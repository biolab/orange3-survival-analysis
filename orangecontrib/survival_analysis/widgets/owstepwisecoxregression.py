import itertools
import pandas as pd
import numpy as np
import pyqtgraph as pg
from typing import Dict, List, Optional, NamedTuple, Tuple, Any

from AnyQt.QtGui import QColor
from AnyQt.QtCore import Qt, QPointF, pyqtSignal as Signal
from lifelines import CoxPHFitter

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting, SettingProvider
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.data.pandas_compat import table_to_frame


class CustomInfiniteLine(pg.InfiniteLine):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent = parent

    def setPos(self, pos):

        if isinstance(pos, (list, tuple)):
            pos_x, pos_y = pos
        elif isinstance(pos, QPointF):
            pos_x, pos_y = round(pos.x()), 0
        else:
            return

        if getattr(self, 'span', None):
            y_min, y_max = self._parent.get_viewbox_y_range()
            max_span = (self._parent.map_x_to_y.get(pos_x, 0) - y_min) / (y_max - y_min)
            self.setSpan(0, max_span)

        super().setPos((pos_x, pos_y))

    def setMouseHover(self, hover):
        self._parent.view_box.setCursor(Qt.PointingHandCursor if hover else Qt.ArrowCursor)
        super().setMouseHover(hover)


class StepwiseCoxRegressionPlot(gui.OWComponent, pg.PlotWidget):
    selection_line_moved = Signal(object)

    def __init__(self, parent: OWWidget = None):
        gui.OWComponent.__init__(self, widget=parent)
        pg.PlotWidget.__init__(self, parent=parent)

        self.view_box = self.getViewBox()
        self.plotItem.setMouseEnabled(x=False, y=False)

        self.map_x_to_y: Optional[Dict[str, str]] = None
        self.plot_line: Optional[pg.PlotDataItem] = None
        self.horizontal_line = CustomInfiniteLine(self, movable=True)
        self.horizontal_line.setPen(color=QColor(Qt.darkGray), width=2, style=Qt.DashLine)
        self.horizontal_line.sigPositionChanged.connect(self.selection_line_moved.emit)

        self.setLabels(left='-log2(p)', bottom='num. of genes')

    def set_plot(self, x, y):
        self.clear()

        self.map_x_to_y = dict(zip(x, y))
        self.plot_line = pg.PlotDataItem(x, y)
        self.plot_line.setPen(color=QColor(Qt.black), width=3)
        self.addItem(self.plot_line)
        self.addItem(self.horizontal_line)
        self.horizontal_line.setBounds((1, len(x)))
        self.horizontal_line.setPos((max(self.map_x_to_y, key=self.map_x_to_y.get), 0))
        self.view_box.invertX(True)

    def get_viewbox_y_range(self):
        return self.view_box.state['targetRange'][1]


class Result(NamedTuple):
    log2p: int
    covariate_to_coef: dict
    removed_covariates: list


def worker(df: pd.DataFrame, initial_covariates: set, time_var: str, event_var: str, state: TaskState):
    progress_steps = iter(np.linspace(0, 100, len(initial_covariates)))

    def fit_cox_models(remaining_covariates: set, combinations_to_check: List[Tuple[str, ...]]):
        results = []
        for covariates in combinations_to_check:
            cph = CoxPHFitter().fit(
                df[[time_var, event_var] + list(covariates)], duration_col=time_var, event_col=event_var
            )
            covariate_to_coef = cph.summary.to_dict('dict')['coef']
            log2p = -np.log2(cph.log_likelihood_ratio_test().p_value)
            result = Result(log2p, covariate_to_coef, [cov for cov in remaining_covariates - set(covariates)])
            results.append(result)
        return results

    removed_covariates = set()
    _trace = fit_cox_models(initial_covariates, [tuple(initial_covariates)])
    while True:
        covariates_to_eval = initial_covariates - removed_covariates

        if len(covariates_to_eval) > 1:
            gene_combinations = list(itertools.combinations(covariates_to_eval, len(covariates_to_eval) - 1))
        else:
            gene_combinations = [tuple(covariates_to_eval)]

        results = fit_cox_models(covariates_to_eval, gene_combinations)

        best_result = max(results, key=lambda result: result.log2p)
        if not best_result.removed_covariates:
            break

        _trace.append(best_result)
        removed_covariates.update(set(best_result.removed_covariates))
        state.set_progress_value(next(progress_steps))

    return _trace


class OWStepwiseCoxRegression(OWWidget, ConcurrentWidgetMixin):
    name = 'Stepwise Cox Regression'
    description = 'Backward feature elimination'
    icon = ''

    graph = SettingProvider(StepwiseCoxRegressionPlot)
    graph_name = 'graph.plotItem'

    settingsHandler = DomainContextHandler()
    time_var = ContextSetting(None, schema_only=True)
    auto_commit: bool = Setting(False, schema_only=True)

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        selected_data = Output('Data', Table)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data: Optional[Table] = None
        self.data_df: Optional[pd.DataFrame] = None
        self.attr_name_to_variable: Optional[dict] = None

        self.trace: Optional[List[Result]] = None

        time_var_model = DomainModel(valid_types=(ContinuousVariable,), order=(4,))
        box = gui.vBox(self.controlArea, 'Time', margin=0)
        gui.comboBox(box, self, 'time_var', model=time_var_model, callback=self.on_controls_changed)

        gui.rubber(self.controlArea)

        self.graph: StepwiseCoxRegressionPlot = StepwiseCoxRegressionPlot(parent=self)
        self.graph.selection_line_moved.connect(self.on_selection_changed)
        self.mainArea.layout().addWidget(self.graph)
        self.graph.setAntialiasing(True)

        gui.rubber(self.controlArea)
        self.commit_button = gui.auto_commit(self.controlArea, self, 'auto_commit', '&Commit', box=False)

    @Inputs.data
    def set_data(self, data: Table):
        self.closeContext()
        if not data:
            return

        self.data = data
        self.attr_name_to_variable = {attr.name: attr for attr in self.data.domain.attributes}
        self.data_df = table_to_frame(data, include_metas=True)
        self.data_df = self.data_df.astype({self.data.domain.class_var.name: np.float64})

        self.controls.time_var.model().set_domain(self.data.domain)
        self.time_var = None
        self.openContext(data)

        if self.time_var:
            self.start(
                worker,
                self.data_df,
                set(self.attr_name_to_variable.keys()),
                self.time_var.name,
                self.data.domain.class_var.name,
            )

    def on_controls_changed(self):
        if self.time_var:
            self.start(
                worker,
                self.data_df,
                set(self.attr_name_to_variable.keys()),
                self.time_var.name,
                self.data.domain.class_var.name,
            )

    def on_selection_changed(self, selection_line):
        self.current_x = selection_line.getXPos() - 1
        self.commit()

    def commit(self):
        self.Outputs.selected_data.send(self.stratify_data(self.data_df, self.trace[self.current_x]))

    def on_done(self, trace):
        # save results
        self.trace = list(reversed(trace))

        # plot lines
        y = [result.log2p for result in trace]
        x = list(reversed(range(1, len(y) + 1)))
        self.graph.set_plot(x, y)

        # send data
        self.commit()

    def on_exception(self, ex):
        raise ex

    def on_partial_result(self, result: Any) -> None:
        pass

    def stratify_data(
        self,
        df: pd.DataFrame,
        result: Result,
    ) -> Table:
        covariates = result.covariate_to_coef.keys()

        risk_score_label = 'Risk Score'
        risk_group_label = 'Risk Group'
        risk_score_var = ContinuousVariable(risk_score_label)
        risk_group_var = DiscreteVariable(risk_group_label, values=['Low Risk', 'High Risk'])

        df[risk_score_label] = df[covariates].dot([result.covariate_to_coef[covariate] for covariate in covariates])
        df[risk_group_label] = (df[risk_score_label] >= df[risk_score_label].median()).astype(int)

        attrs = [self.attr_name_to_variable[name] for name in covariates]
        domain = Domain(attrs, self.data.domain.class_var, self.data.domain.metas + (risk_score_var, risk_group_var))
        data = self.data.transform(domain)
        data[:, risk_score_var] = np.reshape(df[risk_score_label].to_numpy(), (-1, 1))
        data[:, risk_group_var] = np.reshape(df[risk_group_label].to_numpy(), (-1, 1))
        return data

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWStepwiseCoxRegression).run(Table('test_data_full.pkl'))
