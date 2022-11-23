import itertools
import numpy as np
import pyqtgraph as pg
from typing import Dict, List, Optional, NamedTuple, Any

from AnyQt.QtGui import QColor
from AnyQt.QtCore import Qt, QPointF, pyqtSignal as Signal

from Orange.widgets import gui
from Orange.widgets.settings import Setting, SettingProvider
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data import Table, Domain

from orangecontrib.survival_analysis.modeling.cox import (
    CoxRegressionLearner,
    CoxRegressionModel,
)
from orangecontrib.survival_analysis.widgets.data import (
    check_survival_data,
    get_survival_endpoints,
)


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
        self._parent.view_box.setCursor(
            Qt.PointingHandCursor if hover else Qt.ArrowCursor
        )
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
        self.horizontal_line.setPen(
            color=QColor(Qt.darkGray), width=2, style=Qt.DashLine
        )
        self.horizontal_line.sigPositionChanged.connect(self.selection_line_moved.emit)

        self.setLabels(left='-log2(p)', bottom='num. of features')

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
    model: CoxRegressionModel


def worker(data: Table, learner, state: TaskState):
    # No need to check for irregularities, this is done in widget
    time_var, event_var = get_survival_endpoints(data.domain)

    def fit_cox_models(attrs_combinations):
        results = []
        for attrs in attrs_combinations:
            columns = attrs + [time_var.name, event_var.name]
            cph_model = learner(data[:, columns])
            log2p = cph_model.ll_ratio_log2p()
            result = Result(log2p, cph_model)
            results.append(result)
        return results

    attributes = list(data.domain.attributes)
    progress_steps = iter(np.linspace(0, 100, len(attributes)))
    _trace = fit_cox_models([attributes])
    while len(_trace) != len(data.domain.attributes):
        attributes = list(_trace[-1].model.domain.attributes)

        if len(attributes) > 1:
            combinations = [
                list(comb)
                for comb in itertools.combinations(attributes, len(attributes) - 1)
            ]
        else:
            combinations = [attributes]

        results = fit_cox_models(combinations)
        _trace.append(max(results, key=lambda result: result.log2p))
        state.set_progress_value(next(progress_steps))
    return _trace


class OWStepwiseCoxRegression(OWWidget, ConcurrentWidgetMixin):
    name = 'Stepwise Cox Regression'
    description = 'Feature selection with backward elimination.'
    icon = 'icons/owstepwisecoxregression.svg'
    priority = 40
    keywords = ['feature selection', 'backward elimination', 'cox regression']

    graph = SettingProvider(StepwiseCoxRegressionPlot)
    graph_name = 'graph.plotItem'

    auto_commit: bool = Setting(True, schema_only=True)

    class Inputs:
        data = Input('Data', Table)
        learner = Input('Cox Learner', CoxRegressionLearner)

    class Outputs:
        selected_data = Output('Data', Table)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.learner: Optional[CoxRegressionLearner] = CoxRegressionLearner()
        self.data: Optional[Table] = None
        self.trace: Optional[List[Result]] = None
        gui.rubber(self.controlArea)

        self.graph: StepwiseCoxRegressionPlot = StepwiseCoxRegressionPlot(parent=self)
        self.graph.selection_line_moved.connect(self.on_selection_changed)
        self.mainArea.layout().addWidget(self.graph)
        self.graph.setAntialiasing(True)

        gui.rubber(self.controlArea)
        self.commit_button = gui.auto_commit(
            self.controlArea, self, 'auto_commit', '&Commit', box=False
        )

    @Inputs.learner
    def set_learner(self, learner: CoxRegressionLearner()):
        if learner:
            self.learner = learner
        else:
            self.learner = CoxRegressionLearner()

        self.invalidate()

    @Inputs.data
    @check_survival_data
    def set_data(self, data: Table):
        if not data:
            return
        self.data = data
        self.invalidate()

    def invalidate(self):
        if self.data:
            self.start(
                worker,
                self.data,
                self.learner,
            )

    def on_selection_changed(self, selection_line):
        self.current_x = selection_line.getXPos()  # + 1
        self.commit()

    def commit(self):
        if self.current_x:
            result: Result = self.trace[self.current_x - 1]

            domain = Domain(
                result.model.domain.attributes,
                result.model.domain.class_vars,
                self.data.domain.metas,
            )
            data = self.data.transform(domain)
            self.Outputs.selected_data.send(data)

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

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    table = Table('http://datasets.biolab.si/core/melanoma.tab')
    table.attributes['time_var'] = table.domain['time']
    table.attributes['event_var'] = table.domain['event']
    table.attributes['problem_type'] = 'time_to_event'
    metas = [
        meta
        for meta in table.domain.metas
        if meta not in (table.domain['time'], table.domain['event'])
    ]
    domain = Domain(
        table.domain.attributes,
        metas=metas,
        class_vars=[table.domain['time'], table.domain['event']],
    )
    table = table.transform(domain)
    WidgetPreview(OWStepwiseCoxRegression).run(input_data=table)
