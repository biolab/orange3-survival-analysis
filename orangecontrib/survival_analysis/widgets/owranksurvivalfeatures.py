import numpy as np
import multiprocessing
from multiprocessing import cpu_count
from functools import partial
from typing import Any, Optional, List

from lifelines import CoxPHFitter
from statsmodels.stats.multitest import fdrcorrection

from AnyQt.QtWidgets import QButtonGroup, QGridLayout, QRadioButton, QAbstractScrollArea
from AnyQt.QtCore import Qt, QItemSelection, QItemSelectionModel, QItemSelectionRange

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data import Table, Domain
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets.data.owrank import TableView

from orangecontrib.survival_analysis.widgets.data import check_survival_data, TIME_COLUMN, EVENT_COLUMN


def batch_to_process(queue, time_var, event_var, df):
    batch_results = []
    cph = CoxPHFitter()

    for covariate in [col for col in df.columns if col not in (time_var, event_var)]:
        queue.put(covariate)
        # fit cox
        model = cph.fit(df[[time_var, event_var, covariate]], duration_col=time_var, event_col=event_var)
        # log-likelihood ratio test
        ll_ratio_test = model.log_likelihood_ratio_test()
        batch_results.append((covariate, cph.log_likelihood_, ll_ratio_test.test_statistic, ll_ratio_test.p_value))

    return np.array(batch_results)


def worker(table: Table, covariates: List, time_var: str, event_var: str, state: TaskState):
    with multiprocessing.Manager() as _manager:
        _queue = _manager.Queue()
        _cpu_count = cpu_count()

        df = table_to_frame(table, include_metas=True)
        df = df.astype({event_var: np.float64})
        batches = [
            df[[time_var, event_var] + batch] for batch in [covariates[i::_cpu_count] for i in range(_cpu_count)]
        ]
        progress_steps = iter(np.linspace(0, 100, len(covariates)))

        with multiprocessing.Pool(processes=_cpu_count) as pool:
            results = pool.map_async(
                partial(
                    batch_to_process,
                    _queue,
                    time_var,
                    event_var,
                ),
                batches,
            )
            while True:
                try:
                    state.set_progress_value(next(progress_steps))
                except StopIteration:
                    break
                _queue.get()

            stacked_result = np.vstack(results.get())
            covariate_names = stacked_result[:, 0]
            results = stacked_result[:, 1:].astype(float)
            _, pvals_corrected = fdrcorrection(results[:, -1], is_sorted=False)
            results = np.hstack((results, pvals_corrected.reshape(pvals_corrected.shape[0], -1)))
            return covariate_names, results


class OWRankSurvivalFeatures(OWWidget, ConcurrentWidgetMixin):
    name = 'Rank Survival Features'
    # TODO: Add widget metadata
    description = ''
    icon = 'icons/owranksurvivalfeatures.svg'
    priority = 30
    keywords = []

    buttons_area_orientation = Qt.Vertical
    select_none, manual_selection, select_n_best = range(3)

    settingsHandler = DomainContextHandler()
    selected_attrs = ContextSetting([], schema_only=True)
    selection_method = Setting(select_n_best, schema_only=True)
    n_selected = Setting(20, schema_only=True)
    auto_commit: bool = Setting(False, schema_only=True)

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        reduced_data = Output('Reduced Data', Table, default=True)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data: Optional[Table] = None
        self.attr_name_to_variable: Optional[Table] = None
        self.covariates_from_worker_result = None
        self.time_var: Optional[str] = None
        self.event_var: Optional[str] = None

        gui.rubber(self.controlArea)

        sel_method_box = gui.vBox(self.buttonsArea, 'Select Attributes')
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.select_buttons = QButtonGroup()
        self.select_buttons.buttonClicked[int].connect(self.set_selection_method)

        def button(text, buttonid, toolTip=None):
            b = QRadioButton(text)
            self.select_buttons.addButton(b, buttonid)
            if toolTip is not None:
                b.setToolTip(toolTip)
            return b

        b1 = button(self.tr('None'), OWRankSurvivalFeatures.select_none)
        b2 = button(self.tr('Manual'), OWRankSurvivalFeatures.manual_selection)
        b3 = button(self.tr('Best ranked:'), OWRankSurvivalFeatures.select_n_best)

        s = gui.spin(
            sel_method_box,
            self,
            'n_selected',
            1,
            999,
            callback=lambda: self.set_selection_method(OWRankSurvivalFeatures.select_n_best),
            addToLayout=False,
        )

        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 1, 0)
        grid.addWidget(b3, 2, 0)
        grid.addWidget(s, 2, 1)

        sel_method_box.layout().addLayout(grid)

        self.commit_button = gui.auto_commit(self.buttonsArea, self, 'auto_commit', '&Commit', box=False)

        # Main area
        self.model = PyTableModel(parent=self)
        self.table_view = TableView(parent=self)
        self.table_view.setModel(self.model)
        self.model.setHorizontalHeaderLabels(['Log-Likelihood', 'Log-Likelihood Ratio', f'{"p".center(13)}', 'FDR'])
        self.table_view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.table_view.selectionModel().selectionChanged.connect(self.on_select)

        def _set_select_manual():
            self.set_selection_method(OWRankSurvivalFeatures.manual_selection)

        self.table_view.manualSelection.connect(_set_select_manual)
        self.table_view.verticalHeader().sectionClicked.connect(_set_select_manual)

        self.mainArea.layout().addWidget(self.table_view)

    @property
    def covariates(self) -> Optional[List[str]]:
        if not self.data:
            return
        return [attr.name for attr in self.data.domain.attributes]

    @Inputs.data
    @check_survival_data
    def set_data(self, data: Table):
        self.closeContext()
        self.selected_attrs = []
        self.covariates_from_worker_result = []
        self.model.clear()
        self.model.resetSorting()

        if not data:
            return

        self.data = data
        self.attr_name_to_variable = {attr.name: attr for attr in self.data.domain.attributes}

        self.openContext(data)
        self.time_var = self.data.attributes[TIME_COLUMN].name
        self.event_var = self.data.attributes[EVENT_COLUMN].name
        self.start(worker, self.data, self.covariates, self.time_var, self.event_var)

    def commit(self):
        if not self.selected_attrs:
            self.Outputs.reduced_data.send(None)
            self.Outputs.stratified_data.send(None)
        else:
            reduced_domain = Domain(self.selected_attrs, self.data.domain.class_var, self.data.domain.metas)
            data = self.data.transform(reduced_domain)
            self.Outputs.reduced_data.send(data)

    def on_done(self, worker_result):
        covariate_names, results = worker_result

        # wrap everything except covariate names
        self.model.wrap(results.tolist())

        # this is temp solution because covariate orders gets mixed when using multiprocessing
        self.covariates_from_worker_result = covariate_names.tolist()

        # match covariate names to domain variables and set vertical header
        self.model.setVerticalHeaderLabels([self.attr_name_to_variable[name] for name in covariate_names])
        self.table_view.resizeColumnsToContents()

        self.auto_select()

    def on_exception(self, ex):
        raise ex

    def on_partial_result(self, result: Any) -> None:
        pass

    def set_selection_method(self, method):
        self.selection_method = method
        self.select_buttons.button(method).setChecked(True)
        self.auto_select()

    def auto_select(self):
        selection_model = self.table_view.selectionModel()
        row_count = self.model.rowCount()
        column_count = self.model.columnCount()

        if self.selection_method == OWRankSurvivalFeatures.select_none:
            selection = QItemSelection()
        elif self.selection_method == OWRankSurvivalFeatures.select_n_best:
            n_selected = min(self.n_selected, row_count)
            selection = QItemSelection(self.model.index(0, 0), self.model.index(n_selected - 1, column_count - 1))
        else:
            selection = QItemSelection()
            if self.selected_attrs is not None:
                attr_indices = [self.covariates_from_worker_result.index(var.name) for var in self.selected_attrs]
                for row in self.model.mapFromSourceRows(attr_indices):
                    selection.append(
                        QItemSelectionRange(self.model.index(row, 0), self.model.index(row, column_count - 1))
                    )

        selection_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def on_select(self):
        selected_rows = self.table_view.selectionModel().selectedRows(0)
        row_indices = [i.row() for i in selected_rows]
        attr_indices = self.model.mapToSourceRows(row_indices)
        self.selected_attrs = [self.model._headers[Qt.Vertical][row] for row in attr_indices]
        self.commit()


if __name__ == '__main__':
    previewer = WidgetPreview(OWRankSurvivalFeatures)
    previewer.run(Table('iris.tab'))
