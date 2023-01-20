import numpy as np
import multiprocessing
import queue
from multiprocessing import cpu_count
from functools import partial
from typing import Any, Optional

from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.multitest import fdrcorrection

from AnyQt.QtWidgets import QButtonGroup, QGridLayout, QRadioButton, QAbstractScrollArea
from AnyQt.QtCore import (
    Qt,
    QItemSelection,
    QItemSelectionModel,
    QItemSelectionRange,
    QSize,
)

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget, AttributeList
from Orange.data import Table, Domain
from Orange.widgets.data.owrank import TableView

from orangecontrib.survival_analysis.widgets.data import (
    check_survival_data,
    get_survival_endpoints,
)
from orangecontrib.survival_analysis.modeling.cox import to_data_frame


class ScoreMethods:
    multivariate_log_rank = 0
    cox_regression = 1

    labels = ['Multivariate log-rank test', 'Cox regression']


def __process_batch(queue, time_var, event_var, df):
    batch_results = []
    cph = CoxPHFitter()

    for covariate in [col for col in df.columns if col not in (time_var, event_var)]:
        queue.put(covariate)
        # fit cox
        model = cph.fit(
            df[[time_var, event_var, covariate]],
            duration_col=time_var,
            event_col=event_var,
        )
        # log-likelihood ratio test
        ll_ratio_test = model.log_likelihood_ratio_test()
        batch_results.append(
            (
                covariate,
                ll_ratio_test.test_statistic,
                ll_ratio_test.p_value,
            )
        )

    return np.array(batch_results)


def cox_regression_scorer_multicore(
    data: Table, time_var: str, event_var: str, state: TaskState
):
    attr_name_to_variable = {attr.name: attr for attr in data.domain.attributes}
    attrs = list(attr_name_to_variable.keys())

    # order of the data gets lost with map_async
    attr_to_result = {attr.name: None for attr in data.domain.attributes}

    with multiprocessing.Manager() as _manager:
        _queue = _manager.Queue()
        _cpu_count = cpu_count()

        df = to_data_frame(data)
        batches = (
            df[[time_var, event_var] + list(batch)]
            for batch in [attrs[i::_cpu_count] for i in range(_cpu_count)]
        )

        progress_steps = iter(np.linspace(0, 100, len(attrs)))

        with multiprocessing.Pool(processes=_cpu_count) as pool:
            results = pool.map_async(
                partial(
                    __process_batch,
                    _queue,
                    time_var,
                    event_var,
                ),
                batches,
            )

            while True:
                try:
                    state.set_progress_value(next(progress_steps))
                    _queue.get(timeout=3)
                except (queue.Empty, StopIteration):
                    break

            stacked_result = np.vstack(results.get())
            covariate_names = stacked_result[:, 0].tolist()
            results = stacked_result[:, 1:].astype(float)

            # map attr name to results in 'attr_to_result' dict
            for attr_name, row_data in zip(covariate_names, results):
                attr_to_result[attr_name] = row_data.tolist()

            # output sorted data
            return [
                [attr_name_to_variable[attr_name]] + row_data
                for attr_name, row_data in attr_to_result.items()
            ]


def cox_regression_scorer(data: Table, time_var: str, event_var: str, state: TaskState):
    progress_steps = iter(np.linspace(0, 100, len(data.domain.attributes)))

    attr_name_to_variable = {attr.name: attr for attr in data.domain.attributes}

    df = to_data_frame(data)
    cph = CoxPHFitter()
    results = []

    for attr_name in [col for col in df.columns if col not in (time_var, event_var)]:
        # fit cox
        model = cph.fit(
            df[[time_var, event_var, attr_name]],
            duration_col=time_var,
            event_col=event_var,
        )
        # log-likelihood ratio test
        ll_ratio_test = model.log_likelihood_ratio_test()
        results.append(
            [
                attr_name_to_variable[attr_name],
                ll_ratio_test.test_statistic,
                ll_ratio_test.p_value,
            ]
        )
        state.set_progress_value(next(progress_steps))
    return results


def log_rank_scorer(data: Table, time_var: str, event_var: str, state: TaskState):

    time = data.get_column(time_var)
    event = data.get_column(event_var)
    progress_steps = iter(np.linspace(0, 100, len(data.domain.attributes)))

    results = []
    for var in list(data.domain.attributes):
        column_values = mask = data.get_column(var)
        if var.is_continuous:
            mask = column_values > np.median(column_values)
        log_rank = multivariate_logrank_test(time, mask, event)
        results.append([var, log_rank.test_statistic, log_rank.p_value])
        state.set_progress_value(next(progress_steps))

    return results


def worker(data: Table, score_method, state: TaskState):
    time_var, event_var = get_survival_endpoints(data.domain)
    _, columns = data.X.shape

    if score_method == ScoreMethods.multivariate_log_rank:
        scorer = log_rank_scorer
    elif score_method == ScoreMethods.cox_regression:
        scorer = (
            cox_regression_scorer if columns <= 100 else cox_regression_scorer_multicore
        )
    else:
        raise ValueError('Unexpected scorer type')

    results = scorer(data, time_var.name, event_var.name, state)

    # the last column is p values, calculate FDR
    _, pvals_corrected = fdrcorrection([col[-1] for col in results], is_sorted=False)

    return [columns + [fdr] for columns, fdr in zip(results, pvals_corrected)]


class OWRankSurvivalFeatures(OWWidget, ConcurrentWidgetMixin):
    name = 'Rank Survival Features'
    description = 'Ranking of features by univariate Cox regression analysis.'
    icon = 'icons/owranksurvivalfeatures.svg'
    priority = 30
    keywords = ['univariate cox regression', 'rank', 'log-likelihood ratio test']

    buttons_area_orientation = Qt.Vertical
    select_none, manual_selection, select_n_best = range(3)

    settingsHandler = DomainContextHandler()
    selected_attrs = ContextSetting([], schema_only=True)
    selection_method = Setting(select_n_best, schema_only=True)
    n_selected = Setting(5, schema_only=True)
    score_method = Setting(0, schema_only=True)
    auto_commit: bool = Setting(True, schema_only=True)

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        reduced_data = Output('Reduced Data', Table, default=True)
        features = Output("Features", AttributeList, dynamic=False)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data: Optional[Table] = None

        box = gui.vBox(self.controlArea, 'Score method', margin=0)
        gui.comboBox(
            box,
            self,
            'score_method',
            items=ScoreMethods.labels,
            callback=self.start_worker,
        )

        gui.rubber(self.controlArea)

        sel_method_box = gui.vBox(self.buttonsArea, 'Select Attributes')
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.select_buttons = QButtonGroup()
        self.select_buttons.idClicked.connect(self.set_selection_method)

        def button(text, buttonid, tool_tip=None):
            b = QRadioButton(text)
            self.select_buttons.addButton(b, buttonid)
            if tool_tip is not None:
                b.setToolTip(tool_tip)
            return b

        b1 = button(self.tr('None'), OWRankSurvivalFeatures.select_none)
        b2 = button(self.tr('Manual'), OWRankSurvivalFeatures.manual_selection)
        b3 = button(self.tr('Best ranked:'), OWRankSurvivalFeatures.select_n_best)

        s = gui.spin(
            sel_method_box,
            self,
            'n_selected',
            1,
            9999,
            callback=lambda: self.set_selection_method(
                OWRankSurvivalFeatures.select_n_best
            ),
            addToLayout=False,
        )

        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 1, 0)
        grid.addWidget(b3, 2, 0)
        grid.addWidget(s, 2, 1)

        sel_method_box.layout().addLayout(grid)
        self.select_buttons.button(self.selection_method).setChecked(True)

        self.commit_button = gui.auto_commit(
            self.buttonsArea, self, 'auto_commit', '&Commit', box=False
        )

        # Main area
        self.model = PyTableModel(parent=self)
        self.table_view = TableView(parent=self)
        self.table_view.setModel(self.model)
        self.table_view.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContentsOnFirstShow
        )
        self.table_view.selectionModel().selectionChanged.connect(self.on_select)

        def _set_select_manual():
            self.set_selection_method(OWRankSurvivalFeatures.manual_selection)

        self.table_view.manualSelection.connect(_set_select_manual)
        self.table_view.verticalHeader().sectionClicked.connect(_set_select_manual)

        self.mainArea.layout().addWidget(self.table_view)

    @Inputs.data
    @check_survival_data
    def set_data(self, data: Table):
        self.closeContext()
        self.selected_attrs = []
        self.model.clear()
        self.model.resetSorting()

        if not data:
            return

        self.data = data
        self.openContext(data)
        self.start_worker()

    def start_worker(self):
        if self.data is None or self.score_method is None:
            return

        self.start(worker, self.data, self.score_method)

    def commit(self):
        if not self.selected_attrs:
            self.Outputs.reduced_data.send(None)
            self.Outputs.features.send(None)
        else:
            reduced_domain = Domain(
                self.selected_attrs, self.data.domain.class_vars, self.data.domain.metas
            )
            data = self.data.transform(reduced_domain)
            self.Outputs.reduced_data.send(data)
            self.Outputs.features.send(AttributeList(self.selected_attrs))

    def on_done(self, worker_result):
        self.model.wrap(worker_result)

        if self.score_method == ScoreMethods.multivariate_log_rank:
            self.model.setHorizontalHeaderLabels(
                ['', 'Multivariate log-rank test', f'{"p".center(13)}', 'FDR']
            )

        elif self.score_method == ScoreMethods.cox_regression:
            self.model.wrap(worker_result)
            self.model.setHorizontalHeaderLabels(
                ['', 'Log-Likelihood Ratio test', f'{"p".center(13)}', 'FDR']
            )

        self.table_view.resizeColumnsToContents()

        # sort by p values
        self.table_view.sortByColumn(2, Qt.AscendingOrder)
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
            selection = QItemSelection(
                self.model.index(0, 0),
                self.model.index(n_selected - 1, column_count - 1),
            )
        else:
            selection = QItemSelection()
            if self.selected_attrs is not None:
                attr_indices = [
                    self.data.domain.attributes.index(var)
                    for var in self.selected_attrs
                ]

                for row in self.model.mapFromSourceRows(attr_indices):
                    selection.append(
                        QItemSelectionRange(
                            self.model.index(row, 0),
                            self.model.index(row, column_count - 1),
                        )
                    )

        selection_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def on_select(self):
        selected_rows = self.table_view.selectionModel().selectedRows(0)
        row_indices = [i.row() for i in selected_rows]
        attr_indices = self.model.mapToSourceRows(row_indices)
        self.selected_attrs = [self.data.domain[idx] for idx in attr_indices]
        self.commit()

    def sizeHint(self):
        return QSize(750, 600)


if __name__ == '__main__':
    previewer = WidgetPreview(OWRankSurvivalFeatures)
    previewer.run(Table('iris.tab'))
