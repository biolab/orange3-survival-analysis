import numpy as np
from functools import partial
from typing import Optional, Any
from enum import IntEnum

from lifelines.statistics import multivariate_logrank_test

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState

from orangecontrib.survival_analysis.widgets.data import TIME_COLUMN, EVENT_COLUMN
from orangecontrib.survival_analysis.modeling.cox import CoxRegressionLearner, CoxRegressionModel
from orangecontrib.survival_analysis.widgets.data import check_survival_data


class StratifyOn(IntEnum):
    (CoxRiskScore,) = range(1)


class SplittingCriteria(IntEnum):
    Median, Mean, LogRankTest = range(3)


def cutoff_by_log_rank_optimization(durations, events, callback, values):
    results = []

    for cutoff in np.round(values, 2):
        callback()
        strata = (values > cutoff).astype(int)
        _, counts = np.unique(strata, return_counts=True)
        # 10 was chosen arbitrary
        if counts.min() < 10:
            continue

        log_rank_test = multivariate_logrank_test(durations, strata.array, events)
        results.append((cutoff, -np.log2(log_rank_test.p_value)))

    cutoff, _ = max(results, key=lambda x: x[1])
    return cutoff


def cox_risk_score(cox_model: CoxRegressionModel, data: Table):
    return cox_model.predict(data.X)


def stratify(stratify_on: ContinuousVariable, splitting_criteria: int, callback, data: Table):
    stratify_on = stratify_on.compute_value(data)

    if splitting_criteria == SplittingCriteria.Median:
        cutoff = np.median
    elif splitting_criteria == SplittingCriteria.Mean:
        cutoff = np.mean
    elif splitting_criteria == SplittingCriteria.LogRankTest:
        durations, _ = data.get_column_view(data.attributes[TIME_COLUMN])
        events, _ = data.get_column_view(data.attributes[EVENT_COLUMN])
        cutoff = partial(cutoff_by_log_rank_optimization, durations, events, callback)
    else:
        raise ValueError('Unknown splitting criteria')

    return (stratify_on > cutoff(stratify_on)).astype(int)


class OWCohorts(OWWidget, ConcurrentWidgetMixin):
    name = 'Cohorts'
    description = ''
    icon = 'icons/owcohorts.svg'
    priority = 50
    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input('Data', Table)
        learner = Input('Cox Learner', CoxRegressionLearner)

    class Outputs:
        data = Output('Data', Table)

    stratify_on: int = Setting(StratifyOn.CoxRiskScore, schema_only=True)
    splitting_criteria: int = Setting(SplittingCriteria.Median, schema_only=True)
    auto_commit: bool = Setting(True, schema_only=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConcurrentWidgetMixin.__init__(self)
        self.data: Optional[Table] = None
        self.learner: CoxRegressionLearner = CoxRegressionLearner()

        self.stratify_on_options = [
            (StratifyOn.CoxRiskScore, 'Cox Risk Score'),
        ]

        self.splitting_criteria_options = [
            (SplittingCriteria.Median, 'Median'),
            (SplittingCriteria.Mean, 'Mean'),
            (SplittingCriteria.LogRankTest, 'Log Rank Test'),
        ]

        box = gui.vBox(self.controlArea, 'Stratify on', margin=0)
        gui.comboBox(
            box,
            self,
            'stratify_on',
            items=(label for _, label in self.stratify_on_options),
            callback=self.commit.deferred,
        )

        box = gui.vBox(self.controlArea, 'Splitting Criteria', margin=0)
        self.radio_buttons = gui.radioButtons(box, self, 'splitting_criteria', callback=self.commit.deferred)

        for _, opt in self.splitting_criteria_options:
            gui.appendRadioButton(self.radio_buttons, opt)

        self.commit_button = gui.auto_commit(self.controlArea, self, 'auto_commit', '&Commit', box=False)

    @Inputs.learner
    def set_learner(self, learner: CoxRegressionLearner):
        self.learner = learner or CoxRegressionLearner()
        self.commit.now()

    @Inputs.data
    @check_survival_data
    def set_data(self, data: Table) -> None:
        self.data = data
        self.commit.now()

    @gui.deferred
    def commit(
        self,
    ) -> None:
        if not self.data:
            return
        self.start(self.stratify_data, self.data)

    def stratify_data(self, data: Table, state: TaskState) -> Optional[Table]:
        cohort_vars = ()
        steps = iter(np.linspace(0, 100, len(data)))

        def callback():
            try:
                state.set_progress_value(next(steps))
            except StopIteration:
                pass

        if self.stratify_on == StratifyOn.CoxRiskScore:
            cox_model = self.learner(data)
            _, risk_score_label = self.stratify_on_options[self.stratify_on]
            risk_score_var = ContinuousVariable(risk_score_label, compute_value=partial(cox_risk_score, cox_model))
            risk_group_var = DiscreteVariable(
                'Cohorts',
                values=['Low risk', 'High risk'],
                compute_value=partial(stratify, risk_score_var, self.splitting_criteria, callback),
            )

            cohort_vars = (
                risk_score_var,
                risk_group_var,
            )

        domain = Domain(
            self.data.domain.attributes,
            self.data.domain.class_vars,
            self.data.domain.metas + cohort_vars,
        )
        return self.data.transform(domain)

    def on_done(self, result):
        self.Outputs.data.send(result if result else None)

    def on_partial_result(self, result: Any) -> None:
        pass


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWCohorts).run(Table('/Users/jakakokosar/Desktop/melanoma.tab'))
