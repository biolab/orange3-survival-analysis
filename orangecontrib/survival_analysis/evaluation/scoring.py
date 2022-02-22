from lifelines.utils import concordance_index
from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.evaluation.scoring import Score
from orangecontrib.survival_analysis.widgets.data import TIME_COLUMN, EVENT_COLUMN


class SurvivalScorer(Score, abstract=True):
    class_types = (
        ContinuousVariable,
        DiscreteVariable,
    )
    is_built_in = False
    problem_type = 'time_to_event'


class ConcordanceIndex(SurvivalScorer):
    name = 'C-Index'
    long_name = 'Concordance Index'

    def compute_score(self, results):
        data = results.data
        domain = results.domain
        time_var = data.attributes.get(TIME_COLUMN)
        event_var = data.attributes.get(EVENT_COLUMN)

        c_index = concordance_index(
            results.actual[:, domain.class_vars.index(time_var)],
            -results.predicted,
            results.actual[:, domain.class_vars.index(event_var)],
        )
        return [c_index]
