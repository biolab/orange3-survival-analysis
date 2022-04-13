from lifelines.utils import concordance_index
from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.evaluation.scoring import Score
from orangecontrib.survival_analysis.widgets.data import (
    get_survival_endpoints,
    contains_survival_endpoints,
)

__all__ = ['ConcordanceIndex']


class SurvivalScorer(Score, abstract=True):
    class_types = (
        ContinuousVariable,
        DiscreteVariable,
    )

    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        return contains_survival_endpoints(domain)


class ConcordanceIndex(SurvivalScorer):
    name = 'C-Index'
    long_name = 'Concordance Index'

    def compute_score(self, results):
        domain = results.domain
        time_var, event_var = get_survival_endpoints(domain)

        c_index = concordance_index(
            results.actual[:, domain.class_vars.index(time_var)],
            -results.predicted,
            results.actual[:, domain.class_vars.index(event_var)],
        )
        return [c_index]
