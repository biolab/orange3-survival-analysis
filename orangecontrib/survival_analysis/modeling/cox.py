import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from Orange.base import Learner, Model
from Orange.data import Table, Domain, table_from_frame

from orangecontrib.survival_analysis.widgets.data import (
    contains_survival_endpoints,
    get_survival_endpoints,
    MISSING_SURVIVAL_DATA,
)


def to_data_frame(table: Table) -> pd.DataFrame:
    columns = table.domain.attributes + table.domain.class_vars
    df = pd.DataFrame({col.name: table.get_column_view(col)[0] for col in columns})
    df = df.dropna(axis=0)
    return df


class CoxRegressionModel(Model):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.params = vars()

    @property
    def covariates(self):
        return self._model.summary.index.to_list()

    @property
    def coefficients(self):
        return self._model.summary.coef.to_list()

    def ll_ratio_log2p(self):
        """return -log2p from log-likelihood ratio test"""
        return -np.log2(self._model.log_likelihood_ratio_test().p_value)

    def predict(self, x):
        """Predict risk scores."""
        return self._model.predict_partial_hazard(x)

    def summary_to_table(self) -> Table:
        df = self._model.summary

        if 'cmp to' in df.columns:
            df = df.drop(['cmp to'], axis=1)

        table = table_from_frame(df)
        table.name = 'model summary'
        domain = Domain(
            [table.domain['coef']],
            metas=[var for var in table.domain if var.name != 'coef'],
        )
        return table.transform(domain)

    def __call__(self, data, ret=Model.Value):
        return self.predict(data.X).array


class CoxRegressionLearner(Learner):
    __returns__ = CoxRegressionModel
    supports_multiclass = True

    def __init__(self, preprocessors=None, **kwargs):
        self.params = vars()
        super().__init__(preprocessors=preprocessors)

    def incompatibility_reason(self, domain):
        if not contains_survival_endpoints(domain):
            return MISSING_SURVIVAL_DATA

    def fit_storage(self, data):
        return self.fit(data)

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            return self.fit(data)

    def fit(self, data):
        if not contains_survival_endpoints(data.domain):
            raise ValueError(MISSING_SURVIVAL_DATA)
        time_var, event_var = get_survival_endpoints(data.domain)

        df = to_data_frame(data)
        cph = CoxPHFitter(**self.params['kwargs'])
        cph = cph.fit(df, duration_col=time_var.name, event_col=event_var.name)
        return CoxRegressionModel(cph)

    def __call__(self, data, progress_callback=None):
        m = super().__call__(data, progress_callback)
        m.params = self.params
        return m
