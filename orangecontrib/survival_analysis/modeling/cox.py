import numpy as np

from lifelines import CoxPHFitter
from Orange.data.pandas_compat import table_to_frame
from Orange.base import Learner, Model

from orangecontrib.survival_analysis.widgets.data import TIME_COLUMN, EVENT_COLUMN


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

    def predict(self, X):
        """Predict risk scores."""
        return self._model.predict_partial_hazard(X)

    def __call__(self, data, ret=Model.Value):
        return self.predict(data.X).array


class CoxRegressionLearner(Learner):
    __returns__ = CoxRegressionModel
    supports_multiclass = True
    learner_adequacy_err_msg = 'Survival variables expected.'

    def __init__(self, preprocessors=None, **kwargs):
        self.params = vars()
        super().__init__(preprocessors=preprocessors)

    def check_learner_adequacy(self, domain):
        return len(domain.class_vars) == 2

    def fit_storage(self, data):
        return self.fit(data)

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            return self.fit(data)

    def fit(self, data):
        df = table_to_frame(data, include_metas=False)
        time_var = data.attributes.get(TIME_COLUMN)
        event_var = data.attributes.get(EVENT_COLUMN)
        cph = CoxPHFitter(**self.params['kwargs'])
        cph = cph.fit(df, duration_col=time_var.name, event_col=event_var.name)
        return CoxRegressionModel(cph)

    def __call__(self, data, progress_callback=None):
        m = super().__call__(data, progress_callback)
        m.params = self.params
        return m
