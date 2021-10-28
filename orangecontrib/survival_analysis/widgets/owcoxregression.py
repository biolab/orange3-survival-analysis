import numpy as np
import pandas as pd
from typing import Union
from itertools import chain

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLayout, QSizePolicy

from Orange.data import StringVariable
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.signals import Output
from Orange.widgets.settings import (
    ContextSetting,
    DomainContextHandler,
)


from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, TimeVariable
from Orange.data.pandas_compat import table_to_frame
from lifelines import CoxPHFitter


class CoxRegressionModel:
    def __init__(self, model):
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
        return X.dot(self.coefficients)


class CoxRegressionLearner:
    __returns__ = CoxRegressionModel

    def __init__(self, *args, **kwargs):
        self.params = vars()

    def fit(self, df, duration_col, event_col):
        cph = CoxPHFitter(*self.params['args'], **self.params['kwargs'])
        cph = cph.fit(df, duration_col=duration_col, event_col=event_col)
        return CoxRegressionModel(cph)

    def __call__(self, data: Union[Table, pd.DataFrame], duration_col: str, event_col: str):

        if isinstance(data, Table):
            df = table_to_frame(data, include_metas=True)
            df = df[[duration_col, event_col] + [attr.name for attr in data.domain.attributes]]
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError('Wrong data type')

        model = self.fit(df, duration_col, event_col)
        return model


class OWCoxRegression(OWBaseLearner):
    name = 'Cox regression'
    description = (
        'Cox proportional-hazards regression with optional L1 (LASSO), '
        'L2 (ridge) or L1L2 (elastic net) regularization.'
    )

    LEARNER = CoxRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        data = Output('Stratified data', Table, explicit=True)
        coefficients = Output('Coefficients', Table, explicit=True)

    class Inputs(OWBaseLearner.Inputs):
        pass

    REGULARIZATION_TYPES = [
        'No regularization',
        'Ridge regression (L2)',
        'Lasso regression (L1)',
        'Elastic net regression',
    ]
    OLS, Ridge, Lasso, Elastic = 0, 1, 2, 3

    settingsHandler = DomainContextHandler()

    ridge = settings.Setting(False)
    reg_type = settings.Setting(OLS)
    alpha_index: int
    alpha_index = settings.Setting(0)
    l2_ratio: int
    l2_ratio = settings.Setting(0.5)
    autosend = settings.Setting(True)

    time_var = ContextSetting(None, schema_only=True)
    event_var = ContextSetting(None, schema_only=True)

    alphas = list(
        chain(
            [x / 10000 for x in range(1, 10)],
            [x / 1000 for x in range(1, 20)],
            [x / 100 for x in range(2, 20)],
            [x / 10 for x in range(2, 9)],
            range(1, 20),
            range(20, 100, 5),
            range(100, 1001, 100),
        )
    )

    def add_main_layout(self):
        time_var_model = DomainModel(valid_types=(ContinuousVariable,))
        event_var_model = DomainModel(valid_types=DomainModel.PRIMITIVE)

        box = gui.vBox(self.controlArea, 'Time', margin=0)
        gui.comboBox(box, self, 'time_var', model=time_var_model, callback=self.on_controls_changed)

        box = gui.vBox(self.controlArea, 'Event', margin=0)
        gui.comboBox(box, self, 'event_var', model=event_var_model, callback=self.on_controls_changed)

        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.hBox(self.controlArea, 'Regularization')
        gui.radioButtons(box, self, 'reg_type', btnLabels=self.REGULARIZATION_TYPES, callback=self._reg_type_changed)

        gui.separator(box, 20, 20)
        self.alpha_box = box2 = gui.vBox(box, margin=10)
        gui.widgetLabel(box2, 'Regularization strength:')
        gui.hSlider(
            box2,
            self,
            'alpha_index',
            minValue=0,
            maxValue=len(self.alphas) - 1,
            callback=self._alpha_changed,
            createLabel=False,
        )
        box3 = gui.hBox(box2)
        box3.layout().setAlignment(Qt.AlignCenter)
        self.alpha_label = gui.widgetLabel(box3, '')
        self._set_alpha_label()

        gui.separator(box2, 10, 10)
        box4 = gui.vBox(box2, margin=0)
        gui.widgetLabel(box4, 'Elastic net mixing:')
        box5 = gui.hBox(box4)
        gui.widgetLabel(box5, 'L1')
        self.l2_ratio_slider = gui.hSlider(
            box5,
            self,
            'l2_ratio',
            minValue=0.01,
            maxValue=0.99,
            intOnly=False,
            ticks=0.1,
            createLabel=False,
            width=120,
            step=0.01,
            callback=self._l2_ratio_changed,
        )
        gui.widgetLabel(box5, 'L2')
        self.l2_ratio_label = gui.widgetLabel(box4, "", sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.l2_ratio_label.setAlignment(Qt.AlignCenter)

        box5 = gui.hBox(self.controlArea)
        box5.layout().setAlignment(Qt.AlignCenter)
        self._set_l2_ratio_label()
        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.controls.alpha_index.setEnabled(self.reg_type != self.OLS)
        self.l2_ratio_slider.setEnabled(self.reg_type == self.Elastic)

    @Inputs.data
    def set_data(self, data):
        """Set the input train dataset."""
        self.closeContext()
        self.Error.data_error.clear()
        self.data = data

        if not data:
            self.data = None
            self.update_model()
            return

        def _filter_domain_model_options(_domain):
            vars_ = [var for var in _domain.metas if not isinstance(var, TimeVariable)]
            vars_ = [
                var
                for var in vars_
                if isinstance(var, ContinuousVariable) or isinstance(var, DiscreteVariable) and len(var.values) <= 2
            ]

            return Domain(vars_)

        domain = _filter_domain_model_options(data.domain)
        self.controls.time_var.model().set_domain(domain)
        self.controls.event_var.model().set_domain(domain)

        self.time_var = None
        self.event_var = None
        self.openContext(data.domain)

        self.update_model()

    def on_controls_changed(self):
        self.apply()

    def check_data(self):
        # TODO
        pass

    def update_model(self):
        self.show_fitting_failed(None)
        self.model = None
        stratified_data = None

        try:
            if self.time_var is not None and self.event_var is not None:
                self.model: CoxRegressionModel = self.learner(self.data, self.time_var.name, self.event_var.name)
        except BaseException as exc:
            self.show_fitting_failed(exc)

        self.Outputs.model.send(self.model)
        if self.model is not None:
            # create coefficients table
            domain = Domain([ContinuousVariable('coef')], metas=[StringVariable('covariate')])
            coef_table = Table.from_list(domain, list(zip(self.model.coefficients, self.model.covariates)))
            coef_table.name = 'coefficients'
            self.Outputs.coefficients.send(coef_table)

            # stratify output data based on predicted risk scores
            risk_score_label = 'Risk Score'
            risk_group_label = 'Risk Group'
            risk_score_var = ContinuousVariable(risk_score_label)
            risk_group_var = DiscreteVariable(risk_group_label, values=['Low Risk', 'High Risk'])

            risk_scores = self.model.predict(self.data.X)
            risk_groups = (risk_scores > np.median(risk_scores)).astype(int)

            domain = Domain(
                self.data.domain.attributes,
                self.data.domain.class_var,
                self.data.domain.metas + (risk_score_var, risk_group_var),
            )
            stratified_data = self.data.transform(domain)
            stratified_data[:, risk_score_var] = np.reshape(risk_scores, (-1, 1))
            stratified_data[:, risk_group_var] = np.reshape(risk_groups, (-1, 1))

        self.Outputs.data.send(stratified_data)

    def create_learner(self):
        alpha = self.alphas[self.alpha_index]

        if self.reg_type == OWCoxRegression.OLS:
            learner = CoxRegressionLearner()
        elif self.reg_type == OWCoxRegression.Ridge:
            learner = CoxRegressionLearner(penalizer=alpha, l1_ratio=0)
        elif self.reg_type == OWCoxRegression.Lasso:
            learner = CoxRegressionLearner(penalizer=alpha, l1_ratio=1)
        elif self.reg_type == OWCoxRegression.Elastic:
            learner = CoxRegressionLearner(penalizer=alpha, l1_ratio=1 - self.l2_ratio)

        return learner

    def handleNewSignals(self):
        self.apply()

    def _reg_type_changed(self):
        self.controls.alpha_index.setEnabled(self.reg_type != self.OLS)
        self.l2_ratio_slider.setEnabled(self.reg_type == self.Elastic)
        self.apply()

    def _set_alpha_label(self):
        self.alpha_label.setText(f"Alpha: {self.alphas[self.alpha_index]}")

    def _alpha_changed(self):
        self._set_alpha_label()
        self.apply()

    def _set_l2_ratio_label(self):
        self.l2_ratio_label.setText("{:.{}f} : {:.{}f}".format(1 - self.l2_ratio, 2, self.l2_ratio, 2))

    def _l2_ratio_changed(self):
        self._set_l2_ratio_label()
        self.apply()

    def get_learner_parameters(self):
        regularization = 'No Regularization'
        if self.reg_type == OWCoxRegression.Ridge:
            regularization = f'Ridge Regression (L2) with α={self.alphas[self.alpha_index]}'
        elif self.reg_type == OWCoxRegression.Lasso:
            regularization = f'Lasso Regression (L1) with α={self.alphas[self.alpha_index]}'
        elif self.reg_type == OWCoxRegression.Elastic:
            regularization = 'Elastic Net Regression with α={}' ' and L1:L2 ratio of {}:{}'.format(
                self.alphas[self.alpha_index], self.l2_ratio, 1 - self.l2_ratio
            )
        return (('Regularization', regularization),)


if __name__ == "__main__":
    WidgetPreview(OWCoxRegression).run(Table("test_data2.tab"))
