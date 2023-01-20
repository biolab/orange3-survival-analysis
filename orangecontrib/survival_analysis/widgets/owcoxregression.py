from itertools import chain

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLayout, QSizePolicy

from Orange.data import Table
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.signals import Output

from orangecontrib.survival_analysis.modeling.cox import CoxRegressionLearner
from orangecontrib.survival_analysis.widgets.data import check_survival_data


class OWCoxRegression(OWBaseLearner):
    name = 'Cox regression'
    description = (
        'Cox proportional-hazards regression with optional L1 (LASSO), '
        'L2 (ridge) or L1L2 (elastic net) regularization.'
    )
    icon = 'icons/owcoxregression.svg'
    priority = 20
    keywords = ['ridge', 'lasso', 'elastic net', 'cox regression']

    LEARNER = CoxRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output('Coefficients', Table, explicit=True)

    class Inputs(OWBaseLearner.Inputs):
        pass

    REGULARIZATION_TYPES = [
        'No regularization',
        'Ridge regression (L2)',
        'Lasso regression (L1)',
        'Elastic net regression',
    ]
    COX, Ridge, Lasso, Elastic = 0, 1, 2, 3

    ridge = settings.Setting(False)
    reg_type = settings.Setting(COX)
    alpha_index: int
    alpha_index = settings.Setting(0)
    l2_ratio: int
    l2_ratio = settings.Setting(0.5)
    autosend = settings.Setting(True)

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
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.hBox(self.controlArea, 'Regularization')
        gui.radioButtons(
            box,
            self,
            'reg_type',
            btnLabels=self.REGULARIZATION_TYPES,
            callback=self._reg_type_changed,
        )

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
            createLabel=False,
            width=120,
            step=0.01,
            callback=self._l2_ratio_changed,
        )
        gui.widgetLabel(box5, 'L2')
        self.l2_ratio_label = gui.widgetLabel(
            box4, "", sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        )
        self.l2_ratio_label.setAlignment(Qt.AlignCenter)

        box5 = gui.hBox(self.controlArea)
        box5.layout().setAlignment(Qt.AlignCenter)
        self._set_l2_ratio_label()
        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.controls.alpha_index.setEnabled(self.reg_type != self.COX)
        self.l2_ratio_slider.setEnabled(self.reg_type == self.Elastic)

    @Inputs.data
    @check_survival_data
    def set_data(self, data):
        """Set the input train dataset."""
        self.Error.data_error.clear()
        self.data = data
        self.update_model()

    def check_data(self):
        self.valid_data = False
        self.Error.sparse_not_supported.clear()
        if self.data is not None and self.learner is not None:
            self.Error.data_error.clear()
            incompatibility_reason = self.learner.incompatibility_reason(
                self.data.domain
            )
            if incompatibility_reason is not None:
                self.Error.data_error(incompatibility_reason)
            elif not len(self.data):
                self.Error.data_error("Dataset is empty.")
            elif self.data.X.size == 0:
                self.Error.data_error("Data has no features to learn from.")
            elif self.data.is_sparse() and not self.supports_sparse:
                self.Error.sparse_not_supported()
            else:
                self.valid_data = True
        return self.valid_data

    def update_model(self):
        super().update_model()
        if self.model is not None:
            self.Outputs.coefficients.send(self.model.summary_to_table())

    def create_learner(self):
        alpha = self.alphas[self.alpha_index]

        if self.reg_type == OWCoxRegression.COX:
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
        self.controls.alpha_index.setEnabled(self.reg_type != self.COX)
        self.l2_ratio_slider.setEnabled(self.reg_type == self.Elastic)
        self.apply()

    def _set_alpha_label(self):
        self.alpha_label.setText(f"Alpha: {self.alphas[self.alpha_index]}")

    def _alpha_changed(self):
        self._set_alpha_label()
        self.apply()

    def _set_l2_ratio_label(self):
        self.l2_ratio_label.setText(
            "{:.{}f} : {:.{}f}".format(1 - self.l2_ratio, 2, self.l2_ratio, 2)
        )

    def _l2_ratio_changed(self):
        self._set_l2_ratio_label()
        self.apply()

    def get_learner_parameters(self):
        regularization = 'No Regularization'
        if self.reg_type == OWCoxRegression.Ridge:
            regularization = (
                f'Ridge Regression (L2) with α={self.alphas[self.alpha_index]}'
            )
        elif self.reg_type == OWCoxRegression.Lasso:
            regularization = (
                f'Lasso Regression (L1) with α={self.alphas[self.alpha_index]}'
            )
        elif self.reg_type == OWCoxRegression.Elastic:
            regularization = (
                'Elastic Net Regression with α={}'
                ' and L1:L2 ratio of {}:{}'.format(
                    self.alphas[self.alpha_index], self.l2_ratio, 1 - self.l2_ratio
                )
            )
        return (('Regularization', regularization),)


if __name__ == "__main__":
    table = Table('http://datasets.biolab.si/core/melanoma.tab')
    WidgetPreview(OWCoxRegression).run(input_data=table)
