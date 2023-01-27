import os

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.survival_analysis.widgets.owassurvivaldata import OWAsSurvivalData
from orangecontrib.survival_analysis.widgets.owcoxregression import OWCoxRegression
from orangecontrib.survival_analysis.widgets.owsurvivalnomogram import (
    OWSurvivalNomogram,
)


class TestOWSurvivalNomogram(WidgetTest):
    def setUp(self) -> None:
        test_data_path = os.path.join(os.path.dirname(__file__), 'datasets')

        self.as_survival = self.create_widget(OWAsSurvivalData)
        self.cox_regression = self.create_widget(OWCoxRegression)
        self.widget = self.create_widget(OWSurvivalNomogram)

        # setup workflow: data -> as survival data -> cox -> survival nomogram
        self.send_signal(
            self.as_survival.Inputs.data,
            Table(f'{test_data_path}/melanoma.tab'),
        )
        simulate.combobox_activate_item(
            self.as_survival.controls.time_var, self.as_survival._data.columns.Time.name
        )
        simulate.combobox_activate_item(
            self.as_survival.controls.event_var,
            self.as_survival._data.columns.Event.name,
        )

        self.send_signal(
            self.cox_regression.Inputs.data,
            self.get_output(self.as_survival.Outputs.data),
        )
        # print(self.get_output(self.as_survival.Outputs.data))
        print(self.get_output(self.cox_regression.Outputs.coefficients))
        print(self.get_output(self.cox_regression.Outputs.model))
        self.send_signal(
            self.widget.Inputs.model, self.get_output(self.cox_regression.Outputs.model)
        )

    def test_widget_initialization(self):
        self.assertIsNotNone(self.widget.data)
        self.assertEqual(len(self.widget.coefficients), 4)
        self.assertEqual(len(self.widget.data_extremes), 4)
        self.assertEqual(len(self.widget.points), 4)
        self.assertEqual(len(self.widget.feature_items), 4)
