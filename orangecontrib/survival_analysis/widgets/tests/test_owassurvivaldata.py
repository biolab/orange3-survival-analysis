import os

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.survival_analysis.widgets.data import (
    TIME_VAR,
    EVENT_VAR,
    TIME_TO_EVENT_VAR,
)
from orangecontrib.survival_analysis.widgets.owassurvivaldata import OWAsSurvivalData


class TestOWAsSurvivalData(WidgetTest):
    def setUp(self) -> None:
        test_data_path = os.path.join(os.path.dirname(__file__), 'datasets')
        self.widget = self.create_widget(OWAsSurvivalData)
        self.send_signal(
            self.widget.Inputs.data, Table(f'{test_data_path}/toy_example.tab')
        )
        self.assertEqual(self.widget.controls.time_var.count(), 1)
        self.assertEqual(self.widget.controls.event_var.count(), 2)

    def test_output_survival_data(self):
        simulate.combobox_activate_item(
            self.widget.controls.time_var, self.widget._data.columns.Time.name
        )
        self.assertEqual(self.widget.time_var.name, self.widget._data.columns.Time.name)

        simulate.combobox_activate_item(
            self.widget.controls.event_var, self.widget._data.columns.Event.name
        )
        self.assertEqual(
            self.widget.event_var.name, self.widget._data.columns.Event.name
        )

        output_data = self.get_output(self.widget.Outputs.data)
        class_vars = output_data.domain.class_vars
        self.assertIsNotNone(class_vars)
        self.assertTrue(len(class_vars) == 2)
        self.assertTrue(all(TIME_TO_EVENT_VAR in t.attributes for t in class_vars))
        self.assertTrue(
            all(
                t.attributes[TIME_TO_EVENT_VAR] in [TIME_VAR, EVENT_VAR]
                for t in class_vars
            )
        )
