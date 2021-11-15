import os

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.survival_analysis.widgets.data import TIME_COLUMN, EVENT_COLUMN
from orangecontrib.survival_analysis.widgets.owassurvivaldata import OWAsSurvivalData


class TestOWAsSurvivalData(WidgetTest):
    def setUp(self) -> None:
        test_data_path = os.path.join(os.path.dirname(__file__), 'datasets')
        self.widget = self.create_widget(OWAsSurvivalData)
        self.send_signal(self.widget.Inputs.data, Table(f'{test_data_path}/toy_example.tab'))
        self.assertEqual(self.widget.controls.time_var.count(), 1)
        self.assertEqual(self.widget.controls.event_var.count(), 3)

    def test_output_survival_data(self):
        simulate.combobox_activate_item(self.widget.controls.time_var, self.widget._data.columns.Time.name)
        self.assertEqual(self.widget.time_var.name, self.widget._data.columns.Time.name)

        simulate.combobox_activate_item(self.widget.controls.event_var, self.widget._data.columns.Event.name)
        self.assertEqual(self.widget.event_var.name, self.widget._data.columns.Event.name)

        output_data = self.get_output(self.widget.Outputs.data)
        self.assertTrue(len(output_data.attributes))
        self.assertIn(TIME_COLUMN, output_data.attributes)
        self.assertIn(EVENT_COLUMN, output_data.attributes)
        self.assertEqual(output_data.attributes[TIME_COLUMN].name, self.widget._data.columns.Time.name)
        self.assertEqual(output_data.attributes[EVENT_COLUMN].name, self.widget._data.columns.Event.name)
