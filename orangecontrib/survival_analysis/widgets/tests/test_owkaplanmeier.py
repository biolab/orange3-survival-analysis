import pyqtgraph as pg
from Orange.data.table import Table
from orangewidget.tests.base import WidgetTest
from orangecontrib.survival_analysis.widgets.owkaplanmeier import OWKaplanMeier


class TestOWKaplanMeier(WidgetTest):
    # TODO: add more tests

    def setUp(self) -> None:
        self.widget = self.create_widget(OWKaplanMeier)

    def test_controls_changed(self):
        self.send_signal(self.widget.Inputs.data, Table('iris'))
        self.widget.time_var = self.widget.data.domain['sepal length']
        self.widget.event_var = self.widget.data.domain['sepal width']
        self.widget.on_controls_changed()

        # we expect only one curve
        curves = self.widget.graph.curves
        self.assertTrue(len(curves) == 1)

        # there should be only 2 items (empty selection and curve)
        items = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.PlotDataItem)]
        self.assertTrue(len(items) == 2)

        # select group
        self.widget.group_var = self.widget.data.domain['iris']
        self.widget.on_controls_changed()

        # we expect three curves
        curves = self.widget.graph.curves
        self.assertTrue(len(curves) == 3)

        # there should be only 6 items:
        #   - 3 empty selection items
        #   - 3 curves
        items = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.PlotDataItem)]
        self.assertTrue(len(items) == 6)
