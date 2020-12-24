import pyqtgraph as pg
from pyqtgraph.tests import mouseMove, mousePress, mouseRelease, mouseClick

from Orange.data.table import Table, Domain, StringVariable, ContinuousVariable, DiscreteVariable
from orangewidget.tests.base import WidgetTest
from orangecontrib.survival_analysis.widgets.owkaplanmeier import OWKaplanMeier
from pyqtgraph.Qt import QtTest

from AnyQt.QtCore import Qt


class TestOWKaplanMeier(WidgetTest):
    def setUp(self) -> None:
        domain = Domain(
            [],
            metas=[
                StringVariable('Subject'),
                ContinuousVariable('Time'),
                DiscreteVariable('Event', values=('no', 'yes')),
                DiscreteVariable('Group', values=('group1', 'group2')),
            ],
        )
        test_data = Table.from_list(
            domain,
            [
                ['B', 1, 1, 0],
                ['E', 2, 1, 0],
                ['F', 3, 1, 0],
                ['A', 4, 1, 0],
                ['D', 4.5, 1, 0],
                ['C', 5, 0, 0],
                ['U', 0.5, 1, 1],
                ['Z', 0.75, 1, 1],
                ['W', 1, 1, 1],
                ['V', 1.5, 0, 1],
                ['X', 2, 1, 1],
                ['Y', 3.5, 1, 1],
            ],
        )

        self.widget = self.create_widget(OWKaplanMeier)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.widget.time_var = self.widget.data.domain['Time']
        self.widget.event_var = self.widget.data.domain['Event']
        self.widget.on_controls_changed()

        # If we don't do this function ViewBox.mapSceneToView fails with num py.linalg.LinAlgError: Singular matrix
        vb = self.widget.graph.getViewBox()
        vb.resize(200, 200)

    def simulate_mouse_drag(self, start: tuple, end: tuple):
        start = self.widget.graph.view_box.mapViewToScene(pg.Point(start[0], start[1])).toPoint()
        end = self.widget.graph.view_box.mapViewToScene(pg.Point(end[0], end[1])).toPoint()

        mouseMove(self.widget.graph, start)
        QtTest.QTest.qWait(100)
        mousePress(self.widget.graph, start, Qt.LeftButton)
        mouseMove(self.widget.graph, end, Qt.LeftButton)
        mouseRelease(self.widget.graph, end, Qt.LeftButton)
        QtTest.QTest.qWait(100)

    def test_group_variable(self):
        # we expect only one curve
        self.assertTrue(len(self.widget.graph.curves) == 1)

        # there should be only 2 items (empty selection and curve)
        items = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.PlotDataItem)]
        self.assertTrue(len(items) == 2)

        # select group
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_controls_changed()

        # we expect three curves
        self.assertTrue(len(self.widget.graph.curves) == 2)

        # there should be only 4 items:
        #   - 2 empty selection items
        #   - 2 curves
        items = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.PlotDataItem)]
        self.assertTrue(len(items) == 4)

    def test_legend(self):
        self.assertFalse(self.widget.graph.legend.items)
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_controls_changed()
        legend_text = tuple(label.text for _, label in self.widget.graph.legend.items)
        self.assertEqual(self.widget.group_var.values, legend_text)

    def test_curve_highlight(self):
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_controls_changed()

        pos = self.widget.graph.view_box.mapViewToScene(pg.Point(1.5, 0.5)).toPoint()
        mouseMove(self.widget.graph, pos)
        # We need to wait for events to process
        QtTest.QTest.qWait(100)
        self.assertTrue(self.widget.graph.highlighted_curve == 1)

        pos = self.widget.graph.view_box.mapViewToScene(pg.Point(1.5, 0.85)).toPoint()
        mouseMove(self.widget.graph, pos)
        QtTest.QTest.qWait(100)
        self.assertTrue(self.widget.graph.highlighted_curve == 0)

    def test_selection(self):
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected_data)

        self.simulate_mouse_drag((0.1, 1), (6, 1))

        # check if correct curve is selected
        self.assertEqual(1, len(self.widget.graph.selection))
        self.assertIn(0, self.widget.graph.selection.keys())

        # check if correct intervals are selected
        selection_interval = self.widget.graph.selection[0]
        self.assertEqual(0.1, round(selection_interval.x[0], 1))
        self.assertEqual(5.0, round(selection_interval.x[-1]))
        self.assertEqual(1.0, round(selection_interval.y[0]))
        self.assertEqual(0.1, round(selection_interval.y[-1], 1))

        # check output data
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        selected_groups = selected_data.get_column_view('Group')[0]
        self.assertEqual(12, selected_groups.size)
        selected_groups = set(selected_data.get_column_view('Group')[0])
        self.assertEqual(2, len(selected_groups))
        self.assertIn(0, selected_groups)
        self.assertIn(1, selected_groups)

        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_controls_changed()

        self.simulate_mouse_drag((0.1, 1), (6, 1))

        # check if correct curve is selected
        self.assertTrue(1, len(self.widget.graph.selection))
        self.assertIn(0, self.widget.graph.selection.keys())

        # check if correct intervals are selected
        selection_interval = self.widget.graph.selection[0]
        self.assertEqual(0.1, round(selection_interval.x[0], 1))
        self.assertEqual(5.0, round(selection_interval.x[-1]))
        self.assertEqual(1.0, round(selection_interval.y[0]))
        self.assertEqual(0.2, round(selection_interval.y[-1], 1))

        # check output data
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        selected_groups = selected_data.get_column_view('Group')[0]
        self.assertEqual(6, selected_groups.size)
        selected_groups = set(selected_data.get_column_view('Group')[0])
        self.assertEqual(1, len(selected_groups))
        self.assertIn(0, selected_groups)
        self.assertNotIn(1, selected_groups)

        # reset selection
        pos = self.widget.graph.view_box.mapViewToScene(pg.Point(0, 0)).toPoint()
        mouseClick(self.widget.graph, pos, Qt.LeftButton)
        QtTest.QTest.qWait(100)

        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected_data)
        self.assertEqual(0, len(self.widget.graph.selection))

        # test selection of a second group
        self.simulate_mouse_drag((0.4, 0.8), (6, 0.8))

        # check if correct curve is selected
        self.assertTrue(1, len(self.widget.graph.selection))
        self.assertIn(1, self.widget.graph.selection.keys())

        # check output data
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        selected_groups = selected_data.get_column_view('Group')[0]
        self.assertEqual(6, selected_groups.size)
        selected_groups = set(selected_data.get_column_view('Group')[0])
        self.assertEqual(1, len(selected_groups))
        self.assertIn(1, selected_groups)
        self.assertNotIn(0, selected_groups)

    def test_median_value(self):
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_controls_changed()

        # check if X coordinates are correct
        self.assertEqual(3.0, self.widget.graph.curves[0].median_survival)
        self.assertEqual(1.0, self.widget.graph.curves[1].median_survival)

    def test_censored_data(self):
        censored_data = self.widget.graph.curves[0].get_censored_data()

        # check if X coordinates are correct
        self.assertEqual([5.0, 1.5], censored_data[:, 0].tolist())

    def test_display_options(self):
        self.widget.show_confidence_interval = True
        self.widget.show_censored_data = True
        self.widget.show_median_line = True
        self.widget.on_display_option_changed()

        # check if all scene items are plotted after display option has changed
        plot_items = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.PlotDataItem)]
        scatter_items = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.ScatterPlotItem)]
        infinite_line = [item for item in self.widget.graph.sceneObj.items() if isinstance(item, pg.InfiniteLine)]

        self.assertEqual(5, len(plot_items))
        self.assertEqual(6, len(scatter_items))
        self.assertEqual(1, len(infinite_line))
