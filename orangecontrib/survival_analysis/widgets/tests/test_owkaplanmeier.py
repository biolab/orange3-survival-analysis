import os

import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import LabelItem

from AnyQt.QtCore import Qt, QEvent
from AnyQt.QtGui import QMouseEvent
from AnyQt.QtWidgets import QApplication, QGraphicsView
from AnyQt.QtTest import QTest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest, simulate
from orangecontrib.survival_analysis.widgets.owassurvivaldata import OWAsSurvivalData
from orangecontrib.survival_analysis.widgets.owkaplanmeier import OWKaplanMeier

from orangecontrib.survival_analysis.widgets.data import get_survival_endpoints


def mouse_press_and_hold(widget, pos, mouse_button=Qt.LeftButton):
    if isinstance(widget, QGraphicsView):
        widget = widget.viewport()

    event = QMouseEvent(
        QEvent.MouseButtonPress, pos, mouse_button, Qt.NoButton, Qt.NoModifier
    )
    QApplication.sendEvent(widget, event)


def mouse_release(widget, pos, mouse_button=Qt.LeftButton):
    if isinstance(widget, QGraphicsView):
        widget = widget.viewport()

    event = QMouseEvent(
        QEvent.MouseButtonRelease, pos, mouse_button, Qt.NoButton, Qt.NoModifier
    )
    QApplication.sendEvent(widget, event)


def mouse_move(widget, pos, buttons=Qt.NoButton):
    if isinstance(widget, QGraphicsView):
        widget = widget.viewport()

    event = QMouseEvent(QEvent.MouseMove, pos, Qt.NoButton, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


class TestOWKaplanMeier(WidgetTest):
    def setUp(self) -> None:
        self.test_data_path = os.path.join(os.path.dirname(__file__), 'datasets')
        # create widgets
        self.as_survival = self.create_widget(OWAsSurvivalData)
        self.widget = self.create_widget(OWKaplanMeier)

        # handle survival data
        self.send_signal(
            self.as_survival.Inputs.data,
            Table(f'{self.test_data_path}/toy_example.tab'),
        )
        simulate.combobox_activate_item(
            self.as_survival.controls.time_var, self.as_survival._data.columns.Time.name
        )
        simulate.combobox_activate_item(
            self.as_survival.controls.event_var,
            self.as_survival._data.columns.Event.name,
        )
        self.send_signal(
            self.widget.Inputs.data, self.get_output(self.as_survival.Outputs.data)
        )

        # check survival data
        time_var, event_var = get_survival_endpoints(self.widget.data.domain)
        self.assertEqual(time_var.name, 'Time')
        self.assertEqual(event_var.name, 'Event')
        self.assertIn(time_var, self.widget.data.domain.class_vars)
        self.assertIn(event_var, self.widget.data.domain.class_vars)

        # check if missing data detected
        self.assertTrue(self.widget.Warning.missing_values_detected.is_shown())

        self.widget.auto_commit = True

        # If we don't do this function ViewBox.mapSceneToView
        # fails with num py.linalg.LinAlgError: Singular matrix
        vb = self.widget.graph.getViewBox()
        vb.resize(200, 200)

    def simulate_mouse_drag(self, start: tuple, end: tuple):
        start = self.widget.graph.view_box.mapViewToScene(pg.Point(start[0], start[1]))
        end = self.widget.graph.view_box.mapViewToScene(pg.Point(end[0], end[1]))

        mouse_move(self.widget.graph, start)
        # this is somehow not respected in KaplanMeierViewBox.mouseDragEvent
        # so we do it here manualy
        self.widget.graph.plotItem.scene().blockSignals(True)

        QTest.qWait(100)
        mouse_press_and_hold(self.widget.graph, start)
        mouse_move(self.widget.graph, end, Qt.LeftButton)
        QTest.qWait(100)
        mouse_release(self.widget.graph, end)

        self.widget.graph.plotItem.scene().blockSignals(False)

    def test_incorrect_input_data(self):
        self.send_signal(
            self.widget.Inputs.data, Table(f'{self.test_data_path}/toy_example.tab')
        )
        self.assertTrue(self.widget.Error.missing_survival_data.is_shown())
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.widget.time_var)
        self.assertIsNone(self.widget.event_var)

    def test_group_variable(self):
        # we expect only one curve
        self.assertTrue(len(self.widget.graph.curves) == 1)

        # there should be only 2 items (empty selection and curve)
        items = [
            item
            for item in self.widget.graph.sceneObj.items()
            if isinstance(item, pg.PlotDataItem)
        ]
        self.assertTrue(len(items) == 2)

        # select group
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_group_changed()

        # we expect three curves
        self.assertTrue(len(self.widget.graph.curves) == 2)

        # there should be only 4 items:
        #   - 2 empty selection items
        #   - 2 curves
        items = [
            item
            for item in self.widget.graph.sceneObj.items()
            if isinstance(item, pg.PlotDataItem)
        ]
        self.assertTrue(len(items) == 4)

    def test_legend(self):
        legend = tuple(
            label.text
            for label in self.widget.graph.legend.items
            if isinstance(label, LabelItem)
        )
        self.assertIn('All', legend)

        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_group_changed()

        legend = tuple(
            label.text
            for label in self.widget.graph.legend.items
            if isinstance(label, LabelItem)
        )
        for group in self.widget.group_var.values:
            self.assertIn(group, legend)

    def test_curve_highlight(self):
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_group_changed()

        pos = self.widget.graph.view_box.mapViewToScene(pg.Point(1.5, 0.5))
        mouse_move(self.widget.graph, pos)
        # We need to wait for events to process
        QTest.qWait(100)
        self.assertTrue(self.widget.graph.highlighted_curve == 1)

        pos = self.widget.graph.view_box.mapViewToScene(pg.Point(1.5, 0.85))
        mouse_move(self.widget.graph, pos)
        QTest.qWait(100)
        self.assertTrue(self.widget.graph.highlighted_curve == 0)

    def test_selection(self):
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected_data)

        self.widget.graph.legend.hide()
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
        selected_groups = selected_data.get_column('Group')
        self.assertEqual(12, selected_groups.size)
        selected_groups = set(selected_data.get_column('Group'))
        self.assertEqual(2, len(selected_groups))
        self.assertIn(0, selected_groups)
        self.assertIn(1, selected_groups)

        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_group_changed()

        self.widget.graph.legend.hide()
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
        selected_groups = selected_data.get_column('Group')
        self.assertEqual(6, selected_groups.size)
        selected_groups = set(selected_data.get_column('Group'))
        self.assertEqual(1, len(selected_groups))
        self.assertIn(0, selected_groups)
        self.assertNotIn(1, selected_groups)

        # reset selection
        pos = self.widget.graph.view_box.mapViewToScene(pg.Point(0, 0)).toPoint()
        QTest.mouseClick(self.widget.graph.viewport(), Qt.LeftButton, pos=pos)
        QTest.qWait(100)

        selected_data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected_data)
        self.assertEqual(0, len(self.widget.graph.selection))

        # test selection of a second group
        self.widget.graph.legend.hide()
        self.simulate_mouse_drag((0.4, 0.8), (6, 0.8))

        # check if correct curve is selected
        self.assertTrue(1, len(self.widget.graph.selection))
        self.assertIn(1, self.widget.graph.selection.keys())

        # check output data
        selected_data = self.get_output(self.widget.Outputs.selected_data)
        selected_groups = selected_data.get_column('Group')
        self.assertEqual(6, selected_groups.size)
        selected_groups = set(selected_data.get_column('Group'))
        self.assertEqual(1, len(selected_groups))
        self.assertIn(1, selected_groups)
        self.assertNotIn(0, selected_groups)

    def test_median_value(self):
        self.widget.group_var = self.widget.data.domain['Group']
        self.widget.on_group_changed()

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
        plot_items = [
            item
            for item in self.widget.graph.sceneObj.items()
            if isinstance(item, pg.PlotDataItem)
        ]
        scatter_items = [
            item
            for item in self.widget.graph.sceneObj.items()
            if isinstance(item, pg.ScatterPlotItem)
        ]
        infinite_line = [
            item
            for item in self.widget.graph.sceneObj.items()
            if isinstance(item, pg.InfiniteLine)
        ]

        self.assertEqual(5, len(plot_items))
        self.assertEqual(6, len(scatter_items))
        self.assertEqual(1, len(infinite_line))
