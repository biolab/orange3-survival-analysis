import html
import numpy as np
import pyqtgraph as pg

from typing import Dict, List, Optional, NamedTuple
from itertools import zip_longest

from AnyQt.QtGui import QBrush, QColor, QPainterPath, QPalette
from AnyQt.QtCore import Qt, QSize, QEvent
from AnyQt.QtCore import pyqtSignal as Signal
from pyqtgraph.functions import mkPen
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph.graphicsItems.LegendItem import ItemSample, LabelItem
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times, format_p_value
from lifelines.statistics import multivariate_logrank_test

from Orange.data import Table, DiscreteVariable, Domain
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.settings import (
    Setting,
    ContextSetting,
    SettingProvider,
    DomainContextHandler,
)
from Orange.widgets.utils.plot import SELECT, PANNING, ZOOMING, OWPlotGUI
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.annotated_data import (
    create_annotated_table,
    ANNOTATED_DATA_SIGNAL_NAME,
)
from Orange.widgets.visualize.owscatterplotgraph import LegendItem
from Orange.data.filter import IsDefined


from orangecontrib.survival_analysis.widgets.data import (
    check_survival_data,
    get_survival_endpoints,
)


MEDIAN_LINE_PEN_STYLE = {'color': QColor(Qt.darkGray), 'width': 1, 'style': Qt.DashLine}


def create_line_symbol():
    p = QPainterPath()
    p.moveTo(0, 0)
    p.lineTo(0, -1)
    return p


class EstimatedFunctionCurve:
    @staticmethod
    def generate_curve_coordinates(timeline, probabilities):
        intervals = zip_longest(
            zip(timeline, probabilities), zip(timeline[1:], probabilities[:-1])
        )
        x, y = zip(
            *[
                coordinate
                for start, finish in intervals
                for coordinate in (start, finish)
                if coordinate
            ]
        )
        return np.array(x), np.array(y)

    def __init__(self, time, events, label=None, color=None):
        self._kmf = KaplanMeierFitter().fit(
            time.astype(np.float64), events.astype(np.float64)
        )

        self._label: str = label
        self.color: List[int] = color

        # refactor this
        time, survival = self._kmf.survival_function_.reset_index().values.T.tolist()
        lower, upper = self._kmf.confidence_interval_.values.T.tolist()
        self.x, self.y = self.generate_curve_coordinates(time, survival)
        _, self.lower_bound = self.generate_curve_coordinates(time, lower)
        _, self.upper_bound = self.generate_curve_coordinates(time, upper)

        # Estimated function curve
        self.estimated_fun = pg.PlotDataItem(self.x, self.y, pen=self.get_pen())

        # Lower and upper confidence intervals
        pen = self.get_pen(width=1, alpha=70)
        self.lower_conf_limit = pg.PlotDataItem(self.x, self.lower_bound, pen=pen)
        self.upper_conf_limit = pg.PlotDataItem(self.x, self.upper_bound, pen=pen)
        self.confidence_interval = pg.FillBetweenItem(
            self.upper_conf_limit, self.lower_conf_limit, brush=self.get_color(alpha=50)
        )

        self.selection = pg.PlotDataItem(pen=mkPen(color=QColor(Qt.yellow), width=4))
        self.selection.hide()

        self.median_survival = median = np.round(
            median_survival_times(self._kmf.survival_function_.astype(np.float32)), 1
        )
        self.median_vertical = pg.PlotDataItem(
            x=(median, median), y=(0, 0.5), pen=pg.mkPen(**MEDIAN_LINE_PEN_STYLE)
        )

        censored_data = self.get_censored_data()

        self.censored_data = pg.ScatterPlotItem(
            x=censored_data[:, 0],
            y=censored_data[:, 1],
            brush=QBrush(Qt.black),
            pen=self.get_pen(width=1, alpha=255),
            symbol=create_line_symbol(),
            size=15,
        )
        self.censored_data.setZValue(10)

        self.num_of_samples = len(events)
        self.num_of_censored_samples = len(censored_data)

    @property
    def label(self):
        return self._label if self._label else 'All'

    def get_censored_data(self):
        time_events = np.column_stack((self._kmf.durations, self._kmf.event_observed))
        censored_time = time_events[np.argwhere(time_events[:, 1] == 0), 0]
        survival = self._kmf.survival_function_.values
        return np.column_stack(
            (censored_time, survival[np.where(censored_time == self._kmf.timeline)[1]])
        )

    def get_color(self, alpha=255) -> QColor:
        color = QColor(*self.color) if self.color else QColor(Qt.darkGray)
        color.setAlpha(alpha)
        return color

    def get_pen(self, width=3, alpha=100) -> mkPen:
        return mkPen(color=self.get_color(alpha), width=width)

    def set_highlighted(self, highlighted):
        if highlighted:
            estimated_fun_alpha = 200
            conf_limit_alpha = 110
            conf_interval_alpha = 90
        else:
            estimated_fun_alpha = 100
            conf_limit_alpha = 70
            conf_interval_alpha = 50

        self.estimated_fun.setPen(self.get_pen(alpha=estimated_fun_alpha))
        pen = self.get_pen(width=1, alpha=conf_limit_alpha)
        self.lower_conf_limit.setPen(pen)
        self.upper_conf_limit.setPen(pen)
        self.confidence_interval.setBrush(self.get_color(conf_interval_alpha))


class SelectionInterval(NamedTuple):
    x: List[int]
    y: List[int]


class KaplanMeierViewBox(ViewBox):
    selection_changed = Signal(tuple, bool)

    def __init__(self, parent):
        super().__init__(enableMenu=False)
        self.mode = SELECT
        self.parent = parent

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.autoRange()
            self.enableAutoRange()
        else:
            ev.accept()
            self.selection_changed.emit((), True)

    def mouseDragEvent(self, ev, axis=None):
        if self.mode == SELECT and axis is None:
            self.parent.plotItem.scene().blockSignals(True)
            ev.accept()
            if ev.button() == Qt.LeftButton:

                start_pos = self.mapToView(ev.buttonDownPos())
                end_pos = self.mapToView(ev.pos())
                self.selection_changed.emit((start_pos.x(), end_pos.x()), False)
                if ev.isFinish():
                    end_pos = self.mapToView(ev.pos())
                    self.selection_changed.emit((start_pos.x(), end_pos.x()), True)
                    self.parent.plotItem.scene().blockSignals(False)

        elif self.mode == ZOOMING or self.mode == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()


class HLineItemSample(ItemSample):
    def __init__(self):
        super().__init__(None)
        self.pen = pg.mkPen(color=QColor(Qt.darkGray), width=2)

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawLine(0, 0, int(self.width()), 0)


class CustomLegendItem(LegendItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout.setVerticalSpacing(3)
        self._sample_column_label = 'n/N'
        self._median_column_label = 'Median'
        self._log_rank_test_label = 'Log-rank p'

    def set_header(self):
        samples = LabelItem(self._sample_column_label)
        median = LabelItem(self._median_column_label)
        separator = HLineItemSample()

        self.layout.addItem(samples, 1, 2)
        self.layout.addItem(median, 1, 3)
        self.items.append(samples)
        self.items.append(median)

        row = self.layout.rowCount()
        self.layout.addItem(separator, row, 1, row, 3)
        self.items.append(separator)

    def set_curve(self, curve: EstimatedFunctionCurve):
        curve_label = LabelItem(curve.label, color=curve.get_color())
        samples = LabelItem(
            f'{curve.num_of_samples - curve.num_of_censored_samples}/{curve.num_of_samples}'
        )
        median = LabelItem(f'{curve.median_survival}')

        row = self.layout.rowCount()
        self.items.append(curve_label)
        self.items.append(samples)
        self.items.append(median)

        self.layout.addItem(curve_label, row, 1)
        self.layout.addItem(samples, row, 2)
        self.layout.addItem(median, row, 3)

    def set_footer(self, p):
        separator = HLineItemSample()
        log_rank_label = LabelItem(self._log_rank_test_label)
        log_rank_p = LabelItem(html.escape(p))

        row = self.layout.rowCount()
        self.layout.addItem(separator, row, 1, row, 3)
        self.items.append(separator)

        row = self.layout.rowCount()
        self.layout.addItem(log_rank_label, row, 2)
        self.layout.addItem(log_rank_p, row, 3)
        self.items.append(log_rank_label)
        self.items.append(log_rank_p)

    def clear(self):
        for item in self.items:
            self.layout.removeItem(item)
        self.items = []
        self.updateSize()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.PaletteChange:
            color = self.palette().color(QPalette.Text)
            for item in self.items:
                if isinstance(item, LabelItem):
                    item.setText(item.text, color=color)
        super(LegendItem, self).changeEvent(event)


class KaplanMeierPlot(gui.OWComponent, pg.PlotWidget):
    HIGHLIGHT_RADIUS = 20  # in pixels
    selection_changed = Signal()

    selection: Dict[int, Optional[SelectionInterval]] = ContextSetting({})

    def __init__(self, parent: OWWidget = None):
        gui.OWComponent.__init__(self, widget=parent)
        pg.PlotWidget.__init__(self, parent=parent, viewBox=KaplanMeierViewBox(self))

        self.parent: OWWidget = parent
        self.highlighted_curve: Optional[int] = None
        self.curves: Dict[int, EstimatedFunctionCurve] = {}
        self.__selection_items: Dict[int, Optional[pg.PlotDataItem]] = {}

        self.view_box: KaplanMeierViewBox = self.getViewBox()

        self._mouse_moved_signal = pg.SignalProxy(
            self.plotItem.scene().sigMouseMoved,
            slot=self.mouseMovedEvent,
            delay=0.15,
            rateLimit=10,
        )
        self.view_box.selection_changed.connect(self.on_selection_changed)

        self.legend = CustomLegendItem()
        self.legend.setParentItem(self.getViewBox())
        self.legend.restoreAnchor(((1, 0), (1, 0)))
        self.legend.hide()

        self.setLabels(left='Survival Probability', bottom=self.parent.time_var_name)

    def mouseMovedEvent(self, ev):
        pos = self.view_box.mapSceneToView(ev[0])
        mouse_x_pos, mouse_y_pos = pos.x(), pos.y()

        x_pixel_size, y_pixel_size = self.view_box.viewPixelSize()
        x_pixel = self.HIGHLIGHT_RADIUS * x_pixel_size
        y_pixel = self.HIGHLIGHT_RADIUS * y_pixel_size

        for curve_id, curve in self.curves.items():

            points = np.column_stack((curve.x, curve.y))
            line_segments = np.column_stack((points[:-1, :], points[1:, :]))

            mask = np.argwhere(line_segments[:, 0] != line_segments[:, 2])
            horizontal_segments = np.squeeze(line_segments[mask], axis=1)

            mask = np.argwhere(line_segments[:, 1] != line_segments[:, 3])
            vertical_segments = np.squeeze(line_segments[mask], axis=1)

            mouse_on_horizontal_segment = (
                # check X axis
                (horizontal_segments[:, 0] < mouse_x_pos)
                & (mouse_x_pos < horizontal_segments[:, 2])
                # check Y axis
                & (horizontal_segments[:, 1] + y_pixel > mouse_y_pos)
                & (mouse_y_pos > horizontal_segments[:, 3] - y_pixel)
            )
            mouse_on_vertical_segment = (
                # check X axis
                (vertical_segments[:, 0] - x_pixel < mouse_x_pos)
                & (mouse_x_pos < vertical_segments[:, 2] + x_pixel)
                # check Y axis
                & (vertical_segments[:, 1] > mouse_y_pos)
                & (mouse_y_pos > vertical_segments[:, 3])
            )

            if np.any(mouse_on_horizontal_segment) | np.any(mouse_on_vertical_segment):
                self.highlight(curve_id)
                self.view_box.setCursor(Qt.PointingHandCursor)
                return
            else:
                self.view_box.setCursor(Qt.ArrowCursor)
                self.highlight(None)

    def highlight(self, curve_id: Optional[int]):
        old = self.highlighted_curve

        self.highlighted_curve = curve_id
        if self.highlighted_curve is None and old is not None:
            curve = self.curves[old]
            curve.set_highlighted(False)
            return

        if old != self.highlighted_curve:
            curve = self.curves[curve_id]
            curve.set_highlighted(True)

    def clear_selection(self, curve_id: Optional[int] = None):
        """If curve id is None clear all else clear only highlighted curve"""
        if curve_id is not None:
            self.curves[curve_id].selection.hide()
            self.selection = {
                key: val for key, val in self.selection.items() if key != curve_id
            }
            return

        for curve in self.curves.values():
            curve.selection.hide()

        self.selection = {}

    def set_selection(self):
        for curve_id in self.selection.keys():
            self.set_selection_item(curve_id)

    def set_selection_item(self, curve_id: int):
        if curve_id not in self.selection:
            return

        selection = self.selection[curve_id]
        curve = self.curves[curve_id]
        curve.selection.setData(selection.x, selection.y)
        curve.selection.show()

    def on_selection_changed(self, selection_interval, is_finished):
        self.clear_selection(self.highlighted_curve)
        if self.highlighted_curve is None or not selection_interval:
            if is_finished:
                self.selection_changed.emit()
            return

        curve = self.curves[self.highlighted_curve]
        start_x, end_x = sorted(selection_interval)
        if end_x < curve.x[0] or start_x > curve.x[-1]:
            return

        start_x = max(curve.x[0], start_x)
        end_x = min(curve.x[-1], end_x)
        left, right = np.argmax(curve.x > start_x), np.argmax(curve.x > end_x)
        right = right if right else -1

        left_selection = (start_x, curve.y[left])
        right_selection = (end_x, curve.y[right])
        middle_selection = np.column_stack((curve.x, curve.y))[left:right]
        selected = np.vstack((left_selection, middle_selection, right_selection))
        self.selection[self.highlighted_curve] = SelectionInterval(
            selected[:, 0], selected[:, 1]
        )
        self.set_selection_item(self.highlighted_curve)

        if is_finished:
            self.selection_changed.emit()

    def update_plot(self, confidence_interval=False, median=False, censored=False):
        self.clear()
        self.legend.clear()

        if not self.curves:
            return

        if median:
            self.addItem(
                pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen(**MEDIAN_LINE_PEN_STYLE))
            )

        for curve in self.curves.values():
            self.addItem(curve.estimated_fun)
            self.addItem(curve.selection)

            if confidence_interval:
                self.addItem(curve.lower_conf_limit)
                self.addItem(curve.upper_conf_limit)
                self.addItem(curve.confidence_interval)

            if median:
                self.addItem(curve.median_vertical)

            if censored:
                self.addItem(curve.censored_data)

        self.set_selection()
        self.update_legend()
        self.setLabels(bottom=self.parent.time_var_name)

    def update_legend(self):
        self.legend.hide()

        self.legend.set_header()
        for curve in list(self.curves.values()):
            self.legend.set_curve(curve)
        if len(self.curves) > 1:
            self.legend.set_footer(format_p_value(2)(self.parent.log_rank_test.p_value))
        self.legend.updateSize()

        if bool(len(self.legend.items)):
            self.legend.show()

    def select_button_clicked(self):
        self.view_box.mode = SELECT
        self.view_box.setMouseMode(self.view_box.RectMode)

    def pan_button_clicked(self):
        self.view_box.mode = PANNING
        self.view_box.setMouseMode(self.view_box.PanMode)

    def zoom_button_clicked(self):
        self.view_box.mode = ZOOMING
        self.view_box.setMouseMode(self.view_box.RectMode)

    def reset_button_clicked(self):
        self.view_box.autoRange()
        self.view_box.enableAutoRange()


class OWKaplanMeier(OWWidget):
    name = 'Kaplan-Meier Plot'
    description = 'Visualisation of Kaplan-Meier estimator.'
    icon = 'icons/owkaplanmeier.svg'
    priority = 10
    keywords = ['Kaplan-Meier', 'survival curve', 'log-rank']

    show_confidence_interval: bool
    show_confidence_interval = Setting(False)

    show_median_line: bool
    show_median_line = Setting(False)

    show_censored_data: bool
    show_censored_data = Setting(False)

    settingsHandler = DomainContextHandler()
    group_var: Optional[DiscreteVariable] = ContextSetting(None, schema_only=True)

    graph = SettingProvider(KaplanMeierPlot)
    graph_name = 'graph.plotItem'

    auto_commit: bool = Setting(True, schema_only=True)

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        selected_data = Output('Selected Data', Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data: Optional[Table] = None
        self.plot_curves = None
        self.log_rank_test = None

        group_var_model = DomainModel(
            placeholder='(None)', valid_types=(DiscreteVariable,)
        )

        box = gui.vBox(self.controlArea, 'Group', margin=0)
        gui.comboBox(
            box,
            self,
            'group_var',
            model=group_var_model,
            callback=self.on_group_changed,
        )

        box = gui.vBox(self.controlArea, 'Display options')
        gui.checkBox(
            box,
            self,
            'show_confidence_interval',
            label='Confidence intervals',
            callback=self.on_display_option_changed,
        )

        gui.checkBox(
            box,
            self,
            'show_median_line',
            label='Median',
            callback=self.on_display_option_changed,
        )

        gui.checkBox(
            box,
            self,
            'show_censored_data',
            label='Censored data',
            callback=self.on_display_option_changed,
        )

        self.graph: KaplanMeierPlot = KaplanMeierPlot(parent=self)
        self.graph.selection_changed.connect(self.commit.deferred)
        self.mainArea.layout().addWidget(self.graph)

        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.controlArea)

        gui.rubber(self.controlArea)

        self.commit_button = gui.auto_commit(
            self.controlArea, self, 'auto_commit', '&Commit', box=False
        )

    @property
    def time_var(self):
        if not self.data:
            return

        time_var, _ = get_survival_endpoints(self.data.domain)
        return time_var

    @property
    def event_var(self):
        if not self.data:
            return

        _, event_var = get_survival_endpoints(self.data.domain)
        return event_var

    @property
    def time_var_name(self):
        if not self.time_var:
            return 'Time'

        return self.time_var.name

    @Inputs.data
    @check_survival_data
    def set_data(self, data: Table):
        self.closeContext()

        if data and data.has_missing():
            filter_ = IsDefined(columns=data.domain.class_vars)
            self.data = filter_(data)
        else:
            self.data = data

        # exclude class vars
        domain = data.domain if data else None
        if domain is not None:
            domain = Domain(domain.attributes, metas=domain.metas)

        self.controls.group_var.model().set_domain(domain)
        if domain is not None:
            group_var_model = self.controls.group_var.model()
            if 'Cohorts' in domain and domain['Cohorts'] in group_var_model:
                self.group_var = domain['Cohorts']
            else:
                self.group_var = None

        self.graph.selection = {}
        self.openContext(domain)

        self.graph.curves = {
            curve_id: curve
            for curve_id, curve in enumerate(self.generate_plot_curves())
        }
        self.graph.update_plot(**self._get_plot_options())

        self.commit.now()

    def _get_plot_options(self):
        return {
            'confidence_interval': self.show_confidence_interval,
            'median': self.show_median_line,
            'censored': self.show_censored_data,
        }

    def on_display_option_changed(self) -> None:
        self.graph.update_plot(**self._get_plot_options())

    def on_group_changed(self):
        if not self.data:
            return

        self.graph.curves = {
            curve_id: curve
            for curve_id, curve in enumerate(self.generate_plot_curves())
        }
        self.graph.clear_selection()
        self.graph.update_plot(**self._get_plot_options())
        self.commit.now()

    def _get_discrete_var_color(self, index: Optional[int]):
        if self.group_var is not None and index is not None:
            return list(self.group_var.colors[index])

    def generate_plot_curves(self) -> List[EstimatedFunctionCurve]:
        if self.time_var is None or self.event_var is None:
            return []

        data = self.data
        time = data.get_column(self.time_var)
        events = data.get_column(self.event_var)

        if self.group_var:
            groups = data.get_column(self.group_var.name)
            group_indexes = [index for index, _ in enumerate(self.group_var.values)]
            colors = [self._get_discrete_var_color(index) for index in group_indexes]
            masks = groups == np.reshape(group_indexes, (-1, 1))
            self.log_rank_test = multivariate_logrank_test(time, groups, events)

            return [
                EstimatedFunctionCurve(
                    time[mask], events[mask], color=color, label=label
                )
                for mask, color, label in zip(masks, colors, self.group_var.values)
                if mask.any()
            ]

        else:
            return [EstimatedFunctionCurve(time, events)]

    @gui.deferred
    def commit(self):
        if not self.graph.selection:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            return

        data = self.data

        time = data.get_column(self.time_var)
        if self.group_var is None:
            time_interval = self.graph.selection[0].x
            start, end = time_interval[0], time_interval[-1]
            selection = (
                np.argwhere((time >= start) & (time <= end)).reshape(-1).astype(int)
            )
        else:
            selection = []
            group = data.get_column(self.group_var.name)
            for group_id, time_interval in self.graph.selection.items():
                start, end = time_interval.x[0], time_interval.x[-1]
                selection += (
                    np.argwhere((time >= start) & (time <= end) & (group == group_id))
                    .reshape(-1)
                    .astype(int)
                    .tolist()
                )
            selection = sorted(selection)

        self.Outputs.selected_data.send(data[selection, :])
        self.Outputs.annotated_data.send(create_annotated_table(data, selection))

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()

    def sizeHint(self):
        return QSize(1280, 620)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    table = Table('http://datasets.biolab.si/core/melanoma.tab')
    WidgetPreview(OWKaplanMeier).run(input_data=table)
