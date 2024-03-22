"""
The code for Survival Nomogram is taken from the
Orange3 repository - Orange.widgets.visualize.ownomogram - and adapted
for Cox regression model.
"""

from itertools import chain
from collections import OrderedDict
from enum import IntEnum
from functools import singledispatch

import time
import numpy as np


from AnyQt.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
    QGraphicsLineItem,
    QGraphicsWidget,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QGraphicsLinearLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)
from AnyQt.QtGui import QColor, QPainter, QFont, QPen, QBrush, QFontMetrics
from AnyQt.QtCore import Qt, QRectF, QSize, QPropertyAnimation, QObject, pyqtProperty


from Orange.data import Table, DiscreteVariable, ContinuousVariable
from Orange.statistics.util import nanmin, nanmax, nanmean, unique

from Orange.base import Model
from Orange.widgets.settings import Setting, ClassValuesContextHandler
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output, AttributeList
from Orange.widgets import gui


from orangecontrib.survival_analysis.modeling.cox import (
    CoxRegressionModel,
)

DOT_COLOR = QColor(170, 220, 255, 255)


def collides(item, items):
    return any(item.collidesWithItem(i) for i in items)


class SortBy(IntEnum):
    NO_SORTING, NAME, ABSOLUTE, POSITIVE, NEGATIVE = 0, 1, 2, 3, 4

    @staticmethod
    def items():
        return [
            "Original order",
            "Alphabetically",
            "Absolute importance",
            "Positive influence",
            "Negative influence",
        ]


@singledispatch
def _get_labels(*_):
    return []


@_get_labels.register(DiscreteVariable)
def _(var: DiscreteVariable, *_):
    return np.array(var.values)


@_get_labels.register(ContinuousVariable)
def _(_: ContinuousVariable, data_extremes: list, values: np.ndarray, *__):
    diff_ = np.nan_to_num(values[-1] - values[0])
    k = (data_extremes[1] - data_extremes[0]) / diff_ if diff_ else 0
    return [str(np.round(v * k + data_extremes[0], 1)) for v in values]


class GraphicsColorAnimator(QObject):
    @pyqtProperty(QColor)
    def brushColor(self):
        return self.__brushColor

    @brushColor.setter
    def brushColor(self, value):
        self.__brushColor = value
        for item in self.__items:
            item.setBrush(value)

    def __init__(self, parent, duration, key_values):
        super().__init__(parent)
        self.__items = []
        self.__default_color = default_color = key_values[0][1]
        self.__brushColor = default_color

        self.__animation = QPropertyAnimation(self, b'brushColor', self)
        self.__animation.setStartValue(default_color)
        self.__animation.setEndValue(default_color)
        self.__animation.setDuration(duration)
        self.__animation.setKeyValues(key_values)
        self.__animation.setLoopCount(-1)

    def setGraphicsItems(self, items):
        if self.__animation.state() == QPropertyAnimation.Running:
            self.__animation.stop()
        self.__items = items
        for item in items:
            item.hookOnMousePress(self.stop)

    def start(self):
        self.__animation.start()

    def stop(self):
        if self.__animation.state() != QPropertyAnimation.Running:
            return
        self.__animation.stop()
        for item in self.__items:
            item.setBrush(self.__default_color)

    def clear(self):
        for item in self.__items:
            item.unhookOnMousePress()
        self.__items = []


class MovableToolTip(QLabel):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setWindowFlags(Qt.ToolTip)
        self.hide()

    def show(self, pos, text, change_y=True):
        self.setText(text)
        self.adjustSize()

        x, y = pos.x(), (pos.y() + 15 if change_y else self.y())
        avail = self.screen().availableGeometry()
        if x + self.width() > avail.right():
            x -= self.width()
        if y + self.height() > avail.bottom():
            y = pos.y() - 10 - self.height() if change_y else self.y() - self.height()
        self.move(x, y)
        super().show()


class DotItem(QGraphicsEllipseItem):
    TOOLTIP_STYLE = """ul {margin-top: 1px; margin-bottom: 1px;}"""
    TOOLTIP_TEMPLATE = """<html><head><style type="text/css">{}</style>
    </head><body><b>{}</b><hr/>{}</body></html>
    """

    def __init__(self, radius, scale, offset, min_x, max_x):
        super().__init__(0, 0, radius, radius)
        self._min_x = min_x * scale - radius / 2 + offset
        self._max_x = max_x * scale - radius / 2 + offset
        self._scale = scale
        self._offset = offset
        self.setPos(0, -radius / 2)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setBrush(DOT_COLOR)
        self.setPen(QPen(QBrush(QColor(20, 130, 250, 255)), 2))
        self.setZValue(100)
        self.tool_tip = MovableToolTip()
        self.setAcceptHoverEvents(True)

    @property
    def value(self):
        return (self.x() + self.rect().width() / 2 - self._offset) / self._scale

    def move(self, x):
        self.setX(x)

    def move_to_val(self, val):
        x = np.clip(
            self._scale * val - self.rect().width() / 2 + self._offset,
            self._min_x,
            self._max_x,
        )
        self.move(x)

    def hoverEnterEvent(self, event):
        self.tool_tip.show(event.screenPos(), self.get_tooltip_text())

    def hoverLeaveEvent(self, event):
        self.tool_tip.hide()

    def mouseMoveEvent(self, _):
        # Prevent click-moving of these items
        return


class MovableDotItem(DotItem):
    def __init__(self, radius, scale, offset, min_x, max_x):
        self.tooltip_labels = []
        self.tooltip_values = []
        super().__init__(radius, scale, offset, min_x, max_x)
        self._x = min_x * scale - radius / 2 + offset
        self._point_dot = None
        self._total_dot = None
        self._probs_dot = None
        self._vertical_line = None
        self._mousePressFunc = None

    @property
    def vertical_line(self):
        return self._vertical_line

    @vertical_line.setter
    def vertical_line(self, line):
        line.setVisible(False)
        self._vertical_line = line

    @property
    def point_dot(self):
        return self._point_dot

    @point_dot.setter
    def point_dot(self, dot):
        dot.setVisible(False)
        self._point_dot = dot

    @property
    def total_dot(self):
        return self._total_dot

    @total_dot.setter
    def total_dot(self, dot):
        self._total_dot = dot
        self._total_dot.movable_dot_items.append(self)

    @property
    def probs_dot(self):
        return self._probs_dot

    @probs_dot.setter
    def probs_dot(self, dot):
        self._probs_dot = dot
        self._probs_dot.movable_dot_items.append(self)

    def hookOnMousePress(self, func):
        self._mousePressFunc = func

    def unhookOnMousePress(self):
        self._mousePressFunc = None

    def mousePressEvent(self, event):
        if self._mousePressFunc:
            self._mousePressFunc()
            self._mousePressFunc = None
        self.tool_tip.show(event.screenPos(), self.get_tooltip_text(), False)
        self._x = event.pos().x()
        self.setBrush(QColor(50, 180, 250, 255))
        self._show_vertical_line_and_point_dot()
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.tool_tip.show(event.screenPos(), self.get_tooltip_text(), False)
        delta_x = event.pos().x() - self._x
        if self._min_x <= self.x() + delta_x <= self._max_x:
            self.move(self.x() + delta_x)
            mod_tooltip_values = [0] + list(self.tooltip_values)
            if np.round(self.value, 1) in np.round(mod_tooltip_values, 1):
                index = np.where(
                    np.round(mod_tooltip_values, 1) == np.round(self.value, 1)
                )
                time.sleep(0.05)
                self.move_to_val(mod_tooltip_values[index[0][0]])
        elif self.x() + delta_x < self._min_x:
            self.move(self._min_x)
        elif self.x() + delta_x > self._max_x:
            self.move(self._max_x)
        self._show_vertical_line_and_point_dot()
        self.probs_dot.move_to_sum()

    def mouseReleaseEvent(self, event):
        self.tool_tip.hide()
        self.setBrush(QColor(170, 220, 255, 255))
        self.point_dot.setVisible(False)
        self.vertical_line.setVisible(False)
        return super().mousePressEvent(event)

    def _show_vertical_line_and_point_dot(self):
        self.vertical_line.setX(self.x() + self.rect().width() / 2 - self._offset)
        self.vertical_line.setVisible(True)
        self.point_dot.move_to_val(self.value)
        self.point_dot.setVisible(True)


class ContinuousItemMixin:
    def get_tooltip_text(self):
        return self.TOOLTIP_TEMPLATE.format(
            self.TOOLTIP_STYLE,
            f"Points: {np.round(self.value, 2)}",
            f"Value: {np.round(self._get_tooltip_label_value(), 1)}",
        )

    def _get_tooltip_label_value(self):
        if not len(self.tooltip_labels):
            return self.value
        start = float(self.tooltip_labels[0])
        stop = float(self.tooltip_labels[-1])
        delta = self.tooltip_values[-1] - self.tooltip_values[0]
        if not delta:
            return np.nan
        return start + self.value * (stop - start) / delta


class ContinuousMovableDotItem(MovableDotItem, ContinuousItemMixin):
    pass


class DiscreteMovableDotItem(MovableDotItem):
    def get_tooltip_text(self):
        labels = self._get_tooltip_labels_with_percentages()
        return self.TOOLTIP_TEMPLATE.format(
            self.TOOLTIP_STYLE,
            f"Points: {np.round(self.value, 2)}",
            "".join(f"{k}: {v:.0%}<br/>" for k, v in labels)[:-5],
        )

    def _get_tooltip_labels_with_percentages(self):
        if not len(self.tooltip_labels):
            return []

        pos = None
        for i, val in enumerate(self.tooltip_values):
            pos = i
            if val > self.value:
                break
        # pylint: disable=undefined-loop-variable
        diff = self.tooltip_values[pos] - self.tooltip_values[pos - 1]
        p1 = 0 if diff < 1e-6 else (-self.value + self.tooltip_values[pos]) / diff
        return [
            (self.tooltip_labels[pos - 1].replace("<", "&lt;"), abs(p1)),
            (self.tooltip_labels[pos].replace("<", "&lt;"), abs(1 - p1)),
        ]


class Continuous2DMovableDotItem(MovableDotItem, ContinuousItemMixin):
    def __init__(self, radius, scale, offset, min_x, max_x, min_y, max_y):
        super().__init__(radius, scale, offset, min_x, max_x)
        self._min_y = min_y
        self._max_y = max_y
        self._horizontal_line = None

    @property
    def horizontal_line(self):
        return self._horizontal_line

    @horizontal_line.setter
    def horizontal_line(self, line):
        line.setVisible(False)
        self._horizontal_line = line

    def move(self, x):
        super().move(x)
        diff_ = np.nan_to_num(self._max_x - self._min_x)
        k = (x - self._min_x) / diff_ if diff_ else 0
        self.setY(
            self._min_y - self.rect().width() / 2 + (self._max_y - self._min_y) * k
        )

    def mousePressEvent(self, event):
        self._show_horizontal_line()
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._show_horizontal_line()

    def mouseReleaseEvent(self, event):
        self.horizontal_line.setVisible(False)
        return super().mouseReleaseEvent(event)

    def _show_horizontal_line(self):
        self.horizontal_line.setY(
            self.y() + self.rect().width() / 2 - abs(self._max_y - self._min_y) / 2
        )
        self.horizontal_line.setVisible(True)


class ContinuousFeature2DItem(QGraphicsWidget):
    tick_height = 6
    tick_width = 2
    DOT_RADIUS = 12
    y_diff = 80
    n_tck = 4

    def __init__(self, name, _, data_extremes, values, scale, name_offset, offset):
        super().__init__()
        data_start, data_stop = data_extremes[0], data_extremes[1]
        labels = [
            str(
                np.round(
                    data_start + (data_stop - data_start) * i / (self.n_tck - 1), 1
                )
            )
            for i in range(self.n_tck)
        ]

        # leading label
        font = name.document().defaultFont()
        name.setFont(font)
        name.setPos(name_offset, -10)
        name.setParentItem(self)

        # labels
        ascending = data_start < data_stop
        y_start, y_stop = (self.y_diff, 0) if ascending else (0, self.y_diff)
        for i in range(self.n_tck):
            text = QGraphicsSimpleTextItem(labels[i], self)
            w = text.boundingRect().width()
            y = y_start + (y_stop - y_start) / (self.n_tck - 1) * i
            text.setPos(-5 - w, y - 8)
            _ = QGraphicsLineItem(-2, y, 2, y, self)

        # prediction marker
        self.dot = Continuous2DMovableDotItem(
            self.DOT_RADIUS, scale, offset, values[0], values[-1], y_start, y_stop
        )
        self.dot.tooltip_labels = labels
        self.dot.tooltip_values = values
        self.dot.setParentItem(self)
        h_line = QGraphicsLineItem(
            values[0] * scale + offset,
            self.y_diff / 2,
            values[-1] * scale + offset,
            self.y_diff / 2,
            self,
        )
        pen = QPen(Qt.DashLine)
        pen.setBrush(QColor(Qt.red))
        h_line.setPen(pen)
        self.dot.horizontal_line = h_line

        # pylint: disable=unused-variable
        # line
        _ = QGraphicsLineItem(
            values[0] * scale + offset,
            y_start,
            values[-1] * scale + offset,
            y_stop,
            self,
        )

        # ticks
        for value in values:
            diff_ = np.nan_to_num(values[-1] - values[0])
            k = (value - values[0]) / diff_ if diff_ else 0
            y_tick = (y_stop - y_start) * k + y_start - self.tick_height / 2
            x_tick = value * scale - self.tick_width / 2 + offset
            tick = QGraphicsRectItem(
                x_tick, y_tick, self.tick_width, self.tick_height, self
            )
            tick.setBrush(QColor(Qt.black))

        # rect
        rect = QGraphicsRectItem(
            values[0] * scale + offset,
            -self.y_diff * 0.125,
            values[-1] * scale + offset,
            self.y_diff * 1.25,
            self,
        )
        pen = QPen(Qt.DotLine)
        pen.setBrush(QColor(50, 150, 200, 255))
        rect.setPen(pen)
        self.setPreferredSize(self.preferredWidth(), self.y_diff * 1.5)


class RulerItem(QGraphicsWidget):
    tick_height = 6
    tick_width = 0
    DOT_RADIUS = 12
    half_tick_height = 3
    bold_label = True
    DOT_ITEM_CLS = DotItem

    def __init__(self, name, values, scale, name_offset, offset, labels=None):
        super().__init__()

        # leading label
        font = name.document().defaultFont()
        if self.bold_label:
            font.setWeight(QFont.Bold)
        name.setFont(font)
        name.setPos(name_offset, -10)
        name.setParentItem(self)

        # prediction marker
        self.dot = self.DOT_ITEM_CLS(
            self.DOT_RADIUS, scale, offset, values[0], values[-1]
        )
        self.dot.setParentItem(self)

        # pylint: disable=unused-variable
        # line
        _ = QGraphicsLineItem(
            min(values) * scale + offset, 0, max(values) * scale + offset, 0, self
        )

        if labels is None:
            labels = [str(abs(v) if v == -0 else v) for v in values]

        old_x_tick = None
        shown_items = []
        w = QGraphicsSimpleTextItem(labels[0]).boundingRect().width()
        text_finish = values[0] * scale - w + offset - 10
        for i, (label, value) in enumerate(zip(labels, values)):
            text = QGraphicsSimpleTextItem(label)
            x_text = value * scale - text.boundingRect().width() / 2 + offset
            if text_finish > x_text - 10:
                y_text, y_tick = self.DOT_RADIUS * 0.7, 0
                text_finish = values[0] * scale + offset
            else:
                y_text = -text.boundingRect().height() - self.DOT_RADIUS * 0.7
                y_tick = -self.tick_height
                text_finish = x_text + text.boundingRect().width()
            text.setPos(x_text, y_text)
            if not collides(text, shown_items):
                text.setParentItem(self)
                shown_items.append(text)

            x_tick = value * scale - self.tick_width / 2 + offset
            tick = QGraphicsRectItem(
                x_tick, y_tick, self.tick_width, self.tick_height, self
            )
            tick.setBrush(QColor(Qt.black))

            if self.half_tick_height and i:
                x = x_tick - (x_tick - old_x_tick) / 2
                _ = QGraphicsLineItem(x, -self.half_tick_height, x, 0, self)
            old_x_tick = x_tick


class NomogramItem(QGraphicsWidget):
    def __init__(self):
        super().__init__()
        self._items = []
        self.setLayout(QGraphicsLinearLayout(Qt.Vertical))

    def add_items(self, items):
        self._items = items
        for item in items:
            self.layout().addItem(item)


"""         The code below needed changes           """


class ProbabilitiesDotItem(DotItem):
    def __init__(self, radius, scale, offset, min_x, max_x, title, get_probabilities):
        self.title = title
        self.get_probabilities = get_probabilities
        self.movable_dot_items = []
        self._invisible_sum = 0
        super().__init__(radius, scale, offset, min_x, max_x)
        self.setBrush(QColor(150, 150, 150, 255))
        self.setPen(QPen(QBrush(QColor(75, 75, 75, 255)), 2))

    def move_to_sum(self, invisible_sum: float = None):
        total = sum(item.value for item in self.movable_dot_items)

        if invisible_sum is not None:
            self._invisible_sum = invisible_sum
        total += self._invisible_sum

        self.move_to_val(total)
        self.parentItem().rescale()

    def get_tooltip_text(self):
        linear_prediction = self.get_probabilities(self.value)
        risk_score = np.exp(linear_prediction)
        text = (
            f"Total points: {np.round(self.value, 3)} <br/>"
            f"Log-partial hazard: {np.round(linear_prediction, 3)} <br/>"
            f"Partial hazard: {np.round(risk_score, 3)}"
        )

        return self.TOOLTIP_TEMPLATE.format(self.TOOLTIP_STYLE, self.title, text)


class PredictionRulerItem(QGraphicsWidget):
    tick_height = 6
    DOT_RADIUS = 14
    y_diff = 4

    def __init__(
        self,
        min_linear_predictor,
        max_linear_predictor,
        name,
        values,
        scale,
        name_offset,
        offset,
        get_points,
        title,
        get_probabilities,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.get_points = get_points
        self.min_val = min(values)
        self.max_val = max(values)

        # leading labels
        font = name.document().defaultFont()
        font.setWeight(QFont.Bold)
        name_total = QGraphicsTextItem("Total", self)
        name_total.setFont(font)
        name_total.setPos(name_offset, -25)
        name.setFont(font)
        name.setPos(name_offset, 10)
        name.setParentItem(self)

        # prediction marker
        self.dot = ProbabilitiesDotItem(
            self.DOT_RADIUS,
            scale,
            offset,
            values[0],
            values[-1],
            title,
            get_probabilities,
        )
        self.dot.setPos(0, (-self.DOT_RADIUS + self.y_diff) / 2)
        self.dot.setParentItem(self)

        # pylint: disable=unused-variable
        # two lines
        _ = QGraphicsLineItem(
            self.min_val * scale + offset, 0, self.max_val * scale + offset, 0, self
        )
        _ = QGraphicsLineItem(
            self.min_val * scale + offset,
            self.y_diff,
            self.max_val * scale + offset,
            self.y_diff,
            self,
        )

        # ticks and labels
        old_x_tick = values[0] * scale + offset
        for i, value in enumerate(values[1:]):
            x_tick = value * scale + offset
            x = x_tick - (x_tick - old_x_tick) / 2
            _ = QGraphicsLineItem(x, -self.tick_height / 2, x, 0, self)
            old_x_tick = x_tick
            if i == len(values) - 2:
                break
            text = QGraphicsTextItem(str(abs(value) if value == -0 else value), self)
            x_text = value * scale - text.boundingRect().width() / 2 + offset
            y_text = -text.boundingRect().height() - self.DOT_RADIUS * 0.7
            text.setPos(x_text, y_text)
            _ = QGraphicsLineItem(x_tick, -self.tick_height, x_tick, 0, self)

        self.prob_items = [
            (i, QGraphicsTextItem(" " + str(i) + " "), QGraphicsLineItem(0, 0, 0, 0))
            for i in np.round(
                np.linspace(min_linear_predictor, max_linear_predictor, num=15), 3
            )
        ]

    def rescale(self):
        shown_items = []
        for prob, text, tick in self.prob_items:
            pts = self.get_points(prob)
            x = pts * self.scale - text.boundingRect().width() / 2 + self.offset
            text.setPos(x, 10 + self.y_diff)
            x = pts * self.scale + self.offset
            tick.setLine(x, 0 + self.y_diff, x, self.tick_height + self.y_diff)
            text.setParentItem(None)
            tick.setParentItem(None)
            text.setVisible(False)
            tick.setVisible(False)
            if self.min_val < pts < self.max_val:
                tick.setParentItem(self)
                tick.setVisible(True)
                text.setParentItem(self)
                if not collides(text, shown_items):
                    text.setVisible(True)
                    shown_items.append(text)


class DiscreteFeatureItem(RulerItem):
    tick_height = 6
    tick_width = 2
    half_tick_height = 0
    bold_label = False
    DOT_ITEM_CLS = DiscreteMovableDotItem

    def __init__(self, name, variable, values, scale, name_offset, offset):
        labels = _get_labels(variable)
        indices = np.argsort(values)
        labels, values = labels[indices], values[indices]
        super().__init__(name, values, scale, name_offset, offset, labels)
        self.dot.tooltip_labels = labels
        self.dot.tooltip_values = values


class ContinuousFeatureItem(RulerItem):
    tick_height = 6
    tick_width = 2
    half_tick_height = 0
    bold_label = False
    DOT_ITEM_CLS = ContinuousMovableDotItem

    def __init__(
        self, name, variable, data_extremes, values, scale, name_offset, offset
    ):
        labels = _get_labels(variable, data_extremes, values)
        super().__init__(name, values, scale, name_offset, offset, labels)
        self.dot.tooltip_labels = labels
        self.dot.tooltip_values = values


class OWSurvivalNomogram(OWWidget):
    name = 'Survival Nomogram'
    description = 'Survival Nomograms'
    icon = 'icons/owsurvivalnomogram.svg'
    priority = 2000
    keywords = []

    class Inputs:
        model = Input('Survival Model', Model)
        data = Input('Data', Table)

    class Outputs:
        features = Output('Features', AttributeList)

    MAX_N_ATTRS = 1000
    POINT_SCALE = 0
    ALIGN_LEFT = 0
    ALIGN_ZERO = 1
    ACCEPTABLE = (CoxRegressionModel,)
    settingsHandler = ClassValuesContextHandler()
    # target_class_index = ContextSetting(0)
    # normalize_probabilities = Setting(False)
    scale = Setting(1)
    display_index = Setting(1)
    n_attributes = Setting(10)
    sort_index = Setting(SortBy.ABSOLUTE)
    cont_feature_dim_index = Setting(0)

    graph_name = 'scene'

    class Error(OWWidget.Error):
        invalid_model = Msg('Survival Nomogram accepts only Cox Regression Model')

    def __init__(self):
        super().__init__()
        self.instances = None
        self.domain = None
        self.data = None
        self.model = None

        self.points = []
        self.data_extremes = []

        self.feature_items = {}
        self.feature_marker_values = []
        self.scale_marker_values = None
        self.nomogram = None
        self.vertical_line = None
        self.hidden_vertical_line = None
        self.repaint = False

        box = gui.vBox(self.controlArea, "Display features")
        grid = QGridLayout()
        radio_group = gui.radioButtonsInBox(
            box, self, "display_index", [], orientation=grid, callback=self.update_scene
        )
        radio_all = gui.appendRadioButton(radio_group, "All", addToLayout=False)
        radio_best = gui.appendRadioButton(
            radio_group, "Best ranked:", addToLayout=False
        )
        spin_box = gui.hBox(None, margin=0)
        self.n_spin = gui.spin(
            spin_box,
            self,
            "n_attributes",
            1,
            self.MAX_N_ATTRS,
            label=" ",
            controlWidth=60,
            callback=self._n_spin_changed,
        )
        grid.addWidget(radio_all, 1, 1)
        grid.addWidget(radio_best, 2, 1)
        grid.addWidget(spin_box, 2, 2)

        self.sort_combo = gui.comboBox(
            box,
            self,
            "sort_index",
            label="Rank by:",
            items=SortBy.items(),
            orientation=Qt.Horizontal,
            callback=self.update_scene,
        )

        self.cont_feature_dim_combo = gui.comboBox(
            box,
            self,
            "cont_feature_dim_index",
            label="Numeric features: ",
            items=["1D projection", "2D curve"],
            orientation=Qt.Horizontal,
            callback=self.update_scene,
        )

        gui.rubber(self.controlArea)

        class _GraphicsView(QGraphicsView):
            def __init__(self, scene, parent, **kwargs):
                for k, v in dict(  # noqa: C408
                    verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
                    horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
                    viewportUpdateMode=QGraphicsView.BoundingRectViewportUpdate,
                    renderHints=(
                        QPainter.Antialiasing
                        | QPainter.TextAntialiasing
                        | QPainter.SmoothPixmapTransform
                    ),
                    alignment=(Qt.AlignTop | Qt.AlignLeft),
                    sizePolicy=QSizePolicy(
                        QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
                    ),
                ).items():
                    kwargs.setdefault(k, v)

                super().__init__(scene, parent, **kwargs)

        class GraphicsView(_GraphicsView):
            def __init__(self, scene, parent):
                super().__init__(
                    scene,
                    parent,
                    verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                    styleSheet='QGraphicsView {background: white}',
                )
                self.viewport().setMinimumWidth(
                    300
                )  # XXX: This prevents some tests failing
                self._is_resizing = False

            w = self

            def resizeEvent(self, resizeEvent):  # noqa: N803
                # Recompute main scene on window width change
                if resizeEvent.size().width() != resizeEvent.oldSize().width():
                    self._is_resizing = True
                    self.w.update_scene()
                    self._is_resizing = False
                return super().resizeEvent(resizeEvent)

            def is_resizing(self):
                return self._is_resizing

            def sizeHint(self):
                return QSize(500, 200)

        class FixedSizeGraphicsView(_GraphicsView):
            def __init__(self, scene, parent):
                super().__init__(
                    scene,
                    parent,
                    sizePolicy=QSizePolicy(
                        QSizePolicy.MinimumExpanding, QSizePolicy.Minimum
                    ),
                )

            def sizeHint(self):
                return QSize(400, 85)

        scene = self.scene = QGraphicsScene(self)

        top_view = self.top_view = FixedSizeGraphicsView(scene, self)
        mid_view = self.view = GraphicsView(scene, self)
        bottom_view = self.bottom_view = FixedSizeGraphicsView(scene, self)

        for view in (top_view, mid_view, bottom_view):
            self.mainArea.layout().addWidget(view)

        self.dot_animator = GraphicsColorAnimator(
            self,
            3000,
            [
                (0.9, DOT_COLOR),
                (0.925, DOT_COLOR.lighter(115)),
                (0.95, DOT_COLOR),
                (0.975, DOT_COLOR.lighter(115)),
                (1.0, DOT_COLOR),
            ],
        )

    def _n_spin_changed(self):
        self.display_index = 1
        self.update_scene()

    @Inputs.data
    def set_data(self, data):
        self.instances = data
        self.feature_marker_values = []
        # self.set_feature_marker_values()
        self.update_scene()

    @Inputs.model
    def set_model(self, model):
        self.closeContext()
        self.model = model
        self.Error.clear()
        if not model:
            return
        if self.model and not isinstance(self.model, self.ACCEPTABLE):
            self.Error.invalid_model()
            self.model = None
        self.domain = self.model.domain if self.model else None
        self.data = None

        self.calculate_points_range()
        self.openContext(self.domain)

        self.feature_marker_values = []
        self.update_scene()

    def calculate_points_range(self):
        self.points = []
        self.data_extremes = []

        if self.model is None or self.domain is None:
            return
        if not isinstance(self.model, CoxRegressionModel):
            return

        self.data = self.model.original_data  # .transform(self.domain)
        self.coefficients = coeffs = np.array(self.model.coefficients).reshape(-1, 1)

        min_values = nanmin(self.data.X, axis=0)
        max_values = nanmax(self.data.X, axis=0)

        for min_val, max_val, c in zip(min_values, max_values, coeffs.flat):
            self.data_extremes.append(sorted([min_val, max_val], reverse=(c < 0)))

        self.points = np.hstack(
            (coeffs * min_values.reshape(-1, 1), coeffs * max_values.reshape(-1, 1))
        )

    def update_scene(self):
        self.clear_scene()
        if self.domain is None or not len(self.points[0]):
            self.Outputs.features.send(None)
            return

        n_attrs = self.n_attributes if self.display_index else int(1e10)
        attr_inds, attributes = zip(*self.get_ordered_attributes()[:n_attrs])
        self.Outputs.features.send(AttributeList(attributes))

        point_text = QGraphicsTextItem('Points')
        metric = QFontMetrics(point_text.font())

        def text_item(text):
            elided_text = metric.elidedText(text, Qt.ElideRight, 200)
            item = QGraphicsTextItem(elided_text)
            item.setToolTip(text)
            return item

        name_items = [text_item(attr.name) for attr in attributes]

        probs_text = QGraphicsTextItem('Log-partial hazard')
        all_items = name_items + [point_text, probs_text]
        name_offset = -max(t.boundingRect().width() for t in all_items) - 30
        w = self.view.viewport().rect().width()
        max_width = w + name_offset - 30

        points = [self.points[i] for i in attr_inds]
        points = [p - p.min() for p in points]
        max_ = np.nan_to_num(max(max(abs(p)) for p in points))
        d = 100 / max_ if max_ else 1
        minimums = [p.min() for p in self.points]
        points = np.array([p * d for p in points])
        self.scale_marker_values = lambda x: (x - minimums) * d

        # print(minimums)

        point_item, nomogram_head, nomogram_body = self.create_nomogram_header_and_body(
            attributes,
            attr_inds,
            name_items,
            points,
            max_width,
            point_text,
            name_offset,
        )

        probs_item, nomogram_foot = self.create_footer_nomogram(
            probs_text,
            max_width,
            name_offset,
            points,
        )

        for item in self.feature_items.values():
            item.dot.point_dot = point_item.dot
            item.dot.probs_dot = probs_item.dot
            item.dot.vertical_line = self.hidden_vertical_line

        self.dot_animator.setGraphicsItems(
            [item.dot for item in self.feature_items.values()]
        )
        self.dot_animator.start()

        self.nomogram = nomogram = NomogramItem()
        nomogram.add_items([nomogram_head, nomogram_body, nomogram_foot])
        self.scene.addItem(nomogram)

        self.set_feature_marker_values()

        rect = QRectF(
            self.scene.itemsBoundingRect().x(),
            self.scene.itemsBoundingRect().y(),
            self.scene.itemsBoundingRect().width(),
            self.nomogram.preferredSize().height(),
        ).adjusted(10, 0, 20, 0)
        self.scene.setSceneRect(rect)

        # Clip top and bottom (60 and 150) parts from the main view
        self.view.setSceneRect(
            rect.x(), rect.y() + 80, rect.width() - 10, rect.height() - 160
        )
        self.view.viewport().setMaximumHeight(int(rect.height() - 160))
        # Clip main part from top/bottom views
        # below point values are imprecise (less/more than required) but this
        # is not a problem due to clipped scene content still being drawn
        self.top_view.setSceneRect(rect.x(), rect.y() + 3, rect.width() - 10, 20)
        self.bottom_view.setSceneRect(
            rect.x(), rect.height() - 110, rect.width() - 10, 30
        )

    def create_nomogram_header_and_body(
        self,
        attributes,
        attr_inds,
        name_items,
        points,
        max_width,
        point_text,
        name_offset,
    ):
        values = self.get_ruler_values(0, 100, max_width)
        min_p, max_p = min(values), max(values)
        diff_ = np.nan_to_num(max_p - min_p)
        scale_x = self._adjust_scale(
            attributes, points, max_width, diff_, attr_inds, self.data_extremes
        )

        nomogram_header = NomogramItem()
        point_item = RulerItem(
            point_text, values, scale_x, name_offset, -scale_x * min_p
        )
        point_item.setPreferredSize(point_item.preferredWidth(), 35)
        nomogram_header.add_items([point_item])

        nomogram_body = NomogramItem()
        cont_feature_item_class = (
            ContinuousFeature2DItem
            if self.cont_feature_dim_index
            else ContinuousFeatureItem
        )

        feature_items = [
            (
                DiscreteFeatureItem(
                    name_item, attr, point, scale_x, name_offset, -scale_x * min_p
                )
                if attr.is_discrete
                else cont_feature_item_class(
                    name_item,
                    attr,
                    self.data_extremes[i],
                    self.get_ruler_values(
                        point.min(), point.max(), scale_x * point.ptp(), False
                    ),
                    scale_x,
                    name_offset,
                    -scale_x * min_p,
                )
            )
            for i, attr, name_item, point in zip(
                attr_inds, attributes, name_items, points
            )
        ]

        nomogram_body.add_items(feature_items)
        self.feature_items = OrderedDict(sorted(zip(attr_inds, feature_items)))

        x = -scale_x * min_p
        y = nomogram_body.layout().preferredHeight() + 10
        self.vertical_line = QGraphicsLineItem(x, -6, x, y)
        self.vertical_line.setPen(QPen(Qt.DotLine))
        self.vertical_line.setParentItem(point_item)
        self.hidden_vertical_line = QGraphicsLineItem(x, -6, x, y)
        pen = QPen(Qt.DashLine)
        pen.setBrush(QColor(Qt.red))
        self.hidden_vertical_line.setPen(pen)
        self.hidden_vertical_line.setParentItem(point_item)

        return point_item, nomogram_header, nomogram_body

    def create_footer_nomogram(
        self,
        probs_text,
        max_width,
        name_offset,
        points,
    ):
        values = self.get_ruler_values(0, points.sum(), max_width)
        min_sum, max_sum = min(values), max(values)
        diff_ = np.nan_to_num(max_sum - min_sum)
        scale_x = max_width / diff_ if diff_ else max_width

        # min/max linear predictors
        min_lp = sum([p.min() for p in self.points])
        max_lp = sum([p.max() for p in self.points])

        # min/max points range
        min_points, max_points = 0, points.sum()

        def map_between_intervals(x: int, from_interval: tuple, to_interval: tuple):
            """Map x from the interval [a, b] onto the interval [c, d] using:
            p = (lp - a) * ((d - c) / (b - a)) + c
            """
            a, b = from_interval
            c, d = to_interval
            return (x - a) * ((d - c) / (b - a)) + c

        def map_points_to_linear_predictor(p):
            return map_between_intervals(p, (min_points, max_points), (min_lp, max_lp))

        def map_linear_predictor_to_points(lp):
            return map_between_intervals(lp, (min_lp, max_lp), (min_points, max_points))

        probs_item = PredictionRulerItem(
            min_lp,
            max_lp,
            probs_text,
            values,
            scale_x,
            name_offset,
            -scale_x * min_sum,
            get_points=map_linear_predictor_to_points,
            get_probabilities=map_points_to_linear_predictor,
            title='Summary',
        )

        nomogram_footer = NomogramItem()
        nomogram_footer.add_items([probs_item])
        return probs_item, nomogram_footer

    def get_ordered_attributes(self):
        """Return (in_domain_index, attr) pairs, ordered by method in SortBy combo"""
        if self.domain is None or not self.domain.attributes:
            return []

        attrs = self.domain.attributes
        sort_by = self.sort_index
        if sort_by == SortBy.NO_SORTING:
            return list(enumerate(attrs))

        elif sort_by == SortBy.NAME:

            def key(x):
                _, attr = x
                return attr.name.lower()

        elif sort_by == SortBy.ABSOLUTE:

            def key(x):
                i, _ = x
                return -np.abs(self.coefficients[i])

        elif sort_by == SortBy.POSITIVE:

            def key(x):
                i, _ = x
                return self.coefficients[i]

        elif sort_by == SortBy.NEGATIVE:

            def key(x):
                i, _ = x
                return -self.coefficients[i]

        return sorted(enumerate(attrs), key=key)

    def set_feature_marker_values(self):
        if not (len(self.points) and len(self.feature_items)):
            return

        if not len(self.feature_marker_values):
            self._init_feature_marker_values()

        marker_values = self.scale_marker_values(self.feature_marker_values)

        invisible_sum = 0
        for i, marker in enumerate(marker_values):
            try:
                item = self.feature_items[i]
            except KeyError:
                invisible_sum += marker
            else:
                item.dot.move_to_val(marker)

        item.dot.probs_dot.move_to_sum(invisible_sum)

    def _init_feature_marker_values(self):
        self.feature_marker_values = []

        instances = self.instances.transform(self.domain) if self.instances else None
        values = []
        for i, attr in enumerate(self.domain.attributes):
            value, feature_val = 0, None
            if len(self.points):
                if attr.is_discrete:
                    ind, n = unique(self.data.X[:, i], return_counts=True)
                    feature_val = np.nan_to_num(ind[np.argmax(n)])
                else:
                    feature_val = nanmean(self.data.X[:, i])

            # If data is provided on a separate signal, use the first data
            # instance to position the points instead of the mean
            inst_in_dom = instances and attr in instances.domain
            if inst_in_dom and not np.isnan(instances[0][attr]):
                feature_val = instances[0][attr]

            if feature_val is not None:
                value = (
                    self.points[i][int(feature_val)]
                    if attr.is_discrete
                    else self.coefficients[i][0] * feature_val
                )
            values.append(value)
        self.feature_marker_values = np.asarray(values)

    def clear_scene(self):
        self.feature_items = {}
        self.scale_marker_values = None
        self.nomogram = None
        self.vertical_line = None
        self.hidden_vertical_line = None
        self.dot_animator.clear()
        self.scene.clear()

    def send_report(self):
        self.report_plot()

    def reset_settings(self):
        self._reset_settings()
        self.update_scene()

    @staticmethod
    def _adjust_scale(
        attributes, points, max_width, diff, attr_inds, log_reg_cont_data_extremes
    ):
        if not diff:
            return max_width

        def offset(name, point):
            text_ = QGraphicsTextItem(name).boundingRect()
            return scale * point + text_.width() / 2

        lr = log_reg_cont_data_extremes
        scale = max_width / diff
        names = list(
            chain.from_iterable(
                [
                    _get_labels(
                        a,
                        lr and lr[i],
                        OWSurvivalNomogram.get_ruler_values(
                            p.min(), p.max(), scale * p.ptp(), False
                        ),
                    )
                    for i, a, p in zip(attr_inds, attributes, points)
                ]
            )
        )
        points = list(chain.from_iterable(points))

        old_scale = scale + 1
        while old_scale > scale:
            old_scale = scale
            offsets = [offset(n, p) for n, p in zip(names, points)]
            most_right_name = names[np.argmax(offsets)]
            text = QGraphicsTextItem(most_right_name).boundingRect()
            scale = (max_width - text.width() / 2) / diff
        return scale

    @staticmethod
    def get_ruler_values(start, stop, max_width, round_to_nearest=True):
        if max_width == 0:
            return [0]
        diff = np.nan_to_num((stop - start) / max_width)
        if diff <= 0:
            return [0]
        decimals = int(np.floor(np.log10(diff)))
        if diff > 4 * pow(10, decimals):
            step = 5 * pow(10, decimals + 2)
        elif diff > 2 * pow(10, decimals):
            step = 2 * pow(10, decimals + 2)
        elif diff > 1 * pow(10, decimals):
            step = 1 * pow(10, decimals + 2)
        else:
            step = 5 * pow(10, decimals + 1)
        round_by = int(-np.floor(np.log10(step)))
        r = start % step
        if not round_to_nearest:
            _range = np.arange(start + step, stop + r, step) - r
            start, stop = np.floor(start * 100) / 100, np.ceil(stop * 100) / 100
            return np.round(np.hstack((start, _range, stop)), 2)
        return np.round(np.arange(start, stop + r + step, step) - r, round_by)


if __name__ == "__main__":  # pragma: no cover
    from orangecontrib.survival_analysis.modeling.cox import CoxRegressionLearner

    data = Table("https://datasets.biolab.si/core/melanoma.tab")
    # data = Table('https://datasets.biolab.si/core/gbsg2.tab')
    # data = Table('https://datasets.biolab.si/core/veteran.tab')

    clf = CoxRegressionLearner()(data)
    WidgetPreview(OWSurvivalNomogram).run(set_model=clf, set_data=data)
