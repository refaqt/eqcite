from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QWidget, QVBoxLayout


@dataclass
class FigureSpec:
    title: str
    figure: Figure


class MatplotlibFigureWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._scroll_cid: Optional[int] = None

        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setToolTip("Scroll to zoom. Use the toolbar to pan/zoom; Home resets.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._wire_interactions()

    @property
    def figure(self) -> Figure:
        return self._figure

    def replace_figure(self, fig: Figure) -> None:
        # Replace the underlying figure in-place by swapping the canvas (+ toolbar bindings).
        layout = self.layout()
        if layout is None:
            return
        if self._scroll_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._scroll_cid)
            except Exception:
                pass
            self._scroll_cid = None

        self._toolbar.setParent(None)
        self._canvas.setParent(None)
        self._figure = fig
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setToolTip("Scroll to zoom. Use the toolbar to pan/zoom; Home resets.")

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)
        self._wire_interactions()
        self._canvas.draw_idle()

    def redraw(self) -> None:
        self._canvas.draw_idle()

    def _wire_interactions(self) -> None:
        if self._scroll_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._scroll_cid)
            except Exception:
                pass
            self._scroll_cid = None
        self._scroll_cid = self._canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, event) -> None:
        ax = getattr(event, "inaxes", None)
        if ax is None:
            return

        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None and ydata is None:
            return

        step = getattr(event, "step", 0) or 0
        if step == 0:
            button = getattr(event, "button", None)
            if button == "up":
                step = 1
            elif button == "down":
                step = -1
            else:
                return

        base = 1.2
        scale = (1.0 / base) if step > 0 else base

        try:
            if xdata is not None:
                self._zoom_axis(ax, which="x", center=float(xdata), scale=scale)
            if ydata is not None:
                self._zoom_axis(ax, which="y", center=float(ydata), scale=scale)
        except Exception:
            return

        self._canvas.draw_idle()

    @staticmethod
    def _zoom_axis(ax, which: str, center: float, scale: float) -> None:
        if which == "x":
            left, right = ax.get_xlim()
            mode = ax.get_xscale()
            setter = ax.set_xlim
        else:
            left, right = ax.get_ylim()
            mode = ax.get_yscale()
            setter = ax.set_ylim

        if not (math.isfinite(left) and math.isfinite(right) and math.isfinite(center)):
            return
        if left == right:
            return

        if mode == "log":
            if left <= 0 or right <= 0 or center <= 0:
                return
            l = math.log10(left)
            r = math.log10(right)
            c = math.log10(center)
            nl = c + (l - c) * scale
            nr = c + (r - c) * scale
            new_left = 10**nl
            new_right = 10**nr
        else:
            new_left = center + (left - center) * scale
            new_right = center + (right - center) * scale

        if not (math.isfinite(new_left) and math.isfinite(new_right)):
            return
        if new_left == new_right:
            return

        if new_left > new_right:
            new_left, new_right = new_right, new_left
        setter(new_left, new_right)

