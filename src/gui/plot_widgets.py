from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
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
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    @property
    def figure(self) -> Figure:
        return self._figure

    def replace_figure(self, fig: Figure) -> None:
        # Replace the underlying figure in-place by swapping the canvas.
        layout = self.layout()
        if layout is None:
            return
        self._canvas.setParent(None)
        self._figure = fig
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas)
        self._canvas.draw_idle()

    def redraw(self) -> None:
        self._canvas.draw_idle()

