from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)


class SettingsRootPage(QWidget):
    default_save_dir_changed = Signal(str)
    default_load_dir_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Settings")
        title.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        save_row = QWidget()
        save_row_l = QHBoxLayout(save_row)
        save_row_l.setContentsMargins(0, 0, 0, 0)
        save_row_l.addWidget(QLabel("Default save folder"))
        self._save_path = QLineEdit()
        self._save_path.setPlaceholderText("e.g. analyses")
        save_row_l.addWidget(self._save_path, 1)
        save_browse = QPushButton("Browse…")
        save_browse.clicked.connect(self._browse_save)
        save_row_l.addWidget(save_browse)
        self._save_path.editingFinished.connect(self._emit_save_changed)

        load_row = QWidget()
        load_row_l = QHBoxLayout(load_row)
        load_row_l.setContentsMargins(0, 0, 0, 0)
        load_row_l.addWidget(QLabel("Default load folder"))
        self._load_path = QLineEdit()
        self._load_path.setPlaceholderText("e.g. data")
        load_row_l.addWidget(self._load_path, 1)
        load_browse = QPushButton("Browse…")
        load_browse.clicked.connect(self._browse_load)
        load_row_l.addWidget(load_browse)
        self._load_path.editingFinished.connect(self._emit_load_changed)

        layout.addWidget(title)
        layout.addWidget(save_row)
        layout.addWidget(load_row)
        layout.addStretch(1)

    def set_default_save_dir(self, path: str) -> None:
        self._save_path.setText(path)

    def set_default_load_dir(self, path: str) -> None:
        self._load_path.setText(path)

    def _emit_save_changed(self) -> None:
        self.default_save_dir_changed.emit(self._save_path.text().strip())

    def _emit_load_changed(self) -> None:
        self.default_load_dir_changed.emit(self._load_path.text().strip())

    def _browse_save(self) -> None:
        p = QFileDialog.getExistingDirectory(
            self, "Choose default save folder", self._save_path.text() or ""
        )
        if not p:
            return
        self._save_path.setText(p)
        self._emit_save_changed()

    def _browse_load(self) -> None:
        p = QFileDialog.getExistingDirectory(
            self, "Choose default load folder", self._load_path.text() or ""
        )
        if not p:
            return
        self._load_path.setText(p)
        self._emit_load_changed()

