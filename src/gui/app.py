from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QLineEdit {
            border: 1px solid #c8c8c8;
            border-radius: 4px;
            padding: 4px;
        }
        QLineEdit:focus {
            border: 1px solid #6aa6ff;
        }
        QLineEdit:disabled {
            border: 1px solid #dddddd;
            color: #777777;
        }
        """
    )
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

