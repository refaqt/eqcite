"""Tests for default FRF experiment configuration persistence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.gui.pages.frf_experiment_config import FrfSensorConfigPage
from src.gui.pages.frf_model import FrfInputModel
from src.gui.persistence import AppSettings


def _sample_default_experiment_config() -> dict:
    return {
        "sensor_catalog": [
            {"id": "force_sens", "unit": "N", "sensitivity_vs_per_v": 1.0},
            {"id": "accel_sens", "unit": "m/s²", "sensitivity_vs_per_v": 1.0},
        ],
        "mapping": [
            {"name": "Fx", "channel": 0, "sensor_id": "force_sens"},
            {"name": "Ax", "channel": 1, "sensor_id": "accel_sens"},
        ],
        "expressions": [{"name": "F_sum", "expr": "Fx + Fx"}],
        "analysis_io": {
            "x": {"input": "F_sum", "output": "Ax"},
            "y": {"input": "Fx", "output": "Ax"},
        },
        "simulation_signal_map": {"Fx": "F_x", "Ax": "x_tool"},
    }


class TestDefaultExperimentConfigPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_app_settings_round_trip_expressions_and_io(self) -> None:
        defaults = _sample_default_experiment_config()
        settings = AppSettings(default_experiment_config=dict(defaults))
        restored = AppSettings.from_dict(settings.to_dict())
        self.assertEqual(restored.default_experiment_config.get("expressions"), defaults["expressions"])
        self.assertEqual(restored.default_experiment_config.get("analysis_io"), defaults["analysis_io"])

    def test_apply_saved_defaults_restores_expressions_and_io(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        template_path = repo_root / "config" / "config_before_cabinet_assembly.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        model = FrfInputModel.from_experiment_cfg(template)
        page = FrfSensorConfigPage(model)

        saved = _sample_default_experiment_config()
        page.apply_experiment_config(saved)

        self.assertEqual(list(model.expressions or []), saved["expressions"])
        self.assertEqual(dict(model.analysis_io or {}), saved["analysis_io"])

    def test_save_default_emits_expressions_and_io(self) -> None:
        model = FrfInputModel.from_experiment_cfg(_sample_default_experiment_config())
        page = FrfSensorConfigPage(model)
        captured: list[dict] = []
        page.default_save_requested.connect(captured.append)
        page._emit_save_default()
        self.assertEqual(len(captured), 1)
        snap = captured[0]
        self.assertEqual(snap.get("expressions"), model.expressions)
        self.assertEqual(snap.get("analysis_io"), model.analysis_io)


if __name__ == "__main__":
    unittest.main()
