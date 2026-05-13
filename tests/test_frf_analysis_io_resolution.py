"""Tests for FRF analysis_io resolution and force/response binding."""

from __future__ import annotations

import unittest

import numpy as np

from src import experiment_config


def _cfg(
    *,
    catalog: list[dict],
    mapping: list[dict],
    analysis_io: dict,
    expressions: list[dict] | None = None,
) -> dict:
    return {
        "sensor_catalog": catalog,
        "mapping": mapping,
        "expressions": expressions or [],
        "analysis_io": analysis_io,
    }


class TestResolveAnalysisIoNames(unittest.TestCase):
    def test_resolves_input_and_output_for_direction(self) -> None:
        cfg = _cfg(
            catalog=[
                {"id": "f1", "unit": "N", "sensitivity_vs_per_v": 1.0},
                {"id": "a1", "unit": "m/s²", "sensitivity_vs_per_v": 1.0},
            ],
            mapping=[
                {"name": "Fx", "channel": 0, "sensor_id": "f1"},
                {"name": "Ax", "channel": 1, "sensor_id": "a1"},
            ],
            analysis_io={"x": {"input": "Fx", "output": "Ax"}},
        )
        self.assertEqual(experiment_config.resolve_analysis_io_names(cfg, "x"), ("Fx", "Ax"))

    def test_empty_sensor_catalog_raises(self) -> None:
        cfg = {
            "sensor_catalog": [],
            "mapping": [],
            "analysis_io": {"x": {"input": "Fx", "output": "Ax"}},
        }
        with self.assertRaisesRegex(ValueError, "sensor_catalog"):
            experiment_config.resolve_analysis_io_names(cfg, "x")

    def test_missing_analysis_io_block_raises(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "f1", "unit": "N", "sensitivity_vs_per_v": 1.0}],
            mapping=[{"name": "Fx", "channel": 0, "sensor_id": "f1"}],
            analysis_io={},
        )
        with self.assertRaisesRegex(ValueError, "Input/Output"):
            experiment_config.resolve_analysis_io_names(cfg, "x")

    def test_wrong_input_kind_raises(self) -> None:
        cfg = _cfg(
            catalog=[
                {"id": "f1", "unit": "N", "sensitivity_vs_per_v": 1.0},
                {"id": "a1", "unit": "m/s²", "sensitivity_vs_per_v": 1.0},
            ],
            mapping=[
                {"name": "Fx", "channel": 0, "sensor_id": "f1"},
                {"name": "Ax", "channel": 1, "sensor_id": "a1"},
            ],
            analysis_io={"x": {"input": "Ax", "output": "Fx"}},
        )
        with self.assertRaisesRegex(ValueError, "force-classified"):
            experiment_config.resolve_analysis_io_names(cfg, "x")


class TestBindDirectionalIoSignals(unittest.TestCase):
    def test_binds_input_to_force_and_output_to_response(self) -> None:
        force = np.array([1.0, 0.0, 0.0])
        accel = np.array([0.0, 2.0, 0.0])
        data = {"Fx": force, "Ax": accel}
        u, y = experiment_config.bind_directional_io_signals(data, "Fx", "Ax")
        np.testing.assert_array_equal(u, force)
        np.testing.assert_array_equal(y, accel)

    def test_swapped_names_bind_opposite_arrays(self) -> None:
        force = np.array([1.0, 0.0, 0.0])
        accel = np.array([0.0, 2.0, 0.0])
        data = {"Fx": force, "Ax": accel}
        u, y = experiment_config.bind_directional_io_signals(data, "Ax", "Fx")
        np.testing.assert_array_equal(u, accel)
        np.testing.assert_array_equal(y, force)

    def test_missing_signal_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required input signal"):
            experiment_config.bind_directional_io_signals({}, "Fx", "Ax")


if __name__ == "__main__":
    unittest.main()
