"""Tests for experiment variable classification used by the FRF experiment GUI."""

from __future__ import annotations

import unittest

from src import experiment_config


def _cfg(
    *,
    catalog: list[dict],
    mapping: list[dict],
    expressions: list[dict] | None = None,
) -> dict:
    return {
        "sensor_catalog": catalog,
        "mapping": mapping,
        "expressions": expressions or [],
    }


class TestExperimentConfigVariableKinds(unittest.TestCase):
    def test_mapping_units_classify_force_and_acceleration(self) -> None:
        cfg = _cfg(
            catalog=[
                {"id": "force_sens", "unit": "N"},
                {"id": "accel_sens", "unit": "m/s²"},
            ],
            mapping=[
                {"name": "Fx", "channel": 0, "sensor_id": "force_sens"},
                {"name": "Ax", "channel": 1, "sensor_id": "accel_sens"},
            ],
        )
        kinds = experiment_config.build_variable_kinds(cfg)
        self.assertEqual(kinds["Fx"], "force")
        self.assertEqual(kinds["Ax"], "acceleration")

    def test_expression_force_when_all_references_are_newtons(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "f1", "unit": "N"}, {"id": "f2", "unit": "N"}],
            mapping=[
                {"name": "F1", "channel": 0, "sensor_id": "f1"},
                {"name": "F2", "channel": 1, "sensor_id": "f2"},
            ],
            expressions=[{"name": "F_sum", "expr": "F1 + F2"}],
        )
        kinds = experiment_config.build_variable_kinds(cfg)
        self.assertEqual(kinds["F_sum"], "force")

    def test_expression_acceleration_when_all_references_are_accel(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "a1", "unit": "m/s²"}, {"id": "a2", "unit": "m/s²"}],
            mapping=[
                {"name": "A1", "channel": 0, "sensor_id": "a1"},
                {"name": "A2", "channel": 1, "sensor_id": "a2"},
            ],
            expressions=[{"name": "A_sum", "expr": "A1 + A2"}],
        )
        kinds = experiment_config.build_variable_kinds(cfg)
        self.assertEqual(kinds["A_sum"], "acceleration")

    def test_expression_mixed_units_is_ineligible(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "f1", "unit": "N"}, {"id": "a1", "unit": "m/s²"}],
            mapping=[
                {"name": "F1", "channel": 0, "sensor_id": "f1"},
                {"name": "A1", "channel": 1, "sensor_id": "a1"},
            ],
            expressions=[{"name": "mixed", "expr": "F1 + A1"}],
        )
        kinds = experiment_config.build_variable_kinds(cfg)
        self.assertEqual(kinds["mixed"], "none")

    def test_expression_invalid_ast_is_ineligible(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "f1", "unit": "N"}],
            mapping=[{"name": "F1", "channel": 0, "sensor_id": "f1"}],
            expressions=[{"name": "bad", "expr": "F1 +"}],
        )
        kinds = experiment_config.build_variable_kinds(cfg)
        self.assertEqual(kinds["bad"], "none")

    def test_duplicate_mapping_name_raises(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "s1", "unit": "N"}],
            mapping=[
                {"name": "dup", "channel": 0, "sensor_id": "s1"},
                {"name": "dup", "channel": 1, "sensor_id": "s1"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "Duplicate mapping name: 'dup'"):
            experiment_config.build_variable_kinds(cfg)

    def test_unknown_sensor_id_raises(self) -> None:
        cfg = _cfg(
            catalog=[{"id": "s1", "unit": "N"}],
            mapping=[{"name": "Fx", "channel": 0, "sensor_id": "missing"}],
        )
        with self.assertRaisesRegex(ValueError, "Unknown sensor_id 'missing'"):
            experiment_config.build_variable_kinds(cfg)


if __name__ == "__main__":
    unittest.main()
