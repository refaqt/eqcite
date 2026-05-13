import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gui.analysis_export import (
    export_frf_results,
    import_frf_results,
    measurement_keys_for_paths,
    slugify_measurement_key,
)
from src.gui.state import ANALYSIS_FORMAT_VERSION, AnalysisState


class _Step:
    def __init__(self, name: str, y: np.ndarray) -> None:
        self.name = name
        self.y = y


class _Proc:
    def __init__(self) -> None:
        self.time = np.linspace(0.0, 1.0, 5)
        self.u_steps = [_Step("raw", np.ones(5)), _Step("bias_corrected", np.ones(5) * 0.9)]
        self.resp_steps = [_Step("raw", np.ones(5) * 2.0), _Step("bias_corrected", np.ones(5) * 1.8)]
        self.coherence_uy = np.linspace(0.0, 1.0, 4)
        self.f_hz = np.linspace(1.0, 4.0, 4)


class _Measurement:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.proc = _Proc()
        f = self.proc.f_hz
        self.compliance = (f, np.ones(4, dtype=complex))
        self.accelerance = (f, np.ones(4, dtype=complex) * 2)


class AnalysisExportTests(unittest.TestCase):
    def test_measurement_keys_are_unique(self) -> None:
        keys = measurement_keys_for_paths(["a.csv", "a.csv", "b.csv"])
        self.assertEqual(len(keys), 3)
        self.assertEqual(len(set(keys)), 3)

    def test_slugify_measurement_key_avoids_collisions(self) -> None:
        used: set[str] = set()
        self.assertEqual(slugify_measurement_key("run-1", used), "run-1")
        self.assertEqual(slugify_measurement_key("run-1", used), "run-1_2")

    def test_frf_export_import_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = export_frf_results(
                folder,
                x_measurements=[_Measurement("meas_x.csv")],
                y_measurements=[],
                meta={"analysis_type": "frf"},
            )
            bundle = import_frf_results(folder, manifest)
            self.assertEqual(len(bundle["x"]), 1)
            entry = bundle["x"][0]
            self.assertEqual(entry["compliance"][0].shape, entry["compliance"][1].shape)
            freq_csv = folder / manifest["x"][0]["frequency_csv"]
            with freq_csv.open("r", encoding="utf-8") as fp:
                header = fp.readline().strip().split(",")
            self.assertEqual(
                header,
                [
                    "frequency_hz",
                    "coherence",
                    "bode_displacement_real",
                    "bode_displacement_imag",
                    "bode_displacement_magnitude",
                    "bode_displacement_phase_deg",
                    "bode_acceleration_real",
                    "bode_acceleration_imag",
                    "bode_acceleration_magnitude",
                    "bode_acceleration_phase_deg",
                ],
            )


class AnalysisStateTests(unittest.TestCase):
    def test_state_round_trip(self) -> None:
        st = AnalysisState(
            analysis_type="frf",
            title="Example",
            experiment_cfg={"analysis_type": "frf", "data_source": "file"},
            gui_state={"sidebar_index": 2},
            results_manifest={"x": []},
        )
        payload = st.to_dict()
        self.assertEqual(payload["format_version"], ANALYSIS_FORMAT_VERSION)
        restored = AnalysisState.from_dict(payload)
        self.assertEqual(restored.title, "Example")
        self.assertEqual(restored.gui_state["sidebar_index"], 2)

    def test_legacy_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnalysisState.from_dict({"analysis_type": "frf"})


if __name__ == "__main__":
    unittest.main()
