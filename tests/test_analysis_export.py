import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiment_config import experiment_cfg_for_snapshot, uses_new_experiment_schema
from src import frf_estimation
from src.gui.analysis_export import (
    _csd_ur_from_frequency_table,
    _read_frequency_csv,
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
        self.psd_u = np.linspace(0.1, 0.4, 4)
        self.psd_r = np.linspace(0.2, 0.5, 4)
        self.csd_ur = np.ones(4, dtype=complex) * 0.1
        self.fft_f_hz = np.linspace(0.0, 2.0, 3)
        self.fft_u_mag = {"raw": np.array([1.0, 2.0, 3.0])}
        self.fft_resp_mag = {"raw": np.array([0.5, 1.0, 1.5])}
        self.fft_u_complex = {"raw": np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])}
        self.fft_resp_complex = {"raw": np.array([0.5 + 0j, 1.0 + 0j, 1.5 + 0j])}


class _Measurement:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.proc = _Proc()
        f = self.proc.f_hz
        self.compliance = (f, np.ones(4, dtype=complex))
        self.accelerance = (f, np.ones(4, dtype=complex) * 2)
        self.raw_channels = {0: np.linspace(0.0, 1.0, 5), 1: np.linspace(1.0, 0.0, 5)}


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
                header[:6],
                [
                    "frequency_hz",
                    "coherence",
                    "force_psd",
                    "response_psd",
                    "csd_ur_real",
                    "csd_ur_imag",
                ],
            )
            self.assertIn("raw_csv", manifest["x"][0])
            self.assertIn("fft_csv", manifest["x"][0])
            self.assertTrue((folder / manifest["x"][0]["raw_csv"]).exists())
            self.assertTrue((folder / manifest["x"][0]["fft_csv"]).exists())
            np.testing.assert_allclose(entry["psd_u"], np.linspace(0.1, 0.4, 4))
            self.assertIn(0, entry["raw_channels"])
            self.assertIn("raw", entry["fft_u_mag"])

    def test_frf_export_import_preserves_csd_ur(self) -> None:
        meas = _Measurement("meas_x.csv")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = export_frf_results(
                folder,
                x_measurements=[meas],
                y_measurements=[],
                meta={"analysis_type": "frf"},
            )
            bundle = import_frf_results(folder, manifest)
            np.testing.assert_allclose(bundle["x"][0]["csd_ur"], meas.proc.csd_ur)

    def test_average_coherence_from_hydrated_csd(self) -> None:
        m1 = _Measurement("a.csv")
        m2 = _Measurement("b.csv")
        m2.proc.csd_ur = np.array([0.2 + 0.1j, 0.25 + 0.1j, 0.3 + 0.1j, 0.35 + 0.1j])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = export_frf_results(
                folder,
                x_measurements=[m1, m2],
                y_measurements=[],
                meta={"analysis_type": "frf"},
            )
            bundle = import_frf_results(folder, manifest)
            entries = bundle["x"]
            Puu = np.mean(np.stack([e["psd_u"] for e in entries], axis=0), axis=0)
            Prr = np.mean(np.stack([e["psd_r"] for e in entries], axis=0), axis=0)
            Pur = np.mean(np.stack([e["csd_ur"] for e in entries], axis=0), axis=0)
            coh = (np.abs(Pur) ** 2) / ((Puu * Prr) + 1e-30)
            coh = np.clip(coh, 0.0, 1.0)
            self.assertTrue(np.all(coh[1:] > 0))

    def test_average_frf_from_spectra_h1_not_complex_mean(self) -> None:
        f_hz = np.array([1.0, 2.0, 3.0, 4.0])
        p1 = _Proc()
        p1.f_hz = f_hz
        p1.psd_u = np.array([1.0, 1.0, 1.0, 1.0])
        p1.psd_r = np.ones(4)
        p1.csd_ur = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
        p2 = _Proc()
        p2.f_hz = f_hz
        p2.psd_u = np.array([2.0, 2.0, 2.0, 2.0])
        p2.psd_r = np.ones(4)
        p2.csd_ur = np.array([0.0 + 1.0j, 0.0 + 1.0j, 0.0 + 1.0j, 0.0 + 1.0j])
        _, H_accel, _ = frf_estimation.average_frf_from_processing([p1, p2], estimator_type="H1")
        H1_tap = np.stack([p1.csd_ur / p1.psd_u, p2.csd_ur / p2.psd_u], axis=0)
        H_complex_mean = np.mean(H1_tap, axis=0)
        Puy = np.mean(np.stack([p1.csd_ur, p2.csd_ur], axis=0), axis=0)
        Puu = np.mean(np.stack([p1.psd_u, p2.psd_u], axis=0), axis=0)
        H_spectral = Puy / Puu
        np.testing.assert_allclose(H_accel, H_spectral)
        self.assertFalse(np.allclose(H_accel, H_complex_mean))

    def test_export_average_frf_uses_spectral_h1(self) -> None:
        m1 = _Measurement("a.csv")
        m2 = _Measurement("b.csv")
        m1.proc.psd_u = np.array([1.0, 1.0, 1.0, 1.0])
        m1.proc.csd_ur = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
        m2.proc.psd_u = np.array([2.0, 2.0, 2.0, 2.0])
        m2.proc.csd_ur = np.array([0.0 + 1.0j, 0.0 + 1.0j, 0.0 + 1.0j, 0.0 + 1.0j])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = export_frf_results(
                folder,
                x_measurements=[m1, m2],
                y_measurements=[],
                meta={"analysis_type": "frf", "estimator_type": "H1"},
            )
            bundle = import_frf_results(folder, manifest)
            avg_entry = next(e for e in bundle["averages"] if e.get("direction") == "x")
            _, Ha = avg_entry["accelerance"]
            _, H_expected, _ = frf_estimation.average_frf_from_processing(
                [m1.proc, m2.proc], estimator_type="H1"
            )
            np.testing.assert_allclose(Ha, H_expected)

    def test_csd_backfill_from_legacy_frequency_csv(self) -> None:
        f_hz = np.linspace(1.0, 4.0, 4)
        coh = np.linspace(0.2, 0.8, 4)
        psd_u = np.linspace(0.1, 0.4, 4)
        psd_r = np.linspace(0.2, 0.5, 4)
        Hc = np.ones(4, dtype=complex)
        Ha = np.ones(4, dtype=complex) * 2
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            freq_path = folder / "legacy_freq.csv"
            header = [
                "frequency_hz",
                "coherence",
                "force_psd",
                "response_psd",
                "bode_displacement_real",
                "bode_displacement_imag",
                "bode_displacement_magnitude",
                "bode_displacement_phase_deg",
                "bode_acceleration_real",
                "bode_acceleration_imag",
                "bode_acceleration_magnitude",
                "bode_acceleration_phase_deg",
            ]
            with freq_path.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(header)
                for i in range(4):
                    w.writerow(
                        [
                            float(f_hz[i]),
                            float(coh[i]),
                            float(psd_u[i]),
                            float(psd_r[i]),
                            float(np.real(Hc[i])),
                            float(np.imag(Hc[i])),
                            float(np.abs(Hc[i])),
                            0.0,
                            float(np.real(Ha[i])),
                            float(np.imag(Ha[i])),
                            float(np.abs(Ha[i])),
                            0.0,
                        ]
                    )
            freq_tbl = _read_frequency_csv(freq_path)
            Pur = _csd_ur_from_frequency_table(freq_tbl)
            self.assertTrue(np.all(np.abs(Pur) > 0))
            expected = np.sqrt(np.clip(coh, 0.0, 1.0) * psd_u * psd_r + 1e-30)
            np.testing.assert_allclose(np.abs(Pur), expected)

    def test_experiment_cfg_for_snapshot_strips_legacy(self) -> None:
        cfg = {
            "analysis_type": "frf",
            "data_source": "file",
            "sensor_catalog": [{"id": "s0", "unit": "N", "sensitivity_vs_per_v": 1.0}],
            "mapping": [],
            "channel_mapping": {"hammer_x": 6},
            "frf_pairs_x": [{"input": "F_x", "output": "x_tool"}],
            "wiener_mysize": 1000,
            "//": "comment",
        }
        snap = experiment_cfg_for_snapshot(cfg)
        self.assertTrue(uses_new_experiment_schema(snap))
        self.assertNotIn("channel_mapping", snap)
        self.assertNotIn("frf_pairs_x", snap)
        self.assertNotIn("wiener_mysize", snap)
        self.assertNotIn("//", snap)


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
