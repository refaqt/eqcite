import unittest

import numpy as np

from src import frf_estimation, signal_processing


class TestRaisedCosineWindowParams(unittest.TestCase):
    def test_legacy_defaults_convert_to_shape_params(self) -> None:
        out = signal_processing.normalize_raised_cosine_window_params(
            {
                "window_pre_seconds": 0.05,
                "window_post_seconds": 0.1,
                "window_beta": 0.25,
            }
        )
        self.assertAlmostEqual(out["window_rolloff_seconds"], 0.01875)
        self.assertAlmostEqual(out["window_max_start_seconds"], -0.03125)
        self.assertAlmostEqual(out["window_max_end_seconds"], 0.08125)

    def test_invalid_plateau_order_raises(self) -> None:
        time = np.linspace(0.0, 1.0, 1001)
        force = np.zeros_like(time)
        force[500] = 1.0
        params = {
            "window_rolloff_seconds": 0.02,
            "window_max_start_seconds": 0.1,
            "window_max_end_seconds": 0.05,
        }
        with self.assertRaisesRegex(ValueError, "window_max_end_seconds"):
            signal_processing.create_raised_cosine_window(time, force, params, time[1] - time[0])

    def test_non_positive_rolloff_raises(self) -> None:
        time = np.linspace(0.0, 1.0, 1001)
        force = np.zeros_like(time)
        force[500] = 1.0
        params = {
            "window_rolloff_seconds": 0.0,
            "window_max_start_seconds": -0.05,
            "window_max_end_seconds": 0.1,
        }
        with self.assertRaisesRegex(ValueError, "window_rolloff_seconds"):
            signal_processing.create_raised_cosine_window(time, force, params, time[1] - time[0])


class TestRaisedCosineWindowShape(unittest.TestCase):
    def test_plateau_and_rolloff_geometry(self) -> None:
        dt = 0.001
        time = np.arange(0.0, 1.0, dt)
        peak_idx = 500
        force = np.zeros_like(time)
        force[peak_idx] = 1.0

        params = {
            "window_rolloff_seconds": 0.02,
            "window_max_start_seconds": -0.03,
            "window_max_end_seconds": 0.08,
        }
        window, meta = signal_processing.create_raised_cosine_window(time, force, params, dt)

        peak_time = float(time[peak_idx])
        plateau_start = peak_time + params["window_max_start_seconds"]
        plateau_end = peak_time + params["window_max_end_seconds"]
        support_start = plateau_start - params["window_rolloff_seconds"]
        support_end = plateau_end + params["window_rolloff_seconds"]

        plateau_mask = (time >= plateau_start) & (time <= plateau_end)
        self.assertTrue(np.allclose(window[plateau_mask], 1.0))

        outside = (time < support_start) | (time > support_end)
        self.assertTrue(np.allclose(window[outside], 0.0))

        rising = (time > support_start) & (time < plateau_start)
        self.assertGreater(float(np.max(window[rising])), 0.0)
        self.assertLess(float(np.max(window[rising])), 1.0)

        falling = (time > plateau_end) & (time < support_end)
        self.assertGreater(float(np.max(window[falling])), 0.0)
        self.assertLess(float(np.max(window[falling])), 1.0)

        self.assertAlmostEqual(meta["plateau_start_time"], plateau_start)
        self.assertAlmostEqual(meta["plateau_end_time"], plateau_end)

    def test_create_window_accepts_legacy_params(self) -> None:
        dt = 0.001
        time = np.arange(0.0, 1.0, dt)
        force = np.zeros_like(time)
        force[500] = 1.0
        params = {
            "window_pre_seconds": 0.05,
            "window_post_seconds": 0.1,
            "window_beta": 0.25,
        }
        window, _ = frf_estimation.create_window(force, time, params, dt, window_type="raised-cosine")
        self.assertEqual(window.shape, time.shape)
        self.assertGreater(float(np.max(window)), 0.0)


if __name__ == "__main__":
    unittest.main()
