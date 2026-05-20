"""Unit tests for signal preprocessing helpers in ``signal_processing``."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import signal

from src import signal_processing


class TestNoiseWindow(unittest.TestCase):
    def test_calculate_noise_power_is_variance_not_mean_square(self) -> None:
        # Constant nonzero mean: mean(x^2) != var(x)
        w = np.array([3.0, 3.0, 3.0])
        ms = float(np.mean(w**2))
        v = signal_processing.calculate_noise_power_from_window(w, 0, len(w))
        self.assertAlmostEqual(v, 0.0, places=10)
        self.assertGreater(ms, v)

    def test_remove_bias_window_finds_low_variance_segment(self) -> None:
        rng = np.random.default_rng(0)
        quiet = rng.normal(scale=0.01, size=200)
        loud = rng.normal(scale=2.0, size=200)
        x = np.concatenate([quiet, loud])
        y, idx, w = signal_processing.remove_bias_window(x, window_size=80)
        self.assertEqual(w, 80)
        self.assertLessEqual(idx, 120)
        self.assertGreaterEqual(idx, 0)
        # Bias-corrected quiet segment should stay small
        self.assertLess(float(np.std(y[idx : idx + 80])), 0.05)

    def test_wiener_does_not_wipe_sine_with_matched_noise(self) -> None:
        n = 2000
        t = np.linspace(0, 1, n, endpoint=False)
        carrier = np.sin(2 * np.pi * 50 * t)
        rng = np.random.default_rng(1)
        noise = rng.normal(scale=0.05, size=n)
        x = carrier + noise
        y, idx, w = signal_processing.remove_bias_window(x, window_size=150)
        nvar = signal_processing.calculate_noise_power_from_window(y, idx, w)
        nvar = max(nvar, 1e-12)
        out = signal_processing.apply_wiener_filter_legacy(y, nvar, 31)
        rms_in = float(np.sqrt(np.mean(y**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        self.assertGreater(rms_out, 0.1 * rms_in)


class TestWienerOddMysize(unittest.TestCase):
    def test_wiener_mysize_to_odd(self) -> None:
        self.assertEqual(signal_processing.wiener_mysize_to_odd(1000), 999)
        self.assertEqual(signal_processing.wiener_mysize_to_odd(999), 999)
        self.assertEqual(signal_processing.wiener_mysize_to_odd(2), 3)

    def test_legacy_matches_scipy_odd(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        n = 0.5
        a = signal_processing.apply_wiener_filter_legacy(x, n, 10)
        b = signal.wiener(x, mysize=9, noise=n)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


class TestQuietRegionDetrend(unittest.TestCase):
    def test_detrend_removes_linear_offset_from_quiet_regions(self) -> None:
        t = np.linspace(0.0, 1.0, 2001)
        trend = 0.5 * t + 2.0
        pulse = np.exp(-((t - 0.5) ** 2) / (2 * (0.01**2)))
        x = trend + pulse

        y, fitted = signal_processing.detrend_quiet_regions(
            t,
            x,
            impact_start_s=0.4,
            ringdown_end_s=0.6,
        )
        quiet = signal_processing.quiet_regions_mask(t, 0.4, 0.6)
        # Fitted trend should match original trend on quiet regions.
        np.testing.assert_allclose(fitted[quiet], trend[quiet], atol=1e-10)
        # Detrended signal should be near zero in quiet regions.
        self.assertLess(float(np.max(np.abs(y[quiet]))), 1e-9)

    def test_quiet_regions_mask_requires_pre_and_post_samples(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        with self.assertRaises(ValueError):
            signal_processing.quiet_regions_mask(t, impact_start_s=-0.1, ringdown_end_s=0.8)
        with self.assertRaises(ValueError):
            signal_processing.quiet_regions_mask(t, impact_start_s=0.1, ringdown_end_s=2.0)

    def test_detrend_requires_valid_bounds(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        x = np.sin(2 * np.pi * t)
        with self.assertRaises(ValueError):
            signal_processing.detrend_quiet_regions(t, x, impact_start_s=0.6, ringdown_end_s=0.4)

    def test_noise_power_from_mask_matches_variance(self) -> None:
        x = np.array([1.0, 1.0, 2.0, 4.0, 4.0])
        mask = np.array([False, True, True, True, False])
        got = signal_processing.calculate_noise_power_from_mask(x, mask)
        expected = float(np.var(np.array([1.0, 2.0, 4.0]), ddof=0))
        self.assertAlmostEqual(got, expected, places=12)


if __name__ == "__main__":
    unittest.main()
