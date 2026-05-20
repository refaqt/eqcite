#!/usr/bin/env python3
"""
Shared signal processing utilities for FRF and stability analysis.

This module contains CSV/header parsing, basic filtering, and window
construction functions that are used by both frf_estimation and
stability_measurement (and can be reused elsewhere).
"""

from __future__ import annotations

import os
import re
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import signal


# ---------------------------------------------------------------------------
# CSV / header parsing
# ---------------------------------------------------------------------------

def parse_sample_rate_from_header(filepath: str) -> float:
    """
    Parse sample rate from CSV file header.

    The header contains a line like "Samples per sec. = 56000/1" where the number
    is the total samples per second for all channels combined.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, "r") as f:
            first_line = f.readline()
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"Error reading CSV file header: {e}")

    pattern = r"Samples\s+per\s+sec\.\s*=\s*(\d+)/\d+"
    match = re.search(pattern, first_line, re.IGNORECASE)
    if not match:
        raise ValueError(
            "Could not find sample rate in CSV header. "  # noqa: E501
            "Expected format: 'Samples per sec. = XXXXX/1'. "  # noqa: E501
            f"First line: {first_line.strip()}"
        )

    return float(match.group(1))


def load_csv_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, float, float]:
    """
    Load and parse CSV file, return time array, analog input arrays (A0–A7), fs, and dt.

    Returns (time, A0..A7, fs, dt) where A7 is a zero-array if the file
    only contains 7 analog channels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    total_samples_per_sec = parse_sample_rate_from_header(filepath)

    df = pd.read_csv(
        filepath,
        skiprows=5,
        header=None,
        sep=",",
        decimal=".",
        skipinitialspace=True,
    )

    n_cols = df.shape[1]
    if n_cols < 8 or n_cols > 9:
        raise ValueError(
            f"Expected 8 or 9 columns (Time + A0–A6 [+ optional A7]), found {n_cols}"
        )

    num_data_channels = n_cols - 1
    fs = total_samples_per_sec / num_data_channels
    dt = 1.0 / fs

    column_names = ["Time", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]

    for i, _ in enumerate(column_names):
        if i >= df.shape[1]:
            continue
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors="coerce")

    df = df.dropna(how="any")

    time = df.iloc[:, 0].to_numpy()

    def col(idx: int) -> np.ndarray:
        return df.iloc[:, idx].to_numpy()

    A0 = col(1)
    A1 = col(2)
    A2 = col(3)
    A3 = col(4)
    A4 = col(5)
    A5 = col(6)
    A6 = col(7)
    if n_cols == 9:
        A7 = col(8)
    else:
        A7 = np.zeros_like(A0)

    return time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt


# ---------------------------------------------------------------------------
# Basic filtering
# ---------------------------------------------------------------------------

def remove_bias_window(
    signal_data: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, int, int]:
    """
    Remove DC bias by sliding-window minimum variance (quiet segment), same rule as batch FRF.

    Returns:
        (bias_corrected, window_start_idx, actual_window_size)

    The returned index and length identify the segment used for both bias removal
    and (on the corrected signal) Wiener noise variance via
    `calculate_noise_power_from_window`.
    """
    signal = np.asarray(signal_data, dtype=float)
    signal_length = len(signal)
    if signal_length == 0:
        return signal, 0, 0

    window_size = int(window_size)
    if window_size <= 0:
        return signal, 0, 0

    if signal_length < window_size:
        print(
            f"Warning: Signal length ({signal_length}) is less than window size ({window_size}). "
            "Using entire signal for bias calculation."
        )
        bias = float(np.mean(signal))
        return signal - bias, 0, signal_length

    num_windows = signal_length - window_size + 1
    variances = np.zeros(num_windows)
    for i in range(num_windows):
        variances[i] = np.var(signal[i : i + window_size])

    min_var_idx = int(np.argmin(variances))
    bias = float(np.mean(signal[min_var_idx : min_var_idx + window_size]))
    return signal - bias, min_var_idx, window_size


def quiet_regions_mask(
    time_data: np.ndarray,
    impact_start_s: float,
    ringdown_end_s: float,
) -> np.ndarray:
    """
    Build boolean mask for quiet regions used in trend estimation.

    Quiet data is defined as:
      - pre-impact: t < impact_start_s
      - post-ringdown: t > ringdown_end_s
    """
    time = np.asarray(time_data, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if len(time) == 0:
        raise ValueError("time must contain at least one sample")

    t_impact = float(impact_start_s)
    t_ringdown = float(ringdown_end_s)
    if not np.isfinite(t_impact) or not np.isfinite(t_ringdown):
        raise ValueError("impact_start_s and ringdown_end_s must be finite numbers")
    if t_ringdown <= t_impact:
        raise ValueError("ringdown_end_s must be greater than impact_start_s")

    pre_mask = time < t_impact
    post_mask = time > t_ringdown
    if not np.any(pre_mask):
        raise ValueError("No pre-impact quiet samples found before impact_start_s")
    if not np.any(post_mask):
        raise ValueError("No post-ringdown quiet samples found after ringdown_end_s")
    return np.logical_or(pre_mask, post_mask)


def detrend_quiet_regions(
    time_data: np.ndarray,
    signal_data: np.ndarray,
    impact_start_s: float,
    ringdown_end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit linear trend on quiet regions and subtract it over the full signal.
    """
    time = np.asarray(time_data, dtype=float)
    signal = np.asarray(signal_data, dtype=float)
    if time.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and signal must be one-dimensional arrays")
    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")
    if len(time) < 2:
        raise ValueError("At least two samples are required for detrending")
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains non-finite values")

    mask = quiet_regions_mask(time, impact_start_s, ringdown_end_s)
    quiet_t = time[mask]
    quiet_signal = signal[mask]
    if len(quiet_t) < 2:
        raise ValueError("At least two quiet samples are required to fit a linear trend")

    coeffs = np.polyfit(quiet_t, quiet_signal, deg=1)
    trend = np.polyval(coeffs, time)
    return signal - trend, trend


def calculate_noise_power_from_mask(signal_data: np.ndarray, mask: np.ndarray) -> float:
    """
    Noise variance over masked samples (population variance, ddof=0).
    """
    data = np.asarray(signal_data, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if data.ndim != 1:
        raise ValueError("signal_data must be one-dimensional")
    if m.ndim != 1 or len(m) != len(data):
        raise ValueError("mask must be one-dimensional and match signal length")
    if not np.any(m):
        raise ValueError("mask must select at least one sample")
    window = data[m]
    return float(np.var(window, ddof=0))


def calculate_noise_power_from_window(
    signal_data: np.ndarray,
    window_start_idx: int,
    window_size: int,
) -> float:
    """
    Noise variance on ``signal_data[window_start_idx : window_start_idx + window_size]``.

    Uses population variance (``ddof=0``), matching ``scipy.signal.wiener`` local variance
    ``mean(x^2) - mean(x)^2`` so the ``noise`` argument is on the same scale as ``lVar``.
    """
    data = np.asarray(signal_data, dtype=float)
    n = len(data)
    if n == 0:
        return 0.0

    window_start_idx = int(max(0, window_start_idx))
    window_size = int(max(0, window_size))
    if window_size == 0 or window_start_idx >= n:
        return 0.0

    end = min(n, window_start_idx + window_size)
    window = data[window_start_idx:end]
    if len(window) == 0:
        return 0.0

    return float(np.var(window, ddof=0))


def wiener_mysize_to_odd(window_size: int) -> int:
    """SciPy recommends odd ``mysize``; coerce to largest odd <= max(window_size, 3)."""
    w = int(window_size)
    if w < 3:
        return 3
    if w % 2 == 0:
        return w - 1
    return w


def remove_bias(signal_data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Remove DC bias from a signal using a moving-average window.
    """
    data = np.asarray(signal_data, dtype=float)
    if window_size <= 0 or window_size >= len(data):
        return data

    kernel = np.ones(window_size) / window_size
    bias = np.convolve(data, kernel, mode="same")
    return data - bias


def calculate_noise_power(signal_data: np.ndarray, noise_fraction: float = 0.1) -> float:
    """
    Estimate noise power from the quietest portion of a signal.
    """
    data = np.asarray(signal_data, dtype=float)
    n = len(data)
    if n == 0:
        return 0.0

    window = max(1, int(noise_fraction * n))
    if window >= n:
        window = n

    squared = data**2
    cumsum = np.cumsum(squared)
    power_window = cumsum[window - 1 :] - np.concatenate(
        ([0.0], cumsum[:-window])
    )
    min_power = float(np.min(power_window) / window)
    return max(min_power, 0.0)


def apply_wiener_filter(
    signal_data: np.ndarray,
    noise_power: float,
    fs: float,
    nperseg: int = 1024,
) -> np.ndarray:
    """
    Apply a simple frequency-domain Wiener filter to a signal.
    """
    data = np.asarray(signal_data, dtype=float)
    if len(data) == 0:
        return data

    f, Pxx = signal.welch(data, fs=fs, nperseg=min(nperseg, len(data)))
    Pxx_interp = np.interp(
        np.fft.rfftfreq(len(data), 1.0 / fs),
        f,
        Pxx,
    )

    Sxx = Pxx_interp
    Nxx = noise_power
    H = Sxx / (Sxx + Nxx + 1e-12)

    spectrum = np.fft.rfft(data)
    filtered_spectrum = spectrum * H
    filtered = np.fft.irfft(filtered_spectrum, n=len(data))
    return filtered


def apply_wiener_filter_legacy(
    signal_data: np.ndarray,
    noise_power: float,
    window_size: int,
) -> np.ndarray:
    """
    Legacy-compatible Wiener filter wrapper (matches previous `frf_estimation.py` usage).

    This uses `scipy.signal.wiener` with explicit noise power and window size.
    ``window_size`` is coerced to an odd length >= 3 (SciPy recommendation).
    """
    data = np.asarray(signal_data, dtype=float)
    mysize = wiener_mysize_to_odd(window_size)
    return signal.wiener(data, mysize=mysize, noise=float(noise_power))


def apply_lowpass_filter(
    signal_data: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to a signal.
    """
    data = np.asarray(signal_data, dtype=float)
    if cutoff_hz <= 0 or fs <= 0 or cutoff_hz >= fs / 2.0:
        return data

    wn = cutoff_hz / (fs / 2.0)
    b, a = signal.butter(order, wn, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def apply_lowpass_filter_bessel(
    signal_data: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Legacy-compatible low-pass filter (Bessel) used by older FRF/stability code.
    """
    data = np.asarray(signal_data, dtype=float)
    if cutoff_hz <= 0 or fs <= 0 or cutoff_hz >= fs / 2.0:
        return data
    b, a = signal.bessel(N=int(order), Wn=float(cutoff_hz), btype="low", analog=False, fs=float(fs))
    return signal.filtfilt(b, a, data)


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------

_DEFAULT_RC_ROLLOFF_SECONDS = 0.01875
_DEFAULT_RC_MAX_START_SECONDS = -0.03125
_DEFAULT_RC_MAX_END_SECONDS = 0.08125


def normalize_raised_cosine_window_params(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Return peak-relative raised-cosine shape parameters, migrating legacy pre/post/beta when needed.
    """
    p = dict(params or {})
    new_keys = (
        "window_rolloff_seconds",
        "window_max_start_seconds",
        "window_max_end_seconds",
    )
    if all(k in p for k in new_keys):
        rolloff = float(p["window_rolloff_seconds"])
        max_start = float(p["window_max_start_seconds"])
        max_end = float(p["window_max_end_seconds"])
    elif any(k in p for k in ("window_pre_seconds", "window_post_seconds", "window_beta")):
        pre = max(0.0, float(p.get("window_pre_seconds", 0.05)))
        post = max(0.0, float(p.get("window_post_seconds", 0.1)))
        beta = min(max(float(p.get("window_beta", 0.25)), 0.0), 1.0)
        total = pre + post
        rolloff = beta * total / 2.0
        max_start = -pre + rolloff
        max_end = post - rolloff
    else:
        rolloff = float(p.get("window_rolloff_seconds", _DEFAULT_RC_ROLLOFF_SECONDS))
        max_start = float(p.get("window_max_start_seconds", _DEFAULT_RC_MAX_START_SECONDS))
        max_end = float(p.get("window_max_end_seconds", _DEFAULT_RC_MAX_END_SECONDS))

    return {
        "window_rolloff_seconds": rolloff,
        "window_max_start_seconds": max_start,
        "window_max_end_seconds": max_end,
    }


def create_raised_cosine_window(
    time: np.ndarray,
    force_signal: np.ndarray,
    params: Dict[str, Any],
    dt: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a raised cosine window around the dominant force peak.

    Shape parameters are peak-relative seconds: rolloff duration, plateau start, and plateau end.
    """
    time = np.asarray(time, dtype=float)
    force_signal = np.asarray(force_signal)

    if time.shape != force_signal.shape:
        raise ValueError("Time and force_signal must have the same shape for windowing")
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time array must be one-dimensional with at least two samples")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-positive or non-finite sample interval detected")

    shape = normalize_raised_cosine_window_params(params)
    rolloff = float(shape["window_rolloff_seconds"])
    max_start = float(shape["window_max_start_seconds"])
    max_end = float(shape["window_max_end_seconds"])

    if rolloff <= 0:
        raise ValueError("window_rolloff_seconds must be positive")
    if max_end <= max_start:
        raise ValueError("window_max_end_seconds must be greater than window_max_start_seconds")

    peak_idx = int(np.argmax(np.abs(force_signal)))
    peak_time = float(time[peak_idx])

    plateau_start = peak_time + max_start
    plateau_end = peak_time + max_end
    support_start = plateau_start - rolloff
    support_end = plateau_end + rolloff

    support_start = max(support_start, float(time[0]))
    support_end = min(support_end, float(time[-1]))
    if support_end <= support_start:
        raise ValueError("Raised-cosine window support is empty after clamping to the time vector")

    window = np.zeros_like(time, dtype=float)
    t = time
    rising = (t >= support_start) & (t < plateau_start)
    if np.any(rising):
        phase = np.pi * (t[rising] - support_start) / rolloff
        window[rising] = 0.5 * (1.0 - np.cos(phase))

    flat = (t >= plateau_start) & (t <= plateau_end)
    window[flat] = 1.0

    falling = (t > plateau_end) & (t <= support_end)
    if np.any(falling):
        phase = np.pi * (support_end - t[falling]) / rolloff
        window[falling] = 0.5 * (1.0 - np.cos(phase))

    start_idx = int(np.searchsorted(time, support_start, side="left"))
    end_idx = int(np.searchsorted(time, support_end, side="right"))
    start_idx = max(0, min(start_idx, len(time) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(time)))

    metadata = {
        "peak_time": peak_time,
        "start_time": float(time[start_idx]),
        "end_time": float(time[end_idx - 1]),
        "plateau_start_time": float(plateau_start),
        "plateau_end_time": float(plateau_end),
        "peak_index": peak_idx,
        "start_index": start_idx,
        "end_index": end_idx,
        "window_rolloff_seconds": rolloff,
        "window_max_start_seconds": max_start,
        "window_max_end_seconds": max_end,
    }
    return window, metadata


def create_hann_window(time: np.ndarray, center_time: float, width_seconds: float) -> np.ndarray:
    """
    Create a Hann window centered at center_time with given width.
    """
    time = np.asarray(time)
    n = len(time)
    if n == 0:
        return np.zeros(0)

    half_width = width_seconds / 2.0
    start_time = center_time - half_width
    end_time = center_time + half_width

    start_idx = int(np.searchsorted(time, start_time, side="left"))
    end_idx = int(np.searchsorted(time, end_time, side="right"))
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx + 1, min(end_idx, n))

    window = np.zeros_like(time, dtype=float)
    segment_length = end_idx - start_idx
    window[start_idx:end_idx] = signal.windows.hann(segment_length, sym=False)
    return window


def create_constant_window(time: np.ndarray) -> np.ndarray:
    """
    Window of ones over the full time vector.
    """
    return np.ones_like(np.asarray(time), dtype=float)


def create_window(
    time: np.ndarray,
    force: np.ndarray,
    params: Dict[str, Any],
    dt: float,
    window_type: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Dispatch to the configured window type.
    """
    time = np.asarray(time)
    force = np.asarray(force)
    win_type = (window_type or "hann").lower()

    if win_type == "raised-cosine":
        return create_raised_cosine_window(time, force, params, dt)
    if win_type == "constant":
        w = create_constant_window(time)
        return w, {
            "peak_time": float(time[len(time) // 2]) if len(time) else 0.0,
            "start_time": float(time[0]) if len(time) else 0.0,
            "end_time": float(time[-1]) if len(time) else 0.0,
        }

    # Default: Hann around the hammer event (centered at max |force|)
    peak_idx = int(np.argmax(np.abs(force))) if len(force) else 0
    center_time = float(time[peak_idx]) if len(time) else 0.0
    width = params.get("window_hann_width", 0.05)
    w = create_hann_window(time, center_time, width)
    meta = {
        "peak_time": center_time,
        "start_time": float(time[0]) if len(time) else 0.0,
        "end_time": float(time[-1]) if len(time) else 0.0,
    }
    return w, meta

