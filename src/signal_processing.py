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


def apply_savgol_filter(
    signal_data: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """
    Apply a Savitzky–Golay filter to a signal.
    """
    data = np.asarray(signal_data, dtype=float)
    if window_length <= 0 or len(data) < window_length:
        return data

    if window_length % 2 == 0:
        window_length -= 1
        if window_length < 3:
            return data

    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:
            return data

    return signal.savgol_filter(data, window_length, polyorder)


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


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------

def create_raised_cosine_window(
    time: np.ndarray,
    force_signal: np.ndarray,
    pre_seconds: float,
    post_seconds: float,
    beta: float,
    dt: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a raised cosine (Tukey) window around the dominant force peak.
    """
    time = np.asarray(time)
    force_signal = np.asarray(force_signal)

    if time.shape != force_signal.shape:
        raise ValueError("Time and force_signal must have the same shape for windowing")
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Time array must be one-dimensional with at least two samples")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-positive or non-finite sample interval detected")

    peak_idx = int(np.argmax(np.abs(force_signal)))
    peak_time = time[peak_idx]

    pre_seconds = max(0.0, float(pre_seconds))
    post_seconds = max(0.0, float(post_seconds))
    beta = float(beta)
    beta = min(max(beta, 0.0), 1.0)

    start_time = max(peak_time - pre_seconds, time[0])
    end_time = min(peak_time + post_seconds, time[-1])
    if end_time <= start_time:
        window = np.zeros_like(time, dtype=float)
        window[peak_idx] = 1.0
        return window, {
            "peak_time": peak_time,
            "start_time": peak_time,
            "end_time": peak_time,
            "peak_index": peak_idx,
            "start_index": peak_idx,
            "end_index": peak_idx + 1,
        }

    start_idx = int(np.searchsorted(time, start_time, side="left"))
    end_idx = int(np.searchsorted(time, end_time, side="right"))
    if end_idx <= start_idx + 1:
        window = np.zeros_like(time, dtype=float)
        window[peak_idx] = 1.0
        return window, {
            "peak_time": peak_time,
            "start_time": peak_time,
            "end_time": peak_time,
            "peak_index": peak_idx,
            "start_index": peak_idx,
            "end_index": peak_idx + 1,
        }

    segment_length = end_idx - start_idx
    tukey_segment = signal.windows.tukey(segment_length, alpha=beta, sym=False)

    window = np.zeros_like(time, dtype=float)
    window[start_idx:end_idx] = tukey_segment

    metadata = {
        "peak_time": peak_time,
        "start_time": time[start_idx],
        "end_time": time[end_idx - 1],
        "peak_index": peak_idx,
        "start_index": start_idx,
        "end_index": end_idx,
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
        return create_raised_cosine_window(
            time,
            force,
            params.get("window_pre_seconds", 0.05),
            params.get("window_post_seconds", 0.1),
            params.get("window_beta", 0.25),
            dt,
        )
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

