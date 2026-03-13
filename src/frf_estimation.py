#!/usr/bin/env python3
"""
High-level FRF estimation entry points.

This module provides a clean importable API for FRF estimation that can
be called from main.py or a future GUI. It wraps the existing
implementation in frf-estimation.py and adds a simple FRF-from-signals
helper.
"""

from __future__ import annotations

import os
import importlib.util
from typing import Dict, Any, Tuple

import numpy as np
from scipy import signal


_HERE = os.path.dirname(__file__)
_LEGACY_PATH = os.path.join(_HERE, "frf-estimation.py")

_spec = importlib.util.spec_from_file_location("frf_estimation_legacy", _LEGACY_PATH)
_legacy = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_legacy)

# Re-export commonly used legacy functionality so other code can import
# from this module instead of using a hyphenated filename.
get_system_parameters = _legacy.get_system_parameters
plot_bode_diagram = getattr(_legacy, "plot_bode_diagram", None)
plot_time_domain_signals = getattr(_legacy, "plot_time_domain_signals", None)
plot_signal_comparison = getattr(_legacy, "plot_signal_comparison", None)
calculate_derived_signals = getattr(_legacy, "calculate_derived_signals", None)
detect_case3_from_filename = getattr(_legacy, "detect_case3_from_filename", None)
parse_filename = getattr(_legacy, "parse_filename", None)
load_csv_file = getattr(_legacy, "load_csv_file", None)
remove_bias = getattr(_legacy, "remove_bias", None)
apply_wiener_filter = getattr(_legacy, "apply_wiener_filter", None)
apply_savgol_filter = getattr(_legacy, "apply_savgol_filter", None)
apply_lowpass_filter = getattr(_legacy, "apply_lowpass_filter", None)
calculate_noise_power = getattr(_legacy, "calculate_noise_power", None)


def compute_frf_from_signals(
    time: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    params: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a single-input/single-output FRF H(f) = YU*/UU* using Welch CSD.

    Returns (frequencies, H) where H is complex-valued.
    """
    time = np.asarray(time)
    u = np.asarray(input_signal, dtype=float)
    y = np.asarray(output_signal, dtype=float)

    if len(time) < 2:
        raise ValueError("Time vector must contain at least two samples")

    dt = float(time[1] - time[0])
    fs = 1.0 / dt

    if params is None:
        params = get_system_parameters()

    nperseg = int(params.get("welch_nperseg", 2048))
    noverlap = int(params.get("welch_noverlap", nperseg // 2))
    window = params.get("welch_window", "hamming")

    f, Puy = signal.csd(u, y, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    f2, Puu = signal.csd(u, u, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)

    if not np.allclose(f, f2):
        raise RuntimeError("Frequency grids from CSD calculations do not match")

    H = Puy / (Puu + 1e-20)
    return f, H

