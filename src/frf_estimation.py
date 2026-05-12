#!/usr/bin/env python3
"""
Frequency Response Analysis Tool

A Python application for analyzing frequency response measurements from CSV data files.
Processes multi-channel analog input data, calculates spatial coordinates and angular 
displacements, performs FFT analysis, and generates transfer function (compliance) plots.

Author: FRF Analysis Tool
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
from pathlib import Path
import warnings
from scipy import signal
from pyFRF import FRF
import sys
from typing import Any, Dict, Optional, Tuple

from . import signal_processing

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Plot control: if False, only show coherence and Bode plots; if True, show all plots
show_all_figures = True

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================

def get_system_parameters():
    """
    Return dictionary of system parameters.
    
    Returns:
    --------
    dict : System parameters including geometric, angular, and sensor calibration values
    """
    # Geometric parameters (meters)
    L1 = 0.02      # (m) Length parameter 1 - from tool tip to impact point
    L2 = 0.078      # (m) Length parameter 2 - from impact point to bottom accelerometers
    L3 = 0.160       # (m) Length parameter 3 - from bottom to top accelerometers
    
    # Angular parameter
    alpha = 20.123 * np.pi / 180  # Angle in radians (20 degrees)
    
    # Sensor calibration parameters
    r_a = 1 * 9.81 / 0.2    # Accelerometer sensitivity (m/s²/V)
    m_u = 10               # IEPE signal conditioner multiplication factor
    v_u = 2.5e-3           # Force hammer sensitivity (mV/N)
    r_u = 1 / (v_u * m_u)  # Combined output sensitivity (N/V)
    
    # Bias removal parameters
    bias_window_size = 1000  # Number of samples for bias estimation window

    # Wiener: local mean/variance window for scipy.signal.wiener (independent of bias_window_size)
    wiener_mysize = 1000
    
    # Wiener filter parameters
    wiener_filter = True  # Toggle to enable/disable Wiener filter for channels A0-A7
    
    # Savitzky-Golay filter parameters
    savgol_filter = True  # Toggle to enable/disable Savitzky-Golay filter for channels A0-A7
    savgol_window_length = 10  # Window length for Savitzky-Golay filter (must be odd)
    savgol_polyorder = 2  # Polynomial order for Savitzky-Golay filter
    
    # Low-pass filter parameters (for analysis visualization only)
    lowpass_filter = False  # Toggle to enable/disable low-pass filter for channels A0-A7
    lowpass_bandwidth_hz = 100.0  # Low-pass filter bandwidth (Hz) for analysis plots
    
    # Window type parameters
    input_window_type = 'hann'  # Window type for input (force) signal: 'hann', 'exponential', 'raised-cosine', 'signal-based', or 'constant'
    output_window_type = 'exponential'  # Window type for output (acceleration) signals: 'hann', 'exponential', 'raised-cosine', 'signal-based', or 'constant'

    # Window parameter for Hann window
    window_hann_width = 0.05  # Hann window width in seconds

    # Window parameters for raised cosine (Tukey) windowing
    window_pre_seconds = 0.05   # Seconds before peak force to start window
    window_post_seconds = 0.1   # Seconds after peak force to end window
    window_beta = 0.25          # Tukey window roll-off (0 to 1)
    window_rolloff_seconds = 0.005  # Rolloff duration and peak offset timing for signal-based windows (seconds)

    # Window parameters for signal-based raised cosine (Tukey) windowing
    window_trigger_percent = 0.1  # Threshold percentage for trigger point (10% of max)
    window_low_percent = 0.001  # Threshold percentage for low point (2% of max)
    window_sliding_size = 200  # Size of sliding window in samples for low_percent detection
    window_std_multiplier = 3.0  # Multiplier for standard deviation in low_percent condition

    # Window parameters for exponential window
    window_decay_db = 40.0  # Exponential window decay in dB
    
    # Welch method parameters for pyFRF (used for CSD-based transfer function estimation)
    welch_nperseg = 2500       # Length of each segment for Welch method (None = scipy default: 256 or signal length)
    welch_noverlap = 2000      # Number of points to overlap between segments (None = scipy default: nperseg // 2)
    welch_window = 'hamming'      # Window type for Welch method
    
    # FRF estimator type
    estimator_type = 'Hv'      # FRF estimator type: 'H1', 'H2', 'Hv', or 'ODS'

    return {
        'L1': L1,
        'L2': L2,
        'L3': L3,
        'alpha': alpha,
        'r_a': r_a,
        'm_u': m_u,
        'v_u': v_u,
        'r_u': r_u,
        'bias_window_size': bias_window_size,
        'wiener_mysize': wiener_mysize,
        'wiener_filter': wiener_filter,
        'savgol_filter': savgol_filter,
        'savgol_window_length': savgol_window_length,
        'savgol_polyorder': savgol_polyorder,
        'lowpass_filter': lowpass_filter,
        'lowpass_bandwidth_hz': lowpass_bandwidth_hz,
        'input_window_type': input_window_type,
        'output_window_type': output_window_type,
        'window_hann_width': window_hann_width,
        'window_pre_seconds': window_pre_seconds,
        'window_post_seconds': window_post_seconds,
        'window_beta': window_beta,
        'window_rolloff_seconds': window_rolloff_seconds,
        'window_trigger_percent': window_trigger_percent,
        'window_low_percent': window_low_percent,
        'window_sliding_size': window_sliding_size,
        'window_std_multiplier': window_std_multiplier,
        'window_decay_db': window_decay_db,
        'welch_nperseg': welch_nperseg,
        'welch_noverlap': welch_noverlap,
        'welch_window': welch_window,
        'estimator_type': estimator_type
    }

# =============================================================================
# FRF (single-input/single-output) helper
# =============================================================================
def compute_frf_from_signals(
    time: np.ndarray,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FRF H(f) = YU* / UU* using Welch CSD.

    Returns:
        (frequencies, H) where H is complex-valued.
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
    # Clamp Welch settings to valid ranges for short signals (common in simulations/smoke tests).
    nperseg = max(2, min(nperseg, len(u), len(y)))
    noverlap = int(params.get("welch_noverlap", nperseg // 2))
    noverlap = max(0, min(noverlap, nperseg - 1))
    window = params.get("welch_window", "hamming")
    estimator_type = str(params.get("estimator_type", "Hv") or "Hv")

    f, Puy = signal.csd(
        u,
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )
    f2, Puu = signal.csd(
        u,
        u,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )

    if not np.allclose(f, f2):
        raise RuntimeError("Frequency grids from CSD calculations do not match")

    # Branch on estimator type (H1, H2, Hv).
    # eps avoids division by ~0 in the spectral estimates.
    eps = 1e-20
    et = estimator_type.strip()
    if et not in {"H1", "H2", "Hv"}:
        et = "Hv"

    if et == "H1":
        # H1(ω) = G_yu / G_uu
        H = Puy / (Puu + eps)
    else:
        # Need additional spectra for H2 and Hv.
        f3, Pyy = signal.csd(
            y,
            y,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        f4, Pyu = signal.csd(
            y,
            u,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        if not (np.allclose(f, f3) and np.allclose(f, f4)):
            raise RuntimeError("Frequency grids from CSD calculations do not match")

        if et == "H2":
            # H2(ω) = G_yy / G_yu
            H = Pyy / (Pyu + eps)
        else:
            # Hv(ω) = (G_yu/|G_yu|) * sqrt(G_yy/G_uu)
            phase = Puy / (np.abs(Puy) + eps)
            H = phase * np.sqrt((Pyy + eps) / (Puu + eps))
    return f, H

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def parse_sample_rate_from_header(filepath):
    """
    Parse sample rate from CSV file header.
    
    The header contains a line like "Samples per sec. = 56000/1" where the number
    is the total samples per second for all channels combined.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    float : Total samples per second for all channels combined
        
    Raises:
    -------
    ValueError : If sample rate cannot be found in header
    FileNotFoundError : If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read first line to get header
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
    except Exception as e:
        raise ValueError(f"Error reading CSV file header: {e}")
    
    # Search for sample rate in first line
    # Pattern: "Samples per sec. = 56000/1" or similar variations
    pattern = r'Samples\s+per\s+sec\.\s*=\s*(\d+)/\d+'
    
    match = re.search(pattern, first_line, re.IGNORECASE)
    if match:
        total_sample_rate = float(match.group(1))
        return total_sample_rate
    
    # If not found, raise error
    raise ValueError(
        f"Could not find sample rate in CSV header. "
        f"Expected format: 'Samples per sec. = XXXXX/1'. "
        f"First line: {first_line.strip()}"
    )

def load_csv_file(filepath):
    """
    Load and parse CSV file, return time array, analog input arrays (A0-A7), fs, and dt.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple : (time_array, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt)
        Time array, 8 analog input arrays, sampling frequency (Hz), and sample time (s).
        If the CSV contains only 7 analog channels (Time + A0–A6), A7 is returned
        as a zero array with the same length as the time array so that downstream
        code can always assume 8 channels are present.
        fs is the per-channel sampling frequency calculated from the header.
        dt is the sample time (1/fs).
        
    Raises:
    -------
    ValueError : If file format is invalid or data cannot be parsed
    FileNotFoundError : If file doesn't exist
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Parse sample rate from header (total samples per second for all channels)
        total_samples_per_sec = parse_sample_rate_from_header(filepath)
        
        # Read CSV file, skipping first 5 header lines
        df = pd.read_csv(
            filepath, 
            skiprows=5, 
            header=None,
            sep=',',  # Explicit separator
            decimal='.',  # Explicit decimal point
            skipinitialspace=True  # Skip leading whitespace
        )
        
        # Validate column count: accept 8 or 9 columns
        # 8 columns: Time + A0–A6 (7 analog channels)
        # 9 columns: Time + A0–A7 (8 analog channels)
        n_cols = df.shape[1]
        if n_cols < 8 or n_cols > 9:
            raise ValueError(
                f"Expected 8 or 9 columns (Time + A0–A6 [+ optional A7]), "
                f"found {n_cols}"
            )
        
        # Calculate per-channel sampling frequency
        # Number of data channels = total columns - 1 (time column)
        num_data_channels = n_cols - 1
        fs = total_samples_per_sec / num_data_channels
        dt = 1 / fs
        
        # Define column names for clarity (we may not have all of them)
        column_names = ['Time', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
        
        # Convert each column to numeric with error handling
        for i, col_name in enumerate(column_names):
            if i >= df.shape[1]:
                continue
            
            # Try converting directly first
            df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')
            
            # If all values are NaN, try replacing comma with period (European format)
            if df.iloc[:, i].isna().all():
                df.iloc[:, i] = pd.to_numeric(
                    df.iloc[:, i].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
            
            # Check for issues
            nan_count = df.iloc[:, i].isna().sum()
            if nan_count > 0:
                first_nan_idx = df.iloc[:, i].isna().idxmax()
                raise ValueError(
                    f"Column {i} ({col_name}) has {nan_count} non-numeric values. "
                    f"First error at row {first_nan_idx + 5}. "
                    f"Please check the CSV file format."
                )
        
        # Extract time and analog inputs
        time = df.iloc[:, 0].values  # Column 0: Time
        A0 = df.iloc[:, 1].values    # Column 1: A0
        A1 = df.iloc[:, 2].values    # Column 2: A1
        A2 = df.iloc[:, 3].values    # Column 3: A2
        A3 = df.iloc[:, 4].values    # Column 4: A3
        A4 = df.iloc[:, 5].values    # Column 5: A4
        A5 = df.iloc[:, 6].values    # Column 6: A5
        A6 = df.iloc[:, 7].values    # Column 7: A6

        # Column 8 (A7) is optional; if it is not present, create a zero array
        if n_cols >= 9:
            A7 = df.iloc[:, 8].values    # Column 8: A7
        else:
            A7 = np.zeros_like(time)
        
        # Validate data is numeric and finite
        for i, arr in enumerate([time, A0, A1, A2, A3, A4, A5, A6, A7]):
            if arr.dtype == object:
                raise ValueError(f"Column {i} contains non-numeric data")
            if not np.isfinite(arr).all():
                nan_count = (~np.isfinite(arr)).sum()
                raise ValueError(
                    f"Column {i} has {nan_count} non-finite values (NaN or Inf)"
                )
        
        return time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt
        
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or malformed")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading file: {e}")

def parse_filename(filepath):
    """
    Extract date, time, position (X, Y, Z), and direction from filename.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    dict : Extracted metadata including date, time, positions, and direction
    """
    filename = os.path.basename(filepath)
    
    # Pattern: yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_[direction][optional_suffix].csv
    # Make pattern case-insensitive to handle both .csv and .CSV extensions
    # Optional suffix allows patterns like _x_1.csv, _x_test.csv, etc.
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_X([^_]+)_Y([^_]+)_Z([^_]+)_([xy])(?:_[^.]*)?\.csv'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Filename does not match expected format: {filename}")
    
    date_time, x_pos, y_pos, z_pos, direction = match.groups()
    
    return {
        'date_time': date_time,
        'x_position': float(x_pos),
        'y_position': float(y_pos),
        'z_position': float(z_pos),
        'direction': direction
    }

# =============================================================================
# SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def remove_bias(signal, window_size=1000):
    """
    Remove bias from signal by finding the minimal variance window and subtracting its mean.

    Delegates to ``signal_processing.remove_bias_window`` (single implementation).
    """
    return signal_processing.remove_bias_window(signal, window_size)


def calculate_noise_power(signal, window_start_idx, window_size):
    """Noise variance on a slice; same scale as ``scipy.signal.wiener`` ``noise``."""
    return signal_processing.calculate_noise_power_from_window(signal, window_start_idx, window_size)


def apply_wiener_filter(data, noise_power, mysize):
    """
    Apply Wiener filter to the data using scipy.signal.wiener.

    ``mysize`` is coerced to an odd length >= 3 (SciPy recommendation).
    """
    return signal_processing.apply_wiener_filter_legacy(data, noise_power, mysize)

def apply_lowpass_filter(data, cutoff_hz, fs):
    """
    Apply 4th-order Bessel low-pass filter to the data.
    
    Parameters:
    -----------
    data : array_like
        Input signal array
    cutoff_hz : float
        Cutoff frequency (bandwidth) in Hz
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    array_like : Low-pass filtered signal
    """
    # Design 4th-order Bessel low-pass filter
    b, a = signal.bessel(N=4, Wn=cutoff_hz, btype='low', analog=False, fs=fs)
    
    # Apply zero-phase filtering using filtfilt
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def apply_savgol_filter(data, window_length, polyorder):
    """
    Apply Savitzky-Golay filter to the data.
    
    Parameters:
    -----------
    data : array_like
        Input signal array
    window_length : int
        Window length for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter
        
    Returns:
    --------
    array_like : Savitzky-Golay filtered signal
    """
    data = np.asarray(data)
    signal_length = len(data)
    
    # Handle case where signal is shorter than window_length
    if signal_length < window_length:
        print(f"Warning: Signal length ({signal_length}) is less than window length ({window_length}). "
              f"Returning original signal.")
        return data
    
    # Ensure window_length is odd (required by savgol_filter)
    if window_length % 2 == 0:
        window_length = window_length - 1
        if window_length < 3:
            print(f"Warning: Adjusted window length ({window_length}) is too small. "
                  f"Returning original signal.")
            return data
    
    # Ensure polyorder is less than window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
        print(f"Warning: Polynomial order adjusted to {polyorder} (window_length - 1)")
    
    # Apply Savitzky-Golay filter
    filtered_data = signal.savgol_filter(data, window_length, polyorder)
    
    return filtered_data

def calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params, measurement_case='case1'):
    """
    Calculate u, x, y, theta_x, theta_y from analog inputs.
    
    Parameters:
    -----------
    A0, A1, A2, A3, A4, A5, A6, A7 : array_like
        Analog input arrays (Volts)
    params : dict
        System parameters dictionary
    measurement_case : str
        Measurement case identifier: 'case1', 'case2_x', 'case2_y', or 'case3'
        
    Returns:
    --------
    dict : Dictionary containing calculated signals
        u : Force (N)
        x : X-direction acceleration (m/s²)
        y : Y-direction acceleration (m/s²)
        theta_x : Angular displacement about X-axis (rad)
        theta_y : Angular displacement about Y-axis (rad)
        x_tool : X-direction tool acceleration (m/s²) - only for case2_x
        y_tool : Y-direction tool acceleration (m/s²) - only for case2_y
        z : Z-direction acceleration (m/s²) - only for case3
    """
    # Extract parameters
    L1 = params['L1']
    L2 = params['L2']
    L3 = params['L3']
    alpha = params['alpha']
    r_u = params['r_u']
    r_a = params['r_a']
    
    # Force input (convert to Newtons)
    u = A6 * r_u
    
    # Angular displacements
    theta_x = -(A0 + A1 - A2 - A3) / np.sin(alpha) / (2 * L3)
    theta_y = -(A0 - A1 - A2 + A3) / np.cos(alpha) / (2 * L3)
    
    # Initialize result dictionary
    result = {
        'u': u,
        'theta_x': theta_x,
        'theta_y': theta_y
    }
    

    # Case 1: Accelerometers only on z-slide (original equations)
    if measurement_case == 'case1':
        x = (-(A0 - A1 + A2 - A3) / np.cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2) - A4) * r_a
        y = ((A0 + A1 + A2 + A3) / np.sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2) - A5) * r_a
        result['x'] = x
        result['y'] = y
    
    # Case 2.1: Accelerometer at tool, X-direction
    elif measurement_case == 'case2_x':
        x = (-(A0 - A1 + A2 - A3) / np.cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2) - A4) * r_a
        y = ((A0 + A1 + A2 + A3) / np.sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2)) * r_a
        x_tool = (A7 * 5 - A4) * r_a
        result['x'] = x
        result['y'] = y
        result['x_tool'] = x_tool
    
    # Case 2.2: Accelerometer at tool, Y-direction
    elif measurement_case == 'case2_y':
        x = (-(A0 - A1 + A2 - A3) / np.cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2)) * r_a
        y = ((A0 + A1 + A2 + A3) / np.sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2) - A4) * r_a
        y_tool = (A5 - A4) * r_a
        result['x'] = x
        result['y'] = y
        result['y_tool'] = y_tool
    
    # Case 3: XYZ measurement case
    elif measurement_case == 'case3':
        # Convert angles from degrees to radians
        alpha1 = 57.1715 * np.pi / 180  # radians
        beta1 = 76.8368 * np.pi / 180   # radians
        beta2 = 85.2045 * np.pi / 180   # radians
        
        x = ((A3 - A2) / np.cos(alpha1) - A4) * r_a
        y = ((A2 + A3) / np.cos(beta1) - A5) * r_a
        # Note: y already has r_a applied, so divide by r_a before using in z calculation
        z = ((A0 - (y / r_a + A5) * np.cos(beta2)) / np.cos(np.pi / 2 - beta2)) * r_a
        result['x'] = x
        result['y'] = y
        result['z'] = z
    
    else:
        raise ValueError(f"Unknown measurement case: {measurement_case}")
    
    return result


def create_raised_cosine_window(time, force_signal, pre_seconds, post_seconds, beta, dt):
    """
    Build a raised cosine (Tukey) window around the dominant force peak.

    Parameters
    ----------
    time : array_like
        Time vector (seconds), assumed monotonically increasing.
    force_signal : array_like
        Force signal used to detect the peak event.
    pre_seconds : float
        Duration before the peak time where the window becomes non-zero.
    post_seconds : float
        Duration after the peak time where the window remains non-zero.
    beta : float
        Tukey window roll-off factor (0 <= beta <= 1).
    dt : float
        Sample time in seconds

    Returns
    -------
    np.ndarray
        Window array matching the length of `time` with values in [0, 1].
    dict
        Metadata including peak time and window start/end times.
    """
    time = np.asarray(time)
    force_signal = np.asarray(force_signal)

    if time.shape != force_signal.shape:
        raise ValueError("Time and force_signal must have the same shape for windowing")

    if time.ndim != 1:
        raise ValueError("Time array must be one-dimensional")

    if len(time) < 2:
        raise ValueError("Time array must contain at least two samples to create a window")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-positive or non-finite sample interval detected")

    peak_idx = int(np.argmax(np.abs(force_signal)))
    peak_time = time[peak_idx]

    pre_seconds = max(0.0, float(pre_seconds))
    post_seconds = max(0.0, float(post_seconds))
    beta = float(beta)
    beta = min(max(beta, 0.0), 1.0)

    start_time = peak_time - pre_seconds
    end_time = peak_time + post_seconds

    start_time = max(start_time, time[0])
    end_time = min(end_time, time[-1])

    if end_time <= start_time:
        raise ValueError("Window end time must be greater than start time")

    start_idx = int(np.searchsorted(time, start_time, side='left'))
    end_idx = int(np.searchsorted(time, end_time, side='right'))

    if end_idx <= start_idx + 1:
        window = np.zeros_like(time, dtype=float)
        window[peak_idx] = 1.0
        return window, {
            'peak_time': peak_time,
            'start_time': peak_time,
            'end_time': peak_time,
            'peak_index': peak_idx,
            'start_index': peak_idx,
            'end_index': peak_idx + 1
        }

    segment_length = end_idx - start_idx
    tukey_segment = signal.windows.tukey(segment_length, alpha=beta, sym=False)

    window = np.zeros_like(time, dtype=float)
    window[start_idx:end_idx] = tukey_segment

    metadata = {
        'peak_time': peak_time,
        'start_time': time[start_idx],
        'end_time': time[end_idx - 1],
        'peak_index': peak_idx,
        'start_index': start_idx,
        'end_index': end_idx
    }

    return window, metadata

def create_signal_based_window(sig, time, params, dt):
    """
    Build a raised cosine (Tukey) window based on signal-specific thresholds.
    
    The window is determined by:
    - Finding the first sample where |signal| >= trigger_percent * max(|signal|)
    - Setting peak time to rolloff_seconds before this sample
    - Finding the last window where 3*std > low_percent * max(|signal|) using sliding window
    - Setting end rolloff start to rolloff_seconds after this sample
    - Rolloff duration is rolloff_seconds on both sides
    
    Parameters
    ----------
    sig : array_like
        Input signal array used to determine window timing
    time : array_like
        Time vector (seconds), assumed monotonically increasing
    params : dict
        System parameters dictionary containing window configuration
    dt : float
        Sample time in seconds
    
    Returns
    -------
    np.ndarray
        Window array matching the length of `time` with values in [0, 1]
    dict
        Metadata including peak time, start/end times, and threshold indices
    """
    sig = np.asarray(sig)
    time = np.asarray(time)
    
    if time.shape != sig.shape:
        raise ValueError("Time and signal must have the same shape for windowing")
    
    if time.ndim != 1:
        raise ValueError("Time array must be one-dimensional")
    
    if len(time) < 2:
        raise ValueError("Time array must contain at least two samples to create a window")
    
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-positive or non-finite sample interval detected")
    
    # Extract parameters
    rolloff_seconds = params.get('window_rolloff_seconds', 0.005)
    trigger_percent = params.get('window_trigger_percent', 0.1)
    low_percent = params.get('window_low_percent', 0.02)
    window_sliding_size = params.get('window_sliding_size', 100)
    std_multiplier = params.get('window_std_multiplier', 3.0)
    
    # Calculate absolute signal and maximum
    abs_signal = np.abs(sig)
    max_abs_signal = np.max(abs_signal)
    
    # Handle edge case: signal is all zeros or has no valid data
    if max_abs_signal == 0 or not np.isfinite(max_abs_signal):
        # Return all-ones window as fallback
        window = np.ones_like(time, dtype=float)
        return window, {
            'peak_time': time[len(time) // 2],
            'start_time': time[0],
            'end_time': time[-1],
            'peak_index': len(time) // 2,
            'start_index': 0,
            'end_index': len(time),
            'first_trigger_percent_idx': len(time) // 2,
            'last_low_percent_idx': len(time) // 2
        }
    
    # Find first sample where |signal| >= trigger_percent * max(|signal|)
    threshold_trigger_percent = trigger_percent * max_abs_signal
    first_trigger_percent_mask = abs_signal >= threshold_trigger_percent
    first_trigger_percent_indices = np.where(first_trigger_percent_mask)[0]
    
    if len(first_trigger_percent_indices) == 0:
        # No sample reaches trigger threshold, use midpoint as fallback
        first_trigger_percent_idx = len(time) // 2
    else:
        first_trigger_percent_idx = first_trigger_percent_indices[0]
    
    # Find last window where 3*std > low_percent * max(|signal|) using sliding window
    # Use iterative threshold increase if no window is found
    current_threshold_percent = low_percent
    max_iterations = 40
    found_window = False
    
    # Handle case where signal is shorter than window size
    if len(abs_signal) < window_sliding_size:
        # Use smaller window or fallback
        actual_window_size = max(1, len(abs_signal) // 2)
        if actual_window_size < 2:
            # Signal too short, use last sample as fallback
            last_low_percent_idx = len(time) - 1
        else:
            # Try with iterative threshold increase
            for iteration in range(max_iterations):
                threshold_low_percent = current_threshold_percent * max_abs_signal
                # Use the smaller window
                window_data = abs_signal[-actual_window_size:]
                sigma = np.std(window_data)
                if std_multiplier * sigma > threshold_low_percent:
                    last_low_percent_idx = len(time) - actual_window_size
                    found_window = True
                    break
                else:
                    # Increase threshold by 0.01 for next iteration
                    current_threshold_percent += 0.001
            
            if not found_window:
                # Fallback: use last sample
                last_low_percent_idx = len(time) - 1
    else:
        # Slide window of size window_sliding_size over the signal
        for iteration in range(max_iterations):
            threshold_low_percent = current_threshold_percent * max_abs_signal
            last_valid_window_start = None

            for i in range(len(abs_signal) - window_sliding_size + 1):
                window_data = abs_signal[i:i + window_sliding_size]
                sigma = np.std(window_data)
                if std_multiplier * sigma > threshold_low_percent:
                    last_valid_window_start = i
            
            if (last_valid_window_start is not None) & (last_valid_window_start != len(abs_signal) - window_sliding_size) \
                        & (last_valid_window_start < int((window_sliding_size + len(abs_signal))/2)):
                last_low_percent_idx = last_valid_window_start
                found_window = True
                break
            else:
                # Increase threshold by 0.01 for next iteration
                current_threshold_percent += 0.01
        
        if not found_window:
            # Fallback: use last sample
            last_low_percent_idx = len(time) - 1
    
    # Calculate timing
    first_trigger_percent_time = time[first_trigger_percent_idx]
    last_low_percent_time = time[last_low_percent_idx]
    
    # Peak time is rolloff_seconds before the first 10% point
    peak_time = first_trigger_percent_time - rolloff_seconds
    
    # Window start is rolloff_seconds before peak (where rolloff begins)
    start_time = peak_time - rolloff_seconds
    
    # End rolloff starts rolloff_seconds after the last 1% point
    end_rolloff_start_time = last_low_percent_time + rolloff_seconds
    
    # Window end is rolloff_seconds after end rolloff start (where rolloff ends)
    end_time = end_rolloff_start_time + rolloff_seconds
    
    # Clamp times to valid range
    start_time = max(start_time, time[0])
    end_time = min(end_time, time[-1])
    
    # Ensure end_time > start_time
    if end_time <= start_time:
        # Fallback: use a small window around the peak
        peak_idx = int(np.argmax(abs_signal))
        peak_time = time[peak_idx]
        start_time = max(peak_time - rolloff_seconds, time[0])
        end_time = min(peak_time + rolloff_seconds, time[-1])
        if end_time <= start_time:
            end_time = min(start_time + 2 * rolloff_seconds, time[-1])
    
    # Find indices
    start_idx = int(np.searchsorted(time, start_time, side='left'))
    end_idx = int(np.searchsorted(time, end_time, side='right'))
    
    # Ensure valid indices
    start_idx = max(0, min(start_idx, len(time) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(time)))
    
    # Find peak index
    peak_idx = int(np.searchsorted(time, peak_time, side='left'))
    peak_idx = max(0, min(peak_idx, len(time) - 1))
    
    # Create window array
    window = np.zeros_like(time, dtype=float)
    
    # Check if window segment is long enough
    segment_length = end_idx - start_idx
    if segment_length < 2:
        # Window too short, set peak to 1.0
        window[peak_idx] = 1.0
        return window, {
            'peak_time': peak_time,
            'start_time': time[peak_idx],
            'end_time': time[peak_idx],
            'peak_index': peak_idx,
            'start_index': peak_idx,
            'end_index': peak_idx + 1,
            'first_trigger_percent_idx': first_trigger_percent_idx,
            'last_low_percent_idx': last_low_percent_idx
        }
    
    # Calculate rolloff duration in samples
    rolloff_samples = max(1, int(np.round(rolloff_seconds / dt)))
    
    # Calculate total window length in samples
    total_window_samples = end_idx - start_idx
    
    # Calculate alpha parameter for Tukey window
    # alpha = 2 * rolloff_samples / total_window_samples
    # This ensures rolloff regions are approximately rolloff_seconds long on both sides
    if total_window_samples > 2 * rolloff_samples:
        alpha = 2 * rolloff_samples / total_window_samples
        alpha = min(max(alpha, 0.0), 1.0)  # Clamp to [0, 1]
    else:
        # Window too short for proper rolloff, use small alpha
        alpha = 0.1
    
    # Create Tukey window segment with symmetric tapering (rolloff on both sides)
    tukey_segment = signal.windows.tukey(total_window_samples, alpha=alpha, sym=True)
    
    # Place window segment in the full window array
    window[start_idx:end_idx] = tukey_segment
    
    # Metadata
    metadata = {
        'peak_time': peak_time,
        'start_time': time[start_idx],
        'end_time': time[end_idx - 1],
        'peak_index': peak_idx,
        'start_index': start_idx,
        'end_index': end_idx,
        'first_trigger_percent_idx': first_trigger_percent_idx,
        'last_low_percent_idx': last_low_percent_idx
    }
    
    return window, metadata

def create_exponential_window(sig, time, params, dt):
    """
    Build an exponential window centered on the signal peak.
    
    The window is calculated as:
    w = exp(-abs(n - n_p) / tau)
    where:
    - n = sample index (0 to N-1)
    - n_p = sample position of peak of the signal
    - tau = N/2 * 8.69 / D
    - D = decay in dB (window_decay_db parameter)
    
    Parameters
    ----------
    sig : array_like
        Input signal array used to determine window timing
    time : array_like
        Time vector (seconds), assumed monotonically increasing
    params : dict
        System parameters dictionary containing window configuration
    dt : float
        Sample time in seconds (not used in exponential window, but kept for consistency)
    
    Returns
    -------
    np.ndarray
        Window array matching the length of `time` with values in [0, 1]
    dict
        Metadata including peak time and peak index
    """
    sig = np.asarray(sig)
    time = np.asarray(time)
    
    if time.shape != sig.shape:
        raise ValueError("Time and signal must have the same shape for windowing")
    
    if time.ndim != 1:
        raise ValueError("Time array must be one-dimensional")
    
    if len(time) < 1:
        raise ValueError("Time array must contain at least one sample to create a window")
    
    # Extract parameters
    decay_db = params.get('window_decay_db', 60.0)
    
    # Validate decay_db
    if not np.isfinite(decay_db) or decay_db <= 0:
        raise ValueError("window_decay_db must be a positive finite number")
    
    # Calculate absolute signal and find peak
    abs_signal = np.abs(sig)
    max_abs_signal = np.max(abs_signal)
    
    # Handle edge case: signal is all zeros or has no valid data
    if max_abs_signal == 0 or not np.isfinite(max_abs_signal):
        # Return all-ones window as fallback
        window = np.ones_like(time, dtype=float)
        peak_idx = len(time) // 2
        return window, {
            'peak_time': time[peak_idx] if len(time) > 0 else 0.0,
            'start_time': time[0] if len(time) > 0 else 0.0,
            'end_time': time[-1] if len(time) > 0 else 0.0,
            'peak_index': peak_idx,
            'start_index': 0,
            'end_index': len(time)
        }
    
    # Find peak index
    peak_idx = int(np.argmax(abs_signal))
    peak_time = time[peak_idx]
    
    # Calculate tau: tau = N/2 * 8.69 / D
    N = len(time)
    tau = (N / 2.0) * 8.69 / decay_db
    
    # Ensure tau is positive and finite
    if not np.isfinite(tau) or tau <= 0:
        # Fallback: use a reasonable default tau
        tau = N / 10.0
    
    # Create sample indices: n = 0 to N-1
    n = np.arange(N, dtype=float)
    
    # Calculate window: w = exp(-abs(n - n_p) / tau)
    window = np.exp(-np.abs(n - peak_idx) / tau)
    
    # Ensure window values are in [0, 1] range (should already be, but clamp to be safe)
    window = np.clip(window, 0.0, 1.0)
    
    # Metadata
    metadata = {
        'peak_time': peak_time,
        'start_time': time[0],
        'end_time': time[-1],
        'peak_index': peak_idx,
        'start_index': 0,
        'end_index': len(time),
        'tau': tau,
        'decay_db': decay_db
    }
    
    return window, metadata

def create_hann_window(sig, time, params, dt):
    """
    Build a Hann window centered on the signal peak.
    
    The window is calculated as:
    w[n] = 0.5 + 0.5 * cos(2*π*(n - n_peak)/(M-1))
    where:
    - n = sample index (0 to N-1)
    - n_peak = sample position of peak of the signal
    - M = window width in samples = int(window_hann_width / dt)
    - Window is zero outside the range [n_peak - M//2, n_peak + M//2]
    
    Parameters
    ----------
    sig : array_like
        Input signal array used to determine window timing
    time : array_like
        Time vector (seconds), assumed monotonically increasing
    params : dict
        System parameters dictionary containing window configuration
    dt : float
        Sample time in seconds
    
    Returns
    -------
    np.ndarray
        Window array matching the length of `time` with values in [0, 1]
    dict
        Metadata including peak time, peak index, and window start/end indices
    """
    sig = np.asarray(sig)
    time = np.asarray(time)
    
    if time.shape != sig.shape:
        raise ValueError("Time and signal must have the same shape for windowing")
    
    if time.ndim != 1:
        raise ValueError("Time array must be one-dimensional")
    
    if len(time) < 1:
        raise ValueError("Time array must contain at least one sample to create a window")
    
    # Extract parameters
    window_hann_width = params.get('window_hann_width', 0.1)
    
    # Validate window_hann_width
    if not np.isfinite(window_hann_width) or window_hann_width <= 0:
        raise ValueError("window_hann_width must be a positive finite number")
    
    # Calculate absolute signal and find peak
    abs_signal = np.abs(sig)
    max_abs_signal = np.max(abs_signal)
    
    # Handle edge case: signal is all zeros or has no valid data
    if max_abs_signal == 0 or not np.isfinite(max_abs_signal):
        # Return all-ones window as fallback
        window = np.ones_like(time, dtype=float)
        peak_idx = len(time) // 2
        return window, {
            'peak_time': time[peak_idx] if len(time) > 0 else 0.0,
            'start_time': time[0] if len(time) > 0 else 0.0,
            'end_time': time[-1] if len(time) > 0 else 0.0,
            'peak_index': peak_idx,
            'start_index': 0,
            'end_index': len(time),
            'window_width_samples': len(time),
            'window_width_seconds': time[-1] - time[0] if len(time) > 1 else 0.0
        }
    
    # Find peak index
    peak_idx = int(np.argmax(abs_signal))
    peak_time = time[peak_idx]
    
    # Calculate window width in samples: M = int(window_hann_width / dt)
    M = int(window_hann_width / dt)
    
    # Ensure M is at least 2 (need at least 2 samples for the formula)
    if M < 2:
        M = 2
    
    # Calculate window range: [n_peak - M//2, n_peak + M//2]
    start_idx = peak_idx - M // 2
    end_idx = peak_idx + M // 2 + 1  # +1 because end_idx is exclusive
    
    # Clamp to signal bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(time), end_idx)
    
    # Ensure we have at least 2 samples
    if end_idx <= start_idx + 1:
        # Fallback: use a small window around the peak
        start_idx = max(0, peak_idx - 1)
        end_idx = min(len(time), peak_idx + 2)
        if end_idx <= start_idx + 1:
            end_idx = start_idx + 2
        M = end_idx - start_idx
    
    # Create window array (all zeros initially)
    window = np.zeros_like(time, dtype=float)
    
    # Create sample indices for the window region
    n = np.arange(start_idx, end_idx, dtype=float)
    
    # Calculate Hann window: w = 0.5 + 0.5 * cos(2*pi*(n - n_peak)/(M-1))
    # Note: M-1 in denominator ensures the window goes from 0 to 1 over M samples
    window[start_idx:end_idx] = 0.5 + 0.5 * np.cos(2 * np.pi * (n - peak_idx) / (M - 1))
    
    # Ensure window values are in [0, 1] range (should already be, but clamp to be safe)
    window = np.clip(window, 0.0, 1.0)
    
    # Metadata
    metadata = {
        'peak_time': peak_time,
        'start_time': time[start_idx],
        'end_time': time[end_idx - 1] if end_idx > start_idx else time[start_idx],
        'peak_index': peak_idx,
        'start_index': start_idx,
        'end_index': end_idx,
        'window_width_samples': M,
        'window_width_seconds': window_hann_width
    }
    
    return window, metadata

def create_constant_window(sig, time, params, dt):
    """
    Build a constant window (all ones, no windowing applied).
    
    Parameters
    ----------
    sig : array_like
        Input signal array (not used, but kept for consistency with other window functions)
    time : array_like
        Time vector (seconds), assumed monotonically increasing
    params : dict
        System parameters dictionary (not used, but kept for consistency)
    dt : float
        Sample time in seconds (not used, but kept for consistency)
    
    Returns
    -------
    np.ndarray
        Window array matching the length of `time` with all values set to 1.0
    dict
        Metadata including start/end times
    """
    time = np.asarray(time)
    
    if time.ndim != 1:
        raise ValueError("Time array must be one-dimensional")
    
    if len(time) < 1:
        raise ValueError("Time array must contain at least one sample to create a window")
    
    # Create window array of all ones
    window = np.ones_like(time, dtype=float)
    
    # Metadata
    metadata = {
        'start_time': time[0] if len(time) > 0 else 0.0,
        'end_time': time[-1] if len(time) > 0 else 0.0,
        'start_index': 0,
        'end_index': len(time)
    }
    
    return window, metadata

def create_window(sig, time, params, dt, window_type=None):
    """
    Create a window based on the selected window type.
    
    This function acts as a dispatcher that calls the appropriate window creation
    function based on the window_type parameter.
    
    Parameters
    ----------
    sig : array_like
        Input signal array used to determine window timing
    time : array_like
        Time vector (seconds), assumed monotonically increasing
    params : dict
        System parameters dictionary containing window configuration
    dt : float
        Sample time in seconds
    window_type : str, optional
        Window type to use. If None, uses params.get('window_type', 'exponential') for backward compatibility.
        Valid options: 'hann', 'exponential', 'raised-cosine', 'signal-based', 'constant'
    
    Returns
    -------
    np.ndarray
        Window array matching the length of `time` with values in [0, 1]
    dict
        Metadata dictionary with window information
    """
    # If window_type is not provided, use legacy parameter for backward compatibility
    if window_type is None:
        window_type = params.get('window_type', 'exponential')
    
    if window_type == 'raised-cosine':
        # Use raised cosine window
        pre_seconds = params.get('window_pre_seconds', 0.05)
        post_seconds = params.get('window_post_seconds', 0.1)
        beta = params.get('window_beta', 0.25)
        return create_raised_cosine_window(time, sig, pre_seconds, post_seconds, beta, dt)
    
    elif window_type == 'signal-based':
        # Use signal-based window
        return create_signal_based_window(sig, time, params, dt)
    
    elif window_type == 'exponential':
        # Use exponential window
        return create_exponential_window(sig, time, params, dt)
    
    elif window_type == 'hann':
        # Use Hann window
        return create_hann_window(sig, time, params, dt)
    
    elif window_type == 'constant':
        # Use constant window (all ones)
        return create_constant_window(sig, time, params, dt)
    
    else:
        raise ValueError(f"Unknown window type: {window_type}. Must be 'hann', 'exponential', 'raised-cosine', 'signal-based', or 'constant'")

# =============================================================================
# FREQUENCY DOMAIN ANALYSIS
# =============================================================================

def perform_pyfrf_analysis(time, u, x, y, dt, params, tool_signal=None):
    """
    Perform FRF analysis using pyFRF library.
    Uses H₂ estimator: H₂(ω) = G_output_output(ω) / G_output_input(ω).
    Note: x and y are accelerations and will be double-integrated to displacements.
    Also returns acceleration-based transfer functions for comparison.
    Optionally processes tool acceleration signal if provided.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u : array_like
        Force signal (time domain) - input
    x, y : array_like
        Acceleration signals in time domain (m/s²) - outputs
    dt : float
        Sample time in seconds
    params : dict
        System parameters dictionary containing Welch method parameters
    tool_signal : array_like, optional
        Tool acceleration signal in time domain (m/s²) - x_tool or y_tool
        
    Returns:
    --------
    dict : Dictionary containing frequency array and transfer function results
        Includes both displacement (X, Y) and acceleration (X_accel, Y_accel) results
        Optionally includes Tool and Tool_accel if tool_signal is provided
    """
    N = len(time)
    fs = 1 / dt  # Sampling frequency
    fs_int = int(round(fs))  # Convert to int for pyFRF
    
    # Extract Welch method parameters
    nperseg = params.get('welch_nperseg', None)
    noverlap = params.get('welch_noverlap', None)
    window = params.get('welch_window', 'hann')
    estimator_type = params.get('estimator_type', 'H2')
    
    # Create FRF instance for x response (acceleration)
    frf_x = FRF(
        sampling_freq=fs_int,
        exc=u,
        resp=x,
        exc_type='f',  # force
        resp_type='a',  # acceleration
        frf_type=estimator_type,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window
    )
    
    # Create FRF instance for y response (acceleration)
    frf_y = FRF(
        sampling_freq=fs_int,
        exc=u,
        resp=y,
        exc_type='f',  # force
        resp_type='a',  # acceleration
        frf_type=estimator_type,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window
    )
    
    # Get frequency axis
    frequencies = frf_x.get_f_axis()
    
    # Get acceleration transfer functions (accelerance: acceleration/force)
    # get_FRF() returns shape (resp_dofs, exc_dofs, frequency_series) for 3D or (resp_dofs, frequency_series) for 2D
    Gx_accel_full = frf_x.get_FRF(form='accelerance')
    Gy_accel_full = frf_y.get_FRF(form='accelerance')
    
    # Get displacement transfer functions (receptance: displacement/force)
    Gx_disp_full = frf_x.get_FRF(form='receptance')
    Gy_disp_full = frf_y.get_FRF(form='receptance')
    
    # Extract 1D arrays for single input/output - acceleration
    if Gx_accel_full.ndim == 3:
        Gx_accel = Gx_accel_full[0, 0, :]  # Extract (frequency_series,)
    elif Gx_accel_full.ndim == 2:
        Gx_accel = Gx_accel_full[0, :]  # Extract (frequency_series,)
    else:
        Gx_accel = Gx_accel_full  # Already 1D
    
    if Gy_accel_full.ndim == 3:
        Gy_accel = Gy_accel_full[0, 0, :]  # Extract (frequency_series,)
    elif Gy_accel_full.ndim == 2:
        Gy_accel = Gy_accel_full[0, :]  # Extract (frequency_series,)
    else:
        Gy_accel = Gy_accel_full  # Already 1D
    
    # Extract 1D arrays for single input/output - displacement
    if Gx_disp_full.ndim == 3:
        Gx = Gx_disp_full[0, 0, :]  # Extract (frequency_series,)
    elif Gx_disp_full.ndim == 2:
        Gx = Gx_disp_full[0, :]  # Extract (frequency_series,)
    else:
        Gx = Gx_disp_full  # Already 1D
    
    if Gy_disp_full.ndim == 3:
        Gy = Gy_disp_full[0, 0, :]  # Extract (frequency_series,)
    elif Gy_disp_full.ndim == 2:
        Gy = Gy_disp_full[0, :]  # Extract (frequency_series,)
    else:
        Gy = Gy_disp_full  # Already 1D
    
    # Replace NaN values with 0 for displacement FRFs
    Gx = np.nan_to_num(Gx, nan=0.0)
    Gy = np.nan_to_num(Gy, nan=0.0)
    
    # Replace NaN values with 0 for acceleration FRFs
    Gx_accel = np.nan_to_num(Gx_accel, nan=0.0)
    Gy_accel = np.nan_to_num(Gy_accel, nan=0.0)
    
    # Get coherence (returns shape (resp_dofs, frequency_series) for 2D or 1D)
    coherence_ux_full = frf_x.get_coherence()
    coherence_uy_full = frf_y.get_coherence()
    
    if coherence_ux_full.ndim == 2:
        coherence_ux = coherence_ux_full[0, :]  # Extract (frequency_series,)
    else:
        coherence_ux = coherence_ux_full  # Already 1D
    
    if coherence_uy_full.ndim == 2:
        coherence_uy = coherence_uy_full[0, :]  # Extract (frequency_series,)
    else:
        coherence_uy = coherence_uy_full  # Already 1D
    
    # Get PSD of input force (Guu) for U calculation
    # pyFRF stores this internally, but we need to extract it
    # For now, we'll calculate it from the FRF and coherence
    # Actually, we can get it from the internal CSD/PSD if available
    # Let's use scipy.welch for Guu to maintain compatibility
    _, Guu = signal.welch(u, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                         window=window, scaling='density', return_onesided=True)
    
    # Process tool signal if provided
    if tool_signal is not None:
        # Create FRF instance for tool response (acceleration)
        frf_tool = FRF(
            sampling_freq=fs_int,
            exc=u,
            resp=tool_signal,
            exc_type='f',  # force
            resp_type='a',  # acceleration
            frf_type=estimator_type,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )
        
        frequencies_tool = frf_tool.get_f_axis()
        G_tool_accel_full = frf_tool.get_FRF(form='accelerance')
        G_tool_disp_full = frf_tool.get_FRF(form='receptance')
        coherence_tool_full = frf_tool.get_coherence()
        
        # Extract 1D array for single input/output - acceleration
        if G_tool_accel_full.ndim == 3:
            G_tool_accel = G_tool_accel_full[0, 0, :]  # Extract (frequency_series,)
        elif G_tool_accel_full.ndim == 2:
            G_tool_accel = G_tool_accel_full[0, :]  # Extract (frequency_series,)
        else:
            G_tool_accel = G_tool_accel_full  # Already 1D
        
        # Extract 1D array for single input/output - displacement
        if G_tool_disp_full.ndim == 3:
            Tool = G_tool_disp_full[0, 0, :]  # Extract (frequency_series,)
        elif G_tool_disp_full.ndim == 2:
            Tool = G_tool_disp_full[0, :]  # Extract (frequency_series,)
        else:
            Tool = G_tool_disp_full  # Already 1D
        
        if coherence_tool_full.ndim == 2:
            coherence_tool = coherence_tool_full[0, :]  # Extract (frequency_series,)
        else:
            coherence_tool = coherence_tool_full  # Already 1D
        
        # Replace NaN values with 0 for tool FRFs
        Tool = np.nan_to_num(Tool, nan=0.0)
        G_tool_accel = np.nan_to_num(G_tool_accel, nan=0.0)
    else:
        Tool_accel = None
        Tool = None
        coherence_tool = None
        frequencies_tool = None
    
    # Create U array for compatibility with plot_force_psd
    # plot_force_psd calculates: PSD = (2 / (N * fs)) * |U|²
    # For Welch, we have Guu which is already PSD, so: U = sqrt(Guu * N * fs / 2)
    # Interpolate Guu to match frequencies if needed
    if len(Guu) == len(frequencies):
        Guu_aligned = Guu
    else:
        from scipy.interpolate import interp1d
        freq_guu = np.fft.rfftfreq(N, dt)[:len(Guu)]
        valid_guu = np.isfinite(Guu) & (Guu > 0)
        if np.any(valid_guu):
            interp_func = interp1d(freq_guu[valid_guu], Guu[valid_guu], 
                                 kind='linear', bounds_error=False, fill_value=np.nan)
            Guu_aligned = interp_func(frequencies)
        else:
            Guu_aligned = np.full_like(frequencies, np.nan)
    
    valid_mask_guu = np.isfinite(Guu_aligned) & (Guu_aligned > 0)
    U_pos = np.zeros_like(frequencies, dtype=complex)
    if np.any(valid_mask_guu):
        # Calculate U magnitude from PSD (Guu)
        U_magnitude = np.sqrt(Guu_aligned[valid_mask_guu] * N * fs / 2)
        U_pos[valid_mask_guu] = U_magnitude.astype(complex)
    
    # Return results with same structure as perform_welch_csd_analysis
    result = {
        'frequencies': frequencies,
        'U': U_pos,  # U from PSD
        'X': Gx,  # Displacement transfer function
        'Y': Gy,  # Displacement transfer function
        'X_accel': Gx_accel,  # Acceleration transfer function
        'Y_accel': Gy_accel,  # Acceleration transfer function
        'X_full': None,  # Not used - skipping time signal reconstruction
        'Y_full': None,  # Not used - skipping time signal reconstruction
        'coherence_ux': coherence_ux,  # Coherence between input u and output x
        'coherence_uy': coherence_uy,  # Coherence between input u and output y
        'N': N,
        'dt': dt
    }
    
    # Add tool results if available
    if tool_signal is not None:
        result['Tool'] = Tool
        result['Tool_accel'] = G_tool_accel
        result['Tool_full'] = None  # Not used - skipping time signal reconstruction
        result['coherence_tool'] = coherence_tool  # Coherence between input u and tool output
    
    return result

def perform_welch_csd_analysis(time, u, x, y, dt, params, tool_signal=None):
    """
    Perform Welch method with cross-spectral density (CSD) for transfer function estimation.
    Uses H₂ estimator: H₂(ω) = G_output_output(ω) / G_output_input(ω).
    Note: x and y are accelerations and will be double-integrated to displacements.
    Also returns acceleration-based transfer functions for comparison.
    Optionally processes tool acceleration signal if provided.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u : array_like
        Force signal (time domain) - input
    x, y : array_like
        Acceleration signals in time domain (m/s²) - outputs
    dt : float
        Sample time in seconds
    params : dict
        System parameters dictionary containing Welch method parameters
    tool_signal : array_like, optional
        Tool acceleration signal in time domain (m/s²) - x_tool or y_tool
        
    Returns:
    --------
    dict : Dictionary containing frequency array and transfer function results
        Includes both displacement (X, Y) and acceleration (X_accel, Y_accel) results
        Optionally includes Tool and Tool_accel if tool_signal is provided
    """
    N = len(time)
    fs = 1 / dt  # Sampling frequency
    
    # Extract Welch method parameters
    nperseg = params.get('welch_nperseg', None)
    noverlap = params.get('welch_noverlap', None)
    window = params.get('welch_window', 'hann')
    scaling = params.get('welch_scaling', 'density')
    
    # Calculate cross-spectral densities and power spectral densities
    # For acceleration signals (x, y are accelerations)
    frequencies, Gxu_accel = signal.csd(u, x, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                                       window=window, scaling=scaling, return_onesided=True)
    frequencies, Gyu_accel = signal.csd(u, y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                                       window=window, scaling=scaling, return_onesided=True)
    frequencies, Guu = signal.welch(u, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                                    window=window, scaling=scaling, return_onesided=True)
    frequencies, Gxx_accel = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                                          window=window, scaling=scaling, return_onesided=True)
    frequencies, Gyy_accel = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                                          window=window, scaling=scaling, return_onesided=True)
    
    # Calculate H₂ estimator for acceleration: H₂ = G_output_output / G_output_input
    # For Gx: H₂ = Gxx / Gxu (x is output, u is input)
    # For Gy: H₂ = Gyy / Gyu (y is output, u is input)
    threshold = 1e-12
    Gx_accel = np.zeros_like(Gxu_accel, dtype=complex)
    Gy_accel = np.zeros_like(Gyu_accel, dtype=complex)
    
    # Avoid division by zero
    Gxu_mag = np.abs(Gxu_accel)
    Gyu_mag = np.abs(Gyu_accel)
    valid_mask_x = Gxu_mag > threshold
    valid_mask_y = Gyu_mag > threshold
    
    # Calculate H₂ estimator
    Gx_accel[valid_mask_x] = Gxx_accel[valid_mask_x] / Gxu_accel[valid_mask_x]
    Gy_accel[valid_mask_y] = Gyy_accel[valid_mask_y] / Gyu_accel[valid_mask_y]
    
    # Set invalid points to NaN
    Gx_accel[~valid_mask_x] = np.nan
    Gy_accel[~valid_mask_y] = np.nan
    
    # Calculate coherence: γ²(ω) = |Gxy|² / (Gxx · Gyy)
    # For input-output coherence: γ²_ux = |Gxu|² / (Guu · Gxx)
    #                             γ²_uy = |Gyu|² / (Guu · Gyy)
    coherence_ux = np.zeros_like(frequencies, dtype=float)
    coherence_uy = np.zeros_like(frequencies, dtype=float)
    
    # Calculate denominator: Guu · Gxx_accel and Guu · Gyy_accel
    denominator_ux = Guu * Gxx_accel
    denominator_uy = Guu * Gyy_accel
    
    # Calculate |Gxu_accel|² and |Gyu_accel|²
    Gxu_mag_squared = np.abs(Gxu_accel)**2
    Gyu_mag_squared = np.abs(Gyu_accel)**2
    
    # Calculate coherence with threshold to avoid division by zero
    valid_mask_coherence_ux = denominator_ux > threshold
    valid_mask_coherence_uy = denominator_uy > threshold
    
    coherence_ux[valid_mask_coherence_ux] = Gxu_mag_squared[valid_mask_coherence_ux] / denominator_ux[valid_mask_coherence_ux]
    coherence_uy[valid_mask_coherence_uy] = Gyu_mag_squared[valid_mask_coherence_uy] / denominator_uy[valid_mask_coherence_uy]
    
    # Set invalid points to NaN
    coherence_ux[~valid_mask_coherence_ux] = np.nan
    coherence_uy[~valid_mask_coherence_uy] = np.nan
    
    # Clamp coherence values to [0, 1] range (should already be in this range, but ensure it)
    coherence_ux = np.clip(coherence_ux, 0.0, 1.0)
    coherence_uy = np.clip(coherence_uy, 0.0, 1.0)
    
    # Convert acceleration transfer functions to displacement transfer functions
    # Apply double integration transfer function: H(e^(jω)) = dt²/(1-e^(-jω))²
    omega = 2 * np.pi * frequencies
    z = np.exp(1j * omega * dt)
    H = dt**2 / (1 - z**(-1))**2
    
    # Avoid division by zero at DC (frequency = 0)
    Gx = np.zeros_like(Gx_accel, dtype=complex)
    Gy = np.zeros_like(Gy_accel, dtype=complex)
    Gx[1:] = Gx_accel[1:] * H[1:]
    Gy[1:] = Gy_accel[1:] * H[1:]
    Gx[0] = 0
    Gy[0] = 0
    
    # Process tool signal if provided
    if tool_signal is not None:
        frequencies_tool, G_tool_u_accel = signal.csd(u, tool_signal, fs=fs, nperseg=nperseg, 
                                                      noverlap=noverlap, window=window, 
                                                      scaling=scaling, return_onesided=True)
        frequencies_tool, G_tool_tool_accel = signal.welch(tool_signal, fs=fs, nperseg=nperseg, 
                                                           noverlap=noverlap, window=window, 
                                                           scaling=scaling, return_onesided=True)
        
        # Calculate H₂ estimator for tool acceleration
        G_tool_u_mag = np.abs(G_tool_u_accel)
        valid_mask_tool = G_tool_u_mag > threshold
        G_tool_accel = np.zeros_like(G_tool_u_accel, dtype=complex)
        G_tool_accel[valid_mask_tool] = G_tool_tool_accel[valid_mask_tool] / G_tool_u_accel[valid_mask_tool]
        G_tool_accel[~valid_mask_tool] = np.nan
        
        # Convert to displacement using tool frequencies
        omega_tool = 2 * np.pi * frequencies_tool
        z_tool = np.exp(1j * omega_tool * dt)
        H_tool = dt**2 / (1 - z_tool**(-1))**2
        Tool_accel = G_tool_accel
        Tool = np.zeros_like(Tool_accel, dtype=complex)
        Tool[1:] = Tool_accel[1:] * H_tool[1:]
        Tool[0] = 0
        
        # Calculate tool coherence: γ²_tool = |G_tool_u_accel|² / (Guu · G_tool_tool_accel)
        # Need to interpolate Guu to tool frequencies if they differ
        if len(frequencies_tool) == len(frequencies):
            # Same frequency array, use directly
            Guu_tool = Guu
        else:
            # Interpolate Guu to tool frequencies
            from scipy.interpolate import interp1d
            valid_guu = np.isfinite(Guu) & (Guu > 0)
            if np.any(valid_guu):
                interp_func = interp1d(frequencies[valid_guu], Guu[valid_guu], 
                                     kind='linear', bounds_error=False, fill_value=np.nan)
                Guu_tool = interp_func(frequencies_tool)
            else:
                Guu_tool = np.full_like(frequencies_tool, np.nan)
        
        coherence_tool = np.zeros_like(frequencies_tool, dtype=float)
        denominator_tool = Guu_tool * G_tool_tool_accel
        G_tool_u_mag_squared = np.abs(G_tool_u_accel)**2
        valid_mask_coherence_tool = denominator_tool > threshold
        
        coherence_tool[valid_mask_coherence_tool] = G_tool_u_mag_squared[valid_mask_coherence_tool] / denominator_tool[valid_mask_coherence_tool]
        coherence_tool[~valid_mask_coherence_tool] = np.nan
        coherence_tool = np.clip(coherence_tool, 0.0, 1.0)
    else:
        Tool_accel = None
        Tool = None
        coherence_tool = None
    
    # Create U array for compatibility with plot_force_psd
    # plot_force_psd calculates: PSD = (2 / (N * fs)) * |U|²
    # For Welch, we have Guu which is already PSD, so: U = sqrt(Guu * N * fs / 2)
    valid_mask_guu = np.isfinite(Guu) & (Guu > 0)
    U_pos = np.zeros_like(frequencies, dtype=complex)
    if np.any(valid_mask_guu):
        # Calculate U magnitude from PSD (Guu)
        U_magnitude = np.sqrt(Guu[valid_mask_guu] * N * fs / 2)
        U_pos[valid_mask_guu] = U_magnitude.astype(complex)
    
    # Return results directly using Welch frequencies (no interpolation)
    # Skip full spectrum arrays since we're not reconstructing time signals
    result = {
        'frequencies': frequencies,
        'U': U_pos,  # U from PSD
        'X': Gx,  # Displacement transfer function
        'Y': Gy,  # Displacement transfer function
        'X_accel': Gx_accel,  # Acceleration transfer function
        'Y_accel': Gy_accel,  # Acceleration transfer function
        'X_full': None,  # Not used - skipping time signal reconstruction
        'Y_full': None,  # Not used - skipping time signal reconstruction
        'coherence_ux': coherence_ux,  # Coherence between input u and output x
        'coherence_uy': coherence_uy,  # Coherence between input u and output y
        'N': N,
        'dt': dt
    }
    
    # Add tool results if available
    if tool_signal is not None:
        result['Tool'] = Tool
        result['Tool_accel'] = Tool_accel
        result['Tool_full'] = None  # Not used - skipping time signal reconstruction
        result['coherence_tool'] = coherence_tool  # Coherence between input u and tool output
    
    return result

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_csd_psd_before_averaging(csd_psd_list, filepaths, averaged_csd_psd=None, show_plot=True):
    """
    Plot all CSD/PSD terms from individual files, optionally with averaged terms overlaid.
    Creates separate loglog subplots for each term, showing all files as different lines.
    If averaged_csd_psd is provided, plots averaged terms as a black line on top.
    
    Parameters:
    -----------
    csd_psd_list : list
        List of CSD/PSD dictionaries from calculate_csd_psd_terms()
    filepaths : list
        List of file paths for legend labels
    averaged_csd_psd : dict, optional
        Averaged CSD/PSD dictionary from average_csd_psd_terms()
        If provided, will be plotted as a black line on each subplot
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    if not csd_psd_list:
        return
    
    # Check if any file has tool signals
    has_tool = any('G_tool_u_accel' in csd_psd for csd_psd in csd_psd_list)
    
    # Define terms to plot (always present)
    terms_to_plot = [
        ('Gxu_accel', '|Gxu_accel|', 'Cross-spectral density u-x (magnitude)'),
        ('Gyu_accel', '|Gyu_accel|', 'Cross-spectral density u-y (magnitude)'),
        ('Guu', 'Guu', 'Power spectral density of u'),
        ('Gxx_accel', 'Gxx_accel', 'Power spectral density of x'),
        ('Gyy_accel', 'Gyy_accel', 'Power spectral density of y'),
    ]
    
    # Add tool terms if present
    if has_tool:
        terms_to_plot.extend([
            ('G_tool_u_accel', '|G_tool_u_accel|', 'Cross-spectral density u-tool (magnitude)'),
            ('G_tool_tool_accel', 'G_tool_tool_accel', 'Power spectral density of tool'),
        ])
    
    # Determine subplot layout
    n_terms = len(terms_to_plot)
    if n_terms <= 4:
        n_rows, n_cols = 2, 2
    elif n_terms <= 6:
        n_rows, n_cols = 2, 3
    elif n_terms <= 9:
        n_rows, n_cols = 3, 3
    else:
        n_rows, n_cols = 3, 4
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_terms == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get file basenames for legend
    file_basenames = [os.path.basename(fp) for fp in filepaths]
    
    # Plot each term
    for idx, (term_key, term_label, term_title) in enumerate(terms_to_plot):
        ax = axes[idx]
        
        # Plot each file
        for file_idx, (csd_psd, basename) in enumerate(zip(csd_psd_list, file_basenames)):
            # Check if this term exists in this file
            if term_key not in csd_psd:
                continue
            
            # Get data and frequencies
            if term_key in ['G_tool_u_accel', 'G_tool_tool_accel']:
                # Tool terms use frequencies_tool
                if 'frequencies_tool' not in csd_psd:
                    continue
                frequencies = csd_psd['frequencies_tool']
            else:
                frequencies = csd_psd['frequencies']
            
            data = csd_psd[term_key]
            
            # Handle complex vs real values
            if np.iscomplexobj(data):
                # Complex: plot magnitude
                magnitude = np.abs(data)
            else:
                # Real: plot directly
                magnitude = data
            
            # Remove DC component and invalid values
            valid_mask = np.isfinite(magnitude) & (magnitude > 0)
            if len(frequencies) > 1 and frequencies[0] == 0:
                # Skip DC component
                valid_mask = valid_mask & (frequencies > 0)
            
            valid_freq = frequencies[valid_mask]
            valid_mag = magnitude[valid_mask]
            
            if len(valid_freq) > 0:
                ax.loglog(valid_freq, valid_mag, linewidth=1.5, alpha=0.7, label=basename)
        
        # Plot averaged term as black line on top if provided
        if averaged_csd_psd is not None and term_key in averaged_csd_psd:
            # Get data and frequencies for averaged term
            if term_key in ['G_tool_u_accel', 'G_tool_tool_accel']:
                # Tool terms use frequencies_tool
                if 'frequencies_tool' in averaged_csd_psd:
                    frequencies_avg = averaged_csd_psd['frequencies_tool']
                    data_avg = averaged_csd_psd[term_key]
                else:
                    frequencies_avg = None  # Skip if tool frequencies not available
            else:
                frequencies_avg = averaged_csd_psd['frequencies']
                data_avg = averaged_csd_psd[term_key]
            
            # Only plot if we have valid frequencies
            if frequencies_avg is not None:
                # Handle complex vs real values
                if np.iscomplexobj(data_avg):
                    # Complex: plot magnitude
                    magnitude_avg = np.abs(data_avg)
                else:
                    # Real: plot directly
                    magnitude_avg = data_avg
                
                # Remove DC component and invalid values
                valid_mask_avg = np.isfinite(magnitude_avg) & (magnitude_avg > 0)
                if len(frequencies_avg) > 1 and frequencies_avg[0] == 0:
                    # Skip DC component
                    valid_mask_avg = valid_mask_avg & (frequencies_avg > 0)
                
                valid_freq_avg = frequencies_avg[valid_mask_avg]
                valid_mag_avg = magnitude_avg[valid_mask_avg]
                
                if len(valid_freq_avg) > 0:
                    ax.loglog(valid_freq_avg, valid_mag_avg, 'k-', linewidth=2, label='Average', zorder=10)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(term_label)
        ax.set_title(term_title)
        ax.grid(True, which="both", ls="-", alpha=0.3)
        
        # Add legend if there are multiple files or averaged terms
        if len(filepaths) > 1 or averaged_csd_psd is not None:
            ax.legend(fontsize=8, loc='best')
    
    # Hide unused subplots
    for idx in range(n_terms, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot=True):
    """
    Plot comparison of original, bias-corrected, and filtered signals for A0-A7.
    Creates a single figure with 2 rows and 4 columns (8 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    original_signals : dict
        Dictionary with keys 'A0' through 'A7' containing original signals
    corrected_signals : dict
        Dictionary with keys 'A0' through 'A7' containing bias-corrected signals
    filtered_signals : dict
        Dictionary with keys 'A0' through 'A7' containing filtered signals
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    signal_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    
    # Create figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    for idx, name in enumerate(signal_names):
        ax = axes[idx]
        
        ax.plot(time, original_signals[name], 'b-', linewidth=1, alpha=0.7, label='Original')
        ax.plot(time, corrected_signals[name], 'r-', linewidth=1, alpha=0.7, label='Bias Corrected')
        ax.plot(time, filtered_signals[name], 'g-', linewidth=1, alpha=0.7, label='Filtered')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Signal Comparison - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_lowpass_filtered_signals(time, bias_corrected_signals, lowpass_filtered_signals, bandwidth_hz, show_plot=True):
    """
    Plot comparison of bias-corrected and low-pass filtered signals for A0-A7.
    Creates a single figure with 2 rows and 4 columns (8 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    bias_corrected_signals : dict
        Dictionary with keys 'A0' through 'A7' containing bias-corrected signals
    lowpass_filtered_signals : dict
        Dictionary with keys 'A0' through 'A7' containing low-pass filtered signals
    bandwidth_hz : float
        Low-pass filter bandwidth in Hz (for display in labels)
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    signal_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    
    # Create figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    for idx, name in enumerate(signal_names):
        ax = axes[idx]
        
        ax.plot(time, bias_corrected_signals[name], 'r-', linewidth=1, alpha=0.7, label='Bias Corrected')
        ax.plot(time, lowpass_filtered_signals[name], 'b-', linewidth=1, alpha=0.7, label=f'Low-pass Filtered ({bandwidth_hz:.1f} Hz)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Low-Pass Filtered Signal Comparison - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_wiener_filtered_signals(time, bias_corrected_signals, wiener_filtered_signals, show_plot=True):
    """
    Plot comparison of bias-corrected and Wiener filtered signals for A0-A7.
    Creates a single figure with 2 rows and 4 columns (8 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    bias_corrected_signals : dict
        Dictionary with keys 'A0' through 'A7' containing bias-corrected signals
    wiener_filtered_signals : dict
        Dictionary with keys 'A0' through 'A7' containing Wiener filtered signals
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    signal_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    
    # Create figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    for idx, name in enumerate(signal_names):
        ax = axes[idx]
        
        ax.plot(time, bias_corrected_signals[name], 'r-', linewidth=1, alpha=0.7, label='Bias Corrected')
        ax.plot(time, wiener_filtered_signals[name], 'g-', linewidth=1, alpha=0.7, label='Wiener Filtered')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Wiener Filtered Signal Comparison - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_savgol_filtered_signals(time, raw_signals, savgol_filtered_signals, window_length, polyorder, show_plot=True):
    """
    Plot comparison of raw signals (after Wiener filter) and Savitzky-Golay filtered signals for A0-A7.
    Creates a single figure with 2 rows and 4 columns (8 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    raw_signals : dict
        Dictionary with keys 'A0' through 'A7' containing signals after Wiener filter (before Savitzky-Golay)
    savgol_filtered_signals : dict
        Dictionary with keys 'A0' through 'A7' containing Savitzky-Golay filtered signals
    window_length : int
        Window length used for Savitzky-Golay filter (for display in labels)
    polyorder : int
        Polynomial order used for Savitzky-Golay filter (for display in labels)
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    signal_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    
    # Create figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    for idx, name in enumerate(signal_names):
        ax = axes[idx]
        
        ax.plot(time, raw_signals[name], 'r-', linewidth=1, alpha=0.7, label='Raw (after Wiener filter)')
        ax.plot(time, savgol_filtered_signals[name], 'b-', linewidth=1, alpha=0.7, 
                label=f'Savitzky-Golay (w={window_length}, p={polyorder})')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Savitzky-Golay Filtered Signal Comparison - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_time_domain_signals(time, u, x, y, show_plot=True, windows=None, tool_accel=None, tool_label=None, z_accel=None):
    """
    Plot time-domain signals u, x, y in a single figure with synchronized x-axes.
    Optionally includes z acceleration signal as 4th subplot and tool acceleration signal as 5th subplot.
    Each signal can have its own window overlay.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u : array_like
        Force signal (N)
    x : array_like
        X-direction acceleration (m/s²)
    y : array_like
        Y-direction acceleration (m/s²)
    show_plot : bool
        Whether to display the plot
    windows : dict, optional
        Dictionary of window functions to overlay on plots.
        Keys: 'u', 'x', 'y', 'z', 'tool' (optional)
        Values: window arrays matching time length
    tool_accel : array_like, optional
        Tool acceleration signal (m/s²) - x_tool or y_tool
    tool_label : str, optional
        Label for tool acceleration (e.g., 'x_tool' or 'y_tool')
    z_accel : array_like, optional
        Z-direction acceleration signal (m/s²) - only for case3
    """
    if not show_plot:
        return
    
    # Determine number of subplots
    n_plots = 3
    if z_accel is not None:
        n_plots += 1
    if tool_accel is not None:
        n_plots += 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 10 + 2.5 * (n_plots - 3)), sharex=True)
    
    # Plot force u
    axes[0].plot(time, u, 'b-', linewidth=1, label='u (windowed)')
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title('Force Input u(t)')
    axes[0].grid(True, alpha=0.3)
    if windows is not None and 'u' in windows:
        window_u = windows['u']
        scale = np.max(np.abs(u))
        scale = scale if scale > 0 else 1.0
        axes[0].plot(time, window_u * scale, 'k--', linewidth=1, label='Window')
        axes[0].legend(fontsize=8)
    
    # Plot x acceleration
    axes[1].plot(time, x, 'r-', linewidth=1, label='x (windowed)')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('X-Direction Acceleration x(t)')
    axes[1].grid(True, alpha=0.3)
    if windows is not None and 'x' in windows:
        window_x = windows['x']
        scale = np.max(np.abs(x))
        scale = scale if scale > 0 else 1.0
        axes[1].plot(time, window_x * scale, 'k--', linewidth=1, label='Window')
        axes[1].legend(fontsize=8)
    
    # Plot y acceleration
    axes[2].plot(time, y, 'g-', linewidth=1, label='y (windowed)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_title('Y-Direction Acceleration y(t)')
    axes[2].grid(True, alpha=0.3)
    if windows is not None and 'y' in windows:
        window_y = windows['y']
        scale = np.max(np.abs(y))
        scale = scale if scale > 0 else 1.0
        axes[2].plot(time, window_y * scale, 'k--', linewidth=1, label='Window')
        axes[2].legend(fontsize=8)
    
    # Track current axis index
    current_axis = 3
    
    # Plot z acceleration if provided (before tool)
    if z_accel is not None:
        axes[current_axis].plot(time, z_accel, 'c-', linewidth=1, label='z (windowed)')
        axes[current_axis].set_ylabel('Acceleration (m/s²)')
        axes[current_axis].set_title('Z-Direction Acceleration z(t)')
        axes[current_axis].grid(True, alpha=0.3)
        if windows is not None and 'z' in windows:
            window_z = windows['z']
            scale = np.max(np.abs(z_accel))
            scale = scale if scale > 0 else 1.0
            axes[current_axis].plot(time, window_z * scale, 'k--', linewidth=1, label='Window')
            axes[current_axis].legend(fontsize=8)
        current_axis += 1
    
    # Plot tool acceleration if provided
    if tool_accel is not None:
        axes[current_axis].plot(time, tool_accel, 'm-', linewidth=1, label=f'{tool_label} (windowed)')
        axes[current_axis].set_xlabel('Time (s)')
        axes[current_axis].set_ylabel('Acceleration (m/s²)')
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Tool Acceleration'
        axes[current_axis].set_title(f'Tool Acceleration {title_label}(t)')
        axes[current_axis].grid(True, alpha=0.3)
        if windows is not None and 'tool' in windows:
            window_tool = windows['tool']
            scale = np.max(np.abs(tool_accel))
            scale = scale if scale > 0 else 1.0
            axes[current_axis].plot(time, window_tool * scale, 'k--', linewidth=1, label='Window')
            axes[current_axis].legend(fontsize=8)
    else:
        # Set xlabel on the last plotted axis
        if z_accel is not None:
            axes[current_axis - 1].set_xlabel('Time (s)')
        else:
            axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_bode_diagram(frequencies, Gx, Gy, Gx_accel=None, Gy_accel=None, show_plot=True, G_tool=None, G_tool_accel=None, tool_label=None):
    """
    Generate Bode magnitude and phase plots for Gx and Gy in a 2x2 or 2x3 layout.
    Optionally overlay acceleration-based transfer functions.
    Optionally includes tool compliance G_tool as 3rd column.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    Gx, Gy : array_like
        Displacement-based transfer functions (complex)
    Gx_accel, Gy_accel : array_like, optional
        Acceleration-based transfer functions (complex)
    show_plot : bool
        Whether to display the plot
    G_tool : array_like, optional
        Tool compliance transfer function (complex)
    G_tool_accel : array_like, optional
        Tool acceleration-based transfer function (complex)
    tool_label : str, optional
        Label for tool compliance (e.g., 'Gx_tool' or 'Gy_tool')
    """
    if not show_plot:
        return
    
    # Calculate magnitudes and phases for displacement
    Gx_mag = np.abs(Gx)
    Gy_mag = np.abs(Gy)
    Gx_phase = np.angle(Gx, deg=False)  # Get phase in radians first
    Gy_phase = np.angle(Gy, deg=False)  # Get phase in radians first
    
    # Unwrap phase to remove discontinuities
    Gx_phase = np.unwrap(Gx_phase)
    Gy_phase = np.unwrap(Gy_phase)
    
    # Convert to degrees after unwrapping
    Gx_phase = np.degrees(Gx_phase)
    Gy_phase = np.degrees(Gy_phase)
    
    # Calculate magnitudes and phases for acceleration if provided
    if Gx_accel is not None and Gy_accel is not None:
        Gx_accel_mag = np.abs(Gx_accel)
        Gy_accel_mag = np.abs(Gy_accel)
        Gx_accel_phase = np.angle(Gx_accel, deg=False)
        Gy_accel_phase = np.angle(Gy_accel, deg=False)
        
        # Unwrap phase
        Gx_accel_phase = np.unwrap(Gx_accel_phase)
        Gy_accel_phase = np.unwrap(Gy_accel_phase)
        
        # Convert to degrees
        Gx_accel_phase = np.degrees(Gx_accel_phase)
        Gy_accel_phase = np.degrees(Gy_accel_phase)
    else:
        pass
    
    # Process tool compliance if provided
    if G_tool is not None:
        G_tool_mag = np.abs(G_tool)
        G_tool_phase = np.angle(G_tool, deg=False)
        G_tool_phase = np.unwrap(G_tool_phase)
        G_tool_phase = np.degrees(G_tool_phase)
    
    # Calculate magnitudes and phases for tool acceleration if provided
    if G_tool_accel is not None:
        G_tool_accel_mag = np.abs(G_tool_accel)
        G_tool_accel_phase = np.angle(G_tool_accel, deg=False)
        
        # Unwrap phase
        G_tool_accel_phase = np.unwrap(G_tool_accel_phase)
        
        # Convert to degrees
        G_tool_accel_phase = np.degrees(G_tool_accel_phase)
    
    # Remove NaN values for plotting (displacement)
    valid_mask = np.isfinite(Gx_mag) & np.isfinite(Gy_mag) & np.isfinite(Gx_phase) & np.isfinite(Gy_phase)
    valid_freq = frequencies[valid_mask]
    valid_Gx_mag = Gx_mag[valid_mask]
    valid_Gy_mag = Gy_mag[valid_mask]
    valid_Gx_phase = Gx_phase[valid_mask]
    valid_Gy_phase = Gy_phase[valid_mask]
    
    # Process acceleration data if provided
    if Gx_accel is not None and Gy_accel is not None:
        # Skip DC component (index 0) which may contain NaN/Inf, then filter
        # Start from index 1 to avoid DC issues
        if len(frequencies) > 1:
            freq_accel = frequencies[1:]
            Gx_accel_mag_filtered = Gx_accel_mag[1:]
            Gy_accel_mag_filtered = Gy_accel_mag[1:]
            Gx_accel_phase_filtered = Gx_accel_phase[1:]
            Gy_accel_phase_filtered = Gy_accel_phase[1:]
        else:
            freq_accel = frequencies
            Gx_accel_mag_filtered = Gx_accel_mag
            Gy_accel_mag_filtered = Gy_accel_mag
            Gx_accel_phase_filtered = Gx_accel_phase
            Gy_accel_phase_filtered = Gy_accel_phase
        
        # Now filter for finite values
        valid_mask_accel = np.isfinite(Gx_accel_mag_filtered) & np.isfinite(Gy_accel_mag_filtered) & np.isfinite(Gx_accel_phase_filtered) & np.isfinite(Gy_accel_phase_filtered)
        valid_freq_accel = freq_accel[valid_mask_accel]
        valid_Gx_accel_mag = Gx_accel_mag_filtered[valid_mask_accel]
        valid_Gy_accel_mag = Gy_accel_mag_filtered[valid_mask_accel]
        valid_Gx_accel_phase = Gx_accel_phase_filtered[valid_mask_accel]
        valid_Gy_accel_phase = Gy_accel_phase_filtered[valid_mask_accel]
    else:
        # Initialize empty arrays if acceleration data is not available
        valid_freq_accel = np.array([])
        valid_Gx_accel_mag = np.array([])
        valid_Gy_accel_mag = np.array([])
        valid_Gx_accel_phase = np.array([])
        valid_Gy_accel_phase = np.array([])
    
    # Process tool data if provided
    if G_tool is not None:
        valid_mask_tool = np.isfinite(G_tool_mag) & np.isfinite(G_tool_phase)
        valid_freq_tool = frequencies[valid_mask_tool]
        valid_G_tool_mag = G_tool_mag[valid_mask_tool]
        valid_G_tool_phase = G_tool_phase[valid_mask_tool]
    
    # Process tool acceleration data if provided
    if G_tool_accel is not None:
        valid_mask_tool_accel = np.isfinite(G_tool_accel_mag) & np.isfinite(G_tool_accel_phase)
        valid_freq_tool_accel = frequencies[valid_mask_tool_accel]
        valid_G_tool_accel_mag = G_tool_accel_mag[valid_mask_tool_accel]
        valid_G_tool_accel_phase = G_tool_accel_phase[valid_mask_tool_accel]
    
    # Skip DC component for log plot
    if len(valid_freq) > 1 and valid_freq[0] == 0:
        valid_freq = valid_freq[1:]
        valid_Gx_mag = valid_Gx_mag[1:]
        valid_Gy_mag = valid_Gy_mag[1:]
        valid_Gx_phase = valid_Gx_phase[1:]
        valid_Gy_phase = valid_Gy_phase[1:]
    
    # DC component already skipped during filtering, so no need to skip again here
    
    # Skip DC component for tool data
    if G_tool is not None:
        if len(valid_freq_tool) > 1 and valid_freq_tool[0] == 0:
            valid_freq_tool = valid_freq_tool[1:]
            valid_G_tool_mag = valid_G_tool_mag[1:]
            valid_G_tool_phase = valid_G_tool_phase[1:]
    
    # Skip DC component for tool acceleration data
    if G_tool_accel is not None:
        if len(valid_freq_tool_accel) > 1 and valid_freq_tool_accel[0] == 0:
            valid_freq_tool_accel = valid_freq_tool_accel[1:]
            valid_G_tool_accel_mag = valid_G_tool_accel_mag[1:]
            valid_G_tool_accel_phase = valid_G_tool_accel_phase[1:]
    
    # Determine layout: 2x3 if tool is provided, otherwise 2x2
    n_cols = 3 if G_tool is not None else 2
    fig, ax = plt.subplots(2, n_cols, figsize=(14 + 5 * (n_cols - 2), 10))
    
    # Link x-axes within each column for synchronized zooming
    ax[1, 0].sharex(ax[0, 0])  # Gx magnitude and phase
    ax[1, 1].sharex(ax[0, 1])  # Gy magnitude and phase
    if G_tool is not None:
        ax[1, 2].sharex(ax[0, 2])  # Tool magnitude and phase
    
    # Plot Gx magnitude (top-left)
    ax[0, 0].loglog(valid_freq, valid_Gx_mag, 'b-', linewidth=2, label='Gx (Displacement)')
    if Gx_accel is not None and len(valid_freq_accel) > 0:
        ax[0, 0].loglog(valid_freq_accel, valid_Gx_accel_mag, 'b--', linewidth=2, alpha=0.7, label='Gx (Acceleration)')
    ax[0, 0].set_ylabel('Compliance (m/N or m/s²/N)')
    ax[0, 0].set_title('Compliance Gx (X/U) - Magnitude')
    ax[0, 0].grid(True, which="both", ls="-", alpha=0.3)
    ax[0, 0].legend()
    
    # Plot Gy magnitude (top-middle)
    ax[0, 1].loglog(valid_freq, valid_Gy_mag, 'r-', linewidth=2, label='Gy (Displacement)')
    if Gy_accel is not None and len(valid_freq_accel) > 0:
        ax[0, 1].loglog(valid_freq_accel, valid_Gy_accel_mag, 'r--', linewidth=2, alpha=0.7, label='Gy (Acceleration)')
    ax[0, 1].set_ylabel('Compliance (m/N or m/s²/N)')
    ax[0, 1].set_title('Compliance Gy (Y/U) - Magnitude')
    ax[0, 1].grid(True, which="both", ls="-", alpha=0.3)
    ax[0, 1].legend()
    
    # Plot tool magnitude (top-right) if provided
    if G_tool is not None:
        ax[0, 2].loglog(valid_freq_tool, valid_G_tool_mag, 'm-', linewidth=2, label=f'{tool_label} (Displacement)')
        if G_tool_accel is not None:
            ax[0, 2].loglog(valid_freq_tool_accel, valid_G_tool_accel_mag, 'm--', linewidth=2, alpha=0.7, label=f'{tool_label} (Acceleration)')
        ax[0, 2].set_ylabel('Compliance (m/N or m/s²/N)')
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Tool Compliance'
        ax[0, 2].set_title(f'Compliance {title_label} - Magnitude')
        ax[0, 2].grid(True, which="both", ls="-", alpha=0.3)
        ax[0, 2].legend()
    
    # Plot Gx phase (bottom-left)
    ax[1, 0].semilogx(valid_freq, valid_Gx_phase, 'b-', linewidth=2, label='Gx (Displacement)')
    if Gx_accel is not None and len(valid_freq_accel) > 0:
        ax[1, 0].semilogx(valid_freq_accel, valid_Gx_accel_phase, 'b--', linewidth=2, alpha=0.7, label='Gx (Acceleration)')
    ax[1, 0].set_xlabel('Frequency (Hz)')
    ax[1, 0].set_ylabel('Phase (degrees)')
    ax[1, 0].set_title('Compliance Gx (X/U) - Phase')
    ax[1, 0].grid(True, which="both", ls="-", alpha=0.3)
    ax[1, 0].legend()
    # Removed fixed y-limits to allow unwrapped phase to display properly
    
    # Plot Gy phase (bottom-middle)
    ax[1, 1].semilogx(valid_freq, valid_Gy_phase, 'r-', linewidth=2, label='Gy (Displacement)')
    if Gy_accel is not None and len(valid_freq_accel) > 0:
        ax[1, 1].semilogx(valid_freq_accel, valid_Gy_accel_phase, 'r--', linewidth=2, alpha=0.7, label='Gy (Acceleration)')
    ax[1, 1].set_xlabel('Frequency (Hz)')
    ax[1, 1].set_ylabel('Phase (degrees)')
    ax[1, 1].set_title('Compliance Gy (Y/U) - Phase')
    ax[1, 1].grid(True, which="both", ls="-", alpha=0.3)
    ax[1, 1].legend()
    # Removed fixed y-limits to allow unwrapped phase to display properly
    
    # Plot tool phase (bottom-right) if provided
    if G_tool is not None:
        ax[1, 2].semilogx(valid_freq_tool, valid_G_tool_phase, 'm-', linewidth=2, label=f'{tool_label} (Displacement)')
        if G_tool_accel is not None:
            ax[1, 2].semilogx(valid_freq_tool_accel, valid_G_tool_accel_phase, 'm--', linewidth=2, alpha=0.7, label=f'{tool_label} (Acceleration)')
        ax[1, 2].set_xlabel('Frequency (Hz)')
        ax[1, 2].set_ylabel('Phase (degrees)')
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Tool Compliance'
        ax[1, 2].set_title(f'Compliance {title_label} - Phase')
        ax[1, 2].grid(True, which="both", ls="-", alpha=0.3)
        ax[1, 2].legend()
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_real_transfer_functions(frequencies, Gx, Gy, Gx_accel=None, Gy_accel=None, show_plot=True, G_tool=None, G_tool_accel=None, tool_label=None):
    """
    Generate plots of the real and imaginary parts of transfer functions Gx, Gy, and Gtool on linear scale.
    Creates a 2-column layout: accelerations in the left column, displacements in the right column.
    Has 2 or 3 rows depending on whether G_tool is provided.
    Both real (solid) and imaginary (dashed) parts are plotted on each subplot.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    Gx, Gy : array_like
        Displacement-based transfer functions (complex)
    Gx_accel, Gy_accel : array_like, optional
        Acceleration-based transfer functions (complex)
    show_plot : bool
        Whether to display the plot
    G_tool : array_like, optional
        Tool compliance transfer function (complex)
    G_tool_accel : array_like, optional
        Tool acceleration-based transfer function (complex)
    tool_label : str, optional
        Label for tool compliance (e.g., 'Gx_tool' or 'Gy_tool')
    """
    if not show_plot:
        return
    
    # Extract real parts for displacement
    Gx_real = np.real(Gx)
    Gy_real = np.real(Gy)
    
    # Extract imaginary parts for displacement
    Gx_imag = np.imag(Gx)
    Gy_imag = np.imag(Gy)
    
    # Extract real parts for acceleration if provided
    if Gx_accel is not None and Gy_accel is not None:
        Gx_accel_real = np.real(Gx_accel)
        Gy_accel_real = np.real(Gy_accel)
        Gx_accel_imag = np.imag(Gx_accel)
        Gy_accel_imag = np.imag(Gy_accel)
    
    # Extract real parts for tool if provided
    if G_tool is not None:
        G_tool_real = np.real(G_tool)
        G_tool_imag = np.imag(G_tool)
    
    # Extract real parts for tool acceleration if provided
    if G_tool_accel is not None:
        G_tool_accel_real = np.real(G_tool_accel)
        G_tool_accel_imag = np.imag(G_tool_accel)
    
    # Remove NaN values for plotting (displacement)
    valid_mask = np.isfinite(Gx_real) & np.isfinite(Gy_real) & np.isfinite(Gx_imag) & np.isfinite(Gy_imag)
    valid_freq = frequencies[valid_mask]
    valid_Gx_real = Gx_real[valid_mask]
    valid_Gy_real = Gy_real[valid_mask]
    valid_Gx_imag = Gx_imag[valid_mask]
    valid_Gy_imag = Gy_imag[valid_mask]
    
    # Process acceleration data if provided
    if Gx_accel is not None and Gy_accel is not None:
        valid_mask_accel = np.isfinite(Gx_accel_real) & np.isfinite(Gy_accel_real) & np.isfinite(Gx_accel_imag) & np.isfinite(Gy_accel_imag)
        valid_freq_accel = frequencies[valid_mask_accel]
        valid_Gx_accel_real = Gx_accel_real[valid_mask_accel]
        valid_Gy_accel_real = Gy_accel_real[valid_mask_accel]
        valid_Gx_accel_imag = Gx_accel_imag[valid_mask_accel]
        valid_Gy_accel_imag = Gy_accel_imag[valid_mask_accel]
    
    # Process tool data if provided
    if G_tool is not None:
        valid_mask_tool = np.isfinite(G_tool_real) & np.isfinite(G_tool_imag)
        valid_freq_tool = frequencies[valid_mask_tool]
        valid_G_tool_real = G_tool_real[valid_mask_tool]
        valid_G_tool_imag = G_tool_imag[valid_mask_tool]
    
    # Process tool acceleration data if provided
    if G_tool_accel is not None:
        valid_mask_tool_accel = np.isfinite(G_tool_accel_real) & np.isfinite(G_tool_accel_imag)
        valid_freq_tool_accel = frequencies[valid_mask_tool_accel]
        valid_G_tool_accel_real = G_tool_accel_real[valid_mask_tool_accel]
        valid_G_tool_accel_imag = G_tool_accel_imag[valid_mask_tool_accel]
    
    # Determine number of rows: 3 if G_tool is provided, otherwise 2
    n_rows = 3 if G_tool is not None else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 10 + 2.5 * (n_rows - 2)), sharex='col', sharey='none')
    
    # Handle case where n_rows == 1 (shouldn't happen, but just in case)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot Gx acceleration (first row, left column)
    if Gx_accel is not None and Gy_accel is not None:
        axes[0, 0].plot(valid_freq_accel, valid_Gx_accel_imag, 'k--', linewidth=2, label='Gx Imag (Acceleration)')
        axes[0, 0].plot(valid_freq_accel, valid_Gx_accel_real, 'b-', linewidth=2, label='Gx Real (Acceleration)')
    axes[0, 0].set_ylabel('real/imag(Gx)')
    axes[0, 0].set_title('Real and Imaginary Parts of Gx Transfer Function - Acceleration')
    axes[0, 0].grid(True, alpha=0.3)
    if Gx_accel is not None and Gy_accel is not None:
        axes[0, 0].legend()
    
    # Plot Gx displacement (first row, right column)
    axes[0, 1].plot(valid_freq, valid_Gx_imag, 'k--', linewidth=2, label='Gx Imag (Displacement)')
    axes[0, 1].plot(valid_freq, valid_Gx_real, 'b-', linewidth=2, label='Gx Real (Displacement)')
    axes[0, 1].set_ylabel('real/imag(Gx)')
    axes[0, 1].set_title('Real and Imaginary Parts of Gx Transfer Function - Displacement')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot Gy acceleration (second row, left column)
    if Gx_accel is not None and Gy_accel is not None:
        axes[1, 0].plot(valid_freq_accel, valid_Gy_accel_imag, 'k--', linewidth=2, label='Gy Imag (Acceleration)')
        axes[1, 0].plot(valid_freq_accel, valid_Gy_accel_real, 'r-', linewidth=2, label='Gy Real (Acceleration)')
    axes[1, 0].set_ylabel('real/imag(Gy)')
    axes[1, 0].set_title('Real and Imaginary Parts of Gy Transfer Function - Acceleration')
    axes[1, 0].grid(True, alpha=0.3)
    if Gx_accel is not None and Gy_accel is not None:
        axes[1, 0].legend()
    
    # Plot Gy displacement (second row, right column)
    axes[1, 1].plot(valid_freq, valid_Gy_imag, 'k--', linewidth=2, label='Gy Imag (Displacement)')
    axes[1, 1].plot(valid_freq, valid_Gy_real, 'r-', linewidth=2, label='Gy Real (Displacement)')
    axes[1, 1].set_ylabel('real/imag(Gy)')
    axes[1, 1].set_title('Real and Imaginary Parts of Gy Transfer Function - Displacement')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot Gtool acceleration and displacement (third row) if provided
    if G_tool is not None:
        tool_label_display = tool_label if tool_label else 'Gtool'
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Gtool'
        
        # Plot Gtool acceleration (third row, left column)
        if G_tool_accel is not None:
            axes[2, 0].plot(valid_freq_tool_accel, valid_G_tool_accel_imag, 'k--', linewidth=2,
                            label=f'{tool_label_display} Imag (Acceleration)')
            axes[2, 0].plot(valid_freq_tool_accel, valid_G_tool_accel_real, 'm-', linewidth=2, label=f'{tool_label_display} Real (Acceleration)')
        axes[2, 0].set_ylabel(f'real/imag({tool_label_display})')
        axes[2, 0].set_xlabel('Frequency (Hz)')
        axes[2, 0].set_title(f'Real and Imaginary Parts of {title_label} Transfer Function - Acceleration')
        axes[2, 0].grid(True, alpha=0.3)
        if G_tool_accel is not None:
            axes[2, 0].legend()
        
        # Plot Gtool displacement (third row, right column)
        axes[2, 1].plot(valid_freq_tool, valid_G_tool_imag, 'k--', linewidth=2,
                        label=f'{tool_label_display} Imag (Displacement)')
        axes[2, 1].plot(valid_freq_tool, valid_G_tool_real, 'm-', linewidth=2, label=f'{tool_label_display} Real (Displacement)')
        axes[2, 1].set_ylabel(f'real/imag({tool_label_display})')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_title(f'Real and Imaginary Parts of {title_label} Transfer Function - Displacement')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
    else:
        # Set xlabel on the last row (Gy) if G_tool is not present
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_reconstructed_time_signals(time, u, X_freq, Y_freq, show_plot=True, Tool_freq=None, tool_label=None):
    """
    Plot time-domain signals: u (force), x_pos, and y_pos.
    x_pos and y_pos are reconstructed from frequency domain using IFFT.
    Optionally includes tool displacement as 4th subplot.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u : array_like
        Force signal in time domain (N)
    X_freq : array_like
        X displacement in frequency domain (complex, full spectrum)
    Y_freq : array_like
        Y displacement in frequency domain (complex, full spectrum)
    show_plot : bool
        Whether to display the plot
    Tool_freq : array_like, optional
        Tool displacement in frequency domain (complex, full spectrum)
    tool_label : str, optional
        Label for tool displacement (e.g., 'x_tool' or 'y_tool')
    """
    if not show_plot:
        return
    
    # Perform inverse FFT to reconstruct time-domain signals
    x_pos = np.real(np.fft.ifft(X_freq))
    y_pos = np.real(np.fft.ifft(Y_freq))
    
    # Determine number of subplots
    n_plots = 4 if Tool_freq is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 10 + 2.5 * (n_plots - 3)), sharex=True)
    
    # Plot force u
    axes[0].plot(time, u, 'b-', linewidth=1)
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title('Force Input u(t)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot x displacement
    axes[1].plot(time, x_pos, 'r-', linewidth=1)
    axes[1].set_ylabel('Displacement (m)')
    axes[1].set_title('X-Direction Displacement x_pos(t)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot y displacement
    axes[2].plot(time, y_pos, 'g-', linewidth=1)
    axes[2].set_ylabel('Displacement (m)')
    axes[2].set_title('Y-Direction Displacement y_pos(t)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot tool displacement if provided
    if Tool_freq is not None:
        tool_pos = np.real(np.fft.ifft(Tool_freq))
        axes[3].plot(time, tool_pos, 'm-', linewidth=1)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Displacement (m)')
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Tool'
        axes[3].set_title(f'{title_label}-Direction Displacement {tool_label}_pos(t)')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_fft_magnitudes(frequencies, X_accel, Y_accel, Tool_accel=None, show_plot=True):
    """
    Plot magnitude FFT of x, y, and x_tool (if available) in a 3x1 loglog subplot layout.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    X_accel : array_like
        X-direction acceleration FFT (complex)
    Y_accel : array_like
        Y-direction acceleration FFT (complex)
    Tool_accel : array_like, optional
        Tool acceleration FFT (complex) - x_tool or y_tool
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    # Calculate magnitudes
    X_accel_mag = np.abs(X_accel)
    Y_accel_mag = np.abs(Y_accel)
    
    # Remove DC component and prepare data for plotting
    valid_freq = frequencies.copy()
    valid_X_mag = X_accel_mag.copy()
    valid_Y_mag = Y_accel_mag.copy()
    
    # Skip DC component (frequency = 0) for log plot
    if len(valid_freq) > 1 and valid_freq[0] == 0:
        valid_freq = valid_freq[1:]
        valid_X_mag = valid_X_mag[1:]
        valid_Y_mag = valid_Y_mag[1:]
    
    # Remove any NaN or infinite values
    valid_mask_x = np.isfinite(valid_X_mag) & (valid_X_mag > 0)
    valid_mask_y = np.isfinite(valid_Y_mag) & (valid_Y_mag > 0)
    
    valid_freq_x = valid_freq[valid_mask_x]
    valid_X_mag_clean = valid_X_mag[valid_mask_x]
    valid_freq_y = valid_freq[valid_mask_y]
    valid_Y_mag_clean = valid_Y_mag[valid_mask_y]
    
    # Process tool signal if provided
    if Tool_accel is not None:
        Tool_accel_mag = np.abs(Tool_accel)
        valid_Tool_mag = Tool_accel_mag.copy()
        # Remove DC component if present (same as x and y)
        if len(valid_Tool_mag) == len(frequencies) and len(valid_freq) < len(frequencies):
            # DC was removed from frequencies, remove it from tool too
            valid_Tool_mag = valid_Tool_mag[1:]
        # Remove any NaN or infinite values
        valid_mask_tool = np.isfinite(valid_Tool_mag) & (valid_Tool_mag > 0)
        # Use the same frequency array as x and y (already has DC removed)
        if len(valid_Tool_mag) == len(valid_freq):
            valid_freq_tool = valid_freq[valid_mask_tool]
            valid_Tool_mag_clean = valid_Tool_mag[valid_mask_tool]
        else:
            # Lengths don't match - skip tool plot
            valid_freq_tool = None
            valid_Tool_mag_clean = None
    else:
        valid_freq_tool = None
        valid_Tool_mag_clean = None
    
    # Create figure with 3 rows and 1 column, synchronized x-axes
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot X acceleration magnitude
    axes[0].loglog(valid_freq_x, valid_X_mag_clean, 'r-', linewidth=2)
    axes[0].set_ylabel('|X_accel| (m/s²)')
    axes[0].set_title('X-Direction Acceleration FFT Magnitude')
    axes[0].grid(True, which="both", ls="-", alpha=0.3)
    
    # Plot Y acceleration magnitude
    axes[1].loglog(valid_freq_y, valid_Y_mag_clean, 'g-', linewidth=2)
    axes[1].set_ylabel('|Y_accel| (m/s²)')
    axes[1].set_title('Y-Direction Acceleration FFT Magnitude')
    axes[1].grid(True, which="both", ls="-", alpha=0.3)
    
    # Plot tool acceleration magnitude if available
    if Tool_accel is not None and valid_freq_tool is not None and len(valid_freq_tool) > 0:
        axes[2].loglog(valid_freq_tool, valid_Tool_mag_clean, 'm-', linewidth=2)
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('|Tool_accel| (m/s²)')
        axes[2].set_title('Tool Acceleration FFT Magnitude')
        axes[2].grid(True, which="both", ls="-", alpha=0.3)
    else:
        # Hide third subplot if tool signal not available
        axes[2].axis('off')
        axes[1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_force_psd(frequencies, U, dt, N, show_plot=True):
    """
    Plot Power Spectral Density of the input force signal.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    U : array_like
        Force FFT (complex)
    dt : float
        Sample time in seconds
    N : int
        Number of samples
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    # Calculate sampling frequency
    fs = 1 / dt
    
    # Calculate Power Spectral Density from FFT
    # PSD = (2 / (N * fs)) * |U|^2
    # Factor of 2 accounts for one-sided spectrum (positive frequencies only)
    U_magnitude = np.abs(U)
    PSD = (2 / (N * fs)) * U_magnitude**2
    
    # Remove DC component and prepare data for plotting
    valid_freq = frequencies.copy()
    valid_PSD = PSD.copy()
    
    # Skip DC component (frequency = 0) for log plot
    if len(valid_freq) > 1 and valid_freq[0] == 0:
        valid_freq = valid_freq[1:]
        valid_PSD = valid_PSD[1:]
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(valid_PSD) & (valid_PSD > 0)
    valid_freq = valid_freq[valid_mask]
    valid_PSD = valid_PSD[valid_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PSD on log-log scale
    ax.loglog(valid_freq, valid_PSD, 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (N²/Hz)')
    ax.set_title('Power Spectral Density of Input Force')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

def plot_coherence(frequencies, coherence_ux, coherence_uy, show_plot=True, tool_coherence=None, tool_label=None):
    """
    Plot coherence functions for input-output relationships.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    coherence_ux : array_like
        Coherence between input u and output x (0 to 1)
    coherence_uy : array_like
        Coherence between input u and output y (0 to 1)
    show_plot : bool
        Whether to display the plot
    tool_coherence : array_like, optional
        Coherence between input u and tool output (0 to 1)
    tool_label : str, optional
        Label for tool coherence (e.g., 'x_tool' or 'y_tool')
    """
    if not show_plot:
        return
    
    # Remove NaN and invalid values
    valid_mask_ux = np.isfinite(coherence_ux) & (coherence_ux >= 0) & (coherence_ux <= 1)
    valid_mask_uy = np.isfinite(coherence_uy) & (coherence_uy >= 0) & (coherence_uy <= 1)
    
    valid_freq_ux = frequencies[valid_mask_ux]
    valid_coherence_ux = coherence_ux[valid_mask_ux]
    valid_freq_uy = frequencies[valid_mask_uy]
    valid_coherence_uy = coherence_uy[valid_mask_uy]
    
    # Process tool coherence if provided
    if tool_coherence is not None:
        # Handle case where tool frequencies might differ
        if len(tool_coherence) == len(frequencies):
            valid_mask_tool = np.isfinite(tool_coherence) & (tool_coherence >= 0) & (tool_coherence <= 1)
            valid_freq_tool = frequencies[valid_mask_tool]
            valid_coherence_tool = tool_coherence[valid_mask_tool]
        else:
            # Tool has different frequency array - would need frequencies_tool parameter
            # For now, skip tool plot if frequencies don't match
            valid_freq_tool = None
            valid_coherence_tool = None
    else:
        valid_freq_tool = None
        valid_coherence_tool = None
    
    # Skip DC component (frequency = 0) if present
    if len(valid_freq_ux) > 1 and valid_freq_ux[0] == 0:
        valid_freq_ux = valid_freq_ux[1:]
        valid_coherence_ux = valid_coherence_ux[1:]
    if len(valid_freq_uy) > 1 and valid_freq_uy[0] == 0:
        valid_freq_uy = valid_freq_uy[1:]
        valid_coherence_uy = valid_coherence_uy[1:]
    if valid_freq_tool is not None and len(valid_freq_tool) > 1 and valid_freq_tool[0] == 0:
        valid_freq_tool = valid_freq_tool[1:]
        valid_coherence_tool = valid_coherence_tool[1:]
    
    # Determine number of subplots
    n_plots = 3 if (tool_coherence is not None and valid_freq_tool is not None and len(valid_freq_tool) > 0) else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 6 + 2.5 * (n_plots - 2)), sharex=True)
    
    # Plot coherence ux
    axes[0].semilogx(valid_freq_ux, valid_coherence_ux, 'b-', linewidth=2, label='γ²_ux')
    axes[0].axhline(y=0.7, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.7)')
    axes[0].set_ylabel('Coherence')
    axes[0].set_title('Coherence between Input u and Output x (γ²_ux)')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, which="both", ls="-", alpha=0.3)
    axes[0].legend()
    
    # Plot coherence uy
    axes[1].semilogx(valid_freq_uy, valid_coherence_uy, 'r-', linewidth=2, label='γ²_uy')
    axes[1].axhline(y=0.7, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.7)')
    axes[1].set_ylabel('Coherence')
    axes[1].set_title('Coherence between Input u and Output y (γ²_uy)')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, which="both", ls="-", alpha=0.3)
    axes[1].legend()
    
    # Plot tool coherence if provided
    if tool_coherence is not None and valid_freq_tool is not None and len(valid_freq_tool) > 0:
        axes[2].semilogx(valid_freq_tool, valid_coherence_tool, 'm-', linewidth=2, label=f'γ²_{tool_label}' if tool_label else 'γ²_tool')
        axes[2].axhline(y=0.7, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.7)')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Coherence')
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Tool'
        axes[2].set_title(f'Coherence between Input u and Tool Output {title_label} (γ²_{tool_label})' if tool_label else 'Coherence between Input u and Tool Output')
        axes[2].set_ylim([0, 1.05])
        axes[2].grid(True, which="both", ls="-", alpha=0.3)
        axes[2].legend()
    else:
        axes[1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    # Do not call plt.show() here; show all figures once at the end.

# =============================================================================
# FILE OUTPUT
# =============================================================================

def generate_output_filename(original_filepath, suffix):
    """
    Generate output filename based on input filename and suffix (Gx, Gy, Gx_tool, or Gy_tool).
    
    Parameters:
    -----------
    original_filepath : str
        Path to original CSV file
    suffix : str
        Suffix to replace direction indicator ('Gx', 'Gy', 'Gx_tool', or 'Gy_tool')
        
    Returns:
    --------
    str : Generated output filename
    """
    filename = os.path.basename(original_filepath)
    
    # Replace direction indicator with suffix
    # Pattern: ..._x.csv or ..._y.csv (with optional suffix) -> ..._Gx.csv or ..._Gy.csv (preserving suffix)
    # For tool files: ..._x.csv -> ..._Gx_tool.csv
    # Make pattern case-insensitive to handle both .csv and .CSV extensions
    pattern = r'(_[xy])(_[^.]*)?(\.csv)$'
    new_filename = re.sub(pattern, f'_{suffix}\\2\\3', filename, flags=re.IGNORECASE)
    
    # Get directory of original file
    directory = os.path.dirname(original_filepath)
    
    return os.path.join(directory, new_filename)

def save_results(frequencies, Gx, Gy, original_filepath, save_files=True, 
                metadata=None, measurement_case='case1', Gx_accel=None, Gy_accel=None,
                G_tool=None, G_tool_accel=None):
    """
    Save frequency response data to CSV files with comprehensive Bode plot data.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    Gx, Gy : array_like
        Displacement transfer functions (complex)
    original_filepath : str
        Path to original CSV file
    save_files : bool
        Whether to save files
    metadata : dict, optional
        Metadata from parse_filename, contains 'direction' field ('x' or 'y')
    measurement_case : str, optional
        Measurement case identifier ('case1', 'case2_x', 'case2_y')
    Gx_accel, Gy_accel : array_like, optional
        Acceleration transfer functions (complex)
    G_tool, G_tool_accel : array_like, optional
        Tool transfer functions (complex) - displacement and acceleration
        
    Returns:
    --------
    list : List of saved file paths
    """
    if not save_files:
        return []
    
    saved_files = []
    
    # Determine direction from metadata if available, otherwise default to exporting both
    direction = None
    if metadata is not None and 'direction' in metadata:
        direction = metadata['direction'].lower()
    
    # Helper function to create DataFrame for a transfer function
    def create_transfer_function_df(freq, G_accel, G_disp, name):
        """Create DataFrame with all transfer function data."""
        # Calculate acceleration components
        G_accel_real = np.real(G_accel) if G_accel is not None else np.zeros_like(freq)
        G_accel_imag = np.imag(G_accel) if G_accel is not None else np.zeros_like(freq)
        G_accel_amp = np.abs(G_accel) if G_accel is not None else np.zeros_like(freq)
        G_accel_phase = np.angle(G_accel) if G_accel is not None else np.zeros_like(freq)
        
        # Calculate displacement components
        G_disp_real = np.real(G_disp) if G_disp is not None else np.zeros_like(freq)
        G_disp_imag = np.imag(G_disp) if G_disp is not None else np.zeros_like(freq)
        G_disp_amp = np.abs(G_disp) if G_disp is not None else np.zeros_like(freq)
        G_disp_phase = np.angle(G_disp) if G_disp is not None else np.zeros_like(freq)
        
        # Replace NaN/Inf with 0
        G_accel_real = np.nan_to_num(G_accel_real, nan=0.0, posinf=0.0, neginf=0.0)
        G_accel_imag = np.nan_to_num(G_accel_imag, nan=0.0, posinf=0.0, neginf=0.0)
        G_accel_amp = np.nan_to_num(G_accel_amp, nan=0.0, posinf=0.0, neginf=0.0)
        G_accel_phase = np.nan_to_num(G_accel_phase, nan=0.0, posinf=0.0, neginf=0.0)
        G_disp_real = np.nan_to_num(G_disp_real, nan=0.0, posinf=0.0, neginf=0.0)
        G_disp_imag = np.nan_to_num(G_disp_imag, nan=0.0, posinf=0.0, neginf=0.0)
        G_disp_amp = np.nan_to_num(G_disp_amp, nan=0.0, posinf=0.0, neginf=0.0)
        G_disp_phase = np.nan_to_num(G_disp_phase, nan=0.0, posinf=0.0, neginf=0.0)
        
        df = pd.DataFrame({
            'frequency': freq,
            'G_real_acceleration': G_accel_real,
            'G_imag_acceleration': G_accel_imag,
            'G_amplitude_acceleration': G_accel_amp,
            'G_phase_rad_acceleration': G_accel_phase,
            'G_real_displacement': G_disp_real,
            'G_imag_displacement': G_disp_imag,
            'G_amplitude_displacement': G_disp_amp,
            'G_phase_rad_displacement': G_disp_phase
        })
        return df
    
    # Export based on direction from filename
    if direction == 'x':
        # Export Gx file
        if Gx_accel is not None and Gx is not None:
            df_gx = create_transfer_function_df(frequencies, Gx_accel, Gx, 'Gx')
            gx_filename = generate_output_filename(original_filepath, 'Gx')
            df_gx.to_csv(gx_filename, index=False, float_format='%.6e')
            saved_files.append(gx_filename)
        
        # Export Gx_tool if available
        if measurement_case == 'case2_x' and G_tool_accel is not None and G_tool is not None:
            df_gx_tool = create_transfer_function_df(frequencies, G_tool_accel, G_tool, 'Gx_tool')
            gx_tool_filename = generate_output_filename(original_filepath, 'Gx_tool')
            df_gx_tool.to_csv(gx_tool_filename, index=False, float_format='%.6e')
            saved_files.append(gx_tool_filename)
    
    elif direction == 'y':
        # Export Gy file
        if Gy_accel is not None and Gy is not None:
            df_gy = create_transfer_function_df(frequencies, Gy_accel, Gy, 'Gy')
            gy_filename = generate_output_filename(original_filepath, 'Gy')
            df_gy.to_csv(gy_filename, index=False, float_format='%.6e')
            saved_files.append(gy_filename)
        
        # Export Gy_tool if available
        if measurement_case == 'case2_y' and G_tool_accel is not None and G_tool is not None:
            df_gy_tool = create_transfer_function_df(frequencies, G_tool_accel, G_tool, 'Gy_tool')
            gy_tool_filename = generate_output_filename(original_filepath, 'Gy_tool')
            df_gy_tool.to_csv(gy_tool_filename, index=False, float_format='%.6e')
            saved_files.append(gy_tool_filename)
    
    else:
        # Fallback: if direction is not available, export both (for backward compatibility)
        if Gx_accel is not None and Gx is not None:
            df_gx = create_transfer_function_df(frequencies, Gx_accel, Gx, 'Gx')
            gx_filename = generate_output_filename(original_filepath, 'Gx')
            df_gx.to_csv(gx_filename, index=False, float_format='%.6e')
            saved_files.append(gx_filename)
        
        if Gy_accel is not None and Gy is not None:
            df_gy = create_transfer_function_df(frequencies, Gy_accel, Gy, 'Gy')
            gy_filename = generate_output_filename(original_filepath, 'Gy')
            df_gy.to_csv(gy_filename, index=False, float_format='%.6e')
            saved_files.append(gy_filename)
    
    return saved_files

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_multiple_files(filepaths, measurement_case='case1', show_plot=True, save_files=True):
    """
    Process multiple measurement files with CSD/PSD averaging.
    
    Parameters:
    -----------
    filepaths : list
        List of paths to input CSV files
    measurement_case : str
        Measurement case identifier: 'case1', 'case2_x', 'case2_y', or 'case3'
    show_plot : bool
        Whether to display plots (controlled by show_all_figures for which plots)
    save_files : bool
        Whether to save output CSV files
        
    Returns:
    --------
    dict : Contains frequencies, Gx, Gy, and metadata
        Optionally includes G_tool for Case 2
        Note: For case3, z signal is calculated but not included in FRF analysis
    """
    try:
        # Load system parameters
        params = get_system_parameters()
        
        # Parse filename for metadata (use first file)
        metadata = parse_filename(filepaths[0])
        
        # Load all files and find minimum length
        all_data = []
        min_length = None
        min_dt = None
        
        print(f"Loading {len(filepaths)} files...")
        for i, filepath in enumerate(filepaths):
            time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt = load_csv_file(filepath)
            
            # Check sampling rate consistency
            if min_dt is None:
                min_dt = dt
            elif not np.isclose(dt, min_dt, rtol=1e-6):
                print(f"Warning: File {i+1} has different sampling rate. File 1: {1/min_dt:.1f} Hz, File {i+1}: {1/dt:.1f} Hz")
            
            # Track minimum length
            length = len(time)
            if min_length is None or length < min_length:
                min_length = length
            
            all_data.append({
                'time': time,
                'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
                'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7,
                'fs': fs,
                'dt': dt,
                'filepath': filepath
            })
        
        # Truncate all signals to minimum length
        if min_length is not None:
            print(f"Truncating all files to minimum length: {min_length} samples")
            for data in all_data:
                data['time'] = data['time'][:min_length]
                data['A0'] = data['A0'][:min_length]
                data['A1'] = data['A1'][:min_length]
                data['A2'] = data['A2'][:min_length]
                data['A3'] = data['A3'][:min_length]
                data['A4'] = data['A4'][:min_length]
                data['A5'] = data['A5'][:min_length]
                data['A6'] = data['A6'][:min_length]
                data['A7'] = data['A7'][:min_length]
        
        # Process each file and collect windowed signals for pyFRF averaging
        all_windowed_u = []
        all_windowed_x = []
        all_windowed_y = []
        all_windowed_tool = []
        has_tool = False
        
        print(f"Processing {len(filepaths)} files for averaging...")
        for i, data in enumerate(all_data):
            time = data['time']
            A0, A1, A2, A3, A4, A5, A6, A7 = data['A0'], data['A1'], data['A2'], data['A3'], data['A4'], data['A5'], data['A6'], data['A7']
            fs = data['fs']
            dt = data['dt']
            
            # Remove bias from all analog input signals
            window_size = params['bias_window_size']
            A0, bias_idx_A0, actual_window_A0 = remove_bias(A0, window_size)
            A1, bias_idx_A1, actual_window_A1 = remove_bias(A1, window_size)
            A2, bias_idx_A2, actual_window_A2 = remove_bias(A2, window_size)
            A3, bias_idx_A3, actual_window_A3 = remove_bias(A3, window_size)
            A4, bias_idx_A4, actual_window_A4 = remove_bias(A4, window_size)
            A5, bias_idx_A5, actual_window_A5 = remove_bias(A5, window_size)
            A6, bias_idx_A6, actual_window_A6 = remove_bias(A6, window_size)
            A7, bias_idx_A7, actual_window_A7 = remove_bias(A7, window_size)
            
            # Apply Wiener filter to A0-A7 if enabled
            use_wiener = params.get('wiener_filter', True)
            
            # Store original signals before filtering for noise calculation
            original_A0 = A0.copy()
            original_A1 = A1.copy()
            original_A2 = A2.copy()
            original_A3 = A3.copy()
            original_A4 = A4.copy()
            original_A5 = A5.copy()
            original_A6 = A6.copy()
            original_A7 = A7.copy()
            
            if use_wiener:
                # Calculate noise power for each channel A0-A7 from bias window
                noise_A0 = calculate_noise_power(original_A0, bias_idx_A0, actual_window_A0)
                noise_A1 = calculate_noise_power(original_A1, bias_idx_A1, actual_window_A1)
                noise_A2 = calculate_noise_power(original_A2, bias_idx_A2, actual_window_A2)
                noise_A3 = calculate_noise_power(original_A3, bias_idx_A3, actual_window_A3)
                noise_A4 = calculate_noise_power(original_A4, bias_idx_A4, actual_window_A4)
                noise_A5 = calculate_noise_power(original_A5, bias_idx_A5, actual_window_A5)
                noise_A6 = calculate_noise_power(original_A6, bias_idx_A6, actual_window_A6)
                noise_A7 = calculate_noise_power(original_A7, bias_idx_A7, actual_window_A7)

                wiener_mysize = int(params.get("wiener_mysize", 1000))
                # Apply Wiener filter to A0-A7
                A0 = apply_wiener_filter(A0, noise_A0, wiener_mysize)
                A1 = apply_wiener_filter(A1, noise_A1, wiener_mysize)
                A2 = apply_wiener_filter(A2, noise_A2, wiener_mysize)
                A3 = apply_wiener_filter(A3, noise_A3, wiener_mysize)
                A4 = apply_wiener_filter(A4, noise_A4, wiener_mysize)
                A5 = apply_wiener_filter(A5, noise_A5, wiener_mysize)
                A6 = apply_wiener_filter(A6, noise_A6, wiener_mysize)
                A7 = apply_wiener_filter(A7, noise_A7, wiener_mysize)
            
            # Apply Savitzky-Golay filter to A0-A7 if enabled
            use_savgol = params.get('savgol_filter', False)
            if use_savgol:
                window_length = params.get('savgol_window_length', 10)
                polyorder = params.get('savgol_polyorder', 3)
                
                A0 = apply_savgol_filter(A0, window_length, polyorder)
                A1 = apply_savgol_filter(A1, window_length, polyorder)
                A2 = apply_savgol_filter(A2, window_length, polyorder)
                A3 = apply_savgol_filter(A3, window_length, polyorder)
                A4 = apply_savgol_filter(A4, window_length, polyorder)
                A5 = apply_savgol_filter(A5, window_length, polyorder)
                A6 = apply_savgol_filter(A6, window_length, polyorder)
                A7 = apply_savgol_filter(A7, window_length, polyorder)
            
            # Calculate derived signals with measurement case (using filtered A0-A7)
            signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params, measurement_case)
            
            # Build signal-specific windows
            # Each signal gets its own window based on its own characteristics
            
            # Get window types from parameters
            input_window_type = params.get('input_window_type', 'hann')
            output_window_type = params.get('output_window_type', 'exponential')
            
            # Create window for u signal (input window)
            window_u, _ = create_window(
                signals['u'],
                time,
                params,
                dt,
                window_type=input_window_type
            )
            
            # Create window for x signal (output window)
            window_x, _ = create_window(
                signals['x'],
                time,
                params,
                dt,
                window_type=output_window_type
            )
            
            # Create window for y signal (output window)
            window_y, _ = create_window(
                signals['y'],
                time,
                params,
                dt,
                window_type=output_window_type
            )
            
            # Extract z signal if present (case3) and create its window (output window)
            z_signal = None
            if measurement_case == 'case3':
                z_signal = signals.get('z', None)
                if z_signal is not None:
                    window_z, _ = create_window(
                        z_signal,
                        time,
                        params,
                        dt,
                        window_type=output_window_type
                    )
                    z_signal = z_signal * window_z
            
            # Extract tool signal if present and create its window (output window)
            tool_signal = None
            if measurement_case == 'case2_x':
                tool_signal = signals.get('x_tool', None)
                if tool_signal is not None:
                    has_tool = True
                    window_tool, _ = create_window(
                        tool_signal,
                        time,
                        params,
                        dt,
                        window_type=output_window_type
                    )
                    tool_signal = tool_signal * window_tool
            elif measurement_case == 'case2_y':
                tool_signal = signals.get('y_tool', None)
                if tool_signal is not None:
                    has_tool = True
                    window_tool, _ = create_window(
                        tool_signal,
                        time,
                        params,
                        dt,
                        window_type=output_window_type
                    )
                    tool_signal = tool_signal * window_tool
            
            # Apply each window to its corresponding signal
            windowed_signals = {
                'u': signals['u'] * window_u,
                'x': signals['x'] * window_x,
                'y': signals['y'] * window_y
            }
            
            # Collect windowed signals for pyFRF averaging
            all_windowed_u.append(windowed_signals['u'])
            all_windowed_x.append(windowed_signals['x'])
            all_windowed_y.append(windowed_signals['y'])
            if tool_signal is not None:
                all_windowed_tool.append(tool_signal)
            else:
                all_windowed_tool.append(None)
            
            print(f"  Processed file {i+1}/{len(filepaths)}: {os.path.basename(data['filepath'])}")
        
        # Create 3D arrays for pyFRF averaging: (n_measurements, n_dofs, n_samples)
        n_files = len(all_windowed_u)
        n_samples = len(all_windowed_u[0])
        
        # Reshape for pyFRF: (n_files, 1, n_samples)
        exc_3d = np.array(all_windowed_u)[:, np.newaxis, :]  # (n_files, 1, n_samples)
        resp_x_3d = np.array(all_windowed_x)[:, np.newaxis, :]  # (n_files, 1, n_samples)
        resp_y_3d = np.array(all_windowed_y)[:, np.newaxis, :]  # (n_files, 1, n_samples)
        
        # Extract Welch method parameters
        nperseg = params.get('welch_nperseg', None)
        noverlap = params.get('welch_noverlap', None)
        window = params.get('welch_window', 'hann')
        estimator_type = params.get('estimator_type', 'H2')
        fs = 1 / min_dt
        fs_int = int(round(fs))  # Convert to int for pyFRF
        
        # Create FRF instances with built-in averaging for x and y responses
        print("Calculating FRF with pyFRF (averaging across files)...")
        frf_x = FRF(
            sampling_freq=fs_int,
            exc=exc_3d,
            resp=resp_x_3d,
            exc_type='f',  # force
            resp_type='a',  # acceleration
            frf_type=estimator_type,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )
        
        frf_y = FRF(
            sampling_freq=fs_int,
            exc=exc_3d,
            resp=resp_y_3d,
            exc_type='f',  # force
            resp_type='a',  # acceleration
            frf_type=estimator_type,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )
        
        # Get frequency axis
        frequencies = frf_x.get_f_axis()
        
        # Get acceleration transfer functions (accelerance: acceleration/force)
        # get_FRF() returns shape (resp_dofs, exc_dofs, frequency_series) for 3D or (resp_dofs, frequency_series) for 2D
        Gx_accel_full = frf_x.get_FRF(form='accelerance')
        Gy_accel_full = frf_y.get_FRF(form='accelerance')
        
        # Get displacement transfer functions (receptance: displacement/force)
        Gx_disp_full = frf_x.get_FRF(form='receptance')
        Gy_disp_full = frf_y.get_FRF(form='receptance')
        
        # Extract 1D arrays for single input/output - acceleration
        if Gx_accel_full.ndim == 3:
            Gx_accel = Gx_accel_full[0, 0, :]  # Extract (frequency_series,)
        elif Gx_accel_full.ndim == 2:
            Gx_accel = Gx_accel_full[0, :]  # Extract (frequency_series,)
        else:
            Gx_accel = Gx_accel_full  # Already 1D
        
        if Gy_accel_full.ndim == 3:
            Gy_accel = Gy_accel_full[0, 0, :]  # Extract (frequency_series,)
        elif Gy_accel_full.ndim == 2:
            Gy_accel = Gy_accel_full[0, :]  # Extract (frequency_series,)
        else:
            Gy_accel = Gy_accel_full  # Already 1D
        
        # Extract 1D arrays for single input/output - displacement
        if Gx_disp_full.ndim == 3:
            Gx = Gx_disp_full[0, 0, :]  # Extract (frequency_series,)
        elif Gx_disp_full.ndim == 2:
            Gx = Gx_disp_full[0, :]  # Extract (frequency_series,)
        else:
            Gx = Gx_disp_full  # Already 1D
        
        if Gy_disp_full.ndim == 3:
            Gy = Gy_disp_full[0, 0, :]  # Extract (frequency_series,)
        elif Gy_disp_full.ndim == 2:
            Gy = Gy_disp_full[0, :]  # Extract (frequency_series,)
        else:
            Gy = Gy_disp_full  # Already 1D
        
        # Replace NaN values with 0 for displacement FRFs
        Gx = np.nan_to_num(Gx, nan=0.0)
        Gy = np.nan_to_num(Gy, nan=0.0)
        
        # Replace NaN values with 0 for acceleration FRFs
        Gx_accel = np.nan_to_num(Gx_accel, nan=0.0)
        Gy_accel = np.nan_to_num(Gy_accel, nan=0.0)
        
        # Get coherence (returns shape (resp_dofs, frequency_series) for 2D or 1D)
        coherence_ux_full = frf_x.get_coherence()
        coherence_uy_full = frf_y.get_coherence()
        
        if coherence_ux_full.ndim == 2:
            coherence_ux = coherence_ux_full[0, :]  # Extract (frequency_series,)
        else:
            coherence_ux = coherence_ux_full  # Already 1D
        
        if coherence_uy_full.ndim == 2:
            coherence_uy = coherence_uy_full[0, :]  # Extract (frequency_series,)
        else:
            coherence_uy = coherence_uy_full  # Already 1D
        
        # Process tool signal if present
        if has_tool and any(t is not None for t in all_windowed_tool):
            # Filter out None values and create 3D array
            tool_signals = [t for t in all_windowed_tool if t is not None]
            if len(tool_signals) == n_files:
                resp_tool_3d = np.array(tool_signals)[:, np.newaxis, :]  # (n_files, 1, n_samples)
                
                frf_tool = FRF(
                    sampling_freq=fs_int,
                    exc=exc_3d,
                    resp=resp_tool_3d,
                    exc_type='f',  # force
                    resp_type='a',  # acceleration
                    frf_type=estimator_type,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window=window
                )
                
                frequencies_tool = frf_tool.get_f_axis()
                G_tool_accel_full = frf_tool.get_FRF(form='accelerance')
                G_tool_disp_full = frf_tool.get_FRF(form='receptance')
                coherence_tool_full = frf_tool.get_coherence()
                
                # Extract 1D array for single input/output - acceleration
                if G_tool_accel_full.ndim == 3:
                    G_tool_accel = G_tool_accel_full[0, 0, :]  # Extract (frequency_series,)
                elif G_tool_accel_full.ndim == 2:
                    G_tool_accel = G_tool_accel_full[0, :]  # Extract (frequency_series,)
                else:
                    G_tool_accel = G_tool_accel_full  # Already 1D
                
                # Extract 1D array for single input/output - displacement
                if G_tool_disp_full.ndim == 3:
                    Tool = G_tool_disp_full[0, 0, :]  # Extract (frequency_series,)
                elif G_tool_disp_full.ndim == 2:
                    Tool = G_tool_disp_full[0, :]  # Extract (frequency_series,)
                else:
                    Tool = G_tool_disp_full  # Already 1D
                
                if coherence_tool_full.ndim == 2:
                    coherence_tool = coherence_tool_full[0, :]  # Extract (frequency_series,)
                else:
                    coherence_tool = coherence_tool_full  # Already 1D
                
                # Replace NaN values with 0 for tool FRFs
                Tool = np.nan_to_num(Tool, nan=0.0)
                G_tool_accel = np.nan_to_num(G_tool_accel, nan=0.0)
            else:
                G_tool_accel = None
                Tool = None
                coherence_tool = None
                frequencies_tool = None
        else:
            G_tool_accel = None
            Tool = None
            coherence_tool = None
            frequencies_tool = None
        
        # Displacement and acceleration FRFs are now obtained directly from pyFRF
        # No manual double integration needed
        
        # Get PSD of input force (Guu) for U calculation
        # Use scipy.welch on averaged input signal
        u_avg = np.mean(all_windowed_u, axis=0)
        _, Guu = signal.welch(u_avg, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                             window=window, scaling='density', return_onesided=True)
        
        # Interpolate Guu to match frequencies if needed
        if len(Guu) == len(frequencies):
            Guu_aligned = Guu
        else:
            from scipy.interpolate import interp1d
            freq_guu = np.fft.rfftfreq(n_samples, min_dt)[:len(Guu)]
            valid_guu = np.isfinite(Guu) & (Guu > 0)
            if np.any(valid_guu):
                interp_func = interp1d(freq_guu[valid_guu], Guu[valid_guu], 
                                     kind='linear', bounds_error=False, fill_value=np.nan)
                Guu_aligned = interp_func(frequencies)
            else:
                Guu_aligned = np.full_like(frequencies, np.nan)
        
        # Create U array for compatibility with plot_force_psd
        valid_mask_guu = np.isfinite(Guu_aligned) & (Guu_aligned > 0)
        U_pos = np.zeros_like(frequencies, dtype=complex)
        if np.any(valid_mask_guu):
            U_magnitude = np.sqrt(Guu_aligned[valid_mask_guu] * n_samples * fs / 2)
            U_pos[valid_mask_guu] = U_magnitude.astype(complex)
        
        # Create results dictionary with same structure as perform_pyfrf_analysis
        fft_results = {
            'frequencies': frequencies,
            'U': U_pos,  # U from PSD
            'X': Gx,  # Displacement transfer function
            'Y': Gy,  # Displacement transfer function
            'X_accel': Gx_accel,  # Acceleration transfer function
            'Y_accel': Gy_accel,  # Acceleration transfer function
            'X_full': None,  # Not used - skipping time signal reconstruction
            'Y_full': None,  # Not used - skipping time signal reconstruction
            'coherence_ux': coherence_ux,  # Coherence between input u and output x
            'coherence_uy': coherence_uy,  # Coherence between input u and output y
            'N': n_samples,
            'dt': min_dt
        }
        
        # Add tool results if available
        if Tool is not None:
            fft_results['Tool'] = Tool
            fft_results['Tool_accel'] = G_tool_accel
            fft_results['Tool_full'] = None  # Not used - skipping time signal reconstruction
            fft_results['coherence_tool'] = coherence_tool  # Coherence between input u and tool output
        
        # Extract transfer function results
        tf_results = {
            'Gx': fft_results['X'],
            'Gy': fft_results['Y'],
            'valid_mask': np.isfinite(fft_results['X']) & np.isfinite(fft_results['Y'])
        }
        if 'Tool' in fft_results:
            tf_results['G_tool'] = fft_results['Tool']
        
        tf_accel_results = {
            'Gx': fft_results['X_accel'],
            'Gy': fft_results['Y_accel'],
            'valid_mask': np.isfinite(fft_results['X_accel']) & np.isfinite(fft_results['Y_accel'])
        }
        if 'Tool_accel' in fft_results:
            tf_accel_results['G_tool'] = fft_results['Tool_accel']
        
        # Determine tool label for plots
        tool_label = None
        tool_bode_label = None
        if measurement_case == 'case2_x':
            tool_label = 'x_tool'
            tool_bode_label = 'Gx_tool'
        elif measurement_case == 'case2_y':
            tool_label = 'y_tool'
            tool_bode_label = 'Gy_tool'
        
        # Generate plots (controlled by show_all_figures)
        # Only show coherence and Bode plots unless show_all_figures is True
        if show_plot:
            # Plot coherence (always shown for multiple files)
            coherence_ux = fft_results.get('coherence_ux', None)
            coherence_uy = fft_results.get('coherence_uy', None)
            coherence_tool = fft_results.get('coherence_tool', None)
            plot_coherence(fft_results['frequencies'], coherence_ux, coherence_uy, show_plot=True,
                          tool_coherence=coherence_tool, tool_label=tool_label)
            
            # Generate Bode plots (always shown for multiple files)
            plot_bode_diagram(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                             tf_accel_results['Gx'], tf_accel_results['Gy'], show_plot=True,
                             G_tool=tf_results.get('G_tool', None), 
                             G_tool_accel=tf_accel_results.get('G_tool', None),
                             tool_label=tool_bode_label)
            
            # Plot real part of transfer functions (always shown)
            plot_real_transfer_functions(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                                         tf_accel_results['Gx'], tf_accel_results['Gy'], show_plot=True,
                                         G_tool=tf_results.get('G_tool', None), 
                                         G_tool_accel=tf_accel_results.get('G_tool', None),
                                         tool_label=tool_bode_label)
            
            # Show additional plots only if show_all_figures is True
            if show_all_figures:
                # For multiple files, we could show averaged time signals, but skip individual file plots
                # Show averaged FFT magnitudes if desired
                Tool_accel_fft = fft_results.get('Tool_accel', None)
                plot_fft_magnitudes(fft_results['frequencies'], fft_results['X_accel'], fft_results['Y_accel'],
                                   Tool_accel=Tool_accel_fft, show_plot=True)
                
                # Generate PSD plot for input force
                plot_force_psd(fft_results['frequencies'], fft_results['U'],
                               fft_results['dt'], fft_results['N'], show_plot=True)
        
        # Save results (use first file's path for naming)
        saved_files = save_results(
            fft_results['frequencies'], 
            tf_results['Gx'], 
            tf_results['Gy'], 
            filepaths[0], 
            save_files,
            metadata=metadata,
            measurement_case=measurement_case,
            Gx_accel=tf_accel_results['Gx'],
            Gy_accel=tf_accel_results['Gy'],
            G_tool=tf_results.get('G_tool', None),
            G_tool_accel=tf_accel_results.get('G_tool', None)
        )
        
        # Prepare return data
        result = {
            'frequencies': fft_results['frequencies'],
            'Gx': tf_results['Gx'],
            'Gy': tf_results['Gy'],
            'Gx_accel': tf_accel_results['Gx'],
            'Gy_accel': tf_accel_results['Gy'],
            'metadata': metadata,
            'saved_files': saved_files,
            'sample_rate': 1 / fft_results['dt'],
            'N_samples': fft_results['N'],
            'n_files': len(filepaths),
            'averaged': True
        }
        
        # Add tool transfer function if present
        if 'G_tool' in tf_results:
            result['G_tool'] = tf_results['G_tool']
        if 'G_tool' in tf_accel_results:
            result['G_tool_accel'] = tf_accel_results['G_tool']
        
        return result
        
    except Exception as e:
        import traceback
        raise Exception(f"Error processing multiple files: {str(e)}") from e

def process_measurement_file(filepath, measurement_case='case1', show_plot=True, save_files=True):
    """
    Main function to process a single measurement file.
    
    Parameters:
    -----------
    filepath : str
        Path to input CSV file
    measurement_case : str
        Measurement case identifier: 'case1', 'case2_x', 'case2_y', or 'case3'
    show_plot : bool
        Whether to display Bode plots
    save_files : bool
        Whether to save output CSV files
        
    Returns:
    --------
    dict : Contains frequencies, Gx, Gy, and metadata
        Optionally includes G_tool for Case 2
        Note: For case3, z signal is calculated but not included in FRF analysis
    """
    try:
        # Load system parameters
        params = get_system_parameters()
        
        # Parse filename for metadata
        metadata = parse_filename(filepath)
        
        # Load CSV data
        time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt = load_csv_file(filepath)
        
        # Store original signals for comparison plots
        original_signals = {
            'A0': A0.copy(), 'A1': A1.copy(), 'A2': A2.copy(), 'A3': A3.copy(),
            'A4': A4.copy(), 'A5': A5.copy(), 'A6': A6.copy(), 'A7': A7.copy()
        }
        
        # Remove bias from all analog input signals
        window_size = params['bias_window_size']
        # Capture window location for noise calculation for all channels
        A0, bias_idx_A0, actual_window_A0 = remove_bias(A0, window_size)
        A1, bias_idx_A1, actual_window_A1 = remove_bias(A1, window_size)
        A2, bias_idx_A2, actual_window_A2 = remove_bias(A2, window_size)
        A3, bias_idx_A3, actual_window_A3 = remove_bias(A3, window_size)
        A4, bias_idx_A4, actual_window_A4 = remove_bias(A4, window_size)
        A5, bias_idx_A5, actual_window_A5 = remove_bias(A5, window_size)
        A6, bias_idx_A6, actual_window_A6 = remove_bias(A6, window_size)
        A7, bias_idx_A7, actual_window_A7 = remove_bias(A7, window_size)
        
        # Store corrected signals for comparison plots
        corrected_signals = {
            'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
            'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
        }
        
        # Apply Wiener filter to A0-A7 if enabled
        use_wiener = params.get('wiener_filter', True)
        
        # Store original signals before filtering for noise calculation
        original_A0 = A0.copy()
        original_A1 = A1.copy()
        original_A2 = A2.copy()
        original_A3 = A3.copy()
        original_A4 = A4.copy()
        original_A5 = A5.copy()
        original_A6 = A6.copy()
        original_A7 = A7.copy()
        
        if use_wiener:
            # Calculate noise power for each channel A0-A7 from bias window
            noise_A0 = max(calculate_noise_power(original_A0, bias_idx_A0, actual_window_A0), 1e-12)
            noise_A1 = max(calculate_noise_power(original_A1, bias_idx_A1, actual_window_A1), 1e-12)
            noise_A2 = max(calculate_noise_power(original_A2, bias_idx_A2, actual_window_A2), 1e-12)
            noise_A3 = max(calculate_noise_power(original_A3, bias_idx_A3, actual_window_A3), 1e-12)
            noise_A4 = max(calculate_noise_power(original_A4, bias_idx_A4, actual_window_A4), 1e-12)
            noise_A5 = max(calculate_noise_power(original_A5, bias_idx_A5, actual_window_A5), 1e-12)
            noise_A6 = max(calculate_noise_power(original_A6, bias_idx_A6, actual_window_A6), 1e-12)
            noise_A7 = max(calculate_noise_power(original_A7, bias_idx_A7, actual_window_A7), 1e-12)

            wiener_mysize = int(params.get("wiener_mysize", 1000))
            # Apply Wiener filter to A0-A7
            A0 = apply_wiener_filter(A0, noise_A0, wiener_mysize)
            A1 = apply_wiener_filter(A1, noise_A1, wiener_mysize)
            A2 = apply_wiener_filter(A2, noise_A2, wiener_mysize)
            A3 = apply_wiener_filter(A3, noise_A3, wiener_mysize)
            A4 = apply_wiener_filter(A4, noise_A4, wiener_mysize)
            A5 = apply_wiener_filter(A5, noise_A5, wiener_mysize)
            A6 = apply_wiener_filter(A6, noise_A6, wiener_mysize)
            A7 = apply_wiener_filter(A7, noise_A7, wiener_mysize)

        # Store Wiener filtered signals separately for plotting (if Wiener was applied)
        wiener_filtered_signals = None
        if use_wiener:
            wiener_filtered_signals = {
                'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
                'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
            }

        # Store signals before Savitzky-Golay filter (these are the "raw" signals for comparison)
        raw_signals_before_savgol = {
            'A0': A0.copy(), 'A1': A1.copy(), 'A2': A2.copy(), 'A3': A3.copy(),
            'A4': A4.copy(), 'A5': A5.copy(), 'A6': A6.copy(), 'A7': A7.copy()
        }
        
        # Apply Savitzky-Golay filter to A0-A7 if enabled
        use_savgol = params.get('savgol_filter', False)
        if use_savgol:
            window_length = params.get('savgol_window_length', 10)
            polyorder = params.get('savgol_polyorder', 3)
            
            A0 = apply_savgol_filter(A0, window_length, polyorder)
            A1 = apply_savgol_filter(A1, window_length, polyorder)
            A2 = apply_savgol_filter(A2, window_length, polyorder)
            A3 = apply_savgol_filter(A3, window_length, polyorder)
            A4 = apply_savgol_filter(A4, window_length, polyorder)
            A5 = apply_savgol_filter(A5, window_length, polyorder)
            A6 = apply_savgol_filter(A6, window_length, polyorder)
            A7 = apply_savgol_filter(A7, window_length, polyorder)
            
            # Store Savitzky-Golay filtered signals for plotting
            savgol_filtered_signals = {
                'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
                'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
            }
        else:
            savgol_filtered_signals = None

        # Store filtered signals (after all filters) for comparison plots
        filtered_signals = {
            'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
            'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
        }
        
        # Show signal comparison plots (only if show_all_figures is True)
        if show_all_figures:
            plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot)
        
        # Apply low-pass filter to bias-corrected signals for analysis visualization
        use_lowpass = params.get('lowpass_filter', True)
        if use_lowpass:
            lowpass_bandwidth_hz = params['lowpass_bandwidth_hz']
            A0_lowpass = apply_lowpass_filter(corrected_signals['A0'], lowpass_bandwidth_hz, fs)
            A1_lowpass = apply_lowpass_filter(corrected_signals['A1'], lowpass_bandwidth_hz, fs)
            A2_lowpass = apply_lowpass_filter(corrected_signals['A2'], lowpass_bandwidth_hz, fs)
            A3_lowpass = apply_lowpass_filter(corrected_signals['A3'], lowpass_bandwidth_hz, fs)
            A4_lowpass = apply_lowpass_filter(corrected_signals['A4'], lowpass_bandwidth_hz, fs)
            A5_lowpass = apply_lowpass_filter(corrected_signals['A5'], lowpass_bandwidth_hz, fs)
            A6_lowpass = apply_lowpass_filter(corrected_signals['A6'], lowpass_bandwidth_hz, fs)
            A7_lowpass = apply_lowpass_filter(corrected_signals['A7'], lowpass_bandwidth_hz, fs)
            
            # Store low-pass filtered signals for plotting
            lowpass_filtered_signals = {
                'A0': A0_lowpass, 'A1': A1_lowpass, 'A2': A2_lowpass, 'A3': A3_lowpass,
                'A4': A4_lowpass, 'A5': A5_lowpass, 'A6': A6_lowpass, 'A7': A7_lowpass
            }
        else:
            lowpass_filtered_signals = None
        
        # Show low-pass filtered signals plot (only if show_all_figures is True and lowpass_filter is enabled)
        if show_all_figures and use_lowpass:
            lowpass_bandwidth_hz = params['lowpass_bandwidth_hz']
            plot_lowpass_filtered_signals(time, corrected_signals, lowpass_filtered_signals, 
                                         lowpass_bandwidth_hz, show_plot)
        
        # Show Wiener filtered signals plot (if Wiener filter was applied, only if show_all_figures is True)
        if show_all_figures and use_wiener and wiener_filtered_signals is not None:
            # Extract A0-A7 from corrected_signals for comparison
            corrected_signals_A0_A7 = {
                'A0': corrected_signals['A0'],
                'A1': corrected_signals['A1'],
                'A2': corrected_signals['A2'],
                'A3': corrected_signals['A3'],
                'A4': corrected_signals['A4'],
                'A5': corrected_signals['A5'],
                'A6': corrected_signals['A6'],
                'A7': corrected_signals['A7']
            }
            plot_wiener_filtered_signals(time, corrected_signals_A0_A7, wiener_filtered_signals, show_plot)
        
        # Show Savitzky-Golay filtered signals plot (if Savitzky-Golay filter was applied, only if show_all_figures is True)
        if show_all_figures and use_savgol and savgol_filtered_signals is not None:
            window_length = params.get('savgol_window_length', 10)
            polyorder = params.get('savgol_polyorder', 3)
            plot_savgol_filtered_signals(time, raw_signals_before_savgol, savgol_filtered_signals, 
                                        window_length, polyorder, show_plot)
        
        # Calculate derived signals with measurement case (using filtered A0-A7)
        signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params, measurement_case)

        # Build signal-specific windows
        # Each signal gets its own window based on its own characteristics
        
        # Get window types from parameters
        input_window_type = params.get('input_window_type', 'hann')
        output_window_type = params.get('output_window_type', 'exponential')
        
        # Create window for u signal (input window)
        window_u, window_metadata_u = create_window(
            signals['u'],
            time,
            params,
            dt,
            window_type=input_window_type
        )
        
        # Create window for x signal (output window)
        window_x, window_metadata_x = create_window(
            signals['x'],
            time,
            params,
            dt,
            window_type=output_window_type
        )
        
        # Create window for y signal (output window)
        window_y, window_metadata_y = create_window(
            signals['y'],
            time,
            params,
            dt,
            window_type=output_window_type
        )
        
        # Extract z signal if present (case3) and create its window (output window)
        z_signal = None
        window_z = None
        window_metadata_z = None
        if measurement_case == 'case3':
            z_signal = signals.get('z', None)
            if z_signal is not None:
                window_z, window_metadata_z = create_window(
                    z_signal,
                    time,
                    params,
                    dt,
                    window_type=output_window_type
                )
                z_signal = z_signal * window_z
        
        # Extract tool signal if present and create its window (output window)
        tool_signal = None
        tool_label = None
        window_tool = None
        window_metadata_tool = None
        if measurement_case == 'case2_x':
            tool_signal = signals.get('x_tool', None)
            if tool_signal is not None:
                tool_label = 'x_tool'
                window_tool, window_metadata_tool = create_window(
                    tool_signal,
                    time,
                    params,
                    dt,
                    window_type=output_window_type
                )
        elif measurement_case == 'case2_y':
            tool_signal = signals.get('y_tool', None)
            if tool_signal is not None:
                tool_label = 'y_tool'
                window_tool, window_metadata_tool = create_window(
                    tool_signal,
                    time,
                    params,
                    dt,
                    window_type=output_window_type
                )

        # Apply each window to its corresponding signal
        windowed_signals = {
            'u': signals['u'] * window_u,
            'x': signals['x'] * window_x,
            'y': signals['y'] * window_y
        }
        
        # Apply tool window if present
        if tool_signal is not None and window_tool is not None:
            tool_signal = tool_signal * window_tool
        
        # Prepare windows dictionary for plotting
        windows_dict = {
            'u': window_u,
            'x': window_x,
            'y': window_y
        }
        if window_z is not None:
            windows_dict['z'] = window_z
        if window_tool is not None:
            windows_dict['tool'] = window_tool
        
        # Plot time-domain signals with window overlay (only if show_all_figures is True)
        if show_all_figures:
            plot_time_domain_signals(time, windowed_signals['u'], windowed_signals['x'], 
                                    windowed_signals['y'], show_plot, windows=windows_dict,
                                    tool_accel=tool_signal, tool_label=tool_label,
                                    z_accel=z_signal)
        
        # Perform frequency domain analysis using pyFRF
        fft_results = perform_pyfrf_analysis(time, windowed_signals['u'], windowed_signals['x'], 
                                             windowed_signals['y'], dt, params, tool_signal=tool_signal)
        
        # Transfer functions are already calculated in perform_pyfrf_analysis
        # Extract them directly from results
        tf_results = {
            'Gx': fft_results['X'],  # Displacement transfer function
            'Gy': fft_results['Y'],  # Displacement transfer function
            'valid_mask': np.isfinite(fft_results['X']) & np.isfinite(fft_results['Y'])
        }
        if 'Tool' in fft_results:
            tf_results['G_tool'] = fft_results['Tool']
        
        tf_accel_results = {
            'Gx': fft_results['X_accel'],  # Acceleration transfer function
            'Gy': fft_results['Y_accel'],  # Acceleration transfer function
            'valid_mask': np.isfinite(fft_results['X_accel']) & np.isfinite(fft_results['Y_accel'])
        }
        if 'Tool_accel' in fft_results:
            tf_accel_results['G_tool'] = fft_results['Tool_accel']
        

        # Plot FFT magnitudes of x, y, and x_tool (if available, only if show_all_figures is True)
        if show_all_figures:
            Tool_accel_fft = fft_results.get('Tool_accel', None)
            plot_fft_magnitudes(fft_results['frequencies'], fft_results['X_accel'], fft_results['Y_accel'],
                               Tool_accel=Tool_accel_fft, show_plot=show_plot)

        # Generate PSD plot for input force (only if show_all_figures is True)
        if show_all_figures:
            plot_force_psd(fft_results['frequencies'], fft_results['U'],
                           fft_results['dt'], fft_results['N'], show_plot)
        
        # Plot coherence (always shown)
        if 'coherence_ux' in fft_results:
            coherence_ux = fft_results.get('coherence_ux', None)
            coherence_uy = fft_results.get('coherence_uy', None)
            coherence_tool = fft_results.get('coherence_tool', None)
            plot_coherence(fft_results['frequencies'], coherence_ux, coherence_uy, show_plot,
                          tool_coherence=coherence_tool, tool_label=tool_label)
        
        # Determine tool label for Bode plot
        tool_bode_label = None
        if measurement_case == 'case2_x':
            tool_bode_label = 'Gx_tool'
        elif measurement_case == 'case2_y':
            tool_bode_label = 'Gy_tool'
        
        # Generate Bode plots with both displacement and acceleration (always shown)
        plot_bode_diagram(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                         tf_accel_results['Gx'], tf_accel_results['Gy'], show_plot,
                         G_tool=tf_results.get('G_tool', None), 
                         G_tool_accel=tf_accel_results.get('G_tool', None),
                         tool_label=tool_bode_label)
        
        # Plot real part of transfer functions (always shown)
        plot_real_transfer_functions(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                                     tf_accel_results['Gx'], tf_accel_results['Gy'], show_plot,
                                     G_tool=tf_results.get('G_tool', None), 
                                     G_tool_accel=tf_accel_results.get('G_tool', None),
                                     tool_label=tool_bode_label)
        
        # Plot reconstructed time-domain signals (only if show_all_figures is True, skip for Welch method - X_full is None)
        if show_all_figures and fft_results.get('X_full') is not None:
            Tool_freq_full = fft_results.get('Tool_full', None)
            plot_reconstructed_time_signals(time, windowed_signals['u'], fft_results['X_full'], 
                                           fft_results['Y_full'], show_plot,
                                           Tool_freq=Tool_freq_full, tool_label=tool_label)
        
        # Save results
        saved_files = save_results(
            fft_results['frequencies'], 
            tf_results['Gx'], 
            tf_results['Gy'], 
            filepath, 
            save_files,
            metadata=metadata,
            measurement_case=measurement_case,
            Gx_accel=tf_accel_results['Gx'],
            Gy_accel=tf_accel_results['Gy'],
            G_tool=tf_results.get('G_tool', None),
            G_tool_accel=tf_accel_results.get('G_tool', None)
        )
        
        # Prepare return data
        result = {
            'frequencies': fft_results['frequencies'],
            'Gx': tf_results['Gx'],
            'Gy': tf_results['Gy'],
            'Gx_accel': tf_accel_results['Gx'],
            'Gy_accel': tf_accel_results['Gy'],
            'metadata': metadata,
            'saved_files': saved_files,
            'sample_rate': 1 / fft_results['dt'],
            'N_samples': fft_results['N'],
            'windows': windows_dict,
            'window_metadata': {
                'u': window_metadata_u,
                'x': window_metadata_x,
                'y': window_metadata_y
            }
        }
        
        # Add z window metadata if present
        if window_metadata_z is not None:
            result['window_metadata']['z'] = window_metadata_z
        
        # Add tool window metadata if present
        if window_metadata_tool is not None:
            result['window_metadata']['tool'] = window_metadata_tool
        
        # Add tool transfer function if present
        if 'G_tool' in tf_results:
            result['G_tool'] = tf_results['G_tool']
        if 'G_tool' in tf_accel_results:
            result['G_tool_accel'] = tf_accel_results['G_tool']
        
        return result
        
    except Exception as e:
        import traceback
        raise Exception(f"Error processing file {filepath}: {str(e)}") from e

# =============================================================================
# USER INTERFACE
# =============================================================================

def select_measurement_case(root=None):
    """
    Deprecated: interactive measurement-case selection.
    """
    raise RuntimeError(
        "Interactive selection has been removed. Configure analysis via JSON and run src/main.py."
    )

def infer_case2_direction(filepaths, root=None):
    """
    Infer Case 2 direction (x or y) from selected file(s) based on filename suffix.
    
    Parameters:
    -----------
    filepaths : list
        List of file paths to analyze
    root : Any, optional
        Deprecated legacy argument (no longer used for dialogs).
    
    Returns:
    --------
    str : Final measurement case identifier ('case2_x' or 'case2_y')
        Returns None if error occurs or directions are mixed
    
    Raises:
    -------
    ValueError : If files have mixed directions or cannot be parsed
    """
    if not filepaths:
        return None
    
    directions = []
    
    # Parse each file to extract direction
    for filepath in filepaths:
        try:
            metadata = parse_filename(filepath)
            direction = metadata['direction'].lower()
            if direction not in ['x', 'y']:
                raise ValueError(f"Unexpected direction '{direction}' in file: {filepath}")
            directions.append(direction)
        except Exception as e:
            # If parsing fails, show error and return None
            error_msg = f"Error parsing filename: {str(e)}\n\nFile: {os.path.basename(filepath)}"
            print(f"Error: {error_msg}")
            return None
    
    # Check if all directions are the same
    if len(set(directions)) > 1:
        # Mixed directions - show error
        error_msg = (
            "Selected files have mixed directions (both _x and _y).\n\n"
            "Please select files with the same direction suffix.\n\n"
            "Files selected:\n" + "\n".join([f"  - {os.path.basename(fp)} ({d})" 
                                            for fp, d in zip(filepaths, directions)])
        )
        print(f"Error: {error_msg}")
        return None
    
    # All files have the same direction
    direction = directions[0]
    return f'case2_{direction}'

def detect_case3_from_filename(filepaths, root=None):
    """
    Detect Case 3 from selected file(s) based on "XYZ" appearing in filename.
    
    Parameters:
    -----------
    filepaths : list
        List of file paths to analyze
    root : Any, optional
        Deprecated legacy argument (no longer used for dialogs).
    
    Returns:
    --------
    str : 'case3' if detected, None otherwise
    
    Raises:
    -------
    ValueError : If files have mixed cases (some with "XYZ", some without)
    """
    if not filepaths:
        return None
    
    has_xyz = []
    
    # Check each file for "XYZ" in filename (case-insensitive)
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        has_xyz.append('xyz' in filename.lower())
    
    # Check if all files have the same case
    if len(set(has_xyz)) > 1:
        # Mixed cases - show error
        xyz_files = [fp for fp, has in zip(filepaths, has_xyz) if has]
        non_xyz_files = [fp for fp, has in zip(filepaths, has_xyz) if not has]
        error_msg = (
            "Selected files have mixed measurement cases.\n\n"
            "Some files contain 'XYZ' in filename (Case 3), while others do not.\n\n"
            "Files with 'XYZ':\n" + "\n".join([f"  - {os.path.basename(fp)}" for fp in xyz_files]) + "\n\n"
            "Files without 'XYZ':\n" + "\n".join([f"  - {os.path.basename(fp)}" for fp in non_xyz_files])
        )
        print(f"Error: {error_msg}")
        return None
    
    # All files have the same case - return 'case3' if all have XYZ, None otherwise
    if all(has_xyz):
        return 'case3'
    else:
        return None

def wait_for_quit_and_close_plots():
    """
    Show all plots once at the end.

    This is intentionally simple: `plt.show()` blocks while figures are open and
    returns when the user closes the plot windows manually.
    """
    # If no figures were created, nothing to do.
    if len(plt.get_fignums()) == 0:
        return

    plt.show()

def main() -> None:
    """
    Deprecated interactive entry point.

    This project is moving toward `src/main.py` orchestration (and later a GUI).
    """
    raise RuntimeError(
        "Interactive FRF UI has been removed. Run `src/main.py` with a JSON config instead."
    )

