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
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import os
from pathlib import Path
import warnings
from scipy import signal

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
    
    # High-pass filter parameters
    highpass_cutoff_hz = 0.0001  # High-pass filter cutoff frequency (Hz)
    
    # Wiener filter parameters
    wiener_filter = True  # Toggle to enable/disable Wiener filter for channels A0-A5
    
    # Low-pass filter parameters (for analysis visualization only)
    lowpass_bandwidth_hz = 100.0  # Low-pass filter bandwidth (Hz) for analysis plots
    
    # Window parameters for raised cosine (Tukey) windowing
    use_raised_cosine_window = False  # Toggle to enable/disable raised cosine window
    window_pre_seconds = 0.05   # Seconds before peak force to start window
    window_post_seconds = 0.1   # Seconds after peak force to end window
    window_beta = 0.25          # Tukey window roll-off (0 to 1)
    
    # Welch method parameters for CSD-based transfer function estimation
    use_welch_csd = True       # Toggle to enable/disable Welch CSD method (default: True)
    welch_nperseg = 1000       # Length of each segment for Welch method (None = scipy default: 256 or signal length)
    welch_noverlap = None      # Number of points to overlap between segments (None = scipy default: nperseg // 2)
    welch_window = 'hann'      # Window type for Welch method
    welch_scaling = 'density'   # PSD scaling: 'density' or 'spectrum'

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
        'highpass_cutoff_hz': highpass_cutoff_hz,
        'wiener_filter': wiener_filter,
        'lowpass_bandwidth_hz': lowpass_bandwidth_hz,
        'use_raised_cosine_window': use_raised_cosine_window,
        'window_pre_seconds': window_pre_seconds,
        'window_post_seconds': window_post_seconds,
        'window_beta': window_beta,
        'use_welch_csd': use_welch_csd,
        'welch_nperseg': welch_nperseg,
        'welch_noverlap': welch_noverlap,
        'welch_window': welch_window,
        'welch_scaling': welch_scaling
    }

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
    
    The function slides a window across the signal, calculates variance at each position,
    finds the window with minimum variance (assumed to be the quietest/most stable region),
    calculates the mean of that window, and subtracts it from the entire signal.
    
    Parameters:
    -----------
    signal : array_like
        Input signal array
    window_size : int
        Size of the sliding window in samples (default: 1000)
        
    Returns:
    --------
    tuple : (bias_corrected_signal, min_var_idx, window_size)
        bias_corrected_signal : array_like
            Bias-corrected signal
        min_var_idx : int
            Starting index of the window with minimum variance
        window_size : int
            Size of the window used (may differ from input if signal is shorter)
    
    Notes:
    ------
    If the signal length is less than the window size, the mean of the entire signal
    is used for bias removal (with a warning), and min_var_idx will be 0.
    """
    signal = np.asarray(signal)
    signal_length = len(signal)
    
    # Handle case where signal is shorter than window
    if signal_length < window_size:
        print(f"Warning: Signal length ({signal_length}) is less than window size ({window_size}). "
              f"Using entire signal for bias calculation.")
        bias = np.mean(signal)
        actual_window_size = signal_length
        min_var_idx = 0
        return signal - bias, min_var_idx, actual_window_size
    
    # Calculate variance for each window position
    num_windows = signal_length - window_size + 1
    variances = np.zeros(num_windows)
    
    for i in range(num_windows):
        window = signal[i:i + window_size]
        variances[i] = np.var(window)
    
    # Find window with minimum variance
    min_var_idx = np.argmin(variances)
    
    # Calculate mean of the minimal variance window
    bias = np.mean(signal[min_var_idx:min_var_idx + window_size])
    
    # Remove bias from entire signal
    return signal - bias, min_var_idx, window_size

def calculate_noise_power(signal, window_start_idx, window_size):
    """
    Calculate noise power from a specific window using scipy's default method.
    
    This replicates scipy.signal.wiener's default noise calculation, which estimates
    noise power as the average of local variance over the specified window.
    
    Parameters:
    -----------
    signal : array_like
        Input signal array (should be bias-corrected)
    window_start_idx : int
        Starting index of the window for noise calculation
    window_size : int
        Size of the window in samples
        
    Returns:
    --------
    float : Noise power (average local variance)
    """
    signal = np.asarray(signal)
    signal_length = len(signal)
    
    # Ensure window indices are valid
    window_start_idx = max(0, min(window_start_idx, signal_length - 1))
    window_end_idx = min(window_start_idx + window_size, signal_length)
    actual_window_size = window_end_idx - window_start_idx
    
    if actual_window_size < 1:
        # Fallback: use variance of entire signal
        return np.var(signal)
    
    # Extract the window
    window_data = signal[window_start_idx:window_end_idx]
    
    # Calculate local variance in the window
    # scipy's default method: average of local variance
    # For a single window, this is just the variance of that window
    noise_power = np.var(window_data)
    
    return noise_power

def apply_wiener_filter(data, noise_power, mysize):
    """
    Apply Wiener filter to the data using scipy.signal.wiener.
    
    Parameters:
    -----------
    data : array_like
        Input signal array
    noise_power : float
        Noise power parameter for Wiener filter
    mysize : int
        Size of the Wiener filter window (should match bias window size)
        
    Returns:
    --------
    array_like : Wiener filtered signal
    """
    return signal.wiener(data, mysize=mysize, noise=noise_power)

def apply_highpass_filter(data, cutoff_hz, fs):
    """
    Apply 4th-order Bessel high-pass filter to the data.
    
    Parameters:
    -----------
    data : array_like
        Input signal array
    cutoff_hz : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    array_like : High-pass filtered signal
    """
    # Design 4th-order Bessel high-pass filter
    b, a = signal.bessel(N=4, Wn=cutoff_hz, btype='high', analog=False, fs=fs)
    
    # Apply zero-phase filtering using filtfilt
    filtered_data = signal.filtfilt(b, a, data)
    
    # return filtered_data

    return data # no high-pass filter

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
        Measurement case identifier: 'case1', 'case2_x', or 'case2_y'
        
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
        x_tool = (A5 - A4) * r_a
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

# =============================================================================
# FREQUENCY DOMAIN ANALYSIS
# =============================================================================

def perform_fft_analysis(time, u, x, y, dt, tool_signal=None):
    """
    Perform FFT and return frequencies, X, Y, U in frequency domain.
    Note: x and y are accelerations and will be double-integrated to displacements.
    Also returns acceleration FFT results for comparison.
    Optionally processes tool acceleration signal if provided.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u : array_like
        Force signal (time domain)
    x, y : array_like
        Acceleration signals in time domain (m/s²)
    dt : float
        Sample time in seconds
    tool_signal : array_like, optional
        Tool acceleration signal in time domain (m/s²) - x_tool or y_tool
        
    Returns:
    --------
    dict : Dictionary containing frequency array and FFT results
        Includes both displacement (X, Y) and acceleration (X_accel, Y_accel) results
        Optionally includes Tool and Tool_accel if tool_signal is provided
    """
    N = len(time)
    fs = 1 / dt  # Sampling frequency
    
    # Perform FFT
    U = np.fft.fft(u)
    X_accel = np.fft.fft(x)  # FFT of acceleration
    Y_accel = np.fft.fft(y)  # FFT of acceleration
    
    # Generate frequency array
    frequencies = np.fft.fftfreq(N, dt)
    omega = 2 * np.pi * frequencies  # Angular frequency (rad/s)
    
    # Double integration using discrete transfer function
    # H(e^(jω)) = dt²/(1-e^(-jω))²
    z = np.exp(1j * omega * dt)
    H = dt**2 / (1 - z**(-1))**2
    # H = -1 / omega ** 2

    # Apply transfer function to convert acceleration to displacement
    X = np.zeros_like(X_accel, dtype=complex)
    Y = np.zeros_like(Y_accel, dtype=complex)
    
    # Avoid division by zero at DC (frequency = 0)
    X[1:] = X_accel[1:] * H[1:]
    Y[1:] = Y_accel[1:] * H[1:]
    
    # DC component (zero frequency) set to zero
    X[0] = 0
    Y[0] = 0
    
    # Process tool signal if provided
    if tool_signal is not None:
        Tool_accel = np.fft.fft(tool_signal)
        Tool = np.zeros_like(Tool_accel, dtype=complex)
        Tool[1:] = Tool_accel[1:] * H[1:]
        Tool[0] = 0
    else:
        Tool_accel = None
        Tool = None
    
    # Use only positive frequencies (including DC)
    positive_freq_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_freq_mask]
    U_pos = U[positive_freq_mask]
    X_pos = X[positive_freq_mask]
    Y_pos = Y[positive_freq_mask]
    X_accel_pos = X_accel[positive_freq_mask]
    Y_accel_pos = Y_accel[positive_freq_mask]
    
    result = {
        'frequencies': frequencies_pos,
        'U': U_pos,
        'X': X_pos,
        'Y': Y_pos,
        'X_accel': X_accel_pos,
        'Y_accel': Y_accel_pos,
        'X_full': X,  # Full spectrum (positive and negative frequencies)
        'Y_full': Y,  # Full spectrum (positive and negative frequencies)
        'N': N,
        'dt': dt
    }
    
    # Add tool results if available
    if tool_signal is not None:
        Tool_pos = Tool[positive_freq_mask]
        Tool_accel_pos = Tool_accel[positive_freq_mask]
        result['Tool'] = Tool_pos
        result['Tool_accel'] = Tool_accel_pos
        result['Tool_full'] = Tool  # Full spectrum
    
    return result

def calculate_csd_psd_terms(time, u, x, y, dt, params, tool_signal=None):
    """
    Calculate cross-spectral density (CSD) and power spectral density (PSD) terms
    for a single measurement file using Welch method.
    
    This function extracts the CSD/PSD calculation logic for use in multi-file averaging.
    It returns the raw spectral density terms without calculating transfer functions or coherence.
    
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
    dict : Dictionary containing CSD/PSD terms
        'Gxu_accel': Cross-spectral density between u and x (complex)
        'Gyu_accel': Cross-spectral density between u and y (complex)
        'Guu': Power spectral density of u (real)
        'Gxx_accel': Power spectral density of x (real)
        'Gyy_accel': Power spectral density of y (real)
        'frequencies': Frequency array (Hz)
        'dt': Sample time (s)
        'N': Number of samples
        'G_tool_u_accel': Cross-spectral density between u and tool (complex, optional)
        'G_tool_tool_accel': Power spectral density of tool (real, optional)
        'frequencies_tool': Frequency array for tool (Hz, optional)
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
    
    result = {
        'Gxu_accel': Gxu_accel,
        'Gyu_accel': Gyu_accel,
        'Guu': Guu,
        'Gxx_accel': Gxx_accel,
        'Gyy_accel': Gyy_accel,
        'frequencies': frequencies,
        'dt': dt,
        'N': N
    }
    
    # Process tool signal if provided
    if tool_signal is not None:
        frequencies_tool, G_tool_u_accel = signal.csd(u, tool_signal, fs=fs, nperseg=nperseg, 
                                                      noverlap=noverlap, window=window, 
                                                      scaling=scaling, return_onesided=True)
        frequencies_tool, G_tool_tool_accel = signal.welch(tool_signal, fs=fs, nperseg=nperseg, 
                                                           noverlap=noverlap, window=window, 
                                                           scaling=scaling, return_onesided=True)
        result['G_tool_u_accel'] = G_tool_u_accel
        result['G_tool_tool_accel'] = G_tool_tool_accel
        result['frequencies_tool'] = frequencies_tool
    
    return result

def average_csd_psd_terms(csd_psd_list):
    """
    Average cross-spectral density (CSD) and power spectral density (PSD) terms
    across multiple measurement files.
    
    Parameters:
    -----------
    csd_psd_list : list
        List of CSD/PSD dictionaries from calculate_csd_psd_terms()
        
    Returns:
    --------
    dict : Averaged CSD/PSD dictionary with same structure as input
    """
    if not csd_psd_list:
        raise ValueError("Cannot average empty list of CSD/PSD terms")
    
    if len(csd_psd_list) == 1:
        # Single file, return as-is
        return csd_psd_list[0].copy()
    
    # Check that all dictionaries have the same frequency arrays
    first_freq = csd_psd_list[0]['frequencies']
    for i, csd_psd in enumerate(csd_psd_list[1:], 1):
        if not np.allclose(csd_psd['frequencies'], first_freq, rtol=1e-10):
            raise ValueError(f"Frequency arrays differ between files. File 0 and file {i} have different frequency arrays.")
    
    # Check that all have same dt
    first_dt = csd_psd_list[0]['dt']
    for i, csd_psd in enumerate(csd_psd_list[1:], 1):
        if not np.isclose(csd_psd['dt'], first_dt, rtol=1e-10):
            raise ValueError(f"Sample times differ between files. File 0 has dt={first_dt}, file {i} has dt={csd_psd['dt']}")
    
    # Initialize averaged dictionary
    averaged = {
        'frequencies': first_freq.copy(),
        'dt': first_dt,
        'N': csd_psd_list[0]['N']  # Use first file's N (should be same after truncation)
    }
    
    # Average complex CSD terms (element-wise)
    n_files = len(csd_psd_list)
    averaged['Gxu_accel'] = np.mean([csd_psd['Gxu_accel'] for csd_psd in csd_psd_list], axis=0)
    averaged['Gyu_accel'] = np.mean([csd_psd['Gyu_accel'] for csd_psd in csd_psd_list], axis=0)
    
    # Average real PSD terms (element-wise)
    averaged['Guu'] = np.mean([csd_psd['Guu'] for csd_psd in csd_psd_list], axis=0)
    averaged['Gxx_accel'] = np.mean([csd_psd['Gxx_accel'] for csd_psd in csd_psd_list], axis=0)
    averaged['Gyy_accel'] = np.mean([csd_psd['Gyy_accel'] for csd_psd in csd_psd_list], axis=0)
    
    # Handle tool signals if present
    has_tool = all('G_tool_u_accel' in csd_psd for csd_psd in csd_psd_list)
    if has_tool:
        # Check that all tool frequency arrays match
        first_tool_freq = csd_psd_list[0]['frequencies_tool']
        for i, csd_psd in enumerate(csd_psd_list[1:], 1):
            if 'frequencies_tool' not in csd_psd:
                raise ValueError(f"File {i} missing tool frequency array")
            if not np.allclose(csd_psd['frequencies_tool'], first_tool_freq, rtol=1e-10):
                raise ValueError(f"Tool frequency arrays differ between files. File 0 and file {i} have different tool frequency arrays.")
        
        averaged['frequencies_tool'] = first_tool_freq.copy()
        averaged['G_tool_u_accel'] = np.mean([csd_psd['G_tool_u_accel'] for csd_psd in csd_psd_list], axis=0)
        averaged['G_tool_tool_accel'] = np.mean([csd_psd['G_tool_tool_accel'] for csd_psd in csd_psd_list], axis=0)
    
    return averaged

def calculate_tf_from_averaged_csd_psd(averaged_csd_psd, params, dt):
    """
    Calculate transfer functions and coherence from averaged CSD/PSD terms.
    
    Parameters:
    -----------
    averaged_csd_psd : dict
        Averaged CSD/PSD dictionary from average_csd_psd_terms()
    params : dict
        System parameters dictionary
    dt : float
        Sample time in seconds
        
    Returns:
    --------
    dict : Dictionary containing frequency array and transfer function results
        Same structure as perform_welch_csd_analysis() for compatibility
    """
    frequencies = averaged_csd_psd['frequencies']
    N = averaged_csd_psd['N']
    fs = 1 / dt
    
    # Extract averaged CSD/PSD terms
    Gxu_accel = averaged_csd_psd['Gxu_accel']
    Gyu_accel = averaged_csd_psd['Gyu_accel']
    Guu = averaged_csd_psd['Guu']
    Gxx_accel = averaged_csd_psd['Gxx_accel']
    Gyy_accel = averaged_csd_psd['Gyy_accel']
    
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
    
    # Process tool signal if present
    if 'G_tool_u_accel' in averaged_csd_psd:
        G_tool_u_accel = averaged_csd_psd['G_tool_u_accel']
        G_tool_tool_accel = averaged_csd_psd['G_tool_tool_accel']
        frequencies_tool = averaged_csd_psd['frequencies_tool']
        
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
    if Tool is not None:
        result['Tool'] = Tool
        result['Tool_accel'] = Tool_accel
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

def calculate_transfer_functions(X, Y, U, threshold=1e-12, Tool=None):
    """
    Calculate Gx = X/U and Gy = Y/U transfer functions.
    Optionally calculates G_tool = Tool/U if Tool is provided.
    
    Parameters:
    -----------
    X, Y, U : array_like
        FFT results in frequency domain
    threshold : float
        Minimum magnitude threshold to avoid division by zero
    Tool : array_like, optional
        Tool displacement FFT results in frequency domain
        
    Returns:
    --------
    dict : Dictionary containing transfer functions Gx and Gy
        Optionally includes G_tool if Tool is provided
    """
    # Avoid division by zero
    U_magnitude = np.abs(U)
    valid_mask = U_magnitude > threshold
    
    # Initialize arrays
    Gx = np.zeros_like(X, dtype=complex)
    Gy = np.zeros_like(Y, dtype=complex)
    
    # Calculate transfer functions where U is not zero
    Gx[valid_mask] = X[valid_mask] / U[valid_mask]
    Gy[valid_mask] = Y[valid_mask] / U[valid_mask]
    
    # Set invalid points to NaN
    Gx[~valid_mask] = np.nan
    Gy[~valid_mask] = np.nan
    
    result = {
        'Gx': Gx,
        'Gy': Gy,
        'valid_mask': valid_mask
    }
    
    # Calculate tool transfer function if provided
    if Tool is not None:
        G_tool = np.zeros_like(Tool, dtype=complex)
        G_tool[valid_mask] = Tool[valid_mask] / U[valid_mask]
        G_tool[~valid_mask] = np.nan
        result['G_tool'] = G_tool
    
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
    plt.show()

def plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot=True):
    """
    Plot comparison of original, bias-corrected, and high-pass filtered signals for A0-A7.
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
        Dictionary with keys 'A0' through 'A7' containing high-pass filtered signals
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
        ax.plot(time, filtered_signals[name], 'g-', linewidth=1, alpha=0.7, label='High-pass Filtered')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Signal Comparison - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

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
    plt.show()

def plot_wiener_filtered_signals(time, bias_corrected_signals, wiener_filtered_signals, show_plot=True):
    """
    Plot comparison of bias-corrected and Wiener filtered signals for A0-A5.
    Creates a single figure with 2 rows and 3 columns (6 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    bias_corrected_signals : dict
        Dictionary with keys 'A0' through 'A5' containing bias-corrected signals
    wiener_filtered_signals : dict
        Dictionary with keys 'A0' through 'A5' containing Wiener filtered signals
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    signal_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
    
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
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
    plt.show()

def plot_time_domain_signals(time, u, x, y, show_plot=True, window=None, tool_accel=None, tool_label=None):
    """
    Plot time-domain signals u, x, y in a single figure with synchronized x-axes.
    Optionally includes tool acceleration signal as 4th subplot.
    
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
    window : array_like, optional
        Window function to overlay on plots
    tool_accel : array_like, optional
        Tool acceleration signal (m/s²) - x_tool or y_tool
    tool_label : str, optional
        Label for tool acceleration (e.g., 'x_tool' or 'y_tool')
    """
    if not show_plot:
        return
    
    # Determine number of subplots
    n_plots = 4 if tool_accel is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 10 + 2.5 * (n_plots - 3)), sharex=True)
    
    # Plot force u
    axes[0].plot(time, u, 'b-', linewidth=1, label='u (windowed)')
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title('Force Input u(t)')
    axes[0].grid(True, alpha=0.3)
    if window is not None:
        scale = np.max(np.abs(u))
        scale = scale if scale > 0 else 1.0
        axes[0].plot(time, window * scale, 'k--', linewidth=1, label='Window')
        axes[0].legend(fontsize=8)
    
    # Plot x acceleration
    axes[1].plot(time, x, 'r-', linewidth=1, label='x (windowed)')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('X-Direction Acceleration x(t)')
    axes[1].grid(True, alpha=0.3)
    if window is not None:
        scale = np.max(np.abs(x))
        scale = scale if scale > 0 else 1.0
        axes[1].plot(time, window * scale, 'k--', linewidth=1, label='Window')
        axes[1].legend(fontsize=8)
    
    # Plot y acceleration
    axes[2].plot(time, y, 'g-', linewidth=1, label='y (windowed)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_title('Y-Direction Acceleration y(t)')
    axes[2].grid(True, alpha=0.3)
    if window is not None:
        scale = np.max(np.abs(y))
        scale = scale if scale > 0 else 1.0
        axes[2].plot(time, window * scale, 'k--', linewidth=1, label='Window')
        axes[2].legend(fontsize=8)
    
    # Plot tool acceleration if provided
    if tool_accel is not None:
        axes[3].plot(time, tool_accel, 'm-', linewidth=1, label=f'{tool_label} (windowed)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Acceleration (m/s²)')
        title_label = tool_label.replace('_', ' ').title() if tool_label else 'Tool Acceleration'
        axes[3].set_title(f'Tool Acceleration {title_label}(t)')
        axes[3].grid(True, alpha=0.3)
        if window is not None:
            scale = np.max(np.abs(tool_accel))
            scale = scale if scale > 0 else 1.0
            axes[3].plot(time, window * scale, 'k--', linewidth=1, label='Window')
            axes[3].legend(fontsize=8)
    else:
        axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

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
        valid_mask_accel = np.isfinite(Gx_accel_mag) & np.isfinite(Gy_accel_mag) & np.isfinite(Gx_accel_phase) & np.isfinite(Gy_accel_phase)
        valid_freq_accel = frequencies[valid_mask_accel]
        valid_Gx_accel_mag = Gx_accel_mag[valid_mask_accel]
        valid_Gy_accel_mag = Gy_accel_mag[valid_mask_accel]
        valid_Gx_accel_phase = Gx_accel_phase[valid_mask_accel]
        valid_Gy_accel_phase = Gy_accel_phase[valid_mask_accel]
    
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
    
    # Skip DC component for acceleration data
    if Gx_accel is not None and Gy_accel is not None:
        if len(valid_freq_accel) > 1 and valid_freq_accel[0] == 0:
            valid_freq_accel = valid_freq_accel[1:]
            valid_Gx_accel_mag = valid_Gx_accel_mag[1:]
            valid_Gy_accel_mag = valid_Gy_accel_mag[1:]
            valid_Gx_accel_phase = valid_Gx_accel_phase[1:]
            valid_Gy_accel_phase = valid_Gy_accel_phase[1:]
    
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
    if Gx_accel is not None:
        ax[0, 0].loglog(valid_freq_accel, valid_Gx_accel_mag, 'b--', linewidth=2, alpha=0.7, label='Gx (Acceleration)')
    ax[0, 0].set_ylabel('Compliance (m/N or m/s²/N)')
    ax[0, 0].set_title('Compliance Gx (X/U) - Magnitude')
    ax[0, 0].grid(True, which="both", ls="-", alpha=0.3)
    ax[0, 0].legend()
    
    # Plot Gy magnitude (top-middle)
    ax[0, 1].loglog(valid_freq, valid_Gy_mag, 'r-', linewidth=2, label='Gy (Displacement)')
    if Gy_accel is not None:
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
    if Gx_accel is not None:
        ax[1, 0].semilogx(valid_freq_accel, valid_Gx_accel_phase, 'b--', linewidth=2, alpha=0.7, label='Gx (Acceleration)')
    ax[1, 0].set_xlabel('Frequency (Hz)')
    ax[1, 0].set_ylabel('Phase (degrees)')
    ax[1, 0].set_title('Compliance Gx (X/U) - Phase')
    ax[1, 0].grid(True, which="both", ls="-", alpha=0.3)
    ax[1, 0].legend()
    # Removed fixed y-limits to allow unwrapped phase to display properly
    
    # Plot Gy phase (bottom-middle)
    ax[1, 1].semilogx(valid_freq, valid_Gy_phase, 'r-', linewidth=2, label='Gy (Displacement)')
    if Gy_accel is not None:
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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

# =============================================================================
# FILE OUTPUT
# =============================================================================

def generate_output_filename(original_filepath, suffix):
    """
    Generate output filename based on input filename and suffix (Gx or Gy).
    
    Parameters:
    -----------
    original_filepath : str
        Path to original CSV file
    suffix : str
        Suffix to replace direction indicator ('Gx' or 'Gy')
        
    Returns:
    --------
    str : Generated output filename
    """
    filename = os.path.basename(original_filepath)
    
    # Replace direction indicator with suffix
    # Pattern: ..._x.csv or ..._y.csv (with optional suffix) -> ..._Gx.csv or ..._Gy.csv (preserving suffix)
    # Make pattern case-insensitive to handle both .csv and .CSV extensions
    pattern = r'(_[xy])(_[^.]*)?(\.csv)$'
    new_filename = re.sub(pattern, f'_{suffix}\\2\\3', filename, flags=re.IGNORECASE)
    
    # Get directory of original file
    directory = os.path.dirname(original_filepath)
    
    return os.path.join(directory, new_filename)

def save_results(frequencies, Gx, Gy, original_filepath, save_files=True):
    """
    Save frequency response data to CSV files.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    Gx, Gy : array_like
        Transfer functions (complex)
    original_filepath : str
        Path to original CSV file
    save_files : bool
        Whether to save files
        
    Returns:
    --------
    list : List of saved file paths
    """
    if not save_files:
        return []
    
    saved_files = []
    
    # Calculate magnitudes
    Gx_mag = np.abs(Gx)
    Gy_mag = np.abs(Gy)
    
    # Create DataFrame with all data
    df = pd.DataFrame({
        'Frequency_Hz': frequencies,
        'Gx_m_per_N': Gx_mag,
        'Gy_m_per_N': Gy_mag
    })
    
    # Save Gx file
    gx_filename = generate_output_filename(original_filepath, 'Gx')
    df_gx = df[['Frequency_Hz', 'Gx_m_per_N']].copy()
    df_gx.to_csv(gx_filename, index=False, float_format='%.6e')
    saved_files.append(gx_filename)
    
    # Save Gy file
    gy_filename = generate_output_filename(original_filepath, 'Gy')
    df_gy = df[['Frequency_Hz', 'Gy_m_per_N']].copy()
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
        Measurement case identifier: 'case1', 'case2_x', or 'case2_y'
    show_plot : bool
        Whether to display plots (controlled by show_all_figures for which plots)
    save_files : bool
        Whether to save output CSV files
        
    Returns:
    --------
    dict : Contains frequencies, Gx, Gy, and metadata
        Optionally includes G_tool for Case 2
    """
    try:
        # Load system parameters
        params = get_system_parameters()
        
        # Force Welch CSD method for multiple files
        params['use_welch_csd'] = True
        
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
        
        # Process each file and calculate CSD/PSD terms
        csd_psd_list = []
        
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
            A6, _, _ = remove_bias(A6, window_size)
            A7, _, _ = remove_bias(A7, window_size)
            
            # Apply Wiener filter to A0-A5 if enabled, otherwise use high-pass filter
            cutoff_hz = params['highpass_cutoff_hz']
            use_wiener = params.get('wiener_filter', True)
            
            # Store original signals before filtering for noise calculation
            original_A0 = A0.copy()
            original_A1 = A1.copy()
            original_A2 = A2.copy()
            original_A3 = A3.copy()
            original_A4 = A4.copy()
            original_A5 = A5.copy()
            
            if use_wiener:
                # Calculate noise power for each channel A0-A5 from bias window
                noise_A0 = calculate_noise_power(original_A0, bias_idx_A0, actual_window_A0)
                noise_A1 = calculate_noise_power(original_A1, bias_idx_A1, actual_window_A1)
                noise_A2 = calculate_noise_power(original_A2, bias_idx_A2, actual_window_A2)
                noise_A3 = calculate_noise_power(original_A3, bias_idx_A3, actual_window_A3)
                noise_A4 = calculate_noise_power(original_A4, bias_idx_A4, actual_window_A4)
                noise_A5 = calculate_noise_power(original_A5, bias_idx_A5, actual_window_A5)
                
                # Apply Wiener filter to A0-A5
                A0 = apply_wiener_filter(A0, noise_A0, actual_window_A0)
                A1 = apply_wiener_filter(A1, noise_A1, actual_window_A1)
                A2 = apply_wiener_filter(A2, noise_A2, actual_window_A2)
                A3 = apply_wiener_filter(A3, noise_A3, actual_window_A3)
                A4 = apply_wiener_filter(A4, noise_A4, actual_window_A4)
                A5 = apply_wiener_filter(A5, noise_A5, actual_window_A5)
            else:
                # Apply high-pass filter to A0-A5
                A0 = apply_highpass_filter(A0, cutoff_hz, fs)
                A1 = apply_highpass_filter(A1, cutoff_hz, fs)
                A2 = apply_highpass_filter(A2, cutoff_hz, fs)
                A3 = apply_highpass_filter(A3, cutoff_hz, fs)
                A4 = apply_highpass_filter(A4, cutoff_hz, fs)
                A5 = apply_highpass_filter(A5, cutoff_hz, fs)
            
            # Apply high-pass filter to A6, A7 (always use high-pass for these channels)
            A6 = apply_highpass_filter(A6, cutoff_hz, fs)
            A7 = apply_highpass_filter(A7, cutoff_hz, fs)
            
            # Calculate derived signals with measurement case
            signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params, measurement_case)
            
            # Build window around primary force event (raised cosine or no window)
            if params['use_raised_cosine_window']:
                window, window_metadata = create_raised_cosine_window(
                    time,
                    signals['u'],
                    params['window_pre_seconds'],
                    params['window_post_seconds'],
                    params['window_beta'],
                    dt
                )
            else:
                # No windowing: use all-ones window (no effect on signals)
                window = np.ones_like(time, dtype=float)
                window_metadata = {
                    'peak_time': time[len(time) // 2],
                    'start_time': time[0],
                    'end_time': time[-1],
                    'peak_index': len(time) // 2,
                    'start_index': 0,
                    'end_index': len(time)
                }
            
            # Apply window to derived signals used for FRF calculations
            windowed_signals = {
                'u': signals['u'] * window,
                'x': signals['x'] * window,
                'y': signals['y'] * window
            }
            
            # Extract tool signal if present and apply window
            tool_signal = None
            if measurement_case == 'case2_x':
                tool_signal = signals.get('x_tool', None)
                if tool_signal is not None:
                    tool_signal = tool_signal * window
            elif measurement_case == 'case2_y':
                tool_signal = signals.get('y_tool', None)
                if tool_signal is not None:
                    tool_signal = tool_signal * window
            
            # Calculate CSD/PSD terms for this file
            csd_psd = calculate_csd_psd_terms(time, windowed_signals['u'], windowed_signals['x'], 
                                              windowed_signals['y'], dt, params, tool_signal=tool_signal)
            csd_psd_list.append(csd_psd)
            
            print(f"  Processed file {i+1}/{len(filepaths)}: {os.path.basename(data['filepath'])}")
        
        # Average CSD/PSD terms across all files
        print("Averaging CSD/PSD terms...")
        averaged_csd_psd = average_csd_psd_terms(csd_psd_list)
        
        # Plot CSD/PSD terms from all files with averaged terms overlaid
        if show_plot:
            print("Plotting CSD/PSD terms from individual files with averaged terms...")
            plot_csd_psd_before_averaging(csd_psd_list, filepaths, averaged_csd_psd=averaged_csd_psd, show_plot=show_plot)
        
        # Calculate transfer functions and coherence from averaged terms
        print("Calculating transfer functions from averaged CSD/PSD...")
        fft_results = calculate_tf_from_averaged_csd_psd(averaged_csd_psd, params, dt)
        
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
        saved_files = save_results(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                                 filepaths[0], save_files)
        
        # Prepare return data
        result = {
            'frequencies': fft_results['frequencies'],
            'Gx': tf_results['Gx'],
            'Gy': tf_results['Gy'],
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
        
        return result
        
    except Exception as e:
        raise Exception(f"Error processing multiple files: {str(e)}")

def process_measurement_file(filepath, measurement_case='case1', show_plot=True, save_files=True):
    """
    Main function to process a single measurement file.
    
    Parameters:
    -----------
    filepath : str
        Path to input CSV file
    measurement_case : str
        Measurement case identifier: 'case1', 'case2_x', or 'case2_y'
    show_plot : bool
        Whether to display Bode plots
    save_files : bool
        Whether to save output CSV files
        
    Returns:
    --------
    dict : Contains frequencies, Gx, Gy, and metadata
        Optionally includes G_tool for Case 2
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
        # For A0-A5: capture window location for noise calculation
        A0, bias_idx_A0, actual_window_A0 = remove_bias(A0, window_size)
        A1, bias_idx_A1, actual_window_A1 = remove_bias(A1, window_size)
        A2, bias_idx_A2, actual_window_A2 = remove_bias(A2, window_size)
        A3, bias_idx_A3, actual_window_A3 = remove_bias(A3, window_size)
        A4, bias_idx_A4, actual_window_A4 = remove_bias(A4, window_size)
        A5, bias_idx_A5, actual_window_A5 = remove_bias(A5, window_size)
        # For A6-A7: keep simple return (no Wiener filter needed)
        A6, _, _ = remove_bias(A6, window_size)
        A7, _, _ = remove_bias(A7, window_size)
        
        # Store corrected signals for comparison plots
        corrected_signals = {
            'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
            'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
        }
        
        # Apply Wiener filter to A0-A5 if enabled, otherwise use high-pass filter
        cutoff_hz = params['highpass_cutoff_hz']
        use_wiener = params.get('wiener_filter', True)
        
        # Store original signals before filtering for noise calculation
        original_A0 = A0.copy()
        original_A1 = A1.copy()
        original_A2 = A2.copy()
        original_A3 = A3.copy()
        original_A4 = A4.copy()
        original_A5 = A5.copy()
        
        if use_wiener:
            # Calculate noise power for each channel A0-A5 from bias window
            noise_A0 = calculate_noise_power(original_A0, bias_idx_A0, actual_window_A0)
            noise_A1 = calculate_noise_power(original_A1, bias_idx_A1, actual_window_A1)
            noise_A2 = calculate_noise_power(original_A2, bias_idx_A2, actual_window_A2)
            noise_A3 = calculate_noise_power(original_A3, bias_idx_A3, actual_window_A3)
            noise_A4 = calculate_noise_power(original_A4, bias_idx_A4, actual_window_A4)
            noise_A5 = calculate_noise_power(original_A5, bias_idx_A5, actual_window_A5)
            
            # Apply Wiener filter to A0-A5
            A0 = apply_wiener_filter(A0, noise_A0, actual_window_A0)
            A1 = apply_wiener_filter(A1, noise_A1, actual_window_A1)
            A2 = apply_wiener_filter(A2, noise_A2, actual_window_A2)
            A3 = apply_wiener_filter(A3, noise_A3, actual_window_A3)
            A4 = apply_wiener_filter(A4, noise_A4, actual_window_A4)
            A5 = apply_wiener_filter(A5, noise_A5, actual_window_A5)
        else:
            # Apply high-pass filter to A0-A5
            A0 = apply_highpass_filter(A0, cutoff_hz, fs)
            A1 = apply_highpass_filter(A1, cutoff_hz, fs)
            A2 = apply_highpass_filter(A2, cutoff_hz, fs)
            A3 = apply_highpass_filter(A3, cutoff_hz, fs)
            A4 = apply_highpass_filter(A4, cutoff_hz, fs)
            A5 = apply_highpass_filter(A5, cutoff_hz, fs)
        
        # Apply high-pass filter to A6, A7 (always use high-pass for these channels)
        A6 = apply_highpass_filter(A6, cutoff_hz, fs)
        A7 = apply_highpass_filter(A7, cutoff_hz, fs)
        
        # Store filtered signals for comparison plots
        filtered_signals = {
            'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
            'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
        }
        
        # Store Wiener filtered signals separately for plotting (if Wiener was applied)
        wiener_filtered_signals = None
        if use_wiener:
            wiener_filtered_signals = {
                'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
                'A4': A4, 'A5': A5
            }
        
        # Show signal comparison plots (only if show_all_figures is True)
        if show_all_figures:
            plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot)
        
        # Apply low-pass filter to bias-corrected signals for analysis visualization
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
        
        # Show low-pass filtered signals plot (only if show_all_figures is True)
        if show_all_figures:
            plot_lowpass_filtered_signals(time, corrected_signals, lowpass_filtered_signals, 
                                         lowpass_bandwidth_hz, show_plot)
        
        # Show Wiener filtered signals plot (if Wiener filter was applied, only if show_all_figures is True)
        if show_all_figures and use_wiener and wiener_filtered_signals is not None:
            # Extract only A0-A5 from corrected_signals for comparison
            corrected_signals_A0_A5 = {
                'A0': corrected_signals['A0'],
                'A1': corrected_signals['A1'],
                'A2': corrected_signals['A2'],
                'A3': corrected_signals['A3'],
                'A4': corrected_signals['A4'],
                'A5': corrected_signals['A5']
            }
            plot_wiener_filtered_signals(time, corrected_signals_A0_A5, wiener_filtered_signals, show_plot)
        
        # Calculate derived signals with measurement case
        signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params, measurement_case)

        # Build window around primary force event (raised cosine or no window)
        if params['use_raised_cosine_window']:
            # Build raised cosine window around primary force event
            window, window_metadata = create_raised_cosine_window(
                time,
                signals['u'],
                params['window_pre_seconds'],
                params['window_post_seconds'],
                params['window_beta'],
                dt
            )
        else:
            # No windowing: use all-ones window (no effect on signals)
            window = np.ones_like(time, dtype=float)
            window_metadata = {
                'peak_time': time[len(time) // 2],  # Approximate midpoint
                'start_time': time[0],
                'end_time': time[-1],
                'peak_index': len(time) // 2,
                'start_index': 0,
                'end_index': len(time)
            }

        # Apply window to derived signals used for FRF calculations
        windowed_signals = {
            'u': signals['u'] * window,
            'x': signals['x'] * window,
            'y': signals['y'] * window
        }
        
        # Extract tool signal if present and apply window
        tool_signal = None
        tool_label = None
        if measurement_case == 'case2_x':
            tool_signal = signals.get('x_tool', None)
            if tool_signal is not None:
                tool_signal = tool_signal * window
                tool_label = 'x_tool'
        elif measurement_case == 'case2_y':
            tool_signal = signals.get('y_tool', None)
            if tool_signal is not None:
                tool_signal = tool_signal * window
                tool_label = 'y_tool'
        
        # Plot time-domain signals with window overlay (only if show_all_figures is True)
        if show_all_figures:
            plot_time_domain_signals(time, windowed_signals['u'], windowed_signals['x'], 
                                    windowed_signals['y'], show_plot, window=window,
                                    tool_accel=tool_signal, tool_label=tool_label)
        
        # Perform frequency domain analysis using either Welch CSD or FFT method
        if params.get('use_welch_csd', True):
            # Use Welch method with cross-spectral density
            fft_results = perform_welch_csd_analysis(time, windowed_signals['u'], windowed_signals['x'], 
                                                     windowed_signals['y'], dt, params, tool_signal=tool_signal)
            
            # For Welch method, transfer functions are already calculated in perform_welch_csd_analysis
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
        else:
            # Use traditional FFT-based method
            fft_results = perform_fft_analysis(time, windowed_signals['u'], windowed_signals['x'], 
                                               windowed_signals['y'], dt, tool_signal=tool_signal)

            # Calculate displacement transfer functions
            Tool_freq = fft_results.get('Tool', None)
            tf_results = calculate_transfer_functions(fft_results['X'], fft_results['Y'], 
                                                      fft_results['U'], Tool=Tool_freq)
            
            # Calculate acceleration transfer functions
            Tool_accel_freq = fft_results.get('Tool_accel', None)
            tf_accel_results = calculate_transfer_functions(fft_results['X_accel'], fft_results['Y_accel'], 
                                                           fft_results['U'], Tool=Tool_accel_freq)

        # Plot FFT magnitudes of x, y, and x_tool (if available, only if show_all_figures is True)
        if show_all_figures:
            Tool_accel_fft = fft_results.get('Tool_accel', None)
            plot_fft_magnitudes(fft_results['frequencies'], fft_results['X_accel'], fft_results['Y_accel'],
                               Tool_accel=Tool_accel_fft, show_plot=show_plot)

        # Generate PSD plot for input force (only if show_all_figures is True)
        if show_all_figures:
            plot_force_psd(fft_results['frequencies'], fft_results['U'],
                           fft_results['dt'], fft_results['N'], show_plot)
        
        # Plot coherence (always shown, only for Welch CSD method)
        if params.get('use_welch_csd', True) and 'coherence_ux' in fft_results:
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
        
        # Plot reconstructed time-domain signals (only if show_all_figures is True, skip for Welch method - X_full is None)
        if show_all_figures and fft_results.get('X_full') is not None:
            Tool_freq_full = fft_results.get('Tool_full', None)
            plot_reconstructed_time_signals(time, windowed_signals['u'], fft_results['X_full'], 
                                           fft_results['Y_full'], show_plot,
                                           Tool_freq=Tool_freq_full, tool_label=tool_label)
        
        # Save results
        saved_files = save_results(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                                 filepath, save_files)
        
        # Prepare return data
        result = {
            'frequencies': fft_results['frequencies'],
            'Gx': tf_results['Gx'],
            'Gy': tf_results['Gy'],
            'metadata': metadata,
            'saved_files': saved_files,
            'sample_rate': 1 / fft_results['dt'],
            'N_samples': fft_results['N'],
            'window': window,
            'window_metadata': window_metadata
        }
        
        # Add tool transfer function if present
        if 'G_tool' in tf_results:
            result['G_tool'] = tf_results['G_tool']
        
        return result
        
    except Exception as e:
        raise Exception(f"Error processing file {filepath}: {str(e)}")

# =============================================================================
# USER INTERFACE
# =============================================================================

def select_measurement_case(root=None):
    """
    Display dialog for user to select measurement case.
    
    Parameters:
    -----------
    root : tk.Tk, optional
        Root window to use. If None, creates a new root window.
    
    Returns:
    --------
    str : Case identifier ('case1', 'case2_x', or 'case2_y')
        Returns None if user cancels
    """
    # Use provided root or create new one
    created_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()  # Hide root window if we created it
        created_root = True
    
    # Create dialog window
    dialog = tk.Toplevel(root)
    dialog.title("Select Measurement Case")
    dialog.geometry("400x200")
    
    # Make dialog modal
    dialog.grab_set()
    
    # Bring dialog to front and give it focus
    dialog.lift()
    dialog.focus_force()
    
    # Update root to ensure dialog is rendered
    root.update()
    
    # Variable to store result
    result = [None]
    
    def select_case1():
        result[0] = 'case1'
        dialog.destroy()
    
    def select_case2():
        # Close main dialog
        dialog.destroy()
        
        # Create sub-dialog for Case 2 direction
        sub_dialog = tk.Toplevel(root)
        sub_dialog.title("Select Tool Direction")
        sub_dialog.geometry("350x150")
        sub_dialog.grab_set()
        
        # Bring sub-dialog to front and give it focus
        sub_dialog.lift()
        sub_dialog.focus_force()
        root.update()
        
        tk.Label(sub_dialog, text="Select accelerometer direction at tool:", 
                font=("Arial", 10)).pack(pady=10)
        
        def select_x():
            result[0] = 'case2_x'
            sub_dialog.destroy()
        
        def select_y():
            result[0] = 'case2_y'
            sub_dialog.destroy()
        
        btn_x = tk.Button(sub_dialog, text="2.1 X-direction", command=select_x, 
                         width=20, height=2)
        btn_x.pack(pady=5)
        
        btn_y = tk.Button(sub_dialog, text="2.2 Y-direction", command=select_y, 
                         width=20, height=2)
        btn_y.pack(pady=5)
        
        # Update to ensure buttons are visible
        root.update()
        
        sub_dialog.wait_window()
    
    # Main dialog content
    tk.Label(dialog, text="Select measurement case:", 
            font=("Arial", 12, "bold")).pack(pady=15)
    
    btn_case1 = tk.Button(dialog, text="Case 1: Accelerometers only on z-slide", 
                         command=select_case1, width=35, height=2)
    btn_case1.pack(pady=10)
    
    btn_case2 = tk.Button(dialog, text="Case 2: Accelerometer at tool", 
                         command=select_case2, width=35, height=2)
    btn_case2.pack(pady=10)
    
    # Update to ensure dialog content is visible
    root.update()
    
    # Wait for dialog to close
    dialog.wait_window()
    
    # Only destroy root if we created it
    if created_root:
        root.destroy()
    
    return result[0]

def main():
    """
    Entry point with file dialog and user options.
    """
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Select measurement case before file selection
        measurement_case = select_measurement_case(root)
        
        if not measurement_case:
            print("No measurement case selected. Exiting.")
            return
        
        print(f"Selected measurement case: {measurement_case}")
        
        # File selection dialog (allows multiple selection)
        filepaths = filedialog.askopenfilenames(
            title="Select CSV measurement file(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filepaths:
            print("No file selected. Exiting.")
            return
        
        # Convert tuple to list
        filepaths = list(filepaths)
        
        # Determine if single or multiple files
        if len(filepaths) == 1:
            # Single file: use existing process_measurement_file (no averaging)
            filepath = filepaths[0]
            print(f"Processing single file: {filepath}")
            
            # Process the file (always show plots, don't save yet)
            result = process_measurement_file(filepath, measurement_case=measurement_case, show_plot=True, save_files=False)
            
            # Display results
            print(f"\nProcessing completed successfully!")
            print(f"Sample rate: {result['sample_rate']:.0f} Hz")
            print(f"Number of samples: {result['N_samples']}")
            print(f"Frequency range: {result['frequencies'][0]:.2f} - {result['frequencies'][-1]:.2f} Hz")
            
            print(f"\nMetadata:")
            print(f"  Date/Time: {result['metadata']['date_time']}")
            print(f"  Position: X={result['metadata']['x_position']}, Y={result['metadata']['y_position']}, Z={result['metadata']['z_position']}")
            print(f"  Direction: {result['metadata']['direction']}")
            
            # Ask user if they want to save CSV files
            save_files = messagebox.askyesno("Save Options", "Do you want to save the output CSV files?")
            
            if save_files:
                saved_files = save_results(result['frequencies'], result['Gx'], result['Gy'], filepath, save_files=True)
                result['saved_files'] = saved_files
                print(f"\nSaved files:")
                for file in saved_files:
                    print(f"  - {file}")
        else:
            # Multiple files: use process_multiple_files (with averaging)
            print(f"Processing {len(filepaths)} files with averaging...")
            for i, fp in enumerate(filepaths, 1):
                print(f"  File {i}: {os.path.basename(fp)}")
            
            # Process multiple files (always show plots, don't save yet)
            result = process_multiple_files(filepaths, measurement_case=measurement_case, show_plot=True, save_files=False)
            
            # Display results
            print(f"\nProcessing completed successfully!")
            print(f"Number of files averaged: {result['n_files']}")
            print(f"Sample rate: {result['sample_rate']:.0f} Hz")
            print(f"Number of samples: {result['N_samples']}")
            print(f"Frequency range: {result['frequencies'][0]:.2f} - {result['frequencies'][-1]:.2f} Hz")
            
            print(f"\nMetadata (from first file):")
            print(f"  Date/Time: {result['metadata']['date_time']}")
            print(f"  Position: X={result['metadata']['x_position']}, Y={result['metadata']['y_position']}, Z={result['metadata']['z_position']}")
            print(f"  Direction: {result['metadata']['direction']}")
            
            # Ask user if they want to save CSV files
            save_files = messagebox.askyesno("Save Options", "Do you want to save the output CSV files?")
            
            if save_files:
                saved_files = save_results(result['frequencies'], result['Gx'], result['Gy'], filepaths[0], save_files=True)
                result['saved_files'] = saved_files
                print(f"\nSaved files:")
                for file in saved_files:
                    print(f"  - {file}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
    
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
