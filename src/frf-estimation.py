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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
    L1 = 0.01      # (m) Length parameter 1
    L2 = 0.05      # (m) Length parameter 2
    L3 = 0.2       # (m) Length parameter 3
    
    # Angular parameter
    alpha = 20 * np.pi / 180  # Angle in radians (20 degrees)
    
    # Sensor calibration parameters
    r_a = 5 * 9.81 / 10    # Accelerometer sensitivity (m/s²/V)
    m_u = 10               # IEPE signal conditioner multiplication factor
    v_u = 2.5e-3           # Force hammer sensitivity (mV/N)
    r_u = 1 / (v_u * m_u)  # Combined output sensitivity (N/V)
    
    # Bias removal parameters
    bias_window_size = 1000  # Number of samples for bias estimation window
    
    return {
        'L1': L1,
        'L2': L2,
        'L3': L3,
        'alpha': alpha,
        'r_a': r_a,
        'm_u': m_u,
        'v_u': v_u,
        'r_u': r_u,
        'bias_window_size': bias_window_size
    }

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_csv_file(filepath):
    """
    Load and parse CSV file, return time array and analog input arrays (A0-A7).
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple : (time_array, A0, A1, A2, A3, A4, A5, A6, A7)
        Time array and 8 analog input arrays
        
    Raises:
    -------
    ValueError : If file format is invalid or data cannot be parsed
    FileNotFoundError : If file doesn't exist
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read CSV file, skipping first 5 header lines
        df = pd.read_csv(
            filepath, 
            skiprows=5, 
            header=None,
            sep=',',  # Explicit separator
            decimal='.',  # Explicit decimal point
            skipinitialspace=True  # Skip leading whitespace
        )
        
        # Validate column count
        if df.shape[1] < 9:
            raise ValueError(f"Expected 9 columns, found {df.shape[1]}")
        
        # Define column names for clarity
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
        A7 = df.iloc[:, 8].values    # Column 8: A7
        
        # Validate data is numeric and finite
        for i, arr in enumerate([time, A0, A1, A2, A3, A4, A5, A6, A7]):
            if arr.dtype == object:
                raise ValueError(f"Column {i} contains non-numeric data")
            if not np.isfinite(arr).all():
                nan_count = (~np.isfinite(arr)).sum()
                raise ValueError(
                    f"Column {i} has {nan_count} non-finite values (NaN or Inf)"
                )
        
        return time, A0, A1, A2, A3, A4, A5, A6, A7
        
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
    
    # Pattern: yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_[direction].csv
    # Make pattern case-insensitive to handle both .csv and .CSV extensions
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_X([^_]+)_Y([^_]+)_Z([^_]+)_([xy])\.csv'
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

def estimate_sample_time(time_array):
    """
    Calculate sample time from first two time points.
    
    Parameters:
    -----------
    time_array : array_like
        Time array in seconds
        
    Returns:
    --------
    float : Sample time in seconds
    """
    if len(time_array) < 2:
        raise ValueError("Time array must have at least 2 points")
    
    dt = time_array[1] - time_array[0]
    return dt

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
    array_like : Bias-corrected signal
    
    Notes:
    ------
    If the signal length is less than the window size, the mean of the entire signal
    is used for bias removal (with a warning).
    """
    signal = np.asarray(signal)
    signal_length = len(signal)
    
    # Handle case where signal is shorter than window
    if signal_length < window_size:
        print(f"Warning: Signal length ({signal_length}) is less than window size ({window_size}). "
              f"Using entire signal for bias calculation.")
        bias = np.mean(signal)
        return signal - bias
    
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
    return signal - bias

def calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params):
    """
    Calculate u, x, y, theta_x, theta_y from analog inputs.
    
    Parameters:
    -----------
    A0, A1, A2, A3, A4, A5, A6, A7 : array_like
        Analog input arrays (Volts)
    params : dict
        System parameters dictionary
        
    Returns:
    --------
    dict : Dictionary containing calculated signals
        u : Force (N)
        x : X-direction acceleration (m/s²)
        y : Y-direction acceleration (m/s²)
        theta_x : Angular displacement about X-axis (rad)
        theta_y : Angular displacement about Y-axis (rad)
    """
    # Extract parameters
    L1 = params['L1']
    L2 = params['L2']
    L3 = params['L3']
    alpha = params['alpha']
    r_u = params['r_u']
    r_a = params['r_a']
    
    # Force input (convert to Newtons)
    u = A7 * r_u
    
    # Angular displacements
    theta_x = -(A1 + A2 - A3 - A4) * np.sin(alpha) / (2 * L3)
    theta_y = -(A1 - A2 - A3 + A4) * np.cos(alpha) / (2 * L3)
    
    # Linear accelerations (convert from voltage to m/s²)
    x = (-(A1 - A2 + A3 - A4) * np.cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2) - A6) * r_a
    y = ((A1 + A2 + A3 + A4) * np.sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2) - A5) * r_a
    
    return {
        'u': u,
        'x': x,
        'y': y,
        'theta_x': theta_x,
        'theta_y': theta_y
    }

# =============================================================================
# FREQUENCY DOMAIN ANALYSIS
# =============================================================================

def perform_fft_analysis(time, u, x, y):
    """
    Perform FFT and return frequencies, X, Y, U in frequency domain.
    Note: x and y are accelerations and will be double-integrated to displacements.
    Also returns acceleration FFT results for comparison.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u : array_like
        Force signal (time domain)
    x, y : array_like
        Acceleration signals in time domain (m/s²)
        
    Returns:
    --------
    dict : Dictionary containing frequency array and FFT results
        Includes both displacement (X, Y) and acceleration (X_accel, Y_accel) results
    """
    N = len(time)
    dt = time[1] - time[0]
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
    
    # Apply transfer function to convert acceleration to displacement
    X = np.zeros_like(X_accel, dtype=complex)
    Y = np.zeros_like(Y_accel, dtype=complex)
    
    # Avoid division by zero at DC (frequency = 0)
    X[1:] = X_accel[1:] * H[1:]
    Y[1:] = Y_accel[1:] * H[1:]
    
    # DC component (zero frequency) set to zero
    X[0] = 0
    Y[0] = 0
    
    # Use only positive frequencies (including DC)
    positive_freq_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_freq_mask]
    U_pos = U[positive_freq_mask]
    X_pos = X[positive_freq_mask]
    Y_pos = Y[positive_freq_mask]
    X_accel_pos = X_accel[positive_freq_mask]
    Y_accel_pos = Y_accel[positive_freq_mask]
    
    return {
        'frequencies': frequencies_pos,
        'U': U_pos,
        'X': X_pos,
        'Y': Y_pos,
        'X_accel': X_accel_pos,
        'Y_accel': Y_accel_pos,
        'N': N,
        'dt': dt
    }

def calculate_transfer_functions(X, Y, U, threshold=1e-12):
    """
    Calculate Gx = X/U and Gy = Y/U transfer functions.
    
    Parameters:
    -----------
    X, Y, U : array_like
        FFT results in frequency domain
    threshold : float
        Minimum magnitude threshold to avoid division by zero
        
    Returns:
    --------
    dict : Dictionary containing transfer functions Gx and Gy
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
    
    return {
        'Gx': Gx,
        'Gy': Gy,
        'valid_mask': valid_mask
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_signal_comparison(time, original_signals, corrected_signals, show_plot=True):
    """
    Plot comparison of original and bias-corrected signals for A0-A7.
    Creates a single figure with 2 rows and 4 columns (8 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    original_signals : dict
        Dictionary with keys 'A0' through 'A7' containing original signals
    corrected_signals : dict
        Dictionary with keys 'A0' through 'A7' containing corrected signals
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
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Signal Comparison - {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_time_domain_signals(time, u, x, y, show_plot=True):
    """
    Plot time-domain signals u, x, y in a single figure with synchronized x-axes.
    
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
    """
    if not show_plot:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot force u
    axes[0].plot(time, u, 'b-', linewidth=1)
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title('Force Input u(t)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot x acceleration
    axes[1].plot(time, x, 'r-', linewidth=1)
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('X-Direction Acceleration x(t)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot y acceleration
    axes[2].plot(time, y, 'g-', linewidth=1)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_title('Y-Direction Acceleration y(t)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_bode_diagram(frequencies, Gx, Gy, Gx_accel=None, Gy_accel=None, show_plot=True):
    """
    Generate Bode magnitude and phase plots for Gx and Gy in a 2x2 layout.
    Optionally overlay acceleration-based transfer functions.
    
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
    
    # Create figure with 2x2 subplots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    # Link x-axes within each column for synchronized zooming
    ax[1, 0].sharex(ax[0, 0])  # Gx magnitude and phase
    ax[1, 1].sharex(ax[0, 1])  # Gy magnitude and phase
    
    # Plot Gx magnitude (top-left)
    ax[0, 0].loglog(valid_freq, valid_Gx_mag, 'b-', linewidth=2, label='Gx (Displacement)')
    if Gx_accel is not None:
        ax[0, 0].loglog(valid_freq_accel, valid_Gx_accel_mag, 'b--', linewidth=2, alpha=0.7, label='Gx (Acceleration)')
    ax[0, 0].set_ylabel('Compliance (m/N or m/s²/N)')
    ax[0, 0].set_title('Compliance Gx (X/U) - Magnitude')
    ax[0, 0].grid(True, which="both", ls="-", alpha=0.3)
    ax[0, 0].legend()
    
    # Plot Gy magnitude (top-right)
    ax[0, 1].loglog(valid_freq, valid_Gy_mag, 'r-', linewidth=2, label='Gy (Displacement)')
    if Gy_accel is not None:
        ax[0, 1].loglog(valid_freq_accel, valid_Gy_accel_mag, 'r--', linewidth=2, alpha=0.7, label='Gy (Acceleration)')
    ax[0, 1].set_ylabel('Compliance (m/N or m/s²/N)')
    ax[0, 1].set_title('Compliance Gy (Y/U) - Magnitude')
    ax[0, 1].grid(True, which="both", ls="-", alpha=0.3)
    ax[0, 1].legend()
    
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
    
    # Plot Gy phase (bottom-right)
    ax[1, 1].semilogx(valid_freq, valid_Gy_phase, 'r-', linewidth=2, label='Gy (Displacement)')
    if Gy_accel is not None:
        ax[1, 1].semilogx(valid_freq_accel, valid_Gy_accel_phase, 'r--', linewidth=2, alpha=0.7, label='Gy (Acceleration)')
    ax[1, 1].set_xlabel('Frequency (Hz)')
    ax[1, 1].set_ylabel('Phase (degrees)')
    ax[1, 1].set_title('Compliance Gy (Y/U) - Phase')
    ax[1, 1].grid(True, which="both", ls="-", alpha=0.3)
    ax[1, 1].legend()
    # Removed fixed y-limits to allow unwrapped phase to display properly
    
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
    # Pattern: ..._x.csv or ..._y.csv -> ..._Gx.csv or ..._Gy.csv
    # Make pattern case-insensitive to handle both .csv and .CSV extensions
    pattern = r'(_[xy]\.csv)$'
    new_filename = re.sub(pattern, f'_{suffix}.csv', filename, flags=re.IGNORECASE)
    
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

def process_measurement_file(filepath, show_plot=True, save_files=True):
    """
    Main function to process a single measurement file.
    
    Parameters:
    -----------
    filepath : str
        Path to input CSV file
    show_plot : bool
        Whether to display Bode plots
    save_files : bool
        Whether to save output CSV files
        
    Returns:
    --------
    dict : Contains frequencies, Gx, Gy, and metadata
    """
    try:
        # Load system parameters
        params = get_system_parameters()
        
        # Parse filename for metadata
        metadata = parse_filename(filepath)
        
        # Load CSV data
        time, A0, A1, A2, A3, A4, A5, A6, A7 = load_csv_file(filepath)
        
        # Store original signals for comparison plots
        original_signals = {
            'A0': A0.copy(), 'A1': A1.copy(), 'A2': A2.copy(), 'A3': A3.copy(),
            'A4': A4.copy(), 'A5': A5.copy(), 'A6': A6.copy(), 'A7': A7.copy()
        }
        
        # Remove bias from all analog input signals
        window_size = params['bias_window_size']
        A0 = remove_bias(A0, window_size)
        A1 = remove_bias(A1, window_size)
        A2 = remove_bias(A2, window_size)
        A3 = remove_bias(A3, window_size)
        A4 = remove_bias(A4, window_size)
        A5 = remove_bias(A5, window_size)
        A6 = remove_bias(A6, window_size)
        A7 = remove_bias(A7, window_size)
        
        # Store corrected signals for comparison plots
        corrected_signals = {
            'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
            'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
        }
        
        # Show signal comparison plots
        plot_signal_comparison(time, original_signals, corrected_signals, show_plot)
        
        # Calculate derived signals
        signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params)
        
        # Plot time-domain signals
        plot_time_domain_signals(time, signals['u'], signals['x'], signals['y'], show_plot)
        
        # Perform FFT analysis
        fft_results = perform_fft_analysis(time, signals['u'], signals['x'], signals['y'])

        # Generate PSD plot for input force
        plot_force_psd(fft_results['frequencies'], fft_results['U'],
                       fft_results['dt'], fft_results['N'], show_plot)

        # Calculate displacement transfer functions
        tf_results = calculate_transfer_functions(fft_results['X'], fft_results['Y'], fft_results['U'])
        
        # Calculate acceleration transfer functions
        tf_accel_results = calculate_transfer_functions(fft_results['X_accel'], fft_results['Y_accel'], fft_results['U'])
        
        # Generate Bode plots with both displacement and acceleration
        plot_bode_diagram(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], 
                         tf_accel_results['Gx'], tf_accel_results['Gy'], show_plot)
        
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
            'N_samples': fft_results['N']
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Error processing file {filepath}: {str(e)}")

# =============================================================================
# USER INTERFACE
# =============================================================================

def main():
    """
    Entry point with file dialog and user options.
    """
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()
    
    try:
        # File selection dialog
        filepath = filedialog.askopenfilename(
            title="Select CSV measurement file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            print("No file selected. Exiting.")
            return
        
        print(f"Processing file: {filepath}")
        
        # User options
        show_plot = messagebox.askyesno("Display Options", "Do you want to display the Bode plots?")
        save_files = messagebox.askyesno("Save Options", "Do you want to save the output CSV files?")
        
        # Process the file
        result = process_measurement_file(filepath, show_plot, save_files)
        
        # Display results
        print(f"\nProcessing completed successfully!")
        print(f"Sample rate: {result['sample_rate']:.0f} Hz")
        print(f"Number of samples: {result['N_samples']}")
        print(f"Frequency range: {result['frequencies'][0]:.2f} - {result['frequencies'][-1]:.2f} Hz")
        
        if result['saved_files']:
            print(f"\nSaved files:")
            for file in result['saved_files']:
                print(f"  - {file}")
        
        print(f"\nMetadata:")
        print(f"  Date/Time: {result['metadata']['date_time']}")
        print(f"  Position: X={result['metadata']['x_position']}, Y={result['metadata']['y_position']}, Z={result['metadata']['z_position']}")
        print(f"  Direction: {result['metadata']['direction']}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
    
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
