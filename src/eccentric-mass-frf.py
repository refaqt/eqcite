#!/usr/bin/env python3
"""
Eccentric Mass FRF Estimation Tool

A Python application for estimating frequency response functions using an eccentric mass
in the spindle. Processes 6-channel accelerometer data, calculates frequency and amplitude
using sliding Hanning windows, computes input force from eccentricity, and generates bode plots.

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
import time as time_module

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
    L1 = 0.02      # (m) Length parameter 1 - from tool tip to impact point
    L2 = 0.078     # (m) Length parameter 2 - from impact point to bottom accelerometers
    L3 = 0.160     # (m) Length parameter 3 - from bottom to top accelerometers
    
    # Angular parameter
    alpha = 20.123 * np.pi / 180  # Angle in radians (20 degrees)
    
    # Sensor calibration parameters
    r_a = 1 * 9.81 / 0.2    # Accelerometer sensitivity (m/s²/V)
    
    # Bias removal parameters
    bias_window_size = 1000  # Number of samples for bias estimation window
    
    # High-pass filter parameters
    highpass_cutoff_hz = 0.5  # High-pass filter cutoff frequency (Hz)
    
    # Eccentric mass parameters
    e_mass = 100.0  # Eccentricity in g*mm (default value, should be adjusted)
    
    # Sliding window parameters
    window_duration = 0.1  # Window duration in seconds (default 0.1)

    return {
        'L1': L1,
        'L2': L2,
        'L3': L3,
        'alpha': alpha,
        'r_a': r_a,
        'bias_window_size': bias_window_size,
        'highpass_cutoff_hz': highpass_cutoff_hz,
        'e_mass': e_mass,
        'window_duration': window_duration
    }

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_csv_file(filepath):
    """
    Load and parse CSV file, return time array and analog input arrays (A0-A5).
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple : (time_array, A0, A1, A2, A3, A4, A5)
        Time array and 6 analog input arrays.
        
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
        
        # Validate column count: require exactly 7 columns (Time + A0-A5)
        n_cols = df.shape[1]
        if n_cols != 7:
            raise ValueError(
                f"Expected 7 columns (Time + A0–A5), found {n_cols}"
            )
        
        # Define column names for clarity
        column_names = ['Time', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5']
        
        # Convert each column to numeric with error handling
        for i, col_name in enumerate(column_names):
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
        
        # Validate data is numeric and finite
        for i, arr in enumerate([time, A0, A1, A2, A3, A4, A5]):
            if arr.dtype == object:
                raise ValueError(f"Column {i} contains non-numeric data")
            if not np.isfinite(arr).all():
                nan_count = (~np.isfinite(arr)).sum()
                raise ValueError(
                    f"Column {i} has {nan_count} non-finite values (NaN or Inf)"
                )
        
        return time, A0, A1, A2, A3, A4, A5
        
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
        Returns None if filename doesn't match pattern
    """
    filename = os.path.basename(filepath)
    
    # Pattern: yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_[direction].csv
    # Make pattern case-insensitive to handle both .csv and .CSV extensions
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_X([^_]+)_Y([^_]+)_Z([^_]+)_([xy])\.csv'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        return None
    
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
    
    return filtered_data

def calculate_derived_signals(A0, A1, A2, A3, A4, A5, params):
    """
    Calculate x, y, theta_x, theta_y from analog inputs (A0-A5 only).
    
    Parameters:
    -----------
    A0, A1, A2, A3, A4, A5 : array_like
        Analog input arrays (Volts)
    params : dict
        System parameters dictionary
        
    Returns:
    --------
    dict : Dictionary containing calculated signals
        x : X-direction acceleration (m/s²)
        y : Y-direction acceleration (m/s²)
        theta_x : Angular acceleration about X-axis (rad/s²)
        theta_y : Angular acceleration about Y-axis (rad/s²)
    """
    # Extract parameters
    L1 = params['L1']
    L2 = params['L2']
    L3 = params['L3']
    alpha = params['alpha']
    r_a = params['r_a']
    
    # Angular accelerations (already in rad/s² from the calculation)
    theta_x = -(A0 + A1 - A2 - A3) * np.sin(alpha) / (2 * L3)
    theta_y = -(A0 - A1 - A2 + A3) * np.cos(alpha) / (2 * L3)
    
    # Linear accelerations (convert from voltage to m/s²)
    x = (-(A0 - A1 + A2 - A3) * np.cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2) - A4) * r_a
    y = ((A0 + A1 + A2 + A3) * np.sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2) - A5) * r_a
    
    return {
        'x': x,
        'y': y,
        'theta_x': theta_x,
        'theta_y': theta_y
    }

# =============================================================================
# FREQUENCY AND AMPLITUDE CALCULATION
# =============================================================================

def calculate_windowed_frequency_amplitude(time, x, y, theta_x, theta_y, window_duration, fs):
    """
    Calculate frequency and amplitude using sliding Hanning window with FFT.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    x, y, theta_x, theta_y : array_like
        Signal arrays (accelerations)
    window_duration : float
        Window duration in seconds
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    output1 : np.ndarray
        Matrix with columns [time, frequency, amplitude_X, amplitude_Y, 
                            amplitude_theta_X, amplitude_theta_Y]
    output2 : np.ndarray
        Filtered output1 where amplitude_X > 0.5 m/s²
    debug_data : dict
        Debug data for interactive plotting (keyed by center_idx)
    """
    dt = 1.0 / fs
    window_samples = int(window_duration * fs)
    
    # Create Hanning window
    hann_window = np.hanning(window_samples)
    
    n_samples = len(time)
    
    # Initialize output arrays
    frequency_array = np.zeros(n_samples)
    amplitude_x = np.zeros(n_samples)
    amplitude_y = np.zeros(n_samples)
    amplitude_theta_x = np.zeros(n_samples)
    amplitude_theta_y = np.zeros(n_samples)
    
    # Store debug data for interactive plotting
    debug_data = {}
    
    # Process signals
    signals = {'x': x, 'y': y, 'theta_x': theta_x, 'theta_y': theta_y}
    amplitude_arrays = {
        'x': amplitude_x,
        'y': amplitude_y,
        'theta_x': amplitude_theta_x,
        'theta_y': amplitude_theta_y
    }
    
    print(f"\nCalculating windowed frequency and amplitude:")
    print(f"Window duration: {window_duration} seconds")
    print(f"Window samples: {window_samples}")
    print(f"Processing {n_samples} samples...")
    
    # Loop through signal with sliding window
    for i in range(n_samples - window_samples + 1):
        center_idx = i + window_samples // 2
        
        # Calculate frequency from X signal only
        windowed_x = x[i:i+window_samples] * hann_window
        
        # Apply FFT
        fft_result = np.fft.rfft(windowed_x)
        
        # Get frequency bins
        freqs = np.fft.rfftfreq(window_samples, d=dt)
        
        # Find dominant frequency (exclude DC component)
        magnitude = np.abs(fft_result[1:])  # Skip DC component
        freq_bins = freqs[1:]  # Skip DC component
        
        # Initialize debug variables
        interpolated = False
        poly_coeffs = None
        dominant_freq = 0
        max_idx = 0
        
        if len(magnitude) > 0:
            # Find index of maximum magnitude
            max_idx = np.argmax(magnitude)
            
            # 3-point interpolation for better frequency resolution
            dominant_freq = freq_bins[max_idx]
            
            # Check if we have at least 1 point on each side for 3-point interpolation
            if max_idx >= 1 and max_idx < len(magnitude) - 1:
                # Extract 3 magnitude points around the maximum
                mag_3pts = magnitude[max_idx-1:max_idx+2]
                freq_3pts = freq_bins[max_idx-1:max_idx+2]
                
                # Fit 2nd order polynomial through the 3 points
                coeffs = np.polyfit(freq_3pts, mag_3pts, 2)
                
                # Check if polynomial has downward curvature (valid peak)
                if coeffs[0] < 0:
                    # Find polynomial peak: peak = -b/(2a) for ax^2 + bx + c
                    peak_freq = -coeffs[1] / (2 * coeffs[0])
                    
                    # Validate peak is within the 3-point frequency range
                    if freq_3pts[0] <= peak_freq <= freq_3pts[-1]:
                        dominant_freq = peak_freq
                        interpolated = True
                        poly_coeffs = coeffs
        
        # Store frequency at window center position
        if center_idx < n_samples:
            frequency_array[center_idx] = dominant_freq
            
            # Store debug data for X signal (for interactive plotting)
            debug_data[center_idx] = {
                'windowed_signal': windowed_x.copy(),
                'fft_magnitude': magnitude.copy(),
                'fft_freqs': freq_bins.copy(),
                'dominant_freq': dominant_freq,
                'window_start_idx': i,
                'window_end_idx': i + window_samples - 1,
                'max_idx': max_idx,
                'interpolated': interpolated,
                'poly_coeffs': poly_coeffs,
                'window_center_time': time[center_idx]
            }
        
        # Calculate amplitudes for all signals at the dominant frequency
        for sig_name, signal_data in signals.items():
            # Extract windowed segment
            windowed_signal = signal_data[i:i+window_samples] * hann_window
            
            # Apply FFT
            fft_result = np.fft.rfft(windowed_signal)
            
            # Get frequency bins
            freqs = np.fft.rfftfreq(window_samples, d=dt)
            
            # Find magnitude at dominant frequency
            if dominant_freq > 0:
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(freqs - dominant_freq))
                amplitude = np.abs(fft_result[freq_idx])
            else:
                # If no dominant frequency, use maximum magnitude
                magnitude = np.abs(fft_result[1:])  # Skip DC
                if len(magnitude) > 0:
                    amplitude = np.max(magnitude)
                else:
                    amplitude = 0
            
            # Store amplitude at window center position
            if center_idx < n_samples:
                amplitude_arrays[sig_name][center_idx] = amplitude
    
    # Create output1: matrix with all data
    output1 = np.column_stack([
        time,
        frequency_array,
        amplitude_x,
        amplitude_y,
        amplitude_theta_x,
        amplitude_theta_y
    ])
    
    # Create output2: filter where amplitude_X > 0.5 m/s²
    mask = amplitude_x > 0.5
    output2 = output1[mask]
    
    print(f"Output1 shape: {output1.shape}")
    print(f"Output2 shape (filtered): {output2.shape}")
    print(f"Filtered {np.sum(mask)} points where amplitude_X > 0.5 m/s²")
    
    return output1, output2, debug_data

def calculate_input_force(frequency, e_mass):
    """
    Calculate input force U from eccentricity and frequency.
    
    Parameters:
    -----------
    frequency : array_like
        Frequency array in Hz
    e_mass : float
        Eccentricity in g*mm
        
    Returns:
    --------
    array_like : Input force U in Newtons
    """
    # Convert e_mass from g*mm to kg*m
    # 1 g = 0.001 kg, 1 mm = 0.001 m
    # So 1 g*mm = 0.001 * 0.001 = 1e-6 kg*m
    e_mass_kgm = e_mass * 1e-6
    
    # Calculate centrifugal force: U = m * e * (2*pi*f)^2
    # where m*e is the eccentricity (mass * distance)
    U = e_mass_kgm * (2 * np.pi * frequency) ** 2
    
    return U

# =============================================================================
# FRF CALCULATION
# =============================================================================

def calculate_frfs(output2, dt):
    """
    Calculate frequency response functions from output2 data.
    
    Parameters:
    -----------
    output2 : np.ndarray
        Matrix with columns [time, frequency, amplitude_X, amplitude_Y, 
                            amplitude_theta_X, amplitude_theta_Y, U]
    dt : float
        Sample time in seconds
        
    Returns:
    --------
    dict : Dictionary containing FRF results
        frequencies : Frequency array (Hz)
        frf_x_accel : X/U acceleration FRF (m/s²/N)
        frf_y_accel : Y/U acceleration FRF (m/s²/N)
        frf_theta_x_accel : theta_X/U acceleration FRF (rad/s²/N)
        frf_theta_y_accel : theta_Y/U acceleration FRF (rad/s²/N)
        frf_x_disp : X/U displacement FRF (m/N)
        frf_y_disp : Y/U displacement FRF (m/N)
        frf_theta_x_disp : theta_X/U displacement FRF (rad/N)
        frf_theta_y_disp : theta_Y/U displacement FRF (rad/N)
    """
    # Extract data from output2
    frequencies = output2[:, 1]  # Frequency
    amplitude_x = output2[:, 2]
    amplitude_y = output2[:, 3]
    amplitude_theta_x = output2[:, 4]
    amplitude_theta_y = output2[:, 5]
    U = output2[:, 6]  # Input force
    
    # Calculate acceleration FRFs (direct division)
    frf_x_accel = amplitude_x / U
    frf_y_accel = amplitude_y / U
    frf_theta_x_accel = amplitude_theta_x / U
    frf_theta_y_accel = amplitude_theta_y / U
    
    # Calculate displacement FRFs using frequency domain double integration
    # Convert acceleration amplitude to displacement amplitude: X_amp = A_amp / omega^2
    # where omega = 2*pi*f
    omega = 2 * np.pi * frequencies
    
    # Avoid division by zero at DC
    omega_safe = np.where(omega == 0, 1.0, omega)
    
    # Displacement amplitude = Acceleration amplitude / omega^2
    # Note: We use omega^2 (not -omega^2) because we're working with magnitudes
    frf_x_disp = frf_x_accel / (omega_safe**2)
    frf_y_disp = frf_y_accel / (omega_safe**2)
    frf_theta_x_disp = frf_theta_x_accel / (omega_safe**2)
    frf_theta_y_disp = frf_theta_y_accel / (omega_safe**2)
    
    # Set DC components to zero
    dc_mask = frequencies == 0
    if np.any(dc_mask):
        frf_x_disp[dc_mask] = 0
        frf_y_disp[dc_mask] = 0
        frf_theta_x_disp[dc_mask] = 0
        frf_theta_y_disp[dc_mask] = 0
    
    return {
        'frequencies': frequencies,
        'frf_x_accel': frf_x_accel,
        'frf_y_accel': frf_y_accel,
        'frf_theta_x_accel': frf_theta_x_accel,
        'frf_theta_y_accel': frf_theta_y_accel,
        'frf_x_disp': frf_x_disp,
        'frf_y_disp': frf_y_disp,
        'frf_theta_x_disp': frf_theta_x_disp,
        'frf_theta_y_disp': frf_theta_y_disp
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot=True):
    """
    Plot comparison of original, bias-corrected, and high-pass filtered signals for A0-A5.
    Creates a single figure with 2 rows and 3 columns (6 subplots total).
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    original_signals : dict
        Dictionary with keys 'A0' through 'A5' containing original signals
    corrected_signals : dict
        Dictionary with keys 'A0' through 'A5' containing bias-corrected signals
    filtered_signals : dict
        Dictionary with keys 'A0' through 'A5' containing high-pass filtered signals
    show_plot : bool
        Whether to display the plots
    """
    if not show_plot:
        return
    
    signal_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
    
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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

def plot_accelerations(time, x, y, theta_x, theta_y, show_plot=True):
    """
    Plot X, Y, theta_X, and theta_Y accelerations in a 2x2 grid.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    x, y : array_like
        Linear accelerations (m/s²)
    theta_x, theta_y : array_like
        Angular accelerations (rad/s²)
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: X acceleration
    axes[0, 0].plot(time, x, 'b-', linewidth=1)
    axes[0, 0].set_ylabel('Acceleration (m/s²)')
    axes[0, 0].set_title('X-Direction Acceleration')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-right: Y acceleration
    axes[0, 1].plot(time, y, 'r-', linewidth=1)
    axes[0, 1].set_ylabel('Acceleration (m/s²)')
    axes[0, 1].set_title('Y-Direction Acceleration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom-left: theta_X acceleration
    axes[1, 0].plot(time, theta_x, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angular Acceleration (rad/s²)')
    axes[1, 0].set_title('theta_X Angular Acceleration')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: theta_Y acceleration
    axes[1, 1].plot(time, theta_y, 'm-', linewidth=1)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angular Acceleration (rad/s²)')
    axes[1, 1].set_title('theta_Y Angular Acceleration')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

class FrequencyPlotHandler:
    """Handler class for interactive frequency plot with click debugging."""
    
    def __init__(self, time, frequency_array, dt):
        self.time = time
        self.frequency_array = frequency_array
        self.dt = dt
        self.debug_data = {}
        self.fig = None
        self.ax1 = None
        self._last_click_time = 0
        
    def on_click(self, event):
        """Handle click events on the frequency plot."""
        # Check if click is in the frequency plot (ax1)
        if event.inaxes != self.ax1:
            return
        
        # Check if click is a left mouse button
        if event.button != 1:
            return
        
        # Check if zoom/pan tools are active
        if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar:
            toolbar = self.fig.canvas.toolbar
            if hasattr(toolbar, 'mode') and toolbar.mode != '':
                return
        
        # Check if modifier keys are pressed
        if event.key and event.key.lower() in ['ctrl', 'shift', 'alt', 'control']:
            return
        
        # Check if this is a drag operation
        current_time = time_module.time()
        if current_time - self._last_click_time < 0.15:
            return
        self._last_click_time = current_time
        
        # Find nearest datapoint to click location in time
        click_time = event.xdata
        if click_time is None:
            return
        
        # Find nearest time index
        time_diff = np.abs(self.time - click_time)
        nearest_idx = np.argmin(time_diff)
        
        # Check if we have debug data for this position
        if nearest_idx in self.debug_data:
            window_data = self.debug_data[nearest_idx]
            self.plot_debug_window(window_data, nearest_idx)
        else:
            print(f"No debug data available for time index {nearest_idx}")
    
    def plot_debug_window(self, window_data, center_idx):
        """Create debug plot window showing windowed signal, FFT, and frequency details."""
        # Create new figure with 3 subplots
        debug_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Extract debug data
        windowed_signal = window_data['windowed_signal']
        fft_magnitude = window_data['fft_magnitude']
        fft_freqs = window_data['fft_freqs']
        dominant_freq = window_data['dominant_freq']
        window_start_idx = window_data['window_start_idx']
        window_end_idx = window_data['window_end_idx']
        max_idx = window_data['max_idx']
        interpolated = window_data['interpolated']
        poly_coeffs = window_data['poly_coeffs']
        window_center_time = window_data['window_center_time']
        
        # Create time array for windowed signal
        window_time = self.time[window_start_idx:window_end_idx+1]
        
        # Subplot 1: Windowed signal vs time
        ax1.plot(window_time, windowed_signal, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Windowed Signal', fontsize=12)
        ax1.set_title(f'Windowed Signal (Center time: {window_center_time:.3f}s)', 
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Subplot 2: FFT Magnitude vs Frequency
        ax2.plot(fft_freqs, fft_magnitude, 'g-', linewidth=1.0)
        
        # Mark the FFT bin maximum
        ax2.plot(fft_freqs[max_idx], fft_magnitude[max_idx], 'ro', markersize=8, 
                 label=f'FFT Bin Max: {fft_freqs[max_idx]:.2f} Hz')
        
        # If interpolation was used, show the 3 points and fitted parabola
        if interpolated and poly_coeffs is not None:
            # Show the 3 points used for interpolation
            mag_3pts = fft_magnitude[max_idx-1:max_idx+2]
            freq_3pts = fft_freqs[max_idx-1:max_idx+2]
            ax2.plot(freq_3pts, mag_3pts, 'ro', markersize=6, alpha=0.7, 
                     label='Interpolation Points')
            
            # Plot fitted parabola
            freq_fine = np.linspace(freq_3pts[0], freq_3pts[-1], 100)
            parabola = poly_coeffs[0] * freq_fine**2 + poly_coeffs[1] * freq_fine + poly_coeffs[2]
            ax2.plot(freq_fine, parabola, 'r--', linewidth=2, alpha=0.7, 
                     label='Fitted Parabola')
        
        # Mark the calculated dominant frequency
        ax2.axvline(x=dominant_freq, color='r', linestyle='-', linewidth=2, 
                    label=f'Calculated Freq: {dominant_freq:.3f} Hz')
        
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Magnitude', fontsize=12)
        ax2.set_title('FFT Magnitude Spectrum', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Subplot 3: Text summary
        ax3.axis('off')
        summary_text = f"""DEBUG SUMMARY
        
Window Center Time: {window_center_time:.6f} s
Window Indices: {window_start_idx} to {window_end_idx}

Calculated Frequency: {dominant_freq:.6f} Hz
Spindle Speed: {dominant_freq * 60:.2f} RPM

Interpolation Used: {interpolated}
FFT Bin Maximum: {fft_freqs[max_idx]:.6f} Hz

"""
        
        if interpolated and poly_coeffs is not None:
            summary_text += f"""Polynomial Coefficients:
  a (x²): {poly_coeffs[0]:.6e}
  b (x¹): {poly_coeffs[1]:.6e}  
  c (x⁰): {poly_coeffs[2]:.6e}

Validation Checks:
  Downward curvature: {poly_coeffs[0] < 0}
  Peak in range: {fft_freqs[max_idx-1] <= dominant_freq <= fft_freqs[max_idx+1]}
"""
        else:
            summary_text += "Using simple FFT bin maximum (no interpolation)"
        
        ax3.text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def plot_frequency_measurement(time, frequency_array, window_duration, debug_data=None, show_plot=True):
    """
    Plot frequency measurement with interactive debugging.
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    frequency_array : array_like
        Frequency array in Hz
    window_duration : float
        Window duration in seconds
    debug_data : dict, optional
        Debug data for interactive plotting (keyed by center_idx)
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    # Create figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Convert frequency to spindle speed (RPM)
    spindle_rpm = frequency_array * 60
    
    # Plot spindle speed vs time
    ax.plot(time, spindle_rpm, 'r-', linewidth=1.0)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Spindle Speed (RPM)', fontsize=12)
    ax.set_title(f'Frequency Measurement (Window: {window_duration}s)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Calculate statistics
    valid_freqs = frequency_array[frequency_array > 0]
    if len(valid_freqs) > 0:
        valid_rpm = valid_freqs * 60
        stats = {
            'Window duration': f"{window_duration} s",
            'Max spindle speed': f"{np.max(valid_rpm):.0f} RPM",
            'Min spindle speed': f"{np.min(valid_rpm):.0f} RPM",
            'Mean spindle speed': f"{np.mean(valid_rpm):.0f} RPM",
            'Std spindle speed': f"{np.std(valid_rpm):.0f} RPM"
        }
    else:
        stats = {
            'Window duration': f"{window_duration} s",
            'Max spindle speed': "N/A",
            'Min spindle speed': "N/A",
            'Mean spindle speed': "N/A",
            'Std spindle speed': "N/A"
        }
    
    # Add statistics text box
    stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7),
             verticalalignment='bottom')
    
    # Add interactive debugging if debug_data is available
    if debug_data is not None:
        handler = FrequencyPlotHandler(time, frequency_array, 1.0 / (time[1] - time[0]) if len(time) > 1 else 1.0)
        handler.debug_data = debug_data
        handler.fig = fig
        handler.ax1 = ax
        
        # Connect click event handler
        fig.canvas.mpl_connect('button_press_event', handler.on_click)
        
        # Add instruction text
        fig.text(0.5, 0.02, "Click on frequency plot to debug frequency calculation", 
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Print statistics to console
    print("\nFrequency Measurement Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    if debug_data is not None:
        print("\nInteractive Debug: Click on any point in the frequency plot to see detailed analysis!")
    
    plt.show()

def plot_bode_frfs(frf_results, show_plot=True):
    """
    Plot bode diagrams for all FRFs in a 2x2 grid.
    
    Parameters:
    -----------
    frf_results : dict
        Dictionary containing FRF results from calculate_frfs()
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    frequencies = frf_results['frequencies']
    
    # Remove zero frequencies for log plot
    mask = frequencies > 0
    freq_plot = frequencies[mask]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: X/U
    ax = axes[0, 0]
    frf_x_disp = np.abs(frf_results['frf_x_disp'][mask])
    frf_x_accel = np.abs(frf_results['frf_x_accel'][mask])
    ax.loglog(freq_plot, frf_x_disp, 'b-', linewidth=2, label='X/U (Displacement)')
    ax.loglog(freq_plot, frf_x_accel, 'b--', linewidth=2, alpha=0.7, label='X/U (Acceleration)')
    ax.set_ylabel('Magnitude (m/N or m/s²/N)')
    ax.set_title('FRF X/U - Magnitude')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    # Top-right: Y/U
    ax = axes[0, 1]
    frf_y_disp = np.abs(frf_results['frf_y_disp'][mask])
    frf_y_accel = np.abs(frf_results['frf_y_accel'][mask])
    ax.loglog(freq_plot, frf_y_disp, 'r-', linewidth=2, label='Y/U (Displacement)')
    ax.loglog(freq_plot, frf_y_accel, 'r--', linewidth=2, alpha=0.7, label='Y/U (Acceleration)')
    ax.set_ylabel('Magnitude (m/N or m/s²/N)')
    ax.set_title('FRF Y/U - Magnitude')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    # Bottom-left: theta_X/U
    ax = axes[1, 0]
    frf_theta_x_disp = np.abs(frf_results['frf_theta_x_disp'][mask])
    frf_theta_x_accel = np.abs(frf_results['frf_theta_x_accel'][mask])
    ax.loglog(freq_plot, frf_theta_x_disp, 'g-', linewidth=2, label='theta_X/U (Displacement)')
    ax.loglog(freq_plot, frf_theta_x_accel, 'g--', linewidth=2, alpha=0.7, label='theta_X/U (Acceleration)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (rad/N or rad/s²/N)')
    ax.set_title('FRF theta_X/U - Magnitude')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    # Bottom-right: theta_Y/U
    ax = axes[1, 1]
    frf_theta_y_disp = np.abs(frf_results['frf_theta_y_disp'][mask])
    frf_theta_y_accel = np.abs(frf_results['frf_theta_y_accel'][mask])
    ax.loglog(freq_plot, frf_theta_y_disp, 'm-', linewidth=2, label='theta_Y/U (Displacement)')
    ax.loglog(freq_plot, frf_theta_y_accel, 'm--', linewidth=2, alpha=0.7, label='theta_Y/U (Acceleration)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (rad/N or rad/s²/N)')
    ax.set_title('FRF theta_Y/U - Magnitude')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# FILE OUTPUT
# =============================================================================

def save_output(output2, original_filepath, default_folder='data'):
    """
    Save output2 to CSV file.
    
    Parameters:
    -----------
    output2 : np.ndarray
        Matrix with columns [time, frequency, amplitude_X, amplitude_Y, 
                            amplitude_theta_X, amplitude_theta_Y, U]
    original_filepath : str
        Path to original CSV file
    default_folder : str
        Default output folder
        
    Returns:
    --------
    str or None : Path to saved file, or None if not saved
    """
    root = tk.Tk()
    root.withdraw()
    
    # Ask user if they want to save
    save = messagebox.askyesno("Save Output", "Do you want to save the output to a CSV file?")
    
    if not save:
        root.destroy()
        return None
    
    # Create default output directory if it doesn't exist
    output_dir = Path(original_filepath).parent / default_folder
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    original_filename = Path(original_filepath).stem
    output_filename = f"{original_filename}_eccentric_mass_output.csv"
    default_path = output_dir / output_filename
    
    # Ask user for save location
    filepath = filedialog.asksaveasfilename(
        title="Save output CSV file",
        initialdir=str(output_dir),
        initialfile=output_filename,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    
    if not filepath:
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': output2[:, 0],
        'frequency': output2[:, 1],
        'amplitude_X': output2[:, 2],
        'amplitude_Y': output2[:, 3],
        'amplitude_theta_X': output2[:, 4],
        'amplitude_theta_Y': output2[:, 5],
        'U': output2[:, 6]
    })
    
    # Save to CSV
    df.to_csv(filepath, index=False, float_format='%.6e')
    
    print(f"\nOutput saved to: {filepath}")
    return filepath

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_eccentric_mass_file(filepath, show_plot=True):
    """
    Main function to process a single measurement file.
    
    Parameters:
    -----------
    filepath : str
        Path to input CSV file
    show_plot : bool
        Whether to display plots
        
    Returns:
    --------
    dict : Contains processing results and metadata
    """
    try:
        # Load system parameters
        params = get_system_parameters()
        
        # Parse filename for metadata (optional)
        metadata = parse_filename(filepath)
        
        # Load CSV data
        time, A0, A1, A2, A3, A4, A5 = load_csv_file(filepath)
        
        # Calculate sampling frequency
        dt = estimate_sample_time(time)
        fs = 1 / dt
        
        # Store original signals for comparison plots
        original_signals = {
            'A0': A0.copy(), 'A1': A1.copy(), 'A2': A2.copy(),
            'A3': A3.copy(), 'A4': A4.copy(), 'A5': A5.copy()
        }
        
        # Remove bias from all analog input signals
        window_size = params['bias_window_size']
        A0 = remove_bias(A0, window_size)
        A1 = remove_bias(A1, window_size)
        A2 = remove_bias(A2, window_size)
        A3 = remove_bias(A3, window_size)
        A4 = remove_bias(A4, window_size)
        A5 = remove_bias(A5, window_size)
        
        # Store corrected signals for comparison plots
        corrected_signals = {
            'A0': A0, 'A1': A1, 'A2': A2,
            'A3': A3, 'A4': A4, 'A5': A5
        }
        
        # Apply high-pass filter to all analog input signals
        cutoff_hz = params['highpass_cutoff_hz']
        A0 = apply_highpass_filter(A0, cutoff_hz, fs)
        A1 = apply_highpass_filter(A1, cutoff_hz, fs)
        A2 = apply_highpass_filter(A2, cutoff_hz, fs)
        A3 = apply_highpass_filter(A3, cutoff_hz, fs)
        A4 = apply_highpass_filter(A4, cutoff_hz, fs)
        A5 = apply_highpass_filter(A5, cutoff_hz, fs)
        
        # Store filtered signals for comparison plots
        filtered_signals = {
            'A0': A0, 'A1': A1, 'A2': A2,
            'A3': A3, 'A4': A4, 'A5': A5
        }
        
        # Show signal comparison plots
        plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot)
        
        # Calculate derived signals
        signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, params)
        
        # Plot accelerations
        plot_accelerations(time, signals['x'], signals['y'], 
                          signals['theta_x'], signals['theta_y'], show_plot)
        
        # Calculate windowed frequency/amplitude
        window_duration = params['window_duration']
        output1, output2, debug_data = calculate_windowed_frequency_amplitude(
            time, signals['x'], signals['y'], 
            signals['theta_x'], signals['theta_y'],
            window_duration, fs
        )
        
        # Plot frequency measurement
        frequency_array = output1[:, 1]  # Extract frequency column
        plot_frequency_measurement(time, frequency_array, window_duration, debug_data, show_plot)
        
        # Calculate input force U and add to output2
        frequency_output2 = output2[:, 1]  # Frequency from output2
        U = calculate_input_force(frequency_output2, params['e_mass'])
        
        # Add U as 7th column to output2
        output2 = np.column_stack([output2, U])
        
        # Calculate FRFs
        frf_results = calculate_frfs(output2, dt)
        
        # Plot bode diagrams
        plot_bode_frfs(frf_results, show_plot)
        
        # Ask user to save output
        saved_file = save_output(output2, filepath)
        
        # Prepare return data
        result = {
            'output1': output1,
            'output2': output2,
            'frf_results': frf_results,
            'metadata': metadata,
            'saved_file': saved_file,
            'sample_rate': fs,
            'dt': dt
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
        
        # User option for plotting
        show_plot = messagebox.askyesno("Display Options", "Do you want to display the plots?")
        
        # Process the file
        result = process_eccentric_mass_file(filepath, show_plot)
        
        # Display results
        print(f"\nProcessing completed successfully!")
        print(f"Sample rate: {result['sample_rate']:.0f} Hz")
        print(f"Output1 shape: {result['output1'].shape}")
        print(f"Output2 shape: {result['output2'].shape}")
        
        if result['saved_file']:
            print(f"\nSaved file: {result['saved_file']}")
        
        if result['metadata']:
            print(f"\nMetadata:")
            print(f"  Date/Time: {result['metadata']['date_time']}")
            print(f"  Position: X={result['metadata']['x_position']}, "
                  f"Y={result['metadata']['y_position']}, "
                  f"Z={result['metadata']['z_position']}")
            print(f"  Direction: {result['metadata']['direction']}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
    
    finally:
        root.destroy()

if __name__ == "__main__":
    main()

