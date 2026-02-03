#!/usr/bin/env python3
"""
Stability Measurement Analysis Tool

A Python application for analyzing cutting cycle accelerations from CSV data files.
Computes time-frequency spectrograms to detect chatter onset during machining operations.

Author: Stability Analysis Tool
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import warnings
import importlib.util
import re
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

# Import functions from frf-estimation.py
# Use importlib to handle hyphenated module name
spec = importlib.util.spec_from_file_location("frf_estimation", 
                                               os.path.join(os.path.dirname(__file__), "frf-estimation.py"))
frf_estimation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frf_estimation)

# Import required functions
load_csv_file = frf_estimation.load_csv_file
get_system_parameters = frf_estimation.get_system_parameters
remove_bias = frf_estimation.remove_bias
apply_wiener_filter = frf_estimation.apply_wiener_filter
apply_savgol_filter = frf_estimation.apply_savgol_filter
apply_lowpass_filter = frf_estimation.apply_lowpass_filter
calculate_derived_signals = frf_estimation.calculate_derived_signals
plot_time_domain_signals = frf_estimation.plot_time_domain_signals
calculate_noise_power = frf_estimation.calculate_noise_power
plot_signal_comparison = frf_estimation.plot_signal_comparison
detect_case3_from_filename = frf_estimation.detect_case3_from_filename
parse_filename = frf_estimation.parse_filename

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Spectrogram parameters
DEFAULT_WINDOW_DURATION = 0.1  # Window duration in seconds (similar to live_frequency.py)
DEFAULT_OVERLAP = 0.5  # Overlap fraction (0 to 1)
DEFAULT_WINDOW_TYPE = 'hann'  # Window type for spectrogram

# Milling and chatter detection parameters
LOWPASS_CUTOFF_HZ = 2000.0  # Low-pass filter cutoff frequency (Hz)
MILLING_START_THRESHOLD = 20.0  # Threshold for milling start detection (m/s²)
CHATTER_THRESHOLD = 40.0  # Threshold for chatter detection (m/s²)

# =============================================================================
# FILENAME PARSING FUNCTIONS
# =============================================================================

def parse_depth_of_cut_from_filename(filepath):
    """
    Extract depth of cut range from filename pattern Ap_[start]to[end].
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple or None : (depth_start, depth_end) in mm, or None if not found
    """
    filename = os.path.basename(filepath)
    
    # Pattern: Ap_[start]to[end] (case-insensitive)
    pattern = r'Ap_([\d.]+)to([\d.]+)'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        depth_start = float(match.group(1))
        depth_end = float(match.group(2))
        return (depth_start, depth_end)
    
    return None

# =============================================================================
# ROLLING STATISTICS FUNCTIONS
# =============================================================================

def calculate_rolling_std_2x(signal, fs, window_duration, overlap):
    """
    Calculate 2x standard deviation using same window size and hop as spectrogram.
    
    Parameters:
    -----------
    signal : array_like
        Input signal array
    fs : float
        Sampling frequency (Hz)
    window_duration : float
        Window duration in seconds (same as spectrogram)
    overlap : float
        Overlap fraction (0 to 1, same as spectrogram)
        
    Returns:
    --------
    tuple : (rolling_std_2x, time_centers)
        rolling_std_2x : Array of 2x standard deviation values
        time_centers : Array of time centers matching spectrogram time_centers
    """
    signal = np.asarray(signal)
    nperseg = int(window_duration * fs)
    hop = int(nperseg * (1 - overlap))
    
    n_samples = len(signal)
    n_windows = (n_samples - nperseg) // hop + 1
    
    rolling_std_2x = np.zeros(n_windows)
    time_centers = np.zeros(n_windows)
    
    for i in range(n_windows):
        start_idx = i * hop
        end_idx = start_idx + nperseg
        
        if end_idx > n_samples:
            # Last window might be shorter
            window_data = signal[start_idx:]
        else:
            window_data = signal[start_idx:end_idx]
        
        # Calculate 2x standard deviation
        std_val = np.std(window_data)
        rolling_std_2x[i] = 2.0 * std_val
        
        # Calculate time center (middle of window)
        center_idx = start_idx + len(window_data) // 2
        time_centers[i] = center_idx / fs
    
    return rolling_std_2x, time_centers

# =============================================================================
# MILLING DETECTION FUNCTIONS
# =============================================================================

def detect_milling_start(rolling_std_2x, time_centers, threshold):
    """
    Detect milling start by finding first time where 2x std exceeds threshold.
    
    Parameters:
    -----------
    rolling_std_2x : array_like
        Array of 2x standard deviation values
    time_centers : array_like
        Array of time centers (seconds)
    threshold : float
        Threshold value (m/s²)
        
    Returns:
    --------
    float or None : Milling start time in seconds, or None if not detected
    """
    exceeding_indices = np.where(rolling_std_2x >= threshold)[0]
    
    if len(exceeding_indices) > 0:
        return time_centers[exceeding_indices[0]]
    
    return None

def detect_milling_end(rolling_std_2x, time_centers, start_time, threshold):
    """
    Detect milling end by finding first time after start where 2x std drops below threshold.
    
    Parameters:
    -----------
    rolling_std_2x : array_like
        Array of 2x standard deviation values
    time_centers : array_like
        Array of time centers (seconds)
    start_time : float
        Milling start time (seconds)
    threshold : float
        Threshold value (m/s²) - same as milling start threshold
        
    Returns:
    --------
    float or None : Milling end time in seconds, or None if not detected
    """
    if start_time is None:
        return None
    
    # Find index corresponding to start_time
    start_idx = np.argmin(np.abs(time_centers - start_time))
    
    # Look for first time after start where std drops below threshold
    for i in range(start_idx, len(rolling_std_2x)):
        if rolling_std_2x[i] < threshold:
            return time_centers[i]
    
    # If never drops below threshold, return last time
    return time_centers[-1] if len(time_centers) > 0 else None

def detect_chatter_start(rolling_std_2x, time_centers, start_time, threshold):
    """
    Detect chatter start by finding first time after milling start where 2x std exceeds threshold.
    
    Parameters:
    -----------
    rolling_std_2x : array_like
        Array of 2x standard deviation values
    time_centers : array_like
        Array of time centers (seconds)
    start_time : float
        Milling start time (seconds)
    threshold : float
        Chatter threshold value (m/s²)
        
    Returns:
    --------
    float or None : Chatter start time in seconds, or None if not detected
    """
    if start_time is None:
        return None
    
    # Find index corresponding to start_time
    start_idx = np.argmin(np.abs(time_centers - start_time))
    
    # Look for first time after start where std exceeds chatter threshold
    for i in range(start_idx, len(rolling_std_2x)):
        if rolling_std_2x[i] >= threshold:
            return time_centers[i]
    
    return None

# =============================================================================
# SPECTROGRAM FUNCTIONS
# =============================================================================

def calculate_spectrogram(signal, fs, window_duration=0.1, overlap=0.5, window_type='hann', mfft=None):
    """
    Calculate time-frequency spectrogram using scipy.signal.ShortTimeFFT.
    
    Parameters:
    -----------
    signal : array_like
        Acceleration signal array
    fs : float
        Sampling frequency (Hz)
    window_duration : float
        Window duration in seconds (default: 0.1)
    overlap : float
        Overlap fraction (0 to 1, default: 0.5)
    window_type : str
        Window type string (default 'hann') or window array
    mfft : int, optional
        FFT size (None = use next power of 2 >= window size)
        
    Returns:
    --------
    tuple : (SFT, Sxx)
        SFT : ShortTimeFFT instance
        Sxx : 2D array of power spectral density (shape: [frequencies, time_slices])
    """
    # Create window
    nperseg = int(window_duration * fs)
    
    if window_type == 'hann':
        win = hann(nperseg, sym=True)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    # Calculate hop size
    hop = int(nperseg * (1 - overlap))
    
    # Determine FFT size if not provided
    if mfft is None:
        # Use next power of 2 >= nperseg
        mfft = int(2 ** np.ceil(np.log2(nperseg)))
    
    # Create ShortTimeFFT instance
    SFT = ShortTimeFFT(win, hop=hop, fs=fs, mfft=mfft, scale_to='psd')
    
    # Calculate spectrogram
    Sxx = SFT.spectrogram(signal)
    
    return SFT, Sxx

def plot_spectrogram(SFT, Sxx, N, title="Spectrogram", signal_label="Acceleration", show_plot=True,
                     milling_start_time=None, milling_end_time=None):
    """
    Plot 2D intensity plot (spectrogram) using ShortTimeFFT extent information.
    Uses black-to-yellow colormap and optionally marks milling start/end times.
    
    Parameters:
    -----------
    SFT : ShortTimeFFT
        ShortTimeFFT instance (contains frequency and time information)
    Sxx : array_like
        2D array of power spectral density (from SFT.spectrogram())
    N : int
        Number of samples in original signal
    title : str
        Plot title
    signal_label : str
        Label for signal type (e.g., "X Acceleration", "Y Acceleration")
    show_plot : bool
        Whether to display the plot (if False, figure is created but not shown)
    milling_start_time : float, optional
        Milling start time in seconds (for vertical line marker)
    milling_end_time : float, optional
        Milling end time in seconds (for vertical line marker)
        
    Returns:
    --------
    matplotlib.figure.Figure : The figure object
    """
    # Get time and frequency extent
    t_lo, t_hi, f_lo, f_hi = SFT.extent(N)
    
    # Convert to dB for visualization
    Sxx_dB = 10 * np.log10(np.fmax(Sxx, 1e-4))  # Limit range, avoid log(0)
    
    # Create custom black-to-yellow colormap
    colors = ['black', 'yellow']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('black_to_yellow', colors, N=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use imshow with extent for proper axis labeling
    im = ax.imshow(Sxx_dB, origin='lower', aspect='auto',
                   extent=(t_lo, t_hi, f_lo, f_hi),
                   cmap=cmap, interpolation='nearest')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(f'{title} - {signal_label}', fontsize=14, fontweight='bold')
    
    # Add milling start/end markers
    if milling_start_time is not None:
        ax.axvline(x=milling_start_time, color='r', linestyle=':', 
                  linewidth=2, label=f'Milling start: {milling_start_time:.3f} s')
    if milling_end_time is not None:
        ax.axvline(x=milling_end_time, color='r', linestyle=':', 
                  linewidth=2, label=f'Milling end: {milling_end_time:.3f} s')
    
    if milling_start_time is not None or milling_end_time is not None:
        ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power Spectral Density (dB)', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig

def detect_chatter(spectrogram, frequencies, time_centers, threshold_factor=2.0):
    """
    Detect chatter onset by identifying significant magnitude increases.
    
    Parameters:
    -----------
    spectrogram : array_like
        2D array of power spectral density (shape: [frequencies, time_slices])
    frequencies : array_like
        Frequency array (Hz)
    time_centers : array_like
        Time center array (seconds)
    threshold_factor : float
        Factor to multiply baseline magnitude for threshold (default: 2.0)
        
    Returns:
    --------
    dict : Dictionary containing:
        'chatter_times': array of estimated chatter onset times
        'chatter_frequencies': array of frequencies where chatter occurs
        'chatter_mask': 2D boolean array indicating chatter regions
    """
    # Calculate baseline (first 10% of time)
    baseline_end_idx = max(1, int(len(time_centers) * 0.1))
    baseline = np.mean(spectrogram[:, :baseline_end_idx], axis=1, keepdims=True)
    
    # Calculate threshold
    threshold = baseline * threshold_factor
    
    # Find regions where magnitude exceeds threshold
    chatter_mask = spectrogram > threshold
    
    # Find time indices where chatter occurs (any frequency exceeds threshold)
    chatter_time_indices = np.where(np.any(chatter_mask, axis=0))[0]
    
    if len(chatter_time_indices) > 0:
        # First occurrence is the estimated onset
        first_chatter_idx = chatter_time_indices[0]
        chatter_onset_time = time_centers[first_chatter_idx]
        
        # Find frequencies where chatter occurs at onset
        chatter_freq_indices = np.where(chatter_mask[:, first_chatter_idx])[0]
        chatter_frequencies = frequencies[chatter_freq_indices]
    else:
        chatter_onset_time = None
        chatter_frequencies = np.array([])
    
    return {
        'chatter_times': time_centers[chatter_time_indices] if len(chatter_time_indices) > 0 else np.array([]),
        'chatter_frequencies': chatter_frequencies,
        'chatter_onset_time': chatter_onset_time,
        'chatter_mask': chatter_mask
    }

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_stability_measurement(filepath, window_duration=DEFAULT_WINDOW_DURATION, 
                                  overlap=DEFAULT_OVERLAP, show_plot=True):
    """
    Main function to process a stability measurement file.
    
    Parameters:
    -----------
    filepath : str
        Path to input CSV file
    window_duration : float
        Window duration in seconds for spectrogram
    overlap : float
        Overlap fraction (0 to 1) for spectrogram
    show_plot : bool
        Whether to display plots
        
    Returns:
    --------
    dict : Dictionary containing processing results
    """
    try:
        # Load system parameters
        params = get_system_parameters()
        
        # Load CSV data
        print(f"Loading CSV file: {filepath}")
        time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt = load_csv_file(filepath)
        print(f"Loaded {len(time)} samples at {fs:.0f} Hz")
        
        # Store original signals for comparison
        original_signals = {
            'A0': A0.copy(), 'A1': A1.copy(), 'A2': A2.copy(), 'A3': A3.copy(),
            'A4': A4.copy(), 'A5': A5.copy(), 'A6': A6.copy(), 'A7': A7.copy()
        }
        
        # Remove bias from all analog input signals
        window_size = params['bias_window_size']
        print("Removing bias from signals...")
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
        
        # Apply Wiener filter if enabled
        use_wiener = params.get('wiener_filter', True)
        if use_wiener:
            print("Applying Wiener filter...")
            # Calculate noise power for each channel
            noise_A0 = calculate_noise_power(original_signals['A0'], bias_idx_A0, actual_window_A0)
            noise_A1 = calculate_noise_power(original_signals['A1'], bias_idx_A1, actual_window_A1)
            noise_A2 = calculate_noise_power(original_signals['A2'], bias_idx_A2, actual_window_A2)
            noise_A3 = calculate_noise_power(original_signals['A3'], bias_idx_A3, actual_window_A3)
            noise_A4 = calculate_noise_power(original_signals['A4'], bias_idx_A4, actual_window_A4)
            noise_A5 = calculate_noise_power(original_signals['A5'], bias_idx_A5, actual_window_A5)
            noise_A6 = calculate_noise_power(original_signals['A6'], bias_idx_A6, actual_window_A6)
            noise_A7 = calculate_noise_power(original_signals['A7'], bias_idx_A7, actual_window_A7)
            
            # Apply Wiener filter
            A0 = apply_wiener_filter(A0, noise_A0, actual_window_A0)
            A1 = apply_wiener_filter(A1, noise_A1, actual_window_A1)
            A2 = apply_wiener_filter(A2, noise_A2, actual_window_A2)
            A3 = apply_wiener_filter(A3, noise_A3, actual_window_A3)
            A4 = apply_wiener_filter(A4, noise_A4, actual_window_A4)
            A5 = apply_wiener_filter(A5, noise_A5, actual_window_A5)
            A6 = apply_wiener_filter(A6, noise_A6, actual_window_A6)
            A7 = apply_wiener_filter(A7, noise_A7, actual_window_A7)
        
        # Apply Savitzky-Golay filter if enabled
        use_savgol = params.get('savgol_filter', False)
        if use_savgol:
            print("Applying Savitzky-Golay filter...")
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
        
        # Store filtered signals for comparison plots
        filtered_signals = {
            'A0': A0, 'A1': A1, 'A2': A2, 'A3': A3,
            'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7
        }
        
        # Show signal comparison plots
        if show_plot:
            print("Plotting signal comparison (original, bias-corrected, filtered)...")
            plot_signal_comparison(time, original_signals, corrected_signals, filtered_signals, show_plot)
        
        # Detect measurement case from filename
        measurement_case = detect_case3_from_filename([filepath])
        if not measurement_case:
            measurement_case = 'case1'  # Default to case1 if not case3
        
        # Calculate derived signals (x, y accelerations)
        print(f"Calculating derived signals (measurement case: {measurement_case})...")
        signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params, measurement_case=measurement_case)
        x_accel = signals['x']  # X-direction acceleration (m/s²)
        y_accel = signals['y']  # Y-direction acceleration (m/s²)
        
        # Extract z acceleration if case3
        z_accel = signals.get('z', None) if measurement_case == 'case3' else None
        
        # Apply low-pass filter to x and y accelerations
        print(f"Applying low-pass filter (cutoff: {LOWPASS_CUTOFF_HZ} Hz)...")
        x_accel_lowpass = apply_lowpass_filter(x_accel, LOWPASS_CUTOFF_HZ, fs)
        y_accel_lowpass = apply_lowpass_filter(y_accel, LOWPASS_CUTOFF_HZ, fs)
        
        # Plot time-domain signals
        if show_plot:
            print("Plotting time-domain signals...")
            plot_time_domain_signals(time, signals['u'], x_accel, y_accel, show_plot=True, z_accel=z_accel)
            
            # Plot low-pass filtered signal comparison
            print("Plotting low-pass filtered signal comparison...")
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # X acceleration
            axes[0].plot(time, x_accel, 'b-', linewidth=1, alpha=0.7, label='Original X')
            axes[0].plot(time, x_accel_lowpass, 'r-', linewidth=1, alpha=0.7, label=f'Low-pass Filtered X ({LOWPASS_CUTOFF_HZ} Hz)')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Acceleration (m/s²)')
            axes[0].set_title('X Acceleration - Low-Pass Filter Comparison')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Y acceleration
            axes[1].plot(time, y_accel, 'b-', linewidth=1, alpha=0.7, label='Original Y')
            axes[1].plot(time, y_accel_lowpass, 'r-', linewidth=1, alpha=0.7, label=f'Low-pass Filtered Y ({LOWPASS_CUTOFF_HZ} Hz)')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Acceleration (m/s²)')
            axes[1].set_title('Y Acceleration - Low-Pass Filter Comparison')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            plt.tight_layout()
        
        # Calculate rolling 2x standard deviation for detection
        print(f"Calculating rolling 2x standard deviation (window: {window_duration}s, overlap: {overlap*100:.0f}%)...")
        rolling_std_2x_x, time_centers_std_x = calculate_rolling_std_2x(x_accel_lowpass, fs, window_duration, overlap)
        rolling_std_2x_y, time_centers_std_y = calculate_rolling_std_2x(y_accel_lowpass, fs, window_duration, overlap)
        
        # Determine detection direction from filename
        try:
            filename_metadata = parse_filename(filepath)
            detection_direction = filename_metadata['direction'].lower()
        except (ValueError, KeyError):
            # Default to 'x' if parsing fails
            detection_direction = 'x'
            print(f"Warning: Could not parse direction from filename, defaulting to 'x'")
        
        # Select appropriate signal for detection
        if detection_direction == 'x':
            rolling_std_2x_detection = rolling_std_2x_x
            time_centers_detection = time_centers_std_x
            detection_signal_label = 'X'
        else:
            rolling_std_2x_detection = rolling_std_2x_y
            time_centers_detection = time_centers_std_y
            detection_signal_label = 'Y'
        
        # Detect milling start
        print(f"Detecting milling start (threshold: {MILLING_START_THRESHOLD} m/s², direction: {detection_signal_label})...")
        milling_start_time = detect_milling_start(rolling_std_2x_detection, time_centers_detection, MILLING_START_THRESHOLD)
        
        if milling_start_time is not None:
            print(f"Milling start detected at: {milling_start_time:.3f} s")
        else:
            print("Warning: Milling start not detected")
        
        # Detect chatter start (after milling start)
        print(f"Detecting chatter start (threshold: {CHATTER_THRESHOLD} m/s²)...")
        chatter_start_time = detect_chatter_start(rolling_std_2x_detection, time_centers_detection, 
                                                   milling_start_time, CHATTER_THRESHOLD)
        
        if chatter_start_time is not None:
            print(f"Chatter start detected at: {chatter_start_time:.3f} s")
        else:
            print("Warning: Chatter start not detected")
        
        # Detect milling end
        print(f"Detecting milling end (threshold: {MILLING_START_THRESHOLD} m/s²)...")
        milling_end_time = detect_milling_end(rolling_std_2x_detection, time_centers_detection, 
                                              milling_start_time, MILLING_START_THRESHOLD)
        
        if milling_end_time is not None:
            print(f"Milling end detected at: {milling_end_time:.3f} s")
        else:
            print("Warning: Milling end not detected")
        
        # Parse depth of cut from filename and calculate cutting depth at chatter start
        depth_of_cut_info = parse_depth_of_cut_from_filename(filepath)
        cutting_depth_at_chatter = None
        
        if depth_of_cut_info is not None and milling_start_time is not None and milling_end_time is not None and chatter_start_time is not None:
            depth_start, depth_end = depth_of_cut_info
            # Calculate cutting depth at chatter start
            if milling_end_time > milling_start_time:
                cutting_depth_at_chatter = (chatter_start_time - milling_start_time) / (milling_end_time - milling_start_time) * (depth_end - depth_start) + depth_start
                print(f"\nCutting depth at chatter start: {cutting_depth_at_chatter:.3f} mm")
                print(f"  (Depth range: {depth_start:.3f} to {depth_end:.3f} mm)")
            else:
                print("Warning: Invalid milling time range for depth calculation")
        elif depth_of_cut_info is None:
            print("Warning: Could not parse depth of cut from filename (expected pattern: Ap_[start]to[end])")
        
        # Calculate spectrograms for x and y accelerations
        print(f"Calculating spectrograms (window: {window_duration}s, overlap: {overlap*100:.0f}%)...")
        SFT_x, Sxx_x = calculate_spectrogram(x_accel, fs, window_duration, overlap)
        SFT_y, Sxx_y = calculate_spectrogram(y_accel, fs, window_duration, overlap)
        
        # Get frequency and time information
        N = len(time)
        t_lo_x, t_hi_x, f_lo_x, f_hi_x = SFT_x.extent(N)
        t_lo_y, t_hi_y, f_lo_y, f_hi_y = SFT_y.extent(N)
        
        # Create time and frequency arrays for plotting
        # Number of time slices
        n_time_x = Sxx_x.shape[1]
        n_time_y = Sxx_y.shape[1]
        time_centers_x = np.linspace(t_lo_x, t_hi_x, n_time_x)
        time_centers_y = np.linspace(t_lo_y, t_hi_y, n_time_y)
        
        # Frequency bins
        frequencies_x = SFT_x.f
        frequencies_y = SFT_y.f
        
        # Detect chatter before plotting
        print("Detecting chatter...")
        chatter_x = detect_chatter(Sxx_x, frequencies_x, time_centers_x)
        chatter_y = detect_chatter(Sxx_y, frequencies_y, time_centers_y)
        
        # Print chatter detection results
        if chatter_x['chatter_onset_time'] is not None:
            print(f"\nX-direction chatter detected at: {chatter_x['chatter_onset_time']:.3f} s")
            if len(chatter_x['chatter_frequencies']) > 0:
                print(f"  Frequencies: {', '.join([f'{f:.1f}' for f in chatter_x['chatter_frequencies'][:5]])} Hz" + 
                      (f" ({len(chatter_x['chatter_frequencies'])} total)" if len(chatter_x['chatter_frequencies']) > 5 else ""))
        else:
            print("\nNo X-direction chatter detected")
        
        if chatter_y['chatter_onset_time'] is not None:
            print(f"\nY-direction chatter detected at: {chatter_y['chatter_onset_time']:.3f} s")
            if len(chatter_y['chatter_frequencies']) > 0:
                print(f"  Frequencies: {', '.join([f'{f:.1f}' for f in chatter_y['chatter_frequencies'][:5]])} Hz" + 
                      (f" ({len(chatter_y['chatter_frequencies'])} total)" if len(chatter_y['chatter_frequencies']) > 5 else ""))
        else:
            print("\nNo Y-direction chatter detected")
        
        # Plot spectrograms with milling and chatter markers
        if show_plot:
            print("Plotting spectrograms...")
            # Plot X acceleration spectrogram
            fig_x = plot_spectrogram(SFT_x, Sxx_x, N, title="Stability Spectrogram", 
                                     signal_label="X Acceleration", show_plot=False,
                                     milling_start_time=milling_start_time,
                                     milling_end_time=milling_end_time)
            if fig_x is not None:
                ax = fig_x.axes[0]
                if chatter_x['chatter_onset_time'] is not None:
                    ax.axvline(x=chatter_x['chatter_onset_time'], color='r', linestyle='--', 
                              linewidth=2, label=f"Chatter onset: {chatter_x['chatter_onset_time']:.3f} s")
                    ax.legend()
            
            # Plot Y acceleration spectrogram
            fig_y = plot_spectrogram(SFT_y, Sxx_y, N, title="Stability Spectrogram", 
                                     signal_label="Y Acceleration", show_plot=False,
                                     milling_start_time=milling_start_time,
                                     milling_end_time=milling_end_time)
            if fig_y is not None:
                ax = fig_y.axes[0]
                if chatter_y['chatter_onset_time'] is not None:
                    ax.axvline(x=chatter_y['chatter_onset_time'], color='r', linestyle='--', 
                              linewidth=2, label=f"Chatter onset: {chatter_y['chatter_onset_time']:.3f} s")
                    ax.legend()
        
        # Prepare return data
        result = {
            'filepath': filepath,
            'time': time,
            'x_accel': x_accel,
            'y_accel': y_accel,
            'x_accel_lowpass': x_accel_lowpass,
            'y_accel_lowpass': y_accel_lowpass,
            'SFT_x': SFT_x,
            'SFT_y': SFT_y,
            'Sxx_x': Sxx_x,
            'Sxx_y': Sxx_y,
            'frequencies_x': frequencies_x,
            'frequencies_y': frequencies_y,
            'time_centers_x': time_centers_x,
            'time_centers_y': time_centers_y,
            'chatter_x': chatter_x,
            'chatter_y': chatter_y,
            'milling_start_time': milling_start_time,
            'milling_end_time': milling_end_time,
            'chatter_start_time': chatter_start_time,
            'cutting_depth_at_chatter': cutting_depth_at_chatter,
            'rolling_std_2x_x': rolling_std_2x_x,
            'rolling_std_2x_y': rolling_std_2x_y,
            'time_centers_std_x': time_centers_std_x,
            'time_centers_std_y': time_centers_std_y,
            'fs': fs,
            'dt': dt,
            'N': N
        }
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing file {filepath}: {str(e)}"
        print(f"Error: {error_msg}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise Exception(error_msg) from e

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
        
        print(f"Selected file: {filepath}")
        
        # Process the file
        result = process_stability_measurement(filepath, show_plot=True)
        
        print("\nProcessing completed successfully!")
        print(f"Sample rate: {result['fs']:.0f} Hz")
        print(f"Number of samples: {result['N']}")
        print(f"Time duration: {result['time'][-1]:.3f} seconds")
        
        # Wait for user to close plots
        plt.show()
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred: {str(e)}"
        traceback_str = traceback.format_exc()
        messagebox.showerror("Error", error_msg)
        print(f"Error: {error_msg}")
        print("\nFull traceback:")
        print(traceback_str)
    
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
