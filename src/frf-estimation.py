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
    L1 = 0.01      # Length parameter 1
    L2 = 0.05      # Length parameter 2
    L3 = 0.2       # Length parameter 3
    
    # Angular parameter
    alpha = 20 * np.pi / 180  # Angle in radians (20 degrees)
    
    # Sensor calibration parameters
    r_a = 5 * 9.81 / 10    # Accelerometer sensitivity (m/s²/V)
    m_u = 10               # IEPE signal conditioner multiplication factor
    v_u = 2.5e-3           # Force hammer sensitivity (mV/N)
    r_u = 1 / (v_u * m_u)  # Combined output sensitivity (N/V)
    
    return {
        'L1': L1,
        'L2': L2,
        'L3': L3,
        'alpha': alpha,
        'r_a': r_a,
        'm_u': m_u,
        'v_u': v_u,
        'r_u': r_u
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
        df = pd.read_csv(filepath, skiprows=5, header=None)
        
        # Validate column count
        if df.shape[1] < 9:
            raise ValueError(f"Expected 9 columns, found {df.shape[1]}")
        
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
        
        # Validate data is numeric
        for i, arr in enumerate([time, A0, A1, A2, A3, A4, A5, A6, A7]):
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-numeric or infinite values found in column {i}")
        
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
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_X([^_]+)_Y([^_]+)_Z([^_]+)_([xy])\.csv'
    match = re.match(pattern, filename)
    
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
    """
    # Extract parameters
    L1 = params['L1']
    L2 = params['L2']
    L3 = params['L3']
    alpha = params['alpha']
    r_u = params['r_u']
    
    # Force input (convert to Newtons)
    u = A0 * r_u
    
    # Angular displacements
    theta_x = (A1 + A2 - A3 - A4) * np.sin(alpha) / (2 * L3)
    theta_y = (A1 - A2 - A3 + A4) * np.cos(alpha) / (2 * L3)
    
    # Linear displacements
    x = (A1 - A2 + A3 - A4) * np.cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2) + A6
    y = -(A1 + A2 + A3 + A4) * np.sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2) + A5
    
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
    
    Parameters:
    -----------
    time : array_like
        Time array in seconds
    u, x, y : array_like
        Time domain signals
        
    Returns:
    --------
    dict : Dictionary containing frequency array and FFT results
    """
    N = len(time)
    dt = time[1] - time[0]
    
    # Perform FFT
    U = np.fft.fft(u)
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    
    # Generate frequency array
    frequencies = np.fft.fftfreq(N, dt)
    
    # Use only positive frequencies (including DC)
    positive_freq_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_freq_mask]
    U_pos = U[positive_freq_mask]
    X_pos = X[positive_freq_mask]
    Y_pos = Y[positive_freq_mask]
    
    return {
        'frequencies': frequencies_pos,
        'U': U_pos,
        'X': X_pos,
        'Y': Y_pos,
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

def plot_bode_diagram(frequencies, Gx, Gy, show_plot=True):
    """
    Generate Bode magnitude plots for Gx and Gy.
    
    Parameters:
    -----------
    frequencies : array_like
        Frequency array in Hz
    Gx, Gy : array_like
        Transfer functions (complex)
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    # Calculate magnitudes
    Gx_mag = np.abs(Gx)
    Gy_mag = np.abs(Gy)
    
    # Remove NaN values for plotting
    valid_freq = frequencies[np.isfinite(Gx_mag) & np.isfinite(Gy_mag)]
    valid_Gx = Gx_mag[np.isfinite(Gx_mag) & np.isfinite(Gy_mag)]
    valid_Gy = Gy_mag[np.isfinite(Gx_mag) & np.isfinite(Gy_mag)]
    
    # Skip DC component for log plot
    if len(valid_freq) > 1 and valid_freq[0] == 0:
        valid_freq = valid_freq[1:]
        valid_Gx = valid_Gx[1:]
        valid_Gy = valid_Gy[1:]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Gx
    ax1.loglog(valid_freq, valid_Gx, 'b-', linewidth=2, label='Gx')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Compliance (m/N)')
    ax1.set_title('Compliance Gx (X/U)')
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend()
    
    # Plot Gy
    ax2.loglog(valid_freq, valid_Gy, 'r-', linewidth=2, label='Gy')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Compliance (m/N)')
    ax2.set_title('Compliance Gy (Y/U)')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend()
    
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
    pattern = r'(_[xy]\.csv)$'
    new_filename = re.sub(pattern, f'_{suffix}.csv', filename)
    
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
        
        # Calculate derived signals
        signals = calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params)
        
        # Perform FFT analysis
        fft_results = perform_fft_analysis(time, signals['u'], signals['x'], signals['y'])
        
        # Calculate transfer functions
        tf_results = calculate_transfer_functions(fft_results['X'], fft_results['Y'], fft_results['U'])
        
        # Generate plots
        plot_bode_diagram(fft_results['frequencies'], tf_results['Gx'], tf_results['Gy'], show_plot)
        
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
