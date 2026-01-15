#!/usr/bin/env python3
"""
Channel Filling Tool

A Python application for converting 3-channel CSV files to 7-channel CSV files.
Reads a CSV file with 3 data channels and creates a new CSV file with 7 channels
by mapping the input channels to positions 5-7 and filling positions 1-4 with zeros.

Author: Channel Filling Tool
Version: 1.0
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import os

# =============================================================================
# CSV PARSING FUNCTIONS
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

def read_3channel_csv(filepath):
    """
    Read a 3-channel CSV file and return header lines, data arrays, and sample rate.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple : (header_lines, time_array, channel1, channel2, channel3, fs_old)
        header_lines : list of 5 header lines (strings)
        time_array : numpy array of time values
        channel1, channel2, channel3 : numpy arrays of channel data
        fs_old : float, old total samples per second
        
    Raises:
    -------
    ValueError : If file format is invalid or data cannot be parsed
    FileNotFoundError : If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read header lines (first 5 lines)
    header_lines = []
    try:
        with open(filepath, 'r') as f:
            for i in range(5):
                header_lines.append(f.readline().rstrip('\n\r'))
    except Exception as e:
        raise ValueError(f"Error reading CSV file header: {e}")
    
    # Parse sample rate from first line
    fs_old = parse_sample_rate_from_header(filepath)
    
    # Read CSV file, skipping first 5 header lines
    try:
        df = pd.read_csv(
            filepath, 
            skiprows=5, 
            header=None,
            sep=',',  # Explicit separator
            decimal='.',  # Explicit decimal point
            skipinitialspace=True  # Skip leading whitespace
        )
    except Exception as e:
        raise ValueError(f"Error reading CSV data: {e}")
    
    # Validate column count: expect 4 columns (Time + 3 data channels)
    n_cols = df.shape[1]
    if n_cols != 4:
        raise ValueError(
            f"Expected 4 columns (Time + 3 data channels), found {n_cols}"
        )
    
    # Convert each column to numeric with error handling
    for i in range(n_cols):
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
                f"Column {i} has {nan_count} non-numeric values. "
                f"First error at row {first_nan_idx + 5}. "
                f"Please check the CSV file format."
            )
    
    # Extract time and channel data
    time = df.iloc[:, 0].values  # Column 0: Time
    channel1 = df.iloc[:, 1].values  # Column 1: Channel 1
    channel2 = df.iloc[:, 2].values  # Column 2: Channel 2
    channel3 = df.iloc[:, 3].values  # Column 3: Channel 3
    
    # Validate data is numeric and finite
    for i, arr in enumerate([time, channel1, channel2, channel3]):
        if arr.dtype == object:
            raise ValueError(f"Column {i} contains non-numeric data")
        if not np.isfinite(arr).all():
            nan_count = (~np.isfinite(arr)).sum()
            raise ValueError(
                f"Column {i} has {nan_count} non-finite values (NaN or Inf)"
            )
    
    return header_lines, time, channel1, channel2, channel3, fs_old

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def create_7channel_data(time, ch1, ch2, ch3):
    """
    Create 7-channel data from 3-channel input.
    
    Channel mapping:
    - A0-A3 (channels 1-4): zeros
    - A4 (channel 5): ch1
    - A5 (channel 6): ch2
    - A6 (channel 7): ch3
    
    Parameters:
    -----------
    time : array_like
        Time array
    ch1, ch2, ch3 : array_like
        Input channel data arrays
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns [Time, A0, A1, A2, A3, A4, A5, A6]
    """
    n_samples = len(time)
    
    # Create output arrays
    A0 = np.zeros(n_samples)
    A1 = np.zeros(n_samples)
    A2 = np.zeros(n_samples)
    A3 = np.zeros(n_samples)
    A4 = np.asarray(ch1)
    A5 = np.asarray(ch2)
    A6 = np.asarray(ch3)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': time,
        'A0': A0,
        'A1': A1,
        'A2': A2,
        'A3': A3,
        'A4': A4,
        'A5': A5,
        'A6': A6
    })
    
    return df

# =============================================================================
# HEADER PROCESSING FUNCTIONS
# =============================================================================

def update_header_sample_rate(header_lines, fs_new):
    """
    Update the sample rate in the first header line.
    
    Parameters:
    -----------
    header_lines : list
        List of header line strings (5 lines)
    fs_new : float
        New total samples per second
        
    Returns:
    --------
    list : Updated header lines with new sample rate in first line
    """
    # Create a copy of header lines
    updated_header = header_lines.copy()
    
    # Update first line with new sample rate
    # Pattern: "Samples per sec. = XXXXX/1"
    pattern = r'(Samples\s+per\s+sec\.\s*=\s*)(\d+)(/\d+)'
    
    def replace_sample_rate(match):
        return f"{match.group(1)}{int(fs_new)}{match.group(3)}"
    
    updated_header[0] = re.sub(pattern, replace_sample_rate, updated_header[0], flags=re.IGNORECASE)
    
    return updated_header

# =============================================================================
# FILE OUTPUT FUNCTIONS
# =============================================================================

def write_7channel_csv(header_lines, data_df, output_path):
    """
    Write 7-channel CSV file with header and data.
    
    Parameters:
    -----------
    header_lines : list
        List of header line strings (5 lines)
    data_df : pd.DataFrame
        DataFrame with columns [Time, A0, A1, A2, A3, A4, A5, A6]
    output_path : str
        Path to output CSV file
        
    Raises:
    -------
    IOError : If file cannot be written
    """
    try:
        with open(output_path, 'w', newline='') as f:
            # Write header lines
            for line in header_lines:
                f.write(line + '\n')
            
            # Write data
            data_df.to_csv(f, index=False, header=False, float_format='%.6e')
            
    except Exception as e:
        raise IOError(f"Error writing output file: {e}")

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_file(input_filepath, output_filepath=None):
    """
    Process a 3-channel CSV file and create a 7-channel CSV file.
    
    Parameters:
    -----------
    input_filepath : str
        Path to input 3-channel CSV file
    output_filepath : str, optional
        Path to output 7-channel CSV file. If None, generates automatically.
        
    Returns:
    --------
    str : Path to output file
        
    Raises:
    -------
    Exception : If processing fails
    """
    try:
        # Read input file
        header_lines, time, ch1, ch2, ch3, fs_old = read_3channel_csv(input_filepath)
        
        # Calculate new sample rate
        # fs_old is total samples/sec for 3 channels
        # Per-channel rate = fs_old / 3
        # New total rate = (fs_old / 3) * 7
        fs_new = fs_old / 3 * 7
        
        # Create 7-channel data
        data_df = create_7channel_data(time, ch1, ch2, ch3)
        
        # Update header with new sample rate
        updated_header = update_header_sample_rate(header_lines, fs_new)
        
        # Generate output filename if not provided
        if output_filepath is None:
            input_dir = os.path.dirname(input_filepath)
            input_basename = os.path.basename(input_filepath)
            input_name, input_ext = os.path.splitext(input_basename)
            output_filepath = os.path.join(input_dir, f"{input_name}_filled{input_ext}")
        
        # Write output file
        write_7channel_csv(updated_header, data_df, output_filepath)
        
        return output_filepath
        
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

# =============================================================================
# USER INTERFACE
# =============================================================================

def main():
    """
    Entry point with file dialog and processing.
    """
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()
    
    try:
        # File selection dialog
        input_filepath = filedialog.askopenfilename(
            title="Select 3-channel CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not input_filepath:
            print("No file selected. Exiting.")
            return
        
        print(f"Processing file: {input_filepath}")
        
        # Process file
        output_filepath = process_file(input_filepath)
        
        # Show success message
        messagebox.showinfo(
            "Success",
            f"File processed successfully!\n\n"
            f"Output saved to:\n{output_filepath}"
        )
        
        print(f"Output saved to: {output_filepath}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
    
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
