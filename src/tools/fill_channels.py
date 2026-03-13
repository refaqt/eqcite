#!/usr/bin/env python3
"""
Channel Filling Tool

Moved from src/fill_channels.py to src/tools/fill_channels.py.
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd


def parse_sample_rate_from_header(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        with open(filepath, "r") as f:
            first_line = f.readline()
    except Exception as e:
        raise ValueError(f"Error reading CSV file header: {e}")
    pattern = r"Samples\s+per\s+sec\.\s*=\s*(\d+)/\d+"
    match = re.search(pattern, first_line, re.IGNORECASE)
    if match:
        return float(match.group(1))
    raise ValueError(
        "Could not find sample rate in CSV header. "
        "Expected format: 'Samples per sec. = XXXXX/1'. "
        f"First line: {first_line.strip()}"
    )


def read_3channel_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    header_lines = []
    with open(filepath, "r") as f:
        for _ in range(5):
            header_lines.append(f.readline().rstrip("\n\r"))
    fs_old = parse_sample_rate_from_header(filepath)
    df = pd.read_csv(
        filepath,
        skiprows=5,
        header=None,
        sep=",",
        decimal=".",
        skipinitialspace=True,
    )
    n_cols = df.shape[1]
    if n_cols != 4:
        raise ValueError(f"Expected 4 columns (Time + 3 data channels), found {n_cols}")
    for i in range(n_cols):
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors="coerce")
        nan_count = df.iloc[:, i].isna().sum()
        if nan_count > 0:
            first_nan_idx = df.iloc[:, i].isna().idxmax()
            raise ValueError(
                f"Column {i} has {nan_count} non-numeric values. "
                f"First error at row {first_nan_idx + 5}."
            )
    time = df.iloc[:, 0].values
    channel1 = df.iloc[:, 1].values
    channel2 = df.iloc[:, 2].values
    channel3 = df.iloc[:, 3].values
    return header_lines, time, channel1, channel2, channel3, fs_old


def create_7channel_data(time, ch1, ch2, ch3):
    n_samples = len(time)
    A0 = np.zeros(n_samples)
    A1 = np.zeros(n_samples)
    A2 = np.zeros(n_samples)
    A3 = np.zeros(n_samples)
    A4 = np.asarray(ch1)
    A5 = np.asarray(ch2)
    A6 = np.asarray(ch3)
    df = pd.DataFrame(
        {
            "Time": time,
            "A0": A0,
            "A1": A1,
            "A2": A2,
            "A3": A3,
            "A4": A4,
            "A5": A5,
            "A6": A6,
        }
    )
    return df


def update_header_sample_rate(header_lines, fs_new):
    updated_header = header_lines.copy()
    pattern = r"(Samples\s+per\s+sec\.\s*=\s*)(\d+)(/\d+)"

    def replace_sample_rate(match):
        return f"{match.group(1)}{int(fs_new)}{match.group(3)}"

    updated_header[0] = re.sub(pattern, replace_sample_rate, updated_header[0], flags=re.IGNORECASE)
    return updated_header


def write_7channel_csv(header_lines, data_df, output_path):
    with open(output_path, "w", newline="") as f:
        for line in header_lines:
            f.write(line + "\n")
        data_df.to_csv(f, index=False, header=False, float_format="%.6e")


def process_file(input_filepath, output_filepath=None):
    header_lines, time, ch1, ch2, ch3, fs_old = read_3channel_csv(input_filepath)
    fs_new = fs_old / 3 * 7
    data_df = create_7channel_data(time, ch1, ch2, ch3)
    updated_header = update_header_sample_rate(header_lines, fs_new)
    if output_filepath is None:
        input_dir = os.path.dirname(input_filepath)
        input_basename = os.path.basename(input_filepath)
        input_name, input_ext = os.path.splitext(input_basename)
        output_filepath = os.path.join(input_dir, f"{input_name}_filled{input_ext}")
    write_7channel_csv(updated_header, data_df, output_filepath)
    return output_filepath


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        input_filepath = filedialog.askopenfilename(
            title="Select 3-channel CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not input_filepath:
            print("No file selected. Exiting.")
            return
        print(f"Processing file: {input_filepath}")
        output_filepath = process_file(input_filepath)
        messagebox.showinfo(
            "Success",
            f"File processed successfully!\n\nOutput saved to:\n{output_filepath}",
        )
        print(f"Output saved to: {output_filepath}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()

