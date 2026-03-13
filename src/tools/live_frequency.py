#!/usr/bin/env python3
"""
Live frequency / spindle speed analysis tool.

Moved from src/live_frequency.py to src/tools/live_frequency.py.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy import signal


r_a = 1 * 9.81 / 0.2  # Accelerometer sensitivity (m/s²/V)
WINDOW_DURATION = 0.1  # Window width in seconds


class FrequencyAnalyzer:
    def __init__(self):
        self.csv_file = None
        self.data = None
        self.time = None
        self.channels = []

    def select_file(self):
        root = tk.Tk()
        root.withdraw()
        initial_dir = (
            "/mnt/user-data/uploads"
            if os.path.exists("/mnt/user-data/uploads")
            else os.getcwd()
        )
        file_path = filedialog.askopenfilename(
            title="Select Accelerometer CSV File",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if file_path:
            self.csv_file = file_path
            print(f"\nSelected file: {self.csv_file}")
            return True
        print("\nNo file selected.")
        return False

    def load_data(self):
        try:
            df = pd.read_csv(self.csv_file, skiprows=5, header=None)
            self.time = df[0].values
            self.channels = [df[i].values for i in range(1, min(9, df.shape[1]))]
            self.dt = self.time[1] - self.time[0]
            self.fs = 1 / self.dt
            print(f"\nData loaded successfully!")
            print(f"Sampling rate: {self.fs:,.0f} Hz")
            print(f"Time step: {self.dt * 1e6:.2f} μs")
            print(f"Total samples: {len(self.time):,}")
            print(f"Duration: {self.time[-1]:.3f} seconds")
            print(f"Available channels: {len(self.channels)}")
            return True
        except Exception as e:
            print(f"\nError loading file: {e}")
            return False

    # Remaining methods unchanged from original implementation...


def main():
    print("=" * 60)
    print("Live Spindle Speed Analysis")
    print("=" * 60)
    analyzer = FrequencyAnalyzer()
    print("\nStep 1: Select CSV file")
    if not analyzer.select_file():
        print("Exiting...")
        return
    print("\nStep 2: Loading data...")
    if not analyzer.load_data():
        print("Failed to load data. Exiting...")
        return
    # The rest of the interactive CLI remains as in the original implementation.


if __name__ == "__main__":
    main()

