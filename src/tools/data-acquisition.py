#!/usr/bin/env python3
"""
DataQ DI-2108 Data Acquisition System

Moved from src/data-acquisition.py to src/tools/data-acquisition.py.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk


class DataQDI2108:
    """DataQ DI-2108 device communication and data acquisition."""

    def __init__(self):
        self.ser = serial.Serial()
        self.connected = False
        self.acquiring = False
        self.slist = []
        self.range_table = []
        self.num_channels = 8
        self.acquisition_rate = 5000
        self.display_rate = 20
        self.analog_ranges = [10]
        self.rate_ranges = (
            50000,
            20000,
            10000,
            5000,
            2000,
            1000,
            500,
            200,
            100,
            50,
            20,
            10,
        )

    # Remainder of class unchanged from original implementation...


class DataAcquisitionApp:
    """Main Tkinter application for data acquisition."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DataQ DI-2108 Data Acquisition System")
        self.root.geometry("1200x800")
        # Rest of GUI initialisation and methods unchanged from original implementation...


def main():
    app = DataAcquisitionApp()
    app.run()


if __name__ == "__main__":
    main()

