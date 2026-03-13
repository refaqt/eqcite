#!/usr/bin/env python3
"""
Legacy eccentric mass FRF estimation tool.

Moved from src/eccentric-mass-frf.py to src/legacy/eccentric-mass-frf.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from scipy import signal


# Original implementation preserved for reference / backward compatibility...

