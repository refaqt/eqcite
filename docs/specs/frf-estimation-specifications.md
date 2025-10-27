# Frequency Response Analysis Tool - Technical Specifications

## Project Overview

A Python application for analyzing frequency response measurements from CSV data files. The tool processes multi-channel analog input data, calculates spatial coordinates and angular displacements, performs FFT analysis, and generates transfer function (compliance) plots.

## 1. Input Requirements

### 1.1 CSV File Format

**File Naming Convention:**
```
yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_[direction].csv
```

Where:
- `yyyy-mm-dd_hh-mm-ss`: Date and time of measurement
- `X[x-position]`: X-axis position of the machine (numeric value)
- `Y[y-position]`: Y-axis position of the machine (numeric value)
- `Z[z-position]`: Z-axis position of the machine (numeric value)
- `[direction]`: Either "x" or "y" indicating force application direction

**Example:** `2025-10-17_14-30-45_X10_Y20_Z5_x.csv`

### 1.2 CSV Data Structure

- **Header Lines:** First 5 lines contain metadata (sample rate info and column headers)
- **Data Lines:** Starting from line 6
- **Column Count:** 9 columns total
  - Column 0: Time (seconds)
  - Columns 1-8: Analog inputs A0 through A7 (Volts)

**Expected Format:**
```
"Samples per sec. = 80000/1"
[empty line]
[empty line]
"Time","","","","","","","",""
"sec","Volt","Volt","Volt","Volt","Volt","Volt","Volt","Volt"
[data rows]
```

### 1.3 User Interface Requirements

The application must provide:
1. **File selection dialog** to load CSV file
2. **Option to display plots** (boolean toggle)
3. **Option to save output files** (boolean toggle)

## 2. System Parameters

Define the following constants in the application:

```python
# Geometric parameters (meters)
L1 = 0.01      # Length parameter 1
L2 = 0.05      # Length parameter 2
L3 = 0.2       # Length parameter 3 (Note: L3, not L2 again)

# Angular parameter
alpha = 20 * pi / 180  # Angle in radians (20 degrees)

# Sensor calibration parameters
r_a = 5 * 9.81 / 10    # Accelerometer sensitivity (m/s²/V)
m_u = 10               # IEPE signal conditioner multiplication factor
v_u = 2.5e-3           # Force hammer sensitivity (mV/N)
r_u = 1 / (v_u * m_u)  # Combined output sensitivity (N/V)
```

## 3. Data Processing Pipeline

### 3.1 Sample Time Estimation

Calculate the sample time (dt) from the time difference between the first two measurements:
```python
dt = time[1] - time[0]
sample_rate = 1 / dt
```

### 3.2 Signal Calculations

Calculate the following signals from analog inputs:

**Force Input:**
```python
u = A0 * r_u  # Convert to Newtons
```

**Angular Displacements:**
```python
theta_x = (A1 + A2 - A3 - A4) * sin(alpha) / (2 * L3)
theta_y = (A1 - A2 - A3 + A4) * cos(alpha) / (2 * L3)
```

**Linear Displacements:**
```python
x = (A1 - A2 + A3 - A4) * cos(alpha) / 4 - theta_y * (L1 + L2 + L3 / 2) + A6
y = -(A1 + A2 + A3 + A4) * sin(alpha) / 4 + theta_x * (L1 + L2 + L3 / 2) + A5
```

### 3.3 Frequency Domain Analysis

Perform FFT on the calculated signals:
```python
X = fft(x)  # FFT of x displacement
Y = fft(y)  # FFT of y displacement
U = fft(u)  # FFT of force input
```

Calculate transfer functions (frequency response functions):
```python
Gx = X / U  # Compliance in x-direction (m/N)
Gy = Y / U  # Compliance in y-direction (m/N)
```

Calculate frequency vector:
```python
N = length(time_series)
frequencies = fftfreq(N, dt)  # Frequency array in Hz
# Use only positive frequencies: frequencies[:N//2]
```

## 4. Visualization Requirements

### 4.1 Bode Diagram Specifications

Generate two Bode magnitude plots for Gx and Gy with the following characteristics:

**X-Axis:**
- Frequency in Hz
- Logarithmic scale
- Range: From first non-zero frequency to Nyquist frequency

**Y-Axis:**
- Display values in meters/Newton (m/N)
- Scale in decibels (dB)
- Convert magnitude to dB: `20 * log10(|G|)`
- **Important:** Axis should be labeled and scaled in dB, but tick labels should show the original linear values in m/N

**Calculation for axis labels:**
```python
# For a dB value on the axis:
dB_value = 20 * log10(magnitude_m_per_N)
# Display label: magnitude_m_per_N (not dB_value)
```

**Plot Titles:**
- "Compliance Gx (X/U)" for x-direction
- "Compliance Gy (Y/U)" for y-direction

**Additional Requirements:**
- Grid lines enabled
- Legend if multiple data series
- Appropriate axis labels: "Frequency (Hz)" and "Compliance (m/N)"

### 4.2 Display Control

User must be able to choose whether to display plots or skip visualization.

## 5. Output File Requirements

### 5.1 File Format

Generate output files containing frequency response data:

**File Naming Convention:**
```
yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_Gx.csv
yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_Gy.csv
```

The base filename is extracted from the input CSV, with the direction indicator ("x" or "y") replaced by "Gx" or "Gy".

### 5.2 File Content Structure

**CSV format with 3 columns:**

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Frequency (Hz) | Gx (m/N) | Gy (m/N) |

**Specifications:**
- Header row with column names: "Frequency_Hz", "Gx_m_per_N", "Gy_m_per_N"
- One row per frequency point
- Use only positive frequencies (up to Nyquist frequency)
- Values should be complex magnitudes: `|Gx|` and `|Gy|`
- Numeric precision: At least 6 significant figures

**Example:**
```csv
Frequency_Hz,Gx_m_per_N,Gy_m_per_N
0.0,1.234e-6,2.345e-6
1.25,3.456e-6,4.567e-6
2.50,5.678e-6,6.789e-6
...
```

### 5.3 Save Control

User must be able to choose whether to save output files or skip file generation.

## 6. Error Handling

The application must handle the following error conditions:

1. **File Not Found:** Invalid file path or file doesn't exist
2. **Invalid File Format:** CSV doesn't match expected structure
3. **Missing Columns:** Fewer than 9 columns in data
4. **Invalid Data:** Non-numeric values in measurement columns
5. **Zero Division:** Handle cases where U (force) is zero or near-zero
6. **Parsing Errors:** Malformed CSV or encoding issues

For each error, provide clear error messages to the user.

## 7. Dependencies

Required Python libraries:
- `numpy` - Array operations and FFT
- `scipy` - Signal processing (optional, can use numpy.fft)
- `matplotlib` - Plotting Bode diagrams
- `pandas` - CSV file handling
- `tkinter` - File dialog (or alternative GUI library)

## 8. Code Organization

The application should be implemented as a **single Python file** with modular functions. This design facilitates integration into larger programs without file clutter.

**Recommended Function Structure:**

```python
# frequency_response_analysis.py

# Constants and Parameters
def get_system_parameters():
    """Return dictionary of system parameters (L1, L2, L3, alpha, r_a, m_u, v_u, r_u)"""
    pass

# Data Loading
def load_csv_file(filepath):
    """Load and parse CSV file, return time array and analog input arrays (A0-A7)"""
    pass

def parse_filename(filepath):
    """Extract date, time, position (X, Y, Z), and direction from filename"""
    pass

# Signal Processing
def estimate_sample_time(time_array):
    """Calculate sample time from first two time points"""
    pass

def calculate_derived_signals(A0, A1, A2, A3, A4, A5, A6, A7, params):
    """Calculate u, x, y, theta_x, theta_y from analog inputs"""
    pass

# Frequency Domain Analysis
def perform_fft_analysis(time, u, x, y):
    """Perform FFT and return frequencies, X, Y, U in frequency domain"""
    pass

def calculate_transfer_functions(X, Y, U):
    """Calculate Gx = X/U and Gy = Y/U transfer functions"""
    pass

# Visualization
def plot_bode_diagram(frequencies, Gx, Gy, show_plot=True):
    """Generate Bode magnitude plots for Gx and Gy"""
    pass

# File Output
def save_results(frequencies, Gx, Gy, original_filepath, save_files=True):
    """Save frequency response data to CSV files"""
    pass

def generate_output_filename(original_filepath, suffix):
    """Generate output filename based on input filename and suffix (Gx or Gy)"""
    pass

# Main Processing
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
    pass

# User Interface
def main():
    """Entry point with file dialog and user options"""
    pass

if __name__ == "__main__":
    main()
```

**Function Organization Principles:**
- Each function has a single, clear responsibility
- Functions are organized by logical grouping (loading, processing, analysis, output)
- Main processing function `process_measurement_file()` orchestrates the workflow
- User interface is separated in `main()` for easy integration into other programs

## 9. Implementation Notes

### 9.1 FFT Considerations

- Use `numpy.fft.fft()` for forward transform
- Use `numpy.fft.fftfreq()` to generate frequency array
- Extract only positive frequencies: `freqs[:N//2]` and corresponding FFT values
- Handle DC component (frequency = 0) appropriately in transfer functions

### 9.2 Transfer Function Calculation

- Avoid division by zero: skip or interpolate frequencies where |U| < threshold
- Use complex division: `Gx = X / U` (preserves phase information)
- Calculate magnitude: `|Gx| = np.abs(Gx)` for plotting and output

### 9.3 Filename Parsing

Extract position and direction from input filename using regex:
```python
pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_X([^_]+)_Y([^_]+)_Z([^_]+)_([xy])\.csv'
```

### 9.4 Unit Conversions

All internal calculations should maintain consistent units:
- Length: meters (m)
- Force: Newtons (N)
- Time: seconds (s)
- Acceleration: m/s²
- Voltage: Volts (V)

## 10. Testing Requirements

Verify the application with test cases:

1. **Normal Operation:** Process provided test CSV file
2. **Edge Cases:** 
   - Very small force values
   - Files with different sample rates
   - Different filename formats
3. **Error Cases:**
   - Invalid CSV format
   - Missing columns
   - Corrupted data

## 11. Future Enhancements (Optional)

Consider implementing:
- Batch processing of multiple files
- Phase plot in addition to magnitude plot
- Coherence function calculation
- Frequency range selection
- Window functions for FFT (Hanning, Hamming, etc.)
- Data filtering/smoothing options
- Export plots as image files

## 12. Acceptance Criteria

The application is complete when it can:
1. ✓ Load CSV file via file dialog
2. ✓ Parse CSV with correct structure
3. ✓ Calculate all derived signals correctly
4. ✓ Perform FFT and compute transfer functions
5. ✓ Display Bode plots with correct formatting
6. ✓ Save output CSV files with correct naming and content
7. ✓ Provide user options for plot display and file saving
8. ✓ Handle errors gracefully with informative messages

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-17  
**Author:** Project Specifications