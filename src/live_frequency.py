import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import sys
import time
from scipy import signal

# Constants
r_a = 1 * 9.81 / 0.2  # Accelerometer sensitivity (m/s²/V)
WINDOW_DURATION = 0.1  # Window width in seconds


class FrequencyAnalyzer:
    def __init__(self):
        self.csv_file = None
        self.data = None
        self.time = None
        self.channels = []

    def select_file(self):
        """Open file dialog to select CSV file"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Set initial directory to uploads folder if it exists
        initial_dir = "/mnt/user-data/uploads" if os.path.exists("/mnt/user-data/uploads") else os.getcwd()

        file_path = filedialog.askopenfilename(
            title="Select Accelerometer CSV File",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        root.destroy()

        if file_path:
            self.csv_file = file_path
            print(f"\nSelected file: {self.csv_file}")
            return True
        else:
            print("\nNo file selected.")
            return False

    def load_data(self):
        """Load and parse the CSV file"""
        try:
            # Read the CSV file, skipping the header rows
            df = pd.read_csv(self.csv_file, skiprows=5, header=None)

            # Extract time and channels
            self.time = df[0].values
            self.channels = [df[i].values for i in
                             range(1, min(9, df.shape[1]))]  # Channels 1-8 (or fewer if less available)

            # Calculate sampling rate from time data
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

    def process_channel(self, channel_num):
        """Process a selected channel and return acceleration signal"""
        # Convert channel number to index (1-based to 0-based)
        channel_idx = channel_num - 1

        # Get voltage data for selected channel
        voltage = self.channels[channel_idx]

        # Convert voltage to acceleration
        acceleration = voltage * r_a

        # Remove DC component (average value)
        acceleration_ac = acceleration - np.mean(acceleration)

        print(f"\nProcessing Channel {channel_num}:")
        print(f"DC component removed: {np.mean(acceleration):.6f} m/s²")
        print(f"Signal range: {np.min(acceleration_ac):.6f} to {np.max(acceleration_ac):.6f} m/s²")

        return acceleration_ac

    def calculate_windowed_frequency(self, acceleration):
        """Calculate frequency using sliding Hann window with FFT"""
        # Calculate window size in samples
        window_samples = int(WINDOW_DURATION * self.fs)
        
        # Create Hann window
        hann_window = np.hanning(window_samples)
        
        # Initialize frequency array
        n_samples = len(acceleration)
        frequency_array = np.zeros(n_samples)
        
        # Initialize debug data storage
        self.debug_data = {}
        
        print(f"\nCalculating windowed frequency (for spindle speed conversion):")
        print(f"Window duration: {WINDOW_DURATION} seconds")
        print(f"Window samples: {window_samples}")
        print(f"Processing {n_samples} samples...")
        
        # Loop through signal with sliding window
        for i in range(n_samples - window_samples + 1):
            # Extract windowed segment
            windowed_signal = acceleration[i:i+window_samples] * hann_window
            
            # Apply FFT
            fft_result = np.fft.rfft(windowed_signal)
            
            # Get frequency bins
            freqs = np.fft.rfftfreq(window_samples, d=self.dt)
            
            # Find dominant frequency (exclude DC component)
            magnitude = np.abs(fft_result[1:])  # Skip DC component
            freq_bins = freqs[1:]  # Skip DC component
            
            if len(magnitude) > 0:
                # Find index of maximum magnitude
                max_idx = np.argmax(magnitude)
                
                # Initialize debug variables
                interpolated = False
                poly_coeffs = None
                
                # Check if we have at least 1 point on each side for 3-point interpolation
                if max_idx >= 1 and max_idx < len(magnitude) - 1:
                    # Extract 3 magnitude points around the maximum (max_idx-1, max_idx, max_idx+1)
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
                        else:
                            # Fall back to bin maximum
                            dominant_freq = freq_bins[max_idx]
                    else:
                        # Fall back to bin maximum if no valid peak
                        dominant_freq = freq_bins[max_idx]
                else:
                    # Near edges, use simple maximum
                    dominant_freq = freq_bins[max_idx]
            else:
                dominant_freq = 0
                max_idx = 0
            
            # Store frequency at window center position
            center_idx = i + window_samples // 2
            if center_idx < n_samples:
                frequency_array[center_idx] = dominant_freq
                
                # Store debug data for this window position
                self.debug_data[center_idx] = {
                    'windowed_signal': windowed_signal.copy(),
                    'fft_magnitude': magnitude.copy(),
                    'fft_freqs': freq_bins.copy(),
                    'dominant_freq': dominant_freq,
                    'window_start_idx': i,
                    'window_end_idx': i + window_samples - 1,
                    'max_idx': max_idx,
                    'interpolated': interpolated,
                    'poly_coeffs': poly_coeffs,
                    'window_center_time': self.time[center_idx]
                }
        
        # Create time array for frequency results
        time_freq = self.time
        
        return frequency_array, time_freq

    def on_click(self, event):
        """Handle click events on the frequency plot"""
        # Check if click is in the frequency plot (ax1)
        if event.inaxes != self.ax1:
            return
        
        # Check if click is a left mouse button
        if event.button != 1:
            return
        
        # Check if zoom/pan tools are active by checking toolbar mode
        if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar:
            toolbar = self.fig.canvas.toolbar
            if hasattr(toolbar, 'mode') and toolbar.mode != '':
                return  # Navigation tool is active
        
        # Check if modifier keys are pressed (zoom/pan typically use these)
        if event.key and event.key.lower() in ['ctrl', 'shift', 'alt', 'control']:
            return
        
        # Check if this is a drag operation by detecting rapid successive clicks
        current_time = time.time()
        if hasattr(self, '_last_click_time'):
            if current_time - self._last_click_time < 0.15:  # Less than 150ms between clicks
                return
        
        # Store click time for drag detection
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
        """Create debug plot window showing windowed signal, FFT, and frequency details"""
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

    def plot_results(self, acceleration, frequency_array, time_freq, channel_num):
        """Plot frequency and acceleration results"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Store figure and axes as instance variables for event handler access
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

        # Convert frequency to spindle speed (RPM)
        # Assuming 1 Hz = 60 RPM for direct conversion
        spindle_rpm = frequency_array * 60

        # Plot spindle speed vs time (top)
        ax1.plot(time_freq, spindle_rpm, 'r-', linewidth=1.0)
        ax1.set_ylabel('Spindle Speed (RPM)', fontsize=12)
        ax1.set_title(f'Live Spindle Speed Analysis - Channel {channel_num} (Window: {WINDOW_DURATION}s)', 
                      fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.5)

        # Plot acceleration vs time (bottom)
        ax2.plot(self.time, acceleration, 'b-', linewidth=0.8)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax2.set_title('Acceleration Signal', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.5)

        # Calculate statistics
        valid_freqs = frequency_array[frequency_array > 0]  # Exclude zero frequencies
        if len(valid_freqs) > 0:
            valid_rpm = valid_freqs * 60  # Convert to RPM
            stats = {
                'Window duration': f"{WINDOW_DURATION} s",
                'Max spindle speed': f"{np.max(valid_rpm):.0f} RPM",
                'Min spindle speed': f"{np.min(valid_rpm):.0f} RPM",
                'Mean spindle speed': f"{np.mean(valid_rpm):.0f} RPM",
                'Std spindle speed': f"{np.std(valid_rpm):.0f} RPM",
                'Max acceleration': f"{np.max(np.abs(acceleration)):.6f} m/s²",
                'RMS acceleration': f"{np.sqrt(np.mean(acceleration ** 2)):.6f} m/s²"
            }
        else:
            stats = {
                'Window duration': f"{WINDOW_DURATION} s",
                'Max spindle speed': "N/A",
                'Min spindle speed': "N/A",
                'Mean spindle speed': "N/A",
                'Std spindle speed': "N/A",
                'Max acceleration': f"{np.max(np.abs(acceleration)):.6f} m/s²",
                'RMS acceleration': f"{np.sqrt(np.mean(acceleration ** 2)):.6f} m/s²"
            }

        # Add statistics text box
        stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7),
                 verticalalignment='bottom')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for stats

        # Connect click event handler for interactive debugging
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add instruction text to the plot
        fig.text(0.5, 0.02, "Click on frequency plot to debug frequency calculation", 
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Print statistics to console as well
        print("\nStatistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        print("\nInteractive Debug: Click on any point in the frequency plot to see detailed analysis!")

    def get_channel_input(self):
        """Get channel number from user input"""
        while True:
            try:
                channel = int(input(f"\nEnter channel number (1-{len(self.channels)}): "))
                if 1 <= channel <= len(self.channels):
                    return channel
                else:
                    print(f"Please enter a number between 1 and {len(self.channels)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None


def main():
    print("=" * 60)
    print("Live Spindle Speed Analysis")
    print("=" * 60)

    analyzer = FrequencyAnalyzer()

    # Step 1: Select file
    print("\nStep 1: Select CSV file")
    if not analyzer.select_file():
        print("Exiting...")
        return

    # Step 2: Load data
    print("\nStep 2: Loading data...")
    if not analyzer.load_data():
        print("Failed to load data. Exiting...")
        return

    # Main loop for channel analysis
    while True:
        # Step 3: Get channel selection
        channel = analyzer.get_channel_input()
        if channel is None:
            break

        # Step 4: Process and plot
        try:
            # Process channel to get acceleration signal
            acceleration = analyzer.process_channel(channel)
            
            # Calculate windowed frequency
            frequency_array, time_freq = analyzer.calculate_windowed_frequency(acceleration)
            
            # Plot results
            analyzer.plot_results(acceleration, frequency_array, time_freq, channel)
            plt.show()

            # Ask if user wants to analyze another channel
            again = input("\nAnalyze another channel? (y/n): ").lower()
            if again != 'y':
                break

        except Exception as e:
            print(f"Error processing channel: {e}")

    print("\nAnalysis complete. Goodbye!")


if __name__ == "__main__":
    main()
