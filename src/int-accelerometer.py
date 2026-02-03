import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import sys
from scipy import signal

# Constants
r_a = 5 * 9.81 / 10  # Accelerometer sensitivity (m/s²/V)
HIGHPASS_CUTOFF_HZ = 10  # High-pass filter cutoff frequency (Hz)


class AccelerometerAnalyzer:
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

    def apply_highpass_filter(self, data, cutoff_hz):
        """Apply 4th-order Bessel high-pass filter to the data"""
        # Design 4th-order Bessel high-pass filter
        b, a = signal.bessel(N=4, Wn=cutoff_hz, btype='high', analog=False, fs=self.fs)
        
        # Apply zero-phase filtering using filtfilt
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data

    def process_channel(self, channel_num):
        """Process a selected channel with double integration using discrete-time formula"""
        # Convert channel number to index (1-based to 0-based)
        channel_idx = channel_num - 1

        # Get voltage data for selected channel
        voltage = self.channels[channel_idx]

        # Convert voltage to acceleration
        acceleration = voltage * r_a

        # Remove DC component (average value)
        dc_component = np.mean(acceleration)
        acceleration_ac = acceleration - dc_component
        
        # Store unfiltered data for comparison
        acceleration_unfiltered = acceleration_ac.copy()

        # Apply 10Hz high-pass filter
        acceleration_ac = self.apply_highpass_filter(acceleration_ac, HIGHPASS_CUTOFF_HZ)

        print(f"\nProcessing Channel {channel_num}:")
        print(f"DC component removed: {dc_component:.6f} m/s²")
        print(f"Applied {HIGHPASS_CUTOFF_HZ}Hz high-pass filter (4th-order Bessel)")
        print(f"Using discrete-time integration: u(n) = u(n-1) + Ts * i(n)")

        # Get sampling time
        Ts = self.dt

        # Initialize arrays
        n_samples = len(acceleration_ac)
        velocity = np.zeros(n_samples)
        displacement = np.zeros(n_samples)

        # First integration: acceleration to velocity
        # u(n) = u(n-1) + Ts * i(n), with u(0) = 0
        for n in range(1, n_samples):
            velocity[n] = velocity[n - 1] + Ts * acceleration_ac[n]

        # Remove linear drift in velocity by removing DC component (average value)
        velocity_dc = np.mean(velocity)
        velocity_detrended = velocity - velocity_dc
        
        print(f"Velocity DC component removed: {velocity_dc:.6f} m/s")

        # Second integration: velocity to displacement
        # u(n) = u(n-1) + Ts * i(n), with u(0) = 0
        for n in range(1, n_samples):
            displacement[n] = displacement[n - 1] + Ts * velocity_detrended[n]

        # Convert from meters to micrometers
        displacement_micrometers = displacement * 1e6

        return {
            'displacement': displacement_micrometers,
            'acceleration': acceleration_ac,
            'acceleration_unfiltered': acceleration_unfiltered,
            'velocity': velocity_detrended,
            'dc_component': dc_component,
            'channel': channel_num
        }

    def plot_results(self, results):
        """Plot the displacement results"""
        displacement = results['displacement']
        acceleration = results['acceleration']
        acceleration_unfiltered = results['acceleration_unfiltered']
        channel = results['channel']

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot displacement
        ax1.plot(self.time, displacement, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Displacement (μm)', fontsize=12)
        ax1.set_title(f'Double Integration Results - Channel {channel} (Discrete-time: u(n) = u(n-1) + Ts*i(n))',
                      fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.5)

        # Plot both filtered and unfiltered acceleration on same subplot
        ax2.plot(self.time, acceleration_unfiltered, 'g-', linewidth=0.8, alpha=0.7, label='Unfiltered')
        ax2.plot(self.time, acceleration, 'r-', linewidth=0.8, alpha=0.7, label='10Hz HPF')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax2.set_title('Acceleration Signal Comparison (DC removed)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.5)
        ax2.legend()

        # Calculate statistics
        stats = {
            'Sampling time (Ts)': f"{self.dt * 1e6:.2f} μs",
            'Max displacement': f"{np.max(displacement):.3f} μm",
            'Min displacement': f"{np.min(displacement):.3f} μm",
            'Peak-to-peak': f"{np.ptp(displacement):.3f} μm",
            'RMS displacement': f"{np.sqrt(np.mean(displacement ** 2)):.3f} μm",
            'Max accel (unfiltered)': f"{np.max(np.abs(acceleration_unfiltered)):.6f} m/s²",
            'RMS accel (unfiltered)': f"{np.sqrt(np.mean(acceleration_unfiltered ** 2)):.6f} m/s²",
            'Max accel (filtered)': f"{np.max(np.abs(acceleration)):.6f} m/s²",
            'RMS accel (filtered)': f"{np.sqrt(np.mean(acceleration ** 2)):.6f} m/s²"
        }

        # Add statistics text box
        stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7),
                 verticalalignment='bottom')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for stats

        # Print statistics to console as well
        print("\nStatistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

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
    print("Accelerometer Double Integration Analysis")
    print("=" * 60)

    analyzer = AccelerometerAnalyzer()

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
            results = analyzer.process_channel(channel)
            analyzer.plot_results(results)
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