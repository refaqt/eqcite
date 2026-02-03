#!/usr/bin/env python3
"""
DataQ DI-2108 Data Acquisition System
Continuous high-speed streaming with real-time display and on-demand CSV recording.

Based on specifications in docs/data-acquisition-specifications.md
Uses DataQ protocol from docs/Dataq-Instruments-Protocol.pdf
Reference implementation from docs/DataqStarterKit.py
"""

import serial
import serial.tools.list_ports
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque
import os
from datetime import datetime
import queue


class DataQDI2108:
    """DataQ DI-2108 device communication and data acquisition."""
    
    def __init__(self):
        self.ser = serial.Serial()
        self.connected = False
        self.acquiring = False
        self.slist = []
        self.range_table = []
        self.num_channels = 8
        self.acquisition_rate = 5000  # Hz
        self.display_rate = 20  # Hz
        
        # DI-2108 specific configuration
        self.analog_ranges = [10]  # ±10V fixed range
        self.rate_ranges = tuple((50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10))
        
    def discovery(self):
        """Discover and connect to DataQ DI-2108 device."""
        available_ports = list(serial.tools.list_ports.comports())
        hooked_port = ""
        
        for p in available_ports:
            if "VID:PID=0683" in p.hwid:
                hooked_port = p.device
                break
                
        if hooked_port:
            print(f"Found DataQ DI-2108 on {hooked_port}")
            self.ser.timeout = 0
            self.ser.port = hooked_port
            self.ser.baudrate = 115200
            self.ser.open()
            self.connected = True
            return True
        else:
            print("No DataQ DI-2108 device found")
            return False
    
    def send_cmd(self, command):
        """Send command to device and read echo if not acquiring."""
        if not self.connected:
            return
            
        self.ser.write((command + '\r').encode())
        time.sleep(0.1)
        
        if not self.acquiring:
            # Read echo
            while True:
                if self.ser.in_waiting > 0:
                    try:
                        s = self.ser.readline().decode()
                        s = s.strip('\n\r\0')
                        if s != "":
                            print(s)
                            break
                    except:
                        continue
                else:
                    break
    
    def compute_srate_dec(self, target_hz):
        """Compute srate and dec values for target sampling rate."""
        base = 60_000_000
        for dec in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            srate = round(base / (target_hz * dec))
            if 1 <= srate <= 65535:
                actual = base / (srate * dec)
                return int(srate), int(dec), actual
        raise ValueError(f"Cannot achieve sampling rate {target_hz} Hz")
    
    def config_scan_list(self, num_channels):
        """Configure scan list for specified number of channels."""
        self.slist = []
        self.range_table = []
        
        # Configure analog channels A0 to A7 (±10V range)
        for i in range(num_channels):
            # DI-2108 analog channel configuration: 0x0000 + channel number
            slist_item = 0x0000 + i
            self.slist.append(slist_item)
            self.range_table.append(self.analog_ranges[0])  # ±10V
        
        # Send scan list configuration
        for position, item in enumerate(self.slist):
            self.send_cmd(f"slist {position} {item}")
    
    def configure_device(self, num_channels, acquisition_rate):
        """Configure device for continuous acquisition."""
        if not self.connected:
            return False
            
        try:
            # Stop any existing acquisition
            self.send_cmd("stop")
            
            # Configure binary output mode
            self.send_cmd("encode 0")
            
            # Set small packet size for responsiveness
            self.send_cmd("ps 0")
            
            # Configure scan list
            self.config_scan_list(num_channels)
            
            # Set sampling rate
            srate, dec, actual_rate = self.compute_srate_dec(acquisition_rate)
            self.send_cmd(f"dec {dec}")
            self.send_cmd(f"srate {srate}")
            
            print(f"Configured for {actual_rate:.1f} Hz (target: {acquisition_rate} Hz)")
            
            # Start continuous acquisition
            self.send_cmd("start")
            self.acquiring = True
            
            return True
            
        except Exception as e:
            print(f"Device configuration error: {e}")
            return False
    
    def read_data_frame(self):
        """Read one complete data frame from device."""
        if not self.connected or not self.acquiring:
            return None
            
        # Wait for enough data for one frame
        frame_size = 2 * len(self.slist)
        if self.ser.in_waiting < frame_size:
            return None
            
        frame_data = []
        for i in range(len(self.slist)):
            # Read 2 bytes per channel
            bytes_data = self.ser.read(2)
            if len(bytes_data) != 2:
                return None
                
            # Convert to voltage (analog channels only for DI-2108)
            raw = int.from_bytes(bytes_data, byteorder='little', signed=True)
            volts = self.range_table[i] * raw / 32768.0
            frame_data.append(volts)
            
        return frame_data
    
    def close(self):
        """Close device connection."""
        if self.connected:
            self.send_cmd("stop")
            self.ser.close()
            self.connected = False
            self.acquiring = False


class DataAcquisitionApp:
    """Main Tkinter application for data acquisition."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DataQ DI-2108 Data Acquisition System")
        self.root.geometry("1200x800")
        
        # Device and data handling
        self.device = DataQDI2108()
        self.data_queue = queue.Queue()
        self.display_buffers = [deque(maxlen=1000) for _ in range(8)]  # Ring buffers for display
        self.acquisition_buffer = None
        self.acquisition_active = False
        self.acquisition_samples = 0
        self.acquisition_target = 0
        
        # Threading
        self.reader_thread = None
        self.run_event = threading.Event()
        self.data_lock = threading.Lock()
        
        # Configuration
        self.num_channels = 8
        self.channel_names = [f"A{i}" for i in range(8)]
        self.channel_units = ["V"] * 8
        self.acquisition_rate = 5000
        self.display_rate = 20
        self.acquisition_duration = 10.0
        self.measurement_direction = "x"
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.output_directory = "data"
        
        # GUI variables
        self.status_var = tk.StringVar(value="Disconnected")
        self.recording_var = tk.BooleanVar(value=False)
        
        self.setup_gui()
        self.connect_device()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Configuration", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Right panel for plot
        plot_frame = ttk.LabelFrame(main_frame, text="Live Display")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(control_frame)
        self.setup_plot(plot_frame)
        
    def setup_controls(self, parent):
        """Setup control panel widgets."""
        row = 0
        
        # Channel configuration
        ttk.Label(parent, text="Number of Channels:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.num_channels_var = tk.IntVar(value=self.num_channels)
        channels_spin = ttk.Spinbox(parent, from_=1, to=8, textvariable=self.num_channels_var, width=10)
        channels_spin.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Channel names
        ttk.Label(parent, text="Channel Names:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.channel_names_vars = []
        for i in range(8):
            var = tk.StringVar(value=self.channel_names[i])
            self.channel_names_vars.append(var)
            entry = ttk.Entry(parent, textvariable=var, width=8)
            entry.grid(row=row, column=1+i, sticky=tk.W, padx=1, pady=2)
        row += 1
        
        # Acquisition rate
        ttk.Label(parent, text="Acquisition Rate (Hz):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.acquisition_rate_var = tk.IntVar(value=self.acquisition_rate)
        ttk.Entry(parent, textvariable=self.acquisition_rate_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Display rate
        ttk.Label(parent, text="Display Rate (Hz):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.display_rate_var = tk.IntVar(value=self.display_rate)
        ttk.Entry(parent, textvariable=self.display_rate_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Acquisition duration
        ttk.Label(parent, text="Duration (s):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.duration_var = tk.DoubleVar(value=self.acquisition_duration)
        ttk.Entry(parent, textvariable=self.duration_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Measurement direction
        ttk.Label(parent, text="Direction:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.direction_var = tk.StringVar(value=self.measurement_direction)
        direction_frame = ttk.Frame(parent)
        direction_frame.grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(direction_frame, text="X", variable=self.direction_var, value="x").pack(side=tk.LEFT)
        ttk.Radiobutton(direction_frame, text="Y", variable=self.direction_var, value="y").pack(side=tk.LEFT)
        row += 1
        
        # Position inputs
        ttk.Label(parent, text="Position:").grid(row=row, column=0, sticky=tk.W, pady=2)
        pos_frame = ttk.Frame(parent)
        pos_frame.grid(row=row, column=1, columnspan=8, sticky=tk.W, pady=2)
        
        ttk.Label(pos_frame, text="X:").pack(side=tk.LEFT)
        self.pos_x_var = tk.DoubleVar(value=self.position_x)
        ttk.Entry(pos_frame, textvariable=self.pos_x_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(pos_frame, text="Y:").pack(side=tk.LEFT)
        self.pos_y_var = tk.DoubleVar(value=self.position_y)
        ttk.Entry(pos_frame, textvariable=self.pos_y_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(pos_frame, text="Z:").pack(side=tk.LEFT)
        self.pos_z_var = tk.DoubleVar(value=self.position_z)
        ttk.Entry(pos_frame, textvariable=self.pos_z_var, width=8).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Output directory
        ttk.Label(parent, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=2)
        dir_frame = ttk.Frame(parent)
        dir_frame.grid(row=row, column=1, columnspan=8, sticky=tk.W, pady=2)
        
        self.output_dir_var = tk.StringVar(value=self.output_directory)
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=20).pack(side=tk.LEFT)
        ttk.Button(dir_frame, text="Choose...", command=self.choose_output_directory).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=8, pady=10)
        
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        self.start_button = ttk.Button(button_frame, text="Start Acquisition", command=self.toggle_acquisition)
        self.start_button.pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Status
        ttk.Label(parent, text="Status:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(parent, textvariable=self.status_var).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
    def setup_plot(self, parent):
        """Setup the live plot area."""
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Initialize plot lines
        self.lines = []
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
        for i in range(8):
            line, = self.ax.plot([], [], color=colors[i], label=f"A{i}", linewidth=1)
            self.lines.append(line)
        
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.set_title('Live Data Display')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def choose_output_directory(self):
        """Open directory chooser dialog."""
        directory = filedialog.askdirectory(initialdir=self.output_directory)
        if directory:
            self.output_dir_var.set(directory)
            self.output_directory = directory
    
    def connect_device(self):
        """Connect to DataQ device and start streaming."""
        if self.device.discovery():
            self.status_var.set("Connecting...")
            if self.device.configure_device(self.num_channels, self.acquisition_rate):
                self.status_var.set("Streaming")
                self.start_reader_thread()
                self.start_display_timer()
            else:
                self.status_var.set("Configuration Failed")
                messagebox.showerror("Error", "Failed to configure device")
        else:
            self.status_var.set("Device Not Found")
            messagebox.showerror("Error", "DataQ DI-2108 device not found. Please connect device and restart.")
    
    def start_reader_thread(self):
        """Start the data reader thread."""
        self.run_event.set()
        self.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.reader_thread.start()
    
    def reader_loop(self):
        """Main data reading loop running in separate thread."""
        while self.run_event.is_set():
            if self.device.connected and self.device.acquiring:
                frame_data = self.device.read_data_frame()
                if frame_data:
                    timestamp = time.time()
                    
                    with self.data_lock:
                        # Update display buffers
                        for i, value in enumerate(frame_data):
                            if i < len(self.display_buffers):
                                self.display_buffers[i].append((timestamp, value))
                        
                        # Update acquisition buffer if recording
                        if self.acquisition_active and self.acquisition_buffer is not None:
                            if self.acquisition_samples < self.acquisition_target:
                                for i, value in enumerate(frame_data):
                                    if i < self.acquisition_buffer.shape[1]:
                                        self.acquisition_buffer[self.acquisition_samples, i] = value
                                self.acquisition_samples += 1
                            else:
                                # Acquisition complete
                                self.acquisition_active = False
                                self.root.after(0, self.acquisition_complete)
            else:
                time.sleep(0.001)  # Small delay when not acquiring
    
    def start_display_timer(self):
        """Start the display update timer."""
        self.update_display()
    
    def update_display(self):
        """Update the live display plot."""
        if not self.run_event.is_set():
            return
            
        with self.data_lock:
            # Get current time window (last 30 seconds)
            current_time = time.time()
            time_window = 30.0
            
            # Calculate decimation factor
            decimation_factor = max(1, self.acquisition_rate // self.display_rate)
            
            # Update plot data
            for i, line in enumerate(self.lines):
                if i < len(self.display_buffers) and i < self.num_channels:
                    buffer = self.display_buffers[i]
                    if len(buffer) > 0:
                        # Get recent data within time window
                        recent_data = [(t, v) for t, v in buffer if current_time - t <= time_window]
                        
                        if recent_data:
                            # Decimate for display
                            times, values = zip(*recent_data[::decimation_factor])
                            times = [t - current_time for t in times]  # Relative time
                            
                            line.set_data(times, values)
                            
                            # Update line label
                            line.set_label(self.channel_names_vars[i].get())
                        else:
                            line.set_data([], [])
                    else:
                        line.set_data([], [])
            
            # Update plot limits
            if any(len(self.display_buffers[i]) > 0 for i in range(self.num_channels)):
                self.ax.set_xlim(-time_window, 0)
                self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Redraw
            self.canvas.draw_idle()
        
        # Schedule next update
        update_interval = int(1000 / self.display_rate)  # Convert to milliseconds
        self.root.after(update_interval, self.update_display)
    
    def save_parameters(self):
        """Save configuration parameters and reconfigure device."""
        # Update configuration from GUI
        self.num_channels = self.num_channels_var.get()
        self.acquisition_rate = self.acquisition_rate_var.get()
        self.display_rate = self.display_rate_var.get()
        self.acquisition_duration = self.duration_var.get()
        self.measurement_direction = self.direction_var.get()
        self.position_x = self.pos_x_var.get()
        self.position_y = self.pos_y_var.get()
        self.position_z = self.pos_z_var.get()
        self.output_directory = self.output_dir_var.get()
        
        # Update channel names
        self.channel_names = [self.channel_names_vars[i].get() for i in range(8)]
        
        # Reconfigure device
        self.status_var.set("Reconfiguring...")
        if self.device.configure_device(self.num_channels, self.acquisition_rate):
            self.status_var.set("Streaming")
            messagebox.showinfo("Success", "Parameters saved and device reconfigured")
        else:
            self.status_var.set("Reconfiguration Failed")
            messagebox.showerror("Error", "Failed to reconfigure device")
    
    def toggle_acquisition(self):
        """Toggle acquisition recording."""
        if not self.acquisition_active:
            self.start_acquisition()
        else:
            self.stop_acquisition()
    
    def start_acquisition(self):
        """Start recording acquisition data."""
        if not self.device.connected:
            messagebox.showerror("Error", "Device not connected")
            return
        
        # Calculate target samples
        self.acquisition_target = int(self.acquisition_duration * self.acquisition_rate)
        self.acquisition_samples = 0
        
        # Allocate acquisition buffer
        self.acquisition_buffer = np.zeros((self.acquisition_target, self.num_channels))
        
        # Start acquisition
        self.acquisition_active = True
        self.recording_var.set(True)
        self.start_button.config(text="Stop Acquisition")
        self.status_var.set("Recording")
        
        print(f"Started acquisition: {self.acquisition_target} samples at {self.acquisition_rate} Hz")
    
    def stop_acquisition(self):
        """Stop recording acquisition data."""
        self.acquisition_active = False
        self.recording_var.set(False)
        self.start_button.config(text="Start Acquisition")
        self.status_var.set("Streaming")
        
        if self.acquisition_buffer is not None and self.acquisition_samples > 0:
            self.acquisition_complete()
    
    def acquisition_complete(self):
        """Handle completion of acquisition."""
        self.acquisition_active = False
        self.recording_var.set(False)
        self.start_button.config(text="Start Acquisition")
        self.status_var.set("Processing...")
        
        if self.acquisition_buffer is not None and self.acquisition_samples > 0:
            # Save to CSV
            self.save_csv()
            
            # Show post-acquisition plot
            self.show_post_acquisition_plot()
        
        self.status_var.set("Streaming")
    
    def save_csv(self):
        """Save acquisition data to CSV file."""
        if self.acquisition_buffer is None or self.acquisition_samples == 0:
            return
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_directory, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_X{self.position_x}_Y{self.position_y}_Z{self.position_z}_{self.measurement_direction}.csv"
            filepath = os.path.join(self.output_directory, filename)
            
            # Create time vector
            time_vector = np.arange(self.acquisition_samples) / self.acquisition_rate
            
            # Create DataFrame
            data_dict = {'Time': time_vector}
            for i in range(self.num_channels):
                data_dict[self.channel_names[i]] = self.acquisition_buffer[:self.acquisition_samples, i]
            
            df = pd.DataFrame(data_dict)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            messagebox.showinfo("Success", f"Data saved to:\n{filepath}")
            print(f"Saved {self.acquisition_samples} samples to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")
            print(f"CSV save error: {e}")
    
    def show_post_acquisition_plot(self):
        """Show post-acquisition plot with synchronized axes."""
        if self.acquisition_buffer is None or self.acquisition_samples == 0:
            return
        
        # Create new window for post-acquisition plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Post-Acquisition Data")
        plot_window.geometry("1000x600")
        
        # Create matplotlib figure with subplots
        fig = Figure(figsize=(10, 6), dpi=100)
        
        # Create subplots with shared x-axis
        axes = []
        for i in range(self.num_channels):
            if i == 0:
                ax = fig.add_subplot(self.num_channels, 1, i+1)
            else:
                ax = fig.add_subplot(self.num_channels, 1, i+1, sharex=axes[0])
            axes.append(ax)
        
        # Plot data
        time_vector = np.arange(self.acquisition_samples) / self.acquisition_rate
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
        
        for i in range(self.num_channels):
            ax = axes[i]
            data = self.acquisition_buffer[:self.acquisition_samples, i]
            ax.plot(time_vector, data, color=colors[i], linewidth=1)
            ax.set_ylabel(f"{self.channel_names[i]} (V)")
            ax.grid(True, alpha=0.3)
            
            if i == self.num_channels - 1:
                ax.set_xlabel('Time (s)')
        
        # Set title
        fig.suptitle(f"Acquisition Data - {self.measurement_direction.upper()} Direction")
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_closing(self):
        """Handle application closing."""
        # Stop data acquisition
        self.run_event.clear()
        
        # Close device
        if self.device.connected:
            self.device.close()
        
        # Close application
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main entry point."""
    app = DataAcquisitionApp()
    app.run()


if __name__ == "__main__":
    main()
