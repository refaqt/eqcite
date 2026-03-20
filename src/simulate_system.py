#!/usr/bin/env python3
"""
Mass-Spring-Damper System Simulator

Simulates a hammer-excited mass-spring-damper system and generates test data
compatible with frf_estimation.py. Produces CSV output files and visualization plots.

Author: MSD System Simulator
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================

class SimulationParameters:
    """Container for simulation parameters."""
    
    def __init__(self):
        # Timing parameters
        self.fs = 8000           # Sampling rate (Hz)
        self.dt = 1 / self.fs    # Sampling time (s)
        self.T = 1               # Measurement time (s)
        
        # Mass-Spring-Damper parameters
        self.mass = 8.0                                                 # Mass (kg)
        self.stiffness = 1e6                                            # Spring stiffness (N/m)
        self.damping = 0.02 * 2 * np.sqrt(self.stiffness * self.mass)    # Damping coefficient (N·s/m)
        
        # Hammer parameters
        self.hammer_force = 50  # Hammer force magnitude (N)
        self.hammer_bandwidth = 1000  # Hammer bandwidth (Hz)
        
        # Geometric parameter
        self.alpha = 20.123 * np.pi / 180  # Angle (radians)
        
        # Sensor calibration (from frf_estimation.py)
        self.r_u = 1 / (2.5e-3 * 10)  # Force sensitivity (N/V)
        self.r_a = 5 * 9.81 / 10    # Accelerometer sensitivity (m/s²/V)
    
    def get_natural_frequency(self):
        """Calculate natural frequency in Hz."""
        omega_n = np.sqrt(self.stiffness / self.mass)
        return omega_n / (2 * np.pi)
    
    def get_damping_ratio(self):
        """Calculate damping ratio (dimensionless)."""
        omega_n = np.sqrt(self.stiffness / self.mass)
        return self.damping / (2 * self.mass * omega_n)

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_hammer_signal(params):
    """
    Generate hammer excitation signal at t=0.1·T with first-order low-pass filter 
    applied in frequency domain.
    
    Parameters:
    -----------
    params : SimulationParameters
        Simulation parameters
        
    Returns:
    --------
    tuple : (time_array, hammer_force_raw, hammer_force_filtered)
    """
    # Generate time array
    N = int(params.T / params.dt)
    time = np.arange(N) * params.dt
    
    # Create impulse signal at t = 0.1·T
    impulse_index = int(0.1 * N)
    hammer_raw = np.zeros(N)
    hammer_raw[impulse_index] = params.hammer_force
    
    # Apply first-order low-pass filter in frequency domain
    # H_filter(jω) = ωc/(jω + ωc)
    
    # Take FFT of raw impulse
    U_raw = np.fft.fft(hammer_raw)
    
    # Generate frequency array
    frequencies = np.fft.fftfreq(N, params.dt)
    omega = 2 * np.pi * frequencies
    omega_c = 2 * np.pi * params.hammer_bandwidth
    
    # Create filter transfer function
    H_filter = np.zeros(N, dtype=complex)
    H_filter[0] = 1.0  # DC component
    H_filter[1:] = omega_c / (1j * omega[1:] + omega_c)
    
    # Apply filter in frequency domain
    U_filtered = U_raw * H_filter
    
    # Transform back to time domain (take real part to remove numerical noise)
    hammer_filtered = np.real(np.fft.ifft(U_filtered))
    
    return time, hammer_raw, hammer_filtered

# =============================================================================
# SYSTEM SIMULATION
# =============================================================================

def simulate_msd_system(time, force, params):
    """
    Simulate mass-spring-damper system using discrete-time z-transform approach.
    
    Uses bilinear transform (Tustin's method) to convert continuous system to discrete:
    - Continuous: H(s) = 1/(m·s² + c·s + k)
    - Bilinear transform: s = (2/T)·(z-1)/(z+1)
    - Results in: H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²)/(1 + a₁z⁻¹ + a₂z⁻²)
    
    This gives position output for force input.
    
    Parameters:
    -----------
    time : array_like
        Time array
    force : array_like
        Input force signal (filtered)
    params : SimulationParameters
        System parameters
        
    Returns:
    --------
    tuple : (position, velocity, acceleration)
    """
    N = len(time)
    T = params.dt  # Sampling period
    m = params.mass
    c = params.damping
    k = params.stiffness
    
    # Calculate discrete transfer function coefficients using bilinear transform
    # H(s) = 1/(m·s² + c·s + k) with s = (2/T)·(z-1)/(z+1)
    
    # Normalization factor (denominator evaluated at z=1 equivalent)
    norm = 4*m + 2*c*T + k*T**2
    
    # Numerator coefficients: [b₀, b₁, b₂]
    b0 = T**2 / norm
    b1 = 2 * T**2 / norm
    b2 = T**2 / norm
    
    # Denominator coefficients: [1, a₁, a₂] (normalized so a₀ = 1)
    a1 = 2*(k*T**2 - 4*m) / norm
    a2 = (4*m - 2*c*T + k*T**2) / norm
    
    # Take FFT of input force
    U = np.fft.fft(force)
    
    # Generate frequency array and z values
    frequencies = np.fft.fftfreq(N, T)
    omega = 2 * np.pi * frequencies
    
    # Evaluate H(z) at z = e^(jωT) for each frequency
    # z = e^(j·2π·f·T)
    z = np.exp(1j * omega * T)
    z_inv = 1.0 / z  # z⁻¹
    z_inv2 = z_inv ** 2  # z⁻²
    
    # H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²)/(1 + a₁z⁻¹ + a₂z⁻²)
    numerator = b0 + b1 * z_inv + b2 * z_inv2
    denominator = 1.0 + a1 * z_inv + a2 * z_inv2
    
    H_system = numerator / denominator
    
    # Handle DC component (f=0, z=1)
    # At DC, H(z=1) should give static deflection: 1/k
    # But we'll set to 0 to avoid drift in position
    H_system[0] = 0.0
    
    # Multiply in frequency domain: X_position = U · H(z)
    X_position = U * H_system
    
    # Transform back to time domain (position)
    position = np.real(np.fft.ifft(X_position))
    
    # Calculate velocity by differentiating position in frequency domain
    # X_velocity(jω) = jω · X_position(jω)
    X_velocity = 1j * omega * X_position
    velocity = np.real(np.fft.ifft(X_velocity))
    
    # Calculate acceleration by differentiating velocity in frequency domain
    # X_acceleration(jω) = jω · X_velocity(jω)
    X_acceleration = 1j * omega * X_velocity
    acceleration = np.real(np.fft.ifft(X_acceleration))
    
    return position, velocity, acceleration

# =============================================================================
# SENSOR MAPPING
# =============================================================================

def map_to_sensors(acceleration, force, params):
    """
    Map system outputs to sensor channels A0-A7.
    
    Parameters:
    -----------
    acceleration : array_like
        Acceleration signal (m/s²)
    force : array_like
        Force signal (N)
    params : SimulationParameters
        System parameters
        
    Returns:
    --------
    dict : Dictionary with sensor signals A0-A7
    """
    alpha = params.alpha
    r_u = params.r_u
    r_a = params.r_a
    # Convert acceleration to voltage (inverse of sensor calibration)
    # From frf_estimation.py: r_a = 5 * 9.81 / 10 (m/s²/V)
    x_volt = acceleration / r_a
    
    # Map to sensor channels
    A0 = -x_volt * np.cos(alpha)
    A1 = x_volt * np.cos(alpha)
    A2 = -x_volt * np.cos(alpha)
    A3 = x_volt * np.cos(alpha)
    A4 = np.zeros_like(acceleration)
    A5 = np.zeros_like(acceleration)
    A6 = force / r_u  # Convert force to voltage
    A7 = np.zeros_like(acceleration)
    
    return {
        'A0': A0,
        'A1': A1,
        'A2': A2,
        'A3': A3,
        'A4': A4,
        'A5': A5,
        'A6': A6,
        'A7': A7
    }

# =============================================================================
# FILE OUTPUT
# =============================================================================

def save_to_csv(time, sensors, params, position='X0_Y0_Z0', direction='x'):
    """
    Save sensor data to CSV file matching the expected format.
    
    Parameters:
    -----------
    time : array_like
        Time array
    sensors : dict
        Dictionary with sensor signals A0-A7
    params : SimulationParameters
        System parameters
    position : str
        Position string (e.g., 'X0_Y0_Z0')
    direction : str
        Direction ('x' or 'y')
        
    Returns:
    --------
    str : Filename of saved file
    """
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{timestamp}_{position}_{direction}.csv"
    filepath = os.path.join('data', filename)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Calculate sample rate
    sample_rate = int(1 / params.dt)
    
    # Write CSV file with exact format
    with open(filepath, 'w') as f:
        # Line 1: Sample rate
        f.write(f'"Samples per sec. = {8*sample_rate}/1"\n')
        
        # Lines 2-4: Empty strings
        f.write('""\n')
        f.write('""\n')
        
        # Line 5: Headers
        f.write('"Time",' + ','.join(['""'] * 8) + '\n')
        
        # Line 6: Units
        f.write('"sec",' + ','.join(['"Volt"'] * 8) + '\n')
        
        # Lines 7+: Data
        for i in range(len(time)):
            f.write(f'{time[i]:.5f},')
            f.write(f' {sensors["A0"][i]:.5E},')
            f.write(f' {sensors["A1"][i]:.5E},')
            f.write(f' {sensors["A2"][i]:.5E},')
            f.write(f' {sensors["A3"][i]:.5E},')
            f.write(f' {sensors["A4"][i]:.5E},')
            f.write(f' {sensors["A5"][i]:.5E},')
            f.write(f' {sensors["A6"][i]:.5E},')
            f.write(f' {sensors["A7"][i]:.5E}\n')
    
    return filepath

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_time_signals(time, force, acceleration):
    """
    Plot time signals of force (u) and acceleration (x).
    
    Parameters:
    -----------
    time : array_like
        Time array
    force : array_like
        Force signal
    acceleration : array_like
        Acceleration signal
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot force
    ax[0].plot(time, force, 'b-', linewidth=1.5)
    ax[0].set_ylabel('Force u (N)', fontsize=12)
    ax[0].set_title('Hammer Force Excitation', fontsize=14, fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    
    # Plot acceleration
    ax[1].plot(time, acceleration, 'r-', linewidth=1.5)
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel('Acceleration x (m/s²)', fontsize=12)
    ax[1].set_title('System Response (Acceleration)', fontsize=14, fontweight='bold')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_bode_magnitude(time, force, acceleration):
    """
    Plot Bode magnitude diagram (X/U transfer function).
    
    Parameters:
    -----------
    time : array_like
        Time array
    force : array_like
        Force signal
    acceleration : array_like
        Acceleration signal
    """
    # Perform FFT
    N = len(time)
    dt = time[1] - time[0]
    
    U = np.fft.fft(force)
    X_accel = np.fft.fft(acceleration)
    frequencies = np.fft.fftfreq(N, dt)
    
    # Double integration: acceleration -> displacement
    omega = 2 * np.pi * frequencies
    z = np.exp(1j * omega * dt)
    
    # Transfer function for double integration
    H = np.zeros_like(X_accel, dtype=complex)
    H[1:] = dt**2 / (1 - z[1:]**(-1))**2
    X = X_accel * H
    
    # Calculate transfer function X/U
    threshold = 1e-12
    U_magnitude = np.abs(U)
    valid_mask = U_magnitude > threshold
    
    G = np.zeros_like(X, dtype=complex)
    G[valid_mask] = X[valid_mask] / U[valid_mask]
    
    # Get positive frequencies only
    positive_freq_mask = frequencies > 0
    freq_pos = frequencies[positive_freq_mask]
    G_pos = G[positive_freq_mask]
    G_mag = np.abs(G_pos)
    
    # Remove invalid points
    valid_plot_mask = np.isfinite(G_mag) & (G_mag > 0)
    freq_plot = freq_pos[valid_plot_mask]
    G_plot = G_mag[valid_plot_mask]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freq_plot, G_plot, 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Compliance |X/U| (m/N)', fontsize=12)
    ax.set_title('Bode Magnitude Plot - Transfer Function X/U', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_all_sensors(time, acceleration, sensors):
    """
    Plot all sensor signals including acceleration.
    
    Parameters:
    -----------
    time : array_like
        Time array
    acceleration : array_like
        Acceleration signal
    sensors : dict
        Dictionary with sensor signals A0-A7
    """
    fig, axes = plt.subplots(9, 1, figsize=(12, 16), sharex=True)
    
    signals = [
        ('x (m/s²)', acceleration, 'black'),
        ('A0 (V)', sensors['A0'], 'blue'),
        ('A1 (V)', sensors['A1'], 'green'),
        ('A2 (V)', sensors['A2'], 'red'),
        ('A3 (V)', sensors['A3'], 'cyan'),
        ('A4 (V)', sensors['A4'], 'magenta'),
        ('A5 (V)', sensors['A5'], 'orange'),
        ('A6 (V)', sensors['A6'], 'purple'),
        ('A7 (V)', sensors['A7'], 'brown')
    ]
    
    for i, (label, signal, color) in enumerate(signals):
        axes[i].plot(time, signal, color=color, linewidth=1.0)
        axes[i].set_ylabel(label, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add title to first subplot
        if i == 0:
            axes[i].set_title('All Sensor Signals', fontsize=14, fontweight='bold')
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_system_info(params):
    """Print system information."""
    print("\n" + "="*60)
    print("MASS-SPRING-DAMPER SYSTEM SIMULATOR")
    print("="*60)
    print("\nSimulation Parameters:")
    print(f"  Sampling time (dt):      {params.dt*1e6:.1f} µs")
    print(f"  Sampling rate:           {1/params.dt:.0f} Hz")
    print(f"  Measurement time:        {params.T:.2f} s")
    print(f"  Number of samples:       {int(params.T/params.dt)}")
    print("\nSystem Parameters:")
    print(f"  Mass (m):                {params.mass:.2f} kg")
    print(f"  Stiffness (k):           {params.stiffness:.0f} N/m")
    print(f"  Damping (c):             {params.damping:.2f} N·s/m")
    print(f"  Natural frequency:       {params.get_natural_frequency():.2f} Hz")
    print(f"  Damping ratio:           {params.get_damping_ratio():.4f}")
    print("\nExcitation Parameters:")
    print(f"  Hammer force:            {params.hammer_force:.1f} N")
    print(f"  Hammer bandwidth:        {params.hammer_bandwidth:.0f} Hz")
    print("="*60 + "\n")

def main():
    """Main execution function."""
    # Initialize parameters
    params = SimulationParameters()
    
    # Print system information
    print_system_info(params)
    
    # Generate hammer signal
    print("Generating hammer excitation signal...")
    time, force_raw, force = generate_hammer_signal(params)
    
    # Simulate MSD system
    print("Simulating mass-spring-damper system (discrete z-transform)...")
    position, velocity, acceleration = simulate_msd_system(time, force, params)
    
    # Map to sensors
    print("Mapping to sensor channels...")
    sensors = map_to_sensors(acceleration, force, params)
    
    # Save to CSV
    print("Saving data to CSV file...")
    filepath = save_to_csv(time, sensors, params)
    print(f"  File saved: {filepath}")
    
    # Generate plots
    print("\nGenerating plots...")
    print("  Plot 1: Time signals (u and x)")
    plot_time_signals(time, force, acceleration)
    
    print("  Plot 2: Bode magnitude plot")
    plot_bode_magnitude(time, force, acceleration)
    
    print("  Plot 3: All sensor signals")
    plot_all_sensors(time, acceleration, sensors)
    
    print("\nSimulation completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

