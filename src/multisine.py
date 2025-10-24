import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import time


class MultisineOptimizer:
    def __init__(self, fmin_desired, fmax_desired, fs, df_des):
        """
        Initialize the multisine optimizer

        Parameters:
        fmin_desired (float): Minimum desired frequency (Hz)
        fmax_desired (float): Maximum desired frequency (Hz)
        fs (float): Sampling rate (Hz)
        df_des (float): Desired frequency resolution (Hz)
        """
        self.fmin_desired = fmin_desired
        self.fmax_desired = fmax_desired
        self.fs = fs
        self.df_des = df_des
        
        # Calculate signal length
        self.N = round(fs / fmin_desired)
        
        # Calculate actual minimum frequency
        self.fmin = fs / self.N
        
        # Calculate frequency resolution
        # Find the largest df <= df_des such that (fmax_desired - fmin) / df is integer
        frequency_range = fmax_desired - self.fmin
        max_steps = int(frequency_range / df_des)
        self.df = frequency_range / max_steps
        
        # Calculate actual maximum frequency
        self.fmax = self.fmin + max_steps * self.df
        
        # Generate frequency array
        self.frequencies = np.arange(self.fmin, self.fmax + self.df/2, self.df)
        self.K = len(self.frequencies)

    def calculate_crest_factor(self, signal):
        """Calculate the crest factor of a signal"""
        rms = np.sqrt(np.mean(np.abs(signal) ** 2))
        peak = np.max(np.abs(signal))
        return peak / rms

    def schroeder_phases(self):
        """
        Calculate initial phases using Schroeder's method (Eq. 8 in paper)
        φ_k = π*(k-1)²/K
        """
        k = np.arange(1, self.K + 1)
        phases = np.pi * (k - 1) ** 2 / self.K
        return phases

    def generate_multisine(self, phases, amplitudes=None):
        """
        Generate multisine signal from phases and amplitudes

        Parameters:
        phases: Phase values for each frequency component
        amplitudes: Amplitude values (default: flat spectrum with unit amplitude)
        """
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        # Create frequency domain representation
        C = np.zeros(self.N, dtype=complex)
        for k in range(self.K):
            # Calculate frequency index for this frequency component
            freq_index = round(self.frequencies[k] * self.N / self.fs)
            if freq_index > 0 and freq_index < self.N:  # Skip DC and Nyquist
                C[freq_index] = amplitudes[k] * np.exp(1j * phases[k])

        # Generate time domain signal using IFFT
        x = self.N * ifft(C)
        return np.real(x)

    def yang_swapping_method(self, phases, amplitudes=None, max_iterations=100,
                             tolerance=1e-6, b1=10, b2=10000):
        """
        Yang's time-frequency domain swapping algorithm (Algorithm 1 in paper)
        with logarithmic threshold adaptation
        """
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        best_cf = float('inf')
        best_phases = phases.copy()

        current_phases = phases.copy()

        for iteration in range(max_iterations):
            # Generate time domain signal
            x = self.generate_multisine(current_phases, amplitudes)

            # Calculate current CF
            current_cf = self.calculate_crest_factor(x)

            if current_cf < best_cf:
                best_cf = current_cf
                best_phases = current_phases.copy()

            # Calculate adaptive threshold (Eq. 12 in paper)
            a_L = np.log10(iteration + b1) / np.log10(b2)
            a_L = np.clip(a_L, 0.75, 0.95)  # Keep within recommended range

            # Apply amplitude clipping
            threshold = a_L * np.max(np.abs(x))
            x_clipped = np.where(np.abs(x) > threshold,
                                 threshold * np.sign(x),
                                 x)

            # Transform back to frequency domain
            X_clipped = fft(x_clipped) / self.N

            # Extract new phases while keeping amplitudes
            new_phases = np.angle(X_clipped[1:self.K + 1])

            # Check convergence
            phase_change = np.mean(np.abs(new_phases - current_phases))
            if phase_change < tolerance:
                break

            current_phases = new_phases

        return best_phases, best_cf

    def guillaume_optimization(self, initial_phases, amplitudes=None,
                               p_values=[4, 8, 16, 32, 64, 128]):
        """
        Guillaume's optimization method using p-norm minimization
        """
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        current_phases = initial_phases.copy()

        for p in p_values:
            # Define objective function (p-norm of time domain signal)
            def objective(phases):
                x = self.generate_multisine(phases, amplitudes)
                return np.sum(np.abs(x) ** p) ** (1 / p)

            # Optimize phases
            result = minimize(objective, current_phases,
                              method='L-BFGS-B',
                              options={'maxiter': 100})

            current_phases = result.x

        return current_phases, self.calculate_crest_factor(
            self.generate_multisine(current_phases, amplitudes))

    def optimize_multisine(self, amplitudes=None):
        """
        Main optimization routine using Yang's method

        Parameters:
        amplitudes: Amplitude spectrum (default: flat spectrum)
        """
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        # Start with Schroeder phases
        initial_phases = self.schroeder_phases()

        # Apply Yang's optimization method
        phases, cf = self.yang_swapping_method(initial_phases, amplitudes)

        return phases, cf

    def export_multisine(self, filename, phases, amplitudes=None):
        """Export multisine data to CSV file"""
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        # Generate time domain signal
        signal = self.generate_multisine(phases, amplitudes)

        # Create time vector in seconds
        time_vector = np.arange(self.N) / self.fs

        # Create metadata header
        cf = self.calculate_crest_factor(signal)
        header_lines = [
            f"# Multisine Signal Data",
            f"# fmin_desired: {self.fmin_desired:.6f} Hz",
            f"# fmax_desired: {self.fmax_desired:.6f} Hz", 
            f"# fmin: {self.fmin:.6f} Hz",
            f"# fmax: {self.fmax:.6f} Hz",
            f"# fs: {self.fs:.6f} Hz",
            f"# df: {self.df:.6f} Hz",
            f"# df_des: {self.df_des:.6f} Hz",
            f"# N: {self.N}",
            f"# K: {self.K}",
            f"# Crest Factor: {cf:.6f}",
            f"# Columns: Time_s, Signal"
        ]
        
        # Write header and data to CSV
        with open(filename, 'w') as f:
            for line in header_lines:
                f.write(line + '\n')
            f.write('Time_s,Signal\n')
            for i in range(self.N):
                f.write(f'{time_vector[i]:.10e},{signal[i]:.10e}\n')

        return signal


# Example usage
def main():
    """
    Example usage of the adapted multisine generator
    """
    # Example parameters
    fmin_desired = 5.0  # Hz
    fmax_desired = 500.0  # Hz
    fs = 5000.0  # Hz
    df_des = 1.0  # Hz
    
    print("Multisine Generator Example")
    print("=" * 50)
    print(f"Input parameters:")
    print(f"  fmin_desired: {fmin_desired} Hz")
    print(f"  fmax_desired: {fmax_desired} Hz")
    print(f"  fs: {fs} Hz")
    print(f"  df_des: {df_des} Hz")
    print()
    
    # Create optimizer
    optimizer = MultisineOptimizer(fmin_desired, fmax_desired, fs, df_des)
    
    # Display calculated parameters
    print("Calculated parameters:")
    print(f"  fmin: {optimizer.fmin:.6f} Hz")
    print(f"  fmax: {optimizer.fmax:.6f} Hz")
    print(f"  df: {optimizer.df:.6f} Hz")
    print(f"  N (signal length): {optimizer.N}")
    print(f"  K (number of tones): {optimizer.K}")
    print(f"  Frequencies: {optimizer.frequencies[0]:.3f} to {optimizer.frequencies[-1]:.3f} Hz")
    print()
    
    # Optimize multisine
    print("Optimizing multisine...")
    start_time = time.time()
    phases, cf = optimizer.optimize_multisine()
    elapsed_time = time.time() - start_time
    
    print(f"Optimization completed in {elapsed_time:.3f} seconds")
    print(f"Final crest factor: {cf:.6f}")
    print()
    
    # Export to CSV
    filename = "multisine_optimized.csv"
    signal = optimizer.export_multisine(filename, phases)
    print(f"Multisine exported to {filename}")
    
    # Create a simple visualization
    plt.figure(figsize=(12, 8))
    
    # Time domain plot
    plt.subplot(2, 1, 1)
    time_vector = np.arange(optimizer.N) / optimizer.fs
    
    # Show full signal, but limit to reasonable number of samples for readability
    max_samples = min(optimizer.N, 2000)  # Show max 2000 samples or full signal
    plt.plot(time_vector[:max_samples], signal[:max_samples])
    plt.title(f'Multisine Signal (CF={cf:.4f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Frequency domain plot
    plt.subplot(2, 1, 2)
    freqs = np.fft.fftfreq(optimizer.N, 1/optimizer.fs)
    fft_signal = np.fft.fft(signal)
    plt.plot(freqs[:optimizer.N//2], np.abs(fft_signal[:optimizer.N//2]))
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, optimizer.fmax * 1.1)
    
    plt.tight_layout()
    plt.savefig('multisine_result.png', dpi=150)
    plt.show()
    
    return {
        'optimizer': optimizer,
        'phases': phases,
        'crest_factor': cf,
        'signal': signal,
        'filename': filename
    }


if __name__ == "__main__":
    results = main()