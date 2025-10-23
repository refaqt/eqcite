import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import time


class MultisineOptimizer:
    def __init__(self, num_tones, num_samples=4096):
        """
        Initialize the multisine optimizer

        Parameters:
        num_tones (int): Number of frequency components (K)
        num_samples (int): Number of time domain samples (N)
        """
        self.K = num_tones
        self.N = num_samples
        self.frequencies = np.arange(1, self.K + 1) / self.N

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
            freq_index = k + 1  # Skip DC component
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

    def optimize_multisine(self, method='combined', amplitudes=None):
        """
        Main optimization routine

        Parameters:
        method: 'schroeder', 'yang', 'guillaume', or 'combined'
        amplitudes: Amplitude spectrum (default: flat spectrum)
        """
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        # Start with Schroeder phases
        initial_phases = self.schroeder_phases()

        if method == 'schroeder':
            phases = initial_phases
            cf = self.calculate_crest_factor(
                self.generate_multisine(phases, amplitudes))

        elif method == 'yang':
            phases, cf = self.yang_swapping_method(initial_phases, amplitudes)

        elif method == 'guillaume':
            phases, cf = self.guillaume_optimization(initial_phases, amplitudes)

        elif method == 'combined':
            # First apply Yang's method
            phases, cf = self.yang_swapping_method(initial_phases, amplitudes)
            # Then refine with Guillaume's method
            phases, cf = self.guillaume_optimization(phases, amplitudes)

        else:
            raise ValueError(f"Unknown method: {method}")

        return phases, cf

    def export_multisine(self, filename, phases, amplitudes=None):
        """Export multisine data to text file"""
        if amplitudes is None:
            amplitudes = np.ones(self.K)

        # Generate time domain signal
        signal = self.generate_multisine(phases, amplitudes)

        # Create time vector
        time_vector = np.arange(self.N) / self.N

        # Save to file
        header = (f"# Multisine Signal Data\n"
                  f"# Number of tones: {self.K}\n"
                  f"# Number of samples: {self.N}\n"
                  f"# Crest Factor: {self.calculate_crest_factor(signal):.6f}\n"
                  f"# Columns: Time, Signal\n")

        data = np.column_stack((time_vector, signal))
        np.savetxt(filename, data, header=header, fmt='%.10e')

        # Also save frequency domain data
        freq_filename = filename.replace('.txt', '_frequency_domain.txt')
        freq_header = (f"# Multisine Frequency Domain Data\n"
                       f"# Number of tones: {self.K}\n"
                       f"# Columns: Frequency_Index, Frequency, Amplitude, Phase_deg\n")

        freq_data = []
        for k in range(self.K):
            freq_data.append([k + 1, self.frequencies[k], amplitudes[k],
                              np.degrees(phases[k])])

        np.savetxt(freq_filename, freq_data, header=freq_header, fmt='%.10e')

        return signal


# Example usage and comparison
def main():
    # Test with different numbers of tones
    tone_counts = [15, 31]
    methods = ['schroeder', 'yang', 'guillaume', 'combined']

    results = {}

    for K in tone_counts:
        print(f"\nOptimizing multisine with {K} tones:")
        print("-" * 50)

        optimizer = MultisineOptimizer(num_tones=K, num_samples=4096)

        for method in methods:
            start_time = time.time()
            phases, cf = optimizer.optimize_multisine(method=method)
            elapsed_time = time.time() - start_time

            results[f"{K}_{method}"] = {
                'cf': cf,
                'time': elapsed_time,
                'phases': phases
            }

            print(f"{method.capitalize():12} - CF: {cf:.6f}, Time: {elapsed_time:.3f}s")

            # Export the best result for each tone count
            if method == 'combined':
                filename = f"multisine_{K}tones_optimized.txt"
                signal = optimizer.export_multisine(filename, phases)
                print(f"Exported to {filename}")

    # Create visualization for one example
    K = 31
    optimizer = MultisineOptimizer(num_tones=K, num_samples=4096)

    # Compare methods visually
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, method in enumerate(methods):
        phases = results[f"{K}_{method}"]['phases']
        signal = optimizer.generate_multisine(phases)
        cf = results[f"{K}_{method}"]['cf']

        axes[idx].plot(signal[:200])  # Plot first 200 samples
        axes[idx].set_title(f'{method.capitalize()} (CF={cf:.4f})')
        axes[idx].set_xlabel('Sample')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multisine_comparison.png', dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    results = main()