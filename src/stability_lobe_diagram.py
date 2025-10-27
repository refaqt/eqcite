"""
Stability Lobe Diagram Generator for Machine Tool Chatter Prediction
Based on: "Creating a Stability Lobe Diagram" by Jianping Yue (2006)

This script calculates and plots stability lobe diagrams for machining processes
to predict and control machine tool chatter.
"""

import numpy as np
import matplotlib.pyplot as plt


# Material database with Kienzle-Victor model parameters
# Kc1.1: Specific cutting force at h = 1 mm and b = 1 mm [N/mm²]
# mc: Material constant (exponent) for chip thickness effect [-]
MATERIAL_DATABASE = {
    "Aluminum 6061-T6": {
        "Kc1.1": 640,
        "mc": 0.25,
        "description": "Common aluminum alloy"
    },
    "Aluminum 7075-T6": {
        "Kc1.1": 750,
        "mc": 0.22,
        "description": "High-strength aluminum"
    },
    "Aluminum 2024-T3": {
        "Kc1.1": 680,
        "mc": 0.24,
        "description": "Aircraft aluminum"
    },
    "Mild Steel (AISI 1018)": {
        "Kc1.1": 1780,
        "mc": 0.21,
        "description": "Low-carbon steel"
    },
    "Steel (AISI 1045)": {
        "Kc1.1": 2080,
        "mc": 0.17,
        "description": "Medium-carbon steel"
    },
    "Stainless Steel (AISI 304)": {
        "Kc1.1": 2200,
        "mc": 0.14,
        "description": "Austenitic stainless"
    },
    "Stainless Steel (AISI 316)": {
        "Kc1.1": 2300,
        "mc": 0.13,
        "description": "Marine-grade stainless"
    },
    "Tool Steel (H13)": {
        "Kc1.1": 3100,
        "mc": 0.15,
        "description": "Hot-work tool steel"
    },
    "Titanium Alloy (Ti-6Al-4V)": {
        "Kc1.1": 1250,
        "mc": 0.37,
        "description": "Aerospace titanium"
    },
    "Inconel 718": {
        "Kc1.1": 2400,
        "mc": 0.16,
        "description": "Nickel superalloy"
    },
    "Brass (C36000)": {
        "Kc1.1": 520,
        "mc": 0.30,
        "description": "Free-cutting brass"
    },
    "Bronze": {
        "Kc1.1": 850,
        "mc": 0.28,
        "description": "Copper-tin alloy"
    },
    "Cast Iron (Gray)": {
        "Kc1.1": 820,
        "mc": 0.26,
        "description": "Gray cast iron"
    },
    "Cast Iron (Ductile)": {
        "Kc1.1": 1150,
        "mc": 0.22,
        "description": "Ductile cast iron"
    },
    "Copper": {
        "Kc1.1": 900,
        "mc": 0.32,
        "description": "Pure copper"
    },
    "Magnesium Alloy": {
        "Kc1.1": 450,
        "mc": 0.28,
        "description": "Lightweight magnesium"
    },
}


def list_available_materials():
    """
    Print a list of all available materials in the database.
    """
    print("\n" + "="*80)
    print("AVAILABLE MATERIALS (Kienzle-Victor Model)")
    print("="*80)
    print(f"{'Material':<30} {'Kc1.1 (N/mm²)':<15} {'mc':<10} {'Description':<25}")
    print("-"*80)
    for material, data in MATERIAL_DATABASE.items():
        print(f"{material:<30} {data['Kc1.1']:<15} {data['mc']:<10.2f} {data['description']:<25}")
    print("="*80)
    print("\nKienzle-Victor Model: Kc = Kc1.1 × h^(-mc)")
    print("  Kc1.1: Specific cutting force at h=1mm, b=1mm")
    print("  mc: Material constant (higher mc = more size effect)")
    print("  h: Uncut chip thickness (mm)")


def calculate_specific_cutting_force(material_name, chip_thickness):
    """
    Calculate specific cutting force using Kienzle-Victor model.

    Kc = Kc1.1 × h^(-mc)

    Parameters:
    -----------
    material_name : str
        Material name from the MATERIAL_DATABASE
    chip_thickness : float
        Uncut chip thickness in mm

    Returns:
    --------
    Kc : float
        Specific cutting force in N/mm²
    """
    if material_name not in MATERIAL_DATABASE:
        raise ValueError(f"Material '{material_name}' not found. Use list_available_materials() to see options.")

    mat_data = MATERIAL_DATABASE[material_name]
    Kc1_1 = mat_data["Kc1.1"]
    mc = mat_data["mc"]

    # Kienzle-Victor model
    # Ensure chip thickness is not zero to avoid division by zero
    h = max(chip_thickness, 1e-6)
    Kc = Kc1_1 * (h ** (-mc))

    return Kc


def calculate_average_chip_thickness_milling(feed_per_tooth, radial_depth, tool_diameter):
    """
    Calculate average chip thickness for milling operations.

    For end milling, chip thickness varies with angular position:
    h(φ) = fz × sin(φ)

    Parameters:
    -----------
    feed_per_tooth : float
        Feed per tooth in mm/tooth
    radial_depth : float
        Radial depth of cut in mm
    tool_diameter : float
        Tool diameter in mm

    Returns:
    --------
    h_avg : float
        Average chip thickness in mm
    h_max : float
        Maximum chip thickness in mm
    """
    # Radial immersion ratio
    ae_D = radial_depth / tool_diameter

    # For partial immersion, calculate the average over the engagement arc
    # Average of sin(φ) over engagement angle
    if ae_D >= 1.0:
        # Full immersion
        h_avg = feed_per_tooth * (2 / np.pi)  # Average of sin over [0, π]
        h_max = feed_per_tooth
    else:
        # Partial immersion
        # Engagement angle
        phi_end = np.arccos(1 - 2 * ae_D) if ae_D <= 0.5 else np.pi - np.arccos(2 * ae_D - 1)

        # Average chip thickness over engagement
        h_avg = feed_per_tooth * (1 - np.cos(phi_end)) / phi_end
        h_max = feed_per_tooth * np.sin(phi_end)

    return h_avg, h_max


def calculate_cutting_force_coefficients_advanced(material_name, tool_diameter, helix_angle_deg,
                                                   num_teeth, feed_per_tooth, axial_depth,
                                                   radial_depth, rake_angle_deg=10):
    """
    Calculate cutting force coefficients using advanced Kienzle-Victor model.

    This function implements a more sophisticated approach considering:
    - Variable chip thickness along cutting edge
    - Helix angle effects on force distribution
    - Actual engagement geometry

    Parameters:
    -----------
    material_name : str
        Material name from the MATERIAL_DATABASE
    tool_diameter : float
        Tool diameter in mm
    helix_angle_deg : float
        Helix angle in degrees (typically 30-45° for end mills)
    num_teeth : int
        Number of teeth on the cutter
    feed_per_tooth : float
        Feed per tooth in mm/tooth
    axial_depth : float
        Axial depth of cut in mm
    radial_depth : float
        Radial depth of cut in mm
    rake_angle_deg : float
        Rake angle in degrees (default 10°)

    Returns:
    --------
    Kt : float
        Tangential cutting force coefficient in N/mm²
    Kr : float
        Radial cutting force coefficient in N/mm²
    Ka : float
        Axial cutting force coefficient in N/mm²
    ks : float
        Cutting stiffness in N/m² (for stability analysis)
    Kc_avg : float
        Average specific cutting force in N/mm²
    h_avg : float
        Average chip thickness in mm
    """

    # Calculate average chip thickness
    h_avg, h_max = calculate_average_chip_thickness_milling(feed_per_tooth, radial_depth, tool_diameter)

    # Calculate specific cutting force using Kienzle-Victor model
    Kc_avg = calculate_specific_cutting_force(material_name, h_avg)
    Kc_max = calculate_specific_cutting_force(material_name, h_max)

    # Convert angles to radians
    rake_rad = np.deg2rad(rake_angle_deg)
    helix_rad = np.deg2rad(helix_angle_deg)

    # Cutting force coefficients based on orthogonal cutting theory
    # These relate the cutting forces to the chip cross-section

    # Tangential coefficient (main cutting force direction)
    # Kt is approximately equal to Kc for small rake angles
    Kt = Kc_avg * np.cos(rake_rad)

    # Radial coefficient (thrust force)
    # Typically 0.3 to 0.5 times the tangential force
    # Depends on rake angle and friction
    friction_angle = np.deg2rad(25)  # Typical friction angle
    Kr = Kc_avg * np.sin(friction_angle + rake_rad) / np.cos(friction_angle - rake_rad + rake_rad)
    Kr = min(Kr, 0.5 * Kt)  # Limit to reasonable value

    # Axial coefficient (for helical tools)
    # Axial force depends on helix angle
    Ka = Kt * np.tan(helix_rad)

    # Calculate cutting stiffness for stability analysis
    # This is the derivative of cutting force with respect to chip thickness
    # Using Kienzle-Victor: F = Kc × h × b, so dF/dh = Kc × b × (1 - mc) / h
    # For stability, we need the effective stiffness considering geometry

    mat_data = MATERIAL_DATABASE[material_name]
    mc = mat_data["mc"]

    # Effective cutting stiffness (N/m²)
    # This accounts for the change in force with chip thickness
    # ks = dFt/dh = Kt × (1 - mc) / h_avg, converted to N/m²
    ks_per_mm = Kt * (1 - mc) / h_avg ** (-mc)  # N/mm per mm² = N/mm³
    ks = ks_per_mm * 1e6  # Convert N/mm³ to N/m³... wait, units need checking

    # For helical end mills, account for effective engagement
    # Number of teeth engaged simultaneously
    helix_overlap = axial_depth * np.tan(helix_rad) / (np.pi * tool_diameter)
    effective_teeth = min(num_teeth, max(1.0, helix_overlap * num_teeth))

    # Adjust cutting stiffness for multiple teeth engagement
    ks_effective = ks * effective_teeth

    return Kt, Kr, Ka, ks_effective, Kc_avg, h_avg


def calculate_radial_depth_engagement(tool_diameter, radial_depth, milling_type='down'):
    """
    Calculate the start and exit angles for milling engagement.

    Parameters:
    -----------
    tool_diameter : float
        Tool diameter in mm
    radial_depth : float
        Radial depth of cut in mm
    milling_type : str
        'up' for up-milling (conventional), 'down' for down-milling (climb)

    Returns:
    --------
    start_angle : float
        Start engagement angle in radians
    exit_angle : float
        Exit engagement angle in radians
    engagement_angle_deg : float
        Total engagement angle in degrees
    """
    # Radial immersion ratio
    radial_immersion = radial_depth / tool_diameter

    if radial_immersion > 1.0:
        radial_immersion = 1.0  # Full immersion

    # Calculate engagement angles
    if milling_type.lower() == 'up':
        # Up-milling (conventional milling)
        start_angle = 0
        if radial_immersion <= 0.5:
            exit_angle = np.arccos(1 - 2 * radial_immersion)
        else:
            exit_angle = np.pi - np.arccos(2 * radial_immersion - 1)
    elif milling_type.lower() == 'down':
        # Down-milling (climb milling)
        if radial_immersion <= 0.5:
            start_angle = np.arccos(1 - 2 * radial_immersion)
            exit_angle = np.pi
        else:
            start_angle = 0
            exit_angle = np.pi
    else:
        raise ValueError("milling_type must be 'up' or 'down'")

    engagement_angle_deg = np.rad2deg(exit_angle - start_angle)

    return start_angle, exit_angle, engagement_angle_deg


class StabilityLobeDiagram:
    """
    A class to create stability lobe diagrams for machining processes.

    The stability lobe diagram shows the relationship between spindle speed
    and maximum stable depth of cut (chip width).
    """

    def __init__(self, k, ks, zeta, fn, nt):
        """
        Initialize the stability lobe diagram calculator.

        Parameters:
        -----------
        k : float
            Stiffness of the machine tool system (N/m)
        ks : float
            Cutting stiffness or specific power (N/m²)
        zeta : float
            Damping ratio (dimensionless, typically 0.01 to 0.1)
        fn : float
            Natural frequency of the system (Hz)
        nt : int
            Number of teeth on the cutter
        """
        self.k = k
        self.ks = ks
        self.zeta = zeta
        self.fn = fn
        self.nt = nt

        # Calculate minimum chip width (Equation 5)
        self.b_min = (2 * self.k * self.zeta * (1 + self.zeta)) / self.ks

        print(f"Initialized Stability Lobe Diagram")
        print(f"  Stiffness (k): {k} N/m")
        print(f"  Cutting stiffness (ks): {ks} N/m²")
        print(f"  Damping ratio (ζ): {zeta}")
        print(f"  Natural frequency (fn): {fn} Hz")
        print(f"  Number of teeth (nt): {nt}")
        print(f"  Minimum chip width (b_min): {self.b_min*1000:.6f} mm")

    def calculate_rb(self, r):
        """
        Calculate dimensionless chip width ratio rb (Equation 6).

        Parameters:
        -----------
        r : float or numpy array
            Ratio of chatter frequency to natural frequency (f/fn)

        Returns:
        --------
        rb : float or numpy array
            Dimensionless chip width ratio (b/b_min)
        """
        numerator = (1 - r**2)**2 + (2 * self.zeta * r)**2
        denominator = -4 * self.zeta * (1 + self.zeta) * (1 - r**2)
        rb = numerator / denominator
        return rb

    def calculate_rt(self, r, n):
        """
        Calculate tooth frequency ratio rt (Equation 9).

        Parameters:
        -----------
        r : float or numpy array
            Ratio of chatter frequency to natural frequency (f/fn)
        n : int
            Lobe number (0, 1, 2, ...)

        Returns:
        --------
        rt : float or numpy array
            Tooth frequency ratio (ft/fn)
        """
        # Calculate the inverse tangent term
        # Using arctan (not arctan2) to match the paper's formulation
        numerator = -2 * self.zeta * r
        denominator = 1 - r**2
        arctan_term = np.arctan(numerator / denominator)

        # Equation 9
        rt = r / (n + 0.5 + (1 / np.pi) * arctan_term)
        return rt

    def generate_lobe_data(self, n, num_points=1000, N_min=None, N_max=None):
        """
        Generate data points for a single stability lobe.

        Parameters:
        -----------
        n : int
            Lobe number (0, 1, 2, ...)
        num_points : int
            Number of points to calculate for the lobe
        N_min : float or None
            Minimum spindle speed to simulate (RPM)
        N_max : float or None
            Maximum spindle speed to simulate (RPM)

        Returns:
        --------
        rt_values : numpy array
            Tooth frequency ratio values
        rb_values : numpy array
            Dimensionless chip width ratio values
        N_values : numpy array
            Spindle speed values (RPM)
        b_values : numpy array
            Chip width values (m)
        """
        # Create r values starting slightly above 1 to avoid singularity
        # The paper suggests r values that give positive chip widths
        r_start = 1.001
        r_end = 10.0  # Extended range for better coverage
        r_values = np.linspace(r_start, r_end, num_points)

        # Calculate rb for each r value
        rb_values = self.calculate_rb(r_values)

        # Calculate rt for each r value and lobe number n
        rt_values = self.calculate_rt(r_values, n)

        # Calculate actual chip width b
        b_values = rb_values * self.b_min

        # Calculate spindle speed N (RPM)
        # ft = rt * fn, and ft = nt * N / 60
        # Therefore: N = 60 * rt * fn / nt
        N_values = 60 * rt_values * self.fn / self.nt

        # Filter out invalid values (negative or infinite)
        valid_mask = (rb_values > 0) & (rb_values < 1000) & (rt_values > 0) & (N_values > 0)

        # Apply spindle speed range filter if specified
        if N_min is not None:
            valid_mask = valid_mask & (N_values >= N_min)
        if N_max is not None:
            valid_mask = valid_mask & (N_values <= N_max)

        return rt_values[valid_mask], rb_values[valid_mask], N_values[valid_mask], b_values[valid_mask]

    def calculate_optimum_chip_widths(self, max_lobe=5):
        """
        Calculate approximate optimum chip widths at lobe intersections (Section 5.2).

        These are the "sweet spots" that provide maximum stable depth of cut
        at various spindle speeds.

        Parameters:
        -----------
        max_lobe : int
            Maximum lobe number to calculate

        Returns:
        --------
        results : list of dict
            List of dictionaries containing optimum parameters for each lobe
        """
        results = []

        for n in range(max_lobe):
            # Equation 13: rt = 1 / (n + 1)
            rt = 1.0 / (n + 1)

            # Solve Equation 15 for r using iterative method
            # Initial guess for r
            r_guess = 1.2

            # Newton-Raphson iteration to solve Equation 15
            for iteration in range(100):
                # Equation 15 - for the (n+1)th lobe
                numerator = -2 * self.zeta * r_guess
                denominator = 1 - r_guess**2
                arctan_term = np.arctan(numerator / denominator)

                f = r_guess / (n + 1 + 0.5 + (1 / np.pi) * arctan_term) - rt

                # Numerical derivative
                epsilon = 1e-6
                r_plus = r_guess + epsilon
                num_plus = -2 * self.zeta * r_plus
                den_plus = 1 - r_plus**2
                arctan_plus = np.arctan(num_plus / den_plus)
                f_plus = r_plus / (n + 1 + 0.5 + (1 / np.pi) * arctan_plus) - rt

                df = (f_plus - f) / epsilon

                # Update guess
                if abs(df) > 1e-10:
                    r_new = r_guess - f / df
                    if abs(r_new - r_guess) < 1e-8:
                        break
                    r_guess = r_new

            r = r_guess

            # Calculate rb using Equation 6
            rb = self.calculate_rb(r)

            # Calculate actual values
            b = rb * self.b_min
            N = 60 * rt * self.fn / self.nt
            f = r * self.fn

            results.append({
                'lobe_number': n,
                'rt': rt,
                'r': r,
                'rb': rb,
                'N': N,
                'b': b,
                'f': f
            })

        return results

    def plot_stability_diagram(self, num_lobes=4, plot_type='dimensional',
                                show_optimum=True, N_min=None, N_max=None, save_file=None):
        """
        Plot the stability lobe diagram.

        Parameters:
        -----------
        num_lobes : int
            Number of lobes to plot (will plot lobes 0 through num_lobes-1)
        plot_type : str
            'dimensional' for N vs b plot, or 'dimensionless' for rt vs rb plot
        show_optimum : bool
            Whether to show optimum chip width points
        N_min : float or None
            Minimum spindle speed to display (RPM)
        N_max : float or None
            Maximum spindle speed to display (RPM)
        save_file : str or None
            If provided, save the plot to this filename
        """
        plt.figure(figsize=(12, 8))

        # Collect data from all lobes
        all_lobe_data = []
        for n in range(num_lobes):
            rt_values, rb_values, N_values, b_values = self.generate_lobe_data(n, N_min=N_min, N_max=N_max)
            if len(N_values) > 0:  # Only add if there's data
                all_lobe_data.append({
                    'n': n,
                    'rt': rt_values,
                    'rb': rb_values,
                    'N': N_values,
                    'b': b_values
                })

        if not all_lobe_data:
            print("No data to plot in the specified range!")
            return

        # Create a combined stability boundary by taking the minimum at each point
        if plot_type == 'dimensional':
            # Create a fine grid of N values covering all lobes
            all_N = np.concatenate([data['N'] for data in all_lobe_data])
            N_grid = np.linspace(np.min(all_N), np.max(all_N), 2000)

            # For each N value, find the minimum b across all lobes
            b_min_envelope = np.full_like(N_grid, np.inf)

            for data in all_lobe_data:
                # Interpolate this lobe's data onto the grid
                b_interp = np.interp(N_grid, data['N'], data['b'], left=np.inf, right=np.inf)
                # Keep the minimum value
                b_min_envelope = np.minimum(b_min_envelope, b_interp)

            # Remove infinite values
            valid_mask = np.isfinite(b_min_envelope)
            N_envelope = N_grid[valid_mask]
            b_envelope = b_min_envelope[valid_mask]

            # Plot the stability envelope
            plt.plot(N_envelope, b_envelope * 1000, 'b-', linewidth=2.5, label='Stability Boundary')

            # Shade the stable region
            plt.fill_between(N_envelope, 0, b_envelope * 1000, alpha=0.3, color='green', label='Stable Region')

        else:  # dimensionless
            # Create a fine grid of rt values covering all lobes
            all_rt = np.concatenate([data['rt'] for data in all_lobe_data])
            rt_grid = np.linspace(np.min(all_rt), np.max(all_rt), 2000)

            # For each rt value, find the minimum rb across all lobes
            rb_min_envelope = np.full_like(rt_grid, np.inf)

            for data in all_lobe_data:
                # Interpolate this lobe's data onto the grid
                rb_interp = np.interp(rt_grid, data['rt'], data['rb'], left=np.inf, right=np.inf)
                # Keep the minimum value
                rb_min_envelope = np.minimum(rb_min_envelope, rb_interp)

            # Remove infinite values
            valid_mask = np.isfinite(rb_min_envelope)
            rt_envelope = rt_grid[valid_mask]
            rb_envelope = rb_min_envelope[valid_mask]

            # Plot the stability envelope
            plt.plot(rt_envelope, rb_envelope, 'b-', linewidth=2.5, label='Stability Boundary')

            # Shade the stable region
            plt.fill_between(rt_envelope, 0, rb_envelope, alpha=0.3, color='green', label='Stable Region')

        # Add lower border (minimum chip width)
        if plot_type == 'dimensional':
            x_range = plt.xlim()
            plt.axhline(y=self.b_min * 1000, color='darkgreen', linestyle='--',   # Convert to mm
                       label='Absolute Minimum (b_min)', linewidth=2)
            plt.xlim(x_range)
        else:
            x_range = plt.xlim()
            plt.axhline(y=1.0, color='darkgreen', linestyle='--',
                       label='Absolute Minimum (rb = 1)', linewidth=2)
            plt.xlim(x_range)

        # Plot optimum points if requested
        if show_optimum:
            optimum_results = self.calculate_optimum_chip_widths(num_lobes)
            if plot_type == 'dimensional':
                N_opt = [r['N'] for r in optimum_results]
                b_opt = [r['b'] * 1000 for r in optimum_results]  # Convert to mm
                # Filter optimum points by spindle speed range
                if N_min is not None or N_max is not None:
                    filtered_N = []
                    filtered_b = []
                    for N_val, b_val in zip(N_opt, b_opt):
                        if (N_min is None or N_val >= N_min) and (N_max is None or N_val <= N_max):
                            filtered_N.append(N_val)
                            filtered_b.append(b_val)
                    N_opt = filtered_N
                    b_opt = filtered_b
                if N_opt:  # Only plot if there are points in range
                    plt.scatter(N_opt, b_opt, color='red', s=100, marker='s',
                               label='Optimum points', zorder=5)
            else:
                rt_opt = [r['rt'] for r in optimum_results]
                rb_opt = [r['rb'] for r in optimum_results]
                plt.scatter(rt_opt, rb_opt, color='red', s=100, marker='s',
                           label='Optimum points', zorder=5)

        # Labels and formatting
        if plot_type == 'dimensional':
            plt.xlabel('Spindle Speed N (RPM)', fontsize=12, fontweight='bold')
            plt.ylabel('Chip Width b (mm)', fontsize=12, fontweight='bold')
            plt.title('Stability Lobe Diagram - Dimensional', fontsize=14, fontweight='bold')
            # Set x-axis limits if specified
            if N_min is not None or N_max is not None:
                current_xlim = plt.xlim()
                new_xmin = N_min if N_min is not None else current_xlim[0]
                new_xmax = N_max if N_max is not None else current_xlim[1]
                plt.xlim(new_xmin, new_xmax)
            # Set y-axis to start at 0 and end at maximum value in the plot
            if len(b_envelope) > 0:
                y_max = np.max(b_envelope) * 1000 * 1.1  # Add 10% margin
                plt.ylim(0, y_max)
        else:
            plt.xlabel('Tooth Frequency Ratio rt (ft/fn)', fontsize=12, fontweight='bold')
            plt.ylabel('Chip Width Ratio rb (b/b_min)', fontsize=12, fontweight='bold')
            plt.title('Stability Lobe Diagram - Dimensionless', fontsize=14, fontweight='bold')
            # Set y-axis to start at 0 and end at maximum value in the plot
            if len(rb_envelope) > 0:
                y_max = np.max(rb_envelope) * 1.1  # Add 10% margin
                plt.ylim(0, y_max)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')

        # Add stability region annotation
        plt.text(0.02, 0.98, 'STABLE REGION\n(below boundary)',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

        plt.text(0.02, 0.85, 'UNSTABLE REGION\n(above boundary)',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightcoral', alpha=0.5))

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_file}")

        plt.show()

    def print_optimum_results(self, max_lobe=5):
        """
        Print the optimum chip widths in a formatted table.

        Parameters:
        -----------
        max_lobe : int
            Maximum lobe number to calculate
        """
        results = self.calculate_optimum_chip_widths(max_lobe)

        print("\n" + "="*80)
        print("OPTIMUM CHIP WIDTHS (Sweet Spots)")
        print("="*80)
        print(f"{'Lobe':<6} {'rt':<10} {'r':<10} {'rb':<10} {'N (RPM)':<12} {'b (mm)':<12}")
        print("-"*80)

        for r in results:
            print(f"{r['lobe_number']:<6} {r['rt']:<10.4f} {r['r']:<10.4f} "
                  f"{r['rb']:<10.4f} {r['N']:<12.1f} {r['b']*1000:<12.6f}")  # Convert to mm

        print("="*80)
        print("\nNote: These are approximate intersection points between lobes n and n+1.")
        print("They represent the maximum stable chip widths at various spindle speeds.")


def main():
    """
    Example usage of the Stability Lobe Diagram calculator.

    Demonstrates advanced Kienzle-Victor model for cutting force calculation.
    """
    print("\n" + "="*80)
    print("STABILITY LOBE DIAGRAM GENERATOR")
    print("Based on: 'Creating a Stability Lobe Diagram' by Jianping Yue (2006)")
    print("With Advanced Kienzle-Victor Cutting Force Model")
    print("="*80 + "\n")

    # Show available materials
    list_available_materials()

    # Select material and cutting conditions
    material = "Aluminum 6061-T6"
    tool_diameter = 6.0  # mm
    helix_angle = 45.0  # degrees (typical for end mills)
    rake_angle = 10.0  # degrees (typical positive rake)
    num_teeth = 3
    feed_per_tooth = 0.025  # mm/tooth
    axial_depth = 10.0  # mm (length of cut)
    radial_depth = 1  # mm (stepover)
    milling_type = 'down'  # down-milling (climb milling)

    print("\n" + "="*80)
    print("CUTTING CONDITIONS")
    print("="*80)
    print(f"Material: {material}")
    print(f"Tool diameter: {tool_diameter} mm")
    print(f"Helix angle: {helix_angle}°")
    print(f"Rake angle: {rake_angle}°")
    print(f"Number of teeth: {num_teeth}")
    print(f"Feed per tooth: {feed_per_tooth} mm/tooth")
    print(f"Axial depth of cut: {axial_depth} mm")
    print(f"Radial depth of cut: {radial_depth} mm")
    print(f"Milling type: {milling_type}-milling")

    # Calculate cutting force coefficients using advanced model
    Kt, Kr, Ka, ks, Kc_avg, h_avg = calculate_cutting_force_coefficients_advanced(
        material, tool_diameter, helix_angle, num_teeth,
        feed_per_tooth, axial_depth, radial_depth, rake_angle
    )

    print(f"\n" + "-"*80)
    print("KIENZLE-VICTOR MODEL RESULTS")
    print("-"*80)
    print(f"Average chip thickness (h_avg): {h_avg:.4f} mm")
    print(f"Average specific cutting force (Kc): {Kc_avg:.1f} N/mm²")
    print(f"\nCutting Force Coefficients:")
    print(f"  Tangential coefficient (Kt): {Kt:.1f} N/mm²")
    print(f"  Radial coefficient (Kr): {Kr:.1f} N/mm²")
    print(f"  Axial coefficient (Ka): {Ka:.1f} N/mm²")
    print(f"\nStability Analysis Parameters:")
    print(f"  Cutting stiffness (ks): {ks:.2e} N/m²")

    # Calculate engagement angles
    start_angle, exit_angle, engagement_deg = calculate_radial_depth_engagement(
        tool_diameter, radial_depth, milling_type
    )

    print(f"\nMilling Engagement:")
    print(f"  Start angle: {np.rad2deg(start_angle):.1f}°")
    print(f"  Exit angle: {np.rad2deg(exit_angle):.1f}°")
    print(f"  Engagement angle: {engagement_deg:.1f}°")
    print(f"  Radial immersion: {radial_depth/tool_diameter*100:.1f}%")

    # Machine tool parameters (typical for a milling machine)
    k = 1e6  # Stiffness (N/m) - typical for medium rigidity machine
    zeta = 0.02  # Damping ratio (2% damping is typical)
    fn = 50  # Natural frequency (Hz) - typical for medium rigidity machine

    print(f"\n" + "-"*80)
    print("MACHINE TOOL PARAMETERS")
    print("-"*80)
    print(f"Stiffness (k): {k:.2e} N/m")
    print(f"Damping ratio (ζ): {zeta}")
    print(f"Natural frequency (fn): {fn} Hz")
    print("="*80)

    # Define spindle speed range
    N_min = 5000    # Minimum spindle speed (RPM)
    N_max = 24000   # Maximum spindle speed (RPM)

    # Number of lobes to plot
    num_lobes = 4

    # Create the stability lobe diagram object
    sld = StabilityLobeDiagram(k, ks, zeta, fn, num_teeth)

    # Print optimum chip widths
    print("\n")
    sld.print_optimum_results(max_lobe=num_lobes)

    # Plot dimensional stability diagram (N vs b)
    print(f"\nGenerating dimensional stability lobe diagram (N vs b)...")
    print(f"Spindle speed range: {N_min} - {N_max} RPM")
    print(f"Number of lobes: {num_lobes}")
    sld.plot_stability_diagram(num_lobes=num_lobes, plot_type='dimensional',
                               show_optimum=True, N_min=N_min, N_max=N_max,
                               save_file='stability_diagram_dimensional.png')

    # Plot dimensionless stability diagram (rt vs rb)
    print("\nGenerating dimensionless stability lobe diagram (rt vs rb)...")
    sld.plot_stability_diagram(num_lobes=num_lobes, plot_type='dimensionless',
                               show_optimum=True,
                               save_file='stability_diagram_dimensionless.png')

    print("\nDone! Check the generated PNG files for the stability diagrams.")
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("The Kienzle-Victor model: Kc = Kc1.1 × h^(-mc)")
    print("  - Kc1.1: Material-specific constant at h=1mm")
    print("  - mc: Material constant (size effect exponent)")
    print("  - h: Uncut chip thickness")
    print("\nTo modify parameters, edit the values in the main() function:")
    print("  Material Properties:")
    print("    - material: Choose from MATERIAL_DATABASE")
    print("  Tool Geometry:")
    print("    - tool_diameter: End mill diameter (mm)")
    print("    - helix_angle: Helix angle (30-45° typical)")
    print("    - rake_angle: Rake angle (5-15° typical)")
    print("    - num_teeth: Number of flutes (2-6 typical)")
    print("  Cutting Conditions:")
    print("    - feed_per_tooth: Feed per tooth (mm/tooth)")
    print("    - axial_depth: Axial depth of cut (mm)")
    print("    - radial_depth: Radial depth of cut (mm)")
    print("    - milling_type: 'up' or 'down' milling")
    print("  Machine Properties:")
    print("    - k: Machine stiffness (N/m)")
    print("    - zeta: Damping ratio (0.01-0.05 typical)")
    print("    - fn: Natural frequency (Hz)")
    print("  Analysis Range:")
    print("    - N_min, N_max: Spindle speed range (RPM)")
    print("    - num_lobes: Number of stability lobes")
    print("="*80)


if __name__ == "__main__":
    main()