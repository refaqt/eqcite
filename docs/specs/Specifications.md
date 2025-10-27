# Stability Lobe Diagram Generator - Technical Specifications

## 1. Project Overview

### 1.1 Purpose
Create a Python application that generates stability lobe diagrams (SLD) for machine tool chatter prediction and control in machining operations. The application enables users to predict stable cutting conditions by analyzing the relationship between spindle speed and maximum stable depth of cut.

### 1.2 Target Users
- Manufacturing engineers
- CNC machine operators
- Process planners
- Machining researchers and students

### 1.3 Key Benefits
- Predict and prevent machine tool chatter
- Optimize material removal rates (MRR)
- Improve surface quality
- Extend tool life
- Reduce machining time

---

## 2. System Architecture Requirements

### 2.1 Module Structure

The application shall be organized into the following modules:

**Core Modules:**
- `dynamics` - Dynamic system representations
- `cutting_forces` - Cutting force calculations
- `stability_analytical` - Stability calculations for analytical systems
- `stability_frf` - Stability calculations for measured FRF systems
- `geometry` - Milling geometry calculations

**Data Modules:**
- `materials` - Material property database

**Visualization Modules:**
- `plots` - Plotting functions
- `reports` - Report generation

**I/O Modules:**
- `import_frf` - FRF data import
- `export_data` - Data export functions

**Optional Modules:**
- `gui` - Graphical user interface

### 2.2 Core Data Structures

**DynamicSystem (Abstract):**
- Must support two implementations: Analytical and Measured
- Must provide method to identify system type
- Must provide method to generate stability lobes

**AnalyticalSystem:**
- Properties: k [N/m], zeta [-], fn [Hz], m [kg]
- Must calculate FRF from system parameters
- Must calculate minimum chip width
- Must use dimensionless equations for lobe generation

**MeasuredSystem:**
- Properties: frequencies [Hz], FRF_real [m/N], FRF_imag [m/N]
- Must interpolate FRF data
- Must NOT attempt to extract fn or zeta
- Must use direct FRF equations for lobe generation

**Material:**
- Properties: name, Kc1.1 [N/mm²], mc [-], description
- Must calculate specific cutting force using Kienzle-Victor model

**CuttingTool:**
- Properties: diameter [mm], num_teeth, helix_angle [deg], rake_angle [deg]

**CuttingConditions:**
- Properties: feed_per_tooth [mm/tooth], axial_depth [mm], radial_depth [mm], milling_type

**StabilityLobeDiagram:**
- Must work with any DynamicSystem implementation
- Must store cutting_stiffness ks [N/m²]
- Must generate multiple lobes
- Must calculate stability envelope
- Must calculate optimum points (method depends on system type)
- Must validate operating points

---

## 3. Dynamic System Input Modes

### 3.1 Mode 1: Analytical Mass-Spring-Damper Model

**Input Parameters:**
- `k`: Machine tool stiffness [N/m]
  - Valid range: 1e5 to 1e7 N/m
  - Typical: 2e6 N/m for medium rigidity machines
  
- `zeta`: Damping ratio [-]
  - Valid range: 0.001 to 0.1
  - Typical: 0.01 to 0.05 (1-5% damping)
  
- `fn`: Natural frequency [Hz]
  - Valid range: 50 to 1000 Hz
  - Typical: 100-300 Hz for milling machines
  
- `m`: Mass [kg] (optional)
  - If not provided, calculate from: m = k / (2π×fn)²

**FRF Calculation:**
The system shall calculate G_R and G_I using standard second-order system equations:
- G_R = (1 - r²) / [k((1-r²)² + (2ζr)²)]
- G_I = -2ζr / [k((1-r²)² + (2ζr)²)]
- Where r = f/fn (frequency ratio)

**Validation Requirements:**
- Reject if k ≤ 0
- Reject if zeta ≤ 0 or zeta ≥ 1
- Reject if fn ≤ 0
- Reject if calculated or provided m ≤ 0

### 3.2 Mode 2: Measured Frequency Response Function

**Input Requirements:**

The system shall accept FRF data in the following formats:
- CSV file with columns for frequency, real FRF, and imaginary FRF
- JSON file with arrays for the same data
- Direct numerical arrays

**Required Data:**
- `frequencies`: Array of frequency values [Hz]
- `FRF_real`: Array of real part of FRF [m/N]
- `FRF_imag`: Array of imaginary part of FRF [m/N]

**Data Requirements:**
- Minimum 100 data points
- Frequencies must be monotonically increasing
- No NaN or infinite values allowed
- Must contain at least some negative G_R values (required for stability)
- Frequency range should cover expected operating conditions

**Critical Constraint:**
- The system shall NOT attempt to extract fn or zeta from measured FRF
- The system shall NOT attempt to identify or separate individual modes
- The system shall work directly with G_R(f) and G_I(f) as measured

**Validation Requirements:**
- Reject if fewer than 100 points
- Reject if frequencies not monotonically increasing
- Reject if any NaN or infinite values present
- Warn if no negative G_R values found
- Warn if frequency range seems insufficient

---

## 4. Material Database and Cutting Force Model

### 4.1 Material Database

The system shall include a built-in database with the following materials and their Kienzle-Victor parameters:

| Material | Kc1.1 [N/mm²] | mc [-] |
|----------|---------------|--------|
| Aluminum 6061-T6 | 640 | 0.25 |
| Aluminum 7075-T6 | 750 | 0.22 |
| Aluminum 2024-T3 | 680 | 0.24 |
| Mild Steel (AISI 1018) | 1780 | 0.21 |
| Steel (AISI 1045) | 2080 | 0.17 |
| Stainless Steel (AISI 304) | 2200 | 0.14 |
| Stainless Steel (AISI 316) | 2300 | 0.13 |
| Tool Steel (H13) | 3100 | 0.15 |
| Titanium Alloy (Ti-6Al-4V) | 1250 | 0.37 |
| Inconel 718 | 2400 | 0.16 |
| Brass (C36000) | 520 | 0.30 |
| Bronze | 850 | 0.28 |
| Cast Iron (Gray) | 820 | 0.26 |
| Cast Iron (Ductile) | 1150 | 0.22 |
| Copper | 900 | 0.32 |
| Magnesium Alloy | 450 | 0.28 |

**Custom Material Support:**
The system shall allow users to define custom materials by providing:
- Material name (string)
- Kc1.1 value [N/mm²]
- mc value [-]
- Optional description

### 4.2 Kienzle-Victor Cutting Force Model

**Formula:**
```
Kc = Kc1.1 × h^(-mc)
```

**Where:**
- Kc: Specific cutting force [N/mm²]
- Kc1.1: Material-specific constant at h=1mm, b=1mm
- h: Uncut chip thickness [mm]
- mc: Material constant (size effect exponent)

**Implementation Requirements:**
- Handle edge case: use minimum chip thickness of 1e-6 mm to avoid division by zero
- Validate: 0 < mc < 1
- Validate: Kc1.1 > 0

### 4.3 Average Chip Thickness Calculation

**For End Milling Operations:**

The instantaneous chip thickness varies with angular position:
```
h(φ) = fz × sin(φ)
```

**Full Immersion (ae/D ≥ 1.0):**
```
h_avg = fz × (2/π)
h_max = fz
```

**Partial Immersion (ae/D < 1.0):**

Calculate engagement angle:
```
If ae/D ≤ 0.5:
    φ_end = arccos(1 - 2×ae/D)
Else:
    φ_end = π - arccos(2×ae/D - 1)
```

Calculate average chip thickness:
```
h_avg = fz × (1 - cos(φ_end)) / φ_end
h_max = fz × sin(φ_end)
```

**Parameters:**
- fz: Feed per tooth [mm/tooth]
- ae: Radial depth of cut [mm]
- D: Tool diameter [mm]
- φ: Angular position [radians]

---

## 5. Cutting Force Coefficients

### 5.1 Required Calculations

The system shall calculate the following cutting force coefficients:

**Tangential Coefficient (Kt):**
- Based on specific cutting force and rake angle
- Formula: Kt = Kc_avg × cos(rake_angle)

**Radial Coefficient (Kr):**
- Based on friction angle and rake angle
- Typical friction angle: 25°
- Must be limited to reasonable value (≤ 0.5 × Kt)

**Axial Coefficient (Ka):**
- Depends on helix angle
- Formula: Ka = Kt × tan(helix_angle)

**Cutting Stiffness (ks):**
- Must account for material constant mc
- Must consider helix overlap effects
- Must calculate effective number of teeth engaged
- Formula involves: Kt, mc, h_avg, helix angle, axial depth, tool diameter
- Units: [N/m²]

### 5.2 Input Parameters

**Required Inputs:**
- Material properties (Kc1.1, mc)
- Tool diameter [mm]
- Helix angle [degrees] (typical: 30-45°)
- Rake angle [degrees] (typical: 5-15°)
- Number of teeth
- Feed per tooth [mm/tooth]
- Axial depth of cut [mm]
- Radial depth of cut [mm]

**Validation:**
- Tool diameter > 0
- Number of teeth ≥ 1
- 0 < helix_angle < 90
- Feed per tooth > 0
- Axial depth > 0
- Radial depth > 0
- Radial depth ≤ tool diameter

### 5.3 Milling Engagement Geometry

**The system shall calculate:**

**For Up-Milling (Conventional):**
- Start angle = 0
- Exit angle based on radial immersion ratio

**For Down-Milling (Climb):**
- Start and exit angles based on radial immersion ratio

**Outputs Required:**
- Start angle [radians]
- Exit angle [radians]
- Engagement angle [degrees]
- Radial immersion [%]

---

## 6. Stability Lobe Generation

### 6.1 Method 1: Analytical System Algorithm

**For Analytical Systems Using k, ζ, fn:**

**Core Equations:**

**Minimum Chip Width (Equation 5):**
```
b_min = (2 × k × ζ × (1 + ζ)) / ks
```

**Dimensionless Chip Width Ratio (Equation 6):**
```
rb = [(1 - r²)² + (2ζr)²] / [-4ζ(1 + ζ)(1 - r²)]
```

**Regenerative Chatter Equation (Equation 9):**
```
rt = r / [n + 0.5 + (1/π) × arctan(-2ζr / (1 - r²))]
```

**Where:**
- n: Lobe number (0, 1, 2, ...)
- r: Chatter frequency ratio (f/fn)
- rt: Tooth frequency ratio (ft/fn)

**Algorithm Requirements:**

For each lobe number n from 0 to (num_lobes - 1):

1. Generate frequency ratio array:
   - Start: r = 1.001 (avoid singularity at r=1)
   - End: r = 10.0
   - Number of points: 1000 (configurable)

2. Calculate dimensionless chip width ratio (rb) for each r

3. Calculate tooth frequency ratio (rt) for each r using lobe number n

4. Calculate dimensional values:
   - b = rb × b_min [mm]
   - ft = rt × fn [Hz]
   - N = 60 × ft / nt [RPM]

5. Filter valid values:
   - Remove: rb < 0
   - Remove: rb > 1000 (unrealistic)
   - Remove: rt ≤ 0
   - Remove: N ≤ 0
   - Apply user-specified N_min and N_max if provided

6. Store lobe data with arrays: r, rt, rb, N, b

### 6.2 Method 2: Measured FRF Algorithm

**For Measured FRF Systems Using G_R(f) and G_I(f):**

**Core Equations:**

**Chip Width from FRF (Equation 3):**
```
b(f) = -1 / (2 × ks × G_R(f))
```

**Phase Shift (Equation 8):**
```
ε(f) = π + 2 × arctan(G_I(f) / G_R(f))
```

**Regenerative Chatter Relationship (Equation 7):**
```
f / ft = n + ε / (2π)

Therefore:
ft(f, n) = f / (n + ε(f)/(2π))
N(f, n) = 60 × ft(f, n) / nt
```

**Where:**
- f: Chatter frequency [Hz] (from measured FRF)
- ft: Tooth frequency [Hz]
- N: Spindle speed [RPM]
- n: Lobe number (0, 1, 2, ...)
- nt: Number of teeth
- ε: Phase shift angle [radians]

**Algorithm Requirements:**

For each lobe number n from 0 to (num_lobes - 1):

1. Use measured frequency array from FRF data

2. For each frequency f in the array:
   - Check if G_R(f) < 0 (required for stability)
   - Calculate b(f) using Equation 3
   - Calculate ε(f) using Equation 8
   - Calculate ft(f, n) using Equation 7
   - Calculate N(f, n) from tooth frequency

3. Filter valid values:
   - Remove: where G_R ≥ 0
   - Remove: b < 0
   - Remove: b > 100 mm (unrealistic)
   - Remove: N ≤ 0
   - Remove: ε < π or ε > 2π (physical constraint)
   - Apply user-specified N_min and N_max if provided

4. Sort by spindle speed N (ascending) for smooth curves

5. Store lobe data with arrays: f, N, b, ε, G_R, G_I

**Critical Requirements:**
- Phase shift ε MUST be in range [π, 2π] for regenerative chatter
- Only negative G_R values produce valid stability boundaries
- G_R > 0 indicates unconditional instability
- Multimodal FRF will naturally produce complex lobe structures
- No mode decomposition required or allowed

### 6.3 Stability Boundary Envelope

**Requirements:**

The system shall calculate an overall stability boundary envelope by:

1. Generate all lobes (n = 0 to num_lobes-1) using appropriate method

2. Create a fine grid of N values:
   - Start: minimum N across all lobes
   - End: maximum N across all lobes
   - Number of points: 2000

3. For each N value on the grid:
   - Interpolate chip width b from each lobe
   - Take the minimum b value across all lobes
   - Handle cases where lobe doesn't cover that N value

4. Filter out infinite or invalid values

5. Return arrays: N_envelope, b_envelope

**Purpose:**
This envelope represents the true stability boundary accounting for all lobe intersections.

### 6.4 Optimum Chip Width Calculation

**Requirements for Analytical Systems:**

The system shall calculate approximate intersection points between consecutive lobes:

For each lobe n from 0 to (num_lobes - 2):

1. Calculate target tooth frequency ratio:
   ```
   rt = 1 / (n + 1)
   ```

2. Solve for frequency ratio r using Equation 15:
   ```
   r / [n + 1 + 0.5 + (1/π) × arctan(-2ζr / (1 - r²))] = rt
   ```
   - Use Newton-Raphson iteration
   - Initial guess: r = 1.2
   - Maximum iterations: 100
   - Convergence tolerance: 1e-8

3. Calculate optimum values:
   - rb using Equation 6
   - b = rb × b_min [mm]
   - ft = rt × fn [Hz]
   - N = 60 × ft / nt [RPM]
   - f = r × fn [Hz]

4. Store optimum point with: lobe_number, N, b, f, r, rt, rb

**Requirements for Measured FRF Systems:**

The system shall find actual intersection points numerically:

For each pair of consecutive lobes (n and n+1):

1. Interpolate both lobes onto a common N grid

2. Calculate difference: diff(N) = b_lobe_n(N) - b_lobe_n+1(N)

3. Find sign changes in diff (where product of consecutive values < 0)

4. For each sign change:
   - Refine crossing point using linear interpolation
   - Calculate N_optimum and b_optimum at intersection
   - Find corresponding chatter frequency f

5. Store optimum point with: lobe_number, N, b, f

**Note:** FRF optimum points are actual measured intersections, not analytical approximations.

---

## 7. Visualization Requirements

### 7.1 Plot Types

**7.1.1 Dimensional Plot (N vs b)**

**Axes:**
- X-axis: Spindle Speed N [RPM]
- Y-axis: Chip Width b [mm] (or Axial Depth of Cut)

**Required Elements:**
- Stability boundary curve (solid line, prominent)
- Stable region shading (green, semi-transparent)
- Minimum chip width line (dashed, only for analytical systems)
- Optimum points (markers at intersections)
- Grid (light, non-intrusive)

**Annotations Required:**
- "STABLE REGION (below boundary)" label
- "UNSTABLE REGION (above boundary)" label
- Title: "Stability Lobe Diagram - Dimensional"
- Bold axis labels

**Formatting:**
- Figure size: 12" × 8"
- Y-axis: start at 0, end at max value + 10% margin
- X-axis: use specified N_min and N_max or auto-scale

**7.1.2 Frequency-Based Plot (f vs b) - FRF Systems Only**

**Axes:**
- X-axis: Chatter Frequency f [Hz]
- Y-axis: Chip Width b [mm]

**Purpose:**
Shows how stable depth varies with chatter frequency for measured FRF systems. Useful for understanding multimodal behavior.

**Same styling requirements as dimensional plot**

**7.1.3 Individual Lobe Display (Optional)**

**Requirements:**
- Option to show each lobe as a separate colored line
- Lobe numbers in legend
- Useful for debugging and detailed analysis

### 7.2 Export Formats

**7.2.1 Image Export**

The system shall support:
- PNG format (default DPI: 300)
- SVG format (vector, publication quality)

**Filename conventions:**
- `stability_diagram_dimensional.png`
- `stability_diagram_frequency.png` (FRF systems)
- User-configurable prefixes

**7.2.2 Data Export**

**CSV Files Required:**
- `stability_envelope.csv`: columns [N, b]
- `optimum_points.csv`: columns [lobe_number, N, b, f, ...]
- `lobe_data_n{X}.csv`: individual lobe data
  - Analytical: columns [r, rt, rb, N, b]
  - FRF: columns [f, N, b, epsilon, G_R, G_I]

**JSON Export Required:**

Structure:
- `system_type`: "analytical" or "measured"
- `parameters`: all input parameters
- For FRF: `frf_info` with source file, frequency range, number of points
- `stability_envelope`: {N: array, b: array}
- `optimum_points`: array of objects
- `lobes`: dictionary with lobe number keys

### 7.3 Interactive Features

**7.3.1 Point Query Capability**

The system shall support querying any point on the plot to display:
- Spindle speed N [RPM]
- Chip width b [mm]
- Safety factor (distance from boundary)

**7.3.2 Operating Point Validation**

Given an operating point (N, b), the system shall:
- Determine if point is stable or unstable
- Calculate distance from stability boundary [mm]
- Calculate safety factor (b_boundary / b)
- Return boundary value at that spindle speed

**Outputs Required:**
- Boolean: stable/unstable
- Float: distance_from_boundary [mm] (negative if unstable)
- Float: safety_factor (ratio)
- Float: b_boundary at operating N [mm]
- String: message if outside analyzed range

---

## 8. Report Generation

### 8.1 Report Structure

The system shall generate reports in Markdown and optionally PDF format containing:

**Section 1: System Information**
- System type (analytical or measured)
- For analytical: k, ζ, fn, m values
- For measured: FRF source file, frequency range, number of data points

**Section 2: Input Parameters**
- Material name and properties (Kc1.1, mc)
- Tool geometry (diameter, teeth, helix angle, rake angle)
- Cutting conditions (feed, axial depth, radial depth, milling type)

**Section 3: Calculated Values**
- For analytical: minimum chip width b_min [mm]
- Cutting force coefficients (Kt, Kr, Ka)
- Cutting stiffness ks
- Average chip thickness h_avg
- Engagement geometry (angles, immersion)

**Section 4: Optimum Chip Widths Table**

Table with columns:
- Lobe number
- Spindle speed N [RPM]
- Chip width b [mm]
- Chatter frequency f [Hz]
- Notes (e.g., "Sweetest spot" for lobe 0)

**Section 5: Stability Diagrams**
- Embedded images (PNG format)
- For FRF systems: both N vs b and f vs b plots

**Section 6: Recommendations**
- Optimal spindle speed ranges
- Maximum stable depth of cut
- Suggested safety margins (e.g., 80% of boundary value)
- Cautions for specific speed ranges

---

## 9. Input Configuration

### 9.1 Configuration File Format

**File Format:** YAML

**Required Top-Level Keys:**
- `system_type`: "analytical" or "measured"
- `material`: material specification
- `tool`: tool geometry
- `cutting`: cutting conditions
- `analysis`: analysis settings
- `output`: output settings

**For Analytical Systems:**

```yaml
system_type: analytical

analytical:
  k: float        # N/m
  zeta: float     # dimensionless
  fn: float       # Hz
  m: float or null  # kg, auto-calculate if null

material:
  type: "database" or "custom"
  name: string    # if database
  Kc1_1: float    # if custom, N/mm²
  mc: float       # if custom

tool:
  diameter: float       # mm
  num_teeth: integer
  helix_angle: float    # degrees
  rake_angle: float     # degrees

cutting:
  feed_per_tooth: float  # mm/tooth
  axial_depth: float     # mm
  radial_depth: float    # mm
  milling_type: "up" or "down"

analysis:
  num_lobes: integer
  points_per_lobe: integer  # default 1000
  N_min: float or null      # RPM
  N_max: float or null      # RPM

output:
  save_plots: boolean
  plot_format: "png" or "svg"
  plot_dpi: integer
  save_data: boolean
  generate_report: boolean
```

**For Measured FRF Systems:**

```yaml
system_type: measured

measured:
  frf_file: string           # path to CSV/JSON
  frequency_column: string   # column name
  real_column: string        # column name
  imag_column: string        # column name

material:
  # same as analytical

tool:
  # same as analytical

cutting:
  # same as analytical

analysis:
  num_lobes: integer
  N_min: float or null
  N_max: float or null

output:
  # same as analytical
  include_frequency_plot: boolean  # additional for FRF
```

### 9.2 FRF Data File Format

**CSV Format Requirements:**

- Must have header row
- Required columns:
  - Frequency [Hz]
  - Real part of FRF [m/N]
  - Imaginary part of FRF [m/N]
- Optional column:
  - Coherence (for quality assessment)

**Example Structure:**
```
frequency,real_frf,imag_frf,coherence
50.0,-1.5e-8,-2.3e-8,0.95
51.0,-1.6e-8,-2.4e-8,0.96
...
```

**Validation Rules:**
- Frequencies must be in Hz (absolute values, not ratios)
- FRF units must be m/N
- Must contain at least some negative G_R values
- Coherence values should be between 0 and 1 if present
- Minimum 100 data points
- Recommended to cover all relevant modes

### 9.3 Command-Line Interface

**Basic Syntax:**

```
python -m stability_lobe_diagram [config_file.yaml]

or

python -m stability_lobe_diagram [options]
```

**Required Options for Direct Command:**
- `--system`: "analytical" or "measured"

**For Analytical System:**
- `--k`: stiffness [N/m]
- `--zeta`: damping ratio
- `--fn`: natural frequency [Hz]

**For Measured FRF:**
- `--frf-file`: path to FRF data file

**Common Options:**
- `--material`: material name from database or "custom"
- `--diameter`: tool diameter [mm]
- `--teeth`: number of teeth
- `--helix`: helix angle [degrees]
- `--rake`: rake angle [degrees]
- `--feed`: feed per tooth [mm/tooth]
- `--axial`: axial depth [mm]
- `--radial`: radial depth [mm]
- `--milling-type`: "up" or "down"
- `--lobes`: number of lobes to calculate
- `--rpm-range`: "min max" spindle speed range

**Output Options:**
- `--output-dir`: output directory path
- `--save-plots`: flag to save plots
- `--save-data`: flag to save data files
- `--report`: flag to generate report

---

## 10. Validation and Error Handling

### 10.1 Input Validation Rules

**Analytical System Parameters:**
- k must be positive
- zeta must be in range (0, 1)
- fn must be positive
- m (if provided) must be positive

**Measured FRF Parameters:**
- Minimum 100 data points
- Frequencies monotonically increasing
- No NaN or infinite values
- At least some negative G_R values must exist
- All frequencies must be positive

**Material Parameters:**
- Kc1.1 must be positive
- mc must be in range (0, 1)

**Tool Parameters:**
- diameter must be positive
- num_teeth must be ≥ 1
- helix_angle must be in range (0, 90)
- rake_angle can be negative but typically in range (-10, 20)

**Cutting Conditions:**
- feed_per_tooth must be positive
- axial_depth must be positive
- radial_depth must be positive
- radial_depth must be ≤ diameter
- milling_type must be "up" or "down"

**Analysis Settings:**
- num_lobes must be ≥ 1
- If provided, N_min < N_max
- If provided, N_min and N_max must be positive

### 10.2 Warning Messages

**The system shall issue warnings for:**

**Analytical Systems:**
- High damping (ζ > 0.1): "High damping ratio may indicate measurement error"
- Very low stiffness (k < 5e5): "Low stiffness may produce unrealistic results"
- Newton-Raphson non-convergence: "Optimum point calculation did not converge for lobe {n}"

**Measured FRF Systems:**
- No negative G_R: "FRF contains no negative G_R values - cannot calculate stability"
- Few valid points: "Lobe {n} has fewer than 50 valid points - results may be unreliable"
- Many filtered points: "{count} points filtered due to invalid phase shift"
- Narrow frequency range: "FRF frequency range may be insufficient for analysis"
- Low coherence: "FRF data has low coherence in frequency range {f1}-{f2} Hz"

**Both Systems:**
- Very low cutting stiffness: "Low cutting stiffness ({ks}) may produce unrealistic results"
- Empty lobe: "No valid points found for lobe {n} in specified spindle speed range"
- Unstable operating point: "WARNING: Operating point (N={N}, b={b}) is UNSTABLE"
- Missing optimum points: "No optimum points found between lobes {n} and {n+1}"

### 10.3 Error Handling Requirements

**The system shall handle errors by:**

1. **Input Validation Errors:**
   - Reject invalid inputs immediately with clear error message
   - Specify which parameter is invalid and why
   - Suggest valid range or correction

2. **Runtime Errors:**
   - Continue processing with available lobes if one lobe fails
   - Log detailed error information for debugging
   - Provide partial results with warnings about missing data

3. **File I/O Errors:**
   - Handle missing files gracefully
   - Validate file format before processing
   - Provide clear error message if format is incorrect

4. **Numerical Errors:**
   - Detect and handle division by zero
   - Handle convergence failures in iterative methods
   - Filter out invalid numerical results (NaN, Inf)

---

## 11. Testing Requirements

### 11.1 Unit Tests

**Required Test Coverage:**

**Material and Cutting Force Tests:**
- Kienzle-Victor calculation accuracy
- Edge cases (very small/large chip thickness)
- Material parameter validation
- Custom material creation

**Geometry Tests:**
- Chip thickness calculation (full and partial immersion)
- Up-milling vs down-milling engagement
- Boundary cases (ae/D = 0.5, 1.0)
- Force coefficient calculations

**Analytical System Tests:**
- FRF calculation from system parameters
- Minimum chip width calculation
- Lobe generation with various parameters
- Dimensionless to dimensional conversions
- Newton-Raphson convergence

**Measured FRF System Tests:**
- FRF data import and validation
- Interpolation accuracy
- Phase shift calculation
- Lobe generation from FRF data
- Handling of multimodal FRF
- Filtering of invalid points

**Stability Envelope Tests:**
- Envelope calculation from multiple lobes
- Interpolation accuracy
- Handling of non-overlapping lobes

**Operating Point Validation Tests:**
- Stable point detection
- Unstable point detection
- Boundary points
- Points outside analyzed range

### 11.2 Integration Tests

**Required Integration Tests:**

1. **Complete Workflow - Analytical:**
   - Load configuration file
   - Generate all lobes
   - Calculate envelope and optimum points
   - Generate plots
   - Export data
   - Generate report

2. **Complete Workflow - Measured FRF:**
   - Load configuration and FRF file
   - Generate all lobes
   - Calculate envelope and optimum points
   - Generate plots (including frequency plot)
   - Export data
   - Generate report

3. **Comparison Tests:**
   - Generate synthetic FRF from analytical system
   - Feed to measured FRF system
   - Verify lobes match analytical solution within tolerance

4. **Multimodal FRF Test:**
   - Create synthetic multimodal FRF
   - Verify system handles multiple modes correctly
   - Verify lobes show expected complexity

### 11.3 Validation Against Published Results

**Requirements:**

1. Reproduce results from Yue (2006) paper:
   - Use exact parameters from paper example
   - Verify b_min calculation
   - Verify lobe curves match published figures
   - Verify optimum points match published values
   - Tolerance: < 1% difference

2. Validate against provided Python code:
   - Use same test cases
   - Compare numerical results
   - Verify plot appearance
   - Tolerance: < 0.1% difference

3. Cross-validation:
   - Generate lobes using analytical method
   - Convert to synthetic FRF
   - Generate lobes using FRF method
   - Verify both methods produce consistent results
   - Tolerance: < 5% difference (due to discretization)

### 11.4 Physical Constraint Validation

**The system must verify:**

- For analytical systems: b_min > 0
- All frequencies > 0
- All spindle speeds > 0
- All chip widths > 0 in valid regions
- For FRF systems: phase shift ε in [π, 2π]
- Stability boundary is continuous (no jumps)
- Optimum points lie on lobe curves
- Safety factors are correctly calculated

---

## 12. Performance Requirements

### 12.1 Computation Time Limits

**The system shall meet the following performance targets:**

- Single lobe generation (analytical): < 0.1 seconds
- Single lobe generation (FRF, 1000 points): < 0.2 seconds
- Complete diagram (4 lobes, analytical): < 0.5 seconds
- Complete diagram (4 lobes, FRF): < 1.0 second
- Optimum point calculation: < 0.2 seconds
- Stability envelope calculation: < 0.1 seconds
- Complete analysis including plots: < 2 seconds (analytical), < 5 seconds (FRF)

**Performance for Large FRF Datasets:**
- 10,000 FRF points: < 5 seconds total
- 100,000 FRF points: < 30 seconds total

### 12.2 Memory Usage

**The system shall:**
- Peak RAM usage: < 500 MB for typical analysis
- Support FRF files up to 100,000 points
- Scale memory usage linearly with FRF data size
- Release memory after completing analysis

### 12.3 Numerical Stability

**The system shall:**
- Handle near-singular points robustly (r close to 1.0)
- Provide stable results with poor quality FRF data
- Avoid numerical overflow/underflow
- Handle extreme parameter values gracefully
- Detect and warn about questionable results

---

## 13. Documentation Requirements

### 13.1 User Documentation

**Required Content:**

1. **Quick Start Guide:**
   - Installation instructions
   - Basic usage examples for both system types
   - Expected outputs

2. **Detailed User Manual:**
   - When to use analytical vs. measured FRF
   - Parameter selection guidelines
   - Material database reference
   - Interpreting stability diagrams
   - Understanding optimum points
   - Safety margin recommendations

3. **FRF Measurement Guidelines:**
   - How to measure FRF properly
   - Required measurement equipment
   - Data quality assessment
   - Common measurement errors to avoid
   - Frequency range selection

4. **Troubleshooting Guide:**
   - Common errors and solutions
   - Invalid results diagnosis
   - Performance issues
   - Data quality problems

5. **Example Workflows:**
   - Complete example: aluminum milling (analytical)
   - Complete example: steel milling (FRF)
   - Advanced example: multimodal system (FRF)

### 13.2 Theoretical Documentation

**Required Content:**

1. **Mathematical Background:**
   - Regenerative chatter theory overview
   - Derivation of governing equations
   - Analytical method equations (with equation numbers from paper)
   - FRF method equations (with equation numbers from paper)

2. **Key Differences Between Methods:**
   - Why fn and zeta cannot be extracted from multimodal FRF
   - Phase shift physical meaning and constraints
   - When each method is appropriate

3. **Assumptions and Limitations:**
   - Valid operating ranges
   - Model limitations
   - When results may be unreliable
   - Process damping effects (not included)

4. **References:**
   - Yue (2006) paper
   - Original chatter theory papers (Tobias, Tlusty, etc.)
   - Cutting force modeling references
   - MachiningDoctor.com methods

### 13.3 API Documentation

**Required Content:**

1. **Class Reference:**
   - All public classes
   - Properties and their units
   - Methods and their parameters
   - Return value specifications

2. **Function Reference:**
   - All public functions
   - Parameter descriptions with units
   - Return value descriptions
   - Usage examples

3. **Module Organization:**
   - Module hierarchy
   - Import paths
   - Dependencies between modules

4. **Code Examples:**
   - Creating systems programmatically
   - Generating lobes
   - Customizing plots
   - Exporting data
   - Both analytical and FRF workflows

### 13.4 Code Documentation Standards

**Requirements:**

- All public classes and functions must have docstrings
- Docstring format: Google or NumPy style
- All function parameters must have type hints
- Complex algorithms must have inline comments
- Units must be specified in docstrings and comments
- Special emphasis on documenting analytical vs. FRF differences

---

## 14. Success Criteria

### 14.1 Functional Completeness

**The application shall be considered functionally complete when:**

- ✅ Generates correct SLD from analytical systems (k, ζ, fn)
- ✅ Generates correct SLD from measured FRF (no fn/zeta extraction)
- ✅ Correctly implements Kienzle-Victor cutting force model
- ✅ Calculates cutting stiffness using advanced methods
- ✅ Produces accurate dimensional plots (N vs b) for both system types
- ✅ Produces frequency plots (f vs b) for FRF systems
- ✅ Identifies optimum chip widths (analytical for analytical systems, numerical for FRF)
- ✅ Validates operating points with safety factors
- ✅ Exports data in required formats (CSV, JSON, images)
- ✅ Generates comprehensive reports

### 14.2 Quality Standards

**The application shall meet the following quality standards:**

- ✅ All unit tests pass (>100 tests covering both system types)
- ✅ Code coverage > 80%
- ✅ Reproduces Yue (2006) paper results within 1% tolerance
- ✅ FRF method matches analytical solution (single-mode synthetic FRF) within 5%
- ✅ Handles multimodal FRF correctly
- ✅ No critical bugs in validation tests
- ✅ Documentation complete with examples for both workflows

### 14.3 Performance Standards

**The application shall meet the following performance standards:**

- ✅ Complete analytical analysis in < 2 seconds
- ✅ Complete FRF analysis (1000 points) in < 5 seconds
- ✅ Handles FRF files with 10,000+ points
- ✅ Memory usage < 500 MB for typical cases
- ✅ Numerically stable for all valid parameter ranges

### 14.4 Usability Standards

**The application shall meet the following usability standards:**

- ✅ CLI works with configuration files for both system types
- ✅ Clear error messages with guidance for both system types
- ✅ Comprehensive user documentation
- ✅ At least 2 complete example workflows (one per system type)
- ✅ Validation warnings are actionable
- ✅ Plot legends and labels are clear

---

## Appendix A: Mathematical Notation Reference

| Symbol | Description | Units | Used In |
|--------|-------------|-------|---------|
| b | Chip width (axial depth of cut) | mm | Both |
| b_min | Minimum chip width | mm | Analytical only |
| rb | Dimensionless chip width ratio | - | Analytical only |
| k | Machine tool stiffness | N/m | Analytical only |
| ks | Cutting stiffness | N/m² | Both |
| ζ | Damping ratio | - | Analytical only |
| fn | Natural frequency | Hz | Analytical only |
| f | Chatter frequency | Hz | Both |
| ft | Tooth frequency | Hz | Both |
| r | Chatter frequency ratio (f/fn) | - | Analytical only |
| rt | Tooth frequency ratio (ft/fn) | - | Analytical only |
| N | Spindle speed | RPM | Both |
| nt | Number of teeth | - | Both |
| n | Lobe number | - | Both |
| ε | Phase shift angle | rad | FRF only |
| G | Frequency response function | m/N | Both |
| G_R | Real part of FRF | m/N | Both |
| G_I | Imaginary part of FRF | m/N | Both |
| Kc | Specific cutting force | N/mm² | Both |
| Kc1.1 | Specific cutting force at h=1mm | N/mm² | Both |
| mc | Material constant | - | Both |
| h | Uncut chip thickness | mm | Both |
| fz | Feed per tooth | mm/tooth | Both |
| ae | Radial depth of cut | mm | Both |
| D | Tool diameter | mm | Both |
| Kt | Tangential force coefficient | N/mm² | Both |
| Kr | Radial force coefficient | N/mm² | Both |
| Ka | Axial force coefficient | N/mm² | Both |
| φ | Angular position | rad | Both |

---

## Appendix B: Key Equations Summary

### For Analytical Systems (Using k, ζ, fn)

**Equation 3 - Chip Width (General Form):**
```
b = -1 / (2 × ks × G_R)
```

**Equation 5 - Minimum Chip Width:**
```
b_min = (2 × k × ζ × (1 + ζ)) / ks
```

**Equation 6 - Dimensionless Chip Width Ratio:**
```
rb = b / b_min = [(1 - r²)² + (2ζr)²] / [-4ζ(1 + ζ)(1 - r²)]
```

**Equation 9 - Regenerative Chatter (Tooth Frequency Ratio):**
```
rt = r / [n + 0.5 + (1/π) × arctan(-2ζr / (1 - r²))]
```

**Spindle Speed Calculation:**
```
N = 60 × ft / nt = 60 × rt × fn / nt
```

**Equation 15 - For Optimum Points:**
```
r / [n + 1 + 0.5 + (1/π) × arctan(-2ζr / (1 - r²))] = 1/(n+1)
```
(Solve iteratively for r)

### For Measured FRF Systems (Using G_R(f) and G_I(f))

**Equation 3 - Chip Width:**
```
b(f) = -1 / (2 × ks × G_R(f))
```
Valid only where G_R(f) < 0

**Equation 8 - Phase Shift:**
```
ε(f) = π + 2 × arctan(G_I(f) / G_R(f))
```
Physical constraint: π < ε < 2π

**Equation 7 - Regenerative Chatter Relationship:**
```
f / ft = n + ε(f) / (2π)
```

**Tooth Frequency:**
```
ft(f, n) = f / [n + ε(f)/(2π)]
```

**Spindle Speed:**
```
N(f, n) = 60 × ft(f, n) / nt
```

### Kienzle-Victor Model (Both Systems)

**Specific Cutting Force:**
```
Kc = Kc1.1 × h^(-mc)
```

**Average Chip Thickness (Full Immersion):**
```
h_avg = fz × (2/π)
```

**Average Chip Thickness (Partial Immersion):**
```
h_avg = fz × (1 - cos(φ_end)) / φ_end
```

---

## Appendix C: Configuration Examples

### Example 1: Analytical System - Aluminum Milling

```yaml
system_type: analytical

analytical:
  k: 2.0e6
  zeta: 0.05
  fn: 150
  m: null

material:
  type: database
  name: "Aluminum 6061-T6"

tool:
  diameter: 10.0
  num_teeth: 3
  helix_angle: 45.0
  rake_angle: 10.0

cutting:
  feed_per_tooth: 0.025
  axial_depth: 20.0
  radial_depth: 0.5
  milling_type: down

analysis:
  num_lobes: 4
  points_per_lobe: 1000
  N_min: 5000
  N_max: 24000

output:
  save_plots: true
  plot_format: png
  plot_dpi: 300
  save_data: true
  generate_report: true
```

### Example 2: Measured FRF - Steel Milling

```yaml
system_type: measured

measured:
  frf_file: "data/machine_frf_steel.csv"
  frequency_column: "freq"
  real_column: "real"
  imag_column: "imag"

material:
  type: database
  name: "Steel AISI 1045"

tool:
  diameter: 12.0
  num_teeth: 4
  helix_angle: 38.0
  rake_angle: 8.0

cutting:
  feed_per_tooth: 0.03
  axial_depth: 25.0
  radial_depth: 1.5
  milling_type: down

analysis:
  num_lobes: 5
  N_min: 3000
  N_max: 18000

output:
  save_plots: true
  plot_format: png
  plot_dpi: 300
  save_data: true
  generate_report: true
  include_frequency_plot: true
```

### Example 3: Custom Material with Measured FRF

```yaml
system_type: measured

measured:
  frf_file: "data/multimodal_frf.csv"
  frequency_column: "frequency"
  real_column: "real_frf"
  imag_column: "imag_frf"

material:
  type: custom
  Kc1_1: 1900
  mc: 0.19
  name: "Custom Steel Alloy"
  description: "Proprietary steel alloy"

tool:
  diameter: 16.0
  num_teeth: 6
  helix_angle: 42.0
  rake_angle: 12.0

cutting:
  feed_per_tooth: 0.04
  axial_depth: 30.0
  radial_depth: 2.0
  milling_type: down

analysis:
  num_lobes: 6
  N_min: 2000
  N_max: 20000

output:
  save_plots: true
  plot_format: svg
  save_data: true
  generate_report: true
  include_frequency_plot: true
```

---

**Document Version:** 2.1 (Specifications Only)  
**Date:** 2025  
**Status:** Ready for Implementation