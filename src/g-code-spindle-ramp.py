#!/usr/bin/env python3
"""
G-code Spindle Ramp Generator

Generates a G-code file (.nc) that ramps the spindle speed from start to end
speed with specified frequency resolution over a given ramp time.

The script calculates the appropriate wait time (delta_t) between speed
increments to achieve the desired ramp time and frequency resolution.
"""

import os
from datetime import datetime

# =============================================================================
# INPUT PARAMETERS
# =============================================================================

# Ramp time in seconds (total time for the entire ramp)
RAMP_TIME = 10.0

# Frequency resolution in Hz (will be converted to RPM: Hz * 60 = RPM per step)
FREQUENCY_RESOLUTION = 0.2

# Start spindle speed in RPM
START_SPINDLE_SPEED = 1000

# End spindle speed in RPM
END_SPINDLE_SPEED = 24000

# =============================================================================
# VALIDATION AND CALCULATIONS
# =============================================================================

def validate_inputs(ramp_time, frequency_resolution, start_speed, end_speed):
    """
    Validate input parameters.
    
    Parameters:
    -----------
    ramp_time : float
        Ramp time in seconds
    frequency_resolution : float
        Frequency resolution in Hz
    start_speed : float
        Start spindle speed in RPM
    end_speed : float
        End spindle speed in RPM
        
    Raises:
    -------
    ValueError
        If inputs are invalid
    """
    if ramp_time <= 0:
        raise ValueError("RAMP_TIME must be positive")
    if frequency_resolution <= 0:
        raise ValueError("FREQUENCY_RESOLUTION must be positive")
    if start_speed <= 0:
        raise ValueError("START_SPINDLE_SPEED must be positive")
    if end_speed <= 0:
        raise ValueError("END_SPINDLE_SPEED must be positive")
    if end_speed <= start_speed:
        raise ValueError("END_SPINDLE_SPEED must be greater than START_SPINDLE_SPEED")
    if (end_speed - start_speed) / (frequency_resolution * 60) <= 0:
        raise ValueError("Cannot achieve specified frequency resolution with given speed range")


def calculate_delta_t(ramp_time, frequency_resolution, start_speed, end_speed):
    """
    Calculate delta_t (wait time between speed increments).
    
    Parameters:
    -----------
    ramp_time : float
        Total ramp time in seconds
    frequency_resolution : float
        Frequency resolution in Hz
    start_speed : float
        Start spindle speed in RPM
    end_speed : float
        End spindle speed in RPM
        
    Returns:
    --------
    tuple : (delta_t, num_steps, frequency_resolution_rpm)
        delta_t: Wait time in seconds between increments
        num_steps: Number of speed increment steps
        frequency_resolution_rpm: Frequency resolution converted to RPM
    """
    # Convert frequency resolution from Hz to RPM
    frequency_resolution_rpm = frequency_resolution * 60
    
    # Calculate number of steps needed
    speed_range = end_speed - start_speed
    num_steps = speed_range / frequency_resolution_rpm
    
    # Round to integer and recalculate for exact timing
    num_steps_int = int(round(num_steps))
    
    # Calculate delta_t based on actual number of steps
    delta_t = ramp_time / num_steps_int if num_steps_int > 0 else 0
    
    return delta_t, num_steps_int, frequency_resolution_rpm


# =============================================================================
# G-CODE GENERATION
# =============================================================================

def generate_gcode(ramp_time, frequency_resolution, start_speed, end_speed, delta_t, num_steps, frequency_resolution_rpm):
    """
    Generate G-code for spindle speed ramp.
    
    Parameters:
    -----------
    ramp_time : float
        Total ramp time in seconds
    frequency_resolution : float
        Frequency resolution in Hz
    start_speed : float
        Start spindle speed in RPM
    end_speed : float
        End spindle speed in RPM
    delta_t : float
        Wait time in seconds between increments
    num_steps : int
        Number of speed increment steps
    frequency_resolution_rpm : float
        Frequency resolution in RPM
        
    Returns:
    --------
    str
        G-code content as string
    """
    gcode_lines = []
    
    # Header comments
    gcode_lines.append("; G-code Spindle Ramp")
    gcode_lines.append(f"; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    gcode_lines.append(f"; Ramp time: {ramp_time} seconds")
    gcode_lines.append(f"; Frequency resolution: {frequency_resolution} Hz ({frequency_resolution_rpm:.2f} RPM)")
    gcode_lines.append(f"; Start speed: {start_speed} RPM")
    gcode_lines.append(f"; End speed: {end_speed} RPM")
    gcode_lines.append(f"; Number of steps: {num_steps}")
    gcode_lines.append(f"; Delta t (wait time): {delta_t:.6f} seconds")
    gcode_lines.append("")
    
    # Set initial spindle speed
    gcode_lines.append(f"S{int(start_speed)}")
    
    # Start spindle clockwise (M3)
    gcode_lines.append("M3")
    gcode_lines.append("")
    
    # Generate ramp sequence
    current_speed = start_speed
    for step in range(num_steps):
        # Wait for delta_t seconds (G4 P uses seconds)
        gcode_lines.append(f"G4 P{delta_t:.6f}")
        
        # Calculate next speed
        current_speed += frequency_resolution_rpm
        
        # Clamp to end speed to avoid rounding errors
        if current_speed > end_speed:
            current_speed = end_speed
        
        # Set new spindle speed
        gcode_lines.append(f"S{int(round(current_speed))}")
    
    gcode_lines.append("")
    
    # Hold at maximum speed for at least 1 second
    gcode_lines.append("G4 P1.0")
    gcode_lines.append("")
    
    # Stop spindle
    gcode_lines.append("M5")
    
    return "\n".join(gcode_lines)


# =============================================================================
# FILE OUTPUT
# =============================================================================

def generate_filename(start_speed, end_speed, duration):
    """
    Generate structured filename for output G-code file.
    
    Parameters:
    -----------
    start_speed : float
        Start spindle speed in RPM
    end_speed : float
        End spindle speed in RPM
    duration : float
        Ramp duration in seconds
        
    Returns:
    --------
    str
        Filename with format: Spindle_ramp_S(start speed)_S(end_speed)_T(duration in seconds).nc
    """
    return f"Spindle_ramp_S{int(start_speed)}_S{int(end_speed)}_T{int(duration)}.nc"


def write_gcode_file(gcode_content, filename):
    """
    Write G-code content to file.
    
    Parameters:
    -----------
    gcode_content : str
        G-code content as string
    filename : str
        Output filename
    """
    with open(filename, 'w') as f:
        f.write(gcode_content)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    print("G-code Spindle Ramp Generator")
    print("=" * 60)
    
    # Validate inputs
    print("\nValidating inputs...")
    try:
        validate_inputs(RAMP_TIME, FREQUENCY_RESOLUTION, START_SPINDLE_SPEED, END_SPINDLE_SPEED)
        print("  Inputs valid")
    except ValueError as e:
        print(f"  ERROR: {e}")
        return
    
    # Calculate delta_t
    print("\nCalculating parameters...")
    delta_t, num_steps, frequency_resolution_rpm = calculate_delta_t(
        RAMP_TIME, FREQUENCY_RESOLUTION, START_SPINDLE_SPEED, END_SPINDLE_SPEED
    )
    print(f"  Frequency resolution: {FREQUENCY_RESOLUTION} Hz = {frequency_resolution_rpm:.2f} RPM")
    print(f"  Number of steps: {num_steps}")
    print(f"  Delta t (wait time): {delta_t:.6f} seconds")
    
    # Generate G-code
    print("\nGenerating G-code...")
    gcode_content = generate_gcode(
        RAMP_TIME, FREQUENCY_RESOLUTION, START_SPINDLE_SPEED, END_SPINDLE_SPEED,
        delta_t, num_steps, frequency_resolution_rpm
    )
    print("  G-code generated")
    
    # Write to file
    print("\nWriting to file...")
    filename = generate_filename(START_SPINDLE_SPEED, END_SPINDLE_SPEED, RAMP_TIME)
    write_gcode_file(gcode_content, filename)
    filepath = os.path.abspath(filename)
    print(f"  File saved: {filepath}")
    
    print("\n" + "=" * 60)
    print("Generation completed successfully!")


if __name__ == "__main__":
    main()

