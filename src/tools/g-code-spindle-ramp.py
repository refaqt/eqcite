#!/usr/bin/env python3
"""
G-code Spindle Ramp Generator

Moved from src/g-code-spindle-ramp.py to src/tools/g-code-spindle-ramp.py.
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


def validate_inputs(ramp_time, frequency_resolution, start_speed, end_speed):
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
    frequency_resolution_rpm = frequency_resolution * 60
    speed_range = end_speed - start_speed
    num_steps = speed_range / frequency_resolution_rpm
    num_steps_int = int(round(num_steps))
    delta_t = ramp_time / num_steps_int if num_steps_int > 0 else 0
    return delta_t, num_steps_int, frequency_resolution_rpm


def generate_gcode(
    ramp_time,
    frequency_resolution,
    start_speed,
    end_speed,
    delta_t,
    num_steps,
    frequency_resolution_rpm,
):
    gcode_lines = []
    gcode_lines.append("; G-code Spindle Ramp")
    gcode_lines.append(f"; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    gcode_lines.append(f"; Ramp time: {ramp_time} seconds")
    gcode_lines.append(
        f"; Frequency resolution: {frequency_resolution} Hz ({frequency_resolution_rpm:.2f} RPM)"
    )
    gcode_lines.append(f"; Start speed: {start_speed} RPM")
    gcode_lines.append(f"; End speed: {end_speed} RPM")
    gcode_lines.append(f"; Number of steps: {num_steps}")
    gcode_lines.append(f"; Delta t (wait time): {delta_t:.6f} seconds")
    gcode_lines.append("")
    gcode_lines.append(f"S{int(start_speed)}")
    gcode_lines.append("M3")
    gcode_lines.append("")

    current_speed = start_speed
    for _ in range(num_steps):
        gcode_lines.append(f"G4 P{delta_t:.6f}")
        current_speed += frequency_resolution_rpm
        if current_speed > end_speed:
            current_speed = end_speed
        gcode_lines.append(f"S{int(round(current_speed))}")

    gcode_lines.append("")
    gcode_lines.append("G4 P1.0")
    gcode_lines.append("")
    gcode_lines.append("M5")
    return "\n".join(gcode_lines)


def generate_filename(start_speed, end_speed, duration):
    return f"Spindle_ramp_S{int(start_speed)}_S{int(end_speed)}_T{int(duration)}.nc"


def write_gcode_file(gcode_content, filename):
    with open(filename, "w") as f:
        f.write(gcode_content)


def main():
    print("G-code Spindle Ramp Generator")
    print("=" * 60)
    print("\nValidating inputs...")
    try:
        validate_inputs(RAMP_TIME, FREQUENCY_RESOLUTION, START_SPINDLE_SPEED, END_SPINDLE_SPEED)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    print("\nCalculating parameters...")
    delta_t, num_steps, frequency_resolution_rpm = calculate_delta_t(
        RAMP_TIME, FREQUENCY_RESOLUTION, START_SPINDLE_SPEED, END_SPINDLE_SPEED
    )
    print(f"  Frequency resolution: {FREQUENCY_RESOLUTION} Hz = {frequency_resolution_rpm:.2f} RPM")
    print(f"  Number of steps: {num_steps}")
    print(f"  Delta t (wait time): {delta_t:.6f} seconds")

    print("\nGenerating G-code...")
    gcode_content = generate_gcode(
        RAMP_TIME,
        FREQUENCY_RESOLUTION,
        START_SPINDLE_SPEED,
        END_SPINDLE_SPEED,
        delta_t,
        num_steps,
        frequency_resolution_rpm,
    )
    print("  G-code generated")

    print("\nWriting to file...")
    filename = generate_filename(START_SPINDLE_SPEED, END_SPINDLE_SPEED, RAMP_TIME)
    write_gcode_file(gcode_content, filename)
    filepath = os.path.abspath(filename)
    print(f"  File saved: {filepath}")
    print("\nGeneration completed successfully!")


if __name__ == "__main__":
    main()

