#!/usr/bin/env python3
"""
Sensor geometry and coordinate transforms.

This module is the single source of truth for how spindle/tool/workbed
accelerations map to sensor locations (forward model) and how sensor
readings map back to physical coordinates (inverse model).

Angles are in radians; accelerations are in m/s².
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SpindleAngles:
    """Geometry of the spindle accelerometer ring."""

    alpha1: float = math.radians(57.1715)
    beta1: float = math.radians(76.8368)
    beta2: float = math.radians(85.2045)


def default_spindle_angles() -> SpindleAngles:
    """Return default spindle accelerometer angles."""
    return SpindleAngles()


# ---------------------------------------------------------------------------
# Forward model: physical coordinates -> sensor accelerations
# ---------------------------------------------------------------------------

def spindle_to_sensors(
    x_spindle: np.ndarray,
    y_spindle: np.ndarray,
    z_spindle: np.ndarray,
    angles: SpindleAngles | None = None,
) -> Dict[str, np.ndarray]:
    """
    Map spindle accelerations to accelerometer locations on the spindle.

    Returns a dict with keys:
      - 'acc_spindle_top'
      - 'acc_spindle_left'
      - 'acc_spindle_right'
    """
    if angles is None:
        angles = default_spindle_angles()

    alpha1 = angles.alpha1
    beta1 = angles.beta1
    beta2 = angles.beta2

    x_spindle = np.asarray(x_spindle)
    y_spindle = np.asarray(y_spindle)
    z_spindle = np.asarray(z_spindle)

    acc_spindle_top = (
        z_spindle * math.cos(math.pi / 2.0 - beta2)
        + y_spindle * math.cos(beta2)
    )
    acc_spindle_left = (
        -x_spindle * math.cos(alpha1)
        + y_spindle * math.cos(beta1)
        + z_spindle * math.cos(math.pi / 2.0 - beta1)
    )
    acc_spindle_right = (
        x_spindle * math.cos(alpha1)
        + y_spindle * math.cos(beta1)
        + z_spindle * math.cos(math.pi / 2.0 - beta1)
    )

    return {
        "acc_spindle_top": acc_spindle_top,
        "acc_spindle_left": acc_spindle_left,
        "acc_spindle_right": acc_spindle_right,
    }


def tool_to_sensors(
    x_tool: np.ndarray,
    y_tool: np.ndarray,
    acc_workbed_x: np.ndarray,
    acc_workbed_y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Map tool accelerations and workbed reference to tool-mounted sensor values.

    Inverse relation is:
      x_tool = acc_tool_x - acc_workbed_x
      y_tool = acc_tool_y - acc_workbed_y
    """
    x_tool = np.asarray(x_tool)
    y_tool = np.asarray(y_tool)
    acc_workbed_x = np.asarray(acc_workbed_x)
    acc_workbed_y = np.asarray(acc_workbed_y)

    acc_tool_x = x_tool + acc_workbed_x
    acc_tool_y = y_tool + acc_workbed_y

    return {
        "acc_tool_x": acc_tool_x,
        "acc_tool_y": acc_tool_y,
    }


# ---------------------------------------------------------------------------
# Inverse model: sensor accelerations -> physical coordinates
# ---------------------------------------------------------------------------

def sensors_to_spindle(
    acc_spindle_top: np.ndarray,
    acc_spindle_left: np.ndarray,
    acc_spindle_right: np.ndarray,
    acc_workbed_x: np.ndarray,
    acc_workbed_y: np.ndarray,
    angles: SpindleAngles | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover spindle accelerations x_spindle, y_spindle, z_spindle from
    spindle-mounted sensors and workbed references.

    This implements the equations specified in the design:

      x_spindle = (acc_spindle_right - acc_spindle_left) / cos(alpha1) - acc_workbed_x

      y_spindle = (
          acc_spindle_top / cos(pi/2 - beta2)
          - (acc_spindle_left + acc_spindle_right) / (2 * cos(pi/2 - beta1))
      ) / (1 / tan(beta2) - 1 / tan(beta1)) - acc_workbed_y

      z_spindle = (acc_spindle_top - y_spindle * cos(beta2)) / cos(pi / 2 - beta2)

    The original spec referenced (A2 + A3) in the middle term; here we assume
    those correspond to the left/right spindle sensors.
    """
    if angles is None:
        angles = default_spindle_angles()

    alpha1 = angles.alpha1
    beta1 = angles.beta1
    beta2 = angles.beta2

    acc_spindle_top = np.asarray(acc_spindle_top)
    acc_spindle_left = np.asarray(acc_spindle_left)
    acc_spindle_right = np.asarray(acc_spindle_right)
    acc_workbed_x = np.asarray(acc_workbed_x)
    acc_workbed_y = np.asarray(acc_workbed_y)

    # x_spindle from left/right sensors and workbed reference
    x_spindle = (
        (acc_spindle_right - acc_spindle_left) / math.cos(alpha1)
        - acc_workbed_x
    )

    # y_spindle from top and average of left/right (spec-consistent form)
    top_term = acc_spindle_top / math.cos(math.pi / 2.0 - beta2)
    ring_term = (acc_spindle_left + acc_spindle_right) / (
        2.0 * math.cos(math.pi / 2.0 - beta1)
    )
    denom = (1.0 / math.tan(beta2)) - (1.0 / math.tan(beta1))
    y_spindle = (top_term - ring_term) / denom - acc_workbed_y

    # z_spindle from top sensor and y_spindle
    z_spindle = (
        acc_spindle_top - y_spindle * math.cos(beta2)
    ) / math.cos(math.pi / 2.0 - beta2)

    return x_spindle, y_spindle, z_spindle


def sensors_to_tool(
    acc_tool_x: np.ndarray,
    acc_tool_y: np.ndarray,
    acc_workbed_x: np.ndarray,
    acc_workbed_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover tool accelerations from tool-mounted sensors and workbed references:

      x_tool = acc_tool_x - acc_workbed_x
      y_tool = acc_tool_y - acc_workbed_y
    """
    acc_tool_x = np.asarray(acc_tool_x)
    acc_tool_y = np.asarray(acc_tool_y)
    acc_workbed_x = np.asarray(acc_workbed_x)
    acc_workbed_y = np.asarray(acc_workbed_y)

    x_tool = acc_tool_x - acc_workbed_x
    y_tool = acc_tool_y - acc_workbed_y

    return x_tool, y_tool


# ---------------------------------------------------------------------------
# Hammer / force channels
# ---------------------------------------------------------------------------

def sensors_to_inputs(
    hammer_x: np.ndarray | None = None,
    hammer_y: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Map hammer sensor channels to analysis inputs F_x and F_y.
    """
    inputs: Dict[str, np.ndarray] = {}
    if hammer_x is not None:
        inputs["F_x"] = np.asarray(hammer_x)
    if hammer_y is not None:
        inputs["F_y"] = np.asarray(hammer_y)
    return inputs

