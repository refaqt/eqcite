"""
Experiment configuration schema helpers (FRF): sensor catalog, mapping,
expressions, per-direction analysis I/O, and variable kind classification.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Set, Tuple

import numpy as np

from . import expression_eval

VariableKind = Literal["force", "acceleration", "none"]

VALID_UNITS = frozenset({"N", "m/s²"})


def normalize_output_unit(unit: str) -> str:
    u = (unit or "").strip().replace("m/s^2", "m/s²")
    if u not in VALID_UNITS:
        raise ValueError(f"Invalid sensor unit {unit!r}; expected one of {sorted(VALID_UNITS)}")
    return u


def uses_new_experiment_schema(experiment_cfg: Mapping[str, Any]) -> bool:
    sc = experiment_cfg.get("sensor_catalog")
    return isinstance(sc, list) and len(sc) > 0


def _catalog_by_id(sensor_catalog: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in sensor_catalog:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        if sid:
            out[sid] = row
    return out


def mapping_row_unit(
    row: Mapping[str, Any],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    sid = str(row.get("sensor_id") or "").strip()
    if sid not in catalog_by_id:
        raise ValueError(f"Unknown sensor_id {sid!r} in mapping row {row!r}")
    return normalize_output_unit(str(catalog_by_id[sid].get("unit") or ""))


def build_mapping_name_to_unit(experiment_cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Map each mapping `name` -> physical unit ('N' or 'm/s²')."""
    sensor_catalog = experiment_cfg.get("sensor_catalog") or []
    if not isinstance(sensor_catalog, list):
        raise ValueError("sensor_catalog must be a list")
    catalog_by_id = _catalog_by_id([r for r in sensor_catalog if isinstance(r, dict)])

    mapping = experiment_cfg.get("mapping") or []
    if not isinstance(mapping, list):
        raise ValueError("mapping must be a list")

    name_to_unit: Dict[str, str] = {}
    for row in mapping:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if name in name_to_unit:
            raise ValueError(f"Duplicate mapping name: {name!r}")
        name_to_unit[name] = mapping_row_unit(row, catalog_by_id)
    return name_to_unit


def classify_expression_io(
    expr: str,
    mapping_units: Mapping[str, str],
) -> Tuple[bool, bool]:
    """
    Returns (eligible_force_input, eligible_accel_output) for an expression
    referencing only mapping names, per plan rules.
    """
    expr = (expr or "").strip()
    if not expr:
        return False, False
    try:
        names = expression_eval.referenced_mapping_names(expr)
    except Exception:
        return False, False
    if not names:
        return False, False
    units: List[str] = []
    for n in names:
        if n not in mapping_units:
            return False, False
        units.append(mapping_units[n])
    force_ok = all(u == "N" for u in units)
    accel_ok = all(u == "m/s²" for u in units)
    return force_ok, accel_ok


def build_variable_kinds(experiment_cfg: Mapping[str, Any]) -> Dict[str, VariableKind]:
    """
    Classify every user-facing variable (mapping names + expression names)
    as force, acceleration, or none (ineligible for constrained IO dropdowns).
    """
    mapping_units = build_mapping_name_to_unit(experiment_cfg)
    kinds: Dict[str, VariableKind] = {n: "force" if u == "N" else "acceleration" for n, u in mapping_units.items()}

    expressions = experiment_cfg.get("expressions") or []
    if not isinstance(expressions, list):
        raise ValueError("expressions must be a list")

    used_expr_names: Set[str] = set()
    for row in expressions:
        if not isinstance(row, dict):
            continue
        var_name = str(row.get("name") or "").strip()
        expr = str(row.get("expr") or "").strip()
        if not var_name:
            continue
        if var_name in used_expr_names:
            raise ValueError(f"Duplicate expression variable name: {var_name!r}")
        used_expr_names.add(var_name)
        if var_name in kinds:
            raise ValueError(f"Expression name collides with mapping name: {var_name!r}")
        f_ok, a_ok = classify_expression_io(expr, mapping_units)
        if f_ok and not a_ok:
            kinds[var_name] = "force"
        elif a_ok and not f_ok:
            kinds[var_name] = "acceleration"
        else:
            kinds[var_name] = "none"
    return kinds


def validate_analysis_io(
    experiment_cfg: Mapping[str, Any],
    kinds: Optional[Mapping[str, VariableKind]] = None,
) -> None:
    kinds = kinds or build_variable_kinds(experiment_cfg)
    io = experiment_cfg.get("analysis_io") or {}
    if not isinstance(io, dict):
        raise ValueError("analysis_io must be a dict")
    for direction in ("x", "y"):
        block = io.get(direction)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"analysis_io[{direction!r}] must be a dict")
        inp = str(block.get("input") or "").strip()
        out = str(block.get("output") or "").strip()
        if not inp or not out:
            continue
        if kinds.get(inp) != "force":
            raise ValueError(
                f"analysis_io[{direction}].input must be a force-classified variable; got {inp!r} -> {kinds.get(inp)!r}"
            )
        if kinds.get(out) != "acceleration":
            raise ValueError(
                f"analysis_io[{direction}].output must be an acceleration-classified variable; got {out!r} -> {kinds.get(out)!r}"
            )


def resolve_analysis_io_names(
    experiment_cfg: Mapping[str, Any],
    direction: str,
    *,
    kinds: Optional[Mapping[str, VariableKind]] = None,
) -> Tuple[str, str]:
    """Return (input_name, output_name) for one direction from analysis_io (new schema only)."""
    dir_key = str(direction or "").strip().lower()
    if dir_key not in {"x", "y"}:
        raise ValueError(f"direction must be 'x' or 'y'; got {direction!r}")
    if not uses_new_experiment_schema(experiment_cfg):
        raise ValueError(
            "Experiment configuration requires a non-empty sensor_catalog (new schema)."
        )
    io = experiment_cfg.get("analysis_io") or {}
    if not isinstance(io, dict):
        raise ValueError("analysis_io must be a dict")
    block = io.get(dir_key)
    if not isinstance(block, dict):
        raise ValueError(
            f"Configure Experiment configuration → Input/Output for direction {dir_key!r}."
        )
    inp = str(block.get("input") or "").strip()
    out = str(block.get("output") or "").strip()
    if not inp or not out:
        raise ValueError(
            f"Configure Experiment configuration → Input/Output for direction {dir_key!r} "
            f"(missing input and/or output selection)."
        )
    kinds_map = kinds or build_variable_kinds(experiment_cfg)
    if kinds_map.get(inp) != "force":
        raise ValueError(
            f"analysis_io[{dir_key}].input must be a force-classified variable; got {inp!r} -> {kinds_map.get(inp)!r}"
        )
    if kinds_map.get(out) != "acceleration":
        raise ValueError(
            f"analysis_io[{dir_key}].output must be an acceleration-classified variable; got {out!r} -> {kinds_map.get(out)!r}"
        )
    return inp, out


def bind_directional_io_signals(
    data: Mapping[str, Any],
    input_name: str,
    output_name: str,
) -> Tuple[Any, Any]:
    """Load force input and acceleration output arrays from a measurement dict."""
    if input_name not in data:
        raise ValueError(f"Missing required input signal: {input_name!r}")
    if output_name not in data or data.get(output_name) is None:
        raise ValueError(f"Missing required output signal: {output_name!r}")
    u = np.asarray(data[input_name], dtype=float)
    y_accel = np.asarray(data[output_name], dtype=float)
    return u, y_accel


_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def assert_valid_mapping_name(name: str) -> None:
    n = (name or "").strip()
    if not n or not _NAME_RE.match(n):
        raise ValueError("Mapping / variable names must be non-empty and match [A-Za-z0-9_]+ (no spaces)")


def migrate_legacy_experiment_cfg(
    legacy: Mapping[str, Any],
    sensitivities: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Build a new-schema experiment_cfg from legacy channel_mapping +
    sensitivity_mapping + sensitivities.json values.
    Default analysis_io uses hammer_x/hammer_y as inputs and acc_tool_x / acc_tool_y
    as outputs when those mapping names exist and units resolve; otherwise first N / first m/s² mapping names.
    """
    chan_map = dict(legacy.get("channel_mapping") or {})
    sens_map = dict(legacy.get("sensitivity_mapping") or {})

    sens_key_to_id: Dict[str, str] = {}
    catalog: List[Dict[str, Any]] = []
    for sensor_name, sens_key in sens_map.items():
        sk = str(sens_key)
        if sk not in sensitivities:
            continue
        if sk not in sens_key_to_id:
            sid = f"sens_{len(sens_key_to_id)}"
            sens_key_to_id[sk] = sid
            val = float(sensitivities[sk])
            # Heuristic: F_* keys are hammer sensitivities (N/V); a_* are accelerometers (m/s²/V).
            unit = "N" if sk.upper().startswith("F_") else "m/s²"
            catalog.append(
                {
                    "id": sid,
                    "model": sk,
                    "manufacturer": "",
                    "sensitivity_vs_per_v": val,
                    "unit": unit,
                }
            )

    mapping: List[Dict[str, Any]] = []
    for sensor_name, ch in chan_map.items():
        sk = sens_map.get(sensor_name)
        if sk is None or str(sk) not in sens_key_to_id:
            continue
        mapping.append(
            {
                "name": str(sensor_name),
                "channel": int(ch),
                "sensor_id": sens_key_to_id[str(sk)],
            }
        )

    name_to_unit = build_mapping_name_to_unit({"sensor_catalog": catalog, "mapping": mapping})

    def _first_name(pred) -> str:
        for row in mapping:
            nm = str(row.get("name") or "")
            u = name_to_unit.get(nm, "")
            if pred(nm, u):
                return nm
        return ""

    in_x = _first_name(lambda n, u: n == "hammer_x" and u == "N")
    in_y = _first_name(lambda n, u: n == "hammer_y" and u == "N")
    if not in_x:
        in_x = _first_name(lambda n, u: "hammer" in n.lower() and n.endswith("_x") and u == "N")
    if not in_y:
        in_y = _first_name(lambda n, u: "hammer" in n.lower() and n.endswith("_y") and u == "N")
    if not in_x:
        in_x = _first_name(lambda n, u: u == "N")
    if not in_y:
        in_y = _first_name(lambda n, u: u == "N")

    out_x = _first_name(lambda n, u: n == "acc_tool_x" and u == "m/s²")
    out_y = _first_name(lambda n, u: n == "acc_tool_y" and u == "m/s²")
    if not out_x:
        out_x = _first_name(lambda n, u: n.endswith("_x") and u == "m/s²")
    if not out_y:
        out_y = _first_name(lambda n, u: n.endswith("_y") and u == "m/s²")

    analysis_io: Dict[str, Dict[str, str]] = {}
    if in_x and out_x:
        analysis_io["x"] = {"input": in_x, "output": out_x}
    if in_y and out_y:
        analysis_io["y"] = {"input": in_y, "output": out_y}

    out_cfg = dict(legacy)
    out_cfg["sensor_catalog"] = catalog
    out_cfg["mapping"] = mapping
    out_cfg["expressions"] = list(legacy.get("expressions") or [])
    if not out_cfg["expressions"]:
        out_cfg["expressions"] = []
    out_cfg["analysis_io"] = analysis_io
    # Optional: map simulation canonical keys to likely user names for smoke tests / GUI defaults.
    out_cfg.setdefault(
        "simulation_signal_map",
        {
            "hammer_x": "F_x",
            "hammer_y": "F_y",
            "resp_x": "x_tool",
            "resp_y": "y_tool",
        },
    )
    return out_cfg
