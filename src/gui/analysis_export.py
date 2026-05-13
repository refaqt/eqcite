from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def slugify_measurement_key(label: str, used: set[str]) -> str:
  base = re.sub(r"[^a-zA-Z0-9._-]+", "_", (label or "").strip())
  base = re.sub(r"_+", "_", base).strip("_").lower() or "measurement"
  key = base
  n = 2
  while key in used:
    key = f"{base}_{n}"
    n += 1
  used.add(key)
  return key


def measurement_keys_for_paths(filepaths: Sequence[Optional[str]]) -> List[str]:
  used: set[str] = set()
  keys: List[str] = []
  for fp in filepaths:
    label = Path(str(fp or "")).stem if fp else "measurement"
    keys.append(slugify_measurement_key(label, used))
  return keys


def _phase_deg(H: np.ndarray) -> np.ndarray:
  return np.unwrap(np.angle(np.asarray(H, dtype=complex))) * 180.0 / np.pi


def _write_time_csv(
  path: Path,
  *,
  time_s: np.ndarray,
  force_raw: np.ndarray,
  response_raw: np.ndarray,
  u_steps: Sequence[Tuple[str, np.ndarray]],
  resp_steps: Sequence[Tuple[str, np.ndarray]],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  header = ["time_s", "force_raw", "response_raw"]
  for name, _ in u_steps:
    if name == "raw":
      continue
    header.append(f"force_{name}")
  for name, _ in resp_steps:
    if name == "raw":
      continue
    header.append(f"response_{name}")

  with path.open("w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow(header)
    n = len(time_s)
    for i in range(n):
      row: List[Any] = [float(time_s[i]), float(force_raw[i]), float(response_raw[i])]
      for name, arr in u_steps:
        if name == "raw":
          continue
        row.append(float(arr[i]))
      for name, arr in resp_steps:
        if name == "raw":
          continue
        row.append(float(arr[i]))
      w.writerow(row)


def _write_frequency_csv(
  path: Path,
  *,
  f_hz: np.ndarray,
  coherence: np.ndarray,
  compliance: np.ndarray,
  accelerance: np.ndarray,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  f_hz = np.asarray(f_hz, dtype=float)
  coh = np.asarray(coherence, dtype=float)
  Hc = np.asarray(compliance, dtype=complex)
  Ha = np.asarray(accelerance, dtype=complex)
  header = [
    "frequency_hz",
    "coherence",
    "bode_displacement_real",
    "bode_displacement_imag",
    "bode_displacement_magnitude",
    "bode_displacement_phase_deg",
    "bode_acceleration_real",
    "bode_acceleration_imag",
    "bode_acceleration_magnitude",
    "bode_acceleration_phase_deg",
  ]
  with path.open("w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow(header)
    for i in range(len(f_hz)):
      w.writerow(
        [
          float(f_hz[i]),
          float(coh[i]),
          float(np.real(Hc[i])),
          float(np.imag(Hc[i])),
          float(np.abs(Hc[i])),
          float(_phase_deg(Hc)[i]),
          float(np.real(Ha[i])),
          float(np.imag(Ha[i])),
          float(np.abs(Ha[i])),
          float(_phase_deg(Ha)[i]),
        ]
      )


def _save_line_figure(path: Path, *, title: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(x, y, linewidth=1.2)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(True, alpha=0.3)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def _save_coherence_figure(path: Path, *, f_hz: np.ndarray, coherence: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(8, 4))
  mask = f_hz > 0
  ax.semilogx(f_hz[mask], coherence[mask], linewidth=1.6)
  ax.set_title("Coherence")
  ax.set_xlabel("Frequency (Hz)")
  ax.set_ylabel("γ²")
  ax.set_ylim(0.0, 1.05)
  ax.grid(True, which="both", alpha=0.25)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def _save_bode_figure(path: Path, *, f_hz: np.ndarray, H: np.ndarray, title: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
  f = np.asarray(f_hz, dtype=float)
  Hc = np.asarray(H, dtype=complex)
  mask = np.isfinite(f) & (f > 0)
  f2 = f[mask]
  H2 = Hc[mask]
  ax_mag.loglog(f2, np.abs(H2), linewidth=1.6)
  ax_mag.set_ylabel("Magnitude")
  ax_mag.set_title(title)
  ax_mag.grid(True, which="both", alpha=0.3)
  ax_ph.semilogx(f2, _phase_deg(H2), linewidth=1.2)
  ax_ph.set_xlabel("Frequency (Hz)")
  ax_ph.set_ylabel("Phase (deg)")
  ax_ph.grid(True, which="both", alpha=0.3)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)


def _steps_as_tuples(steps: Sequence[Any]) -> List[Tuple[str, np.ndarray]]:
  out: List[Tuple[str, np.ndarray]] = []
  for step in steps:
    name = str(getattr(step, "name", "") or "")
    y = np.asarray(getattr(step, "y", []), dtype=float)
    out.append((name, y))
  return out


def _export_measurement_artifacts(
  folder: Path,
  *,
  direction: str,
  key: str,
  measurement: Any,
) -> Dict[str, Any]:
  proc = measurement.proc
  time_s = np.asarray(proc.time, dtype=float)
  u_steps = _steps_as_tuples(proc.u_steps)
  r_steps = _steps_as_tuples(proc.resp_steps)
  force_raw = u_steps[0][1] if u_steps else np.zeros_like(time_s)
  response_raw = r_steps[0][1] if r_steps else np.zeros_like(time_s)

  rel_time = f"results/tables/time/{direction}/{key}.csv"
  rel_freq = f"results/tables/frequency/{direction}/{key}.csv"
  time_path = folder / rel_time
  freq_path = folder / rel_freq
  _write_time_csv(
    time_path,
    time_s=time_s,
    force_raw=force_raw,
    response_raw=response_raw,
    u_steps=u_steps,
    resp_steps=r_steps,
  )
  f_hz, Hc = measurement.compliance
  _, Ha = measurement.accelerance
  _write_frequency_csv(
    freq_path,
    f_hz=np.asarray(f_hz, dtype=float),
    coherence=np.asarray(proc.coherence_uy, dtype=float),
    compliance=np.asarray(Hc, dtype=complex),
    accelerance=np.asarray(Ha, dtype=complex),
  )

  fig_root = folder / "results" / "figures" / "time" / direction / key
  _save_line_figure(fig_root / "force.png", title="Force", x=time_s, y=force_raw, xlabel="Time (s)", ylabel="Force")
  _save_line_figure(
    fig_root / "response.png",
    title="Response",
    x=time_s,
    y=response_raw,
    xlabel="Time (s)",
    ylabel="Response",
  )
  for name, arr in u_steps:
    if name == "raw":
      continue
    _save_line_figure(
      fig_root / f"force_{name}.png",
      title=f"Force ({name})",
      x=time_s,
      y=arr,
      xlabel="Time (s)",
      ylabel="Force",
    )
  for name, arr in r_steps:
    if name == "raw":
      continue
    _save_line_figure(
      fig_root / f"response_{name}.png",
      title=f"Response ({name})",
      x=time_s,
      y=arr,
      xlabel="Time (s)",
      ylabel="Response",
    )

  freq_fig_root = folder / "results" / "figures" / "frequency" / direction / key
  _save_coherence_figure(freq_fig_root / "coherence.png", f_hz=np.asarray(f_hz), coherence=np.asarray(proc.coherence_uy))
  _save_bode_figure(
    freq_fig_root / "bode_displacement_force.png",
    f_hz=np.asarray(f_hz),
    H=np.asarray(Hc, dtype=complex),
    title="Bode (displacement/force)",
  )
  _save_bode_figure(
    freq_fig_root / "bode_acceleration_force.png",
    f_hz=np.asarray(f_hz),
    H=np.asarray(Ha, dtype=complex),
    title="Bode (acceleration/force)",
  )

  return {
    "direction": direction,
    "key": key,
    "filepath": measurement.filepath,
    "time_csv": rel_time,
    "frequency_csv": rel_freq,
    "time_figures": str(fig_root.relative_to(folder)).replace("\\", "/"),
    "frequency_figures": str(freq_fig_root.relative_to(folder)).replace("\\", "/"),
  }


def _avg_complex_frf(items: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
  if not items:
    return None
  f0 = np.asarray(items[0][0], dtype=float)
  Hs = [np.asarray(it[1], dtype=complex) for it in items]
  return f0, np.mean(np.stack(Hs, axis=0), axis=0)


def _avg_coherence(procs: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
  f0 = np.asarray(procs[0].f_hz, dtype=float)
  Puu = np.mean(np.stack([np.asarray(p.psd_u, dtype=float) for p in procs], axis=0), axis=0)
  Prr = np.mean(np.stack([np.asarray(p.psd_r, dtype=float) for p in procs], axis=0), axis=0)
  Pur = np.mean(np.stack([np.asarray(p.csd_ur, dtype=complex) for p in procs], axis=0), axis=0)
  coh = (np.abs(Pur) ** 2) / ((Puu * Prr) + 1e-30)
  return f0, np.clip(coh, 0.0, 1.0)


def _avg_time_steps(procs: Sequence[Any]) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
  time_s = np.asarray(procs[0].time, dtype=float)
  u_names = [s.name for s in procs[0].u_steps]
  r_names = [s.name for s in procs[0].resp_steps]
  u_steps: List[Tuple[str, np.ndarray]] = []
  r_steps: List[Tuple[str, np.ndarray]] = []
  for name in u_names:
    stack = np.stack([np.asarray(next(s for s in p.u_steps if s.name == name).y, dtype=float) for p in procs], axis=0)
    u_steps.append((name, np.mean(stack, axis=0)))
  for name in r_names:
    stack = np.stack([np.asarray(next(s for s in p.resp_steps if s.name == name).y, dtype=float) for p in procs], axis=0)
    r_steps.append((name, np.mean(stack, axis=0)))
  return time_s, u_steps, r_steps


def _export_average_artifacts(folder: Path, *, direction: str, measurements: Sequence[Any]) -> Optional[Dict[str, Any]]:
  if not measurements:
    return None
  procs = [m.proc for m in measurements]
  time_s, u_steps, r_steps = _avg_time_steps(procs)
  force_raw = u_steps[0][1] if u_steps else np.zeros_like(time_s)
  response_raw = r_steps[0][1] if r_steps else np.zeros_like(time_s)
  comp = _avg_complex_frf([m.compliance for m in measurements])
  acc = _avg_complex_frf([m.accelerance for m in measurements])
  if comp is None or acc is None:
    return None
  f_hz, Hc = comp
  _, Ha = acc
  f_coh, coh = _avg_coherence(procs)

  key = "average"
  rel_time = f"results/tables/time/{direction}/{key}.csv"
  rel_freq = f"results/tables/frequency/{direction}/{key}.csv"
  _write_time_csv(
    folder / rel_time,
    time_s=time_s,
    force_raw=force_raw,
    response_raw=response_raw,
    u_steps=u_steps,
    resp_steps=r_steps,
  )
  _write_frequency_csv(
    folder / rel_freq,
    f_hz=np.asarray(f_hz, dtype=float),
    coherence=np.interp(np.asarray(f_hz), f_coh, coh),
    compliance=np.asarray(Hc, dtype=complex),
    accelerance=np.asarray(Ha, dtype=complex),
  )

  fig_root = folder / "results" / "figures" / "time" / direction / key
  _save_line_figure(fig_root / "force.png", title="Force (average)", x=time_s, y=force_raw, xlabel="Time (s)", ylabel="Force")
  _save_line_figure(
    fig_root / "response.png",
    title="Response (average)",
    x=time_s,
    y=response_raw,
    xlabel="Time (s)",
    ylabel="Response",
  )
  for name, arr in u_steps:
    if name == "raw":
      continue
    _save_line_figure(fig_root / f"force_{name}.png", title=f"Force ({name}, average)", x=time_s, y=arr, xlabel="Time (s)", ylabel="Force")
  for name, arr in r_steps:
    if name == "raw":
      continue
    _save_line_figure(
      fig_root / f"response_{name}.png",
      title=f"Response ({name}, average)",
      x=time_s,
      y=arr,
      xlabel="Time (s)",
      ylabel="Response",
    )

  freq_fig_root = folder / "results" / "figures" / "frequency" / direction / key
  _save_coherence_figure(freq_fig_root / "coherence.png", f_hz=np.asarray(f_hz), coherence=np.interp(np.asarray(f_hz), f_coh, coh))
  _save_bode_figure(freq_fig_root / "bode_displacement_force.png", f_hz=np.asarray(f_hz), H=np.asarray(Hc, dtype=complex), title="Bode (displacement/force, average)")
  _save_bode_figure(freq_fig_root / "bode_acceleration_force.png", f_hz=np.asarray(f_hz), H=np.asarray(Ha, dtype=complex), title="Bode (acceleration/force, average)")

  return {
    "direction": direction,
    "key": key,
    "time_csv": rel_time,
    "frequency_csv": rel_freq,
    "time_figures": str(fig_root.relative_to(folder)).replace("\\", "/"),
    "frequency_figures": str(freq_fig_root.relative_to(folder)).replace("\\", "/"),
  }


def export_frf_results(folder: Path, *, x_measurements: Sequence[Any], y_measurements: Sequence[Any], meta: Dict[str, Any]) -> Dict[str, Any]:
  folder = Path(folder).resolve()
  manifest: Dict[str, Any] = {"analysis_type": "frf", "x": [], "y": [], "averages": []}
  warnings: List[str] = []

  for direction, measurements in (("x", x_measurements), ("y", y_measurements)):
    keys = measurement_keys_for_paths([getattr(m, "filepath", None) for m in measurements])
    for key, m in zip(keys, measurements):
      try:
        manifest[direction].append(_export_measurement_artifacts(folder, direction=direction, key=key, measurement=m))
      except Exception as e:
        warnings.append(f"Failed to export {direction} measurement: {e}")
    try:
      avg_entry = _export_average_artifacts(folder, direction=direction, measurements=measurements)
      if avg_entry is not None:
        manifest["averages"].append(avg_entry)
    except Exception as e:
      warnings.append(f"Failed to export {direction} average: {e}")

  log_path = folder / "results" / "logs" / "export_log.json"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  log_path.write_text(
    json.dumps(
      {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "meta": dict(meta or {}),
        "manifest": manifest,
        "warnings": warnings,
      },
      indent=2,
    ),
    encoding="utf-8",
  )
  manifest["warnings"] = warnings
  return manifest


def _read_frequency_csv(path: Path) -> Dict[str, np.ndarray]:
  with path.open("r", encoding="utf-8", newline="") as fp:
    rows = list(csv.reader(fp))
  if not rows:
    raise ValueError(f"Empty frequency CSV: {path}")
  header = rows[0]
  data = np.array([[float(x) for x in row] for row in rows[1:]], dtype=float)
  out: Dict[str, np.ndarray] = {}
  for i, name in enumerate(header):
    out[name] = data[:, i]
  return out


def _read_time_csv(path: Path) -> Dict[str, np.ndarray]:
  with path.open("r", encoding="utf-8", newline="") as fp:
    rows = list(csv.reader(fp))
  if not rows:
    raise ValueError(f"Empty time CSV: {path}")
  header = rows[0]
  data = np.array([[float(x) for x in row] for row in rows[1:]], dtype=float)
  out: Dict[str, np.ndarray] = {}
  for i, name in enumerate(header):
    out[name] = data[:, i]
  return out


def _hydrate_measurement_from_csv(
  folder: Path,
  entry: Dict[str, Any],
  *,
  direction: str,
) -> Dict[str, Any]:
  time_tbl = _read_time_csv(folder / str(entry["time_csv"]))
  freq_tbl = _read_frequency_csv(folder / str(entry["frequency_csv"]))
  time_s = time_tbl["time_s"]
  u_steps = [("raw", time_tbl["force_raw"])]
  r_steps = [("raw", time_tbl["response_raw"])]
  for col, arr in time_tbl.items():
    if col.startswith("force_") and col != "force_raw":
      u_steps.append((col[len("force_") :], arr))
    if col.startswith("response_") and col != "response_raw":
      r_steps.append((col[len("response_") :], arr))

  f_hz = freq_tbl["frequency_hz"]
  Hc = freq_tbl["bode_displacement_real"] + 1j * freq_tbl["bode_displacement_imag"]
  Ha = freq_tbl["bode_acceleration_real"] + 1j * freq_tbl["bode_acceleration_imag"]
  return {
    "direction": direction,
    "filepath": entry.get("filepath"),
    "key": entry.get("key"),
    "time_s": time_s,
    "u_steps": u_steps,
    "resp_steps": r_steps,
    "f_hz": f_hz,
    "coherence": freq_tbl["coherence"],
    "compliance": (f_hz, Hc),
    "accelerance": (f_hz, Ha),
    "psd_u": np.zeros_like(f_hz),
    "psd_r": np.zeros_like(f_hz),
    "csd_ur": np.zeros_like(f_hz, dtype=complex),
    "fft_f_hz": np.fft.rfftfreq(len(time_s), d=(time_s[1] - time_s[0]) if len(time_s) > 1 else 1.0),
    "fft_u_mag": {},
    "fft_resp_mag": {},
    "fft_u_complex": {},
    "fft_resp_complex": {},
    "io_input_name": "",
    "io_output_name": "",
  }


def import_frf_results(folder: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
  folder = Path(folder).resolve()
  out: Dict[str, Any] = {"x": [], "y": [], "averages": []}
  for direction in ("x", "y"):
    for entry in manifest.get(direction, []) or []:
      if not isinstance(entry, dict):
        continue
      out[direction].append(_hydrate_measurement_from_csv(folder, entry, direction=direction))
  for entry in manifest.get("averages", []) or []:
    if not isinstance(entry, dict):
      continue
    direction = str(entry.get("direction") or "x")
    out["averages"].append(_hydrate_measurement_from_csv(folder, entry, direction=direction))
  return out


def export_stability_results(folder: Path, *, result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
  folder = Path(folder).resolve()
  tab_dir = folder / "results" / "tables" / "stability"
  fig_dir = folder / "results" / "figures" / "stability"
  tab_dir.mkdir(parents=True, exist_ok=True)
  fig_dir.mkdir(parents=True, exist_ok=True)

  manifest: Dict[str, Any] = {"analysis_type": "stability_measurement", "tables": [], "figures": []}
  numeric_keys = [
    "time_centers_std_x",
    "time_centers_std_y",
    "rolling_std_2x_x",
    "rolling_std_2x_y",
    "milling_start_time",
    "milling_end_time",
    "N",
  ]
  for key in numeric_keys:
    if key not in result:
      continue
    rel = f"results/tables/stability/{key}.csv"
    arr = np.asarray(result[key])
    with (folder / rel).open("w", newline="", encoding="utf-8") as fp:
      w = csv.writer(fp)
      w.writerow([key])
      for v in arr.reshape(-1):
        w.writerow([float(v)])
    manifest["tables"].append(rel)

  for axis, sft_key, sxx_key in (("x", "SFT_x", "Sxx_x"), ("y", "SFT_y", "Sxx_y")):
    if sft_key in result and sxx_key in result:
      rel = f"results/tables/stability/spectrogram_{axis}.csv"
      SFT = np.asarray(result[sft_key])
      Sxx = np.asarray(result[sxx_key])
      with (folder / rel).open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["time", "frequency_hz", "magnitude"])
        for i in range(SFT.shape[0]):
          for j in range(SFT.shape[1]):
            w.writerow([float(SFT[i, j]), float(j), float(Sxx[i, j])])
      manifest["tables"].append(rel)

  log_path = folder / "results" / "logs" / "export_log.json"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  log_path.write_text(
    json.dumps(
      {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "meta": dict(meta or {}),
        "manifest": manifest,
      },
      indent=2,
    ),
    encoding="utf-8",
  )
  return manifest


def import_stability_results(folder: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
  folder = Path(folder).resolve()
  out: Dict[str, Any] = {}
  for rel in manifest.get("tables", []) or []:
    path = folder / str(rel)
    if not path.exists():
      continue
    key = path.stem
    with path.open("r", encoding="utf-8", newline="") as fp:
      rows = list(csv.reader(fp))
    if not rows:
      continue
    if key.startswith("spectrogram_"):
      continue
    vals = [float(r[0]) for r in rows[1:]]
    out[key] = np.asarray(vals)
  return out
