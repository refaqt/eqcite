from __future__ import annotations

import numpy as np

from . import frf_estimation
from . import load_data


def smoke_compute_frf_from_signals() -> None:
    fs = 8000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.2, dt)

    # Simple synthetic SISO system: y = u convolved with an exponential kernel.
    u = np.zeros_like(t)
    u[len(u) // 10] = 1.0

    tau = 0.01
    kernel_t = np.arange(0.0, 0.05, dt)
    kernel = np.exp(-kernel_t / tau)
    y = np.convolve(u, kernel, mode="same")

    params = frf_estimation.get_system_parameters()
    params["welch_nperseg"] = 2048
    params["welch_noverlap"] = 2000  # intentionally too large; should be clamped

    for et in ["H1", "H2", "Hv"]:
        params2 = dict(params)
        params2["estimator_type"] = et
        f, H = frf_estimation.compute_frf_from_signals(t, u, y, params2)
        assert f.ndim == 1 and H.ndim == 1
        assert len(f) == len(H)
        assert np.isfinite(f).all()
        assert np.isfinite(np.real(H)).all()
        assert np.isfinite(np.imag(H)).all()


def smoke_load_simulation_with_config() -> None:
    exp_cfg = {
        "data_source": "simulation",
        "simulation": {
            "fs": 8000,
            "T": 0.2,
            "model": {"mass": 8.0, "stiffness": 1e6, "damping": 0.02},
            "hammer": {"force": 50.0, "bandwidth_hz": 1000.0},
        },
    }
    sens = {}
    directional = load_data.load_directional_data(exp_cfg, sens)
    assert "x" in directional and "y" in directional
    for key in ("x", "y"):
        lst = directional[key]
        assert isinstance(lst, list) and len(lst) >= 1
        d = lst[0]
        assert d["time"] is not None
        assert "F_x" in d and "F_y" in d
        assert "x_tool" in d and "y_tool" in d

    exp_cfg2 = dict(exp_cfg)
    exp_cfg2["sensor_catalog"] = [
        {"id": "s0", "model": "", "manufacturer": "", "sensitivity_vs_per_v": 1.0, "unit": "m/s²"}
    ]
    exp_cfg2["mapping"] = []
    exp_cfg2["expressions"] = []
    exp_cfg2["simulation_signal_map"] = {
        "hammer_x": "F_x",
        "hammer_y": "F_y",
        "resp_x": "x_tool",
        "resp_y": "y_tool",
    }
    exp_cfg2["analysis_io"] = {
        "x": {"input": "hammer_x", "output": "resp_x"},
        "y": {"input": "hammer_y", "output": "resp_y"},
    }
    directional2 = load_data.load_directional_data(exp_cfg2, {})
    d2 = directional2["x"][0]
    assert "hammer_x" in d2 and "resp_x" in d2

    # Validate semantic: config damping is damping ratio ζ (dimensionless).
    from . import simulate_system

    sim = simulate_system.simulate_two_axis_system(exp_cfg["simulation"])
    assert abs(sim["meta"]["damping_ratio"] - 0.02) < 1e-12


def run() -> None:
    smoke_compute_frf_from_signals()
    smoke_load_simulation_with_config()
    # Minimal sanity: complex-mean averaging differs from magnitude-mean.
    z = np.array([1.0 + 0.0j, -1.0 + 0.0j])
    assert abs(np.mean(z)) < 1e-12
    assert abs(np.mean(np.abs(z)) - 1.0) < 1e-12


if __name__ == "__main__":
    run()
