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
        d = directional[key]
        assert d["time"] is not None
        assert "F_x" in d and "F_y" in d
        assert "x_tool" in d and "y_tool" in d

    # Validate semantic: config damping is damping ratio ζ (dimensionless).
    from . import simulate_system

    sim = simulate_system.simulate_two_axis_system(exp_cfg["simulation"])
    assert abs(sim["meta"]["damping_ratio"] - 0.02) < 1e-12


def run() -> None:
    smoke_compute_frf_from_signals()
    smoke_load_simulation_with_config()


if __name__ == "__main__":
    run()
