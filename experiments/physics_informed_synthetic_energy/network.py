"""Vectorized balanced radial AC power flow, independently checked with pandapower.

Five-bus illustrative feeder, not a named standard benchmark. All impedances,
powers and current ratings are per unit on 1 MVA / 20 kV three-phase bases.
No shunts, taps or voltage-controlled generation. Bus 0 is the slack.
"""
from __future__ import annotations

import numpy as np

PARENTS = np.array([0, 1, 1, 3])
Z = np.array([0.03 + 0.02j, 0.04 + 0.025j, 0.035 + 0.025j, 0.04 + 0.03j])
RATINGS = np.array([0.95, 0.70, 0.60, 0.45])  # current p.u., thermal limits
ANCHOR = np.array([0.10, 0.10, 0.10, 0.10, 0.0])  # four loads and zero solar at bus 4, MW
Q_RATIO = 0.25  # fixed reactive consumption, MVAr/MW; solar unity power factor


def admittance() -> np.ndarray:
    y = np.zeros((5, 5), complex)
    for child, (parent, z) in enumerate(zip(PARENTS, Z, strict=True), 1):
        y[parent, parent] += 1 / z
        y[child, child] += 1 / z
        y[parent, child] -= 1 / z
        y[child, parent] -= 1 / z
    return y


def solve(x: np.ndarray, max_iter: int = 100, tolerance: float = 1e-10) -> dict:
    """Solve independent hourly states of arbitrary batch shape (..., 5).

    Residual uses complex nodal injections, independently of the fixed-point
    iteration stopping test. Nonconvergence is retained and fails feasibility.
    """
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 5 or not np.isfinite(x).all():
        raise ValueError("Expected finite four-load plus solar vectors")
    shape = x.shape[:-1]
    flat = x.reshape(-1, 5)
    s = (flat[:, :4] * (1 + 1j * Q_RATIO)).astype(complex)
    s[:, 3] -= flat[:, 4]
    v = np.ones((len(flat), 5), complex)
    for _ in range(max_iter):
        safe = np.where(abs(v[:, 1:]) > 1e-8, v[:, 1:], 1e-8 + 0j)
        branch = np.conj(s / safe)
        for child in range(4, 0, -1):
            parent = PARENTS[child - 1]
            if parent:
                branch[:, parent - 1] += branch[:, child - 1]
        new = np.ones_like(v)
        for child in range(1, 5):
            new[:, child] = new[:, PARENTS[child - 1]] - Z[child - 1] * branch[:, child - 1]
        delta = abs(new - v).max(axis=1)
        v = new
        if np.all(delta < tolerance):
            break
    branch = (v[:, PARENTS] - v[:, 1:]) / Z
    nodal = v * np.conj(v @ admittance().T)
    residual = abs(nodal[:, 1:] + s).max(axis=1)
    converged = (delta < tolerance) & (residual < 1e-8) & np.isfinite(residual)
    voltage_bad = (abs(v).min(axis=1) < 0.95) | (abs(v).max(axis=1) > 1.05)
    thermal_bad = (abs(branch) / RATINGS).max(axis=1) > 1
    bounds_bad = (flat[:, :4] < 0).any(axis=1) | (flat[:, :4] > 2).any(axis=1)
    bounds_bad |= (flat[:, 4] < 0) | (flat[:, 4] > 1)
    # Substation import/export capability: explicit slack P and Q bounds.
    slack_bad = (abs(nodal[:, 0].real) > 1.5) | (abs(nodal[:, 0].imag) > 0.75)
    feasible = converged & ~voltage_bad & ~thermal_bad & ~bounds_bad & ~slack_bad
    out = {"voltage": v, "line_current": abs(branch), "line_loading": abs(branch) / RATINGS,
           "slack_p": nodal[:, 0].real, "slack_q": nodal[:, 0].imag,
           "residual": residual, "converged": converged, "voltage_bad": voltage_bad,
           "thermal_bad": thermal_bad, "bounds_bad": bounds_bad, "slack_bad": slack_bad,
           "feasible": feasible}
    return {key: val.reshape(shape + val.shape[1:]) for key, val in out.items()}


def project(x: np.ndarray, steps: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Box clipping then feasible contraction of each whole window to an anchor.

    One contraction coefficient per trajectory preserves more temporal shape
    than independent hour clipping. Bisection keeps a verified feasible lower
    endpoint; it does not claim a globally nearest-point projection.
    """
    x = np.asarray(x, float)
    if x.ndim != 3 or x.shape[-1] != 5 or not np.isfinite(x).all():
        raise ValueError("Expected finite (trajectory, hour, 5) windows")
    clipped = np.clip(x, 0, np.array([2, 2, 2, 2, 1]))
    good = solve(clipped)["feasible"].all(axis=1)
    low = np.where(good, 1.0, 0.0)
    high = np.ones(len(x))
    if not solve(ANCHOR)["feasible"]:
        raise RuntimeError("Projection anchor is infeasible")
    for _ in range(steps):
        mid = (low + high) / 2
        candidate = ANCHOR + mid[:, None, None] * (clipped - ANCHOR)
        ok = solve(candidate)["feasible"].all(axis=1)
        low = np.where(ok, mid, low)
        high = np.where(ok, high, mid)
    corrected = ANCHOR + low[:, None, None] * (clipped - ANCHOR)
    if not solve(corrected)["feasible"].all():
        raise RuntimeError("Final projection did not pass constraints")
    return corrected, low


def pandapower_check(x: np.ndarray) -> dict:
    """Independent established solver comparison on supplied operating points."""
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=1.0)
    for _ in range(5):
        pp.create_bus(net, vn_kv=20)
    pp.create_ext_grid(net, bus=0, vm_pu=1)
    for child, (parent, z, rating) in enumerate(zip(PARENTS, Z, RATINGS, strict=True), 1):
        pp.create_line_from_parameters(net, int(parent), child, length_km=1,
                                       r_ohm_per_km=z.real * 400, x_ohm_per_km=z.imag * 400,
                                       c_nf_per_km=0, max_i_ka=rating / (np.sqrt(3) * 20))
        pp.create_load(net, child, p_mw=0, q_mvar=0)
    pp.create_sgen(net, bus=4, p_mw=0, q_mvar=0)
    errors_v, errors_i, residuals = [], [], []
    flat = np.asarray(x).reshape(-1, 5)
    own = solve(flat)
    failures = 0
    for k, point in enumerate(flat):
        net.load.p_mw = point[:4]
        net.load.q_mvar = point[:4] * Q_RATIO
        net.sgen.p_mw = point[4]
        try:
            pp.runpp(net, algorithm="nr", numba=False, init="flat", tolerance_mva=1e-10, max_iteration=50)
        except pp.LoadflowNotConverged:
            failures += 1
            continue
        errors_v.append(float(np.max(abs(net.res_bus.vm_pu.to_numpy() - abs(own["voltage"][k])))))
        errors_i.append(float(np.max(abs(net.res_line.loading_percent.to_numpy() / 100 - own["line_loading"][k]))))
        residuals.append(float(own["residual"][k]))
    if failures or not errors_v or max(errors_v) > 1e-7 or max(errors_i) > 1e-7:
        raise AssertionError(f"Independent solver disagreement: {failures=}, {errors_v=}")
    return {"points_checked": len(flat), "failures": failures, "max_voltage_error_pu": max(errors_v),
            "max_line_loading_error_fraction": max(errors_i), "max_nodal_residual_mva": max(residuals),
            "pandapower_version": pp.__version__}
