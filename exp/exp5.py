"""Experiment 5: descriptive Gaussian fits of five trend models under the
declared covariance family (paper Table 1, Appendix H).

Fits constant drift, power law, drift-and-log-age, logistic and Gompertz
trend models to the realized weekly Bitcoin log-price series by Gaussian
maximum likelihood, profiling the deviation model out of the declared
1,216-member covariance grid (and, for logistic/Gompertz, profiling the
nonlinear parameters through the paper's fixed two-stage search). Also runs
the "origin profile" of Appendix H: holding the deviation model fixed at the
value selected by the genesis (tau=0) power-law fit, refits the power law's
intercept and slope for ages shifted by tau in {-104, ..., 104} weeks
(step 8, 27 points) and reports the log-likelihood profile.

Note: the realized sample has 663 anchors, not the paper's 820, because the
paper's exact 2011-2013 daily series is not obtainable from the data path
used here (see data/data_loader.py and results/RESULTS.md); everything else
follows the paper's construction unchanged.
"""
from __future__ import annotations

import functools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Must be set before numpy/OpenBLAS is imported: this script parallelizes the
# grid loop with Python *threads* (numpy's linalg calls release the GIL), but
# if OpenBLAS itself is also multi-threaded, N_WORKERS threads each spawning
# OpenBLAS's own thread pool oversubscribes the CPU and causes severe lock
# contention (observed as a near-hang, 10x+ slower than the serial loop).
# Pinning BLAS to one thread per call lets the outer thread pool be the only
# source of parallelism, which is what actually gives a speed-up here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from data.data_loader import load_primary_weekly_series
from src import config
from src.deviation_family import DeviationModel, build_grid, model_covariance
from src.trend_models import TrendFit, fit_linear_model, gls_profile_loglik

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ORIGIN_TAU_WEEKS = np.arange(-104, 105, 8)  # 27 points, step 8 (paper Appendix H)

# The declared grid has 1,216 members and each covariance evaluation costs an
# O(n_obs^2 * n_full) matrix build + linear solve (tens of ms each), so a
# fully serial sweep over 3 linear models + 2 nonlinear models (each profiling
# the whole grid once) is slow. Everything below only calls the existing
# public primitives (model_covariance, gls_profile_loglik, fit_linear_model)
# but spreads the 1,216-member grid loop over a thread pool: numpy's
# linalg.solve/slogdet release the GIL while they run in OpenBLAS, so plain
# threads (not processes) give real parallel speed-up here without any of the
# pickling / fork-after-threading-deadlock pitfalls of a process pool, and
# without duplicating the (small) data arrays across workers. The statistics
# computed are identical to a serial loop.
N_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _design_matrix(name: str, s: np.ndarray) -> np.ndarray:
    """Design matrix for a linear trend model."""
    if name == "constant_drift":
        return np.column_stack([np.ones_like(s), s])
    if name == "power_law":
        return np.column_stack([np.ones_like(s), np.log(s)])
    if name == "drift_and_log_age":
        return np.column_stack([np.ones_like(s), s, np.log(s)])
    raise ValueError(name)


def _fit_linear_chunk(y: np.ndarray, anchor_weeks: np.ndarray, n_full: int, name: str,
                       s: np.ndarray, chunk_models: list[DeviationModel],
                       n_trend_params: int) -> TrendFit:
    """Run fit_linear_model on one grid chunk, in a worker thread."""
    design_fn = functools.partial(_design_matrix, name, s)
    return fit_linear_model(y, anchor_weeks, n_full, design_fn, chunk_models, n_trend_params, name)


def _parallel_fit_linear(y: np.ndarray, anchor_weeks: np.ndarray, n_full: int, name: str,
                          s: np.ndarray, grid: list[DeviationModel],
                          n_trend_params: int) -> TrendFit:
    """Profile a linear model over the full grid, split across worker threads."""
    idx_chunks = np.array_split(np.arange(len(grid)), N_WORKERS)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_fit_linear_chunk, y, anchor_weeks, n_full, name, s,
                              [grid[i] for i in idxs], n_trend_params)
                   for idxs in idx_chunks]
        results = [f.result() for f in futures]
    best_chunk = int(np.argmax([r.log_likelihood for r in results]))
    best = results[best_chunk]
    best.theta_index = int(idx_chunks[best_chunk][best.theta_index])
    return best


def _nonlinear_design(kind: str, s: np.ndarray, nl_params: np.ndarray) -> np.ndarray:
    """Design matrix for the nonlinear trend models (formulas given in Appendix H)."""
    if kind == "logistic":
        r, s0 = nl_params
        col = -np.log1p(np.exp(-r * (s - s0)))
    else:
        (r,) = nl_params
        col = -np.exp(-r * s)
    return np.column_stack([np.ones_like(s), col])


def _nonlinear_loglik(nl_params: np.ndarray, kind: str, y: np.ndarray, s: np.ndarray,
                      Omega: np.ndarray) -> float:
    X = _nonlinear_design(kind, s, nl_params)
    return gls_profile_loglik(y, X, Omega)[2]


def _profile_nonlinear_chunk(y: np.ndarray, s: np.ndarray, anchor_weeks: np.ndarray, n_full: int,
                              chunk_models: list[DeviationModel], kind: str,
                              nl_params: np.ndarray) -> tuple:
    """Best (log-lik, local index, beta, kappa2) over one grid chunk, fixed nl_params
    (run in a worker thread)."""
    X = _nonlinear_design(kind, s, nl_params)
    best = None
    for local_idx, model in enumerate(chunk_models):
        Omega = model_covariance(model, anchor_weeks, n_full)
        beta, kappa2, loglik = gls_profile_loglik(y, X, Omega)
        if best is None or loglik > best[0]:
            best = (loglik, local_idx, beta, kappa2)
    return best


def _fit_nonlinear_parallel(y: np.ndarray, s: np.ndarray, anchor_weeks: np.ndarray, n_full: int,
                             grid: list[DeviationModel], kind: str,
                             start_grid: list[np.ndarray],
                             bounds: list[tuple[float, float]]) -> TrendFit:
    """Same 3-stage search as src.trend_models.fit_nonlinear_model, with the
    expensive stage-2 grid profiling spread across worker processes."""
    n_trend = 3
    rw_idx = next(i for i, m in enumerate(grid)
                  if m.persistence.kind == "ar" and m.persistence.param == 0.0
                  and m.cycle.r == 0.0 and m.lam == 1.0)
    Omega_rw = model_covariance(grid[rw_idx], anchor_weeks, n_full)

    best_stage1 = None
    for start in start_grid:
        res = minimize(lambda p: -_nonlinear_loglik(p, kind, y, s, Omega_rw),
                        start, method="L-BFGS-B", bounds=bounds)
        if res.success and (best_stage1 is None or -res.fun > best_stage1[1]):
            best_stage1 = (res.x, -res.fun)
    nl_params = best_stage1[0]

    idx_chunks = np.array_split(np.arange(len(grid)), N_WORKERS)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_profile_nonlinear_chunk, y, s, anchor_weeks, n_full,
                              [grid[i] for i in idxs], kind, nl_params)
                   for idxs in idx_chunks]
        chunk_results = [f.result() for f in futures]
    best_chunk = int(np.argmax([r[0] for r in chunk_results]))
    best_loglik, local_idx, best_beta, best_kappa2 = chunk_results[best_chunk]
    best_idx = int(idx_chunks[best_chunk][local_idx])

    # Stage 3: refit nl params at the selected member; keep only if it does not lower L.
    Omega_sel = model_covariance(grid[best_idx], anchor_weeks, n_full)
    res = minimize(lambda p: -_nonlinear_loglik(p, kind, y, s, Omega_sel),
                    nl_params, method="L-BFGS-B", bounds=bounds)
    if res.success and -res.fun > best_loglik:
        nl_params = res.x
        X = _nonlinear_design(kind, s, nl_params)
        best_beta, best_kappa2, best_loglik = gls_profile_loglik(y, X, Omega_sel)

    params = {"r": float(nl_params[0])} if kind == "gompertz" else \
        {"r": float(nl_params[0]), "s0": float(nl_params[1])}
    return TrendFit(kind, params, best_idx, best_loglik, n_trend, best_beta, best_kappa2)


def _logistic_start_grid(s: np.ndarray) -> list[np.ndarray]:
    """24 evenly spaced starting points for (r, s0).

    Not fully specified by the paper beyond "24 evenly spaced starts"; chosen
    here as a representative 4 (log-spaced r) x 6 (quantile-spaced s0) grid.
    """
    r_vals = np.geomspace(1e-3, 3e-1, 4)
    s0_vals = np.quantile(s, np.linspace(0.05, 0.95, 6))
    return [np.array([r, s0]) for r in r_vals for s0 in s0_vals]


def _gompertz_start_grid() -> list[np.ndarray]:
    """24 evenly (log-)spaced starting points for r, spanning a plausible decade."""
    return [np.array([r]) for r in np.geomspace(1e-4, 5e-2, 24)]


def _origin_profile(y: np.ndarray, s: np.ndarray, anchor_weeks: np.ndarray, n_full: int,
                     theta_model) -> list[dict]:
    """Appendix H origin profile: refit power law with shifted age, fixed theta."""
    profile = []
    for tau in ORIGIN_TAU_WEEKS:
        s_shifted = s + tau
        mask = s_shifted > 0  # log(s_shifted) requires positive age
        s_used, y_used, anchor_used = s_shifted[mask], y[mask], anchor_weeks[mask]
        Omega = model_covariance(theta_model, anchor_used, n_full)
        X = np.column_stack([np.ones_like(s_used), np.log(s_used)])
        _, _, loglik = gls_profile_loglik(y_used, X, Omega)
        profile.append({"tau_weeks": int(tau), "log_likelihood": loglik,
                         "n_used": int(mask.sum())})
    baseline = next(p["log_likelihood"] for p in profile if p["tau_weeks"] == 0)
    for p in profile:
        p["delta_loglik"] = p["log_likelihood"] - baseline
    return profile


def _derive_params(fit: TrendFit) -> dict:
    """Map each TrendFit's internal beta/nonlinear params to the reported parameters."""
    if fit.model_name == "constant_drift":
        return {"a": float(fit.beta[0]), "c": float(fit.beta[1])}
    if fit.model_name == "power_law":
        return {"a": float(fit.beta[0]), "b": float(fit.beta[1])}
    if fit.model_name == "drift_and_log_age":
        return {"a": float(fit.beta[0]), "c": float(fit.beta[1]), "b": float(fit.beta[2])}
    if fit.model_name == "logistic":
        return {"K": float(np.exp(fit.beta[0])), "r": float(fit.params["r"]),
                "s0": float(fit.params["s0"])}
    if fit.model_name == "gompertz":
        return {"K": float(np.exp(fit.beta[0])), "beta": float(-fit.beta[1]),
                "r": float(fit.params["r"])}
    raise ValueError(fit.model_name)


def _trend_curve(model_name: str, params: dict, s_grid: np.ndarray) -> np.ndarray:
    if model_name == "constant_drift":
        return params["a"] + params["c"] * s_grid
    if model_name == "power_law":
        return params["a"] + params["b"] * np.log(s_grid)
    if model_name == "drift_and_log_age":
        return params["a"] + params["c"] * s_grid + params["b"] * np.log(s_grid)
    if model_name == "logistic":
        return np.log(params["K"]) - np.log1p(np.exp(-params["r"] * (s_grid - params["s0"])))
    if model_name == "gompertz":
        return np.log(params["K"]) - params["beta"] * np.exp(-params["r"] * s_grid)
    raise ValueError(model_name)


def _plot_origin_profile(origin_profile: list[dict], peak: dict) -> None:
    taus = [p["tau_weeks"] for p in origin_profile]
    deltas = [p["delta_loglik"] for p in origin_profile]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(taus, deltas, marker="o")
    ax.axvline(peak["tau_weeks"], color="red", linestyle="--",
               label=f"peak tau = {peak['tau_weeks']} weeks")
    ax.set_xlabel("Origin shift tau (weeks)")
    ax.set_ylabel("Delta log-likelihood vs. tau=0")
    ax.set_title("Power-law origin profile (Appendix H)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_origin_profile.png", dpi=150)
    plt.close(fig)


def _plot_fitted_trends(s: np.ndarray, y: np.ndarray, model_results: list[dict]) -> None:
    s_grid = np.linspace(float(s.min()), float(s.max()), 500)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(s, y, s=6, color="gray", alpha=0.5, label="observed log price")
    for res in model_results:
        curve = _trend_curve(res["model_name"], res["params"], s_grid)
        ax.plot(s_grid, curve, label=res["model_name"])
    ax.set_xlabel("Age (weeks since genesis)")
    ax.set_ylabel("log price")
    ax.set_title("Fitted trend models (Table 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_fitted_trends.png", dpi=150)
    plt.close(fig)


def run() -> dict:
    t0 = time.time()
    df = load_primary_weekly_series(
        config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END,
        dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,),
    )
    n_obs = len(df)
    s = df["age_weeks"].to_numpy()
    y = df["y"].to_numpy()
    anchor_weeks = np.round(s).astype(int)
    n_full = int(anchor_weeks.max()) + 1

    grid = build_grid()

    fits: dict[str, TrendFit] = {}
    fits["constant_drift"] = _parallel_fit_linear(
        y, anchor_weeks, n_full, "constant_drift", s, grid, 2)
    print(f"constant_drift done ({time.time() - t0:.1f}s)", flush=True)
    fits["power_law"] = _parallel_fit_linear(
        y, anchor_weeks, n_full, "power_law", s, grid, 2)
    print(f"power_law done ({time.time() - t0:.1f}s)", flush=True)
    fits["drift_and_log_age"] = _parallel_fit_linear(
        y, anchor_weeks, n_full, "drift_and_log_age", s, grid, 3)
    print(f"drift_and_log_age done ({time.time() - t0:.1f}s)", flush=True)

    fits["logistic"] = _fit_nonlinear_parallel(
        y, s, anchor_weeks, n_full, grid, "logistic",
        start_grid=_logistic_start_grid(s), bounds=[(1e-4, 1.0), (0.0, float(s.max()) * 2)])
    print(f"logistic done ({time.time() - t0:.1f}s)", flush=True)
    fits["gompertz"] = _fit_nonlinear_parallel(
        y, s, anchor_weeks, n_full, grid, "gompertz",
        start_grid=_gompertz_start_grid(), bounds=[(1e-5, 1e-1)])
    print(f"gompertz done ({time.time() - t0:.1f}s)", flush=True)

    powerlaw_loglik = fits["power_law"].log_likelihood
    model_results = []
    for name in config.TREND_MODEL_NAMES:
        fit = fits[name]
        model_results.append({
            "model_name": name,
            "params": _derive_params(fit),
            "log_likelihood": fit.log_likelihood,
            "delta_loglik_vs_powerlaw": fit.log_likelihood - powerlaw_loglik,
            "theta_index": fit.theta_index,
            "n_trend_params": fit.n_trend_params,
        })

    theta_model = grid[fits["power_law"].theta_index]
    origin_profile = _origin_profile(y, s, anchor_weeks, n_full, theta_model)
    peak = max(origin_profile, key=lambda p: p["log_likelihood"])

    elapsed = time.time() - t0
    out = {
        "n_observations": n_obs,
        "n_grid_models": len(grid),
        "models": model_results,
        "origin_profile": origin_profile,
        "origin_profile_peak_tau_weeks": peak["tau_weeks"],
        "origin_profile_peak_delta_loglik": peak["delta_loglik"],
        "elapsed_seconds": elapsed,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp5_trend_fits.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    _plot_origin_profile(origin_profile, peak)
    _plot_fitted_trends(s, y, model_results)
    return out


if __name__ == "__main__":
    run()
