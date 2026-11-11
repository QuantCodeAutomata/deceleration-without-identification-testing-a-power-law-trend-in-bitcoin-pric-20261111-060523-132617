"""Experiment 6: prospective forecasts (paper Section "Prospective forecasts",
Appendix H).

Produces pre-registered predictive densities for Bitcoin's closing price on
the last Saturday of each of 2027, 2028, 2029 and 2030, under the five trend
models fitted in experiment 5 (constant drift, power law, drift-and-log-age,
logistic, Gompertz). Each model's forecast is an equal-weight mixture of
Gaussian predictive densities: one centred at the model's fitted parameters,
plus a parametric bootstrap of refits in which the deviation-model grid
search is restricted to the fitted grid member and its immediate neighbours
("one step in exactly one coordinate" of the persistence/cycle/volatility-
shape grid, paper Appendix H). Future volatility is held flat at its 2026
level (`config.FUTURE_VOLATILITY_FLAT`), which enters only through the
declared deviation family's covariance, not through any new parameter here.

This module reuses the fitting procedure of experiment 5 verbatim (same
design matrices, same two-stage nonlinear search, same declared 1,216-member
grid) by importing `exp.exp5` rather than duplicating it, so the five point
estimates are guaranteed identical to what experiment 5 itself would report
on the same data.

Custom implementation; the conditional (future | observed) Gaussian formulas
are the standard multivariate-normal partitioned-covariance identities
applied to the declared deviation family's exact covariance (paper Appendix
B), not provided by any general-purpose forecasting library.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

# Must be set before numpy/OpenBLAS is imported, for the same reason as
# exp_5: this script's own loops call numpy linalg many times in sequence
# (not across worker threads here, since the per-draw cost is already small),
# but an oversubscribed multi-threaded BLAS still slows down a long sequence
# of small linalg calls through repeated thread-pool setup/teardown.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data.data_loader import GENESIS_TIMESTAMP, load_primary_weekly_series
from src import config
from src.deviation_family import DeviationModel, build_grid, model_covariance, simulate_path
from src.trend_models import (TrendFit, _gompertz_design, _logistic_design,
                               fit_linear_model, gls_profile_loglik)

import exp.exp5 as exp5

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Same bounds as experiment 5's nonlinear fits (Appendix H); duplicated here
# (rather than re-deriving from exp5's start-grid helpers) because the
# bootstrap refit only needs the bounds, not the 24-point start grid.
LOGISTIC_BOUNDS = [(1e-4, 1.0), (0.0, None)]  # upper bound on s0 filled in from s.max() at call time
GOMPERTZ_BOUNDS = [(1e-5, 1e-1)]

MC_DRAWS = 20_000  # Monte Carlo draws from the mixture, per model (paper's reporting: median, 90% interval)
MC_SEED = 20261111  # fixed seed for the Monte Carlo mixture sampling step (development seed)
BOOTSTRAP_SEED = 20261110  # fixed seed for the parametric bootstrap simulations (development seed)

# Soft wall-clock budget for the whole bootstrap + forecasting stage (after the
# five point-estimate fits, which alone cost several minutes: see exp5's own
# reported `elapsed_seconds`). A single calibration draw is timed separately
# for a linear and a nonlinear model (their per-draw costs differ by roughly
# an order of magnitude), and if the full 5-model sweep at the requested
# count would blow this budget, the bootstrap count actually used is reduced
# below `config.FORECAST_N_BOOTSTRAP_RUN` and the true count is recorded in
# the output JSON's "n_bootstrap_used" field.
TIME_BUDGET_SECONDS = 200.0


def _design_matrix_for(model_name: str, s_arr: np.ndarray, fit: TrendFit) -> np.ndarray:
    """Design matrix for `model_name` at ages `s_arr`, using the nonlinear
    parameters carried by `fit` for logistic/Gompertz (paper Table 1)."""
    if model_name == "logistic":
        return _logistic_design(s_arr, fit.params["r"], fit.params["s0"])
    if model_name == "gompertz":
        return _gompertz_design(s_arr, fit.params["r"])
    return exp5._design_matrix(model_name, s_arr)


def _grid_axis_sizes() -> tuple[int, int, int]:
    """Sizes of the three independent axes `build_grid()` is the product of
    (persistence x cycle x volatility-shape), in the same enumeration order
    used by `build_grid()` (src/deviation_family.py)."""
    n_persistence = (len(config.AR_C_VALUES) + len(config.FRACTIONAL_D_VALUES)
                      + len(config.LOCAL_LEVEL_G_VALUES))
    n_cycle = 1 + (len(config.CYCLE_R_VALUES) - 1) * len(config.CYCLE_PERIOD_YEARS) * len(config.CYCLE_KAPPA_VALUES)
    n_lam = len(config.VOLATILITY_LAMBDA_VALUES)
    return n_persistence, n_cycle, n_lam


def neighbour_indices(theta_index: int) -> list[int]:
    """Grid indices that differ from `theta_index` by one step in exactly one
    of the three axes (persistence, cycle, volatility-shape), adjacent in
    each axis's own declared value list (paper Appendix H, "one step in
    exactly one coordinate"). Boundary members simply have fewer neighbours.

    Valid because `build_grid()` builds the grid as
    `itertools.product(persistences, cycles, lam_values)`, i.e. a flat index
    `p_idx * (n_cycle * n_lam) + c_idx * n_lam + l_idx` with the last axis
    varying fastest.
    """
    n_p, n_c, n_l = _grid_axis_sizes()
    l_idx = theta_index % n_l
    rem = theta_index // n_l
    c_idx = rem % n_c
    p_idx = rem // n_c

    neighbours: set[int] = set()
    for p2 in (p_idx - 1, p_idx + 1):
        if 0 <= p2 < n_p:
            neighbours.add(p2 * n_c * n_l + c_idx * n_l + l_idx)
    for c2 in (c_idx - 1, c_idx + 1):
        if 0 <= c2 < n_c:
            neighbours.add(p_idx * n_c * n_l + c2 * n_l + l_idx)
    for l2 in (l_idx - 1, l_idx + 1):
        if 0 <= l2 < n_l:
            neighbours.add(p_idx * n_c * n_l + c_idx * n_l + l2)
    neighbours.discard(theta_index)
    return sorted(neighbours)


def _nonlinear_ll(nl_params: np.ndarray, kind: str, y: np.ndarray, s: np.ndarray,
                   Omega: np.ndarray) -> tuple[float, np.ndarray, float]:
    if kind == "logistic":
        r, s0 = nl_params
        X = _logistic_design(s, r, s0)
    else:
        (r,) = nl_params
        X = _gompertz_design(s, r)
    beta, kappa2, loglik = gls_profile_loglik(y, X, Omega)
    return loglik, beta, kappa2


def _fit_nonlinear_restricted(y: np.ndarray, s: np.ndarray, anchor_weeks: np.ndarray,
                               n_full: int, restricted_models: list[DeviationModel],
                               kind: str, nl_params_start: np.ndarray,
                               bounds: list[tuple[float, float]],
                               reference_omega: np.ndarray) -> TrendFit:
    """Bootstrap-refit variant of `src.trend_models.fit_nonlinear_model`,
    restricted to a small grid (the fitted member and its neighbours) instead
    of the full 1,216-member grid.

    `fit_nonlinear_model`'s stage 1 looks up a fixed "random walk, no cycle"
    grid member to search the nonlinear parameters at; that member need not
    be among the few neighbours considered here, so stage 1 is adapted to
    search instead at `reference_omega` (the covariance of the model the
    bootstrap draw was actually simulated from) and start only from the
    original fit's nonlinear parameters, rather than the original 24-point
    start grid. Both are good choices precisely because the bootstrap sample
    is drawn from the fitted model and its own parameters already close to
    optimal; a full 24-point restart per bootstrap draw would also be far
    too slow for the computing budget. Stage 2/3 (grid profiling, then a
    refit-if-better pass) are otherwise identical to `fit_nonlinear_model`.
    """
    res1 = minimize(lambda p: -_nonlinear_ll(p, kind, y, s, reference_omega)[0],
                     nl_params_start, method="L-BFGS-B", bounds=bounds)
    if not res1.success or not np.isfinite(res1.fun):
        raise RuntimeError("bootstrap nonlinear stage-1 search failed")
    nl_params = res1.x

    best = None
    for local_idx, model in enumerate(restricted_models):
        Omega = model_covariance(model, anchor_weeks, n_full)
        loglik, beta, kappa2 = _nonlinear_ll(nl_params, kind, y, s, Omega)
        if not np.isfinite(loglik):
            continue
        if best is None or loglik > best[0]:
            best = (loglik, local_idx, beta, kappa2)
    if best is None:
        raise RuntimeError("bootstrap nonlinear stage-2: no finite grid member")
    best_loglik, best_local_idx, best_beta, best_kappa2 = best

    Omega_sel = model_covariance(restricted_models[best_local_idx], anchor_weeks, n_full)
    res2 = minimize(lambda p: -_nonlinear_ll(p, kind, y, s, Omega_sel)[0],
                     nl_params, method="L-BFGS-B", bounds=bounds)
    if res2.success and np.isfinite(res2.fun) and -res2.fun > best_loglik:
        nl_params = res2.x
        best_loglik, best_beta, best_kappa2 = _nonlinear_ll(nl_params, kind, y, s, Omega_sel)

    if not np.isfinite(best_loglik):
        raise RuntimeError("bootstrap nonlinear refit non-finite")
    params = ({"r": float(nl_params[0])} if kind == "gompertz"
              else {"r": float(nl_params[0]), "s0": float(nl_params[1])})
    return TrendFit(kind, params, best_local_idx, float(best_loglik), 3, best_beta, best_kappa2)


def _target_week_ages(years: tuple[int, ...]) -> np.ndarray:
    """Age in weeks from genesis of the last Saturday of each year in `years`,
    end-of-day, using the exact same convention as
    `data.data_loader.build_weekly_series` (paper Section 2)."""
    genesis_naive = GENESIS_TIMESTAMP.tz_localize(None)
    ages = []
    for year in years:
        last_saturday = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="W-SAT")[-1]
        end_of_day = last_saturday + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ages.append((end_of_day - genesis_naive).total_seconds() / (7 * 24 * 3600))
    return np.array(ages)


def _conditional_future(model_name: str, fit: TrendFit, grid: list[DeviationModel],
                         anchor_weeks: np.ndarray, resid_obs: np.ndarray,
                         s_future: np.ndarray, anchor_future: np.ndarray,
                         n_full_ext: int) -> tuple[np.ndarray, np.ndarray]:
    """Conditional predictive mean and covariance of the future log-prices
    given the observed residuals, under one Gaussian component (`fit`'s
    deviation model, beta and kappa2). Partitioned-Gaussian identities
    (paper Appendix H): with actual covariance kappa2 * Omega, kappa2
    cancels in the conditional mean but not in the conditional covariance.
    """
    model = grid[fit.theta_index]
    n_obs = len(anchor_weeks)
    anchor_combined = np.concatenate([anchor_weeks, anchor_future])
    Omega_ext = model_covariance(model, anchor_combined, n_full_ext)
    Omega11 = Omega_ext[:n_obs, :n_obs]
    Omega12 = Omega_ext[:n_obs, n_obs:]
    Omega22 = Omega_ext[n_obs:, n_obs:]

    solved_resid = np.linalg.solve(Omega11, resid_obs)
    mu_cond = Omega12.T @ solved_resid
    solved_Omega12 = np.linalg.solve(Omega11, Omega12)
    Sigma_cond = fit.kappa2 * (Omega22 - Omega12.T @ solved_Omega12)

    X_future = _design_matrix_for(model_name, s_future, fit)
    mean_future = X_future @ fit.beta + mu_cond
    return mean_future, Sigma_cond


def _mc_quantiles(components: list[tuple[np.ndarray, np.ndarray]], years: tuple[int, ...],
                   seed: int, n_draws: int = MC_DRAWS) -> list[dict]:
    """Equal-weight mixture of Gaussian `components` (mean, covariance pairs,
    in log-price space); Monte Carlo median and 90% interval of price at each
    target year (paper's reporting convention: USD medians and intervals)."""
    rng = np.random.default_rng(seed)
    n_components = len(components)
    n_targets = len(years)
    comp_choice = rng.integers(0, n_components, size=n_draws)
    price_samples = np.empty((n_draws, n_targets))
    for k, (mean, cov) in enumerate(components):
        mask = comp_choice == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        draws = rng.multivariate_normal(mean, cov, size=n_k)
        price_samples[mask] = np.exp(draws)
    forecasts = []
    for j, year in enumerate(years):
        col = price_samples[:, j]
        forecasts.append({
            "year": int(year),
            "median_usd": float(np.median(col)),
            "p05_usd": float(np.percentile(col, 5)),
            "p95_usd": float(np.percentile(col, 95)),
        })
    return forecasts


def _bootstrap_one_model(model_name: str, fit: TrendFit, y: np.ndarray, s: np.ndarray,
                          anchor_weeks: np.ndarray, n_full: int, grid: list[DeviationModel],
                          n_bootstrap: int, seed: int) -> list[TrendFit]:
    """Parametric bootstrap refits of `fit`, restricted to the fitted grid
    member and its neighbours (paper Appendix H). Returns the list of
    successful refits (each a `TrendFit` with a *global* grid index)."""
    rng = np.random.default_rng(seed)
    indices = [fit.theta_index] + neighbour_indices(fit.theta_index)
    restricted_models = [grid[i] for i in indices]
    X_fit = _design_matrix_for(model_name, s, fit)
    is_linear = model_name in ("constant_drift", "power_law", "drift_and_log_age")

    if not is_linear:
        reference_omega = model_covariance(grid[fit.theta_index], anchor_weeks, n_full)
        if model_name == "logistic":
            bounds = [LOGISTIC_BOUNDS[0], (0.0, float(s.max()) * 2)]
            nl_start = np.array([fit.params["r"], fit.params["s0"]])
        else:
            bounds = GOMPERTZ_BOUNDS
            nl_start = np.array([fit.params["r"]])

    successes: list[TrendFit] = []
    for _ in range(n_bootstrap):
        u_full = np.sqrt(fit.kappa2) * simulate_path(grid[fit.theta_index], n_full, rng)
        y_boot = X_fit @ fit.beta + u_full[anchor_weeks]
        try:
            if is_linear:
                design_fn = functools.partial(exp5._design_matrix, model_name, s)
                refit = fit_linear_model(y_boot, anchor_weeks, n_full, design_fn,
                                          restricted_models, fit.n_trend_params, model_name)
            else:
                refit = _fit_nonlinear_restricted(y_boot, s, anchor_weeks, n_full,
                                                   restricted_models, model_name, nl_start,
                                                   bounds, reference_omega)
            if not np.isfinite(refit.log_likelihood):
                continue
            refit.theta_index = indices[refit.theta_index]
            successes.append(refit)
        except Exception:
            continue
    return successes


def run(n_bootstrap_run: int = config.FORECAST_N_BOOTSTRAP_RUN) -> dict:
    """Full experiment 6 pipeline: fit the five trend models (reusing
    experiment 5's procedure), bootstrap-refit each with a grid restricted to
    the fitted member's neighbours, build the per-model Gaussian mixture of
    conditional predictive densities at the four target years, and save
    results/plot."""
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

    s_future = _target_week_ages(config.FORECAST_TARGET_YEARS)
    anchor_future = np.round(s_future).astype(int)
    n_full_ext = int(max(anchor_weeks.max(), anchor_future.max())) + 1

    # Stage 1: point-estimate fits, exactly as experiment 5 (Appendix H).
    fits: dict[str, TrendFit] = {}
    fits["constant_drift"] = exp5._parallel_fit_linear(y, anchor_weeks, n_full, "constant_drift", s, grid, 2)
    fits["power_law"] = exp5._parallel_fit_linear(y, anchor_weeks, n_full, "power_law", s, grid, 2)
    fits["drift_and_log_age"] = exp5._parallel_fit_linear(y, anchor_weeks, n_full, "drift_and_log_age", s, grid, 3)
    fits["logistic"] = exp5._fit_nonlinear_parallel(
        y, s, anchor_weeks, n_full, grid, "logistic",
        start_grid=exp5._logistic_start_grid(s), bounds=[(1e-4, 1.0), (0.0, float(s.max()) * 2)])
    fits["gompertz"] = exp5._fit_nonlinear_parallel(
        y, s, anchor_weeks, n_full, grid, "gompertz",
        start_grid=exp5._gompertz_start_grid(), bounds=[(1e-5, 1e-1)])
    print(f"point-estimate fits done ({time.time() - t0:.1f}s)", flush=True)

    # Calibrate the bootstrap count against the time budget *before* committing
    # to a count for all five models: linear and nonlinear refits have very
    # different per-draw costs (the nonlinear refit's own inner L-BFGS-B
    # search makes it several times slower than a linear GLS solve), so
    # timing only the first model (always "constant_drift", linear) would
    # badly underestimate the cost of the two nonlinear models later in
    # `config.TREND_MODEL_NAMES`. One single-draw timing per class (linear,
    # nonlinear) is cheap and representative.
    n_bootstrap_used = n_bootstrap_run
    linear_names = [m for m in config.TREND_MODEL_NAMES
                    if m in ("constant_drift", "power_law", "drift_and_log_age")]
    nonlinear_names = [m for m in config.TREND_MODEL_NAMES if m not in linear_names]
    if n_bootstrap_used > 1:
        t_cal = time.time()
        t_lin0 = time.time()
        _bootstrap_one_model(linear_names[0], fits[linear_names[0]], y, s, anchor_weeks,
                              n_full, grid, 1, BOOTSTRAP_SEED - 1)
        per_draw_linear = time.time() - t_lin0
        t_nl0 = time.time()
        _bootstrap_one_model(nonlinear_names[0], fits[nonlinear_names[0]], y, s, anchor_weeks,
                              n_full, grid, 1, BOOTSTRAP_SEED - 1)
        per_draw_nonlinear = time.time() - t_nl0
        projected_total = (len(linear_names) * per_draw_linear
                           + len(nonlinear_names) * per_draw_nonlinear) * n_bootstrap_used
        print(f"calibration: {per_draw_linear:.2f}s/draw (linear), "
              f"{per_draw_nonlinear:.2f}s/draw (nonlinear), projected total "
              f"{projected_total:.0f}s ({time.time() - t_cal:.1f}s spent calibrating)", flush=True)
        if projected_total > TIME_BUDGET_SECONDS:
            scale = TIME_BUDGET_SECONDS / projected_total
            n_bootstrap_used = max(1, int(n_bootstrap_used * scale))
            print(f"reducing n_bootstrap_used to {n_bootstrap_used} "
                  f"(projected {projected_total:.0f}s > budget {TIME_BUDGET_SECONDS:.0f}s)",
                  flush=True)

    model_results = []
    for i, model_name in enumerate(config.TREND_MODEL_NAMES):
        t_model = time.time()
        fit = fits[model_name]
        successes = _bootstrap_one_model(model_name, fit, y, s, anchor_weeks, n_full,
                                          grid, n_bootstrap_used, BOOTSTRAP_SEED + i)

        all_fits = [fit] + successes
        components = []
        for comp_fit in all_fits:
            comp_resid_obs = y - _design_matrix_for(model_name, s, comp_fit) @ comp_fit.beta
            mean_future, Sigma_cond = _conditional_future(
                model_name, comp_fit, grid, anchor_weeks, comp_resid_obs,
                s_future, anchor_future, n_full_ext)
            components.append((mean_future, Sigma_cond))

        forecasts = _mc_quantiles(components, config.FORECAST_TARGET_YEARS, seed=MC_SEED + i)

        model_results.append({
            "model_name": model_name,
            "fitted_log_likelihood": fit.log_likelihood,
            "theta_index": fit.theta_index,
            "params": fit.params,
            "n_successful_refits": len(successes),
            "n_attempted_refits": n_bootstrap_used,
            "forecasts": forecasts,
        })
        print(f"{model_name}: {len(successes)}/{n_bootstrap_used} successful refits "
              f"({time.time() - t_model:.1f}s)", flush=True)

    elapsed = time.time() - t0
    out = {
        "n_observations": n_obs,
        "target_years": list(config.FORECAST_TARGET_YEARS),
        "future_volatility_annual": config.FUTURE_VOLATILITY_FLAT,
        "n_bootstrap_used": n_bootstrap_used,
        "n_bootstrap_paper": config.FORECAST_N_BOOTSTRAP_PAPER,
        "models": model_results,
        "elapsed_seconds": elapsed,
        "notes": {
            "bootstrap_count_reduced":
                f"config.FORECAST_N_BOOTSTRAP_RUN={config.FORECAST_N_BOOTSTRAP_RUN} requested; "
                f"{n_bootstrap_used} actually used, instead of the paper's "
                f"{config.FORECAST_N_BOOTSTRAP_PAPER}, for computing time; see results/RESULTS.md. "
                "Pass --full to attempt the paper's count.",
            "nonlinear_bootstrap_search":
                "logistic/Gompertz bootstrap refits search the nonlinear parameters from a "
                "single start (the original fit's own parameters) against the fitted model's "
                "own covariance as the stage-1 reference, instead of the original 24-point "
                "start grid against a fixed random-walk reference, because the small "
                "neighbour-restricted grid need not contain that reference member; see "
                "_fit_nonlinear_restricted docstring.",
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp6_forecast_densities.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    _plot_forecast_fan(model_results)
    return out


def _plot_forecast_fan(model_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    years = [fc["year"] for fc in model_results[0]["forecasts"]]
    for res in model_results:
        medians = [fc["median_usd"] for fc in res["forecasts"]]
        p05 = [fc["p05_usd"] for fc in res["forecasts"]]
        p95 = [fc["p95_usd"] for fc in res["forecasts"]]
        line, = ax.plot(years, medians, marker="o", label=res["model_name"])
        ax.fill_between(years, p05, p95, color=line.get_color(), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("Year (last Saturday)")
    ax.set_ylabel("BTC/USD closing price (log scale)")
    ax.set_title("Registered prospective forecasts: median and 90% interval (Table 5)")
    ax.set_xticks(years)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_forecast_fan.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="use the paper's bootstrap count (49) instead of the reduced "
                             "development count; much slower, not run by default")
    args = parser.parse_args()
    n_boot = config.FORECAST_N_BOOTSTRAP_PAPER if args.full else config.FORECAST_N_BOOTSTRAP_RUN
    run(n_bootstrap_run=n_boot)
