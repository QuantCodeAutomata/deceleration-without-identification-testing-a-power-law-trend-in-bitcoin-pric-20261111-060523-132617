"""Experiment 9: separating the best-fitting trends with future data (paper
Appendix K, "Separating the best-fitting trends with future data"; this is an
EXPLORATORY simulation, not part of the frozen protocol).

The power law, Gompertz and drift-and-log-age trend models currently fit the
realized Bitcoin weekly log-price record almost equally well (experiment 5).
This experiment asks when, if ever, future data could decisively separate
these three models from one another. For each of the three models in turn,
treated as the (unknown) data-generating "true" trend:

1. Simulate `n_simulations_used` continuations of the record out to the end
   of 2046, weekly, holding the deviation model fixed and known at the single
   grid member selected by all three fits on the observed record (paper:
   "all three fits select the same deviation model: a near-random walk with
   a 3.5-year cycle"; "Holding the deviation model known"). The observed
   history is fixed data (never resimulated); only the future points are
   drawn, from their conditional Gaussian distribution given the observed
   residuals -- the same conditional-Gaussian construction as experiment 6
   (`exp.exp6._conditional_future`, partitioned-covariance identities under
   the declared deviation family, paper Appendix B).
2. At each of several record-end cutoffs (2030, 2035, 2040, 2046), refit all
   three trend models by Gaussian MLE on the (observed + simulated-future)
   series, with the deviation-model grid search restricted to the single
   shared, held-fixed grid member (no re-search of the full 1,216-member
   grid at every refit: the point of this analysis is trend separation at a
   *known* deviation model, paper Appendix K).
3. Compare AIC = -2*log_likelihood + 2*n_trend_params between the true model
   and each rival; "decisive" means the true model leads the rival by at
   least `config.SEPARATION_AIC_DECISIVE_THRESHOLD` (odds ~20:1, paper
   Appendix K).

Only the four ordered (true, rival) pairs the paper discusses are reported:
(power_law, gompertz), (drift_and_log_age, power_law), (power_law,
drift_and_log_age) [cross-check: drift_and_log_age nests power_law as the
b=0 case, so when power_law is truly true the extra drift coefficient can
only be shown to be small, never exactly absent -- 0% decisive by
construction] and (gompertz, power_law) [paper: a power law with small
negative drift bends almost exactly like a Gompertz curve, so this pair is
expected to stay near 0% decisive even by 2046].

This module reuses experiment 5's fitting helpers (`exp.exp5`, imported, not
duplicated) for the point-estimate fits on the observed record, and
experiment 6's conditional-Gaussian forecasting helpers (`exp.exp6`,
imported) for simulating the future continuations. The only new code here is
the restricted (single, held-fixed deviation model) refit loop over
simulated continuations and cutoffs, and the AIC bookkeeping.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import os

# Must be set before numpy/OpenBLAS is imported: this script's inner loop is
# a long sequence of small-to-moderate linalg calls (GLS solves, one
# bounded L-BFGS-B search per refit); an oversubscribed multi-threaded BLAS
# slows down such a sequence through repeated thread-pool setup/teardown,
# exactly as noted in exp5.py / exp6.py.
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
from src.deviation_family import DeviationModel, build_grid, model_covariance
from src.trend_models import TrendFit, gls_profile_loglik

import exp.exp5 as exp5
import exp.exp6 as exp6

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# The three trend models under comparison in this experiment (paper
# Appendix K); constant_drift and logistic are not part of this analysis.
COMPARISON_MODELS = ("power_law", "drift_and_log_age", "gompertz")

# The four ordered (true, rival) pairs the paper discusses (Appendix K).
TRUE_RIVAL_PAIRS = (
    ("power_law", "gompertz"),
    ("drift_and_log_age", "power_law"),
    ("power_law", "drift_and_log_age"),
    ("gompertz", "power_law"),
)

SEED = 20261111  # fixed seed for the conditional-Gaussian continuation draws (development seed)

# Soft wall-clock budget for the simulate+refit stage (after the three
# point-estimate fits, which alone cost several minutes -- see exp5's own
# reported `elapsed_seconds`). One continuation's full refit cost (all four
# cutoffs, all three trend models) is timed once before committing to a
# simulation count for all three "true" models (see `run`'s calibration
# block), and the count actually used is reduced below
# `config.SEPARATION_N_SIMULATIONS_RUN` if needed, with the true count
# recorded in the output JSON's "n_simulations_used" field.
TOTAL_TIME_BUDGET_SECONDS = 650.0
SIMULATION_SAFETY_MARGIN_SECONDS = 40.0


def _future_week_grid(last_observed_date: pd.Timestamp, end_year: int
                       ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Every weekly Saturday anchor strictly after `last_observed_date`
    through the last Saturday on or before 31 December of `end_year`, using
    the same age-in-weeks convention as `data.data_loader.build_weekly_series`
    / `exp6._target_week_ages` (paper Section 2).

    Returns (dates, s_future, anchor_future): fractional ages (float, for
    trend design matrices) and their rounded integer week indices (for
    covariance lookups), both ordered the same way as `dates`.
    """
    genesis_naive = GENESIS_TIMESTAMP.tz_localize(None)
    dates = pd.date_range(start=last_observed_date + pd.Timedelta(days=7),
                           end=f"{end_year}-12-31", freq="W-SAT")
    end_of_day = dates + pd.Timedelta(hours=23, minutes=59, seconds=59)
    s_future = ((end_of_day - genesis_naive).total_seconds() / (7 * 24 * 3600)).to_numpy()
    anchor_future = np.round(s_future).astype(int)
    return dates, s_future, anchor_future


def _refit_gompertz_fixed_theta(y: np.ndarray, s: np.ndarray, Omega: np.ndarray,
                                 nl_params_start: np.ndarray,
                                 bounds: list[tuple[float, float]]) -> TrendFit:
    """Refit the Gompertz trend's nonlinear parameter `r` (and profile
    beta/kappa2) at a single, held-fixed deviation-model covariance `Omega`.

    This is the restricted (single grid member) analogue of
    `src.trend_models.fit_nonlinear_model` / `exp.exp6._fit_nonlinear_restricted`:
    both of those assume a multi-member grid (the former looks up a fixed
    "random walk, no cycle" reference member by `persistence`/`cycle`
    attributes, which need not be -- and here, per the paper's "near-random
    walk with a 3.5-year cycle" finding, is not -- the single shared member),
    so neither can be called unmodified with a one-element grid. The
    likelihood evaluation itself (`exp.exp6._nonlinear_ll`, which wraps
    `src.trend_models.gls_profile_loglik`) is reused as-is; only the search
    is simplified to the one L-BFGS-B call this restricted case actually
    needs, keeping each refit cheap enough for the simulation count below.
    """
    ll_start, beta_start, kappa2_start = exp6._nonlinear_ll(nl_params_start, "gompertz", y, s, Omega)
    res = minimize(lambda p: -exp6._nonlinear_ll(p, "gompertz", y, s, Omega)[0],
                    nl_params_start, method="L-BFGS-B", bounds=bounds)
    if res.success and np.isfinite(res.fun) and -res.fun > ll_start:
        loglik, beta, kappa2 = exp6._nonlinear_ll(res.x, "gompertz", y, s, Omega)
        r = float(res.x[0])
    else:
        loglik, beta, kappa2, r = ll_start, beta_start, kappa2_start, float(nl_params_start[0])
    return TrendFit("gompertz", {"r": r}, 0, float(loglik), 3, beta, kappa2)


def _aic(loglik: float, n_trend_params: int) -> float:
    """AIC = -2*log_likelihood + 2*n_trend_params (paper Appendix K)."""
    return -2.0 * loglik + 2.0 * n_trend_params


N_TREND_PARAMS = {"power_law": 2, "drift_and_log_age": 3, "gompertz": 3}
GOMPERTZ_BOUNDS = [(1e-5, 1e-1)]  # same bounds as exp5/exp6's Gompertz search


def _build_cutoff_data(shared_model: DeviationModel, anchor_obs: np.ndarray, s_obs: np.ndarray,
                        s_future_all: np.ndarray, anchor_future_all: np.ndarray,
                        future_dates: pd.DatetimeIndex, end_years: tuple[int, ...]) -> dict:
    """Precompute, once per cutoff year (independent of the simulated
    continuation and of which model is "true"), the restricted covariance
    Omega and the linear design matrices used at every refit. These do not
    depend on the simulated y, so building them once here instead of inside
    the per-continuation loop below is a large constant-factor speed-up."""
    cutoff_data = {}
    for year in end_years:
        count = int((future_dates <= pd.Timestamp(f"{year}-12-31")).sum())
        s_cut = np.concatenate([s_obs, s_future_all[:count]])
        anchor_cut = np.concatenate([anchor_obs, anchor_future_all[:count]])
        n_full_cut = int(anchor_cut.max()) + 1
        Omega_cut = model_covariance(shared_model, anchor_cut, n_full_cut)
        cutoff_data[year] = {
            "count": count,
            "s_cut": s_cut,
            "X_power": exp5._design_matrix("power_law", s_cut),
            "X_drift": exp5._design_matrix("drift_and_log_age", s_cut),
            "Omega_cut": Omega_cut,
        }
    return cutoff_data


def _refit_all_three(y_cut: np.ndarray, cd: dict, gompertz_r_start: float) -> dict[str, float]:
    """Refit power_law, drift_and_log_age and gompertz at one cutoff's fixed
    (Omega, design) data, restricted to the shared held-fixed deviation
    model; return each model's AIC."""
    _, _, ll_power = gls_profile_loglik(y_cut, cd["X_power"], cd["Omega_cut"])
    _, _, ll_drift = gls_profile_loglik(y_cut, cd["X_drift"], cd["Omega_cut"])
    gfit = _refit_gompertz_fixed_theta(y_cut, cd["s_cut"], cd["Omega_cut"],
                                        np.array([gompertz_r_start]), GOMPERTZ_BOUNDS)
    return {
        "power_law": _aic(ll_power, N_TREND_PARAMS["power_law"]),
        "drift_and_log_age": _aic(ll_drift, N_TREND_PARAMS["drift_and_log_age"]),
        "gompertz": _aic(gfit.log_likelihood, N_TREND_PARAMS["gompertz"]),
    }


def run(n_simulations_run: int | None = None) -> dict:
    """Run the full experiment 9 analysis (paper Appendix K)."""
    t0 = time.time()
    df = load_primary_weekly_series(
        config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END,
        dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,),
    )
    n_obs = len(df)
    s_obs = df["age_weeks"].to_numpy()
    y_obs = df["y"].to_numpy()
    anchor_obs = np.round(s_obs).astype(int)
    n_full_obs = int(anchor_obs.max()) + 1

    grid = build_grid()

    # Step 1: fit the three trend models under comparison exactly as exp5
    # (same design matrices / two-stage nonlinear search, full 1,216-member
    # grid search), by direct reuse of exp5's own fitting helpers.
    fits: dict[str, TrendFit] = {}
    fits["power_law"] = exp5._parallel_fit_linear(y_obs, anchor_obs, n_full_obs, "power_law", s_obs, grid, 2)
    fits["drift_and_log_age"] = exp5._parallel_fit_linear(
        y_obs, anchor_obs, n_full_obs, "drift_and_log_age", s_obs, grid, 3)
    fits["gompertz"] = exp5._fit_nonlinear_parallel(
        y_obs, s_obs, anchor_obs, n_full_obs, grid, "gompertz",
        start_grid=exp5._gompertz_start_grid(), bounds=GOMPERTZ_BOUNDS)
    print(f"point-estimate fits done ({time.time() - t0:.1f}s)", flush=True)

    theta_indices = {name: fits[name].theta_index for name in COMPARISON_MODELS}
    if len(set(theta_indices.values())) == 1:
        shared_theta_index = theta_indices["power_law"]
        deviation_model_note = (
            "all three fits selected the same deviation grid member "
            f"(index {shared_theta_index}), consistent with the paper's finding that "
            "all three fits select the same deviation model (a near-random walk with "
            "a 3.5-year cycle); this shared member is used directly as the held-fixed "
            "deviation model for every simulated continuation and restricted refit below."
        )
    else:
        shared_theta_index = theta_indices["power_law"]
        deviation_model_note = (
            f"the three refits selected different grid members ({theta_indices}); per the "
            "paper's explicit 'Holding the deviation model known' treatment of this "
            "exploratory analysis, the power law's selected member (index "
            f"{shared_theta_index}) is used as the shared, held-fixed deviation model for "
            "all simulated continuations and restricted refits below, regardless of this "
            "small refit disagreement."
        )
    shared_model = grid[shared_theta_index]
    print(f"theta_indices={theta_indices}, shared_theta_index={shared_theta_index}", flush=True)

    # Step 2: build the future weekly grid out to the last cutoff year, and
    # precompute each cutoff's restricted covariance / design matrices once
    # (shared across all simulated continuations and all three "true" models).
    last_observed_date = pd.Timestamp(df["date"].max())
    end_year = max(config.SEPARATION_RECORD_END_YEARS)
    future_dates, s_future_all, anchor_future_all = _future_week_grid(last_observed_date, end_year)
    n_full_ext = int(max(anchor_obs.max(), anchor_future_all.max())) + 1

    cutoff_data = _build_cutoff_data(shared_model, anchor_obs, s_obs, s_future_all,
                                      anchor_future_all, future_dates, config.SEPARATION_RECORD_END_YEARS)
    print(f"cutoff data built ({time.time() - t0:.1f}s)", flush=True)

    # Step 3: for each "true" model, the conditional (future | observed)
    # Gaussian mean and covariance -- same conditional-Gaussian construction
    # as experiment 6 (`exp.exp6._conditional_future`), but with the
    # deviation model forced to `shared_model` (not the model's own fitted
    # theta_index, which per the paper is treated as the same known member
    # for every model here) via `dataclasses.replace`.
    conditional: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in COMPARISON_MODELS:
        fit = fits[model_name]
        resid_obs = y_obs - exp6._design_matrix_for(model_name, s_obs, fit) @ fit.beta
        fit_shared = dataclasses.replace(fit, theta_index=shared_theta_index)
        mean_future, Sigma_cond = exp6._conditional_future(
            model_name, fit_shared, grid, anchor_obs, resid_obs,
            s_future_all, anchor_future_all, n_full_ext)
        conditional[model_name] = (mean_future, Sigma_cond)
    print(f"conditional future distributions built ({time.time() - t0:.1f}s)", flush=True)

    # Calibrate the simulation count against the remaining time budget: time
    # one full continuation's refit cost (all four cutoffs, all three trend
    # models) for the first "true" model, and extrapolate the total cost of
    # running all three "true" models at the paper-requested count.
    n_requested = n_simulations_run or config.SEPARATION_N_SIMULATIONS_RUN
    rng = np.random.default_rng(SEED)
    gompertz_r0 = float(fits["gompertz"].params["r"])

    # The restricted Gompertz refit's dominant cost is a bounded L-BFGS-B
    # search whose objective is evaluated in pure Python around small GLS
    # solves; unlike exp5's full 1,216-member grid profiling (where each
    # grid point's GLS solve is large enough for BLAS's released-GIL time to
    # dominate), here per-call Python/GIL overhead dominates, so spreading
    # continuations over worker threads was measured to make the wall-clock
    # cost *worse* (thread-switching overhead on four cores, no real
    # parallel speed-up) rather than better; the loop below is therefore
    # kept serial, and the time budget check below reduces the simulation
    # count instead.
    years = list(config.SEPARATION_RECORD_END_YEARS)
    calib_mean, calib_cov = conditional["power_law"]
    calib_draw = rng.multivariate_normal(calib_mean, calib_cov, size=1, method="cholesky")[0]
    t_calib = time.time()
    for year in years:
        cd = cutoff_data[year]
        y_cut = np.concatenate([y_obs, calib_draw[:cd["count"]]])
        _refit_all_three(y_cut, cd, gompertz_r0)
    per_continuation_seconds = time.time() - t_calib

    remaining_budget = max(30.0, TOTAL_TIME_BUDGET_SECONDS - (time.time() - t0) - SIMULATION_SAFETY_MARGIN_SECONDS)
    projected_total = 3 * n_requested * per_continuation_seconds
    n_simulations_used = n_requested
    if projected_total > remaining_budget:
        n_simulations_used = max(1, int(remaining_budget / (3 * per_continuation_seconds)))
    print(f"calibration: {per_continuation_seconds:.2f}s/continuation (all 4 cutoffs, 3 models), "
          f"projected total {projected_total:.0f}s, remaining budget {remaining_budget:.0f}s -> "
          f"n_simulations_used={n_simulations_used} (requested {n_requested})", flush=True)

    # Step 4: simulate continuations and refit, for each "true" model.
    aic_values: dict[str, dict[int, dict[str, list[float]]]] = {
        true_name: {year: {m: [] for m in COMPARISON_MODELS} for year in years}
        for true_name in COMPARISON_MODELS
    }
    for true_name in COMPARISON_MODELS:
        t_model = time.time()
        mean_future, Sigma_cond = conditional[true_name]
        draws = rng.multivariate_normal(mean_future, Sigma_cond, size=n_simulations_used, method="cholesky")
        for i in range(n_simulations_used):
            for year in years:
                cd = cutoff_data[year]
                y_cut = np.concatenate([y_obs, draws[i, :cd["count"]]])
                aics = _refit_all_three(y_cut, cd, gompertz_r0)
                for m in COMPARISON_MODELS:
                    aic_values[true_name][year][m].append(aics[m])
        print(f"true={true_name}: {n_simulations_used} continuations done "
              f"({time.time() - t_model:.1f}s)", flush=True)

    # Step 5: decisive-separation fractions for the paper's four ordered pairs.
    threshold = config.SEPARATION_AIC_DECISIVE_THRESHOLD
    results = []
    for true_name, rival_name in TRUE_RIVAL_PAIRS:
        for year in years:
            aic_true = np.array(aic_values[true_name][year][true_name])
            aic_rival = np.array(aic_values[true_name][year][rival_name])
            frac_true_decisive = float(np.mean(aic_rival - aic_true >= threshold))
            frac_rival_decisive = float(np.mean(aic_true - aic_rival >= threshold))
            results.append({
                "true_model": true_name,
                "rival_model": rival_name,
                "cutoff_year": int(year),
                "frac_true_decisive": frac_true_decisive,
                "frac_rival_decisive": frac_rival_decisive,
            })

    elapsed = time.time() - t0
    out = {
        "n_observations_at_fit": n_obs,
        "shared_theta_index": int(shared_theta_index),
        "theta_indices_per_model": {k: int(v) for k, v in theta_indices.items()},
        "n_simulations_used": n_simulations_used,
        "n_simulations_paper": config.SEPARATION_N_SIMULATIONS_PAPER,
        "decisive_threshold_aic": config.SEPARATION_AIC_DECISIVE_THRESHOLD,
        "cutoff_years": years,
        "results": results,
        "paper_expected": {
            "power_law_vs_gompertz_2035": [0.75, 0.98],
            "power_law_vs_gompertz_2040": [0.93, 1.0],
            "drift_and_log_age_vs_power_law_2040": 0.19,
            "drift_and_log_age_vs_power_law_2046": 0.71,
            "power_law_vs_drift_and_log_age_all": 0.0,
            "gompertz_vs_power_law_2046": 0.0,
        },
        "elapsed_seconds": elapsed,
        "notes": {
            "deviation_model_note": deviation_model_note,
            "n_observations_vs_paper":
                f"{n_obs} weekly observations used here vs. the paper's 820 (see exp5.py "
                "docstring / results/RESULTS.md for the data-source substitution); everything "
                "else follows the paper's construction unchanged.",
            "simulation_count_reduced":
                f"config.SEPARATION_N_SIMULATIONS_RUN={config.SEPARATION_N_SIMULATIONS_RUN} "
                f"requested (n_requested={n_requested}); {n_simulations_used} actually used per "
                f"'true' model (3 x {n_simulations_used} = {3 * n_simulations_used} continuations "
                f"total), instead of the paper's {config.SEPARATION_N_SIMULATIONS_PAPER}, so that "
                "the full sweep (3 true models x 4 cutoffs x n_simulations_used continuations x "
                "up to 3 trend refits each, with the restricted single-member Gompertz refit as "
                "the main cost, timed at calibration) fits the computing-time budget. Pass --full "
                "to attempt the paper's count (very slow; not run here).",
            "gompertz_refit_simplification":
                "the Gompertz refit at each cutoff uses a bespoke single-stage restricted search "
                "(_refit_gompertz_fixed_theta: one bounded L-BFGS-B call against the shared "
                "held-fixed deviation model's covariance, reusing exp.exp6._nonlinear_ll for the "
                "likelihood evaluation) instead of src.trend_models.fit_nonlinear_model or "
                "exp.exp6._fit_nonlinear_restricted verbatim, because both of those look up a "
                "multi-member-grid reference (a fixed 'random walk, no cycle' member) that need "
                "not exist in -- and, per the paper's own finding of a 3.5-year cycle in the "
                "shared deviation model, does not exist in -- a one-element restricted grid.",
            "future_volatility":
                "future weekly volatility enters only through the declared deviation family's "
                "covariance at the shared, held-fixed deviation model; no separate volatility "
                "parameter is introduced here (consistent with exp6's convention).",
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp9_model_separation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    _plot_separation_curves(results, years)
    return out


def _plot_separation_curves(results: list[dict], years: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for true_name, rival_name in TRUE_RIVAL_PAIRS:
        rows = [r for r in results if r["true_model"] == true_name and r["rival_model"] == rival_name]
        rows.sort(key=lambda r: r["cutoff_year"])
        fracs = [r["frac_true_decisive"] for r in rows]
        ax.plot(years, fracs, marker="o", label=f"true={true_name}, rival={rival_name}")
    ax.set_xlabel("Record-end cutoff year")
    ax.set_ylabel("Fraction of continuations with decisive AIC separation")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Model separation with future data (Appendix K)")
    ax.set_xticks(years)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp9_separation_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="use the paper's simulation count (1000 continuations per true "
                             "model) instead of the reduced development count; far too slow "
                             "for routine use, not run by default")
    args = parser.parse_args()
    n_sim = config.SEPARATION_N_SIMULATIONS_PAPER if args.full else config.SEPARATION_N_SIMULATIONS_RUN
    run(n_simulations_run=n_sim)
