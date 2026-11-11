"""Experiment 7: conventional diagnostics robustness study (paper Appendix G).

Self-contained pure simulation study: no Bitcoin price data, no Dataloader
calls. It shows that conventional residual diagnostics (ADF, KPSS,
Newey-West HAC) do not keep nominal size/coverage when deviations are as
persistent as Bitcoin's, which is why the paper's main test (exp1) uses
simulation-based calibration instead of these conventional tools.

For each of 6 deviation types, simulate many DAILY histories of a true
power-law trend (exponent b=5.7, a=0) plus a deviation path, fit an OLS
power-law trend to the simulated series, and check:
  (a) does ADF reject the unit root in the residuals (p < 0.05)?
  (b) does KPSS fail to reject stationarity (p > 0.05)?
  (c) do both (a) and (b) hold at once?
  (d) does the Newey-West 95% CI for the log-age slope cover the true b=5.7?
  (e) does the post-2020 data stay within +/-2*sigma_hat of a trend fitted
      only through 2020 ("band check")?

Daily-frequency translation (the paper gives these only at the weekly
latent-grid level in src/deviation_family.py; that module's API is weekly
by construction -- e.g. rho = 1 - c/821 -- so it is simpler and safer to
re-derive small daily analogues here rather than force a weekly module onto
a daily grid). All such choices are documented inline below.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from src import config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SEED = 20261101  # arbitrary; the paper does not specify a seed for this diagnostic study

# Daily observation count and age measurement inferred from the weekly design, since
# the paper does not give exact figures for a daily simulation: 820 weeks * 7 = 5740
# days, matching the real record length approximately.
N_DAYS = config.PRIMARY_N_PAPER * 7  # 5740
TRUE_B = config.PUBLISHED_EXPONENT  # 5.7, the paper's published log-age exponent
TRUE_A = 0.0  # only the exponent and residual diagnostics matter here

# "Four-year window taken literally" as ~1460 daily Newey-West lags.
NW_LAGS = round(4 * 365)

# adfuller's default Schwert-rule maxlag for n=5740 is ~39; each candidate lag refit
# does a full SVD-based OLS and, in this sandboxed environment, that makes a single
# ADF call cost seconds instead of milliseconds. Cap the lag search at 20 days (still
# several weeks of autocorrelation) for computational feasibility; autolag='AIC' still
# selects the best lag among 0..ADF_MAXLAG exactly as specified.
ADF_MAXLAG = 20

# Band check: fit through end of 2020 (day_index < 10 calendar years from 2011),
# check the remainder lies within +/-2*sigma_hat of that fit's prediction.
BAND_CUTOFF_DAY = (2020 - 2011 + 1) * 365  # ~3650

# Representative stochastic (Harvey trigonometric) cycle configuration: period ~3.5
# years (the shortest of the paper's CYCLE_PERIOD_YEARS), amplitude r=5.86 (the
# middle of the paper's non-zero CYCLE_R_VALUES), kappa=3 (the middle of
# CYCLE_KAPPA_VALUES). This experiment illustrates diagnostic robustness with one
# plausible cycle, not a recalibration of the full 19-cycle grid.
CYCLE_PERIOD_DAYS = 3.5 * 365
CYCLE_R = 5.86
CYCLE_KAPPA = 3

# AR persistence: rho chosen so that rho**halflife_days == 0.5.
AR_HALFLIFE_LONG_DAYS = 2.2 * 365
AR_HALFLIFE_SHORT_DAYS = 0.55 * 365

DEVIATION_TYPES = [
    "random_walk_profile",
    "random_walk_constant",
    "rw_cycle_profile",
    "rw_cycle_constant",
    "ar_hl2.2y_cycle",
    "ar_hl0.55y_cycle",
]

# Hardcoded literal reference values from the paper (Appendix G), for comparison only;
# they are not fed into the simulation in any way.
PAPER_EXPECTED = {
    "random_walk_profile": {"adf": 0.44, "kpss": 0.06, "both": 0.05, "nw": 0.35, "band": 0.32},
    "random_walk_constant": {"adf": 0.18, "kpss": 0.04, "both": 0.03, "nw": 0.34, "band": 0.09},
    "rw_cycle_profile": {"adf": 0.20, "kpss": 0.15, "both": 0.04, "nw": 0.36, "band": 0.24},
    "rw_cycle_constant": {"adf": 0.06, "kpss": 0.12, "both": 0.01, "nw": 0.33, "band": 0.10},
    "ar_hl2.2y_cycle": {"adf": 0.20, "kpss": 0.20, "both": 0.06, "nw": 0.44, "band": 0.38},
    "ar_hl0.55y_cycle": {"adf": 0.31, "kpss": 0.60, "both": 0.19, "nw": 0.62, "band": 0.44},
}


def daily_sigma_profile(day_index: np.ndarray) -> np.ndarray:
    """Daily innovation s.d. from config.ANNUAL_VOLATILITY (Appendix A).

    Each day is mapped to a calendar year assuming the series starts in 2011
    (year = 2011 + day_index // 365), clamped to the declared support
    [2011, 2026]. The annual sigma is converted to a daily sigma via
    sigma_daily = sigma_annual / sqrt(365).
    """
    years = 2011 + day_index // 365
    years = np.clip(years, min(config.ANNUAL_VOLATILITY), max(config.ANNUAL_VOLATILITY))
    annual = np.array([config.ANNUAL_VOLATILITY[int(y)] for y in years])
    return annual / np.sqrt(365.0)


# Constant volatility = the time-series average of the annual volatility profile,
# converted to daily units the same way as the profile case (documented choice; the
# paper does not specify a constant-volatility value for this robustness study).
CONSTANT_SIGMA = float(np.mean(list(config.ANNUAL_VOLATILITY.values()))) / np.sqrt(365.0)


def simulate_cycle(n: int, rng: np.random.Generator) -> np.ndarray:
    """Harvey stochastic trigonometric cycle, daily analogue of the weekly recursion
    in src/deviation_family.py (same functional form, period/rho_c/omega measured in
    days instead of weeks), using one representative configuration (see module docstring).
    """
    omega = 2 * np.pi / CYCLE_PERIOD_DAYS
    rho_c = 0.5 ** (1.0 / (CYCLE_KAPPA * CYCLE_PERIOD_DAYS))
    scale = CYCLE_R * config.SIGMA_BAR
    inn = rng.standard_normal((n, 2)) * np.sqrt(1 - rho_c ** 2) * scale
    innov = (inn[:, 0] + 1j * inn[:, 1]).astype(complex)
    state0 = rng.standard_normal(2) * scale  # stationary start
    z0 = complex(state0[0], state0[1])
    c = rho_c * np.exp(-1j * omega)
    x_ext = np.concatenate(([z0], innov[:-1]))
    z = lfilter(np.array([1.0 + 0j]), np.array([1.0 + 0j, -c]), x_ext)
    return z.real


def simulate_random_walk(n: int, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n) * sigma
    return np.cumsum(e)


def simulate_ar(n: int, rho: float, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n) * sigma
    return lfilter([1.0], [1.0, -rho], e)


def simulate_deviation(kind: str, day_index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(day_index)
    profile_sigma = daily_sigma_profile(day_index)
    if kind == "random_walk_profile":
        return simulate_random_walk(n, profile_sigma, rng)
    if kind == "random_walk_constant":
        return simulate_random_walk(n, np.full(n, CONSTANT_SIGMA), rng)
    if kind == "rw_cycle_profile":
        return simulate_random_walk(n, profile_sigma, rng) + simulate_cycle(n, rng)
    if kind == "rw_cycle_constant":
        return simulate_random_walk(n, np.full(n, CONSTANT_SIGMA), rng) + simulate_cycle(n, rng)
    if kind == "ar_hl2.2y_cycle":
        rho = 0.5 ** (1.0 / AR_HALFLIFE_LONG_DAYS)
        return simulate_ar(n, rho, profile_sigma, rng) + simulate_cycle(n, rng)
    if kind == "ar_hl0.55y_cycle":
        rho = 0.5 ** (1.0 / AR_HALFLIFE_SHORT_DAYS)
        return simulate_ar(n, rho, profile_sigma, rng) + simulate_cycle(n, rng)
    raise ValueError(f"unknown deviation type {kind}")


def diagnose_one(y: np.ndarray, age_weeks: np.ndarray, day_index: np.ndarray) -> dict:
    """Apply the conventional diagnostics (a)-(e) to one simulated history."""
    X = np.column_stack([np.ones_like(age_weeks), np.log(age_weeks)])
    fit = OLS(y, X).fit()
    resid = fit.resid

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        adf_p = adfuller(resid, regression="c", autolag="AIC", maxlag=ADF_MAXLAG)[1]
        kpss_p = kpss(resid, regression="c", nlags="auto")[1]

    adf_reject = bool(adf_p < 0.05)
    kpss_accept = bool(kpss_p > 0.05)

    hac_fit = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    slope, se = hac_fit.params[1], hac_fit.bse[1]
    lo, hi = slope - 1.96 * se, slope + 1.96 * se
    nw_covers = bool(lo <= TRUE_B <= hi)

    mask_fit = day_index < BAND_CUTOFF_DAY
    band_fit = OLS(y[mask_fit], X[mask_fit]).fit()
    sigma_hat = float(np.std(band_fit.resid, ddof=X.shape[1]))
    pred_all = X @ band_fit.params
    mask_rest = ~mask_fit
    if mask_rest.any():
        in_band = bool(np.all(np.abs(y[mask_rest] - pred_all[mask_rest]) <= 2 * sigma_hat))
    else:
        in_band = True

    return {
        "adf_reject": adf_reject,
        "kpss_accept": kpss_accept,
        "both": adf_reject and kpss_accept,
        "nw_covers": nw_covers,
        "in_band": in_band,
    }


def run(full: bool = False) -> dict:
    t0 = time.time()
    day_index = np.arange(N_DAYS)
    # age in weeks = day_index/7 + 1, kept strictly positive (small-offset convention).
    age_weeks = day_index / 7.0 + 1.0
    true_trend = TRUE_A + TRUE_B * np.log(age_weeks)

    # Time ONE simulated history + its diagnostics to estimate total runtime and pick
    # n_histories accordingly (reduced from the paper's 500 histories per type for
    # computational feasibility; use --full for the paper configuration).
    probe_rng = np.random.default_rng(SEED + 999)
    t_probe = time.time()
    u_probe = simulate_deviation(DEVIATION_TYPES[0], day_index, probe_rng)
    _ = diagnose_one(true_trend + u_probe, age_weeks, day_index)
    per_history_seconds = time.time() - t_probe

    if full:
        n_histories = 500
    else:
        # Reduced from the paper's 500 histories per type for computational
        # feasibility; use --full for the paper configuration. Step down through a
        # candidate sequence until the estimated total runtime fits the ~4 minute
        # budget, down to a floor of 10 histories per type.
        budget_seconds = 240.0
        for n_histories in (60, 30, 15, 10):
            est_total = per_history_seconds * len(DEVIATION_TYPES) * n_histories
            if est_total <= budget_seconds:
                break
        print(
            f"per-history time: {per_history_seconds:.4f}s, "
            f"estimated total: {est_total:.1f}s, n_histories={n_histories}"
        )

    rng = np.random.default_rng(SEED)
    results = []
    for kind in DEVIATION_TYPES:
        adf_c = kpss_c = both_c = nw_c = band_c = 0
        for _ in range(n_histories):
            u = simulate_deviation(kind, day_index, rng)
            d = diagnose_one(true_trend + u, age_weeks, day_index)
            adf_c += d["adf_reject"]
            kpss_c += d["kpss_accept"]
            both_c += d["both"]
            nw_c += d["nw_covers"]
            band_c += d["in_band"]
        results.append(
            {
                "deviation_type": kind,
                "n_histories": n_histories,
                "adf_reject_rate": adf_c / n_histories,
                "kpss_accept_rate": kpss_c / n_histories,
                "both_rate": both_c / n_histories,
                "newey_west_coverage": nw_c / n_histories,
                "band_rate": band_c / n_histories,
                "paper_expected": PAPER_EXPECTED[kind],
            }
        )
        print(f"{kind}: done ({n_histories} histories)")

    elapsed = time.time() - t0
    out = {
        "seed": SEED,
        "n_days": N_DAYS,
        "nw_lags": NW_LAGS,
        "band_cutoff_day": BAND_CUTOFF_DAY,
        "cycle_config": {"period_days": CYCLE_PERIOD_DAYS, "r": CYCLE_R, "kappa": CYCLE_KAPPA},
        "n_histories": n_histories,
        "per_history_seconds": per_history_seconds,
        "full_paper_configuration": full,
        "results": results,
        "elapsed_seconds": elapsed,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp7_conventional_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="use the paper's 500 histories per deviation type"
    )
    args = parser.parse_args()
    run(full=args.full)
