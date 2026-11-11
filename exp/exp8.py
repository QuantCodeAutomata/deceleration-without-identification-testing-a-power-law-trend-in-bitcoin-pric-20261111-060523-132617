"""Experiment 8 (exploratory, not part of the frozen protocol): power of the
S1 test against record length (paper Appendix J).

Projects how the achieved power of S1, and the Neyman-Pearson oracle upper
bound on power, improve as the Bitcoin weekly record lengthens from 2026
(today) to 2030, 2036 and 2046, under two future-volatility assumptions: flat
at the 2026 level, or declining at the past rate. Three representative
deviation models stand in for the declared 1,216-member family's alternative
("(a) random walk + cycle with lambda=1, (b) AR (c=5) + cycle with lambda=1,
(c) random walk with lambda=0", paper Section 5/Appendix J), each evaluated
at the paper's published exponent b=5.7.

Record lengths beyond 2026 have no realized price data, so for each end year
a synthetic weekly Saturday anchor grid is built exactly as exp1.py's data
loader builds the realized one (same genesis-relative age-in-weeks formula),
continued with `pandas.date_range(..., freq="W-SAT")`. This reproduces
exp1/exp2's exact 2026 setup (820 anchors, q=6) as a check, then extends it.

Every element of this script (grid/stat/oracle API calls, calibration
pattern, representative-model lookup) follows exp1.py (S1 calibration and
application) and exp2.py (oracle bounds) exactly; nothing here invents a
different API.

PERFORMANCE: building Sigma_theta for the full 1,216-member family at one
record length takes ~90s (n_obs=820) and grows like n_obs^2 (n_obs reaches
~1,860 by 2046); doing this 8 times (4 record lengths x 2 volatility
scenarios), both for calibration and again inside `oracle_power_bound`'s
null-model loop, does not fit a ~12-minute budget. Strided subgrids of the
declared family (every 16th member for calibration/achieved-power, every
32nd for the oracle bound) are used instead; see the saved "notes" for the
exact sizes and the measured first-cell timing. Use --full to select the
paper-sized draw counts and the complete 1,216-member family (slow; not run
by default, see module docstring note below `if __name__`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# Must be set before numpy/BLAS is imported (see exp2.py's identical note):
# avoids BLAS spinning up background threads and oversubscribing this
# single-process, CPU-bound script.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.calibration import CalibrationSetup, calibrate_critical_value, precompute_m_sigma
from src.deviation_family import DeviationModel, build_grid, simulate_paths_batch
from src.direction_density import build_grid_cache, s1_statistic_batch
from src.oracle import oracle_power_bound
from src.projection import cosine_weights, project_m

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Representative nonzero cycle (middle of each declared grid of values), same as
# experiment 2 (exp2.py's REPR_CYCLE_R/REPR_CYCLE_PERIOD/REPR_CYCLE_KAPPA), copied
# here as literal values since module-level constants are not imported across exp
# scripts.
REPR_CYCLE_R = config.CYCLE_R_VALUES[1]  # 5.86
REPR_CYCLE_PERIOD = config.CYCLE_PERIOD_YEARS[1]  # 4.0 years
REPR_CYCLE_KAPPA = config.CYCLE_KAPPA_VALUES[1]  # 3
# The grid's only r=0 cycle (build_grid() has a single no-cycle entry).
NO_CYCLE_PERIOD = config.CYCLE_PERIOD_YEARS[0]
NO_CYCLE_KAPPA = config.CYCLE_KAPPA_VALUES[0]

# Reduced subgrids of the declared 1,216-member family (see module docstring
# PERFORMANCE note). Not specified by the paper; chosen for computational
# tractability of this exploratory, 8-cell (record length x volatility
# scenario) simulation study, documented in the saved "notes".
CALIBRATION_NULL_STRIDE_RUN = 16  # ~76 members, used for calibration AND the achieved-power cache
ORACLE_NULL_STRIDE_RUN = 32  # ~38 members, used only inside oracle_power_bound's null search

N_DRAWS_POWER_RUN = 100  # achieved-power Monte Carlo histories per (end_year, scenario, model)
N_DRAWS_ORACLE_RUN = 50  # oracle Monte Carlo draws per Neyman-Pearson test; a first pass with
# exp2.py's N_DRAWS_RUN=15 (chosen there because cost is dominated by per-null covariance
# construction, not n_draws) gave an oracle_bound estimate too noisy for this script's much
# smaller null subgrid (stderr ~0.13 at n_draws=15, causing oracle_bound < achieved_power in
# several cells from Monte Carlo noise alone); profiling showed raising n_draws to 50 only
# adds ~35% to the oracle cost (covariance construction still dominates), which fits the
# time budget and roughly halves the sampling noise (stderr ~0.07).

B_TRUE = config.PUBLISHED_EXPONENT  # paper's published exponent (Section 5), used for all power
# evaluations in this experiment (achieved power and the oracle bound).

DECLINE_FIT_YEARS = tuple(range(2017, 2027))  # "declining since the early volatile years"
# segment used to fit the extrapolation; not specified by the paper (see
# `declining_volatility_fit`).
VOLATILITY_FLOOR = 0.02  # floor for the extrapolated volatility, avoids a degenerate zero

CALIBRATION_SEED_BASE = 20261101  # new seed family for exp8 (does NOT reuse
# config.S1_CALIBRATION_SEED, which is reserved for the primary registered test).


def b_grid() -> np.ndarray:
    """B = {0, 0.75, ..., 30}, 41 points (paper Appendix B), exactly as exp1.py builds it."""
    return np.arange(0.0, config.B_GRID_MAX + config.B_GRID_STEP / 2, config.B_GRID_STEP)


def _find_model(grid: list[DeviationModel], kind: str, param: float, r: float,
                 period: float, kappa: int, lam: float) -> DeviationModel:
    """Locate the unique grid member matching the given fields (adapted from
    exp2.py's `_find_model` helper)."""
    for model in grid:
        if (model.persistence.kind == kind
                and math.isclose(model.persistence.param, param, abs_tol=1e-9)
                and math.isclose(model.cycle.r, r, abs_tol=1e-9)
                and math.isclose(model.cycle.period_years, period, abs_tol=1e-9)
                and model.cycle.kappa == kappa
                and math.isclose(model.lam, lam, abs_tol=1e-9)):
            return model
    raise ValueError(
        f"no grid model with kind={kind} param={param} r={r} period={period} "
        f"kappa={kappa} lam={lam}")


def representative_models(grid: list[DeviationModel]) -> list[tuple[str, DeviationModel]]:
    """The 3 representative deviation models (paper Section 5/Appendix J):
    (a) random walk + cycle, lambda=1; (b) AR c=5 + cycle, lambda=1;
    (c) random walk, lambda=0 (no cycle)."""
    rw_cycle = _find_model(grid, "ar", 0.0, REPR_CYCLE_R, REPR_CYCLE_PERIOD,
                            REPR_CYCLE_KAPPA, lam=1.0)
    ar5_cycle = _find_model(grid, "ar", 5.0, REPR_CYCLE_R, REPR_CYCLE_PERIOD,
                             REPR_CYCLE_KAPPA, lam=1.0)
    rw_lam0 = _find_model(grid, "ar", 0.0, 0.0, NO_CYCLE_PERIOD, NO_CYCLE_KAPPA, lam=0.0)
    return [
        ("random_walk_cycle_lam1", rw_cycle),
        ("ar_c5_cycle_lam1", ar5_cycle),
        ("random_walk_lam0", rw_lam0),
    ]


def build_anchors(end_year: int) -> np.ndarray:
    """Synthetic weekly Saturday age-in-weeks grid `s` from the paper's sample
    start to the last Saturday on/before 19 September of `end_year` (paper
    Section 2, Appendix J). Uses the exact genesis-relative age formula
    data_loader.build_weekly_series applies to the realized series (age_weeks
    = seconds from genesis to 23:59:59 UTC that Saturday, divided by a week),
    so end_year=2026 reproduces exp1.py's 820-anchor record exactly (checked
    in `run`). `pandas.date_range(..., freq="W-SAT")` continues the same
    weekly Saturday sequence for later years, with no realized-price gaps to
    drop since there is no data beyond 2026.
    """
    end_date = f"{end_year}-09-19"
    saturdays = pd.date_range(config.REALIZED_SAMPLE_START, end_date, freq="W-SAT")
    dropped = pd.Timestamp(config.PRIMARY_DROPPED_ANCHOR)
    genesis = pd.Timestamp(config.GENESIS_TIMESTAMP).tz_localize(None)
    s = np.array([
        (sat + pd.Timedelta(hours=23, minutes=59, seconds=59) - genesis).total_seconds()
        / (7 * 24 * 3600)
        for sat in saturdays if sat != dropped
    ])
    return s


def build_setup(end_year: int) -> CalibrationSetup:
    """Primary-test setup (X=[1,s], departure regressor log(s), paper Section 4)
    at the synthetic record length ending in `end_year`."""
    s = build_anchors(end_year)
    n = len(s)
    q = config.PROJECTION_Q_FEATURES[end_year]
    anchor_weeks = np.round(s).astype(int)
    n_full = int(anchor_weeks.max()) + 1
    X = np.column_stack([np.ones(n), s])
    Psi = cosine_weights(n, q)
    m_regressor = np.log(s)
    return CalibrationSetup(anchor_weeks=anchor_weeks, n_full=n_full, X=X, Psi=Psi, q=q,
                             m_regressor=m_regressor)


def declining_volatility_fit() -> tuple[float, float]:
    """OLS fit of log-volatility against calendar year over the declining
    decade 2017-2026 (not specified in the paper; chosen because this is the
    segment after the 2011-2016 settling-in period and before the roughly
    flat 2023-2026 tail). Returns (intercept, slope) of log(vol) = a + b*year.
    """
    years = np.array(DECLINE_FIT_YEARS, dtype=float)
    vols = np.array([config.ANNUAL_VOLATILITY[int(y)] for y in years])
    slope, intercept = np.polyfit(years, np.log(vols), 1)
    return float(intercept), float(slope)


def extend_volatility(scenario: str, up_to_year: int) -> dict[int, float]:
    """A copy of config.ANNUAL_VOLATILITY extended with the scenario's
    projected values for every year from 2027 to `up_to_year` (paper Appendix
    J: "flat at the 2026 level, or declining at the past rate")."""
    extended = dict(config.ANNUAL_VOLATILITY)
    last_year = max(config.ANNUAL_VOLATILITY)
    if scenario == "flat":
        flat_value = config.ANNUAL_VOLATILITY[2026]
        for y in range(last_year + 1, up_to_year + 1):
            extended[y] = flat_value
    elif scenario == "declining":
        intercept, slope = declining_volatility_fit()
        for y in range(last_year + 1, up_to_year + 1):
            extended[y] = max(float(np.exp(intercept + slope * y)), VOLATILITY_FLOOR)
    else:
        raise ValueError(scenario)
    return extended


def achieved_power(model: DeviationModel, setup: CalibrationSetup, cache, bgrid: np.ndarray,
                    critical_value: float, rng: np.random.Generator, n_draws: int) -> float:
    """Monte-Carlo achieved power of S1 at b=B_TRUE under `model` (paper
    Section 5/Appendix J): simulate `n_draws` full-grid histories, project
    them the same way `src.calibration.simulate_s1_under_null` projects null
    draws but adding the b*m alternative mean (as `src.oracle.simulate_z`
    does per-draw), batch the S1 evaluation, and take the empirical rejection
    rate against `critical_value`.
    """
    u_full = simulate_paths_batch(model, setup.n_full, n_draws, rng)  # (n_draws, n_full)
    u_anchor = u_full[:, setup.anchor_weeks]  # (n_draws, n_obs)
    m = project_m(setup.m_regressor, setup.X, setup.Psi)
    XtX_inv = np.linalg.inv(setup.X.T @ setup.X)
    L = setup.Psi.T @ (np.eye(setup.X.shape[0]) - setup.X @ XtX_inv @ setup.X.T)  # (q, n_obs)
    z_batch = u_anchor @ L.T + B_TRUE * m[None, :]
    s1 = s1_statistic_batch(z_batch, cache, bgrid, setup.q)
    return float(np.mean(s1 > critical_value))


def run(full: bool = False) -> dict:
    t0 = time.time()
    grid = build_grid()
    bgrid = b_grid()
    reps = representative_models(grid)

    calib_stride = 1 if full else CALIBRATION_NULL_STRIDE_RUN
    oracle_stride = 1 if full else ORACLE_NULL_STRIDE_RUN
    n_draws_power = config.DIAGNOSTICS_N_HISTORIES_PAPER if full else N_DRAWS_POWER_RUN
    n_draws_oracle = config.ORACLE_N_SIMULATIONS_PAPER if full else N_DRAWS_ORACLE_RUN
    n_screen = config.CALIBRATION_SCREEN_DRAWS_PAPER if full else config.CALIBRATION_SCREEN_DRAWS_RUN
    n_refine = config.CALIBRATION_REFINE_DRAWS_PAPER if full else config.CALIBRATION_REFINE_DRAWS_RUN
    n_refine_models = (config.CALIBRATION_N_REFINE_PAPER if full
                       else config.CALIBRATION_N_REFINE_RUN)

    calib_null_grid = grid[::calib_stride]
    oracle_null_grid = grid[::oracle_stride]

    n_observations: dict[str, int] = {}
    results: list[dict] = []
    first_cell_seconds: float | None = None

    for end_year in config.PROJECTION_END_YEARS:
        setup = build_setup(end_year)
        n_observations[str(end_year)] = len(setup.m_regressor)
        m = project_m(setup.m_regressor, setup.X, setup.Psi)  # model-independent

        for scenario in ("flat", "declining"):
            original_volatility = dict(config.ANNUAL_VOLATILITY)
            try:
                extended_volatility = extend_volatility(scenario, end_year)
                config.ANNUAL_VOLATILITY.update(extended_volatility)

                t_cell = time.time()
                seed = CALIBRATION_SEED_BASE + end_year + (0 if scenario == "flat" else 100000)
                m_list, sigma_list = precompute_m_sigma(setup, calib_null_grid)
                calib = calibrate_critical_value(
                    setup, calib_null_grid, m_list, sigma_list, bgrid, seed=seed,
                    n_screen=n_screen, n_refine=n_refine, n_refine_models=n_refine_models,
                )
                critical_value = calib["critical_value"]
                cache = build_grid_cache(m_list, sigma_list)

                for i, (name, model) in enumerate(reps):
                    rng_power = np.random.default_rng(seed + 1000 + i)
                    power = achieved_power(model, setup, cache, bgrid, critical_value,
                                            rng_power, n_draws_power)

                    rng_oracle = np.random.default_rng(seed + 2000 + i)
                    bound, worst_idx = oracle_power_bound(
                        model, oracle_null_grid, B_TRUE, m, setup, rng_oracle,
                        n_draws_oracle, alpha=config.ORACLE_ALPHA)

                    results.append({
                        "end_year": end_year,
                        "volatility_scenario": scenario,
                        "model": name,
                        "critical_value": critical_value,
                        "achieved_power": power,
                        "oracle_bound": bound,
                        "worst_null_index": worst_idx,
                        "n_draws_power": n_draws_power,
                        "n_null_grid": len(oracle_null_grid),
                    })

                cell_seconds = time.time() - t_cell
                if first_cell_seconds is None:
                    first_cell_seconds = cell_seconds
                    projected_total = cell_seconds * len(config.PROJECTION_END_YEARS) * 2
                    print(f"first cell (end_year={end_year}, scenario={scenario}) took "
                          f"{cell_seconds:.1f}s (calibration + {len(reps)} models' achieved "
                          f"power + oracle bound); naive linear extrapolation to all "
                          f"{len(config.PROJECTION_END_YEARS) * 2} cells: "
                          f"{projected_total:.1f}s (record length grows faster than linear, "
                          f"so later cells cost more than this estimate)", flush=True)
            finally:
                config.ANNUAL_VOLATILITY.clear()
                config.ANNUAL_VOLATILITY.update(original_volatility)

    elapsed = time.time() - t0

    # Plot: one line per (scenario x bound-type), power averaged over the 3
    # representative models (paper Appendix J projects a single power-vs-record-
    # length curve per scenario; this experiment evaluates 3 alternatives, so the
    # plot shows their mean to keep one line per scenario/bound-type as requested).
    fig, ax = plt.subplots(figsize=(7, 5))
    years = list(config.PROJECTION_END_YEARS)
    for scenario in ("flat", "declining"):
        for bound_type, key in (("achieved power", "achieved_power"), ("oracle bound", "oracle_bound")):
            y = []
            for end_year in years:
                rows = [r[key] for r in results
                        if r["end_year"] == end_year and r["volatility_scenario"] == scenario]
                y.append(float(np.mean(rows)))
            style = "-" if bound_type == "achieved power" else "--"
            ax.plot(years, y, style, marker="o", label=f"{scenario}, {bound_type}")
    ax.set_xlabel("record end year")
    ax.set_ylabel("power at b=5.7 (mean over 3 representative models)")
    ax.set_ylim(0, 1)
    ax.set_title("S1 power vs. record length (paper Appendix J, exploratory)")
    ax.legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp8_power_curves.png", dpi=150)
    plt.close(fig)

    out = {
        "end_years": list(config.PROJECTION_END_YEARS),
        "q_features": {str(y): config.PROJECTION_Q_FEATURES[y] for y in config.PROJECTION_END_YEARS},
        "n_observations": n_observations,
        "volatility_scenarios": ["flat", "declining"],
        "results": results,
        "paper_oracle_range_2026": [0.34, 0.43],
        "paper_achieved_power_2026": config.PAPER_VALIDATED_POWER_AT_5_7,
        "paper_oracle_range_2046": [0.64, 0.86],
        "notes": {
            "exploratory": (
                "Experiment 8 is an exploratory projection (paper Appendix J), not part "
                "of the frozen registered protocol."
            ),
            "synthetic_anchors": (
                "Record lengths beyond 2026 have no realized price data; anchors are a "
                "synthetic weekly Saturday grid (pandas.date_range(..., freq='W-SAT')) using "
                "the same genesis-relative age-in-weeks formula as data_loader.build_weekly_"
                "series. end_year=2026 reproduces exp1/exp2's 820-anchor, q=6 setup exactly."
            ),
            "calibration_null_subgrid": (
                f"Calibration (and the achieved-power statistic's GridCache) uses every "
                f"{CALIBRATION_NULL_STRIDE_RUN}th member of the declared 1,216-model family "
                f"({len(calib_null_grid)} members), not the full family: building Sigma_theta "
                f"for all 1,216 members at one record length alone measured ~90s at n_obs=820 "
                f"and scales like n_obs^2 (n_obs reaches ~1,860 by 2046); doing this for 8 "
                f"(record length x volatility scenario) cells would not fit the time budget. "
                f"Use --full for the complete family."
            ),
            "oracle_null_subgrid": (
                f"The oracle bound uses a further-reduced subgrid, every "
                f"{ORACLE_NULL_STRIDE_RUN}th member ({len(oracle_null_grid)} members): "
                f"oracle_power_bound's cost is dominated by per-null covariance construction "
                f"(same O(n_obs^2) cost as above), repeated independently for each of the 3 "
                f"representative alternatives at each of the 8 cells. Use --full for the "
                f"complete family."
            ),
            "n_draws_power": (
                f"{n_draws_power} Monte-Carlo histories per (end_year, scenario, model) "
                f"achieved-power estimate (reduced from a starting point of 150 to fit the "
                f"time budget; --full uses config.DIAGNOSTICS_N_HISTORIES_PAPER)."
            ),
            "n_draws_oracle": (
                f"{n_draws_oracle} draws per Neyman-Pearson test inside the oracle search: "
                f"oracle cost is dominated by per-null covariance construction, not n_draws "
                f"(exp2.py's N_DRAWS_RUN note), so this stays small even in --full."
            ),
            "calibration_draws": {
                "n_screen": n_screen, "n_refine": n_refine, "n_refine_models": n_refine_models,
                "note": "config.CALIBRATION_*_RUN (or *_PAPER with --full), as in exp1.py.",
            },
            "seed_family": (
                f"seeds = {CALIBRATION_SEED_BASE} + end_year (+100000 for the declining "
                f"scenario); a new seed family for this experiment, distinct from "
                f"config.S1_CALIBRATION_SEED (reserved for the primary registered test)."
            ),
            "representative_cycle": {
                "r": REPR_CYCLE_R, "period_years": REPR_CYCLE_PERIOD, "kappa": REPR_CYCLE_KAPPA,
                "note": "same representative cycle as experiment 2",
            },
            "representative_models": (
                "(a) random_walk_cycle_lam1: AR c=0 (random walk) + the representative cycle, "
                "lambda=1; (b) ar_c5_cycle_lam1: AR c=5 + the representative cycle, lambda=1; "
                "(c) random_walk_lam0: AR c=0, no cycle, lambda=0 (paper Section 5/Appendix J)."
            ),
            "volatility_scenarios": {
                "flat": "future annual volatility held at config.ANNUAL_VOLATILITY[2026].",
                "declining": (
                    f"OLS fit of log-volatility against year over {DECLINE_FIT_YEARS[0]}-"
                    f"{DECLINE_FIT_YEARS[-1]} (not specified by the paper beyond 'declining at "
                    f"the past rate'), extrapolated forward and floored at {VOLATILITY_FLOOR}."
                ),
                "implementation": (
                    "src.deviation_family.sigma_profile reads config.ANNUAL_VOLATILITY as a "
                    "module global when no override is passed (which is how model_covariance/"
                    "simulate_path/simulate_paths_batch call it); src/config.py and src/"
                    "deviation_family.py may not be edited, so each scenario's extra years "
                    "(2027..end_year) are added to config.ANNUAL_VOLATILITY in place for the "
                    "duration of that cell's simulations only, saved and restored via "
                    "try/finally immediately afterward. config.CALIBRATION_SUPPORT_YEARS "
                    "(captured at import time from the original 2011-2026 keys) is untouched, "
                    "so the volatility-shape RMS normalization stays fixed across every record "
                    "length and scenario, matching its documented 'held fixed' design."
                ),
            },
            "b_grid": "np.arange(0, 30+0.375, 0.75), 41 points, exactly as exp1.py's b_grid().",
            "plot": (
                "exp8_power_curves.png averages achieved power and the oracle bound over the "
                "3 representative models at each (end_year, scenario), to show exactly one "
                "line per (scenario x bound-type) as requested."
            ),
            "first_cell_seconds": first_cell_seconds,
        },
        "full_paper_configuration": full,
        "elapsed_seconds": elapsed,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp8_power_vs_record_length.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper-sized draw counts and the complete 1,216-member "
                              "family for both calibration and the oracle bound (slow: not "
                              "run by default, see module PERFORMANCE note)")
    args = parser.parse_args()
    run(full=args.full)
