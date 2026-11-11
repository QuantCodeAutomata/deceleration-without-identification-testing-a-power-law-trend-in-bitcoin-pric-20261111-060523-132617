"""Experiment 2: oracle power bounds for identifiability of the trend and
reversion questions (paper Section 5, Appendix J).

For the TREND question, the alternative is a power-law trend (b=5.7) plus one
representative deviation model; the null is constant drift (b=0) plus any
member of the declared 1,216-model family. The oracle upper bound on power
for one alternative is the smallest Neyman-Pearson power over every null in
the family (the worst-case null the data could not rule out).

For the REVERSION question, the alternatives are mean-reverting AR deviation
models (half-lives 0.55, 1.1, 2.2 years, each with a cycle) and the nulls are
restricted to the non-reverting (random-walk, c=0) members of the family.

Both bounds reuse `src.oracle.oracle_power_bound`, which already implements
the Neyman-Pearson logic and the worst-null search; this script only selects
the representative alternative models from the declared grid and wires the
primary setup (X=[1,s], departure regressor log(s), q=6 cosine features) to
the current 2026 record length.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

# Must be set before numpy/BLAS is imported anywhere in this process: OpenBLAS/MKL spin up
# background worker threads on first use, and fork()-ing a multi-threaded process (the
# default ProcessPoolExecutor start method on Linux) after that can deadlock in the child
# (locks held by threads that do not survive the fork). Forcing single-threaded BLAS here
# keeps each process single-threaded, which is fork-safe and also avoids CPU oversubscription
# when several worker processes run matrix ops concurrently on this 4-CPU machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data.data_loader import load_primary_weekly_series
from src import config
from src.calibration import CalibrationSetup, precompute_m_sigma
from src.deviation_family import DeviationModel, build_grid
from src.oracle import oracle_power_bound
from src.pipeline import primary_setup

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Representative nonzero cycle (middle of each declared grid of values), shared by every
# alternative model below that is described in the paper as "with a cycle".
REPR_CYCLE_R = config.CYCLE_R_VALUES[1]  # 5.86
REPR_CYCLE_PERIOD = config.CYCLE_PERIOD_YEARS[1]  # 4.0 years
REPR_CYCLE_KAPPA = config.CYCLE_KAPPA_VALUES[1]  # 3
# The grid's only r=0 cycle (build_grid() has a single no-cycle entry).
NO_CYCLE_PERIOD = config.CYCLE_PERIOD_YEARS[0]
NO_CYCLE_KAPPA = config.CYCLE_KAPPA_VALUES[0]

# Profiling (see report) found oracle_power_bound's cost against the full 1,216-member
# null grid is dominated by per-null covariance construction (~0.1-0.2s/null, essentially
# independent of n_draws), not by the Monte Carlo draws themselves. config.ORACLE_N_
# SIMULATIONS_RUN (2000) would therefore take >30 minutes per scenario on the full grid;
# reduced further here, not specified in the paper, chosen for computational feasibility.
N_DRAWS_RUN = 15
N_DRAWS_FULL = config.ORACLE_N_SIMULATIONS_PAPER

WEEKS_PER_YEAR = config.WEEKS_PER_YEAR


def _find_model(grid: list[DeviationModel], kind: str, param: float, r: float,
                 period: float, kappa: int, lam: float) -> DeviationModel:
    """Locate the unique grid member matching the given fields."""
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


def half_life_weeks(c: float) -> float:
    """Half-life (weeks) implied by AR persistence parameter c (rho = 1 - c/821)."""
    rho = 1.0 - c / config.AR_SCALE
    if rho >= 1.0:
        return math.inf
    return math.log(0.5) / math.log(rho)


def closest_ar_c_for_half_life(target_weeks: float) -> float:
    """The declared AR_C_VALUES member (excluding c=0, a non-reverting random walk)
    whose implied half-life is closest to `target_weeks`."""
    candidates = [c for c in config.AR_C_VALUES if c > 0]
    return min(candidates, key=lambda c: abs(half_life_weeks(c) - target_weeks))


def trend_alt_models(grid: list[DeviationModel]) -> list[dict]:
    """The 3 representative TREND alternatives (paper Section 5)."""
    rw_cycle = _find_model(grid, "ar", 0.0, REPR_CYCLE_R, REPR_CYCLE_PERIOD,
                            REPR_CYCLE_KAPPA, lam=1.0)
    target_weeks = 2.2 * WEEKS_PER_YEAR
    c_reverting = closest_ar_c_for_half_life(target_weeks)
    reverting_cycle = _find_model(grid, "ar", c_reverting, REPR_CYCLE_R, REPR_CYCLE_PERIOD,
                                   REPR_CYCLE_KAPPA, lam=1.0)
    rw_const_vol = _find_model(grid, "ar", 0.0, 0.0, NO_CYCLE_PERIOD, NO_CYCLE_KAPPA, lam=0.0)
    return [
        {"description": "random walk with a cycle (c=0, lam=1)", "model": rw_cycle},
        {"description": (f"slowly reverting AR, half-life~{half_life_weeks(c_reverting):.1f} "
                          f"weeks (c={c_reverting}), with a cycle"), "model": reverting_cycle},
        {"description": "random walk with constant volatility (c=0, r=0, lam=0)",
         "model": rw_const_vol},
    ]


def reversion_alt_models(grid: list[DeviationModel]) -> list[dict]:
    """The 3 mean-reverting REVERSION alternatives, each with the representative cycle."""
    out = []
    for half_life_years in (0.55, 1.1, 2.2):
        target_weeks = half_life_years * WEEKS_PER_YEAR
        c = closest_ar_c_for_half_life(target_weeks)
        model = _find_model(grid, "ar", c, REPR_CYCLE_R, REPR_CYCLE_PERIOD,
                             REPR_CYCLE_KAPPA, lam=1.0)
        out.append({"half_life_years": half_life_years, "ar_c": c,
                    "implied_half_life_weeks": half_life_weeks(c), "model": model})
    return out


def reversion_null_grid(grid: list[DeviationModel]) -> list[DeviationModel]:
    """Non-reverting null family: the AR, c=0 (random-walk) members of the declared
    grid, across every cycle and volatility-shape variation (76 of 1,216 members).
    Simplifying choice (paper does not pin down this subgrid precisely): restricting
    to AR c=0 keeps the null family unambiguously "non-reverting" and the search
    tractable; see 'assumptions' in the saved result.
    """
    return [m for m in grid if m.persistence.kind == "ar" and m.persistence.param == 0.0]


def _run_scenario(label: str, alt_model: DeviationModel, null_grid: list[DeviationModel],
                   b: float, m: np.ndarray, setup: CalibrationSetup, seed: int,
                   n_draws: int, alpha: float) -> dict:
    rng = np.random.default_rng(seed)
    bound, worst_idx = oracle_power_bound(alt_model, null_grid, b, m, setup, rng, n_draws, alpha)
    return {"label": label, "bound": bound, "worst_null_index": worst_idx}


def run(full: bool = False) -> dict:
    t0 = time.time()
    df = load_primary_weekly_series(
        config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END,
        dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,),
    )
    setup = primary_setup(df, config.Q_FEATURES_PRIMARY)
    grid = build_grid()
    # m is identical for every grid member (it depends only on setup.m_regressor), so
    # one call with an arbitrary single-model list gives the shared m-vector.
    m_list, _ = precompute_m_sigma(setup, [grid[0]])
    m = m_list[0]

    n_draws = N_DRAWS_FULL if full else N_DRAWS_RUN
    alpha = config.ORACLE_ALPHA
    b = config.PUBLISHED_EXPONENT

    trend_alts = trend_alt_models(grid)
    reversion_alts = reversion_alt_models(grid)
    null_grid_reversion = reversion_null_grid(grid)

    tasks = []
    for i, alt in enumerate(trend_alts):
        tasks.append(("trend", i, alt["description"], alt["model"], grid, b, seed := 1000 + i))
    for i, alt in enumerate(reversion_alts):
        tasks.append(("reversion", i, f"half_life={alt['half_life_years']}yr",
                      alt["model"], null_grid_reversion, 0.0, 2000 + i))

    results_by_key: dict[tuple[str, int], dict] = {}
    max_workers = min(4, len(tasks))
    # spawn, not the Linux default fork: see the BLAS/fork-safety note above.
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
        futures = {
            pool.submit(_run_scenario, f"{kind}:{idx}", model, null_grid, b_val, m, setup,
                        seed, n_draws, alpha): (kind, idx, desc)
            for kind, idx, desc, model, null_grid, b_val, seed in tasks
        }
        for fut, (kind, idx, desc) in futures.items():
            res = fut.result()
            results_by_key[(kind, idx)] = {"description": desc, **res}

    trend_oracle_bounds = []
    for i, alt in enumerate(trend_alts):
        res = results_by_key[("trend", i)]
        trend_oracle_bounds.append({
            "description": alt["description"], "b": b,
            "bound": res["bound"], "worst_null_index": res["worst_null_index"],
        })

    reversion_oracle_bounds = []
    for i, alt in enumerate(reversion_alts):
        res = results_by_key[("reversion", i)]
        reversion_oracle_bounds.append({
            "half_life_years": alt["half_life_years"], "ar_c": alt["ar_c"],
            "bound": res["bound"], "worst_null_index": res["worst_null_index"],
        })

    elapsed = time.time() - t0
    out = {
        "n_observations": len(df),
        "sample_start": config.REALIZED_SAMPLE_START,
        "sample_end": config.PRIMARY_SAMPLE_END,
        "q_features": config.Q_FEATURES_PRIMARY,
        "n_draws": n_draws,
        "alpha": alpha,
        "trend_oracle_bounds": trend_oracle_bounds,
        "reversion_oracle_bounds": reversion_oracle_bounds,
        "n_null_grid_trend": len(grid),
        "n_null_grid_reversion": len(null_grid_reversion),
        "assumptions": {
            "representative_cycle": {"r": REPR_CYCLE_R, "period_years": REPR_CYCLE_PERIOD,
                                      "kappa": REPR_CYCLE_KAPPA},
            "reversion_null_family": (
                "restricted to AR, c=0 (random-walk) members of the declared grid "
                "across all cycle/volatility-shape variations (76 of 1216 members), "
                "not the full 1216-member family, as a tractable, unambiguously "
                "non-reverting null subgrid."),
            "n_draws_reduced": (
                "n_draws reduced below config.ORACLE_N_SIMULATIONS_RUN for the "
                "non-full run: oracle_power_bound's cost against a 1216-member null "
                "grid is dominated by per-null covariance construction, not n_draws; "
                "use --full for config.ORACLE_N_SIMULATIONS_PAPER draws."),
        },
        "full_paper_configuration": full,
        "elapsed_seconds": elapsed,
        "paper_trend_range": [0.34, 0.43],
        "paper_reversion_range": [0.08, 0.20],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp2_oracle_bounds.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full Monte Carlo draw count (slow)")
    args = parser.parse_args()
    run(full=args.full)
