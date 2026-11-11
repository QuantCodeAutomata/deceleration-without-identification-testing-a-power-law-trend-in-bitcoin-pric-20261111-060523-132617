"""Experiment 3: secondary tests for extra drift and curvature in log age,
starting from power-law residuals (paper Section 4, Appendix B/C).

Both tests residualize on [1, log s] (removing a pure power-law fit) instead
of [1, s], then look for a departure in one of two directions: an extra
linear-age drift, or curvature in log age. Each test is two-sided (the
coefficient on the departure regressor may be positive or negative) and is
calibrated by the same simulation rule as the primary S1 test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data.data_loader import load_primary_weekly_series
from src import config
from src.calibration import calibrate_critical_value, precompute_m_sigma
from src.deviation_family import build_grid
from src.direction_density import two_sided_statistic, two_sided_statistic_batch
from src.pipeline import apply_two_sided, secondary_setup

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def b_grid_pos() -> np.ndarray:
    """Positive half of the +/-B grid for the departure coefficient. The paper
    references '+/-B' without stating the grid; the same spacing and range as
    the primary test's B = {0, 0.75, ..., 30} is used (Appendix B), applied to
    both signs through the two-sided statistic's symmetry."""
    return np.arange(0.0, config.B_GRID_MAX + config.B_GRID_STEP / 2, config.B_GRID_STEP)


def run(full: bool = False) -> dict:
    t0 = time.time()
    df = load_primary_weekly_series(
        config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END,
        dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,),
    )
    n_obs = len(df)
    grid = build_grid()
    bgrid = b_grid_pos()

    n_screen = config.CALIBRATION_SCREEN_DRAWS_PAPER if full else config.CALIBRATION_SCREEN_DRAWS_RUN
    n_refine = config.CALIBRATION_REFINE_DRAWS_PAPER if full else config.CALIBRATION_REFINE_DRAWS_RUN
    n_refine_models = config.CALIBRATION_N_REFINE_PAPER if full else config.CALIBRATION_N_REFINE_RUN

    out = {
        "n_observations": n_obs,
        "q_features": config.Q_FEATURES_PRIMARY,
        "n_grid_models": len(grid),
        "full_paper_configuration": full,
        "tests": {},
    }

    seeds = {"drift": config.DRIFT_TEST_SEED, "curvature": config.CURVATURE_TEST_SEED}
    paper_cv = {"drift": config.PAPER_DRIFT_CRITICAL_VALUE,
                "curvature": config.PAPER_CURVATURE_CRITICAL_VALUE}

    for departure in ("drift", "curvature"):
        setup, v_scaled = secondary_setup(df, departure, config.Q_FEATURES_PRIMARY)
        m_list, sigma_list = precompute_m_sigma(setup, grid)

        calib = calibrate_critical_value(
            setup, grid, m_list, sigma_list, bgrid, seed=seeds[departure],
            n_screen=n_screen, n_refine=n_refine, n_refine_models=n_refine_models,
            statistic_fn=two_sided_statistic_batch,
        )
        critical_value = calib["critical_value"]

        result = apply_two_sided(df, departure, grid, bgrid, config.Q_FEATURES_PRIMARY)
        statistic = result["statistic"]
        reject = bool(statistic > critical_value)

        out["tests"][departure] = {
            "critical_value": critical_value,
            "largest_refined_quantile": calib["largest_refined_quantile"],
            "seed": seeds[departure],
            "statistic": statistic,
            "reject_power_law_adequacy": reject,
            "is_zero": bool(np.isclose(statistic, 0.0)),
            "paper_critical_value": paper_cv[departure],
        }

    elapsed = time.time() - t0
    out["elapsed_seconds"] = elapsed
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp3_secondary_tests.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's exact calibration draw counts")
    args = parser.parse_args()
    run(full=args.full)
