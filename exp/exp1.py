"""Experiment 1: primary S1 test of constant drift vs. a positive log-age
trend under persistent deviations (paper Section 4, Box 1, Appendix B/C).

Pipeline: build the 820-anchor weekly Bitcoin log-price series, project it
with the cosine low-frequency basis after removing [1, s], calibrate S1's
critical value by simulation under the declared 1,216-member null family,
apply S1 to the data, and invert the test to get a lower confidence bound
for the log-age coefficient b.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data.data_loader import load_primary_weekly_series
from src import config
from src.calibration import calibrate_critical_value
from src.deviation_family import build_grid
from src.pipeline import apply_s1, confidence_bound_for_b, primary_setup

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def b_grid() -> np.ndarray:
    """B = {0, 0.75, ..., 30}, 41 points (paper Appendix B)."""
    return np.arange(0.0, config.B_GRID_MAX + config.B_GRID_STEP / 2, config.B_GRID_STEP)


def run(full: bool = False) -> dict:
    t0 = time.time()
    df = load_primary_weekly_series(
        config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END,
        dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,),
    )
    n_obs = len(df)
    grid = build_grid()
    bgrid = b_grid()

    setup = primary_setup(df, config.Q_FEATURES_PRIMARY)
    from src.calibration import precompute_m_sigma
    m_list, sigma_list = precompute_m_sigma(setup, grid)

    n_screen = config.CALIBRATION_SCREEN_DRAWS_PAPER if full else config.CALIBRATION_SCREEN_DRAWS_RUN
    n_refine = config.CALIBRATION_REFINE_DRAWS_PAPER if full else config.CALIBRATION_REFINE_DRAWS_RUN
    n_refine_models = config.CALIBRATION_N_REFINE_PAPER if full else config.CALIBRATION_N_REFINE_RUN

    calib = calibrate_critical_value(
        setup, grid, m_list, sigma_list, bgrid, seed=config.S1_CALIBRATION_SEED,
        n_screen=n_screen, n_refine=n_refine, n_refine_models=n_refine_models,
    )
    critical_value = calib["critical_value"]

    result = apply_s1(df, grid, bgrid, config.Q_FEATURES_PRIMARY)
    s1 = result["s1"]
    reject = bool(s1 > critical_value)

    bound = confidence_bound_for_b(df, grid, bgrid, critical_value, config.Q_FEATURES_PRIMARY)
    lower_bound = bound["lower_bound"]
    b_5_7_accepted = bool(np.any(np.isclose(bound["accepted_trials"], config.PUBLISHED_EXPONENT)))

    elapsed = time.time() - t0
    out = {
        "n_observations": n_obs,
        "sample_start": config.REALIZED_SAMPLE_START,
        "sample_end": config.PRIMARY_SAMPLE_END,
        "q_features": config.Q_FEATURES_PRIMARY,
        "n_grid_models": len(grid),
        "calibration": {
            "critical_value": critical_value,
            "largest_refined_quantile": calib["largest_refined_quantile"],
            "n_screen": n_screen, "n_refine": n_refine, "n_refine_models": n_refine_models,
            "seed": config.S1_CALIBRATION_SEED,
        },
        "s1_statistic": s1,
        "reject_constant_drift": reject,
        "lower_confidence_bound_b": lower_bound,
        "b_5_7_accepted": b_5_7_accepted,
        "paper_s1": config.PAPER_S1_STATISTIC,
        "paper_critical_value": config.PAPER_S1_CRITICAL_VALUE,
        "paper_lower_bound": config.PAPER_LOWER_BOUND_B,
        "full_paper_configuration": full,
        "elapsed_seconds": elapsed,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp1_primary_s1_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's exact calibration draw counts")
    args = parser.parse_args()
    run(full=args.full)
