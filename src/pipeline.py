"""Shared pipeline tying together data, the declared family, projection,
calibration and the statistic, for the primary test (exp_1), the secondary
tests (exp_3), and the sensitivity analyses (exp_4).

All experiments that apply S1 (or its secondary-test analogue) to a weekly
log-price record call these functions, so the construction is identical
everywhere except for the one element each analysis is designed to change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.calibration import CalibrationSetup, calibrate_critical_value, precompute_m_sigma
from src.deviation_family import DeviationModel, build_grid, model_covariance
from src.direction_density import s1_statistic, two_sided_statistic
from src.lower_bound import lower_confidence_bound, trial_grid
from src.projection import apply_residualizer, cosine_weights, project_m, project_z, sigma_theta


def primary_design(df: pd.DataFrame) -> np.ndarray:
    """X = [1, s] (intercept, linear age): removed before testing for the log-age
    term (paper Section 4)."""
    s = df["age_weeks"].to_numpy()
    return np.column_stack([np.ones_like(s), s])


def power_law_design(df: pd.DataFrame) -> np.ndarray:
    """X = [1, log s] (intercept, log-age): removed for the secondary tests that
    start from power-law residuals (paper Section 4, exp_3)."""
    log_s = np.log(df["age_weeks"].to_numpy())
    return np.column_stack([np.ones_like(log_s), log_s])


def build_setup(df: pd.DataFrame, design_fn, m_regressor: np.ndarray, q: int,
                anchor_week_rounding: str = "nearest") -> CalibrationSetup:
    """Shared setup: weekly latent grid indices for each anchor, cosine basis,
    and the design matrix to residualize on.
    """
    n = len(df)
    anchor_weeks = np.round(df["age_weeks"].to_numpy()).astype(int)
    n_full = int(anchor_weeks.max()) + 1
    Psi = cosine_weights(n, q)
    X = design_fn(df)
    return CalibrationSetup(anchor_weeks=anchor_weeks, n_full=n_full, X=X, Psi=Psi, q=q,
                             m_regressor=m_regressor)


def primary_setup(df: pd.DataFrame, q: int = config.Q_FEATURES_PRIMARY) -> CalibrationSetup:
    """The primary test's setup: X=[1,s], departure regressor log(s) (paper Section 4)."""
    log_s = np.log(df["age_weeks"].to_numpy())
    return build_setup(df, primary_design, log_s, q)


def secondary_setup(df: pd.DataFrame, departure: str, q: int = config.Q_FEATURES_PRIMARY
                     ) -> tuple[CalibrationSetup, np.ndarray]:
    """Secondary-test setup: X=[1,log s], departure regressor v (paper Section 4, exp_3).

    `v` is scaled to unit range after removing its own best power-law fit
    (the paper's "scaled to unit range after removing its best power-law fit",
    Section 4; the exact scaling beyond "unit range" is not specified, so the
    regressor is divided by its own max absolute value after this removal).
    """
    s = df["age_weeks"].to_numpy()
    log_s = np.log(s)
    if departure == "drift":
        v_raw = s
    elif departure == "curvature":
        v_raw = (log_s - log_s.mean()) ** 2
    else:
        raise ValueError(departure)
    X_power_law = power_law_design(df)
    v_resid = apply_residualizer(X_power_law, v_raw)
    v_scaled = v_resid / np.max(np.abs(v_resid))
    setup = build_setup(df, power_law_design, v_scaled, q)
    return setup, v_scaled


def application_vectors(setup: CalibrationSetup, y: np.ndarray, grid: list[DeviationModel]
                         ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """z, and the per-model (m, Sigma) lists, for the observed data."""
    z = project_z(y, setup.X, setup.Psi)
    m_list, sigma_list = precompute_m_sigma(setup, grid)
    return z, m_list, sigma_list


def apply_s1(df: pd.DataFrame, grid: list[DeviationModel], b_grid: np.ndarray,
             q: int = config.Q_FEATURES_PRIMARY) -> dict:
    """Apply the primary S1 statistic to the observed weekly series (paper Section 4)."""
    setup = primary_setup(df, q)
    y = df["y"].to_numpy()
    z, m_list, sigma_list = application_vectors(setup, y, grid)
    s1, null_idx, alt_ll = s1_statistic(z, m_list, sigma_list, b_grid, q)
    return {"s1": s1, "z": z, "setup": setup, "m_list": m_list, "sigma_list": sigma_list,
            "best_null_idx": null_idx}


def apply_two_sided(df: pd.DataFrame, departure: str, grid: list[DeviationModel],
                     b_grid_pos: np.ndarray, q: int = config.Q_FEATURES_PRIMARY) -> dict:
    """Apply a secondary (extra-drift or curvature) two-sided statistic (paper
    Section 4, exp_3)."""
    setup, v_scaled = secondary_setup(df, departure, q)
    y = df["y"].to_numpy()
    z, m_list, sigma_list = application_vectors(setup, y, grid)
    stat, null_idx, alt_ll = two_sided_statistic(z, m_list, sigma_list, b_grid_pos, q)
    return {"statistic": stat, "z": z, "setup": setup, "m_list": m_list,
            "sigma_list": sigma_list, "best_null_idx": null_idx}


def confidence_bound_for_b(df: pd.DataFrame, grid: list[DeviationModel], b_grid: np.ndarray,
                            critical_value: float, q: int = config.Q_FEATURES_PRIMARY) -> dict:
    """Lower 95% confidence bound for b from the primary test (paper Appendix B)."""
    result = apply_s1(df, grid, b_grid, q)
    trials = trial_grid()
    L, accepted = lower_confidence_bound(result["z"], result["m_list"], result["sigma_list"],
                                          b_grid, q, critical_value, trials)
    return {"lower_bound": L, "accepted_trials": accepted, **result}
