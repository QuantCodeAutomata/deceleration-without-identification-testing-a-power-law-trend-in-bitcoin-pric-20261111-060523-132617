"""Simulation-based calibration of the S1 critical value (paper Section 4, Appendix C).

Custom implementation: no general library runs this specific two-stage
screen-then-refine Monte-Carlo calibration with an order-statistic upper
confidence limit for a quantile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src import config
from src.deviation_family import DeviationModel, model_covariance, simulate_paths_batch
from src.direction_density import GridCache, build_grid_cache, s1_statistic_batch
from src.projection import cosine_weights, project_m, sigma_theta


@dataclass
class CalibrationSetup:
    """Pre-computed, model-independent pieces shared by every simulation draw."""
    anchor_weeks: np.ndarray
    n_full: int
    X: np.ndarray  # design matrix removed before projection, e.g. [1, s] or [1, log s]
    Psi: np.ndarray
    q: int
    m_regressor: np.ndarray  # the regressor whose projection gives m (e.g. log(s))


def precompute_m_sigma(setup: CalibrationSetup, grid: list[DeviationModel]
                        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """m_theta and Sigma_theta for every grid member (independent of simulated data)."""
    m_list, sigma_list = [], []
    m_proj = project_m(setup.m_regressor, setup.X, setup.Psi)
    for model in grid:
        omega = model_covariance(model, setup.anchor_weeks, setup.n_full)
        sigma_list.append(sigma_theta(omega, setup.X, setup.Psi))
        m_list.append(m_proj)
    return m_list, sigma_list


def simulate_s1_under_null(setup: CalibrationSetup, null_model: DeviationModel,
                            m_list, sigma_list,
                            n_draws: int, b_grid: np.ndarray, rng: np.random.Generator,
                            statistic_fn=s1_statistic_batch) -> np.ndarray:
    """Draw `n_draws` histories of the statistic under b=0, with deviations from
    `null_model`.

    `m_list`/`sigma_list` may be the raw per-model lists or a prebuilt `GridCache`
    (with `sigma_list=None`): the cache is built once per calibration call and
    reused across every draw and null model. Simulation and projection are
    batched over all `n_draws` at once (`simulate_paths_batch`, one matrix
    multiply for the projection), and the direction-density evaluation is
    batched in memory-bounded chunks (`s1_statistic_batch` /
    `two_sided_statistic_batch`), which is what makes the Monte-Carlo loop
    (millions of draws over 1,216 models) tractable. `statistic_fn` defaults to
    the primary one-sided batched statistic; the secondary (two-sided) tests of
    exp_3 pass `two_sided_statistic_batch`, which shares the same call signature.
    """
    cache = m_list if isinstance(m_list, GridCache) else build_grid_cache(m_list, sigma_list)
    u_full_batch = simulate_paths_batch(null_model, setup.n_full, n_draws, rng)  # (n_draws, n_full)
    u_anchor = u_full_batch[:, setup.anchor_weeks]  # (n_draws, n_obs)
    # L maps an n_obs-vector to its q-dim projection: z = Psi^T (I - X(X'X)^{-1}X') u.
    XtX_inv = np.linalg.inv(setup.X.T @ setup.X)
    L = setup.Psi.T @ (np.eye(setup.X.shape[0]) - setup.X @ XtX_inv @ setup.X.T)  # (q, n_obs)
    z_batch = u_anchor @ L.T  # (n_draws, q)
    return statistic_fn(z_batch, cache, b_grid, setup.q)


def upper_confidence_limit_for_quantile(draws: np.ndarray, quantile: float,
                                         confidence: float) -> float:
    """Order-statistic upper confidence limit for the `quantile`-quantile of `draws`
    (paper Appendix C: "largest refined 97.5% order-statistic upper confidence limit
    for the 95% quantile"). Uses the binomial order-statistic method: find the
    smallest order statistic k such that P(Binomial(n, quantile) < k) >= confidence,
    i.e. the UCL is the k-th order statistic with coverage `confidence`.
    """
    n = len(draws)
    sorted_draws = np.sort(draws)
    # Smallest k (1-indexed) such that P(X <= k-1) >= confidence, X ~ Binomial(n, quantile)
    k = int(stats.binom.ppf(confidence, n, quantile)) + 1
    k = min(max(k, 1), n)
    return float(sorted_draws[k - 1])


def calibrate_critical_value(setup: CalibrationSetup, grid: list[DeviationModel],
                              m_list: list[np.ndarray], sigma_list: list[np.ndarray],
                              b_grid: np.ndarray, seed: int,
                              n_screen: int = config.CALIBRATION_SCREEN_DRAWS_RUN,
                              n_refine: int = config.CALIBRATION_REFINE_DRAWS_RUN,
                              n_refine_models: int = config.CALIBRATION_N_REFINE_RUN,
                              statistic_fn=s1_statistic_batch
                              ) -> dict:
    """Two-stage calibration (paper Section 4 / Box 1 / Appendix C):
    1. Screen every null member with `n_screen` draws, take its 95% empirical quantile.
    2. Refine the `n_refine_models` hardest (highest-quantile) members with `n_refine` draws.
    3. Critical value = largest refined UCL for the 95% quantile.

    `statistic_fn` selects the statistic being calibrated: the primary one-sided
    `s1_statistic` (default, exp_1/exp_4) or the secondary `two_sided_statistic`
    (exp_3), applying the same calibration rule to either (paper Appendix C).
    """
    rng = np.random.default_rng(seed)
    cache = build_grid_cache(m_list, sigma_list)
    screen_quantiles = np.empty(len(grid))
    for idx, model in enumerate(grid):
        draws = simulate_s1_under_null(setup, model, cache, None, n_screen, b_grid, rng,
                                        statistic_fn=statistic_fn)
        screen_quantiles[idx] = np.quantile(draws, config.CALIBRATION_QUANTILE)
    hardest_idx = np.argsort(screen_quantiles)[::-1][:n_refine_models]

    refined_ucls = np.empty(len(hardest_idx))
    refined_quantiles = np.empty(len(hardest_idx))
    for j, idx in enumerate(hardest_idx):
        draws = simulate_s1_under_null(setup, grid[idx], cache, None, n_refine, b_grid, rng,
                                        statistic_fn=statistic_fn)
        refined_quantiles[j] = np.quantile(draws, config.CALIBRATION_QUANTILE)
        refined_ucls[j] = upper_confidence_limit_for_quantile(
            draws, config.CALIBRATION_QUANTILE, config.CALIBRATION_UCL_LEVEL)

    best = int(np.argmax(refined_ucls))
    return {
        "critical_value": float(refined_ucls[best]),
        "largest_refined_quantile": float(refined_quantiles[best]),
        "hardest_model_indices": [int(i) for i in hardest_idx],
        "screen_quantiles": screen_quantiles,
        "refined_ucls": refined_ucls,
        "n_screen": n_screen,
        "n_refine": n_refine,
        "n_refine_models": n_refine_models,
        "seed": seed,
    }
