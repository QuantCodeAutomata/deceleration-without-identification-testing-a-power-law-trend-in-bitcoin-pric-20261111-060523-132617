"""Gaussian maximum-likelihood fits of the five trend models (paper Table 1,
Appendix H). Linear models (constant drift, power law, drift and log age) are
profiled exactly over the declared deviation grid by generalized least
squares; the two nonlinear models (logistic, Gompertz) use the paper's fixed
two-stage search.

Custom implementation (paper Appendix H): GLS under a declared covariance
grid, and the specific two-stage nonlinear search, are not provided by a
general-purpose fitting library. `scipy.optimize.minimize(method="L-BFGS-B")`
is used for the nonlinear inner loop, as named in the paper's implementation
notes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.deviation_family import DeviationModel, model_covariance


@dataclass
class TrendFit:
    model_name: str
    params: dict
    theta_index: int
    log_likelihood: float
    n_trend_params: int
    beta: np.ndarray | None = None
    kappa2: float | None = None


def gls_profile_loglik(y: np.ndarray, X: np.ndarray, Omega: np.ndarray
                        ) -> tuple[np.ndarray, float, float]:
    """GLS fit of y = X beta + kappa*u, u ~ N(0, Omega), kappa^2 profiled out.

    beta_hat = (X^T Omega^-1 X)^-1 X^T Omega^-1 y
    kappa_hat^2 = residual sum of squares (under Omega^-1 metric) / n
    log L = -0.5 n log(2 pi) - 0.5 log|Omega| - 0.5 n log(kappa_hat^2) - 0.5 n
    """
    n = len(y)
    Omega_inv_X = np.linalg.solve(Omega, X)
    Omega_inv_y = np.linalg.solve(Omega, y)
    XtOiX = X.T @ Omega_inv_X
    beta = np.linalg.solve(XtOiX, X.T @ Omega_inv_y)
    resid = y - X @ beta
    rss = float(resid @ np.linalg.solve(Omega, resid))
    kappa2 = rss / n
    sign, logdet = np.linalg.slogdet(Omega)
    loglik = -0.5 * n * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * n * np.log(kappa2) - 0.5 * n
    return beta, kappa2, float(loglik)


def fit_linear_model(y: np.ndarray, anchor_weeks: np.ndarray, n_full: int,
                      design_fn, grid: list[DeviationModel], n_trend_params: int,
                      model_name: str) -> TrendFit:
    """Profile a linear trend model over the declared grid; `design_fn(s)` returns X."""
    best = None
    for idx, model in enumerate(grid):
        Omega = model_covariance(model, anchor_weeks, n_full)
        X = design_fn()
        beta, kappa2, loglik = gls_profile_loglik(y, X, Omega)
        if best is None or loglik > best.log_likelihood:
            best = TrendFit(model_name, {}, idx, loglik, n_trend_params, beta, kappa2)
    return best


def _logistic_design(s: np.ndarray, r: float, s0: float) -> np.ndarray:
    col = -np.log1p(np.exp(-r * (s - s0)))
    return np.column_stack([np.ones_like(s), col])


def _gompertz_design(s: np.ndarray, r: float) -> np.ndarray:
    col = -np.exp(-r * s)
    return np.column_stack([np.ones_like(s), col])


def _nonlinear_profile_loglik(nl_params: np.ndarray, kind: str, y: np.ndarray, s: np.ndarray,
                               Omega: np.ndarray) -> tuple[float, np.ndarray, float]:
    if kind == "logistic":
        r, s0 = nl_params
        X = _logistic_design(s, r, s0)
    else:
        (r,) = nl_params
        X = _gompertz_design(s, r)
    beta, kappa2, loglik = gls_profile_loglik(y, X, Omega)
    return loglik, beta, kappa2


def fit_nonlinear_model(y: np.ndarray, s: np.ndarray, anchor_weeks: np.ndarray, n_full: int,
                         grid: list[DeviationModel], kind: str, start_grid: list[np.ndarray],
                         bounds: list[tuple[float, float]]) -> TrendFit:
    """Fixed two-stage search (paper Appendix H):
    1. From `start_grid` starting points, bounded L-BFGS-B fits of nonlinear
       parameters at 24 evenly spaced grid members (approximated here at the
       genesis-fit member, then refined across the grid in stage 2).
    2. With the best nonlinear parameters fixed, profile the deviation model
       over the whole grid.
    3. Refit nonlinear parameters at the selected member; keep only if it does
       not lower the likelihood.
    """
    n_trend = 3
    # Stage 1: search nonlinear params at a representative (random-walk, no-cycle) member.
    rw_idx = next(i for i, m in enumerate(grid)
                  if m.persistence.kind == "ar" and m.persistence.param == 0.0
                  and m.cycle.r == 0.0 and m.lam == 1.0)
    Omega_rw = model_covariance(grid[rw_idx], anchor_weeks, n_full)
    best_stage1 = None
    for start in start_grid:
        res = minimize(lambda p: -_nonlinear_profile_loglik(p, kind, y, s, Omega_rw)[0],
                        start, method="L-BFGS-B", bounds=bounds)
        if res.success and (best_stage1 is None or -res.fun > best_stage1[1]):
            best_stage1 = (res.x, -res.fun)
    nl_params = best_stage1[0]

    # Stage 2: profile the deviation model over the whole grid at fixed nl_params.
    best = None
    for idx, model in enumerate(grid):
        Omega = model_covariance(model, anchor_weeks, n_full)
        loglik, beta, kappa2 = _nonlinear_profile_loglik(nl_params, kind, y, s, Omega)
        if best is None or loglik > best[0]:
            best = (loglik, idx, beta, kappa2)
    best_loglik, best_idx, best_beta, best_kappa2 = best

    # Stage 3: refit nl params at the selected member; keep only if it does not lower L.
    Omega_sel = model_covariance(grid[best_idx], anchor_weeks, n_full)
    res = minimize(lambda p: -_nonlinear_profile_loglik(p, kind, y, s, Omega_sel)[0],
                    nl_params, method="L-BFGS-B", bounds=bounds)
    if res.success and -res.fun > best_loglik:
        nl_params = res.x
        best_loglik, best_beta, best_kappa2 = -res.fun, *_nonlinear_profile_loglik(
            nl_params, kind, y, s, Omega_sel)[1:]

    params = {"r": nl_params[0]} if kind == "gompertz" else {"r": nl_params[0], "s0": nl_params[1]}
    return TrendFit(kind, params, best_idx, best_loglik, n_trend, best_beta, best_kappa2)
