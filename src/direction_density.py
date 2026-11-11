"""Direction density and the S1 statistic (paper Appendix B, Box 1).

Custom implementation of the exact direction-density recurrence: no general
library (scipy.stats, statsmodels) implements the density of a Gaussian
vector's direction under the declared mean/covariance family with the scaled,
overflow-stable J_n recurrence the paper specifies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc


def log_J0_scaled(x: np.ndarray) -> np.ndarray:
    """log(J_0(x) * exp(-x^2/2)) = log(sqrt(pi/2)) + log(erfc(-x/sqrt(2))).

    J_0(x) = sqrt(pi/2) exp(x^2/2) erfc(-x/sqrt(2)); stable for all real x since
    erfc(-x/sqrt(2)) in (0, 2] and the exp(x^2/2) factor is absorbed.
    """
    return 0.5 * np.log(np.pi / 2) + np.log(np.clip(erfc(-x / np.sqrt(2)), 1e-300, None))


def scaled_J_sequence(x: np.ndarray, n_max: int) -> np.ndarray:
    """J_n(x) * exp(-x^2/2) for n = 0..n_max, via the exact recurrence (paper Appendix B):
    J_0 = sqrt(pi/2) e^{x^2/2} erfc(-x/sqrt2); J_1 = 1 + x J_0; J_n = (n-1) J_{n-2} + x J_{n-1}.
    Evaluated in scaled form (multiplied by e^{-x^2/2}), stable for x >= 0.
    """
    x = np.asarray(x, dtype=float)
    shape = x.shape
    out = np.empty((n_max + 1,) + shape)
    exp_term = np.exp(-x ** 2 / 2)
    J0_scaled = np.sqrt(np.pi / 2) * erfc(-x / np.sqrt(2))  # = J_0 * exp(-x^2/2)
    out[0] = J0_scaled
    if n_max >= 1:
        out[1] = exp_term + x * out[0]
    for n in range(2, n_max + 1):
        out[n] = (n - 1) * out[n - 2] + x * out[n - 1]
    return out


def log_direction_density(u: np.ndarray, b: float, m: np.ndarray, Sigma: np.ndarray,
                           q: int) -> float:
    """log f(u; b*m, Sigma) up to the constant common to all models (paper Appendix B)."""
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return -np.inf
    Sigma_inv_u = np.linalg.solve(Sigma, u)
    Sigma_inv_m = np.linalg.solve(Sigma, m)
    a = float(u @ Sigma_inv_u)
    g = float(u @ Sigma_inv_m)
    c = float(m @ Sigma_inv_m)
    if a <= 0:
        return -np.inf
    # J_{q-1}(x) = scaled(x) * exp(x^2/2); the scaled recurrence is stable for x >= 0,
    # so x < 0 is evaluated at |x| via J_n(-|x|) = ... computed directly is unstable,
    # but b, a >= 0 here and g's sign already determines whether b > 0 can be optimal
    # (see the g <= 0 short-circuit in s1_statistic), so x = b*g/sqrt(a) >= 0 whenever
    # this branch is reached with b > 0.
    x = max(b * g / np.sqrt(a), 0.0)
    log_Jq1_scaled = scaled_J_sequence(np.array([x]), q - 1)[-1][0]
    log_Jq1 = np.log(max(log_Jq1_scaled, 1e-300)) + x ** 2 / 2
    return -0.5 * logdet - 0.5 * b ** 2 * c - (q / 2) * np.log(a) + log_Jq1


@dataclass
class GridCache:
    """Precomputed, draw-independent pieces for every grid member, so that a single
    draw's S1 evaluation is a handful of vectorized array operations instead of a
    Python loop over 1,216 models x 41 b-grid points (paper Appendix B, Box 1).
    """
    m: np.ndarray          # (n_models, q)
    Sigma_inv: np.ndarray  # (n_models, q, q)
    logdet: np.ndarray     # (n_models,)
    c: np.ndarray          # (n_models,) = m^T Sigma_inv m
    V: np.ndarray          # (n_models, q) = Sigma_inv @ m


def build_grid_cache(m_by_model: list[np.ndarray], Sigma_by_model: list[np.ndarray]) -> GridCache:
    """Build a `GridCache` from the per-model (m, Sigma) lists (paper Appendix B)."""
    n_models = len(Sigma_by_model)
    q = m_by_model[0].shape[0]
    m_arr = np.stack(m_by_model)
    Sigma_inv = np.empty((n_models, q, q))
    logdet = np.empty(n_models)
    V = np.empty((n_models, q))
    c = np.empty(n_models)
    for i in range(n_models):
        Sigma = Sigma_by_model[i]
        _, logdet[i] = np.linalg.slogdet(Sigma)
        Sigma_inv[i] = np.linalg.inv(Sigma)
        V[i] = Sigma_inv[i] @ m_arr[i]
        c[i] = float(m_arr[i] @ V[i])
    return GridCache(m=m_arr, Sigma_inv=Sigma_inv, logdet=logdet, c=c, V=V)


def _log_density_grid(u: np.ndarray, cache: GridCache, b_grid: np.ndarray, q: int) -> np.ndarray:
    """log f(u; b*m_theta, Sigma_theta) for every (model, b) pair at once.

    Returns an (n_models, n_b) array (paper Appendix B direction-density formula,
    evaluated with the scaled, overflow-stable J_{q-1} recurrence).
    """
    a = np.einsum("j,mjk,k->m", u, cache.Sigma_inv, u)       # (n_models,)
    g = np.einsum("j,mj->m", u, cache.V)                     # (n_models,)
    a_safe = np.clip(a, 1e-300, None)
    x = np.clip(np.outer(g / np.sqrt(a_safe), b_grid), 0.0, None)  # (n_models, n_b)
    scaled_J = scaled_J_sequence(x, q - 1)[-1]                # (n_models, n_b)
    log_Jq1 = np.log(np.clip(scaled_J, 1e-300, None)) + x ** 2 / 2
    ll = (-0.5 * cache.logdet[:, None] - 0.5 * (b_grid ** 2)[None, :] * cache.c[:, None]
          - (q / 2) * np.log(a_safe)[:, None] + log_Jq1)
    ll[a <= 0] = -np.inf
    return ll


def s1_statistic(u: np.ndarray, m_by_model, Sigma_by_model,
                  b_grid: np.ndarray, q: int) -> tuple[float, int, float]:
    """S1(u) = max_{theta,b} log f(u; b m_theta, Sigma_theta) - max_theta log f(u; 0, Sigma_theta).

    `m_by_model`/`Sigma_by_model` may be the usual per-model lists or a prebuilt
    `GridCache` (reused across draws to avoid rebuilding Sigma_inv every time).
    Returns (S1, best null-model index, best alternative log-likelihood).
    A zero projected vector, or any non-finite value, stops the computation (paper
    Appendix B). `b_grid` must include 0: the alternative's grid already contains
    the null case, so B's maximum automatically dominates the b=0 term.
    """
    if not np.all(np.isfinite(u)) or np.allclose(u, 0.0):
        raise ValueError("zero or non-finite projected vector: S1 computation stops")
    cache = m_by_model if isinstance(m_by_model, GridCache) else build_grid_cache(m_by_model, Sigma_by_model)
    ll = _log_density_grid(u, cache, b_grid, q)
    zero_col = int(np.argmin(np.abs(b_grid)))
    best_null = float(np.max(ll[:, zero_col]))
    best_null_idx = int(np.argmax(ll[:, zero_col]))
    best_alt = float(np.max(ll))
    if not np.isfinite(best_null) or not np.isfinite(best_alt):
        raise ValueError("non-finite maximized log-likelihood: S1 computation stops")
    return max(best_alt - best_null, 0.0), best_null_idx, best_alt


def two_sided_statistic(u: np.ndarray, m_by_model, Sigma_by_model,
                         b_grid_pos: np.ndarray, q: int) -> tuple[float, int, float]:
    """Two-sided version used by the secondary tests (paper Appendix B): the grid of
    coefficients is +/-B, and f(u; -b m, Sigma) = f(-u; b m, Sigma) for b > 0, so each
    negative grid point is evaluated as the positive point applied to -u. Vectorized
    the same way as `s1_statistic` (see `GridCache`).
    """
    if not np.all(np.isfinite(u)) or np.allclose(u, 0.0):
        raise ValueError("zero or non-finite projected vector: statistic computation stops")
    cache = m_by_model if isinstance(m_by_model, GridCache) else build_grid_cache(m_by_model, Sigma_by_model)
    b_grid_full = np.concatenate([[0.0], b_grid_pos])
    ll_pos = _log_density_grid(u, cache, b_grid_full, q)
    ll_neg = _log_density_grid(-u, cache, b_grid_full, q)
    best_null = float(max(ll_pos[:, 0].max(), ll_neg[:, 0].max()))
    best_null_idx = int(np.argmax(ll_pos[:, 0]))
    best_alt = float(max(ll_pos.max(), ll_neg.max()))
    if not np.isfinite(best_null) or not np.isfinite(best_alt):
        raise ValueError("non-finite maximized log-likelihood: statistic computation stops")
    return max(best_alt - best_null, 0.0), best_null_idx, best_alt
