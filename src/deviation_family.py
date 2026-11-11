"""The declared 1,216-member deviation family (paper Appendix B).

Deviations u live on a weekly latent grid k = 0, 1, ... from the genesis
week and are the sum of one persistence component and one cycle, independent
of each other. This module builds the grid of (persistence, cycle,
volatility-shape) combinations and computes the exact covariance of the
latent process at an arbitrary set of weekly indices (not just contiguous
ones), without ever forming the full genesis-to-date covariance matrix.

Custom implementation (paper Appendix B, Eq. for sigma_k, AR/fractional/
local-level persistence, Harvey cycle covariance). No general-purpose
library implements this declared family; `arch`/`statsmodels` cover single
GARCH/ARMA models, not a finite grid of competing persistence x cycle x
volatility-shape specifications combined through a shared Monte-Carlo and
calibration pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from src import config


@dataclass(frozen=True)
class PersistenceSpec:
    kind: str  # "ar", "fractional", "local_level"
    param: float  # c (AR), d (fractional), g (local level)


@dataclass(frozen=True)
class CycleSpec:
    r: float
    period_years: float
    kappa: int


@dataclass(frozen=True)
class DeviationModel:
    persistence: PersistenceSpec
    cycle: CycleSpec
    lam: float  # volatility-shape exponent


def build_grid() -> list[DeviationModel]:
    """The full 1,216-member grid: 16 persistence x 19 cycle x 4 volatility shapes."""
    persistences = (
        [PersistenceSpec("ar", c) for c in config.AR_C_VALUES]
        + [PersistenceSpec("fractional", d) for d in config.FRACTIONAL_D_VALUES]
        + [PersistenceSpec("local_level", g) for g in config.LOCAL_LEVEL_G_VALUES]
    )
    assert len(persistences) == 16
    cycles = [CycleSpec(0.0, config.CYCLE_PERIOD_YEARS[0], config.CYCLE_KAPPA_VALUES[0])]
    for r in config.CYCLE_R_VALUES:
        if r == 0.0:
            continue
        for period, kappa in product(config.CYCLE_PERIOD_YEARS, config.CYCLE_KAPPA_VALUES):
            cycles.append(CycleSpec(r, period, kappa))
    assert len(cycles) == 19
    grid = [
        DeviationModel(p, c, lam)
        for p, c, lam in product(persistences, cycles, config.VOLATILITY_LAMBDA_VALUES)
    ]
    assert len(grid) == 1216
    return grid


def year_of_week(week_index: np.ndarray, genesis_year: int = 2009) -> np.ndarray:
    """Calendar year of latent week `week_index`, weeks since the genesis week."""
    genesis = np.datetime64(config.GENESIS_TIMESTAMP[:10])
    dates = genesis + (week_index * 7).astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    return years


def sigma_profile(lam: float, week_index: np.ndarray,
                   annual_volatility: dict[int, float] | None = None) -> np.ndarray:
    """sigma_k = sigma_bar * h_k^lambda / rms(h^lambda) (paper Appendix B).

    h_k is the annual volatility of the calendar year of week k, using 2011's
    value before 2011. The root mean square is taken over the fixed
    calibration support (every year with a declared volatility value).
    """
    annual_volatility = annual_volatility or config.ANNUAL_VOLATILITY
    years = year_of_week(week_index)
    first_year = config.CALIBRATION_SUPPORT_YEARS[0]
    years_clamped = np.clip(years, first_year, max(annual_volatility))
    h_k = np.array([annual_volatility[int(y)] for y in years_clamped])
    support_vals = np.array([annual_volatility[y] for y in config.CALIBRATION_SUPPORT_YEARS])
    rms = np.sqrt(np.mean(support_vals ** (2 * lam))) if lam != 0 else 1.0
    h_k_lam = h_k ** lam if lam != 0 else np.ones_like(h_k)
    return config.SIGMA_BAR * h_k_lam / rms


def _fractional_psi(d: float, n: int) -> np.ndarray:
    """Coefficients of (1-L)^{-d}: psi_0=1, psi_j = (j-1+d)/j * psi_{j-1}."""
    psi = np.empty(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = (j - 1 + d) / j * psi[j - 1]
    return psi


def _persistence_L_sub(spec: PersistenceSpec, anchor_weeks: np.ndarray, n_full: int) -> tuple:
    """Return list of (psi_array, weight) pairs whose L_sub @ diag(sigma) representation
    sums (with weight**2) to the persistence component's covariance."""
    if spec.kind == "ar":
        rho = 1.0 - spec.param / config.AR_SCALE
        h = np.arange(n_full)
        psi = rho ** h
        return [(psi, 1.0)]
    if spec.kind == "fractional":
        psi = _fractional_psi(spec.param, n_full)
        return [(psi, 1.0)]
    if spec.kind == "local_level":
        psi_eps = np.zeros(n_full)
        psi_eps[0] = 1.0
        psi_eta = np.ones(n_full)
        return [(psi_eps, 1.0), (psi_eta, spec.param)]
    raise ValueError(f"unknown persistence kind {spec.kind}")


def _L_sub_from_psi(psi: np.ndarray, anchor_weeks: np.ndarray, n_full: int) -> np.ndarray:
    """L_sub[i, j] = psi[anchor_weeks[i] - j] for j <= anchor_weeks[i], else 0."""
    diff = anchor_weeks[:, None] - np.arange(n_full)[None, :]
    mask = diff >= 0
    idx = np.clip(diff, 0, n_full - 1)
    return np.where(mask, psi[idx], 0.0)


def persistence_covariance(spec: PersistenceSpec, lam: float, anchor_weeks: np.ndarray,
                            n_full: int) -> np.ndarray:
    """Covariance of the persistence component at `anchor_weeks` (n_obs x n_obs)."""
    sigma = sigma_profile(lam, np.arange(n_full))
    total = np.zeros((len(anchor_weeks), len(anchor_weeks)))
    for psi, weight in _persistence_L_sub(spec, anchor_weeks, n_full):
        L_sub = _L_sub_from_psi(psi, anchor_weeks, n_full)
        scaled = L_sub * sigma[None, :]
        total += (weight ** 2) * (scaled @ scaled.T)
    return total


def cycle_covariance(spec: CycleSpec, anchor_weeks: np.ndarray) -> np.ndarray:
    """Harvey's stochastic trigonometric cycle covariance (paper Appendix B).

    Cov(psi_k, psi_l) = (r*sigma_bar)^2 * rho_c^|k-l| * cos(omega*(k-l)), stationary
    from genesis, so it depends only on the anchor week-index difference.
    """
    if spec.r == 0.0:
        return np.zeros((len(anchor_weeks), len(anchor_weeks)))
    period_weeks = spec.period_years * config.WEEKS_PER_YEAR
    omega = 2 * np.pi / period_weeks
    rho_c = 0.5 ** (1.0 / (spec.kappa * period_weeks))
    dk = anchor_weeks[:, None] - anchor_weeks[None, :]
    scale = (spec.r * config.SIGMA_BAR) ** 2
    return scale * (rho_c ** np.abs(dk)) * np.cos(omega * dk)


def model_covariance(model: DeviationModel, anchor_weeks: np.ndarray, n_full: int) -> np.ndarray:
    """Exact covariance Omega_theta of u at `anchor_weeks` (n_obs x n_obs)."""
    return (
        persistence_covariance(model.persistence, model.lam, anchor_weeks, n_full)
        + cycle_covariance(model.cycle, anchor_weeks)
    )


def simulate_path(model: DeviationModel, n_full: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate one full-grid path u_0..u_{n_full-1} under `model` (for Monte-Carlo
    calibration; O(n) per persistence draw, closed-form Harvey recursion for the cycle).
    """
    sigma = sigma_profile(model.lam, np.arange(n_full))
    e = rng.standard_normal(n_full) * sigma
    spec = model.persistence
    if spec.kind == "ar":
        # u_k = rho*u_{k-1} + e_k, u_{-1}=0: AR(1) recursion via lfilter (vectorized,
        # equivalent to the explicit loop but avoids an O(n_full) Python loop per draw).
        rho = 1.0 - spec.param / config.AR_SCALE
        u = lfilter([1.0], [1.0, -rho], e)
    elif spec.kind == "fractional":
        psi = _fractional_psi(spec.param, n_full)
        # causal convolution: u_k = sum_{j<=k} psi[k-j] e[j]
        u = np.convolve(e, psi)[:n_full]
    elif spec.kind == "local_level":
        eps = e
        eta = rng.standard_normal(n_full) * sigma
        w = np.cumsum(eta)
        u = eps + spec.param * w
    else:
        raise ValueError(spec.kind)

    cyc = model.cycle
    if cyc.r != 0.0:
        # Harvey's stochastic cycle state (s0_k, s1_k) rotates by omega and shrinks by
        # rho_c each week. Writing z_k = s0_k + i*s1_k, the update s_k -> rot(s_k) + inn_k
        # is exactly the complex AR(1) z_{k+1} = rho_c*exp(-i*omega)*z_k + (inn0_k + i*inn1_k),
        # so the whole path is one lfilter call instead of an O(n_full) Python loop.
        period_weeks = cyc.period_years * config.WEEKS_PER_YEAR
        omega = 2 * np.pi / period_weeks
        rho_c = 0.5 ** (1.0 / (cyc.kappa * period_weeks))
        scale = cyc.r * config.SIGMA_BAR
        inn = rng.standard_normal((n_full, 2)) * np.sqrt(1 - rho_c ** 2) * scale
        innov = (inn[:, 0] + 1j * inn[:, 1]).astype(complex)
        state0 = rng.standard_normal(2) * scale  # stationary from genesis (paper Appendix B)
        z0 = complex(state0[0], state0[1])
        c = rho_c * np.exp(-1j * omega)
        x_ext = np.concatenate(([z0], innov[:-1]))
        z = lfilter(np.array([1.0 + 0j]), np.array([1.0 + 0j, -c]), x_ext)
        u = u + z.real
    return u


def simulate_paths_batch(model: DeviationModel, n_full: int, n_draws: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Simulate `n_draws` independent full-grid paths at once (shape (n_draws,
    n_full)), batching every recursion across draws so a Monte-Carlo calibration
    needs a handful of vectorized array operations per null model instead of a
    Python-level loop over draws (paper Appendix B/C: the persistence and cycle
    recursions are linear in the innovations, so batching over draws is exact,
    not an approximation).
    """
    sigma = sigma_profile(model.lam, np.arange(n_full))
    e = rng.standard_normal((n_draws, n_full)) * sigma[None, :]
    spec = model.persistence
    if spec.kind == "ar":
        rho = 1.0 - spec.param / config.AR_SCALE
        u = lfilter([1.0], [1.0, -rho], e, axis=1)
    elif spec.kind == "fractional":
        psi = _fractional_psi(spec.param, n_full)
        # Full causal convolution for every draw at once via one Toeplitz matmul
        # (lower-triangular Toeplitz of psi), equivalent to np.convolve(e, psi)[:n] per row.
        L = toeplitz(psi, np.zeros(n_full))
        u = e @ L.T
    elif spec.kind == "local_level":
        eps = e
        eta = rng.standard_normal((n_draws, n_full)) * sigma[None, :]
        w = np.cumsum(eta, axis=1)
        u = eps + spec.param * w
    else:
        raise ValueError(spec.kind)

    cyc = model.cycle
    if cyc.r != 0.0:
        period_weeks = cyc.period_years * config.WEEKS_PER_YEAR
        omega = 2 * np.pi / period_weeks
        rho_c = 0.5 ** (1.0 / (cyc.kappa * period_weeks))
        scale = cyc.r * config.SIGMA_BAR
        inn = rng.standard_normal((n_draws, n_full, 2)) * np.sqrt(1 - rho_c ** 2) * scale
        innov = (inn[..., 0] + 1j * inn[..., 1]).astype(complex)  # (n_draws, n_full)
        state0 = rng.standard_normal((n_draws, 2)) * scale
        z0 = state0[:, 0] + 1j * state0[:, 1]  # (n_draws,)
        c = rho_c * np.exp(-1j * omega)
        x_ext = np.concatenate([z0[:, None], innov[:, :-1]], axis=1)
        z = lfilter(np.array([1.0 + 0j]), np.array([1.0 + 0j, -c]), x_ext, axis=1)
        u = u + z.real
    return u
