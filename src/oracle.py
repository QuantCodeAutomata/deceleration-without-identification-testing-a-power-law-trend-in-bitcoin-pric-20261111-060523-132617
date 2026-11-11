"""Neyman-Pearson oracle power bounds (paper Section 5, Appendix J).

For one alternative (a trend coefficient b plus one deviation model) and one
null member (b=0 plus another deviation model from the declared family), the
most powerful test at level alpha is the likelihood-ratio test between the two
fully specified Gaussians of the projected vector z = Psi^T M_X y. Its exact
power is obtained by Monte Carlo: simulate z under each hypothesis, calibrate
the rejection threshold from the null draws' (1-alpha) quantile of the
log-likelihood ratio, and measure the rejection rate under the alternative.
The oracle upper bound for the whole declared family is the smallest such
power over every null member: no test that holds level alpha against the
whole family can beat the Neyman-Pearson test against its hardest null.

Custom implementation (paper Appendix J): this null-worst-case Neyman-Pearson
bound under a finite competing-model grid is specific to the paper's design;
no general hypothesis-testing library computes it.
"""
from __future__ import annotations

import numpy as np

from src.deviation_family import DeviationModel, model_covariance, simulate_path


def mvn_log_pdf(z: np.ndarray, mean: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(z; mean, Sigma) evaluated at rows of `z` (n_draws x q)."""
    q = Sigma.shape[0]
    sign, logdet = np.linalg.slogdet(Sigma)
    diff = z - mean[None, :]
    Sigma_inv_diff = np.linalg.solve(Sigma, diff.T).T
    quad = np.einsum("ij,ij->i", diff, Sigma_inv_diff)
    return -0.5 * q * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * quad


def log_likelihood_ratio(z: np.ndarray, mean0: np.ndarray, Sigma0: np.ndarray,
                          mean1: np.ndarray, Sigma1: np.ndarray) -> np.ndarray:
    """log p1(z) - log p0(z), the Neyman-Pearson statistic (higher favors H1)."""
    return mvn_log_pdf(z, mean1, Sigma1) - mvn_log_pdf(z, mean0, Sigma0)


def simulate_z(model: DeviationModel, b: float, m: np.ndarray, setup, rng: np.random.Generator,
               n_draws: int) -> np.ndarray:
    """Simulate `n_draws` draws of z = Psi^T M_X (u + b*regressor_component) under `model`."""
    out = np.empty((n_draws, setup.q))
    for i in range(n_draws):
        u_full = simulate_path(model, setup.n_full, rng)
        u_anchor = u_full[setup.anchor_weeks]
        resid = u_anchor - setup.X @ np.linalg.solve(setup.X.T @ setup.X, setup.X.T @ u_anchor)
        out[i] = setup.Psi.T @ resid + b * m
    return out


def np_test_power(alt_model: DeviationModel, null_model: DeviationModel, b: float,
                   m: np.ndarray, setup, rng: np.random.Generator, n_draws: int,
                   alpha: float = 0.05) -> float:
    """Power of the exact Neyman-Pearson test of H0: (b=0, null_model) against
    H1: (b, alt_model), at level alpha (paper Appendix J)."""
    Sigma0 = model_covariance(null_model, setup.anchor_weeks, setup.n_full)
    Sigma0 = _project_sigma(Sigma0, setup)
    Sigma1 = model_covariance(alt_model, setup.anchor_weeks, setup.n_full)
    Sigma1 = _project_sigma(Sigma1, setup)
    mean0 = np.zeros(setup.q)
    mean1 = b * m

    z0 = simulate_z(null_model, 0.0, m, setup, rng, n_draws)
    lr0 = log_likelihood_ratio(z0, mean0, Sigma0, mean1, Sigma1)
    threshold = np.quantile(lr0, 1 - alpha)

    z1 = simulate_z(alt_model, b, m, setup, rng, n_draws)
    lr1 = log_likelihood_ratio(z1, mean0, Sigma0, mean1, Sigma1)
    return float(np.mean(lr1 > threshold))


def _project_sigma(omega: np.ndarray, setup) -> np.ndarray:
    from src.projection import apply_residualizer
    A = apply_residualizer(setup.X, setup.Psi)
    return A.T @ omega @ A


def oracle_power_bound(alt_model: DeviationModel, null_grid: list[DeviationModel], b: float,
                        m: np.ndarray, setup, rng: np.random.Generator, n_draws: int,
                        alpha: float = 0.05) -> tuple[float, int]:
    """min over null_grid of the NP test's power against `alt_model` at coefficient `b`
    (paper Section 5: "the smallest such power over all null members is the oracle
    upper bound"). Returns (bound, worst-null index).
    """
    powers = np.empty(len(null_grid))
    for idx, null_model in enumerate(null_grid):
        powers[idx] = np_test_power(alt_model, null_model, b, m, setup, rng, n_draws, alpha)
    worst = int(np.argmin(powers))
    return float(powers[worst]), worst
