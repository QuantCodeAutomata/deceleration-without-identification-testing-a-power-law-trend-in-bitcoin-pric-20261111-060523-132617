"""Cosine low-frequency projection and residualization (paper Appendix B, Box 1).

Custom implementation of the Mueller-Watson (2008) low-frequency projection
as specified by the paper: no general-purpose library implements this exact
cosine basis combined with linear/power-law residualization for a declared
covariance family.
"""
from __future__ import annotations

import numpy as np


def residualizer(X: np.ndarray) -> np.ndarray:
    """M_X = I - X (X^T X)^{-1} X^T, the projection that removes the columns of X.

    Only used where the explicit n x n matrix is convenient (tests); the hot
    path below applies M_X without ever forming it, via `apply_residualizer`.
    """
    n = X.shape[0]
    beta_proj = X @ np.linalg.solve(X.T @ X, X.T)
    return np.eye(n) - beta_proj


def apply_residualizer(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """M_X @ V = V - X (X^T X)^{-1} X^T V, without forming the n x n matrix M_X."""
    return V - X @ np.linalg.solve(X.T @ X, X.T @ V)


def cosine_weights(n: int, q: int) -> np.ndarray:
    """Psi_{tj} = sqrt(2) cos(pi j tau_t), j=1..q, tau_t = (t-1)/(n-1) in [0,1].

    tau_t is "the position of anchor t in the record" (paper Appendix B); the
    paper does not spell out the normalization, so linear positions over the
    full record, from 0 to 1, are used (Mueller-Watson 2008 low-frequency
    weights), consistent with the stated period range ("between half a cycle
    and three cycles over the record", Section 4).
    """
    tau = np.arange(n) / (n - 1)
    j = np.arange(1, q + 1)
    return np.sqrt(2) * np.cos(np.pi * np.outer(tau, j))


def project_z(y: np.ndarray, X: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """z = Psi^T M_X y."""
    return Psi.T @ apply_residualizer(X, y)


def project_m(regressor: np.ndarray, X: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """m = Psi^T M_X regressor (e.g. regressor = log(s) for the primary test)."""
    return Psi.T @ apply_residualizer(X, regressor)


def sigma_theta(omega_anchors: np.ndarray, X: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """Sigma_theta = Psi^T M_X Omega_theta M_X Psi (q x q)."""
    A = apply_residualizer(X, Psi)
    return A.T @ omega_anchors @ A
