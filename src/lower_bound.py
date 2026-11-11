"""Lower confidence bound for the log-age coefficient b (paper Appendix B).

Custom implementation: inverting the S1 test over a trial grid by direct
enumeration is specific to this paper's calibrated-direction-test
construction; no general library provides it.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.direction_density import GridCache, build_grid_cache, s1_statistic

# The primary test's design has a single departure regressor, so every grid
# member shares the same projected m (it depends only on the regressor and
# the design matrix, not on the deviation model); shifting z by t*m is
# therefore unambiguous and the same for every model.


def trial_grid() -> np.ndarray:
    """Trial set T: -30 to 40 in steps of 0.1, extended by doubling to +/-10,000
    (paper Appendix B)."""
    core = np.round(np.arange(config.LOWER_BOUND_TRIAL_MIN, config.LOWER_BOUND_TRIAL_MAX
                               + config.LOWER_BOUND_TRIAL_STEP / 2,
                               config.LOWER_BOUND_TRIAL_STEP), 10)
    extension = np.array([-config.LOWER_BOUND_TRIAL_EXTENSION,
                           config.LOWER_BOUND_TRIAL_EXTENSION])
    return np.unique(np.concatenate([core, extension]))


def lower_confidence_bound(z: np.ndarray, m_list, sigma_list,
                            b_grid: np.ndarray, q: int, critical_value: float,
                            trials: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """L_T(z) = min { t in T : S1(z - t*m) <= critical_value } (paper Appendix B).

    Returns (L_T, accepted_trials). +inf if the confidence set is empty. Every
    grid member shares the same projected regressor m (see module note), so
    z - t*m is a single shifted vector fed back into the same S1 statistic.
    `m_list`/`sigma_list` may be raw per-model lists or a prebuilt `GridCache`.
    """
    trials = trials if trials is not None else trial_grid()
    cache = m_list if isinstance(m_list, GridCache) else build_grid_cache(m_list, sigma_list)
    m_common = cache.m[0]
    accepted = []
    for t in trials:
        z_shift = z - t * m_common
        if not np.all(np.isfinite(z_shift)) or np.allclose(z_shift, 0.0):
            continue
        try:
            s1_t, _, _ = s1_statistic(z_shift, cache, None, b_grid, q)
        except ValueError:
            continue
        if s1_t <= critical_value:
            accepted.append(t)
    if not accepted:
        return float("inf"), np.array([])
    return float(min(accepted)), np.array(accepted)
