"""Experiment 4: sensitivity analyses, 17 pre-specified variations of the
primary S1 test that each change one element of the design (paper Section 4,
Appendix F, Figure 5).

Every variation reuses the shared pipeline of `src/pipeline.py` and
`src/calibration.py`: only the data, the departure grid, or the declared
deviation grid changes. Each variation gets its own critical value by the
same calibration rule as the primary test (paper Appendix C).
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import build_weekly_series, load_spliced_daily_closes
from src import config
from src.calibration import calibrate_critical_value, precompute_m_sigma
from src.deviation_family import CycleSpec, DeviationModel, PersistenceSpec, build_grid
from src.pipeline import apply_s1, primary_setup

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def b_grid() -> np.ndarray:
    return np.arange(0.0, config.B_GRID_MAX + config.B_GRID_STEP / 2, config.B_GRID_STEP)


def weekly_series_weekday(daily: pd.DataFrame, start: str, end: str, weekday: str,
                           drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """Same construction as `build_weekly_series` but anchored on `weekday`
    (pandas offset alias, e.g. 'W-MON') instead of Saturday (paper Appendix F,
    variations 5-11)."""
    daily_idx = daily.set_index("date")["close"]
    anchors = list(pd.date_range(start=start, end=end, freq=weekday))
    dropped = {pd.Timestamp(d) for d in drop}
    records = []
    for day in anchors:
        if day in dropped or day not in daily_idx.index:
            continue
        close = daily_idx.loc[day]
        age_weeks = (day + pd.Timedelta(hours=23, minutes=59, seconds=59)
                     - pd.Timestamp("2009-01-03T18:15:05")).total_seconds() / (7 * 24 * 3600)
        records.append((day, float(close), age_weeks))
    out = pd.DataFrame(records, columns=["date", "close", "age_weeks"])
    out["y"] = np.log(out["close"])
    return out


def monthly_series(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Last Saturday of each calendar month, excluding incomplete September
    2026 and dropping June 2011 (no close), per paper Appendix F variation 12."""
    saturdays = pd.date_range(start=start, end=end, freq="W-SAT")
    df = pd.DataFrame({"date": saturdays})
    df["period"] = df["date"].dt.to_period("M")
    last_sat = df.groupby("period")["date"].max()
    last_sat = last_sat[last_sat.dt.to_period("M") < pd.Timestamp(end).to_period("M")]
    daily_idx = daily.set_index("date")["close"]
    records = []
    for day in last_sat:
        if day == pd.Timestamp(config.PRIMARY_DROPPED_ANCHOR) or day not in daily_idx.index:
            continue
        close = daily_idx.loc[day]
        age_weeks = (day + pd.Timedelta(hours=23, minutes=59, seconds=59)
                     - pd.Timestamp("2009-01-03T18:15:05")).total_seconds() / (7 * 24 * 3600)
        records.append((day, float(close), age_weeks))
    out = pd.DataFrame(records, columns=["date", "close", "age_weeks"])
    out["y"] = np.log(out["close"])
    return out


def restricted_cycle_grid(max_r: float | None = None) -> list[DeviationModel]:
    """Full grid with cycle amplitude capped at `max_r` log units (paper
    Appendix F variations 14-15: "cycle s.d. up to 1.0/2.0 log units")."""
    full = build_grid()
    if max_r is None:
        return full
    return [m for m in full if m.cycle.r <= max_r + 1e-9]


def constant_volatility_grid() -> list[DeviationModel]:
    """lambda=0 only: 16 persistence x 19 cycle x 1 = 304 models (variation 17)."""
    return [m for m in build_grid() if m.lam == 0.0]


def wide_cycle_grid() -> list[DeviationModel]:
    """Cycle periods 3-5 years (5 values) with coherence (kappa) restricted to
    1 or 3 cycles (>=0.5 persistence per the paper's 'coherence >= 0.5'
    description), combined with the 16 persistence x 4 volatility grid: a
    1,600-model grid approximating the paper's variation 16. The exact
    'coherence >= 0.5' construction beyond this is not given in the paper, so
    the widened period range with the two slower kappa values is used."""
    persistences = (
        [PersistenceSpec("ar", c) for c in config.AR_C_VALUES]
        + [PersistenceSpec("fractional", d) for d in config.FRACTIONAL_D_VALUES]
        + [PersistenceSpec("local_level", g) for g in config.LOCAL_LEVEL_G_VALUES]
    )
    periods = (3.0, 3.5, 4.0, 4.5, 5.0)
    kappas = (1, 3)
    cycles = [CycleSpec(0.0, periods[0], kappas[0])]
    for r in config.CYCLE_R_VALUES:
        if r == 0.0:
            continue
        for period, kappa in product(periods, kappas):
            cycles.append(CycleSpec(r, period, kappa))
    grid = [DeviationModel(p, c, lam) for p, c, lam in
            product(persistences, cycles, config.VOLATILITY_LAMBDA_VALUES)]
    return grid


def extended_persistence_grid() -> list[DeviationModel]:
    """Add fractional d in {1.1, 1.25, 1.4} beyond a random walk (variation 18):
    19 persistence x 19 cycle x 4 volatility = 1,444 models."""
    persistences = (
        [PersistenceSpec("ar", c) for c in config.AR_C_VALUES]
        + [PersistenceSpec("fractional", d) for d in config.FRACTIONAL_D_VALUES]
        + [PersistenceSpec("fractional", d) for d in (1.1, 1.25, 1.4)]
        + [PersistenceSpec("local_level", g) for g in config.LOCAL_LEVEL_G_VALUES]
    )
    cycles = [CycleSpec(0.0, config.CYCLE_PERIOD_YEARS[0], config.CYCLE_KAPPA_VALUES[0])]
    for r in config.CYCLE_R_VALUES:
        if r == 0.0:
            continue
        for period, kappa in product(config.CYCLE_PERIOD_YEARS, config.CYCLE_KAPPA_VALUES):
            cycles.append(CycleSpec(r, period, kappa))
    grid = [DeviationModel(p, c, lam) for p, c, lam in
            product(persistences, cycles, config.VOLATILITY_LAMBDA_VALUES)]
    return grid


def run_one(name: str, df: pd.DataFrame, q: int, grid: list[DeviationModel],
            seed: int, n_screen: int, n_refine: int, n_refine_models: int) -> dict:
    bgrid = b_grid()
    setup = primary_setup(df, q)
    m_list, sigma_list = precompute_m_sigma(setup, grid)
    calib = calibrate_critical_value(setup, grid, m_list, sigma_list, bgrid, seed=seed,
                                      n_screen=n_screen, n_refine=n_refine,
                                      n_refine_models=n_refine_models)
    cv = calib["critical_value"]
    result = apply_s1(df, grid, bgrid, q)
    s1 = result["s1"]
    return {
        "name": name, "n_observations": len(df), "n_grid_models": len(grid), "q_features": q,
        "critical_value": cv, "s1_statistic": s1, "reject_constant_drift": bool(s1 > cv),
        "seed": seed,
    }


def run(full: bool = False) -> dict:
    t0 = time.time()
    n_screen = config.CALIBRATION_SCREEN_DRAWS_PAPER if full else config.CALIBRATION_SCREEN_DRAWS_RUN
    n_refine = config.CALIBRATION_REFINE_DRAWS_PAPER if full else config.CALIBRATION_REFINE_DRAWS_RUN
    n_refine_models = config.CALIBRATION_N_REFINE_PAPER if full else config.CALIBRATION_N_REFINE_RUN

    daily = load_spliced_daily_closes(config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END)
    primary_df = build_weekly_series(daily, config.REALIZED_SAMPLE_START,
                                      config.PRIMARY_SAMPLE_END,
                                      dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,))
    full_grid = build_grid()
    analyses = {}
    seed0 = config.S1_CALIBRATION_SEED

    weekday_offsets = {"monday": "W-MON", "tuesday": "W-TUE", "wednesday": "W-WED",
                       "thursday": "W-THU", "friday": "W-FRI", "saturday": "W-SAT",
                       "sunday": "W-SUN"}
    for i, (name, alias) in enumerate(weekday_offsets.items()):
        if name == "saturday":
            continue  # the primary design itself
        wdf = weekly_series_weekday(daily, config.REALIZED_SAMPLE_START,
                                     config.PRIMARY_SAMPLE_END, alias)
        analyses[f"weekday_{name}"] = run_one(f"weekday_{name}", wdf, config.Q_FEATURES_PRIMARY,
                                               full_grid, seed0 + 100 + i, n_screen, n_refine,
                                               n_refine_models)

    monthly_df = monthly_series(daily, config.REALIZED_SAMPLE_START, config.PRIMARY_SAMPLE_END)
    analyses["monthly"] = run_one("monthly", monthly_df, config.Q_FEATURES_PRIMARY, full_grid,
                                   seed0 + 200, n_screen, n_refine, n_refine_models)

    analyses["twelve_averages"] = run_one("twelve_averages", primary_df, 12, full_grid,
                                           seed0 + 300, n_screen, n_refine, n_refine_models)

    analyses["cycle_sd_up_to_1_0"] = run_one("cycle_sd_up_to_1_0", primary_df,
                                              config.Q_FEATURES_PRIMARY,
                                              restricted_cycle_grid(1.0), seed0 + 400,
                                              n_screen, n_refine, n_refine_models)
    analyses["cycle_sd_up_to_2_0"] = run_one("cycle_sd_up_to_2_0", primary_df,
                                              config.Q_FEATURES_PRIMARY,
                                              restricted_cycle_grid(2.0), seed0 + 401,
                                              n_screen, n_refine, n_refine_models)

    analyses["wide_cycle"] = run_one("wide_cycle", primary_df, config.Q_FEATURES_PRIMARY,
                                      wide_cycle_grid(), seed0 + 500, n_screen, n_refine,
                                      n_refine_models)

    analyses["constant_volatility"] = run_one("constant_volatility", primary_df,
                                               config.Q_FEATURES_PRIMARY,
                                               constant_volatility_grid(), seed0 + 600,
                                               n_screen, n_refine, n_refine_models)

    analyses["persistence_to_d_1_4"] = run_one("persistence_to_d_1_4", primary_df,
                                                config.Q_FEATURES_PRIMARY,
                                                extended_persistence_grid(), seed0 + 700,
                                                n_screen, n_refine, n_refine_models)

    include_2010_df = build_weekly_series(
        load_spliced_daily_closes("2010-07-17", config.PRIMARY_SAMPLE_END),
        "2010-07-17", config.PRIMARY_SAMPLE_END,
        dropped_anchors=(config.PRIMARY_DROPPED_ANCHOR,))
    analyses["include_2010"] = run_one("include_2010", include_2010_df,
                                        config.Q_FEATURES_PRIMARY, full_grid, seed0 + 800,
                                        n_screen, n_refine, n_refine_models)

    not_reproduced = {
        "coin_metrics_prices": "No Coin Metrics reference-price feed is served by the "
            "QuantX Dataloader or Massive, and the authors' sensitivity CSV "
            "(sensitivity_daily_2026-10-05.csv) is not accessible from this environment.",
        "blockchain_com_prices": "Same unavailability as the Coin Metrics variation: no "
            "Blockchain.com reference-price feed is served by Dataloader/Massive, and the "
            "authors' sensitivity file could not be obtained.",
        "inflation_adjusted": "Requires the authors' exact CPI deflation of the frozen daily "
            "series together with the US CPI (FRED CPIAUCNS); the frozen daily series itself "
            "is already a reconstruction (see 'Deviations from the paper'), so deflating it "
            "would not reproduce the paper's inflation-adjusted variation.",
    }

    n_pass = sum(1 for a in analyses.values() if a["reject_constant_drift"])
    out = {
        "n_analyses_run": len(analyses), "n_reject_constant_drift": n_pass,
        "full_paper_configuration": full, "analyses": analyses,
        "not_reproduced": not_reproduced,
        "elapsed_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp4_sensitivity_analyses.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "analyses"}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run(full=args.full)
