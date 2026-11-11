"""Parameters for the replication, grouped by experiment.

Every numeric value is taken from the paper's text, Appendix A-D, or Appendix
F-K, as cited in the comments. Where the paper leaves a detail unspecified,
the comment says so explicitly.
"""

# ---------------------------------------------------------------------------
# Shared data construction (Section 2, Appendix A)
# ---------------------------------------------------------------------------

GENESIS_TIMESTAMP = "2009-01-03T18:15:05Z"  # Bitcoin genesis block (Appendix A)
WEEK_SECONDS = 7 * 24 * 3600  # one week, used to convert timestamps to age in weeks

PRIMARY_SAMPLE_START = "2011-01-01"  # paper's sample start (Section 2)
PRIMARY_SAMPLE_END = "2026-09-19"  # paper's sample end (Section 2), last Saturday anchor
PRIMARY_DROPPED_ANCHOR = "2011-06-25"  # Saturday with no close, dropped (Section 2)
PRIMARY_N_PAPER = 820  # number of weekly observations reported in the paper (Section 2)

# The paper's exact daily series (Mt. Gox until 2012-12-31, Bitstamp from 2013-01-01,
# bitcoincharts.com / Internet Archive, SHA-256 5552aa23e3...) is not obtainable from
# the Dataloader/Massive market-data path: a smoke test of X:BTCUSD on the QuantX
# Dataloader found no coverage before November 2013 (Massive's own crypto history
# starts there). This is a genuine data-coverage limitation, recorded in
# results/RESULTS.md, not a convenience choice. The realized sample therefore starts
# once continuous daily coverage is available.
REALIZED_SAMPLE_START = "2014-01-04"  # first Saturday with reliable Massive BTC coverage

# ---------------------------------------------------------------------------
# Annual volatility profile of weekly log returns (Appendix A)
# The paper states this is the only real-data input to the simulated deviation
# family. Values are taken directly from the paper's table (Appendix A), not
# recomputed, because the realized price sample does not cover 2011-2013.
# ---------------------------------------------------------------------------
ANNUAL_VOLATILITY = {
    2011: 0.294, 2012: 0.093, 2013: 0.212, 2014: 0.089,
    2015: 0.089, 2016: 0.063, 2017: 0.140, 2018: 0.134,
    2019: 0.085, 2020: 0.109, 2021: 0.111, 2022: 0.092,
    2023: 0.071, 2024: 0.065, 2025: 0.051, 2026: 0.065,
}
CALIBRATION_SUPPORT_YEARS = sorted(ANNUAL_VOLATILITY)  # "fixed calibration support" for the
# rms of h^lambda (Appendix B); not precisely specified by the paper beyond "fixed", so
# the full set of years with a declared volatility value is used, held fixed across
# every analysis at every record length.

SIGMA_BAR = 0.128  # reference innovation scale, Appendix B; the statistic is scale-free

# ---------------------------------------------------------------------------
# Declared deviation family (Appendix B): 16 persistence x 19 cycle x 4 volatility = 1216
# ---------------------------------------------------------------------------
AR_C_VALUES = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0)  # rho = 1 - c/821, c=0 is random walk
FRACTIONAL_D_VALUES = (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)  # type II fractional integration
LOCAL_LEVEL_G_VALUES = (0.1, 0.3, 0.5)  # local-level mixing weight

CYCLE_R_VALUES = (0.0, 5.86, 11.72)  # cycle s.d. up to 1.5 log units (r=0: no cycle)
CYCLE_PERIOD_YEARS = (3.5, 4.0, 4.5)  # Harvey cycle period P, in years
CYCLE_KAPPA_VALUES = (1, 3, 5)  # number of cycles after which the envelope halves
WEEKS_PER_YEAR = 365.25 / 7.0  # conversion of P (years) to weeks (Appendix B)

VOLATILITY_LAMBDA_VALUES = (0.0, 0.5, 1.0, 1.25)  # volatility shape exponent

AR_SCALE = 821  # denominator of rho = 1 - c/821 (Appendix B)

# ---------------------------------------------------------------------------
# Projection (Appendix B): cosine basis and trend grid
# ---------------------------------------------------------------------------
Q_FEATURES_PRIMARY = 6  # number of cosine projections for S1 (Section 4, Box 1)
B_GRID_STEP = 0.75  # step of the b grid for S1 (Appendix B)
B_GRID_MAX = 30.0  # upper end of the b grid for S1 (Appendix B)
PUBLISHED_EXPONENT = 5.7  # the log-age exponent used for power evaluations (Section 5)

LOWER_BOUND_TRIAL_MIN = -30.0  # trial grid T for the lower confidence bound (Appendix B)
LOWER_BOUND_TRIAL_MAX = 40.0
LOWER_BOUND_TRIAL_STEP = 0.1
LOWER_BOUND_TRIAL_EXTENSION = 10000.0  # "extended by doubling to +/-10,000" (Appendix B)

# ---------------------------------------------------------------------------
# Calibration (Appendix C): simulate under the null, refine the hardest members
# ---------------------------------------------------------------------------
# Paper configuration: 2000 draws for every one of the 1216 members, then 20000
# draws for the 60 hardest. This is reduced for the computing budget of this
# replication (see results/RESULTS.md, "Deviations from the paper"); the paper
# configuration remains selectable with --full.
CALIBRATION_SCREEN_DRAWS_PAPER = 2000
CALIBRATION_REFINE_DRAWS_PAPER = 20000
CALIBRATION_N_REFINE_PAPER = 60

CALIBRATION_SCREEN_DRAWS_RUN = 300
CALIBRATION_REFINE_DRAWS_RUN = 2000
CALIBRATION_N_REFINE_RUN = 15

CALIBRATION_QUANTILE = 0.95  # nominal size, Section 4 / Box 1
CALIBRATION_UCL_LEVEL = 0.975  # "largest refined 97.5% order-statistic upper confidence
# limit for the 95% quantile" (Section 4, Appendix C)

S1_CALIBRATION_SEED = 20261024  # seed used for the primary S1 critical value (Section 4)
DRIFT_TEST_SEED = 20261027  # seed for the extra-drift secondary test (Appendix C)
CURVATURE_TEST_SEED = 20261028  # seed for the curvature secondary test (Appendix C)

# ---------------------------------------------------------------------------
# Validation gates (Section 5)
# ---------------------------------------------------------------------------
GATE_G1_N_MODELS_PAPER = 50
GATE_G1_DRAWS_PAPER = 20000
GATE_G1_N_MODELS_RUN = 20
GATE_G1_DRAWS_RUN = 2000
GATE_G1_MAX_SIZE = 0.06  # Clopper-Pearson 6% simultaneous bound (Section 5)
GATE_G3_B_VALUE = 5.7

# ---------------------------------------------------------------------------
# Application result reported by the paper (Section 4), for comparison only
# ---------------------------------------------------------------------------
PAPER_S1_STATISTIC = 3.410
PAPER_S1_CRITICAL_VALUE = 2.631683520034235
PAPER_LOWER_BOUND_B = 1.5
PAPER_VALIDATED_POWER_AT_5_7 = 0.26

# ---------------------------------------------------------------------------
# Trend models (Table 1, exp_5 / exp_6 / exp_9)
# ---------------------------------------------------------------------------
TREND_MODEL_NAMES = ("constant_drift", "power_law", "drift_and_log_age", "logistic", "gompertz")
NONLINEAR_START_GRID_SIZE = 24  # evenly spaced grid members for the two-stage search (Appendix H)

# ---------------------------------------------------------------------------
# Forecasting (exp_6): predictive-density mixture
# ---------------------------------------------------------------------------
FORECAST_N_BOOTSTRAP_PAPER = 49  # bootstrap refits, plus the fitted-parameter Gaussian = 50
FORECAST_N_BOOTSTRAP_RUN = 19  # reduced bootstrap count for the computing budget
FORECAST_TARGET_YEARS = (2027, 2028, 2029, 2030)
FUTURE_VOLATILITY_FLAT = ANNUAL_VOLATILITY[2026]  # future volatility held at its 2026 level

# ---------------------------------------------------------------------------
# Oracle power bounds (exp_2) and record-length projection (exp_8)
# ---------------------------------------------------------------------------
ORACLE_ALPHA = 0.05  # Neyman-Pearson test size
ORACLE_N_SIMULATIONS_PAPER = 20000
ORACLE_N_SIMULATIONS_RUN = 2000

PROJECTION_END_YEARS = (2026, 2030, 2036, 2046)
PROJECTION_Q_FEATURES = {2026: 6, 2030: 8, 2036: 10, 2046: 14}  # Appendix J

# ---------------------------------------------------------------------------
# Conventional-diagnostics simulation study (exp_7)
# ---------------------------------------------------------------------------
DIAGNOSTICS_N_HISTORIES_PAPER = 500
DIAGNOSTICS_N_HISTORIES_RUN = 150
DIAGNOSTICS_TRUE_EXPONENT = 5.7
DIAGNOSTICS_NW_LAGS_WEEKS = 208  # four-year window in weekly lags (Appendix G)
DIAGNOSTICS_BAND_FIT_END_YEAR = 2020  # fit through 2020, check 2021-2026 band (Appendix G)
DIAGNOSTICS_SIGNIFICANCE = 0.05

# ---------------------------------------------------------------------------
# Separating trend models with future data (exp_9)
# ---------------------------------------------------------------------------
SEPARATION_N_SIMULATIONS_PAPER = 1000
SEPARATION_N_SIMULATIONS_RUN = 150
SEPARATION_RECORD_END_YEARS = (2030, 2035, 2040, 2046)
SEPARATION_AIC_DECISIVE_THRESHOLD = 3.0  # "decisive" when AIC leads by at least 3 (Appendix K)

# ---------------------------------------------------------------------------
# Random seeds for reproducibility (not specified at code level by the paper
# beyond the three calibration seeds above; development seeds are ours)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 2026
