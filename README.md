# Deceleration without identification: testing a power-law trend in Bitcoin prices under persistent noise

Replication of the paper's confirmatory and exploratory tests of a power-law
(log-age) trend in Bitcoin's weekly log price, against a declared family of
1,216 competing persistent-and-cyclical deviation models.

## Layout

- `src/` — shared library: the declared deviation family (`deviation_family.py`),
  the cosine low-frequency projection (`projection.py`), the direction-density
  S1 statistic (`direction_density.py`), simulation calibration
  (`calibration.py`), the lower confidence bound (`lower_bound.py`), the
  Neyman-Pearson oracle bound (`oracle.py`), the five trend-model Gaussian
  fits (`trend_models.py`), the shared pipeline (`pipeline.py`), and
  `config.py` (every paper-derived parameter).
- `exp/` — one script per experiment (`exp1` .. `exp9`), run as
  `python -m exp.exp1`.
- `data/` — `data_loader.py` loads Bitcoin daily/weekly closes through the
  QuantX Dataloader REST API; no synthetic data generator is used (every
  experiment's simulations are generated inline from the declared deviation
  family in `src/`, which is the paper's own model, not a data-fixture
  generator).
- `tests/` — unit tests of `src/` on synthetic/small inputs, plus
  `market_data`-marked tests of the real Bitcoin series.
- `results/` — `RESULTS.md` and every generated table, plot and metrics file.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                        # full suite, requires API keys
```

## Experiments

```bash
python -m exp.exp1   # primary S1 test
python -m exp.exp2   # oracle power bounds
python -m exp.exp3   # secondary drift/curvature tests
python -m exp.exp4   # 17 sensitivity analyses
python -m exp.exp5   # descriptive Gaussian fits of five trend models
python -m exp.exp6   # registered predictive densities 2027-2030
python -m exp.exp7   # conventional diagnostics robustness study
python -m exp.exp8   # power vs. record length (2026-2046)
python -m exp.exp9   # separating best-fitting trends with future data
```

Each experiment can also be run with `--full` for the paper's exact
simulation counts (Monte Carlo draws, bootstrap refits); the default,
reduced counts are documented in `results/RESULTS.md`.

Re-running an experiment overwrites its files in `results/`; the numbers and
verdicts in `results/RESULTS.md` must then be checked against the new files.

## Data

Bitcoin OHLCV bars are loaded only through the QuantX Dataloader REST API
(`data/data_loader.py`). The paper's own 2011-2013 daily series (Mt. Gox then
Bitstamp) is not available from this path: Massive's BTC/USD history starts
on 2013-11-01, so the realized weekly sample used here starts on
2014-01-04. See `results/RESULTS.md`, "Deviations from the paper", for the
full explanation and its consequences.
