"""Bitcoin daily and weekly closing prices from the QuantX Dataloader REST API.

The paper's exact daily series (last Mt. Gox trade of each UTC day until
31 December 2012, then Bitstamp's official daily close, built from the
bitcoincharts.com archive preserved by the Internet Archive, SHA-256
5552aa23e37dd5d972a5956c44aa6682a7aa60ab2fe3bbf06e99a9e5023dacf7) is not
served by the Dataloader or Massive: a smoke test of X:BTCUSD found no
Massive coverage before 1 November 2013 (Massive's own crypto history starts
there), so the pre-registered 2011-2013 segment cannot be reconstructed from
this path. This is a genuine data-coverage limitation (see
results/RESULTS.md, "Deviations from the paper"), not a convenience
substitution: the realized sample starts once continuous daily coverage is
available, keeping the paper's own Saturday-anchor sampling rule, age
definition and response unchanged from that point on.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed

GENESIS_TIMESTAMP = pd.Timestamp("2009-01-03T18:15:05Z")


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _fetch_ohlcvs(ticker: str, start_date: str, end_date: str) -> dict:
    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    body = {
        "tickers": [ticker],
        "timeframe": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
    }
    headers = _headers()
    response = None
    for attempt in range(3):
        response = requests.post(API_ROOT + "/fetch-ohlcvs", json=body, headers=headers,
                                  timeout=120)
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2 ** attempt))
    if response.status_code != 200:
        raise DataloaderError(f"fetch-ohlcvs {ticker} {start_date}..{end_date}: "
                               f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def load_btc_daily_closes(start_date: str, end_date: str, ticker: str = "X:BTCUSD",
                           use_cache: bool = True) -> pd.DataFrame:
    """Daily BTC/USD closing prices (UTC date, close), from Massive via the
    Dataloader. Returns a DataFrame with columns `date` (UTC calendar date) and
    `close`.
    """
    key = hashlib.sha256(f"{ticker}|{start_date}|{end_date}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"btc_daily_{key}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    answer = _fetch_ohlcvs(ticker, start_date, end_date)
    result = answer["result"]
    details = result.get("ticker_details") or []
    failed = [d for d in details if d.get("error")]
    if failed:
        raise DataloaderError(f"{ticker}: {failed}")
    rows = result.get("ohlcv_df", {}).get("rows")
    if not rows:
        raise DataloaderError(f"empty OHLCV result for {ticker} {start_date}..{end_date}")
    wide = pd.DataFrame(rows)
    out = pd.DataFrame({
        "date": pd.to_datetime(wide["timestamp"], utc=True).dt.tz_localize(None).dt.normalize(),
        "close": wide["close"].astype(float),
    }).dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    meta_path = CACHE_DIR / f"btc_daily_{key}_meta.json"
    meta_path.write_text(json.dumps({
        "ticker": ticker, "start_date": start_date, "end_date": end_date,
        "ticker_details": details, "n_rows": len(out),
    }, indent=2, default=str))
    return out


def saturdays_between(start: str, end: str) -> list[pd.Timestamp]:
    """Every Saturday (inclusive) from `start` to `end` (paper Section 2)."""
    rng = pd.date_range(start=start, end=end, freq="W-SAT")
    return list(rng)


def build_weekly_series(daily: pd.DataFrame, start: str, end: str,
                         dropped_anchors: tuple[str, ...] = ()) -> pd.DataFrame:
    """Sample every Saturday's UTC close as the weekly anchor (paper Section 2).

    Returns columns: date (Saturday), close, age_weeks (s_t, weeks from genesis
    block to end of that day), y (log close). An anchor with no close (or
    explicitly dropped, e.g. 25 June 2011) is omitted, matching the paper's
    "one anchor has no close and is dropped" rule.
    """
    daily = daily.set_index("date")["close"]
    anchors = saturdays_between(start, end)
    dropped = {pd.Timestamp(d) for d in dropped_anchors}
    records = []
    for sat in anchors:
        if sat in dropped:
            continue
        if sat not in daily.index:
            continue
        close = daily.loc[sat]
        age_weeks = (sat + pd.Timedelta(hours=23, minutes=59, seconds=59)
                     - GENESIS_TIMESTAMP.tz_localize(None)).total_seconds() / (7 * 24 * 3600)
        records.append((sat, float(close), age_weeks))
    out = pd.DataFrame(records, columns=["date", "close", "age_weeks"])
    out["y"] = np.log(out["close"])
    return out


def load_primary_weekly_series(start: str, end: str, dropped_anchors: tuple[str, ...] = (),
                                use_cache: bool = True) -> pd.DataFrame:
    """The 820-anchor weekly series used by exp_1, exp_3, exp_4, exp_5, exp_6
    (realized sample: see module docstring for the data-coverage deviation).
    """
    daily = load_btc_daily_closes(start, end, use_cache=use_cache)
    return build_weekly_series(daily, start, end, dropped_anchors)
