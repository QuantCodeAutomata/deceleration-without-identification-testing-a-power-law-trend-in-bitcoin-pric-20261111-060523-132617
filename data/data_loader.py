"""Bitcoin daily and weekly closing prices.

The paper's exact daily series (last Mt. Gox trade of each UTC day until
31 December 2012, then Bitstamp's official daily close) is not served by the
Dataloader or Massive: a smoke test of X:BTCUSD found no Massive coverage
before 1 November 2013 (Massive's own crypto history starts there). The
2011-2013 segment is instead rebuilt from the same bitcoincharts.com archive
the paper cites, preserved by the Internet Archive's Wayback Machine
(`mtgoxUSD.csv.gz` and `bitstampUSD.csv.gz`, trade-level CSV dumps of
unix-timestamp, price, volume). The last trade of each UTC day gives the Mt.
Gox daily close through 2012-12-31; the last trade of each UTC day in the
archived Bitstamp dump gives the daily close from 2013-01-01 through the day
before Massive's own BTC/USD coverage begins (2013-10-31); Massive/Dataloader
X:BTCUSD supplies every day from 2013-11-01 onward. This matches the paper's
construction (Section 2) except that the exact frozen CSV (SHA-256
5552aa23e37dd5d972a5956c44aa6682a7aa60ab2fe3bbf06e99a9e5023dacf7) is not
reachable; the Wayback Machine snapshot used here is a different capture of
the same public archive (see results/RESULTS.md, "Deviations from the
paper").
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed

GENESIS_TIMESTAMP = pd.Timestamp("2009-01-03T18:15:05Z")

# bitcoincharts.com trade dumps, preserved by the Internet Archive (paper Section 2).
ARCHIVE_URLS = {
    "mtgox": "https://web.archive.org/web/2016id_/http://api.bitcoincharts.com/v1/csv/mtgoxUSD.csv.gz",
    "bitstamp": "https://web.archive.org/web/2016id_/http://api.bitcoincharts.com/v1/csv/bitstampUSD.csv.gz",
}
MTGOX_LAST_DAY = date(2012, 12, 31)  # last day of Mt. Gox segment (paper Section 2)
BITSTAMP_ARCHIVE_START = date(2013, 1, 1)
BITSTAMP_ARCHIVE_END = date(2013, 10, 31)  # day before Massive's own BTC/USD coverage begins


def _load_archive_daily_closes(exchange: str, start: date, end: date) -> dict[date, float]:
    """Last trade of each UTC day between `start` and `end` (inclusive), streamed
    from the exchange's bitcoincharts.com dump without ever holding the whole
    multi-million-row file in memory (only one (date, price) pair is kept)."""
    cache_path = CACHE_DIR / f"archive_{exchange}_{start}_{end}.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return {date.fromisoformat(k): v for k, v in raw.items()}

    response = requests.get(ARCHIVE_URLS[exchange], stream=True, timeout=180)
    if response.status_code != 200:
        raise DataloaderError(f"archive {exchange}: HTTP {response.status_code}")
    daily: dict[date, float] = {}
    with gzip.GzipFile(fileobj=response.raw) as gz:
        for line in gz:
            parts = line.decode("ascii", "ignore").strip().split(",")
            if len(parts) != 3:
                continue
            try:
                ts = int(parts[0])
                price = float(parts[1])
            except ValueError:
                continue
            d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            if d < start:
                continue
            if d > end:
                break
            daily[d] = price  # later trades in the same day overwrite earlier ones
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({k.isoformat(): v for k, v in daily.items()}))
    return daily


def load_archived_segment(use_cache: bool = True) -> pd.DataFrame:
    """Mt. Gox (through 2012-12-31) then Bitstamp (2013-01-01 through
    2013-10-31) daily closes from the bitcoincharts.com archive (paper
    Section 2). Returns columns `date`, `close`.
    """
    cache_path = CACHE_DIR / "archived_segment.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])
    mtgox = _load_archive_daily_closes("mtgox", date(2011, 1, 1), MTGOX_LAST_DAY)
    bitstamp = _load_archive_daily_closes("bitstamp", BITSTAMP_ARCHIVE_START, BITSTAMP_ARCHIVE_END)
    records = [(pd.Timestamp(d), p) for d, p in sorted({**mtgox, **bitstamp}.items())]
    out = pd.DataFrame(records, columns=["date", "close"]).sort_values("date").reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


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


def load_spliced_daily_closes(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily BTC/USD closes spliced from the archived Mt. Gox/Bitstamp segment
    (through 2013-10-31) and Massive/Dataloader X:BTCUSD (2013-11-01 onward),
    matching the paper's exchange sequence (Section 2). `start`/`end` bound the
    requested window; the archive is only consulted for the portion before
    2013-11-01.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    massive_start = pd.Timestamp("2013-11-01")
    frames = []
    if start_ts < massive_start:
        archived = load_archived_segment(use_cache=use_cache)
        archived = archived[(archived["date"] >= start_ts) & (archived["date"] < massive_start)]
        frames.append(archived)
    if end_ts >= massive_start:
        massive_daily = load_btc_daily_closes(
            max(start_ts, massive_start).strftime("%Y-%m-%d"), end, use_cache=use_cache)
        frames.append(massive_daily)
    out = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return out.drop_duplicates(subset="date", keep="last")


def load_primary_weekly_series(start: str, end: str, dropped_anchors: tuple[str, ...] = (),
                                use_cache: bool = True) -> pd.DataFrame:
    """The 820-anchor weekly series used by exp_1, exp_3, exp_4, exp_5, exp_6,
    spliced from the archived Mt. Gox/Bitstamp segment and Massive/Dataloader
    X:BTCUSD (see module docstring for the data-source deviation from the
    paper's exact frozen CSV).
    """
    daily = load_spliced_daily_closes(start, end, use_cache=use_cache)
    return build_weekly_series(daily, start, end, dropped_anchors)
