"""Shared pytest fixtures and marker registration.

`market_data` tests load a reduced slice of real Bitcoin prices through
`data/data_loader.py` and are skipped when `DATALOADER_API_KEY` is absent.
"""
from __future__ import annotations

import os

import pytest

from data.data_loader import load_primary_weekly_series


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: requires DATALOADER_API_KEY/MASSIVE_API_KEY and network access"
    )


def _has_market_data_credentials() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def reduced_weekly_series(request: pytest.FixtureRequest):
    """A short, cached slice of the real weekly Bitcoin series (about one year)."""
    if not _has_market_data_credentials():
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set: market_data test skipped")
    cache = request.config.cache
    key = "qca/reduced_weekly_series"
    cached = cache.get(key, None)
    if cached is not None:
        import pandas as pd
        return pd.DataFrame(cached)
    df = load_primary_weekly_series("2014-01-04", "2015-01-03")
    cache.set(key, df.assign(date=df["date"].astype(str)).to_dict("list"))
    return df
