"""Parquet-based market data cache for incremental daily updates.

Saves OHLCV and macro DataFrames to parquet files so the live trading
script can load cached history and fetch only new day(s) instead of
re-downloading 10 years of data every run.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Files stored in the cache directory
_FILES = {
    "prices": "prices.parquet",
    "open_prices": "open_prices.parquet",
    "high_prices": "high_prices.parquet",
    "low_prices": "low_prices.parquet",
    "volume": "volume.parquet",
}
_SERIES_FILES = {
    "spy": "spy.parquet",
    "vix": "vix.parquet",
    "yield_spread": "yield_spread.parquet",
    "dxy": "dxy.parquet",
    "credit_spread": "credit_spread.parquet",
    "kbe": "kbe.parquet",
}
_METADATA = "metadata.json"


def save_market_data(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    volume: pd.DataFrame,
    spy: Optional[pd.Series],
    vix: Optional[pd.Series],
    yield_spread: Optional[pd.Series],
    dxy: Optional[pd.Series],
    credit_spread: Optional[pd.Series],
    cache_dir: str | Path,
    kbe: Optional[pd.Series] = None,
) -> None:
    """Save all market data to parquet files in cache_dir."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # DataFrames
    frames = {
        "prices": prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "volume": volume,
    }
    for key, df in frames.items():
        if df is not None:
            df.to_parquet(cache_dir / _FILES[key])

    # Series
    series = {
        "spy": spy,
        "vix": vix,
        "yield_spread": yield_spread,
        "dxy": dxy,
        "credit_spread": credit_spread,
        "kbe": kbe,
    }
    for key, s in series.items():
        if s is not None:
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.to_frame(name=key).to_parquet(cache_dir / _SERIES_FILES[key])

    # Metadata
    last_date = str(prices.index[-1].date()) if len(prices) > 0 else ""
    meta = {
        "last_date": last_date,
        "tickers": list(prices.columns),
        "n_assets": len(prices.columns),
        "n_days": len(prices),
        "saved_at": datetime.now().isoformat(),
    }
    with open(cache_dir / _METADATA, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Market data saved: %d assets, %d days, last=%s",
                len(prices.columns), len(prices), last_date)


def load_market_data(cache_dir: str | Path) -> tuple:
    """Load cached market data from parquet files.

    Returns:
        (prices, open_prices, high_prices, low_prices, volume,
         spy, vix, yield_spread, dxy, credit_spread, metadata)
    """
    cache_dir = Path(cache_dir)

    # DataFrames
    frames = {}
    for key, fname in _FILES.items():
        path = cache_dir / fname
        frames[key] = pd.read_parquet(path) if path.exists() else None

    # Series
    series = {}
    for key, fname in _SERIES_FILES.items():
        path = cache_dir / fname
        if path.exists():
            df = pd.read_parquet(path)
            series[key] = df.iloc[:, 0]
        else:
            series[key] = None

    # Metadata
    meta_path = cache_dir / _METADATA
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    logger.info("Market data loaded: %d assets, %d days",
                frames["prices"].shape[1] if frames["prices"] is not None else 0,
                frames["prices"].shape[0] if frames["prices"] is not None else 0)

    return (
        frames["prices"],
        frames["open_prices"],
        frames["high_prices"],
        frames["low_prices"],
        frames["volume"],
        series["spy"],
        series["vix"],
        series["yield_spread"],
        series["dxy"],
        series["credit_spread"],
        series.get("kbe"),
        metadata,
    )


def get_cache_last_date(cache_dir: str | Path) -> Optional[str]:
    """Return the last date in the cache, or None if no cache."""
    meta_path = Path(cache_dir) / _METADATA
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("last_date")


def fetch_incremental(
    tickers: list[str],
    last_date: str,
    fred_api_key: str = "",
) -> tuple:
    """Fetch only new data since last_date.

    Returns same tuple format as fetch_full_history in run_live.py:
    (prices, open_p, high_p, low_p, volume, spy, vix, ys, dxy, credit_spread, kbe)
    All may be empty DataFrames/None if no new data.
    """
    from NR.data.fetcher import MarketDataFetcher
    from NR.config import DataConfig

    start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    if pd.Timestamp(start) > pd.Timestamp(end):
        logger.info("No new data: last_date=%s >= today", last_date)
        return None, None, None, None, None, None, None, None, None, None, None

    config = DataConfig(period="1y", chunk_size=20, max_retries=3,
                        trading_days_per_year=252)
    fetcher = MarketDataFetcher(config)

    # Fetch OHLCV — use start/end dates instead of period
    import yfinance as yf
    open_p = close_p = high_p = low_p = vol = None

    try:
        data = yf.download(tickers, start=start, end=end,
                           group_by="ticker", auto_adjust=True,
                           threads=True, progress=False)
        if data.empty:
            logger.info("No new trading days since %s", last_date)
            return None, None, None, None, None, None, None, None, None, None, None

        if len(tickers) == 1:
            close_p = data[["Close"]].rename(columns={"Close": tickers[0]})
            open_p = data[["Open"]].rename(columns={"Open": tickers[0]})
            high_p = data[["High"]].rename(columns={"High": tickers[0]})
            low_p = data[["Low"]].rename(columns={"Low": tickers[0]})
            vol = data[["Volume"]].rename(columns={"Volume": tickers[0]})
        else:
            close_p = data.xs("Close", level=1, axis=1) if "Close" in data.columns.get_level_values(1) else None
            open_p = data.xs("Open", level=1, axis=1) if "Open" in data.columns.get_level_values(1) else None
            high_p = data.xs("High", level=1, axis=1) if "High" in data.columns.get_level_values(1) else None
            low_p = data.xs("Low", level=1, axis=1) if "Low" in data.columns.get_level_values(1) else None
            vol = data.xs("Volume", level=1, axis=1) if "Volume" in data.columns.get_level_values(1) else None
    except Exception as e:
        logger.warning("Incremental OHLCV fetch failed: %s", e)
        return None, None, None, None, None, None, None, None, None, None, None

    if close_p is None or close_p.empty:
        return None, None, None, None, None, None, None, None, None, None, None

    # Macro
    spy = vix = ys = dxy = None
    try:
        spy_data = yf.download("SPY", start=start, end=end,
                               auto_adjust=True, progress=False)
        if not spy_data.empty:
            spy = spy_data["Close"]
            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]
            spy.name = "SPY"
    except Exception:
        pass

    try:
        macro = yf.download(["^VIX", "^TNX", "UUP"], start=start, end=end,
                            auto_adjust=True, progress=False)
        if not macro.empty and len(macro.columns.names) > 1:
            closes = macro.xs("Close", level=1, axis=1)
            for col in closes.columns:
                s = closes[col]
                if "VIX" in col.upper():
                    vix = s
                elif "TNX" in col.upper():
                    ys = s
                elif "UUP" in col.upper():
                    dxy = s
    except Exception:
        pass

    credit_spread = None
    if fred_api_key:
        try:
            from NR.macro.fred_client import FREDClient
            fred = FREDClient(api_key=fred_api_key)
            cs = fred.get_credit_spread()
            cs = cs.reindex(close_p.index, method="ffill").ffill().bfill()
            cs.name = "HY_OAS"
            credit_spread = cs
        except Exception:
            pass

    kbe = None
    try:
        kbe_data = yf.download("KBE", start=start, end=end,
                               auto_adjust=True, progress=False)
        if not kbe_data.empty:
            kbe = kbe_data["Close"]
            if isinstance(kbe, pd.DataFrame):
                kbe = kbe.iloc[:, 0]
            kbe.name = "KBE"
    except Exception:
        pass

    n_days = len(close_p)
    logger.info("Incremental fetch: %d new trading days (%s to %s)",
                n_days, close_p.index[0].date(), close_p.index[-1].date())

    return close_p, open_p, high_p, low_p, vol, spy, vix, ys, dxy, credit_spread, kbe


def append_data(
    cached: tuple,
    new_data: tuple,
) -> tuple:
    """Append new day(s) to cached data. Deduplicates by date index.

    Both arguments are tuples of (prices, open_p, high_p, low_p, volume,
    spy, vix, yield_spread, dxy, credit_spread, kbe).

    Returns tuple in same format.
    """
    result = []
    for i in range(11):  # 11 data elements
        old = cached[i]
        new = new_data[i]

        if old is None and new is None:
            result.append(None)
        elif old is None:
            result.append(new)
        elif new is None:
            result.append(old)
        else:
            combined = pd.concat([old, new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            result.append(combined)

    return tuple(result)
