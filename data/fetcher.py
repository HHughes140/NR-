import logging
import time
import warnings
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

from NR.config import DataConfig
from NR.exceptions import DataFetchError, DataValidationError


class MarketDataFetcher:
    """Fetches and caches market data from yfinance."""

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self._cache: dict[str, pd.DataFrame] = {}

    def _resample_to_4h(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample 1h OHLCV bars to 4h bars using last value per block."""
        resampled = data.resample("4h").last()
        return resampled.dropna(how="all")

    def _chunked_download(self, tickers: list[str]) -> pd.DataFrame:
        """Download data for many tickers in chunks with retry.

        Splits large ticker lists into batches to avoid yfinance
        timeouts and rate limits. Retries failed batches with
        exponential backoff.
        """
        chunk_size = self.config.chunk_size
        max_retries = self.config.max_retries
        period = self.config.period
        interval = self.config.interval

        # yfinance limits intraday data to ~730 days for 1h
        if interval in ("1h", "60m"):
            valid_intraday = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y")
            if period not in valid_intraday:
                logger.info("Capping period to 2y for intraday interval %s", interval)
                period = "2y"

        # For small lists, download directly (threads OK for small N)
        if len(tickers) <= chunk_size:
            return yf.download(
                tickers, period=period,
                interval=interval, auto_adjust=True,
                progress=False,
            )

        # For large universes, disable yfinance internal threading to avoid
        # thread exhaustion (getaddrinfo failures) and SQLite cache lock
        # contention (OperationalError) when downloading 500+ tickers.
        all_data = []
        n_chunks = (len(tickers) + chunk_size - 1) // chunk_size
        for chunk_idx, i in enumerate(range(0, len(tickers), chunk_size)):
            chunk = tickers[i:i + chunk_size]
            logger.info("Downloading chunk %d/%d (%d tickers: %s...)",
                        chunk_idx + 1, n_chunks, len(chunk), chunk[0])

            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        chunk, period=period,
                        interval=interval, auto_adjust=True,
                        progress=False, threads=False,
                    )
                    if not data.empty:
                        all_data.append(data)
                        break
                except Exception as e:
                    logger.warning("Chunk %d attempt %d failed: %s",
                                   chunk_idx + 1, attempt + 1, e)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + 1)
                        continue

            if i + chunk_size < len(tickers):
                time.sleep(1.5)

        if not all_data:
            raise DataFetchError(f"All chunks failed for {len(tickers)} tickers")

        if len(all_data) == 1:
            return all_data[0]

        combined = pd.concat(all_data, axis=1)
        return combined

    def _clean_prices(self, prices: pd.DataFrame,
                       min_history_pct: float = 0.80) -> pd.DataFrame:
        """Clean price matrix for large universes.

        1. Drop tickers with zero data.
        2. Forward-fill gaps up to 2 days.
        3. Drop tickers that still lack min_history_pct of rows
           (e.g. recent IPOs in a 10-year window).
        4. Drop any remaining rows with NaN.

        This keeps the full time series rather than truncating
        to the most recently listed ticker.
        """
        n_rows = len(prices)

        # 1. Drop all-NaN columns
        valid_cols = prices.columns[prices.notna().any()]
        no_data = set(prices.columns) - set(valid_cols)
        if no_data:
            warnings.warn(f"Dropped {len(no_data)} tickers with no data: {sorted(no_data)}")
        prices = prices[valid_cols]

        # 2. Forward-fill small gaps
        prices = prices.ffill(limit=2)

        # 3. Drop tickers without enough history
        coverage = prices.notna().sum() / n_rows
        short_history = coverage[coverage < min_history_pct].index.tolist()
        if short_history:
            logger.info("Dropped %d tickers with <%.0f%% history: %s",
                        len(short_history), min_history_pct * 100,
                        short_history[:10])
            prices = prices.drop(columns=short_history)

        # 4. Drop remaining NaN rows
        prices = prices.dropna()
        return prices

    def _validate_prices(self, prices: pd.DataFrame):
        """Run data quality checks on cleaned price matrix.

        Checks for negative prices, extreme daily moves, and
        low data coverage. Logs warnings for non-fatal issues
        and raises DataValidationError for critical failures.
        """
        if prices.empty:
            raise DataValidationError("Price matrix is empty after cleaning")

        # Check for negative or zero prices
        neg_mask = prices <= 0
        if neg_mask.any().any():
            bad_tickers = list(prices.columns[neg_mask.any()])
            raise DataValidationError(
                f"Negative or zero prices found in: {bad_tickers}")

        # Check for NaN/inf
        if not prices.apply(lambda s: pd.to_numeric(s, errors='coerce')).notnull().all().all():
            raise DataValidationError("Non-numeric values in price data")

        # Warn on extreme daily moves (>50% in one day)
        daily_returns = prices.pct_change().dropna()
        extreme = (daily_returns.abs() > 0.50)
        if extreme.any().any():
            n_extreme = int(extreme.sum().sum())
            bad = list(prices.columns[extreme.any()])
            logger.warning(
                "%d extreme daily moves (>50%%) detected in: %s",
                n_extreme, bad[:10])

        # Warn if very few observations
        if len(prices) < 30:
            logger.warning(
                "Only %d price observations — results may be unreliable",
                len(prices))

    def fetch_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch adjusted close prices for multiple tickers.

        Returns DataFrame indexed by date, columns = tickers.
        """
        if not tickers:
            raise DataValidationError("Ticker list must be non-empty")
        if len(tickers) != len(set(tickers)):
            dupes = [t for t in tickers if tickers.count(t) > 1]
            logger.warning("Duplicate tickers removed: %s", set(dupes))
            tickers = list(dict.fromkeys(tickers))

        cache_key = f"prices_{'_'.join(sorted(tickers))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            data = self._chunked_download(tickers)
        except Exception as e:
            raise DataFetchError(f"Failed to fetch prices for {tickers}: {e}") from e

        if data.empty:
            raise DataFetchError(f"No data returned for {tickers}")

        # yf.download returns MultiIndex columns when multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers

        # Resample 1h → 4h if timeframe is "4h"
        if getattr(self.config, 'timeframe', '1d') == '4h':
            prices = self._resample_to_4h(prices)

        prices = self._clean_prices(prices)
        self._validate_prices(prices)
        self._cache[cache_key] = prices
        return prices

    def fetch_prices_and_volume(self, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch close prices and volume for multiple tickers."""
        cache_key = f"pv_{'_'.join(sorted(tickers))}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached["prices"], cached["volume"]

        try:
            data = self._chunked_download(tickers)
        except Exception as e:
            raise DataFetchError(f"Failed to fetch data for {tickers}: {e}") from e

        if data.empty:
            raise DataFetchError(f"No data returned for {tickers}")

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
            volume = data["Volume"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers
            volume = data[["Volume"]]
            volume.columns = tickers

        # Resample 1h → 4h if timeframe is "4h"
        if getattr(self.config, 'timeframe', '1d') == '4h':
            prices = self._resample_to_4h(prices)
            volume = self._resample_to_4h(volume)

        prices = self._clean_prices(prices)
        volume = volume[prices.columns].reindex(prices.index).fillna(0)

        self._cache[cache_key] = {"prices": prices, "volume": volume}
        return prices, volume

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Fetch full OHLCV for a single ticker."""
        t = yf.Ticker(ticker)
        data = t.history(period=self.config.period, interval=self.config.interval)
        if data.empty:
            raise DataFetchError(f"No OHLCV data for {ticker}")
        return data

    def fetch_ohlcv_bulk(self, tickers: list[str]) -> tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch OHLCV for all tickers in one download.

        Returns (open_prices, close_prices, high_prices, low_prices, volume)
        DataFrames aligned by date.
        """
        cache_key = f"ohlcv_bulk_{'_'.join(sorted(tickers))}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return (cached["open"], cached["close"],
                    cached["high"], cached["low"], cached["volume"])

        try:
            data = self._chunked_download(tickers)
        except Exception as e:
            raise DataFetchError(f"Failed to fetch OHLCV for {tickers}: {e}") from e

        if data.empty:
            raise DataFetchError(f"No OHLCV data returned for {tickers}")

        if isinstance(data.columns, pd.MultiIndex):
            open_prices = data["Open"]
            close_prices = data["Close"]
            high_prices = data["High"]
            low_prices = data["Low"]
            volume = data["Volume"]
        else:
            open_prices = data[["Open"]]
            open_prices.columns = tickers
            close_prices = data[["Close"]]
            close_prices.columns = tickers
            high_prices = data[["High"]]
            high_prices.columns = tickers
            low_prices = data[["Low"]]
            low_prices.columns = tickers
            volume = data[["Volume"]]
            volume.columns = tickers

        close_prices = self._clean_prices(close_prices)
        common_idx = close_prices.index
        surviving_tickers = close_prices.columns
        open_prices = open_prices[surviving_tickers].reindex(common_idx).ffill()
        high_prices = high_prices[surviving_tickers].reindex(common_idx).ffill()
        low_prices = low_prices[surviving_tickers].reindex(common_idx).ffill()
        volume = volume[surviving_tickers].reindex(common_idx).fillna(0)

        self._cache[cache_key] = {
            "open": open_prices, "close": close_prices,
            "high": high_prices, "low": low_prices, "volume": volume,
        }
        return open_prices, close_prices, high_prices, low_prices, volume

    def fetch_benchmark(self) -> pd.DataFrame:
        """Fetch benchmark prices."""
        return self.fetch_prices([self.config.benchmark])

    def fetch_option_chain(self, ticker: str,
                           expiration: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch option chain (calls, puts) for a ticker."""
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            raise DataFetchError(f"No options data for {ticker}")

        exp = expiration or expirations[0]
        chain = t.option_chain(exp)
        return chain.calls, chain.puts

    def fetch_option_expirations(self, ticker: str) -> list[str]:
        """Available option expiration dates."""
        return list(yf.Ticker(ticker).options)

    def fetch_market_caps(self, tickers: list[str]) -> pd.Series:
        """Fetch market caps."""
        caps = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                caps[ticker] = info.get("marketCap", 0)
            except Exception:
                caps[ticker] = 0
        return pd.Series(caps)

    def get_risk_free_rate(self) -> float:
        """Fetch current risk-free rate proxy (13-week T-bill)."""
        try:
            data = yf.download("^IRX", period="5d", progress=False)
            if not data.empty:
                rate_col = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
                return float(rate_col.iloc[-1]) / 100.0
        except Exception:
            pass
        return self.config.risk_free_rate

    def fetch_macro_tickers(self, macro_tickers: list[str],
                            period: str = "2y") -> pd.DataFrame:
        """Fetch macro indicator prices (VIX, TNX, DXY, etc.)."""
        try:
            data = yf.download(macro_tickers, period=period, progress=False)
            if data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                return data["Close"].dropna()
            return data[["Close"]].dropna()
        except Exception:
            return pd.DataFrame()

    def fetch_rss_headlines(self, ticker: str) -> list[dict]:
        """Fetch Yahoo Finance RSS headlines for a ticker."""
        import urllib.request
        import xml.etree.ElementTree as ET

        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            return [{"title": item.findtext("title", ""),
                     "pubDate": item.findtext("pubDate", ""),
                     "ticker": ticker}
                    for item in root.iter("item") if item.findtext("title")]
        except Exception:
            return []

    def clear_cache(self):
        self._cache.clear()
