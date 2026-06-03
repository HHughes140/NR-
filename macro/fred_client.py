
from typing import Optional

import pandas as pd

from NR.config import MacroConfig
from NR.exceptions import FREDError


class FREDClient:
    """Fetches macroeconomic data from FRED API. Requires fredapi (lazy import)."""

    def __init__(self, api_key: str, config: MacroConfig = None):
        self.api_key = api_key
        self.config = config or MacroConfig()
        self._fred = None

    def _get_fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except ImportError:
                raise FREDError(
                    "fredapi not installed. Install with: pip install fredapi")
        return self._fred

    def fetch_series(self, series_id: str, start_date: str = None,
                     end_date: str = None) -> pd.Series:
        """Fetch a single FRED series."""
        try:
            fred = self._get_fred()
            data = fred.get_series(series_id,
                                   observation_start=start_date,
                                   observation_end=end_date)
            return data.dropna()
        except Exception as e:
            raise FREDError(f"Failed to fetch {series_id}: {e}") from e

    def fetch_all_macro(self, start_date: str = None) -> pd.DataFrame:
        """Fetch all configured FRED series into a single DataFrame."""
        frames = {}
        for name, series_id in self.config.fred_series.items():
            try:
                frames[name] = self.fetch_series(series_id, start_date)
            except FREDError:
                continue

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df.ffill()  # Forward-fill different publication frequencies
        return df

    def get_yield_spread(self, start_date: str = None) -> pd.Series:
        """10Y-2Y yield spread."""
        return self.fetch_series("T10Y2Y", start_date=start_date)

    def get_credit_spread(self, start_date: str = None) -> pd.Series:
        """ICE BofA US High Yield OAS (BAMLH0A0HYM2)."""
        return self.fetch_series("BAMLH0A0HYM2", start_date=start_date)

    def get_ted_spread(self, start_date: str = None) -> pd.Series:
        """TED spread (TEDRATE)."""
        return self.fetch_series("TEDRATE", start_date=start_date)

    def get_latest_values(self) -> dict:
        """Most recent value for each configured series."""
        result = {}
        for name, series_id in self.config.fred_series.items():
            try:
                s = self.fetch_series(series_id)
                result[name] = float(s.iloc[-1])
            except (FREDError, IndexError):
                continue
        return result
