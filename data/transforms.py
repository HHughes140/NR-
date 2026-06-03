import numpy as np
import pandas as pd

from NR.exceptions import DataValidationError


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute returns from price DataFrame.

    Args:
        method: "simple" for (P1-P0)/P0, "log" for ln(P1/P0).
    """
    if method not in ("simple", "log"):
        raise DataValidationError(f"Unknown return method: {method}")
    if len(prices) < 2:
        raise DataValidationError(
            f"Need at least 2 price observations, got {len(prices)}")
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def align_series(*series: pd.DataFrame) -> list[pd.DataFrame]:
    """Align multiple DataFrames to a common date index (inner join)."""
    if not series:
        raise DataValidationError("align_series requires at least one input")
    common_index = series[0].index
    for s in series[1:]:
        common_index = common_index.intersection(s.index)
    common_index = common_index.sort_values()
    return [s.loc[common_index] for s in series]


def resample_returns(returns: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Resample daily returns to lower frequency via compounding."""
    return returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)


def rolling_returns(prices: pd.Series, window: int) -> pd.Series:
    """Compute rolling N-period return."""
    return prices / prices.shift(window) - 1


def annualize_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    """Convert total return over `periods` to annualized return."""
    years = periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def annualize_volatility(daily_vol: float, periods_per_year: int = 252) -> float:
    """Annualize daily standard deviation."""
    return daily_vol * np.sqrt(periods_per_year)
