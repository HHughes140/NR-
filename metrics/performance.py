from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from NR.metrics.drawdown import DrawdownAnalyzer


class PerformanceMetrics:
    """Comprehensive portfolio performance metrics.

    All methods accept daily returns as pd.Series.
    """

    def __init__(self, risk_free_rate: float = 0.05, periods_per_year: int = 252):
        self.rf = risk_free_rate
        self.ppy = periods_per_year
        self.rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        excess = returns - self.rf_daily
        if excess.std() == 0:
            return 0.0
        return excess.mean() / excess.std() * np.sqrt(self.ppy)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (penalizes only downside deviation)."""
        excess = returns - self.rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = np.sqrt((downside ** 2).mean())
        if downside_std == 0:
            return np.inf
        return excess.mean() / downside_std * np.sqrt(self.ppy)

    def alpha_beta(self, returns: pd.Series,
                   benchmark_returns: pd.Series) -> tuple[float, float]:
        """CAPM alpha (annualized) and beta via OLS regression."""
        slope, intercept, _, _, _ = stats.linregress(benchmark_returns, returns)
        beta = slope
        alpha = intercept * self.ppy  # annualize daily alpha
        return alpha, beta

    def information_ratio(self, returns: pd.Series,
                          benchmark_returns: pd.Series) -> float:
        """Annualized information ratio: active return / tracking error."""
        active = returns - benchmark_returns
        if active.std() == 0:
            return 0.0
        return active.mean() / active.std() * np.sqrt(self.ppy)

    def treynor_ratio(self, returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
        """Annualized Treynor ratio: excess return / beta."""
        _, beta = self.alpha_beta(returns, benchmark_returns)
        if beta == 0:
            return 0.0
        annual_return = (1 + returns.mean()) ** self.ppy - 1
        return (annual_return - self.rf) / beta

    def calmar_ratio(self, returns: pd.Series) -> float:
        """Annualized Calmar ratio: return / |max drawdown|."""
        annual_return = (1 + returns.mean()) ** self.ppy - 1
        mdd = DrawdownAnalyzer.max_drawdown(returns)
        if mdd == 0:
            return np.inf
        return annual_return / abs(mdd)

    def compute_all(self, returns: pd.Series,
                    benchmark_returns: Optional[pd.Series] = None) -> dict:
        """Compute all available metrics."""
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** self.ppy - 1
        annual_vol = returns.std() * np.sqrt(self.ppy)

        result = {
            "total_return": total_return,
            "annualized_return": annual_return,
            "annualized_volatility": annual_vol,
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "max_drawdown": DrawdownAnalyzer.max_drawdown(returns),
            "calmar_ratio": self.calmar_ratio(returns),
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "positive_days_pct": float((returns > 0).mean()),
        }

        if benchmark_returns is not None:
            alpha, beta = self.alpha_beta(returns, benchmark_returns)
            result["alpha"] = alpha
            result["beta"] = beta
            result["information_ratio"] = self.information_ratio(returns, benchmark_returns)
            result["treynor_ratio"] = self.treynor_ratio(returns, benchmark_returns)

        return result
