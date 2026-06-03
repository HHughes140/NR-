from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from NR.config import OptimizationConfig
from NR.optimization.weights import PortfolioWeights


class EfficientFrontier:
    """Full Modern Portfolio Theory implementation.

    Computes the efficient frontier, tangency portfolio, minimum variance
    portfolio, and capital market line.
    """

    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                 asset_names: list[str], risk_free_rate: float = 0.05,
                 config: OptimizationConfig = None,
                 periods_per_year: int = 252):
        self.mu = np.array(expected_returns)
        self.cov = np.array(cov_matrix)
        self.names = asset_names
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        self.config = config or OptimizationConfig()

        self.frontier_returns: Optional[np.ndarray] = None
        self.frontier_volatilities: Optional[np.ndarray] = None
        self.frontier_weights: Optional[np.ndarray] = None
        self.frontier_sharpes: Optional[np.ndarray] = None
        self.tangency_weights: Optional[np.ndarray] = None
        self.min_var_weights: Optional[np.ndarray] = None

    def compute(self) -> "EfficientFrontier":
        """Compute the full efficient frontier."""
        n = len(self.mu)
        n_points = self.config.frontier_points

        # Find min-variance and max-return bounds
        self.min_var_weights = PortfolioWeights.minimum_variance(
            self.cov, allow_short=self.config.allow_short)
        min_ret = self.min_var_weights @ self.mu

        if self.config.allow_short:
            max_ret = self.mu.max() * 1.5
        else:
            max_ret = self.mu.max()

        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_vols = []
        frontier_weights = []

        bounds = None
        if not self.config.allow_short:
            bounds = [(self.config.min_weight, self.config.max_weight)] * n

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ self.mu - t},
            ]
            x0 = np.ones(n) / n
            result = minimize(lambda w: w @ self.cov @ w, x0,
                              method="SLSQP", bounds=bounds,
                              constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-12})

            if result.success:
                vol = np.sqrt(result.x @ self.cov @ result.x)
                frontier_vols.append(vol)
                frontier_weights.append(result.x)
            else:
                frontier_vols.append(np.nan)
                frontier_weights.append(np.full(n, np.nan))

        self.frontier_returns = target_returns
        self.frontier_volatilities = np.array(frontier_vols)
        self.frontier_weights = np.array(frontier_weights)

        # Sharpe ratios along frontier
        self.frontier_sharpes = np.where(
            self.frontier_volatilities > 0,
            (self.frontier_returns - self.rf_daily) / self.frontier_volatilities,
            0.0,
        )

        # Tangency portfolio
        self.tangency_weights = PortfolioWeights.maximum_sharpe(
            self.mu, self.cov, self.rf, allow_short=self.config.allow_short)

        return self

    def minimum_variance_portfolio(self) -> dict:
        if self.min_var_weights is None:
            self.min_var_weights = PortfolioWeights.minimum_variance(
                self.cov, allow_short=self.config.allow_short)
        return self.portfolio_stats(self.min_var_weights)

    def tangency_portfolio(self) -> dict:
        if self.tangency_weights is None:
            self.tangency_weights = PortfolioWeights.maximum_sharpe(
                self.mu, self.cov, self.rf, allow_short=self.config.allow_short)
        return self.portfolio_stats(self.tangency_weights)

    def capital_market_line(self, max_vol: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """CML from risk-free rate through tangency portfolio."""
        tang = self.tangency_portfolio()
        tang_ret = tang["return"]
        tang_vol = tang["volatility"]

        if max_vol is None:
            max_vol = tang_vol * 2.0

        vols = np.linspace(0, max_vol, 100)
        slope = (tang_ret - self.rf_daily) / tang_vol
        rets = self.rf_daily + slope * vols
        return vols, rets

    def optimal_portfolio(self, target_return: Optional[float] = None,
                          target_volatility: Optional[float] = None) -> dict:
        """Find optimal portfolio for a given target return or volatility."""
        n = len(self.mu)
        bounds = None
        if not self.config.allow_short:
            bounds = [(self.config.min_weight, self.config.max_weight)] * n

        if target_return is not None:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: w @ self.mu - target_return},
            ]
            result = minimize(lambda w: w @ self.cov @ w, np.ones(n) / n,
                              method="SLSQP", bounds=bounds,
                              constraints=constraints)
        elif target_volatility is not None:
            def neg_return(w):
                return -(w @ self.mu)

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "ineq", "fun": lambda w: target_volatility ** 2 - w @ self.cov @ w},
            ]
            result = minimize(neg_return, np.ones(n) / n,
                              method="SLSQP", bounds=bounds,
                              constraints=constraints)
        else:
            raise ValueError("Specify exactly one of target_return or target_volatility")

        return self.portfolio_stats(result.x)

    def portfolio_stats(self, weights: np.ndarray) -> dict:
        ret = float(weights @ self.mu)
        vol = float(np.sqrt(weights @ self.cov @ weights))
        sharpe = (ret - self.rf_daily) / vol if vol > 0 else 0.0
        return {
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe,
            "weights": dict(zip(self.names, weights)),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Efficient frontier as a DataFrame."""
        data = {
            "return": self.frontier_returns,
            "volatility": self.frontier_volatilities,
            "sharpe": self.frontier_sharpes,
        }
        for i, name in enumerate(self.names):
            data[f"w_{name}"] = self.frontier_weights[:, i]
        return pd.DataFrame(data)
