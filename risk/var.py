import numpy as np
import pandas as pd
from scipy.stats import norm


class ValueAtRisk:
    """Value at Risk using parametric, historical, and Monte Carlo methods."""

    @staticmethod
    def parametric(returns: pd.Series, confidence: float = 0.95,
                   portfolio_value: float = 1.0) -> float:
        """Gaussian VaR assuming normal distribution."""
        mu = returns.mean()
        sigma = returns.std()
        z = norm.ppf(1 - confidence)
        return -(mu + z * sigma) * portfolio_value

    @staticmethod
    def historical(returns: pd.Series, confidence: float = 0.95,
                   portfolio_value: float = 1.0) -> float:
        """Historical VaR from empirical return distribution."""
        var_pct = np.percentile(returns, (1 - confidence) * 100)
        return -var_pct * portfolio_value

    @staticmethod
    def monte_carlo(mean_return: float, cov_matrix: np.ndarray,
                    weights: np.ndarray, confidence: float = 0.95,
                    portfolio_value: float = 1.0,
                    n_simulations: int = 10_000, horizon: int = 1) -> float:
        """Monte Carlo VaR via multivariate normal simulation."""
        n_assets = len(weights)
        mean_vec = np.full(n_assets, mean_return) if np.isscalar(mean_return) else np.array(mean_return)

        # Simulate multivariate normal returns
        rng = np.random.default_rng(42)
        simulated = rng.multivariate_normal(mean_vec * horizon, cov_matrix * horizon,
                                            size=n_simulations)
        portfolio_returns = simulated @ weights
        var_pct = np.percentile(portfolio_returns, (1 - confidence) * 100)
        return -var_pct * portfolio_value

    @staticmethod
    def component_var(returns: pd.DataFrame, weights: np.ndarray,
                      confidence: float = 0.95) -> pd.Series:
        """Component VaR decomposition showing each asset's contribution."""
        cov = np.array(returns.cov())
        port_vol = np.sqrt(weights @ cov @ weights)
        z = norm.ppf(confidence)
        portfolio_var = z * port_vol

        # Marginal VaR
        marginal = (cov @ weights) / port_vol * z
        # Component VaR = weight * marginal VaR
        component = weights * marginal

        return pd.Series(component, index=returns.columns, name="component_var")

    @staticmethod
    def marginal_var(returns: pd.DataFrame, weights: np.ndarray,
                     confidence: float = 0.95) -> pd.Series:
        """Marginal VaR: sensitivity of portfolio VaR to weight changes."""
        cov = np.array(returns.cov())
        port_vol = np.sqrt(weights @ cov @ weights)
        z = norm.ppf(confidence)
        marginal = (cov @ weights) / port_vol * z

        return pd.Series(marginal, index=returns.columns, name="marginal_var")
