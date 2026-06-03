import numpy as np
import pandas as pd
from scipy.stats import norm


class ConditionalVaR:
    """Conditional VaR (Expected Shortfall / CVaR).

    Average loss given that loss exceeds VaR threshold.
    """

    @staticmethod
    def parametric(returns: pd.Series, confidence: float = 0.95,
                   portfolio_value: float = 1.0) -> float:
        """Gaussian CVaR."""
        mu = returns.mean()
        sigma = returns.std()
        alpha = 1 - confidence
        z_alpha = norm.ppf(alpha)
        # ES = -mu + sigma * phi(z_alpha) / alpha
        es = -mu + sigma * norm.pdf(z_alpha) / alpha
        return es * portfolio_value

    @staticmethod
    def historical(returns: pd.Series, confidence: float = 0.95,
                   portfolio_value: float = 1.0) -> float:
        """Historical CVaR: mean of returns below VaR threshold."""
        threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= threshold]
        if len(tail_returns) == 0:
            return 0.0
        return -tail_returns.mean() * portfolio_value

    @staticmethod
    def monte_carlo(mean_return: float, cov_matrix: np.ndarray,
                    weights: np.ndarray, confidence: float = 0.95,
                    portfolio_value: float = 1.0,
                    n_simulations: int = 10_000) -> float:
        """Monte Carlo CVaR from simulated tail."""
        n_assets = len(weights)
        mean_vec = np.full(n_assets, mean_return) if np.isscalar(mean_return) else np.array(mean_return)

        rng = np.random.default_rng(42)
        simulated = rng.multivariate_normal(mean_vec, cov_matrix,
                                            size=n_simulations)
        portfolio_returns = simulated @ weights
        threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
        tail = portfolio_returns[portfolio_returns <= threshold]
        if len(tail) == 0:
            return 0.0
        return -tail.mean() * portfolio_value
