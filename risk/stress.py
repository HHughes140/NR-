import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from NR.config import RiskConfig


class StressTester:
    """Stress testing and scenario analysis."""

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def historical_scenario(self, returns: pd.DataFrame, weights: np.ndarray,
                            start_date: str, end_date: str) -> dict:
        """Replay a historical period on the current portfolio."""
        period = returns.loc[start_date:end_date]
        if period.empty:
            return {"portfolio_return": 0.0, "max_drawdown": 0.0,
                    "worst_day": 0.0, "days": 0}

        port_returns = period @ weights
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return {
            "portfolio_return": float(cumulative.iloc[-1] - 1),
            "max_drawdown": float(drawdown.min()),
            "worst_day": float(port_returns.min()),
            "days": len(period),
        }

    def factor_shock(self, cov_matrix: np.ndarray, weights: np.ndarray,
                     benchmark_returns: pd.Series, returns: pd.DataFrame,
                     market_shock: float) -> float:
        """Estimate portfolio loss from a market factor shock using beta."""
        # Compute betas for each asset
        betas = np.array([
            sp_stats.linregress(benchmark_returns, returns.iloc[:, i]).slope
            for i in range(returns.shape[1])
        ])
        return float(weights @ (betas * market_shock))

    def scenario_analysis(self, returns: pd.DataFrame, weights: np.ndarray,
                          benchmark_returns: pd.Series,
                          scenarios: dict[str, float] = None) -> pd.DataFrame:
        """Run multiple predefined scenarios."""
        scenarios = scenarios or self.config.stress_scenarios
        cov = np.array(returns.cov())

        results = []
        for name, shock in scenarios.items():
            port_loss = self.factor_shock(cov, weights, benchmark_returns,
                                          returns, shock)
            results.append({
                "scenario": name,
                "market_return": shock,
                "portfolio_return": port_loss,
                "portfolio_loss_pct": port_loss,
            })
        return pd.DataFrame(results)

    def monte_carlo_stress(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                           weights: np.ndarray, n_simulations: int = 10_000,
                           horizon_days: int = 252) -> dict:
        """Full Monte Carlo simulation over a given horizon."""
        rng = np.random.default_rng(42)
        n_assets = len(weights)

        # Simulate all paths at once: (n_simulations, horizon_days, n_assets)
        all_returns = rng.multivariate_normal(
            mean_returns, cov_matrix, size=(n_simulations, horizon_days)
        )
        # Portfolio daily returns: (n_sims, horizon)
        port_daily = all_returns @ weights
        # Cumulative wealth paths
        paths = np.cumprod(1 + port_daily, axis=1)

        terminal = paths[:, -1]
        terminal_returns = terminal - 1

        percentiles = {
            "5th": float(np.percentile(terminal_returns, 5)),
            "25th": float(np.percentile(terminal_returns, 25)),
            "50th": float(np.percentile(terminal_returns, 50)),
            "75th": float(np.percentile(terminal_returns, 75)),
            "95th": float(np.percentile(terminal_returns, 95)),
        }

        return {
            "paths": paths,
            "terminal_distribution": terminal_returns,
            "probability_of_loss": float(np.mean(terminal_returns < 0)),
            "expected_shortfall_5pct": float(np.mean(terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)])),
            "percentiles": percentiles,
        }
