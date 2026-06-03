import numpy as np
from scipy.optimize import minimize

from NR.config import OptimizationConfig
from NR.backtest.transaction_costs import TransactionCostModel


class CostAwareOptimizer:
    """Optimization that penalizes turnover from current positions."""

    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                 current_weights: np.ndarray, tickers: list,
                 risk_free_rate: float = 0.05,
                 cost_model: TransactionCostModel = None,
                 config: OptimizationConfig = None,
                 periods_per_year: int = 252):
        self.mu = expected_returns
        self.cov = cov_matrix
        self.w_current = current_weights
        self.tickers = tickers
        self.rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        self.cost_model = cost_model or TransactionCostModel()
        self.config = config or OptimizationConfig()

    def optimize(self, turnover_penalty: float = 0.01,
                 portfolio_value: float = 100_000.0) -> dict:
        """Optimize weights considering turnover cost.

        turnover_penalty: higher = more reluctant to trade.
        """
        n = len(self.mu)

        def objective(w):
            # Negative Sharpe + turnover penalty
            port_return = w @ self.mu
            port_vol = np.sqrt(w @ self.cov @ w)
            sharpe = (port_return - self.rf_daily) / port_vol if port_vol > 1e-10 else 0
            turnover = np.sum(np.abs(w - self.w_current)) / 2
            return -sharpe + turnover_penalty * turnover

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        x0 = self.w_current.copy()

        result = minimize(objective, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-15})

        new_weights = result.x
        turnover = np.sum(np.abs(new_weights - self.w_current)) / 2
        cost = self.cost_model.portfolio_rebalance_cost(
            self.w_current, new_weights, portfolio_value)

        def _sharpe(w):
            r = w @ self.mu
            v = np.sqrt(w @ self.cov @ w)
            return (r - self.rf_daily) / v if v > 1e-10 else 0

        sharpe_before = _sharpe(self.w_current)
        sharpe_after = _sharpe(new_weights)

        return {
            "weights": new_weights,
            "weights_dict": dict(zip(self.tickers, new_weights)),
            "turnover": turnover,
            "estimated_cost": cost,
            "sharpe_before": sharpe_before,
            "sharpe_after": sharpe_after,
            "sharpe_net_of_costs": sharpe_after - cost / portfolio_value,
        }
