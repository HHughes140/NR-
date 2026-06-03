import numpy as np

from NR.config import BacktestConfig


class TransactionCostModel:
    """Models trading costs: fixed, proportional, market impact (sqrt), slippage."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def compute_cost(self, trade_value: float, adv: float = None) -> float:
        """Total cost for a single trade."""
        trade_abs = abs(trade_value)
        fixed = self.config.fixed_cost_per_trade
        proportional = trade_abs * self.config.proportional_cost_bps / 10_000
        slippage = trade_abs * self.config.slippage_bps / 10_000

        impact = 0.0
        if adv is not None and adv > 0:
            impact = self.config.market_impact_coefficient * np.sqrt(
                trade_abs / adv) * trade_abs

        return fixed + proportional + impact + slippage

    def portfolio_rebalance_cost(self, current_weights: np.ndarray,
                                  target_weights: np.ndarray,
                                  portfolio_value: float,
                                  adv: np.ndarray = None) -> float:
        """Total cost of rebalancing from current to target weights."""
        trades = np.abs(target_weights - current_weights) * portfolio_value
        total = 0.0
        for i, trade_val in enumerate(trades):
            if trade_val < 1e-6:
                continue
            a = adv[i] if adv is not None else None
            total += self.compute_cost(trade_val, a)
        return total

    def cost_breakdown(self, trade_value: float, adv: float = None) -> dict:
        """Itemized cost breakdown."""
        trade_abs = abs(trade_value)
        fixed = self.config.fixed_cost_per_trade
        proportional = trade_abs * self.config.proportional_cost_bps / 10_000
        slippage = trade_abs * self.config.slippage_bps / 10_000
        impact = 0.0
        if adv is not None and adv > 0:
            impact = self.config.market_impact_coefficient * np.sqrt(
                trade_abs / adv) * trade_abs
        return {
            "fixed": fixed, "proportional": proportional,
            "market_impact": impact, "slippage": slippage,
            "total": fixed + proportional + impact + slippage,
        }

    @staticmethod
    def compute_borrow_cost(
        weights: np.ndarray,
        portfolio_value: float,
        borrow_annual_bps: float = 50.0,
    ) -> float:
        """Daily borrow cost for short positions.

        Args:
            weights: N-vector of portfolio weights (negative = short).
            portfolio_value: Current portfolio value in dollars.
            borrow_annual_bps: Annual borrow cost in basis points.

        Returns:
            Daily dollar cost of carrying short positions.
        """
        short_notional = float(np.sum(np.abs(np.minimum(weights, 0.0)))) * portfolio_value
        daily_rate = borrow_annual_bps / 10_000 / 252
        return short_notional * daily_rate

    def almgren_chriss_cost(
        self,
        trade_value: float,
        sigma: float,
        adv: float,
        spread_bps: float = 5.0,
        commission_bps: float = 1.0,
        eta: float = 0.1,
        gamma: float = 0.05,
        impact_exponent: float = 0.6,
    ) -> dict:
        """Almgren-Chriss market impact model.

        Args:
            trade_value: Dollar value of trade (signed).
            sigma: Daily volatility of the asset (price * daily_std).
            adv: Average daily dollar volume.
            spread_bps: Spread cost in basis points.
            commission_bps: Commission in basis points.
            eta: Temporary impact coefficient.
            gamma: Permanent impact coefficient.
            impact_exponent: Exponent on participation rate.

        Returns dict with spread, commission, temporary_impact,
        permanent_impact, participation_rate, total.
        """
        abs_trade = abs(trade_value)
        if abs_trade < 1e-6:
            return {"spread": 0.0, "commission": 0.0,
                    "temporary_impact": 0.0, "permanent_impact": 0.0,
                    "participation_rate": 0.0, "total": 0.0}

        participation_rate = abs_trade / max(adv, 1.0)

        spread = abs_trade * spread_bps / 10_000
        commission = abs_trade * commission_bps / 10_000
        temp_impact = eta * sigma * (participation_rate ** impact_exponent) * abs_trade
        perm_impact = gamma * sigma * participation_rate * abs_trade

        total = spread + commission + temp_impact + perm_impact
        return {
            "spread": spread, "commission": commission,
            "temporary_impact": temp_impact, "permanent_impact": perm_impact,
            "participation_rate": participation_rate, "total": total,
        }
