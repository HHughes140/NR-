from typing import Optional

import numpy as np
import pandas as pd

from NR.metrics.performance import PerformanceMetrics
from NR.metrics.drawdown import DrawdownAnalyzer


class BacktestReport:
    """Generate comprehensive backtest reports from engine results."""

    def __init__(self, results: dict, periods_per_year: int = 252):
        self.results = results
        self.equity = results["equity_curve"]
        self.returns = results["returns"]
        self.spy_equity = results.get("spy_equity_curve")
        self.spy_returns = results.get("spy_returns")
        self.validation_log = results.get("validation_log", {})
        self.ppy = periods_per_year
        self.perf = PerformanceMetrics(periods_per_year=periods_per_year)

    def summary(self) -> dict:
        """One-page summary of backtest results."""
        metrics = self.results["metrics"]
        return {
            "initial_capital": self.results["initial_capital"],
            "final_value": self.results["final_value"],
            "total_return": float(self.equity.iloc[-1] / self.equity.iloc[0] - 1),
            "annualized_return": metrics["annualized_return"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar_ratio": metrics["calmar_ratio"],
            "total_cost": self.results["total_cost"],
            "cost_drag_annual": self.results["cost_drag_annual"],
            "total_turnover": self.results["total_turnover"],
            "n_rebalances": len(self.results["weights_history"]),
            "positive_days_pct": metrics["positive_days_pct"],
            "best_day": metrics["best_day"],
            "worst_day": metrics["worst_day"],
        }

    def monthly_returns(self) -> pd.DataFrame:
        """Monthly return table (rows = years, cols = months)."""
        monthly = self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.year, monthly.index.month],
            names=["year", "month"])
        return monthly.unstack(level="month")

    def drawdown_analysis(self) -> dict:
        """Detailed drawdown statistics."""
        dd = DrawdownAnalyzer.drawdown_series(self.returns)
        mdd = DrawdownAnalyzer.max_drawdown(self.returns)
        duration = DrawdownAnalyzer.max_drawdown_duration(self.returns)
        top = DrawdownAnalyzer.top_drawdowns(self.returns, n=5)
        current_dd = float(dd["drawdown"].iloc[-1]) if len(dd) > 0 else 0.0
        return {
            "max_drawdown": mdd,
            "max_drawdown_duration_days": duration,
            "top_5_drawdowns": top,
            "current_drawdown": current_dd,
        }

    def cost_analysis(self) -> dict:
        """Transaction cost statistics."""
        costs = self.results["cost_history"]
        if not costs:
            return {"total": 0, "mean_per_rebalance": 0, "max_single": 0}
        cost_vals = [c["cost"] for c in costs]
        turnover_vals = [c["turnover"] for c in costs]
        return {
            "total_cost": sum(cost_vals),
            "mean_per_rebalance": float(np.mean(cost_vals)),
            "max_single_cost": float(np.max(cost_vals)),
            "mean_turnover": float(np.mean(turnover_vals)),
            "max_turnover": float(np.max(turnover_vals)),
        }

    def decision_scoring(self) -> pd.DataFrame:
        """Score each rebalance decision by subsequent performance."""
        decisions = self.results["decision_log"]
        if not decisions:
            return pd.DataFrame()

        records = []
        equity = self.equity
        for i, dec in enumerate(decisions):
            date = dec["date"]
            if date not in equity.index:
                continue

            val_at = float(equity.loc[date])
            # Forward return to next rebalance or end
            future = equity.loc[date:]
            if len(future) < 2:
                continue

            # 21-day forward return (or whatever is available)
            end_idx = min(21, len(future) - 1)
            fwd_return = float(future.iloc[end_idx] / future.iloc[0] - 1)

            records.append({
                "date": date,
                "regime": dec.get("regime"),
                "turnover": dec["turnover"],
                "cost": dec["cost"],
                "forward_return_21d": fwd_return,
                "cost_adjusted_return": fwd_return - dec["cost"] / val_at,
            })

        return pd.DataFrame(records).set_index("date")

    def benchmark_comparison(self) -> dict:
        """Compare portfolio vs SPY buy-and-hold."""
        if self.spy_returns is None:
            return {"skipped": True}

        perf = PerformanceMetrics(periods_per_year=self.ppy)
        portfolio_metrics = perf.compute_all(self.returns, self.spy_returns)
        spy_metrics = perf.compute_all(self.spy_returns)

        aligned_spy = self.spy_returns.reindex(self.returns.index, fill_value=0)
        active_returns = self.returns - aligned_spy
        tracking_error = float(active_returns.std() * np.sqrt(self.ppy))

        port_total = float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)
        spy_total = float(self.spy_equity.iloc[-1] / self.spy_equity.iloc[0] - 1)

        return {
            "portfolio_total_return": port_total,
            "spy_total_return": spy_total,
            "portfolio_sharpe": portfolio_metrics["sharpe_ratio"],
            "spy_sharpe": spy_metrics["sharpe_ratio"],
            "alpha": portfolio_metrics.get("alpha", 0),
            "beta": portfolio_metrics.get("beta", 1),
            "information_ratio": portfolio_metrics.get("information_ratio", 0),
            "tracking_error": tracking_error,
        }

    def validation_report(self) -> dict:
        """Framework validation metrics: L lead-lag, kappa regime stats."""
        result = {}

        L_hist = self.validation_log.get("L_history", [])
        kappa_hist = self.validation_log.get("kappa_history", [])

        if L_hist:
            L_values = [h["L"] for h in L_hist]
            result["L_mean"] = float(np.mean(L_values))
            result["L_std"] = float(np.std(L_values))
            result["L_max"] = float(np.max(L_values))

            # L -> volatility lead-lag correlations
            try:
                L_series = pd.Series(
                    L_values,
                    index=pd.DatetimeIndex([h["date"] for h in L_hist]))
                vol_series = self.returns.rolling(21).std() * np.sqrt(self.ppy)
                vol_aligned = vol_series.reindex(L_series.index, method="nearest")
                for lag in [1, 5, 10, 21]:
                    shifted = L_series.shift(lag)
                    corr = shifted.corr(vol_aligned)
                    if not np.isnan(corr):
                        result[f"L_vol_corr_lag{lag}d"] = round(float(corr), 4)
            except Exception:
                pass

        if kappa_hist:
            kappa_values = [h["kappa"] for h in kappa_hist]
            result["kappa_mean"] = float(np.mean(kappa_values))
            result["kappa_std"] = float(np.std(kappa_values))
            n_total = len(kappa_values)
            result["pct_momentum_regime"] = float(
                sum(1 for k in kappa_values if k > 1.2) / n_total)
            result["pct_mean_reversion_regime"] = float(
                sum(1 for k in kappa_values if k < 0.8) / n_total)

        return result

    def text_report(self) -> str:
        """Human-readable text summary."""
        s = self.summary()
        lines = [
            "=" * 60,
            "WALK-FORWARD BACKTEST REPORT",
            "=" * 60,
            f"Period: {self.equity.index[0].strftime('%Y-%m-%d')} to "
            f"{self.equity.index[-1].strftime('%Y-%m-%d')}",
            f"Initial Capital:     ${s['initial_capital']:>14,.2f}",
            f"Final Value:         ${s['final_value']:>14,.2f}",
            f"Total Return:        {s['total_return']:>14.2%}",
            f"Annualized Return:   {s['annualized_return']:>14.2%}",
            f"Annualized Vol:      {s['annualized_volatility']:>14.2%}",
            f"Sharpe Ratio:        {s['sharpe_ratio']:>14.3f}",
            f"Sortino Ratio:       {s['sortino_ratio']:>14.3f}",
            f"Max Drawdown:        {s['max_drawdown']:>14.2%}",
            f"Calmar Ratio:        {s['calmar_ratio']:>14.3f}",
            "",
            "--- Costs ---",
            f"Total Trading Costs: ${s['total_cost']:>14,.2f}",
            f"Annual Cost Drag:    {s['cost_drag_annual']:>14.4%}",
            f"Total Turnover:      {s['total_turnover']:>14.2f}x",
            f"Rebalances:          {s['n_rebalances']:>14d}",
            "",
            "--- Daily Stats ---",
            f"Positive Days:       {s['positive_days_pct']:>14.1%}",
            f"Best Day:            {s['best_day']:>14.2%}",
            f"Worst Day:           {s['worst_day']:>14.2%}",
            "=" * 60,
        ]

        # Benchmark comparison (if SPY data available)
        if self.spy_returns is not None:
            try:
                bench = self.benchmark_comparison()
                if not bench.get("skipped"):
                    lines.extend([
                        "",
                        "--- vs SPY Buy-and-Hold ---",
                        f"SPY Total Return:    {bench['spy_total_return']:>14.2%}",
                        f"Alpha (annualized):  {bench['alpha']:>14.4f}",
                        f"Beta:                {bench['beta']:>14.3f}",
                        f"Information Ratio:   {bench['information_ratio']:>14.3f}",
                        f"Tracking Error:      {bench['tracking_error']:>14.2%}",
                    ])
            except Exception:
                pass

        return "\n".join(lines)
