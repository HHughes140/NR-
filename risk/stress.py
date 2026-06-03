from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from NR.config import RiskConfig


# ── Insurance fundamental stress scenarios ────────────────────────
# Each scenario defines deltas to apply to factor z-scores.
# Positive = factor value increases; negative = decreases.
# These are calibrated to historical insurance industry events.

INSURANCE_SCENARIOS = {
    "catastrophe_year": {
        "description": "Major hurricane/earthquake year (e.g., Katrina 2005, Sandy 2012)",
        "shocks": {
            "combined_ratio": 0.20,       # +20pts (95% -> 115%)
            "loss_ratio": 0.15,           # +15pts
            "lae_ratio": 0.05,            # +5pts (claims cost more to settle)
            "reserve_ratio": 0.30,        # +30% reserves increase
            "premium_growth_yoy": 0.10,   # +10% post-cat rate hardening
            "operating_margin": -0.15,    # margin crushed
            "roe": -0.08,                 # ROE drops
            "book_value_growth": -0.05,   # surplus eroded
        },
    },
    "hard_market": {
        "description": "Insurance pricing cycle hardens (rates rise across industry)",
        "shocks": {
            "premium_growth_yoy": 0.15,   # +15% rate increases
            "combined_ratio": -0.05,      # -5pts improvement
            "expense_ratio": -0.02,       # operating leverage
            "operating_margin": 0.05,     # margin expansion
            "roe": 0.04,                  # profitability improves
            "book_value_growth": 0.03,    # surplus grows
        },
    },
    "soft_market": {
        "description": "Rate war -- competitors undercut pricing",
        "shocks": {
            "premium_growth_yoy": -0.05,  # -5% revenue decline
            "combined_ratio": 0.08,       # +8pts deterioration
            "expense_ratio": 0.02,        # fixed costs on lower base
            "loss_ratio": 0.05,           # adverse selection
            "operating_margin": -0.06,    # margin compression
            "roe": -0.03,                 # lower profitability
        },
    },
    "investment_crash": {
        "description": "Bond/equity portfolio marks down (2008-style)",
        "shocks": {
            "investment_yield": -0.03,    # -300bps yield collapse
            "investment_to_assets": -0.05, # portfolio shrinks
            "book_value_growth": -0.10,   # unrealized losses hit book
            "roe": -0.06,                 # total return drops
            "leverage_ratio": 0.20,       # leverage spikes (equity down)
        },
    },
    "reserve_strengthening": {
        "description": "Prior-year adverse reserve development",
        "shocks": {
            "reserve_ratio": 0.25,        # +25% reserve increase
            "combined_ratio": 0.10,       # +10pts from prior year charges
            "loss_ratio": 0.08,           # loss picks revised up
            "lae_ratio": 0.03,            # settlement costs rise
            "operating_margin": -0.08,    # margin hit
            "book_value_growth": -0.04,   # surplus charge
            "roe": -0.05,
        },
    },
    "rate_spike": {
        "description": "Rapid interest rate increase (Fed tightening)",
        "shocks": {
            "investment_yield": 0.02,     # +200bps new money yield
            "book_value_growth": -0.06,   # unrealized bond losses
            "leverage_ratio": 0.10,       # equity shrinks from marks
            "investment_to_assets": -0.02, # portfolio value drops
        },
    },
    "pandemic": {
        "description": "COVID-style event (business interruption + market crash)",
        "shocks": {
            "combined_ratio": 0.12,       # BI claims
            "loss_ratio": 0.10,
            "lae_ratio": 0.02,
            "reserve_ratio": 0.15,
            "investment_yield": -0.02,    # flight to safety
            "book_value_growth": -0.08,
            "roe": -0.07,
            "operating_margin": -0.10,
        },
    },
    "social_inflation": {
        "description": "Rising litigation costs and nuclear verdicts",
        "shocks": {
            "lae_ratio": 0.06,            # +6pts LAE from verdicts
            "loss_ratio": 0.04,           # higher ultimate losses
            "combined_ratio": 0.10,
            "reserve_ratio": 0.20,        # reserves inadequate
            "operating_margin": -0.05,
        },
    },
}


class StressTester:
    """Stress testing and scenario analysis.

    Supports three modes:
      1. Market stress: traditional equity drawdown scenarios.
      2. Fundamental stress: project return impact from hypothetical
         changes in insurance fundamentals using factor betas.
      3. Historical analog: find past quarters with similar fundamental
         shifts and measure what actually happened to stock prices.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    # ── Market stress (original) ───────────────────────────────

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
        betas = np.array([
            sp_stats.linregress(benchmark_returns, returns.iloc[:, i]).slope
            for i in range(returns.shape[1])
        ])
        return float(weights @ (betas * market_shock))

    def scenario_analysis(self, returns: pd.DataFrame, weights: np.ndarray,
                          benchmark_returns: pd.Series,
                          scenarios: dict[str, float] = None) -> pd.DataFrame:
        """Run multiple predefined market scenarios."""
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

        all_returns = rng.multivariate_normal(
            mean_returns, cov_matrix, size=(n_simulations, horizon_days)
        )
        port_daily = all_returns @ weights
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
            "expected_shortfall_5pct": float(np.mean(
                terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)])),
            "percentiles": percentiles,
        }

    # ── Fundamental stress (insurance factors) ─────────────────

    def fundamental_stress(self, factor_model, scenario: dict[str, float],
                           weights: np.ndarray = None) -> dict:
        """Project return impact from hypothetical fundamental changes.

        Uses the fitted InsuranceFundamentalFactorModel betas to estimate
        how a given set of factor shocks would affect each ticker and the
        portfolio.

        Args:
            factor_model: A fitted InsuranceFundamentalFactorModel instance.
            scenario: Dict mapping factor name -> delta (e.g.,
                      {"combined_ratio": 0.15, "roe": -0.05}).
            weights: Portfolio weights. If None, uses equal weight.

        Returns:
            Dict with per-ticker projected returns, portfolio impact,
            and factor-level breakdown.
        """
        if factor_model.betas is None and factor_model.factor_returns is None:
            raise ValueError("Factor model must be fitted first")

        tickers = factor_model.tickers
        n = len(tickers)
        if weights is None:
            weights = np.ones(n) / n

        # Build shock vector aligned to factor order
        factor_names = factor_model.factor_names
        shock_vector = np.array([
            scenario.get(f, 0.0) for f in factor_names
        ])

        # Per-ticker impact
        if factor_model.betas is not None:
            # Time-series mode: betas are per-ticker
            betas = factor_model.betas.fillna(0).values
            ticker_impacts = betas @ shock_vector
        else:
            # Cross-sectional mode: use average factor returns as proxy
            fr = factor_model.factor_returns
            avg_betas = fr.mean().values
            ticker_impacts = np.full(n, float(avg_betas @ shock_vector))

        # Factor-level contribution
        factor_contributions = {}
        for i, fname in enumerate(factor_names):
            if fname in scenario:
                if factor_model.betas is not None:
                    avg_beta = float(factor_model.betas[fname].mean())
                else:
                    avg_beta = float(factor_model.factor_returns[fname].mean())
                contribution = avg_beta * scenario[fname]
                factor_contributions[fname] = {
                    "shock": scenario[fname],
                    "avg_beta": avg_beta,
                    "contribution": contribution,
                }

        portfolio_impact = float(weights @ ticker_impacts)

        # Annualize: quarterly factor change -> annualized return impact
        # Factor changes are quarterly, but betas are fit on daily returns.
        # A quarterly fundamental shift persists ~63 trading days.
        annualized_impact = portfolio_impact * 252

        per_ticker = {}
        for i, t in enumerate(tickers):
            per_ticker[t] = {
                "projected_return": float(ticker_impacts[i]),
                "annualized": float(ticker_impacts[i] * 252),
                "weight": float(weights[i]),
                "weighted_contribution": float(
                    weights[i] * ticker_impacts[i]),
            }

        return {
            "portfolio_impact_daily": portfolio_impact,
            "portfolio_impact_annualized": annualized_impact,
            "per_ticker": per_ticker,
            "factor_contributions": factor_contributions,
            "scenario_applied": scenario,
        }

    def run_insurance_scenarios(self, factor_model,
                                weights: np.ndarray = None,
                                scenarios: dict = None) -> pd.DataFrame:
        """Run all predefined insurance stress scenarios.

        Args:
            factor_model: A fitted InsuranceFundamentalFactorModel.
            weights: Portfolio weights. None = equal weight.
            scenarios: Custom scenario dict. None = use INSURANCE_SCENARIOS.

        Returns:
            DataFrame with one row per scenario: name, description,
            portfolio impact (daily and annualized), and top contributing
            factors.
        """
        if scenarios is None:
            scenarios = INSURANCE_SCENARIOS

        results = []
        for name, spec in scenarios.items():
            shocks = spec["shocks"]
            description = spec.get("description", "")

            result = self.fundamental_stress(factor_model, shocks, weights)

            # Find top 3 contributing factors
            contribs = result["factor_contributions"]
            sorted_contribs = sorted(
                contribs.items(),
                key=lambda x: abs(x[1]["contribution"]),
                reverse=True,
            )
            top_factors = ", ".join(
                f"{f}({c['contribution']:+.4f})"
                for f, c in sorted_contribs[:3]
            )

            results.append({
                "scenario": name,
                "description": description,
                "portfolio_impact_daily": result["portfolio_impact_daily"],
                "portfolio_impact_annualized": result["portfolio_impact_annualized"],
                "n_factors_shocked": len(shocks),
                "top_contributors": top_factors,
            })

        return pd.DataFrame(results)

    # ── Historical analog search ───────────────────────────────

    def find_historical_analogs(self, factor_model,
                                scenario: dict[str, float],
                                tolerance: float = 0.5,
                                forward_days: list[int] = None
                                ) -> pd.DataFrame:
        """Find past quarters where fundamentals shifted similarly.

        Searches the quarterly data for periods where the factor changes
        were within tolerance of the specified scenario. Then measures
        actual stock returns in the following 1/3/6 months.

        Args:
            factor_model: A fitted InsuranceFundamentalFactorModel with
                          _quarterly_data and _returns populated.
            scenario: Dict mapping factor name -> delta to search for.
            tolerance: How close the historical change must be to the
                       scenario delta (in units of historical std dev of
                       changes). Lower = stricter matching.
            forward_days: Horizons to measure forward returns.
                          Default: [21, 63, 126] (1m, 3m, 6m).

        Returns:
            DataFrame with matched quarters, the actual factor changes,
            and realized forward returns at each horizon.
        """
        if forward_days is None:
            forward_days = [21, 63, 126]

        quarterly_data = factor_model._quarterly_data
        returns = factor_model._returns

        if quarterly_data is None or returns is None:
            raise ValueError(
                "Factor model must have _quarterly_data and _returns. "
                "Call fit() first.")

        scenario_factors = list(scenario.keys())
        matches = []

        for ticker, qdata in quarterly_data.items():
            if ticker not in returns.columns:
                continue

            # Compute quarter-over-quarter changes
            available = [f for f in scenario_factors if f in qdata.columns]
            if not available:
                continue

            changes = qdata[available].diff()
            change_std = changes.std()

            # Check each transition (row i-1 -> row i)
            for i in range(1, len(changes)):
                date = changes.index[i]
                row_changes = changes.iloc[i]

                # Score: how many factors are within tolerance
                matched_factors = 0
                total_distance = 0.0
                factor_detail = {}

                for factor in available:
                    actual_change = row_changes[factor]
                    if pd.isna(actual_change):
                        continue
                    target_change = scenario[factor]
                    std = change_std[factor]
                    if std < 1e-10:
                        std = abs(target_change) if abs(target_change) > 0 else 1
                    distance = abs(actual_change - target_change) / std
                    factor_detail[factor] = {
                        "actual": float(actual_change),
                        "target": target_change,
                        "distance_std": float(distance),
                    }
                    total_distance += distance
                    if distance <= tolerance:
                        matched_factors += 1

                if matched_factors == 0:
                    continue

                match_pct = matched_factors / len(available)
                avg_distance = total_distance / len(available)

                # Measure forward returns
                fwd_returns = {}
                ticker_returns = returns[ticker]
                date_loc = ticker_returns.index.searchsorted(date)
                for horizon in forward_days:
                    end_loc = date_loc + horizon
                    if end_loc < len(ticker_returns):
                        fwd = float(
                            ticker_returns.iloc[date_loc:end_loc].sum())
                        fwd_returns[f"fwd_{horizon}d"] = fwd
                    else:
                        fwd_returns[f"fwd_{horizon}d"] = np.nan

                matches.append({
                    "ticker": ticker,
                    "date": date,
                    "match_pct": match_pct,
                    "avg_distance_std": avg_distance,
                    "matched_factors": matched_factors,
                    "total_factors": len(available),
                    **fwd_returns,
                    **{f"delta_{k}": v["actual"]
                       for k, v in factor_detail.items()},
                })

        if not matches:
            return pd.DataFrame()

        df = pd.DataFrame(matches).sort_values(
            "avg_distance_std", ascending=True)
        return df

    def compare_projection_vs_history(self, factor_model,
                                       scenario: dict[str, float],
                                       weights: np.ndarray = None,
                                       tolerance: float = 0.75
                                       ) -> dict:
        """Compare beta projection against historical analogs.

        Runs both the linear projection and historical search, then
        compares them. If they agree, the model is reliable. If they
        diverge, the linear model is missing non-linear effects.

        Args:
            factor_model: Fitted InsuranceFundamentalFactorModel.
            scenario: Factor shocks to test.
            weights: Portfolio weights.
            tolerance: Strictness for analog matching.

        Returns:
            Dict with projection, analog stats, and agreement assessment.
        """
        # Linear projection
        projection = self.fundamental_stress(
            factor_model, scenario, weights)

        # Historical analogs
        analogs = self.find_historical_analogs(
            factor_model, scenario, tolerance=tolerance)

        result = {
            "projection": projection,
            "n_analogs_found": len(analogs),
        }

        if analogs.empty:
            result["analogs"] = None
            result["agreement"] = "no_analogs"
            result["warning"] = (
                "No historical analogs found. Linear projection is "
                "the only estimate -- treat with caution.")
            return result

        # Aggregate analog forward returns
        fwd_cols = [c for c in analogs.columns if c.startswith("fwd_")]
        analog_stats = {}
        for col in fwd_cols:
            vals = analogs[col].dropna()
            if len(vals) == 0:
                continue
            analog_stats[col] = {
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n": len(vals),
                "pct_negative": float((vals < 0).mean()),
            }

        result["analog_stats"] = analog_stats
        result["analogs"] = analogs

        # Agreement check: compare projection to 63-day analog mean
        projected_quarterly = projection["portfolio_impact_daily"] * 63
        if "fwd_63d" in analog_stats:
            analog_mean = analog_stats["fwd_63d"]["mean"]
            analog_std = analog_stats["fwd_63d"]["std"]

            if analog_std > 1e-10:
                z_diff = abs(projected_quarterly - analog_mean) / analog_std
            else:
                z_diff = 0.0

            result["projected_63d"] = projected_quarterly
            result["analog_mean_63d"] = analog_mean
            result["divergence_z"] = float(z_diff)

            if z_diff < 1.0:
                result["agreement"] = "strong"
            elif z_diff < 2.0:
                result["agreement"] = "moderate"
            else:
                result["agreement"] = "weak"
                result["warning"] = (
                    f"Linear projection ({projected_quarterly:.4f}) "
                    f"diverges from historical mean ({analog_mean:.4f}) "
                    f"by {z_diff:.1f} std devs. Non-linear effects "
                    f"likely present.")
        else:
            result["agreement"] = "insufficient_data"

        return result

    # ── Export ──────────────────────────────────────────────────

    def export_to_excel(self, factor_model,
                        weights: np.ndarray = None,
                        scenarios: dict = None,
                        output_path: str = "stress_results.xlsx") -> str:
        """Export all stress test results to a multi-sheet Excel workbook.

        Sheets:
          1. Scenarios     -- run_insurance_scenarios() output
          2. Per-Ticker    -- per-ticker projected returns for each scenario
          3. Factor Detail -- factor-level contribution breakdown
          4. Analogs       -- historical analog matches (if found)

        Args:
            factor_model: A fitted InsuranceFundamentalFactorModel.
            weights: Portfolio weights. None = equal weight.
            scenarios: Custom scenarios dict. None = INSURANCE_SCENARIOS.
            output_path: File path for the .xlsx output.

        Returns:
            Absolute path of the written file.
        """
        if scenarios is None:
            scenarios = INSURANCE_SCENARIOS

        scenario_df = self.run_insurance_scenarios(
            factor_model, weights, scenarios)

        # Per-ticker and factor detail for each scenario
        ticker_rows = []
        factor_rows = []

        for name, spec in scenarios.items():
            shocks = spec["shocks"]
            result = self.fundamental_stress(factor_model, shocks, weights)

            for ticker, info in result["per_ticker"].items():
                ticker_rows.append({
                    "scenario": name,
                    "ticker": ticker,
                    "projected_return_daily": info["projected_return"],
                    "projected_return_annualized": info["annualized"],
                    "weight": info["weight"],
                    "weighted_contribution": info["weighted_contribution"],
                })

            for fname, finfo in result["factor_contributions"].items():
                factor_rows.append({
                    "scenario": name,
                    "factor": fname,
                    "shock_applied": finfo["shock"],
                    "avg_beta": finfo["avg_beta"],
                    "contribution": finfo["contribution"],
                })

        ticker_df = pd.DataFrame(ticker_rows)
        factor_df = pd.DataFrame(factor_rows)

        # Historical analogs for each scenario
        analog_rows = []
        for name, spec in scenarios.items():
            shocks = spec["shocks"]
            analogs = self.find_historical_analogs(
                factor_model, shocks, tolerance=0.75)
            if not analogs.empty:
                analogs = analogs.copy()
                analogs.insert(0, "scenario", name)
                analog_rows.append(analogs)

        analog_df = (pd.concat(analog_rows, ignore_index=True)
                     if analog_rows else pd.DataFrame())

        path = Path(output_path).resolve()
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            scenario_df.to_excel(writer, sheet_name="Scenarios", index=False)
            if not ticker_df.empty:
                ticker_df.to_excel(
                    writer, sheet_name="Per-Ticker", index=False)
            if not factor_df.empty:
                factor_df.to_excel(
                    writer, sheet_name="Factor Detail", index=False)
            if not analog_df.empty:
                analog_df.to_excel(
                    writer, sheet_name="Analogs", index=False)

        return str(path)
