from typing import Optional

import numpy as np
import pandas as pd


class FactorAttribution:
    """Return attribution decomposing portfolio performance by source.

    Breaks down returns into: factor exposures, sentiment contribution,
    alpha (unexplained), and cost drag.
    """

    def __init__(self, risk_free_rate: float = 0.05, periods_per_year: int = 252):
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        self.ppy = periods_per_year

    def attribute_returns(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame = None,
        factor_betas: np.ndarray = None,
        sentiment_returns: pd.Series = None,
        cost_series: pd.Series = None,
    ) -> pd.DataFrame:
        """Period-by-period return attribution.

        Args:
            portfolio_returns: Daily portfolio returns.
            factor_returns: DataFrame of factor returns (Mkt-RF, SMB, HML, etc.).
            factor_betas: Portfolio's factor loadings (1D array matching columns).
            sentiment_returns: Return component attributed to sentiment signals.
            cost_series: Transaction costs incurred each period.

        Returns:
            DataFrame with columns for each attribution source.
        """
        idx = portfolio_returns.index
        attr = pd.DataFrame(index=idx)
        attr["total"] = portfolio_returns

        explained = pd.Series(0.0, index=idx)

        # Factor contribution
        if factor_returns is not None and factor_betas is not None:
            common = idx.intersection(factor_returns.index)
            for i, col in enumerate(factor_returns.columns):
                if i < len(factor_betas):
                    contrib = factor_betas[i] * factor_returns.loc[common, col]
                    attr[col] = contrib.reindex(idx, fill_value=0.0)
                    explained = explained.add(attr[col], fill_value=0.0)

        # Sentiment contribution
        if sentiment_returns is not None:
            attr["sentiment"] = sentiment_returns.reindex(idx, fill_value=0.0)
            explained = explained.add(attr["sentiment"], fill_value=0.0)

        # Cost drag
        if cost_series is not None:
            attr["cost_drag"] = -cost_series.reindex(idx, fill_value=0.0).abs()
            explained = explained.add(attr["cost_drag"], fill_value=0.0)

        # Alpha = unexplained residual
        attr["alpha"] = portfolio_returns - explained

        return attr

    def summary(self, attribution: pd.DataFrame) -> dict:
        """Summarize attribution into annualized contributions."""
        result = {}
        for col in attribution.columns:
            cumulative = attribution[col].sum()
            annualized = attribution[col].mean() * self.ppy
            result[col] = {
                "cumulative": float(cumulative),
                "annualized": float(annualized),
                "pct_of_total": float(
                    cumulative / attribution["total"].sum()
                    if attribution["total"].sum() != 0 else 0
                ),
            }
        return result

    def risk_attribution(
        self,
        asset_returns: pd.DataFrame,
        weights: np.ndarray,
        factor_returns: pd.DataFrame = None,
        factor_betas: pd.DataFrame = None,
    ) -> dict:
        """Decompose portfolio variance into factor and specific risk.

        Args:
            asset_returns: Per-asset daily returns.
            weights: Portfolio weights.
            factor_returns: Factor return series.
            factor_betas: Per-asset factor betas (assets x factors DataFrame).

        Returns:
            Dict with total, factor, and specific variance.
        """
        port_returns = asset_returns @ weights
        total_var = float(port_returns.var()) * self.ppy

        if factor_returns is None or factor_betas is None:
            return {"total_variance": total_var, "factor_variance": 0.0,
                    "specific_variance": total_var, "factor_pct": 0.0}

        # Portfolio factor loadings
        port_betas = weights @ factor_betas.values
        common = asset_returns.index.intersection(factor_returns.index)
        fac = factor_returns.loc[common]

        factor_cov = fac.cov().values * self.ppy
        factor_var = float(port_betas @ factor_cov @ port_betas)
        specific_var = max(total_var - factor_var, 0.0)

        return {
            "total_variance": total_var,
            "factor_variance": factor_var,
            "specific_variance": specific_var,
            "factor_pct": float(factor_var / total_var) if total_var > 0 else 0.0,
        }
