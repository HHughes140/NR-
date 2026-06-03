"""Insurance company fundamental factor model.

Regresses insurance stock returns against insurance-specific fundamentals
(combined ratio, LAE, reserves, premium growth, investment yield, etc.)
to decompose price movements into fundamental-driven and idiosyncratic
components.

Handles the quarterly-to-daily frequency mismatch via step-function
forward-fill (point-in-time, no look-ahead bias).
"""

from typing import Optional
import logging
import time

import numpy as np
import pandas as pd

from NR.config import InsuranceFactorConfig
from NR.exceptions import InsuranceDataError

logger = logging.getLogger(__name__)


# ── yfinance field mappings ──────────────────────────────────────────

INCOME_FIELDS = {
    "total_revenue": ["Total Revenue", "Operating Revenue"],
    "lae": ["Loss Adjustment Expense"],
    "claims": ["Net Policyholder Benefits And Claims",
               "Policyholder Benefits Gross"],
    "sga": ["Selling General And Administration",
            "General And Administrative Expense"],
    "total_expenses": ["Total Expenses"],
    "net_interest_income": ["Net Interest Income"],
    "other_income": ["Other Income Expense"],
    "ebit": ["EBIT"],
    "net_income": ["Net Income"],
}

BALANCE_FIELDS = {
    "total_assets": ["Total Assets"],
    "investments": ["Investments And Advances"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_debt": ["Total Debt", "Long Term Debt"],
    "tangible_book_value": ["Tangible Book Value"],
}


def _safe_get(df: pd.DataFrame, candidates: list[str],
              col) -> Optional[float]:
    """Extract a value from a financial statement, trying multiple field names.

    Returns None if all candidates are missing or NaN.
    """
    for field_name in candidates:
        if field_name in df.index:
            val = df.loc[field_name, col]
            if pd.notna(val):
                return float(val)
    return None


class InsuranceFundamentalFactorModel:
    """Factor model for insurance company stock returns.

    Decomposes insurance stock returns into components driven by
    insurance-specific fundamentals: combined ratio, LAE ratio,
    premium growth, investment yield, reserve adequacy, leverage,
    and book value growth.

    Two regression modes:
      - cross_sectional (Fama-MacBeth): daily cross-sectional regressions
        yield factor return premiums. Answers: 'which fundamentals does
        the market reward?'
      - time_series (per-ticker): per-ticker regressions yield factor
        loadings. Answers: 'how much of this stock's movement is
        explained by its fundamentals?'
    """

    def __init__(self, config: InsuranceFactorConfig = None,
                 periods_per_year: int = 252):
        self.config = config or InsuranceFactorConfig()
        self.ppy = periods_per_year

        # Raw data
        self._quarterly_data: Optional[dict[str, pd.DataFrame]] = None
        self._daily_factors: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None

        # Regression results
        self._betas: Optional[pd.DataFrame] = None
        self._alphas: Optional[pd.Series] = None
        self._r_squared: Optional[pd.Series] = None
        self._residuals: Optional[pd.DataFrame] = None
        self._factor_returns: Optional[pd.DataFrame] = None
        self._factor_t_stats: Optional[pd.Series] = None

        # Universe
        self._tickers: Optional[list[str]] = None
        self._factor_names: Optional[list[str]] = None

    # ── Data Fetching ──────────────────────────────────────────

    def fetch_fundamentals(self, tickers: list[str] = None
                           ) -> dict[str, pd.DataFrame]:
        """Fetch quarterly fundamentals for insurance tickers.

        Returns dict mapping ticker -> DataFrame of quarterly factor values.
        """
        import yfinance as yf

        if tickers is None:
            tickers = list(self.config.default_tickers)

        results = {}
        for ticker in tickers:
            try:
                data = self._fetch_single_ticker(ticker, yf)
                if data is not None and len(data) >= self.config.min_quarters:
                    results[ticker] = data
                    logger.info("Fetched %d quarters for %s", len(data), ticker)
                else:
                    logger.warning("Insufficient data for %s, skipping", ticker)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", ticker, e)
            time.sleep(0.5)  # rate limit

        if len(results) < self.config.min_tickers:
            raise InsuranceDataError(
                f"Only {len(results)} tickers have sufficient data "
                f"(need {self.config.min_tickers})")

        self._quarterly_data = results
        self._tickers = list(results.keys())
        return results

    def _fetch_single_ticker(self, ticker: str, yf) -> Optional[pd.DataFrame]:
        """Fetch and compute factors for a single ticker."""
        t = yf.Ticker(ticker)
        income = t.quarterly_income_stmt
        balance = t.quarterly_balance_sheet

        if income is None or income.empty or balance is None or balance.empty:
            return None

        return self._compute_factors(income, balance)

    def _compute_factors(self, income: pd.DataFrame,
                         balance: pd.DataFrame) -> pd.DataFrame:
        """Compute insurance factor ratios from raw financial statements.

        yfinance returns index=line items, columns=quarter dates (descending).
        """
        quarters = sorted(income.columns)
        # Align balance sheet quarters to income quarters
        balance_quarters = sorted(balance.columns)

        rows = []
        for q in quarters:
            revenue = _safe_get(income, INCOME_FIELDS["total_revenue"], q)
            if revenue is None or revenue <= 0:
                continue

            claims = _safe_get(income, INCOME_FIELDS["claims"], q)
            lae = _safe_get(income, INCOME_FIELDS["lae"], q)
            sga = _safe_get(income, INCOME_FIELDS["sga"], q)
            total_expenses = _safe_get(income, INCOME_FIELDS["total_expenses"], q)
            net_interest = _safe_get(income, INCOME_FIELDS["net_interest_income"], q)
            other_income = _safe_get(income, INCOME_FIELDS["other_income"], q)
            ebit = _safe_get(income, INCOME_FIELDS["ebit"], q)
            net_income = _safe_get(income, INCOME_FIELDS["net_income"], q)

            # Find matching balance sheet quarter (closest date <= q)
            bq = None
            for bdate in balance_quarters:
                if bdate <= q:
                    bq = bdate
            if bq is None and balance_quarters:
                bq = balance_quarters[0]

            total_assets = None
            investments = None
            equity = None
            total_liabilities = None
            total_debt = 0.0

            if bq is not None:
                total_assets = _safe_get(balance, BALANCE_FIELDS["total_assets"], bq)
                investments = _safe_get(balance, BALANCE_FIELDS["investments"], bq)
                equity = _safe_get(balance, BALANCE_FIELDS["equity"], bq)
                total_liabilities = _safe_get(
                    balance, BALANCE_FIELDS["total_liabilities"], bq)
                total_debt = _safe_get(
                    balance, BALANCE_FIELDS["total_debt"], bq) or 0.0

            row = {"date": q}

            # Loss ratio
            if claims is not None:
                row["loss_ratio"] = abs(claims) / revenue

            # LAE ratio
            if lae is not None:
                row["lae_ratio"] = abs(lae) / revenue

            # Expense ratio (fallback: total_expenses - claims - lae)
            if sga is not None:
                row["expense_ratio"] = abs(sga) / revenue
            elif total_expenses is not None and claims is not None:
                implied = abs(total_expenses) - abs(claims) - abs(lae or 0)
                row["expense_ratio"] = max(implied, 0) / revenue

            # Combined ratio
            lr = row.get("loss_ratio", 0)
            lae_r = row.get("lae_ratio", 0)
            er = row.get("expense_ratio", 0)
            if lr > 0 or er > 0:
                row["combined_ratio"] = lr + lae_r + er

            # Investment yield
            if investments is not None and investments > 0:
                inv_income = (net_interest or 0) + (other_income or 0)
                row["investment_yield"] = inv_income / investments

            # Reserve ratio: (liabilities - debt) / premium
            if total_liabilities is not None:
                reserves_proxy = total_liabilities - total_debt
                row["reserve_ratio"] = reserves_proxy / revenue

            # Leverage
            if equity is not None and equity > 0 and total_liabilities is not None:
                row["leverage_ratio"] = total_liabilities / equity

            # ROE
            if equity is not None and equity > 0 and net_income is not None:
                row["roe"] = net_income / equity

            # Operating margin
            if ebit is not None:
                row["operating_margin"] = ebit / revenue

            # Investment to assets
            if (investments is not None and total_assets is not None
                    and total_assets > 0):
                row["investment_to_assets"] = investments / total_assets

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("date").sort_index()

        # Premium growth YoY (compare to 4 quarters ago)
        revenues = []
        for q in quarters:
            rev = _safe_get(income, INCOME_FIELDS["total_revenue"], q)
            if rev is not None:
                revenues.append((q, rev))
        if len(revenues) >= 5:
            rev_s = pd.Series(dict(revenues)).sort_index()
            yoy = rev_s.pct_change(4)
            df["premium_growth_yoy"] = yoy.reindex(df.index)

        # Book value growth QoQ
        equities = []
        for q in sorted(balance.columns):
            eq = _safe_get(balance, BALANCE_FIELDS["equity"], q)
            if eq is not None:
                equities.append((q, eq))
        if len(equities) >= 2:
            eq_s = pd.Series(dict(equities)).sort_index()
            qoq = eq_s.pct_change(1)
            df["book_value_growth"] = qoq.reindex(df.index)

        return df

    # ── Frequency Conversion ──────────────────────────────────

    def build_daily_factors(self, returns: pd.DataFrame,
                            quarterly_data: dict[str, pd.DataFrame] = None
                            ) -> pd.DataFrame:
        """Convert quarterly fundamentals to daily factor panel.

        Uses step-function forward-fill: Q1 values apply every day from
        Q1 report date until Q2 reports. No look-ahead bias.

        Returns DataFrame with MultiIndex (date, ticker), columns = factor names.
        """
        if quarterly_data is None:
            quarterly_data = self._quarterly_data
        if quarterly_data is None:
            raise InsuranceDataError(
                "No quarterly data. Call fetch_fundamentals() first.")

        daily_index = returns.index
        tickers = [t for t in returns.columns if t in quarterly_data]

        # Union of all factor names across tickers
        all_factors = set()
        for t in tickers:
            all_factors.update(quarterly_data[t].columns)
        factor_names = sorted(
            all_factors & set(self.config.factors_enabled))

        if not factor_names:
            raise InsuranceDataError("No enabled factors found in data")

        # Build panel: for each ticker, forward-fill quarterly into daily
        panels = []
        for ticker in tickers:
            qdata = quarterly_data[ticker]
            available_cols = [c for c in factor_names if c in qdata.columns]
            if not available_cols:
                continue
            daily = qdata[available_cols].reindex(daily_index, method="ffill")
            daily["ticker"] = ticker
            panels.append(daily)

        if not panels:
            raise InsuranceDataError("No tickers with valid daily factors")

        panel = pd.concat(panels)
        panel = panel.set_index("ticker", append=True)

        if self.config.cross_sectional_zscore:
            panel = self._zscore_cross_sectional(panel, factor_names)

        self._daily_factors = panel
        self._factor_names = factor_names
        self._tickers = tickers
        return panel

    def _zscore_cross_sectional(self, panel: pd.DataFrame,
                                 factor_names: list[str]) -> pd.DataFrame:
        """Z-score factor values across tickers for each date."""
        result = panel.copy()
        wp = self.config.winsorize_pct

        for date in panel.index.get_level_values(0).unique():
            mask = result.index.get_level_values(0) == date
            for col in factor_names:
                if col not in result.columns:
                    continue
                vals = result.loc[mask, col].dropna()
                if len(vals) < 3:
                    continue
                # Winsorize
                if wp > 0:
                    lo = vals.quantile(wp)
                    hi = vals.quantile(1 - wp)
                    vals = vals.clip(lo, hi)
                mu = vals.mean()
                std = vals.std()
                if std > 1e-10:
                    result.loc[mask, col] = (
                        result.loc[mask, col] - mu) / std

        return result

    # ── Regression ─────────────────────────────────────────────

    def fit(self, returns: pd.DataFrame = None,
            tickers: list[str] = None) -> "InsuranceFundamentalFactorModel":
        """Fit the insurance fundamental factor model.

        Orchestrates: fetch fundamentals -> fetch prices -> build daily
        factors -> regress.
        """
        if tickers is None:
            tickers = list(self.config.default_tickers)

        # Fetch fundamentals
        if self._quarterly_data is None:
            self.fetch_fundamentals(tickers)

        # Fetch returns if not provided
        if returns is None:
            returns = self._fetch_returns(self._tickers)

        self._returns = returns

        # Build daily factor panel
        daily_factors = self.build_daily_factors(returns)

        # Run regression
        if self.config.regression_mode == "cross_sectional":
            self._fit_cross_sectional(returns, daily_factors)
        else:
            self._fit_time_series(returns, daily_factors)

        return self

    def _fetch_returns(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch daily returns from yfinance."""
        import yfinance as yf

        prices = yf.download(
            tickers, period=self.config.fetch_period,
            interval="1d", threads=False)

        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices["Adj Close"]
        elif len(tickers) == 1:
            prices = prices[["Adj Close"]]
            prices.columns = tickers

        prices = prices.dropna(how="all").ffill().dropna()
        returns = prices.pct_change().dropna()
        return returns

    def _fit_cross_sectional(self, returns: pd.DataFrame,
                              factors: pd.DataFrame) -> None:
        """Fama-MacBeth: cross-sectional regression each day."""
        factor_cols = [c for c in self._factor_names
                       if c in factors.columns]
        dates = returns.index
        factor_dates = factors.index.get_level_values(0).unique()

        daily_betas = []
        daily_r2 = []
        valid_dates = []

        for date in dates:
            if date not in factor_dates:
                continue

            try:
                day_factors = factors.loc[date]
            except KeyError:
                continue

            day_returns = returns.loc[date]

            # Align tickers
            if isinstance(day_factors.index, pd.MultiIndex):
                factor_tickers = day_factors.index.get_level_values(0)
            else:
                factor_tickers = day_factors.index
            common = factor_tickers.intersection(day_returns.index)
            if len(common) < self.config.min_tickers:
                continue

            y = day_returns[common].values
            X = day_factors.loc[common, factor_cols].values

            # Drop NaN rows
            valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if valid.sum() < self.config.min_tickers:
                continue

            y = y[valid]
            X = X[valid]

            # OLS with intercept
            X_const = np.column_stack([np.ones(len(X)), X])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            betas = coeffs[1:]
            predicted = X_const @ coeffs
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum((y - predicted) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

            daily_betas.append(betas)
            daily_r2.append(r2)
            valid_dates.append(date)

        if not daily_betas:
            raise InsuranceDataError(
                "No valid cross-sections for regression")

        self._factor_returns = pd.DataFrame(
            daily_betas, columns=factor_cols, index=valid_dates)
        self._r_squared = pd.Series(daily_r2, index=valid_dates,
                                     name="r_squared")

        # Newey-West t-stats
        t_stats = {}
        for col in factor_cols:
            series = self._factor_returns[col].dropna().values
            if len(series) < 10:
                t_stats[col] = 0.0
                continue
            mean = np.mean(series)
            T = len(series)
            lag = max(1, int(4 * (T / 100) ** (2 / 9)))
            nw_se = self._newey_west_se(series, lag)
            t_stats[col] = mean / nw_se if nw_se > 1e-12 else 0.0

        self._factor_t_stats = pd.Series(t_stats)
        self._factor_names = factor_cols

    def _fit_time_series(self, returns: pd.DataFrame,
                          factors: pd.DataFrame) -> None:
        """Per-ticker time-series regression."""
        factor_cols = [c for c in self._factor_names
                       if c in factors.columns]
        betas_dict = {}
        alphas_dict = {}
        r2_dict = {}
        residuals_dict = {}

        for ticker in self._tickers:
            if ticker not in returns.columns:
                continue

            y_full = returns[ticker]

            # Get this ticker's daily factor values
            try:
                ticker_factors = factors.xs(ticker, level="ticker")
            except KeyError:
                continue

            available_cols = [c for c in factor_cols
                              if c in ticker_factors.columns]
            if not available_cols:
                continue

            # Align dates
            common = y_full.index.intersection(ticker_factors.index)
            if len(common) < self.config.min_observations:
                continue

            y = y_full.loc[common].values
            X = ticker_factors.loc[common, available_cols].values

            # Drop NaN
            valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if valid.sum() < self.config.min_observations:
                continue

            y = y[valid]
            X = X[valid]
            valid_dates = common[valid]

            # OLS
            X_const = np.column_stack([np.ones(len(X)), X])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            alpha = coeffs[0]
            beta = coeffs[1:]
            predicted = X_const @ coeffs
            resid = y - predicted
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum(resid ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

            # Pad beta to full factor_cols length (NaN for missing)
            full_beta = np.full(len(factor_cols), np.nan)
            for i, col in enumerate(available_cols):
                idx = factor_cols.index(col)
                full_beta[idx] = beta[i]

            betas_dict[ticker] = full_beta
            alphas_dict[ticker] = alpha * self.ppy
            r2_dict[ticker] = r2
            residuals_dict[ticker] = pd.Series(resid, index=valid_dates)

        if not betas_dict:
            raise InsuranceDataError(
                "No tickers with sufficient data for regression")

        self._betas = pd.DataFrame(betas_dict, index=factor_cols).T
        self._alphas = pd.Series(alphas_dict)
        self._r_squared = pd.Series(r2_dict)
        self._residuals = pd.DataFrame(residuals_dict)
        self._factor_names = factor_cols

    @staticmethod
    def _newey_west_se(x: np.ndarray, max_lag: int) -> float:
        """Newey-West standard error for the mean of x."""
        T = len(x)
        x_dm = x - x.mean()
        gamma_0 = np.sum(x_dm ** 2) / T

        nw_var = gamma_0
        for j in range(1, max_lag + 1):
            weight = 1 - j / (max_lag + 1)
            gamma_j = np.sum(x_dm[j:] * x_dm[:-j]) / T
            nw_var += 2 * weight * gamma_j

        return np.sqrt(max(nw_var, 0) / T)

    # ── Results ────────────────────────────────────────────────

    def risk_decomposition(self, weights: np.ndarray = None) -> dict:
        """Decompose portfolio variance into fundamental factor contributions.

        Works for both regression modes. In cross-sectional mode, uses
        the factor return covariance. In time-series mode, uses
        portfolio betas and factor covariance.
        """
        if self.config.regression_mode == "cross_sectional":
            return self._risk_decomp_cross_sectional()
        else:
            return self._risk_decomp_time_series(weights)

    def _risk_decomp_cross_sectional(self) -> dict:
        """Risk decomposition using Fama-MacBeth factor return series."""
        if self._factor_returns is None:
            raise InsuranceDataError("Call fit() first")

        fr = self._factor_returns.dropna()
        total_var = float(fr.sum(axis=1).var()) * self.ppy
        if total_var < 1e-12:
            total_var = 1e-12

        per_factor = {}
        for col in fr.columns:
            fvar = float(fr[col].var()) * self.ppy
            per_factor[col] = fvar / total_var if total_var > 0 else 0

        factor_var = sum(per_factor.values()) * total_var
        avg_r2 = float(self._r_squared.mean()) if self._r_squared is not None else 0

        return {
            "total_risk": total_var,
            "factor_risk": factor_var,
            "idiosyncratic_risk": total_var - factor_var,
            "factor_pct": avg_r2,
            "avg_daily_r_squared": avg_r2,
            "per_factor": per_factor,
        }

    def _risk_decomp_time_series(self, weights: np.ndarray = None) -> dict:
        """Risk decomposition from time-series betas."""
        if self._betas is None:
            raise InsuranceDataError("Call fit() first")

        n = len(self._tickers)
        if weights is None:
            weights = np.ones(n) / n

        betas = self._betas.fillna(0).values
        port_betas = weights @ betas

        # Factor covariance from daily factor changes
        if self._daily_factors is not None:
            factor_cols = self._factor_names
            # Average factor values across tickers per day
            df = self._daily_factors[factor_cols].groupby(level=0).mean()
            factor_changes = df.diff().dropna()
            factor_cov = factor_changes.cov().values * self.ppy
        else:
            factor_cov = np.eye(len(self._factor_names))

        factor_var = float(port_betas @ factor_cov @ port_betas)

        # Idiosyncratic
        if self._residuals is not None:
            idio_vars = np.array([
                self._residuals[t].var() if t in self._residuals.columns else 0
                for t in self._tickers
            ])
            idio_var = float(np.sum((weights ** 2) * idio_vars)) * self.ppy
        else:
            idio_var = 0

        total_var = factor_var + idio_var
        if total_var < 1e-12:
            total_var = 1e-12

        per_factor = {}
        for i, fname in enumerate(self._factor_names):
            contrib = port_betas[i] ** 2 * factor_cov[i, i]
            per_factor[fname] = float(contrib / total_var)

        return {
            "total_risk": total_var,
            "factor_risk": factor_var,
            "idiosyncratic_risk": idio_var,
            "factor_pct": factor_var / total_var,
            "per_factor": per_factor,
        }

    def factor_summary(self) -> pd.DataFrame:
        """Summary statistics for each factor.

        Returns DataFrame with mean return/beta, t-stat, annualized
        return, and information ratio.
        """
        if self.config.regression_mode == "cross_sectional":
            return self._summary_cross_sectional()
        else:
            return self._summary_time_series()

    def _summary_cross_sectional(self) -> pd.DataFrame:
        """Summary from Fama-MacBeth factor returns."""
        if self._factor_returns is None:
            raise InsuranceDataError("Call fit() first")

        fr = self._factor_returns
        rows = {}
        for col in fr.columns:
            s = fr[col].dropna()
            mean_ret = float(s.mean())
            std_ret = float(s.std()) if len(s) > 1 else 0
            ann_ret = mean_ret * self.ppy
            ann_std = std_ret * np.sqrt(self.ppy) if std_ret > 0 else 0
            t_stat = float(self._factor_t_stats.get(col, 0))
            ir = ann_ret / ann_std if ann_std > 1e-10 else 0

            rows[col] = {
                "mean_daily_return": mean_ret,
                "t_stat": t_stat,
                "annualized_return": ann_ret,
                "annualized_std": ann_std,
                "info_ratio": ir,
                "significant": abs(t_stat) >= 1.96,
            }

        return pd.DataFrame(rows).T

    def _summary_time_series(self) -> pd.DataFrame:
        """Summary from per-ticker betas."""
        if self._betas is None:
            raise InsuranceDataError("Call fit() first")

        rows = {}
        for col in self._betas.columns:
            betas = self._betas[col].dropna()
            mean_beta = float(betas.mean())
            std_beta = float(betas.std()) if len(betas) > 1 else 0
            n = len(betas)
            se = std_beta / np.sqrt(n) if n > 0 else 0
            t_stat = mean_beta / se if se > 1e-10 else 0

            rows[col] = {
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "t_stat": t_stat,
                "n_tickers": n,
                "significant": abs(t_stat) >= 1.96,
            }

        return pd.DataFrame(rows).T

    def factor_correlation(self) -> pd.DataFrame:
        """Correlation matrix between factors.

        Flags multicollinearity. Combined ratio = loss + lae + expense
        by construction, so these will be correlated.
        """
        if self.config.regression_mode == "cross_sectional":
            if self._factor_returns is None:
                raise InsuranceDataError("Call fit() first")
            return self._factor_returns.corr()
        else:
            if self._betas is None:
                raise InsuranceDataError("Call fit() first")
            return self._betas.corr()

    def ticker_exposures(self) -> Optional[pd.DataFrame]:
        """Current factor exposures for each ticker.

        Returns the most recent day's factor values for each ticker.
        """
        if self._daily_factors is None:
            return None

        last_date = self._daily_factors.index.get_level_values(0).max()
        try:
            return self._daily_factors.loc[last_date]
        except KeyError:
            return None

    # ── Properties ─────────────────────────────────────────────

    @property
    def betas(self) -> Optional[pd.DataFrame]:
        return self._betas

    @property
    def alphas(self) -> Optional[pd.Series]:
        return self._alphas

    @property
    def r_squared(self) -> Optional[pd.Series]:
        return self._r_squared

    @property
    def factor_returns(self) -> Optional[pd.DataFrame]:
        return self._factor_returns

    @property
    def factor_names(self) -> list[str]:
        return list(self._factor_names) if self._factor_names else []

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers) if self._tickers else []
