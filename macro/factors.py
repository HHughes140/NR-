from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from NR.config import MacroConfig
from NR.exceptions import MacroDataError


class FactorModel:
    """Fama-French factor model for return decomposition and estimation.

    Supports 3-factor (MKT, SMB, HML) and 5-factor (+RMW, +CMA).
    Uses pandas-datareader (lazy import) for factor data.
    """

    def __init__(self, config: MacroConfig = None, periods_per_year: int = 252):
        self.config = config or MacroConfig()
        self.ppy = periods_per_year
        self._factor_data: Optional[pd.DataFrame] = None
        self._betas: Optional[pd.DataFrame] = None
        self._alphas: Optional[pd.Series] = None
        self._residuals: Optional[pd.DataFrame] = None
        self._r_squared: Optional[pd.Series] = None
        self._asset_names: Optional[list] = None

    def fetch_factors(self, start_date: str = None,
                      end_date: str = None) -> pd.DataFrame:
        """Fetch Fama-French factor returns.

        Tries pandas-datareader first; falls back to direct CSV download
        from Ken French's data library if datareader fails.
        """
        data = None

        # Attempt 1: pandas-datareader
        try:
            import pandas_datareader.data as web
            dataset = ("F-F_Research_Data_5_Factors_2x3_daily"
                       if self.config.ff_factors >= 5
                       else "F-F_Research_Data_Factors_daily")
            ff = web.DataReader(dataset, "famafrench",
                                start=start_date, end=end_date)
            data = ff[0] / 100.0
        except Exception:
            pass

        # Attempt 2: Direct CSV download from Ken French's site
        if data is None:
            data = self._fetch_factors_direct(start_date, end_date)

        if data is None:
            raise MacroDataError(
                "Failed to fetch Fama-French data via datareader and direct download")

        # FF5+MOM (6 factors): merge momentum factor
        if self.config.ff_factors == 6:
            try:
                mom = self._fetch_momentum_direct(start_date, end_date)
                if mom is not None and len(mom) > 0:
                    mom_col = mom.columns[0]
                    data["Mom"] = mom[mom_col].reindex(data.index).fillna(0.0)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Momentum factor fetch failed (%s), using FF5 only", e)

        self._factor_data = data
        return data

    def _fetch_factors_direct(self, start_date=None, end_date=None):
        """Download FF factors directly from Ken French's data library."""
        import io
        import zipfile
        import urllib.request

        base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        if self.config.ff_factors >= 5:
            filename = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        else:
            filename = "F-F_Research_Data_Factors_daily_CSV.zip"

        try:
            url = base_url + filename
            with urllib.request.urlopen(url, timeout=30) as resp:
                zip_data = resp.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.CSV')
                            or n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    raw = f.read().decode('utf-8')

            # Parse: skip header lines, find the daily data table
            lines = raw.strip().split('\n')
            start_idx = None
            for i, line in enumerate(lines):
                parts = line.strip().split(',')
                if len(parts) >= 4 and len(parts[0].strip()) == 8:
                    try:
                        int(parts[0].strip())
                        start_idx = i
                        break
                    except ValueError:
                        continue

            if start_idx is None:
                return None

            # Find end of daily data (blank line or non-numeric first col)
            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                parts = lines[i].strip().split(',')
                if not parts[0].strip() or len(parts[0].strip()) != 8:
                    try:
                        int(parts[0].strip())
                    except ValueError:
                        end_idx = i
                        break

            # Build DataFrame
            data_lines = lines[start_idx:end_idx]
            if not data_lines:
                return None

            rows = []
            dates = []
            for line in data_lines:
                parts = [p.strip() for p in line.split(',')]
                try:
                    date = pd.Timestamp(parts[0][:4] + '-' +
                                        parts[0][4:6] + '-' +
                                        parts[0][6:8])
                    vals = [float(v) for v in parts[1:] if v]
                    dates.append(date)
                    rows.append(vals)
                except (ValueError, IndexError):
                    continue

            if not rows:
                return None

            if self.config.ff_factors >= 5:
                cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
            else:
                cols = ["Mkt-RF", "SMB", "HML", "RF"]

            # Truncate cols to match actual data width
            max_cols = len(rows[0])
            cols = cols[:max_cols]

            df = pd.DataFrame(rows, index=dates, columns=cols)
            df = df / 100.0  # percentage to decimal

            # Filter date range
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            return df

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Direct FF factor download failed: %s", e)
            return None

    def _fetch_momentum_direct(self, start_date=None, end_date=None):
        """Download momentum factor directly from Ken French's data library."""
        import io
        import zipfile
        import urllib.request

        url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "F-F_Momentum_Factor_daily_CSV.zip")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                zip_data = resp.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.CSV')
                            or n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    raw = f.read().decode('utf-8')

            lines = raw.strip().split('\n')
            rows = []
            dates = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and len(parts[0]) == 8:
                    try:
                        date = pd.Timestamp(parts[0][:4] + '-' +
                                            parts[0][4:6] + '-' +
                                            parts[0][6:8])
                        val = float(parts[1])
                        dates.append(date)
                        rows.append([val])
                    except (ValueError, IndexError):
                        continue

            if not rows:
                return None

            df = pd.DataFrame(rows, index=dates, columns=["Mom"])
            df = df / 100.0

            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            return df
        except Exception:
            return None

    def fit(self, asset_returns: pd.DataFrame,
            factor_data: pd.DataFrame = None) -> "FactorModel":
        """Regress each asset's excess returns on factor returns."""
        if factor_data is not None:
            self._factor_data = factor_data
        if self._factor_data is None:
            self.fetch_factors()

        self._asset_names = list(asset_returns.columns)
        factors = self._factor_data.copy()

        # Align dates
        common = asset_returns.index.intersection(factors.index)
        if len(common) < 30:
            raise MacroDataError(
                f"Only {len(common)} overlapping dates between assets and factors")

        asset_ret = asset_returns.loc[common]
        factors = factors.loc[common]

        # Get risk-free rate and factor columns
        rf = factors["RF"] if "RF" in factors.columns else 0.0
        factor_cols = [c for c in factors.columns if c != "RF"]
        if self.config.ff_factors == 3:
            factor_cols = [c for c in factor_cols if c in
                           ["Mkt-RF", "SMB", "HML"]]
        elif self.config.ff_factors == 5:
            factor_cols = [c for c in factor_cols if c in
                           ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]

        X = factors[factor_cols].values
        betas_dict = {}
        alphas_dict = {}
        r2_dict = {}
        residuals_dict = {}

        for ticker in self._asset_names:
            excess_ret = asset_ret[ticker].values - (rf.values if isinstance(rf, pd.Series) else rf)
            # OLS: y = alpha + X @ beta + epsilon
            X_with_const = np.column_stack([np.ones(len(X)), X])
            coeffs, residual_ss, _, _ = np.linalg.lstsq(X_with_const, excess_ret,
                                                          rcond=None)
            alpha = coeffs[0]
            beta = coeffs[1:]
            predicted = X_with_const @ coeffs
            resid = excess_ret - predicted
            ss_tot = np.sum((excess_ret - excess_ret.mean()) ** 2)
            ss_res = np.sum(resid ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            betas_dict[ticker] = beta
            alphas_dict[ticker] = alpha * self.ppy  # annualize
            r2_dict[ticker] = r2
            residuals_dict[ticker] = resid

        self._betas = pd.DataFrame(betas_dict, index=factor_cols).T
        self._alphas = pd.Series(alphas_dict)
        self._r_squared = pd.Series(r2_dict)
        self._residuals = pd.DataFrame(residuals_dict, index=common)

        return self

    def expected_returns(self, factor_premiums: np.ndarray = None) -> np.ndarray:
        """Expected returns from factor model."""
        if self._betas is None:
            raise MacroDataError("Call fit() first")

        if factor_premiums is None:
            # Use historical factor means (annualized)
            factor_cols = self._betas.columns
            factor_premiums = self._factor_data[factor_cols].mean().values

        daily_alpha = self._alphas.values / self.ppy
        return daily_alpha + self._betas.values @ factor_premiums

    def risk_decomposition(self, weights: np.ndarray) -> dict:
        """Decompose portfolio risk into factor and idiosyncratic components."""
        if self._betas is None:
            raise MacroDataError("Call fit() first")

        factor_cols = self._betas.columns
        factor_cov = np.array(self._factor_data[factor_cols].cov())
        port_betas = weights @ self._betas.values  # factor exposures

        factor_var = port_betas @ factor_cov @ port_betas
        idio_vars = np.array([self._residuals[t].var() for t in self._asset_names])
        idio_var = np.sum((weights ** 2) * idio_vars)
        total_var = factor_var + idio_var

        # Per-factor contribution
        contributions = {}
        for i, factor in enumerate(factor_cols):
            contributions[factor] = float(
                port_betas[i] ** 2 * factor_cov[i, i])

        return {
            "factor_risk": float(factor_var),
            "idiosyncratic_risk": float(idio_var),
            "total_risk": float(total_var),
            "factor_pct": float(factor_var / total_var) if total_var > 0 else 0,
            "factor_contributions": contributions,
        }

    def factor_attribution(self, portfolio_returns: pd.Series,
                           weights: np.ndarray) -> pd.DataFrame:
        """Period-by-period attribution of portfolio return to each factor."""
        if self._betas is None:
            raise MacroDataError("Call fit() first")

        factor_cols = list(self._betas.columns)
        port_betas = weights @ self._betas.values
        common = portfolio_returns.index.intersection(self._factor_data.index)

        factors = self._factor_data.loc[common, factor_cols]
        port_ret = portfolio_returns.loc[common]
        daily_alpha = (weights @ self._alphas.values) / self.ppy

        attribution = pd.DataFrame(index=common)
        attribution["total"] = port_ret
        attribution["alpha"] = daily_alpha
        for i, factor in enumerate(factor_cols):
            attribution[factor] = port_betas[i] * factors[factor]
        attribution["residual"] = (port_ret - attribution[factor_cols].sum(axis=1)
                                    - daily_alpha)
        return attribution

    @property
    def factor_names(self) -> list:
        return list(self._betas.columns) if self._betas is not None else []

    @property
    def betas(self) -> Optional[pd.DataFrame]:
        return self._betas

    @property
    def alphas(self) -> Optional[pd.Series]:
        return self._alphas

    @property
    def r_squared(self) -> Optional[pd.Series]:
        return self._r_squared
