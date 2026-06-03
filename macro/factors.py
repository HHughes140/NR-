from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from NR.config import MacroConfig
from NR.exceptions import MacroDataError

import logging
logger = logging.getLogger(__name__)


class FactorModel:
    """Factor model for insurance-based trading.

    Two-layer decomposition:
      1. Market beta regression (Mkt-RF only) -- directional exposure
         of each underlying, which drives delta risk on short options.
      2. Vol-surface factor analysis -- the factors that actually move
         an option book: vol level, vol term structure, skew, correlation,
         and variance risk premium.

    The equity-style factors (SMB, HML, RMW, CMA) are dropped. An option
    seller doesn't care if the underlying is value or growth -- they care
    about the vol surface above it.
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
        # Vol-surface factor state
        self._vol_factors: Optional[pd.DataFrame] = None
        self._vol_diagnostics: Optional[dict] = None

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_factors(self, start_date: str = None,
                      end_date: str = None) -> pd.DataFrame:
        """Fetch Fama-French market factor (Mkt-RF + RF only).

        We only use Mkt-RF for beta regression. SMB/HML/RMW/CMA are
        fetched but stored separately in case risk_decomposition is
        called -- they are not used in the core fit.
        """
        data = None

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

        if data is None:
            data = self._fetch_factors_direct(start_date, end_date)

        if data is None:
            raise MacroDataError(
                "Failed to fetch Fama-French data via datareader and direct download")

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

            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                parts = lines[i].strip().split(',')
                if not parts[0].strip() or len(parts[0].strip()) != 8:
                    try:
                        int(parts[0].strip())
                    except ValueError:
                        end_idx = i
                        break

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

            max_cols = len(rows[0])
            cols = cols[:max_cols]

            df = pd.DataFrame(rows, index=dates, columns=cols)
            df = df / 100.0

            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            return df

        except Exception as e:
            logger.warning("Direct FF factor download failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Market beta regression (what matters for delta exposure)
    # ------------------------------------------------------------------

    def fit(self, asset_returns: pd.DataFrame,
            factor_data: pd.DataFrame = None) -> "FactorModel":
        """Regress each asset on Mkt-RF to get market beta and residuals.

        This is the only regression that matters for insurance trading:
        market beta tells you how much delta exposure each underlying
        contributes to your short option book.
        """
        if factor_data is not None:
            self._factor_data = factor_data
        if self._factor_data is None:
            self.fetch_factors()

        self._asset_names = list(asset_returns.columns)
        factors = self._factor_data.copy()

        common = asset_returns.index.intersection(factors.index)
        if len(common) < 30:
            raise MacroDataError(
                f"Only {len(common)} overlapping dates between assets and factors")

        asset_ret = asset_returns.loc[common]
        factors = factors.loc[common]

        rf = factors["RF"] if "RF" in factors.columns else 0.0
        # Only regress on Mkt-RF -- the one factor that matters for delta
        factor_cols = ["Mkt-RF"] if "Mkt-RF" in factors.columns else []
        if not factor_cols:
            raise MacroDataError("Mkt-RF column not found in factor data")

        X = factors[factor_cols].values
        betas_dict = {}
        alphas_dict = {}
        r2_dict = {}
        residuals_dict = {}

        for ticker in self._asset_names:
            excess_ret = asset_ret[ticker].values - (
                rf.values if isinstance(rf, pd.Series) else rf)
            X_with_const = np.column_stack([np.ones(len(X)), X])
            coeffs, _, _, _ = np.linalg.lstsq(X_with_const, excess_ret,
                                               rcond=None)
            alpha = coeffs[0]
            beta = coeffs[1:]
            predicted = X_with_const @ coeffs
            resid = excess_ret - predicted
            ss_tot = np.sum((excess_ret - excess_ret.mean()) ** 2)
            ss_res = np.sum(resid ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            betas_dict[ticker] = beta
            alphas_dict[ticker] = alpha * self.ppy
            r2_dict[ticker] = r2
            residuals_dict[ticker] = resid

        self._betas = pd.DataFrame(betas_dict, index=factor_cols).T
        self._alphas = pd.Series(alphas_dict)
        self._r_squared = pd.Series(r2_dict)
        self._residuals = pd.DataFrame(residuals_dict, index=common)

        return self

    # ------------------------------------------------------------------
    # Vol-surface factor construction
    # ------------------------------------------------------------------

    def compute_vol_factors(self, vix: pd.Series,
                            vix9d: pd.Series = None,
                            vix3m: pd.Series = None,
                            vvix: pd.Series = None,
                            skew: pd.Series = None,
                            spx_returns: pd.Series = None) -> pd.DataFrame:
        """Construct the vol-surface factors that drive option P&L.

        Returns a DataFrame with the following factors:
            vol_level       : VIX level (z-scored). High = expensive premium,
                              but also higher claim risk.
            vol_term_slope  : (VIX3M - VIX) / VIX. Positive = contango
                              (normal, safe to sell). Negative = backwardation
                              (fear, dangerous).
            vol_near_spread : (VIX - VIX9D) / VIX. Near-term vs 30-day.
                              Positive = near-term fear fading. Negative =
                              near-term panic intensifying.
            vol_of_vol      : VVIX level (z-scored). High = unstable vol
                              regime, options can reprice violently.
            skew_factor     : CBOE SKEW (z-scored). High = market paying up
                              for tail protection, puts are expensive.
            vrp             : Variance risk premium (IV - RV). Positive =
                              premium available to collect. This is the
                              insurance margin.
            realized_vol    : 21-day realized vol of SPX. The "actual claims"
                              vs the "premium charged" (implied vol).
        """
        rv_window = self.config.vrp_realized_vol_window
        result = pd.DataFrame(index=vix.index)

        # Vol level (z-scored over trailing year)
        vix_mean = vix.rolling(252, min_periods=63).mean()
        vix_std = vix.rolling(252, min_periods=63).std()
        result["vol_level"] = (vix - vix_mean) / (vix_std + 1e-8)

        # Term structure slope: contango (+) vs backwardation (-)
        if vix3m is not None:
            common = vix.index.intersection(vix3m.index)
            vix_a = vix.reindex(common)
            vix3m_a = vix3m.reindex(common)
            slope = (vix3m_a - vix_a) / (vix_a + 1e-8)
            result["vol_term_slope"] = slope.reindex(vix.index)
        else:
            result["vol_term_slope"] = np.nan

        # Near-term spread: 30d vs 9d
        if vix9d is not None:
            common = vix.index.intersection(vix9d.index)
            vix_a = vix.reindex(common)
            vix9d_a = vix9d.reindex(common)
            near = (vix_a - vix9d_a) / (vix_a + 1e-8)
            result["vol_near_spread"] = near.reindex(vix.index)
        else:
            result["vol_near_spread"] = np.nan

        # Vol of vol (z-scored)
        if vvix is not None:
            vvix_mean = vvix.rolling(252, min_periods=63).mean()
            vvix_std = vvix.rolling(252, min_periods=63).std()
            result["vol_of_vol"] = ((vvix - vvix_mean) / (vvix_std + 1e-8)
                                    ).reindex(vix.index)
        else:
            result["vol_of_vol"] = np.nan

        # Skew (z-scored)
        if skew is not None:
            skew_mean = skew.rolling(252, min_periods=63).mean()
            skew_std = skew.rolling(252, min_periods=63).std()
            result["skew_factor"] = ((skew - skew_mean) / (skew_std + 1e-8)
                                     ).reindex(vix.index)
        else:
            result["skew_factor"] = np.nan

        # Realized vol
        if spx_returns is not None:
            rv = spx_returns.rolling(rv_window, min_periods=10).std() * np.sqrt(252) * 100
            result["realized_vol"] = rv.reindex(vix.index)
            # Variance risk premium: IV - RV (in vol points)
            result["vrp"] = (vix - result["realized_vol"]).reindex(vix.index)
        else:
            result["realized_vol"] = np.nan
            result["vrp"] = np.nan

        self._vol_factors = result
        return result

    def vol_regime_diagnostics(self, vol_factors: pd.DataFrame = None) -> dict:
        """Diagnose current vol regime for premium-selling decisions.

        Returns a dict with actionable signals:
            safe_to_sell    : bool -- all conditions favor selling premium
            vrp_sufficient  : bool -- variance risk premium > min threshold
            vrp_rich        : bool -- VRP is very rich, can size up
            term_contango   : bool -- futures in contango (normal)
            vvix_stable     : bool -- vol of vol is not elevated
            skew_normal     : bool -- tail risk demand is not extreme
            regime_signal   : str  -- 'green' / 'yellow' / 'red'
            deployment_pct  : float -- suggested deployment (0.0 to 1.0)
        """
        if vol_factors is None:
            vol_factors = self._vol_factors
        if vol_factors is None or vol_factors.empty:
            return {"safe_to_sell": False, "regime_signal": "red",
                    "deployment_pct": 0.0, "reason": "no vol factor data"}

        latest = vol_factors.iloc[-1]
        cfg = self.config
        diag = {}

        # VRP check
        vrp = latest.get("vrp", np.nan)
        diag["vrp"] = float(vrp) if not np.isnan(vrp) else None
        diag["vrp_sufficient"] = (not np.isnan(vrp) and
                                  vrp >= cfg.vrp_min_threshold)
        diag["vrp_rich"] = (not np.isnan(vrp) and
                            vrp >= cfg.vrp_rich_threshold)

        # Term structure check
        slope = latest.get("vol_term_slope", np.nan)
        diag["vol_term_slope"] = float(slope) if not np.isnan(slope) else None
        diag["term_contango"] = (not np.isnan(slope) and
                                 slope >= cfg.term_structure_contango_threshold)
        diag["term_backwardation_warning"] = (
            not np.isnan(slope) and
            slope <= cfg.term_structure_backwardation_warning)
        diag["term_backwardation_crisis"] = (
            not np.isnan(slope) and
            slope <= cfg.term_structure_backwardation_crisis)

        # VVIX check
        vol_of_vol_z = latest.get("vol_of_vol", np.nan)
        diag["vol_of_vol_z"] = float(vol_of_vol_z) if not np.isnan(vol_of_vol_z) else None
        # Use raw VVIX if available, otherwise use z-score heuristic
        diag["vvix_stable"] = (np.isnan(vol_of_vol_z) or vol_of_vol_z < 1.5)
        diag["vvix_elevated"] = (not np.isnan(vol_of_vol_z) and vol_of_vol_z >= 1.5)

        # Skew check
        skew_z = latest.get("skew_factor", np.nan)
        diag["skew_z"] = float(skew_z) if not np.isnan(skew_z) else None
        diag["skew_normal"] = (np.isnan(skew_z) or skew_z < 1.5)
        diag["skew_extreme"] = (not np.isnan(skew_z) and skew_z >= 2.0)

        # Aggregate regime signal
        red_flags = 0
        yellow_flags = 0

        if diag.get("term_backwardation_crisis"):
            red_flags += 1
        elif diag.get("term_backwardation_warning"):
            yellow_flags += 1

        if not diag["vrp_sufficient"]:
            yellow_flags += 1
        if diag.get("vvix_elevated"):
            yellow_flags += 1
        if diag.get("skew_extreme"):
            yellow_flags += 1

        if red_flags > 0:
            diag["regime_signal"] = "red"
            diag["deployment_pct"] = 0.0
        elif yellow_flags >= 2:
            diag["regime_signal"] = "yellow"
            diag["deployment_pct"] = 0.50
        elif yellow_flags == 1:
            diag["regime_signal"] = "yellow"
            diag["deployment_pct"] = 0.75
        else:
            diag["regime_signal"] = "green"
            diag["deployment_pct"] = 1.0

        # Boost if VRP is rich and everything else is green
        if diag["vrp_rich"] and diag["regime_signal"] == "green":
            diag["deployment_pct"] = 1.0

        diag["safe_to_sell"] = diag["regime_signal"] == "green"

        self._vol_diagnostics = diag
        return diag

    # ------------------------------------------------------------------
    # Insurance-relevant analytics
    # ------------------------------------------------------------------

    def correlation_concentration(self, asset_returns: pd.DataFrame,
                                  window: int = 63) -> pd.DataFrame:
        """Track rolling average pairwise correlation.

        High average correlation = your "policies" are not independent.
        In insurance terms: all your claims trigger at once. This is the
        correlation spike risk that destroys short-option portfolios.

        Returns a DataFrame with:
            avg_correlation : rolling average pairwise correlation
            max_correlation : rolling max pairwise correlation
            pct_above_70    : fraction of pairs with corr > 0.70
        """
        n = asset_returns.shape[1]
        if n < 2:
            return pd.DataFrame(index=asset_returns.index)

        result_data = []
        for end in range(window, len(asset_returns) + 1):
            chunk = asset_returns.iloc[end - window:end]
            corr = chunk.corr().values
            # Upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            pairs = corr[mask]
            if len(pairs) == 0:
                continue
            result_data.append({
                "date": asset_returns.index[end - 1],
                "avg_correlation": float(np.mean(pairs)),
                "max_correlation": float(np.max(pairs)),
                "pct_above_70": float(np.mean(pairs > 0.70)),
            })

        if not result_data:
            return pd.DataFrame(index=asset_returns.index)

        return pd.DataFrame(result_data).set_index("date")

    def idiosyncratic_risk_ratio(self) -> Optional[pd.Series]:
        """Fraction of each asset's variance that is idiosyncratic.

        High ratio = the asset's risk is diversifiable = good for
        insurance (independent claims). Low ratio = the asset moves
        with the market = correlated claims, harder to diversify.
        """
        if self._r_squared is None:
            return None
        return 1.0 - self._r_squared

    def expected_returns(self, factor_premiums: np.ndarray = None) -> np.ndarray:
        """Expected returns from market beta model."""
        if self._betas is None:
            raise MacroDataError("Call fit() first")

        if factor_premiums is None:
            factor_cols = self._betas.columns
            factor_premiums = self._factor_data[factor_cols].mean().values

        daily_alpha = self._alphas.values / self.ppy
        return daily_alpha + self._betas.values @ factor_premiums

    def risk_decomposition(self, weights: np.ndarray) -> dict:
        """Decompose portfolio risk into market and idiosyncratic components."""
        if self._betas is None:
            raise MacroDataError("Call fit() first")

        factor_cols = self._betas.columns
        factor_cov = np.array(self._factor_data[factor_cols].cov())
        port_betas = weights @ self._betas.values

        factor_var = port_betas @ factor_cov @ port_betas
        idio_vars = np.array([self._residuals[t].var() for t in self._asset_names])
        idio_var = np.sum((weights ** 2) * idio_vars)
        total_var = factor_var + idio_var

        contributions = {}
        for i, factor in enumerate(factor_cols):
            contributions[factor] = float(
                port_betas[i] ** 2 * factor_cov[i, i])

        return {
            "market_risk": float(factor_var),
            "idiosyncratic_risk": float(idio_var),
            "total_risk": float(total_var),
            "market_pct": float(factor_var / total_var) if total_var > 0 else 0,
            "factor_contributions": contributions,
        }

    def factor_attribution(self, portfolio_returns: pd.Series,
                           weights: np.ndarray) -> pd.DataFrame:
        """Period-by-period attribution of portfolio return to market vs alpha."""
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    @property
    def vol_factors(self) -> Optional[pd.DataFrame]:
        return self._vol_factors

    @property
    def vol_diagnostics(self) -> Optional[dict]:
        return self._vol_diagnostics
