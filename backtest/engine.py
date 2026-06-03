import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from NR.config import (
    BacktestConfig, DataConfig, OptimizationConfig, MacroConfig, SentimentConfig,
    BayesianConfig,
)
from NR.backtest.transaction_costs import TransactionCostModel
from NR.linalg.covariance import CovarianceEstimator
from NR.optimization.weights import PortfolioWeights
from NR.metrics.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime-adaptive D-hat phase weights
# ---------------------------------------------------------------------------
# In momentum regimes (kappa > 1.2), dislocation (follow the flow) dominates.
# In mean-reversion regimes (kappa < 0.8), convergence (fade the flow) dominates.
_PHASE_WEIGHTS_MOMENTUM = {
    "dislocation": 3.5, "discovery": 2.0, "convergence": -0.5, "normal": 0.6}
_PHASE_WEIGHTS_NEUTRAL = {
    "dislocation": 3.0, "discovery": 1.5, "convergence": -1.0, "normal": 0.5}
_PHASE_WEIGHTS_MEAN_REV = {
    "dislocation": 1.0, "discovery": 0.5, "convergence": -3.0, "normal": -0.5}


def _regime_phase_weights(
    kappa: float,
    momentum_threshold: float = 1.2,
    mean_rev_threshold: float = 0.8,
) -> dict:
    """Return D-hat phase weights adapted to the current reflexivity regime."""
    if kappa > momentum_threshold:
        return _PHASE_WEIGHTS_MOMENTUM
    elif kappa < mean_rev_threshold:
        return _PHASE_WEIGHTS_MEAN_REV
    else:
        return _PHASE_WEIGHTS_NEUTRAL


def _regime_blend_weights(kappa: float, dynamics_config) -> tuple:
    """Return (dhat_weight, reversal_weight) for regime-blended alpha."""
    mt = getattr(dynamics_config, 'momentum_threshold', 1.2)
    mrt = getattr(dynamics_config, 'mean_reversion_threshold', 0.8)
    if kappa > mt:
        w = dynamics_config.regime_blend_momentum_dhat_weight
    elif kappa < mrt:
        w = dynamics_config.regime_blend_meanrev_dhat_weight
    else:
        w = dynamics_config.regime_blend_neutral_dhat_weight
    return w, 1.0 - w


def _compute_regime_gate(kappa: float, vix_zscore: float, L: float,
                         dynamics_config) -> float:
    """Return raw deployment fraction based on regime state.

    Three states:
      1.0  — full deployment: momentum regime, low VIX, low contraction
      min  — reduced deployment: mean-reversion with no signal edge
      0.5  — ambiguous: elevated VIX or unclear regime
    """
    cash_kappa = getattr(dynamics_config, 'regime_gate_cash_kappa', 0.8)
    vix_thresh = getattr(dynamics_config, 'regime_gate_vix_threshold', 1.5)
    ambig_deploy = getattr(dynamics_config, 'regime_gate_ambiguous_deploy', 0.5)
    min_deploy = getattr(dynamics_config, 'regime_gate_min_deploy', 0.2)
    mt = getattr(dynamics_config, 'momentum_threshold', 1.2)

    if kappa > mt and vix_zscore < vix_thresh and L < 0.5:
        return 1.0  # full deployment
    elif kappa < cash_kappa and vix_zscore < vix_thresh:
        return min_deploy  # reduced — mean-reversion with no signal edge
    else:
        return ambig_deploy  # ambiguous — half deployment


class WalkForwardBacktester:
    """Rolling-window walk-forward backtester with market dynamics.

    At each rebalance date (scheduled or CUSUM-triggered):
    1. Estimate covariance from lookback window
    2. Optionally adjust for regime and sentiment
    3. Optionally apply market dynamics (Granger, CUSUM, volume dislocation)
    4. Compute optimal weights
    5. Apply transaction costs
    6. Track portfolio value forward

    Records equity curve, weights over time, costs, and dynamics events.
    """

    def __init__(self, config: BacktestConfig = None,
                 opt_config: OptimizationConfig = None,
                 data_config: DataConfig = None):
        self.config = config or BacktestConfig()
        self.opt_config = opt_config or OptimizationConfig()
        self.data_config = data_config or DataConfig()
        self.cost_model = TransactionCostModel(self.config)
        self._results: Optional[dict] = None
        self._last_causal_M = None  # Persisted from _apply_dynamics_v2

    @property
    def bars_per_year(self) -> int:
        """Annualization constant from data_config."""
        return self.data_config.bars_per_year

    @staticmethod
    def _align_ff_factors(
        ff_raw: pd.DataFrame,
        factor_cols: list,
        target_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Align daily FF factors to target index (may be intraday).

        For daily data, timestamps match directly.  For intraday data,
        daily values are forward-filled to every bar within that day.
        Handles timezone mismatches between FF data (naive) and
        returns index (may be UTC-aware).
        """
        ff = ff_raw[factor_cols].copy()
        ff.index = pd.to_datetime(ff.index)

        # Timezone alignment
        tgt_tz = getattr(target_index, 'tz', None)
        src_tz = getattr(ff.index, 'tz', None)
        if tgt_tz is not None and src_tz is None:
            ff.index = ff.index.tz_localize(tgt_tz)
        elif tgt_tz is None and src_tz is not None:
            ff.index = ff.index.tz_localize(None)

        # Combine source + target indices, sort, forward-fill, then select
        # target rows.  This is more robust than reindex(method='ffill')
        # when source dates don't appear in target (intraday case).
        combined = ff.reindex(ff.index.union(target_index)).ffill()
        return combined.reindex(target_index).fillna(0.0)

    def run(self, prices: pd.DataFrame,
            volume: pd.DataFrame = None,
            regime_detector=None,
            vix_series: pd.Series = None,
            yield_spread_series: pd.Series = None,
            sentiment_engine=None,
            bayesian_updater=None,
            dynamics_config=None,
            weight_method: str = "max_sharpe") -> dict:
        """Run walk-forward backtest.

        Args:
            prices: Daily close prices for all assets.
            volume: Daily volume for all assets (required for dynamics).
            regime_detector: Optional RegimeDetector instance.
            vix_series: VIX series for regime detection.
            yield_spread_series: Yield spread for regime detection.
            sentiment_engine: Optional CompositeSentiment instance.
            bayesian_updater: Optional BayesianReturnUpdater instance.
            dynamics_config: Optional DynamicsConfig for market dynamics.
            weight_method: 'max_sharpe', 'min_variance', 'risk_parity', 'equal'.

        Returns:
            Dict with equity_curve, weights_history, costs, decisions,
            metrics, and dynamics_log.
        """
        returns = prices.pct_change().dropna()
        tickers = list(prices.columns)
        n_assets = len(tickers)
        lookback = self.config.lookback_window
        rebal_freq = self.config.rebalance_frequency
        capital = self.config.initial_capital

        dates = returns.index
        if len(dates) < lookback + rebal_freq:
            raise ValueError(
                f"Need at least {lookback + rebal_freq} return observations, "
                f"got {len(dates)}")

        # Initialize dynamics modules
        use_dynamics = (dynamics_config is not None and
                        dynamics_config.enabled and
                        volume is not None)

        factor_tracker = None
        transition_detector = None
        signal_gen = None
        causal_graph = None
        vol_returns = None
        rolling_mean_vol = None
        ff_daily_factors_simple = None
        dynamics_log = []
        extra_rebalance_count = 0

        if use_dynamics:
            from NR.dynamics.factor_tracker import FactorTracker
            from NR.dynamics.transition_detector import TransitionDetector
            from NR.dynamics.signal_generator import SignalGenerator
            from NR.dynamics.causal_graph import GrangerCausalGraph

            factor_tracker = FactorTracker(
                n_factors=dynamics_config.n_factors,
                variance_threshold=dynamics_config.pca_variance_threshold,
                use_kalman_dfm=getattr(dynamics_config, 'use_kalman_dfm', False),
                kalman_process_noise=getattr(dynamics_config, 'kalman_dfm_process_noise', 1e-4),
                kalman_obs_noise=getattr(dynamics_config, 'kalman_dfm_obs_noise', 1e-3),
                loadings_source="graph_vae" if getattr(dynamics_config, 'use_graph_vae', False) else "pca",
                use_rbpf=getattr(dynamics_config, 'use_rbpf', False),
                rbpf_n_particles=getattr(dynamics_config, 'rbpf_n_particles', 50),
                rbpf_resample_threshold=getattr(dynamics_config, 'rbpf_resample_threshold', 0.5),
                rbpf_process_noise_scale=getattr(dynamics_config, 'rbpf_process_noise_scale', 0.01),
                graph_vae_use_directed=getattr(dynamics_config, 'graph_vae_use_directed', True),
                ff_eigenvalue_test=getattr(dynamics_config, 'ff_eigenvalue_test', False),
                ff_eigenvalue_significance=getattr(dynamics_config, 'ff_eigenvalue_significance', 0.05),
                ff_k_min=getattr(dynamics_config, 'ff_k_min', 1),
                ff_k_max=getattr(dynamics_config, 'ff_k_max', 0))
            transition_detector = TransitionDetector(
                n_factors=dynamics_config.n_factors,
                drift=dynamics_config.cusum_drift,
                threshold=dynamics_config.cusum_threshold,
                dislocation_decay_days=dynamics_config.dislocation_decay_days,
                convergence_trigger_days=dynamics_config.convergence_trigger_days)
            signal_gen = SignalGenerator(
                dislocation_return_scale=dynamics_config.dislocation_return_scale,
                granger_alpha=dynamics_config.granger_alpha)
            causal_graph = GrangerCausalGraph(
                lags=dynamics_config.granger_lags,
                p_threshold=dynamics_config.granger_p_threshold,
                min_edges=dynamics_config.granger_min_edges,
                max_assets=getattr(dynamics_config, 'max_granger_assets', 0),
                use_transfer_entropy=getattr(dynamics_config, 'use_transfer_entropy', False),
                te_bins=getattr(dynamics_config, 'te_bins', 6),
                te_k=getattr(dynamics_config, 'te_k', 1),
                te_l=getattr(dynamics_config, 'te_l', 1),
                te_threshold=getattr(dynamics_config, 'te_threshold', 0.01),
                te_method=getattr(dynamics_config, 'te_method', 'binned'),
                te_knn_k=getattr(dynamics_config, 'te_knn_k', 5),
                te_bootstrap_n=getattr(dynamics_config, 'te_bootstrap_n', 0),
                te_bootstrap_alpha=getattr(dynamics_config, 'te_bootstrap_alpha', 0.05))

            # POFM: Fetch Fama-French factor data for FF5 residualization
            ff_daily_factors_simple = None
            if getattr(dynamics_config, 'ff_residualize_pca', False):
                try:
                    from NR.macro.factors import FactorModel
                    from NR.config import MacroConfig
                    ff_n = getattr(dynamics_config, 'ff_factors', 5)
                    ff_config = MacroConfig(ff_factors=ff_n)
                    ff_model = FactorModel(ff_config)
                    ff_raw = ff_model.fetch_factors(
                        start_date=str(returns.index[0].date()),
                        end_date=str(returns.index[-1].date()))
                    factor_cols = [c for c in ff_raw.columns if c != "RF"]
                    ff_daily_factors_simple = self._align_ff_factors(
                        ff_raw, factor_cols, returns.index)
                    logger.info("POFM: loaded %d FF factors, %d dates",
                                len(factor_cols), len(ff_daily_factors_simple))
                except Exception as e:
                    logger.warning("FF factor fetch failed (%s), POFM disabled", e)

            # Align volume with returns
            vol_returns = volume.reindex(returns.index).fillna(0)
            rolling_mean_vol = vol_returns.rolling(
                dynamics_config.volume_lookback, min_periods=5).mean()

        # State
        current_weights = np.ones(n_assets) / n_assets
        portfolio_value = capital
        cov_est = CovarianceEstimator()
        pw = PortfolioWeights()

        equity = []
        weights_history = []
        cost_history = []
        decision_log = []

        # Day-by-day loop
        day_idx = lookback
        while day_idx < len(dates):
            date = dates[day_idx]
            days_since_start = day_idx - lookback

            # Determine if rebalance
            is_scheduled = (days_since_start % rebal_freq == 0)
            is_triggered = False

            if (transition_detector is not None and
                    not is_scheduled and
                    (dynamics_config.max_extra_rebalances < 0 or
                     extra_rebalance_count < dynamics_config.max_extra_rebalances)):
                if getattr(dynamics_config, 'use_optimal_stopping_rebalance', False):
                    cusum_mag = float(np.mean(transition_detector._cusum))
                    is_triggered = transition_detector.should_rebalance_optimal(
                        current_weights=current_weights,
                        target_weights=current_weights,
                        transaction_cost_bps=self.config.proportional_cost_bps,
                        signal_strength=cusum_mag,
                        min_interval=dynamics_config.min_rebalance_interval,
                    )
                else:
                    is_triggered = transition_detector.should_rebalance(
                        dynamics_config.min_rebalance_interval)

            if is_scheduled or is_triggered:
                # === REBALANCE BLOCK ===
                window_start = max(0, day_idx - lookback)
                window_returns = returns.iloc[window_start:day_idx]

                if len(window_returns) < 30:
                    day_idx += 1
                    continue

                # 1. Estimate covariance (POFM when FF data available)
                ff_win_simple = None
                if ff_daily_factors_simple is not None:
                    try:
                        ff_win_simple = ff_daily_factors_simple.iloc[
                            window_start:day_idx].values
                    except Exception:
                        ff_win_simple = None
                if ff_win_simple is not None and ff_win_simple.shape[0] == len(window_returns):
                    cov = CovarianceEstimator.pofm(
                        window_returns.values, ff_win_simple)
                else:
                    cov, _ = cov_est.ledoit_wolf(window_returns.values)

                # 2. Regime adjustment
                regime = None
                if regime_detector is not None and vix_series is not None:
                    try:
                        vix_val = float(vix_series.asof(date))
                        ys_val = (float(yield_spread_series.asof(date))
                                  if yield_spread_series is not None else 0.5)
                        regime = regime_detector.classify(vix_val, ys_val)
                        cov = CovarianceEstimator.regime_adjusted(
                            cov, regime,
                            regime_detector.config.regime_cov_multipliers)
                    except (KeyError, TypeError):
                        pass

                # 3. Expected returns
                mu = window_returns.mean().values

                if regime_detector is not None and regime is not None:
                    mu = mu + regime_detector.get_return_adjustment(regime)

                if sentiment_engine is not None:
                    try:
                        sent = sentiment_engine.compute(tickers=tickers)
                        mu = mu + sent["return_adjustment"]
                        cov = cov * sent["cov_multiplier"]
                    except Exception as e:
                        logger.warning("Sentiment failed on %s: %s", date, e)

                if bayesian_updater is not None:
                    try:
                        realized = window_returns.iloc[-rebal_freq:].mean().values
                        mu = bayesian_updater.update(
                            mu, realized, len(window_returns))
                    except Exception as e:
                        logger.warning("Bayesian update failed on %s: %s", date, e)

                # 4. Market dynamics adjustments
                if use_dynamics:
                    try:
                        mu, cov = self._apply_dynamics(
                            mu, cov, returns, window_returns, tickers,
                            day_idx, vol_returns, rolling_mean_vol,
                            factor_tracker, causal_graph,
                            transition_detector, signal_gen,
                            dynamics_config, dynamics_log, date,
                            ff_daily_factors=ff_daily_factors_simple)
                    except Exception as e:
                        logger.warning("Dynamics failed on %s: %s", date, e)

                # 5. Optimize
                try:
                    if weight_method == "max_sharpe":
                        new_weights = pw.maximum_sharpe(mu, cov)
                    elif weight_method == "min_variance":
                        new_weights = pw.minimum_variance(cov)
                    elif weight_method == "risk_parity":
                        new_weights = pw.risk_parity(cov)
                    else:
                        new_weights = pw.equal_weight(n_assets)
                except Exception as e:
                    logger.warning("Optimization failed on %s: %s", date, e)
                    new_weights = current_weights.copy()

                # 5d. Volatility targeting
                if getattr(self.config, 'vol_target_enabled', False):
                    vt_lookback = getattr(self.config, 'vol_target_lookback', 21)
                    vt_target = getattr(self.config, 'vol_target_annualized', 0.15)
                    if day_idx >= vt_lookback:
                        port_rets = (returns.iloc[day_idx - vt_lookback:day_idx]
                                     .values @ current_weights)
                        realized_vol = float(port_rets.std()) * np.sqrt(self.bars_per_year)
                        if realized_vol > 1e-6:
                            vol_scalar = vt_target / realized_vol
                            vt_max_lev = getattr(
                                self.config, 'vol_target_max_leverage', 1.5)
                            vt_min_exp = getattr(
                                self.config, 'vol_target_min_exposure', 0.3)
                            vol_scalar = float(
                                np.clip(vol_scalar, vt_min_exp, vt_max_lev))
                            new_weights = new_weights * vol_scalar
                            total = new_weights.sum()
                            if total > 1e-12:
                                new_weights = new_weights / total

                # 6. Transaction costs
                if self.config.use_almgren_chriss and vol_returns is not None:
                    cost = self._compute_rebalance_cost(
                        current_weights, new_weights, portfolio_value,
                        tickers, vol_returns, None, returns, day_idx)
                else:
                    cost = self.cost_model.portfolio_rebalance_cost(
                        current_weights, new_weights, portfolio_value)
                turnover = float(
                    np.sum(np.abs(new_weights - current_weights)) / 2)

                portfolio_value -= cost

                weights_history.append({
                    "date": date,
                    "weights": dict(zip(tickers, new_weights)),
                    "turnover": turnover,
                })
                cost_history.append({
                    "date": date, "cost": cost, "turnover": turnover,
                })
                decision_log.append({
                    "date": date,
                    "regime": regime,
                    "turnover": turnover,
                    "cost": cost,
                    "method": weight_method,
                    "triggered": is_triggered,
                })

                if is_triggered:
                    extra_rebalance_count += 1
                    dynamics_log.append({
                        "date": str(date),
                        "type": "triggered_rebalance",
                        "extra_count": extra_rebalance_count,
                    })

                current_weights = new_weights

            # === DAILY PORTFOLIO UPDATE ===
            day_ret = returns.iloc[day_idx]
            day_port_ret = float(current_weights @ day_ret.values)
            portfolio_value *= (1 + day_port_ret)

            # Daily borrow cost for short positions (v1)
            if self.config.allow_short and np.any(current_weights < 0):
                borrow = TransactionCostModel.compute_borrow_cost(
                    current_weights, portfolio_value,
                    self.config.short_borrow_annual_bps)
                portfolio_value -= borrow

            equity.append({
                "date": date,
                "value": portfolio_value,
                "weights": current_weights.copy(),
            })

            # === DAILY DYNAMICS UPDATE ===
            if (use_dynamics and transition_detector is not None and
                    factor_tracker.loadings is not None):
                try:
                    day_vol = vol_returns.iloc[day_idx].values
                    mean_vol = rolling_mean_vol.iloc[day_idx].values
                    B = factor_tracker.loadings

                    if not np.any(np.isnan(mean_vol)):
                        D_k = signal_gen.compute_volume_dislocation(
                            day_ret.values, day_vol, B, mean_vol)

                        # Information-time weighting
                        info_weight = 1.0
                        if getattr(dynamics_config, 'information_time_cusum', False):
                            safe_mean = np.maximum(mean_vol, 1e-8)
                            vol_ratio = day_vol / safe_mean
                            valid = np.isfinite(vol_ratio)
                            if valid.any():
                                info_weight = float(np.mean(vol_ratio[valid]))

                        update_result = transition_detector.update(
                            D_k, information_weight=info_weight)

                        if update_result["events"]:
                            new_phases = [ev[0] for ev in update_result["events"]]
                            dynamics_log.append({
                                "date": str(date),
                                "type": "phase_transition",
                                "events": update_result["events"],
                                "new_phase": new_phases[0] if len(new_phases) == 1 else ",".join(new_phases),
                                "cusum": [float(c) for c in update_result["cusum"]],
                            })
                except Exception as e:
                    logger.warning("Daily CUSUM update failed on %s: %s", date, e)

            day_idx += 1

        # Build equity curve
        equity_df = pd.DataFrame(equity)
        if equity_df.empty:
            raise ValueError("No backtest periods were generated")
        equity_df = equity_df.set_index("date")

        equity_returns = equity_df["value"].pct_change().dropna()
        perf = PerformanceMetrics(periods_per_year=self.bars_per_year)

        total_costs = sum(c["cost"] for c in cost_history)
        total_turnover = sum(c["turnover"] for c in cost_history)

        bpy = self.bars_per_year
        self._results = {
            "equity_curve": equity_df["value"],
            "returns": equity_returns,
            "weights_history": weights_history,
            "cost_history": cost_history,
            "decision_log": decision_log,
            "total_cost": total_costs,
            "total_turnover": total_turnover,
            "cost_drag_annual": total_costs / capital / max(len(equity_returns) / bpy, 1 / bpy),
            "metrics": perf.compute_all(equity_returns),
            "initial_capital": capital,
            "final_value": portfolio_value,
            "dynamics_log": dynamics_log,
            "extra_rebalances": extra_rebalance_count,
        }
        return self._results

    def _apply_dynamics(
        self, mu, cov, returns_full, window_returns, tickers,
        day_idx, vol_returns, rolling_mean_vol,
        factor_tracker, causal_graph,
        transition_detector, signal_gen,
        dynamics_config, dynamics_log, date,
        ff_daily_factors=None,
    ):
        """Apply market dynamics adjustments to mu and cov."""
        # Fit PCA on current window
        window_returns_df = pd.DataFrame(
            window_returns.values, columns=tickers,
            index=window_returns.index)

        # Prepare FF factor data for this window (POFM)
        ff_window = None
        if ff_daily_factors is not None:
            try:
                ff_window = ff_daily_factors.reindex(
                    window_returns_df.index).ffill().fillna(0.0).values
            except Exception:
                ff_window = None

        factor_tracker.fit(window_returns_df, ff_factor_data=ff_window)

        # Endogeneity diagnostic (UECL bias estimate)
        if (getattr(dynamics_config, 'endogeneity_diagnostic', False)
                and ff_window is not None):
            try:
                from NR.dynamics.factor_tracker import FactorTracker as _FT
                diag = _FT.compute_endogeneity_diagnostic(
                    window_returns.values, ff_window)
                dynamics_log.append({
                    "date": str(date),
                    "type": "endogeneity_diagnostic",
                    "frobenius_norm": diag['frobenius_norm'],
                    "mean_abs_bias": [float(x) for x in diag['mean_abs_bias']],
                    "max_abs_bias": [float(x) for x in diag['max_abs_bias']],
                    "n_latent_used": diag['n_latent_used'],
                })
            except Exception as e:
                logger.warning("Endogeneity diagnostic failed on %s: %s", date, e)

        K = factor_tracker.n_active_factors
        B = factor_tracker.loadings

        # Resize transition detector if factor count changed (preserves CUSUM history)
        if transition_detector.n_factors != K:
            transition_detector.resize(K)

        # Build Granger causal graph on FF-residualized returns when POFM active
        granger_input = window_returns.values
        if ff_window is not None:
            try:
                F_gc = np.column_stack([np.ones(ff_window.shape[0]), ff_window])
                coeffs_gc = np.linalg.lstsq(F_gc, granger_input, rcond=None)[0]
                granger_input = granger_input - F_gc @ coeffs_gc
            except Exception:
                pass
        causal_graph.fit(granger_input)
        correlation = CovarianceEstimator.to_correlation(cov)
        causal_M = causal_graph.build_propagation_matrix(
            alpha=dynamics_config.granger_alpha,
            correlation=correlation,
            safety_factor=getattr(dynamics_config, 'alpha_safety_factor', 0.7))

        # Get phase risk scale
        phase_scale = transition_detector.get_risk_scale(
            dynamics_config.phase_risk_scales)

        # Adjust covariance
        cov = signal_gen.compute_cov_adjustment(cov, phase_scale, causal_M)

        # Compute average dislocation over recent volume_lookback days
        vl = dynamics_config.volume_lookback
        recent_end = day_idx
        recent_start = max(0, day_idx - vl)

        D_k_avg = np.zeros(K)
        count = 0

        mean_vol_idx = min(day_idx - 1, len(rolling_mean_vol) - 1)
        if mean_vol_idx >= 0:
            mean_vol = rolling_mean_vol.iloc[mean_vol_idx].values
        else:
            mean_vol = np.ones(len(tickers))

        if not np.any(np.isnan(mean_vol)):
            for t_idx in range(recent_start, recent_end):
                if t_idx >= len(returns_full) or t_idx >= len(vol_returns):
                    continue
                r_t = returns_full.iloc[t_idx].values
                v_t = vol_returns.iloc[t_idx].values
                D_k_t = signal_gen.compute_volume_dislocation(
                    r_t, v_t, B, mean_vol)
                D_k_avg += D_k_t
                count += 1

        if count > 0:
            D_k_avg /= count

        # Apply return adjustment
        phases = transition_detector._phase
        mu_adj = signal_gen.compute_return_adjustment(D_k_avg, B, phases)
        mu = mu + mu_adj

        # Log dynamics state
        dynamics_log.append({
            "date": str(date),
            "type": "rebalance_dynamics",
            "n_factors": K,
            "granger_edges": causal_graph.n_edges,
            "phase": transition_detector.get_aggregate_phase(),
            "phase_scale": phase_scale,
            "loading_change": factor_tracker.loading_change_norm(),
        })

        return mu, cov

    # ================================================================
    # EVENT-DRIVEN BACKTEST ENGINE (v2)
    # ================================================================

    def run_event_driven(
        self,
        prices: pd.DataFrame,
        open_prices: pd.DataFrame = None,
        high_prices: pd.DataFrame = None,
        low_prices: pd.DataFrame = None,
        volume: pd.DataFrame = None,
        spy_prices: pd.Series = None,
        dynamics_config=None,
        regime_detector=None,
        vix_series: pd.Series = None,
        yield_spread_series: pd.Series = None,
        dxy_series: pd.Series = None,
        tnx_series: pd.Series = None,
        event_attribution_config=None,
        position_manager_config=None,
        weight_method: str = "risk_parity",
        sentiment_engine=None,
        institutional_config=None,
        credit_spread_series: pd.Series = None,
        kbe_series: pd.Series = None,
        agent_config=None,
        resume_checkpoint: dict = None,
    ) -> dict:
        """Event-driven backtest with lookahead prevention.

        Three-stage day loop:
          Stage 1: State Update — mark-to-market with day i prices
          Stage 2: Signal Detection — estimates from data[0:i] only
          Stage 3: Execution — optimize, impact model, trade

        Args:
            prices: Daily close prices (N assets).
            open_prices: Daily open prices (N assets, for OFI).
            volume: Daily volume (N assets).
            spy_prices: SPY close prices for benchmark tracking.
            dynamics_config: DynamicsConfig instance.
            regime_detector: Optional RegimeDetector.
            vix_series: VIX series for regime detection.
            yield_spread_series: Yield spread for regime.
            weight_method: Optimization method.
            agent_config: Optional AgentConfig for LLM orchestration.

        Returns:
            Dict with equity_curve, spy_equity_curve, returns,
            weights_history, cost_history, decision_log, dynamics_log,
            validation_log, metrics (inc. alpha, beta, info_ratio).
        """
        returns = prices.pct_change().dropna()
        tickers = list(prices.columns)
        n_assets = len(tickers)
        lookback = self.config.lookback_window
        rebal_freq = self.config.rebalance_frequency
        capital = self.config.initial_capital

        dates = returns.index
        if len(dates) < lookback + rebal_freq:
            raise ValueError(
                f"Need at least {lookback + rebal_freq} return observations, "
                f"got {len(dates)}")

        # Initialize dynamics
        use_dynamics = (dynamics_config is not None and
                        dynamics_config.enabled and
                        volume is not None)

        factor_tracker = None
        transition_detector = None
        signal_gen = None
        causal_graph = None
        dynamics_log = []
        extra_rebalance_count = 0

        if use_dynamics:
            from NR.dynamics.factor_tracker import FactorTracker
            from NR.dynamics.transition_detector import TransitionDetector
            from NR.dynamics.signal_generator import SignalGenerator
            from NR.dynamics.causal_graph import GrangerCausalGraph

            factor_tracker = FactorTracker(
                n_factors=dynamics_config.n_factors,
                variance_threshold=dynamics_config.pca_variance_threshold,
                use_kalman_dfm=getattr(dynamics_config, 'use_kalman_dfm', False),
                kalman_process_noise=getattr(dynamics_config, 'kalman_dfm_process_noise', 1e-4),
                kalman_obs_noise=getattr(dynamics_config, 'kalman_dfm_obs_noise', 1e-3),
                loadings_source="graph_vae" if getattr(dynamics_config, 'use_graph_vae', False) else "pca",
                use_rbpf=getattr(dynamics_config, 'use_rbpf', False),
                rbpf_n_particles=getattr(dynamics_config, 'rbpf_n_particles', 50),
                rbpf_resample_threshold=getattr(dynamics_config, 'rbpf_resample_threshold', 0.5),
                rbpf_process_noise_scale=getattr(dynamics_config, 'rbpf_process_noise_scale', 0.01),
                graph_vae_use_directed=getattr(dynamics_config, 'graph_vae_use_directed', True),
                ff_eigenvalue_test=getattr(dynamics_config, 'ff_eigenvalue_test', False),
                ff_eigenvalue_significance=getattr(dynamics_config, 'ff_eigenvalue_significance', 0.05),
                ff_k_min=getattr(dynamics_config, 'ff_k_min', 1),
                ff_k_max=getattr(dynamics_config, 'ff_k_max', 0))
            transition_detector = TransitionDetector(
                n_factors=dynamics_config.n_factors,
                drift=dynamics_config.cusum_drift,
                threshold=dynamics_config.cusum_threshold,
                dislocation_decay_days=dynamics_config.dislocation_decay_days,
                convergence_trigger_days=dynamics_config.convergence_trigger_days,
                burn_in=dynamics_config.cusum_burn_in,
                use_welford=dynamics_config.cusum_use_welford)
            signal_gen = SignalGenerator(
                dislocation_return_scale=dynamics_config.dislocation_return_scale,
                granger_alpha=dynamics_config.granger_alpha)
            causal_graph = GrangerCausalGraph(
                lags=dynamics_config.granger_lags,
                p_threshold=dynamics_config.granger_p_threshold,
                min_edges=dynamics_config.granger_min_edges,
                max_assets=getattr(dynamics_config, 'max_granger_assets', 0),
                use_transfer_entropy=getattr(dynamics_config, 'use_transfer_entropy', False),
                te_bins=getattr(dynamics_config, 'te_bins', 6),
                te_k=getattr(dynamics_config, 'te_k', 1),
                te_l=getattr(dynamics_config, 'te_l', 1),
                te_threshold=getattr(dynamics_config, 'te_threshold', 0.01),
                te_method=getattr(dynamics_config, 'te_method', 'binned'),
                te_knn_k=getattr(dynamics_config, 'te_knn_k', 5),
                te_bootstrap_n=getattr(dynamics_config, 'te_bootstrap_n', 0),
                te_bootstrap_alpha=getattr(dynamics_config, 'te_bootstrap_alpha', 0.05))

        # POFM: Fetch Fama-French factor data for FF5 residualization
        ff_daily_factors = None
        rmw_betas = None  # Per-stock RMW profitability betas for quality gate
        if use_dynamics and getattr(dynamics_config, 'ff_residualize_pca', False):
            try:
                from NR.macro.factors import FactorModel
                from NR.config import MacroConfig
                ff_n = getattr(dynamics_config, 'ff_factors', 5)
                ff_config = MacroConfig(ff_factors=ff_n)
                ff_model = FactorModel(ff_config)
                ff_raw = ff_model.fetch_factors(
                    start_date=str(returns.index[0].date()),
                    end_date=str(returns.index[-1].date()))
                factor_cols = [c for c in ff_raw.columns if c != "RF"]
                ff_daily_factors = self._align_ff_factors(
                    ff_raw, factor_cols, returns.index)
                logger.info("POFM: loaded %d FF factors, %d dates",
                            len(factor_cols), len(ff_daily_factors))
                # Compute per-stock FF5 betas for quality gating
                try:
                    ff_model.fit(returns)
                    _ff_betas = ff_model.betas  # DataFrame: ticker x factor
                    if _ff_betas is not None and "RMW" in _ff_betas.columns:
                        rmw_betas = (_ff_betas["RMW"]
                                     .reindex(tickers)
                                     .fillna(0.0).values)
                        logger.info("FF5 RMW betas computed: mean=%.3f, "
                                    "std=%.3f, %%negative=%.1f%%",
                                    rmw_betas.mean(), rmw_betas.std(),
                                    100 * (rmw_betas < 0).mean())
                except Exception as e:
                    logger.warning("FF5 beta computation failed: %s", e)
            except Exception as e:
                logger.warning("FF factor fetch failed (%s), POFM disabled", e)
                ff_daily_factors = None

        # Manifold L tracking (rolling factor covariance history)
        cov_history = []

        # LSTM factor forecaster
        lstm_forecaster = None
        if use_dynamics and getattr(dynamics_config, 'use_lstm_forecaster', False):
            try:
                from NR.dynamics.lstm_forecaster import LSTMFactorForecaster
                lstm_forecaster = LSTMFactorForecaster(
                    n_factors=dynamics_config.n_factors,
                    hidden_dim=getattr(dynamics_config, 'lstm_hidden_dim', 32),
                    num_layers=getattr(dynamics_config, 'lstm_num_layers', 2),
                    seq_len=getattr(dynamics_config, 'lstm_seq_len', 63),
                    dropout=getattr(dynamics_config, 'lstm_dropout', 0.1),
                    learning_rate=getattr(dynamics_config, 'lstm_lr', 1e-3),
                    n_epochs=getattr(dynamics_config, 'lstm_epochs', 50),
                )
            except ImportError:
                logger.warning("PyTorch unavailable, LSTM forecaster disabled")

        # RL allocator (PPO)
        rl_allocator = None
        rl_env = None
        if use_dynamics and getattr(dynamics_config, 'use_rl_allocator', False):
            try:
                from NR.dynamics.rl_allocator import (
                    PPOAllocator, PortfolioEnv)
                rl_env = PortfolioEnv(
                    n_assets, cost_model=self.cost_model,
                    reward_type=getattr(dynamics_config, 'rl_reward_type', 'return'))
                rl_allocator = PPOAllocator(
                    n_assets=n_assets,
                    n_factors=dynamics_config.n_factors,
                    hidden_dim=getattr(dynamics_config, 'rl_hidden_dim', 128),
                    lr=getattr(dynamics_config, 'rl_lr', 3e-4),
                    gamma=getattr(dynamics_config, 'rl_gamma', 0.99),
                    clip_epsilon=getattr(dynamics_config, 'rl_clip_epsilon', 0.2),
                    n_update_epochs=getattr(dynamics_config, 'rl_update_epochs', 4),
                    buffer_size=getattr(dynamics_config, 'rl_buffer_size', 32),
                    entropy_coef=getattr(dynamics_config, 'rl_entropy_coef', 0.01),
                    value_coef=getattr(dynamics_config, 'rl_value_coef', 0.5),
                )
                pretrained = getattr(dynamics_config, 'rl_pretrained_path', None)
                if pretrained:
                    rl_allocator.load(pretrained)
                rl_env.reset(dynamics_config.n_factors, capital)
            except ImportError:
                logger.warning("PyTorch unavailable, RL allocator disabled")

        # OFI-based volume decomposer
        volume_decomposer = None
        dhat_smoothed = None
        _xi_z_smoothed = None
        if use_dynamics and getattr(dynamics_config, 'use_ofi_dhat', False):
            from NR.dynamics.volume_decomposer import VolumeDecomposer
            volume_decomposer = VolumeDecomposer(
                ofi_lookback=dynamics_config.ofi_dhat_lookback)
            volume_decomposer.reset(n_assets)

        # Fiber Bundle State Space
        fiber_bundle = None
        if use_dynamics and getattr(dynamics_config, 'fiber_bundle_enabled', False):
            from NR.dynamics.fiber_bundle import StratifiedStateSpace
            fiber_bundle = StratifiedStateSpace(
                initial_K=dynamics_config.n_factors,
                blend_window=getattr(dynamics_config, 'bundle_blend_window', 5),
                metric_coupling=getattr(dynamics_config, 'bundle_metric_coupling', 0.0),
                proper_time_method=getattr(dynamics_config, 'proper_time_method', 'realized_variance'),
                cusum_drift=dynamics_config.cusum_drift,
                cusum_threshold=dynamics_config.cusum_threshold,
                dislocation_decay_days=dynamics_config.dislocation_decay_days,
                convergence_trigger_days=dynamics_config.convergence_trigger_days,
            )

        # CDR (Conviction-Direction-Risk) state
        _cdr_holding_days = np.zeros(n_assets)
        _cdr_last_rebal_day = -9999

        # Correlation Manifold Dynamics (stored on self for _apply_dynamics_v2 access)
        self._manifold_evolution = None
        self._manifold_state = None
        self._grassmannian_tracker = None
        self._soft_phase_estimator = None
        self._operator_spectrum = None
        if use_dynamics and getattr(dynamics_config, 'correlation_manifold_enabled', False):
            from NR.dynamics.correlation_flow import CorrelationFlowTracker
            from NR.dynamics.manifold_dynamics import (
                ManifoldDynamicsEngine, MarketFieldEquation)
            from NR.dynamics.unified_state import MarketEvolution
            from NR.dynamics.manifold import SPDManifold as _SPD

            _spd = _SPD()
            _flow_tracker = CorrelationFlowTracker(
                max_history=getattr(dynamics_config, 'correlation_flow_max_history', 63),
                meta_correlation_window=getattr(dynamics_config, 'correlation_flow_meta_window', 21),
                velocity_ema_alpha=getattr(dynamics_config, 'correlation_flow_velocity_ema', 0.3),
            )
            _dyn_engine = ManifoldDynamicsEngine(
                potential_strength=getattr(dynamics_config, 'manifold_dynamics_potential_strength', 0.1),
                damping=getattr(dynamics_config, 'manifold_dynamics_damping', 0.05),
                max_velocity_norm=getattr(dynamics_config, 'manifold_dynamics_max_velocity', 1.0),
                integrator=getattr(dynamics_config, 'manifold_dynamics_integrator', 'verlet'),
            )
            _field_eq = MarketFieldEquation(
                coupling_constant=getattr(dynamics_config, 'field_equation_coupling', 0.01),
                volume_weight=getattr(dynamics_config, 'field_equation_volume_weight', 0.4),
                return_weight=getattr(dynamics_config, 'field_equation_return_weight', 0.3),
                weight_change_weight=getattr(dynamics_config, 'field_equation_weight_change_weight', 0.3),
            )
            self._manifold_evolution = MarketEvolution(
                dynamics_engine=_dyn_engine,
                flow_tracker=_flow_tracker,
                field_equation=_field_eq,
                manifold=_spd,
                blend_weight=getattr(dynamics_config, 'manifold_dynamics_blend_weight', 0.3),
                energy_divergence_threshold=getattr(dynamics_config, 'manifold_energy_divergence_threshold', 10.0),
            )
            if getattr(dynamics_config, 'grassmannian_enabled', False):
                from NR.dynamics.grassmannian import GrassmannianTracker
                self._grassmannian_tracker = GrassmannianTracker(
                    blend_window=getattr(dynamics_config, 'grassmannian_blend_window', 5))
            if getattr(dynamics_config, 'soft_phases_enabled', False):
                from NR.dynamics.soft_phase import SoftPhaseEstimator
                self._soft_phase_estimator = SoftPhaseEstimator(
                    steepness=getattr(dynamics_config, 'soft_phase_steepness', 5.0),
                    rbpf_blend=getattr(dynamics_config, 'soft_phase_rbpf_blend', 0.5))
            if getattr(dynamics_config, 'operator_spectrum_enabled', False):
                from NR.dynamics.operator_spectrum import OperatorSpectrum
                self._operator_spectrum = OperatorSpectrum(
                    marginal_threshold=getattr(dynamics_config, 'operator_spectrum_threshold', 0.95))

        # Pass graph_weighted_pca to factor tracker
        if use_dynamics and factor_tracker is not None:
            factor_tracker.graph_weighted_pca = getattr(
                dynamics_config, 'graph_weighted_pca', False)

        # Cross-sectional alpha engine
        alpha_engine = None
        if use_dynamics and (getattr(dynamics_config, 'alpha_enabled', False)
                             or getattr(dynamics_config, 'regime_blend_alpha', False)):
            from NR.dynamics.alpha_signals import CrossSectionalAlpha
            alpha_engine = CrossSectionalAlpha(
                momentum_window=dynamics_config.alpha_momentum_window,
                momentum_skip=dynamics_config.alpha_momentum_skip,
                reversal_window=dynamics_config.alpha_reversal_window,
                vol_adjust=dynamics_config.alpha_vol_adjust)

        # Do-calculus causal filter
        causal_filter = None
        if use_dynamics and getattr(dynamics_config, 'causal_filter_enabled', False):
            from NR.dynamics.causal_filter import CausalFilter
            causal_filter = CausalFilter(
                min_edge_weight=dynamics_config.causal_min_edge_weight,
                max_confounders=dynamics_config.causal_max_confounders,
                regularization=dynamics_config.causal_regularization)

        # Event attribution
        event_attributor = None
        attribution_log = []
        if (use_dynamics and event_attribution_config is not None
                and event_attribution_config.enabled):
            from NR.dynamics.event_attributor import EventAttributor
            event_attributor = EventAttributor(event_attribution_config)

        # Position lifecycle manager
        position_manager = None
        use_lifecycle = (position_manager_config is not None
                         and position_manager_config.enabled
                         and use_dynamics)
        if use_lifecycle:
            from NR.positioning.lifecycle import (
                PositionLifecycleManager)
            position_manager = PositionLifecycleManager(
                config=position_manager_config,
                tickers=tickers,
            )

        # LLM Semantic Orchestration agents
        llm_orchestrator = None
        if agent_config is None:
            agent_config = getattr(self, 'agent_config', None)
        if agent_config is not None and agent_config.enabled:
            try:
                from NR.agents import LLMOrchestrator
                llm_orchestrator = LLMOrchestrator(
                    agent_config, tickers, mode=agent_config.mode)
                logger.info("LLM Orchestrator initialized (mode=%s)",
                            agent_config.mode)
            except Exception as e:
                logger.warning("LLM Orchestrator init failed: %s", e)

        # Beta-Binomial edge updater for Granger graph
        _edge_updater = None
        if use_dynamics and getattr(dynamics_config, 'use_bayesian_edges', False):
            from NR.dynamics.edge_updater import BetaBinomialEdgeUpdater
            _edge_updater = BetaBinomialEdgeUpdater(
                n_assets=n_assets,
                prior_alpha=getattr(dynamics_config, 'bayesian_edge_prior_alpha', 1.0),
                prior_beta=getattr(dynamics_config, 'bayesian_edge_prior_beta', 1.0),
                decay=getattr(dynamics_config, 'bayesian_edge_decay', 0.995))

        # Sentiment-phase cross-validator
        _sentiment_phase_validator = None
        if use_dynamics and sentiment_engine is not None:
            from NR.sentiment.phase_validator import SentimentPhaseValidator
            _sentiment_phase_validator = SentimentPhaseValidator()

        # Markov regime detector
        _markov_detector = None
        _macro_cfg = regime_detector.config if regime_detector is not None else None
        if getattr(_macro_cfg, 'use_markov_regime', False):
            from NR.macro.regime import MarkovRegimeDetector
            _markov_detector = MarkovRegimeDetector(
                burn_in=getattr(_macro_cfg, 'markov_burn_in', 63))

        # Insider activity signal
        _insider_signal = None
        _institutional_cache = None
        _institutional_consensus = None
        _propagation_signal = None
        _hf_crowding_signal = None
        _all_inst_trades = {}
        if institutional_config is not None and institutional_config.enabled:
            try:
                from NR.institutional.cache import InstitutionalCache
                _institutional_cache = InstitutionalCache(
                    institutional_config.cache_dir)
                _institutional_cache.connect()
            except Exception:
                _institutional_cache = None

            if getattr(institutional_config, 'consensus_weight_in_entry', 0) > 0:
                try:
                    from NR.institutional.consensus import InstitutionalConsensus
                    _institutional_consensus = InstitutionalConsensus(
                        lookback_quarters=institutional_config.consensus_lookback_quarters)
                except Exception:
                    pass

            # Load institutional trades and pre-compute propagation signal
            if _institutional_cache is not None:
                try:
                    from NR.institutional.trade_differ import TradeDiffer
                    from NR.institutional.cusip_mapper import CusipMapper
                    mapper = CusipMapper(cache=_institutional_cache)
                    differ = TradeDiffer(cusip_mapper=mapper)

                    for cik in (institutional_config.target_ciks or []):
                        report_dates = _institutional_cache.get_all_report_dates(cik)
                        if len(report_dates) < 2:
                            continue
                        filings = []
                        for rd in report_dates:
                            h = _institutional_cache.get_holdings(cik, rd)
                            if h is not None and not h.empty:
                                filings.append((rd, h))
                        if len(filings) >= 2:
                            rds = [f[0] for f in filings]
                            trades = differ.diff_all_quarters(
                                [f[1] for f in filings], rds,
                                cik, "Institution")
                            valid = [t for t in trades if t.ticker]
                            if valid:
                                _all_inst_trades[cik] = valid

                    if (_all_inst_trades
                            and getattr(institutional_config, 'propagation_enabled', False)):
                        from NR.institutional.propagation_signal import (
                            PropagationSpeedSignal)
                        _propagation_signal = PropagationSpeedSignal(
                            lookback_quarters=getattr(
                                institutional_config,
                                'propagation_lookback_quarters', 4),
                            decay_halflife_quarters=getattr(
                                institutional_config,
                                'propagation_decay_halflife_quarters', 2.0),
                        )
                        filing_dates = {}
                        try:
                            filing_dates = _institutional_cache.get_filing_dates_dict()
                        except Exception:
                            pass
                        _propagation_signal.precompute(
                            all_trades=_all_inst_trades,
                            returns_df=returns,
                            filing_dates=filing_dates,
                        )
                except Exception as e:
                    import traceback
                    logger.warning("Institutional trade loading failed: %s", e)
                    traceback.print_exc()

        # HF crowding signal (from 13F trades)
        if (_all_inst_trades
                and institutional_config is not None
                and getattr(institutional_config, 'crowding_enabled', False)):
            try:
                from NR.institutional.crowding import HFCrowdingSignal
                _hf_crowding_signal = HFCrowdingSignal(
                    n_institutions=len(
                        institutional_config.target_ciks or []),
                    lookback_quarters=getattr(
                        institutional_config, 'crowding_lookback_quarters', 4),
                )
                _hf_crowding_signal.precompute(
                    all_trades=_all_inst_trades, tickers=tickers)
            except Exception as e:
                logger.warning("HF crowding init failed: %s", e)

        if use_dynamics and getattr(dynamics_config, 'use_insider_signal', False):
            from NR.signals.insider_activity import InsiderActivitySignal
            _insider_signal = InsiderActivitySignal(cache=_institutional_cache)

        # Precompute volume statistics
        vol_aligned = None
        rolling_mean_vol = None
        open_aligned = None
        high_aligned = None
        low_aligned = None
        spy_rolling_avg = None
        momentum_series = None

        if use_dynamics and volume is not None:
            vol_aligned = volume.reindex(returns.index).fillna(0)
            vl = dynamics_config.volume_lookback
            rolling_mean_vol = vol_aligned.rolling(vl, min_periods=5).mean()

            if open_prices is not None:
                open_aligned = open_prices.reindex(returns.index).ffill()
            if high_prices is not None:
                high_aligned = high_prices.reindex(returns.index).ffill()
            if low_prices is not None:
                low_aligned = low_prices.reindex(returns.index).ffill()

            # Momentum proxy for v_pos (63-day cumulative return)
            mom_w = dynamics_config.momentum_proxy_window
            momentum_series = returns.rolling(mom_w, min_periods=10).sum()

        # SPY benchmark
        spy_returns = None
        spy_initial_price = None
        if spy_prices is not None:
            spy_returns = spy_prices.pct_change().dropna()

        # SPY volume stats (for v_mkt)
        spy_vol_series = None
        spy_vol_avg = None

        # State
        current_weights = np.ones(n_assets) / n_assets
        portfolio_value = capital
        cov_est = CovarianceEstimator()
        pw = PortfolioWeights()

        equity = []
        spy_equity = []
        weights_history = []
        cost_history = []
        decision_log = []
        L_history = []
        kappa_history = []
        dk_series = []
        is_exited = False  # Track if we've emergency-exited
        cash_weight = 0.0  # Fraction of portfolio in cash
        _dd_ema = 0.0      # EMA-smoothed drawdown for graduated scaling
        _gate_ema = 1.0    # EMA-smoothed regime gate (starts fully deployed)
        lifecycle_result = None  # Position lifecycle state
        daily_alpha_scores = np.zeros(n_assets)  # Alpha scores for lifecycle
        _mispricing_history = []          # Rolling mispricing for cumulative signal
        _dispersion_history = []          # Rolling eigenvalue dispersion
        _curvature_gate = 1.0            # Curvature regime gate [0.3, 1.5]
        _propagation_mispricing_scores = None  # Persisted from last rebalance
        _latent_shock_scores = None       # Persisted from last rebalance
        _sia_outgoing_scores = None       # Persisted from last rebalance
        _factor_tau_history = []          # Eigenvalue history for τ_k stability
        _current_tau_k = None             # Current info absorption rates
        _factor_interaction_F = None      # K x K factor interaction matrix
        _latency_alpha_z = None           # Persisted latency alpha z-scores
        _lead_lag_signal = None           # LeadLagSignal instance
        _lead_lag_scores = None           # Persisted lead-lag catch-up scores

        # Initialize lead-lag signal if enabled
        if use_dynamics and getattr(dynamics_config, 'lead_lag_enabled', False):
            from NR.propagation.lead_lag_signal import (
                LeadLagSignal, LeadLagConfig,
            )
            _ll_cfg = LeadLagConfig(
                enabled=True,
                leader_threshold=dynamics_config.lead_lag_leader_threshold,
                max_lag_days=dynamics_config.lead_lag_max_lag_days,
                decay_halflife=dynamics_config.lead_lag_decay_halflife,
                trailing_vol_window=dynamics_config.lead_lag_trailing_vol_window,
                gap_ema_alpha=dynamics_config.lead_lag_gap_ema_alpha,
                min_leader_count=dynamics_config.lead_lag_min_leader_count,
                max_leader_fraction=dynamics_config.lead_lag_max_leader_fraction,
                cascade_clip=dynamics_config.lead_lag_cascade_clip,
            )
            _lead_lag_signal = LeadLagSignal(_ll_cfg)
            _lead_lag_signal.reset(n_assets)

        # Pre-convert to numpy for hot-path daily accesses
        returns_arr = returns.values
        prices_arr = prices.values

        # ========== CHECKPOINT RESTORE ==========
        _resuming = False
        _start_day_idx = lookback

        if resume_checkpoint is not None:
            _ckpt = resume_checkpoint
            _ckpt_tickers = _ckpt.get('tickers', [])
            if set(_ckpt_tickers) != set(tickers):
                logger.warning(
                    "Universe changed (%d -> %d tickers), ignoring checkpoint",
                    len(_ckpt_tickers), len(tickers))
            elif _ckpt.get('day_idx', 0) >= len(dates) - 1:
                logger.info("Checkpoint is up-to-date (day_idx=%d, "
                            "total_days=%d), no new data to process",
                            _ckpt['day_idx'], len(dates))
                _resuming = True
                _start_day_idx = len(dates)  # skip loop entirely
            else:
                _resuming = True
                _start_day_idx = _ckpt['day_idx'] + 1
                logger.info("Resuming from checkpoint: day_idx=%d -> %d "
                            "(%d new days), portfolio=%.0f",
                            _ckpt['day_idx'], len(dates) - 1,
                            len(dates) - _start_day_idx,
                            _ckpt.get('portfolio_value', 0))

            # Restore all state from checkpoint (for both up-to-date
            # and normal resume paths)
            if _resuming:
                # Restore portfolio state
                current_weights = _ckpt['current_weights']
                portfolio_value = _ckpt['portfolio_value']
                cash_weight = _ckpt.get('cash_weight', 0.0)
                is_exited = _ckpt.get('is_exited', False)
                _dd_ema = _ckpt.get('_dd_ema', 0.0)
                _gate_ema = _ckpt.get('_gate_ema', 1.0)
                self._portfolio_peak = _ckpt.get('_portfolio_peak',
                                                  portfolio_value)

                # Restore sub-component objects
                if _ckpt.get('factor_tracker') is not None:
                    factor_tracker = _ckpt['factor_tracker']
                if _ckpt.get('transition_detector') is not None:
                    transition_detector = _ckpt['transition_detector']
                if _ckpt.get('signal_gen') is not None:
                    signal_gen = _ckpt['signal_gen']
                if _ckpt.get('causal_graph') is not None:
                    causal_graph = _ckpt['causal_graph']
                if _ckpt.get('volume_decomposer') is not None:
                    volume_decomposer = _ckpt['volume_decomposer']
                if _ckpt.get('alpha_engine') is not None:
                    alpha_engine = _ckpt['alpha_engine']
                if _ckpt.get('causal_filter') is not None:
                    causal_filter = _ckpt['causal_filter']
                if _ckpt.get('position_manager') is not None:
                    position_manager = _ckpt['position_manager']
                if _ckpt.get('llm_orchestrator') is not None:
                    llm_orchestrator = _ckpt['llm_orchestrator']
                if _ckpt.get('_edge_updater') is not None:
                    _edge_updater = _ckpt['_edge_updater']
                if _ckpt.get('_sentiment_phase_validator') is not None:
                    _sentiment_phase_validator = _ckpt[
                        '_sentiment_phase_validator']
                if _ckpt.get('_markov_detector') is not None:
                    _markov_detector = _ckpt['_markov_detector']
                if _ckpt.get('_lead_lag_signal') is not None:
                    _lead_lag_signal = _ckpt['_lead_lag_signal']
                if _ckpt.get('fiber_bundle') is not None:
                    fiber_bundle = _ckpt['fiber_bundle']
                if _ckpt.get('lstm_forecaster') is not None:
                    lstm_forecaster = _ckpt['lstm_forecaster']

                # Restore smoothed / accumulated signals
                dhat_smoothed = _ckpt.get('dhat_smoothed')
                _xi_z_smoothed = _ckpt.get('_xi_z_smoothed')
                daily_alpha_scores = _ckpt.get(
                    'daily_alpha_scores', np.zeros(n_assets))
                _mispricing_history = _ckpt.get('_mispricing_history', [])
                _dispersion_history = _ckpt.get('_dispersion_history', [])
                _curvature_gate = _ckpt.get('_curvature_gate', 1.0)
                _propagation_mispricing_scores = _ckpt.get(
                    '_propagation_mispricing_scores')
                _latent_shock_scores = _ckpt.get('_latent_shock_scores')
                _sia_outgoing_scores = _ckpt.get('_sia_outgoing_scores')
                _factor_tau_history = _ckpt.get('_factor_tau_history', [])
                _current_tau_k = _ckpt.get('_current_tau_k')
                _factor_interaction_F = _ckpt.get('_factor_interaction_F')
                _latency_alpha_z = _ckpt.get('_latency_alpha_z')
                _lead_lag_scores = _ckpt.get('_lead_lag_scores')

                # Restore CDR state
                _cdr_holding_days = _ckpt.get(
                    '_cdr_holding_days', np.zeros(n_assets))
                _cdr_last_rebal_day = _ckpt.get('_cdr_last_rebal_day', -9999)

                # Restore dynamics metrics
                extra_rebalance_count = _ckpt.get(
                    'extra_rebalance_count', 0)
                cov_history = _ckpt.get('cov_history', [])
                kappa_current = _ckpt.get('kappa_current', 1.0)
                L_current = _ckpt.get('L_current', 0.0)

                # Restore histories
                equity = _ckpt.get('equity', [])
                spy_equity = _ckpt.get('spy_equity', [])
                weights_history = _ckpt.get('weights_history', [])
                cost_history = _ckpt.get('cost_history', [])
                decision_log = _ckpt.get('decision_log', [])
                dynamics_log = _ckpt.get('dynamics_log', [])
                L_history = _ckpt.get('L_history', [])
                kappa_history = _ckpt.get('kappa_history', [])
                dk_series = _ckpt.get('dk_series', [])
                attribution_log = _ckpt.get('attribution_log', [])
                lifecycle_result = _ckpt.get('lifecycle_result')

                # Restore self attributes
                self._last_causal_M = _ckpt.get('_last_causal_M')
                self._last_info_weight = _ckpt.get('_last_info_weight', 1.0)
                if _ckpt.get('_last_soft_phases') is not None:
                    self._last_soft_phases = _ckpt['_last_soft_phases']
                if _ckpt.get('_last_spectrum') is not None:
                    self._last_spectrum = _ckpt['_last_spectrum']

                # Restore SPY initial price for benchmark continuity
                spy_initial_price = _ckpt.get('spy_initial_price')

        # ========== DAY LOOP ==========
        total_days = len(dates) - lookback
        for day_idx in range(_start_day_idx, len(dates)):
            date = dates[day_idx]
            days_since_start = day_idx - lookback

            _log_interval = self.data_config.scale_window(250)
            if days_since_start > 0 and days_since_start % _log_interval == 0:
                logger.info(
                    "Backtest progress: day %d/%d (%.0f%%) | %s | value=%.0f",
                    days_since_start, total_days,
                    100 * days_since_start / total_days,
                    str(date)[:10], portfolio_value)

            update_result = None

            # ===== STAGE 1: STATE UPDATE =====
            day_ret = returns_arr[day_idx]
            day_port_ret = (float(current_weights @ day_ret)
                            + cash_weight * self.config.cash_return_daily)
            portfolio_value *= (1 + day_port_ret)

            # Daily borrow cost for short positions (v2)
            if self.config.allow_short and np.any(current_weights < 0):
                borrow = TransactionCostModel.compute_borrow_cost(
                    current_weights, portfolio_value,
                    self.config.short_borrow_annual_bps)
                portfolio_value -= borrow

            # Portfolio drawdown limit — graduated scaling with EMA smoothing
            _portfolio_peak = max(
                getattr(self, '_portfolio_peak', portfolio_value),
                portfolio_value)
            self._portfolio_peak = _portfolio_peak
            max_dd_limit = getattr(self.config, 'max_portfolio_drawdown', 1.0)
            if max_dd_limit < 1.0 and _portfolio_peak > 0:
                _port_dd = (
                    (_portfolio_peak - portfolio_value) / _portfolio_peak)
                _dd_ema = 0.9 * _dd_ema + 0.1 * _port_dd
                dd_floor = max_dd_limit * 0.3  # start scaling at 30% of limit
                if _dd_ema >= max_dd_limit:
                    dd_scale = 0.0
                elif _dd_ema > dd_floor:
                    dd_scale = (
                        1.0 - (_dd_ema - dd_floor) / (max_dd_limit - dd_floor))
                else:
                    dd_scale = 1.0
                if dd_scale < 1.0:
                    current_weights *= dd_scale
                    cash_weight = 1.0 - max(current_weights.sum(), 0.0)
                    dynamics_log.append({
                        "date": str(date),
                        "type": "drawdown_scaling",
                        "raw_dd": float(_port_dd),
                        "ema_dd": float(_dd_ema),
                        "dd_scale": float(dd_scale),
                    })

            equity.append({
                "date": date,
                "value": portfolio_value,
                "weights": current_weights.copy(),
            })

            # Agent learning store P&L tracking
            if llm_orchestrator is not None and agent_config.learning_enabled:
                llm_orchestrator.learning_store.update_pnl(
                    date, portfolio_value)

            # SPY benchmark tracking
            if spy_prices is not None:
                if spy_initial_price is None:
                    spy_idx = spy_prices.index.get_indexer([dates[lookback]], method="nearest")[0]
                    spy_initial_price = float(spy_prices.iloc[spy_idx])
                try:
                    spy_val = float(spy_prices.asof(date))
                    spy_equity.append({
                        "date": date,
                        "value": capital * spy_val / spy_initial_price,
                    })
                except Exception:
                    pass

            # ===== STAGE 2: SIGNAL DETECTION (data up to day_idx) =====
            _effective_rebal_freq = rebal_freq
            if getattr(dynamics_config, 'cdr_enabled', False):
                _effective_rebal_freq = getattr(
                    dynamics_config, 'cdr_rebalance_freq', rebal_freq)
            is_scheduled = (days_since_start % _effective_rebal_freq == 0)
            is_triggered = False
            should_exit = False
            L_current = 0.0
            kappa_current = 1.0
            D_k_current = None  # Track latest dislocation for exit check

            if use_dynamics and signal_gen is not None:
                # Daily CUSUM update using yesterday's data
                if (factor_tracker.loadings is not None and
                        vol_aligned is not None and
                        rolling_mean_vol is not None and
                        day_idx > 0):
                    try:
                        B = factor_tracker.loadings
                        prev_idx = day_idx - 1
                        prev_ret = returns_arr[prev_idx]
                        prev_vol = vol_aligned.iloc[prev_idx].values
                        mean_vol = rolling_mean_vol.iloc[prev_idx].values

                        if not np.any(np.isnan(mean_vol)):
                            # OFI-based D-hat path (proper volume decomposition)
                            use_ofi = (volume_decomposer is not None
                                       and high_aligned is not None
                                       and low_aligned is not None
                                       and open_aligned is not None)
                            use_v3 = (not use_ofi
                                      and getattr(dynamics_config, 'use_directional_volume', False)
                                      and high_aligned is not None
                                      and low_aligned is not None)
                            use_v2 = (not use_ofi and not use_v3
                                      and open_aligned is not None
                                      and dynamics_config.volume_decomposition)

                            if use_ofi:
                                prev_close = prices.iloc[prev_idx].values
                                prev_open = open_aligned.iloc[prev_idx].values
                                prev_high = high_aligned.iloc[prev_idx].values
                                prev_low = low_aligned.iloc[prev_idx].values
                                vol_decomp = volume_decomposer.update_day(
                                    prev_close, prev_open, prev_high,
                                    prev_low, prev_vol)
                                D_k = signal_gen.compute_dhat_from_ofi(
                                    vol_decomp['ofi'],
                                    vol_decomp['impact_fraction'], B)
                                dhat_smoothed = signal_gen.smooth_dhat(
                                    D_k, dhat_smoothed,
                                    span=dynamics_config.dhat_ema_span)
                            elif use_v3:
                                prev_close = prices.iloc[prev_idx].values
                                prev_open = open_aligned.iloc[prev_idx].values
                                prev_high = high_aligned.iloc[prev_idx].values
                                prev_low = low_aligned.iloc[prev_idx].values
                                vol_comp = signal_gen.decompose_volume_v3(
                                    prev_close, prev_open, prev_high,
                                    prev_low, prev_vol, mean_vol)
                                D_k = signal_gen.compute_volume_dislocation_v3(
                                    vol_comp, B)
                            elif use_v2:
                                prev_close = prices.iloc[prev_idx].values
                                prev_open = open_aligned.iloc[prev_idx].values
                                mom = (momentum_series.iloc[prev_idx].values
                                       if momentum_series is not None
                                       else np.zeros(n_assets))
                                spy_vol = 1.0
                                spy_avg = 1.0
                                vol_comp = signal_gen.decompose_volume(
                                    prev_close, prev_open, prev_vol,
                                    mean_vol, spy_vol, spy_avg, mom)
                                D_k = signal_gen.compute_volume_dislocation_v2(
                                    vol_comp, B)
                            else:
                                D_k = signal_gen.compute_volume_dislocation(
                                    prev_ret, prev_vol, B, mean_vol)

                            # Optional Kalman daily update
                            if getattr(dynamics_config, 'use_kalman_dfm', False):
                                factor_tracker.kalman_update(prev_ret)

                            # Information-time weighting
                            info_weight = 1.0
                            if getattr(dynamics_config, 'information_time_cusum', False):
                                safe_mean = np.maximum(mean_vol, 1e-8)
                                vol_ratio = prev_vol / safe_mean
                                valid = np.isfinite(vol_ratio)
                                if valid.any():
                                    info_weight = float(np.mean(vol_ratio[valid]))

                            # Kappa-adaptive CUSUM threshold: tighter during
                            # mean-reversion (more signal), looser during momentum
                            if kappa_current < 0.8:
                                transition_detector.threshold = dynamics_config.cusum_threshold
                            elif kappa_current > 1.2:
                                transition_detector.threshold = dynamics_config.cusum_threshold * 1.5
                            else:
                                transition_detector.threshold = dynamics_config.cusum_threshold * 1.25

                            # Persist for manifold dynamics dt scaling
                            self._last_info_weight = info_weight

                            update_result = transition_detector.update(
                                D_k, information_weight=info_weight)
                            D_k_current = D_k
                            dk_series.append({
                                "date": date,
                                "D_k": D_k.copy(),
                                "D_standardized": update_result.get("D_standardized"),
                                "phases": update_result.get("phases"),
                                "directions": update_result.get("directions"),
                                "cusum": update_result.get("cusum"),
                            })
                            if update_result["events"]:
                                new_phases = [ev[0] for ev in update_result["events"]]
                                dynamics_log.append({
                                    "date": str(date),
                                    "type": "phase_transition",
                                    "events": update_result["events"],
                                    "new_phase": new_phases[0] if len(new_phases) == 1 else ",".join(new_phases),
                                    "cusum": [float(c) for c in update_result["cusum"]],
                                })
                            # === AGENT: Strategy 2 — CUSUM Validation ===
                            if (llm_orchestrator is not None
                                    and agent_config.cusum_validator_enabled
                                    and update_result["events"]):
                                try:
                                    dislocation_evts = [
                                        ev for ev in update_result["events"]
                                        if ev[0] == "dislocation"
                                    ]
                                    for _ev in dislocation_evts:
                                        _fidx = _ev[1]
                                        _cusum_state = {
                                            "D_k": D_k.copy(),
                                            "phases": update_result.get("phases", []),
                                            "factor_loadings": B,
                                            "tickers": tickers,
                                            "returns_recent": returns_arr[max(0, day_idx - 5):day_idx],
                                            "volume_recent": (
                                                vol_aligned.iloc[max(0, prev_idx - 4):prev_idx + 1].values
                                                if vol_aligned is not None else None),
                                            "volume_baseline": mean_vol if 'mean_vol' in dir() else None,
                                            "dislocated_factor_idx": _fidx,
                                            "cusum_pos": transition_detector._cusum_pos.copy(),
                                            "cusum_neg": transition_detector._cusum_neg.copy(),
                                        }
                                        _cv_result = llm_orchestrator.validate_cusum(_cusum_state)
                                        if _cv_result["type"] == "signal":
                                            D_k[_fidx] *= agent_config.cusum_signal_amplify
                                        elif _cv_result["type"] == "noise":
                                            D_k[_fidx] *= agent_config.cusum_noise_dampen
                                        elif _cv_result["type"] == "contrarian":
                                            D_k[_fidx] *= -1.0
                                        dhat_smoothed = signal_gen.smooth_dhat(
                                            D_k, dhat_smoothed,
                                            span=dynamics_config.dhat_ema_span)
                                        D_k_current = D_k
                                        dynamics_log.append({
                                            "date": str(date),
                                            "type": "agent_cusum_validation",
                                            "factor_idx": _fidx,
                                            "validation": _cv_result,
                                        })
                                        self._agent_cusum_today = _cv_result
                                except Exception as _cv_e:
                                    logger.warning("Agent CUSUM validation failed: %s", _cv_e)

                    except Exception as e:
                        logger.warning("Daily CUSUM update failed on %s: %s", date, e)

                # === STAGE 2.4: FIBER BUNDLE UPDATE ===
                if fiber_bundle is not None and volume_decomposer is not None:
                    try:
                        B_fb = factor_tracker.loadings if factor_tracker else None
                        K_fb = B_fb.shape[1] if B_fb is not None else 0
                        if K_fb > 0 and day_idx > 0:
                            factor_rets_fb = returns_arr[day_idx - 1] @ B_fb

                            if K_fb != fiber_bundle.current_K:
                                vol_decomp_fb = vol_decomp if 'vol_decomp' in dir() else {
                                    "v_trade": vol_aligned.iloc[day_idx - 1].values if vol_aligned is not None else np.zeros(n_assets),
                                    "v_pos": np.zeros(n_assets),
                                    "v_mkt": 0.0,
                                }
                                fiber_bundle.transition_stratum(K_fb, vol_decomp_fb)

                            vol_data_fb = vol_decomp if 'vol_decomp' in dir() else {
                                "v_trade": vol_aligned.iloc[day_idx - 1].values if vol_aligned is not None else np.zeros(n_assets),
                                "v_pos": np.zeros(n_assets),
                                "v_mkt": 0.0,
                            }

                            eigs_fb = (factor_tracker._current_eigenvalues[:K_fb]
                                       if hasattr(factor_tracker, '_current_eigenvalues')
                                       and factor_tracker._current_eigenvalues is not None
                                       else None)

                            bundle_result = fiber_bundle.update(
                                dt=1.0,
                                factor_returns=factor_rets_fb,
                                volume_data=vol_data_fb,
                                realized_variances=eigs_fb,
                                information_times=_current_tau_k,
                            )

                            if getattr(dynamics_config, 'bundle_trajectory_log', False):
                                dynamics_log.append({
                                    "date": str(date),
                                    "type": "fiber_bundle",
                                    "tau_global": bundle_result["tau_global"],
                                    "phases": bundle_result["phases"],
                                    "proper_time_rates": [float(r) for r in bundle_result["proper_time_rates"]],
                                    "state_dim": len(bundle_result["state_vector"]),
                                })

                            crossings = fiber_bundle.bundle.detect_phase_transition()
                            for phase_name, fidx, info in crossings:
                                dynamics_log.append({
                                    "date": str(date),
                                    "type": "bundle_phase_crossing",
                                    "phase": phase_name,
                                    "factor_idx": fidx,
                                    "crossing_speed": info["crossing_speed"],
                                    "tau_n": info["tau_n_at_crossing"],
                                    "tau_global": info["tau_global_at_crossing"],
                                })
                    except Exception as e:
                        logger.warning("Fiber bundle update failed on %s: %s", date, e)

                # === STAGE 2.45: DAILY MANIFOLD EVOLUTION ===
                # Evolves correlation geometry every day (not just at rebalance)
                if self._manifold_evolution is not None and D_k_current is not None:
                    try:
                        K_m = factor_tracker.n_active_factors
                        factor_cov_K = factor_tracker.factor_covariance
                        if factor_cov_K is None:
                            eigs = factor_tracker._current_eigenvalues
                            if eigs is not None:
                                factor_cov_K = np.diag(eigs[:K_m])
                        if factor_cov_K is not None:
                            B_m = factor_tracker.loadings
                            # Factor returns for backreaction
                            z_daily = (factor_tracker.project_returns(
                                returns_arr[day_idx - 1:day_idx]).flatten()[:K_m]
                                if day_idx > 0 else np.zeros(K_m))
                            D_k_m = D_k_current[:K_m] if len(D_k_current) >= K_m else np.zeros(K_m)
                            phases_m = (transition_detector._phase[:K_m]
                                        if transition_detector else ["normal"] * K_m)
                            # Soft phases (daily)
                            soft_probs_daily = None
                            if self._soft_phase_estimator is not None:
                                rbpf_probs = None
                                if (factor_tracker._rbpf is not None
                                        and factor_tracker._rbpf._initialized):
                                    rbpf_probs = factor_tracker._rbpf.regime_probabilities
                                soft_probs_daily = self._soft_phase_estimator.compute_soft_phases(
                                    cusum_pos=transition_detector._cusum_pos[:K_m],
                                    cusum_neg=transition_detector._cusum_neg[:K_m],
                                    threshold=dynamics_config.cusum_threshold,
                                    phase_day_counters=transition_detector._phase_day_counter[:K_m],
                                    dislocation_decay_days=dynamics_config.dislocation_decay_days,
                                    convergence_trigger_days=dynamics_config.convergence_trigger_days,
                                    rbpf_regime_probs=rbpf_probs,
                                )
                                # Persist for rebalance use
                                self._last_soft_phases = soft_probs_daily

                            # Information time: scale dt by vol_ratio
                            dt_daily = 1.0
                            if getattr(dynamics_config, 'information_time_cusum', False):
                                dt_daily = float(np.clip(
                                    getattr(self, '_last_info_weight', 1.0), 0.1, 5.0))

                            # Initialize or evolve
                            if self._manifold_state is None:
                                self._manifold_state = self._manifold_evolution.initialize_from_pca(
                                    factor_cov_K, current_weights, D_k_m, phases_m, z_daily)
                            else:
                                self._manifold_state, _m_diag = self._manifold_evolution.evolve(
                                    state=self._manifold_state,
                                    sigma_observed=factor_cov_K,
                                    sigma_equilibrium=None,
                                    factor_returns=z_daily,
                                    D_k=D_k_m,
                                    weight_delta=np.zeros(n_assets),
                                    B=B_m,
                                    phase_probs=soft_probs_daily,
                                    dt=dt_daily,
                                )
                                # Log every 5 days to avoid bloat
                                if day_idx % 5 == 0:
                                    dynamics_log.append({
                                        "date": str(date),
                                        "type": "correlation_manifold",
                                        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                           for k, v in _m_diag.items()
                                           if not isinstance(v, np.ndarray)},
                                    })
                    except Exception as e:
                        logger.warning("Daily manifold evolution failed on %s: %s", date, e)

                # === STAGE 2.5: EVENT ATTRIBUTION ===
                event_modifier = 1.0
                if (event_attributor is not None
                        and update_result is not None
                        and update_result.get("events")):
                    disloc_events = [
                        e for e in update_result["events"]
                        if e[0] == "dislocation"
                    ]
                    if disloc_events and factor_tracker.loadings is not None:
                        try:
                            attributions = event_attributor.attribute(
                                transition_result=update_result,
                                B=factor_tracker.loadings,
                                tickers=tickers,
                                date=date,
                                D_k=D_k_current,
                            )
                            event_modifier = (
                                event_attributor.get_aggregate_modifier(
                                    attributions))

                            for attr in attributions:
                                attribution_log.append(attr.to_dict())
                                dynamics_log.append({
                                    "date": str(date),
                                    "type": "event_attribution",
                                    "event_type": attr.event_type,
                                    "confidence": attr.confidence,
                                    "modifier": attr.signal_modifier,
                                    "factor_idx": attr.factor_idx,
                                    "top_assets": [
                                        a["ticker"]
                                        for a in attr.contributing_assets[:3]
                                    ],
                                })

                                if event_attribution_config.store_results:
                                    try:
                                        event_attributor.store_attribution(
                                            attr)
                                    except Exception:
                                        pass
                        except Exception as e:
                            logger.warning(
                                "Event attribution failed on %s: %s",
                                date, e)

                # Factor loadings for L and kappa (use most recent fit)
                B_current = factor_tracker.loadings  # N×K or None

                # Compute contraction constant L (on factor returns, not raw)
                if dynamics_config.contraction_enabled:
                    ac_window = dynamics_config.autocorr_window
                    ac_start = max(0, day_idx - ac_window)
                    if day_idx - ac_start >= 20:
                        ac_returns = returns_arr[ac_start:day_idx]
                        # Hierarchy: manifold > Kalman F > OLS VAR > spectral radius
                        if getattr(dynamics_config, 'use_manifold_L', False):
                            factor_cov = factor_tracker.factor_covariance
                            if factor_cov is not None:
                                cov_history.append(factor_cov)
                                if len(cov_history) > 3:
                                    L_current = signal_gen.compute_contraction_constant_manifold(
                                        cov_history[-ac_window:])
                            else:
                                # Fall through to next method
                                F_mat = factor_tracker.transition_matrix
                                if F_mat is not None:
                                    L_current = float(np.clip(
                                        np.max(np.abs(np.linalg.eigvals(F_mat))),
                                        0.0, 0.99))
                        else:
                            F_mat = factor_tracker.transition_matrix
                            if (getattr(dynamics_config, 'use_kalman_dfm', False)
                                    and F_mat is not None):
                                L_current = float(np.clip(
                                    np.max(np.abs(np.linalg.eigvals(F_mat))),
                                    0.0, 0.99))
                            elif getattr(dynamics_config, 'use_ols_contraction', False):
                                L_current = signal_gen.compute_contraction_constant_ols(
                                    ac_returns, B=B_current)
                            else:
                                L_current = signal_gen.compute_contraction_constant(
                                    ac_returns, B=B_current)
                        L_history.append({"date": date, "L": L_current})

                # Compute reflexivity kappa (on factor returns, not raw)
                if dynamics_config.reflexivity_enabled:
                    k_window = dynamics_config.kappa_momentum_window
                    k_start = max(0, day_idx - k_window)
                    if day_idx - k_start >= 20:
                        k_returns = returns_arr[k_start:day_idx]
                        if getattr(dynamics_config, 'use_regression_kappa', False):
                            kappa_current = signal_gen.compute_reflexivity_kappa_regression(
                                k_returns, dynamics_config.kappa_autocorr_lag,
                                B=B_current)
                        else:
                            kappa_current = signal_gen.compute_reflexivity_kappa(
                                k_returns, dynamics_config.kappa_autocorr_lag,
                                B=B_current)
                        kappa_history.append({"date": date, "kappa": kappa_current})

                    # Sentiment-kappa blending
                    if (sentiment_engine is not None
                            and getattr(dynamics_config, 'sentiment_kappa_blend', False)):
                        try:
                            sent = sentiment_engine.compute(tickers=tickers)
                            s = sent["directional_score"]
                            scale = getattr(dynamics_config, 'sentiment_kappa_scale', 0.3)
                            kappa_current = kappa_current * (1.0 + scale * s)
                            kappa_current = np.clip(kappa_current, 0.1, 10.0)
                        except Exception:
                            pass

                # === AGENT: Strategy 3 — Macro Reflexivity kappa override ===
                if (llm_orchestrator is not None
                        and agent_config.macro_reflexivity_enabled):
                    try:
                        _macro_state = {
                            "kappa_current": kappa_current,
                            "vix_current": (
                                float(vix_series.iloc[day_idx])
                                if vix_series is not None and day_idx < len(vix_series)
                                else None),
                            "vix_history": (
                                vix_series.iloc[max(0, day_idx - 20):day_idx + 1].values
                                if vix_series is not None else None),
                            "credit_spread": (
                                float(credit_spread_series.iloc[day_idx])
                                if credit_spread_series is not None
                                and day_idx < len(credit_spread_series) else None),
                            "credit_spread_history": (
                                credit_spread_series.iloc[
                                    max(0, day_idx - 20):day_idx + 1].values
                                if credit_spread_series is not None else None),
                            "yield_spread": (
                                float(yield_spread_series.iloc[day_idx])
                                if yield_spread_series is not None
                                and day_idx < len(yield_spread_series) else None),
                            "regime": regime if 'regime' in dir() else None,
                        }
                        _macro_result = llm_orchestrator.assess_macro(_macro_state)
                        if _macro_result.get("kappa_override") is not None:
                            _ko = _macro_result["kappa_override"]
                            _max_adj = agent_config.max_kappa_adjustment
                            kappa_current = float(np.clip(
                                _ko,
                                kappa_current - _max_adj,
                                kappa_current + _max_adj))
                            kappa_current = float(np.clip(kappa_current, 0.1, 10.0))
                        self._agent_macro_today = _macro_result
                    except Exception as _me:
                        logger.warning("Agent macro reflexivity failed: %s", _me)

                # Compute Hurst exponent for signal modulation
                hurst_current = 0.5  # neutral default
                if (getattr(dynamics_config, 'use_hurst_exponent', False)
                        and signal_gen is not None):
                    h_window = dynamics_config.autocorr_window
                    h_start = max(0, day_idx - h_window)
                    if day_idx - h_start >= 20:
                        h_returns = returns_arr[h_start:day_idx]
                        B_h = factor_tracker.loadings if factor_tracker else None
                        try:
                            hurst_current = signal_gen.compute_hurst_exponent(
                                h_returns, B=B_h,
                                min_lag=dynamics_config.hurst_min_lag,
                                max_lag=dynamics_config.hurst_max_lag)
                        except Exception:
                            hurst_current = 0.5

                # Apply Hurst modulation to D_k
                if (getattr(dynamics_config, 'use_hurst_exponent', False)
                        and D_k_current is not None
                        and signal_gen is not None):
                    try:
                        D_k_current = signal_gen.modulate_signal_by_hurst(
                            D_k_current, hurst_current)
                    except Exception:
                        pass

                # Exit conditions (only trigger once, not every day)
                if not is_exited:
                    if dynamics_config.exit_on_max_L and L_current > dynamics_config.max_L:
                        should_exit = True
                    if dynamics_config.exit_on_phase3:
                        if transition_detector.get_aggregate_phase() == "convergence":
                            should_exit = True
                    # Exit when dislocation collapses (D_k → 0)
                    if (dynamics_config.exit_on_dislocation_collapse
                            and D_k_current is not None):
                        if np.linalg.norm(D_k_current) < dynamics_config.dislocation_collapse_threshold:
                            agg_phase = transition_detector.get_aggregate_phase()
                            if agg_phase in ("discovery", "convergence"):
                                should_exit = True

                # CUSUM-triggered rebalance
                # Sentiment-modulated CUSUM threshold
                if (sentiment_engine is not None
                        and transition_detector is not None
                        and getattr(dynamics_config, 'sentiment_cusum_modulation', False)):
                    try:
                        sent = sentiment_engine.compute(tickers=tickers)
                        s = sent["directional_score"]  # -1 to +1
                        scale = getattr(dynamics_config, 'sentiment_cusum_scale', 0.3)
                        base_threshold = dynamics_config.cusum_threshold
                        # Fear → lower threshold (easier trigger), Greed → higher
                        transition_detector.threshold = base_threshold * (1.0 - scale * s)
                    except Exception:
                        pass
                if (not is_scheduled and
                        (dynamics_config.max_extra_rebalances < 0 or
                         extra_rebalance_count < dynamics_config.max_extra_rebalances)):
                    if getattr(dynamics_config, 'use_optimal_stopping_rebalance', False):
                        cusum_mag = float(np.mean(transition_detector._cusum))
                        is_triggered = transition_detector.should_rebalance_optimal(
                            current_weights=current_weights,
                            target_weights=current_weights,
                            transaction_cost_bps=self.config.proportional_cost_bps,
                            signal_strength=cusum_mag,
                            min_interval=dynamics_config.min_rebalance_interval,
                        )
                    else:
                        is_triggered = transition_detector.should_rebalance(
                            dynamics_config.min_rebalance_interval)

            # ===== STAGE 2.9: POSITION LIFECYCLE UPDATE =====
            _hf_derisk_score = 0.0
            _crowding_scores = None
            if (use_lifecycle and position_manager is not None
                    and days_since_start >= position_manager_config.burn_in_days):
                try:
                    # Compute daily alpha scores for lifecycle
                    # D-hat component
                    _dhat_daily_z = None
                    if (dhat_smoothed is not None
                            and factor_tracker is not None
                            and factor_tracker.loadings is not None):
                        from NR.dynamics.alpha_signals import (
                            CrossSectionalAlpha)
                        B_lc = factor_tracker.loadings
                        K_lc = min(len(dhat_smoothed), B_lc.shape[1])
                        phases_lc = (transition_detector._phase
                                     if transition_detector is not None
                                     else [])
                        if K_lc > 0:
                            # Use raw D_k without phase weighting — phase
                            # weights are z-score invariant and weakening
                            # the signal hurts portfolio beta. Phase info
                            # is handled by the entry model instead.
                            _raw = B_lc[:, :K_lc] @ dhat_smoothed[:K_lc]
                            _raw *= max(1.0 - L_current, 0.0)
                            _dhat_daily_z = (
                                CrossSectionalAlpha._cross_sectional_zscore(
                                    _raw))

                    if _dhat_daily_z is not None:
                        daily_alpha_scores = _dhat_daily_z
                    else:
                        daily_alpha_scores = np.zeros(n_assets)

                    # Gradient flow alpha (daily path): replaces D-hat
                    if (getattr(dynamics_config, 'gradient_flow_enabled', False)
                            and dhat_smoothed is not None
                            and factor_tracker is not None
                            and factor_tracker.loadings is not None):
                        try:
                            B_gf = factor_tracker.loadings
                            K_gf = min(len(dhat_smoothed), B_gf.shape[1])
                            if K_gf > 0:
                                gf_raw = signal_gen.compute_gradient_flow_alpha(
                                    dhat_smoothed[:K_gf], B_gf[:, :K_gf],
                                    L_current, kappa_current,
                                    momentum_blend_cap=dynamics_config.gradient_flow_momentum_blend_cap)
                                from NR.dynamics.alpha_signals import CrossSectionalAlpha
                                gf_z = CrossSectionalAlpha._cross_sectional_zscore(gf_raw)
                                gf_z *= _curvature_gate
                                daily_alpha_scores = daily_alpha_scores + 0.5 * gf_z
                        except Exception as e:
                            logger.warning("Daily gradient flow failed: %s", e)

                    # Compute fresh propagation mispricing daily (not just rebalance)
                    if (getattr(dynamics_config, 'propagation_mispricing_enabled', False)
                            and hasattr(self, '_last_causal_M')
                            and self._last_causal_M is not None
                            and day_idx >= 2):
                        try:
                            _causal_M_daily = self._last_causal_M
                            prev_returns_daily = returns_arr[day_idx - 1]
                            today_returns_daily = returns_arr[day_idx]
                            predicted_daily = _causal_M_daily @ prev_returns_daily
                            mispricing_raw_daily = predicted_daily - today_returns_daily
                            _, _prop_cumul_daily = signal_gen.compute_propagation_mispricing_signal(
                                mispricing_raw_daily, _mispricing_history,
                                cumulative_window=dynamics_config.propagation_mispricing_cumulative_window)
                            _propagation_mispricing_scores = _prop_cumul_daily
                        except Exception as e:
                            logger.warning("Daily mispricing computation failed: %s", e)

                    # Propagation mispricing additive alpha (daily path)
                    if (getattr(dynamics_config, 'propagation_mispricing_enabled', False)
                            and _propagation_mispricing_scores is not None):
                        try:
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            mis_z = CrossSectionalAlpha._cross_sectional_zscore(
                                _propagation_mispricing_scores)
                            daily_alpha_scores = (daily_alpha_scores
                                + dynamics_config.propagation_mispricing_alpha_weight * mis_z)
                        except Exception as e:
                            logger.warning("Daily mispricing alpha failed: %s", e)

                    # Latency alpha (daily path): use persisted τ_k from rebalance
                    if (_latency_alpha_z is not None
                            and getattr(dynamics_config, 'information_time_alpha_enabled', False)):
                        it_weight = dynamics_config.information_time_alpha_weight
                        daily_alpha_scores = daily_alpha_scores + it_weight * _latency_alpha_z

                    # Ξ Contrarian daily alpha (lifecycle path)
                    if (getattr(dynamics_config, 'xi_contrarian_enabled', False)
                            and dhat_smoothed is not None
                            and factor_tracker is not None
                            and factor_tracker.loadings is not None
                            and day_idx >= 5):
                        try:
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            B_xi = factor_tracker.loadings
                            K_xi = min(len(dhat_smoothed), B_xi.shape[1])
                            if K_xi > 0:
                                conviction_raw = np.abs(B_xi[:, :K_xi] @ dhat_smoothed[:K_xi])
                                recent_ret = returns_arr[day_idx - 5:day_idx]
                                factor_rets_5d = np.mean(recent_ret @ B_xi[:, :K_xi], axis=0)
                                factor_direction = np.sign(factor_rets_5d) * np.abs(dhat_smoothed[:K_xi])
                                flow_momentum = B_xi[:, :K_xi] @ factor_direction
                                N_xi = len(flow_momentum)
                                align_rank = np.argsort(np.argsort(flow_momentum)) / max(N_xi - 1, 1)
                                xi_contrarian = conviction_raw * (1.5 - align_rank)
                                xi_z = CrossSectionalAlpha._cross_sectional_zscore(xi_contrarian)
                                # EMA smoothing of xi_z signal
                                xi_ema = getattr(dynamics_config, 'xi_signal_ema_alpha', 0.6)
                                if _xi_z_smoothed is not None and len(_xi_z_smoothed) == len(xi_z):
                                    xi_z = xi_ema * xi_z + (1.0 - xi_ema) * _xi_z_smoothed
                                _xi_z_smoothed = xi_z.copy()
                                xi_tilt = getattr(dynamics_config, 'xi_contrarian_tilt', 0.5)
                                # Phase-dependent scaling
                                xi_phase_scales = getattr(dynamics_config, 'xi_phase_scales', None)
                                if xi_phase_scales and transition_detector is not None:
                                    priority = {"dislocation": 3, "convergence": 2,
                                                "discovery": 1, "normal": 0}
                                    dominant = max(transition_detector._phase,
                                                   key=lambda p: priority.get(p, 0))
                                    phase_scale = xi_phase_scales.get(dominant, 0.0)
                                    xi_tilt = xi_tilt * phase_scale
                                if xi_tilt > 1e-8:
                                    daily_alpha_scores = xi_tilt * xi_z + (1.0 - xi_tilt) * daily_alpha_scores
                        except Exception as e:
                            logger.warning("Daily xi contrarian failed: %s", e)

                    # Trailing volatility for sizer (~63 days scaled to bars)
                    _vf_lb = self.data_config.scale_window(63)
                    _vf_start = max(0, day_idx - _vf_lb)
                    _trailing_vol = (
                        np.std(returns_arr[_vf_start:day_idx], axis=0, ddof=1)
                        * np.sqrt(self.bars_per_year))

                    # Compute PageRank and IRF from causal graph (persists from last fit)
                    _pr_scores = None
                    _irf_scores = None
                    if causal_graph is not None:
                        try:
                            _pr_scores = causal_graph.pagerank()
                            _irf_scores = causal_graph.impulse_response(horizon=5)
                        except Exception:
                            pass

                    # Compute insider signal from Form 4 data
                    _insider_scores = None
                    if (getattr(dynamics_config, 'use_insider_signal', False)
                            and _insider_signal is not None):
                        try:
                            _ins = _insider_signal.compute(tickers, date=date)
                            if _ins.get("available", False):
                                _insider_scores = _ins["scores"]
                        except Exception:
                            pass

                    # Compute institutional consensus signal
                    _inst_consensus_scores = None
                    if _institutional_consensus is not None:
                        try:
                            _inst_consensus_scores = _institutional_consensus.compute(
                                tickers=tickers,
                                all_trades=_all_inst_trades,
                                as_of_date=str(date)[:10],
                            )
                        except Exception:
                            pass

                    # Compute propagation speed signal
                    _propagation_scores = None
                    if _propagation_signal is not None:
                        try:
                            _propagation_scores = _propagation_signal.compute(
                                tickers=tickers,
                                as_of_date=str(date)[:10],
                            )
                        except Exception:
                            pass

                    # Compute HF crowding signal
                    _crowding_scores = None
                    _hf_derisk_score = 0.0
                    if _hf_crowding_signal is not None:
                        try:
                            _crowding_scores = _hf_crowding_signal.compute(
                                tickers=tickers,
                                as_of_date=str(date)[:10],
                            )
                            _hf_derisk_score = _hf_crowding_signal.aggregate_derisk_score(
                                tickers=tickers,
                                as_of_date=str(date)[:10],
                            )
                        except Exception:
                            pass

                    # Sentiment-phase validation multiplier
                    _sent_val_mult = 1.0
                    if (_sentiment_phase_validator is not None
                            and sentiment_engine is not None):
                        try:
                            _sv_sent = sentiment_engine.compute(tickers=tickers)
                            _sv_phases = (transition_detector._phase
                                          if transition_detector is not None
                                          else [])
                            _sent_val_mult = _sentiment_phase_validator.validate(
                                _sv_sent["directional_score"], _sv_phases)
                        except Exception:
                            _sent_val_mult = 1.0

                    # ATR for stop loss scaling
                    _atr_values = None
                    if (high_aligned is not None and low_aligned is not None
                            and position_manager_config is not None):
                        atr_lb = getattr(position_manager_config, 'atr_lookback', 14)
                        atr_mult = getattr(position_manager_config, 'atr_stop_multiplier', 0.0)
                        if atr_mult > 0 and day_idx >= atr_lb:
                            _atr_values = np.zeros(n_assets)
                            for j in range(n_assets):
                                h_vals = high_aligned.iloc[day_idx - atr_lb:day_idx, j].values
                                l_vals = low_aligned.iloc[day_idx - atr_lb:day_idx, j].values
                                c_vals = prices_arr[day_idx - atr_lb:day_idx, j]
                                if len(h_vals) >= 2 and len(c_vals) >= 2:
                                    prev_c = np.concatenate([[c_vals[0]], c_vals[:-1]])
                                    tr = np.maximum(
                                        h_vals - l_vals,
                                        np.maximum(
                                            np.abs(h_vals - prev_c),
                                            np.abs(l_vals - prev_c)))
                                    _atr_values[j] = float(np.mean(tr))

                    # Compute entry-path predictive signals from persisted causal_M
                    if (hasattr(self, '_last_causal_M')
                            and self._last_causal_M is not None):
                        try:
                            _causal_M = self._last_causal_M
                            M_abs = np.abs(_causal_M)
                            _sia_outgoing_scores = M_abs.sum(axis=1)
                            if day_idx >= 5:
                                incoming = M_abs.sum(axis=0)
                                asymmetry = _sia_outgoing_scores / np.maximum(incoming, 1e-12)
                                recent_5d = returns_arr[day_idx - 5:day_idx]
                                mean_5d = np.mean(recent_5d, axis=0)
                                abs_move = np.abs(mean_5d)
                                ms = np.std(abs_move)
                                if ms > 1e-12:
                                    mz = np.abs((abs_move - np.mean(abs_move)) / ms)
                                else:
                                    mz = np.zeros_like(abs_move)
                                _latent_shock_scores = asymmetry * np.maximum(2.0 - mz, 0.0)
                        except Exception as e:
                            logger.warning("Latent/SIA signals failed: %s", e)

                    # Lead-lag catch-up signal (daily, uses persisted M)
                    if (_lead_lag_signal is not None
                            and hasattr(self, '_last_causal_M')
                            and self._last_causal_M is not None):
                        try:
                            _lead_lag_scores = _lead_lag_signal.compute(
                                M=self._last_causal_M,
                                returns_buffer=returns_arr,
                                day_idx=day_idx,
                            )
                        except Exception as e:
                            logger.warning("Lead-lag signal failed: %s", e)

                    # === AGENT: Strategy 4 — Exit Watchdog ===
                    if (llm_orchestrator is not None
                            and agent_config.exit_watchdog_enabled
                            and position_manager is not None):
                        try:
                            position_manager.tick_overrides()
                            _near = position_manager.get_positions_near_stop(
                                prices_arr[day_idx], threshold_pct=0.5)
                            if _near:
                                _exit_state = {
                                    "positions_near_stop": _near,
                                    "returns_today": (
                                        returns_arr[day_idx]
                                        if day_idx > 0 else np.zeros(n_assets)),
                                    "factor_loadings": (
                                        factor_tracker.loadings
                                        if factor_tracker is not None else None),
                                    "tickers": tickers,
                                    "volume_today": (
                                        vol_aligned.iloc[day_idx - 1].values
                                        if vol_aligned is not None
                                        and day_idx > 0 else None),
                                    "volume_baseline": (
                                        mean_vol if 'mean_vol' in dir()
                                        else None),
                                }
                                _ew_results = llm_orchestrator.assess_exits(
                                    _exit_state)
                                for _er in _ew_results:
                                    _min_conf = agent_config.exit_min_confidence
                                    if (_er["action"] == "hold"
                                            and _er["confidence"] >= _min_conf):
                                        position_manager.set_exit_override(
                                            _er["ticker_idx"], "hold",
                                            duration=1)
                                    elif _er["action"] == "widen_stop":
                                        position_manager.adjust_stop(
                                            _er["ticker_idx"],
                                            _er.get("stop_adjustment", 1.3))
                                if _ew_results:
                                    self._agent_exit_today = _ew_results
                        except Exception as _ew_e:
                            logger.warning(
                                "Agent exit watchdog failed: %s", _ew_e)

                    lifecycle_result = position_manager.update_daily(
                        date=date,
                        prices_today=prices_arr[day_idx],
                        D_k=D_k_current,
                        L=L_current,
                        kappa=kappa_current,
                        phases=(transition_detector._phase
                                if transition_detector is not None
                                else []),
                        factor_loadings=(factor_tracker.loadings
                                         if factor_tracker is not None
                                         else None),
                        alpha_scores=daily_alpha_scores,
                        causal_graph=(causal_graph.graph
                                      if causal_graph is not None
                                      and hasattr(causal_graph, 'graph')
                                      else None),
                        regime=regime if 'regime' in dir() else None,
                        trailing_vol=_trailing_vol,
                        day_idx=day_idx,
                        prices_df=prices,
                        high_prices_df=high_aligned,
                        low_prices_df=low_aligned,
                        pagerank_scores=_pr_scores,
                        irf_scores=_irf_scores,
                        insider_scores=_insider_scores,
                        institutional_consensus_scores=_inst_consensus_scores,
                        propagation_scores=_propagation_scores,
                        atr_values=_atr_values,
                        sentiment_validator_multiplier=_sent_val_mult,
                        rmw_betas=rmw_betas,
                        propagation_mispricing_scores=_propagation_mispricing_scores,
                        latent_shock_scores=_latent_shock_scores,
                        sia_outgoing_scores=_sia_outgoing_scores,
                        lead_lag_scores=_lead_lag_scores,
                        crowding_scores=_crowding_scores,
                    )

                    if lifecycle_result["transitions"]:
                        dynamics_log.append({
                            "date": str(date),
                            "type": "lifecycle_transitions",
                            "transitions": lifecycle_result["transitions"],
                            "active_count": int(
                                lifecycle_result["active_mask"].sum()),
                            "cash_fraction": float(
                                1.0 - lifecycle_result[
                                    "target_sizes"].sum()),
                        })

                        # Lifecycle transaction costs
                        if getattr(dynamics_config,
                                   'lifecycle_transaction_costs', False):
                            try:
                                lc_new_w = current_weights.copy()
                                _ts = lifecycle_result["target_sizes"]
                                for _tr in lifecycle_result["transitions"]:
                                    _tk = _tr["ticker"]
                                    _idx = tickers.index(_tk)
                                    if _tr["new_state"] == "active":
                                        if _idx < len(_ts):
                                            lc_new_w[_idx] = _ts[_idx]
                                    elif _tr["new_state"] in (
                                            "exit", "cooldown"):
                                        lc_new_w[_idx] = 0.0
                                _lc_delta = np.sum(
                                    np.abs(lc_new_w - current_weights))
                                if _lc_delta > 1e-6:
                                    _lc_cost = self._compute_rebalance_cost(
                                        current_weights, lc_new_w,
                                        portfolio_value, tickers,
                                        vol_aligned, prices, returns,
                                        day_idx)
                                    portfolio_value -= _lc_cost
                                    cost_history.append({
                                        "date": date,
                                        "cost": _lc_cost,
                                        "turnover": float(_lc_delta / 2),
                                        "type": "lifecycle",
                                    })
                            except Exception as _lc_e:
                                logger.warning(
                                    "Lifecycle cost failed on %s: %s",
                                    date, _lc_e)
                except Exception as e:
                    logger.warning(
                        "Lifecycle update failed on %s: %s", date, e)
                    lifecycle_result = None

            # ===== STAGE 3: EXECUTION =====
            if should_exit:
                # Emergency de-risk: cash or equal weight
                if self.config.exit_to_cash:
                    new_weights = np.zeros(n_assets)
                else:
                    new_weights = np.ones(n_assets) / n_assets
                cost = self._compute_rebalance_cost(
                    current_weights, new_weights, portfolio_value,
                    tickers, vol_aligned, prices, returns, day_idx)
                turnover = float(np.sum(np.abs(new_weights - current_weights)) / 2)
                portfolio_value -= cost
                if self.config.allow_short:
                    cash_weight = 1.0 - new_weights.sum()
                else:
                    cash_weight = max(0.0, 1.0 - new_weights.sum())

                weights_history.append({
                    "date": date, "weights": dict(zip(tickers, new_weights)),
                    "turnover": turnover, "cash_weight": cash_weight,
                })
                cost_history.append({"date": date, "cost": cost, "turnover": turnover})
                exit_method = "cash_exit" if self.config.exit_to_cash else "equal_exit"
                decision_log.append({
                    "date": date, "regime": "exit", "turnover": turnover,
                    "cost": cost, "method": exit_method, "triggered": True,
                    "L": L_current, "kappa": kappa_current,
                })
                current_weights = new_weights
                is_exited = True

            elif is_scheduled or is_triggered:
                # === REBALANCE (clears exit state) ===
                is_exited = False
                window_start = max(0, day_idx - lookback)
                window_returns = returns.iloc[window_start:day_idx]

                if len(window_returns) < 30:
                    continue

                # Position lifecycle: if active and no positions, go to cash
                plm_active_mask = None
                if lifecycle_result is not None:
                    plm_active_mask = lifecycle_result["active_mask"]
                    if not np.any(plm_active_mask):
                        new_weights = np.zeros(n_assets)
                        cash_weight = 1.0
                        turnover = float(
                            np.sum(np.abs(new_weights - current_weights)) / 2)
                        cost = self._compute_rebalance_cost(
                            current_weights, new_weights, portfolio_value,
                            tickers, vol_aligned, prices, returns, day_idx
                        ) if vol_aligned is not None else 0.0
                        portfolio_value -= cost
                        weights_history.append({
                            "date": date,
                            "weights": dict(zip(tickers, new_weights)),
                            "turnover": turnover,
                            "cash_weight": cash_weight,
                        })
                        cost_history.append({
                            "date": date, "cost": cost,
                            "turnover": turnover,
                        })
                        decision_log.append({
                            "date": date, "regime": "lifecycle_cash",
                            "turnover": turnover, "cost": cost,
                            "method": "lifecycle_no_active",
                            "triggered": is_triggered,
                        })
                        current_weights = new_weights
                        if is_triggered:
                            extra_rebalance_count += 1
                        day_idx += 1
                        continue

                # 1. Covariance (POFM when FF data available)
                ff_win_ev = None
                if ff_daily_factors is not None:
                    try:
                        ff_win_ev = ff_daily_factors.iloc[
                            window_start:day_idx].values
                    except Exception:
                        ff_win_ev = None
                if ff_win_ev is not None and ff_win_ev.shape[0] == len(window_returns):
                    cov = CovarianceEstimator.pofm(
                        window_returns.values, ff_win_ev)
                else:
                    cov, _ = cov_est.ledoit_wolf(window_returns.values)

                # 2. Regime adjustment
                regime = None
                if regime_detector is not None and vix_series is not None:
                    try:
                        vix_val = float(vix_series.asof(date))
                        ys_val = (float(yield_spread_series.asof(date))
                                  if yield_spread_series is not None else 0.5)
                        cs_val = (float(credit_spread_series.asof(date))
                                  if credit_spread_series is not None else None)
                        _bd_val = None
                        if kbe_series is not None:
                            try:
                                _bd_window = getattr(
                                    regime_detector.config,
                                    'regime_bank_drawdown_window', 63)
                                _kbe_w = kbe_series.loc[:date].tail(_bd_window)
                                if len(_kbe_w) >= 10:
                                    _kbe_peak = float(_kbe_w.max())
                                    if _kbe_peak > 0:
                                        _bd_val = float(
                                            (_kbe_peak - _kbe_w.iloc[-1])
                                            / _kbe_peak)
                            except Exception:
                                pass
                        regime = regime_detector.classify(
                            vix_val, ys_val, credit_spread=cs_val,
                            bank_drawdown=_bd_val)

                        # Markov regime overlay (optional)
                        if _markov_detector is not None:
                            try:
                                _mom_val = 0.0
                                if momentum_series is not None:
                                    _mom_val = float(
                                        momentum_series.iloc[day_idx].mean())
                                _cs_markov = cs_val if cs_val is not None else 4.0
                                markov_regime = _markov_detector.update(
                                    vix_val, ys_val, _mom_val, _cs_markov)
                                # Use Markov regime when it disagrees with
                                # threshold detector in crisis/recovery states
                                if markov_regime in ("crisis", "recovery"):
                                    regime = markov_regime
                            except Exception:
                                pass

                        cov = CovarianceEstimator.regime_adjusted(
                            cov, regime,
                            regime_detector.config.regime_cov_multipliers)
                    except (KeyError, TypeError):
                        pass

                # 3. Expected returns
                mu = window_returns.mean().values
                mu_baseline = mu.copy()

                if regime_detector is not None and regime is not None:
                    mu = mu + regime_detector.get_return_adjustment(regime)

                # 3b. Sentiment return and covariance adjustment
                if sentiment_engine is not None:
                    try:
                        _cs_sent = (float(credit_spread_series.asof(date))
                                    if credit_spread_series is not None else None)
                        sent = sentiment_engine.compute(
                            tickers=tickers, vix_series=vix_series,
                            credit_spread=_cs_sent,
                            kbe_series=kbe_series,
                            hf_derisk_score=_hf_derisk_score)
                        mu = mu + sent["return_adjustment"]
                        cov = cov * sent["cov_multiplier"]
                    except Exception as e:
                        logger.warning("Sentiment failed on %s: %s", date, e)

                # 4. Dynamics adjustments
                if use_dynamics:
                    try:
                        mu, cov = self._apply_dynamics_v2(
                            mu, cov, returns, window_returns, tickers,
                            day_idx, vol_aligned, rolling_mean_vol,
                            open_aligned, high_aligned, low_aligned,
                            momentum_series, prices,
                            factor_tracker, causal_graph,
                            transition_detector, signal_gen,
                            dynamics_config, dynamics_log, date,
                            dhat_smoothed=dhat_smoothed,
                            kappa_current=kappa_current,
                            vix_series=vix_series,
                            yield_spread_series=yield_spread_series,
                            dxy_series=dxy_series,
                            ff_daily_factors=ff_daily_factors,
                            edge_updater=_edge_updater,
                            credit_spread_series=credit_spread_series,
                            kbe_series=kbe_series)
                        # Reset lead-lag EMA when M changes at rebalance
                        if _lead_lag_signal is not None:
                            _lead_lag_signal.reset(n_assets)

                        # === AGENT: Strategy 1 — Semantic Factor Labeling ===
                        if (llm_orchestrator is not None
                                and agent_config.semantic_factor_enabled
                                and factor_tracker is not None
                                and factor_tracker.loadings is not None):
                            try:
                                _fl_state = {
                                    "factor_loadings": factor_tracker.loadings,
                                    "tickers": tickers,
                                    "eigenvalues": getattr(
                                        factor_tracker, '_current_eigenvalues', None),
                                    "phases": (
                                        transition_detector._phase
                                        if transition_detector is not None else []),
                                }
                                _fl_result = llm_orchestrator.label_factors(_fl_state)
                                if _fl_result:
                                    self._last_factor_labels = _fl_result
                                    self._persistence_modifier = (
                                        llm_orchestrator._persistence_modifier)
                                    dynamics_log.append({
                                        "date": str(date),
                                        "type": "agent_factor_labels",
                                        "labels": [
                                            {"label": fl["label"],
                                             "persistence": fl["persistence"],
                                             "type": fl.get("type", "unknown")}
                                            for fl in _fl_result],
                                    })
                                    self._agent_factors_today = _fl_result
                            except Exception as _fl_e:
                                logger.warning(
                                    "Agent factor labeling failed: %s", _fl_e)

                    except Exception as e:
                        logger.warning("Dynamics v2 failed on %s: %s", date, e)

                    # Phase-differentiated position scaling is applied to
                    # weights via apply_dislocation_sizing, NOT to mu.
                    # Multiplying mu by phase_scale is wrong: negative mu
                    # values get amplified in the wrong direction.

                    # Directional dislocation tilt for CUSUM-triggered rebalances
                    if is_triggered and signal_gen is not None:
                        try:
                            cusum_vals = transition_detector._cusum
                            phases = transition_detector._phase
                            detected_dirs = transition_detector.directions
                            B = factor_tracker.loadings
                            if B is not None and len(cusum_vals) == B.shape[1]:
                                factor_returns = window_returns.values @ B
                                factor_std = factor_returns.std(axis=0)

                                dislocation_tilt = signal_gen.compute_directional_dislocation(
                                    cusum_vals, detected_dirs, B,
                                    factor_std, phases)
                                dislocation_tilt = dislocation_tilt * event_modifier
                                mu = mu + dislocation_tilt
                        except Exception as e:
                            logger.warning("Directional dislocation failed on %s: %s", date, e)

                    # Kappa-based regime tilt
                    if dynamics_config.reflexivity_enabled:
                        _kappa_lb = self.data_config.scale_window(21)
                        recent_ret = window_returns.iloc[-_kappa_lb:].mean().values
                        if kappa_current > dynamics_config.momentum_threshold:
                            mu = mu + 0.0005 * np.sign(recent_ret)
                        elif kappa_current < dynamics_config.mean_reversion_threshold:
                            mu = mu - 0.0003 * np.sign(recent_ret)

                # 4a-lstm. LSTM factor return forecast (blends with mu)
                if (lstm_forecaster is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None):
                    try:
                        B_lstm = factor_tracker.loadings
                        factor_rets = window_returns.values @ B_lstm
                        lstm_seq = getattr(dynamics_config, 'lstm_seq_len', 63)
                        if len(factor_rets) >= lstm_seq + 1:
                            lstm_forecaster.fit(factor_rets, verbose=False)
                            mu_lstm = lstm_forecaster.predict_asset_returns(
                                factor_rets, B_lstm)
                            blend = getattr(
                                dynamics_config, 'lstm_blend_weight', 0.5)
                            mu = blend * mu_lstm + (1 - blend) * mu
                    except Exception as e:
                        logger.warning(
                            "LSTM forecast failed on %s: %s", date, e)

                # 4a-info. Information time & factor interaction
                if (getattr(dynamics_config, 'information_time_alpha_enabled', False)
                        and dhat_smoothed is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None
                        and transition_detector is not None):
                    try:
                        B_it = factor_tracker.loadings
                        K_it = min(len(dhat_smoothed), B_it.shape[1])
                        if K_it > 0:
                            # Factor returns for ACF computation
                            fr_it = window_returns.values @ B_it[:, :K_it]

                            # Factor interaction: cross-factor propagation
                            if getattr(dynamics_config, 'factor_interaction_enabled', False):
                                _factor_interaction_F = signal_gen.compute_factor_interaction_matrix(
                                    fr_it, K_it)
                                dhat_smoothed = signal_gen.apply_factor_interaction(
                                    dhat_smoothed, _factor_interaction_F,
                                    blend=dynamics_config.factor_interaction_blend)

                            # Information absorption rate τ_k
                            cusum_vals_it = np.array([
                                max(abs(transition_detector._cusum_pos[k]),
                                    abs(transition_detector._cusum_neg[k]))
                                for k in range(K_it)
                            ])
                            phase_days_it = np.array([
                                transition_detector._phase_day_counter[k]
                                for k in range(K_it)
                            ])
                            eigs_it = (factor_tracker._current_eigenvalues[:K_it]
                                       if factor_tracker._current_eigenvalues is not None
                                       else np.ones(K_it))

                            _current_tau_k = signal_gen.compute_information_time(
                                factor_returns_window=fr_it,
                                D_k=dhat_smoothed[:K_it],
                                cusum_values=cusum_vals_it,
                                phase_day_counters=phase_days_it,
                                eigenvalues=eigs_it,
                                factor_tau_history=_factor_tau_history,
                                acf_lookback=dynamics_config.factor_acf_lookback,
                            )

                            # Latency alpha: trade the gap
                            latency_raw = signal_gen.compute_latency_alpha(
                                dhat_smoothed[:K_it], _current_tau_k,
                                B_it[:, :K_it], L_current)
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            _latency_alpha_z = CrossSectionalAlpha._cross_sectional_zscore(
                                latency_raw)
                    except Exception as e:
                        logger.warning("Information time failed on %s: %s", date, e)

                # 4b. Compute per-stock alpha scores
                alpha_scores = np.zeros(n_assets)
                use_dhat_alpha = getattr(dynamics_config, 'dhat_primary_alpha', False)
                use_regime_blend = getattr(dynamics_config, 'regime_blend_alpha', False)

                # D-hat primary alpha: project factor dislocation to stock space
                dhat_alpha_z = None
                if (use_dhat_alpha and dhat_smoothed is not None
                        and factor_tracker.loadings is not None):
                    try:
                        B_alpha = factor_tracker.loadings
                        K_alpha = min(len(dhat_smoothed), B_alpha.shape[1])
                        if K_alpha > 0:
                            # Use raw D_k without phase weighting — phase
                            # weights are z-score invariant and weakening
                            # the signal hurts beta. Phase info is handled
                            # by the entry model instead.
                            raw_alpha = B_alpha[:, :K_alpha] @ dhat_smoothed[:K_alpha]
                            raw_alpha *= (1.0 - L_current)
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            dhat_alpha_z = CrossSectionalAlpha._cross_sectional_zscore(raw_alpha)
                    except Exception as e:
                        logger.warning("D-hat alpha failed on %s: %s", date, e)

                # Regime-blended alpha: blend D-hat with reversal
                if (use_regime_blend and alpha_engine is not None
                        and dhat_alpha_z is not None):
                    try:
                        from NR.dynamics.alpha_signals import CrossSectionalAlpha
                        if len(window_returns) >= dynamics_config.alpha_reversal_window:
                            alpha_engine.compute(window_returns.values)
                            reversal_alpha_z = alpha_engine.regime_composite(
                                kappa_current)
                            w_dhat, w_rev = _regime_blend_weights(
                                kappa_current, dynamics_config)
                            blended = w_dhat * dhat_alpha_z + w_rev * reversal_alpha_z
                            alpha_scores = CrossSectionalAlpha._cross_sectional_zscore(
                                blended)
                        else:
                            alpha_scores = dhat_alpha_z
                    except Exception as e:
                        logger.warning("Regime blend failed on %s: %s", date, e)
                        alpha_scores = dhat_alpha_z
                elif dhat_alpha_z is not None:
                    alpha_scores = dhat_alpha_z

                # Fallback: cross-sectional momentum/reversal alpha
                elif (alpha_engine is not None
                        and len(window_returns) >= dynamics_config.alpha_momentum_window):
                    try:
                        alpha_engine.compute(window_returns.values)
                        alpha_scores = alpha_engine.regime_composite(
                            kappa_current)
                        mu = mu + dynamics_config.alpha_mu_scale * alpha_scores
                    except Exception as e:
                        logger.warning("Alpha signals failed on %s: %s", date, e)

                # 4c. Causal deconfounding + leadership
                if (causal_filter is not None and use_dynamics
                        and causal_graph is not None
                        and causal_graph.graph is not None):
                    try:
                        causal_filter.fit(causal_graph.graph, window_returns.values)
                        cov = causal_filter.deconfounded_covariance(cov)
                        leadership = causal_filter.leadership_scores
                        if leadership is not None and len(leadership) == n_assets:
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            lead_z = CrossSectionalAlpha._cross_sectional_zscore(leadership)
                            alpha_scores = alpha_scores + dynamics_config.causal_leadership_weight * lead_z
                    except Exception as e:
                        logger.warning("Causal filter failed on %s: %s", date, e)

                # 4c-2. Gradient flow alpha (replaces D-hat when enabled)
                if (getattr(dynamics_config, 'gradient_flow_enabled', False)
                        and dhat_smoothed is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None):
                    try:
                        B_gf = factor_tracker.loadings
                        K_gf = min(len(dhat_smoothed), B_gf.shape[1])
                        if K_gf > 0:
                            gf_raw = signal_gen.compute_gradient_flow_alpha(
                                dhat_smoothed[:K_gf], B_gf[:, :K_gf],
                                L_current, kappa_current,
                                momentum_blend_cap=dynamics_config.gradient_flow_momentum_blend_cap)
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            gf_z = CrossSectionalAlpha._cross_sectional_zscore(gf_raw)
                            # Curvature regime gate (Signal 4)
                            if getattr(dynamics_config, 'curvature_regime_enabled', False):
                                _fcov = factor_tracker.factor_covariance
                                if _fcov is not None:
                                    from NR.dynamics.manifold import SPDManifold
                                    _spd = SPDManifold()
                                    _disp_info = _spd.eigenvalue_dispersion(_fcov)
                                    _curvature_gate = signal_gen.compute_curvature_regime_gate(
                                        _disp_info["dispersion"], _dispersion_history,
                                        dynamics_config.curvature_dispersion_lookback)
                            gf_z *= _curvature_gate
                            alpha_scores = alpha_scores + 0.5 * gf_z
                    except Exception as e:
                        logger.warning("Gradient flow alpha failed on %s: %s", date, e)

                # 4c-3. Propagation mispricing additive alpha (Signal 1)
                if (getattr(dynamics_config, 'propagation_mispricing_enabled', False)
                        and hasattr(self, '_last_causal_M')
                        and self._last_causal_M is not None
                        and day_idx >= 2):
                    try:
                        _causal_M = self._last_causal_M
                        prev_returns = returns_arr[day_idx - 1]
                        today_returns_sig = returns_arr[day_idx]
                        predicted = _causal_M @ prev_returns
                        mispricing_raw = predicted - today_returns_sig
                        _, _prop_cumul = signal_gen.compute_propagation_mispricing_signal(
                            mispricing_raw, _mispricing_history,
                            cumulative_window=dynamics_config.propagation_mispricing_cumulative_window)
                        _propagation_mispricing_scores = _prop_cumul
                        from NR.dynamics.alpha_signals import CrossSectionalAlpha
                        mis_z = CrossSectionalAlpha._cross_sectional_zscore(_prop_cumul)
                        alpha_scores = alpha_scores + dynamics_config.propagation_mispricing_alpha_weight * mis_z
                    except Exception as e:
                        logger.warning("Propagation mispricing failed on %s: %s", date, e)

                # 4c-4. Information time latency alpha
                if (_latency_alpha_z is not None
                        and getattr(dynamics_config, 'information_time_alpha_enabled', False)):
                    it_weight = dynamics_config.information_time_alpha_weight
                    alpha_scores = alpha_scores + it_weight * _latency_alpha_z

                # 4c-5. Ξ Contrarian alpha: |D-hat| × contrarian factor momentum
                # High-conviction stocks that lag their factor = mean-reversion alpha
                if (getattr(dynamics_config, 'xi_contrarian_enabled', False)
                        and dhat_smoothed is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None
                        and day_idx >= 5):
                    try:
                        from NR.dynamics.alpha_signals import CrossSectionalAlpha
                        B_xi = factor_tracker.loadings
                        K_xi = min(len(dhat_smoothed), B_xi.shape[1])
                        if K_xi > 0:
                            # Factor engagement: |B @ D_k|
                            conviction_raw = np.abs(B_xi[:, :K_xi] @ dhat_smoothed[:K_xi])

                            # Factor momentum (5-day): project recent returns onto factors
                            recent_ret = returns_arr[day_idx - 5:day_idx]
                            factor_rets_5d = np.mean(recent_ret @ B_xi[:, :K_xi], axis=0)

                            # Flow-momentum alignment per stock
                            factor_direction = np.sign(factor_rets_5d) * np.abs(dhat_smoothed[:K_xi])
                            flow_momentum = B_xi[:, :K_xi] @ factor_direction

                            # Contrarian: overweight stocks AGAINST factor momentum
                            N_xi = len(flow_momentum)
                            align_rank = np.argsort(np.argsort(flow_momentum)) / max(N_xi - 1, 1)
                            xi_contrarian = conviction_raw * (1.5 - align_rank)

                            xi_z = CrossSectionalAlpha._cross_sectional_zscore(xi_contrarian)
                            # EMA smoothing of xi_z signal
                            xi_ema = getattr(dynamics_config, 'xi_signal_ema_alpha', 0.6)
                            if _xi_z_smoothed is not None and len(_xi_z_smoothed) == len(xi_z):
                                xi_z = xi_ema * xi_z + (1.0 - xi_ema) * _xi_z_smoothed
                            _xi_z_smoothed = xi_z.copy()
                            xi_tilt = getattr(dynamics_config, 'xi_contrarian_tilt', 0.5)
                            # Phase-dependent scaling: only apply during dislocation/discovery
                            xi_phase_scales = getattr(dynamics_config, 'xi_phase_scales', None)
                            if xi_phase_scales and transition_detector is not None:
                                priority = {"dislocation": 3, "convergence": 2,
                                            "discovery": 1, "normal": 0}
                                dominant = max(transition_detector._phase,
                                               key=lambda p: priority.get(p, 0))
                                phase_scale = xi_phase_scales.get(dominant, 0.0)
                                xi_tilt = xi_tilt * phase_scale
                            if xi_tilt > 1e-8:
                                alpha_scores = xi_tilt * xi_z + (1.0 - xi_tilt) * alpha_scores
                    except Exception as e:
                        logger.warning("Xi contrarian alpha failed on %s: %s", date, e)

                # 4d. Volatility filter + lifecycle mask
                active_mask = np.ones(n_assets, dtype=bool)
                if plm_active_mask is not None:
                    active_mask = plm_active_mask.copy()
                max_vol = self.config.max_asset_volatility
                if max_vol > 0:
                    vf_lb = self.config.vol_filter_lookback
                    vf_start = max(0, day_idx - vf_lb)
                    trailing_vol = returns.iloc[vf_start:day_idx].std().values
                    ann_vol = trailing_vol * np.sqrt(365)
                    active_mask = ann_vol <= max_vol
                    min_assets = self.config.vol_filter_min_assets
                    if active_mask.sum() < min_assets:
                        sorted_idx = np.argsort(ann_vol)
                        active_mask[:] = False
                        active_mask[sorted_idx[:min_assets]] = True

                # 5. Optimize (RL or classical)
                if (rl_allocator is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None):
                    try:
                        from NR.dynamics.rl_allocator import (
                            PortfolioState, PortfolioEnv)
                        cov_upper = cov[np.triu_indices(n_assets)]
                        regime_int = PortfolioEnv.REGIME_MAP.get(
                            str(regime) if regime else "", 0)
                        phase_arr = np.array([
                            PortfolioEnv.PHASE_MAP.get(p, 0)
                            for p in transition_detector._phase
                        ])
                        rl_state = PortfolioState(
                            expected_returns=mu,
                            cov_features=cov_upper,
                            regime=regime_int,
                            L=L_current,
                            kappa=kappa_current,
                            phases=phase_arr,
                            current_weights=current_weights,
                        )
                        deterministic = not getattr(
                            dynamics_config, 'rl_training_mode', True)
                        new_weights = rl_allocator.select_action(
                            rl_state, deterministic=deterministic)

                        # Store reward from previous rebalance period
                        if (len(equity) > 1
                                and rl_allocator._last_obs is not None):
                            recent_eq = [
                                e["value"]
                                for e in equity[-rebal_freq:]
                            ]
                            if len(recent_eq) >= 2:
                                realized_ret = (
                                    recent_eq[-1] / recent_eq[0] - 1)
                                reward, _ = rl_env.step(
                                    new_weights, realized_ret,
                                    portfolio_value)
                                should_update = (
                                    rl_allocator.store_transition(
                                        reward, new_weights))
                                if (should_update and getattr(
                                        dynamics_config,
                                        'rl_training_mode', True)):
                                    update_info = rl_allocator.update()
                                    dynamics_log.append({
                                        "date": str(date),
                                        "type": "rl_update",
                                        **update_info,
                                    })
                    except Exception as e:
                        logger.warning(
                            "RL allocator failed on %s: %s, "
                            "falling back to classical", date, e)
                        new_weights = self._optimize(
                            weight_method, mu, cov, n_assets, pw,
                            allow_short=self.config.allow_short,
                            alpha_scores=alpha_scores)
                else:
                    # Classical optimization (with optional vol filter subsetting)
                    opt_cov = cov
                    opt_alpha = alpha_scores
                    opt_mu = mu
                    opt_current = current_weights
                    opt_n = n_assets
                    filtered = not np.all(active_mask)

                    if filtered:
                        active_idx = np.where(active_mask)[0]
                        opt_n = len(active_idx)
                        opt_cov = cov[np.ix_(active_idx, active_idx)]
                        opt_alpha = alpha_scores[active_idx]
                        opt_mu = mu[active_idx]
                        opt_current = current_weights[active_idx]
                        cs = opt_current.sum()
                        if cs > 1e-12:
                            opt_current = opt_current / cs
                        else:
                            opt_current = np.ones(opt_n) / opt_n

                    has_alpha = np.any(np.abs(opt_alpha) > 1e-8)
                    turnover_pen = self.config.turnover_penalty_bps
                    tilt = dynamics_config.alpha_tilt_strength if dynamics_config else 0.5
                    # Phase-dynamic alpha tilt: scale tilt during stress phases
                    # Skip phase gating when using xi_alpha_tilt — the alpha scores
                    # already carry phase information via the Ξ signal
                    if (tilt > 0 and dynamics_config
                            and not getattr(dynamics_config, 'xi_alpha_tilt_enabled', False)
                            and getattr(dynamics_config, 'xi_phase_scales', None)
                            and transition_detector is not None):
                        priority = {"dislocation": 3, "convergence": 2,
                                    "discovery": 1, "normal": 0}
                        dominant = max(transition_detector._phase,
                                       key=lambda p: priority.get(p, 0))
                        phase_scale = dynamics_config.xi_phase_scales.get(dominant, 0.0)
                        tilt = tilt * phase_scale
                    core_tickers_cfg = self.config.core_tickers
                    core_pct = self.config.core_allocation_pct

                    # Resolve active tickers for core-satellite
                    if filtered:
                        active_tickers = [tickers[i] for i in np.where(active_mask)[0]]
                    else:
                        active_tickers = list(tickers)

                    _cdr_mom_ready = (day_idx >= getattr(dynamics_config, 'cdr_momentum_window', 252)
                                      or days_since_start >= 30)  # lookback already provides history
                    _cdr_cond = (getattr(dynamics_config, 'cdr_enabled', False)
                            and dhat_smoothed is not None
                            and factor_tracker is not None
                            and factor_tracker.loadings is not None
                            and _cdr_mom_ready)
                    if _cdr_cond:
                        # CDR: Conviction x Direction x Risk
                        try:
                            from NR.dynamics.alpha_signals import CrossSectionalAlpha
                            B_cdr = factor_tracker.loadings
                            K_cdr = min(len(dhat_smoothed), B_cdr.shape[1])

                            # C_i = |B @ D_k| per asset (raw conviction)
                            conviction_raw = np.abs(
                                B_cdr[:, :K_cdr] @ dhat_smoothed[:K_cdr]
                            ) if K_cdr > 0 else np.ones(n_assets)

                            # Optionally blend Xi contrarian into conviction
                            _xi_blend = getattr(
                                dynamics_config,
                                'cdr_xi_conviction_blend', 0.0)
                            if _xi_blend > 0 and day_idx >= 5 and K_cdr > 0:
                                try:
                                    _xi_ret = returns_arr[day_idx - 5:day_idx]
                                    _xi_fret = np.mean(
                                        _xi_ret @ B_cdr[:, :K_cdr], axis=0)
                                    _xi_fdir = (np.sign(_xi_fret)
                                                * np.abs(dhat_smoothed[:K_cdr]))
                                    _xi_fmom = B_cdr[:, :K_cdr] @ _xi_fdir
                                    _xi_N = len(_xi_fmom)
                                    _xi_rank = (
                                        np.argsort(np.argsort(_xi_fmom))
                                        / max(_xi_N - 1, 1))
                                    _xi_conv = conviction_raw * (
                                        1.5 - _xi_rank)
                                    _xi_conv = np.maximum(_xi_conv, 0.0)
                                    conviction_raw = (
                                        (1.0 - _xi_blend) * conviction_raw
                                        + _xi_blend * _xi_conv)
                                except Exception as _xi_e:
                                    logger.warning(
                                        "CDR Xi conviction blend failed "
                                        "on %s: %s", date, _xi_e)

                            # Phase gating on conviction
                            cdr_phase_scales = getattr(
                                dynamics_config, 'cdr_phase_scales', None)
                            phase_scale = 1.0
                            if (cdr_phase_scales
                                    and transition_detector is not None):
                                priority = {"dislocation": 3,
                                            "convergence": 2,
                                            "discovery": 1, "normal": 0}
                                dominant = max(
                                    transition_detector._phase,
                                    key=lambda p: priority.get(p, 0))
                                phase_scale = cdr_phase_scales.get(
                                    dominant, 0.3)
                            conviction_gated = conviction_raw * phase_scale

                            # D_i = direction from momentum + reversal blend
                            cdr_mom_win = getattr(
                                dynamics_config, 'cdr_momentum_window', 252)
                            cdr_mom_skip = getattr(
                                dynamics_config, 'cdr_momentum_skip', 21)
                            cdr_rev_win = getattr(
                                dynamics_config, 'cdr_reversal_window', 5)
                            cdr_mw = getattr(
                                dynamics_config, 'cdr_momentum_weight', 0.7)
                            cdr_rw = getattr(
                                dynamics_config, 'cdr_reversal_weight', 0.3)

                            mom_start = max(
                                0, day_idx - cdr_mom_win)
                            mom_end = max(
                                0, day_idx - cdr_mom_skip)
                            if mom_end > mom_start:
                                mom_ret = np.sum(
                                    returns_arr[mom_start:mom_end],
                                    axis=0)
                            else:
                                mom_ret = np.zeros(n_assets)
                            z_mom = CrossSectionalAlpha._cross_sectional_zscore(
                                mom_ret)

                            rev_start = max(0, day_idx - cdr_rev_win)
                            rev_ret = np.sum(
                                returns_arr[rev_start:day_idx],
                                axis=0)
                            z_rev = CrossSectionalAlpha._cross_sectional_zscore(
                                -rev_ret)  # negated for mean-reversion

                            # Phase-dependent direction switching
                            if (transition_detector is not None
                                    and cdr_phase_scales):
                                if dominant == "dislocation":
                                    # Pure momentum in dislocation
                                    direction = z_mom
                                elif dominant == "convergence":
                                    # Contrarian in convergence
                                    direction = -z_mom
                                else:
                                    direction = (
                                        cdr_mw * z_mom + cdr_rw * z_rev)
                            else:
                                direction = (
                                    cdr_mw * z_mom + cdr_rw * z_rev)

                            # R_i = 1/sigma from trailing vol
                            cdr_vol_lb = getattr(
                                dynamics_config, 'cdr_vol_lookback', 63)
                            vol_start = max(0, day_idx - cdr_vol_lb)
                            trailing_std = returns.iloc[
                                vol_start:day_idx].std().values
                            trailing_std = np.maximum(trailing_std, 1e-8)
                            inv_vol_cdr = 1.0 / trailing_std

                            # Log CDR screener data for post-hoc analysis
                            dynamics_log.append({
                                "date": str(date),
                                "type": "cdr_screener",
                                "conviction_gated": conviction_gated.copy(),
                                "direction": direction.copy(),
                                "inv_vol_cdr": inv_vol_cdr.copy(),
                            })

                            # Apply vol filter if active
                            if filtered:
                                active_idx = np.where(active_mask)[0]
                                conv_f = conviction_gated[active_idx]
                                dir_f = direction[active_idx]
                                ivol_f = inv_vol_cdr[active_idx]
                                cur_f = opt_current
                                hd_f = _cdr_holding_days[active_idx]
                            else:
                                conv_f = conviction_gated
                                dir_f = direction
                                ivol_f = inv_vol_cdr
                                cur_f = current_weights
                                hd_f = _cdr_holding_days

                            sub_weights = pw.conviction_direction_risk(
                                conviction=conv_f,
                                direction=dir_f,
                                inv_vol=ivol_f,
                                conviction_max=getattr(
                                    dynamics_config,
                                    'cdr_conviction_max', 3.0),
                                short_exposure=getattr(
                                    dynamics_config,
                                    'cdr_short_exposure', 0.0),
                                current_weights=cur_f,
                                max_weight_change=getattr(
                                    dynamics_config,
                                    'cdr_max_weight_change', 0.20),
                                holding_days=hd_f,
                                min_hold_days=getattr(
                                    dynamics_config,
                                    'cdr_min_hold_days', 15),
                                conviction_percentile=getattr(
                                    dynamics_config,
                                    'cdr_conviction_percentile', 0.30),
                                direction_threshold=getattr(
                                    dynamics_config,
                                    'cdr_direction_threshold', 0.3),
                            )
                            _cdr_last_rebal_day = day_idx
                        except Exception as e:
                            logger.warning(
                                "CDR weight construction failed on "
                                "%s: %s, falling back", date, e)
                            sub_weights = self._optimize(
                                weight_method, opt_mu, opt_cov,
                                opt_n, pw,
                                allow_short=self.config.allow_short,
                                alpha_scores=opt_alpha)

                    elif weight_method == "equal_weight":
                        sub_weights = np.ones(opt_n) / opt_n
                    elif core_tickers_cfg and core_pct > 0:
                        sub_weights = pw.core_satellite_weights(
                            cov_matrix=opt_cov,
                            tickers=active_tickers,
                            core_tickers=core_tickers_cfg,
                            core_pct=core_pct,
                            core_method=self.config.core_weight_method,
                            satellite_alpha=opt_alpha if has_alpha else None,
                            tilt_strength=tilt,
                            current_weights=opt_current,
                            turnover_penalty_bps=turnover_pen,
                        )
                    elif (getattr(dynamics_config, 'xi_alpha_tilt_enabled', False)
                          and has_alpha
                          and dhat_smoothed is not None
                          and factor_tracker is not None
                          and factor_tracker.loadings is not None):
                        B_wt = factor_tracker.loadings
                        K_wt = min(len(dhat_smoothed), B_wt.shape[1])
                        dhat_abs = np.abs(B_wt[:, :K_wt] @ dhat_smoothed[:K_wt]) if K_wt > 0 else np.ones(opt_n)
                        if filtered:
                            dhat_abs = dhat_abs[np.where(active_mask)[0]]
                        # Conviction filter: concentrate on top percentile by |D̂|
                        conv_pct = getattr(dynamics_config, 'xi_conviction_percentile', 0.0)
                        if conv_pct > 0 and len(dhat_abs) > 10:
                            n_keep = max(10, int(len(dhat_abs) * conv_pct))
                            threshold = np.sort(dhat_abs)[-n_keep]
                            conv_mask = dhat_abs >= threshold
                            conv_idx = np.where(conv_mask)[0]
                            conv_cov = opt_cov[np.ix_(conv_idx, conv_idx)]
                            conv_alpha = opt_alpha[conv_idx]
                            conv_dhat = dhat_abs[conv_idx]
                            conv_current = opt_current[conv_idx] if opt_current is not None else None
                            conv_weights = pw.xi_alpha_tilt(
                                conv_cov, conv_alpha, conv_dhat,
                                tilt_strength=tilt,
                                z_clip=getattr(dynamics_config, 'xi_tilt_z_clip', 2.0),
                                current_weights=conv_current,
                                blend_rate=getattr(dynamics_config, 'xi_tilt_blend_rate', 0.67),
                                rebalance_band=getattr(dynamics_config, 'xi_tilt_rebalance_band', 0.0),
                            )
                            sub_weights = np.zeros(opt_n)
                            sub_weights[conv_idx] = conv_weights
                        else:
                            sub_weights = pw.xi_alpha_tilt(
                                opt_cov, opt_alpha, dhat_abs,
                                tilt_strength=tilt,
                                z_clip=getattr(dynamics_config, 'xi_tilt_z_clip', 2.0),
                                current_weights=opt_current,
                                blend_rate=getattr(dynamics_config, 'xi_tilt_blend_rate', 0.67),
                                rebalance_band=getattr(dynamics_config, 'xi_tilt_rebalance_band', 0.0),
                            )
                    elif (turnover_pen > 0
                          and weight_method not in ("max_sharpe",
                                                    "capped_max_sharpe",
                                                    "blended_max_sharpe",
                                                    "min_variance",
                                                    "equal_weight")):
                        sub_weights = pw.turnover_penalized_risk_parity(
                            opt_cov, opt_current,
                            penalty_bps=turnover_pen,
                            alpha_scores=opt_alpha if has_alpha else None,
                            tilt_strength=tilt,
                        )
                    elif ((has_alpha or weight_method == "signal_aware_risk_parity")
                          and weight_method not in ("equal_weight",)):
                        sub_weights = pw.signal_aware_risk_parity(
                            opt_cov, opt_alpha, tilt,
                            allow_short=self.config.allow_short)
                    else:
                        sub_weights = self._optimize(
                            weight_method, opt_mu, opt_cov, opt_n, pw,
                            allow_short=self.config.allow_short,
                            alpha_scores=opt_alpha)

                    if filtered:
                        new_weights = np.zeros(n_assets)
                        new_weights[np.where(active_mask)[0]] = sub_weights
                    else:
                        new_weights = sub_weights

                # 4c-bis. Core-satellite alpha overlay
                # Blend: (1-α) × core_RP + α × concentrated_L/S_Ξ
                if (getattr(dynamics_config, 'xi_overlay_enabled', False)
                        and _xi_z_smoothed is not None
                        and len(_xi_z_smoothed) == n_assets):
                    try:
                        opct = getattr(dynamics_config, 'xi_overlay_pct', 0.25)
                        overlay = pw.xi_alpha_overlay(
                            cov, _xi_z_smoothed,
                            long_pct=getattr(dynamics_config, 'xi_overlay_long_pct', 0.20),
                            short_pct=getattr(dynamics_config, 'xi_overlay_short_pct', 0.20),
                        )
                        new_weights = (1.0 - opct) * new_weights + opct * overlay
                    except Exception as e:
                        logger.warning(
                            "Xi overlay failed on %s: %s", date, e)

                # 4d. Framework dislocation sizing overlay
                # w_s ∝ D_k * beta_sk * (1/lambda_s) * (1-L)
                # Phase-gated: only apply during dislocation/discovery
                # Skip when overlay active — disloc sizing clips negatives & renorms to 1.0
                _disloc_tilt = tilt if 'tilt' in dir() else (
                    dynamics_config.alpha_tilt_strength if dynamics_config else 0.0)
                if (use_dynamics
                        and not getattr(dynamics_config, 'xi_overlay_enabled', False)
                        and signal_gen is not None
                        and dhat_smoothed is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None
                        and factor_tracker._current_eigenvalues is not None
                        and _disloc_tilt > 1e-8):
                    try:
                        new_weights = signal_gen.apply_dislocation_sizing(
                            new_weights, dhat_smoothed,
                            factor_tracker.loadings,
                            factor_tracker._current_eigenvalues,
                            L_current,
                            transition_detector._phase,
                            blend_strength=0.1,
                            kappa=(kappa_current
                                   if dynamics_config.kappa_sizing_modulation
                                   else None),
                        )
                    except Exception as e:
                        logger.warning(
                            "Dislocation sizing failed on %s: %s", date, e)

                # 5a. Self-referential feedback loop
                # weights → flow impact → D_k' → mu' → weights'
                # Skip when overlay active — self-ref re-optimizes and renorms to 1.0
                if (getattr(dynamics_config, 'self_referential_enabled', False)
                        and not getattr(dynamics_config, 'xi_overlay_enabled', False)
                        and use_dynamics
                        and signal_gen is not None
                        and factor_tracker is not None
                        and factor_tracker.loadings is not None
                        and dhat_smoothed is not None):
                    try:
                        B_sr = factor_tracker.loadings
                        mean_vol_idx_sr = min(
                            day_idx - 1, len(rolling_mean_vol) - 1)
                        if mean_vol_idx_sr >= 0:
                            mean_vol_sr = rolling_mean_vol.iloc[
                                mean_vol_idx_sr].values
                        else:
                            mean_vol_sr = np.ones(n_assets)

                        new_weights, mu = self._self_referential_loop(
                            new_weights, current_weights, mu, mu_baseline,
                            cov, dhat_smoothed, B_sr, mean_vol_sr,
                            signal_gen, transition_detector,
                            dynamics_config, pw, weight_method,
                            alpha_scores, active_mask, n_assets,
                            dynamics_log, date,
                            kappa_current=kappa_current,
                        )
                    except Exception as e:
                        logger.warning(
                            "Self-referential loop failed on %s: %s",
                            date, e)

                # 5a2. Operator spectrum hedging: reduce exposure to unstable modes
                if (getattr(dynamics_config, 'operator_spectrum_hedge', False)
                        and self._operator_spectrum is not None
                        and getattr(self, '_last_spectrum', None) is not None):
                    try:
                        _spec = self._last_spectrum
                        hedge = self._operator_spectrum.hedging_direction(
                            _spec["eigenvectors"], _spec["eigenvalues"],
                            new_weights)
                        if np.any(np.abs(hedge) > 1e-12):
                            new_weights = new_weights + hedge
                            # Renormalize to maintain net exposure
                            _net = new_weights.sum()
                            if abs(_net) > 1e-12:
                                new_weights *= 1.0 / _net
                    except Exception:
                        pass

                # 5b. Apply weight caps
                if (self.config.vol_scaled_caps
                        or self.config.max_weight_per_asset < 1.0
                        or self.config.min_weight_per_asset > 0.0):
                    per_asset_max = None
                    if self.config.vol_scaled_caps:
                        vol_lb = self.config.vol_cap_lookback
                        vol_start = max(0, day_idx - vol_lb)
                        trailing_vol = returns.iloc[vol_start:day_idx].std().values
                        median_vol = np.median(trailing_vol[trailing_vol > 0])
                        if median_vol > 1e-12:
                            vol_ratio = np.maximum(trailing_vol / median_vol, 0.1)
                            per_asset_max = np.maximum(
                                self.config.vol_cap_base / vol_ratio,
                                self.config.vol_cap_floor)
                    # Preserve intended net exposure (< 1.0 when overlay has shorts)
                    _intended_net = new_weights.sum()
                    if abs(_intended_net) < 1e-12:
                        _intended_net = 1.0
                    new_weights = pw.clip_and_renormalize(
                        new_weights,
                        min_weight=self.config.min_weight_per_asset,
                        max_weight=self.config.max_weight_per_asset,
                        per_asset_max=per_asset_max,
                        net_exposure=_intended_net,
                    )

                # 5b2. Conviction-based position filtering
                # (Skip when lifecycle manager is active — it handles selection)
                max_pos = self.config.max_positions
                min_pos_w = self.config.min_position_weight
                conv_thresh = self.config.conviction_threshold
                if (plm_active_mask is None
                        and (max_pos > 0 or min_pos_w > 0 or conv_thresh > 0)):
                    # Zero out low-conviction positions
                    if conv_thresh > 0 and np.any(np.abs(alpha_scores) > 1e-8):
                        new_weights[np.abs(alpha_scores) < conv_thresh] = 0.0
                    # Zero out below minimum weight (use abs for long-short)
                    if min_pos_w > 0:
                        new_weights[np.abs(new_weights) < min_pos_w] = 0.0
                    # Keep only top N positions by |weight|
                    if max_pos > 0 and np.count_nonzero(new_weights) > max_pos:
                        sorted_idx = np.argsort(np.abs(new_weights))[::-1]
                        new_weights[sorted_idx[max_pos:]] = 0.0
                    # Renormalize (preserve net exposure for L/S overlay)
                    total = new_weights.sum()
                    _target_net = _intended_net if '_intended_net' in dir() else 1.0
                    if abs(total) > 1e-12:
                        new_weights *= _target_net / total
                    else:
                        new_weights = np.ones(n_assets) / n_assets

                # 5b3. Lifecycle sizing overlay
                if (lifecycle_result is not None
                        and plm_active_mask is not None):
                    target_sizes = lifecycle_result["target_sizes"]
                    active_idx = np.where(plm_active_mask)[0]
                    if len(active_idx) > 0:
                        lc_weights = np.zeros(n_assets)
                        lc_weights[active_idx] = target_sizes[active_idx]
                        total_lc = lc_weights.sum()
                        total_opt = new_weights.sum()
                        if total_lc > 1e-12 and total_opt > 1e-12:
                            lc_weights = lc_weights / total_lc * total_opt
                            blend = position_manager_config.lifecycle_sizing_blend
                            new_weights = ((1 - blend) * new_weights
                                           + blend * lc_weights)
                        new_weights = np.minimum(
                            new_weights,
                            position_manager_config.max_position_size)
                        total = new_weights.sum()
                        if total > position_manager_config.max_total_exposure:
                            new_weights *= (
                                position_manager_config.max_total_exposure
                                / total)

                # 5c. Cross-asset exposure reduction
                if self.config.cross_asset_enabled:
                    risk_signals = []
                    ca_lb = self.config.cross_asset_lookback
                    if dxy_series is not None:
                        try:
                            dxy_now = float(dxy_series.asof(date))
                            dxy_past = float(dxy_series.asof(
                                dates[max(0, day_idx - ca_lb)]))
                            if dxy_past > 0:
                                risk_signals.append(
                                    (dxy_now / dxy_past - 1) * 10)
                        except (KeyError, TypeError):
                            pass
                    if tnx_series is not None:
                        try:
                            tnx_now = float(tnx_series.asof(date))
                            tnx_past = float(tnx_series.asof(
                                dates[max(0, day_idx - ca_lb)]))
                            if tnx_past > 0:
                                risk_signals.append(
                                    (tnx_now / tnx_past - 1) * 5)
                        except (KeyError, TypeError):
                            pass
                    if vix_series is not None:
                        try:
                            vix_now = float(vix_series.asof(date))
                            risk_signals.append(vix_now / 20.0 - 1.0)
                        except (KeyError, TypeError):
                            pass
                    if risk_signals:
                        risk_score = np.clip(np.mean(risk_signals), 0, 1)
                        exposure = 1.0 - self.config.cross_asset_max_reduction * risk_score
                        new_weights = new_weights * exposure

                # 5d. Volatility targeting (with leverage)
                if getattr(self.config, 'vol_target_enabled', False):
                    vt_lookback = getattr(self.config, 'vol_target_lookback', 21)
                    vt_target = getattr(self.config, 'vol_target_annualized', 0.15)
                    if day_idx >= vt_lookback:
                        port_rets = (returns.iloc[day_idx - vt_lookback:day_idx]
                                     .values @ current_weights)
                        realized_vol = float(port_rets.std()) * np.sqrt(self.bars_per_year)
                        if realized_vol > 1e-6:
                            vol_scalar = vt_target / realized_vol
                            vt_max_lev = getattr(
                                self.config, 'vol_target_max_leverage', 1.5)
                            vt_min_exp = getattr(
                                self.config, 'vol_target_min_exposure', 0.3)
                            vol_scalar = float(
                                np.clip(vol_scalar, vt_min_exp, vt_max_lev))
                            new_weights = new_weights * vol_scalar

                # 5e. Regime gate — scale deployment based on kappa/VIX/L
                regime_gate_val = 1.0
                if (use_dynamics and dynamics_config is not None
                        and getattr(dynamics_config, 'regime_gate_enabled', False)):
                    # Compute VIX z-score from rolling window
                    _vix_z = 0.0
                    if vix_series is not None:
                        try:
                            vix_now = float(vix_series.asof(date))
                            _vix_lb = min(day_idx, 252)
                            if _vix_lb >= 20:
                                _vix_vals = np.array([
                                    float(vix_series.asof(dates[j]))
                                    for j in range(day_idx - _vix_lb, day_idx)])
                                _vix_mu = _vix_vals.mean()
                                _vix_sd = _vix_vals.std()
                                if _vix_sd > 1e-6:
                                    _vix_z = (vix_now - _vix_mu) / _vix_sd
                        except (KeyError, TypeError, ValueError):
                            pass
                    _raw_gate = _compute_regime_gate(
                        kappa_current, _vix_z, L_current, dynamics_config)
                    # EMA smooth to prevent whipsaw on kappa boundary
                    _gate_alpha = getattr(
                        dynamics_config, 'regime_gate_ema_alpha', 0.3)
                    _gate_ema = _gate_alpha * _raw_gate + (1 - _gate_alpha) * _gate_ema
                    regime_gate_val = _gate_ema
                    if regime_gate_val < 1.0 - 1e-6:
                        new_weights = new_weights * regime_gate_val
                        dynamics_log.append({
                            "date": str(date), "type": "regime_gate",
                            "gate": float(regime_gate_val),
                            "raw_gate": float(_raw_gate),
                            "kappa": kappa_current,
                            "vix_z": float(_vix_z),
                            "L": L_current,
                        })

                # 6. Filter dust trades (skip weight changes below threshold)
                min_tw = self.config.min_trade_weight
                if min_tw > 0:
                    intended_total = new_weights.sum()
                    trade_delta = np.abs(new_weights - current_weights)
                    dust = trade_delta < min_tw
                    new_weights[dust] = current_weights[dust]
                    total_w = new_weights.sum()
                    if total_w > 1e-12 and intended_total > 1e-12:
                        new_weights *= intended_total / total_w

                # 6b. Min trade size filter (dynamics config, pct-based)
                _min_ts = (getattr(dynamics_config, 'min_trade_size_pct', 0.0)
                           if dynamics_config is not None else 0.0)
                if _min_ts > 0:
                    intended_total_6b = new_weights.sum()
                    trade_delta = np.abs(new_weights - current_weights)
                    small = trade_delta < _min_ts
                    new_weights[small] = current_weights[small]
                    total_w = new_weights.sum()
                    if total_w > 1e-12 and intended_total_6b > 1e-12:
                        new_weights *= intended_total_6b / total_w

                # 7. Update cash weight from any exposure reduction
                if self.config.allow_short:
                    cash_weight = 1.0 - new_weights.sum()
                else:
                    cash_weight = max(0.0, 1.0 - new_weights.sum())

                # 8. Transaction costs
                cost = self._compute_rebalance_cost(
                    current_weights, new_weights, portfolio_value,
                    tickers, vol_aligned, prices, returns, day_idx)
                turnover = float(np.sum(np.abs(new_weights - current_weights)) / 2)

                portfolio_value -= cost

                weights_history.append({
                    "date": date, "weights": dict(zip(tickers, new_weights)),
                    "turnover": turnover, "cash_weight": cash_weight,
                })
                cost_history.append({"date": date, "cost": cost, "turnover": turnover})
                decision_log.append({
                    "date": date, "regime": regime, "turnover": turnover,
                    "cost": cost, "method": weight_method,
                    "triggered": is_triggered,
                    "L": L_current, "kappa": kappa_current,
                    "regime_gate": regime_gate_val,
                    "curvature_gate": _curvature_gate,
                })

                if is_triggered:
                    extra_rebalance_count += 1
                    dynamics_log.append({
                        "date": str(date), "type": "triggered_rebalance",
                        "extra_count": extra_rebalance_count,
                    })

                # CDR holding days: reset for changed positions, increment for held
                if getattr(dynamics_config, 'cdr_enabled', False):
                    changed = np.abs(new_weights - current_weights) > 1e-8
                    _cdr_holding_days[changed] = 0
                    _cdr_holding_days[~changed] += 1

                current_weights = new_weights

            # CDR holding days: increment on non-rebalance days
            elif getattr(dynamics_config, 'cdr_enabled', False):
                _cdr_holding_days += 1

            # === AGENT: End-of-day decision logging ===
            if llm_orchestrator is not None and agent_config.learning_enabled:
                _agent_day_decisions = {}
                if hasattr(self, '_agent_cusum_today'):
                    _agent_day_decisions["cusum_validation"] = self._agent_cusum_today
                    del self._agent_cusum_today
                if hasattr(self, '_agent_macro_today'):
                    _agent_day_decisions["macro_reflexivity"] = self._agent_macro_today
                    del self._agent_macro_today
                if hasattr(self, '_agent_exit_today'):
                    _agent_day_decisions["exit_overrides"] = self._agent_exit_today
                    del self._agent_exit_today
                if hasattr(self, '_agent_factors_today'):
                    _agent_day_decisions["factor_labels"] = self._agent_factors_today
                    del self._agent_factors_today
                _agent_state = {
                    "kappa_current": kappa_current,
                    "vix_current": (
                        float(vix_series.iloc[day_idx])
                        if vix_series is not None
                        and day_idx < len(vix_series) else None),
                    "credit_spread": (
                        float(credit_spread_series.iloc[day_idx])
                        if credit_spread_series is not None
                        and day_idx < len(credit_spread_series) else None),
                    "regime": regime if 'regime' in dir() else None,
                    "L_current": L_current,
                }
                llm_orchestrator.learning_store.log_decision(
                    date, _agent_state, _agent_day_decisions)
                llm_orchestrator._daily_decisions.append({
                    "date": str(date),
                    "decisions": _agent_day_decisions,
                })

        # ========== UP-TO-DATE SHORTCUT ==========
        # If checkpoint was up-to-date and loop ran 0 iterations,
        # return results built from restored state directly.
        if _resuming and _start_day_idx >= len(dates):
            logger.info("Checkpoint up-to-date — returning restored state "
                        "(portfolio=%.0f, %d equity points, %d weights)",
                        portfolio_value, len(equity), len(weights_history))
            # Re-save checkpoint as-is
            self._checkpoint = resume_checkpoint
            # Build minimal results from restored state
            equity_df = pd.DataFrame(equity)
            if not equity_df.empty:
                equity_df = equity_df.set_index("date")
                equity_returns = equity_df["value"].pct_change().dropna()
            else:
                equity_df = pd.DataFrame(
                    {"value": [portfolio_value]},
                    index=[dates[-1]])
                equity_returns = pd.Series(dtype=float)

            perf = PerformanceMetrics(periods_per_year=self.bars_per_year)
            spy_equity_series = None
            spy_returns_series = None
            if spy_equity:
                spy_df = pd.DataFrame(spy_equity).set_index("date")
                spy_equity_series = spy_df["value"]
                spy_returns_series = spy_equity_series.pct_change().dropna()

            metrics = perf.compute_all(
                equity_returns, spy_returns_series
            ) if len(equity_returns) > 0 else {}

            total_costs = sum(c["cost"] for c in cost_history)
            total_turnover = sum(c["turnover"] for c in cost_history)
            bpy = self.bars_per_year
            self._results = {
                "equity_curve": equity_df["value"] if "value" in equity_df.columns else equity_df.iloc[:, 0],
                "spy_equity_curve": spy_equity_series,
                "returns": equity_returns,
                "spy_returns": spy_returns_series,
                "weights_history": weights_history,
                "cost_history": cost_history,
                "decision_log": decision_log,
                "total_cost": total_costs,
                "total_turnover": total_turnover,
                "cost_drag_annual": total_costs / capital / max(len(equity_returns) / bpy, 1 / bpy) if equity_returns is not None and len(equity_returns) > 0 else 0.0,
                "metrics": metrics,
                "initial_capital": capital,
                "final_value": portfolio_value,
                "dynamics_log": dynamics_log,
                "attribution_log": attribution_log,
                "extra_rebalances": extra_rebalance_count,
                "validation_log": {
                    "L_history": L_history,
                    "kappa_history": kappa_history,
                },
                "dk_series": dk_series,
                "lifecycle_log": (position_manager.lifecycle_log
                                  if position_manager is not None else []),
                "fiber_bundle": fiber_bundle.bundle if fiber_bundle is not None else None,
                "agent_stats": (llm_orchestrator.finalize()
                                if llm_orchestrator is not None else None),
                "agent_decision_stats": (llm_orchestrator.stats
                                         if llm_orchestrator is not None
                                         else None),
            }
            return self._results

        # ========== CHECKPOINT SAVE ==========
        # Determine last processed day_idx
        _ckpt_day_idx = (day_idx if 'day_idx' in dir() and equity
                         else (_start_day_idx - 1
                               if _start_day_idx > lookback
                               else lookback - 1))
        self._checkpoint = {
            # Loop position
            'day_idx': _ckpt_day_idx,
            'days_since_start': max(_ckpt_day_idx - lookback, 0),

            # Portfolio state
            'current_weights': current_weights.copy(),
            'portfolio_value': portfolio_value,
            'cash_weight': cash_weight,
            'is_exited': is_exited,
            '_dd_ema': _dd_ema,
            '_gate_ema': _gate_ema,
            '_portfolio_peak': getattr(self, '_portfolio_peak',
                                       portfolio_value),

            # Sub-component objects
            'factor_tracker': factor_tracker,
            'transition_detector': transition_detector,
            'signal_gen': signal_gen,
            'causal_graph': causal_graph,
            'volume_decomposer': volume_decomposer,
            'alpha_engine': alpha_engine,
            'causal_filter': causal_filter,
            'position_manager': position_manager,
            'llm_orchestrator': llm_orchestrator,
            '_edge_updater': _edge_updater,
            '_sentiment_phase_validator': _sentiment_phase_validator,
            '_markov_detector': _markov_detector,
            '_lead_lag_signal': _lead_lag_signal,
            'fiber_bundle': fiber_bundle,
            'lstm_forecaster': lstm_forecaster,

            # Smoothed / accumulated signals
            'dhat_smoothed': dhat_smoothed,
            '_xi_z_smoothed': _xi_z_smoothed,
            'daily_alpha_scores': daily_alpha_scores.copy()
                if daily_alpha_scores is not None else None,
            '_mispricing_history': _mispricing_history,
            '_dispersion_history': _dispersion_history,
            '_curvature_gate': _curvature_gate,
            '_propagation_mispricing_scores':
                _propagation_mispricing_scores,
            '_latent_shock_scores': _latent_shock_scores,
            '_sia_outgoing_scores': _sia_outgoing_scores,
            '_factor_tau_history': _factor_tau_history,
            '_current_tau_k': _current_tau_k,
            '_factor_interaction_F': _factor_interaction_F,
            '_latency_alpha_z': _latency_alpha_z,
            '_lead_lag_scores': _lead_lag_scores,

            # CDR state
            '_cdr_holding_days': _cdr_holding_days.copy(),
            '_cdr_last_rebal_day': _cdr_last_rebal_day,

            # Dynamics metrics
            'extra_rebalance_count': extra_rebalance_count,
            'cov_history': cov_history[-63:],
            'kappa_current': kappa_current,
            'L_current': L_current,

            # History logs (trimmed for checkpoint size)
            'equity': equity[-252:],
            'spy_equity': spy_equity[-252:],
            'weights_history': weights_history,
            'cost_history': cost_history[-252:],
            'decision_log': decision_log[-50:],
            'dynamics_log': dynamics_log[-252:],
            'L_history': L_history[-252:],
            'kappa_history': kappa_history[-252:],
            'dk_series': dk_series[-63:],
            'attribution_log': attribution_log[-50:],
            'lifecycle_result': lifecycle_result,

            # Self-stored state
            '_last_causal_M': getattr(self, '_last_causal_M', None),
            '_last_info_weight': getattr(self, '_last_info_weight', 1.0),
            '_last_soft_phases': getattr(self, '_last_soft_phases', None),
            '_last_spectrum': getattr(self, '_last_spectrum', None),

            # SPY initial price for benchmark continuity
            'spy_initial_price': spy_initial_price,

            # Precomputed data
            'ff_daily_factors': ff_daily_factors,
            'rmw_betas': rmw_betas,

            # Universe metadata for mismatch detection
            'tickers': tickers,
            'n_assets': n_assets,
            'lookback': lookback,
        }

        # ========== BUILD RESULTS ==========
        equity_df = pd.DataFrame(equity)
        if equity_df.empty:
            raise ValueError("No backtest periods were generated")
        equity_df = equity_df.set_index("date")

        equity_returns = equity_df["value"].pct_change().dropna()
        perf = PerformanceMetrics(periods_per_year=self.bars_per_year)

        # SPY equity curve
        spy_equity_series = None
        spy_returns_series = None
        if spy_equity:
            spy_df = pd.DataFrame(spy_equity).set_index("date")
            spy_equity_series = spy_df["value"]
            spy_returns_series = spy_equity_series.pct_change().dropna()

        metrics = perf.compute_all(equity_returns, spy_returns_series)

        # Tracking error
        if spy_returns_series is not None:
            aligned_spy = spy_returns_series.reindex(equity_returns.index, fill_value=0)
            active_returns = equity_returns - aligned_spy
            metrics["tracking_error"] = float(active_returns.std() * np.sqrt(self.bars_per_year))

        total_costs = sum(c["cost"] for c in cost_history)
        total_turnover = sum(c["turnover"] for c in cost_history)

        bpy = self.bars_per_year
        self._results = {
            "equity_curve": equity_df["value"],
            "spy_equity_curve": spy_equity_series,
            "returns": equity_returns,
            "spy_returns": spy_returns_series,
            "weights_history": weights_history,
            "cost_history": cost_history,
            "decision_log": decision_log,
            "total_cost": total_costs,
            "total_turnover": total_turnover,
            "cost_drag_annual": total_costs / capital / max(len(equity_returns) / bpy, 1 / bpy),
            "metrics": metrics,
            "initial_capital": capital,
            "final_value": portfolio_value,
            "dynamics_log": dynamics_log,
            "attribution_log": attribution_log,
            "extra_rebalances": extra_rebalance_count,
            "validation_log": {
                "L_history": L_history,
                "kappa_history": kappa_history,
            },
            "dk_series": dk_series,
            "lifecycle_log": (position_manager.lifecycle_log
                              if position_manager is not None else []),
            "fiber_bundle": fiber_bundle.bundle if fiber_bundle is not None else None,
            "agent_stats": (llm_orchestrator.finalize()
                            if llm_orchestrator is not None else None),
            "agent_decision_stats": (llm_orchestrator.stats
                                     if llm_orchestrator is not None
                                     else None),
        }
        return self._results

    def save_checkpoint(self, path) -> None:
        """Serialize engine checkpoint state to disk (pickle)."""
        import pickle
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = getattr(self, '_checkpoint', None)
        if ckpt is None:
            logger.warning("No checkpoint to save (run_event_driven not "
                           "called yet)")
            return
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / 1e6
        logger.info("Checkpoint saved: %s (%.1f MB, day_idx=%d)",
                    path, size_mb, ckpt.get('day_idx', -1))

    @staticmethod
    def load_checkpoint(path) -> dict:
        """Load engine checkpoint from disk."""
        import pickle
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        logger.info("Checkpoint loaded: day_idx=%d, tickers=%d, "
                    "portfolio=%.0f",
                    checkpoint.get('day_idx', -1),
                    checkpoint.get('n_assets', 0),
                    checkpoint.get('portfolio_value', 0))
        return checkpoint

    def _optimize(self, method: str, mu: np.ndarray, cov: np.ndarray,
                  n_assets: int, pw: PortfolioWeights,
                  allow_short: bool = False,
                  alpha_scores: np.ndarray = None) -> np.ndarray:
        """Unified optimization dispatch."""
        try:
            if method == "risk_parity":
                if allow_short and alpha_scores is not None:
                    return pw.signal_aware_risk_parity(
                        cov, alpha_scores, allow_short=True)
                return pw.risk_parity(cov)
            elif method == "max_sharpe":
                return pw.maximum_sharpe(mu, cov, allow_short=allow_short)
            elif method == "capped_max_sharpe":
                # Cap per-asset mu shift to ±max_bps from cross-sectional mean
                max_bps = 5e-4  # 5 bps daily
                mu_mean = np.mean(mu)
                mu_capped = np.clip(mu, mu_mean - max_bps, mu_mean + max_bps)
                return pw.maximum_sharpe(mu_capped, cov, allow_short=allow_short)
            elif method == "blended_max_sharpe":
                # 70% risk parity base + 30% max_sharpe tilt
                w_rp = pw.risk_parity(cov)
                w_ms = pw.maximum_sharpe(mu, cov, allow_short=allow_short)
                blend = 0.3
                w = (1 - blend) * w_rp + blend * w_ms
                if not allow_short:
                    w = np.maximum(w, 0)
                s = w.sum()
                return w / s if abs(s) > 1e-12 else np.ones(n_assets) / n_assets
            elif method == "min_variance":
                return pw.minimum_variance(cov, allow_short=allow_short)
            else:
                return pw.equal_weight(n_assets)
        except Exception:
            return np.ones(n_assets) / n_assets

    def _compute_rebalance_cost(
        self, current_weights, new_weights, portfolio_value,
        tickers, volume_df, prices, returns, day_idx,
    ) -> float:
        """Compute rebalance cost using Almgren-Chriss or legacy model."""
        if not self.config.use_almgren_chriss:
            return self.cost_model.portfolio_rebalance_cost(
                current_weights, new_weights, portfolio_value)

        total = 0.0
        n = len(tickers)
        for i in range(n):
            trade_weight = new_weights[i] - current_weights[i]
            trade_value = trade_weight * portfolio_value
            if abs(trade_value) < 1.0:
                continue

            # Vol from recent returns (21 days scaled to bars)
            vol_bars = self.data_config.scale_window(21)
            vol_start = max(0, day_idx - vol_bars)
            asset_returns = returns.iloc[vol_start:day_idx, i].values
            sigma = float(np.std(asset_returns)) if len(asset_returns) > 5 else 0.02

            # ADV (average daily dollar volume)
            adv = 1e8  # default fallback
            if volume_df is not None:
                try:
                    vol_slice = volume_df.iloc[max(0, day_idx - vol_bars):day_idx, i]
                    price_val = float(prices.iloc[day_idx - 1, i])
                    adv = float(vol_slice.mean()) * price_val
                except Exception:
                    pass

            result = self.cost_model.almgren_chriss_cost(
                trade_value, sigma, adv,
                self.config.spread_bps, self.config.commission_bps,
                self.config.eta, self.config.gamma,
                self.config.impact_exponent)
            total += result["total"]

        return total

    def _self_referential_loop(
        self, new_weights, current_weights, mu, mu_baseline,
        cov, D_k_base, B, mean_volume,
        signal_gen, transition_detector, dynamics_config,
        pw, weight_method, alpha_scores, active_mask, n_assets,
        dynamics_log, date, kappa_current=None,
    ):
        """Self-referential feedback: weights → flow → D_k' → mu' → weights'.

        The core theoretical insight: proposed trades generate order flow,
        which creates factor dislocation, which changes the signal that
        generated the trades. Iteration finds the fixed point where the
        portfolio is consistent with its own market impact.

        Convergence is guaranteed when the contraction constant L < 1
        (Banach fixed-point theorem).

        Args:
            new_weights: N-vector of proposed weights from optimizer.
            current_weights: N-vector of current portfolio weights.
            mu: N-vector of current expected returns.
            mu_baseline: N-vector of expected returns before dynamics.
            cov: N×N covariance matrix.
            D_k_base: K-vector of base factor dislocation from volume data.
            B: N×K factor loadings.
            mean_volume: N-vector of rolling mean volume.
            signal_gen: SignalGenerator instance.
            transition_detector: TransitionDetector instance.
            dynamics_config: DynamicsConfig instance.
            pw: PortfolioWeights instance.
            weight_method: Optimization method string.
            alpha_scores: N-vector of alpha scores.
            active_mask: N-vector boolean mask for active assets.
            n_assets: Number of assets.
            dynamics_log: List to append diagnostics to.
            date: Current date for logging.
            kappa_current: Optional reflexivity coefficient.

        Returns:
            (new_weights, mu) after self-referential convergence.
        """
        max_iter = getattr(dynamics_config, 'self_ref_max_iter', 5)
        tol = getattr(dynamics_config, 'self_ref_tol', 1e-4)
        coeff = getattr(dynamics_config, 'flow_impact_coefficient', 0.01)
        phases = transition_detector._phase
        soft_phases = getattr(self, '_last_soft_phases', None)

        kappa_for_signal = (
            kappa_current
            if getattr(dynamics_config, 'kappa_signal_modulation', False)
            else None)

        D_k = D_k_base.copy()

        for iteration in range(max_iter):
            # 1. Compute flow impact from proposed trades
            weight_delta = new_weights - current_weights
            D_k_synthetic = signal_gen.estimate_flow_impact(
                weight_delta, B, mean_volume, coeff)

            # 2. Update D_k: base signal + flow-induced dislocation
            D_k_new = D_k_base + D_k_synthetic

            # 3. Check convergence
            dk_residual = float(np.linalg.norm(D_k_new - D_k))
            if dk_residual < tol:
                dynamics_log.append({
                    "date": str(date),
                    "type": "self_referential",
                    "converged": True,
                    "iterations": iteration + 1,
                    "residual": dk_residual,
                })
                break

            D_k = D_k_new

            # 4. Recompute mu from updated D_k (soft phases when available)
            if (self._soft_phase_estimator is not None
                    and soft_phases is not None
                    and len(D_k) <= soft_phases.shape[0]):
                _pw_map = {"dislocation": 1.0, "discovery": 0.5,
                           "convergence": -0.3, "normal": 0.3}
                _epw = self._soft_phase_estimator.expected_phase_weight(
                    soft_phases[:len(D_k)], _pw_map)
                _dk_eff = D_k.copy()
                if kappa_for_signal is not None:
                    _dk_eff = signal_gen.modulate_signal_by_kappa(
                        _dk_eff, kappa_for_signal,
                        dynamics_config.momentum_threshold,
                        dynamics_config.mean_reversion_threshold)
                mu_adj = signal_gen.dislocation_return_scale * (B @ (_dk_eff * _epw))
            else:
                mu_adj = signal_gen.compute_return_adjustment(
                    D_k, B, phases,
                    kappa=kappa_for_signal,
                    momentum_threshold=dynamics_config.momentum_threshold,
                    mean_reversion_threshold=dynamics_config.mean_reversion_threshold)
            mu = mu_baseline + mu_adj

            # 5. Re-optimize with updated mu
            filtered = not np.all(active_mask)
            opt_mu = mu
            opt_cov = cov
            opt_alpha = alpha_scores
            opt_current = current_weights
            opt_n = n_assets

            if filtered:
                active_idx = np.where(active_mask)[0]
                opt_n = len(active_idx)
                opt_cov = cov[np.ix_(active_idx, active_idx)]
                opt_alpha = alpha_scores[active_idx]
                opt_mu = mu[active_idx]
                opt_current = current_weights[active_idx]
                cs = opt_current.sum()
                if cs > 1e-12:
                    opt_current = opt_current / cs
                else:
                    opt_current = np.ones(opt_n) / opt_n

            has_alpha = np.any(np.abs(opt_alpha) > 1e-8)
            turnover_pen = self.config.turnover_penalty_bps
            tilt = (dynamics_config.alpha_tilt_strength
                    if dynamics_config else 0.5)
            # Phase-dynamic alpha tilt (v1 path)
            if (tilt > 0 and dynamics_config
                    and getattr(dynamics_config, 'xi_phase_scales', None)
                    and transition_detector is not None):
                priority = {"dislocation": 3, "convergence": 2,
                            "discovery": 1, "normal": 0}
                dominant = max(transition_detector._phase,
                               key=lambda p: priority.get(p, 0))
                phase_scale = dynamics_config.xi_phase_scales.get(dominant, 0.0)
                tilt = tilt * phase_scale

            if weight_method == "equal_weight":
                sub_weights = np.ones(opt_n) / opt_n
            elif (turnover_pen > 0
                  and weight_method not in ("max_sharpe",
                                            "capped_max_sharpe",
                                            "blended_max_sharpe",
                                            "min_variance",
                                            "equal_weight")):
                sub_weights = pw.turnover_penalized_risk_parity(
                    opt_cov, opt_current,
                    penalty_bps=turnover_pen,
                    alpha_scores=opt_alpha if has_alpha else None,
                    tilt_strength=tilt,
                )
            elif (has_alpha
                  and weight_method not in ("equal_weight",)):
                sub_weights = pw.signal_aware_risk_parity(
                    opt_cov, opt_alpha, tilt,
                    allow_short=self.config.allow_short)
            else:
                sub_weights = self._optimize(
                    weight_method, opt_mu, opt_cov, opt_n, pw,
                    allow_short=self.config.allow_short,
                    alpha_scores=opt_alpha)

            if filtered:
                new_weights = np.zeros(n_assets)
                new_weights[np.where(active_mask)[0]] = sub_weights
            else:
                new_weights = sub_weights
        else:
            # Did not converge within max_iter
            dynamics_log.append({
                "date": str(date),
                "type": "self_referential",
                "converged": False,
                "iterations": max_iter,
                "residual": dk_residual,
            })

        return new_weights, mu

    def _apply_dynamics_v2(
        self, mu, cov, returns_full, window_returns, tickers,
        day_idx, vol_aligned, rolling_mean_vol,
        open_aligned, high_aligned, low_aligned,
        momentum_series, prices,
        factor_tracker, causal_graph,
        transition_detector, signal_gen,
        dynamics_config, dynamics_log, date,
        dhat_smoothed=None,
        kappa_current=None,
        vix_series=None,
        yield_spread_series=None,
        dxy_series=None,
        ff_daily_factors=None,
        edge_updater=None,
        credit_spread_series=None,
        kbe_series=None,
    ):
        """Apply market dynamics with volume vector decomposition."""
        _t_total = time.perf_counter()

        # Fit PCA (or Graph VAE or graph-weighted PCA) on lookback window
        window_returns_df = pd.DataFrame(
            window_returns.values, columns=tickers,
            index=window_returns.index)

        # Prepare FF factor data for this window (POFM) — needed for both
        # Granger residualization and factor_tracker.fit()
        ff_window = None
        if ff_daily_factors is not None:
            try:
                ff_window = ff_daily_factors.reindex(
                    window_returns_df.index).ffill().fillna(0.0).values
            except Exception:
                ff_window = None

        # Build Granger causal graph first (needed for propagation_matrix)
        # Use FF-residualized returns when POFM active
        _t0 = time.perf_counter()
        granger_input = window_returns.values
        if ff_window is not None:
            try:
                F_gc = np.column_stack([np.ones(ff_window.shape[0]), ff_window])
                coeffs_gc = np.linalg.lstsq(F_gc, granger_input, rcond=None)[0]
                granger_input = granger_input - F_gc @ coeffs_gc
            except Exception:
                pass
        causal_graph.fit(granger_input)

        # Beta-Binomial edge update: modulate Granger F-stats by posterior belief
        if edge_updater is not None and causal_graph._p_values is not None:
            try:
                edge_updater.update(
                    causal_graph._graph, causal_graph._p_values,
                    dynamics_config.granger_p_threshold)
                causal_graph._graph = edge_updater.apply_to_graph(
                    causal_graph._graph)
            except Exception:
                pass

        correlation = CovarianceEstimator.to_correlation(cov)
        causal_M = causal_graph.build_propagation_matrix(
            alpha=dynamics_config.granger_alpha,
            correlation=correlation,
            safety_factor=getattr(dynamics_config, 'alpha_safety_factor', 0.7))
        _granger_ms = (time.perf_counter() - _t0) * 1000
        self._last_causal_M = causal_M  # Persist for daily predictive signals

        # Build institutional features (always when volume data available)
        inst_feats = None
        inst_v_pos_change = None
        if (vol_aligned is not None and open_aligned is not None
                and len(vol_aligned) > 0):
            try:
                T_win = len(window_returns)
                N_assets = len(tickers)
                ofi_buffer = np.zeros((T_win, N_assets))
                win_start = max(0, day_idx - T_win)

                mean_vol_idx_inst = min(day_idx - 1, len(rolling_mean_vol) - 1)
                if mean_vol_idx_inst >= 0:
                    mean_vol_inst = rolling_mean_vol.iloc[mean_vol_idx_inst].values
                else:
                    mean_vol_inst = np.ones(N_assets)

                for t_offset in range(T_win):
                    t_abs = win_start + t_offset
                    if (t_abs >= len(vol_aligned)
                            or t_abs >= len(prices)
                            or t_abs >= len(open_aligned)):
                        continue
                    close_t = prices.iloc[t_abs].values
                    open_t = open_aligned.iloc[t_abs].values
                    v_t = vol_aligned.iloc[t_abs].values
                    safe_mv = np.maximum(mean_vol_inst, 1.0)
                    ofi_buffer[t_offset] = (
                        np.sign(close_t - open_t) * v_t / safe_mv)

                inst_feats = ofi_buffer
                # v_pos_change: diff of rolling OFI sum
                ofi_lookback = getattr(dynamics_config, 'ofi_smoothing', 5)
                v_pos = np.zeros_like(ofi_buffer)
                for t in range(T_win):
                    start = max(0, t - ofi_lookback + 1)
                    v_pos[t] = ofi_buffer[start:t + 1].sum(axis=0)
                inst_v_pos_change = np.zeros_like(v_pos)
                inst_v_pos_change[1:] = v_pos[1:] - v_pos[:-1]
            except Exception as e:
                logger.warning("Institutional features failed on %s: %s", date, e)
                inst_feats = None
                inst_v_pos_change = None

        # Build macro data for continuous factor integration
        macro_data = None
        if getattr(dynamics_config, 'macro_factor_enabled', False):
            macro_data = {}
            window_dates = window_returns_df.index
            if vix_series is not None:
                try:
                    macro_data['vix'] = np.array([
                        float(vix_series.asof(d)) for d in window_dates])
                except (KeyError, TypeError):
                    pass
            if yield_spread_series is not None:
                try:
                    macro_data['yield_spread'] = np.array([
                        float(yield_spread_series.asof(d))
                        for d in window_dates])
                except (KeyError, TypeError):
                    pass
            if dxy_series is not None:
                try:
                    macro_data['dxy'] = np.array([
                        float(dxy_series.asof(d)) for d in window_dates])
                except (KeyError, TypeError):
                    pass
            if credit_spread_series is not None:
                try:
                    macro_data['credit_spread'] = np.array([
                        float(credit_spread_series.asof(d))
                        for d in window_dates])
                except (KeyError, TypeError):
                    pass
            if kbe_series is not None:
                try:
                    macro_data['bank_returns'] = np.array([
                        float(kbe_series.asof(d))
                        for d in window_dates])
                except (KeyError, TypeError):
                    pass
            if not macro_data:
                macro_data = None

        _t0 = time.perf_counter()
        inst_weight = getattr(
            dynamics_config, 'institutional_feature_weight', 0.1)
        coord_lb = getattr(
            dynamics_config, 'institutional_coordination_lookback', 21)

        factor_tracker.fit(
            window_returns_df,
            adjacency=causal_graph.graph if (
                getattr(dynamics_config, 'use_graph_vae', False)
                and causal_graph is not None) else None,
            propagation_matrix=causal_M,
            institutional_features=inst_feats,
            institutional_v_pos_change=inst_v_pos_change,
            institutional_feature_weight=inst_weight,
            institutional_coordination_lookback=coord_lb,
            macro_data=macro_data,
            ff_factor_data=ff_window)
        _pca_ms = (time.perf_counter() - _t0) * 1000

        # Endogeneity diagnostic (UECL bias estimate)
        if (getattr(dynamics_config, 'endogeneity_diagnostic', False)
                and ff_window is not None):
            try:
                from NR.dynamics.factor_tracker import FactorTracker as _FT
                diag = _FT.compute_endogeneity_diagnostic(
                    window_returns.values, ff_window)
                dynamics_log.append({
                    "date": str(date),
                    "type": "endogeneity_diagnostic",
                    "frobenius_norm": diag['frobenius_norm'],
                    "mean_abs_bias": [float(x) for x in diag['mean_abs_bias']],
                    "max_abs_bias": [float(x) for x in diag['max_abs_bias']],
                    "n_latent_used": diag['n_latent_used'],
                })
            except Exception as e:
                logger.warning("Endogeneity diagnostic failed on %s: %s", date, e)

        K = factor_tracker.n_active_factors
        B = factor_tracker.loadings

        # Resize transition detector if factor count changed (preserves CUSUM history)
        if transition_detector.n_factors != K:
            transition_detector.resize(K)

        # Get phase risk scale — soft phases override discrete when available
        soft_phases = getattr(self, '_last_soft_phases', None)
        if (self._soft_phase_estimator is not None and soft_phases is not None):
            phase_scale = self._soft_phase_estimator.soft_risk_scale(
                soft_phases, dynamics_config.phase_risk_scales)
        elif (factor_tracker._rbpf is not None
                and factor_tracker._rbpf._initialized):
            # RBPF soft regime blending
            regime_probs = factor_tracker._rbpf.regime_probabilities
            phase_scale = sum(
                regime_probs[phase]
                * dynamics_config.phase_risk_scales.get(phase, 1.0)
                for phase in regime_probs
            )
        else:
            phase_scale = transition_detector.get_risk_scale(
                dynamics_config.phase_risk_scales)

        # Adjust covariance (uniform phase scaling + Granger propagation)
        cov = signal_gen.compute_cov_adjustment(cov, phase_scale, causal_M)

        # Factor-specific covariance adjustment by phase
        # Channels dislocation signal through cov (which risk_parity uses)
        phases = transition_detector._phase
        eigenvalues = factor_tracker._current_eigenvalues
        if eigenvalues is not None and len(eigenvalues) == B.shape[1]:
            cov = signal_gen.compute_factor_cov_adjustment(
                cov, B, eigenvalues, phases)

        # RBPF covariance blending: reconstruct N×N from RBPF factor cov
        if (getattr(dynamics_config, 'rbpf_cov_blend', False)
                and factor_tracker._rbpf is not None
                and factor_tracker._rbpf._initialized):
            rbpf_factor_cov = factor_tracker.factor_covariance
            if rbpf_factor_cov is not None:
                ess = factor_tracker._rbpf.effective_sample_size
                ess_ratio = ess / factor_tracker._rbpf.M
                cov = signal_gen.compute_rbpf_blended_covariance(
                    cov, rbpf_factor_cov, B,
                    blend_weight=dynamics_config.rbpf_cov_blend_weight,
                    ess_ratio=(ess_ratio if dynamics_config.rbpf_cov_ess_adaptive
                               else 1.0))

        # Correlation Manifold Dynamics: use daily-evolved state at rebalance
        # Daily evolution happens in Stage 2.45; here we apply results to cov
        self._last_spectrum = None
        if self._manifold_evolution is not None and self._manifold_state is not None:
            try:
                # Grassmannian smoothing of loadings at rebalance
                if self._grassmannian_tracker is not None:
                    B = self._grassmannian_tracker.update(
                        B, eigenvalues[:K] if eigenvalues is not None else np.ones(K))

                # Lift evolved K×K factor covariance to N×N
                cov = self._manifold_evolution.lift_to_full(
                    self._manifold_state.sigma, B, cov)

                # Operator spectrum analysis + hedging
                if self._operator_spectrum is not None:
                    try:
                        F_k = factor_tracker.transition_matrix
                        if F_k is not None:
                            J = self._operator_spectrum.compute_jacobian(
                                F_k, causal_M, B,
                                damping=getattr(dynamics_config, 'fixed_point_damping', 0.5))
                            spectrum = self._operator_spectrum.analyze_spectrum(J)
                            self._manifold_state.operator_spectrum = spectrum["eigenvalues"]
                            self._last_spectrum = spectrum
                            dynamics_log.append({
                                "date": str(date),
                                "type": "operator_spectrum",
                                "spectral_radius": spectrum["spectral_radius"],
                                "n_marginal": spectrum["n_marginal"],
                                "n_unstable": spectrum["n_unstable"],
                                "dominant_halflife": spectrum["dominant_mode_halflife"],
                            })
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Manifold rebalance integration failed on %s: %s", date, e)

        # Compute D-hat for return adjustment
        _t0 = time.perf_counter()
        # When OFI D-hat is enabled, use the daily EMA-smoothed D-hat directly
        # (already computed in the day loop). Otherwise fall back to 20-day averaging.
        if (getattr(dynamics_config, 'use_ofi_dhat', False)
                and dhat_smoothed is not None
                and len(dhat_smoothed) == K):
            D_k_avg = dhat_smoothed
        else:
            vl = dynamics_config.volume_lookback
            recent_end = day_idx
            recent_start = max(0, day_idx - vl)

            D_k_avg = np.zeros(K)
            count = 0

            mean_vol_idx = min(day_idx - 1, len(rolling_mean_vol) - 1)
            if mean_vol_idx >= 0:
                mean_vol = rolling_mean_vol.iloc[mean_vol_idx].values
            else:
                mean_vol = np.ones(len(tickers))

            use_v3 = (getattr(dynamics_config, 'use_directional_volume', False)
                      and high_aligned is not None and low_aligned is not None)
            use_v2 = (open_aligned is not None and
                       dynamics_config.volume_decomposition)

            if not np.any(np.isnan(mean_vol)):
                for t_idx in range(recent_start, recent_end):
                    if t_idx >= len(returns_full) or t_idx >= len(vol_aligned):
                        continue
                    r_t = returns_full.iloc[t_idx].values
                    v_t = vol_aligned.iloc[t_idx].values

                    if use_v3:
                        close_t = prices.iloc[t_idx].values
                        open_t = open_aligned.iloc[t_idx].values
                        high_t = high_aligned.iloc[t_idx].values
                        low_t = low_aligned.iloc[t_idx].values
                        vol_comp = signal_gen.decompose_volume_v3(
                            close_t, open_t, high_t, low_t, v_t, mean_vol)
                        D_k_t = signal_gen.compute_volume_dislocation_v3(vol_comp, B)
                    elif use_v2:
                        close_t = prices.iloc[t_idx].values
                        open_t = open_aligned.iloc[t_idx].values
                        mom_t = (momentum_series.iloc[t_idx].values
                                 if momentum_series is not None
                                 else np.zeros(len(tickers)))
                        vol_comp = signal_gen.decompose_volume(
                            close_t, open_t, v_t, mean_vol, 1.0, 1.0, mom_t)
                        D_k_t = signal_gen.compute_volume_dislocation_v2(vol_comp, B)
                    else:
                        D_k_t = signal_gen.compute_volume_dislocation(
                            r_t, v_t, B, mean_vol)

                    D_k_avg += D_k_t
                    count += 1

            if count > 0:
                D_k_avg /= count

        # Apply return adjustment (with optional kappa signal modulation)
        phases = transition_detector._phase
        kappa_for_signal = (kappa_current
                            if getattr(dynamics_config, 'kappa_signal_modulation', False)
                            else None)
        # Use soft phase weights for return adjustment when available (Point E)
        soft_phases = getattr(self, '_last_soft_phases', None)
        if (self._soft_phase_estimator is not None
                and soft_phases is not None
                and len(D_k_avg) <= soft_phases.shape[0]):
            # Soft-weighted mu: mu_i = scale * sum_k(D_k_eff * B_ik * E[phase_weight_k])
            phase_weights_map = {
                "dislocation": 1.0, "discovery": 0.5,
                "convergence": -0.3, "normal": 0.3,
            }
            expected_pw = self._soft_phase_estimator.expected_phase_weight(
                soft_phases[:len(D_k_avg)], phase_weights_map)
            D_k_eff = D_k_avg.copy()
            if kappa_for_signal is not None:
                D_k_eff = signal_gen.modulate_signal_by_kappa(
                    D_k_eff, kappa_for_signal,
                    dynamics_config.momentum_threshold,
                    dynamics_config.mean_reversion_threshold)
            weighted_D = D_k_eff * expected_pw
            mu_adj = signal_gen.dislocation_return_scale * (B @ weighted_D)
        else:
            mu_adj = signal_gen.compute_return_adjustment(
                D_k_avg, B, phases,
                kappa=kappa_for_signal,
                momentum_threshold=dynamics_config.momentum_threshold,
                mean_reversion_threshold=dynamics_config.mean_reversion_threshold)
        mu = mu + mu_adj
        _volume_ms = (time.perf_counter() - _t0) * 1000

        # Fixed-point equilibrium iteration
        _fp_ms = 0.0
        if getattr(dynamics_config, 'fixed_point_enabled', False):
            try:
                _t_fp = time.perf_counter()
                from NR.dynamics.fixed_point import (
                    MarketOperator, ExtendedMarketOperator, FixedPointSolver)

                eigenvalues = factor_tracker._current_eigenvalues
                use_extended = getattr(dynamics_config,
                                       'fixed_point_iterate_dk', False)

                # The upstream build_propagation_matrix now clamps alpha
                # to safety_factor/rho(A), keeping rho(M) reasonable.
                # This secondary clamp is a safety net for edge cases.
                fp_damping = getattr(dynamics_config,
                                     'fixed_point_damping', 0.5)
                sr_M = float(np.max(np.abs(np.linalg.eigvals(causal_M))))
                effective_alpha = getattr(causal_graph, '_last_alpha', None)
                alpha_rho_product = getattr(
                    causal_graph, '_last_effective_product', None)

                if sr_M * fp_damping >= 0.99:
                    safe_scale = 0.95 / (sr_M * fp_damping)
                    causal_M_fp = causal_M * safe_scale
                    logger.debug(
                        "Fixed-point: clamped causal_M by %.3f "
                        "(sr=%.4f, damping=%.2f)", safe_scale, sr_M,
                        fp_damping)
                else:
                    causal_M_fp = causal_M

                # Store pre-iteration state as base for the operator
                mu_base_fp = mu.copy()
                cov_base_fp = cov.copy()

                if use_extended:
                    market_op = ExtendedMarketOperator(
                        signal_gen=signal_gen,
                        transition_detector=transition_detector,
                        causal_M=causal_M_fp,
                        B=B,
                        eigenvalues=(eigenvalues if eigenvalues is not None
                                     else np.ones(K)),
                        phase_risk_scales=dynamics_config.phase_risk_scales,
                        D_k=D_k_avg,
                        mu_base=mu_base_fp,
                        cov_base=cov_base_fp,
                        damping=fp_damping,
                    )
                else:
                    market_op = MarketOperator(
                        signal_gen=signal_gen,
                        transition_detector=transition_detector,
                        causal_M=causal_M_fp,
                        B=B,
                        eigenvalues=(eigenvalues if eigenvalues is not None
                                     else np.ones(K)),
                        phase_risk_scales=dynamics_config.phase_risk_scales,
                        D_k=D_k_avg,
                        mu_base=mu_base_fp,
                        cov_base=cov_base_fp,
                        damping=fp_damping,
                    )

                # Estimate noise cov and transition matrix for Lyapunov solver.
                # Lyapunov eqn: Sigma = F Sigma F^T + Q where F = alpha*A
                # (transition matrix, rho < 1), NOT M = (I-alpha*A)^{-1}
                # (propagation matrix, rho > 1).
                noise_cov = None
                lyap_transition = None
                if getattr(dynamics_config, 'use_lyapunov_equilibrium', False):
                    lyap_transition = causal_graph.transition_matrix
                    if lyap_transition is not None:
                        F_cov = lyap_transition @ cov @ lyap_transition.T
                        noise_cov = cov - F_cov
                        noise_cov = (noise_cov + noise_cov.T) / 2
                        min_eig = float(np.min(
                            np.linalg.eigvalsh(noise_cov)))
                        if min_eig < 1e-10:
                            noise_cov += (abs(min_eig) + 1e-8) * np.eye(
                                cov.shape[0])

                solver = FixedPointSolver(
                    max_iter=getattr(dynamics_config,
                                     'fixed_point_max_iter', 50),
                    tol=getattr(dynamics_config, 'fixed_point_tol', 1e-6),
                    use_lyapunov=getattr(dynamics_config,
                                         'use_lyapunov_equilibrium', False),
                    use_manifold=getattr(dynamics_config,
                                         'use_manifold_L', False),
                )

                mu, cov = solver.solve(
                    market_op, mu, cov, lyap_transition, noise_cov,
                    D_k_init=(D_k_avg if use_extended else None))

                _fp_ms = (time.perf_counter() - _t_fp) * 1000
                dynamics_log.append({
                    "date": str(date),
                    "type": "fixed_point",
                    "iterations": solver.n_iterations,
                    "residual": float(solver.residual),
                    "converged": solver.converged,
                    "spectral_radius_M": sr_M,
                    "damping": fp_damping,
                    "effective_alpha": effective_alpha,
                    "alpha_rho_product": alpha_rho_product,
                })
                # Update manifold dynamics equilibrium from fixed-point result
                if (self._manifold_evolution is not None
                        and solver.converged
                        and self._manifold_evolution.engine.initialized):
                    try:
                        sigma_eq_factor = B.T @ cov @ B
                        self._manifold_evolution.engine.update_equilibrium(sigma_eq_factor)
                    except Exception:
                        pass
            except Exception as e:
                _fp_ms = (time.perf_counter() - _t_fp) * 1000
                logger.warning("Fixed-point iteration failed on %s: %s",
                               date, e)

        # Log dynamics state
        _total_ms = (time.perf_counter() - _t_total) * 1000
        L_val = causal_graph.contraction_constant
        dynamics_log.append({
            "date": str(date),
            "type": "rebalance_dynamics",
            "n_factors": K,
            "granger_edges": causal_graph.n_edges,
            "phase": transition_detector.get_aggregate_phase(),
            "phase_scale": phase_scale,
            "loading_change": factor_tracker.loading_change_norm(),
            "contraction_L": L_val,
            "timing_ms": {
                "granger": _granger_ms,
                "pca_fit": _pca_ms,
                "volume": _volume_ms,
                "fixed_point": _fp_ms,
                "total": _total_ms,
            },
            "curvature_gate": getattr(self, '_curvature_gate_val', 1.0),
        })

        return mu, cov

    @property
    def results(self) -> Optional[dict]:
        return self._results
