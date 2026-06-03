import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize


class PortfolioWeights:
    """Heuristic and analytical portfolio weighting methods."""

    @staticmethod
    def equal_weight(n_assets: int) -> np.ndarray:
        return np.ones(n_assets) / n_assets

    @staticmethod
    def market_cap_weight(market_caps: pd.Series) -> np.ndarray:
        caps = np.array(market_caps, dtype=float)
        return caps / caps.sum()

    @staticmethod
    def inverse_volatility(returns: pd.DataFrame) -> np.ndarray:
        """Less volatile assets get higher weight."""
        vols = returns.std().values
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()

    @staticmethod
    def risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
        """Equal risk contribution via Spinu (2013) log-barrier formulation.

        Uses L-BFGS-B with analytical gradient for N > 50,
        Nelder-Mead for small N.
        """
        n = cov_matrix.shape[0]

        if n <= 50:
            def objective(y):
                y = np.abs(y)
                w = y / y.sum()
                port_vol = np.sqrt(w @ cov_matrix @ w)
                return port_vol - np.sum(np.log(w + 1e-16)) / n

            y0 = np.ones(n)
            result = minimize(objective, y0, method="Nelder-Mead",
                              options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12})
            y = np.abs(result.x)
            w = y / y.sum()
            return w

        # Gradient-based for large N via log-parameterization
        def obj_and_grad(log_y):
            y = np.exp(log_y)
            s = y.sum()
            w = y / s
            sigma_w = cov_matrix @ w
            port_var = w @ sigma_w
            port_vol = np.sqrt(max(port_var, 1e-20))

            obj = port_vol - np.sum(np.log(w + 1e-16)) / n

            dvol_dw = sigma_w / port_vol
            dlog_dw = -1.0 / (n * (w + 1e-16))
            dobj_dw = dvol_dw + dlog_dw

            # Chain rule through softmax: d/d(log_y_i) = y_i * (dobj/dw_i - w^T dobj/dw)
            wtg = w @ dobj_dw
            grad = y * (dobj_dw - wtg) / s

            return obj, grad

        log_y0 = np.zeros(n)
        result = minimize(obj_and_grad, log_y0, method="L-BFGS-B",
                          jac=True,
                          options={"maxiter": 500, "ftol": 1e-12})
        y = np.exp(result.x)
        w = y / y.sum()
        return w

    @staticmethod
    def minimum_variance(cov_matrix: np.ndarray, allow_short: bool = False) -> np.ndarray:
        """Minimum variance portfolio."""
        n = cov_matrix.shape[0]

        if not allow_short:
            ones = np.ones(n)
            try:
                w_unconstrained = np.linalg.solve(cov_matrix, ones)
            except np.linalg.LinAlgError:
                w_unconstrained = np.linalg.lstsq(cov_matrix, ones, rcond=None)[0]
            w_unconstrained /= w_unconstrained.sum()

            # If all positive, we're done
            if np.all(w_unconstrained >= -1e-10):
                return np.maximum(w_unconstrained, 0.0) / np.maximum(w_unconstrained, 0.0).sum()

            # Otherwise, constrained optimization with gradient
            def obj_and_grad(w):
                val = w @ cov_matrix @ w
                grad = 2.0 * cov_matrix @ w
                return val, grad

            x0 = np.ones(n) / n
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1,
                           "jac": lambda w: np.ones(n)}
            bounds = [(0.0, 1.0)] * n

            result = minimize(obj_and_grad, x0, method="SLSQP", jac=True,
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-15})
            return result.x
        else:
            ones = np.ones(n)
            try:
                w = np.linalg.solve(cov_matrix, ones)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(cov_matrix, ones, rcond=None)[0]
            return w / w.sum()

    @staticmethod
    def maximum_sharpe(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                       risk_free_rate: float = 0.05,
                       allow_short: bool = False,
                       periods_per_year: int = 252) -> np.ndarray:
        """Tangency portfolio: maximum Sharpe ratio."""
        n = len(expected_returns)
        rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = expected_returns - rf_daily

        # Try analytical solution first: w* = Sigma^{-1} @ excess_returns
        try:
            w_analytical = np.linalg.solve(cov_matrix, excess)
        except np.linalg.LinAlgError:
            w_analytical = np.linalg.lstsq(cov_matrix, excess, rcond=None)[0]
        w_analytical /= w_analytical.sum()

        if not allow_short and np.any(w_analytical < -1e-10):
            # Need constrained optimization
            def neg_sharpe(w):
                port_return = w @ expected_returns
                port_vol = np.sqrt(w @ cov_matrix @ w)
                if port_vol < 1e-10:
                    return 0.0
                return -(port_return - rf_daily) / port_vol

            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            bounds = [(0.0, 1.0)] * n
            x0 = np.ones(n) / n

            result = minimize(neg_sharpe, x0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-15})
            return result.x

        if not allow_short:
            return np.maximum(w_analytical, 0.0) / np.maximum(w_analytical, 0.0).sum()
        return w_analytical

    @staticmethod
    def risk_parity_long_short(
        cov_matrix: np.ndarray,
        alpha_scores: np.ndarray,
        short_threshold_z: float = -0.5,
        net_exposure: float = 1.0,
    ) -> np.ndarray:
        """Risk parity with short positions for negative-alpha assets.

        Computes standard risk-parity for weight magnitudes, then flips
        sign for assets with alpha z-score below short_threshold_z.
        Renormalizes so sum(w) = net_exposure.

        Args:
            cov_matrix: N x N covariance matrix.
            alpha_scores: N-vector of cross-sectional z-scores.
            short_threshold_z: Z-score cutoff below which assets go short.
            net_exposure: Target net exposure (sum of weights).

        Returns:
            N-vector of signed weights summing to net_exposure.
        """
        w_rp = PortfolioWeights.risk_parity(cov_matrix)
        alpha = np.asarray(alpha_scores, dtype=float)

        # Flip sign for low-alpha assets
        short_mask = alpha < short_threshold_z
        w_signed = w_rp.copy()
        w_signed[short_mask] *= -1.0

        total = w_signed.sum()
        if abs(total) < 1e-12:
            return w_rp * net_exposure
        return w_signed * (net_exposure / total)

    @staticmethod
    def signal_aware_risk_parity(
        cov_matrix: np.ndarray,
        alpha_scores: np.ndarray,
        tilt_strength: float = 0.5,
        allow_short: bool = False,
    ) -> np.ndarray:
        """Risk parity with alpha signal tilt.

        Computes standard risk-parity weights, then applies exponential
        tilt proportional to alpha z-scores:
            w_tilted_i = w_rp_i * exp(tilt_strength * z_i)
            w_final = w_tilted / sum(w_tilted)

        When allow_short=True, delegates to risk_parity_long_short.

        Args:
            cov_matrix: N x N covariance matrix.
            alpha_scores: N-vector of cross-sectional z-scores.
            tilt_strength: Aggressiveness [0, inf). 0 = pure risk parity.
            allow_short: If True, use long-short risk parity.

        Returns:
            N-vector of tilted weights.
        """
        if allow_short:
            return PortfolioWeights.risk_parity_long_short(
                cov_matrix, alpha_scores)
        w_rp = PortfolioWeights.risk_parity(cov_matrix)
        tilt = np.exp(tilt_strength * np.asarray(alpha_scores))
        w_tilted = w_rp * tilt
        total = w_tilted.sum()
        if total < 1e-12:
            return w_rp
        return w_tilted / total

    @staticmethod
    def robust_maximum_sharpe(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        kappa: float = 0.1,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Distributionally Robust Optimization (DRO) maximum Sharpe.

        Solves worst-case optimization within a Wasserstein ball of
        radius kappa around the estimated return distribution:

            max_w  min_{P in B(P_hat, kappa)}  E_P[w^T r] / sqrt(w^T Sigma w)

        The dual formulation (Esfahani & Kuhn 2018) reduces to
        penalizing the mean return by kappa * ||w||_2:

            max_w  (w^T mu - kappa * ||w||_2) / sqrt(w^T Sigma w)

        Higher kappa = more robust (lower return, less sensitivity to
        estimation error). kappa=0 recovers standard maximum Sharpe.

        The robustness parameter should scale with the contraction
        constant L: kappa_eff = kappa * (1 + L) when regime is unstable.

        Args:
            expected_returns: N-vector of expected returns.
            cov_matrix: N x N covariance matrix.
            kappa: Wasserstein ball radius (robustness parameter).
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading periods per year.

        Returns:
            N-vector of DRO-optimal weights (positive, sums to 1).
        """
        n = len(expected_returns)
        rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess = expected_returns - rf_daily

        def neg_robust_sharpe(w):
            port_return = w @ excess
            port_vol = np.sqrt(max(w @ cov_matrix @ w, 1e-20))
            w_norm = np.sqrt(max(w @ w, 1e-20))
            # Robust return = nominal return - kappa * ||w||
            robust_return = port_return - kappa * w_norm
            return -robust_return / port_vol

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(neg_robust_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-15})

        w = result.x
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total < 1e-12:
            return np.ones(n) / n
        return w / total

    @staticmethod
    def clip_and_renormalize(
        weights: np.ndarray,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        per_asset_max: np.ndarray = None,
        net_exposure: float = 1.0,
    ) -> np.ndarray:
        """Clip weights to per-asset bounds and renormalize.

        Iteratively clips and redistributes excess weight among uncapped
        assets (max 10 rounds to converge). Normalizes to net_exposure.
        """
        w = weights.copy()
        n = len(w)
        upper = np.full(n, max_weight) if per_asset_max is None else per_asset_max.copy()
        upper = np.minimum(upper, max_weight)
        lower = np.full(n, min_weight)

        for _ in range(10):
            w = np.clip(w, lower, upper)
            total = w.sum()
            if abs(total) < 1e-12:
                return np.ones(n) / n * net_exposure
            w *= net_exposure / total
            if not np.any((w > upper + 1e-10) | (w < lower - 1e-10)):
                break

        return w

    @staticmethod
    def turnover_penalized_risk_parity(
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        penalty_bps: float = 50.0,
        alpha_scores: np.ndarray = None,
        tilt_strength: float = 0.5,
    ) -> np.ndarray:
        """Risk parity blended toward current weights to reduce turnover.

        Computes target via risk_parity (or signal_aware variant), then
        blends with current weights: w = blend * target + (1-blend) * current.
        blend = 1 / (1 + penalty_bps / 100).
        """
        n = cov_matrix.shape[0]
        if alpha_scores is not None and np.any(np.abs(alpha_scores) > 1e-8):
            target = PortfolioWeights.signal_aware_risk_parity(
                cov_matrix, alpha_scores, tilt_strength)
        else:
            target = PortfolioWeights.risk_parity(cov_matrix)

        blend = 1.0 / (1.0 + penalty_bps / 100.0)
        w = blend * target + (1.0 - blend) * current_weights
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total < 1e-12:
            return target
        return w / total

    @staticmethod
    def xi_alpha_tilt(
        cov_matrix: np.ndarray,
        alpha_scores: np.ndarray,
        dhat_abs: np.ndarray,
        tilt_strength: float = 0.5,
        z_clip: float = 2.0,
        current_weights: np.ndarray = None,
        blend_rate: float = 0.67,
        rebalance_band: float = 0.0,
    ) -> np.ndarray:
        """Alpha-aware inverse-vol weighting with exponential tilt and soft conviction gate.

        w_i ∝ (1/σ_i) × exp(λ × clip(z_i, -c, c)) × gate(|D̂_i|)

        - Exponential tilt guarantees positive weights and smooth scaling
        - Soft gate via rank(|D̂|) replaces hard conviction cutoff
        - Turnover blend with previous weights controls trading costs
        - Rebalance band: only trade positions deviating > band from current
        - When tilt_strength=0: degrades to inv_vol × dhat_gate (no alpha)
        """
        n = cov_matrix.shape[0]

        # 1. Inverse volatility base (from covariance diagonal)
        vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        inv_vol = 1.0 / vols

        # 2. Exponential alpha tilt with z-score clipping
        z_clipped = np.clip(np.asarray(alpha_scores, dtype=float), -z_clip, z_clip)
        exp_tilt = np.exp(tilt_strength * z_clipped)

        # 3. Soft conviction gate: rank(|D̂|) mapped to [0.5, 1.5]
        dhat = np.asarray(dhat_abs, dtype=float)
        if n > 1:
            dhat_rank = np.argsort(np.argsort(dhat)) / (n - 1)
        else:
            dhat_rank = np.array([0.5])
        dhat_gate = 0.5 + dhat_rank

        # 4. Combine and normalize
        raw = inv_vol * exp_tilt * dhat_gate
        total = raw.sum()
        if total < 1e-12:
            target = np.ones(n) / n
        else:
            target = raw / total

        # 5. Turnover blend with previous weights
        if current_weights is not None and blend_rate < 1.0:
            blended = blend_rate * target + (1.0 - blend_rate) * current_weights
            blended = np.maximum(blended, 0.0)
            s = blended.sum()
            if s < 1e-12:
                return target
            blended = blended / s

            # 6. Rebalance band: only trade positions with large deviations
            if rebalance_band > 0:
                deviation = np.abs(blended - current_weights)
                hold_mask = deviation <= rebalance_band
                if hold_mask.any() and not hold_mask.all():
                    result = blended.copy()
                    result[hold_mask] = current_weights[hold_mask]
                    rs = result.sum()
                    if rs > 1e-12:
                        return result / rs
            return blended

        return target

    @staticmethod
    def xi_alpha_overlay(
        cov_matrix: np.ndarray,
        alpha_scores: np.ndarray,
        long_pct: float = 0.20,
        short_pct: float = 0.20,
    ) -> np.ndarray:
        """Long/short overlay: top quintile long, bottom quintile short, vol-weighted.

        Returns dollar-neutral weights (sum ≈ 0).
        Long sleeve: top long_pct by alpha_scores, inverse-vol weighted.
        Short sleeve: bottom short_pct by alpha_scores, inverse-vol weighted.
        """
        n = len(alpha_scores)
        if n < 4:
            return np.zeros(n)
        scores = np.asarray(alpha_scores, dtype=float)
        ranks = np.argsort(np.argsort(scores)) / max(n - 1, 1)

        long_mask = ranks >= (1.0 - long_pct)
        short_mask = ranks <= short_pct

        vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        inv_vol = 1.0 / vols

        long_w = np.where(long_mask, inv_vol, 0.0)
        short_w = np.where(short_mask, inv_vol, 0.0)

        ls = long_w.sum()
        ss = short_w.sum()
        if ls > 1e-12:
            long_w /= ls
        if ss > 1e-12:
            short_w /= ss

        return long_w - short_w

    @staticmethod
    def conviction_direction_risk(
        conviction: np.ndarray,
        direction: np.ndarray,
        inv_vol: np.ndarray,
        conviction_max: float = 3.0,
        short_exposure: float = 0.0,
        current_weights: np.ndarray = None,
        max_weight_change: float = 0.20,
        holding_days: np.ndarray = None,
        min_hold_days: int = 15,
        conviction_percentile: float = 0.30,
        direction_threshold: float = 0.3,
    ) -> np.ndarray:
        """Conviction x Direction x Risk portfolio construction.

        w_i = C_i * D_i * R_i with conviction gating and controlled
        long/short normalization.

        Concentration is achieved through two gates:
        - conviction_percentile: only top X% by conviction get allocated
        - direction_threshold: only assets with |direction_z| > threshold

        Direction magnitude is preserved (not reduced to sign) so
        stronger signals get proportionally more weight.

        Args:
            conviction: Per-asset conviction (N,). Rank-normalized internally.
            direction: Per-asset directional z-score (N,). Magnitude preserved.
            inv_vol: Per-asset inverse volatility 1/sigma (N,).
            conviction_max: Max conviction after rank scaling.
            short_exposure: Short sleeve gross exposure (0 = long-only).
            current_weights: Previous weights for turnover control (N,).
            max_weight_change: Max absolute weight change per rebalance.
            holding_days: Days each position has been held (N,).
            min_hold_days: Don't exit positions held fewer than this.
            conviction_percentile: Fraction of assets to keep (0.30 = top 30%).
            direction_threshold: Min |direction_z| to take a position.

        Returns:
            N-vector of signed weights. Long sums to ~1.0,
            short sums to ~-short_exposure.
        """
        N = len(conviction)
        if N == 0:
            return np.array([])

        # Conviction: rank-normalize to [0, conviction_max]
        ranks = sp_stats.rankdata(conviction) / N
        C = ranks * conviction_max

        # Conviction gate: zero out bottom (1 - percentile)
        if 0 < conviction_percentile < 1.0:
            cutoff_rank = 1.0 - conviction_percentile
            C[ranks <= cutoff_rank] = 0.0

        # Direction: preserve magnitude, apply threshold
        D = direction.copy()
        D[np.abs(direction) < direction_threshold] = 0.0

        # Risk: inverse volatility (already provided)
        R = np.maximum(inv_vol, 0.0)

        # Raw weights: conviction × direction × risk
        raw = C * D * R

        # Separate long/short
        long_mask = raw > 0
        short_mask = raw < 0
        w = np.zeros(N)

        # Normalize long bucket
        long_sum = np.sum(raw[long_mask])
        if long_sum > 1e-12:
            target_long = 1.0 + short_exposure if short_exposure > 0 else 1.0
            w[long_mask] = raw[long_mask] / long_sum * target_long

        # Normalize short bucket
        if short_exposure > 0:
            short_sum = np.sum(np.abs(raw[short_mask]))
            if short_sum > 1e-12:
                w[short_mask] = raw[short_mask] / short_sum * short_exposure
            elif long_sum > 1e-12:
                # No shorts signaled — rescale longs to net=1.0
                w[long_mask] = raw[long_mask] / long_sum * 1.0
        else:
            # Long-only: zero out any negative weights
            w = np.maximum(w, 0.0)
            ws = w.sum()
            if ws > 1e-12:
                w /= ws

        # Min-hold constraint: don't exit positions held < min_hold_days
        if current_weights is not None and holding_days is not None:
            for i in range(N):
                if (abs(current_weights[i]) > 1e-8
                        and holding_days[i] < min_hold_days):
                    # Keep current position if we'd be exiting or flipping
                    if (abs(w[i]) < 1e-8
                            or np.sign(w[i]) != np.sign(current_weights[i])):
                        w[i] = current_weights[i]

        # Max weight change constraint
        if current_weights is not None and max_weight_change > 0:
            delta = w - current_weights
            clipped = np.clip(delta, -max_weight_change, max_weight_change)
            w = current_weights + clipped

        return w

    @staticmethod
    def core_satellite_weights(
        cov_matrix: np.ndarray,
        tickers: list,
        core_tickers: list,
        core_pct: float = 0.50,
        core_method: str = "inverse_volatility",
        satellite_cov: np.ndarray = None,
        satellite_alpha: np.ndarray = None,
        tilt_strength: float = 0.5,
        current_weights: np.ndarray = None,
        turnover_penalty_bps: float = 0.0,
    ) -> np.ndarray:
        """Core-satellite allocation.

        Allocates core_pct to core_tickers (split by core_method),
        then (1-core_pct) to remaining tickers via risk parity.
        """
        n = len(tickers)
        weights = np.zeros(n)
        sat_pct = 1.0 - core_pct

        core_idx = [i for i, t in enumerate(tickers) if t in core_tickers]
        sat_idx = [i for i, t in enumerate(tickers) if t not in core_tickers]

        if not core_idx:
            return PortfolioWeights.risk_parity(cov_matrix)

        # Core weights
        n_core = len(core_idx)
        if core_method == "inverse_volatility":
            core_vol = np.array([np.sqrt(cov_matrix[i, i]) for i in core_idx])
            inv_vol = 1.0 / np.maximum(core_vol, 1e-12)
            core_w = inv_vol / inv_vol.sum()
        elif core_method == "risk_parity":
            core_cov = cov_matrix[np.ix_(core_idx, core_idx)]
            core_w = PortfolioWeights.risk_parity(core_cov)
        else:
            core_w = np.ones(n_core) / n_core

        for i, idx in enumerate(core_idx):
            weights[idx] = core_pct * core_w[i]

        # Satellite weights
        if sat_idx:
            n_sat = len(sat_idx)
            sat_cov = cov_matrix[np.ix_(sat_idx, sat_idx)]
            sat_alpha = satellite_alpha[sat_idx] if satellite_alpha is not None else None
            has_alpha = sat_alpha is not None and np.any(np.abs(sat_alpha) > 1e-8)

            if turnover_penalty_bps > 0 and current_weights is not None:
                sat_current = current_weights[sat_idx]
                cs = sat_current.sum()
                sat_current = sat_current / cs if cs > 1e-12 else np.ones(n_sat) / n_sat
                sat_w = PortfolioWeights.turnover_penalized_risk_parity(
                    sat_cov, sat_current,
                    penalty_bps=turnover_penalty_bps,
                    alpha_scores=sat_alpha if has_alpha else None,
                    tilt_strength=tilt_strength)
            elif has_alpha:
                sat_w = PortfolioWeights.signal_aware_risk_parity(
                    sat_cov, sat_alpha, tilt_strength)
            else:
                sat_w = PortfolioWeights.risk_parity(sat_cov)

            for i, idx in enumerate(sat_idx):
                weights[idx] = sat_pct * sat_w[i]

        return weights
