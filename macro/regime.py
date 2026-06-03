from typing import Optional

import numpy as np
import pandas as pd

from NR.config import MacroConfig


class RegimeDetector:
    """Threshold-based market regime classifier.

    Regimes: expansion, contraction, crisis, recovery.
    Uses VIX level, yield curve slope, and VIX trend.
    """

    def __init__(self, config: MacroConfig = None):
        self.config = config or MacroConfig()
        self._history: list[tuple] = []
        self._prev_regime: Optional[str] = None

    def classify(self, vix: float, yield_spread: float = 0.5,
                 vix_sma_ratio: float = 1.0,
                 credit_spread: float = None,
                 bank_drawdown: float = None) -> str:
        """Classify current regime from macro signals.

        Args:
            vix: Current VIX level.
            yield_spread: 10Y-2Y spread (negative = inversion).
            vix_sma_ratio: VIX / VIX_SMA50. <1 means VIX declining.
            credit_spread: HY OAS in percentage points (e.g. 5.0 = 500bp).
            bank_drawdown: KBE 63-day peak-to-trough drawdown (0.15 = 15%).
        """
        # Credit spread override — catches stress before VIX spikes
        if credit_spread is not None:
            cs_crisis = getattr(self.config, 'regime_credit_spread_crisis', 8.0)
            cs_stress = getattr(self.config, 'regime_credit_spread_stress', 5.0)
            if credit_spread >= cs_crisis:
                self._prev_regime = "crisis"
                return "crisis"
            if credit_spread >= cs_stress:
                if self._prev_regime == "crisis" and vix_sma_ratio < 1.0:
                    self._prev_regime = "recovery"
                    return "recovery"
                self._prev_regime = "contraction"
                return "contraction"

        # Bank stress amplifier — catches bank-led stress (e.g. SVB 2023)
        if bank_drawdown is not None:
            bd_threshold = getattr(
                self.config, 'regime_bank_drawdown_threshold', 0.15)
            if bank_drawdown > bd_threshold and self._prev_regime != "crisis":
                self._prev_regime = "contraction"
                return "contraction"

        if vix >= self.config.regime_vix_crisis:
            regime = "crisis"
        elif (vix >= self.config.regime_vix_high or
              yield_spread < self.config.regime_yield_curve_inversion):
            if self._prev_regime == "crisis" and vix_sma_ratio < 1.0:
                regime = "recovery"
            else:
                regime = "contraction"
        elif self._prev_regime in ("crisis", "contraction") and vix_sma_ratio < 0.9:
            regime = "recovery"
        else:
            regime = "expansion"

        self._prev_regime = regime
        return regime

    def classify_series(self, vix_series: pd.Series,
                        yield_spread_series: pd.Series = None,
                        credit_spread_series: pd.Series = None,
                        kbe_series: pd.Series = None) -> pd.Series:
        """Classify regime for each date in a time series."""
        if yield_spread_series is None:
            yield_spread_series = pd.Series(0.5, index=vix_series.index)

        vix_sma = vix_series.rolling(50, min_periods=1).mean()
        bd_window = getattr(
            self.config, 'regime_bank_drawdown_window', 63)
        regimes = []
        self._prev_regime = None

        for date in vix_series.index:
            vix = vix_series.loc[date]
            ys = yield_spread_series.loc[date] if date in yield_spread_series.index else 0.5
            ratio = vix / vix_sma.loc[date] if vix_sma.loc[date] > 0 else 1.0
            cs = None
            if credit_spread_series is not None and date in credit_spread_series.index:
                cs = credit_spread_series.loc[date]
            bd = None
            if kbe_series is not None:
                kbe_w = kbe_series.loc[:date].tail(bd_window)
                if len(kbe_w) >= 10:
                    peak = kbe_w.max()
                    if peak > 0:
                        bd = float((peak - kbe_w.iloc[-1]) / peak)
            regime = self.classify(vix, ys, ratio, credit_spread=cs,
                                   bank_drawdown=bd)
            regimes.append(regime)

        return pd.Series(regimes, index=vix_series.index, name="regime")

    def get_cov_multiplier(self, regime: str) -> float:
        return self.config.regime_cov_multipliers.get(regime, 1.0)

    def get_return_adjustment(self, regime: str) -> float:
        return self.config.regime_return_adjustments.get(regime, 0.0)


class MarkovRegimeDetector:
    """Gaussian observation HMM with 4 hidden states.

    States: expansion, contraction, crisis, recovery.
    Observations: [VIX, yield_spread, momentum, credit_spread].

    Uses forward algorithm (no hmmlearn dependency):
    - Transition matrix A estimated from RegimeDetector history
    - Emission model: Gaussian per state (mean, var)
    - Forward pass: alpha_t(s) = p(obs|s) * sum_s'(A[s',s] * alpha_{t-1}(s'))
    """

    def __init__(self, n_states: int = 4, burn_in: int = 63):
        self.n_states = n_states
        self.states = ["expansion", "contraction", "crisis", "recovery"]
        # Initialize uniform transition matrix
        self.A = np.ones((n_states, n_states)) / n_states
        # Emission parameters: mean and var per state per observable
        # Columns: [VIX, yield_spread, momentum, credit_spread(HY OAS)]
        self._emission_mean = np.array([
            [15.0, 0.5, 0.05, 3.5],    # expansion: tight spreads
            [25.0, 0.0, -0.02, 5.0],   # contraction: widening
            [35.0, -0.3, -0.10, 10.0],  # crisis: very wide
            [22.0, 0.3, 0.03, 4.5],    # recovery: normalizing
        ])
        self._emission_var = np.array([
            [25.0, 0.25, 0.01, 1.0],
            [36.0, 0.25, 0.01, 4.0],
            [100.0, 0.25, 0.04, 25.0],
            [25.0, 0.25, 0.01, 2.0],
        ])
        self._belief = np.ones(n_states) / n_states
        self._history: list = []
        self._burn_in = burn_in
        self._fitted = False

    def fit(self, vix_series, yield_series=None, momentum_series=None,
            credit_spread_series=None):
        """Estimate emission parameters from historical data.

        Uses VIX-threshold regime labels for supervised initialization.
        """
        if len(vix_series) < self._burn_in:
            return

        # Label regimes using VIX thresholds (supervised initialization)
        labels = []
        for v in vix_series:
            if v >= 35:
                labels.append(2)  # crisis
            elif v >= 25:
                labels.append(1)  # contraction
            elif v >= 20:
                labels.append(3)  # recovery
            else:
                labels.append(0)  # expansion

        labels = np.array(labels)
        vix_arr = np.asarray(vix_series, dtype=float)
        yield_arr = (np.asarray(yield_series, dtype=float)
                     if yield_series is not None
                     else np.full(len(vix_arr), 0.5))
        mom_arr = (np.asarray(momentum_series, dtype=float)
                   if momentum_series is not None
                   else np.zeros(len(vix_arr)))
        cs_arr = (np.asarray(credit_spread_series, dtype=float)
                  if credit_spread_series is not None
                  else np.full(len(vix_arr), 4.0))

        obs = np.column_stack([vix_arr, yield_arr, mom_arr, cs_arr])

        # Estimate emission parameters per state
        for s in range(self.n_states):
            mask = labels == s
            if mask.sum() > 2:
                self._emission_mean[s] = obs[mask].mean(axis=0)
                self._emission_var[s] = np.maximum(obs[mask].var(axis=0), 1e-4)

        # Estimate transition matrix from label sequence
        counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(labels)):
            counts[labels[t - 1], labels[t]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        # Unobserved states get uniform transition probability
        zero_rows = (row_sums == 0).ravel()
        row_sums[row_sums == 0] = 1.0
        self.A = counts / row_sums
        self.A[zero_rows] = 1.0 / self.n_states

        self._fitted = True

    def update(self, vix: float, yield_spread: float = 0.5,
               momentum: float = 0.0, credit_spread: float = 4.0) -> str:
        """Forward step: update belief state given new observation."""
        obs = np.array([vix, yield_spread, momentum, credit_spread])
        # Emission likelihood per state
        likelihoods = np.array([
            self._gaussian_likelihood(obs, s) for s in range(self.n_states)
        ])
        # Forward: predict then update
        predicted = self.A.T @ self._belief
        self._belief = likelihoods * predicted
        total = self._belief.sum()
        if total > 1e-12:
            self._belief /= total
        else:
            self._belief = np.ones(self.n_states) / self.n_states
        self._history.append(self.current_regime)
        return self.current_regime

    def _gaussian_likelihood(self, obs: np.ndarray, state: int) -> float:
        """Compute multivariate Gaussian likelihood (diagonal cov)."""
        diff = obs - self._emission_mean[state]
        var = self._emission_var[state]
        # Product of univariate Gaussians (diagonal covariance)
        log_lik = -0.5 * np.sum(diff ** 2 / var + np.log(2 * np.pi * var))
        return np.exp(np.clip(log_lik, -500, 0))

    @property
    def current_regime(self) -> str:
        """Most likely current regime."""
        return self.states[np.argmax(self._belief)]

    @property
    def regime_probabilities(self) -> dict:
        """Current belief distribution over regimes."""
        return dict(zip(self.states, self._belief))
