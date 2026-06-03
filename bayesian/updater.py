from typing import Optional

import numpy as np

from NR.config import BayesianConfig
from NR.bayesian.forecast_store import ForecastStore


class BayesianReturnUpdater:
    """Bayesian shrinkage of expected returns toward realized outcomes.

    Prior: model-estimated expected returns.
    Likelihood: realized returns from recent periods.
    Posterior: weighted combination, with weights determined by forecast accuracy.

    The posterior expected return for each asset is:
        mu_posterior = tau * mu_prior + (1 - tau) * mu_realized

    where tau starts at config.prior_weight and decreases as more realized
    data accumulates and the realized mean is more reliable.
    """

    def __init__(self, config: BayesianConfig = None,
                 forecast_store: ForecastStore = None,
                 periods_per_year: int = 252):
        self.config = config or BayesianConfig()
        self.ppy = periods_per_year
        self.store = forecast_store or ForecastStore(self.config.store_path)
        self._posterior: Optional[np.ndarray] = None
        self._prior: Optional[np.ndarray] = None
        self._realized_mean: Optional[np.ndarray] = None

    def update(self, prior_returns: np.ndarray,
               realized_returns: np.ndarray = None,
               n_observations: int = None) -> np.ndarray:
        """Compute posterior expected returns.

        Args:
            prior_returns: Model-estimated expected daily returns (n_assets,).
            realized_returns: Mean realized daily returns (n_assets,).
                If None, uses forecast store history.
            n_observations: Number of observations backing the realized mean.

        Returns:
            Posterior expected daily returns (n_assets,).
        """
        self._prior = prior_returns

        if realized_returns is None:
            realized_returns = self._realized_from_store(len(prior_returns))
            if realized_returns is None:
                self._posterior = prior_returns.copy()
                return self._posterior

        self._realized_mean = realized_returns

        # Adaptive tau: shrink toward realized as observations grow.
        # Use sqrt decay (gentle) rather than exponential per-day
        # (which collapses tau to ~0 for any real dataset).
        tau = self.config.prior_weight
        if n_observations is not None and n_observations > 0:
            excess = max(0, n_observations - self.config.min_observations)
            tau = tau / np.sqrt(1 + excess / self.ppy)

        tau = max(0.1, min(tau, 0.95))  # Clamp to [0.1, 0.95]

        self._posterior = tau * prior_returns + (1 - tau) * realized_returns
        return self._posterior

    def _realized_from_store(self, n_assets: int) -> Optional[np.ndarray]:
        """Extract realized return mean from forecast store."""
        errors = self.store.get_errors()
        if len(errors) < self.config.min_observations:
            return None

        # Use recent realized values
        recent = errors[-self.config.min_observations:]
        realized_list = [r["realized"] for r in recent if r["realized"] is not None]
        if not realized_list:
            return None

        # Average across periods, handling varying lengths
        arrays = [np.array(r) for r in realized_list if len(r) == n_assets]
        if not arrays:
            return None

        return np.mean(arrays, axis=0)

    def update_with_accuracy_weights(
        self,
        prior_returns: np.ndarray,
        recent_forecasts: list[np.ndarray],
        recent_realized: list[np.ndarray],
    ) -> np.ndarray:
        """Update using accuracy-weighted historical forecasts.

        More accurate past forecasts contribute more to the posterior.
        """
        self._prior = prior_returns

        if not recent_forecasts or not recent_realized:
            self._posterior = prior_returns.copy()
            return self._posterior

        n = len(recent_forecasts)
        decay = self.config.decay_factor

        # Compute per-period accuracy
        weights = []
        for i, (fc, real) in enumerate(zip(recent_forecasts, recent_realized)):
            mse = np.mean((np.array(fc) - np.array(real)) ** 2)
            accuracy = 1.0 / (1.0 + mse)
            recency = decay ** (n - 1 - i)
            weights.append(accuracy * recency)

        weights = np.array(weights)
        if weights.sum() == 0:
            self._posterior = prior_returns.copy()
            return self._posterior
        weights = weights / weights.sum()

        # Weighted realized mean
        realized_mean = np.zeros_like(prior_returns)
        for w, real in zip(weights, recent_realized):
            realized_mean += w * np.array(real)

        self._realized_mean = realized_mean

        # Blend with prior
        tau = self.config.prior_weight
        self._posterior = tau * prior_returns + (1 - tau) * realized_mean
        return self._posterior

    def store_forecast(self, tickers: list[str], forecast: np.ndarray,
                       metadata: dict = None):
        """Store a forecast for future comparison."""
        self.store.store_forecast(tickers, forecast, metadata)

    def record_realized(self, date: str, realized: np.ndarray):
        """Record realized returns and update the store."""
        self.store.update_realized(date, realized)

    def flush(self):
        """Flush deferred writes in the forecast store to disk."""
        self.store.flush()

    @property
    def posterior(self) -> Optional[np.ndarray]:
        return self._posterior

    @property
    def prior(self) -> Optional[np.ndarray]:
        return self._prior

    @property
    def shrinkage_applied(self) -> Optional[float]:
        """How much the prior was shrunk toward realized (0 = no shrinkage, 1 = full)."""
        if self._prior is None or self._posterior is None:
            return None
        diff_prior = np.linalg.norm(self._posterior - self._prior)
        diff_total = np.linalg.norm(self._realized_mean - self._prior) if self._realized_mean is not None else 1.0
        if diff_total < 1e-12:
            return 0.0
        return float(np.clip(diff_prior / diff_total, 0.0, 1.0))
