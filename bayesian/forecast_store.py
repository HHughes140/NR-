import json
import os
from datetime import datetime
from typing import Optional

import numpy as np


class ForecastStore:
    """Persist forecasts and realized returns for Bayesian learning.

    Stores records as JSON lines. Each record has:
    - date, tickers, forecast (expected returns), realized (actual returns),
      error (forecast - realized), metadata (regime, sentiment, etc.).
    """

    def __init__(self, store_path: str = None):
        self.store_path = store_path or os.path.expanduser(
            "~/.NR/forecasts.jsonl")
        self._records: list[dict] = []
        self._loaded = False
        self._dirty = False

    def _ensure_dir(self):
        d = os.path.dirname(self.store_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.store_path):
            return
        with open(self.store_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def store_forecast(self, tickers: list[str], forecast: np.ndarray,
                       metadata: dict = None):
        """Store a new forecast."""
        self._load()
        if self._dirty:
            self._rewrite()  # Flush pending update_realized changes first
        record = {
            "date": datetime.now().isoformat(),
            "tickers": tickers,
            "forecast": forecast.tolist(),
            "realized": None,
            "error": None,
            "metadata": metadata or {},
        }
        self._records.append(record)
        self._ensure_dir()
        with open(self.store_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def update_realized(self, date: str, realized: np.ndarray):
        """Update the most recent forecast matching date prefix with realized returns."""
        self._load()
        for record in reversed(self._records):
            if record["date"].startswith(date) and record["realized"] is None:
                record["realized"] = realized.tolist()
                record["error"] = (
                    np.array(record["forecast"]) - realized).tolist()
                self._dirty = True
                return
        # No matching unrealized forecast — store as standalone
        self._records.append({
            "date": date,
            "tickers": [],
            "forecast": None,
            "realized": realized.tolist(),
            "error": None,
            "metadata": {},
        })
        self._dirty = True

    def _rewrite(self):
        """Rewrite the entire store (after updating records in place)."""
        self._ensure_dir()
        with open(self.store_path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")
        self._dirty = False

    def flush(self):
        """Write pending changes to disk (deferred from update_realized)."""
        if self._dirty:
            self._rewrite()

    def get_errors(self, n_recent: int = None,
                   event_type: str = None) -> list[dict]:
        """Get forecast errors for Bayesian updating.

        Args:
            n_recent: Limit to the N most recent error records.
            event_type: If provided, filter to records whose metadata
                contains this event_type.  Enables accuracy weights
                conditioned on event classification.
        """
        self._load()
        errors = [r for r in self._records if r["error"] is not None]
        if event_type is not None:
            errors = [
                r for r in errors
                if r.get("metadata", {}).get("event_type") == event_type
            ]
        if n_recent is not None:
            errors = errors[-n_recent:]
        return errors

    def get_accuracy_weights(self, n_recent: int = 20,
                             decay: float = 0.95) -> np.ndarray:
        """Compute accuracy-based weights for covariance weighting.

        More accurate recent forecasts get higher weight.
        Returns weights for recent observations (most recent last).
        Falls back to uniform weights over n_recent if no errors exist.
        """
        errors = self.get_errors(n_recent)
        if not errors:
            w = np.ones(n_recent) / n_recent
            return w

        mse_list = []
        for record in errors:
            err = np.array(record["error"])
            mse_list.append(float(np.mean(err ** 2)))

        mse = np.array(mse_list)
        # Invert MSE to get accuracy (lower error → higher weight)
        accuracy = 1.0 / (1.0 + mse)
        # Apply recency decay
        n = len(accuracy)
        recency = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights = accuracy * recency
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        return weights

    def summary(self) -> dict:
        """Summary statistics of forecast performance."""
        self._load()
        errors = self.get_errors()
        if not errors:
            return {"n_forecasts": len(self._records), "n_with_realized": 0,
                    "mean_abs_error": None, "mean_squared_error": None}

        all_errors = []
        for record in errors:
            all_errors.extend(record["error"])
        all_errors = np.array(all_errors)

        return {
            "n_forecasts": len(self._records),
            "n_with_realized": len(errors),
            "mean_abs_error": float(np.mean(np.abs(all_errors))),
            "mean_squared_error": float(np.mean(all_errors ** 2)),
            "mean_bias": float(np.mean(all_errors)),
        }

    def clear(self):
        """Clear all records."""
        self._records = []
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        self._loaded = True
