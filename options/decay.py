import numpy as np
import pandas as pd

from NR.options.black_scholes import BlackScholes
from NR.options.greeks import Greeks


class OptionDecay:
    """Time decay (theta) analysis and visualization data generation."""

    @staticmethod
    def theta_decay_curve(S, K, r, sigma, days_to_expiry: int,
                          option_type="call", q=0) -> pd.DataFrame:
        """Option value at each day from now to expiry."""
        price_fn = BlackScholes.call_price if option_type == "call" else BlackScholes.put_price
        rows = []
        for d in range(days_to_expiry, -1, -1):
            T = d / 365.0
            total_value = price_fn(S, K, T, r, sigma, q) if T > 0 else max(S - K, 0) if option_type == "call" else max(K - S, 0)
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            time_value = total_value - intrinsic
            daily_theta = Greeks.theta(S, K, T, r, sigma, option_type, q) if T > 0 else 0.0
            rows.append({
                "days_remaining": d,
                "total_value": total_value,
                "intrinsic_value": intrinsic,
                "time_value": max(time_value, 0),
                "daily_theta": daily_theta,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def time_value_surface(S, K_range, days_range, r, sigma,
                           option_type="call", q=0) -> pd.DataFrame:
        """Time value across strikes and days-to-expiry for 3D surface."""
        price_fn = BlackScholes.call_price if option_type == "call" else BlackScholes.put_price
        rows = []
        for K in K_range:
            for d in days_range:
                T = d / 365.0
                intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
                total = price_fn(S, K, T, r, sigma, q) if T > 0 else intrinsic
                rows.append({
                    "strike": K,
                    "days_remaining": d,
                    "time_value": max(total - intrinsic, 0),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def theta_by_moneyness(S, r, sigma, T, option_type="call", q=0,
                           moneyness_range=(-0.3, 0.3), n_points=50) -> pd.DataFrame:
        """Theta as a function of moneyness (S/K - 1)."""
        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_points)
        rows = []
        for m in moneyness:
            K = S / (1 + m)
            theta = Greeks.theta(S, K, T, r, sigma, option_type, q)
            rows.append({"moneyness": m, "strike": K, "theta": theta})
        return pd.DataFrame(rows)
