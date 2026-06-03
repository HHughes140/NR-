import numpy as np
from scipy.stats import norm

from NR.options.black_scholes import BlackScholes


class Greeks:
    """Analytical Greeks for European options under Black-Scholes."""

    @staticmethod
    def delta(S, K, T, r, sigma, option_type="call", q=0) -> float:
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        if option_type == "call":
            return np.exp(-q * T) * norm.cdf(d1)
        return -np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def gamma(S, K, T, r, sigma, q=0) -> float:
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(S, K, T, r, sigma, option_type="call", q=0) -> float:
        """Daily theta (annual theta / 365)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)

        common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == "call":
            annual = common + q * S * np.exp(-q * T) * norm.cdf(d1) \
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            annual = common - q * S * np.exp(-q * T) * norm.cdf(-d1) \
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)

        return annual / 365.0

    @staticmethod
    def vega(S, K, T, r, sigma, q=0) -> float:
        """Vega per 1% change in volatility."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0

    @staticmethod
    def rho(S, K, T, r, sigma, option_type="call", q=0) -> float:
        """Rho per 1% change in interest rate."""
        if T <= 0:
            return 0.0
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    @staticmethod
    def all_greeks(S, K, T, r, sigma, option_type="call", q=0) -> dict:
        """All five Greeks plus the option price."""
        price_fn = BlackScholes.call_price if option_type == "call" else BlackScholes.put_price
        return {
            "price": price_fn(S, K, T, r, sigma, q),
            "delta": Greeks.delta(S, K, T, r, sigma, option_type, q),
            "gamma": Greeks.gamma(S, K, T, r, sigma, q),
            "theta": Greeks.theta(S, K, T, r, sigma, option_type, q),
            "vega": Greeks.vega(S, K, T, r, sigma, q),
            "rho": Greeks.rho(S, K, T, r, sigma, option_type, q),
        }

    @staticmethod
    def greek_surface(S, K_range, T_range, r, sigma, greek: str,
                      option_type="call", q=0) -> np.ndarray:
        """Compute a Greek over a grid of (strike, time)."""
        greek_fn = getattr(Greeks, greek)
        surface = np.zeros((len(K_range), len(T_range)))
        for i, K in enumerate(K_range):
            for j, T in enumerate(T_range):
                kwargs = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
                if greek not in ("gamma", "vega"):
                    kwargs["option_type"] = option_type
                if greek != "rho":
                    kwargs["q"] = q
                else:
                    kwargs["q"] = q
                surface[i, j] = greek_fn(**kwargs)
        return surface
