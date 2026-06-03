import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from NR.exceptions import ImpliedVolError


class BlackScholes:
    """Black-Scholes option pricing model for European options.

    Parameters across all methods:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual, continuous)
        sigma: Volatility (annual)
        q: Dividend yield (annual, continuous)
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0) -> float:
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0) -> float:
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float,
                   q: float = 0) -> float:
        """European call price."""
        if T <= 0:
            return max(S - K, 0.0)
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float,
                  q: float = 0) -> float:
        """European put price."""
        if T <= 0:
            return max(K - S, 0.0)
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def put_call_parity_check(call: float, put: float, S: float, K: float,
                              T: float, r: float, q: float = 0) -> float:
        """Residual of put-call parity (should be ~0 for European options)."""
        return call - put - S * np.exp(-q * T) + K * np.exp(-r * T)

    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float,
                           r: float, option_type: str = "call", q: float = 0,
                           tol: float = 1e-6, max_iter: int = 100) -> float:
        """Solve for implied volatility using Brent's method."""
        price_fn = BlackScholes.call_price if option_type == "call" else BlackScholes.put_price

        def objective(sigma):
            return price_fn(S, K, T, r, sigma, q) - market_price

        try:
            return brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
        except ValueError as e:
            raise ImpliedVolError(
                f"Could not find implied vol for price={market_price}, "
                f"S={S}, K={K}, T={T}: {e}"
            ) from e

    @staticmethod
    def price_surface(S: float, K_range: np.ndarray, T_range: np.ndarray,
                      r: float, sigma: float, option_type: str = "call",
                      q: float = 0) -> np.ndarray:
        """Option prices over a grid of (strike, time)."""
        price_fn = BlackScholes.call_price if option_type == "call" else BlackScholes.put_price
        surface = np.zeros((len(K_range), len(T_range)))
        for i, K in enumerate(K_range):
            for j, T in enumerate(T_range):
                surface[i, j] = price_fn(S, K, T, r, sigma, q)
        return surface
