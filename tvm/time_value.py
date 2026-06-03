import numpy as np
from scipy.optimize import brentq


class TimeValueOfMoney:
    """Core time value of money calculations with discrete and continuous compounding."""

    @staticmethod
    def present_value(future_value: float, rate: float, periods: int,
                      continuous: bool = False) -> float:
        if continuous:
            return future_value * np.exp(-rate * periods)
        return future_value / (1 + rate) ** periods

    @staticmethod
    def future_value(present_value: float, rate: float, periods: int,
                     continuous: bool = False) -> float:
        if continuous:
            return present_value * np.exp(rate * periods)
        return present_value * (1 + rate) ** periods

    @staticmethod
    def discount_factor(rate: float, periods: int, continuous: bool = False) -> float:
        if continuous:
            return np.exp(-rate * periods)
        return 1 / (1 + rate) ** periods

    @staticmethod
    def discount_factor_curve(rate: float, max_periods: int,
                              continuous: bool = False) -> np.ndarray:
        """Array of discount factors for periods 0..max_periods."""
        t = np.arange(max_periods + 1)
        if continuous:
            return np.exp(-rate * t)
        return 1 / (1 + rate) ** t

    @staticmethod
    def annuity_pv(payment: float, rate: float, periods: int) -> float:
        """PV of ordinary annuity."""
        if rate == 0:
            return payment * periods
        return payment * (1 - (1 + rate) ** (-periods)) / rate

    @staticmethod
    def annuity_due_pv(payment: float, rate: float, periods: int) -> float:
        """PV of annuity due (payments at start of period)."""
        return TimeValueOfMoney.annuity_pv(payment, rate, periods) * (1 + rate)

    @staticmethod
    def perpetuity_pv(payment: float, rate: float, growth: float = 0.0) -> float:
        """PV of perpetuity or growing perpetuity (requires rate > growth)."""
        if growth >= rate:
            raise ValueError("Growth rate must be less than discount rate for convergence")
        return payment / (rate - growth)

    @staticmethod
    def effective_annual_rate(nominal_rate: float, compounding_periods: int) -> float:
        """EAR = (1 + r/m)^m - 1"""
        return (1 + nominal_rate / compounding_periods) ** compounding_periods - 1

    @staticmethod
    def continuous_to_discrete(continuous_rate: float, compounding_periods: int = 1) -> float:
        """Convert continuous rate to equivalent discrete rate."""
        return compounding_periods * (np.exp(continuous_rate / compounding_periods) - 1)

    @staticmethod
    def discrete_to_continuous(discrete_rate: float, compounding_periods: int = 1) -> float:
        """Convert discrete rate to equivalent continuous rate."""
        return compounding_periods * np.log(1 + discrete_rate / compounding_periods)

    @staticmethod
    def npv(cashflows: list[float], rate: float) -> float:
        """Net present value: sum(CF_t / (1+r)^t)"""
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))

    @staticmethod
    def irr(cashflows: list[float], guess: float = 0.1) -> float:
        """Internal rate of return via Brent's method."""
        def _npv(r):
            return sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
        return brentq(_npv, -0.999, 10.0, xtol=1e-10)
