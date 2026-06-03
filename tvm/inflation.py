import numpy as np
import pandas as pd


class InflationAdjuster:
    """Inflation adjustment and purchasing power calculations."""

    @staticmethod
    def real_return(nominal_return: float, inflation_rate: float) -> float:
        """Fisher equation: (1 + r_real) = (1 + r_nominal) / (1 + r_inflation)"""
        return (1 + nominal_return) / (1 + inflation_rate) - 1

    @staticmethod
    def nominal_to_real_series(nominal_returns: pd.Series,
                               inflation_rate: float,
                               periods_per_year: int = 252) -> pd.Series:
        """Convert series of nominal returns to real returns."""
        # Convert annual inflation to per-period
        inflation_per_period = (1 + inflation_rate) ** (1 / periods_per_year) - 1
        return (1 + nominal_returns) / (1 + inflation_per_period) - 1

    @staticmethod
    def purchasing_power_erosion(years: int, inflation_rate: float) -> float:
        """What $1 today is worth in `years` at given inflation."""
        return 1 / (1 + inflation_rate) ** years

    @staticmethod
    def purchasing_power_curve(years: int, inflation_rate: float) -> np.ndarray:
        """Array of purchasing power factors for years 0..years."""
        t = np.arange(years + 1)
        return 1 / (1 + inflation_rate) ** t

    @staticmethod
    def inflation_adjusted_value(nominal_value: float, years: int,
                                 inflation_rate: float) -> float:
        """Real value = nominal / (1 + inflation)^years"""
        return nominal_value / (1 + inflation_rate) ** years

    @staticmethod
    def required_nominal_return(target_real_return: float,
                                inflation_rate: float) -> float:
        """What nominal return is needed to achieve target real return."""
        return (1 + target_real_return) * (1 + inflation_rate) - 1

    @staticmethod
    def real_wealth_growth(initial_value: float, nominal_return: float,
                           inflation_rate: float, years: int) -> pd.DataFrame:
        """Year-by-year table of nominal vs real portfolio value."""
        data = []
        for y in range(years + 1):
            nominal = initial_value * (1 + nominal_return) ** y
            pp_factor = 1 / (1 + inflation_rate) ** y
            real = nominal * pp_factor
            cum_real_return = real / initial_value - 1
            data.append({
                "year": y,
                "nominal_value": nominal,
                "real_value": real,
                "purchasing_power_factor": pp_factor,
                "cumulative_real_return": cum_real_return,
            })
        return pd.DataFrame(data).set_index("year")

    @staticmethod
    def breakeven_inflation(nominal_yield: float, real_yield: float) -> float:
        """Breakeven inflation rate implied by nominal vs TIPS yields."""
        return (1 + nominal_yield) / (1 + real_yield) - 1
