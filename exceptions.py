class PortfolioModelError(Exception):
    """Base exception for portfolio model."""


class DataFetchError(PortfolioModelError):
    """Failed to fetch data from yfinance."""


class InsufficientDataError(PortfolioModelError):
    def __init__(self, message: str, required: int, available: int):
        super().__init__(message)
        self.required = required
        self.available = available


class OptimizationError(PortfolioModelError):
    """Optimization did not converge or constraints are infeasible."""


class OptionsError(PortfolioModelError):
    """Options pricing or Greeks computation failed."""


class ImpliedVolError(OptionsError):
    """Implied volatility solver did not converge."""


class MacroDataError(PortfolioModelError):
    """Failed to fetch macro or factor data."""


class FREDError(MacroDataError):
    """FRED API request failed."""


class SentimentError(PortfolioModelError):
    """Sentiment analysis failed."""


class BacktestError(PortfolioModelError):
    """Backtesting engine error."""


class TradingError(PortfolioModelError):
    """Position tracking or rebalancing error."""


class PropagationError(PortfolioModelError):
    """Correlation propagation or factor decomposition failed."""


class ConfigValidationError(PortfolioModelError):
    """Configuration parameter out of valid range or inconsistent."""


class DataValidationError(PortfolioModelError):
    """Data quality check failed (NaN, negative prices, etc.)."""


class InstitutionalDataError(PortfolioModelError):
    """SEC EDGAR data fetch or parsing failed."""
