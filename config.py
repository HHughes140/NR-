import math
from dataclasses import dataclass, field
from typing import Optional

from NR.exceptions import ConfigValidationError


def _check(condition: bool, msg: str):
    """Raise ConfigValidationError if condition is False."""
    if not condition:
        raise ConfigValidationError(msg)


# Timeframe presets: bars per trading day
TIMEFRAME_PRESETS = {
    "1d": {"equity": 1, "crypto": 1},
    "4h": {"equity": 2, "crypto": 6},
    "1h": {"equity": 7, "crypto": 24},
}


@dataclass
class DataConfig:
    period: str = "2y"
    interval: str = "1d"
    benchmark: str = "SPY"
    risk_free_rate: float = 0.05
    trading_days_per_year: int = 252
    cache_dir: Optional[str] = None
    chunk_size: int = 20
    max_retries: int = 3
    timeframe: str = "1d"
    bars_per_day: Optional[int] = None

    def __post_init__(self):
        if self.timeframe != "1d" and self.interval == "1d":
            self.interval = "1h"

    @property
    def _effective_bars_per_day(self) -> int:
        if self.bars_per_day is not None:
            return self.bars_per_day
        preset = TIMEFRAME_PRESETS.get(self.timeframe)
        return preset["equity"] if preset else 1

    @property
    def bars_per_year(self) -> int:
        return self._effective_bars_per_day * self.trading_days_per_year

    def scale_window(self, daily_window: int) -> int:
        return daily_window * self._effective_bars_per_day

    def validate(self):
        _check(0 <= self.risk_free_rate <= 1, "risk_free_rate must be in [0, 1]")
        _check(self.trading_days_per_year > 0, "trading_days_per_year must be > 0")
        _check(self.chunk_size > 0, "chunk_size must be > 0")
        _check(self.max_retries > 0, "max_retries must be > 0")
        _check(len(self.benchmark) > 0, "benchmark must be non-empty")
        _check(self.timeframe in TIMEFRAME_PRESETS,
               f"timeframe must be one of {list(TIMEFRAME_PRESETS)}")
        if self.bars_per_day is not None:
            _check(self.bars_per_day > 0, "bars_per_day must be > 0")


@dataclass
class OptimizationConfig:
    allow_short: bool = False
    max_weight: float = 1.0
    min_weight: float = 0.0
    target_return: Optional[float] = None
    frontier_points: int = 100
    shrinkage_target: str = "constant_correlation"
    dro_enabled: bool = False
    dro_kappa: float = 0.1

    def validate(self):
        _check(self.max_weight >= self.min_weight,
               "max_weight must be >= min_weight")
        _check(self.min_weight >= 0, "min_weight must be >= 0")
        _check(self.max_weight <= 1 or self.allow_short,
               "max_weight must be <= 1 when shorting disabled")
        _check(self.frontier_points > 0, "frontier_points must be > 0")
        _check(self.shrinkage_target in (
            "constant_correlation", "identity", "f_matrix"),
            f"Unknown shrinkage_target: {self.shrinkage_target}")


@dataclass
class RiskConfig:
    """Risk configuration calibrated for insurance-based trading.

    Higher confidence levels than equity portfolios because option
    selling has asymmetric payoffs -- the tail is what kills you.
    Scenarios include vol-specific catastrophes (Volmageddon, correlation
    spikes) in addition to standard equity drawdowns.
    """
    var_confidence: float = 0.99           # 99% for option books (vs 95% equity)
    cvar_confidence: float = 0.99
    monte_carlo_simulations: int = 50_000  # more sims for tail accuracy
    ewma_halflife: int = 30                # faster decay -- vol regime shifts matter more
    # Stress scenarios: insurance-relevant tail events
    stress_scenarios: dict = field(default_factory=lambda: {
        # Equity crashes (underlying risk)
        "2008_crisis": -0.38,
        "covid_crash": -0.34,
        "dot_com": -0.49,
        # Vol-specific catastrophes
        "volmageddon_2018": -0.96,         # XIV lost 96% in one day
        "aug_2015_flash": -0.11,           # S&P flash crash, VIX tripled
        "feb_2020_vol_spike": -0.12,       # VIX 13 -> 82 in 4 weeks
        # Correlation spike (diversification failure)
        "correlation_spike": -0.25,        # everything correlates to 1
        # Rate shock (affects put pricing via rho)
        "rate_shock": -0.20,
        # Mild drawdown (high frequency, tests premium buffer)
        "garden_variety_5pct": -0.05,
        "garden_variety_10pct": -0.10,
    })
    # Tail risk multipliers for option books
    # In a crisis, realized vol can be 3-5x implied -- these scale
    # the covariance matrix beyond the regime multipliers to stress-test
    # the book under extreme conditions
    tail_vol_multiplier: float = 3.0       # stress-test at 3x current vol
    tail_correlation_override: float = 0.85  # assume 85% correlation in crisis

    def validate(self):
        _check(0 < self.var_confidence < 1, "var_confidence must be in (0, 1)")
        _check(0 < self.cvar_confidence < 1, "cvar_confidence must be in (0, 1)")
        _check(self.monte_carlo_simulations >= 100,
               "monte_carlo_simulations must be >= 100")
        _check(self.ewma_halflife > 0, "ewma_halflife must be > 0")
        _check(self.tail_vol_multiplier >= 1.0,
               "tail_vol_multiplier must be >= 1.0")
        _check(0 < self.tail_correlation_override <= 1.0,
               "tail_correlation_override must be in (0, 1]")


@dataclass
class OptionsConfig:
    """Options configuration for insurance-based premium collection.

    Extends basic Black-Scholes pricing with parameters for systematic
    premium selling: target DTE, delta ranges, roll timing, margin.
    """
    pricing_model: str = "black_scholes"
    implied_vol_tolerance: float = 1e-6
    implied_vol_max_iter: int = 100
    dividend_yield: float = 0.0

    # --- Premium collection parameters ---
    # Target days-to-expiry for selling (30-45 DTE is the theta sweet spot)
    target_dte_min: int = 25
    target_dte_max: int = 50
    target_dte_ideal: int = 35             # peak theta/gamma ratio
    # Delta targets for short options (absolute value)
    # 0.16 delta ~ 1 standard deviation OTM ~ 84% probability of expiring worthless
    short_put_delta: float = 0.16
    short_call_delta: float = 0.16
    # Minimum premium to collect (annualized % of notional)
    min_premium_annualized: float = 0.08   # don't sell for less than 8% annualized
    # Roll timing: roll when DTE drops below this or premium captured exceeds target
    roll_dte_trigger: int = 10             # roll at 10 DTE (avoid gamma risk)
    roll_profit_target: float = 0.50       # roll when 50% of max profit captured
    roll_loss_trigger: float = 2.0         # roll/close when loss = 2x premium collected

    # --- Margin / collateral ---
    margin_method: str = "notional"        # "notional", "span", or "portfolio"
    notional_margin_pct: float = 0.20      # 20% of notional as collateral
    max_portfolio_margin_usage: float = 0.70  # never use more than 70% of available margin
    # Buying power reserve -- always keep this fraction undeployed
    buying_power_reserve: float = 0.30

    # --- Greeks limits (portfolio level) ---
    max_portfolio_delta: float = 0.15      # net delta exposure as fraction of NAV
    max_portfolio_gamma: float = 0.05      # max negative gamma as fraction of NAV
    max_portfolio_vega: float = 0.10       # max short vega as fraction of NAV
    min_portfolio_theta: float = 0.0       # theta must be positive (collecting premium)

    def validate(self):
        _check(self.pricing_model in ("black_scholes",),
               f"Unknown pricing_model: {self.pricing_model}")
        _check(self.implied_vol_tolerance > 0,
               "implied_vol_tolerance must be > 0")
        _check(self.implied_vol_max_iter > 0,
               "implied_vol_max_iter must be > 0")
        _check(self.dividend_yield >= 0, "dividend_yield must be >= 0")
        _check(0 < self.target_dte_min < self.target_dte_max,
               "target_dte_min must be in (0, target_dte_max)")
        _check(self.target_dte_min <= self.target_dte_ideal <= self.target_dte_max,
               "target_dte_ideal must be in [target_dte_min, target_dte_max]")
        _check(0 < self.short_put_delta < 0.50,
               "short_put_delta must be in (0, 0.50)")
        _check(0 < self.short_call_delta < 0.50,
               "short_call_delta must be in (0, 0.50)")
        _check(self.min_premium_annualized >= 0,
               "min_premium_annualized must be >= 0")
        _check(0 < self.roll_dte_trigger < self.target_dte_min,
               "roll_dte_trigger must be in (0, target_dte_min)")
        _check(0 < self.roll_profit_target <= 1.0,
               "roll_profit_target must be in (0, 1]")
        _check(self.roll_loss_trigger > 0,
               "roll_loss_trigger must be > 0")
        _check(self.margin_method in ("notional", "span", "portfolio"),
               f"Unknown margin_method: {self.margin_method}")
        _check(0 < self.notional_margin_pct <= 1.0,
               "notional_margin_pct must be in (0, 1]")
        _check(0 < self.max_portfolio_margin_usage <= 1.0,
               "max_portfolio_margin_usage must be in (0, 1]")
        _check(0 < self.buying_power_reserve < 1.0,
               "buying_power_reserve must be in (0, 1)")
        _check(self.max_portfolio_delta >= 0,
               "max_portfolio_delta must be >= 0")
        _check(self.max_portfolio_vega >= 0,
               "max_portfolio_vega must be >= 0")


@dataclass
class MacroConfig:
    """Macro configuration calibrated for insurance-based vol trading.

    Replaces generic equity regime detection with volatility-surface-aware
    indicators: VIX term structure, VVIX (vol of vol), SKEW index,
    variance risk premium, and realized-vs-implied spread.

    Regime detection is tuned for option selling: the key question is
    not 'is the market going up or down' but 'is it safe to sell
    premium right now'.
    """
    fred_api_key: Optional[str] = None

    # --- Volatility surface tickers ---
    vix_ticker: str = "^VIX"               # spot implied vol (30-day)
    vix9d_ticker: str = "^VIX9D"           # 9-day VIX (near-term fear)
    vix3m_ticker: str = "^VIX3M"           # 3-month VIX (term structure)
    vvix_ticker: str = "^VVIX"             # vol of vol (tail risk pricing)
    skew_ticker: str = "^SKEW"             # CBOE SKEW (tail risk demand)
    # VIX futures for contango/backwardation (primary premium signal)
    vix_front_future: str = "VX=F"         # front-month VIX future
    # Underlying
    spx_ticker: str = "^GSPC"              # S&P 500 for realized vol
    tnx_ticker: str = "^TNX"               # 10Y yield (rho exposure)
    irx_ticker: str = "^IRX"               # 3-month T-bill (risk-free)
    dxy_ticker: str = "DX-Y.NYB"           # dollar index
    gold_ticker: str = "GC=F"              # gold (tail hedge proxy)

    # --- FRED series (insurance-relevant) ---
    fred_series: dict = field(default_factory=lambda: {
        # Volatility & tail risk
        "financial_stress": "STLFSI2",     # St. Louis Financial Stress Index
        "financial_conditions": "NFCI",    # Chicago Fed National Financial Conditions
        # Credit (default risk = claim risk)
        "hy_oas": "BAMLH0A0HYM2",         # HY OAS -- credit stress
        "ig_oas": "BAMLC0A0CM",            # IG OAS -- broad credit
        "bbb_spread": "BAMLC0A4CBBB",     # BBB spread
        "ted_spread": "TEDRATE",           # interbank stress
        # Rates (affects option pricing via rho and discounting)
        "fed_funds": "FEDFUNDS",
        "yield_spread": "T10Y2Y",          # recession signal
        "real_rate_5y": "DFII5",           # 5Y real rate (TIPS)
        "breakeven_5y": "T5YIE",           # 5Y inflation breakeven
        # Economic backdrop
        "unemployment": "UNRATE",
        "cpi": "CPIAUCSL",
    })

    # --- Variance risk premium tracking ---
    # VRP = implied vol - realized vol. Positive VRP = premium available.
    # This is the core "insurance margin" -- what you collect minus what
    # you actually pay out.
    vrp_realized_vol_window: int = 21      # 21-day realized vol (matches monthly options)
    vrp_min_threshold: float = 2.0         # minimum VRP (vol points) to sell premium
    vrp_rich_threshold: float = 6.0        # VRP above this = premium is very rich, size up
    vrp_ewma_halflife: int = 10            # half-life for smoothing VRP signal

    # --- VIX term structure ---
    # Contango (VIX futures > spot VIX) = normal, safe to sell premium
    # Backwardation (spot > futures) = fear, dangerous to sell
    term_structure_contango_threshold: float = 0.03   # 3% contango = healthy
    term_structure_backwardation_warning: float = -0.02  # -2% = caution
    term_structure_backwardation_crisis: float = -0.10   # -10% = stop selling

    # --- VVIX (vol of vol) thresholds ---
    # High VVIX = unstable vol regime, options can reprice violently
    vvix_normal: float = 90.0              # below this = stable vol environment
    vvix_elevated: float = 110.0           # above this = vol is volatile, reduce size
    vvix_crisis: float = 140.0             # above this = stop selling premium

    # --- SKEW index thresholds ---
    # High SKEW = market pricing tail risk, put skew is steep
    skew_normal: float = 130.0             # typical range
    skew_elevated: float = 145.0           # unusual tail demand
    skew_extreme: float = 155.0            # extreme -- puts are very expensive

    # --- Regime detection (vol-selling calibration) ---
    # These are tighter than equity thresholds because option sellers
    # need earlier warning -- by the time VIX hits 35, your short puts
    # are already deep in trouble.
    regime_vix_elevated: float = 20.0      # caution: reduce new premium sales
    regime_vix_high: float = 25.0          # warning: close expiring positions, no new sales
    regime_vix_crisis: float = 30.0        # crisis: hedge everything, consider closing book
    regime_yield_curve_inversion: float = 0.0
    regime_credit_spread_stress: float = 4.0   # tighter than equity (400bp)
    regime_credit_spread_crisis: float = 6.0   # tighter than equity (600bp)
    # Bank sector stress
    regime_bank_drawdown_threshold: float = 0.10  # tighter (10% vs 15%)
    regime_bank_drawdown_window: int = 63

    # --- Regime covariance multipliers ---
    # Higher crisis multiplier than equity because option payoffs are
    # convex -- a 2x vol move doesn't cause 2x loss, it causes 4x loss.
    regime_cov_multipliers: dict = field(default_factory=lambda: {
        "expansion": 0.7,      # vol is low, can tighten estimates
        "contraction": 1.5,    # vol expanding, widen
        "crisis": 3.0,         # option book needs 3x cov (convexity)
        "recovery": 1.2,       # vol still elevated but declining
    })
    # Separate correlation multipliers (corr spikes are distinct from vol spikes)
    regime_correlation_multipliers: dict = field(default_factory=lambda: {
        "expansion": 0.9,      # diversification works
        "contraction": 1.2,    # correlations rising
        "crisis": 1.8,         # everything moves together
        "recovery": 1.1,       # correlations normalizing
    })
    # Return adjustments for premium estimation
    regime_return_adjustments: dict = field(default_factory=lambda: {
        "expansion": 0.0002,   # normal premium accrual
        "contraction": -0.0003,
        "crisis": -0.001,      # larger adjustment -- options move faster
        "recovery": 0.0004,    # post-crisis premium is rich
    })

    # --- Premium sizing by regime ---
    # Fraction of max capacity to deploy in each regime
    regime_deployment: dict = field(default_factory=lambda: {
        "expansion": 1.0,      # full deployment
        "contraction": 0.50,   # half size
        "crisis": 0.0,         # no new premium sales
        "recovery": 0.75,      # rebuilding book
    })

    # --- Factor model ---
    # For insurance trading, 3-factor is sufficient -- we care about
    # market beta, not size/value/momentum factor decomposition.
    ff_factors: int = 3
    ff_frequency: str = "daily"

    # --- Markov regime switching ---
    use_markov_regime: bool = True         # on by default for insurance
    markov_burn_in: int = 63

    def validate(self):
        _check(self.regime_vix_elevated < self.regime_vix_high,
               "regime_vix_elevated must be < regime_vix_high")
        _check(self.regime_vix_high < self.regime_vix_crisis,
               "regime_vix_high must be < regime_vix_crisis")
        _check(self.regime_vix_elevated > 0,
               "regime_vix_elevated must be > 0")
        _check(self.regime_credit_spread_stress < self.regime_credit_spread_crisis,
               "regime_credit_spread_stress must be < regime_credit_spread_crisis")
        _check(all(v > 0 for v in self.regime_cov_multipliers.values()),
               "regime_cov_multipliers values must be > 0")
        _check(all(v > 0 for v in self.regime_correlation_multipliers.values()),
               "regime_correlation_multipliers values must be > 0")
        _check(all(0 <= v <= 1 for v in self.regime_deployment.values()),
               "regime_deployment values must be in [0, 1]")
        _check(self.ff_factors in (0, 3, 5, 6),
               "ff_factors must be 0, 3, 5, or 6")
        _check(self.ff_frequency in ("daily", "monthly"),
               f"Unknown ff_frequency: {self.ff_frequency}")
        _check(self.vrp_realized_vol_window > 0,
               "vrp_realized_vol_window must be > 0")
        _check(self.vrp_min_threshold >= 0,
               "vrp_min_threshold must be >= 0")
        _check(self.vrp_rich_threshold > self.vrp_min_threshold,
               "vrp_rich_threshold must be > vrp_min_threshold")
        _check(self.vvix_normal < self.vvix_elevated < self.vvix_crisis,
               "VVIX thresholds must be in ascending order")
        _check(self.term_structure_backwardation_crisis
               < self.term_structure_backwardation_warning
               < self.term_structure_contango_threshold,
               "term structure thresholds must be in ascending order")


@dataclass
class BayesianConfig:
    enabled: bool = True
    prior_weight: float = 0.6
    decay_factor: float = 0.95
    min_observations: int = 20
    store_path: Optional[str] = None

    def validate(self):
        _check(0 < self.prior_weight < 1, "prior_weight must be in (0, 1)")
        _check(0 < self.decay_factor < 1, "decay_factor must be in (0, 1)")
        _check(self.min_observations > 0, "min_observations must be > 0")


@dataclass
class BacktestConfig:
    """Backtest configuration for insurance-based strategies.

    Options-aware cost model with wider spreads, and portfolio-level
    drawdown limits calibrated for premium-selling books.
    """
    lookback_window: int = 252
    rebalance_frequency: int = 7           # weekly -- options need more frequent management
    # Transaction costs (options have wider spreads than equity)
    fixed_cost_per_trade: float = 1.0      # per-contract commission
    proportional_cost_bps: float = 30.0    # options spreads are wider (30bps vs 10)
    market_impact_coefficient: float = 0.15
    slippage_bps: float = 15.0             # options slippage is higher
    initial_capital: float = 100_000.0
    # Almgren-Chriss (less relevant for options but kept for underlying hedges)
    use_almgren_chriss: bool = False
    spread_bps: float = 5.0
    commission_bps: float = 1.0
    eta: float = 0.1
    gamma: float = 0.05
    impact_exponent: float = 0.6
    # Benchmark
    benchmark_ticker: str = "SPY"
    # Weight caps
    max_weight_per_asset: float = 1.0
    min_weight_per_asset: float = 0.0
    vol_scaled_caps: bool = True           # on by default for insurance
    vol_cap_base: float = 0.10             # tighter than equity (10% vs 15%)
    vol_cap_floor: float = 0.01
    vol_cap_lookback: int = 63
    # Turnover penalty
    turnover_penalty_bps: float = 0.0
    # Volatility filter
    max_asset_volatility: float = 0.0
    vol_filter_lookback: int = 63
    vol_filter_min_assets: int = 5
    # Cash / risk-off
    exit_to_cash: bool = True              # on by default -- insurance needs cash option
    cash_return_daily: float = 0.0
    # Volatility targeting (critical for option books)
    vol_target_enabled: bool = True
    vol_target_annualized: float = 0.12    # lower target than equity (12% vs 20%)
    vol_target_lookback: int = 21          # faster lookback for vol changes
    vol_target_max_leverage: float = 1.0   # no leverage
    vol_target_min_exposure: float = 0.30  # can go as low as 30% deployed
    # Position filtering
    max_positions: int = 0
    min_position_weight: float = 0.0
    conviction_threshold: float = 0.0
    min_trade_weight: float = 0.0
    # Short selling (needed for hedging)
    allow_short: bool = True
    short_borrow_annual_bps: float = 50.0
    # Portfolio-level drawdown limit (ruin constraint)
    # Insurance books need strict drawdown limits -- a 25% drawdown
    # on a premium-selling book is catastrophic and hard to recover from.
    max_portfolio_drawdown: float = 0.20   # exit at -20% (vs 100% disabled for equity)

    def validate(self):
        _check(self.lookback_window > 0, "lookback_window must be > 0")
        _check(self.rebalance_frequency > 0,
               "rebalance_frequency must be > 0")
        _check(self.lookback_window >= self.rebalance_frequency,
               "lookback_window must be >= rebalance_frequency")
        _check(self.initial_capital > 0, "initial_capital must be > 0")
        _check(self.fixed_cost_per_trade >= 0,
               "fixed_cost_per_trade must be >= 0")
        _check(0 <= self.proportional_cost_bps <= 200,
               "proportional_cost_bps must be in [0, 200]")
        _check(0 <= self.slippage_bps <= 100,
               "slippage_bps must be in [0, 100]")
        _check(self.max_weight_per_asset >= self.min_weight_per_asset,
               "max_weight_per_asset must be >= min_weight_per_asset")
        if self.allow_short:
            _check(self.min_weight_per_asset >= -1.0,
                   "min_weight_per_asset must be >= -1.0 when allow_short=True")
        else:
            _check(self.min_weight_per_asset >= 0.0,
                   "min_weight_per_asset must be >= 0.0 when allow_short=False")
        if self.use_almgren_chriss:
            _check(self.eta > 0, "Almgren-Chriss eta must be > 0")
            _check(self.gamma >= 0, "Almgren-Chriss gamma must be >= 0")
            _check(0 < self.impact_exponent <= 1,
                   "impact_exponent must be in (0, 1]")
        if self.vol_scaled_caps:
            _check(self.vol_cap_base > 0, "vol_cap_base must be > 0")
            _check(self.vol_cap_floor > 0, "vol_cap_floor must be > 0")
            _check(self.vol_cap_base >= self.vol_cap_floor,
                   "vol_cap_base must be >= vol_cap_floor")
        _check(0 < self.max_portfolio_drawdown <= 1.0,
               "max_portfolio_drawdown must be in (0, 1]")

    @classmethod
    def for_timeframe(cls, data_config: "DataConfig", **overrides) -> "BacktestConfig":
        scale = data_config.scale_window
        defaults = {
            "lookback_window": scale(252),
            "rebalance_frequency": scale(7),
            "vol_cap_lookback": scale(63),
            "vol_filter_lookback": scale(63),
            "vol_target_lookback": scale(21),
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class InsuranceFactorConfig:
    """Configuration for insurance fundamental factor model.

    Controls which fundamentals to extract from quarterly financials,
    how to handle missing data, and regression parameters.
    """
    # Default insurance universe (major US insurers)
    default_tickers: tuple = (
        "PGR", "ALL", "TRV", "CB", "AIG", "MET", "AFL", "HIG",
        "CINF", "WRB", "GL", "AIZ", "BRK-B", "L", "RE",
        "RNR", "ERIE", "ACGL", "AJG", "BRO", "MMC", "AON",
    )

    # Factor selection -- which insurance fundamentals to include
    factors_enabled: tuple = (
        "loss_ratio",
        "lae_ratio",
        "expense_ratio",
        "combined_ratio",
        "premium_growth_yoy",
        "investment_yield",
        "reserve_ratio",
        "leverage_ratio",
        "roe",
        "book_value_growth",
        "operating_margin",
        "investment_to_assets",
    )

    # Data handling
    min_quarters: int = 4          # require at least 4 quarters of data
    min_tickers: int = 6           # min universe size for cross-sectional regression
    cross_sectional_zscore: bool = True  # z-score factors across universe each day
    winsorize_pct: float = 0.025   # winsorize at 2.5% / 97.5%

    # Regression
    regression_mode: str = "cross_sectional"  # "cross_sectional" or "time_series"
    min_observations: int = 60     # minimum daily observations for regression

    # Data fetch
    fetch_period: str = "5y"       # yfinance period for price data
    chunk_size: int = 10
    max_retries: int = 3

    def validate(self):
        _check(self.min_quarters >= 2, "min_quarters must be >= 2")
        _check(self.min_tickers >= 3, "min_tickers must be >= 3")
        _check(0 <= self.winsorize_pct < 0.25,
               "winsorize_pct must be in [0, 0.25)")
        _check(self.regression_mode in ("cross_sectional", "time_series"),
               f"Unknown regression_mode: {self.regression_mode}")
        _check(self.min_observations >= 30,
               "min_observations must be >= 30")


@dataclass
class TradingConfig:
    enabled: bool = True
    cost_method: str = "average"
    min_turnover_threshold: float = 0.02
    rebalance_benefit_threshold: float = 0.001
    short_term_tax_rate: float = 0.37
    long_term_tax_rate: float = 0.20
    long_term_holding_days: int = 365
    initial_cash: float = 100_000.0

    def validate(self):
        _check(self.cost_method in ("average", "worst_case"),
               f"Unknown cost_method: {self.cost_method}")
        _check(0 <= self.min_turnover_threshold <= 1,
               "min_turnover_threshold must be in [0, 1]")
        _check(0 <= self.short_term_tax_rate <= 1,
               "short_term_tax_rate must be in [0, 1]")
        _check(0 <= self.long_term_tax_rate <= 1,
               "long_term_tax_rate must be in [0, 1]")
        _check(self.long_term_holding_days > 0,
               "long_term_holding_days must be > 0")
        _check(self.initial_cash > 0, "initial_cash must be > 0")
