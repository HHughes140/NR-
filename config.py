import math
from dataclasses import dataclass, field
from typing import Optional

from NR.exceptions import ConfigValidationError


def _check(condition: bool, msg: str):
    """Raise ConfigValidationError if condition is False."""
    if not condition:
        raise ConfigValidationError(msg)


# Timeframe presets: bars per trading day for equity and crypto markets
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
    # Multi-timeframe support
    timeframe: str = "1d"               # "1d", "4h", or "1h"
    bars_per_day: Optional[int] = None  # None = use equity preset for timeframe

    def __post_init__(self):
        # Auto-set yfinance interval from timeframe (4h downloads 1h then resamples)
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
        """Annualization constant: bars per trading day * trading days per year."""
        return self._effective_bars_per_day * self.trading_days_per_year

    def scale_window(self, daily_window: int) -> int:
        """Convert a window expressed in trading days to bars."""
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
    # Distributionally Robust Optimization (DRO)
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
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    monte_carlo_simulations: int = 10_000
    ewma_halflife: int = 60
    stress_scenarios: dict = field(default_factory=lambda: {
        "2008_crisis": -0.38,
        "covid_crash": -0.34,
        "dot_com": -0.49,
        "rate_shock": -0.20,
        "custom_mild": -0.10,
    })

    def validate(self):
        _check(0 < self.var_confidence < 1, "var_confidence must be in (0, 1)")
        _check(0 < self.cvar_confidence < 1, "cvar_confidence must be in (0, 1)")
        _check(self.monte_carlo_simulations >= 100,
               "monte_carlo_simulations must be >= 100")
        _check(self.ewma_halflife > 0, "ewma_halflife must be > 0")


@dataclass
class OptionsConfig:
    pricing_model: str = "black_scholes"
    implied_vol_tolerance: float = 1e-6
    implied_vol_max_iter: int = 100
    dividend_yield: float = 0.0

    def validate(self):
        _check(self.pricing_model in ("black_scholes",),
               f"Unknown pricing_model: {self.pricing_model}")
        _check(self.implied_vol_tolerance > 0,
               "implied_vol_tolerance must be > 0")
        _check(self.implied_vol_max_iter > 0,
               "implied_vol_max_iter must be > 0")
        _check(self.dividend_yield >= 0, "dividend_yield must be >= 0")


@dataclass
class MacroConfig:
    fred_api_key: Optional[str] = None
    vix_ticker: str = "^VIX"
    tnx_ticker: str = "^TNX"
    irx_ticker: str = "^IRX"
    dxy_ticker: str = "DX-Y.NYB"
    gold_ticker: str = "GC=F"
    oil_ticker: str = "CL=F"
    fred_series: dict = field(default_factory=lambda: {
        "gdp": "GDP", "cpi": "CPIAUCSL", "unemployment": "UNRATE",
        "fed_funds": "FEDFUNDS", "yield_spread": "T10Y2Y",
        "consumer_confidence": "UMCSENT",
        "hy_oas": "BAMLH0A0HYM2", "ted_spread": "TEDRATE",
        "bbb_spread": "BAMLC0A4CBBB",
        "financial_stress": "STLFSI2",
    })
    # Bank sector stress
    regime_bank_drawdown_threshold: float = 0.15
    regime_bank_drawdown_window: int = 63
    regime_vix_high: float = 25.0
    regime_vix_crisis: float = 35.0
    regime_yield_curve_inversion: float = 0.0
    regime_credit_spread_stress: float = 5.0   # HY OAS > 500bp = contraction
    regime_credit_spread_crisis: float = 8.0   # HY OAS > 800bp = crisis
    regime_cov_multipliers: dict = field(default_factory=lambda: {
        "expansion": 0.8, "contraction": 1.3, "crisis": 2.0, "recovery": 1.0,
    })
    regime_return_adjustments: dict = field(default_factory=lambda: {
        "expansion": 0.0001, "contraction": -0.0002,
        "crisis": -0.0005, "recovery": 0.0003,
    })
    ff_factors: int = 3
    ff_frequency: str = "daily"
    # Markov regime switching
    use_markov_regime: bool = False
    markov_burn_in: int = 63

    def validate(self):
        _check(self.regime_vix_high < self.regime_vix_crisis,
               "regime_vix_high must be < regime_vix_crisis")
        _check(self.regime_vix_high > 0, "regime_vix_high must be > 0")
        _check(self.regime_credit_spread_stress < self.regime_credit_spread_crisis,
               "regime_credit_spread_stress must be < regime_credit_spread_crisis")
        _check(all(v > 0 for v in self.regime_cov_multipliers.values()),
               "regime_cov_multipliers values must be > 0")
        _check(self.ff_factors in (0, 3, 5, 6),
               "ff_factors must be 0, 3, 5, or 6")
        _check(self.ff_frequency in ("daily", "monthly"),
               f"Unknown ff_frequency: {self.ff_frequency}")


@dataclass
class SentimentConfig:
    enabled: bool = True
    news_lookback_days: int = 7
    news_decay_halflife: float = 3.0
    mode: str = "contrarian"
    news_weight: float = 0.18
    vix_weight: float = 0.22
    put_call_weight: float = 0.10
    breadth_weight: float = 0.15
    credit_spread_weight: float = 0.13
    vix_term_weight: float = 0.10
    bank_performance_weight: float = 0.07
    hf_crowding_weight: float = 0.05
    return_adjustment_scale: float = 0.0002
    volatility_adjustment_scale: float = 0.3

    def validate(self):
        _check(self.news_lookback_days > 0, "news_lookback_days must be > 0")
        _check(self.news_decay_halflife > 0, "news_decay_halflife must be > 0")
        _check(self.mode in ("contrarian", "momentum"),
               f"Unknown sentiment mode: {self.mode}")
        total = (self.news_weight + self.vix_weight
                 + self.put_call_weight + self.breadth_weight
                 + self.credit_spread_weight + self.vix_term_weight
                 + self.bank_performance_weight + self.hf_crowding_weight)
        _check(abs(total - 1.0) < 0.01,
               f"Sentiment weights must sum to 1.0, got {total:.3f}")
        _check(self.return_adjustment_scale >= 0,
               "return_adjustment_scale must be >= 0")
        _check(self.volatility_adjustment_scale >= 0,
               "volatility_adjustment_scale must be >= 0")


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
    lookback_window: int = 252
    rebalance_frequency: int = 21
    fixed_cost_per_trade: float = 0.0
    proportional_cost_bps: float = 10.0
    market_impact_coefficient: float = 0.1
    slippage_bps: float = 5.0
    initial_capital: float = 100_000.0
    # Almgren-Chriss market impact model
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
    vol_scaled_caps: bool = False
    vol_cap_base: float = 0.15
    vol_cap_floor: float = 0.02
    vol_cap_lookback: int = 63
    # Turnover penalty
    turnover_penalty_bps: float = 0.0
    # Volatility filter
    max_asset_volatility: float = 0.0
    vol_filter_lookback: int = 63
    vol_filter_min_assets: int = 5
    # Core-satellite allocation
    core_tickers: list = field(default_factory=list)
    core_allocation_pct: float = 0.0
    core_weight_method: str = "inverse_volatility"
    # Cash / risk-off
    exit_to_cash: bool = False
    cash_return_daily: float = 0.0
    # Cross-asset signals
    cross_asset_enabled: bool = False
    cross_asset_dxy_ticker: str = "DX-Y.NYB"
    cross_asset_tnx_ticker: str = "^TNX"
    cross_asset_lookback: int = 21
    cross_asset_max_reduction: float = 0.5
    # Volatility targeting
    vol_target_enabled: bool = False
    vol_target_annualized: float = 0.20
    vol_target_lookback: int = 30
    vol_target_max_leverage: float = 1.0
    vol_target_min_exposure: float = 0.75
    # Conviction-based position filtering
    max_positions: int = 0              # 0 = no limit; >0 = keep top N by weight
    min_position_weight: float = 0.0    # Zero out positions below this threshold
    conviction_threshold: float = 0.0   # Min |alpha_score| z-score to hold (0 = disabled)
    # Minimum trade size filter — skip weight changes smaller than this
    min_trade_weight: float = 0.0       # 0 = disabled; e.g. 0.001 = skip <0.1% trades
    # Short selling
    allow_short: bool = False
    short_borrow_annual_bps: float = 50.0  # Annual borrow cost for short positions
    # Portfolio-level drawdown limit (exit all if portfolio drops this much from peak)
    max_portfolio_drawdown: float = 1.0    # 1.0 = disabled; e.g. 0.25 = exit at -25%

    def validate(self):
        _check(self.lookback_window > 0, "lookback_window must be > 0")
        _check(self.rebalance_frequency > 0,
               "rebalance_frequency must be > 0")
        _check(self.lookback_window >= self.rebalance_frequency,
               "lookback_window must be >= rebalance_frequency")
        _check(self.initial_capital > 0, "initial_capital must be > 0")
        _check(self.fixed_cost_per_trade >= 0,
               "fixed_cost_per_trade must be >= 0")
        _check(0 <= self.proportional_cost_bps <= 100,
               "proportional_cost_bps must be in [0, 100]")
        _check(0 <= self.slippage_bps <= 50,
               "slippage_bps must be in [0, 50]")
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

    @classmethod
    def for_timeframe(cls, data_config: "DataConfig", **overrides) -> "BacktestConfig":
        """Create BacktestConfig with windows auto-scaled to the timeframe."""
        scale = data_config.scale_window
        defaults = {
            "lookback_window": scale(252),
            "rebalance_frequency": scale(21),
            "vol_cap_lookback": scale(63),
            "vol_filter_lookback": scale(63),
            "cross_asset_lookback": scale(21),
            "vol_target_lookback": scale(30),
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_crypto(cls, data_config: "DataConfig", **overrides) -> "BacktestConfig":
        """Create BacktestConfig with crypto-appropriate costs and windows.

        Crypto markets have higher transaction costs than equities:
          - Exchange fees: 5-10bps per trade
          - Bid-ask spread on altcoins: 5-20bps
          - Slippage: 5-30bps depending on liquidity
        """
        scale = data_config.scale_window
        defaults = {
            "lookback_window": scale(252),
            "rebalance_frequency": scale(21),
            "vol_cap_lookback": scale(63),
            "vol_filter_lookback": scale(63),
            "cross_asset_lookback": scale(21),
            "vol_target_lookback": scale(30),
            # Crypto-appropriate costs
            "use_almgren_chriss": True,
            "spread_bps": 10.0,
            "commission_bps": 5.0,
            "slippage_bps": 10.0,
            "eta": 0.15,
            "gamma": 0.08,
            "impact_exponent": 0.6,
            "proportional_cost_bps": 15.0,
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class PropagationConfig:
    """Configuration for correlation propagation and factor decomposition."""
    enabled: bool = True
    base_alpha: float = 0.5
    max_neumann_terms: int = 50
    regime_alpha_scales: dict = field(default_factory=lambda: {
        "expansion": 0.3, "contraction": 0.6,
        "crisis": 0.9, "recovery": 0.4,
    })
    factor_source: str = "factor_model"  # "factor_model" or "pca"
    pca_variance_threshold: float = 0.90
    tensor_window_size: int = 60
    tensor_step_size: int = 5
    kalman_process_noise: float = 1e-5
    kalman_observation_noise: float = 1e-3
    kalman_initial_uncertainty: float = 1.0
    online_enabled: bool = False
    mean_reversion_speed: float = 0.1
    dynamics_enabled: bool = True
    apply_propagation_to_cov: bool = True

    def validate(self):
        _check(0 < self.base_alpha < 1, "base_alpha must be in (0, 1)")
        _check(self.max_neumann_terms > 0, "max_neumann_terms must be > 0")
        _check(all(0 < v < 1 for v in self.regime_alpha_scales.values()),
               "regime_alpha_scales values must be in (0, 1)")
        _check(0 < self.pca_variance_threshold < 1,
               "pca_variance_threshold must be in (0, 1)")
        _check(self.tensor_window_size > 0,
               "tensor_window_size must be > 0")
        _check(self.factor_source in ("factor_model", "pca"),
               f"Unknown factor_source: {self.factor_source}")


@dataclass
class DynamicsConfig:
    """Configuration for Unified Market Dynamics Framework."""
    enabled: bool = True
    # Factor discovery (rolling PCA)
    n_factors: int = 3
    pca_variance_threshold: float = 0.85
    # Granger causality
    granger_lags: int = 5
    granger_p_threshold: float = 0.01
    granger_alpha: float = 0.3
    adaptive_alpha: bool = True
    alpha_safety_factor: float = 0.7
    granger_min_edges: int = 3
    # Volume asymmetry
    volume_lookback: int = 21
    # CUSUM detection
    cusum_drift: float = 1.0
    cusum_threshold: float = 1.5
    # Three-phase model
    # Neutral defaults: per-factor eigenvalue scaling in signal_generator
    # handles phase-differentiated risk. Setting these != 1.0 applies a
    # uniform multiplier on top of the per-factor adjustment, which can
    # create conflicting signals (e.g. 0.8 * 2.0 = 1.6 for dislocation).
    phase_risk_scales: dict = field(default_factory=lambda: {
        "dislocation": 1.3, "discovery": 1.0,
        "convergence": 1.2, "normal": 1.0,
    })
    dislocation_decay_days: int = 10
    convergence_trigger_days: int = 5
    # Signal
    dislocation_return_scale: float = 0.001
    # Dynamic rebalancing
    min_rebalance_interval: int = 5
    max_extra_rebalances: int = 50
    # Volume vector decomposition
    volume_decomposition: bool = True
    ofi_smoothing: int = 5
    momentum_proxy_window: int = 63
    # CUSUM burn-in
    cusum_burn_in: int = 63
    cusum_use_welford: bool = True
    # Contraction constant L
    contraction_enabled: bool = True
    autocorr_window: int = 63
    max_L: float = 0.95
    L_position_scale: bool = True
    # Reflexivity kappa
    reflexivity_enabled: bool = True
    kappa_momentum_window: int = 63
    kappa_autocorr_lag: int = 5
    kappa_clip_max: float = 5.0
    momentum_threshold: float = 1.2
    mean_reversion_threshold: float = 0.8
    # Kappa signal modulation: amplify/flip D_k based on reflexivity regime
    kappa_signal_modulation: bool = False
    kappa_sizing_modulation: bool = False
    # Exit logic
    exit_on_dislocation_collapse: bool = True
    exit_on_max_L: bool = True
    exit_on_phase3: bool = True
    dislocation_collapse_threshold: float = 0.05
    # Granger scaling for large universes
    max_granger_assets: int = 100
    # Transfer entropy (replaces Granger F-test when enabled)
    use_transfer_entropy: bool = False
    te_bins: int = 6
    te_k: int = 1
    te_l: int = 1
    te_threshold: float = 0.10
    te_method: str = "binned"  # "binned" or "knn"
    te_knn_k: int = 5
    te_bootstrap_n: int = 0  # 0 = disabled
    te_bootstrap_alpha: float = 0.05
    # Kalman DFM (post-PCA factor smoothing)
    use_kalman_dfm: bool = False
    kalman_dfm_process_noise: float = 1e-4
    kalman_dfm_obs_noise: float = 1e-3
    # OLS contraction constant (VAR(1) for L)
    use_ols_contraction: bool = False
    # Regression kappa (multivariate regression for kappa)
    use_regression_kappa: bool = False
    # Directional volume decomposition (requires OHLC)
    use_directional_volume: bool = False
    # Cross-sectional alpha signals
    alpha_enabled: bool = False
    alpha_momentum_window: int = 252
    alpha_momentum_skip: int = 21
    alpha_reversal_window: int = 5
    alpha_vol_adjust: bool = True
    alpha_momentum_weight: float = 0.5
    alpha_reversal_weight: float = 0.3
    alpha_vol_mom_weight: float = 0.2
    alpha_tilt_strength: float = 0.5
    alpha_mu_scale: float = 0.0005
    # Causal filtering (do-calculus)
    causal_filter_enabled: bool = False
    causal_min_edge_weight: float = 0.05
    causal_max_confounders: int = 10
    causal_regularization: float = 1e-4
    causal_leadership_weight: float = 0.2
    # Graph VAE (replaces PCA when enabled)
    use_graph_vae: bool = False
    graph_vae_hidden_dim: int = 32
    graph_vae_epochs: int = 100
    graph_vae_lr: float = 1e-3
    graph_vae_kl_weight: float = 0.01
    graph_vae_dropout: float = 0.1
    graph_vae_use_directed: bool = True
    # OFI-based D-hat (proper volume decomposition)
    use_ofi_dhat: bool = False
    ofi_dhat_lookback: int = 21
    dhat_ema_span: int = 5
    dhat_primary_alpha: bool = False
    # Regime-blended alpha: blend D-hat with reversal signal based on kappa
    regime_blend_alpha: bool = False
    regime_blend_momentum_dhat_weight: float = 0.8
    regime_blend_neutral_dhat_weight: float = 0.5
    regime_blend_meanrev_dhat_weight: float = 0.2
    # Lifecycle transition costs
    lifecycle_transaction_costs: bool = False
    # Regime gate: reduce deployment when signal has no edge
    regime_gate_enabled: bool = False
    regime_gate_cash_kappa: float = 0.8       # full cash below this kappa (if VIX normal)
    regime_gate_vix_threshold: float = 1.5    # VIX z-score threshold for ambiguous regime
    regime_gate_ambiguous_deploy: float = 0.5 # deployment fraction in ambiguous regime
    regime_gate_min_deploy: float = 0.2      # floor even in cash regime (prevents whipsaw)
    regime_gate_ema_alpha: float = 0.3       # EMA smoothing for gate transitions
    # Minimum trade size filter (suppress trades below threshold)
    min_trade_size_pct: float = 0.0           # 0 = disabled, e.g. 0.005 = 0.5%
    # Predictive signal engine
    propagation_mispricing_enabled: bool = False
    propagation_mispricing_alpha_weight: float = 0.3   # blend weight into alpha scores
    propagation_mispricing_cumulative_window: int = 5   # rolling cumulative lookback
    gradient_flow_enabled: bool = False                 # equation of motion alpha
    gradient_flow_momentum_blend_cap: float = 0.8       # max D-hat weight in blend
    curvature_regime_enabled: bool = False               # eigenvalue dispersion gate
    curvature_dispersion_lookback: int = 21              # rolling z-score window
    # Ξ Contrarian: |D-hat| × contrarian factor momentum
    xi_contrarian_enabled: bool = False                   # engagement × factor mean-reversion
    xi_contrarian_tilt: float = 0.5                       # blend weight (1.0 = pure Ξ, 0.0 = original alpha)
    xi_phase_scales: dict = field(default_factory=lambda: {
        "dislocation": 1.5,   # strongest signal (IC t=3.21)
        "discovery": 1.0,     # good signal (IC t=2.45)
        "convergence": 0.0,   # no signal
        "normal": 0.0,        # weak/noise
    })
    # Ξ alpha tilt weighting: exp(λ × z(Ξ)) × (1/σ) × rank(|D̂|)
    xi_alpha_tilt_enabled: bool = False              # use xi_alpha_tilt instead of risk_parity
    xi_signal_ema_alpha: float = 0.6                 # signal smoothing: 0.6*today + 0.4*prev
    xi_tilt_blend_rate: float = 0.67                 # turnover blend toward previous weights
    xi_tilt_z_clip: float = 2.0                      # z-score clip before exp()
    xi_tilt_rebalance_band: float = 0.0              # only trade if |target-current| > band
    xi_conviction_percentile: float = 0.0            # top X% by |D̂| (0=no filter, 0.4=top 40%)
    # Alpha overlay: concentrated L/S Ξ on top of core
    xi_overlay_enabled: bool = False                 # enable core + overlay architecture
    xi_overlay_pct: float = 0.25                     # fraction of portfolio in alpha overlay
    xi_overlay_long_pct: float = 0.20                # top 20% by Ξ = long sleeve
    xi_overlay_short_pct: float = 0.20               # bottom 20% by Ξ = short sleeve
    # Lead-lag propagation catch-up signal
    lead_lag_enabled: bool = False                       # master switch
    lead_lag_leader_threshold: float = 2.0               # sigma cutoff for leader detection
    lead_lag_max_lag_days: int = 3                        # lookback horizon (days)
    lead_lag_decay_halflife: float = 1.5                  # exponential decay halflife (days)
    lead_lag_trailing_vol_window: int = 63                # vol estimation window (days)
    lead_lag_gap_ema_alpha: float = 0.5                   # EMA smoothing for turnover control
    lead_lag_min_leader_count: int = 3                    # min leaders needed to generate signal
    lead_lag_max_leader_fraction: float = 0.25            # cap broad market moves
    lead_lag_cascade_clip: float = 3.0                    # outlier clip on predicted cascade
    # Information time & factor vectors
    information_time_alpha_enabled: bool = False          # latency alpha from τ_k
    information_time_alpha_weight: float = 0.5           # weight in alpha blend
    factor_interaction_enabled: bool = False              # cross-factor propagation
    factor_interaction_blend: float = 0.3                # blend weight for F.T @ D_k
    factor_acf_lookback: int = 21                        # ACF window for τ_k
    # Graph-weighted PCA (information propagation channels)
    graph_weighted_pca: bool = False
    # Correlation-aware diversification penalty
    correlation_penalty_enabled: bool = False
    correlation_penalty_threshold: float = 0.80
    correlation_penalty_strength: float = 0.5
    # LSTM factor forecaster (requires torch)
    use_lstm_forecaster: bool = False
    lstm_hidden_dim: int = 32
    lstm_num_layers: int = 2
    lstm_seq_len: int = 63
    lstm_dropout: float = 0.1
    lstm_lr: float = 1e-3
    lstm_epochs: int = 50
    lstm_blend_weight: float = 0.5
    # RL allocator — PPO (requires torch)
    use_rl_allocator: bool = False
    rl_hidden_dim: int = 128
    rl_lr: float = 3e-4
    rl_gamma: float = 0.99
    rl_clip_epsilon: float = 0.2
    rl_update_epochs: int = 4
    rl_buffer_size: int = 32
    rl_entropy_coef: float = 0.01
    rl_value_coef: float = 0.5
    rl_reward_type: str = "return"
    rl_pretrained_path: Optional[str] = None
    rl_training_mode: bool = True
    # RBPF (Rao-Blackwellized Particle Filter on SPD manifold)
    use_rbpf: bool = False
    rbpf_n_particles: int = 50
    rbpf_resample_threshold: float = 0.5
    rbpf_process_noise_scale: float = 0.01
    # RBPF covariance blending: reconstruct N×N from RBPF factor cov
    rbpf_cov_blend: bool = False
    rbpf_cov_blend_weight: float = 0.3
    rbpf_cov_ess_adaptive: bool = True
    # Fixed-point equilibrium iteration
    fixed_point_enabled: bool = False
    fixed_point_max_iter: int = 50
    fixed_point_tol: float = 1e-6
    fixed_point_iterate_dk: bool = False
    fixed_point_damping: float = 0.3
    # Self-referential feedback loop: weights → flow impact → D_k' → signals'
    self_referential_enabled: bool = False
    flow_impact_coefficient: float = 0.01
    self_ref_max_iter: int = 5
    self_ref_tol: float = 1e-4
    # Institutional activity as endogenous factor input
    # (always computed when volume data available — weight=0 disables)
    institutional_feature_weight: float = 0.1
    institutional_coordination_lookback: int = 21
    # Continuous macro factor integration
    macro_factor_enabled: bool = False
    macro_residualize_pca: bool = True
    macro_tickers: list = field(default_factory=lambda: ["vix", "yield_spread", "dxy"])
    # POFM: Fama-French residualization before PCA
    ff_residualize_pca: bool = False       # Regress out FF factors before PCA
    ff_factors: int = 5                    # 3, 5, or 6 (FF5+MOM) factors
    ff_eigenvalue_test: bool = False       # Data-driven K via eigenvalue significance test
    ff_eigenvalue_significance: float = 0.05  # Significance level for eigenvalue test
    ff_k_min: int = 1                      # Minimum latent factors (safety floor)
    ff_k_max: int = 0                      # Maximum latent factors (0 = use n_factors)
    endogeneity_diagnostic: bool = False   # Log UECL endogeneity diagnostic per rebalance
    use_manifold_L: bool = False
    use_lyapunov_equilibrium: bool = False
    use_nonlinear_operator: bool = False
    # Hurst exponent (continuous kappa replacement)
    use_hurst_exponent: bool = False
    hurst_min_lag: int = 2
    hurst_max_lag: int = 20
    # Information-time CUSUM weighting
    information_time_cusum: bool = False
    info_weight_vol_lookback: int = 21
    # Liquidity-adjusted Granger weights
    liquidity_adjusted_granger: bool = False
    # Optimal stopping rebalance (value-of-information)
    use_optimal_stopping_rebalance: bool = False
    optimal_stopping_cost_bps: float = 10.0
    # Sentiment integration (A/B testable)
    sentiment_cusum_modulation: bool = False   # modulate CUSUM threshold with sentiment
    sentiment_kappa_blend: bool = False         # blend kappa with sentiment direction
    sentiment_cusum_scale: float = 0.3         # max ±30% threshold adjustment
    sentiment_kappa_scale: float = 0.3         # max ±30% kappa adjustment
    # Bayesian edge updates (Beta-Binomial conjugate)
    use_bayesian_edges: bool = False
    bayesian_edge_prior_alpha: float = 1.0
    bayesian_edge_prior_beta: float = 1.0
    bayesian_edge_decay: float = 0.995
    # Insider activity signal
    use_insider_signal: bool = False
    # Fiber Bundle State Space
    fiber_bundle_enabled: bool = False
    proper_time_method: str = "realized_variance"   # or "information_time"
    bundle_metric_coupling: float = 0.0             # off-diagonal base-fiber coupling
    bundle_trajectory_log: bool = False              # log full state to dynamics_log
    bundle_blend_window: int = 5                     # days to blend during K transitions
    # Conviction-Direction-Risk (CDR) weight construction
    cdr_enabled: bool = False                          # master switch for C×D×R weights
    cdr_conviction_max: float = 3.0                    # max rank-scaled conviction
    cdr_momentum_weight: float = 0.7                   # 12-1 momentum weight in direction blend
    cdr_reversal_weight: float = 0.3                   # 5-day reversal weight in direction blend
    cdr_short_exposure: float = 0.0                    # short sleeve gross (0 = long-only)
    cdr_min_hold_days: int = 15                        # don't exit before this many days
    cdr_rebalance_freq: int = 5                        # rebalance every N days
    cdr_max_weight_change: float = 0.20                # max absolute weight change per rebalance
    cdr_phase_scales: dict = field(default_factory=lambda: {
        "dislocation": 1.0,   # full conviction — follow the break
        "discovery": 1.0,     # full conviction — new regime
        "convergence": 0.3,   # dampen — mean reversion
        "normal": 0.3,        # dampen — low signal
    })
    cdr_vol_lookback: int = 63                         # trailing vol window for R_i
    cdr_momentum_window: int = 252                     # 12-month momentum lookback
    cdr_momentum_skip: int = 21                        # skip recent month
    cdr_reversal_window: int = 5                       # short-term reversal window
    cdr_conviction_percentile: float = 0.30            # top 30% by conviction
    cdr_direction_threshold: float = 0.3               # min |z| for direction
    cdr_xi_conviction_blend: float = 0.0               # blend Xi contrarian into CDR conviction [0,1]
    # === Correlation Manifold Dynamics (closes 7 approximation gaps) ===
    # Master switch for second-order geodesic dynamics on SPD(K)
    correlation_manifold_enabled: bool = False
    # Correlation flow tracker (Approximation 5: velocity tensor)
    correlation_flow_max_history: int = 63             # rolling covariance snapshot buffer
    correlation_flow_meta_window: int = 21             # meta-correlation computation window
    correlation_flow_velocity_ema: float = 0.3         # EMA smoothing for velocity field
    # Manifold dynamics engine (Approximation 2: geodesic flow; Approximation 4: continuous FP)
    manifold_dynamics_potential_strength: float = 0.1  # lambda: harmonic pull toward equilibrium
    manifold_dynamics_damping: float = 0.05            # gamma: friction coefficient
    manifold_dynamics_max_velocity: float = 1.0        # safety clamp on velocity norm
    manifold_dynamics_integrator: str = "verlet"       # "verlet" (symplectic) or "euler"
    manifold_dynamics_blend_weight: float = 0.3        # blend evolved sigma with PCA/RBPF
    # Field equation / backreaction (G = kappa * T)
    field_equation_coupling: float = 0.01              # kappa: stress-energy coupling constant
    field_equation_volume_weight: float = 0.4          # volume stress weight in T
    field_equation_return_weight: float = 0.3          # return stress weight in T
    field_equation_weight_change_weight: float = 0.3   # weight change stress weight in T
    # Operator spectrum (Approximation 6: full spectral decomposition)
    operator_spectrum_enabled: bool = False
    operator_spectrum_hedge: bool = False               # hedge unstable modes
    operator_spectrum_threshold: float = 0.95           # marginal/unstable boundary
    # Grassmannian factor flow (Approximation 3: smooth K transitions)
    grassmannian_enabled: bool = False
    grassmannian_blend_window: int = 5                 # days to smooth K transitions
    # Soft phases (Approximation 7: continuous phase distribution)
    soft_phases_enabled: bool = False
    soft_phase_steepness: float = 5.0                  # sigmoid sharpness at threshold
    soft_phase_rbpf_blend: float = 0.5                 # weight RBPF probs vs CUSUM probs
    # Safety
    manifold_energy_divergence_threshold: float = 10.0

    def validate(self):
        _check(self.n_factors > 0, "n_factors must be > 0")
        _check(0 < self.pca_variance_threshold < 1,
               "pca_variance_threshold must be in (0, 1)")
        _check(self.granger_lags > 0, "granger_lags must be > 0")
        _check(0 < self.granger_p_threshold < 1,
               "granger_p_threshold must be in (0, 1)")
        _check(0 < self.granger_alpha < 1,
               "granger_alpha must be in (0, 1)")
        _check(0 < self.alpha_safety_factor < 1,
               "alpha_safety_factor must be in (0, 1)")
        _check(self.volume_lookback > 0, "volume_lookback must be > 0")
        _check(self.cusum_drift > 0, "cusum_drift must be > 0")
        _check(self.cusum_threshold > 0, "cusum_threshold must be > 0")
        _check(all(v > 0 for v in self.phase_risk_scales.values()),
               "phase_risk_scales values must be > 0")
        _check(self.dislocation_decay_days > 0,
               "dislocation_decay_days must be > 0")
        _check(0 < self.max_L <= 1, "max_L must be in (0, 1]")
        _check(self.momentum_threshold > self.mean_reversion_threshold,
               "momentum_threshold must be > mean_reversion_threshold")
        _check(self.cusum_burn_in > 0, "cusum_burn_in must be > 0")
        for _bw in (self.regime_blend_momentum_dhat_weight,
                     self.regime_blend_neutral_dhat_weight,
                     self.regime_blend_meanrev_dhat_weight):
            _check(0 <= _bw <= 1, "regime_blend dhat weights must be in [0, 1]")
        _check(self.max_granger_assets > 0,
               "max_granger_assets must be > 0")
        if self.rbpf_cov_blend:
            _check(0 <= self.rbpf_cov_blend_weight <= 1,
                   "rbpf_cov_blend_weight must be in [0, 1]")
        if self.use_lstm_forecaster:
            _check(self.lstm_hidden_dim > 0,
                   "lstm_hidden_dim must be > 0")
            _check(self.lstm_seq_len > 0, "lstm_seq_len must be > 0")
            _check(0 <= self.lstm_blend_weight <= 1,
                   "lstm_blend_weight must be in [0, 1]")
        if self.use_rl_allocator:
            _check(self.rl_hidden_dim > 0, "rl_hidden_dim must be > 0")
            _check(self.rl_buffer_size > 0, "rl_buffer_size must be > 0")
            _check(0 < self.rl_gamma <= 1, "rl_gamma must be in (0, 1]")
            _check(0 < self.rl_clip_epsilon < 1,
                   "rl_clip_epsilon must be in (0, 1)")
            _check(self.rl_reward_type in ("return", "sharpe"),
                   f"Unknown rl_reward_type: {self.rl_reward_type}")
        if self.ff_residualize_pca:
            _check(self.ff_factors in (3, 5, 6),
                   "ff_factors must be 3, 5, or 6 when ff_residualize_pca is enabled")
        if self.ff_eigenvalue_test:
            _check(0 < self.ff_eigenvalue_significance < 1,
                   "ff_eigenvalue_significance must be in (0, 1)")
            _check(self.ff_k_min >= 1, "ff_k_min must be >= 1")
            _check(self.ff_k_max >= 0, "ff_k_max must be >= 0")
        if self.fiber_bundle_enabled:
            _check(self.proper_time_method in ("realized_variance", "information_time"),
                   "proper_time_method must be 'realized_variance' or 'information_time'")
            _check(0.0 <= self.bundle_metric_coupling <= 1.0,
                   "bundle_metric_coupling must be in [0, 1]")
            _check(self.bundle_blend_window > 0,
                   "bundle_blend_window must be > 0")
        if self.cdr_enabled:
            _check(self.cdr_conviction_max > 0,
                   "cdr_conviction_max must be > 0")
            _check(0.0 <= self.cdr_short_exposure <= 1.0,
                   "cdr_short_exposure must be in [0, 1]")
            _check(self.cdr_min_hold_days >= 0,
                   "cdr_min_hold_days must be >= 0")
            _check(self.cdr_rebalance_freq > 0,
                   "cdr_rebalance_freq must be > 0")
            _check(0 < self.cdr_max_weight_change <= 1.0,
                   "cdr_max_weight_change must be in (0, 1]")
            _check(self.cdr_vol_lookback > 0,
                   "cdr_vol_lookback must be > 0")
            _check(0.0 <= self.cdr_xi_conviction_blend <= 1.0,
                   "cdr_xi_conviction_blend must be in [0, 1]")
        if self.correlation_manifold_enabled:
            _check(0 < self.manifold_dynamics_potential_strength < 10,
                   "manifold_dynamics_potential_strength must be in (0, 10)")
            _check(0 <= self.manifold_dynamics_damping < 1,
                   "manifold_dynamics_damping must be in [0, 1)")
            _check(self.manifold_dynamics_max_velocity > 0,
                   "manifold_dynamics_max_velocity must be > 0")
            _check(self.manifold_dynamics_integrator in ("verlet", "euler"),
                   f"Unknown integrator: {self.manifold_dynamics_integrator}")
            _check(0 <= self.manifold_dynamics_blend_weight <= 1,
                   "manifold_dynamics_blend_weight must be in [0, 1]")
            _check(self.field_equation_coupling > 0,
                   "field_equation_coupling must be > 0")
            _check(self.manifold_energy_divergence_threshold > 0,
                   "manifold_energy_divergence_threshold must be > 0")
        if self.grassmannian_enabled:
            _check(self.grassmannian_blend_window > 0,
                   "grassmannian_blend_window must be > 0")
        if self.lead_lag_enabled:
            _check(self.lead_lag_leader_threshold > 0,
                   "lead_lag_leader_threshold must be > 0")
            _check(1 <= self.lead_lag_max_lag_days <= 10,
                   "lead_lag_max_lag_days must be in [1, 10]")
            _check(self.lead_lag_decay_halflife > 0,
                   "lead_lag_decay_halflife must be > 0")
            _check(self.lead_lag_trailing_vol_window > 0,
                   "lead_lag_trailing_vol_window must be > 0")
            _check(0 < self.lead_lag_gap_ema_alpha <= 1,
                   "lead_lag_gap_ema_alpha must be in (0, 1]")

    @classmethod
    def for_large_universe(cls, n_assets: int = 500) -> "DynamicsConfig":
        """Factory for S&P 500 scale backtests.

        Adjusts defaults for computational feasibility at N=500:
        more factors, fewer Granger lags, subsampled causality.
        """
        return cls(
            enabled=True,
            n_factors=5,
            pca_variance_threshold=0.80,
            granger_lags=3,
            granger_p_threshold=0.01,
            granger_alpha=0.3,
            adaptive_alpha=True,
            alpha_safety_factor=0.7,
            granger_min_edges=5,
            max_granger_assets=min(100, n_assets),
            volume_lookback=21,
            cusum_drift=1.0,
            cusum_threshold=1.5,
            dislocation_return_scale=0.0005,
            min_rebalance_interval=5,
            max_extra_rebalances=30,
            volume_decomposition=True,
            cusum_burn_in=63,
            cusum_use_welford=True,
            contraction_enabled=True,
            autocorr_window=63,
            max_L=0.95,
            L_position_scale=True,
            reflexivity_enabled=True,
            kappa_momentum_window=63,
            kappa_autocorr_lag=5,
            use_transfer_entropy=True,
            use_ols_contraction=True,
            use_regression_kappa=True,
            use_kalman_dfm=True,
            use_directional_volume=True,
            # Alpha signals
            alpha_enabled=False,   # D-hat is primary alpha, not momentum/reversal
            alpha_tilt_strength=1.0,
            alpha_mu_scale=0.0003,
            # Causal filtering
            causal_filter_enabled=True,
            causal_max_confounders=10,
            causal_leadership_weight=0.2,
            # Graph VAE (off by default — requires torch)
            use_graph_vae=False,
            # OFI-based D-hat
            use_ofi_dhat=True,
            ofi_dhat_lookback=21,
            dhat_ema_span=5,
            dhat_primary_alpha=True,
            # Graph-weighted PCA
            graph_weighted_pca=True,
            # RBPF and fixed-point (off by default — research features)
            use_rbpf=False,
            fixed_point_enabled=False,
            # POFM: FF5 residualization before PCA
            ff_residualize_pca=True,
            ff_factors=5,
            ff_eigenvalue_test=True,
            ff_k_min=2,
            ff_k_max=8,
        )

    @classmethod
    def for_timeframe(cls, data_config: "DataConfig", **overrides) -> "DynamicsConfig":
        """Create DynamicsConfig with temporal windows scaled to the timeframe.

        CUSUM parameters are scaled by √(bars_per_day):
          - drift scales DOWN (hourly returns are smaller)
          - threshold scales UP (need more evidence at higher frequency)
        This prevents noise-driven phase transitions at intraday frequencies.
        """
        scale = data_config.scale_window
        bpd = data_config._effective_bars_per_day
        cusum_scale = math.sqrt(bpd)  # √1 = 1.0 for daily, √24 ≈ 4.9 for 1h
        defaults = {
            "volume_lookback": scale(21),
            "dislocation_decay_days": scale(10),
            "convergence_trigger_days": scale(5),
            "min_rebalance_interval": scale(5),
            "cusum_burn_in": scale(63),
            "cusum_drift": 1.0 / cusum_scale,
            "cusum_threshold": 1.5 * cusum_scale,
            "autocorr_window": scale(63),
            "kappa_momentum_window": scale(63),
            "momentum_proxy_window": scale(63),
            "alpha_momentum_window": scale(252),
            "alpha_momentum_skip": scale(21),
            "alpha_reversal_window": scale(5),
            "institutional_coordination_lookback": scale(21),
            "ofi_dhat_lookback": scale(21),
            "lstm_seq_len": scale(63),
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass
class AgentConfig:
    """Configuration for LLM Semantic Orchestration agents.

    Four strategies that contextualize mathematical signals:
    1. Semantic Factor — labels PCA factors, predicts persistence
    2. CUSUM Validator — validates dislocation events (signal vs noise)
    3. Macro Reflexivity — overrides kappa from macro regime shifts
    4. Exit Watchdog — analyzes positions near trailing stop
    """
    enabled: bool = False                           # Master switch
    mode: str = "backtest"                          # "backtest" (heuristic) or "live" (API)
    openai_api_key: Optional[str] = None            # From env OPENAI_API_KEY if None
    openai_model: str = "gpt-4o"                    # Model for live mode
    openai_timeout: float = 10.0                    # API timeout seconds

    # Strategy switches
    semantic_factor_enabled: bool = True             # Strategy 1: factor labeling
    cusum_validator_enabled: bool = True             # Strategy 2: CUSUM validation
    macro_reflexivity_enabled: bool = True           # Strategy 3: kappa override
    exit_watchdog_enabled: bool = True               # Strategy 4: exit analysis

    # Strategy 2: CUSUM validator parameters
    cusum_dispersion_threshold_high: float = 0.6     # >this = broad cascade
    cusum_dispersion_threshold_low: float = 0.3      # <this = sector-specific
    cusum_volume_spike_threshold: float = 2.0        # volume ratio for panic detection
    cusum_signal_amplify: float = 1.5                # D_k multiplier for "signal" type
    cusum_noise_dampen: float = 0.3                  # D_k multiplier for "noise" type

    # Strategy 3: Macro reflexivity parameters
    macro_vix_zscore_threshold: float = 2.0          # VIX 3-day z-score trigger
    macro_credit_spread_velocity: float = 0.5        # 5-day spread change trigger (pp)
    macro_kappa_momentum_override: float = 1.5       # kappa when momentum detected
    macro_kappa_meanrev_override: float = 0.7        # kappa when mean-reversion detected

    # Strategy 4: Exit watchdog parameters
    exit_idiosyncratic_threshold: float = 2.0        # sigma for stock-specific detection
    exit_sector_decline_threshold: float = -0.015    # -1.5% for broad liquidation
    exit_volume_capitulation: float = 3.0            # volume ratio for panic detection
    exit_stop_widen_factor: float = 1.3              # widen stop by 30% on "hold"
    exit_min_confidence: float = 0.6                 # min confidence to override exit

    # Learning store
    llm_log_dir: str = "results/agent_decisions"     # Decision log directory
    learning_enabled: bool = True                    # Track decisions + P&L
    learning_pnl_horizon: int = 30                   # Days to wait for P&L evaluation

    # Safety
    max_kappa_adjustment: float = 0.5                # Max |kappa_override - kappa_base|
    fallback_on_error: bool = True                   # Revert to base signal on any error

    def validate(self):
        _check(self.mode in ("backtest", "live"),
               f"Unknown agent mode: {self.mode}")
        _check(0 < self.cusum_signal_amplify <= 3.0,
               "cusum_signal_amplify must be in (0, 3]")
        _check(0 < self.cusum_noise_dampen <= 1.0,
               "cusum_noise_dampen must be in (0, 1]")
        _check(0 < self.max_kappa_adjustment <= 2.0,
               "max_kappa_adjustment must be in (0, 2]")
        _check(0 <= self.exit_min_confidence <= 1.0,
               "exit_min_confidence must be in [0, 1]")
        _check(1.0 <= self.exit_stop_widen_factor <= 2.0,
               "exit_stop_widen_factor must be in [1, 2]")


@dataclass
class FactorLabConfig:
    """Configuration for FactorLab walk-forward alpha evaluation."""
    lookback: int = 252
    rebalance_frequency: int = 21
    n_quantiles: int = 5
    long_short: bool = True
    ic_lags: list = field(default_factory=lambda: [1, 5, 21])
    institutional_lookback: int = 63
    volume_spike_threshold: float = 2.0
    macro_momentum_window: int = 21
    macro_zscore_window: int = 63

    def validate(self):
        _check(self.lookback > 0, "lookback must be > 0")
        _check(self.rebalance_frequency > 0,
               "rebalance_frequency must be > 0")
        _check(2 <= self.n_quantiles <= 20,
               "n_quantiles must be in [2, 20]")
        _check(self.institutional_lookback > 0,
               "institutional_lookback must be > 0")
        _check(self.volume_spike_threshold > 0,
               "volume_spike_threshold must be > 0")
        _check(self.macro_momentum_window > 0,
               "macro_momentum_window must be > 0")
        _check(self.macro_zscore_window > 0,
               "macro_zscore_window must be > 0")
        _check(all(lag > 0 for lag in self.ic_lags),
               "All ic_lags must be > 0")


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


@dataclass
class EventAttributionConfig:
    """Configuration for event attribution on CUSUM-detected dislocations."""
    enabled: bool = True
    # Asset attribution
    top_n_assets: int = 5
    min_contribution_pct: float = 0.05
    # News source
    news_lookback_days: int = 3
    news_source: str = "yahoo_rss"
    newsapi_key: Optional[str] = None
    # LLM escalation
    llm_enabled: bool = False
    llm_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_confidence_threshold: float = 0.5
    llm_max_tokens: int = 500
    # Signal modifiers per event type
    signal_modifiers: dict = field(default_factory=lambda: {
        "earnings_surprise": 1.0,
        "index_rebalance": -0.5,
        "unknown_no_news": 1.5,
        "m_and_a": 0.7,
        "regulatory": 0.3,
        "macro": 0.8,
        "technical": 1.0,
    })
    # Classification
    min_headline_count: int = 2
    # Backtest mode
    backtest_mode: bool = False
    cache_path: Optional[str] = None
    # Persistence
    store_results: bool = True

    def validate(self):
        _check(self.top_n_assets > 0, "top_n_assets must be > 0")
        _check(0 < self.min_contribution_pct < 1,
               "min_contribution_pct must be in (0, 1)")
        _check(self.news_lookback_days > 0,
               "news_lookback_days must be > 0")
        _check(self.news_source in ("yahoo_rss", "newsapi"),
               f"Unknown news_source: {self.news_source}")
        _check(0 < self.llm_confidence_threshold <= 1,
               "llm_confidence_threshold must be in (0, 1]")
        _check(all(isinstance(v, (int, float))
                   for v in self.signal_modifiers.values()),
               "signal_modifiers values must be numeric")


@dataclass
class PositionManagerConfig:
    """Configuration for Position Lifecycle Manager.

    Controls the entry/exit state machine that selectively enters and
    exits individual positions based on composite signal scoring.
    """
    enabled: bool = True

    # --- Entry model weights (sum to composite score) ---
    entry_factor_weight: float = 0.30    # weight for factor exposure signal
    entry_alpha_weight: float = 0.25     # weight for alpha z-score
    entry_granger_weight: float = 0.15   # weight for Granger in-degree bonus
    entry_pagerank_weight: float = 0.10  # weight for PageRank centrality
    entry_irf_weight: float = 0.10       # weight for IRF lead-lag
    entry_insider_weight: float = 0.0    # disabled until data source connected
    entry_institutional_weight: float = 0.0  # weight for institutional consensus signal
    entry_propagation_weight: float = 0.10   # weight for propagation speed signal
    # Predictive signal entry weights
    entry_mispricing_weight: float = 0.0     # propagation mispricing (M @ r_{t-1} - r_t)
    entry_latent_shock_weight: float = 0.0   # latent shock source (high M + low recent move)
    entry_sia_weight: float = 0.0            # SIA outgoing importance (row sums of |M|)
    entry_lead_lag_weight: float = 0.0       # lead-lag catch-up signal (component 12)
    entry_crowding_weight: float = 0.0       # HF crowding contrarian signal (component 13)

    # --- Entry thresholds ---
    entry_threshold: float = 1.0         # z-score cutoff above median
    max_positions: int = 12              # max simultaneous ACTIVE positions
    min_positions: int = 3               # min before forcing lower threshold

    # --- Cash re-entry patience ---
    cash_reentry_patience: int = 2       # consecutive all-cash rebalances before lowering threshold
    reentry_threshold_factor: float = 0.3  # multiply entry_threshold by this after patience expires

    # --- Exit model ---
    signal_decay_threshold: float = 0.0   # exit when signal drops below 0 (negative = thesis lost)
    min_holding_days: int = 10            # no signal/thesis exits before this many days
    trailing_stop_pct: float = 0.15       # 15% drawdown from peak price (fallback)
    atr_stop_multiplier: float = 0.0     # ATR multiplier for trailing stop (0 = use fixed pct)
    atr_lookback: int = 14               # ATR computation window in days
    max_holding_days: int = 126           # two quarters max hold
    exit_L_threshold: float = 0.92        # per-position contraction breach
    dhat_exit_threshold: float = 0.03     # D_hat norm must be below this for thesis_reversal exit

    # --- Regime thresholds (shared entry/exit) ---
    kappa_momentum_threshold: float = 1.2
    kappa_mean_rev_threshold: float = 0.8
    regime_momentum_multiplier: float = 1.3
    regime_mean_rev_multiplier: float = 1.0

    # --- Quality gate ---
    entry_rmw_quality_gate: bool = False
    entry_rmw_min_beta: float = -0.1

    # --- Position sizing ---
    max_position_size: float = 0.15       # 15% max per asset
    max_total_exposure: float = 1.0       # no leverage
    min_position_size: float = 0.02       # 2% minimum

    # --- Cooldown ---
    cooldown_days: int = 10               # days before re-entry after exit

    # --- Short selling ---
    allow_short: bool = False             # enable short entries (negative scores)

    # --- Burn-in ---
    burn_in_days: int = 63               # days before lifecycle activates

    # --- Sizing blend ---
    lifecycle_sizing_blend: float = 0.4   # blend weight for lifecycle vs optimizer

    # --- Lorentzian kNN Classifier ---
    lorentzian_enabled: bool = False          # master switch
    lorentzian_k: int = 8                     # k nearest neighbors
    lorentzian_max_bars: int = 2000           # max training history
    lorentzian_label_bars: int = 4            # forward bars for training labels
    lorentzian_warmup_bars: int = 200         # bars before classifier activates
    lorentzian_rq_h: float = 8.0             # Rational Quadratic bandwidth
    lorentzian_rq_r: float = 8.0             # RQ relative weighting
    lorentzian_rq_x: int = 25                # RQ lookback
    lorentzian_gauss_h: float = 6.0          # Gaussian bandwidth
    lorentzian_gauss_x: int = 25             # Gaussian lookback
    lorentzian_vol_filter: bool = True        # suppress in extreme vol
    lorentzian_regime_filter: bool = True     # suppress in strong downtrend
    lorentzian_regime_threshold: float = -0.1 # slope threshold
    lorentzian_entry_gate: bool = False       # entry gate too restrictive; use exit kernel only
    lorentzian_exit_kernel: bool = True       # use kernel bearish change as exit

    def validate(self):
        _check(self.entry_factor_weight >= 0,
               "entry_factor_weight must be >= 0")
        _check(self.entry_alpha_weight >= 0,
               "entry_alpha_weight must be >= 0")
        _check(self.entry_granger_weight >= 0,
               "entry_granger_weight must be >= 0")
        _check(0 < self.entry_threshold <= 3.0,
               "entry_threshold must be in (0, 3.0]")
        _check(self.max_positions > 0,
               "max_positions must be > 0")
        _check(self.min_positions > 0,
               "min_positions must be > 0")
        _check(self.min_positions <= self.max_positions,
               "min_positions must be <= max_positions")
        _check(-1 <= self.signal_decay_threshold <= 1,
               "signal_decay_threshold must be in [-1, 1]")
        _check(self.min_holding_days >= 0,
               "min_holding_days must be >= 0")
        _check(0 < self.trailing_stop_pct < 1,
               "trailing_stop_pct must be in (0, 1)")
        _check(self.atr_stop_multiplier >= 0,
               "atr_stop_multiplier must be >= 0")
        _check(self.atr_lookback > 0,
               "atr_lookback must be > 0")
        _check(self.entry_propagation_weight >= 0,
               "entry_propagation_weight must be >= 0")
        _check(self.entry_mispricing_weight >= 0,
               "entry_mispricing_weight must be >= 0")
        _check(self.entry_latent_shock_weight >= 0,
               "entry_latent_shock_weight must be >= 0")
        _check(self.entry_sia_weight >= 0,
               "entry_sia_weight must be >= 0")
        _check(self.entry_lead_lag_weight >= 0,
               "entry_lead_lag_weight must be >= 0")
        _check(self.max_holding_days > 0,
               "max_holding_days must be > 0")
        _check(0 < self.exit_L_threshold <= 1,
               "exit_L_threshold must be in (0, 1]")
        _check(self.kappa_momentum_threshold > self.kappa_mean_rev_threshold,
               "kappa_momentum_threshold must be > kappa_mean_rev_threshold")
        _check(0 < self.max_position_size <= 1,
               "max_position_size must be in (0, 1]")
        _check(0 < self.max_total_exposure <= 2.0,
               "max_total_exposure must be in (0, 2.0]")
        _check(self.cooldown_days >= 0,
               "cooldown_days must be >= 0")
        _check(self.burn_in_days >= 0,
               "burn_in_days must be >= 0")
        _check(0 <= self.lifecycle_sizing_blend <= 1,
               "lifecycle_sizing_blend must be in [0, 1]")
        if self.lorentzian_enabled:
            _check(self.lorentzian_k > 0,
                   "lorentzian_k must be > 0")
            _check(self.lorentzian_max_bars > 0,
                   "lorentzian_max_bars must be > 0")
            _check(self.lorentzian_label_bars > 0,
                   "lorentzian_label_bars must be > 0")
            _check(self.lorentzian_warmup_bars >= 50,
                   "lorentzian_warmup_bars must be >= 50")
            _check(self.lorentzian_rq_h > 0,
                   "lorentzian_rq_h must be > 0")
            _check(self.lorentzian_rq_r > 0,
                   "lorentzian_rq_r must be > 0")
            _check(self.lorentzian_rq_x > 0,
                   "lorentzian_rq_x must be > 0")
            _check(self.lorentzian_gauss_h > 0,
                   "lorentzian_gauss_h must be > 0")
            _check(self.lorentzian_gauss_x > 0,
                   "lorentzian_gauss_x must be > 0")


@dataclass
class InstitutionalConfig:
    """Configuration for institutional trade analysis system."""
    enabled: bool = False

    # EDGAR access
    edgar_user_agent: str = ""     # Required: "Name email@example.com"

    # Target institutions (CIK numbers)
    target_ciks: list = field(default_factory=list)

    # Cache
    cache_dir: Optional[str] = None  # None = ~/.NR/institutional_cache

    # Signal generation
    consensus_lookback_quarters: int = 4
    consensus_weight_in_entry: float = 0.0  # 0 = disabled
    insider_lookback_days: int = 90
    insider_weight_in_entry: float = 0.0    # 0 = disabled

    # Analysis
    forward_return_horizons: list = field(
        default_factory=lambda: [5, 21, 63])
    n_strategy_clusters: int = 4

    # Point-in-time snapshots
    snapshot_lookback_days: int = 252
    snapshot_n_factors: int = 5

    # Propagation speed signal
    propagation_enabled: bool = False
    propagation_lookback_quarters: int = 4
    propagation_decay_halflife_quarters: float = 2.0

    # HF crowding signal
    crowding_enabled: bool = False
    crowding_weight_in_entry: float = 0.0
    crowding_lookback_quarters: int = 4

    # Rate limiting
    edgar_requests_per_second: float = 8.0

    def validate(self):
        if self.enabled:
            _check(len(self.edgar_user_agent) > 0,
                   "edgar_user_agent required when institutional analysis enabled")
            _check(len(self.target_ciks) > 0,
                   "At least one target CIK required")
        _check(self.consensus_lookback_quarters > 0,
               "consensus_lookback_quarters must be > 0")
        _check(0 <= self.consensus_weight_in_entry <= 1,
               "consensus_weight_in_entry must be in [0, 1]")
        _check(0 <= self.insider_weight_in_entry <= 1,
               "insider_weight_in_entry must be in [0, 1]")
        _check(self.edgar_requests_per_second > 0,
               "edgar_requests_per_second must be > 0")
