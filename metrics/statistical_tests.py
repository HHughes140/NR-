"""Statistical validation tests for portfolio backtest results.

Provides rigorous hypothesis testing for backtest metrics:
- Bootstrap confidence intervals for Sharpe ratio
- Newey-West HAC alpha significance testing
- Lo (2002) autocorrelation-corrected Sharpe ratio test
- Spearman rank information coefficient
"""

import numpy as np
from scipy import stats


class StatisticalTests:
    """Statistical tests for validating backtest results."""

    @staticmethod
    def bootstrap_sharpe_ci(
        returns: np.ndarray,
        n_bootstrap: int = 10_000,
        ci: float = 0.95,
        block_size: int = 20,
        seed: int = 42,
        periods_per_year: int = 252,
    ) -> tuple[float, float, float]:
        """Block bootstrap confidence interval for Sharpe ratio.

        Uses circular block bootstrap to preserve autocorrelation structure
        in returns, giving more realistic confidence intervals than IID bootstrap.

        Args:
            returns: 1D array of portfolio returns.
            n_bootstrap: Number of bootstrap resamples.
            ci: Confidence level (e.g. 0.95 for 95% CI).
            block_size: Block length for block bootstrap.
            seed: Random seed for reproducibility.

        Returns:
            (sharpe, ci_lower, ci_upper) — annualized Sharpe and CI bounds.
        """
        returns = np.asarray(returns)
        T = len(returns)
        if T < 2:
            return 0.0, 0.0, 0.0

        rng = np.random.default_rng(seed)

        # Point estimate
        sharpe = float(np.mean(returns) / (np.std(returns, ddof=1) + 1e-12) * np.sqrt(periods_per_year))

        # Circular block bootstrap
        n_blocks = (T + block_size - 1) // block_size
        boot_sharpes = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            starts = rng.integers(0, T, size=n_blocks)
            indices = np.concatenate([
                np.arange(s, s + block_size) % T for s in starts
            ])[:T]
            sample = returns[indices]
            std = np.std(sample, ddof=1)
            if std > 1e-12:
                boot_sharpes[b] = np.mean(sample) / std * np.sqrt(periods_per_year)
            else:
                boot_sharpes[b] = 0.0

        alpha = 1 - ci
        lower = float(np.percentile(boot_sharpes, 100 * alpha / 2))
        upper = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))

        return sharpe, lower, upper

    @staticmethod
    def alpha_significance(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        confidence: float = 0.95,
        max_lags: int = None,
        periods_per_year: int = 252,
    ) -> tuple[float, float, float, bool]:
        """Newey-West HAC t-test on CAPM alpha.

        Tests H0: alpha = 0 using heteroskedasticity and autocorrelation
        consistent (HAC) standard errors, which are robust to serial
        correlation in residuals.

        Args:
            portfolio_returns: 1D array of portfolio daily returns.
            benchmark_returns: 1D array of benchmark daily returns.
            confidence: Confidence level for significance.
            max_lags: Newey-West lag truncation (default: floor(T^{1/3})).

        Returns:
            (alpha_annualized, t_stat, p_value, significant)
        """
        port = np.asarray(portfolio_returns)
        bench = np.asarray(benchmark_returns)
        T = len(port)

        if max_lags is None:
            max_lags = max(1, int(T ** (1 / 3)))

        # OLS: port = alpha + beta * bench + eps
        X = np.column_stack([np.ones(T), bench])
        beta = np.linalg.lstsq(X, port, rcond=None)[0]
        residuals = port - X @ beta
        alpha_daily = beta[0]

        # Newey-West HAC variance of alpha
        # Meat: S = sum_{j=-L}^{L} w(j) * Gamma(j) where Gamma(j) = (1/T) sum e_t * x_t * x_{t-j}^T * e_{t-j}
        XtX_inv = np.linalg.inv(X.T @ X / T)
        S = np.zeros((2, 2))
        for lag in range(max_lags + 1):
            weight = 1.0 - lag / (max_lags + 1)  # Bartlett kernel
            if lag == 0:
                G = sum(residuals[t] ** 2 * np.outer(X[t], X[t]) for t in range(T)) / T
                S += G
            else:
                G = sum(
                    residuals[t] * residuals[t - lag] * np.outer(X[t], X[t - lag])
                    for t in range(lag, T)
                ) / T
                S += weight * (G + G.T)

        V = XtX_inv @ S @ XtX_inv / T
        se_alpha = np.sqrt(max(V[0, 0], 1e-20))

        t_stat = alpha_daily / se_alpha
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 2))
        alpha_annual = alpha_daily * periods_per_year

        return float(alpha_annual), float(t_stat), float(p_value), p_value < (1 - confidence)

    @staticmethod
    def sharpe_ratio_test(
        returns: np.ndarray,
        null_sharpe: float = 0.0,
        max_lags: int = None,
        periods_per_year: int = 252,
    ) -> tuple[float, float, float, float]:
        """Lo (2002) autocorrelation-corrected Sharpe ratio test.

        Standard Sharpe SE = 1/sqrt(T), but with autocorrelation the
        effective sample size is smaller. Lo's correction inflates the
        SE by a factor eta = sqrt(1 + 2*sum_{k=1}^{q} (1-k/(q+1)) * rho_k).

        Args:
            returns: 1D array of portfolio returns.
            null_sharpe: Null hypothesis Sharpe (annualized).
            max_lags: Lag truncation for autocorrelation (default: floor(T^{1/3})).

        Returns:
            (sharpe_annualized, corrected_se, t_stat, p_value)
        """
        returns = np.asarray(returns)
        T = len(returns)
        if T < 10:
            return 0.0, 1.0, 0.0, 1.0

        if max_lags is None:
            max_lags = max(1, int(T ** (1 / 3)))

        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        if sigma < 1e-12:
            return 0.0, 1.0, 0.0, 1.0

        sharpe_daily = mu / sigma
        sharpe_annual = sharpe_daily * np.sqrt(periods_per_year)

        # Autocorrelation correction factor
        demeaned = returns - mu
        gamma0 = np.mean(demeaned ** 2)
        eta_sq = 1.0
        for k in range(1, max_lags + 1):
            rho_k = np.mean(demeaned[k:] * demeaned[:-k]) / gamma0
            bartlett_weight = 1 - k / (max_lags + 1)
            eta_sq += 2 * bartlett_weight * rho_k

        eta_sq = max(eta_sq, 0.01)  # Floor to avoid negative

        # Corrected SE of annualized Sharpe
        naive_se = 1.0 / np.sqrt(T) * np.sqrt(periods_per_year)
        corrected_se = naive_se * np.sqrt(eta_sq)

        t_stat = (sharpe_annual - null_sharpe) / corrected_se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return float(sharpe_annual), float(corrected_se), float(t_stat), float(p_value)

    @staticmethod
    def information_coefficient(
        signal: np.ndarray,
        forward_returns: np.ndarray,
    ) -> tuple[float, float, float]:
        """Spearman rank information coefficient.

        Measures the rank correlation between a cross-sectional signal
        and subsequent forward returns.

        Args:
            signal: N-vector of signal values (e.g. expected return tilts).
            forward_returns: N-vector of realized forward returns.

        Returns:
            (ic, t_stat, p_value)
        """
        signal = np.asarray(signal)
        forward_returns = np.asarray(forward_returns)

        if len(signal) < 3:
            return 0.0, 0.0, 1.0

        rho, p_value = stats.spearmanr(signal, forward_returns)
        N = len(signal)
        t_stat = rho * np.sqrt((N - 2) / (1 - rho ** 2 + 1e-12))

        return float(rho), float(t_stat), float(p_value)
