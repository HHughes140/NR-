import numpy as np
import pandas as pd


class CovarianceEstimator:
    """Multiple covariance estimation methods for portfolio analysis."""

    @staticmethod
    def sample(returns: pd.DataFrame) -> np.ndarray:
        """Standard sample covariance matrix."""
        return np.array(returns.cov())

    @staticmethod
    def ledoit_wolf(returns: pd.DataFrame,
                    target: str = "constant_correlation") -> tuple[np.ndarray, float]:
        """Ledoit-Wolf shrinkage estimator.

        Implements the analytical formula from Ledoit & Wolf (2004).

        Args:
            target: "constant_correlation" or "identity".

        Returns:
            (shrunk_covariance, shrinkage_intensity)
        """
        X = np.array(returns)
        T, N = X.shape
        X_demean = X - X.mean(axis=0)

        # Sample covariance (1/T normalization for the formula)
        sample = X_demean.T @ X_demean / T

        # Structured target F
        if target == "identity":
            mu = np.trace(sample) / N
            F = mu * np.eye(N)
        else:  # constant_correlation
            var = np.diag(sample)
            std = np.sqrt(var)
            # Avoid division by zero
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = sample / np.outer(std, std)
            np.fill_diagonal(corr, 1.0)
            corr = np.nan_to_num(corr, nan=0.0)
            avg_corr = (corr.sum() - N) / (N * (N - 1))
            F = avg_corr * np.outer(std, std)
            np.fill_diagonal(F, var)

        # Frobenius norm squared of (S - F)
        delta = np.sum((sample - F) ** 2)

        if delta == 0:
            return sample * T / (T - 1), 0.0

        # Estimate pi: asymptotic variance of sqrt(T) * vec(S)
        # pi = (1/T^2) * sum_t ||x_t x_t^T - S||_F^2
        # Vectorized: ||outer_t - S||^2 = ||x_t||^4 - 2*x_t^T S x_t + ||S||^2
        norms_sq = np.sum(X_demean ** 2, axis=1)
        term1 = np.sum(norms_sq ** 2)
        XS = X_demean @ sample
        term2 = 2.0 * np.sum(XS * X_demean)
        term3 = T * np.sum(sample ** 2)
        pi = (term1 - term2 + term3) / (T * T)

        # Shrinkage intensity
        alpha = min(max(pi / delta, 0.0), 1.0)

        shrunk = alpha * F + (1 - alpha) * sample
        # Scale back to 1/(T-1) normalization for consistency
        shrunk = shrunk * T / (T - 1)
        return shrunk, alpha

    @staticmethod
    def ewma(returns: pd.DataFrame, halflife: int = 60) -> np.ndarray:
        """Exponentially weighted moving average covariance."""
        lam = 1 - np.log(2) / halflife
        T, N = returns.shape
        X = np.array(returns - returns.mean())

        # Compute weights
        weights = np.array([(1 - lam) * lam ** i for i in range(T - 1, -1, -1)])
        weights /= weights.sum()

        # Weighted covariance
        weighted_X = X * np.sqrt(weights[:, np.newaxis])
        return weighted_X.T @ weighted_X

    @staticmethod
    def to_correlation(cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        d = np.sqrt(np.diag(cov_matrix))
        d_inv = 1 / d
        corr = cov_matrix * np.outer(d_inv, d_inv)
        np.fill_diagonal(corr, 1.0)
        return corr

    @staticmethod
    def is_positive_definite(matrix: np.ndarray) -> bool:
        """Check via Cholesky decomposition."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
        """Project to nearest PD matrix (Higham 2002)."""
        B = (matrix + matrix.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if CovarianceEstimator.is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(matrix))
        I = np.eye(matrix.shape[0])
        k = 1
        while not CovarianceEstimator.is_positive_definite(A3):
            min_eig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-min_eig * k ** 2 + spacing)
            k += 1
        return A3

    @staticmethod
    def regime_adjusted(base_cov: np.ndarray, regime: str,
                        multipliers: dict = None) -> np.ndarray:
        """Scale covariance by regime-specific multiplier."""
        defaults = {"expansion": 0.8, "contraction": 1.3,
                     "crisis": 2.0, "recovery": 1.0}
        multipliers = multipliers or defaults
        m = multipliers.get(regime, 1.0)
        return base_cov * m

    @staticmethod
    def sentiment_adjusted(base_cov: np.ndarray, sentiment_score: float,
                           scale: float = 0.3) -> np.ndarray:
        """Inflate covariance at extreme sentiment levels."""
        multiplier = 1.0 + scale * abs(sentiment_score)
        return base_cov * multiplier

    @staticmethod
    def pofm(returns: np.ndarray,
             ff_factor_data: np.ndarray,
             n_latent: int = 0,
             threshold_method: str = "soft",
             threshold_scale: float = 3.0) -> np.ndarray:
        """POFM three-component covariance estimator.

        Partially Observable Factor Model covariance via three-component
        decomposition (Chen, Lu & Xie 2025):

            Σ = β' Σ_F β  +  θ' Λ θ  +  r*

        where:
            β' Σ_F β  = observable factor covariance (FF5/FF5+MOM)
            θ' Λ θ    = latent factor covariance (PCA on residuals)
            r*         = thresholded idiosyncratic covariance

        Step 1: OLS regression → β, residuals.
        Step 2: Ledoit-Wolf shrinkage on residual covariance.
        Step 3: PCA to separate latent low-rank from idiosyncratic.
        Step 4: Adaptive threshold on idiosyncratic only.
        Step 5: Reconstruct three-component covariance.

        Args:
            returns: T x N array of asset returns.
            ff_factor_data: T x K_ff array of FF factor returns.
            n_latent: Number of latent factors for residual PCA.
                0 = auto-select via variance threshold (85%).
            threshold_method: "soft" or "hard" on idiosyncratic cov.
            threshold_scale: Multiplier for threshold chi = scale * varpi.
                Default 3.0 per Chen et al. (2025).

        Returns:
            N x N positive semi-definite covariance matrix.
        """
        T, N = returns.shape
        K_ff = ff_factor_data.shape[1]

        # Step 1: OLS regression with intercept
        F = np.column_stack([np.ones(T), ff_factor_data])
        coeffs = np.linalg.lstsq(F, returns, rcond=None)[0]  # (K_ff+1, N)
        betas = coeffs[1:]  # (K_ff, N) — exclude intercept
        residuals = returns - F @ coeffs  # (T, N)

        # Observable factor covariance
        Sigma_F = np.cov(ff_factor_data, rowvar=False)  # (K_ff, K_ff)
        B = betas.T  # (N, K_ff)
        cov_obs = B @ Sigma_F @ B.T  # (N, N)

        # Step 2: Ledoit-Wolf shrinkage on residual covariance
        resid_df = pd.DataFrame(residuals)
        cov_res, _ = CovarianceEstimator.ledoit_wolf(resid_df)

        # Step 3: PCA to separate latent low-rank from idiosyncratic
        cov_res_sym = (cov_res + cov_res.T) / 2
        evals, evecs = np.linalg.eigh(cov_res_sym)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        if n_latent <= 0:
            # Auto-select K via 85% variance threshold
            total_var = np.sum(np.maximum(evals, 0))
            if total_var > 0:
                cumulative = np.cumsum(np.maximum(evals, 0) / total_var)
                n_latent = int(np.searchsorted(cumulative, 0.85) + 1)
            else:
                n_latent = 1
        n_latent = min(n_latent, N)

        # Latent factor covariance (low-rank)
        V_k = evecs[:, :n_latent]
        Lambda_k = np.diag(np.maximum(evals[:n_latent], 0))
        cov_latent = V_k @ Lambda_k @ V_k.T  # (N, N)

        # Step 4: Idiosyncratic = residual cov - latent (threshold only this)
        cov_idio = cov_res - cov_latent

        # Adaptive threshold: chi = threshold_scale * varpi
        # varpi = sqrt(log(p) / T) per Chen et al. (2025)
        varpi = np.sqrt(np.log(max(N, 2)) / max(T, 1))
        chi = threshold_scale * varpi

        if threshold_method == "soft":
            sign = np.sign(cov_idio)
            cov_idio_thresh = sign * np.maximum(np.abs(cov_idio) - chi, 0)
        else:
            cov_idio_thresh = cov_idio * (np.abs(cov_idio) > chi)

        # Preserve diagonal (never threshold variances)
        np.fill_diagonal(cov_idio_thresh, np.diag(cov_idio))

        # Step 5: Three-component reconstruction
        Sigma_full = cov_obs + cov_latent + cov_idio_thresh

        # Ensure symmetry and PSD
        Sigma_full = (Sigma_full + Sigma_full.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_full))
        if min_eig < 0:
            Sigma_full += (-min_eig + 1e-8) * np.eye(N)

        return Sigma_full

    @staticmethod
    def accuracy_weighted(returns: pd.DataFrame,
                          observation_weights: np.ndarray) -> np.ndarray:
        """Weighted sample covariance using per-observation weights."""
        X = np.array(returns)
        w = observation_weights / observation_weights.sum()
        X_mean = (w[:, np.newaxis] * X).sum(axis=0)
        X_centered = X - X_mean
        weighted = X_centered * np.sqrt(w[:, np.newaxis])
        return weighted.T @ weighted
