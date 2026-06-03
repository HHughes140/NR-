from typing import Optional

import numpy as np
import pandas as pd

from NR.linalg.covariance import CovarianceEstimator


class MatrixOps:
    """Core linear algebra operations for portfolio analysis."""

    @staticmethod
    def eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues (descending) and eigenvectors of a symmetric matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """Ratio of largest to smallest eigenvalue."""
        eigenvalues = np.linalg.eigvalsh(matrix)
        return abs(eigenvalues.max() / eigenvalues.min())

    @staticmethod
    def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        """Matrix square root via eigendecomposition."""
        eigenvalues, V = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        return V @ np.diag(np.sqrt(eigenvalues)) @ V.T

    @staticmethod
    def matrix_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        """Matrix inverse square root via eigendecomposition."""
        eigenvalues, V = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        return V @ np.diag(1.0 / np.sqrt(eigenvalues)) @ V.T


class PCA:
    """Principal Component Analysis for return factor extraction."""

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.loadings_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.asset_names_: Optional[list[str]] = None

    def fit(self, returns: pd.DataFrame) -> "PCA":
        """Fit PCA on returns matrix (T x N)."""
        self.asset_names_ = list(returns.columns)
        X = np.array(returns)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = CovarianceEstimator.sample(returns)
        eigenvalues, eigenvectors = MatrixOps.eigendecomposition(cov)

        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues / total_var

        if self.n_components is None:
            self.n_components = len(eigenvalues)

        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.eigenvectors_ = eigenvectors[:, :self.n_components]
        self.loadings_ = self.eigenvectors_ * np.sqrt(self.eigenvalues_)

        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Project returns onto principal components."""
        X = np.array(returns) - self.mean_
        scores = X @ self.eigenvectors_
        columns = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(scores, index=returns.index, columns=columns)

    def inverse_transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct approximate returns from factor scores."""
        reconstructed = np.array(factors) @ self.eigenvectors_.T + self.mean_
        return pd.DataFrame(reconstructed, index=factors.index,
                            columns=self.asset_names_)

    @property
    def variance_explained(self) -> np.ndarray:
        """Cumulative variance explained by each component."""
        return np.cumsum(self.explained_variance_ratio_[:self.n_components])

    def select_components(self, threshold: float = 0.90) -> int:
        """Number of components needed to explain threshold fraction of variance."""
        cumulative = np.cumsum(self.explained_variance_ratio_)
        return int(np.searchsorted(cumulative, threshold) + 1)
