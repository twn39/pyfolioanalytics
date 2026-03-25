from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def ac_ranking(
    R: pd.DataFrame, order: list[int | str], max_value: float | None = None
) -> np.ndarray:
    """
    Asset Ranking based on Almgren and Chriss (2005).
    Converts a relative ranking of assets into an expected return vector.
    """
    n_assets = R.shape[1]
    asset_names = list(R.columns)

    # Convert string order to indices
    idx_order = []
    for item in order:
        if isinstance(item, str):
            idx_order.append(asset_names.index(item))
        else:
            idx_order.append(item)

    if len(idx_order) != n_assets:
        raise ValueError("Order length must match number of assets")

    if max_value is None:
        max_value = float(np.median(R.mean().values))

    # 1. Compute analytical centroid
    c_hat = centroid_analytical(n_assets)

    # 2. Scale centroid (R implementation maps to [-0.05, 0.05] then shifts)
    c_hat_scaled = scale_range(c_hat, -0.05, 0.05)

    # 3. Assign to assets based on order
    # order[0] is the index of the asset with LOWEST expected return
    # c_hat from centroid() is [highest, ..., lowest]
    out = np.zeros(n_assets)
    out[np.array(idx_order)[::-1]] = c_hat_scaled
    return out


def centroid_analytical(n: int) -> np.ndarray:
    """
    Analytical solution to the centroid for single complete sort.
    """
    A = 0.4424
    B = 0.1185
    beta = 0.21
    alpha = A - B * n ** (-beta)
    j = np.arange(1, n + 1)
    # R: qnorm((n + 1 - j - alpha) / (n - 2 * alpha + 1))
    c_hat = norm.ppf((n + 1 - j - alpha) / (n - 2 * alpha + 1))
    return c_hat


def scale_range(x: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    old_min, old_max = np.min(x), np.max(x)
    old_range = old_max - old_min
    new_range = new_max - new_min
    if old_range == 0:
        return np.full_like(x, (new_min + new_max) / 2)
    return ((x - old_min) * new_range) / old_range + new_min


def centroid_complete_mc(order: list[int], simulations: int = 1000) -> np.ndarray:
    n = len(order)
    sims = np.random.randn(simulations, n)
    sims.sort(axis=1)
    sims = sims[:, ::-1]  # Decreasing: [largest, ..., smallest]

    means = np.mean(sims, axis=0)
    out = np.zeros(n)
    out[np.array(order)[::-1]] = means
    return out


def statistical_factor_model(R: pd.DataFrame, k: int = 3) -> dict[str, Any]:
    """
    Extract statistical factors using PCA.
    Returns:
    - factors: Factor returns (T x k)
    - loadings: Factor loadings (N x k)
    - alpha: Intercepts (N x 1)
    - residuals: Residual returns (T x N)
    """
    T, N = R.shape
    # Center returns
    mu = R.mean()
    R_centered = R - mu

    # PCA via SVD
    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)

    # Factors (principal components)
    # R = U S V'
    # Factors = U S
    factors_mat = U[:, :k] @ np.diag(S[:k])
    factors = pd.DataFrame(
        factors_mat, index=R.index, columns=[f"Factor.{i + 1}" for i in range(k)]
    )

    # Loadings (eigenvectors)
    # Vt is (N x N), top k rows are loadings
    loadings = Vt[:k, :].T

    # Alphas and Residuals
    # R = alpha + Loadings * Factors + Residuals
    # For statistical factors, alpha is often mean return
    alpha = mu.values.reshape(-1, 1)

    # Reconstruction
    R_hat = factors_mat @ loadings.T
    residuals = R_centered.values - R_hat

    return {
        "factors": factors,
        "loadings": pd.DataFrame(loadings, index=R.columns, columns=factors.columns),
        "alpha": pd.Series(alpha.flatten(), index=R.columns),
        "residuals": pd.DataFrame(residuals, index=R.index, columns=R.columns),
    }


def factor_model_covariance(model_results: dict[str, Any]) -> np.ndarray:
    """
    Calculate the factor model covariance matrix.
    Sigma = Beta * Sigma_f * Beta' + Diag(Sigma_e)
    """
    B = model_results["loadings"].values
    factors = model_results["factors"].values
    residuals = model_results["residuals"].values

    # Covariance of factors
    Sigma_f = np.cov(factors, rowvar=False)

    # Diagonal matrix of residual variances
    Sigma_e = np.diag(np.var(residuals, axis=0, ddof=1))

    return B @ Sigma_f @ B.T + Sigma_e
