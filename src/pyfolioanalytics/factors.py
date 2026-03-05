import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def statistical_factor_model(
    R: pd.DataFrame,
    k: int = 3
) -> Dict[str, Any]:
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
    factors = pd.DataFrame(factors_mat, index=R.index, columns=[f"Factor.{i+1}" for i in range(k)])
    
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
        "residuals": pd.DataFrame(residuals, index=R.index, columns=R.columns)
    }

def factor_model_covariance(
    model_results: Dict[str, Any]
) -> np.ndarray:
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
