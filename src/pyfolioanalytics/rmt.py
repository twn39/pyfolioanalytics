import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from scipy.optimize import minimize_scalar

def marchenko_pastur_pdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """
    Theoretical Marchenko-Pastur PDF.
    q = T / N (observations / assets)
    """
    e_min = sigma2 * (1 - np.sqrt(1.0 / q))**2
    e_max = sigma2 * (1 + np.sqrt(1.0 / q))**2
    
    # Avoid division by zero
    x_safe = np.maximum(x, 1e-10)
    pdf = q / (2 * np.pi * sigma2 * x_safe) * np.sqrt(np.maximum(0, (e_max - x) * (x - e_min)))
    return pdf

def find_max_eigenvalue(eigenvalues: np.ndarray, q: float) -> Tuple[float, float]:
    """
    Find the maximum eigenvalue that belongs to the noise component.
    Returns (e_max, fitted_sigma2).
    Histogram is computed once and captured in closure to avoid redundant work
    inside the scalar minimiser (which may call the objective hundreds of times).
    """
    # Pre-compute empirical PDF once — eigenvalues don't change during fitting
    hist, bin_edges = np.histogram(eigenvalues, bins=1000, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _fit_err(sigma2: float) -> float:
        if sigma2 <= 0:
            return 1e10
        pdf_theo = marchenko_pastur_pdf(bin_centers, q, sigma2)
        return float(np.sum((hist - pdf_theo) ** 2))

    res = minimize_scalar(_fit_err, bounds=(1e-5, 1.0), args=(), method='bounded')
    sigma2 = res.x
    e_max = sigma2 * (1 + np.sqrt(1.0 / q))**2
    return e_max, sigma2

def denoise_covariance(
    sigma: np.ndarray, 
    q: float, 
    method: str = "fixed", 
    alpha: float = 0.0,
    is_correlation: bool = False
) -> np.ndarray:
    """
    Denoise a covariance or correlation matrix using Marchenko-Pastur theorem.
    """
    if not is_correlation:
        std = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(std, std)
    else:
        corr = sigma
        
    vals, vecs = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    e_max, sigma2 = find_max_eigenvalue(vals, q)
    
    # Identify noise components
    n_noise = int(np.sum(vals <= e_max))
    if n_noise == 0:
        # Fallback
        e_max = float((1 + np.sqrt(1.0 / q))**2)
        n_noise = int(np.sum(vals <= e_max))
        if n_noise == 0:
            return sigma
        
    vals_denoised = np.copy(vals)
    if method == "spectral":
        vals_denoised[-n_noise:] = 1e-10
    elif method == "fixed":
        avg_noise = np.mean(vals[-n_noise:])
        vals_denoised[-n_noise:] = avg_noise
    elif method == "shrunk":
        avg_noise = np.mean(vals[-n_noise:])
        vals_denoised[-n_noise:] = alpha * vals[-n_noise:] + (1 - alpha) * avg_noise
        
    # Reconstruction
    corr_denoised = vecs @ np.diag(vals_denoised) @ vecs.T
    
    # Rescale to ensure diagonal is 1.0
    d = np.diag(corr_denoised)
    corr_denoised = corr_denoised / np.sqrt(np.outer(d, d))
    
    if not is_correlation:
        return corr_denoised * np.outer(std, std)
    else:
        return corr_denoised

def gerber_statistic(
    R: pd.DataFrame, 
    threshold: float = 0.5, 
    method: int = 1, 
    standardize: bool = False
) -> pd.DataFrame:
    """
    Compute the Gerber Statistic (robust co-movement measure).
    """
    T, N = R.shape
    X = R.values
    
    if standardize:
        mu = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
        X = (X - mu) / std
        U = (X >= threshold)
        D = (X <= -threshold)
    else:
        std = np.std(X, axis=0, ddof=1)
        U = (X >= threshold * std)
        D = (X <= -threshold * std)
        
    UmD = U.astype(float) - D.astype(float)
    
    if method == 0:
        UpD = U.astype(float) + D.astype(float)
        H = UmD.T @ UmD
        denom = UpD.T @ UpD
        rho = np.divide(H, denom, out=np.zeros_like(H), where=denom!=0)
    elif method == 1:
        N_mat = (~U & ~D).astype(float)
        H = UmD.T @ UmD
        denom = T - (N_mat.T @ N_mat)
        rho = np.divide(H, denom, out=np.zeros_like(H), where=denom!=0)
    elif method == 2:
        H = UmD.T @ UmD
        h = np.sqrt(np.diag(H))
        denom = np.outer(h, h)
        rho = np.divide(H, denom, out=np.zeros_like(H), where=denom!=0)
    else:
        raise ValueError("Method must be 0, 1, or 2")
        
    std_orig = np.std(R.values, axis=0, ddof=1)
    sigma = rho * np.outer(std_orig, std_orig)
    return pd.DataFrame(sigma, index=R.columns, columns=R.columns)

def detone_covariance(sigma: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Remove the market component (first principal components) from covariance.
    Eigenvalues are set to a small positive value rather than 0 to keep the
    matrix invertible (avoids singular covariance passed to MVO solvers).
    """
    vals, vecs = np.linalg.eigh(sigma)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    vals_detoned = np.copy(vals)
    # Use a tiny positive floor instead of 0 so the matrix stays positive-definite
    vals_detoned[:n_components] = 1e-8
    
    sigma_detoned = vecs @ np.diag(vals_detoned) @ vecs.T
    return sigma_detoned

def bootstrap_uncertainty_set(
    R: pd.DataFrame, 
    n_sim: int = 1000, 
    q: float = 0.05,
    random_state: int | None = None,
) -> Dict[str, Any]:
    """
    Construct uncertainty sets for mu and sigma using bootstrap.
    """
    T, N = R.shape
    rng = np.random.default_rng(random_state)
    mu_sims = []
    sigma_sims = []
    
    for _ in range(n_sim):
        idx = rng.choice(T, size=T, replace=True)
        R_boot = R.iloc[idx]
        mu_sims.append(R_boot.mean().values)
        sigma_sims.append(R_boot.cov().values)
        
    mu_sims = np.array(mu_sims)
    sigma_sims = np.array(sigma_sims)
    
    # 1. Box uncertainty for mu
    mu_lb = np.quantile(mu_sims, q/2, axis=0)
    mu_ub = np.quantile(mu_sims, 1-q/2, axis=0)
    
    # 2. Ellipsoidal uncertainty for mu
    mu_mean = np.mean(mu_sims, axis=0)
    mu_cov = np.cov(mu_sims, rowvar=False)
    
    # 3. Ellipsoidal uncertainty for sigma
    sigma_vecs = sigma_sims.reshape(n_sim, -1)
    sigma_mean = np.mean(sigma_sims, axis=0)
    sigma_cov = np.cov(sigma_vecs, rowvar=False)
    
    return {
        "mu_lb": mu_lb,
        "mu_ub": mu_ub,
        "mu_mean": mu_mean,
        "mu_cov": mu_cov,
        "sigma_mean": sigma_mean,
        "sigma_cov": sigma_cov
    }
