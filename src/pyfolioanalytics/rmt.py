import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Any

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

def _fit_err(sigma2: float, eigenvalues: np.ndarray, q: float, n_bins: int = 1000) -> float:
    """
    SSE between empirical and theoretical MP PDF.
    """
    # Empirical PDF
    hist, bin_edges = np.histogram(eigenvalues, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Theoretical PDF at bin centers
    pdf_theo = marchenko_pastur_pdf(bin_centers, q, sigma2)
    
    return float(np.sum((hist - pdf_theo)**2))

def find_max_eigenvalue(eigenvalues: np.ndarray, q: float) -> Tuple[float, float]:
    """
    Find the maximum eigenvalue that belongs to the noise component.
    Returns (e_max, fitted_sigma2).
    """
    # Initial guess for sigma2: usually 1.0 for correlation matrix
    res = minimize(_fit_err, x0=[1.0], args=(eigenvalues, q), bounds=[(1e-5, 10.0)])
    sigma2 = res.x[0]
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
    
    e_max, _ = find_max_eigenvalue(vals, q)
    
    # Identify noise components
    n_noise = np.sum(vals <= e_max)
    if n_noise == 0:
        return sigma
        
    vals_denoised = np.copy(vals)
    if method == "spectral":
        # Using a small epsilon to maintain positive definiteness and avoid numerical issues
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
