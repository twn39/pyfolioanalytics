import numpy as np
import pandas as pd
from pyfolioanalytics.rmt import (
    denoise_covariance, 
    gerber_statistic, 
    detone_covariance, 
    bootstrap_uncertainty_set
)

def test_marchenko_pastur_denoising():
    # Setup random data with high noise to signal ratio
    np.random.seed(42)
    # T > N for standard MP, but let's use a case where noise is visible
    T, N = 500, 100
    # True signal: only 2 large components
    signal_core = np.random.randn(T, 2) @ np.random.randn(2, N) * 0.5
    # Lots of pure noise
    noise = np.random.randn(T, N)
    X = signal_core + noise
    
    sigma = np.cov(X, rowvar=False)
    q = T / N
    
    # Denoise fixed
    sigma_denoised = denoise_covariance(sigma, q, method="fixed")
    
    assert sigma_denoised.shape == (N, N)
    # Check if eigenvalues changed
    vals_raw = np.linalg.eigvalsh(sigma)
    vals_denoised = np.linalg.eigvalsh(sigma_denoised)
    
    # In MP denoising, many small eigenvalues are replaced by their average
    # This should change the spectrum
    assert not np.allclose(vals_raw, vals_denoised)
    
    # Standard deviation should be preserved on diagonal
    std_raw = np.sqrt(np.diag(sigma))
    std_denoised = np.sqrt(np.diag(sigma_denoised))
    np.testing.assert_allclose(std_raw, std_denoised, atol=1e-10)

def test_denoise_method_spectral():
    np.random.seed(42)
    # Use small T/N to ensure lots of noise eigenvalues
    T, N = 100, 50
    X = np.random.randn(T, N)
    sigma = np.cov(X, rowvar=False)
    q = T / N
    
    sigma_denoised = denoise_covariance(sigma, q, method="spectral")
    vals = np.linalg.eigvalsh(sigma_denoised)
    # Some should be very small (epsilon = 1e-10)
    assert np.any(np.abs(vals) < 1e-8)

def test_denoise_method_shrunk_and_correlation():
    np.random.seed(42)
    T, N = 100, 20
    X = np.random.randn(T, N)
    sigma = np.cov(X, rowvar=False)
    q = T / N
    corr = np.corrcoef(X, rowvar=False)

    sigma_denoised_shrunk = denoise_covariance(sigma, q, method="shrunk", alpha=0.5)
    assert sigma_denoised_shrunk.shape == (N, N)

    # Test is_correlation=True
    corr_denoised = denoise_covariance(corr, q, is_correlation=True)
    np.testing.assert_allclose(np.diag(corr_denoised), 1.0, atol=1e-10)

def test_denoise_fallback():
    np.random.seed(42)
    T, N = 100, 2
    # Create very clean signal (no noise) to trigger n_noise == 0 fallback
    X = np.random.randn(T, 1) @ np.ones((1, N)) * 10
    sigma = np.cov(X, rowvar=False)
    q = T / N
    
    # If no eigenvalues are classified as noise, it should fallback or return original
    sigma_denoised = denoise_covariance(sigma, q, method="fixed")
    assert sigma_denoised.shape == (N, N)

def test_gerber_statistic():
    np.random.seed(42)
    T, N = 50, 3
    R = pd.DataFrame(np.random.randn(T, N), columns=['A', 'B', 'C'])
    
    # Method 0
    cov_0 = gerber_statistic(R, method=0)
    assert cov_0.shape == (3, 3)
    
    # Method 1
    cov_1 = gerber_statistic(R, method=1)
    assert cov_1.shape == (3, 3)
    
    # Method 2, with standardize
    cov_2 = gerber_statistic(R, method=2, standardize=True)
    assert cov_2.shape == (3, 3)

    import pytest
    with pytest.raises(ValueError):
        gerber_statistic(R, method=3)

def test_detone_covariance():
    np.random.seed(42)
    N = 4
    sigma = np.diag([4.0, 3.0, 2.0, 1.0])
    
    detoned = detone_covariance(sigma, n_components=1)
    vals = np.linalg.eigvalsh(detoned)
    # The largest eigenvalue (which was 4.0) should be dropped to a very small positive value (~1e-8)
    assert np.min(vals) > 0
    assert np.max(vals) <= 3.01  # Should be close to 3.0 now

def test_bootstrap_uncertainty_set():
    np.random.seed(42)
    T, N = 50, 2
    R = pd.DataFrame(np.random.randn(T, N), columns=['A', 'B'])
    
    res = bootstrap_uncertainty_set(R, n_sim=10, random_state=42)
    assert "mu_lb" in res
    assert "mu_ub" in res
    assert "mu_cov" in res
    assert "sigma_cov" in res
    
    assert res["mu_lb"].shape == (N,)
    assert res["mu_cov"].shape == (N, N)
    assert res["sigma_cov"].shape == (N*N, N*N)
