import numpy as np
from pyfolioanalytics.rmt import denoise_covariance

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
    
    # Denoise
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
