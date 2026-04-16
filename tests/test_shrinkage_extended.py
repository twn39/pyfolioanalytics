import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.moments import set_portfolio_moments, capm_returns, ema_returns, shrunk_covariance
from pypfopt import expected_returns, risk_models


def test_shrinkage_identity(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))

    moments = set_portfolio_moments(
        R, port, method="shrinkage", shrinkage_target="identity"
    )
    assert "sigma" in moments
    assert moments["sigma"].shape == (5, 5)
    # Check PSD
    eigvals = np.linalg.eigvalsh(moments["sigma"])
    assert np.all(eigvals >= -1e-8)


def test_shrinkage_oas(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))

    moments = set_portfolio_moments(R, port, method="shrinkage", shrinkage_target="oas")
    assert "sigma" in moments
    assert moments["sigma"].shape == (5, 5)
    # Check PSD
    eigvals = np.linalg.eigvalsh(moments["sigma"])
    assert np.all(eigvals >= -1e-8)


def test_shrinkage_constant_correlation(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))

    moments = set_portfolio_moments(
        R, port, method="shrinkage", shrinkage_target="constant_correlation"
    )
    assert "sigma" in moments
    assert moments["sigma"].shape == (5, 5)

    # Check PSD
    eigvals = np.linalg.eigvalsh(moments["sigma"])
    assert np.all(eigvals >= -1e-8)

    # It shouldn't be exactly the sample covariance
    sample_cov = R.cov().values
    assert not np.allclose(moments["sigma"], sample_cov)


def test_capm_parity_with_pypfopt(stocks_data):
    R = stocks_data.iloc[:, :5]
    
    # 1. Compute with pyfolioanalytics (native math)
    native_capm = capm_returns(R, risk_free_rate=0.02)
    
    # 2. Compute with PyPortfolioOpt
    # PyPfOpt calculates returns from prices, but we can pass returns_data=True
    # However, PyPfOpt annualizes by default (frequency=252). Our capm_returns works on the raw scale provided.
    # To match exactly, we must account for PyPfOpt's implicit annualized mean return logic.
    pypfopt_capm = expected_returns.capm_return(R, returns_data=True, risk_free_rate=0.02, frequency=1) # frequency=1 to match raw daily scale
    
    # 3. Assert exact parity
    np.testing.assert_allclose(native_capm.flatten(), pypfopt_capm.values, rtol=1e-5)


def test_ema_parity_with_pypfopt(stocks_data):
    R = stocks_data.iloc[:, :5]
    
    # 1. Native EMA
    native_ema = ema_returns(R, span=180)
    
    # 2. PyPortfolioOpt EMA
    pypfopt_ema = expected_returns.ema_historical_return(R, returns_data=True, span=180, frequency=1)
    
    # 3. Assert exact parity
    np.testing.assert_allclose(native_ema.flatten(), pypfopt_ema.values, rtol=1e-5)


def test_ledoit_wolf_constant_correlation_parity(stocks_data):
    R = stocks_data.iloc[:, :5]
    
    # 1. Compute with pyfolioanalytics (native math)
    native_cov = shrunk_covariance(R, method="ledoit_wolf", shrinkage_target="constant_correlation")
    
    # 2. Compute with PyPortfolioOpt
    cs = risk_models.CovarianceShrinkage(R, returns_data=True, frequency=1)
    pypfopt_cov = cs.ledoit_wolf(shrinkage_target="constant_correlation")
    
    # 3. Assert exact parity
    np.testing.assert_allclose(native_cov, pypfopt_cov.values, rtol=1e-5)

def test_decoupled_moments_interface(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))

    moments = set_portfolio_moments(
        R, port, sigma_method="ledoit_wolf", shrinkage_target="constant_correlation", mu_method="capm", risk_free_rate=0.01
    )
    
    assert "sigma" in moments
    assert "mu" in moments
    
    # Sigma should be shrunk
    assert not np.allclose(moments["sigma"], R.cov().values)
    
    # Mu should not be standard mean
    assert not np.allclose(moments["mu"].flatten(), R.mean().values)
