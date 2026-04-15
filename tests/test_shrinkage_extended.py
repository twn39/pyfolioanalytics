import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.moments import set_portfolio_moments


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
