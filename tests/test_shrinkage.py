import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.moments import set_portfolio_moments
from sklearn.covariance import ledoit_wolf


def test_ledoit_wolf_shrinkage():
    # Generate synthetic data
    np.random.seed(42)
    n_assets = 5
    n_obs = 100
    asset_names = [f"A{i}" for i in range(n_assets)]
    R = pd.DataFrame(np.random.randn(n_obs, n_assets), columns=asset_names)

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_objective("StdDev")

    # Test method="shrinkage"
    moments = set_portfolio_moments(R, portfolio, method="shrinkage")

    # Calculate truth using sklearn directly
    # Note: sklearn's ledoit_wolf function returns (shrunk_cov, shrinkage_constant)
    # The LedoitWolf() class used in moments.py is a wrapper.
    expected_sigma, _ = ledoit_wolf(R.values)

    np.testing.assert_allclose(moments["sigma"], expected_sigma, rtol=1e-7)

    # Check sample cov is different
    sample_cov = R.cov().values
    assert not np.allclose(moments["sigma"], sample_cov, rtol=1e-3)
