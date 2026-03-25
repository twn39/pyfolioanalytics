import pytest
import numpy as np
import pandas as pd
import riskfolio as rp
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.moments import set_portfolio_moments


def test_shrinkage_identity_cv(stocks_data):
    # Select first 5 assets for performance
    R = stocks_data.iloc[:, :5]

    # 1. PyFolioAnalytics calculation
    port = Portfolio(assets=list(R.columns))
    py_moments = set_portfolio_moments(
        R, port, method="shrinkage", shrinkage_target="identity"
    )
    py_sigma = py_moments["sigma"]

    # 2. Riskfolio-Lib calculation
    # Riskfolio-Lib's 'ledoit' actually shrinks towards the identity matrix (Ledoit and Wolf 2004)
    rp_port = rp.Portfolio(returns=R)
    rp_port.assets_stats(method_mu="hist", method_cov="ledoit")
    rp_sigma = getattr(rp_port.cov, 'values', rp_port.cov)

    # Note: Riskfolio-Lib scales covariance differently internally (annualized or empirical adjustment)
    # We test for matrix correlation parity (the structure of the shrinkage should be identical)

    # Convert to correlation matrices to compare the structural shrinkage
    def cov_to_corr(cov):
        d = np.diag(cov)
        std = np.sqrt(d)
        return cov / np.outer(std, std)

    py_corr = cov_to_corr(py_sigma)
    rp_corr = cov_to_corr(rp_sigma)

    # Assert structural parity
    np.testing.assert_allclose(py_corr, rp_corr, atol=0.01)
