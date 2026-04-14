import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

import json
import os


@pytest.fixture
def riskfolio_cv():
    path = "data/riskfolio_cv.json"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found. Run scripts/generate_riskfolio_cv.py first.")
    with open(path, "r") as f:
        return json.load(f)


def test_kelly_optimization_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.iloc[:100].columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    res = optimize_portfolio(stocks_data.iloc[:100], portfolio, optimize_method="Kelly")
    w_ours = res["weights"].values

    # Ground truth from JSON
    w_expected = pd.Series(riskfolio_cv["kelly"]["weights"]).values

    np.testing.assert_allclose(w_ours, w_expected, atol=1e-5)


def test_max_diversification_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.iloc[:100].columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    res = optimize_portfolio(stocks_data.iloc[:100], portfolio, optimize_method="MDIV")
    w_ours = res["weights"].values

    # Ground truth from JSON
    w_expected = pd.Series(riskfolio_cv["mdiv"]["weights"]).values

    np.testing.assert_allclose(w_ours, w_expected, atol=1e-5)


def test_mdiv_ratio_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.iloc[:100].columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    res = optimize_portfolio(stocks_data.iloc[:100], portfolio, optimize_method="MDIV")

    # Check if Diversification Ratio is calculated correctly
    w = res["weights"].values
    cov = stocks_data.iloc[:100].cov().values
    vols = np.sqrt(np.diag(cov))

    p_vol = np.sqrt(w @ cov @ w)
    weighted_vol = w @ vols
    div_ratio = weighted_vol / p_vol

    expected_ratio = riskfolio_cv["mdiv"]["div_ratio"]
    assert np.isclose(div_ratio, expected_ratio, atol=1e-5)


def test_min_uci_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.iloc[:100].columns.tolist()
    port = Portfolio(assets=assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="UCI")
    
    res = optimize_portfolio(stocks_data.iloc[:100], port, optimize_method="ROI")
    w_ours = res["weights"].values
    
    # Ground truth from JSON
    w_expected = pd.Series(riskfolio_cv["min_uci"]["weights"]).values

    # We use a relatively high tolerance (1e-4) as Riskfolio and our ConvexOptimizer 
    # might use different default kwargs for the underlying SOCP solver (ECOS/SCS)
    np.testing.assert_allclose(w_ours, w_expected, atol=1e-4)


def test_max_martin_ratio_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.iloc[:100].columns.tolist()
    port = Portfolio(assets=assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    # No risk-free rate in R / Riskfolio default unless specified
    port.add_objective(type="return", name="mean", arguments={"risk_free_rate": 0.0})
    port.add_objective(type="risk", name="UCI")
    
    res = optimize_portfolio(stocks_data.iloc[:100], port, optimize_method="ROI")
    w_ours = res["weights"].values
    
    # Ground truth from JSON
    w_expected = pd.Series(riskfolio_cv["max_martin"]["weights"]).values

    np.testing.assert_allclose(w_ours, w_expected, atol=5e-4)
