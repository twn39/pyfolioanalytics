import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

import json
import os

@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True).iloc[:100]
    return df

@pytest.fixture
def riskfolio_cv():
    path = "data/riskfolio_cv.json"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found. Run scripts/generate_riskfolio_cv.py first.")
    with open(path, "r") as f:
        return json.load(f)

def test_kelly_optimization_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="Kelly")
    w_ours = res["weights"].values
    
    # Ground truth from JSON
    w_expected = pd.Series(riskfolio_cv["kelly"]["weights"]).values
    
    np.testing.assert_allclose(w_ours, w_expected, atol=1e-5)

def test_max_diversification_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="MDIV")
    w_ours = res["weights"].values
    
    # Ground truth from JSON
    w_expected = pd.Series(riskfolio_cv["mdiv"]["weights"]).values
    
    np.testing.assert_allclose(w_ours, w_expected, atol=1e-5)

def test_mdiv_ratio_parity(stocks_data, riskfolio_cv):
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="MDIV")
    
    # Check if Diversification Ratio is calculated correctly
    w = res["weights"].values
    cov = stocks_data.cov().values
    vols = np.sqrt(np.diag(cov))
    
    p_vol = np.sqrt(w @ cov @ w)
    weighted_vol = w @ vols
    div_ratio = weighted_vol / p_vol
    
    expected_ratio = riskfolio_cv["mdiv"]["div_ratio"]
    assert np.isclose(div_ratio, expected_ratio, atol=1e-5)
