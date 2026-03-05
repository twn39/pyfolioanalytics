import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True).iloc[:100]
    return df

def test_kelly_optimization(stocks_data):
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="Kelly")
    
    assert res["weights"] is not None
    assert np.allclose(res["weights"].sum(), 1.0)
    assert np.all(res["weights"] >= -1e-7)
    # Kelly should favor high growth assets
    # Check if weights are concentrated
    assert (res["weights"] > 0.05).any()

def test_max_diversification(stocks_data):
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="MDIV")
    
    assert res["weights"] is not None
    assert np.allclose(res["weights"].sum(), 1.0)
    assert np.all(res["weights"] >= -1e-7)
    
    # Check if Diversification Ratio is calculated
    # Ratio = (w' @ sigma) / sqrt(w' @ Sigma @ w)
    w = res["weights"].values
    cov = stocks_data.cov().values
    vols = np.sqrt(np.diag(cov))
    
    p_vol = np.sqrt(w @ cov @ w)
    weighted_vol = w @ vols
    div_ratio = weighted_vol / p_vol
    
    # Typically div_ratio > 1 for diversified portfolios
    assert div_ratio >= 1.0
    
    # Compare with Equal Weight diversification
    w_ew = np.full(len(assets), 1.0 / len(assets))
    p_vol_ew = np.sqrt(w_ew @ cov @ w_ew)
    weighted_vol_ew = w_ew @ vols
    div_ratio_ew = weighted_vol_ew / p_vol_ew
    
    assert div_ratio >= div_ratio_ew - 1e-7
