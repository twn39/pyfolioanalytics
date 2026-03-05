import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True).iloc[:100]
    return df

def test_kelly_optimization_parity(stocks_data):
    # Our implementation
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="Kelly")
    w_ours = res["weights"].values
    
    # Riskfolio-lib implementation
    import riskfolio as rp
    port = rp.Portfolio(returns=stocks_data)
    port.assets_stats(method_mu='hist', method_cov='hist')
    # Use same constraints (long-only, full investment)
    w_rp = port.optimization(model='Classic', rm='MV', obj='Sharpe', kelly='exact', rf=0)
    w_rf = w_rp["weights"].values
    
    np.testing.assert_allclose(w_ours, w_rf, atol=1e-5)

def test_max_diversification_parity(stocks_data):
    # Our implementation
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="MDIV")
    w_ours = res["weights"].values
    
    # Riskfolio-lib implementation
    # MDIV is Sharpe Ratio with mu = volatilities
    import riskfolio as rp
    port = rp.Portfolio(returns=stocks_data)
    # Force mu to be volatilities
    vols = stocks_data.std().to_frame().T
    port.mu = vols
    port.cov = stocks_data.cov()
    
    w_rp = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0)
    w_rf = w_rp["weights"].values
    
    np.testing.assert_allclose(w_ours, w_rf, atol=1e-5)

def test_kelly_optimization_internal(stocks_data):
    # (Existing internal tests...)
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="Kelly")
    assert res["weights"] is not None
    assert np.allclose(res["weights"].sum(), 1.0)

def test_max_diversification_internal(stocks_data):
    # (Existing internal tests...)
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    res = optimize_portfolio(stocks_data, portfolio, optimize_method="MDIV")
    assert res["weights"] is not None
    assert np.allclose(res["weights"].sum(), 1.0)
