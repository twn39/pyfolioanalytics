import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.backtest import backtest_portfolio

def test_realistic_backtest_drift():
    # Simple artificial data
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    R = pd.DataFrame({
        "A": [0.0, 0.1, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0],
        "B": [0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0]
    }, index=dates)
    
    port = Portfolio(assets=["A", "B"])
    # initial optimization uses just the start, so we force equal weight by some logic or pass custom weights
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="StdDev")
    
    res = backtest_portfolio(R, port, rebalance_periods="W", ptc=0.01)
    
    # Check that it returns net_returns and turnover
    assert hasattr(res, 'net_returns')
    assert hasattr(res, 'turnover')
    assert hasattr(res, 'eop_weights') # end of period weights (drifted)
    
    assert res.turnover.sum() > 0
    assert (res.returns > res.net_returns).any() # fees deducted