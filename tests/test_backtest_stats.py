import pandas as pd
import numpy as np
from pyfolioanalytics.backtest import BacktestResult

def test_backtest_result_stats():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Simulate a steady growth of 0.1% per day
    returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
    weights = pd.DataFrame({"A": [0.5]*100, "B": [0.5]*100}, index=dates)
    turnover = pd.Series([0.0]*100, index=dates)
    turnover.iloc[0] = 1.0 # First day setup
    
    res = BacktestResult(
        weights=weights,
        returns=returns,
        opt_results=[],
        eop_weights=weights,
        turnover=turnover,
        net_returns=returns - 0.0001 # Small fee
    )
    
    # We will add summary stats to BacktestResult next. 
    # Let's ensure the object structure holds these values.
    assert (res.net_returns < res.returns).all()
    assert res.turnover.sum() == 1.0

def test_backtest_summary():
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Simulate a steady growth of 0.1% per day
    returns = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)
    weights = pd.DataFrame({"A": [0.5]*252, "B": [0.5]*252}, index=dates)
    turnover = pd.Series([0.0]*252, index=dates)
    turnover.iloc[0] = 1.0 # First day setup
    
    res = BacktestResult(
        weights=weights,
        returns=returns,
        opt_results=[],
        eop_weights=weights,
        turnover=turnover,
        net_returns=returns - 0.0001 # Small fee
    )
    
    summary_df = res.summary()
    assert "Gross" in summary_df.columns
    assert "Net" in summary_df.columns
    assert "Sharpe Ratio" in summary_df.index
    assert "Max Drawdown" in summary_df.index
    assert "Total Turnover" in summary_df.index
    
    assert summary_df.loc["Total Turnover", "Gross"] == 1.0
    assert summary_df.loc["Total Return", "Gross"] > summary_df.loc["Total Return", "Net"]
