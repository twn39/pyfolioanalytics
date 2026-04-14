import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.backtest import backtest_portfolio

def test_aum_based_fees(stocks_data):
    R = stocks_data.iloc[:500, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="StdDev")
    
    # 1. Base case: no fees, no ptc
    res_base = backtest_portfolio(R, port, rebalance_periods="ME")
    base_cum_ret = (1 + res_base.net_returns).prod() - 1
    
    # 2. Add Management Fee (e.g. 10% annualized just to see effect)
    res_mgt = backtest_portfolio(R, port, rebalance_periods="ME", management_fee=0.10)
    mgt_cum_ret = (1 + res_mgt.net_returns).prod() - 1
    assert mgt_cum_ret < base_cum_ret
    
    # 3. Add Performance Fee (20% above HWM)
    res_perf = backtest_portfolio(R, port, rebalance_periods="ME", performance_fee=0.20)
    perf_cum_ret = (1 + res_perf.net_returns).prod() - 1
    assert perf_cum_ret < base_cum_ret
    
    # 4. Both Fees + PTC
    res_all = backtest_portfolio(R, port, rebalance_periods="ME", management_fee=0.02, performance_fee=0.20, ptc=0.005)
    all_cum_ret = (1 + res_all.net_returns).prod() - 1
    assert all_cum_ret < perf_cum_ret
    
    # Compare tearsheet
    summ = res_all.summary()
    assert summ.loc["Total Return", "Net"] < summ.loc["Total Return", "Gross"]

