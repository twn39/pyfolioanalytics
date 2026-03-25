import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_turnover_constraint():
    # Generate synthetic data
    np.random.seed(42)
    n_assets = 5
    n_obs = 100
    asset_names = [f"A{i}" for i in range(n_assets)]
    R = pd.DataFrame(np.random.randn(n_obs, n_assets), columns=asset_names)
    
    # 1. Base optimization without turnover constraint
    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint("full_investment")
    portfolio.add_constraint("long_only")
    portfolio.add_objective("StdDev")
    
    res_base = optimize_portfolio(R, portfolio)
    w_base = res_base["weights"].values
    
    # 2. Add turnover constraint relative to a far-away initial weight
    # If we set w_init to be very different, the constraint should pull the new weights toward it
    w_init = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Calculate distance from base to init
    dist_base = np.sum(np.abs(w_base - w_init))
    
    # Set a turnover target smaller than the current distance
    turnover_target = dist_base * 0.5
    
    p_to = Portfolio(assets=asset_names)
    p_to.add_constraint("full_investment")
    p_to.add_constraint("long_only")
    p_to.add_constraint("turnover", turnover_target=turnover_target, weight_initial=w_init)
    p_to.add_objective("StdDev")
    
    res_to = optimize_portfolio(R, p_to)
    w_to = res_to["weights"].values
    
    # Verify turnover constraint is satisfied
    actual_turnover = np.sum(np.abs(w_to - w_init))
    assert actual_turnover <= turnover_target + 1e-6
    
    # 3. Add proportional transaction costs (ptc) to the objective
    ptc_rate = 0.05
    p_tc = Portfolio(assets=asset_names)
    p_tc.add_constraint("full_investment")
    p_tc.add_constraint("long_only")
    p_tc.add_constraint("transaction_cost", ptc=ptc_rate, weight_initial=w_init)
    p_tc.add_objective("StdDev")
    
    res_tc = optimize_portfolio(R, p_tc)
    w_tc = res_tc["weights"].values
    
    # Cost should pull weights closer to w_init compared to base
    dist_tc = np.sum(np.abs(w_tc - w_init))
    assert dist_tc < dist_base - 1e-4, f"TC penalty should reduce turnover, but {dist_tc} >= {dist_base}"
    assert "transaction_cost" in res_tc["objective_measures"]
    
    # Verify weights are different from base
    assert not np.allclose(w_to, w_base, atol=1e-4)
    
    # Verify it is closer to w_init than w_base was
    assert actual_turnover < dist_base
