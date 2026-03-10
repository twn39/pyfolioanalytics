import numpy as np
import pandas as pd
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_leverage_exposure_logic():
    # 1. Setup synthetic data where leverage will be active
    np.random.seed(42)
    T, N = 100, 4
    # Create returns such that some assets are much better than others
    R_raw = np.random.randn(T, N) * 0.01
    R_raw[:, 0] += 0.05 # Strong positive asset
    R_raw[:, 1] -= 0.05 # Strong negative asset
    
    asset_names = ["A", "B", "C", "D"]
    R = pd.DataFrame(R_raw, columns=asset_names)
    
    # Define Portfolio
    pspec = Portfolio(assets=asset_names)
    pspec.add_constraint(type="full_investment")
    # Allow shorting
    pspec.add_constraint(type="box", min=-1.0, max=1.0)
    
    # CASE A: No leverage constraint (unconstrained 130/30 or more)
    pspec_unconstrained = pspec.copy()
    pspec_unconstrained.add_objective(type="return", name="mean")
    res_un = optimize_portfolio(R=R, portfolio=pspec_unconstrained, optimize_method="cvxpy")
    
    # CASE B: Strict leverage constraint (120/20 => Gross=1.4)
    pspec_constrained = pspec.copy()
    leverage_limit = 1.4
    pspec_constrained.add_constraint(type="leverage_exposure", leverage=leverage_limit)
    pspec_constrained.add_objective(type="return", name="mean")
    res_con = optimize_portfolio(R=R, portfolio=pspec_constrained, optimize_method="cvxpy")
    
    # Assertions
    assert res_un['status'] == "optimal"
    assert res_con['status'] == "optimal"
    
    w_un = res_un['weights']
    w_con = res_con['weights']
    
    gross_un = np.sum(np.abs(w_un))
    gross_con = np.sum(np.abs(w_con))
    
    print(f"DEBUG: Gross Leverage Unconstrained: {gross_un:.4f}")
    print(f"DEBUG: Gross Leverage Constrained: {gross_con:.4f}")
    
    # 1. Constrained leverage must be within limit
    assert gross_con <= leverage_limit + 1e-8
    
    # 2. In this specific setup, unconstrained leverage should be much higher than constrained
    assert gross_un > gross_con
    
    # 3. Sum of weights must still be 1 (full investment)
    assert np.abs(np.sum(w_con) - 1.0) < 1e-8
    
    # 4. Verify the impact on return (constrained return should be lower than unconstrained)
    ret_un = w_un @ np.mean(R_raw, axis=0)
    ret_con = w_con @ np.mean(R_raw, axis=0)
    assert ret_un > ret_con
