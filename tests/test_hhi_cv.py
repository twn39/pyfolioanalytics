import numpy as np
import pandas as pd
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_hhi_diversification_logic():
    # 1. Setup synthetic data
    np.random.seed(42)
    T, N = 100, 10
    R_raw = np.random.randn(T, N) * 0.01
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R = pd.DataFrame(R_raw, columns=asset_names)
    
    # Define Portfolio
    pspec = Portfolio(assets=asset_names)
    pspec.add_constraint(type="full_investment")
    pspec.add_constraint(type="box", min=0, max=1)
    
    # CASE A: Min Variance without HHI constraint
    res_un = optimize_portfolio(R=R, portfolio=pspec, optimize_method="cvxpy")
    div_un = 1.0 - np.sum(res_un['weights']**2)
    
    # CASE B: Add strict Diversification Constraint (Div >= 0.9)
    # This means weights must be more spread out than in min-var
    div_target = 0.9
    pspec_con = pspec.copy()
    pspec_con.add_constraint(type="diversification", div_target=div_target)
    res_con = optimize_portfolio(R=R, portfolio=pspec_con, optimize_method="cvxpy")
    
    assert res_un['status'] == "optimal"
    assert res_con['status'] == "optimal"
    
    div_actual = 1.0 - np.sum(res_con['weights']**2)
    
    # Python CVXPY MUST satisfy the constraint
    assert div_actual >= div_target - 1e-8
    
    # In this setup, unconstrained min-var should be less diversified than constrained
    assert div_un < div_actual
    
    print(f"DEBUG: Unconstrained Div: {div_un:.4f}, Constrained Div: {div_actual:.4f}")

def test_hhi_alias():
    # Test that 'HHI' type also works and hhi_target is correctly converted
    pspec = Portfolio(assets=5)
    # HHI <= 0.2  => Div >= 0.8
    pspec.add_constraint(type="HHI", hhi_target=0.2)
    
    constr = pspec.get_constraints()
    assert constr['div_target'] == 0.8
