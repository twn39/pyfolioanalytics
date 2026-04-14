import numpy as np
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_l_inf_tracking_error(stocks_data):
    R = stocks_data.iloc[:, :5]
    assets = list(R.columns)
    
    # 1. Establish benchmark (e.g. Equal Weight)
    w_bench = np.full(5, 1.0 / 5)
    
    port = Portfolio(assets=assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    # Restrict maximum absolute deviation from benchmark to 5% (0.05)
    port.add_constraint(type="tracking_error", benchmark=w_bench, target=0.05, p_norm="inf")
    
    # Objective: Minimize standard deviation
    port.add_objective(type="risk", name="StdDev")
    
    res = optimize_portfolio(R, port, optimize_method="ROI")
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    w_opt = res["weights"]
    
    # Check bounds
    assert np.allclose(w_opt.sum(), 1.0)
    assert np.all(w_opt >= -1e-6)
    
    # Check L-inf tracking error strictly bounds single deviations
    diff = np.abs(w_opt - w_bench)
    assert np.max(diff) <= 0.05 + 1e-5
    
    # Confirm it actually tried to move away from benchmark (min_vol != EW usually)
    assert np.max(diff) > 0.01  # it should have used the slack
    
def test_l1_tracking_error(stocks_data):
    R = stocks_data.iloc[:, :5]
    assets = list(R.columns)
    w_bench = np.full(5, 1.0 / 5)
    
    port = Portfolio(assets=assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    # L1 tracking error (like Active Share * 2) bounded to 10%
    port.add_constraint(type="tracking_error", benchmark=w_bench, target=0.10, p_norm=1)
    port.add_objective(type="risk", name="StdDev")
    
    res = optimize_portfolio(R, port, optimize_method="ROI")
    w_opt = res["weights"]
    diff = np.abs(w_opt - w_bench)
    assert np.sum(diff) <= 0.10 + 1e-5

