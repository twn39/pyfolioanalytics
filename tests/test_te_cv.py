import json
import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_tracking_error_cross_validation():
    with open("data/te_cv.json", "r") as f:
        cv_data = json.load(f)
    
    R_data = np.array(cv_data["returns"])
    asset_names = [f"A{i+1}" for i in range(R_data.shape[1])]
    R = pd.DataFrame(R_data, columns=asset_names)
    
    w_b = np.array(cv_data["benchmark_weights"])
    te_target = cv_data["te_target"]
    if isinstance(te_target, list):
        te_target = te_target[0]
    r_weights = np.array(cv_data["opt_weights"])
    
    # Python Optimization
    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint("full_investment")
    portfolio.add_constraint("long_only")
    
    # Map benchmark_weights back to dict for the portfolio
    benchmark_dict = dict(zip(asset_names, w_b))
    portfolio.add_constraint("tracking_error", target=te_target, benchmark=benchmark_dict)
    
    # Objective: Minimize Variance (StdDev)
    portfolio.add_objective(type="risk", name="StdDev")
    
    res_py = optimize_portfolio(R, portfolio)
    py_weights = res_py["weights"].values
    
    # Check parity with R
    np.testing.assert_allclose(py_weights, r_weights, atol=1e-5)
    
    # Verify tracking error calculation
    assert "tracking_error" in res_py["objective_measures"]
    te_actual = res_py["objective_measures"]["tracking_error"]
    np.testing.assert_allclose(te_actual, cv_data["te_actual"], rtol=1e-5)
    assert te_actual <= te_target + 1e-6
