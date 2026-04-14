import numpy as np
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

import json
import os

def test_l_inf_tracking_error(stocks_data):
    R = stocks_data.iloc[:100, :5]
    assets = list(R.columns)
    
    # Load ground truth
    json_path = "data/tracking_cv.json"
    if not os.path.exists(json_path):
        pytest.skip("data/tracking_cv.json not found. Run scripts/generate_tracking_cv.py first.")
        
    with open(json_path, "r") as f:
        gt = json.load(f)
        
    w_expected = np.array(gt["L_inf_weights"])
    
    w_bench = np.full(5, 1.0 / 5)
    
    port = Portfolio(assets=assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    port.add_constraint(type="tracking_error", benchmark=w_bench, target=0.05, p_norm="inf")
    
    port.add_objective(type="risk", name="StdDev")
    
    res = optimize_portfolio(R, port, optimize_method="ROI", solver="SCS")
    w_ours = res["weights"].values
    
    np.testing.assert_allclose(w_ours, w_expected, atol=5e-4)
    
def test_l1_tracking_error(stocks_data):
    R = stocks_data.iloc[:100, :5]
    assets = list(R.columns)
    
    json_path = "data/tracking_cv.json"
    with open(json_path, "r") as f:
        gt = json.load(f)
        
    w_expected = np.array(gt["L_1_weights"])
    
    w_bench = np.full(5, 1.0 / 5)
    
    port = Portfolio(assets=assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    port.add_constraint(type="tracking_error", benchmark=w_bench, target=0.10, p_norm=1)
    port.add_objective(type="risk", name="StdDev")
    
    res = optimize_portfolio(R, port, optimize_method="ROI", solver="SCS")
    w_ours = res["weights"].values
    
    np.testing.assert_allclose(w_ours, w_expected, atol=5e-4)

