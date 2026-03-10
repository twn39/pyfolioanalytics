import json
import numpy as np
import pandas as pd
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def load_cv_data():
    with open("data/factor_exposure_cv.json", "r") as f:
        return json.load(f)

def test_factor_exposure_cv():
    data = load_cv_data()
    R_raw = np.array(data['input_R'])
    B = np.array(data['B'])
    lower = np.array(data['lower'])
    upper = np.array(data['upper'])
    expected_weights = np.array(data['weights'])
    
    T, N = R_raw.shape
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R = pd.DataFrame(R_raw, columns=asset_names)
    
    # Define Portfolio
    pspec = Portfolio(assets=asset_names)
    pspec.add_constraint(type="full_investment")
    pspec.add_constraint(type="box", min=0, max=0.4)
    
    # Add Factor Exposure Constraint
    pspec.add_constraint(type="factor_exposure", B=B, lower=lower, upper=upper)
    
    # Maximize Return Objective
    pspec.add_objective(type="return", name="mean")
    
    # Solve
    res = optimize_portfolio(R=R, portfolio=pspec, optimize_method="cvxpy")
    
    assert res['status'] == "optimal"
    
    # Parity check on weights
    # Since it's a linear problem, there might be multiple optima if Returns are linear,
    # but ROI and CVXPY usually pick similar points if possible.
    # However, exposure should DEFINITELY be the same and within bounds.
    np.testing.assert_allclose(res['weights'], expected_weights, rtol=1e-5, atol=1e-5)
    
    # Check exposure actuals
    actual_exposure = B.T @ res['weights']
    np.testing.assert_allclose(actual_exposure, np.array(data['exposure_actual']), rtol=1e-6, atol=1e-6)
    
    # Validate bounds
    assert np.all(actual_exposure >= lower - 1e-9)
    assert np.all(actual_exposure <= upper + 1e-9)
