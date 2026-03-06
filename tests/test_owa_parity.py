import pytest
import pandas as pd
import numpy as np
import json
import os
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import owa_risk, owa_gmd_weights, owa_cvar_weights

def test_owa_weights_and_risk_parity():
    # Load EDHEC data
    data_path = os.path.join(os.path.dirname(__file__), "../data/edhec.csv")
    df = pd.read_csv(data_path, index_col=0)
    # Use same assets as julia script (2:11 -> first 10 columns)
    R = df.iloc[:, :10]
    T, N = R.shape
    
    # Load Julia ground truth
    json_path = os.path.join(os.path.dirname(__file__), "../data/owa_parity.json")
    if not os.path.exists(json_path):
        pytest.skip("owa_parity.json ground truth not found. Run julia script first.")
    
    with open(json_path, 'r') as f:
        gt = json.load(f)
    
    # 1. Verify GMD weights
    py_w_gmd = owa_gmd_weights(T)
    # Python uses non-increasing weights for convex minimization of decreasingly sorted losses.
    # Julia uses increasing weights.
    np.testing.assert_allclose(py_w_gmd[0], gt["owa_gmd_weights_julia"][-1], rtol=1e-7)
    np.testing.assert_allclose(py_w_gmd[-1], gt["owa_gmd_weights_julia"][0], rtol=1e-7)
    
    # 2. Verify CVaR weights
    py_w_cvar = owa_cvar_weights(T, p=0.95)
    # Julia is negative, Python is positive
    np.testing.assert_allclose(py_w_cvar[0], abs(gt["owa_cvar_weights_julia"][0]), rtol=1e-7)
    
    # 3. Verify risk calculation for dummy portfolio
    dummy_w = np.array(gt["dummy_weights"])
    # risk_gmd: Julia is dot(negative_to_positive_weights, sorted_losses)
    # My weights are translated exactly.
    py_risk_gmd = owa_risk(dummy_w, R.values, py_w_gmd)
    np.testing.assert_allclose(py_risk_gmd, gt["expected_gmd_risk_julia"], rtol=1e-7)
    
    # risk_cvar: Julia is negative, Python is positive
    py_risk_cvar = owa_risk(dummy_w, R.values, py_w_cvar)
    np.testing.assert_allclose(py_risk_cvar, abs(gt["expected_cvar_risk_julia"]), rtol=1e-7)

def test_owa_optimization_basic():
    data_path = os.path.join(os.path.dirname(__file__), "../data/edhec.csv")
    df = pd.read_csv(data_path, index_col=0)
    R = df.iloc[:, :10]
    
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")
    
    # Minimize GMD Risk
    w_gmd = owa_gmd_weights(len(R))
    port.add_objective(type="risk", name="OWA", arguments={"owa_weights": w_gmd})
    
    res = optimize_portfolio(R, port)
    
    assert res["status"] == "optimal"
    assert np.isclose(res["weights"].sum(), 1.0, atol=1e-6)
    assert np.all(res["weights"] >= -1e-6)
