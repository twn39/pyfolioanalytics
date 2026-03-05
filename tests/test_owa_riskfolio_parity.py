import pytest
import pandas as pd
import numpy as np
import json
import os
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import owa_gmd_weights

def test_owa_gmd_riskfolio_parity():
    # Load EDHEC data
    data_path = os.path.join(os.path.dirname(__file__), "../data/edhec.csv")
    df = pd.read_csv(data_path, sep=';', index_col=0)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100.0
    R = df.iloc[:, :10]
    
    # Load Riskfolio ground truth
    json_path = os.path.join(os.path.dirname(__file__), "../data/owa_riskfolio_parity.json")
    if not os.path.exists(json_path):
        pytest.skip("owa_riskfolio_parity.json ground truth not found.")
    
    with open(json_path, 'r') as f:
        gt = json.load(f)
    
    # Create Portfolio for pyfolioanalytics
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")
    
    # OWA GMD Weights for pyfolioanalytics
    w_gmd = owa_gmd_weights(len(R))
    port.add_objective(type="risk", name="OWA", arguments={"owa_weights": w_gmd})
    
    res = optimize_portfolio(R, port)
    
    assert res["status"] == "optimal"
    
    # Parity check
    py_weights = res["weights"].values
    rf_weights = np.array(gt["weights"])
    
    from pyfolioanalytics.risk import owa_risk
    py_risk_with_py_w = owa_risk(py_weights, R.values, w_gmd)
    py_risk_with_rf_w = owa_risk(rf_weights, R.values, w_gmd)
    
    print(f"\nPy risk with Py weights: {py_risk_with_py_w}")
    print(f"Py risk with Rf weights: {py_risk_with_rf_w}")
    
    # If py_risk_with_py_w <= py_risk_with_rf_w, then py solver is actually better or equal!
    # This would mean the formulation is correct.
    
    # Check risk parity
    np.testing.assert_allclose(py_risk_with_py_w, py_risk_with_rf_w, rtol=1e-4)

    # Use higher tolerance for optimized weights
    np.testing.assert_allclose(py_weights, rf_weights, atol=2e-2, rtol=1e-1)
