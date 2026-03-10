import numpy as np
import pandas as pd
import json
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_advanced_risk_real_cv():
    # 1. Load Data
    edhec = pd.read_csv("data/edhec.csv")
    with open("data/advanced_risk_real_cv.json", "r") as f:
        cv_data = json.load(f)
        
    asset_names = cv_data["asset_names"]
    R_sub = edhec[asset_names]
    
    # 2. Test RLDaR CV
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=1.0)
    port.add_objective(name="RLDaR", type="risk", arguments={"p": 0.95, "kappa": 0.3})
    
    res_rldar = optimize_portfolio(R_sub, port)
    assert res_rldar["status"] in ["optimal", "feasible"]
    
    w_py = res_rldar["weights"].values
    w_rp = np.array(cv_data["w_rldar"])
    
    # RLDaR is sensitive due to power cones and drawdown variables. 1e-3 is reasonable.
    np.testing.assert_allclose(w_py, w_rp, atol=1e-3)
    
    # 3. Test L-Moment CRM CV
    port.clear_objectives()
    port.add_objective(name="L_Moment_CRM", type="risk", arguments={"k": 4, "method": "MSD"})
    
    res_crm = optimize_portfolio(R_sub, port)
    assert res_crm["status"] in ["optimal", "feasible"]
    
    w_py_crm = res_crm["weights"].values
    w_ref_crm = np.array(cv_data["w_crm_msd"])
    
    # Check weights
    np.testing.assert_allclose(w_py_crm, w_ref_crm, atol=1e-4)
