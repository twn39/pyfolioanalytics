import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import RLDaR

def test_rldar_optimization_smoke():
    np.random.seed(42)
    T, N = 100, 5
    R_raw = np.random.randn(T, N) * 0.01 + 0.001
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=0.5)
    
    # Minimize RLDaR
    port.add_objective(name="RLDaR", type="risk", arguments={"p": 0.95, "kappa": 0.3})
    
    res = optimize_portfolio(R_df, port)
    
    assert res["status"] in ["optimal", "feasible"]
    weights = res["weights"]
    assert np.allclose(weights.sum(), 1.0)
    
    # Check measure
    val = RLDaR(weights.values, R_raw, p=0.95, kappa=0.3)
    assert val > 0

def test_l_moment_crm_optimization_smoke():
    np.random.seed(42)
    T, N = 100, 5
    R_raw = np.random.randn(T, N) * 0.01 + 0.001
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    
    # Minimize L-Moment CRM (weighted sum of L1, L2, L3, L4)
    # Using 'MSD' method for OWA weights
    port.add_objective(name="L_Moment_CRM", type="risk", arguments={"k": 4, "method": "MSD"})
    
    res = optimize_portfolio(R_df, port)
    
    assert res["status"] in ["optimal", "feasible"]
    weights = res["weights"]
    assert np.allclose(weights.sum(), 1.0)
    
    # Verify measure calculation in results
    assert "L_Moment_CRM" in res["objective_measures"]
    assert res["objective_measures"]["L_Moment_CRM"] > 0
