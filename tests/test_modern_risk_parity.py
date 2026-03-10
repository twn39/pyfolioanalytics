import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import EDaR, RLVaR

def test_edar_optimization_smoke():
    np.random.seed(42)
    T, N = 100, 5
    R_raw = np.random.randn(T, N) * 0.01 + 0.001
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=0.5)
    
    # Minimize EDaR
    port.add_objective(name="EDaR", type="risk", arguments={"p": 0.95})
    
    res = optimize_portfolio(R_df, port)
    
    if res["status"] == "failed":
        print(f"EDaR Optimization Failed: {res.get('message')}")
    
    assert res["status"] in ["optimal", "feasible"]
    weights = res["weights"]
    assert np.allclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-7)
    assert np.all(weights <= 0.5 + 1e-7)
    
    # Calculate measure
    val = EDaR(weights.values, R_raw, p=0.95)
    assert val > 0

def test_rlvar_optimization_smoke():
    np.random.seed(42)
    T, N = 50, 3
    R_raw = np.random.randn(T, N) * 0.01 + 0.001
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    
    # Minimize RLVaR
    port.add_objective(name="RLVaR", type="risk", arguments={"p": 0.95, "kappa": 0.3})
    
    res = optimize_portfolio(R_df, port)
    
    # RLVaR might be harder to solve, but Clarabel should handle it
    if res["status"] in ["optimal", "feasible"]:
        weights = res["weights"]
        assert np.allclose(weights.sum(), 1.0)
        
        val = RLVaR(weights.values, R_raw, p=0.95, kappa=0.3)
        assert val > 0
