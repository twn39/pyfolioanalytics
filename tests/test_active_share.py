import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_active_share_constraint_smoke():
    np.random.seed(42)
    T, N = 100, 5
    R_raw = np.random.randn(T, N) * 0.01 + 0.001
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=1.0)
    
    # Benchmark: Equal Weight (0.2 each)
    benchmark = {name: 0.2 for name in asset_names}
    
    # Add Active Share constraint: very tight, must be close to benchmark
    port.add_constraint(type="active_share", target=0.05, benchmark=benchmark)
    
    # Objective: Maximize return (which usually pushes weights to boundaries)
    port.add_objective(type="return")
    
    res = optimize_portfolio(R_df, port)
    
    assert res["status"] in ["optimal", "feasible"]
    weights = res["weights"]
    
    # Calculate actual Active Share
    w_py = weights.values
    w_b = np.array([0.2] * N)
    actual_as = 0.5 * np.sum(np.abs(w_py - w_b))
    
    # Should be <= 0.05
    assert actual_as <= 0.05 + 1e-7

def test_active_share_large_target():
    np.random.seed(42)
    T, N = 100, 5
    R_raw = np.random.randn(T, N) * 0.01 + 0.001
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R_df = pd.DataFrame(R_raw, columns=asset_names)
    
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    
    # Benchmark: Equal Weight
    benchmark = {name: 0.2 for name in asset_names}
    
    # Large Active Share target (0.8), should allow more deviation
    port.add_constraint(type="active_share", target=0.8, benchmark=benchmark)
    port.add_objective(type="return")
    
    res = optimize_portfolio(R_df, port)
    
    assert res["status"] in ["optimal", "feasible"]
    weights = res["weights"]
    w_py = weights.values
    w_b = np.array([0.2] * N)
    actual_as = 0.5 * np.sum(np.abs(w_py - w_b))
    
    # Should be significantly larger than 0.05 if it's maximizing return
    assert actual_as > 0.1
    assert actual_as <= 0.8 + 1e-7
