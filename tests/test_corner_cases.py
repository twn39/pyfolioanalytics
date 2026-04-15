import numpy as np
import pandas as pd

from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_single_asset_portfolio():
    """Test N=1 edge case for optimization and risk attribution."""
    # N=1 means optimization should trivially return 1.0 (if fully invested)
    R = pd.DataFrame(np.random.randn(100, 1), columns=["Asset_X"])
    
    port = Portfolio(assets=["Asset_X"])
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="StdDev")
    
    # Should not crash on shape errors or division by zero
    res = optimize_portfolio(R, port)
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    assert len(res["weights"]) == 1
    assert np.isclose(res["weights"]["Asset_X"], 1.0)
    
def test_dollar_neutral_with_ratio():
    """Test target_sum=0 (Market Neutral) when executing Charnes-Cooper Ratio Optimization."""
    # Normally Ratio = Return / Risk, scaled by kappa. 
    # With sum=0, sum(y) = 0 * kappa = 0.
    R = pd.DataFrame(np.random.randn(100, 3) * 0.01, columns=["A", "B", "C"])
    R["A"] += 0.005  # Make A slightly profitable
    
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="dollar_neutral") # min_sum = 0, max_sum = 0
    port.add_constraint(type="box", min=[-1, -1, -1], max=[1, 1, 1])
    
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="StdDev")
    
    res = optimize_portfolio(R, port, optimize_method="ROI")
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    w = res["weights"].values
    
    # Check that it actually kept it neutral
    assert np.isclose(np.sum(w), 0.0, atol=1e-5)
    
def test_singular_covariance():
    """Test optimization behavior when assets are perfectly collinear (Singular Covariance Matrix)."""
    # Create two identical assets
    np.random.seed(42)
    s = np.random.randn(100)
    R = pd.DataFrame({
        "A": s * 0.01 + 0.002,
        "B": s * 0.01 + 0.002,
        "C": np.random.randn(100) * 0.01
    })
    
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="CVaR")
    
    # CVXPY SOCP formulation should gracefully handle collinearity
    res = optimize_portfolio(R, port)
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    assert np.isclose(res["weights"].sum(), 1.0)

