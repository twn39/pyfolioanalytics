import numpy as np
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_cross_solver_stability(stocks_data):
    """
    Test that mathematically equivalent convex problems yield the same optimal weights
    regardless of the underlying solver engine (CLARABEL vs OSQP vs SCS).
    This acts as a proof of mathematical stability for our risk models.
    """
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    # Use StdDev (Quadratic) to test cross-solver capabilities for all convex solvers
    port.add_objective(type="risk", name="StdDev")
    
    # 1. Force SCS Solver (First-order Splitting Conic Solver)
    res_scs = optimize_portfolio(R, port, optimize_method="ROI", solver="SCS")
    assert res_scs["status"] in ["optimal", "optimal_inaccurate"]
    w_scs = res_scs["weights"]
    
    # 2. Force OSQP Solver (Operator Splitting Quadratic Program)
    res_osqp = optimize_portfolio(R, port, optimize_method="ROI", solver="OSQP")
    assert res_osqp["status"] in ["optimal", "optimal_inaccurate"]
    w_osqp = res_osqp["weights"]
    
    # 3. Force CLARABEL Solver (Modern robust Interior Point)
    res_clarabel = optimize_portfolio(R, port, optimize_method="ROI", solver="CLARABEL")
    assert res_clarabel["status"] in ["optimal", "optimal_inaccurate"]
    w_clarabel = res_clarabel["weights"]
    
    # Assert Cross-Solver Homotopy / Equivalence
    # Because solvers have different precision limits (SCS is often ~1e-4), we use a tolerant atol.
    assert np.allclose(w_scs, w_osqp, atol=2e-3), f"SCS vs OSQP mismatch: {w_scs} vs {w_osqp}"
    assert np.allclose(w_osqp, w_clarabel, atol=2e-3), f"OSQP vs CLARABEL mismatch: {w_osqp} vs {w_clarabel}"

def test_fallback_cascade(stocks_data):
    """
    Test the cascade fallback mechanism by asking for a dummy non-existent solver.
    It should warn, then fall back to CLARABEL/OSQP/SCS and succeed.
    """
    R = stocks_data.iloc[:, :3]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="StdDev")
    
    with pytest.warns(UserWarning, match="attempting fallback solvers"):
        res = optimize_portfolio(R, port, optimize_method="ROI", solver="NON_EXISTENT_SOLVER_123")
        
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    assert np.isclose(np.sum(res["weights"]), 1.0)

