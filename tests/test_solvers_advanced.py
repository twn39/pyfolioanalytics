import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.solvers import solve_deoptim

def test_solve_deoptim_smoke():
    np.random.seed(42)
    T, N = 50, 4
    R = pd.DataFrame(np.random.randn(T, N) * 0.01 + 0.001, columns=[f"A{i}" for i in range(N)])
    
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="StdDev")
    port.add_objective(type="risk_budget", name="StdDev", arguments={"min_concentration": True})
    
    from pyfolioanalytics.moments import set_portfolio_moments
    moments = set_portfolio_moments(R, port)
    
    res = solve_deoptim(moments, port.get_constraints(), port.objectives, R=R.values, itermax=5)
    assert res["status"] in ["optimal", "Maximum number of iterations has been exceeded."]
    assert len(res["weights"]) == N
    assert np.isclose(np.sum(res["weights"]), 1.0, atol=1e-5)

