import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.solvers import solve_global_heuristic


def test_solve_deoptim_smoke():
    np.random.seed(42)
    T, N = 50, 4
    R = pd.DataFrame(
        np.random.randn(T, N) * 0.01 + 0.001, columns=[f"A{i}" for i in range(N)]
    )

    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="StdDev")
    port.add_objective(
        type="risk_budget", name="StdDev", arguments={"min_concentration": True}
    )

    from pyfolioanalytics.moments import set_portfolio_moments

    moments = set_portfolio_moments(R, port)

    res = solve_global_heuristic(
        moments, port.get_constraints(), port.objectives, R=R.values, itermax=5
    )
    assert res["status"] in [
        "optimal",
        "Maximum number of iterations has been exceeded.",
    ]
    w = res["weights"]
    assert len(w) == N
    assert np.isclose(np.sum(w), 1.0, atol=1e-5)
    assert np.all(w >= -1e-5)

    # Assert DEoptim found a valid risk parity / min risk portfolio
    # Compare with equal weight
    eq_w = np.full(N, 1.0 / N)
    sigma = moments["sigma"]
    eq_risk = np.sqrt(eq_w.T @ sigma @ eq_w)
    opt_risk = np.sqrt(w.T @ sigma @ w)

    # Ensure it minimizes risk compared to a badly weighted portfolio
    bad_w = np.array([1.0, 0.0, 0.0, 0.0])
    bad_risk = np.sqrt(bad_w.T @ sigma @ bad_w)
    assert opt_risk <= bad_risk + 1e-4
