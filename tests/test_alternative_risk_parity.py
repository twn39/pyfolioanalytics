import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import numerical_risk_contribution, CVaR, MAD


def test_numerical_risk_contribution_euler(stocks_data):
    # Setup simple data
    R = stocks_data.iloc[:, :4]
    w = np.array([0.4, 0.3, 0.2, 0.1])

    # Check CVaR
    rc_cvar = numerical_risk_contribution(w, R.values, CVaR, p=0.95)
    total_cvar = CVaR(w, R.values, p=0.95)

    # By Euler's theorem, sum of RC should equal the total risk
    assert np.isclose(np.sum(rc_cvar), total_cvar, rtol=1e-5)

    # Check MAD
    rc_mad = numerical_risk_contribution(w, R.values, MAD)
    total_mad = MAD(w, R.values)
    assert np.isclose(np.sum(rc_mad), total_mad, rtol=1e-5)


def test_cvar_risk_parity(stocks_data):
    # Setup simple data
    R = stocks_data.iloc[:, :4]

    # Create portfolio
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")

    # Add an objective for equal risk contribution (CVaR / CVaR)
    port.add_objective(
        type="risk_budget", name="CVaR", arguments={"p": 0.95}, min_concentration=True
    )

    # Test random optimization using transform method
    res = optimize_portfolio(R, port)

    assert res["status"] in ["optimal", "feasible", "optimal_inaccurate"]
    assert res["weights"] is not None

    # Check that it solved the risk parity properly
    pct_contrib = res["objective_measures"]["pct_contrib_CVaR"]

    # Each asset should contribute roughly 25% to the total risk
    target = np.full(4, 0.25)

    # The optimization might not be perfect with nonlinear solvers, but should be close
    np.testing.assert_allclose(pct_contrib, target, atol=0.05)


def test_mad_risk_parity(stocks_data):
    R = stocks_data.iloc[:, :4]

    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")

    # MAD Risk parity
    port.add_objective(type="risk_budget", name="MAD", min_concentration=True)

    res = optimize_portfolio(R, port)

    assert res["status"] in ["optimal", "feasible", "optimal_inaccurate"]

    pct_contrib = res["objective_measures"]["pct_contrib_MAD"]
    target = np.full(4, 0.25)

    np.testing.assert_allclose(pct_contrib, target, atol=0.05)


def test_evar_risk_parity(stocks_data):
    R = stocks_data.iloc[:, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(
        type="risk_budget", name="EVaR", arguments={"p": 0.95}, min_concentration=True
    )
    res = optimize_portfolio(R, port)

    assert res["status"] in ["optimal", "feasible", "optimal_inaccurate"]
    pct_contrib = res["objective_measures"]["pct_contrib_EVaR"]
    target = np.full(4, 0.25)

    # EVaR is smooth, so the parity should be extremely precise
    np.testing.assert_allclose(pct_contrib, target, atol=0.01)
