import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_deoptim(stocks_data):
    R = stocks_data.iloc[:, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="StdDev")
    
    res = optimize_portfolio(R, port, optimize_method="DEoptim", itermax=10)
    assert res["status"] == "optimal"
    assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)
    assert np.all(res["weights"] >= -1e-3)

def test_gensa(stocks_data):
    R = stocks_data.iloc[:, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    # Add a non-convex objective: risk budget
    port.add_objective(type="risk_budget", name="StdDev", min_concentration=True)
    
    # Run with dual_annealing
    res = optimize_portfolio(R, port, optimize_method="GenSA", itermax=20)
    assert res["status"] == "optimal"
    assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)
    assert np.all(res["weights"] >= -1e-3)

def test_pso_alias(stocks_data):
    R = stocks_data.iloc[:, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk", name="StdDev")
    
    # Ensure PSO routes correctly (via surrogate)
    res = optimize_portfolio(R, port, optimize_method="PSO", itermax=5)
    assert res["status"] == "optimal"

