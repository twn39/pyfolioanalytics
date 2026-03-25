import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_random_portfolios_optimization(stocks_data):
    # Setup simple data
    R = stocks_data.iloc[:, :4]
    
    # Create portfolio
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="weight_sum", min_sum=0.99, max_sum=1.01)
    port.add_constraint(type="box", min=0.05, max=0.6)
    
    # Add an objective that random search should optimize
    port.add_objective(type="risk", name="StdDev")
    port.add_objective(type="return", name="mean")
    
    # Test random optimization using transform method
    res = optimize_portfolio(
        R, port, optimize_method="random", rp_method="transform", permutations=500
    )
    
    assert res["status"] == "optimal"
    assert res["weights"] is not None
    weights = res["weights"]
    
    # Check constraints rigorously
    assert np.all(weights >= 0.05 - 1e-5), f"Min weight violation: {weights.min()}"
    assert np.all(weights <= 0.6 + 1e-5), f"Max weight violation: {weights.max()}"
    assert abs(weights.sum() - 1.0) <= 0.01 + 1e-5, f"Sum violation: {weights.sum()}"
    
    # Check if optimization is better than naive equal weight
    eq_w = np.full(4, 0.25)
    from pyfolioanalytics.optimize import calculate_objective_measures
    eq_meas = calculate_objective_measures(eq_w, res["moments"], port.objectives)
    
    # Usually random search with StdDev objective should find a portfolio with lower risk than equal weight
    assert res["objective_measures"]["StdDev"] <= eq_meas["StdDev"] + 1e-4
    
    # Check objectives are computed
    assert "StdDev" in res["objective_measures"]
    assert "mean" in res["objective_measures"]

