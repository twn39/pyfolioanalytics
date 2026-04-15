import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import UCI

def test_uci_calculation(stocks_data):
    R = stocks_data.iloc[:, :5].values
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Calculate UCI directly
    uci_val = UCI(weights, R)
    assert uci_val > 0
    assert np.isfinite(uci_val)
    
    # Test homogeneous scaling: UCI(c*w) = c * UCI(w)
    uci_val_scaled = UCI(weights * 2.0, R)
    assert np.isclose(uci_val_scaled, uci_val * 2.0)

def test_uci_optimization(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    port.add_objective(type="risk", name="UCI")
    
    res = optimize_portfolio(R, port, optimize_method="ROI")
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    w = res["weights"]
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= -1e-6)
    
def test_martin_ratio_optimization(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="UCI")
    
    # This should trigger the new Charnes-Cooper transformation!
    res = optimize_portfolio(R, port, optimize_method="ROI")
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    w = res["weights"]
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= -1e-6)

