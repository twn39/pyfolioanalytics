import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_charnes_cooper_sharpe(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    # Enable both return (maximize) and risk (minimize), which triggers max_ratio default (Sharpe)
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="StdDev")

    res = optimize_portfolio(R, port, optimize_method="ROI", max_ratio=True)
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    weights = res["weights"]
    
    # Verify sum = 1
    assert np.isclose(np.sum(weights), 1.0)
    
    # Compare with CLA max sharpe
    res_cla = optimize_portfolio(R, port, optimize_method="CLA")
    weights_cla = res_cla["weights"]
    assert np.allclose(weights, weights_cla, atol=1e-3)

def test_charnes_cooper_starr(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    # STARR: maximize return / ES
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="ES", arguments={"p": 0.95})
    
    res = optimize_portfolio(R, port, optimize_method="ROI")
    assert res["status"] in ["optimal", "optimal_inaccurate"]
    weights = res["weights"]
    
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= -1e-6)

