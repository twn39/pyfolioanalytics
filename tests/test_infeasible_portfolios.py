import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_infeasible_mvo():
    R = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="weight_sum", min_sum=2.0, max_sum=1.0)
    port.add_objective(type="risk", name="var")
    res = optimize_portfolio(R, port)
    assert res["status"] in ["infeasible", "failed"]

def test_infeasible_mad():
    R = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="weight_sum", min_sum=2.0, max_sum=1.0)
    port.add_objective(type="risk", name="MAD")
    res = optimize_portfolio(R, port)
    assert res["status"] in ["infeasible", "failed"]

def test_infeasible_rlvar():
    R = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="weight_sum", min_sum=2.0, max_sum=1.0)
    port.add_objective(type="risk", name="RLVaR")
    res = optimize_portfolio(R, port)
    assert res["status"] in ["infeasible", "failed"]

def test_infeasible_evar():
    R = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="weight_sum", min_sum=2.0, max_sum=1.0)
    port.add_objective(type="risk", name="EVaR")
    res = optimize_portfolio(R, port)
    assert res["status"] in ["infeasible", "failed"]

def test_infeasible_owa():
    R = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="weight_sum", min_sum=2.0, max_sum=1.0)
    
    from pyfolioanalytics.risk import owa_gmd_weights
    port.add_objective(type="risk", name="OWA", arguments={"owa_weights": owa_gmd_weights(100)})
    res = optimize_portfolio(R, port)
    assert res["status"] in ["infeasible", "failed"]

def test_infeasible_kelly():
    R = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    port = Portfolio(assets=["A", "B", "C"])
    port.add_constraint(type="weight_sum", min_sum=2.0, max_sum=1.0)
    port.add_objective(type="return", name="Kelly")
    res = optimize_portfolio(R, port, optimize_method="Kelly")
    assert res["status"] in ["infeasible", "failed"]

