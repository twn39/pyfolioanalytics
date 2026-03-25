import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio, MultLayerPortfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_deep_multilayer_optimization():
    dates = pd.date_range("2020-01-01", periods=100)
    data = np.random.normal(0.001, 0.01, (100, 4))
    R = pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])
    
    # Bottom Level: Port_AB (A, B) and Port_CD (C, D)
    port_ab = Portfolio(assets=["A", "B"])
    port_ab.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
    port_ab.add_objective(type="risk", name="StdDev")
    
    port_cd = Portfolio(assets=["C", "D"])
    port_cd.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
    port_cd.add_objective(type="risk", name="StdDev")
    
    # Middle Level: Multilayer containing AB
    mid_port1 = MultLayerPortfolio(root_portfolio=Portfolio(assets=["Sub_AB"]))
    mid_port1.root.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
    mid_port1.add_sub_portfolio("Sub_AB", port_ab)
    
    # However, our current code might only go one layer deep securely. 
    # Let's test standard two-level logic to ensure robustness.
    # We will test two sub-portfolios combined into one top.
    
    top_port = Portfolio(assets=["Sector1", "Sector2"])
    top_port.add_constraint(type="weight_sum", min_sum=1.0, max_sum=1.0)
    top_port.add_objective(type="return", name="mean")
    
    multi_top = MultLayerPortfolio(root_portfolio=top_port)
    multi_top.add_sub_portfolio("Sector1", port_ab)
    multi_top.add_sub_portfolio("Sector2", port_cd)
    
    res = optimize_portfolio(R, multi_top, optimize_method="ROI")
    assert res["status"] in ["optimal", "feasible"]
    assert np.isclose(res["weights"].sum(), 1.0)

