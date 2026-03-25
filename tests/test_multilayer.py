import numpy as np
from pyfolioanalytics.portfolio import Portfolio, MultLayerPortfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_multilayer_optimization(stocks_data):
    R = stocks_data.iloc[:, :5]  # Use 5 assets
    assets = list(R.columns)
    
    # Define sub-portfolio 1: Tech
    tech_assets = assets[:3]
    port_tech = Portfolio(assets=tech_assets)
    port_tech.add_constraint(type="full_investment")
    port_tech.add_objective(type="risk", name="StdDev")
    
    # Define sub-portfolio 2: Consumer
    cons_assets = assets[3:]
    port_cons = Portfolio(assets=cons_assets)
    port_cons.add_constraint(type="full_investment")
    port_cons.add_objective(type="return", name="mean")
    
    # Define Top-Level Portfolio
    port_top = Portfolio(assets=["TechSector", "ConsumerSector"])
    port_top.add_constraint(type="full_investment")
    # Say we want minimum variance at the top level between the two sectors
    port_top.add_objective(type="risk", name="StdDev")
    
    # Create Multilayer structure
    multi_port = MultLayerPortfolio(root_portfolio=port_top)
    multi_port.add_sub_portfolio("TechSector", port_tech)
    multi_port.add_sub_portfolio("ConsumerSector", port_cons)
    
    # Optimize
    res = optimize_portfolio(R, multi_port, optimize_method="ROI")
    
    assert res["status"] == "optimal"
    weights = res["weights"]
    
    # Final weights should be size of all original assets
    assert len(weights) == 5
    assert set(weights.index) == set(assets)
    
    # Sum of all weights should be 1.0 (since top level is full_investment)
    assert np.isclose(weights.sum(), 1.0)
    
    # Check that it returns sub results
    assert "sub_results" in res
    assert "TechSector" in res["sub_results"]
    assert "ConsumerSector" in res["sub_results"]

