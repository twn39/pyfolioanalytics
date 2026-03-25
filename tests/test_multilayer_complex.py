import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio, MultLayerPortfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_multilayer_with_raw_assets_mixed():
    dates = pd.date_range("2020-01-01", periods=100)
    data = np.random.normal(0.001, 0.01, (100, 4))
    R = pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])

    # Sub-portfolio for A and B
    port_ab = Portfolio(assets=["A", "B"])
    port_ab.add_constraint(type="full_investment")
    port_ab.add_objective(type="risk", name="StdDev")

    # Top portfolio uses Sub_AB and raw assets C, D
    port_top = Portfolio(assets=["Sub_AB", "C", "D"])
    port_top.add_constraint(type="full_investment")
    port_top.add_constraint(type="box", min=0.1, max=0.5)
    port_top.add_objective(type="return", name="mean")

    multi_port = MultLayerPortfolio(root_portfolio=port_top)
    multi_port.add_sub_portfolio("Sub_AB", port_ab)

    res = optimize_portfolio(R, multi_port, optimize_method="ROI")
    assert res["status"] in ["optimal", "feasible"]

    w = res["weights"]
    assert len(w) == 4
    assert np.isclose(w.sum(), 1.0)

    # C and D should respect the box constraints from port_top (0.1 to 0.5)
    assert w["C"] >= 0.1 - 1e-4
    assert w["C"] <= 0.5 + 1e-4
    assert w["D"] >= 0.1 - 1e-4
    assert w["D"] <= 0.5 + 1e-4
