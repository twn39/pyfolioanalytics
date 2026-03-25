import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio, MultLayerPortfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_three_level_multilayer():
    dates = pd.date_range("2020-01-01", periods=100)
    data = np.random.normal(0.001, 0.01, (100, 3))
    R = pd.DataFrame(data, index=dates, columns=["A", "B", "C"])

    # L3: Bottom
    p_a = Portfolio(assets=["A"])
    p_a.add_constraint("full_investment")
    p_a.add_objective("risk", name="StdDev")

    # L2: Middle
    p_mid = Portfolio(assets=["Meta_A", "B"])
    p_mid.add_constraint("full_investment")
    p_mid.add_objective("risk", name="StdDev")
    multi_mid = MultLayerPortfolio(p_mid)
    multi_mid.add_sub_portfolio("Meta_A", p_a)

    # L1: Top
    p_top = Portfolio(assets=["Meta_Mid", "C"])
    p_top.add_constraint("full_investment")
    p_top.add_objective("return", name="mean")
    multi_top = MultLayerPortfolio(p_top)
    multi_top.add_sub_portfolio("Meta_Mid", multi_mid)

    res = optimize_portfolio(R, multi_top)
    assert res["status"] in ["optimal", "feasible"]
    assert len(res["weights"]) == 3
    assert np.isclose(res["weights"].sum(), 1.0)
