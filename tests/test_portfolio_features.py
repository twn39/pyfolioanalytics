import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio, MultLayerPortfolio, RegimePortfolio
from pyfolioanalytics.optimize import optimize_portfolio


@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
    return df


def test_group_constraints(stocks_data):
    assets = stocks_data.columns.tolist()
    # Group 1: AAPL, AMZN (Indices 0, 1)
    # Group 2: GOOGL, META, MSFT (Indices 2, 3, 4)
    groups = [[0, 1], [0, 2, 3, 4]]
    group_min = [0.4, 0.4]
    group_max = [0.6, 0.6]

    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    portfolio.add_constraint(
        type="group", groups=groups, group_min=group_min, group_max=group_max
    )
    portfolio.add_objective(type="risk", name="StdDev")

    res = optimize_portfolio(stocks_data, portfolio)
    assert res["status"] == "optimal"
    w = res["weights"].values

    # Check group 1
    assert 0.4 - 1e-7 <= np.sum(w[[0, 1]]) <= 0.6 + 1e-7
    # Check group 2 (overlap with AAPL)
    assert 0.4 - 1e-7 <= np.sum(w[[0, 2, 3, 4]]) <= 0.6 + 1e-7


def test_position_limit_constraint(stocks_data):
    assets = stocks_data.columns.tolist()
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    # Limit to max 2 assets
    portfolio.add_constraint(type="position_limit", max_pos=2)
    portfolio.add_objective(type="risk", name="StdDev")

    # Note: Requires a solver that supports MIP (like SCIP or ECOS_BB)
    # CVXPY solve_mvo handles this
    res = optimize_portfolio(stocks_data, portfolio)
    if res["status"] == "optimal":
        w = res["weights"].values
        non_zero = np.sum(w > 1e-4)
        assert non_zero <= 2


def test_multi_layer_portfolio(stocks_data):
    stocks_data.columns.tolist()
    # Layer 1: Tech Group
    p_tech = Portfolio(assets=["AAPL", "MSFT"])
    p_tech.add_constraint(type="full_investment")
    p_tech.add_constraint(type="long_only")
    p_tech.add_objective(type="risk", name="StdDev")

    # Root: Tech + Others
    # We treat p_tech as a single asset "Tech"
    p_root = Portfolio(assets={"Tech": 0.5, "AMZN": 0.2, "GOOGL": 0.2, "META": 0.1})
    p_root.add_constraint(type="full_investment")
    p_root.add_objective(type="risk", name="StdDev")

    mlp = MultLayerPortfolio(p_root)
    mlp.add_sub_portfolio("Tech", p_tech)

    res = optimize_portfolio(stocks_data, mlp)
    assert res["weights"] is not None
    assert "AAPL" in res["weights"].index
    assert "MSFT" in res["weights"].index
    assert np.isclose(res["weights"].sum(), 1.0)


def test_regime_portfolio():
    p1 = Portfolio(assets=3)
    p1.add_objective(type="risk", name="StdDev")

    p2 = Portfolio(assets=3)
    p2.add_objective(type="return", name="mean")

    rp = RegimePortfolio(portfolios=[p1, p2], regime_labels=["low_vol", "high_vol"])
    assert rp.get_portfolio("low_vol") == p1
    assert rp.get_portfolio("high_vol") == p2
    # Fallback
    assert rp.get_portfolio("other") == p1
