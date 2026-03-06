import pandas as pd
import numpy as np
import pytest
import os
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio, create_efficient_frontier
from pyfolioanalytics.backtest import optimize_portfolio_rebalancing
from pyfolioanalytics.moments import set_portfolio_moments


@pytest.fixture
def stock_returns():
    path = "data/stock_returns.csv"
    if not os.path.exists(path):
        pytest.skip("Stock returns data not found.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def test_mvo_on_real_data(stock_returns):
    R = stock_returns
    asset_names = R.columns.tolist()

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    portfolio.add_objective(type="risk", name="StdDev")

    res = optimize_portfolio(R, portfolio, optimize_method="ROI")

    assert res["status"] == "optimal"
    assert len(res["weights"]) == 5
    assert np.abs(np.sum(res["weights"].values) - 1.0) < 1e-7
    assert np.all(res["weights"].values >= -1e-7)


def test_erc_on_real_data(stock_returns):
    R = stock_returns
    asset_names = R.columns.tolist()

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    portfolio.add_objective(type="risk_budget", name="StdDev", min_concentration=True)

    res = optimize_portfolio(R, portfolio)

    assert res["status"] == "optimal"
    pct_rc = res["objective_measures"]["pct_contrib_StdDev"]
    # Check if contributions are nearly equal (1/5 = 0.2)
    np.testing.assert_allclose(pct_rc, np.full(5, 0.2), atol=1e-2)


def test_frontier_on_real_data(stock_returns):
    R = stock_returns
    asset_names = R.columns.tolist()

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    frontier = create_efficient_frontier(R, portfolio, n_portfolios=5)

    assert len(frontier) >= 3
    assert "mean" in frontier.columns
    assert "StdDev" in frontier.columns
    # Check if risk is increasing with return
    assert np.all(np.diff(frontier["StdDev"].values) >= -1e-7)


def test_backtest_on_real_data(stock_returns):
    R = stock_returns.iloc[-200:]  # Last ~10 months
    asset_names = R.columns.tolist()

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    portfolio.add_objective(type="risk", name="StdDev")

    # Quarterly rebalancing
    res = optimize_portfolio_rebalancing(
        R, portfolio, rebalance_on="quarters", training_period=60
    )

    assert len(res.weights) > 0
    assert res.portfolio_returns.shape[0] > 0


def test_black_litterman_on_real_data(stock_returns):
    R = stock_returns
    asset_names = R.columns.tolist()

    portfolio = Portfolio(assets=asset_names)

    # View: AAPL (index 0) will return 5% annualized (approx 0.0002 daily)
    P = np.zeros((1, 5))
    P[0, 0] = 1.0
    q = np.array([[0.0002]])

    moments = set_portfolio_moments(R, portfolio, method="black_litterman", P=P, q=q)

    assert moments["mu"].shape == (5, 1)
    assert moments["sigma"].shape == (5, 5)
    # Posterior mean for AAPL should be influenced by view
    assert moments["mu"][0, 0] != R.iloc[:, 0].mean()
