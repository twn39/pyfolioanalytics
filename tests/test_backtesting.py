import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.portfolio import Portfolio, RegimePortfolio
from pyfolioanalytics.backtest import optimize_portfolio_rebalancing


@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
    return df


def test_optimize_portfolio_rebalancing_basics(stocks_data):
    # Test simple monthly rebalancing
    R = stocks_data.iloc[:100]  # Use a subset
    assets = R.columns.tolist()

    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")
    portfolio.add_objective(type="risk", name="StdDev")

    res = optimize_portfolio_rebalancing(
        R, portfolio, optimize_method="ROI", rebalance_on="months", training_period=30
    )

    assert res.weights.shape[0] > 0
    assert np.allclose(res.weights.sum(axis=1), 1.0)
    assert len(res.portfolio_returns) > 0


def test_optimize_portfolio_rebalancing_rolling(stocks_data):
    # Test rolling window
    R = stocks_data.iloc[:120]
    assets = R.columns.tolist()

    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_objective(type="risk", name="StdDev")

    res = optimize_portfolio_rebalancing(
        R,
        portfolio,
        optimize_method="ROI",
        rebalance_on="months",
        training_period=60,
        rolling_window=60,
    )

    assert res.weights.shape[0] > 0
    # Every optimization should use 60 days
    for opt in res.opt_results:
        # Check training data size in moments
        assert opt["moments"]["mu"].shape[0] == 5


def test_regime_portfolio_backtest(stocks_data):
    R = stocks_data.iloc[:200]
    assets = R.columns.tolist()

    # Regime 1: Min StdDev
    p1 = Portfolio(assets=assets)
    p1.add_objective(type="risk", name="StdDev")

    # Regime 2: Max Return
    p2 = Portfolio(assets=assets)
    p2.add_objective(type="return", name="mean")

    reg_port = RegimePortfolio(portfolios=[p1, p2], regime_labels=[0, 1])

    # Create a dummy regime signal (0 for first half, 1 for second half)
    regimes = pd.Series(0, index=R.index)
    regimes.iloc[len(regimes) // 2 :] = 1

    res = optimize_portfolio_rebalancing(
        R,
        reg_port,
        optimize_method="ROI",
        rebalance_on="months",
        training_period=60,
        regimes=regimes,
    )

    assert res.weights.shape[0] > 0
    # The objectives in results should change according to regime
    # First rebalance (month 3, 4 etc) should be from p1
    # Later should be from p2
    found_stddev = False
    found_mean = False
    for opt in res.opt_results:
        obs = [o["name"] for o in opt["portfolio"].objectives]
        if "StdDev" in obs:
            found_stddev = True
        if "mean" in obs:
            found_mean = True

    assert found_stddev
    assert found_mean
