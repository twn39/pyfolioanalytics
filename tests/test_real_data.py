import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio, create_efficient_frontier
from pyfolioanalytics.backtest import optimize_portfolio_rebalancing
from pyfolioanalytics.moments import set_portfolio_moments


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
    # ERC: each asset should contribute ~20%. SLSQP with multi-start should be tight.
    np.testing.assert_allclose(pct_rc, np.full(5, 0.2), atol=5e-3)


def test_frontier_on_real_data(stock_returns):
    R = stock_returns
    asset_names = R.columns.tolist()

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    frontier = create_efficient_frontier(R, portfolio, n_portfolios=5)

    assert len(frontier) == 5
    assert "mean" in frontier.columns
    assert "StdDev" in frontier.columns
    
    # 1. Monotonicity checks: As risk increases, expected return must strictly increase
    returns = frontier["mean"].values
    risks = frontier["StdDev"].values
    
    assert np.all(np.diff(returns) > 1e-7), "Frontier returns must be strictly increasing"
    assert np.all(np.diff(risks) > 1e-7), "Frontier risks (StdDev) must be strictly increasing"
    
    # 2. Feasibility check: The sum of weights must equal 1 for every portfolio on the frontier
    weight_cols = [col for col in frontier.columns if col not in ["mean", "sd", "StdDev"]]
    weights = frontier[weight_cols].values
    assert np.allclose(weights.sum(axis=1), 1.0)
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
