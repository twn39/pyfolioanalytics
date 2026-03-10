import numpy as np
import pandas as pd
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.cla import CLA
from pyfolioanalytics.moments import set_portfolio_moments


def test_cla_min_vol():
    # Setup some test data
    np.random.seed(42)
    n_assets = 5
    returns = np.random.randn(100, n_assets) * 0.01 + 0.001
    R = pd.DataFrame(returns, columns=[f"Asset.{i + 1}" for i in range(n_assets)])

    port = Portfolio(assets=n_assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=1.0)
    port.add_objective(type="risk", name="StdDev")

    moments = set_portfolio_moments(R, port)

    # Solve with standard MVO (CVXPY)
    res_mvo = optimize_portfolio(R, port, optimize_method="ROI")

    # Solve with CLA
    res_cla = optimize_portfolio(R, port, optimize_method="CLA")

    sigma = moments["sigma"]
    w_mvo = res_mvo["weights"].values
    w_cla = res_cla["weights"].values

    var_mvo = w_mvo.T @ sigma @ w_mvo
    var_cla = w_cla.T @ sigma @ w_cla

    assert res_cla["status"] == "optimal"
    assert np.isclose(var_mvo, var_cla, atol=1e-6)


def test_cla_max_sharpe():
    np.random.seed(42)
    n_assets = 5
    returns = np.random.randn(100, n_assets) * 0.01 + 0.001
    returns[:, 0] += 0.005
    R = pd.DataFrame(returns, columns=[f"Asset.{i + 1}" for i in range(n_assets)])

    port = Portfolio(assets=n_assets)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=0.5)
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="var", risk_aversion=1.0)

    res_cla = optimize_portfolio(R, port, optimize_method="CLA")
    assert res_cla["status"] == "optimal"


def test_cla_frontier():
    np.random.seed(42)
    n_assets = 4
    mu = np.array([0.01, 0.02, 0.015, 0.012])
    sigma = np.diag([0.05, 0.08, 0.06, 0.07]) ** 2
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets)

    cla = CLA(mu, sigma, lb, ub)
    cla.solve()
    mu_f, sigma_f, weights_f = cla.efficient_frontier(points=50)
    assert len(mu_f) >= 1


def test_cla_non_psd_raises():
    """CLA must raise ValueError when given a non-PSD covariance matrix."""
    n = 3
    mu = np.array([0.01, 0.02, 0.015])
    # Construct a definitly-negative definite matrix
    bad_cov = -np.eye(n)
    lb = np.zeros(n)
    ub = np.ones(n)

    with pytest.raises(ValueError, match="not positive semi-definite"):
        CLA(mu, bad_cov, lb, ub)
