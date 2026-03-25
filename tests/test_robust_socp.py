import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_socp_return_uncertainty():
    np.random.seed(42)
    n_assets = 5
    asset_names = [f"A{i}" for i in range(n_assets)]
    R = pd.DataFrame(np.random.randn(100, n_assets) * 0.01, columns=asset_names)

    port = Portfolio(assets=asset_names)
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="var", risk_aversion=1.0)

    # 1. Base MVO
    res_base = optimize_portfolio(R, port)

    # 2. Ellipsoidal Robust Return
    sigma_mu = np.diag([0.001**2] * n_assets)
    res_robust = optimize_portfolio(
        R,
        port,
        delta_mu=0.001,
        robust_mu_type="ellipsoidal",
        sigma_mu=sigma_mu,
        k_mu=2.0,
    )

    assert res_robust["status"] in ["optimal", "feasible"]
    # Robust weights should be different from base weights
    assert not np.allclose(res_base["weights"], res_robust["weights"], atol=1e-2)
    assert np.isclose(res_robust["weights"].sum(), 1.0)


def test_socp_covariance_uncertainty():
    np.random.seed(42)
    n_assets = 3
    asset_names = [f"A{i}" for i in range(n_assets)]
    R = pd.DataFrame(np.random.randn(100, n_assets) * 0.01, columns=asset_names)

    port = Portfolio(assets=asset_names)
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="var")

    # Base Min Vol
    res_base = optimize_portfolio(R, port)

    # 2. Ellipsoidal Robust Sigma
    # Uncertainty of vec(Sigma)
    sigma_sigma = np.diag([1e-6] * (n_assets**2))

    res_robust = optimize_portfolio(
        R, port, robust_sigma_type="ellipsoidal", sigma_sigma=sigma_sigma, k_sigma=1.0
    )

    assert res_robust["status"] in ["optimal", "feasible"]
    assert np.isclose(res_robust["weights"].sum(), 1.0)
    # Should produce different weights
    assert not np.allclose(res_base["weights"], res_robust["weights"], atol=1e-3)
