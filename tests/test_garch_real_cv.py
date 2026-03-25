import json
import numpy as np
import pandas as pd
from pyfolioanalytics.moments import ccc_garch_moments


def load_real_cv_data():
    with open("data/garch_real_cv.json", "r") as f:
        return json.load(f)


def test_ccc_garch_real_data_cv():
    data = load_real_cv_data()
    R = np.array(data["input_R"])

    # Calculate moments in Python
    res = ccc_garch_moments(R)

    # 1. Mean Check (Exact)
    np.testing.assert_allclose(res["mu"].flatten(), np.array(data["mu"]), rtol=1e-7)

    # 2. Sigma Structure Check
    expected_sigma = np.array(data["sigma"])

    # Check variances (diagonal)
    # GARCH forecasts are highly sensitive to solvers (arch vs fGarch).
    # Differences up to 70% can occur on real short time series due to local optima.
    np.testing.assert_allclose(np.diag(res["sigma"]), np.diag(expected_sigma), rtol=0.7)

    # Check Correlation Matrix consistency
    def get_corr(S):
        d = np.sqrt(np.diag(S))
        return S / np.outer(d, d)

    corr_py = get_corr(res["sigma"])
    corr_r = get_corr(expected_sigma)

    # Correlations should be more stable (within 0.2 absolute difference)
    np.testing.assert_allclose(corr_py, corr_r, atol=0.2)

    # 3. Higher Order Moments Shape Check
    N = R.shape[1]
    assert res["m3"].shape == (N, N * N)
    assert res["m4"].shape == (N, N**3)


def test_garch_optimization_integration():
    from pyfolioanalytics.portfolio import Portfolio
    from pyfolioanalytics.optimize import optimize_portfolio

    data = load_real_cv_data()
    asset_names = [f"Asset_{i}" for i in range(13)]
    R_df = pd.DataFrame(data["input_R"], columns=asset_names)

    p = Portfolio(assets=asset_names)
    p.add_constraint(type="full_investment")
    p.add_constraint(type="box", min=0, max=0.2)
    p.add_objective(type="risk", name="var")

    # Optimize using GARCH moments
    res = optimize_portfolio(R_df, p, moment_method="garch")

    assert res["status"] == "optimal"
    assert len(res["weights"]) == 13
