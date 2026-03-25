import json
import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio


def load_real_cv_data():
    with open("data/factor_exposure_real_cv.json", "r") as f:
        return json.load(f)


def test_factor_exposure_real_data_cv():
    data = load_real_cv_data()
    R_raw = np.array(data["input_R"])
    B = np.array(data["B"])
    lower = np.array(data["lower"])
    upper = np.array(data["upper"])
    expected_weights = np.array(data["weights"])

    T, N = R_raw.shape
    # Assets names in R were from edhec (13 assets)
    # We should match them. Portfolio class by default uses Asset.1, etc. if names not provided.
    # To match R's order and names:
    asset_names = [
        "Convertible Arbitrage",
        "CTA Global",
        "Distressed Securities",
        "Emerging Markets",
        "Equity Market Neutral",
        "Event Driven",
        "Fixed Income Arbitrage",
        "Global Macro",
        "Long/Short Equity",
        "Merger Arbitrage",
        "Relative Value",
        "Short Selling",
        "Funds of Funds",
    ]
    R = pd.DataFrame(R_raw, columns=asset_names)

    # Define Portfolio
    pspec = Portfolio(assets=asset_names)
    pspec.add_constraint(type="full_investment")
    pspec.add_constraint(type="box", min=0, max=0.2)

    # Add Factor Exposure Constraint
    pspec.add_constraint(type="factor_exposure", B=B, lower=lower, upper=upper)

    # Minimize Variance Objective
    pspec.add_objective(type="risk", name="var")

    # Solve
    res = optimize_portfolio(R=R, portfolio=pspec, optimize_method="cvxpy")

    assert res["status"] == "optimal"

    # Parity check on weights (rtol=1e-5 since QP solvers are very stable)
    np.testing.assert_allclose(res["weights"], expected_weights, rtol=1e-5, atol=1e-5)

    # Check exposure actuals
    actual_exposure = B.T @ res["weights"]
    expected_exposure = np.array(data["exposure_actual"])
    np.testing.assert_allclose(actual_exposure, expected_exposure, rtol=1e-6, atol=1e-6)

    # Hard validation against constraints
    assert np.all(actual_exposure >= lower - 1e-9)
    assert np.all(actual_exposure <= upper + 1e-9)
