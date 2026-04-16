import json
import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_ptc_cross_validation():
    with open("data/ptc_cv.json", "r") as f:
        cv_data = json.load(f)

    R_data = np.array(cv_data["returns"])
    asset_names = [f"A{i + 1}" for i in range(R_data.shape[1])]
    R = pd.DataFrame(R_data, columns=asset_names)

    w_init = np.array(cv_data["w_init"])
    ptc_val = cv_data["ptc"]
    if isinstance(ptc_val, list):
        ptc_val = ptc_val[0]
    r_weights = np.array(cv_data["opt_weights"])

    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint("full_investment")
    portfolio.add_constraint("long_only")
    # Add PTC as transaction_cost type
    portfolio.add_constraint("transaction_cost", ptc=ptc_val, weight_initial=w_init)

    # PortfolioAnalytics R script used Mean and StdDev objectives with risk_aversion=2
    portfolio.add_objective(type="return", name="mean")
    portfolio.add_objective(type="risk", name="var", risk_aversion=2.0)

    res_py = optimize_portfolio(R, portfolio)
    py_weights = res_py["weights"].values

    # Check if weights are close to R's solver results
    # Our solver uses: Minimize( risk_aversion * risk_term - (w @ mu) + tc_penalty )
    # which exactly matches R's standard quadratic utility formulation (no 0.5 factor).
    np.testing.assert_allclose(py_weights, r_weights, atol=1e-4)

    # Verify transaction cost calculation in objective_measures
    expected_tc = np.sum(np.abs(py_weights - w_init) * ptc_val)
    assert "transaction_cost" in res_py["objective_measures"]
    np.testing.assert_allclose(
        res_py["objective_measures"]["transaction_cost"], expected_tc, rtol=1e-7
    )
