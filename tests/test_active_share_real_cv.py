import numpy as np
import pandas as pd
import json
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_active_share_real_data_cv():
    # 1. Load Data
    edhec = pd.read_csv("data/edhec.csv")
    with open("data/active_share_real_cv.json", "r") as f:
        cv_data = json.load(f)

    asset_names = cv_data["asset_names"]
    R_sub = edhec[asset_names]

    # 2. Setup pyfolioanalytics Portfolio
    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=0.0, max=1.0)

    benchmark = dict(zip(asset_names, cv_data["w_b"]))
    port.add_constraint(
        type="active_share", target=cv_data["as_target"], benchmark=benchmark
    )

    # Objective: Maximize return
    port.add_objective(type="return")

    res = optimize_portfolio(R_sub, port)

    assert res["status"] in ["optimal", "feasible"]
    w_py = res["weights"].values
    w_ref = np.array(cv_data["w_optimal"])

    # Compare weights
    np.testing.assert_allclose(w_py, w_ref, atol=1e-6)

    # Verify Active Share value
    w_b = np.array(cv_data["w_b"])
    actual_as = 0.5 * np.sum(np.abs(w_py - w_b))
    assert actual_as <= cv_data["as_target"] + 1e-8
