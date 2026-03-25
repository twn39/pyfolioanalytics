import numpy as np
import pandas as pd
import json
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_modern_parity_real_cv():
    # 1. Load Data
    edhec = pd.read_csv("data/edhec.csv")
    with open("data/modern_risk_parity_real_cv.json", "r") as f:
        cv_data = json.load(f)

    asset_names = cv_data["asset_names"]
    R_sub = edhec[asset_names]

    port = Portfolio(assets=asset_names)
    port.add_constraint(type="full_investment")
    # Long-only by default in PortfolioAnalytics if not specified,
    # but we should ensure it matches Riskfolio's default (0, 1)
    port.add_constraint(type="box", min=0.0, max=1.0)

    # 2. Test EDaR CV
    port.add_objective(name="EDaR", type="risk", arguments={"p": 0.95})
    res_edar = optimize_portfolio(R_sub, port)

    assert res_edar["status"] in ["optimal", "feasible"]
    w_edar_py = res_edar["weights"].values
    w_edar_rp = np.array(cv_data["w_edar"])

    # Riskfolio and our implementation use slightly different auxiliary variable setups for drawdowns
    # (absolute vs relative drawdowns etc). Let's check if the weights are close.
    np.testing.assert_allclose(w_edar_py, w_edar_rp, atol=1e-3)

    # 3. Test RLVaR CV
    # Reset objectives
    port.objectives = []
    port.add_objective(name="RLVaR", type="risk", arguments={"p": 0.95, "kappa": 0.3})
    res_rlvar = optimize_portfolio(R_sub, port)

    assert res_rlvar["status"] in ["optimal", "feasible"]
    w_rlvar_py = res_rlvar["weights"].values
    w_rlvar_rp = np.array(cv_data["w_rlvar"])

    np.testing.assert_allclose(w_rlvar_py, w_rlvar_rp, atol=1e-3)
