import pandas as pd
import numpy as np
import os
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio


def test_noc_on_real_data():
    # Load EDHEC data
    data_path = os.path.join(os.path.dirname(__file__), "../data/edhec.csv")
    df = pd.read_csv(data_path, index_col=0)
    R = df.iloc[:, :10]

    # Create Portfolio
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")

    # Standard MVO (Min Risk)
    port.add_objective(type="risk", name="StdDev")
    res_mvo = optimize_portfolio(R, port, optimize_method="ROI")
    assert res_mvo["status"] == "optimal"

    # NOC optimization
    # bins default to N/T, but let's use a fixed value for stability check
    res_noc = optimize_portfolio(R, port, optimize_method="NOC", bins=20.0)
    assert res_noc["status"] in ["optimal", "optimal_inaccurate"]

    w_mvo = res_mvo["weights"]
    w_noc = res_noc["weights"]

    print(f"\nMVO weights sum: {w_mvo.sum()}")
    print(f"NOC weights sum: {w_noc.sum()}")

    # Centering check: NOC should have more "even" weights (higher entropy / less zeros potentially)
    # Most MVO solutions are sparse. NOC should be less sparse.
    mvo_zeros = np.sum(w_mvo < 1e-4)
    noc_zeros = np.sum(w_noc < 1e-4)

    print(f"MVO zero weights: {mvo_zeros}")
    print(f"NOC zero weights: {noc_zeros}")

    # In MVO, often many assets are 0. NOC log-barrier pushes them away from 0.
    assert noc_zeros <= mvo_zeros

    # Check risk/return
    mu = R.mean().values
    sigma = R.cov().values

    rk_mvo = w_mvo.values @ sigma @ w_mvo.values
    rt_mvo = w_mvo.values @ mu

    rk_noc = w_noc.values @ sigma @ w_noc.values
    rt_noc = w_noc.values @ mu

    print(f"MVO: Risk={rk_mvo:.6f}, Ret={rt_mvo:.6f}")
    print(f"NOC: Risk={rk_noc:.6f}, Ret={rt_noc:.6f}")

    # NOC risk should be slightly higher than MVO min risk (since it's in the near-optimal region)
    assert rk_noc >= rk_mvo - 1e-9
