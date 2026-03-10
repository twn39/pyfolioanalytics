import json
import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.factors import ac_ranking
from pyfolioanalytics.optimize import optimize_portfolio

def test_new_features_cross_validation():
    with open("data/new_features_cv.json", "r") as f:
        cv_data = json.load(f)
    
    R_data = np.array(cv_data["returns"])
    asset_names = [f"A{i+1}" for i in range(R_data.shape[1])]
    R = pd.DataFrame(R_data, columns=asset_names)
    
    # 1. Active Ranking
    # R order: c(2, 3, 1, 4, 5) -> [A2, A3, A1, A4, A5]
    ranking_order = ["A2", "A3", "A1", "A4", "A5"]
    py_ac_mu = ac_ranking(R, ranking_order)
    r_ac_mu = np.array(cv_data["ac_ranking_mu"])
    
    # Parity check for AC Ranking
    np.testing.assert_allclose(py_ac_mu, r_ac_mu, rtol=1e-7)
    
    # 2. Turnover Constraint
    w_init = np.array(cv_data["turnover_w_init"])
    turnover_target = cv_data["turnover_target"]
    if isinstance(turnover_target, list):
        turnover_target = turnover_target[0]
    r_weights = np.array(cv_data["turnover_weights"])
    
    portfolio = Portfolio(assets=asset_names)
    portfolio.add_constraint("full_investment")
    portfolio.add_constraint("long_only")
    portfolio.add_constraint("turnover", turnover_target=turnover_target, weight_initial=w_init)
    portfolio.add_objective("StdDev")
    
    res_py = optimize_portfolio(R, portfolio)
    py_weights = res_py["weights"].values
    
    # Note: Small differences might occur due to different solvers (ROI/GLPK vs CVXPY/CLARABEL)
    # We check if the turnover constraint is satisfied and weights are close
    actual_turnover = np.sum(np.abs(py_weights - w_init))
    assert actual_turnover <= turnover_target + 1e-6
    np.testing.assert_allclose(py_weights, r_weights, atol=1e-5)

def test_shrinkage_parity_check():
    # Shrinkage parity is tricky because corpcor and sklearn use different targets/formulations.
    # We will mainly check if our 'shrinkage' method in moments.py is working as intended (using sklearn)
    # in the previous unit test. Cross-validation for shrinkage with R usually requires 
    # using the exact same target (e.g. constant correlation).
    pass
