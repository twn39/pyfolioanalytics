import pytest
import numpy as np
import pandas as pd
import json
import os
from pyfolioanalytics.rmt import denoise_covariance
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_rmt_parity_with_julia():
    # Load ground truth generated from Julia/PortfolioOptimisers logic
    json_path = os.path.join("data", "rmt_cv.json")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    sigma_raw = np.array(data["sigma_raw"])
    sigma_ref = np.array(data["sigma_denoised"])
    q = data["q"]
    
    # Run our Python MP Denoising
    sigma_py = denoise_covariance(sigma_raw, q, method="fixed")
    
    # Parity check
    np.testing.assert_allclose(sigma_py, sigma_ref, rtol=1e-7, atol=1e-8)

def test_socp_robust_parity_manual():
    # Since direct SOCP ground truth from other libraries is hard to align
    # due to formulation differences, we use a manual SOCP implementation
    # using CVXPY directly here to verify our solver's internal logic.
    # This is effectively a white-box cross-validation of the formulation.
    
    np.random.seed(42)
    n = 5
    mu = np.random.randn(n) * 0.01
    sigma = np.diag([0.02**2] * n)
    sigma_mu = np.diag([0.005**2] * n) # Uncertainty of mu
    k_mu = 1.5
    
    R = pd.DataFrame(np.random.randn(100, n), columns=[f"A{i}" for i in range(n)])
    port = Portfolio(assets=[f"A{i}" for i in range(n)])
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")
    port.add_objective(type="return", name="mean")
    port.add_objective(type="risk", name="var", risk_aversion=1.0)
    
    # 1. Run our solver
    res_py = optimize_portfolio(R, port, 
                                 delta_mu=0.001, # Trigger robust logic
                                 robust_mu_type="ellipsoidal", 
                                 sigma_mu=sigma_mu, 
                                 k_mu=k_mu)
    
    w_py = res_py["weights"].values
    
    # 2. Manual CVXPY implementation of the same SOCP
    import cvxpy as cp
    w = cp.Variable(n)
    G_mu = np.linalg.cholesky(sigma_mu).T
    
    # Same as solve_mvo robust logic:
    # Minimize 0.5 * risk_aversion * w'Sw - (mu'w - k_mu * ||G_mu * w||_2)
    risk = cp.quad_form(w, sigma)
    ret_robust = mu @ w - k_mu * cp.norm(G_mu @ w)
    obj = cp.Minimize(0.5 * 1.0 * risk - ret_robust)
    
    cons = [cp.sum(w) == 1.0, w >= 0]
    prob = cp.Problem(obj, cons)
    # Use exact same solver moments for manual check
    # But optimize_portfolio re-calculates moments from R
    # So we need to overwrite the R-based moments in our comparison
    # or pass pre-calculated moments.
    
    # Let's use the actual moments from the result
    moments = res_py["moments"]
    mu_act = moments["mu"].flatten()
    sigma_act = moments["sigma"]
    
    risk_act = cp.quad_form(w, sigma_act)
    ret_robust_act = mu_act @ w - k_mu * cp.norm(G_mu @ w)
    prob_act = cp.Problem(cp.Minimize(0.5 * 1.0 * risk_act - ret_robust_act), cons)
    prob_act.solve()
    
    w_manual = w.value
    
    # Verify parity
    np.testing.assert_allclose(w_py, w_manual, atol=1e-6)

def test_socp_covariance_parity_manual():
    # White-box validation of Ellipsoidal Covariance Uncertainty formulation
    # Risk: Tr(Sigma * (W + E)) + k_sigma * sigma_risk
    # s.t. || G_sigma * vec(W + E) ||_2 <= sigma_risk, E >> 0, [W w; w' 1] >> 0
    
    np.random.seed(42)
    n = 3
    mu = np.array([0.01, 0.02, 0.015])
    sigma = np.diag([0.05, 0.08, 0.06])**2
    # Uncertainty of vec(Sigma) - small for stability
    sigma_sigma = np.diag([1e-6] * (n**2))
    k_sigma = 0.5
    
    R = pd.DataFrame(np.random.randn(100, n) * 0.01, columns=[f"A{i}" for i in range(n)])
    port = Portfolio(assets=[f"A{i}" for i in range(n)])
    port.add_constraint(type="long_only")
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="var")
    
    # 1. Run our solver
    res_py = optimize_portfolio(R, port, 
                                 robust_sigma_type="ellipsoidal", 
                                 sigma_sigma=sigma_sigma, 
                                 k_sigma=k_sigma)
    w_py = res_py["weights"].values
    
    # 2. Manual CVXPY implementation
    import cvxpy as cp
    w = cp.Variable(n)
    W = cp.Variable((n, n), symmetric=True)
    E = cp.Variable((n, n), symmetric=True)
    sigma_risk = cp.Variable()
    
    G_sigma = np.linalg.cholesky(sigma_sigma).T
    
    # Use actual solver sigma for parity
    sigma_act = res_py["moments"]["sigma"]
    
    # Robust risk expression
    risk_robust = cp.trace(sigma_act @ (W + E)) + k_sigma * sigma_risk
    
    cons = [
        cp.sum(w) == 1.0,
        w >= 0,
        cp.norm(G_sigma @ cp.vec(W + E, order="C")) <= sigma_risk,
        E >> 0,
        cp.vstack([cp.hstack([W, cp.reshape(w, (n, 1), order="C")]),
                   cp.hstack([cp.reshape(w, (1, n), order="C"), np.array([[1.0]])])]) >> 0
    ]
    
    prob = cp.Problem(cp.Minimize(risk_robust), cons)
    prob.solve(solver=cp.SCS) # Using SCS for SDP/SOCP
    
    w_manual = w.value
    
    # Verify parity
    # Using 1e-3 tolerance as solvers (CLARABEL vs SCS) might have different precision defaults
    np.testing.assert_allclose(w_py, w_manual, atol=1e-3)
