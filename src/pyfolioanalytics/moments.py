import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .portfolio import Portfolio
from .factors import statistical_factor_model, factor_model_covariance

def M3_MM(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    M3 = np.zeros((N, N * N))
    for t in range(T):
        rt = R[t, :].reshape(-1, 1)
        M3 += np.dot(rt, np.kron(rt.T, rt.T))
    return M3 / T

def M4_MM(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    M4 = np.zeros((N, N**3))
    for t in range(T):
        rt = R[t, :].reshape(-1, 1)
        M4 += np.dot(rt, np.kron(rt.T, np.kron(rt.T, rt.T)))
    return M4 / T

def set_portfolio_moments(
    R: pd.DataFrame,
    portfolio: Portfolio,
    method: str = "sample",
    **kwargs
) -> Dict[str, Any]:
    moments = {}
    asset_names = list(portfolio.assets.keys())
    R_filtered = R[asset_names]
    
    if method == "sample":
        moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        moments["sigma"] = R_filtered.cov().values
    elif method == "factor_model":
        k = kwargs.get("k", 3)
        fm = statistical_factor_model(R_filtered, k=k)
        moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        moments["sigma"] = factor_model_covariance(fm)
    elif method == "black_litterman":
        from .black_litterman import black_litterman
        sigma = R_filtered.cov().values
        w_mkt = kwargs.get("w_mkt", np.full((len(asset_names), 1), 1.0 / len(asset_names)))
        P = kwargs.get("P")
        q = kwargs.get("q")
        tau = kwargs.get("tau", 0.05)
        risk_aversion = kwargs.get("risk_aversion", 2.5)
        res_bl = black_litterman(sigma, w_mkt, P, q, tau, risk_aversion)
        moments["mu"] = res_bl["mu"]
        moments["sigma"] = res_bl["sigma"]
    elif method == "shrinkage":
        from sklearn.covariance import LedoitWolf
        moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        lw = LedoitWolf().fit(R_filtered.values)
        moments["sigma"] = lw.covariance_
    elif method == "meucci":
        from .meucci import entropy_pooling, meucci_moments
        T, N = R_filtered.shape
        prior_probs = kwargs.get("prior_probs", np.full(T, 1.0 / T))
        Aeq = kwargs.get("Aeq")
        beq = kwargs.get("beq")
        # Find posterior probabilities
        p = entropy_pooling(prior_probs, Aeq=Aeq, beq=beq)
        # Calculate moments
        res_m = meucci_moments(R_filtered.values, p)
        moments["mu"] = res_m["mu"]
        moments["sigma"] = res_m["sigma"]
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")
        
    # Check if higher order moments are needed
    needs_m3_m4 = any(obj["name"] in ["VaR", "ES", "mVaR", "mES"] 
                     for obj in portfolio.objectives if obj.get("enabled", True))
    if needs_m3_m4:
        R_centered = R_filtered.values - moments["mu"].T
        moments["m3"] = M3_MM(R_centered)
        moments["m4"] = M4_MM(R_centered)
        
    return moments
