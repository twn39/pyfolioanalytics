import pandas as pd
import numpy as np
from typing import Dict, Any
from .portfolio import Portfolio
from .factors import statistical_factor_model, factor_model_covariance, ac_ranking


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


def M3_SFM(R: pd.DataFrame, k: int = 1) -> np.ndarray:
    """
    Coskewness matrix estimate via Statistical Factor Model.
    """
    from .factors import statistical_factor_model
    fm = statistical_factor_model(R, k=k)
    B = fm["loadings"].values
    f = fm["factors"].values
    res = fm["residuals"].values
    T, N = R.shape
    
    # Factor M3
    f_centered = f - np.mean(f, axis=0)
    M3_f = M3_MM(f_centered)
    
    # Residual M3 (diagonal-like)
    stockM3 = np.sum(res**3, axis=0) / (T - k - 1)
    
    # S = B * M3_f * (B.T kron B.T)
    Bt = B.T
    S = (B @ M3_f) @ np.kron(Bt, Bt)
    
    # D residual matrix (N x N^2)
    D = np.zeros((N, N**2))
    for i in range(N):
        D[i, i * N + i] = stockM3[i]
        
    return S + D


def M4_SFM(R: pd.DataFrame, k: int = 1) -> np.ndarray:
    """
    Cokurtosis matrix estimate via Statistical Factor Model.
    """
    from .factors import statistical_factor_model
    fm = statistical_factor_model(R, k=k)
    B = fm["loadings"].values
    f = fm["factors"].values
    res = fm["residuals"].values
    T, N = R.shape
    
    # Factor M4
    f_centered = f - np.mean(f, axis=0)
    M4_f = M4_MM(f_centered)
    
    # Factor M2 (Covariance)
    # R's cov(f) uses T-1
    f2 = np.cov(f, rowvar=False).reshape(k, k)
    
    # Residual moments
    stockM2 = np.sum(res**2, axis=0) / (T - k - 1)
    stockM4 = np.sum(res**4, axis=0) / (T - k - 1)
    
    # S = B * M4_f * (B.T kron B.T kron B.T)
    Bt = B.T
    S = (B @ M4_f) @ np.kron(Bt, np.kron(Bt, Bt))
    
    # D residual matrix (N x N^3)
    # This is complex in MF. For SFM k=1 it's easier.
    # In PA, it calls a C routine. We'll implement the structured residual part.
    # D residual matrix (N x N^3)
    D = np.zeros((N, N**3))

    # Full Kronecker residual terms for SFM (k=1)
    if k == 1:
        # Match residualcokurtosisSF from PortfolioAnalytics
        b = B.flatten()
        f2_val = f2.item()
        s2 = stockM2
        s4 = stockM4

        # We need to fill D[l, i*N*N + j*N + k] = kijkl
        # Following the C code's logic:
        for i in range(N):
            for j in range(N):
                for k_idx in range(N):
                    for l in range(N):
                        kijkl = 0.0
                        if (i == j) or (i == k_idx) or (i == l) or (j == k_idx) or (j == l) or (k_idx == l):
                            if (i == j) and (i == k_idx) and (i == l):
                                kijkl = 6 * b[i] * b[i] * f2_val * s2[i] + s4[i]
                            elif ((i == j) and (i == k_idx)) or ((i == j) and (i == l)) or ((i == k_idx) and (i == l)) or ((j == k_idx) and (j == l)):
                                if (i == j) and (i == k_idx):
                                    kijkl = 3 * b[i] * b[l] * f2_val * s2[i]
                                elif (i == j) and (i == l):
                                    kijkl = 3 * b[i] * b[k_idx] * f2_val * s2[i]
                                elif (i == k_idx) and (i == l):
                                    kijkl = 3 * b[i] * b[j] * f2_val * s2[i]
                                elif (j == k_idx) and (j == l):
                                    kijkl = 3 * b[j] * b[i] * f2_val * s2[j]
                            elif ((i == j) and (k_idx == l)) or ((i == k_idx) and (j == l)) or ((i == l) and (j == k_idx)):
                                if (i == j) and (k_idx == l):
                                    kijkl = b[i] * b[i] * f2_val * s2[k_idx] + b[k_idx] * b[k_idx] * f2_val * s2[i] + s2[i] * s2[k_idx]
                                elif (i == k_idx) and (j == l):
                                    kijkl = b[i] * b[i] * f2_val * s2[j] + b[j] * b[j] * f2_val * s2[i] + s2[i] * s2[j]
                                elif (i == l) and (j == k_idx):
                                    kijkl = b[i] * b[i] * f2_val * s2[j] + b[j] * b[j] * f2_val * s2[i] + s2[i] * s2[j]
                            else:
                                if i == j:
                                    kijkl = b[k_idx] * b[l] * f2_val * s2[i]
                                elif i == k_idx:
                                    kijkl = b[j] * b[l] * f2_val * s2[i]
                                elif i == l:
                                    kijkl = b[j] * b[k_idx] * f2_val * s2[i]
                                elif j == k_idx:
                                    kijkl = b[i] * b[l] * f2_val * s2[j]
                                elif j == l:
                                    kijkl = b[i] * b[k_idx] * f2_val * s2[j]
                                elif k_idx == l:
                                    kijkl = b[i] * b[j] * f2_val * s2[k_idx]

                        D[l, i * N * N + j * N + k_idx] = kijkl
    else:

        # Multi-factor residual approximation
        for i in range(N):
            D[i, i * N**2 + i * N + i] = stockM4[i]
            
    return S + D


def shrink_comoments(M_sample: np.ndarray, M_target: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (1 - alpha) * M_sample + alpha * M_target


def set_portfolio_moments(
    R: pd.DataFrame, portfolio: Any, method: str = "sample", **kwargs
) -> Dict[str, Any]:
    # Handle Multi-Layer Portfolio
    if hasattr(portfolio, "root"):
        portfolio = portfolio.root

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
    elif method == "ac_ranking":
        order = kwargs.get("order")
        if order is None:
            raise ValueError("Method 'ac_ranking' requires an 'order' argument.")
        moments["mu"] = ac_ranking(R_filtered, order).reshape(-1, 1)
        moments["sigma"] = R_filtered.cov().values
    elif method == "black_litterman":
        from .black_litterman import black_litterman

        sigma = R_filtered.cov().values
        w_mkt = kwargs.get(
            "w_mkt", np.full((len(asset_names), 1), 1.0 / len(asset_names))
        )
        from typing import cast
        P = cast(np.ndarray, kwargs.get("P"))
        q = cast(np.ndarray, kwargs.get("q"))
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
    elif method == "robust":
        from sklearn.covariance import MinCovDet

        mcd = MinCovDet(random_state=42).fit(R_filtered.values)
        moments["mu"] = mcd.location_.reshape(-1, 1)
        moments["sigma"] = mcd.covariance_
    elif method == "denoised":
        from .rmt import denoise_covariance

        T, N = R_filtered.shape
        q = T / N
        sigma = R_filtered.cov().values
        denoised_sigma = denoise_covariance(
            sigma, q, method=kwargs.get("denoise_method", "fixed")
        )
        moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        moments["sigma"] = denoised_sigma
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

    # Check if higher order moments are needed
    needs_m3_m4 = any(
        obj["name"] in ["VaR", "ES", "mVaR", "mES"]
        for obj in portfolio.objectives
        if obj.get("enabled", True)
    )
    if needs_m3_m4:
        R_centered = R_filtered.values - moments["mu"].T
        
        # Determine calculation method for M3/M4
        comoment_method = kwargs.get("comoment_method", "sample")
        alpha = kwargs.get("comoment_alpha", 0.0)
        k_factors = kwargs.get("k", 1)
        
        if comoment_method == "sample":
            moments["m3"] = M3_MM(R_centered)
            moments["m4"] = M4_MM(R_centered)
        elif comoment_method == "factor_model":
            moments["m3"] = M3_SFM(R_filtered, k=k_factors)
            moments["m4"] = M4_SFM(R_filtered, k=k_factors)
        elif comoment_method == "shrinkage":
            m3_sample = M3_MM(R_centered)
            m4_sample = M4_MM(R_centered)
            m3_target = M3_SFM(R_filtered, k=k_factors)
            m4_target = M4_SFM(R_filtered, k=k_factors)
            moments["m3"] = shrink_comoments(m3_sample, m3_target, alpha=alpha)
            moments["m4"] = shrink_comoments(m4_sample, m4_target, alpha=alpha)

    return moments
