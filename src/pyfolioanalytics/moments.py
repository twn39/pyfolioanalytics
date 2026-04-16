from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .factors import ac_ranking, factor_model_covariance, statistical_factor_model

def clean_returns_boudt(R: pd.DataFrame | np.ndarray, alpha: float = 0.05) -> pd.DataFrame | np.ndarray:
    """
    Robust return cleaning (Winsorization) using the Boudt et al. (2008) method.
    Identifies multivariate outliers using Mahalanobis distance based on MCD 
    (Minimum Covariance Determinant) robust estimates and scales them back to 
    the boundaries of the chi-squared distribution.
    """
    isinstance(R, pd.DataFrame)
    R_vals = R.values if isinstance(R, pd.DataFrame) else np.asarray(R)
    T, N = R_vals.shape
    
    # 1. Robust Mean and Covariance estimation (MCD)
    try:
        from sklearn.covariance import MinCovDet
        # Ensure sufficient observations for MCD, otherwise fallback to standard
        if T > 2 * N:
            mcd = MinCovDet(random_state=42).fit(R_vals)
            mu_mcd = mcd.location_
            cov_mcd = mcd.covariance_
        else:
            raise ValueError("Not enough observations for MCD")
    except Exception as e:
        warnings.warn(f"MCD fitting failed ({str(e)}), falling back to sample moments for Boudt cleaning.")
        mu_mcd = np.mean(R_vals, axis=0)
        cov_mcd = np.cov(R_vals, rowvar=False)
        
    # Calculate pseudo-inverse to handle ill-conditioned covariance
    cov_inv = np.linalg.pinv(cov_mcd)
    
    # 2. Squared Mahalanobis Distance D^2
    diff = R_vals - mu_mcd
    # Vectorized computation of (R_t - mu)^T * Sigma^-1 * (R_t - mu)
    D_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
    
    # 3. Chi-Square threshold (df = N assets)
    threshold = chi2.ppf(1 - alpha, df=N)
    
    # 4. Outlier detection and scaling factor computation
    scaling_factors = np.ones(T)
    outliers = D_sq > threshold
    if np.any(outliers):
        scaling_factors[outliers] = np.sqrt(threshold / D_sq[outliers])
    
    # 5. Winsorization (scaling back outliers)
    R_clean = mu_mcd + diff * scaling_factors[:, np.newaxis]
    
    if isinstance(R, pd.DataFrame):
        return pd.DataFrame(R_clean, index=R.index, columns=R.columns)
    return R_clean

def M3_MM(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    # Vectorized: M3[i,j,k] = mean(R[:,i] * R[:,j] * R[:,k])
    M3 = np.einsum("ti,tj,tk->ijk", R, R, R) / T
    return M3.reshape(N, N * N)


def M4_MM(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    # Vectorized: M4[i,j,k,l] = mean(R[:,i] * R[:,j] * R[:,k] * R[:,l])
    M4 = np.einsum("ti,tj,tk,tl->ijkl", R, R, R, R) / T
    return M4.reshape(N, N**3)


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
                    for l_idx in range(N):
                        kijkl = 0.0
                        if (
                            (i == j)
                            or (i == k_idx)
                            or (i == l_idx)
                            or (j == k_idx)
                            or (j == l_idx)
                            or (k_idx == l_idx)
                        ):
                            if (i == j) and (i == k_idx) and (i == l_idx):
                                kijkl = 6 * b[i] * b[i] * f2_val * s2[i] + s4[i]
                            elif (
                                ((i == j) and (i == k_idx))
                                or ((i == j) and (i == l_idx))
                                or ((i == k_idx) and (i == l_idx))
                                or ((j == k_idx) and (j == l_idx))
                            ):
                                if (i == j) and (i == k_idx):
                                    kijkl = 3 * b[i] * b[l_idx] * f2_val * s2[i]
                                elif (i == j) and (i == l_idx):
                                    kijkl = 3 * b[i] * b[k_idx] * f2_val * s2[i]
                                elif (i == k_idx) and (i == l_idx):
                                    kijkl = 3 * b[i] * b[j] * f2_val * s2[i]
                                elif (j == k_idx) and (j == l_idx):
                                    kijkl = 3 * b[j] * b[i] * f2_val * s2[j]
                            elif (
                                ((i == j) and (k_idx == l_idx))
                                or ((i == k_idx) and (j == l_idx))
                                or ((i == l_idx) and (j == k_idx))
                            ):
                                if (i == j) and (k_idx == l_idx):
                                    kijkl = (
                                        b[i] * b[i] * f2_val * s2[k_idx]
                                        + b[k_idx] * b[k_idx] * f2_val * s2[i]
                                        + s2[i] * s2[k_idx]
                                    )
                                elif (i == k_idx) and (j == l_idx):
                                    kijkl = (
                                        b[i] * b[i] * f2_val * s2[j]
                                        + b[j] * b[j] * f2_val * s2[i]
                                        + s2[i] * s2[j]
                                    )
                                elif (i == l_idx) and (j == k_idx):
                                    kijkl = (
                                        b[i] * b[i] * f2_val * s2[j]
                                        + b[j] * b[j] * f2_val * s2[i]
                                        + s2[i] * s2[j]
                                    )
                            else:
                                if i == j:
                                    kijkl = b[k_idx] * b[l_idx] * f2_val * s2[i]
                                elif i == k_idx:
                                    kijkl = b[j] * b[l_idx] * f2_val * s2[i]
                                elif i == l_idx:
                                    kijkl = b[j] * b[k_idx] * f2_val * s2[i]
                                elif j == k_idx:
                                    kijkl = b[i] * b[l_idx] * f2_val * s2[j]
                                elif j == l_idx:
                                    kijkl = b[i] * b[k_idx] * f2_val * s2[j]
                                elif k_idx == l_idx:
                                    kijkl = b[i] * b[j] * f2_val * s2[k_idx]

                        D[l_idx, i * N * N + j * N + k_idx] = kijkl
    else:
        # Multi-factor residual approximation
        for i in range(N):
            D[i, i * N**2 + i * N + i] = stockM4[i]

    return S + D


def shrink_comoments(
    M_sample: np.ndarray, M_target: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    return (1 - alpha) * M_sample + alpha * M_target


def ewma_moments(R: np.ndarray, span: int = 36) -> dict[str, Any]:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) mean and covariance.
    """
    alpha = 2.0 / (span + 1)
    T, N = R.shape
    weights = (1 - alpha) ** np.arange(T - 1, -1, -1)
    weights /= weights.sum()

    mu = np.average(R, weights=weights, axis=0)
    R_centered = R - mu
    # unbiased-like normalization can be done, but keeping it simple with weights
    cov = (weights * R_centered.T) @ R_centered
    return {"mu": mu.reshape(-1, 1), "sigma": cov}


def semi_covariance(R: np.ndarray, benchmark: float = 0.0) -> np.ndarray:
    """
    Calculate semi-covariance matrix (downside covariance), penalizing returns below benchmark.
    """
    R_down = np.minimum(R - benchmark, 0.0)
    T = R.shape[0]
    return (R_down.T @ R_down) / T


def ema_returns(R: pd.DataFrame, span: int = 252) -> np.ndarray:
    """
    Calculate the exponentially-weighted mean of historical returns.
    """
    ewm_mean = R.ewm(span=span).mean().iloc[-1]
    return ewm_mean.values.reshape(-1, 1)


def capm_returns(
    R: pd.DataFrame,
    market_returns: pd.Series | None = None,
    market_caps: pd.Series | dict[str, float] | None = None,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """
    Calculate the expected returns based on the Capital Asset Pricing Model (CAPM).
    Matches PyPfOpt's capm_return logic strictly.
    """
    returns = R.copy()
    
    if market_returns is not None:
        if isinstance(market_returns, pd.DataFrame):
            market_returns = market_returns.iloc[:, 0]
    else:
        # Construct proxy for market
        if market_caps is not None:
            mc = pd.Series(market_caps)
            mc = mc.reindex(R.columns).fillna(0.0)
            if mc.sum() > 0:
                weights = mc / mc.sum()
            else:
                weights = pd.Series(1.0 / len(R.columns), index=R.columns)
            market_returns = R.dot(weights)
        else:
            market_returns = R.mean(axis=1)

    returns["mkt"] = market_returns
    cov = returns.cov()
    
    # Beta = Cov(R_i, R_m) / Var(R_m)
    if cov.loc["mkt", "mkt"] > 0:
        betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    else:
        betas = pd.Series(0.0, index=returns.columns)
        
    betas = betas.drop("mkt")

    # Assuming daily returns, compounding to match PyPfOpt's annualized mkt_mean_ret logic
    # BUT we want to return raw scale to match PyFolioAnalytics API. We use raw means.
    # PyPfOpt allows toggling compounding. We'll stick to raw mean.
    mkt_mean_ret = (1 + returns["mkt"]).prod() ** (1.0 / returns["mkt"].count()) - 1.0
    
    # Expected return = Rf + Beta * (E[Rm] - Rf)
    expected_returns = risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)
    
    return expected_returns.values.reshape(-1, 1)


def shrunk_covariance(
    R: pd.DataFrame, 
    method: str = "ledoit_wolf", 
    shrinkage_target: str = "constant_variance"
) -> np.ndarray:
    """
    Native implementation of advanced covariance shrinkage (Ledoit-Wolf & OAS).
    """
    from sklearn.covariance import LedoitWolf, OAS
    
    X = np.nan_to_num(R.values)
    t, n = X.shape

    if method == "oas":
        oas = OAS(assume_centered=False).fit(X)
        return oas.covariance_
        
    elif method == "ledoit_wolf":
        if shrinkage_target == "constant_variance":
            # Scikit-learn's default implementation
            lw = LedoitWolf(assume_centered=False).fit(X)
            return lw.covariance_
            
        elif shrinkage_target == "constant_correlation":
            # Native implementation matching Ledoit & Wolf (2003) / PyPfOpt
            S = np.cov(X, rowvar=False)
            
            var = np.diag(S).reshape(-1, 1)
            std = np.sqrt(var)
            _var = np.tile(var, (n,))
            _std = np.tile(std, (n,))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                cor_mat = S / (_std * _std.T)
                cor_mat[np.isnan(cor_mat) | np.isinf(cor_mat)] = 0.0
                
            r_bar = (np.sum(cor_mat) - n) / (n * (n - 1)) if n > 1 else 1.0
            
            F = r_bar * (_std * _std.T)
            F[np.eye(n) == 1] = var.reshape(-1)
            
            Xm = X - X.mean(axis=0)
            y = Xm**2
            
            # Estimate pi
            pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S**2
            pi_hat = np.sum(pi_mat)
            
            # Theta matrix, expanded term by term
            term1 = np.dot((Xm**3).T, Xm) / t
            help_ = np.dot(Xm.T, Xm) / t
            help_diag = np.diag(help_)
            term2 = np.tile(help_diag, (n, 1)).T * S
            term3 = help_ * _var
            term4 = _var * S
            
            theta_mat = term1 - term2 - term3 + term4
            theta_mat[np.eye(n) == 1] = np.zeros(n)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_std = np.where(std > 1e-10, 1.0 / std, 0.0)
                
            rho_hat = np.sum(np.diag(pi_mat)) + r_bar * np.sum(
                np.dot(inv_std, std.T) * theta_mat
            )
            
            # Estimate gamma
            gamma_hat = np.linalg.norm(S - F, "fro") ** 2
            
            # Compute shrinkage constant
            if gamma_hat < 1e-10:
                delta = 0.0
            else:
                kappa_hat = (pi_hat - rho_hat) / gamma_hat
                delta = max(0.0, min(1.0, kappa_hat / t))
                
            return delta * F + (1.0 - delta) * S
            
        else:
            raise ValueError(f"Unknown shrinkage target: {shrinkage_target}")
    else:
        raise ValueError(f"Unknown shrinkage method: {method}")


def ccc_garch_moments(R: np.ndarray, mu: np.ndarray | None = None) -> dict[str, Any]:
    """
    Constant Conditional Correlation (CCC) GARCH Moment Model.
    Equivalent to PortfolioAnalytics::CCCgarch.MM.
    """
    import warnings

    from arch import arch_model

    T, N = R.shape
    if mu is None:
        mu = np.mean(R, axis=0)

    R_centered = R - mu
    S = np.zeros((T, N))
    nextS = np.zeros(N)

    for i in range(N):
        # Scale returns by 100 for stability (arch library recommendation)
        scale_factor = 100.0
        y = R_centered[:, i] * scale_factor

        # Fit GARCH(1,1)
        model = arch_model(y, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp="off", show_warning=False)

        # alpha1 check (on scaled parameters)
        alpha1 = res.params.get("alpha[1]", 0.0)

        if alpha1 < 0.01:
            sigmat_scaled = np.full(T, np.std(y))
            nextSt_scaled = np.std(y)
        else:
            sigmat_scaled = res.conditional_volatility
            forecast = res.forecast(horizon=1)
            nextSt_scaled = np.sqrt(forecast.variance.values[-1, 0])

        # De-scale results
        S[:, i] = sigmat_scaled / scale_factor
        nextS[i] = nextSt_scaled / scale_factor

    # Standardized residuals
    U = R_centered / S

    # Constant Correlation Matrix
    Rcor = np.corrcoef(U, rowvar=False)

    # Conditional Covariance Matrix for next period
    D = np.diag(nextS)
    sigma = D @ Rcor @ D

    # Rescale U for higher order moments matching R
    # uncS = sqrt(diag(cov(U)))
    uncS = np.std(U, axis=0)
    U_rescaled = U * (nextS / uncS)

    return {
        "mu": mu.reshape(-1, 1),
        "sigma": sigma,
        "m3": M3_MM(U_rescaled),
        "m4": M4_MM(U_rescaled),
    }


def set_portfolio_moments(
    R: pd.DataFrame, portfolio: Any, method: str = "sample", **kwargs
) -> dict[str, Any]:
    # Handle Multi-Layer Portfolio
    if hasattr(portfolio, "root"):
        portfolio = portfolio.root

    moments = {}
    asset_names = list(portfolio.assets.keys())
    R_filtered = R[asset_names]
    
    # Handle Return Cleaning (e.g. Boudt robust winsorization)
    clean_method = kwargs.get("clean_returns")
    if clean_method == "boudt":
        alpha = kwargs.get("clean_alpha", 0.05)
        R_filtered = pd.DataFrame(clean_returns_boudt(R_filtered, alpha=alpha), columns=R_filtered.columns, index=R_filtered.index)

    # Resolve covariance and expected return methods
    sigma_method = kwargs.get("sigma_method", method)
    mu_method = kwargs.get("mu_method", method)

    # 1. Covariance Matrix Estimation
    if sigma_method == "sample":
        moments["sigma"] = R_filtered.cov().values
    elif sigma_method == "factor_model":
        k = kwargs.get("k", 3)
        fm = statistical_factor_model(R_filtered, k=k)
        moments["sigma"] = factor_model_covariance(fm)
    elif sigma_method == "shrinkage" or sigma_method == "ledoit_wolf" or sigma_method == "oas":
        target = kwargs.get("shrinkage_target", "constant_variance")
        # Handle legacy "shrinkage" method mappings
        if sigma_method == "shrinkage":
            if target == "identity":
                target = "constant_variance"
            method_arg = "oas" if target == "oas" else "ledoit_wolf"
        else:
            method_arg = sigma_method
        
        moments["sigma"] = shrunk_covariance(R_filtered, method=method_arg, shrinkage_target=target)
    elif sigma_method == "robust" or sigma_method == "mcd":
        from sklearn.covariance import MinCovDet
        mcd = MinCovDet(random_state=42).fit(R_filtered.values)
        moments["sigma"] = mcd.covariance_
        # MCD also robustly estimates the mean
        if kwargs.get("mu_method") is None and (method == "mcd" or method == "robust"):
            moments["mu"] = mcd.location_.reshape(-1, 1)
    elif sigma_method == "denoised":
        from .rmt import denoise_covariance
        T, N = R_filtered.shape
        q = T / N
        sigma = R_filtered.cov().values
        moments["sigma"] = denoise_covariance(
            sigma, q, method=kwargs.get("denoise_method", "fixed")
        )
    elif sigma_method == "garch":
        garch_res = ccc_garch_moments(R_filtered.values)
        moments["sigma"] = garch_res["sigma"]
        if kwargs.get("mu_method") is None and method == "garch":
            moments["mu"] = garch_res["mu"]
    elif sigma_method == "ewma":
        span = kwargs.get("span", 36)
        res_ewma = ewma_moments(R_filtered.values, span=span)
        moments["sigma"] = res_ewma["sigma"]
        if kwargs.get("mu_method") is None and method == "ewma":
            moments["mu"] = res_ewma["mu"]
    elif sigma_method == "semi_covariance":
        benchmark = kwargs.get("benchmark", 0.0)
        moments["sigma"] = semi_covariance(R_filtered.values, benchmark=benchmark)
    elif sigma_method == "black_litterman":
        from .black_litterman import black_litterman
        sigma = R_filtered.cov().values
        w_mkt = kwargs.get("w_mkt", np.full((len(asset_names), 1), 1.0 / len(asset_names)))
        P = kwargs.get("P")
        q = kwargs.get("q")
        tau = kwargs.get("tau", 0.05)
        risk_aversion = kwargs.get("risk_aversion", 2.5)
        res_bl = black_litterman(sigma, w_mkt, P, q, tau, risk_aversion)
        moments["sigma"] = res_bl["sigma"]
        if kwargs.get("mu_method") is None and method == "black_litterman":
            moments["mu"] = res_bl["mu"]
    elif sigma_method == "meucci":
        from .meucci import entropy_pooling, meucci_moments
        T, N = R_filtered.shape
        prior_probs = kwargs.get("prior_probs", np.full(T, 1.0 / T))
        Aeq = kwargs.get("Aeq")
        beq = kwargs.get("beq")
        p = entropy_pooling(prior_probs, Aeq=Aeq, beq=beq)
        res_m = meucci_moments(R_filtered.values, p)
        moments["sigma"] = res_m["sigma"]
        if kwargs.get("mu_method") is None and method == "meucci":
            moments["mu"] = res_m["mu"]
    elif sigma_method == "ac_ranking":
        moments["sigma"] = R_filtered.cov().values
    else:
        raise NotImplementedError(f"Covariance method '{sigma_method}' is not implemented.")

    # 2. Expected Returns Estimation
    if "mu" not in moments or mu_method != method:
        if mu_method == "sample" or mu_method == "historical" or mu_method == "semi_covariance" or mu_method == "shrinkage" or mu_method == "denoised":
            moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        elif mu_method == "ema":
            span = kwargs.get("ema_span", 252)
            moments["mu"] = ema_returns(R_filtered, span=span)
        elif mu_method == "capm":
            moments["mu"] = capm_returns(
                R_filtered, 
                market_returns=kwargs.get("market_returns"),
                market_caps=kwargs.get("market_caps"),
                risk_free_rate=kwargs.get("risk_free_rate", 0.0)
            )
        elif mu_method == "ac_ranking":
            order = kwargs.get("order")
            if order is None:
                raise ValueError("Method 'ac_ranking' requires an 'order' argument.")
            moments["mu"] = ac_ranking(R_filtered, order).reshape(-1, 1)
        elif mu_method == "factor_model":
            moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        elif "mu" not in moments:
             # Ultimate fallback
             moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
             
    # Ensure sigma is set if somehow still missing
    if "sigma" not in moments:
        moments["sigma"] = R_filtered.cov().values

    # Only compute higher-order moments for modified (Cornish-Fisher) VaR/ES,
    # not for Gaussian — avoids O(T·N⁴) work when not needed.
    needs_m3_m4 = any(
        obj["name"] in ["VaR", "ES", "mVaR", "mES"]
        and obj.get("arguments", {}).get("method", "gaussian") == "modified"
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
