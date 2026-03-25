from typing import Any

import numpy as np
import pandas as pd

from .factors import ac_ranking, factor_model_covariance, statistical_factor_model


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
        from sklearn.covariance import OAS, LedoitWolf

        moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        target = kwargs.get("shrinkage_target", "identity")

        if target == "identity":
            lw = LedoitWolf().fit(R_filtered.values)
            moments["sigma"] = lw.covariance_
        elif target == "oas":
            oas = OAS().fit(R_filtered.values)
            moments["sigma"] = oas.covariance_
        elif target == "constant_correlation":
            # Ledoit-Wolf shrinkage towards constant correlation
            # Equivalent to R's RiskPortfolios::covMcd or Riskfolio-Lib's cov_ledoit
            X = R_filtered.values
            T, N = X.shape

            # De-mean returns
            Y = X - np.mean(X, axis=0)

            # Sample covariance matrix (T-1)
            S = np.cov(X, rowvar=False)

            # Sample correlation matrix
            std = np.sqrt(np.diag(S))
            Corr = S / np.outer(std, std)

            # Mean correlation
            r_bar = (np.sum(Corr) - N) / (N * (N - 1))

            # Target matrix F
            F = r_bar * np.outer(std, std)
            np.fill_diagonal(F, np.diag(S))

            # Estimate Pi_mat (asymptotic variance of sample covariance)
            # Y is TxN
            # We need cov(y_it * y_jt) ->
            # Pi is sum_t [ (Y_it Y_jt - S_ij)^2 ] / T

            Pi_mat = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    Pi_mat[i, j] = np.sum((Y[:, i] * Y[:, j] - S[i, j]) ** 2) / T

            pi_hat = np.sum(Pi_mat)

            # Estimate Rho
            term1 = np.sum(np.diag(Pi_mat))
            term2 = 0.0

            for i in range(N):
                for j in range(N):
                    if i != j:
                        term2 += (r_bar / 2) * (
                            np.sqrt(S[j, j] / S[i, i])
                            * np.sum(
                                (Y[:, i] ** 2 * Y[:, j] - S[i, i] * S[i, j])
                                * (Y[:, i] * Y[:, j] - S[i, j])
                            )
                            / T
                            + np.sqrt(S[i, i] / S[j, j])
                            * np.sum(
                                (Y[:, j] ** 2 * Y[:, i] - S[j, j] * S[i, j])
                                * (Y[:, i] * Y[:, j] - S[i, j])
                            )
                            / T
                        ) - r_bar * np.sqrt(S[i, i] * S[j, j]) * Pi_mat[i, j]

            rho_hat = term1 + term2

            # Estimate Gamma
            gamma_hat = np.sum((S - F) ** 2)

            # Shrinkage intensity
            kappa_hat = (pi_hat - rho_hat) / gamma_hat
            delta = max(0.0, min(1.0, kappa_hat / T))

            moments["sigma"] = delta * F + (1 - delta) * S
        else:
            raise ValueError(f"Unknown shrinkage target: {target}")
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
    elif method == "garch":
        moments = ccc_garch_moments(R_filtered.values)
    elif method == "ewma":
        span = kwargs.get("span", 36)
        res_ewma = ewma_moments(R_filtered.values, span=span)
        moments["mu"] = res_ewma["mu"]
        moments["sigma"] = res_ewma["sigma"]
    elif method == "semi_covariance":
        benchmark = kwargs.get("benchmark", 0.0)
        moments["mu"] = R_filtered.mean().values.reshape(-1, 1)
        moments["sigma"] = semi_covariance(R_filtered.values, benchmark=benchmark)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

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
