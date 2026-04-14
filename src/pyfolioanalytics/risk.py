import math

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import binom
from scipy.stats import norm


def MAD(weights: np.ndarray, R: np.ndarray) -> float:
    """
    Mean Absolute Deviation (MAD) of the portfolio returns.
    MAD = 1/T * sum(|R_p - E[R_p]|)
    """
    p_returns = R @ weights
    mu_p = np.mean(p_returns)
    return np.mean(np.abs(p_returns - mu_p))


def semi_MAD(weights: np.ndarray, R: np.ndarray) -> float:
    """
    Semi Mean Absolute Deviation (Downside MAD) of the portfolio returns.
    Semi-MAD = 1/T * sum(max(E[R_p] - R_p, 0))
    """
    p_returns = R @ weights
    mu_p = np.mean(p_returns)
    return np.mean(np.maximum(mu_p - p_returns, 0.0))


def VaR(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    m3: np.ndarray | None = None,
    m4: np.ndarray | None = None,
    p: float = 0.95,
    method: str = "gaussian",
) -> float:
    p_ret = np.dot(weights, mu).item()
    p_var = np.dot(weights.T, np.dot(sigma, weights)).item()
    p_sd = np.sqrt(p_var)
    z = norm.ppf(1 - p)

    if method == "gaussian":
        return -(p_ret + z * p_sd)
    elif method == "modified" and m3 is not None and m4 is not None:
        w_sq = np.kron(weights, weights)
        skew = np.dot(weights, np.dot(m3, w_sq)) / (p_sd**3)
        w_cu = np.kron(weights, w_sq)
        kurt = np.dot(weights, np.dot(m4, w_cu)) / (p_sd**4)
        ex_kurt = kurt - 3

        z_m = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * ex_kurt / 24
            - (2 * z**3 - 5 * z) * (skew**2) / 36
        )
        return -(p_ret + z_m * p_sd)
    else:
        return -(p_ret + z * p_sd)


def ES(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    m3: np.ndarray | None = None,
    m4: np.ndarray | None = None,
    p: float = 0.95,
    method: str = "gaussian",
) -> float:
    p_ret = np.dot(weights, mu).item()
    p_var = np.dot(weights.T, np.dot(sigma, weights)).item()
    p_sd = np.sqrt(p_var)
    alpha = 1 - p
    z = norm.ppf(alpha)

    if method == "gaussian":
        es = -(p_ret + p_sd * (-norm.pdf(z) / alpha))
        return es
    elif method == "modified" and m3 is not None and m4 is not None:
        w_sq = np.kron(weights, weights)
        skew = np.dot(weights, np.dot(m3, w_sq)) / (p_sd**3)
        w_cu = np.kron(weights, w_sq)
        kurt = np.dot(weights, np.dot(m4, w_cu)) / (p_sd**4)
        ex_kurt = kurt - 3

        z_m = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * ex_kurt / 24
            - (2 * z**3 - 5 * z) * (skew**2) / 36
        )
        return -(p_ret + p_sd * (-norm.pdf(z_m) / alpha))
    else:
        return -(p_ret + p_sd * (-norm.pdf(z) / alpha))


def calculate_drawdowns(p_returns: np.ndarray) -> np.ndarray:
    cum_ret = np.cumsum(p_returns)
    peak = np.maximum.accumulate(cum_ret)
    return cum_ret - peak


def max_drawdown(weights: np.ndarray, R: np.ndarray) -> float:
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    return -np.min(drawdowns)


def average_drawdown(weights: np.ndarray, R: np.ndarray) -> float:
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    return -np.mean(drawdowns)


def risk_contribution(weights: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Component Contribution to Risk (CCR) for portfolio standard deviation.
    Sum(risk_contribution) == StdDev(portfolio).
    """
    p_var = float(weights.T @ sigma @ weights)
    if p_var < 1e-14:
        return np.zeros_like(weights)
    p_sd = np.sqrt(p_var)
    marginal_contribution = (sigma @ weights) / p_sd
    return weights * marginal_contribution


def risk_decomposition(
    weights: np.ndarray, sigma: np.ndarray, type: str = "StdDev"
) -> dict:
    """
    Comprehensive risk decomposition.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    sigma : np.ndarray
        Covariance matrix.
    type : str
        'StdDev' (default) or 'var'.

    Returns
    -------
    dict
        {
            'total': Total risk (sd or var),
            'mcr': Marginal Contribution to Risk,
            'ccr': Component Contribution to Risk,
            'pcr': Percentage Contribution to Risk
        }
    """
    weights = weights.flatten()
    p_var = float(weights.T @ sigma @ weights)

    if type == "StdDev":
        total_risk = np.sqrt(p_var)
        if total_risk < 1e-14:
            mcr = np.zeros_like(weights)
            ccr = np.zeros_like(weights)
            pcr = np.zeros_like(weights)
        else:
            mcr = (sigma @ weights) / total_risk
            ccr = weights * mcr
            pcr = ccr / total_risk
    elif type == "var":
        total_risk = p_var
        mcr = 2 * (sigma @ weights)
        ccr = weights * (
            sigma @ weights
        )  # In PA, CCR for var is often w_i * (Sigma*w)_i
        if p_var < 1e-14:
            pcr = np.zeros_like(weights)
        else:
            pcr = ccr / p_var
    else:
        raise ValueError("Type must be 'StdDev' or 'var'")

    return {"total": total_risk, "mcr": mcr, "ccr": ccr, "pcr": pcr}


def factor_risk_decomposition(
    weights: np.ndarray,
    B: np.ndarray,
    sigma_f: np.ndarray,
    residual_sigma: np.ndarray | None = None,
    type: str = "StdDev",
) -> dict:
    """
    Factor risk decomposition (Attribution).

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N,).
    B : np.ndarray
        Factor loading matrix (N, K).
    sigma_f : np.ndarray
        Factor covariance matrix (K, K).
    residual_sigma : np.ndarray, optional
        Residual (idiosyncratic) covariance matrix (N, N). Usually diagonal.
    """
    weights = weights.flatten()
    # Total systematic covariance: B @ sigma_f @ B.T
    sigma_sys = B @ sigma_f @ B.T

    if residual_sigma is not None:
        sigma_total = sigma_sys + residual_sigma
    else:
        sigma_total = sigma_sys

    # Asset level decomposition first
    asset_decomp = risk_decomposition(weights, sigma_total, type=type)

    # Factor level decomposition
    # Exposure e = B.T @ w (K,)
    exposure = B.T @ weights

    if type == "StdDev":
        total_risk = asset_decomp["total"]
        if total_risk < 1e-14:
            mcr_f = np.zeros_like(exposure)
            ccr_f = np.zeros_like(exposure)
        else:
            # MCR_f = (sigma_f @ exposure) / total_risk
            mcr_f = (sigma_f @ exposure) / total_risk
            ccr_f = exposure * mcr_f
    else:  # var
        total_risk = asset_decomp["total"]
        mcr_f = 2 * (sigma_f @ exposure)
        ccr_f = exposure * (sigma_f @ exposure)

    # Calculate Residual Contribution
    ccr_resid = total_risk - np.sum(ccr_f)

    return {
        "total": total_risk,
        "exposure": exposure,
        "mcr_factor": mcr_f,
        "ccr_factor": ccr_f,
        "pcr_factor": ccr_f / total_risk
        if total_risk > 1e-14
        else np.zeros_like(ccr_f),
        "ccr_residual": ccr_resid,
        "pcr_residual": ccr_resid / total_risk if total_risk > 1e-14 else 0.0,
    }


def _solve_evar_scalar(losses: np.ndarray, alpha: float) -> float:
    """Shared log-sum-exp minimisation used by EVaR and EDaR."""
    T = len(losses)

    def evar_obj(z: float) -> float:
        if z <= 0:
            return 1e10
        # Log-sum-exp trick for numerical stability
        scaled = losses / z
        m = np.max(scaled)
        return z * (m + np.log(np.sum(np.exp(scaled - m)) / (T * alpha)))

    res = minimize_scalar(evar_obj, bounds=(1e-6, 100), method="bounded")
    return float(res.fun)


def EVaR(weights: np.ndarray, R: np.ndarray, p: float = 0.95) -> float:
    return _solve_evar_scalar(-(R @ weights), 1 - p)


def CDaR(weights: np.ndarray, R: np.ndarray, p: float = 0.95) -> float:
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    losses = -drawdowns
    sorted_losses = np.sort(losses)[::-1]
    n = len(sorted_losses)
    cutoff_idx = int(np.ceil((1 - p) * n))
    if cutoff_idx == 0:
        return sorted_losses[0]
    return np.mean(sorted_losses[:cutoff_idx])


def EDaR(weights: np.ndarray, R: np.ndarray, p: float = 0.95) -> float:
    p_returns = R @ weights
    # Apply EVaR to the drawdown series (not the raw returns)
    return _solve_evar_scalar(-calculate_drawdowns(p_returns), 1 - p)


def UCI(weights: np.ndarray, R: np.ndarray, **kwargs) -> float:
    """
    Calculate the Ulcer Index (Root Mean Square Drawdown) of a portfolio
    using uncompounded cumulative returns.
    """
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    return float(np.sqrt(np.mean(drawdowns ** 2)))


def RLVaR(
    weights: np.ndarray, R: np.ndarray, p: float = 0.95, kappa: float = 0.3
) -> float:
    losses = -(R @ weights)
    T = len(losses)
    alpha = 1 - p
    Z = cp.Variable(T)
    nu = cp.Variable(T)
    tau = cp.Variable(T)
    ones = np.ones(T)
    c = ((1 / (alpha * T)) ** kappa - (1 / (alpha * T)) ** (-kappa)) / (2 * kappa)
    constraints = [
        cp.sum(Z) == 1,
        cp.sum(nu - tau) / (2 * kappa) <= c,
        cp.PowCone3D(nu, ones, Z, 1 / (1 + kappa)),
        cp.PowCone3D(Z, ones, tau, 1 - kappa),
    ]
    prob = cp.Problem(cp.Maximize(Z @ losses), constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-5)
    except Exception:
        prob.solve()
    return float(prob.value)


def RLDaR(
    weights: np.ndarray, R: np.ndarray, p: float = 0.95, kappa: float = 0.3
) -> float:
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    losses = -drawdowns
    T = len(losses)
    alpha = 1 - p
    Z = cp.Variable(T)
    nu = cp.Variable(T)
    tau = cp.Variable(T)
    ones = np.ones(T)
    c = ((1 / (alpha * T)) ** kappa - (1 / (alpha * T)) ** (-kappa)) / (2 * kappa)
    constraints = [
        cp.sum(Z) == 1,
        cp.sum(nu - tau) / (2 * kappa) <= c,
        cp.PowCone3D(nu, ones, Z, 1 / (1 + kappa)),
        cp.PowCone3D(Z, ones, tau, 1 - kappa),
    ]
    prob = cp.Problem(cp.Maximize(Z @ losses), constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-5)
    except Exception:
        prob.solve()
    return float(prob.value)


def owa_risk(weights: np.ndarray, R: np.ndarray, owa_weights: np.ndarray) -> float:
    losses = -(R @ weights)
    sorted_losses = np.sort(losses)[::-1]
    return np.dot(owa_weights, sorted_losses)


def owa_l_moment_weights(T: int, k: int = 2) -> np.ndarray:
    # Vectorized via broadcasting over the double loop on i and j
    i_arr = np.arange(1, T + 1, dtype=float)  # shape (T,)
    j_arr = np.arange(k, dtype=float)  # shape (k,)
    # Expand dims for broadcasting: (T, k)
    i_col = i_arr[:, np.newaxis]
    j_row = j_arr[np.newaxis, :]
    signs = (-1.0) ** j_row
    # scipy binom is element-wise on arrays
    terms = (
        signs
        * binom(k - 1, j_row)
        * binom(i_col - 1, k - 1 - j_row)
        * binom(T - i_col, j_row)
    )
    w = terms.sum(axis=1) / (k * binom(T, k))
    return w


def l_moment(R: np.ndarray, weights: np.ndarray, k: int = 2) -> float:
    p_returns = R @ weights
    T = len(p_returns)
    w_owa = owa_l_moment_weights(T, k=k)
    sorted_returns = np.sort(p_returns)
    return float(np.dot(w_owa, sorted_returns))


def owa_l_moment_crm_weights(
    T: int,
    k: int = 4,
    method: str = "MSD",
    g: float = 0.5,
    max_phi: float = 0.5,
    solver: str | None = None,
) -> np.ndarray:
    ws = np.empty((T, 0))
    for i in range(2, k + 1):
        w_i = ((-1) ** i * owa_l_moment_weights(T, k=i)).reshape(-1, 1)
        ws = np.concatenate([ws, w_i], axis=1)

    if method == "CRRA":
        phis = []
        e = 1
        for i in range(1, k):
            e *= g + i - 1
            phis.append(e / math.factorial(i + 1))
        phis = np.array(phis)
        phis = phis / np.sum(phis)
        a = ws @ phis.reshape(-1, 1)
        w = a.flatten()[::-1]
        for i in range(1, len(w)):
            w[i] = np.min(w[: i + 1])
        return w

    else:
        theta = cp.Variable((T, 1))
        n_phis = ws.shape[1]
        phi = cp.Variable((n_phis, 1))
        constraints = [
            cp.sum(phi) == 1,
            theta == ws @ phi,
            phi <= max_phi,
            phi >= 0,
            phi[1:] <= phi[:-1],
            theta[1:] >= theta[:-1],
        ]
        if method == "ME":
            theta_ = cp.Variable((T, 1))
            obj = cp.sum(cp.entr(theta_)) * 1000
            constraints += [theta_ >= theta, theta_ >= -theta]
            objective = cp.Maximize(obj)
        elif method == "MSS":
            obj = cp.norm(theta, 2) * 1000
            objective = cp.Minimize(obj)
        elif method == "MSD":
            obj = cp.norm(theta[1:] - theta[:-1], 2) * 1000
            objective = cp.Minimize(obj)
        else:
            raise ValueError(f"Unknown method {method}")

        problem = cp.Problem(objective, constraints)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Solution may be inaccurate")
            try:
                if solver:
                    problem.solve(solver=solver)
                else:
                    try:
                        problem.solve(solver=cp.CLARABEL, verbose=False)
                    except Exception:
                        problem.solve()
            except Exception:
                problem.solve()

        if phi.value is None:
            return np.ones(T) / T  # Fallback

        phis = phi.value.flatten()
        a = (ws @ phis.reshape(-1, 1)).flatten()
        w = a[::-1]
        for i in range(1, len(w)):
            w[i] = np.min(w[: i + 1])
        return w


def owa_gmd_weights(T: int) -> np.ndarray:
    w = (4 * np.arange(1, T + 1) - 2 * (T + 1)) / (T * (T - 1))
    return np.sort(w)[::-1]


def owa_cvar_weights(T: int, p: float = 0.05) -> np.ndarray:
    # Handle both p=0.05 (alpha) and p=0.95 (confidence level)
    alpha = p if p < 0.5 else 1 - p
    k = int(np.ceil(alpha * T))
    w = np.zeros(T)
    w[: k - 1] = 1 / (alpha * T)
    w[k - 1] = (alpha * T - (k - 1)) / (alpha * T)
    return w


def numerical_risk_contribution(
    weights: np.ndarray, R: np.ndarray, risk_func, **kwargs
) -> np.ndarray:
    """
    Compute the marginal risk contribution of any risk measure using numerical differentiation (Euler's theorem).
    """
    eps = 1e-6
    n = len(weights)
    grad = np.zeros(n)

    for i in range(n):
        w_plus = weights.copy()
        w_plus[i] += eps
        r_plus = risk_func(w_plus, R, **kwargs)

        w_minus = weights.copy()
        w_minus[i] -= eps
        r_minus = risk_func(w_minus, R, **kwargs)

        grad[i] = (r_plus - r_minus) / (2 * eps)

    rc = weights * grad

    # Optional scaling to exact risk value to fix floating point drift
    total_risk = risk_func(weights, R, **kwargs)
    sum_rc = np.sum(rc)
    if abs(sum_rc) > 1e-12 and abs(total_risk) > 1e-12:
        rc = rc * (total_risk / sum_rc)

    return rc


def CVaR(weights: np.ndarray, R: np.ndarray, p: float = 0.95) -> float:
    """Historical Conditional Value at Risk (Expected Shortfall)."""
    port_returns = R @ weights
    var_threshold = np.percentile(port_returns, 100 * (1 - p))
    # Average of returns below the VaR threshold
    tail_returns = port_returns[port_returns <= var_threshold]
    if len(tail_returns) == 0:
        return -var_threshold
    return -np.mean(tail_returns)
