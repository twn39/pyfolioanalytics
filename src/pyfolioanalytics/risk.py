import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.special import binom
from typing import Optional, List, Dict, Any, Union
import math
import cvxpy as cp


def VaR(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    m3: Optional[np.ndarray] = None,
    m4: Optional[np.ndarray] = None,
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
    m3: Optional[np.ndarray] = None,
    m4: Optional[np.ndarray] = None,
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
    p_var = weights.T @ sigma @ weights
    marginal_contribution = (sigma @ weights) / np.sqrt(p_var)
    return weights * marginal_contribution


def EVaR(weights: np.ndarray, R: np.ndarray, p: float = 0.95) -> float:
    losses = -(R @ weights)
    T = len(losses)
    alpha = 1 - p

    def evar_obj(z):
        if z <= 0:
            return 1e10
        m = np.max(losses / z)
        val = z * (m + np.log(np.sum(np.exp(losses / z - m)) / (T * alpha)))
        return val

    res = minimize_scalar(evar_obj, bounds=(1e-6, 100), method="bounded")
    return float(res.fun)


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
    drawdowns = calculate_drawdowns(p_returns)
    losses = -drawdowns 
    T = len(losses)
    alpha = 1 - p

    def evar_obj(z):
        if z <= 0:
            return 1e10
        m = np.max(losses / z)
        val = z * (m + np.log(np.sum(np.exp(losses / z - m)) / (T * alpha)))
        return val

    res = minimize_scalar(evar_obj, bounds=(1e-6, 100), method="bounded")
    return float(res.fun)


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
    except:
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
    except:
        prob.solve()
    return float(prob.value)


def owa_risk(weights: np.ndarray, R: np.ndarray, owa_weights: np.ndarray) -> float:
    losses = -(R @ weights)
    sorted_losses = np.sort(losses)[::-1]
    return np.dot(owa_weights, sorted_losses)


def owa_l_moment_weights(T: int, k: int = 2) -> np.ndarray:
    w = []
    for i in range(1, T + 1):
        a = 0
        for j in range(k):
            a += (
                (-1) ** j
                * binom(k - 1, j)
                * binom(i - 1, k - 1 - j)
                * binom(T - i, j)
            )
        a *= 1.0 / (k * binom(T, k))
        w.append(a)
    return np.array(w)


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
    solver: Optional[str] = None,
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
            w[i] = np.min(w[:i+1])
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
                    except:
                        problem.solve()
            except:
                problem.solve()

        if phi.value is None:
            return np.ones(T) / T # Fallback

        phis = phi.value.flatten()
        a = (ws @ phis.reshape(-1, 1)).flatten()
        w = a[::-1]
        for i in range(1, len(w)):
            w[i] = np.min(w[:i+1])
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
