import numpy as np
from scipy.stats import norm
from typing import Optional, Dict, Union

def VaR(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    m3: Optional[np.ndarray] = None,
    m4: Optional[np.ndarray] = None,
    p: float = 0.95,
    method: str = "gaussian"
) -> float:
    z_p = norm.ppf(1 - p)
    p_mu = np.dot(weights, mu).item()
    p_var = np.dot(weights.T, np.dot(sigma, weights))
    p_sd = np.sqrt(p_var)
    if method == "gaussian":
        return -(p_mu + z_p * p_sd)
    elif method == "modified" and m3 is not None and m4 is not None:
        p_skew = np.dot(weights, np.dot(m3, np.kron(weights, weights))) / (p_sd ** 3)
        p_kurt = np.dot(weights, np.dot(m4, np.kron(weights, np.kron(weights, weights)))) / (p_sd ** 4)
        h_p = z_p + (1/6) * (z_p**2 - 1) * p_skew + (1/24) * (z_p**3 - 3*z_p) * (p_kurt - 3) - (1/36) * (2*z_p**3 - 5*z_p) * (p_skew**2)
        return -(p_mu + h_p * p_sd)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

def ES(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    m3: Optional[np.ndarray] = None,
    m4: Optional[np.ndarray] = None,
    p: float = 0.95,
    method: str = "gaussian"
) -> float:
    if method == "gaussian":
        p_mu = np.dot(weights, mu).item()
        p_var = np.dot(weights.T, np.dot(sigma, weights))
        p_sd = np.sqrt(p_var)
        alpha = 1 - p
        return -(p_mu - p_sd * (norm.pdf(norm.ppf(alpha)) / alpha))
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

def risk_contribution(
    weights: np.ndarray,
    sigma: np.ndarray,
    name: str = "StdDev"
) -> np.ndarray:
    if name in ["StdDev", "var"]:
        p_var = np.dot(weights.T, np.dot(sigma, weights))
        p_sd = np.sqrt(p_var)
        marginal_risk = np.dot(sigma, weights)
        if name == "StdDev":
            return weights * marginal_risk / p_sd
        else:
            return weights * marginal_risk
    else:
        raise NotImplementedError(f"Risk measure '{name}' is not supported for contribution.")

def calculate_drawdowns(returns: np.ndarray) -> np.ndarray:
    wealth_index = np.cumprod(1 + returns)
    previous_peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index / previous_peaks) - 1
    return drawdowns

def max_drawdown(weights: np.ndarray, R: np.ndarray) -> float:
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    return -np.min(drawdowns)

def average_drawdown(weights: np.ndarray, R: np.ndarray) -> float:
    """
    Calculate the Average Drawdown matching R PerformanceAnalytics behavior.
    Average of the trough values of all drawdown periods.
    """
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    
    # Identify trough values of drawdown periods
    # A trough is the minimum value within a period where drawdown < 0
    troughs = []
    in_drawdown = False
    current_min = 0.0
    
    for d in drawdowns:
        if d < 0:
            if not in_drawdown:
                in_drawdown = True
                current_min = d
            else:
                current_min = min(current_min, d)
        else:
            if in_drawdown:
                troughs.append(current_min)
                in_drawdown = False
                
    # If the last period is in drawdown
    if in_drawdown:
        troughs.append(current_min)
        
    if not troughs: return 0.0
    return -np.mean(troughs)

def CDaR(weights: np.ndarray, R: np.ndarray, p: float = 0.95) -> float:
    p_returns = R @ weights
    drawdowns = calculate_drawdowns(p_returns)
    sorted_drawdowns = np.sort(drawdowns)
    n = len(sorted_drawdowns)
    cutoff_idx = int(np.ceil((1 - p) * n))
    if cutoff_idx == 0: return -sorted_drawdowns[0]
    return -np.mean(sorted_drawdowns[:cutoff_idx])
