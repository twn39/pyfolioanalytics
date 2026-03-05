import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from .portfolio import Portfolio
from .risk import risk_contribution

def solve_mvo(
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    n = len(moments["mu"])
    w = cp.Variable(n)
    mu = moments["mu"].flatten()
    sigma = moments["sigma"]
    asset_names = list(constraints["min"].index)
    
    # 1. Handle Robustness (Worst-case Mu)
    delta_mu = constraints.get("delta_mu")
    if delta_mu is not None:
        # For long positions, worst-case is mu - delta
        # For short positions, worst-case is mu + delta
        # Using a conservative approach: mu_robust = mu - delta * sign(w)
        # However, CVXPY requires DCP compliance. 
        # For long_only (min >= 0), mu - delta is linear and sufficient.
        if np.all(constraints["min"].values >= 0):
            mu_robust = mu - delta_mu.values
        else:
            # General case: worst-case return is w@mu - delta@|w|
            # We implement this as a penalty in the objective
            mu_robust = mu
            # This would require modifying the objective function to:
            # min risk - w@mu + delta@abs(w)
    else:
        mu_robust = mu

    # 2. Base constraints
    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
        
    cp_constraints.extend([
        w >= constraints["min"].values,
        w <= constraints["max"].values
    ])
    
    # Return Constraint (using robust mu if available)
    min_return = constraints.get("min_return")
    if min_return is not None:
        cp_constraints.append(w @ mu_robust >= min_return)
    
    # Position Limits
    max_pos = constraints.get("max_pos")
    if max_pos is not None:
        z = cp.Variable(n, boolean=True)
        W_max = np.maximum(1.0, constraints["max"].values)
        cp_constraints.append(w <= cp.multiply(z, W_max))
        cp_constraints.append(cp.sum(z) <= max_pos)
        
    # Group Constraints
    if constraints.get("groups") is not None:
        groups = constraints["groups"]
        group_min = constraints["group_min"]
        group_max = constraints["group_max"]
        for i, group in enumerate(groups):
            indices = [asset_names.index(item) if isinstance(item, str) else item for item in group]
            if group_min is not None: cp_constraints.append(cp.sum(w[indices]) >= group_min[i])
            if group_max is not None: cp_constraints.append(cp.sum(w[indices]) <= group_max[i])
                
    # Turnover/TC
    w_init = constraints.get("weight_initial")
    ptc = constraints.get("ptc")
    tc_penalty = 0
    if w_init is not None:
        if constraints.get("turnover_target") is not None:
            cp_constraints.append(cp.sum(cp.abs(w - w_init)) <= constraints["turnover_target"])
        if ptc is not None:
            tc_penalty = cp.sum(cp.multiply(cp.abs(w - w_init), ptc))

    # Objectives
    return_obj = next((o for o in objectives if o["type"] in ["return", "return_objective"]), None)
    risk_obj = next((o for o in objectives if o["type"] in ["risk", "portfolio_risk_objective"]), None)
    
    if min_return is not None:
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma) + tc_penalty), cp_constraints)
    elif return_obj and risk_obj:
        risk_aversion = risk_obj.get("risk_aversion", 1.0)
        prob = cp.Problem(cp.Minimize(0.5 * risk_aversion * cp.quad_form(w, sigma) - w @ mu_robust + tc_penalty), cp_constraints)
    elif risk_obj:
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma) + tc_penalty), cp_constraints)
    elif return_obj:
        mult = return_obj.get("multiplier", -1.0)
        if mult < 0:
            prob = cp.Problem(cp.Maximize(w @ mu_robust - tc_penalty), cp_constraints)
        else:
            prob = cp.Problem(cp.Minimize(w @ mu_robust + tc_penalty), cp_constraints)
    else:
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma) + tc_penalty), cp_constraints)
        
    try:
        if max_pos is not None: prob.solve(solver=cp.SCIP, verbose=False)
        else: prob.solve(verbose=False)
    except:
        prob.solve(verbose=False)
    
    if prob.status not in ["optimal", "feasible"]: return {"status": prob.status, "weights": None}
    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}

def solve_evar(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Minimize Entropic Value at Risk (EVaR).
    """
    T, n = R.shape
    w = cp.Variable(n)
    t = cp.Variable()
    z = cp.Variable(nonneg=True)
    
    # Probability level p
    evar_obj = next((o for o in objectives if o["name"] == "EVaR"), None)
    p = evar_obj.get("arguments", {}).get("p", 0.95) if evar_obj else 0.95
    alpha = 1 - p
    
    # Constraints (Linear ones)
    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend([w >= constraints["min"].values, w <= constraints["max"].values])
    
    # EVaR objective via Exponential Cone
    # Formula: min t + z * log( sum(exp(-R_i*w - t)/z) / (T*alpha) )
    # Equivalent to convex form in CVXPY:
    objective = cp.Minimize(t + z * (cp.log_sum_exp(cp.hstack([(-R[i] @ w - t) for i in range(T)]) / z) - np.log(T * alpha)))
    
    # Note: Perspective function log_sum_exp(x/z)*z is convex. 
    # CVXPY supports this via specific atoms or reformulations.
    # For now, we use a simpler approximation or solve iteratively if needed.
    # Riskfolio implementation uses a specialized CVXPY reformulation.
    # Simplest: Minimize z * log_sum_exp(-R*w/z - t/z) + t
    
    # Let's use the SCIP or another non-linear solver if available
    try:
        prob = cp.Problem(objective, cp_constraints)
        prob.solve(verbose=False)
    except:
        return {"status": "failed", "weights": None}
        
    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}

def solve_nonlinear(moments: Dict[str, Any], constraints: Dict[str, Any], objectives: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    n = len(moments["mu"]); sigma = moments["sigma"]; mu = moments["mu"].flatten(); w0 = np.full(n, 1.0 / n)
    R = kwargs.get("R")
    def objective_fn(w):
        if np.sum(w) == 0: return 1e10
        out = 0.0
        for obj in objectives:
            if not obj.get("enabled", True): continue
            mult = obj.get("multiplier", 1.0); name = obj["name"]
            if name == "mean": out += mult * np.dot(w, mu)
            elif name in ["var", "StdDev"]:
                p_var = np.dot(w.T, np.dot(sigma, w))
                val = p_var if name == "var" else np.sqrt(p_var)
                out += mult * val
            if obj["type"] == "risk_budget":
                p_var = np.dot(w.T, np.dot(sigma, w))
                if p_var <= 0: return 1e10
                rc = w * np.dot(sigma, w) / p_var
                if obj.get("min_concentration") or obj.get("min_difference"):
                    target = np.full(n, 1.0 / n); out += 1e4 * np.sum((rc - target)**2)
                elif obj.get("max_prisk") is not None:
                    max_p = np.array(obj["max_prisk"]); out += 1e4 * np.sum(np.maximum(0, rc - max_p)**2)
            if name == "EVaR" and R is not None:
                from .risk import EVaR
                out += mult * EVaR(w, R, p=obj.get("arguments", {}).get("p", 0.95))
        return out
    cons = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-8: cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - constraints["min_sum"]})
    else: 
        cons.append({'type': 'ineq', 'fun': lambda w: np.sum(w) - constraints["min_sum"]})
        cons.append({'type': 'ineq', 'fun': lambda w: constraints["max_sum"] - np.sum(w)})
    bounds = list(zip(constraints["min"].values, constraints["max"].values))
    res = minimize(objective_fn, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-12, 'maxiter': 1000})
    return {"status": "optimal" if res.success else res.message, "weights": res.x if res.success else None, "obj_value": res.fun}

def solve_deoptim(moments: Dict[str, Any], constraints: Dict[str, Any], objectives: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    from .optimize import calculate_objective_measures; from scipy.optimize import LinearConstraint
    n = len(moments["mu"]); bounds = list(zip(constraints["min"].values, constraints["max"].values)); R = kwargs.get("R")
    def de_objective(w):
        measures = calculate_objective_measures(w, moments, objectives, R=R)
        total_score = 0.0
        for obj in objectives:
            if not obj.get("enabled", True): continue
            mult = obj.get("multiplier", 1.0); val = measures.get(obj["name"], 0.0); total_score += val * mult
            if obj["type"] == "risk_budget":
                rc_name = f"pct_contrib_{obj['name']}"
                if rc_name in measures:
                    pct_rc = measures[rc_name]
                    if obj.get("min_concentration") or obj.get("min_difference"):
                        target = np.full(n, 1.0 / n); total_score += 1e2 * np.sum((pct_rc - target)**2)
        return total_score
    lc = LinearConstraint(np.ones(n), constraints["min_sum"], constraints["max_sum"])
    res = differential_evolution(de_objective, bounds, constraints=(lc,), maxiter=kwargs.get("itermax", 100), popsize=15, tol=1e-7, polish=True)
    return {"status": "optimal" if res.success else res.message, "weights": res.x, "obj_value": res.fun}

def solve_kelly(
    R: np.ndarray,
    constraints: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Maximize expected logarithmic growth (Kelly Criterion).
    Objective: max 1/T * sum(log(1 + R @ w))
    """
    T, n = R.shape
    w = cp.Variable(n)
    
    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend([w >= constraints["min"].values, w <= constraints["max"].values])
    
    # Ensure 1 + Rw > 0 to avoid bankruptcy (log domain)
    # We add a small epsilon for stability
    cp_constraints.append(1 + R @ w >= 1e-4)
    
    objective = cp.Maximize(cp.sum(cp.log(1 + R @ w)) / T)
    prob = cp.Problem(objective, cp_constraints)
    
    try:
        prob.solve(verbose=False)
    except:
        return {"status": "failed", "weights": None}
        
    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}

def solve_mdiv(
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Maximize Diversification Ratio: (w' @ sigma_assets) / sqrt(w' @ Sigma @ w)
    Solved by fixing numerator to 1 and minimizing convex denominator.
    """
    n = len(moments["mu"])
    sigma_assets = np.sqrt(np.diag(moments["sigma"]))
    Sigma = moments["sigma"]
    
    # Transformation: w_hat / k where numerator = 1
    w_hat = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    
    # Num = 1
    cp_constraints = [w_hat @ sigma_assets == 1]
    
    # Original constraints transformed
    # min_sum <= sum(w_hat)/k <= max_sum  =>  min_sum * k <= sum(w_hat) <= max_sum * k
    cp_constraints.append(cp.sum(w_hat) >= constraints["min_sum"] * k)
    cp_constraints.append(cp.sum(w_hat) <= constraints["max_sum"] * k)
    
    # min <= w_hat / k <= max => k*min <= w_hat <= k*max
    cp_constraints.append(w_hat >= k * constraints["min"].values)
    cp_constraints.append(w_hat <= k * constraints["max"].values)
    
    # Objective: Minimize risk (transformed)
    # The diversification ratio is 1 / sqrt(w_hat' @ Sigma @ w_hat)
    # So we minimize w_hat' @ Sigma @ w_hat
    prob = cp.Problem(cp.Minimize(cp.quad_form(w_hat, Sigma)), cp_constraints)
    
    try:
        prob.solve(verbose=False)
    except:
        return {"status": "failed", "weights": None}
        
    if prob.status in ["optimal", "feasible"] and k.value > 1e-9:
        w_final = w_hat.value / k.value
        return {"status": prob.status, "weights": w_final, "obj_value": 1.0 / np.sqrt(prob.value)}
    
    return {"status": "failed", "weights": None}

def solve_owa(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Minimize Ordered Weighted Averaging (OWA) risk.
    """
    T, n = R.shape
    w = cp.Variable(n)
    
    # 1. Constraints
    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend([w >= constraints["min"].values, w <= constraints["max"].values])

    # 2. OWA Objective
    # OWA = sum(owa_weights * sorted_losses)
    # LP Formulation for non-increasing weights:
    # OWA = sum_{k=1}^T (w_k - w_{k+1}) * (sum of top k losses)
    
    owa_obj_config = next((o for o in objectives if o["name"] == "OWA"), None)
    if not owa_obj_config:
        return {"status": "failed", "message": "OWA objective config not found"}
    
    owa_weights = owa_obj_config.get("arguments", {}).get("owa_weights")
    if owa_weights is None:
        # Default to GMD if not provided
        from .risk import owa_gmd_weights
        owa_weights = owa_gmd_weights(T)
    
    # Delta weights: delta_w[k] = w[k] - w[k+1]
    # Correct LP formulation: OWA = sum( (w_k - w_{k+1}) * TopK_Sum )
    # This is only valid if weights are non-increasing: w1 >= w2 >= ... >= wT >= 0
    # If some weights are negative (like GMD), we shift them up or use the alternate formulation.
    # However, for pure risk measures like CVaR, they are non-negative and non-increasing.
    
    # Ensure weights are non-increasing for convex optimization
    if np.any(np.diff(owa_weights) > 1e-12):
        # If not non-increasing, the problem is not convex for minimization
        # We enforce it or raise an error? For now, let's just make them non-increasing.
        owa_weights = np.sort(owa_weights)[::-1]
    
    delta_w = owa_weights[:-1] - owa_weights[1:]
    
    # Variables for top-k sums
    # Sum of k largest losses = min k*zeta_k + sum(d_{t,k})
    # This only works for k < T where we actually need the sorting trick.
    # For k = T, sum of top T losses is just the sum of ALL losses (linear).
    
    if T > 1:
        zeta = cp.Variable(T - 1)
        d = cp.Variable((T, T - 1), nonneg=True)
        
        losses = -R @ w
        for k in range(1, T):
            cp_constraints.append(d[:, k-1] >= losses - zeta[k-1])
        
        # OWA = sum_{k=1}^{T-1} (w_k - w_{k+1}) * TopK_Sum + w_T * Total_Sum
        top_k_sums = [ (k * zeta[k-1] + cp.sum(d[:, k-1])) for k in range(1, T) ]
        owa_expr = cp.sum([delta_w[i] * top_k_sums[i] for i in range(T-1)]) + owa_weights[-1] * cp.sum(losses)
    else:
        owa_expr = owa_weights[0] * (-R @ w)
    
    prob = cp.Problem(cp.Minimize(owa_expr), cp_constraints)
    
    try:
        prob.solve(verbose=False)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
        
    if prob.status not in ["optimal", "feasible"]:
        return {"status": prob.status, "weights": None}
        
    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}
