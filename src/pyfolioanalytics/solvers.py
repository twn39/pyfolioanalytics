import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.optimize import minimize, differential_evolution


def solve_mvo(
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    n = len(moments["mu"])
    w = cp.Variable(n)
    mu = moments["mu"].flatten()
    sigma = moments["sigma"]
    asset_names = list(constraints["min"].index)

    # 1. Handle Robustness (Worst-case Mu)
    delta_mu = constraints.get("delta_mu")
    mu_robust = mu
    robust_mu_type = constraints.get("robust_mu_type", "box")

    if delta_mu is not None:
        if robust_mu_type == "box":
            if np.all(constraints["min"].values >= 0):
                mu_robust = mu - delta_mu.values
            else:
                mu_robust = mu
        elif robust_mu_type == "ellipsoidal":
            # Return objective will be w @ mu - k * ||G @ w||_2
            # where G is Cholesky factor of sigma_mu (uncertainty of mu)
            k_mu = constraints.get("k_mu", 1.0)
            sigma_mu = constraints.get("sigma_mu")
            if sigma_mu is not None:
                G_mu = np.linalg.cholesky(sigma_mu).T
                mu_robust = mu # Base mu
                # We will add penalty term later in objective
            else:
                robust_mu_type = "box" # Fallback

    # 2. Base constraints

    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])

    cp_constraints.extend(
        [w >= constraints["min"].values, w <= constraints["max"].values]
    )

    # Factor Exposure Constraint
    B = constraints.get("B")
    if B is not None:
        lower = constraints.get("lower")
        upper = constraints.get("upper")
        # B is N x K, w is N x 1 => B.T @ w is K x 1
        cp_constraints.append(B.T @ w >= lower)
        cp_constraints.append(B.T @ w <= upper)

    # Return Constraint
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
            indices = [
                asset_names.index(item) if isinstance(item, str) else item
                for item in group
            ]
            if group_min is not None:
                cp_constraints.append(cp.sum(w[indices]) >= group_min[i])
            if group_max is not None:
                cp_constraints.append(cp.sum(w[indices]) <= group_max[i])

    # Tracking Error Constraint
    te_target = constraints.get("target")
    te_benchmark = constraints.get("benchmark")
    if te_target is not None and te_benchmark is not None:
        # TE = sqrt((w - w_b)' * Sigma * (w - w_b)) <= target
        # For convex optimization, we use (w - w_b)' * Sigma * (w - w_b) <= target^2
        w_b = te_benchmark
        if isinstance(w_b, dict):
            w_b = np.array([w_b.get(name, 0.0) for name in asset_names])
        elif isinstance(w_b, pd.Series):
            w_b = w_b.reindex(asset_names, fill_value=0.0).values
        
        diff = w - w_b
        # We can use quad_form or norm if we take Cholesky
        # PortfolioAnalytics usually treats target as StdDev target
        cp_constraints.append(cp.quad_form(diff, sigma) <= te_target**2)

    # Active Share Constraint
    as_target = constraints.get("active_share_target")
    as_benchmark = constraints.get("active_share_benchmark")
    if as_target is not None and as_benchmark is not None:
        # AS = 0.5 * sum(|w - w_b|) <= target
        w_b = as_benchmark
        if isinstance(w_b, dict):
            w_b = np.array([w_b.get(name, 0.0) for name in asset_names])
        elif isinstance(w_b, pd.Series):
            w_b = w_b.reindex(asset_names, fill_value=0.0).values
        
        # AS constraint
        cp_constraints.append(0.5 * cp.norm(w - w_b, 1) <= as_target)

    # Turnover/TC
    w_init = constraints.get("weight_initial")
    ptc = constraints.get("ptc")
    tc_penalty = 0
    if w_init is not None:
        if constraints.get("turnover_target") is not None:
            cp_constraints.append(
                cp.sum(cp.abs(w - w_init)) <= constraints["turnover_target"]
            )
        if ptc is not None:
            tc_penalty = cp.sum(cp.multiply(cp.abs(w - w_init), ptc))

    # Objectives
    return_obj = next(
        (o for o in objectives if o["type"] in ["return", "return_objective"]), None
    )
    risk_obj = next(
        (o for o in objectives if o["type"] in ["risk", "portfolio_risk_objective"]),
        None,
    )

    # Uncertainty terms
    ret_uncertainty = 0
    if robust_mu_type == "ellipsoidal" and delta_mu is not None:
        sigma_mu = constraints.get("sigma_mu")
        if sigma_mu is not None:
            k_mu = constraints.get("k_mu", 1.0)
            G_mu = np.linalg.cholesky(sigma_mu).T
            ret_uncertainty = k_mu * cp.norm(G_mu @ w)

    robust_sigma_type = constraints.get("robust_sigma_type", "none")
    risk_uncertainty = 0
    if robust_sigma_type == "ellipsoidal":
        # Robust Risk: Tr(Sigma * (W + E)) + k_sigma * sigma_risk
        # where [W w; w' k] >> 0 and E >= 0
        sigma_sigma = constraints.get("sigma_sigma")
        if sigma_sigma is not None:
            k_sigma = constraints.get("k_sigma", 1.0)
            G_sigma = np.linalg.cholesky(sigma_sigma).T
            
            W = cp.Variable((n, n), symmetric=True)
            E = cp.Variable((n, n), symmetric=True)
            sigma_risk = cp.Variable()
            
            # Conic constraints for robustness
            # PortfolioAnalytics/Optimisers.jl approach:
            # || G_sigma * vec(W + E) ||_2 <= sigma_risk
            cp_constraints.append(cp.norm(G_sigma @ cp.vec(W + E, order="C")) <= sigma_risk)
            cp_constraints.append(E >> 0)
            
            # [W w; w' 1] >> 0 (using 1 since we assume k=1 for non-ratio)
            # This is equivalent to W >> w @ w'
            L = cp.vstack([cp.hstack([W, cp.reshape(w, (n, 1), order="C")]),
                           cp.hstack([cp.reshape(w, (1, n), order="C"), np.array([[1.0]])])])
            cp_constraints.append(L >> 0)
            
            risk_uncertainty = cp.trace(sigma @ (W + E)) + k_sigma * sigma_risk
            # Override standard risk term if robust
            risk_term = risk_uncertainty
        else:
            risk_term = cp.quad_form(w, sigma)
    else:
        risk_term = cp.quad_form(w, sigma)

    if min_return is not None:
        prob = cp.Problem(
            cp.Minimize(risk_term + tc_penalty), cp_constraints
        )
    elif return_obj and risk_obj:
        risk_aversion = risk_obj.get("risk_aversion", 1.0)
        # Utility: 0.5 * lambda * risk_term - (mu'w - penalty)
        prob = cp.Problem(
            cp.Minimize(
                0.5 * risk_aversion * risk_term
                - (w @ mu_robust - ret_uncertainty)
                + tc_penalty
            ),
            cp_constraints,
        )
    elif risk_obj:
        prob = cp.Problem(
            cp.Minimize(risk_term + tc_penalty), cp_constraints
        )
    elif return_obj:
        mult = return_obj.get("multiplier", -1.0)
        if mult < 0:
            prob = cp.Problem(
                cp.Minimize(-(w @ mu_robust - ret_uncertainty) + tc_penalty),
                cp_constraints,
            )
        else:
            prob = cp.Problem(
                cp.Minimize((w @ mu_robust - ret_uncertainty) + tc_penalty),
                cp_constraints,
            )
    else:
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, sigma) + tc_penalty), cp_constraints
        )


    try:
        if max_pos is not None:
            prob.solve(solver=cp.SCIP, verbose=False)
        else:
            prob.solve(verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if prob.status not in ["optimal", "feasible"]:
        return {"status": prob.status, "weights": None}
    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}


def solve_evar(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    T, n = R.shape
    w = cp.Variable(n)
    t = cp.Variable()
    z = cp.Variable(nonneg=True)

    evar_obj_conf = next((o for o in objectives if o["name"] == "EVaR"), None)
    p = evar_obj_conf.get("arguments", {}).get("p", 0.95) if evar_obj_conf else 0.95
    alpha = 1 - p

    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend(
        [w >= constraints["min"].values, w <= constraints["max"].values]
    )

    objective = cp.Minimize(
        t
        + z
        * (
            cp.log_sum_exp(cp.hstack([(-R[i] @ w - t) for i in range(T)]) / z)
            - np.log(T * alpha)
        )
    )

    try:
        prob = cp.Problem(objective, cp_constraints)
        prob.solve(verbose=False)
    except Exception:
        return {"status": "failed", "weights": None}

    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}


def solve_nonlinear(
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    n = len(moments["mu"])
    sigma = moments["sigma"]
    mu = moments["mu"].flatten()
    R = kwargs.get("R")

    def objective_fn(w: np.ndarray) -> float:
        if np.sum(w) == 0:
            return 1e10
        out = 0.0
        for obj in objectives:
            if not obj.get("enabled", True):
                continue
            mult = obj.get("multiplier", 1.0)
            name = obj["name"]
            if name == "mean":
                out += mult * np.dot(w, mu)
            elif name in ["var", "StdDev"]:
                p_var = np.dot(w.T, np.dot(sigma, w))
                val = p_var if name == "var" else np.sqrt(p_var)
                out += mult * val
            if obj["type"] == "risk_budget":
                p_var = float(np.dot(w.T, np.dot(sigma, w)))
                if p_var <= 1e-14:
                    return 1e10
                rc = w * np.dot(sigma, w) / p_var
                # Adaptive penalty: scales with 1/p_var so it stays well-conditioned
                # regardless of whether returns are in % or decimal form.
                penalty_scale = 1.0 / p_var
                if obj.get("min_concentration") or obj.get("min_difference"):
                    target = np.full(n, 1.0 / n)
                    out += penalty_scale * np.sum((rc - target) ** 2)
                elif obj.get("max_prisk") is not None:
                    max_p = np.array(obj["max_prisk"])
                    out += penalty_scale * np.sum(np.maximum(0, rc - max_p) ** 2)
            if name == "EVaR" and R is not None:
                from .risk import EVaR
                out += mult * EVaR(w, R, p=obj.get("arguments", {}).get("p", 0.95))
        return out

    cons = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-8:
        cons.append({"type": "eq", "fun": lambda w: np.sum(w) - constraints["min_sum"]})
    else:
        cons.append(
            {"type": "ineq", "fun": lambda w: np.sum(w) - constraints["min_sum"]}
        )
        cons.append(
            {"type": "ineq", "fun": lambda w: constraints["max_sum"] - np.sum(w)}
        )

    bounds = list(zip(constraints["min"].values, constraints["max"].values))
    minimize_opts = {"ftol": 1e-12, "maxiter": 1000}

    # Multi-start restarts: ERC is well-posed but SLSQP can get stuck for
    # highly correlated assets.  5 Dirichlet draws cover the simplex better.
    best_res = None
    rng = np.random.default_rng(0)
    starting_points = [np.full(n, 1.0 / n)] + [
        rng.dirichlet(np.ones(n)) for _ in range(4)
    ]
    for w0 in starting_points:
        res = minimize(
            objective_fn,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options=minimize_opts,
        )
        if res.success and (best_res is None or res.fun < best_res.fun):
            best_res = res

    if best_res is None:
        best_res = res  # Last attempt even if unsuccessful

    return {
        "status": "optimal" if best_res.success else best_res.message,
        "weights": best_res.x if best_res.success else None,
        "obj_value": best_res.fun,
    }


def solve_deoptim(
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    from .optimize import calculate_objective_measures
    from scipy.optimize import LinearConstraint

    n = len(moments["mu"])
    bounds = list(zip(constraints["min"].values, constraints["max"].values))
    R = kwargs.get("R")

    def de_objective(w):
        measures = calculate_objective_measures(w, moments, objectives, R=R)
        total_score = 0.0
        for obj in objectives:
            if not obj.get("enabled", True):
                continue
            mult = obj.get("multiplier", 1.0)
            val = measures.get(obj["name"], 0.0)
            total_score += val * mult
            if obj["type"] == "risk_budget":
                rc_name = f"pct_contrib_{obj['name']}"
                if rc_name in measures:
                    pct_rc = measures[rc_name]
                    if obj.get("min_concentration") or obj.get("min_difference"):
                        target = np.full(n, 1.0 / n)
                        total_score += 1e2 * np.sum((pct_rc - target) ** 2)
        return total_score

    lc = LinearConstraint(np.ones(n), constraints["min_sum"], constraints["max_sum"])
    res = differential_evolution(
        de_objective,
        bounds,
        constraints=(lc,),
        maxiter=kwargs.get("itermax", 100),
        popsize=15,
        tol=1e-7,
        polish=True,
    )
    return {
        "status": "optimal" if res.success else res.message,
        "weights": res.x,
        "obj_value": res.fun,
    }


def solve_kelly(R: np.ndarray, constraints: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    T, n = R.shape
    w = cp.Variable(n)

    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend(
        [w >= constraints["min"].values, w <= constraints["max"].values]
    )
    cp_constraints.append(1 + R @ w >= 1e-4)

    objective = cp.Maximize(cp.sum(cp.log(1 + R @ w)) / T)
    prob = cp.Problem(objective, cp_constraints)

    try:
        prob.solve(verbose=False)
    except Exception:
        return {"status": "failed", "weights": None}

    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}


def solve_mdiv(
    moments: Dict[str, Any], constraints: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    n = len(moments["mu"])
    sigma_assets = np.sqrt(np.diag(moments["sigma"]))
    Sigma = moments["sigma"]

    w_hat = cp.Variable(n)
    k = cp.Variable(nonneg=True)

    cp_constraints = [w_hat @ sigma_assets == 1]
    cp_constraints.append(cp.sum(w_hat) >= constraints["min_sum"] * k)
    cp_constraints.append(cp.sum(w_hat) <= constraints["max_sum"] * k)
    cp_constraints.append(w_hat >= k * constraints["min"].values)
    cp_constraints.append(w_hat <= k * constraints["max"].values)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w_hat, Sigma)), cp_constraints)

    try:
        prob.solve(verbose=False)
    except Exception:
        return {"status": "failed", "weights": None}

    if (
        prob.status in ["optimal", "feasible"]
        and k.value is not None
        and k.value > 1e-9
        and w_hat.value is not None
    ):
        w_final = w_hat.value / k.value
        return {
            "status": prob.status,
            "weights": w_final,
            "obj_value": 1.0 / np.sqrt(prob.value),
        }

    return {"status": "failed", "weights": None}


def solve_owa(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    T, n = R.shape
    w = cp.Variable(n)

    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend(
        [w >= constraints["min"].values, w <= constraints["max"].values]
    )

    owa_obj_config = next((o for o in objectives if o["name"] == "OWA"), None)
    if not owa_obj_config:
        return {"status": "failed", "message": "OWA objective config not found"}

    owa_weights = owa_obj_config.get("arguments", {}).get("owa_weights")
    if owa_weights is None:
        from .risk import owa_gmd_weights
        owa_weights = owa_gmd_weights(T)

    # Sort weights descending for convex risk measure (Standard for sorted losses)
    if np.any(np.diff(owa_weights) > 1e-12):
        owa_weights = np.sort(owa_weights)[::-1]

    delta_w = owa_weights[:-1] - owa_weights[1:]

    # Linear programming formulation for OWA
    if T > 1:
        zeta = cp.Variable(T - 1)
        d = cp.Variable((T, T - 1), nonneg=True)
        losses = -R @ w
        for k in range(1, T):
            cp_constraints.append(d[:, k - 1] >= losses - zeta[k - 1])

        top_k_sums = [(k * zeta[k - 1] + cp.sum(d[:, k - 1])) for k in range(1, T)]
        owa_expr = cp.sum(
            [delta_w[i] * top_k_sums[i] for i in range(T - 1)]
        ) + owa_weights[-1] * cp.sum(losses)
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


def solve_edar(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    T, n = R.shape
    w = cp.Variable(n)
    alpha_p = next(
        (o.get("arguments", {}).get("p", 0.95) for o in objectives if o["name"] == "EDaR"),
        0.95,
    )
    alpha = 1 - alpha_p

    # Drawdown constraints
    u = cp.Variable(T + 1)
    cum_ret = cp.Variable(T + 1)
    d = cp.Variable(T)
    
    cp_constraints = []
    # Portfolio constraints (sum, bounds)
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend([w >= constraints["min"].values, w <= constraints["max"].values])

    cp_constraints.append(cum_ret[0] == 0)
    cp_constraints.append(u[0] == 0)
    for t in range(T):
        cp_constraints.append(cum_ret[t+1] == cum_ret[t] + R[t] @ w)
        cp_constraints.append(u[t+1] >= cum_ret[t+1])
        cp_constraints.append(u[t+1] >= u[t])
        cp_constraints.append(d[t] == u[t+1] - cum_ret[t+1])

    # EVaR applied to d (DCP compliant exponential cone formulation)
    t_evar = cp.Variable()
    z_evar = cp.Variable(nonneg=True)
    ui = cp.Variable(T)
    
    # Minimize t + z * log(sum(exp((di - t)/z))/(T*alpha))
    # This is equivalent to minimizing t subject to:
    # sum(exp((di - t)/z)) <= T * alpha
    # which is sum(ui) <= T * alpha where ExpCone(di - t, z, ui)
    
    cp_constraints.append(cp.sum(ui) <= T * alpha * z_evar)
    for i in range(T):
        cp_constraints.append(cp.ExpCone(d[i] - t_evar, z_evar, ui[i]))

    prob = cp.Problem(cp.Minimize(t_evar), cp_constraints)
    try:
        prob.solve(verbose=False)
    except Exception as e:
        return {"status": "failed", "message": str(e)}

    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}


def solve_rlvar(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    T, n = R.shape
    w = cp.Variable(n)
    obj_config = next((o for o in objectives if o["name"] == "RLVaR"), {})
    alpha_p = obj_config.get("arguments", {}).get("p", 0.95)
    kappa = obj_config.get("arguments", {}).get("kappa", 0.3)
    alpha = 1 - alpha_p

    cp_constraints = []
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend([w >= constraints["min"].values, w <= constraints["max"].values])

    # RLVaR primal formulation
    t = cp.Variable()
    z = cp.Variable(nonneg=True)
    psi = cp.Variable(T)
    theta = cp.Variable(T)
    epsilon = cp.Variable(T)
    omega = cp.Variable(T)
    
    # Scale returns to improve stability
    scale = 100.0
    losses = -(R * scale) @ w
    
    ln_k = ((1 / (alpha * T)) ** kappa - (1 / (alpha * T)) ** (-kappa)) / (2 * kappa)
    
    cp_constraints.append(losses - t + epsilon + omega <= 0)
    
    # Correct 3D Power Cone application
    x1 = cp.vstack([z * (1 + kappa) / (2 * kappa)] * T).flatten(order="C")
    y1 = psi * (1 + kappa) / kappa
    cp_constraints.append(cp.PowCone3D(x1, y1, epsilon, 1 / (1 + kappa)))
    
    x2 = omega / (1 - kappa)
    y2 = theta / kappa
    z2 = cp.vstack([-z / (2 * kappa)] * T).flatten(order="C")
    cp_constraints.append(cp.PowCone3D(x2, y2, z2, 1 - kappa))
    
    obj = t + z * ln_k + cp.sum(psi + theta)
    prob = cp.Problem(cp.Minimize(obj), cp_constraints)
    
    # Try different solvers
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
    except Exception:
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve()

    return {"status": prob.status, "weights": w.value, "obj_value": prob.value / scale if w.value is not None else None}


def solve_rldar(
    R: np.ndarray,
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    T, n = R.shape
    w = cp.Variable(n)
    obj_config = next((o for o in objectives if o["name"] == "RLDaR"), {})
    alpha_p = obj_config.get("arguments", {}).get("p", 0.95)
    kappa = obj_config.get("arguments", {}).get("kappa", 0.3)
    alpha = 1 - alpha_p

    # Scale returns to improve stability
    scale = 100.0
    
    cp_constraints = []
    # Portfolio constraints
    if abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-10:
        cp_constraints.append(cp.sum(w) == constraints["min_sum"])
    else:
        cp_constraints.append(cp.sum(w) >= constraints["min_sum"])
        cp_constraints.append(cp.sum(w) <= constraints["max_sum"])
    cp_constraints.extend([w >= constraints["min"].values, w <= constraints["max"].values])

    # Drawdown tracking
    u = cp.Variable(T + 1)
    cum_ret = cp.Variable(T + 1)
    d = cp.Variable(T)
    cp_constraints.append(cum_ret[0] == 0)
    cp_constraints.append(u[0] == 0)
    for t in range(T):
        cp_constraints.append(cum_ret[t+1] == cum_ret[t] + (R[t] * scale) @ w)
        cp_constraints.append(u[t+1] >= cum_ret[t+1])
        cp_constraints.append(u[t+1] >= u[t])
        cp_constraints.append(d[t] == u[t+1] - cum_ret[t+1])

    # RLVaR primal formulation applied to d
    t_rlvar = cp.Variable()
    z_rlvar = cp.Variable(nonneg=True)
    psi = cp.Variable(T)
    theta = cp.Variable(T)
    epsilon = cp.Variable(T)
    omega = cp.Variable(T)
    
    ln_k = ((1 / (alpha * T)) ** kappa - (1 / (alpha * T)) ** (-kappa)) / (2 * kappa)
    cp_constraints.append(d - t_rlvar + epsilon + omega <= 0)
    
    x1 = cp.vstack([z_rlvar * (1 + kappa) / (2 * kappa)] * T).flatten(order="C")
    y1 = psi * (1 + kappa) / kappa
    cp_constraints.append(cp.PowCone3D(x1, y1, epsilon, 1 / (1 + kappa)))
    
    x2 = omega / (1 - kappa)
    y2 = theta / kappa
    z2 = cp.vstack([-z_rlvar / (2 * kappa)] * T).flatten(order="C")
    cp_constraints.append(cp.PowCone3D(x2, y2, z2, 1 - kappa))
    
    obj = t_rlvar + z_rlvar * ln_k + cp.sum(psi + theta)
    prob = cp.Problem(cp.Minimize(obj), cp_constraints)
    
    try:
        prob.solve(solver=cp.SCS, eps=1e-5)
    except Exception:
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            prob.solve()

    return {"status": prob.status, "weights": w.value, "obj_value": prob.value / scale if w.value is not None else None}


def solve_portfolio_cvxpy(
    R: np.ndarray,
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """
    Standard MVO solver using CVXPY.
    """
    # Reuse solve_mvo logic
    return solve_mvo(moments, constraints, objectives, **kwargs)


def solve_noc(
    R: np.ndarray,
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    T, n = R.shape
    mu = moments.get("mu", np.mean(R, axis=0)).flatten()
    sigma = moments.get("sigma", np.cov(R, rowvar=False))

    bins = kwargs.get("bins")
    if bins is None:
        bins = T / n if T > n else 20.0

    w = cp.Variable(n)
    base_constraints = [
        cp.sum(w) >= constraints["min_sum"],
        cp.sum(w) <= constraints["max_sum"],
        w >= constraints["min"].values,
        w <= constraints["max"].values,
    ]

    prob_min = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), base_constraints)
    prob_min.solve(verbose=False)
    if prob_min.status not in ["optimal", "feasible"]:
        return {
            "status": "failed",
            "weights": None,
            "message": "Min risk anchor failed",
        }
    w_min = w.value
    rk_min = prob_min.value
    rt_min = w_min @ mu

    prob_max = cp.Problem(cp.Maximize(w @ mu), base_constraints)
    prob_max.solve(verbose=False)
    if prob_max.status not in ["optimal", "feasible"]:
        return {
            "status": "failed",
            "weights": None,
            "message": "Max return anchor failed",
        }
    w_max = w.value
    rt_max = prob_max.value
    rk_max = cp.quad_form(w_max, sigma).value

    res_opt = solve_mvo(moments, constraints, objectives)
    if res_opt["status"] not in ["optimal", "feasible"]:
        return {"status": "failed", "weights": None, "message": "Target anchor failed"}
    w_opt = res_opt["weights"]
    rk_opt = w_opt.T @ sigma @ w_opt
    rt_opt = w_opt @ mu

    rk_delta = (rk_max - rk_min) / bins
    rt_delta = (rt_max - rt_min) / bins
    rk_limit = rk_opt + rk_delta
    rt_limit = rt_opt - rt_delta

    centering_constraints = base_constraints + [
        cp.quad_form(w, sigma) <= rk_limit,
        w @ mu >= rt_limit,
    ]

    lb = constraints["min"].values
    ub = constraints["max"].values

    log_args = [
        cp.reshape(rk_limit - cp.quad_form(w, sigma), (1,), order="C"),
        cp.reshape(w @ mu - rt_limit, (1,), order="C"),
    ]
    for i in range(n):
        if ub[i] > lb[i] + 1e-12:
            log_args.append(cp.reshape(w[i] - lb[i], (1,), order="C"))
            log_args.append(cp.reshape(ub[i] - w[i], (1,), order="C"))

    vec_args = cp.vstack(log_args)
    centering_obj = cp.Maximize(cp.sum(cp.log(vec_args)))

    prob_noc = cp.Problem(centering_obj, centering_constraints)
    try:
        prob_noc.solve(verbose=False)
    except Exception as e:
        return {"status": "failed", "weights": None, "message": str(e)}

    if prob_noc.status not in ["optimal", "optimal_inaccurate", "feasible"]:
        return {"status": prob_noc.status, "weights": None}

    return {"status": prob_noc.status, "weights": w.value, "obj_value": prob_noc.value}


def solve_cla(
    moments: Dict[str, Any],
    constraints: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    from .cla import CLA

    mu = moments["mu"].flatten()
    sigma = moments["sigma"]
    lb = constraints["min"].values
    ub = constraints["max"].values

    cla = CLA(mu, sigma, lb, ub)
    cla.solve()

    return_obj = next(
        (o for o in objectives if o["type"] in ["return", "return_objective"]), None
    )
    risk_obj = next(
        (o for o in objectives if o["type"] in ["risk", "portfolio_risk_objective"]),
        None,
    )

    if return_obj and risk_obj:
        rf = risk_obj.get("arguments", {}).get("risk_free_rate", 0.0)
        weights = cla.max_sharpe(risk_free_rate=rf)
    elif return_obj:
        weights = cla.w[0].flatten()
    else:
        weights = cla.min_volatility()

    return {"status": "optimal", "weights": weights, "cla_object": cla}
