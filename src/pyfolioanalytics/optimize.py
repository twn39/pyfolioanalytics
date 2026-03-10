import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .portfolio import Portfolio
from .moments import set_portfolio_moments, M3_MM, M4_MM
from .risk import (
    VaR,
    ES,
    EVaR,
    risk_contribution,
    max_drawdown,
    average_drawdown,
    CDaR,
    EDaR,
    RLVaR,
    RLDaR,
    l_moment,
    owa_risk,
    owa_gmd_weights,
    owa_l_moment_crm_weights,
)
from .ml import hrp_optimization, herc_optimization, nco_optimization
from .solvers import (
    solve_portfolio_cvxpy,
    solve_kelly,
    solve_mdiv,
    solve_noc,
    solve_cla,
    solve_owa,
    solve_edar,
    solve_rlvar,
    solve_rldar,
    solve_evar,
)


def calculate_objective_measures(
    weights: np.ndarray,
    moments: Dict[str, Any],
    objectives: List[Dict[str, Any]],
    R: Optional[np.ndarray] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    measures = {}
    mu = moments.get("mu")
    sigma = moments.get("sigma")
    m3 = moments.get("m3")
    m4 = moments.get("m4")

    if mu is not None and len(mu) == len(weights):
        measures["mean"] = np.dot(weights, mu).item()
    if sigma is not None and sigma.shape[0] == len(weights):
        p_var = np.dot(weights.T, np.dot(sigma, weights))
        p_sd = np.sqrt(max(0, float(p_var)))
        measures["sd"] = p_sd
        measures["StdDev"] = p_sd

    for obj in objectives:
        if not obj.get("enabled", True):
            continue
        obj_name = obj["name"]
        obj_type = obj.get("type")
        obj_args = obj.get("arguments", {})
        
        if obj_name == "VaR":
            measures[obj_name] = VaR(weights, mu, sigma, m3, m4, **obj_args)
        elif obj_name == "ES":
            measures[obj_name] = ES(weights, mu, sigma, m3, m4, **obj_args)
        elif obj_name == "EVaR" and R is not None:
            measures[obj_name] = EVaR(weights, R, **obj_args)
        elif obj_name == "EDaR" and R is not None:
            measures[obj_name] = EDaR(weights, R, **obj_args)
        elif obj_name == "RLVaR" and R is not None:
            measures[obj_name] = RLVaR(weights, R, **obj_args)
        elif obj_name == "RLDaR" and R is not None:
            measures[obj_name] = RLDaR(weights, R, **obj_args)
        elif obj_name == "L_Moment_CRM" and R is not None:
            T = R.shape[0]
            w_owa = owa_l_moment_crm_weights(T, **obj_args)
            measures[obj_name] = owa_risk(weights, R, w_owa)
        elif obj_name == "max_drawdown" and R is not None:
            measures[obj_name] = max_drawdown(weights, R)
        elif obj_name == "average_drawdown" and R is not None:
            measures[obj_name] = average_drawdown(weights, R)
        elif obj_name == "CDaR" and R is not None:
            measures[obj_name] = CDaR(weights, R, **obj_args)
        elif obj_name == "OWA" and R is not None:
            owa_weights = obj_args.get("owa_weights")
            if owa_weights is None:
                owa_weights = owa_gmd_weights(R.shape[0])
            measures[obj_name] = owa_risk(weights, R, owa_weights)
        
        # Track tracking error for objectives or constraints
        te_target = obj_args.get("target") or (constraints.get("target") if constraints and (obj_name == "tracking_error" or obj_type == "tracking_error") else None)
        te_benchmark = obj_args.get("benchmark") or (constraints.get("benchmark") if constraints else None)
        
        if (obj_name == "tracking_error" or obj_type == "tracking_error" or (obj_name == "StdDev" and te_benchmark is not None)) and sigma is not None:
            if te_benchmark is not None:
                w_b = te_benchmark
                if isinstance(w_b, dict) and constraints is not None:
                    asset_names = list(constraints["min"].index)
                    w_b = np.array([w_b.get(name, 0.0) for name in asset_names])
                elif isinstance(w_b, pd.Series):
                    w_b = w_b.values
                
                if isinstance(w_b, (list, np.ndarray)) and len(w_b) == len(weights):
                    diff = weights - w_b
                    te_var = np.dot(diff.T, np.dot(sigma, diff))
                    measures["tracking_error"] = np.sqrt(max(0, float(te_var)))

        if obj_type == "risk_budget" and sigma is not None:
            rc = risk_contribution(weights, sigma)
            measures["risk_contribution_" + obj_name] = rc
            measures["pct_contrib_" + obj_name] = rc / np.sum(rc)

    if constraints is not None:
        if "weight_initial" in constraints and "ptc" in constraints:
            w_init = constraints["weight_initial"]
            ptc = constraints["ptc"]
            measures["transaction_cost"] = np.sum(np.abs(weights - w_init) * ptc)
        if "weight_initial" in constraints and "turnover_target" in constraints:
            w_init = constraints["weight_initial"]
            measures["turnover"] = np.sum(np.abs(weights - w_init))

    return measures


def optimize_portfolio(
    R: pd.DataFrame, portfolio: Portfolio, optimize_method: str = "ROI", **kwargs
) -> Dict[str, Any]:
    # 1. Dispatch Multi-Layer
    if hasattr(portfolio, "sub_portfolios") and len(getattr(portfolio, "sub_portfolios", {})) > 0:
        return optimize_portfolio_multi_layer(R, portfolio, **kwargs)

    # 2. Setup Moments
    moment_method = kwargs.get("moment_method", "sample")
    moments = set_portfolio_moments(R, portfolio, method=moment_method, **kwargs)

    # 3. Setup Constraints
    constraints = portfolio.get_constraints()
    for k in ["delta_mu", "robust_mu_type", "sigma_mu", "k_mu", "robust_sigma_type", "sigma_sigma", "k_sigma"]:
        if k in kwargs: constraints[k] = kwargs[k]

    # 4. Specialized ML methods
    if optimize_method == "HRP":
        w_hrp = hrp_optimization(R, **kwargs)
        return {"weights": w_hrp, "objective_measures": calculate_objective_measures(w_hrp.values, moments, portfolio.objectives, R=R.values, constraints=constraints), "status": "optimal", "moments": moments, "portfolio": portfolio}
    if optimize_method == "HERC":
        w_herc = herc_optimization(R, **kwargs)
        return {"weights": w_herc, "objective_measures": calculate_objective_measures(w_herc.values, moments, portfolio.objectives, R=R.values, constraints=constraints), "status": "optimal", "moments": moments, "portfolio": portfolio}
    if optimize_method == "NCO":
        w_nco = nco_optimization(R, **kwargs)
        return {"weights": w_nco, "objective_measures": calculate_objective_measures(w_nco.values, moments, portfolio.objectives, R=R.values, constraints=constraints), "status": "optimal", "moments": moments, "portfolio": portfolio}

    # 5. Direct optimization for specific measures (only if NO risk budget)
    result = None
    enabled_objs = [obj for obj in portfolio.objectives if obj.get("enabled", True)]
    # IMPORTANT: Risk budget must use solve_mvo/solve_portfolio_cvxpy
    has_risk_budget = any(obj.get("type") == "risk_budget" for obj in enabled_objs)
    
    if not has_risk_budget:
        if any(obj["name"] == "EVaR" for obj in enabled_objs):
            result = solve_evar(R.values, constraints, portfolio.objectives, **kwargs)
        elif any(obj["name"] == "OWA" for obj in enabled_objs):
            result = solve_owa(R.values, constraints, portfolio.objectives, **kwargs)
        elif any(obj["name"] == "EDaR" for obj in enabled_objs):
            result = solve_edar(R.values, constraints, portfolio.objectives, **kwargs)
        elif any(obj["name"] == "RLVaR" for obj in enabled_objs):
            result = solve_rlvar(R.values, constraints, portfolio.objectives, **kwargs)
        elif any(obj["name"] == "RLDaR" for obj in enabled_objs):
            result = solve_rldar(R.values, constraints, portfolio.objectives, **kwargs)
        elif any(obj["name"] == "L_Moment_CRM" for obj in enabled_objs):
            obj_conf = next(o for o in portfolio.objectives if o["name"] == "L_Moment_CRM")
            w_owa = owa_l_moment_crm_weights(R.shape[0], **obj_conf.get("arguments", {}))
            import copy
            new_objectives = copy.deepcopy(portfolio.objectives)
            for o in new_objectives:
                if o["name"] == "L_Moment_CRM":
                    o["name"] = "OWA"
                    o["arguments"]["owa_weights"] = w_owa
            result = solve_owa(R.values, constraints, new_objectives, **kwargs)

    if result is None:
        if optimize_method == "Kelly": result = solve_kelly(R.values, constraints, **kwargs)
        elif optimize_method == "MDIV": result = solve_mdiv(moments, constraints, **kwargs)
        elif optimize_method == "NOC": result = solve_noc(R.values, moments, constraints, portfolio.objectives, **kwargs)
        elif optimize_method == "CLA": result = solve_cla(moments, constraints, portfolio.objectives, **kwargs)
        else: 
            # Risk budget or standard MVO
            from .solvers import solve_mvo
            result = solve_mvo(moments, constraints, portfolio.objectives, **kwargs)

    if result.get("status") in ["optimal", "feasible", "optimal_inaccurate"]:
        w = result["weights"]
        assets_keys = list(portfolio.assets.keys())
        return {
            "weights": pd.Series(w, index=assets_keys),
            "objective_measures": calculate_objective_measures(w, moments, portfolio.objectives, R=R.values, constraints=constraints),
            "status": result["status"], "moments": moments, "portfolio": portfolio,
        }
    else:
        return {"status": result.get("status", "failed"), "message": result.get("message", "Optimization failed"), "moments": moments, "portfolio": portfolio}


def optimize_portfolio_multi_layer(
    R: pd.DataFrame, portfolio: Any, **kwargs
) -> Dict[str, Any]:
    sub_results = {}
    sub_returns = {}
    for meta_asset, sub_port in portfolio.sub_portfolios.items():
        res = optimize_portfolio(R, sub_port, **kwargs)
        sub_results[meta_asset] = res
        sub_returns[meta_asset] = R[list(sub_port.assets.keys())] @ res["weights"]

    meta_R = pd.DataFrame(sub_returns)
    other_assets = [a for a in portfolio.root.assets.keys() if a not in portfolio.sub_portfolios]
    if other_assets:
        meta_R = pd.concat([meta_R, R[other_assets]], axis=1)

    root_res = optimize_portfolio(meta_R, portfolio.root, **kwargs)
    
    final_weights = pd.Series(0.0, index=R.columns)
    root_weights = root_res["weights"]

    for meta_asset, w_meta in root_weights.items():
        if meta_asset in sub_results:
            w_sub = sub_results[meta_asset]["weights"]
            final_weights.loc[w_sub.index] += w_sub * w_meta
        else:
            final_weights.loc[meta_asset] += w_meta

    full_assets_port = Portfolio(assets=list(R.columns))
    moments = set_portfolio_moments(R, full_assets_port)
    measures = calculate_objective_measures(final_weights.values, moments, portfolio.root.objectives, R=R.values)
    
    return {"weights": final_weights, "objective_measures": measures, "root_result": root_res, "sub_results": sub_results, "status": root_res["status"], "portfolio": portfolio}


def create_efficient_frontier(
    R: pd.DataFrame, portfolio: Portfolio, n_portfolios: int = 10, **kwargs
) -> pd.DataFrame:
    port_min = portfolio.copy().clear_objectives().add_objective(type="risk", name="StdDev")
    res_min = optimize_portfolio(R, port_min, **kwargs)
    if res_min["status"] not in ["optimal", "feasible", "optimal_inaccurate"]: raise ValueError("Min risk portfolio failed")

    port_max = portfolio.copy().clear_objectives().add_objective(type="return")
    res_max = optimize_portfolio(R, port_max, **kwargs)
    if res_max["status"] not in ["optimal", "feasible", "optimal_inaccurate"]: raise ValueError("Max return portfolio failed")

    target_returns = np.linspace(res_min["objective_measures"]["mean"], res_max["objective_measures"]["mean"], n_portfolios)
    frontier_data = []
    for ret in target_returns:
        port_tmp = portfolio.copy().clear_objectives().add_objective(type="risk", name="StdDev").add_objective(type="return", name="mean", target=ret)
        res = optimize_portfolio(R, port_tmp, **kwargs)
        if res["status"] in ["optimal", "feasible", "optimal_inaccurate"]:
            row = res["objective_measures"].copy()
            for asset, weight in res["weights"].items(): row[asset] = weight
            frontier_data.append(row)
    return pd.DataFrame(frontier_data)
