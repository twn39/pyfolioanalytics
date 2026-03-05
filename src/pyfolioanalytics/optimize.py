import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from .portfolio import Portfolio, MultLayerPortfolio
from .moments import set_portfolio_moments
from .solvers import solve_mvo, solve_nonlinear, solve_deoptim, solve_evar
from .risk import VaR, ES, risk_contribution, max_drawdown, CDaR, EVaR
from .random_portfolios import random_portfolios
from .ml import hrp_optimization, herc_optimization, nco_optimization

def calculate_objective_measures(
    weights: np.ndarray, 
    moments: Dict[str, Any], 
    objectives: List[Dict[str, Any]],
    R: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    mu = moments["mu"]
    sigma = moments["sigma"]
    m3 = moments.get("m3")
    m4 = moments.get("m4")
    measures = {}
    measures["mean"] = np.dot(weights, mu).item()
    p_var = np.dot(weights.T, np.dot(sigma, weights))
    measures["StdDev"] = np.sqrt(p_var)
    for obj in objectives:
        obj_name = obj["name"]
        obj_type = obj["type"]
        if obj_name in ["VaR", "mVaR"]:
            p = obj.get("arguments", {}).get("p", 0.95)
            method = "modified" if obj_name == "mVaR" or obj.get("arguments", {}).get("method") == "modified" else "gaussian"
            measures[obj_name] = VaR(weights, mu, sigma, m3, m4, p=p, method=method)
        elif obj_name in ["ES", "mES", "ETL", "mETL"]:
            p = obj.get("arguments", {}).get("p", 0.95)
            method = obj.get("arguments", {}).get("method", "gaussian")
            measures[obj_name] = ES(weights, mu, sigma, m3, m4, p=p, method=method)
        elif obj_name == "EVaR" and R is not None:
            p = obj.get("arguments", {}).get("p", 0.95)
            measures[obj_name] = EVaR(weights, R, p=p)
        elif obj_name == "MaxDrawdown" and R is not None:
            measures[obj_name] = max_drawdown(weights, R)
        elif obj_name == "AverageDrawdown" and R is not None:
            from .risk import average_drawdown
            measures[obj_name] = average_drawdown(weights, R)
        elif obj_name == "CDaR" and R is not None:
            p = obj.get("arguments", {}).get("p", 0.95)
            measures[obj_name] = CDaR(weights, R, p=p)
        if obj_type == "risk_budget":
            rc_name = obj_name if obj_name in ["StdDev", "var"] else "StdDev"
            rc = risk_contribution(weights, sigma, name=rc_name)
            measures[f"pct_contrib_{obj_name}"] = rc / np.sum(rc)
    return measures

def equal_weight(R: pd.DataFrame, portfolio: Portfolio, **kwargs) -> Dict[str, Any]:
    nassets = len(portfolio.assets)
    constraints = portfolio.get_constraints()
    max_sum = constraints["max_sum"]
    weights = np.full(nassets, max_sum / nassets)
    moments = set_portfolio_moments(R, portfolio)
    measures = calculate_objective_measures(weights, moments, portfolio.objectives, R=R.values)
    return {"weights": pd.Series(weights, index=portfolio.assets.keys()), "objective_measures": measures, "moments": moments, "portfolio": portfolio}

def inverse_volatility_weight(R: pd.DataFrame, portfolio: Portfolio, **kwargs) -> Dict[str, Any]:
    nassets = len(portfolio.assets)
    constraints = portfolio.get_constraints()
    max_sum = constraints["max_sum"]
    asset_names = list(portfolio.assets.keys())
    vols = R[asset_names].std()
    inv_vols = 1.0 / vols
    weights = (inv_vols / inv_vols.sum()) * max_sum
    moments = set_portfolio_moments(R, portfolio)
    measures = calculate_objective_measures(weights.values, moments, portfolio.objectives, R=R.values)
    return {"weights": pd.Series(weights, index=portfolio.assets.keys()), "objective_measures": measures, "moments": moments, "portfolio": portfolio}

def optimize_portfolio(R: pd.DataFrame, portfolio: Union[Portfolio, MultLayerPortfolio], optimize_method: str = "ROI", **kwargs) -> Dict[str, Any]:
    if isinstance(portfolio, MultLayerPortfolio):
        sub_weights = {}
        sub_returns = {}
        for meta_asset, sub_port in portfolio.sub_portfolios.items():
            res_sub = optimize_portfolio(R, sub_port, optimize_method=optimize_method, **kwargs)
            if res_sub["weights"] is not None:
                sub_weights[meta_asset] = res_sub["weights"]
                sub_returns[meta_asset] = R[list(sub_port.assets.keys())] @ res_sub["weights"].values
            else:
                return {"status": f"Sub-portfolio {meta_asset} failed", "weights": None}
        R_meta = pd.DataFrame(sub_returns)
        remaining_assets = [a for a in portfolio.root.assets.keys() if a not in portfolio.sub_portfolios]
        if remaining_assets: R_meta = pd.concat([R_meta, R[remaining_assets]], axis=1)
        res_root = optimize_portfolio(R_meta, portfolio.root, optimize_method=optimize_method, **kwargs)
        if res_root["weights"] is not None:
            final_weights = {asset: 0.0 for asset in R.columns}
            for meta_asset, w_meta in res_root["weights"].items():
                if meta_asset in sub_weights:
                    for asset, w_sub in sub_weights[meta_asset].items(): final_weights[asset] += w_meta * w_sub
                else:
                    final_weights[meta_asset] += w_meta
            w_series = pd.Series(final_weights)
            moments = set_portfolio_moments(R, Portfolio(assets=list(final_weights.keys())))
            measures = calculate_objective_measures(w_series.values, moments, portfolio.root.objectives, R=R.values)
            return {"weights": w_series, "objective_measures": measures, "status": res_root["status"]}
        else:
            return res_root

    moments = set_portfolio_moments(R, portfolio)
    constraints = portfolio.get_constraints()
    objectives = portfolio.objectives
    
    if optimize_method == "random":
        search_size = kwargs.get("search_size", 1000)
        rp = random_portfolios(portfolio, permutations=search_size)
        best_w = None
        min_obj = np.inf
        target_obj = next((o for o in objectives if o.get("enabled", True)), None)
        for i in range(rp.shape[0]):
            w = rp[i, :]
            measures = calculate_objective_measures(w, moments, objectives, R=R.values)
            if target_obj:
                val = measures.get(target_obj["name"], measures.get("StdDev"))
                mult = target_obj.get("multiplier", 1.0)
                score = val * mult
                if score < min_obj:
                    min_obj = score
                    best_w = w
        if best_w is not None:
            weights = pd.Series(best_w, index=portfolio.assets.keys())
            measures = calculate_objective_measures(best_w, moments, objectives, R=R.values)
            return {"weights": weights, "objective_measures": measures, "status": "optimal", "moments": moments, "portfolio": portfolio}
        else:
            return {"status": "failed", "weights": None, "portfolio": portfolio}

    if optimize_method == "HRP":
        w_hrp = hrp_optimization(R, **kwargs)
        measures = calculate_objective_measures(w_hrp.values, moments, objectives, R=R.values)
        return {"weights": w_hrp, "objective_measures": measures, "status": "optimal", "moments": moments, "portfolio": portfolio}

    if optimize_method == "HERC":
        w_herc = herc_optimization(R, **kwargs)
        measures = calculate_objective_measures(w_herc.values, moments, objectives, R=R.values)
        return {"weights": w_herc, "objective_measures": measures, "status": "optimal", "moments": moments, "portfolio": portfolio}

    if optimize_method == "NCO":
        w_nco = nco_optimization(R, **kwargs)
        measures = calculate_objective_measures(w_nco.values, moments, objectives, R=R.values)
        return {"weights": w_nco, "objective_measures": measures, "status": "optimal", "moments": moments, "portfolio": portfolio}

    # New: Direct EVaR optimization via CVXPY
    if any(obj["name"] == "EVaR" for obj in objectives if obj.get("enabled", True)):
        result = solve_evar(R.values, constraints, objectives, **kwargs)
    elif optimize_method == "Kelly":
        from .solvers import solve_kelly
        result = solve_kelly(R.values, constraints, **kwargs)
    elif optimize_method == "MDIV":
        from .solvers import solve_mdiv
        result = solve_mdiv(moments, constraints, **kwargs)
    else:
        has_risk_budget = any(obj["type"] == "risk_budget" for obj in objectives if obj.get("enabled", True))
        if has_risk_budget or optimize_method == "DEoptim":
            if optimize_method == "DEoptim":
                result = solve_deoptim(moments, constraints, objectives, R=R.values, **kwargs)
            else:
                result = solve_nonlinear(moments, constraints, objectives, R=R.values, **kwargs)
        else:
            result = solve_mvo(moments, constraints, objectives, **kwargs)
    
    if result["weights"] is not None:
        weights = pd.Series(result["weights"], index=portfolio.assets.keys())
        measures = calculate_objective_measures(weights.values, moments, objectives, R=R.values)
        return {"weights": weights, "objective_measures": measures, "status": result["status"], "moments": moments, "portfolio": portfolio}
    else:
        return {"status": result["status"], "weights": None, "portfolio": portfolio}

def create_efficient_frontier(R: pd.DataFrame, portfolio: Portfolio, n_portfolios: int = 25, optimize_method: str = "ROI", **kwargs) -> pd.DataFrame:
    p_min_risk = portfolio.copy()
    p_min_risk.clear_objectives()
    p_min_risk.add_objective(type="risk", name="StdDev")
    res_min = optimize_portfolio(R, p_min_risk, optimize_method=optimize_method)
    p_max_ret = portfolio.copy()
    p_max_ret.clear_objectives()
    p_max_ret.add_objective(type="return", name="mean", multiplier=-1.0)
    res_max = optimize_portfolio(R, p_max_ret, optimize_method=optimize_method)
    if res_min["weights"] is None or res_max["weights"] is None: raise ValueError("Could not bound frontier.")
    min_ret = res_min["objective_measures"]["mean"]
    max_ret = res_max["objective_measures"]["mean"]
    target_returns = np.linspace(min_ret, max_ret, n_portfolios)
    frontier_data = []
    for target in target_returns:
        p_tmp = portfolio.copy()
        p_tmp.clear_objectives()
        p_tmp.add_constraint(type="return", min_return=target)
        p_tmp.add_objective(type="risk", name="StdDev")
        res = optimize_portfolio(R, p_tmp, optimize_method=optimize_method)
        if res["weights"] is not None:
            row = {"mean": res["objective_measures"]["mean"], "StdDev": res["objective_measures"]["StdDev"]}
            for asset, weight in res["weights"].items(): row[asset] = weight
            frontier_data.append(row)
    return pd.DataFrame(frontier_data)
