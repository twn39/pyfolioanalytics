from typing import Any

import numpy as np
import pandas as pd

from .convex_solvers import RISK_STRATEGIES, ConvexOptimizer
from .ml import herc_optimization, hrp_optimization, nco_optimization
from .moments import set_portfolio_moments
from .portfolio import Portfolio
from .random_portfolios import random_portfolios
from .risk import (
    ES,
    MAD,
    CDaR,
    EDaR,
    EVaR,
    RLDaR,
    RLVaR,
    VaR,
    average_drawdown,
    max_drawdown,
    owa_gmd_weights,
    owa_l_moment_crm_weights,
    owa_risk,
    risk_contribution,
    semi_MAD,
)
from .solvers import (
    solve_cla,
    solve_kelly,
    solve_mdiv,
    solve_noc,
    solve_nonlinear,
)


def calculate_objective_measures(
    weights: np.ndarray,
    moments: dict[str, Any],
    objectives: list[dict[str, Any]],
    R: np.ndarray | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, float]:
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

        if obj_name == "VaR" and mu is not None and sigma is not None:
            measures[obj_name] = VaR(weights, mu, sigma, m3, m4, **obj_args)
        elif obj_name == "ES" and mu is not None and sigma is not None:
            measures[obj_name] = ES(weights, mu, sigma, m3, m4, **obj_args)
        elif obj_name == "EVaR" and R is not None:
            measures[obj_name] = EVaR(weights, R, **obj_args)
        elif obj_name == "EDaR" and R is not None:
            measures[obj_name] = EDaR(weights, R, **obj_args)
        elif obj_name == "MAD" and R is not None:
            measures[obj_name] = MAD(weights, R)
        elif obj_name == "semi_MAD" and R is not None:
            measures[obj_name] = semi_MAD(weights, R)
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
        te_benchmark = obj_args.get("benchmark") or (
            constraints.get("benchmark") if constraints else None
        )

        if (
            obj_name == "tracking_error"
            or obj_type == "tracking_error"
            or (obj_name == "StdDev" and te_benchmark is not None)
        ) and sigma is not None:
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

        if obj_type == "risk_budget":
            if obj_name in ["StdDev", "Variance"] and sigma is not None:
                rc = risk_contribution(weights, sigma)
            else:
                import pyfolioanalytics.risk as pr

                from .risk import numerical_risk_contribution

                func = getattr(pr, obj_name, None)
                if func is None:
                    # Fallback to StdDev if not found or something else
                    rc = (
                        risk_contribution(weights, sigma)
                        if sigma is not None
                        else np.zeros_like(weights)
                    )
                else:
                    if R is None:
                        raise ValueError(
                            f"Historical returns R must be provided for alternative risk parity using {obj_name}"
                        )
                    rc = numerical_risk_contribution(weights, R, func, **obj_args)

            measures["risk_contribution_" + obj_name] = rc
            sum_rc = np.sum(rc)
            if sum_rc > 1e-12:
                measures["pct_contrib_" + obj_name] = rc / sum_rc
            else:
                measures["pct_contrib_" + obj_name] = np.zeros_like(rc)

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
    R: pd.DataFrame, portfolio: Portfolio | Any, optimize_method: str = "ROI", **kwargs
) -> dict[str, Any]:
    # 1. Dispatch Multi-Layer
    if (
        hasattr(portfolio, "sub_portfolios")
        and len(getattr(portfolio, "sub_portfolios", {})) > 0
    ):
        return optimize_portfolio_multi_layer(R, portfolio, **kwargs)

    # 2. Setup Moments
    moment_method = kwargs.get("moment_method", "sample")
    moments = set_portfolio_moments(R, portfolio, method=moment_method, **kwargs)

    # 3. Setup Constraints
    constraints = portfolio.get_constraints()
    for k in [
        "delta_mu",
        "robust_mu_type",
        "sigma_mu",
        "k_mu",
        "robust_sigma_type",
        "sigma_sigma",
        "k_sigma",
    ]:
        if k in kwargs:
            constraints[k] = kwargs[k]

    # 4. Specialized ML methods
    if optimize_method == "HRP":
        w_hrp = hrp_optimization(R, **kwargs)
        return {
            "weights": w_hrp,
            "objective_measures": calculate_objective_measures(
                w_hrp.values,
                moments,
                portfolio.objectives,
                R=R.values,
                constraints=constraints,
            ),
            "status": "optimal",
            "moments": moments,
            "portfolio": portfolio,
        }
    if optimize_method == "HERC":
        w_herc = herc_optimization(R, **kwargs)
        return {
            "weights": w_herc,
            "objective_measures": calculate_objective_measures(
                w_herc.values,
                moments,
                portfolio.objectives,
                R=R.values,
                constraints=constraints,
            ),
            "status": "optimal",
            "moments": moments,
            "portfolio": portfolio,
        }
    if optimize_method == "NCO":
        w_nco = nco_optimization(R, **kwargs)
        return {
            "weights": w_nco,
            "objective_measures": calculate_objective_measures(
                w_nco.values,
                moments,
                portfolio.objectives,
                R=R.values,
                constraints=constraints,
            ),
            "status": "optimal",
            "moments": moments,
            "portfolio": portfolio,
        }

    # 5. Random Portfolios Engine
    if optimize_method == "random":
        rp_kwargs = kwargs.copy()
        permutations = rp_kwargs.pop("permutations", 2000)
        rp_method = rp_kwargs.pop("rp_method", "transform")
        # generate random portfolios
        rp_weights = random_portfolios(
            portfolio, permutations=permutations, method=rp_method, **rp_kwargs
        )

        if len(rp_weights) == 0:
            return {
                "weights": None,
                "status": "infeasible",
                "moments": moments,
                "portfolio": portfolio,
            }

        best_score = float("inf")
        best_w = None
        best_measures = {}
        R_vals = R.values if R is not None else None

        enabled_objs = [obj for obj in portfolio.objectives if obj.get("enabled", True)]

        for w in rp_weights:
            measures = calculate_objective_measures(
                w, moments, enabled_objs, R=R_vals, constraints=constraints
            )

            # Penalize constraint violations
            penalty = 0.0

            # Position Limit Penalty
            if "max_pos" in constraints:
                pos_count = np.sum(w > 1e-6)
                if pos_count > constraints["max_pos"]:
                    penalty += (pos_count - constraints["max_pos"]) * 1e4

            # Score objective
            score = penalty
            for obj in enabled_objs:
                mult = obj.get("multiplier", 1.0)
                val = measures.get(obj["name"], 0.0)
                target = obj.get("target")
                if target is not None:
                    # Target penalty: (val - target)^2
                    score += mult * (val - target) ** 2
                else:
                    score += mult * val

            if score < best_score:
                best_score = score
                best_w = w
                best_measures = measures

        final_w = pd.Series(best_w, index=R.columns)
        return {
            "weights": final_w,
            "objective_measures": best_measures,
            "status": "optimal",
            "moments": moments,
            "portfolio": portfolio,
        }

    # 6. Direct optimization for specific measures
    result = None
    enabled_objs = [obj for obj in portfolio.objectives if obj.get("enabled", True)]
    has_risk_budget = any(obj.get("type") == "risk_budget" for obj in enabled_objs)

    if not has_risk_budget and optimize_method not in [
        "Kelly",
        "MDIV",
        "NOC",
        "CLA",
        "random",
        "HRP",
        "HERC",
        "NCO",
    ]:
        risk_obj = next(
            (
                o
                for o in enabled_objs
                if o.get("type") in ["risk", "portfolio_risk_objective"]
            ),
            None,
        )
        risk_name = risk_obj.get("name", "StdDev") if risk_obj else "StdDev"

        if risk_name == "L_Moment_CRM":
            import copy

            from .risk import owa_l_moment_crm_weights

            w_owa = owa_l_moment_crm_weights(
                R.shape[0], **risk_obj.get("arguments", {}) if risk_obj and risk_obj.get("arguments") else {}
            )
            opt_portfolio = copy.deepcopy(portfolio)
            for o in opt_portfolio.objectives:
                if o.get("name") == "L_Moment_CRM":
                    o["name"] = "OWA"
                    if "arguments" not in o:
                        o["arguments"] = {}
                    o["arguments"]["owa_weights"] = w_owa
            risk_name = "OWA"
            opt_objs = opt_portfolio.objectives
        else:
            opt_objs = portfolio.objectives

        if risk_name in RISK_STRATEGIES or risk_name in ["var"]:
            opt = ConvexOptimizer(
                moments,
                constraints,
                opt_objs,
                R=R.values if R is not None else None,
                **kwargs,
            )
            result = opt.solve()

    if result is None:
        if optimize_method in ["DEoptim", "GenSA", "PSO"]:
            from .solvers import solve_global_heuristic
            result = solve_global_heuristic(
                moments, constraints, portfolio.objectives, method=optimize_method, R=R.values if R is not None else None, **kwargs
            )
        elif optimize_method == "Kelly":
            result = solve_kelly(R.values, constraints, **kwargs)
        elif optimize_method == "MDIV":
            result = solve_mdiv(moments, constraints, **kwargs)
        elif optimize_method == "NOC":
            result = solve_noc(
                R.values, moments, constraints, portfolio.objectives, **kwargs
            )
        elif optimize_method == "CLA":
            result = solve_cla(moments, constraints, portfolio.objectives, **kwargs)
        elif has_risk_budget:
            result = solve_nonlinear(
                moments,
                constraints,
                portfolio.objectives,
                R=R.values if R is not None else None,
                **kwargs,
            )
        else:
            # Fallback (should be covered by ConvexOptimizer now)
            opt = ConvexOptimizer(
                moments,
                constraints,
                opt_objs,
                R=R.values if R is not None else None,
                **kwargs,
            )
            result = opt.solve()

    if result.get("status") in ["optimal", "feasible", "optimal_inaccurate"]:
        w = result["weights"]
        assets_keys = list(portfolio.assets.keys())
        return {
            "weights": pd.Series(w, index=assets_keys),
            "objective_measures": calculate_objective_measures(
                w, moments, portfolio.objectives, R=R.values, constraints=constraints
            ),
            "status": result["status"],
            "moments": moments,
            "portfolio": portfolio,
        }
    else:
        return {
            "status": result.get("status", "failed"),
            "message": result.get("message", "Optimization failed"),
            "moments": moments,
            "portfolio": portfolio,
        }


def optimize_portfolio_multi_layer(
    R: pd.DataFrame, portfolio: Any, **kwargs
) -> dict[str, Any]:
    sub_results = {}
    sub_returns = {}
    for meta_asset, sub_port in portfolio.sub_portfolios.items():
        res = optimize_portfolio(R, sub_port, **kwargs)
        sub_results[meta_asset] = res
        # Multiply underlying asset returns by their weights in the sub-portfolio
        leaf_assets = list(res["weights"].index)
        sub_returns[meta_asset] = R[leaf_assets] @ res["weights"]

    meta_R = pd.DataFrame(sub_returns)
    other_assets = [
        a for a in portfolio.root.assets.keys() if a not in portfolio.sub_portfolios
    ]
    if other_assets:
        meta_R = pd.concat([meta_R, R[other_assets]], axis=1)

    root_res = optimize_portfolio(meta_R, portfolio.root, **kwargs)

    # final_weights should accumulate weights for all leaf assets found in R
    final_weights = pd.Series(0.0, index=R.columns)
    root_weights = root_res["weights"]

    for meta_asset, w_meta in root_weights.items():
        if meta_asset in sub_results:
            w_sub = sub_results[meta_asset]["weights"]
            # Add scaled sub-weights to the final weights
            for asset, w in w_sub.items():
                if asset in final_weights.index:
                    final_weights.loc[asset] += w * w_meta
        else:
            if meta_asset in final_weights.index:
                final_weights.loc[meta_asset] += w_meta

    full_assets_port = Portfolio(assets=list(R.columns))
    moments = set_portfolio_moments(R, full_assets_port)
    measures = calculate_objective_measures(
        final_weights.values, moments, portfolio.root.objectives, R=R.values
    )

    return {
        "weights": final_weights,
        "objective_measures": measures,
        "root_result": root_res,
        "sub_results": sub_results,
        "status": root_res["status"],
        "portfolio": portfolio,
    }


def create_efficient_frontier(
    R: pd.DataFrame, portfolio: Portfolio, n_portfolios: int = 10, **kwargs
) -> pd.DataFrame:
    port_min = (
        portfolio.copy().clear_objectives().add_objective(type="risk", name="StdDev")
    )
    res_min = optimize_portfolio(R, port_min, **kwargs)
    if res_min["status"] not in ["optimal", "feasible", "optimal_inaccurate"]:
        raise ValueError("Min risk portfolio failed")

    port_max = portfolio.copy().clear_objectives().add_objective(type="return")
    res_max = optimize_portfolio(R, port_max, **kwargs)
    if res_max["status"] not in ["optimal", "feasible", "optimal_inaccurate"]:
        raise ValueError("Max return portfolio failed")

    target_returns = np.linspace(
        res_min["objective_measures"]["mean"],
        res_max["objective_measures"]["mean"],
        n_portfolios,
    )
    frontier_data = []
    for ret in target_returns:
        port_tmp = (
            portfolio.copy()
            .clear_objectives()
            .add_objective(type="risk", name="StdDev")
            .add_objective(type="return", name="mean", target=ret)
        )
        res = optimize_portfolio(R, port_tmp, **kwargs)
        if res["status"] in ["optimal", "feasible", "optimal_inaccurate"]:
            row = res["objective_measures"].copy()
            for asset, weight in res["weights"].items():
                row[asset] = weight
            frontier_data.append(row)
    return pd.DataFrame(frontier_data)
