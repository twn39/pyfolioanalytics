from dataclasses import fields as _dc_fields
from typing import Any

import numpy as np
import pandas as pd

from .convex_solvers import RISK_STRATEGIES, ConvexOptimizer
from .ml import herc_optimization, hrp_optimization, nco_optimization
from .moments import MomentConfig, set_portfolio_moments
from .portfolio import Portfolio, SubPortfolioConfig
from .random_portfolios import random_portfolios
from .risk import (
    ES,
    LPM,
    MAD,
    UCI,
    CDaR,
    EDaR,
    EVaR,
    RLDaR,
    RLVaR,
    VaR,
    average_drawdown,
    hhi,
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

# Pre-computed set of MomentConfig field names for O(1) kwargs filtering.
# Built once at import time; avoids per-call overhead in optimize_portfolio.
_MOMENT_CONFIG_KEYS: frozenset[str] = frozenset(
    f.name for f in _dc_fields(MomentConfig)
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
        elif obj_name == "LPM" and R is not None:
            measures[obj_name] = LPM(weights, R, **obj_args)
        elif obj_name == "UCI" and R is not None:
            measures[obj_name] = UCI(weights, R, **obj_args)
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


def _check_returns(R: pd.DataFrame | np.ndarray | None):
    if R is None:
        return
    if isinstance(R, pd.DataFrame):
        vals = R.values
    else:
        vals = R

    if np.isnan(vals).any():
        raise ValueError("Historical returns 'R' contain NaN values. Please clean or impute missing data before optimization.")
    if np.isinf(vals).any():
        raise ValueError("Historical returns 'R' contain infinite values.")


def optimize_portfolio(
    R: pd.DataFrame, portfolio: Portfolio | Any, optimize_method: str = "ROI", **kwargs
) -> dict[str, Any]:
    _check_returns(R)

    # 1. Dispatch Multi-Layer
    if (
        hasattr(portfolio, "sub_portfolios")
        and len(getattr(portfolio, "sub_portfolios", {})) > 0
    ):
        # Forward optimize_method explicitly so the root portfolio uses it;
        # before this fix optimize_method was silently dropped here.
        return optimize_portfolio_multi_layer(
            R, portfolio, optimize_method=optimize_method, **kwargs
        )

    # 2. Setup Moments
    # Pre-filter kwargs to MomentConfig fields so that solver/optimizer
    # kwargs (solver, itermax, permutations, …) never reach the moment
    # estimator.  Responsibility for separation lives here at the call
    # boundary, not inside MomentConfig.
    moment_method = kwargs.get("moment_method", "sample")
    moment_kwargs = {k: v for k, v in kwargs.items() if k in _MOMENT_CONFIG_KEYS}
    moment_config = MomentConfig(method=moment_method, **moment_kwargs)
    moments = set_portfolio_moments(R, portfolio, config=moment_config)

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

            # Score objective — mirrors R's constrained_objective scoring.
            score = penalty
            RPENALTY = 1e4
            for obj in enabled_objs:
                mult = obj.get("multiplier", 1.0)
                val = measures.get(obj["name"], 0.0)
                obj_type = obj.get("type", "")
                target = obj.get("target")

                if obj_type == "minmax":
                    # Penalise only when the value escapes [min_val, max_val].
                    obj_min = obj.get("min_val")
                    obj_max = obj.get("max_val")
                    if obj_min is not None and val < obj_min:
                        score += RPENALTY * mult * (obj_min - val)
                    if obj_max is not None and val > obj_max:
                        score += RPENALTY * mult * (val - obj_max)
                elif obj_type == "weight_concentration":
                    # Inline HHI — only weights needed, no moment estimation.
                    # Groups are already 0-based (normalised in add_objective).
                    conc_aversion = obj.get("conc_aversion", 0.0)
                    conc_groups   = obj.get("conc_groups")
                    result = hhi(w, groups=conc_groups)
                    if conc_groups is None:
                        score += RPENALTY * float(conc_aversion) * float(result)
                    else:
                        group_hhi_arr = result["Groups_HHI"]
                        alpha = np.asarray(conc_aversion, dtype=float)
                        if len(alpha) == len(group_hhi_arr):
                            score += RPENALTY * float(np.dot(alpha, group_hhi_arr))
                elif target is not None:
                    score += RPENALTY * abs(mult) * abs(val - target)
                    score += mult * val
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
    opt_objs = portfolio.objectives
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
            max_pos = constraints.get("max_pos")
            assets_keys = list(portfolio.assets.keys())
            R_values = R.values if R is not None else None

            if max_pos is not None and max_pos < len(assets_keys):
                # ── Primary path: exact MILP cardinality solver ───────────────
                # Supported risk types (StdDev, CVaR, MAD, SemiVar) are solved
                # exactly via branch-and-bound (CVXPY + HiGHS).  Unsupported
                # cone types (EVaR, RLVaR, …) return weights=None and we fall
                # through to the two-step heuristic below.
                from .solvers import solve_milp_cardinality
                result = solve_milp_cardinality(
                    moments, constraints, opt_objs,
                    max_pos=max_pos, R=R_values,
                )

                # ── Fallback path: two-step heuristic ────────────────────────
                # Triggered when MILP solver cannot handle the risk type or
                # encounters a numerical failure.
                if result.get("weights") is None:
                    opt = ConvexOptimizer(
                        moments, constraints, opt_objs,
                        R=R_values, **kwargs,
                    )
                    relaxed_result = opt.solve()

                    if relaxed_result.get("status") in ["optimal", "optimal_inaccurate"]:
                        w_relaxed = relaxed_result["weights"]
                        top_indices = np.argsort(w_relaxed)[-max_pos:]

                        locked_constraints = constraints.copy()
                        locked_max = locked_constraints["max"].copy()
                        locked_min = locked_constraints["min"].copy()

                        mask = np.ones(len(assets_keys), dtype=bool)
                        mask[top_indices] = False
                        locked_max.iloc[mask] = locked_min.iloc[mask]
                        locked_constraints["max"] = locked_max

                        opt_locked = ConvexOptimizer(
                            moments, locked_constraints, opt_objs,
                            R=R_values, **kwargs,
                        )
                        result = opt_locked.solve()
                    else:
                        result = relaxed_result
            else:
                opt = ConvexOptimizer(
                    moments,
                    constraints,
                    opt_objs,
                    R=R_values,
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
            # We first attempt exact convex risk parity for cohesive measures (Spinu formulation)
            opt = ConvexOptimizer(
                moments, constraints, opt_objs, R=R.values if R is not None else None, **kwargs
            )
            result = opt.solve()
            if result.get("status") not in ["optimal", "optimal_inaccurate"]:
                # Fallback to SLSQP nonlinear penalty if exact approach fails or measure is not supported
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
    R: pd.DataFrame,
    portfolio: Any,
    optimize_method: str = "ROI",
    **kwargs,
) -> dict[str, Any]:
    """Optimise a multi-layer (hierarchical) portfolio.

    Each sub-portfolio is optimised independently using its own
    ``optimize_method`` and ``search_size`` as stored in its
    ``SubPortfolioConfig``.  The resulting proxy returns are fed into the
    root-level optimisation which uses the ``optimize_method`` supplied to
    this function (forwarded from the top-level ``optimize_portfolio()``
    call).

    Parameters
    ----------
    R:
        Full asset returns DataFrame (all leaf assets as columns).
    portfolio:
        A ``MultLayerPortfolio`` instance.
    optimize_method:
        Optimisation method for the *root* portfolio.  Sub-portfolios use
        their own per-``SubPortfolioConfig`` method.
    **kwargs:
        Additional kwargs passed to the *root* portfolio optimisation only.
        Sub-portfolio kwargs are configured inside ``add_sub_portfolio()``.
    """
    sub_results: dict[str, Any] = {}
    sub_returns: dict[str, pd.Series] = {}

    for meta_asset, config in portfolio.sub_portfolios.items():
        # ── Unpack config, supporting legacy bare-Portfolio storage ──────────
        if isinstance(config, SubPortfolioConfig):
            sub_port        = config.portfolio
            sub_method      = config.optimize_method
            sub_search_size = config.search_size
            sub_kwargs      = dict(config.kwargs)   # per-sub extra kwargs
        else:
            # Legacy: bare Portfolio stored before SubPortfolioConfig existed.
            sub_port        = config
            sub_method      = "ROI"
            sub_search_size = 20_000
            sub_kwargs      = {}

        # ── Subset R to the assets in this sub-portfolio ─────────────────────
        # Mirrors R's ``R[,names(tmp$portfolio$assets)]``.
        # For nested MultLayerPortfolio, use leaf_assets() to resolve the real
        # (non-virtual) asset names — root.assets contains meta-asset names
        # that are not columns of R.
        if hasattr(sub_port, "leaf_assets"):
            # Nested MultLayerPortfolio: recurse to leaf level.
            sub_asset_names = sub_port.leaf_assets()
        else:
            # Plain Portfolio: assets are real leaf assets.
            sub_asset_names = list(sub_port.assets.keys())
        R_sub = R[sub_asset_names]

        # ── Optimise this sub-portfolio with its own independent parameters ──
        # search_size is mapped to `permutations` (the random-engine kwarg).
        res = optimize_portfolio(
            R_sub,
            sub_port,
            optimize_method=sub_method,
            permutations=sub_search_size,   # maps R's search_size → random engine
            **sub_kwargs,
        )
        sub_results[meta_asset] = res

        # Proxy return series = weighted leaf returns of the sub-portfolio.
        w_sub = res["weights"].reindex(sub_asset_names).fillna(0.0)
        sub_returns[meta_asset] = R_sub @ w_sub

    # ── Build meta-asset returns DataFrame for the root optimisation ─────────
    meta_R = pd.DataFrame(sub_returns)
    other_assets = [
        a for a in portfolio.root.assets.keys()
        if a not in portfolio.sub_portfolios
    ]
    if other_assets:
        meta_R = pd.concat([meta_R, R[other_assets]], axis=1)

    # ── Optimise root portfolio with the top-level method ────────────────────
    root_res = optimize_portfolio(
        meta_R, portfolio.root, optimize_method=optimize_method, **kwargs
    )

    # ── Assemble final leaf-level weights ────────────────────────────────────
    final_weights = pd.Series(0.0, index=R.columns)
    for meta_asset, w_meta in root_res["weights"].items():
        if meta_asset in sub_results:
            w_sub = sub_results[meta_asset]["weights"]
            for asset, w in w_sub.items():
                if asset in final_weights.index:
                    final_weights.loc[asset] += w * w_meta
        else:
            if meta_asset in final_weights.index:
                final_weights.loc[meta_asset] += w_meta

    # ── Compute objective measures on final leaf-level weights ───────────────
    full_port = Portfolio(assets=list(R.columns))
    moments = set_portfolio_moments(R, full_port)
    measures = calculate_objective_measures(
        final_weights.values, moments, portfolio.root.objectives, R=R.values
    )

    return {
        "weights":            final_weights,
        "objective_measures": measures,
        "root_result":        root_res,
        "sub_results":        sub_results,
        "status":             root_res["status"],
        "portfolio":          portfolio,
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
