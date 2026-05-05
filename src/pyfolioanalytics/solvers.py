from typing import Any

import cvxpy as cp
import numpy as np
from scipy.optimize import differential_evolution, minimize

from .risk import hhi as _hhi


def create_penalized_objective(
    moments: dict,
    constraints: dict,
    objectives: list,
    R=None
):
    from .optimize import calculate_objective_measures
    n = len(moments["mu"])

    def objective_fn(w):
        measures = calculate_objective_measures(w, moments, objectives, R=R)
        out = 0.0
        PENALTY = 1e4
        TOLERANCE = 1.49e-8  # Approx .Machine$double.eps^0.5

        for obj in objectives:
            if not obj.get("enabled", True):
                continue
            mult = obj.get("multiplier", 1.0)

            # (1) Return Objective
            if obj["type"] == "return":
                val = measures.get(obj["name"], 0.0)
                if "target" in obj and obj["target"] is not None:
                    out += PENALTY * abs(mult) * abs(val - obj["target"])
                out += mult * val

            # (2) Risk / Turnover Objective
            elif obj["type"] in ["risk", "turnover"]:
                val = measures.get(obj["name"], 0.0)
                if "target" in obj and obj["target"] is not None:
                    out += PENALTY * abs(mult) * abs(val - obj["target"])
                out += abs(mult) * val  # R uses abs() for risk multipliers

            # (3) Risk Budget Objective
            elif obj["type"] == "risk_budget":
                rc_name = f"pct_contrib_{obj['name']}"
                if rc_name in measures:
                    pct_rc = measures[rc_name] * 100.0  # R uses percentages (0-100)
                    # min/max risk budget limits
                    if "max_prisk" in obj and obj["max_prisk"] is not None:
                        violations = np.maximum(0, pct_rc - obj["max_prisk"])
                        out += PENALTY * mult * np.sum(violations)
                    if "min_prisk" in obj and obj["min_prisk"] is not None:
                        violations = np.maximum(0, obj["min_prisk"] - pct_rc)
                        out += PENALTY * mult * np.sum(violations)

                    # Concentration
                    if obj.get("min_difference"):
                        max_diff = np.sqrt(np.sum((pct_rc / 100.0)**2))
                        out += PENALTY * mult * max_diff
                    if obj.get("min_concentration"):
                        act_hhi = np.sum((pct_rc / 100.0)**2)
                        min_hhi = np.sum(np.full(n, 1.0/n)**2)
                        out += PENALTY * mult * abs(act_hhi - min_hhi)

            # (4) Weight Concentration Objective (HHI)
            # Mirrors R's constrained_objective.R lines 738-752.
            #
            # Path A (no groups): penalty * conc_aversion * HHI_global
            # Path B (groups)   : penalty * sum(conc_aversion_k * HHI_group_k)
            #
            # NOTE: R line 746 has a known bracket-placement bug —
            #   ``length(x > 1)`` is always 1, so the grouped branch fires
            #   even for a scalar conc_aversion.  We use the *intended*
            #   condition ``conc_groups is not None`` instead.
            elif obj["type"] == "weight_concentration":
                conc_aversion = obj.get("conc_aversion", 0.0)
                conc_groups   = obj.get("conc_groups")     # list[list[int]] | None

                result = _hhi(w, groups=conc_groups)       # groups already 0-based

                if conc_groups is None:
                    # Path A: result is a scalar float.
                    out += PENALTY * float(conc_aversion) * float(result)
                else:
                    # Path B: result is {'HHI': float, 'Groups_HHI': ndarray}.
                    group_hhi = result["Groups_HHI"]
                    alpha = np.asarray(conc_aversion, dtype=float)

                    if len(alpha) == len(group_hhi):
                        # Dot product: sum(alpha_k * HHI_group_k)
                        out += PENALTY * float(np.dot(alpha, group_hhi))
                    else:
                        # Dimension mismatch — user configuration error;
                        # fall back to global HHI with the mean aversion.
                        import warnings
                        warnings.warn(
                            f"weight_concentration: 'conc_aversion' length "
                            f"({len(alpha)}) != number of groups "
                            f"({len(group_hhi)}). Falling back to global HHI.",
                            stacklevel=2,
                        )
                        out += PENALTY * float(np.mean(alpha)) * float(result["HHI"])

            # (5) MinMax Objective
            # Mirrors R's constrained_objective (lines 680-689):
            # penalise the measure only when it falls outside [min_val, max_val].
            # Both bounds are optional so a one-sided range is also valid.
            elif obj["type"] == "minmax":
                val = measures.get(obj["name"], 0.0)
                obj_min = obj.get("min_val")
                obj_max = obj.get("max_val")
                mult = obj.get("multiplier", 1.0)
                if obj_min is not None and val < obj_min:
                    out += PENALTY * mult * (obj_min - val)
                if obj_max is not None and val > obj_max:
                    out += PENALTY * mult * (val - obj_max)

        # --- Evaluate Constraints (Penalties) ---

        # (1) Weight Sum
        sum_w = np.sum(w)
        if "max_sum" in constraints and constraints["max_sum"] is not None:
            if sum_w > constraints["max_sum"]:
                out += PENALTY * (sum_w - constraints["max_sum"])
        if "min_sum" in constraints and constraints["min_sum"] is not None:
            if sum_w < constraints["min_sum"]:
                out += PENALTY * (constraints["min_sum"] - sum_w)

        # (2) Position Limit Constraint
        max_pos = constraints.get("max_pos")
        if max_pos is not None:
            nzassets = np.sum(np.abs(w) > TOLERANCE)
            if nzassets > max_pos:
                out += PENALTY * (nzassets - max_pos)

        # (3) Turnover Constraint
        turnover_target = constraints.get("turnover_target")
        w_init = constraints.get("weight_initial")
        if turnover_target is not None and w_init is not None:
            to = np.sum(np.abs(w - w_init))
            # R penalizes if it exceeds +/- 5% of target
            if to < turnover_target * 0.95 or to > turnover_target * 1.05:
                out += PENALTY * abs(to - turnover_target)

        # (4) Transaction Cost
        ptc = constraints.get("ptc")
        if ptc is not None and w_init is not None:
            tc = np.sum(np.abs(w - w_init) * ptc)
            out += tc  # R does NOT multiply by PENALTY for transaction costs, just mult=1

        # (5) Leverage Constraint
        leverage = constraints.get("leverage")
        if leverage is not None:
            lev_w = np.sum(np.abs(w))
            if lev_w > leverage:
                out += (PENALTY / 100.0) * abs(lev_w - leverage)

        return out

    return objective_fn

def _apply_linear_constraints(cp_constraints, w, constraints):
    if "linear_A" in constraints and "linear_b" in constraints:
        for A, b in zip(constraints["linear_A"], constraints["linear_b"]):
            cp_constraints.append(A @ w <= b)
    if "linear_A_eq" in constraints and "linear_b_eq" in constraints:
        for A_eq, b_eq in zip(constraints["linear_A_eq"], constraints["linear_b_eq"]):
            cp_constraints.append(A_eq @ w == b_eq)


def solve_nonlinear(
    moments: dict[str, Any],
    constraints: dict[str, Any],
    objectives: list[dict[str, Any]],
    **kwargs,
) -> dict[str, Any]:
    n = len(moments["mu"])
    R = kwargs.get("R")

    objective_fn = create_penalized_objective(moments, constraints, objectives, R=R)

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
    minimize_opts = {"ftol": 1e-7, "maxiter": 1000}

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
        "status": "optimal"
        if best_res.success
        else ("optimal_inaccurate" if best_res.x is not None else "failed"),
        "weights": best_res.x if best_res.success else None,
        "obj_value": best_res.fun,
    }


def solve_global_heuristic(
    moments: dict[str, Any],
    constraints: dict[str, Any],
    objectives: list[dict[str, Any]],
    method: str = "DEoptim",
    **kwargs,
) -> dict[str, Any]:
    from scipy.optimize import LinearConstraint, dual_annealing


    n = len(moments["mu"])
    bounds = list(zip(constraints["min"].values, constraints["max"].values))
    R = kwargs.get("R")

    objective_fn = create_penalized_objective(moments, constraints, objectives, R=R)

    max_iter = kwargs.get("itermax", 100)

    if method == "GenSA":
        # dual_annealing is Python's highly optimized Generalized Simulated Annealing
        res = dual_annealing(objective_fn, bounds, maxiter=max_iter)
    elif method in ["DEoptim", "PSO"]:
        # We use differential_evolution as a stable and fast alternative/surrogate for PSO
        lc = LinearConstraint(np.ones(n), constraints["min_sum"], constraints["max_sum"])
        res = differential_evolution(
            objective_fn,
            bounds,
            constraints=(lc,),
            maxiter=max_iter,
            popsize=15,
            tol=1e-7,
            polish=True,
        )
    else:
        raise ValueError(f"Unknown heuristic method: {method}")

    w_out = res.x
    sum_w = np.sum(w_out)
    if sum_w > 0 and abs(constraints["min_sum"] - constraints["max_sum"]) < 1e-5:
        w_out = (w_out / sum_w) * constraints["min_sum"]

    return {
        "status": "optimal" if res.success else res.message,
        "weights": w_out,
        "obj_value": res.fun,
    }


def solve_kelly(R: np.ndarray, constraints: dict[str, Any], **kwargs) -> dict[str, Any]:
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
    _apply_linear_constraints(cp_constraints, w, constraints)
    cp_constraints.append(1 + R @ w >= 1e-4)

    objective = cp.Maximize(cp.sum(cp.log(1 + R @ w)) / T)
    prob = cp.Problem(objective, cp_constraints)

    try:
        prob.solve(verbose=False)
    except Exception:
        return {"status": "failed", "weights": None}

    return {"status": prob.status, "weights": w.value, "obj_value": prob.value}


def solve_mdiv(
    moments: dict[str, Any], constraints: dict[str, Any], **kwargs
) -> dict[str, Any]:
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


def solve_noc(
    R: np.ndarray,
    moments: dict[str, Any],
    constraints: dict[str, Any],
    objectives: list[dict[str, Any]],
    **kwargs,
) -> dict[str, Any]:
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

    from .convex_solvers import ConvexOptimizer
    res_opt = ConvexOptimizer(moments, constraints, objectives).solve()
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
    moments: dict[str, Any],
    constraints: dict[str, Any],
    objectives: list[dict[str, Any]],
    **kwargs,
) -> dict[str, Any]:
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


# ── MILP Cardinality Solver ────────────────────────────────────────────────────

# Risk-measure names that MILP cardinality solver can handle.
# Measures not in this set require exponential/power cones that HiGHS cannot
# solve; those will fall back to the two-step heuristic in optimize.py.
_MILP_SUPPORTED_RISKS: frozenset[str] = frozenset(
    {"STDDEV", "VARIANCE", "MV", "CVAR", "ES", "MAD", "SEMIVARIANCE", "SEMISTDDEV"}
)


def solve_milp_cardinality(
    moments: dict[str, Any],
    constraints: dict[str, Any],
    objectives: list[dict[str, Any]],
    max_pos: int,
    R: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact cardinality-constrained portfolio optimisation via MILP/MIQP.

    Introduces binary selection variables ``z_i ∈ {0,1}`` (z_i = 1 means
    asset i is held) and big-M linking constraints to enforce
    ``sum(z_i) <= max_pos`` exactly.  The resulting mixed-integer problem is
    solved by HiGHS (already a project dependency) via CVXPY.

    Supported risk objectives: StdDev / Variance, CVaR / ES, MAD, SemiVar.
    Unsupported objectives (EVaR, RLVaR, …) return ``{'weights': None}`` so
    that the caller can fall back to the two-step heuristic.

    Parameters
    ----------
    moments:
        Dict with keys ``'mu'`` (N,) and ``'sigma'`` (N, N).
    constraints:
        Standard constraints dict produced by ``Portfolio.get_constraints()``.
    objectives:
        List of objective dicts (same format as ``ConvexOptimizer``).
    max_pos:
        Maximum number of non-zero positions.
    R:
        Historical returns array (T, N).  Required for CVaR / MAD objectives.

    Returns
    -------
    dict
        ``{'status': str, 'weights': ndarray | None, 'obj_value': float | None,
           'n_positions': int | None}``.
        ``weights`` is ``None`` if the solver failed or the risk type is
        unsupported (caller should fall back).
    """
    n = len(moments["mu"])
    mu = np.asarray(moments["mu"]).flatten()
    sigma = np.asarray(moments["sigma"])

    lb = constraints["min"].values.astype(float)
    ub = constraints["max"].values.astype(float)
    min_sum = float(constraints["min_sum"])
    max_sum = float(constraints["max_sum"])

    # ── Identify risk objective ────────────────────────────────────────────────
    risk_obj = next((o for o in objectives if o.get("type") == "risk"), None)
    risk_name = (risk_obj.get("name", "StdDev") if risk_obj else "StdDev").upper()

    # Fast-exit for unsupported cones — caller will fall back to heuristic
    if risk_name not in _MILP_SUPPORTED_RISKS:
        return {
            "status": "unsupported_risk",
            "weights": None,
            "obj_value": None,
            "n_positions": None,
        }

    # ── CVXPY variables ────────────────────────────────────────────────────────
    w = cp.Variable(n, name="weights")
    z = cp.Variable(n, boolean=True, name="selected")

    # ── Constraints ───────────────────────────────────────────────────────────
    cvx_cons: list = []

    # 1. Weight sum
    if abs(min_sum - max_sum) < 1e-8:
        cvx_cons.append(cp.sum(w) == min_sum)
    else:
        cvx_cons.append(cp.sum(w) >= min_sum)
        cvx_cons.append(cp.sum(w) <= max_sum)

    # 2. Big-M linking: w_i ∈ [lb_i * z_i,  ub_i * z_i]
    #    Use cp.multiply() for elementwise multiplication (not *, which CVXPY
    #    1.1+ treats as matrix multiplication and raises a DeprecationWarning).
    #    Using the weight bounds as M avoids artificially large M values, which
    #    would cause numerical ill-conditioning.
    cvx_cons.append(w <= cp.multiply(ub, z))
    if np.any(lb > 1e-12):
        cvx_cons.append(w >= cp.multiply(lb, z))
    else:
        cvx_cons.append(w >= 0.0)

    # 3. Cardinality (core constraint)
    cvx_cons.append(cp.sum(z) <= max_pos)

    # 4. Pass-through: custom linear constraints
    if "linear_A" in constraints and "linear_b" in constraints:
        for A, b in zip(constraints["linear_A"], constraints["linear_b"]):
            cvx_cons.append(np.asarray(A) @ w <= float(b))
    if "linear_A_eq" in constraints and "linear_b_eq" in constraints:
        for A_eq, b_eq in zip(constraints["linear_A_eq"], constraints["linear_b_eq"]):
            cvx_cons.append(np.asarray(A_eq) @ w == float(b_eq))

    # ── Objective ─────────────────────────────────────────────────────────────
    args = risk_obj.get("arguments", {}) if risk_obj else {}

    if risk_name in ("STDDEV", "VARIANCE", "MV"):
        # Minimise portfolio variance (MIQP).  CVXPY + HiGHS handles this via
        # branch-and-bound on the continuous relaxation.
        objective = cp.Minimize(cp.quad_form(w, sigma))

    elif risk_name in ("CVAR", "ES"):
        if R is None:
            return {
                "status": "failed",
                "weights": None,
                "obj_value": None,
                "n_positions": None,
            }
        alpha = float(args.get("alpha", 0.05))
        T = R.shape[0]
        u = cp.Variable(T, nonneg=True, name="cvar_u")
        xi = cp.Variable(name="cvar_xi")
        cvx_cons += [u >= -R @ w - xi]
        objective = cp.Minimize(xi + cp.sum(u) / (alpha * T))

    elif risk_name == "MAD":
        if R is None:
            return {
                "status": "failed",
                "weights": None,
                "obj_value": None,
                "n_positions": None,
            }
        T = R.shape[0]
        mu_r = R.mean(axis=0)          # shape (N,)
        mu_port = mu_r @ w             # scalar CVXPY expression
        R_w = R @ w                    # shape (T,) CVXPY expression
        mad_u = cp.Variable(T, nonneg=True, name="mad_u")
        cvx_cons += [mad_u >= R_w - mu_port, mad_u >= -(R_w - mu_port)]
        objective = cp.Minimize(cp.sum(mad_u) / T)

    elif risk_name in ("SEMIVARIANCE", "SEMISTDDEV"):
        # Minimise downside semi-variance
        target = float(args.get("target", float(mu.mean())))
        if R is not None:
            T = R.shape[0]
            d = cp.Variable(T, nonneg=True, name="semi_d")
            cvx_cons += [d >= target - R @ w]
            objective = cp.Minimize(cp.sum_squares(d) / T)
        else:
            # Approximate with Sigma-based lower semi-variance (Hogan-Warren)
            objective = cp.Minimize(cp.quad_form(w, sigma))

    else:
        # Unreachable given the guard above, but be explicit
        return {
            "status": "unsupported_risk",
            "weights": None,
            "obj_value": None,
            "n_positions": None,
        }

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob = cp.Problem(objective, cvx_cons)

    # Solver preference: HiGHS first (open-source, always available),
    # then commercial solvers if the user has them installed.
    _SOLVERS = ["HIGHS", "GUROBI", "CPLEX", "MOSEK", "SCIP"]
    for solver_name in _SOLVERS:
        try:
            prob.solve(solver=solver_name, verbose=False)
            if w.value is not None:
                break
        except (cp.error.SolverError, Exception):
            continue

    if w.value is None or prob.status in ("infeasible", "unbounded"):
        return {
            "status": prob.status or "failed",
            "weights": None,
            "obj_value": None,
            "n_positions": None,
        }

    # ── Post-processing ───────────────────────────────────────────────────────
    # Clip near-zero weights (MIQP/MILP solvers can leave small numerical residuals)
    w_out = w.value.copy()
    w_out[np.abs(w_out) < 1e-6] = 0.0

    # Re-normalise if fully-invested (sum-to-one) to correct tiny rounding errors
    w_sum = float(np.sum(w_out))
    if w_sum > 1e-10 and abs(min_sum - max_sum) < 1e-8:
        w_out = w_out / w_sum * min_sum

    return {
        "status": "optimal",
        "weights": w_out,
        "obj_value": float(prob.value) if prob.value is not None else None,
        "n_positions": int(np.sum(w_out > 1e-6)),
    }
