from abc import ABC, abstractmethod
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


class ConvexOptimizer:
    """
    Unified Convex Optimization Model for Portfolio Analytics.
    Dispatches to different Risk Models using the Strategy Pattern.
    """

    def __init__(
        self,
        moments: dict[str, Any],
        constraints: dict[str, Any],
        objectives: list[dict[str, Any]],
        R: np.ndarray | None = None,
        **kwargs,
    ):
        self.moments = moments
        self.constraints = constraints
        self.objectives = objectives
        self.R = R
        self.kwargs = kwargs

        self.n = len(moments["mu"])
        self.T = R.shape[0] if R is not None else 0
        self.w = cp.Variable(self.n)

        self.cp_constraints = []
        self.objective_terms = []

        self.mu_robust = moments["mu"].flatten()
        self.ret_uncertainty = 0.0

    def add_constraint(self, constraint):
        self.cp_constraints.append(constraint)

    def _build_base_constraints(self, kappa: Any = 1.0):
        c = self.constraints
        w = self.w  # w acts as y in ratio case

        # 1. Sum constraints
        if abs(c["min_sum"] - c["max_sum"]) < 1e-10:
            self.add_constraint(cp.sum(w) == c["min_sum"] * kappa)
        else:
            self.add_constraint(cp.sum(w) >= c["min_sum"] * kappa)
            self.add_constraint(cp.sum(w) <= c["max_sum"] * kappa)

        # 2. Box constraints
        self.add_constraint(w >= c["min"].values * kappa)
        self.add_constraint(w <= c["max"].values * kappa)

        # 3. Factor Exposure
        if c.get("B") is not None:
            self.add_constraint(c["B"].T @ w >= c["lower"] * kappa)
            self.add_constraint(c["B"].T @ w <= c["upper"] * kappa)

        # 4. Leverage
        if c.get("leverage") is not None:
            self.add_constraint(cp.norm(w, 1) <= c["leverage"] * kappa)

        # 5. Diversification
        if c.get("div_target") is not None:
            div_val = np.sqrt(max(0.0, 1.0 - c["div_target"]))
            self.add_constraint(cp.norm(w, 2) <= div_val * kappa)

        # 6.5 Group Constraints
        if c.get("groups") is not None:
            groups = c["groups"]
            group_min = c.get("group_min")
            group_max = c.get("group_max")
            asset_names = list(c["min"].index)
            for i, group in enumerate(groups):
                indices = [
                    asset_names.index(item) if isinstance(item, str) else item
                    for item in group
                ]
                if group_min is not None:
                    self.add_constraint(cp.sum(w[indices]) >= group_min[i] * kappa)
                if group_max is not None:
                    self.add_constraint(cp.sum(w[indices]) <= group_max[i] * kappa)

        # 6. Linear Constraints
        if "linear_A" in c and "linear_b" in c:
            for A, b in zip(c["linear_A"], c["linear_b"]):
                self.add_constraint(A @ w <= b * kappa)
        if "linear_A_eq" in c and "linear_b_eq" in c:
            for A_eq, b_eq in zip(c["linear_A_eq"], c["linear_b_eq"]):
                self.add_constraint(A_eq @ w == b_eq * kappa)

        # 7. Position Limits (requires MIQP)
        if c.get("max_pos") is not None:
            z = cp.Variable(self.n, boolean=True)
            W_max = np.maximum(1.0, c["max"].values) * 1000.0 if isinstance(kappa, cp.Variable) else np.maximum(1.0, c["max"].values)
            self.add_constraint(w <= cp.multiply(z, W_max))
            self.add_constraint(cp.sum(z) <= c["max_pos"])

        # 8. Tracking Error & Active Share
        te_target = c.get("target")
        te_benchmark = c.get("benchmark")
        if te_target is not None and te_benchmark is not None:
            p_norm = c.get("p_norm", 2)
            w_b = self._get_benchmark_weights(te_benchmark)
            diff = w - kappa * w_b
            
            if p_norm == 2:
                # Traditional L2 Tracking Error scaled by covariance
                sigma = self.moments["sigma"]
                G = np.linalg.cholesky(sigma + 1e-8 * np.eye(self.n)).T
                self.add_constraint(cp.norm(G @ diff, 2) <= te_target * kappa)
            elif p_norm == 1:
                # L1 Tracking Error (Active Share like constraint)
                self.add_constraint(cp.norm(diff, 1) <= te_target * kappa)
            elif p_norm == "inf" or p_norm == np.inf:
                # L-infinity Tracking Error (Maximum single-asset deviation)
                self.add_constraint(cp.norm(diff, "inf") <= te_target * kappa)
            else:
                raise ValueError(f"Unsupported p_norm {p_norm} for tracking_error constraint.")

        as_target = c.get("active_share_target")
        as_benchmark = c.get("active_share_benchmark")
        if as_target is not None and as_benchmark is not None:
            w_b = self._get_benchmark_weights(as_benchmark)
            self.add_constraint(0.5 * cp.norm(w - kappa * w_b, 1) <= as_target * kappa)

        # 9. Turnover and Transaction Costs
        w_init = c.get("weight_initial")
        if w_init is not None:
            turnover_target = c.get("turnover_target")
            if turnover_target is not None:
                self.add_constraint(cp.sum(cp.abs(w - kappa * w_init)) <= turnover_target * kappa)

            ptc = c.get("ptc")
            if ptc is not None:
                tc_penalty = cp.sum(cp.multiply(cp.abs(w - kappa * w_init), ptc))
                self.objective_terms.append(tc_penalty)

        # 10. Robust Return Preparation
        robust_mu_type = c.get("robust_mu_type", "box")
        delta_mu = c.get("delta_mu")
        if delta_mu is not None:
            if robust_mu_type == "box":
                if np.all(c["min"].values >= 0):
                    self.mu_robust = self.mu_robust - delta_mu.values
            elif robust_mu_type == "ellipsoidal" and c.get("sigma_mu") is not None:
                G_mu = np.linalg.cholesky(c["sigma_mu"]).T
                k_mu = c.get("k_mu", 1.0)
                self.ret_uncertainty = k_mu * cp.norm(G_mu @ w)

        min_return = c.get("min_return")
        if min_return is not None:
            self.add_constraint(w @ self.mu_robust - self.ret_uncertainty >= min_return * kappa)

    def _get_benchmark_weights(self, benchmark) -> np.ndarray:
        if isinstance(benchmark, pd.Series):
            return benchmark.values
        if isinstance(benchmark, dict):
            asset_names = list(self.constraints["min"].index)
            return np.array([benchmark.get(name, 0.0) for name in asset_names])
        return benchmark

    def _execute_solve(self, prob: cp.Problem) -> None:
        """
        Robust solver cascade. Attempts to use user-specified solver via kwargs first.
        If unspecified or if it fails, falls back through a sequence of modern, stable solvers.
        """
        if not prob.is_dcp() and not prob.is_dqcp():
            raise ValueError("Problem does not follow Disciplined Convex Programming (DCP) rules.")
            
        solver = self.kwargs.get("solver")
        solver_kwargs = self.kwargs.get("solver_kwargs", {"verbose": False})
        # Default fallback solvers order: CLARABEL (modern interior point, highly stable), ECOS, SCS (good for exponential cones but lower precision)
        fallback_solvers = [cp.CLARABEL, cp.ECOS, cp.SCS]
        
        # 1. If user explicitly specified a solver
        if solver is not None:
            if isinstance(solver, str):
                solver = getattr(cp, solver.upper(), solver)
            try:
                prob.solve(solver=solver, **solver_kwargs)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return
            except Exception as e:
                import warnings
                warnings.warn(f"User-specified solver {solver} failed ({str(e)}), attempting fallback solvers.")
                
        # 2. Cascade through fallback solvers
        for fb_solver in fallback_solvers:
            if fb_solver in cp.installed_solvers():
                try:
                    # SCS requires slightly different kwargs for tolerance sometimes, but standard cvxpy kwargs should pass
                    prob.solve(solver=fb_solver, **solver_kwargs)
                    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        return
                except Exception:
                    continue
                    
        # 3. Ultimate fallback to default CVXPY solve
        try:
            prob.solve(**solver_kwargs)
        except Exception:
            pass

    def solve(self) -> dict[str, Any]:
        return_obj = None
        risk_obj = None

        for obj in self.objectives:
            if not obj.get("enabled", True):
                continue
            if obj["type"] in ["return", "return_objective"]:
                return_obj = obj
            elif obj["type"] in ["risk", "portfolio_risk_objective"]:
                risk_obj = obj

        # Check if we should do Ratio Optimization
        is_ratio_opt = False
        if return_obj and risk_obj:
            ret_target = return_obj.get("target")
            risk_target = risk_obj.get("target")
            if ret_target is None and risk_target is None:
                is_ratio_opt = self.kwargs.get("max_ratio", False)

        if is_ratio_opt:
            return self._solve_ratio(return_obj, risk_obj)

        self._build_base_constraints()

        # Process Risk Objective Term
        risk_term = 0.0
        if risk_obj is not None:
            risk_name = risk_obj.get("name", "StdDev")
            strategy_cls = RISK_STRATEGIES.get(risk_name)
            if not strategy_cls:
                raise ValueError(f"Unsupported convex risk measure: {risk_name}")

            strategy = strategy_cls()
            risk_term = strategy.build(self, risk_obj.get("arguments", {}))

            # Apply risk target as constraint if provided
            risk_target = risk_obj.get("target")
            if risk_target is not None:
                self.add_constraint(risk_term <= risk_target)

        # Build Final Objective Expression
        if return_obj and risk_obj:
            risk_aversion = risk_obj.get("risk_aversion", 1.0)
            ret_target = return_obj.get("target")
            if ret_target is not None:
                # If return target is specified, enforce it as constraint
                self.add_constraint(
                    self.w @ self.mu_robust - self.ret_uncertainty >= ret_target
                )
                self.objective_terms.append(risk_term)
            else:
                # Maximize Return - Risk Penalty
                self.objective_terms.append(
                    0.5 * risk_aversion * risk_term
                    - (self.w @ self.mu_robust - self.ret_uncertainty)
                )
        elif risk_obj:
            self.objective_terms.append(risk_term)
        elif return_obj:
            ret_target = return_obj.get("target")
            if ret_target is not None:
                self.add_constraint(
                    self.w @ self.mu_robust - self.ret_uncertainty >= ret_target
                )
            mult = return_obj.get("multiplier", -1.0)
            if mult < 0:
                self.objective_terms.append(
                    -(self.w @ self.mu_robust - self.ret_uncertainty)
                )
            else:
                self.objective_terms.append(
                    self.w @ self.mu_robust - self.ret_uncertainty
                )
        else:
            self.objective_terms.append(cp.quad_form(self.w, self.moments["sigma"]))

        prob = cp.Problem(cp.Minimize(sum(self.objective_terms)), self.cp_constraints)
        
        try:
            self._execute_solve(prob)
        except Exception as e:
            return {"status": "failed", "weights": None, "message": str(e)}

        return {"status": prob.status, "weights": self.w.value, "obj_value": prob.value}

    def _solve_ratio(self, return_obj: dict[str, Any], risk_obj: dict[str, Any]) -> dict[str, Any]:
        """
        Solve Ratio Optimization (e.g., Sharpe, STARR) using Charnes-Cooper transformation.
        """
        kappa = cp.Variable(nonneg=True)
        # w is now acting as y = kappa * w_real
        self._build_base_constraints(kappa=kappa)

        risk_name = risk_obj.get("name", "StdDev")
        strategy_cls = RISK_STRATEGIES.get(risk_name)
        if not strategy_cls:
            raise ValueError(f"Unsupported convex risk measure: {risk_name}")
        strategy = strategy_cls()
        
        # In ratio optimization, strategy.build(self) receives w acting as y.
        # Since most coherent risk measures are positive homogeneous, Risk(y) = kappa * Risk(w).
        risk_term_y = strategy.build(self, risk_obj.get("arguments", {}))
        
        # Enforce scaled risk <= 1
        self.add_constraint(risk_term_y <= 1.0)

        # Maximize scaled return: y^T * mu - kappa * R_f
        rf = 0.0
        if "arguments" in return_obj and "risk_free_rate" in return_obj["arguments"]:
            rf = return_obj["arguments"]["risk_free_rate"]
        elif "risk_free_rate" in self.kwargs:
            rf = self.kwargs["risk_free_rate"]

        obj_expr = -(self.w @ self.mu_robust - self.ret_uncertainty - kappa * rf)
        
        # Add turnover/ptc penalties scaled appropriately if they were added to objective_terms
        prob = cp.Problem(cp.Minimize(obj_expr + sum(self.objective_terms)), self.cp_constraints)

        try:
            self._execute_solve(prob)
        except Exception as e:
            return {"status": "failed", "weights": None, "message": str(e)}

        if prob.status in ["optimal", "optimal_inaccurate"] and kappa.value is not None and kappa.value > 1e-8:
            real_weights = self.w.value / kappa.value
        else:
            real_weights = self.w.value if self.w.value is not None else np.zeros(self.n)

        return {"status": prob.status, "weights": real_weights, "obj_value": prob.value}


class RiskModelStrategy(ABC):
    @abstractmethod
    def build(
        self, optimizer: ConvexOptimizer, arguments: dict[str, Any]
    ) -> cp.Expression:
        pass


class MeanVarianceStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        sigma = optimizer.moments["sigma"]
        robust_sigma_type = optimizer.constraints.get("robust_sigma_type", "none")
        if (
            robust_sigma_type == "ellipsoidal"
            and optimizer.constraints.get("sigma_sigma") is not None
        ):
            sigma_sigma = optimizer.constraints["sigma_sigma"]
            k_sigma = optimizer.constraints.get("k_sigma", 1.0)
            G_sigma = np.linalg.cholesky(sigma_sigma).T

            W = cp.Variable((optimizer.n, optimizer.n), symmetric=True)
            E = cp.Variable((optimizer.n, optimizer.n), symmetric=True)
            sigma_risk = cp.Variable()

            optimizer.add_constraint(
                cp.norm(G_sigma @ cp.vec(W + E, order="C")) <= sigma_risk
            )
            optimizer.add_constraint(E >> 0)

            L = cp.vstack(
                [
                    cp.hstack([W, cp.reshape(optimizer.w, (optimizer.n, 1), order="C")]),
                    cp.hstack(
                        [cp.reshape(optimizer.w, (1, optimizer.n), order="C"), np.array([[1.0]])]
                    ),
                ]
            )
            optimizer.add_constraint(L >> 0)

            return cp.trace(sigma @ (W + E)) + k_sigma * sigma_risk
        return cp.quad_form(optimizer.w, sigma)


class EVaRStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("EVaR requires historical returns R.")
        T = optimizer.T
        p = arguments.get("p", 0.95)
        alpha = 1.0 - p

        t = cp.Variable()
        z = cp.Variable(nonneg=True)
        ui = cp.Variable(T)

        optimizer.add_constraint(cp.sum(ui) <= T * alpha * z)
        for i in range(T):
            optimizer.add_constraint(cp.ExpCone(-optimizer.R[i] @ optimizer.w - t, z, ui[i]))

        return t


class EDaRStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("EDaR requires historical returns R.")
        T = optimizer.T
        p = arguments.get("p", 0.95)
        alpha = 1.0 - p

        u = cp.Variable(T + 1)
        cum_ret = cp.Variable(T + 1)
        d = cp.Variable(T)

        optimizer.add_constraint(cum_ret[0] == 0)
        optimizer.add_constraint(u[0] == 0)
        for i in range(T):
            optimizer.add_constraint(cum_ret[i + 1] == cum_ret[i] + optimizer.R[i] @ optimizer.w)
            optimizer.add_constraint(u[i + 1] >= cum_ret[i + 1])
            optimizer.add_constraint(u[i + 1] >= u[i])
            optimizer.add_constraint(d[i] == u[i + 1] - cum_ret[i + 1])

        t = cp.Variable()
        z = cp.Variable(nonneg=True)
        ui = cp.Variable(T)

        optimizer.add_constraint(cp.sum(ui) <= T * alpha * z)
        for i in range(T):
            optimizer.add_constraint(cp.ExpCone(d[i] - t, z, ui[i]))

        return t


class MADStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("MAD requires historical returns R.")
        mu_vec = np.mean(optimizer.R, axis=0)
        T = optimizer.T
        y = cp.Variable(T)
        dev = optimizer.R @ optimizer.w - mu_vec @ optimizer.w
        optimizer.add_constraint(y >= dev)
        optimizer.add_constraint(y >= -dev)
        return cp.sum(y) / T


class SemiMADStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("semi_MAD requires historical returns R.")
        mu_vec = np.mean(optimizer.R, axis=0)
        T = optimizer.T
        y = cp.Variable(T)
        dev = optimizer.R @ optimizer.w - mu_vec @ optimizer.w
        optimizer.add_constraint(y >= -dev)
        optimizer.add_constraint(y >= 0)
        return cp.sum(y) / T


class OWAStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("OWA requires historical returns R.")
        T = optimizer.T
        owa_weights = arguments.get("owa_weights")
        if owa_weights is None:
            from .risk import owa_gmd_weights

            owa_weights = owa_gmd_weights(T)

        if len(owa_weights) != T:
            raise ValueError(f"owa_weights must have length {T}")

        if np.any(np.diff(owa_weights) > 1e-12):
            owa_weights = np.sort(owa_weights)[::-1]

        delta_w = owa_weights[:-1] - owa_weights[1:]

        if T > 1:
            zeta = cp.Variable(T - 1)
            d = cp.Variable((T, T - 1), nonneg=True)
            losses = -optimizer.R @ optimizer.w
            for k in range(1, T):
                optimizer.add_constraint(d[:, k - 1] >= losses - zeta[k - 1])

            top_k_sums = [(k * zeta[k - 1] + cp.sum(d[:, k - 1])) for k in range(1, T)]
            owa_expr = cp.sum(
                [delta_w[i] * top_k_sums[i] for i in range(T - 1)]
            ) + owa_weights[-1] * cp.sum(losses)
        else:
            owa_expr = owa_weights[0] * (-optimizer.R @ optimizer.w)

        return owa_expr


# Mapping from name to Strategy class
RISK_STRATEGIES: dict[str, type[RiskModelStrategy]] = {
    "StdDev": MeanVarianceStrategy,
    "var": MeanVarianceStrategy,
    "EVaR": EVaRStrategy,
    "EDaR": EDaRStrategy,
    "MAD": MADStrategy,
    "semi_MAD": SemiMADStrategy,
    "OWA": OWAStrategy,
}


class RLVaRStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("RLVaR requires historical returns R.")
        T = optimizer.T
        alpha_p = arguments.get("p", 0.95)
        kappa = arguments.get("kappa", 0.3)
        alpha = 1.0 - alpha_p

        t = cp.Variable()
        z = cp.Variable(nonneg=True)
        psi = cp.Variable(T)
        theta = cp.Variable(T)
        epsilon = cp.Variable(T)
        omega = cp.Variable(T)

        scale = 100.0
        losses = -(optimizer.R * scale) @ optimizer.w

        ln_k = ((1 / (alpha * T)) ** kappa - (1 / (alpha * T)) ** (-kappa)) / (
            2 * kappa
        )

        optimizer.add_constraint(losses - t + epsilon + omega <= 0)

        x1 = cp.vstack([z * (1 + kappa) / (2 * kappa)] * T).flatten(order="C")
        y1 = psi * (1 + kappa) / kappa
        optimizer.add_constraint(cp.PowCone3D(x1, y1, epsilon, 1 / (1 + kappa)))

        x2 = omega / (1 - kappa)
        y2 = theta / kappa
        z2 = cp.vstack([-z / (2 * kappa)] * T).flatten(order="C")
        optimizer.add_constraint(cp.PowCone3D(x2, y2, z2, 1 - kappa))

        return (t + z * ln_k + cp.sum(psi + theta)) / scale


class RLDaRStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("RLDaR requires historical returns R.")
        T = optimizer.T
        alpha_p = arguments.get("p", 0.95)
        kappa = arguments.get("kappa", 0.3)
        alpha = 1.0 - alpha_p

        scale = 100.0
        u = cp.Variable(T + 1)
        cum_ret = cp.Variable(T + 1)
        d = cp.Variable(T)

        optimizer.add_constraint(cum_ret[0] == 0)
        optimizer.add_constraint(u[0] == 0)
        for i in range(T):
            optimizer.add_constraint(
                cum_ret[i + 1] == cum_ret[i] + (optimizer.R[i] * scale) @ optimizer.w
            )
            optimizer.add_constraint(u[i + 1] >= cum_ret[i + 1])
            optimizer.add_constraint(u[i + 1] >= u[i])
            optimizer.add_constraint(d[i] == u[i + 1] - cum_ret[i + 1])

        t_rlvar = cp.Variable()
        z_rlvar = cp.Variable(nonneg=True)
        psi = cp.Variable(T)
        theta = cp.Variable(T)
        epsilon = cp.Variable(T)
        omega = cp.Variable(T)

        ln_k = ((1 / (alpha * T)) ** kappa - (1 / (alpha * T)) ** (-kappa)) / (
            2 * kappa
        )
        optimizer.add_constraint(d - t_rlvar + epsilon + omega <= 0)

        x1 = cp.vstack([z_rlvar * (1 + kappa) / (2 * kappa)] * T).flatten(order="C")
        y1 = psi * (1 + kappa) / kappa
        optimizer.add_constraint(cp.PowCone3D(x1, y1, epsilon, 1 / (1 + kappa)))

        x2 = omega / (1 - kappa)
        y2 = theta / kappa
        z2 = cp.vstack([-z_rlvar / (2 * kappa)] * T).flatten(order="C")
        optimizer.add_constraint(cp.PowCone3D(x2, y2, z2, 1 - kappa))

        return (t_rlvar + z_rlvar * ln_k + cp.sum(psi + theta)) / scale


class CVaRStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("CVaR/ES requires historical returns R.")
        T = optimizer.T
        p = arguments.get("p", 0.95)
        alpha = 1.0 - p

        t = cp.Variable()
        u = cp.Variable(T)

        for i in range(T):
            optimizer.add_constraint(u[i] >= -optimizer.R[i] @ optimizer.w - t)

        optimizer.add_constraint(u >= 0)
        return t + cp.sum(u) / (T * alpha)


class UlcerIndexStrategy(RiskModelStrategy):
    def build(self, optimizer: ConvexOptimizer, arguments: dict[str, Any]) -> cp.Expression:
        if optimizer.R is None:
            raise ValueError("Ulcer Index (UCI) requires historical returns R.")
        T = optimizer.T

        u = cp.Variable(T + 1)
        cum_ret = cp.Variable(T + 1)
        d = cp.Variable(T)

        optimizer.add_constraint(cum_ret[0] == 0)
        optimizer.add_constraint(u[0] == 0)
        for i in range(T):
            optimizer.add_constraint(cum_ret[i + 1] == cum_ret[i] + optimizer.R[i] @ optimizer.w)
            optimizer.add_constraint(u[i + 1] >= cum_ret[i + 1])
            optimizer.add_constraint(u[i + 1] >= u[i])
            optimizer.add_constraint(d[i] == u[i + 1] - cum_ret[i + 1])

        return cp.norm(d, 2) / np.sqrt(T)

RISK_STRATEGIES["UCI"] = UlcerIndexStrategy
RISK_STRATEGIES["RLVaR"] = RLVaRStrategy
RISK_STRATEGIES["RLDaR"] = RLDaRStrategy
RISK_STRATEGIES["CVaR"] = CVaRStrategy
RISK_STRATEGIES["ES"] = CVaRStrategy
