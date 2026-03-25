from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def entropy_prog(
    p: np.ndarray,
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    Aeq: np.ndarray | None = None,
    beq: np.ndarray | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Entropy pooling program for blending views on scenarios with a prior scenario-probability distribution.

    Equivalent to PortfolioAnalytics::EntropyProg in R.
    Finds posterior probabilities p_ that minimize KL divergence from prior p,
    subject to linear constraints: A @ p_ <= b and Aeq @ p_ = beq.

    Parameters
    ----------
    p : np.ndarray
        Prior probabilities (must sum to 1).
    A : np.ndarray, optional
        Inequality constraint matrix.
    b : np.ndarray, optional
        Inequality constraint vector.
    Aeq : np.ndarray, optional
        Equality constraint matrix.
    beq : np.ndarray, optional
        Equality constraint vector.
    verbose : bool
        If True, print optimization status.

    Returns
    -------
    dict
        {'p_': revised probabilities, 'optimizationPerformance': info}
    """
    p = p.reshape(-1, 1)
    J = len(p)

    # Validate sum(p) == 1
    if not (0.999 < np.sum(p) < 1.001):
        raise ValueError("Sum of prior probabilities must equal 1")

    # Handle empty constraints
    k_ineq = A.shape[0] if A is not None and A.size > 0 else 0
    k_eq = Aeq.shape[0] if Aeq is not None and Aeq.size > 0 else 0

    if k_ineq + k_eq == 0:
        raise ValueError(
            "At least one equality or inequality constraint must be specified"
        )

    # Starting guess for dual variables (Lagrange multipliers)
    x0 = np.zeros(k_ineq + k_eq)

    def dual_objective(x):
        l = x[:k_ineq].reshape(-1, 1) if k_ineq > 0 else None
        v = x[k_ineq:].reshape(-1, 1) if k_eq > 0 else None

        exponent = np.zeros((J, 1))
        if l is not None:
            exponent -= A.T @ l
        if v is not None:
            exponent -= Aeq.T @ v

        log_p = np.log(np.maximum(p, 1e-32))
        x_p = np.exp(log_p - 1 + exponent)
        x_p = np.maximum(x_p, 1e-32)

        # We want to maximize L = -sum(x_p) - l'b - v'beq
        # So we minimize -L = sum(x_p) + l'b + v'beq
        obj = np.sum(x_p)
        if l is not None:
            obj += (l.T @ b.reshape(-1, 1))[0, 0]
        if v is not None:
            obj += (v.T @ beq.reshape(-1, 1))[0, 0]

        return obj

    def dual_gradient(x):
        l = x[:k_ineq].reshape(-1, 1) if k_ineq > 0 else None
        v = x[k_ineq:].reshape(-1, 1) if k_eq > 0 else None

        exponent = np.zeros((J, 1))
        if l is not None:
            exponent -= A.T @ l
        if v is not None:
            exponent -= Aeq.T @ v

        log_p = np.log(np.maximum(p, 1e-32))
        x_p = np.exp(log_p - 1 + exponent)
        x_p = np.maximum(x_p, 1e-32)

        # Grad of -L w.r.t l: -(A*x_p - b) = b - A*x_p
        # Grad of -L w.r.t v: -(Aeq*x_p - beq) = beq - Aeq*x_p
        grad = []
        if l is not None:
            grad.append((b.reshape(-1, 1) - A @ x_p).flatten())
        if v is not None:
            grad.append((beq.reshape(-1, 1) - Aeq @ x_p).flatten())

        return np.concatenate(grad)

    # Inequality multipliers must be >= 0
    bounds = [(0, None)] * k_ineq + [(None, None)] * k_eq

    res = minimize(
        dual_objective,
        x0,
        jac=dual_gradient,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-12,
        options={"ftol": 1e-12, "gtol": 1e-12},
    )

    if not res.success:
        if verbose:
            print(f"Entropy pooling optimization did not converge: {res.message}")
        # Even if not success, we follow R behavior: if it fails, it might stop or return partial
        # But we want to be robust.

    # Recover posterior probabilities
    l = res.x[:k_ineq].reshape(-1, 1) if k_ineq > 0 else None
    v = res.x[k_ineq:].reshape(-1, 1) if k_eq > 0 else None

    exponent = np.zeros((J, 1))
    if l is not None:
        exponent -= A.T @ l
    if v is not None:
        exponent -= Aeq.T @ v

    log_p = np.log(np.maximum(p, 1e-32))
    p_post = np.exp(log_p - 1 + exponent)

    sum_p = np.sum(p_post)
    if sum_p > 1e-12:
        p_post = p_post / sum_p
    else:
        # Optimization failed catastrophically, fallback to prior
        if verbose:
            print(
                "Entropy pooling sum of probabilities is near zero. Falling back to prior."
            )
        p_post = p.flatten()

    return {
        "p_": p_post.flatten(),
        "optimizationPerformance": {
            "converged": res.success and (sum_p > 1e-12),
            "iterations": res.nit,
            "sumOfProbabilities": np.sum(p_post),
        },
    }


def centroid_ranking(N: int) -> np.ndarray:
    """
    Computes Almgren-Chriss centroids for asset ranking.

    Parameters
    ----------
    N : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Centroid vector in descending order (highest ranking first).
    """
    # c_n = 1/n * sum_{i=n}^N (1/i)
    c = np.zeros(N)
    inv_i = 1.0 / np.arange(1, N + 1)
    for n in range(1, N + 1):
        c[n - 1] = np.mean(inv_i[n - 1 :])
    return c


def meucci_ranking(
    R: pd.DataFrame, order: list[int | str], p: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """
    Asset Ranking using Meucci's Entropy Pooling.

    Equivalent to PortfolioAnalytics::meucci.ranking.
    """
    if isinstance(R, pd.DataFrame):
        asset_names = list(R.columns)
        X = R.values
    else:
        X = np.asanyarray(R)
        asset_names = [str(i) for i in range(X.shape[1])]

    J, N = X.shape
    if p is None:
        p = np.full(J, 1.0 / J)

    # Map order to indices if names are provided
    order_idx = [asset_names.index(o) if isinstance(o, str) else o for o in order]
    k = len(order_idx)

    # Equality constraints: sum(p) = 1
    Aeq = np.ones((1, J))
    beq = np.array([1.0])

    # Inequality constraints: E[R_{order[i]}] <= E[R_{order[i+1]}]
    # A * p_ <= 0  => sum(p_j * (R_{j, order[i]} - R_{j, order[i+1]})) <= 0
    V = X[:, order_idx[:-1]] - X[:, order_idx[1:]]
    A = V.T  # (k-1) x J
    b = np.zeros(k - 1)

    res = entropy_prog(p, A=A, b=b, Aeq=Aeq, beq=beq)
    p_post = res["p_"]

    return meucci_moments(X, p_post)


def meucci_views(
    R: pd.DataFrame, views: list[dict[str, Any]], p: np.ndarray | None = None
) -> np.ndarray:
    """
    Automate the generation of constraints for Meucci's Entropy Pooling based on high-level view descriptions.

    Each view in 'views' is a dict:
    - {'type': 'relative', 'asset_high': 'A', 'asset_low': 'B'} -> E[R_A] > E[R_B]
    - {'type': 'absolute', 'asset': 'A', 'value': 0.02} -> E[R_A] = 0.02
    - {'type': 'inequality', 'asset': 'A', 'value': 0.05, 'direction': 'less'} -> E[R_A] <= 0.05
    """
    if isinstance(R, pd.DataFrame):
        asset_names = list(R.columns)
        X = R.values
    else:
        X = np.asanyarray(R)
        asset_names = [str(i) for i in range(X.shape[1])]

    J, N = X.shape
    if p is None:
        p = np.full(J, 1.0 / J)

    Aeq_list = [np.ones((1, J))]  # sum(p) = 1
    beq_list = [1.0]

    Aineq_list = []
    bineq_list = []

    for view in views:
        v_type = view["type"]
        if v_type == "relative":
            idx_h = asset_names.index(view["asset_high"])
            idx_l = asset_names.index(view["asset_low"])
            # E[R_h - R_l] >= 0  => sum(p_j * (R_jl - R_jh)) <= 0
            Aineq_list.append((X[:, idx_l] - X[:, idx_h]).reshape(1, -1))
            bineq_list.append(0.0)

        elif v_type == "absolute":
            idx = asset_names.index(view["asset"])
            Aeq_list.append(X[:, idx].reshape(1, -1))
            beq_list.append(view["value"])

        elif v_type == "inequality":
            idx = asset_names.index(view["asset"])
            direction = view.get("direction", "less")
            if direction == "less":
                Aineq_list.append(X[:, idx].reshape(1, -1))
                bineq_list.append(view["value"])
            else:
                Aineq_list.append(-X[:, idx].reshape(1, -1))
                bineq_list.append(-view["value"])

    Aeq = np.vstack(Aeq_list)
    beq = np.array(beq_list)

    Aineq = np.vstack(Aineq_list) if Aineq_list else None
    bineq = np.array(bineq_list) if bineq_list else None

    res = entropy_prog(p, A=Aineq, b=bineq, Aeq=Aeq, beq=beq)
    return res["p_"]


def meucci_moments(R: np.ndarray, posterior_probs: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calculate adjusted mean and covariance based on posterior probabilities.
    """
    p = posterior_probs.reshape(-1, 1)
    T, N = R.shape

    # Posterior Mean
    mu_p = R.T @ p

    # Posterior Covariance
    # Sigma = sum(p_t * (R_t - mu)(R_t - mu)')
    R_centered = R - mu_p.T
    sigma_p = (R_centered.T * p.flatten()) @ R_centered

    return {"mu": mu_p.flatten(), "sigma": sigma_p}


# Maintain backward compatibility for original names
def entropy_pooling(
    prior_probs: np.ndarray,
    Aeq: np.ndarray | None = None,
    beq: np.ndarray | None = None,
    Aineq: np.ndarray | None = None,
    bineq: np.ndarray | None = None,
) -> np.ndarray:
    """Deprecated: use entropy_prog instead."""
    res = entropy_prog(prior_probs, A=Aineq, b=bineq, Aeq=Aeq, beq=beq)
    return res["p_"]
