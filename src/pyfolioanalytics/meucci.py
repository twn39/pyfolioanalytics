import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, List, Any, Union


def entropy_pooling(
    prior_probs: np.ndarray,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    Aineq: Optional[np.ndarray] = None,
    bineq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Entropy Pooling algorithm.
    Finds posterior probabilities p that minimize KL divergence from prior q,
    subject to linear constraints on p: Aeq @ p = beq and Aineq @ p <= bineq.
    """
    q = prior_probs.reshape(-1, 1)
    J = len(prior_probs)

    # Standard Aeq for sum(p) = 1
    if Aeq is None:
        Aeq = np.ones((1, J))
        beq = np.array([[1.0]])

    # Dual problem for Entropy Pooling
    # The constraints are on the expectations: E[V] <= m
    # where V is the view matrix (scenarios x views)
    # p * V' <= m
    
    # In this implementation, we use a simpler version focusing on 
    # the constraints being directly on p if provided, or via expectations.
    # For fully flexible views, we typically have constraints on E[X] = sum(p_j * X_j)
    
    k_eq = Aeq.shape[0]
    k_ineq = Aineq.shape[0] if Aineq is not None else 0
    
    x0 = np.zeros(k_eq + k_ineq)
    
    def dual_objective(x):
        lambda_eq = x[:k_eq]
        lambda_ineq = x[k_eq:]
        
        # p_j = q_j * exp(-1 - Aeq' * lambda_eq - Aineq' * lambda_ineq)
        # We need sum(p_j) = 1, but Aeq[0,:] is already ones(1, T) and beq[0] is 1.0.
        # The dual objective for entropy pooling (minimizing sum p*ln(p/q)) is:
        # L(p, lambda) = sum p*ln(p/q) + lambda_eq'(Aeq*p - beq) + lambda_ineq'(Aineq*p - bineq)
        # First order condition: ln(p_j/q_j) + 1 + Aeq_j'*lambda_eq + Aineq_j'*lambda_ineq = 0
        # p_j = q_j * exp(-1 - Aeq_j'*lambda_eq - Aineq_j'*lambda_ineq)
        
        # To simplify, we let G_j = Aeq_j'*lambda_eq + Aineq_j'*lambda_ineq
        # p_j(lambda) = q_j * exp(-G_j) / sum(q_i * exp(-G_i))
        # The dual objective is: ln(sum q_i * exp(-G_i)) + lambda_eq'*beq + lambda_ineq'*bineq
        
        exponent = -(Aeq.T @ lambda_eq)
        if k_ineq > 0:
            exponent -= (Aineq.T @ lambda_ineq)
            
        # Log-Sum-Exp trick
        max_exp = np.max(exponent)
        obj = np.log(np.sum(q.flatten() * np.exp(exponent - max_exp))) + max_exp
        
        obj += (lambda_eq @ beq.flatten())
        if k_ineq > 0:
            obj += (lambda_ineq @ bineq.flatten())
            
        return obj

    # Bounds for inequality multipliers (must be >= 0)
    bounds = [(None, None)] * k_eq + [(0, None)] * k_ineq
    
    res = minimize(
        dual_objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-12,
        options={"ftol": 1e-12, "gtol": 1e-12},
    )

    if not res.success:
        warnings.warn(
            f"Entropy pooling optimization did not converge: {res.message}. "
            "Returning prior probabilities unchanged — views may not be reflected.",
            RuntimeWarning,
            stacklevel=2,
        )
        return prior_probs

    # Recover posterior probabilities
    lambda_eq = res.x[:k_eq]
    lambda_ineq = res.x[k_eq:]
    exponent = -(Aeq.T @ lambda_eq)
    if k_ineq > 0:
        exponent -= (Aineq.T @ lambda_ineq)
    
    p = q.flatten() * np.exp(exponent)
    p = p / np.sum(p)
    return p


def meucci_views(
    R: pd.DataFrame, 
    views: List[Dict[str, Any]],
    prior_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Automate the generation of constraints for Meucci's Entropy Pooling.
    
    Each view in 'views' is a dict:
    - {'type': 'relative', 'asset_high': 'A', 'asset_low': 'B'} -> E[R_A] > E[R_B]
    - {'type': 'absolute', 'asset': 'A', 'value': 0.02} -> E[R_A] = 0.02
    - {'type': 'inequality', 'asset': 'A', 'value': 0.05, 'direction': 'less'} -> E[R_A] <= 0.05
    """
    T, N = R.shape
    if prior_probs is None:
        prior_probs = np.full(T, 1.0 / T)
        
    asset_names = list(R.columns)
    X = R.values
    
    Aeq_list = [np.ones((1, T))] # sum(p) = 1
    beq_list = [1.0]
    
    Aineq_list = []
    bineq_list = []
    
    for view in views:
        v_type = view['type']
        if v_type == 'relative':
            idx_h = asset_names.index(view['asset_high'])
            idx_l = asset_names.index(view['asset_low'])
            # E[R_h - R_l] >= 0  => sum(p_j * (R_jh - R_jl)) >= 0
            # sum(p_j * (R_jl - R_jh)) <= 0
            # Aineq: (R_jl - R_jh) [1 x T]
            Aineq_list.append((X[:, idx_l] - X[:, idx_h]).reshape(1, -1))
            bineq_list.append(0.0)
            
        elif v_type == 'absolute':
            idx = asset_names.index(view['asset'])
            # sum(p_j * R_ji) = value
            Aeq_list.append(X[:, idx].reshape(1, -1))
            beq_list.append(view['value'])
            
        elif v_type == 'inequality':
            idx = asset_names.index(view['asset'])
            direction = view.get('direction', 'less')
            if direction == 'less':
                # sum(p_j * R_ji) <= value
                Aineq_list.append(X[:, idx].reshape(1, -1))
                bineq_list.append(view['value'])
            else:
                # sum(p_j * R_ji) >= value  => sum(p_j * -R_ji) <= -value
                Aineq_list.append(-X[:, idx].reshape(1, -1))
                bineq_list.append(-view['value'])

    Aeq = np.vstack(Aeq_list)
    beq = np.array(beq_list).reshape(-1, 1)
    
    Aineq = np.vstack(Aineq_list) if Aineq_list else None
    bineq = np.array(bineq_list).reshape(-1, 1) if bineq_list else None
    
    return entropy_pooling(prior_probs, Aeq, beq, Aineq, bineq)


def meucci_ranking(
    R: pd.DataFrame, 
    order: List[Union[int, str]],
    prior_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a relative ranking into posterior probabilities.
    'order' is ascending: [lowest, ..., highest]
    """
    asset_names = list(R.columns)
    views = []
    for i in range(len(order) - 1):
        views.append({
            'type': 'relative',
            'asset_high': asset_names[order[i+1]] if isinstance(order[i+1], int) else order[i+1],
            'asset_low': asset_names[order[i]] if isinstance(order[i], int) else order[i]
        })
    return meucci_views(R, views, prior_probs)


def meucci_moments(R: np.ndarray, posterior_probs: np.ndarray) -> Dict[str, np.ndarray]:
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

    return {"mu": mu_p, "sigma": sigma_p}
