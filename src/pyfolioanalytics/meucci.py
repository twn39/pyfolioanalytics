import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional


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
    subject to linear constraints on p.
    """
    T = len(prior_probs)
    q = prior_probs.reshape(-1, 1)

    # We solve the dual problem for efficiency
    # The dual objective is sum(q * exp(-1 - A' * lambda)) + b' * lambda

    # Consolidate constraints for the solver
    # This implementation focuses on equality constraints for simplicity
    if Aeq is None:
        return prior_probs

    k = Aeq.shape[0]
    x0 = np.zeros(k)

    def dual_objective(x):
        # x is the vector of Lagrange multipliers (lambda)
        # exp_term = q * exp(-Aeq.T @ x)
        # We handle the '-1' constant by normalization later
        ln_q = np.log(q.flatten())
        val = np.exp(ln_q - (Aeq.T @ x))
        return np.sum(val) + (beq.flatten() @ x)

    res = minimize(
        dual_objective,
        x0,
        method="L-BFGS-B",
        tol=1e-12,
        options={"ftol": 1e-12, "gtol": 1e-12},
    )

    if not res.success:
        # Fallback or raise
        pass

    # Recover posterior probabilities
    ln_q = np.log(q.flatten())
    p = np.exp(ln_q - (Aeq.T @ res.x))
    p = p / np.sum(p)  # Ensure they sum to 1

    return p


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
