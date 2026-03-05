import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union

def black_litterman(
    sigma: np.ndarray,
    w_mkt: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    Omega: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Standard Black-Litterman Model.
    - sigma: Covariance matrix (N x N)
    - w_mkt: Market weights (N x 1)
    - P: View matrix (K x N)
    - q: View returns (K x 1)
    - tau: Scalar indicating confidence in prior (default 0.05)
    - risk_aversion: Lambda (default 2.5)
    - Omega: View uncertainty matrix (K x K). If None, calculated via He-Litterman.
    """
    # 1. Implied Equilibrium Returns
    Pi = risk_aversion * sigma @ w_mkt
    
    # 2. View Uncertainty (Omega)
    if Omega is None:
        # He-Litterman method: Omega = diag(P * (tau * sigma) * P')
        Omega = np.diag(np.diag(P @ (tau * sigma) @ P.T))
        
    # 3. Posterior Mean
    # mu_bl = Pi + tau*sigma*P' * (P*tau*sigma*P' + Omega)^-1 * (q - P*Pi)
    M_inv = np.linalg.inv(P @ (tau * sigma) @ P.T + Omega)
    mu_bl = Pi + (tau * sigma @ P.T) @ M_inv @ (q - P @ Pi)
    
    # 4. Posterior Covariance
    # sigma_bl = (1+tau)*sigma - tau^2 * sigma * P' * (P*tau*sigma*P' + Omega)^-1 * P * sigma
    sigma_bl = (1 + tau) * sigma - (tau**2 * sigma @ P.T) @ M_inv @ (P @ sigma)
    
    return {
        "mu": mu_bl,
        "sigma": sigma_bl
    }
