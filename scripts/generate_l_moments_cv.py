import numpy as np
import pandas as pd
import json
import riskfolio.src.OwaWeights as owa
import riskfolio.src.RiskFunctions as rf

from sklearn.covariance import MinCovDet

def generate_l_moments_cv():
    # Setup some test data
    np.random.seed(42)
    T, N = 100, 5
    R_raw = np.random.randn(T, N) * 0.01
    asset_names = [f"Asset.{i+1}" for i in range(N)]
    R = pd.DataFrame(R_raw, columns=asset_names)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    # 1. L-Moment Weights (k=2, 3, 4)
    owa_2 = owa.owa_l_moment(T, k=2).flatten().tolist()
    owa_3 = owa.owa_l_moment(T, k=3).flatten().tolist()
    owa_4 = owa.owa_l_moment(T, k=4).flatten().tolist()
    
    # 2. L-Moments for a portfolio
    p_returns = R.values @ weights
    l2 = rf.L_Moment(p_returns, k=2)
    l3 = rf.L_Moment(p_returns, k=3)
    l4 = rf.L_Moment(p_returns, k=4)

    # 3. Robust Covariance (MCD) via Sklearn
    mcd = MinCovDet(random_state=42).fit(R_raw)
    mu_robust = mcd.location_.flatten().tolist()
    sigma_robust = mcd.covariance_.tolist()
    
    results = {
        "T": T,
        "N": N,
        "returns": R_raw.tolist(),
        "weights": weights.tolist(),
        "owa_weights_k2": owa_2,
        "owa_weights_k3": owa_3,
        "owa_weights_k4": owa_4,
        "l2": l2,
        "l3": l3,
        "l4": l4,
        "mu_robust": mu_robust,
        "sigma_robust": sigma_robust
    }
    
    with open("data/l_moments_cv.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    generate_l_moments_cv()
