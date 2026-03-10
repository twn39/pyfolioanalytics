import numpy as np
import pandas as pd
import cvxpy as cp
import json

def generate_active_share_cv():
    # 1. Load Data
    data = pd.read_csv("data/edhec.csv")
    asset_names = ["Convertible Arbitrage", "CTA Global", "Distressed Securities", "Emerging Markets", "Equity Market Neutral"]
    Y = data[asset_names].values
    mu = np.mean(Y, axis=0)
    T, N = Y.shape
    
    # 2. Setup Reference Optimizer (Direct CVXPY)
    w = cp.Variable(N)
    
    # Equal weight benchmark
    w_b = np.full(N, 1.0 / N)
    
    # Target Active Share
    as_target = 0.1
    
    # Problem: Maximize Return subject to Active Share <= 0.1
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        0.5 * cp.norm(w - w_b, 1) <= as_target
    ]
    
    prob = cp.Problem(cp.Maximize(w @ mu), constraints)
    prob.solve()
    
    results = {
        "asset_names": asset_names,
        "w_b": w_b.tolist(),
        "as_target": as_target,
        "w_optimal": w.value.tolist(),
        "obj_value": float(prob.value)
    }
    
    with open("data/active_share_real_cv.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    generate_active_share_cv()
