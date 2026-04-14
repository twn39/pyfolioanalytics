import numpy as np
import pandas as pd
import json
import os
import cvxpy as cp

def generate_tracking_cv():
    data_path = "data/stock_returns.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).iloc[:100, :5]
    assets = df.columns.tolist()
    N = len(assets)

    mu = df.mean().values
    Sigma = df.cov().values

    w_bench = np.full(N, 1.0 / N)
    
    # 1. L-infinity Tracking Error (Max deviation <= 0.05)
    target_L_inf = 0.05
    w_inf = cp.Variable(N)
    
    # Ground truth formulated directly via generic CVXPY bypassing PyFolioAnalytics structure
    # This acts as an independent mathematical verification.
    obj_inf = cp.Minimize(cp.quad_form(w_inf, Sigma))
    constraints_inf = [
        cp.sum(w_inf) == 1,
        w_inf >= 0,
        cp.norm(w_inf - w_bench, "inf") <= target_L_inf
    ]
    prob_inf = cp.Problem(obj_inf, constraints_inf)
    prob_inf.solve(solver=cp.SCS)
    
    # 2. L-1 Tracking Error (Active Share style <= 0.10)
    target_L_1 = 0.10
    w_1 = cp.Variable(N)
    obj_1 = cp.Minimize(cp.quad_form(w_1, Sigma))
    constraints_1 = [
        cp.sum(w_1) == 1,
        w_1 >= 0,
        cp.norm(w_1 - w_bench, 1) <= target_L_1
    ]
    prob_1 = cp.Problem(obj_1, constraints_1)
    prob_1.solve(solver=cp.SCS)
    
    # 3. L-infinity + Charnes-Cooper Ratio Optimization
    # Maximize Return / Risk, subject to sum=1, L-inf <= 0.05
    # Risk here is sqrt(w^T Sigma w). Ratio is (mu^T w) / sqrt(w^T Sigma w).
    # Substitute y = w * kappa
    kappa = cp.Variable(nonneg=True)
    y = cp.Variable(N)
    obj_ratio = cp.Minimize(- (mu @ y)) # max return
    constraints_ratio = [
        cp.sum(y) == kappa,
        y >= 0,
        cp.norm(y - kappa * w_bench, "inf") <= target_L_inf * kappa,
        cp.quad_form(y, Sigma) <= 1.0 # Risk <= 1
    ]
    prob_ratio = cp.Problem(obj_ratio, constraints_ratio)
    prob_ratio.solve(solver=cp.SCS)
    w_ratio = y.value / kappa.value

    results = {
        "L_inf_weights": w_inf.value.tolist(),
        "L_1_weights": w_1.value.tolist(),
        "L_inf_ratio_weights": w_ratio.tolist()
    }

    output_path = "data/tracking_cv.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_tracking_cv()
