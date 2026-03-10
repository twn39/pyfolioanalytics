import numpy as np
import pandas as pd
import json
import riskfolio as rp
import riskfolio.src.OwaWeights as owa

import cvxpy as cp

def solve_owa_manual(R, owa_weights):
    T, n = R.shape
    w = cp.Variable(n)
    
    # Sort weights descending for convex risk
    owa_weights = np.sort(owa_weights)[::-1]
    
    delta_w = owa_weights[:-1] - owa_weights[1:]
    zeta = cp.Variable(T - 1)
    d = cp.Variable((T, T - 1), nonneg=True)
    losses = -R @ w
    
    constraints = [cp.sum(w) == 1, w >= 0]
    for k in range(1, T):
        constraints.append(d[:, k - 1] >= losses - zeta[k - 1])

    top_k_sums = [(k * zeta[k - 1] + cp.sum(d[:, k - 1])) for k in range(1, T)]
    owa_expr = cp.sum([delta_w[i] * top_k_sums[i] for i in range(T - 1)]) + owa_weights[-1] * cp.sum(losses)
    
    prob = cp.Problem(cp.Minimize(owa_expr), constraints)
    prob.solve()
    return w.value

def generate_advanced_risk_cv():
    # 1. Load Real Data (EDHEC)
    data = pd.read_csv("data/edhec.csv")
    asset_names = ["Convertible Arbitrage", "CTA Global", "Distressed Securities", "Emerging Markets", "Equity Market Neutral"]
    Y = data[asset_names]
    R = Y.values
    
    # 2. Setup Riskfolio Portfolio for RLDaR
    port = rp.Portfolio(returns=Y)
    port.assets_stats(method_mu='hist', method_cov='hist')
    w_rldar = port.optimization(model='Classic', rm='RLDaR', obj='MinRisk', rf=0, l=0, hist=True)
    
    # 3. Manual OWA for L-Moment CRM (k=4, MSD)
    T = R.shape[0]
    w_owa_crm = owa.owa_l_moment_crm(T, k=4, method="MSD")
    w_crm = solve_owa_manual(R, w_owa_crm.flatten())
    
    results = {
        "asset_names": asset_names,
        "w_rldar": w_rldar.iloc[:, 0].tolist(),
        "w_crm_msd": w_crm.tolist(),
        "owa_weights_crm": w_owa_crm.flatten().tolist()
    }
    
    with open("data/advanced_risk_real_cv.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    generate_advanced_risk_cv()
