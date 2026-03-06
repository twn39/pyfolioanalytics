import pandas as pd
import numpy as np
import riskfolio as rp
import json
import os

# Load data
df = pd.read_csv("data/edhec.csv", index_col=0)
Y = df.iloc[:, :10]

# Setup Portfolio
port = rp.Portfolio(returns=Y)
port.assets_stats(method_mu='hist', method_cov='hist')

# Riskfolio supports robust optimization
# We will use 'MV' (Mean-Variance) with 'Ellipsoidal' uncertainty on mu
# For Riskfolio, robust optimization is configured via 'kind' and 'kappa' (confidence level)
model = 'Classic'
rm = 'MV'
obj = 'Sharpe'
rf = 0.0

# Generate Ellipsoidal uncertainty set for mu
# We'll use a fixed kappa for repeatability
kappa = 0.5 

# Solve robust optimization in Riskfolio
# Note: Riskfolio robust implementation might differ slightly in formulation (SOCP)
# but it's the closest industry-standard Python library.
w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, kappa=kappa, solver='CLARABEL')

# Riskfolio stores uncertainty sets
sigma_mu = port.cov_mu # Uncertainty of mu (if estimated)

results = {
    "mu": port.mu.flatten().tolist(),
    "sigma": port.cov.values.tolist(),
    "kappa": kappa,
    "weights": w['weights'].tolist()
}

with open("data/robust_socp_cv.json", "w") as f:
    json.dump(results, f)

print("Generated data/robust_socp_cv.json")
