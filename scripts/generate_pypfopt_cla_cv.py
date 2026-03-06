import numpy as np
import pandas as pd
import json
from pypfopt.cla import CLA

# Correctly load EDHEC data
df = pd.read_csv("data/edhec.csv", sep=";", index_col=0)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.replace("%", "").astype(float) / 100
df = df.apply(pd.to_numeric, errors='coerce')

# Use a subset
df_subset = df.iloc[:100, :10]
mu = df_subset.mean() * 12
S = df_subset.cov() * 12

cla = CLA(mu, S, weight_bounds=(0, 1))

# 1. Max Sharpe
try:
    weights_ms = cla.max_sharpe()
    print("PyPfOpt Max Sharpe weights:", weights_ms)
except Exception as e:
    print("Max Sharpe failed:", e)
    weights_ms = None

# 2. Min Volatility
weights_mv = cla.min_volatility()
print("PyPfOpt Min Vol weights:", weights_mv)

# 3. Efficient Frontier
mu_f, sigma_f, weights_f = cla.efficient_frontier(points=20)

# Prepare output
output = {
    "mu": mu.tolist(),
    "sigma": S.values.tolist(),
    "max_sharpe_weights": list(weights_ms.values()) if weights_ms is not None else None,
    "min_volatility_weights": list(weights_mv.values()),
    "frontier_means": [float(x) for x in mu_f],
    "frontier_stds": [float(x) for x in sigma_f],
    "frontier_weights": [list(w.values()) if isinstance(w, dict) else w.flatten().tolist() for w in weights_f]
}

with open("data/pypfopt_cla_cv.json", "w") as f:
    json.dump(output, f)

print("Generated data/pypfopt_cla_cv.json")
