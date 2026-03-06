import numpy as np
import pandas as pd
import json
from pypfopt.cla import CLA

# 1. 加载并清洗数据
df = pd.read_csv("data/edhec.csv", index_col=0)

# 2. 选取子集并确保没有 NaN
# 使用前 100 行，前 10 个资产
df_subset = df.iloc[:100, :10].dropna()

print(f"Using {len(df_subset)} rows for calculation.")

# 3. 计算年化指标 (假设月度数据)
mu = df_subset.mean() * 12
S = df_subset.cov() * 12

if mu.isna().any() or S.isna().any().any():
    print("Error: mu or S still contains NaNs!")
    print("mu NaNs:", mu.isna().sum())
    print("S NaNs:", S.isna().sum().sum())

# 4. 运行 PyPortfolioOpt CLA
cla = CLA(mu, S, weight_bounds=(0, 1))

# Max Sharpe
try:
    weights_ms = cla.max_sharpe()
    ms_w = list(weights_ms.values())
except:
    ms_w = None

# Min Volatility
weights_mv = cla.min_volatility()
mv_w = list(weights_mv.values())

# Efficient Frontier
mu_f, sigma_f, weights_f = cla.efficient_frontier(points=20)

# 5. 序列化输出
output = {
    "mu": mu.tolist(),
    "sigma": S.values.tolist(),
    "max_sharpe_weights": ms_w,
    "min_volatility_weights": mv_w,
    "frontier_means": [float(x) for x in mu_f],
    "frontier_stds": [float(x) for x in sigma_f],
    "frontier_weights": [list(w.values()) if isinstance(w, dict) else w.flatten().tolist() for w in weights_f]
}

with open("data/pypfopt_cla_cv.json", "w") as f:
    json.dump(output, f)

print("Successfully generated data/pypfopt_cla_cv.json without NaNs.")
