import numpy as np
import pandas as pd
import json
from pyfolioanalytics.factors import statistical_factor_model
from pyfolioanalytics.moments import M4_MM

with open("data/comoments_cv.json", "r") as f:
    cv_data = json.load(f)

R_raw = np.array(cv_data["returns"])
R_df = pd.DataFrame(R_raw, columns=["A", "B", "C", "D"])

fm1 = statistical_factor_model(R_df, k=1)
print("PY FM1 Loadings:")
print(fm1["loadings"])

f = fm1["factors"].values
f_centered = f - np.mean(f, axis=0)
m4_f = M4_MM(f_centered)
print("PY FM1 Factor M4:")
print(m4_f)

res = fm1["residuals"].values
T, N = R_df.shape
stockM4 = np.sum(res**4, axis=0) / (T - 1 - 1)
print("PY FM1 Residual M4:")
print(stockM4)
