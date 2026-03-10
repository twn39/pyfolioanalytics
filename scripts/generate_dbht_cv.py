import numpy as np
import pandas as pd
import json
import riskfolio as rp

def generate_dbht_cv():
    # 1. Load Real Data (EDHEC)
    data = pd.read_csv("data/edhec.csv")
    # DBHT is better with more assets, using 12 assets
    asset_names = data.columns[1:13].tolist()
    Y = data[asset_names]
    
    # 2. Setup Riskfolio Hierarchical Portfolio
    port = rp.HCPortfolio(returns=Y)
    
    # 3. Calculate DBHT clusters and HRP weights
    # Riskfolio: model='HRP', codependence='pearson', linkage='DBHT'
    w_hrp_dbht = port.optimization(model='HRP', codependence='pearson', linkage='DBHT', rm='MV', rf=0)
    
    # We also want to capture the cluster labels directly if possible
    # Riskfolio's HCPortfolio stores some internal state after optimization
    # But let's calculate the DBHT labels directly using its internal function to be sure
    corr = Y.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    S = corr + 1.0
    T8, Rpm, Adjv, Dpm, Mv, Z = rp.DBHT.DBHTs(dist, S, leaf_order=False)
    
    results = {
        "asset_names": asset_names,
        "clusters": T8.flatten().tolist(),
        "w_hrp_dbht": w_hrp_dbht.iloc[:, 0].tolist(),
        "linkage_matrix": Z.tolist()
    }
    
    with open("data/dbht_real_cv.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    generate_dbht_cv()
