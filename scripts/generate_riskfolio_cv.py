import riskfolio as rp
import pandas as pd
import numpy as np
import json
import os

def generate_riskfolio_cv():
    # Load data
    data_path = "data/stock_returns.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).iloc[:100]
    assets = df.columns.tolist()

    results = {}

    # 1. Kelly Optimization
    port = rp.Portfolio(returns=df)
    port.assets_stats(method_mu='hist', method_cov='hist')
    w_kelly = port.optimization(model='Classic', rm='MV', obj='Sharpe', kelly='exact', rf=0)
    
    results["kelly"] = {
        "weights": w_kelly["weights"].to_dict()
    }

    # 2. Maximum Diversification
    port_mdiv = rp.Portfolio(returns=df)
    vols = df.std().to_frame().T
    port_mdiv.mu = vols
    port_mdiv.cov = df.cov()
    w_mdiv = port_mdiv.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0)
    
    # Calculate Diversification Ratio
    w = w_mdiv["weights"].values
    cov = df.cov().values
    asset_vols = np.sqrt(np.diag(cov))
    p_vol = np.sqrt(w @ cov @ w)
    div_ratio = (w @ asset_vols) / p_vol

    results["mdiv"] = {
        "weights": w_mdiv["weights"].to_dict(),
        "div_ratio": float(div_ratio)
    }

    # Save to JSON
    output_path = "data/riskfolio_cv.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_riskfolio_cv()
