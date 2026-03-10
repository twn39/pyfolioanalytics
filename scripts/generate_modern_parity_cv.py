import numpy as np
import pandas as pd
import json
import riskfolio as rp

def generate_modern_parity_cv():
    # 1. Load Real Data (EDHEC)
    data = pd.read_csv("data/edhec.csv")
    asset_names = ["Convertible Arbitrage", "CTA Global", "Distressed Securities", "Emerging Markets", "Equity Market Neutral"]
    Y = data[asset_names]
    
    # 2. Setup Riskfolio Portfolio
    port = rp.Portfolio(returns=Y)
    port.assets_stats(method_mu='hist', method_cov='hist')
    
    # 3. Minimize EDaR
    # Riskfolio uses 'EDaR' as model name
    w_edar = port.optimization(model='Classic', rm='EDaR', obj='MinRisk', rf=0, l=0, hist=True)
    
    # 4. Minimize RLVaR
    # Riskfolio uses 'RLVaR'
    w_rlvar = port.optimization(model='Classic', rm='RLVaR', obj='MinRisk', rf=0, l=0, hist=True)
    
    # Get risk values from the optimized portfolios
    risk_edar = rp.RiskFunctions.EDaR_Abs(Y @ w_edar.values, alpha=0.05)[0]
    risk_rlvar = rp.RiskFunctions.RLVaR_Hist(Y @ w_rlvar.values, alpha=0.05, kappa=0.3)
    
    results = {
        "asset_names": asset_names,
        "w_edar": w_edar.iloc[:, 0].tolist(),
        "risk_edar": float(risk_edar),
        "w_rlvar": w_rlvar.iloc[:, 0].tolist(),
        "risk_rlvar": float(risk_rlvar)
    }
    
    with open("data/modern_risk_parity_real_cv.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    generate_modern_parity_cv()
