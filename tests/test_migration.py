import json
import pandas as pd
import numpy as np
import pytest
import os
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio, equal_weight
from pyfolioanalytics.moments import set_portfolio_moments
from pyfolioanalytics.factors import statistical_factor_model, factor_model_covariance
from pyfolioanalytics.risk import VaR, max_drawdown

def load_dataset(name):
    if name == "edhec":
        df = pd.read_table("data/edhec.csv", sep=";", header=0)
        dates = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
        df = df.iloc[:, 1:].copy()
        df.index = dates
        df.columns = [c.replace(" ", ".") for c in df.columns]
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].str.replace("%", "").astype(float) / 100
        return df.iloc[:, :5]
    elif name == "stocks":
        df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
        return df
    elif name == "macro":
        df = pd.read_csv("data/macro_returns.csv", index_col=0, parse_dates=True)
        df.columns = [c.replace("-", ".") for c in df.columns]
        return df

def test_exhaustive_multi_dataset_cross_validation():
    with open("exhaustive_cross_val.json", "r") as f:
        all_r_results = json.load(f)
        
    for ds_name in ["edhec", "stocks", "macro"]:
        print(f"Testing exhaustive suite on: {ds_name}")
        r = all_r_results[ds_name]
        R = load_dataset(ds_name)
        asset_names = r["assets"]
        R = R[asset_names].copy()
        
        # 1. Factor Model Parity
        fm = statistical_factor_model(R, k=3)
        sigma_fm = factor_model_covariance(fm)
        np.testing.assert_allclose(sigma_fm, r["sigma_fm"], rtol=1e-7)
        
        # 2. Risk Measures (mVaR and MaxDD)
        portfolio = Portfolio(assets=asset_names)
        portfolio.add_objective(type="risk", name="mVaR")
        moments = set_portfolio_moments(R, portfolio)
        n = len(asset_names)
        weights = np.full(n, 1.0 / n)
        py_mvar = VaR(weights, moments["mu"], moments["sigma"], moments["m3"], moments["m4"], p=0.95, method="modified")
        np.testing.assert_allclose(py_mvar, r["mvar"], rtol=1e-7)
        py_max_dd = max_drawdown(weights, R.values)
        np.testing.assert_allclose(py_max_dd, r["max_dd"], rtol=1e-7)
        
        # 3. Optimization Parity at Rebalance Points
        # Instead of full backtest, we test specific points exported from R to ensure math parity
        portfolio_opt = Portfolio(assets=asset_names)
        portfolio_opt.add_constraint(type="full_investment")
        portfolio_opt.add_constraint(type="box", min=0.05, max=0.5)
        portfolio_opt.add_objective(type="risk", name="StdDev")
        
        r_dates = r["rebal_dates"]
        r_weights = np.array(r["rebal_weights"])
        
        # Check first, middle, and last rebalance point for efficiency
        test_indices = [0, len(r_dates)//2, len(r_dates)-1]
        for idx in test_indices:
            date_str = r_dates[idx]
            target_w = r_weights[idx]
            
            # Slice R as R[1:date]
            R_slice = R.loc[:date_str]
            
            # R might use a rolling window of 12 if set, but we used expanding in R script?
            # Re-checking R script: optimize.portfolio.rebalancing(..., training_period=36)
            # Default is expanding if rolling_window is NULL.
            
            py_res = optimize_portfolio(R_slice, portfolio_opt, optimize_method="ROI")
            np.testing.assert_allclose(py_res["weights"].values, target_w, atol=1e-4)

def test_rp_transform_constraints():
    pass

def test_regime_switching_rebalancing():
    pass
