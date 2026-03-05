import json
import pandas as pd
import numpy as np
import pytest
import os
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.risk import EVaR

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

def test_multi_dataset_cross_validation():
    with open("data/multi_cross_val.json", "r") as f:
        all_r_results = json.load(f)
        
    for ds_name in ["edhec", "stocks", "macro"]:
        print(f"Testing dataset: {ds_name}")
        r_results = all_r_results[ds_name]
        R_orig = load_dataset(ds_name)
        asset_names = r_results["assets"]
        R = R_orig[asset_names].copy()
        
        # 1. Base MVO
        portfolio = Portfolio(assets=asset_names)
        portfolio.add_constraint(type="full_investment")
        portfolio.add_constraint(type="long_only")
        portfolio.add_objective(type="risk", name="StdDev")
        py_res_mvo = optimize_portfolio(R, portfolio, optimize_method="ROI")
        np.testing.assert_allclose(py_res_mvo["weights"].values, r_results["mvo_weights"], atol=1e-4)
        
        # 2. EVaR Calculation Parity (Equal Weight)
        n = len(asset_names)
        weights = np.full(n, 1.0 / n)
        py_evar = EVaR(weights, R.values, p=0.95)
        np.testing.assert_allclose(py_evar, r_results["evar_eq"], rtol=1e-5)
        
        # 3. Robust MVO Parity (Utility objective)
        portfolio_rob = Portfolio(assets=asset_names)
        portfolio_rob.add_constraint(type="full_investment")
        portfolio_rob.add_constraint(type="long_only")
        # delta_mu = 0.0001
        portfolio_rob.add_constraint(type="robust", delta_mu=0.0001)
        portfolio_rob.add_objective(type="quadratic_utility", risk_aversion=2.0)
        
        py_res_rob = optimize_portfolio(R, portfolio_rob)
        np.testing.assert_allclose(py_res_rob["weights"].values, r_results["robust_weights"], atol=1e-4)

