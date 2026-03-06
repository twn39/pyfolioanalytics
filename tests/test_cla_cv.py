import pytest
import numpy as np
import json
import os
from pyfolioanalytics.cla import CLA

def test_cla_parity_with_r():
    # Load ground truth
    json_path = os.path.join("data", "cla_cv.json")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    mu = np.array(data["mu"])
    sigma = np.array(data["sigma"])
    # r_weights shape is (N_PORTFOLIOS, N_ASSETS) with "rowmajor" export
    r_weights = np.array(data["frontier_weights"])
    r_means = np.array(data["frontier_means"])
    r_stds = np.array(data["frontier_stds"])
    
    n_assets = len(mu)
    n_portfolios = len(r_means)
    
    lb = np.zeros(n_assets)
    ub = np.ones(n_assets)
    
    # Run Python CLA
    cla = CLA(mu, sigma, lb, ub)
    cla.solve()
    
    # Generate a high-resolution frontier in Python
    py_means, py_stds, py_weights = cla.efficient_frontier(points=1000)
    
    for i in range(n_portfolios):
        target_ret = r_means[i]
        target_std = r_stds[i]
        target_w = r_weights[i] # Rows are portfolios now
        
        idx = np.argmin(np.abs(py_means - target_ret))
        
        # Parity checks
        assert np.isclose(py_means[idx], target_ret, atol=1e-5)
        # Check standard deviation (Risk)
        np.testing.assert_allclose(py_stds[idx], target_std, rtol=2e-3, atol=1e-4)
        # Check weights
        np.testing.assert_allclose(py_weights[idx], target_w, atol=1e-2)
