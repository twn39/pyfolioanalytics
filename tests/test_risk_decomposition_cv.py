import json
import numpy as np
from pyfolioanalytics.risk import risk_decomposition

def load_cv_data():
    with open("data/risk_decomposition_cv.json", "r") as f:
        return json.load(f)

def test_std_dev_decomposition_cv():
    data = load_cv_data()
    sigma = np.array(data['sigma'])
    weights = np.array(data['weights'])
    
    # Calculate decomposition using Python
    res = risk_decomposition(weights, sigma, type="StdDev")
    
    expected = data['std_dev_decomp']
    
    # Total StdDev
    np.testing.assert_allclose(res['total'], expected['total'], rtol=1e-7)
    
    # Marginal Contribution (MCR)
    np.testing.assert_allclose(res['mcr'], np.array(expected['mcr']).flatten(), rtol=1e-7)
    
    # Component Contribution (CCR)
    np.testing.assert_allclose(res['ccr'], np.array(expected['ccr']).flatten(), rtol=1e-7)
    
    # Percentage Contribution (PCR)
    np.testing.assert_allclose(res['pcr'], np.array(expected['pcr']).flatten(), rtol=1e-7)
    
    # Euler Check: Sum(CCR) == Total
    np.testing.assert_allclose(np.sum(res['ccr']), res['total'], rtol=1e-10)

def test_var_decomposition_cv():
    data = load_cv_data()
    sigma = np.array(data['sigma'])
    weights = np.array(data['weights'])
    
    # Calculate decomposition using Python
    res = risk_decomposition(weights, sigma, type="var")
    
    expected = data['var_decomp']
    
    # Total Variance
    np.testing.assert_allclose(res['total'], expected['total'], rtol=1e-7)
    
    # Component Contribution (CCR)
    np.testing.assert_allclose(res['ccr'], np.array(expected['ccr']).flatten(), rtol=1e-7)
    
    # Percentage Contribution (PCR)
    np.testing.assert_allclose(res['pcr'], np.array(expected['pcr']).flatten(), rtol=1e-7)
    
    # Euler Check: Sum(CCR) == Total
    np.testing.assert_allclose(np.sum(res['ccr']), res['total'], rtol=1e-10)
