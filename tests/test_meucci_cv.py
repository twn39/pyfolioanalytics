import json
import numpy as np
import pytest
from pyfolioanalytics.meucci import entropy_prog, meucci_ranking, centroid_ranking

def load_cv_data():
    with open("data/meucci_cv.json", "r") as f:
        return json.load(f)

def test_entropy_prog_eq():
    data = load_cv_data()
    prior_probs = np.array(data['prior_probs'])
    R = np.array(data['input_R'])
    T = R.shape[0]
    print(f"DEBUG: T={T}, prior_probs.shape={prior_probs.shape}, R.shape={R.shape}")
    
    Aeq = np.vstack([np.ones(T), R[:, 0]])
    beq = np.array([1.0, 0.005])
    
    res = entropy_prog(prior_probs, Aeq=Aeq, beq=beq)
    
    expected_p = np.array(data['entropy_prog_eq']['p_posterior'])
    np.testing.assert_allclose(res['p_'], expected_p, rtol=1e-7, atol=1e-7)

def test_entropy_prog_ineq():
    data = load_cv_data()
    prior_probs = np.array(data['prior_probs'])
    R = np.array(data['input_R'])
    T = R.shape[0]
    
    # View: E[R3 - R2] <= 0 (i.e. E[R2] >= E[R3])
    Aineq = (R[:, 2] - R[:, 1]).reshape(1, -1)
    bineq = np.array([0.0])
    Aeq = np.ones((1, T))
    beq = np.array([1.0])
    
    res = entropy_prog(prior_probs, A=Aineq, b=bineq, Aeq=Aeq, beq=beq)
    
    expected_p = np.array(data['entropy_prog_ineq']['p_posterior'])
    np.testing.assert_allclose(res['p_'], expected_p, rtol=1e-6, atol=1e-6)

def test_meucci_ranking_cv():
    data = load_cv_data()
    R = np.array(data['input_R'])
    
    # Order: asset 2 < asset 3 < asset 1 < asset 4 < asset 5 (0-based: 1, 2, 0, 3, 4)
    # R order was c(2, 3, 1, 4, 5) which is 1-based.
    order = [1, 2, 0, 3, 4]
    
    res = meucci_ranking(R, order)
    
    expected_mu = np.array(data['meucci_ranking']['mu'])
    expected_sigma = np.array(data['meucci_ranking']['sigma'])
    
    np.testing.assert_allclose(res['mu'], expected_mu, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(res['sigma'], expected_sigma, rtol=1e-7, atol=1e-7)

def test_centroid_ranking():
    c = centroid_ranking(3)
    expected = np.array([11/18, 5/12, 1/3])
    np.testing.assert_allclose(c, expected)
