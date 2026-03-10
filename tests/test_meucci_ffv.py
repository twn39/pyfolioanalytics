import numpy as np
import pandas as pd
import pytest
from pyfolioanalytics.meucci import meucci_views, meucci_moments, meucci_ranking

def test_meucci_relative_view():
    np.random.seed(42)
    T, N = 100, 2
    # Create two assets with same mean
    R_raw = np.random.randn(T, N) * 0.01
    R = pd.DataFrame(R_raw, columns=['A', 'B'])
    
    # View: A > B
    views = [{'type': 'relative', 'asset_high': 'A', 'asset_low': 'B'}]
    
    p_post = meucci_views(R, views)
    moments = meucci_moments(R.values, p_post)
    
    mu_a = moments['mu'][0, 0]
    mu_b = moments['mu'][1, 0]
    
    # After view A > B, mu_a should be > mu_b
    assert mu_a > mu_b
    # Sum of probabilities should be 1
    assert np.allclose(np.sum(p_post), 1.0)

def test_meucci_absolute_view():
    np.random.seed(42)
    T, N = 100, 1
    R_raw = np.random.randn(T, N) * 0.01
    R = pd.DataFrame(R_raw, columns=['A'])
    
    # Current mean is near 0
    target_mu = 0.01 # Within reasonable bounds of random N(0, 0.01)
    views = [{'type': 'absolute', 'asset': 'A', 'value': target_mu}]
    
    p_post = meucci_views(R, views)
    moments = meucci_moments(R.values, p_post)
    
    mu_post = moments['mu'][0, 0]
    # Post mu should be exactly target_mu
    assert np.allclose(mu_post, target_mu, atol=1e-5)

def test_meucci_ranking():
    np.random.seed(42)
    T, N = 100, 3
    R_raw = np.random.randn(T, N) * 0.01
    R = pd.DataFrame(R_raw, columns=['A', 'B', 'C'])
    
    # Order: C < B < A
    order = ['C', 'B', 'A']
    p_post = meucci_ranking(R, order)
    moments = meucci_moments(R.values, p_post)
    
    mu = moments['mu'].flatten()
    # A is at index 0, B at 1, C at 2 in columns
    # Order implies mu[2] < mu[1] < mu[0]
    assert mu[2] < mu[1] < mu[0]

def test_meucci_cv():
    import json
    with open("data/meucci_cv.json", "r") as f:
        cv_data = json.load(f)
        
    R_raw = np.array(cv_data["returns"])
    R_df = pd.DataFrame(R_raw, columns=["A", "B", "C"])
    
    # 1. Ranking CV (C < B < A in R indexing 3, 2, 1)
    # R: c(3, 2, 1) => A=0, B=1, C=2. Ascending order of indices: 2, 1, 0
    order = ["C", "B", "A"]
    p_ranking_actual = meucci_ranking(R_df, order)
    
    # Compare posterior mu
    moments_ranking = meucci_moments(R_raw, p_ranking_actual)
    np.testing.assert_allclose(moments_ranking["mu"].flatten(), cv_data["mu_ranking"], rtol=1e-4)
    
    # 2. Absolute view CV (A = 0.01)
    views = [{"type": "absolute", "asset": "A", "value": cv_data["target_mu_A"]}]
    p_absolute_actual = meucci_views(R_df, views)
    
    np.testing.assert_allclose(p_absolute_actual.flatten(), np.array(cv_data["p_absolute"]).flatten(), rtol=1e-4)
    
    moments_absolute = meucci_moments(R_raw, p_absolute_actual)
    np.testing.assert_allclose(moments_absolute["mu"].flatten(), cv_data["mu_absolute"], rtol=1e-4)
