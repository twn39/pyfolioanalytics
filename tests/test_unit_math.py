import numpy as np
import pytest

from pyfolioanalytics.moments import M3_MM, M4_MM
from pyfolioanalytics.risk import calculate_drawdowns, owa_cvar_weights, owa_gmd_weights

def test_calculate_drawdowns_unit():
    """Pure unit test for drawdown calculation using a deterministic array."""
    # Returns: +10%, -20%, +30%, -10%
    p_returns = np.array([0.1, -0.2, 0.3, -0.1])
    
    # Cumulative returns: [0.1, -0.1, 0.2, 0.1]
    # Running Max (Peak): [0.1, 0.1, 0.2, 0.2]
    # Drawdowns (Cum - Peak): [0.0, -0.2, 0.0, -0.1]
    
    drawdowns = calculate_drawdowns(p_returns)
    expected_drawdowns = np.array([0.0, -0.2, 0.0, -0.1])
    
    assert np.allclose(drawdowns, expected_drawdowns)

def test_m3_mm_unit():
    """Pure unit test for co-skewness tensor flattening."""
    # T=2 observations, N=2 assets
    R = np.array([[1.0, 0.0],
                  [0.0, 2.0]])
    
    # M3_ijk = mean(R_ti * R_tj * R_tk)
    # Expected M3[0,0,0] = (1^3 + 0^3)/2 = 0.5
    # Expected M3[1,1,1] = (0^3 + 2^3)/2 = 4.0
    # Expected M3[0,1,0] = (1*0*1 + 0*2*0)/2 = 0.0
    
    M3 = M3_MM(R)
    
    # Resulting shape should be (N, N^2) -> (2, 4)
    assert M3.shape == (2, 4)
    
    # Flattened indices for (i, j, k) -> i, j*N + k
    # M3[0, 0] corresponds to i=0, j=0, k=0 -> 0.5
    assert np.isclose(M3[0, 0], 0.5)
    # M3[1, 3] corresponds to i=1, j=1, k=1 -> 4.0
    assert np.isclose(M3[1, 3], 4.0)
    # M3[0, 1] corresponds to i=0, j=0, k=1 -> 0.0
    assert np.isclose(M3[0, 1], 0.0)

def test_owa_cvar_weights_unit():
    """Pure unit test for CVaR OWA weight generation."""
    T = 100
    p = 0.05
    # 5% of 100 is 5.
    # The worst 5 observations should have equal weight 1/5 = 0.2, rest 0.
    
    w = owa_cvar_weights(T, p)
    
    assert len(w) == 100
    assert np.allclose(w[:5], 0.2)
    assert np.allclose(w[5:], 0.0)
    assert np.isclose(np.sum(w), 1.0)

def test_owa_gmd_weights_unit():
    """Pure unit test for Gini Mean Difference (GMD) OWA weight generation."""
    T = 5
    # For GMD, weights are based on the rank (L-moments k=2)
    w = owa_gmd_weights(T)
    
    assert len(w) == 5
    assert np.isclose(np.sum(w), 0.0)  # GMD weights sum to 0
    # Weights should be strictly decreasing
    assert np.all(np.diff(w) < 0)
    # They should be symmetric around 0
    assert np.allclose(w, -w[::-1])

