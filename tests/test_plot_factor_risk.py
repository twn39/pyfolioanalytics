import numpy as np
import pandas as pd
from pyfolioanalytics.plots import plot_factor_risk_decomposition
from pyfolioanalytics.risk import factor_risk_decomposition

def test_plot_factor_risk_decomposition():
    # Setup dummy data
    N, K = 5, 2
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Factor loadings (B)
    B = np.array([
        [1.0, 0.5],
        [0.8, 0.2],
        [1.2, -0.4],
        [0.9, 0.0],
        [1.1, 0.1]
    ])
    
    # Factor covariance (sigma_f)
    sigma_f = np.array([
        [0.04, 0.005],
        [0.005, 0.02]
    ])
    
    # Idiosyncratic risk
    residual_sigma = np.diag([0.01, 0.015, 0.005, 0.02, 0.01])
    
    # Run the attribution
    decomp = factor_risk_decomposition(weights, B, sigma_f, residual_sigma, type="var")
    
    # Plot PCR
    fig_pcr = plot_factor_risk_decomposition(decomp, factor_names=["Market", "Value"], percentage=True)
    
    # Check that Plotly structure was generated
    assert len(fig_pcr.data) == 2 # 1 for Factors, 1 for Idiosyncratic
    
    factor_trace = fig_pcr.data[0]
    idiosyncratic_trace = fig_pcr.data[1]
    
    # Factors should have K elements, Idiosyncratic should have 1 element (summed)
    assert len(factor_trace.x) == K
    assert len(idiosyncratic_trace.x) == 1
    
    # The sum of all PCR elements (factors + idiosyncratic) should be 1.0 (100%)
    total_pcr_plotted = np.sum(factor_trace.x) + idiosyncratic_trace.x[0]
    assert np.isclose(total_pcr_plotted, 1.0)
    
    # Check X axis tickformat
    assert fig_pcr.layout.xaxis.tickformat == ".1%"

