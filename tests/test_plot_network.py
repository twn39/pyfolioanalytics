import numpy as np
import pandas as pd
from pyfolioanalytics.plots import plot_network_allocation

def test_plot_network_allocation():
    # Setup dummy returns for 5 assets
    np.random.seed(42)
    # create some correlated blocks
    factor1 = np.random.randn(100)
    factor2 = np.random.randn(100)
    
    R = pd.DataFrame({
        "A": factor1 + np.random.randn(100) * 0.1,
        "B": factor1 + np.random.randn(100) * 0.1,
        "C": factor2 + np.random.randn(100) * 0.1,
        "D": factor2 + np.random.randn(100) * 0.1,
        "E": np.random.randn(100) # uncorrelated
    })
    
    weights = pd.Series({"A": 0.4, "B": 0.1, "C": 0.2, "D": 0.2, "E": 0.1})
    
    fig = plot_network_allocation(weights, R)
    
    # Assert traces exist
    assert len(fig.data) == 2 # 1 for edges, 1 for nodes
    
    # Trace 0 is edges
    assert fig.data[0].mode == 'lines'
    # Trace 1 is nodes
    assert fig.data[1].mode == 'markers+text'
    
    # Check node size scaling (A should be largest)
    sizes = fig.data[1].marker.size
    assert max(sizes) == sizes[0] # A is index 0
    assert min(sizes) == sizes[1] # B is index 1
    
    # Layout checks
    assert not fig.layout.xaxis.showgrid
    assert not fig.layout.yaxis.showgrid

