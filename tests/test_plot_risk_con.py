import numpy as np
import pandas as pd
from pyfolioanalytics.plots import plot_risk_decomposition

def test_plot_risk_decomposition_percentage():
    # 3 assets, total risk contribution is 0.1
    ccr = pd.Series({"A": 0.05, "B": 0.03, "C": 0.02})
    
    # Base plot (Percentage = True, default)
    fig_pct = plot_risk_decomposition(ccr, percentage=True)
    
    # Check that x values were converted to percentages (0.5, 0.3, 0.2)
    # The sorted order should be C (0.2), B (0.3), A (0.5)
    x_data = fig_pct.data[0].x
    assert np.isclose(x_data[0], 0.2)
    assert np.isclose(x_data[-1], 0.5)
    
    # Absolute plot (Percentage = False)
    fig_abs = plot_risk_decomposition(ccr, percentage=False)
    x_data_abs = fig_abs.data[0].x
    assert np.isclose(x_data_abs[0], 0.02)
    
def test_plot_risk_decomposition_erc_budget():
    ccr = pd.Series({"A": 0.05, "B": 0.03, "C": 0.02})
    custom_budget = {"A": 0.6, "B": 0.3, "C": 0.1}
    
    fig = plot_risk_decomposition(ccr, percentage=True, erc_line=True, custom_budget=custom_budget)
    
    # We should have 1 Bar trace and 1 Scatter trace
    assert len(fig.data) == 2
    assert fig.data[1].type == 'scatter'
    assert fig.data[1].name == 'Target Budget'
    
    # The layout should have a shape (the vline for ERC)
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].type == 'line'
    assert np.isclose(fig.layout.shapes[0].x0, 1/3)

