import numpy as np
import pandas as pd
from pyfolioanalytics.plots import plot_return_histogram

def test_plot_return_histogram():
    # Generate some normal data with a fat left tail
    np.random.seed(42)
    normal_rets = np.random.normal(0.001, 0.02, 1000)
    crash_rets = np.random.normal(-0.10, 0.05, 50)
    rets = pd.Series(np.concatenate([normal_rets, crash_rets]))
    
    # Generate plot
    fig = plot_return_histogram(rets, alpha=0.05)
    
    # We should have traces for: Histogram, KDE, VaR line, CVaR line, and Shaded Tail Area
    # (ff.create_distplot adds 1 hist trace and 1 KDE trace by default, making 5 total)
    assert len(fig.data) >= 5
    
    # Check trace names
    trace_names = [t.name for t in fig.data if hasattr(t, 'name')]
    assert any("VaR (95%)" in name for name in trace_names if name is not None)
    assert any("CVaR/ES (95%)" in name for name in trace_names if name is not None)
    
    # Check X-axis tickformat is set for percentages
    assert fig.layout.xaxis.tickformat == ".2%"

