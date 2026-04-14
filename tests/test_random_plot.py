import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.random_portfolios import random_portfolios
from pyfolioanalytics.moments import set_portfolio_moments
from pyfolioanalytics.plots import plot_random_portfolios

def test_plot_random_portfolios(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_constraint(type="position_limit", max_pos=3) # creates non-convex space
    
    # Needs some objectives to calculate measures
    port.add_objective(type="risk", name="StdDev")
    port.add_objective(type="return", name="mean")
    
    # Generate random portfolios
    w_mat = random_portfolios(port, permutations=50, method="sample")
    moments = set_portfolio_moments(R, port)
    
    # Dummy optimal weights
    opt_w = pd.Series([0.5, 0.5, 0, 0, 0], index=R.columns)
    
    fig = plot_random_portfolios(
        random_weights=w_mat, 
        moments=moments, 
        objectives=port.objectives,
        R=R,
        optimal_weights=opt_w
    )
    
    # verify plot object created correctly
    assert fig is not None
    assert len(fig.data) == 2 # 1 for scattergl, 1 for star
    
    # WebGL scatter should exist
    assert fig.data[0].type == 'scattergl'

