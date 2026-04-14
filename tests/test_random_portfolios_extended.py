import numpy as np
import pytest
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.random_portfolios import random_portfolios

def test_rp_sample_basic(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="box", min=[0.05]*5, max=[0.4]*5)
    
    w_mat = random_portfolios(port, permutations=500, method="sample")
    assert w_mat.shape[1] == 5
    assert w_mat.shape[0] > 0
    # Check constraints
    assert np.allclose(w_mat.sum(axis=1), 1.0)
    assert np.all(w_mat >= 0.05 - 1e-5)
    assert np.all(w_mat <= 0.4 + 1e-5)

def test_rp_grid_basic(stocks_data):
    R = stocks_data.iloc[:, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    
    w_mat = random_portfolios(port, permutations=100, method="grid")
    assert w_mat.shape[1] == 4
    assert w_mat.shape[0] > 0
    assert np.allclose(w_mat.sum(axis=1), 1.0)
    assert np.all(w_mat >= -1e-5)

def test_rp_sample_max_pos(stocks_data):
    R = stocks_data.iloc[:, :5]  # stocks_data only has 5 columns
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_constraint(type="position_limit", max_pos=3)
    
    w_mat = random_portfolios(port, permutations=500, method="sample")
    assert w_mat.shape[1] == 5
    assert w_mat.shape[0] > 0
    assert np.allclose(w_mat.sum(axis=1), 1.0)
    # Check max_pos constraint: count non-zero weights per portfolio
    nonzero_counts = np.sum(w_mat > 1e-5, axis=1)
    assert np.all(nonzero_counts <= 3)

