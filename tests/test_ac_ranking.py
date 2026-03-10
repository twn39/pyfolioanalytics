import numpy as np
import pandas as pd
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.moments import set_portfolio_moments
from pyfolioanalytics.factors import ac_ranking

def test_ac_ranking_logic():
    # Synthetic data
    n_assets = 3
    asset_names = ["A", "B", "C"]
    R = pd.DataFrame(np.random.randn(10, n_assets), columns=asset_names)
    
    # We expect A < B < C
    order = ["A", "B", "C"]
    mu_rank = ac_ranking(R, order)
    
    # Order of results should match the ranks
    # order[0] is lowest, order[-1] is highest
    # mu_rank is [mu_A, mu_B, mu_C]
    # So mu_rank[0] < mu_rank[1] < mu_rank[2]
    assert mu_rank[0] < mu_rank[1]
    assert mu_rank[1] < mu_rank[2]

def test_ac_ranking_moments():
    n_assets = 4
    asset_names = ["A", "B", "C", "D"]
    R = pd.DataFrame(np.random.randn(10, n_assets), columns=asset_names)
    
    portfolio = Portfolio(assets=asset_names)
    portfolio.add_objective("StdDev")
    
    order = ["B", "A", "D", "C"] # B < A < D < C
    moments = set_portfolio_moments(R, portfolio, method="ac_ranking", order=order)
    
    mu = moments["mu"].flatten()
    # Indices in asset_names: A=0, B=1, C=2, D=3
    # B (idx 1) should be smallest, C (idx 2) largest
    assert mu[1] == np.min(mu)
    assert mu[2] == np.max(mu)
    assert mu[1] < mu[0] < mu[3] < mu[2]
