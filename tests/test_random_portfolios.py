import numpy as np
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.random_portfolios import random_portfolios, rp_transform


def test_rp_transform_basic():
    """Test basic weight transformation to meet sum and box constraints."""
    n_assets = 5
    portfolio = Portfolio(assets=n_assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    # Generate 10 random portfolios via transform
    # The current rp_transform implementation in src takes an integer 'permutations'
    # and returns an array of portfolios.
    rps = random_portfolios(portfolio, permutations=10, method="transform")

    assert rps.shape == (10, n_assets)
    for i in range(10):
        assert np.isclose(np.sum(rps[i]), 1.0)
        assert np.all(rps[i] >= -1e-12)


def test_random_portfolios_generation_simplex():
    """Test generating a set of random portfolios using simplex method."""
    n_assets = 10
    portfolio = Portfolio(assets=n_assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    n_permutations = 50
    rps = random_portfolios(portfolio, permutations=n_permutations, method="simplex")

    assert rps.shape == (n_permutations, n_assets)
    for i in range(n_permutations):
        assert np.isclose(np.sum(rps[i]), 1.0)
        assert np.all(rps[i] >= -1e-12)


def test_random_portfolios_transform_simple():
    """Test generating random portfolios using transform method."""
    n_assets = 5
    portfolio = Portfolio(assets=n_assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    # Test rp_transform directly
    rps = rp_transform(portfolio, permutations=5)
    assert rps.shape == (5, n_assets)
    assert np.allclose(rps.sum(axis=1), 1.0)
