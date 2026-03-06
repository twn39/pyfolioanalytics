import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.moments import set_portfolio_moments
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.factors import statistical_factor_model, factor_model_covariance
from pyfolioanalytics.meucci import entropy_pooling, meucci_moments


@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
    return df


def test_statistical_factor_model(stocks_data):
    # Test PCA extraction
    k = 3
    res = statistical_factor_model(stocks_data, k=k)

    assert res["factors"].shape == (len(stocks_data), k)
    assert res["loadings"].shape == (stocks_data.shape[1], k)
    assert res["alpha"].shape == (stocks_data.shape[1],)

    # Test covariance reconstruction
    sigma_fm = factor_model_covariance(res)
    assert sigma_fm.shape == (stocks_data.shape[1], stocks_data.shape[1])
    # Diagonals should be positive
    assert np.all(np.diag(sigma_fm) > 0)

    # Check if Sigma_fm is close to sample covariance for high k
    res_full = statistical_factor_model(stocks_data, k=stocks_data.shape[1])
    sigma_full = factor_model_covariance(res_full)
    np.testing.assert_allclose(sigma_full, stocks_data.cov().values, atol=1e-5)


def test_shrinkage_moments(stocks_data):
    portfolio = Portfolio(assets=stocks_data.columns.tolist())
    moments = set_portfolio_moments(stocks_data, portfolio, method="shrinkage")

    assert moments["mu"].shape == (5, 1)
    assert moments["sigma"].shape == (5, 5)
    # Shrinkage covariance should differ from sample covariance
    assert not np.array_equal(moments["sigma"], stocks_data.cov().values)


def test_meucci_entropy_pooling(stocks_data):
    # Prior: equal probabilities
    T = len(stocks_data)
    prior = np.full(T, 1.0 / T)

    # Constraint: Posterior mean of the first asset must be exactly 0
    # sum(p_t * R_{t,1}) = 0
    R1 = stocks_data.iloc[:, 0].values
    Aeq = R1.reshape(1, -1)
    beq = np.zeros(1)

    # Add normalization constraint sum(p) = 1 (Entropy pooling handles this implicitly or via Aeq)
    # The current implementation handles normalization internally.

    p_post = entropy_pooling(prior, Aeq=Aeq, beq=beq)

    assert len(p_post) == T
    assert np.isclose(np.sum(p_post), 1.0)
    assert np.all(p_post >= 0)

    # Verify constraint satisfaction
    post_mean = np.sum(p_post * R1)
    assert np.abs(post_mean) < 1e-8

    # Test shifted moments
    moments = meucci_moments(stocks_data.values, p_post)
    assert np.isclose(moments["mu"][0, 0], 0.0, atol=1e-8)


def test_set_portfolio_moments_meucci(stocks_data):
    portfolio = Portfolio(assets=stocks_data.columns.tolist())
    T = len(stocks_data)
    R1 = stocks_data.iloc[:, 0].values
    Aeq = R1.reshape(1, -1)
    beq = np.array([0.001])  # Target mean

    moments = set_portfolio_moments(
        stocks_data, portfolio, method="meucci", Aeq=Aeq, beq=beq
    )
    assert np.isclose(moments["mu"][0, 0], 0.001, atol=1e-7)
