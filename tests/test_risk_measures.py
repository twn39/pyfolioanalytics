import pytest
import pandas as pd
import numpy as np
from pyfolioanalytics.risk import (
    VaR,
    ES,
    EVaR,
    max_drawdown,
    average_drawdown,
    CDaR,
    calculate_drawdowns,
)
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.random_portfolios import random_portfolios


@pytest.fixture
def stocks_data():
    df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
    return df


def test_drawdowns(stocks_data):
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    R = stocks_data.values
    p_returns = R @ weights

    dd = calculate_drawdowns(p_returns)
    assert len(dd) == len(p_returns)
    assert np.all(dd <= 0)
    assert dd[0] <= 0  # Initial could be 0 or small negative

    mdd = max_drawdown(weights, R)
    assert mdd >= 0
    assert mdd == -np.min(dd)

    add = average_drawdown(weights, R)
    assert add >= 0
    assert add <= mdd


def test_var_measures(stocks_data):
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    mu = stocks_data.mean().values
    sigma = stocks_data.cov().values

    var_g = VaR(weights, mu, sigma, p=0.95, method="gaussian")
    assert var_g > 0

    es_g = ES(weights, mu, sigma, p=0.95, method="gaussian")
    assert es_g >= var_g  # ES should be worse than VaR


def test_entropic_var(stocks_data):
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    R = stocks_data.values

    evar = EVaR(weights, R, p=0.95)
    assert evar > 0

    # Check if EVaR is sensitive to p
    evar_low = EVaR(weights, R, p=0.90)
    assert evar > evar_low


def test_cdar(stocks_data):
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    R = stocks_data.values

    cdar_95 = CDaR(weights, R, p=0.95)
    cdar_90 = CDaR(weights, R, p=0.90)

    assert cdar_95 >= 0
    assert cdar_95 >= cdar_90


def test_random_portfolios():
    assets = ["A", "B", "C"]
    portfolio = Portfolio(assets=assets)
    portfolio.add_constraint(type="full_investment")
    portfolio.add_constraint(type="long_only")

    # Simplex
    rp_s = random_portfolios(portfolio, permutations=50, method="simplex")
    assert rp_s.shape == (50, 3)
    assert np.allclose(rp_s.sum(axis=1), 1.0)
    assert np.all(rp_s >= 0)

    # Transform
    rp_t = random_portfolios(portfolio, permutations=50, method="transform")
    assert rp_t.shape == (50, 3)
    assert np.allclose(rp_t.sum(axis=1), 1.0)
    assert np.all(rp_t >= 0)
