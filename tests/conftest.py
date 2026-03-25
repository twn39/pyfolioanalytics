"""
Shared pytest fixtures for the pyfolioanalytics test suite.

Session-scoped fixtures are used for data-loading operations so that CSV
files are only read once per pytest run, not once per test function.
"""

import os
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def stock_returns():
    """Session-scoped AAPL/MSFT etc. daily return data (stock_returns.csv)."""
    path = "data/stock_returns.csv"
    if not os.path.exists(path):
        pytest.skip("Stock returns data not found.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# Alias so tests that import `stocks_data` continue to work unchanged
@pytest.fixture(scope="session")
def stocks_data(stock_returns):
    return stock_returns


@pytest.fixture(scope="session")
def edhec_data():
    """Session-scoped EDHEC hedge-fund index data (edhec.csv)."""
    path = "data/edhec.csv"
    if not os.path.exists(path):
        pytest.skip("EDHEC data not found.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


@pytest.fixture(scope="session")
def small_returns():
    """Tiny synthetic return matrix for fast unit tests (no disk IO)."""
    rng = np.random.default_rng(42)
    T, N = 120, 5
    data = rng.standard_normal((T, N)) * 0.01 + 0.0005
    cols = [f"Asset.{i + 1}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)
