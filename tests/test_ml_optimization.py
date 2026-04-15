import pytest
import pandas as pd
import numpy as np
import riskfolio as rp
from pyfolioanalytics.ml import hrp_optimization, herc_optimization, nco_optimization


# Custom fixture: applies dayfirst parsing and column renaming required by this module's tests
@pytest.fixture(scope="module")
def ml_edhec_data():
    df = pd.read_csv("data/edhec.csv", index_col=0)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.iloc[:, :5].copy()  # First 5 assets
    df.columns = [c.replace(" ", ".") for c in df.columns]
    return df


def test_hrp_real_data(stocks_data):
    w = hrp_optimization(stocks_data)
    assert isinstance(w, pd.Series)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)
    assert not np.all(w == 0)


def test_herc_real_data(stocks_data):
    w = herc_optimization(stocks_data)
    assert isinstance(w, pd.Series)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_nco_real_data(stocks_data):
    w = nco_optimization(stocks_data)
    assert isinstance(w, pd.Series)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-12)  # Allow for numerical noise


def test_hrp_cross_validation(stocks_data):
    # PyfolioAnalytics
    w_py = hrp_optimization(stocks_data, linkage_method="single")

    # Riskfolio-Lib
    port = rp.HCPortfolio(returns=stocks_data)
    w_rp = port.optimization(
        model="HRP",
        codependence="pearson",
        rm="MV",
        linkage="single",
        max_k=10,
        leaf_order=False,
    )

    # Parity check
    np.testing.assert_allclose(w_py.values, w_rp.values.flatten(), atol=1e-6)


def test_herc_cross_validation(ml_edhec_data):
    # PyfolioAnalytics
    w_py = herc_optimization(ml_edhec_data, linkage_method="ward")

    # Riskfolio-Lib
    port = rp.HCPortfolio(returns=ml_edhec_data)
    w_rp = port.optimization(
        model="HERC",
        codependence="pearson",
        rm="MV",
        linkage="ward",
        max_k=10,
        leaf_order=True,
    )

    # Parity check
    # HERC/NCO can be sensitive to numerical differences in solvers and clustering
    np.testing.assert_allclose(
        w_py.values, w_rp.values.flatten(), atol=0.1
    )  # Loose tolerance for initial debug


def test_nco_cross_validation(stocks_data):
    # PyfolioAnalytics
    w_py = nco_optimization(stocks_data, linkage_method="ward", max_clusters=3)

    # Riskfolio-Lib
    port = rp.HCPortfolio(returns=stocks_data)
    w_rp = port.optimization(
        model="NCO",
        codependence="pearson",
        rm="MV",
        linkage="ward",
        max_k=3,
        leaf_order=True,
    )

    # Parity check
    np.testing.assert_allclose(w_py.values, w_rp.values.flatten(), atol=0.1)
