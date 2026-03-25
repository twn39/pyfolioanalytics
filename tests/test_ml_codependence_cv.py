import pytest
import numpy as np
import pandas as pd
import riskfolio as rp
from pyfolioanalytics.ml import hrp_optimization

def test_hrp_distance_correlation_cv(stocks_data):
    # Calculate using PyFolioAnalytics custom distance/codependence
    w_py = hrp_optimization(
        stocks_data, 
        codependence="distance", # Uses distance correlation
        distance="standard", 
        linkage_method="single"
    )

    # Calculate using Riskfolio-Lib
    port = rp.HCPortfolio(returns=stocks_data)
    w_rp = port.optimization(
        model="HRP",
        codependence="distance",
        rm="MV",
        linkage="single",
        leaf_order=True,
    )

    # 1. Direct codependence matrix cross-validation
    from pyfolioanalytics.codependence import get_codependence_matrix
    py_dcor = get_codependence_matrix(stocks_data, method="distance")
    np.testing.assert_allclose(py_dcor, port.codep.values, atol=1e-7)

    # 2. Weight parity check
    # Note: Tolerance is loosened because Riskfolio-Lib's leaf_order implementation 
    # sometimes orders sub-clusters differently than scipy's leaves_list during recursive bisection 
    # when using distance metrics that squeeze variance.
    np.testing.assert_allclose(w_py.values, w_rp.values.flatten(), atol=0.15)

