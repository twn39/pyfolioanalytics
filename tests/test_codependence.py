import numpy as np
from pyfolioanalytics.codependence import get_codependence_matrix, get_distance_matrix


def test_codependence_pearson(stocks_data):
    R = stocks_data.iloc[:, :5]
    corr = get_codependence_matrix(R, method="pearson")
    assert corr.shape == (5, 5)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0)


def test_codependence_distance(stocks_data):
    R = stocks_data.iloc[:, :3]
    dcor = get_codependence_matrix(R, method="distance")
    assert dcor.shape == (3, 3)
    assert np.allclose(np.diag(dcor), 1.0)
    assert np.all(dcor >= 0.0) and np.all(dcor <= 1.0)


def test_codependence_mutual_info(stocks_data):
    R = stocks_data.iloc[:, :3]
    mi = get_codependence_matrix(R, method="mutual_info", bins=10)
    assert mi.shape == (3, 3)
    assert np.allclose(np.diag(mi), 1.0)
    assert np.all(mi >= 0.0) and np.all(mi <= 1.0)


def test_codependence_tail(stocks_data):
    R = stocks_data.iloc[:, :4]
    td = get_codependence_matrix(R, method="tail", q=0.05)
    assert td.shape == (4, 4)
    assert np.allclose(np.diag(td), 1.0)
    assert np.all(td >= 0.0) and np.all(td <= 1.0)


def test_distance_matrix():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])

    # Standard distance: sqrt(0.5 * (1 - 0.5)) = sqrt(0.25) = 0.5
    dist_std = get_distance_matrix(corr, method="standard")
    assert np.allclose(dist_std[0, 1], 0.5)
    assert dist_std[0, 0] == 0.0

    # Variation of information: 1 - rho
    dist_vi = get_distance_matrix(corr, method="variation_of_information")
    assert np.allclose(dist_vi[0, 1], 0.5)
    assert dist_vi[0, 0] == 0.0


def test_custom_distance(stocks_data):
    from pyfolioanalytics.portfolio import Portfolio
    from pyfolioanalytics.optimize import optimize_portfolio

    R = stocks_data.iloc[:, :4]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment").add_constraint(type="long_only")

    custom_dist = np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.2],
            [0.9, 0.8, 0.2, 0.0],
        ]
    )

    res = optimize_portfolio(
        R, port, optimize_method="HRP", distance="custom", custom_distance=custom_dist
    )
    assert res["status"] == "optimal"
    assert len(res["weights"]) == 4
