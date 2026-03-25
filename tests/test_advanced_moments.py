import numpy as np
import pandas as pd
import json
from pyfolioanalytics.risk import owa_l_moment_weights, l_moment
from pyfolioanalytics.moments import set_portfolio_moments
from pyfolioanalytics.portfolio import Portfolio


def test_l_moments_cv():
    with open("data/l_moments_cv.json", "r") as f:
        cv_data = json.load(f)

    T = cv_data["T"]
    returns = np.array(cv_data["returns"])
    weights = np.array(cv_data["weights"])

    # Test weights
    for k in [2, 3, 4]:
        expected_w = np.array(cv_data[f"owa_weights_k{k}"])
        actual_w = owa_l_moment_weights(T, k=k)
        np.testing.assert_allclose(actual_w.flatten(), expected_w.flatten(), rtol=1e-10)

    # Test L-moments
    actual_l2 = l_moment(returns, weights, k=2)
    actual_l3 = l_moment(returns, weights, k=3)
    actual_l4 = l_moment(returns, weights, k=4)

    np.testing.assert_allclose(actual_l2, cv_data["l2"], rtol=1e-10)
    np.testing.assert_allclose(actual_l3, cv_data["l3"], rtol=1e-10)
    np.testing.assert_allclose(actual_l4, cv_data["l4"], rtol=1e-10)


def test_robust_covariance_cv():
    with open("data/l_moments_cv.json", "r") as f:
        cv_data = json.load(f)

    R_df = pd.DataFrame(cv_data["returns"])
    asset_names = [f"Asset.{i + 1}" for i in range(len(R_df.columns))]
    R_df.columns = asset_names
    port = Portfolio(assets=asset_names)

    # Call set_portfolio_moments with method="robust"
    moments = set_portfolio_moments(R_df, port, method="robust")

    # Compare mu and sigma
    expected_mu = np.array(cv_data["mu_robust"]).reshape(-1, 1)
    expected_sigma = np.array(cv_data["sigma_robust"])

    np.testing.assert_allclose(moments["mu"], expected_mu, rtol=1e-10)
    np.testing.assert_allclose(moments["sigma"], expected_sigma, rtol=1e-10)


def test_denoised_covariance_fixed():
    # Test denoise consistency (internal logic verification)
    np.random.seed(42)
    T, N = 200, 50
    R = np.random.randn(T, N)
    R_df = pd.DataFrame(R)
    asset_names = [f"Asset.{i + 1}" for i in range(N)]
    R_df.columns = asset_names
    port = Portfolio(assets=asset_names)

    moments_raw = set_portfolio_moments(R_df, port, method="sample")
    moments_denoised = set_portfolio_moments(
        R_df, port, method="denoised", denoise_method="fixed"
    )

    # Denoised sigma should be different from raw sigma
    assert not np.array_equal(moments_raw["sigma"], moments_denoised["sigma"])
    # But it should be a valid covariance matrix
    assert np.all(np.linalg.eigvals(moments_denoised["sigma"]) >= -1e-10)


def test_ewma_moments():
    np.random.seed(42)
    T, N = 100, 5
    R = np.random.randn(T, N)
    R_df = pd.DataFrame(R, columns=[f"Asset.{i + 1}" for i in range(N)])
    port = Portfolio(assets=list(R_df.columns))

    moments = set_portfolio_moments(R_df, port, method="ewma", span=36)
    assert "mu" in moments
    assert "sigma" in moments
    assert moments["mu"].shape == (N, 1)
    assert moments["sigma"].shape == (N, N)
    # Check PSD
    assert np.all(np.linalg.eigvals(moments["sigma"]) >= -1e-10)


def test_semi_covariance_moments():
    np.random.seed(42)
    T, N = 100, 5
    R = np.random.randn(T, N)
    R_df = pd.DataFrame(R, columns=[f"Asset.{i + 1}" for i in range(N)])
    port = Portfolio(assets=list(R_df.columns))

    moments = set_portfolio_moments(R_df, port, method="semi_covariance", benchmark=0.0)
    assert "mu" in moments
    assert "sigma" in moments
    assert moments["mu"].shape == (N, 1)
    assert moments["sigma"].shape == (N, N)
    # Semi-covariance must be PSD
    assert np.all(np.linalg.eigvals(moments["sigma"]) >= -1e-10)
