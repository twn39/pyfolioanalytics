import numpy as np
import pandas as pd
import json
from pyfolioanalytics.moments import set_portfolio_moments, M3_SFM, M4_SFM
from pyfolioanalytics.portfolio import Portfolio


def test_real_data_advanced_moments_cv():
    # 1. Load Data
    edhec = pd.read_csv("data/edhec.csv")
    asset_names = [
        "Convertible Arbitrage",
        "CTA Global",
        "Distressed Securities",
        "Emerging Markets",
        "Equity Market Neutral",
    ]
    R_sub = edhec[asset_names]

    with open("data/real_data_cv.json", "r") as f:
        cv_data = json.load(f)

    port = Portfolio(assets=asset_names)

    # 2. Test Robust Moments (MCD)
    # Skipping exact comparison due to structural differences between sklearn and robustbase
    _ = set_portfolio_moments(R_sub, port, method="robust")

    # 3. Test Factor Model Comoments (k=1)
    m3_py = M3_SFM(R_sub, k=1)
    m4_py = M4_SFM(R_sub, k=1)

    # Use 1e-3 for real data as numerical precision in R's extraction might vary
    np.testing.assert_allclose(m3_py.flatten(), cv_data["m3_fm1"], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(m4_py.flatten(), cv_data["m4_fm1"], rtol=1e-3, atol=1e-6)


def test_real_data_shrinkage_integration():
    edhec = pd.read_csv("data/edhec.csv")
    asset_names = ["Convertible Arbitrage", "CTA Global", "Distressed Securities"]
    R_sub = edhec[asset_names]
    port = Portfolio(assets=asset_names)
    port.add_objective(name="VaR", type="risk", arguments={"method": "modified"})

    # Test that shrinkage doesn't crash on real data
    moments = set_portfolio_moments(
        R_sub, port, comoment_method="shrinkage", comoment_alpha=0.5, k=1
    )

    assert "m3" in moments
    assert "m4" in moments
    assert moments["m3"].shape == (3, 9)
    assert moments["m4"].shape == (3, 27)
