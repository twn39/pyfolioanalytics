import pandas as pd
import numpy as np
from pyfolioanalytics.discrete_allocation import DiscreteAllocation as PA_DA
from pypfopt.discrete_allocation import DiscreteAllocation as PP_DA


def test_discrete_allocation_pypfopt_parity():
    # Setup common inputs
    weights = {"AAPL": 0.45, "GOOG": 0.25, "TSLA": 0.15, "MSFT": 0.10, "AMZN": 0.05}
    latest_prices = pd.Series(
        {"AAPL": 180.5, "GOOG": 2850.2, "TSLA": 700.1, "MSFT": 310.4, "AMZN": 3300.5}
    )
    total_val = 50000.0

    # 1. Test Greedy Portfolio Parity
    pa_da = PA_DA(weights, latest_prices, total_portfolio_value=total_val)
    pa_alloc, pa_leftover = pa_da.greedy_portfolio()

    pp_da = PP_DA(weights, latest_prices, total_portfolio_value=total_val)
    pp_alloc, pp_leftover = pp_da.greedy_portfolio()

    # Assert Greedy parity
    assert pa_alloc == pp_alloc
    np.testing.assert_allclose(pa_leftover, pp_leftover, rtol=1e-7)

    # 2. Test LP Portfolio Parity
    # Note: Both use CVXPY, so they should find the same optimal integer solution
    pa_alloc_lp, pa_left_lp = pa_da.lp_portfolio()
    pp_alloc_lp, pp_left_lp = pp_da.lp_portfolio()

    # Assert LP parity
    assert pa_alloc_lp == pp_alloc_lp
    np.testing.assert_allclose(pa_left_lp, pp_left_lp, rtol=1e-7)


def test_discrete_allocation_shorts_parity():
    # Setup inputs with shorts
    weights = {"AAPL": 1.2, "GOOG": -0.2}
    latest_prices = pd.Series({"AAPL": 150.0, "GOOG": 2800.0})
    total_val = 10000.0

    pa_da = PA_DA(weights, latest_prices, total_portfolio_value=total_val)
    pa_alloc, pa_leftover = pa_da.greedy_portfolio()

    pp_da = PP_DA(weights, latest_prices, total_portfolio_value=total_val)
    pp_alloc, pp_leftover = pp_da.greedy_portfolio()

    # Assert parity for shorts
    assert pa_alloc == pp_alloc
    np.testing.assert_allclose(pa_leftover, pp_leftover, rtol=1e-7)
