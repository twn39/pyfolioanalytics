import pandas as pd
from pyfolioanalytics.discrete_allocation import DiscreteAllocation, get_latest_prices


def test_get_latest_prices():
    df = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22]})
    latest = get_latest_prices(df)
    assert latest["A"] == 12
    assert latest["B"] == 22


def test_greedy_portfolio_long_only():
    weights = {"AAPL": 0.5, "GOOG": 0.5}
    latest_prices = pd.Series({"AAPL": 150, "GOOG": 2800})
    total_val = 10000

    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_val)
    allocation, leftover = da.greedy_portfolio()

    # AAPL: 0.5 * 10000 / 150 = 33.33 -> 33 shares
    # GOOG: 0.5 * 10000 / 2800 = 1.78 -> 1 share
    # Total spent: 33 * 150 + 1 * 2800 = 4950 + 2800 = 7750
    # Leftover: 10000 - 7750 = 2250
    # Round 2: Can we buy more GOOG? No (2800 > 2250). Can we buy more AAPL? Yes (150 < 2250).
    # Buying as many AAPL as possible: 2250 // 150 = 15 shares.
    # Final AAPL: 33 + 15 = 48 shares.
    # Total spent: 48 * 150 + 1 * 2800 = 7200 + 2800 = 10000. Leftover: 0.

    assert allocation["AAPL"] == 48
    assert allocation["GOOG"] == 1
    assert leftover == 0


def test_lp_portfolio_long_only():
    weights = {"AAPL": 0.6, "GOOG": 0.4}
    latest_prices = pd.Series({"AAPL": 150, "GOOG": 2800})
    total_val = 10000

    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_val)
    allocation, leftover = da.lp_portfolio()

    total_spent = sum(allocation[t] * latest_prices[t] for t in allocation)
    assert total_spent <= total_val
    assert leftover >= 0
    # LP might trade off some cash for better weight alignment
    assert leftover < total_val * 0.2


def test_greedy_portfolio_shorts():
    weights = {"AAPL": 1.3, "GOOG": -0.3}
    latest_prices = pd.Series({"AAPL": 150, "GOOG": 2800})
    total_val = 10000

    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_val)
    allocation, leftover = da.greedy_portfolio()

    assert allocation["AAPL"] > 0
    assert allocation["GOOG"] < 0

    allocation["AAPL"] * latest_prices["AAPL"]
    total_short = -allocation["GOOG"] * latest_prices["GOOG"]

    # Short value should be approx 0.3 * 10000 = 3000
    assert 2800 <= total_short <= 5600  # At least 1 share
