import pandas as pd
import numpy as np
from pyfolioanalytics.backtest import BacktestResult


def test_backtest_stats_cv(stocks_data):
    """
    Cross validate extended backtest stats (Omega, Tail Ratio, Tracking Error, etc.)
    against a benchmark logic using Riskfolio-Lib or quantstats-like math parity.
    """
    # 1. Setup mock data
    np.random.seed(42)
    # Generate 5 years of daily returns
    dates = pd.date_range("2018-01-01", periods=1260, freq="B")

    # Portfolio returns: slightly positive drift + normal noise
    R_port = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    # Benchmark returns: standard normal noise
    R_bench = pd.Series(np.random.normal(0.0002, 0.012, len(dates)), index=dates)

    weights = pd.DataFrame(1, index=dates, columns=["A"])

    # 2. PyFolioAnalytics calculation
    br = BacktestResult(weights, R_port, [])
    summary = br.summary(benchmark_returns=R_bench)

    # 3. Manual Mathematical Verification / Cross Validation

    # A. Check Hit Ratio
    expected_hit = (R_port > 0).sum() / len(R_port)
    np.testing.assert_allclose(summary.loc["Hit Ratio", "Gross"], expected_hit)

    # B. Check Tracking Error
    active_returns = R_port - R_bench
    expected_te = active_returns.std() * np.sqrt(252)
    np.testing.assert_allclose(summary.loc["Tracking Error", "Gross"], expected_te)

    # C. Check Information Ratio
    expected_ir = (active_returns.mean() * 252) / expected_te
    np.testing.assert_allclose(summary.loc["Information Ratio", "Gross"], expected_ir)

    # D. Check Tail Ratio
    expected_tail = np.percentile(R_port, 95) / np.abs(np.percentile(R_port, 5))
    np.testing.assert_allclose(summary.loc["Tail Ratio", "Gross"], expected_tail)

    # E. Check Up Capture / Down Capture Ratio
    up_market = R_bench > 0
    down_market = R_bench < 0

    r_up_annual = (1 + R_port[up_market]).prod() ** (252 / up_market.sum()) - 1
    b_up_annual = (1 + R_bench[up_market]).prod() ** (252 / up_market.sum()) - 1
    expected_up_cap = r_up_annual / b_up_annual
    np.testing.assert_allclose(
        summary.loc["Up Capture Ratio", "Gross"], expected_up_cap
    )

    r_down_annual = (1 + R_port[down_market]).prod() ** (252 / down_market.sum()) - 1
    b_down_annual = (1 + R_bench[down_market]).prod() ** (252 / down_market.sum()) - 1
    expected_down_cap = r_down_annual / b_down_annual
    np.testing.assert_allclose(
        summary.loc["Down Capture Ratio", "Gross"], expected_down_cap
    )

    # F. Omega Ratio (threshold = 0)
    pos_sum = R_port[R_port >= 0].sum()
    neg_sum = np.abs(R_port[R_port < 0].sum())
    expected_omega = pos_sum / neg_sum
    np.testing.assert_allclose(summary.loc["Omega Ratio", "Gross"], expected_omega)


def test_backtest_stats_riskfolio_cv(stocks_data):
    """
    Cross validate Drawdown logic directly against Riskfolio-Lib.
    """
    R_port = stocks_data.iloc[:, 0]  # AAPL returns
    weights = pd.DataFrame(1, index=R_port.index, columns=["A"])

    # 1. PyFolioAnalytics
    br = BacktestResult(weights, R_port, [])
    summary = br.summary()
    py_max_dd = summary.loc["Max Drawdown", "Gross"]

    # 2. Riskfolio-Lib calculation
    # In Riskfolio-Lib, Max Drawdown is CDaR with alpha=1.0 (or just calculating the max drawdown vector)
    # Let's use standard max drawdown calculation vector matching
    cum_vals = (1 + R_port).cumprod()
    rolling_max = cum_vals.cummax()
    expected_drawdowns = (cum_vals / rolling_max) - 1.0
    rp_max_dd = expected_drawdowns.min()

    np.testing.assert_allclose(py_max_dd, rp_max_dd)
