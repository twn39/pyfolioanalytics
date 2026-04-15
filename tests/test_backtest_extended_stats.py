import pandas as pd
import numpy as np
from pyfolioanalytics.backtest import BacktestResult


def test_extended_summary_stats():
    # Construct a synthetic returns series with some positive and negative days
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.05, -0.03, 0.02, 0.01, -0.01, 0.04], index=dates
    )
    benchmark = pd.Series(
        [0.005, -0.01, 0.02, -0.005, 0.03, -0.02, 0.01, 0.005, -0.005, 0.02],
        index=dates,
    )
    weights = pd.DataFrame(1, index=dates, columns=["A"])

    br = BacktestResult(weights, returns, [])

    # Calculate summary without benchmark
    summary = br.summary()
    assert "Omega Ratio" in summary.index
    assert "Tail Ratio" in summary.index
    assert "Hit Ratio" in summary.index
    assert "Skewness" in summary.index
    assert "Kurtosis" in summary.index
    assert "Daily VaR (95%)" in summary.index
    assert "Daily CVaR (95%)" in summary.index
    assert np.isnan(summary.loc["Tracking Error", "Gross"])

    # Calculate summary with benchmark
    summary_bench = br.summary(benchmark_returns=benchmark)
    assert not np.isnan(summary_bench.loc["Tracking Error", "Gross"])
    assert not np.isnan(summary_bench.loc["Information Ratio", "Gross"])
    assert not np.isnan(summary_bench.loc["Up Capture Ratio", "Gross"])
    assert not np.isnan(summary_bench.loc["Down Capture Ratio", "Gross"])


def test_underwater_drawdown_plot():
    from pyfolioanalytics.plots import plot_underwater_drawdown

    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.05, -0.02, 0.03, 0.05], index=dates)
    fig = plot_underwater_drawdown(returns)
    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6
