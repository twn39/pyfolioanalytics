import pandas as pd
import numpy as np
from pyfolioanalytics import (
    plot_weights,
    plot_efficient_frontier,
    plot_risk_decomposition,
    plot_performance
)
import plotly.graph_objects as go

def test_plot_weights():
    dates = pd.date_range("2020-01-01", periods=5)
    weights = pd.DataFrame({
        "AssetA": [0.4, 0.45, 0.5, 0.4, 0.6],
        "AssetB": [0.6, 0.55, 0.5, 0.6, 0.4]
    }, index=dates)
    fig = plot_weights(weights)
    assert isinstance(fig, go.Figure)

def test_plot_efficient_frontier():
    frontier_data = pd.DataFrame({
        "StdDev": [0.1, 0.15, 0.2],
        "mean": [0.05, 0.08, 0.12],
        "AssetA": [1.0, 0.5, 0.0],
        "AssetB": [0.0, 0.5, 1.0]
    })
    moments = {
        "mu": np.array([0.05, 0.12]),
        "sigma": np.array([[0.01, 0.0], [0.0, 0.04]])
    }
    fig = plot_efficient_frontier(frontier_data, moments)
    assert isinstance(fig, go.Figure)

def test_plot_risk_decomposition():
    ccr = pd.Series([0.02, 0.05, 0.01], index=["AssetA", "AssetB", "AssetC"])
    fig = plot_risk_decomposition(ccr)
    assert isinstance(fig, go.Figure)

def test_plot_performance():
    dates = pd.date_range("2020-01-01", periods=100)
    returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
    fig = plot_performance(returns)
    assert isinstance(fig, go.Figure)
