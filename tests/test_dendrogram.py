import pandas as pd
import numpy as np
from pyfolioanalytics.plots import plot_dendrogram
import plotly.graph_objects as go


def test_plot_dendrogram():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100)
    R = pd.DataFrame(
        np.random.randn(100, 5),
        index=dates,
        columns=["AssetA", "AssetB", "AssetC", "AssetD", "AssetE"],
    )

    fig = plot_dendrogram(R)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Hierarchical Clustering Dendrogram"
    assert len(fig.data) > 0  # Should have line traces for the tree
