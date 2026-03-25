from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_weights(
    weights: pd.DataFrame, title: str = "Portfolio Weights over Time"
) -> go.Figure:
    """
    Plot an area chart of portfolio weights over time.
    """
    fig = go.Figure()

    for col in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[col],
                mode="lines",
                stackgroup="one",
                name=str(col),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_efficient_frontier(
    frontier_data: pd.DataFrame,
    moments: dict[str, Any] | None = None,
    risk_col: str = "StdDev",
    return_col: str = "mean",
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Plot the efficient frontier curve and optionally the underlying asset scatters.
    """
    fig = go.Figure()

    if risk_col not in frontier_data.columns and "sd" in frontier_data.columns:
        risk_col = "sd"

    if risk_col not in frontier_data.columns or return_col not in frontier_data.columns:
        raise ValueError(
            f"frontier_data must contain {risk_col} and {return_col} columns"
        )

    # Plot Efficient Frontier
    fig.add_trace(
        go.Scatter(
            x=frontier_data[risk_col],
            y=frontier_data[return_col],
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(shape="spline", smoothing=1.3, width=2),
            marker=dict(size=6),
        )
    )

    # Plot Individual Assets if moments are provided
    if moments is not None and "mu" in moments and "sigma" in moments:
        mu = moments["mu"]
        sigma = moments["sigma"]
        # Determine std deviation (annualized or period depending on inputs, usually matches frontier)
        asset_sd = np.sqrt(np.diag(sigma))
        asset_returns = mu

        # We need asset names, which are not directly in moments usually,
        # but we can try to guess from frontier_data columns or just use indices.
        asset_names = [
            col for col in frontier_data.columns if col not in [risk_col, return_col]
        ]
        if len(asset_names) != len(asset_returns):
            asset_names = [f"Asset {i}" for i in range(len(asset_returns))]

        fig.add_trace(
            go.Scatter(
                x=asset_sd,
                y=asset_returns,
                mode="markers",
                name="Assets",
                text=asset_names,
                marker=dict(size=8, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"Risk ({risk_col})",
        yaxis_title=f"Return ({return_col})",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
        template="plotly_white",
    )

    return fig


def plot_risk_decomposition(
    ccr: pd.Series, title: str = "Component Contribution to Risk"
) -> go.Figure:
    """
    Plot a bar chart showing the Component Contribution to Risk (CCR) of each asset.
    """
    fig = go.Figure()

    # Sort for better visualization
    ccr_sorted = ccr.sort_values(ascending=True)

    fig.add_trace(
        go.Bar(
            x=ccr_sorted.values,
            y=ccr_sorted.index,
            orientation="h",
            marker=dict(color=ccr_sorted.values, colorscale="Viridis", showscale=False),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Risk Contribution",
        yaxis_title="Asset",
        xaxis_tickformat=".2%",
        template="plotly_white",
    )

    return fig


def plot_dendrogram(
    R: pd.DataFrame,
    linkage_method: str = "ward",
    title: str = "Hierarchical Clustering Dendrogram",
) -> go.Figure:
    """
    Plot a dendrogram using hierarchical clustering on asset returns.
    This visualizes the clustering process used in HRP/HERC.
    """
    import plotly.figure_factory as ff
    from scipy.cluster.hierarchy import linkage

    # 1. Calculate correlation and distance matrix
    corr = R.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    # Fill diagonal with 0 just in case of numerical issues
    np.fill_diagonal(dist, 0)

    # 2. Extract asset names
    labels = R.columns.tolist()

    # 3. Define custom distance and linkage functions for Plotly's create_dendrogram
    # plotly create_dendrogram expects X to be the observation matrix.
    # It passes X to distfun. We will pre-compute distance and pass a dummy distfun,
    # or just pass the distance matrix directly and override linkagefun.

    # Actually, the most robust way with plotly is to let it do the linkage
    # Since we already have the distance matrix (not observation matrix),
    # we can pass the distance matrix if we define distfun to just return squareform
    from scipy.spatial.distance import squareform

    dist_condensed = squareform(dist)

    def custom_linkage(x):
        # x is ignored because we capture dist_condensed from closure
        return linkage(dist_condensed, method=linkage_method)

    # Plotly's create_dendrogram requires an X of shape (N, ...).
    # We just pass a dummy array of shape (N, 1) and ignore it in custom_linkage.
    dummy_X = np.zeros((len(labels), 1))

    fig = ff.create_dendrogram(
        dummy_X,
        labels=labels,
        linkagefun=custom_linkage,
        color_threshold=None,  # Default threshold
    )

    fig.update_layout(
        title=title,
        xaxis_title="Assets",
        yaxis_title="Distance",
        template="plotly_white",
        width=800,
        height=500,
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    return fig


def plot_performance(
    returns: pd.Series, title: str = "Portfolio Performance"
) -> go.Figure:
    """
    Plot cumulative returns curve and underwater drawdown plot.
    """
    # Calculate cumulative return
    cum_returns = (1 + returns).cumprod()

    # Calculate rolling maximum and drawdown
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns / rolling_max) - 1.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Returns", "Drawdown"),
    )

    # Cumulative Returns trace
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode="lines",
            name="Cumulative Return",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=1,
    )

    # Drawdown trace
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(width=1, color="red"),
            fillcolor="rgba(255, 0, 0, 0.3)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title, height=700, hovermode="x unified", template="plotly_white"
    )

    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def plot_underwater_drawdown(backtest_res, title="Underwater Drawdown", show=True):
    import plotly.graph_objects as go

    returns = backtest_res.net_returns
    cum_vals = (1 + returns).cumprod()
    rolling_max = cum_vals.cummax()
    drawdowns = (cum_vals / rolling_max) - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns,
            fill="tozeroy",
            name="Drawdown",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title="Drawdown",
        yaxis_tickformat=".2%",
        template="plotly_white",
    )
    if show:
        fig.show()
    return fig
