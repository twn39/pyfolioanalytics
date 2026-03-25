import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional


def plot_weights(weights: pd.DataFrame, title: str = "Portfolio Weights over Time") -> go.Figure:
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
                name=str(col)
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


def plot_efficient_frontier(
    frontier_data: pd.DataFrame,
    moments: Optional[Dict[str, Any]] = None,
    risk_col: str = "StdDev",
    return_col: str = "mean",
    title: str = "Efficient Frontier"
) -> go.Figure:
    """
    Plot the efficient frontier curve and optionally the underlying asset scatters.
    """
    fig = go.Figure()
    
    if risk_col not in frontier_data.columns and "sd" in frontier_data.columns:
        risk_col = "sd"

    if risk_col not in frontier_data.columns or return_col not in frontier_data.columns:
        raise ValueError(f"frontier_data must contain {risk_col} and {return_col} columns")

    # Plot Efficient Frontier
    fig.add_trace(
        go.Scatter(
            x=frontier_data[risk_col],
            y=frontier_data[return_col],
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(shape="spline", smoothing=1.3, width=2),
            marker=dict(size=6)
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
        asset_names = [col for col in frontier_data.columns if col not in [risk_col, return_col]]
        if len(asset_names) != len(asset_returns):
            asset_names = [f"Asset {i}" for i in range(len(asset_returns))]
            
        fig.add_trace(
            go.Scatter(
                x=asset_sd,
                y=asset_returns,
                mode="markers",
                name="Assets",
                text=asset_names,
                marker=dict(size=8, symbol="diamond")
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"Risk ({risk_col})",
        yaxis_title=f"Return ({return_col})",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
        template="plotly_white"
    )
    
    return fig


def plot_risk_decomposition(
    ccr: pd.Series,
    title: str = "Component Contribution to Risk"
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
            marker=dict(
                color=ccr_sorted.values,
                colorscale="Viridis",
                showscale=False
            )
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Risk Contribution",
        yaxis_title="Asset",
        xaxis_tickformat=".2%",
        template="plotly_white"
    )
    
    return fig


def plot_performance(
    returns: pd.Series,
    title: str = "Portfolio Performance"
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
        subplot_titles=("Cumulative Returns", "Drawdown")
    )
    
    # Cumulative Returns trace
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode="lines",
            name="Cumulative Return",
            line=dict(width=2, color="blue")
        ),
        row=1, col=1
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
            fillcolor="rgba(255, 0, 0, 0.3)"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        height=700,
        hovermode="x unified",
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig
