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


def plot_random_portfolios(
    random_weights: np.ndarray | pd.DataFrame,
    moments: dict[str, Any],
    objectives: list[dict[str, Any]],
    R: pd.DataFrame | np.ndarray | None = None,
    risk_col: str = "StdDev",
    return_col: str = "mean",
    optimal_weights: pd.Series | np.ndarray | None = None,
    title: str = "Random Portfolios Feasible Space",
) -> go.Figure:
    """
    Plot a scatter cloud of random portfolios to visualize the feasible risk-return space.
    Optionally overlays the optimal portfolio as a highlighted star.
    """
    from .optimize import calculate_objective_measures

    # Ensure random_weights is a 2D numpy array
    if isinstance(random_weights, pd.DataFrame):
        w_mat = random_weights.values
    else:
        w_mat = random_weights

    n_ports = w_mat.shape[0]
    returns_arr = np.zeros(n_ports)
    risks_arr = np.zeros(n_ports)
    ratios_arr = np.zeros(n_ports)
    hover_texts = []

    # Prepare historical returns R for calculating complex risk measures if needed
    R_vals = R.values if isinstance(R, pd.DataFrame) else R

    # Pre-extract asset names for hover text if available
    asset_names = None
    if isinstance(random_weights, pd.DataFrame):
        asset_names = random_weights.columns.tolist()
    elif isinstance(R, pd.DataFrame):
        asset_names = R.columns.tolist()

    for i in range(n_ports):
        w = w_mat[i]
        measures = calculate_objective_measures(w, moments, objectives, R=R_vals)
        
        # Default fallbacks if the measure wasn't calculated (e.g., objective disabled)
        ret = measures.get(return_col, np.dot(w, moments["mu"]).item())
        if risk_col in measures:
            risk = measures[risk_col]
        else:
            # Fallback to standard deviation if specific risk wasn't calculated
            risk = np.sqrt(max(0.0, np.dot(w.T, np.dot(moments["sigma"], w))))
            
        returns_arr[i] = ret
        risks_arr[i] = risk
        ratio = ret / risk if risk > 1e-8 else 0.0
        ratios_arr[i] = ratio

        # Construct hover text
        hover = f"Return: {ret:.4%}<br>Risk ({risk_col}): {risk:.4%}<br>Ratio: {ratio:.4f}"
        if asset_names is not None:
            # Find top 3 allocations
            top_indices = np.argsort(w)[-3:][::-1]
            top_allocs = "<br>".join([f"{asset_names[idx]}: {w[idx]:.1%}" for idx in top_indices if w[idx] > 0.01])
            if top_allocs:
                hover += f"<br><br>Top Allocations:<br>{top_allocs}"
        hover_texts.append(hover)

    fig = go.Figure()

    # Use Scattergl for high performance rendering of thousands of points
    fig.add_trace(
        go.Scattergl(
            x=risks_arr,
            y=returns_arr,
            mode="markers",
            name="Random Portfolios",
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=5,
                color=ratios_arr,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Return/Risk Ratio"),
                opacity=0.6,
                line=dict(width=0.5, color="white")
            ),
        )
    )

    # Overlay optimal portfolio if provided
    if optimal_weights is not None:
        if isinstance(optimal_weights, pd.Series):
            w_opt = optimal_weights.values
        else:
            w_opt = optimal_weights
            
        opt_measures = calculate_objective_measures(w_opt, moments, objectives, R=R_vals)
        opt_ret = opt_measures.get(return_col, np.dot(w_opt, moments["mu"]).item())
        
        if risk_col in opt_measures:
            opt_risk = opt_measures[risk_col]
        else:
            opt_risk = np.sqrt(max(0.0, np.dot(w_opt.T, np.dot(moments["sigma"], w_opt))))

        fig.add_trace(
            go.Scatter(
                x=[opt_risk],
                y=[opt_ret],
                mode="markers",
                name="Optimal Portfolio",
                hoverinfo="x+y+name",
                marker=dict(
                    symbol="star",
                    size=16,
                    color="gold",
                    line=dict(width=2, color="DarkSlateGrey")
                )
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"Risk ({risk_col})",
        yaxis_title=f"Return ({return_col})",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    )

    return fig


def plot_efficient_frontier(
    frontier_data: pd.DataFrame,
    moments: dict[str, Any] | None = None,
    risk_col: str = "StdDev",
    return_col: str = "mean",
    title: str = "Efficient Frontier",
    annualize_factor: int | float | None = None,
) -> go.Figure:
    """
    Plot the efficient frontier curve and optionally the underlying asset scatters.
    If `annualize_factor` is provided (e.g., 252 for daily data), returns are multiplied 
    by this factor, and risk measures are scaled appropriately (sqrt for volatility-based, 
    linear for drawdowns).
    """
    fig = go.Figure()

    if risk_col not in frontier_data.columns and "sd" in frontier_data.columns:
        risk_col = "sd"

    if risk_col not in frontier_data.columns or return_col not in frontier_data.columns:
        raise ValueError(
            f"frontier_data must contain {risk_col} and {return_col} columns"
        )
        
    # Determine scaling factors
    ret_scale = 1.0
    risk_scale = 1.0
    if annualize_factor is not None:
        ret_scale = float(annualize_factor)
        # Volatility/Variance based risks scale with sqrt(T), Drawdowns scale linearly, Variance scales with T
        if risk_col in ["StdDev", "sd", "CVaR", "ES", "VaR", "EVaR", "RLVaR", "MAD", "semi_MAD"]:
            risk_scale = np.sqrt(annualize_factor)
        elif risk_col in ["var", "Variance"]:
            risk_scale = float(annualize_factor)
        else: # Drawdowns or unclassified
            risk_scale = 1.0 # Standard practice: Drawdowns are not annualized, or scaled linearly? Usually unscaled or linear. We keep it 1.0 for DD to be safe, or user can pre-scale. Let's use 1.0 for DD.
            if "DaR" in risk_col or "drawdown" in risk_col.lower():
                risk_scale = 1.0
            else:
                risk_scale = np.sqrt(annualize_factor) # default to sqrt
        
        # Update axis titles to reflect annualization
        x_title_ext = f" (Annualized x{annualize_factor})" if risk_scale != 1.0 else ""
        y_title_ext = f" (Annualized x{annualize_factor})"
    else:
        x_title_ext = ""
        y_title_ext = ""

    # Calculate ratios for color mapping
    risk_vals = frontier_data[risk_col].values * risk_scale
    ret_vals = frontier_data[return_col].values * ret_scale
    ratios = np.zeros_like(risk_vals)
    for i in range(len(risk_vals)):
        ratios[i] = ret_vals[i] / risk_vals[i] if risk_vals[i] > 1e-8 else 0.0

    # Plot Efficient Frontier
    fig.add_trace(
        go.Scatter(
            x=risk_vals,
            y=ret_vals,
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(shape="spline", smoothing=1.3, width=2, color="gray"),
            marker=dict(
                size=8, 
                color=ratios, 
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Return/Risk Ratio"),
                line=dict(width=0.5, color="DarkSlateGrey")
            ),
            hovertemplate=f"Return: %{{y:.4%}}<br>Risk ({risk_col}): %{{x:.4%}}<br>Ratio: %{{marker.color:.4f}}<extra></extra>"
        )
    )

    # Plot Individual Assets if moments are provided
    if moments is not None and "mu" in moments and "sigma" in moments:
        mu = moments["mu"].flatten() * ret_scale
        sigma = moments["sigma"]
        
        # Determine std deviation (annualized or period depending on inputs, usually matches frontier)
        # Note: if risk_col is not StdDev, plotting assets on the same X-axis might be mathematically inconsistent 
        # unless we compute the specific risk measure for each asset. For simplicity, we approximate with StdDev * scale
        asset_sd = np.sqrt(np.diag(sigma)) * (np.sqrt(annualize_factor) if annualize_factor else 1.0)
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
                name="Assets (StdDev)",
                text=asset_names,
                hovertemplate="<b>%{text}</b><br>Return: %{y:.4%}<br>Risk: %{x:.4%}<extra></extra>",
                marker=dict(size=10, symbol="diamond", color="#ff7f0e", line=dict(width=1, color="black")),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"Risk ({risk_col}){x_title_ext}",
        yaxis_title=f"Return ({return_col}){y_title_ext}",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    )

    return fig


def plot_risk_decomposition(
    ccr: pd.Series, 
    title: str = "Component Contribution to Risk",
    percentage: bool = True,
    erc_line: bool = False,
    custom_budget: pd.Series | dict | None = None,
) -> go.Figure:
    """
    Plot a horizontal bar chart showing the Component Contribution to Risk of each asset.
    Supports converting to Percentage Contribution to Risk (PCR) and overlaying reference lines.
    """
    fig = go.Figure()

    total_risk = ccr.sum()
    if percentage and total_risk > 0:
        data_to_plot = ccr / total_risk
        x_title = "Percentage Contribution to Risk (PCR)"
        hover_template = "%{y}: %{x:.2%}<extra></extra>"
    else:
        data_to_plot = ccr
        x_title = "Absolute Contribution to Risk (CCR)"
        hover_template = "%{y}: %{x:.4f}<extra></extra>"

    # Sort for better visualization
    data_sorted = data_to_plot.sort_values(ascending=True)

    fig.add_trace(
        go.Bar(
            x=data_sorted.values,
            y=data_sorted.index,
            orientation="h",
            hovertemplate=hover_template,
            marker=dict(
                color=data_sorted.values,
                colorscale="Viridis",
                showscale=False,
            ),
            name="Risk Contribution"
        )
    )

    # Add Equal Risk Contribution (ERC) vertical line
    if percentage and erc_line:
        n_assets = len(ccr)
        if n_assets > 0:
            target = 1.0 / n_assets
            fig.add_vline(
                x=target,
                line_dash="dash",
                line_color="red",
                annotation_text=f"ERC ({target:.1%})",
                annotation_position="top right",
                opacity=0.8
            )
            
    # Add custom budget markers if provided
    if custom_budget is not None and percentage:
        if isinstance(custom_budget, dict):
            budget_series = pd.Series(custom_budget)
        else:
            budget_series = custom_budget
            
        # Reindex to match sorted data order
        budget_sorted = budget_series.reindex(data_sorted.index).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=budget_sorted.values,
                y=budget_sorted.index,
                mode="markers",
                name="Target Budget",
                marker=dict(symbol="line-ns-open", size=12, color="red", line=dict(width=2))
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Asset",
        xaxis_tickformat=".1%" if percentage else ".4f",
        template="plotly_white",
        showlegend=False if custom_budget is None else True
    )

    return fig


def plot_factor_risk_decomposition(
    factor_decomp_result: dict[str, Any],
    factor_names: list[str] | None = None,
    percentage: bool = True,
    title: str = "Factor Risk Decomposition (Attribution)",
) -> go.Figure:
    """
    Plot the systematic (factor) and idiosyncratic risk contributions of a portfolio.
    Requires the output dictionary from `risk.factor_risk_decomposition`.
    """
    fig = go.Figure()

    total_risk = factor_decomp_result.get("total", 0.0)
    
    if percentage and total_risk > 0:
        factor_contrib = factor_decomp_result["pcr_factor"]
        resid_contrib = factor_decomp_result["pcr_residual"]
        x_title = "Percentage Contribution to Risk (PCR)"
        hover_template = "%{y}: %{x:.2%}<extra></extra>"
    else:
        factor_contrib = factor_decomp_result["ccr_factor"]
        resid_contrib = factor_decomp_result["ccr_residual"]
        x_title = "Absolute Contribution to Risk (CCR)"
        hover_template = "%{y}: %{x:.4f}<extra></extra>"

    num_factors = len(factor_contrib)
    if factor_names is None:
        factor_names = [f"Factor {i+1}" for i in range(num_factors)]

    # 1. Systematic Factors Trace
    # Use blue for factors
    fig.add_trace(
        go.Bar(
            x=factor_contrib,
            y=factor_names,
            orientation="h",
            name="Systematic Factors",
            marker=dict(color="#1f77b4"),
            hovertemplate=hover_template
        )
    )

    # 2. Idiosyncratic Risk Trace
    # Use orange/grey to separate specific risk
    fig.add_trace(
        go.Bar(
            x=[np.sum(resid_contrib)],  # Residuals are typically a vector of asset-specific risks, sum them for total idiosyncratic
            y=["Idiosyncratic (Specific)"],
            orientation="h",
            name="Idiosyncratic Risk",
            marker=dict(color="#ff7f0e"),
            hovertemplate=hover_template
        )
    )

    # 3. Layout & Aesthetics
    fig.update_layout(
        title=title,
        barmode="relative",  # Allows positive and negative bars to exist cleanly
        xaxis_title=x_title,
        yaxis=dict(title="Risk Source", autorange="reversed"),  # Reverse so Factor 1 is at top
        xaxis_tickformat=".1%" if percentage else ".4f",
        template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    )

    # Add a zero line to clearly show negative risk contributions (hedges)
    fig.add_vline(x=0, line_width=1, line_color="black")

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


def plot_network_allocation(
    weights: pd.Series,
    R: pd.DataFrame,
    title: str = "Network Topology Allocation (Minimum Spanning Tree)",
) -> go.Figure:
    """
    Plots an asset correlation network based on the Minimum Spanning Tree (MST).
    Nodes are sized by their portfolio allocation weights.
    Requires `networkx` to be installed.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("The 'networkx' package is required to plot network allocations. Please install it using 'pip install networkx'.")
        
    # 1. Filter: Build distance matrix and extract MST
    corr = R.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    
    G = nx.Graph()
    assets = list(weights.index)
    N = len(assets)
    
    # Build fully connected graph
    for i in range(N):
        G.add_node(i, label=assets[i], weight=weights.iloc[i])
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=dist[i, j], corr=corr[i, j])
            
    # Extract Minimum Spanning Tree (based on minimal distance -> maximum correlation)
    mst = nx.minimum_spanning_tree(G, weight="weight")
    
    # 2. Calculate Spring Layout (highly correlated assets pull each other closer)
    pos = nx.spring_layout(mst, weight="weight", seed=42)
    
    # 3. Construct Plotly Render Data
    edge_x, edge_y = [], []
    for edge in mst.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    node_x = [pos[i][0] for i in range(N)]
    node_y = [pos[i][1] for i in range(N)]
    
    # Map weights to sizes and colors
    # Ensure minimum size for visibility, scale up proportionally
    w_vals = weights.values
    node_text = [f"{assets[i]}<br>Weight: {w_vals[i]:.2%}" for i in range(N)]
    # Scaling factor for node sizes
    node_size = [max(18, w_vals[i] * 120) for i in range(N)] 
    
    fig = go.Figure()
    
    # Draw Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=1.5, color='#888'), 
        hoverinfo='none', 
        mode='lines',
        name='MST Connections'
    ))
    
    # Draw Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        mode='markers+text', 
        text=assets, 
        textfont=dict(
            family="sans serif",
            size=11,
            color="black"
        ),
        hovertext=node_text, 
        hoverinfo='text',
        textposition="top center",
        name='Assets',
        marker=dict(
            showscale=True, 
            colorscale='YlGnBu', 
            color=w_vals, 
            size=node_size,
            line=dict(width=1.5, color='DarkSlateGrey'), 
            colorbar=dict(title="Allocation Weight", tickformat=".1%")
        )
    ))
    
    fig.update_layout(
        title=title, 
        showlegend=False, 
        template="plotly_white", 
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def plot_return_histogram(
    returns: pd.Series,
    alpha: float = 0.05,
    title: str = "Portfolio Returns Distribution",
) -> go.Figure:
    """
    Plot a histogram of portfolio returns overlaid with a Kernel Density Estimate (KDE).
    Adds vertical lines indicating Value at Risk (VaR) and Conditional Value at Risk (CVaR/ES) 
    at the specified significance level alpha.
    """
    import plotly.figure_factory as ff
    
    # Drop NaNs
    rets = returns.dropna()
    if len(rets) == 0:
        return go.Figure()

    # Calculate VaR (historical simulation)
    # Note: VaR/CVaR in risk.py are defined on losses (positive values), 
    # but for the histogram X-axis, we want the actual negative return threshold.
    var_threshold = np.percentile(rets.values, alpha * 100)
    
    # Calculate CVaR/ES (Expected Shortfall)
    # We can use the risk.py function which expects weights=1 and R=rets.values.reshape(-1, 1)
    # Alternatively, direct calculation:
    tail_losses = rets[rets <= var_threshold]
    cvar_threshold = tail_losses.mean() if len(tail_losses) > 0 else var_threshold

    # Create distribution plot with histogram and KDE
    fig = ff.create_distplot(
        [rets.values],
        group_labels=["Returns"],
        bin_size=(rets.max() - rets.min()) / 50,
        show_rug=False,
        colors=["#1f77b4"]
    )
    
    # Find max y to draw lines
    max_y = max([max(trace.y) for trace in fig.data if hasattr(trace, 'y') and trace.y is not None])

    # Add VaR line
    fig.add_trace(
        go.Scatter(
            x=[var_threshold, var_threshold],
            y=[0, max_y],
            mode="lines",
            name=f"VaR ({(1-alpha):.0%})",
            line=dict(color="orange", width=2, dash="dash"),
            hoverinfo="name+x"
        )
    )

    # Add CVaR line
    fig.add_trace(
        go.Scatter(
            x=[cvar_threshold, cvar_threshold],
            y=[0, max_y],
            mode="lines",
            name=f"CVaR/ES ({(1-alpha):.0%})",
            line=dict(color="red", width=2, dash="dot"),
            hoverinfo="name+x"
        )
    )
    
    # Shade the tail region
    # Find KDE trace (it's a scatter trace added by create_distplot)
    kde_trace = next((trace for trace in fig.data if getattr(trace, 'type', '') == 'scatter' and getattr(trace, 'mode', '') == "lines" and getattr(trace, 'name', '') == "Returns"), None)
    if kde_trace is not None:
        x_kde = np.array(kde_trace.x)
        y_kde = np.array(kde_trace.y)
        
        tail_mask = x_kde <= var_threshold
        x_tail = x_kde[tail_mask]
        y_tail = y_kde[tail_mask]
        
        if len(x_tail) > 0:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_tail, x_tail[::-1]]),
                    y=np.concatenate([y_tail, np.zeros_like(y_tail)]),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color="rgba(255,0,0,0)"),
                    name="Tail Risk Area",
                    hoverinfo="skip",
                    showlegend=False
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Returns",
        yaxis_title="Density",
        xaxis_tickformat=".2%",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    )

    return fig


def plot_performance(
    returns: pd.Series, 
    title: str = "Portfolio Performance",
) -> go.Figure:
    """
    Plot cumulative returns curve and compounded underwater drawdown plot.
    """
    # Calculate cumulative return
    cum_returns = (1 + returns).cumprod()

    # Calculate compounded drawdown
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns / rolling_max) - 1.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Returns", "Compounded Drawdown"),
    )

    # Cumulative Returns trace
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode="lines",
            name="Cumulative Return",
            line=dict(width=2, color="#1f77b4"),  # Muted professional blue
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
            name="Compounded Drawdown",
            line=dict(width=1, color="#ff7f0e"),  # Warm amber/orange
            fillcolor="rgba(255, 127, 14, 0.25)",
        ),
        row=2,
        col=1,
    )
    
    fig.update_layout(
        title=title, height=700, hovermode="x unified", template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=10)
        )
    )

    fig.update_yaxes(title_text="Cumulative Wealth", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def plot_underwater_drawdown(
    returns: pd.Series, 
    title: str = "Underwater Drawdown Analysis",
    alpha: float = 0.05
) -> go.Figure:
    """
    Plot a comprehensive two-pane underwater drawdown chart.
    Top Pane: Historical Compounded Cumulative Returns.
    Bottom Pane: Historical Uncompounded (Absolute) Drawdown overlaid with key risk metrics 
    (Max Drawdown, CDaR, Average Drawdown, UCI).
    """
    from .risk import CDaR, UCI, average_drawdown, max_drawdown
    
    # 1. Calculate Compounded Cumulative Returns (Wealth)
    cum_returns = (1 + returns).cumprod()
    
    # 2. Calculate Uncompounded Drawdown
    uncomp_cum = returns.cumsum()
    rolling_max_uncomp = np.maximum(0, uncomp_cum.cummax())
    uncomp_drawdown = uncomp_cum - rolling_max_uncomp

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Historical Compounded Cumulative Returns", "Historical Uncompounded Drawdown"),
    )

    # Top Pane: Cumulative Returns trace
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode="lines",
            name="Cumulative Return",
            line=dict(width=1.5, color="#1f77b4"),  # Muted blue
        ),
        row=1,
        col=1,
    )

    # Bottom Pane: Uncompounded Drawdown trace
    fig.add_trace(
        go.Scatter(
            x=uncomp_drawdown.index,
            y=uncomp_drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Uncompounded DD",
            line=dict(width=1, color="#d62728"),  # Muted professional red
            fillcolor="rgba(214, 39, 40, 0.25)",
        ),
        row=2,
        col=1,
    )
    
    # Bottom Pane: Overlay Metrics
    w_dummy = np.array([1.0])
    r_mat = returns.values.reshape(-1, 1)
    
    mdd = -max_drawdown(w_dummy, r_mat)
    avg_dd = -average_drawdown(w_dummy, r_mat)
    cdar = -CDaR(w_dummy, r_mat, p=1-alpha)
    uci = -UCI(w_dummy, r_mat)
    
    metrics = [
        ("Average Drawdown", avg_dd, "#2ca02c", "solid"), # Green
        ("Ulcer Index", uci, "#9467bd", "dash"), # Purple
        (f"CDaR ({(1-alpha):.0%})", cdar, "#e377c2", "solid"), # Pink
        ("Maximum Drawdown", mdd, "#7f7f7f", "dashdot"), # Gray
    ]
    
    metrics.sort(key=lambda x: x[1], reverse=True)
    
    for name, val, color, dash in metrics:
        fig.add_trace(
            go.Scatter(
                x=[uncomp_drawdown.index[0], uncomp_drawdown.index[-1]],
                y=[val, val],
                mode="lines",
                name=f"{name}: {val:.2%}",
                line=dict(color=color, width=2, dash=dash),
                hoverinfo="name"
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        title=title, height=600, hovermode="x unified", template="plotly_white",
        legend=dict(
            yanchor="bottom",
            y=0.01,  # Place legend at the absolute bottom right
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=10)
        )
    )

    fig.update_yaxes(title_text="Cumulative Wealth", tickformat=".1%", row=1, col=1)
    # The lower pane values can be very large negative numbers (e.g. -2.0)
    # Plotly's .1% format will display -2.0 as -200.0%. This matches Riskfolio.
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig
