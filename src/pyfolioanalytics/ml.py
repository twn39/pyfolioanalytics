import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from typing import List, Dict, Any, Optional

def get_ivp(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov)
    diag = np.where(diag < 1e-12, 1e-12, diag)
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov: np.ndarray, cluster_items: List[int]) -> float:
    cov_slice = cov[cluster_items][:, cluster_items]
    w_ivp = get_ivp(cov_slice)
    return np.dot(w_ivp, np.dot(cov_slice, w_ivp))

def get_quasi_diag(link: np.ndarray) -> List[int]:
    return leaves_list(link).tolist()

def get_recursive_bisection(cov: np.ndarray, sort_items: List[int], method: str = "HRP") -> pd.Series:
    w = pd.Series(1.0, index=sort_items)
    items = [sort_items]
    while len(items) > 0:
        items = [i for i in items if len(i) > 1]
        if not items: break
        curr_items = items.pop(0)
        mid = len(curr_items) // 2
        left_items = curr_items[:mid]
        right_items = curr_items[mid:]
        v_left = get_cluster_var(cov, left_items)
        v_right = get_cluster_var(cov, right_items)
        if method == "HRP":
            alpha = 1 - v_left / (v_left + v_right)
        elif method == "HERC":
            std_l = np.sqrt(v_left); std_r = np.sqrt(v_right)
            alpha = 1 - std_l / (std_l + std_r)
        w.loc[left_items] *= alpha
        w.loc[right_items] *= (1 - alpha)
        items.append(left_items)
        items.append(right_items)
    return w

def hrp_optimization(R: pd.DataFrame, **kwargs) -> pd.Series:
    asset_names = R.columns.tolist()
    corr = R.corr().values; cov = R.cov().values
    dist = np.sqrt(0.5 * (1 - corr)); np.fill_diagonal(dist, 0)
    method = kwargs.get("linkage_method", "single")
    link = linkage(squareform(dist), method=method)
    sort_indices = get_quasi_diag(link)
    weights = get_recursive_bisection(cov, sort_indices, method="HRP")
    final_w = pd.Series(0.0, index=asset_names)
    for i, w_val in weights.items(): final_w[asset_names[i]] = w_val
    return final_w

def herc_optimization(R: pd.DataFrame, **kwargs) -> pd.Series:
    asset_names = R.columns.tolist()
    corr = R.corr().values; cov = R.cov().values
    dist = np.sqrt(0.5 * (1 - corr)); np.fill_diagonal(dist, 0)
    method = kwargs.get("linkage_method", "ward")
    link = linkage(squareform(dist), method=method)
    sort_indices = get_quasi_diag(link)
    weights = get_recursive_bisection(cov, sort_indices, method="HERC")
    final_w = pd.Series(0.0, index=asset_names)
    for i, w_val in weights.items(): final_w[asset_names[i]] = w_val
    return final_w

def nco_optimization(R: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Nested Clustered Optimization (NCO).
    """
    from .portfolio import Portfolio
    from .optimize import optimize_portfolio
    
    asset_names = R.columns.tolist()
    n = len(asset_names)
    corr = R.corr().values
    cov = R.cov().values
    
    # 1. Clustering
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist), method=kwargs.get("linkage_method", "ward"))
    
    # Determine clusters (e.g., using max clusters or distance threshold)
    max_k = kwargs.get("max_clusters", min(n // 2 + 1, 10))
    clusters = fcluster(link, max_k, criterion='maxclust')
    
    cluster_weights = {} # Intra-cluster
    cluster_returns = {}
    
    # 2. Intra-cluster Optimization (MVO Min Variance)
    for k in np.unique(clusters):
        idx = np.where(clusters == k)[0]
        c_assets = [asset_names[i] for i in idx]
        
        # Simple internal portfolio for min variance
        p_sub = Portfolio(assets=c_assets)
        p_sub.add_constraint(type="full_investment")
        p_sub.add_constraint(type="long_only")
        p_sub.add_objective(type="risk", name="StdDev")
        
        res = optimize_portfolio(R[c_assets], p_sub)
        w_sub = res["weights"]
        cluster_weights[k] = w_sub
        cluster_returns[k] = R[c_assets] @ w_sub.values
        
    # 3. Inter-cluster Optimization
    R_meta = pd.DataFrame(cluster_returns)
    p_root = Portfolio(assets=list(R_meta.columns))
    p_root.add_constraint(type="full_investment")
    p_root.add_constraint(type="long_only")
    p_root.add_objective(type="risk", name="StdDev")
    
    res_root = optimize_portfolio(R_meta, p_root)
    w_inter = res_root["weights"]
    
    # 4. Final Weights
    final_w = pd.Series(0.0, index=asset_names)
    for k, w_k in w_inter.items():
        for asset, w_in in cluster_weights[k].items():
            final_w[asset] = w_k * w_in
            
    return final_w
