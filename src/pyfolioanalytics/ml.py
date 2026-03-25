import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .codependence import get_codependence_matrix, get_distance_matrix
from .dbht import DBHTs


def get_ivp(cov: np.ndarray) -> np.ndarray:
    """
    Inverse-Variance Portfolio (IVP) weights.
    """
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov: np.ndarray, cluster_indices: list[int]) -> float:
    """
    Variance of a cluster based on IVP weights within the cluster.
    """
    cov_slice = cov[np.ix_(cluster_indices, cluster_indices)]
    w = get_ivp(cov_slice)
    return float(w.T @ cov_slice @ w)


def get_recursive_bisection(
    cov: np.ndarray, sort_indices: list[int], method: str = "HRP"
) -> pd.Series:
    """
    Recursive bisection to allocate weights.
    """
    w = pd.Series(1.0, index=range(len(sort_indices)))
    items = [sort_indices]

    while len(items) > 0:
        items = [i for i in items if len(i) > 1]
        if not items:
            break
        curr_items = items.pop(0)
        mid = len(curr_items) // 2
        left_items = curr_items[:mid]
        right_items = curr_items[mid:]

        v_left = get_cluster_var(cov, left_items)
        v_right = get_cluster_var(cov, right_items)

        if method == "HRP":
            alpha = 1 - v_left / (v_left + v_right)
        elif method == "HERC":
            std_l = np.sqrt(v_left)
            std_r = np.sqrt(v_right)
            alpha = 1 - std_l / (std_l + std_r)
        else:
            alpha = 0.5

        w.loc[left_items] *= alpha
        w.loc[right_items] *= 1 - alpha

        items.append(left_items)
        items.append(right_items)
    return w


def hrp_optimization(R: pd.DataFrame, **kwargs) -> pd.Series:
    asset_names = R.columns.tolist()
    cov = R.cov().values

    # 1. Compute Codependence & Distance Matrix
    codep_method = kwargs.get("codependence", "pearson")
    dist_method = kwargs.get("distance", "standard")

    if codep_method == "custom" and "custom_matrix" in kwargs:
        corr = kwargs["custom_matrix"]
    else:
        codep_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["codependence", "distance", "method"]
        }
        corr = get_codependence_matrix(R, method=codep_method, **codep_kwargs)

    if dist_method == "custom" and "custom_distance" in kwargs:
        dist = kwargs["custom_distance"]
    else:
        dist_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["codependence", "distance", "method"]
        }
        dist = get_distance_matrix(corr, method=dist_method, **dist_kwargs)

    clustering = kwargs.get("clustering", "linkage")

    # 2. Hierarchical Clustering
    if clustering == "DBHT":
        S = corr + 1.0
        res = DBHTs(dist, S)
        Z = res[5]
    else:
        method = kwargs.get("linkage_method", "single")
        Z = linkage(squareform(dist), method=method)

    sort_indices = leaves_list(Z).tolist()

    weights = get_recursive_bisection(cov, sort_indices, method="HRP")
    final_w = pd.Series(0.0, index=asset_names)
    for i, w_val in weights.items():
        final_w.iloc[i] = w_val
    return final_w


def herc_optimization(R: pd.DataFrame, **kwargs) -> pd.Series:
    asset_names = R.columns.tolist()
    cov = R.cov().values

    # 1. Compute Codependence & Distance Matrix
    codep_method = kwargs.get("codependence", "pearson")
    dist_method = kwargs.get("distance", "standard")

    if codep_method == "custom" and "custom_matrix" in kwargs:
        corr = kwargs["custom_matrix"]
    else:
        codep_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["codependence", "distance", "method"]
        }
        corr = get_codependence_matrix(R, method=codep_method, **codep_kwargs)

    if dist_method == "custom" and "custom_distance" in kwargs:
        dist = kwargs["custom_distance"]
    else:
        dist_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["codependence", "distance", "method"]
        }
        dist = get_distance_matrix(corr, method=dist_method, **dist_kwargs)

    clustering = kwargs.get("clustering", "linkage")

    # 2. Hierarchical Clustering
    if clustering == "DBHT":
        S = corr + 1.0
        res = DBHTs(dist, S)
        Z = res[5]
    else:
        method = kwargs.get("linkage_method", "ward")
        Z = linkage(squareform(dist), method=method)

    sort_indices = leaves_list(Z).tolist()

    weights = get_recursive_bisection(cov, sort_indices, method="HERC")
    final_w = pd.Series(0.0, index=asset_names)
    for i, w_val in weights.items():
        final_w.iloc[i] = w_val
    return final_w


def nco_optimization(R: pd.DataFrame, **kwargs) -> pd.Series:
    from .solvers import solve_mvo

    asset_names = R.columns.tolist()

    # 1. Compute Codependence & Distance Matrix
    codep_method = kwargs.get("codependence", "pearson")
    dist_method = kwargs.get("distance", "standard")

    if codep_method == "custom" and "custom_matrix" in kwargs:
        corr = kwargs["custom_matrix"]
    else:
        codep_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["codependence", "distance", "method"]
        }
        corr = get_codependence_matrix(R, method=codep_method, **codep_kwargs)

    if dist_method == "custom" and "custom_distance" in kwargs:
        dist = kwargs["custom_distance"]
    else:
        dist_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["codependence", "distance", "method"]
        }
        dist = get_distance_matrix(corr, method=dist_method, **dist_kwargs)

    clustering = kwargs.get("clustering", "linkage")

    # 2. Hierarchical Clustering
    if clustering == "DBHT":
        S = corr + 1.0
        res = DBHTs(dist, S)
        clusters = res[0].flatten() + 1
    else:
        method = kwargs.get("linkage_method", "ward")
        link = linkage(squareform(dist), method=method)
        max_clusters = kwargs.get("max_clusters", 3)
        clusters = fcluster(link, max_clusters, criterion="maxclust")

    cluster_ids = np.unique(clusters)
    w_intra = pd.Series(0.0, index=asset_names)

    # 2. Intra-cluster optimization
    for cid in cluster_ids:
        c_assets = [asset_names[i] for i, v in enumerate(clusters) if v == cid]
        R_c = R[c_assets]
        mu_c = R_c.mean().values.reshape(-1, 1)
        sigma_c = R_c.cov().values

        # Simple Min Variance for intra-cluster
        res = solve_mvo(
            {"mu": mu_c, "sigma": sigma_c},
            {
                "min_sum": 1.0,
                "max_sum": 1.0,
                "min": pd.Series(0.0, index=c_assets),
                "max": pd.Series(1.0, index=c_assets),
            },
            [],
        )
        if res["weights"] is not None:
            w_intra[c_assets] = res["weights"]

    # 3. Inter-cluster optimization
    R_inter = pd.DataFrame(index=R.index)
    for cid in cluster_ids:
        c_assets = [asset_names[i] for i, v in enumerate(clusters) if v == cid]
        R_inter[f"cluster_{cid}"] = R[c_assets] @ w_intra[c_assets].values

    mu_inter = R_inter.mean().values.reshape(-1, 1)
    sigma_inter = R_inter.cov().values

    res_inter = solve_mvo(
        {"mu": mu_inter, "sigma": sigma_inter},
        {
            "min_sum": 1.0,
            "max_sum": 1.0,
            "min": pd.Series(0.0, index=R_inter.columns),
            "max": pd.Series(1.0, index=R_inter.columns),
        },
        [],
    )

    # 4. Final weights
    w_inter = res_inter["weights"]
    final_w = pd.Series(0.0, index=asset_names)
    for i, cid in enumerate(cluster_ids):
        c_assets = [asset_names[j] for j, v in enumerate(clusters) if v == cid]
        final_w[c_assets] = w_intra[c_assets] * w_inter[i]

    return final_w
