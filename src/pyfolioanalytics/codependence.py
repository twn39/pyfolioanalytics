import numpy as np
import pandas as pd


def get_codependence_matrix(
    R: pd.DataFrame, method: str = "pearson", **kwargs
) -> np.ndarray:
    """
    Calculate codependence (similarity) matrix for clustering.
    Returns a matrix bounded in [-1, 1] or [0, 1] representing similarity.
    """
    if method == "pearson":
        return R.corr(method="pearson").values
    elif method == "spearman":
        return R.corr(method="spearman").values
    elif method == "abs_pearson":
        return np.abs(R.corr(method="pearson").values)
    elif method == "distance":
        # Distance correlation
        return _distance_correlation_matrix(R.values)
    elif method == "mutual_info":
        return _mutual_info_matrix(R.values, **kwargs)
    elif method == "tail":
        return _tail_dependence_matrix(R.values, **kwargs)
    elif method == "custom" and "custom_matrix" in kwargs:
        return kwargs["custom_matrix"]
    else:
        raise ValueError(f"Unknown codependence method: {method}")


def get_distance_matrix(
    codependence: np.ndarray, method: str = "standard", **kwargs
) -> np.ndarray:
    """
    Convert a codependence (similarity) matrix into a distance matrix.
    """
    # Clip to avoid numerical precision issues
    rho = np.clip(codependence, -1.0, 1.0)

    if method == "standard":
        # sqrt(0.5 * (1 - rho))
        dist = np.sqrt(0.5 * (1.0 - rho))
    elif method == "absolute":
        # sqrt(1 - |rho|)
        dist = np.sqrt(1.0 - np.abs(rho))
    elif method == "variation_of_information":
        # Variation of Information requires Mutual Information as input.
        # Here we assume `codependence` is the normalized mutual information [0, 1]
        dist = 1.0 - rho
    elif method == "custom" and "custom_distance" in kwargs:
        dist = kwargs["custom_distance"]
    else:
        raise ValueError(f"Unknown distance metric: {method}")

    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry
    dist = (dist + dist.T) / 2.0
    # Clip tiny negative values due to float precision
    dist = np.clip(dist, 0.0, None)
    return dist


def _distance_correlation_matrix(X: np.ndarray) -> np.ndarray:
    n_samples, n_features = X.shape
    dcor = np.ones((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = _dcor_1d(X[:, i], X[:, j])
            dcor[i, j] = val
            dcor[j, i] = val
    return dcor


def _dcor_1d(x: np.ndarray, y: np.ndarray) -> float:
    # A fast, simplified calculation for distance correlation of 1D arrays
    n = len(x)
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])

    A_mean_row = A.mean(axis=1, keepdims=True)
    A_mean_col = A.mean(axis=0, keepdims=True)
    A_mean_all = A.mean()
    a = A - A_mean_row - A_mean_col + A_mean_all

    B_mean_row = B.mean(axis=1, keepdims=True)
    B_mean_col = B.mean(axis=0, keepdims=True)
    B_mean_all = B.mean()
    b = B - B_mean_row - B_mean_col + B_mean_all

    dcov2_xy = (a * b).sum() / (n * n)
    dcov2_xx = (a * a).sum() / (n * n)
    dcov2_yy = (b * b).sum() / (n * n)

    denom = np.sqrt(dcov2_xx * dcov2_yy)
    if denom < 1e-12:
        return 0.0
    return np.sqrt(max(0, dcov2_xy / denom))


def _mutual_info_matrix(X: np.ndarray, bins: int = 20) -> np.ndarray:
    """
    Calculate Normalized Mutual Information matrix.
    Using histogram-based entropy.
    """
    n_samples, n_features = X.shape
    mi_mat = np.ones((n_features, n_features))

    # Pre-bin all columns
    binned_X = np.zeros_like(X, dtype=int)
    entropies = np.zeros(n_features)

    for i in range(n_features):
        hist, edges = np.histogram(X[:, i], bins=bins)
        p = hist / n_samples
        p = p[p > 0]
        entropies[i] = -np.sum(p * np.log2(p))
        binned_X[:, i] = np.digitize(X[:, i], edges[:-1]) - 1

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Joint histogram
            hist2d, _, _ = np.histogram2d(binned_X[:, i], binned_X[:, j], bins=bins)
            p_xy = hist2d / n_samples
            p_xy = p_xy[p_xy > 0]
            h_xy = -np.sum(p_xy * np.log2(p_xy))

            # Mutual Information: I(X; Y) = H(X) + H(Y) - H(X,Y)
            I_xy = entropies[i] + entropies[j] - h_xy

            # Variation of Information: VI(X, Y) = H(X, Y) - I(X, Y) = H(X) + H(Y) - 2I(X, Y)
            # Normalized Mutual Info: I(X; Y) / sqrt(H(X)*H(Y))
            denom = np.sqrt(entropies[i] * entropies[j])
            if denom < 1e-12:
                nmi = 0.0
            else:
                nmi = max(0.0, min(1.0, I_xy / denom))

            mi_mat[i, j] = nmi
            mi_mat[j, i] = nmi

    return mi_mat


def _tail_dependence_matrix(X: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Lower tail dependence matrix"""
    n_samples, n_features = X.shape
    td_mat = np.ones((n_features, n_features))

    # Find quantile thresholds
    thresholds = np.quantile(X, q, axis=0)

    for i in range(n_features):
        is_tail_i = X[:, i] <= thresholds[i]
        p_i = np.mean(is_tail_i)
        for j in range(i + 1, n_features):
            is_tail_j = X[:, j] <= thresholds[j]
            p_ij = np.mean(is_tail_i & is_tail_j)

            # td = P(X <= q_x | Y <= q_y) + P(Y <= q_y | X <= q_x) / 2
            p_j = np.mean(is_tail_j)
            if p_i == 0 or p_j == 0:
                td = 0.0
            else:
                td = 0.5 * (p_ij / p_i + p_ij / p_j)

            td_mat[i, j] = td
            td_mat[j, i] = td
    return td_mat
