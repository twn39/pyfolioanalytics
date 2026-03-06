import numpy as np
from .portfolio import Portfolio


def rp_simplex(portfolio: Portfolio, permutations: int = 1000, **kwargs) -> np.ndarray:
    n = len(portfolio.assets)
    constraints = portfolio.get_constraints()
    min_w = constraints["min"].values
    max_w = constraints["max"].values
    portfolios = []
    count = 0
    max_iter = permutations * 50
    while len(portfolios) < permutations and count < max_iter:
        count += 1
        w = np.random.dirichlet(np.ones(n), size=1).flatten()
        w = w * constraints["max_sum"]  # Scale to sum
        if np.all(w >= min_w - 1e-9) and np.all(w <= max_w + 1e-9):
            portfolios.append(w)
    return np.array(portfolios)


def rp_transform(
    portfolio: Portfolio, permutations: int = 1000, **kwargs
) -> np.ndarray:
    """
    Generate random portfolios using the transform method (random walk).
    Ensures weights satisfy box and weight_sum constraints.
    """
    n = len(portfolio.assets)
    constraints = portfolio.get_constraints()
    min_w = constraints["min"].values
    max_w = constraints["max"].values
    target_sum = constraints["min_sum"]  # Assume min_sum == max_sum for simplicity

    # 1. Start with a feasible portfolio (Equal Weight if possible, else Simplex)
    w_start = np.full(n, target_sum / n)
    if not (np.all(w_start >= min_w) and np.all(w_start <= max_w)):
        # Try to find one via simplex
        seeds = rp_simplex(portfolio, permutations=1)
        if len(seeds) > 0:
            w_start = seeds[0]
        else:
            # Fallback to clipping (not ideal but provides a starting point)
            w_start = np.clip(w_start, min_w, max_w)
            w_start = w_start / w_start.sum() * target_sum

    portfolios = [w_start]
    w_curr = w_start.copy()

    # 2. Random Walk via Pairwise Transformation
    while len(portfolios) < permutations:
        # Pick two random assets
        idx = np.random.choice(n, 2, replace=False)
        i, j = idx[0], idx[1]

        # Determine available slack for movement
        # w_i can increase by up to (max_w[i] - w_i)
        # w_j can decrease by up to (w_j - min_w[j])
        # Delta is the amount to move from j to i
        max_delta = min(max_w[i] - w_curr[i], w_curr[j] - min_w[j])
        min_delta = -min(w_curr[i] - min_w[i], max_w[j] - w_curr[j])

        if max_delta > min_delta:
            delta = np.random.uniform(min_delta, max_delta)
            w_curr[i] += delta
            w_curr[j] -= delta

            # Record if significant move or just periodically
            if abs(delta) > 1e-6:
                portfolios.append(w_curr.copy())
        else:
            # Stalled, try to re-seed or just continue
            pass

    return np.array(portfolios)[:permutations]


def random_portfolios(
    portfolio: Portfolio, permutations: int = 1000, method: str = "simplex", **kwargs
) -> np.ndarray:
    if method == "simplex":
        return rp_simplex(portfolio, permutations, **kwargs)
    elif method == "transform":
        return rp_transform(portfolio, permutations, **kwargs)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")
