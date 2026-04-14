import itertools

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


def _project_and_normalize(w: np.ndarray, min_w: np.ndarray, max_w: np.ndarray, target_sum: float, max_pos: int | None = None) -> np.ndarray:
    """Helper to project weights into box constraints and normalize to target sum."""
    n = len(w)
    
    # Enforce max_pos before any projection
    if max_pos is not None and max_pos < n:
        threshold_idx = np.argsort(w)[-max_pos:]
        mask = np.zeros(n, dtype=bool)
        mask[threshold_idx] = True
        w[~mask] = 0.0
        
    w_proj = np.clip(w, min_w, max_w)
    
    # If max_pos is applied, ensure we don't accidentally re-allocate weights to masked assets
    active_mask = np.ones(n, dtype=bool)
    if max_pos is not None and max_pos < n:
        # Only assets that survived the max_pos mask AND aren't forced to 0 by min_w
        active_mask = mask
    
    for _ in range(10):  # Iterative projection
        current_sum = w_proj.sum()
        if abs(current_sum - target_sum) < 1e-6:
            break
        
        # Calculate how much needs to be added/removed
        diff = target_sum - current_sum
        
        # Find which weights can still be modified
        if diff > 0:
            modifiable = (w_proj < max_w - 1e-8) & active_mask
        else:
            modifiable = (w_proj > min_w + 1e-8) & active_mask
            
        if not np.any(modifiable):
            break
            
        # Distribute difference proportionally among modifiable weights
        w_proj[modifiable] += diff / np.sum(modifiable)
        w_proj = np.clip(w_proj, min_w, max_w)
        
    return w_proj


def rp_sample(portfolio: Portfolio, permutations: int = 1000, **kwargs) -> np.ndarray:
    """
    Generate random portfolios using random sampling from a discrete weight grid, 
    similar to R's randomize_portfolio / rp_sample method.
    """
    n = len(portfolio.assets)
    constraints = portfolio.get_constraints()
    min_w = constraints["min"].values
    max_w = constraints["max"].values
    target_sum = constraints.get("min_sum", 1.0)
    max_pos = constraints.get("max_pos")

    # Define weight sequence grid
    wt_min = max(0.0, np.min(min_w)) if np.all(np.isfinite(min_w)) else 0.0
    wt_max = min(1.0, np.max(max_w)) if np.all(np.isfinite(max_w)) else 1.0
    
    # Generate random weights from grid [wt_min, wt_max] with 0.002 steps
    weight_seq = np.arange(wt_min, wt_max + 0.002, 0.002)
    if max_pos is not None and 0.0 not in weight_seq:
        weight_seq = np.append([0.0], weight_seq)
        
    # Sample completely random weights from the sequence
    random_weights = np.random.choice(weight_seq, size=(permutations, n))
    
    valid_portfolios = []
    
    for i in range(permutations):
        w = random_weights[i].copy()
        
        # 1 & 2. Normalize and project into box constraints, respecting max_pos
        w_proj = _project_and_normalize(w, min_w, max_w, target_sum, max_pos=max_pos)
        
        # 3. Validate constraints (sum and box are usually fixed by projection)
        if abs(w_proj.sum() - target_sum) < 1e-4 and np.all(w_proj >= min_w - 1e-6) and np.all(w_proj <= max_w + 1e-6):
            if max_pos is None or np.sum(w_proj > 1e-5) <= max_pos:
                valid_portfolios.append(w_proj)

    # Return unique portfolios
    if not valid_portfolios:
        return np.empty((0, n))
        
    res = np.array(valid_portfolios)
    return np.unique(res, axis=0)


def rp_grid(portfolio: Portfolio, permutations: int = 2000, **kwargs) -> np.ndarray:
    """
    Generate random portfolios based on grid search method.
    Calculates step size based on required permutations and number of assets,
    generates grid, and normalizes.
    """
    n = len(portfolio.assets)
    constraints = portfolio.get_constraints()
    min_w = constraints["min"].values
    max_w = constraints["max"].values
    target_sum = constraints.get("min_sum", 1.0)
    max_pos = constraints.get("max_pos")
    
    # Calculate number of levels per asset to hit ~permutations total
    # levels^n = permutations -> levels = permutations^(1/n)
    levels = max(2, int(np.ceil(permutations ** (1 / n))))
    
    # Create grid for each asset
    grids = []
    for i in range(n):
        mn = min_w[i] if np.isfinite(min_w[i]) else 0.0
        mx = max_w[i] if np.isfinite(max_w[i]) else 1.0
        grids.append(np.linspace(mn, mx, levels))
        
    # Generate all combinations
    product_iter = itertools.product(*grids)
    
    valid_portfolios = []
    count = 0
    max_iter = permutations * 10  # Prevent infinite loops
    
    for combo in product_iter:
        if count >= max_iter or len(valid_portfolios) >= permutations:
            break
        count += 1
        
        w = np.array(combo)
        
        # Normalize and project
        w_proj = _project_and_normalize(w, min_w, max_w, target_sum, max_pos=max_pos)
        
        if abs(w_proj.sum() - target_sum) < 1e-4 and np.all(w_proj >= min_w - 1e-6) and np.all(w_proj <= max_w + 1e-6):
            if max_pos is None or np.sum(w_proj > 1e-5) <= max_pos:
                valid_portfolios.append(w_proj)
            
    res = np.array(valid_portfolios)
    if len(res) > 0:
        return np.unique(res, axis=0)
    return np.empty((0, n))


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
            if w_start.sum() > 0:
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
    portfolio: Portfolio, permutations: int = 1000, method: str = "sample", **kwargs
) -> np.ndarray:
    """
    Generate random portfolios.
    
    Available methods:
      - 'sample': Random walk sampling from an initialized grid (R PortfolioAnalytics default)
      - 'simplex': Simplex sampling with dirichlet distribution
      - 'grid': Grid search permutation approach
      - 'transform': Pairwise transformation random walk
    """
    if method == "simplex":
        return rp_simplex(portfolio, permutations, **kwargs)
    elif method == "sample":
        return rp_sample(portfolio, permutations, **kwargs)
    elif method == "grid":
        return rp_grid(portfolio, permutations, **kwargs)
    elif method == "transform":
        return rp_transform(portfolio, permutations, **kwargs)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")
