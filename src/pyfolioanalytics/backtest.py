import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from .portfolio import Portfolio, RegimePortfolio
from .optimize import optimize_portfolio

class BacktestResult:
    def __init__(self, weights: pd.DataFrame, portfolio_returns: pd.Series, opt_results: List[Dict[str, Any]]):
        self.weights = weights
        self.portfolio_returns = portfolio_returns
        self.opt_results = opt_results

    def __repr__(self):
        return f"BacktestResult(periods={len(self.weights)})"

def optimize_portfolio_rebalancing(
    R: pd.DataFrame,
    portfolio: Union[Portfolio, RegimePortfolio],
    optimize_method: str = "ROI",
    rebalance_on: str = "months",
    training_period: Optional[int] = None,
    rolling_window: Optional[int] = None,
    regimes: Optional[pd.Series] = None,
    **kwargs
) -> BacktestResult:
    """
    Execute portfolio optimization at specified rebalancing intervals.
    """
    # Ensure R is numeric
    R = R.copy()
    for col in R.columns:
        if R[col].dtype == object:
            try:
                R[col] = R[col].str.replace("%", "").astype(float) / 100
            except:
                pass
    R = R.astype(float)
    
    # Identify rebalancing dates
    if rebalance_on == "months":
        rebalance_dates = R.resample('ME').last().index
    elif rebalance_on == "quarters":
        rebalance_dates = R.resample('QE').last().index
    elif rebalance_on == "years":
        rebalance_dates = R.resample('YE').last().index
    else:
        rebalance_dates = R.index
        
    rebalance_dates = list(rebalance_dates)
    if R.index[-1] not in rebalance_dates:
        rebalance_dates.append(R.index[-1])
        
    rebalance_dates = sorted(list(set([d for d in rebalance_dates if d in R.index])))
    
    if training_period is None:
        training_period = 0
        
    all_weights = []
    opt_results = []
    
    # Iterate through rebalancing dates
    for i, date in enumerate(rebalance_dates):
        idx = R.index.get_loc(date)
        
        if idx < (training_period - 1):
            continue
            
        start_idx = 0
        if rolling_window is not None:
            start_idx = max(0, idx - rolling_window)
            
        R_train = R.iloc[start_idx:idx+1]
        
        # Select appropriate portfolio if regime switching
        current_port = portfolio
        if isinstance(portfolio, RegimePortfolio):
            if regimes is None:
                raise ValueError("RegimePortfolio requires a regimes signal.")
            # Take the regime at the current date
            regime_val = regimes.loc[date]
            current_port = portfolio.get_portfolio(regime_val)
            
        # Optimize
        res = optimize_portfolio(R_train, current_port, optimize_method=optimize_method, **kwargs)
        
        if res["weights"] is not None:
            w = res["weights"]
            w.name = date
            all_weights.append(w)
            opt_results.append(res)
            
    # 3. Consolidate results
    weights_df = pd.concat(all_weights, axis=1).T
    
    # Calculate portfolio returns (simple approach: weights fixed during the period)
    # Note: A more precise backtest would account for drift within periods
    returns_list = []
    for i in range(len(weights_df) - 1):
        start_date = weights_df.index[i]
        end_date = weights_df.index[i+1]
        period_returns = R.loc[start_date:end_date].iloc[1:] # Exclude the rebalance day return if it was used for optimization
        w = weights_df.iloc[i].values
        p_ret = period_returns @ w
        returns_list.append(p_ret)
        
    portfolio_returns = pd.concat(returns_list) if returns_list else pd.Series()
    
    return BacktestResult(weights_df, portfolio_returns, opt_results)
