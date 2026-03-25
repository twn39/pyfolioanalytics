import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from .portfolio import Portfolio, RegimePortfolio
from .optimize import optimize_portfolio


class BacktestResult:
    def __init__(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
        opt_results: List[Dict[str, Any]],
        eop_weights: Optional[pd.DataFrame] = None,
        turnover: Optional[pd.Series] = None,
        net_returns: Optional[pd.Series] = None,
    ):
        self.weights = weights
        self.returns = returns
        self.portfolio_returns = returns  # Alias for backward compatibility
        self.opt_results = opt_results
        self.eop_weights = eop_weights if eop_weights is not None else weights.copy()
        self.turnover = turnover if turnover is not None else pd.Series(0.0, index=returns.index)
        self.net_returns = net_returns if net_returns is not None else returns.copy()

    def summary(self, risk_free_rate: float = 0.0, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate key performance and risk metrics (Tearsheet) for the backtest.
        Returns a DataFrame comparing Gross vs Net returns.
        """
        metrics = {}
        
        for ret_type, ret_series in [("Gross", self.returns), ("Net", self.net_returns)]:
            if ret_series.empty:
                continue
                
            # Basic stats
            # Assuming daily returns for annualized factors if not specified, 
            # ideally we should infer freq from index, but 252 is standard for daily trading
            # We'll calculate total return first to be safe
            cum_ret = (1 + ret_series).prod() - 1
            n_days = len(ret_series)
            cagr = (1 + cum_ret) ** (252 / max(1, n_days)) - 1
            
            ann_vol = ret_series.std() * np.sqrt(252)
            
            # Sharpe Ratio
            excess_ret = ret_series - (risk_free_rate / 252)
            sharpe = (excess_ret.mean() / ret_series.std()) * np.sqrt(252) if ret_series.std() > 0 else np.nan
            
            # Sortino Ratio
            downside_ret = ret_series[ret_series < 0]
            downside_vol = downside_ret.std() * np.sqrt(252)
            sortino = (excess_ret.mean() * np.sqrt(252)) / downside_vol if downside_vol > 0 else np.nan
            
            # Max Drawdown
            cum_vals = (1 + ret_series).cumprod()
            rolling_max = cum_vals.cummax()
            drawdowns = (cum_vals / rolling_max) - 1.0
            max_dd = drawdowns.min()
            
            # Calmar Ratio
            calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else np.nan
            
            # Omega Ratio (threshold = 0)
            returns_less_thresh = ret_series[ret_series < 0]
            returns_above_thresh = ret_series[ret_series >= 0]
            if len(returns_less_thresh) > 0 and returns_less_thresh.sum() != 0:
                omega = returns_above_thresh.sum() / abs(returns_less_thresh.sum())
            else:
                omega = np.nan
                
            # Skewness & Kurtosis
            skew = ret_series.skew()
            kurt = ret_series.kurt()
            
            # Value at Risk / Expected Shortfall (historical 95%)
            var_95 = -np.percentile(ret_series, 5) if len(ret_series) > 0 else np.nan
            es_95 = -ret_series[ret_series <= -var_95].mean() if len(ret_series[ret_series <= -var_95]) > 0 else np.nan
            
            # Tail Ratio: 95th percentile / abs(5th percentile)
            pct_95 = np.percentile(ret_series, 95)
            tail_ratio = pct_95 / abs(var_95) if abs(var_95) > 0 else np.nan
            
            # Hit Ratios
            positive_periods = (ret_series > 0).sum()
            hit_ratio = positive_periods / n_days if n_days > 0 else np.nan

            # Relative metrics if benchmark provided
            info_ratio = np.nan
            tracking_error = np.nan
            up_capture = np.nan
            down_capture = np.nan
            
            if benchmark_returns is not None:
                # Align dates
                aligned = pd.concat([ret_series, benchmark_returns], axis=1).dropna()
                if len(aligned) > 0:
                    ret_a = aligned.iloc[:, 0]
                    bench_a = aligned.iloc[:, 1]
                    
                    active_ret = ret_a - bench_a
                    tracking_error = active_ret.std() * np.sqrt(252)
                    if tracking_error > 0:
                        info_ratio = (active_ret.mean() * 252) / tracking_error
                        
                    # Capture Ratios
                    up_market = bench_a > 0
                    down_market = bench_a < 0
                    
                    if up_market.any():
                        r_up = (1 + ret_a[up_market]).prod() ** (252 / up_market.sum()) - 1
                        b_up = (1 + bench_a[up_market]).prod() ** (252 / up_market.sum()) - 1
                        up_capture = r_up / b_up if b_up > 0 else np.nan
                        
                    if down_market.any():
                        r_down = (1 + ret_a[down_market]).prod() ** (252 / down_market.sum()) - 1
                        b_down = (1 + bench_a[down_market]).prod() ** (252 / down_market.sum()) - 1
                        down_capture = r_down / b_down if b_down < 0 else np.nan

            metrics[ret_type] = {
                "Total Return": cum_ret,
                "CAGR": cagr,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Omega Ratio": omega,
                "Max Drawdown": max_dd,
                "Calmar Ratio": calmar,
                "Tail Ratio": tail_ratio,
                "Hit Ratio": hit_ratio,
                "Skewness": skew,
                "Kurtosis": kurt,
                "Daily VaR (95%)": var_95,
                "Daily CVaR (95%)": es_95,
                "Tracking Error": tracking_error,
                "Information Ratio": info_ratio,
                "Up Capture Ratio": up_capture,
                "Down Capture Ratio": down_capture
            }
            
        # Add turnover stat (same for gross/net)
        total_turnover = self.turnover.sum()
        avg_turnover = self.turnover[self.turnover > 0].mean() if (self.turnover > 0).any() else 0.0
        
        df = pd.DataFrame(metrics)
        df.loc["Total Turnover"] = total_turnover
        df.loc["Avg Turnover per Rebalance"] = avg_turnover
        
        return df


def backtest_portfolio(
    R: pd.DataFrame,
    portfolio: Union[Portfolio, RegimePortfolio],
    rebalance_periods: str = "ME",
    optimize_method: str = "ROI",
    ptc: float = 0.0,
    **kwargs,
) -> BacktestResult:
    """
    Simple walk-forward backtest with rebalancing, including weight drift,
    turnover calculation, and proportional transaction costs (PTC).
    """
    # Handle rebalance_on from PortfolioAnalytics style
    rebalance_on = kwargs.get("rebalance_on")
    if rebalance_on:
        mapping = {
            "months": "ME",
            "quarters": "QE",
            "years": "YE",
            "weeks": "W",
            "days": "D",
        }
        rebalance_periods = mapping.get(rebalance_on, rebalance_periods)

    # Ensure R index is datetime
    if not isinstance(R.index, pd.DatetimeIndex):
        R.index = pd.to_datetime(R.index)

    # Identify rebalancing dates
    rebal_dates = pd.date_range(
        start=R.index[0], end=R.index[-1], freq=rebalance_periods
    )
    if rebal_dates[0] > R.index[0]:
        rebal_dates = rebal_dates.insert(0, R.index[0])

    rolling_window = kwargs.get("rolling_window")
    regimes = kwargs.get("regimes")

    all_bop_weights = []
    all_eop_weights = []
    all_returns = []
    all_net_returns = []
    all_turnover = []
    all_opt_results = []
    
    current_weights = pd.Series(1.0 / len(R.columns), index=R.columns)
    last_eop_weights = pd.Series(0.0, index=R.columns)

    for i in range(len(rebal_dates) - 1):
        start_date = rebal_dates[i]
        end_date = rebal_dates[i + 1]

        # Data for optimization
        if rolling_window:
            # Find integer index of start_date
            loc = R.index.get_indexer([start_date], method="pad")[0]
            start_idx = max(0, loc - rolling_window)
            R_train = R.iloc[start_idx:loc]
        else:
            # Strict less-than prevents rebalance-date return leaking into training
            R_train = R.loc[R.index < start_date]

        if len(R_train) >= 2:
            if isinstance(portfolio, RegimePortfolio):
                if regimes is not None:
                    # Use the regime of the current rebalance date
                    current_regime = regimes.asof(start_date)
                    active_portfolio = portfolio.get_portfolio(current_regime)
                else:
                    active_portfolio = portfolio.get_portfolio("default")
            else:
                active_portfolio = portfolio

            res = optimize_portfolio(
                R_train, active_portfolio, optimize_method=optimize_method, **kwargs
            )
            if res["weights"] is not None:
                current_weights = res["weights"]
                opt_info = {
                    "date": start_date,
                    "weights": current_weights,
                    "portfolio": active_portfolio,
                    "status": res["status"],
                }
                # Ensure moments and other metadata are passed through if present
                if "moments" in res:
                    opt_info["moments"] = res["moments"]
                all_opt_results.append(opt_info)

        # Apply weights to the period
        R_period = R[start_date:end_date]
        if R_period.empty:
            continue
            
        # 1. Calculate Turnover at rebalance date (start_date)
        # Sum of absolute changes between target current_weights and last end-of-period weights
        turnover_val = np.abs(current_weights - last_eop_weights).sum() / 2.0
        
        # 2. Calculate Drift
        # eop_weights(t) = bop_weights(t) * (1 + R(t)) / (1 + R_port(t))
        bop_weights_matrix = np.zeros(R_period.shape)
        eop_weights_matrix = np.zeros(R_period.shape)
        port_ret_array = np.zeros(len(R_period))
        net_ret_array = np.zeros(len(R_period))
        turnover_array = np.zeros(len(R_period))
        
        # First day of the period pays the turnover cost
        turnover_array[0] = turnover_val
        
        w = current_weights.values
        R_vals = R_period.values
        
        for t in range(len(R_period)):
            # Beginning of period weights
            bop_weights_matrix[t, :] = w
            
            # Portfolio return for the day
            r_p = np.dot(w, R_vals[t])
            port_ret_array[t] = r_p
            
            # Net return
            if t == 0:
                net_ret_array[t] = r_p - turnover_val * ptc
            else:
                net_ret_array[t] = r_p
                
            # End of period weights (drift)
            # w_{t+1} = w_t * (1 + R_t) / (1 + R_p)
            w_next = w * (1 + R_vals[t])
            sum_w = np.sum(w_next)
            if sum_w > 0:
                w = w_next / sum_w
            eop_weights_matrix[t, :] = w
            
        last_eop_weights = pd.Series(w, index=R.columns)
        
        all_bop_weights.append(pd.DataFrame(bop_weights_matrix, index=R_period.index, columns=R.columns))
        all_eop_weights.append(pd.DataFrame(eop_weights_matrix, index=R_period.index, columns=R.columns))
        all_returns.append(pd.Series(port_ret_array, index=R_period.index))
        all_net_returns.append(pd.Series(net_ret_array, index=R_period.index))
        all_turnover.append(pd.Series(turnover_array, index=R_period.index))

    if not all_bop_weights:
        return BacktestResult(pd.DataFrame(), pd.Series(), [])

    # Filter out duplicate dates at period boundaries
    # rebal_dates chunks might overlap on end_date/start_date depending on indexing.
    # R[start_date:end_date] is inclusive on both sides in pandas for datetime slices.
    # To avoid double counting the boundary days, we should drop duplicates.
    
    full_weights = pd.concat(all_bop_weights)
    full_eop = pd.concat(all_eop_weights)
    full_returns = pd.concat(all_returns)
    full_net_returns = pd.concat(all_net_returns)
    full_turnover = pd.concat(all_turnover)
    
    # Remove duplicates, keeping the first occurrence (which would be the rebalance day with new weights)
    full_weights = full_weights[~full_weights.index.duplicated(keep='first')]
    full_eop = full_eop[~full_eop.index.duplicated(keep='first')]
    full_returns = full_returns[~full_returns.index.duplicated(keep='first')]
    full_net_returns = full_net_returns[~full_net_returns.index.duplicated(keep='first')]
    full_turnover = full_turnover[~full_turnover.index.duplicated(keep='first')]

    return BacktestResult(
        weights=full_weights, 
        returns=full_returns, 
        opt_results=all_opt_results,
        eop_weights=full_eop,
        turnover=full_turnover,
        net_returns=full_net_returns
    )


# Alias for backward compatibility
optimize_portfolio_rebalancing = backtest_portfolio
