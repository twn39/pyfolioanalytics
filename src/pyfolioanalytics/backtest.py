from typing import Any

import numpy as np
import pandas as pd

from .optimize import optimize_portfolio
from .portfolio import Portfolio, RegimePortfolio


class BacktestResult:
    def __init__(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
        opt_results: list[dict[str, Any]],
        eop_weights: pd.DataFrame | None = None,
        turnover: pd.Series | None = None,
        net_returns: pd.Series | None = None,
    ):
        self.weights = weights
        self.returns = returns
        self.portfolio_returns = returns  # Alias for backward compatibility
        self.opt_results = opt_results
        self.eop_weights = eop_weights if eop_weights is not None else weights.copy()
        self.turnover = (
            turnover if turnover is not None else pd.Series(0.0, index=returns.index)
        )
        self.net_returns = net_returns if net_returns is not None else returns.copy()

    def summary(
        self, risk_free_rate: float = 0.0, benchmark_returns: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Calculate key performance and risk metrics (Tearsheet) for the backtest.
        Returns a DataFrame comparing Gross vs Net returns.
        """
        metrics = {}

        for ret_type, ret_series in [
            ("Gross", self.returns),
            ("Net", self.net_returns),
        ]:
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
            sharpe = (
                (excess_ret.mean() / ret_series.std()) * np.sqrt(252)
                if ret_series.std() > 0
                else np.nan
            )

            # Sortino Ratio
            downside_ret = ret_series[ret_series < 0]
            downside_vol = downside_ret.std() * np.sqrt(252)
            sortino = (
                (excess_ret.mean() * np.sqrt(252)) / downside_vol
                if downside_vol > 0
                else np.nan
            )

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
            es_95 = (
                -ret_series[ret_series <= -var_95].mean()
                if len(ret_series[ret_series <= -var_95]) > 0
                else np.nan
            )

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
                        r_up = (1 + ret_a[up_market]).prod() ** (
                            252 / up_market.sum()
                        ) - 1
                        b_up = (1 + bench_a[up_market]).prod() ** (
                            252 / up_market.sum()
                        ) - 1
                        up_capture = r_up / b_up if b_up > 0 else np.nan

                    if down_market.any():
                        r_down = (1 + ret_a[down_market]).prod() ** (
                            252 / down_market.sum()
                        ) - 1
                        b_down = (1 + bench_a[down_market]).prod() ** (
                            252 / down_market.sum()
                        ) - 1
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
                "Down Capture Ratio": down_capture,
            }

        # Add turnover stat (same for gross/net)
        total_turnover = self.turnover.sum()
        avg_turnover = (
            self.turnover[self.turnover > 0].mean()
            if (self.turnover > 0).any()
            else 0.0
        )

        df = pd.DataFrame(metrics)
        df.loc["Total Turnover"] = total_turnover
        df.loc["Avg Turnover per Rebalance"] = avg_turnover

        return df


def backtest_portfolio(
    R: pd.DataFrame,
    portfolio: Portfolio | RegimePortfolio,
    rebalance_periods: str = "ME",
    optimize_method: str = "ROI",
    ptc: float = 0.0,
    **kwargs,
) -> BacktestResult:
    """
    Walk-forward backtest with rebalancing, including vectorized weight drift,
    turnover calculation, and proportional transaction costs (PTC).
    Also supports AUM-based fees (Management Fee and Performance Fee with High Water Mark)
    and R-style training_period/rolling_window logic.
    """
    initial_aum = kwargs.get("initial_aum", 1.0)
    management_fee = kwargs.get("management_fee", 0.0)
    performance_fee = kwargs.get("performance_fee", 0.0)
    
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

    if not isinstance(R.index, pd.DatetimeIndex):
        R.index = pd.to_datetime(R.index)

    # R PortfolioAnalytics logic for training_period and rolling_window
    training_period = kwargs.get("training_period", 0)
    rolling_window = kwargs.get("rolling_window")

    # The first possible rebalance date must be at least `training_period` observations into the dataset
    if training_period > 0 and len(R) > training_period:
        valid_start_date = R.index[training_period]
    else:
        valid_start_date = R.index[0]

    # Identify all potential rebalancing dates
    raw_rebal_dates = pd.date_range(start=R.index[0], end=R.index[-1], freq=rebalance_periods)
    
    # Filter dates to respect training_period and ensure we cover the very end
    rebal_dates = raw_rebal_dates[raw_rebal_dates >= valid_start_date]
    if len(rebal_dates) == 0 or rebal_dates[0] > valid_start_date:
        rebal_dates = rebal_dates.insert(0, valid_start_date)
    if rebal_dates[-1] < R.index[-1]:
        # Add a dummy end date that goes just past the end of the data to capture the last slice
        rebal_dates = rebal_dates.insert(len(rebal_dates), R.index[-1] + pd.Timedelta(days=1))

    regimes = kwargs.get("regimes")

    all_bop_weights = []
    all_eop_weights = []
    all_returns = []
    all_net_returns = []
    all_turnover = []
    all_opt_results = []

    current_weights = pd.Series(1.0 / len(R.columns), index=R.columns)
    last_eop_weights = pd.Series(0.0, index=R.columns)

    nav = initial_aum
    hwm = initial_aum
    daily_mgt_fee_rate = management_fee / 252.0

    for i in range(len(rebal_dates) - 1):
        start_date = rebal_dates[i]
        end_date = rebal_dates[i + 1]

        # Extract strict half-open interval [start_date, end_date) to prevent overlapping days
        R_period = R.loc[(R.index >= start_date) & (R.index < end_date)]
        if R_period.empty:
            continue

        # Data for optimization (Strictly before start_date to prevent lookahead bias)
        loc = R.index.get_indexer([start_date], method="bfill")[0]
        if rolling_window and loc >= rolling_window:
            start_idx = loc - rolling_window
            R_train = R.iloc[start_idx:loc]
        else:
            # If rolling_window is None or not enough history, use expanding window
            R_train = R.iloc[:loc]

        if len(R_train) >= 2:
            if isinstance(portfolio, RegimePortfolio):
                if regimes is not None:
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
                if "moments" in res:
                    opt_info["moments"] = res["moments"]
                all_opt_results.append(opt_info)

        # 1. Turnover at rebalance
        turnover_val = np.abs(current_weights - last_eop_weights).sum() / 2.0
        
        # 2. Vectorized Drift and NAV Calculation
        T, N = R_period.shape
        R_vals = R_period.values
        w = current_weights.values
        
        # Calculate daily growth factor for each asset: (1 + R_i,t)
        asset_growth = 1.0 + R_vals
        
        # Cumulative growth of each asset over the period
        cum_asset_growth = np.cumprod(asset_growth, axis=0)
        
        # Value of each asset over time, assuming we start with $w allocation
        value_matrix = w.reshape(1, N) * cum_asset_growth
        
        # Total portfolio value over time (relative to starting $1)
        port_rel_value = np.sum(value_matrix, axis=1)
        
        # EOP Weights = current value of asset / total portfolio value
        with np.errstate(divide='ignore', invalid='ignore'):
            eop_weights_matrix = value_matrix / port_rel_value.reshape(-1, 1)
            eop_weights_matrix[port_rel_value == 0] = 0.0
        
        # BOP Weights: first day is target `w`, subsequent days are previous day's EOP
        bop_weights_matrix = np.zeros_like(eop_weights_matrix)
        bop_weights_matrix[0, :] = w
        if T > 1:
            bop_weights_matrix[1:, :] = eop_weights_matrix[:-1, :]
            
        # Portfolio Gross Returns: V_t / V_{t-1} - 1
        port_ret_array = np.zeros(T)
        port_ret_array[0] = np.dot(w, R_vals[0])
        if T > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                port_ret_array[1:] = (port_rel_value[1:] / port_rel_value[:-1]) - 1.0
            port_ret_array[1:][port_rel_value[:-1] == 0] = 0.0
            
        # NAV and Net Returns tracking
        net_ret_array = np.zeros(T)
        turnover_array = np.zeros(T)
        turnover_array[0] = turnover_val
        
        # Vectorized NAV calculation
        # The portfolio grows by port_ret_array each day, and shrinks by daily_mgt_fee_rate
        # For the first day, we must apply the turnover PTC drop BEFORE market growth
        nav_trajectory = np.zeros(T)
        
        nav_after_ptc = nav * (1.0 - turnover_val * ptc)
        nav_trajectory[0] = nav_after_ptc * (1.0 + port_ret_array[0]) * (1.0 - daily_mgt_fee_rate)
        
        # Cumulative compounding for the rest of the period
        if T > 1:
            net_daily_growth = (1.0 + port_ret_array[1:]) * (1.0 - daily_mgt_fee_rate)
            nav_trajectory[1:] = nav_trajectory[0] * np.cumprod(net_daily_growth)
            
        # Apply Performance Fee on the last day of the period
        if performance_fee > 0:
            final_nav = nav_trajectory[-1]
            if final_nav > hwm:
                perf_fee_amount = (final_nav - hwm) * performance_fee
                nav_trajectory[-1] -= perf_fee_amount
                hwm = nav_trajectory[-1]
                
        # Calculate Net Returns
        net_ret_array[0] = (nav_trajectory[0] / nav) - 1.0
        if T > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                net_ret_array[1:] = (nav_trajectory[1:] / nav_trajectory[:-1]) - 1.0
            net_ret_array[1:][nav_trajectory[:-1] == 0] = 0.0
            
        # Update state for next period
        nav = nav_trajectory[-1]
        last_eop_weights = pd.Series(eop_weights_matrix[-1, :], index=R.columns)

        all_bop_weights.append(pd.DataFrame(bop_weights_matrix, index=R_period.index, columns=R.columns))
        all_eop_weights.append(pd.DataFrame(eop_weights_matrix, index=R_period.index, columns=R.columns))
        all_returns.append(pd.Series(port_ret_array, index=R_period.index))
        all_net_returns.append(pd.Series(net_ret_array, index=R_period.index))
        all_turnover.append(pd.Series(turnover_array, index=R_period.index))

    if not all_bop_weights:
        return BacktestResult(pd.DataFrame(), pd.Series(), [])

    # Concatenate all periods (No overlaps due to half-open intervals!)
    full_weights = pd.concat(all_bop_weights)
    full_eop = pd.concat(all_eop_weights)
    full_returns = pd.concat(all_returns)
    full_net_returns = pd.concat(all_net_returns)
    full_turnover = pd.concat(all_turnover)

    return BacktestResult(
        weights=full_weights,
        returns=full_returns,
        opt_results=all_opt_results,
        eop_weights=full_eop,
        turnover=full_turnover,
        net_returns=full_net_returns,
    )


# Alias for backward compatibility
optimize_portfolio_rebalancing = backtest_portfolio
