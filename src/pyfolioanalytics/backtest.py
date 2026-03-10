import pandas as pd
from typing import Dict, Any, List, Union
from .portfolio import Portfolio, RegimePortfolio
from .optimize import optimize_portfolio


class BacktestResult:
    def __init__(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
        opt_results: List[Dict[str, Any]],
    ):
        self.weights = weights
        self.returns = returns
        self.portfolio_returns = returns  # Alias for backward compatibility
        self.opt_results = opt_results


def backtest_portfolio(
    R: pd.DataFrame,
    portfolio: Union[Portfolio, RegimePortfolio],
    rebalance_periods: str = "ME",
    optimize_method: str = "ROI",
    **kwargs,
) -> BacktestResult:
    """
    Simple walk-forward backtest with rebalancing.
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

    all_weights = []
    all_opt_results = []
    current_weights = pd.Series(1.0 / len(R.columns), index=R.columns)

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
            active_portfolio = portfolio
            if isinstance(portfolio, RegimePortfolio):
                if regimes is not None:
                    # Use the regime of the current rebalance date
                    current_regime = regimes.asof(start_date)
                    active_portfolio = portfolio.get_portfolio(current_regime)
                else:
                    active_portfolio = portfolio.get_portfolio("default")

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
        if not R_period.empty:
            weights_df = pd.DataFrame(
                [current_weights] * len(R_period), index=R_period.index
            )
            all_weights.append(weights_df)

    if not all_weights:
        return BacktestResult(pd.DataFrame(), pd.Series(), [])

    full_weights = pd.concat(all_weights)
    port_returns = (full_weights * R.loc[full_weights.index]).sum(axis=1)

    return BacktestResult(full_weights, port_returns, all_opt_results)


# Alias for backward compatibility
optimize_portfolio_rebalancing = backtest_portfolio
