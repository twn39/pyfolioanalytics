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
        self.opt_results = opt_results


def backtest_portfolio(
    R: pd.DataFrame,
    portfolio: Union[Portfolio, RegimePortfolio],
    rebalance_periods: str = "M",
    optimize_method: str = "ROI",
    **kwargs,
) -> BacktestResult:
    """
    Simple walk-forward backtest.
    """
    # Ensure R index is datetime
    if not isinstance(R.index, pd.DatetimeIndex):
        R.index = pd.to_datetime(R.index)

    # Standardize column names if needed (handle dots vs spaces)
    for col in R.columns:
        if isinstance(col, str) and "%" in col:
            try:
                R[col] = R[col].str.replace("%", "").astype(float) / 100
            except Exception:
                pass
    R = R.astype(float)

    # Identify rebalancing dates
    rebal_dates = pd.date_range(
        start=R.index[0], end=R.index[-1], freq=rebalance_periods
    )
    # Ensure the first date is the start of the data
    if rebal_dates[0] > R.index[0]:
        rebal_dates = rebal_dates.insert(0, R.index[0])

    all_weights = []
    all_opt_results = []

    current_weights = pd.Series(0.0, index=R.columns)

    # For walk-forward, we need a lookback window or we use expanding window
    # Default: use all data up to rebalance date

    for i in range(len(rebal_dates) - 1):
        start_date = rebal_dates[i]
        end_date = rebal_dates[i + 1]

        # Data for optimization (lookback)
        R_train = R[:start_date]
        if len(R_train) < 2:
            # Not enough data to optimize, use equal weight or initial if provided
            n = len(R.columns)
            current_weights = pd.Series(1.0 / n, index=R.columns)
        else:
            # Determine which portfolio to use (for RegimePortfolio)
            if isinstance(portfolio, RegimePortfolio):
                # Simple logic: use the regime of the last data point
                # TODO: Implement more complex regime detection
                regime = "default"  # Placeholder
                active_portfolio = portfolio.get_portfolio(regime)
            else:
                active_portfolio = portfolio

            res = optimize_portfolio(
                R_train, active_portfolio, optimize_method=optimize_method, **kwargs
            )
            if res["weights"] is not None:
                current_weights = res["weights"]
                all_opt_results.append(
                    {
                        "date": start_date,
                        "weights": current_weights,
                        "portfolio": active_portfolio,
                        "status": res["status"],
                    }
                )

        # Period returns
        R_period = R[start_date:end_date]
        # In practice, weights are applied at the CLOSE of start_date
        # and we earn returns from the next day.
        # This is a simplified implementation.

        weights_df = pd.DataFrame(
            [current_weights] * len(R_period), index=R_period.index
        )
        all_weights.append(weights_df)

    full_weights = pd.concat(all_weights)
    # Calculate portfolio returns
    # Note: sum(w * r)
    port_returns = (full_weights * R.loc[full_weights.index]).sum(axis=1)

    return BacktestResult(full_weights, port_returns, all_opt_results)
