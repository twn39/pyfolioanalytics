
import cvxpy as cp
import numpy as np
import pandas as pd


def get_latest_prices(prices: pd.DataFrame) -> pd.Series:
    """
    Retrieve the most recent asset prices from a dataframe.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    return prices.ffill().iloc[-1]


class DiscreteAllocation:
    """
    Generate a discrete portfolio allocation from continuous weights.
    """

    def __init__(
        self,
        weights: dict[str, float],
        latest_prices: pd.Series,
        total_portfolio_value: float = 10000.0,
        short_ratio: float | None = None,
    ):
        if not isinstance(weights, dict):
            raise TypeError("weights must be a dictionary of {ticker: weight}")
        if not isinstance(latest_prices, pd.Series):
            raise TypeError("latest_prices must be a pandas Series")
        if total_portfolio_value <= 0:
            raise ValueError("total_portfolio_value must be greater than zero")

        self.weights = list(weights.items())
        self.latest_prices = latest_prices
        self.total_portfolio_value = total_portfolio_value

        if short_ratio is None:
            self.short_ratio = sum([-w for _, w in self.weights if w < 0])
        else:
            self.short_ratio = short_ratio

        self.allocation: dict[str, int] = {}

    def _remove_zero_positions(self, allocation: dict[str, int]) -> dict[str, int]:
        return {k: v for k, v in allocation.items() if v != 0}

    def greedy_portfolio(self, reinvest: bool = False) -> tuple[dict[str, int], float]:
        """
        Convert continuous weights into a discrete portfolio allocation
        using a greedy iterative approach.
        """
        # Sort in descending order of weight
        self.weights.sort(key=lambda x: x[1], reverse=True)

        # Handle shorts if present
        if any(w < 0 for _, w in self.weights):
            longs = {t: w for t, w in self.weights if w >= 0}
            shorts = {t: -w for t, w in self.weights if w < 0}

            long_total = sum(longs.values())
            short_total = sum(shorts.values())

            # Re-normalize
            longs = {t: w / long_total for t, w in longs.items()}
            shorts = {t: w / short_total for t, w in shorts.items()}

            short_val = self.total_portfolio_value * self.short_ratio
            long_val = self.total_portfolio_value
            if reinvest:
                long_val += short_val

            da_long = DiscreteAllocation(longs, self.latest_prices[list(longs.keys())], total_portfolio_value=long_val)
            long_alloc, long_leftover = da_long.greedy_portfolio()

            da_short = DiscreteAllocation(shorts, self.latest_prices[list(shorts.keys())], total_portfolio_value=short_val)
            short_alloc, short_leftover = da_short.greedy_portfolio()
            short_alloc = {t: -v for t, v in short_alloc.items()}

            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            self.allocation = self._remove_zero_positions(self.allocation)
            return self.allocation, long_leftover + short_leftover

        # Long-only greedy allocation
        available_funds = self.total_portfolio_value
        shares_bought = []
        buy_prices = []
        tickers = []

        # First round: Floor allocation
        for ticker, weight in self.weights:
            price = self.latest_prices[ticker]
            n_shares = int(weight * self.total_portfolio_value / price)
            cost = n_shares * price
            available_funds -= cost
            shares_bought.append(n_shares)
            buy_prices.append(price)
            tickers.append(ticker)

        # Second round: Fill remaining funds by minimizing deficit
        while available_funds > 0:
            current_vals = np.array(buy_prices) * np.array(shares_bought)
            current_total = np.sum(current_vals)

            if current_total == 0:
                # If nothing bought in round 1, just pick the one with highest weight
                ideal_weights = np.array([w for _, w in self.weights])
                idx = np.argmax(ideal_weights)
            else:
                current_weights = current_vals / current_total
                ideal_weights = np.array([w for _, w in self.weights])
                deficit = ideal_weights - current_weights
                idx = np.argmax(deficit)

            ticker = tickers[idx]
            price = buy_prices[idx]

            # Find next best if can't afford
            if price > available_funds:
                # Try to find any asset that fits
                found = False
                # Sort remaining by deficit
                if current_total > 0:
                    deficit = ideal_weights - (np.array(buy_prices) * np.array(shares_bought)) / current_total
                    sort_idx = np.argsort(deficit)[::-1]
                    for i in sort_idx:
                        if buy_prices[i] <= available_funds:
                            idx = i
                            ticker = tickers[idx]
                            price = buy_prices[idx]
                            found = True
                            break
                if not found:
                    break

            shares_bought[idx] += 1
            available_funds -= price

        self.allocation = self._remove_zero_positions(dict(zip(tickers, shares_bought)))
        return self.allocation, available_funds

    def lp_portfolio(self, reinvest: bool = False, solver: str | None = None) -> tuple[dict[str, int], float]:
        """
        Convert continuous weights into a discrete portfolio allocation
        using integer programming.
        """
        if any(w < 0 for _, w in self.weights):
            longs = {t: w for t, w in self.weights if w >= 0}
            shorts = {t: -w for t, w in self.weights if w < 0}

            long_total = sum(longs.values())
            short_total = sum(shorts.values())

            longs = {t: w / long_total for t, w in longs.items()}
            shorts = {t: w / short_total for t, w in shorts.items()}

            short_val = self.total_portfolio_value * self.short_ratio
            long_val = self.total_portfolio_value
            if reinvest:
                long_val += short_val

            da_long = DiscreteAllocation(longs, self.latest_prices[list(longs.keys())], total_portfolio_value=long_val)
            long_alloc, long_leftover = da_long.lp_portfolio(solver=solver)

            da_short = DiscreteAllocation(shorts, self.latest_prices[list(shorts.keys())], total_portfolio_value=short_val)
            short_alloc, short_leftover = da_short.lp_portfolio(solver=solver)
            short_alloc = {t: -v for t, v in short_alloc.items()}

            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            self.allocation = self._remove_zero_positions(self.allocation)
            return self.allocation, long_leftover + short_leftover

        prices = self.latest_prices.values
        n = len(prices)
        ideal_weights = np.array([w for _, w in self.weights])

        x = cp.Variable(n, integer=True)
        remaining_funds = self.total_portfolio_value - prices @ x

        # Absolute difference between desired dollar amount and actual dollar amount
        delta = ideal_weights * self.total_portfolio_value - cp.multiply(x, prices)
        u = cp.Variable(n)

        constraints = [
            delta <= u,
            -delta <= u,
            x >= 0,
            remaining_funds >= 0
        ]

        objective = cp.sum(u) + remaining_funds
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=solver)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return self.greedy_portfolio()

        vals = np.rint(np.array(x.value)).astype(int)
        tickers = [t for t, _ in self.weights]
        self.allocation = self._remove_zero_positions(dict(zip(tickers, vals)))

        return self.allocation, float(remaining_funds.value)
