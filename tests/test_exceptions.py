import numpy as np
import pytest

from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.convex_solvers import ConvexOptimizer

def test_nan_returns(stocks_data):
    # Inject NaN
    R = stocks_data.iloc[:, :5].copy()
    R.iloc[0, 0] = np.nan
    
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="StdDev")
    
    with pytest.raises(ValueError, match="NaN values"):
        optimize_portfolio(R, port)

def test_inf_returns(stocks_data):
    # Inject Inf
    R = stocks_data.iloc[:, :5].copy()
    R.iloc[0, 0] = np.inf
    
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="StdDev")
    
    with pytest.raises(ValueError, match="infinite values"):
        optimize_portfolio(R, port)

def test_missing_R_for_cvar(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="CVaR")
    
    moments = {
        "mu": R.mean().values.reshape(-1, 1),
        "sigma": R.cov().values
    }
    
    opt = ConvexOptimizer(moments, port.get_constraints(), port.objectives, R=None)
    # The cvxpy formulation should explicitly reject this
    with pytest.raises(ValueError, match="requires historical returns R"):
        opt.solve()

def test_missing_R_for_edar(stocks_data):
    R = stocks_data.iloc[:, :5]
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_objective(type="risk", name="EDaR")
    
    moments = {
        "mu": R.mean().values.reshape(-1, 1),
        "sigma": R.cov().values
    }
    
    opt = ConvexOptimizer(moments, port.get_constraints(), port.objectives, R=None)
    with pytest.raises(ValueError, match="requires historical returns R"):
        opt.solve()

def test_missing_benchmark_tracking_error():
    port = Portfolio(assets=["A", "B", "C"])
    with pytest.raises(ValueError, match="requires a 'benchmark'"):
        port.add_constraint(type="tracking_error", target=0.05)

def test_missing_benchmark_active_share():
    port = Portfolio(assets=["A", "B", "C"])
    with pytest.raises(ValueError, match="requires a 'benchmark'"):
        port.add_constraint(type="active_share", target=0.6)

