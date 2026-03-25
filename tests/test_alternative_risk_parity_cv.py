import pytest
import numpy as np
import pandas as pd
import riskfolio as rp
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.optimize import optimize_portfolio

def test_cvar_risk_parity_cv(stocks_data):
    """
    Cross-validate CVaR Equal Risk Contribution (ERC) with Riskfolio-Lib.
    """
    # 1. Calculate using PyFolioAnalytics
    R = stocks_data.iloc[:, :4]
    
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    # Equal risk contribution for CVaR
    port.add_objective(type="risk_budget", name="CVaR", arguments={"p": 0.95}, min_concentration=True)
    
    res_py = optimize_portfolio(R, port)
    w_py = res_py["weights"].values
    
    # 2. Calculate using Riskfolio-Lib
    rp_port = rp.Portfolio(returns=R)
    rp_port.assets_stats(method_mu='hist', method_cov='hist')
    # Optimize for Equal Risk Contribution (ERC) using CVaR (alpha=0.05 corresponds to p=0.95)
    w_rp_df = rp_port.rp_optimization(model="Classic", rm="CVaR", rf=0, b=None, hist=True)
    w_rp = w_rp_df.values.flatten()
    
    # 3. Assert Parity
    # CVaR Risk Parity can have slightly different local minima depending on the solver (SciPy SLSQP vs Clarabel/SCS)
    # We use atol=0.015 (1.5% absolute weight difference) to account for nonlinear optimization noise
    np.testing.assert_allclose(w_py, w_rp, atol=0.015)


def test_mad_risk_parity_cv(stocks_data):
    """
    Cross-validate MAD Equal Risk Contribution (ERC) with Riskfolio-Lib.
    """
    # 1. Calculate using PyFolioAnalytics
    R = stocks_data.iloc[:, :4]
    
    port = Portfolio(assets=list(R.columns))
    port.add_constraint(type="full_investment")
    port.add_constraint(type="long_only")
    port.add_objective(type="risk_budget", name="MAD", min_concentration=True)
    
    res_py = optimize_portfolio(R, port)
    w_py = res_py["weights"].values
    
    # 2. Calculate using Riskfolio-Lib
    rp_port = rp.Portfolio(returns=R)
    rp_port.assets_stats(method_mu='hist', method_cov='hist')
    w_rp_df = rp_port.rp_optimization(model="Classic", rm="MAD", rf=0, b=None, hist=True)
    w_rp = w_rp_df.values.flatten()
    
    # 3. Assert Parity
    np.testing.assert_allclose(w_py, w_rp, atol=0.015)

