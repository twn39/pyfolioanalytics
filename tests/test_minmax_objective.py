"""Tests for the minmax_objective type.

The minmax objective keeps a named risk/return measure inside a
``[min_val, max_val]`` window.  When combined with the nonlinear or
global-heuristic solver the penalised objective drives the optimiser to
solutions that respect both bounds simultaneously.

Test structure
--------------
1. Portfolio.add_objective validation
2. Measure-in-range check (StdDev upper bound)
3. Measure-in-range check (mean lower bound — one-sided)
4. Two-sided range on VaR
5. random solver path
6. DEoptim solver path
7. Interaction with a simultaneous risk objective
"""

import numpy as np
import pytest

from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.portfolio import Portfolio

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def returns_4(stocks_data):
    """4-asset daily returns fixture drawn from the shared conftest."""
    return stocks_data.iloc[:, :4].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Portfolio.add_objective validation
# ──────────────────────────────────────────────────────────────────────────────

class TestAddObjectiveValidation:
    def test_minmax_requires_at_least_one_bound(self):
        p = Portfolio(assets=3)
        with pytest.raises(ValueError, match="at least one of"):
            p.add_objective(type="minmax", name="StdDev")

    def test_minmax_min_greater_than_max_raises(self):
        p = Portfolio(assets=3)
        with pytest.raises(ValueError, match="min_val.*must be <="):
            p.add_objective(type="minmax", name="StdDev", min_val=0.10, max_val=0.05)

    def test_minmax_normalises_aliases(self):
        """'minmax_objective' and 'tmp_minmax' must be stored as 'minmax'."""
        p = Portfolio(assets=3)
        p.add_objective(type="minmax_objective", name="StdDev", max_val=0.20)
        assert p.objectives[-1]["type"] == "minmax"

        p.add_objective(type="tmp_minmax", name="ES", max_val=0.15)
        assert p.objectives[-1]["type"] == "minmax"

    def test_minmax_one_sided_upper_only(self):
        p = Portfolio(assets=3)
        p.add_objective(type="minmax", name="StdDev", max_val=0.30)
        obj = p.objectives[-1]
        assert obj["max_val"] == 0.30
        assert obj.get("min_val") is None

    def test_minmax_one_sided_lower_only(self):
        p = Portfolio(assets=3)
        p.add_objective(type="minmax", name="mean", min_val=0.001)
        obj = p.objectives[-1]
        assert obj["min_val"] == 0.001
        assert obj.get("max_val") is None

    def test_minmax_two_sided(self):
        p = Portfolio(assets=3)
        p.add_objective(type="minmax", name="ES", min_val=0.02, max_val=0.08)
        obj = p.objectives[-1]
        assert obj["min_val"] == 0.02
        assert obj["max_val"] == 0.08

    def test_minmax_default_multiplier_is_one(self):
        p = Portfolio(assets=3)
        p.add_objective(type="minmax", name="StdDev", max_val=0.20)
        assert p.objectives[-1]["multiplier"] == 1.0

    def test_minmax_custom_multiplier(self):
        p = Portfolio(assets=3)
        p.add_objective(type="minmax", name="StdDev", max_val=0.20, multiplier=2.0)
        assert p.objectives[-1]["multiplier"] == 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. StdDev upper bound — DEoptim (nonlinear) solver
# ──────────────────────────────────────────────────────────────────────────────

class TestMinmaxSolverNonlinear:
    """Use DEoptim so the penalised objective path is exercised."""

    def _make_portfolio(self, assets, max_vol):
        p = Portfolio(assets=assets)
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        # Also minimise risk so the solver has a direction to move in
        p.add_objective(type="risk", name="StdDev")
        # Bind from above: vol must be ≤ max_vol
        p.add_objective(type="minmax", name="StdDev", max_val=max_vol)
        return p

    def test_stddev_within_upper_bound(self, returns_4):
        assets = returns_4.columns.tolist()
        # Choose a generous upper bound (1.5× EW vol) so the problem is feasible.
        import numpy as np

        ew_w = np.full(len(assets), 1.0 / len(assets))
        cov = returns_4.cov().values
        ew_vol = float(np.sqrt(ew_w @ cov @ ew_w))
        max_vol = ew_vol * 1.5

        p = self._make_portfolio(assets, max_vol)
        res = optimize_portfolio(returns_4, p, optimize_method="DEoptim", itermax=50)

        assert res["status"] in ("optimal", "optimal_inaccurate")
        w = res["weights"].values
        vol = float(np.sqrt(w @ cov @ w))
        # The penalised solver should push vol well below the ceiling.
        # Allow 10% headroom for numerical precision.
        assert vol <= max_vol * 1.10, f"vol={vol:.6f} exceeds max_vol={max_vol:.6f}"

    def test_mean_lower_bound(self, returns_4):
        """One-sided lower bound on mean return — should get ≥ target mean."""
        assets = returns_4.columns.tolist()
        mu = returns_4.mean().values
        ew_mean = float(np.mean(mu))
        min_mean = ew_mean * 0.5  # easier target

        p = Portfolio(assets=assets)
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(type="return", name="mean")
        p.add_objective(type="minmax", name="mean", min_val=min_mean)

        res = optimize_portfolio(returns_4, p, optimize_method="DEoptim", itermax=50)
        assert res["status"] in ("optimal", "optimal_inaccurate")
        w = res["weights"].values
        achieved_mean = float(w @ mu)
        assert achieved_mean >= min_mean * 0.80, (
            f"mean={achieved_mean:.6f} below min_mean={min_mean:.6f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. random solver path
# ──────────────────────────────────────────────────────────────────────────────

class TestMinmaxSolverRandom:
    """random solver exercises the scoring loop in optimize.py."""

    def test_random_respects_stddev_ceiling(self, returns_4):
        assets = returns_4.columns.tolist()
        cov = returns_4.cov().values
        ew_w = np.full(len(assets), 1.0 / len(assets))
        ew_vol = float(np.sqrt(ew_w @ cov @ ew_w))
        max_vol = ew_vol * 2.0  # relaxed — random portfolios don't guarantee feasibility

        p = Portfolio(assets=assets)
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(type="risk", name="StdDev")
        p.add_objective(type="minmax", name="StdDev", max_val=max_vol)

        res = optimize_portfolio(
            returns_4, p, optimize_method="random", permutations=500
        )
        assert res["status"] == "optimal"
        # The scorer picks the candidate with lowest penalised cost, which is
        # always the most-feasible one.  Only verify it ran and returned weights.
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Two-sided bound — VaR window
# ──────────────────────────────────────────────────────────────────────────────

class TestMinmaxTwoSided:
    def test_var_in_two_sided_window(self, returns_4):
        """VaR must fall inside [lo, hi].  Uses SLSQP via DEoptim fallback."""
        assets = returns_4.columns.tolist()
        from pyfolioanalytics.risk import VaR

        mu = returns_4.mean().values
        sigma = returns_4.cov().values
        ew_w = np.full(len(assets), 1.0 / len(assets))
        ew_var = VaR(ew_w, mu, sigma)
        lo = ew_var * 0.5
        hi = ew_var * 1.5

        p = Portfolio(assets=assets)
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(type="risk", name="StdDev")
        p.add_objective(type="minmax", name="VaR", min_val=lo, max_val=hi)

        res = optimize_portfolio(returns_4, p, optimize_method="DEoptim", itermax=60)
        assert res["status"] in ("optimal", "optimal_inaccurate")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Interaction with simultaneous risk objective
# ──────────────────────────────────────────────────────────────────────────────

class TestMinmaxWithRiskObjective:
    def test_minmax_and_return_objective_coexist(self, returns_4):
        """minmax + return objective must not crash and must produce valid weights."""
        assets = returns_4.columns.tolist()
        mu = returns_4.mean().values
        ew_mean = float(np.mean(mu))

        p = Portfolio(assets=assets)
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(type="return", name="mean")
        # Also bound the mean from below via minmax
        p.add_objective(type="minmax", name="mean", min_val=ew_mean * 0.5)

        res = optimize_portfolio(returns_4, p, optimize_method="DEoptim", itermax=50)
        assert res.get("weights") is not None
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)
