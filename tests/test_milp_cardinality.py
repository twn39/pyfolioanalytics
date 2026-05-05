"""Tests for solve_milp_cardinality() and max_pos routing in optimize_portfolio."""

import numpy as np
import pandas as pd

from pyfolioanalytics.solvers import solve_milp_cardinality

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_correlated_moments(n: int = 5, seed: int = 0) -> tuple:
    """Return (moments, constraints, R) for a small correlated universe."""
    rng = np.random.default_rng(seed)
    # Random correlation matrix with some high-correlation pairs
    A = rng.standard_normal((n, n))
    sigma = (A @ A.T) / n + np.eye(n) * 0.01
    mu = rng.normal(0.001, 0.005, n)
    R = rng.multivariate_normal(mu, sigma, size=200)
    moments = {"mu": mu.reshape(-1, 1), "sigma": sigma}
    constraints = {
        "min_sum": 1.0,
        "max_sum": 1.0,
        "min": pd.Series(np.zeros(n)),
        "max": pd.Series(np.ones(n)),
    }
    return moments, constraints, R


# ── Unit tests for solve_milp_cardinality ────────────────────────────────────

class TestSolveMilpCardinality:

    def test_basic_max_pos_satisfied(self):
        """Returned weights must have at most max_pos non-zero entries."""
        moments, constraints, R = _make_correlated_moments(n=8)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "StdDev"}],
            max_pos=3,
        )
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        n_pos = int(np.sum(result["weights"] > 1e-6))
        assert n_pos <= 3, f"max_pos=3 violated: {n_pos} positions"

    def test_weights_sum_to_one(self):
        """Fully-invested portfolio should sum to 1.0 after normalisation."""
        moments, constraints, R = _make_correlated_moments(n=6)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "StdDev"}],
            max_pos=2,
        )
        assert result["weights"] is not None
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-5

    def test_weights_non_negative(self):
        """Long-only portfolio (lb=0) must return non-negative weights."""
        moments, constraints, _ = _make_correlated_moments(n=5)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "StdDev"}],
            max_pos=2,
        )
        assert result["weights"] is not None
        assert np.all(result["weights"] >= -1e-8)

    def test_n_positions_field(self):
        """'n_positions' key must equal actual non-zero count."""
        moments, constraints, _ = _make_correlated_moments(n=6)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "StdDev"}],
            max_pos=3,
        )
        assert result["n_positions"] is not None
        actual = int(np.sum(result["weights"] > 1e-6))
        assert result["n_positions"] == actual

    def test_cvar_objective(self):
        """CVaR objective with R data should also satisfy max_pos."""
        moments, constraints, R = _make_correlated_moments(n=6)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "CVaR", "arguments": {"alpha": 0.05}}],
            max_pos=3,
            R=R,
        )
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert np.sum(result["weights"] > 1e-6) <= 3

    def test_mad_objective(self):
        """MAD objective should work with R data."""
        moments, constraints, R = _make_correlated_moments(n=5)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "MAD"}],
            max_pos=2,
            R=R,
        )
        assert result["status"] == "optimal"
        assert result["weights"] is not None
        assert np.sum(result["weights"] > 1e-6) <= 2

    def test_unsupported_risk_returns_none_weights(self):
        """EVaR requires exponential cone; HiGHS cannot solve it → None weights."""
        moments, constraints, _ = _make_correlated_moments(n=5)
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "EVaR"}],
            max_pos=2,
        )
        # Must signal 'unsupported_risk' without raising, so caller can fall back
        assert result["weights"] is None
        assert result["status"] == "unsupported_risk"

    def test_milp_better_than_heuristic_correlated_assets(self):
        """
        Construct a case where top-2 relaxed weights pick two highly correlated
        assets, while the globally optimal 2-asset subset has lower variance.
        MILP must find the lower-variance solution.
        """
        # Assets 0 & 1 are nearly identical (ρ ≈ 0.95).
        # Asset 2 is uncorrelated with both.
        # Optimal 2-asset subset: {0, 2} or {1, 2}, NOT {0, 1}.
        sigma = np.array([
            [0.04,  0.038,  0.001],
            [0.038, 0.04,   0.001],
            [0.001, 0.001,  0.04 ],
        ], dtype=float)
        mu = np.array([0.01, 0.01, 0.005])
        moments = {"mu": mu.reshape(-1, 1), "sigma": sigma}
        constraints = {
            "min_sum": 1.0, "max_sum": 1.0,
            "min": pd.Series([0.0, 0.0, 0.0]),
            "max": pd.Series([1.0, 1.0, 1.0]),
        }
        result = solve_milp_cardinality(
            moments, constraints,
            [{"type": "risk", "name": "StdDev"}],
            max_pos=2,
        )
        assert result["weights"] is not None
        w = result["weights"]

        # MILP variance
        var_milp = float(w @ sigma @ w)

        # Two-step heuristic: solve relaxed, pick top-2, solve again
        # The relaxed equal-weight solution (approx 1/3 each) gives top-2 = {0,1}
        w_heuristic = np.array([0.5, 0.5, 0.0])
        var_heuristic = float(w_heuristic @ sigma @ w_heuristic)

        # MILP must find strictly lower or equal variance
        assert var_milp <= var_heuristic + 1e-8, (
            f"MILP var={var_milp:.6f} > heuristic var={var_heuristic:.6f}"
        )
        # And the selected pair should include asset 2 (the uncorrelated one)
        assert w[2] > 1e-6, "MILP should select the uncorrelated asset 2"


# ── Integration test: max_pos via optimize_portfolio ─────────────────────────

class TestMaxPosIntegration:

    def test_optimize_portfolio_max_pos_strict(self):
        """optimize_portfolio with max_pos=3 must return ≤3 non-zero positions."""
        from pyfolioanalytics.optimize import optimize_portfolio
        from pyfolioanalytics.portfolio import Portfolio

        rng = np.random.default_rng(7)
        R = pd.DataFrame(
            rng.normal(0.001, 0.02, (200, 8)),
            columns=[f"A{i}" for i in range(8)],
        )
        port = Portfolio(list(R.columns))
        port.add_constraint("full_investment")
        port.add_constraint("long_only")
        port.add_constraint("position_limit", max_pos=3)
        port.add_objective("risk", name="StdDev")

        res = optimize_portfolio(R, port, optimize_method="ROI")
        assert res["weights"] is not None
        n_pos = int(np.sum(res["weights"].values > 1e-6))
        assert n_pos <= 3, f"max_pos=3 violated: got {n_pos} positions"

    def test_optimize_portfolio_max_pos_evar_fallback(self):
        """EVaR with max_pos falls back gracefully to two-step heuristic."""
        from pyfolioanalytics.optimize import optimize_portfolio
        from pyfolioanalytics.portfolio import Portfolio

        rng = np.random.default_rng(9)
        R = pd.DataFrame(
            rng.normal(0.001, 0.02, (150, 6)),
            columns=[f"B{i}" for i in range(6)],
        )
        port = Portfolio(list(R.columns))
        port.add_constraint("full_investment")
        port.add_constraint("long_only")
        port.add_constraint("position_limit", max_pos=3)
        port.add_objective("risk", name="EVaR")

        # Should not raise; may or may not satisfy max_pos exactly (heuristic)
        res = optimize_portfolio(R, port, optimize_method="ROI")
        assert res is not None  # at minimum, no crash
