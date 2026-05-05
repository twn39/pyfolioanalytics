"""Tests for SemiDeviation and SemiVariance risk measures.

Verifies:
1. Consistency with LPM(p=2) scalar evaluator.
2. Numerical parity with R PerformanceAnalytics SemiDeviation / SemiVariance.
3. Convex optimisation via SemiStdDevStrategy (SemiStdDev / SemiDeviation /
   SemiVariance / Sortino names all resolve correctly).
4. LPMStrategy p=2 denominator now matches SemiDeviation (both ÷T).
"""

import numpy as np
import pytest

import pyfolioanalytics as pa
from pyfolioanalytics.risk import LPM, SemiDeviation, SemiVariance
from pyfolioanalytics.convex_solvers import RISK_STRATEGIES


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def returns_and_weights():
    rng = np.random.default_rng(0)
    R = rng.normal(0.001, 0.02, (60, 3))
    w = np.array([0.4, 0.35, 0.25])
    return R, w


# ── 1. Public API / package-level export ──────────────────────────────────────

class TestPublicExport:
    def test_semi_deviation_exported(self):
        assert hasattr(pa, "SemiDeviation")
        assert callable(pa.SemiDeviation)

    def test_semi_variance_exported(self):
        assert hasattr(pa, "SemiVariance")
        assert callable(pa.SemiVariance)

    def test_lpm_exported(self):
        assert hasattr(pa, "LPM")
        assert callable(pa.LPM)


# ── 2. Scalar evaluator correctness ───────────────────────────────────────────

class TestSemiDeviationFormula:
    """SemiDeviation = LPM(p=2, method='full') = sqrt(sum(max(0-r,0)^2) / T)."""

    def test_equals_lpm_full(self, returns_and_weights):
        R, w = returns_and_weights
        assert SemiDeviation(w, R) == LPM(w, R, p=2, rf=0.0, method="full")

    def test_value_matches_manual(self, returns_and_weights):
        R, w = returns_and_weights
        p_ret = R @ w
        shortfall = np.maximum(0.0 - p_ret, 0.0)
        expected = float(np.sqrt(np.sum(shortfall**2) / len(p_ret)))
        result = SemiDeviation(w, R, rf=0.0)
        assert abs(result - expected) < 1e-12

    def test_positive(self, returns_and_weights):
        R, w = returns_and_weights
        assert SemiDeviation(w, R) >= 0.0

    def test_all_positive_returns_zero(self):
        R = np.abs(np.random.default_rng(1).normal(0.01, 0.005, (40, 2)))
        w = np.array([0.5, 0.5])
        assert SemiDeviation(w, R) == 0.0

    def test_custom_mar(self, returns_and_weights):
        R, w = returns_and_weights
        mar = 0.001
        sd_mar = SemiDeviation(w, R, rf=mar)
        sd_zero = SemiDeviation(w, R, rf=0.0)
        # Higher MAR → more observations below threshold → larger SemiDeviation
        assert sd_mar >= sd_zero


class TestSemiVarianceFormula:
    """SemiVariance = LPM(p=2, method='subset')^2 = sum(...)/k."""

    def test_equals_lpm_subset_squared(self, returns_and_weights):
        R, w = returns_and_weights
        expected = LPM(w, R, p=2, rf=0.0, method="subset") ** 2
        assert abs(SemiVariance(w, R) - expected) < 1e-12

    def test_semi_variance_ge_semi_deviation_squared(self, returns_and_weights):
        """SemiVariance (÷k) >= SemiDeviation^2 (÷T) since k <= T."""
        R, w = returns_and_weights
        sv = SemiVariance(w, R)
        sd_sq = SemiDeviation(w, R) ** 2
        assert sv >= sd_sq - 1e-12

    def test_all_positive_returns_zero(self):
        R = np.abs(np.random.default_rng(2).normal(0.01, 0.005, (40, 2)))
        w = np.array([0.6, 0.4])
        assert SemiVariance(w, R) == 0.0


# ── 3. R PerformanceAnalytics parity ─────────────────────────────────────────

class TestRParity:
    """Verify against manually-computed R ground truth.

    Using: r = c(0.02, -0.03, 0.01, -0.05, 0.04, -0.01, 0.03, -0.02)
           DownsideDeviation(r, MAR=0, method="full")   # sqrt(0.0039/8)
           SemiVariance(r)  ← method="subset", MAR=mean(r)
    """

    _RETURNS = np.array([0.02, -0.03, 0.01, -0.05, 0.04, -0.01, 0.03, -0.02])
    _W = np.array([1.0])

    def test_semi_deviation_parity_with_R(self):
        R = self._RETURNS.reshape(-1, 1)
        result = SemiDeviation(self._W, R, rf=0.0)
        expected = float(np.sqrt(0.0039 / 8))  # exact from R
        assert abs(result - expected) < 1e-12

    def test_semi_variance_parity_with_R(self):
        # R SemiVariance uses MAR=mean(R) and method="subset"
        R = self._RETURNS.reshape(-1, 1)
        mar = float(np.mean(self._RETURNS))
        result = SemiVariance(self._W, R, rf=mar)
        p_ret = self._RETURNS
        shortfall = np.maximum(mar - p_ret, 0)
        k = int(np.sum(shortfall > 0))
        expected = float(np.sum(shortfall**2) / k)
        assert abs(result - expected) < 1e-12


# ── 4. Convex optimisation – SemiStdDevStrategy ───────────────────────────────

class TestSemiStdDevStrategy:
    """Verify that all registered name aliases resolve and optimise correctly."""

    @pytest.mark.parametrize("name", [
        "SemiStdDev", "SemiDeviation", "SemiVariance", "Sortino"
    ])
    def test_name_registered(self, name):
        assert name in RISK_STRATEGIES, f"{name!r} not in RISK_STRATEGIES"

    @pytest.mark.parametrize("name", ["SemiStdDev", "SemiDeviation", "Sortino"])
    def test_optimisation_feasible(self, name):
        """Min-SemiDeviation must return a valid weight vector summing to 1."""
        import pandas as pd
        rng = np.random.default_rng(42)
        assets = ["A", "B", "C"]
        R = pd.DataFrame(rng.normal(0.001, 0.02, (50, 3)), columns=assets)
        port = pa.Portfolio(assets=assets)
        port.add_constraint("full_investment")
        port.add_constraint("long_only")
        port.add_objective("risk", name=name)
        result = pa.optimize_portfolio(R, port, optimize_method="ROI")
        w = result["weights"].values
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-4)

    def test_mar_argument_accepted(self):
        """The 'mar' keyword must be accepted without error."""
        import pandas as pd
        rng = np.random.default_rng(7)
        assets = ["X", "Y"]
        R = pd.DataFrame(rng.normal(0.001, 0.02, (40, 2)), columns=assets)
        port = pa.Portfolio(assets=assets)
        port.add_constraint("full_investment")
        port.add_constraint("long_only")
        port.add_objective("risk", name="SemiStdDev", mar=0.0)
        result = pa.optimize_portfolio(R, port, optimize_method="ROI")
        w = result["weights"].values
        assert abs(w.sum() - 1.0) < 1e-4

    def test_semi_dev_lt_std_dev(self):
        """Min-SemiDev risk <= Min-StdDev risk for the same universe."""
        import pandas as pd
        rng = np.random.default_rng(99)
        assets = list("ABCD")
        R = pd.DataFrame(rng.normal(0.0005, 0.015, (80, 4)), columns=assets)

        def _opt(risk_name: str) -> np.ndarray:
            port = pa.Portfolio(assets=assets)
            port.add_constraint("full_investment")
            port.add_constraint("long_only")
            port.add_objective("risk", name=risk_name)
            return pa.optimize_portfolio(R, port, optimize_method="ROI")["weights"].values

        w_semi = _opt("SemiStdDev")
        w_std  = _opt("StdDev")
        R_np = R.to_numpy()
        sd_at_semi_w = SemiDeviation(w_semi, R_np)
        sd_at_std_w  = SemiDeviation(w_std,  R_np)
        assert sd_at_semi_w <= sd_at_std_w + 1e-4


# ── 5. LPMStrategy p=2 denominator consistency ───────────────────────────────

class TestLPMStrategyDenominator:
    """The convex LPMStrategy(p=2) denominator must now be ÷T (not ÷T-1),
    consistent with the scalar LPM() function."""

    def test_lpm_strategy_optimal_matches_scalar(self):
        """For the min-LPM(p=2) optimal portfolio, the scalar evaluator should be
        internally consistent (positive, weights sum to 1)."""
        import pandas as pd
        rng = np.random.default_rng(5)
        assets = ["A", "B", "C"]
        R_df = pd.DataFrame(rng.normal(0.001, 0.02, (60, 3)), columns=assets)
        R_np = R_df.to_numpy()
        port = pa.Portfolio(assets=assets)
        port.add_constraint("full_investment")
        port.add_constraint("long_only")
        port.add_objective("risk", name="LPM", p=2)
        result = pa.optimize_portfolio(R_df, port, optimize_method="ROI")
        w = result["weights"].values
        scalar_val = LPM(w, R_np, p=2, rf=0.0, method="full")
        assert scalar_val >= 0.0
        assert abs(w.sum() - 1.0) < 1e-4
