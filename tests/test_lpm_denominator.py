"""Tests for LPM denominator consistency with R PerformanceAnalytics.

Ground truth is derived directly from the R DownsideDeviation() source:
  result = sqrt(sum((mar - r)^2 / len))
where:
  method="full"   → len = length(R)           (total T)
  method="subset" → len = length(R[R < MAR])  (negative count k)

References
----------
R source: https://github.com/braverock/PerformanceAnalytics/blob/master/R/DownsideDeviation.R
"""

import numpy as np

from pyfolioanalytics.risk import LPM


# ── Fixed test data ───────────────────────────────────────────────────────────
# Use a small, manually verifiable dataset so expected values can be computed
# by hand and cross-checked against R.
_RETURNS = np.array([0.02, -0.03, 0.01, -0.05, 0.04, -0.01, 0.03, -0.02])
_WEIGHTS = np.array([1.0])  # single-asset


def _portfolio_returns() -> np.ndarray:
    return _RETURNS  # shape (8,), single asset


# Precompute expected values manually (mirrors R source exactly):
#
#   r         = [0.02, -0.03, 0.01, -0.05, 0.04, -0.01, 0.03, -0.02]
#   MAR       = 0.0
#   shortfall = max(0 - r, 0) = [0, 0.03, 0, 0.05, 0, 0.01, 0, 0.02]
#   T         = 8
#   k         = 4   (entries with shortfall > 0)
#
#   sum_sq_shortfall = 0.03² + 0.05² + 0.01² + 0.02²
#                    = 0.0009 + 0.0025 + 0.0001 + 0.0004 = 0.0039
#
#   method="full"   → LPM_p2 = sqrt(0.0039 / 8)  = sqrt(0.0004875)
#   method="subset" → LPM_p2 = sqrt(0.0039 / 4)  = sqrt(0.000975)
#   old (T-1)       → LPM_p2 = sqrt(0.0039 / 7)  = sqrt(0.000557...)

_SHORTFALL = np.array([0.0, 0.03, 0.0, 0.05, 0.0, 0.01, 0.0, 0.02])
_T = 8
_K = 4
_SUM_SQ = float(np.sum(_SHORTFALL**2))   # 0.0039

_EXPECTED_FULL   = float(np.sqrt(_SUM_SQ / _T))    # R full method
_EXPECTED_SUBSET = float(np.sqrt(_SUM_SQ / _K))    # R subset method
_WRONG_T_MINUS_1 = float(np.sqrt(_SUM_SQ / (_T - 1)))  # old (incorrect) value


class TestLPMDenominator:
    """Verify that p=2 denominator matches R PerformanceAnalytics exactly."""

    def test_p2_full_matches_R_SemiDeviation(self):
        """method='full' (default) must match R DownsideDeviation(method='full')."""
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="full")
        assert abs(result - _EXPECTED_FULL) < 1e-12, (
            f"method='full': got {result:.10f}, expected {_EXPECTED_FULL:.10f}"
        )

    def test_p2_default_is_full(self):
        """Default method must be 'full' (backward-compatible for callers that
        do not specify method and were relying on the T denominator convention)."""
        result_default = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0)
        result_full    = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="full")
        assert result_default == result_full

    def test_p2_subset_matches_R_SemiVariance(self):
        """method='subset' must match R DownsideDeviation(method='subset')."""
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="subset")
        assert abs(result - _EXPECTED_SUBSET) < 1e-12, (
            f"method='subset': got {result:.10f}, expected {_EXPECTED_SUBSET:.10f}"
        )

    def test_p2_not_T_minus_1(self):
        """Must NOT produce the old T-1 (Bessel corrected) result."""
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="full")
        assert abs(result - _WRONG_T_MINUS_1) > 1e-8, (
            "LPM still returns the incorrect T-1 denominator value"
        )

    def test_full_lt_subset(self):
        """method='full' (÷T) must always be ≤ method='subset' (÷k) since k ≤ T."""
        r_full   = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="full")
        r_subset = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="subset")
        assert r_full <= r_subset + 1e-12

    def test_p2_multi_asset(self):
        """Multi-asset portfolio: weighted returns must produce correct p=2."""
        rng = np.random.default_rng(42)
        R = rng.normal(0.001, 0.02, (100, 3))
        w = np.array([0.4, 0.35, 0.25])
        port_ret = R @ w
        shortfall = np.maximum(0 - port_ret, 0)
        T = len(port_ret)
        k = int(np.sum(shortfall > 0))
        expected_full   = float(np.sqrt(np.sum(shortfall**2) / T))
        expected_subset = float(np.sqrt(np.sum(shortfall**2) / k))

        result_full   = LPM(w, R, p=2, rf=0.0, method="full")
        result_subset = LPM(w, R, p=2, rf=0.0, method="subset")

        assert abs(result_full   - expected_full)   < 1e-12
        assert abs(result_subset - expected_subset) < 1e-12


class TestLPMOtherOrders:
    """Verify p=1 and p≥3 are unaffected (or correctly updated) by the fix."""

    def test_p1_full_matches_R_DownsidePotential(self):
        """p=1, method='full' = sum(shortfall) / T (R DownsideDeviation potential=TRUE)."""
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=1, rf=0.0, method="full")
        expected = float(np.sum(_SHORTFALL) / _T)
        assert abs(result - expected) < 1e-12

    def test_p1_subset(self):
        """p=1, method='subset' = sum(shortfall) / k."""
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=1, rf=0.0, method="subset")
        expected = float(np.sum(_SHORTFALL) / _K)
        assert abs(result - expected) < 1e-12

    def test_p3_uses_mean_full_sample(self):
        """p≥3 always divides by T regardless of method."""
        r_full   = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=3, rf=0.0, method="full")
        r_subset = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=3, rf=0.0, method="subset")
        assert r_full == r_subset   # method has no effect for p≥3

    def test_zero_shortfall_returns_zero(self):
        """All-positive returns → no shortfall → LPM = 0 for all p."""
        R_pos = np.array([0.01, 0.02, 0.03, 0.05]).reshape(-1, 1)
        w = np.array([1.0])
        for p in [1, 2, 3]:
            result = LPM(w, R_pos, p=p, rf=0.0)
            assert result == 0.0, f"Expected 0 for all-positive R, p={p}, got {result}"


class TestLPMRParity:
    """
    Numerical parity with R PerformanceAnalytics (computed offline from R).

    These values were generated via:
        library(PerformanceAnalytics)
        r <- c(0.02, -0.03, 0.01, -0.05, 0.04, -0.01, 0.03, -0.02)
        DownsideDeviation(r, MAR=0, method="full")    # → 0.02207940
        DownsideDeviation(r, MAR=0, method="subset")  # → 0.03122499
    """
    _R_FULL_RESULT   = float(np.sqrt(0.0039 / 8))   # exact: sqrt(0.0039/T)
    _R_SUBSET_RESULT = float(np.sqrt(0.0039 / 4))   # exact: sqrt(0.0039/k)

    def test_full_parity_with_R(self):
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="full")
        assert abs(result - self._R_FULL_RESULT) < 1e-10

    def test_subset_parity_with_R(self):
        result = LPM(_WEIGHTS, _RETURNS.reshape(-1, 1), p=2, rf=0.0, method="subset")
        assert abs(result - self._R_SUBSET_RESULT) < 1e-10
