"""Tests for owa_l_moment_crm_weights CRRA branch correctness.

Cross-validates against Riskfolio-Lib's owa_l_moment_crm() which is the
authoritative reference implementation (Cajas, 2024).
"""

import math

import numpy as np
import pytest
from scipy.special import binom

from pyfolioanalytics.risk import owa_l_moment_crm_weights


# ── Reference implementation (verbatim from Riskfolio-Lib OwaWeights.py) ─────

def _riskfolio_owa_l_moment(T: int, k: int) -> np.ndarray:
    """Exact copy of Riskfolio-Lib owa_l_moment() for cross-validation."""
    w = []
    for i in range(1, T + 1):
        a = 0
        for j in range(k):
            a += (-1) ** j * binom(k - 1, j) * binom(i - 1, k - 1 - j) * binom(T - i, j)
        a *= 1 / (k * binom(T, k))
        w.append(a)
    return np.array(w).reshape(-1, 1)


def _riskfolio_crm_crra(T: int, k: int, g: float) -> np.ndarray:
    """Exact copy of Riskfolio-Lib owa_l_moment_crm(..., method='CRRA')."""
    ws = np.empty((T, 0))
    for i in range(2, k + 1):
        w_i = (-1) ** i * _riskfolio_owa_l_moment(T, i)
        ws = np.concatenate([ws, w_i], axis=1)

    phis = []
    e = 1
    for i in range(1, k):
        e *= g + i - 1
        phis.append(e / math.factorial(i + 1))
    phis = np.array(phis)
    phis = phis / np.sum(phis)
    phis = phis.reshape(-1, 1)
    a = ws @ phis

    w = np.zeros_like(a)
    w[0] = a[0]
    for i in range(1, len(a)):
        w[i, 0] = np.max(a[: i + 1, 0])  # cummax — non-decreasing

    return w.flatten()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCRRAMonotonicity:
    """Verify that the CRRA branch produces a non-decreasing OWA weight vector."""

    @pytest.mark.parametrize("T,k,g", [
        (20,  3, 0.5),
        (20,  4, 0.5),
        (50,  4, 0.3),
        (100, 5, 0.7),
        (12,  3, 0.9),
    ])
    def test_non_decreasing(self, T, k, g):
        """OWA weights must be non-decreasing (CRM condition)."""
        w = owa_l_moment_crm_weights(T, k=k, method="CRRA", g=g)
        diffs = np.diff(w)
        assert np.all(diffs >= -1e-10), (
            f"CRRA weights not non-decreasing for T={T}, k={k}, g={g}. "
            f"Min diff: {diffs.min():.4e}"
        )

    def test_first_le_last(self):
        """Bug-check: wrong version had w[0] > w[-1] (non-increasing).
        Correct version must have w[0] <= w[-1]."""
        w = owa_l_moment_crm_weights(20, k=4, method="CRRA", g=0.5)
        assert w[0] <= w[-1], (
            f"w[0]={w[0]:.6f} > w[-1]={w[-1]:.6f} — reversal bug still present"
        )


class TestCRRARiskfolioParty:
    """Numerical parity with Riskfolio-Lib owa_l_moment_crm() reference."""

    @pytest.mark.parametrize("T,k,g", [
        (20, 3, 0.5),
        (20, 4, 0.5),
        (30, 4, 0.3),
        (50, 5, 0.7),
    ])
    def test_matches_riskfolio(self, T, k, g):
        result_py  = owa_l_moment_crm_weights(T, k=k, method="CRRA", g=g)
        result_ref = _riskfolio_crm_crra(T, k=k, g=g)
        np.testing.assert_allclose(
            result_py, result_ref, rtol=1e-10, atol=1e-12,
            err_msg=f"Mismatch vs Riskfolio for T={T}, k={k}, g={g}",
        )

    def test_shape(self):
        """Return shape must be (T,)."""
        T = 25
        w = owa_l_moment_crm_weights(T, k=4, method="CRRA", g=0.5)
        assert w.shape == (T,), f"Expected ({T},), got {w.shape}"


class TestCRRAPhiFormula:
    """Verify the phi (risk-aversion coefficient) computation is correct."""

    def test_phi_normalised(self):
        """After normalisation, phi must sum to 1."""
        # Reconstruct phis manually for k=4, g=0.5
        k, g = 4, 0.5
        phis = []
        e = 1
        for i in range(1, k):
            e *= g + i - 1
            phis.append(e / math.factorial(i + 1))
        phis = np.array(phis) / np.sum(phis)
        assert abs(np.sum(phis) - 1.0) < 1e-12

    @pytest.mark.parametrize("g", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_phi_all_positive(self, g):
        """All phi values must be positive for g in (0, 1)."""
        k = 4
        phis = []
        e = 1
        for i in range(1, k):
            e *= g + i - 1
            phis.append(e / math.factorial(i + 1))
        assert all(p > 0 for p in phis), f"Negative phi for g={g}: {phis}"


class TestOtherMethodsUnchanged:
    """Ensure MSD / ME / MSS methods still produce non-decreasing weights
    (regression guard — CRRA fix must not affect the convex-optimisation path)."""

    @pytest.mark.parametrize("method", ["MSD", "MSS"])
    def test_convex_path_non_decreasing(self, method):
        w = owa_l_moment_crm_weights(20, k=3, method=method)
        assert np.all(np.diff(w) >= -1e-6), (
            f"method={method} weights not non-decreasing after CRRA fix"
        )
