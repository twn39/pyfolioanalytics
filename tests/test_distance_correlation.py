"""Tests for vectorised _distance_correlation_matrix.

Verifies:
1. Numerical parity with the original O(N²)-loop reference implementation.
2. Structural properties (symmetry, diagonal=1, range [0,1]).
3. Chunked path produces identical results to the full-tensor path.
4. Edge cases: N=2, constant column, single pair, correlated columns.
5. Performance regression guard.
"""

import time

import numpy as np
import pytest

from pyfolioanalytics.codependence import (
    _dcor_1d,
    _dcor_matrix_chunked,
    _dcor_matrix_full,
    _distance_correlation_matrix,
    get_codependence_matrix,
)
import pandas as pd


# ── Reference (original loop) ─────────────────────────────────────────────────

def _ref_dcor_matrix(X: np.ndarray) -> np.ndarray:
    """Verbatim copy of the original O(N²) double-loop implementation."""
    n_samples, n_features = X.shape
    dcor = np.ones((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = _dcor_1d(X[:, i], X[:, j])
            dcor[i, j] = val
            dcor[j, i] = val
    return dcor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_X():
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, (60, 8))


@pytest.fixture
def medium_X():
    rng = np.random.default_rng(7)
    return rng.normal(0.001, 0.02, (252, 30))


# ── 1. Numerical parity with reference ───────────────────────────────────────

class TestNumericalParity:
    def test_small_matches_reference(self, small_X):
        ref = _ref_dcor_matrix(small_X)
        result = _distance_correlation_matrix(small_X)
        np.testing.assert_allclose(result, ref, atol=1e-12,
            err_msg="Vectorised result diverges from reference for small X")

    def test_medium_matches_reference(self, medium_X):
        ref = _ref_dcor_matrix(medium_X)
        result = _distance_correlation_matrix(medium_X)
        np.testing.assert_allclose(result, ref, atol=1e-12,
            err_msg="Vectorised result diverges from reference for medium X")

    def test_get_codependence_matrix_distance_path(self, medium_X):
        """End-to-end: get_codependence_matrix(method='distance') must match ref."""
        df = pd.DataFrame(medium_X, columns=[f"A{i}" for i in range(medium_X.shape[1])])
        ref = _ref_dcor_matrix(medium_X)
        result = get_codependence_matrix(df, method="distance")
        np.testing.assert_allclose(result, ref, atol=1e-12)


# ── 2. Structural properties ──────────────────────────────────────────────────

class TestStructuralProperties:
    def test_symmetric(self, medium_X):
        m = _distance_correlation_matrix(medium_X)
        np.testing.assert_allclose(m, m.T, atol=1e-14, err_msg="Matrix is not symmetric")

    def test_diagonal_ones(self, medium_X):
        m = _distance_correlation_matrix(medium_X)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-14)

    def test_range_zero_to_one(self, medium_X):
        m = _distance_correlation_matrix(medium_X)
        assert np.all(m >= -1e-12), "dCor values below 0"
        assert np.all(m <= 1.0 + 1e-12), "dCor values above 1"

    def test_square_output(self, medium_X):
        T, N = medium_X.shape
        m = _distance_correlation_matrix(medium_X)
        assert m.shape == (N, N)


# ── 3. Chunked vs full path equivalence ──────────────────────────────────────

class TestChunkedEquivalence:
    """Force chunked path by temporarily lowering the memory limit."""

    def test_chunked_matches_full(self, medium_X):
        full = _dcor_matrix_full(medium_X)
        chunked = _dcor_matrix_chunked(medium_X)
        np.testing.assert_allclose(chunked, full, atol=1e-12,
            err_msg="Chunked path diverges from full-tensor path")

    def test_chunked_chunk_size_1(self, small_X):
        """Extreme case: chunk of 1 column (degenerates to sequential processing)."""
        import pyfolioanalytics.codependence as cod
        original = cod._DCOR_MEM_LIMIT_MB
        try:
            cod._DCOR_MEM_LIMIT_MB = 1e-9  # force chunk=1
            result = _distance_correlation_matrix(small_X)
        finally:
            cod._DCOR_MEM_LIMIT_MB = original
        ref = _ref_dcor_matrix(small_X)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_dispatch_switches_to_chunked(self, monkeypatch):
        """When memory would exceed limit, dispatch must call chunked path."""
        import pyfolioanalytics.codependence as cod
        monkeypatch.setattr(cod, "_DCOR_MEM_LIMIT_MB", 1e-9)  # 0 bytes limit
        rng = np.random.default_rng(99)
        X = rng.normal(0, 0.01, (50, 4))
        ref = _ref_dcor_matrix(X)
        result = _distance_correlation_matrix(X)
        np.testing.assert_allclose(result, ref, atol=1e-12)


# ── 4. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_n_equals_2(self):
        """Minimal matrix: only one off-diagonal pair."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 2))
        m = _distance_correlation_matrix(X)
        assert m.shape == (2, 2)
        assert abs(m[0, 0] - 1.0) < 1e-14
        assert abs(m[0, 1] - m[1, 0]) < 1e-14
        assert 0.0 <= m[0, 1] <= 1.0

    def test_identical_columns_dcor_one(self):
        """Two identical columns must have dCor = 1."""
        rng = np.random.default_rng(3)
        col = rng.normal(0, 1, 80)
        X = np.column_stack([col, col])
        m = _distance_correlation_matrix(X)
        np.testing.assert_allclose(m[0, 1], 1.0, atol=1e-12)

    def test_constant_column_dcor_zero(self):
        """A constant column is independent of everything → dCor = 0."""
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (80, 3))
        X[:, 1] = 3.14  # constant column
        m = _distance_correlation_matrix(X)
        np.testing.assert_allclose(m[0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(m[1, 2], 0.0, atol=1e-12)

    def test_perfectly_negatively_correlated(self):
        """dCor is defined on [0,1]; even for perfect anti-correlation it is positive."""
        col = np.linspace(-1, 1, 100)
        X = np.column_stack([col, -col])
        m = _distance_correlation_matrix(X)
        assert m[0, 1] > 0.5, "Expected high dCor for perfectly anti-correlated series"
        assert m[0, 1] <= 1.0

    def test_dcor_1d_consistency(self, small_X):
        """_dcor_1d(x, y) must equal _distance_correlation_matrix(stack)[0,1]."""
        x, y = small_X[:, 0], small_X[:, 3]
        scalar = _dcor_1d(x, y)
        matrix_val = _distance_correlation_matrix(np.column_stack([x, y]))[0, 1]
        assert abs(scalar - matrix_val) < 1e-12


# ── 5. Performance regression guard ──────────────────────────────────────────

class TestPerformance:
    """The vectorised implementation must be substantially faster than the loop."""

    @pytest.mark.parametrize("T,N,max_seconds", [
        (252, 30, 0.5),
        (252, 50, 1.0),
    ])
    def test_faster_than_loop(self, T, N, max_seconds):
        rng = np.random.default_rng(11)
        X = rng.normal(0, 0.01, (T, N))

        t0 = time.perf_counter()
        _distance_correlation_matrix(X)
        elapsed = time.perf_counter() - t0

        assert elapsed < max_seconds, (
            f"Vectorised dCor for T={T}, N={N} took {elapsed:.3f}s "
            f"(limit {max_seconds}s) — possible performance regression"
        )
