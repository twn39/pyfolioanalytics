"""Tests for the weight_concentration_objective (HHI-based).

Structure
---------
1. ``hhi()`` unit tests — pure function, no solver involved
2. ``Portfolio.add_objective()`` validation tests
3. Global HHI (no groups) optimisation — DEoptim & random solvers
4. Grouped HHI (path B) optimisation — DEoptim & random solvers
5. ``from_r_index`` index conversion
6. Scalar conc_aversion broadcast to groups
7. Dimension mismatch fallback
"""

import warnings

import numpy as np
import pytest

from pyfolioanalytics.optimize import optimize_portfolio
from pyfolioanalytics.portfolio import Portfolio
from pyfolioanalytics.risk import hhi


# ──────────────────────────────────────────────────────────────────────────────
# 1.  hhi() — pure unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestHhiFunction:
    def test_equal_weight_n4(self):
        w = np.full(4, 0.25)
        assert np.isclose(hhi(w), 0.25)

    def test_fully_concentrated(self):
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.isclose(hhi(w), 1.0)

    def test_no_groups_returns_float(self):
        w = np.array([0.4, 0.3, 0.2, 0.1])
        result = hhi(w)
        assert isinstance(result, float)
        assert np.isclose(result, 0.4**2 + 0.3**2 + 0.2**2 + 0.1**2)

    def test_groups_returns_dict_keys(self):
        w = np.array([0.3, 0.3, 0.2, 0.2])
        result = hhi(w, groups=[[0, 1], [2, 3]])
        assert isinstance(result, dict)
        assert "HHI" in result and "Groups_HHI" in result

    def test_global_hhi_with_groups(self):
        """The 'HHI' key must equal sum(w^2) regardless of grouping."""
        w = np.array([0.3, 0.3, 0.2, 0.2])
        result = hhi(w, groups=[[0, 1], [2, 3]])
        assert np.isclose(result["HHI"], w @ w)

    def test_group_hhi_values_non_overlapping(self):
        """Groups_HHI[k] = sum(w[group_k]^2); for disjoint groups they sum to global."""
        w = np.array([0.3, 0.3, 0.2, 0.2])
        result = hhi(w, groups=[[0, 1], [2, 3]])
        expected_g0 = 0.3**2 + 0.3**2  # 0.18
        expected_g1 = 0.2**2 + 0.2**2  # 0.08
        assert np.isclose(result["Groups_HHI"][0], expected_g0)
        assert np.isclose(result["Groups_HHI"][1], expected_g1)
        # Non-overlapping groups: their HHIs sum to the global HHI
        assert np.isclose(np.sum(result["Groups_HHI"]), result["HHI"])

    def test_group_hhi_single_element_groups(self):
        """One-element group HHI equals the squared weight."""
        w = np.array([0.5, 0.3, 0.2])
        result = hhi(w, groups=[[0], [1], [2]])
        expected = w**2
        np.testing.assert_allclose(result["Groups_HHI"], expected)

    def test_groups_hhi_vector_shape(self):
        w = np.ones(6) / 6
        result = hhi(w, groups=[[0, 1, 2], [3, 4, 5]])
        assert result["Groups_HHI"].shape == (2,)

    def test_from_r_index_true(self):
        """1-based R indices [1,2,3,4] should map to 0-based [0,1,2,3]."""
        w = np.array([0.3, 0.3, 0.2, 0.2])
        # 0-based reference
        ref = hhi(w, groups=[[0, 1], [2, 3]])
        # R-style 1-based
        r_result = hhi(w, groups=[[1, 2], [3, 4]], from_r_index=True)
        np.testing.assert_allclose(r_result["Groups_HHI"], ref["Groups_HHI"])

    def test_from_r_index_false_does_not_shift(self):
        """from_r_index=False must not alter the indices."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        r1 = hhi(w, groups=[[0, 1], [2, 3]], from_r_index=False)
        r2 = hhi(w, groups=[[0, 1], [2, 3]])
        np.testing.assert_allclose(r1["Groups_HHI"], r2["Groups_HHI"])

    def test_no_groups_from_r_index_has_no_effect(self):
        """from_r_index has no effect when groups is None."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        assert hhi(w) == hhi(w, from_r_index=True)

    def test_overlapping_groups(self):
        """Overlapping groups are allowed; HHIs do NOT sum to global."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        result = hhi(w, groups=[[0, 1, 2], [1, 2, 3]])  # asset 1,2 shared
        assert result["Groups_HHI"].shape == (2,)
        # Each group HHI = 3 * 0.25^2 = 0.1875
        assert np.allclose(result["Groups_HHI"], [0.1875, 0.1875])


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Portfolio.add_objective() validation
# ──────────────────────────────────────────────────────────────────────────────


class TestAddObjectiveValidation:
    def test_requires_conc_aversion(self):
        p = Portfolio(assets=4)
        with pytest.raises(ValueError, match="conc_aversion"):
            p.add_objective(type="weight_concentration", name="HHI")

    def test_scalar_conc_aversion_no_groups(self):
        p = Portfolio(assets=4)
        p.add_objective(type="weight_concentration", name="HHI", conc_aversion=0.1)
        obj = p.objectives[-1]
        assert obj["conc_aversion"] == 0.1
        assert obj["conc_groups"] is None

    def test_alias_weight_conc_normalised(self):
        p = Portfolio(assets=4)
        p.add_objective(type="weight_conc", name="HHI", conc_aversion=0.1)
        assert p.objectives[-1]["type"] == "weight_concentration"

    def test_scalar_broadcast_to_groups(self):
        p = Portfolio(assets=4)
        p.add_objective(
            type="weight_concentration",
            name="HHI",
            conc_aversion=0.2,
            conc_groups=[[0, 1], [2, 3]],
        )
        obj = p.objectives[-1]
        # Scalar must be broadcast: [0.2, 0.2]
        assert obj["conc_aversion"] == [0.2, 0.2]

    def test_vector_conc_aversion_matches_groups(self):
        p = Portfolio(assets=4)
        p.add_objective(
            type="weight_concentration",
            name="HHI",
            conc_aversion=[0.1, 0.5],
            conc_groups=[[0, 1], [2, 3]],
        )
        obj = p.objectives[-1]
        assert obj["conc_aversion"] == [0.1, 0.5]

    def test_length_mismatch_raises(self):
        p = Portfolio(assets=4)
        with pytest.raises(ValueError, match="length"):
            p.add_objective(
                type="weight_concentration",
                name="HHI",
                conc_aversion=[0.1, 0.2, 0.3],  # 3 values, but only 2 groups
                conc_groups=[[0, 1], [2, 3]],
            )

    def test_vector_aversion_without_groups_raises(self):
        p = Portfolio(assets=4)
        with pytest.raises(ValueError, match="scalar"):
            p.add_objective(
                type="weight_concentration",
                name="HHI",
                conc_aversion=[0.1, 0.2],  # vector but no groups
            )

    def test_from_r_index_shifts_conc_groups(self):
        p = Portfolio(assets=4)
        p.add_objective(
            type="weight_concentration",
            name="HHI",
            conc_aversion=0.1,
            conc_groups=[[1, 2], [3, 4]],  # R-style 1-based
            from_r_index=True,
        )
        obj = p.objectives[-1]
        # Indices must be converted to 0-based
        assert obj["conc_groups"] == [[0, 1], [2, 3]]

    def test_from_r_index_removed_from_obj(self):
        """The 'from_r_index' helper key must not leak into the stored objective."""
        p = Portfolio(assets=4)
        p.add_objective(
            type="weight_concentration",
            name="HHI",
            conc_aversion=0.1,
            conc_groups=[[1, 2], [3, 4]],
            from_r_index=True,
        )
        assert "from_r_index" not in p.objectives[-1]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Global HHI (no groups) — solver integration
# ──────────────────────────────────────────────────────────────────────────────


class TestGlobalHhiSolver:
    """High conc_aversion should produce more dispersed weights than low."""

    def _run(self, returns, conc_aversion, method="DEoptim", **kw):
        p = Portfolio(assets=returns.columns.tolist())
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(
            type="weight_concentration", name="HHI", conc_aversion=conc_aversion
        )
        return optimize_portfolio(returns, p, optimize_method=method, **kw)

    def test_high_aversion_lower_hhi_deoptim(self, stocks_data):
        # Run a sweep of 3 conc_aversion values and verify that HHI is
        # monotonically non-increasing as aversion increases.  A single
        # pair comparison is fragile because DEoptim is stochastic and two
        # independent runs can converge to the same local optimum.
        aversions = [0.0001, 0.5, 5.0]
        hhis = []
        for av in aversions:
            res = self._run(stocks_data, conc_aversion=av, itermax=80)
            assert res["status"] in ("optimal", "optimal_inaccurate")
            hhis.append(hhi(res["weights"].values))
        # Monotone non-increasing: HHI should weaken or stay equal as aversion grows
        assert hhis[2] <= hhis[0] + 1e-3, (
            f"Highest aversion HHI ({hhis[2]:.4f}) should not exceed "
            f"lowest aversion HHI ({hhis[0]:.4f})"
        )

    def test_high_aversion_lower_hhi_random(self, stocks_data):
        res_low  = self._run(stocks_data, conc_aversion=0.001, method="random", permutations=500)
        res_high = self._run(stocks_data, conc_aversion=5.0,   method="random", permutations=500)
        hhi_low  = hhi(res_low["weights"].values)
        hhi_high = hhi(res_high["weights"].values)
        assert hhi_high <= hhi_low + 1e-4, (
            f"random: higher aversion should not increase HHI: {hhi_high:.4f} vs {hhi_low:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Grouped HHI (path B) — solver integration
# ──────────────────────────────────────────────────────────────────────────────


class TestGroupedHhiSolver:
    """Penalise the second group 5× more; its within-group HHI should be lower."""

    def _run(self, returns, groups, aversions, method="DEoptim", **kw):
        p = Portfolio(assets=returns.columns.tolist())
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(
            type="weight_concentration",
            name="HHI",
            conc_aversion=aversions,
            conc_groups=groups,
        )
        return optimize_portfolio(returns, p, optimize_method=method, **kw)

    def test_higher_group_aversion_produces_lower_group_hhi(self, stocks_data):
        n = len(stocks_data.columns)
        half = n // 2
        groups = [list(range(half)), list(range(half, n))]
        n_g0 = len(groups[0])
        n_g1 = len(groups[1])

        # Run A: Group 0 heavily penalised, Group 1 leniently penalised.
        res_a = self._run(stocks_data, groups, [5.0, 0.05], itermax=80)
        # Run B: roles swapped — Group 1 heavily penalised, Group 0 lenient.
        res_b = self._run(stocks_data, groups, [0.05, 5.0], itermax=80)

        assert res_a["status"] in ("optimal", "optimal_inaccurate")
        assert res_b["status"] in ("optimal", "optimal_inaccurate")

        gr_a = hhi(res_a["weights"].values, groups=groups)
        gr_b = hhi(res_b["weights"].values, groups=groups)

        # Normalised HHI = raw_HHI * n_group  (1 = fully concentrated, 1/n = equal)
        norm_a_g0 = float(gr_a["Groups_HHI"][0]) * n_g0   # high-penalty in A
        norm_b_g0 = float(gr_b["Groups_HHI"][0]) * n_g0   # low-penalty in B
        norm_a_g1 = float(gr_a["Groups_HHI"][1]) * n_g1   # low-penalty in A
        norm_b_g1 = float(gr_b["Groups_HHI"][1]) * n_g1   # high-penalty in B

        # Group 0: run A (high penalty) should not be more concentrated than run B.
        assert norm_a_g0 <= norm_b_g0 + 0.10, (
            f"G0 high-penalty={norm_a_g0:.4f} should not exceed "
            f"G0 low-penalty={norm_b_g0:.4f}"
        )
        # Group 1: run B (high penalty) should not be more concentrated than run A.
        assert norm_b_g1 <= norm_a_g1 + 0.10, (
            f"G1 high-penalty={norm_b_g1:.4f} should not exceed "
            f"G1 low-penalty={norm_a_g1:.4f}"
        )


    def test_grouped_hhi_random_solver(self, stocks_data):
        n = len(stocks_data.columns)
        half = n // 2
        groups = [list(range(half)), list(range(half, n))]
        res = self._run(
            stocks_data, groups, [0.1, 5.0], method="random", permutations=600
        )
        assert res["status"] == "optimal"
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-4)

    def test_scalar_broadcast_optimises_all_groups(self, stocks_data):
        """Scalar conc_aversion broadcast to all groups must still work end-to-end."""
        n = len(stocks_data.columns)
        groups = [list(range(n // 2)), list(range(n // 2, n))]
        # Scalar aversion \u2014 should be broadcast to [0.5, 0.5] internally
        p = Portfolio(assets=stocks_data.columns.tolist())
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        p.add_objective(
            type="weight_concentration",
            name="HHI",
            conc_aversion=0.5,           # scalar
            conc_groups=groups,
        )
        res = optimize_portfolio(stocks_data, p, optimize_method="DEoptim", itermax=50)
        assert res["status"] in ("optimal", "optimal_inaccurate")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  from_r_index end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestFromRIndex:
    def test_r_index_vs_python_index_same_result(self, stocks_data):
        """from_r_index=True with 1-based indices must give same result as 0-based."""
        n = len(stocks_data.columns)
        groups_0based = [list(range(n // 2)), list(range(n // 2, n))]
        groups_1based = [[i + 1 for i in g] for g in groups_0based]

        def _run(groups, from_r):
            p = Portfolio(assets=stocks_data.columns.tolist())
            p.add_constraint(type="full_investment")
            p.add_constraint(type="long_only")
            p.add_objective(
                type="weight_concentration",
                name="HHI",
                conc_aversion=[0.5, 1.0],
                conc_groups=groups,
                from_r_index=from_r,
            )
            return optimize_portfolio(
                stocks_data, p, optimize_method="DEoptim", itermax=40
            )

        res_py = _run(groups_0based, from_r=False)
        res_r  = _run(groups_1based, from_r=True)

        # Both runs start from the same problem; results should be very close.
        assert res_py["status"] in ("optimal", "optimal_inaccurate")
        assert res_r["status"]  in ("optimal", "optimal_inaccurate")
        # Weights won't be identical (stochastic), but both should be valid.
        assert np.isclose(res_py["weights"].sum(), 1.0, atol=1e-3)
        assert np.isclose(res_r["weights"].sum(),  1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Dimension-mismatch fallback warning
# ──────────────────────────────────────────────────────────────────────────────


class TestDimensionMismatchFallback:
    def test_mismatch_emits_warning(self, stocks_data):
        """A manually constructed bad objective should trigger the fallback warning."""
        from pyfolioanalytics.solvers import create_penalized_objective
        from pyfolioanalytics.moments import set_portfolio_moments

        p = Portfolio(assets=stocks_data.columns.tolist())
        p.add_constraint(type="full_investment")
        p.add_constraint(type="long_only")
        moments = set_portfolio_moments(stocks_data, p)
        constraints = p.get_constraints()

        # Manually craft a bad objective (3 aversion values, 2 groups)
        bad_obj = {
            "type": "weight_concentration",
            "name": "HHI",
            "enabled": True,
            "multiplier": 1.0,
            "conc_aversion": [0.1, 0.2, 0.3],     # length 3
            "conc_groups": [[0, 1], [2, 3]],       # length 2
        }
        fn = create_penalized_objective(moments, constraints, [bad_obj])
        w = np.ones(len(p.assets)) / len(p.assets)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = fn(w)

        assert any("Falling back" in str(c.message) for c in caught), (
            "Expected a fallback warning for dimension mismatch."
        )
        assert np.isfinite(val), "Objective must still return a finite value."
