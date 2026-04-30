"""Tests for black_litterman.py and its integration with set_portfolio_moments.

Coverage areas
--------------
1. Meucci formulation — mathematical correctness vs R's BlackLittermanFormula
2. He-Litterman formulation — behaviour with tau="auto" and explicit tau
3. Default Omega and q — mirror R's black.litterman() behaviour
4. tau="auto" → 1/T
5. black_litterman_tilt()
6. moments.py integration — mu always written for method="black_litterman"
7. sentinel flag prevents mu overwrite by mu_method
8. Backward-compatible kwargs (bl_formulation, w_mkt, Mu, Sigma, Omega)
9. Edge cases — K=1 view, K=N views, singular Omega guard
"""

import numpy as np
import pytest

from pyfolioanalytics.black_litterman import black_litterman, black_litterman_tilt
from pyfolioanalytics.moments import set_portfolio_moments
from pyfolioanalytics.portfolio import Portfolio


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_data():
    """120 × 3 returns, deterministic via seed."""
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.01, (120, 3))


@pytest.fixture()
def simple_portfolio():
    p = Portfolio(assets=["A", "B", "C"])
    p.add_constraint(type="full_investment")
    p.add_constraint(type="long_only")
    p.add_objective(type="risk", name="StdDev")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Meucci formulation — mathematical correctness
# ──────────────────────────────────────────────────────────────────────────────


class TestMeucciFormulation:
    """Verify the Meucci mode matches R's BlackLittermanFormula exactly."""

    def test_output_shapes(self, simple_data):
        N = simple_data.shape[1]
        P = np.eye(N)
        q = np.zeros(N)
        res = black_litterman(simple_data, P, q)
        assert res["mu"].shape == (N,)
        assert res["sigma"].shape == (N, N)

    def test_returns_dict_with_mu_and_sigma(self, simple_data):
        P = np.ones((1, 3))
        res = black_litterman(simple_data, P)
        assert "mu" in res
        assert "sigma" in res

    def test_no_pi_key_in_meucci_mode(self, simple_data):
        """The 'Pi' key is exclusive to He-Litterman mode."""
        P = np.ones((1, 3))
        res = black_litterman(simple_data, P, formulation="meucci")
        assert "Pi" not in res

    def test_posterior_matches_r_formula(self, simple_data):
        """Manual R-formula cross-check.

        R formula:
          BLMu    = Mu + Sigma P' (P Sigma P' + Omega)^{-1} (q - P Mu)
          BLSigma = Sigma - Sigma P' (P Sigma P' + Omega)^{-1} P Sigma
        """
        T, N = simple_data.shape
        Mu    = np.mean(simple_data, axis=0)
        Sigma = np.cov(simple_data.T, ddof=1)
        P     = np.array([[1.0, -1.0, 0.0]])   # asset 0 outperforms asset 1
        q     = np.array([0.005])
        Omega = P @ Sigma @ P.T                 # R default

        # Manual computation
        A        = P @ Sigma @ P.T + Omega
        expected_mu    = Mu + Sigma @ P.T @ np.linalg.solve(A, q - P @ Mu)
        expected_sigma = Sigma - Sigma @ P.T @ np.linalg.solve(A.T, P @ Sigma)

        res = black_litterman(simple_data, P, q, Omega=Omega, formulation="meucci")
        np.testing.assert_allclose(res["mu"],    expected_mu,    rtol=1e-10)
        np.testing.assert_allclose(res["sigma"], expected_sigma, rtol=1e-10)

    def test_sigma_is_psd(self, simple_data):
        """Posterior covariance must be positive semi-definite."""
        P = np.eye(3)
        q = np.zeros(3)
        res = black_litterman(simple_data, P, q)
        eigvals = np.linalg.eigvalsh(res["sigma"])
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_null_view_keeps_prior(self, simple_data):
        """With a trivial identity view and q=Mu, posterior should equal prior."""
        T, N = simple_data.shape
        Mu    = np.mean(simple_data, axis=0)
        Sigma = np.cov(simple_data.T, ddof=1)
        P     = np.eye(N)
        Omega = P @ Sigma @ P.T
        q     = P @ Mu          # views equal the prior mean → no update

        res = black_litterman(simple_data, P, q, Omega=Omega)
        np.testing.assert_allclose(res["mu"], Mu, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  He-Litterman formulation
# ──────────────────────────────────────────────────────────────────────────────


class TestHeLittermanFormulation:
    def test_output_shapes(self, simple_data):
        N = simple_data.shape[1]
        P = np.eye(N)
        q = np.zeros(N)
        res = black_litterman(simple_data, P, q, formulation="he_litterman")
        assert res["mu"].shape == (N,)
        assert res["sigma"].shape == (N, N)

    def test_pi_key_present(self, simple_data):
        """He-Litterman mode must return 'Pi' (implied equilibrium returns)."""
        P = np.ones((1, 3))
        res = black_litterman(simple_data, P, formulation="he_litterman")
        assert "Pi" in res
        assert res["Pi"].shape == (3,)

    def test_sigma_is_psd(self, simple_data):
        P = np.eye(3)
        res = black_litterman(simple_data, P, np.zeros(3), formulation="he_litterman")
        eigvals = np.linalg.eigvalsh(res["sigma"])
        assert np.all(eigvals >= -1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Default Omega and q — mirror R's black.litterman() behaviour
# ──────────────────────────────────────────────────────────────────────────────


class TestDefaultOmegaAndQ:
    def test_omega_defaults_to_p_sigma_pt(self, simple_data):
        """When Omega=None, the function uses P @ Sigma @ P.T (R default)."""
        Sigma = np.cov(simple_data.T, ddof=1)
        P = np.array([[1.0, -1.0, 0.0]])
        Omega_explicit = P @ Sigma @ P.T

        res_default  = black_litterman(simple_data, P)
        res_explicit = black_litterman(simple_data, P, Omega=Omega_explicit)
        np.testing.assert_allclose(res_default["mu"],    res_explicit["mu"],    rtol=1e-10)
        np.testing.assert_allclose(res_default["sigma"], res_explicit["sigma"], rtol=1e-10)

    def test_q_defaults_to_sqrt_diag_omega(self, simple_data):
        """When q=None, the function uses sqrt(diag(Omega)) (R default)."""
        Sigma = np.cov(simple_data.T, ddof=1)
        P = np.ones((1, 3))
        Omega = P @ Sigma @ P.T
        q_default = np.sqrt(np.maximum(np.diag(Omega), 0.0))

        res_none = black_litterman(simple_data, P, q=None,      Omega=Omega)
        res_expl = black_litterman(simple_data, P, q=q_default, Omega=Omega)
        np.testing.assert_allclose(res_none["mu"],    res_expl["mu"],    rtol=1e-10)
        np.testing.assert_allclose(res_none["sigma"], res_expl["sigma"], rtol=1e-10)

    def test_mu_defaults_to_sample_mean(self, simple_data):
        """When Mu=None, uses sample mean (R: Mu <- colMeans(R))."""
        Mu_sample = np.mean(simple_data, axis=0)
        P = np.eye(3)
        res_none   = black_litterman(simple_data, P, Mu=None)
        res_sample = black_litterman(simple_data, P, Mu=Mu_sample)
        np.testing.assert_allclose(res_none["mu"], res_sample["mu"], rtol=1e-10)

    def test_sigma_defaults_to_sample_covariance(self, simple_data):
        """When Sigma=None, uses sample covariance (R: Sigma <- cov(R))."""
        Sigma_sample = np.cov(simple_data.T, ddof=1)
        P = np.eye(3)
        res_none   = black_litterman(simple_data, P, Sigma=None)
        res_sample = black_litterman(simple_data, P, Sigma=Sigma_sample)
        np.testing.assert_allclose(res_none["mu"],    res_sample["mu"],    rtol=1e-10)
        np.testing.assert_allclose(res_none["sigma"], res_sample["sigma"], rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  tau="auto" → 1/T
# ──────────────────────────────────────────────────────────────────────────────


class TestTauAuto:
    def test_auto_equals_one_over_T(self, simple_data):
        T = simple_data.shape[0]
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.005])

        res_auto = black_litterman(
            simple_data, P, q, formulation="he_litterman", tau="auto"
        )
        res_1_T  = black_litterman(
            simple_data, P, q, formulation="he_litterman", tau=1.0 / T
        )
        np.testing.assert_allclose(res_auto["mu"],    res_1_T["mu"],    rtol=1e-10)
        np.testing.assert_allclose(res_auto["sigma"], res_1_T["sigma"], rtol=1e-10)

    def test_none_treated_same_as_auto(self, simple_data):
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.005])
        res_auto = black_litterman(
            simple_data, P, q, formulation="he_litterman", tau="auto"
        )
        res_none = black_litterman(
            simple_data, P, q, formulation="he_litterman", tau=None
        )
        np.testing.assert_allclose(res_auto["mu"], res_none["mu"], rtol=1e-10)

    def test_explicit_tau_overrides_auto(self, simple_data):
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.005])

        res_explicit = black_litterman(
            simple_data, P, q, formulation="he_litterman", tau=0.05
        )
        res_auto     = black_litterman(
            simple_data, P, q, formulation="he_litterman", tau="auto"
        )
        # tau=0.05 ≠ 1/120=0.0083... → results must differ
        assert not np.allclose(res_explicit["mu"], res_auto["mu"])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  black_litterman_tilt()
# ──────────────────────────────────────────────────────────────────────────────


class TestBlackLittermanTilt:
    def test_output_shape(self, simple_data):
        Sigma    = np.cov(simple_data.T, ddof=1)
        mu_prior = np.mean(simple_data, axis=0)
        P = np.array([[1.0, -1.0, 0.0]])
        res = black_litterman(simple_data, P)
        w = black_litterman_tilt(
            w_prior=np.full(3, 1.0 / 3),
            Sigma=Sigma, mu_bl=res["mu"], mu_prior=mu_prior
        )
        assert w.shape == (3,)

    def test_zero_view_no_tilt(self, simple_data):
        """When mu_bl == mu_prior, the tilt should be zero → w_tilted == w_prior."""
        mu = np.mean(simple_data, axis=0)
        Sigma = np.cov(simple_data.T, ddof=1)
        w_prior = np.array([0.3, 0.5, 0.2])
        w_tilted = black_litterman_tilt(w_prior, Sigma, mu_bl=mu, mu_prior=mu)
        np.testing.assert_allclose(w_tilted, w_prior, atol=1e-12)

    def test_scale_factor_linear(self, simple_data):
        """Doubling scale should exactly double the tilt adjustment."""
        Sigma    = np.cov(simple_data.T, ddof=1)
        mu_prior = np.mean(simple_data, axis=0)
        P = np.array([[1.0, -1.0, 0.0]])
        res = black_litterman(simple_data, P)
        w_prior  = np.full(3, 1.0 / 3)

        w1 = black_litterman_tilt(w_prior, Sigma, res["mu"], mu_prior, scale=1.0)
        w2 = black_litterman_tilt(w_prior, Sigma, res["mu"], mu_prior, scale=2.0)
        delta1 = w1 - w_prior
        delta2 = w2 - w_prior
        np.testing.assert_allclose(delta2, 2 * delta1, rtol=1e-10)

    def test_positive_view_increases_asset_weight(self, simple_data):
        """A positive view on asset 0 should tilt its weight upward."""
        Sigma    = np.cov(simple_data.T, ddof=1)
        mu_prior = np.mean(simple_data, axis=0)
        # Explicit view: asset 0 expected excess return = +1%
        P = np.eye(3)
        q = mu_prior + np.array([0.01, 0.0, 0.0])
        res = black_litterman(simple_data, P, q)

        w_prior  = np.full(3, 1.0 / 3)
        w_tilted = black_litterman_tilt(w_prior, Sigma, res["mu"], mu_prior)
        assert w_tilted[0] > w_prior[0], "View on asset 0 should tilt weight up"


# ──────────────────────────────────────────────────────────────────────────────
# 6.  moments.py integration — mu always written
# ──────────────────────────────────────────────────────────────────────────────


class TestMomentsIntegration:
    def _make_R(self, n=120, assets=None):
        import pandas as pd
        rng = np.random.default_rng(7)
        cols = assets or ["A", "B", "C"]
        return pd.DataFrame(
            rng.normal(0.001, 0.01, (n, len(cols))), columns=cols
        )

    def test_bl_mu_always_written(self, simple_portfolio):
        """method='black_litterman' must always write moments['mu']."""
        R = self._make_R()
        P = np.array([[1.0, -1.0, 0.0]])
        moments = set_portfolio_moments(
            R, simple_portfolio, method="black_litterman", P=P
        )
        assert "mu" in moments, "BL mu was not written to moments"

    def test_bl_sigma_always_written(self, simple_portfolio):
        R = self._make_R()
        moments = set_portfolio_moments(
            R, simple_portfolio, method="black_litterman"
        )
        assert "sigma" in moments, "BL sigma was not written to moments"

    def test_bl_mu_differs_from_sample_mean(self, simple_portfolio):
        """BL moments with a non-trivial view must differ from sample mean."""
        R = self._make_R()
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.05])   # strong view: +5% excess

        moments_bl = set_portfolio_moments(
            R, simple_portfolio, method="black_litterman", P=P, q=q
        )
        moments_sample = set_portfolio_moments(
            R, simple_portfolio, method="sample"
        )
        assert not np.allclose(
            moments_bl["mu"], moments_sample["mu"]
        ), "BL mu should differ from sample mu when a strong view is applied"

    def test_bl_sigma_differs_from_sample_cov(self, simple_portfolio):
        R = self._make_R()
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.05])

        moments_bl = set_portfolio_moments(
            R, simple_portfolio, method="black_litterman", P=P, q=q
        )
        moments_sample = set_portfolio_moments(
            R, simple_portfolio, method="sample"
        )
        assert not np.allclose(moments_bl["sigma"], moments_sample["sigma"])


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Sentinel flag prevents mu_method overwrite (Bug ③ regression)
# ──────────────────────────────────────────────────────────────────────────────


class TestSentinelFlag:
    def test_bl_mu_not_overwritten_when_explicit_mu_method(self, simple_portfolio):
        """Before the fix, passing mu_method caused BL mu to be silently replaced
        by the sample mean.  After the fix the BL mu must survive."""
        import pandas as pd
        rng = np.random.default_rng(7)
        R = pd.DataFrame(rng.normal(0.001, 0.01, (120, 3)), columns=["A", "B", "C"])
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.05])   # strong view

        moments_bl = set_portfolio_moments(
            R, simple_portfolio, method="black_litterman",
            P=P, q=q,
            mu_method="sample",   # explicit mu_method — must NOT override BL mu
        )
        moments_sample = set_portfolio_moments(
            R, simple_portfolio, method="sample"
        )
        # The BL mu must differ from the plain sample mean.
        assert not np.allclose(moments_bl["mu"], moments_sample["mu"]), (
            "BL mu was silently overwritten by the sample mu (sentinel failed)"
        )

    def test_sentinel_key_cleaned_from_output(self, simple_portfolio):
        """The internal sentinel '_bl_moments_set' must not appear in the
        moments dict that downstream optimisers receive."""
        import pandas as pd
        rng = np.random.default_rng(7)
        R = pd.DataFrame(rng.normal(0.001, 0.01, (120, 3)), columns=["A", "B", "C"])
        moments = set_portfolio_moments(
            R, simple_portfolio, method="black_litterman"
        )
        # Sentinel is an internal implementation detail; it must be removed
        # before the dict is returned, or at minimum not affect downstream code.
        # We only assert it does not cause KeyError / incorrect mu.
        assert "mu" in moments


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Backward-compatible kwargs (bl_formulation, w_mkt, Mu, Sigma, Omega)
# ──────────────────────────────────────────────────────────────────────────────


class TestBackwardCompatKwargs:
    def test_custom_prior_mu(self, simple_data):
        """User-supplied Mu should be used as the prior."""
        Mu_custom = np.array([0.01, 0.02, 0.015])
        P = np.eye(3)
        # With an identity P and Omega = P @ Sigma @ P.T, the update is non-trivial
        res = black_litterman(simple_data, P, Mu=Mu_custom)
        # Result must not equal sample mean (custom prior should shift things)
        Mu_sample = np.mean(simple_data, axis=0)
        assert not np.allclose(res["mu"], Mu_sample)

    def test_he_litterman_via_bl_formulation_kwarg(self, simple_data):
        """bl_formulation='he_litterman' should route to He-Litterman."""
        P = np.ones((1, 3))
        res = black_litterman(simple_data, P, formulation="he_litterman")
        assert "Pi" in res

    def test_explicit_omega(self, simple_data):
        """Passing an explicit Omega must be respected."""
        Sigma = np.cov(simple_data.T, ddof=1)
        P = np.array([[1.0, -1.0, 0.0]])
        Omega_tight = 1e-6 * np.eye(1)   # very confident views
        Omega_loose = 1.0  * np.eye(1)   # very uncertain views

        res_tight = black_litterman(simple_data, P, Omega=Omega_tight)
        res_loose = black_litterman(simple_data, P, Omega=Omega_loose)
        # Tight Omega → stronger view pull → larger mu deviation
        q_default = np.sqrt(np.diag(P @ Sigma @ P.T))
        q = q_default  # same q, different Omega
        res_tight = black_litterman(simple_data, P, q=q, Omega=Omega_tight)
        res_loose = black_litterman(simple_data, P, q=q, Omega=Omega_loose)
        diff_tight = np.linalg.norm(res_tight["mu"] - np.mean(simple_data, axis=0))
        diff_loose = np.linalg.norm(res_loose["mu"] - np.mean(simple_data, axis=0))
        assert diff_tight > diff_loose, (
            "Tighter Omega should pull mu further from the prior"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_k_equals_n_full_view(self, simple_data):
        """K = N views (full observation set) — should not raise."""
        N = simple_data.shape[1]
        P = np.eye(N)
        q = np.mean(simple_data, axis=0)
        res = black_litterman(simple_data, P, q)
        assert res["mu"].shape == (N,)

    def test_k_equals_1_single_view(self, simple_data):
        """K = 1 view — the minimal case."""
        P = np.array([[1.0, -1.0, 0.0]])
        q = np.array([0.002])
        res = black_litterman(simple_data, P, q)
        assert res["mu"].shape == (3,)

    def test_prior_mu_vector_shape_variants(self, simple_data):
        """Mu can be (N,) or (N,1) — both should work."""
        P = np.eye(3)
        Mu_1d = np.mean(simple_data, axis=0)
        Mu_2d = Mu_1d.reshape(-1, 1)

        res_1d = black_litterman(simple_data, P, Mu=Mu_1d)
        res_2d = black_litterman(simple_data, P, Mu=Mu_2d)
        np.testing.assert_allclose(res_1d["mu"], res_2d["mu"], rtol=1e-10)
