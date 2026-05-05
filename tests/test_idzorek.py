"""Tests for Idzorek view-confidence method in Black-Litterman.

Covers:
  - idzorek_omega() boundary conditions and mathematical properties
  - Parity with PyPortfolioOpt's idzorek_method (rtol=1e-12)
  - black_litterman() omega='idzorek' shorthand (Meucci and He-Litterman)
  - MomentConfig integration through BlackLittermanEstimator
  - Error handling (bad confidence values, missing view_confidences)
"""

import numpy as np
import pytest

from pyfolioanalytics.black_litterman import black_litterman, idzorek_omega


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def market_data():
    rng = np.random.default_rng(42)
    T, N = 120, 4
    R = rng.normal(0, 0.01, (T, N))
    Sigma = np.cov(R.T, ddof=1)
    Mu = R.mean(axis=0)
    return {"R": R, "Sigma": Sigma, "Mu": Mu, "T": T, "N": N}


@pytest.fixture(scope="module")
def two_views(market_data):
    """Two-view setup: asset0 > asset1, asset2 > asset3."""
    P = np.array([[1.0, -1.0, 0.0, 0.0],
                  [0.0,  0.0, 1.0, -1.0]])
    q = np.array([0.005, 0.003])
    return {"P": P, "q": q}


# ── idzorek_omega: boundary conditions ────────────────────────────────────────

class TestIdzorekOmegaBoundary:
    def test_conf_half_equals_proportional_prior(self, market_data, two_views):
        """conf=0.5 → ω_k = τ · P_k Σ P_k'  (matches He-Litterman prior)."""
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        tau = 1 / market_data["T"]

        Omega_idz = idzorek_omega([0.5, 0.5], P, Sigma, tau=tau)
        Omega_prop = np.diag(np.diag(tau * P @ Sigma @ P.T))

        np.testing.assert_allclose(Omega_idz, Omega_prop, rtol=1e-12)

    def test_high_confidence_small_omega(self, market_data, two_views):
        """conf=0.99 → very small Ω (posterior close to view)."""
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        tau = 1 / market_data["T"]

        Omega_high = idzorek_omega([0.99, 0.99], P, Sigma, tau=tau)
        Omega_low  = idzorek_omega([0.01, 0.01], P, Sigma, tau=tau)

        assert np.all(np.diag(Omega_high) < np.diag(Omega_low))

    def test_conf_one_gives_zero_omega(self, market_data, two_views):
        """conf=1.0 → ω_k = 0 (α = 0 → complete confidence)."""
        Sigma = market_data["Sigma"]
        P = two_views["P"]

        Omega = idzorek_omega([1.0, 1.0], P, Sigma, tau=1.0)
        np.testing.assert_allclose(Omega, np.zeros((2, 2)), atol=1e-15)

    def test_diagonal_structure(self, market_data, two_views):
        """Returned matrix must be diagonal."""
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        Omega = idzorek_omega([0.8, 0.6], P, Sigma)
        off_diag = Omega - np.diag(np.diag(Omega))
        np.testing.assert_array_equal(off_diag, 0.0)

    def test_monotone_in_confidence(self, market_data, two_views):
        """Higher confidence → smaller ω (monotone relationship)."""
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        confs = [0.1, 0.3, 0.5, 0.7, 0.9]
        omegas = [idzorek_omega([c, 0.5], P, Sigma)[0, 0] for c in confs]
        # omegas should be strictly decreasing
        for a, b in zip(omegas, omegas[1:]):
            assert a > b

    def test_tau_scaling(self, market_data, two_views):
        """Doubling tau doubles each ω_k."""
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        Omega1 = idzorek_omega([0.7, 0.4], P, Sigma, tau=0.01)
        Omega2 = idzorek_omega([0.7, 0.4], P, Sigma, tau=0.02)
        np.testing.assert_allclose(2 * Omega1, Omega2, rtol=1e-12)


# ── idzorek_omega: error handling ─────────────────────────────────────────────

class TestIdzorekOmegaErrors:
    def test_zero_confidence_raises(self, market_data, two_views):
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            idzorek_omega([0.0, 0.5], P, Sigma)

    def test_negative_confidence_raises(self, market_data, two_views):
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        with pytest.raises(ValueError):
            idzorek_omega([-0.1, 0.5], P, Sigma)

    def test_confidence_above_one_raises(self, market_data, two_views):
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        with pytest.raises(ValueError):
            idzorek_omega([1.1, 0.5], P, Sigma)

    def test_length_mismatch_raises(self, market_data, two_views):
        Sigma = market_data["Sigma"]
        P = two_views["P"]
        with pytest.raises(ValueError, match="must match"):
            idzorek_omega([0.5], P, Sigma)   # 1 conf but 2 views


# ── PyPortfolioOpt parity ─────────────────────────────────────────────────────

class TestIdzorekPyPfOptParity:
    """Verify numerical equivalence with PyPortfolioOpt's idzorek_method."""

    @pytest.fixture(autouse=True)
    def skip_if_no_pypfopt(self):
        try:
            from pypfopt.black_litterman import BlackLittermanModel  # noqa: F401
        except ImportError:
            pytest.skip("pyportfolioopt not installed")

    def test_single_view_parity(self, market_data):
        from pypfopt.black_litterman import BlackLittermanModel

        Sigma = market_data["Sigma"]
        T = market_data["T"]
        tau = 1.0 / T
        P = np.array([[1.0, -1.0, 0.0, 0.0]])
        q = np.array([0.005])
        pi = np.zeros(4)  # dummy pi for PyPfOpt call
        confs = np.array([0.75])

        our_omega = idzorek_omega(confs, P, Sigma, tau=tau)
        ref_omega = BlackLittermanModel.idzorek_method(confs, Sigma, pi, q, P, tau)

        np.testing.assert_allclose(our_omega, ref_omega, rtol=1e-12)

    def test_two_view_parity(self, market_data, two_views):
        from pypfopt.black_litterman import BlackLittermanModel

        Sigma = market_data["Sigma"]
        T = market_data["T"]
        tau = 1.0 / T
        P = two_views["P"]
        q = two_views["q"]
        pi = np.zeros(4)
        confs = np.array([0.80, 0.60])

        our_omega = idzorek_omega(confs, P, Sigma, tau=tau)
        ref_omega = BlackLittermanModel.idzorek_method(confs, Sigma, pi, q, P, tau)

        np.testing.assert_allclose(our_omega, ref_omega, rtol=1e-12)


# ── black_litterman() omega='idzorek' shorthand ───────────────────────────────

class TestBlackLittermanIdzorekShorthand:
    def test_meucci_idzorek_returns_correct_shapes(self, market_data, two_views):
        R = market_data["R"]
        P, q = two_views["P"], two_views["q"]
        confs = np.array([0.75, 0.60])

        res = black_litterman(R, P, q, Omega="idzorek", view_confidences=confs)

        assert res["mu"].shape == (market_data["N"],)
        assert res["sigma"].shape == (market_data["N"], market_data["N"])

    def test_meucci_idzorek_equivalent_to_explicit_omega(self, market_data, two_views):
        """omega='idzorek' must give same result as manually computed Omega."""
        R = market_data["R"]
        P, q = two_views["P"], two_views["q"]
        Sigma = market_data["Sigma"]
        confs = np.array([0.80, 0.55])

        # Manual: compute Omega explicitly, pass as array
        Omega_explicit = idzorek_omega(confs, P, Sigma, tau=1.0)  # Meucci: tau=1
        res_explicit = black_litterman(R, P, q, Omega=Omega_explicit)

        # Shorthand
        res_shorthand = black_litterman(R, P, q, Omega="idzorek",
                                        view_confidences=confs)

        np.testing.assert_allclose(res_shorthand["mu"],    res_explicit["mu"],    rtol=1e-12)
        np.testing.assert_allclose(res_shorthand["sigma"], res_explicit["sigma"], rtol=1e-12)

    def test_he_litterman_idzorek_returns_pi(self, market_data, two_views):
        R = market_data["R"]
        P, q = two_views["P"], two_views["q"]
        confs = np.array([0.70, 0.50])

        res = black_litterman(
            R, P, q,
            Omega="idzorek", view_confidences=confs,
            formulation="he_litterman",
        )

        assert "Pi" in res
        assert res["mu"].shape == (market_data["N"],)

    def test_he_litterman_idzorek_vs_explicit_omega(self, market_data, two_views):
        """He-Litterman: idzorek uses tau, so we verify vs explicit Omega(tau)."""
        R = market_data["R"]
        T = market_data["T"]
        P, q = two_views["P"], two_views["q"]
        Sigma = market_data["Sigma"]
        confs = np.array([0.65, 0.80])
        tau = 1.0 / T

        Omega_explicit = idzorek_omega(confs, P, Sigma, tau=tau)
        res_explicit = black_litterman(
            R, P, q, Omega=Omega_explicit,
            formulation="he_litterman", tau=tau,
        )
        res_shorthand = black_litterman(
            R, P, q, Omega="idzorek", view_confidences=confs,
            formulation="he_litterman", tau=tau,
        )

        np.testing.assert_allclose(res_shorthand["mu"],    res_explicit["mu"],    rtol=1e-12)
        np.testing.assert_allclose(res_shorthand["sigma"], res_explicit["sigma"], rtol=1e-12)

    def test_missing_view_confidences_raises(self, market_data, two_views):
        R = market_data["R"]
        P, q = two_views["P"], two_views["q"]

        with pytest.raises(ValueError, match="view_confidences"):
            black_litterman(R, P, q, Omega="idzorek")


# ── MomentConfig integration ──────────────────────────────────────────────────

class TestMomentConfigIdzorekIntegration:
    def test_config_idzorek_meucci(self, market_data, two_views):
        import pandas as pd
        from pyfolioanalytics import Portfolio
        from pyfolioanalytics.moments import MomentConfig, set_portfolio_moments

        R_df = pd.DataFrame(market_data["R"], columns=list("ABCD"))
        port = Portfolio(assets=list("ABCD"))
        port.add_objective("risk", name="StdDev")

        confs = np.array([0.75, 0.60])
        cfg = MomentConfig(
            method="black_litterman",
            P=two_views["P"],
            q=two_views["q"],
            Omega="idzorek",
            view_confidences=confs,
        )
        moments = set_portfolio_moments(R_df, port, config=cfg)

        assert "mu" in moments and "sigma" in moments
        assert moments["mu"].shape[0] == 4
        assert moments["sigma"].shape == (4, 4)

    def test_config_idzorek_matches_direct_call(self, market_data, two_views):
        """MomentConfig path must produce identical results to black_litterman()."""
        import pandas as pd
        from pyfolioanalytics import Portfolio
        from pyfolioanalytics.moments import MomentConfig, set_portfolio_moments

        R = market_data["R"]
        R_df = pd.DataFrame(R, columns=list("ABCD"))
        port = Portfolio(assets=list("ABCD"))
        port.add_objective("risk", name="StdDev")

        P, q = two_views["P"], two_views["q"]
        confs = np.array([0.80, 0.55])

        # Direct call
        res_direct = black_litterman(R, P, q, Omega="idzorek", view_confidences=confs)

        # Via MomentConfig
        cfg = MomentConfig(
            method="black_litterman",
            P=P, q=q,
            Omega="idzorek",
            view_confidences=confs,
        )
        moments = set_portfolio_moments(R_df, port, config=cfg)

        np.testing.assert_allclose(moments["mu"].ravel(), res_direct["mu"],    rtol=1e-12)
        np.testing.assert_allclose(moments["sigma"],       res_direct["sigma"], rtol=1e-12)
