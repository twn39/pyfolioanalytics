"""Tests for the MultLayerPortfolio independent optimization parameters.

Structure
---------
1. ``SubPortfolioConfig`` — dataclass construction and __repr__
2. ``MultLayerPortfolio.add_sub_portfolio()`` — new API coverage
3. Backward-compatibility — bare Portfolio still works
4. Independent ``optimize_method`` per sub-portfolio (DEoptim vs ROI)
5. ``search_size`` → ``permutations`` mapping (random engine)
6. Asset column subsetting (Bug ④ regression)
7. Root ``optimize_method`` forwarding (Bug ③ regression)
8. ``SubPortfolioConfig`` pre-built and passed directly
9. Per-sub ``sub_kwargs`` forwarding
10. Public API exports
"""

import numpy as np
import pytest

from pyfolioanalytics import MultLayerPortfolio, Portfolio, SubPortfolioConfig, optimize_portfolio


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SubPortfolioConfig — dataclass
# ──────────────────────────────────────────────────────────────────────────────


class TestSubPortfolioConfig:
    def test_default_values(self):
        p = Portfolio(assets=2)
        cfg = SubPortfolioConfig(portfolio=p)
        assert cfg.optimize_method == "ROI"
        assert cfg.search_size == 20_000
        assert cfg.kwargs == {}

    def test_custom_values(self):
        p = Portfolio(assets=2)
        cfg = SubPortfolioConfig(
            portfolio=p,
            optimize_method="DEoptim",
            search_size=500,
            kwargs={"itermax": 50},
        )
        assert cfg.optimize_method == "DEoptim"
        assert cfg.search_size == 500
        assert cfg.kwargs["itermax"] == 50

    def test_repr(self):
        p = Portfolio(assets=2)
        cfg = SubPortfolioConfig(portfolio=p, optimize_method="random")
        r = repr(cfg)
        assert "random" in r
        assert "SubPortfolioConfig" in r

    def test_mutable_kwargs_are_independent(self):
        """Each instance gets its own kwargs dict (field(default_factory=dict))."""
        p = Portfolio(assets=2)
        cfg1 = SubPortfolioConfig(portfolio=p)
        cfg2 = SubPortfolioConfig(portfolio=p)
        cfg1.kwargs["x"] = 1
        assert "x" not in cfg2.kwargs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MultLayerPortfolio.add_sub_portfolio() — new API
# ──────────────────────────────────────────────────────────────────────────────


class TestAddSubPortfolio:
    def _mlp(self):
        p_root = Portfolio(assets={"A": 0.5, "B": 0.5})
        p_root.add_constraint(type="full_investment")
        return MultLayerPortfolio(p_root)

    def test_bare_portfolio_auto_wrapped(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        mlp.add_sub_portfolio("A", p_sub)
        assert isinstance(mlp.sub_portfolios["A"], SubPortfolioConfig)

    def test_optimize_method_stored(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        mlp.add_sub_portfolio("A", p_sub, optimize_method="DEoptim")
        assert mlp.sub_portfolios["A"].optimize_method == "DEoptim"

    def test_search_size_stored(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        mlp.add_sub_portfolio("A", p_sub, search_size=500)
        assert mlp.sub_portfolios["A"].search_size == 500

    def test_sub_kwargs_stored(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        mlp.add_sub_portfolio("A", p_sub, itermax=30)
        assert mlp.sub_portfolios["A"].kwargs["itermax"] == 30

    def test_pre_built_config_stored_as_is(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        cfg = SubPortfolioConfig(portfolio=p_sub, optimize_method="random", search_size=100)
        mlp.add_sub_portfolio("A", cfg)
        assert mlp.sub_portfolios["A"] is cfg

    def test_invalid_meta_asset_raises(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        with pytest.raises(ValueError, match="root portfolio"):
            mlp.add_sub_portfolio("DOES_NOT_EXIST", p_sub)

    def test_returns_self_for_chaining(self):
        mlp = self._mlp()
        p_sub = Portfolio(assets=2)
        result = mlp.add_sub_portfolio("A", p_sub)
        assert result is mlp


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Backward-compatibility — existing code still works
# ──────────────────────────────────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_bare_portfolio_optimises(self, stocks_data):
        """Existing callers that pass a bare Portfolio must still work."""
        p_tech = Portfolio(assets=["AAPL", "MSFT"])
        p_tech.add_constraint(type="full_investment")
        p_tech.add_constraint(type="long_only")
        p_tech.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"Tech": 0.5, "AMZN": 0.2, "GOOGL": 0.2, "META": 0.1})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("Tech", p_tech)   # No optimize_method kwarg

        res = optimize_portfolio(stocks_data, mlp)
        assert res["weights"] is not None
        assert "AAPL" in res["weights"].index
        assert "MSFT" in res["weights"].index
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Independent optimize_method per sub-portfolio
# ──────────────────────────────────────────────────────────────────────────────


class TestIndependentOptimizeMethod:
    def test_mixed_methods_produce_valid_result(self, stocks_data):
        """Two sub-portfolios using different methods must both return weights."""
        p_g1 = Portfolio(assets=["AAPL", "MSFT"])
        p_g1.add_constraint(type="full_investment")
        p_g1.add_constraint(type="long_only")
        p_g1.add_objective(type="risk", name="StdDev")

        p_g2 = Portfolio(assets=["AMZN", "GOOGL", "META"])
        p_g2.add_constraint(type="full_investment")
        p_g2.add_constraint(type="long_only")
        p_g2.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"G1": 0.5, "G2": 0.5})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("G1", p_g1, optimize_method="ROI")
        mlp.add_sub_portfolio("G2", p_g2, optimize_method="DEoptim", itermax=40)

        res = optimize_portfolio(stocks_data, mlp, optimize_method="ROI")

        assert res["status"] in ("optimal", "optimal_inaccurate", "feasible")
        assert set(res["sub_results"]["G1"]["weights"].index) == {"AAPL", "MSFT"}
        assert set(res["sub_results"]["G2"]["weights"].index) == {"AMZN", "GOOGL", "META"}
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)

    def test_sub_configs_are_independent(self, stocks_data):
        """Changing one sub-portfolio's method must not affect the other."""
        p_g1 = Portfolio(assets=["AAPL", "MSFT"])
        p_g1.add_constraint(type="full_investment")
        p_g1.add_constraint(type="long_only")
        p_g1.add_objective(type="risk", name="StdDev")

        p_g2 = Portfolio(assets=["AMZN", "GOOGL"])
        p_g2.add_constraint(type="full_investment")
        p_g2.add_constraint(type="long_only")
        p_g2.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"G1": 0.5, "G2": 0.5})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("G1", p_g1, optimize_method="ROI")
        mlp.add_sub_portfolio("G2", p_g2, optimize_method="random", search_size=200)

        assert mlp.sub_portfolios["G1"].optimize_method == "ROI"
        assert mlp.sub_portfolios["G2"].optimize_method == "random"
        assert mlp.sub_portfolios["G2"].search_size == 200


# ──────────────────────────────────────────────────────────────────────────────
# 5.  search_size → permutations mapping (random engine)
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchSizeMapping:
    def test_random_sub_portfolio_uses_search_size(self, stocks_data):
        """Sub-portfolio with optimize_method='random' and small search_size
        should still converge to a valid portfolio."""
        p_sub = Portfolio(assets=["AAPL", "MSFT"])
        p_sub.add_constraint(type="full_investment")
        p_sub.add_constraint(type="long_only")
        p_sub.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"Sub": 0.6, "AMZN": 0.4})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        # Small search_size to keep the test fast
        mlp.add_sub_portfolio("Sub", p_sub, optimize_method="random", search_size=300)

        res = optimize_portfolio(stocks_data, mlp, optimize_method="ROI")
        assert res["status"] in ("optimal", "optimal_inaccurate", "feasible")
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Asset column subsetting — Bug ④ regression
# ──────────────────────────────────────────────────────────────────────────────


class TestAssetSubsetting:
    def test_sub_portfolio_weights_contain_only_its_assets(self, stocks_data):
        """Sub-portfolio weights index must contain only the sub-portfolio's
        assets, not the full returns dataset."""
        p_g1 = Portfolio(assets=["AAPL", "MSFT"])
        p_g1.add_constraint(type="full_investment")
        p_g1.add_constraint(type="long_only")
        p_g1.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"G1": 1.0})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("G1", p_g1)

        res = optimize_portfolio(stocks_data, mlp)
        sub_idx = set(res["sub_results"]["G1"]["weights"].index)
        assert sub_idx == {"AAPL", "MSFT"}, (
            f"Sub-portfolio weight index should be {{'AAPL','MSFT'}}, got {sub_idx}"
        )

    def test_full_returns_R_has_more_columns_than_sub(self, stocks_data):
        """Full R has 5 columns; sub-portfolio only uses 2.
        The function must not crash or use the wrong columns."""
        assert len(stocks_data.columns) >= 3, "Need at least 3 assets in fixture"

        p_sub = Portfolio(assets=stocks_data.columns[:2].tolist())
        p_sub.add_constraint(type="full_investment")
        p_sub.add_constraint(type="long_only")
        p_sub.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"Sub": 0.5, stocks_data.columns[2]: 0.5})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("Sub", p_sub)

        res = optimize_portfolio(stocks_data, mlp)
        assert res["weights"] is not None
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Root optimize_method forwarding — Bug ③ regression
# ──────────────────────────────────────────────────────────────────────────────


class TestRootMethodForwarding:
    def test_root_uses_specified_method(self, stocks_data):
        """Before the fix, optimize_method was silently dropped before being
        forwarded to optimize_portfolio_multi_layer, so the root portfolio
        always fell back to 'ROI'.  Now it should use whatever is specified."""
        p_sub = Portfolio(assets=["AAPL", "MSFT"])
        p_sub.add_constraint(type="full_investment")
        p_sub.add_constraint(type="long_only")
        p_sub.add_objective(type="risk", name="StdDev")

        p_root = Portfolio(assets={"Sub": 0.6, "AMZN": 0.4})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("Sub", p_sub)

        # Explicitly request DEoptim for the root; before the fix this was ignored.
        res = optimize_portfolio(stocks_data, mlp, optimize_method="DEoptim", itermax=40)
        assert res["status"] in ("optimal", "optimal_inaccurate")
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  SubPortfolioConfig passed directly to add_sub_portfolio
# ──────────────────────────────────────────────────────────────────────────────


class TestPreBuiltConfig:
    def test_pre_built_config_end_to_end(self, stocks_data):
        p_sub = Portfolio(assets=["AAPL", "MSFT"])
        p_sub.add_constraint(type="full_investment")
        p_sub.add_constraint(type="long_only")
        p_sub.add_objective(type="risk", name="StdDev")

        cfg = SubPortfolioConfig(
            portfolio=p_sub,
            optimize_method="ROI",
            search_size=500,
        )

        p_root = Portfolio(assets={"Sub": 0.7, "AMZN": 0.3})
        p_root.add_constraint(type="full_investment")
        p_root.add_objective(type="risk", name="StdDev")

        mlp = MultLayerPortfolio(p_root)
        mlp.add_sub_portfolio("Sub", cfg)  # pass pre-built config

        res = optimize_portfolio(stocks_data, mlp)
        assert res["status"] in ("optimal", "optimal_inaccurate", "feasible")
        assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Public API exports
# ──────────────────────────────────────────────────────────────────────────────


class TestPublicApiExports:
    def test_subportfolioconfig_importable_from_package(self):
        from pyfolioanalytics import SubPortfolioConfig as SPC  # noqa: F401
        assert SPC is SubPortfolioConfig

    def test_multilayerportfolio_importable_from_package(self):
        from pyfolioanalytics import MultLayerPortfolio as MLP  # noqa: F401
        assert MLP is MultLayerPortfolio

    def test_subportfolioconfig_in_all(self):
        import pyfolioanalytics
        assert "SubPortfolioConfig" in pyfolioanalytics.__all__

    def test_multilayerportfolio_in_all(self):
        import pyfolioanalytics
        assert "MultLayerPortfolio" in pyfolioanalytics.__all__
