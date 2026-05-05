"""Tests for _infer_periods_per_year() and BacktestResult.summary() annualisation."""

import numpy as np
import pandas as pd
import pytest

from pyfolioanalytics.backtest import BacktestResult, _infer_periods_per_year


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_result(rets: pd.Series) -> BacktestResult:
    """Wrap a return series in a minimal BacktestResult."""
    return BacktestResult(
        weights=pd.DataFrame(index=rets.index),
        returns=rets,
        opt_results=[],
    )


# ── _infer_periods_per_year ───────────────────────────────────────────────────

class TestInferPeriodsPerYear:
    def test_business_daily(self):
        idx = pd.bdate_range("2020-01-01", periods=252)
        assert _infer_periods_per_year(idx) == 252.0

    def test_calendar_daily(self):
        idx = pd.date_range("2020-01-01", periods=365, freq="D")
        assert _infer_periods_per_year(idx) == 365.0

    def test_weekly(self):
        idx = pd.date_range("2020-01-06", periods=52, freq="W-MON")
        assert _infer_periods_per_year(idx) == 52.0

    def test_monthly_ME(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        assert _infer_periods_per_year(idx) == 12.0

    def test_monthly_MS(self):
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        assert _infer_periods_per_year(idx) == 12.0

    def test_quarterly(self):
        idx = pd.date_range("2020-03-31", periods=12, freq="QE")
        assert _infer_periods_per_year(idx) == 4.0

    def test_annual(self):
        idx = pd.date_range("2015-12-31", periods=10, freq="YE")
        assert _infer_periods_per_year(idx) == 1.0

    def test_single_element_falls_back(self):
        """Single-element index cannot determine frequency → fall back to 252."""
        idx = pd.DatetimeIndex(["2020-01-31"])
        assert _infer_periods_per_year(idx) == 252.0

    def test_irregular_monthly_fallback(self):
        """Irregular index with ~30-day gaps → Tier-2 median → 12."""
        # Simulate slightly irregular monthly dates (not a clean freq string)
        dates = pd.to_datetime(
            ["2020-01-31", "2020-03-02", "2020-03-30", "2020-04-29",
             "2020-06-01", "2020-06-30"]
        )
        result = _infer_periods_per_year(pd.DatetimeIndex(dates))
        assert result == 12.0

    def test_explicit_override_ignores_index(self):
        """periods_per_year kwarg must take precedence over auto-inference."""
        monthly_idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        rets = pd.Series([0.01] * 12, index=monthly_idx)
        result = _make_result(rets)
        summary = result.summary(periods_per_year=52.0)   # force weekly
        # Annualised vol = std * sqrt(52) — confirm not sqrt(12)
        expected_vol = rets.std() * np.sqrt(52.0)
        assert abs(summary.loc["Annualized Volatility", "Gross"] - expected_vol) < 1e-12


# ── BacktestResult.summary() correctness ─────────────────────────────────────

class TestSummaryAnnualisation:
    """Verify that metrics scale correctly for non-daily data."""

    @pytest.fixture
    def monthly_result(self):
        rets = pd.Series(
            [0.01] * 24,
            index=pd.date_range("2020-01-31", periods=24, freq="ME"),
        )
        return _make_result(rets)

    def test_ppy_inferred_as_12_for_monthly(self, monthly_result):
        """Auto-inferred ppy should be 12 for a monthly index."""
        rets = monthly_result.returns
        ppy = _infer_periods_per_year(rets.index)
        assert ppy == 12.0

    def test_annualized_volatility_monthly(self, monthly_result):
        rets = monthly_result.returns
        summary = monthly_result.summary()
        expected = rets.std() * np.sqrt(12.0)
        assert abs(summary.loc["Annualized Volatility", "Gross"] - expected) < 1e-12

    def test_sharpe_ratio_monthly(self, monthly_result):
        # Use a result with varied returns so std != 0
        rng = np.random.default_rng(42)
        rets = pd.Series(
            rng.normal(0.01, 0.02, 24),
            index=pd.date_range("2020-01-31", periods=24, freq="ME"),
        )
        result = _make_result(rets)
        summary = result.summary(risk_free_rate=0.0)
        expected = (rets.mean() / rets.std()) * np.sqrt(12.0)
        assert abs(summary.loc["Sharpe Ratio", "Gross"] - expected) < 1e-10

    def test_sharpe_daily_unchanged(self):
        """Daily data should still produce the same Sharpe as before."""
        rets = pd.Series(
            [0.001] * 252,
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        result = _make_result(rets)
        summary = result.summary(risk_free_rate=0.0)
        expected = (rets.mean() / rets.std()) * np.sqrt(252.0)
        assert abs(summary.loc["Sharpe Ratio", "Gross"] - expected) < 1e-10

    def test_cagr_monthly(self, monthly_result):
        rets = monthly_result.returns
        summary = monthly_result.summary()
        cum = (1 + rets).prod() - 1
        expected_cagr = (1 + cum) ** (12.0 / len(rets)) - 1
        assert abs(summary.loc["CAGR", "Gross"] - expected_cagr) < 1e-12

    def test_monthly_vs_daily_sharpe_ratio(self):
        """Monthly Sharpe must NOT equal daily Sharpe for identical raw return series."""
        rng = np.random.default_rng(0)
        raw = rng.normal(0.01, 0.02, 24).tolist()
        monthly_rets = pd.Series(
            raw,
            index=pd.date_range("2020-01-31", periods=24, freq="ME"),
        )
        daily_rets = pd.Series(
            raw,
            index=pd.bdate_range("2020-01-01", periods=24),
        )
        s_monthly = _make_result(monthly_rets).summary().loc["Sharpe Ratio", "Gross"]
        s_daily   = _make_result(daily_rets).summary().loc["Sharpe Ratio", "Gross"]
        # Both std are identical; only ppy differs → ratio = sqrt(12/252)
        assert not np.isclose(s_monthly, s_daily, rtol=0.01)
        ratio = s_monthly / s_daily
        assert abs(ratio - np.sqrt(12.0 / 252.0)) < 1e-8
