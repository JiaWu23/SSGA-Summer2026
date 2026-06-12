"""Tests for the opt-in inverse-volatility weighting scheme (compute_budgets)."""

import numpy as np
import pandas as pd
import pytest

from src.config import PortfolioConfig
from src.portfolio import INV_VOL_COLUMN, compute_budgets


def make_panel(vols, signals):
    dates = pd.to_datetime(["2024-01-05"]) .repeat(len(vols))
    tickers = [f"T{i}" for i in range(len(vols))]
    idx = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame({INV_VOL_COLUMN: vols, "M1_signal": signals}, index=idx)


def test_equal_scheme_returns_plain_float():
    """Default scheme must leave the baseline code path bit-for-bit unchanged."""
    cfg = PortfolioConfig()
    df = make_panel([0.10, 0.20, 0.30], [1, 1, 1])
    out = compute_budgets(df, cfg)
    assert isinstance(out, float)
    assert out == cfg.base_budget_per_asset


def test_calmer_asset_gets_more_capital():
    cfg = PortfolioConfig(weighting_scheme="inverse_vol")
    df = make_panel([0.10, 0.20, 0.40], [1, 1, 1])
    b = compute_budgets(df, cfg)
    assert b.iloc[0] > b.iloc[1] > b.iloc[2]
    # 1/0.1 : 1/0.2 : 1/0.4 = 4 : 2 : 1 shares
    np.testing.assert_allclose(b.iloc[0] / b.iloc[2], 4.0, rtol=1e-9)


def test_total_deployed_budget_matches_equal_scheme():
    """Inverse-vol redistributes capital; it must not change the total deployed."""
    cfg = PortfolioConfig(weighting_scheme="inverse_vol")
    df = make_panel([0.08, 0.15, 0.30, 0.25], [1, 1, 1, 0])
    b = compute_budgets(df, cfg)
    selected = df["M1_signal"] != 0
    total = b[selected].sum()
    np.testing.assert_allclose(total, selected.sum() * cfg.base_budget_per_asset, rtol=1e-9)


def test_missing_vol_falls_back_to_equal_shares():
    cfg = PortfolioConfig(weighting_scheme="inverse_vol")
    df = make_panel([np.nan, np.nan, np.nan], [1, 1, 1])
    # NaN vol -> clip leaves NaN -> inv NaN; whole-date fallback to equal shares
    df[INV_VOL_COLUMN] = np.nan
    b = compute_budgets(df, cfg)
    np.testing.assert_allclose(b.values, cfg.base_budget_per_asset, rtol=1e-9)


def test_vol_floor_guards_division():
    cfg = PortfolioConfig(weighting_scheme="inverse_vol", inv_vol_floor_ann=0.02)
    df = make_panel([1e-9, 0.20], [1, 1])
    b = compute_budgets(df, cfg)
    assert np.isfinite(b).all()
    # floored asset behaves as if vol == 0.02 -> share ratio 10:1
    np.testing.assert_allclose(b.iloc[0] / b.iloc[1], 10.0, rtol=1e-9)


def test_unknown_scheme_raises():
    cfg = PortfolioConfig(weighting_scheme="banana")
    df = make_panel([0.1], [1])
    with pytest.raises(ValueError):
        compute_budgets(df, cfg)
