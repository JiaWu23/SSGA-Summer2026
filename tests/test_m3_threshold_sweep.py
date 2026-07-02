"""Tests for M3 threshold sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.m3_threshold_sweep import linear_gated_size, recommend_m3_thresholds, sweep_m3_thresholds


def _mini_cfg():
    from src.config import load_config

    cfg = load_config("config/config.yaml")
    cfg.split.test_start = "2020-01-01"
    cfg.split.test_end = "2021-12-31"
    return cfg


def _synthetic_panel(n_weeks: int = 40, n_assets: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2018-01-05", periods=n_weeks, freq="W-FRI")
    tickers = [f"T{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(42)
    p = rng.uniform(0.45, 0.75, len(idx))
    m1 = rng.choice([-1, 0, 1], len(idx), p=[0.2, 0.5, 0.3])
    fwd = rng.normal(0.001, 0.02, len(idx))
    ret1w = rng.normal(0.0005, 0.01, len(idx))
    df = pd.DataFrame(
        {
            "M1_signal": m1,
            "M1_conviction": rng.uniform(0.5, 1.0, len(idx)),
            "p_success": p,
            "meta_label": (fwd > 0).astype(int),
            "forward_return_4w": fwd,
            "return_1w": ret1w,
        },
        index=idx,
    )
    return df


def test_linear_gated_size_zeros_below_threshold():
    p = pd.Series([0.4, 0.55, 0.7], index=[0, 1, 2])
    out = linear_gated_size(p, 0.55)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == pytest.approx(0.1)
    assert out.iloc[2] == pytest.approx(0.4)


def test_recommend_m3_thresholds_prefers_meaningful_rejection():
    sweep = pd.DataFrame(
        [
            {
                "m3_mode": "binary",
                "threshold": 0.55,
                "test_sharpe": 0.8,
                "m3_rejection_share": 0.001,
                "m2_recall": 0.999,
                "meaningful_rejection": False,
                "test_ann_return": 0.1,
                "sharpe_edge_vs_m1": 0.0,
                "m2_precision": 0.5,
            },
            {
                "m3_mode": "binary",
                "threshold": 0.62,
                "test_sharpe": 0.85,
                "m3_rejection_share": 0.12,
                "m2_recall": 0.88,
                "meaningful_rejection": True,
                "test_ann_return": 0.12,
                "sharpe_edge_vs_m1": 0.05,
                "m2_precision": 0.55,
            },
        ]
    )
    rec = recommend_m3_thresholds(sweep, baseline_threshold=0.55)
    assert rec["binary"]["recommended_threshold"] == 0.62
    assert rec["binary"]["meaningful_rejection"] is True


def test_sweep_m3_thresholds_runs_on_synthetic():
    from src.backtest import returns_wide_from_panel

    panel = _synthetic_panel()
    cfg = _mini_cfg()
    test = panel.loc[panel.index.get_level_values("date") >= "2020-01-01"]
    returns_wide = returns_wide_from_panel(panel.reset_index(), ["T0", "T1", "T2"])
    train = panel.loc[panel.index.get_level_values("date") < "2020-01-01"]
    train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
    sweep = sweep_m3_thresholds(
        panel,
        test,
        returns_wide,
        cfg,
        train_proba=train_proba,
        threshold_grid=(0.50, 0.60, 0.70),
    )
    assert len(sweep) == 6
    assert set(sweep["m3_mode"]) == {"binary", "linear_gated"}
    assert sweep["threshold"].nunique() == 3
