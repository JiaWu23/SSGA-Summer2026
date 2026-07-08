"""Tests for extended M2 diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics import (
    m2_calibration_table,
    m2_classification_metrics,
    m2_probability_decile_returns,
)


def test_m2_extended_metrics_include_auc_pr():
    y = pd.Series([1, 0, 1, 0, 1, 0])
    p = pd.Series([0.7, 0.3, 0.6, 0.4, 0.8, 0.2])
    m = m2_classification_metrics(y, p, threshold=0.5)
    assert "auc" in m
    assert "auc_pr" in m
    assert "base_rate" in m
    assert m["base_rate"] == 0.5
    assert "mean_p_winners" in m
    assert "mean_p_losers" in m


def test_m2_calibration_table_buckets():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 50))
    p = pd.Series(rng.uniform(0.2, 0.9, 50))
    cal = m2_calibration_table(y, p, bins=5)
    assert not cal.empty
    assert "mean_pred" in cal.columns
    assert "realized" in cal.columns


def test_m2_probability_decile_returns():
    dates = pd.date_range("2021-01-01", periods=6, freq="W-FRI")
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "date": d,
                "ticker": "SPY",
                "M1_signal": 1,
                "p_success": 0.3 + i * 0.1,
                "trade_return": 0.001 * i,
                "meta_label": 1 if i >= 3 else 0,
            }
        )
    panel = pd.DataFrame(rows).set_index(["date", "ticker"])
    dec = m2_probability_decile_returns(panel, bins=3)
    assert not dec.empty
    assert "mean_trade_return" in dec.columns
