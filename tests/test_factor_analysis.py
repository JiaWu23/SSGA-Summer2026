"""Tests for M1 factor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor_analysis import (
    build_weight_variants,
    composite_score_from_components,
    compute_factor_correlation,
    compute_factor_covariance,
    compute_factor_ic,
    FACTOR_COLS,
    ic_proportional_weights,
)


def _synthetic_panel(n_dates: int = 12, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["SPY", "TLT"]
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "forward_return_4w": rng.normal(0.001, 0.02),
                    "momentum_score": rng.normal(0, 1),
                    "trend_score": rng.normal(0, 1),
                    "macro_score": rng.normal(0, 0.5),
                    "risk_penalty": rng.uniform(0, 1),
                    "M1_score": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_compute_factor_ic_returns_rows():
    panel = _synthetic_panel()
    ic = compute_factor_ic(panel, period_label="test")
    assert not ic.empty
    assert set(ic["factor"]).issuperset(set(FACTOR_COLS))


def test_factor_correlation_and_covariance():
    panel = _synthetic_panel()
    corr = compute_factor_correlation(panel)
    cov = compute_factor_covariance(panel)
    assert corr.shape == (4, 4)
    assert cov.shape == (4, 4)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_ic_proportional_weights_normalize():
    ic = pd.DataFrame(
        {
            "period": ["train", "train", "train", "train"],
            "factor": FACTOR_COLS,
            "ic_mean": [0.03, 0.01, -0.02, 0.02],
        }
    )
    w = ic_proportional_weights(ic, period="train")
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["macro"] == 0.0


def test_composite_score_from_components():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2021-01-01"), "SPY")], names=["date", "ticker"])
    comps = pd.DataFrame(
        {
            "momentum_score": [1.0],
            "trend_score": [2.0],
            "macro_score": [0.5],
            "risk_penalty": [1.0],
        },
        index=idx,
    )
    weights = {"momentum": 0.45, "trend": 0.25, "macro": 0.20, "risk_penalty": 0.10}
    score = composite_score_from_components(comps, weights)
    expected = 0.45 * 1.0 + 0.25 * 2.0 + 0.20 * 0.5 - 0.10 * 1.0
    assert abs(float(score.iloc[0]) - expected) < 1e-9


def test_build_weight_variants_includes_trend_heavy():
    ic = pd.DataFrame(
        {
            "period": ["train"] * 4,
            "factor": FACTOR_COLS,
            "ic_mean": [0.03, 0.01, 0.0, 0.02],
        }
    )
    corr = pd.DataFrame(
        np.eye(4),
        index=FACTOR_COLS,
        columns=FACTOR_COLS,
    )
    corr.loc["momentum_score", "trend_score"] = 0.77
    corr.loc["trend_score", "momentum_score"] = 0.77
    baseline = {"momentum": 0.45, "trend": 0.25, "macro": 0.20, "risk_penalty": 0.10}
    names = [v[0] for v in build_weight_variants(baseline, ic, corr)]
    assert "trend_heavy" in names
    assert "technical_ic_blend" in names
