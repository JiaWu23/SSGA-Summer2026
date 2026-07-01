"""Tests for M1 factor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor_analysis import (
    compute_factor_correlation,
    compute_factor_covariance,
    compute_factor_ic,
    FACTOR_COLS,
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
