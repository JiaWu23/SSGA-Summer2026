"""Tests for market regime analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.regime_analysis import (
    build_regime_timeline,
    regime_transition_summary,
)


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    tickers = ["SPY", "TLT"]
    rows = []
    for i, d in enumerate(dates):
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "risk_off": float(i % 4 == 0),
                    "curve_inverted": float(i % 5 == 0),
                    "vix_level": 15.0 + i,
                    "credit_stress": 2.0,
                    "yield_curve": 0.5,
                    "growth_trend": 0.01,
                    "inflation_trend": 0.02,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_build_regime_timeline_one_row_per_date():
    panel = _synthetic_panel()
    timeline = build_regime_timeline(panel)
    assert len(timeline) == 20
    assert "risk_off" in timeline.columns


def test_regime_transition_summary_counts():
    panel = _synthetic_panel()
    timeline = build_regime_timeline(panel)
    transitions = regime_transition_summary(timeline)
    assert not transitions.empty
    assert "n_transitions" in transitions.columns
    assert (transitions["n_transitions"] >= 0).all()
