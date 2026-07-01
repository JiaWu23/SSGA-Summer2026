"""Tests for M3 bet-sizing and allocation states."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_m3 import allocation_state, attach_m3_to_panel, compute_m3_size
from src.position_sizing import SizingMode


def _panel() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2021-01-01"), "SPY"),
            (pd.Timestamp("2021-01-01"), "TLT"),
            (pd.Timestamp("2021-01-08"), "SPY"),
        ],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "M1_signal": [1, 0, 1],
            "p_success": [0.7, np.nan, 0.4],
        },
        index=idx,
    )


def test_allocation_state_labels():
    panel = _panel()
    m3 = pd.Series([0.8, 0.0, 0.0], index=panel.index)
    states = allocation_state(panel["M1_signal"], m3)
    assert states.loc[(pd.Timestamp("2021-01-01"), "SPY")] == "m3_active"
    assert states.loc[(pd.Timestamp("2021-01-01"), "TLT")] == "no_signal"
    assert states.loc[(pd.Timestamp("2021-01-08"), "SPY")] == "m3_zero"


def test_binary_m3_zero_below_threshold():
    p = pd.Series([0.4, 0.6])
    sizes = compute_m3_size(p, SizingMode.BINARY, threshold=0.55)
    assert sizes.iloc[0] == 0.0
    assert sizes.iloc[1] == 1.0


def test_attach_m3_adds_columns(cfg, synthetic_panel):
    panel = synthetic_panel.copy()
    panel["p_success"] = 0.6
    panel.loc[panel["M1_signal"] == 0, "p_success"] = np.nan
    train_proba = panel.loc[panel["M1_signal"] != 0, "p_success"].head(10)
    out = attach_m3_to_panel(panel, cfg, train_proba=train_proba)
    assert "M3_size" in out.columns
    assert "M3_size_binary" in out.columns
    assert "allocation_state" in out.columns
    assert set(out["allocation_state"].unique()) <= {"no_signal", "m3_zero", "m3_active"}
