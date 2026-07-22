"""Tests for multi-frequency time alignment (interpolate train / ffill eval)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.time_alignment import align_dataframe_to_index, align_series_to_index


def test_monthly_to_weekly_train_interpolates_interior():
    # Two month-end observations; weekly Fridays between should be interpolated on train.
    s = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="CPI",
    )
    target = pd.date_range("2020-01-31", "2020-02-28", freq="W-FRI")
    aligned = align_series_to_index(
        s, target, train_end="2020-12-31", interpolate_train=True
    )
    assert aligned.notna().all()
    # Midpoint week should sit between 100 and 110 (not stuck at 100 via ffill-only).
    mid = aligned.iloc[len(aligned) // 2]
    assert 100.0 < float(mid) < 110.0


def test_eval_forward_fills_without_using_next_observation():
    # Observation at start of eval and a later jump; weeks before the jump must stay at 100.
    s = pd.Series(
        [100.0, 200.0],
        index=pd.to_datetime(["2021-01-01", "2021-03-01"]),
        name="CPI",
    )
    target = pd.date_range("2021-01-01", "2021-02-26", freq="W-FRI")
    aligned = align_series_to_index(
        s, target, train_end="2020-12-31", interpolate_train=True
    )
    # Entire pre-March window is after train_end → ffill only from 100.
    assert (aligned == 100.0).all()
    # Confirm the next observation is not blended into earlier eval weeks.
    assert float(aligned.iloc[-1]) == 100.0


def test_ffill_only_when_interpolate_disabled():
    s = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="CPI",
    )
    target = pd.date_range("2020-01-31", "2020-02-28", freq="W-FRI")
    aligned = align_series_to_index(
        s, target, train_end="2020-12-31", interpolate_train=False
    )
    # Without interpolate, weeks after Jan stay at 100 until Feb observation week.
    assert float(aligned.iloc[0]) == 100.0
    # Last Friday before Feb 29 week should still be 100 if Feb obs maps later.
    assert aligned.nunique(dropna=True) <= 2


def test_align_dataframe_columnwise():
    df = pd.DataFrame(
        {
            "A": pd.Series([1.0, 3.0], index=pd.to_datetime(["2020-01-31", "2020-03-31"])),
            "B": pd.Series([10.0, 30.0], index=pd.to_datetime(["2020-01-31", "2020-03-31"])),
        }
    )
    target = pd.date_range("2020-01-31", "2020-03-27", freq="W-FRI")
    out = align_dataframe_to_index(df, target, train_end="2020-12-31", interpolate_train=True)
    assert list(out.columns) == ["A", "B"]
    assert len(out) == len(target)
    assert out["A"].notna().sum() > 2
    mid_a = float(out["A"].iloc[len(out) // 2])
    assert 1.0 < mid_a < 3.0


def test_leading_nans_before_first_observation():
    s = pd.Series([5.0], index=pd.to_datetime(["2020-03-01"]), name="X")
    target = pd.date_range("2020-01-03", "2020-03-06", freq="W-FRI")
    aligned = align_series_to_index(s, target, train_end="2020-12-31")
    assert np.isnan(aligned.iloc[0])
    assert float(aligned.dropna().iloc[0]) == 5.0
