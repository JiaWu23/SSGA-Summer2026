"""Align differently sampled / misaligned series onto a common timestamp grid.

Training: time-interpolate between known observations on the common calendar,
then forward-fill any remaining gaps within the train window.

Test / eval: forward-fill only — propagate the last known value until the next
observation. No interpolation that would blend in future releases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_series_to_index(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    train_end: pd.Timestamp | str | None = None,
    interpolate_train: bool = True,
) -> pd.Series:
    """
    Align a sparse or irregular series onto ``target_index``.

    Parameters
    ----------
    series
        Observation series (any frequency). Duplicate timestamps keep the last value.
    target_index
        Common calendar (e.g. weekly W-FRI market dates).
    train_end
        Inclusive end of the train window. Dates after this use forward-fill only.
        If None, the whole span uses forward-fill only (safe default for eval-only).
    interpolate_train
        If True and ``train_end`` is set, fill interior train gaps with time
        interpolation before forward-fill.
    """
    name = series.name
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        target = pd.DatetimeIndex(pd.to_datetime(target_index)).tz_localize(None)
        target = target.sort_values().unique()
        return pd.Series(np.nan, index=target, dtype=float, name=name)

    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()

    target = pd.DatetimeIndex(pd.to_datetime(target_index)).tz_localize(None)
    target = target.sort_values().unique()

    if train_end is None or not interpolate_train:
        return s.reindex(target.union(s.index)).ffill().reindex(target).rename(name)

    train_end_ts = pd.Timestamp(train_end)
    grid = target.union(s.index).sort_values()
    dense = s.reindex(grid)

    train_mask = dense.index <= train_end_ts
    train_dense = dense.loc[train_mask].copy()
    if int(train_dense.notna().sum()) >= 2:
        train_dense = train_dense.interpolate(method="time", limit_area="inside")
    train_dense = train_dense.ffill()

    out = dense.copy()
    out.loc[train_mask] = train_dense
    # Restore raw post-train observations, then ffill through eval timestamps.
    out = out.ffill()
    return out.reindex(target).rename(name)


def align_dataframe_to_index(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    train_end: pd.Timestamp | str | None = None,
    interpolate_train: bool = True,
) -> pd.DataFrame:
    """Column-wise ``align_series_to_index`` onto a shared target calendar."""
    if df.empty:
        target = pd.DatetimeIndex(pd.to_datetime(target_index)).tz_localize(None)
        target = target.sort_values().unique()
        return pd.DataFrame(index=target, columns=df.columns, dtype=float)

    aligned = {
        col: align_series_to_index(
            df[col],
            target_index,
            train_end=train_end,
            interpolate_train=interpolate_train,
        )
        for col in df.columns
    }
    return pd.DataFrame(aligned)


def align_long_macro_to_index(
    macro_long: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    train_end: pd.Timestamp | str | None = None,
    interpolate_train: bool = True,
    date_col: str = "date",
    series_col: str = "series",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Align a long-format macro panel (date, series, value) onto ``target_index``.

    Returns the same long schema with one row per (target date, series) after
    train interpolate / eval forward-fill.
    """
    if macro_long.empty:
        return macro_long.iloc[0:0].copy()

    wide = (
        macro_long.pivot(index=date_col, columns=series_col, values=value_col)
        .sort_index()
    )
    wide.index = pd.to_datetime(wide.index)
    aligned = align_dataframe_to_index(
        wide,
        target_index,
        train_end=train_end,
        interpolate_train=interpolate_train,
    )
    out = (
        aligned.rename_axis(date_col)
        .reset_index()
        .melt(id_vars=date_col, var_name=series_col, value_name=value_col)
        .dropna(subset=[value_col])
        .sort_values([date_col, series_col])
        .reset_index(drop=True)
    )
    return out
