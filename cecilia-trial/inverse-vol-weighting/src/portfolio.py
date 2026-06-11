"""Portfolio weight construction with risk constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PortfolioConfig

WEEKS_PER_YEAR = 52

INV_VOL_COLUMN = "vol_26w"  # lagged trailing vol feature (already .shift(1) in feature_engineering)


def compute_budgets(df: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series | float:
    """Per-(date, ticker) capital budget for selected assets.

    "equal" (default): every pick gets base_budget_per_asset — returns the plain
    float so the baseline code path is bit-for-bit unchanged.

    "inverse_vol" (opt-in): each week, the total deployed budget is identical to
    the equal scheme (n_picks * base_budget_per_asset), but it is redistributed
    across that week's picks proportionally to 1 / trailing 26-week volatility,
    so calmer assets receive more capital and jumpier assets less (risk-balanced
    allocation; Maillard, Roncalli & Teiletche 2010). Uses the lagged vol_26w
    feature column, so no look-ahead is introduced. Falls back to equal shares
    on any week where vol data is missing.
    """
    scheme = getattr(cfg, "weighting_scheme", "equal")
    if scheme == "equal":
        return cfg.base_budget_per_asset
    if scheme != "inverse_vol":
        raise ValueError(f"Unknown weighting_scheme: {scheme!r} (use 'equal' or 'inverse_vol')")
    if INV_VOL_COLUMN not in df.columns:
        raise ValueError(
            f"weighting_scheme='inverse_vol' requires the '{INV_VOL_COLUMN}' feature column"
        )

    selected = df["M1_signal"].fillna(0) != 0
    floor = max(float(getattr(cfg, "inv_vol_floor_ann", 0.02)), 1e-6)
    inv_vol = 1.0 / df[INV_VOL_COLUMN].clip(lower=floor)
    inv_vol = inv_vol.where(selected, 0.0)

    grouped = inv_vol.groupby(level="date")
    share = inv_vol / grouped.transform("sum").replace(0.0, np.nan)
    n_picks = selected.groupby(level="date").transform("sum")
    # Equal-share fallback for weeks where every pick lacks vol data.
    share = share.fillna(1.0 / n_picks.replace(0, np.nan)).fillna(0.0)

    budgets = share * n_picks * cfg.base_budget_per_asset
    # Unselected rows keep the plain base budget (multiplied by signal=0 anyway).
    budgets = budgets.where(selected, cfg.base_budget_per_asset)
    return budgets.rename("budget")


def raw_weights(
    signals: pd.Series,
    sizes: pd.Series,
    base_budget: float,
    conviction: pd.Series | None = None,
) -> pd.Series:
    conv = conviction if conviction is not None else 1.0
    if isinstance(conv, pd.Series):
        return (signals * sizes * conv * base_budget).rename("raw_weight")
    return (signals * sizes * base_budget).rename("raw_weight")


def apply_constraints(
    weights: pd.Series,
    cfg: PortfolioConfig,
) -> pd.Series:
    w = weights.copy().astype(float)
    if not cfg.allow_short:
        w = w.clip(lower=0.0)
    w = w.clip(lower=-cfg.max_abs_asset_weight, upper=cfg.max_abs_asset_weight)
    gross = w.abs().sum()
    if gross > cfg.max_gross_exposure and gross > 0:
        w = w * (cfg.max_gross_exposure / gross)
    return w.rename("weight")


def apply_constraints_by_date(df: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    """Apply portfolio constraints to each date's cross-section of raw weights."""
    out = []
    for _date, grp in df.groupby(level="date"):
        w = apply_constraints(grp["raw_weight"], cfg)
        w.index = grp.index
        out.append(w)
    return pd.concat(out).sort_index()


def build_weights_panel(
    panel: pd.DataFrame,
    cfg: PortfolioConfig,
    sizes: pd.Series,
    conviction: pd.Series | None = None,
) -> pd.DataFrame:
    out = panel.copy()
    out["size"] = sizes.reindex(out.index)
    if conviction is not None:
        out["M1_conviction"] = conviction.reindex(out.index).fillna(0.0)
    conv = out["M1_conviction"] if "M1_conviction" in out.columns else None
    out["raw_weight"] = raw_weights(out["M1_signal"], out["size"], compute_budgets(out, cfg), conviction=conv)
    out["weight"] = apply_constraints_by_date(out, cfg)
    return out


def weights_to_wide(weights_long: pd.DataFrame) -> pd.DataFrame:
    df = weights_long.reset_index()
    return df.pivot(index="date", columns="ticker", values="weight").fillna(0.0).sort_index()


def build_weights_from_signals(
    panel: pd.DataFrame,
    signals: pd.Series,
    *,
    conviction: pd.Series | None = None,
    portfolio_cfg: PortfolioConfig,
    m2_sizes: pd.Series | None = None,
) -> pd.DataFrame:
    """Build wide weight matrix from M1 signals (used in tuning and backtest)."""
    df = panel.reset_index().copy() if not isinstance(panel.index, pd.MultiIndex) else panel.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"])
    df["M1_signal"] = signals.reindex(df.index).fillna(0)
    sizes = m2_sizes.reindex(df.index).fillna(1.0) if m2_sizes is not None else pd.Series(1.0, index=df.index)
    df["size"] = sizes
    if conviction is not None:
        df["M1_conviction"] = conviction.reindex(df.index).fillna(0.0)
    conv = df["M1_conviction"] if "M1_conviction" in df.columns else None
    df["raw_weight"] = raw_weights(df["M1_signal"], df["size"], compute_budgets(df, portfolio_cfg), conviction=conv)
    df["weight"] = apply_constraints_by_date(df, portfolio_cfg)
    return weights_to_wide(df.reset_index())


def apply_vol_target_wide(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    """Scale gross exposure to hit annualized vol target using trailing realized vol."""
    target = cfg.vol_target_ann
    if target is None or target <= 0:
        return weights

    lookback = max(4, int(cfg.vol_target_lookback_weeks))
    aligned_ret = returns_wide.reindex(weights.index).fillna(0.0)
    w = weights.reindex(aligned_ret.index).ffill().fillna(0.0)
    port_ret = (w.shift(1) * aligned_ret).sum(axis=1)
    realized_vol = port_ret.rolling(lookback, min_periods=8).std() * np.sqrt(WEEKS_PER_YEAR)
    scale = (target / realized_vol).clip(upper=cfg.vol_target_max_scale)
    scale = scale.fillna(1.0).replace([np.inf, -np.inf], 1.0)

    scaled = w.mul(scale, axis=0)
    out = []
    for date in scaled.index:
        row = scaled.loc[date]
        gross = row.abs().sum()
        if gross > cfg.max_gross_exposure and gross > 0:
            row = row * (cfg.max_gross_exposure / gross)
        out.append(row)
    return pd.DataFrame(out, index=scaled.index, columns=scaled.columns)
