"""Market regime timeline, transitions, and performance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.diagnostics import (
    annualized_return,
    compute_ic,
    m2_classification_metrics,
    sharpe_ratio,
    strategy_metrics_on_period,
)

REGIME_FLAG_COLS = ["risk_off", "curve_inverted", "inflation_up", "growth_down"]
REGIME_LEVEL_COLS = ["vix_level", "credit_stress", "yield_curve", "growth_trend", "inflation_trend"]


def _panel_to_date_level(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per date with regime features (broadcast across tickers)."""
    df = panel.reset_index()
    regime_cols = [c for c in REGIME_FLAG_COLS + REGIME_LEVEL_COLS if c in df.columns]
    if not regime_cols:
        return pd.DataFrame()
    return df.groupby("date", as_index=True)[regime_cols].first().sort_index()


def build_regime_timeline(panel: pd.DataFrame) -> pd.DataFrame:
    timeline = _panel_to_date_level(panel)
    if timeline.empty:
        return timeline
    if "vix_level" in timeline.columns:
        vix = timeline["vix_level"].dropna()
        if len(vix) >= 4:
            q25, q50, q75 = vix.quantile([0.25, 0.5, 0.75])
            timeline["vix_quartile"] = pd.cut(
                timeline["vix_level"],
                bins=[-np.inf, q25, q50, q75, np.inf],
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            )
    return timeline.reset_index()


def regime_transition_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    """Count on/off flips and average spell duration for binary regime flags."""
    if timeline.empty:
        return pd.DataFrame()
    df = timeline.set_index("date") if "date" in timeline.columns else timeline
    rows = []
    for col in REGIME_FLAG_COLS:
        if col not in df.columns:
            continue
        s = df[col].fillna(0).astype(int)
        flips = int((s.diff().abs() == 1).sum())
        spells_on = []
        spells_off = []
        current_val = None
        current_len = 0
        for val in s:
            if val == current_val:
                current_len += 1
            else:
                if current_val is not None and current_len > 0:
                    (spells_on if current_val == 1 else spells_off).append(current_len)
                current_val = val
                current_len = 1
        if current_val is not None and current_len > 0:
            (spells_on if current_val == 1 else spells_off).append(current_len)
        rows.append(
            {
                "regime_flag": col,
                "n_transitions": flips,
                "pct_on": float(s.mean()),
                "avg_spell_on_weeks": float(np.mean(spells_on)) if spells_on else float("nan"),
                "avg_spell_off_weeks": float(np.mean(spells_off)) if spells_off else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _returns_by_regime_mask(
    returns: pd.Series,
    timeline: pd.DataFrame,
    flag_col: str,
    flag_value: int,
) -> pd.Series:
    tl = timeline.set_index("date") if "date" in timeline.columns else timeline
    if flag_col not in tl.columns:
        return pd.Series(dtype=float)
    mask_dates = tl.index[tl[flag_col].fillna(0).astype(int) == flag_value]
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    return r[r.index.isin(mask_dates)]


def performance_by_regime(
    results: dict[str, BacktestResult],
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Strategy metrics when each binary regime flag is on vs off."""
    rows = []
    strategy_keys = [k for k in results if k in ("m1_only", "m1_m2_m3_ecdf", "m1_m2_ecdf", "equal_weight_1_7")]
    for flag in REGIME_FLAG_COLS:
        if flag not in timeline.columns:
            continue
        for val, label in [(1, "on"), (0, "off")]:
            for strat in strategy_keys:
                rets = _returns_by_regime_mask(results[strat].returns, timeline, flag, val)
                if rets.empty or len(rets) < 4:
                    continue
                rows.append(
                    {
                        "regime_flag": flag,
                        "regime_state": label,
                        "strategy": strat,
                        "n_weeks": len(rets),
                        "annualized_return": annualized_return(rets),
                        "sharpe": sharpe_ratio(rets),
                        "hit_rate": float((rets > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def m1_ic_by_regime(test_panel: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tl = timeline.set_index("date") if "date" in timeline.columns else timeline
    for flag in REGIME_FLAG_COLS:
        if flag not in tl.columns:
            continue
        for val, label in [(1, "on"), (0, "off")]:
            dates = tl.index[tl[flag].fillna(0).astype(int) == val]
            sub = test_panel.reset_index()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub[sub["date"].isin(dates)].set_index(["date", "ticker"])
            if sub.empty:
                continue
            ic = compute_ic(sub)
            rows.append(
                {
                    "regime_flag": flag,
                    "regime_state": label,
                    "ic_mean": float(ic.mean()) if not ic.empty else float("nan"),
                    "n_weeks": int(len(ic)),
                }
            )
    return pd.DataFrame(rows)


def m2_auc_by_regime(test_panel: pd.DataFrame, timeline: pd.DataFrame, threshold: float = 0.55) -> pd.DataFrame:
    rows = []
    tl = timeline.set_index("date") if "date" in timeline.columns else timeline
    for flag in REGIME_FLAG_COLS:
        if flag not in tl.columns:
            continue
        for val, label in [(1, "on"), (0, "off")]:
            dates = tl.index[tl[flag].fillna(0).astype(int) == val]
            sub = test_panel.reset_index()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub[sub["date"].isin(dates)].set_index(["date", "ticker"])
            if sub.empty or "meta_label" not in sub.columns:
                continue
            m = m2_classification_metrics(sub["meta_label"], sub["p_success"], threshold=threshold)
            if not m:
                continue
            rows.append(
                {
                    "regime_flag": flag,
                    "regime_state": label,
                    "auc": m.get("auc", float("nan")),
                    "n_trades": int(sub["meta_label"].notna().sum()),
                    "base_rate": float(sub["meta_label"].mean()),
                }
            )
    return pd.DataFrame(rows)


def regime_market_context_table(
    panel: pd.DataFrame,
    *,
    train_end: str,
    test_start: str,
) -> pd.DataFrame:
    """Summary stats of macro/VIX features for train vs test."""
    timeline = build_regime_timeline(panel)
    if timeline.empty:
        return pd.DataFrame()
    tl = timeline.set_index("date")
    cols = [c for c in REGIME_LEVEL_COLS + REGIME_FLAG_COLS if c in tl.columns]
    rows = []
    for period, mask in [
        ("train", tl.index <= pd.Timestamp(train_end)),
        ("test", tl.index >= pd.Timestamp(test_start)),
    ]:
        sub = tl.loc[mask, cols]
        for col in cols:
            s = sub[col].dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "period": period,
                    "feature": col,
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "pct_on": float((s > 0).mean()) if col in REGIME_FLAG_COLS else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def save_regime_charts(
    timeline: pd.DataFrame,
    perf_by_regime: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    if timeline.empty:
        return saved

    tl = timeline.set_index("date") if "date" in timeline.columns else timeline

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    if "vix_level" in tl.columns:
        axes[0].plot(tl.index, tl["vix_level"], color="#C44E52", linewidth=1)
        axes[0].set_ylabel("VIX")
        axes[0].set_title("VIX Level")
        axes[0].grid(True, alpha=0.3)

    for i, flag in enumerate(REGIME_FLAG_COLS[:2]):
        if flag in tl.columns:
            axes[i + 1].fill_between(tl.index, 0, tl[flag].fillna(0), alpha=0.5, step="mid")
            axes[i + 1].set_ylabel(flag)
            axes[i + 1].set_ylim(-0.1, 1.1)
            axes[i + 1].grid(True, alpha=0.3)
    p = output_dir / "vix_and_flags.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    flag_cols = [c for c in REGIME_FLAG_COLS if c in tl.columns]
    if flag_cols:
        fig, ax = plt.subplots(figsize=(12, 3))
        for col in flag_cols:
            ax.plot(tl.index, tl[col].fillna(0) + flag_cols.index(col) * 0.05, label=col, drawstyle="steps-post")
        ax.legend(fontsize=8, ncol=4)
        ax.set_title("Regime Flags Over Time")
        ax.set_ylabel("Flag (offset)")
        p = output_dir / "regime_timeline.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if not perf_by_regime.empty:
        pivot = perf_by_regime.pivot_table(
            index="regime_flag",
            columns=["strategy", "regime_state"],
            values="sharpe",
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.5)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels([f"{a}_{b}" for a, b in pivot.columns], rotation=45, ha="right")
            ax.set_yticklabels(pivot.index)
            ax.set_title("Sharpe by Regime Flag and Strategy")
            fig.colorbar(im, ax=ax)
            p = output_dir / "performance_by_regime_heatmap.png"
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved.append(p.name)

    return saved


def run_regime_analysis(
    panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    results: dict[str, BacktestResult],
    output_dir: Path,
    *,
    cfg_train_end: str,
    cfg_test_start: str,
    m2_threshold: float = 0.55,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    timeline = build_regime_timeline(panel)
    timeline.to_csv(output_dir / "regime_timeline.csv", index=False)

    transitions = regime_transition_summary(timeline)
    transitions.to_csv(output_dir / "regime_transitions.csv", index=False)

    perf = performance_by_regime(results, timeline)
    perf.to_csv(output_dir / "performance_by_regime.csv", index=False)

    ic_regime = m1_ic_by_regime(test_panel, timeline)
    ic_regime.to_csv(output_dir / "m1_ic_by_regime.csv", index=False)

    auc_regime = m2_auc_by_regime(test_panel, timeline, threshold=m2_threshold)
    auc_regime.to_csv(output_dir / "m2_auc_by_regime.csv", index=False)

    context = regime_market_context_table(panel, train_end=cfg_train_end, test_start=cfg_test_start)
    context.to_csv(output_dir / "regime_market_context.csv", index=False)

    charts = save_regime_charts(timeline, perf, figures_dir)

    return {
        "regime_timeline": timeline,
        "regime_transitions": transitions,
        "performance_by_regime": perf,
        "m1_ic_by_regime": ic_regime,
        "m2_auc_by_regime": auc_regime,
        "regime_market_context": context,
        "charts": charts,
    }
