"""M3 allocation-state diagnostics and mode comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.model_m3 import allocation_state, attach_m3_to_panel, compute_m3_size
from src.position_sizing import SizingMode, fit_ecdf


def _filter_period(
    panel: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        return panel
    dates = pd.to_datetime(panel.index.get_level_values("date"))
    mask = pd.Series(True, index=panel.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return panel[mask.values]


def m3_allocation_summary(panel: pd.DataFrame, *, period_label: str = "full") -> pd.DataFrame:
    if "allocation_state" not in panel.columns:
        return pd.DataFrame()
    counts = panel["allocation_state"].value_counts(normalize=False)
    total = len(panel)
    rows = []
    for state in ("no_signal", "m3_zero", "m3_active"):
        n = int(counts.get(state, 0))
        rows.append(
            {
                "period": period_label,
                "allocation_state": state,
                "count": n,
                "share": n / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def m3_rejection_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    candidates = df[df["M1_signal"] != 0].copy()
    if candidates.empty:
        return pd.DataFrame()
    rows = []
    for state in ("m3_zero", "m3_active"):
        sub = candidates[candidates["allocation_state"] == state]
        if sub.empty:
            continue
        rows.append(
            {
                "allocation_state": state,
                "n": len(sub),
                "mean_p_success": float(sub["p_success"].mean()) if "p_success" in sub.columns else float("nan"),
                "median_p_success": float(sub["p_success"].median()) if "p_success" in sub.columns else float("nan"),
                "mean_trade_return": float(sub["trade_return"].mean()) if "trade_return" in sub.columns else float("nan"),
                "hit_rate": float(sub["meta_label"].mean()) if "meta_label" in sub.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def m3_mode_comparison(panel: pd.DataFrame, cfg: PipelineConfig, train_proba: pd.Series | None) -> pd.DataFrame:
    """Compare allocation states across M3 sizing rules on the same M1+M2 inputs."""
    enriched = attach_m3_to_panel(panel, cfg, train_proba=train_proba)
    p = enriched["p_success"]
    threshold = cfg.m3.threshold or cfg.m2.threshold
    train_sorted = fit_ecdf(train_proba) if train_proba is not None and len(train_proba.dropna()) else None
    m1_sig = enriched["M1_signal"]
    rows = []
    for mode, col in [
        ("binary", "M3_size_binary"),
        ("linear", "M3_size_linear"),
        ("ecdf", "M3_size_ecdf"),
    ]:
        if col not in enriched.columns:
            sizes = compute_m3_size(p, mode, threshold=threshold, train_proba=train_proba, train_sorted=train_sorted)
        else:
            sizes = enriched[col]
        state = allocation_state(m1_sig, sizes)
        candidates = state[m1_sig != 0]
        n_cand = len(candidates)
        n_zero = int((candidates == "m3_zero").sum())
        n_active = int((candidates == "m3_active").sum())
        rows.append(
            {
                "m3_mode": mode,
                "m1_candidates": n_cand,
                "m3_zero_count": n_zero,
                "m3_active_count": n_active,
                "m3_zero_share": n_zero / n_cand if n_cand else float("nan"),
                "mean_m3_size_on_candidates": float(sizes[m1_sig != 0].mean()) if n_cand else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def save_m3_allocation_chart(summary: pd.DataFrame, output_dir: Path) -> str | None:
    if summary.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    test = summary[summary["period"] == "test"]
    plot_df = test if not test.empty else summary[summary["period"] == "full"]
    if plot_df.empty:
        plot_df = summary
    fig, ax = plt.subplots(figsize=(8, 4))
    states = plot_df["allocation_state"].tolist()
    shares = plot_df["share"].values * 100
    colors = {"no_signal": "#CCCCCC", "m3_zero": "#C44E52", "m3_active": "#55A868"}
    bar_colors = [colors.get(s, "#888888") for s in states]
    ax.bar(states, shares, color=bar_colors)
    ax.set_ylabel("Share of asset-weeks (%)")
    ax.set_title("M3 Allocation States (test period)")
    p = output_dir / "m3_allocation_states.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p.name


def run_m3_diagnostics(
    panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    cfg: PipelineConfig,
    output_dir: Path,
    *,
    train_proba: pd.Series | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        m3_allocation_summary(panel, period_label="full"),
        m3_allocation_summary(
            train_panel, period_label="train", 
        ),
        m3_allocation_summary(test_panel, period_label="test"),
    ]
    allocation = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    allocation.to_csv(output_dir / "m3_allocation_summary.csv", index=False)

    rejection = m3_rejection_analysis(test_panel)
    rejection.to_csv(output_dir / "m3_rejection_analysis.csv", index=False)

    mode_cmp = m3_mode_comparison(panel, cfg, train_proba)
    mode_cmp.to_csv(output_dir / "m3_mode_comparison.csv", index=False)

    chart = save_m3_allocation_chart(allocation, figures_dir)

    return {
        "allocation_summary": allocation,
        "rejection_analysis": rejection,
        "mode_comparison": mode_cmp,
        "chart": chart,
    }
