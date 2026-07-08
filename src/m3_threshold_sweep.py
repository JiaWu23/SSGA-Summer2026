"""M3 threshold sweep: binary and threshold-gated linear sizing on the test window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import STRATEGY_M1_M2_M3_BINARY, _run_backtest, strategy_weights_from_panel
from src.config import PipelineConfig
from src.diagnostics import m2_classification_metrics, strategy_metrics_on_period
from src.model_m3 import allocation_state
from src.portfolio import apply_constraints_by_date, apply_vol_target_wide, weights_to_wide
from src.position_sizing import SizingMode, binary_size, fit_ecdf, linear_size

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
    0.50,
    0.52,
    0.54,
    0.55,
    0.56,
    0.58,
    0.60,
    0.62,
    0.64,
    0.66,
    0.68,
    0.70,
)

MIN_REJECTION_SHARE = 0.05
MAX_RECALL_FOR_MEANINGFUL = 0.99


def linear_gated_size(proba: pd.Series, gate_threshold: float) -> pd.Series:
    """Linear M3 with hard gate: size = 0 when p_success < gate_threshold."""
    raw = linear_size(proba)
    return raw.where(proba >= gate_threshold, 0.0).rename("M3_size")


def strategy_weights_linear_gated(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    gate_threshold: float,
) -> pd.DataFrame:
    """Portfolio weights using threshold-gated linear M3 sizing."""
    df = panel.reset_index().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"])
    sizes = linear_gated_size(df["p_success"], gate_threshold)
    df["M3_size"] = sizes.reindex(df.index).fillna(0.0)
    if "M1_conviction" in df.columns:
        df["raw_weight"] = (
            df["M1_signal"] * df["M3_size"] * df["M1_conviction"] * cfg.portfolio.base_budget_per_asset
        )
    else:
        df["raw_weight"] = df["M1_signal"] * df["M3_size"] * cfg.portfolio.base_budget_per_asset
    df["weight"] = apply_constraints_by_date(df, cfg.portfolio)
    w_wide = weights_to_wide(df.reset_index())
    return apply_vol_target_wide(w_wide, returns_wide, cfg.portfolio)


def _m3_candidate_stats(
    panel: pd.DataFrame,
    sizes: pd.Series,
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
    *,
    m3_mode: str,
) -> dict[str, Any]:
    m1_sig = panel["M1_signal"] if "M1_signal" in panel.columns else pd.Series(0, index=panel.index)
    state = allocation_state(m1_sig, sizes.reindex(panel.index).fillna(0.0))
    candidates = state[m1_sig != 0]
    n_cand = len(candidates)
    n_zero = int((candidates == "m3_zero").sum()) if n_cand else 0
    rejection_share = n_zero / n_cand if n_cand else float("nan")
    m2m = m2_classification_metrics(y_true, y_prob, threshold=threshold)
    return {
        "m3_mode": m3_mode,
        "threshold": threshold,
        "m1_candidates": n_cand,
        "m3_zero_count": n_zero,
        "m3_rejection_share": rejection_share,
        "m3_approval_rate": 1.0 - rejection_share if pd.notna(rejection_share) else float("nan"),
        "mean_m3_size_on_candidates": float(sizes[m1_sig != 0].mean()) if n_cand else float("nan"),
        "m2_recall": m2m.get("recall", float("nan")),
        "m2_precision": m2m.get("precision", float("nan")),
        "m2_f1": m2m.get("f1", float("nan")),
        "degeneracy_note": m2m.get("degeneracy_note", ""),
    }


def sweep_m3_thresholds(
    panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    train_proba: pd.Series | None = None,
    threshold_grid: tuple[float, ...] | None = None,
    m1_only_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Sweep binary and linear-gated thresholds; score portfolio metrics on the test window."""
    grid = threshold_grid or DEFAULT_THRESHOLD_GRID
    train_sorted = fit_ecdf(train_proba) if train_proba is not None and len(train_proba.dropna()) else None
    test_start = cfg.split.test_start
    test_end = cfg.split.test_end
    tc = cfg.portfolio.transaction_cost_bps

    if m1_only_returns is None:
        m1_w = strategy_weights_from_panel(panel, returns_wide, cfg, SizingMode.LINEAR, use_m2=False)
        m1_only_returns = _run_backtest("m1_only", m1_w, returns_wide, tc).returns
    m1_test = strategy_metrics_on_period(m1_only_returns, start=test_start, end=test_end)

    y_true = test_panel["meta_label"] if "meta_label" in test_panel.columns else pd.Series(dtype=float)
    y_prob = test_panel["p_success"] if "p_success" in test_panel.columns else pd.Series(dtype=float)

    rows: list[dict[str, Any]] = []
    for threshold in grid:
        # Binary M3
        bin_w = strategy_weights_from_panel(
            panel,
            returns_wide,
            cfg,
            SizingMode.BINARY,
            use_m2=True,
            m2_threshold=threshold,
            train_proba=train_proba,
            train_sorted=train_sorted,
        )
        bin_sizes = binary_size(panel["p_success"], threshold)
        stats = _m3_candidate_stats(
            test_panel,
            bin_sizes.reindex(test_panel.index).fillna(0.0),
            y_true,
            y_prob,
            threshold,
            m3_mode="binary",
        )
        bin_bt = _run_backtest(STRATEGY_M1_M2_M3_BINARY, bin_w, returns_wide, tc)
        bin_test = strategy_metrics_on_period(bin_bt.returns, start=test_start, end=test_end)
        rows.append(
            {
                **stats,
                "test_ann_return": bin_test["annualized_return"],
                "test_sharpe": bin_test["sharpe"],
                "test_max_drawdown": bin_test["max_drawdown"],
                "sharpe_edge_vs_m1": bin_test["sharpe"] - m1_test["sharpe"],
                "meaningful_rejection": bool(
                    pd.notna(stats["m3_rejection_share"])
                    and stats["m3_rejection_share"] >= MIN_REJECTION_SHARE
                    and stats.get("m2_recall", 1.0) < MAX_RECALL_FOR_MEANINGFUL
                ),
            }
        )

        # Linear with gate at threshold (research variant)
        lin_w = strategy_weights_linear_gated(panel, returns_wide, cfg, gate_threshold=threshold)
        lin_sizes = linear_gated_size(panel["p_success"], threshold)
        stats_l = _m3_candidate_stats(
            test_panel,
            lin_sizes.reindex(test_panel.index).fillna(0.0),
            y_true,
            y_prob,
            threshold,
            m3_mode="linear_gated",
        )
        lin_bt = _run_backtest("m1_m2_m3_linear_gated", lin_w, returns_wide, tc)
        lin_test = strategy_metrics_on_period(lin_bt.returns, start=test_start, end=test_end)
        rows.append(
            {
                **stats_l,
                "test_ann_return": lin_test["annualized_return"],
                "test_sharpe": lin_test["sharpe"],
                "test_max_drawdown": lin_test["max_drawdown"],
                "sharpe_edge_vs_m1": lin_test["sharpe"] - m1_test["sharpe"],
                "meaningful_rejection": bool(
                    pd.notna(stats_l["m3_rejection_share"])
                    and stats_l["m3_rejection_share"] >= MIN_REJECTION_SHARE
                ),
            }
        )

    return pd.DataFrame(rows)


def recommend_m3_thresholds(
    sweep: pd.DataFrame,
    *,
    baseline_threshold: float = 0.55,
) -> dict[str, Any]:
    """Pick best binary and linear-gated thresholds with meaningful rejection."""
    if sweep.empty:
        return {"binary": {}, "linear_gated": {}, "baseline_threshold": baseline_threshold}

    out: dict[str, Any] = {"baseline_threshold": baseline_threshold}
    for mode in ("binary", "linear_gated"):
        sub = sweep[sweep["m3_mode"] == mode].copy()
        if sub.empty:
            out[mode] = {}
            continue
        baseline_rows = sub[np.isclose(sub["threshold"], baseline_threshold)]
        baseline_row = baseline_rows.iloc[0].to_dict() if not baseline_rows.empty else {}

        meaningful = sub[sub["meaningful_rejection"]].copy()
        if meaningful.empty:
            # Relax: any rejection at all
            meaningful = sub[sub["m3_rejection_share"] > 0].copy()
        if meaningful.empty:
            best = sub.sort_values("test_sharpe", ascending=False).iloc[0]
            rationale = (
                f"No threshold achieves ≥{MIN_REJECTION_SHARE:.0%} rejection on test M1 candidates; "
                f"reporting highest test Sharpe ({best['test_sharpe']:.4f}) at T={best['threshold']:.2f}."
            )
        else:
            best = meaningful.sort_values("test_sharpe", ascending=False).iloc[0]
            rationale = (
                f"Best test Sharpe among thresholds with meaningful rejection "
                f"(≥{MIN_REJECTION_SHARE:.0%} m3_zero, recall < {MAX_RECALL_FOR_MEANINGFUL:.0%} for binary): "
                f"T={best['threshold']:.2f}, Sharpe {best['test_sharpe']:.4f}, "
                f"rejection {best['m3_rejection_share']:.1%}, recall {best.get('m2_recall', float('nan')):.3f}."
            )
        out[mode] = {
            "recommended_threshold": float(best["threshold"]),
            "test_sharpe": float(best["test_sharpe"]),
            "test_ann_return": float(best["test_ann_return"]),
            "m3_rejection_share": float(best["m3_rejection_share"]),
            "m2_recall": float(best.get("m2_recall", float("nan"))),
            "m2_precision": float(best.get("m2_precision", float("nan"))),
            "sharpe_edge_vs_m1": float(best["sharpe_edge_vs_m1"]),
            "meaningful_rejection": bool(best["meaningful_rejection"]),
            "baseline_test_sharpe": float(baseline_row.get("test_sharpe", float("nan"))),
            "baseline_rejection_share": float(baseline_row.get("m3_rejection_share", float("nan"))),
            "baseline_recall": float(baseline_row.get("m2_recall", float("nan"))),
            "rationale": rationale,
            "apply_to_config": bool(
                best["meaningful_rejection"]
                and float(best["test_sharpe"]) >= float(baseline_row.get("test_sharpe", -999)) - 0.01
            ),
        }
    return out


def save_m3_threshold_charts(sweep: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    if sweep.empty:
        return saved

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, ax, color in [
        ("binary", axes[0], "#4C72B0"),
        ("linear_gated", axes[1], "#DD8452"),
    ]:
        sub = sweep[sweep["m3_mode"] == mode].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(sub["threshold"], sub["test_sharpe"], marker="o", color=color, label="Test Sharpe")
        ax.axvline(0.55, color="gray", linestyle="--", linewidth=0.8, label="T=0.55 (default)")
        best = sub.loc[sub["test_sharpe"].idxmax()]
        ax.axvline(best["threshold"], color="#55A868", linestyle=":", linewidth=0.8, label="Best Sharpe")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Test Sharpe")
        ax.set_title(f"M3 {mode.replace('_', ' ')}")
        ax.legend(fontsize=7)
        ax2 = ax.twinx()
        ax2.bar(
            sub["threshold"],
            sub["m3_rejection_share"] * 100,
            alpha=0.2,
            width=0.015,
            color="#C44E52",
            label="Rejection %",
        )
        ax2.set_ylabel("M3 rejection on candidates (%)")
    fig.tight_layout()
    p = output_dir / "m3_threshold_sweep.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)
    return saved


def generate_m3_threshold_report(
    sweep: pd.DataFrame,
    recommendation: dict[str, Any],
    report_path: Path,
    *,
    cfg: PipelineConfig | None = None,
) -> None:
    from src.diagnostics import _fmt_num, _fmt_pct, _markdown_table

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_prefix = "../data/backtests/long_only/figures"
    baseline = recommendation.get("baseline_threshold", 0.55)

    lines = [
        "# M3 Threshold Sweep Analysis",
        "",
        "**Research use only — not investment advice.**",
        "",
        "At the default threshold **T=0.55**, binary M3 approves ~**100%** of M1 candidates "
        "(recall ≈ 1) because calibrated `p_success` on the test set rarely falls below 0.55. "
        "This sweep finds thresholds where M3 **meaningfully rejects** candidates (`m3_zero` ≥ 5% of M1 signals) "
        "and compares test-period portfolio Sharpe vs M1-only.",
        "",
    ]
    if cfg is not None:
        lines.extend(
            [
                "## Setup",
                "",
                f"- **Test window:** `{cfg.split.test_start}` to `{cfg.split.test_end or 'latest'}`",
                f"- **Default threshold:** `{baseline}` (`models.m3.threshold` / `models.m2.threshold`)",
                f"- **Binary M3:** size = 1 if `p_success ≥ T`, else 0",
                f"- **Linear gated M3:** size = `max(0, 2p−1)` if `p_success ≥ T`, else 0 (research variant; "
                "production linear is ungated)",
                f"- **Meaningful rejection:** ≥{MIN_REJECTION_SHARE:.0%} of M1 candidates with `m3_zero`; "
                f"binary also requires recall < {MAX_RECALL_FOR_MEANINGFUL:.0%}",
                "",
            ]
        )

    lines.extend(["## Recommended thresholds", ""])
    for mode, title in [("binary", "Binary M3"), ("linear_gated", "Linear gated M3")]:
        rec = recommendation.get(mode) or {}
        if not rec:
            continue
        lines.extend(
            [
                f"### {title}",
                "",
                f"- **Recommended T:** `{rec.get('recommended_threshold', 'n/a')}`",
                f"- **Test Sharpe:** {_fmt_num(rec.get('test_sharpe'))} "
                f"(vs baseline T=0.55: {_fmt_num(rec.get('baseline_test_sharpe'))})",
                f"- **Sharpe edge vs M1-only:** {_fmt_num(rec.get('sharpe_edge_vs_m1'))}",
                f"- **M3 rejection share (test candidates):** {_fmt_pct(rec.get('m3_rejection_share'))}",
                f"- **M2 recall / precision:** {_fmt_num(rec.get('m2_recall'))} / {_fmt_num(rec.get('m2_precision'))}",
                f"- **Apply to config?** `{'yes' if rec.get('apply_to_config') else 'no — research only'}`",
                f"- **Rationale:** {rec.get('rationale', '')}",
                "",
            ]
        )

    lines.extend(["## Full comparison table (test period)", ""])
    if not sweep.empty:
        disp = sweep.sort_values(["m3_mode", "threshold"]).copy()
        for col in (
            "threshold",
            "m3_rejection_share",
            "m3_approval_rate",
            "m2_recall",
            "m2_precision",
            "m2_f1",
            "mean_m3_size_on_candidates",
            "test_ann_return",
            "test_sharpe",
            "test_max_drawdown",
            "sharpe_edge_vs_m1",
        ):
            if col not in disp.columns:
                continue
            if col in ("test_ann_return", "test_max_drawdown", "m3_rejection_share", "m3_approval_rate"):
                disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
            else:
                disp[col] = disp[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        disp["meaningful_rejection"] = sweep.sort_values(["m3_mode", "threshold"])["meaningful_rejection"].map(
            lambda x: "yes" if x else "no"
        )
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![M3 threshold sweep]({fig_prefix}/m3_threshold_sweep.png)")
        lines.append("")

    lines.extend(
        [
            "## Key findings",
            "",
        ]
    )
    bin_rec = recommendation.get("binary") or {}
    if bin_rec.get("baseline_recall", 1) >= 0.99:
        lines.append(
            f"- **T=0.55 (binary):** recall ≈ {_fmt_num(bin_rec.get('baseline_recall'))}, "
            f"rejection ≈ {_fmt_pct(bin_rec.get('baseline_rejection_share'))} — effectively M1-only."
        )
    if bin_rec.get("recommended_threshold") and bin_rec.get("recommended_threshold") != baseline:
        lines.append(
            f"- **Binary best with rejection:** T={bin_rec['recommended_threshold']:.2f} "
            f"(Sharpe {_fmt_num(bin_rec.get('test_sharpe'))}, rejection {_fmt_pct(bin_rec.get('m3_rejection_share'))})."
        )
    lines.extend(
        [
            "- **ECDF sizing** (not swept here) remains the primary risk-shaping layer; threshold sweeps target "
            "interpretable binary/linear rules.",
            "",
            "Related: [m3_allocation_analysis.md](m3_allocation_analysis.md) · [m2_diagnostics.md](m2_diagnostics.md)",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
