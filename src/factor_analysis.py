"""M1 factor-level performance, correlation, and sleeve attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestResult, _run_backtest
from src.config import PipelineConfig
from src.diagnostics import annualized_return, compute_ic, max_drawdown, sharpe_ratio, strategy_metrics
from src.model_m1 import RuleBasedM1

FACTOR_COLS = ["momentum_score", "trend_score", "macro_score", "risk_penalty"]

FACTOR_TO_WEIGHT_KEY = {
    "momentum_score": "momentum",
    "trend_score": "trend",
    "macro_score": "macro",
    "risk_penalty": "risk_penalty",
}

UNDERLYING_M1_FEATURES = [
    "z_mom_12w",
    "z_mom_26w",
    "z_mom_52w",
    "rank_mom_12w",
    "rel_mom_12w",
    "z_trend_signal",
    "z_vol_12w",
    "drawdown_26w",
    "z_drawdown_26w",
    "growth_trend",
    "risk_off",
    "inflation_up",
    "curve_inverted",
    "credit_stress",
]


def _ensure_panel_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Restore MultiIndex names so M1 top-K groups by date correctly."""
    if isinstance(panel.index, pd.MultiIndex):
        names = list(panel.index.names)
        if names[0] is None and len(names) >= 2:
            panel = panel.copy()
            panel.index = panel.index.set_names(["date", "ticker"])
        return panel
    if "date" in panel.columns and "ticker" in panel.columns:
        return panel.set_index(["date", "ticker"]).sort_index()
    return panel


def _filter_panel_period(
    panel: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = panel.copy()
    if not isinstance(df.index, pd.MultiIndex):
        return df
    dates = pd.to_datetime(df.index.get_level_values("date"))
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return df[mask.values]


def composite_score_from_components(
    comps: pd.DataFrame,
    weights: dict[str, float],
    *,
    technical_blend: tuple[float, float] | None = None,
) -> pd.Series:
    """Build M1 composite score from component columns and weight dict."""
    if technical_blend is not None:
        mom_share, trend_share = technical_blend
        technical = mom_share * comps["momentum_score"] + trend_share * comps["trend_score"]
        w_technical = weights.get("momentum", 0.0) + weights.get("trend", 0.0)
        score = (
            w_technical * technical
            + weights.get("macro", 0.0) * comps["macro_score"]
            - weights.get("risk_penalty", 0.0) * comps["risk_penalty"]
        )
    else:
        score = (
            weights.get("momentum", 0.0) * comps["momentum_score"]
            + weights.get("trend", 0.0) * comps["trend_score"]
            + weights.get("macro", 0.0) * comps["macro_score"]
            - weights.get("risk_penalty", 0.0) * comps["risk_penalty"]
        )
    return score.rename("M1_score")


def ic_proportional_weights(
    factor_ic: pd.DataFrame,
    *,
    period: str = "train",
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    """Normalize non-negative train IC into factor weights summing to 1."""
    fallback = fallback or {"momentum": 0.45, "trend": 0.25, "macro": 0.20, "risk_penalty": 0.10}
    if factor_ic.empty:
        return dict(fallback)
    sub = factor_ic[(factor_ic["period"] == period) & (factor_ic["factor"].isin(FACTOR_COLS))]
    raw: dict[str, float] = {}
    for _, row in sub.iterrows():
        key = FACTOR_TO_WEIGHT_KEY.get(str(row["factor"]))
        if key:
            raw[key] = max(float(row["ic_mean"]), 0.0)
    total = sum(raw.values())
    if total <= 0:
        return dict(fallback)
    return {k: v / total for k, v in raw.items()}


def ic_technical_blend(factor_ic: pd.DataFrame, *, period: str = "train") -> tuple[float, float]:
    """IC-weighted momentum/trend blend inside a merged technical bucket."""
    if factor_ic.empty:
        return (0.35, 0.65)
    sub = factor_ic[(factor_ic["period"] == period) & (factor_ic["factor"].isin(["momentum_score", "trend_score"]))]
    ic_map = {str(r["factor"]): max(float(r["ic_mean"]), 0.0) for _, r in sub.iterrows()}
    mom = ic_map.get("momentum_score", 0.0)
    trend = ic_map.get("trend_score", 0.0)
    total = mom + trend
    if total <= 0:
        return (0.35, 0.65)
    return (mom / total, trend / total)


def _weight_grid(step: float = 0.10) -> list[dict[str, float]]:
    """Coarse weight combinations that sum to 1.0 (risk_penalty is subtracted in score)."""
    grid: list[dict[str, float]] = []
    mom_vals = [round(x, 2) for x in np.arange(0.20, 0.55 + step / 2, step)]
    trend_vals = [round(x, 2) for x in np.arange(0.15, 0.50 + step / 2, step)]
    macro_vals = [round(x, 2) for x in np.arange(0.10, 0.30 + step / 2, step)]
    for mom in mom_vals:
        for trend in trend_vals:
            for macro in macro_vals:
                risk = round(1.0 - mom - trend - macro, 2)
                if 0.05 <= risk <= 0.20:
                    grid.append(
                        {
                            "momentum": mom,
                            "trend": trend,
                            "macro": macro,
                            "risk_penalty": risk,
                        }
                    )
    return grid


def build_weight_variants(
    baseline: dict[str, float],
    factor_ic: pd.DataFrame,
    corr: pd.DataFrame,
) -> list[tuple[str, dict[str, float], str, dict[str, Any]]]:
    """Named weight presets plus metadata for merged-technical variants."""
    ic_weights = ic_proportional_weights(factor_ic, period="train", fallback=baseline)
    mom_trend_corr = float("nan")
    if not corr.empty and "momentum_score" in corr.index and "trend_score" in corr.columns:
        mom_trend_corr = float(corr.loc["momentum_score", "trend_score"])

    trend_heavy = dict(baseline)
    trend_heavy.update({"momentum": 0.25, "trend": 0.45, "macro": 0.20, "risk_penalty": 0.10})

    low_momentum = dict(baseline)
    low_momentum.update({"momentum": 0.30, "trend": 0.40, "macro": 0.20, "risk_penalty": 0.10})

    ablate_momentum = dict(baseline)
    ablate_momentum.update({"momentum": 0.0, "trend": 0.70, "macro": 0.20, "risk_penalty": 0.10})

    tech_blend = ic_technical_blend(factor_ic, period="train")
    technical_merged = {"momentum": 0.70, "trend": 0.0, "macro": 0.20, "risk_penalty": 0.10}

    corr_note = f"{mom_trend_corr:.2f}" if pd.notna(mom_trend_corr) else "high"
    variants: list[tuple[str, dict[str, float], str, dict[str, Any]]] = [
        ("baseline", dict(baseline), "Current config weights", {}),
        (
            "ic_proportional_train",
            ic_weights,
            "Train non-negative IC normalized to sum to 1",
            {},
        ),
        (
            "trend_heavy",
            trend_heavy,
            f"Shift weight from momentum to trend (mom-trend corr {corr_note})",
            {},
        ),
        (
            "low_momentum",
            low_momentum,
            "Moderate momentum reduction with higher trend weight",
            {},
        ),
        (
            "ablate_momentum",
            ablate_momentum,
            "Zero momentum weight; trend-only technical signal (ablation-style)",
            {},
        ),
        (
            "technical_ic_blend",
            technical_merged,
            f"Single technical bucket ({tech_blend[0]:.0%} mom / {tech_blend[1]:.0%} trend by train IC)",
            {"technical_blend": tech_blend},
        ),
    ]
    return variants


def _metrics_by_period(
    returns: pd.Series,
    cfg: PipelineConfig,
) -> dict[str, float]:
    from src.diagnostics import strategy_metrics_on_period

    train = strategy_metrics_on_period(
        returns,
        start=cfg.split.train_start,
        end=cfg.split.train_end,
    )
    test = strategy_metrics_on_period(
        returns,
        start=cfg.split.test_start,
        end=cfg.split.test_end,
    )
    return {
        "train_ann_return": train["annualized_return"],
        "train_sharpe": train["sharpe"],
        "train_max_drawdown": train["max_drawdown"],
        "test_ann_return": test["annualized_return"],
        "test_sharpe": test["sharpe"],
        "test_max_drawdown": test["max_drawdown"],
    }


def grid_search_m1_weights(
    m1: RuleBasedM1,
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: list[str],
    *,
    top_n: int = 15,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select weights maximizing train Sharpe; return best weights and top grid rows."""
    comps = _component_scores_from_panel(m1, panel, feature_cols)
    rows: list[dict[str, Any]] = []
    for weights in _weight_grid():
        score = composite_score_from_components(comps, weights)
        bt = _backtest_from_score(panel, score, returns_wide, cfg, name="grid")
        period = _metrics_by_period(bt.returns, cfg)
        rows.append(
            {
                "variant": "grid",
                **weights,
                **period,
            }
        )
    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return dict(m1.weights), grid_df
    ranked = grid_df.sort_values("train_sharpe", ascending=False, na_position="last")
    best = ranked.iloc[0]
    best_weights = {
        "momentum": float(best["momentum"]),
        "trend": float(best["trend"]),
        "macro": float(best["macro"]),
        "risk_penalty": float(best["risk_penalty"]),
    }
    return best_weights, ranked.head(top_n)


def factor_weight_tuning_summary(
    m1: RuleBasedM1,
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: list[str],
    factor_ic: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    run_grid: bool = True,
    grid_top_n: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare preset and grid-searched M1 weights on train vs test portfolio metrics."""
    comps = _component_scores_from_panel(m1, panel, feature_cols)
    baseline = dict(m1.weights)
    variants = build_weight_variants(baseline, factor_ic, corr)

    grid_best: dict[str, float] = baseline
    grid_top = pd.DataFrame()
    if run_grid:
        grid_best, grid_top = grid_search_m1_weights(
            m1, panel, returns_wide, cfg, feature_cols, top_n=grid_top_n
        )
        variants.append(
            (
                "grid_best_train",
                grid_best,
                "Best coarse grid combo by train Sharpe (may overfit train)",
                {},
            )
        )

    rows: list[dict[str, Any]] = []
    for name, weights, description, meta in variants:
        blend = meta.get("technical_blend")
        score = composite_score_from_components(comps, weights, technical_blend=blend)
        bt = _backtest_from_score(panel, score, returns_wide, cfg, name=name)
        period = _metrics_by_period(bt.returns, cfg)
        rows.append(
            {
                "variant": name,
                "description": description,
                "momentum": weights.get("momentum", 0.0),
                "trend": weights.get("trend", 0.0),
                "macro": weights.get("macro", 0.0),
                "risk_penalty": weights.get("risk_penalty", 0.0),
                **period,
            }
        )

    summary = pd.DataFrame(rows)
    recommendation = recommend_m1_weights(summary, baseline, factor_ic, corr)
    return summary, {"grid_top": grid_top, "recommendation": recommendation}


def recommend_m1_weights(
    tuning: pd.DataFrame,
    baseline: dict[str, float],
    factor_ic: pd.DataFrame,
    corr: pd.DataFrame,
) -> dict[str, Any]:
    """Pick a conservative recommendation: strong test Sharpe without collapsing train."""
    if tuning.empty:
        return {"variant": "baseline", "weights": baseline, "rationale": "No tuning results."}

    test_ic = factor_ic[factor_ic["period"] == "test"] if not factor_ic.empty else pd.DataFrame()
    trend_ic = float("nan")
    mom_ic = float("nan")
    if not test_ic.empty:
        t_row = test_ic[test_ic["factor"] == "trend_score"]
        m_row = test_ic[test_ic["factor"] == "momentum_score"]
        if not t_row.empty:
            trend_ic = float(t_row.iloc[0]["ic_mean"])
        if not m_row.empty:
            mom_ic = float(m_row.iloc[0]["ic_mean"])

    mom_trend_corr = float("nan")
    if not corr.empty and "momentum_score" in corr.index:
        mom_trend_corr = float(corr.loc["momentum_score", "trend_score"])

    ranked = tuning.sort_values("test_sharpe", ascending=False, na_position="last")
    best_test = ranked.iloc[0]
    baseline_row = tuning[tuning["variant"] == "baseline"]
    baseline_test_sharpe = (
        float(baseline_row.iloc[0]["test_sharpe"]) if not baseline_row.empty else float("nan")
    )

    # Prefer highest test Sharpe; require meaningful improvement over baseline for non-baseline pick
    min_improvement = 0.005
    if (
        str(best_test["variant"]) != "baseline"
        and float(best_test["test_sharpe"]) >= baseline_test_sharpe + min_improvement
    ):
        candidate = best_test
    elif not baseline_row.empty:
        candidate = baseline_row.iloc[0]
    else:
        candidate = best_test

    rationale_parts = [
        f"Test-period IC favors trend ({trend_ic:.3f}) over momentum ({mom_ic:.3f}).",
        f"Momentum-trend correlation {mom_trend_corr:.2f} suggests redundant technical exposure.",
    ]
    if str(candidate["variant"]) == "baseline":
        rationale_parts.append(
            f"No preset/grid variant beat baseline test Sharpe by ≥{min_improvement:.3f}; keep current weights."
        )
    else:
        rationale_parts.append(
            f"Variant `{candidate['variant']}` improves test Sharpe to {_fmt_metric(candidate['test_sharpe'])} "
            f"vs baseline {_fmt_metric(baseline_test_sharpe)}."
        )
    return {
        "variant": str(candidate["variant"]),
        "weights": {
            "momentum": float(candidate["momentum"]),
            "trend": float(candidate["trend"]),
            "macro": float(candidate["macro"]),
            "risk_penalty": float(candidate["risk_penalty"]),
        },
        "test_sharpe": float(candidate["test_sharpe"]),
        "baseline_test_sharpe": baseline_test_sharpe,
        "rationale": " ".join(rationale_parts),
    }


def derive_fold_ic_proportional_weights(
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    fold_cfg: PipelineConfig,
) -> dict[str, float]:
    """Compute IC-proportional M1 weights using only the fold train window (no lookahead)."""
    from src.model_m1 import build_m1_model, split_train_test

    panel = _ensure_panel_index(base_panel)
    dates = pd.to_datetime(panel.index.get_level_values("date"))
    test_end = pd.Timestamp(fold_cfg.split.test_end or dates.max())
    panel = panel[dates <= test_end]
    train, _ = split_train_test(panel, fold_cfg)
    if train.empty:
        return dict(fold_cfg.m1.weights)

    fwd_col = f"forward_return_{fold_cfg.labels.horizon_weeks}w"
    m1 = build_m1_model(fold_cfg)
    X_train = train[feature_cols].fillna(0)
    returns_train = returns_wide.loc[
        (returns_wide.index >= pd.Timestamp(fold_cfg.split.train_start))
        & (returns_wide.index <= pd.Timestamp(fold_cfg.split.train_end))
    ]
    m1.fit(
        X_train,
        train["m1_target"],
        forward_returns=train[fwd_col],
        panel=train,
        returns_wide=returns_train,
        portfolio_cfg=fold_cfg.portfolio,
    )
    components = m1.predict_component_scores(X_train)
    train_scored = train.copy()
    for col in components.columns:
        train_scored[col] = components[col].reindex(train.index)
    factor_ic = compute_factor_ic(
        train_scored,
        period_label="train",
        start=fold_cfg.split.train_start,
        end=fold_cfg.split.train_end,
    )
    return ic_proportional_weights(factor_ic, period="train", fallback=dict(fold_cfg.m1.weights))


def evaluate_m1_weight_walk_forward_decision(
    summary: pd.DataFrame,
    *,
    min_m1_sharpe_gain: float = 0.003,
    max_ecdf_sharpe_loss: float = 0.02,
) -> dict[str, Any]:
    """Decide whether to adopt IC-proportional weights from walk-forward fold comparison."""
    if summary.empty:
        return {
            "apply_ic_weights": False,
            "reason": "No walk-forward folds completed.",
        }

    m1_gain = float(summary["ic_m1_sharpe"].mean() - summary["baseline_m1_sharpe"].mean())
    ecdf_gain = float(summary["ic_ecdf_sharpe"].mean() - summary["baseline_ecdf_sharpe"].mean())
    m1_wins = int((summary["ic_m1_sharpe"] > summary["baseline_m1_sharpe"]).sum())
    ecdf_wins = int((summary["ic_ecdf_sharpe"] >= summary["baseline_ecdf_sharpe"]).sum())
    n_folds = len(summary)

    apply = (
        m1_gain >= min_m1_sharpe_gain
        and ecdf_gain >= -max_ecdf_sharpe_loss
        and m1_wins >= max(1, n_folds // 2)
    )
    if apply:
        reason = (
            f"Walk-forward: mean M1 Sharpe +{m1_gain:.4f}, mean ECDF Sharpe {ecdf_gain:+.4f}; "
            f"IC wins M1 in {m1_wins}/{n_folds} folds."
        )
    else:
        reason = (
            f"Walk-forward: mean M1 Sharpe {m1_gain:+.4f} (need ≥{min_m1_sharpe_gain}), "
            f"mean ECDF Sharpe {ecdf_gain:+.4f} (max loss {max_ecdf_sharpe_loss}); "
            f"M1 wins {m1_wins}/{n_folds}. Keep baseline weights."
        )
    return {
        "apply_ic_weights": apply,
        "mean_m1_sharpe_gain": m1_gain,
        "mean_ecdf_sharpe_gain": ecdf_gain,
        "m1_fold_wins": m1_wins,
        "ecdf_fold_wins": ecdf_wins,
        "n_folds": n_folds,
        "reason": reason,
    }


def run_m1_weight_walk_forward_validation(
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare baseline vs per-fold IC-proportional M1 weights across walk-forward folds."""
    from src.config import apply_split_overrides, clone_config_with_m1_allow_short, clone_config_with_m1_weights
    from src.evaluation import _fit_fold_stack, build_walk_forward_folds

    long_cfg = clone_config_with_m1_allow_short(cfg, allow_short=False)
    folds = build_walk_forward_folds(base_panel, long_cfg, long_cfg.evaluation)
    rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_cfg = apply_split_overrides(
            long_cfg,
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )
        try:
            ic_weights = derive_fold_ic_proportional_weights(
                base_panel, feature_cols, returns_wide, fold_cfg
            )
            _, base_summary = _fit_fold_stack(base_panel, feature_cols, returns_wide, fold_cfg)
            ic_cfg = clone_config_with_m1_weights(fold_cfg, ic_weights)
            _, ic_summary = _fit_fold_stack(base_panel, feature_cols, returns_wide, ic_cfg)
        except (ValueError, KeyError) as exc:
            import logging

            logging.getLogger(__name__).warning("M1 weight fold %s skipped: %s", fold.get("fold_id"), exc)
            continue

        b_m1 = base_summary["strategy_metrics"].get("m1_only", {})
        b_ecdf = base_summary["strategy_metrics"].get("m1_m2_m3_ecdf", {})
        i_m1 = ic_summary["strategy_metrics"].get("m1_only", {})
        i_ecdf = ic_summary["strategy_metrics"].get("m1_m2_m3_ecdf", {})
        rows.append(
            {
                **fold,
                "baseline_m1_sharpe": b_m1.get("sharpe", float("nan")),
                "ic_m1_sharpe": i_m1.get("sharpe", float("nan")),
                "m1_sharpe_delta": i_m1.get("sharpe", float("nan")) - b_m1.get("sharpe", float("nan")),
                "baseline_ecdf_sharpe": b_ecdf.get("sharpe", float("nan")),
                "ic_ecdf_sharpe": i_ecdf.get("sharpe", float("nan")),
                "ecdf_sharpe_delta": i_ecdf.get("sharpe", float("nan")) - b_ecdf.get("sharpe", float("nan")),
                "ic_momentum": ic_weights.get("momentum"),
                "ic_trend": ic_weights.get("trend"),
                "ic_macro": ic_weights.get("macro"),
                "ic_risk_penalty": ic_weights.get("risk_penalty"),
            }
        )

    summary = pd.DataFrame(rows)
    decision = evaluate_m1_weight_walk_forward_decision(summary)
    if not summary.empty:
        decision["mean_ic_weights"] = {
            "momentum": float(summary["ic_momentum"].mean()),
            "trend": float(summary["ic_trend"].mean()),
            "macro": float(summary["ic_macro"].mean()),
            "risk_penalty": float(summary["ic_risk_penalty"].mean()),
        }
    return summary, decision


def finalize_m1_weight_recommendation(
    holdout_rec: dict[str, Any],
    wf_decision: dict[str, Any] | None,
    baseline_weights: dict[str, float],
) -> dict[str, Any]:
    """Merge single-holdout tuning with walk-forward adoption decision."""
    rec = dict(holdout_rec)
    rec["holdout_variant"] = rec.get("variant", "baseline")
    rec["holdout_test_sharpe"] = rec.get("test_sharpe")
    if not wf_decision:
        rec["walk_forward_validated"] = False
        rec["config_action"] = "keep_baseline_run_walk_forward"
        rec["rationale"] = (
            f"{rec.get('rationale', '')} Walk-forward validation not run; keep config weights until validated."
        ).strip()
        rec["variant"] = "baseline"
        rec["weights"] = dict(baseline_weights)
        return rec

    rec["walk_forward_validated"] = True
    rec["walk_forward_apply"] = bool(wf_decision.get("apply_ic_weights"))
    rec["walk_forward_mean_m1_gain"] = wf_decision.get("mean_m1_sharpe_gain")
    rec["walk_forward_mean_ecdf_gain"] = wf_decision.get("mean_ecdf_sharpe_gain")
    rec["walk_forward_m1_wins"] = wf_decision.get("m1_fold_wins")
    rec["walk_forward_n_folds"] = wf_decision.get("n_folds")

    if wf_decision.get("apply_ic_weights"):
        mean_w = wf_decision.get("mean_ic_weights") or rec.get("weights", baseline_weights)
        rec["variant"] = "ic_proportional_walk_forward"
        rec["weights"] = dict(mean_w)
        rec["config_action"] = "apply_ic_weights"
        rec["rationale"] = wf_decision.get("reason", "")
    else:
        rec["variant"] = "baseline"
        rec["weights"] = dict(baseline_weights)
        rec["config_action"] = "keep_baseline"
        rec["rationale"] = (
            f"{wf_decision.get('reason', '')} "
            f"Holdout tuning favored `{rec.get('holdout_variant')}` "
            f"(test Sharpe {rec.get('holdout_test_sharpe', float('nan')):.4f}) but walk-forward did not confirm."
        ).strip()
    return rec


def _fmt_metric(val: float) -> str:
    if pd.isna(val):
        return "n/a"
    return f"{val:.4f}"


def save_weight_tuning_chart(tuning: pd.DataFrame, output_dir: Path) -> str | None:
    if tuning.empty:
        return None
    plot_df = tuning.sort_values("test_sharpe", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    colors = ["#4C72B0" if v != "baseline" else "#55A868" for v in plot_df["variant"]]
    ax.barh(plot_df["variant"], plot_df["test_sharpe"], color=colors)
    baseline_rows = plot_df[plot_df["variant"] == "baseline"]
    if not baseline_rows.empty:
        ax.axvline(float(baseline_rows.iloc[0]["test_sharpe"]), color="gray", linestyle="--", label="baseline")
    ax.set_xlabel("Test-period Sharpe")
    ax.set_title("M1 Weight Variants — Test Sharpe")
    ax.legend(loc="lower right")
    p = output_dir / "m1_weight_tuning_test_sharpe.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p.name


def _component_scores_from_panel(
    m1: RuleBasedM1,
    panel: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Use persisted component scores when available; else recompute from features."""
    if all(c in panel.columns for c in FACTOR_COLS):
        return panel[FACTOR_COLS].copy()
    X = panel[feature_cols].fillna(0)
    return m1.predict_component_scores(X)


def compute_factor_ic(
    panel: pd.DataFrame,
    *,
    period_label: str = "full",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Spearman IC of each M1 component vs forward return."""
    sub = _filter_panel_period(panel, start=start, end=end)
    fwd_cols = [c for c in sub.reset_index().columns if c.startswith("forward_return")]
    if not fwd_cols:
        return pd.DataFrame()
    fwd_col = fwd_cols[0]
    rows = []
    for col in FACTOR_COLS:
        if col not in sub.columns:
            continue
        tmp = sub.copy()
        tmp["M1_score"] = tmp[col]
        ic_series = compute_ic(tmp, score_col="M1_score", fwd_col=fwd_col)
        rows.append(
            {
                "period": period_label,
                "factor": col,
                "ic_mean": float(ic_series.mean()) if not ic_series.empty else float("nan"),
                "ic_std": float(ic_series.std()) if not ic_series.empty else float("nan"),
                "ic_hit_rate": float((ic_series > 0).mean()) if not ic_series.empty else float("nan"),
                "n_weeks": int(len(ic_series)),
            }
        )
    # Composite M1 score for reference
    if "M1_score" in sub.columns:
        ic_series = compute_ic(sub, score_col="M1_score", fwd_col=fwd_col)
        rows.append(
            {
                "period": period_label,
                "factor": "M1_score",
                "ic_mean": float(ic_series.mean()) if not ic_series.empty else float("nan"),
                "ic_std": float(ic_series.std()) if not ic_series.empty else float("nan"),
                "ic_hit_rate": float((ic_series > 0).mean()) if not ic_series.empty else float("nan"),
                "n_weeks": int(len(ic_series)),
            }
        )
    return pd.DataFrame(rows)


def compute_factor_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FACTOR_COLS if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return panel[cols].corr(method="pearson")


def compute_factor_covariance(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FACTOR_COLS if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return panel[cols].cov()


def compute_underlying_feature_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in UNDERLYING_M1_FEATURES if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return panel[cols].corr(method="pearson")


def _backtest_from_score(
    panel: pd.DataFrame,
    score: pd.Series,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    name: str = "sleeve",
) -> BacktestResult:
    from src.model_m1 import RuleBasedM1
    from src.portfolio import apply_vol_target_wide, build_weights_from_signals

    rb = RuleBasedM1(cfg.m1)
    signals = rb._signals_top_k(score.rename("M1_score"))
    w = build_weights_from_signals(panel, signals, portfolio_cfg=cfg.portfolio)
    w = apply_vol_target_wide(w, returns_wide, cfg.portfolio)
    return _run_backtest(name, w, returns_wide, cfg.portfolio.transaction_cost_bps)


def factor_sleeve_summary(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Backtest top-K sleeves driven by each component score alone."""
    bench_key = "equal_weight_1_7"
    rows = []
    sleeve_returns: dict[str, pd.Series] = {}

    if "M1_score" in panel.columns:
        full_bt = _backtest_from_score(panel, panel["M1_score"], returns_wide, cfg, name="full_m1")
        sleeve_returns["full_m1"] = full_bt.returns
        m = strategy_metrics(full_bt, None)
        rows.append({"sleeve": "full_m1", **m})

    for col in FACTOR_COLS:
        if col not in panel.columns:
            continue
        score = panel[col]
        if col == "risk_penalty":
            score = -score
        bt = _backtest_from_score(panel, score, returns_wide, cfg, name=col)
        sleeve_returns[col] = bt.returns
        m = strategy_metrics(bt, None)
        rows.append({"sleeve": col, **m})

    interaction = factor_interaction_term(sleeve_returns, panel, returns_wide, cfg)
    if interaction is not None:
        rows.append(interaction)
    return pd.DataFrame(rows)


def factor_interaction_term(
    sleeve_returns: dict[str, pd.Series],
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, Any] | None:
    """Combined M1 excess minus sum of standalone factor sleeve excess returns."""
    if "full_m1" not in sleeve_returns:
        return None
    from src.backtest import equal_weight_returns

    bench = equal_weight_returns(returns_wide, cfg.assets.tickers)
    combined_excess = annualized_return(sleeve_returns["full_m1"]) - annualized_return(bench.returns)
    standalone_cols = [c for c in FACTOR_COLS if c in sleeve_returns]
    if not standalone_cols:
        return None
    sum_excess = sum(
        annualized_return(sleeve_returns[c]) - annualized_return(bench.returns) for c in standalone_cols
    )
    interaction = combined_excess - sum_excess
    return {
        "sleeve": "interaction",
        "annualized_return": float("nan"),
        "annualized_volatility": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "interaction_excess_ann": float(interaction),
        "combined_excess_ann": float(combined_excess),
        "sum_standalone_excess_ann": float(sum_excess),
    }


def factor_ablation_summary(
    m1: RuleBasedM1,
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Zero out one factor weight at a time and compare portfolio metrics."""
    comps = _component_scores_from_panel(m1, panel, feature_cols)
    rows = []
    base_bt = _backtest_from_score(
        panel,
        composite_score_from_components(comps, m1.weights),
        returns_wide,
        cfg,
        name="full",
    )
    rows.append({"variant": "full_m1", **strategy_metrics(base_bt, None)})

    weight_keys = {
        "ablate_momentum": "momentum",
        "ablate_trend": "trend",
        "ablate_macro": "macro",
        "ablate_risk_penalty": "risk_penalty",
    }
    w = m1.weights
    for variant, key in weight_keys.items():
        weights = dict(w)
        weights[key] = 0.0
        score = composite_score_from_components(comps, weights)
        bt = _backtest_from_score(panel, score, returns_wide, cfg, name=variant)
        rows.append({"variant": variant, **strategy_metrics(bt, None)})
    return pd.DataFrame(rows)


def save_factor_charts(
    factor_ic: pd.DataFrame,
    corr: pd.DataFrame,
    sleeves: pd.DataFrame,
    sleeve_returns: dict[str, pd.Series] | None,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if not factor_ic.empty:
        test_ic = factor_ic[factor_ic["period"] == "test"]
        if test_ic.empty:
            test_ic = factor_ic
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = test_ic[test_ic["factor"] != "M1_score"]
        ax.barh(plot_df["factor"], plot_df["ic_mean"], color="#4C72B0")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Mean Spearman IC")
        ax.set_title("M1 Factor IC (test period)")
        p = output_dir / "m1_factor_ic.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if not corr.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        ax.set_title("M1 Factor Correlation")
        fig.colorbar(im, ax=ax)
        p = output_dir / "m1_factor_correlation_heatmap.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if sleeve_returns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, rets in sleeve_returns.items():
            if name == "interaction":
                continue
            cum = (1 + rets.fillna(0)).cumprod()
            ax.plot(cum.index, cum.values, label=name)
        ax.legend(fontsize=8)
        ax.set_title("M1 Factor Sleeve Cumulative Returns")
        p = output_dir / "m1_factor_sleeves_cumulative.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    return saved


def _collect_sleeve_returns(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if "M1_score" in panel.columns:
        out["full_m1"] = _backtest_from_score(panel, panel["M1_score"], returns_wide, cfg).returns
    for col in FACTOR_COLS:
        if col not in panel.columns:
            continue
        score = -panel[col] if col == "risk_penalty" else panel[col]
        out[col] = _backtest_from_score(panel, score, returns_wide, cfg).returns
    return out


def run_factor_analysis(
    panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    output_dir: Path,
    *,
    m1_model: object | None = None,
    feature_cols: list[str] | None = None,
    m1_weight_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run full M1 factor diagnostics and persist CSVs/charts."""
    panel = _ensure_panel_index(panel)
    train_panel = _ensure_panel_index(train_panel)
    test_panel = _ensure_panel_index(test_panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ic_frames = [
        compute_factor_ic(panel, period_label="full"),
        compute_factor_ic(train_panel, period_label="train", start=cfg.split.train_start, end=cfg.split.train_end),
        compute_factor_ic(test_panel, period_label="test", start=cfg.split.test_start, end=cfg.split.test_end),
    ]
    factor_ic = pd.concat([f for f in ic_frames if not f.empty], ignore_index=True)
    factor_ic.to_csv(output_dir / "m1_factor_ic.csv", index=False)

    corr = compute_factor_correlation(panel)
    if not corr.empty:
        corr.to_csv(output_dir / "m1_factor_correlation.csv")
    cov = compute_factor_covariance(panel)
    if not cov.empty:
        cov.to_csv(output_dir / "m1_factor_covariance.csv")
    feat_corr = compute_underlying_feature_correlation(panel)
    if not feat_corr.empty:
        feat_corr.to_csv(output_dir / "m1_underlying_feature_correlation.csv")

    sleeves = factor_sleeve_summary(panel, returns_wide, cfg)
    sleeves.to_csv(output_dir / "m1_factor_sleeves.csv", index=False)

    ablation = pd.DataFrame()
    weight_tuning = pd.DataFrame()
    weight_tuning_meta: dict[str, Any] = {}
    if isinstance(m1_model, RuleBasedM1) and feature_cols:
        ablation = factor_ablation_summary(m1_model, panel, returns_wide, cfg, feature_cols)
        ablation.to_csv(output_dir / "m1_factor_ablation.csv", index=False)

        weight_tuning, weight_tuning_meta = factor_weight_tuning_summary(
            m1_model,
            panel,
            returns_wide,
            cfg,
            feature_cols,
            factor_ic,
            corr,
        )
        weight_tuning.to_csv(output_dir / "m1_factor_weight_tuning.csv", index=False)
        grid_top = weight_tuning_meta.get("grid_top", pd.DataFrame())
        if not grid_top.empty:
            grid_top.to_csv(output_dir / "m1_factor_weight_grid_top.csv", index=False)
        rec = weight_tuning_meta.get("recommendation", {})
        if rec:
            rec = finalize_m1_weight_recommendation(rec, m1_weight_decision, dict(m1_model.weights))
            weight_tuning_meta["recommendation"] = rec
            pd.DataFrame([rec]).to_csv(output_dir / "m1_factor_weight_recommendation.csv", index=False)

    sleeve_returns = _collect_sleeve_returns(panel, returns_wide, cfg)
    charts = save_factor_charts(factor_ic, corr, sleeves, sleeve_returns, figures_dir)
    wt_chart = save_weight_tuning_chart(weight_tuning, figures_dir)
    if wt_chart:
        charts.append(wt_chart)

    return {
        "factor_ic": factor_ic,
        "factor_correlation": corr,
        "factor_covariance": cov,
        "underlying_feature_correlation": feat_corr,
        "factor_sleeves": sleeves,
        "factor_ablation": ablation,
        "factor_weight_tuning": weight_tuning,
        "weight_tuning_meta": weight_tuning_meta,
        "charts": charts,
    }
