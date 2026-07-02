"""Extended evaluation: walk-forward validation and transaction-cost sensitivity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import (
    STRATEGY_M1_M2_M3_ECDF,
    _run_backtest,
    run_all_strategies,
)
from src.config import EvaluationConfig, PipelineConfig, apply_split_overrides
from src.diagnostics import m2_classification_metrics, strategy_metrics_on_period
from src.labels import build_meta_labels, get_m2_training_mask
from src.model_m1 import build_m1_model, split_train_test
from src.model_m2 import fit_m2, predict_m2
from src.model_m3 import attach_m3_to_panel

logger = logging.getLogger(__name__)

TC_SENSITIVITY_STRATEGIES = ("equal_weight_1_7", "m1_only", STRATEGY_M1_M2_M3_ECDF)


def _panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(panel.index, pd.MultiIndex):
        return pd.to_datetime(panel.index.get_level_values("date")).unique()
    return pd.to_datetime(panel["date"]).unique()


def build_walk_forward_folds(
    panel: pd.DataFrame,
    cfg: PipelineConfig,
    eval_cfg: EvaluationConfig | None = None,
) -> list[dict[str, str]]:
    """Expanding-window folds: each test block is followed by extending train through that block."""
    eval_cfg = eval_cfg or EvaluationConfig()
    dates = pd.DatetimeIndex(sorted(_panel_dates(panel)))
    if dates.empty:
        return []

    panel_end = dates.max()
    train_end = pd.Timestamp(eval_cfg.walk_forward_first_train_end)
    cfg_train_start = pd.Timestamp(cfg.split.train_start)
    cfg_train_end = pd.Timestamp(cfg.split.train_end)
    if train_end < cfg_train_start:
        train_end = cfg_train_end
    test_years = max(1, int(eval_cfg.walk_forward_test_years))
    folds: list[dict[str, str]] = []

    while train_end < panel_end:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if test_start > panel_end:
            break
        if test_end > panel_end:
            test_end = panel_end

        fold_cfg = apply_split_overrides(
            cfg,
            train_end=train_end.date().isoformat(),
            test_start=test_start.date().isoformat(),
            test_end=test_end.date().isoformat(),
        )
        try:
            from src.config import validate_split_dates

            validate_split_dates(fold_cfg)
        except ValueError as exc:
            logger.warning("Skipping walk-forward fold ending %s: %s", train_end.date(), exc)
            break

        test_weeks = int((dates[(dates >= test_start) & (dates <= test_end)]).size)
        if test_weeks < 26:
            logger.info("Walk-forward: stopping — test window has only %d weeks", test_weeks)
            break

        folds.append(
            {
                "fold_id": str(len(folds) + 1),
                "train_start": fold_cfg.split.train_start,
                "train_end": fold_cfg.split.train_end,
                "test_start": fold_cfg.split.test_start,
                "test_end": fold_cfg.split.test_end or panel_end.date().isoformat(),
                "test_weeks": test_weeks,
            }
        )
        if test_end >= panel_end:
            break
        train_end = test_end

    return folds


def _fit_fold_stack(
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    fold_cfg: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train M1/M2 on fold train window; return panel with predictions through fold test_end."""
    dates = pd.to_datetime(
        base_panel.index.get_level_values("date")
        if isinstance(base_panel.index, pd.MultiIndex)
        else base_panel["date"]
    )
    test_end = pd.Timestamp(fold_cfg.split.test_end or dates.max())
    panel = base_panel[dates <= test_end].copy()
    train, test = split_train_test(panel, fold_cfg)
    if train.empty or test.empty:
        raise ValueError("Empty train or test slice in walk-forward fold")

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

    X_panel = panel[feature_cols].fillna(0)
    m1_signals = m1.predict_signal(X_panel)
    m1_scores = m1.predict_score(X_panel)
    m1_conviction = m1.predict_conviction(X_panel)
    m1_components = m1.predict_component_scores(X_panel)

    panel = build_meta_labels(panel, m1_signals, m1_scores, fold_cfg)
    panel["M1_conviction"] = m1_conviction.reindex(panel.index).fillna(0.0)
    for col in m1_components.columns:
        panel[col] = m1_components[col].reindex(panel.index)

    m2_model, _ = fit_m2(panel, fold_cfg)
    panel = predict_m2(m2_model, panel, fold_cfg)
    train, test = split_train_test(panel, fold_cfg)
    train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
    panel = attach_m3_to_panel(panel, fold_cfg, train_proba=train_proba)

    results = run_all_strategies(panel, returns_wide, fold_cfg, train_proba=train_proba)

    test_mask = get_m2_training_mask(test)
    test_idx = test.loc[test_mask].index
    y_true = test.loc[test_mask, "meta_label"]
    y_prob = panel.loc[test_idx, "p_success"]
    m2_metrics = m2_classification_metrics(y_true, y_prob, threshold=fold_cfg.m2.threshold)

    strategy_metrics: dict[str, dict[str, float]] = {}
    for key in ("m1_only", STRATEGY_M1_M2_M3_ECDF, "equal_weight_1_7"):
        if key not in results:
            continue
        strategy_metrics[key] = strategy_metrics_on_period(
            results[key].returns,
            start=fold_cfg.split.test_start,
            end=fold_cfg.split.test_end,
        )

    return panel, {"results": results, "m2_metrics": m2_metrics, "strategy_metrics": strategy_metrics}


def run_walk_forward_evaluation(
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    eval_cfg: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Run expanding-window walk-forward folds; return one row per fold."""
    eval_cfg = eval_cfg or cfg.evaluation
    if not eval_cfg.walk_forward_enabled:
        return pd.DataFrame()

    folds = build_walk_forward_folds(base_panel, cfg, eval_cfg)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_cfg = apply_split_overrides(
            cfg,
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )
        try:
            _, summary = _fit_fold_stack(base_panel, feature_cols, returns_wide, fold_cfg)
        except (ValueError, KeyError) as exc:
            logger.warning("Walk-forward fold %s failed: %s", fold.get("fold_id"), exc)
            continue

        sm = summary["strategy_metrics"]
        m2m = summary["m2_metrics"]
        m1 = sm.get("m1_only", {})
        ecdf = sm.get(STRATEGY_M1_M2_M3_ECDF, {})
        ew = sm.get("equal_weight_1_7", {})
        rows.append(
            {
                **fold,
                "m1_only_sharpe": m1.get("sharpe", float("nan")),
                "m1_only_ann_return": m1.get("annualized_return", float("nan")),
                "ecdf_sharpe": ecdf.get("sharpe", float("nan")),
                "ecdf_ann_return": ecdf.get("annualized_return", float("nan")),
                "ecdf_sharpe_edge_vs_m1": ecdf.get("sharpe", float("nan")) - m1.get("sharpe", float("nan")),
                "ecdf_return_edge_vs_m1": ecdf.get("annualized_return", float("nan"))
                - m1.get("annualized_return", float("nan")),
                "equal_weight_sharpe": ew.get("sharpe", float("nan")),
                "m2_auc": m2m.get("auc", float("nan")),
                "m2_auc_pr": m2m.get("auc_pr", float("nan")),
                "m2_n_trades": m2m.get("n_trades", 0),
            }
        )

    return pd.DataFrame(rows)


def run_transaction_cost_sensitivity(
    results: dict[str, Any],
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    test_start: str,
    test_end: str | None = None,
    eval_cfg: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Re-score strategy returns at multiple transaction-cost levels (same weights)."""
    eval_cfg = eval_cfg or cfg.evaluation
    rows: list[dict[str, Any]] = []
    aligned_ret = returns_wide.sort_index()

    for bps in eval_cfg.transaction_cost_bps_grid:
        for strat in TC_SENSITIVITY_STRATEGIES:
            res = results.get(strat)
            if res is None:
                continue
            bt = _run_backtest(strat, res.weights, aligned_ret, float(bps))
            period = strategy_metrics_on_period(
                bt.returns,
                start=test_start,
                end=test_end,
            )
            rows.append(
                {
                    "transaction_cost_bps": bps,
                    "strategy": strat,
                    "annualized_return": period["annualized_return"],
                    "sharpe": period["sharpe"],
                    "max_drawdown": period["max_drawdown"],
                    "hit_rate": period["hit_rate"],
                    "n_weeks": period["n_weeks"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pivot = df.pivot(index="transaction_cost_bps", columns="strategy", values="sharpe")
    if "m1_only" in pivot.columns and STRATEGY_M1_M2_M3_ECDF in pivot.columns:
        df["ecdf_sharpe_edge_vs_m1"] = np.nan
        for bps in df["transaction_cost_bps"].unique():
            mask = df["transaction_cost_bps"] == bps
            ecdf_sh = df.loc[mask & (df["strategy"] == STRATEGY_M1_M2_M3_ECDF), "sharpe"]
            m1_sh = df.loc[mask & (df["strategy"] == "m1_only"), "sharpe"]
            if not ecdf_sh.empty and not m1_sh.empty:
                edge = float(ecdf_sh.iloc[0] - m1_sh.iloc[0])
                df.loc[mask & (df["strategy"] == STRATEGY_M1_M2_M3_ECDF), "ecdf_sharpe_edge_vs_m1"] = edge
    return df


def save_evaluation_charts(
    walk_forward: pd.DataFrame,
    tc_sensitivity: pd.DataFrame,
    output_dir: Any,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if not walk_forward.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        x = walk_forward["fold_id"].astype(str)
        width = 0.25
        idx = np.arange(len(x))
        ax.bar(idx - width, walk_forward["m1_only_sharpe"], width, label="M1 only")
        ax.bar(idx, walk_forward["ecdf_sharpe"], width, label="M1+M2+M3 ECDF")
        ax.bar(idx + width, walk_forward["equal_weight_sharpe"], width, label="Equal weight")
        ax.set_xticks(idx)
        ax.set_xticklabels(x)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Test Sharpe")
        ax.set_title("Walk-Forward Test Sharpe by Fold")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8)
        p = output_dir / "walk_forward_sharpe.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if not tc_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for strat, label in [
            ("m1_only", "M1 only"),
            (STRATEGY_M1_M2_M3_ECDF, "M1+M2+M3 ECDF"),
            ("equal_weight_1_7", "Equal weight"),
        ]:
            sub = tc_sensitivity[tc_sensitivity["strategy"] == strat]
            if sub.empty:
                continue
            ax.plot(sub["transaction_cost_bps"], sub["sharpe"], marker="o", label=label)
        ax.set_xlabel("Transaction cost (bps per unit turnover)")
        ax.set_ylabel("Test-period Sharpe")
        ax.set_title("Transaction-Cost Sensitivity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = output_dir / "transaction_cost_sensitivity.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    return saved


def run_extended_evaluation(
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    production_results: dict[str, Any],
    test_panel: pd.DataFrame,
    eval_cfg: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Walk-forward validation + transaction-cost sensitivity on the production test window."""
    eval_cfg = eval_cfg or cfg.evaluation
    test_start = cfg.split.test_start
    test_end = cfg.split.test_end

    if eval_cfg.walk_forward_enabled:
        n_folds = len(build_walk_forward_folds(base_panel, cfg, eval_cfg))
        logger.info("Running walk-forward evaluation (up to %d folds)", n_folds)
        walk_forward = run_walk_forward_evaluation(base_panel, feature_cols, returns_wide, cfg, eval_cfg)
    else:
        logger.info("Walk-forward evaluation disabled")
        walk_forward = pd.DataFrame()

    logger.info("Running transaction-cost sensitivity on test window")
    tc_sensitivity = run_transaction_cost_sensitivity(
        production_results,
        returns_wide,
        cfg,
        test_start=test_start,
        test_end=test_end,
        eval_cfg=eval_cfg,
    )

    summary: dict[str, Any] = {
        "walk_forward": walk_forward,
        "transaction_cost_sensitivity": tc_sensitivity,
    }
    if not walk_forward.empty:
        summary["walk_forward_mean_ecdf_edge"] = float(walk_forward["ecdf_sharpe_edge_vs_m1"].mean())
        summary["walk_forward_mean_m2_auc"] = float(walk_forward["m2_auc"].mean())
    if not tc_sensitivity.empty:
        ecdf_rows = tc_sensitivity[tc_sensitivity["strategy"] == STRATEGY_M1_M2_M3_ECDF]
        if not ecdf_rows.empty:
            summary["ecdf_sharpe_at_default_bps"] = float(
                ecdf_rows.loc[ecdf_rows["transaction_cost_bps"] == cfg.portfolio.transaction_cost_bps, "sharpe"].iloc[0]
                if (ecdf_rows["transaction_cost_bps"] == cfg.portfolio.transaction_cost_bps).any()
                else float("nan")
            )
            summary["ecdf_edge_persists_at_25bps"] = bool(
                (ecdf_rows.set_index("transaction_cost_bps")["ecdf_sharpe_edge_vs_m1"].get(25.0, float("-inf")) > 0)
                if "ecdf_sharpe_edge_vs_m1" in ecdf_rows.columns
                else False
            )
    return summary


def generate_evaluation_report(
    eval_summary: dict[str, Any],
    report_path: Path,
    *,
    mode_name: str = "long_only",
    cfg: PipelineConfig | None = None,
) -> None:
    """Write walk-forward and transaction-cost sensitivity markdown report."""
    from src.diagnostics import _fmt_num, _markdown_table

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_prefix = f"../data/backtests/{mode_name}/figures"
    walk_forward = eval_summary.get("walk_forward", pd.DataFrame())
    tc = eval_summary.get("transaction_cost_sensitivity", pd.DataFrame())

    lines = [
        "# Extended Evaluation: Walk-Forward & Transaction Costs",
        "",
        "**Research use only — not investment advice.**",
        "",
        "This report validates the M1+M2+M3 ECDF stack on **multiple out-of-sample windows** "
        "(expanding train, rolling 2-year test blocks) and measures **Sharpe sensitivity** to "
        "transaction costs versus M1-only and equal-weight baselines.",
        "",
    ]
    if cfg is not None:
        ev = cfg.evaluation
        lines.extend(
            [
                "## Configuration",
                "",
                f"- Walk-forward enabled: `{ev.walk_forward_enabled}`",
                f"- First train end: `{ev.walk_forward_first_train_end}`",
                f"- Test block length: `{ev.walk_forward_test_years}` year(s)",
                f"- Transaction-cost grid (bps): `{list(ev.transaction_cost_bps_grid)}`",
                f"- Production test window: `{cfg.split.test_start}` to `{cfg.split.test_end or 'latest'}`",
                "",
            ]
        )

    lines.extend(["## Walk-forward validation", ""])
    if walk_forward.empty:
        lines.append("*Walk-forward evaluation disabled or no valid folds.*")
        lines.append("")
    else:
        mean_edge = eval_summary.get("walk_forward_mean_ecdf_edge")
        mean_auc = eval_summary.get("walk_forward_mean_m2_auc")
        if mean_edge is not None:
            lines.append(
                f"- **Mean ECDF Sharpe edge vs M1-only (across folds):** {_fmt_num(mean_edge)}"
            )
        if mean_auc is not None:
            lines.append(f"- **Mean M2 AUC (test, across folds):** {_fmt_num(mean_auc)}")
        lines.append("")
        disp = walk_forward.copy()
        for col in (
            "m1_only_sharpe",
            "ecdf_sharpe",
            "ecdf_sharpe_edge_vs_m1",
            "equal_weight_sharpe",
            "m2_auc",
        ):
            if col in disp.columns:
                disp[col] = disp[col].map(_fmt_num)
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![Walk-forward Sharpe]({fig_prefix}/walk_forward_sharpe.png)")
        lines.append("")

    lines.extend(["## Transaction-cost sensitivity (production test window)", ""])
    if tc.empty:
        lines.append("*No transaction-cost sensitivity results.*")
    else:
        disp = tc.copy()
        for col in ("annualized_return", "sharpe", "max_drawdown", "hit_rate", "ecdf_sharpe_edge_vs_m1"):
            if col in disp.columns:
                disp[col] = disp[col].map(_fmt_num)
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![Transaction-cost sensitivity]({fig_prefix}/transaction_cost_sensitivity.png)")
        lines.append("")
        if eval_summary.get("ecdf_edge_persists_at_25bps"):
            lines.append(
                "- **ECDF Sharpe edge vs M1-only remains positive at 25 bps** turnover cost."
            )
        elif "ecdf_edge_persists_at_25bps" in eval_summary:
            lines.append(
                "- ECDF Sharpe edge vs M1-only **does not persist** at 25 bps under this test window."
            )

    report_path.write_text("\n".join(lines))
