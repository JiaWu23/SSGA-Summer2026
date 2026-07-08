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
from src.diagnostics import m2_classification_metrics, strategy_metrics_on_period, information_ratio
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
        results = summary["results"]
        m1 = sm.get("m1_only", {})
        ecdf = sm.get(STRATEGY_M1_M2_M3_ECDF, {})
        ew = sm.get("equal_weight_1_7", {})

        def _fold_ir(strat_key: str) -> float:
            res = results.get(strat_key)
            if res is None:
                return float("nan")
            r = res.returns.copy()
            r.index = pd.to_datetime(r.index)
            ts = pd.Timestamp(fold_cfg.split.test_start)
            te = pd.Timestamp(fold_cfg.split.test_end or r.index.max())
            r = r[(r.index >= ts) & (r.index <= te)]
            ew_r = results["equal_weight_1_7"].returns.reindex(r.index).fillna(0.0)
            return information_ratio(r, ew_r)

        m1_ir = _fold_ir("m1_only")
        ecdf_ir = _fold_ir(STRATEGY_M1_M2_M3_ECDF)

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
                "m1_ir": m1_ir,
                "ecdf_ir": ecdf_ir,
                "ir_edge_vs_ew": ecdf_ir,
                "m2_auc": m2m.get("auc", float("nan")),
                "m2_auc_pr": m2m.get("auc_pr", float("nan")),
                "m2_n_trades": m2m.get("n_trades", 0),
            }
        )

    return pd.DataFrame(rows)


def run_walk_forward_ir_evaluation(
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    winner_spec: Any | None = None,
    eval_cfg: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Walk-forward IR vs EW for ECDF, M1, and optional intervention winner."""
    from src.backtest import _run_backtest, strategy_weights_from_panel
    from src.ir_interventions import build_intervention_weights
    from src.position_sizing import SizingMode

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
            panel, summary = _fit_fold_stack(base_panel, feature_cols, returns_wide, fold_cfg)
        except (ValueError, KeyError) as exc:
            logger.warning("Walk-forward IR fold %s failed: %s", fold.get("fold_id"), exc)
            continue

        results = summary["results"]
        train, _ = split_train_test(panel, fold_cfg)
        train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
        ew_r = results["equal_weight_1_7"].returns

        def _ir(strat: str) -> float:
            r = results[strat].returns.copy()
            r.index = pd.to_datetime(r.index)
            ts = pd.Timestamp(fold_cfg.split.test_start)
            te = pd.Timestamp(fold_cfg.split.test_end or r.index.max())
            r = r[(r.index >= ts) & (r.index <= te)]
            return information_ratio(r, ew_r.reindex(r.index).fillna(0.0))

        row: dict[str, Any] = {
            **fold,
            "m1_ir": _ir("m1_only"),
            "ecdf_ir": _ir(STRATEGY_M1_M2_M3_ECDF),
            "ir_edge_vs_ew": _ir(STRATEGY_M1_M2_M3_ECDF),
            "winner_ir": float("nan"),
            "winner_ir_edge_vs_ew": float("nan"),
        }

        if winner_spec is not None and winner_spec.kind != "baseline":
            m1_w = strategy_weights_from_panel(panel, returns_wide, fold_cfg, SizingMode.LINEAR, use_m2=False)
            ew_w = results["equal_weight_1_7"].weights
            w = build_intervention_weights(
                panel,
                returns_wide,
                fold_cfg,
                train_proba,
                m1_w,
                ew_w,
                winner_spec,
            )
            tc = fold_cfg.portfolio.transaction_cost_bps
            bt = _run_backtest(winner_spec.name, w, returns_wide, tc)
            r = bt.returns.copy()
            r.index = pd.to_datetime(r.index)
            ts = pd.Timestamp(fold_cfg.split.test_start)
            te = pd.Timestamp(fold_cfg.split.test_end or r.index.max())
            r = r[(r.index >= ts) & (r.index <= te)]
            wir = information_ratio(r, ew_r.reindex(r.index).fillna(0.0))
            row["winner_ir"] = wir
            row["winner_ir_edge_vs_ew"] = wir

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_walk_forward_ir_stability(
    wf_ir: pd.DataFrame,
    *,
    ir_col: str = "ecdf_ir",
    min_positive_fold_share: float = 0.5,
) -> dict[str, Any]:
    if wf_ir.empty or ir_col not in wf_ir.columns:
        return {"stable_ir": False, "n_folds": 0, "verdict": "insufficient_data"}

    ir = wf_ir[ir_col].astype(float)
    positive = int((ir > 0).sum())
    n = len(ir)
    mean_ir = float(ir.mean())
    stable = mean_ir > 0 and (positive / n) >= min_positive_fold_share
    return {
        "stable_ir": stable,
        "n_folds": n,
        "positive_ir_folds": positive,
        "mean_ir": mean_ir,
        "verdict": "stable_majority" if stable else "unstable",
    }


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
        edge_chart = save_walk_forward_edge_chart(walk_forward, output_dir)
        if edge_chart:
            saved.append(edge_chart)

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
    m1_weight_walk_forward: pd.DataFrame | None = None,
    m1_weight_decision: dict[str, Any] | None = None,
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

    m1_weight_wf = m1_weight_walk_forward if m1_weight_walk_forward is not None else pd.DataFrame()
    m1_weight_decision_out = m1_weight_decision or {}
    if m1_weight_wf.empty and m1_weight_decision_out == {} and eval_cfg.walk_forward_enabled:
        try:
            from src.factor_analysis import run_m1_weight_walk_forward_validation

            logger.info("Running M1 IC-proportional weight walk-forward validation")
            m1_weight_wf, m1_weight_decision_out = run_m1_weight_walk_forward_validation(
                base_panel, feature_cols, returns_wide, cfg
            )
        except Exception as exc:
            logger.warning("M1 weight walk-forward validation failed: %s", exc)

    summary: dict[str, Any] = {
        "walk_forward": walk_forward,
        "transaction_cost_sensitivity": tc_sensitivity,
        "m1_weight_walk_forward": m1_weight_wf,
        "m1_weight_decision": m1_weight_decision_out,
    }
    if not walk_forward.empty:
        summary["walk_forward_mean_ecdf_edge"] = float(walk_forward["ecdf_sharpe_edge_vs_m1"].mean())
        summary["walk_forward_mean_m2_auc"] = float(walk_forward["m2_auc"].mean())
        summary["walk_forward_stability"] = analyze_walk_forward_stability(
            walk_forward,
            production_test_start=cfg.split.test_start,
        )
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


def analyze_walk_forward_stability(
    walk_forward: pd.DataFrame,
    *,
    production_test_start: str = "2021-01-01",
    min_positive_fold_share: float = 0.5,
) -> dict[str, Any]:
    """Summarize whether ECDF Sharpe edge vs M1-only is stable across walk-forward folds."""
    if walk_forward.empty:
        return {
            "verdict": "insufficient_data",
            "stable_ecdf_edge": False,
            "n_folds": 0,
            "summary": "No walk-forward folds completed.",
        }

    wf = walk_forward.copy()
    edge = wf["ecdf_sharpe_edge_vs_m1"].astype(float)
    wf["is_production_window"] = pd.to_datetime(wf["test_start"]) >= pd.Timestamp(production_test_start)
    pre_prod = wf[~wf["is_production_window"]]
    prod = wf[wf["is_production_window"]]

    positive = int((edge > 0).sum())
    n = len(wf)
    mean_edge = float(edge.mean())
    median_edge = float(edge.median())
    mean_ecdf = float(wf["ecdf_sharpe"].mean())
    mean_m1 = float(wf["m1_only_sharpe"].mean())
    mean_ew = float(wf["equal_weight_sharpe"].mean())
    ecdf_beats_ew = int((wf["ecdf_sharpe"] > wf["equal_weight_sharpe"]).sum())

    stable = mean_edge > 0 and (positive / n) >= min_positive_fold_share

    if stable and (edge > 0).all():
        verdict = "stable_all_folds"
    elif stable:
        verdict = "stable_majority"
    elif mean_edge > 0:
        verdict = "mixed_positive_mean"
    else:
        verdict = "unstable"

    production_edge = (
        float(prod["ecdf_sharpe_edge_vs_m1"].mean()) if not prod.empty else float("nan")
    )
    pre_edge = (
        float(pre_prod["ecdf_sharpe_edge_vs_m1"].mean()) if not pre_prod.empty else float("nan")
    )
    production_only_outlier = (
        not prod.empty
        and not pre_prod.empty
        and pd.notna(production_edge)
        and pd.notna(pre_edge)
        and production_edge > 0.05
        and pre_edge <= 0
    )

    return {
        "verdict": verdict,
        "stable_ecdf_edge": stable,
        "n_folds": n,
        "positive_edge_folds": positive,
        "positive_edge_pct": positive / n,
        "mean_ecdf_edge": mean_edge,
        "median_ecdf_edge": median_edge,
        "std_ecdf_edge": float(edge.std()) if n > 1 else 0.0,
        "mean_ecdf_sharpe": mean_ecdf,
        "mean_m1_sharpe": mean_m1,
        "mean_equal_weight_sharpe": mean_ew,
        "ecdf_beats_equal_weight_folds": ecdf_beats_ew,
        "pre_production_mean_edge": pre_edge,
        "pre_production_positive_folds": int((pre_prod["ecdf_sharpe_edge_vs_m1"] > 0).sum())
        if not pre_prod.empty
        else 0,
        "pre_production_n_folds": len(pre_prod),
        "production_mean_edge": production_edge,
        "production_n_folds": len(prod),
        "production_only_outlier": production_only_outlier,
        "mean_m2_auc": float(wf["m2_auc"].mean()) if "m2_auc" in wf.columns else float("nan"),
        "summary": _walk_forward_verdict_text(
            verdict,
            mean_edge=mean_edge,
            positive=positive,
            n=n,
            production_only_outlier=production_only_outlier,
        ),
    }


def _walk_forward_verdict_text(
    verdict: str,
    *,
    mean_edge: float,
    positive: int,
    n: int,
    production_only_outlier: bool,
) -> str:
    if verdict == "stable_all_folds":
        base = f"ECDF Sharpe edge vs M1-only is positive in all {n} folds (mean +{mean_edge:.3f})."
    elif verdict == "stable_majority":
        base = (
            f"ECDF edge is positive in {positive}/{n} folds with mean +{mean_edge:.3f} — "
            "stable under a majority-fold criterion."
        )
    elif verdict == "mixed_positive_mean":
        base = (
            f"Mean ECDF edge is +{mean_edge:.3f} but only {positive}/{n} folds are positive — "
            "mixed stability."
        )
    elif verdict == "unstable":
        base = (
            f"ECDF edge is not stable: mean {mean_edge:+.3f}, positive in only {positive}/{n} folds."
        )
    else:
        base = "Insufficient walk-forward data."
    if production_only_outlier:
        base += " Production 2021+ window is stronger than pre-2021 folds."
    return base


def save_walk_forward_edge_chart(walk_forward: pd.DataFrame, output_dir: Any) -> str | None:
    """Bar chart of ECDF Sharpe edge vs M1 by fold."""
    if walk_forward.empty or "ecdf_sharpe_edge_vs_m1" not in walk_forward.columns:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = walk_forward["fold_id"].astype(str)
    edge = walk_forward["ecdf_sharpe_edge_vs_m1"]
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in edge]
    ax.bar(x, edge, color=colors)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("ECDF Sharpe edge vs M1-only")
    ax.set_title("Walk-Forward ECDF Sharpe Edge by Fold")
    for i, (_, row) in enumerate(walk_forward.iterrows()):
        label = str(row["test_start"])[:4] if "test_start" in walk_forward.columns else str(i + 1)
        ax.annotate(
            label,
            (i, edge.iloc[i]),
            ha="center",
            va="bottom" if edge.iloc[i] >= 0 else "top",
            fontsize=7,
        )
    p = output_dir / "walk_forward_ecdf_edge.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p.name


def generate_walk_forward_analysis_report(
    walk_forward: pd.DataFrame,
    stability: dict[str, Any],
    report_path: Path,
    *,
    mode_name: str = "long_only",
    cfg: PipelineConfig | None = None,
    tc_sensitivity: pd.DataFrame | None = None,
) -> None:
    """Dedicated walk-forward report: stability verdict, Q&A, fold table."""
    from src.diagnostics import _fmt_num, _fmt_pct, _markdown_table

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_prefix = f"../data/backtests/{mode_name}/figures"
    ev = cfg.evaluation if cfg else None
    prod_start = cfg.split.test_start if cfg else "2021-01-01"

    lines = [
        "# Walk-Forward Analysis: ECDF Edge Stability",
        "",
        "**Research use only — not investment advice.**",
        "",
        "This report answers whether **M1+M2+M3 ECDF** improves risk-adjusted returns vs **M1-only** "
        "across **multiple out-of-sample windows**, not only the production test period (2021+).",
        "",
    ]
    if cfg is not None and ev is not None:
        lines.extend(
            [
                "## Method",
                "",
                f"- **Design:** expanding train window; first train end `{ev.walk_forward_first_train_end}`; "
                f"**{ev.walk_forward_test_years}-year** test blocks",
                f"- **Production window (config):** `{prod_start}` onward — compared but not the sole metric",
                "- **Per fold:** refit M1, M2, M3; backtest long-only; score test-block Sharpe",
                "- **Edge:** `ECDF Sharpe − M1-only Sharpe` on each fold's test window",
                "",
            ]
        )

    lines.extend(
        [
            "## Executive verdict",
            "",
            f"**{stability.get('summary', 'N/A')}**",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Folds completed | {stability.get('n_folds', 0)} |",
            f"| Stable (majority + positive mean edge)? | **{'Yes' if stability.get('stable_ecdf_edge') else 'No'}** |",
            f"| Mean ECDF Sharpe edge vs M1 | {_fmt_num(stability.get('mean_ecdf_edge', float('nan')))} |",
            f"| Median edge | {_fmt_num(stability.get('median_ecdf_edge', float('nan')))} |",
            f"| Folds with positive edge | {stability.get('positive_edge_folds', 0)} / {stability.get('n_folds', 0)} "
            f"({stability.get('positive_edge_pct', 0):.0%}) |",
            f"| Mean ECDF / M1 / EW Sharpe | "
            f"{_fmt_num(stability.get('mean_ecdf_sharpe', float('nan')))} / "
            f"{_fmt_num(stability.get('mean_m1_sharpe', float('nan')))} / "
            f"{_fmt_num(stability.get('mean_equal_weight_sharpe', float('nan')))} |",
            f"| ECDF beats equal-weight (folds) | {stability.get('ecdf_beats_equal_weight_folds', 0)} / "
            f"{stability.get('n_folds', 0)} |",
            f"| Mean M2 AUC (test, across folds) | {_fmt_num(stability.get('mean_m2_auc', float('nan')))} |",
            "",
        ]
    )

    lines.extend(
        [
            "## Pre-2021 vs production window",
            "",
            "| Segment | Folds | Mean ECDF edge vs M1 | Positive folds |",
            "| --- | ---: | ---: | ---: |",
            f"| Pre-`{prod_start}` test blocks | {stability.get('pre_production_n_folds', 0)} | "
            f"{_fmt_num(stability.get('pre_production_mean_edge', float('nan')))} | "
            f"{stability.get('pre_production_positive_folds', 0)} |",
            f"| `{prod_start}`+ test blocks | {stability.get('production_n_folds', 0)} | "
            f"{_fmt_num(stability.get('production_mean_edge', float('nan')))} | "
            f"{stability.get('positive_edge_folds', 0) - stability.get('pre_production_positive_folds', 0)} |",
            "",
        ]
    )
    if stability.get("production_only_outlier"):
        lines.append(
            "> **Note:** The production 2021+ fold shows a larger edge than pre-2021 folds. "
            "Headline OOS metrics should be reported alongside this multi-window table."
        )
        lines.append("")

    lines.extend(
        [
            "## Key questions",
            "",
            "### 1. Is ECDF Sharpe edge vs M1 stable across folds?",
            "",
        ]
    )
    if stability.get("stable_ecdf_edge"):
        lines.append(
            f"**Yes (under majority criterion):** mean edge {_fmt_num(stability.get('mean_ecdf_edge'))}, "
            f"positive in {stability.get('positive_edge_folds')}/{stability.get('n_folds')} folds."
        )
    else:
        lines.append(
            f"**No / mixed:** mean edge {_fmt_num(stability.get('mean_ecdf_edge'))}, "
            f"positive in only {stability.get('positive_edge_folds')}/{stability.get('n_folds')} folds."
        )
    lines.extend(
        [
            "",
            "### 2. Is the 2021+ production result representative?",
            "",
        ]
    )
    if stability.get("production_only_outlier"):
        lines.append(
            "**Partially.** Pre-2021 folds show weaker or negative edge; the 2021+ block contributes "
            "disproportionately to the full-sample ECDF advantage."
        )
    elif (
        stability.get("pre_production_mean_edge", 0) > 0
        and stability.get("production_mean_edge", 0) > 0
        and stability.get("pre_production_mean_edge", 0) >= stability.get("production_mean_edge", 0)
    ):
        lines.append(
            "**Broadly yes, not 2021-specific.** Both eras show positive mean ECDF edge; "
            f"pre-2021 mean edge ({stability.get('pre_production_mean_edge'):.3f}) is actually "
            f"**higher** than production-era folds ({stability.get('production_mean_edge'):.3f}), "
            "so the single 2021+ headline is not an isolated outlier."
        )
    elif stability.get("pre_production_mean_edge", 0) > 0 and stability.get("production_mean_edge", 0) > 0:
        lines.append(
            "**Yes.** Both pre-production and production-era folds show positive mean ECDF edge vs M1-only."
        )
    else:
        lines.append(
            "**Mixed.** Compare the fold table below — some eras favor ECDF sizing, others favor M1-only levels."
        )
    lines.extend(
        [
            "",
            "### 3. Does ECDF add value beyond equal-weight?",
            "",
            f"ECDF Sharpe exceeds equal-weight in **{stability.get('ecdf_beats_equal_weight_folds', 0)}** of "
            f"**{stability.get('n_folds', 0)}** folds (mean ECDF Sharpe "
            f"{_fmt_num(stability.get('mean_ecdf_sharpe'))} vs EW "
            f"{_fmt_num(stability.get('mean_equal_weight_sharpe'))}).",
            "",
            "### 4. What is M2 doing across folds?",
            "",
            f"Mean test AUC **{_fmt_num(stability.get('mean_m2_auc'))}** — ranking remains modest; "
            "ECDF edge is driven by **vol/drawdown shaping** from `p_success`, not binary filtering.",
            "",
            "## Fold-level results",
            "",
        ]
    )
    if not walk_forward.empty:
        disp = walk_forward.copy()
        for col in (
            "m1_only_sharpe",
            "m1_only_ann_return",
            "ecdf_sharpe",
            "ecdf_ann_return",
            "ecdf_sharpe_edge_vs_m1",
            "ecdf_return_edge_vs_m1",
            "equal_weight_sharpe",
            "m2_auc",
        ):
            if col in disp.columns:
                if "ann_return" in col or "return_edge" in col:
                    disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
                else:
                    disp[col] = disp[col].map(_fmt_num)
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![Walk-forward Sharpe by fold]({fig_prefix}/walk_forward_sharpe.png)")
        lines.append("")
        lines.append(f"![ECDF Sharpe edge by fold]({fig_prefix}/walk_forward_ecdf_edge.png)")
        lines.append("")

    if tc_sensitivity is not None and not tc_sensitivity.empty:
        lines.extend(["## Transaction-cost sensitivity (production window)", ""])
        disp = tc_sensitivity.copy()
        for col in ("annualized_return", "sharpe", "max_drawdown", "hit_rate", "ecdf_sharpe_edge_vs_m1"):
            if col in disp.columns:
                disp[col] = disp[col].map(_fmt_num)
        lines.append(_markdown_table(disp))
        lines.append("")

    lines.extend(
        [
            "## Implications",
            "",
            "- Report **fold-level** ECDF edge alongside the single 2021+ test table in `final_report.md`.",
            "- If edge is fold-dependent, prefer **regime-conditioned M3** or accept ECDF as a drawdown tool, not return engine.",
            "- M1-only remains the return-oriented sleeve when ECDF edge is negative on a fold.",
            "",
            "Related: [evaluation_analysis.md](evaluation_analysis.md) · [final_report.md](final_report.md)",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


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
