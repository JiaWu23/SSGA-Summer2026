"""Run strategy walk-forward validation and write ECDF stability report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.backtest import returns_wide_from_panel
from src.config import clone_config_with_m1_allow_short, load_config
from src.evaluation import (
    analyze_walk_forward_stability,
    generate_walk_forward_analysis_report,
    run_extended_evaluation,
    run_walk_forward_evaluation,
    save_evaluation_charts,
)
from src.feature_engineering import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_panel(root: Path) -> pd.DataFrame:
    path = root / "data/features/model_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the pipeline first.")
    panel = pd.read_parquet(path)
    if "date" in panel.columns and "ticker" in panel.columns:
        panel = panel.set_index(["date", "ticker"]).sort_index()
    return panel


def run_walk_forward_research(
    config_path: Path,
    *,
    project_root: Path | None = None,
    skip_tc: bool = False,
) -> dict:
    root = project_root or Path.cwd()
    cfg = load_config(config_path if config_path.is_absolute() else root / config_path)
    cfg = clone_config_with_m1_allow_short(cfg, allow_short=False)

    if not cfg.evaluation.walk_forward_enabled:
        logger.warning("evaluation.walk_forward_enabled is false in config; enabling for this run.")
        from src.config import apply_config_overrides

        cfg = apply_config_overrides(cfg, {"evaluation": {"walk_forward_enabled": True}})

    panel = _load_panel(root)
    feature_cols = get_feature_columns(panel.reset_index())
    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)

    logger.info("Running strategy walk-forward evaluation (long-only)")
    walk_forward = run_walk_forward_evaluation(panel, feature_cols, returns_wide, cfg)

    stability = analyze_walk_forward_stability(walk_forward, production_test_start=cfg.split.test_start)

    tc = pd.DataFrame()
    if not skip_tc:
        from src.run_pipeline import run_all_strategies
        from src.model_m1 import build_m1_model, split_train_test
        from src.model_m2 import fit_m2, predict_m2
        from src.model_m3 import attach_m3_to_panel
        from src.labels import build_meta_labels

        train, test = split_train_test(panel, cfg)
        m1 = build_m1_model(cfg)
        X_train = train[feature_cols].fillna(0)
        fwd_col = f"forward_return_{cfg.labels.horizon_weeks}w"
        returns_train = returns_wide.loc[
            (returns_wide.index >= pd.Timestamp(cfg.split.train_start))
            & (returns_wide.index <= pd.Timestamp(cfg.split.train_end))
        ]
        m1.fit(
            X_train,
            train["m1_target"],
            forward_returns=train[fwd_col],
            panel=train,
            returns_wide=returns_train,
            portfolio_cfg=cfg.portfolio,
        )
        X_panel = panel[feature_cols].fillna(0)
        panel_scored = build_meta_labels(
            panel,
            m1.predict_signal(X_panel),
            m1.predict_score(X_panel),
            cfg,
        )
        m2_model, _ = fit_m2(panel_scored, cfg)
        panel_scored = predict_m2(m2_model, panel_scored, cfg)
        train, _ = split_train_test(panel_scored, cfg)
        train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
        panel_scored = attach_m3_to_panel(panel_scored, cfg, train_proba=train_proba)
        results = run_all_strategies(panel_scored, returns_wide, cfg, train_proba=train_proba)
        from src.evaluation import run_transaction_cost_sensitivity

        tc = run_transaction_cost_sensitivity(
            results,
            returns_wide,
            cfg,
            test_start=cfg.split.test_start,
            test_end=cfg.split.test_end,
        )

    eval_dir = root / "data/backtests/long_only/evaluation"
    fig_dir = root / "data/backtests/long_only/figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not walk_forward.empty:
        walk_forward.to_csv(eval_dir / "walk_forward_summary.csv", index=False)
    if not tc.empty:
        tc.to_csv(eval_dir / "transaction_cost_sensitivity.csv", index=False)
    (eval_dir / "walk_forward_stability.json").write_text(json.dumps(stability, indent=2))

    save_evaluation_charts(walk_forward, tc, fig_dir)

    reports_dir = root / "reports"
    generate_walk_forward_analysis_report(
        walk_forward,
        stability,
        reports_dir / "walk_forward_analysis.md",
        cfg=cfg,
        tc_sensitivity=tc,
    )
    from src.evaluation import generate_evaluation_report

    eval_summary = {
        "walk_forward": walk_forward,
        "transaction_cost_sensitivity": tc,
        "walk_forward_mean_ecdf_edge": stability.get("mean_ecdf_edge"),
        "walk_forward_mean_m2_auc": stability.get("mean_m2_auc"),
        "walk_forward_stability": stability,
        "ecdf_edge_persists_at_25bps": bool(
            not tc.empty
            and (tc[tc["strategy"] == "m1_m2_m3_ecdf"]["ecdf_sharpe_edge_vs_m1"].get(25.0, 0) > 0)
            if "ecdf_sharpe_edge_vs_m1" in tc.columns
            else False
        ),
    }
    generate_evaluation_report(
        eval_summary,
        reports_dir / "evaluation_analysis.md",
        mode_name="long_only",
        cfg=cfg,
    )

    logger.info("Verdict: %s", stability.get("summary"))
    return {"walk_forward": walk_forward, "stability": stability, "tc": tc}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward ECDF stability research")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-tc", action="store_true", help="Skip transaction-cost sensitivity")
    args = parser.parse_args(argv)
    try:
        run_walk_forward_research(Path(args.config), skip_tc=args.skip_tc)
        return 0
    except Exception:
        logger.exception("Walk-forward research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
