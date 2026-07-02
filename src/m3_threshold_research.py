"""Run M3 threshold sweep and write analysis report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.backtest import returns_wide_from_panel
from src.config import clone_config_with_m1_allow_short, load_config
from src.feature_engineering import get_feature_columns
from src.labels import build_meta_labels
from src.model_m1 import build_m1_model, split_train_test
from src.model_m2 import fit_m2, predict_m2
from src.model_m3 import attach_m3_to_panel
from src.m3_threshold_sweep import (
    generate_m3_threshold_report,
    recommend_m3_thresholds,
    save_m3_threshold_charts,
    sweep_m3_thresholds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_predictions_panel(root: Path) -> pd.DataFrame | None:
    path = root / "data/predictions/long_only/panel_with_predictions.parquet"
    if not path.exists():
        return None
    panel = pd.read_parquet(path)
    if "date" in panel.columns and "ticker" in panel.columns:
        panel = panel.set_index(["date", "ticker"]).sort_index()
    return panel


def _fit_production_panel(root: Path, cfg) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(root / "data/features/model_panel.parquet")
    if "date" in panel.columns:
        panel = panel.set_index(["date", "ticker"]).sort_index()
    feature_cols = get_feature_columns(panel.reset_index())
    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)
    train, _ = split_train_test(panel, cfg)
    m1 = build_m1_model(cfg)
    X_train = train[feature_cols].fillna(0)
    fwd = f"forward_return_{cfg.labels.horizon_weeks}w"
    rt = returns_wide.loc[
        (returns_wide.index >= pd.Timestamp(cfg.split.train_start))
        & (returns_wide.index <= pd.Timestamp(cfg.split.train_end))
    ]
    m1.fit(
        X_train,
        train["m1_target"],
        forward_returns=train[fwd],
        panel=train,
        returns_wide=rt,
        portfolio_cfg=cfg.portfolio,
    )
    X = panel[feature_cols].fillna(0)
    panel = build_meta_labels(panel, m1.predict_signal(X), m1.predict_score(X), cfg)
    m2, _ = fit_m2(panel, cfg)
    panel = predict_m2(m2, panel, cfg)
    train, _ = split_train_test(panel, cfg)
    tp = train.loc[train["M1_signal"] != 0, "p_success"]
    panel = attach_m3_to_panel(panel, cfg, train_proba=tp)
    return panel, feature_cols


def run_m3_threshold_research(
    config_path: Path,
    *,
    project_root: Path | None = None,
) -> dict:
    root = project_root or Path.cwd()
    cfg = clone_config_with_m1_allow_short(
        load_config(config_path if config_path.is_absolute() else root / config_path),
        allow_short=False,
    )

    panel = _load_predictions_panel(root)
    if panel is None or "p_success" not in panel.columns:
        logger.info("Fitting production panel (no cached predictions)")
        panel, _ = _fit_production_panel(root, cfg)
    else:
        logger.info("Using cached predictions panel")

    train, test = split_train_test(panel, cfg)
    train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)

    logger.info("Sweeping M3 thresholds on test window %s+", cfg.split.test_start)
    sweep = sweep_m3_thresholds(
        panel,
        test,
        returns_wide,
        cfg,
        train_proba=train_proba,
    )
    baseline_t = cfg.m3.threshold or cfg.m2.threshold
    recommendation = recommend_m3_thresholds(sweep, baseline_threshold=float(baseline_t))

    out_dir = root / "data/backtests/long_only"
    eval_dir = out_dir / "evaluation"
    fig_dir = out_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(eval_dir / "m3_threshold_sweep.csv", index=False)
    (eval_dir / "m3_threshold_recommendation.json").write_text(json.dumps(recommendation, indent=2))
    save_m3_threshold_charts(sweep, fig_dir)
    generate_m3_threshold_report(
        sweep,
        recommendation,
        root / "reports/m3_threshold_analysis.md",
        cfg=cfg,
    )

    bin_rec = recommendation.get("binary") or {}
    logger.info(
        "Binary: baseline T=0.55 recall≈%.3f; recommended T=%s Sharpe=%.4f",
        bin_rec.get("baseline_recall", float("nan")),
        bin_rec.get("recommended_threshold"),
        bin_rec.get("test_sharpe", float("nan")),
    )
    return {"sweep": sweep, "recommendation": recommendation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3 threshold sweep research")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    try:
        run_m3_threshold_research(Path(args.config))
        return 0
    except Exception:
        logger.exception("M3 threshold research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
