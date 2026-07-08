"""Walk-forward validation of IC-proportional M1 weights; optional config update."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.backtest import returns_wide_from_panel
from src.config import clone_config_with_m1_allow_short, clone_config_with_m1_weights, load_config
from src.factor_analysis import run_m1_weight_walk_forward_validation
from src.feature_engineering import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_base_panel(root: Path) -> pd.DataFrame:
    panel_path = root / "data/features/model_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing {panel_path}. Run the pipeline once to build the feature panel."
        )
    panel = pd.read_parquet(panel_path)
    if "date" in panel.columns and "ticker" in panel.columns:
        panel = panel.set_index(["date", "ticker"]).sort_index()
    return panel


def _round_weights(weights: dict[str, float], decimals: int = 4) -> dict[str, float]:
    rounded = {k: round(float(v), decimals) for k, v in weights.items()}
    total = sum(rounded.values())
    if total > 0:
        rounded = {k: v / total for k, v in rounded.items()}
    return rounded


def apply_weights_to_config(config_path: Path, weights: dict[str, float]) -> None:
    data = yaml.safe_load(config_path.read_text())
    weights = _round_weights(weights)
    data.setdefault("models", {}).setdefault("m1", {})["weights"] = weights
    config_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def run_research(
    config_path: Path,
    *,
    project_root: Path | None = None,
    apply: bool = False,
    output_dir: Path | None = None,
) -> dict:
    root = project_root or Path.cwd()
    cfg = load_config(config_path if config_path.is_absolute() else root / config_path)
    cfg = clone_config_with_m1_allow_short(cfg, allow_short=False)

    panel = _load_base_panel(root)
    feature_cols = get_feature_columns(panel.reset_index())
    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)

    logger.info("Running M1 weight walk-forward validation (long-only)")
    summary, decision = run_m1_weight_walk_forward_validation(
        panel, feature_cols, returns_wide, cfg
    )

    out_dir = output_dir or (root / "data/backtests/long_only/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        summary.to_csv(out_dir / "m1_weight_walk_forward.csv", index=False)
    (out_dir / "m1_weight_walk_forward_decision.json").write_text(json.dumps(decision, indent=2))

    logger.info("Decision: %s", decision.get("reason"))
    if decision.get("apply_ic_weights") and apply:
        mean_w = decision.get("mean_ic_weights") or {}
        if mean_w:
            target = config_path if config_path.is_absolute() else root / config_path
            apply_weights_to_config(target, mean_w)
            logger.info("Updated %s with mean walk-forward IC weights: %s", target, mean_w)
        else:
            logger.warning("apply_ic_weights=True but mean_ic_weights missing; config unchanged.")
    elif decision.get("apply_ic_weights"):
        logger.info("Walk-forward supports IC weights; re-run with --apply to update config.")

    return {"summary": summary, "decision": decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validate IC-proportional M1 weights")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write mean IC weights to config when walk-forward decision is positive",
    )
    args = parser.parse_args(argv)
    try:
        run_research(Path(args.config), apply=args.apply)
        return 0
    except Exception:
        logger.exception("M1 weight research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
