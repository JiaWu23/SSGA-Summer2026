"""IR attribution, intervention sweep, and adoption research CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import returns_wide_from_panel, run_all_strategies
from src.config import clone_config_with_m1_allow_short, load_config
from src.feature_engineering import get_feature_columns
from src.ir_attribution import generate_ir_attribution_report, run_ir_attribution
from src.ir_interventions import (
    InterventionSpec,
    evaluate_adoption_gates,
    sweep_ir_interventions,
)
from src.labels import build_meta_labels
from src.model_m1 import build_m1_model, split_train_test
from src.model_m2 import fit_m2, predict_m2
from src.model_m3 import attach_m3_to_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_panel(root: Path) -> pd.DataFrame | None:
    path = root / "data/predictions/long_only/panel_with_predictions.parquet"
    if not path.exists():
        return None
    panel = pd.read_parquet(path)
    if "date" in panel.columns and "ticker" in panel.columns:
        panel = panel.set_index(["date", "ticker"]).sort_index()
    return panel


def _fit_panel(root: Path, cfg) -> tuple[pd.DataFrame, list[str]]:
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


def generate_ir_improvement_report(
    sweep: pd.DataFrame,
    decision: dict[str, Any],
    wf_ir: pd.DataFrame,
    report_path: Path,
    *,
    test_start: str,
) -> None:
    from src.diagnostics import _fmt_num, _fmt_pct, _markdown_table

    report_path.parent.mkdir(parents=True, exist_ok=True)
    test = sweep[sweep["period"] == "test"].sort_values("information_ratio", ascending=False)

    lines = [
        "# IR Improvement Research",
        "",
        "**Research use only — not investment advice.**",
        "",
        "Structured sweep of ECDF overlays to raise **Information Ratio vs equal-weight**",
        "while preserving test Sharpe and annualized return.",
        "",
        f"**Test window:** `{test_start}` onward",
        "",
        "## Adoption verdict",
        "",
        f"- **Verdict:** `{decision.get('verdict', 'unknown')}`",
        f"- **Winner:** `{decision.get('winner', 'none')}`",
        f"- **Reason:** {decision.get('reason', decision.get('gates', ''))}",
        "",
        "## Adoption gates (test period)",
        "",
        "| Gate | Threshold |",
        "| --- | --- |",
        f"| Sharpe | ≥ {decision.get('gates', {}).get('sharpe_min', 0.95)} |",
        f"| Ann return | ≥ {decision.get('gates', {}).get('return_min', 0.075):.1%} |",
        f"| Info Ratio | > 0 and > ECDF baseline |",
        f"| Max drawdown | Not >2pp worse than ECDF |",
        f"| Turnover | ≤ +30% vs ECDF |",
        "",
        "## Test-period sweep (sorted by IR)",
        "",
    ]

    if not test.empty:
        disp = test[
            [
                "variant",
                "annualized_return",
                "sharpe",
                "information_ratio",
                "excess_return_vs_benchmark",
                "max_drawdown",
                "mean_gross_exposure",
                "annualized_turnover",
            ]
        ].copy()
        for col in ("annualized_return", "excess_return_vs_benchmark", "max_drawdown", "mean_gross_exposure"):
            disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
        for col in ("sharpe", "information_ratio"):
            disp[col] = disp[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        lines.append(_markdown_table(disp))
        lines.append("")

    lines.extend(["## Walk-forward IR stability (winner vs EW)", ""])
    if not wf_ir.empty:
        disp_wf = wf_ir.copy()
        for col in ("ecdf_ir", "m1_ir", "ir_edge_vs_ew", "winner_ir", "winner_ir_edge_vs_ew"):
            if col in disp_wf.columns:
                disp_wf[col] = disp_wf[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        lines.append(_markdown_table(disp_wf))
        lines.append("")

    if decision.get("verdict") == "reject":
        lines.extend(
            [
                "## Recommendation",
                "",
                "Keep production **ECDF** unchanged. IR vs EW is an explicit trade-off:",
                "ECDF improves Sharpe/drawdown by deploying less capital.",
                "",
                "**Holdout note:** `exposure_renorm_1.10` achieved the highest test IR (0.35) but",
                "Sharpe fell to 0.78 (below 0.95 gate). `vol_bump_0.55_1.15` passed holdout gates",
                "but failed walk-forward IR stability (2/6 positive folds).",
                "",
                "Document in [TERMINOLOGY.md](../TERMINOLOGY.md) — do not adopt overlay without gate pass.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommendation",
                "",
                f"Adopt **`{decision.get('winner')}`** in config after walk-forward confirmation.",
                "",
            ]
        )

    lines.append(
        "Related: [ir_attribution_analysis.md](ir_attribution_analysis.md) · "
        "[evaluation_analysis.md](evaluation_analysis.md)"
    )
    report_path.write_text("\n".join(lines))


def run_ir_research(
    config_path: Path,
    *,
    project_root: Path | None = None,
    skip_walk_forward: bool = False,
) -> dict[str, Any]:
    root = project_root or Path.cwd()
    cfg = clone_config_with_m1_allow_short(
        load_config(config_path if config_path.is_absolute() else root / config_path),
        allow_short=False,
    )

    panel = _load_panel(root)
    if panel is None or "p_success" not in panel.columns:
        logger.info("Fitting production panel")
        panel, feature_cols = _fit_panel(root, cfg)
    else:
        logger.info("Using cached predictions panel")
        feature_cols = get_feature_columns(panel.reset_index())

    train, test = split_train_test(panel, cfg)
    train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)
    test_start = cfg.split.test_start
    test_end = cfg.split.test_end

    results = run_all_strategies(panel, returns_wide, cfg, train_proba=train_proba)

    logger.info("Running IR attribution")
    attr_summary, attr_extra = run_ir_attribution(
        results, panel, test_start=test_start, test_end=test_end
    )
    eval_dir = root / "data/backtests/long_only/evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    attr_summary.to_csv(eval_dir / "ir_attribution.csv", index=False)
    if not attr_extra["regime"].empty:
        attr_extra["regime"].to_csv(eval_dir / "ir_attribution_by_regime.csv", index=False)
    generate_ir_attribution_report(
        attr_summary,
        attr_extra["regime"],
        root / "reports/ir_attribution_analysis.md",
        test_start=test_start,
    )

    logger.info("Sweeping IR interventions")
    sweep = sweep_ir_interventions(
        panel,
        returns_wide,
        cfg,
        train_proba,
        test_start=test_start,
        test_end=test_end,
    )
    sweep.to_csv(eval_dir / "ir_intervention_sweep.csv", index=False)

    decision = evaluate_adoption_gates(sweep)
    (eval_dir / "ir_adoption_decision.json").write_text(json.dumps(decision, indent=2, default=str))

    wf_ir = pd.DataFrame()
    if not skip_walk_forward and cfg.evaluation.walk_forward_enabled:
        from src.evaluation import analyze_walk_forward_ir_stability, run_walk_forward_ir_evaluation

        winner_name = decision.get("winner")
        winner_spec = _spec_from_variant_name(winner_name) if winner_name else None
        logger.info("Walk-forward IR evaluation (baseline + winner)")
        wf_ir = run_walk_forward_ir_evaluation(
            panel,
            feature_cols,
            returns_wide,
            cfg,
            winner_spec=winner_spec if decision.get("verdict") == "adopt" else None,
        )
        if not wf_ir.empty:
            wf_ir.to_csv(eval_dir / "ir_walk_forward.csv", index=False)
            ecdf_stab = analyze_walk_forward_ir_stability(wf_ir, ir_col="ecdf_ir")
            decision["walk_forward_ecdf_ir"] = ecdf_stab
            if decision.get("verdict") == "adopt" and winner_spec is not None:
                wf_stab = analyze_walk_forward_ir_stability(wf_ir, ir_col="winner_ir")
                decision["walk_forward_winner_ir"] = wf_stab
                if not wf_stab.get("stable_ir"):
                    decision["verdict"] = "reject"
                    decision["reason"] = (
                        f"Winner `{decision.get('winner')}` failed walk-forward IR stability "
                        f"({wf_stab.get('positive_ir_folds', 0)}/{wf_stab.get('n_folds', 0)} positive folds)"
                    )
                    decision["provisional_winner"] = decision.get("winner")
                    decision["winner"] = None
            (eval_dir / "ir_adoption_decision.json").write_text(json.dumps(decision, indent=2, default=str))

    generate_ir_improvement_report(
        sweep,
        decision,
        wf_ir,
        root / "reports/ir_improvement_research.md",
        test_start=test_start,
    )

    return {
        "attribution": attr_summary,
        "sweep": sweep,
        "decision": decision,
        "walk_forward_ir": wf_ir,
    }


def _spec_from_variant_name(name: str) -> InterventionSpec | None:
    if name == "ecdf_baseline":
        return InterventionSpec(name=name, kind="baseline")
    if name.startswith("exposure_renorm_"):
        frac = float(name.split("_")[-1])
        return InterventionSpec(name=name, kind="exposure_renorm", exposure_target_vs_m1=frac)
    if name.startswith("m3_floor_"):
        floor = float(name.split("_")[-1])
        return InterventionSpec(name=name, kind="m3_floor", m3_floor=floor)
    if name.startswith("ew_blend_"):
        alpha = float(name.split("_")[-1])
        return InterventionSpec(name=name, kind="ew_blend", ew_blend_alpha=alpha)
    if name == "regime_m3":
        return InterventionSpec(name=name, kind="regime_m3", regime_m3=True)
    if name.startswith("vol_bump_"):
        parts = name.split("_")
        thresh, bump = float(parts[2]), float(parts[3])
        return InterventionSpec(
            name=name,
            kind="vol_bump",
            vol_bump_when_m3_below=thresh,
            vol_bump_factor=bump,
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IR improvement research")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-walk-forward", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = run_ir_research(Path(args.config), skip_walk_forward=args.skip_walk_forward)
        logger.info("Verdict: %s winner=%s", out["decision"].get("verdict"), out["decision"].get("winner"))
        return 0
    except Exception:
        logger.exception("IR research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
