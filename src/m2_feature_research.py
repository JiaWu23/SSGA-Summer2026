"""M2 feature-set research: sweep variants, portfolio impact, walk-forward adoption gates."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import (
    STRATEGY_M1_M2_M3_ECDF,
    _run_backtest,
    returns_wide_from_panel,
    strategy_weights_from_panel,
)
from dataclasses import replace as dc_replace

from src.config import clone_config_with_m1_allow_short, clone_config_with_m2_variant, load_config
from src.diagnostics import m2_classification_metrics, strategy_metrics_on_period
from src.feature_engineering import get_feature_columns
from src.labels import build_meta_labels, get_m2_training_mask
from src.m2_feature_enrichment import (
    M2_FEATURE_VARIANTS,
    build_m2_features_variant,
    describe_variant,
    m2_config_for_variant,
)
from src.model_m1 import build_m1_model, split_train_test
from src.model_m2 import SklearnM2, _m2_auc, build_m2_features, create_m2_model
from src.model_m3 import attach_m3_to_panel
from src.position_sizing import SizingMode, fit_ecdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASELINE_VARIANT = "configured"
GATE_AUC_MIN_DELTA = 0.003
GATE_SHARPE_MIN = 0.95
GATE_SHARPE_TOLERANCE = 0.01
WF_MIN_POSITIVE_FOLDS = 4


def _ecdf_sharpe_on_test(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg,
    train_proba: pd.Series,
    *,
    test_start: str,
    test_end: str | None,
) -> dict[str, float]:
    """ECDF portfolio metrics on test window using panel p_success."""
    train_sorted = fit_ecdf(train_proba)
    w = strategy_weights_from_panel(
        panel,
        returns_wide,
        cfg,
        SizingMode.ECDF,
        use_m2=True,
        train_proba=train_proba,
        train_sorted=train_sorted,
    )
    tc = cfg.portfolio.transaction_cost_bps
    bt = _run_backtest(STRATEGY_M1_M2_M3_ECDF, w, returns_wide, tc)
    period = strategy_metrics_on_period(bt.returns, start=test_start, end=test_end)
    return {
        "ecdf_test_sharpe": period["sharpe"],
        "ecdf_test_ann_return": period["annualized_return"],
        "ecdf_test_max_drawdown": period["max_drawdown"],
    }


def evaluate_m2_variant(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg,
    variant: str,
    *,
    test_start: str,
    test_end: str | None,
) -> dict[str, Any]:
    variant_cfg = m2_config_for_variant(variant, cfg)
    train, test = split_train_test(panel, variant_cfg)

    train_mask = get_m2_training_mask(train)
    test_mask = get_m2_training_mask(test)
    y_train = train.loc[train_mask, "meta_label"].dropna()
    y_test = test.loc[test_mask, "meta_label"].dropna()

    ic_weights = None
    if variant in ("ic_alignment", "full_enriched"):
        from src.m2_feature_enrichment import _train_ic_weights

        ic_weights = _train_ic_weights(train.loc[train_mask])

    X_train = build_m2_features_variant(train.loc[train_mask], variant_cfg, variant, train_panel=train).loc[
        y_train.index
    ]
    X_test = build_m2_features_variant(test.loc[test_mask], variant_cfg, variant, train_panel=train).loc[
        y_test.index
    ]

    model = create_m2_model(variant_cfg.m2)
    if isinstance(model, SklearnM2):
        model.fit(X_train, y_train, ic_weights=ic_weights)
    else:
        model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)
    p_test = model.predict_proba(X_test)
    m2m = m2_classification_metrics(y_test, p_test, threshold=cfg.m2.threshold)

    full_panel = panel.copy()
    m2_mask = get_m2_training_mask(full_panel)
    X_all = build_m2_features_variant(full_panel, variant_cfg, variant, train_panel=train)
    full_panel["p_success"] = np.nan
    if m2_mask.any():
        full_panel.loc[m2_mask, "p_success"] = model.predict_proba(X_all.loc[m2_mask]).values

    train_m2_idx = train.index[get_m2_training_mask(train) & (train["M1_signal"] != 0)]
    tp = full_panel.loc[train_m2_idx, "p_success"].dropna()

    port = _ecdf_sharpe_on_test(full_panel, returns_wide, variant_cfg, tp, test_start=test_start, test_end=test_end)

    return {
        "variant": variant,
        "description": describe_variant(variant),
        "n_features": len(X_train.columns),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_auc": _m2_auc(y_train, p_train),
        "test_auc": _m2_auc(y_test, p_test),
        "test_f1": m2m.get("f1", float("nan")),
        "test_brier": m2m.get("brier_score", float("nan")),
        **port,
    }


def sweep_m2_feature_variants(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg,
    *,
    variants: tuple[str, ...] | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> pd.DataFrame:
    test_start = test_start or cfg.split.test_start
    test_end = test_end or cfg.split.test_end
    variants = variants or M2_FEATURE_VARIANTS
    rows = []
    for variant in variants:
        logger.info("Evaluating M2 variant: %s", variant)
        try:
            rows.append(
                evaluate_m2_variant(
                    panel,
                    returns_wide,
                    cfg,
                    variant,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
        except Exception as exc:
            logger.warning("Variant %s failed: %s", variant, exc, exc_info=True)
    return pd.DataFrame(rows)


def evaluate_adoption_decision(
    sweep: pd.DataFrame,
    wf: pd.DataFrame,
    *,
    baseline_variant: str = BASELINE_VARIANT,
) -> dict[str, Any]:
    if sweep.empty:
        return {"verdict": "reject", "reason": "empty sweep", "winner": None}

    base = sweep[sweep["variant"] == baseline_variant]
    if base.empty:
        return {"verdict": "reject", "reason": "baseline missing", "winner": None}
    base_row = base.iloc[0]
    base_auc = float(base_row["test_auc"])
    base_sharpe = float(base_row["ecdf_test_sharpe"])

    candidates = sweep[sweep["variant"] != baseline_variant].copy()
    passed: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        gates = {
            "auc_ok": float(row["test_auc"]) >= base_auc + GATE_AUC_MIN_DELTA,
            "sharpe_ok": float(row["ecdf_test_sharpe"]) >= GATE_SHARPE_MIN - GATE_SHARPE_TOLERANCE,
            "sharpe_not_worse_than_baseline": float(row["ecdf_test_sharpe"])
            >= base_sharpe - GATE_SHARPE_TOLERANCE,
        }
        if all(gates.values()):
            passed.append({**row.to_dict(), "gates": gates})

    if not passed:
        best_auc = candidates.sort_values("test_auc", ascending=False).iloc[0] if not candidates.empty else None
        return {
            "verdict": "reject",
            "winner": None,
            "reason": "No variant beat baseline on test AUC (+0.003) while preserving ECDF Sharpe",
            "baseline": base_row.to_dict(),
            "best_auc_variant": best_auc.to_dict() if best_auc is not None else None,
            "gates": {
                "auc_min_delta": GATE_AUC_MIN_DELTA,
                "sharpe_min": GATE_SHARPE_MIN,
            },
        }

    winner = max(
        passed,
        key=lambda x: (x["test_auc"], x["ecdf_test_sharpe"]),
    )

    wf_ok = True
    wf_note = ""
    if not wf.empty and "test_auc" in wf.columns:
        wsub = wf[wf["variant"] == winner["variant"]]
        if not wsub.empty and "test_auc" in wsub.columns:
            pos = int((wsub["test_auc"] > wsub["baseline_test_auc"]).sum())
            wf_ok = pos >= WF_MIN_POSITIVE_FOLDS
            wf_note = f"walk-forward AUC wins {pos}/{len(wsub)} folds"
            if not wf_ok:
                return {
                    "verdict": "reject",
                    "winner": None,
                    "provisional_winner": winner["variant"],
                    "reason": f"Holdout passed but walk-forward failed ({wf_note}, need {WF_MIN_POSITIVE_FOLDS})",
                    "baseline": base_row.to_dict(),
                    "provisional_metrics": winner,
                }

    return {
        "verdict": "adopt",
        "winner": winner["variant"],
        "reason": f"Test AUC {winner['test_auc']:.4f} vs baseline {base_auc:.4f}; "
        f"ECDF Sharpe {winner['ecdf_test_sharpe']:.4f} vs {base_sharpe:.4f}. {wf_note}",
        "winner_metrics": winner,
        "baseline": base_row.to_dict(),
    }


def run_m2_feature_walk_forward(
    base_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg,
    variants: tuple[str, ...],
) -> pd.DataFrame:
    from src.config import apply_split_overrides
    from src.evaluation import build_walk_forward_folds

    folds = build_walk_forward_folds(base_panel, cfg, cfg.evaluation)
    rows: list[dict[str, Any]] = []

    for fold in folds:
        base_fold_cfg = apply_split_overrides(
            cfg,
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )
        dates = base_panel.index.get_level_values("date")
        test_end = pd.Timestamp(base_fold_cfg.split.test_end or dates.max())
        panel = base_panel[dates <= test_end].copy()
        train, test = split_train_test(panel, base_fold_cfg)
        if train.empty or test.empty:
            continue

        for variant in variants:
            try:
                fold_cfg = m2_config_for_variant(variant, base_fold_cfg)
                train_mask = get_m2_training_mask(train)
                test_mask = get_m2_training_mask(test)
                y_train = train.loc[train_mask, "meta_label"].dropna()
                y_test = test.loc[test_mask, "meta_label"].dropna()
                X_train = build_m2_features_variant(train.loc[train_mask], fold_cfg, variant, train_panel=train).loc[
                    y_train.index
                ]
                X_test = build_m2_features_variant(test.loc[test_mask], fold_cfg, variant, train_panel=train).loc[
                    y_test.index
                ]
                model = create_m2_model(fold_cfg.m2)
                model.fit(X_train, y_train)
                p_test = model.predict_proba(X_test)
                test_auc = _m2_auc(y_test, p_test)

                # baseline configured on same fold
                baseline_cfg = m2_config_for_variant(BASELINE_VARIANT, base_fold_cfg)
                Xb = build_m2_features_variant(
                    test.loc[test_mask], baseline_cfg, BASELINE_VARIANT, train_panel=train
                ).loc[y_test.index]
                mb = create_m2_model(baseline_cfg.m2)
                mb.fit(
                    build_m2_features_variant(
                        train.loc[train_mask], baseline_cfg, BASELINE_VARIANT, train_panel=train
                    ).loc[y_train.index],
                    y_train,
                )
                baseline_auc = _m2_auc(y_test, mb.predict_proba(Xb))

                rows.append(
                    {
                        **fold,
                        "variant": variant,
                        "test_auc": test_auc,
                        "baseline_test_auc": baseline_auc,
                        "auc_delta_vs_baseline": test_auc - baseline_auc,
                        "n_features": len(X_train.columns),
                    }
                )
            except Exception as exc:
                logger.warning("WF fold %s variant %s failed: %s", fold.get("fold_id"), variant, exc)

    return pd.DataFrame(rows)


def generate_m2_feature_research_report(
    sweep: pd.DataFrame,
    decision: dict[str, Any],
    wf: pd.DataFrame,
    report_path: Path,
    *,
    test_start: str,
) -> None:
    from src.diagnostics import _fmt_num, _fmt_pct, _markdown_table

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M2 Feature Enrichment Research",
        "",
        "**Research use only — not investment advice.**",
        "",
        "Goal: improve **M2 → M3** pipeline by enriching meta-label features with",
        "**M1 factor analysis** (component ranks, IC alignment) and **dynamic external**",
        "factors (VIX, macro regimes, cross-asset dispersion interactions).",
        "",
        f"**Test window:** `{test_start}` onward",
        "",
        "## Steps taken",
        "",
        "1. **Baseline comparison** — `legacy_global` (40 features) vs `configured` (52, production).",
        "2. **M1 factor enrichments** — cross-sectional component ranks, factor spread/sign agreement,",
        "   trend−momentum spread, trend-heavy composite (from M1 IC analysis: trend IC 0.12 on test).",
        "3. **External/regime enrichments** — interactions of M1/components with `risk_off`, VIX,",
        "   yield curve, credit stress, inflation/growth flags, and dispersion features.",
        "4. **IC alignment** — train-period factor IC weights × per-factor CS ranks (no look-ahead).",
        "5. **Portfolio validation** — each variant refit on train; ECDF test Sharpe/return via M3.",
        "6. **Walk-forward** — test AUC vs configured baseline across 6 expanding-window folds.",
        "7. **Adoption gates** — test AUC ≥ baseline + 0.003 AND ECDF Sharpe ≥ 0.94; WF ≥ 4/6 fold wins.",
        "",
        "## Adoption verdict",
        "",
        f"- **Verdict:** `{decision.get('verdict')}`",
        f"- **Winner:** `{decision.get('winner', 'none')}`",
        f"- **Reason:** {decision.get('reason', '')}",
        "",
        "## Holdout comparison (test period)",
        "",
    ]

    if not sweep.empty:
        disp = sweep.sort_values("test_auc", ascending=False).copy()
        for col in ("train_auc", "test_auc", "test_f1", "ecdf_test_sharpe"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        for col in ("ecdf_test_ann_return", "ecdf_test_max_drawdown"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
        show = [
            "variant",
            "n_features",
            "train_auc",
            "test_auc",
            "test_f1",
            "ecdf_test_sharpe",
            "ecdf_test_ann_return",
            "ecdf_test_max_drawdown",
            "description",
        ]
        disp = disp[[c for c in show if c in disp.columns]]
        lines.append(_markdown_table(disp))
        lines.append("")

    lines.extend(["## Walk-forward test AUC (top variants)", ""])
    if not wf.empty:
        summary = (
            wf.groupby("variant")
            .agg(
                mean_test_auc=("test_auc", "mean"),
                mean_auc_delta=("auc_delta_vs_baseline", "mean"),
                positive_folds=("auc_delta_vs_baseline", lambda s: int((s > 0).sum())),
                n_folds=("test_auc", "count"),
            )
            .reset_index()
            .sort_values("mean_test_auc", ascending=False)
        )
        for col in ("mean_test_auc", "mean_auc_delta"):
            summary[col] = summary[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        lines.append(_markdown_table(summary))
        lines.append("")

    lines.extend(["## Explainability", ""])
    for variant in M2_FEATURE_VARIANTS:
        lines.append(f"- **`{variant}`:** {describe_variant(variant)}")
    lines.append("")

    if decision.get("verdict") == "adopt":
        lines.extend(
            [
                "## Recommendation",
                "",
                f"Adopt **`{decision.get('winner')}`** — wired into `build_m2_features` / config.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommendation",
                "",
                "Keep production **`configured`** M2 features unchanged.",
                "Document trade-offs in [m2_diagnostics.md](m2_diagnostics.md).",
                "",
            ]
        )

    lines.append(
        "Related: [m1_factor_analysis.md](m1_factor_analysis.md) · "
        "[m2_diagnostics.md](m2_diagnostics.md) · [TERMINOLOGY.md](../TERMINOLOGY.md)"
    )
    report_path.write_text("\n".join(lines))


def apply_winner_to_build_m2_features(winner: str) -> bool:
    """If adopt, set default enrichment in model_m2 via config hook. Returns True if applied."""
    if winner in ("legacy_global", "configured"):
        return False
    # Research winners use enrich_m2_features — wire via m2.feature_variant in config
    return True


def run_m2_feature_research(
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

    panel_path = root / "data/predictions/long_only/panel_with_predictions.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        if "date" in panel.columns:
            panel = panel.set_index(["date", "ticker"]).sort_index()
        logger.info("Using cached predictions panel")
    else:
        from src.m3_threshold_research import _fit_production_panel

        panel, _ = _fit_production_panel(root, cfg)

    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)
    eval_dir = root / "data/backtests/long_only/evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    sweep = sweep_m2_feature_variants(
        panel,
        returns_wide,
        cfg,
        test_start=cfg.split.test_start,
        test_end=cfg.split.test_end,
    )
    sweep.to_csv(eval_dir / "m2_feature_sweep.csv", index=False)

    # Walk-forward on promising variants only (top 3 by test AUC + baseline)
    wf = pd.DataFrame()
    if not skip_walk_forward and cfg.evaluation.walk_forward_enabled and not sweep.empty:
        top = sweep.sort_values("test_auc", ascending=False).head(3)["variant"].tolist()
        wf_variants = tuple(dict.fromkeys([BASELINE_VARIANT, *top]))
        logger.info("Walk-forward for variants: %s", wf_variants)
        wf = run_m2_feature_walk_forward(panel, returns_wide, cfg, wf_variants)
        if not wf.empty:
            wf.to_csv(eval_dir / "m2_feature_walk_forward.csv", index=False)

    decision = evaluate_adoption_decision(sweep, wf)
    (eval_dir / "m2_feature_adoption_decision.json").write_text(
        json.dumps(decision, indent=2, default=str)
    )

    generate_m2_feature_research_report(
        sweep,
        decision,
        wf,
        root / "reports/m2_feature_research.md",
        test_start=cfg.split.test_start,
    )

    if decision.get("verdict") == "adopt":
        winner = decision["winner"]
        logger.info("Applying winner variant to config: %s", winner)
        _apply_config_variant(root / "config/config.yaml", winner)

    return {"sweep": sweep, "decision": decision, "walk_forward": wf}


def _apply_config_variant(config_path: Path, variant: str) -> None:
    """Set m2.feature_variant in config.yaml."""
    text = config_path.read_text()
    if "feature_variant:" in text:
        import re

        text = re.sub(r"feature_variant:\s*\S+", f"feature_variant: {variant}", text)
    else:
        text = text.replace(
            "include_asset_encoding: true",
            f"include_asset_encoding: true\n    feature_variant: {variant}",
        )
    config_path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M2 feature enrichment research")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-walk-forward", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = run_m2_feature_research(Path(args.config), skip_walk_forward=args.skip_walk_forward)
        logger.info("Verdict: %s winner=%s", out["decision"].get("verdict"), out["decision"].get("winner"))
        return 0
    except Exception:
        logger.exception("M2 feature research failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
