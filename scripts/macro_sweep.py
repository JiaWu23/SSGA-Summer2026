#!/usr/bin/env python3
"""Macro-feature sweep for the meta-labeling pipeline.

Runs the full pipeline (with walk-forward) once per macro-feature configuration,
varying ONLY `macro.fred_series` and `features.macro_lag_weeks`. Everything else
(M1 weights, top-k, M3 rule, portfolio constraints) is held at the config baseline
so any Sharpe difference is attributable to the macro signal set, not other knobs.

Selection metric is deliberately WALK-FORWARD mean Sharpe + fold consistency, NOT
in-sample Sharpe -- picking the best in-sample macro set across many trials is the
overfitting trap. We report per-fold detail so a reviewer can see stability.

Usage:
    python scripts/macro_sweep.py                 # full sweep
    python scripts/macro_sweep.py --only baseline # single variant (smoke)
    python scripts/macro_sweep.py --dry-run       # list variants, no runs
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "config.yaml"
WF_CSV = ROOT / "data" / "backtests" / "long_only" / "evaluation" / "walk_forward_summary.csv"
SWEEP_DIR = ROOT / "runs" / "macro_sweep"

# ---- Macro variants -------------------------------------------------------
# Existing derivable series and what each feeds:
#   CPIAUCSL -> inflation_up | INDPRO -> growth_trend | T10Y2Y -> curve_inverted/yield_curve
#   BAA10Y -> credit_stress  | FEDFUNDS -> policy_rate_change | UNRATE -> unemployment_change
#   DGS10 -> per-asset carry  (VIX -> risk_off is always on, independent of this list)
ALL = ["CPIAUCSL", "UNRATE", "INDPRO", "FEDFUNDS", "DGS10", "T10Y2Y", "BAA10Y"]

# v2 (controlled): fred_series is held at ALL 7 for EVERY variant so the download, panel and
# sample window are byte-identical. Only `model_series` (which macro signals actually feed the
# model) varies -- excluded series are zeroed, not dropped. Any Sharpe difference is therefore
# attributable to the macro signal, not to a shifted sample. VIX risk_off is always present.
VARIANTS: dict[str, dict] = {
    "baseline":            {"model": ALL, "lag": 4},
    "macro_off":           {"model": [], "lag": 4},          # ablation: does macro help at all?
    "growth_inflation":    {"model": ["INDPRO", "CPIAUCSL"], "lag": 4},   # Bridgewater 2x2 core
    "rates_credit":        {"model": ["T10Y2Y", "BAA10Y", "DGS10", "FEDFUNDS"], "lag": 4},
    "recession_signals":   {"model": ["T10Y2Y", "UNRATE", "BAA10Y"], "lag": 4},
    "parsimonious":        {"model": ["INDPRO", "T10Y2Y", "BAA10Y"], "lag": 4},  # 1 growth/1 rates/1 credit
    "drop_lagging_cpi":    {"model": [s for s in ALL if s != "CPIAUCSL"], "lag": 4},
    "baseline_lag2":       {"model": ALL, "lag": 2},
    "baseline_lag8":       {"model": ALL, "lag": 8},
}


def summarize_wf(df: pd.DataFrame) -> dict:
    """Collapse a walk_forward_summary.csv into headline + stability stats."""
    def mean(col: str) -> float:
        return float(df[col].mean()) if col in df else float("nan")

    def pos_frac(col: str) -> float:
        return float((df[col] > 0).mean()) if col in df else float("nan")

    n = len(df)
    return {
        "n_folds": n,
        "m1_only_wf_sharpe": mean("m1_only_sharpe"),
        "ecdf_wf_sharpe": mean("ecdf_sharpe"),
        "ecdf_edge_vs_m1_mean": mean("ecdf_sharpe_edge_vs_m1"),
        "ecdf_edge_folds_pos": pos_frac("ecdf_sharpe_edge_vs_m1"),
        "m1_ir_vs_ew_mean": mean("m1_ir"),
        "ir_edge_vs_ew_mean": mean("ir_edge_vs_ew"),
        "ir_edge_folds_pos": pos_frac("ir_edge_vs_ew"),
        "m2_auc_mean": mean("m2_auc"),
        "m2_auc_folds_above_half": pos_frac("m2_auc") if "m2_auc" not in df
        else float((df["m2_auc"] > 0.5).mean()),
        "ew_wf_sharpe": mean("equal_weight_sharpe"),
    }


def run_variant(name: str, spec: dict, python: str, m1_macro_free: bool = False) -> dict | None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE_CONFIG.read_text())
    cfg = copy.deepcopy(base)
    cfg["macro"]["fred_series"] = ALL                 # fixed data/sample for every variant
    cfg["macro"]["model_series"] = spec["model"]      # only the model's macro inputs vary
    cfg["features"]["macro_lag_weeks"] = spec["lag"]
    if m1_macro_free:
        # v3: M1 is a PURE technical side model (macro weight -> 0, proportional rescale of the
        # original 0.45/0.25/0.10 non-macro weights). Macro then lives ONLY in M2 -> textbook
        # meta-labeling separation. M1-only is now identical across variants (fixed control);
        # any ECDF/M2-AUC difference is the value macro adds to the META-LABEL, not to M1.
        cfg["models"]["m1"]["weights"] = {
            "momentum": 0.5625, "trend": 0.3125, "macro": 0.0, "risk_penalty": 0.125,
        }
    cfg_path = SWEEP_DIR / f"config_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"\n=== [{name}] model_series={spec['model'] or 'NONE'} lag={spec['lag']} ===", flush=True)
    if WF_CSV.exists():
        WF_CSV.unlink()  # ensure we read THIS run's result, not a stale one

    proc = subprocess.run(
        [python, "-m", "src.run_pipeline", "--config", str(cfg_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    log_path = SWEEP_DIR / f"{name}.log"
    log_path.write_text(proc.stdout + "\n===STDERR===\n" + proc.stderr)
    if proc.returncode != 0 or not WF_CSV.exists():
        print(f"    FAILED (rc={proc.returncode}); see {log_path}", flush=True)
        return None

    shutil.copy(WF_CSV, SWEEP_DIR / f"{name}_walk_forward.csv")
    stats = summarize_wf(pd.read_csv(WF_CSV))
    stats["variant"] = name
    stats["model_series"] = ",".join(spec["model"]) or "(none)"
    stats["macro_lag_weeks"] = spec["lag"]
    print(f"    m1_only WF Sharpe={stats['m1_only_wf_sharpe']:.3f} | "
          f"ecdf edge={stats['ecdf_edge_vs_m1_mean']:+.3f} "
          f"({stats['ecdf_edge_folds_pos']*100:.0f}% folds+) | "
          f"IR edge={stats['ir_edge_vs_ew_mean']:+.3f}", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single named variant")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--m1-macro-free", action="store_true",
                    help="v3: zero M1's macro weight so macro feeds ONLY M2 (clean meta-labeling)")
    args = ap.parse_args()

    names = [args.only] if args.only else list(VARIANTS)
    if args.dry_run:
        for nm in names:
            print(f"{nm:20s} {VARIANTS[nm]}")
        return 0

    python = sys.executable
    rows = []
    for nm in names:
        res = run_variant(nm, VARIANTS[nm], python, m1_macro_free=args.m1_macro_free)
        if res:
            rows.append(res)
            pd.DataFrame(rows).to_csv(SWEEP_DIR / "results.csv", index=False)  # checkpoint each run

    if not rows:
        print("No successful runs.")
        return 1

    df = pd.DataFrame(rows).sort_values("m1_only_wf_sharpe", ascending=False)
    cols = ["variant", "m1_only_wf_sharpe", "ecdf_wf_sharpe", "ecdf_edge_vs_m1_mean",
            "ecdf_edge_folds_pos", "ir_edge_vs_ew_mean", "m2_auc_mean", "model_series"]
    df[cols].to_csv(SWEEP_DIR / "results_ranked.csv", index=False)
    (SWEEP_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print("\n===== RANKED (by walk-forward M1-only Sharpe) =====")
    print(df[cols].to_string(index=False))
    print(f"\nOutputs in {SWEEP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
