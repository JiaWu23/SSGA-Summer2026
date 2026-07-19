#!/usr/bin/env python3
"""M1 factor-weight sweep (Axis A) — walk-forward, sample-fixed.

Motivation: the production weights 45/25/20/10 are an economic prior, never tuned; and
momentum_score vs trend_score are ~0.78 correlated (measured on the model panel), so the
45/25 split puts ~70% of M1's weight on one underlying trend-following signal. This sweep
tests whether that split matters and whether momentum-alone or trend-alone reproduces it --
a practical stand-in for "merge momentum + trend into a single technical factor".

Ranks on WALK-FORWARD mean M1-only Sharpe (not in-sample) so we do not overfit the weights.
The macro set is held at the config default for every variant, so only the M1 weights vary.

Usage:
    python scripts/m1_weight_sweep.py            # full sweep
    python scripts/m1_weight_sweep.py --dry-run
    python scripts/m1_weight_sweep.py --only baseline
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "config.yaml"
WF_CSV = ROOT / "data" / "backtests" / "long_only" / "evaluation" / "walk_forward_summary.csv"
SWEEP_DIR = ROOT / "runs" / "m1_weight_sweep"

# weights: (momentum, trend, macro, risk_penalty)
VARIANTS: dict[str, tuple] = {
    "baseline":          (0.45, 0.25, 0.20, 0.10),   # current prior
    "momentum_only":     (0.70, 0.00, 0.20, 0.10),   # drop trend -> momentum is the sole technical factor
    "trend_only":        (0.00, 0.70, 0.20, 0.10),   # drop momentum
    "technical_5050":    (0.35, 0.35, 0.20, 0.10),   # equal mom/trend
    "no_macro":          (0.56, 0.31, 0.00, 0.13),   # technical-only, prior non-macro weights rescaled
    "macro_heavy":       (0.35, 0.20, 0.35, 0.10),   # push macro to see if it can carry weight
    "risk_heavy":        (0.40, 0.22, 0.18, 0.20),   # more downside penalty
    "momentum_pure":     (0.85, 0.00, 0.00, 0.15),   # momentum + risk only (leanest technical)
}


def summarize_wf(df: pd.DataFrame) -> dict:
    def mean(c):
        return float(df[c].mean()) if c in df else float("nan")

    def pos(c):
        return float((df[c] > 0).mean()) if c in df else float("nan")

    return {
        "n_folds": len(df),
        "m1_only_wf_sharpe": mean("m1_only_sharpe"),
        "m1_wf_ann_return": mean("m1_only_ann_return"),
        "m1_folds_pos": pos("m1_only_sharpe"),
        "ecdf_edge_vs_m1_mean": mean("ecdf_sharpe_edge_vs_m1"),
        "ir_edge_vs_ew_mean": mean("ir_edge_vs_ew"),
        "m2_auc_mean": mean("m2_auc"),
        "ew_wf_sharpe": mean("equal_weight_sharpe"),
    }


def run_variant(name: str, w: tuple, python: str) -> dict | None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(yaml.safe_load(BASE_CONFIG.read_text()))
    cfg["models"]["m1"]["weights"] = {
        "momentum": w[0], "trend": w[1], "macro": w[2], "risk_penalty": w[3],
    }
    cfg_path = SWEEP_DIR / f"config_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"\n=== [{name}] weights mom/trend/macro/risk = {w} ===", flush=True)
    if WF_CSV.exists():
        WF_CSV.unlink()
    proc = subprocess.run(
        [python, "-m", "src.run_pipeline", "--config", str(cfg_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    (SWEEP_DIR / f"{name}.log").write_text(proc.stdout + "\n===STDERR===\n" + proc.stderr)
    if proc.returncode != 0 or not WF_CSV.exists():
        print(f"    FAILED (rc={proc.returncode})", flush=True)
        return None
    shutil.copy(WF_CSV, SWEEP_DIR / f"{name}_walk_forward.csv")
    s = summarize_wf(pd.read_csv(WF_CSV))
    s["variant"] = name
    s["weights"] = "/".join(str(x) for x in w)
    print(f"    M1 WF Sharpe={s['m1_only_wf_sharpe']:.3f} "
          f"(folds+ {s['m1_folds_pos']*100:.0f}%) | ann={s['m1_wf_ann_return']*100:.2f}%", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = [args.only] if args.only else list(VARIANTS)
    if args.dry_run:
        for nm in names:
            print(f"{nm:16s} {VARIANTS[nm]}")
        return 0
    python = sys.executable
    rows = []
    for nm in names:
        r = run_variant(nm, VARIANTS[nm], python)
        if r:
            rows.append(r)
            pd.DataFrame(rows).to_csv(SWEEP_DIR / "results.csv", index=False)
    if not rows:
        print("No successful runs.")
        return 1
    df = pd.DataFrame(rows).sort_values("m1_only_wf_sharpe", ascending=False)
    cols = ["variant", "m1_only_wf_sharpe", "m1_folds_pos", "m1_wf_ann_return",
            "ecdf_edge_vs_m1_mean", "m2_auc_mean", "weights"]
    df[cols].to_csv(SWEEP_DIR / "results_ranked.csv", index=False)
    (SWEEP_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print("\n===== RANKED (walk-forward M1-only Sharpe) =====")
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
