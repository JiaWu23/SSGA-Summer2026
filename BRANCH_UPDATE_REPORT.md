# Branch Update — Executive Summary

**Branch:** `vitaly_week5` vs `main`  
**Date:** July 2026  
**Audience:** Project managers, team leads, stakeholders  

**Research use only — not investment advice.**

---

## Purpose

This update makes the multi-asset meta-labeling pipeline **explainable to reviewers** and aligns internal terminology with **Joubert (2022)**: M1 (side), M2 (trade quality probability), M3 (bet sizing), then portfolio risk controls. Portfolio returns are **unchanged in substance**; the branch adds **diagnostics, naming clarity, and companion reports** so the team can defend design choices in meetings and code review.

**Full technical detail:** [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md)

---

## What Changed (One Paragraph)

Two workstreams landed in three commits (~156 files, +6,300 lines, mostly reports and diagnostics):

1. **Analytics layer** — M1 factor IC and ablation, M2 calibration/decile/AUC-PR charts, market/regime conditioning, four new companion reports wired into `final_report.md`.
2. **M3 formalization** — Explicit bet-sizing layer (`M3_size`), allocation states (`no_signal` / `m3_zero` / `m3_active`), persisted panel columns, strategy rename to `m1_m2_m3_*`, and M3 allocation diagnostics.

**Core model logic is unchanged:** same M1 top-K rule engine, M2 logistic regression, portfolio caps, 12% vol target, train/test split, and data sources.

---

## Headline Results (Long-Only, Test Period 2021+)

These numbers match `main` economics; interpretation is now clearer.

| Strategy | Ann. Return | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: |
| Equal Weight (1/7) | 7.34% | 0.69 | -23.9% |
| **M1 Only** | **8.40%** | **0.79** | **-21.0%** |
| M1 + M2 + M3 (Binary) | 8.34% | 0.78 | -21.0% |
| M1 + M2 + M3 (ECDF) | 6.19% | 0.83 | -15.0% |
| M1 + M2 + M3 (Linear) | 1.82% | 0.85 | -4.6% |

**Key narrative for stakeholders:**

- **M1** selects ~3 ETFs per week (~43% of asset-weeks are long candidates).
- **M2** assigns P(success) with ~59% base rate of profitable trades; AUC-ROC ≈ 0.57 (weak ranking).
- **M3 binary** at threshold 0.55 approves ~99% of candidates (recall ≈ 1), so it looks like M1-only — that is expected, not a bug.
- **M3 ECDF** improves Sharpe mainly by **lowering volatility and drawdown**, sometimes at the cost of mean return — consistent with Joubert Experiment 3.

When presenting OOS performance, **lead with the test-period table above**, not full-sample metrics.

Source: [reports/final_report.md](reports/final_report.md)

---

## Architecture: Before vs After

| | `main` | `vitaly_week5` |
| --- | --- | --- |
| Stack | M1 → M2 → unnamed sizing → portfolio | M1 → M2 → **M3** → portfolio |
| M2 output | `p_success` | `p_success` (unchanged) |
| Sizing output | Ephemeral `size` in backtest | Persisted `M3_size` on panel |
| Strategy names | `m1_m2_binary`, `m1_m2_linear`, `m1_m2_ecdf` | `m1_m2_m3_*` (legacy aliases kept) |

```mermaid
flowchart LR
  M1[M1 side signal]
  M2[M2 p_success]
  M3[M3 M3_size]
  PF[Portfolio caps and vol target]
  W[final_weight]

  M1 --> M2 --> M3 --> PF --> W
```

---

## New Artifacts (Open Today)

| Report | Content |
| --- | --- |
| [reports/m1_factor_analysis.md](reports/m1_factor_analysis.md) | Per-factor IC, correlation, sleeve backtests |
| [reports/m2_diagnostics.md](reports/m2_diagnostics.md) | Calibration, decile returns, AUC-ROC guide |
| [reports/market_regime_analysis.md](reports/market_regime_analysis.md) | Regime timeline, performance by macro flags |
| [reports/m3_allocation_analysis.md](reports/m3_allocation_analysis.md) | Allocation states, M3 rule comparison |
| [reports/final_report.md](reports/final_report.md) | Deep Diagnostics section links all companions |

**Data artifacts** under `data/backtests/long_only/`: `m1_factor_ic.csv`, `m2_calibration_table.csv`, `m3_allocation_summary.csv`, figures, and `panel_with_predictions.parquet` with `M3_size` and `allocation_state`.

---

## Quality & Merge Readiness

| Check | Status |
| --- | --- |
| Unit + integration tests | **51/51 passing** (was 48 on `main`) |
| Backward-compatible strategy keys | Legacy `m1_m2_*` aliases preserved |
| Documentation | `ARCHITECTURE_BRIEFING.md`, `docs/MODELING_SPEC.md` updated |
| No change to M1/M2 training logic | Confirmed |
| Minor follow-ups before/after merge | See detailed report § Known gaps |

---

## Recommended Next Steps (Prioritized)

**Merge hygiene**

1. Review and merge PR `vitaly_week5` → `main` with this document linked in the PR description.
2. Consider adding `runs/` to `.gitignore` (run snapshots currently in diff).

**Research (high value)**

3. **Improve M2 ranking** — AUC 0.57 is weak; richer features or per-asset heads.
4. **M3 threshold sweep** — T=0.55 is too permissive for binary sizing to add value.
5. **M1 factor ablation** — momentum/trend correlation 0.77; tune weights using new IC outputs.
6. **Regime-conditioned M3** — scale bets by `risk_off` / inflation flags.

**Evaluation**

7. Walk-forward validation across multiple test windows.
8. Transaction-cost sensitivity on ECDF Sharpe edge.

Full roadmap: [reports/branch_update_vitaly_week5.md#recommended-next-steps](reports/branch_update_vitaly_week5.md#recommended-next-steps)

---

## Related Documents

| Document | Role |
| --- | --- |
| [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md) | Full technical diff and file map |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Ongoing project narrative (updated for M3) |
| [ARCHITECTURE_BRIEFING.md](ARCHITECTURE_BRIEFING.md) | Architecture for banking/systematic audiences |
| [reports/final_report.md](reports/final_report.md) | Latest pipeline metrics and charts |
