# Branch Update — Executive Summary

**Branch:** `vitaly_week5` vs `main`  
**Date:** July 2026  
**Audience:** Project managers, team leads, stakeholders  

**Research use only — not investment advice.**

---

## Purpose

This branch makes the multi-asset meta-labeling pipeline **defensible in review** and aligns terminology with **Joubert (2022)**: M1 (side), M2 (trade quality probability), M3 (bet sizing), then portfolio risk controls. It adds **diagnostics, two measurable model improvements, and extended evaluation** — without changing the M1 rule engine or train/test split.

**Full technical detail:** [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md)

---

## What Changed

Five workstreams landed across the branch (~160 files, +7,400 lines — mostly diagnostics, reports, and tests):

| # | Workstream | Headline outcome |
| --- | --- | --- |
| 1 | **Analytics layer** | 6 companion reports; factor IC, M2 calibration, regime conditioning, M3 allocation states |
| 2 | **M3 formalization** | Persisted `M3_size`; strategy rename to `m1_m2_m3_*`; allocation states on panel |
| 3 | **M1 weight tuning** | IC-driven weights improve test Sharpe **0.787 → 0.795** (+0.008) — research only, not in config |
| 4 | **M2 ranking** | Test AUC **0.573 → 0.589** (+0.016) via enriched meta-features (52 vs 40 inputs) |
| 5 | **Extended evaluation** | Transaction-cost sensitivity; walk-forward module (configurable) |

**Unchanged vs `main`:** M1 top-K selection, train/test dates, portfolio caps, 12% vol target, data sources.

---

## Headline Results (Long-Only, Test Period 2021+)

| Strategy | Ann. Return | Sharpe | Max Drawdown | vs `main` Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Equal Weight (1/7) | 7.34% | 0.69 | -23.9% | same |
| **M1 Only** | **8.40%** | **0.79** | -21.0% | same |
| M1 + M2 + M3 (Binary) | 8.40% | 0.79 | -21.0% | same |
| M1 + M2 + M3 (Linear) | 1.87% | 0.86 | -4.4% | +0.002 |
| **M1 + M2 + M3 (ECDF)** | **7.02%** | **0.96** | **-11.3%** | **+0.113** |

**Stakeholder narrative:**

- **M1** selects ~3 ETFs per week (~43% of asset-weeks are long candidates); economics match `main`.
- **M2** test AUC improved from **~0.57 to 0.59**; still weak ranking — value is in **M3 ECDF sizing**, not binary filter at 0.55.
- **M3 binary** at T=0.55 approves ~99% of candidates (recall ≈ 1) — equals M1-only by design.
- **M3 ECDF** vs `main`: Sharpe **0.85 → 0.96**, drawdown **-16.3% → -11.3%** on the 2021+ test window.
- **Transaction costs:** ECDF Sharpe edge vs M1 remains **+0.046** even at **25 bps** turnover.

Source: [reports/final_report.md](reports/final_report.md) · [reports/evaluation_analysis.md](reports/evaluation_analysis.md)

---

## Architecture: Before vs After

| | `main` | `vitaly_week5` |
| --- | --- | --- |
| Stack | M1 → M2 → unnamed sizing → portfolio | M1 → M2 → **M3** → portfolio |
| M2 output | `p_success` | `p_success` (unchanged role) |
| Sizing output | Ephemeral in backtest | Persisted `M3_size` on panel |
| Strategy names | `m1_m2_binary`, `m1_m2_ecdf` | `m1_m2_m3_*` (legacy aliases kept) |
| Reports | `final_report.md` only | +5 companion reports + evaluation |

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

## New Artifacts

| Report | Content |
| --- | --- |
| [reports/m1_factor_analysis.md](reports/m1_factor_analysis.md) | Per-factor IC, correlation 0.77 mom/trend, weight tuning |
| [reports/m2_diagnostics.md](reports/m2_diagnostics.md) | AUC 0.589, calibration, architecture benchmark |
| [reports/market_regime_analysis.md](reports/market_regime_analysis.md) | Regime timeline, performance by macro flags |
| [reports/m3_allocation_analysis.md](reports/m3_allocation_analysis.md) | Allocation states, M3 rule comparison |
| [reports/evaluation_analysis.md](reports/evaluation_analysis.md) | Transaction-cost sensitivity (+0.177 edge @ 5 bps) |
| [reports/final_report.md](reports/final_report.md) | Deep Diagnostics links all companions |

**Data:** `data/backtests/long_only/` — factor IC, weight tuning, M2 benchmark, M3 allocation, evaluation CSVs, figures, `panel_with_predictions.parquet` with `M3_size`.

---

## Quality & Merge Readiness

| Check | Status |
| --- | --- |
| Unit + integration tests | **61/61 passing** (was 48 on `main`) |
| Backward-compatible strategy keys | Legacy `m1_m2_*` aliases preserved |
| Documentation | `ARCHITECTURE_BRIEFING.md`, `docs/MODELING_SPEC.md`, branch reports |
| M1-only economics | Unchanged vs `main` |
| ECDF sleeve | Improved vs `main` on test Sharpe and drawdown |

---

## Recommended Next Steps

**Merge**

1. Review and merge PR `vitaly_week5` → `main` with this document in the PR description.
2. Add `runs/` to `.gitignore`.

**Research (high value)**

3. **Apply IC-proportional M1 weights** after walk-forward confirmation (+0.008 test Sharpe in research).
4. **Run full walk-forward** (`evaluation.walk_forward_enabled: true`) — multi-window OOS validation.
5. **M3 threshold sweep** — T=0.55 is too permissive for binary sizing to add value.
6. **Regime-conditioned M3** — ECDF Sharpe **1.21** in risk-off vs **0.86** in risk-on (full sample).
7. **Short-side logic** — long/short test Sharpe **0.47** vs long-only **0.79**.

**Ruled out (tested)**

8. Per-asset M2 heads / tree models — test AUC ~0.48–0.50 (overfit).

Full roadmap: [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md)

---

## Related Documents

| Document | Role |
| --- | --- |
| [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md) | Full technical diff and file map |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Ongoing project narrative |
| [ARCHITECTURE_BRIEFING.md](ARCHITECTURE_BRIEFING.md) | Architecture for banking/systemic audiences |
| [reports/final_report.md](reports/final_report.md) | Latest pipeline metrics and charts |
