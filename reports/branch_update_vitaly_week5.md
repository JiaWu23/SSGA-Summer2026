# Branch Update — `vitaly_week5` vs `main` (Technical)

**Branch:** `vitaly_week5`  
**Baseline:** `main`  
**Date:** July 2026  
**Audience:** Engineers, quants, code reviewers  

**Research use only — not investment advice.**

Executive summary for PMs: [BRANCH_UPDATE_REPORT.md](../BRANCH_UPDATE_REPORT.md)

---

## Summary

This branch does **not** replace the core M1 top-K rule engine or portfolio construction. It adds **explainability**, **Joubert-aligned naming (M3)**, **research modules**, and **two measurable model improvements** (M2 ranking + ECDF sizing input). Portfolio economics for M1-only are **unchanged** vs `main`; the ECDF sleeve improves on `main` because enriched M2 features change `p_success` and therefore ECDF bet fractions.

| Area | `main` | `vitaly_week5` |
| --- | --- | --- |
| Stack naming | M1 → M2 → sizing → portfolio | M1 → M2 → **M3** → portfolio |
| Persisted sizing | Ephemeral in backtest | `M3_size`, `allocation_state` on panel |
| Strategy keys | `m1_m2_*` | `m1_m2_m3_*` (legacy aliases kept) |
| Companion reports | None | 6 markdown reports + Deep Diagnostics |
| Tests | 48 passing | **61 passing** (+13) |
| M2 feature count | 40 | **52** (+ M1 components, interactions, asset dummies) |
| M2 test AUC-ROC | **0.5727** (legacy LR) | **0.5890** (+0.016) |
| ECDF test Sharpe (2021+) | **0.8513** | **0.9641** (+0.113) |
| ECDF test max drawdown | -16.29% | **-11.33%** (+4.96 pp shallower) |

---

## Headline OOS Metrics (Long-Only, Test 2021+)

Same train/test split on both branches: train through **2020-12-31**, test from **2021-01-01**.

| Strategy | `main` Ann. Return | `main` Sharpe | Branch Ann. Return | Branch Sharpe | Sharpe Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal Weight (1/7) | 7.34% | 0.685 | 7.34% | 0.685 | 0.000 |
| **M1 Only** | **8.40%** | **0.787** | **8.40%** | **0.787** | 0.000 |
| M1 + M2 + M3 (Binary) | 8.40% | 0.787 | 8.40% | 0.787 | 0.000 |
| M1 + M2 + M3 (Linear) | 1.92% | 0.858 | 1.87% | 0.860 | +0.002 |
| **M1 + M2 + M3 (ECDF)** | **6.93%** | **0.851** | **7.02%** | **0.964** | **+0.113** |

**Interpretation:** M1-only is identical (same rule engine and weights). ECDF improves because M2 probabilities feed M3 sizing — richer M2 features shift the ECDF map without changing M1 selection. Binary M3 at T=0.55 still equals M1-only (recall ≈ 1.0).

Source: [final_report.md](final_report.md) (branch) vs `git show main:reports/final_report.md`.

---

## Workstream 1 — Analytics & Explainability

**New modules:** `src/factor_analysis.py`, `src/regime_analysis.py`, extended `src/diagnostics.py`, `src/m3_diagnostics.py`

| Deliverable | Key numbers |
| --- | --- |
| [m1_factor_analysis.md](m1_factor_analysis.md) | Test IC: trend **0.121**, momentum **0.073**, M1 composite **0.106**; momentum–trend correlation **0.773** |
| [m2_diagnostics.md](m2_diagnostics.md) | Test AUC-ROC **0.588**, AUC-PR **0.663**, base rate **58.9%**; top decile hit rate **76.5%** vs bottom **50.6%** |
| [market_regime_analysis.md](market_regime_analysis.md) | ECDF Sharpe **1.21** in `risk_off=on` vs **0.86** when off; inflation-up EW **-6.85%** ann. vs M1 **+0.18%** |
| [m3_allocation_analysis.md](m3_allocation_analysis.md) | **42.9%** of asset-weeks are M1 candidates with M3_size > 0; binary M3 zero-share **0.14%** |
| [evaluation_analysis.md](evaluation_analysis.md) | ECDF Sharpe edge vs M1 at 5 bps: **+0.177**; at 25 bps: **+0.046** (still positive) |

**`main` had:** single `final_report.md` with strategy tables only — no factor IC, calibration charts, regime conditioning, or M3 state breakdown.

---

## Workstream 2 — M3 Formalization (Joubert)

**New:** `src/model_m3.py`, `M3Config`, panel columns `M3_size` / `allocation_state`

| Concept | Branch behavior |
| --- | --- |
| Allocation states | `no_signal` (57.1%) · `m3_zero` · `m3_active` (42.9% of weeks) |
| M3 modes | binary / linear / ECDF — deterministic maps from `p_success` |
| Strategy rename | `m1_m2_m3_ecdf` replaces `m1_m2_ecdf`; aliases preserved in backtest |

Mean M3_size on candidates (full sample): binary **0.999**, linear **0.184**, ECDF **0.532**.

---

## Workstream 3 — M1 Factor Weight Tuning (Research)

**New functions in** `src/factor_analysis.py`: IC-proportional weights, grid search, recommendation CSV.

| Variant | Test Sharpe | vs Baseline |
| --- | ---: | ---: |
| Baseline (45/25/20/10) | 0.787 | — |
| **IC-proportional (train)** | **0.795** | **+0.008** |
| Trend-heavy (25/45/20/10) | 0.734 | -0.053 |
| Grid-best (train Sharpe) | 0.747 | -0.040 |

Recommended weights (not applied to config): momentum **49%**, trend **6%**, macro **15%**, risk penalty **30%**.

### Walk-forward validation (6 expanding-window folds)

| Metric | Baseline vs IC-proportional (per-fold) | Mean Δ |
| --- | --- | ---: |
| M1 test Sharpe | IC wins **2 / 6** folds | **-0.035** |
| ECDF test Sharpe | IC wins **3 / 6** folds | **-0.084** |
| **Config decision** | — | **Keep baseline 45/25/20/10** |

Single-holdout gain (+0.008 Sharpe on 2021+) does not survive multi-window OOS. Artifacts: `data/backtests/long_only/evaluation/m1_weight_walk_forward.csv`.

Ablation insight: zeroing trend hurts full-sample Sharpe (**0.822** with trend ablated vs **0.702** full M1); momentum and trend overlap via ρ=0.77.

---

## Workstream 4 — M2 Ranking Improvement

**Changes in** `src/model_m2.py`: M1 component features, cross-sectional rank, interactions (`m1_x_vol`, `m1_x_risk_off`), asset-class dummies.

| Architecture | Train AUC | Test AUC | Features |
| --- | ---: | ---: | ---: |
| Legacy global LR (`main`) | 0.646 | **0.573** | 40 |
| **Configured global LR (branch default)** | 0.646 | **0.589** | 52 |
| Per-asset heads (research) | — | ~0.48–0.50 | overfit |
| Gradient boosting (research) | — | ~0.48–0.50 | overfit |

**Net:** +**0.016** test AUC (+2.8% relative) with no change to classifier family. Value remains in **M3 ECDF sizing**, not binary threshold at 0.55.

---

## Workstream 5 — Extended Evaluation

**New:** `src/evaluation.py`, `EvaluationConfig`, [evaluation_analysis.md](evaluation_analysis.md)

### Transaction-cost sensitivity (production test window)

| Cost (bps) | M1-only Sharpe | ECDF Sharpe | ECDF edge vs M1 |
| --- | ---: | ---: | ---: |
| 0 | 0.814 | 1.024 | +0.210 |
| 5 (default) | 0.787 | 0.964 | +0.177 |
| 10 | 0.760 | 0.904 | +0.144 |
| 25 | 0.679 | 0.726 | **+0.046** |

### Walk-forward validation

Expanding-window folds completed (**6 folds**, 2-year test blocks). See [walk_forward_analysis.md](walk_forward_analysis.md).

| Metric | Value |
| --- | ---: |
| Mean ECDF Sharpe edge vs M1 | **+0.177** |
| Folds with positive edge | **4 / 6** (67%) |
| Pre-2021 mean edge | +0.243 |
| 2021+ mean edge | +0.112 |
| Stable (majority criterion)? | **Yes** |

Single-holdout 2021+ edge (+0.177 @ 5 bps TC) is **consistent** with multi-window mean; edge is **not** a 2021-only artifact.

---

## File Map (High Signal)

| Path | Role |
| --- | --- |
| `src/model_m3.py` | M3 sizing rules |
| `src/m3_diagnostics.py` | Allocation summaries |
| `src/factor_analysis.py` | IC, ablation, weight tuning, walk-forward weight validation |
| `src/m1_weight_research.py` | CLI: `python -m src.m1_weight_research` |
| `src/regime_analysis.py` | Regime timeline & conditioning |
| `src/model_m2.py` | Enriched M2 features |
| `src/evaluation.py` | Walk-forward + TC sensitivity |
| `src/diagnostics.py` | Report generation orchestration |
| `config/config.yaml` | `models.m3`, `evaluation`, `models.m2` meta features |

---

## Quality & Merge Readiness

| Check | Status |
| --- | --- |
| Tests | **61/61** passing (unit + integration) |
| Backward-compatible keys | `m1_m2_ecdf` → `m1_m2_m3_ecdf` alias |
| Docs | `ARCHITECTURE_BRIEFING.md`, `docs/MODELING_SPEC.md`, this report |
| Portfolio returns (M1-only) | Unchanged vs `main` |
| ECDF sleeve | Improved vs `main` (Sharpe +0.113 test) |

---

## Recommended Next Steps

### Merge & hygiene
1. Merge PR with [BRANCH_UPDATE_REPORT.md](../BRANCH_UPDATE_REPORT.md) linked.

### High-value research (prioritized)
2. **Regime-conditioned M3** — scale ECDF bets by `risk_off` / `inflation_up`.
3. **M3 threshold sweep** — T=0.55 → recall ≈ 1.0.
4. **Short-side M1** — long/short test Sharpe **0.474** vs long-only **0.787**.

### Completed / ruled out
- **IC-proportional M1 weights:** walk-forward **rejected** (mean M1 Δ **-0.035**, ECDF Δ **-0.084**). Config unchanged.
- Per-asset M2 heads / tree models — overfit (~0.48–0.50 test AUC).
- Trend-heavy M1 weights — test Sharpe **0.734** (worse than baseline).

### Data & production
9. Institutional point-in-time data before external claims.
10. Capacity, borrow, and live execution constraints not modeled.

---

## Related Documents

| Document | Role |
| --- | --- |
| [BRANCH_UPDATE_REPORT.md](../BRANCH_UPDATE_REPORT.md) | PM-facing executive summary |
| [final_report.md](final_report.md) | Latest pipeline metrics |
| [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) | Ongoing project narrative |
| [ARCHITECTURE_BRIEFING.md](../ARCHITECTURE_BRIEFING.md) | Architecture for systematic audiences |
