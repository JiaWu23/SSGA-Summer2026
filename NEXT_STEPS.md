# Next Steps and Review Notes

This branch is in a strong reviewer-facing state: Joubert M1/M2/M3 terminology, six companion reports, factor/regime/M3 diagnostics, M2 ranking improvement, and extended evaluation (transaction-cost sensitivity + walk-forward module).

**Branch vs `main`:** [BRANCH_UPDATE_REPORT.md](BRANCH_UPDATE_REPORT.md) · [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md)

## Current Assessment

The strongest interpretation is a **long-only weekly top-K allocator**:

- **M1** ranks seven ETFs and selects top-3 each week (test Sharpe **0.787**, unchanged vs `main`).
- **M2** assigns P(success) with test AUC **0.589** (up from **~0.573** on `main`).
- **M3 ECDF** converts probabilities to bet fractions — test Sharpe **0.964** vs **0.851** on `main` ECDF (+**0.113**).
- **Long-only** is the main sleeve; long/short test Sharpe **0.474** vs **0.787** long-only.

## What Is Strong (Branch Additions vs `main`)

| Area | Outcome |
| --- | --- |
| Explainability | 6 companion reports + Deep Diagnostics in `final_report.md` |
| M3 formalization | `M3_size`, allocation states, `m1_m2_m3_*` naming |
| M2 ranking | +0.016 test AUC with enriched meta-features (52 vs 40 inputs) |
| M1 weight research | IC-proportional weights → +0.008 test Sharpe (not in config) |
| Transaction costs | ECDF edge **+0.177** @ 5 bps, **+0.046** @ 25 bps vs M1-only |
| Tests | **61/61** passing (was 48) |

## Main Remaining Risks

1. **Walk-forward not yet run on production config** — module exists (`evaluation.walk_forward_enabled`); last pipeline run disabled it for speed. Need full run (~20 min) for multi-window OOS table.

2. **M2 AUC still modest (~0.59)** — per-asset heads and tree models overfit (test AUC ~0.48–0.50); value remains in M3 ECDF sizing, not binary filter at T=0.55.

3. **M1 weight tuning not applied** — research shows +0.008 Sharpe; needs walk-forward confirmation before config change.

4. **Benchmark context** — equal-weight at 0 bps costs; strategies at 5 bps; M1 ~81% gross exposure.

5. **Research-grade data** — yfinance/FRED, not institutional point-in-time feeds.

## Recommended Next Work (Prioritized)

### 1. Run Full Walk-Forward Evaluation

Enable in `config/config.yaml`:

```yaml
evaluation:
  walk_forward_enabled: true
```

Run `python -m src.run_pipeline` (~20 min). Success criteria: mean ECDF Sharpe edge vs M1-positive across folds.

### 2. Apply IC-Proportional M1 Weights (After Walk-Forward)

Research weights (momentum 49%, trend 6%, macro 15%, risk 30%) improved test Sharpe **0.787 → 0.795**. Do not merge to config until walk-forward confirms stability.

### 3. M3 Threshold Sweep

T=0.55 → recall ≈ 1.0; binary M3 equals M1-only. Sweep T ∈ [0.55, 0.70] for meaningful rejection vs Sharpe tradeoff.

### 4. Regime-Conditioned M3

Full-sample ECDF Sharpe **1.21** in `risk_off=on` vs **0.86** when off. Scale ECDF bets by regime flags.

### 5. Short-Side Research (Separate Track)

Long/short M1 test Sharpe **0.474** — do not force symmetry with long-side logic.

### 6. Merge Hygiene

- Merge PR with branch reports linked

### Ruled Out (Tested on Branch)

- Per-asset M2 heads — test AUC ~0.48–0.50
- Gradient boosting / random forest M2 — overfit
- Trend-heavy M1 weights — test Sharpe 0.734 (worse than baseline)

## Completed Since `main`

- [x] M1 factor IC, ablation, weight tuning research
- [x] M2 deep diagnostics, architecture benchmark, enriched features
- [x] Market/regime analysis
- [x] M3 formalization and allocation diagnostics
- [x] Transaction-cost sensitivity
- [x] Walk-forward module (implementation; full run pending)

See [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md) for the full roadmap.
