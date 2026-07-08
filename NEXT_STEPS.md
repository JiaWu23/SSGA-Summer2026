# Next Steps and Review Notes

**Branch vs `main`:** [BRANCH_UPDATE_REPORT.md](BRANCH_UPDATE_REPORT.md) · [reports/branch_update_vitaly_week5.md](reports/branch_update_vitaly_week5.md)

## Walk-forward ECDF validation (completed)

Full strategy walk-forward ran with `walk_forward_enabled: true`. Report: [reports/walk_forward_analysis.md](reports/walk_forward_analysis.md).

| Question | Answer |
| --- | --- |
| Is ECDF edge stable across folds? | **Yes (majority):** +0.177 mean edge, **4/6** folds positive |
| Is 2021+ an outlier? | **No** — pre-2021 mean edge (+0.243) ≥ production-era (+0.112) |
| ECDF vs equal-weight | Wins **3/6** folds on Sharpe |
| TC @ 25 bps (production window) | Edge **+0.046** Sharpe vs M1-only |

**Weak folds:** 2015–2016 (edge −0.19), 2025–2026 partial (edge −0.12). **Strong folds:** 2017–2018 (+0.73), 2021–2022 (+0.42).

## M1 weights

Walk-forward **rejected** IC-proportional weights — config unchanged at **45/25/20/10**.

## IR vs equal-weight (completed)

Reports: [ir_attribution_analysis.md](reports/ir_attribution_analysis.md) · [ir_improvement_research.md](reports/ir_improvement_research.md)

| Finding | Detail |
| --- | --- |
| Why IR drops | ECDF deploys ~52% gross vs EW 100%; lower absolute return vs EW |
| Holdout best IR | `exposure_renorm_1.10` IR **0.35** but Sharpe **0.78** (fails gate) |
| Holdout gate pass | `vol_bump_0.55_1.15` IR **0.08**, Sharpe **0.96**, return **8.1%** |
| Walk-forward | Winner **rejected** — IR positive in only **2/6** folds |
| **Config** | **Unchanged** — keep ECDF baseline |

## Recommended next work

1. **Regime-conditioned M3** — edge varies by fold/regime; fold 1 risk-off may need sizing down.
2. **M3 threshold sweep** — binary T=0.55 still non-binding.
3. **Short-side research** — long/short test Sharpe 0.47 vs 0.79 long-only.

## CLI

```bash
python -m src.walk_forward_research      # strategy walk-forward + reports
python -m src.m1_weight_research       # M1 IC weight walk-forward
python -m src.ir_research              # IR attribution + intervention sweep
python -m src.m2_feature_research      # M2 feature enrichment sweep
```

## M2 feature enrichment (completed)

Report: [reports/m2_feature_research.md](reports/m2_feature_research.md)

| Variant | Test AUC | ECDF Sharpe | vs configured |
| --- | ---: | ---: | --- |
| **m1_components_rich** | **0.594** | **1.049** | +0.005 AUC, +0.085 Sharpe |
| ic_alignment | 0.594 | 1.048 | +0.005 AUC |
| configured (production) | 0.589 | 0.964 | baseline |
| full_enriched | 0.584 | 1.067 | overfits; WF fails |

**Walk-forward rejected** `m1_components_rich` (2/6 fold AUC wins). **Config unchanged** — keep 52-feature `configured` set.

Holdout insight: M1 factor CS ranks help M2 ranking and ECDF Sharpe on 2021+, but gains are not stable across regimes.
