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

## Recommended next work

1. **Regime-conditioned M3** — edge varies by fold/regime; fold 1 risk-off may need sizing down.
2. **M3 threshold sweep** — binary T=0.55 still non-binding.
3. **Short-side research** — long/short test Sharpe 0.47 vs 0.79 long-only.

## CLI

```bash
python -m src.walk_forward_research      # strategy walk-forward + reports
python -m src.m1_weight_research       # M1 IC weight walk-forward
```
