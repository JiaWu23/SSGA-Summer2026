# IR Improvement Research

**Research use only — not investment advice.**

Structured sweep of ECDF overlays to raise **Information Ratio vs equal-weight**
while preserving test Sharpe and annualized return.

**Test window:** `2021-01-01` onward

## Adoption verdict

- **Verdict:** `reject`
- **Winner:** `None`
- **Reason:** Winner `vol_bump_0.55_1.15` failed walk-forward IR stability (2/6 positive folds)

## Adoption gates (test period)

| Gate | Threshold |
| --- | --- |
| Sharpe | ≥ 0.95 |
| Ann return | ≥ 7.5% |
| Info Ratio | > 0 and > ECDF baseline |
| Max drawdown | Not >2pp worse than ECDF |
| Turnover | ≤ +30% vs ECDF |

## Test-period sweep (sorted by IR)

| variant | annualized_return | sharpe | information_ratio | excess_return_vs_benchmark | max_drawdown | mean_gross_exposure | annualized_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exposure_renorm_1.10 | 9.2043% | 0.7814 | 0.3523 | 1.8649% | -20.6362% | 90.1693% | 11.689844504675952 |
| exposure_renorm_1.00 | 8.3964% | 0.7841 | 0.2008 | 1.0570% | -18.9074% | 81.9721% | 10.627131367887229 |
| vol_bump_0.55_1.15 | 8.0659% | 0.9631 | 0.0795 | 0.7265% | -12.9517% | 60.1834% | 9.42577567414971 |
| vol_bump_0.60_1.15 | 8.0659% | 0.9631 | 0.0795 | 0.7265% | -12.9517% | 60.1834% | 9.42577567414971 |
| vol_bump_0.55_1.10 | 7.7179% | 0.9635 | 0.0175 | 0.3786% | -12.4140% | 57.5667% | 9.01595934049103 |
| m3_floor_0.7 | 7.3597% | 0.8421 | -0.0356 | 0.0204% | -16.6141% | 66.3025% | 5.678398590529322 |
| exposure_renorm_0.85 | 7.1724% | 0.7880 | -0.0655 | -0.1669% | -16.2603% | 69.6763% | 9.033061662704144 |
| regime_m3 | 7.2143% | 0.9261 | -0.0669 | -0.1250% | -13.5184% | 56.6716% | 9.081093675789473 |
| m3_floor_0.6 | 7.1545% | 0.8718 | -0.0813 | -0.1849% | -14.9151% | 62.1957% | 6.146161746221188 |
| m3_floor_0.4 | 7.0516% | 0.9337 | -0.1013 | -0.2877% | -12.5139% | 56.3891% | 7.1325914585058925 |
| ew_blend_0.9 | 7.0697% | 0.9445 | -0.1018 | -0.2697% | -12.6522% | 57.1000% | 7.376694005856296 |
| ew_blend_0.8 | 7.1146% | 0.9207 | -0.1018 | -0.2247% | -13.9583% | 61.8667% | 6.557061338538929 |
| ecdf_baseline | 7.0210% | 0.9641 | -0.1018 | -0.3184% | -11.3317% | 52.3333% | 8.196326673173662 |
| ew_blend_0.7 | 7.1558% | 0.8936 | -0.1018 | -0.1835% | -15.2501% | 66.6333% | 5.737428671221563 |
| m3_floor_0.5 | 7.0300% | 0.9010 | -0.1059 | -0.3093% | -13.5231% | 58.8927% | 6.634984075260907 |

## Walk-forward IR stability (winner vs EW)

| fold_id | train_start | train_end | test_start | test_end | test_weeks | m1_ir | ecdf_ir | ir_edge_vs_ew | winner_ir | winner_ir_edge_vs_ew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2006-01-01 | 2014-12-31 | 2015-01-01 | 2016-12-31 | 105 | -0.3370 | -0.5614 | -0.5614 | -0.5836 | -0.5836 |
| 2 | 2006-01-01 | 2016-12-31 | 2017-01-01 | 2018-12-31 | 104 | 0.9530 | 0.4284 | 0.4284 | 0.6149 | 0.6149 |
| 3 | 2006-01-01 | 2018-12-31 | 2019-01-01 | 2020-12-31 | 104 | -0.6691 | -0.7326 | -0.7326 | -0.6683 | -0.6683 |
| 4 | 2006-01-01 | 2020-12-31 | 2021-01-01 | 2022-12-31 | 105 | 0.6357 | 0.8779 | 0.8779 | 0.9858 | 0.9858 |
| 5 | 2006-01-01 | 2022-12-31 | 2023-01-01 | 2024-12-31 | 104 | -0.3689 | -0.8313 | -0.8313 | -0.7013 | -0.7013 |
| 6 | 2006-01-01 | 2024-12-31 | 2025-01-01 | 2026-06-12 | 76 | 0.2117 | -1.1561 | -1.1561 | -0.6706 | -0.6706 |

## Recommendation

Keep production **ECDF** unchanged. IR vs EW is an explicit trade-off:
ECDF improves Sharpe/drawdown by deploying less capital.
Document in [TERMINOLOGY.md](../TERMINOLOGY.md) — do not adopt overlay without gate pass.

Related: [ir_attribution_analysis.md](ir_attribution_analysis.md) · [evaluation_analysis.md](evaluation_analysis.md)