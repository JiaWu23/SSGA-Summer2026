# Walk-Forward Analysis: ECDF Edge Stability

**Research use only — not investment advice.**

This report answers whether **M1+M2+M3 ECDF** improves risk-adjusted returns vs **M1-only** across **multiple out-of-sample windows**, not only the production test period (2021+).

## Method

- **Design:** expanding train window; first train end `2014-12-31`; **2-year** test blocks
- **Production window (config):** `2021-01-01` onward — compared but not the sole metric
- **Per fold:** refit M1, M2, M3; backtest long-only; score test-block Sharpe
- **Edge:** `ECDF Sharpe − M1-only Sharpe` on each fold's test window

## Executive verdict

**ECDF edge is not stable: mean -0.190, positive in only 1/6 folds.**

| Metric | Value |
| --- | --- |
| Folds completed | 6 |
| Stable (majority + positive mean edge)? | **No** |
| Mean ECDF Sharpe edge vs M1 | -0.1898 |
| Median edge | -0.3831 |
| Folds with positive edge | 1 / 6 (17%) |
| Mean ECDF / M1 / EW Sharpe | 0.5187 / 0.7085 / 0.7084 |
| ECDF beats equal-weight (folds) | 1 / 6 |
| Mean M2 AUC (test, across folds) | 0.4864 |

## Pre-2021 vs production window

| Segment | Folds | Mean ECDF edge vs M1 | Positive folds |
| --- | ---: | ---: | ---: |
| Pre-`2021-01-01` test blocks | 3 | -0.4057 | 0 |
| `2021-01-01`+ test blocks | 3 | 0.0261 | 1 |

## Key questions

### 1. Is ECDF Sharpe edge vs M1 stable across folds?

**No / mixed:** mean edge -0.1898, positive in only 1/6 folds.

### 2. Is the 2021+ production result representative?

**Mixed.** Compare the fold table below — some eras favor ECDF sizing, others favor M1-only levels.

### 3. Does ECDF add value beyond equal-weight?

ECDF Sharpe exceeds equal-weight in **1** of **6** folds (mean ECDF Sharpe 0.5187 vs EW 0.7084).

### 4. What is M2 doing across folds?

Mean test AUC **0.4864** — ranking remains modest; ECDF edge is driven by **vol/drawdown shaping** from `p_success`, not binary filtering.

## Fold-level results

| fold_id | train_start | train_end | test_start | test_end | test_weeks | m1_only_sharpe | m1_only_ann_return | ecdf_sharpe | ecdf_ann_return | ecdf_sharpe_edge_vs_m1 | ecdf_return_edge_vs_m1 | equal_weight_sharpe | m1_ir | ecdf_ir | ir_edge_vs_ew | m2_auc | m2_auc_pr | m2_n_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2006-01-01 | 2014-12-31 | 2015-01-01 | 2016-12-31 | 105 | -0.1285 | -1.0150% | -0.5124 | -2.3030% | -0.3839 | -1.2880% | 0.0996 | -0.39060885532756784 | -0.5257141908472133 | -0.5257141908472133 | 0.4636 | 0.47169860418726905 | 315 |
| 2 | 2006-01-01 | 2016-12-31 | 2017-01-01 | 2018-12-31 | 104 | 0.9473 | 7.9767% | 0.2507 | 1.1642% | -0.6966 | -6.8125% | 0.5140 | 0.9009464574584505 | -0.5754901022661866 | -0.5754901022661866 | 0.4114 | 0.5486849104711295 | 312 |
| 3 | 2006-01-01 | 2018-12-31 | 2019-01-01 | 2020-12-31 | 104 | 0.5695 | 6.5965% | 0.4329 | 4.0281% | -0.1366 | -2.5684% | 0.8547 | -0.7643713943700546 | -0.9837458961189556 | -0.9837458961189556 | 0.4982 | 0.666760410786915 | 312 |
| 4 | 2006-01-01 | 2020-12-31 | 2021-01-01 | 2022-12-31 | 105 | -0.0569 | -0.5679% | 0.8621 | 3.1047% | 0.9190 | 3.6726% | -0.2795 | 0.431591604309763 | 0.5526835711766697 | 0.5526835711766697 | 0.5804 | 0.6402936075840315 | 315 |
| 5 | 2006-01-01 | 2022-12-31 | 2023-01-01 | 2024-12-31 | 104 | 1.0932 | 9.7612% | 0.6346 | 3.0389% | -0.4586 | -6.7223% | 1.1727 | -0.4132921203939574 | -1.1444725123832062 | -1.1444725123832062 | 0.4292 | 0.5927951847204773 | 312 |
| 6 | 2006-01-01 | 2024-12-31 | 2025-01-01 | 2026-07-10 | 80 | 1.8267 | 21.5409% | 1.4444 | 13.1697% | -0.3823 | -8.3712% | 1.8891 | 0.4996151931784539 | -1.076273669633716 | -1.076273669633716 | 0.5358 | 0.6519009203096195 | 240 |

![Walk-forward Sharpe by fold](../data/backtests/long_only/figures/walk_forward_sharpe.png)

![ECDF Sharpe edge by fold](../data/backtests/long_only/figures/walk_forward_ecdf_edge.png)

## Transaction-cost sensitivity (production window)

| transaction_cost_bps | strategy | annualized_return | sharpe | max_drawdown | hit_rate | n_weeks | ecdf_sharpe_edge_vs_m1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | equal_weight_1_7 | 0.0781 | 0.7445 | -0.2234 | 0.5640 | 289 | — |
| 0.0 | m1_only | 0.0921 | 0.9045 | -0.1998 | 0.6021 | 289 | — |
| 0.0 | m1_m2_m3_ecdf | 0.0484 | 0.9637 | -0.0749 | 0.5606 | 289 | 0.0592 |
| 5.0 | equal_weight_1_7 | 0.0781 | 0.7445 | -0.2234 | 0.5640 | 289 | — |
| 5.0 | m1_only | 0.0892 | 0.8754 | -0.2017 | 0.5986 | 289 | — |
| 5.0 | m1_m2_m3_ecdf | 0.0458 | 0.9137 | -0.0751 | 0.5502 | 289 | 0.0383 |
| 10.0 | equal_weight_1_7 | 0.0781 | 0.7445 | -0.2234 | 0.5640 | 289 | — |
| 10.0 | m1_only | 0.0863 | 0.8464 | -0.2035 | 0.5917 | 289 | — |
| 10.0 | m1_m2_m3_ecdf | 0.0433 | 0.8638 | -0.0754 | 0.5398 | 289 | 0.0174 |
| 25.0 | equal_weight_1_7 | 0.0781 | 0.7445 | -0.2234 | 0.5640 | 289 | — |
| 25.0 | m1_only | 0.0777 | 0.7597 | -0.2090 | 0.5848 | 289 | — |
| 25.0 | m1_m2_m3_ecdf | 0.0358 | 0.7142 | -0.0761 | 0.5190 | 289 | -0.0455 |

## Implications

- Report **fold-level** ECDF edge alongside the single 2021+ test table in `final_report.md`.
- If edge is fold-dependent, prefer **regime-conditioned M3** or accept ECDF as a drawdown tool, not return engine.
- M1-only remains the return-oriented sleeve when ECDF edge is negative on a fold.

Related: [evaluation_analysis.md](evaluation_analysis.md) · [final_report.md](final_report.md)
