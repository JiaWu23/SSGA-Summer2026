# Extended Evaluation: Walk-Forward & Transaction Costs

**Research use only — not investment advice.**

This report validates the M1+M2+M3 ECDF stack on **multiple out-of-sample windows** (expanding train, rolling 2-year test blocks) and measures **Sharpe sensitivity** to transaction costs versus M1-only and equal-weight baselines.

## Configuration

- Walk-forward enabled: `True`
- First train end: `2014-12-31`
- Test block length: `2` year(s)
- Transaction-cost grid (bps): `[0.0, 5.0, 10.0, 25.0]`
- Production test window: `2021-01-01` to `latest`

## Walk-forward validation

- **Mean ECDF Sharpe edge vs M1-only (across folds):** -0.1898
- **Mean M2 AUC (test, across folds):** 0.4864

| fold_id | train_start | train_end | test_start | test_end | test_weeks | m1_only_sharpe | m1_only_ann_return | ecdf_sharpe | ecdf_ann_return | ecdf_sharpe_edge_vs_m1 | ecdf_return_edge_vs_m1 | equal_weight_sharpe | m1_ir | ecdf_ir | ir_edge_vs_ew | m2_auc | m2_auc_pr | m2_n_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2006-01-01 | 2014-12-31 | 2015-01-01 | 2016-12-31 | 105 | -0.1285 | -0.01014977063872069 | -0.5124 | -0.023029761415103378 | -0.3839 | -0.012879990776382688 | 0.0996 | -0.39060885532756784 | -0.5257141908472133 | -0.5257141908472133 | 0.4636 | 0.47169860418726905 | 315 |
| 2 | 2006-01-01 | 2016-12-31 | 2017-01-01 | 2018-12-31 | 104 | 0.9473 | 0.07976682299483073 | 0.2507 | 0.01164155627375485 | -0.6966 | -0.06812526672107588 | 0.5140 | 0.9009464574584505 | -0.5754901022661866 | -0.5754901022661866 | 0.4114 | 0.5486849104711295 | 312 |
| 3 | 2006-01-01 | 2018-12-31 | 2019-01-01 | 2020-12-31 | 104 | 0.5695 | 0.06596479944866518 | 0.4329 | 0.04028077168547761 | -0.1366 | -0.025684027763187567 | 0.8547 | -0.7643713943700546 | -0.9837458961189556 | -0.9837458961189556 | 0.4982 | 0.666760410786915 | 312 |
| 4 | 2006-01-01 | 2020-12-31 | 2021-01-01 | 2022-12-31 | 105 | -0.0569 | -0.00567882643499229 | 0.8621 | 0.031047281248522962 | 0.9190 | 0.03672610768351525 | -0.2795 | 0.431591604309763 | 0.5526835711766697 | 0.5526835711766697 | 0.5804 | 0.6402936075840315 | 315 |
| 5 | 2006-01-01 | 2022-12-31 | 2023-01-01 | 2024-12-31 | 104 | 1.0932 | 0.09761181374899408 | 0.6346 | 0.030388516013010847 | -0.4586 | -0.06722329773598323 | 1.1727 | -0.4132921203939574 | -1.1444725123832062 | -1.1444725123832062 | 0.4292 | 0.5927951847204773 | 312 |
| 6 | 2006-01-01 | 2024-12-31 | 2025-01-01 | 2026-07-10 | 80 | 1.8267 | 0.21540919382035018 | 1.4444 | 0.13169699052958905 | -0.3823 | -0.08371220329076112 | 1.8891 | 0.4996151931784539 | -1.076273669633716 | -1.076273669633716 | 0.5358 | 0.6519009203096195 | 240 |

![Walk-forward Sharpe](../data/backtests/long_only/figures/walk_forward_sharpe.png)

## Transaction-cost sensitivity (production test window)

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

![Transaction-cost sensitivity](../data/backtests/long_only/figures/transaction_cost_sensitivity.png)

- ECDF Sharpe edge vs M1-only **does not persist** at 25 bps under this test window.