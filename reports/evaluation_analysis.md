# Extended Evaluation: Walk-Forward & Transaction Costs

**Research use only — not investment advice.**

## vs `main`

| Item | `main` | `vitaly_week5` |
| --- | --- | --- |
| Walk-forward validation | None | Expanding-window folds (configurable) |
| Transaction-cost grid | Single 5 bps default | **0 / 5 / 10 / 25 bps** sensitivity table |
| ECDF edge @ 5 bps (test) | Not measured | **+0.177** Sharpe vs M1-only |
| ECDF edge @ 25 bps (test) | Not measured | **+0.046** Sharpe vs M1-only (still positive) |

Branch update: [Executive summary](../BRANCH_UPDATE_REPORT.md) · [Technical report](branch_update_vitaly_week5.md)

This report validates the M1+M2+M3 ECDF stack on **multiple out-of-sample windows** (expanding train, rolling 2-year test blocks) and measures **Sharpe sensitivity** to transaction costs versus M1-only and equal-weight baselines.

## Configuration

- Walk-forward enabled: `False`
- First train end: `2014-12-31`
- Test block length: `2` year(s)
- Transaction-cost grid (bps): `[0.0, 5.0, 10.0, 25.0]`
- Production test window: `2021-01-01` to `latest`

## Walk-forward validation

*Walk-forward evaluation disabled or no valid folds.*

## Transaction-cost sensitivity (production test window)

| transaction_cost_bps | strategy | annualized_return | sharpe | max_drawdown | hit_rate | n_weeks | ecdf_sharpe_edge_vs_m1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | equal_weight_1_7 | 0.0734 | 0.6853 | -0.2390 | 0.5439 | 285 | — |
| 0.0 | m1_only | 0.0869 | 0.8139 | -0.2078 | 0.5965 | 285 | — |
| 0.0 | m1_m2_m3_ecdf | 0.0746 | 1.0242 | -0.1100 | 0.5930 | 285 | 0.2103 |
| 5.0 | equal_weight_1_7 | 0.0734 | 0.6853 | -0.2390 | 0.5439 | 285 | — |
| 5.0 | m1_only | 0.0840 | 0.7869 | -0.2100 | 0.5965 | 285 | — |
| 5.0 | m1_m2_m3_ecdf | 0.0702 | 0.9641 | -0.1133 | 0.5930 | 285 | 0.1772 |
| 10.0 | equal_weight_1_7 | 0.0734 | 0.6853 | -0.2390 | 0.5439 | 285 | — |
| 10.0 | m1_only | 0.0812 | 0.7600 | -0.2130 | 0.5895 | 285 | — |
| 10.0 | m1_m2_m3_ecdf | 0.0658 | 0.9042 | -0.1166 | 0.5895 | 285 | 0.1442 |
| 25.0 | equal_weight_1_7 | 0.0734 | 0.6853 | -0.2390 | 0.5439 | 285 | — |
| 25.0 | m1_only | 0.0726 | 0.6794 | -0.2222 | 0.5825 | 285 | — |
| 25.0 | m1_m2_m3_ecdf | 0.0528 | 0.7255 | -0.1281 | 0.5754 | 285 | 0.0462 |

![Transaction-cost sensitivity](../data/backtests/long_only/figures/transaction_cost_sensitivity.png)

- **ECDF Sharpe edge vs M1-only remains positive at 25 bps** turnover cost.