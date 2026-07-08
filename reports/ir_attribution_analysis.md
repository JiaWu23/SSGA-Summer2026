# IR Attribution Analysis

**Research use only — not investment advice.**

Information Ratio (IR) measures **consistency of beating equal-weight (EW)** week-by-week:
`IR = mean(strategy − EW) × √52 / tracking_error`.

M2/M3 can **raise Sharpe** while **lowering IR** when the strategy deploys less capital
or lags broad EW rallies — especially in selective top-K sleeves.

**Test window:** `2021-01-01` onward

## Test-period IR vs EW

| strategy | annualized_return | sharpe | excess_return_vs_benchmark | information_ratio | mean_active_return_ann | tracking_error_ann | mean_gross_exposure | mean_gross_vs_ew | return_correlation_vs_ew | pct_weeks_ew_outperformed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m1_only | 8.4044% | 0.7869 | 1.0651% | 0.2005 | 0.9870% | 4.9229% | 81.9721% | -18.0279% | 0.8941 | 42.4561% |
| m1_m2_m3_ecdf | 7.0210% | 0.9641 | -0.3184% | -0.1018 | -0.6047% | 5.9383% | 52.3333% | -47.6667% | 0.8492 | 49.8246% |
| m1_m2_m3_linear | 1.8668% | 0.8597 | -5.4725% | -0.6565 | -5.7852% | 8.8117% | 16.4049% | -83.5951% | 0.8977 | 54.3860% |
| m1_m2_m3_binary | 8.4044% | 0.7869 | 1.0651% | 0.2005 | 0.9870% | 4.9229% | 81.9721% | -18.0279% | 0.8941 | 42.4561% |

## Active return by regime (test)

| regime_flag | regime_state | n_weeks | mean_active_return_ann | information_ratio | pct_weeks_ew_outperformed | strategy | period |
| --- | --- | --- | --- | --- | --- | --- | --- |
| risk_off | on | 50 | -2.5782% | -0.3803 | 42.0000% | m1_only | test |
| risk_off | off | 235 | 1.7455% | 0.3930 | 42.5532% | m1_only | test |
| curve_inverted | off | 285 | 0.9870% | 0.2005 | 42.4561% | m1_only | test |
| inflation_up | on | 115 | 0.2853% | 0.0563 | 46.9565% | m1_only | test |
| inflation_up | off | 170 | 1.4617% | 0.3020 | 39.4118% | m1_only | test |
| growth_down | on | 122 | 1.4732% | 0.3120 | 40.9836% | m1_only | test |
| growth_down | off | 163 | 0.6231% | 0.1226 | 43.5583% | m1_only | test |
| risk_off | on | 50 | 0.5130% | 0.0748 | 50.0000% | m1_m2_m3_ecdf | test |
| risk_off | off | 235 | -0.8425% | -0.1468 | 49.7872% | m1_m2_m3_ecdf | test |
| curve_inverted | off | 285 | -0.6047% | -0.1018 | 49.8246% | m1_m2_m3_ecdf | test |
| inflation_up | on | 115 | 2.6160% | 0.3557 | 44.3478% | m1_m2_m3_ecdf | test |
| inflation_up | off | 170 | -2.7834% | -0.5862 | 53.5294% | m1_m2_m3_ecdf | test |
| growth_down | on | 122 | -2.8404% | -0.4786 | 50.0000% | m1_m2_m3_ecdf | test |
| growth_down | off | 163 | 1.0687% | 0.1796 | 49.6933% | m1_m2_m3_ecdf | test |
| risk_off | on | 50 | -13.0689% | -1.2450 | 60.0000% | m1_m2_m3_linear | test |
| risk_off | off | 235 | -4.2355% | -0.5030 | 53.1915% | m1_m2_m3_linear | test |
| curve_inverted | off | 285 | -5.7852% | -0.6565 | 54.3860% | m1_m2_m3_linear | test |
| inflation_up | on | 115 | -0.0952% | -0.0095 | 46.9565% | m1_m2_m3_linear | test |
| inflation_up | off | 170 | -9.6343% | -1.2289 | 59.4118% | m1_m2_m3_linear | test |
| growth_down | on | 122 | -7.9930% | -0.9283 | 53.2787% | m1_m2_m3_linear | test |
| growth_down | off | 163 | -4.1327% | -0.4603 | 55.2147% | m1_m2_m3_linear | test |
| risk_off | on | 50 | -2.5782% | -0.3803 | 42.0000% | m1_m2_m3_binary | test |
| risk_off | off | 235 | 1.7455% | 0.3930 | 42.5532% | m1_m2_m3_binary | test |
| curve_inverted | off | 285 | 0.9870% | 0.2005 | 42.4561% | m1_m2_m3_binary | test |
| inflation_up | on | 115 | 0.2853% | 0.0563 | 46.9565% | m1_m2_m3_binary | test |
| inflation_up | off | 170 | 1.4617% | 0.3020 | 39.4118% | m1_m2_m3_binary | test |
| growth_down | on | 122 | 1.4732% | 0.3120 | 40.9836% | m1_m2_m3_binary | test |
| growth_down | off | 163 | 0.6231% | 0.1226 | 43.5583% | m1_m2_m3_binary | test |

## Key findings

- **M1-only** test IR 0.2005 with 1.0651% excess return vs EW.
- **ECDF** test IR -0.1018 — Sharpe 0.9641 vs M1 0.7869, but gross exposure ~52.3333% (-47.6667% vs EW).
- EW outperforms on ~49.8246% of ECDF weeks (active return < 0).

Related: [ir_improvement_research.md](ir_improvement_research.md) · [TERMINOLOGY.md](../TERMINOLOGY.md)
