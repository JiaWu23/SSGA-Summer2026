# Market & Regime Analysis

**Research use only — not investment advice.**

## Regime Feature Definitions

| Flag / Series | Definition |
| --- | --- |
| `risk_off` | VIX above its 75th percentile (156-week rolling) |
| `curve_inverted` | 10Y–2Y Treasury spread < 0 |
| `inflation_up` | CPI YoY above its 156-week median |
| `growth_down` | Industrial production YoY below its 156-week median |
| `vix_level`, `credit_stress`, `yield_curve` | Continuous macro/risk levels (lagged) |

![Regime timeline](../data/backtests/long_only/figures/regime_timeline.png)

![VIX and flags](../data/backtests/long_only/figures/vix_and_flags.png)

## Regime Transitions

| regime_flag | n_transitions | pct_on | avg_spell_on_weeks | avg_spell_off_weeks |
| --- | --- | --- | --- | --- |
| risk_off | 90 | 0.22 | 3.91 | 13.33 |
| curve_inverted | 0 | 0.0 | nan | 789.0 |
| inflation_up | 20 | 0.41 | 29.55 | 46.4 |
| growth_down | 14 | 0.5 | 56.71 | 49.0 |

## Strategy Performance by Regime

| regime_flag | regime_state | strategy | n_weeks | annualized_return | sharpe | hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| risk_off | on | equal_weight_1_7 | 176 | 23.3987% | 1.4106 | 63.6364% |
| risk_off | on | m1_only | 176 | 14.4643% | 1.1909 | 66.4773% |
| risk_off | on | m1_m2_m3_ecdf | 176 | 5.8263% | 0.7957 | 63.6364% |
| risk_off | on | m1_m2_ecdf | 176 | 5.8263% | 0.7957 | 63.6364% |
| risk_off | off | equal_weight_1_7 | 613 | 2.1203% | 0.2365 | 54.9755% |
| risk_off | off | m1_only | 613 | 3.5786% | 0.4115 | 56.1175% |
| risk_off | off | m1_m2_m3_ecdf | 613 | 0.6833% | 0.1430 | 53.1811% |
| risk_off | off | m1_m2_ecdf | 613 | 0.6833% | 0.1430 | 53.1811% |
| curve_inverted | off | equal_weight_1_7 | 789 | 6.5241% | 0.5838 | 56.9075% |
| curve_inverted | off | m1_only | 789 | 5.9134% | 0.6170 | 58.4284% |
| curve_inverted | off | m1_m2_m3_ecdf | 789 | 1.8085% | 0.3316 | 55.5133% |
| curve_inverted | off | m1_m2_ecdf | 789 | 1.8085% | 0.3316 | 55.5133% |
| inflation_up | on | equal_weight_1_7 | 325 | -2.3169% | -0.1856 | 53.8462% |
| inflation_up | on | m1_only | 325 | 0.0641% | 0.0062 | 54.4615% |
| inflation_up | on | m1_m2_m3_ecdf | 325 | 0.0344% | 0.0058 | 50.7692% |
| inflation_up | on | m1_m2_ecdf | 325 | 0.0344% | 0.0058 | 50.7692% |
| inflation_up | off | equal_weight_1_7 | 464 | 13.1889% | 1.3069 | 59.0517% |
| inflation_up | off | m1_only | 464 | 10.2129% | 1.1309 | 61.2069% |
| inflation_up | off | m1_m2_m3_ecdf | 464 | 3.0697% | 0.6041 | 58.8362% |
| inflation_up | off | m1_m2_ecdf | 464 | 3.0697% | 0.6041 | 58.8362% |
| growth_down | on | equal_weight_1_7 | 397 | 6.6826% | 0.5639 | 55.6675% |
| growth_down | on | m1_only | 397 | 5.2205% | 0.5465 | 58.4383% |
| growth_down | on | m1_m2_m3_ecdf | 397 | 0.8536% | 0.1540 | 54.4081% |
| growth_down | on | m1_m2_ecdf | 397 | 0.8536% | 0.1540 | 54.4081% |
| growth_down | off | equal_weight_1_7 | 392 | 6.3638% | 0.6083 | 58.1633% |
| growth_down | off | m1_only | 392 | 6.6199% | 0.6875 | 58.4184% |
| growth_down | off | m1_m2_m3_ecdf | 392 | 2.7848% | 0.5190 | 56.6327% |
| growth_down | off | m1_m2_ecdf | 392 | 2.7848% | 0.5190 | 56.6327% |

![Performance heatmap](../data/backtests/long_only/figures/performance_by_regime_heatmap.png)

## M1 IC by Regime (Test)

| regime_flag | regime_state | ic_mean | n_weeks |
| --- | --- | --- | --- |
| risk_off | on | -0.0264 | 50 |
| risk_off | off | 0.1236 | 239 |
| curve_inverted | off | 0.0972 | 289 |
| inflation_up | on | 0.0413 | 119 |
| inflation_up | off | 0.1351 | 170 |
| growth_down | on | 0.108 | 122 |
| growth_down | off | 0.0892 | 167 |

## M2 AUC by Regime (Test)

| regime_flag | regime_state | auc | n_trades | base_rate |
| --- | --- | --- | --- | --- |
| risk_off | on | 0.4769 | 150 | 0.5667 |
| risk_off | off | 0.552 | 717 | 0.6165 |
| curve_inverted | off | 0.5389 | 867 | 0.6078 |
| inflation_up | on | 0.5248 | 357 | 0.5182 |
| inflation_up | off | 0.4968 | 510 | 0.6706 |
| growth_down | on | 0.4393 | 366 | 0.6421 |
| growth_down | off | 0.5874 | 501 | 0.5828 |

## Train vs Test Macro Context

| period | feature | mean | std | min | max | pct_on |
| --- | --- | --- | --- | --- | --- | --- |
| train | vix_level | 17.1436 | 6.7668 | 9.51 | 42.9609 | nan |
| train | credit_stress | 2.5748 | 0.4674 | 1.6799 | 3.5604 | nan |
| train | yield_curve | 1.1926 | 0.7216 | 0.0699 | 2.6001 | nan |
| train | growth_trend | 0.0032 | 0.036 | -0.1605 | 0.0435 | nan |
| train | inflation_trend | 0.0171 | 0.0088 | -0.0011 | 0.0375 | nan |
| train | risk_off | 0.252 | 0.4346 | 0.0 | 1.0 | 0.252 |
| train | curve_inverted | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| train | inflation_up | 0.412 | 0.4927 | 0.0 | 1.0 | 0.412 |
| train | growth_down | 0.55 | 0.498 | 0.0 | 1.0 | 0.55 |
| test | vix_level | 19.1801 | 5.1846 | 11.93 | 42.9609 | nan |
| test | credit_stress | 1.875 | 0.2017 | 1.6799 | 2.42 | nan |
| test | yield_curve | 0.4238 | 0.4242 | 0.0699 | 1.58 | nan |
| test | growth_trend | 0.0081 | 0.0187 | -0.0568 | 0.0435 | nan |
| test | inflation_trend | 0.032 | 0.0063 | 0.0132 | 0.0375 | nan |
| test | risk_off | 0.173 | 0.3789 | 0.0 | 1.0 | 0.173 |
| test | curve_inverted | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| test | inflation_up | 0.4118 | 0.493 | 0.0 | 1.0 | 0.4118 |
| test | growth_down | 0.4221 | 0.4948 | 0.0 | 1.0 | 0.4221 |
