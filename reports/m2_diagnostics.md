# M2 Diagnostics & AUC-ROC Guide

**Research use only — not investment advice.**

## vs `main`

| Metric | `main` (legacy 40 features) | Branch (52 enriched features) | Δ |
| --- | ---: | ---: | ---: |
| Test AUC-ROC | 0.5727 | **0.5890** | **+0.016** |
| Test AUC (reported) | ~0.57 | **0.5884** | +0.018 |
| Feature count | 40 | 52 | +12 |
| Calibration / decile charts | Placeholder or absent | Real ROC + decile return charts | new |
| Architecture benchmark CSV | No | `m2_architecture_benchmark.csv` | new |

Per-asset heads and tree models were tested; test AUC **~0.48–0.50** (overfit) — **not** adopted.

Branch update: [Executive summary](../BRANCH_UPDATE_REPORT.md) · [Technical report](branch_update_vitaly_week5.md)

## Classifier Metrics (Test Set)

| Metric | Value | Meaning |
| --- | --- | --- |
| Accuracy | 0.5895 | Share of correct meta-label predictions |
| Precision | 0.5895 | Approved trades that were actually profitable |
| Recall | 1.0000 | Profitable trades that M2 approved |
| F1 Score | 0.7417 | Balance of precision and recall |
| AUC-ROC | 0.5884 | Ranking quality: P(random winner scored higher than random loser) |
| AUC-PR | 0.6634 | Precision-recall AUC; more informative when base rate ≠ 50% |
| Base Rate | 58.9474% | Fraction of M1 trades that beat the cost hurdle |
| Brier Score | 0.2403 | Probability calibration error (lower is better) |
| Mean P (winners) | 0.5973 | Average M2 probability on profitable trades |
| Mean P (losers) | 0.5934 | Average M2 probability on unprofitable trades |
| Mean IC | 0.1061 | Spearman rank correlation of M1 scores vs forward returns |
| Note | — | Binary M3 at this threshold approves all trades; strategy equals M1-only. |

## Understanding AUC-ROC

AUC-ROC measures **ranking quality**, not accuracy. If you randomly pick one winning trade and one losing trade, AUC is the probability M2 assigns a higher `P(success)` to the winner. At **0.5884**, discrimination is only slightly above random (0.50).

| AUC | Interpretation |
| --- | --- |
| 0.50 | Random ranking — no discrimination |
| 0.55–0.60 | Weak but common for noisy financial labels |
| 0.70+ | Moderate discrimination |

**Base rate** (fraction of profitable M1 trades): 58.9474%. When base rate ≠ 50%, **AUC-PR** (0.6634) is often more informative than AUC-ROC.

- **AUC vs Brier:** Brier scores calibration (predicted vs realized); AUC scores ranking. A model can be calibrated but still rank poorly.
- **AUC vs precision/recall:** AUC is threshold-independent. At threshold **0.55**, recall=1.0000 — if recall ≈ 1.0, binary M3 at that threshold approves all trades and adds no filter.
- **Economic role:** M2 outputs probabilities only; **M3** converts them to bet fractions. Threshold approval at 0.55 is an M3 binary sizing rule, not M2 classification output.

![ROC and calibration](../data/backtests/long_only/figures/m2_roc_calibration.png)

## Calibration by Probability Bucket

| bucket | n | mean_pred | realized |
| --- | --- | --- | --- |
| (0.558, 0.579] | 86 | 0.5724 | 50.0000% |
| (0.579, 0.585] | 85 | 0.5824 | 49.4118% |
| (0.585, 0.59] | 86 | 0.5876 | 61.6279% |
| (0.59, 0.593] | 85 | 0.5916 | 47.0588% |
| (0.593, 0.596] | 86 | 0.5948 | 53.4884% |
| (0.596, 0.599] | 85 | 0.5976 | 63.5294% |
| (0.599, 0.602] | 85 | 0.6007 | 50.5882% |
| (0.602, 0.606] | 86 | 0.6041 | 66.2791% |
| (0.606, 0.613] | 85 | 0.6089 | 70.5882% |
| (0.613, 0.639] | 86 | 0.6169 | 76.7442% |

## Economic View: Return by Probability Decile

| decile | n | mean_p_success | mean_trade_return | hit_rate |
| --- | --- | --- | --- | --- |
| (0.558, 0.579] | 85 | 0.5724 | -0.1014% | 50.5882% |
| (0.579, 0.585] | 84 | 0.5825 | 0.3375% | 51.1905% |
| (0.585, 0.59] | 84 | 0.5878 | 0.8093% | 65.4762% |
| (0.59, 0.593] | 84 | 0.5918 | -0.1769% | 45.2381% |
| (0.593, 0.596] | 85 | 0.5949 | 0.8813% | 55.2941% |
| (0.596, 0.599] | 84 | 0.5977 | 1.0617% | 63.0952% |
| (0.599, 0.602] | 84 | 0.6008 | 0.2831% | 53.5714% |
| (0.602, 0.607] | 84 | 0.6043 | 1.4028% | 65.4762% |
| (0.607, 0.613] | 84 | 0.6090 | 1.3195% | 71.4286% |
| (0.613, 0.639] | 85 | 0.6169 | 2.4405% | 76.4706% |

![Decile returns](../data/backtests/long_only/figures/m2_decile_returns.png)

## Feature Importance (Top 15)

| feature | coefficient |
| --- | --- |
| growth_trend | -0.5126 |
| growth_down | -0.4836 |
| inflation_up | -0.4732 |
| drawdown_26w | 0.4619 |
| mom_vol_interaction | -0.4551 |
| z_vol_12w | -0.4326 |
| m1_x_risk_off | -0.4201 |
| trend_signal | 0.3875 |
| vix_level | 0.3836 |
| inflation_trend | 0.3637 |
| mom_52w | -0.3486 |
| z_mom_52w | 0.3227 |
| vol_12w | 0.3054 |
| z_trend_signal | -0.2596 |
| mom_26w | -0.2584 |

![Feature importance](../data/backtests/long_only/figures/m2_feature_importance.png)

## Architecture Benchmark (train vs test AUC)

Compares legacy global logistic regression against enriched features and per-asset heads.

| variant | model_type | architecture | n_features | n_train | n_test | train_auc | test_auc | asset_heads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_global | logistic_regression | global | 40 | 2103 | 855 | 0.6464 | 0.5727 | 0 |
| configured | logistic_regression | global | 52 | 2103 | 855 | 0.6460 | 0.5890 | 0 |

## Metrics by Asset

| ticker | n_trades | base_rate | auc | approval_rate | mean_trade_return |
| --- | --- | --- | --- | --- | --- |
| GLD | 218 | 0.6009 | 0.6324 | 1.0 | 0.0146 |
| SPY | 219 | 0.6301 | 0.4711 | 1.0 | 0.0081 |
| TLT | 17 | 0.2941 | 0.6 | 1.0 | -0.0172 |
| VEA | 180 | 0.6 | 0.6721 | 1.0 | 0.0127 |
| VNQ | 107 | 0.5327 | 0.494 | 1.0 | -0.0081 |
| VWO | 114 | 0.5702 | 0.7042 | 1.0 | 0.009 |

## Metrics by Regime Flag

| risk_off | n_trades | base_rate | auc | approval_rate | mean_trade_return | curve_inverted | inflation_up | growth_down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 705.0 | 0.6057 | 0.584 | 1.0 | 0.0082 | nan | nan | nan |
| 1.0 | 150.0 | 0.5133 | 0.6195 | 1.0 | 0.0088 | nan | nan | nan |
| nan | 855.0 | 0.5895 | 0.5884 | 1.0 | 0.0083 | 0.0 | nan | nan |
| nan | 510.0 | 0.6627 | 0.5627 | 1.0 | 0.0139 | nan | 0.0 | nan |
| nan | 345.0 | 0.4812 | 0.5148 | 1.0 | -0.0004 | nan | 1.0 | nan |
| nan | 489.0 | 0.5583 | 0.6051 | 1.0 | 0.0056 | nan | nan | 0.0 |
| nan | 366.0 | 0.6311 | 0.5645 | 1.0 | 0.0117 | nan | nan | 1.0 |
