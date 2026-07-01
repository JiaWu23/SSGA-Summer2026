# M2 Diagnostics & AUC-ROC Guide

**Research use only — not investment advice.**

## Classifier Metrics (Test Set)

| Metric | Value | Meaning |
| --- | --- | --- |
| Accuracy | 0.5930 | Share of correct meta-label predictions |
| Precision | 0.5915 | Approved trades that were actually profitable |
| Recall | 1.0000 | Profitable trades that M2 approved |
| F1 Score | 0.7434 | Balance of precision and recall |
| AUC-ROC | 0.5727 | Ranking quality: P(random winner scored higher than random loser) |
| AUC-PR | 0.6652 | Precision-recall AUC; more informative when base rate ≠ 50% |
| Base Rate | 58.9474% | Fraction of M1 trades that beat the cost hurdle |
| Brier Score | 0.2404 | Probability calibration error (lower is better) |
| Mean P (winners) | 0.5953 | Average M2 probability on profitable trades |
| Mean P (losers) | 0.5915 | Average M2 probability on unprofitable trades |
| Mean IC | 0.1061 | Spearman rank correlation of M1 scores vs forward returns |
| Note | — | M2 approves all trades at this threshold; binary M2 equals M1-only. |

## Understanding AUC-ROC

AUC-ROC measures **ranking quality**, not accuracy. If you randomly pick one winning trade and one losing trade, AUC is the probability M2 assigns a higher `P(success)` to the winner. At **0.5727**, discrimination is only slightly above random (0.50).

| AUC | Interpretation |
| --- | --- |
| 0.50 | Random ranking — no discrimination |
| 0.55–0.60 | Weak but common for noisy financial labels |
| 0.70+ | Moderate discrimination |

**Base rate** (fraction of profitable M1 trades): 58.9474%. When base rate ≠ 50%, **AUC-PR** (0.6652) is often more informative than AUC-ROC.

- **AUC vs Brier:** Brier scores calibration (predicted vs realized); AUC scores ranking. A model can be calibrated but still rank poorly.
- **AUC vs precision/recall:** AUC is threshold-independent. At threshold **0.55**, recall=1.0000 — if recall ≈ 1.0, binary M2 approves all trades and adds no filter.
- **Economic role:** M2 value in this pipeline is mainly **ECDF sizing** (risk shaping), not rejecting trades.

![ROC and calibration](../data/backtests/long_only/figures/m2_roc_calibration.png)

## Calibration by Probability Bucket

| bucket | n | mean_pred | realized |
| --- | --- | --- | --- |
| (0.545, 0.574] | 86 | 0.5659 | 51.1628% |
| (0.574, 0.582] | 85 | 0.5778 | 65.8824% |
| (0.582, 0.587] | 86 | 0.5847 | 51.1628% |
| (0.587, 0.591] | 85 | 0.5890 | 49.4118% |
| (0.591, 0.594] | 86 | 0.5928 | 54.6512% |
| (0.594, 0.598] | 85 | 0.5960 | 45.8824% |
| (0.598, 0.601] | 85 | 0.5993 | 54.1176% |
| (0.601, 0.606] | 86 | 0.6036 | 65.1163% |
| (0.606, 0.613] | 85 | 0.6095 | 75.2941% |
| (0.613, 0.639] | 86 | 0.6189 | 76.7442% |

## Economic View: Return by Probability Decile

| decile | n | mean_p_success | mean_trade_return | hit_rate |
| --- | --- | --- | --- | --- |
| (0.545, 0.574] | 85 | 0.5658 | 0.2486% | 51.7647% |
| (0.574, 0.582] | 84 | 0.5781 | 1.5045% | 71.4286% |
| (0.582, 0.587] | 84 | 0.5849 | 0.2821% | 51.1905% |
| (0.587, 0.591] | 84 | 0.5892 | 0.1053% | 51.1905% |
| (0.591, 0.595] | 85 | 0.5930 | 0.2524% | 52.9412% |
| (0.595, 0.598] | 84 | 0.5961 | -0.1977% | 47.6190% |
| (0.598, 0.601] | 84 | 0.5995 | 0.5376% | 54.7619% |
| (0.601, 0.606] | 84 | 0.6038 | 1.3399% | 64.2857% |
| (0.606, 0.613] | 84 | 0.6096 | 1.6920% | 76.1905% |
| (0.613, 0.639] | 85 | 0.6190 | 2.4953% | 76.4706% |

![Decile returns](../data/backtests/long_only/figures/m2_decile_returns.png)

## Feature Importance (Top 15)

| feature | coefficient |
| --- | --- |
| growth_trend | -0.5363 |
| growth_down | -0.5020 |
| vol_12w | 0.4722 |
| inflation_up | -0.4623 |
| drawdown_26w | 0.4229 |
| mom_vol_interaction | -0.3908 |
| risk_off | -0.3547 |
| trend_signal | 0.3542 |
| z_vol_12w | -0.3493 |
| inflation_trend | 0.3291 |
| z_mom_52w | 0.3238 |
| vix_level | 0.3170 |
| mom_52w | -0.3091 |
| z_mom_12w | 0.2639 |
| mom_26w | -0.2598 |

![Feature importance](../data/backtests/long_only/figures/m2_feature_importance.png)

## Metrics by Asset

| ticker | n_trades | base_rate | auc | approval_rate | mean_trade_return |
| --- | --- | --- | --- | --- | --- |
| GLD | 218 | 0.6009 | 0.6051 | 1.0 | 0.0146 |
| SPY | 219 | 0.6301 | 0.4426 | 1.0 | 0.0081 |
| TLT | 17 | 0.2941 | 0.7 | 1.0 | -0.0172 |
| VEA | 180 | 0.6 | 0.6731 | 0.9889 | 0.0127 |
| VNQ | 107 | 0.5327 | 0.4049 | 1.0 | -0.0081 |
| VWO | 114 | 0.5702 | 0.7096 | 0.9912 | 0.009 |

## Metrics by Regime Flag

| risk_off | n_trades | base_rate | auc | approval_rate | mean_trade_return | curve_inverted | inflation_up | growth_down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 705.0 | 0.6057 | 0.5685 | 0.9957 | 0.0082 | nan | nan | nan |
| 1.0 | 150.0 | 0.5133 | 0.6026 | 1.0 | 0.0088 | nan | nan | nan |
| nan | 855.0 | 0.5895 | 0.5727 | 0.9965 | 0.0083 | 0.0 | nan | nan |
| nan | 510.0 | 0.6627 | 0.5816 | 1.0 | 0.0139 | nan | 0.0 | nan |
| nan | 345.0 | 0.4812 | 0.4735 | 0.9913 | -0.0004 | nan | 1.0 | nan |
| nan | 489.0 | 0.5583 | 0.5957 | 1.0 | 0.0056 | nan | nan | 0.0 |
| nan | 366.0 | 0.6311 | 0.5627 | 0.9918 | 0.0117 | nan | nan | 1.0 |
