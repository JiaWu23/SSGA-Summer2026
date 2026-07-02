# Modeling Specification: M1, M2, Meta-Labels, and Position Sizing

## M1 Primary Model

### Objective

M1 generates candidate trade sides for each asset and date.

```text
M1_output ∈ {-1, 0, 1}
```

Interpretation:

```text
-1 = open/maintain short position
 0 = close existing position or remain flat
 1 = open/maintain long position
```

M1 should prioritize broad opportunity detection. It does not need perfect precision because M2 will filter false positives.

---

## M1 Baseline: Rule-Based Factor Score

Build normalized features:

```text
momentum_score = average(z_mom_12w, z_mom_26w, z_mom_52w)
trend_score = z_trend_signal
risk_penalty = z_vol_12w + z_drawdown_26w_abs
macro_score = regime tilts by asset class
```

Example score:

```text
M1_score = 0.40 * momentum_score
         + 0.25 * trend_score
         + 0.20 * macro_score
         - 0.15 * risk_penalty
```

Signal conversion:

```text
if M1_score > long_threshold: M1_signal = 1
elif M1_score < short_threshold: M1_signal = -1
else: M1_signal = 0
```

Thresholds must be selected on training data only.

---

## M1 Alternative: Supervised Classifier

Define M1 target from forward returns:

```text
future_return_h = price_{t+h} / price_t - 1

if future_return_h > positive_threshold:
    y_m1 = 1
elif future_return_h < negative_threshold:
    y_m1 = -1
else:
    y_m1 = 0
```

Candidate models:

- multinomial logistic regression
- random forest classifier
- gradient boosting classifier

Use time-series validation only.

---

## Meta-Label Construction for M2

For each non-zero M1 signal:

```text
trade_return = M1_signal_t * forward_return_h
meta_label = 1 if trade_return > transaction_cost_threshold else 0
```

Rows where `M1_signal = 0` should usually be excluded from M2 training because there is no proposed trade to validate.

Required columns:

```text
date
ticker
M1_signal
M1_score
forward_return_h
trade_return
meta_label
```

---

## M2 Secondary Model

### Objective

M2 estimates:

```text
p_success = P(meta_label = 1 | M2_features)
```

### Baseline Models

Start with:

1. Logistic regression with class weighting.
2. Random forest classifier.

Then optionally add:

- gradient boosting
- calibrated classifier
- ensemble

### M2 Feature Groups

| Group | Examples |
|---|---|
| Information advantage | M1 input features reused by M2 |
| M1 output | M1 score, signal, score percentile, distance from threshold |
| M1 components | `momentum_score`, `trend_score`, `macro_score`, `risk_penalty` |
| Cross-sectional context | `m1_cs_rank` (weekly rank of M1 score), `m1_score_abs` |
| Interactions | `m1_x_vol`, `m1_x_risk_off`, `m1_x_macro` |
| Asset encoding | One-hot asset class (equity, bond, credit, gold, reit) |
| False-positive modeling | VIX, dispersion, volatility, correlation, liquidity, rolling M1 hit rate |
| Regime | inflation, growth, credit, rates, risk-off flags |
| LLM-derived | sentiment, macro narrative, uncertainty, policy narrative |

### Architecture (config: `models.m2`)

| Setting | Values | Notes |
|---|---|---|
| `architecture` | `global` (default), `per_asset` | Per-asset heads available but tend to overfit with ~300 labels/asset |
| `use_meta_features` | `true` (default) | Adds M1 context + interactions; improved test AUC vs legacy |
| `include_asset_encoding` | `true` (default) | Asset-class dummies |
| `type` | `logistic_regression`, `random_forest`, `gradient_boosting` | Prefer calibrated logistic for ranking stability |
| `min_asset_samples` | 80 | Minimum rows to fit a per-asset head |

Diagnostics write `m2_architecture_benchmark.csv` comparing legacy vs configured model AUC.

---

## M2 Probability Calibration

Add optional probability calibration:

- Platt scaling / sigmoid calibration.
- Isotonic calibration.

Evaluate:

- Brier score.
- Calibration curve.
- Reliability plot.
- Strategy performance after sizing.

Important: fit calibrator using training/validation data only, not test data.

---

## M3 Bet Sizing (Joubert M3 layer)

M2 outputs `p_success`. **M3** maps that probability to a bet fraction `M3_size` ∈ [0, 1]. This is not a classifier—binary thresholding is an all-or-nothing M3 rule.

Reference: Joubert (2022), *Journal of Financial Data Science*, Exhibit 2 (M3 bet-sizing algorithm).

### Size Inputs

```text
M1_signal ∈ {-1, 0, 1}
p_success ∈ [0, 1]   # M2 output
M3_size ∈ [0, 1]     # M3 output
base_risk_budget per asset
portfolio risk constraints (post-M3)
```

### Method 1: Binary M3 (threshold T)

```text
if p_success >= threshold:
    M3_size = 1
else:
    M3_size = 0
```

### Method 2: Linear M3

```text
M3_size = max(0, 2 * p_success - 1)
```

This means:

```text
p = 0.50 -> M3_size = 0.00
p = 0.75 -> M3_size = 0.50
p = 1.00 -> M3_size = 1.00
```

### Method 3: ECDF M3

Fit the empirical distribution of M2 probabilities on training data.

```text
M3_size_t = percentile_rank_train_distribution(p_success_t)
```

### Allocation states (interpretation)

```text
M1 = 0              -> no_signal   (no buy candidate)
M1 ≠ 0, M3_size = 0 -> m3_zero     (candidate rejected by M3 rule)
M1 ≠ 0, M3_size > 0 -> m3_active   (positive bet fraction)
```

---

## Portfolio Weight Construction

Raw signed asset weight:

```text
raw_weight_asset_t = M1_signal_asset_t * M3_size_asset_t * base_budget_asset_t
```

Default base budget:

```text
base_budget_asset = 1 / 7
```

Risk constraints:

```text
max_abs_asset_weight = 0.25
max_gross_exposure = 1.00
allow_short = true
```

Normalize:

```text
if sum(abs(raw_weights)) > max_gross_exposure:
    weights = raw_weights * max_gross_exposure / sum(abs(raw_weights))
else:
    weights = raw_weights
```

---

## Time-Series Validation

Use expanding-window validation.

Example folds:

```text
Fold 1: train 2006-2012, validate 2013
Fold 2: train 2006-2013, validate 2014
Fold 3: train 2006-2014, validate 2015
...
Final test: 2021-latest
```

Never shuffle.

---

## Model Selection Criteria

Do not choose a model only because it has the highest backtested return.

Evaluate:

1. Out-of-sample Sharpe.
2. Max drawdown.
3. Turnover and transaction-cost sensitivity.
4. M2 precision/recall/F1/AUC.
5. Calibration quality.
6. Stability across folds.
7. Simplicity and interpretability.

---

## Required Comparisons

For final report, compare:

```text
Benchmark: equal-weight 1/7
Benchmark: 60/40
M1-only strategy
M1 + M2 binary filter
M1 + M2 linear probability sizing
M1 + M2 ECDF sizing
```

---

## Failure Modes to Monitor

- M1 produces too few signals for M2 training.
- M1 has very low recall.
- M2 predicts probabilities clustered near 0.5.
- Strategy improvement comes only from lower exposure, not better selection.
- Test-period turnover is too high.
- Performance is concentrated in one asset or regime.
- LLM features appear predictive only because of hidden future leakage.
