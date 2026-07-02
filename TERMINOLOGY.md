# Terminology Guide

**Audience:** software engineers, data scientists, and reviewers who are **not** finance specialists.

This project is a **research backtest** of a weekly multi-asset allocation strategy. Reports use finance and machine-learning jargon. This file explains those terms in plain language and maps them to what the code actually does.

**Related docs:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) · [reports/final_report.md](reports/final_report.md)

---

## How to read this guide

- **Bold terms** in reports usually refer to entries below.
- **Strategy names** (e.g. `equal_weight_1_7`, `m1_m2_m3_ecdf`) are identifiers in code and CSV outputs.
- **M1 / M2 / M3** are layers in the [Joubert meta-labeling framework](#joubert-framework-m1--m2--m3) — not separate products.

---

## Pipeline architecture

### Joubert framework (M1 → M2 → M3)

A three-layer design from Marcos López de Prado’s meta-labeling literature (often cited via Joubert 2022). Think of it as: **pick trades → score trade quality → decide how much capital to risk**.

| Layer | Plain English | Code output |
| --- | --- | --- |
| **M1** | “Should we be long, short, or flat in this asset this week?” | `M1_signal` (−1, 0, +1), `M1_score` |
| **M2** | “If M1 wants a trade, how likely is it to be profitable?” | `p_success` (probability 0–1) |
| **M3** | “Given that probability, what fraction of the base bet should we take?” | `M3_size` (0–1) |
| **Portfolio** | Risk limits: caps per asset, vol target, transaction costs | Final `weight` per asset-week |

### M1 (side / opportunity model)

Rule-based model that scores each ETF each week using factors (momentum, trend, macro, risk). It does **not** use machine learning in the default setup.

- **`M1_signal`**: +1 = go long, −1 = go short, 0 = no trade.
- **`M1_score`**: Continuous score before discretizing to a signal.
- **`M1_conviction`**: Optional scaling of position size by how strong the M1 score is (currently **off** in production config).
- **Factor families**: Groups of features — momentum, trend, macro, risk — combined with configurable weights.

### M2 (meta-label model)

A **classifier** (logistic regression) trained only on weeks where M1 actually proposed a trade (`M1_signal ≠ 0`).

- **`meta_label`**: Training label — 1 if the M1 trade would have been profitable over the forward horizon (after costs), 0 otherwise.
- **`p_success`**: M2’s predicted probability that the trade succeeds. This is the main M2 output used downstream.
- **`predicted_meta_label`**: Diagnostic only — “would we approve at threshold T?” Not the primary production output.

**Meta-labeling** means: the first model (M1) picks *direction*; the second model (M2) judges *whether that specific trade is worth taking*.

### M3 (bet-sizing layer)

Deterministic rule that converts `p_success` into a **bet fraction** `M3_size` ∈ [0, 1]. M3 is **not** another classifier.

| M3 mode | Code name | Plain English |
| --- | --- | --- |
| **Binary** | `binary` | Full size if `p_success ≥ T`, else zero |
| **Linear** | `linear` | Size = `max(0, 2 × p_success − 1)` — scales smoothly with confidence |
| **ECDF** | `ecdf` | Size = percentile rank of `p_success` vs training distribution — relative sizing |
| **Passthrough** | `passthrough` | Diagnostic: use raw `p_success` as size (research only) |
| **Linear gated** | (research) | Linear size, but zero if `p_success` below a gate threshold |

**Allocation states** (on the panel):

| State | Meaning |
| --- | --- |
| `no_signal` | M1 said flat — no candidate trade |
| `m3_zero` | M1 wanted a trade, but M3 sized it to zero (rejected) |
| `m3_active` | M1 wanted a trade and M3 assigned positive size |

### Backtest

Running the strategy on **historical** prices as if trades had been executed in the past. No real money, no live broker. Outputs: return series, weights, Sharpe, drawdown, etc.

### Walk-forward evaluation

Instead of one train/test split, the model is re-trained on expanding windows and tested on the next block of time (e.g. 2-year folds). Used to check whether edge is **stable across regimes**, not just lucky on 2021+.

### Train / test split

| Term | Meaning |
| --- | --- |
| **In-sample / train** | Dates used to fit M2 and calibrate thresholds (default: through 2020-12-31) |
| **Out-of-sample / test / OOS** | Dates held back for honest evaluation (default: 2021-01-01 onward) |
| **Full sample** | Train + test combined — easier to look good; always check test-only tables too |

---

## Strategies and benchmarks (code names)

These strings appear in backtest outputs, charts, and `metrics_table`.

| Code / report name | Plain English |
| --- | --- |
| **`equal_weight_1_7`** / **Equal Weight 1/7** | Each week, split money **equally** across all 7 ETFs (~14.3% each). Passive benchmark. |
| **`sixty_forty`** / **60/40** | Classic benchmark: 60% equities (SPY), 40% bonds (TLT). Not equal-weight. |
| **`m1_only`** / **M1 Only** | Follow M1 signals only; no M2/M3 sizing overlay (beyond portfolio vol target). |
| **`m1_m2_m3_binary`** | M1 + M2 probability + binary M3 (all-or-nothing at threshold T). |
| **`m1_m2_m3_linear`** | M1 + M2 + linear M3 sizing. |
| **`m1_m2_m3_ecdf`** | M1 + M2 + ECDF M3 sizing — best risk-adjusted variant in current research. |
| **`m1_m2_passthrough`** | Diagnostic: M3 size = raw `p_success` (not a production recommendation). |

**Long-only** vs **long/short**:

- **Long-only**: Only buy assets (positive weights). Shorts disabled (`allow_short: false`).
- **Long/short**: M1 may emit −1 and bet against an asset. Often weaker in this ETF universe.

---

## Investments and instruments

### ETF (Exchange-Traded Fund)

A fund that trades on a stock exchange like a share, but holds a **basket** of assets. This project uses ETFs as **proxies** for whole asset classes (e.g. SPY ≈ U.S. large-cap stocks).

### Ticker

Short symbol for a tradable instrument: `SPY`, `TLT`, `GLD`, etc.

### Asset class

Category of investments with similar behavior: equities, government bonds, credit, gold, real estate (REITs), etc.

### Tradable universe

The seven ETFs the strategy can actually allocate to: SPY, TLT, GLD, VEA, VWO, HYG, VNQ.

### Macro / risk indicators (not traded)

Series used only as **features** — VIX (fear gauge), CPI, unemployment, yields, etc. from **FRED** (Federal Reserve Economic Data). Lagged in the pipeline to reduce look-ahead bias.

### Adjusted close

Stock/ETF price series corrected for splits and dividends. Used so returns are comparable over time.

### Weekly rebalance

Decisions and weight updates happen once per week (Friday close in this project), not every day.

---

## Portfolio and trading concepts

### Weight

Fraction of portfolio capital allocated to an asset. Sum of absolute weights = **gross exposure**. Example: 0.25 on SPY = 25% of capital in SPY.

### Long / short / flat

| Position | Meaning |
| --- | --- |
| **Long** | Own the asset — profit if price rises |
| **Short** | Bet against the asset — profit if price falls (not used in long-only mode) |
| **Flat** | No position (weight = 0) |

### Gross exposure

Sum of **absolute** weights across assets. 1.0 ≈ fully invested; 0.5 ≈ half the capital is deployed on average.

### Turnover

How much the portfolio **changes** from week to week (sum of absolute weight changes). High turnover → more trading → more transaction costs.

### Transaction cost / TC

Friction from trading: spreads, commissions, market impact. Configured in **bps** (see below). Subtracted from returns in backtests.

### Volatility targeting / vol target

Scale portfolio weights so realized volatility hovers near a target (e.g. **12% annualized**). When markets are wild, the strategy **shrinks** positions; when calm, it may **scale up** (within caps).

### Exposure cap / constraint

Limits such as: max 25% in one ETF (`max_abs_asset_weight`), max total gross exposure 100% (`max_gross_exposure`).

### Top-K allocation (`top_k`)

Each week, rank all ETFs by M1 score and trade only the **top K** names (default K=3). Relative ranking, not a fixed score cutoff.

### Threshold allocation

Alternative M1 mode: go long only if score exceeds a learned cutoff (vs top-K ranking).

### Forward return

Return over the **next** N weeks (default 4). Used for labels and IC — you must not use future data when making today’s decision (pipeline enforces this with features/labels aligned in time).

### Horizon (`horizon_weeks`)

How many weeks ahead the label looks (default 4). “Did this trade work over the next month-ish?”

### Regime

Broad market environment: e.g. crisis, low-vol bull market, rising rates. Performance often differs by regime — see [market_regime_analysis.md](reports/market_regime_analysis.md).

---

## Performance metrics

### Return (annualized)

Average growth rate per year, compounded. 8% annualized ≈ portfolio grew ~8% per year on average over the period.

### Volatility (annualized)

How much returns **bounce around** — standard deviation scaled to a year. Higher vol = bumpier ride.

### Sharpe ratio

**Return per unit of risk** (excess return over risk-free rate, divided by volatility). Higher is better.

- **0.5–0.7**: decent for a simple strategy  
- **~1.0**: strong risk-adjusted (context-dependent)  
- Compare strategies at similar vol and cost assumptions

**Sharpe edge vs M1**: Difference in Sharpe vs the M1-only baseline — “how much did M2/M3 help risk-adjusted performance?”

### Max drawdown

Largest **peak-to-trough** loss. −21% means the portfolio fell 21% from its prior high before recovering. Measures “worst pain” along the path.

### Hit rate / weekly hit rate

Fraction of weeks with **positive** portfolio return (or profitable trades, depending on context).

### Excess return vs benchmark

Strategy return minus benchmark return (e.g. vs equal-weight). Can be negative even when Sharpe improves if the strategy runs lower risk.

### Information ratio (IR)

**Excess return vs equal-weight, per unit of tracking error:**

`IR = mean(strategy − EW) × √52 / std(strategy − EW)`

- **IR > 0**: strategy tends to beat EW consistently week-to-week  
- **IR < 0**: strategy tends to lag EW even if absolute Sharpe looks fine  

### Sharpe vs Information Ratio

| Metric | Question |
| --- | --- |
| **Sharpe** | Return per unit of **total** volatility (vs cash) |
| **IR** | Consistency of **beating equal-weight** |

M1+M2+M3 ECDF often **raises Sharpe** but **lowers IR** because ECDF scales positions down. When all seven ETFs rally, EW captures the full move; a selective, under-invested sleeve lags. See [reports/ir_attribution_analysis.md](reports/ir_attribution_analysis.md).

### Cumulative return

Total growth over the full period, not annualized.

### Base rate

In classification: fraction of positive labels (e.g. % of M1 trades that were actually profitable). Important for interpreting precision/recall.

---

## Machine learning and statistics

### AUC-ROC

**Area Under the ROC Curve.** Measures **ranking quality**: if you pick one random winning trade and one random losing trade, what’s the probability the model scores the winner higher?

- **0.50** = random coin flip  
- **0.55–0.60** = weak but common in finance  
- **0.70+** = moderate discrimination  

Does **not** mean “70% accurate.”

### AUC-PR

Area under the **precision–recall** curve. More informative when successes are **not** 50% of the sample (common here: base rate ~59%).

### Accuracy

Fraction of predictions that are correct. Misleading when classes are imbalanced.

### Precision

Of trades the model **approved**, what fraction were actually winners? “When we say yes, how often are we right?”

### Recall

Of trades that **were** winners, what fraction did the model approve? “Did we catch the good ones?”

- Recall ≈ **1.0** at T=0.55 means binary M3 approves **everyone** — no filtering.

### F1 score

Harmonic mean of precision and recall — single number balancing both.

### Brier score

Measures **calibration**: how close predicted probabilities are to true frequencies. Lower is better. A model can rank well (high AUC) but be poorly calibrated (high Brier).

### Calibration

If the model says “60% chance of success” many times, do ~60% of those trades actually win? M2 uses calibration on the train set when `calibrate: true`.

### Confusion matrix

Table of predicted vs actual labels (true positive, false positive, etc.).

### IC (Information Coefficient)

**Spearman rank correlation** between a score (e.g. M1) and **forward** return. Measures predictive **ranking** of assets, not classification.

- **0.10** is a modest but meaningful IC in many quant contexts  
- Reported per asset and for factor components

### Spearman correlation

Correlation on **ranks** instead of raw values — robust to outliers and non-linear monotonic relationships.

### ECDF (Empirical Cumulative Distribution Function)

A way to map a value to its **percentile** in a reference sample. Here: “where does this week’s `p_success` sit vs all training `p_success` values?” That percentile becomes **M3_size** under ECDF sizing.

- Higher `p_success` relative to history → larger bet fraction  
- No fixed threshold like 0.55 — sizing is **relative**

### Degeneracy (M3 threshold context)

When a threshold is so low that **every** trade passes (recall ≈ 1). The binary layer does nothing; strategy collapses to M1-only.

### Meaningful rejection

Research term: M3 actually zeros out a non-trivial share of M1 candidates (e.g. ≥5% `m3_zero`), so the filter is doing work.

### Overfitting

Model memorizes train data and fails on test data. Why we hold out 2021+ and use walk-forward.

### Winsorize

Clip extreme feature values to reduce outlier influence (e.g. top/bottom 1% on train set).

### Look-ahead bias

Accidentally using **future** information in past decisions. Invalidates backtests. Macro series are lagged; labels use forward returns only for supervision, not features at decision time.

---

## Units and shorthand

| Term | Meaning |
| --- | --- |
| **bps** | Basis points. 1 bp = 0.01%. 5 bps transaction cost ≈ 0.05% per unit of turnover. |
| **ann.** | Annualized (scaled to per-year units). |
| **TC** | Transaction cost. |
| **OOS** | Out-of-sample. |
| **EW** | Equal weight (benchmark). |
| **DD** | Drawdown. |
| **REIT** | Real Estate Investment Trust — property-linked equities (VNQ). |
| **FRED** | U.S. Federal Reserve economic data API. |
| **VIX** | CBOE Volatility Index — market “fear” gauge. |
| **W-FRI** | Weekly data aligned to Friday. |

---

## Config parameters (quick reference)

| Parameter | Plain English |
| --- | --- |
| `models.m3.threshold` / `models.m2.threshold` | Cutoff T for binary/gated sizing on `p_success` (default 0.55) |
| `portfolio.transaction_cost_bps` | Trading cost assumption (default 5 bps) |
| `portfolio.vol_target_ann` | Target annual volatility (default 12%) |
| `labels.positive_threshold` | Min forward return to call a long trade “successful” |
| `models.m1.top_k` | How many ETFs to select each week |
| `split.train_end` / `split.test_start` | Where in-sample ends and OOS begins |

Full table: [final_report.md — Configuration Parameters](reports/final_report.md#configuration-parameters-affecting-performance).

---

## Report and file glossary

| File / section | What it contains |
| --- | --- |
| `final_report.md` | Main results: benchmarks, long-only vs long/short, M2 quality |
| `m1_factor_analysis.md` | Which factor families predict returns (IC, correlations) |
| `m2_diagnostics.md` | Classifier metrics, calibration, AUC guide |
| `m3_allocation_analysis.md` | How often M3 rejects vs activates trades |
| `m3_threshold_analysis.md` | Sweep of binary/linear thresholds vs Sharpe |
| `evaluation_analysis.md` | Walk-forward folds, transaction-cost stress tests |
| `walk_forward_analysis.md` | Is ECDF edge stable across time windows? |
| `market_regime_analysis.md` | Performance conditioned on volatility/macro regimes |
| `BRANCH_UPDATE_REPORT.md` | Executive summary vs `main` branch |

---

## Mental model (one paragraph)

Each **Friday**, the pipeline scores seven ETFs with **M1**, picks candidates (e.g. top 3 longs), asks **M2** “how likely is each trade to work?”, converts that to a bet size with **M3**, applies **portfolio** risk limits and **vol targeting**, and simulates what would have happened historically. **Equal-weight** and **60/40** are simple benchmarks. **Sharpe** and **drawdown** tell you whether the strategy improved the return-versus-pain tradeoff — not just raw return.

---

*Research use only — not investment advice.*
