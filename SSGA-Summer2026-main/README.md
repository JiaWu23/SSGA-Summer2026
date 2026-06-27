# Multi-Asset Meta-Labeling Research Pipeline

A weekly, multi-asset allocation framework using a two-stage meta-labeling design.
Research and educational use only — not investment advice.

## Overview

The pipeline allocates across a seven-sleeve global asset universe and separates the
allocation decision into two stages plus a portfolio layer:

- **M1 — static directional model.** A simple, linear, rule-based signal that ranks assets each week using fixed-weight price factors only. No learning, no regime switching, no dynamic adjustment. Deliberately kept lean so M2 has meaningful signal to evalute
- **M2 — dynamic meta-label.** Asks "given the current market regime, how
much should I trust M1's signal?" Sizes M1's active tilt up or down based on M1's
recent track record and regime context. All dynamic logic lives here.
- **Portfolio.** Benchmark-relative active weights (benchmark ± bounded tilt),
  volatility targeting, position caps, and a two-layer cost model (expense ratio +
  transaction cost). The headline metric is the information ratio.

Design principle: **static factors live in M1, dynamic factors live in M2.**


## M1 — Static Directional Model

M1 is a pure cross-sectional ranker. Every week it scores each asset using four
static price factors, blends them with fixed weights, and tilts the portfolio toward
higher-scored assets. Nothing changes week to week except the input prices.

### M1 Factors (all static, all price-based, no learning)

| Factor | Weight | Definition |
|---|---:|---|
| **Technical** | 0.40 
| `0.5 × z(momentum) + 0.5 × z(trend)`. Momentum = avg pct_change over [12, 26] weeks. Trend = MA(10)/MA(40) − 1. Both cross-sectionally z-scored. |
| **Relative Momentum** | 0.30 | Each asset's 12-week return minus the equal-weight universe average. Captures strength relative to the group, not just in absolute terms. |
| **52-Week High Proximity** | 0.20 | Price / rolling 52-week high. Assets near their high have persistent price strength. |
| **Short-Term Reversal** | 0.10 | Negative of the 1-week return. Mild mean-reversion counterbalance to momentum. |

All factors are cross-sectionally z-scored so they live on a comparable scale.
Every input uses `shift(1)` — strictly no look-ahead.

### M1 Factor Weights vs Benchmark

The benchmark and the M1 factors are two separate things:

- **Benchmark** = equal weight (1/N), defined in config under `baselines`.
  Each asset gets 1/7 = 14.3% by default.
- **M1 factors** = signals that decide how to tilt AWAY from the benchmark.

The flow is:
Benchmark (equal weight 1/N)

+ M1 active tilt (driven by the 4 factors above)

= final portfolio weights


## Asset Universe

| Asset Class | ETF Proxy | Index (target) |
|---|---|---|
| U.S. Equity | SPY | S&P 500 Index |
| Developed Intl Equity | VEA | MSCI EAFE Index |
| Emerging Markets Equity | VWO | MSCI Emerging Markets Index |
| U.S. Treasury | TLT | S&P U.S. Treasury Bond 7-10yr Index |
| High Yield Bond | HYG | ICE BofA U.S. High Yield (total return) |
| Gold | GLD | Gold Spot Price |
| Real Estate | VNQ | Nasdaq U.S. Benchmark REIT Index |

Note: HYG OAS (credit spread) is kept as an M2 regime feature only — it is a spread
series, not total return, so it cannot be used directly as an M1 price signal.


## Usage

```bash
pip install -r requirements.txt      # pandas, numpy, scikit-learn, yfinance, pyyaml, pyarrow
python fetch_indices.py              # download index/proxy series into data/raw/index/
python run_all.py                    # strategy + attribution + walk-forward -> reports/
```

Individual stages:
```bash
python run_m1.py            # M1 allocation for the latest week
python run_strategy.py      # full M1 / M1+M2 backtest + M2 evaluation suite
python run_attribution.py   # factor and cost attribution
python run_walkforward.py   # walk-forward validation across windows
python -m pytest tests/     # correctness tests (no look-ahead, embargo, constraints)
```

Configuration lives in `config/config.yaml`.


## Repository layout

| Path | Contents |
|---|---|
| `src/data.py` | market + macro ingestion (yfinance, FRED), weekly resampling, index loader |
| `src/features.py` | no-look-ahead factors: momentum, trend, relative momentum, 52w high proximity, reversal, regime |
| `src/m1.py` | static linear directional model |
| `src/m2.py` | dynamic regime-aware meta-label (rolling logistic, embargo) |
| `src/portfolio.py` | benchmark-relative weights, vol targeting, two-layer costs |
| `src/backtest.py` | returns, Sharpe, drawdown, information ratio, baselines |
| `src/evaluation.py` | M2 classifier metrics (F1, AUC-ROC, AUC-PR, calibration) |
| `config/config.yaml` | all parameters |
| `FACTORS.md` | detailed factor definitions and worked example |
| `DATA_SOURCES.md` | index vs ETF mapping and data source decisions |

## Latest Results

Full sample 2000–2026, out-of-sample (OOS) from 2021.

| Strategy | Sharpe | Max DD | Sharpe (OOS) | Info Ratio (OOS) |
|---|---:|---:|---:|---:|
| Equal-Weight | 0.57 | -39% | 0.71 | — |
| Moderate Growth | 0.56 | -40% | 0.63 | -0.38 |
| Institutional | 0.61 | -35% | 0.59 | -0.80 |
| **M1-only** | **0.56** | **-32%** | **0.70** | **-0.11** |
| M1 + M2 | 0.55 | -32% | 0.72 | -0.08 |

M1 reduces max drawdown meaningfully (-32% vs -39% equal-weight). IR is negative
because M1 slightly underperforms equal-weight on return while taking active risk —
this is the open research question for M1 factor improvement and window tuning.

---

## Methodology Notes

### No Look-Ahead and Information Leakage

Information leakage is one of the most critical issues in financial ML.
A model that accidentally sees future information during training will
look great in backtests but fail completely in live trading. Here is
exactly how we prevent leakage at every stage:

**M1 — feature construction:**
- All rolling features use `.shift(1)` — M1 only sees last week's prices,
  never the current week
- Momentum: `prices.pct_change(window).shift(1)` — return up to last Friday
- Trend: `MA_short / MA_long - 1`, both computed on `.shift(1)` prices
- Relative momentum, 52w high proximity, reversal — all shifted by 1 week
- M1 never sees the return it is trying to predict

**M2 — evaluation data from M1:**
- Rolling hit rate: `labels.rolling(12).mean().shift(1)` — M2 only sees
  whether M1's *past* bets paid off, never the current week's outcome
- Rolling IR: computed on past active returns, shifted by 1 week
- Signal strength: M1's score from *last* week, not the current week
- Regime features (VIX, yield curve, NFCI): all use last available
  observation, macro data lagged 4 weeks for publication delay

**M2 — training and label embargo:**
- Meta-label for week t = did M1's bet pay off over weeks t+1 to t+4
- This means the label for week t is NOT observable until week t+4
- We enforce a **4-week embargo**: when predicting at week t, M2 only
  trains on labels that resolved at least 4 weeks before t
- This prevents M2 from training on labels that haven't happened yet
- In code: `cutoff = unique_dates[max(0, i - horizon - embargo)]`

**Walk-forward validation:**
- Train through 2020, test from 2021 onward — strict temporal split
- M2 refits every 4 weeks using only past data
- No hyperparameter tuning on the test set

**Summary — three layers of leakage prevention:**

| Layer | Method | Where in code |
|---|---|---|
| Feature leakage | `.shift(1)` on all rolling features | `src/features.py` |
| Label leakage | 4-week embargo between signal and label | `src/m2.py` `run_rolling()` |
| Validation leakage | strict train/test temporal split | `config/config.yaml` `split` section |
---


## Open questions / discussion

- **M1 window testing** — compare momentum windows [12, 26] vs [24, 52] on IR
- **M2 implementation** — build rolling M1 hit rate feature; add NFCI, STLFSI4,
  bond-equity correlation as regime inputs
- **Index data** — wire MSCI EAFE, MSCI EM, Treasury 7-10Y index series for
  longer pre-2007 history
- **Universe** — evaluate adding investment-grade credit (LQD) and broad commodity (DBC)

## Limitations

- Research-grade data (yfinance, FRED, ETF proxies); free index history is limited,
  so a true long-history index study requires institutional data.
- Historical simulation only — no capacity, market impact, borrow, or live execution.
- Some diagnostics are full-sample; production validation would extend the
  walk-forward and purged cross-validation.
