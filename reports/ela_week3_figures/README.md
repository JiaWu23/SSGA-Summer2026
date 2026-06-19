# Ela — Week 3 Figures (clean-rebuild pipeline)

These four figures come from the clean-rebuild meta-labeling pipeline
(static linear M1 + dynamic logistic M2, benchmark-relative active weights,
2-layer costs). Full sample 2000–2026, OOS from 2021.

## Figures

| File | What it shows |
|------|---------------|
| `equity_curves.png` | Growth of $1 (log scale): Equal-Weight vs M1-only vs M1+M2. |
| `walk_forward.png` | Sharpe by chronological window ('14–'16, '16–'18, '18–'20, '21–now). |
| `m2_calibration.png` | M2 predicted P(success) vs realized. Flat line ≈ no signal yet. |
| `attribution.png` | M1 sub-factor Sharpe: Momentum 0.65 / Trend 0.62 / merged Technical 0.67. |
| `asset_class_returns.png` | Cumulative growth of the 3 asset classes (Equity / Fixed Income / Alternatives), with faded individual members. |
| `group_weights.png` | How M1 rotates active allocation across the 3 classes over time (dashed = equal-weight benchmark). |
| `asset_active_tilt.png` | Average over/underweight per asset vs the 1/N benchmark — M1 most underweights TLT (-3.0%), most overweights GLD (+1.7%). |

## Asset universe & grouping

7 assets in 3 classes: **Equity** = SPY (US), VEA (developed ex-US), VWO (EM);
**Fixed Income** = TLT (Treasuries), HYG (high yield); **Alternatives** = GLD
(gold), VNQ (REITs). Equal-weight benchmark = 1/7 each (Equity 42.9%, FI 28.6%,
Alt 28.6%). Asset-group figures are produced by `make_asset_figures.py`.

## Headline numbers (full sample / OOS)

| Strategy | Sharpe | MaxDD | OOS Sharpe | OOS Info Ratio |
|----------|--------|-------|-----------|----------------|
| Equal-Weight | 0.619 | -29.9% | 0.764 | — |
| **M1-only** | **0.646** | **-21.8%** | **0.808** | **+0.179** |
| M1 + M2 | 0.605 | -25.2% | 0.777 | -0.004 |

**Read:** M1 adds risk-adjusted value (best Sharpe, lowest drawdown, the only
positive OOS info ratio). M2 does **not** add value yet — calibration is flat,
AUC ≈ 0.50 (coin flip). Making M2 carry signal is the open task. Note: results
still use free/proxy data (ETF proxies + short macro history), not Bloomberg
index data.

## M1 — design

M1 is intentionally simple: **static, linear, no learning.**

1. **Momentum + Trend** are collinear, so they are merged into one **"technical"**
   group (`0.5·momentum + 0.5·trend`). Keeping collinear regressors separate
   makes attribution impossible.
2. Scores are **cross-sectionally z-scored** so assets are comparable.
3. **Benchmark-relative active weights**: benchmark = equal weight (1/N). Highest
   score → positive tilt, lowest → negative tilt (synthetic underweight, no real
   shorting). Weights clipped + renormalized, fully invested (gross = 1).
4. Macro and risk/vol are **removed from M1** (they are time-varying = dynamic →
   they belong in M2).

## M2 — design

M2 is the **dynamic, regime-aware** meta-labeling layer. It asks: *"given the
current regime, how much do I trust M1's view here?"*

1. **Meta-label**: did M1's active bet pay on a benchmark-relative basis?
   (overweight beat the basket, or underweight lagged it → success).
2. **Features**: momentum, trend, macro, vol — kept **separate** (un-merged) —
   plus regime features (VIX, yield curve, credit spread, growth, inflation).
   Separation is what lets M2 do dynamic factor-timing.
3. **Logistic regression**, rolling ~12-month window, periodic refit, with strict
   no-look-ahead + embargo so the 4-week label cannot leak.
4. **Output**: P(success) → a sizing multiplier in [0, 1].
