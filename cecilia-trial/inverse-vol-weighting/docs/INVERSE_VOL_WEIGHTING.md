# Inverse-Volatility Weighting (opt-in)

Author: Cecilia (Anh-Chi Pham)

## What this adds

A new optional capital weighting scheme for the weekly picks.

Baseline behavior (unchanged, default): each selected asset receives the same
capital budget (`base_budget_per_asset` = 1/7).

New opt-in behavior (`portfolio.weighting_scheme: inverse_vol`): each week, the
same total capital is deployed, but it is split across the picks in proportion
to 1 / trailing 26-week volatility. Calmer assets receive more capital, jumpier
assets receive less, so each pick contributes a more similar amount of RISK
instead of a similar amount of dollars.

This is different from the conviction sizing that was tested and rejected
earlier: conviction sizing scaled positions by SIGNAL strength; this scales by
RISK. Reference: Maillard, Roncalli & Teiletche (2010), "The Properties of
Equally Weighted Risk Contribution Portfolios," Journal of Portfolio Management.

## Design rules respected

- No look-ahead: the budget uses the existing `vol_26w` feature column, which
  is already lagged one week in `feature_engineering.py` (`.shift(1)`).
- Same total deployment: weekly deployed budget equals the equal scheme
  (n_picks x base_budget), so any performance difference comes purely from the
  redistribution, not from taking more exposure.
- Baseline untouched: with `weighting_scheme: equal` (default), the function
  returns the plain float and the original code path is bit-for-bit identical.
  Verified: re-running `config/config.yaml` after this change reproduces the
  committed `metrics_table.csv` with max absolute difference 0.0.
- Robustness: volatility is floored at `inv_vol_floor_ann` (default 2%
  annualized) to guard division blow-ups; weeks with missing vol data fall back
  to equal shares.

## How to run the A/B

```
python -m src.run_pipeline --config config/config.yaml              # equal (baseline)
python -m src.run_pipeline --config config/config_inverse_vol.yaml  # inverse-vol
```

## Results (long-only)

Full sample:

| strategy      | equal: ann.ret / Sharpe / maxDD | inverse-vol: ann.ret / Sharpe / maxDD |
|---------------|---------------------------------|----------------------------------------|
| m1_only       | 7.32% / 0.70 / -21.0%           | 7.67% / 0.76 / -20.1%                  |
| m1_m2_binary  | 7.16% / 0.69 / -23.5%           | 7.58% / 0.75 / -20.1%                  |
| m1_m2_linear  | 1.80% / 0.84 / -5.4%            | 1.84% / 0.91 / -4.3%                   |
| m1_m2_ecdf    | 6.51% / 0.91 / -18.8%           | 6.83% / 1.00 / -14.9%                  |

Test period only (2021+), i.e. out-of-sample:

| strategy      | equal: ann.ret / Sharpe / maxDD | inverse-vol: ann.ret / Sharpe / maxDD |
|---------------|---------------------------------|----------------------------------------|
| m1_only       | 8.40% / 0.81 / -21.0%           | 8.86% / 0.86 / -20.1%                  |
| m1_m2_ecdf    | 6.93% / 0.86 / -16.3%           | 7.26% / 0.91 / -14.9%                  |

Headline: inverse-vol weighting improves annualized return, Sharpe ratio, and
maximum drawdown simultaneously for every strategy variant, both full-sample
and in the held-out test period. The largest gain is on the flagship
m1_m2_ecdf variant: full-sample Sharpe 0.91 -> 1.00 and max drawdown
-18.8% -> -14.9%.

Interpretation: with equal budgets, the jumpiest pick (often VNQ or VWO)
dominates the portfolio's realized risk. Redistributing toward calmer picks
keeps the same opportunities while reducing concentration of risk, which both
smooths the path (Sharpe, drawdown) and slightly helps compounding (return).

Shipped opt-in / default off, consistent with the convention that changes to
strategy behavior require team review before becoming the default.

## Files

- Modified additively: `src/config.py` (2 new fields), `src/portfolio.py`
  (new `compute_budgets`), `src/backtest.py` (route budgets through it).
- New: `config/config_inverse_vol.yaml`, `tests/test_inverse_vol_weighting.py`
  (6 tests), this document.
- All 47 tests pass (41 pre-existing + 6 new).
