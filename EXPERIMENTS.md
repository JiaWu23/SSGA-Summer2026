# Week 7 — Experiments Log

All experiments are ranked on **walk-forward** performance (6 expanding folds, 2015–2026),
not in-sample. Default config is unchanged; every experiment is opt-in. Scripts:
`scripts/macro_sweep.py`, `scripts/m1_weight_sweep.py`. Outputs under `runs/`.

## 1 · Baseline reproduction
- **Question:** Does our environment reproduce the committed results exactly?
- **Setup:** Default config, 6-fold walk-forward.
- **Result:** Reproduced **bit-for-bit**. M1-only walk-forward Sharpe **0.709**; ECDF edge vs
  M1 **−0.19** (positive in 1 of 6 folds); IR vs equal-weight **−0.63**. Confirms the older
  report's **+0.177 / 4-of-6** figure was a stale ETF-era number, not the index-era result.

## 2 · Macro sweep v1 — discarded (methodology bug)
- **Question:** Which set of macro series maximizes Sharpe?
- **Setup:** Vary `macro.fred_series` (the series actually downloaded), 9 variants.
- **Result:** **Invalid.** Different series shifted the sample window (a confound), so the
  Sharpe differences were not attributable to the macro signal. The tell-tale sign was
  `macro_off` coming out byte-identical to `baseline`. Discarded — this motivated the fix below.

## 3 · Macro sweep v2 — controlled (macro in both M1 and M2)
- **Question:** With the sample held fixed, does macro actually help?
- **Setup:** Added `MacroConfig.model_series` — all 7 series are always downloaded (sample
  identical for every variant); excluded macro signals are **zeroed, not dropped**. 9 macro sets.
- **Result:** **Macro barely moves M1.** The best set (growth+inflation, 0.714) beats macro
  entirely OFF (0.701) by only **+0.013 Sharpe** (paired t ≈ 0.19 — noise). Rates/credit signals
  actively **hurt** M1 (0.668). The ECDF edge is negative in every variant.

## 4 · Macro sweep v3 — clean separation (macro only in M2)
- **Question:** Does macro help the meta-label once M1 is a pure technical model?
- **Setup:** `--m1-macro-free` sets M1's macro weight to 0 (weights mom 0.56 / trend 0.31 /
  macro 0 / risk 0.13), so M1-only is fixed at 0.678 and macro affects only M2. 9 macro sets.
- **Result:** **Two sets flip positive.** `drop_lagging_cpi` (all macro except CPI): ECDF edge
  **+0.067**, M2 AUC 0.525; `parsimonious`: +0.009. The clean-separation ECDF ≈ **0.745**, which
  exceeds the original macro-in-M1 M1-only of **0.709** — the first configuration where the
  meta-label adds value. Still small (2 of 6 folds); needs exposure-matching + Deflated Sharpe.

## 5 · M1 factor-weight sweep (Axis A)
- **Question:** Were the 45/25/20/10 weights ever justified? Which weights are best?
- **Setup:** 8 weight combinations, ranked on walk-forward Sharpe.

| M1 weights (mom / trend / macro / risk) | Walk-fwd Sharpe | Folds positive |
| --- | ---: | ---: |
| **risk_heavy 0.40 / 0.22 / 0.18 / 0.20** | **0.811** | **6 / 6** |
| baseline 0.45 / 0.25 / 0.20 / 0.10 | 0.709 | 4 / 6 |
| no_macro 0.56 / 0.31 / 0.00 / 0.13 | 0.690 | 4 / 6 |
| technical 0.35 / 0.35 / 0.20 / 0.10 | 0.672 | 4 / 6 |
| trend_only 0.00 / 0.70 / 0.20 / 0.10 | 0.521 | 5 / 6 |
| momentum_only 0.70 / 0.00 / 0.20 / 0.10 | 0.500 | 4 / 6 |

- **Result:** A heavier downside/risk-penalty weight lifts Sharpe **0.709 → 0.811, positive in
  all 6 folds** — the most robust improvement found. Either momentum or trend **alone** (~0.50)
  is far worse than both together (0.71).

## 6 · Momentum/trend collinearity
- **Question:** Are momentum and trend redundant?
- **Setup:** Correlation of the M1 component scores on the model panel (n = 5523).
- **Result:** corr(momentum, trend) = **0.78**; macro is nearly orthogonal to both (≈ 0.09).
  So they are correlated — but not redundant (Experiment 5 shows dropping either one hurts).

## 7 · Pure single-factor sweep (isolating momentum / trend / macro / risk)

- **Question:** Which single factor performs best when isolated (100% weight on
  one factor, 0% on the other three)? Hypothesis stated before viewing results:
  risk would perform best, based on the pattern observed in Experiment 5 (higher
  risk-penalty weight tracked with higher walk-forward Sharpe); momentum and
  trend were expected to place in the middle (they carry the most weight in the
  baseline config); macro was expected to be weakest, consistent with
  `macro_heavy` underperforming in Experiment 5.
- **Setup:** 4 pure single-factor configs — `momentum_100` (1/0/0/0),
  `trend_100` (0/1/0/0), `macro_100` (0/0/1/0), `risk_100` (0/0/0/1) — same
  walk-forward methodology as Experiment 5.

| Factor (100% weight) | Walk-fwd Sharpe | Folds positive | Ann. return |
| --- | ---: | ---: | ---: |
| **risk_100** | **0.922** | 5 / 6 | 5.70% |
| trend_100 | 0.481 | 5 / 6 | 5.29% |
| momentum_100 | 0.446 | 4 / 6 | 5.15% |
| macro_100 | 0.237 | 4 / 6 | 2.35% |

- **Result:** Hypothesis largely confirmed. `risk_100` is the strongest single
  factor (0.922), exceeding even the mixed `risk_heavy` config from Experiment
  5 (0.811); `macro_100` is the weakest (0.237), consistent with `macro_heavy`
  underperforming earlier. One deviation from the hypothesis: momentum
  (0.446) placed slightly below trend (0.481) rather than above it — in this
  7-asset universe and sample period, the trend signal alone edges out
  momentum alone.
- **Caveat — read before citing this result:** `risk_100`'s high Sharpe comes
  primarily from **low volatility, not high return** — its annualized return
  (5.70%) is the second-lowest of the four. This should not be read as "the
  risk factor alone picks the highest-returning assets"; it picks the
  *steadiest* ones. This nuance matters if presenting this result as evidence
  for reweighting M1.


---

**Summary:** 6 experiments — 1 reproduction, 3 macro sweeps (one discarded via a methodology
fix), 1 weight sweep, 1 correlation check. Headline wins: **risk-heavy M1 weights (0.811, 6/6
folds)** and **clean-separation macro-in-M2 (first positive meta-label edge)**.
