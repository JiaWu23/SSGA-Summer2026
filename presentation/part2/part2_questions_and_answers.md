# Part 2 Question Bank — M1 + M2: Selection & Meta-Labeling

Source: `Final_Presentation_Full_Deck_aug5.pptx`, slides 8–15, cross-checked against the project configuration, implementation, and saved experiment outputs.

## Fast Opening Questions

### 1. What is the one-sentence summary of Part 2?

M1 transparently selects candidate sleeves; M2 estimates the probability that each M1 proposal will beat a four-week cost hurdle, but cannot create a trade itself.

### 2. What exactly does M1 output?

It outputs an `M1_score` and an `M1_signal`. In the production long-only, top-K setup, the three highest-scoring sleeves receive a signal of 1 each week and the others receive 0.

### 3. What exactly does M2 output?

One calibrated number, `p_success`, interpreted as the estimated probability that the proposed M1 trade earns more than the label’s transaction-cost hurdle over the next four weeks.

### 4. Why separate M1 and M2?

The separation preserves accountability. M1 answers “what should be considered?” using a transparent economic rule. M2 answers “how likely is that proposal to work?” using learned historical relationships.

### 5. Can M2 create a trade that M1 did not select?

No. M2 is trained and scored only on nonzero M1 rows, and the portfolio formula multiplies by `M1_signal`. If that signal is zero, the position is zero regardless of `p_success` or M3 size.

## M1 Formula and Selection

### 6. What is the exact baseline M1 formula?

`M1_score = 0.45 × momentum_score + 0.25 × trend_score + 0.20 × macro_score − 0.10 × risk_penalty`.

### 7. Why do the four weights sum to 100% if risk is subtracted?

The numbers represent the intended relative emphasis of the four components. Risk receives a positive 10% magnitude in the configuration, and the formula applies it with a negative sign because it is a penalty.

### 8. Are the five momentum-related quantities averaged equally?

No. The three z-scored horizons are first averaged into one part. That part, the centered 12-week rank, and 12-week relative momentum are then averaged. Each horizon therefore contributes roughly one ninth of `momentum_score`, while rank and relative momentum each contribute roughly one third.

### 9. Why include both a z-score and a rank?

They capture different information. A z-score reflects the magnitude of separation from the cross-sectional mean; a rank is more robust to outliers and captures ordering.

### 10. What does `rel_mom_12w` measure?

It measures a sleeve’s 12-week momentum relative to the equal-weight basket, so M1 can distinguish broad market movement from sleeve-specific leadership.

### 11. How is the trend signal calculated?

It is the 10-week moving average divided by the 40-week moving average minus one, shifted by one week to avoid using contemporaneous information, and then standardized across sleeves.

### 12. Is the trend component a machine-learning model?

No. It is a deterministic moving-average signal followed by cross-sectional standardization.

### 13. What is in the risk penalty?

Standardized 12-week volatility plus two drawdown terms: the absolute magnitude of a negative 26-week drawdown and the positive part of standardized 26-week drawdown. The terms are summed, and the total is subtracted from M1’s score.

### 14. Are the risk terms averaged or summed?

Summed. This was checked directly against `src/model_m1.py`.

### 15. Is there a sign issue in the standardized drawdown term?

This is a fair technical challenge. Raw drawdown is zero or negative, so a positive cross-sectional z-score normally means a shallower, not deeper, drawdown. The implemented `max(0, z_drawdown_26w)` therefore does not straightforwardly mean “worse than peers.” The absolute drawdown term still penalizes damage, but the relative term’s economic direction should be reviewed in a follow-up sensitivity test. Do not claim that this second term unambiguously penalizes the worst relative drawdowns.

### 16. What macro variables affect M1?

M1 uses five derived regime flags: growth trend, risk-off, inflation-up, curve inversion, and credit stress. Their signs and weights depend on asset class.

### 17. What are the exact asset-class macro tilts?

- Equity: `0.35 growth − 0.35 risk_off − 0.15 inflation_up`
- REIT: `0.25 growth − 0.30 risk_off − 0.25 inflation_up`
- Bond: `0.40 risk_off + 0.30 curve_inverted − 0.30 inflation_up`
- Credit: `0.30 growth − 0.40 credit_stress − 0.20 risk_off`
- Gold: `0.40 inflation_up + 0.35 risk_off − 0.15 growth`

### 18. Why can the same macro state affect sleeves differently?

Because the economic transmission differs. Risk-off may hurt equities and credit while supporting safe-haven bonds or gold; rising inflation can help gold but hurt nominal bonds and rate-sensitive REITs.

### 19. Is VIX treated like the monthly macro series?

No. VIX is public, real-time market data and follows the price-data path, with its engineered features shifted one week. Monthly FRED inputs receive the separate four-week publication lag.

### 20. Does M1 allow short positions?

The framework supports shorts, but the production results discussed in Part 2 are the long-only run. M1 selects the top three sleeves and does not open short positions in that run.

### 21. Why top three?

Top three provides a simple, stable breadth rule within a seven-sleeve universe. It limits concentration while preserving selectivity. It is a configured design choice, not a universally optimal number.

## Momentum Windows

### 22. Why 12, 26, and 52 weeks?

They represent a recent pulse, a half-year regime bridge, and a full-year trend. Combining them balances responsiveness and persistence.

### 23. Why does M2 also include four-week momentum, but M1 does not?

M1 deliberately uses a slower, auditable selector. M2 receives four-week momentum as extra short-horizon context when judging whether an already-proposed trade is likely to work.

### 24. Which horizon is most variable?

For raw accumulated momentum dispersion, 52 weeks is highest, followed by 26 and then 12. For week-to-week reactivity, 12 weeks is highest. “Variable” must be defined before answering.

### 25. Why z-score momentum cross-sectionally?

The goal is to compare sleeves at the same date. Cross-sectional standardization answers which assets are stronger or weaker relative to the opportunity set, rather than whether all assets rose together.

### 26. Doesn’t using three correlated horizons triple-count momentum?

There is overlap, but the horizons carry different timing information. The inner average also limits their combined influence: all three together form only one of the three momentum sub-parts.

### 27. Are momentum and trend redundant?

They are correlated—about 0.78 at the component level—but not redundant in the experiment. Removing either technical leg reduced walk-forward Sharpe relative to the mixed baseline.

## Weight-Sweep Evidence

### 28. How was the weight sweep evaluated?

Using six expanding-window, out-of-sample folds. The test blocks begin in 2015 and run in roughly two-year increments through the partial 2025–2026 fold. Variants were ranked on mean M1-only walk-forward Sharpe.

### 29. What does “6/6 folds positive” mean?

It means the risk-heavy M1 portfolio had a Sharpe above zero in every test fold. It does not merely mean six successful trades, and it should not be confused with statistical significance.

### 30. Did risk-heavy also beat baseline in every fold?

Yes. The saved fold files show a higher Sharpe than baseline in all six test folds, although some improvements were small.

### 31. What were the key mixed-weight results?

- Risk-heavy, `40/22/18/20`: Sharpe 0.811, 6/6 positive folds
- Baseline, `45/25/20/10`: 0.709, 4/6
- No macro, `56/31/0/13`: 0.690, 4/6
- Technical 50/50, `35/35/20/10`: 0.672, 4/6
- Momentum-plus-risk, `85/0/0/15`: 0.642, 4/6
- Macro-heavy, `35/20/35/10`: 0.638, 4/6
- Trend-only technical, `0/70/20/10`: 0.521, 5/6
- Momentum-only technical, `70/0/20/10`: 0.500, 4/6

### 32. Why keep baseline if risk-heavy is better?

Because the sweep is one experiment across a limited menu and has not yet received the same stress testing as the established configuration. Changing M1 also changes the candidate trades and M2’s training population. Risk-heavy is the leading candidate, not yet a production conclusion.

### 33. Isn’t keeping baseline arbitrary?

It is conservative, not evidence that baseline is optimal. The correct claim is that baseline is the current reference configuration and risk-heavy has stronger recent walk-forward evidence.

### 34. Was the 45/25/20/10 baseline originally optimized?

No. It began as an economic prior and later served as the repeatedly tested reference. The weight sweep was designed specifically to challenge it.

### 35. Did you correct for testing multiple weight combinations?

No formal multiple-testing adjustment or Deflated Sharpe Ratio is reported for this sweep. That is one reason not to declare the best observed configuration globally optimal.

### 36. Do you have confidence intervals or a significance test for the 0.102 Sharpe improvement?

Not in the deck. Six folds are informative but still a small sample. A bootstrap, Deflated Sharpe Ratio, or paired fold-level analysis would strengthen the claim.

### 37. Were transaction costs included?

Yes, the pipeline’s portfolio backtest uses the configured five-basis-point transaction-cost assumption. The meta-label itself uses a separate ten-basis-point success hurdle.

### 38. Why are the label hurdle and backtest cost different?

They serve different roles: the ten-basis-point label threshold defines a minimum four-week trade success, while the five-basis-point parameter charges simulated turnover. The mismatch is a design choice and should be included in sensitivity testing rather than treated as an identity.

### 39. Was the weight sweep testing M1 or the full M1–M2–M3 stack?

The ranking on Slide 11 is based on M1-only walk-forward Sharpe. M2 diagnostics were also produced, but they were not the criterion used to rank those M1 weights.

## Single-Factor Experiment

### 40. Why state the hypothesis before seeing the result?

It separates prediction from post-hoc storytelling. The recorded hypothesis was risk first, momentum and trend in the middle, and macro last.

### 41. What were the pure-factor results?

- Risk-only: Sharpe 0.922, 5/6 positive folds, 5.70% annualized return
- Trend-only: 0.481, 5/6, 5.29%
- Momentum-only: 0.446, 4/6, 5.15%
- Macro-only: 0.237, 4/6, 2.35%

### 42. Why does risk-only have the highest Sharpe but only the second-lowest return?

Sharpe is return divided by volatility. Risk-only selected steadier sleeves and reduced volatility enough to lift the ratio without producing the highest return.

### 43. Does risk-only prove the other factors are unnecessary?

No. It optimizes a risk-adjusted ratio in this particular sample and universe. It has a lower return, one negative fold, and may concentrate the portfolio in defensive sleeves. It is evidence for stronger risk control, not proof that return-seeking signals should be discarded.

### 44. Why is risk-heavy positive in 6/6 folds while risk-only is positive in only 5/6?

The other components can diversify or rescue the risk signal in a difficult period. Risk-only has the higher mean Sharpe, but the mixed risk-heavy model is more consistent by the positive-fold count.

### 45. Why did trend beat momentum in the pure-factor test?

In this seven-sleeve universe and sample, the smoothed 10-versus-40-week relationship generalized slightly better than the isolated multi-horizon momentum score. The difference is small and should not be generalized beyond this design.

### 46. Why is macro weakest?

The macro tilt is a fixed, intuition-driven mapping, and the controlled experiments found only modest M1 benefit from macro. Monthly data is also slow relative to weekly price signals.

## Meta-Labels and M2 Training

### 47. What is the exact meta-label definition?

For a nonzero M1 signal, `trade_return = M1_signal × forward_return_4w`. The meta-label is 1 if that trade return exceeds 0.001, or 10 basis points, and 0 otherwise.

### 48. Why a four-week horizon?

It aligns the success label with a medium-short holding period inside a weekly process. It is a configured research choice and should be tested against alternative horizons.

### 49. What happens when M1 is flat?

The meta-label is undefined, the row is excluded from M2 training, and `p_success` remains undefined. There is no proposed trade to evaluate.

### 50. How many M2 observations were used?

The saved long-only split had 1,500 eligible training rows and 867 eligible test rows. These are sleeve-week trade proposals, not 1,500 independent calendar weeks.

### 51. Are those 1,500 rows independent?

Not fully. Sleeves on the same date share macro conditions, and overlapping four-week outcomes create temporal dependence. That is why time-based walk-forward validation is more credible than a random train/test split.

### 52. Why logistic regression?

It is transparent, regularized by default, relatively data-efficient, and produces a natural probability-like score. More complex tree and per-asset alternatives showed signs of overfitting in project benchmarks.

### 53. What does balanced class weighting do?

It gives the two label classes inverse-frequency weight during fitting so the more common class does not dominate the loss. It changes training emphasis; it does not make the observed base rate 50%.

### 54. What was the base success rate?

About 60.8% in the reported test set. That is why raw accuracy must be interpreted relative to the base rate.

### 55. What is `p_success` mathematically?

The logistic model forms `b0 + b1x1 + … + b52x52`, maps it through `1 / (1 + exp(−score))`, and then applies sigmoid calibration to the probability scale.

### 56. Does M2 predict returns?

No. It predicts a binary event: whether the M1 trade beats the specified four-week cost hurdle. It does not estimate return magnitude.

### 57. Why calibrate probabilities?

M3 uses probability magnitude for sizing. Calibration attempts to align predicted probabilities with realized frequencies, so a 0.60 forecast has a meaningful scale rather than being only a ranking score.

### 58. Does `p_success = 0.60` guarantee a 60% chance of success?

No. It means that, if calibration generalizes, observations with similar forecasts should succeed about 60% of the time on average. It is not certainty for a single trade.

### 59. How good is M2?

The deck reports test AUC of about 0.539 and Brier score about 0.239. Calibration is roughly usable, but ranking power is weak—only slightly above random—so the project does not claim a strong predictive edge.

### 60. Why can calibration be acceptable when AUC is weak?

AUC measures ordering: whether winners rank above losers. Brier score measures probability error. A model can estimate the overall success frequency reasonably while still doing a poor job separating individual winners from losers.

### 61. Is the three-fold calibration time-series aware?

Not strictly. The implementation uses scikit-learn’s standard three-fold sigmoid calibration inside the training sample, not an explicitly chronological calibration splitter. The final 2021+ test set remains untouched, but a purist time-series version should use forward-only calibration folds.

## Missing Values and Leakage Controls

### 62. Which values are imputed?

Feature values only. In the saved training rows, 23 of 52 columns had gaps, mostly rolling momentum, trend, volatility, drawdown, correlation, dispersion, and derived interaction features.

### 63. Why are 52-week features missing?

They require 52 prior weekly closes. The feature is then shifted one additional week to prevent look-ahead, extending the initial warm-up gap.

### 64. Why can a cross-sectional z-score be missing?

If every sleeve has the same value on a date, cross-sectional standard deviation is zero, so a z-score is undefined. A common example is all sleeves being at zero drawdown.

### 65. What were the largest missing counts?

In the saved 1,500-row training sample: 159 for `mom_52w` and `z_mom_52w`, 120 for trend and standardized trend, and 81 for several 26-week risk and correlation features.

### 66. How does median imputation work?

Each feature gets its own median learned from the training rows available to that fitted pipeline. The same fitted median is used later for prediction.

### 67. Why median rather than mean?

The median is less sensitive to outliers and skewed financial features. It is a simple defensive default rather than a claim that missingness contains no information.

### 68. Why not fill missing values with zero?

Zero has an economic meaning for many features and could falsely imply neutral momentum, volatility, or macro conditions. The training median is a less assertive central replacement.

### 69. Why not drop every row with a missing feature?

Dropping would disproportionately remove early history and shrink an already modest training sample. Median imputation preserves rows while keeping the procedure reproducible.

### 70. Does imputation introduce look-ahead?

Not into the held-out test period: medians are learned on training data and reused at prediction time. Within the calibration routine, each cloned pipeline fits preprocessing on its own calibration-training fold. The broader limitation is that the default calibration folds are not explicitly forward-chaining.

### 71. Are labels or future returns ever imputed?

No. Labels, forward returns, `p_success`, and raw prices are not M2 feature inputs and are not filled by this preprocessing step.

### 72. Could missingness itself carry information?

Yes. A missingness indicator could distinguish warm-up periods or zero-dispersion states. The current model does not add such indicators; testing them is a reasonable extension.

## The 52 Features

### 73. How do the 52 inputs break down?

- 22 price and technical
- 14 external macro and carry
- 6 M1 context variables
- 5 dynamic interactions
- 5 asset-class indicators

### 74. What are the six M1 context variables?

The four component scores—momentum, trend, macro, and risk penalty—plus `M1_signal` and `M1_score`.

### 75. What are the five dynamic interactions?

`m1_cs_rank`, `m1_score_abs`, `m1_x_vol`, `m1_x_risk_off`, and `m1_x_macro`.

### 76. What question do the interaction terms answer?

They ask whether M1’s historical success changes with call strength, cross-sectional rank, volatility, risk-off conditions, or macro context.

### 77. What are the five asset identities?

Bond, credit, equity, gold, and REIT. The seven sleeves map into these five economic classes.

### 78. Why use asset-class indicators rather than one model per sleeve?

Indicators let one global model learn class-level differences while sharing data across sleeves. Separate per-asset heads have far fewer observations and showed more overfitting risk.

### 79. Do zero-one dummy variables dominate continuous features?

Not merely because they are coded as zero and one: all features are standardized. They can still matter through learned coefficients, and rare categories can produce larger standardized values, so standardization controls units but does not guarantee low influence.

### 80. Doesn’t including all five dummies plus an intercept create collinearity?

Yes, the full one-hot set is linearly dependent with the intercept. Regularized logistic regression can still produce stable predictions, but dropping one reference class would make coefficients easier to interpret.

### 81. Are 52 features too many for 1,500 training rows?

It is a meaningful overfitting risk, especially with correlated features and dependent sleeve-week rows. Logistic regularization and walk-forward testing help, but feature ablation or stronger sparsity controls would be prudent.

### 82. Does M2 reuse information already used by M1?

Yes. It receives M1 components and some underlying regime features so it can learn when M1’s fixed recipe tends to work. The downside is duplication and collinearity.

### 83. Is this “true” meta-labeling?

Not in the strictest sense yet. M2 sees M1’s outputs and ingredients, but not a direct rolling history of M1’s errors or residuals. The deck identifies rolling hit-rate or residual features as the top next step.

### 84. Why not feed future return or the label into M2?

Those are outcomes used to train and evaluate the model. Including them as predictors would be direct target leakage.

### 85. Does M3 use any of the 52 features directly?

No. M3 sees only `p_success` and maps it to a position-size rule.

## Tough Interpretation Questions

### 86. What is the strongest claim supported by Part 2?

M1’s selection rule is transparent and appears improvable through greater risk emphasis; M2 is a controlled probability layer that cannot originate trades, but its predictive ranking is still weak.

### 87. What should we not claim?

Do not claim that 45/25/20/10 is optimal, that risk-only maximizes return, that a calibrated probability is certain, that M2 has strong discrimination, or that the architecture is already strict meta-labeling.

### 88. What is the biggest M1 follow-up?

Validate risk-heavy and risk-only with broader threshold sensitivity, multiple-testing-aware statistics, additional universes or histories, and a direct review of the standardized drawdown sign.

### 89. What is the biggest M2 follow-up?

Replace or augment raw M1 ingredients with true track-record variables such as rolling hit rate and residual error, then evaluate with forward-only calibration and walk-forward AUC and portfolio metrics.

### 90. If M2’s AUC is only 0.539, why keep it?

Because probability-based sizing can still reduce exposure and drawdown, and calibration can remain useful even when ranking is weak. But it should be described as a risk-throttling research layer, not a proven source of return.

### 91. Could the drawdown benefit be just de-leveraging?

Yes. Lower gross exposure mechanically lowers volatility and drawdown. A fair attribution requires exposure-matched or constant-haircut controls before assigning the benefit to predictive skill.

### 92. How do you know there is no look-ahead in price features?

Momentum, moving-average trend, volatility, drawdown, and correlation features are shifted by one week. The M2 label uses future returns only as the outcome, never as an input.

### 93. How do you know there is no look-ahead in macro data?

Monthly FRED data is aligned to the weekly grid and then lagged four weeks. Test and evaluation periods use forward-fill rather than future-blended interpolation, and VIX is treated separately as market data and shifted one week. One limitation is that the training period permits time interpolation before the four-week lag; a strict release-date audit should replace that with an as-of merge or forward-fill everywhere.

### 94. What would make you change the production M1 weights?

A preregistered follow-up in which risk-heavy remains superior across folds, threshold choices, realistic cost assumptions, and preferably another universe or longer independent period—without weakening downstream M2/M3 behavior.

### 95. What is the clean transition to Part 3?

“M1 determines eligibility and M2 supplies only `p_success`. Part 3 shows how M3 maps that probability into an actual bet size and how portfolio-level constraints are applied.”

## Numbers to Memorize

| Item | Number |
|---|---:|
| Baseline M1 weights | 45 / 25 / 20 / 10 |
| M1 selections per week | Top 3 of 7 sleeves |
| Momentum windows in M1 | 12 / 26 / 52 weeks |
| Momentum–trend component correlation | About 0.78 |
| Walk-forward folds | 6 |
| Baseline walk-forward Sharpe | 0.709 |
| Risk-heavy walk-forward Sharpe | 0.811 |
| Risk-heavy positive folds | 6 / 6 |
| Risk-only walk-forward Sharpe | 0.922 |
| Risk-only annualized return | 5.70% |
| M2 features | 52 |
| M2 train rows | 1,500 |
| M2 test rows | 867 |
| Feature columns with training NaNs | 23 / 52 |
| Largest 52-week missing count | 159 |
| Label horizon | 4 weeks |
| Meta-label hurdle | 0.10% / 10 bps |
| Binary sizing threshold | 0.55 |
| Test AUC | About 0.539 |
| Test Brier score | About 0.239 |
