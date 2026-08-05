# Part 2 Speaker Script — M1 + M2: Selection & Meta-Labeling

Source deck: `Final_Presentation_Full_Deck_aug5.pptx`, slides 8–15  
Target delivery: 8:00 at roughly 140–145 words per minute  
Speaking text: about 1,145 words

## Slide 8 — Part 2: M1 + M2 [0:00–0:25]

“Part 2 covers the two decision layers at the front of the pipeline. M1 is the transparent selector: each week, it scores seven sleeves and proposes the top three. M2 is the learned meta-label: it cannot propose a new trade; it estimates the probability that an M1 proposal succeeds. I’ll unpack M1, show the evidence behind its weights, and then trace how M2 produces one calibrated probability.”

## Slide 9 — Four Components, Four Exact Formulas [0:25–1:50]

“M1 combines four components: momentum at 45 percent, trend at 25 percent, macro at 20 percent, and a risk penalty at 10 percent. The first three are added; the risk term is subtracted.

Momentum has three equally weighted parts—not five equally weighted features. The first part averages the cross-sectional z-scores of 12-, 26-, and 52-week momentum. The second is the 12-week percentile rank, centered around zero. The third is 12-week performance relative to the equal-weight basket. We then average those three parts. So each horizon inside the first part contributes roughly one ninth of the momentum score, while rank and relative momentum each contribute one third.

Trend is simpler: the 10-week moving-average level relative to the 40-week average, shifted to prevent look-ahead and then z-scored across sleeves.

The risk penalty is 12-week standardized volatility plus two drawdown terms. The terms are summed, not averaged, and the resulting penalty is subtracted from the alpha score. In plain English, the model prefers a strong signal that is not accompanied by excessive volatility or drawdown damage.

Macro is the only asset-class-specific component. The same regime flags are interpreted differently: for example, risk-off hurts equities but can support bonds and gold. The result is one auditable M1 score, used to select the weekly candidates.”

## Slide 10 — Why 12, 26, and 52 Weeks? [1:50–2:40]

“Why use three momentum windows? Because they answer different questions. Twelve weeks is the recent pulse. It reacts fastest to a change in market opinion, including policy repricing or a sudden risk-on or risk-off move. Twenty-six weeks bridges short-term noise and a persistent half-year regime. Fifty-two weeks tests whether leadership survived a full annual and seasonal cycle.

There is an important distinction here. In raw accumulated return, the 52-week window has the greatest dispersion because it compounds more shocks; then 26 weeks; then 12. But in week-to-week responsiveness, the order reverses: 12 weeks is most reactive. Combining all three reduces the chance that M1 either chases a short burst or waits too long to recognize a genuine change.”

## Slide 11 — Weight-Sweep Evidence [2:40–3:45]

“The next question is whether 45–25–20–10 is actually the best weighting. We tested eight configurations using six expanding walk-forward folds from 2015 through 2026.

The strongest candidate was risk-heavy: 40 percent momentum, 22 trend, 18 macro, and 20 risk. Its mean walk-forward Sharpe was 0.811 versus 0.709 for the current baseline. It produced a positive Sharpe in all six folds, compared with four of six for baseline, and it actually beat baseline in every fold.

That is meaningful evidence, but it is not proof of a global optimum. This is one sweep over a small menu, and we have already seen a trend-heavy result look good on one holdout and fail broader validation. Changing the production weights also changes every downstream M1 proposal and M2 training sample. So our defensible decision is to keep 45–25–20–10 as the established baseline today, while treating risk-heavy as the leading candidate for the next controlled configuration change.”

## Slide 12 — Single-Factor Experiment [3:45–4:35]

“We then isolated each factor at 100 percent. Before looking at the results, we wrote down the hypothesis: risk would be strongest, momentum and trend would be in the middle, and macro would be weakest.

That hypothesis was largely confirmed. Risk-only led with a 0.922 walk-forward Sharpe and five positive folds. Trend was 0.481, momentum 0.446, and macro 0.237. The one deviation was that trend slightly beat momentum.

The caveat matters more than the ranking: risk-only earned 5.70 percent annualized, the second-lowest return of the four. Its Sharpe is high mainly because volatility is low. So this does not say risk alone finds the highest-return assets. It says the risk measure is effective at finding the steadiest sleeves.”

## Slide 13 — M2: 52 Features to One Probability [4:35–5:50]

“Once M1 proposes a trade, M2 asks a narrower question: what is the probability this trade beats its four-week, cost-adjusted success hurdle?

M2 follows five explicit steps. First, missing feature values are replaced with medians learned only from the training data. Second, every feature is standardized to mean zero and unit variance. Third, a class-balanced logistic regression combines the 52 standardized inputs into a linear score and maps it through the logistic function. We chose this model for transparency and regularization rather than a more complex black box. Fourth, three-fold sigmoid, or Platt, calibration adjusts the probability scale. Fifth, M2 outputs one value: `p_success`.

Calibration is easy to overlook but essential because M3 sizes positions using the magnitude of the probability, not only its rank. A raw score can order trades correctly while still being overconfident. Calibration is intended to make a value like 0.60 mean approximately a 60 percent success rate across similar observations. It is a population estimate, not a guarantee for an individual trade.”

## Slide 14 — Missing-Value Treatment [5:50–6:50]

“In the saved long-only training run, 23 of the 52 M2 columns had at least one missing value. Most came from rolling-feature warm-up periods. A 52-week momentum feature needs 52 prior weekly closes, and our one-week shift adds another no-look-ahead delay. Cross-sectional z-scores can also be undefined when all sleeves have the same value.

The model was fitted on 1,500 eligible training rows. The largest gaps were 159 for 52-week momentum and its z-score, 120 for trend and standardized trend, and 81 for several 26-week risk and correlation features.

Each column receives its own training median—not zero and not a test-period value. The fitted medians are reused at prediction time. Labels, future returns, probabilities, and raw prices are not M2 inputs and are never imputed. This preserves early usable rows without manufacturing outcomes.”

## Slide 15 — Feature Groups and Structural Gating [6:50–8:00]

“The 52 inputs fall into five groups: 22 price and technical features, 14 macro and carry features, six M1 context variables, five dynamic interactions, and five asset-class indicators.

The interactions ask whether context changes the quality of M1’s call. They include M1’s rank, absolute score, and its score interacted with volatility, risk-off, and macro. Asset-class indicators allow different base success odds for bonds, credit, equities, gold, and REITs. Every input is standardized, so original measurement scale alone cannot dominate the regression.

The structural point is the most important. M2 is trained and scored only where M1 is nonzero, and portfolio weight multiplies the M1 signal by M3 size and base budget. If M1 is zero, the position is mathematically zero. M3 simply converts `p_success` into a binary, linear, or percentile-based size; it does not fit another model.

So the handoff to Part 3 is clean: M1 decides what is eligible, M2 estimates whether that proposal is likely to work, and M3 decides how much risk to allocate.”

## Delivery Notes

- Do not read the equations character by character. Point to each term and explain its job.
- Pause briefly after “beat baseline in every fold” and after the risk-only caveat.
- On Slide 13, emphasize that calibration makes probability magnitude usable; it does not make the forecast certain.
- If running long, shorten Slide 10 by omitting the examples of policy repricing and seasonal cycles.
- If running short, add: “Momentum and trend have a 0.78 component correlation, but dropping either weakened the mixed model, so correlated does not mean redundant.”
