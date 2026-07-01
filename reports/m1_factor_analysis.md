# M1 Factor Analysis

**Research use only — not investment advice.**

## Factor Weights

M1 composite score uses momentum **45%**, trend **25%**, macro **20%**, risk penalty **10%**.

## Per-Factor Information Coefficient

Spearman rank correlation of each component score vs 4-week forward return.

| period | factor | ic_mean | ic_std | ic_hit_rate | n_weeks |
| --- | --- | --- | --- | --- | --- |
| full | momentum_score | 0.0438 | 0.4876 | 0.5325 | 986 |
| full | trend_score | 0.0391 | 0.5039 | 0.5000 | 986 |
| full | macro_score | -0.0035 | 0.4567 | 0.4696 | 986 |
| full | risk_penalty | 0.0259 | 0.4517 | 0.4980 | 986 |
| full | M1_score | 0.0544 | 0.4290 | 0.5294 | 986 |
| train | momentum_score | 0.0318 | 0.5009 | 0.5221 | 701 |
| train | trend_score | 0.0042 | 0.5249 | 0.4622 | 701 |
| train | macro_score | 0.0096 | 0.4726 | 0.4893 | 701 |
| train | risk_penalty | 0.0194 | 0.4567 | 0.4922 | 701 |
| train | M1_score | 0.0337 | 0.4360 | 0.5093 | 701 |
| test | momentum_score | 0.0732 | 0.4528 | 0.5579 | 285 |
| test | trend_score | 0.1210 | 0.4406 | 0.5930 | 285 |
| test | macro_score | -0.0364 | 0.4133 | 0.4211 | 285 |
| test | risk_penalty | 0.0419 | 0.4396 | 0.5123 | 285 |
| test | M1_score | 0.1061 | 0.4071 | 0.5789 | 285 |

![Factor IC](../data/backtests/long_only/figures/m1_factor_ic.png)

## Factor Correlation Matrix

| factor | momentum_score | trend_score | macro_score | risk_penalty |
| --- | --- | --- | --- | --- |
| momentum_score | 1.0 | 0.7729 | 0.0566 | 0.1442 |
| trend_score | 0.7729 | 1.0 | 0.0407 | 0.068 |
| macro_score | 0.0566 | 0.0407 | 1.0 | 0.4475 |
| risk_penalty | 0.1442 | 0.068 | 0.4475 | 1.0 |

![Factor correlation](../data/backtests/long_only/figures/m1_factor_correlation_heatmap.png)

## Factor Covariance Matrix

| factor | momentum_score | trend_score | macro_score | risk_penalty |
| --- | --- | --- | --- | --- |
| momentum_score | 0.120129 | 0.241269 | 0.059976 | 0.040011 |
| trend_score | 0.241269 | 0.811243 | 0.111886 | 0.049028 |
| macro_score | 0.059976 | 0.111886 | 9.332746 | 1.094667 |
| risk_penalty | 0.040011 | 0.049028 | 1.094667 | 0.641275 |

## Factor Sleeve Backtests

Each row is a portfolio using only that factor family for top-K selection (risk penalty inverted).

| sleeve | annualized_return | annualized_volatility | sharpe | max_drawdown | rolling_12m_max_drawdown | turnover | annualized_turnover | hit_rate | interaction_excess_ann | combined_excess_ann | sum_standalone_excess_ann |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_m1 | 7.3198% | 0.10426328396287823 | 0.7021 | -21.0040% | -0.21003977682248687 | 0.09598374622952195 | 4.991154803935141 | 0.5882352941176471 | nan | nan | nan |
| momentum_score | 6.3889% | 0.09109130539601205 | 0.7014 | -22.9859% | -0.2298586658264773 | 0.1415511355791459 | 7.360659050115586 | 0.5801217038539553 | nan | nan | nan |
| trend_score | 5.9278% | 0.09500978494940132 | 0.6239 | -20.4921% | -0.20492128230007478 | 0.0727334545751878 | 3.782139637909766 | 0.5628803245436106 | nan | nan | nan |
| macro_score | 3.9420% | 0.07453143398102433 | 0.5289 | -16.0349% | -0.14757287217727255 | 0.09547596128479639 | 4.964749986809412 | 0.5365111561866126 | nan | nan | nan |
| risk_penalty | 3.7912% | 0.05496016113287806 | 0.6898 | -10.0161% | -0.10016070168620173 | 0.14817504363698755 | 7.705102269123353 | 0.46551724137931033 | nan | nan | nan |
| interaction | — | nan | — | — | nan | nan | nan | nan | 0.09357350006798582 | -0.00042615811935098336 | -0.0939996581873368 |

![Factor sleeves](../data/backtests/long_only/figures/m1_factor_sleeves_cumulative.png)

## Factor Ablation (zero one weight at a time)

| variant | annualized_return | annualized_volatility | sharpe | max_drawdown | rolling_12m_max_drawdown | turnover | annualized_turnover | hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_m1 | 7.3198% | 0.10426328396287823 | 0.7021 | -21.0040% | -0.21003977682248687 | 0.09598374622952195 | 4.991154803935141 | 0.5882352941176471 |
| ablate_momentum | 6.5050% | 0.10012209922881003 | 0.6497 | -21.5613% | -0.21561286482389153 | 0.08291160601671349 | 4.311403512869101 | 0.5831643002028397 |
| ablate_trend | 7.7928% | 0.09480828958789554 | 0.8220 | -21.7232% | -0.21723248023110975 | 0.15048207307685527 | 7.825067799996474 | 0.5963488843813387 |
| ablate_macro | 6.4507% | 0.09507144503145203 | 0.6785 | -21.0133% | -0.2101325937730114 | 0.08945231363731157 | 4.651520309140202 | 0.5750507099391481 |
| ablate_risk_penalty | 7.3412% | 0.10163519089115651 | 0.7223 | -20.2830% | -0.20282995973854834 | 0.09425142924051848 | 4.901074320506961 | 0.5821501014198783 |

## Interaction Term

Combined M1 excess minus sum of standalone factor sleeves: **9.3574%** annualized. Positive values suggest factors reinforce; negative suggests overlap.
