# M2 Feature Enrichment Research

**Research use only — not investment advice.**

Goal: improve **M2 → M3** pipeline by enriching meta-label features with
**M1 factor analysis** (component ranks, IC alignment) and **dynamic external**
factors (VIX, macro regimes, cross-asset dispersion interactions).

**Test window:** `2021-01-01` onward

## Steps taken

1. **Baseline comparison** — `legacy_global` (40 features) vs `configured` (52, production).
2. **M1 factor enrichments** — cross-sectional component ranks, factor spread/sign agreement,
   trend−momentum spread, trend-heavy composite (from M1 IC analysis: trend IC 0.12 on test).
3. **External/regime enrichments** — interactions of M1/components with `risk_off`, VIX,
   yield curve, credit stress, inflation/growth flags, and dispersion features.
4. **IC alignment** — train-period factor IC weights × per-factor CS ranks (no look-ahead).
5. **Portfolio validation** — each variant refit on train; ECDF test Sharpe/return via M3.
6. **Walk-forward** — test AUC vs configured baseline across 6 expanding-window folds.
7. **Adoption gates** — test AUC ≥ baseline + 0.003 AND ECDF Sharpe ≥ 0.94; WF ≥ 4/6 fold wins.

## Adoption verdict

- **Verdict:** `reject`
- **Winner:** `None`
- **Reason:** Holdout passed but walk-forward failed (walk-forward AUC wins 2/6 folds, need 4)

## Holdout comparison (test period)

| variant | n_features | train_auc | test_auc | test_f1 | ecdf_test_sharpe | ecdf_test_ann_return | ecdf_test_max_drawdown | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m1_components_rich | 72 | 0.6404 | 0.5942 | 0.7434 | 1.0490 | 7.2338% | -8.5323% | Configured + M1 factor CS ranks, spread, sign agreement, conviction, trend−momentum. |
| ic_alignment | 74 | 0.6405 | 0.5941 | 0.7434 | 1.0484 | 7.2312% | -8.5454% | Configured + per-factor CS ranks + train IC-weighted alignment (no look-ahead). |
| trend_emphasis | 76 | 0.6437 | 0.5934 | 0.7434 | 1.0402 | 7.3309% | -9.3436% | Configured + trend-heavy composite (M1 IC analysis: trend strongest on test). |
| configured | 52 | 0.6460 | 0.5890 | 0.7417 | 0.9641 | 7.0210% | -11.3317% | Current production: 52 features with M1 meta + asset class dummies. |
| regime_external_rich | 182 | 0.6589 | 0.5872 | 0.7422 | 1.0557 | 6.3543% | -7.1688% | Configured + VIX/macro/regime × M1/component interactions + dispersion overlays. |
| full_enriched | 204 | 0.6682 | 0.5838 | 0.7415 | 1.0674 | 6.6408% | -8.6708% | M1 component rich + regime external + train IC-weighted factor alignment score. |

## Walk-forward test AUC (top variants)

| variant | mean_test_auc | mean_auc_delta | positive_folds | n_folds |
| --- | --- | --- | --- | --- |
| configured | 0.5476 | 0.0000 | 0 | 6 |
| trend_emphasis | 0.5339 | -0.0138 | 2 | 6 |
| m1_components_rich | 0.5313 | -0.0163 | 2 | 6 |
| ic_alignment | 0.5313 | -0.0163 | 2 | 6 |

## Explainability

- **`legacy_global`:** 40 base factors only; no M1 meta or asset encoding (main baseline).
- **`configured`:** Current production: 52 features with M1 meta + asset class dummies.
- **`m1_components_rich`:** Configured + M1 factor CS ranks, spread, sign agreement, conviction, trend−momentum.
- **`regime_external_rich`:** Configured + VIX/macro/regime × M1/component interactions + dispersion overlays.
- **`full_enriched`:** M1 component rich + regime external + train IC-weighted factor alignment score.
- **`trend_emphasis`:** Configured + trend-heavy composite (M1 IC analysis: trend strongest on test).
- **`ic_alignment`:** Configured + per-factor CS ranks + train IC-weighted alignment (no look-ahead).

## Recommendation

Keep production **`configured`** M2 features unchanged.
Document trade-offs in [m2_diagnostics.md](m2_diagnostics.md).

Related: [m1_factor_analysis.md](m1_factor_analysis.md) · [m2_diagnostics.md](m2_diagnostics.md) · [TERMINOLOGY.md](../TERMINOLOGY.md)