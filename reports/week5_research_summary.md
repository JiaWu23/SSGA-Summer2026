# Week 5 Research Summary — `vitaly_week5` vs `main`

**Branch:** `vitaly_week5` · **Baseline:** `main` · **July 2026**  
**Research use only — not investment advice.**

Short index of questions explored, directions taken, and what changed. Detail lives in linked companion reports.

**Also see:** [BRANCH_UPDATE_REPORT.md](../BRANCH_UPDATE_REPORT.md) · [branch_update_vitaly_week5.md](branch_update_vitaly_week5.md) · [TERMINOLOGY.md](../TERMINOLOGY.md)

---

## Why this branch

Make the M1 → M2 → M3 pipeline **reviewable and explainable** (Joubert framework), add **diagnostics**, and improve **M2 ranking + ECDF sizing** without changing the M1 rule engine, train/test split, or top-K allocator.

**Unchanged vs `main`:** M1 factor weights (45/25/20/10), train through 2020 / test from 2021, portfolio caps, 12% vol target, data sources.

---

## Headline outcomes (long-only, test 2021+)

| Area | `main` | Branch | Adopted? |
| --- | ---: | ---: | --- |
| M1-only Sharpe | 0.79 | 0.79 | — (unchanged) |
| M2 test AUC | ~0.573 | **0.589** | Yes (52 enriched features) |
| ECDF test Sharpe | 0.85 | **0.96** | Yes (via better `p_success` → M3) |
| ECDF max drawdown | −16.3% | **−11.3%** | Yes |
| Tests | 48 | **61+** | Yes |

---

## Research directions (questions → work → verdict)

### 1. Explainability & analytics

**Question:** Can we defend each layer with factor, regime, and allocation evidence?

**Done:** Six companion reports + Deep Diagnostics in `final_report.md` — M1 factor IC, M2 calibration/AUC, regime conditioning, M3 allocation states, walk-forward, transaction-cost sensitivity.

**Modules:** `factor_analysis.py`, `regime_analysis.py`, `m3_diagnostics.py`, extended `diagnostics.py`.

---

### 2. M3 formalization (Joubert)

**Question:** Is bet sizing a separate, explicit layer?

**Done:** `model_m3.py`, persisted `M3_size` / `allocation_state`, strategy keys `m1_m2_m3_*` (legacy aliases kept).

**Finding:** ECDF is the production risk-shaping layer; binary at T=0.55 is degenerate (recall ≈ 1 → equals M1-only).

---

### 3. M1 factor weight tuning

**Question:** Should IC-proportional weights beat fixed 45/25/20/10?

**Done:** Holdout grid + 6-fold walk-forward (`m1_weight_research.py`).

**Verdict:** **Rejected** — holdout looked better (+0.008 Sharpe) but walk-forward mean M1 Δ −0.035; **config unchanged**.

---

### 4. M2 ranking improvement

**Question:** Can richer M2 inputs improve `p_success`?

**Done:** 40 → 52 features (M1 components, CS rank, vol/risk/macro interactions, asset-class dummies). Architecture benchmark: trees/per-asset heads overfit (~0.48–0.50 test AUC).

**Verdict:** **Adopted** — test AUC **+0.016**; drives ECDF Sharpe lift on branch.

---

### 5. Extended evaluation (walk-forward + costs)

**Question:** Is ECDF edge stable OOS, not just 2021+?

**Done:** Expanding-window walk-forward (`walk_forward_research.py`, `evaluation.py`); TC grid 0–25 bps.

**Verdict:** **Stable (majority)** — mean ECDF Sharpe edge vs M1 **+0.177**, **4/6** folds positive; edge **+0.046** @ 25 bps on production window. Weak folds: 2015–16, 2025–26 partial.

---

### 6. M3 threshold sweep

**Question:** Can binary/linear M3 reject meaningfully at T=0.55?

**Done:** Sweep T ∈ [0.50, 0.70] (`m3_threshold_research.py`).

**Finding:** T=0.55 approves ~100% of candidates; T=0.60 gives meaningful rejection (63%) and strong holdout Sharpe (0.94) but sits on a sharp cliff.

**Verdict:** **Config unchanged** — ECDF still preferred; threshold report documents trade-offs only.

---

### 7. Information Ratio vs equal-weight

**Question:** Why does Info Ratio fall when M2/M3 are added, and can we fix it?

**Done:** IR attribution + 15 portfolio overlays (`ir_research.py`, `ir_interventions.py`).

**Finding:** IR drops because ECDF **under-invests** vs EW (~52% gross); Sharpe rises but weekly EW tracking worsens.

**Verdict:** **No overlay adopted** — best holdout `vol_bump` failed walk-forward (2/6 positive IR folds); `exposure_renorm` helps IR but kills Sharpe.

---

### 8. M2 feature enrichment (M1 + external factors)

**Question:** Can M1 factor analysis and regime/dispersion features improve M2 further?

**Done:** Seven variants including `m1_components_rich`, `regime_external_rich`, `ic_alignment` (`m2_feature_research.py`).

**Finding:** `m1_components_rich` best holdout — test AUC **0.594**, ECDF Sharpe **1.05** vs configured **0.589 / 0.96**; `full_enriched` (204 features) overfits.

**Verdict:** **Config unchanged** — walk-forward rejected (2/6 fold AUC wins). Production stays **52-feature `configured`**.

---

## Code & config changes (summary)

| Category | Change |
| --- | --- |
| **Architecture** | M1 → M2 → **M3** → portfolio; `M3_size` on panel |
| **M2 features** | 52 enriched inputs (in production) |
| **M2 `feature_variant`** | Hook added; default `configured` (research variants not promoted) |
| **Evaluation** | `walk_forward_enabled: true`; per-fold IR columns |
| **Research CLIs** | `walk_forward_research`, `m1_weight_research`, `m3_threshold_research`, `ir_research`, `m2_feature_research` |
| **Docs** | `TERMINOLOGY.md`, `BRANCH_UPDATE_REPORT.md`, 10+ reports under `reports/` |

---

## Report index

| Report | Topic |
| --- | --- |
| [m1_factor_analysis.md](m1_factor_analysis.md) | Factor IC, sleeves, weight tuning |
| [m2_diagnostics.md](m2_diagnostics.md) | AUC, calibration, feature importance |
| [m2_feature_research.md](m2_feature_research.md) | M2 enrichment sweep |
| [m3_allocation_analysis.md](m3_allocation_analysis.md) | M3 states and sizing modes |
| [m3_threshold_analysis.md](m3_threshold_analysis.md) | Threshold sweep |
| [market_regime_analysis.md](market_regime_analysis.md) | Regime performance |
| [evaluation_analysis.md](evaluation_analysis.md) | Walk-forward + TC sensitivity |
| [walk_forward_analysis.md](walk_forward_analysis.md) | ECDF edge stability Q&A |
| [ir_attribution_analysis.md](ir_attribution_analysis.md) | Why IR drops vs EW |
| [ir_improvement_research.md](ir_improvement_research.md) | IR overlay sweep |
| [final_report.md](final_report.md) | Full pipeline metrics |

---

## What we learned (one paragraph)

**M1** economics are solid and unchanged. **M2** adds modest ranking (AUC ~0.59); its main value is feeding **M3 ECDF**, which materially improves Sharpe and drawdown vs `main`. **Binary M3** at 0.55 and **IC-proportional M1 weights** look good on holdout but fail stability checks. **IR vs equal-weight** is an explicit trade-off: ECDF sizes down capital, so Sharpe and IR move in opposite directions. **Regime** matters (2015–16 weak); next high-value direction is regime-aware M3, not more M2 features without walk-forward proof.

---

## Open directions (not done)

1. Regime-conditioned M3 sizing  
2. Promote M3 threshold only with walk-forward validation  
3. Long/short sleeve (test Sharpe ~0.47 vs 0.79 long-only)  
4. Re-test `m1_components_rich` with regime-robust training or fold-aware gates  

---

## Reproduce research

```bash
python -m src.run_pipeline                    # full pipeline + companion reports
python -m src.walk_forward_research
python -m src.m1_weight_research
python -m src.m3_threshold_research
python -m src.ir_research
python -m src.m2_feature_research
```
