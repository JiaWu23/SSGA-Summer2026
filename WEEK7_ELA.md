# Week 7 — Ela · Deliverables, controlled experiments, reproducibility

Branch: `week7-ela`. Default config and pipeline behaviour are **unchanged**; every experiment
is opt-in. All numbers below come from reproduced, walk-forward runs.

## What this branch adds

| Area | File(s) |
| --- | --- |
| Sample-controlled macro ablation (`MacroConfig.model_series` — excluded macro signals are **zeroed, not dropped**, so the sample/panel is identical across ablations) | `src/config.py`, `src/feature_engineering.py` |
| Macro sweep (which macro signals feed the model; `--m1-macro-free` for macro-in-M2) | `scripts/macro_sweep.py` |
| M1 factor-weight sweep, walk-forward (Axis A) | `scripts/m1_weight_sweep.py` |
| Deliverables deck generator + deck | `scripts/make_deliverables_deck.py`, `presentation/SSGA_Deliverables_Deck.pptx` |

Reproduce:

```bash
python scripts/macro_sweep.py                 # macro in M1 vs M2 (walk-forward)
python scripts/macro_sweep.py --m1-macro-free # macro only in M2
python scripts/m1_weight_sweep.py             # M1 weight sweep (Axis A)
python scripts/make_deliverables_deck.py      # regenerate the deck
```

## How it maps to the reviewer's four deliverables

1. **Architecture diagram (inputs, outputs, training, inference).** Deck slides 3–4.
2. **Worked numerical example (one asset → final weight).** Deck slide 8 (SP500 → ~12%).
3. **Structured experimental matrix (broader M3 & allocation alternatives).** Deck slide 9; Axis A run in `runs/m1_weight_sweep/`.
4. **Exposure-controlled + walk-forward results (genuine vs mechanical).** Deck slide 10; the exposure-matched tests (constant-haircut, vol-matched, `Spearman(M3_size, realized PnL)`) are specified as the decisive next step.

Also addressed: **meta-labeling explanation** (the five sub-questions — training sample, meta-label
construction, 4-week horizon + threshold, features, calibration) on deck slides 5–6; and
**one reproducible result set** (deck slide 13).

## Key findings (all walk-forward, 6 folds)

- **Baseline reproduced bit-for-bit** (M1-only Sharpe 0.709; ECDF edge vs M1 **−0.19, 1/6 folds positive**; IR vs EW −0.63). The older report's **+0.177 / 4-of-6** figure was a stale ETF-era number — reconcile to the index-era run.
- **M1 weights were never tuned.** A heavier downside/risk-penalty weight (`0.40/0.22/0.18/0.20`) gives **walk-forward Sharpe 0.811, positive in all 6 folds** — the most robust improvement found (baseline 0.709).
- **Momentum & trend are 0.78 correlated but not redundant:** either alone (~0.50) is far worse than both together (0.71).
- **Macro is not a lever for M1's selection** (best macro set beats macro-off by only +0.013, t≈0.19). But once M1 is a clean technical model, the right macro set in **M2** (dropping backward-looking CPI) turns the meta-label **positive** — first configuration where meta-labeling adds value. Small (2/6 folds); needs exposure-matching + Deflated Sharpe before any claim.

## Next

Exposure-matched M3 comparison (the decisive test); regime-dependent M3; deepen the
technical-M1 + macro-in-M2 design; broaden the asset-allocation axis. This is treated as the
**beginning** of the investigation, not a conclusion about M3 or meta-labeling.
