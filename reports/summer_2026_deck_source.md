# Summer 2026 Deck Source — Jun 28 to Aug 8, 2026

See `scripts/make_summer_deck.py` → `reports/Summer_2026_Meta_Labeling_Deck.pptx`.

**Universe:** 7 index sleeves (`SP500`, `MSCI_EAFE`, `MSCI_EM`, `UST_7_10`, `US_HIGH_YIELD`, `GOLD_SPOT`, `US_REIT`).

**Disclaimer:** Research use only — not investment advice.

---

## Timeline

| Period | Work |
|--------|------|
| Jun 28 – early Jul | Static M1, LR M2, macro→M2, index signal switch |
| Jul | M3 formalization, companion reports, M2 AUC lift |
| Jul–Aug | Walk-forward, M1 weights, M3 threshold, IR, M2 enrichment research |
| Aug 8 | Index-first migration, validation, docs alignment, summer deck |

---

## Headline OOS (long-only, test 2021+, index sleeves)

| Strategy | Sharpe (approx.) |
|----------|-----------------|
| M1 Only | 0.88 |
| M1+M2+M3 Binary | 0.95 |
| M1+M2+M3 ECDF | 0.91 |

See `reports/final_report.md` for latest pipeline numbers.

---

## Key topics for slides

1. M1→M2→M3 architecture (Joubert)
2. Index universe & data policy
3. M3 allocation states and `m3_zero`
4. Walk-forward adoption gates
5. Adopted vs rejected research
6. 74 tests passing
