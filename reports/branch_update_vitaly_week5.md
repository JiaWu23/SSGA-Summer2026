# Branch Update — Technical (`vitaly_week5`)

**Status:** August 2026 — merged into active root with index-first universe.

**Executive summary:** [BRANCH_UPDATE_REPORT.md](../BRANCH_UPDATE_REPORT.md)  
**Research index:** [week5_research_summary.md](week5_research_summary.md)  
**Data policy:** [DATA_SOURCES_AND_ETL.md](../DATA_SOURCES_AND_ETL.md)

---

## Summary

The `vitaly_week5` branch added Joubert-aligned M3 formalization, companion diagnostics, M2 feature enrichment (52 inputs), walk-forward evaluation, and research modules (M1 weights, M3 threshold, IR attribution, M2 variants). **August 2026** completed index sleeve migration (`SP500`, …) via `IndexProvider`.

## Headline OOS (long-only, test 2021+, index sleeves)

See [final_report.md](final_report.md) for latest numbers. Representative test-window metrics:

| Strategy | Sharpe (approx.) |
| --- | ---: |
| M1 Only | 0.88 |
| M1+M2+M3 Binary | 0.95 |
| M1+M2+M3 ECDF | 0.91 |

## Config adopted vs rejected

| Item | Verdict |
| --- | --- |
| M3 layer + ECDF | Adopted |
| 52-feature M2 | Adopted |
| Index sleeve IDs | Adopted |
| IC-proportional M1 weights | Rejected (walk-forward) |
| M3 T=0.55 promotion | Rejected |
| IR overlays | Rejected |
| M2 `m1_components_rich` | Rejected (walk-forward) |

## Artifacts

- `src/model_m3.py`, `src/m3_diagnostics.py`, research CLIs under `src/*_research.py`
- `reports/` companion analyses + `Summer_2026_Meta_Labeling_Deck.pptx`
- `TERMINOLOGY.md`, updated `DATA_SOURCES_AND_ETL.md`
