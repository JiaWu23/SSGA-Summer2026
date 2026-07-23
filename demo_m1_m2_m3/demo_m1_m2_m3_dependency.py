"""Small demonstration that M2 and M3 only act on M1-generated candidates.

This script mirrors the logic in the repository:
- M2 is evaluated only where M1_signal != 0
- M3 sizes those candidates
- Final position weight is M1_signal * M3_size * budget
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig


def main() -> None:
    cfg = PipelineConfig()

    panel = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-09",
                "2024-01-09",
            ],
            "ticker": ["A", "B", "C", "A", "B"],
            "M1_signal": [1, -1, 0, 0, 1],
            "p_success": [0.82, 0.71, 0.95, 0.91, 0.66],
            "M3_size": [0.70, 0.55, 0.80, 0.85, 0.40],
        }
    )

    panel["eligible_for_m2"] = panel["M1_signal"] != 0
    panel["weight"] = panel["M1_signal"] * panel["M3_size"] * cfg.portfolio.base_budget_per_asset
    panel["position_created"] = panel["weight"] != 0

    print("=== Demo: M1 -> M2 -> M3 -> final weight ===")
    print(
        panel[
            ["date", "ticker", "M1_signal", "p_success", "M3_size", "weight", "eligible_for_m2", "position_created"]
        ].to_string(index=False)
    )

    print("\n=== Case 1: M1_signal != 0 ===")
    case_active = panel[panel["M1_signal"] != 0]
    print(case_active[["date", "ticker", "M1_signal", "p_success", "M3_size", "weight", "position_created"]].to_string(index=False))

    print("\n=== Case 2: M1_signal = 0 ===")
    case_flat = panel[panel["M1_signal"] == 0]
    print(case_flat[["date", "ticker", "M1_signal", "p_success", "M3_size", "weight", "position_created"]].to_string(index=False))

    print("\n=== Highlighted comparison ===")
    highlighted = panel[(panel["M1_signal"] == 0) & (panel["p_success"] > 0.5) & (panel["M3_size"] > 0.0)]
    print(highlighted[["date", "ticker", "M1_signal", "p_success", "M3_size", "weight", "position_created"]].to_string(index=False))

    print("\nInterpretation:")
    print("- Case 1: when M1_signal != 0, M2 and M3 can act on an existing trade candidate and a position is created.")
    print("- Case 2: when M1_signal = 0, M2 and M3 cannot create a position because the trade candidate does not exist.")
    print("- The final weight is zero whenever M1_signal is zero, even if p_success and M3_size are positive.")


if __name__ == "__main__":
    main()
