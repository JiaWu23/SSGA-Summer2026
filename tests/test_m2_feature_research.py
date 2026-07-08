"""Tests for M2 feature enrichment research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.m2_feature_enrichment import build_m2_features_variant, enrich_m2_features
from src.m2_feature_research import evaluate_adoption_decision


def test_enrich_adds_component_ranks():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-03", periods=4, freq="W-FRI"), ["SPY", "TLT"]],
        names=["date", "ticker"],
    )
    panel = pd.DataFrame(
        {
            "momentum_score": np.random.randn(len(idx)),
            "trend_score": np.random.randn(len(idx)),
            "macro_score": np.random.randn(len(idx)),
            "risk_penalty": np.random.randn(len(idx)),
            "M1_score": np.random.randn(len(idx)),
            "M1_signal": 1,
        },
        index=idx,
    )
    base = pd.DataFrame({"f1": [1.0] * len(idx)}, index=idx)
    out = enrich_m2_features(base, panel, "m1_components_rich")
    assert "momentum_score_cs_rank" in out.columns
    assert "factor_spread" in out.columns


def test_adoption_rejects_weak_variant():
    sweep = pd.DataFrame(
        [
            {
                "variant": "configured",
                "test_auc": 0.589,
                "ecdf_test_sharpe": 0.964,
            },
            {
                "variant": "full_enriched",
                "test_auc": 0.590,
                "ecdf_test_sharpe": 0.90,
            },
        ]
    )
    d = evaluate_adoption_decision(sweep, pd.DataFrame())
    assert d["verdict"] == "reject"
