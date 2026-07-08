"""Tests for walk-forward ECDF stability analysis."""

from __future__ import annotations

import pandas as pd

from src.evaluation import analyze_walk_forward_stability


def _sample_wf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": ["1", "2", "3", "4"],
            "test_start": ["2015-01-01", "2017-01-01", "2019-01-01", "2021-01-01"],
            "test_end": ["2016-12-31", "2018-12-31", "2020-12-31", "2022-12-31"],
            "m1_only_sharpe": [0.5, 0.6, 0.7, 0.8],
            "ecdf_sharpe": [0.6, 0.7, 0.65, 0.95],
            "ecdf_sharpe_edge_vs_m1": [0.1, 0.1, -0.05, 0.15],
            "equal_weight_sharpe": [0.55, 0.58, 0.62, 0.68],
            "m2_auc": [0.58, 0.59, 0.57, 0.60],
        }
    )


def test_analyze_walk_forward_stable_majority():
    s = analyze_walk_forward_stability(_sample_wf(), production_test_start="2021-01-01")
    assert s["n_folds"] == 4
    assert s["positive_edge_folds"] == 3
    assert s["stable_ecdf_edge"] is True
    assert s["verdict"] == "stable_majority"


def test_analyze_walk_forward_detects_production_outlier():
    s = analyze_walk_forward_stability(_sample_wf(), production_test_start="2021-01-01")
    assert s["production_mean_edge"] == 0.15
    assert s["pre_production_mean_edge"] < 0.15
