"""Tests for walk-forward M1 weight validation."""

from __future__ import annotations

import pandas as pd

from src.factor_analysis import evaluate_m1_weight_walk_forward_decision


def test_decision_applies_when_m1_and_ecdf_improve():
    summary = pd.DataFrame(
        {
            "baseline_m1_sharpe": [0.70, 0.75, 0.80],
            "ic_m1_sharpe": [0.72, 0.78, 0.82],
            "baseline_ecdf_sharpe": [0.85, 0.90, 0.88],
            "ic_ecdf_sharpe": [0.86, 0.91, 0.89],
        }
    )
    d = evaluate_m1_weight_walk_forward_decision(summary)
    assert d["apply_ic_weights"] is True
    assert d["m1_fold_wins"] == 3


def test_decision_rejects_when_ecdf_degrades_too_much():
    summary = pd.DataFrame(
        {
            "baseline_m1_sharpe": [0.70, 0.75],
            "ic_m1_sharpe": [0.75, 0.80],
            "baseline_ecdf_sharpe": [0.90, 0.88],
            "ic_ecdf_sharpe": [0.65, 0.70],
        }
    )
    d = evaluate_m1_weight_walk_forward_decision(summary)
    assert d["apply_ic_weights"] is False


def test_finalize_rejects_ic_when_walk_forward_fails():
    from src.factor_analysis import finalize_m1_weight_recommendation

    holdout = {
        "variant": "ic_proportional_train",
        "weights": {"momentum": 0.49, "trend": 0.06, "macro": 0.15, "risk_penalty": 0.30},
        "test_sharpe": 0.795,
        "rationale": "Holdout improvement.",
    }
    wf = {
        "apply_ic_weights": False,
        "mean_m1_sharpe_gain": -0.035,
        "mean_ecdf_sharpe_gain": -0.084,
        "m1_fold_wins": 2,
        "n_folds": 6,
        "reason": "Walk-forward rejected.",
    }
    baseline = {"momentum": 0.45, "trend": 0.25, "macro": 0.20, "risk_penalty": 0.10}
    out = finalize_m1_weight_recommendation(holdout, wf, baseline)
    assert out["config_action"] == "keep_baseline"
    assert out["variant"] == "baseline"
    assert out["weights"]["momentum"] == 0.45
