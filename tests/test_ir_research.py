"""Tests for IR attribution and interventions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ir_attribution import decompose_ir
from src.ir_interventions import InterventionSpec, evaluate_adoption_gates


def test_decompose_ir_positive_when_strategy_beats_benchmark():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-03", periods=52, freq="W-FRI")
    bench = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    strat = bench + 0.0005
    d = decompose_ir(strat, bench)
    assert d["information_ratio"] > 0
    assert d["mean_active_return_ann"] > 0


def test_evaluate_adoption_gates_picks_winner():
    sweep = pd.DataFrame(
        [
            {
                "variant": "ecdf_baseline",
                "period": "test",
                "sharpe": 0.96,
                "annualized_return": 0.07,
                "information_ratio": -0.10,
                "max_drawdown": -0.11,
                "annualized_turnover": 8.0,
            },
            {
                "variant": "ew_blend_0.8",
                "period": "test",
                "sharpe": 0.97,
                "annualized_return": 0.076,
                "information_ratio": 0.05,
                "max_drawdown": -0.12,
                "annualized_turnover": 8.5,
            },
        ]
    )
    d = evaluate_adoption_gates(sweep)
    assert d["verdict"] == "adopt"
    assert d["winner"] == "ew_blend_0.8"


def test_evaluate_adoption_gates_rejects_when_none_pass():
    sweep = pd.DataFrame(
        [
            {
                "variant": "ecdf_baseline",
                "period": "test",
                "sharpe": 0.96,
                "annualized_return": 0.07,
                "information_ratio": -0.10,
                "max_drawdown": -0.11,
                "annualized_turnover": 8.0,
            },
            {
                "variant": "m3_floor_0.4",
                "period": "test",
                "sharpe": 0.90,
                "annualized_return": 0.06,
                "information_ratio": -0.05,
                "max_drawdown": -0.15,
                "annualized_turnover": 7.0,
            },
        ]
    )
    d = evaluate_adoption_gates(sweep)
    assert d["verdict"] == "reject"
