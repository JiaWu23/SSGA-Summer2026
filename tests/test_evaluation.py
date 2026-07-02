"""Tests for walk-forward evaluation and transaction-cost sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import STRATEGY_M1_M2_M3_ECDF, BacktestResult, _run_backtest
from src.config import EvaluationConfig, load_config
from src.evaluation import (
    build_walk_forward_folds,
    generate_evaluation_report,
    run_transaction_cost_sensitivity,
    save_evaluation_charts,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config/config.yaml"


def _synthetic_panel_index(n_weeks: int = 520) -> pd.DataFrame:
    dates = pd.date_range("2006-01-06", periods=n_weeks, freq="W-FRI")
    tickers = ["SPY", "TLT"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame({"score": 0.0}, index=idx)


def test_build_walk_forward_folds_expanding():
    cfg = load_config(CONFIG)
    panel = _synthetic_panel_index(n_weeks=900)
    eval_cfg = EvaluationConfig(
        walk_forward_first_train_end="2014-12-31",
        walk_forward_test_years=2,
    )
    folds = build_walk_forward_folds(panel, cfg, eval_cfg)
    assert len(folds) >= 2
    assert folds[0]["train_end"] == "2014-12-31"
    assert folds[0]["test_start"] == "2015-01-01"
    assert folds[1]["train_end"] == folds[0]["test_end"]


def test_run_transaction_cost_sensitivity_monotonic_cost_impact():
    cfg = load_config(CONFIG)
    dates = pd.date_range("2021-01-08", periods=40, freq="W-FRI")
    tickers = ["SPY", "TLT"]
    weights = pd.DataFrame(
        0.5,
        index=dates,
        columns=tickers,
    )
    weights.iloc[::2] = 0.25
    weights.iloc[1::2] = 0.75
    returns = pd.DataFrame(
        np.random.default_rng(0).normal(0.001, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    bt = _run_backtest("m1_only", weights, returns, transaction_cost_bps=5.0)
    results = {
        "m1_only": BacktestResult(
            name="m1_only",
            returns=bt.returns,
            weights=weights,
            turnover=bt.turnover,
            transaction_costs=bt.transaction_costs,
            gross_returns=bt.gross_returns,
        ),
        STRATEGY_M1_M2_M3_ECDF: BacktestResult(
            name=STRATEGY_M1_M2_M3_ECDF,
            returns=bt.returns * 0.95,
            weights=weights,
            turnover=bt.turnover,
            transaction_costs=bt.transaction_costs,
            gross_returns=bt.gross_returns,
        ),
    }
    eval_cfg = EvaluationConfig(transaction_cost_bps_grid=(0.0, 10.0, 25.0))
    df = run_transaction_cost_sensitivity(
        results,
        returns,
        cfg,
        test_start="2021-01-01",
        eval_cfg=eval_cfg,
    )
    assert not df.empty
    assert set(df["strategy"].unique()) >= {"m1_only", STRATEGY_M1_M2_M3_ECDF}
    assert set(df["transaction_cost_bps"]) == {0.0, 10.0, 25.0}
    assert "ecdf_sharpe_edge_vs_m1" in df.columns


def test_save_evaluation_charts_and_report(tmp_path: Path):
    walk_forward = pd.DataFrame(
        {
            "fold_id": ["1", "2"],
            "m1_only_sharpe": [0.5, 0.6],
            "ecdf_sharpe": [0.7, 0.65],
            "ecdf_sharpe_edge_vs_m1": [0.2, 0.05],
            "equal_weight_sharpe": [0.4, 0.45],
        }
    )
    tc = pd.DataFrame(
        {
            "transaction_cost_bps": [0.0, 10.0],
            "strategy": ["m1_only", "m1_only"],
            "sharpe": [0.8, 0.7],
            "annualized_return": [0.05, 0.04],
            "max_drawdown": [-0.1, -0.12],
            "hit_rate": [0.55, 0.54],
            "n_weeks": [100, 100],
        }
    )
    fig_dir = tmp_path / "figures"
    saved = save_evaluation_charts(walk_forward, tc, fig_dir)
    assert "walk_forward_sharpe.png" in saved
    assert "walk_forward_ecdf_edge.png" in saved
    assert "transaction_cost_sensitivity.png" in saved

    eval_summary = {
        "walk_forward": walk_forward,
        "transaction_cost_sensitivity": tc,
        "walk_forward_mean_ecdf_edge": 0.125,
        "walk_forward_mean_m2_auc": 0.58,
        "ecdf_edge_persists_at_25bps": True,
    }
    report_path = tmp_path / "evaluation_analysis.md"
    generate_evaluation_report(eval_summary, report_path, mode_name="long_only")
    text = report_path.read_text()
    assert "Walk-forward validation" in text
    assert "Transaction-cost sensitivity" in text


def test_config_loads_evaluation_section():
    cfg = load_config(CONFIG)
    assert cfg.evaluation.walk_forward_enabled is True
    assert cfg.evaluation.walk_forward_first_train_end == "2014-12-31"
    assert 25.0 in cfg.evaluation.transaction_cost_bps_grid
