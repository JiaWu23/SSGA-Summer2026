"""Tests for improved M2 ranking features and per-asset heads."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import M2Config
from src.model_m2 import (
    PerAssetM2,
    SklearnM2,
    build_m2_features,
    create_m2_model,
    fit_m2,
    predict_m2,
)


def _labeled_panel(n_dates: int = 40, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["SPY", "TLT", "GLD"]
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-05", periods=n_dates, freq="W-FRI")
    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "z_mom_12w": rng.normal(0, 1),
                    "z_trend_signal": rng.normal(0, 1),
                    "z_vol_12w": rng.normal(0, 1),
                    "risk_off": float(rng.choice([0, 1])),
                    "M1_signal": 1,
                    "M1_score": rng.normal(0.5, 0.3),
                    "momentum_score": rng.normal(0, 1),
                    "trend_score": rng.normal(0, 1),
                    "macro_score": rng.normal(0, 0.5),
                    "risk_penalty": rng.uniform(0, 1),
                    "meta_label": int(rng.random() > 0.4),
                    "trade_return": rng.normal(0.001, 0.02),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def test_build_m2_features_includes_meta_columns(cfg):
    panel = _labeled_panel()
    cfg.models["m2"] = {
        **cfg.models.get("m2", {}),
        "use_meta_features": True,
        "include_asset_encoding": True,
    }
    X = build_m2_features(panel, cfg)
    assert "M1_score" in X.columns
    assert "m1_cs_rank" in X.columns
    assert any(c.startswith("asset_class_") for c in X.columns)


def test_per_asset_m2_fits_and_predicts(cfg):
    panel = _labeled_panel(n_dates=60)
    cfg.models["m2"] = {
        "type": "logistic_regression",
        "calibrate": False,
        "architecture": "per_asset",
        "min_asset_samples": 30,
        "use_meta_features": True,
        "include_asset_encoding": True,
    }
    model, X = fit_m2(panel, cfg)
    assert isinstance(model, PerAssetM2)
    assert len(model.asset_models) >= 1
    out = predict_m2(model, panel, cfg)
    assert out["p_success"].notna().sum() > 0


def test_create_m2_model_global(cfg):
    cfg.models["m2"] = {"architecture": "global"}
    assert isinstance(create_m2_model(cfg.m2), SklearnM2)
