"""M3 bet-sizing layer — deterministic rules mapping M2 probability to position fraction.

M3 is NOT a classifier. Per Joubert (2022), M2 outputs P(true positive); M3 converts that
probability into a bet fraction f in [0, 1] before portfolio constraints and vol targeting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import M3Config, PipelineConfig
from src.position_sizing import SizingMode, compute_sizes, fit_ecdf


def compute_m3_size(
    p_success: pd.Series,
    mode: SizingMode | str,
    *,
    threshold: float = 0.55,
    train_proba: pd.Series | None = None,
    train_sorted: np.ndarray | None = None,
) -> pd.Series:
    """Map M2 probabilities to bet-sizing multipliers in [0, 1]."""
    return compute_sizes(
        p_success,
        mode,
        threshold=threshold,
        train_proba=train_proba,
        train_sorted=train_sorted,
    ).rename("M3_size")


def passthrough_m3_size(p_success: pd.Series) -> pd.Series:
    """Diagnostic rule: M3_size = p_success (unscaled probability)."""
    return p_success.clip(0.0, 1.0).rename("M3_size")


def allocation_state(m1_signal: pd.Series, m3_size: pd.Series) -> pd.Series:
    """Label each asset-week: no_signal | m3_zero | m3_active."""
    sig = m1_signal.fillna(0)
    size = m3_size.fillna(0.0)
    out = pd.Series("no_signal", index=sig.index, dtype=object)
    active_mask = sig != 0
    out.loc[active_mask & (size <= 0.0)] = "m3_zero"
    out.loc[active_mask & (size > 0.0)] = "m3_active"
    return out.rename("allocation_state")


def attach_m3_to_panel(
    panel: pd.DataFrame,
    cfg: PipelineConfig,
    train_proba: pd.Series | None = None,
) -> pd.DataFrame:
    """Persist M3_size (default mode) and diagnostic columns for all M3 rules."""
    out = panel.copy()
    m3_cfg = cfg.m3
    train_sorted = fit_ecdf(train_proba) if train_proba is not None and len(train_proba.dropna()) else None
    threshold = m3_cfg.threshold if m3_cfg.threshold is not None else cfg.m2.threshold

    p = out["p_success"] if "p_success" in out.columns else pd.Series(np.nan, index=out.index)

    for mode, col in [
        (SizingMode.BINARY, "M3_size_binary"),
        (SizingMode.LINEAR, "M3_size_linear"),
        (SizingMode.ECDF, "M3_size_ecdf"),
    ]:
        sizes = compute_m3_size(
            p,
            mode,
            threshold=threshold,
            train_proba=train_proba,
            train_sorted=train_sorted,
        )
        out[col] = sizes.reindex(out.index)

    default_col = {
        "binary": "M3_size_binary",
        "linear": "M3_size_linear",
        "ecdf": "M3_size_ecdf",
    }.get(m3_cfg.mode, "M3_size_linear")
    out["M3_size"] = out[default_col]

    m1_sig = out["M1_signal"] if "M1_signal" in out.columns else pd.Series(0, index=out.index)
    out["allocation_state"] = allocation_state(m1_sig, out["M3_size"])
    return out


def m3_config_from_pipeline(cfg: PipelineConfig) -> M3Config:
    return cfg.m3
