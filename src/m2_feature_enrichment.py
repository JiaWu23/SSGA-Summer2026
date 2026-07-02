"""Extended M2 feature sets for research: M1 factor context + dynamic external factors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.config import PipelineConfig, clone_config_with_m2_variant
from src.factor_analysis import FACTOR_COLS, compute_factor_ic
from src.model_m2 import (
    M1_COMPONENT_COLS,
    M2_META_DERIVED_COLS,
    _asset_class_dummies,
    _cross_sectional_m1_rank,
    build_m2_features,
)

M2_FEATURE_VARIANTS: tuple[str, ...] = (
    "legacy_global",
    "configured",
    "m1_components_rich",
    "regime_external_rich",
    "full_enriched",
    "trend_emphasis",
    "ic_alignment",
)

REGIME_COLS = (
    "risk_off",
    "curve_inverted",
    "inflation_up",
    "growth_down",
    "vix_level",
    "vix_change_4w",
    "credit_stress",
    "yield_curve",
    "growth_trend",
    "inflation_trend",
    "policy_rate_change",
    "unemployment_change",
)

DISPERSION_COLS = (
    "cross_asset_dispersion_4w",
    "cross_asset_dispersion_12w",
    "average_pairwise_correlation_26w",
)


def m2_config_for_variant(variant: str, base_cfg: PipelineConfig) -> PipelineConfig:
    """Map research variant name to PipelineConfig with appropriate M2 flags."""
    if variant == "legacy_global":
        return clone_config_with_m2_variant(
            base_cfg,
            variant,
            use_meta_features=False,
            include_asset_encoding=False,
            type="logistic_regression",
            calibrate=True,
            architecture="global",
        )
    return clone_config_with_m2_variant(
        base_cfg,
        variant,
        use_meta_features=True,
        include_asset_encoding=True,
        type="logistic_regression",
        calibrate=True,
        architecture="global",
    )


def _cs_rank(series: pd.Series, panel: pd.DataFrame) -> pd.Series:
    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values("date")
        return series.groupby(dates).rank(pct=True)
    return series.rank(pct=True)


def _train_ic_weights(train_panel: pd.DataFrame) -> dict[str, float]:
    """Positive IC weights from train period only (no look-ahead)."""
    ic_df = compute_factor_ic(train_panel, period_label="train")
    if ic_df.empty:
        return {c: 0.25 for c in FACTOR_COLS}
    weights: dict[str, float] = {}
    for col in FACTOR_COLS:
        sub = ic_df[ic_df["factor"] == col]
        ic = float(sub["ic_mean"].iloc[0]) if not sub.empty else 0.0
        weights[col] = max(0.0, ic)
    total = sum(weights.values())
    if total <= 0:
        return {c: 1.0 / len(FACTOR_COLS) for c in FACTOR_COLS}
    return {k: v / total for k, v in weights.items()}


def enrich_m2_features(
    base: pd.DataFrame,
    panel: pd.DataFrame,
    variant: str,
    *,
    train_panel: pd.DataFrame | None = None,
    ic_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add research feature blocks on top of build_m2_features output."""
    if variant in ("legacy_global", "configured"):
        return base

    extra = pd.DataFrame(index=panel.index)
    comps = [c for c in M1_COMPONENT_COLS if c in panel.columns]

    if variant in ("m1_components_rich", "full_enriched", "ic_alignment", "trend_emphasis"):
        for col in comps:
            extra[f"{col}_cs_rank"] = _cs_rank(panel[col], panel)
        if len(comps) >= 2:
            comp_df = panel[comps]
            extra["factor_spread"] = comp_df.max(axis=1) - comp_df.min(axis=1)
            extra["factor_mean"] = comp_df.mean(axis=1)
            signs = np.sign(comp_df.fillna(0))
            extra["factor_sign_agreement"] = signs.std(axis=1).rsub(1.0)
        if "M1_conviction" in panel.columns:
            extra["m1_conviction"] = panel["M1_conviction"]
        if "trend_score" in panel.columns and "momentum_score" in panel.columns:
            extra["trend_minus_momentum"] = panel["trend_score"] - panel["momentum_score"]
            extra["trend_over_momentum"] = panel["trend_score"] / (
                panel["momentum_score"].abs() + 1e-6
            )

    if variant == "trend_emphasis":
        if all(c in panel.columns for c in ("trend_score", "momentum_score", "macro_score")):
            extra["trend_heavy_score"] = (
                0.55 * panel["trend_score"]
                + 0.25 * panel["momentum_score"]
                + 0.20 * panel["macro_score"]
            )
            extra["trend_heavy_cs_rank"] = _cs_rank(extra["trend_heavy_score"], panel)

    if variant in ("regime_external_rich", "full_enriched"):
        for regime in REGIME_COLS:
            if regime not in panel.columns:
                continue
            if "M1_score" in panel.columns:
                extra[f"m1_x_{regime}"] = panel["M1_score"] * panel[regime]
            for col in comps:
                extra[f"{col}_x_{regime}"] = panel[col] * panel[regime]
        for disp in DISPERSION_COLS:
            if disp in panel.columns and "M1_score" in panel.columns:
                extra[f"m1_x_{disp}"] = panel["M1_score"] * panel[disp]
        if "credit_stress" in panel.columns and "macro_score" in panel.columns:
            extra["macro_x_credit_stress"] = panel["macro_score"] * panel["credit_stress"]
        if "yield_curve" in panel.columns and "trend_score" in panel.columns:
            extra["trend_x_yield_curve"] = panel["trend_score"] * panel["yield_curve"]

    if variant in ("ic_alignment", "full_enriched"):
        ic_w = ic_weights if ic_weights is not None else (
            _train_ic_weights(train_panel) if train_panel is not None else {c: 0.25 for c in FACTOR_COLS}
        )
        align = pd.Series(0.0, index=panel.index)
        for col, w in ic_w.items():
            rank_col = f"{col}_cs_rank"
            if rank_col in extra.columns:
                align = align + extra[rank_col] * w
            elif col in panel.columns:
                align = align + _cs_rank(panel[col], panel) * w
        extra["factor_ic_alignment"] = align

    extra = extra.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    combined = pd.concat([base, extra], axis=1)
    return combined.loc[:, list(dict.fromkeys(combined.columns))]


def build_m2_features_variant(
    panel: pd.DataFrame,
    cfg: PipelineConfig,
    variant: str,
    *,
    train_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build full M2 feature matrix for a named research variant."""
    variant_cfg = m2_config_for_variant(variant, cfg)
    base = build_m2_features(panel, variant_cfg)
    return enrich_m2_features(base, panel, variant, train_panel=train_panel)


def describe_variant(variant: str) -> str:
    descriptions: dict[str, str] = {
        "legacy_global": "40 base factors only; no M1 meta or asset encoding (main baseline).",
        "configured": "Current production: 52 features with M1 meta + asset class dummies.",
        "m1_components_rich": "Configured + M1 factor CS ranks, spread, sign agreement, conviction, trend−momentum.",
        "regime_external_rich": "Configured + VIX/macro/regime × M1/component interactions + dispersion overlays.",
        "full_enriched": "M1 component rich + regime external + train IC-weighted factor alignment score.",
        "trend_emphasis": "Configured + trend-heavy composite (M1 IC analysis: trend strongest on test).",
        "ic_alignment": "Configured + per-factor CS ranks + train IC-weighted alignment (no look-ahead).",
    }
    return descriptions.get(variant, variant)
