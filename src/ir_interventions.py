"""Research interventions to improve IR while preserving ECDF risk shaping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import (
    STRATEGY_M1_M2_M3_ECDF,
    _run_backtest,
    equal_weight_returns,
    strategy_weights_from_panel,
)
from src.config import PipelineConfig
from src.diagnostics import (
    annualized_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    strategy_metrics_on_period,
)
from src.portfolio import WEEKS_PER_YEAR, apply_constraints_by_date, apply_vol_target_wide, weights_to_wide
from src.position_sizing import SizingMode, compute_sizes, fit_ecdf

REGIME_M3_OFF_SCALE = 1.15
REGIME_M3_ON_SCALE = 0.90


@dataclass
class InterventionSpec:
    """Research-only portfolio overlay on ECDF weights."""

    name: str
    kind: str
    # B: m3 floor
    m3_floor: float | None = None
    # A: exposure renormalize to fraction of M1 gross (1.0 = match M1 mean)
    exposure_target_vs_m1: float | None = None
    # C: EW blend — alpha on strategy, (1-alpha) on EW
    ew_blend_alpha: float | None = None
    # D: regime-conditioned M3
    regime_m3: bool = False
    # E: vol-target bump when mean M3 size is low
    vol_bump_when_m3_below: float | None = None
    vol_bump_factor: float = 1.15


def _panel_df(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.reset_index().copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"])


def _build_raw_ecdf_weights(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    train_proba: pd.Series | None,
    *,
    m3_floor: float | None = None,
    regime_m3: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide weights before optional post-processing; also returns per-date mean M3 size."""
    df = _panel_df(panel)
    threshold = cfg.m3.threshold or cfg.m2.threshold
    train_sorted = fit_ecdf(train_proba) if train_proba is not None and len(train_proba.dropna()) else None
    sizes = compute_sizes(
        df["p_success"],
        SizingMode.ECDF,
        threshold=threshold,
        train_proba=train_proba,
        train_sorted=train_sorted,
    ).reindex(df.index).fillna(0.0)

    if m3_floor is not None:
        sizes = sizes.clip(lower=float(m3_floor))

    if regime_m3 and "risk_off" in df.columns:
        risk = df["risk_off"].fillna(0).astype(float)
        mult = risk.map({1.0: REGIME_M3_ON_SCALE, 0.0: REGIME_M3_OFF_SCALE}).fillna(1.0)
        sizes = (sizes * mult).clip(0.0, 1.0)

    df["M3_size"] = sizes
    m3_by_date = sizes.groupby(level="date").mean()

    if "M1_conviction" in df.columns:
        df["raw_weight"] = (
            df["M1_signal"] * df["M3_size"] * df["M1_conviction"] * cfg.portfolio.base_budget_per_asset
        )
    else:
        df["raw_weight"] = df["M1_signal"] * df["M3_size"] * cfg.portfolio.base_budget_per_asset
    df["weight"] = apply_constraints_by_date(df, cfg.portfolio)
    w_wide = weights_to_wide(df.reset_index())

    vol_cfg = cfg.portfolio
    if vol_cfg.vol_target_ann and vol_cfg.vol_target_ann > 0:
        w_wide = apply_vol_target_wide(w_wide, returns_wide, vol_cfg)

    return w_wide, m3_by_date.to_frame("mean_m3_size")


def _renormalize_to_m1_gross(
    w: pd.DataFrame,
    m1_w: pd.DataFrame,
    target_fraction: float,
) -> pd.DataFrame:
    """Scale weekly gross exposure toward M1-only gross × target_fraction."""
    m1_gross = m1_w.abs().sum(axis=1).reindex(w.index).fillna(0.0)
    target = m1_gross * float(target_fraction)
    out = []
    for date in w.index:
        row = w.loc[date]
        gross = row.abs().sum()
        tgt = target.loc[date] if date in target.index else gross
        if gross > 1e-12 and tgt > 0:
            row = row * (tgt / gross)
        out.append(row)
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def _blend_with_ew(w: pd.DataFrame, ew_w: pd.DataFrame, alpha: float) -> pd.DataFrame:
    ew = ew_w.reindex(w.index).ffill().fillna(0.0)
    blended = w * alpha + ew * (1.0 - alpha)
    return blended


def _vol_bump_weights(
    w: pd.DataFrame,
    m3_by_date: pd.DataFrame,
    *,
    m3_threshold: float,
    bump_factor: float,
    cfg_portfolio,
    returns_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Re-apply vol target with higher effective target when mean M3 size is low."""
    if not cfg_portfolio.vol_target_ann or cfg_portfolio.vol_target_ann <= 0:
        return w
    scale_up = pd.Series(1.0, index=w.index)
    m3_mean = m3_by_date["mean_m3_size"].reindex(w.index).fillna(1.0)
    low = m3_mean < m3_threshold
    scale_up.loc[low] = bump_factor
  # multiply weights by extra scale on low-M3 weeks (before gross cap)
    bumped = w.mul(scale_up, axis=0)
    out = []
    for date in bumped.index:
        row = bumped.loc[date]
        gross = row.abs().sum()
        if gross > cfg_portfolio.max_gross_exposure and gross > 0:
            row = row * (cfg_portfolio.max_gross_exposure / gross)
        out.append(row)
    return pd.DataFrame(out, index=bumped.index, columns=bumped.columns)


def build_intervention_weights(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    train_proba: pd.Series | None,
    m1_weights: pd.DataFrame,
    ew_weights: pd.DataFrame,
    spec: InterventionSpec,
) -> pd.DataFrame:
    m3_floor = spec.m3_floor if spec.kind == "m3_floor" else None
    regime_m3 = spec.regime_m3 or spec.kind == "regime_m3"

    w, m3_by_date = _build_raw_ecdf_weights(
        panel,
        returns_wide,
        cfg,
        train_proba,
        m3_floor=m3_floor,
        regime_m3=regime_m3,
    )

    if spec.kind == "exposure_renorm" and spec.exposure_target_vs_m1 is not None:
        w = _renormalize_to_m1_gross(w, m1_weights, spec.exposure_target_vs_m1)

    if spec.kind == "ew_blend" and spec.ew_blend_alpha is not None:
        w = _blend_with_ew(w, ew_weights, spec.ew_blend_alpha)

    if spec.kind == "vol_bump" and spec.vol_bump_when_m3_below is not None:
        w = _vol_bump_weights(
            w,
            m3_by_date,
            m3_threshold=spec.vol_bump_when_m3_below,
            bump_factor=spec.vol_bump_factor,
            cfg_portfolio=cfg.portfolio,
            returns_wide=returns_wide,
        )

    return w


def default_intervention_grid() -> list[InterventionSpec]:
    specs: list[InterventionSpec] = [
        InterventionSpec(name="ecdf_baseline", kind="baseline"),
    ]
    for frac in (0.85, 1.0, 1.1):
        specs.append(
            InterventionSpec(
                name=f"exposure_renorm_{frac:.2f}",
                kind="exposure_renorm",
                exposure_target_vs_m1=frac,
            )
        )
    for floor in (0.4, 0.5, 0.6, 0.7):
        specs.append(InterventionSpec(name=f"m3_floor_{floor:.1f}", kind="m3_floor", m3_floor=floor))
    for alpha in (0.7, 0.8, 0.9):
        specs.append(
            InterventionSpec(name=f"ew_blend_{alpha:.1f}", kind="ew_blend", ew_blend_alpha=alpha)
        )
    specs.append(InterventionSpec(name="regime_m3", kind="regime_m3", regime_m3=True))
    for thresh, bump in ((0.55, 1.10), (0.55, 1.15), (0.60, 1.15)):
        specs.append(
            InterventionSpec(
                name=f"vol_bump_{thresh:.2f}_{bump:.2f}",
                kind="vol_bump",
                vol_bump_when_m3_below=thresh,
                vol_bump_factor=bump,
            )
        )
    return specs


def _metrics_row(
    name: str,
    bt,
    bench_returns: pd.Series,
    *,
    period_label: str,
    start: str | None,
    end: str | None,
    baseline_turnover: float | None = None,
) -> dict[str, Any]:
    r = bt.returns.copy()
    r.index = pd.to_datetime(r.index)
    if start:
        r = r[r.index >= pd.Timestamp(start)]
    if end:
        r = r[r.index <= pd.Timestamp(end)]
    bench = bench_returns.reindex(r.index).fillna(0.0)
    period = strategy_metrics_on_period(r, start=None, end=None)
    gross = bt.weights.reindex(r.index).ffill().abs().sum(axis=1).mean()
    row = {
        "variant": name,
        "period": period_label,
        "annualized_return": period["annualized_return"],
        "sharpe": period["sharpe"],
        "max_drawdown": period["max_drawdown"],
        "excess_return_vs_benchmark": period["annualized_return"] - annualized_return(bench),
        "information_ratio": information_ratio(r, bench),
        "mean_gross_exposure": float(gross),
        "annualized_turnover": float(bt.turnover.reindex(r.index).mean() * WEEKS_PER_YEAR),
        "hit_rate": period["hit_rate"],
        "n_weeks": period["n_weeks"],
    }
    if baseline_turnover is not None and baseline_turnover > 0:
        row["turnover_vs_baseline"] = row["annualized_turnover"] / baseline_turnover - 1.0
    return row


def sweep_ir_interventions(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    train_proba: pd.Series | None,
    *,
    test_start: str,
    test_end: str | None = None,
    specs: list[InterventionSpec] | None = None,
) -> pd.DataFrame:
    specs = specs or default_intervention_grid()
    tc = cfg.portfolio.transaction_cost_bps

    m1_w = strategy_weights_from_panel(panel, returns_wide, cfg, SizingMode.LINEAR, use_m2=False)
    ew_res = equal_weight_returns(returns_wide, cfg.assets.tickers)
    ew_w = ew_res.weights

    baseline_w = strategy_weights_from_panel(
        panel,
        returns_wide,
        cfg,
        SizingMode.ECDF,
        use_m2=True,
        train_proba=train_proba,
    )
    baseline_bt = _run_backtest(STRATEGY_M1_M2_M3_ECDF, baseline_w, returns_wide, tc)
    baseline_turnover = float(baseline_bt.turnover.mean() * WEEKS_PER_YEAR)

    rows: list[dict[str, Any]] = []
    bench_returns = ew_res.returns

    for spec in specs:
        if spec.kind == "baseline":
            w = baseline_w
        else:
            w = build_intervention_weights(
                panel, returns_wide, cfg, train_proba, m1_w, ew_w, spec
            )
        bt = _run_backtest(spec.name, w, returns_wide, tc)
        for period_label, start, end in [("full", None, None), ("test", test_start, test_end)]:
            rows.append(
                _metrics_row(
                    spec.name,
                    bt,
                    bench_returns,
                    period_label=period_label,
                    start=start,
                    end=end,
                    baseline_turnover=baseline_turnover if spec.kind != "baseline" else None,
                )
            )

    return pd.DataFrame(rows)


# Adoption gates (test 2021+)
GATE_SHARPE_MIN = 0.95
GATE_RETURN_MIN = 0.075
GATE_IR_MIN = 0.0
GATE_DD_WORSE_THAN_BASELINE_PP = 0.02
GATE_TURNOVER_INCREASE_MAX = 0.30


def evaluate_adoption_gates(
    sweep: pd.DataFrame,
    *,
    baseline_variant: str = "ecdf_baseline",
) -> dict[str, Any]:
    test = sweep[sweep["period"] == "test"].copy()
    if test.empty:
        return {"winner": None, "verdict": "reject", "reason": "empty sweep"}

    base = test[test["variant"] == baseline_variant]
    if base.empty:
        base_row = test.iloc[0]
    else:
        base_row = base.iloc[0]

    base_sharpe = float(base_row["sharpe"])
    base_ir = float(base_row["information_ratio"])
    base_dd = float(base_row["max_drawdown"])
    base_turnover = float(base_row["annualized_turnover"])

    candidates = test[test["variant"] != baseline_variant].copy()
    passed: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        gates = {
            "sharpe_ok": float(row["sharpe"]) >= GATE_SHARPE_MIN,
            "return_ok": float(row["annualized_return"]) >= GATE_RETURN_MIN,
            "ir_ok": float(row["information_ratio"]) > GATE_IR_MIN
            and float(row["information_ratio"]) > base_ir,
            "dd_ok": float(row["max_drawdown"]) >= base_dd - GATE_DD_WORSE_THAN_BASELINE_PP,
            "turnover_ok": float(row["annualized_turnover"]) <= base_turnover * (1 + GATE_TURNOVER_INCREASE_MAX),
        }
        if all(gates.values()):
            passed.append({**row.to_dict(), "gates": gates})

    if not passed:
        best_ir = candidates.sort_values("information_ratio", ascending=False).iloc[0]
        return {
            "winner": None,
            "verdict": "reject",
            "reason": "No variant passed all adoption gates on test period",
            "baseline": base_row.to_dict(),
            "best_ir_variant": best_ir.to_dict(),
            "gates": {
                "sharpe_min": GATE_SHARPE_MIN,
                "return_min": GATE_RETURN_MIN,
                "ir_min": GATE_IR_MIN,
                "dd_worse_than_baseline_pp": GATE_DD_WORSE_THAN_BASELINE_PP,
                "turnover_increase_max": GATE_TURNOVER_INCREASE_MAX,
            },
        }

    winner = max(
        passed,
        key=lambda x: (x["information_ratio"], x["sharpe"], x["annualized_return"]),
    )
    return {
        "winner": winner["variant"],
        "verdict": "adopt",
        "winner_metrics": winner,
        "baseline": base_row.to_dict(),
        "gates": {
            "sharpe_min": GATE_SHARPE_MIN,
            "return_min": GATE_RETURN_MIN,
            "ir_min": GATE_IR_MIN,
        },
    }
