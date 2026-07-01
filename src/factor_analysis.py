"""M1 factor-level performance, correlation, and sleeve attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestResult, _run_backtest
from src.config import PipelineConfig
from src.diagnostics import annualized_return, compute_ic, max_drawdown, sharpe_ratio, strategy_metrics
from src.model_m1 import RuleBasedM1

FACTOR_COLS = ["momentum_score", "trend_score", "macro_score", "risk_penalty"]

UNDERLYING_M1_FEATURES = [
    "z_mom_12w",
    "z_mom_26w",
    "z_mom_52w",
    "rank_mom_12w",
    "rel_mom_12w",
    "z_trend_signal",
    "z_vol_12w",
    "drawdown_26w",
    "z_drawdown_26w",
    "growth_trend",
    "risk_off",
    "inflation_up",
    "curve_inverted",
    "credit_stress",
]


def _filter_panel_period(
    panel: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = panel.copy()
    if not isinstance(df.index, pd.MultiIndex):
        return df
    dates = pd.to_datetime(df.index.get_level_values("date"))
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return df[mask.values]


def compute_factor_ic(
    panel: pd.DataFrame,
    *,
    period_label: str = "full",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Spearman IC of each M1 component vs forward return."""
    sub = _filter_panel_period(panel, start=start, end=end)
    fwd_cols = [c for c in sub.reset_index().columns if c.startswith("forward_return")]
    if not fwd_cols:
        return pd.DataFrame()
    fwd_col = fwd_cols[0]
    rows = []
    for col in FACTOR_COLS:
        if col not in sub.columns:
            continue
        tmp = sub.copy()
        tmp["M1_score"] = tmp[col]
        ic_series = compute_ic(tmp, score_col="M1_score", fwd_col=fwd_col)
        rows.append(
            {
                "period": period_label,
                "factor": col,
                "ic_mean": float(ic_series.mean()) if not ic_series.empty else float("nan"),
                "ic_std": float(ic_series.std()) if not ic_series.empty else float("nan"),
                "ic_hit_rate": float((ic_series > 0).mean()) if not ic_series.empty else float("nan"),
                "n_weeks": int(len(ic_series)),
            }
        )
    # Composite M1 score for reference
    if "M1_score" in sub.columns:
        ic_series = compute_ic(sub, score_col="M1_score", fwd_col=fwd_col)
        rows.append(
            {
                "period": period_label,
                "factor": "M1_score",
                "ic_mean": float(ic_series.mean()) if not ic_series.empty else float("nan"),
                "ic_std": float(ic_series.std()) if not ic_series.empty else float("nan"),
                "ic_hit_rate": float((ic_series > 0).mean()) if not ic_series.empty else float("nan"),
                "n_weeks": int(len(ic_series)),
            }
        )
    return pd.DataFrame(rows)


def compute_factor_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FACTOR_COLS if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return panel[cols].corr(method="pearson")


def compute_factor_covariance(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FACTOR_COLS if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return panel[cols].cov()


def compute_underlying_feature_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in UNDERLYING_M1_FEATURES if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return panel[cols].corr(method="pearson")


def _backtest_from_score(
    panel: pd.DataFrame,
    score: pd.Series,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    name: str = "sleeve",
) -> BacktestResult:
    from src.model_m1 import RuleBasedM1
    from src.portfolio import apply_vol_target_wide, build_weights_from_signals

    rb = RuleBasedM1(cfg.m1)
    signals = rb._signals_top_k(score.rename("M1_score"))
    w = build_weights_from_signals(panel, signals, portfolio_cfg=cfg.portfolio)
    w = apply_vol_target_wide(w, returns_wide, cfg.portfolio)
    return _run_backtest(name, w, returns_wide, cfg.portfolio.transaction_cost_bps)


def factor_sleeve_summary(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Backtest top-K sleeves driven by each component score alone."""
    bench_key = "equal_weight_1_7"
    rows = []
    sleeve_returns: dict[str, pd.Series] = {}

    if "M1_score" in panel.columns:
        full_bt = _backtest_from_score(panel, panel["M1_score"], returns_wide, cfg, name="full_m1")
        sleeve_returns["full_m1"] = full_bt.returns
        m = strategy_metrics(full_bt, None)
        rows.append({"sleeve": "full_m1", **m})

    for col in FACTOR_COLS:
        if col not in panel.columns:
            continue
        score = panel[col]
        if col == "risk_penalty":
            score = -score
        bt = _backtest_from_score(panel, score, returns_wide, cfg, name=col)
        sleeve_returns[col] = bt.returns
        m = strategy_metrics(bt, None)
        rows.append({"sleeve": col, **m})

    interaction = factor_interaction_term(sleeve_returns, panel, returns_wide, cfg)
    if interaction is not None:
        rows.append(interaction)
    return pd.DataFrame(rows)


def factor_interaction_term(
    sleeve_returns: dict[str, pd.Series],
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, Any] | None:
    """Combined M1 excess minus sum of standalone factor sleeve excess returns."""
    if "full_m1" not in sleeve_returns:
        return None
    from src.backtest import equal_weight_returns

    bench = equal_weight_returns(returns_wide, cfg.assets.tickers)
    combined_excess = annualized_return(sleeve_returns["full_m1"]) - annualized_return(bench.returns)
    standalone_cols = [c for c in FACTOR_COLS if c in sleeve_returns]
    if not standalone_cols:
        return None
    sum_excess = sum(
        annualized_return(sleeve_returns[c]) - annualized_return(bench.returns) for c in standalone_cols
    )
    interaction = combined_excess - sum_excess
    return {
        "sleeve": "interaction",
        "annualized_return": float("nan"),
        "annualized_volatility": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "interaction_excess_ann": float(interaction),
        "combined_excess_ann": float(combined_excess),
        "sum_standalone_excess_ann": float(sum_excess),
    }


def factor_ablation_summary(
    m1: RuleBasedM1,
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Zero out one factor weight at a time and compare portfolio metrics."""
    from copy import deepcopy

    X = panel[feature_cols].fillna(0)
    rows = []
    base_bt = _backtest_from_score(panel, m1.predict_score(X), returns_wide, cfg, name="full")
    rows.append({"variant": "full_m1", **strategy_metrics(base_bt, None)})

    weight_keys = {
        "ablate_momentum": "momentum",
        "ablate_trend": "trend",
        "ablate_macro": "macro",
        "ablate_risk_penalty": "risk_penalty",
    }
    comps = m1.predict_component_scores(X)
    w = m1.weights
    for variant, key in weight_keys.items():
        weights = dict(w)
        weights[key] = 0.0
        score = (
            weights.get("momentum", 0.0) * comps["momentum_score"]
            + weights.get("trend", 0.0) * comps["trend_score"]
            + weights.get("macro", 0.0) * comps["macro_score"]
            - weights.get("risk_penalty", 0.0) * comps["risk_penalty"]
        )
        bt = _backtest_from_score(panel, score, returns_wide, cfg, name=variant)
        rows.append({"variant": variant, **strategy_metrics(bt, None)})
    return pd.DataFrame(rows)


def save_factor_charts(
    factor_ic: pd.DataFrame,
    corr: pd.DataFrame,
    sleeves: pd.DataFrame,
    sleeve_returns: dict[str, pd.Series] | None,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if not factor_ic.empty:
        test_ic = factor_ic[factor_ic["period"] == "test"]
        if test_ic.empty:
            test_ic = factor_ic
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = test_ic[test_ic["factor"] != "M1_score"]
        ax.barh(plot_df["factor"], plot_df["ic_mean"], color="#4C72B0")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Mean Spearman IC")
        ax.set_title("M1 Factor IC (test period)")
        p = output_dir / "m1_factor_ic.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if not corr.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        ax.set_title("M1 Factor Correlation")
        fig.colorbar(im, ax=ax)
        p = output_dir / "m1_factor_correlation_heatmap.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if sleeve_returns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, rets in sleeve_returns.items():
            if name == "interaction":
                continue
            cum = (1 + rets.fillna(0)).cumprod()
            ax.plot(cum.index, cum.values, label=name)
        ax.legend(fontsize=8)
        ax.set_title("M1 Factor Sleeve Cumulative Returns")
        p = output_dir / "m1_factor_sleeves_cumulative.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    return saved


def _collect_sleeve_returns(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if "M1_score" in panel.columns:
        out["full_m1"] = _backtest_from_score(panel, panel["M1_score"], returns_wide, cfg).returns
    for col in FACTOR_COLS:
        if col not in panel.columns:
            continue
        score = -panel[col] if col == "risk_penalty" else panel[col]
        out[col] = _backtest_from_score(panel, score, returns_wide, cfg).returns
    return out


def run_factor_analysis(
    panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    output_dir: Path,
    *,
    m1_model: object | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Run full M1 factor diagnostics and persist CSVs/charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ic_frames = [
        compute_factor_ic(panel, period_label="full"),
        compute_factor_ic(train_panel, period_label="train", start=cfg.split.train_start, end=cfg.split.train_end),
        compute_factor_ic(test_panel, period_label="test", start=cfg.split.test_start, end=cfg.split.test_end),
    ]
    factor_ic = pd.concat([f for f in ic_frames if not f.empty], ignore_index=True)
    factor_ic.to_csv(output_dir / "m1_factor_ic.csv", index=False)

    corr = compute_factor_correlation(panel)
    if not corr.empty:
        corr.to_csv(output_dir / "m1_factor_correlation.csv")
    cov = compute_factor_covariance(panel)
    if not cov.empty:
        cov.to_csv(output_dir / "m1_factor_covariance.csv")
    feat_corr = compute_underlying_feature_correlation(panel)
    if not feat_corr.empty:
        feat_corr.to_csv(output_dir / "m1_underlying_feature_correlation.csv")

    sleeves = factor_sleeve_summary(panel, returns_wide, cfg)
    sleeves.to_csv(output_dir / "m1_factor_sleeves.csv", index=False)

    ablation = pd.DataFrame()
    if isinstance(m1_model, RuleBasedM1) and feature_cols:
        ablation = factor_ablation_summary(m1_model, panel, returns_wide, cfg, feature_cols)
        ablation.to_csv(output_dir / "m1_factor_ablation.csv", index=False)

    sleeve_returns = _collect_sleeve_returns(panel, returns_wide, cfg)
    charts = save_factor_charts(factor_ic, corr, sleeves, sleeve_returns, figures_dir)

    return {
        "factor_ic": factor_ic,
        "factor_correlation": corr,
        "factor_covariance": cov,
        "underlying_feature_correlation": feat_corr,
        "factor_sleeves": sleeves,
        "factor_ablation": ablation,
        "charts": charts,
    }
