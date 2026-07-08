"""Backtest diagnostics and reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.backtest import BacktestResult, METRICS_TABLE_STRATEGIES
from src.config import PipelineConfig

logger = logging.getLogger(__name__)

WEEKS_PER_YEAR = 52


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    cum = (1 + r).prod()
    years = len(r) / WEEKS_PER_YEAR
    if years <= 0:
        return 0.0
    return float(cum ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(r.std() * np.sqrt(WEEKS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    vol = annualized_volatility(returns)
    if vol == 0:
        return 0.0
    ann = annualized_return(returns) - rf
    return float(ann / vol)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def rolling_max_drawdown(returns: pd.Series, window: int = 52) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    roll_max = equity.rolling(window, min_periods=1).max()
    return equity / roll_max - 1


def information_ratio(strategy: pd.Series, benchmark: pd.Series) -> float:
    active = strategy - benchmark
    std = active.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(active.mean() * np.sqrt(WEEKS_PER_YEAR) / std)


def compute_per_asset_ic(
    panel: pd.DataFrame,
    *,
    score_col: str = "M1_score",
    fwd_col: str | None = None,
) -> pd.DataFrame:
    """Spearman IC between M1 score and forward return, per ticker."""
    df = panel.reset_index()
    if fwd_col is None:
        fwd_cols = [c for c in df.columns if c.startswith("forward_return")]
        fwd_col = fwd_cols[0] if fwd_cols else None
    if fwd_col is None or "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "ic", "n_obs", "hit_rate"])

    rows = []
    for ticker, grp in df.groupby("ticker"):
        sub = grp[[score_col, fwd_col]].dropna()
        if len(sub) < 10:
            continue
        ic = sub[score_col].corr(sub[fwd_col], method="spearman")
        active = grp[grp["M1_signal"] != 0] if "M1_signal" in grp.columns else sub
        hit = float((active[fwd_col] > 0).mean()) if len(active) > 0 and fwd_col in active.columns else float("nan")
        rows.append({"ticker": ticker, "ic": ic, "n_obs": len(sub), "hit_rate": hit})
    return pd.DataFrame(rows)


def analyze_m1_exposure(
    result: BacktestResult,
    benchmark: BacktestResult | None = None,
) -> dict[str, Any]:
    """Gross exposure, cash weight, and active-share style stats for M1-only weights."""
    w = result.weights
    gross = w.abs().sum(axis=1)
    cash = (1.0 - gross).clip(lower=0.0)
    long_gross = w.clip(lower=0).sum(axis=1)
    short_gross = (-w.clip(upper=0)).sum(axis=1)
    n_active = (w.abs() > 1e-8).sum(axis=1)

    stats = {
        "mean_gross_exposure": float(gross.mean()),
        "median_gross_exposure": float(gross.median()),
        "mean_cash_weight": float(cash.mean()),
        "mean_long_gross": float(long_gross.mean()),
        "mean_short_gross": float(short_gross.mean()),
        "mean_active_names": float(n_active.mean()),
        "pct_weeks_below_half_invested": float((gross < 0.5).mean()),
    }

    if benchmark is not None:
        bench_gross = benchmark.weights.abs().sum(axis=1)
        stats["mean_gross_vs_benchmark"] = float(gross.mean() - bench_gross.mean())
        strat_r = result.returns.fillna(0)
        bench_r = benchmark.returns.reindex(strat_r.index).fillna(0)
        # Correlation of weight changes vs benchmark as crude active share proxy
        w_diff = w.diff().abs().sum(axis=1).fillna(0)
        stats["mean_weekly_turnover"] = float(w_diff.mean())
        stats["return_correlation_vs_benchmark"] = float(strat_r.corr(bench_r))

    return {
        "summary": stats,
        "gross_exposure": gross.rename("gross_exposure"),
        "cash_weight": cash.rename("cash_weight"),
    }


def threshold_sensitivity_summary(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    period_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Evaluate M1 threshold quantiles on a period (train for tuning charts)."""
    from src.backtest import _run_backtest
    from src.model_m1 import _signals_from_thresholds
    from src.portfolio import apply_vol_target_wide, build_weights_from_signals

    df = panel.copy()
    if period_mask is not None:
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values("date")
            df = df[period_mask.reindex(dates).fillna(False).values]
        else:
            df = df[period_mask.values]

    if "M1_score" not in df.columns:
        return pd.DataFrame()

    scores = df["M1_score"]
    score_vals = scores.dropna()
    if score_vals.empty:
        return pd.DataFrame()

    rows = []
    for long_q in np.arange(cfg.m1.long_quantile_min, cfg.m1.long_quantile_max + 1e-9, cfg.m1.quantile_step):
        long_t = float(score_vals.quantile(long_q))
        short_t = float(score_vals.min() - 1.0)
        sig = _signals_from_thresholds(scores, long_t, short_t, cfg.m1.allow_short)
        w = build_weights_from_signals(df, sig, portfolio_cfg=cfg.portfolio)
        w = apply_vol_target_wide(w, returns_wide, cfg.portfolio)
        bt = _run_backtest("sens", w, returns_wide, cfg.portfolio.transaction_cost_bps)
        rows.append(
            {
                "long_quantile": round(long_q, 2),
                "long_threshold": long_t,
                "annualized_return": annualized_return(bt.returns),
                "sharpe": sharpe_ratio(bt.returns),
                "max_drawdown": max_drawdown(bt.returns),
                "mean_gross": float(w.abs().sum(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def save_m1_exposure_charts(exposure: dict[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    gross = exposure.get("gross_exposure")
    if gross is None or gross.empty:
        return saved

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(gross.index, gross.values, color="#8172B3", linewidth=1.5)
    axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="100% gross")
    axes[0].set_ylabel("Gross exposure")
    axes[0].set_title("M1 Portfolio Gross Exposure Over Time", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    cash = exposure.get("cash_weight")
    if cash is not None and not cash.empty:
        axes[1].fill_between(cash.index, cash.values, alpha=0.4, color="#C44E52")
        axes[1].set_ylabel("Implied cash weight")
    axes[1].set_title("Uninvested Capital (1 − gross exposure)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    p = output_dir / "m1_exposure_over_time.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)
    return saved


def save_threshold_sensitivity_chart(sens_df: pd.DataFrame, output_dir: Path) -> str | None:
    if sens_df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sens_df["long_quantile"], sens_df["sharpe"], "o-", color="#8172B3", label="Sharpe")
    ax1.set_xlabel("Long signal quantile (train)")
    ax1.set_ylabel("Sharpe", color="#8172B3")
    ax1.tick_params(axis="y", labelcolor="#8172B3")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sens_df["long_quantile"], sens_df["annualized_return"] * 100, "s--", color="#C44E52", label="Ann. return %")
    ax2.set_ylabel("Ann. return (%)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax1.set_title("M1 Threshold Sensitivity (train period)", fontweight="bold")
    p = output_dir / "m1_threshold_sensitivity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p.name


def build_m1_exposure_report_section(
    exposure_analysis: dict[str, Any] | None,
    per_asset_ic: pd.DataFrame | None,
    *,
    chart_rel: str | None = None,
    sens_chart_rel: str | None = None,
) -> list[str]:
    if exposure_analysis is None:
        return []
    stats = exposure_analysis.get("summary", {})
    lines = [
        "## M1 Exposure & Signal Quality Diagnostics",
        "",
        "Understanding **how much capital M1 deploys** versus benchmark buy-and-hold helps separate "
        "low return from low edge.",
        "",
        "### Portfolio Exposure (M1 only)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean gross exposure | {_fmt_pct(stats.get('mean_gross_exposure', float('nan')))} |",
        f"| Median gross exposure | {_fmt_pct(stats.get('median_gross_exposure', float('nan')))} |",
        f"| Mean implied cash (1 − gross) | {_fmt_pct(stats.get('mean_cash_weight', float('nan')))} |",
        f"| Mean active names per week | {_fmt_num(stats.get('mean_active_names', float('nan')), 2)} |",
        f"| Weeks below 50% invested | {_fmt_pct(stats.get('pct_weeks_below_half_invested', float('nan')))} |",
    ]
    if "mean_gross_vs_benchmark" in stats:
        lines.append(f"| Mean gross vs equal-weight | {_fmt_pct(stats.get('mean_gross_vs_benchmark', float('nan')))} |")
    lines.append("")
    if chart_rel:
        lines.extend([f"![M1 exposure over time]({chart_rel})", ""])

    if per_asset_ic is not None and not per_asset_ic.empty:
        lines.extend(
            [
                "### Per-Asset IC (M1 score vs forward return)",
                "",
                "| Ticker | IC | Observations | Hit rate (active) |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in per_asset_ic.sort_values("ic", ascending=False).iterrows():
            lines.append(
                f"| {row['ticker']} | {_fmt_num(row['ic'])} | {int(row['n_obs'])} | "
                f"{_fmt_pct(row['hit_rate']) if not np.isnan(row['hit_rate']) else '—'} |"
            )
        lines.append("")

    if sens_chart_rel:
        lines.extend(
            [
                "### Threshold sensitivity (train period)",
                "",
                f"![Threshold sensitivity]({sens_chart_rel})",
                "",
            ]
        )
    return lines


def compute_ic(panel: pd.DataFrame, score_col: str = "M1_score", fwd_col: str | None = None) -> pd.Series:
    df = panel.reset_index()
    if fwd_col is None:
        fwd_cols = [c for c in df.columns if c.startswith("forward_return")]
        fwd_col = fwd_cols[0] if fwd_cols else None
    if fwd_col is None:
        return pd.Series(dtype=float)
    ics = []
    dates = []
    for date, grp in df.groupby("date"):
        if grp[score_col].notna().sum() < 2:
            continue
        ic = grp[score_col].corr(grp[fwd_col], method="spearman")
        ics.append(ic)
        dates.append(date)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="IC")


def strategy_metrics(result: BacktestResult, benchmark: BacktestResult | None = None) -> dict[str, float]:
    r = result.returns
    m: dict[str, float] = {
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "rolling_12m_max_drawdown": float(rolling_max_drawdown(r, 52).min()),
        "turnover": float(result.turnover.mean()),
        "annualized_turnover": float(result.turnover.mean() * WEEKS_PER_YEAR),
        "hit_rate": float((r > 0).mean()),
    }
    if benchmark is not None:
        m["excess_return_vs_benchmark"] = m["annualized_return"] - annualized_return(benchmark.returns)
        m["information_ratio"] = information_ratio(r, benchmark.returns)
    return m


def strategy_metrics_on_period(
    returns: pd.Series,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, float]:
    """Compute strategy metrics on a date-filtered return slice."""
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    if r.empty:
        return {
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "n_weeks": 0,
        }
    return {
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "n_weeks": int(len(r)),
    }


def m2_classification_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, Any]:
    mask = y_true.notna() & y_prob.notna()
    y = y_true[mask].astype(int)
    p = y_prob[mask]
    if len(y) == 0:
        return {}
    pred = (p >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "base_rate": float(y.mean()),
        "n_trades": int(len(y)),
        "mean_p_winners": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_p_losers": float(p[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mae": float(np.abs(p - y).mean()),
        "mean_diff": float((p - y).mean()),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        metrics["auc"] = float("nan")
    try:
        metrics["auc_pr"] = float(average_precision_score(y, p))
    except ValueError:
        metrics["auc_pr"] = float("nan")
    if metrics.get("recall", 0) >= 0.999 and metrics.get("precision", 0) > 0:
        metrics["degeneracy_note"] = (
            "Binary M3 at this threshold approves all trades; strategy equals M1-only."
        )
    return metrics


def m2_calibration_table(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> pd.DataFrame:
    mask = y_true.notna() & y_prob.notna()
    df = pd.DataFrame({"y": y_true[mask].astype(int), "p": y_prob[mask]})
    if df.empty or df["p"].nunique() < 2:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["p"], q=min(bins, df["p"].nunique()), duplicates="drop")
    g = df.groupby("bucket", observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_pred": g["p"].mean(),
            "realized": g["y"].mean(),
        }
    ).reset_index()


def m2_probability_decile_returns(panel: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    trades = df[df["M1_signal"] != 0].dropna(subset=["p_success", "trade_return"])
    if trades.empty or trades["p_success"].nunique() < 2:
        return pd.DataFrame()
    trades = trades.copy()
    trades["decile"] = pd.qcut(trades["p_success"], q=min(bins, trades["p_success"].nunique()), duplicates="drop")
    g = trades.groupby("decile", observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_p_success": g["p_success"].mean(),
            "mean_trade_return": g["trade_return"].mean(),
            "hit_rate": g["meta_label"].mean(),
        }
    ).reset_index()


def m2_feature_importance(m2_model: object, panel: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    from src.model_m2 import resolve_m2_for_importance

    resolved = resolve_m2_for_importance(m2_model)
    if resolved is None or resolved.pipeline is None:
        return pd.DataFrame()
    if cfg.m2.type != "logistic_regression":
        return pd.DataFrame()

    pipe = resolved.pipeline
    estimators: list = []
    if hasattr(pipe, "calibrated_classifiers_"):
        estimators = [cc.estimator for cc in pipe.calibrated_classifiers_]
    else:
        estimators = [pipe]

    coefs = []
    for est in estimators:
        if not hasattr(est, "named_steps") or "clf" not in est.named_steps:
            continue
        clf = est.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            continue
        coefs.append(clf.coef_.ravel())
    if not coefs:
        return pd.DataFrame()

    mean_coef = np.mean(coefs, axis=0)

    feature_names = list(resolved.feature_cols)

    if len(feature_names) != len(mean_coef):
        logger.warning(
            "M2 feature importance length mismatch: %s feature names vs %s coefficients. "
            "Using aligned/generated feature names for diagnostics.",
            len(feature_names),
            len(mean_coef),
        )

        if len(feature_names) > len(mean_coef):
            feature_names = feature_names[: len(mean_coef)]
        else:
            feature_names = feature_names + [
                f"derived_feature_{i}"
                for i in range(len(feature_names), len(mean_coef))
            ]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": mean_coef,
            "abs_coefficient": np.abs(mean_coef),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return importance.head(15).reset_index(drop=True)


def m2_metrics_by_dimension(
    panel: pd.DataFrame,
    dimension: str,
    threshold: float = 0.55,
) -> pd.DataFrame:
    df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    trades = df[(df["M1_signal"] != 0) & df["meta_label"].notna() & df["p_success"].notna()]
    if trades.empty or dimension not in trades.columns:
        return pd.DataFrame()

    rows = []
    for val, grp in trades.groupby(dimension):
        m = m2_classification_metrics(grp["meta_label"], grp["p_success"], threshold)
        if not m:
            continue
        rows.append(
            {
                dimension: val,
                "n_trades": m.get("n_trades", len(grp)),
                "base_rate": m.get("base_rate", float("nan")),
                "auc": m.get("auc", float("nan")),
                "approval_rate": float((grp["p_success"] >= threshold).mean()),
                "mean_trade_return": float(grp["trade_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_m2_deep_charts(
    y_true: pd.Series,
    y_prob: pd.Series,
    calibration: pd.DataFrame,
    decile_returns: pd.DataFrame,
    feature_importance: pd.DataFrame,
    m2_metrics: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    mask = y_true.notna() & y_prob.notna()
    y = y_true[mask].astype(int)
    p = y_prob[mask]
    if len(y) > 0 and y.nunique() >= 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, color="#4C72B0", label=f"AUC={m2_metrics.get('auc', float('nan')):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (0.50)")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("M2 ROC Curve (Test)")
        axes[0].legend(fontsize=8)

        if not calibration.empty:
            axes[1].plot(calibration["mean_pred"], calibration["realized"], "o-", color="#55A868")
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[1].set_xlabel("Mean Predicted P(success)")
            axes[1].set_ylabel("Realized Success Rate")
            axes[1].set_title("Calibration Reliability")
        else:
            axes[1].set_title("Calibration (insufficient buckets)")
        p = figures_dir / "m2_roc_calibration.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if not feature_importance.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        top = feature_importance.head(15)
        colors = ["#55A868" if c > 0 else "#C44E52" for c in top["coefficient"]]
        ax.barh(top["feature"], top["coefficient"], color=colors)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Standardized coefficient")
        ax.set_title("M2 Feature Importance (logistic regression)")
        ax.invert_yaxis()
        p = figures_dir / "m2_feature_importance.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    if not decile_returns.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(decile_returns))
        ax.bar(x, decile_returns["mean_trade_return"] * 100, color="#8172B3")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"D{i+1}" for i in x], fontsize=8)
        ax.set_ylabel("Mean trade return (%)")
        ax.set_title("Realized Return by M2 Probability Decile")
        p = figures_dir / "m2_decile_returns.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    return saved


def run_m2_deep_diagnostics(
    test_panel: pd.DataFrame,
    m2_model: object | None,
    cfg: PipelineConfig,
    threshold: float,
    output_dir: Path,
    *,
    train_panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    y_true = test_panel["meta_label"]
    y_prob = test_panel["p_success"]

    m2_metrics = m2_classification_metrics(y_true, y_prob, threshold=threshold)
    calibration = m2_calibration_table(y_true, y_prob)
    if not calibration.empty:
        calibration.to_csv(output_dir / "m2_calibration_table.csv", index=False)

    decile_returns = m2_probability_decile_returns(test_panel)
    if not decile_returns.empty:
        decile_returns.to_csv(output_dir / "m2_decile_returns.csv", index=False)

    feature_importance = pd.DataFrame()
    if m2_model is not None:
        feature_importance = m2_feature_importance(m2_model, test_panel, cfg)
        if not feature_importance.empty:
            feature_importance.to_csv(output_dir / "m2_feature_importance.csv", index=False)

    by_asset = m2_metrics_by_dimension(test_panel, "ticker", threshold)
    if not by_asset.empty:
        by_asset.to_csv(output_dir / "m2_metrics_by_asset.csv", index=False)

    regime_dims = ["risk_off", "curve_inverted", "inflation_up", "growth_down"]
    by_regime_frames = []
    for dim in regime_dims:
        if dim in test_panel.reset_index().columns:
            by_regime_frames.append(m2_metrics_by_dimension(test_panel, dim, threshold))
    by_regime = pd.concat(by_regime_frames, ignore_index=True) if by_regime_frames else pd.DataFrame()
    if not by_regime.empty:
        by_regime.to_csv(output_dir / "m2_metrics_by_regime.csv", index=False)

    charts = save_m2_deep_charts(
        y_true, y_prob, calibration, decile_returns, feature_importance, m2_metrics, figures_dir
    )

    architecture_benchmark = pd.DataFrame()
    if train_panel is not None and m2_model is not None:
        from src.model_m2 import m2_architecture_benchmark

        architecture_benchmark = m2_architecture_benchmark(train_panel, test_panel, cfg)
        if not architecture_benchmark.empty:
            architecture_benchmark.to_csv(output_dir / "m2_architecture_benchmark.csv", index=False)

    return {
        "m2_metrics": m2_metrics,
        "calibration_table": calibration,
        "decile_returns": decile_returns,
        "feature_importance": feature_importance,
        "metrics_by_asset": by_asset,
        "metrics_by_regime": by_regime,
        "architecture_benchmark": architecture_benchmark,
        "charts": charts,
    }



def build_metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    bench = results.get("equal_weight_1_7")
    rows = []
    for name in METRICS_TABLE_STRATEGIES:
        res = results.get(name)
        if res is None:
            continue
        row = {"strategy": name, **strategy_metrics(res, bench)}
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics_table_on_period(
    results: dict[str, BacktestResult],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build strategy metrics table for a date-filtered period."""
    bench = results.get("equal_weight_1_7")
    bench_returns = None
    if bench is not None:
        bench_returns = bench.returns.copy()
        bench_returns.index = pd.to_datetime(bench_returns.index)
        if start is not None:
            bench_returns = bench_returns[bench_returns.index >= pd.Timestamp(start)]
        if end is not None:
            bench_returns = bench_returns[bench_returns.index <= pd.Timestamp(end)]

    rows = []
    for name in METRICS_TABLE_STRATEGIES:
        res = results.get(name)
        if res is None:
            continue
        row = {"strategy": name, **strategy_metrics_on_period(res.returns, start=start, end=end)}
        if bench_returns is not None:
            r = res.returns.copy()
            r.index = pd.to_datetime(r.index)
            if start is not None:
                r = r[r.index >= pd.Timestamp(start)]
            if end is not None:
                r = r[r.index <= pd.Timestamp(end)]
            row["excess_return_vs_benchmark"] = row["annualized_return"] - annualized_return(bench_returns)
            row["information_ratio"] = information_ratio(r, bench_returns)
        rows.append(row)
    return pd.DataFrame(rows)


STRATEGY_LABELS: dict[str, str] = {
    "equal_weight_1_7": "Equal Weight (1/7)",
    "sixty_forty": "60/40 Benchmark",
    "m1_only": "M1 Only",
    "m1_m2_m3_binary": "M1 + M2 + M3 (Binary threshold)",
    "m1_m2_m3_linear": "M1 + M2 + M3 (Linear)",
    "m1_m2_m3_ecdf": "M1 + M2 + M3 (ECDF)",
    "m1_m2_passthrough": "M1 + M2 + M3 (Passthrough diagnostic)",
    "m1_m2_binary": "M1 + M2 + M3 (Binary threshold)",
    "m1_m2_linear": "M1 + M2 + M3 (Linear)",
    "m1_m2_ecdf": "M1 + M2 + M3 (ECDF)",
}

REPORT_CHART_STRATEGIES = [
    "equal_weight_1_7",
    "sixty_forty",
    "m1_only",
    "m1_m2_m3_linear",
    "m1_m2_m3_ecdf",
]


def _strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def _fmt_pct(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def _fmt_num(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{decimals}f}"


def format_metrics_table_for_report(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Return a display-friendly metrics table with readable labels and rounding."""
    df = metrics_table.copy()
    df["strategy"] = df["strategy"].map(_strategy_label)
    display = pd.DataFrame(
        {
            "Strategy": df["strategy"],
            "Ann. Return": df["annualized_return"].map(lambda x: _fmt_pct(x)),
            "Ann. Volatility": df["annualized_volatility"].map(lambda x: _fmt_pct(x)),
            "Sharpe": df["sharpe"].map(lambda x: _fmt_num(x)),
            "Max Drawdown": df["max_drawdown"].map(lambda x: _fmt_pct(x)),
            "Excess vs EW": df["excess_return_vs_benchmark"].map(lambda x: _fmt_pct(x)),
            "Info Ratio": df["information_ratio"].map(lambda x: _fmt_num(x)),
            "Weekly Hit Rate": df["hit_rate"].map(lambda x: _fmt_pct(x)),
        }
    )
    return display


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def _build_executive_summary(metrics_table: pd.DataFrame, m2_metrics: dict[str, Any]) -> list[str]:
    raw = metrics_table.set_index("strategy")
    ew = raw.loc["equal_weight_1_7"] if "equal_weight_1_7" in raw.index else None
    m1 = raw.loc["m1_only"] if "m1_only" in raw.index else None
    m2_linear = None
    if "m1_m2_m3_linear" in raw.index:
        m2_linear = raw.loc["m1_m2_m3_linear"]
    elif "m1_m2_linear" in raw.index:
        m2_linear = raw.loc["m1_m2_linear"]

    lines = [
        "This report compares a **meta-labeling pipeline** against standard benchmarks on seven global index sleeves "
        "(SP500, MSCI_EAFE, MSCI_EM, UST_7_10, US_HIGH_YIELD, GOLD_SPOT, US_REIT). M1 proposes long/short/flat signals; M2 estimates trade quality "
        "and scales position size.",
        "",
        "**Research use only — not investment advice.**",
        "",
    ]

    if ew is not None:
        lines.append(
            f"- The **equal-weight benchmark** returned {_fmt_pct(ew['annualized_return'])} per year "
            f"with Sharpe {_fmt_num(ew['sharpe'])} and max drawdown {_fmt_pct(ew['max_drawdown'])}."
        )
    if m1 is not None:
        lines.append(
            f"- **M1 alone** produced lower return ({_fmt_pct(m1['annualized_return'])}) but also lower volatility "
            f"({_fmt_pct(m1['annualized_volatility'])}) than buy-and-hold benchmarks."
        )
    if m2_linear is not None and m1 is not None:
        vol_drop = m1["annualized_volatility"] - m2_linear["annualized_volatility"]
        lines.append(
            f"- **M1 + M2 + M3 (linear sizing)** improved risk-adjusted metrics: Sharpe {_fmt_num(m2_linear['sharpe'])} "
            f"vs {_fmt_num(m1['sharpe'])} for M1-only, with max drawdown {_fmt_pct(m2_linear['max_drawdown'])}. "
            "Much of the improvement comes from **lower exposure**, not higher raw returns."
        )
        if vol_drop > 0:
            lines.append(
                f"- M2 filtering reduced annualized volatility by roughly {_fmt_pct(vol_drop)} relative to M1-only."
            )
        lines.append(
            f"- Max drawdown moved from {_fmt_pct(m1['max_drawdown'])} (M1-only) to "
            f"{_fmt_pct(m2_linear['max_drawdown'])} (M1 + M2 linear)."
        )

    if m2_metrics:
        recall = m2_metrics.get("recall", float("nan"))
        precision = m2_metrics.get("precision", float("nan"))
        auc = m2_metrics.get("auc", float("nan"))
        lines.extend(
            [
                "",
                "**M2 meta-labeling (out-of-sample test period):**",
                f"- Precision {_fmt_num(precision)} — when M2 approves a trade, about {_fmt_pct(precision)} are profitable.",
                f"- Recall {_fmt_num(recall)} — M2 approves {_fmt_pct(recall)} of truly profitable M1 signals.",
                f"- AUC {_fmt_num(auc)} — modest discrimination between winning and losing trades.",
            ]
        )

    return lines


def save_report_charts(
    results: dict[str, BacktestResult],
    m2_metrics: dict[str, Any],
    reports_dir: Path,
    *,
    subdir: str | None = None,
) -> list[str]:
    """Create presentation-ready charts saved alongside final_report.md."""
    out_dir = reports_dir / subdir if subdir else reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir
    saved: list[str] = []
    palette = {
        "equal_weight_1_7": "#4C72B0",
        "sixty_forty": "#55A868",
        "m1_only": "#C44E52",
        "m1_m2_m3_linear": "#8172B3",
        "m1_m2_m3_ecdf": "#CCB974",
        "m1_m2_linear": "#8172B3",
        "m1_m2_ecdf": "#CCB974",
    }

    # 1. Cumulative returns (key strategies only)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in REPORT_CHART_STRATEGIES:
        if name not in results:
            continue
        cum = (1 + results[name].returns.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=_strategy_label(name), color=palette.get(name), linewidth=2)
    ax.set_title("Cumulative Growth of $1", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio value")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    p = reports_dir / "strategy_cumulative_returns.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 2. Drawdown (key strategies)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in REPORT_CHART_STRATEGIES:
        if name not in results:
            continue
        eq = (1 + results[name].returns.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, label=_strategy_label(name), color=palette.get(name), linewidth=1.8)
    ax.set_title("Drawdown Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    p = reports_dir / "strategy_drawdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 3. Sharpe comparison bar chart
    metrics = build_metrics_table(results)
    metrics["label"] = metrics["strategy"].map(_strategy_label)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [palette.get(s, "#888888") for s in metrics["strategy"]]
    ax.bar(metrics["label"], metrics["sharpe"], color=colors, edgecolor="white")
    ax.set_title("Sharpe Ratio by Strategy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(metrics["sharpe"]):
        ax.text(i, v + 0.02, _fmt_num(v, 2), ha="center", fontsize=8)
    p = reports_dir / "strategy_sharpe_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 4. Risk vs return scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    vol_pct = metrics["annualized_volatility"] * 100
    ret_pct = metrics["annualized_return"] * 100
    for _, row in metrics.iterrows():
        name = row["strategy"]
        ax.scatter(
            row["annualized_volatility"] * 100,
            row["annualized_return"] * 100,
            s=120,
            color=palette.get(name, "#888888"),
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            _strategy_label(name),
            (row["annualized_volatility"] * 100, row["annualized_return"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    sharpe_raw = metrics["annualized_return"] / metrics["annualized_volatility"].replace(0, np.nan)
    best_idx = sharpe_raw.idxmax()
    best = metrics.loc[best_idx]
    bx = float(best["annualized_volatility"] * 100)
    by = float(best["annualized_return"] * 100)
    if np.isfinite(bx) and np.isfinite(by) and (bx != 0 or by != 0):
        x_end = max(float(vol_pct.max()) * 1.1, bx * 1.05, bx + 1.0)
        scale = x_end / bx if bx != 0 else 1.0
        ax.plot(
            [0, bx * scale],
            [0, by * scale],
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            zorder=2,
            label=f"Best return/risk ({_strategy_label(best['strategy'])})",
        )
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Risk vs Return", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    p = reports_dir / "strategy_risk_return.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 5. M2 classification summary
    if m2_metrics and "confusion_matrix" in m2_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        cm = np.array(m2_metrics["confusion_matrix"])
        im = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("M2 Confusion Matrix (Test)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=axes[0], fraction=0.046)

        cls_names = ["Precision", "Recall", "F1", "AUC"]
        cls_vals = [
            m2_metrics.get("precision", 0),
            m2_metrics.get("recall", 0),
            m2_metrics.get("f1", 0),
            m2_metrics.get("auc", 0),
        ]
        axes[1].barh(cls_names, cls_vals, color="#8172B3")
        axes[1].set_xlim(0, 1)
        axes[1].set_title("M2 Test Metrics")
        for i, v in enumerate(cls_vals):
            axes[1].text(v + 0.02, i, _fmt_num(v, 2), va="center", fontsize=9)
        p = reports_dir / "m2_classification_summary.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    return saved


def save_figures(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    m2_metrics: dict[str, Any],
    ic_series: pd.Series,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # 1. Cumulative returns
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        cum = (1 + res.returns.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Cumulative Returns")
    p = figures_dir / "cumulative_returns.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        eq = (1 + res.returns.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Drawdown")
    p = figures_dir / "drawdown.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 3. Rolling Sharpe
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        roll = res.returns.rolling(52).mean() / res.returns.rolling(52).std() * np.sqrt(WEEKS_PER_YEAR)
        ax.plot(roll.index, roll.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Rolling 52-week Sharpe")
    p = figures_dir / "rolling_sharpe.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 4. Rolling 12m max drawdown
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        rdd = rolling_max_drawdown(res.returns, 52)
        ax.plot(rdd.index, rdd.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Rolling 52-week Max Drawdown")
    p = figures_dir / "rolling_max_drawdown.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 5. Turnover
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        ax.plot(res.turnover.index, res.turnover.values, label=name, alpha=0.7)
    ax.legend(fontsize=8)
    ax.set_title("Turnover")
    p = figures_dir / "turnover.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 6. Asset weights (m1_m2_linear if present)
    key = "m1_m2_m3_linear" if "m1_m2_m3_linear" in results else (
        "m1_m2_linear" if "m1_m2_linear" in results else next(iter(results))
    )
    w = results[key].weights
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in w.columns:
        ax.plot(w.index, w[col], label=col, alpha=0.7)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Asset Weights ({key})")
    p = figures_dir / "asset_weights.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 7. M1 signal heatmap
    df = panel.reset_index()
    if "M1_signal" in df.columns:
        sig = df.pivot(index="date", columns="ticker", values="M1_signal")
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(sig.T, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_yticks(range(len(sig.columns)))
        ax.set_yticklabels(sig.columns)
        ax.set_title("M1 Signal Heatmap")
        fig.colorbar(im, ax=ax)
        p = figures_dir / "m1_signal_heatmap.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 8. M2 probability histogram
    if "p_success" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["p_success"].dropna(), bins=30, edgecolor="black")
        ax.set_title("M2 Probability Histogram")
        p = figures_dir / "m2_probability_histogram.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 9. M2 calibration / ROC (real curves when deep diagnostics ran; fallback title only)
    if m2_metrics and "auc" in m2_metrics:
        mask = panel.reset_index()
        if "meta_label" in mask.columns and "p_success" in mask.columns:
            y_t = mask["meta_label"]
            y_p = mask["p_success"]
            cal = m2_calibration_table(y_t, y_p)
            m = m2_classification_metrics(y_t, y_p)
            if len(y_t.dropna()) > 0 and y_t.dropna().astype(int).nunique() >= 2:
                save_m2_deep_charts(y_t, y_p, cal, pd.DataFrame(), pd.DataFrame(), m, figures_dir)
                saved.append(str(figures_dir / "m2_roc_calibration.png"))

    # 10. IC time series
    if not ic_series.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ic_series.index, ic_series.values)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("Information Coefficient Time Series")
        p = figures_dir / "ic_timeseries.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


def generate_final_report(
    metrics_table: pd.DataFrame,
    m2_metrics: dict[str, Any],
    report_path: Path,
    *,
    effective_start: str | None = None,
    effective_end: str | None = None,
    results: dict[str, BacktestResult] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = report_path.parent

    chart_files: list[str] = []
    if results is not None:
        chart_files = save_report_charts(results, m2_metrics, reports_dir)

    display_table = format_metrics_table_for_report(metrics_table)
    summary_lines = _build_executive_summary(metrics_table, m2_metrics)

    m2_display = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": _fmt_num(m2_metrics.get("accuracy", float("nan"))), "Meaning": "Share of correct meta-label predictions"},
            {"Metric": "Precision", "Value": _fmt_num(m2_metrics.get("precision", float("nan"))), "Meaning": "Approved trades that were actually profitable"},
            {"Metric": "Recall", "Value": _fmt_num(m2_metrics.get("recall", float("nan"))), "Meaning": "Profitable trades that M2 approved"},
            {"Metric": "F1 Score", "Value": _fmt_num(m2_metrics.get("f1", float("nan"))), "Meaning": "Balance of precision and recall"},
            {"Metric": "AUC", "Value": _fmt_num(m2_metrics.get("auc", float("nan"))), "Meaning": "Ranking quality of M2 probabilities"},
            {"Metric": "Brier Score", "Value": _fmt_num(m2_metrics.get("brier_score", float("nan"))), "Meaning": "Probability calibration error (lower is better)"},
            {
                "Metric": "Mean IC",
                "Value": _fmt_num(m2_metrics.get("information_coefficient_mean", float("nan"))),
                "Meaning": "Spearman rank correlation of M1 scores vs forward returns",
            },
        ]
    )

    lines = [
        "# Final Report: AI-Augmented Multi-Asset Meta-Labeling Pipeline",
        "",
        "## Executive Summary",
        "",
        *summary_lines,
        "",
        "## Sample Period",
        "",
        f"| Item | Value |",
        f"| --- | --- |",
        f"| Effective start | {effective_start or 'N/A'} |",
        f"| Effective end | {effective_end or 'N/A'} |",
        f"| Train period | 2006-01-01 to 2020-12-31 |",
        f"| Test period (M2 evaluation) | 2021-01-01 onward |",
        f"| Assets | SP500, MSCI_EAFE, MSCI_EM, UST_7_10, US_HIGH_YIELD, GOLD_SPOT, US_REIT |",
        "",
        "## Strategy Comparison",
        "",
        "All return and risk figures are **annualized** from weekly data. Benchmark for excess return and information ratio is **equal-weight 1/7**.",
        "",
        _markdown_table(display_table),
        "",
        "### How to read the metrics",
        "",
        "| Metric | Interpretation |",
        "| --- | --- |",
        "| **Ann. Return** | Geometric average yearly portfolio return after transaction costs |",
        "| **Ann. Volatility** | Standard deviation of weekly returns, scaled to a year |",
        "| **Sharpe** | Return per unit of risk (higher is better; assumes 0% risk-free rate) |",
        "| **Max Drawdown** | Largest peak-to-trough loss over the displayed period |",
        "| **Excess vs EW** | Strategy return minus equal-weight benchmark return |",
        "| **Info Ratio** | Consistency of outperformance vs equal-weight (mean active return / tracking error) |",
        "| **Weekly Hit Rate** | Fraction of weeks with positive net strategy return |",
        "",
        "## Visual Summary",
        "",
    ]

    chart_descriptions = {
        "strategy_cumulative_returns.png": "Growth of $1 invested — compares benchmarks, M1-only, and meta-labeled strategies.",
        "strategy_drawdown.png": "Peak-to-trough losses over time — shows how deeply each strategy declined.",
        "strategy_sharpe_comparison.png": "Sharpe ratio bar chart — risk-adjusted performance at a glance.",
        "strategy_risk_return.png": "Return vs volatility scatter; dashed line from origin through the best return/risk (Sharpe) strategy.",
        "m2_classification_summary.png": "M2 confusion matrix and precision/recall on the out-of-sample test set.",
    }
    for chart in chart_files:
        desc = chart_descriptions.get(chart, "")
        lines.append(f"### {chart.replace('_', ' ').replace('.png', '').title()}")
        lines.append("")
        if desc:
            lines.append(desc)
            lines.append("")
        lines.append(f"![{chart}]({chart})")
        lines.append("")

    lines.extend(
        [
            "## M2 Meta-Labeling Quality (Test Set)",
            "",
            "M2 predicts whether each non-zero M1 signal will be profitable after a 4-week horizon. "
            "Metrics below are computed on **out-of-sample** dates from 2021 onward.",
            "",
            _markdown_table(m2_display),
            "",
        ]
    )

    if "confusion_matrix" in m2_metrics:
        cm = m2_metrics["confusion_matrix"]
        lines.extend(
            [
                "**Confusion matrix** (rows = actual, columns = predicted):",
                "",
                f"| | Predicted 0 | Predicted 1 |",
                f"| --- | ---: | ---: |",
                f"| Actual 0 | {cm[0][0]} | {cm[0][1]} |",
                f"| Actual 1 | {cm[1][0]} | {cm[1][1]} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Key Takeaways",
            "",
            "1. **Benchmarks** (equal-weight, 60/40) delivered the highest raw returns but with larger drawdowns.",
            "2. **M1-only** generated more active exposure with mixed results versus passive benchmarks.",
            "3. **M2 meta-labeling** improved Sharpe ratios mainly by **reducing position size** on low-confidence signals.",
            "4. M2 has **low recall** — it filters aggressively, trading less often but with better risk control.",
            "5. Results are **historical** and sensitive to data source quality (yfinance/FRED).",
            "",
            "## Pipeline Architecture",
            "",
            "```",
            "Raw Data → Ingest → Validation → Features → M1 → M2 → Sizing → Backtest → Diagnostics",
            "```",
            "",
            "## Look-Ahead Controls",
            "",
            "- Features use only data available at signal time (`shift(1)` on rolling windows)",
            "- Macro series lagged 4 weeks to approximate release delay",
            "- Strict chronological train/test split (train ≤ 2020, test ≥ 2021)",
            "- Label columns excluded from model feature matrices",
            "",
            "## AI / LLM Usage",
            "",
            "LLM-derived features are **disabled** in this run. See `runs/*/research_log.jsonl` for design decisions.",
            "",
            "## Limitations",
            "",
            "- yfinance and FRED are research-grade fallbacks, not institutional data",
            "- Partial FRED refreshes preserve cached series or use clearly logged proxy macro fallbacks",
            "- Data provenance, ETL, validation, cache behavior, and fallback caveats are documented in `../DATA_SOURCES_AND_ETL.md`",
            "- Index/proxy backtests may include survivorship and data-vendor effects",
            "- Past performance does not predict future results",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


MODE_LABELS = {
    "long_only": "Long Only (no shorts)",
    "long_short": "Long / Short",
}


def _m2_metrics_table(m2_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"Metric": "Accuracy", "Value": _fmt_num(m2_metrics.get("accuracy", float("nan"))), "Meaning": "Share of correct meta-label predictions"},
        {"Metric": "Precision", "Value": _fmt_num(m2_metrics.get("precision", float("nan"))), "Meaning": "Approved trades that were actually profitable"},
        {"Metric": "Recall", "Value": _fmt_num(m2_metrics.get("recall", float("nan"))), "Meaning": "Profitable trades that M2 approved"},
        {"Metric": "F1 Score", "Value": _fmt_num(m2_metrics.get("f1", float("nan"))), "Meaning": "Balance of precision and recall"},
        {"Metric": "AUC-ROC", "Value": _fmt_num(m2_metrics.get("auc", float("nan"))), "Meaning": "Ranking quality: P(random winner scored higher than random loser)"},
        {"Metric": "AUC-PR", "Value": _fmt_num(m2_metrics.get("auc_pr", float("nan"))), "Meaning": "Precision-recall AUC; more informative when base rate ≠ 50%"},
        {"Metric": "Base Rate", "Value": _fmt_pct(m2_metrics.get("base_rate", float("nan"))), "Meaning": "Fraction of M1 trades that beat the cost hurdle"},
        {"Metric": "Brier Score", "Value": _fmt_num(m2_metrics.get("brier_score", float("nan"))), "Meaning": "Probability calibration error (lower is better)"},
        {"Metric": "Mean P (winners)", "Value": _fmt_num(m2_metrics.get("mean_p_winners", float("nan"))), "Meaning": "Average M2 probability on profitable trades"},
        {"Metric": "Mean P (losers)", "Value": _fmt_num(m2_metrics.get("mean_p_losers", float("nan"))), "Meaning": "Average M2 probability on unprofitable trades"},
        {
            "Metric": "Mean IC",
            "Value": _fmt_num(m2_metrics.get("information_coefficient_mean", float("nan"))),
            "Meaning": "Spearman rank correlation of M1 scores vs forward returns",
        },
    ]
    if m2_metrics.get("degeneracy_note"):
        rows.append({"Metric": "Note", "Value": "—", "Meaning": m2_metrics["degeneracy_note"]})
    return pd.DataFrame(rows)


M1_SIGNAL_LABELS: dict[int, str] = {
    -1: "Short (−1)",
    0: "Flat (0)",
    1: "Long (+1)",
}


def analyze_m1_signal_m2_performance(
    panel: pd.DataFrame,
    threshold: float,
    *,
    period_label: str = "test",
) -> dict[str, Any]:
    """Summarize trade outcomes and M2 quality grouped by M1 signal (−1, 0, +1)."""
    df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    if df.empty:
        return {"period": period_label, "threshold": threshold, "by_signal": pd.DataFrame()}

    rows: list[dict[str, Any]] = []
    for sig in sorted(df["M1_signal"].dropna().unique()):
        sig_int = int(sig)
        sub = df[df["M1_signal"] == sig_int]
        row: dict[str, Any] = {
            "m1_signal": sig_int,
            "signal_label": M1_SIGNAL_LABELS.get(sig_int, str(sig_int)),
            "observations": len(sub),
            "share_of_panel": len(sub) / len(df),
        }
        if sig_int == 0:
            rows.append(row)
            continue

        trades = sub[sub["meta_label"].notna() & sub["p_success"].notna()]
        row["labeled_trades"] = len(trades)
        if trades.empty:
            rows.append(row)
            continue

        approved = trades[trades["p_success"] >= threshold]
        rejected = trades[trades["p_success"] < threshold]

        row["m1_hit_rate"] = float(trades["meta_label"].mean())
        row["mean_trade_return"] = float(trades["trade_return"].mean())
        row["median_trade_return"] = float(trades["trade_return"].median())
        row["m2_approval_rate"] = float(len(approved) / len(trades))
        row["hit_rate_m2_approved"] = float(approved["meta_label"].mean()) if len(approved) else float("nan")
        row["hit_rate_m2_rejected"] = float(rejected["meta_label"].mean()) if len(rejected) else float("nan")
        row["mean_return_m2_approved"] = float(approved["trade_return"].mean()) if len(approved) else float("nan")
        row["mean_return_m2_rejected"] = float(rejected["trade_return"].mean()) if len(rejected) else float("nan")

        m2_group = m2_classification_metrics(trades["meta_label"], trades["p_success"], threshold)
        row["m2_accuracy"] = m2_group.get("accuracy", float("nan"))
        row["m2_precision"] = m2_group.get("precision", float("nan"))
        row["m2_recall"] = m2_group.get("recall", float("nan"))
        row["m2_f1"] = m2_group.get("f1", float("nan"))
        row["m2_auc"] = m2_group.get("auc", float("nan"))
        rows.append(row)

    by_signal = pd.DataFrame(rows)
    return {"period": period_label, "threshold": threshold, "by_signal": by_signal}


def format_m1_signal_analysis_table(analysis: dict[str, Any]) -> pd.DataFrame:
    """Display table for M1-signal-grouped M2 analysis."""
    df = analysis.get("by_signal", pd.DataFrame())
    if df.empty:
        return pd.DataFrame()

    display = pd.DataFrame(
        {
            "M1 Signal": df["signal_label"],
            "Observations": df["observations"],
            "Share": df["share_of_panel"].map(lambda x: _fmt_pct(x)),
            "Labeled Trades": df.get("labeled_trades", pd.Series(dtype=float)).fillna(0).astype(int),
            "M1 Hit Rate": df.get("m1_hit_rate", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "Mean Trade Return": df.get("mean_trade_return", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "M2 Approval Rate": df.get("m2_approval_rate", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "Hit Rate (M2 Approved)": df.get("hit_rate_m2_approved", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "M2 Precision": df.get("m2_precision", pd.Series(dtype=float)).map(lambda x: _fmt_num(x)),
            "M2 Recall": df.get("m2_recall", pd.Series(dtype=float)).map(lambda x: _fmt_num(x)),
            "M2 F1": df.get("m2_f1", pd.Series(dtype=float)).map(lambda x: _fmt_num(x)),
        }
    )
    return display


def save_m1_signal_m2_chart(analysis: dict[str, Any], output_path: Path) -> str:
    """Visualize M1 signal grouping and M2 filtering on the test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = analysis.get("by_signal", pd.DataFrame())
    if df.empty:
        return ""

    active = df[df["m1_signal"] != 0].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: observation counts by M1 signal
    ax = axes[0, 0]
    colors_count = {1: "#55A868", 0: "#CCCCCC", -1: "#C44E52"}
    bar_colors = [colors_count.get(int(s), "#888888") for s in df["m1_signal"]]
    ax.bar(df["signal_label"], df["observations"], color=bar_colors)
    ax.set_title("Observations by M1 Signal", fontweight="bold")
    ax.set_ylabel("Count (asset-weeks)")
    ax.tick_params(axis="x", rotation=15)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["observations"], f"{int(row['observations'])}", ha="center", va="bottom", fontsize=8)

    # Panel B: mean trade return for active signals
    ax = axes[0, 1]
    if not active.empty and "mean_trade_return" in active.columns:
        ret_pct = active["mean_trade_return"].fillna(0) * 100
        bar_colors_a = [colors_count.get(int(s), "#888888") for s in active["m1_signal"]]
        bars = ax.bar(active["signal_label"], ret_pct, color=bar_colors_a)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title("Mean Forward Trade Return by M1 Signal", fontweight="bold")
        ax.set_ylabel("Mean return (%)")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, ret_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )

    # Panel C: M1 hit rate vs M2-approved hit rate
    ax = axes[1, 0]
    if not active.empty:
        x = np.arange(len(active))
        width = 0.35
        m1_hr = active["m1_hit_rate"].fillna(0) * 100
        m2_hr = active["hit_rate_m2_approved"].fillna(0) * 100
        ax.bar(x - width / 2, m1_hr, width, label="M1 (all trades)", color="#4C72B0")
        ax.bar(x + width / 2, m2_hr, width, label="M2 approved only", color="#8172B3")
        ax.set_xticks(x)
        ax.set_xticklabels(active["signal_label"], rotation=15, ha="right")
        ax.set_ylabel("Hit rate (%)")
        ax.set_title("Profitability: M1 vs M2-Filtered", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(100, float(m1_hr.max()) * 1.15, float(m2_hr.max()) * 1.15))

    # Panel D: M2 classification metrics by active signal group
    ax = axes[1, 1]
    if not active.empty:
        metric_names = ["m2_precision", "m2_recall", "m2_f1", "m2_auc"]
        metric_labels = ["Precision", "Recall", "F1", "AUC"]
        x = np.arange(len(active))
        n_metrics = len(metric_names)
        width = 0.8 / n_metrics
        palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
        for j, (col, label) in enumerate(zip(metric_names, metric_labels)):
            if col not in active.columns:
                continue
            offset = (j - (n_metrics - 1) / 2) * width
            vals = active[col].fillna(0)
            ax.bar(x + offset, vals, width, label=label, color=palette[j % len(palette)])
        ax.set_xticks(x)
        ax.set_xticklabels(active["signal_label"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("M2 Classifier Quality by M1 Signal", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)

    threshold = analysis.get("threshold", 0.5)
    fig.suptitle(
        f"M2 Performance by M1 Signal Group (test set, M2 threshold={threshold})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    chart_name = output_path.name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_name


def save_m1_signal_m2_mode_comparison_chart(
    mode_analyses: list[tuple[str, dict[str, Any]]],
    output_path: Path,
) -> str:
    """Compare long-only vs long/short M1 signal outcomes and M2 filtering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mode_colors = {"long_only": "#4C72B0", "long_short": "#C44E52"}

    # Left: mean trade return by signal and mode
    ax = axes[0]
    plot_rows = []
    for mode_name, analysis in mode_analyses:
        df = analysis.get("by_signal", pd.DataFrame())
        active = df[df["m1_signal"] != 0]
        for _, row in active.iterrows():
            plot_rows.append(
                {
                    "mode": MODE_LABELS.get(mode_name, mode_name),
                    "mode_key": mode_name,
                    "signal": row["signal_label"],
                    "mean_return_pct": float(row.get("mean_trade_return", 0) or 0) * 100,
                }
            )
    if plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        signals = plot_df["signal"].unique()
        modes = [MODE_LABELS.get(m, m) for m, _ in mode_analyses]
        x = np.arange(len(signals))
        width = 0.8 / max(len(modes), 1)
        for i, (mode_name, _) in enumerate(mode_analyses):
            mode_label = MODE_LABELS.get(mode_name, mode_name)
            subset = plot_df[plot_df["mode"] == mode_label]
            vals = [subset.loc[subset["signal"] == s, "mean_return_pct"].iloc[0] if s in subset["signal"].values else 0 for s in signals]
            offset = (i - (len(modes) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=mode_label, color=mode_colors.get(mode_name, "#888888"))
        ax.set_xticks(x)
        ax.set_xticklabels(signals, rotation=15, ha="right")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel("Mean trade return (%)")
        ax.set_title("M1 Trade Return by Signal & Mode", fontweight="bold")
        ax.legend(fontsize=8)

    # Right: M1 vs M2-approved hit rate for Long (+1) across modes
    ax = axes[1]
    hit_rows = []
    for mode_name, analysis in mode_analyses:
        df = analysis.get("by_signal", pd.DataFrame())
        for sig_val, label in [(1, "Long (+1)"), (-1, "Short (−1)")]:
            row = df[df["m1_signal"] == sig_val]
            if row.empty or pd.isna(row.iloc[0].get("m1_hit_rate", np.nan)):
                continue
            hit_rows.append(
                {
                    "group": f"{label}\n({MODE_LABELS.get(mode_name, mode_name)})",
                    "mode_key": mode_name,
                    "m1_hit": float(row.iloc[0]["m1_hit_rate"]) * 100,
                    "m2_hit": float(row.iloc[0].get("hit_rate_m2_approved", 0) or 0) * 100,
                }
            )
    if hit_rows:
        hit_df = pd.DataFrame(hit_rows)
        x = np.arange(len(hit_df))
        width = 0.35
        ax.bar(x - width / 2, hit_df["m1_hit"], width, label="M1 all trades", color="#4C72B0")
        ax.bar(x + width / 2, hit_df["m2_hit"], width, label="M2 approved", color="#8172B3")
        ax.set_xticks(x)
        ax.set_xticklabels(hit_df["group"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Hit rate (%)")
        ax.set_title("Hit Rate: M1 vs M2-Filtered by Mode", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)

    fig.suptitle("M2 Exploration: M1 Signal Groups (Long-Only vs Long/Short)", fontweight="bold", y=1.02)
    fig.tight_layout()
    chart_name = output_path.name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_name


def build_m1_signal_m2_report_section(
    mode_results: list[Any],
    *,
    comparison_chart: str | None = None,
) -> list[str]:
    """Markdown section for M1-signal-grouped M2 analysis."""
    lines = [
        "## M2 Performance by M1 Signal",
        "",
        "M1 outputs three signal types per asset-week: **short (−1)**, **flat (0)**, or **long (+1)**. "
        "M2 only trains and predicts on non-zero signals. Below we break out **test-set** trade outcomes "
        "and classifier quality within each M1 group.",
        "",
        "- **M1 hit rate**: share of trades with positive forward return (after cost hurdle)",
        "- **M2 approval rate**: share of trades where `p_success` ≥ threshold",
        "- **Hit rate (M2 approved)**: profitability among trades M2 kept",
        "",
    ]
    if comparison_chart:
        lines.extend(
            [
                "### Long-Only vs Long/Short Comparison",
                "",
                f"![M2 by M1 signal comparison]({comparison_chart})",
                "",
                "*Left: mean forward trade return by M1 signal. Right: M1 vs M2-filtered hit rates "
                "(long-only has no short bucket).*",
                "",
            ]
        )

    for mode in mode_results:
        analysis = getattr(mode, "m1_signal_analysis", None)
        if not analysis:
            continue
        label = MODE_LABELS.get(mode.mode_name, mode.mode_name)
        chart_name = getattr(mode, "m1_signal_chart", None)
        lines.extend(
            [
                f"### {label}",
                "",
                f"`allow_short={mode.allow_short}` — M2 threshold = {analysis.get('threshold', 'N/A')}",
                "",
            ]
        )
        table = format_m1_signal_analysis_table(analysis)
        if not table.empty:
            lines.append(_markdown_table(table))
            lines.append("")
        chart_rel = getattr(mode, "m1_signal_chart_rel", None)
        if chart_rel:
            lines.append(f"![M2 by M1 signal — {label}]({chart_rel})")
        elif chart_name:
            lines.append(f"![M2 by M1 signal — {label}]({chart_name})")
            lines.append("")
        if mode.mode_name == "long_only":
            lines.extend(
                [
                    "*Long-only mode: M1 never emits −1; shorts are disabled at the signal layer.*",
                    "",
                ]
            )

    return lines


def save_m1_mode_comparison_chart(mode_results: list[Any], reports_dir: Path) -> str:
    """Chart comparing M1-only performance across long-only vs long-short modes."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    colors = {"long_only": "#4C72B0", "long_short": "#C44E52"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode in mode_results:
        name = mode.mode_name
        res = mode.results["m1_only"]
        cum = (1 + res.returns.fillna(0)).cumprod()
        axes[0].plot(
            cum.index,
            cum.values,
            label=MODE_LABELS.get(name, name),
            color=colors.get(name, "#888888"),
            linewidth=2,
        )
    axes[0].set_title("M1 Only — Cumulative Growth of $1", fontweight="bold")
    axes[0].set_ylabel("Portfolio value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rows = []
    for mode in mode_results:
        m1_row = mode.metrics_table.set_index("strategy").loc["m1_only"]
        rows.append(
            {
                "Mode": MODE_LABELS.get(mode.mode_name, mode.mode_name),
                "Ann. Return (%)": m1_row["annualized_return"] * 100,
                "Sharpe": m1_row["sharpe"],
                "Max DD (%)": m1_row["max_drawdown"] * 100,
            }
        )
    compare = pd.DataFrame(rows)
    x = np.arange(len(compare))
    width = 0.25
    axes[1].bar(x - width, compare["Ann. Return (%)"], width, label="Ann. Return %", color="#55A868")
    axes[1].bar(x, compare["Sharpe"] * 10, width, label="Sharpe (×10)", color="#8172B3")
    axes[1].bar(x + width, compare["Max DD (%)"].abs(), width, label="|Max DD| %", color="#CCB974")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(compare["Mode"], rotation=15, ha="right")
    axes[1].set_title("M1 Only — Key Metrics by Mode", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    chart_name = "m1_mode_comparison.png"
    fig.tight_layout()
    fig.savefig(reports_dir / chart_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_name


def build_performance_parameters_section(cfg: PipelineConfig) -> list[str]:
    """Document tunable parameters that influence strategy performance."""
    m1 = cfg.m1
    w = m1.weights
    test_end_disp = cfg.split.test_end or "latest (open-ended)"
    return [
        "## Configuration Parameters Affecting Performance",
        "",
        "The pipeline reads defaults from `config/config.yaml`. **Split dates** can also be set "
        "at runtime without editing the file (see CLI below). Other parameters require config edits.",
        "",
        "### Train / Test Split",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `split.data_start` | {cfg.data_start_resolved()} | Earliest downloaded price date (can precede train for feature warmup) |",
        f"| `split.train_start` | {cfg.split.train_start} | Intended train window start (clipped to effective panel start) |",
        f"| `split.train_end` | {cfg.split.train_end} | Last in-sample date; **primary knob for tuning in-sample fit** |",
        f"| `split.test_start` | {cfg.split.test_start} | Out-of-sample evaluation begins here (M2 metrics, IC, and test-period strategy tables) |",
        f"| `split.test_end` | {test_end_disp} | Optional cap on the evaluation window |",
        f"| `split.require_full_universe` | {cfg.split.require_full_universe} | If true, only weeks with all 7 sleeves (~2011+); if false, partial groups allowed |",
        "",
        "**Can train_start be before 2006?** Yes in config/CLI, but with `require_full_universe: true` "
        "(default) the **effective** sample starts when all seven index/proxy sleeves have sufficient public data coverage "
        "both exist. Dates before that are dropped. Set `require_full_universe: false` or `--partial-universe` "
        "to train on subsets when earlier public proxy coverage is incomplete.",
        "",
        "**CLI overrides** (ISO dates, applied after loading config):",
        "",
        "```bash",
        "# Shorter/longer train, earlier/later test — compare Sharpe in reports/final_report.md",
        "python -m src.run_pipeline --train-end 2018-12-31 --test-start 2019-01-01",
        "python -m src.run_pipeline --train-end 2015-12-31 --test-start 2016-01-01",
        "python -m src.run_pipeline --train-start 2008-01-01 --train-end 2012-12-31 --test-start 2013-01-01",
        "",
        "# Earlier history: partial universe before all seven index/proxy sleeves have sufficient public data coverageed",
        "python -m src.run_pipeline --data-start 2004-01-01 --train-start 2005-01-01 --train-end 2006-12-31 "
        "--test-start 2007-01-01 --partial-universe --refresh-data",
        "```",
        "",
        "Shorter train windows reduce overfitting risk but give fewer M2 labels; varying `train_end` is the "
        "fastest way to test whether performance is stable across in-sample cutoffs.",
        "",
        "### M1 Rule-Based Side Model",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `models.m1.weights` | momentum={w['momentum']}, trend={w['trend']}, macro={w['macro']}, risk={w['risk_penalty']} | Relative importance of factor families in the composite score |",
        f"| `models.m1.optimize_thresholds` | {m1.optimize_thresholds} | When true, long/short cutoffs are tuned on the train set only |",
        f"| `models.m1.long_quantile` / `short_quantile` | {m1.long_quantile} / {m1.short_quantile} | Starting quantiles for threshold search (higher long quantile → fewer longs) |",
        f"| `models.m1.allow_short` | {m1.allow_short} | Default shorting flag; pipeline always runs both long-only and long/short modes |",
        f"| `models.m1.asset_class_tilts` | {m1.asset_class_tilts} | Macro tilts by asset class (equity, bonds, credit, gold, REIT) |",
        f"| `models.m1.allocation_mode` | {m1.allocation_mode} | `threshold` (absolute cutoffs) or `top_k` (weekly cross-sectional rank) |",
        f"| `models.m1.top_k` | {m1.top_k} | Number of names to long each week when `allocation_mode=top_k` |",
        f"| `models.m1.conviction_sizing` | {m1.conviction_sizing} | Scale weights by normalized M1 score before M2 sizing |",
        f"| `models.m1.tune_objective` | {m1.tune_objective} | `trade` or `portfolio` Sharpe for threshold tuning (threshold mode only) |",
        "",
        "### M2 Meta-Labeling",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `models.m2.threshold` | {cfg.m2.threshold} | Minimum P(success) to take full size; higher → fewer trades, often lower turnover |",
        f"| `models.m2.calibrate` | {cfg.m2.calibrate} | Probability calibration on train data; improves threshold interpretability |",
        f"| `models.m2.type` | {cfg.m2.type} | Classifier used for meta-labels |",
        "",
        "### Labels (M1 targets & M2 supervision)",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `labels.horizon_weeks` | {cfg.labels.horizon_weeks} | Forward return horizon for profitability labels |",
        f"| `labels.positive_threshold` | {cfg.labels.positive_threshold} | Minimum forward return to label a long as successful |",
        f"| `labels.negative_threshold` | {cfg.labels.negative_threshold} | Forward return threshold for short success |",
        f"| `labels.transaction_cost_threshold` | {cfg.labels.transaction_cost_threshold} | Cost hurdle embedded in label construction |",
        "",
        "### Portfolio & Costs",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `portfolio.transaction_cost_bps` | {cfg.portfolio.transaction_cost_bps} | Round-trip cost per unit turnover; higher values drag net returns |",
        f"| `portfolio.max_gross_exposure` | {cfg.portfolio.max_gross_exposure} | Cap on sum of absolute weights |",
        f"| `portfolio.max_abs_asset_weight` | {cfg.portfolio.max_abs_asset_weight} | Per-asset weight ceiling |",
        f"| `portfolio.sizing_mode` | {cfg.portfolio.sizing_mode} | Default M3 bet-sizing rule (binary / linear / ecdf) |",
        f"| `models.m3.mode` | {cfg.m3.mode} | M3 sizing rule applied to M2 probabilities (Joubert bet-sizing layer) |",
        f"| `models.m3.threshold` | {cfg.m3.threshold or cfg.m2.threshold} | M3 binary threshold T (all-or-nothing sizing only) |",
        f"| `portfolio.vol_target_ann` | {cfg.portfolio.vol_target_ann} | Annualized vol target for gross scaling (null disables) |",
        f"| `portfolio.vol_target_lookback_weeks` | {cfg.portfolio.vol_target_lookback_weeks} | Trailing window for realized vol estimate |",
        "",
        "### Features",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `features.momentum_windows` | {cfg.features.momentum_windows} | Lookback weeks for momentum factors |",
        f"| `features.macro_lag_weeks` | {cfg.features.macro_lag_weeks} | Release lag applied to macro series (reduces look-ahead) |",
        f"| `features.winsorize_pct` | {cfg.features.winsorize_pct} | Train-set winsorization of extreme feature values |",
        "",
    ]


def _df_to_markdown_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_No data._"
    return _markdown_table(df)


def build_deep_diagnostics_summary_section(mode_results: list[Any]) -> list[str]:
    """Executive summary bullets for final_report deep diagnostics section."""
    long_mode = next((m for m in mode_results if m.mode_name == "long_only"), mode_results[0] if mode_results else None)
    lines = [
        "## Deep Diagnostics",
        "",
        "Branch update (vs `main`): [Executive summary](../BRANCH_UPDATE_REPORT.md) · "
        "[Technical report](branch_update_vitaly_week5.md)",
        "",
        "**Terminology:** [TERMINOLOGY.md](../TERMINOLOGY.md) — plain-language glossary for finance and ML terms used in these reports.",
        "",
        "Companion reports provide factor-level, M2 input, regime, M3 allocation, and AUC-ROC detail:",
        "",
        "- [M1 Factor Analysis](m1_factor_analysis.md) — per-factor IC, correlation/covariance, sleeve backtests",
        "- [M2 Diagnostics](m2_diagnostics.md) — calibration, decile returns, feature importance, AUC-ROC guide",
        "- [M2 Feature Research](m2_feature_research.md) — M1 factor + external factor enrichment sweep",
        "- [Market & Regime Analysis](market_regime_analysis.md) — regime timeline, transitions, conditioned performance",
        "- [M3 Allocation Analysis](m3_allocation_analysis.md) — M1 vs M3=0 vs M3>0 states and sizing rules",
        "- [M3 Threshold Analysis](m3_threshold_analysis.md) — binary/linear threshold sweep with rejection vs Sharpe trade-off",
        "- [IR Attribution Analysis](ir_attribution_analysis.md) — why Info Ratio falls vs equal-weight when M2/M3 added",
        "- [IR Improvement Research](ir_improvement_research.md) — intervention sweep and adoption verdict",
        "- [Extended Evaluation](evaluation_analysis.md) — walk-forward folds and transaction-cost sensitivity",
        "- [Walk-Forward Analysis](walk_forward_analysis.md) — ECDF edge stability across OOS windows",
        "",
    ]
    if long_mode is None:
        return lines

    fs = getattr(long_mode, "factor_summary", None) or {}
    rs = getattr(long_mode, "regime_summary", None) or {}
    m2d = getattr(long_mode, "m2_deep_summary", None) or {}
    factor_ic = fs.get("factor_ic", pd.DataFrame())
    if not factor_ic.empty:
        test_ic = factor_ic[(factor_ic["period"] == "test") & (factor_ic["factor"] != "M1_score")]
        if not test_ic.empty:
            best = test_ic.loc[test_ic["ic_mean"].idxmax()]
            lines.append(
                f"- **M1 factors (test):** strongest IC is `{best['factor']}` "
                f"(mean IC {_fmt_num(best['ic_mean'])})."
            )
    m2m = m2d.get("m2_metrics") or getattr(long_mode, "m2_metrics", {})
    if m2m:
        auc = m2m.get("auc", float("nan"))
        lines.append(
            f"- **M2 AUC-ROC (test, long-only):** {_fmt_num(auc)} — weak ranking quality; "
            "value is mainly in M3 ECDF sizing, not M3 binary threshold at 0.55."
        )
    perf = rs.get("performance_by_regime", pd.DataFrame())
    if not perf.empty:
        lines.append("- **Regime:** strategy Sharpe varies by `risk_off` / curve / inflation flags — see regime report.")
    m3d = getattr(long_mode, "m3_summary", None) or {}
    evald = getattr(long_mode, "eval_summary", None) or {}
    wf_stab = evald.get("walk_forward_stability") or {}
    if evald:
        wf = evald.get("walk_forward", pd.DataFrame())
        if not wf.empty and "ecdf_sharpe_edge_vs_m1" in wf.columns and not wf_stab:
            mean_edge = wf["ecdf_sharpe_edge_vs_m1"].mean()
            lines.append(
                f"- **Walk-forward (long-only):** mean ECDF Sharpe edge vs M1-only is {_fmt_num(mean_edge)} "
                f"across {len(wf)} fold(s)."
            )
    if wf_stab:
        lines.append(
            f"- **Walk-forward ECDF edge:** mean {_fmt_num(wf_stab.get('mean_ecdf_edge', float('nan')))} "
            f"({wf_stab.get('positive_edge_folds', 0)}/{wf_stab.get('n_folds', 0)} folds positive) — "
            f"see [Walk-Forward Analysis](walk_forward_analysis.md)."
        )
    if evald.get("ecdf_edge_persists_at_25bps"):
        lines.append(
            "- **Transaction costs:** ECDF Sharpe edge vs M1-only remains positive at 25 bps on the production test window."
        )
    if m3d.get("allocation_summary") is not None and not m3d["allocation_summary"].empty:
        test_alloc = m3d["allocation_summary"]
        if "period" in test_alloc.columns:
            test_alloc = test_alloc[test_alloc["period"] == "test"]
        active = test_alloc[test_alloc["allocation_state"] == "m3_active"]
        if not active.empty:
            lines.append(
                f"- **M3 allocation (test):** {_fmt_pct(active.iloc[0]['share'])} of asset-weeks are "
                "M1 candidates with M3_size > 0 (active bets before portfolio constraints)."
            )
    lines.append(
        "- **M1/M2/M3 stack:** M2 outputs P(success); M3 converts it to bet fraction; "
        "M3=0 with M1≠0 means a candidate was rejected by the sizing rule, not absent from M1."
    )
    lines.append("")
    return lines


def generate_m3_allocation_report(
    m3_summary: dict[str, Any],
    report_path: Path,
    *,
    mode_name: str = "long_only",
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_prefix = f"../data/backtests/{mode_name}/figures"
    allocation = m3_summary.get("allocation_summary", pd.DataFrame())
    rejection = m3_summary.get("rejection_analysis", pd.DataFrame())
    mode_cmp = m3_summary.get("mode_comparison", pd.DataFrame())

    lines = [
        "# M3 Bet-Sizing & Allocation Analysis",
        "",
        "**Research use only — not investment advice.**",
        "",
        "## M1 / M2 / M3 roles (Joubert framework)",
        "",
        "| Layer | Output | Question answered |",
        "| --- | --- | --- |",
        "| **M1** | `M1_signal` ∈ {-1, 0, 1} | Which side? (buy candidate or not) |",
        "| **M2** | `p_success` ∈ [0, 1] | How likely is the M1 trade profitable? |",
        "| **M3** | `M3_size` ∈ [0, 1] | How much capital to bet? (before portfolio caps) |",
        "",
        "M3 is a **deterministic sizing rule**, not a classifier. Binary thresholding at T=0.55 is an "
        "all-or-nothing M3 rule, not a separate M2 model.",
        "",
        "## Allocation states (long-only interpretation)",
        "",
        "| State | Condition | Meaning |",
        "| --- | --- | --- |",
        "| `no_signal` | M1 = 0 | No buy candidate from M1 (not selected in top-K) |",
        "| `m3_zero` | M1 ≠ 0 and M3_size = 0 | Buy candidate existed; M3 allocated zero capital |",
        "| `m3_active` | M1 ≠ 0 and M3_size > 0 | Buy candidate received positive bet fraction |",
        "",
        "## Allocation summary by period",
        "",
    ]
    if not allocation.empty:
        disp = allocation.copy()
        disp["share"] = disp["share"].map(lambda x: _fmt_pct(x))
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![M3 allocation states]({fig_prefix}/m3_allocation_states.png)")
        lines.append("")

    lines.extend(["## M3 rejection analysis (test, M1 candidates only)", ""])
    if not rejection.empty:
        disp = rejection.copy()
        for col in ("mean_p_success", "median_p_success", "mean_trade_return", "hit_rate"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_pct(x) if "return" in col or col == "hit_rate" else _fmt_num(x))
        lines.append(_markdown_table(disp))
    lines.extend(["", "## M3 rule comparison (binary vs linear vs ECDF)", ""])
    if not mode_cmp.empty:
        disp = mode_cmp.copy()
        if "m3_zero_share" in disp.columns:
            disp["m3_zero_share"] = disp["m3_zero_share"].map(lambda x: _fmt_pct(x))
        if "mean_m3_size_on_candidates" in disp.columns:
            disp["mean_m3_size_on_candidates"] = disp["mean_m3_size_on_candidates"].map(lambda x: _fmt_num(x))
        lines.append(_markdown_table(disp))
    lines.append("")
    report_path.write_text("\n".join(lines))


def generate_m1_factor_analysis_report(
    factor_summary: dict[str, Any],
    report_path: Path,
    *,
    cfg: PipelineConfig | None = None,
    mode_name: str = "long_only",
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    w = cfg.m1.weights if cfg else {}
    factor_ic = factor_summary.get("factor_ic", pd.DataFrame())
    corr = factor_summary.get("factor_correlation", pd.DataFrame())
    cov = factor_summary.get("factor_covariance", pd.DataFrame())
    sleeves = factor_summary.get("factor_sleeves", pd.DataFrame())
    ablation = factor_summary.get("factor_ablation", pd.DataFrame())
    weight_tuning = factor_summary.get("factor_weight_tuning", pd.DataFrame())
    weight_meta = factor_summary.get("weight_tuning_meta", {})
    recommendation = weight_meta.get("recommendation", {}) if isinstance(weight_meta, dict) else {}
    fig_prefix = f"../data/backtests/{mode_name}/figures"

    lines = [
        "# M1 Factor Analysis",
        "",
        "**Research use only — not investment advice.**",
        "",
        "## Factor Weights",
        "",
        f"M1 composite score uses momentum **{w.get('momentum', 0.45):.0%}**, trend **{w.get('trend', 0.25):.0%}**, "
        f"macro **{w.get('macro', 0.20):.0%}**, risk penalty **{w.get('risk_penalty', 0.10):.0%}**.",
        "",
        "## Per-Factor Information Coefficient",
        "",
        "Spearman rank correlation of each component score vs 4-week forward return.",
        "",
    ]
    if not factor_ic.empty:
        display_ic = factor_ic.copy()
        for col in ("ic_mean", "ic_std", "ic_hit_rate"):
            if col in display_ic.columns:
                display_ic[col] = display_ic[col].map(lambda x: _fmt_num(x))
        lines.append(_markdown_table(display_ic))
        lines.append("")
        lines.append(f"![Factor IC]({fig_prefix}/m1_factor_ic.png)")
        lines.append("")

    lines.extend(["## Factor Correlation Matrix", ""])
    if not corr.empty:
        lines.append(_markdown_table(corr.round(4).reset_index().rename(columns={"index": "factor"})))
        lines.append("")
        lines.append(f"![Factor correlation]({fig_prefix}/m1_factor_correlation_heatmap.png)")
        lines.append("")

    lines.extend(["## Factor Covariance Matrix", ""])
    if not cov.empty:
        lines.append(_markdown_table(cov.round(6).reset_index().rename(columns={"index": "factor"})))
        lines.append("")

    lines.extend(
        [
            "## Factor Sleeve Backtests",
            "",
            "Each row is a portfolio using only that factor family for top-K selection (risk penalty inverted).",
            "",
        ]
    )
    if not sleeves.empty:
        disp = sleeves.copy()
        for col in ("annualized_return", "max_drawdown"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
        if "sharpe" in disp.columns:
            disp["sharpe"] = disp["sharpe"].map(lambda x: _fmt_num(x))
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![Factor sleeves]({fig_prefix}/m1_factor_sleeves_cumulative.png)")
        lines.append("")

    lines.extend(["## Factor Ablation (zero one weight at a time)", ""])
    if not ablation.empty:
        disp = ablation.copy()
        for col in ("annualized_return", "max_drawdown"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_pct(x))
        if "sharpe" in disp.columns:
            disp["sharpe"] = disp["sharpe"].map(lambda x: _fmt_num(x))
        lines.append(_markdown_table(disp))
        lines.append("")

    lines.extend(
        [
            "## Weight Tuning (IC + ablation inspired)",
            "",
            "Compares preset and grid-searched M1 factor weights. Grid search selects by **train** Sharpe; "
            "test columns are out-of-sample. High momentum–trend correlation suggests shifting weight toward "
            "the stronger test-period IC factor (typically trend).",
            "",
        ]
    )
    if recommendation:
        rec_w = recommendation.get("weights", {})
        if isinstance(rec_w, str):
            import ast

            try:
                rec_w = ast.literal_eval(rec_w)
            except (SyntaxError, ValueError):
                rec_w = {}
        action = recommendation.get("config_action", "research_only")
        title = "### Weight recommendation"
        if action == "keep_baseline":
            title = "### Weight recommendation — **keep baseline** (walk-forward validated)"
        elif action == "apply_ic_weights":
            title = "### Weight recommendation — **apply IC weights** (walk-forward validated)"
        lines.extend([title, ""])
        if recommendation.get("walk_forward_validated"):
            lines.extend(
                [
                    f"- **Walk-forward:** {recommendation.get('walk_forward_n_folds', 'n/a')} folds; "
                    f"M1 wins {recommendation.get('walk_forward_m1_wins', 'n/a')}; "
                    f"mean M1 Sharpe Δ {recommendation.get('walk_forward_mean_m1_gain', float('nan')):+.4f}; "
                    f"mean ECDF Sharpe Δ {recommendation.get('walk_forward_mean_ecdf_gain', float('nan')):+.4f}",
                    f"- **Holdout variant:** `{recommendation.get('holdout_variant', 'n/a')}` "
                    f"(test Sharpe {recommendation.get('holdout_test_sharpe', float('nan')):.4f})",
                    "",
                ]
            )
        lines.extend(
            [
                f"- **Adopted variant:** `{recommendation.get('variant', 'baseline')}`",
                f"- **Weights:** momentum {rec_w.get('momentum', 0):.0%}, trend {rec_w.get('trend', 0):.0%}, "
                f"macro {rec_w.get('macro', 0):.0%}, risk penalty {rec_w.get('risk_penalty', 0):.0%}",
                f"- **Config action:** `{action}`",
                f"- **Rationale:** {recommendation.get('rationale', '')}",
                "",
            ]
        )
    if not weight_tuning.empty:
        disp = weight_tuning.copy()
        for col in ("train_ann_return", "test_ann_return", "train_max_drawdown", "test_max_drawdown"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
        for col in ("train_sharpe", "test_sharpe", "momentum", "trend", "macro", "risk_penalty"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        lines.append(_markdown_table(disp))
        lines.append("")
        lines.append(f"![Weight tuning test Sharpe]({fig_prefix}/m1_weight_tuning_test_sharpe.png)")
        lines.append("")

    interaction = sleeves[sleeves["sleeve"] == "interaction"] if not sleeves.empty else pd.DataFrame()
    if not interaction.empty and "interaction_excess_ann" in interaction.columns:
        val = interaction.iloc[0]["interaction_excess_ann"]
        lines.extend(
            [
                "## Interaction Term",
                "",
                f"Combined M1 excess minus sum of standalone factor sleeves: **{_fmt_pct(val)}** annualized. "
                "Positive values suggest factors reinforce; negative suggests overlap.",
                "",
            ]
        )
    report_path.write_text("\n".join(lines))


def generate_m2_diagnostics_report(
    m2_deep_summary: dict[str, Any],
    m2_metrics: dict[str, Any],
    report_path: Path,
    *,
    mode_name: str = "long_only",
    threshold: float = 0.55,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_prefix = f"../data/backtests/{mode_name}/figures"
    metrics = {**m2_metrics, **(m2_deep_summary.get("m2_metrics") or {})}
    calibration = m2_deep_summary.get("calibration_table", pd.DataFrame())
    deciles = m2_deep_summary.get("decile_returns", pd.DataFrame())
    importance = m2_deep_summary.get("feature_importance", pd.DataFrame())
    by_asset = m2_deep_summary.get("metrics_by_asset", pd.DataFrame())
    by_regime = m2_deep_summary.get("metrics_by_regime", pd.DataFrame())
    arch_bench = m2_deep_summary.get("architecture_benchmark", pd.DataFrame())

    auc = metrics.get("auc", float("nan"))
    base = metrics.get("base_rate", float("nan"))
    lines = [
        "# M2 Diagnostics & AUC-ROC Guide",
        "",
        "**Research use only — not investment advice.**",
        "",
        "## Classifier Metrics (Test Set)",
        "",
        _markdown_table(_m2_metrics_table(metrics)),
        "",
        "## Understanding AUC-ROC",
        "",
        "AUC-ROC measures **ranking quality**, not accuracy. If you randomly pick one winning trade and one losing trade, "
        f"AUC is the probability M2 assigns a higher `P(success)` to the winner. At **{_fmt_num(auc)}**, discrimination is "
        "only slightly above random (0.50).",
        "",
        "| AUC | Interpretation |",
        "| --- | --- |",
        "| 0.50 | Random ranking — no discrimination |",
        "| 0.55–0.60 | Weak but common for noisy financial labels |",
        "| 0.70+ | Moderate discrimination |",
        "",
        f"**Base rate** (fraction of profitable M1 trades): {_fmt_pct(base)}. When base rate ≠ 50%, **AUC-PR** "
        f"({_fmt_num(metrics.get('auc_pr', float('nan')))}) is often more informative than AUC-ROC.",
        "",
        "- **AUC vs Brier:** Brier scores calibration (predicted vs realized); AUC scores ranking. A model can be "
        "calibrated but still rank poorly.",
        "- **AUC vs precision/recall:** AUC is threshold-independent. At threshold "
        f"**{threshold}**, recall={_fmt_num(metrics.get('recall', float('nan')))} — "
        "if recall ≈ 1.0, binary M3 at that threshold approves all trades and adds no filter.",
        "- **Economic role:** M2 outputs probabilities only; **M3** converts them to bet fractions. "
        "Threshold approval at 0.55 is an M3 binary sizing rule, not M2 classification output.",
        "",
        f"![ROC and calibration]({fig_prefix}/m2_roc_calibration.png)",
        "",
        "## Calibration by Probability Bucket",
        "",
    ]
    if not calibration.empty:
        cal_disp = calibration.copy()
        cal_disp["mean_pred"] = cal_disp["mean_pred"].map(lambda x: _fmt_num(x))
        cal_disp["realized"] = cal_disp["realized"].map(lambda x: _fmt_pct(x))
        lines.append(_markdown_table(cal_disp))
    else:
        lines.append("_Insufficient data for calibration buckets._")
    lines.extend(["", "## Economic View: Return by Probability Decile", ""])
    if not deciles.empty:
        d_disp = deciles.copy()
        d_disp["mean_p_success"] = d_disp["mean_p_success"].map(lambda x: _fmt_num(x))
        d_disp["mean_trade_return"] = d_disp["mean_trade_return"].map(lambda x: _fmt_pct(x))
        d_disp["hit_rate"] = d_disp["hit_rate"].map(lambda x: _fmt_pct(x))
        lines.append(_markdown_table(d_disp))
        lines.append("")
        lines.append(f"![Decile returns]({fig_prefix}/m2_decile_returns.png)")
    lines.extend(["", "## Feature Importance (Top 15)", ""])
    if not importance.empty:
        imp_disp = importance.copy()
        imp_disp["coefficient"] = imp_disp["coefficient"].map(lambda x: _fmt_num(x, 4))
        lines.append(_markdown_table(imp_disp[["feature", "coefficient"]]))
        lines.append("")
        lines.append(f"![Feature importance]({fig_prefix}/m2_feature_importance.png)")
    lines.extend(["", "## Architecture Benchmark (train vs test AUC)", ""])
    if not arch_bench.empty:
        bench_disp = arch_bench.copy()
        for col in ("train_auc", "test_auc"):
            if col in bench_disp.columns:
                bench_disp[col] = bench_disp[col].map(lambda x: _fmt_num(x))
        lines.append(
            "Compares legacy global logistic regression against enriched features and per-asset heads."
        )
        lines.append("")
        lines.append(_markdown_table(bench_disp))
    else:
        lines.append("_Architecture benchmark not available._")
    lines.extend(["", "## Metrics by Asset", ""])
    if not by_asset.empty:
        lines.append(_markdown_table(by_asset.round(4)))
    lines.extend(["", "## Metrics by Regime Flag", ""])
    if not by_regime.empty:
        lines.append(_markdown_table(by_regime.round(4)))
    lines.append("")
    report_path.write_text("\n".join(lines))


def generate_market_regime_report(
    regime_summary: dict[str, Any],
    report_path: Path,
    *,
    mode_name: str = "long_only",
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_prefix = f"../data/backtests/{mode_name}/figures"
    transitions = regime_summary.get("regime_transitions", pd.DataFrame())
    perf = regime_summary.get("performance_by_regime", pd.DataFrame())
    ic_regime = regime_summary.get("m1_ic_by_regime", pd.DataFrame())
    auc_regime = regime_summary.get("m2_auc_by_regime", pd.DataFrame())
    context = regime_summary.get("regime_market_context", pd.DataFrame())

    lines = [
        "# Market & Regime Analysis",
        "",
        "**Research use only — not investment advice.**",
        "",
        "## Regime Feature Definitions",
        "",
        "| Flag / Series | Definition |",
        "| --- | --- |",
        "| `risk_off` | VIX above its 75th percentile (156-week rolling) |",
        "| `curve_inverted` | 10Y–2Y Treasury spread < 0 |",
        "| `inflation_up` | CPI YoY above its 156-week median |",
        "| `growth_down` | Industrial production YoY below its 156-week median |",
        "| `vix_level`, `credit_stress`, `yield_curve` | Continuous macro/risk levels (lagged) |",
        "",
        f"![Regime timeline]({fig_prefix}/regime_timeline.png)",
        "",
        f"![VIX and flags]({fig_prefix}/vix_and_flags.png)",
        "",
        "## Regime Transitions",
        "",
    ]
    if not transitions.empty:
        lines.append(_markdown_table(transitions.round(2)))
    lines.extend(["", "## Strategy Performance by Regime", ""])
    if not perf.empty:
        p_disp = perf.copy()
        p_disp["annualized_return"] = p_disp["annualized_return"].map(lambda x: _fmt_pct(x))
        p_disp["sharpe"] = p_disp["sharpe"].map(lambda x: _fmt_num(x))
        p_disp["hit_rate"] = p_disp["hit_rate"].map(lambda x: _fmt_pct(x))
        lines.append(_markdown_table(p_disp))
        lines.append("")
        lines.append(f"![Performance heatmap]({fig_prefix}/performance_by_regime_heatmap.png)")
    lines.extend(["", "## M1 IC by Regime (Test)", ""])
    if not ic_regime.empty:
        lines.append(_markdown_table(ic_regime.round(4)))
    lines.extend(["", "## M2 AUC by Regime (Test)", ""])
    if not auc_regime.empty:
        lines.append(_markdown_table(auc_regime.round(4)))
    lines.extend(["", "## Train vs Test Macro Context", ""])
    if not context.empty:
        lines.append(_markdown_table(context.round(4)))
    lines.append("")
    report_path.write_text("\n".join(lines))


def generate_companion_reports(
    mode_results: list[Any],
    reports_root: Path,
    *,
    cfg: PipelineConfig | None = None,
) -> None:
    """Write companion markdown reports from long_only mode diagnostics."""
    long_mode = next((m for m in mode_results if m.mode_name == "long_only"), None)
    if long_mode is None:
        return
    fs = getattr(long_mode, "factor_summary", None) or {}
    rs = getattr(long_mode, "regime_summary", None) or {}
    m2d = getattr(long_mode, "m2_deep_summary", None) or {}
    m3d = getattr(long_mode, "m3_summary", None) or {}
    threshold = cfg.m2.threshold if cfg else 0.55

    if fs:
        generate_m1_factor_analysis_report(
            fs, reports_root / "m1_factor_analysis.md", cfg=cfg, mode_name="long_only"
        )
    if m2d or long_mode.m2_metrics:
        generate_m2_diagnostics_report(
            m2d,
            long_mode.m2_metrics,
            reports_root / "m2_diagnostics.md",
            mode_name="long_only",
            threshold=threshold,
        )
    if rs:
        generate_market_regime_report(rs, reports_root / "market_regime_analysis.md", mode_name="long_only")
    if m3d:
        generate_m3_allocation_report(m3d, reports_root / "m3_allocation_analysis.md", mode_name="long_only")
    m3_thresh = reports_root / "m3_threshold_analysis.md"
    if not m3_thresh.exists() and cfg is not None:
        try:
            from src.m3_threshold_research import run_m3_threshold_research

            run_m3_threshold_research(Path("config/config.yaml"), project_root=reports_root.parent)
        except Exception:
            logger.exception("M3 threshold sweep failed; skipping report")
    evald = getattr(long_mode, "eval_summary", None) or {}
    if evald:
        from src.evaluation import generate_evaluation_report

        generate_evaluation_report(
            evald,
            reports_root / "evaluation_analysis.md",
            mode_name="long_only",
            cfg=cfg,
        )
        wf = evald.get("walk_forward", pd.DataFrame())
        stability = evald.get("walk_forward_stability")
        if stability and not wf.empty:
            from src.evaluation import generate_walk_forward_analysis_report

            generate_walk_forward_analysis_report(
                wf,
                stability,
                reports_root / "walk_forward_analysis.md",
                mode_name="long_only",
                cfg=cfg,
                tc_sensitivity=evald.get("transaction_cost_sensitivity"),
            )


def generate_dual_mode_report(
    mode_results: list[Any],
    report_path: Path,
    *,
    final_dir: Path | None = None,
    mode_comparison_dir: Path | None = None,
    cfg: PipelineConfig | None = None,
    effective_start: str | None = None,
    effective_end: str | None = None,
    asset_analysis_sections: list[str] | None = None,
) -> None:
    """Build a final report comparing long-only and long-short M1 runs."""
    reports_root = report_path.parent
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_dir = final_dir or reports_root / "final"
    mode_comparison_dir = mode_comparison_dir or reports_root / "mode_comparison"
    final_dir.mkdir(parents=True, exist_ok=True)
    mode_comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_chart = f"mode_comparison/{save_m1_mode_comparison_chart(mode_results, mode_comparison_dir)}"
    m1_m2_comparison_file = save_m1_signal_m2_mode_comparison_chart(
    [
        (m.mode_name, m.m1_signal_analysis)
        for m in mode_results
        if getattr(m, "m1_signal_analysis", None)
    ],
    mode_comparison_dir / "m2_m1_signal_comparison.png",
)

    m1_m2_comparison_chart = f"mode_comparison/{m1_m2_comparison_file}"

    import shutil

    for mode in mode_results:
        analysis = getattr(mode, "m1_signal_analysis", None)
        mode_chart_dir = final_dir / mode.mode_name
        mode_chart_dir.mkdir(parents=True, exist_ok=True)
        if analysis:
            chart_path = mode_chart_dir / "m2_m1_signal_analysis.png"
            save_m1_signal_m2_chart(analysis, chart_path)
            mode.m1_signal_chart = chart_path.name
            mode.m1_signal_chart_rel = f"final/{mode.mode_name}/{chart_path.name}"
        bt_figures = getattr(mode, "backtests_dir", None)
        if bt_figures is not None:
            src_fig = Path(bt_figures) / "figures"
            if src_fig.exists():
                dst_fig = mode_chart_dir / "figures"
                dst_fig.mkdir(parents=True, exist_ok=True)
                for name in ("m1_exposure_over_time.png", "m1_threshold_sensitivity.png"):
                    src = src_fig / name
                    if src.exists():
                        shutil.copy2(src, dst_fig / name)

    lines = [
        "# Final Report: AI-Augmented Multi-Asset Meta-Labeling Pipeline",
        "",
        "This run executes the pipeline **twice**: once with M1 **long-only** (no short signals) "
        "and once with M1 **long/short** enabled.",
        "",
        "**Research use only — not investment advice.**",
        "",
        "## Sample Period",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Effective start | {effective_start or 'N/A'} |",
        f"| Effective end | {effective_end or 'N/A'} |",
    ]
    if cfg is not None:
        test_end_disp = cfg.split.test_end or "latest"
        universe = "all 7 index sleeves each week" if cfg.split.require_full_universe else "partial (per-sleeve availability)"
        lines.extend(
            [
                f"| Data download from | {cfg.data_start_resolved()} |",
                f"| Train period (requested) | {cfg.split.train_start} to {cfg.split.train_end} |",
                f"| Test period (M2 evaluation) | {cfg.split.test_start} to {test_end_disp} |",
                f"| Universe mode | {universe} |",
                f"| Assets | {', '.join(cfg.assets.tickers)} |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"| Train period | (see config) |",
                f"| Test period (M2 evaluation) | (see config) |",
                f"| Assets | SP500, MSCI_EAFE, MSCI_EM, UST_7_10, US_HIGH_YIELD, GOLD_SPOT, US_REIT |",
                "",
            ]
        )

    if cfg is not None:
        lines.extend(build_performance_parameters_section(cfg))

    if asset_analysis_sections:
        lines.extend(asset_analysis_sections)

    lines.extend(build_m1_signal_m2_report_section(mode_results, comparison_chart=m1_m2_comparison_chart))

    lines.extend(
        [
        "## M1 Mode Comparison (M1 Only)",
        "",
        "| Mode | Ann. Return | Sharpe | Max Drawdown |",
        "| --- | --- | --- | --- |",
        ]
    )

    for mode in mode_results:
        m1_row = mode.metrics_table.set_index("strategy").loc["m1_only"]
        label = MODE_LABELS.get(mode.mode_name, mode.mode_name)
        lines.append(
            f"| {label} | {_fmt_pct(m1_row['annualized_return'])} | {_fmt_num(m1_row['sharpe'])} | {_fmt_pct(m1_row['max_drawdown'])} |"
        )

    lines.extend(
        [
            "",
            f"![M1 mode comparison]({comparison_chart})",
            "",
            "*Left: cumulative M1-only returns. Right: return, Sharpe (×10), and drawdown by mode.*",
            "",
        ]
    )

    for mode in mode_results:
        if mode.mode_name == "long_only":
            lines.extend(
                build_m1_exposure_report_section(
                    getattr(mode, "m1_exposure_analysis", None),
                    getattr(mode, "per_asset_ic", None),
                    chart_rel=getattr(mode, "m1_exposure_chart_rel", None),
                    sens_chart_rel=getattr(mode, "m1_sens_chart_rel", None),
                )
            )
            break

    for mode in mode_results:
        label = MODE_LABELS.get(mode.mode_name, mode.mode_name)
        chart_prefix = f"final/{mode.mode_name}/"
        save_report_charts(mode.results, mode.m2_metrics, final_dir, subdir=mode.mode_name)
        display_table = format_metrics_table_for_report(mode.metrics_table)
        test_start = cfg.split.test_start if cfg is not None else None
        test_end = cfg.split.test_end if cfg is not None else None
        test_metrics_table = build_metrics_table_on_period(
            mode.results,
            start=test_start,
            end=test_end,
        )
        test_display_table = format_metrics_table_for_report(test_metrics_table)

        lines.extend(
            [
                f"## Results: {label}",
                "",
                f"`allow_short={mode.allow_short}` — outputs in `data/backtests/{mode.mode_name}/`",
                "",
                "### Full-Sample Strategy Metrics",
                "",
                "These metrics cover the full effective panel, including train and test periods. "
                "They are useful for long-run behavior but should not be read as pure OOS performance.",
                "",
                _markdown_table(display_table),
                "",
                "### Test-Period Strategy Metrics",
                "",
                f"These metrics start at `{test_start or 'configured test_start'}` and are the cleanest portfolio-level OOS view in this report.",
                "",
                _markdown_table(test_display_table),
                "",
                f"### Charts ({label})",
                "",
            ]
        )
        for chart in [
            "strategy_cumulative_returns.png",
            "strategy_drawdown.png",
            "strategy_sharpe_comparison.png",
            "strategy_risk_return.png",
            "m2_classification_summary.png",
            "m2_m1_signal_analysis.png",
        ]:
            lines.append(f"![{chart}]({chart_prefix}{chart})")
            lines.append("")

        lines.extend(
            [
                f"### M2 Quality — {label} (Test Set)",
                "",
                _markdown_table(_m2_metrics_table(mode.m2_metrics)),
                "",
            ]
        )

    lines.extend(
        [
            "### How to read the metrics",
            "",
            "| Metric | Interpretation |",
            "| --- | --- |",
            "| **Ann. Return** | Geometric average yearly portfolio return after transaction costs |",
            "| **Ann. Volatility** | Standard deviation of weekly returns, scaled to a year |",
            "| **Sharpe** | Return per unit of risk (higher is better; assumes 0% risk-free rate) |",
            "| **Max Drawdown** | Largest peak-to-trough loss over the displayed period |",
            "| **Excess vs EW** | Strategy return minus equal-weight benchmark return |",
            "| **Info Ratio** | Consistency of outperformance vs equal-weight |",
            "| **Weekly Hit Rate** | Fraction of weeks with positive net strategy return |",
            "",
        ]
    )
    lines.extend(build_deep_diagnostics_summary_section(mode_results))
    lines.extend(
        [
            "## Key Takeaways",
            "",
            "1. **Long-only M1** avoids short exposure, which often hurts in upward-trending equity samples.",
            "2. **Long/short M1** can increase activity but shorts may reduce returns if poorly timed.",
            "3. **M2 meta-labeling** adjusts position size on top of whichever M1 mode is used.",
            "4. Compare both modes above to see whether shorts add value in this universe.",
            "",
            "## Look-Ahead Controls",
            "",
            "- Features use only data available at signal time (`shift(1)` on rolling windows)",
            f"- Macro series lagged {cfg.features.macro_lag_weeks} weeks to approximate release delay"
            if cfg is not None
            else "- Macro series lagged (see config features.macro_lag_weeks) to approximate release delay",
            f"- Strict chronological train/test split (train {cfg.split.train_start}–{cfg.split.train_end}, "
            f"test {cfg.split.test_start}–{cfg.split.test_end or 'latest'})"
            if cfg is not None
            else "- Strict chronological train/test split (see config split section)",
            "",
            "## Limitations",
            "",
            "- yfinance and FRED are research-grade fallbacks, not institutional data",
            "- Data provenance, ETL, validation, cache behavior, and fallback caveats are documented in `../DATA_SOURCES_AND_ETL.md`",
            "- Past performance does not predict future results",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def run_diagnostics(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    cfg_threshold: float,
    output_dir: Path,
    *,
    cfg: PipelineConfig | None = None,
    returns_wide: pd.DataFrame | None = None,
    train_panel: pd.DataFrame | None = None,
    m1_model: object | None = None,
    m2_model: object | None = None,
    m1_weight_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics_table = build_metrics_table(results)
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)

    ic = compute_ic(test_panel)
    ic_mean = float(ic.mean()) if not ic.empty else float("nan")
    per_asset_ic = compute_per_asset_ic(test_panel)
    if not per_asset_ic.empty:
        per_asset_ic.to_csv(output_dir / "m1_per_asset_ic.csv", index=False)

    m2_metrics = m2_classification_metrics(
        test_panel["meta_label"],
        test_panel["p_success"],
        threshold=cfg_threshold,
    )
    if not ic.empty:
        m2_metrics["information_coefficient_mean"] = ic_mean

    factor_summary: dict[str, Any] = {}
    regime_summary: dict[str, Any] = {}
    m2_deep_summary: dict[str, Any] = {}
    m3_summary: dict[str, Any] = {}
    if cfg is not None and returns_wide is not None and train_panel is not None:
        from src.factor_analysis import run_factor_analysis
        from src.feature_engineering import get_feature_columns
        from src.m3_diagnostics import run_m3_diagnostics
        from src.regime_analysis import run_regime_analysis

        feature_cols = get_feature_columns(panel.reset_index())
        feature_cols = [c for c in feature_cols if c in panel.columns]
        train_proba = train_panel.loc[train_panel["M1_signal"] != 0, "p_success"]
        factor_summary = run_factor_analysis(
            panel,
            train_panel,
            test_panel,
            returns_wide,
            cfg,
            output_dir,
            m1_model=m1_model,
            feature_cols=feature_cols,
            m1_weight_decision=m1_weight_decision,
        )
        regime_summary = run_regime_analysis(
            panel,
            train_panel,
            test_panel,
            results,
            output_dir,
            cfg_train_end=cfg.split.train_end,
            cfg_test_start=cfg.split.test_start,
            m2_threshold=cfg_threshold,
        )
        if m2_model is not None:
            m2_deep_summary = run_m2_deep_diagnostics(
                test_panel, m2_model, cfg, cfg_threshold, output_dir, train_panel=train_panel
            )
            if m2_deep_summary.get("m2_metrics"):
                m2_metrics.update(
                    {k: v for k, v in m2_deep_summary["m2_metrics"].items() if k not in m2_metrics}
                )
        m3_summary = run_m3_diagnostics(
            panel,
            train_panel,
            test_panel,
            cfg,
            output_dir,
            train_proba=train_proba,
        )

    m1_signal_analysis = analyze_m1_signal_m2_performance(
        test_panel, cfg_threshold, period_label="test"
    )
    m1_signal_analysis["by_signal"].to_csv(output_dir / "m1_signal_m2_analysis.csv", index=False)

    m1_exposure_analysis = None
    threshold_sensitivity = pd.DataFrame()
    if "m1_only" in results:
        bench = results.get("equal_weight_1_7")
        m1_exposure_analysis = analyze_m1_exposure(results["m1_only"], bench)
        pd.DataFrame([m1_exposure_analysis["summary"]]).to_csv(output_dir / "m1_exposure_summary.csv", index=False)

    if cfg is not None and returns_wide is not None and train_panel is not None:
        threshold_sensitivity = threshold_sensitivity_summary(train_panel, returns_wide, cfg)
        if not threshold_sensitivity.empty:
            threshold_sensitivity.to_csv(output_dir / "m1_threshold_sensitivity.csv", index=False)

    figures_dir = output_dir / "figures"
    save_figures(results, test_panel, m2_metrics, ic, figures_dir)
    save_m1_signal_m2_chart(m1_signal_analysis, figures_dir / "m2_m1_signal_analysis.png")

    exposure_chart = None
    sens_chart = None
    if m1_exposure_analysis is not None:
        saved = save_m1_exposure_charts(m1_exposure_analysis, figures_dir)
        exposure_chart = saved[0] if saved else None
    if not threshold_sensitivity.empty:
        sens_chart = save_threshold_sensitivity_chart(threshold_sensitivity, figures_dir)

    summary = {
        "metrics_table": metrics_table.to_dict(orient="records"),
        "m2_metrics": m2_metrics,
        "m1_signal_analysis": m1_signal_analysis,
        "ic_mean": ic_mean,
        "per_asset_ic": per_asset_ic,
        "m1_exposure_analysis": m1_exposure_analysis,
        "threshold_sensitivity": threshold_sensitivity,
        "exposure_chart": exposure_chart,
        "sens_chart": sens_chart,
        "factor_summary": factor_summary,
        "regime_summary": regime_summary,
        "m2_deep_summary": m2_deep_summary,
        "m3_summary": m3_summary,
    }
    json_summary = {
        **summary,
        "m1_signal_analysis": {
            "period": m1_signal_analysis["period"],
            "threshold": m1_signal_analysis["threshold"],
            "by_signal": m1_signal_analysis["by_signal"].to_dict(orient="records"),
        },
        "per_asset_ic": per_asset_ic.to_dict(orient="records") if not per_asset_ic.empty else [],
        "m1_exposure_analysis": m1_exposure_analysis["summary"] if m1_exposure_analysis else {},
        "threshold_sensitivity": threshold_sensitivity.to_dict(orient="records")
        if not threshold_sensitivity.empty
        else [],
    }
    with (output_dir / "diagnostics_summary.json").open("w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    return summary
