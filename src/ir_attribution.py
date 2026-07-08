"""Information-ratio attribution vs equal-weight benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, STRATEGY_M1_M2_M3_ECDF
from src.diagnostics import (
    WEEKS_PER_YEAR,
    annualized_return,
    information_ratio,
    strategy_metrics,
    strategy_metrics_on_period,
)
from src.regime_analysis import REGIME_FLAG_COLS, _panel_to_date_level

ATTRIBUTION_STRATEGIES = (
    "m1_only",
    STRATEGY_M1_M2_M3_ECDF,
    "m1_m2_m3_linear",
    "m1_m2_m3_binary",
)


def decompose_ir(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """IR components on aligned weekly returns."""
    s = strategy.dropna()
    b = benchmark.reindex(s.index).fillna(0.0)
    active = s - b
    te = float(active.std() * np.sqrt(WEEKS_PER_YEAR))
    mean_active_ann = float(active.mean() * WEEKS_PER_YEAR)
    ir = information_ratio(s, b)
    corr = float(s.corr(b)) if len(s) > 1 else float("nan")
    ew_won = float((active < 0).mean()) if len(active) else float("nan")
    return {
        "mean_active_return_ann": mean_active_ann,
        "tracking_error_ann": te,
        "information_ratio": ir,
        "return_correlation_vs_ew": corr,
        "pct_weeks_ew_outperformed": ew_won,
    }


def exposure_vs_benchmark(
    weights: pd.DataFrame,
    ew_weights: pd.DataFrame,
) -> dict[str, float]:
    gross = weights.abs().sum(axis=1)
    ew_gross = ew_weights.abs().sum(axis=1).reindex(gross.index).fillna(1.0)
    diff = gross - ew_gross
    return {
        "mean_gross_exposure": float(gross.mean()),
        "mean_gross_vs_ew": float(diff.mean()),
        "median_gross_exposure": float(gross.median()),
        "pct_weeks_below_half_invested": float((gross < 0.5).mean()),
    }


def active_return_by_regime(
    strategy: pd.Series,
    benchmark: pd.Series,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Mean annualized active return and IR by binary regime flag."""
    if timeline.empty:
        return pd.DataFrame()
    tl = timeline.set_index("date") if "date" in timeline.columns else timeline
    tl.index = pd.to_datetime(tl.index)
    rows: list[dict[str, Any]] = []
    s = strategy.copy()
    s.index = pd.to_datetime(s.index)
    b = benchmark.reindex(s.index).fillna(0.0)
    active = s - b

    for flag in REGIME_FLAG_COLS:
        if flag not in tl.columns:
            continue
        for val, label in [(1, "on"), (0, "off")]:
            dates = tl.index[tl[flag].fillna(0).astype(int) == val]
            sub = active[active.index.isin(dates)]
            if len(sub) < 4:
                continue
            rows.append(
                {
                    "regime_flag": flag,
                    "regime_state": label,
                    "n_weeks": len(sub),
                    "mean_active_return_ann": float(sub.mean() * WEEKS_PER_YEAR),
                    "information_ratio": information_ratio(
                        s[s.index.isin(dates)],
                        b[b.index.isin(dates)],
                    ),
                    "pct_weeks_ew_outperformed": float((sub < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def attribute_strategy_ir(
    result: BacktestResult,
    benchmark: BacktestResult,
    *,
    period_label: str,
    ew_weights: pd.DataFrame | None = None,
    timeline: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Full IR attribution for one strategy vs EW on a period."""
    r = result.returns.copy()
    r.index = pd.to_datetime(r.index)
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    bench_r = benchmark.returns.reindex(r.index).fillna(0.0)

    base = strategy_metrics_on_period(r, start=None, end=None)
    base["excess_return_vs_benchmark"] = base["annualized_return"] - annualized_return(bench_r)
    base["information_ratio"] = information_ratio(r, bench_r)
    ir_parts = decompose_ir(r, bench_r)

    out: dict[str, Any] = {
        "strategy": result.name,
        "period": period_label,
        **base,
        **ir_parts,
    }

    if ew_weights is not None:
        w = result.weights.reindex(r.index).ffill().fillna(0.0)
        ew_w = ew_weights.reindex(r.index).ffill().fillna(0.0)
        out.update(exposure_vs_benchmark(w, ew_w))

    if timeline is not None and not timeline.empty:
        regime_rows = active_return_by_regime(r, bench_r, timeline)
        out["regime_attribution"] = regime_rows

    return out


def run_ir_attribution(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    *,
    test_start: str,
    test_end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Attribute IR for key strategies on full and test periods."""
    bench = results.get("equal_weight_1_7")
    if bench is None:
        raise ValueError("equal_weight_1_7 required for IR attribution")

    ew_weights = bench.weights
    timeline = _panel_to_date_level(panel)

    summary_rows: list[dict[str, Any]] = []
    regime_frames: list[pd.DataFrame] = []

    for strat in ATTRIBUTION_STRATEGIES:
        res = results.get(strat)
        if res is None:
            continue
        for period_label, start, end in [
            ("full", None, None),
            ("test", test_start, test_end),
        ]:
            attr = attribute_strategy_ir(
                res,
                bench,
                period_label=period_label,
                ew_weights=ew_weights,
                timeline=timeline,
                start=start,
                end=end,
            )
            regime_df = attr.pop("regime_attribution", None)
            if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
                regime_df = regime_df.copy()
                regime_df["strategy"] = strat
                regime_df["period"] = period_label
                regime_frames.append(regime_df)
            summary_rows.append(attr)

    summary = pd.DataFrame(summary_rows)
    regime = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    return summary, {"regime": regime}


def generate_ir_attribution_report(
    summary: pd.DataFrame,
    regime: pd.DataFrame,
    report_path: Path,
    *,
    test_start: str,
) -> None:
    from src.diagnostics import _fmt_num, _fmt_pct, _markdown_table

    report_path.parent.mkdir(parents=True, exist_ok=True)
    test = summary[summary["period"] == "test"].copy() if not summary.empty else pd.DataFrame()

    lines = [
        "# IR Attribution Analysis",
        "",
        "**Research use only — not investment advice.**",
        "",
        "Information Ratio (IR) measures **consistency of beating equal-weight (EW)** week-by-week:",
        "`IR = mean(strategy − EW) × √52 / tracking_error`.",
        "",
        "M2/M3 can **raise Sharpe** while **lowering IR** when the strategy deploys less capital",
        "or lags broad EW rallies — especially in selective top-K sleeves.",
        "",
        f"**Test window:** `{test_start}` onward",
        "",
        "## Test-period IR vs EW",
        "",
    ]

    if not test.empty:
        disp = test[
            [
                "strategy",
                "annualized_return",
                "sharpe",
                "excess_return_vs_benchmark",
                "information_ratio",
                "mean_active_return_ann",
                "tracking_error_ann",
                "mean_gross_exposure",
                "mean_gross_vs_ew",
                "return_correlation_vs_ew",
                "pct_weeks_ew_outperformed",
            ]
        ].copy()
        for col in (
            "annualized_return",
            "excess_return_vs_benchmark",
            "mean_active_return_ann",
            "tracking_error_ann",
            "mean_gross_exposure",
            "mean_gross_vs_ew",
            "pct_weeks_ew_outperformed",
        ):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
        for col in ("sharpe", "information_ratio", "return_correlation_vs_ew"):
            if col in disp.columns:
                disp[col] = disp[col].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
        lines.append(_markdown_table(disp))
        lines.append("")

    lines.extend(["## Active return by regime (test)", ""])
    if not regime.empty:
        rt = regime[regime["period"] == "test"].copy()
        if not rt.empty:
            for col in ("mean_active_return_ann", "pct_weeks_ew_outperformed"):
                if col in rt.columns:
                    rt[col] = rt[col].map(lambda x: _fmt_pct(x) if pd.notna(x) else "—")
            if "information_ratio" in rt.columns:
                rt["information_ratio"] = rt["information_ratio"].map(
                    lambda x: _fmt_num(x) if pd.notna(x) else "—"
                )
            lines.append(_markdown_table(rt))
            lines.append("")

    m1 = test[test["strategy"] == "m1_only"] if not test.empty else pd.DataFrame()
    ecdf = test[test["strategy"] == STRATEGY_M1_M2_M3_ECDF] if not test.empty else pd.DataFrame()
    lines.extend(["## Key findings", ""])
    if not m1.empty and not ecdf.empty:
        m1r, ecdfr = m1.iloc[0], ecdf.iloc[0]
        lines.append(
            f"- **M1-only** test IR {_fmt_num(m1r.get('information_ratio'))} with "
            f"{_fmt_pct(m1r.get('excess_return_vs_benchmark'))} excess return vs EW."
        )
        lines.append(
            f"- **ECDF** test IR {_fmt_num(ecdfr.get('information_ratio'))} — Sharpe "
            f"{_fmt_num(ecdfr.get('sharpe'))} vs M1 {_fmt_num(m1r.get('sharpe'))}, but "
            f"gross exposure ~{_fmt_pct(ecdfr.get('mean_gross_exposure'))} "
            f"({_fmt_pct(ecdfr.get('mean_gross_vs_ew'))} vs EW)."
        )
        lines.append(
            f"- EW outperforms on ~{_fmt_pct(ecdfr.get('pct_weeks_ew_outperformed'))} of ECDF weeks "
            "(active return < 0)."
        )
    lines.extend(
        [
            "",
            "Related: [ir_improvement_research.md](ir_improvement_research.md) · [TERMINOLOGY.md](../TERMINOLOGY.md)",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
