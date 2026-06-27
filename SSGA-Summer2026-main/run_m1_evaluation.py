"""M1 Evaluation — evaluation data (e1, e2 ... ek) that flows from M1 to M2.

Computes per-week, per-asset M1 evaluation metrics:
  1. Rolling hit rate     — what % of M1's bets paid off over trailing 12 weeks
  2. Rolling IR           — M1's information ratio over a trailing window
  3. Signal strength      — how strong/confident was M1's score each week
  4. Per-factor contribution — which factor drove the signal each week

These are the evaluation data M2 should receive, NOT M1's raw factor scores.

Usage:
    python run_m1_evaluation.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest import equal_weight_returns, metrics, portfolio_returns
from src.config import load_config
from src.data import IndexFileProvider, ingest_macro_data, ingest_market_data
from src.features import (
    get_vix_series, high_proximity_score, macro_wide, momentum_score,
    pivot_prices, regime_features, relative_momentum_score, reversal_score,
    risk_score, trend_score,
)
from src.m1 import M1Model
from src.m2 import build_meta_labels
from src.portfolio import apply_vol_target, build_weights, cost_drag

logging.basicConfig(level=logging.ERROR, format="%(message)s")
pd.set_option("display.width", 160)


def rolling_hit_rate(labels: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Trailing hit rate of M1's bets per asset.
    For each (date, asset): what % of M1's last N bets on this asset paid off?
    Shifted by 1 so M2 never sees the current week's outcome — no look-ahead."""
    return labels.rolling(window, min_periods=4).mean().shift(1)


def rolling_ir(active_returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Trailing information ratio of M1's active bets per asset.
    IR = mean(active_ret) / std(active_ret) * sqrt(52).
    Shifted by 1 — no look-ahead."""
    mean_r = active_returns.rolling(window, min_periods=4).mean().shift(1)
    std_r  = active_returns.rolling(window, min_periods=4).std().shift(1)
    return (mean_r / std_r.replace(0, np.nan)) * np.sqrt(52)


def signal_strength(score: pd.DataFrame) -> pd.DataFrame:
    """Absolute value of M1's cross-sectional score — how confident is M1?
    High = strong conviction. Low = weak/noisy signal.
    Shifted by 1 — no look-ahead."""
    return score.abs().shift(1)


def factor_contribution(factors: dict[str, pd.DataFrame],
                        weights: dict[str, float]) -> pd.DataFrame:
    """For each week, which factor contributed most to the M1 score?
    Returns the dominant factor name as a string per (date, asset)."""
    contributions = {
        name: weights.get(name, 0.0) * frame.abs()
        for name, frame in factors.items()
    }
    contrib_df = pd.concat(contributions, axis=1)
    # dominant factor = the one with the highest weighted absolute contribution
    dominant = contrib_df.groupby(level=1, axis=1).idxmax(axis=1)
    return dominant


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main() -> None:
    cfg = load_config()

    # ── data ──────────────────────────────────────────────────────────────────
    provider = IndexFileProvider(cfg.raw_dir / "index") if cfg.data.use_index_signal else None
    market = ingest_market_data(
        cfg.data.universe, cfg.vix_ticker, cfg.data.data_start, None,
        cfg.raw_dir, cfg.processed_dir, provider=provider,
        use_cache=not cfg.data.use_index_signal,
    )
    macro = ingest_macro_data(
        cfg.data.macro_series, cfg.data.data_start, None,
        cfg.raw_dir, cfg.processed_dir, market_weekly=market,
    )

    prices  = pivot_prices(market[market["ticker"].isin(cfg.data.universe)])
    returns = prices.pct_change()

    # ── M1 factors ────────────────────────────────────────────────────────────
    mom     = momentum_score(prices, cfg.m1.momentum_windows)
    trd     = trend_score(prices, cfg.m1.trend_windows)
    rel_m   = relative_momentum_score(prices, window=12)
    hi_prox = high_proximity_score(prices, window=52)
    rev     = reversal_score(prices)

    factor_weights = {
        "technical":      0.40,
        "rel_momentum":   0.30,
        "high_proximity": 0.20,
        "reversal":       0.10,
    }

    factors = {
        "technical":      0.5 * mom + 0.5 * trd,
        "rel_momentum":   rel_m,
        "high_proximity": hi_prox,
        "reversal":       rev,
    }

    m1    = M1Model(cfg.m1)
    score = m1.score({k: v for k, v in zip(factor_weights.keys(), factors.values())})

    w_m1          = build_weights(score, cfg)
    w_m1          = apply_vol_target(w_m1, returns, cfg)
    benchmark_w   = 1.0 / len(prices.columns)
    labels        = build_meta_labels(prices, w_m1, cfg.labels.horizon_weeks,
                                      benchmark_w, cfg.labels.positive_threshold)

    # active returns per asset (M1 tilt return vs equal weight)
    ew_ret       = returns.mean(axis=1)
    active_ret   = returns.sub(ew_ret, axis=0)

    # ── evaluation data (e1...ek) ─────────────────────────────────────────────
    hit_rate = rolling_hit_rate(labels, window=12)
    roll_ir  = rolling_ir(active_ret, window=12)
    strength = signal_strength(score)

    test = cfg.split.test_start

    # ── 1. Rolling Hit Rate ───────────────────────────────────────────────────
    print_section("E1: ROLLING 12-WEEK HIT RATE (per asset)")
    print("  What % of M1's last 12 bets paid off per asset?\n")
    print("  Full sample mean hit rate per asset:")
    print(hit_rate.mean().to_string(float_format=lambda v: f"  {v:.3f}"))
    print(f"\n  OOS mean hit rate (from {test}):")
    print(hit_rate[hit_rate.index >= test].mean().to_string(float_format=lambda v: f"  {v:.3f}"))
    print("\n  Interpretation: >0.5 = M1 correct more than half the time on this asset")

    # ── 2. Rolling IR ─────────────────────────────────────────────────────────
    print_section("E2: ROLLING 12-WEEK INFORMATION RATIO (per asset)")
    print("  M1's trailing IR per asset — how efficiently is it generating alpha?\n")
    print("  Full sample mean rolling IR per asset:")
    print(roll_ir.mean().to_string(float_format=lambda v: f"  {v:.3f}"))
    print(f"\n  OOS mean rolling IR (from {test}):")
    print(roll_ir[roll_ir.index >= test].mean().to_string(float_format=lambda v: f"  {v:.3f}"))
    print("\n  Interpretation: IR > 0 = M1 adding value on this asset recently")

    # ── 3. Signal Strength ────────────────────────────────────────────────────
    print_section("E3: M1 SIGNAL STRENGTH (per asset)")
    print("  How confident/strong is M1's score each week?\n")
    print("  Full sample mean signal strength per asset:")
    print(strength.mean().to_string(float_format=lambda v: f"  {v:.3f}"))
    print(f"\n  OOS mean signal strength (from {test}):")
    print(strength[strength.index >= test].mean().to_string(float_format=lambda v: f"  {v:.3f}"))
    print("\n  Interpretation: higher = M1 more decisive, lower = M1 unsure")

    # ── 4. Overall M1 hit rate by year ────────────────────────────────────────
    print_section("E4: M1 HIT RATE BY YEAR (universe average)")
    print("  Did M1 work better in some years than others?\n")
    yearly = labels.mean(axis=1).groupby(labels.index.year).mean()
    for year, hr in yearly.items():
        bar = "█" * int(hr * 20) if not pd.isna(hr) else ""
        flag = " <- OOS" if year >= int(test[:4]) else ""
        print(f"  {year}: {hr:.3f}  {bar}{flag}")
    print("\n  Interpretation: years where hit rate > 0.5 = M1 worked that year")


if __name__ == "__main__":
    main()
