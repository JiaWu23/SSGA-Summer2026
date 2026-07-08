"""M1 Comparison Runner — Points 3 & 4 for external meeting.

Runs four M1 variants side by side and prints a comparison table:
  1. Equal-weight benchmark (baseline)
  2. Momentum-only M1
  3. Trend-only M1
  4. 4-Factor M1 (momentum + trend + rel_momentum + high_proximity + reversal)

Usage:
    python run_comparison.py

No config changes needed — variants are defined directly in this script.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.backtest import equal_weight_returns, metrics, portfolio_returns, static_portfolio_returns
from src.config import load_config
from src.data import IndexFileProvider, ingest_macro_data, ingest_market_data
from src.features import (
    get_vix_series, high_proximity_score, macro_wide, momentum_score,
    pivot_prices, regime_features, relative_momentum_score, reversal_score,
    risk_score, trend_score,
)
from src.m1 import M1Model
from src.portfolio import apply_vol_target, build_weights, cost_drag

logging.basicConfig(level=logging.ERROR, format="%(message)s")
pd.set_option("display.width", 160)


def run_variant(name: str, score: pd.DataFrame, cfg, returns: pd.DataFrame) -> tuple[str, dict]:
    """Run one M1 variant and return its metrics."""
    w = build_weights(score, cfg)   # converts M1 score into portfolio weights
    w = apply_vol_target(w, returns, cfg)# adjusts those weights to hit the vol target
    r = portfolio_returns(w, returns, cost_drag(w, cfg))
    ew = equal_weight_returns(returns)
    test = cfg.split.test_start

    full = metrics(r, ew)
    oos  = metrics(r[r.index >= test], ew[ew.index >= test])

    return name, {
        "ann_ret":    full["ann_return"],
        "sharpe":     full["sharpe"],
        "maxDD":      full["max_drawdown"],
        "IR_full":    full.get("info_ratio", float("nan")),
        "sharpe_oos": oos["sharpe"],
        "IR_oos":     oos.get("info_ratio", float("nan")),
    }


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
    regime  = regime_features(macro_wide(macro), get_vix_series(market))

    # ── build all factors once ────────────────────────────────────────────────
    mom      = momentum_score(prices, cfg.m1.momentum_windows)
    trd      = trend_score(prices, cfg.m1.trend_windows)
    rel_m    = relative_momentum_score(prices, window=12)
    hi_prox  = high_proximity_score(prices, window=52)
    rev      = reversal_score(prices)

    m1 = M1Model(cfg.m1)

    # ── four M1 variants ──────────────────────────────────────────────────────
    #
    # Variant 1: Momentum-only
    #   Pure cross-sectional momentum signal. No trend, no extra factors.
    #   This is the simplest possible M1 — the baseline the advisor asked for.
    score_mom_only = m1.score({"technical": mom})

    # Variant 2: Trend-only
    #   Pure moving-average trend signal. No momentum, no extra factors.
    #   Useful to see if trend alone adds value vs momentum alone.
    score_trd_only = m1.score({"technical": trd})

    # Variant 3: Technical combined (momentum + trend, equal weight)
    #   The original M1 from last week. Blends momentum and trend 50/50.
    #   Benchmark to check whether adding extra factors actually helps.
    score_tech = m1.score({"technical": 0.5 * mom + 0.5 * trd})

    # Variant 4: Full 4-factor M1 (this week's update)
    #   Adds relative momentum, 52-week high proximity, and short-term reversal
    #   on top of the technical signal. All factors are static and price-based.
    score_4f = m1.score({
        "technical":      0.40 * (0.5 * mom + 0.5 * trd),
        "rel_momentum":   0.30 * rel_m,
        "high_proximity": 0.20 * hi_prox,
        "reversal":       0.10 * rev,
    })

    # ── run all variants ──────────────────────────────────────────────────────
    ew      = equal_weight_returns(returns)
    test    = cfg.split.test_start
    ew_full = metrics(ew)
    ew_oos  = metrics(ew[ew.index >= test])

    rows = {}

    # Equal-weight benchmark
    rows["Equal-Weight (benchmark)"] = {
        "ann_ret":    ew_full["ann_return"],
        "sharpe":     ew_full["sharpe"],
        "maxDD":      ew_full["max_drawdown"],
        "IR_full":    float("nan"),
        "sharpe_oos": ew_oos["sharpe"],
        "IR_oos":     float("nan"),
    }

    for name, score in [
        ("M1: Momentum-Only",          score_mom_only),
        ("M1: Trend-Only",             score_trd_only),
        ("M1: Technical (mom+trend)",  score_tech),
        ("M1: 4-Factor (this week)",   score_4f),
    ]:
        _, row = run_variant(name, score, cfg, returns)
        rows[name] = row

    # ── print results ─────────────────────────────────────────────────────────
    table = pd.DataFrame(rows).T
    print(f"\n{'='*80}")
    print("  M1 VARIANT COMPARISON")
    print(f"  Full sample {prices.index[0].date()} – {prices.index[-1].date()}")
    print(f"  Out-of-sample (OOS) from {test}")
    print(f"{'='*80}\n")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

    
if __name__ == "__main__":
    main()
