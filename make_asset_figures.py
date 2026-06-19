"""Generate ASSET-GROUP figures (PNG) into reports/figures/.

    python make_asset_figures.py

Groups the 7-asset universe into 3 asset classes and shows:
  asset_class_returns.png   - cumulative growth of each asset class (+ members)
  group_weights.png         - M1 active allocation across groups over time
  asset_active_tilt.png     - avg over/underweight vs 1/N benchmark, per asset
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import IndexFileProvider, ingest_market_data  # noqa: E402
from src.features import momentum_score, pivot_prices, trend_score  # noqa: E402
from src.m1 import M1Model  # noqa: E402
from src.portfolio import build_weights  # noqa: E402

logging.basicConfig(level=logging.ERROR)
NAVY, ACCENT, GREY = "#0B2A4A", "#1F6FB2", "#888888"

# 7-asset universe -> 3 asset classes
GROUPS = {
    "Equity": ["SPY", "VEA", "VWO"],
    "Fixed Income": ["TLT", "HYG"],
    "Alternatives": ["GLD", "VNQ"],
}
GROUP_COLOR = {"Equity": NAVY, "Fixed Income": ACCENT, "Alternatives": GREY}
LABEL = {"SPY": "SPY (US eq)", "VEA": "VEA (dev ex-US)", "VWO": "VWO (EM)",
         "TLT": "TLT (Treasuries)", "HYG": "HYG (high yield)",
         "GLD": "GLD (gold)", "VNQ": "VNQ (REITs)"}


def main() -> None:
    cfg = load_config()
    figdir = cfg.root / "reports" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    provider = IndexFileProvider(cfg.raw_dir / "index") if cfg.data.use_index_signal else None
    market = ingest_market_data(cfg.data.universe, cfg.vix_ticker, cfg.data.data_start, None,
                                cfg.raw_dir, cfg.processed_dir, provider=provider,
                                use_cache=not cfg.data.use_index_signal)
    prices = pivot_prices(market[market["ticker"].isin(cfg.data.universe)])
    returns = prices.pct_change()

    # M1 active weights (gross=1, fully invested) -- before vol targeting so
    # group shares sum to 100%.
    mom = momentum_score(prices, cfg.m1.momentum_windows)
    trd = trend_score(prices, cfg.m1.trend_windows)
    score = M1Model(cfg.m1).score({"technical": 0.5 * mom + 0.5 * trd})
    w = build_weights(score, cfg)

    # --- 1. asset-class cumulative returns ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for grp, members in GROUPS.items():
        comp = returns[members].mean(axis=1)               # equal-weight within class
        curve = (1 + comp.fillna(0)).cumprod()
        ax.plot(curve.index, curve.values, color=GROUP_COLOR[grp], linewidth=2.4, label=grp, zorder=3)
        for m in members:                                   # faded member lines
            c = (1 + returns[m].fillna(0)).cumprod()
            ax.plot(c.index, c.values, color=GROUP_COLOR[grp], linewidth=0.9, alpha=0.35, zorder=2)
    ax.set_yscale("log"); ax.set_title("Asset classes — cumulative growth (log scale)",
                                       color=NAVY, fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1"); ax.legend(frameon=False, title="bold = class composite")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figdir / "asset_class_returns.png", dpi=130); plt.close(fig)

    # --- 2. M1 group allocation over time (stacked area) ---
    gw = pd.DataFrame({grp: w[members].sum(axis=1) for grp, members in GROUPS.items()}).dropna()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stackplot(gw.index, [gw[g] for g in GROUPS], labels=list(GROUPS),
                 colors=[GROUP_COLOR[g] for g in GROUPS], alpha=0.85)
    # benchmark group weights (equal-weight 1/7 each)
    n = len(cfg.data.universe)
    bench = {g: len(m) / n for g, m in GROUPS.items()}
    cum = 0.0
    for g in GROUPS:
        cum += bench[g]
        ax.axhline(cum, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("M1 active allocation across asset classes (dashed = equal-weight benchmark)",
                 color=NAVY, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Portfolio weight"); ax.set_ylim(0, 1); ax.margins(x=0)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "group_weights.png", dpi=130); plt.close(fig)

    # --- 3. average active tilt per asset (vs 1/N) ---
    bench_w = 1.0 / n
    tilt = (w - bench_w).mean().reindex([a for g in GROUPS.values() for a in g])
    colors = [GROUP_COLOR[g] for g, members in GROUPS.items() for _ in members]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar([LABEL[a] for a in tilt.index], tilt.values * 100, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title("Average active weight vs equal-weight benchmark (M1)",
                 color=NAVY, fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg over / underweight (%)")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, tilt.values * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.12),
                f"{v:+.1f}", ha="center", color=NAVY, fontsize=9, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUPS]
    ax.legend(handles, list(GROUPS), frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(figdir / "asset_active_tilt.png", dpi=130); plt.close(fig)

    print("saved asset-group figures ->", figdir)


if __name__ == "__main__":
    main()
