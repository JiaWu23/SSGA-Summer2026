"""Market and macro data providers."""

from __future__ import annotations

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class MarketDataProvider(ABC):
    @abstractmethod
    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        ...


class MacroDataProvider(ABC):
    @abstractmethod
    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        ...


class BloombergProvider(MarketDataProvider, MacroDataProvider):
    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "BloombergProvider is a placeholder. Export data from Bloomberg Terminal "
            "or implement the Bloomberg API wrapper and save to data/raw/."
        )

    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "BloombergProvider is a placeholder. Implement Bloomberg macro fetch or use FredProvider."
        )


class IndexProvider(MarketDataProvider):
    """
    Index-first market data provider.

    Pipeline asset ids are index sleeve names such as SP500, MSCI_EAFE,
    UST_7_10, GOLD_SPOT, etc. The Yahoo/FRED symbols are only used for
    public data retrieval. The final ticker column keeps the index asset id.
    """

    def __init__(self, index_dir: Path, index_sources: dict[str, dict[str, str]]) -> None:
        self.index_dir = index_dir
        self.index_sources = index_sources
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_download(
        self,
        asset_id: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        source = self.index_sources.get(asset_id, {})
        provider = source.get("provider", "manual")
        symbol = source.get("symbol", asset_id)

        safe_name = asset_id.replace("^", "")
        cache_path = self.index_dir / f"{safe_name}.csv"

        if cache_path.exists():
            df = pd.read_csv(cache_path)
            return df

        if provider == "manual":
            raise FileNotFoundError(
                f"Missing manual index file for {asset_id}: {cache_path}"
            )

        if provider == "fred":
            raw = _fetch_fred_csv(symbol)
            date_col = raw.columns[0]
            value_col = raw.columns[1]
            df = raw.rename(
                columns={
                    date_col: "date",
                    value_col: "adj_close",
                }
            )

        elif provider == "yahoo":
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )

            if raw.empty:
                raise ValueError(f"No Yahoo data returned for {asset_id} / {symbol}")

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            df = raw[[price_col]].reset_index()
            df.columns = ["date", "adj_close"]

        else:
            raise ValueError(f"Unknown provider for {asset_id}: {provider}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        df = df.dropna(subset=["date", "adj_close"])
        df = df[df["date"] >= pd.Timestamp(start)]

        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]

        df[["date", "adj_close"]].to_csv(cache_path, index=False)
        logger.info("Saved index/proxy data for %s to %s", asset_id, cache_path)

        return df

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for asset_id in tickers:
            logger.info("Loading index/proxy series for %s", asset_id)

            df = self._load_or_download(asset_id, start=start, end=end)

            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
            df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
            df = df.dropna(subset=["date", "adj_close"])

            df = df[df["date"] >= pd.Timestamp(start)]
            if end is not None:
                df = df[df["date"] <= pd.Timestamp(end)]

            df["ticker"] = asset_id
            df["open"] = df["adj_close"]
            df["high"] = df["adj_close"]
            df["low"] = df["adj_close"]
            df["close"] = df["adj_close"]
            df["volume"] = 0

            frames.append(df[["date", "ticker", *PRICE_COLUMNS]])

        if not frames:
            raise ValueError("No index market data was loaded.")

        daily = pd.concat(frames, ignore_index=True)
        daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

        if frequency == "daily":
            return daily

        return resample_to_weekly(daily)

def _fetch_fred_csv(series_id: str, retries: int = 3, pause: float = 1.0, cache_path: Path | None = None) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-fsSL", "--max-time", "30", url],
                capture_output=True,
                text=True,
                check=True,
            )
            df = pd.read_csv(StringIO(result.stdout))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
            return df
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            last_err = exc
            logger.warning("FRED curl fetch %s attempt %d failed: %s", series_id, attempt + 1, exc)
            time.sleep(pause)
    try:
        resp = requests.get(url, headers={"User-Agent": "finance-meta-labeling-pipeline/0.1"}, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        return df
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Failed to fetch FRED series {series_id}") from exc or last_err


def build_proxy_macro(market_weekly: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Fallback macro panel derived from market data when FRED is unavailable."""
    dates = pd.to_datetime(market_weekly["date"].unique())
    rows = []
    vix = market_weekly[market_weekly["ticker"].isin(["VIX", "^VIX"])]
    vix_series = vix.set_index("date")["adj_close"] if not vix.empty else pd.Series(20.0, index=dates)
    for d in dates:
        for sid in series:
            vix_val = float(vix_series.get(d, 20.0))
            val = vix_val if sid == "BAA10Y" else vix_val / 100.0
            rows.append({"date": d, "series": sid, "value": val})
    return pd.DataFrame(rows)


class FredProvider(MacroDataProvider):
    """Fetch macro series from FRED public CSV endpoint (no API key required)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir

    def get_macro(
        self,
        series: list[str],
        start: str,
        end: str | None = None,
        frequency: str = "weekly",
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for sid in series:
            try:
                logger.info("Downloading FRED series %s", sid)
                cache_path = (self.cache_dir / f"fred_{sid}.csv") if self.cache_dir else None
                s = _fetch_fred_csv(sid, cache_path=cache_path)
                s = s.rename(columns={"observation_date": "date", sid: "value"})
                if "DATE" in s.columns:
                    s = s.rename(columns={"DATE": "date"})
                s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
                s["value"] = pd.to_numeric(s["value"], errors="coerce")
                s = s[s["date"] >= pd.Timestamp(start)]
                if end is not None:
                    s = s[s["date"] <= pd.Timestamp(end)]
                s["series"] = sid
                frames.append(s[["date", "series", "value"]])
            except Exception as exc:
                logger.warning("Skipping FRED series %s: %s", sid, exc)
        if not frames:
            raise RuntimeError("No FRED series could be downloaded")

        daily = pd.concat(frames, ignore_index=True)
        daily = daily.sort_values(["series", "date"]).reset_index(drop=True)
        if frequency == "daily":
            return daily
        return resample_macro_to_weekly(daily)


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV panel to weekly (Friday or last trading day)."""
    parts: list[pd.DataFrame] = []
    for ticker, grp in daily.groupby("ticker"):
        g = grp.set_index("date").sort_index()
        weekly = pd.DataFrame(
            {
                "open": g["open"].resample("W-FRI").first(),
                "high": g["high"].resample("W-FRI").max(),
                "low": g["low"].resample("W-FRI").min(),
                "close": g["close"].resample("W-FRI").last(),
                "adj_close": g["adj_close"].resample("W-FRI").last(),
                "volume": g["volume"].resample("W-FRI").sum(),
            }
        )
        weekly = weekly.dropna(subset=["adj_close"])
        weekly = weekly.reset_index().rename(columns={"date": "date"})
        weekly["ticker"] = ticker
        parts.append(weekly)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def resample_macro_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Map each macro observation to its week-ending Friday; keep sparse weeks.

    Gaps between monthly/irregular releases are filled later in feature engineering
    via ``src.time_alignment`` (train: interpolate; test/eval: forward-fill) onto
    the market weekly calendar before the publication lag is applied.
    """
    parts: list[pd.DataFrame] = []
    for series, grp in daily.groupby("series"):
        g = grp.set_index("date").sort_index()
        weekly = g["value"].resample("W-FRI").last().dropna().reset_index()
        weekly["series"] = series
        weekly = weekly.rename(columns={"value": "value"})
        parts.append(weekly)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "series"]).reset_index(drop=True)


def ingest_market_data(
    tickers: list[str],
    vix_ticker: str,
    start: str,
    end: str | None,
    raw_dir: Path,
    processed_dir: Path,
    provider: MarketDataProvider | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_weekly = processed_dir / "market_weekly.parquet"
    if use_cache and cache_weekly.exists():
        logger.warning("Using CACHED market data from %s (not re-downloading)", cache_weekly)
        return pd.read_parquet(cache_weekly)

    all_tickers = list(dict.fromkeys(tickers + [vix_ticker]))
    daily_provider = provider or YFinanceProvider()
    daily = daily_provider.get_prices(all_tickers, start=start, end=end, frequency="daily")
    weekly = resample_to_weekly(daily)
    daily.to_parquet(raw_dir / "market_daily.parquet", index=False)
    weekly.to_parquet(cache_weekly, index=False)
    logger.info("Saved market data: %d weekly rows", len(weekly))
    return weekly


def ingest_macro_data(
    series: list[str],
    start: str,
    end: str | None,
    raw_dir: Path,
    processed_dir: Path,
    provider: MacroDataProvider | None = None,
    *,
    market_weekly: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_daily = raw_dir / "macro_daily.parquet"
    cache_weekly = processed_dir / "macro_weekly.parquet"

    if use_cache and cache_weekly.exists():
        logger.warning("Using CACHED macro data from %s (not re-downloading)", cache_weekly)
        return pd.read_parquet(cache_weekly)

    try:
        provider = provider or FredProvider(cache_dir=raw_dir)
        daily = provider.get_macro(series, start=start, end=end, frequency="daily")
        missing = [sid for sid in series if sid not in set(daily["series"])]
        if missing:
            if cache_weekly.exists():
                cached = pd.read_parquet(cache_weekly)
                cached_missing = cached[cached["series"].isin(missing)].copy()
                if not cached_missing.empty:
                    logger.warning(
                        "FRED refresh missed %s; preserving cached rows for missing series",
                        ", ".join(sorted(cached_missing["series"].unique())),
                    )
                    daily = pd.concat([daily, cached_missing], ignore_index=True)
                    missing = [sid for sid in missing if sid not in set(cached_missing["series"])]
            if missing and market_weekly is not None:
                logger.warning(
                    "FRED refresh missed %s; filling missing series with market-derived proxies",
                    ", ".join(missing),
                )
                daily = pd.concat([daily, build_proxy_macro(market_weekly, missing)], ignore_index=True)
                missing = []
            if missing and cache_weekly.exists():
                logger.warning(
                    "FRED refresh incomplete (%s missing); using existing macro cache instead",
                    ", ".join(missing),
                )
                return pd.read_parquet(cache_weekly)
        weekly = resample_macro_to_weekly(daily)
        daily.to_parquet(cache_daily, index=False)
        weekly.to_parquet(cache_weekly, index=False)
        logger.info("Saved macro data: %d weekly rows", len(weekly))
        return weekly
    except Exception as exc:
        if cache_weekly.exists():
            logger.warning("Macro fetch failed (%s); using cached %s", exc, cache_weekly)
            return pd.read_parquet(cache_weekly)
        if market_weekly is not None:
            logger.warning("Macro fetch failed (%s); using market-derived proxy macro", exc)
            daily = build_proxy_macro(market_weekly, series)
            weekly = resample_macro_to_weekly(daily)
            daily.to_parquet(cache_daily, index=False)
            weekly.to_parquet(cache_weekly, index=False)
            return weekly
        raise
