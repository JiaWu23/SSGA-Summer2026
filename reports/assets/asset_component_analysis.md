# Asset & Component Analysis

Standalone buy-and-hold performance for each index sleeve in the universe, plus documentation of all data inputs.

**Research use only — not investment advice.**

## Data & Components Used

The pipeline combines **seven index sleeves** (asset-class benchmarks) plus **macro/risk indicators** for regime features. Prices are resampled to **weekly** (Friday close) from daily adjusted-close data.

| Field | Value |
| --- | --- |
| Sample start | 2011-06-03 |
| Sample end | 2026-07-10 |
| Frequency | Weekly (W-FRI) |
| Price field | Adjusted close |

### Index Sleeves (tradable universe)

| Ticker | Instrument | Proxy / Benchmark | Asset Class | Role in Portfolio | Data Source |
| --- | --- | --- | --- | --- | --- |
| SP500 | S&P 500 Index | S&P 500 Index (^GSPC) | Equity | U.S. large-cap equity index sleeve | Yahoo Finance |
| MSCI_EAFE | MSCI EAFE Index Proxy | MSCI EAFE public proxy (EFA) | Equity | developed international equity index sleeve | Yahoo Finance |
| MSCI_EM | MSCI Emerging Markets Index Proxy | MSCI Emerging Markets public proxy (EEM) | Equity | emerging markets equity index sleeve | Yahoo Finance |
| UST_7_10 | U.S. Treasury 7-10 Year Index Proxy | U.S. Treasury 7-10 Year public proxy (IEF) | Fixed Income | intermediate Treasury duration sleeve | Yahoo Finance |
| US_HIGH_YIELD | U.S. High Yield Bond Index Proxy | U.S. High Yield public proxy (HYG) | Fixed Income | credit risk and high-yield fixed income sleeve | Yahoo Finance |
| GOLD_SPOT | Gold Spot / Gold Index Proxy | Gold futures / spot proxy (GC=F) | Commodity | commodity and inflation-hedging sleeve | Yahoo Finance |
| US_REIT | U.S. REIT Index | NASDAQ U.S. Benchmark REIT Index | Real Estate | real estate equity index sleeve | FRED: NASDAQNQUSB351020 |

### Macro & Risk Indicators (features only)

These series are **not traded** in the backtest. They feed M1/M2 regime and false-positive features, lagged by 4 weeks to approximate publication delay.

| Series | Description | Use | Source |
| --- | --- | --- | --- |
| CPIAUCSL | Consumer Price Index | Inflation trend and regime indicator | FRED — lagged 4 weeks in features |
| UNRATE | Unemployment Rate | Labor market / growth proxy | FRED — lagged 4 weeks in features |
| INDPRO | Industrial Production Index | Economic growth proxy | FRED — lagged 4 weeks in features |
| FEDFUNDS | Federal Funds Rate | Monetary policy stance | FRED — lagged 4 weeks in features |
| DGS10 | 10-Year Treasury Yield | Long-term interest rate level | FRED — lagged 4 weeks in features |
| T10Y2Y | 10Y–2Y Treasury Spread | Yield curve slope / recession signal | FRED — lagged 4 weeks in features |
| BAA10Y | Baa–10Y Credit Spread | Credit stress indicator | FRED — lagged 4 weeks in features |
| VIX | CBOE Volatility Index | Equity risk sentiment (risk-on / risk-off) | yfinance (^VIX) — used in features, not traded |

## Individual Asset Performance (Buy-and-Hold)

Each row below is a **standalone buy-and-hold** of one index sleeve: 100% allocated to that asset, rebalanced weekly, **no transaction costs**, no M1/M2 overlay. This shows how each building block performed on its own before any strategy logic. Charts also overlay **M1** and **M1+M2** portfolio models (long-only and long/short) for comparison.

### Full Sample

| Ticker | Asset | Class | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Total Return | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SP500 | S&P 500 Index | Equity | 12.2423% | 16.2364% | 0.7540 | -31.8103% | 475.5222% | 58.5025% |
| MSCI_EAFE | MSCI EAFE Index Proxy | Equity | 6.7988% | 17.1525% | 0.3964 | -33.4150% | 170.9527% | 55.8376% |
| MSCI_EM | MSCI Emerging Markets Index Proxy | Equity | 4.4273% | 19.6649% | 0.2251 | -39.1578% | 92.7985% | 53.8071% |
| UST_7_10 | U.S. Treasury 7-10 Year Index Proxy | Fixed Income | 1.9271% | 6.3495% | 0.3035 | -23.2555% | 33.5420% | 54.4416% |
| US_HIGH_YIELD | U.S. High Yield Bond Index Proxy | Fixed Income | 4.7054% | 9.0782% | 0.5183 | -20.7473% | 100.7274% | 58.1218% |
| GOLD_SPOT | Gold Spot / Gold Index Proxy | Commodity | 6.6472% | 16.2561% | 0.4089 | -43.6303% | 165.1813% | 54.9492% |
| US_REIT | U.S. REIT Index | Real Estate | 3.6860% | 20.1949% | 0.1825 | -39.9657% | 73.0690% | 53.8071% |

![Individual asset cumulative returns](asset_cumulative_returns.png)

![Individual asset metrics](asset_metrics_bars.png)

### Train Period (2006-01-01 to 2020-12-31)

| Ticker | Asset | Class | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Total Return | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SP500 | S&P 500 Index | Equity | 11.5243% | 16.4455% | 0.7008 | -31.8103% | 184.8157% | 59.7194% |
| MSCI_EAFE | MSCI EAFE Index Proxy | Equity | 5.0103% | 17.7039% | 0.2830 | -33.4150% | 59.8619% | 55.1102% |
| MSCI_EM | MSCI Emerging Markets Index Proxy | Equity | 2.7042% | 20.3061% | 0.1332 | -36.6879% | 29.1823% | 52.5050% |
| UST_7_10 | U.S. Treasury 7-10 Year Index Proxy | Fixed Income | 4.1628% | 5.8635% | 0.7100 | -8.5046% | 47.9020% | 57.1142% |
| US_HIGH_YIELD | U.S. High Yield Bond Index Proxy | Fixed Income | 5.2681% | 10.1607% | 0.5185 | -20.7473% | 63.6672% | 60.5210% |
| GOLD_SPOT | Gold Spot / Gold Index Proxy | Commodity | 2.0883% | 16.1884% | 0.1290 | -43.6303% | 21.9368% | 52.9058% |
| US_REIT | U.S. REIT Index | Real Estate | 3.7112% | 21.2509% | 0.1746 | -39.9657% | 41.8622% | 55.9118% |

### Test Period (2021-01-01 to latest)

| Ticker | Asset | Class | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Total Return | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SP500 | S&P 500 Index | Equity | 13.4929% | 15.8962% | 0.8488 | -24.8230% | 102.0683% | 56.4014% |
| MSCI_EAFE | MSCI EAFE Index Proxy | Equity | 9.9590% | 16.1793% | 0.6155 | -28.8835% | 69.4917% | 57.0934% |
| MSCI_EM | MSCI Emerging Markets Index Proxy | Equity | 7.4707% | 18.5344% | 0.4031 | -39.1578% | 49.2453% | 56.0554% |
| UST_7_10 | U.S. Treasury 7-10 Year Index Proxy | Fixed Income | -1.8209% | 7.0924% | -0.2567 | -21.7531% | -9.7091% | 49.8270% |
| US_HIGH_YIELD | U.S. High Yield Bond Index Proxy | Fixed Income | 3.7409% | 6.8301% | 0.5477 | -15.3952% | 22.6436% | 53.9792% |
| GOLD_SPOT | Gold Spot / Gold Index Proxy | Commodity | 15.0032% | 16.3472% | 0.9178 | -22.0208% | 117.4743% | 58.4775% |
| US_REIT | U.S. REIT Index | Real Estate | 3.6424% | 18.2635% | 0.1994 | -37.3946% | 21.9980% | 50.1730% |

![Train vs test asset returns](asset_train_test_returns.png)

### Per-Asset Highlights

- **SP500** (S&P 500 Index (^GSPC)): 12.2423% annualized, Sharpe 0.7540, max drawdown -31.8103% — U.S. large-cap equity index sleeve.
- **MSCI_EAFE** (MSCI EAFE public proxy (EFA)): 6.7988% annualized, Sharpe 0.3964, max drawdown -33.4150% — developed international equity index sleeve.
- **GOLD_SPOT** (Gold futures / spot proxy (GC=F)): 6.6472% annualized, Sharpe 0.4089, max drawdown -43.6303% — commodity and inflation-hedging sleeve.
- **US_HIGH_YIELD** (U.S. High Yield public proxy (HYG)): 4.7054% annualized, Sharpe 0.5183, max drawdown -20.7473% — credit risk and high-yield fixed income sleeve.
- **MSCI_EM** (MSCI Emerging Markets public proxy (EEM)): 4.4273% annualized, Sharpe 0.2251, max drawdown -39.1578% — emerging markets equity index sleeve.
- **US_REIT** (NASDAQ U.S. Benchmark REIT Index): 3.6860% annualized, Sharpe 0.1825, max drawdown -39.9657% — real estate equity index sleeve.
- **UST_7_10** (U.S. Treasury 7-10 Year public proxy (IEF)): 1.9271% annualized, Sharpe 0.3035, max drawdown -23.2555% — intermediate Treasury duration sleeve.

## Notes

- **SP500** represents the U.S. large-cap equity index sleeve.
- **UST_7_10** and **US_HIGH_YIELD** represent the fixed-income sleeves, separating Treasury duration from credit risk.
- **MSCI_EAFE** and **MSCI_EM** split developed international and emerging market equity exposure.
- **GOLD_SPOT** provides commodity/gold exposure; **US_REIT** provides listed real estate exposure.
- Strategy results in the main report combine these components via M1 signals and M2 sizing.
- Data provenance, ETL, validation, cache behavior, and fallback caveats are documented in `../../DATA_SOURCES_AND_ETL.md`.
