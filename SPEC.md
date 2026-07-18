# OPEX Friday Market Microstructure Dashboard — Specification v1

**Owner:** Aditya Kumar · **Date:** 2026-07-16 · **Status:** awaiting sign-off (no code until approved)

## 1. What this is

A local, interactive dashboard that studies market behavior at four intraday windows on Fridays
(with special weight on monthly OPEX / witching Fridays), across 15 curated thematic industries
plus a "lipstick index" macro-regime panel, extended over a rolling 52-week history, with a
model-driven scoreboard per investment horizon.

**Audience:** a single non-technical but business-savvy user. Design goal: "wow" — clean,
interactive, self-explanatory, zero terminal knowledge needed after launch.

## 2. The four time windows (America/Chicago, CDT/CST aware)

| Window | CDT | ET | Market meaning |
|---|---|---|---|
| W1 Pre-open | 08:00–08:30 | 09:00–09:30 | Final pre-market half hour into the opening auction |
| W2 Midday | 10:15–10:45 | 11:15–11:45 | Midday flow inflection (~European close) |
| W3 Power hour close | 14:30–15:00 | 15:30–16:00 | Closing half hour + closing auction (MOC imbalances) |
| W4 After hours | 15:00–17:00 | 16:00–18:00 | Post-close reaction window |

Every metric in §5 is computed **per window, per ticker, per Friday**, then aggregated to industry level.
Fridays are tagged: `regular`, `monthly_opex` (3rd Friday), `quad_witching` (3rd Friday of Mar/Jun/Sep/Dec).

## 3. Data sources (all free tier)

| Source | Used for | Limits we accept |
|---|---|---|
| **Alpaca Market Data API** (free) | 1-min bars incl. pre/post market, historical to 2016 → makes the 52-week window lookback possible; trade-level prints for block detection; options chain + options trades (indicative feed — capability to be verified empirically on day 1) | IEX-only real-time feed; SIP historical OK; options feed is "indicative" quality |
| **yfinance** | Options chain snapshots (strikes, IV, OI, volume, bid/ask), fundamentals, shares outstanding, fallback prices | 1-min bars only ~30 days back; chain is snapshot-only (no history) — see §7 |
| **SEC EDGAR** (free, official) | 13F quarterly institutional holdings deltas; Form 4 insider trades; 8-K event feed; 10-Q revenue for lipstick index | 13F has 45-day lag, quarterly, no timestamps |
| **Web-search agents** (public pages only) | Sentiment & event research with citations (Yahoo Finance news, Finviz, Reuters, MarketWatch, press releases, public Seeking Alpha headlines) | No paywalled scraping; every claim cited with URL + date |

**Empirical feed test results (2026-07-16, live against Alpaca free tier):**
- Chain snapshots: ✅ full chains returned (SPY 5,458 contracts). Quotes on 100% of contracts; latest trade on ~77%; Alpaca-supplied greeks/IV only on recently-active contracts (~30%) → we compute our own Black–Scholes greeks from quotes, as planned.
- **Historical options trades: ✅ available free** — trade-by-trade prints (price, size, exchange, condition codes). This upgrades block detection from "best effort" to real tape analysis, and likely allows *backfilling* past block-trade history (verify depth at build time).
- Historical options quotes: ❌ 404 — not offered. At-ask classification therefore uses the **tick rule** on the trade sequence plus comparison against our four daily window snapshot quotes (the agreed best-effort method). OI/IV history still accrues only from capture start.

**Key honesty notes (agreed in discussion):**
- "Institutional buying/selling at those times" is approximated by: block prints (≥10,000 shares or ≥$200k notional) from trade tape + volume anomaly z-scores + quarterly 13F deltas. True intraday institutional attribution does not exist in public data.
- "Block options trades filled at the ask" is **best-effort on free data**: primary = Alpaca options trades classified by quote rule (fill ≥ ask ⇒ buyer-initiated / "at ask"); fallback = volume-vs-open-interest deltas between our four daily chain snapshots.
- Options **history starts the day we start capturing** (chains cannot be backfilled from yfinance). Equity minute-bar history CAN be backfilled 52 weeks via Alpaca.

## 4. The universe — 15 industries + lipstick panel (~205 tickers, top ≤15 each, US-listed incl. ADRs)

Lists are curated by liquidity/relevance; every ticker is validated against live data at build time
(delisted/acquired names dropped and logged). No ticker appears in two industries (clean rollups).

1. **Photonics:** LITE, COHR, IPGP, MKSI, FN, LASR, AAOI, POET, LPTH, KOPN, VIAV, CIEN, GLW, OLED, VECO
2. **Clean energy:** FSLR, ENPH, NEE, BE, PLUG, RUN, SEDG, ARRY, NXT, SHLS, CSIQ, JKS, BEP, AES, ORA
3. **Bitcoin mining:** MARA, RIOT, CLSK, CORZ, IREN, CIFR, HUT, WULF, BITF, HIVE, BTDR, BTBT, CAN, APLD, GLXY
4. **Semiconductors:** NVDA, TSM, AVGO, AMD, ASML, QCOM, TXN, AMAT, MU, LRCX, KLAC, ADI, MRVL, INTC, ARM
5. **Electronics (hardware/EMS/components):** AAPL, SONY, DELL, SMCI, HPE, HPQ, APH, TEL, JBL, FLEX, CLS, SANM, GRMN, LOGI, VSH
6. **Nuclear energy:** CEG, CCJ, VST, TLN, BWXT, OKLO, SMR, LEU, UEC, UUUU, NXE, DNN, NNE, LTBR, EU
7. **Battery storage:** FLNC, EOSE, STEM, ENS, ENVX, QS, SLDP, SES, AMPX, MVST, GWH, ALB, LAC, SQM
8. **Cooling (data-center/thermal):** VRT, MOD, NVT, JCI, CARR, TT, LII, AAON, WSO, FIX, LMB, CSWI, EMR
9. **Turbines (gas/wind/aero-derivative):** GEV, GE, HWM, CW, CAT, CMI, ETN, BW, PSIX, RRX, AGX
10. **AI vision (perception/imaging/lidar):** AMBA, CGNX, MBLY, TDY, OUST, LAZR, INVZ, AEVA, MVIS, HIMX, VUZI, REKR, EVLV
11. **Medical devices:** MDT, ABT, BSX, SYK, EW, ZBH, BDX, DXCM, PODD, RMD, GEHC, PEN, IRTC, TNDM, NVCR
12. **Medical robotics:** ISRG, PRCT, GMED, ARAY, STXS, RBOT, EDAP, MYO, TMCI, SISI-check → validated at build
13. **Miniature robotics & small drones:** SERV, RCAT, UMAC, ONDS, DPRO, MBOT, RR, PDYN, KSCP, AVAV-alt (thin universe — honestly ~8–10 investable names; panel says so)
14. **Defense:** LMT, RTX, NOC, GD, LHX, HII, TDG, HEI, LDOS, AXON, PLTR, KTOS, AVAV, RKLB, BA
15. **eVTOL / advanced air mobility:** JOBY, ACHR, EVTL, EH, EVEX, BLDE, SRFM, VTOL, HOVR, XTIA, AIRO (thin universe, ~11 names)

**Lipstick index panel (macro-regime, quarterly cadence):** EL, ELF, COTY, ULTA, ODD, IPAR, SBH, EPC —
quarterly share behavior + SEC 10-Q revenue trends, displayed as a consumer trade-down regime gauge.

**Benchmarks for alpha:** SPY (market), QQQ, IWM, plus matched ETFs per industry (SMH, SOXX, ICLN, URA/NLR, ITA, IHI, BOTZ, WGMI, XLE, XLI, XLV).

## 5. Metrics & models (all computed; descriptive AND predictive)

### Per-window core metrics (per ticker, per Friday, 52-week history)
- **Price/return:** window OHLC, window return, VWAP deviation, gap vs prior close
- **Volume:** raw, RVOL (vs 52-wk same-window average), volume z-score
- **Volatility:** realized vol (1-min log returns), Parkinson high-low estimator, window-vol percentile vs history
- **Institutional proxy:** block print count & notional (≥10k sh / ≥$200k), buy/sell classification via Lee–Ready tick rule, block imbalance ratio, Amihud illiquidity
- **13F overlay (quarterly):** # institutions, net share change QoQ, top holders delta (SEC)

### Options layer (from capture start; snapshots at each window)
- Chain snapshot per window: volume, OI, IV by strike/expiry
- **At-ask block detection:** puts & calls, quote-rule classification, premium size ranking
- **Greeks:** Black–Scholes delta, gamma, theta, vega, rho per contract from snapshot IV
- **Dealer gamma exposure (GEX)** aggregated per ticker and industry — headline metric on OPEX Fridays
- Put/call ratio, IV skew (25Δ), IV term structure

### Statistical models
- **Poisson:** arrival intensity λ of (a) block trades and (b) price jumps per window; over-dispersion test (negative binomial fallback); "expected blocks this window" vs actual
- **Merton jump-diffusion:** MLE calibration on daily log returns per ticker & industry (σ_diffusion, λ_jump, μ_J, σ_J); jump share of total variance; risk decomposition panel
- **Jump detection:** Lee–Mykland test on intraday returns → timestamped jump events
- **Hawkes (self-exciting) process** on jump events → P(≥k jump events in next 6 months) per industry via Monte Carlo, blended with a forward event calendar (Fed/FOMC dates, earnings dates, sector catalysts from cited research)
- **Alpha:** rolling CAPM alpha/beta vs SPY and matched sector ETF (daily, 63/252-day); **window micro-alpha** = window return − β × SPY window return
- **Momentum:** 1w / 1m / 3m / 6m / 12m−1m returns, RSI(14), MACD, distance to 52-wk high, industry-relative rank
- **Sentiment:** agent-gathered public headlines per industry & top tickers, LLM-scored −1…+1 with cited sources and dates; 4-week sentiment trend

### Scoreboard (per horizon: 1-day, 2-week, 1-month, 6-month, 12-month)
Composite z-score blend per horizon (weights differ: short horizons weight window flow/options/momentum;
long horizons weight Merton risk decomposition, alpha, 13F trend, sentiment, event-cluster probability).
Each row shows: score, rank, the evidence behind it (expandable), historical hit-rate of the score decile
(honest backtest), and a **$200 accessibility flag** (fractional vs whole-share).

> **Framing (agreed):** this is a statistical screen, not financial advice. The tool ranks and shows
> probabilities with confidence intervals; the investment decision is the user's. No "buy X" language.

## 6. Dashboard (local Streamlit app, dark professional theme, Plotly interactive)

Launch: one double-click `.bat` → opens in browser. Pages:

1. **Friday Command Center** — latest Friday, 4 windows × 15 industries heatmap (return / RVOL / vol / block imbalance toggle), OPEX badge, biggest movers & blocks
2. **Industry Explorer** — drill into one industry: 52-week window history, constituent table, sparklines
3. **Ticker Deep Dive** — everything for one symbol incl. minute chart with jump markers & block prints overlaid
4. **Options Flow & GEX** — at-ask blocks table (puts/calls, premium ranked), GEX by industry, IV skew
5. **Models Lab** — Merton decomposition, Poisson λ dashboards, Hawkes 6-month event-cluster probabilities
6. **Sentiment & Events** — cited headlines, sentiment gauges, next-week outlook Q&A (every claim sourced), forward event calendar
7. **Scoreboard** — the 5-horizon ranked screen with $200 accessibility and hit-rate honesty panel
8. **Lipstick Index** — quarterly cosmetics share behavior + SEC revenue trend vs SPX as regime gauge

UX: plain-English tooltips on every metric ("RVOL 3.2 = traded 3.2× its normal volume for this half hour"),
color-blind-safe palette, no jargon without a hover explainer.

## 7. Pipeline & storage

- Python 3.12, DuckDB + Parquet store (fast, zero-admin), `.env` for Alpaca keys (never committed)
- **Backfill job (one-time, ~hours):** 52 weeks × ~205 tickers of Friday minute bars + trades from Alpaca; daily bars 2 years; SEC 13F/Form 4 pulls
- **Snapshot collector (scheduled):** options chains + quotes captured at W1–W4 each trading day via Windows Task Scheduler (options history accrues from capture start)
- **Nightly update job:** append latest bars, recompute models, refresh scoreboard
- **Weekly agent job:** sentiment + event research with citations
- All model outputs cached to the store; dashboard reads the store only (instant load)

## 8. Build status (all delivered 2026-07-17)

1. ✅ Snapshot collector + open interest, scheduled 4×/weekday
2. ✅ Equity backfill (52 Fridays minute bars, 2y daily) + Parquet store
3. ✅ Core window metrics → pages 1–3
4. ✅ Options layer + Greeks/GEX + **trade tape & block classification** → page 4
5. ✅ Models (Poisson → Merton → Hawkes) → page 5
6. ✅ Sentiment/events agent with citations → page 6
7. ✅ Scoreboard + proxy backtest honesty panel → page 7
8. ✅ SEC: 13F institutional panel + Form 4 insiders → page 8 (Institutions & Insiders)
9. ✅ Lipstick panel incl. XBRL quarterly revenue → page 9
10. ✅ Fresh-machine bootstrap (`setup_fresh.py`), portable launchers,
    schedule registration script (`collector/register_schedule.ps1` — the
    trade-tape and nightly-refresh jobs are optional and not auto-registered)
