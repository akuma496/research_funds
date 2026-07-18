"""Per-ticker fundamentals: cash reserves (SEC XBRL), short interest, analyst
targets, and tracked-ETF top-10 membership (yfinance).

Writes data/store/fundamentals.parquet
Run:   python pipeline/fundamentals.py     (~5-8 min, safe to re-run)
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
UA = {"User-Agent": "Aditya Kumar akuma496@asu.edu (personal research dashboard)"}
CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
TRACKED_ETFS = ["SPY", "QQQ", "IWM", "SMH", "ICLN", "URA", "ITA", "IHI",
                "BOTZ", "XLI", "XLV", "XLE", "LAZR"]

_last = [0.0]

def sec_get(url):
    wait = 0.13 - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read().decode())


def sec_cash(uni: pd.DataFrame) -> pd.DataFrame:
    sec = sec_get("https://www.sec.gov/files/company_tickers.json")
    ciks = {v["ticker"]: int(v["cik_str"]) for v in sec.values()}
    rows = []
    for t in uni["ticker"]:
        cik = ciks.get(t)
        if not cik:
            continue
        for tag in CASH_TAGS:
            try:
                d = sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/"
                            f"CIK{cik:010d}/us-gaap/{tag}.json")
            except Exception:
                continue
            facts = [(e["end"], e["val"]) for e in d.get("units", {}).get("USD", [])]
            if facts:
                end, val = max(facts)  # latest reported instant
                rows.append({"ticker": t, "cash": float(val), "cash_date": end})
                break
    return pd.DataFrame(rows)


def yf_stats(uni: pd.DataFrame) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for t in uni["ticker"]:
        try:
            info = yf.Ticker(t).info or {}
            rows.append({
                "ticker": t,
                "short_pct_float": info.get("shortPercentOfFloat"),
                "shares_short": info.get("sharesShort"),
                "days_to_cover": info.get("shortRatio"),
                "float_shares": info.get("floatShares"),
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "n_analysts": info.get("numberOfAnalystOpinions"),
            })
        except Exception:
            rows.append({"ticker": t})
        time.sleep(0.25)
    return pd.DataFrame(rows)


def etf_membership(uni: pd.DataFrame) -> pd.DataFrame:
    """Which of our tracked ETFs list a stock among their TOP-10 holdings.
    (Free data only exposes top holdings — a floor, not a full count.)"""
    import yfinance as yf
    ours = set(uni["ticker"])
    membership = {}
    for etf in TRACKED_ETFS:
        try:
            th = yf.Ticker(etf).funds_data.top_holdings
            for sym in th.index.astype(str):
                if sym in ours:
                    membership.setdefault(sym, []).append(etf)
        except Exception as e:
            print(f"  WARN {etf}: top holdings unavailable ({e})")
        time.sleep(0.25)
    return pd.DataFrame([{"ticker": t, "etf_top10_count": len(v),
                          "etf_top10_list": ",".join(v)}
                         for t, v in membership.items()])


def main():
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    print("=== SEC cash reserves ===")
    cash = sec_cash(uni)
    print(f"  cash for {len(cash)} companies")
    print("=== yfinance short interest / analyst targets ===")
    ys = yf_stats(uni)
    print(f"  stats for {ys['short_pct_float'].notna().sum()} tickers with short data")
    print("=== tracked-ETF top-10 membership ===")
    em = etf_membership(uni)
    print(f"  {len(em)} of our tickers appear in tracked ETFs' top-10 holdings")

    out = uni[["ticker"]].merge(cash, on="ticker", how="left") \
                         .merge(ys, on="ticker", how="left") \
                         .merge(em, on="ticker", how="left")
    out["etf_top10_count"] = out["etf_top10_count"].fillna(0).astype(int)
    out.to_parquet(STORE / "fundamentals.parquet", index=False)
    print(f"fundamentals: {len(out)} rows written")


if __name__ == "__main__":
    main()
