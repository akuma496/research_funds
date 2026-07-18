"""Parse raw collector output into the Parquet store.

Inputs  (written by collector/*.py):
    data/raw/equity/minute/<date>.json.gz     one file per Friday
    data/raw/equity/daily/daily_bars.json.gz
    data/raw/options/<date>/<window>/<sym>.json.gz

Outputs (data/store/):
    bars_minute.parquet   ticker, ts (UTC), o h l c, v, n, vw, date
    bars_daily.parquet    ticker, date, o h l c, v, vw
    options_chain.parquet one row per contract per captured window snapshot
    universe.parquet      ticker, industry

Re-runnable: rebuilds the store from all raw files each time (raw is source of truth).
"""

import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
STORE = ROOT / "data" / "store"

OCC_RE = re.compile(r"^([A-Z.]+)(\d{6})([CP])(\d{8})$")


def load_overrides() -> dict:
    """ticker -> valid_from date string. Bars before valid_from are dropped —
    used when a ticker is recycled to a different security (e.g. LAZR)."""
    f = ROOT / "data" / "ticker_overrides.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return dict(zip(df["ticker"], df["valid_from"]))


def apply_overrides(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    for t, valid_from in load_overrides().items():
        drop = (df["ticker"] == t) & (df[date_col].astype(str) < valid_from)
        if drop.any():
            print(f"  overrides: dropped {drop.sum():,} {t} rows before {valid_from}")
            df = df[~drop]
    return df


def parse_minute_bars() -> pd.DataFrame:
    frames = []
    for f in sorted((RAW / "equity" / "minute").glob("*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        rows = []
        for sym, bars in d["bars"].items():
            for b in bars:
                rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("n", 0), b.get("vw", 0.0)))
        if rows:
            df = pd.DataFrame(rows, columns=["ticker", "ts", "o", "h", "l", "c", "v", "n", "vw"])
            df["date"] = d["date"]
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def parse_daily_bars() -> pd.DataFrame:
    d = json.load(gzip.open(RAW / "equity" / "daily" / "daily_bars.json.gz", "rt"))
    rows = []
    for sym, bars in d["bars"].items():
        for b in bars:
            rows.append((sym, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw", 0.0)))
    return pd.DataFrame(rows, columns=["ticker", "date", "o", "h", "l", "c", "v", "vw"])


def parse_options() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "options").glob("*/*/*.json.gz")):
        window, date = f.parent.name, f.parent.parent.name
        d = json.load(gzip.open(f, "rt"))
        oi_map = d.get("contracts") or {}
        for occ, s in (d.get("snapshots") or {}).items():
            m = OCC_RE.match(occ)
            if not m:
                continue
            und, exp, cp, strike = m.group(1), m.group(2), m.group(3), int(m.group(4)) / 1000.0
            q = s.get("latestQuote") or {}
            t = s.get("latestTrade") or {}
            g = s.get("greeks") or {}
            rows.append((date, window, und, occ, "20" + exp[:2] + "-" + exp[2:4] + "-" + exp[4:6],
                         cp, strike,
                         q.get("bp"), q.get("ap"), q.get("bs"), q.get("as"), q.get("t"),
                         t.get("p"), t.get("s"), t.get("t"),
                         s.get("impliedVolatility"), g.get("delta"), g.get("gamma"),
                         g.get("theta"), g.get("vega"), g.get("rho"),
                         float(oi_map.get(occ, {}).get("oi") or 0) or None,
                         d.get("fetched_at")))
    cols = ["date", "window", "underlying", "occ", "expiry", "cp", "strike",
            "bid", "ask", "bid_size", "ask_size", "quote_ts",
            "last_price", "last_size", "trade_ts",
            "iv_feed", "delta_feed", "gamma_feed", "theta_feed", "vega_feed", "rho_feed",
            "open_interest", "fetched_at"]
    return pd.DataFrame(rows, columns=cols)


def main():
    STORE.mkdir(parents=True, exist_ok=True)

    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    uni.to_parquet(STORE / "universe.parquet", index=False)

    mb = apply_overrides(parse_minute_bars(), "date")
    mb.to_parquet(STORE / "bars_minute.parquet", index=False)
    print(f"bars_minute: {len(mb):,} rows, {mb['date'].nunique()} Fridays, {mb['ticker'].nunique()} tickers")

    db = apply_overrides(parse_daily_bars(), "date")
    db.to_parquet(STORE / "bars_daily.parquet", index=False)
    print(f"bars_daily:  {len(db):,} rows, {db['ticker'].nunique()} tickers")

    oc = parse_options()
    oc.to_parquet(STORE / "options_chain.parquet", index=False)
    print(f"options_chain: {len(oc):,} rows, "
          f"{oc.groupby(['date','window']).ngroups if len(oc) else 0} snapshots")


if __name__ == "__main__":
    main()
