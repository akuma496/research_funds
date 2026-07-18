"""Equity history backfill from Alpaca (free tier, stdlib only).

Fetches and stores raw:
  1. 1-minute bars (all sessions incl. pre/post market) for every ticker in
     data/universe.csv, for the last 52 Fridays
       -> data/raw/equity/minute/<YYYY-MM-DD>.json.gz   (one file per Friday)
  2. Daily bars for 2 years for the whole universe
       -> data/raw/equity/daily/daily_bars.json.gz

Safe to re-run: already-downloaded Friday files are skipped, so an interrupted
run just resumes. Usage:

    python collector/equity_backfill.py            # full backfill
    python collector/equity_backfill.py --fridays 4    # most recent N Fridays only
"""

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe.csv"
OUT_MIN = ROOT / "data" / "raw" / "equity" / "minute"
OUT_DAY = ROOT / "data" / "raw" / "equity" / "daily"
LOG_FILE = ROOT / "data" / "raw" / "equity" / "backfill.log"

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
MIN_REQUEST_INTERVAL = 0.35   # ~170 req/min, under the 200/min free cap
MAX_RETRIES = 4
PAGE_LIMIT = 10000


def load_env() -> dict:
    env = {}
    env_file = ROOT / ".env"
    if not env_file.exists():
        sys.exit("FATAL: .env not found at repo root")
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") if not env.get(k)]
    if missing:
        sys.exit(f"FATAL: missing in .env: {', '.join(missing)}")
    return env


def load_universe() -> list:
    with open(UNIVERSE, newline="") as f:
        return [r["ticker"].strip() for r in csv.DictReader(f) if r["ticker"].strip()]


_last_request = [0.0]

def api_get(url: str, headers: dict):
    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(MAX_RETRIES):
        _last_request[0] = time.monotonic()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def fetch_bars(symbols, timeframe, start, end, headers) -> dict:
    """Fetch bars for many symbols over a range, following pagination.
    Returns {symbol: [bars...]}."""
    merged, token, pages = {}, None, 0
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "start": start, "end": end, "limit": str(PAGE_LIMIT),
              "adjustment": "split", "feed": "sip", "sort": "asc"}
    while True:
        if token:
            params["page_token"] = token
        data = api_get(BARS_URL + "?" + urllib.parse.urlencode(params), headers)
        for sym, bars in (data.get("bars") or {}).items():
            merged.setdefault(sym, []).extend(bars)
        pages += 1
        token = data.get("next_page_token")
        if not token or pages >= 500:
            break
    return merged


def last_fridays(n: int, include_today: bool = False) -> list:
    d = date.today()
    d -= timedelta(days=(d.weekday() - 4) % 7)  # most recent Friday (or today)
    if d >= date.today() and not include_today:  # exclude today/future
        d -= timedelta(days=7)
    return [d - timedelta(weeks=i) for i in range(n)]


def log(msg: str):
    line = f"{datetime.now().astimezone().isoformat()} {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fridays", type=int, default=52)
    ap.add_argument("--include-today", action="store_true",
                    help="include today's session (use after the close)")
    args = ap.parse_args()

    env = load_env()
    headers = {"APCA-API-KEY-ID": env["ALPACA_API_KEY"],
               "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"],
               "Accept": "application/json"}
    symbols = load_universe()
    OUT_MIN.mkdir(parents=True, exist_ok=True)
    OUT_DAY.mkdir(parents=True, exist_ok=True)

    # --- 1. minute bars for the last N Fridays ---
    fridays = last_fridays(args.fridays, include_today=args.include_today)
    log(f"START backfill: {len(fridays)} Fridays x {len(symbols)} tickers")
    done = failed = 0
    for d in fridays:
        out = OUT_MIN / f"{d.isoformat()}.json.gz"
        if out.exists():
            done += 1
            continue
        try:
            bars = fetch_bars(symbols, "1Min", f"{d.isoformat()}T08:00:00Z",
                              f"{d.isoformat()}T23:59:00Z", headers)
            n = sum(len(v) for v in bars.values())
            with gzip.open(out, "wt") as f:
                json.dump({"date": d.isoformat(), "timeframe": "1Min",
                           "n_bars": n, "bars": bars}, f)
            done += 1
            log(f"  {d} minute bars: {len(bars)} symbols, {n} bars")
        except Exception as e:
            failed += 1
            log(f"  FAIL {d}: {e}")
    log(f"minute backfill done: ok={done} failed={failed}")

    # --- 2. daily bars, 2 years ---
    out = OUT_DAY / "daily_bars.json.gz"
    start = (date.today() - timedelta(days=731)).isoformat()
    end = (date.today() if args.include_today
           else date.today() - timedelta(days=1)).isoformat()
    try:
        bars = fetch_bars(symbols, "1Day", f"{start}T00:00:00Z", f"{end}T23:59:00Z", headers)
        n = sum(len(v) for v in bars.values())
        with gzip.open(out, "wt") as f:
            json.dump({"start": start, "end": end, "timeframe": "1Day",
                       "n_bars": n, "bars": bars}, f)
        log(f"daily bars: {len(bars)} symbols, {n} bars -> {out.name}")
    except Exception as e:
        log(f"FAIL daily bars: {e}")
        sys.exit(1)
    log("BACKFILL COMPLETE")


if __name__ == "__main__":
    main()
