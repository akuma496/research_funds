"""Options chain snapshot collector.

Captures the full options chain (quotes, trades, IV, greeks) for every ticker in
data/universe.csv from Alpaca's free indicative feed, and stores the raw API
responses as gzipped JSON under data/raw/options/<date>/<window>/<TICKER>.json.gz.

Chains cannot be backfilled, so history for the options layer of the dashboard
accrues from whenever this script starts running. Scheduled 4x per trading day
(see setup notes in README_collector.md); can also be run manually:

    python collector/options_snapshot.py            # window inferred from clock
    python collector/options_snapshot.py --window W3
    python collector/options_snapshot.py --test     # 3 tickers only, prints summary

Stdlib only — no packages to install.
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
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe.csv"
OUT_ROOT = ROOT / "data" / "raw" / "options"
LOG_FILE = OUT_ROOT / "collector.log"

BASE = "https://data.alpaca.markets/v1beta1/options/snapshots/"
EXPIRY_HORIZON_DAYS = 45      # nearest expiries are what matter for OPEX flow
PAGE_LIMIT = 1000
MIN_REQUEST_INTERVAL = 0.35   # stay under Alpaca's 200 req/min free limit
MAX_RETRIES = 4

# Window boundaries in local (Central) time — matches SPEC.md section 2.
# W1 pre-open baseline, W2 midday, W3 power hour, W4 post-close.
def infer_window(now: datetime) -> str:
    hm = now.hour * 60 + now.minute
    if hm < 9 * 60:
        return "W1"
    if hm < 12 * 60 + 30:
        return "W2"
    if hm < 15 * 60:
        return "W3"
    return "W4"


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
        return [row["ticker"].strip() for row in csv.DictReader(f) if row["ticker"].strip()]


_last_request = [0.0]

def api_get(url: str, headers: dict):
    """GET with rate limiting and backoff. Returns parsed JSON or raises."""
    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(MAX_RETRIES):
        _last_request[0] = time.monotonic()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
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


def fetch_chain(sym: str, headers: dict) -> dict:
    """Fetch full chain snapshot for one underlying, following pagination."""
    max_expiry = (datetime.now() + timedelta(days=EXPIRY_HORIZON_DAYS)).strftime("%Y-%m-%d")
    params = {"feed": "indicative", "limit": str(PAGE_LIMIT), "expiration_date_lte": max_expiry}
    snapshots, pages, token = {}, 0, None
    while True:
        if token:
            params["page_token"] = token
        data = api_get(BASE + sym + "?" + urllib.parse.urlencode(params), headers)
        snapshots.update(data.get("snapshots") or {})
        pages += 1
        token = data.get("next_page_token")
        if not token or pages >= 20:
            break
    return {"underlying": sym, "fetched_at": datetime.now().astimezone().isoformat(),
            "pages": pages, "n_contracts": len(snapshots), "snapshots": snapshots}


def log(msg: str):
    line = f"{datetime.now().astimezone().isoformat()} {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["W1", "W2", "W3", "W4"])
    ap.add_argument("--test", action="store_true", help="fetch 3 tickers only")
    args = ap.parse_args()

    env = load_env()
    headers = {"APCA-API-KEY-ID": env["ALPACA_API_KEY"],
               "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"],
               "Accept": "application/json"}

    now = datetime.now()
    window = args.window or infer_window(now)
    out_dir = OUT_ROOT / now.strftime("%Y-%m-%d") / window
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = load_universe()
    if args.test:
        tickers = ["SPY", "NVDA", "JOBY"]

    log(f"START window={window} tickers={len(tickers)} out={out_dir}")
    ok = empty = failed = 0
    for sym in tickers:
        try:
            chain = fetch_chain(sym, headers)
            if chain["n_contracts"] == 0:
                empty += 1  # no listed options (common for micro-caps) — still record it
            else:
                ok += 1
            with gzip.open(out_dir / f"{sym}.json.gz", "wt") as f:
                json.dump(chain, f)
            if args.test:
                log(f"  {sym}: {chain['n_contracts']} contracts in {chain['pages']} page(s)")
        except Exception as e:
            failed += 1
            log(f"  FAIL {sym}: {e}")
    log(f"DONE window={window} ok={ok} empty={empty} failed={failed}")
    if failed and failed >= len(tickers) // 2:
        sys.exit(1)  # mostly failed — surface a nonzero exit for Task Scheduler history


if __name__ == "__main__":
    main()
