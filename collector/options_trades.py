"""Options trade-tape collector: every print for contracts that traded on a date.

Contract discovery uses the day's chain snapshots (which contracts show a trade
that day); trades then come from Alpaca's historical options trades endpoint,
which is free. Runs after the close (scheduled 15:25 Central).

    python collector/options_trades.py                    # today
    python collector/options_trades.py --date 2026-07-17
    python collector/options_trades.py --backfill-days 5  # past weekdays too,
                                                          # using the latest
                                                          # snapshot's contracts

Output: data/raw/options_trades/<date>.json.gz  {contract: [trades...]}
Stdlib only.
"""

import argparse
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
SNAP_ROOT = ROOT / "data" / "raw" / "options"
OUT_ROOT = ROOT / "data" / "raw" / "options_trades"
LOG_FILE = OUT_ROOT / "trades.log"

TRADES_URL = "https://data.alpaca.markets/v1beta1/options/trades"
BATCH = 90
MIN_REQUEST_INTERVAL = 0.35
MAX_RETRIES = 4


def load_env() -> dict:
    env = {}
    env_file = ROOT / ".env"
    if not env_file.exists():
        sys.exit("FATAL: .env not found")
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    if not env.get("ALPACA_API_KEY") or not env.get("ALPACA_SECRET_KEY"):
        sys.exit("FATAL: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
    return env


_last = [0.0]

def api_get(url, headers):
    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(MAX_RETRIES):
        _last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60) as r:
                return json.loads(r.read().decode())
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


def log(msg):
    line = f"{datetime.now().astimezone().isoformat()} {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def contracts_for(day: str) -> tuple:
    """Contract symbols that traded on `day`, from that day's snapshots.
    Falls back to the latest snapshot dir (all contracts) when the day has none."""
    snap_dir = SNAP_ROOT / day
    fallback = False
    if not snap_dir.exists() or not any(snap_dir.glob("*/*.json.gz")):
        dirs = sorted(d for d in SNAP_ROOT.iterdir() if d.is_dir())
        if not dirs:
            sys.exit("No chain snapshots found at all — run options_snapshot.py first.")
        snap_dir, fallback = dirs[-1], True
    syms = set()
    for f in snap_dir.glob("*/*.json.gz"):
        d = json.load(gzip.open(f, "rt"))
        for occ, s in (d.get("snapshots") or {}).items():
            t = s.get("latestTrade") or {}
            if fallback or str(t.get("t", ""))[:10] == day:
                syms.add(occ)
    return sorted(syms), fallback


def fetch_day(day: str, headers) -> None:
    out = OUT_ROOT / f"{day}.json.gz"
    if out.exists():
        log(f"skip {day}: already collected")
        return
    syms, fallback = contracts_for(day)
    if fallback:
        log(f"{day}: no same-day snapshots — using latest snapshot's full contract "
            f"list ({len(syms)} contracts); trades before their listing date are naturally absent")
    log(f"{day}: fetching trades for {len(syms)} contracts")
    all_trades, n_prints = {}, 0
    for i in range(0, len(syms), BATCH):
        chunk = syms[i:i + BATCH]
        params = {"symbols": ",".join(chunk), "start": f"{day}T08:00:00Z",
                  "end": f"{day}T23:59:59Z", "limit": "10000", "sort": "asc"}
        token = None
        while True:
            if token:
                params["page_token"] = token
            data = api_get(TRADES_URL + "?" + urllib.parse.urlencode(params), headers)
            for sym, trades in (data.get("trades") or {}).items():
                all_trades.setdefault(sym, []).extend(trades)
                n_prints += len(trades)
            token = data.get("next_page_token")
            if not token:
                break
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump({"date": day, "n_contracts": len(all_trades),
                   "n_prints": n_prints, "trades": all_trades}, f)
    log(f"{day}: DONE {len(all_trades)} contracts, {n_prints:,} prints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--backfill-days", type=int, default=0)
    args = ap.parse_args()

    env = load_env()
    headers = {"APCA-API-KEY-ID": env["ALPACA_API_KEY"],
               "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"]}

    days = [args.date]
    d = date.fromisoformat(args.date)
    for _ in range(args.backfill_days):
        d -= timedelta(days=1)
        while d.weekday() > 4:
            d -= timedelta(days=1)
        days.append(d.isoformat())
    for day in days:
        try:
            fetch_day(day, headers)
        except Exception as e:
            log(f"FAIL {day}: {e}")


if __name__ == "__main__":
    main()
