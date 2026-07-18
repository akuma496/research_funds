"""SEC EDGAR integration: 13F institutional panel, Form 4 insiders, lipstick revenue.

Three independent parts (each fails soft so one outage doesn't kill the rest):

1. 13F panel — the latest two quarterly 13F-HR filings from ~13 giant
   institutions, holdings matched to our universe by issuer name. Gives
   per-ticker "panel shares held" for the last two quarters. Honest limits:
   it's a panel, not all institutions, and 13F data lags ~45 days.
2. Form 4 insiders — open-market insider buys (code P) vs sells (code S)
   per ticker over the last 90 days.
3. Lipstick revenue — quarterly revenue for the beauty basket from XBRL.

Writes data/store/{inst_13f,insiders,lipstick_revenue}.parquet
Run:   python pipeline/sec_refresh.py            (all three, ~10-20 min)
       python pipeline/sec_refresh.py --part 13f|insiders|revenue
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
UA = {"User-Agent": "Aditya Kumar akuma496@asu.edu (personal research dashboard)"}
RATE = 0.13  # stay under SEC's 10 req/s

FILERS = {  # validated at runtime; wrong/renamed CIKs are skipped with a warning
    "Vanguard Group": 102909,
    "BlackRock": 1364742,
    "State Street": 93751,
    "FMR (Fidelity)": 315066,
    "Geode Capital": 1214717,
    "Berkshire Hathaway": 1067983,
    "Citadel Advisors": 1423053,
    "Millennium Management": 1273087,
    "Renaissance Technologies": 1037389,
    "Bridgewater Associates": 1350694,
    "Point72": 1603466,
    "ARK Investment": 1697748,
    "Norges Bank": 1374170,
}

REVENUE_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"]

_last = [0.0]

def get(url, binary=False):
    wait = RATE - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=120) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 2:
                time.sleep(3)
                continue
            raise


def universe() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "universe.csv")


def cik_map() -> dict:
    sec = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    return {v["ticker"]: int(v["cik_str"]) for v in sec.values()}


SUFFIXES = {"INC", "CORP", "CO", "LTD", "PLC", "SA", "NV", "LP", "LLC", "CL",
            "COM", "NEW", "DEL", "DE", "NY", "PA", "TX", "UK", "THE", "HOLDINGS",
            "HOLDING", "GRP", "COMPANIES", "ENTERPRISES", "TECHNOLOGIES"}

def norm_tokens(name: str) -> list:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    return [t for t in s.split() if t and t not in SUFFIXES]


def build_name_index(uni: pd.DataFrame) -> dict:
    """normalized-name keys -> ticker, for matching 13F issuer names."""
    idx = {}
    ambiguous = set()
    for _, r in uni.iterrows():
        toks = norm_tokens(r.get("name", r["ticker"]))
        if not toks:
            continue
        for key in (" ".join(toks), " ".join(toks[:2]), toks[0]):
            if len(key) < 4 and key != toks[0]:
                continue
            if key in idx and idx[key] != r["ticker"]:
                ambiguous.add(key)
            else:
                idx[key] = r["ticker"]
    for k in ambiguous:
        idx.pop(k, None)
    return idx


def match_issuer(issuer: str, idx: dict) -> str | None:
    toks = norm_tokens(issuer)
    if not toks:
        return None
    for key in (" ".join(toks), " ".join(toks[:2]), toks[0] if len(toks[0]) >= 5 else None):
        if key and key in idx:
            return idx[key]
    return None


# ------------------------------------------------------------------ 13F panel
def run_13f():
    uni = universe()
    idx = build_name_index(uni)
    rows = []
    for filer, cik in FILERS.items():
        try:
            sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
            rec = sub["filings"]["recent"]
            hits = [(rec["accessionNumber"][i], rec["reportDate"][i])
                    for i, f in enumerate(rec["form"]) if f.startswith("13F-HR")][:2]
            if not hits:
                print(f"  WARN {filer}: no 13F-HR found — skipping")
                continue
            for acc, report_date in hits:
                accn = acc.replace("-", "")
                base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}"
                index = json.loads(get(f"{base}/index.json"))
                # the information table is reliably the largest XML that isn't
                # the primary document (filers name it inconsistently)
                cands = [i for i in index["directory"]["item"]
                         if i["name"].lower().endswith(".xml")
                         and "primary_doc" not in i["name"].lower()]
                xml_name = (max(cands, key=lambda i: int(i.get("size") or 0))["name"]
                            if cands else None)
                if not xml_name:
                    print(f"  WARN {filer} {report_date}: no infotable xml")
                    continue
                blob = get(f"{base}/{xml_name}", binary=True)
                n_matched = 0
                for _, el in ET.iterparse(pd.io.common.BytesIO(blob)):
                    if el.tag.endswith("infoTable"):
                        issuer = shares = None
                        for c in el.iter():
                            if c.tag.endswith("nameOfIssuer"):
                                issuer = c.text
                            elif c.tag.endswith("sshPrnamt"):
                                shares = c.text
                        t = match_issuer(issuer or "", idx)
                        if t and shares:
                            rows.append((t, filer, report_date, float(shares)))
                            n_matched += 1
                        el.clear()
                print(f"  {filer} {report_date}: {n_matched} matched holdings")
        except Exception as e:
            print(f"  WARN {filer}: {e}")
    df = (pd.DataFrame(rows, columns=["ticker", "filer", "report_date", "shares"])
            .groupby(["ticker", "filer", "report_date"], as_index=False)["shares"].sum())
    df.to_parquet(STORE / "inst_13f.parquet", index=False)
    print(f"inst_13f: {len(df):,} filer-holdings, {df.ticker.nunique()} tickers, "
          f"{df.filer.nunique()} filers")


# ------------------------------------------------------------------ Form 4
def run_insiders():
    uni = universe()
    ciks = cik_map()
    since = (date.today() - timedelta(days=90)).isoformat()
    rows = []
    for _, r in uni.iterrows():
        t = r["ticker"]
        cik = ciks.get(t)
        if not cik:
            continue
        try:
            sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
            rec = sub["filings"]["recent"]
            f4 = [(rec["accessionNumber"][i], rec["filingDate"][i],
                   rec["primaryDocument"][i])
                  for i, f in enumerate(rec["form"])
                  if f == "4" and rec["filingDate"][i] >= since][:8]
            buy_sh = sell_sh = buy_val = sell_val = 0.0
            for acc, fdate, doc in f4:
                accn = acc.replace("-", "")
                xml = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}")
                try:
                    root = ET.fromstring(xml)
                except ET.ParseError:
                    continue
                for tr in root.iter("nonDerivativeTransaction"):
                    code = (tr.findtext(".//transactionCode") or "").strip()
                    sh = float(tr.findtext(".//transactionShares/value") or 0)
                    px = float(tr.findtext(".//transactionPricePerShare/value") or 0)
                    if code == "P":
                        buy_sh += sh
                        buy_val += sh * px
                    elif code == "S":
                        sell_sh += sh
                        sell_val += sh * px
            if f4:
                rows.append((t, len(f4), buy_sh, sell_sh, buy_val, sell_val,
                             buy_val - sell_val))
        except Exception as e:
            print(f"  WARN {t}: {e}")
    df = pd.DataFrame(rows, columns=["ticker", "n_form4_90d", "buy_shares",
                                     "sell_shares", "buy_value", "sell_value",
                                     "net_value"])
    df.to_parquet(STORE / "insiders.parquet", index=False)
    print(f"insiders: {len(df)} tickers with Form 4 activity in 90 days")


# ------------------------------------------------------------------ revenue
def run_revenue():
    uni = universe()
    lips = uni[uni.industry == "lipstick"]["ticker"].tolist()
    ciks = cik_map()
    rows = []
    for t in lips:
        cik = ciks.get(t)
        if not cik:
            continue
        for tag in REVENUE_TAGS:
            try:
                d = json.loads(get(f"https://data.sec.gov/api/xbrl/companyconcept/"
                                   f"CIK{cik:010d}/us-gaap/{tag}.json"))
            except Exception:
                continue
            got = {}
            for e in d.get("units", {}).get("USD", []):
                try:
                    dur = (date.fromisoformat(e["end"])
                           - date.fromisoformat(e["start"])).days
                except Exception:
                    continue
                if 80 <= dur <= 100:  # quarterly durations only
                    got[e["end"]] = e["val"]  # later filings overwrite (restatements)
            if got:
                rows += [(t, end, val) for end, val in got.items()]
                break
    df = (pd.DataFrame(rows, columns=["ticker", "period_end", "revenue"])
            .sort_values(["ticker", "period_end"]))
    df.to_parquet(STORE / "lipstick_revenue.parquet", index=False)
    print(f"lipstick_revenue: {len(df)} ticker-quarters for {df.ticker.nunique()} companies")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["13f", "insiders", "revenue"])
    args = ap.parse_args()
    STORE.mkdir(parents=True, exist_ok=True)
    if args.part in (None, "revenue"):
        print("=== lipstick revenue (XBRL) ===")
        run_revenue()
    if args.part in (None, "13f"):
        print("=== 13F panel ===")
        run_13f()
    if args.part in (None, "insiders"):
        print("=== Form 4 insiders ===")
        run_insiders()
    print("SEC REFRESH COMPLETE")


if __name__ == "__main__":
    main()
