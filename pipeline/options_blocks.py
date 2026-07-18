"""Classify the options trade tape into buyer/seller-initiated prints and blocks.

Side classification, best-effort on free data (per SPEC):
  1. Quote rule against the day's chain-snapshot quotes (last captured window):
     price >= ask*0.995 -> buy ("at ask"), price <= bid*1.005 -> sell.
  2. Tick rule fallback for prints between the quotes: uptick -> buy,
     downtick -> sell, unchanged -> carry previous sign.

Block = >=50 contracts in one print OR >=$25,000 premium.

Reads  data/raw/options_trades/*.json.gz + data/store/options_chain.parquet
Writes data/store/options_tape.parquet  (prints with size >= 10, all blocks)

Run:  python pipeline/options_blocks.py
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "options_trades"
STORE = ROOT / "data" / "store"

OCC_RE = re.compile(r"^([A-Z.]+)(\d{6})([CP])(\d{8})$")
MIN_KEEP_SIZE = 10
BLOCK_SIZE = 50
BLOCK_PREMIUM = 25_000


def load_tape() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        day = d["date"]
        for occ, trades in d["trades"].items():
            m = OCC_RE.match(occ)
            if not m:
                continue
            und, exp, cp, strike = (m.group(1), m.group(2), m.group(3),
                                    int(m.group(4)) / 1000.0)
            prev_p, sign = None, 0
            for t in trades:
                p, s = t.get("p"), t.get("s", 0)
                if p is None:
                    continue
                if prev_p is not None:
                    if p > prev_p:
                        sign = 1
                    elif p < prev_p:
                        sign = -1
                prev_p = p
                if s >= MIN_KEEP_SIZE:
                    rows.append((day, occ, und, cp, strike,
                                 "20" + exp[:2] + "-" + exp[2:4] + "-" + exp[4:6],
                                 t.get("t"), p, s, p * s * 100, sign,
                                 t.get("x"), t.get("c")))
    cols = ["date", "occ", "underlying", "cp", "strike", "expiry", "ts",
            "price", "size", "premium", "tick_sign", "exchange", "cond"]
    return pd.DataFrame(rows, columns=cols)


def main():
    tape = load_tape()
    if tape.empty:
        print("no trade-tape raw files yet — run collector/options_trades.py first")
        return

    # quote rule via the day's last captured chain snapshot
    oc = pd.read_parquet(STORE / "options_chain.parquet",
                         columns=["date", "window", "occ", "bid", "ask"])
    oc = (oc.sort_values("window").groupby(["date", "occ"]).last().reset_index())
    tape = tape.merge(oc, on=["date", "occ"], how="left")

    at_ask = (tape["ask"] > 0) & (tape["price"] >= tape["ask"] * 0.995)
    at_bid = (tape["bid"] > 0) & (tape["price"] <= tape["bid"] * 1.005)
    tape["side"] = np.select(
        [at_ask, at_bid, tape["tick_sign"] > 0, tape["tick_sign"] < 0],
        ["buy (at ask)", "sell (at bid)", "buy (tick)", "sell (tick)"],
        default="unclear")
    tape["is_block"] = (tape["size"] >= BLOCK_SIZE) | (tape["premium"] >= BLOCK_PREMIUM)

    tape.to_parquet(STORE / "options_tape.parquet", index=False)
    blocks = tape[tape.is_block]
    print(f"options_tape: {len(tape):,} prints (size>={MIN_KEEP_SIZE}) across "
          f"{tape['date'].nunique()} day(s) | blocks: {len(blocks):,} "
          f"({blocks['premium'].sum() / 1e6:,.1f}M premium) | "
          f"side known: {(tape['side'] != 'unclear').mean():.0%}")


if __name__ == "__main__":
    main()
