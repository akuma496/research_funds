"""Paper portfolio: the scoreboard's honest forward test.

Each time it runs on a new data Friday, it freezes the top-10 liquid names per
horizon as a paper basket (equal weight, assumed filled at that Friday's close,
no costs). Baskets are never edited after creation — future refreshes only add
new ones — so performance vs SPY is a true out-of-sample record of whether the
scoreboard means anything. Evaluation happens live in the dashboard.

Writes (append-only) data/store/paper_baskets.parquet
Run:   python pipeline/paper_portfolio.py   (also part of refresh_all)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
BASKETS = STORE / "paper_baskets.parquet"
HORIZONS = ["1d", "2w", "1m", "6m", "12m"]
TOP_N = 10


def main():
    sb = pd.read_parquet(STORE / "scoreboard.parquet")
    wm = pd.read_parquet(STORE / "window_metrics.parquet", columns=["date"])
    db = pd.read_parquet(STORE / "bars_daily.parquet", columns=["ticker", "date", "c"])
    as_of = wm["date"].max()

    closes = db[db.date == as_of].set_index("ticker")["c"]
    if closes.empty or "SPY" not in closes:
        print(f"paper_portfolio: no daily closes for {as_of} yet — skipped")
        return

    existing = pd.read_parquet(BASKETS) if BASKETS.exists() else pd.DataFrame()
    rows = []
    pool = sb[sb.get("liquid", True)]
    for h in HORIZONS:
        if not existing.empty and ((existing.as_of == as_of)
                                   & (existing.horizon == h)).any():
            continue
        top = pool.sort_values(f"score_{h}", ascending=False).head(TOP_N)
        for _, r in top.iterrows():
            px = closes.get(r.ticker)
            if pd.isna(px):
                continue
            rows.append({"as_of": as_of, "horizon": h, "ticker": r.ticker,
                         "entry_price": float(px), "score": float(r[f"score_{h}"]),
                         "spy_entry": float(closes["SPY"]),
                         "created_at": datetime.now().astimezone().isoformat()})
    if not rows:
        print(f"paper_portfolio: baskets for {as_of} already exist — nothing to add")
        return
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out.to_parquet(BASKETS, index=False)
    print(f"paper_portfolio: froze {len(rows)} holdings across "
          f"{len(set(r['horizon'] for r in rows))} horizons as of {as_of} "
          f"(total record: {out.groupby(['as_of', 'horizon']).ngroups} baskets)")


if __name__ == "__main__":
    main()
