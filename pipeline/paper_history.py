"""Backfill historical paper baskets for every Tuesday and Friday of the past year.

True point-in-time discipline: each historical basket is picked using ONLY
price-derived signals computable on that date (momentum blend minus volatility,
liquidity gate) — sentiment/13F/options can't be reconstructed historically, so
those baskets are labeled basket_type="proxy" to distinguish them from the full
"live" baskets frozen going forward. Together they form a walk-forward record.

Labels: "Fri 2026-06-19 · expiration", "Tue 2026-06-16", etc.

Appends to data/store/paper_baskets.parquet (idempotent).
Run:   python pipeline/paper_history.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
BASKETS = STORE / "paper_baskets.parquet"
TOP_N = 10
LOOKBACK_WEEKS = 52


def zs(row: pd.Series) -> pd.Series:
    return (row - row.mean()) / row.std()


def label_for(d: pd.Timestamp) -> str:
    if d.weekday() == 4:
        third = 15 + (4 - pd.Timestamp(d.year, d.month, 15).weekday()) % 7
        tag = " · expiration" if d.day == third else ""
        return f"Fri {d.date()}{tag}"
    return f"{d.strftime('%a')} {d.date()}"


def main():
    db = pd.read_parquet(STORE / "bars_daily.parquet")
    px = db.pivot(index="date", columns="ticker", values="c").sort_index()
    px.index = pd.to_datetime(px.index)
    dv = (db.pivot(index="date", columns="ticker", values="v").sort_index()
            .set_axis(px.index) * px)  # dollar volume

    mom21, mom63 = px / px.shift(21) - 1, px / px.shift(63) - 1
    vol63 = np.log(px / px.shift(1)).rolling(63).std()
    dv21 = dv.rolling(21).mean()

    days = px.index[px.index >= px.index.max() - pd.Timedelta(weeks=LOOKBACK_WEEKS)]
    days = [d for d in days if d.weekday() in (1, 4)]  # Tuesdays & Fridays

    existing = pd.read_parquet(BASKETS) if BASKETS.exists() else pd.DataFrame()
    if not existing.empty and "basket_type" not in existing.columns:
        existing["basket_type"] = "live"
        existing["label"] = existing["as_of"].map(
            lambda s: label_for(pd.Timestamp(s)))
    have = (set(zip(existing.as_of, existing.horizon))
            if not existing.empty else set())

    rows = []
    for d in days:
        key = (d.strftime("%Y-%m-%d"), "proxy")
        if key in have or pd.isna(px.loc[d, "SPY"]):
            continue
        score = (zs(mom21.loc[d]) + zs(mom63.loc[d]) - zs(vol63.loc[d]))
        liquid = dv21.loc[d] >= 1_000_000
        score = score[liquid & score.notna()]
        top = score.nlargest(TOP_N)
        if len(top) < TOP_N:
            continue
        for t, sc in top.items():
            rows.append({"as_of": d.strftime("%Y-%m-%d"), "horizon": "proxy",
                         "ticker": t, "entry_price": float(px.loc[d, t]),
                         "score": float(sc), "spy_entry": float(px.loc[d, "SPY"]),
                         "created_at": "backfill",
                         "basket_type": "proxy", "label": label_for(d)})
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out.to_parquet(BASKETS, index=False)
    n_new = len(rows) // TOP_N if rows else 0
    print(f"paper_history: added {n_new} proxy baskets "
          f"(total {out.groupby(['as_of', 'horizon']).ngroups} baskets on record)")


if __name__ == "__main__":
    main()
