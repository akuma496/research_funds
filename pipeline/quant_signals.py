"""Quant signals: price tiers, drift vs alpha vs mismatch, four screens,
and Monte-Carlo price targets.

Definitions (plain English in the dashboard tooltips):
- tier: price bucket (sub-$20 / $80 / $100 / $150 / $200 / above).
- drift_ann: annualized average daily log return over 6 months — the price's
  current "cruising direction" regardless of the market.
- mr_z ("mismatch"): how many standard deviations price sits away from its own
  50-day average — the statistical stretch a mean-reversion trade feeds on.
- half_life: how fast the stretch historically decays (days), from an AR(1) fit.
- Screens are RELATIVE likelihood rankings (percentiles), not calibrated
  probabilities: squeeze (short interest x ignition), mean-reversal (stretch x
  reversion speed), sideways (low vol x no drift x tight range), black swan
  (jump exposure x fat tails). Top 25 of each flagged.
- Price targets: 2,000 Merton jump-diffusion paths per ticker to 2026-09-30 and
  2026-12-31 -> 10th/50th/90th percentile prices, beside the analyst mean.

Writes data/store/{quant_signals,price_targets}.parquet
Run:   python pipeline/quant_signals.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
TARGET_DATES = ["2026-09-30", "2026-12-31"]
TOP_N = 25


def zs(s: pd.Series) -> pd.Series:
    return ((s - s.mean()) / s.std()).clip(-3, 3).fillna(0.0)


def tier(p: float) -> str:
    for cut, name in [(20, "sub-$20"), (80, "sub-$80"), (100, "sub-$100"),
                      (150, "sub-$150"), (200, "sub-$200")]:
        if p < cut:
            return name
    return "above-$200"


def main():
    db = pd.read_parquet(STORE / "bars_daily.parquet")
    df = pd.read_parquet(STORE / "daily_features.parquet")
    mp = pd.read_parquet(STORE / "merton_params.parquet")
    fund_f = STORE / "fundamentals.parquet"
    fund = pd.read_parquet(fund_f) if fund_f.exists() else pd.DataFrame({"ticker": []})

    px = db.pivot(index="date", columns="ticker", values="c").sort_index()
    rows = []
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 60:
            continue
        r = np.log(s / s.shift(1)).dropna()
        m50, sd50 = s.rolling(50).mean().iloc[-1], s.rolling(50).std().iloc[-1]
        mr_z = (s.iloc[-1] - m50) / sd50 if sd50 else np.nan
        # AR(1) on deviation from 50d mean -> reversion half-life
        dev = (s - s.rolling(50).mean()).dropna()
        hl = np.nan
        if len(dev) > 60:
            x, y = dev.shift(1).dropna(), dev.diff().dropna()
            x, y = x.align(y, join="inner")
            if len(x) > 30 and x.var() > 0:
                phi = float(np.cov(x, y)[0, 1] / x.var())
                if -1 < phi < 0:
                    hl = float(-np.log(2) / np.log(1 + phi))
        rr = r.tail(126)
        rows.append({
            "ticker": t, "price": float(s.iloc[-1]), "tier": tier(float(s.iloc[-1])),
            "drift_ann": float(rr.mean() * 252),
            "mr_z": float(mr_z) if pd.notna(mr_z) else np.nan,
            "mismatch_pct": float(s.iloc[-1] / m50 - 1) if m50 else np.nan,
            "half_life_d": hl,
            "range20_pct": float((s.tail(20).max() - s.tail(20).min())
                                 / s.tail(20).mean()),
            "kurtosis": float(rr.kurtosis()),
        })
    q = pd.DataFrame(rows)
    q = (q.merge(df[["ticker", "alpha_ann", "beta", "vol_ann", "rsi14", "mom_1w",
                     "dollar_vol_21d", "data_artifact"]], on="ticker", how="left")
          .merge(mp[["ticker", "jump_var_share", "sigma_j", "jumps_per_year"]],
                 on="ticker", how="left")
          .merge(fund, on="ticker", how="left", suffixes=("", "_f")))
    q = q[~q["data_artifact"].fillna(False)]

    # ---- four screens (percentile ranks 0-100; top-25 flags) ----
    q["squeeze_score"] = (2 * zs(q.get("short_pct_float", pd.Series(dtype=float)))
                          + zs(q.get("days_to_cover", pd.Series(dtype=float)))
                          + zs(q["mom_1w"]))          # crowded short + ignition
    q.loc[q.get("short_pct_float").isna(), "squeeze_score"] = np.nan
    q["meanrev_score"] = (zs(q["mr_z"].abs())          # stretched...
                          + zs(1 / q["half_life_d"].clip(lower=2))  # ...and snaps back fast
                          + zs((q["rsi14"] - 50).abs()))
    q["meanrev_direction"] = np.where(q["mr_z"] < 0, "reverts UP", "reverts DOWN")
    q["sideways_score"] = (zs(-q["vol_ann"]) + zs(-q["drift_ann"].abs())
                           + zs(-q["range20_pct"]))
    q["blackswan_score"] = (zs(q["jump_var_share"]) + zs(q["sigma_j"])
                            + zs(q["kurtosis"]) + zs(q["vol_ann"]))
    for sc in ["squeeze", "meanrev", "sideways", "blackswan"]:
        col = f"{sc}_score"
        q[f"{sc}_pctl"] = q[col].rank(pct=True).mul(100).round(0)
        thresh = q[col].nlargest(TOP_N).min()
        q[f"{sc}_top25"] = q[col] >= thresh
    q.to_parquet(STORE / "quant_signals.parquet", index=False)
    print(f"quant_signals: {len(q)} tickers "
          f"(squeeze data for {q['squeeze_score'].notna().sum()})")

    # ---- Monte-Carlo price targets ----
    rng = np.random.default_rng(11)
    last_date = pd.Timestamp(db["date"].max())
    tgt_rows = []
    mpx = mp.set_index("ticker")
    for _, r in q.iterrows():
        t = r["ticker"]
        if t not in mpx.index:
            continue
        m = mpx.loc[t]
        # blend empirical drift (capped) with zero — pure historical drift
        # extrapolation overshoots badly on hot names
        mu_d = float(np.clip(r["drift_ann"] / 252, -0.002, 0.002)) * 0.5
        for dstr in TARGET_DATES:
            n = int(np.busday_count(last_date.date(), pd.Timestamp(dstr).date()))
            if n <= 0:
                continue
            sig, lam, mj, sj = m["sigma_d"], m["lambda_d"], m["mu_j"], m["sigma_j"]
            diff = rng.normal(mu_d - sig ** 2 / 2, sig, (2000, n)).sum(axis=1)
            nj = rng.poisson(lam * n, 2000)
            jumps = rng.normal(mj * nj, sj * np.sqrt(np.maximum(nj, 1)) * (nj > 0))
            fin = r["price"] * np.exp(diff + jumps)
            p10, p50, p90 = np.percentile(fin, [10, 50, 90])
            tgt_rows.append({"ticker": t, "target_date": dstr,
                             "p10": p10, "p50": p50, "p90": p90,
                             "analyst_mean": r.get("target_mean")})
    tg = pd.DataFrame(tgt_rows)
    tg.to_parquet(STORE / "price_targets.parquet", index=False)
    print(f"price_targets: {len(tg)} ticker-dates")


if __name__ == "__main__":
    main()
