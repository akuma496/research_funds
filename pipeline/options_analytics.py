"""Options analytics: Black-Scholes IV/greeks, GEX, at-ask classification.

Reads  data/store/{options_chain,bars_daily}.parquet
Writes data/store/options_enriched.parquet  per contract per snapshot
       data/store/options_agg.parquet       per underlying per snapshot

Greeks are computed here from snapshot quotes (the free feed only supplies its
own greeks for ~30% of contracts). Risk-free rate fixed at 4% — fine at these
horizons. GEX sign convention: dealers long calls / short puts, so call gamma
positive, put gamma negative; units = $ notional gamma per 1% move.

Run:  python pipeline/options_analytics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
RISK_FREE = 0.04


def bs_price(S, K, T, sig, r, cp):
    d1 = (np.log(S / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(S, K, T, mid, cp):
    intrinsic = max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    if mid <= intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(lambda v: bs_price(S, K, T, v, RISK_FREE, cp) - mid,
                      1e-3, 8.0, xtol=1e-5, maxiter=60)
    except Exception:
        return np.nan


def greeks(S, K, T, sig, cp):
    d1 = (np.log(S / K) + (RISK_FREE + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    delta = norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sig * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sig / (2 * np.sqrt(T))
             - (1 if cp == "C" else -1) * RISK_FREE * K * np.exp(-RISK_FREE * T)
             * norm.cdf(d2 if cp == "C" else -d2)) / 365
    return delta, gamma, vega, theta


def main():
    oc = pd.read_parquet(STORE / "options_chain.parquet")
    db = pd.read_parquet(STORE / "bars_daily.parquet")
    spot_last = db.sort_values("date").groupby("ticker")["c"].last()

    oc = oc.copy()
    oc["mid"] = np.where((oc.bid > 0) & (oc.ask > 0), (oc.bid + oc.ask) / 2, np.nan)
    oc["spot"] = oc["underlying"].map(spot_last)
    days = (pd.to_datetime(oc["expiry"]) - pd.to_datetime(oc["date"])).dt.days
    oc["T"] = np.maximum(days, 0.25) / 365.0
    oc["moneyness"] = oc["strike"] / oc["spot"]

    # IV: prefer feed value, invert from mid otherwise
    iv = oc["iv_feed"].copy()
    need = iv.isna() & oc["mid"].notna() & oc["spot"].notna() & (oc["mid"] > 0)
    iv.loc[need] = [implied_vol(s, k, t, m, c) for s, k, t, m, c in
                    zip(oc.loc[need, "spot"], oc.loc[need, "strike"],
                        oc.loc[need, "T"], oc.loc[need, "mid"], oc.loc[need, "cp"])]
    oc["iv"] = iv

    ok = oc["iv"].notna() & (oc["iv"] > 0) & oc["spot"].notna()
    g = np.array([greeks(s, k, t, v, c) if o else (np.nan,) * 4
                  for o, s, k, t, v, c in zip(ok, oc["spot"], oc["strike"],
                                              oc["T"], oc["iv"].fillna(1), oc["cp"])])
    oc[["delta", "gamma", "vega", "theta"]] = g

    sign = np.where(oc["cp"] == "C", 1.0, -1.0)
    oc["gex"] = (sign * oc["gamma"] * oc["open_interest"].fillna(0)
                 * 100 * oc["spot"] ** 2 * 0.01)

    # trade classification vs snapshot quote (same trading date only)
    traded_today = oc["trade_ts"].astype(str).str[:10] == oc["date"]
    oc["at_ask"] = traded_today & (oc.ask > 0) & (oc.last_price >= oc.ask * 0.99)
    oc["at_bid"] = traded_today & (oc.bid > 0) & (oc.last_price <= oc.bid * 1.01)
    oc["premium"] = np.where(traded_today, oc.last_price * oc.last_size * 100, 0.0)

    oc.to_parquet(STORE / "options_enriched.parquet", index=False)

    def agg(gr: pd.DataFrame) -> pd.Series:
        traded = gr[gr.premium > 0]
        calls, puts = traded[traded.cp == "C"], traded[traded.cp == "P"]
        atm = gr[(gr.moneyness - 1).abs() < 0.03]
        return pd.Series({
            "spot": gr["spot"].iloc[0],
            "n_contracts": len(gr),
            "total_oi": gr["open_interest"].sum(),
            "gex_total": gr["gex"].sum(skipna=True),
            "atm_iv": atm["iv"].median(),
            "skew_25d": (gr.loc[(gr.cp == "P") & (gr.delta.between(-0.35, -0.15)), "iv"].median()
                         - gr.loc[(gr.cp == "C") & (gr.delta.between(0.15, 0.35)), "iv"].median()),
            "pc_ratio_traded": len(puts) / max(len(calls), 1),
            "call_prem_at_ask": calls.loc[calls.at_ask, "premium"].sum(),
            "put_prem_at_ask": puts.loc[puts.at_ask, "premium"].sum(),
            "call_prem_at_bid": calls.loc[calls.at_bid, "premium"].sum(),
            "put_prem_at_bid": puts.loc[puts.at_bid, "premium"].sum(),
        })

    ag = (oc.groupby(["date", "window", "underlying"])
            .apply(agg, include_groups=False).reset_index())
    ag.to_parquet(STORE / "options_agg.parquet", index=False)
    print(f"options_enriched: {len(oc):,} rows | iv coverage "
          f"{oc['iv'].notna().mean():.0%} | options_agg: {len(ag):,} rows")


if __name__ == "__main__":
    main()
