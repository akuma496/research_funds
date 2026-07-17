"""Compute all window metrics and statistical models from the Parquet store.

Reads  data/store/{bars_minute,bars_daily,universe}.parquet
Writes data/store/window_metrics.parquet   per ticker x Friday x window
       data/store/daily_features.parquet   per ticker (latest snapshot of momentum/alpha/vol)
       data/store/jumps_intraday.parquet   Lee-Mykland jump events on Friday minute bars
       data/store/jumps_daily.parquet      3-sigma daily jump events (2y, all weekdays)
       data/store/merton_params.parquet    jump-diffusion MLE per ticker
       data/store/hawkes_industry.parquet  cluster probabilities per industry
       data/store/backtest_deciles.parquet proxy score decile hit rates

Run:  python pipeline/compute.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"

WINDOWS = {  # local America/Chicago [start, end) in minutes-from-midnight
    "W1": (8 * 60, 8 * 60 + 30),
    "W2": (10 * 60 + 15, 10 * 60 + 45),
    "W3": (14 * 60 + 30, 15 * 60),
    "W4": (15 * 60, 17 * 60),
}
TRADING_DAYS = 252


# ---------------------------------------------------------------- window metrics
def window_metrics(mb: pd.DataFrame) -> pd.DataFrame:
    mb = mb.copy()
    local = mb["ts"].dt.tz_convert("America/Chicago")
    mins = local.dt.hour * 60 + local.dt.minute
    mb["window"] = None
    for w, (a, b) in WINDOWS.items():
        mb.loc[(mins >= a) & (mins < b), "window"] = w
    mb = mb.dropna(subset=["window"])

    def agg(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("ts")
        lr = np.log(g["c"] / g["c"].shift(1)).dropna()
        vol_sum = g["v"].sum()
        vwap = (g["vw"] * g["v"]).sum() / vol_sum if vol_sum else np.nan
        park = np.sqrt(np.mean(np.log(g["h"] / g["l"]) ** 2) / (4 * np.log(2)))
        return pd.Series({
            "open": g["o"].iloc[0], "close": g["c"].iloc[-1],
            "high": g["h"].max(), "low": g["l"].min(),
            "ret": g["c"].iloc[-1] / g["o"].iloc[0] - 1,
            "volume": vol_sum, "n_bars": len(g),
            "rv": lr.std() * np.sqrt(max(len(lr), 1)),   # window realized vol
            "parkinson": park,
            "vwap": vwap,
            "vwap_dev": (g["c"].iloc[-1] - vwap) / vwap if vwap else np.nan,
        })

    wm = (mb.groupby(["ticker", "date", "window"]).apply(agg, include_groups=False)
            .reset_index())
    # relative volume + z-score vs same ticker/window across all Fridays
    grp = wm.groupby(["ticker", "window"])["volume"]
    wm["rvol"] = wm["volume"] / grp.transform("mean")
    sd = grp.transform("std")
    wm["vol_z"] = (wm["volume"] - grp.transform("mean")) / sd.replace(0, np.nan)
    return wm


# ---------------------------------------------------------------- daily features
def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def daily_features(db: pd.DataFrame, wm: pd.DataFrame) -> pd.DataFrame:
    px = db.pivot(index="date", columns="ticker", values="c").sort_index()
    px.index = pd.to_datetime(px.index)
    rets = np.log(px / px.shift(1))
    spy = rets.get("SPY")
    rows = []
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 70:
            continue
        r = rets[t].dropna()
        last = s.iloc[-1]
        def mom(days):
            return float(s.iloc[-1] / s.iloc[-days - 1] - 1) if len(s) > days else np.nan
        # rolling 126d CAPM vs SPY
        j = pd.concat([r, spy], axis=1, keys=["r", "m"]).dropna().tail(126)
        beta = alpha_ann = np.nan
        if len(j) > 60:
            cov = np.cov(j["r"], j["m"])
            beta = cov[0, 1] / cov[1, 1]
            alpha_ann = float((j["r"].mean() - beta * j["m"].mean()) * TRADING_DAYS)
        rows.append({
            "ticker": t, "price": last,
            "mom_1w": mom(5), "mom_1m": mom(21), "mom_3m": mom(63),
            "mom_6m": mom(126), "mom_12m1m": (mom(252) - mom(21)) if len(s) > 252 else np.nan,
            "rsi14": rsi(s),
            "vol_ann": float(r.tail(126).std() * np.sqrt(TRADING_DAYS)),
            "dist_52w_high": float(last / s.tail(252).max() - 1),
            "beta": beta, "alpha_ann": alpha_ann,
            "dollar_vol_21d": float((db[db.ticker == t].tail(21)["c"]
                                     * db[db.ticker == t].tail(21)["v"]).mean()),
            "whole_share_200": bool(last <= 200),
        })
    df = pd.DataFrame(rows)

    # window micro-alpha: last Friday W3 return minus beta * SPY W3 return
    last_date = wm["date"].max()
    w3 = wm[(wm.date == last_date) & (wm.window == "W3")].set_index("ticker")["ret"]
    spy_w3 = w3.get("SPY", 0.0)
    df["micro_alpha_w3"] = df.apply(
        lambda x: w3.get(x.ticker, np.nan) - (x.beta if pd.notna(x.beta) else 1.0) * spy_w3, axis=1)
    return df


# ---------------------------------------------------------------- jumps
def intraday_jumps(mb: pd.DataFrame) -> pd.DataFrame:
    """Lee-Mykland style: |minute return| / local bipower vol > threshold."""
    out = []
    K = 60
    thresh = 4.0
    for (t, d), g in mb.groupby(["ticker", "date"]):
        g = g.sort_values("ts").reset_index(drop=True)
        lr = np.log(g["c"] / g["c"].shift(1))
        bp = (lr.abs() * lr.abs().shift(1)).rolling(K).mean() * (np.pi / 2)
        sig = np.sqrt(bp.clip(lower=1e-10))
        L = lr / sig
        mask = (L.abs() > thresh) & lr.notna() & bp.notna()
        for i in g.index[mask]:
            out.append({"ticker": t, "date": d, "ts": g.at[i, "ts"],
                        "ret_1min": float(lr.iloc[i])})
    return pd.DataFrame(out, columns=["ticker", "date", "ts", "ret_1min"])


def daily_jumps(db: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, g in db.groupby("ticker"):
        g = g.sort_values("date")
        r = np.log(g["c"] / g["c"].shift(1))
        sig = r.rolling(63).std()
        z = r / sig.replace(0, np.nan)
        hit = g[(z.abs() > 3) & r.notna()]
        for i, h in hit.iterrows():
            out.append({"ticker": t, "date": h["date"], "ret": float(r.loc[i]), "z": float(z.loc[i])})
    return pd.DataFrame(out, columns=["ticker", "date", "ret", "z"])


# ---------------------------------------------------------------- Merton MLE
def merton_mle(r: np.ndarray, kmax: int = 8):
    r = r[np.isfinite(r)]
    if len(r) < 120:
        return None
    s0 = r.std()

    def nll(p):
        mu, sig, lam, muj, sigj = p
        dens = np.zeros_like(r)
        pk = np.exp(-lam)
        for k in range(kmax + 1):
            if k:
                pk = pk * lam / k
            dens += pk * norm.pdf(r, mu + k * muj, np.sqrt(sig ** 2 + k * sigj ** 2))
        return -np.log(np.clip(dens, 1e-300, None)).sum()

    x0 = [r.mean(), s0 * 0.8, 0.05, 0.0, 2 * s0]
    bounds = [(-.05, .05), (1e-4, 5 * s0), (1e-4, 1.0), (-.5, .5), (1e-4, 10 * s0)]
    try:
        res = minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
        mu, sig, lam, muj, sigj = res.x
        jvar = lam * (muj ** 2 + sigj ** 2)
        return {"mu_d": mu, "sigma_d": sig, "lambda_d": lam, "mu_j": muj, "sigma_j": sigj,
                "sigma_ann": sig * np.sqrt(TRADING_DAYS),
                "jumps_per_year": lam * TRADING_DAYS,
                "jump_var_share": jvar / (sig ** 2 + jvar),
                "converged": bool(res.success)}
    except Exception:
        return None


def merton_all(db: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in db.groupby("ticker"):
        g = g.sort_values("date")
        r = np.log(g["c"] / g["c"].shift(1)).dropna().values
        m = merton_mle(r)
        if m:
            m["ticker"] = t
            rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Hawkes per industry
def hawkes_fit(times: np.ndarray, T: float):
    """Univariate exp-kernel Hawkes MLE. times sorted, in trading days."""
    if len(times) < 5:
        return None

    def nll(p):
        mu, a, b = p
        A = 0.0
        ll = np.log(mu)  # first event
        for i in range(1, len(times)):
            A = np.exp(-b * (times[i] - times[i - 1])) * (1 + A)
            ll += np.log(mu + a * b * A)
        comp = mu * T + a * np.sum(1 - np.exp(-b * (T - times)))
        return -(ll - comp)

    best = None
    for a0 in (0.2, 0.5):
        try:
            res = minimize(nll, [len(times) / T * 0.7, a0, 0.5],
                           bounds=[(1e-5, 5), (1e-4, 0.95), (0.01, 5)], method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    return best.x if best is not None else None


def hawkes_simulate(mu, a, b, horizon=126, n_paths=400, seed=7):
    """Ogata thinning; exp kernel decays, so intensity just after the current
    point is a valid upper bound until the next event."""
    rng = np.random.default_rng(seed)
    counts, clustered = [], 0
    for _ in range(n_paths):
        t, S, events = 0.0, 0.0, []   # S = sum of exp(-b*(t - t_i)) at time t
        while True:
            lam_bar = mu + a * b * S
            w = rng.exponential(1 / max(lam_bar, 1e-9))
            t += w
            if t >= horizon:
                break
            S *= np.exp(-b * w)
            if rng.uniform() <= (mu + a * b * S) / lam_bar:
                events.append(t)
                S += 1.0
        counts.append(len(events))
        ev = np.array(events)
        if len(ev) >= 3 and np.any(ev[2:] - ev[:-2] <= 10):  # 3 events within 10 days
            clustered += 1
    return float(np.mean(counts)), clustered / n_paths


def hawkes_industry(dj: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    dj = dj.merge(uni, on="ticker")
    all_days = np.sort(dj["date"].unique())
    day_index = {d: i for i, d in enumerate(np.sort(pd.unique(dj["date"])))}
    rows = []
    n_members = uni.groupby("industry")["ticker"].nunique()
    for ind, g in dj.groupby("industry"):
        # event = an industry-wide shock day: >=20% of constituents (min 2)
        # jump on the same day. Any-single-ticker days fire near-daily in a
        # 15-name industry and make P(cluster) degenerate at 1.0.
        need = max(2, int(round(0.2 * n_members.get(ind, 10))))
        per_day = g.groupby("date")["ticker"].nunique()
        days = np.sort(per_day[per_day >= need].index.values)
        times = np.array([day_index[d] for d in days], dtype=float)
        times = times - times.min() + 0.5
        T = float(max(day_index.values())) + 1.0
        fit = hawkes_fit(times, T)
        if fit is None:
            continue
        mu, a, b = fit
        exp_events, p_cluster = hawkes_simulate(mu, a, b)
        rows.append({"industry": ind, "mu": mu, "alpha_branching": a, "beta_decay": b,
                     "n_events_2y": len(times),
                     "expected_events_6m": exp_events,
                     "p_cluster_6m": p_cluster})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- proxy backtest
def backtest_deciles(db: pd.DataFrame) -> pd.DataFrame:
    px = db.pivot(index="date", columns="ticker", values="c").sort_index()
    px.index = pd.to_datetime(px.index)
    wk = px.resample("W-FRI").last()
    mom1m, mom3m = wk / wk.shift(4) - 1, wk / wk.shift(13) - 1
    vol = np.log(wk / wk.shift(1)).rolling(13).std()
    fwd = wk.shift(-1) / wk - 1
    spy_fwd = fwd.get("SPY")
    recs = []
    for dt in wk.index[14:-1]:
        z = lambda s: (s - s.mean()) / s.std()
        score = z(mom1m.loc[dt]) + z(mom3m.loc[dt]) - z(vol.loc[dt])
        f = pd.concat([score.rename("score"), fwd.loc[dt].rename("fwd")], axis=1).dropna()
        if len(f) < 50:
            continue
        f["decile"] = pd.qcut(f["score"], 10, labels=False, duplicates="drop")
        f["beat_spy"] = f["fwd"] > spy_fwd.loc[dt]
        recs.append(f.assign(week=dt))
    allr = pd.concat(recs)
    out = allr.groupby("decile").agg(avg_fwd_1w=("fwd", "mean"),
                                     hit_rate_vs_spy=("beat_spy", "mean"),
                                     n=("fwd", "size")).reset_index()
    return out


# ---------------------------------------------------------------- main
def main():
    uni = pd.read_parquet(STORE / "universe.parquet")
    mb = pd.read_parquet(STORE / "bars_minute.parquet")
    db = pd.read_parquet(STORE / "bars_daily.parquet")

    wm = window_metrics(mb)
    wm.to_parquet(STORE / "window_metrics.parquet", index=False)
    print(f"window_metrics: {len(wm):,} rows")

    df = daily_features(db, wm)
    df.to_parquet(STORE / "daily_features.parquet", index=False)
    print(f"daily_features: {len(df):,} tickers")

    ij = intraday_jumps(mb)
    ij.to_parquet(STORE / "jumps_intraday.parquet", index=False)
    print(f"jumps_intraday: {len(ij):,} events")

    dj = daily_jumps(db)
    dj.to_parquet(STORE / "jumps_daily.parquet", index=False)
    print(f"jumps_daily: {len(dj):,} events")

    mp = merton_all(db)
    mp.to_parquet(STORE / "merton_params.parquet", index=False)
    print(f"merton_params: {len(mp):,} tickers, {int(mp.converged.sum())} converged")

    hk = hawkes_industry(dj, uni)
    hk.to_parquet(STORE / "hawkes_industry.parquet", index=False)
    print(f"hawkes_industry: {len(hk):,} industries")

    bt = backtest_deciles(db)
    bt.to_parquet(STORE / "backtest_deciles.parquet", index=False)
    print("backtest_deciles done")
    print("COMPUTE COMPLETE")


if __name__ == "__main__":
    main()
