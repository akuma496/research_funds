"""Portfolio-25: the highest-composite-probability basket with calibrated risk.

Selection: top 25 liquid names by blended 1m/6m scoreboard score, max 3 per
industry (forced diversification). Weights: inverse-volatility (riskier names
get less money), then the whole basket is scaled so its expected annualized
volatility is ~15% — the rest sits in cash. That's the "risk calibration":
it maximizes the historical probability of a profitable period; nothing can
make profit certain, and the stats below say exactly how often it wasn't.

Backtest: this basket's weekly returns over the past 2 years vs SPY, plus an
OLS regression of its weekly excess returns on SPY's (alpha, beta, R^2,
t-stats). CAVEAT the dashboard repeats: the basket was chosen with TODAY'S
information, so the backtest flatters it (look-ahead). The paper-portfolio
page is the honest forward test.

Writes data/store/{portfolio25,portfolio25_curve}.parquet, portfolio25_stats.json
Run:   python pipeline/portfolio25.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"
N, MAX_PER_IND, VOL_TARGET, RF = 25, 3, 0.15, 0.04


def main():
    sb = pd.read_parquet(STORE / "scoreboard.parquet")
    db = pd.read_parquet(STORE / "bars_daily.parquet")

    pool = sb[sb.get("liquid", True)].copy()
    pool["blend"] = 0.5 * pool["score_1m"] + 0.5 * pool["score_6m"]
    pool = pool.sort_values("blend", ascending=False)
    picks, count = [], {}
    for _, r in pool.iterrows():
        if count.get(r.industry, 0) >= MAX_PER_IND:
            continue
        picks.append(r)
        count[r.industry] = count.get(r.industry, 0) + 1
        if len(picks) == N:
            break
    port = pd.DataFrame(picks)

    px = (db.pivot(index="date", columns="ticker", values="c").sort_index())
    px.index = pd.to_datetime(px.index)
    rets = np.log(px / px.shift(1))

    inv_vol = 1 / port.set_index("ticker")["vol_ann"]
    w = (inv_vol / inv_vol.sum())
    cov = rets[w.index].tail(252).cov() * 252
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    scale = min(VOL_TARGET / port_vol, 1.0) if port_vol > 0 else 1.0
    w_final = w * scale
    cash_weight = 1 - float(w_final.sum())

    port = port.set_index("ticker")
    port["weight"] = w_final
    port = port.reset_index()[["ticker", "industry", "price", "weight", "blend",
                               "score_1m", "score_6m", "vol_ann", "alpha_ann"]]
    port.to_parquet(STORE / "portfolio25.parquet", index=False)

    # ---- weekly backtest of this (look-ahead) basket ----
    wk = px.resample("W-FRI").last()
    wret = wk.pct_change()
    bt = (wret[w_final.index] * w_final).sum(axis=1, min_count=5).dropna()
    spy = wret["SPY"].reindex(bt.index)
    curve = pd.DataFrame({"date": bt.index, "portfolio": (1 + bt).cumprod(),
                          "spy": (1 + spy).cumprod()})
    curve.to_parquet(STORE / "portfolio25_curve.parquet", index=False)

    ex_p, ex_m = bt - RF / 52, spy - RF / 52
    X = np.column_stack([np.ones(len(ex_m)), ex_m.values])
    coef, res, *_ = np.linalg.lstsq(X, ex_p.values, rcond=None)
    resid = ex_p.values - X @ coef
    se = np.sqrt(np.sum(resid ** 2) / (len(ex_p) - 2)
                 * np.linalg.inv(X.T @ X).diagonal())
    r2 = 1 - np.sum(resid ** 2) / np.sum((ex_p - ex_p.mean()) ** 2)
    dd = (curve["portfolio"] / curve["portfolio"].cummax() - 1).min()
    stats = {
        "n_weeks": int(len(bt)),
        "cagr": float((1 + bt).prod() ** (52 / len(bt)) - 1),
        "vol_ann": float(bt.std() * np.sqrt(52)),
        "sharpe": float(ex_p.mean() / bt.std() * np.sqrt(52)) if bt.std() else None,
        "max_drawdown": float(dd),
        "pct_positive_weeks": float((bt > 0).mean()),
        "pct_beat_spy_weeks": float((bt > spy).mean()),
        "alpha_weekly": float(coef[0]), "alpha_ann": float(coef[0] * 52),
        "alpha_t": float(coef[0] / se[0]) if se[0] else None,
        "beta": float(coef[1]), "beta_t": float(coef[1] / se[1]) if se[1] else None,
        "r_squared": float(r2),
        "vol_target": VOL_TARGET, "cash_weight": cash_weight,
        "realized_port_vol_est": port_vol,
    }
    (STORE / "portfolio25_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"portfolio25: {len(port)} names, cash {cash_weight:.0%}, "
          f"backtest alpha {stats['alpha_ann']:+.1%}/yr (t={stats['alpha_t']:.1f}), "
          f"beta {stats['beta']:.2f}, maxDD {dd:.0%}, "
          f"{stats['pct_positive_weeks']:.0%} positive weeks")


if __name__ == "__main__":
    main()
