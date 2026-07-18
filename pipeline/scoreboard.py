"""Five-horizon composite scoreboard.

Blends z-scored features into one score per horizon per ticker. This is a
statistical screen — it ranks evidence, it does not give financial advice,
and the dashboard presents it with that framing.

Reads  data/store/{daily_features,merton_params,hawkes_industry,window_metrics,
                   universe}.parquet and data/store/sentiment.json (optional)
Writes data/store/scoreboard.parquet

Run:  python pipeline/scoreboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"

# feature -> weight, per horizon. Positive weight = higher is better.
WEIGHTS = {
    "1d":  {"micro_alpha_w3": .25, "volz_w3": .10, "mom_1w": .20, "rsi_c": .10,
            "sentiment": .15, "alpha_ann": .10, "neg_vol": .10},
    "2w":  {"mom_1w": .15, "mom_1m": .25, "alpha_ann": .15, "sentiment": .15,
            "dist_52w_high": .10, "neg_vol": .10, "neg_jump": .10},
    "1m":  {"mom_1m": .25, "mom_3m": .20, "alpha_ann": .20, "sentiment": .15,
            "neg_jump": .10, "dist_52w_high": .10},
    "6m":  {"mom_3m": .15, "mom_6m": .25, "alpha_ann": .20, "neg_jump": .15,
            "neg_cluster": .10, "sentiment": .15},
    "12m": {"mom_12m1m": .30, "alpha_ann": .25, "neg_vol": .15, "neg_jump": .15,
            "sentiment": .15},
}


def zscore(s: pd.Series) -> pd.Series:
    return ((s - s.mean()) / s.std()).clip(-3, 3).fillna(0.0)


def main():
    uni = pd.read_parquet(STORE / "universe.parquet")
    df = pd.read_parquet(STORE / "daily_features.parquet").merge(uni, on="ticker")
    mp = pd.read_parquet(STORE / "merton_params.parquet")[["ticker", "jump_var_share",
                                                           "jumps_per_year", "sigma_ann"]]
    hk = pd.read_parquet(STORE / "hawkes_industry.parquet")[["industry", "p_cluster_6m"]]
    wm = pd.read_parquet(STORE / "window_metrics.parquet")

    df = df.merge(mp, on="ticker", how="left").merge(hk, on="industry", how="left")

    last_date = wm["date"].max()
    w3 = wm[(wm.date == last_date) & (wm.window == "W3")][["ticker", "vol_z"]]
    df = df.merge(w3.rename(columns={"vol_z": "volz_w3"}), on="ticker", how="left")

    sent_file = STORE / "sentiment.json"
    sent = {}
    if sent_file.exists():
        raw = json.loads(sent_file.read_text(encoding="utf-8"))
        sent = {k: v.get("score", 0.0) for k, v in raw.get("industries", {}).items()}
    df["sentiment_raw"] = df["industry"].map(sent).fillna(0.0)

    if "data_artifact" in df.columns and df["data_artifact"].any():
        bad = df.loc[df.data_artifact, "ticker"].tolist()
        print(f"excluded from ranking (corporate-action artifact in returns): {bad}")
        df = df[~df.data_artifact]
    ranked = df[df.industry != "benchmark"].copy()
    feats = pd.DataFrame({
        "mom_1w": zscore(ranked.mom_1w), "mom_1m": zscore(ranked.mom_1m),
        "mom_3m": zscore(ranked.mom_3m), "mom_6m": zscore(ranked.mom_6m),
        "mom_12m1m": zscore(ranked.mom_12m1m),
        "alpha_ann": zscore(ranked.alpha_ann),
        "rsi_c": zscore(50 - (ranked.rsi14 - 50).abs()),   # reward mid-range RSI
        "dist_52w_high": zscore(ranked.dist_52w_high),
        "neg_vol": zscore(-ranked.vol_ann),
        "neg_jump": zscore(-ranked.jump_var_share),
        "neg_cluster": zscore(-ranked.p_cluster_6m),
        "micro_alpha_w3": zscore(ranked.micro_alpha_w3),
        "volz_w3": zscore(ranked.volz_w3),
        "sentiment": zscore(ranked.sentiment_raw) if ranked.sentiment_raw.abs().sum() else 0.0,
    }, index=ranked.index)

    for h, w in WEIGHTS.items():
        sc = sum(feats[f] * wt for f, wt in w.items())
        ranked[f"score_{h}"] = sc
        ranked[f"rank_{h}"] = sc.rank(ascending=False).astype(int)

    # liquidity gate: a $200 account loses more to spread than to being wrong
    # in names trading under ~$1M/day — flag them so the UI can hide by default
    ranked["liquid"] = ranked["dollar_vol_21d"] >= 1_000_000

    keep = (["ticker", "industry", "price", "whole_share_200", "liquid", "sentiment_raw",
             "mom_1w", "mom_1m", "mom_3m", "mom_6m", "mom_12m1m", "alpha_ann", "beta",
             "rsi14", "vol_ann", "dist_52w_high", "jump_var_share", "jumps_per_year",
             "p_cluster_6m", "micro_alpha_w3", "volz_w3", "dollar_vol_21d"]
            + [c for c in ranked.columns if c.startswith(("score_", "rank_"))])
    ranked[keep].to_parquet(STORE / "scoreboard.parquet", index=False)
    print(f"scoreboard: {len(ranked)} tickers x {len(WEIGHTS)} horizons "
          f"(sentiment {'loaded' if sent else 'not yet gathered — zeros'})")


if __name__ == "__main__":
    main()
