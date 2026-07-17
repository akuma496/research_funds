"""OPEX Friday Market Microstructure Dashboard.

Launch:  launch_dashboard.bat   (or: python -m streamlit run app/dashboard.py)
Reads the Parquet store only — run the pipeline scripts to refresh data.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "store"

# ---- palette (dark mode steps, validated) ------------------------------------
SERIES = ["#3987e5", "#199e70", "#c98500", "#008300",
          "#9085e9", "#e66767", "#d55181", "#d95926"]
SURFACE, PAGE = "#1a1a19", "#0d0d0d"
INK, INK2, MUTED, GRID = "#ffffff", "#c3c2b7", "#898781", "#2c2c2a"
DIVERGING = [[0.0, "#e66767"], [0.5, "#383835"], [1.0, "#3987e5"]]  # red=down, blue=up
SEQ_BLUE = [[0.0, "#104281"], [0.5, "#3987e5"], [1.0, "#cde2fb"]]

WINDOW_LABEL = {
    "W1": "W1 · Pre-open 8:00–8:30",
    "W2": "W2 · Midday 10:15–10:45",
    "W3": "W3 · Power hour 2:30–3:00",
    "W4": "W4 · After hours 3:00–5:00",
}
IND_LABEL = {
    "photonics": "Photonics", "clean_energy": "Clean energy",
    "bitcoin_mining": "Bitcoin mining", "semiconductors": "Semiconductors",
    "electronics": "Electronics", "nuclear": "Nuclear energy",
    "battery_storage": "Battery storage", "cooling": "Cooling",
    "turbines": "Turbines", "ai_vision": "AI vision",
    "medical_devices": "Medical devices", "medical_robotics": "Medical robotics",
    "miniature_robotics": "Miniature robotics", "defense": "Defense",
    "evtol": "eVTOL", "lipstick": "Lipstick index", "benchmark": "Benchmarks",
}

st.set_page_config(page_title="OPEX Friday Dashboard", page_icon="📈", layout="wide")


def style_fig(fig, height=430):
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=INK2, size=13),
        margin=dict(l=10, r=10, t=48, b=10),
        title_font=dict(color=INK, size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#2c2c2a", font_color=INK),
        colorway=SERIES,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#383835", linecolor="#383835",
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#383835", linecolor="#383835",
                     tickfont=dict(color=MUTED))
    return fig


@st.cache_data(ttl=600)
def load(name):
    p = STORE / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def friday_tag(d: str) -> str:
    dt = date.fromisoformat(d)
    third = 15 + (4 - date(dt.year, dt.month, 15).weekday()) % 7
    if dt.day == third:
        return "⚡ QUAD WITCHING" if dt.month in (3, 6, 9, 12) else "🎯 MONTHLY OPEX"
    return ""


def ind_name(k):
    return IND_LABEL.get(k, k)


# ================================================================ 1 · command center
def page_command_center():
    st.title("📈 Friday Command Center")
    wm, uni = load("window_metrics"), load("universe")
    dates = sorted(wm["date"].unique(), reverse=True)
    c1, c2 = st.columns([1, 2])
    day = c1.selectbox("Friday", dates, format_func=lambda d: f"{d}  {friday_tag(d)}")
    metric = c2.radio("Show", ["Return", "Relative volume", "Volatility", "Volume surprise"],
                      horizontal=True,
                      help="Return = price change inside the half-hour window. "
                           "Relative volume = volume vs the 52-week normal for that window (1 = normal). "
                           "Volatility = realized minute-to-minute movement. "
                           "Volume surprise = how many standard deviations above normal volume ran.")
    tag = friday_tag(day)
    if tag:
        st.info(f"{tag} — options expiration Fridays historically concentrate "
                "institutional rebalancing at the open and close.")

    col = {"Return": "ret", "Relative volume": "rvol",
           "Volatility": "rv", "Volume surprise": "vol_z"}[metric]
    d = wm[wm.date == day].merge(uni, on="ticker")
    d = d[~d.industry.isin(["benchmark"])]
    piv = d.pivot_table(index="industry", columns="window", values=col, aggfunc="median")
    piv = piv.reindex(columns=["W1", "W2", "W3", "W4"])
    piv.index = [ind_name(i) for i in piv.index]

    if col == "ret":
        scale, mid, fmt = DIVERGING, 0.0, ".2%"
    elif col == "rvol":
        scale, mid, fmt = SEQ_BLUE, None, ".2f"
    elif col == "rv":
        scale, mid, fmt = SEQ_BLUE, None, ".3%"
    else:
        scale, mid, fmt = DIVERGING, 0.0, ".2f"
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[WINDOW_LABEL[w] for w in piv.columns], y=piv.index,
        colorscale=scale, zmid=mid, xgap=2, ygap=2,
        text=piv.values, texttemplate="%{text:" + fmt + "}",
        hovertemplate="%{y}<br>%{x}<br>" + metric + ": %{z:" + fmt + "}<extra></extra>"))
    fig.update_layout(title=f"{metric} by industry and time window — {day}")
    st.plotly_chart(style_fig(fig, 560), width="stretch")
    st.caption("Each cell is the median across the industry's tickers. "
               "Times are US Central. Diverging colors: red = down/below normal, blue = up/above.")

    st.subheader("Biggest single-stock moves that day")
    top = d.reindex(d["ret"].abs().sort_values(ascending=False).index)[
        ["ticker", "industry", "window", "ret", "rvol", "vol_z", "volume"]].head(15)
    top["industry"] = top["industry"].map(ind_name)
    top["window"] = top["window"].map(lambda w: WINDOW_LABEL[w].split("·")[0].strip())
    st.dataframe(top.style.format({"ret": "{:+.2%}", "rvol": "{:.1f}×",
                                   "vol_z": "{:+.1f}σ", "volume": "{:,.0f}"}),
                 width="stretch", hide_index=True)


# ================================================================ 2 · industry explorer
def page_industry():
    st.title("🏭 Industry Explorer")
    wm, uni, df = load("window_metrics"), load("universe"), load("daily_features")
    inds = [i for i in sorted(uni.industry.unique()) if i not in ("benchmark",)]
    ind = st.selectbox("Industry", inds, format_func=ind_name)
    members = uni[uni.industry == ind]["ticker"].tolist()
    d = wm[wm.ticker.isin(members)]

    hist = (d.groupby(["date", "window"])["ret"].median().reset_index()
              .pivot(index="date", columns="window", values="ret").sort_index())
    fig = go.Figure()
    for i, w in enumerate(["W1", "W2", "W3", "W4"]):
        if w in hist:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[w], name=WINDOW_LABEL[w],
                                     mode="lines", line=dict(width=2, color=SERIES[i])))
    fig.update_layout(title=f"{ind_name(ind)} — median window return, every Friday for 52 weeks",
                      yaxis_tickformat=".1%")
    st.plotly_chart(style_fig(fig), width="stretch")
    st.caption("Each line is one of your four time windows. Persistent gaps between lines "
               "reveal which part of the day this industry systematically moves.")

    stats = (d.groupby("window")["ret"].agg(["mean", "std"])
               .rename(columns={"mean": "avg return", "std": "volatility"}))
    stats.index = [WINDOW_LABEL[w] for w in stats.index]
    c1, c2 = st.columns(2)
    c1.dataframe(stats.style.format("{:.3%}"), width="stretch")
    m = df[df.ticker.isin(members)][["ticker", "price", "mom_1m", "mom_6m",
                                     "alpha_ann", "vol_ann", "whole_share_200"]]
    m = m.rename(columns={"whole_share_200": "≤$200/share"})
    c2.dataframe(m.sort_values("mom_1m", ascending=False)
                  .style.format({"price": "${:.2f}", "mom_1m": "{:+.1%}",
                                 "mom_6m": "{:+.1%}", "alpha_ann": "{:+.1%}",
                                 "vol_ann": "{:.0%}"}),
                 width="stretch", hide_index=True)


# ================================================================ 3 · ticker deep dive
def page_ticker():
    st.title("🔬 Ticker Deep Dive")
    mb_dates = sorted(load("window_metrics")["date"].unique(), reverse=True)
    uni = load("universe")
    c1, c2 = st.columns(2)
    tick = c1.selectbox("Ticker", sorted(uni.ticker.unique()))
    day = c2.selectbox("Friday", mb_dates)

    mb = pd.read_parquet(STORE / "bars_minute.parquet",
                         filters=[("ticker", "==", tick), ("date", "==", day)])
    if mb.empty:
        st.warning(f"{tick} has no minute bars on {day} — it may not have traded.")
        return
    mb = mb.sort_values("ts")
    local = mb["ts"].dt.tz_convert("America/Chicago")

    fig = go.Figure(go.Candlestick(
        x=local, open=mb["o"], high=mb["h"], low=mb["l"], close=mb["c"],
        increasing_line_color="#199e70", decreasing_line_color="#e66767", name=tick))
    for w, (a, b) in {"W1": (8.0, 8.5), "W2": (10.25, 10.75),
                      "W3": (14.5, 15.0), "W4": (15.0, 17.0)}.items():
        d0 = local.iloc[0].normalize()
        fig.add_vrect(x0=d0 + pd.Timedelta(hours=a), x1=d0 + pd.Timedelta(hours=b),
                      fillcolor="#3987e5", opacity=0.10, line_width=0,
                      annotation_text=w, annotation_font_color=MUTED)
    jumps = load("jumps_intraday")
    j = jumps[(jumps.ticker == tick) & (jumps.date == day)]
    if not j.empty:
        jl = pd.to_datetime(j["ts"]).dt.tz_convert("America/Chicago")
        jp = mb.set_index("ts").reindex(pd.to_datetime(j["ts"]))["c"]
        fig.add_trace(go.Scatter(x=jl, y=jp.values, mode="markers", name="Jump detected",
                                 marker=dict(size=11, color="#c98500", symbol="diamond",
                                             line=dict(width=2, color=SURFACE))))
    fig.update_layout(title=f"{tick} — minute bars on {day} {friday_tag(day)} (Central time)",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(style_fig(fig, 500), width="stretch")
    st.caption("Shaded bands are your four windows. Orange diamonds are statistically "
               "abnormal one-minute price jumps (Lee-Mykland test).")

    wm = load("window_metrics")
    wmt = wm[(wm.ticker == tick) & (wm.date == day)][
        ["window", "ret", "volume", "rvol", "vol_z", "rv", "vwap_dev"]]
    wmt["window"] = wmt["window"].map(WINDOW_LABEL)
    st.dataframe(wmt.style.format({"ret": "{:+.2%}", "volume": "{:,.0f}", "rvol": "{:.1f}×",
                                   "vol_z": "{:+.1f}σ", "rv": "{:.3%}", "vwap_dev": "{:+.2%}"}),
                 width="stretch", hide_index=True)


# ================================================================ 4 · options flow
def page_options():
    st.title("🎯 Options Flow & GEX")
    ag, oe = load("options_agg"), load("options_enriched")
    if ag.empty:
        st.warning("No options snapshots captured yet. The collector runs 4× daily; "
                   "history builds from today forward.")
        return
    snaps = ag[["date", "window"]].drop_duplicates().sort_values(["date", "window"])
    labels = [f"{d} · {w}" for d, w in snaps.values]
    pick = st.selectbox("Snapshot", labels, index=len(labels) - 1)
    day, win = pick.split(" · ")
    a = ag[(ag.date == day) & (ag.window == win)].copy()
    e = oe[(oe.date == day) & (oe.window == win)]
    n_und = a["underlying"].nunique()
    if n_und < 20:
        st.info(f"This snapshot covers {n_und} underlying(s) — full-universe capture "
                "starts with the next scheduled run. Chain history accrues from capture start.")

    has_oi = a["total_oi"].fillna(0).sum() > 0
    c1, c2 = st.columns(2)
    if has_oi:
        g = a.sort_values("gex_total")
        fig = go.Figure(go.Bar(x=g["gex_total"], y=g["underlying"], orientation="h",
                               marker_color=["#e66767" if v < 0 else "#3987e5"
                                             for v in g["gex_total"]]))
        fig.update_layout(title="Dealer gamma exposure (GEX) — $ per 1% move")
        c1.plotly_chart(style_fig(fig), width="stretch")
        c1.caption("Positive (blue): dealers dampen moves. Negative (red): dealers "
                   "amplify moves — fuel for sharp swings, especially on OPEX day.")
    else:
        c1.info("Open interest arrives with the next scheduled capture — GEX will "
                "appear here from the first full-universe snapshot onward.")
    sk = a.dropna(subset=["skew_25d"]).sort_values("skew_25d")
    if not sk.empty:
        fig = go.Figure(go.Bar(x=sk["skew_25d"], y=sk["underlying"], orientation="h",
                               marker_color="#9085e9"))
        fig.update_layout(title="25-delta put−call IV skew (fear gauge)")
        c2.plotly_chart(style_fig(fig), width="stretch")
        c2.caption("Higher = puts pricier than calls = more downside hedging demand.")

    st.subheader("Aggressive prints (filled at the ask)")
    st.caption("The latest trade on each contract, classified against the quote. "
               "'At ask' = buyer paid up — aggressive buying. Ranked by premium.")
    blk = e[e.at_ask & (e.premium > 0)].sort_values("premium", ascending=False)[
        ["underlying", "occ", "cp", "strike", "expiry", "last_price", "last_size",
         "premium", "iv", "delta"]].head(25)
    if blk.empty:
        st.write("No at-ask prints in this snapshot.")
    else:
        blk["cp"] = blk["cp"].map({"C": "CALL", "P": "PUT"})
        st.dataframe(blk.style.format({"strike": "{:.1f}", "last_price": "{:.2f}",
                                       "premium": "${:,.0f}", "iv": "{:.0%}",
                                       "delta": "{:+.2f}"}),
                     width="stretch", hide_index=True)

    t = a[["underlying", "spot", "atm_iv", "pc_ratio_traded",
           "call_prem_at_ask", "put_prem_at_ask"]]
    st.dataframe(t.style.format({"spot": "${:.2f}", "atm_iv": "{:.0%}",
                                 "pc_ratio_traded": "{:.2f}",
                                 "call_prem_at_ask": "${:,.0f}",
                                 "put_prem_at_ask": "${:,.0f}"}),
                 width="stretch", hide_index=True)


# ================================================================ 5 · models lab
def page_models():
    st.title("🧪 Models Lab")
    mp, uni = load("merton_params").merge(load("universe"), on="ticker"), load("universe")
    hk, ji = load("hawkes_industry"), load("jumps_intraday")

    st.subheader("Merton jump-diffusion — how much of each stock's risk is jump risk?")
    st.caption("Every stock's 2-year daily returns fitted to a jump-diffusion. "
               "Right = more volatile overall; higher = more of that risk arrives as "
               "sudden jumps rather than smooth wiggle.")
    m = mp[mp.industry != "benchmark"]
    fig = go.Figure(go.Scatter(
        x=m["sigma_ann"], y=m["jump_var_share"], mode="markers",
        marker=dict(size=9, color=m["jumps_per_year"], colorscale=SEQ_BLUE,
                    colorbar=dict(title="jumps/yr", tickfont=dict(color=MUTED)),
                    line=dict(width=1, color=SURFACE)),
        text=m["ticker"] + " · " + m["industry"].map(ind_name),
        hovertemplate="%{text}<br>diffusion vol %{x:.0%} · jump share %{y:.0%}<extra></extra>"))
    fig.update_layout(title="Jump risk map (each dot = one stock)",
                      xaxis_title="baseline volatility (annualized)",
                      yaxis_title="share of variance from jumps",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 520), width="stretch")

    c1, c2 = st.columns(2)
    c1.subheader("Poisson jump intensity by window")
    c1.caption("Average number of abnormal 1-minute jumps per Friday, by time window — "
               "which half hour actually produces the fireworks.")
    if not ji.empty:
        local = pd.to_datetime(ji["ts"]).dt.tz_convert("America/Chicago")
        mins = local.dt.hour * 60 + local.dt.minute
        wmap = pd.Series("outside", index=ji.index)
        for w, (a, b) in {"W1": (480, 510), "W2": (615, 645),
                          "W3": (870, 900), "W4": (900, 1020)}.items():
            wmap[(mins >= a) & (mins < b)] = w
        lam = (wmap[wmap != "outside"].value_counts() / ji["date"].nunique()).sort_index()
        fig = go.Figure(go.Bar(x=[WINDOW_LABEL[w] for w in lam.index], y=lam.values,
                               marker_color=SERIES[0],
                               text=[f"{v:.1f}" for v in lam.values], textposition="outside"))
        fig.update_layout(title="Jump events per Friday (all tickers pooled)")
        c1.plotly_chart(style_fig(fig, 380), width="stretch")

    c2.subheader("Hawkes clustering — next 6 months")
    c2.caption("Big daily moves self-excite: one shock raises the odds of another. "
               "P(cluster) = probability of ≥3 industry-wide shock days within any "
               "10-day stretch in the next 6 months (simulated).")
    h = hk[hk.industry != "benchmark"].sort_values("p_cluster_6m", ascending=False)
    fig = go.Figure(go.Bar(x=h["p_cluster_6m"], y=h["industry"].map(ind_name),
                           orientation="h", marker_color=SERIES[5],
                           text=[f"{v:.0%}" for v in h["p_cluster_6m"]],
                           textposition="outside"))
    fig.update_layout(title="P(shock cluster within 6 months)", xaxis_tickformat=".0%",
                      xaxis_range=[0, 1.15])
    c2.plotly_chart(style_fig(fig, 380), width="stretch")

    with st.expander("Full Merton parameter table"):
        st.dataframe(mp[["ticker", "industry", "sigma_ann", "jumps_per_year",
                         "mu_j", "sigma_j", "jump_var_share"]]
                     .sort_values("jump_var_share", ascending=False)
                     .style.format({"sigma_ann": "{:.0%}", "jumps_per_year": "{:.1f}",
                                    "mu_j": "{:+.3f}", "sigma_j": "{:.3f}",
                                    "jump_var_share": "{:.0%}"}),
                     width="stretch", hide_index=True)


# ================================================================ 6 · sentiment
def page_sentiment():
    st.title("📰 Sentiment & Events")
    f = STORE / "sentiment.json"
    if not f.exists():
        st.info("The research pass hasn't produced results yet. Sentiment is gathered "
                "from public sources by an automated research agent — check back shortly.")
        return
    data = json.loads(f.read_text(encoding="utf-8"))
    st.caption(f"Generated {data.get('generated_at', '?')} from public sources. "
               "Every claim is cited — click through and judge the source yourself.")

    inds = data.get("industries", {})
    s = (pd.DataFrame([(k, v.get("score", 0)) for k, v in inds.items()],
                      columns=["industry", "score"]).sort_values("score"))
    fig = go.Figure(go.Bar(x=s["score"], y=s["industry"].map(ind_name), orientation="h",
                           marker_color=["#e66767" if v < 0 else "#3987e5"
                                         for v in s["score"]],
                           text=[f"{v:+.1f}" for v in s["score"]], textposition="outside"))
    fig.update_layout(title="Industry sentiment (−1 bearish … +1 bullish)",
                      xaxis_range=[-1.2, 1.2])
    st.plotly_chart(style_fig(fig, 520), width="stretch")

    for k in sorted(inds, key=lambda k: -inds[k].get("score", 0)):
        v = inds[k]
        with st.expander(f"{ind_name(k)} — {v.get('score', 0):+.1f}"):
            st.write(v.get("summary", ""))
            for c in v.get("citations", []):
                st.markdown(f"- [{c.get('title', 'source')}]({c.get('url', '')}) — "
                            f"{c.get('source', '')}, {c.get('date', '')}")

    st.subheader("Scheduled events — next two weeks")
    for ev in data.get("next_week_outlook", []):
        st.markdown(f"**{ev.get('date', '?')}** — {ev.get('event', '')}  \n"
                    f"*{ev.get('relevance', '')}*")
        for c in ev.get("citations", []):
            st.markdown(f"  - [{c.get('title', 'source')}]({c.get('url', '')})")


# ================================================================ 7 · scoreboard
def page_scoreboard():
    st.title("🏆 Scoreboard")
    st.warning("**This is a statistical screen, not financial advice.** It ranks the "
               "evidence the models produce. Scores are estimates with real uncertainty; "
               "past patterns don't guarantee future results. The decision is yours.",
               icon="⚖️")
    sb, bt = load("scoreboard"), load("backtest_deciles")
    h = st.radio("Horizon", ["1d", "2w", "1m", "6m", "12m"], horizontal=True,
                 format_func=lambda x: {"1d": "Tomorrow", "2w": "2 weeks", "1m": "1 month",
                                        "6m": "6 months", "12m": "12 months"}[x])
    c1, c2 = st.columns(2)
    only200 = c1.toggle("Only whole shares ≤ $200", value=False,
                        help="With ~$200 you can buy a whole share of these. "
                             "Fractional shares make the rest accessible too.")
    inds = c2.multiselect("Industries", sorted(sb.industry.unique()), format_func=ind_name)

    d = sb.copy()
    if only200:
        d = d[d.whole_share_200]
    if inds:
        d = d[d.industry.isin(inds)]
    d = d.sort_values(f"score_{h}", ascending=False)

    top = d.head(20)[["ticker", "industry", "price", f"score_{h}", "mom_1m", "mom_6m",
                      "alpha_ann", "vol_ann", "jump_var_share", "p_cluster_6m",
                      "sentiment_raw", "whole_share_200"]]
    top = top.rename(columns={f"score_{h}": "score", "sentiment_raw": "sentiment",
                              "whole_share_200": "≤$200"})
    top["industry"] = top["industry"].map(ind_name)
    st.dataframe(top.style.format({"price": "${:.2f}", "score": "{:+.2f}",
                                   "mom_1m": "{:+.1%}", "mom_6m": "{:+.1%}",
                                   "alpha_ann": "{:+.1%}", "vol_ann": "{:.0%}",
                                   "jump_var_share": "{:.0%}", "p_cluster_6m": "{:.0%}",
                                   "sentiment": "{:+.1f}"}),
                 width="stretch", hide_index=True)
    st.caption("Score = weighted blend of momentum, alpha vs S&P, jump risk, event-cluster "
               "probability, Friday-window flow, and cited sentiment — weights differ by "
               "horizon (short horizons lean on flow; long horizons on risk-adjusted trend).")

    if not bt.empty:
        st.subheader("Does the score mean anything? (honesty panel)")
        st.caption("Proxy backtest: a simplified momentum-and-volatility version of this score, "
                   "computed weekly over the past 2 years. Bars = how often each score decile "
                   "beat the S&P the following week. Decile 9 = top scores. If the right side "
                   "isn't taller than the left, respect that.")
        fig = go.Figure(go.Bar(x=bt["decile"].astype(str), y=bt["hit_rate_vs_spy"],
                               marker_color=SERIES[0],
                               text=[f"{v:.0%}" for v in bt["hit_rate_vs_spy"]],
                               textposition="outside"))
        fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED,
                      annotation_text="coin flip", annotation_font_color=MUTED)
        fig.update_layout(title="Weekly hit rate vs S&P by score decile",
                          xaxis_title="score decile (9 = best scores)",
                          yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 380), width="stretch")


# ================================================================ 8 · lipstick
def page_lipstick():
    st.title("💄 Lipstick Index")
    st.caption("The trade-down thesis: when budgets tighten, big purchases stall but small "
               "affordable luxuries (cosmetics) hold up — so beauty names outperforming the "
               "market can be an early caution sign for consumer stress.")
    db, uni = load("bars_daily"), load("universe")
    lips = uni[uni.industry == "lipstick"]["ticker"].tolist()
    px = db[db.ticker.isin(lips + ["SPY"])].pivot(index="date", columns="ticker", values="c")
    px.index = pd.to_datetime(px.index)
    basket = px[lips].div(px[lips].iloc[0]).mean(axis=1)
    spy = px["SPY"] / px["SPY"].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px.index, y=basket * 100, name="Beauty basket",
                             line=dict(width=2.5, color=SERIES[6])))
    fig.add_trace(go.Scatter(x=px.index, y=spy * 100, name="S&P 500 (SPY)",
                             line=dict(width=2.5, color=SERIES[0])))
    fig.update_layout(title="Beauty basket vs the market (indexed to 100, 2 years)")
    st.plotly_chart(style_fig(fig), width="stretch")

    rel = (basket / spy)
    q = rel.resample("QE").last().pct_change().dropna()
    fig = go.Figure(go.Bar(x=q.index.to_period("Q").astype(str),
                           y=q.values,
                           marker_color=["#e66767" if v < 0 else "#199e70" for v in q.values],
                           text=[f"{v:+.1%}" for v in q.values], textposition="outside"))
    fig.update_layout(title="Quarterly relative performance: beauty vs S&P "
                            "(positive = trade-down signal firming)",
                      yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    members = load("daily_features")[lambda d: d.ticker.isin(lips)][
        ["ticker", "price", "mom_1m", "mom_3m", "mom_6m", "beta"]]
    st.dataframe(members.style.format({"price": "${:.2f}", "mom_1m": "{:+.1%}",
                                       "mom_3m": "{:+.1%}", "mom_6m": "{:+.1%}",
                                       "beta": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("Quarterly revenue from SEC filings joins this panel in a future update — "
               "share behavior is the live proxy meanwhile.")


# ================================================================ nav
pages = st.navigation([
    st.Page(page_command_center, title="Friday Command Center", icon="📈", default=True),
    st.Page(page_industry, title="Industry Explorer", icon="🏭"),
    st.Page(page_ticker, title="Ticker Deep Dive", icon="🔬"),
    st.Page(page_options, title="Options Flow & GEX", icon="🎯"),
    st.Page(page_models, title="Models Lab", icon="🧪"),
    st.Page(page_sentiment, title="Sentiment & Events", icon="📰"),
    st.Page(page_scoreboard, title="Scoreboard", icon="🏆"),
    st.Page(page_lipstick, title="Lipstick Index", icon="💄"),
])
pages.run()
