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
    "OPEN": "OPEN · First half hour 8:30–9:00",
    "W2": "W2 · Midday 10:15–10:45",
    "W3": "W3 · Power hour 2:30–3:00",
    "W4": "W4 · After hours 3:00–5:00",
}
WINDOW_ORDER = ["W1", "OPEN", "W2", "W3", "W4"]
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


# Every table column: (friendly header, plain-English hover explanation).
COLUMN_HELP = {
    "ticker": ("Ticker", "The stock's exchange symbol."),
    "company": ("Company", "Company (or fund) name."),
    "industry": ("Industry", "Which of the 15 tracked industries it belongs to."),
    "window": ("Time window", "Which of the four Friday windows (US Central): "
               "pre-open 8:00–8:30, midday 10:15–10:45, power hour 2:30–3:00, after hours 3:00–5:00."),
    "ret": ("Return", "Price change during that time window."),
    "volume": ("Volume", "Shares traded during the window."),
    "rvol": ("Rel. volume", "Volume vs this stock's 52-week normal for the same window "
             "on the same kind of Friday (expiration Fridays are compared only to other "
             "expiration Fridays). 2.0× means double the usual amount."),
    "vol_z": ("Volume surprise", "How unusual the volume was, in standard deviations, "
              "vs the same window on the same kind of Friday. Above +2σ is genuinely "
              "abnormal — someone showed up."),
    "liquid": ("Liquid", "Yes = trades at least ~$1M per day, so a small order "
               "doesn't get eaten by the bid-ask spread."),
    "ftype": ("Friday type", "Expiration = monthly options expiration (incl. quad "
              "witching); these Fridays run structurally hotter than regular ones."),
    "rv": ("Volatility", "Realized minute-to-minute price movement inside the window."),
    "vwap_dev": ("VWAP gap", "Close vs the volume-weighted average price. Positive = "
                 "finished above the average price paid — buyers won the window."),
    "price": ("Price", "Latest share price."),
    "mom_1w": ("1-week", "Price change over the last week."),
    "mom_1m": ("1-month", "Price change over the last month."),
    "mom_3m": ("3-month", "Price change over the last 3 months."),
    "mom_6m": ("6-month", "Price change over the last 6 months."),
    "mom_12m1m": ("12m − 1m", "Return over the past year excluding the latest month — "
                  "the classic academic momentum measure."),
    "alpha_ann": ("Alpha /yr", "Yearly return above what its market exposure would predict. "
                  "Positive = beating the market risk-adjusted; negative = lagging it."),
    "beta": ("Beta", "Market sensitivity: 1.0 moves with the S&P 500, 2.0 swings twice as hard."),
    "rsi14": ("RSI", "Momentum oscillator 0–100. Above ~70 is often called overbought, below ~30 oversold."),
    "vol_ann": ("Volatility /yr", "Annualized daily volatility — how bumpy the ride is. "
                "The S&P is roughly 15–20%."),
    "dist_52w_high": ("Off 52w high", "How far below its one-year high the price sits."),
    "jump_var_share": ("Jump risk", "Share of total risk arriving as sudden jumps rather than "
                       "day-to-day wiggle (Merton model). High = gap-prone."),
    "jumps_per_year": ("Jumps /yr", "Model-estimated abnormal jump days per year."),
    "mu_j": ("Avg jump", "Average size of a jump when one happens; negative = jumps tend to be drops."),
    "sigma_j": ("Jump spread", "How variable the jump sizes are."),
    "sigma_ann": ("Baseline vol", "The smooth, non-jump part of volatility, annualized."),
    "p_cluster_6m": ("Cluster risk", "Model probability of a burst of industry-wide shock days "
                     "(3+ within two weeks) in the next 6 months."),
    "sentiment": ("Sentiment", "News/analyst tone from cited public sources: −1 bearish to +1 bullish."),
    "score": ("Score", "Composite of all signals, weighted for this horizon. "
              "Higher = stronger combined evidence. A ranking, not advice."),
    "≤$200": ("≤$200", "Yes = one whole share costs $200 or less."),
    "≤$200/share": ("≤$200/share", "Yes = one whole share costs $200 or less."),
    "spot": ("Stock price", "The underlying share price at snapshot time."),
    "atm_iv": ("ATM impl. vol", "Expected future volatility priced into at-the-money options."),
    "pc_ratio_traded": ("Put/Call", "Traded put contracts per call contract. Above 1 = more "
                        "put (downside) activity."),
    "call_prem_at_ask": ("Call $ at ask", "Dollars of call premium bought aggressively (at the ask)."),
    "put_prem_at_ask": ("Put $ at ask", "Dollars of put premium bought aggressively (at the ask)."),
    "occ": ("Contract", "The standardized option contract code (ticker, expiry, C/P, strike)."),
    "cp": ("Type", "CALL = upside bet or hedge; PUT = downside bet or protection."),
    "strike": ("Strike", "The price at which the option can be exercised."),
    "expiry": ("Expires", "The option's expiration date."),
    "last_price": ("Last price", "Price of the most recent trade on this contract."),
    "last_size": ("Last size", "Contracts in that most recent trade."),
    "premium": ("Premium", "Dollars paid: price × contracts × 100 shares."),
    "iv": ("Impl. vol", "Volatility implied by this option's price."),
    "delta": ("Delta", "Moves ~this much per $1 stock move; also roughly the market's "
              "probability it finishes in the money."),
    "side": ("Side", "Buyer- or seller-initiated, judged from the quote (at ask/at bid) "
             "or the price sequence (tick)."),
    "size": ("Contracts", "Number of contracts in this single print. 50+ is block-sized."),
    "ts": ("Time", "Time of the print (US Central)."),
    "n_contracts": ("# contracts", "Option contracts listed in the snapshot."),
    "total_oi": ("Open interest", "Total contracts outstanding across the chain."),
    "n_form4_90d": ("# filings", "Form 4 insider filings in the last 90 days."),
    "buy_shares": ("Shares bought", "Shares insiders bought on the open market (code P)."),
    "sell_shares": ("Shares sold", "Shares insiders sold (code S)."),
    "buy_value": ("$ bought", "Dollar value of insider open-market buys."),
    "sell_value": ("$ sold", "Dollar value of insider sales."),
    "net_value": ("Net insider $", "Buys minus sells. Positive = insiders putting their own money in."),
    "chg": ("Change QoQ", "Panel holdings: latest quarter vs prior quarter."),
    "micro_alpha_w3": ("Power-hour alpha", "Last Friday's power-hour return beyond what the "
                       "market's move explains."),
    "px": ("Price", "Current share price."),
    "tier": ("Price tier", "Price bucket: sub-$20 / $80 / $100 / $150 / $200 / above."),
    "cash": ("Cash reserves", "Cash & equivalents from the company's latest SEC filing."),
    "drift_ann": ("Drift /yr", "The price's own annualized direction of travel over "
                  "6 months, regardless of the market."),
    "mr_z": ("Mismatch (σ)", "How stretched price is vs its own 50-day average, in "
             "standard deviations. Beyond ±2 is a big rubber band."),
    "mismatch_pct": ("Mismatch %", "Price vs its 50-day average in percent."),
    "half_life_d": ("Half-life (d)", "How many days it historically takes for half of a "
                    "stretch to snap back. Short = fast mean-reverter."),
    "short_pct_float": ("Short % float", "Share of freely-trading shares sold short. "
                        "Above ~15% is crowded; above 25% is squeeze fuel."),
    "days_to_cover": ("Days to cover", "Days of normal volume shorts would need to buy "
                      "back. Higher = more explosive if forced."),
    "squeeze_pctl": ("Squeeze rank", "Relative squeeze-setup rank, 0–100 percentile. "
                     "A ranking, not a calibrated probability."),
    "meanrev_pctl": ("Reversion rank", "Relative mean-reversion-setup rank, 0–100."),
    "meanrev_direction": ("Direction", "Which way the snap-back would go from here."),
    "sideways_pctl": ("Sideways rank", "Relative range-bound likelihood rank, 0–100."),
    "blackswan_pctl": ("Tail-risk rank", "Relative exposure to violent gaps, 0–100. "
                       "This ranks exposure — no one can predict the swan itself."),
    "sep_p50": ("Sept target", "Model median price for 2026-09-30 (2,000 simulated "
                "paths from this stock's own jump-diffusion). Wide bands are honest."),
    "dec_p50": ("Dec target", "Model median price for 2026-12-31."),
    "analyst_mean": ("Street target", "Wall Street analysts' mean 12-month price target."),
    "etf_top10_count": ("In ETFs", "How many of our tracked ETFs hold it in their "
                        "top-10 holdings (free data floor, not a full count)."),
    "weight": ("Weight", "Share of the portfolio's money in this name (inverse to its "
               "volatility, so risky names get less)."),
    "label": ("Basket day", "Which day picked the basket; expiration Fridays tagged."),
    "volz_w3": ("Power-hour vol z", "Volume surprise in last Friday's power hour."),
    "dollar_vol_21d": ("$ volume /day", "Average daily dollars traded (21 days) — liquidity."),
}


def col_cfg(df, extra: dict = None) -> dict:
    cfg = {c: st.column_config.Column(label=lab, help=hlp)
           for c, (lab, hlp) in COLUMN_HELP.items() if c in df.columns}
    for c, (lab, hlp) in (extra or {}).items():
        cfg[c] = st.column_config.Column(label=lab, help=hlp)
    return cfg


@st.cache_data(ttl=600)
def name_map() -> dict:
    u = load("universe")
    if "name" not in u.columns:
        return {}
    return dict(zip(u["ticker"], u["name"]))


def t_label(t: str) -> str:
    n = name_map().get(t, "")
    return f"{t} — {n}" if n and n != t else t


@st.cache_data(ttl=600)
def price_map() -> dict:
    df = load("daily_features")
    if df.empty:
        return {}
    return dict(zip(df["ticker"], df["price"]))


def add_name(df: pd.DataFrame, after: str = "ticker") -> pd.DataFrame:
    """Company name + current share price beside every ticker column."""
    df = df.copy()
    df.insert(df.columns.get_loc(after) + 1, "company", df[after].map(name_map()))
    if "price" not in df.columns and "px" not in df.columns:
        df.insert(df.columns.get_loc("company") + 1, "px",
                  df[after].map(price_map()))
    return df


def analyst(lines: list):
    """Auto-generated analyst read — data-driven sentences, no chatter."""
    lines = [l for l in lines if l]
    if lines:
        st.info("🧑‍💼 **Analyst read**\n\n" + "\n".join(f"- {l}" for l in lines))


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
    piv = piv.reindex(columns=[w for w in WINDOW_ORDER if w in piv.columns])
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

    # data-driven analyst summary of the selected Friday
    w3 = d[d.window == "W3"]
    if not w3.empty:
        by_ind = w3.groupby("industry")["ret"].median().sort_values()
        breadth = (by_ind > 0).mean()
        mover = d.reindex(d["ret"].abs().sort_values(ascending=False).index).iloc[0]
        hot = d[d.vol_z > 2]
        analyst([
            f"Power hour: {ind_name(by_ind.index[-1])} led ({by_ind.iloc[-1]:+.2%} median), "
            f"{ind_name(by_ind.index[0])} lagged ({by_ind.iloc[0]:+.2%}); "
            f"{breadth:.0%} of industries closed the window positive — "
            + ("broad risk-on." if breadth > 0.65 else
               "broad risk-off." if breadth < 0.35 else "a mixed, stock-picker's tape."),
            f"Biggest single move: {t_label(mover.ticker)} {mover.ret:+.1%} in the "
            f"{WINDOW_LABEL.get(mover.window, mover.window).split('·')[-1].strip()} window "
            f"on {mover.rvol:.1f}× normal volume.",
            f"{len(hot)} ticker-windows ran >2σ volume vs their normal for this kind of "
            f"Friday — " + ("unusual participation; someone repositioned."
                            if len(hot) > 25 else "participation was ordinary."),
            (f"This was an expiration Friday — open/close flow reflects options "
             f"rebalancing, so fade the noise in W1/W3." if tag else None)])

    st.subheader("Biggest single-stock moves that day")
    top = d.reindex(d["ret"].abs().sort_values(ascending=False).index)[
        ["ticker", "industry", "window", "ret", "rvol", "vol_z", "volume"]].head(15)
    top = add_name(top)
    top["industry"] = top["industry"].map(ind_name)
    top["window"] = top["window"].map(lambda w: WINDOW_LABEL[w].split("·")[0].strip())
    st.dataframe(top.style.format({"ret": "{:+.2%}", "rvol": "{:.1f}×",
                                   "vol_z": "{:+.1f}σ", "volume": "{:,.0f}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(top))


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
    for i, w in enumerate(WINDOW_ORDER):
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
    c1.dataframe(stats.style.format("{:.3%}"), width="stretch",
                 column_config=col_cfg(stats, extra={
                     "avg return": ("Avg return", "Average return in this window across all 52 Fridays."),
                     "volatility": ("Spread", "How much that window's return varies Friday to Friday.")}))
    m = df[df.ticker.isin(members)][["ticker", "price", "mom_1m", "mom_6m",
                                     "alpha_ann", "vol_ann", "whole_share_200"]]
    m = add_name(m)
    m = m.rename(columns={"whole_share_200": "≤$200/share"})
    c2.dataframe(m.sort_values("mom_1m", ascending=False)
                  .style.format({"price": "${:.2f}", "mom_1m": "{:+.1%}",
                                 "mom_6m": "{:+.1%}", "alpha_ann": "{:+.1%}",
                                 "vol_ann": "{:.0%}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(m))


# ================================================================ 3 · ticker deep dive
def page_ticker():
    st.title("🔬 Ticker Deep Dive")
    mb_dates = sorted(load("window_metrics")["date"].unique(), reverse=True)
    uni = load("universe")
    c1, c2 = st.columns(2)
    tick = c1.selectbox("Ticker", sorted(uni.ticker.unique()), format_func=t_label)
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
    for w, (a, b) in {"W1": (8.0, 8.5), "OPEN": (8.5, 9.0), "W2": (10.25, 10.75),
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
    fig.update_layout(title=f"{t_label(tick)} — minute bars on {day} {friday_tag(day)} (Central time)",
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
                 width="stretch", hide_index=True, column_config=col_cfg(wmt))

    # ---- quant panel: drift vs alpha vs mismatch + price targets ----
    q, tg = load("quant_signals"), load("price_targets")
    qt = q[q.ticker == tick]
    if not qt.empty:
        r = qt.iloc[0]
        st.subheader("Quant read")
        c = st.columns(5)
        c[0].metric("Drift /yr", f"{r.drift_ann:+.0%}",
                    help=COLUMN_HELP["drift_ann"][1])
        c[1].metric("Alpha /yr", f"{r.alpha_ann:+.0%}" if pd.notna(r.alpha_ann) else "—",
                    help=COLUMN_HELP["alpha_ann"][1])
        c[2].metric("Mismatch", f"{r.mr_z:+.1f}σ" if pd.notna(r.mr_z) else "—",
                    help=COLUMN_HELP["mr_z"][1])
        c[3].metric("Half-life", f"{r.half_life_d:.0f}d" if pd.notna(r.half_life_d) else "—",
                    help=COLUMN_HELP["half_life_d"][1])
        c[4].metric("Price tier", r.tier, help=COLUMN_HELP["tier"][1])
        drift_vs_alpha = ("its own drift and its market-adjusted alpha agree — the move "
                          "is genuinely its own" if pd.notna(r.alpha_ann)
                          and np.sign(r.drift_ann) == np.sign(r.alpha_ann)
                          else "drift and alpha disagree — the raw trend is mostly a "
                               "market/beta effect, not stock-specific strength")
        stretch = (f"price sits {r.mr_z:+.1f}σ from its 50-day average "
                   f"({r.mismatch_pct:+.1%}), and historically half of such a stretch "
                   f"decays in ~{r.half_life_d:.0f} days"
                   if pd.notna(r.mr_z) and pd.notna(r.half_life_d) else None)
        analyst([f"{t_label(tick)}: drift {r.drift_ann:+.0%}/yr vs alpha "
                 f"{(r.alpha_ann if pd.notna(r.alpha_ann) else 0):+.0%}/yr — {drift_vs_alpha}.",
                 stretch])
    tgt = tg[tg.ticker == tick] if not tg.empty else pd.DataFrame()
    if not tgt.empty:
        st.subheader("Model price targets (2,000 simulated paths)")
        cols = st.columns(len(tgt))
        for col, (_, row) in zip(cols, tgt.iterrows()):
            when = "End of Sept" if "09-30" in row.target_date else "End of Dec"
            col.metric(f"{when} 2026", f"${row.p50:,.2f}",
                       help=f"Median of 2,000 jump-diffusion paths. 80% of paths "
                            f"ended between ${row.p10:,.2f} and ${row.p90:,.2f} — "
                            f"that width is the honest uncertainty.")
            col.caption(f"80% band: ${row.p10:,.2f}–${row.p90:,.2f}"
                        + (f" · Street: ${row.analyst_mean:,.2f}"
                           if pd.notna(row.analyst_mean) else ""))
        st.caption("Model targets come from each stock's own volatility and jump "
                   "behavior — they say what's *typical* for this stock, not what "
                   "news will happen. The Street number is analysts' 12-month view.")


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
    if not e.empty and "fetched_at" in e.columns and e["fetched_at"].notna().any():
        cap = pd.to_datetime(e["fetched_at"].dropna().iloc[0])
        st.caption(f"🕐 Chain actually captured at **{cap.strftime('%H:%M %Z')}** — if that's "
                   "far from the window's nominal time (e.g., the laptop was asleep), read "
                   "this snapshot as the state at capture time, not at the window.")
    n_und = a["underlying"].nunique()
    if n_und < 20:
        st.info(f"This snapshot covers {n_und} underlying(s) — full-universe capture "
                "starts with the next scheduled run. Chain history accrues from capture start.")

    has_oi = a["total_oi"].fillna(0).sum() > 0
    c1, c2 = st.columns(2)
    if has_oi:
        g = a.sort_values("gex_total")
        fig = go.Figure(go.Bar(x=g["gex_total"], y=g["underlying"], orientation="h",
                               customdata=g["underlying"].map(name_map()),
                               hovertemplate="%{y} — %{customdata}<br>GEX %{x:$,.0f}<extra></extra>",
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
                               customdata=sk["underlying"].map(name_map()),
                               hovertemplate="%{y} — %{customdata}<br>skew %{x:.3f}<extra></extra>",
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
        blk = add_name(blk, after="underlying")
        blk["cp"] = blk["cp"].map({"C": "CALL", "P": "PUT"})
        st.dataframe(blk.style.format({"strike": "{:.1f}", "last_price": "{:.2f}",
                                       "premium": "${:,.0f}", "iv": "{:.0%}",
                                       "delta": "{:+.2f}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(blk, extra={
                         "underlying": ("Ticker", "The stock the option is on.")}))

    t = add_name(a[["underlying", "spot", "atm_iv", "pc_ratio_traded",
                    "call_prem_at_ask", "put_prem_at_ask"]], after="underlying")
    st.dataframe(t.style.format({"spot": "${:.2f}", "atm_iv": "{:.0%}",
                                 "pc_ratio_traded": "{:.2f}",
                                 "call_prem_at_ask": "${:,.0f}",
                                 "put_prem_at_ask": "${:,.0f}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(t, extra={
                     "underlying": ("Ticker", "The stock the options are on.")}))

    # ---- block tape (full trade-by-trade prints, collected after each close) ----
    tape = load("options_tape")
    st.subheader("Block tape — the day's biggest options prints")
    if tape.empty or day not in set(tape["date"]):
        st.info("No trade tape collected for this date yet. The tape collector runs "
                "at 3:25 PM Central each weekday and covers every print of the day.")
        return
    tp = tape[(tape.date == day) & tape.is_block].copy()
    st.caption(f"{len(tp):,} block prints (≥50 contracts or ≥$25k premium) from "
               f"{tape[tape.date == day].shape[0]:,} sizeable prints collected for {day}. "
               "Side: quote rule first, tick rule fallback.")
    buys_all = tp[tp.side.str.startswith("buy")]
    cb = buys_all.loc[buys_all.cp == "C", "premium"].sum()
    pb = buys_all.loc[buys_all.cp == "P", "premium"].sum()
    if cb + pb > 0:
        top_und = buys_all.groupby("underlying")["premium"].sum().idxmax()
        analyst([
            f"Aggressive block flow: ${cb / 1e6:,.0f}M into calls vs "
            f"${pb / 1e6:,.0f}M into puts ({cb / (cb + pb):.0%} calls) — "
            + ("buyers leaned bullish." if cb > 1.5 * pb else
               "buyers leaned defensive." if pb > 1.5 * cb else
               "no clear directional lean."),
            f"Heaviest aggressive interest: {t_label(top_und)}.",
            "Caveat: spread legs aren't filtered out yet, so read this as where the "
            "action is, not a pure directional bet."])
    buys = tp[tp.side.str.startswith("buy")]
    net = (buys.groupby(["underlying", "cp"])["premium"].sum().unstack(fill_value=0)
               .rename(columns={"C": "call_buys", "P": "put_buys"}))
    net["total"] = net.sum(axis=1)
    net = net.sort_values("total", ascending=False).head(15)
    hover = pd.Series(net.index, index=net.index).map(name_map())
    fig = go.Figure()
    fig.add_trace(go.Bar(y=net.index, x=net.get("call_buys", 0), name="Call buying",
                         customdata=hover,
                         hovertemplate="%{y} — %{customdata}<br>call buys %{x:$,.0f}<extra></extra>",
                         orientation="h", marker_color="#199e70"))
    fig.add_trace(go.Bar(y=net.index, x=net.get("put_buys", 0), name="Put buying",
                         customdata=hover,
                         hovertemplate="%{y} — %{customdata}<br>put buys %{x:$,.0f}<extra></extra>",
                         orientation="h", marker_color="#e66767"))
    fig.update_layout(barmode="stack", title="Aggressive block premium by underlying "
                                             "(buyer-initiated only)")
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    show = tp.sort_values("premium", ascending=False).head(30)[
        ["underlying", "cp", "strike", "expiry", "ts", "price", "size",
         "premium", "side"]]
    show = add_name(show, after="underlying")
    show["cp"] = show["cp"].map({"C": "CALL", "P": "PUT"})
    show["ts"] = pd.to_datetime(show["ts"]).dt.tz_convert("America/Chicago").dt.strftime("%H:%M:%S")
    st.dataframe(show.style.format({"strike": "{:.1f}", "price": "{:.2f}",
                                    "premium": "${:,.0f}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(show, extra={
                     "underlying": ("Ticker", "The stock the option is on."),
                     "price": ("Price", "Price paid per contract in this print.")}))


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
        text=m["ticker"].map(t_label) + " · " + m["industry"].map(ind_name),
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
        for w, (a, b) in {"W1": (480, 510), "OPEN": (510, 540), "W2": (615, 645),
                          "W3": (870, 900), "W4": (900, 1020)}.items():
            wmap[(mins >= a) & (mins < b)] = w
        lam_order = [w for w in WINDOW_ORDER if w in set(wmap)]
        lam = (wmap[wmap != "outside"].value_counts() / ji["date"].nunique())
        lam = lam.reindex([w for w in lam_order if w in lam.index])
        fig = go.Figure(go.Bar(x=[WINDOW_LABEL[w] for w in lam.index], y=lam.values,
                               marker_color=[("#898781" if w in ("W1", "W4") else SERIES[0])
                                             for w in lam.index],
                               text=[f"{v:.1f}" for v in lam.values], textposition="outside"))
        fig.update_layout(title="Jump events per Friday (all tickers pooled)")
        c1.plotly_chart(style_fig(fig, 380), width="stretch")
        c1.caption("⚠️ The grey bars (pre-open and after-hours) are inflated: thin "
                   "books make ordinary bid-ask bounce look like jumps. Trust the "
                   "regular-session bars; treat the grey ones as an upper bound.")

    c2.subheader("Hawkes clustering — 1 to 6 months")
    c2.caption("Big daily moves self-excite: one shock raises the odds of another. "
               "Each line: probability of a shock cluster (≥3 industry-wide shock "
               "days within 10 days) by horizon. Steep early rise = danger is near-term.")
    h = hk[hk.industry != "benchmark"].sort_values("p_cluster_6m", ascending=False)
    hcols = [c for c in ["p_cluster_1m", "p_cluster_2m", "p_cluster_3m",
                         "p_cluster_4m", "p_cluster_5m", "p_cluster_6m"]
             if c in h.columns]
    fig = go.Figure()
    for i, (_, r) in enumerate(h.head(6).iterrows()):
        fig.add_trace(go.Scatter(x=[c.split("_")[-1] for c in hcols],
                                 y=[r[c] for c in hcols], name=ind_name(r.industry),
                                 mode="lines+markers",
                                 line=dict(width=2, color=SERIES[i % len(SERIES)])))
    fig.update_layout(title="P(shock cluster) by horizon — 6 riskiest industries",
                      yaxis_tickformat=".0%")
    c2.plotly_chart(style_fig(fig, 380), width="stretch")

    hh = load("hawkes_history")
    st.subheader("How cluster risk is moving over time")
    if not hh.empty and hh["as_of"].nunique() > 1:
        snaps = sorted(hh["as_of"].unique())
        piv = hh.pivot_table(index="industry", columns="as_of",
                             values="p_cluster_6m")
        chg = (piv[snaps[-1]] - piv[snaps[0]]).dropna().sort_values()
        fig = go.Figure(go.Bar(x=chg.values, y=[ind_name(i) for i in chg.index],
                               orientation="h",
                               marker_color=["#199e70" if v < 0 else "#e66767"
                                             for v in chg.values],
                               text=[f"{v:+.0%}" for v in chg.values],
                               textposition="outside"))
        fig.update_layout(title=f"6-month cluster risk: change from {snaps[0]} to {snaps[-1]}",
                          xaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
        st.caption("Green = risk cooling since the first snapshot; red = heating. "
                   "A new snapshot is stored at every data refresh.")
    else:
        st.caption("Only one snapshot recorded so far — this comparison fills in "
                   "automatically as refreshes accumulate.")

    with st.expander("Full Merton parameter table"):
        mt = add_name(mp[["ticker", "industry", "sigma_ann", "jumps_per_year",
                          "mu_j", "sigma_j", "jump_var_share"]].sort_values(
                              "jump_var_share", ascending=False))
        st.dataframe(mt.style.format({"sigma_ann": "{:.0%}", "jumps_per_year": "{:.1f}",
                                      "mu_j": "{:+.3f}", "sigma_j": "{:.3f}",
                                      "jump_var_share": "{:.0%}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(mt))


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
        verdict = v.get("verdict") or v.get("summary", "")
        with st.expander(f"{ind_name(k)}  {v.get('score', 0):+.1f} — {verdict[:80]}"):
            st.markdown(f"**Verdict:** {verdict}")
            if v.get("drivers"):
                st.markdown("**Why:**\n" + "\n".join(f"- {x}" for x in v["drivers"]))
            if v.get("risks"):
                st.markdown("**What breaks it:**\n"
                            + "\n".join(f"- {x}" for x in v["risks"]))
            if v.get("watch"):
                st.markdown("**Watch:**\n" + "\n".join(f"- 📅 {x}" for x in v["watch"]))
            if v.get("action"):
                st.markdown(f"**So what:** *{v['action']}*")
            elif v.get("summary") and not v.get("verdict"):
                st.write(v["summary"])
            for c in v.get("citations", []):
                st.markdown(f"- [{c.get('title', 'source')}]({c.get('url', '')}) — "
                            f"{c.get('source', '')}, {c.get('date', '')}")

    fund = load("fundamentals")
    if not fund.empty and fund["etf_top10_count"].max() > 0:
        st.subheader("ETF ownership — who's in the big funds")
        st.caption("How many of our tracked ETFs (SPY, QQQ, SMH, sector and thematic "
                   "funds) hold each stock among their TOP-10 holdings. Free data only "
                   "exposes top holdings, so this is a floor — inclusion in a fund's "
                   "top 10 means passive money buys it automatically on every inflow.")
        em = fund[fund.etf_top10_count > 0].sort_values("etf_top10_count",
                                                        ascending=False)
        em = add_name(em[["ticker", "etf_top10_count", "etf_top10_list"]])
        st.dataframe(em, width="stretch", hide_index=True,
                     column_config=col_cfg(em, extra={
                         "etf_top10_list": ("Which ETFs", "The tracked ETFs holding it "
                                            "in their top 10.")}))

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
    c1, c2, c3 = st.columns(3)
    only200 = c1.toggle("Only whole shares ≤ $200", value=False,
                        help="With ~$200 you can buy a whole share of these. "
                             "Fractional shares make the rest accessible too.")
    liq = c2.toggle("Hide illiquid (<$1M/day)", value=True,
                    help="Stocks trading under ~$1M/day have wide bid-ask spreads — "
                         "a small account loses more to the spread than most weekly "
                         "moves deliver. On by default for your protection.")
    inds = c3.multiselect("Industries", sorted(sb.industry.unique()), format_func=ind_name)

    d = sb.copy()
    q, tg, fund = load("quant_signals"), load("price_targets"), load("fundamentals")
    if not q.empty:
        d = d.merge(q[["ticker", "tier"]], on="ticker", how="left")
    if not fund.empty:
        d = d.merge(fund[["ticker", "cash"]], on="ticker", how="left")
    if not tg.empty:
        piv = tg.pivot(index="ticker", columns="target_date", values="p50")
        ren = {c: ("sep_p50" if "09-30" in c else "dec_p50") for c in piv.columns}
        d = d.merge(piv.rename(columns=ren).reset_index(), on="ticker", how="left")
    if only200:
        d = d[d.whole_share_200]
    if liq and "liquid" in d.columns:
        d = d[d.liquid]
    if inds:
        d = d[d.industry.isin(inds)]
    d = d.sort_values(f"score_{h}", ascending=False)

    top3 = d.head(3)
    analyst([
        f"Top of this board: " + ", ".join(
            f"{t_label(r.ticker)} ({r[f'score_{h}']:+.2f})" for _, r in top3.iterrows()) + ".",
        f"What's driving them: median 1-month momentum {top3.mom_1m.median():+.1%}, "
        f"median alpha {top3.alpha_ann.median():+.1%}/yr, sentiment "
        f"{top3.sentiment_raw.median():+.1f} — "
        + ("trend and narrative agree." if top3.sentiment_raw.median() > 0
           else "the tape is ahead of the narrative; that divergence is the risk."),
        "Scores are relative evidence, not probabilities of profit — cross-check the "
        "same names on the Screens page for their risk setups."])

    show_cols = ["ticker", "industry", "price", "tier", "cash", f"score_{h}",
                 "sep_p50", "dec_p50", "mom_1m", "mom_6m", "alpha_ann", "vol_ann",
                 "jump_var_share", "sentiment_raw", "whole_share_200"]
    top = d.head(20)[[c for c in show_cols if c in d.columns]]
    top = add_name(top)
    top = top.rename(columns={f"score_{h}": "score", "sentiment_raw": "sentiment",
                              "whole_share_200": "≤$200"})
    top["industry"] = top["industry"].map(ind_name)
    st.dataframe(top.style.format({"price": "${:.2f}", "score": "{:+.2f}",
                                   "cash": "${:,.0f}", "sep_p50": "${:,.2f}",
                                   "dec_p50": "${:,.2f}",
                                   "mom_1m": "{:+.1%}", "mom_6m": "{:+.1%}",
                                   "alpha_ann": "{:+.1%}", "vol_ann": "{:.0%}",
                                   "jump_var_share": "{:.0%}",
                                   "sentiment": "{:+.1f}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(top))
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


# ================================================================ screens
def page_screens():
    st.title("🔍 Screens")
    q = load("quant_signals")
    if q.empty:
        st.info("Run the data refresh to build the screens.")
        return
    st.caption("Four setups, top 25 each, refreshed with the data and rolling "
               "roughly a week to a month forward. Ranks are relative percentiles "
               "across the universe — a 95 means 'better setup than 95% of our "
               "stocks', not '95% chance it happens'. Statistical screens, not advice.")

    fmt_money = lambda v: f"${v / 1e9:.1f}B" if pd.notna(v) and v >= 1e9 else (
        f"${v / 1e6:.0f}M" if pd.notna(v) else "—")
    tabs = st.tabs(["🚀 Short squeeze", "🪃 Mean reversal", "😴 Sideways", "🦢 Black swan"])

    with tabs[0]:
        d = q[q.squeeze_top25].sort_values("squeeze_score", ascending=False)
        top = d.iloc[0] if len(d) else None
        analyst([
            f"Most crowded setup: {t_label(top.ticker)} — {top.short_pct_float:.0%} of "
            f"float short, {top.days_to_cover:.1f} days to cover, and a "
            f"{top.mom_1w:+.1%} week that pressures shorts." if top is not None else None,
            f"{(d.short_pct_float > 0.20).sum()} of the 25 carry >20% of float short — "
            "genuine squeeze fuel; the rest are moderate.",
            "A squeeze needs a spark (news, earnings, forced covering) — this screen "
            "finds the dry tinder, not the match."])
        show = add_name(d[["ticker", "short_pct_float", "days_to_cover", "mom_1w",
                           "squeeze_pctl"]])
        st.dataframe(show.style.format({"short_pct_float": "{:.1%}",
                                        "days_to_cover": "{:.1f}", "mom_1w": "{:+.1%}",
                                        "px": "${:.2f}", "squeeze_pctl": "{:.0f}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(show))

    with tabs[1]:
        d = q[q.meanrev_top25].sort_values("meanrev_score", ascending=False)
        ups = (d.meanrev_direction == "reverts UP").sum()
        analyst([
            f"{ups} of 25 are stretched BELOW their average (snap-back would be up); "
            f"{25 - ups} are stretched above (snap-back would be down).",
            f"Fastest spring: {t_label(d.iloc[0].ticker)} — {d.iloc[0].mr_z:+.1f}σ from "
            f"its 50-day mean with a {d.iloc[0].half_life_d:.0f}-day half-life."
            if len(d) else None])
        show = add_name(d[["ticker", "mr_z", "mismatch_pct", "half_life_d", "rsi14",
                           "meanrev_direction", "meanrev_pctl"]])
        st.dataframe(show.style.format({"mr_z": "{:+.1f}", "mismatch_pct": "{:+.1%}",
                                        "half_life_d": "{:.0f}", "rsi14": "{:.0f}",
                                        "px": "${:.2f}", "meanrev_pctl": "{:.0f}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(show))

    with tabs[2]:
        d = q[q.sideways_top25].sort_values("sideways_score", ascending=False)
        analyst([
            f"These 25 combine the lowest volatility, flattest drift, and tightest "
            f"20-day ranges — median annualized vol {d.vol_ann.median():.0%} vs "
            f"{q.vol_ann.median():.0%} for the whole universe.",
            "Sideways names are premium-selling and patience territory, not "
            "breakout territory."])
        show = add_name(d[["ticker", "vol_ann", "drift_ann", "range20_pct",
                           "sideways_pctl"]])
        st.dataframe(show.style.format({"vol_ann": "{:.0%}", "drift_ann": "{:+.1%}",
                                        "range20_pct": "{:.1%}", "px": "${:.2f}",
                                        "sideways_pctl": "{:.0f}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(show))

    with tabs[3]:
        d = q[q.blackswan_top25].sort_values("blackswan_score", ascending=False)
        analyst([
            f"Highest tail exposure: {t_label(d.iloc[0].ticker)} — "
            f"{d.iloc[0].jump_var_share:.0%} of its risk arrives as jumps."
            if len(d) else None,
            "This ranks EXPOSURE to violent gaps (both directions), built from jump "
            "share, jump size, and fat tails. Nobody can predict the swan itself — "
            "size positions in these names as if the gap will happen to you.",
            f"Median cash reserve among the 25: {fmt_money(d.get('cash', pd.Series(dtype=float)).median())} — "
            "cash is the survival buffer if a swan lands."])
        cols = ["ticker", "jump_var_share", "sigma_j", "kurtosis", "vol_ann",
                "blackswan_pctl"] + (["cash"] if "cash" in d.columns else [])
        show = add_name(d[cols])
        st.dataframe(show.style.format({"jump_var_share": "{:.0%}", "sigma_j": "{:.3f}",
                                        "kurtosis": "{:.1f}", "vol_ann": "{:.0%}",
                                        "px": "${:.2f}", "blackswan_pctl": "{:.0f}",
                                        "cash": "${:,.0f}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(show))


# ================================================================ portfolio 25
def page_p25():
    st.title("🎛️ Portfolio 25")
    port, curve = load("portfolio25"), load("portfolio25_curve")
    stats_f = STORE / "portfolio25_stats.json"
    if port.empty or not stats_f.exists():
        st.info("Run the data refresh to build Portfolio 25.")
        return
    s = json.loads(stats_f.read_text())
    st.caption("The 25 highest-composite-score liquid names (max 3 per industry), "
               "weighted inversely to volatility and scaled so the whole basket "
               "targets ~15% annual volatility — the risk calibration that maximizes "
               "the historical share of profitable periods. Paper only. Nothing makes "
               "profit certain: the drawdown number below is this portfolio's own history "
               "saying so.")
    analyst([
        f"Backtest (2 years, weekly): {s['pct_positive_weeks']:.0%} of weeks positive, "
        f"{s['pct_beat_spy_weeks']:.0%} beat the S&P; worst peak-to-trough "
        f"{s['max_drawdown']:.0%}.",
        f"Regression vs S&P: beta {s['beta']:.2f} (t={s['beta_t']:.1f}), annualized "
        f"alpha {s['alpha_ann']:+.1%} (t={s['alpha_t']:.1f}), R² {s['r_squared']:.2f}. "
        + ("Alpha is statistically meaningful." if abs(s.get("alpha_t") or 0) > 2
           else "Alpha is NOT statistically significant — treat it as noise until "
                "the forward test confirms it."),
        f"Risk calibration: {1 - s['cash_weight']:.0%} invested / "
        f"{s['cash_weight']:.0%} cash to hit the {s['vol_target']:.0%} vol target.",
        "⚠️ This basket was selected with today's information, so the backtest "
        "flatters it (look-ahead bias). The Paper Portfolio page is the honest test."])

    if not curve.empty:
        c = curve.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c["date"], y=c["portfolio"] * 100, name="Portfolio 25",
                                 line=dict(width=2.5, color=SERIES[0])))
        fig.add_trace(go.Scatter(x=c["date"], y=c["spy"] * 100, name="S&P 500",
                                 line=dict(width=2.5, color=SERIES[5])))
        fig.update_layout(title="Growth of 100 — weekly, 2 years (look-ahead backtest)")
        st.plotly_chart(style_fig(fig), width="stretch")

    hold = add_name(port.sort_values("weight", ascending=False))
    hold["industry"] = hold["industry"].map(ind_name)
    st.dataframe(hold.style.format({"price": "${:.2f}", "weight": "{:.1%}",
                                    "blend": "{:+.2f}", "score_1m": "{:+.2f}",
                                    "score_6m": "{:+.2f}", "vol_ann": "{:.0%}",
                                    "alpha_ann": "{:+.1%}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(hold))


# ================================================================ 8 · paper portfolio
def page_paper():
    st.title("🧾 Paper Portfolio")
    st.caption("The scoreboard's honest forward test. Every data Friday, the top 10 "
               "liquid names per horizon are frozen as a paper basket — equal weight, "
               "assumed filled at that Friday's close, zero costs, never edited "
               "afterwards. If the scoreboard has real predictive power, these baskets "
               "beat SPY over time; if they don't, believe the baskets, not the scores. "
               "No real money is involved.")
    bk, db = load("paper_baskets"), load("bars_daily")
    if bk.empty:
        st.info("No baskets frozen yet — they're created automatically by the data refresh.")
        return
    if "basket_type" not in bk.columns:
        bk["basket_type"], bk["label"] = "live", bk["as_of"]
    latest_close = db.sort_values("date").groupby("ticker")["c"].last()
    latest_date = db["date"].max()
    px = db.pivot_table(index="date", columns="ticker", values="c").sort_index()
    dates_idx = list(px.index)
    target_days = {"1d": 1, "2w": 10, "1m": 21, "6m": 126, "12m": 252, "proxy": 5}
    hname = {"1d": "Tomorrow", "2w": "2 weeks", "1m": "1 month",
             "6m": "6 months", "12m": "12 months", "proxy": "1 week (history)"}

    # ---- the year-long historical record (proxy baskets, 1-week horizon) ----
    hist = []
    for (as_of, h), g in bk[bk.basket_type == "proxy"].groupby(["as_of", "horizon"]):
        if as_of not in px.index:
            continue
        i = dates_idx.index(as_of)
        j = min(i + 5, len(dates_idx) - 1)
        if j == i:
            continue
        exit_d = dates_idx[j]
        rets = px.loc[exit_d].reindex(g.ticker).values / g.entry_price.values - 1
        spy = px.loc[exit_d, "SPY"] / g.spy_entry.iloc[0] - 1
        hist.append({"as_of": as_of, "label": g.label.iloc[0],
                     "excess": float(pd.Series(rets).mean() - spy),
                     "settled": exit_d < latest_date})
    if hist:
        hd = pd.DataFrame(hist).sort_values("as_of")
        settled = hd[hd.settled]
        if len(settled) > 10:
            mean_ex = settled["excess"].mean()
            t_stat = mean_ex / (settled["excess"].std() / np.sqrt(len(settled)))
            hit = (settled["excess"] > 0).mean()
            st.subheader("One year of history — does the picking rule work?")
            analyst([
                f"{len(settled)} historical baskets (every Tuesday and Friday for a "
                f"year, price-signals only): {hit:.0%} beat the S&P the following "
                f"week; average edge {mean_ex:+.2%}/week (t-statistic {t_stat:.1f}).",
                ("That t-statistic clears 2 — the edge is statistically real, though "
                 "costs would eat part of it." if abs(t_stat) > 2 else
                 "That t-statistic is below 2 — the historical edge is NOT "
                 "statistically distinguishable from luck. Respect that."),
                "These historical baskets use only price-derived signals (momentum, "
                "volatility) because sentiment and options data can't be reconstructed "
                "backwards — the live baskets going forward use the full score."])
            fig = go.Figure(go.Scatter(x=pd.to_datetime(settled["as_of"]),
                                       y=settled["excess"].cumsum(),
                                       mode="lines", line=dict(width=2, color=SERIES[0])))
            fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
            fig.update_layout(title="Cumulative weekly edge vs S&P (sum of excess returns)",
                              yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 340), width="stretch")

    summary = []
    for (as_of, h), g in bk.groupby(["as_of", "horizon"]):
        rets = latest_close.reindex(g.ticker).values / g.entry_price.values - 1
        basket = float(pd.Series(rets).mean())
        spy = float(latest_close["SPY"] / g.spy_entry.iloc[0] - 1)
        elapsed = max((pd.Timestamp(latest_date) - pd.Timestamp(as_of)).days, 0)
        summary.append({"as_of": as_of, "label": g.label.iloc[0], "horizon": h,
                        "type": g.basket_type.iloc[0], "days": elapsed,
                        "basket_ret": basket, "spy_ret": spy,
                        "excess": basket - spy})
    sm = (pd.DataFrame(summary).sort_values(["as_of", "horizon"], ascending=[False, True])
            .head(40))
    sm["horizon"] = sm["horizon"].map(hname)
    st.subheader("The record so far")
    st.dataframe(sm.style.format({"basket_ret": "{:+.2%}", "spy_ret": "{:+.2%}",
                                  "excess": "{:+.2%}"}),
                 width="stretch", hide_index=True,
                 column_config=col_cfg(sm, extra={
                     "as_of": ("Frozen on", "The Friday whose scoreboard picked this basket."),
                     "horizon": ("Horizon", "Which scoreboard horizon picked it."),
                     "days": ("Days elapsed", "Calendar days since the basket was frozen."),
                     "target_days": ("Days to verdict", "Calendar days until this basket's "
                                     "horizon is up and its result counts."),
                     "basket_ret": ("Basket", "Equal-weight return of the 10 names since freezing."),
                     "spy_ret": ("S&P 500", "SPY's return over the same period."),
                     "excess": ("Vs S&P", "Basket minus SPY. Positive = the scoreboard added value.")}))

    picks = sorted(bk["as_of"].unique(), reverse=True)
    c1, c2 = st.columns(2)
    sel_date = c1.selectbox("Basket date", picks)
    sel_h = c2.selectbox("Horizon", [h for h in target_days if
                                     ((bk.as_of == sel_date) & (bk.horizon == h)).any()],
                         format_func=lambda x: hname[x])
    g = bk[(bk.as_of == sel_date) & (bk.horizon == sel_h)].copy()
    g["latest"] = latest_close.reindex(g.ticker).values
    g["ret"] = g["latest"] / g["entry_price"] - 1
    hold = add_name(g[["ticker", "entry_price", "latest", "ret", "score"]]
                    .sort_values("ret", ascending=False))
    st.dataframe(hold.style.format({"entry_price": "${:.2f}", "latest": "${:.2f}",
                                    "ret": "{:+.2%}", "score": "{:+.2f}"}),
                 width="stretch", hide_index=True,
                 column_config=col_cfg(hold, extra={
                     "entry_price": ("Entry", "Close price on the Friday the basket was frozen."),
                     "latest": ("Latest", f"Most recent close ({latest_date})."),
                     "ret": ("Return", "Change since the basket was frozen.")}))
    st.caption("Fills are assumed at the freezing Friday's close with zero costs — "
               "real-world results would be slightly worse. Judge each basket only "
               "after its horizon is up.")


# ================================================================ 9 · institutions
def page_institutions():
    st.title("🏛️ Institutions & Insiders")
    inst, ins = load("inst_13f"), load("insiders")

    st.subheader("Institutional panel — 13F quarterly holdings")
    st.caption("Holdings of ~13 giant institutions (Vanguard, BlackRock, Berkshire, "
               "Citadel, Renaissance…) from their quarterly SEC 13F filings, summed per "
               "stock. This is a large panel, not every institution, and 13F data lags "
               "up to 45 days — it shows positioning, not today's trades.")
    if inst.empty:
        st.info("13F data not fetched yet — run `python pipeline/sec_refresh.py --part 13f`.")
    else:
        qs = sorted(inst["report_date"].unique())
        if len(qs) >= 2:
            latest, prior = qs[-1], qs[-2]
            piv = (inst.groupby(["ticker", "report_date"])["shares"].sum()
                       .unstack("report_date"))
            piv = piv[[prior, latest]].dropna()
            piv["chg"] = piv[latest] / piv[prior] - 1
            piv = piv[piv[prior] > 100_000]  # ignore dust positions
            movers = pd.concat([piv.nlargest(10, "chg"), piv.nsmallest(10, "chg")])
            movers = movers.sort_values("chg")
            fig = go.Figure(go.Bar(
                x=movers["chg"], y=[t_label(t) for t in movers.index], orientation="h",
                marker_color=["#e66767" if v < 0 else "#199e70" for v in movers["chg"]],
                text=[f"{v:+.0%}" for v in movers["chg"]], textposition="outside"))
            fig.update_layout(title=f"Biggest panel position changes: {prior} → {latest}",
                              xaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 560), width="stretch")
            tbl = piv.sort_values("chg", ascending=False).reset_index()
            tbl = add_name(tbl)
            with st.expander("Full panel table"):
                st.dataframe(tbl.style.format({prior: "{:,.0f}", latest: "{:,.0f}",
                                               "chg": "{:+.1%}"}),
                             width="stretch", hide_index=True,
                             column_config=col_cfg(tbl, extra={
                                 prior: (f"Shares {prior}", "Panel shares held at the prior quarter-end."),
                                 latest: (f"Shares {latest}", "Panel shares held at the latest reported quarter-end.")}))
        else:
            st.write("Only one quarter of panel data available so far.")

    st.subheader("Insider activity — Form 4, last 90 days")
    st.caption("Open-market buys (code P) vs sells (code S) by officers and directors, "
               "from SEC Form 4 filings. Insider buying with their own money is one of "
               "the stronger public signals; routine selling is usually noise.")
    if ins.empty:
        st.info("Insider data not fetched yet — run `python pipeline/sec_refresh.py --part insiders`.")
        return
    act = ins[(ins.buy_value > 0) | (ins.sell_value > 0)].copy()
    act = act.reindex(act["net_value"].abs().sort_values(ascending=False).index).head(20)
    act = act.sort_values("net_value")
    fig = go.Figure(go.Bar(
        x=act["net_value"], y=[t_label(t) for t in act["ticker"]], orientation="h",
        marker_color=["#e66767" if v < 0 else "#199e70" for v in act["net_value"]],
        text=[f"${v / 1e6:+,.1f}M" for v in act["net_value"]], textposition="outside"))
    fig.update_layout(title="Net insider dollars (buys − sells), 90 days")
    st.plotly_chart(style_fig(fig, 560), width="stretch")
    full = add_name(ins.sort_values("net_value", ascending=False))
    with st.expander("All insider activity"):
        st.dataframe(full.style.format({"buy_shares": "{:,.0f}", "sell_shares": "{:,.0f}",
                                        "buy_value": "${:,.0f}", "sell_value": "${:,.0f}",
                                        "net_value": "${:+,.0f}"}),
                     width="stretch", hide_index=True, column_config=col_cfg(full))


# ================================================================ 9 · lipstick
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

    members = add_name(load("daily_features")[lambda d: d.ticker.isin(lips)][
        ["ticker", "price", "mom_1m", "mom_3m", "mom_6m", "beta"]])
    st.dataframe(members.style.format({"price": "${:.2f}", "mom_1m": "{:+.1%}",
                                       "mom_3m": "{:+.1%}", "mom_6m": "{:+.1%}",
                                       "beta": "{:.2f}"}),
                 width="stretch", hide_index=True, column_config=col_cfg(members))
    rev = load("lipstick_revenue")
    if not rev.empty:
        st.subheader("Quarterly revenue — straight from SEC filings")
        st.caption("Year-over-year revenue growth per company (XBRL data from their "
                   "10-Q/10-K filings). Beauty revenue holding up while the economy "
                   "wobbles is the actual lipstick-index signal.")
        rev = rev.copy()
        rev["period_end"] = pd.to_datetime(rev["period_end"])
        fig = go.Figure()
        for i, (t, g) in enumerate(rev.groupby("ticker")):
            g = g.sort_values("period_end").tail(12)
            yoy = g.set_index("period_end")["revenue"].pct_change(4).dropna()
            if len(yoy):
                fig.add_trace(go.Scatter(x=yoy.index, y=yoy.values, name=t_label(t),
                                         mode="lines+markers",
                                         line=dict(width=2, color=SERIES[i % len(SERIES)]),
                                         marker=dict(size=7)))
        fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
        fig.update_layout(title="Revenue growth, year over year", yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig), width="stretch")


# ================================================================ nav
pages = st.navigation([
    st.Page(page_command_center, title="Friday Command Center", icon="📈", default=True),
    st.Page(page_industry, title="Industry Explorer", icon="🏭"),
    st.Page(page_ticker, title="Ticker Deep Dive", icon="🔬"),
    st.Page(page_options, title="Options Flow & GEX", icon="🎯"),
    st.Page(page_models, title="Models Lab", icon="🧪"),
    st.Page(page_sentiment, title="Sentiment & Events", icon="📰"),
    st.Page(page_scoreboard, title="Scoreboard", icon="🏆"),
    st.Page(page_screens, title="Screens", icon="🔍"),
    st.Page(page_p25, title="Portfolio 25", icon="🎛️"),
    st.Page(page_paper, title="Paper Portfolio", icon="🧾"),
    st.Page(page_institutions, title="Institutions & Insiders", icon="🏛️"),
    st.Page(page_lipstick, title="Lipstick Index", icon="💄"),
])
pages.run()
