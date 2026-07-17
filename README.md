# OPEX Friday Market Microstructure Dashboard

Personal research dashboard studying market behavior at four intraday windows
(pre-open, midday, power hour, after hours — US Central) across 15 thematic
industries, with options flow, jump models, sentiment, and a 5-horizon
statistical scoreboard. Full design: [SPEC.md](SPEC.md).

## Daily use

**Open the dashboard:** double-click `launch_dashboard.bat` → browser opens at
http://localhost:8501. Close the black console window to stop it.

**Refresh data** (evenings, or after a Friday):

```
python pipeline\refresh_all.py
```

Options chain snapshots are captured automatically 4× every weekday by
Windows Task Scheduler (see [collector/README_collector.md](collector/README_collector.md)) —
the laptop just needs to be on or asleep.

## Layout

| Path | What |
|---|---|
| `collector/` | Data capture: options snapshots (scheduled), equity backfill |
| `pipeline/` | parse_raw → compute (metrics + Merton/Poisson/Hawkes) → options_analytics → scoreboard |
| `app/dashboard.py` | 8-page Streamlit app |
| `data/raw/` | Raw API responses (gzipped JSON, git-ignored) |
| `data/store/` | Parquet tables + sentiment.json the app reads (git-ignored) |
| `data/universe.csv` | The 219-ticker universe, tagged by industry |

Requires: `.env` with `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (never commit),
Python 3.14 with streamlit, plotly, pandas, numpy, scipy, pyarrow.

> The Scoreboard is a statistical screen, not financial advice — it ranks model
> evidence; decisions are yours.
