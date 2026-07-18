# OPEX Friday Market Microstructure Dashboard

Personal research dashboard studying market behavior at four intraday windows
(pre-open, midday, power hour, after hours — US Central) across 15 thematic
industries, with options flow + block tape, jump models, SEC institutional and
insider data, sentiment, and a 5-horizon statistical scoreboard.
Full design: [SPEC.md](SPEC.md).

## Fresh setup (new machine or new user)

1. Install Python 3.12+ (python.org, check "Add to PATH").
2. Free account at alpaca.markets → generate API keys.
3. Create `.env` in this folder: `ALPACA_API_KEY=...` and `ALPACA_SECRET_KEY=...`
4. `python setup_fresh.py` — installs packages, backfills a year of data (~10 min), builds everything.
5. Optional automation: `powershell -ExecutionPolicy Bypass -File collector\register_schedule.ps1`
6. Optional SEC data (13F/insiders/revenue, ~15 min): `python pipeline\sec_refresh.py`

## Daily use

**Open the dashboard:** double-click `launch_dashboard.bat` → browser opens at
http://localhost:8501. Close the black console window to stop it.

**Refresh data** (evenings, or after a Friday):

```
collector\run_refresh.bat        (fetches the day's bars + trade tape, rebuilds everything)
```

**Refresh SEC data** (weekly is plenty): `python pipeline\sec_refresh.py`

Options chain snapshots are captured automatically 4× every weekday by
Windows Task Scheduler (see [collector/README_collector.md](collector/README_collector.md)) —
the laptop just needs to be on or asleep.

## Layout

| Path | What |
|---|---|
| `collector/` | Data capture: options snapshots (scheduled), trade tape, equity backfill |
| `pipeline/` | parse_raw → compute (metrics + Merton/Poisson/Hawkes) → options_analytics → options_blocks → scoreboard; sec_refresh (13F/insiders/revenue) |
| `app/dashboard.py` | 9-page Streamlit app |
| `data/raw/` | Raw API responses (gzipped JSON, git-ignored) |
| `data/store/` | Parquet tables + sentiment.json the app reads (git-ignored) |
| `data/universe.csv` | The 219-ticker universe, tagged by industry |

Requires: `.env` with `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (never commit),
Python 3.14 with streamlit, plotly, pandas, numpy, scipy, pyarrow.

> The Scoreboard is a statistical screen, not financial advice — it ranks model
> evidence; decisions are yours.
