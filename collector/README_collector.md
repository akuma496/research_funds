# Options snapshot collector

Captures the full options chain (quotes, latest trades, IV, greeks) for all ~219
tickers in `data/universe.csv` from Alpaca's free indicative feed. Raw responses are
stored under `data/raw/options/<YYYY-MM-DD>/<window>/<TICKER>.json.gz`; the dashboard
pipeline parses them later. Options chains cannot be backfilled, so this history
starts accruing the day the collector first runs.

## Schedule (Windows Task Scheduler, local Central time, Mon–Fri)

| Task name | Window | Time |
|---|---|---|
| ResearchFunds Options Snapshot W1 | pre-open baseline (fresh overnight OI) | 08:15 |
| ResearchFunds Options Snapshot W2 | midday | 10:30 |
| ResearchFunds Options Snapshot W3 | power hour | 14:45 |
| ResearchFunds Options Snapshot W4 | post-close state | 15:10 |

Tasks are set to wake the machine from sleep and to run late if the scheduled time
was missed (the actual fetch timestamp is recorded inside every file, so a late run
is labeled honestly). **The laptop must be powered on (or asleep, not shut down).**

## Manual use

```
collector\run_snapshot.bat --test          # 3 tickers, prints contract counts
collector\run_snapshot.bat                 # full universe, window inferred from clock
collector\run_snapshot.bat --window W3     # full universe, explicit window label
```

Activity log: `data/raw/options/collector.log`.

## Requirements

- `.env` at repo root with `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (never commit it).
- Python 3.x — stdlib only, no packages needed.

## Removing the schedule

```powershell
Get-ScheduledTask -TaskName 'ResearchFunds Options Snapshot *' | Unregister-ScheduledTask -Confirm:$false
```
