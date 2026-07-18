@echo off
rem Nightly data refresh — fetches the day's bars and trade tape, rebuilds the store.
cd /d "%~dp0.."
python collector\equity_backfill.py --include-today --fridays 52
python collector\options_trades.py
python pipeline\refresh_all.py
