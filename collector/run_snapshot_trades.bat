@echo off
rem Trade-tape collector launcher — used by Task Scheduler (15:25) and manually.
cd /d "%~dp0.."
python collector\options_trades.py %*
