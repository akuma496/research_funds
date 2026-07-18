@echo off
rem Options snapshot collector launcher — used by Task Scheduler and for manual runs.
rem Portable: resolves the repo from this file's own location.
cd /d "%~dp0.."
python collector\options_snapshot.py %*
