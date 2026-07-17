@echo off
rem Options snapshot collector launcher — used by Task Scheduler and for manual runs.
cd /d "C:\Users\adity\Documents\research_funds"
"C:\Users\adity\AppData\Local\Python\pythoncore-3.14-64\python.exe" collector\options_snapshot.py %*
