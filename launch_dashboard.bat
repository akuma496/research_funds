@echo off
rem One-click launcher for the OPEX Friday dashboard.
cd /d "%~dp0"
start "" http://localhost:8501
python -m streamlit run app\dashboard.py
