@echo off
rem One-click launcher for the OPEX Friday dashboard.
cd /d "C:\Users\adity\Documents\research_funds"
start "" http://localhost:8501
"C:\Users\adity\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app\dashboard.py
