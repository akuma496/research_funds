"""One-command fresh setup: install deps, backfill a year of data, build the store.

For a new machine / new user:
  1. Install Python 3.12+ from python.org (check "Add to PATH").
  2. Create a free account at alpaca.markets and generate API keys.
  3. Create a file named `.env` in this folder containing:
         ALPACA_API_KEY=your_key_id
         ALPACA_SECRET_KEY=your_secret
  4. Run:  python setup_fresh.py
  5. Register the automatic daily captures (optional but recommended):
         powershell -ExecutionPolicy Bypass -File collector\register_schedule.ps1
  6. Open the dashboard:  double-click launch_dashboard.bat

The backfill downloads ~100 MB (52 Fridays of minute bars + 2 years of daily
bars for ~219 tickers) and takes ~10 minutes on the free API tier. Options
chain history cannot be backfilled — it accrues from the first day the
scheduled captures run.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(desc, args, check=True):
    print(f"\n=== {desc} ===", flush=True)
    r = subprocess.run(args, cwd=ROOT)
    if check and r.returncode:
        sys.exit(f"FAILED: {desc} (exit {r.returncode})")


def main():
    if sys.version_info < (3, 11):
        sys.exit("Python 3.11+ required.")
    env = ROOT / ".env"
    if not env.exists():
        sys.exit("Create a .env file first — see the instructions at the top of this script.")
    txt = env.read_text()
    if "ALPACA_API_KEY" not in txt or "ALPACA_SECRET_KEY" not in txt:
        sys.exit(".env must define ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    run("Installing Python packages", [sys.executable, "-m", "pip", "install",
                                       "-q", "-r", "requirements.txt"])
    run("Testing Alpaca connection (3 tickers)",
        [sys.executable, "collector/options_snapshot.py", "--test"])
    run("Backfilling 52 Fridays of minute bars + 2y daily bars (~10 min)",
        [sys.executable, "collector/equity_backfill.py"])
    run("Taking a first full options snapshot (~5 min)",
        [sys.executable, "collector/options_snapshot.py"])
    run("Building the data store and running all models",
        [sys.executable, "pipeline/refresh_all.py"])

    print("""
SETUP COMPLETE.
  Open the dashboard:      double-click launch_dashboard.bat
  Automate daily captures: powershell -ExecutionPolicy Bypass -File collector\\register_schedule.ps1
""")


if __name__ == "__main__":
    main()
