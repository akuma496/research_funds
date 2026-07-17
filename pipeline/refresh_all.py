"""Run the full pipeline in order: parse raw -> compute -> options -> scoreboard.

Use after new data arrives (e.g., every evening, or after Friday's captures):
    python pipeline/refresh_all.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("parse_raw.py", "compute.py", "options_analytics.py", "scoreboard.py"):
    print(f"=== {script} ===", flush=True)
    r = subprocess.run([sys.executable, str(HERE / script)])
    if r.returncode:
        sys.exit(f"{script} failed with exit {r.returncode}")
print("REFRESH COMPLETE")
