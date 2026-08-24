"""Root re-export / entrypoint for Streamlit dashboard"""
import os
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Execute dashboard.py content
dashboard_file = Path(__file__).parent / "dashboard.py"
with open(dashboard_file, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, dashboard_file, "exec"))
