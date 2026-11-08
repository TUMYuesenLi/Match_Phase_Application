"""Run statistics, spatial estimation and full software-output export."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for script in [
    "01_phase_profiles.py",
    "02_team_strength.py",
    "03_phase_behaviours.py",
    "04_success_models.py",
    "05_case_study.py",
    "06_spatial_density.py",
    "07_compile_statistical_output.py",
]:
    print("Running", script, flush=True)
    subprocess.run([sys.executable, str(root / "code" / script)], check=True)
