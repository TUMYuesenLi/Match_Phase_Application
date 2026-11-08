"""Shared file locations and the six phase names used in the CSV files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PHASES = [
    "Build Up",
    "Progression",
    "Counter attack",
    "Maintenance",
    "Sustained Threat",
    "Finishing",
]


def output_folder(name):
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder
