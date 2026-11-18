"""Configuration loading and shared constants."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "experiment_config.yaml"

SCENARIOS = ["clean", "realistic", "stressed"]
SCENARIO_LABEL = {"clean": "A_clean", "realistic": "B_realistic", "stressed": "C_stressed"}

INDICATOR_FIELDS = [
    "market_arrivals_tonnes", "estimated_demand_index", "rainfall_mm", "temperature_c",
    "drought_index", "fuel_cost_index", "transport_cost_kes_tonne", "regional_supply_index",
]
# indicators whose records can arrive one week late (supply/demand/logistics sources)
DELAYABLE_FIELDS = ["market_arrivals_tonnes", "estimated_demand_index",
                    "transport_cost_kes_tonne", "regional_supply_index"]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def market_ids(cfg): return list(cfg["markets"].keys())
def crop_ids(cfg): return list(cfg["crops"].keys())


def data_dir(cfg=None) -> Path:
    d = ROOT / "data"; d.mkdir(exist_ok=True); return d

def results_dir() -> Path:
    d = ROOT / "results"; d.mkdir(exist_ok=True); return d

def figures_dir() -> Path:
    d = ROOT / "figures"; d.mkdir(exist_ok=True); return d

def reports_dir() -> Path:
    d = ROOT / "reports"; d.mkdir(exist_ok=True); return d
