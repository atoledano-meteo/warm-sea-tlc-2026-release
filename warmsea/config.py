"""Folders set in config.yaml (relative paths are relative to the repository)."""

from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
_c = yaml.safe_load((REPO / 'config.yaml').read_text())
DATASET, ERA5, OBS, OUTPUT = ((REPO / _c[k]).resolve() for k in ('dataset', 'era5', 'observations', 'output'))
