"""Latent heat flux and 0-1 km equivalent potential temperature of the low-level maritime flow (Text S3), from the
full WRF output (not archived). Writes the dataset's diagnostics/maritime_flow.csv.

    python scripts/maritime_flow.py --runs /path/to/runs --out /path/to/diagnostics

Both are means over the sea cells (no lakes) of 40.5-42.0 N, 1.5-4.0 E on d02, from 1200 UTC 3 to 0600 UTC 4 October:
LH from every 15 min output, and theta-e (MetPy, Bolton 1980) over the model levels below 1 km above ground at the
full hours.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from metpy import calc as mc
from metpy.units import units

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.runs import SST_RUNS

RUNS = [r for r in SST_RUNS if not r.startswith('tmynn')]
HOURS = pd.date_range('2026-10-03 12:00', '2026-10-04 06:00', freq='h')
BOX = (40.5, 42.0, 1.5, 4.0)


def theta_e(p, tk, r):
    """Equivalent potential temperature (K) from pressure (Pa), temperature (K) and mixing ratio (kg/kg)."""
    p = np.asarray(p) * units.Pa
    td = mc.dewpoint(mc.vapor_pressure(p, np.asarray(r) * units.dimensionless))
    return mc.equivalent_potential_temperature(p, np.asarray(tk) * units.K, td).magnitude


def run_series(run_dir):
    rows = []
    for t in HOURS:
        with Dataset(run_dir / 'wrf' / f'wrfout_d02_{t:%Y-%m-%d_%H}:00:00') as d:
            if t == HOURS[0]:
                lat, lon = np.asarray(d['XLAT'][0]), np.asarray(d['XLONG'][0])
                sea = (np.asarray(d['LANDMASK'][0]) <= 0.5) & (np.asarray(d['LAKEMASK'][0]) <= 0.5)
                sea &= (lat >= BOX[0]) & (lat <= BOX[1]) & (lon >= BOX[2]) & (lon <= BOX[3])
                hgt = np.asarray(d['HGT'][0], 'f8')[sea]
            nf = 1 if t == HOURS[-1] else d.dimensions['Time'].size
            for k in range(nf):
                lh = np.asarray(d['LH'][k], 'f8')[sea].mean()
                row = {'time': t + pd.Timedelta(minutes=60 // d.dimensions['Time'].size * k), 'latent_heat_W_m2': lh}
                if k == 0:
                    g = lambda v, s=sea: np.asarray(d[v][0], 'f8')[:, s]
                    p = g('P') + g('PB')
                    tk = (g('T') + 300) * (p / 1e5) ** 0.2857
                    ph = (np.asarray(d['PH'][0], 'f8')[:, sea] + np.asarray(d['PHB'][0], 'f8')[:, sea]) / 9.81 - hgt
                    z = 0.5 * (ph[1:] + ph[:-1])
                    row['theta_e_0_1km_K'] = float(theta_e(p, tk, g('QVAPOR'))[z < 1000].mean())
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--runs', type=Path, required=True, help='one folder per run (names in warmsea/runs.py)')
    a.add_argument('--out', type=Path, required=True)
    o = a.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    t = pd.concat([run_series(o.runs / r).assign(run=r) for r in RUNS])
    t[['run', 'time', 'latent_heat_W_m2', 'theta_e_0_1km_K']].to_csv(
        o.out / 'maritime_flow.csv', index=False, float_format='%.4f'
    )
