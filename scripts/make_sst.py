"""SST states of the experiments, on the WRF grids.

CURR is the Copernicus Mediterranean L4 subskin SST (hourly, with the diurnal cycle) interpolated bilinearly to the
marine cells of each domain, with the nearest valid pixel within 25 km where the interpolation has no value
(`current`). It is archived in the dataset (sst/curr.nc). From it:

  CLIM    CURR minus the 1991-2020 anomaly of the daily Copernicus UHR foundation SST (`climatological`)
  +2K     CURR + 2 K (`plus2k`)

on marine cells only (land and lakes unchanged). WRF reads these fields through wrflowinp (sst_update = 1).

    python scripts/make_sst.py --anomaly /path/to/SST_MED_SSTA_L4_NRT_OBSERVATIONS_010_004_d --out /path/to/folder

The anomaly files are the daily product of 2 to 5 October 2026 (Copernicus Marine, SST_MED_SSTA_L4_NRT_OBSERVATIONS_010_004_d).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.config import DATASET

FILL_KM, TAPER_KM = 30.0, 150.0


def current(lat, lon, field, target_lat, target_lon, marine):
    """Subskin SST (lat, lon, field ascending) on the marine cells: bilinear, else the nearest valid pixel within 25 km."""
    valid = np.isfinite(field) & (field > 250) & (field < 320)
    sy, sx = np.nonzero(valid)
    k = np.array([111.2, 111.2 * np.cos(np.radians(40))])
    tree = cKDTree(np.column_stack((lat[sy], lon[sx])) * k)
    ty, tx = np.nonzero(marine)
    pos = np.column_stack((target_lat[ty, tx], target_lon[ty, tx]))
    dist, near = tree.query(pos * k)
    v = RegularGridInterpolator((lat, lon), field, bounds_error=False, fill_value=np.nan)(pos)
    miss = ~np.isfinite(v)
    v[miss] = field[sy[near[miss]], sx[near[miss]]]
    out = np.zeros(target_lat.shape, 'f4')
    ok = dist <= 25
    out[ty[ok], tx[ok]] = v[ok]
    return out


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) * 6371.0


def anomaly_on_grid(lat2d, lon2d, marine, alat, alon, a):
    """Mean of the ~1 km anomaly pixels whose nearest marine cell is each cell; cells without pixels take the nearest
    covered value within 30 km, tapered linearly to 0 over the next 150 km (outside the Mediterranean product)."""
    tree = cKDTree(_xyz(lat2d[marine], lon2d[marine]))
    LA, LO = np.meshgrid(alat, alon, indexing='ij')
    box = (
        (LA >= lat2d.min() - 0.2)
        & (LA <= lat2d.max() + 0.2)
        & (LO >= lon2d.min() - 0.2)
        & (LO <= lon2d.max() + 0.2)
        & np.isfinite(a)
    )
    dist, idx = tree.query(_xyz(LA[box], LO[box]))
    keep = dist < 10.0
    n = marine.sum()
    s = np.bincount(idx[keep], a[box][keep], n)
    c = np.bincount(idx[keep], None, n)
    val = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    covered = np.isfinite(val)
    pts = _xyz(lat2d[marine], lon2d[marine])
    d2, j = cKDTree(pts[covered]).query(pts[~covered])
    nearv = val[covered][j]
    val[~covered] = np.where(d2 <= FILL_KM, nearv, nearv * np.clip(1 - (d2 - FILL_KM) / TAPER_KM, 0, 1))
    field = np.zeros(lat2d.shape, 'f4')
    field[marine] = val
    return field


def climatological(curr, times, lat2d, lon2d, marine, anomaly_dir):
    """CURR minus the daily anomaly (00 UTC files) interpolated linearly in time to each hour."""
    days = {}
    for f in sorted(anomaly_dir.glob('*.nc')):
        with xr.open_dataset(f) as d:
            days[pd.Timestamp(f.name[:8])] = anomaly_on_grid(
                lat2d, lon2d, marine, d.lat.values, d.lon.values, d.sst_anomaly.isel(time=0).values.astype('f4')
            )
    keys = sorted(days)
    out = curr.astype('f8').copy()
    for i, t in enumerate(pd.to_datetime(times)):
        k = min(max(j for j, x in enumerate(keys) if x <= t), len(keys) - 2)
        w = (t - keys[k]) / (keys[k + 1] - keys[k])
        out[i] = np.where(marine, curr[i] - ((1 - w) * days[keys[k]] + w * days[keys[k + 1]]), curr[i])
    return out


def plus2k(curr, marine, delta=2.0):
    return np.where(marine[None], curr.astype('f8') + delta, curr)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--anomaly', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    o = a.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    G = xr.open_dataset(DATASET / 'grid.nc')
    C = xr.open_dataset(DATASET / 'sst/curr.nc')
    res = {'clim': {}, 'plus2k': {}}
    for d in ('d01', 'd02'):
        marine = (G[f'landmask_{d}'].values < 0.5) & (G[f'lakemask_{d}'].values < 0.5)
        curr = C[f'sst_{d}'].values
        res['clim'][f'sst_{d}'] = (
            ('time', f'y_{d}', f'x_{d}'),
            climatological(
                curr, C.time.values, G[f'xlat_{d}'].values, G[f'xlong_{d}'].values, marine, o.anomaly
            ).astype('f4'),
        )
        res['plus2k'][f'sst_{d}'] = (('time', f'y_{d}', f'x_{d}'), plus2k(curr, marine).astype('f4'))
    for k, v in res.items():
        xr.Dataset(v, coords={'time': C.time.values}, attrs={'units': 'K'}).to_netcdf(o.out / f'{k}.nc')
        print('written', o.out / f'{k}.nc')
