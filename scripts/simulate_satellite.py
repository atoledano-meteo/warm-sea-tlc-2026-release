"""MTG-I1 FCI 10.5 um brightness temperature simulated from the WRF 3 km columns with RTTOV 14.1 (Figures 4 and S4).

RTTOV is distributed by the NWP SAF under its own licence and is not included. Set RTTOV_DIR to the RTTOV 14.1 folder
(with the Python wrapper built). Coefficients: rtcoef_mtg_1_fci_o3co2.dat (54 levels) and the hydrotable
rttov_hydrotable_mtg_1_fci.dat (all-sky).

Per column, at the hourly output time:
  layers   the WRF mass levels (temperature from theta, pressure P+PB), top first; half levels midway between them,
           bounded by P_TOP and PSFC (PSFC raised to half a layer below the lowest level where it is lower)
  gases    water vapour as specific humidity; O3 and CO2 from the RTTOV reference profile
  clouds   liquid QCLOUD (parameterised effective diameter); ice QICE+QSNOW+QGRAUP (Baran ice); cloud fraction 0.999999
           where liquid + ice > 1e-6 kg/kg; rain left out
  surface  TSK, T2, Q2, U10, V10, land or sea from XLAND, terrain height; RTTOV emissivity models
  geometry MTG-I1 at 0 E

    RTTOV_DIR=/path/to/rttov141 python scripts/simulate_satellite.py --runs /path/to/runs --out /path/to/folder
writes <run>_<yyyymmddhh>.npz (lon, lat, ir105, wv63 in degC) for the map times of Figures 4 and S4.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.runs import MAP_TIMES
from warmsea.wrf import specific_humidity, temperature

CHANNELS = {10: 'wv63', 14: 'ir105'}
R_E, R_SAT = 6371.0, 42164.0
BOX = (-1.0, 7.0, 37.9, 44.1)  # lon0, lon1, lat0, lat1


def sat_angles(lat, lon, lon_sat=0.0):
    """Zenith and azimuth (deg, clockwise from north) of a geostationary satellite."""
    la, lo = np.radians(lat), np.radians(lon - lon_sat)
    p = R_E * np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    v = np.array([R_SAT, 0, 0])[:, None] - p
    up = p / R_E
    east = np.stack([-np.sin(lo), np.cos(lo), 0 * lo])
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    vn = v / np.linalg.norm(v, axis=0)
    return np.degrees(np.arccos((vn * up).sum(0))), np.degrees(
        np.arctan2((vn * east).sum(0), (vn * north).sum(0))
    ) % 360


def load(rt, nthreads=8):
    sys.path.insert(0, str(rt / 'wrapper'))
    import pyrttov

    r = pyrttov.Rttov()
    r.FileCoef = str(rt / 'rtcoef_rttov14/rttov13pred54L/rtcoef_mtg_1_fci_o3co2.dat')
    r.FileHydrotable = str(rt / 'rtcoef_rttov14/hydrotable_visir/rttov_hydrotable_mtg_1_fci.dat')
    r.Options.Nthreads = nthreads
    r.Options.VerboseWrapper = False
    r.Options.Hydrometeors = True
    r.loadInst(list(CHANNELS))
    return r, pyrttov


def columns(path, box):
    with Dataset(path) as d:
        lat, lon = np.asarray(d['XLAT'][0], 'f8'), np.asarray(d['XLONG'][0], 'f8')
        m = (lon >= box[0]) & (lon <= box[1]) & (lat >= box[2]) & (lat <= box[3])
        g = lambda v: np.asarray(d[v][0], 'f8')[..., m]
        p = g('P') + g('PB')
        th = g('T') + 300
        qv = np.maximum(g('QVAPOR'), 1e-9)  # RTTOV rejects the tiny negative values
        return dict(
            lat=lat[m],
            lon=lon[m],
            p=p / 100,
            t=temperature(p, th),
            q=specific_humidity(qv),
            liq=np.maximum(g('QCLOUD'), 0),
            ice=np.maximum(g('QICE') + g('QSNOW') + g('QGRAUP'), 0),
            psfc=g('PSFC') / 100,
            ptop=float(d['P_TOP'][0]) / 100,
            tsk=g('TSK'),
            t2=g('T2'),
            q2=g('Q2'),
            u10=g('U10'),
            v10=g('V10'),
            sea=g('XLAND') > 1.5,
            hgt=g('HGT'),
            mask=m,
            lat2=lat,
            lon2=lon,
        )


def simulate(r, pyrttov, c, t, chunk=8000):
    nl, n = c['p'].shape
    out = np.full((n, len(CHANNELS)), np.nan)
    flip = lambda a: np.ascontiguousarray(a[::-1].T)
    P, T, Q, L, I = (flip(c[v]) for v in ('p', 't', 'q', 'liq', 'ice'))
    half = np.empty((n, nl + 1))
    half[:, 0] = c['ptop']
    half[:, 1:-1] = 0.5 * (P[:, 1:] + P[:, :-1])
    half[:, -1] = np.maximum(c['psfc'], P[:, -1] + 0.5 * (P[:, -1] - P[:, -2]))
    zen, azi = sat_angles(c['lat'], c['lon'])
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        m = e - s
        sl = slice(s, e)
        pr = pyrttov.Profiles(m, nl + 1, 1)
        pr.GasUnits = 1
        pr.MmrHydro = True
        pr.PHalf = half[sl]
        pr.P = P[sl]
        pr.T = T[sl]
        pr.Q = Q[sl]
        pr.Hydro6 = L[sl]
        pr.Hydro8 = I[sl]
        pr.HydroFrac1 = np.where(L[sl] + I[sl] > 1e-6, 0.999999, 0.0)
        pr.ClwdeParam = np.ones(m, np.int32)
        pr.Angles = np.column_stack([zen[sl], azi[sl], np.zeros(m), np.zeros(m)])
        pr.SurfGeom = np.column_stack([c['lat'][sl], c['lon'][sl], c['hgt'][sl] / 1000])
        pr.SurfType = np.column_stack([np.where(c['sea'][sl], 1, 0), np.where(c['sea'][sl], 1, 0)]).astype(np.int32)[
            :, None, :
        ]
        pr.Skin = np.column_stack(
            [c['tsk'][sl], np.full(m, 35.0), np.zeros(m), np.zeros(m), np.tile([3.0, 5.0, 15.0, 0.1, 0.3], (m, 1))]
        )[:, None, :]
        pr.NearSurface = np.column_stack(
            [c['t2'][sl], c['q2'][sl] / (1 + c['q2'][sl]), c['u10'][sl], c['v10'][sl], np.full(m, 1e5)]
        )[:, None, :]
        pr.DateTimes = np.tile([t.year, t.month, t.day, t.hour, 15, 0], (m, 1)).astype(np.int32)
        r.Profiles = pr
        r.runDirect()
        out[sl] = r.BtRefl
    return {name: out[:, i] for i, name in enumerate(CHANNELS.values())}


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--runs', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    o = a.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    r, pyrttov = load(Path(os.environ['RTTOV_DIR']))
    for run, t in MAP_TIMES.items():
        t = pd.Timestamp(t)
        c = columns(o.runs / run / 'wrf' / f'wrfout_d02_{t:%Y-%m-%d_%H}:00:00', BOX)
        bt = simulate(r, pyrttov, c, t)
        f = {}
        for k, v in bt.items():
            x = np.full(c['lat2'].shape, np.nan)
            x[c['mask']] = v - 273.15
            f[k] = x
        np.savez_compressed(o.out / f'{run}_{t:%Y%m%d%H}.npz', lon=c['lon2'], lat=c['lat2'], **f)
