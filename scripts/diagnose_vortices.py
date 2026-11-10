"""MCV and surface-cyclone diagnostics of every run, from the full WRF output (not archived). Writes the tables in the
dataset's diagnostics/ folder.

    python scripts/diagnose_vortices.py --runs /path/to/runs --cache /path/to/cache --out /path/to/diagnostics [steps]

steps: extract mcv cyclones profiles (default: all). `extract` needs wrf-python (we used 1.3.4.1 with NumPy 1.26 in a
separate environment); the other steps need MetPy and CPyS 1.0.0. --runs holds one folder per run (paper names, see
warmsea/runs.py) with wrf/wrfout_d02_* files.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.runs import SST_RUNS
from warmsea import vortex as V
from warmsea.wrf import virtual_temperature

TIMES = pd.date_range('2026-10-03 18:00', '2026-10-05 00:00', freq='h')
P_LEVELS = np.array([925, 900, 850, 800, 750, 700, 650, 600, 575, 550, 525, 500, 450, 400, 350, 300, 250, 200.0])
Z_LEVELS = np.unique(np.r_[300.0, 500.0, 750.0, np.arange(1000.0, 6001.0, 250.0), np.arange(7000.0, 12001.0, 1000.0)])


def wrfout(o, run, t):
    return o.runs / run / 'wrf' / f'wrfout_d02_{t:%Y-%m-%d_%H}:00:00'


def extract(job):
    """Heights, virtual temperature and winds on pressure and height levels, sea-level pressure and 10 m wind (wrf-python)."""
    import wrf

    o, run, t = job
    dest = o.cache / run / f'{t:%Y%m%d%H}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = lambda a: np.asarray(np.ma.filled(a, np.nan), float)
    with Dataset(wrfout(o, run, t)) as ds:
        g = lambda k: arr(ds[k][0])
        p = g('P') + g('PB')
        z = wrf.destagger((g('PH') + g('PHB')) / 9.81, 0, meta=False)
        tk = wrf.tk(p, g('T') + 300, meta=False)
        q = np.maximum(g('QVAPOR'), 0)
        tv = virtual_temperature(tk, q)
        u = wrf.destagger(g('U'), -1, meta=False)
        v = wrf.destagger(g('V'), -2, meta=False)
        ip = lambda a: arr(wrf.interplevel(a, np.log(p / 100), np.log(P_LEVELS), meta=False, missing=np.nan))
        iz = lambda a: arr(wrf.interplevel(a, z, Z_LEVELS, meta=False, missing=np.nan))
        f = dict(
            zp=ip(z),
            tvp=ip(tv),
            up=ip(u),
            vp=ip(v),
            uz=iz(u),
            vz=iz(v),
            slp=arr(wrf.slp(z, tk, p, q, meta=False)),
            u10=g('U10'),
            v10=g('V10'),
        )
        if not (dest.parent / 'grid.npz').exists():
            attrs = {
                k: float(ds.getncattr(k)) for k in ['DX', 'DY', 'TRUELAT1', 'TRUELAT2', 'MOAD_CEN_LAT', 'STAND_LON']
            }
            np.savez_compressed(
                dest.parent / 'grid.npz',
                lat=g('XLAT'),
                lon=g('XLONG'),
                sea=(g('LANDMASK') < 0.5) & (g('LAKEMASK') < 0.5),
                hgt=g('HGT'),
                mapfactor=g('MAPFAC_M'),
                ca=g('COSALPHA'),
                sa=g('SINALPHA'),
                pressure_hPa=P_LEVELS,
                height_m=Z_LEVELS,
                attrs=json.dumps(attrs),
            )
    np.savez_compressed(dest, **{k: np.asarray(v, 'f4') for k, v in f.items()})


def grid(o, run):
    g = dict(np.load(o.cache / run / 'grid.npz'))
    return g, json.loads(str(g['attrs']))


def mcv(job):
    o, run = job
    g, attrs = grid(o, run)
    vort = []
    for t in TIMES:
        with np.load(o.cache / run / f'{t:%Y%m%d%H}.npz') as d:
            vort.append(V.layer_vorticity(d, g, attrs).astype('f4'))
    acc = []
    for t in [TIMES[0] - pd.Timedelta(hours=1)] + list(TIMES):  # hourly model rainfall for the rain association
        with Dataset(wrfout(o, run, t)) as nc:
            acc.append(sum(np.asarray(nc[v][0], 'f8') for v in ('RAINC', 'RAINNC', 'RAINSH') if v in nc.variables))
    rain = np.diff(np.array(acc), axis=0).astype('f4')
    return V.track_mcv(vort, list(TIMES), rain, g['lat'], g['lon'], g['sea'], attrs['DX'] / 1000).assign(run=run)


def cyclones(job):
    """Surface tracks passing within 100 km of the MCV; structure of each of their hourly outputs."""
    o, run, m = job
    g, attrs = grid(o, run)
    slps = []
    for t in TIMES:
        with np.load(o.cache / run / f'{t:%Y%m%d%H}.npz') as d:
            slps.append(d['slp'])
    tr = V.track_surface(slps, list(TIMES), g['lat'], g['lon'], g['hgt'], ~g['sea'], g['mapfactor'], attrs['DX'] / 1000)
    j = tr.merge(m[['time', 'lat', 'lon']], on='time', suffixes=('', '_mcv'))
    ids = j[V.dist(j.lat, j.lon, j.lat_mcv, j.lon_mcv) <= 100].track_id.unique()
    p = g['pressure_hPa']
    cps = (p % 50 == 0) | (p == 925)
    rows = []
    for tid in ids:
        t_ = tr[tr.track_id == tid].sort_values('time').reset_index(drop=True)
        for k, r in t_.iterrows():
            heading, u, v = V.motion(t_, k)
            with np.load(o.cache / run / f'{r.time:%Y%m%d%H}.npz') as d:
                s = V.structure(d, g, (r.lat, r.lon), (u, v))
                c = V.phase_space(d['zp'][cps], p[cps], g['lat'], g['lon'], (r.lat, r.lon), heading, attrs)
            thermal = bool(c['mVTL'] > 0 and c['mVTU'] > 0 and abs(c['B']) < 10)
            rows.append(
                dict(
                    run=run,
                    track_id=tid,
                    time=r.time,
                    lat=r.lat,
                    lon=r.lon,
                    land=r.land,
                    heading=heading,
                    **c,
                    **s,
                    thermal=thermal,
                    structural_coherence=bool(
                        thermal and s['closure_earth'] and s['warm_anomaly_both'] and s['cavicchia_winds']
                    ),
                )
            )
    return pd.DataFrame(rows)


def associate(m, h):
    """The surface cyclone of the MCV: the track within 100 km of the MCV center at the most common hours (ties: deeper)."""
    rows, links = [], []
    for run, mm in m.groupby('run', sort=False):
        mm = mm.set_index('time')
        row = dict(run=run, mcv_peak_f=mm.zeta_f.max(), mcv_outputs=len(mm))
        best = None
        for tid, t in h[h.run == run].groupby('track_id'):
            j = t.set_index('time').join(mm[['lat', 'lon']], rsuffix='_mcv', how='inner')
            n = int((V.dist(j.lat, j.lon, j.lat_mcv, j.lon_mcv) <= 100).sum())
            sea = t[~t.land].sort_values('time')
            key = (n, -sea.slp_deficit.min() if len(sea) else 0.0)
            if n and (best is None or key > best[0]):
                best = (key, tid, sea, t.sort_values('time'))
        if best:
            (n, _), tid, sea, full = best
            row.update(
                track_id=tid,
                associated_hours=n,
                marine_outputs=len(sea),
                first=sea.time.min(),
                last=sea.time.max(),
                deficit_hPa=sea.slp_deficit.min(),
                v10max_ms=sea.v10max80.max(),
                thermal_pct=100 * sea.thermal.mean(),
                closure_pct=100 * sea.closure_earth.mean(),
                coherent=int(sea.structural_coherence.sum()),
                coherent_longest_run=V.persistence(full.time, full.structural_coherence, ~full.land),
            )
            links.append(sea)
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(links)


def profiles(job):
    """Relative vorticity within 50 km of the MCV center, on height levels and at 10 m, in units of f (Figure S3d)."""
    o, run, m = job
    g, a = grid(o, run)
    rows = []
    for _, r in m.iterrows():
        mask = V.dist(g['lat'], g['lon'], r.lat, r.lon) <= 50.0
        j, i = np.where(mask)
        win = np.s_[j.min() - 2 : j.max() + 3, i.min() - 2 : i.max() + 3]
        k_ = mask[win]
        with np.load(o.cache / run / f'{r.time:%Y%m%d%H}.npz') as d:

            def zeta(u, v, win=win, inside=k_):
                return (
                    np.nanmean(V.relative_vorticity(u[win], v[win], a['DX'], a['DY'], g['mapfactor'][win])[inside])
                    / V.F
                )

            rows.append(
                dict(
                    run=run,
                    time=r.time,
                    land=r.land,
                    z10=zeta(d['u10'], d['v10']),
                    **{str(int(h)): zeta(d['uz'][k], d['vz'][k]) for k, h in enumerate(g['height_m'])},
                )
            )
    return rows


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--runs', type=Path, required=True)
    a.add_argument('--cache', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--workers', type=int, default=4)
    a.add_argument('steps', nargs='*', default=['extract', 'mcv', 'cyclones', 'profiles'])
    o = a.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    runs = list(SST_RUNS)
    with ProcessPoolExecutor(o.workers) as pool:
        if 'extract' in o.steps:
            list(pool.map(extract, [(o, r, t) for r in runs for t in TIMES]))
        if 'mcv' in o.steps:
            pd.concat(pool.map(mcv, [(o, r) for r in runs])).to_csv(o.out / 'mcv_tracks.csv', index=False)
        m = pd.read_csv(o.out / 'mcv_tracks.csv', parse_dates=['time'])
        if 'cyclones' in o.steps:
            h = pd.concat(pool.map(cyclones, [(o, r, m[m.run == r]) for r in runs]))
            s, links = associate(m, h)
            s.to_csv(o.out / 'mcv_cyclone_summary.csv', index=False, float_format='%.2f')
            links.to_csv(o.out / 'associated_cyclone_hourly.csv', index=False, float_format='%.4f')
        if 'profiles' in o.steps:
            pd.DataFrame(
                [row for rows in pool.map(profiles, [(o, r, m[m.run == r]) for r in runs]) for row in rows]
            ).to_csv(o.out / 'mcv_vertical_profiles.csv', index=False, float_format='%.4f')
