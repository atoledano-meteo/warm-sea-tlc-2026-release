"""Dataset files from the full WRF output (not archived).

    python scripts/export_dataset.py --runs RUNS --rttov RTTOV --diagnostics DIAGNOSTICS --weights WEIGHTS \
        --reference REFERENCE --scores SCORES --out DATASET [steps]
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.runs import SST_RUNS, MAP_TIMES, SECTIONS
from warmsea.wrf import mslp, temperature, virtual_temperature

ZLIB = {'zlib': True, 'complevel': 4}
SOURCE = (
    'WRF-ARW 4.6.1; Toledano et al., A Warmer Sea Turns a Mediterranean Convective Vortex Into a Tropical-Like Cyclone'
)
LATLON = {'lat': {'units': 'degree_north'}, 'lon': {'units': 'degree_east'}}


def wrfout(run_dir, t, dom=2):
    return run_dir / 'wrf' / f'wrfout_d{dom:02d}_{t:%Y-%m-%d_%H}:00:00'


def write(ds, path, title, single=True, **attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs = {'title': title, 'source': SOURCE, **attrs}
    for k, a in LATLON.items():
        if k in ds.variables:
            ds[k].attrs.update(a)
    enc = {v: dict(ZLIB, dtype='float32') if single else ZLIB for v in ds.data_vars if ds[v].dtype.kind == 'f'}
    ds.to_netcdf(path, encoding=enc)


def grid(src, out):
    """Model grids, terrain and land mask of both domains (Figure 2a)."""
    v = {}
    for d in ('d01', 'd02'):
        with Dataset(src / f'wrf/wrfinput_{d}') as n:
            for k in ('XLONG', 'XLAT', 'HGT', 'LANDMASK', 'LAKEMASK'):
                attrs = {'wrf_name': k, 'description': n[k].description, 'units': n[k].units}
                v[f'{k.lower()}_{d}'] = ((f'y_{d}', f'x_{d}'), np.asarray(n[k][0], 'f4'), attrs)
    write(xr.Dataset(v), out / 'grid.nc', 'WRF grids: d01 (9 km) and d02 (3 km)')


def sst(runs, out):
    """Hourly SST read by WRF (wrflowinp) for each SST state, on both domains."""
    for state in ('clim', 'curr', 'plus2k'):  # +2K-ALL reads the +2K SST (its wrflowinp differs only over land)
        run = f'{state}_ctrl'
        v = {}
        for d in ('d01', 'd02'):
            with Dataset(runs / run / f'wrf/wrflowinp_{d}') as n:
                times = pd.to_datetime([b''.join(t).decode() for t in n['Times'][:]], format='%Y-%m-%d_%H:%M:%S')
                attrs = {'wrf_name': 'SST', 'description': n['SST'].description, 'units': n['SST'].units}
                v[f'sst_{d}'] = (('time', f'y_{d}', f'x_{d}'), np.asarray(n['SST'][:], 'f4'), attrs)
        ds = xr.Dataset(v, coords={'time': times})
        write(
            ds,
            out / 'sst' / f'{state}.nc',
            f'Hourly SST read by WRF (wrflowinp), {state.upper()} state; values over land are not used',
        )


def rainfall(runs, out):
    """Hourly rainfall (RAINC + RAINNC + RAINSH) on the 3 km domain, 48 h from 2200 UTC 2 October."""
    for name in SST_RUNS:
        run_dir = runs / name
        with xr.open_dataset(run_dir / 'validation/tolemeteo_primary/rainfall_48h_native.nc') as d:
            ds = xr.Dataset(
                {'rain': (('time', 'y', 'x'), d.rainfall_mm.values)},
                coords={
                    'time': d.time.values,
                    'lat': (('y', 'x'), d.latitude.values),
                    'lon': (('y', 'x'), d.longitude.values),
                },
            )
        ds.rain.attrs = {'units': 'mm', 'description': 'RAINC + RAINNC + RAINSH accumulated in the hour ending at time'}
        write(ds, out / 'rainfall' / f'{name}.nc', 'Hourly rainfall on d02', run=name)


def verification_region(runs, out):
    with xr.open_dataset(runs / 'curr_ctrl/validation/tolemeteo_primary/cellwise_comparison.nc') as d:
        ds = xr.Dataset(
            {'mask': (('y', 'x'), d.valid_comparison_mask.values.astype('i1'))},
            coords={'lat': (('y', 'x'), d.latitude.values), 'lon': (('y', 'x'), d.longitude.values)},
        )
    ds.mask.attrs['description'] = '1 inside the verification region'
    write(ds, out / 'verification_region.nc', 'Verification region on the 3 km verification grid (UTM 31N)')


def surface_fields(run_dir, t, centre):
    """10 m wind, column-maximum reflectivity and sea-level pressure (below 300 m) around the pressure minimum over the
    sea within 60 km of a first-guess centre."""
    with Dataset(wrfout(run_dir, t)) as d:
        g = lambda v: np.asarray(d[v][0], 'f8')
        lat, lon, hgt = g('XLAT'), g('XLONG'), g('HGT')
        sea = g('XLAND') > 1.5
        p = gaussian_filter(mslp(g('PSFC'), g('T2'), hgt), 1)
        ca, sa = g('COSALPHA'), g('SINALPHA')
        u, v = g('U10') * ca - g('V10') * sa, g('V10') * ca + g('U10') * sa
        dbz = np.asarray(d['REFL_10CM'][0], 'f8').max(0)
    y0, x0 = centre
    r0 = np.hypot((lat - y0) * 111, (lon - x0) * 111 * np.cos(np.radians(y0)))
    k = np.argmin(np.where((r0 < 60) & sea, p, 1e9))
    return dict(
        lat=lat, lon=lon, p=np.where(hgt < 300, p, np.nan), u=u, v=v, dbz=dbz, yc=lat.flat[k], xc=lon.flat[k], sea=sea
    )


def first_guess(run_dir, t, box=(40.0, 42.5, 1.0, 5.5)):
    """Lowest sea-level pressure over the sea cells of the marine region 40.0-42.5 N, 1.0-5.5 E."""
    with Dataset(wrfout(run_dir, t)) as d:
        g = lambda v: np.asarray(d[v][0], 'f8')
        lat, lon = np.asarray(d['XLAT'][0]), np.asarray(d['XLONG'][0])
        sea = np.asarray(d['LANDMASK'][0]) <= 0.5
        sl = mslp(g('PSFC'), g('T2'), g('HGT'))
    inside = (lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3]) & sea
    j, i = np.unravel_index(np.argmin(np.where(inside, sl, 1e9)), sl.shape)
    return float(lat[j, i]), float(lon[j, i])


def section(run_dir, t, yc, xc, motion, half=150.0, ns=121, zk=np.arange(100, 14001, 100.0)):
    """West-east section through (yc, xc): virtual-temperature anomaly relative to the marine columns 100-150 km from
    the centre, condensate, storm-relative meridional wind, and sea-level pressure and rain rate along it."""
    with Dataset(wrfout(run_dir, t)) as d, Dataset(wrfout(run_dir, t - pd.Timedelta(hours=1))) as d0:
        g = lambda v, n=d: np.asarray(n[v][0], 'f8')
        lat, lon, hgt = g('XLAT'), g('XLONG'), g('HGT')
        sea = g('XLAND') > 1.5
        p = g('P') + g('PB')
        tk = temperature(p, g('T') + 300)
        qv = g('QVAPOR')
        z = 0.5 * ((g('PH') + g('PHB'))[1:] + (g('PH') + g('PHB'))[:-1]) / 9.81
        u = 0.5 * (g('U')[:, :, 1:] + g('U')[:, :, :-1])
        v = 0.5 * (g('V')[:, 1:, :] + g('V')[:, :-1, :])
        v = v * g('COSALPHA') + u * g('SINALPHA')
        cond = sum(g(n) for n in ('QCLOUD', 'QICE', 'QSNOW', 'QGRAUP')) * 1e3
        slp = mslp(g('PSFC'), g('T2'), hgt)
        tv = virtual_temperature(tk, qv)
        rr = (g('RAINNC') + g('RAINC')) - (g('RAINNC', d0) + g('RAINC', d0))
    kx = 111.2 * np.cos(np.radians(yc))
    xs = np.linspace(-half, half, ns)
    idx = [
        np.unravel_index(np.argmin((lat - yc) ** 2 + ((lon - (xc + x / kx)) * np.cos(np.radians(yc))) ** 2), lat.shape)
        for x in xs
    ]
    jj, ii = map(np.array, zip(*idx))
    col = lambda a, J, I: (
        np.array([np.interp(zk, z[:, j, i], a[:, j, i], left=np.nan, right=np.nan) for j, i in zip(J, I)]).T
    )
    out = {k: col(a, jj, ii) for k, a in (('cond', cond), ('v', v), ('tv', tv))}
    r = np.hypot((lat - yc) * 111.2, (lon - xc) * kx)
    rj, ri = np.where((r >= 100) & (r <= 150) & sea)
    out['ta'] = out['tv'] - np.nanmean(col(tv, rj, ri), 1, keepdims=True)
    out['vr'] = out['v'] - motion[1]
    out.update(x=xs, z=zk, hgt=hgt[jj, ii], sea=sea[jj, ii], slp=slp[jj, ii], rain=rr[jj, ii])
    return out


def maps(runs, rttov, out):
    """Fields of the map panels of Figures 4 and S4 at one time, and the simulated 10.5 um brightness temperature."""
    for name, t in MAP_TIMES.items():
        t = pd.Timestamp(t)
        F = surface_fields(runs / name, t, first_guess(runs / name, t))
        b = np.load(rttov / f'{name}_{t:%Y%m%d%H}.npz')  # written by simulate_satellite.py
        ds = xr.Dataset(
            {k: (('y', 'x'), F[k]) for k in ('p', 'u', 'v', 'dbz')} | {'bt': (('yb', 'xb'), b['ir105'])},
            coords={
                'lat': (('y', 'x'), F['lat']),
                'lon': (('y', 'x'), F['lon']),
                'lat_bt': (('yb', 'xb'), b['lat']),
                'lon_bt': (('yb', 'xb'), b['lon']),
            },
        )
        for k, units, desc in (
            ('p', 'hPa', 'sea-level pressure from PSFC, T2 and HGT (6.5 K/km), smoothed; missing above 300 m'),
            ('u', 'm s-1', '10 m eastward wind (U10, V10 rotated to Earth coordinates)'),
            ('v', 'm s-1', '10 m northward wind (U10, V10 rotated to Earth coordinates)'),
            ('dbz', 'dBZ', 'column maximum of REFL_10CM'),
            ('bt', 'degC', 'MTG-I1 FCI 10.5 um brightness temperature simulated with RTTOV 14.1'),
        ):
            ds[k].attrs = {'units': units, 'description': desc}
        ds.lat_bt.attrs.update(LATLON['lat'])
        ds.lon_bt.attrs.update(LATLON['lon'])
        write(
            ds,
            out / 'maps' / f'{name}_{t:%Y%m%d%H}.nc',
            'Map fields of Figures 4 and S4',
            single=False,  # double precision: contours as in the paper
            run=name,
            time=f'{t:%Y-%m-%d %H:%M} UTC',
            centre_lat=F['yc'],
            centre_lon=F['xc'],
            centre='sea-level pressure minimum',
        )


def sections(runs, out):
    """Five-hour mean west-east sections of Figure 4g,h."""
    for name, t in SECTIONS.items():
        t = pd.Timestamp(t)
        hrs = pd.date_range(t - pd.Timedelta(hours=2), t + pd.Timedelta(hours=2), freq='h')
        C = {
            h: surface_fields(runs / name, h, first_guess(runs / name, h))
            for h in pd.date_range(hrs[0] - pd.Timedelta(hours=1), hrs[-1] + pd.Timedelta(hours=1), freq='h')
        }
        S = []
        for h in hrs:
            a, b = C[h - pd.Timedelta(hours=1)], C[h + pd.Timedelta(hours=1)]
            motion = (
                (b['xc'] - a['xc']) * 111.2e3 * np.cos(np.radians(C[h]['yc'])) / 7200,
                (b['yc'] - a['yc']) * 111.2e3 / 7200,
            )
            S.append(section(runs / name, h, C[h]['yc'], C[h]['xc'], motion))
        M = {k: np.nanmean([s[k] for s in S], 0) for k in ('ta', 'cond', 'vr', 'hgt', 'slp', 'rain')}
        ds = xr.Dataset(
            {
                'ta': (('z', 'x'), M['ta']),
                'cond': (('z', 'x'), M['cond']),
                'vr': (('z', 'x'), M['vr']),
                'hgt': ('x', M['hgt']),
                'slp': ('x', M['slp']),
                'rain': ('x', M['rain']),
            },
            coords={'x': S[0]['x'], 'z': S[0]['z']},
        )
        for k, units, desc in (
            ('ta', 'K', 'virtual-temperature anomaly relative to the marine columns 100-150 km from the centre'),
            ('cond', 'g kg-1', 'QCLOUD + QICE + QSNOW + QGRAUP'),
            ('vr', 'm s-1', 'storm-relative northward wind'),
            ('hgt', 'm', 'terrain height (HGT)'),
            ('slp', 'hPa', 'sea-level pressure from PSFC, T2 and HGT (6.5 K/km)'),
            ('rain', 'mm h-1', 'RAINC + RAINNC in the previous hour'),
            ('x', 'km', 'distance east of the sea-level pressure minimum'),
            ('z', 'm', 'height above sea level'),
        ):
            ds[k].attrs = {'units': units, 'description': desc}
        write(
            ds,
            out / 'sections' / f'{name}.nc',
            'Five-hour mean west-east section of Figure 4g,h',
            single=False,
            run=name,
            start=f'{hrs[0]:%Y-%m-%d %H:%M} UTC',
            end=f'{hrs[-1]:%Y-%m-%d %H:%M} UTC',
        )


def diagnostics(diag, out):
    """Tables written by diagnose_vortices.py and air_sea_exchange.py."""
    (out / 'diagnostics').mkdir(parents=True, exist_ok=True)
    for f in ('mcv_vertical_profiles', 'associated_cyclone_hourly', 'mcv_cyclone_summary', 'air_sea_exchange'):
        shutil.copy(diag / f'{f}.csv', out / 'diagnostics' / f'{f}.csv')


def regrid(weights, out):
    """Area weights from the 3 km model grid to the 3 km verification grid (sparse matrix, rows: verification cells)."""
    shutil.copy(weights, out / 'regrid_model_to_verification.npz')


def scores(reports, out):
    """Rainfall scores and peak MCV vorticity of the configurations of Table S1. The scores compare with the rainfall
    reference, which is not archived (see README)."""
    d = pd.read_csv(reports / 'zeus_reruns/selection_metrics.csv').set_index('case')
    mcv = pd.concat(
        [
            pd.read_csv(reports / 'zeus_reruns/selection_mcv.csv').set_index('case').mcv_peak_f,
            pd.read_csv(reports / 'selection_mcv.csv').set_index('case').mcv_peak_f.filter(like='rep_zeus'),
        ]
    )
    name = {
        f'{"" if f == "gfs" else f + "_"}{mp}_{pbl}': f'sel_{f}_{mp}_{pbl}'
        for f in ('gfs', 'ifs', 'era5t')
        for mp in ('thompson', 'morrison')
        for pbl in ('mynn', 'ysu')
    }
    name.update(
        {
            'era5t_thompson_mynn': 'tmynn_curr_ctrl',
            'rep_zeus_control': 'curr_ctrl',
            'rep_zeus_skebs1': 'curr_skb1',
            'rep_zeus_skebs2': 'curr_skb2',
        }
    )
    d = d.drop(index='era5t_morrison_ysu')  # same run as rep_zeus_control (curr_ctrl)
    t = pd.DataFrame(
        {
            'run': d.index.map(name),
            'forcing': d.forcing,
            'physics': d.physics.str.replace('--', '-'),
            'zeta_m_f': mcv.reindex(d.index).values,
            'bias_mm': d.bias_all,
            'fss50': d.FSS50_51_all,
            'fss100': d.FSS100_27_all,
            'p99_mm': d.p99_all,
            'obs_p99_mm': d.obs_p99_all,
            'top20_mm': d.top20_bias,
            'r_hourly': d.hourly_corr,
            'lag_h': d.peak3h_lag,
        }
    )
    (out / 'scores').mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'scores' / 'selection.csv', index=False, float_format='%.4f')


def gauges(reference, out):
    """The 599 rain gauges of the rainfall reference: identifiers and positions only (no data)."""
    h = (
        pd.read_csv(reference / 'gauge_window_totals.csv', dtype={'station_id': str})
        .groupby(['network', 'station_id'], as_index=False)[['latitude', 'longitude']]
        .first()
    )
    h['data'] = 'hourly'
    x = pd.read_csv(reference / 'xom_native_assimilation.csv', dtype={'station_id': str})
    a = pd.read_csv(reference / 'andorra_native_assimilation.csv', dtype={'station_id': str})
    a = a[a.included_after_completeness_QC]
    d = [
        g.groupby(['network', 'station_id'], as_index=False)[['latitude', 'longitude']].first().assign(data='daily')
        for g in (x, a)
    ]
    t = pd.concat([h, *d], ignore_index=True)
    t['network'] = t.network.replace({'MF': 'Meteo-France', 'ANDORRA': 'Meteo Andorra'})
    t.to_csv(out / 'rain_gauges.csv', index=False, float_format='%.5f')


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument(
        '--runs', type=Path, required=True, help='one folder per run of the SST experiments (names in warmsea/runs.py)'
    )
    a.add_argument('--diagnostics', type=Path, help='folder with the tables of diagnose_vortices.py')
    a.add_argument('--rttov', type=Path, help='simulated brightness temperatures (simulate_satellite.py)')
    a.add_argument('--weights', type=Path, help='model-to-verification-grid weights (npz)')
    a.add_argument('--reference', type=Path, help='rainfall reference folder (gauge tables)')
    a.add_argument('--scores', type=Path, help='folder with the selection scores (selection/zeus_reruns)')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument(
        'steps',
        nargs='*',
        help='grid sst rainfall verification_region maps sections diagnostics regrid scores gauges (default: all)',
    )
    o = a.parse_args()
    steps = dict(
        grid=lambda: grid(o.runs / 'curr_ctrl', o.out),
        sst=lambda: sst(o.runs, o.out),
        rainfall=lambda: rainfall(o.runs, o.out),
        verification_region=lambda: verification_region(o.runs, o.out),
        maps=lambda: maps(o.runs, o.rttov, o.out),
        sections=lambda: sections(o.runs, o.out),
        diagnostics=lambda: diagnostics(o.diagnostics, o.out),
        regrid=lambda: regrid(o.weights, o.out),
        scores=lambda: scores(o.scores, o.out),
        gauges=lambda: gauges(o.reference, o.out),
    )
    for k in o.steps or steps:
        steps[k]()
