"""Readers for the observations of Figures 1, S1 and S2 (files as delivered by each provider; see README)."""

import json
from functools import lru_cache
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer

from warmsea.config import OBS

KT = 0.514444


# ------------------------------------------------------------------ MTG-I1 FCI Level 1c (EUMETSAT, EO:EUM:DAT:0662)
def mtg_bt(folder, channel='ir_105'):
    """Brightness temperature (degC) of the FCI body chunks in folder (one repeat cycle), with lon/lat of the fixed
    grid (geostationary projection at 0 E, sweep axis y, GRS80)."""
    rows = []
    for f in sorted(folder.glob('*BODY*.nc')):
        with Dataset(f) as n:
            g = n[f'data/{channel}/measured']
            L = np.ma.filled(g['effective_radiance'][:].astype('f8'), np.nan)
            x = np.asarray(g['x'][:], 'f8')
            y = np.asarray(g['y'][:], 'f8')
            a, b = (
                float(g['radiance_to_bt_conversion_coefficient_a'][:]),
                float(g['radiance_to_bt_conversion_coefficient_b'][:]),
            )
            nu = float(g['radiance_to_bt_conversion_coefficient_wavenumber'][:])
            c1, c2 = (
                float(g['radiance_to_bt_conversion_constant_c1'][:]),
                float(g['radiance_to_bt_conversion_constant_c2'][:]),
            )
        L = np.where(L > 0, L, np.nan)
        rows.append((y, x, (c2 * nu / np.log(1 + c1 * nu**3 / L) - b) / a - 273.15))
    rows.sort(key=lambda r: r[0][0])
    y = np.concatenate([r[0] for r in rows])
    bt = np.concatenate([r[2] for r in rows])
    x = rows[0][1]
    tr = Transformer.from_crs(
        '+proj=geos +h=35786400 +lon_0=0 +sweep=y +a=6378137 +rf=298.257222101 +units=m', 'EPSG:4326', always_xy=True
    )
    lon, lat = tr.transform(*np.meshgrid(-x * 35786400.0, y * 35786400.0))  # FCI x grows westward
    bad = ~np.isfinite(lon) | (np.abs(lon) > 180)
    lon, lat = np.where(bad, np.nan, lon), np.where(bad, np.nan, lat)
    cols = np.flatnonzero(np.nanmax(np.where((lon > -3) & (lon < 9), 1, 0), axis=0) > 0)
    sl = slice(cols.min(), cols.max() + 1)
    return lon[:, sl], lat[:, sl], bt[:, sl]


# ------------------------------------------------------------------ Barcelona sounding (SMC, 1 s ascent data)
def sounding(path):
    """Ascent of a 1 s sounding file (tab-separated: time, altitude, pressure, temperature, RH, dew point, wind speed
    in knots, direction, east and north wind in knots, lat, lon); isolated dew-point drops are removed."""
    x = pd.read_csv(path, sep='\t').dropna(axis=1, how='all')
    x.columns = [c.replace('ftr_', '') for c in x.columns]
    x = x.rename(
        columns={
            'time': 's',
            'alt': 'z',
            'pres': 'p',
            'temp': 't',
            'hum': 'rh',
            'DP': 'td',
            'WF': 'ws',
            'WD': 'wd',
            'VEF': 'u',
            'VNF': 'v',
            'LAT': 'lat',
            'LON': 'lon',
        }
    )
    x[['ws', 'u', 'v']] *= KT
    x = x.iloc[: int(x.p.idxmin()) + 1]  # ascent only
    x = x[x.p <= x.p.cummin() + 1e-6].reset_index(drop=True)  # monotonic pressure
    med = x.td.rolling(21, center=True, min_periods=5).median()
    x.loc[(x.td - med).abs() > 5, ['td', 'rh']] = np.nan
    return x


def layer_means(x, dp):
    b = (x.p // dp) * dp + dp / 2
    out = x.groupby(b)[['z', 't', 'td', 'u', 'v', 'lat', 'lon', 's']].mean()
    out.index.name = 'p'
    return out.reset_index().sort_values('p', ascending=False).reset_index(drop=True)


def precipitable_water(x):
    """Precipitable water (mm) up to 300 hPa from 5 hPa layer means (MetPy)."""
    import metpy.calc as mc
    from metpy.units import units

    b = layer_means(x, 5.0).dropna(subset=['t', 'td'])
    return float(mc.precipitable_water(b.p.values * units.hPa, b.td.values * units.degC, top=300 * units.hPa).m)


# ------------------------------------------------------------------ county boundaries (ICGC) and Catalan land
@lru_cache(maxsize=1)
def counties():
    import cartopy.io.shapereader as shpreader

    r = shpreader.Reader(next((OBS / 'icgc').glob('*comarques*.shp')))
    try:
        return tuple(r.geometries())  # UTM 31N
    finally:
        r.close()


@lru_cache(maxsize=1)
def land():
    from shapely.ops import unary_union

    return unary_union(counties())


def land_mask(glon, glat, buffer_m=0.0):
    from shapely import contains_xy

    g = land().buffer(buffer_m) if buffer_m else land()
    X, Y = np.meshgrid(glon, glat)
    ux, uy = Transformer.from_crs(4326, 32631, always_xy=True).transform(X, Y)
    return contains_xy(g, ux, uy)


# ------------------------------------------------------------------ surface stations
def stations():
    """Hourly station table (stations/hourly_surface.csv): network, station_id, lat, lon, alt, time (UTC),
    psfc_hPa, mslp_hPa, wspd_ms, wdir_deg, gust_ms. AEMET, Meteo-France, XEMA and Puertos del Estado buoys."""
    return pd.read_csv(OBS / 'stations/hourly_surface.csv', parse_dates=['time'], dtype={'station_id': str})


def pressure_perturbation(base=('2026-10-03 12:00', '2026-10-03 18:00')):
    """Hourly pressure perturbation per station: pressure minus the station's mean over base, minus the median of
    these anomalies over all stations at the same time. XEMA station pressure (variable 34) from the Meteocat open-data
    files; AEMET and Meteo-France station pressure, or sea-level pressure for the buoys. Stations above 600 m excluded."""
    xd = OBS / 'stations'
    r = pd.DataFrame(json.loads((xd / 'xema_p34_t32_20261003_04.json').read_text()))
    meta = pd.DataFrame(json.loads((xd / 'xema_stations_metadata.json').read_text()))
    meta = meta.assign(
        lat=meta.latitud.astype(float), lon=meta.longitud.astype(float), alt=meta.altitud.astype(float)
    ).set_index('codi_estacio')
    r['time'] = pd.to_datetime(r.data_lectura)
    r['v'] = r.valor_lectura.astype(float)
    r = r[r.time.dt.minute == 0]
    w = (
        r.pivot_table(index=['codi_estacio', 'time'], columns='codi_variable', values='v')
        .reset_index()
        .dropna(subset=['34', '32'])
    )
    w = w.join(meta[['lat', 'lon', 'alt']], on='codi_estacio')
    w = w[w.alt < 600]
    xema = pd.DataFrame({'id': 'XEMA:' + w.codi_estacio, 'lat': w.lat, 'lon': w.lon, 'time': w.time, 'p': w['34']})
    a = stations()
    a = a[a.network.isin(['AEMET', 'PORTUS', 'MF'])].copy()
    a['p'] = a.psfc_hPa.where(a.psfc_hPa.notna(), a.mslp_hPa)
    a.loc[a.psfc_hPa.notna() & (a.alt.fillna(0) >= 600), 'p'] = np.nan
    other = pd.DataFrame(
        {'id': a.network + ':' + a.station_id, 'lat': a.lat, 'lon': a.lon, 'time': a.time, 'p': a.p}
    ).dropna()
    p = pd.concat([xema, other], ignore_index=True)
    p = p[p.lat.between(40.3, 42.9) & p.lon.between(0.2, 3.9)]
    b = p[(p.time >= base[0]) & (p.time <= base[1])].groupby('id').p.mean()
    p = p.join(b.rename('base'), on='id').dropna(subset=['base'])
    p['anom'] = p.p - p.base
    return pd.concat(
        [g.assign(pert=g.anom.values - np.median(g.anom.values)) for t, g in p.groupby('time') if len(g) >= 12],
        ignore_index=True,
    )


def barnes(lon, lat, val, glon, glat, L=20.0, mask_km=25.0):
    """Two-pass Barnes analysis (length scale L km; second pass 0.6 L on the residuals), masked beyond mask_km from
    the nearest station."""
    kx = np.cos(np.radians(41.5)) * 111.2
    X, Y = np.meshgrid(glon * kx, glat * 111.2)
    sx, sy = lon * kx, lat * 111.2
    d2 = (X[..., None] - sx) ** 2 + (Y[..., None] - sy) ** 2
    w = np.exp(-d2 / L**2)
    field = (w * val).sum(-1) / w.sum(-1)
    ws = np.exp(-((sx[:, None] - sx[None]) ** 2 + (sy[:, None] - sy[None]) ** 2) / L**2)
    res = val - (ws * val[None]).sum(-1) / ws.sum(-1)
    w2 = np.exp(-d2 / (0.6 * L) ** 2)
    field = field + (w2 * res).sum(-1) / np.maximum(w2.sum(-1), 1e-12)
    return np.where(np.sqrt(d2.min(-1)) <= mask_km, field, np.nan)


def thin(df, km):
    """Keep the strongest wind first, then any station more than km from those kept."""
    keep, taken = [], []
    for i, r in df.sort_values('wspd_ms', ascending=False).iterrows():
        if all(np.hypot((r.lon - lo) * 83.0, (r.lat - la) * 111.2) > km for la, lo in taken):
            keep.append(i)
            taken.append((r.lat, r.lon))
    return df.loc[keep]
