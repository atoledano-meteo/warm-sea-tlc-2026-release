"""Read OPERA ODIM HDF5 files: one sweep of a radar volume (Figure S1b,c) and the maximum-reflectivity composite
(Figure S1a)."""

import h5py
import numpy as np
from pyproj import Geod, Transformer

EARTH = 6371000.0
GEOD = Geod(a=EARTH, b=EARTH)  # spherical Earth, as in the beam model


def _txt(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _decode(group):
    a = group['what'].attrs
    raw = group['data'][:]
    v = raw.astype('f4') * float(a['gain']) + float(a['offset'])
    v[(raw == a['nodata']) | (raw == a['undetect'])] = np.nan
    return _txt(a['quantity']), v


def sweep(path, dataset='dataset1'):
    """Fields (DBZH, VRADH, ...) of one sweep, rays sorted by azimuth, with gate longitude and latitude
    (4/3 Earth radius beam model). The range of the first gate (rstart) is in metres."""
    with h5py.File(path) as h:
        lat0, lon0, h0 = (float(h['where'].attrs[k]) for k in ('lat', 'lon', 'height'))
        d = h[dataset]
        w = d['where'].attrs
        how = {**dict(h['how'].attrs), **dict(d['how'].attrs)}
        az = (
            np.asarray(how['startazA']) + ((np.asarray(how['stopazA']) - how['startazA'] + 180) % 360 - 180) / 2
        ) % 360
        order = np.argsort(az)
        az = az[order]
        ranges = float(w['rstart']) + (np.arange(int(w['nbins'])) + 0.5) * float(w['rscale'])
        el = (
            ((np.asarray(how['startelA']) + how['stopelA']) / 2)[order]
            if 'startelA' in how
            else np.full(len(az), float(w['elangle']))
        )
        fields = {q: v[order] for q, v in (_decode(d[k]) for k in sorted(d) if k.startswith('data') and 'what' in d[k])}
    ke = 4 / 3 * EARTH
    a = np.radians(el)[:, None]
    r = ranges[None, :]
    arc = np.arctan2(r * np.cos(a), ke + r * np.sin(a))
    aa = np.broadcast_to(az[:, None], arc.shape)
    lon, lat, _ = GEOD.fwd(np.full(aa.shape, lon0), np.full(aa.shape, lat0), aa, ke * arc)
    return lon, lat, fields


def composite(path, extent):
    """OPERA composite (first dataset) cropped to extent = (lon0, lon1, lat0, lat1), at the native pixel centres."""
    with h5py.File(path) as h:
        w = h['where'].attrs
        crs = _txt(w['projdef'])
        fwd = Transformer.from_crs(4326, crs, always_xy=True)
        inv = Transformer.from_crs(crs, 4326, always_xy=True)
        x0, y0 = fwd.transform(float(w['UL_lon']), float(w['UL_lat']))
        dx, dy = float(w['xscale']), float(w['yscale'])
        ny, nx = int(w['ysize']), int(w['xsize'])
        xx, yy = np.meshgrid(np.linspace(extent[0], extent[1], 21), np.linspace(extent[2], extent[3], 21))
        xs, ys = fwd.transform(xx, yy)
        c0 = max(0, int(np.floor((xs.min() - x0) / dx)) - 2)
        c1 = min(nx, int(np.ceil((xs.max() - x0) / dx)) + 3)
        r0 = max(0, int(np.floor((y0 - ys.max()) / dy)) - 2)
        r1 = min(ny, int(np.ceil((y0 - ys.min()) / dy)) + 3)
        g = h['dataset1/data1']
        a = g['what'].attrs
        raw = g['data'][r0:r1, c0:c1]
        z = raw.astype('f4') * float(a['gain']) + float(a['offset'])
        z[(raw == a['nodata']) | (raw == a['undetect'])] = np.nan
        t = h['dataset1/what'].attrs
        times = (_txt(t['startdate']) + _txt(t['starttime']), _txt(t['enddate']) + _txt(t['endtime']))
    x, y = np.meshgrid(x0 + (np.arange(c0, c1) + 0.5) * dx, y0 - (np.arange(r0, r1) + 0.5) * dy)
    lon, lat = inv.transform(x, y)
    return lon, lat, z, times
