"""MCV and surface-cyclone diagnostics (Text S2), from the hourly cache of scripts/diagnose_vortices.py."""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Proj
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter, label
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress

# Coriolis parameter at 41.3 N, f = 2 * Omega * sin(41.3 deg) = 2 * 7.2921e-5 * 0.6600 = 9.63e-5 s-1, rounded as in
# Text S2.1; vorticity is given in units of this fixed f.
F = 9.6e-5
GEO = Geod(a=6370000, b=6370000)  # WRF sphere
GENESIS = (40.8, 41.9, 1.8, 3.4)  # lat0, lat1, lon0, lon1 (Section 2.1)
TRACK_START, SEED_END, ASSOC_END = (
    pd.Timestamp('2026-10-03 18:00'),
    pd.Timestamp('2026-10-04 06:00'),
    pd.Timestamp('2026-10-04 08:00'),
)


def dist(la, lo, lb, lob):
    return np.hypot((la - lb) * 111.2, (lo - lob) * 111.2 * np.cos(np.radians(la)))


def relative_vorticity(u, v, dx, dy, mapfactor):
    """Grid-relative winds on the Lambert grid (MetPy, map-factor aware), s-1."""
    import metpy.calc as mpcalc
    from metpy.units import units

    return mpcalc.vorticity(
        u * units('m/s'),
        v * units('m/s'),
        dx=dx * units.meter,
        dy=dy * units.meter,
        parallel_scale=mapfactor,
        meridional_scale=mapfactor,
    ).magnitude


# ------------------------------------------------------------------ MCV
def layer_vorticity(d, g, attrs, levels=(500, 600)):
    """Relative vorticity averaged over the 600-500 hPa layer (trapezoid in pressure)."""
    p = g['pressure_hPa']
    sel = (p >= levels[0]) & (p <= levels[1])
    curl = relative_vorticity(d['up'][sel], d['vp'][sel], attrs['DX'], attrs['DY'], g['mapfactor'])
    o = np.argsort(p[sel])
    x = p[sel][o]
    return np.trapezoid(curl[o], x, axis=0) / (x[-1] - x[0])


def filtered(field, sigma, dx):
    """Gaussian-smoothed vorticity in units of f, normalized near missing values; -inf where the kernel is incomplete."""
    valid = np.isfinite(field)
    w = gaussian_filter(valid.astype(float), sigma / dx)
    return np.where(
        w > 0.99, gaussian_filter(np.where(valid, field, 0), sigma / dx) / np.maximum(w, 1e-12) / F, -np.inf
    )


def track_mcv(
    vort,
    times,
    rain,
    lat,
    lon,
    sea,
    dx,
    sigma=20.0,
    sigma_track=40.0,
    merge_km=40.0,
    link_km=100.0,
    max_gap=1,
    region_km=50.0,
    rain_km=50.0,
):
    """MCV of the simulated system.
    1. candidates: local maxima >= 0.4 f of the vorticity filtered at 40 km, at least 40 km apart;
    2. linking: one-to-one assignment minimizing the mean distance to the last and the extrapolated position, at most
       100 km per hour, one missing hour allowed;
    3. MCV: among tracks with a maximum >= 0.75 f (20 km filter) within 50 km of the genesis region between 1800 UTC 3 and
       0600 UTC 4 October, the one with the largest model rainfall within 50 km up to 0800 UTC 4 October.
    Intensity zeta_f: maximum of the 20 km filtered field within 40 km of the center."""
    size = int(2 * merge_km / dx) + 1
    Z, tracks, active = [], [], []
    for k, field in enumerate(vort):
        q = filtered(field, sigma_track, dx)
        Z.append(filtered(field, sigma, dx))
        pts = list(zip(*np.where((q == maximum_filter(q, size=size)) & (q >= 0.4))))
        used = set()
        nxt = []
        if active and pts:
            cost = np.full((len(active), len(pts)), 1e9)
            for a, idx in enumerate(active):
                h = tracks[idx]
                k1, j1, i1 = h[-1]
                gap = k - k1
                pla, plo = lat[j1, i1], lon[j1, i1]
                ela, elo = pla, plo
                if len(h) > 1:
                    k2, j2, i2 = h[-2]
                    r = gap / (k1 - k2)
                    ela, elo = pla + r * (pla - lat[j2, i2]), plo + r * (plo - lon[j2, i2])
                for b, (j, i) in enumerate(pts):
                    step = dist(lat[j, i], lon[j, i], pla, plo)
                    if step <= link_km * gap:
                        cost[a, b] = 0.5 * (step + dist(lat[j, i], lon[j, i], ela, elo))
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e8:
                    tracks[active[a]].append((k,) + pts[b])
                    used.add(b)
                    nxt.append(active[a])
        for b, (j, i) in enumerate(pts):
            if b not in used:
                tracks.append([(k, j, i)])
                nxt.append(len(tracks) - 1)
        active = nxt + [x for x in active if x not in nxt and k - tracks[x][-1][0] <= max_gap]
    zeta = lambda k, j, i: float(np.max(Z[k][dist(lat, lon, lat[j, i], lon[j, i]) <= merge_km]))
    y0, y1, x0, x1 = GENESIS
    region = lambda j, i: (
        np.hypot(
            max(y0 - lat[j, i], 0, lat[j, i] - y1) * 111.2,
            max(x0 - lon[j, i], 0, lon[j, i] - x1) * 111.2 * np.cos(np.radians(lat[j, i])),
        )
        <= region_km
    )
    cand = [
        tr for tr in tracks if any(times[k] <= SEED_END and region(j, i) and zeta(k, j, i) >= 0.75 for k, j, i in tr)
    ]
    if not cand:
        return pd.DataFrame()
    assoc = lambda tr: sum(
        float(rain[k][dist(lat, lon, lat[j, i], lon[j, i]) <= rain_km].mean())
        for k, j, i in tr
        if times[k] <= ASSOC_END
    )
    best = max(cand, key=assoc)
    return pd.DataFrame(
        [
            dict(
                time=times[k],
                lat=float(lat[j, i]),
                lon=float(lon[j, i]),
                zeta_f=zeta(k, j, i),
                land=not bool(sea[j, i]),
                rain_assoc_mm=assoc(best),
                candidates=len(cand),
            )
            for k, j, i in best
        ]
    )


# ------------------------------------------------------------------ surface cyclones (Picornell et al., 2014)
# A minimum is a cyclone if the mean pressure gradient to 50 km is at least 0.5 hPa per 100 km in 6 of 8 directions, and an
# intense cyclone (one of their medicane criteria) if it is at least 3.2 hPa per 100 km in 6 of 8 directions.
SURFACE = dict(
    gradient_radius_km=50,
    minimum_gradient_hPa_km=0.005,
    intense_gradient_hPa_km=0.032,
    minimum_sectors=6,
    maximum_displacement_km_hour=100,
    maximum_land_outputs=3,
)


def pressure_minima(slp, lat, lon, hgt, land, mapfactor, dx, sigma=10.0):
    """Minima of the smoothed sea-level pressure (cells below 300 m) with a pressure rise of at least 0.005 hPa/km to
    50 km in at least 6 of 8 directions; intense_sectors counts the directions with at least 0.032 hPa/km."""
    valid = hgt < 300
    smooth = gaussian_filter(np.where(valid, slp, 0), sigma / dx) / np.maximum(
        gaussian_filter(valid.astype(float), sigma / dx), 1e-12
    )
    minima = (smooth == minimum_filter(smooth, size=3, mode='nearest')) & valid
    minima[:3] = False
    minima[-3:] = False
    minima[:, :3] = False
    minima[:, -3:] = False
    lab, n = label(minima)
    points = []
    c = SURFACE
    for k in range(1, n + 1):
        jj, ii = np.where(lab == k)
        q = np.argmin(smooth[jj, ii])
        j, i = int(jj[q]), int(ii[q])
        ang = np.arange(8) * np.pi / 4
        rr = c['gradient_radius_km'] / dx * mapfactor[j, i]
        y, x = j + rr * np.sin(ang), i + rr * np.cos(ang)
        ring = map_coordinates(smooth, [y, x], order=1, mode='constant', cval=np.nan)
        ok = map_coordinates(valid.astype(float), [y, x], order=0, mode='constant', cval=0) > 0.5
        slope = (ring - smooth[j, i]) / c['gradient_radius_km']
        count = int(np.sum(ok & (slope >= c['minimum_gradient_hPa_km'])))
        if count >= c['minimum_sectors']:
            points.append(
                dict(
                    j=j,
                    i=i,
                    lat=float(lat[j, i]),
                    lon=float(lon[j, i]),
                    smoothed_slp=float(smooth[j, i]),
                    gradient=float(np.nanmean(slope[ok])),
                    sectors=count,
                    intense_sectors=int(np.sum(ok & (slope >= c['intense_gradient_hPa_km']))),
                    land=bool(land[j, i]),
                )
            )
    return points


def track_surface(slps, times, lat, lon, hgt, land, mapfactor, dx):
    """Link pressure minima hour to hour (one-to-one, mean distance to the last and extrapolated positions, at most
    100 km per hour; at most 3 consecutive land outputs). New tracks start at any marine minimum; all tracks are kept."""
    tracks, active = [], []
    c = SURFACE
    for n, time in enumerate(times):
        pts = pressure_minima(slps[n], lat, lon, hgt, land, mapfactor, dx)
        used = set()
        nxt = []
        if active and pts:
            cost = np.full((len(active), len(pts)), 1e9)
            for a, idx in enumerate(active):
                h = tracks[idx]
                prev = h[-1]
                if (time - prev['time']).total_seconds() != 3600:
                    continue
                pla, plo = prev['lat'], prev['lon']
                if len(h) > 1:
                    pla += prev['lat'] - h[-2]['lat']
                    plo += prev['lon'] - h[-2]['lon']
                for b, p in enumerate(pts):
                    if dist(p['lat'], p['lon'], prev['lat'], prev['lon']) <= c['maximum_displacement_km_hour']:
                        cost[a, b] = 0.5 * (
                            dist(p['lat'], p['lon'], pla, plo) + dist(p['lat'], p['lon'], prev['lat'], prev['lon'])
                        )
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] >= 1e8:
                    continue
                idx = active[a]
                p = pts[b]
                used.add(b)
                nland = 0
                for row in reversed(tracks[idx]):
                    if not row['land']:
                        break
                    nland += 1
                if p['land'] and nland >= c['maximum_land_outputs']:
                    continue
                tracks[idx].append(dict(time=time, track_id=idx, **p))
                nxt.append(idx)
        for b, p in enumerate(pts):
            if b not in used and not p['land']:
                idx = len(tracks)
                tracks.append([dict(time=time, track_id=idx, **p)])
                nxt.append(idx)
        active = nxt
    return pd.DataFrame([r for t in tracks for r in t])


# ------------------------------------------------------------------ structure around a surface center
def motion(track, k):
    a = track.iloc[max(k - 1, 0)]
    b = track.iloc[min(k + 1, len(track) - 1)]
    dt = (b.time - a.time).total_seconds()
    az, _, d = GEO.inv(a.lon, a.lat, b.lon, b.lat)
    if dt <= 0 or d < 1.0:
        return np.nan, 0.0, 0.0
    s = d / dt
    return (90 - az) % 360, s * np.sin(np.radians(az)), s * np.cos(np.radians(az))


def phase_space(zp, pl, lat, lon, center, heading, attrs, radius=100.0, layers=(925, 700, 400), dr=0.75, daz=0.5):
    """Cyclone phase space (Hart, 2003) with CPyS on geodesic circles within radius km: -VTL, -VTU and B."""
    import cpys

    proj = Proj(
        proj='lcc',
        lat_1=attrs['TRUELAT1'],
        lat_2=attrs['TRUELAT2'],
        lat_0=attrs['MOAD_CEN_LAT'],
        lon_0=attrs['STAND_LON'],
        a=6370000,
        b=6370000,
    )
    x0, y0 = proj(float(lon[0, 0]), float(lat[0, 0]))
    r = np.linspace(0, radius, int(np.ceil(radius / dr)) + 1)
    az = np.arange(0, 360, daz)
    rr, aa = np.meshgrid(r, az, indexing='ij')
    lo, la, _ = GEO.fwd(np.full(rr.shape, center[1]), np.full(rr.shape, center[0]), 90 - aa, rr * 1000)
    x, y = proj(lo, la)
    coords = [(y - y0) / attrs['DY'], (x - x0) / attrs['DX']]
    sampled = np.array([map_coordinates(z, coords, order=1, mode='constant', cval=np.nan) for z in zp])
    da = xr.DataArray(sampled[None], dims=['snapshot', 'plev', 'r', 'az'], coords={'plev': pl * 100, 'r': r, 'az': az})
    bot, mid, top = layers
    lower = float(cpys.vt(da, p_bottom=bot * 100, p_top=mid * 100)[0])
    upper = float(cpys.vt(da, p_bottom=mid * 100, p_top=top * 100)[0])
    b = (
        float(cpys.b(da.sel(plev=mid * 100) - da.sel(plev=bot * 100), np.array([center[0]]), np.array([heading]))[0])
        if np.isfinite(heading)
        else np.nan
    )
    spread = np.nanmax(sampled, axis=(1, 2)) - np.nanmin(sampled, axis=(1, 2))  # check with an independent fit
    own = [
        linregress(np.log(pl[(pl <= lo_) & (pl >= hi)]), spread[(pl <= lo_) & (pl >= hi)]).slope
        for lo_, hi in ((bot, mid), (mid, top))
    ]
    assert np.allclose([lower, upper], own, atol=1e-9)
    return dict(mVTL=lower, mVTU=upper, B=b)


def structure(d, g, center):
    """Depth and maximum 10 m wind speed of the cyclone around a center (Text S2.2)."""
    lat, lon = g['lat'], g['lon']
    _, _, rr = GEO.inv(np.full(lon.shape, center[1]), np.full(lat.shape, center[0]), lon, lat)
    r = rr / 1000
    w = g['mapfactor'] ** -2
    sea = g['sea']

    def mean(a, m):
        ok = np.isfinite(a) & m
        return float(np.sum(a[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan

    mx = lambda a, m: float(np.nanmax(a[m])) if np.isfinite(a[m]).any() else np.nan
    return dict(
        slp_deficit=-mx(-d['slp'], (r < 60) & sea) - mean(d['slp'], (r >= 120) & (r <= 200) & sea),
        v10max80=mx(np.hypot(d['u10'], d['v10']), (r <= 80) & sea),
    )


def persistence(times, passes, marine):
    """Longest run of consecutive hourly outputs that pass (land outputs and missing hours break it)."""
    times = pd.to_datetime(times)
    passes = np.asarray(passes, bool) & np.asarray(marine, bool)
    longest = run = 0
    prev = None
    for t, ok in zip(times, passes):
        run = run + 1 if ok and prev is not None and t - prev == pd.Timedelta(hours=1) else int(ok)
        longest = max(longest, run)
        prev = t
    return longest
