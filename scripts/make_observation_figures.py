"""Figures 1, S1 and S2, which show observations.

    python scripts/make_observation_figures.py            # all three
    python scripts/make_observation_figures.py S1

The observations are not redistributed; the folder set as `observations` in config.yaml must hold the files listed
in the README. Figures 1b, 1c and S2 also need the rainfall reference, which is not public.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.config import DATASET, ERA5, OBS, OUTPUT
from warmsea.style import (
    plt,
    pe,
    cf,
    W,
    PC,
    REGIONAL,
    KT,
    rain_cmap,
    ir_cmap,
    letter,
    utc,
    latlon,
    coast,
    boxline,
    save,
)
from warmsea import observations as O
from warmsea.radar import sweep, composite
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import pyproj

FIG = OUTPUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
T0 = pd.Timestamp('2026-10-04 00:00')
GELIDA, OPOUL = (1.884890031, 41.40817998), (2.86499, 42.91839)
SITE = (41.3844, 2.1181)  # Barcelona sounding (Faculty of Physics)
ZOOM = [1.0, 3.3, 40.75, 42.4]  # Figure 1c
INK = '#1a1a19'
RAIN_LEVELS = [0.5, 2, 5, 10, 20, 30, 50, 80]
RAIN_COLORS = ['#cde2fb', '#9ec5f4', '#6da7ec', '#2a78d6', '#1baf7a', '#eda100', '#eb6834', '#e34948']
REF = OBS / 'rainfall_reference'


def counties(ax):
    ax.add_feature(
        ShapelyFeature(O.counties(), ccrs.UTM(31), facecolor='none', edgecolor='#8a8983', linewidth=0.45), zorder=2
    )


def surface_panel(fig, cell, t):
    """Rain in the previous 2 h, pressure perturbation (over land, within 25 km of a station) and station winds."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import PathPatch
    from cartopy.mpl.path import shapely_to_path
    from shapely.geometry import Point, box

    g2 = cell.subgridspec(2, 3, height_ratios=[1, 1e-3], width_ratios=[1, 1e-3, 1e-3], wspace=0, hspace=0)
    ax = fig.add_subplot(g2[0, 0], projection=REGIONAL)
    glon, glat = np.arange(ZOOM[0] - 0.3, ZOOM[1] + 0.301, 0.02), np.arange(ZOOM[2] - 0.3, ZOOM[3] + 0.301, 0.02)
    with xr.open_dataset(REF / 'precip_obs_hourly_native.nc') as h:
        r = h.precipitation_mm.sel(
            latitude=slice(ZOOM[2] - 0.05, ZOOM[3] + 0.05), longitude=slice(ZOOM[0] - 0.05, ZOOM[1] + 0.05)
        ).load()
    cmap = ListedColormap(RAIN_COLORS).with_extremes(over='#a1203f')
    norm = BoundaryNorm(RAIN_LEVELS, cmap.N)
    r2 = r.sel(time=slice(np.datetime64(t - pd.Timedelta(hours=1)), np.datetime64(t))).sum('time').values
    im = ax.pcolormesh(
        r.longitude,
        r.latitude,
        np.ma.masked_less(r2, 0.5),
        cmap=cmap,
        norm=norm,
        shading='auto',
        rasterized=True,
        transform=PC,
        zorder=1,
    )
    ax.add_feature(cf.COASTLINE.with_scale('10m'), lw=0.8, color=INK, zorder=3)
    ax.set_extent(ZOOM, crs=PC)
    counties(ax)
    for name, la, lo in (('Barcelona', 41.39, 2.17), ('Girona', 41.98, 2.82), ('Tarragona', 41.12, 1.25)):
        ax.plot(lo, la, 's', ms=3.5, color=INK, transform=PC, zorder=6)
        ax.text(lo + 0.03, la + 0.025, name, fontsize=8.5, color=INK, transform=PC, zorder=6)
    p = O.pressure_perturbation()
    g = p[p.time == t]
    f = np.where(
        O.land_mask(glon, glat, buffer_m=8000.0),
        O.barnes(g.lon.values, g.lat.values, g.pert.values, glon, glat),
        np.nan,
    )
    neg = ax.contour(
        glon,
        glat,
        f,
        levels=[-2.5, -2, -1.5, -1, -0.5],
        colors='#0b2a7a',
        linewidths=1.7,
        linestyles='solid',
        transform=PC,
        zorder=4,
    )
    pos = ax.contour(glon, glat, f, levels=[0.5, 1, 1.5, 2], colors='#b3261e', linewidths=1.0, transform=PC, zorder=4)
    ax.clabel(neg, fmt='%.1f', fontsize=8)
    ax.clabel(pos, fmt='+%.1f', fontsize=7)
    frame = box(*ax.get_xlim()[:1], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
    clip = PathPatch(
        shapely_to_path(ax.projection.project_geometry(O.land(), ccrs.UTM(31)).intersection(frame)),
        transform=ax.transData,
        fc='none',
        ec='none',
    )
    ax.add_patch(clip)
    for cs in (neg, pos):  # isolines clipped at the coastline; labels over the sea dropped
        cs.set_clip_path(clip)
        for tx in list(cs.labelTexts):
            if not O.land().contains(Point(ccrs.UTM(31).transform_point(*tx.get_position(), ax.projection))):
                tx.remove()
    d = O.stations()
    w = d[
        (d.time == t)
        & d.wdir_deg.notna()
        & d.wspd_ms.notna()
        & (d.alt.fillna(0) < 300)
        & d.network.isin(['AEMET', 'XEMA', 'MF', 'PORTUS'])
    ]
    w = w[w.lat.between(ZOOM[2] - 0.2, ZOOM[3] + 0.2) & w.lon.between(ZOOM[0] - 0.2, ZOOM[1] + 0.2)].copy()
    w['u'] = -w.wspd_ms * np.sin(np.radians(w.wdir_deg))
    w['v'] = -w.wspd_ms * np.cos(np.radians(w.wdir_deg))
    wt = O.thin(w[w.lat.between(ZOOM[2], ZOOM[3]) & w.lon.between(ZOOM[0], ZOOM[1])], km=10.0)
    ax.barbs(
        wt.lon.values,
        wt.lat.values,
        (wt.u / KT).values,
        (wt.v / KT).values,
        length=5.0,
        linewidth=0.75,
        color=INK,
        transform=PC,
        zorder=7,
        sizes={'emptybarb': 0.12, 'spacing': 0.16, 'height': 0.45},
        path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
    )
    ax.set_extent(ZOOM, crs=PC)
    for tx in list(ax.texts):
        x_, y_ = ax.transAxes.inverted().transform(tx.get_transform().transform(tx.get_position()))
        if not (0.04 < x_ < 0.96 and 0.04 < y_ < 0.96):  # contour labels cut by the frame
            tx.remove()
            continue
        tx.set_fontsize(6)
        tx.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])
    return ax, im


def fig1():
    """(a) MTG-I1 10.5 um; (b) 48 h rainfall reference with the gauges; (c) surface stations; (d) Barcelona sounding."""
    fig = plt.figure(figsize=(W, 7.0))
    grid = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.22, left=0.08, right=0.94, top=0.97, bottom=0.06)
    # (a)
    ax = fig.add_subplot(grid[0, 0], projection=REGIONAL)
    ax.set_extent([-0.5, 5.5, 39.3, 43.6], crs=PC)
    cm, nm = ir_cmap()
    lon, lat, bt = O.mtg_bt(OBS / 'mtg' / f'{T0:%Y%m%dT%H%M}')
    m = ax.pcolormesh(lon, lat, bt, transform=PC, cmap=cm, norm=nm, shading='auto', rasterized=True)
    coast(ax, '10m', color='#ff2fb3')
    latlon(ax, range(0, 6, 2), range(40, 44))
    letter(ax, 'a')
    ax.text(
        0.97,
        0.03,
        utc(T0),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=6,
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1),
        zorder=20,
    )
    cb = fig.colorbar(m, ax=ax, orientation='horizontal', fraction=0.04, pad=0.07, aspect=30, ticks=range(-80, 41, 40))
    cb.set_label('Brightness temperature (°C)', fontsize=6.3)
    cb.ax.tick_params(labelsize=6)
    # (b)
    cmap, norm = rain_cmap()
    ax = fig.add_subplot(grid[0, 1], projection=REGIONAL)
    ax.set_extent([0.15, 3.35, 40.55, 42.85], crs=PC)
    N = xr.open_dataset(REF / 'precip_obs_native.nc')
    nat = N.precipitation_mm.sel(period='2026-10-03_04').values
    m = ax.pcolormesh(
        N.longitude.values,
        N.latitude.values,
        np.where(nat >= 0.1, nat, np.nan),
        transform=PC,
        cmap=cmap,
        norm=norm,
        shading='auto',
        rasterized=True,
        zorder=1,
    )
    counties(ax)
    ax.add_feature(cf.COASTLINE.with_scale('10m'), lw=0.6, color='#222', zorder=8)
    g = pd.read_csv(DATASET / 'rain_gauges.csv')
    g = g[g.data == 'hourly']
    ax.scatter(g.longitude, g.latitude, s=1.6, color='k', alpha=0.35, lw=0, transform=PC, zorder=9)
    ax.plot(*GELIDA, marker='^', ms=4, mfc='white', mec='k', mew=0.6, transform=PC, zorder=11)
    boxline(ax, *ZOOM, color='k', lw=0.8, ls='--', zorder=12)
    latlon(ax, range(0, 4), range(41, 43))
    letter(ax, 'b')
    cb = fig.colorbar(
        m,
        ax=ax,
        orientation='horizontal',
        fraction=0.04,
        pad=0.07,
        aspect=30,
        ticks=[1, 10, 50, 100, 200, 400],
        extend='max',
    )
    cb.set_label('48 h rainfall (mm)', fontsize=6.3)
    cb.ax.tick_params(labelsize=6)
    # (c)
    ax, im = surface_panel(fig, grid[1, 0], T0)
    latlon(ax, [1.5, 2.5], [41, 42], left=False, bottom=True, right=True)
    letter(ax, 'c')
    ax.text(
        0.97,
        0.03,
        utc(T0),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=6,
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1),
        zorder=20,
    )
    cb = fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        fraction=0.04,
        pad=0.07,
        aspect=30,
        extend='max',
        ticks=RAIN_LEVELS,
        format='%g',
    )
    cb.set_label('2 h rainfall (mm)', fontsize=6.3)
    cb.ax.tick_params(labelsize=6)
    # (d) 10 hPa layer means; winds as 50 hPa vector means
    from metpy.plots import SkewT

    sk = SkewT(fig, rotation=45, aspect=80, subplot=grid[1, 1])
    ax = sk.ax
    with xr.open_dataset(ERA5 / 'temperature_october_1991_2020.nc') as ds:
        cl = ds.t.mean('valid_time').interp(latitude=SITE[0], longitude=SITE[1])
        sk.plot(
            cl.pressure_level.values, cl.values - 273.15, color='#7a5c2e', lw=1.1, ls=':', label='ERA5 October mean'
        )
    col = '#a8141e'
    raw = O.sounding(OBS / 'sounding' / '26100400.txt')
    b = O.layer_means(raw, 10.0)
    sk.plot(b.p.values, b.t.values, color=col, lw=1.3, label='Temperature')
    up = b.p.values >= 250  # humidity not used above 250 hPa
    sk.plot(b.p.values[up], b.td.values[up], color=col, lw=1.1, ls='--', label='Dew point')
    w = O.layer_means(raw, 50.0)
    w = w[(w.p <= 1000) & (w.p >= 150)]
    sk.plot_barbs(
        w.p.values,
        w.u.values / KT,
        w.v.values / KT,
        xloc=0.95,
        color=col,
        length=5.2,
        linewidth=0.6,
        barb_increments={'half': 5, 'full': 10, 'flag': 50},
    )
    sk.plot_dry_adiabats(lw=0.35, alpha=0.35)
    sk.plot_moist_adiabats(lw=0.35, alpha=0.35)
    sk.plot_mixing_lines(lw=0.35, alpha=0.3)
    ax.set_ylim(1030, 100)
    ax.set_xlim(-25, 35)
    ax.set_yticks(
        [1000, 850, 700, 500, 400, 300, 200, 150, 100], ['1000', '850', '700', '500', '400', '300', '200', '150', '100']
    )
    ax.minorticks_off()
    ax.grid(color='.88', lw=0.4)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (hPa)')
    letter(ax, 'd', x=0.93)
    ax.legend(
        fontsize=5.8,
        loc='lower left',
        frameon=True,
        framealpha=0.9,
        edgecolor='none',
        handlelength=1.8,
        labelspacing=0.2,
        borderpad=0.3,
    )
    save(fig, FIG / 'fig1_observations.png')
    pw = {
        n: O.precipitable_water(O.sounding(OBS / 'sounding' / f'{n}.txt'))
        for n in ('26100300', '26100400')
        if (OBS / 'sounding' / f'{n}.txt').exists()
    }
    print('precipitable water (mm):', {k: round(v, 1) for k, v in pw.items()})


def rings(ax, km=(50, 100, 150)):
    g = pyproj.Geod(ellps='WGS84')
    az = np.arange(361)
    for d in km:
        lo, la, _ = g.fwd(np.full(361, GELIDA[0]), np.full(361, GELIDA[1]), az, np.full(361, d * 1e3))
        ax.plot(lo, la, color='#333', lw=0.4, ls=':', transform=PC, zorder=9)
    ax.plot(*GELIDA, marker='^', ms=4, mfc='white', mec='k', mew=0.7, transform=PC, zorder=11)


def figS1():
    """(a) OPERA maximum reflectivity; Gelida 0.5 deg (b) reflectivity and (c) radial velocity where reflectivity >= 15 dBZ."""
    REXT, OEXT = [-0.05, 3.85, 39.95, 43.0], [-0.5, 6.0, 38.9, 44.0]
    lon, lat, f = sweep(OBS / 'radar' / 'esgld@20261004T0007@0.5_1.5@DBZH_VRADH.h5')
    lon, lat = lon.astype('f4'), lat.astype('f4')  # single precision, as in the paper figure
    ve = np.where(np.isfinite(f['VRADH']) & (f['DBZH'] >= 15), f['VRADH'], np.nan)
    olo, ola, oz, _ = composite(OBS / 'radar' / 'OPERA@20261004T0005@0@DBZH.h5', (-1.5, 6.0, 39.0, 44.0))
    fig = plt.figure(figsize=(W, 2.7))
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    P = [
        (olo, ola, np.where(oz >= 5, oz, np.nan), OEXT, 'viridis', 5, 55),
        (lon, lat, np.where(f['DBZH'] >= 5, f['DBZH'], np.nan), REXT, 'viridis', 5, 55),
        (lon, lat, ve, REXT, 'RdBu_r', -30, 30),
    ]
    for k, (lo, la, z, ext, cmap, vmin, vmax) in enumerate(P):
        ax = fig.add_subplot(gs[0, k], projection=REGIONAL)
        ax.set_extent(ext, crs=PC)
        coast(ax)
        rings(ax)
        ax.plot(*OPOUL, marker='^', ms=4, mfc='white', mec='k', mew=0.7, transform=PC, zorder=11)
        im = ax.pcolormesh(
            lo,
            la,
            np.ma.masked_invalid(z),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=PC,
            shading='auto',
            rasterized=True,
            zorder=2,
        )
        letter(ax, 'abc'[k])
        if k == 0:  # radar names in panel a
            for (x, y), nm in ((GELIDA, 'Gelida'), (OPOUL, 'Opoul')):
                ax.text(
                    x + 0.15,
                    y,
                    nm,
                    transform=PC,
                    fontsize=5.5,
                    va='center',
                    zorder=12,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
                )
        latlon(
            ax,
            np.arange(-2, 8, 2 if ext is OEXT else 1),
            np.arange(38, 46, 2 if ext is OEXT else 1),
            left=(k == 0 or ext is OEXT),
            bottom=True,
        )
        lab, e = ('Reflectivity (dBZ)', 'max') if k < 2 else ('Radial velocity (m s$^{-1}$)', 'both')
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.16, extend=e)
        cb.set_label(lab, fontsize=6)
        cb.ax.tick_params(labelsize=6)
    save(fig, FIG / 'figS1_observations.png')


def figS2():
    """Observed and CURR CTRL 48 h rainfall on the 3 km verification grid, and their difference."""
    from matplotlib.colors import BoundaryNorm

    V = xr.open_dataset(DATASET / 'verification_region.nc')
    with xr.open_dataset(REF / 'precip_obs_common_3km.nc') as r:
        obs48 = r.observed_48h_mm.values
        fixed = r.fixed_all_model_comparison_mask.values > 0
    Wt = load_npz(DATASET / 'regrid_model_to_verification.npz')
    with xr.open_dataset(DATASET / 'rainfall/curr_ctrl.nc') as d:
        model = (Wt @ d.rain.values.sum(0).astype('f8').ravel()) / np.maximum(np.asarray(Wt.sum(1)).ravel(), 1e-12)
    valid = fixed & (V.mask.values > 0)
    obs = np.where(valid, obs48, np.nan)
    model = np.where(valid, model.reshape(valid.shape), np.nan)
    diff = model - obs
    cmap, norm = rain_cmap()
    dnorm = BoundaryNorm([-200, -100, -50, -25, -10, 10, 25, 50, 100, 200], plt.get_cmap('BrBG').N, extend='both')
    fig = plt.figure(figsize=(W, 2.8))
    gs = fig.add_gridspec(1, 3, wspace=0.28, bottom=0.25)
    axs, ims = [], []
    for j, (field, title) in enumerate(zip((obs, model, diff), ('Observed', 'CURR CTRL', 'CURR CTRL minus observed'))):
        ax = fig.add_subplot(gs[0, j], projection=REGIONAL)
        ax.set_extent([0.15, 3.35, 40.55, 42.85], crs=PC)
        im = ax.pcolormesh(
            V.lon,
            V.lat,
            field,
            transform=PC,
            cmap=cmap if j < 2 else 'BrBG',
            norm=norm if j < 2 else dnorm,
            shading='auto',
            rasterized=True,
        )
        coast(ax, '10m')
        latlon(ax, range(1, 4), range(41, 43), left=True)
        ax.set_title(title, fontsize=7)
        letter(ax, 'abc'[j])
        axs.append(ax)
        ims.append(im)
    for a in axs:
        a.apply_aspect()
    p0, p1, p2 = (a.get_position() for a in axs)
    y0, hc = p0.y0 - 0.17, 0.035  # a and b share one colour bar
    for (x0, x1), im, ex, tk, lab in (
        (
            (p0.x0 + 0.2 * p0.width, p1.x1 - 0.2 * p1.width),
            ims[0],
            'max',
            [1, 10, 50, 100, 200, 400],
            '48 h rainfall (mm)',
        ),
        ((p2.x0, p2.x1), ims[2], 'both', [-100, -25, 25, 100], 'Difference (mm)'),
    ):
        cb = fig.colorbar(im, cax=fig.add_axes([x0, y0, x1 - x0, hc]), orientation='horizontal', extend=ex, ticks=tk)
        cb.set_label(lab, fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    save(fig, FIG / 'figS2_verification.png')


FIGS = {'1': fig1, 'S1': figS1, 'S2': figS2}

if __name__ == '__main__':
    for k in sys.argv[1:] or FIGS:
        FIGS[k]()
