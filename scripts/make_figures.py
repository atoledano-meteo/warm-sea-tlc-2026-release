"""Figures of the letter and the Supporting Information.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py 3 4 S3     # some of them

Model fields are read from the dataset; ERA5 and observations from the folders in config.yaml.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.config import DATASET, ERA5, OUTPUT
from warmsea.style import (
    plt,
    pe,
    matplotlib,
    cf,
    W,
    PC,
    REGIONAL,
    SSTC,
    GENESIS,
    rain_cmap,
    ir_cmap,
    WIND_CMAP,
    WIND_NORM,
    letter,
    utc,
    latlon,
    coast,
    boxline,
    outline,
    save,
)
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

FIG = OUTPUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
T0 = pd.Timestamp('2026-10-04 00:00')


def rain(run):
    with xr.open_dataset(DATASET / 'rainfall' / f'{run}.nc') as d:
        return d.lon.values, d.lat.values, d.rain.values


# ------------------------------------------------------------------ Figure 2
def fig2():
    """(a) WRF domains and terrain; (b) SST anomaly removed in CLIM; (c) ERA5 upper levels and (d) low levels."""
    fig = plt.figure(figsize=(W, 5.1))
    grid = fig.add_gridspec(2, 2, top=0.97, bottom=0.08, wspace=0.16, hspace=0.3)
    G = xr.open_dataset(DATASET / 'grid.nc')
    d01 = (G.xlong_d01.values, G.xlat_d01.values)
    d02 = (G.xlong_d02.values, G.xlat_d02.values)
    # (a)
    ax = fig.add_subplot(grid[0, 0], projection=REGIONAL)
    ax.set_extent([-19.5, 19.5, 28.5, 50.5], crs=PC)
    lm, hg = G.landmask_d01.values, G.hgt_d01.values
    ax.pcolormesh(
        *d01,
        np.where(lm > 0.5, np.nan, 1),
        transform=PC,
        cmap=matplotlib.colors.ListedColormap(['#cfe3f1']),
        shading='auto',
        rasterized=True,
    )
    tm = ax.pcolormesh(
        *d01,
        np.where(lm > 0.5, hg, np.nan),
        transform=PC,
        vmin=0,
        vmax=2500,
        shading='auto',
        rasterized=True,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list('t', ['#e9ecd7', '#c9c2a3', '#a3896a', '#7a5f4a']),
    )
    coast(ax, '50m', lw=0.4)
    outline(ax, *d01, color='k', lw=0.9, zorder=10)
    outline(ax, *d02, color='k', lw=0.9, ls='--', zorder=10)
    ax.text(
        -13.5,
        46.6,
        'd01, 9 km',
        transform=PC,
        fontsize=6,
        zorder=11,
        path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
    )
    ax.text(-3.6, 43.6, 'd02, 3 km', transform=PC, fontsize=6, zorder=11)
    for x, y, t in ((-15.0, 38.0, 'Atlantic\nOcean'), (5.6, 38.6, 'Mediterranean\nSea')):
        ax.text(
            x,
            y,
            t,
            transform=PC,
            fontsize=6,
            ha='center',
            va='center',
            style='italic',
            color='#1f4e73',
            zorder=12,
            linespacing=1.0,
            path_effects=[pe.withStroke(linewidth=1.8, foreground='white')],
        )
    cb = fig.colorbar(tm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.11, aspect=30, extend='max')
    cb.set_label('Topography (m a.s.l.)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    latlon(ax, range(-20, 21, 10), range(30, 51, 10))
    letter(ax, 'a')
    # (b) 48 h mean (2200 UTC 2 to 2200 UTC 4 October) of CURR minus CLIM over the sea
    ax = fig.add_subplot(grid[0, 1], projection=REGIONAL)
    ax.set_extent([-5.5, 15.0, 35.3, 45.5], crs=PC)
    sea = (G.landmask_d01.values < 0.5) & (G.lakemask_d01.values < 0.5)
    sst = {
        s: xr.open_dataset(DATASET / 'sst' / f'{s}.nc').sst_d01.values[10:59].astype('f8').mean(0)
        for s in ('curr', 'clim')
    }
    m = ax.pcolormesh(
        *d01,
        np.where(sea, sst['curr'], np.nan) - np.where(sea, sst['clim'], np.nan),
        transform=PC,
        cmap='RdBu_r',
        norm=TwoSlopeNorm(0, -4, 4),
        shading='auto',
        rasterized=True,
    )
    ax.add_feature(cf.LAND.with_scale('50m'), color='#ecebe6', zorder=2)
    coast(ax, '50m')
    outline(ax, *d02, color='k', lw=0.8, ls='--', zorder=10)
    ax.text(
        -3.6,
        43.6,
        'd02, 3 km',
        transform=PC,
        fontsize=6,
        zorder=11,
        path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
    )
    boxline(ax, *GENESIS, color='k', lw=0.9, zorder=12)
    ax.text(
        2.4,
        40.3,
        'Balearic\nSea',
        transform=PC,
        fontsize=5.5,
        ha='center',
        va='center',
        style='italic',
        color='#1f2f45',
        zorder=12,
        linespacing=1.0,
        path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
    )
    cb = fig.colorbar(m, ax=ax, orientation='horizontal', fraction=0.05, pad=0.11, aspect=30)
    cb.set_label('1991–2020 SST anomaly (K)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    latlon(ax, range(0, 16, 5), range(36, 46, 3))
    letter(ax, 'b')
    # (c) 300 hPa PV, 500 hPa height, 250 hPa jet
    P = xr.open_dataset(ERA5 / 'pressure_levels_20261004_00.nc').sel(valid_time=T0)
    S = xr.open_dataset(ERA5 / 'single_levels_20261004_00.nc').sel(valid_time=T0)
    ax = fig.add_subplot(grid[1, 0], projection=REGIONAL)
    ax.set_extent([-28, 24, 29, 58], crs=PC)
    m = ax.pcolormesh(
        P.longitude,
        P.latitude,
        P.pv.sel(pressure_level=300) * 1e6,
        cmap='Purples',
        norm=BoundaryNorm(np.arange(0, 5.5, 0.5), 256, extend='max'),
        transform=PC,
        shading='auto',
        rasterized=True,
    )
    cs = ax.contour(
        P.longitude,
        P.latitude,
        P.z.sel(pressure_level=500) / 98.1,
        levels=np.arange(558, 600, 3),
        colors='k',
        linewidths=0.5,
        transform=PC,
    )
    ax.clabel(cs, fontsize=6, fmt='%d')
    jet = gaussian_filter(np.hypot(P.u.sel(pressure_level=250), P.v.sel(pressure_level=250)).values, 1.5)
    cj = ax.contour(
        P.longitude,
        P.latitude,
        jet,
        levels=[30, 40, 50],
        colors='#d95f02',
        linewidths=[0.8, 1.3, 1.8],
        transform=PC,
        zorder=7,
    )
    for t in ax.clabel(cj, fontsize=6, fmt='%d', inline_spacing=1):
        t.set_color('#d95f02')
        t.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])
    coast(ax, '50m', lw=0.4)
    cb = fig.colorbar(
        m, ax=ax, orientation='horizontal', fraction=0.05, pad=0.11, aspect=30, extend='max', ticks=range(0, 6)
    )
    cb.set_label('300 hPa potential vorticity (PVU)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    latlon(ax, range(-20, 21, 20), range(30, 61, 10))
    letter(ax, 'c')
    ax.set_title(utc(T0), fontsize=7)
    # (d) precipitable water, 850 hPa wind (where the surface pressure exceeds 850 hPa) and sea-level pressure
    ax = fig.add_subplot(grid[1, 1], projection=REGIONAL)
    ax.set_extent([-9, 13, 34, 47.5], crs=PC)
    m = ax.pcolormesh(
        S.longitude, S.latitude, S.tcwv, cmap='GnBu', vmin=10, vmax=50, transform=PC, shading='auto', rasterized=True
    )
    cs = ax.contour(
        S.longitude,
        S.latitude,
        gaussian_filter(S.msl.values / 100, 1.5),
        levels=np.arange(1008, 1036, 4),
        colors='k',
        linewidths=0.8,
        transform=PC,
        zorder=6,
    )
    for t in ax.clabel(
        cs,
        fontsize=6.5,
        fmt='%d',
        inline_spacing=2,
        manual=[REGIONAL.transform_point(x, y, PC) for x, y in ((6.5, 37.6), (-4.5, 44.6), (9, 45.5))],
    ):
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground='white')])
    above = (S.sp.interp(latitude=P.latitude, longitude=P.longitude, method='nearest') > 85000).values
    u8, v8 = [np.where(above, P[c].sel(pressure_level=850).values, np.nan) for c in ('u', 'v')]
    k = (slice(None, None, 5), slice(None, None, 5))
    LO, LA = np.meshgrid(P.longitude.values, P.latitude.values)
    q = ax.quiver(LO[k], LA[k], u8[k], v8[k], transform=PC, scale=230, width=0.004, headwidth=3.5, color='.1', zorder=8)
    ax.add_patch(
        plt.Rectangle(
            (0.66, 0.005),
            0.335,
            0.075,
            transform=ax.transAxes,
            facecolor='white',
            alpha=0.85,
            edgecolor='none',
            zorder=8.5,
        )
    )
    ax.quiverkey(
        q, 0.9, 0.04, 10, '10 m s$^{-1}$', labelpos='W', fontproperties={'size': 6}, coordinates='axes', zorder=9
    )
    ax.add_feature(cf.COASTLINE.with_scale('50m'), lw=0.45, color='#555', zorder=3)
    cb = fig.colorbar(m, ax=ax, orientation='horizontal', fraction=0.05, pad=0.11, aspect=30, extend='both')
    cb.set_label('Precipitable water (mm)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    latlon(ax, range(-5, 16, 5), range(35, 50, 5))
    letter(ax, 'd')
    ax.set_title(utc(T0), fontsize=7)
    save(fig, FIG / 'fig2_environment.png')


# ------------------------------------------------------------------ Figure 3
def exceed(v, thr):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.array([(v >= x).mean() for x in thr]) * 100


def fig3():
    """(a-c) 48 h rainfall of CTRL in CLIM, CURR and +2K; (d, e) exceedance curves of all members over the mapped area."""
    cmap, norm = rain_cmap()
    EXT = [-1.0, 7.6, 38.3, 44.5]
    V = xr.open_dataset(DATASET / 'verification_region.nc')
    fig = plt.figure(figsize=(W, 5.0))
    g0 = fig.add_gridspec(4, 1, height_ratios=[1, 0.045, 0.06, 0.95], hspace=0.3)
    gs = g0[0].subgridspec(1, 3, wspace=0.06)
    for j, (lab, sst) in enumerate((('CLIM', 'clim'), ('CURR', 'curr'), ('+2K', 'plus2k'))):
        ax = fig.add_subplot(gs[0, j], projection=REGIONAL)
        ax.set_extent(EXT, crs=PC)
        lo, la, h = rain(f'{sst}_ctrl')
        tot = h.sum(0)
        ax.pcolormesh(
            lo,
            la,
            np.where(tot >= 0.1, tot, np.nan),
            transform=PC,
            cmap=cmap,
            norm=norm,
            shading='auto',
            rasterized=True,
        )
        coast(ax, '50m')
        latlon(ax, range(0, 8, 2), range(39, 45, 2), left=(j == 0))
        if j == 1:
            ax.contour(
                V.lon,
                V.lat,
                V.mask.astype(float),
                levels=[0.5],
                colors='k',
                linewidths=1.1,
                linestyles='--',
                transform=PC,
                zorder=6,
            )
        ax.set_title(lab, fontsize=7.5, color=SSTC[sst])
        letter(ax, 'abc'[j])
    cax = fig.add_subplot(g0[1].subgridspec(1, 5)[0, 1:4])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm, cmap),
        cax=cax,
        orientation='horizontal',
        ticks=[1, 10, 50, 100, 200, 400],
        extend='max',
    )
    cb.set_label('48 h rainfall (mm)', fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    gb = g0[3].subgridspec(1, 2, wspace=0.28)
    ad, ae = fig.add_subplot(gb[0, 0]), fig.add_subplot(gb[0, 1])
    T48 = np.r_[np.arange(1, 50, 1), np.arange(50, 601, 5)]
    T1 = np.r_[np.arange(0.5, 10, 0.5), np.arange(10, 151, 2)]
    LS = {'ctrl': '-', 'skb1': (0, (4, 2)), 'skb2': (0, (1, 1.4))}
    for sst in ('clim', 'curr', 'plus2k'):
        for mem in ('ctrl', 'skb1', 'skb2'):
            lo, la, h = rain(f'{sst}_{mem}')
            k = (lo >= EXT[0]) & (lo <= EXT[1]) & (la >= EXT[2]) & (la <= EXT[3])
            hh = h.reshape(len(h), -1)[:, k.ravel()]
            kw = dict(color=SSTC[sst], ls=LS[mem], lw=1.5 if mem == 'ctrl' else 1.0)
            ad.plot(T48, exceed(hh.sum(0), T48), **kw)
            ae.plot(T1, exceed(hh.ravel(), T1), **kw)
    for a, xl, L in ((ad, '48 h rainfall (mm)', 'd'), (ae, 'Hourly rainfall (mm h$^{-1}$)', 'e')):
        a.set_yscale('log')
        a.set_xscale('log')
        a.grid(color='.9', which='major', lw=0.5)
        a.grid(color='.96', which='minor', lw=0.35)
        a.set_xlabel(xl, fontsize=6.5)
        a.tick_params(labelsize=6)
        letter(a, L, x=0.95)
        a.set_ylabel('Area (%)' if L == 'd' else 'Hourly grid-cell values (%)', fontsize=6.5)
    ad.set_ylim(0.001, 100)
    ae.set_ylim(0.0001, 10)
    ad.set_yticks([0.001, 0.01, 0.1, 1, 10, 100], ['0.001', '0.01', '0.1', '1', '10', '100'])
    ae.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10], ['0.0001', '0.001', '0.01', '0.1', '1', '10'])
    ad.set_xlim(1, 600)
    ae.set_xlim(0.5, 150)
    ad.set_xticks([1, 10, 50, 100, 200, 500], ['1', '10', '50', '100', '200', '500'])
    ae.set_xticks([0.5, 1, 5, 10, 50, 100], ['0.5', '1', '5', '10', '50', '100'])
    L2 = lambda **k: plt.Line2D([], [], **k)
    h = [
        L2(color=SSTC['clim'], lw=1.4),
        L2(color=SSTC['curr'], lw=1.4),
        L2(color=SSTC['plus2k'], lw=1.4),
        L2(color='.4', ls='-', lw=1.5),
        L2(color='.4', ls=(0, (4, 2)), lw=1.0),
        L2(color='.4', ls=(0, (1, 1.4)), lw=1.0),
    ]
    ad.legend(
        h,
        ['CLIM', 'CURR', '+2K', 'CTRL', 'SKB1', 'SKB2'],
        loc='lower left',
        frameon=False,
        fontsize=5.8,
        handlelength=2.2,
        ncol=2,
    )
    save(fig, FIG / 'fig3_rainfall.png')


# ------------------------------------------------------------------ Figures 4 and S4
MAPS = {
    'curr_skb1': '2026100403',
    'plus2k_skb1': '2026100417',
    'plus2k_ctrl': '2026100406',
    'plus2k_skb2': '2026100322',
}


def draw_section(ax, top, S, title, color, left=True, right=True):
    x, zk = S.x.values, S.z.values / 1e3
    X2, Z2 = np.meshgrid(x, zk)
    m = ax.contourf(X2, Z2, S.ta, levels=np.arange(-4, 4.01, 0.5), cmap='RdBu_r', extend='both')
    ax.contourf(X2, Z2, S.cond, levels=[0.1, 50], colors=['#9a9a9a'], alpha=0.2)
    ax.contour(X2, Z2, S.cond, levels=[0.1], colors='#5f5c55', linewidths=0.45)
    cs = ax.contour(X2, Z2, S.vr, levels=[-30, -20, -10, 10, 20, 30], colors='k', linewidths=0.8)
    ax.clabel(cs, fontsize=6, fmt='%d', inline_spacing=1)
    ax.fill_between(x, 0, S.hgt / 1e3, color='#3d3b36', zorder=5)
    ax.set_ylim(0, 12)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel('Distance from the center, west → east (km)')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    top.bar(x, S.rain, width=2.4, color='#2a78d6', alpha=0.8)
    top.set_ylim(0, 40)
    t2 = top.twinx()
    t2.plot(x, S.slp, color='k', lw=1.1)
    t2.set_ylim(998, 1020)
    top.tick_params(labelbottom=False, labelsize=6, labelleft=left)
    t2.tick_params(labelsize=6, labelright=right)
    if left:
        top.set_ylabel('Rain\n(mm h$^{-1}$)', fontsize=6, color='#2a78d6')
    if right:
        t2.set_ylabel('Pressure\n(hPa)', fontsize=6)
    t2.spines['right'].set_visible(True)
    top.set_title(title, loc='left', fontsize=7.5, fontweight='bold', color=color)
    return m


def section_numbers(S):
    """Numbers quoted in Section 3.4: Tv anomaly of the central columns (|x| <= 20 km), largest storm-relative
    meridional wind within 100 km at each height, and position of the heaviest rain."""
    x, z, ta, vr = S.x.values, S.z.values, S.ta.values, S.vr.values
    c = np.abs(x) <= 20
    vmax = np.nanmax(np.abs(vr[:, np.abs(x) <= 100]), 1)
    k = int(np.nanargmax(vmax))
    lay = lambda a, b: float(np.nanmean(ta[(z >= a) & (z <= b)][:, c]))
    return dict(
        ta_0_1km=lay(0, 1000),
        ta_3_9km=lay(3000, 9000),
        ta_max_center=float(np.nanmax(ta[:, c])),
        z_ta_max_km=float(z[np.nanargmax(np.nanmax(ta[:, c], 1))] / 1e3),
        vr_max=float(vmax[k]),
        z_vr_max_km=float(z[k] / 1e3),
        vr_1km=float(vmax[np.argmin(np.abs(z - 1000))]),
        rain_max_x_km=float(x[np.nanargmax(S.rain.values)]),
    )


def fig4(rows=(('curr_skb1', 'CURR'), ('plus2k_skb1', '+2K')), sections=True, name='fig4_structure'):
    """Maps of simulated 10.5 um brightness temperature, reflectivity and 10 m wind with sea-level pressure; for Figure 4
    also the 5 h mean sections."""
    n = len(rows)
    if sections:
        fig = plt.figure(figsize=(W, 2.4 * n + 3.4))
        g0 = fig.add_gridspec(3, 1, height_ratios=[0.85 * n, 0.07, 1.15], hspace=0.34)
        gmap = g0[0]
    else:
        fig = plt.figure(figsize=(W, 2.6 * n + 0.4))
        gmap = fig.add_gridspec(1, 1, bottom=0.12)[0]
    gm = gmap.subgridspec(n, 3, wspace=0.06, hspace=0.2)
    cmap, norm = ir_cmap()
    L = iter('abcdefghijk')
    for i, (run, lab) in enumerate(rows):
        F = xr.open_dataset(DATASET / 'maps' / f'{run}_{MAPS[run]}.nc')
        t = pd.Timestamp(F.attrs['time'])
        xc, yc = F.attrs['centre_lon'], F.attrs['centre_lat']
        ext = [xc - 2.9, xc + 2.9, yc - 2.2, yc + 2.2]
        for j in range(3):
            ax = fig.add_subplot(gm[i, j], projection=REGIONAL)
            ax.set_extent(ext, crs=PC)
            if j == 0:
                mm = ax.pcolormesh(
                    F.lon_bt, F.lat_bt, F.bt, transform=PC, cmap=cmap, norm=norm, shading='auto', rasterized=True
                )
                cc = '#ff2fb3'
                if n > 1:
                    ax.text(
                        -0.25,
                        0.5,
                        lab,
                        transform=ax.transAxes,
                        rotation=90,
                        ha='right',
                        va='center',
                        fontsize=6.8,
                        color=SSTC['curr' if run.startswith('curr') else 'plus2k'],
                        fontweight='bold',
                    )
            elif j == 1:
                mm = ax.pcolormesh(
                    F.lon,
                    F.lat,
                    F.dbz.where(F.dbz > 5),
                    transform=PC,
                    cmap='viridis',
                    vmin=5,
                    vmax=55,
                    shading='auto',
                    rasterized=True,
                )
                cc = '#222'
            else:
                mm = ax.pcolormesh(
                    F.lon,
                    F.lat,
                    np.hypot(F.u, F.v),
                    transform=PC,
                    cmap=WIND_CMAP,
                    norm=WIND_NORM,
                    shading='auto',
                    rasterized=True,
                )
                cc = '#3a2f22'
                cs = ax.contour(
                    F.lon,
                    F.lat,
                    F.p,
                    levels=np.arange(990, 1025, 2),
                    colors='#202020',
                    linewidths=0.5,
                    transform=PC,
                    zorder=9,
                )
                kx = 111.2 * np.cos(np.radians(yc))
                ax.plot([xc - 150 / kx, xc + 150 / kx], [yc] * 2, transform=PC, color='k', lw=0.6, ls=':', zorder=13)
                for t_ in ax.clabel(
                    cs, levels=[l for l in cs.levels if l % 4 == 0], fontsize=5.2, fmt='%d', inline_spacing=1
                ):
                    t_.set_color('#111111')
                    t_.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])
                    t_.set_zorder(10)
            coast(ax, color=cc if j < 2 else '#8a8780', lw=0.6 if j < 2 else 1.1)
            latlon(ax, np.arange(-4, 12, 2), np.arange(36, 47, 1), left=(j == 0), bottom=(i == n - 1), right=(j == 2))
            if i == 0 and sections:  # Figure 4 only; in Figure S4 the colorbars name the fields
                ax.set_title(
                    ('Simulated 10.5 µm', 'Reflectivity', '10 m wind speed, sea-level pressure')[j], fontsize=7
                )
            ax.text(
                0.97,
                0.03,
                utc(t),
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=5.8,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1),
                zorder=20,
            )
            letter(ax, next(L))
            if i == n - 1:
                lab_, ticks = (
                    ('Brightness temperature (°C)', range(-80, 41, 40)),
                    ('Reflectivity (dBZ)', None),
                    ('10 m wind speed (m s$^{-1}$)', [0, 10, 20, 30]),
                )[j]
                if sections:
                    cb = fig.colorbar(
                        mm,
                        cax=ax.inset_axes([0, -0.16, 1, 0.045]),
                        orientation='horizontal',
                        ticks=ticks,
                        format=('%g' if j == 2 else None),
                    )
                else:
                    cb = fig.colorbar(
                        mm,
                        cax=ax.inset_axes([0, -0.24, 1, 0.045]),
                        orientation='horizontal',
                        ticks=ticks,
                        format=('%g' if j == 2 else None),
                    )
                cb.set_label(lab_, fontsize=7.5 if sections else 6)
                cb.ax.tick_params(labelsize=6.5 if sections else 5.8)
    if not sections:
        save(fig, FIG / f'{name}.png')
        return
    gs = g0[2].subgridspec(2, 3, height_ratios=[0.28, 1], width_ratios=[1, 1, 0.035], hspace=0.08, wspace=0.12)
    numbers = {}
    for jcol, (run, lab, col) in enumerate(
        (('curr_skb1', f'{next(L)}  CURR', SSTC['curr']), ('plus2k_skb1', f'{next(L)}  +2K', SSTC['plus2k']))
    ):
        S = xr.open_dataset(DATASET / 'sections' / f'{run}.nc')
        t0, t1 = pd.Timestamp(S.attrs['start']), pd.Timestamp(S.attrs['end'])
        top = fig.add_subplot(gs[0, jcol])
        ax = fig.add_subplot(gs[1, jcol], sharex=top)
        m = draw_section(ax, top, S, lab, col, left=jcol == 0, right=jcol == 1)
        numbers[run] = section_numbers(S)
        ax.text(
            0.98,
            0.97,
            f'mean {t0:%H%M}–{t1:%H%M} UTC {t0.day} {t0:%B}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=6,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1),
            zorder=20,
        )
        if jcol == 0:
            ax.set_ylabel('Height (km)')
        else:
            ax.tick_params(labelleft=False)
    cb = fig.colorbar(m, cax=fig.add_subplot(gs[1, 2]), ticks=range(-4, 5))
    cb.set_label('$T_v$ anomaly (K)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    (OUTPUT / 'fig4_section_numbers.json').write_text(json.dumps(numbers, indent=1) + '\n')
    save(fig, FIG / f'{name}.png')


def figS4():
    fig4(rows=(('plus2k_ctrl', 'CTRL'), ('plus2k_skb2', 'SKB2')), sections=False, name='figS4_members')


# ------------------------------------------------------------------ Figure S3
def figS3():
    """(a) Tracks of the surface cyclones associated with the MCV, over the sea; (b, c) cyclone phase space;
    (d) mean vorticity profile within 50 km of the MCV center, 2200-0800 UTC."""
    H = pd.read_csv(DATASET / 'diagnostics/associated_cyclone_hourly.csv', parse_dates=['time'])
    RUNS = [
        ('curr_ctrl', SSTC['curr'], 'o'),
        ('curr_skb1', SSTC['curr'], 's'),
        ('plus2k_ctrl', SSTC['plus2k'], 'o'),
        ('plus2k_skb1', SSTC['plus2k'], 's'),
        ('plus2k_skb2', SSTC['plus2k'], '^'),
        ('plus2kall_ctrl', SSTC['plus2kall'], 'o'),
        ('plus2kall_skb1', SSTC['plus2kall'], 's'),
    ]
    fig = plt.figure(figsize=(W, 5.6))
    g = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.3, left=0.07, right=0.99, top=0.97, bottom=0.13)
    ax = fig.add_subplot(g[0, 0], projection=REGIONAL)
    ax.set_extent([1.2, 5.6, 40.2, 42.6], crs=PC)
    coast(ax, '10m')
    pa, pb, pc = fig.add_subplot(g[0, 1]), fig.add_subplot(g[1, 0]), fig.add_subplot(g[1, 1])
    for run, col, mk in RUNS:
        d = H[H.run == run].sort_values('time')
        if d.empty:
            continue
        ok = d.structural_coherence.astype(bool).values
        ax.plot(d.lon, d.lat, color=col, lw=0.7, transform=PC, zorder=3)
        ax.scatter(d.lon[ok], d.lat[ok], s=11, marker=mk, color=col, lw=0.4, transform=PC, zorder=4)
        ax.scatter(
            d.lon[~ok], d.lat[~ok], s=11, marker=mk, facecolor='white', edgecolor=col, lw=0.6, transform=PC, zorder=4
        )
        if run in ('curr_skb1', 'plus2k_ctrl', 'plus2k_skb1', 'plus2k_skb2'):
            sm = d[['B', 'mVTL', 'mVTU']].rolling(5, center=True, min_periods=3).mean()  # 5 h centred running means
            for a, (xk, yk) in ((pa, ('mVTL', 'B')), (pb, ('mVTL', 'mVTU'))):
                a.scatter(d[xk], d[yk], s=6, marker=mk, facecolor='none', edgecolor=col, lw=0.35, alpha=0.45, zorder=2)
                if len(d) >= 5:
                    a.plot(sm[xk], sm[yk], color=col, lw=1.1, ls='-', zorder=3)
                    a.plot(sm[xk].iloc[-1], sm[yk].iloc[-1], marker=mk, ms=4, color=col, zorder=4)
                    a.plot(sm[xk].iloc[0], sm[yk].iloc[0], marker=mk, ms=4, mfc='white', mec=col, mew=0.8, zorder=4)
    latlon(ax, range(2, 6), range(41, 43))
    letter(ax, 'a')
    pa.axvline(0, color='.5', lw=0.5)
    pa.axhline(10, color='.5', lw=0.5, ls='--')
    pa.axhline(-10, color='.5', lw=0.5, ls='--')
    pa.set(xlabel='$-V_T^L$ (m)', ylabel='$B$ (m)', ylim=(-16, 16))
    letter(pa, 'b')
    pb.axhline(0, color='.5', lw=0.5)
    pb.axvline(0, color='.5', lw=0.5)
    pb.set(xlabel='$-V_T^L$ (m)', ylabel='$-V_T^U$ (m)')
    letter(pb, 'c')
    for x_, y_, t_, ha in (
        (0.98, -7.5, 'Symmetric warm core', 'right'),
        (0.02, -7.5, 'Symmetric cold core', 'left'),
        (0.98, 13, 'Asymmetric warm core', 'right'),
        (0.02, 13, 'Asymmetric cold core', 'left'),
    ):
        pa.text(x_, y_, t_, transform=pa.get_yaxis_transform(), ha=ha, va='center', fontsize=5.5, color='.45', zorder=1)
    for x_, y_, t_, ha, va in (
        (0.98, 0.98, 'Deep warm core', 'right', 'top'),
        (0.98, 0.02, 'Shallow warm core', 'right', 'bottom'),
        (0.02, 0.02, 'Deep cold core', 'left', 'bottom'),
    ):
        pb.text(x_, y_, t_, transform=pb.transAxes, ha=ha, va=va, fontsize=5.5, color='.45', zorder=1)
    P = pd.read_csv(DATASET / 'diagnostics/mcv_vertical_profiles.csv', parse_dates=['time'])
    P = P[(P.time >= '2026-10-03 22') & (P.time <= '2026-10-04 08')]
    zc = [c for c in P.columns if c.isdigit()]
    zk = np.r_[0.01, np.array(zc, float) / 1000]
    for sst, slab in (('clim', 'CLIM'), ('curr', 'CURR'), ('plus2k', '+2K'), ('plus2kall', '+2K-ALL')):
        for mem, ls, mk in (('ctrl', '-', 'o'), ('skb1', '--', 's'), ('skb2', ':', '^')):
            d = P[P.run == f'{sst}_{mem}']
            pc.plot(
                np.r_[d.z10.mean(), d[zc].mean().values],
                zk,
                color=SSTC[sst],
                ls=ls,
                lw=1.0,
                marker=mk,
                ms=2.6,
                markevery=(1, 4),
                label=slab if mem == 'ctrl' else None,
            )
    pc.axvline(0, color='.5', lw=0.5)
    pc.set(xlabel='Relative vorticity ($f$)', ylabel='Height (km)', ylim=(0, 12), xlim=(-1, 9))
    style = [
        plt.Line2D([], [], color='.3', ls=ls, marker=mk, ms=2.6, lw=1, label=l)
        for ls, mk, l in (('-', 'o', 'CTRL'), ('--', 's', 'SKB1'), (':', '^', 'SKB2'))
    ]
    exps = [
        plt.Line2D([], [], color=SSTC[k], lw=1.5, label=l)
        for k, l in (('clim', 'CLIM'), ('curr', 'CURR'), ('plus2k', '+2K'), ('plus2kall', '+2K-ALL'))
    ]
    letter(pc, 'd')
    for a in (pa, pb, pc):
        a.tick_params(labelsize=6.5)
        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)
        a.grid(color='.92', lw=0.4)
    fig.legend(  # colors: experiment; markers and line styles: member
        handles=exps + style,
        loc='lower center',
        ncol=7,
        fontsize=6.2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    save(fig, FIG / 'figS3_cyclones.png')


FIGS = {'2': fig2, '3': fig3, '4': fig4, 'S3': figS3, 'S4': figS4}

if __name__ == '__main__':
    for k in sys.argv[1:] or FIGS:
        FIGS[k]()
