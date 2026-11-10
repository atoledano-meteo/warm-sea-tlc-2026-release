"""Common figure style: fonts and sizes (AGU), colours, map helpers."""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cf

plt.rcParams.update(
    {
        'font.family': 'Nimbus Sans',
        'font.size': 7.5,
        'axes.titlesize': 8,
        'axes.labelsize': 7.5,
        'legend.fontsize': 6.5,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Nimbus Sans',
        'mathtext.it': 'Nimbus Sans:italic',
        'mathtext.bf': 'Nimbus Sans:bold',
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)

W = 6.69  # AGU maximum figure width (170 mm), inches
PC = ccrs.PlateCarree()
REGIONAL = ccrs.Mercator(central_longitude=3)
SSTC = {'clim': '#2a78d6', 'curr': '#1a1a19', 'plus2k': '#d6362f', 'plus2kall': '#8c6d1f'}
GENESIS = (1.8, 3.4, 40.8, 41.9)  # lon0, lon1, lat0, lat1 (Section 2.1)
KT = 0.514444  # m/s per knot


def rain_cmap():
    bounds = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 35, 50, 65, 80, 100, 150, 200, 300, 400, 500]
    colors = [
        '#ffffff',
        '#d4f4fa',
        '#8fd3f2',
        '#4fb3ea',
        '#2f7fdc',
        '#1f4fc8',
        '#11877a',
        '#2fa64c',
        '#93c83e',
        '#f2e23a',
        '#f8b22c',
        '#f37a20',
        '#e2321f',
        '#a8141e',
        '#d23ea0',
        '#9b2fae',
        '#4f3a9e',
    ]
    cmap = ListedColormap(colors).with_extremes(over=colors[-1])
    return cmap, BoundaryNorm(bounds, cmap.N, clip=True)


def ir_cmap():
    """Enhanced infrared scale: magenta below -80 C, greys -80 to -70 C, colours -70 to -20 C, greys above -20 C."""
    nodes = [
        (-90, '#a21caf'),
        (-80, '#f0abfc'),
        (-80, '#d4d4d4'),
        (-70, '#1a1a1a'),
        (-70, '#4a0000'),
        (-60, '#e00000'),
        (-55, '#ff7a00'),
        (-50, '#ffe600'),
        (-45, '#8fe000'),
        (-40, '#00c800'),
        (-35, '#00309a'),
        (-30, '#1d6fe0'),
        (-20, '#b4f0ff'),
        (-20, '#ffffff'),
        (10, '#808080'),
        (40, '#000000'),
    ]
    lo, hi = nodes[0][0], nodes[-1][0]
    return LinearSegmentedColormap.from_list('ir', [((v - lo) / (hi - lo), c) for v, c in nodes]), Normalize(lo, hi)


WIND_CMAP = ListedColormap(
    [
        '#ffffff',
        '#e3eef8',
        '#bcd7ee',
        '#84b6df',
        '#4e93cc',
        '#2f9e6b',
        '#93c83e',
        '#f2e23a',
        '#f8a22c',
        '#e8551f',
        '#b3141e',
        '#7d0d5e',
        '#4f3a9e',
    ]
).with_extremes(over='#24164f')
WIND_NORM = BoundaryNorm([0, 4, 6, 8, 10, 12, 14, 17.2, 20, 22.5, 25, 28, 32.7, 40], WIND_CMAP.N)


def letter(ax, s, x=0.01, y=0.99):
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        fontweight='bold',
        zorder=30,
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
    )


def utc(t):
    """Time label as in the text: 0300 UTC 4 October."""
    return f'{t:%H%M} UTC {t.day} {t:%B}'


def latlon(ax, xs, ys, left=True, bottom=True, right=False, top=False):
    sides = {
        k: v
        for k, v, on in (('left', 'y', left), ('right', 'y', right), ('bottom', 'x', bottom), ('top', 'x', top))
        if on
    }
    g = ax.gridlines(
        crs=PC,
        xlocs=xs,
        ylocs=ys,
        draw_labels=sides or False,
        x_inline=False,
        y_inline=False,
        lw=0.3,
        color='.55',
        alpha=0.6,
        ls=':',
        zorder=9,
    )
    g.xlabel_style = g.ylabel_style = {'size': 6, 'color': '.25'}
    g.rotate_labels = False
    return g


def coast(ax, scale='10m', lw=0.6, color='#222'):
    ax.add_feature(cf.COASTLINE.with_scale(scale), lw=lw, color=color, zorder=8)
    ax.add_feature(cf.BORDERS.with_scale(scale), lw=0.35, color='#666', zorder=8)


def boxline(ax, lon0, lon1, lat0, lat1, **kw):
    x = np.r_[np.linspace(lon0, lon1, 30), np.full(30, lon1), np.linspace(lon1, lon0, 30), np.full(30, lon0)]
    y = np.r_[np.full(30, lat0), np.linspace(lat0, lat1, 30), np.full(30, lat1), np.linspace(lat1, lat0, 30)]
    ax.plot(x, y, transform=PC, **kw)


def outline(ax, lon, lat, **kw):
    x = np.r_[lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]]
    y = np.r_[lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]]
    ax.plot(x, y, transform=PC, **kw)


def save(fig, path):
    fig.savefig(path, dpi=450, facecolor='white', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('written', path)
