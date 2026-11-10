"""Download the ERA5 fields used in Figures 1d and 2c,d from the Copernicus Climate Data Store.

Needs a CDS account and the cdsapi package (https://cds.climate.copernicus.eu/how-to-api).

    python scripts/download_era5.py
"""

import sys
from pathlib import Path
import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warmsea.config import ERA5

REQUESTS = {
    'pressure_levels_20261004_00.nc': (
        'reanalysis-era5-pressure-levels',
        {
            'product_type': ['reanalysis'],
            'date': ['2026-10-04'],
            'time': ['00:00'],
            'variable': ['potential_vorticity', 'geopotential', 'u_component_of_wind', 'v_component_of_wind'],
            'pressure_level': ['250', '300', '500', '850'],
            'area': [60, -30, 28, 25],
        },
    ),
    'single_levels_20261004_00.nc': (
        'reanalysis-era5-single-levels',
        {
            'product_type': ['reanalysis'],
            'date': ['2026-10-04'],
            'time': ['00:00'],
            'variable': ['mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour'],
            'area': [60, -30, 28, 25],
        },
    ),
    'temperature_october_1991_2020.nc': (
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'product_type': ['monthly_averaged_reanalysis'],
            'month': ['10'],
            'time': ['00:00'],
            'year': [str(y) for y in range(1991, 2021)],
            'variable': ['temperature'],
            'area': [42, 1, 41, 3],
            'pressure_level': [
                '1000',
                '975',
                '950',
                '925',
                '900',
                '875',
                '850',
                '825',
                '800',
                '775',
                '750',
                '700',
                '650',
                '600',
                '550',
                '500',
                '450',
                '400',
                '350',
                '300',
                '250',
                '225',
                '200',
                '175',
                '150',
                '125',
                '100',
                '70',
                '50',
            ],
        },
    ),
}

if __name__ == '__main__':
    ERA5.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    for name, (dataset, request) in REQUESTS.items():
        if not (ERA5 / name).exists():
            c.retrieve(dataset, request | {'data_format': 'netcdf', 'download_format': 'unarchived'}, str(ERA5 / name))
