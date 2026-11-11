# A Warmer Sea Turns a Mediterranean Convective Vortex Into a Tropical-Like Cyclone

Code and model configuration for the paper by Toledano, Udina, Sola, Bech, Busquets and Polls (submitted to *Geophysical Research Letters*). The model fields are in the companion dataset at Zenodo (https://doi.org/10.5281/zenodo.24110631).

## Contents

| Path                                    |                                                                                              |
| --------------------------------------- | -------------------------------------------------------------------------------------------- |
| `namelists/wps/`                      | WPS namelists for the ERA5T (hourly), IFS and GFS (3-hourly) initial and boundary conditions |
| `namelists/wrf/<run>/`                | WRF-ARW 4.6.1 namelist of each run; `iofields_d0*.txt` are shared by all runs              |
| `scripts/make_figures.py`             | Figures 2, 3, 4, S3 and S4, from the dataset and ERA5                                        |
| `scripts/make_observation_figures.py` | Figures 1, S1 and S2, from the observations                                                  |
| `scripts/make_tables.py`              | Tables S1 to S3, and the air–sea exchange changes                                            |
| `scripts/download_era5.py`            | ERA5 fields of Figures 1d and 2c,d                                                           |
| `scripts/make_sst.py`                 | CLIM and +2K SST from the current SST                                                        |
| `scripts/warm_met_em.py`              | +2K-ALL initial and boundary conditions                                                      |
| `scripts/simulate_satellite.py`       | Simulated 10.5 µm brightness temperature (RTTOV)                                            |
| `scripts/diagnose_vortices.py`        | MCV and surface-cyclone diagnostics                                                          |
| `scripts/air_sea_exchange.py`         | Latent heat flux and θe over the sea of the genesis region                                   |
| `scripts/export_dataset.py`           | How the dataset was extracted from the full WRF output                                       |
| `warmsea/`                            | Run names, readers and the vortex diagnostics                                                |

## Runs

Run names follow the paper: `<sst>_<member>` for the Morrison–YSU runs, with `tmynn_` in front for the
Thompson–MYNN runs.

|                                        |                                                 |
| -------------------------------------- | ----------------------------------------------- |
| `clim`, `curr`, `plus2k`         | climatological, current, and current + 2 K SST  |
| `plus2kall`                          | uniform warming of sea, land, and air (+2K-ALL) |
| `ctrl`, `skb1`, `skb2`           | deterministic run and the two SKEBS members     |
| `sel_<forcing>_<microphysics>_<pbl>` | other configurations of Table S1                |

In Table S1, ERA5T Morrison–YSU is `curr_ctrl` and ERA5T Thompson–MYNN is `tmynn_curr_ctrl`.

All runs start at 1200 UTC 2 October 2026. The SST experiments differ only in the SST that WRF reads from `wrflowinp` (dataset `sst/`); the SKEBS members differ from CTRL only in the `&stoch` section. +2K-ALL uses the +2K SST and the met_em files of `warm_met_em.py`.

## Reproducing the figures and tables

```bash
conda env create -f environment.yml
conda activate warm-sea-tlc
# download and unpack the dataset; set its folder in config.yaml
python scripts/download_era5.py        # needs a Copernicus Climate Data Store account
python scripts/make_figures.py
python scripts/make_tables.py
```

## Observations

Observations are not redistributed. `make_observation_figures.py` reads them from the `observations` folder of
`config.yaml`:

| Folder                  | Contents                                                                                                    | Source                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `mtg/20261004T0000/`  | FCI Level 1c body chunks 33–37 of the 0000 UTC 4 October cycle                                             | EUMETSAT Data Store, EO:EUM:DAT:0662                     |
| `radar/`              | Gelida volume `esgld@20261004T0007@0.5_1.5@DBZH_VRADH.h5` and composite `OPERA@20261004T0005@0@DBZH.h5` | EUMETNET OPERA Open Radar Data                           |
| `sounding/`           | Barcelona 1 s soundings `26100300.txt`, `26100400.txt`                                                  | Servei Meteorològic de Catalunya                        |
| `stations/`           | `hourly_surface.csv` (columns in `warmsea/observations.py`), and the XEMA pressure files                | AEMET, Météo-France, Servei Meteorològic de Catalunya |
| `icgc/`               | County boundaries (`*comarques*.shp`)                                                                     | Institut Cartogràfic i Geològic de Catalunya           |
| `rainfall_reference/` | 0.25 km radar–gauge analysis                                                                               | not distributed                                          |

## From the WRF output

The full output (about 100 GB per run) is not archived. With it, the dataset is rebuilt in this order:

```bash
python scripts/diagnose_vortices.py --runs RUNS --cache CACHE --out DIAGNOSTICS
python scripts/air_sea_exchange.py --runs RUNS --out DIAGNOSTICS
RTTOV_DIR=... python scripts/simulate_satellite.py --runs RUNS --out RTTOV
python scripts/export_dataset.py --runs RUNS --rttov RTTOV --diagnostics DIAGNOSTICS ... --out DATASET
```

## License

MIT (see `LICENSE`).
