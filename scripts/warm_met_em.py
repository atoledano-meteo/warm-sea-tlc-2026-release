"""Inputs of +2K-ALL (Schär et al., 1996): the WPS met_em files with the atmosphere, land and sea 2 K warmer.

In each met_em file (both domains, every hour, so initial and boundary conditions):
  TT                       +2 K at every level
  RH                       unchanged (specific humidity rises with temperature)
  GHT                      + (Rd * 2 K / g) ln(PSFC / p) above the surface (hydrostatic thickening); terrain unchanged
  PSFC, PMSL, winds        unchanged
  SKINTEMP, SST, soil      +2 K
The SST that WRF then reads (wrflowinp) is the +2K state of make_sst.py.

    python scripts/warm_met_em.py /path/to/met_em /path/to/met_em_plus2kall
"""

import shutil
import sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

DT, RD, G = 2.0, 287.04, 9.81
SURFACE = ['SKINTEMP', 'SST', 'ST000007', 'ST007028', 'ST028100', 'ST100289', 'SOILTEMP', 'ST']

if __name__ == '__main__':
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob('met_em.d0*.nc')):
        shutil.copy2(f, out / f.name)
        with Dataset(out / f.name, 'r+') as d:
            d['TT'][:] = d['TT'][:] + DT
            p, psfc = d['PRES'][:], d['PSFC'][:]
            dz = RD * DT / G * np.log(np.maximum(psfc[:, None], p) / p)
            dz[:, 0] = 0.0
            d['GHT'][:] = d['GHT'][:] + dz
            for v in SURFACE:
                if v in d.variables:
                    a = d[v][:]
                    d[v][:] = np.where(a > 0, a + DT, a)  # zero and missing fills kept
        print(f.name)
