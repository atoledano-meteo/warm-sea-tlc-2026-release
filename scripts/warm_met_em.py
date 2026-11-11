"""WPS met_em files of +2K-ALL (Text S1): atmosphere, land and sea 2 K warmer at constant relative humidity.

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
