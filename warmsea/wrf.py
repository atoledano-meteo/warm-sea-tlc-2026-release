"""Small WRF helpers. Thermodynamic conversions use MetPy."""

import numpy as np
from metpy import calc as mc
from metpy.units import units


def temperature(p, theta):
    """Temperature (K) from pressure (Pa) and potential temperature (K)."""
    return mc.temperature_from_potential_temperature(np.asarray(p) * units.Pa, np.asarray(theta) * units.K).m_as('K')


def virtual_temperature(t, r):
    """Virtual temperature (K) from temperature (K) and water vapor mixing ratio (kg/kg)."""
    return mc.virtual_temperature(np.asarray(t) * units.K, np.asarray(r) * units('kg/kg')).m_as('K')


def specific_humidity(r):
    """Specific humidity (kg/kg) from mixing ratio (kg/kg)."""
    return mc.specific_humidity_from_mixing_ratio(np.asarray(r) * units('kg/kg')).m_as('kg/kg')


def mslp(psfc, t2, z):
    """Sea-level pressure (hPa) from surface pressure (Pa), 2 m temperature (K) and terrain height (m), with a
    standard lapse rate of 6.5 K/km."""
    return psfc * (1 + 0.0065 * z / (t2 + 0.0065 * z * 0.5)) ** 5.257 / 100
