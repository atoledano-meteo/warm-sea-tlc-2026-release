"""Small WRF helpers."""


def mslp(psfc, t2, z):
    """Sea-level pressure (hPa) from surface pressure (Pa), 2 m temperature (K) and terrain height (m), with a
    standard lapse rate of 6.5 K/km."""
    return psfc * (1 + 0.0065 * z / (t2 + 0.0065 * z * 0.5)) ** 5.257 / 100
