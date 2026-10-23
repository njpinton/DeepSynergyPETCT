"""Attenuation conversions between Hounsfield units (HU), CT attenuation and 511 keV PET attenuation.

All ``mu`` arrays returned here are **per pixel** (cm^-1 multiplied by the pixel size in cm),
so that a sum along a ray directly gives the line integral used by the projectors.
"""

from __future__ import annotations

import numpy as np

# NIST XCOM mass attenuation coefficients (mu/rho, cm^2/g) at 80 keV.
MU_WATER_80KEV = 1.837e-01             # cm^-1 (density 1 g/cm^3)
MU_AIR_80KEV = 1.662e-01 * 1.205e-03   # cm^-1 (density 1.205e-3 g/cm^3)

# Bilinear HU -> mu(511 keV) model (Carney et al. 2006), as in OMEGA's attenuationCT_to_511.m.
_BILINEAR = {
    80: ([9.3e-5, 3.28e-5, 0.41e-5], [0.093, 0.093, 0.122]),
    100: ([9.3e-5, 4.00e-5, 0.50e-5], [0.093, 0.093, 0.128]),
    120: ([9.3e-5, 4.71e-5, 0.589e-5], [0.093, 0.093, 0.134]),
    140: ([9.3e-5, 5.59e-5, 0.698e-5], [0.093, 0.093, 0.142]),
}


def hu_to_mu(hu: np.ndarray, vox_mm: float, mu_water: float = MU_WATER_80KEV,
             mu_air: float = MU_AIR_80KEV) -> np.ndarray:
    """HU -> CT attenuation per pixel (clipped at 0)."""
    mu_cm = hu / 1000.0 * (mu_water - mu_air) + mu_water
    return np.clip(mu_cm, 0.0, None) * vox_mm / 10.0


def mu_to_hu(mu: np.ndarray, vox_mm: float, mu_water: float = MU_WATER_80KEV,
             mu_air: float = MU_AIR_80KEV) -> np.ndarray:
    """CT attenuation per pixel -> HU."""
    mu_cm = mu * 10.0 / vox_mm
    return 1000.0 * (mu_cm - mu_water) / (mu_water - mu_air)


def hu_to_mu511(hu: np.ndarray, vox_mm: float, kvp: float = 120) -> np.ndarray:
    """HU (2D or 3D) -> 511 keV attenuation per pixel, piecewise linear in HU."""
    if kvp in _BILINEAR:
        a, b = (np.asarray(v) for v in _BILINEAR[kvp])
    else:
        kvps = np.array(sorted(_BILINEAR))
        a = np.array([np.interp(kvp, kvps, [_BILINEAR[k][0][j] for k in kvps]) for j in range(3)])
        b = np.array([np.interp(kvp, kvps, [_BILINEAR[k][1][j] for k in kvps]) for j in range(3)])

    knots_hu = np.array([-1000.0, 0.0, 1000.0, 3000.0])
    knots_mu = np.array([b[0] - 1000 * a[0], b[1], b[1] + 1000 * a[1], b[1] + 1000 * a[1] + 2000 * a[2]])

    hu = np.asarray(hu, dtype=np.float64)
    mu_cm = np.interp(hu, knots_hu, knots_mu)
    # linear extrapolation beyond the outer knots
    lo, hi = hu < knots_hu[0], hu > knots_hu[-1]
    slope_lo = (knots_mu[1] - knots_mu[0]) / (knots_hu[1] - knots_hu[0])
    slope_hi = (knots_mu[3] - knots_mu[2]) / (knots_hu[3] - knots_hu[2])
    mu_cm[lo] = knots_mu[0] + slope_lo * (hu[lo] - knots_hu[0])
    mu_cm[hi] = knots_mu[-1] + slope_hi * (hu[hi] - knots_hu[-1])
    return np.clip(mu_cm, 0.0, None) * vox_mm / 10.0


def attenuation_factors(mu511: np.ndarray, projector) -> np.ndarray:
    """Survival probabilities exp(-R mu) on the PET sinogram grid.

    Uses the projector's plain line integrals (no PET count scale, no PSF): projecting
    mu with the PET count scale makes these factors vanish.
    """
    return np.exp(-np.clip(projector.radon(mu511), 0.0, None))
