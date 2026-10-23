"""2D tomographic projectors.

``RotationProjector`` is a pure NumPy/OpenCV parallel-beam projector (rotate, then sum),
ported from the thesis code (``radon2``/``bradon2``). Its back-projection smears and
rotates back, which makes it an approximate (not exact) adjoint of the forward projection.

``AstraFanBeamProjector`` wraps the ASTRA toolbox fan-beam geometry used for CT in the
thesis. ASTRA is optional: it is only imported when this class is instantiated.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    n0, n1 = img.shape
    m = cv2.getRotationMatrix2D(((n1 - 1) / 2.0, (n0 - 1) / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (n1, n0), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


class RotationProjector:
    """Parallel-beam projector A = scale * R * H, with H an optional Gaussian PSF.

    Args:
        img_size: number of pixels per side (square images).
        n_angles: number of projection angles over [0, 360).
        fwhm_mm: PSF full width at half maximum (0 disables the PSF).
        vox_mm: pixel size in mm.
        scale: multiplicative factor (sets the count level for PET).
    """

    def __init__(self, img_size: int, n_angles: int = 120, fwhm_mm: float = 0.0,
                 vox_mm: float = 1.0, scale: float = 1.0):
        self.img_size = img_size
        self.n_angles = n_angles
        self.fwhm_mm = fwhm_mm
        self.vox_mm = vox_mm
        self.scale = scale
        self.theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)
        self.sino_shape = (img_size, n_angles)

    def _psf(self, img: np.ndarray) -> np.ndarray:
        if self.fwhm_mm <= 0:
            return img
        sigma_vox = self.fwhm_mm * FWHM_TO_SIGMA / self.vox_mm
        return gaussian_filter(img, sigma_vox, mode="constant")

    def radon(self, img: np.ndarray) -> np.ndarray:
        """Line integrals R x (no PSF, no scale), e.g. for attenuation."""
        img = np.asarray(img, dtype=np.float32)
        sino = np.empty(self.sino_shape, dtype=np.float64)
        for i, t in enumerate(self.theta):
            sino[:, i] = _rotate(img, t).sum(axis=1)
        return sino

    def radon_t(self, sino: np.ndarray) -> np.ndarray:
        """Back-projection R^T y (no PSF, no scale)."""
        n = self.img_size
        img = np.zeros((n, n), dtype=np.float64)
        for i, t in enumerate(self.theta):
            smear = np.repeat(sino[:, i:i + 1].astype(np.float32), n, axis=1)
            img += _rotate(smear, -t)
        return img

    def forward(self, img: np.ndarray) -> np.ndarray:
        return self.scale * self.radon(self._psf(img))

    def back(self, sino: np.ndarray) -> np.ndarray:
        return self.scale * self._psf(self.radon_t(sino))


class AstraFanBeamProjector:
    """Fan-beam CT projector (ASTRA toolbox), geometry as in the thesis experiments.

    Distances are in mm and converted to pixel units with ``vox_mm``.
    """

    def __init__(self, img_size: int, n_angles: int = 120, vox_mm: float = 0.9765625,
                 det_width_mm: float = 1.2, det_count: int = 750,
                 source_origin_mm: float = 600.0, origin_det_mm: float = 600.0,
                 proj_type: str = "line_fanflat"):
        import astra  # optional dependency

        self._astra = astra
        self.img_size = img_size
        vol_geom = astra.create_vol_geom(img_size)
        proj_geom = astra.create_proj_geom(
            "fanflat", det_width_mm / vox_mm, det_count,
            np.linspace(0, 2 * np.pi, n_angles, endpoint=False),
            source_origin_mm / vox_mm, origin_det_mm / vox_mm)
        self.proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)

    def forward(self, img: np.ndarray) -> np.ndarray:
        sino_id, sino = self._astra.create_sino(img, self.proj_id)
        self._astra.data2d.delete(sino_id)
        return sino

    def back(self, sino: np.ndarray) -> np.ndarray:
        bp_id, img = self._astra.create_backprojection(sino, self.proj_id)
        self._astra.data2d.delete(bp_id)
        return img

    def radon(self, img: np.ndarray) -> np.ndarray:
        return self.forward(img)
