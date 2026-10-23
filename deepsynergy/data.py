"""Synthetic paired PET/CT thorax phantoms and training-patch extraction.

The phantoms share their anatomy between modalities (body, lungs, spine, heart, lesions), which
is what the multi-branch prior learns to exploit. PET is returned normalised to [0, 1];
CT is returned in HU.
"""

from __future__ import annotations

import numpy as np

from .physics.attenuation import hu_to_mu
from .prior import PatchOperator

# (HU, PET activity) per tissue
TISSUES = {
    "air": (-1000.0, 0.0),
    "soft": (40.0, 0.12),
    "lung": (-800.0, 0.03),
    "bone": (700.0, 0.08),
    "heart": (40.0, 0.45),
}


def _ellipse(xx, yy, cx, cy, rx, ry, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    u = ((xx - cx) * c + (yy - cy) * s) / rx
    v = (-(xx - cx) * s + (yy - cy) * c) / ry
    return u ** 2 + v ** 2 <= 1.0


def thorax_phantom(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One random 2D thorax slice. Returns (pet in [0, 1], ct in HU)."""
    lin = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(lin, lin)
    j = lambda scale: rng.uniform(-scale, scale)  # noqa: E731  random jitter

    hu = np.full((size, size), TISSUES["air"][0])
    pet = np.zeros((size, size))

    def paint(mask, tissue, hu_jitter=15.0, pet_jitter=0.15):
        h, a = TISSUES[tissue]
        hu[mask] = h + j(hu_jitter)
        pet[mask] = a * (1 + j(pet_jitter))

    paint(_ellipse(xx, yy, j(0.03), j(0.03), 0.85 + j(0.05), 0.6 + j(0.05)), "soft")
    for side in (-1, 1):
        paint(_ellipse(xx, yy, side * (0.38 + j(0.04)), -0.05 + j(0.04), 0.27 + j(0.04),
                       0.38 + j(0.05), side * j(0.3)), "lung", hu_jitter=60)
    paint(_ellipse(xx, yy, 0.12 + j(0.05), 0.05 + j(0.05), 0.17 + j(0.03), 0.14 + j(0.03), j(0.5)), "heart")
    paint(_ellipse(xx, yy, j(0.02), 0.45 + j(0.03), 0.09 + j(0.01), 0.09 + j(0.01)), "bone", hu_jitter=100)
    paint(_ellipse(xx, yy, j(0.02), 0.45 + j(0.03), 0.04, 0.04), "soft")  # spinal canal

    # lesions: soft-tissue density inside a lung, hot on PET
    for _ in range(rng.integers(1, 3)):
        side = rng.choice([-1, 1])
        r = rng.uniform(0.03, 0.08)
        mask = _ellipse(xx, yy, side * (0.38 + j(0.12)), -0.05 + j(0.2), r, r * (1 + j(0.3)))
        hu[mask] = 30.0 + j(20)
        pet[mask] = rng.uniform(0.6, 1.0)

    return np.clip(pet, 0, 1), hu


def make_phantoms(n: int, size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs = [thorax_phantom(size, rng) for _ in range(n)]
    return np.stack([p for p, _ in pairs]), np.stack([c for _, c in pairs])


def make_training_patches(pet: np.ndarray, ct_hu: np.ndarray, vox_mm: float, patch: int,
                          stride: int, drop_empty: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract paired patches and normalise both modalities to [0, 1].

    PET is scaled by 1 / max and CT is converted to attenuation per pixel, then scaled by
    1 / max. The scales are returned in ``meta`` and must be stored with the model.
    """
    ct_mu = hu_to_mu(ct_hu, vox_mm)
    op = PatchOperator(pet.shape[1:], patch, stride)
    p_pet = np.concatenate([op.extract(img) for img in pet])
    p_ct = np.concatenate([op.extract(img) for img in ct_mu])
    if drop_empty:
        keep = (p_pet.max(axis=(1, 2)) > 0) | (p_ct.max(axis=(1, 2)) > 0)
        p_pet, p_ct = p_pet[keep], p_ct[keep]
    pet_scale = 1.0 / float(p_pet.max())
    ct_scale = 1.0 / float(p_ct.max())
    meta = {"pet_scale": pet_scale, "ct_scale": ct_scale, "patch": patch, "vox_mm": vox_mm}
    return (p_pet * pet_scale).astype(np.float32), (p_ct * ct_scale).astype(np.float32), meta
