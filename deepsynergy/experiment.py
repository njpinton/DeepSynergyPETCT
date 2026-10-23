"""Glue: build projectors from a Config, simulate noisy data, set up the PET/CT problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .physics import (AstraFanBeamProjector, CTProblem, PETProblem, RotationProjector,
                      attenuation_factors, hu_to_mu, hu_to_mu511, mu_to_hu, simulate_ct, simulate_pet)


def build_projectors(cfg: Config):
    n, vox = cfg.image.size, cfg.image.vox_mm
    pet = RotationProjector(n, cfg.pet.n_angles, cfg.pet.fwhm_mm, vox, cfg.pet.scale)
    if cfg.ct.backend == "rotation":
        ct = RotationProjector(n, cfg.ct.n_angles, 0.0, vox, 1.0)
    elif cfg.ct.backend == "astra":
        ct = AstraFanBeamProjector(n, cfg.ct.n_angles, vox, cfg.ct.det_width_mm, cfg.ct.det_count,
                                   cfg.ct.source_origin_mm, cfg.ct.origin_det_mm)
    else:
        raise ValueError(f"unknown CT backend: {cfg.ct.backend}")
    return pet, ct


@dataclass
class Simulation:
    pet: PETProblem          # attenuation factors estimated from the reconstructed CT
    ct: CTProblem
    pet_true: np.ndarray
    mu_true: np.ndarray      # CT attenuation per pixel
    ct_init: np.ndarray      # PWLS CT used for attenuation correction


def simulate(cfg: Config, pet_true: np.ndarray, ct_hu: np.ndarray,
             rng: np.random.Generator) -> Simulation:
    """Simulate PET and CT data from a ground-truth pair and prepare the reconstruction problems.

    The PET data is generated with the true attenuation; the reconstruction uses attenuation
    derived from a PWLS reconstruction of the simulated CT, as in a real PET/CT workflow.
    """
    if pet_true.shape != (cfg.image.size,) * 2 or ct_hu.shape != pet_true.shape:
        raise ValueError(f"images must be {cfg.image.size}x{cfg.image.size}")
    vox = cfg.image.vox_mm
    pet_proj, ct_proj = build_projectors(cfg)

    mu_true = hu_to_mu(ct_hu, vox)
    y_ct = simulate_ct(mu_true, ct_proj, cfg.ct.i0, rng, cfg.ct.background)
    ct = CTProblem(y_ct, ct_proj, cfg.ct.i0, cfg.ct.background)

    att_true = attenuation_factors(hu_to_mu511(ct_hu, vox, cfg.pet.kvp), pet_proj)
    y_pet = simulate_pet(pet_true, pet_proj, att_true, rng, cfg.pet.background)

    ct_init = ct.pwls(np.ones_like(mu_true), cfg.recon.init_iters)
    att_rec = attenuation_factors(hu_to_mu511(mu_to_hu(ct_init, vox), vox, cfg.pet.kvp), pet_proj)
    pet = PETProblem(y_pet, pet_proj, att_rec, cfg.pet.background)
    return Simulation(pet=pet, ct=ct, pet_true=pet_true, mu_true=mu_true, ct_init=ct_init)
