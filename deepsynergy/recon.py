"""Synergistic PET/CT reconstruction with a patch-based multi-branch generative prior.

Alternating scheme (thesis Ch. 5, eqs. z-update / x-update; TRPMS 2025):

  z_p  <- argmin_z  eta ||G1(z) - s1 P_p x1||^2 + (1-eta) ||G2(z) - s2 P_p x2||^2 + alpha ||z||^2
  x1   <- argmin_x  PET Poisson loss(x)  + (beta_pet s1^2 / 2) sum_p ||P_p x - G1(z_p)/s1||^2   (EM surrogate)
  x2   <- argmin_x  CT WLS loss(x)       + (beta_ct  s2^2 / 2) sum_p ||P_p x - G2(z_p)/s2||^2   (separable surrogate)

The patch penalty equals (beta s^2 / 2) * D * ||x - x_gen||^2 + const, with D = P^T P 1 the
patch-overlap count and x_gen = P^T g / D, which gives the closed-form x-updates in
``PETProblem.prior_step`` and ``CTProblem.prior_step``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import ReconConfig
from .physics.ct import CTProblem
from .physics.pet import PETProblem
from .prior import LatentFit, PatchOperator, PatchPrior

log = logging.getLogger(__name__)


@dataclass
class ReconResult:
    pet: np.ndarray
    ct: np.ndarray
    pet_gen: np.ndarray | None = None
    ct_gen: np.ndarray | None = None
    z: np.ndarray | None = None
    history: list = field(default_factory=list)


def baseline(pet: PETProblem, ct: CTProblem, n_iter: int) -> ReconResult:
    """Independent MLEM (attenuation-corrected) and PWLS reconstructions."""
    n = pet.projector.img_size
    return ReconResult(pet=pet.mlem(np.ones((n, n)), n_iter), ct=ct.pwls(np.ones((n, n)), n_iter))


def joint_reconstruct(pet: PETProblem, ct: CTProblem, prior: PatchPrior, cfg: ReconConfig,
                      init: ReconResult | None = None, rng: np.random.Generator | None = None,
                      callback=None) -> ReconResult:
    """Run the alternating z / x scheme. ``callback(it, result)`` is called after every outer iteration."""
    rng = rng or np.random.default_rng(0)
    init = init or baseline(pet, ct, cfg.init_iters)
    x_pet, x_ct = init.pet.copy(), init.ct.copy()

    ops = PatchOperator(x_pet.shape, prior.patch, cfg.stride)
    w_pet = cfg.beta_pet * prior.pet_scale ** 2 * ops.overlap
    w_ct = cfg.beta_ct * prior.ct_scale ** 2 * ops.overlap
    fit = LatentFit(max_iter=cfg.lbfgs_max_iter, factr=cfg.lbfgs_factr, alpha=cfg.alpha)
    z = rng.uniform(-cfg.z_init_range, cfg.z_init_range, (ops.n_patches, prior.latent_dim))

    result = ReconResult(pet=x_pet, ct=x_ct)
    for it in range(cfg.outer_iters):
        t0 = time.perf_counter()
        if it < cfg.z_fit_iters:
            z = prior.fit_latents(ops.extract(x_pet), ops.extract(x_ct), z, cfg.eta, fit)
        g_pet, g_ct = prior.generate(z)
        x_pet_gen, x_ct_gen = ops.average(g_pet), ops.average(g_ct)

        for _ in range(cfg.pet_subiters):
            x_pet = pet.prior_step(x_pet, x_pet_gen, w_pet)
        for _ in range(cfg.ct_subiters):
            x_ct = ct.prior_step(x_ct, x_ct_gen, w_ct)

        result = ReconResult(pet=x_pet, ct=x_ct, pet_gen=x_pet_gen, ct_gen=x_ct_gen, z=z,
                             history=result.history)
        log.info("outer iteration %d/%d done in %.1fs", it + 1, cfg.outer_iters, time.perf_counter() - t0)
        if callback:
            callback(it, result)
    return result
