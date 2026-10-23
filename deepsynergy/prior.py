"""Patch-based multi-branch generative prior (thesis Ch. 5, TRPMS 2025).

R(x1, x2) = sum_p min_z  eta ||G1(z) - s1 P_p x1||^2 + (1 - eta) ||G2(z) - s2 P_p x2||^2 + alpha ||z||^2

where G = (G1, G2) is the two-branch decoder, P_p extracts patch p and s_k are the
normalisation constants used when training the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b


def patch_starts(n: int, patch: int, stride: int) -> list[int]:
    """Patch start positions along one axis; the last patch is aligned to the border so
    that every pixel is covered."""
    if patch > n:
        raise ValueError(f"patch size {patch} larger than image size {n}")
    starts = list(range(0, n - patch + 1, stride))
    if starts[-1] != n - patch:
        starts.append(n - patch)
    return starts


class PatchOperator:
    """Patch extraction P = [P_1; ...; P_P] and its adjoint P^T (reassembly by summation)."""

    def __init__(self, img_shape: tuple[int, int], patch: int, stride: int):
        self.img_shape = img_shape
        self.patch = patch
        self.rows = patch_starts(img_shape[0], patch, stride)
        self.cols = patch_starts(img_shape[1], patch, stride)
        self.n_patches = len(self.rows) * len(self.cols)
        self.overlap = self.adjoint(np.ones((self.n_patches, patch, patch)))  # P^T P 1

    def extract(self, img: np.ndarray) -> np.ndarray:
        p = self.patch
        return np.stack([img[r:r + p, c:c + p] for r in self.rows for c in self.cols])

    def adjoint(self, patches: np.ndarray) -> np.ndarray:
        p = self.patch
        img = np.zeros(self.img_shape)
        k = 0
        for r in self.rows:
            for c in self.cols:
                img[r:r + p, c:c + p] += patches[k]
                k += 1
        return img

    def average(self, patches: np.ndarray) -> np.ndarray:
        """P^T g / P^T P 1: the image that best fits the patches in the least-squares sense."""
        return self.adjoint(patches) / self.overlap


@dataclass
class LatentFit:
    max_iter: int = 1000
    factr: float = 10.0
    pgtol: float = 1e-12
    alpha: float = 0.0


class PatchPrior:
    """Wraps a two-branch decoder (latent -> [PET patch, CT patch]) for patch-wise z-fitting."""

    def __init__(self, decoder: tf.keras.Model, pet_scale: float, ct_scale: float,
                 batch_size: int = 4096):
        self.decoder = decoder
        self.latent_dim = int(decoder.input_shape[1])
        self.patch = int(decoder.output_shape[0][1])
        self.pet_scale = pet_scale
        self.ct_scale = ct_scale
        self.batch_size = batch_size

    @tf.function(reduce_retracing=True)
    def _value_and_grad(self, z, u1, u2, eta, alpha):
        with tf.GradientTape() as tape:
            tape.watch(z)
            g1, g2 = self.decoder(z, training=False)
            g1, g2 = g1[..., 0], g2[..., 0]
            f = (eta * tf.reduce_sum(tf.square(g1 - u1))
                 + (1.0 - eta) * tf.reduce_sum(tf.square(g2 - u2))
                 + alpha * tf.reduce_sum(tf.square(z)))
        return f, tape.gradient(f, z)

    def generate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decoded PET and CT patches, in image units (normalisation undone)."""
        pet, ct = [], []
        for i in range(0, len(z), self.batch_size):
            g1, g2 = self.decoder(z[i:i + self.batch_size].astype(np.float32), training=False)
            pet.append(g1.numpy()[..., 0])
            ct.append(g2.numpy()[..., 0])
        return np.concatenate(pet) / self.pet_scale, np.concatenate(ct) / self.ct_scale

    def fit_latents(self, pet_patches: np.ndarray, ct_patches: np.ndarray, z0: np.ndarray,
                    eta: float, opts: LatentFit) -> np.ndarray:
        """z-update: all patches at once. The objective is a sum of independent per-patch
        terms, so one L-BFGS over the stacked latents solves every patch problem."""
        u1 = tf.constant(np.clip(pet_patches * self.pet_scale, 0, None), tf.float32)
        u2 = tf.constant(np.clip(ct_patches * self.ct_scale, 0, None), tf.float32)
        shape = z0.shape
        eta_t, alpha_t = tf.constant(eta, tf.float32), tf.constant(opts.alpha, tf.float32)

        def objective(v):
            f, g = self._value_and_grad(tf.constant(v.reshape(shape), tf.float32), u1, u2, eta_t, alpha_t)
            return float(f), g.numpy().astype(np.float64).ravel()

        z, _, _ = fmin_l_bfgs_b(objective, z0.ravel().astype(np.float64), factr=opts.factr,
                                pgtol=opts.pgtol, maxiter=opts.max_iter, iprint=-1)
        return z.reshape(shape)
