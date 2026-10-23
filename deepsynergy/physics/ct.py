"""CT data model and penalised weighted least squares (PWLS).

Measurement model: y ~ Poisson(I0 exp(-A mu) + b). The PWLS step uses the separable
quadratic surrogate with denominator D = A^T W A 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def simulate_ct(mu: np.ndarray, projector, i0: float, rng: np.random.Generator,
                background: float = 0.0) -> np.ndarray:
    ybar = i0 * np.exp(-projector.forward(mu)) + background
    return rng.poisson(ybar).astype(np.float64)


@dataclass
class CTProblem:
    y: np.ndarray
    projector: object
    i0: float
    background: float = 0.0

    def __post_init__(self):
        d = self.y - self.background
        valid = d > 0
        self.l = np.where(valid, np.log(self.i0 / np.where(valid, d, 1.0)), 0.0)
        self.w = np.where(valid, d ** 2 / np.where(self.y > 0, self.y, 1.0), 0.0)
        n = self.projector.img_size
        self.denom = self.projector.back(self.w * self.projector.forward(np.ones((n, n))))

    def wls_step(self, x: np.ndarray) -> np.ndarray:
        grad = self.projector.back(self.w * (self.projector.forward(x) - self.l))
        step = np.divide(grad, self.denom, out=np.zeros_like(grad), where=self.denom > 0)
        return np.clip(np.where(self.denom > 0, x - step, 0.0), 0.0, None)

    def pwls(self, x: np.ndarray, n_iter: int) -> np.ndarray:
        for _ in range(n_iter):
            x = self.wls_step(x)
        return x

    def prior_step(self, x: np.ndarray, x_gen: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """WLS surrogate step fused with (weight / 2) * ||x - x_gen||^2 (thesis x-update)."""
        x_rec = self.wls_step(x)
        den = self.denom + weight
        out = np.divide(self.denom * x_rec + weight * x_gen, den, out=np.zeros_like(x), where=den > 0)
        return np.clip(out, 0.0, None)
