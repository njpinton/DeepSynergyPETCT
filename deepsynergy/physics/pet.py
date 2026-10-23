"""PET data model and attenuation-corrected MLEM.

Measurement model: y ~ Poisson(a * A x + r), where A is the PET projector, a = exp(-R mu511)
are the attenuation factors and r is a (known) mean background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def simulate_pet(x: np.ndarray, projector, att: np.ndarray, rng: np.random.Generator,
                 background: float = 0.0) -> np.ndarray:
    ybar = att * projector.forward(x) + background
    return rng.poisson(np.clip(ybar, 0.0, None)).astype(np.float64)


@dataclass
class PETProblem:
    y: np.ndarray
    projector: object
    att: np.ndarray
    background: float = 0.0

    def __post_init__(self):
        # sensitivity image A^T a
        self.sens = self.projector.back(self.att)

    def em_step(self, x: np.ndarray) -> np.ndarray:
        """One attenuation-corrected MLEM update."""
        ybar = self.att * self.projector.forward(x) + self.background
        ratio = np.divide(self.y, ybar, out=np.zeros_like(ybar), where=ybar > 0)
        num = self.projector.back(self.att * ratio)
        return np.divide(x * num, self.sens, out=np.zeros_like(x), where=self.sens > 0)

    def mlem(self, x: np.ndarray, n_iter: int) -> np.ndarray:
        for _ in range(n_iter):
            x = self.em_step(x)
        return x

    def prior_step(self, x: np.ndarray, x_gen: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """EM update fused with the quadratic prior (weight / 2) * ||x - x_gen||^2.

        Positive root of  weight x^2 + (sens - weight x_gen) x - sens x_em = 0,
        i.e. the minimiser of the EM surrogate plus the prior (thesis x-update).
        """
        x_em = self.em_step(x)
        b = weight * x_gen - self.sens
        c = self.sens * x_em
        sq = np.sqrt(b ** 2 + 4.0 * weight * c)
        # two algebraically equal forms of the root; pick the one without cancellation
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = (b + sq) / (2.0 * weight)
            neg = 2.0 * c / (sq - b)
        out = np.where(b > 0, pos, neg)
        return np.clip(np.nan_to_num(out), 0.0, None)
