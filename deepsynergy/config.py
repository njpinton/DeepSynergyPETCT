"""Experiment configuration (JSON <-> dataclasses)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class ImageConfig:
    size: int = 128
    vox_mm: float = 3.9


@dataclass
class PETConfig:
    n_angles: int = 120
    fwhm_mm: float = 0.0
    scale: float = 1.0          # count level: y = Poisson(scale * att * R x)
    background: float = 0.0
    kvp: float = 120            # CT tube voltage used for the 511 keV conversion


@dataclass
class CTConfig:
    backend: str = "rotation"   # "rotation" (parallel beam, no extra deps) or "astra" (fan beam)
    n_angles: int = 120
    i0: float = 1e4             # incident photons per detector bin
    background: float = 0.0
    det_width_mm: float = 1.2   # astra only
    det_count: int = 750
    source_origin_mm: float = 600.0
    origin_det_mm: float = 600.0


@dataclass
class ReconConfig:
    init_iters: int = 30        # MLEM / PWLS iterations for the initial images (and baselines)
    outer_iters: int = 5        # alternating z / x updates
    z_fit_iters: int = 5        # outer iterations in which z is re-fitted
    pet_subiters: int = 30
    ct_subiters: int = 100
    eta: float = 0.5            # PET weight in the z-update (CT gets 1 - eta)
    beta_pet: float = 1.0
    beta_ct: float = 1.0
    stride: int = 16
    z_init_range: float = 3.0   # z initialised uniformly in [-r, r]
    lbfgs_max_iter: int = 1000
    lbfgs_factr: float = 10.0
    alpha: float = 0.0          # latent prior weight (alpha ||z||^2)


@dataclass
class Config:
    image: ImageConfig = field(default_factory=ImageConfig)
    pet: PETConfig = field(default_factory=PETConfig)
    ct: CTConfig = field(default_factory=CTConfig)
    recon: ReconConfig = field(default_factory=ReconConfig)
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return _from_dict(cls, json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def _from_dict(cls, data: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        kwargs[name] = _from_dict(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)
