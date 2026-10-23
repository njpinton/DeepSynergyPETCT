"""Saving / loading trained decoders together with their normalisation constants."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from .prior import PatchPrior

META_FILE = "meta.json"


def save_model(vae, out_dir: str | Path, meta: dict) -> Path:
    """Save decoder + metadata. ``meta`` must contain ``pet_scale`` and ``ct_scale``
    (factors that map image values to the [0, 1] range the model was trained on)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vae.decoder.save(out / "decoder.keras")
    vae.encoder.save_weights(out / "encoder.weights.h5")
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    return out


def load_prior(model_dir: str | Path, pet_scale: float | None = None,
               ct_scale: float | None = None) -> PatchPrior:
    """Load a decoder as a PatchPrior.

    Supports this repo's format (``decoder.keras`` + ``meta.json``) and the thesis
    SavedModel format (``decoder/`` directory), for which the scales must be given.
    """
    d = Path(model_dir)
    meta = json.loads((d / META_FILE).read_text()) if (d / META_FILE).exists() else {}
    pet_scale = pet_scale if pet_scale is not None else meta.get("pet_scale")
    ct_scale = ct_scale if ct_scale is not None else meta.get("ct_scale")
    if pet_scale is None or ct_scale is None:
        raise ValueError(f"{d}: no {META_FILE}; pass pet_scale and ct_scale explicitly")
    path = d / "decoder.keras" if (d / "decoder.keras").exists() else d / "decoder"
    decoder = tf.keras.models.load_model(path, compile=False)
    return PatchPrior(decoder, pet_scale=pet_scale, ct_scale=ct_scale)
