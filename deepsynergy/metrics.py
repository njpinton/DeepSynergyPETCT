"""Image quality metrics with an explicit data range (the reference image maximum)."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, img: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, img, data_range=float(ref.max())))


def ssim(ref: np.ndarray, img: np.ndarray) -> float:
    return float(structural_similarity(ref, img, data_range=float(ref.max())))


def report(ref: np.ndarray, img: np.ndarray) -> dict:
    return {"psnr": psnr(ref, img), "ssim": ssim(ref, img)}
