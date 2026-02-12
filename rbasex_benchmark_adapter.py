from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _sanitize_xy(xy_data: np.ndarray) -> np.ndarray:
    if xy_data is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(xy_data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    arr = arr[:, :2]
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def _xy_to_centered_image(
    xy_data: np.ndarray,
    *,
    pixel_size_mm: float = 0.05,
    n_pixels: int = 512,
) -> np.ndarray:
    arr = _sanitize_xy(xy_data)
    image = np.zeros((int(n_pixels), int(n_pixels)), dtype=float)
    if arr.size == 0:
        return image

    px = float(max(pixel_size_mm, 1e-9))
    cx = (int(n_pixels) - 1) / 2.0
    cy = (int(n_pixels) - 1) / 2.0

    x_idx = np.rint(arr[:, 0] / px + cx).astype(np.int32)
    y_idx = np.rint(arr[:, 1] / px + cy).astype(np.int32)
    keep = (x_idx >= 0) & (x_idx < int(n_pixels)) & (y_idx >= 0) & (y_idx < int(n_pixels))
    if not np.any(keep):
        return image
    np.add.at(image, (y_idx[keep], x_idx[keep]), 1.0)
    return image


def fit_xy_rbasex_benchmark(
    xy_data: np.ndarray,
    n_peaks: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    from Abel_rbasex_reconstruction import reconstruct_rbasex

    n_pixels = int(kwargs.get("n_pixels", 512))
    pixel_size_mm = float(kwargs.get("pixel_size_mm", 0.05))
    image = _xy_to_centered_image(xy_data, pixel_size_mm=pixel_size_mm, n_pixels=n_pixels)

    params, _metadata = reconstruct_rbasex(image, config=None, verbose=False)
    if not params:
        return []

    peaks: List[Dict[str, float]] = []
    for p in params:
        peaks.append(
            {
                "r0": float(p.get("r", 0.0)),
                "sigma": float(p.get("sigma", 0.0)),
                "beta": float(p.get("beta", 0.0)),
                "amp": float(p.get("amp", p.get("branching_ratio", 0.0))),
            }
        )

    if n_peaks is not None and int(n_peaks) > 0:
        peaks = sorted(peaks, key=lambda t: float(t.get("amp", 0.0)), reverse=True)[: int(n_peaks)]
    peaks = sorted(peaks, key=lambda t: float(t.get("r0", 0.0)))
    return peaks

