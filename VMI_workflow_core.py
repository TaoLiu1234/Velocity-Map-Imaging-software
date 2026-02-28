#!/usr/bin/env python3
from __future__ import annotations

"""Core data-processing helpers for VMI_workflow.

This module contains pure functions (no Qt dependency), so it is easy to test
and easy to read for beginners.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CacheData:
    """In-memory cache for raw arrays loaded from three data files."""

    trigger_indices: np.ndarray
    electron_points: np.ndarray
    ion_points: np.ndarray


def ensure_2d(array: np.ndarray, expected_cols: int, name: str) -> np.ndarray:
    """Validate and normalize an array to shape (N, expected_cols)."""
    if array.ndim == 1:
        if array.size != expected_cols:
            raise ValueError(f"{name} does not have {expected_cols} columns.")
        array = array.reshape(1, expected_cols)
    if array.ndim != 2 or array.shape[1] != expected_cols:
        raise ValueError(f"{name} does not have {expected_cols} columns.")
    return array


def select_increment_pairs(
    trigger_indices: np.ndarray,
    progress_callback: Callable[[float], None] | None = None,
    progress_step: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Select +1/+1 rows using a rolling anchor-window search.

    Rule:
    - Keep one current anchor (first valid row is initial anchor, not returned).
    - Scan forward. A candidate row is accepted if it is +1/+1 versus *any* valid
      row seen since current anchor (including anchor itself).
    - Once accepted, that candidate becomes the new anchor and the search window
      is reset from this row.

    Input columns are expected as [electron_index, ion_index].
    NaN rows are ignored automatically.
    """
    # Pre-filter valid rows in vectorized form to minimize Python-loop overhead.
    e_idx = trigger_indices[:, 0]
    i_idx = trigger_indices[:, 1]
    valid = ~np.isnan(e_idx) & ~np.isnan(i_idx)
    if not np.any(valid):
        if progress_callback is not None:
            progress_callback(1.0)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    e_valid = np.rint(e_idx[valid]).astype(np.int64, copy=False)
    i_valid = np.rint(i_idx[valid]).astype(np.int64, copy=False)
    n = int(e_valid.size)
    if n <= 1:
        if progress_callback is not None:
            progress_callback(1.0)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if progress_callback is not None:
        progress_callback(0.0)
    progress_step = max(1, int(progress_step))
    progress_den = max(1, n - 1)

    selected_mask = np.zeros(n, dtype=bool)
    segment_id = 0

    # Fast path for non-negative 31-bit indices: pack (e,i) into one int64 key.
    # This avoids Python tuple allocation in the main loop.
    e_max = int(np.max(e_valid))
    i_max = int(np.max(i_valid))
    e_min = int(np.min(e_valid))
    i_min = int(np.min(i_valid))
    if e_min >= 0 and i_min >= 0 and e_max <= 0x7FFFFFFF and i_max <= 0x7FFFFFFF:
        e_i64 = e_valid.astype(np.int64, copy=False)
        i_i64 = i_valid.astype(np.int64, copy=False)
        key_now = (e_i64 << 32) | i_i64
        key_prev = ((e_i64 - 1) << 32) | (i_i64 - 1)

        seen_segment: dict[int, int] = {int(key_now[0]): segment_id}
        for idx in range(1, n):
            prev = int(key_prev[idx])
            if seen_segment.get(prev, -1) == segment_id:
                selected_mask[idx] = True
                segment_id += 1
            seen_segment[int(key_now[idx])] = segment_id
            if progress_callback is not None and (idx % progress_step == 0 or idx == n - 1):
                progress_callback(float(idx) / float(progress_den))
    else:
        # General path keeps exact integer-pair keys.
        seen_segment_pair: dict[tuple[int, int], int] = {(int(e_valid[0]), int(i_valid[0])): segment_id}
        for idx in range(1, n):
            prev_key = (int(e_valid[idx] - 1), int(i_valid[idx] - 1))
            if seen_segment_pair.get(prev_key, -1) == segment_id:
                selected_mask[idx] = True
                segment_id += 1
            seen_segment_pair[(int(e_valid[idx]), int(i_valid[idx]))] = segment_id
            if progress_callback is not None and (idx % progress_step == 0 or idx == n - 1):
                progress_callback(float(idx) / float(progress_den))

    return e_valid[selected_mask], i_valid[selected_mask]


def select_all_one_pairs(trigger_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select all non-NaN trigger rows as electron/ion index pairs.

    Input columns are expected as [electron_index, ion_index].
    Rows containing NaN in either column are dropped.
    """
    e_idx = trigger_indices[:, 0]
    i_idx = trigger_indices[:, 1]
    valid = ~np.isnan(e_idx) & ~np.isnan(i_idx)
    if not np.any(valid):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.rint(e_idx[valid]).astype(np.int64), np.rint(i_idx[valid]).astype(np.int64)


def density_counts_from_bins(x: np.ndarray, y: np.ndarray, bin_size: float | None = None) -> np.ndarray:
    """Estimate local density for each point by 2D bin counting.

    Returns one density value per input point. The GUI uses this value as
    point color in scatter plots.
    """
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    max_bins_axis = 900
    if bin_size is not None and bin_size > 0:
        nx = max(1, int(np.ceil(x_span / bin_size)))
        ny = max(1, int(np.ceil(y_span / bin_size)))
        nx = min(nx, max_bins_axis)
        ny = min(ny, max_bins_axis)
    else:
        target = int(np.clip(np.sqrt(n), 30, 180))
        nx = target
        ny = target

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    flat = x_bin * ny + y_bin
    counts_flat = np.bincount(flat, minlength=nx * ny).astype(np.float64)
    return counts_flat[flat]


def geometric_median(points_xy: np.ndarray, max_iter: int = 120, tol: float = 1e-6) -> np.ndarray:
    """Compute geometric median center (robust to outliers)."""
    if points_xy.shape[0] == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    if points_xy.shape[0] == 1:
        return points_xy[0].astype(np.float64)

    y = np.mean(points_xy, axis=0).astype(np.float64)
    for _ in range(max_iter):
        d = np.linalg.norm(points_xy - y, axis=1)
        nonzero = d > 1e-12
        if not np.any(nonzero):
            return y
        inv = 1.0 / d[nonzero]
        y_next = np.sum(points_xy[nonzero] * inv[:, None], axis=0) / np.sum(inv)
        if np.linalg.norm(y_next - y) <= tol:
            return y_next
        y = y_next
    return y


def circle_fit_kasa(points_xy: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a circle with an algebraic least-squares method (Kasa fit)."""
    if points_xy.shape[0] < 3:
        return None
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy, c0 = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = cx * cx + cy * cy + c0
    if r2 <= 0:
        return None
    return cx, cy, float(np.sqrt(r2))


def edge_circle_center(points_xy: np.ndarray, init_center: np.ndarray, angle_bins: int = 180) -> np.ndarray:
    """Estimate center from edge envelope, then circle-fit edge points."""
    if points_xy.shape[0] < 24:
        return np.mean(points_xy, axis=0).astype(np.float64)

    center = init_center.astype(np.float64)
    for _ in range(2):
        dx = points_xy[:, 0] - center[0]
        dy = points_xy[:, 1] - center[1]
        rr = np.hypot(dx, dy)
        theta = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
        bins = np.clip((theta * angle_bins).astype(np.int64), 0, angle_bins - 1)

        edge_indices = []
        for b in range(angle_bins):
            idx = np.where(bins == b)[0]
            if idx.size == 0:
                continue
            local = idx[np.argmax(rr[idx])]
            edge_indices.append(local)

        if len(edge_indices) < 24:
            break

        edge_points = points_xy[np.array(edge_indices, dtype=np.int64)]
        fit = circle_fit_kasa(edge_points)
        if fit is None:
            break
        new_center = np.array([fit[0], fit[1]], dtype=np.float64)
        if np.linalg.norm(new_center - center) <= 1e-6:
            center = new_center
            break
        center = new_center

    return center


def build_centered_histogram(
    points: np.ndarray, bin_size: float, force_lim: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a square 2D histogram around (0,0) for centered points."""
    if points.shape[0] == 0 or bin_size <= 0:
        return None
    x = points[:, 0].astype(np.float64, copy=False)
    y = points[:, 1].astype(np.float64, copy=False)
    if force_lim is None:
        max_abs = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), bin_size)
    else:
        max_abs = max(force_lim, bin_size)
    n_half = max(1, int(np.ceil(max_abs / bin_size)) + 2)
    n_side = 2 * n_half + 1
    edges = (np.arange(n_side + 1, dtype=np.float64) - n_half - 0.5) * bin_size
    h2d, xedges, yedges = np.histogram2d(x, y, bins=[edges, edges])
    return h2d.astype(np.float64), xedges, yedges


def build_denoised_centered_histogram(
    centered_signal: np.ndarray,
    centered_noise: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    bin_size: float,
) -> dict | None:
    """Build centered histogram and subtract constant inner-ring noise estimate.

    Noise model used here:
    - Estimate noise density from outer ring count / outer-ring area.
    - Convert to expected noise per bin.
    - Subtract this constant from all bins inside inner ring.
    """
    if centered_signal.shape[0] == 0 or bin_size <= 0:
        return None

    lim_signal = max(
        float(np.max(np.abs(centered_signal[:, 0]))),
        float(np.max(np.abs(centered_signal[:, 1]))),
        inner_radius,
    )
    lim_noise = 0.0
    if centered_noise.shape[0] > 0:
        lim_noise = max(
            float(np.max(np.abs(centered_noise[:, 0]))),
            float(np.max(np.abs(centered_noise[:, 1]))),
            outer_radius,
        )
    lim = max(lim_signal, lim_noise, outer_radius, bin_size)

    signal_built = build_centered_histogram(centered_signal, bin_size, force_lim=lim)
    if signal_built is None:
        return None
    signal_hist, xedges, yedges = signal_built

    noise_count = float(centered_noise.shape[0])
    area_inner = np.pi * inner_radius**2
    area_outer_ring = np.pi * max(outer_radius**2 - inner_radius**2, 1e-12)
    bin_area = float(bin_size * bin_size)
    noise_est = np.zeros_like(signal_hist, dtype=np.float64)
    expected_inner_noise_total = 0.0

    if noise_count > 0 and area_outer_ring > 0 and bin_area > 0:
        noise_density = noise_count / area_outer_ring
        expected_per_bin = noise_density * bin_area
        expected_inner_noise_total = noise_count * (area_inner / area_outer_ring)

        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        xv, yv = np.meshgrid(xcenters, ycenters, indexing="ij")
        inner_bin_mask = (xv * xv + yv * yv) <= (inner_radius * inner_radius)
        noise_est[inner_bin_mask] = expected_per_bin

    denoised = signal_hist - noise_est
    denoised[denoised < 0] = 0
    removed_total = float(np.sum(signal_hist) - np.sum(denoised))

    return {
        "hist_signal": signal_hist,
        "hist_noise_est": noise_est,
        "hist_denoised": denoised,
        "xedges": xedges,
        "yedges": yedges,
        "bin_size": float(bin_size),
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "signal_count": int(centered_signal.shape[0]),
        "noise_count": int(centered_noise.shape[0]),
        "outer_noise_count": noise_count,
        "expected_inner_noise_total": float(expected_inner_noise_total),
        "removed_total": removed_total,
    }
