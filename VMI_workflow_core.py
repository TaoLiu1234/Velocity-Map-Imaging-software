#!/usr/bin/env python3
from __future__ import annotations

"""Core data-processing helpers for VMI_workflow.

This module contains pure functions (no Qt dependency), so it is easy to test
and easy to read for beginners.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

@dataclass
class CacheData:
    """In-memory cache for raw arrays loaded from three data files."""

    trigger_indices: np.ndarray
    electron_points: np.ndarray
    ion_points: np.ndarray


def fast_read_csv_float64(
    path: str,
    *,
    n_columns: int,
    use_columns: tuple[int, ...],
) -> np.ndarray:
    """Read a headerless numeric CSV as float64 using a compiled C++ parser.

    Prefers pyarrow (C++ Arrow CSV reader; ~10x faster than np.loadtxt on
    large files), falling back to np.loadtxt with the exact same semantics:
    - returns (N, len(use_columns)) float64 array;
    - text `NaN` is parsed as numpy NaN (real data files contain "NaN");
    - invalid/missing rows become NaN, matching np.loadtxt behaviour.

    Parameters:
    - `path`: CSV file path.
    - `n_columns`: total number of columns in the file (trigger files have 4,
      electron/ion point files have 3).
    - `use_columns`: 0-based column indices to keep, in output order.
    """
    use_columns = tuple(int(c) for c in use_columns)
    if not use_columns:
        return np.zeros((0, 0), dtype=np.float64)
    n_columns = max(int(n_columns), max(use_columns) + 1)

    # Fast path: pyarrow C++ reader (preferred, battle-tested parser).
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except Exception:
        pa = None
        pacsv = None
    if pacsv is not None and pa is not None:
        try:
            names = [f"c{i}" for i in range(n_columns)]
            table = pacsv.read_csv(
                path,
                read_options=pa.csv.ReadOptions(column_names=names),
            )
            cols = []
            for i in use_columns:
                arr = table.column(names[i]).to_numpy(zero_copy_only=False)
                cols.append(np.asarray(arr, dtype=np.float64))
            if not cols:
                return np.zeros((0, 0), dtype=np.float64)
            out = np.column_stack(cols)
            return out
        except Exception:
            # Fall through to the np.loadtxt path on any parse anomaly
            # (e.g. ragged rows or malformed fields) rather than surfacing
            # library-specific errors to the user.
            pass

    # Fallback path: identical semantics to the historical implementation.
    return np.loadtxt(
        path,
        delimiter=",",
        dtype=np.float64,
        usecols=list(use_columns),
    )

def ensure_2d(array: np.ndarray, expected_cols: int, name: str) -> np.ndarray:
    """Validate and normalize an array to shape (N, expected_cols)."""
    if array.ndim == 1:
        if array.size != expected_cols:
            raise ValueError(f"{name} does not have {expected_cols} columns.")
        array = array.reshape(1, expected_cols)
    if array.ndim != 2 or array.shape[1] != expected_cols:
        raise ValueError(f"{name} does not have {expected_cols} columns.")
    return array


def _select_strict_delta_pairs(
    trigger_indices: np.ndarray,
    *,
    delta_e: int,
    delta_i: int,
    progress_callback: Callable[[float], None] | None = None,
    progress_step: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Select rows whose delta vs previous raw row matches (delta_e, delta_i).

    Both current row and immediately previous row must be valid (non-NaN in both
    columns). The returned indices are from the current row.
    """
    _ = progress_step  # Kept for API compatibility with previous implementation.
    n_rows = int(trigger_indices.shape[0])
    if progress_callback is not None:
        progress_callback(0.0)
    if n_rows <= 1:
        if progress_callback is not None:
            progress_callback(1.0)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    e_prev = trigger_indices[:-1, 0]
    i_prev = trigger_indices[:-1, 1]
    e_curr = trigger_indices[1:, 0]
    i_curr = trigger_indices[1:, 1]

    valid = (
        ~np.isnan(e_prev)
        & ~np.isnan(i_prev)
        & ~np.isnan(e_curr)
        & ~np.isnan(i_curr)
    )
    if not np.any(valid):
        if progress_callback is not None:
            progress_callback(1.0)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    e_prev_i = np.rint(e_prev[valid]).astype(np.int64, copy=False)
    i_prev_i = np.rint(i_prev[valid]).astype(np.int64, copy=False)
    e_curr_i = np.rint(e_curr[valid]).astype(np.int64, copy=False)
    i_curr_i = np.rint(i_curr[valid]).astype(np.int64, copy=False)

    match = ((e_curr_i - e_prev_i) == int(delta_e)) & ((i_curr_i - i_prev_i) == int(delta_i))
    if progress_callback is not None:
        progress_callback(1.0)
    if not np.any(match):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return e_curr_i[match], i_curr_i[match]


def select_increment_pairs(
    trigger_indices: np.ndarray,
    progress_callback: Callable[[float], None] | None = None,
    progress_step: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict 1e/1i selector: current row must be +1/+1 vs previous raw row.

    Input columns are expected as [electron_index, ion_index].
    Rows with NaN in either column are invalid for strict-adjacent comparison.
    """
    return _select_strict_delta_pairs(
        trigger_indices,
        delta_e=1,
        delta_i=1,
        progress_callback=progress_callback,
        progress_step=progress_step,
    )


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

def select_one_e_two_i_pairs(
    trigger_indices: np.ndarray,
    progress_callback: Callable[[float], None] | None = None,
    progress_step: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict 1e/2i selector: current row must be +1/+2 vs previous raw row.

    For each accepted row ``(e, i)`` this returns two pairs:
    ``(e, i-1)`` and ``(e, i)``.
    """
    e_selected, i_selected = _select_strict_delta_pairs(
        trigger_indices,
        delta_e=1,
        delta_i=2,
        progress_callback=progress_callback,
        progress_step=progress_step,
    )
    if e_selected.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    out_e = np.repeat(e_selected, 2).astype(np.int64, copy=False)
    out_i = np.empty(out_e.shape[0], dtype=np.int64)
    out_i[0::2] = i_selected - 1
    out_i[1::2] = i_selected

    # Safety guard for malformed low-index rows.
    keep = out_i >= 0
    if not np.all(keep):
        out_e = out_e[keep]
        out_i = out_i[keep]
    return out_e, out_i


def select_one_e_three_i_pairs(
    trigger_indices: np.ndarray,
    progress_callback: Callable[[float], None] | None = None,
    progress_step: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict 1e/3i selector: current row must be +1/+3 vs previous raw row.

    For each accepted row ``(e, i)`` this returns three pairs:
    ``(e, i-2)``, ``(e, i-1)``, and ``(e, i)``.
    """
    e_selected, i_selected = _select_strict_delta_pairs(
        trigger_indices,
        delta_e=1,
        delta_i=3,
        progress_callback=progress_callback,
        progress_step=progress_step,
    )
    if e_selected.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    out_e = np.repeat(e_selected, 3).astype(np.int64, copy=False)
    out_i = np.empty(out_e.shape[0], dtype=np.int64)
    out_i[0::3] = i_selected - 2
    out_i[1::3] = i_selected - 1
    out_i[2::3] = i_selected

    # Safety guard for malformed low-index rows.
    keep = out_i >= 0
    if not np.all(keep):
        out_e = out_e[keep]
        out_i = out_i[keep]
    return out_e, out_i


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


def circle_fit_kasa(
    points_xy: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Fit a circle with an algebraic least-squares method (Kasa fit)."""
    if points_xy.shape[0] < 3:
        return None
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    x = pts[:, 0].astype(np.float64, copy=False)
    y = pts[:, 1].astype(np.float64, copy=False)
    finite = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != pts.shape[0]:
            return None
        finite = finite & np.isfinite(w) & (w > 0.0)
    else:
        w = None
    if np.count_nonzero(finite) < 3:
        return None
    x = x[finite]
    y = y[finite]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    if w is not None:
        sqrt_w = np.sqrt(np.maximum(w[finite], 1e-12))
        a = a * sqrt_w[:, None]
        b = b * sqrt_w
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
    for _ in range(3):
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


def quadrant_symmetry_center(
    points_xy: np.ndarray,
    fallback_xy: tuple[float, float] | np.ndarray,
    *,
    grid_bins: int = 72,
    sample_limit: int = 28_000,
) -> np.ndarray:
    """Estimate center from diagonal quadrant symmetry using raw point matches.

    Around one candidate center, the cloud is split into four quadrants:
    right-up, left-up, left-down, right-down. Opposite quadrants should match
    after 180-degree rotation, so we:

    - fold each quadrant into first-quadrant coordinates `(abs(dx), abs(dy))`
    - match points between diagonal quadrants with nearest-neighbor search
    - score one center by the weighted mismatch of those paired points
    - update the center from the weighted midpoints of matched opposite points

    This keeps the objective on raw scatter points rather than 2D histogram
    bins, which is noticeably smoother and more stable under small center
    changes.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    fallback = np.asarray(fallback_xy, dtype=np.float64).reshape(-1)
    if fallback.size < 2:
        fallback = np.array([0.0, 0.0], dtype=np.float64)
    else:
        fallback = fallback[:2].astype(np.float64, copy=False)

    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0:
        return fallback.copy()

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite, :2]
    if pts.shape[0] == 0:
        return fallback.copy()

    if sample_limit > 0 and pts.shape[0] > int(sample_limit):
        step = max(1, int(np.ceil(float(pts.shape[0]) / float(sample_limit))))
        pts = pts[::step]

    if pts.shape[0] < 20:
        return np.mean(pts, axis=0).astype(np.float64)
    if cKDTree is None:
        return edge_circle_center(pts, fallback)

    seed_center = edge_circle_center(pts, fallback)
    dx_seed = pts[:, 0] - float(seed_center[0])
    dy_seed = pts[:, 1] - float(seed_center[1])
    rr_seed = np.hypot(dx_seed, dy_seed)
    rr_pos = rr_seed[np.isfinite(rr_seed) & (rr_seed > 0.0)]
    if rr_pos.size == 0:
        return seed_center.astype(np.float64, copy=False)

    def _dominant_radius(rr: np.ndarray) -> tuple[float, float]:
        rr_use = rr[np.isfinite(rr) & (rr > 0.0)]
        if rr_use.size == 0:
            return 1.0, 0.2
        hi = float(np.quantile(rr_use, 0.995))
        hi = max(hi, float(np.max(rr_use)))
        bins_r = max(60, min(180, int(np.sqrt(rr_use.size) * 2.0)))
        hist, edges = np.histogram(rr_use, bins=bins_r, range=(0.0, hi))
        if hist.size == 0:
            return max(float(np.median(rr_use)), 1.0), max(0.03 * float(np.median(rr_use)), 0.2)
        centers = 0.5 * (edges[:-1] + edges[1:])
        peak_idx = int(np.argmax(hist))
        peak_r = float(centers[peak_idx]) if centers.size > 0 else float(np.median(rr_use))
        dr = abs(float(edges[1] - edges[0])) if edges.size >= 2 else max(0.03 * peak_r, 0.2)
        return max(peak_r, 1e-9), max(dr, 1e-9)

    peak_r, dr = _dominant_radius(rr_seed)
    shell_sigma = max(2.2 * dr, 0.05 * peak_r, 0.6)
    axis_sigma = max(0.35 * shell_sigma, 0.35)
    match_sigma = max(1.6 * dr, 0.045 * peak_r, 0.45)
    search_radius = max(0.8, min(0.15 * max(peak_r, 1.0), 4.5))
    max_step = max(0.12, min(0.09 * max(peak_r, 1.0), 2.0))
    move_tol = max(1e-3, 2e-4 * max(peak_r, 1.0))
    rel_tol = 1e-6
    _ = grid_bins  # kept for call compatibility; raw-point matcher does not bin quadrants.

    # The point set is frozen for the whole candidate search, so the KD-tree is
    # built once here instead of per candidate center. Opposite-quadrant
    # matches are evaluated by querying this raw tree with 180-degree antipodal
    # points (see _pair_contrib below) rather than rebuilding folded trees.
    # `leafsize=8` tuned for query throughput; nearest-neighbour results are
    # exact (and identical) for any leafsize.
    tree_pts = cKDTree(pts, leafsize=8)

    def _nearest_in_dst_mask(query_pts: np.ndarray, dst_mask: np.ndarray) -> np.ndarray:
        """Nearest raw point inside `dst_mask` for each antipodal query point.

        Queries the shared raw tree with growing k until the found neighbour
        satisfies the destination-quadrant mask (neighbours come back sorted
        by distance, so the first hit is the masked nearest neighbour).
        """
        n_all = int(pts.shape[0])
        matched = np.full(int(query_pts.shape[0]), -1, dtype=np.int64)
        todo = np.arange(int(query_pts.shape[0]), dtype=np.int64)
        # Fast path: the common case resolves with the plain k=1 query.
        _dist1, idx1 = tree_pts.query(query_pts[todo], k=1)
        idx1 = np.asarray(idx1, dtype=np.int64).reshape(-1)
        hit = dst_mask[idx1]
        matched[todo[hit]] = idx1[hit]
        todo = todo[~hit]
        k = 4
        while todo.size > 0:
            kk = k if k < n_all else n_all
            _dist_k, idx_k = tree_pts.query(query_pts[todo], k=kk)
            if kk == 1:
                idx_k = np.asarray(idx_k, dtype=np.int64).reshape(-1, 1)
            else:
                idx_k = np.asarray(idx_k, dtype=np.int64)
            valid = dst_mask[idx_k]
            first = np.argmax(valid, axis=1)
            hit = valid[np.arange(todo.size), first]
            rows = np.flatnonzero(hit)
            matched[todo[rows]] = idx_k[rows, first[rows]]
            todo = todo[~hit]
            if kk >= n_all:
                break
            k = kk * 4
        # `dst_mask` is never empty here (checked by the caller); once k
        # reaches the full point count every query has resolved.
        return matched

    def _pair_point_stats(center_xy: np.ndarray) -> tuple[float, np.ndarray, float]:
        dx = pts[:, 0] - float(center_xy[0])
        dy = pts[:, 1] - float(center_xy[1])
        rr = np.hypot(dx, dy)
        shell_resid = (rr - peak_r) / max(shell_sigma, 1e-9)
        radial_weight = np.exp(-0.5 * np.clip(shell_resid * shell_resid, 0.0, 25.0))
        # Points near the candidate axes flip quadrant labels too easily, so
        # reduce their leverage instead of letting them dominate the update.
        axis_weight = np.tanh(np.abs(dx) / axis_sigma) * np.tanh(np.abs(dy) / axis_sigma)
        point_weight = np.clip(radial_weight * axis_weight, 0.0, None)
        shell_keep = np.abs(shell_resid) <= 2.4
        if np.count_nonzero(shell_keep) >= max(64, pts.shape[0] // 20):
            point_weight = np.where(shell_keep, point_weight, 0.0)
        if not np.any(point_weight > 1e-12):
            return np.inf, np.asarray(center_xy, dtype=np.float64).reshape(2).copy(), 0.0

        q_ru = (dx >= 0.0) & (dy >= 0.0)
        q_lu = (dx < 0.0) & (dy >= 0.0)
        q_ld = (dx < 0.0) & (dy < 0.0)
        q_rd = (dx >= 0.0) & (dy < 0.0)
        midpoint_sum = np.zeros(2, dtype=np.float64)
        midpoint_wsum = 0.0
        score_sum = 0.0
        score_wsum = 0.0
        pair_balance_penalty = 0.0

        def _pair_contrib(src_mask: np.ndarray, dst_mask: np.ndarray) -> tuple[np.ndarray, float, float, float]:
            src_idx = np.flatnonzero(src_mask)
            dst_idx = np.flatnonzero(dst_mask)
            if src_idx.size == 0 or dst_idx.size == 0:
                return np.zeros(2, dtype=np.float64), 0.0, 0.0, 0.0

            # Folded-space nearest neighbour between opposite quadrants equals
            # the raw nearest neighbour of the 180-degree antipode
            # `center + (center - src)` restricted to the destination
            # quadrant, so the shared raw tree replaces the per-candidate
            # folded trees. Distances are recomputed in folded coordinates
            # below with the exact arithmetic the folded trees produced.
            center_fx = float(center_xy[0])
            center_fy = float(center_xy[1])
            query_pts = np.column_stack(
                (
                    center_fx + (center_fx - pts[src_idx, 0]),
                    center_fy + (center_fy - pts[src_idx, 1]),
                )
            )
            matched_idx = _nearest_in_dst_mask(query_pts, dst_mask)

            src_fold_x = np.abs(dx[src_idx])
            src_fold_y = np.abs(dy[src_idx])
            dist = np.sqrt(
                (src_fold_x - np.abs(dx[matched_idx])) ** 2
                + (src_fold_y - np.abs(dy[matched_idx])) ** 2
            )

            w_src = point_weight[src_idx]
            w_dst = point_weight[matched_idx]
            match_weight = np.exp(-0.5 * np.clip((dist / max(match_sigma, 1e-9)) ** 2, 0.0, 25.0))
            w_eff = np.clip(w_src * w_dst * match_weight, 0.0, None)
            if dist.size >= 24:
                dist_cut = float(np.quantile(dist, 0.70))
                keep = dist <= dist_cut
                if np.count_nonzero(keep) >= max(12, dist.size // 4):
                    matched_idx = matched_idx[keep]
                    src_idx = src_idx[keep]
                    dist = dist[keep]
                    w_eff = w_eff[keep]
            w_sum = float(np.sum(w_eff))
            if w_sum <= 1e-12:
                return np.zeros(2, dtype=np.float64), 0.0, 0.0, 0.0

            mid = 0.5 * (pts[src_idx] + pts[matched_idx])
            midpoint_acc = np.sum(w_eff[:, None] * mid, axis=0)
            score_acc = float(np.sum(w_eff * dist * dist))
            return midpoint_acc, w_sum, score_acc, float(np.sum(w_src))

        for mask_a, mask_b in ((q_ru, q_ld), (q_lu, q_rd)):
            mass_a = float(np.sum(point_weight[mask_a]))
            mass_b = float(np.sum(point_weight[mask_b]))
            pair_mass = mass_a + mass_b
            if pair_mass > 1e-12:
                pair_balance_penalty += abs(mass_a - mass_b) / pair_mass

            mid_ab, w_ab, score_ab, _src_mass_ab = _pair_contrib(mask_a, mask_b)
            midpoint_sum += mid_ab
            midpoint_wsum += w_ab
            score_sum += score_ab
            score_wsum += w_ab

            mid_ba, w_ba, score_ba, _src_mass_ba = _pair_contrib(mask_b, mask_a)
            midpoint_sum += mid_ba
            midpoint_wsum += w_ba
            score_sum += score_ba
            score_wsum += w_ba

        if score_wsum <= 1e-12 or midpoint_wsum <= 1e-12:
            return np.inf, np.asarray(center_xy, dtype=np.float64).reshape(2).copy(), 0.0

        center_update = midpoint_sum / midpoint_wsum
        score = float(score_sum / score_wsum + 0.15 * (match_sigma**2) * pair_balance_penalty)
        if not np.isfinite(score) or not np.isfinite(center_update).all():
            return np.inf, np.asarray(center_xy, dtype=np.float64).reshape(2).copy(), 0.0
        return score, center_update.astype(np.float64, copy=False), score_wsum

    def _coarse_search(center_start: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
        best_center_local = np.asarray(center_start, dtype=np.float64).reshape(2).copy()
        best_score_local, _center_hint, _mass = _pair_point_stats(best_center_local)
        levels = (
            (float(radius), 5),
            (max(float(radius) * 0.35, 0.24), 5),
        )
        for rad, steps in levels:
            base = best_center_local.copy()
            offsets = np.linspace(-rad, rad, int(steps), dtype=np.float64)
            for off_x in offsets:
                for off_y in offsets:
                    candidate = np.array([base[0] + off_x, base[1] + off_y], dtype=np.float64)
                    cand_score, _cand_hint, _cand_mass = _pair_point_stats(candidate)
                    if cand_score + 1e-12 < best_score_local:
                        best_score_local = cand_score
                        best_center_local = candidate
        return best_center_local, best_score_local

    center, best_score = _coarse_search(seed_center, search_radius)
    best_center = center.copy()
    best_mass = 0.0
    for _ in range(14):
        score_now, center_hint, matched_mass = _pair_point_stats(center)
        if score_now + 1e-12 < best_score:
            best_score = score_now
            best_center = center.copy()
            best_mass = matched_mass

        step = center_hint - center
        step_norm = float(np.linalg.norm(step))
        if (not np.isfinite(step_norm)) or step_norm <= move_tol:
            break
        if step_norm > max_step:
            step *= max_step / step_norm
            step_norm = max_step

        accepted = False
        alpha = 1.0
        while alpha >= 1.0 / 32.0:
            candidate = center + alpha * step
            cand_score, _cand_hint, cand_mass = _pair_point_stats(candidate)
            if cand_score + 1e-12 < score_now:
                improvement = float(score_now - cand_score)
                center = candidate
                score_now = cand_score
                matched_mass = cand_mass
                accepted = True
                if cand_score + 1e-12 < best_score:
                    best_score = cand_score
                    best_center = candidate.copy()
                    best_mass = cand_mass
                if alpha * step_norm <= move_tol or improvement <= rel_tol * max(abs(score_now), 1.0):
                    break
                break
            alpha *= 0.5
        if not accepted:
            break
        if matched_mass <= 1e-12:
            break

    # Final small local search helps if the midpoint iteration lands between
    # two nearly equivalent match configurations.
    for radius, steps in (
        (max(0.35 * max_step, 0.10), 5),
        (max(0.12 * max_step, move_tol), 5),
    ):
        base = best_center.copy()
        offsets = np.linspace(-float(radius), float(radius), int(steps), dtype=np.float64)
        improved = False
        for off_x in offsets:
            for off_y in offsets:
                candidate = np.array([base[0] + off_x, base[1] + off_y], dtype=np.float64)
                cand_score, _cand_hint, cand_mass = _pair_point_stats(candidate)
                if cand_score + 1e-12 < best_score:
                    best_score = cand_score
                    best_center = candidate
                    best_mass = cand_mass
                    improved = True
        if (not improved) and float(radius) <= move_tol:
            break

    _ = best_mass
    return best_center


def _smooth_profile_1d(values: np.ndarray) -> np.ndarray:
    """Small symmetric smoothing used for radial/column peak picking."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 2:
        return arr.astype(np.float64, copy=True)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= float(np.sum(kernel))
    return np.convolve(arr, kernel, mode="same")


def _smooth_profile_axis0_kernel5(values_2d: np.ndarray) -> np.ndarray:
    """Apply the same 5-tap symmetric smoother along axis=0 of a 2D array."""
    arr = np.asarray(values_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 2:
        return arr.astype(np.float64, copy=True)
    # Equivalent to applying `_smooth_profile_1d` per column (`mode="same"` => zero padding).
    pad = np.pad(arr, ((2, 2), (0, 0)), mode="constant", constant_values=0.0)
    return (
        pad[0:-4, :]
        + (2.0 * pad[1:-3, :])
        + (3.0 * pad[2:-2, :])
        + (2.0 * pad[3:-1, :])
        + pad[4:, :]
    ) / 9.0


def _pick_profile_upper_edge_index(
    profile: np.ndarray,
    *,
    lo: int,
    hi: int,
    target_idx: int,
    threshold_frac: float,
    tail_mass_fraction: float,
) -> int | None:
    """Pick a stable upper-envelope index from one radial profile window."""
    prof = np.asarray(profile, dtype=np.float64).reshape(-1)
    if prof.size == 0:
        return None

    lo_i = int(max(0, min(lo, prof.size - 1)))
    hi_i = int(max(lo_i + 1, min(hi, prof.size)))
    window = np.clip(prof[lo_i:hi_i], 0.0, None)
    if window.size == 0:
        return None

    local_peak = int(np.clip(target_idx, lo_i, hi_i - 1)) - lo_i
    if window[local_peak] <= 0.0:
        local_peak = int(np.argmax(window))

    peak_level = float(np.max(window))
    if (not np.isfinite(peak_level)) or peak_level <= 0.0:
        return None

    support = window >= max(1.0, float(threshold_frac) * peak_level)
    if not np.any(support):
        return int(lo_i + local_peak)
    if not support[local_peak]:
        supported_idx = np.flatnonzero(support)
        if supported_idx.size == 0:
            return int(lo_i + local_peak)
        nearest = int(np.argmin(np.abs(supported_idx - local_peak)))
        local_peak = int(supported_idx[nearest])

    run_lo = local_peak
    run_hi = local_peak
    while run_lo > 0 and support[run_lo - 1]:
        run_lo -= 1
    while run_hi + 1 < support.size and support[run_hi + 1]:
        run_hi += 1

    component = window[run_lo : run_hi + 1]
    comp_total = float(np.sum(component))
    if (not np.isfinite(comp_total)) or comp_total <= 0.0:
        return int(lo_i + local_peak)

    tail_frac = float(np.clip(tail_mass_fraction, 1e-3, 0.999))
    reverse_cum = np.cumsum(component[::-1], dtype=np.float64)
    edge_rev = int(np.searchsorted(reverse_cum, tail_frac * comp_total, side="left"))
    edge_rev = int(np.clip(edge_rev, 0, component.size - 1))
    edge_idx = run_hi - edge_rev
    return int(lo_i + edge_idx)


def _select_polar_peak_line(
    hist: np.ndarray,
    r_centers: np.ndarray,
    *,
    peak_mode: str,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick one radial ridge per theta column from a polar histogram."""
    hist_arr = np.asarray(hist, dtype=np.float64)
    r_arr = np.asarray(r_centers, dtype=np.float64).reshape(-1)
    if hist_arr.ndim != 2 or hist_arr.size == 0 or r_arr.size != hist_arr.shape[0]:
        n_theta = int(hist_arr.shape[1]) if hist_arr.ndim == 2 else 0
        return (
            np.zeros(n_theta, dtype=np.float64),
            np.zeros(n_theta, dtype=np.float64),
            np.zeros(n_theta, dtype=bool),
        )

    n_r, n_theta = hist_arr.shape
    mode = str(peak_mode).strip().lower()
    if mode not in {"dominant", "outermost"}:
        mode = "dominant"
    manual_target = (
        target_radius is not None
        and np.isfinite(float(target_radius))
        and float(target_radius) > 0.0
        and r_arr.size > 0
    )
    if r_arr.size >= 2:
        dr = float(np.median(np.diff(r_arr)))
    else:
        dr = max(float(target_radius) * 0.05, 1.0) if manual_target else 1.0
    if (not np.isfinite(dr)) or dr <= 0.0:
        dr = 1.0
    if manual_target:
        target_r = float(target_radius)
        if target_window is not None and np.isfinite(float(target_window)) and float(target_window) > 0.0:
            win = max(float(target_window), 2.0 * dr)
        else:
            win = max(6.0 * dr, 0.08 * target_r, 1.5)
        in_window = np.abs(r_arr - target_r) <= win
        if not np.any(in_window):
            nearest = int(np.argmin(np.abs(r_arr - target_r)))
            pad = max(2, int(np.ceil(win / dr)))
            lo_window = max(0, nearest - pad)
            hi_window = min(n_r, nearest + pad + 1)
            in_window = np.zeros(n_r, dtype=bool)
            in_window[lo_window:hi_window] = True
        manual_lo = int(np.flatnonzero(in_window)[0])
        manual_hi = int(np.flatnonzero(in_window)[-1] + 1)
    else:
        manual_lo = 0
        manual_hi = n_r

    if mode == "dominant":
        if manual_target:
            hist_window = hist_arr[manual_lo:manual_hi, :]
            peak_idx_local = np.argmax(hist_window, axis=0) if hist_window.size > 0 else np.zeros(n_theta, dtype=np.int64)
            peak_idx = manual_lo + peak_idx_local
        else:
            peak_idx = np.argmax(hist_arr, axis=0) if hist_arr.size > 0 else np.zeros(n_theta, dtype=np.int64)
        peak_r = r_arr[peak_idx] if r_arr.size > 0 else np.zeros(n_theta, dtype=np.float64)
        peak_counts = hist_arr[peak_idx, np.arange(n_theta)] if n_theta > 0 else np.zeros(0, dtype=np.float64)
        peak_mask = peak_counts > 0.0
        return peak_r, peak_counts, peak_mask

    global_profile = _smooth_profile_1d(np.sum(hist_arr, axis=1))
    if global_profile.size == 0 or float(np.max(global_profile)) <= 0.0:
        return (
            np.zeros(n_theta, dtype=np.float64),
            np.zeros(n_theta, dtype=np.float64),
            np.zeros(n_theta, dtype=bool),
        )

    max_global = float(np.max(global_profile))
    peak_candidates = []
    for idx in range(global_profile.size):
        left = global_profile[idx - 1] if idx > 0 else -np.inf
        right = global_profile[idx + 1] if idx + 1 < global_profile.size else -np.inf
        if global_profile[idx] >= left and global_profile[idx] >= right:
            peak_candidates.append(idx)
    if peak_candidates:
        if manual_target:
            local_peaks = [idx for idx in peak_candidates if manual_lo <= idx < manual_hi]
            significant = [idx for idx in local_peaks if global_profile[idx] >= max(1.0, 0.04 * max_global)]
            if significant:
                target_idx = int(min(significant, key=lambda idx: abs(float(r_arr[idx]) - float(target_radius))))
            elif local_peaks:
                target_idx = int(min(local_peaks, key=lambda idx: abs(float(r_arr[idx]) - float(target_radius))))
            else:
                target_idx = int(manual_lo + np.argmax(global_profile[manual_lo:manual_hi]))
        else:
            significant = [idx for idx in peak_candidates if global_profile[idx] >= max(1.0, 0.08 * max_global)]
            if significant:
                support_scores = []
                for idx in significant:
                    band_lo = max(0, idx - 1)
                    band_hi = min(n_r, idx + 2)
                    support = float(np.mean(np.sum(hist_arr[band_lo:band_hi, :], axis=0) > 0.0))
                    support_scores.append((idx, support))
                max_support = max(score for _idx, score in support_scores) if support_scores else 0.0
                support_cut = max(0.10, 0.45 * max_support)
                supported = [idx for idx, score in support_scores if score >= support_cut]
                target_idx = int(supported[-1] if supported else significant[-1])
            else:
                target_idx = int(peak_candidates[-1])
    else:
        if manual_target:
            target_idx = int(manual_lo + np.argmax(global_profile[manual_lo:manual_hi]))
        else:
            target_idx = int(np.argmax(global_profile))

    if manual_target:
        lo = manual_lo
        hi = manual_hi
    else:
        band_bins = max(4, int(np.ceil(0.08 * max(1, n_r))))
        lo = max(0, target_idx - band_bins)
        hi = min(n_r, target_idx + band_bins + 1)
    peak_r = np.zeros(n_theta, dtype=np.float64)
    peak_counts = np.zeros(n_theta, dtype=np.float64)
    peak_mask = np.zeros(n_theta, dtype=bool)
    threshold_frac = 0.18 if manual_target else 0.12
    tail_mass_fraction = 0.20 if manual_target else 0.24
    for col in range(n_theta):
        col_profile = _smooth_profile_1d(hist_arr[:, col])
        col_window = col_profile[lo:hi]
        if col_window.size == 0:
            continue
        col_max = float(np.max(col_window))
        if (not np.isfinite(col_max)) or col_max <= 0.0:
            continue
        local_peak_idx = int(lo + np.argmax(col_window))
        idx_local = _pick_profile_upper_edge_index(
            col_profile,
            lo=lo,
            hi=hi,
            target_idx=local_peak_idx,
            threshold_frac=threshold_frac,
            tail_mass_fraction=tail_mass_fraction,
        )
        if idx_local is None:
            idx_local = local_peak_idx
        peak_r[col] = float(r_arr[idx_local])
        peak_counts[col] = float(hist_arr[idx_local, col])
        peak_mask[col] = peak_counts[col] > 0.0
    return peak_r, peak_counts, peak_mask


def build_polar_histogram(
    points_xy: np.ndarray,
    center_xy: tuple[float, float] | np.ndarray,
    *,
    theta_bins: int = 360,
    radial_bins: int = 220,
    r_max: float | None = None,
    peak_mode: str = "dominant",
    target_radius: float | None = None,
    target_window: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Convert XY points into a binned polar (r,theta) matrix around one center.

    The returned `hist` array is shaped as `(n_r, n_theta)` so it can be drawn
    directly with `pcolormesh(theta_edges, r_edges, hist)`.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] < 2 or center.size < 2:
        return {
            "hist": np.zeros((0, 0), dtype=np.float64),
            "r_edges": np.zeros(0, dtype=np.float64),
            "theta_edges": np.zeros(0, dtype=np.float64),
            "r_centers": np.zeros(0, dtype=np.float64),
            "theta_centers": np.zeros(0, dtype=np.float64),
            "peak_r": np.zeros(0, dtype=np.float64),
            "peak_counts": np.zeros(0, dtype=np.float64),
            "peak_mask": np.zeros(0, dtype=bool),
            "peak_r_mean": 0.0,
            "peak_r_std": np.inf,
            "straightness_score": np.inf,
            "valid_theta_fraction": 0.0,
        }

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite, :2]
    if pts.shape[0] == 0:
        return {
            "hist": np.zeros((0, 0), dtype=np.float64),
            "r_edges": np.zeros(0, dtype=np.float64),
            "theta_edges": np.zeros(0, dtype=np.float64),
            "r_centers": np.zeros(0, dtype=np.float64),
            "theta_centers": np.zeros(0, dtype=np.float64),
            "peak_r": np.zeros(0, dtype=np.float64),
            "peak_counts": np.zeros(0, dtype=np.float64),
            "peak_mask": np.zeros(0, dtype=bool),
            "peak_r_mean": 0.0,
            "peak_r_std": np.inf,
            "straightness_score": np.inf,
            "valid_theta_fraction": 0.0,
        }

    theta_bins = max(24, int(theta_bins))
    radial_bins = max(24, int(radial_bins))

    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    rr = np.hypot(dx, dy)
    tt = np.degrees(np.arctan2(dy, dx))
    finite_rt = np.isfinite(rr) & np.isfinite(tt)
    rr = rr[finite_rt]
    tt = tt[finite_rt]
    if rr.size == 0:
        return {
            "hist": np.zeros((0, 0), dtype=np.float64),
            "r_edges": np.zeros(0, dtype=np.float64),
            "theta_edges": np.zeros(0, dtype=np.float64),
            "r_centers": np.zeros(0, dtype=np.float64),
            "theta_centers": np.zeros(0, dtype=np.float64),
            "peak_r": np.zeros(0, dtype=np.float64),
            "peak_counts": np.zeros(0, dtype=np.float64),
            "peak_mask": np.zeros(0, dtype=bool),
            "peak_r_mean": 0.0,
            "peak_r_std": np.inf,
            "straightness_score": np.inf,
            "valid_theta_fraction": 0.0,
        }

    if r_max is None:
        r_hi = float(np.max(rr)) if rr.size > 0 else 1.0
    else:
        r_hi = float(r_max)
    if (not np.isfinite(r_hi)) or r_hi <= 0.0:
        r_hi = float(np.max(rr)) if rr.size > 0 else 1.0
    r_hi = max(r_hi, 1e-9)

    theta_edges = np.linspace(-180.0, 180.0, theta_bins + 1, dtype=np.float64)
    r_edges = np.linspace(0.0, r_hi, radial_bins + 1, dtype=np.float64)
    hist, _r_edges, _theta_edges = np.histogram2d(rr, tt, bins=[r_edges, theta_edges])
    hist = hist.astype(np.float64, copy=False)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    peak_r, peak_counts, peak_mask = _select_polar_peak_line(
        hist,
        r_centers,
        peak_mode=peak_mode,
        target_radius=target_radius,
        target_window=target_window,
    )
    hist_peak_r = np.asarray(peak_r, dtype=np.float64)
    hist_peak_counts = np.asarray(peak_counts, dtype=np.float64)
    hist_peak_mask = np.asarray(peak_mask, dtype=bool)
    if str(peak_mode).strip().lower() == "outermost":
        scatter_line = _build_scatter_peak_line_profile(
            pts[:, :2],
            center[:2],
            peak_mode=peak_mode,
            theta_bins=theta_bins,
            sample_limit=64_000,
            target_radius=target_radius,
            target_window=target_window,
        )
        if scatter_line is not None:
            peak_r_sc = np.asarray(scatter_line.get("peak_r", hist_peak_r), dtype=np.float64)
            peak_counts_sc = np.asarray(scatter_line.get("peak_counts", hist_peak_counts), dtype=np.float64)
            peak_mask_sc = np.asarray(scatter_line.get("peak_mask", hist_peak_mask), dtype=bool)
            use_scatter_line = (
                peak_r_sc.size == theta_centers.size
                and peak_counts_sc.size == theta_centers.size
                and peak_mask_sc.size == theta_centers.size
                and np.isfinite(peak_r_sc).any()
                and (np.count_nonzero(peak_mask_sc) >= max(8, theta_centers.size // 40))
            )
            if use_scatter_line:
                peak_r = peak_r_sc
                peak_counts = peak_counts_sc
                peak_mask = peak_mask_sc
                theta_centers_sc = np.asarray(scatter_line.get("theta_centers_deg", theta_centers), dtype=np.float64)
                if theta_centers_sc.size == theta_centers.size:
                    theta_centers = theta_centers_sc
            else:
                # Keep histogram-derived line when scatter-line support is too sparse;
                # this avoids the "missing peak curve" impression in the GUI.
                peak_r = hist_peak_r
                peak_counts = hist_peak_counts
                peak_mask = hist_peak_mask
    valid_theta_fraction = float(np.mean(peak_mask)) if peak_mask.size > 0 else 0.0

    if np.any(peak_mask):
        weights = peak_counts[peak_mask]
        peak_r_valid = peak_r[peak_mask]
        peak_r_mean = float(np.average(peak_r_valid, weights=weights))
        peak_r_std = float(np.sqrt(np.average((peak_r_valid - peak_r_mean) ** 2, weights=weights)))
        denom = max(peak_r_mean, 1.0)
        coverage_penalty = 0.25 if str(peak_mode).strip().lower() == "outermost" else 0.75
        straightness_score = float((peak_r_std / denom) + coverage_penalty * (1.0 - valid_theta_fraction))
    else:
        peak_r_mean = 0.0
        peak_r_std = np.inf
        straightness_score = np.inf

    return {
        "hist": hist,
        "r_edges": r_edges,
        "theta_edges": theta_edges,
        "r_centers": r_centers,
        "theta_centers": theta_centers,
        "peak_r": peak_r,
        "peak_counts": peak_counts,
        "peak_mask": peak_mask,
        "peak_r_mean": peak_r_mean,
        "peak_r_std": peak_r_std,
        "straightness_score": straightness_score,
        "valid_theta_fraction": valid_theta_fraction,
        "peak_mode": str(peak_mode),
        "target_radius": (float(target_radius) if target_radius is not None and np.isfinite(float(target_radius)) else np.nan),
        "target_window": (float(target_window) if target_window is not None and np.isfinite(float(target_window)) else np.nan),
    }


def polar_outermost_center(
    points_xy: np.ndarray,
    fallback_xy: tuple[float, float] | np.ndarray,
    *,
    theta_bins: int = 180,
    radial_bins: int = 200,
    sample_limit: int = 64_000,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> np.ndarray:
    """Estimate center by straightening the outermost ring from raw scatter points."""
    _ = radial_bins
    return _scatter_peak_line_center(
        points_xy,
        fallback_xy,
        peak_mode="outermost",
        theta_bins=theta_bins,
        sample_limit=sample_limit,
        target_radius=target_radius,
        target_window=target_window,
    )


def _theta_bin_indices(theta_rad: np.ndarray, n_theta: int) -> np.ndarray:
    """Map `[-pi, pi]` angles into stable theta-bin indices."""
    n_bins = max(1, int(n_theta))
    scaled = (np.asarray(theta_rad, dtype=np.float64) + np.pi) / (2.0 * np.pi)
    return np.clip((scaled * float(n_bins)).astype(np.int64), 0, n_bins - 1)


def _pick_scatter_shell_radius(
    rr: np.ndarray,
    peak_mode: str,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> tuple[float, float]:
    """Pick one shell radius from raw scatter radii measured at one reference center."""
    rr_pos = np.asarray(rr, dtype=np.float64).reshape(-1)
    rr_pos = rr_pos[np.isfinite(rr_pos) & (rr_pos > 0.0)]
    if rr_pos.size == 0:
        return 0.0, 0.0

    bins_r = max(96, min(320, int(np.sqrt(rr_pos.size) * 3.0)))
    hi = float(np.quantile(rr_pos, 0.998))
    hi = max(hi, float(np.max(rr_pos)), 1.0)
    hist, edges = np.histogram(rr_pos, bins=bins_r, range=(0.0, hi))
    if hist.size == 0 or edges.size < 2:
        return float(np.median(rr_pos)), max(0.03 * float(np.median(rr_pos)), 0.12)

    profile = _smooth_profile_1d(hist.astype(np.float64, copy=False))
    if profile.size == 0 or float(np.max(profile)) <= 0.0:
        return float(np.median(rr_pos)), max(0.03 * float(np.median(rr_pos)), 0.12)

    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx_list: list[int] = []
    for idx in range(profile.size):
        left = profile[idx - 1] if idx > 0 else -np.inf
        right = profile[idx + 1] if idx + 1 < profile.size else -np.inf
        if profile[idx] >= left and profile[idx] >= right:
            peak_idx_list.append(idx)
    if not peak_idx_list:
        peak_idx_list = [int(np.argmax(profile))]

    max_profile = float(np.max(profile))
    mode = str(peak_mode).strip().lower()
    manual_target = target_radius is not None and np.isfinite(float(target_radius)) and float(target_radius) > 0.0
    bin_width = float(edges[1] - edges[0]) if edges.size >= 2 else max(0.03 * float(np.median(rr_pos)), 0.12)
    if manual_target:
        target_r = float(target_radius)
        if target_window is not None and np.isfinite(float(target_window)) and float(target_window) > 0.0:
            search_half_width = max(float(target_window), 2.0 * bin_width)
        else:
            search_half_width = max(6.0 * bin_width, 0.08 * target_r, 1.5)
        local_peaks = [idx for idx in peak_idx_list if abs(float(centers[idx]) - target_r) <= search_half_width]
        significant = [idx for idx in local_peaks if profile[idx] >= max(1.0, 0.04 * max_profile)]
        if significant:
            target_idx = int(min(significant, key=lambda idx: abs(float(centers[idx]) - target_r)))
        elif local_peaks:
            target_idx = int(min(local_peaks, key=lambda idx: abs(float(centers[idx]) - target_r)))
        else:
            in_window = np.where(np.abs(centers - target_r) <= search_half_width)[0]
            if in_window.size == 0:
                target_idx = int(np.argmin(np.abs(centers - target_r)))
            else:
                lo_local = int(in_window[0])
                hi_local = int(in_window[-1] + 1)
                target_idx = int(lo_local + np.argmax(profile[lo_local:hi_local]))
    elif mode == "outermost":
        significant = [idx for idx in peak_idx_list if profile[idx] >= max(3.0, 0.08 * max_profile)]
        target_idx = int(significant[-1] if significant else peak_idx_list[-1])
    else:
        significant = [idx for idx in peak_idx_list if profile[idx] >= max(3.0, 0.04 * max_profile)]
        candidates = significant if significant else peak_idx_list
        target_idx = int(max(candidates, key=lambda idx: float(profile[idx])))

    peak_r = float(centers[target_idx]) if centers.size > 0 else float(np.median(rr_pos))
    peak_level = float(profile[target_idx])
    cut_level = max(1.0, 0.45 * peak_level)
    lo = target_idx
    hi_idx = target_idx
    while lo > 0 and profile[lo - 1] >= cut_level:
        lo -= 1
    while hi_idx + 1 < profile.size and profile[hi_idx + 1] >= cut_level:
        hi_idx += 1

    shell_width = 0.5 * max(float(edges[hi_idx + 1] - edges[lo]), bin_width)
    shell_width = max(shell_width, 1.5 * bin_width, 0.03 * max(peak_r, 1.0), 0.12)
    if manual_target and target_window is not None and np.isfinite(float(target_window)) and float(target_window) > 0.0:
        shell_width = max(min(shell_width, float(target_window)), 1.5 * bin_width, 0.12)
    return max(peak_r, 1e-9), max(shell_width, 1e-9)


def _build_scatter_peak_line_model(
    points_xy: np.ndarray,
    reference_center: np.ndarray,
    *,
    peak_mode: str,
    theta_bins: int,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> dict[str, np.ndarray | float | int] | None:
    """Freeze one shell from the scatter cloud so later optimization is monotonic."""
    pts = np.asarray(points_xy, dtype=np.float64)
    center = np.asarray(reference_center, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0 or center.size < 2:
        return None

    theta_bins_local = max(72, min(int(theta_bins), 360))
    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    rr = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    peak_r, shell_width = _pick_scatter_shell_radius(
        rr,
        peak_mode,
        target_radius=target_radius,
        target_window=target_window,
    )
    if (not np.isfinite(peak_r)) or (not np.isfinite(shell_width)) or peak_r <= 0.0 or shell_width <= 0.0:
        return None

    seed_bins_all = _theta_bin_indices(theta, theta_bins_local)
    manual_band = target_radius is not None and target_window is not None and np.isfinite(float(target_radius)) and np.isfinite(float(target_window)) and float(target_radius) > 0.0 and float(target_window) > 0.0
    if manual_band:
        width_options = (max(float(target_window), 0.12),)
    else:
        width_options = (
            shell_width,
            max(shell_width * 1.5, 0.04 * peak_r),
            max(shell_width * 2.1, 0.06 * peak_r),
        )
    best_mask: np.ndarray | None = None
    best_rank = (-np.inf, -np.inf, -np.inf)
    min_keep = max(24, theta_bins_local // 8)
    for width in width_options:
        if manual_band:
            mask = np.abs(rr - float(target_radius)) <= float(width)
        else:
            mask = np.abs(rr - peak_r) <= float(width)
        n_keep = int(np.count_nonzero(mask))
        if n_keep < min_keep:
            continue
        counts = np.bincount(seed_bins_all[mask], minlength=theta_bins_local).astype(np.float64)
        coverage = float(np.mean(counts > 0.0)) if counts.size > 0 else 0.0
        density_score = min(1.0, float(n_keep) / float(max(theta_bins_local, 1)))
        rank = (
            coverage,
            density_score,
            -abs(float(width) - float(shell_width)),
        )
        if rank > best_rank:
            best_rank = rank
            best_mask = mask

    if best_mask is None:
        order = np.argsort(np.abs(rr - peak_r))
        n_pick = min(rr.size, max(48, 2 * theta_bins_local))
        if n_pick < 24:
            return None
        best_mask = np.zeros(rr.size, dtype=bool)
        best_mask[order[:n_pick]] = True

    shell_pts = pts[best_mask, :2]
    shell_rr = rr[best_mask]
    shell_theta = theta[best_mask]
    if shell_pts.shape[0] < 24:
        return None

    shell_bins = _theta_bin_indices(shell_theta, theta_bins_local)
    shell_counts = np.bincount(shell_bins, minlength=theta_bins_local).astype(np.float64)
    coverage = float(np.mean(shell_counts > 0.0)) if shell_counts.size > 0 else 0.0
    if coverage <= 0.08:
        return None

    mode = str(peak_mode).strip().lower()
    shell_scale = max(shell_width, 0.03 * max(peak_r, 1.0), 0.12)
    shell_resid = (shell_rr - peak_r) / max(shell_scale, 1e-12)
    radial_weight = np.exp(-0.5 * np.clip(shell_resid * shell_resid, 0.0, 30.0))
    angle_weight = 1.0 / np.maximum(shell_counts[shell_bins], 1.0)
    point_weight = radial_weight * angle_weight
    if manual_band and mode == "outermost":
        max_per_theta = 24
    elif mode == "outermost":
        max_per_theta = 8
    else:
        max_per_theta = 12
    keep_idx_list: list[np.ndarray] = []
    if shell_bins.size > 0:
        sort_order = np.argsort(shell_bins, kind="stable")
        counts_per_bin = np.bincount(shell_bins, minlength=theta_bins_local).astype(np.int64, copy=False)
        starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), counts_per_bin[:-1])))
        for b in range(theta_bins_local):
            start = int(starts[b])
            stop = int(start + counts_per_bin[b])
            if stop <= start:
                continue
            idx = sort_order[start:stop]
            if idx.size > max_per_theta:
                if mode == "outermost":
                    order = np.argsort(shell_rr[idx])
                else:
                    order = np.argsort(point_weight[idx])
                idx = idx[order[-max_per_theta:]]
            keep_idx_list.append(idx)
    if keep_idx_list:
        keep_idx = np.concatenate(keep_idx_list)
        shell_pts = shell_pts[keep_idx]
        shell_rr = shell_rr[keep_idx]
        shell_theta = shell_theta[keep_idx]
        shell_bins = shell_bins[keep_idx]
        point_weight = point_weight[keep_idx]

    if manual_band:
        band_lo = max(0.0, float(target_radius) - float(target_window))
        band_hi = max(band_lo + 1e-6, float(target_radius) + float(target_window))
    else:
        band_lo = max(0.0, float(peak_r) - 1.35 * float(shell_scale))
        band_hi = max(band_lo + 1e-6, float(peak_r) + (1.15 if mode == "outermost" else 0.85) * float(shell_scale))

    if manual_band and mode == "outermost":
        band_width = max(band_hi - band_lo, 1e-6)
        edge_rank = np.clip((shell_rr - band_lo) / band_width, 0.0, 1.0)
        point_weight = point_weight * (0.20 + 0.80 * edge_rank * edge_rank)

    total_weight = float(np.sum(point_weight))
    if (not np.isfinite(total_weight)) or total_weight <= 0.0:
        point_weight = np.full(shell_pts.shape[0], 1.0 / float(max(1, shell_pts.shape[0])), dtype=np.float64)
    else:
        point_weight = point_weight / total_weight

    theta_step = (2.0 * np.pi) / float(theta_bins_local)
    theta_centers = np.linspace(-np.pi, np.pi, theta_bins_local, endpoint=False, dtype=np.float64) + 0.5 * theta_step
    angular_sigma = max((2.4 if mode == "outermost" else 3.2) * theta_step, 0.10)
    kappa = 1.0 / max(angular_sigma * angular_sigma, 1e-9)
    beta_base = 8.0 if (mode == "outermost" and manual_band) else (4.8 if mode == "outermost" else 1.0)
    beta = beta_base / max(shell_scale, 1e-9)

    seed_delta = shell_theta[:, None] - theta_centers[None, :]
    seed_kernel = np.exp(kappa * (np.cos(seed_delta) - 1.0))
    seed_support = np.sum(point_weight[:, None] * seed_kernel, axis=0)
    support_floor = max(0.05 * float(np.max(seed_support)), 0.25 / float(max(theta_bins_local, 1)))
    anchor_mask = seed_support >= support_floor
    if np.count_nonzero(anchor_mask) < max(16, theta_bins_local // 6):
        order = np.argsort(seed_support)
        keep = min(theta_bins_local, max(16, theta_bins_local // 4))
        anchor_mask = np.zeros(theta_bins_local, dtype=bool)
        anchor_mask[order[-keep:]] = True
    anchor_centers = theta_centers[anchor_mask]
    anchor_support = seed_support[anchor_mask]
    anchor_total = float(np.sum(anchor_support))
    if (not np.isfinite(anchor_total)) or anchor_total <= 0.0:
        anchor_weight = np.full(anchor_centers.size, 1.0 / float(max(1, anchor_centers.size)), dtype=np.float64)
    else:
        anchor_weight = anchor_support / anchor_total
    edge_reference = float(peak_r if (manual_band and mode == "outermost") else (band_hi if mode == "outermost" else peak_r))
    band_softness = max(0.08, 0.20 * min(max(band_hi - band_lo, 1e-6), max(shell_scale, 0.6)))
    roughness_weight = 0.18 if (mode == "outermost" and manual_band) else (0.16 if mode == "outermost" else 0.04)
    curvature_weight = 0.55 if (mode == "outermost" and manual_band) else (0.12 if mode == "outermost" else 0.0)
    mean_penalty_weight = 0.10 if (mode == "outermost" and manual_band) else (0.70 if mode == "outermost" else 0.18)
    if manual_band and mode == "outermost":
        loss_mode = "roi_edge_quantile"
    elif manual_band:
        loss_mode = "roi_shell_variance"
    else:
        loss_mode = "ridge_line"
    sector_mass = np.bincount(shell_bins, weights=point_weight, minlength=theta_bins_local).astype(np.float64)
    sector_weight = sector_mass / max(float(np.sum(sector_mass)), 1e-12)

    return {
        "points": shell_pts.astype(np.float64, copy=False),
        "weights": point_weight.astype(np.float64, copy=False),
        "peak_radius": float(peak_r),
        "shell_width": float(shell_scale),
        "coverage": float(coverage),
        "peak_mode": mode,
        "anchor_centers": anchor_centers.astype(np.float64, copy=False),
        "anchor_weights": anchor_weight.astype(np.float64, copy=False),
        "seed_support": anchor_support.astype(np.float64, copy=False),
        "theta_centers": theta_centers.astype(np.float64, copy=False),
        "sector_count": int(theta_bins_local),
        "beta": float(beta),
        "kappa": float(kappa),
        "reference_radius": float(edge_reference),
        "mean_penalty_weight": float(mean_penalty_weight),
        "band_lo": float(band_lo),
        "band_hi": float(band_hi),
        "band_softness": float(band_softness),
        "roughness_weight": float(roughness_weight),
        "curvature_weight": float(curvature_weight),
        "loss_mode": loss_mode,
        "edge_beta": float(max(beta, 18.0 / max(band_hi - band_lo, 0.5))),
        "edge_quantile": float(0.75 if (manual_band and mode == "outermost") else 0.90),
        "sector_index": shell_bins.astype(np.int64, copy=False),
        "sector_weight": sector_weight.astype(np.float64, copy=False),
        "point_var_weight": float(0.30 if manual_band else 0.0),
        "sector_var_weight": float(0.85 if manual_band else 0.0),
        "edge_point_reg_weight": float(0.04 if (manual_band and mode == "outermost") else 0.0),
        "edge_sector_reg_weight": float(0.10 if (manual_band and mode == "outermost") else 0.0),
        "coverage_penalty_weight": float(0.18 if (manual_band and mode == "outermost") else 0.0),
    }


def _scatter_sector_edge_profile(
    rr_safe: np.ndarray,
    dr: np.ndarray,
    sector_index: np.ndarray,
    sector_count: int,
    weights: np.ndarray,
    *,
    edge_beta: float,
    reference_radius: float,
    sector_weight_base: np.ndarray,
    return_grad: bool = False,
) -> dict[str, np.ndarray | float] | None:
    """Evaluate one scatter-based outer-edge line from fixed theta sectors."""
    n_sector = max(int(sector_count), 0)
    if n_sector <= 0:
        return None
    idx = np.asarray(sector_index, dtype=np.int64).reshape(-1)
    rr = np.asarray(rr_safe, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if idx.size != rr.size or w.size != rr.size or dr.shape != (rr.size, 2):
        return None

    positive_mass = np.bincount(idx, weights=w, minlength=n_sector).astype(np.float64)
    positive = positive_mass[positive_mass > 0.0]
    mass_floor = max(1e-9, 0.25 * float(np.mean(positive)) if positive.size > 0 else 1e-9)
    edge_scale = float(max(edge_beta, 1e-9))
    z = w * np.exp(np.clip(edge_scale * (rr - float(reference_radius)), -60.0, 60.0))
    den = np.bincount(idx, weights=z, minlength=n_sector).astype(np.float64)
    num = np.bincount(idx, weights=z * rr, minlength=n_sector).astype(np.float64)
    valid_mask = (positive_mass > mass_floor) & np.isfinite(den) & (den > 1e-18)
    if np.count_nonzero(valid_mask) < max(10, n_sector // 8):
        return None

    edge_r = np.zeros(n_sector, dtype=np.float64)
    edge_r[valid_mask] = num[valid_mask] / den[valid_mask]

    base = np.asarray(sector_weight_base, dtype=np.float64).reshape(-1)
    if base.size != n_sector:
        base = positive_mass
    gamma = base[valid_mask]
    gamma_sum = float(np.sum(gamma))
    if (not np.isfinite(gamma_sum)) or gamma_sum <= 0.0:
        return None
    gamma = gamma / gamma_sum

    result: dict[str, np.ndarray | float] = {
        "edge_r": edge_r,
        "valid_mask": valid_mask,
        "gamma": gamma,
        "support": positive_mass,
    }
    if not return_grad:
        return result

    d_z = z[:, None] * edge_scale * dr
    d_den_x = np.bincount(idx, weights=d_z[:, 0], minlength=n_sector).astype(np.float64)
    d_den_y = np.bincount(idx, weights=d_z[:, 1], minlength=n_sector).astype(np.float64)
    d_num_x = np.bincount(idx, weights=(d_z[:, 0] * rr) + (z * dr[:, 0]), minlength=n_sector).astype(np.float64)
    d_num_y = np.bincount(idx, weights=(d_z[:, 1] * rr) + (z * dr[:, 1]), minlength=n_sector).astype(np.float64)
    d_edge = np.zeros((n_sector, 2), dtype=np.float64)
    d_edge[valid_mask, 0] = (d_num_x[valid_mask] - edge_r[valid_mask] * d_den_x[valid_mask]) / den[valid_mask]
    d_edge[valid_mask, 1] = (d_num_y[valid_mask] - edge_r[valid_mask] * d_den_y[valid_mask]) / den[valid_mask]
    result["d_edge"] = d_edge
    return result


def _weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return one weighted quantile from a 1D scatter sample."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vals.size == 0 or w.size != vals.size:
        return np.nan
    finite = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(finite):
        return np.nan
    vals = vals[finite]
    w = w[finite]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    total = float(np.sum(w))
    if (not np.isfinite(total)) or total <= 0.0:
        return np.nan
    cdf = np.cumsum(w)
    q = float(np.clip(quantile, 0.0, 1.0)) * total
    return float(np.interp(q, cdf, vals, left=vals[0], right=vals[-1]))


def _scatter_sector_quantile_profile(
    rr_safe: np.ndarray,
    theta: np.ndarray,
    sector_count: int,
    weights: np.ndarray,
    *,
    quantile: float,
    sector_weight_base: np.ndarray | None = None,
) -> dict[str, np.ndarray | float] | None:
    """Track one robust outer line from scatter radii using sector-wise quantiles."""
    n_sector = max(int(sector_count), 0)
    if n_sector <= 0:
        return None
    rr = np.asarray(rr_safe, dtype=np.float64).reshape(-1)
    tt = np.asarray(theta, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if rr.size == 0 or tt.size != rr.size or w.size != rr.size:
        return None

    idx = _theta_bin_indices(tt, n_sector)
    support = np.bincount(idx, weights=w, minlength=n_sector).astype(np.float64)
    counts = np.bincount(idx, minlength=n_sector).astype(np.int64)
    positive_support = support[support > 0.0]
    positive_counts = counts[counts > 0]
    support_floor = max(1e-9, 0.22 * float(np.mean(positive_support)) if positive_support.size > 0 else 1e-9)
    count_floor = max(4, int(np.floor(0.25 * float(np.mean(positive_counts))))) if positive_counts.size > 0 else 4

    edge_r = np.zeros(n_sector, dtype=np.float64)
    valid_mask = np.zeros(n_sector, dtype=bool)
    if idx.size > 0:
        order = np.argsort(idx, kind="stable")
        rr_sorted = rr[order]
        w_sorted = w[order]
        starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), counts[:-1])))
    else:
        rr_sorted = np.empty(0, dtype=np.float64)
        w_sorted = np.empty(0, dtype=np.float64)
        starts = np.zeros(n_sector, dtype=np.int64)

    for sector in range(n_sector):
        if support[sector] <= support_floor or counts[sector] < count_floor:
            continue
        start = int(starts[sector])
        stop = int(start + counts[sector])
        if stop <= start:
            continue
        # `order` sorted by sector keeps contiguous slices and avoids `idx == sector` scans.
        edge_value = _weighted_quantile_1d(rr_sorted[start:stop], w_sorted[start:stop], quantile)
        if np.isfinite(edge_value):
            edge_r[sector] = float(edge_value)
            valid_mask[sector] = True
    if np.count_nonzero(valid_mask) < max(10, n_sector // 8):
        return None

    base = np.asarray(sector_weight_base, dtype=np.float64).reshape(-1) if sector_weight_base is not None else support
    if base.size != n_sector:
        base = support
    gamma = base[valid_mask]
    gamma_sum = float(np.sum(gamma))
    if (not np.isfinite(gamma_sum)) or gamma_sum <= 0.0:
        return None
    gamma = gamma / gamma_sum
    return {
        "edge_r": edge_r,
        "valid_mask": valid_mask,
        "gamma": gamma,
        "support": support,
        "counts": counts.astype(np.float64),
    }


def _should_use_iterative_outer_roi_edge_fit(target_radius: float | None, target_window: float | None) -> bool:
    """Use the dedicated outer-edge circle fit only for reasonably narrow manual ROIs."""
    if target_radius is None or target_window is None:
        return False
    if not (np.isfinite(float(target_radius)) and np.isfinite(float(target_window))):
        return False
    target_r = float(target_radius)
    target_w = float(target_window)
    if target_r <= 0.0 or target_w <= 0.0:
        return False
    return target_w <= max(2.5, 0.08 * target_r)


def _pick_smooth_polar_edge_path(
    contrast_map: np.ndarray,
    inside_map: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    r_step: float,
    band_width: float,
    target_row: int,
    max_iters: int = 8,
) -> np.ndarray:
    """Pick one smooth circular edge path across theta columns from per-pixel contrast."""
    contrast = np.asarray(contrast_map, dtype=np.float64)
    inside = np.asarray(inside_map, dtype=np.float64)
    rows = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    if contrast.ndim != 2 or inside.shape != contrast.shape or rows.size != contrast.shape[0] or contrast.shape[1] == 0:
        return np.zeros(0, dtype=np.int64)

    positive_contrast = contrast[contrast > 0.0]
    contrast_scale = float(np.quantile(positive_contrast, 0.90)) if positive_contrast.size > 0 else 1.0
    contrast_scale = max(contrast_scale, 1e-9)
    positive_inside = inside[inside > 0.0]
    inside_scale = float(np.quantile(positive_inside, 0.90)) if positive_inside.size > 0 else 1.0
    inside_scale = max(inside_scale, 1e-9)

    score = np.clip(contrast / contrast_scale, 0.0, None) + (0.10 * np.clip(inside / inside_scale, 0.0, None))
    if not np.any(score > 0.0):
        nearest = int(np.argmin(np.abs(rows - int(target_row))))
        return np.full(contrast.shape[1], rows[nearest], dtype=np.int64)

    target_penalty = ((rows.astype(np.float64) - float(target_row)) / max(4.0, 0.75 * max(float(np.ptp(rows)), 1.0))) ** 2
    score = score - (0.03 * target_penalty[:, None])

    path_idx = np.argmax(score, axis=0).astype(np.int64)
    path_rows = rows[path_idx]
    jump_scale = max(1.5, min(4.0, max(2.0 * float(r_step), 0.12 * float(band_width)) / max(float(r_step), 1e-9)))
    max_jump_bins = max(3, int(np.ceil(jump_scale + 1.0)))
    smooth_weight = 0.42
    curvature_weight = 0.18

    for _ in range(max(1, int(max_iters))):
        prev_rows = path_rows.copy()
        changed = False
        for col in range(score.shape[1]):
            prev_row = prev_rows[(col - 1) % score.shape[1]]
            next_row = prev_rows[(col + 1) % score.shape[1]]
            center_row = 0.5 * (float(prev_row) + float(next_row))
            delta_prev = (rows.astype(np.float64) - float(prev_row)) / jump_scale
            delta_next = (rows.astype(np.float64) - float(next_row)) / jump_scale
            delta_center = (rows.astype(np.float64) - center_row) / jump_scale
            allowed = (np.abs(rows - int(round(center_row))) <= max_jump_bins)
            objective = (
                score[:, col]
                - (smooth_weight * (delta_prev * delta_prev + delta_next * delta_next))
                - (curvature_weight * (delta_center * delta_center))
            )
            objective = np.where(allowed, objective, -np.inf)
            best_idx = int(np.argmax(objective))
            best_row = int(rows[best_idx])
            if best_row != int(path_rows[col]):
                path_rows[col] = best_row
                path_idx[col] = best_idx
                changed = True
        if not changed:
            break

    return path_rows.astype(np.int64, copy=False)


def _scatter_roi_outer_edge_profile(
    points_xy: np.ndarray,
    center_xy: tuple[float, float] | np.ndarray,
    *,
    theta_bins: int,
    target_radius: float,
    target_window: float,
    sample_limit: int = 48_000,
) -> dict[str, np.ndarray | float] | None:
    """Detect one manual-band outer edge by local outward step contrast per theta."""
    pts = np.asarray(points_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0 or center.size < 2:
        return None

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite, :2]
    if pts.shape[0] == 0:
        return None

    band_lo = max(0.0, float(target_radius) - float(target_window))
    band_hi = max(band_lo + 1e-6, float(target_radius) + float(target_window))
    search_lo = float(band_lo)
    search_hi = float(band_hi)
    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    rr = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    band_mask = np.isfinite(rr) & np.isfinite(theta) & (rr >= search_lo) & (rr <= search_hi)
    if np.count_nonzero(band_mask) < 48:
        return None

    rr_band = rr[band_mask]
    theta_band = theta[band_mask]
    theta_bins_local = max(96, min(int(theta_bins), 360))
    theta_edges = np.linspace(-np.pi, np.pi, theta_bins_local + 1, dtype=np.float64)
    theta_step = (2.0 * np.pi) / float(theta_bins_local)
    theta_centers = theta_edges[:-1] + 0.5 * theta_step
    if sample_limit > 0 and rr_band.size > int(sample_limit):
        theta_index = _theta_bin_indices(theta_band, theta_bins_local)
        target_per_sector = max(1, int(np.ceil(float(sample_limit) / float(theta_bins_local))))
        counts_per_sector = np.bincount(theta_index, minlength=theta_bins_local).astype(np.int64, copy=False)
        order = np.argsort(theta_index, kind="stable")
        starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), counts_per_sector[:-1])))
        keep_idx_list: list[np.ndarray] = []
        for sector in range(theta_bins_local):
            start = int(starts[sector])
            stop = int(start + counts_per_sector[sector])
            if stop <= start:
                continue
            idx = order[start:stop]
            if idx.size <= target_per_sector:
                keep_idx_list.append(idx)
                continue
            take = np.linspace(0, idx.size - 1, num=target_per_sector, dtype=np.float64)
            keep_idx_list.append(idx[np.rint(take).astype(np.int64)])
        if keep_idx_list:
            keep_idx = np.concatenate(keep_idx_list)
            rr_band = rr_band[keep_idx]
            theta_band = theta_band[keep_idx]
        else:
            return None
        if rr_band.size < 48:
            return None

    radial_bins = int(np.clip(np.ceil(24.0 * max(search_hi - search_lo, 1.0)), 96, 260))
    r_edges = np.linspace(search_lo, search_hi, radial_bins + 1, dtype=np.float64)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_step = float(r_edges[1] - r_edges[0]) if r_edges.size >= 2 else max((search_hi - search_lo) / max(radial_bins, 1), 1e-6)
    band_width = max(band_hi - band_lo, 1e-6)

    hist2d, _r_tmp, _theta_tmp = np.histogram2d(rr_band, theta_band, bins=[r_edges, theta_edges])
    hist2d = hist2d.astype(np.float64, copy=False)
    if hist2d.size == 0 or float(np.sum(hist2d)) <= 0.0:
        return None

    hist_smooth = _smooth_profile_axis0_kernel5(hist2d)
    if hist_smooth.shape[1] >= 5:
        # Theta is periodic, so denoise across neighboring columns with wraparound.
        hist_smooth = (
            np.roll(hist_smooth, 2, axis=1)
            + (2.0 * np.roll(hist_smooth, 1, axis=1))
            + (3.0 * hist_smooth)
            + (2.0 * np.roll(hist_smooth, -1, axis=1))
            + np.roll(hist_smooth, -2, axis=1)
        ) / 9.0
    half_window_bins = int(np.clip(np.round(0.06 * band_width / max(r_step, 1e-9)), 2, 7))
    edge_positions = r_edges[1:-1]
    if hist_smooth.shape[0] <= (2 * half_window_bins):
        return None
    candidate_idx = np.arange(half_window_bins - 1, hist_smooth.shape[0] - half_window_bins, dtype=np.int64)
    if candidate_idx.size == 0:
        return None
    candidate_r = edge_positions[candidate_idx]

    edge_r = np.zeros(theta_bins_local, dtype=np.float64)
    edge_theta = theta_centers.astype(np.float64, copy=True)
    peak_counts = np.zeros(theta_bins_local, dtype=np.float64)
    peak_mask = np.zeros(theta_bins_local, dtype=bool)
    contrast_map = np.zeros((candidate_idx.size, theta_bins_local), dtype=np.float64)
    inside_map = np.zeros((candidate_idx.size, theta_bins_local), dtype=np.float64)
    fallback_local = np.full(theta_bins_local, -1, dtype=np.int64)
    best_contrast_by_sector = np.zeros(theta_bins_local, dtype=np.float64)
    valid_local_candidates = [np.zeros(0, dtype=np.int64) for _ in range(theta_bins_local)]
    strong_local_candidates = [np.zeros(0, dtype=np.int64) for _ in range(theta_bins_local)]
    for sector in range(theta_bins_local):
        col = hist_smooth[:, sector]
        if col.size == 0:
            continue
        col_max = float(np.max(col))
        if (not np.isfinite(col_max)) or col_max <= 0.0:
            continue

        prefix = np.concatenate(([0.0], np.cumsum(col, dtype=np.float64)))
        inside_lo = candidate_idx - half_window_bins + 1
        inside_hi = candidate_idx + 1
        outside_lo = candidate_idx + 1
        outside_hi = candidate_idx + 1 + half_window_bins

        inside_sum = prefix[inside_hi] - prefix[inside_lo]
        outside_sum = prefix[outside_hi] - prefix[outside_lo]
        inside_mean = inside_sum / float(half_window_bins)
        outside_mean = outside_sum / float(half_window_bins)
        contrast = np.clip(inside_mean - outside_mean, 0.0, None)
        contrast = np.where(np.isfinite(contrast), contrast, 0.0)
        if contrast.size == 0:
            continue

        inside_cut = max(1e-9, 0.12 * col_max)
        valid_local = (inside_mean >= inside_cut) & (contrast > 0.0)
        valid_idx = np.flatnonzero(valid_local)
        if valid_idx.size == 0:
            continue

        best_contrast = float(np.max(contrast[valid_idx]))
        if best_contrast < max(1e-9, 0.06 * col_max):
            continue
        best_contrast_by_sector[sector] = best_contrast
        valid_local_candidates[sector] = valid_idx.astype(np.int64, copy=False)
        contrast_map[:, sector] = np.where(valid_local, contrast, 0.0)
        inside_map[:, sector] = np.where(valid_local, inside_mean, 0.0)

        local_peak_mask = np.ones_like(contrast, dtype=bool)
        if contrast.size >= 2:
            local_peak_mask[1:] &= contrast[1:] >= contrast[:-1]
            local_peak_mask[:-1] &= contrast[:-1] <= contrast[1:]
        peak_local = np.flatnonzero(valid_local & local_peak_mask)
        if peak_local.size == 0:
            peak_local = valid_idx

        strong_local = peak_local[contrast[peak_local] >= 0.75 * best_contrast]
        if strong_local.size == 0:
            strong_local = np.array([int(peak_local[np.argmax(contrast[peak_local])])], dtype=np.int64)
        strong_local_candidates[sector] = strong_local.astype(np.int64, copy=False)
        fallback_local[sector] = int(np.max(strong_local))

    smooth_rows = np.zeros(0, dtype=np.int64)
    if np.any(contrast_map > 0.0):
        outer_target_r = min(
            search_hi - (0.5 * r_step),
            max(search_lo, band_hi - max(1.5 * r_step, 0.10 * band_width)),
        )
        target_local = int(np.argmin(np.abs(candidate_r - outer_target_r)))
        smooth_rows = _pick_smooth_polar_edge_path(
            contrast_map,
            inside_map,
            candidate_idx,
            r_step=r_step,
            band_width=band_width,
            target_row=int(candidate_idx[target_local]),
            max_iters=10,
        )
    path_gap_bins = max(2, half_window_bins + 1)

    for sector in range(theta_bins_local):
        chosen_local = int(fallback_local[sector])
        candidate_pool = strong_local_candidates[sector]
        if candidate_pool.size == 0:
            candidate_pool = valid_local_candidates[sector]
        if smooth_rows.size == theta_bins_local and candidate_pool.size > 0:
            path_row = int(smooth_rows[sector])
            nearby = candidate_pool[np.abs(candidate_idx[candidate_pool] - path_row) <= path_gap_bins]
            if nearby.size > 0:
                candidate_pool = nearby
            sector_contrast = np.maximum(contrast_map[candidate_pool, sector], 0.0)
            if np.any(sector_contrast > 0.0):
                distance = np.abs(candidate_idx[candidate_pool].astype(np.float64) - float(path_row))
                contrast_bonus = sector_contrast / max(best_contrast_by_sector[sector], 1e-9)
                objective = distance - (0.25 * contrast_bonus)
                order = np.lexsort((-candidate_idx[candidate_pool], objective))
                chosen_local = int(candidate_pool[int(order[0])])
        if chosen_local < 0:
            continue

        chosen_idx = int(candidate_idx[chosen_local])
        refine_lo = max(0, chosen_local - 1)
        refine_hi = min(candidate_idx.size, chosen_local + 2)
        refine_idx = np.arange(refine_lo, refine_hi, dtype=np.int64)
        if smooth_rows.size == theta_bins_local:
            path_row = int(smooth_rows[sector])
            close_to_path = np.abs(candidate_idx[refine_idx] - path_row) <= max(path_gap_bins, 2)
            if np.any(close_to_path):
                refine_idx = refine_idx[close_to_path]
        refine_w = np.maximum(contrast_map[refine_idx, sector], 0.0)
        if np.any(refine_w > 0.0):
            edge_val = float(np.average(candidate_r[refine_idx], weights=refine_w))
            support_val = float(np.sum(refine_w) * float(half_window_bins))
        else:
            edge_val = float(edge_positions[chosen_idx])
            support_val = float(max(best_contrast_by_sector[sector], 1e-9) * float(half_window_bins))
        edge_r[sector] = edge_val
        edge_theta[sector] = float(theta_centers[sector])
        peak_counts[sector] = max(support_val, 1e-9)
        peak_mask[sector] = True

    valid_sectors = np.flatnonzero(peak_mask)
    if valid_sectors.size >= max(16, theta_bins_local // 10):
        prev_r = edge_r.copy()
        prev_support = peak_counts.copy()
        spike_tol = max(2.5 * r_step, 0.22 * band_width)
        for sector in valid_sectors:
            neigh_r = []
            neigh_s = []
            for off in (-2, -1, 1, 2):
                nbr = (int(sector) + off) % theta_bins_local
                if peak_mask[nbr]:
                    neigh_r.append(prev_r[nbr])
                    neigh_s.append(prev_support[nbr])
            if len(neigh_r) < 3:
                continue
            local_med = float(np.median(np.asarray(neigh_r, dtype=np.float64)))
            local_support = float(np.median(np.asarray(neigh_s, dtype=np.float64)))
            if abs(prev_r[sector] - local_med) > spike_tol and prev_support[sector] < 0.70 * max(local_support, 1e-9):
                edge_r[sector] = local_med

    if np.count_nonzero(peak_mask) < max(16, theta_bins_local // 10):
        return None
    weights = peak_counts[peak_mask]
    if np.any(weights > 0.0):
        weights = weights / max(float(np.sum(weights)), 1e-12)
    else:
        weights = np.full(np.count_nonzero(peak_mask), 1.0 / float(np.count_nonzero(peak_mask)), dtype=np.float64)
    peak_r_valid = edge_r[peak_mask]
    peak_r_mean = float(np.average(peak_r_valid, weights=weights))
    peak_r_std = float(np.sqrt(np.average((peak_r_valid - peak_r_mean) ** 2, weights=weights)))
    valid_theta_fraction = float(np.mean(peak_mask))
    return {
        "center_xy": np.asarray(center[:2], dtype=np.float64),
        "theta_centers_deg": np.degrees(theta_centers),
        "edge_theta_rad": edge_theta,
        "peak_r": edge_r,
        "peak_counts": peak_counts,
        "peak_mask": peak_mask,
        "peak_r_mean": peak_r_mean,
        "peak_r_std": peak_r_std,
        "valid_theta_fraction": valid_theta_fraction,
        "search_lo": float(search_lo),
        "band_lo": float(band_lo),
        "band_hi": float(band_hi),
    }


def _outer_edge_profile_loss(profile: dict[str, np.ndarray | float]) -> float:
    """Score how flat and complete one detected outer edge line is."""
    peak_r = np.asarray(profile.get("peak_r", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    peak_mask = np.asarray(profile.get("peak_mask", np.zeros(0, dtype=bool)), dtype=bool)
    peak_counts = np.asarray(profile.get("peak_counts", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if peak_r.size == 0 or peak_mask.size != peak_r.size or peak_counts.size != peak_r.size:
        return np.inf
    if np.count_nonzero(peak_mask) < max(16, peak_r.size // 10):
        return np.inf
    weights = peak_counts[peak_mask]
    if np.any(weights > 0.0):
        weights = weights / max(float(np.sum(weights)), 1e-12)
    else:
        weights = np.full(np.count_nonzero(peak_mask), 1.0 / float(np.count_nonzero(peak_mask)), dtype=np.float64)
    peak_r_valid = peak_r[peak_mask]
    mean_r = float(np.average(peak_r_valid, weights=weights))
    if (not np.isfinite(mean_r)) or mean_r <= 0.0:
        return np.inf
    edge_var = float(np.average((peak_r_valid - mean_r) ** 2, weights=weights))
    loss = edge_var / max(mean_r * mean_r, 1e-12)

    pair_mask = peak_mask & np.roll(peak_mask, -1)
    if np.any(pair_mask):
        diff = np.roll(peak_r, -1) - peak_r
        pair_weights = np.sqrt(np.maximum(peak_counts * np.roll(peak_counts, -1), 1e-12))
        pair_weights = pair_weights[pair_mask]
        if np.any(pair_weights > 0.0):
            pair_weights = pair_weights / max(float(np.sum(pair_weights)), 1e-12)
        else:
            pair_weights = np.full(np.count_nonzero(pair_mask), 1.0 / float(np.count_nonzero(pair_mask)), dtype=np.float64)
        loss += 0.14 * float(np.average(diff[pair_mask] * diff[pair_mask], weights=pair_weights)) / max(mean_r * mean_r, 1e-12)

    trip_mask = peak_mask & np.roll(peak_mask, -1) & np.roll(peak_mask, -2)
    if np.any(trip_mask):
        curve = np.roll(peak_r, -2) - (2.0 * np.roll(peak_r, -1)) + peak_r
        trip_weights = np.cbrt(
            np.maximum(
                peak_counts * np.roll(peak_counts, -1) * np.roll(peak_counts, -2),
                1e-12,
            )
        )
        trip_weights = trip_weights[trip_mask]
        if np.any(trip_weights > 0.0):
            trip_weights = trip_weights / max(float(np.sum(trip_weights)), 1e-12)
        else:
            trip_weights = np.full(np.count_nonzero(trip_mask), 1.0 / float(np.count_nonzero(trip_mask)), dtype=np.float64)
        loss += 0.40 * float(np.average(curve[trip_mask] * curve[trip_mask], weights=trip_weights)) / max(mean_r * mean_r, 1e-12)

    valid_theta_fraction = float(profile.get("valid_theta_fraction", 0.0))
    loss += 0.30 * max(0.0, 1.0 - valid_theta_fraction) ** 2
    return float(loss)


def _fit_circle_from_outer_edge_profile(
    profile: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, float] | None:
    """Fit a circle to one detected outer edge line with light robust trimming."""
    peak_r = np.asarray(profile.get("peak_r", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    peak_mask = np.asarray(profile.get("peak_mask", np.zeros(0, dtype=bool)), dtype=bool)
    peak_counts = np.asarray(profile.get("peak_counts", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    edge_theta = np.asarray(profile.get("edge_theta_rad", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    center = np.asarray(profile.get("center_xy", np.zeros(2, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if (
        peak_r.size == 0
        or peak_mask.size != peak_r.size
        or peak_counts.size != peak_r.size
        or edge_theta.size != peak_r.size
        or center.size < 2
        or np.count_nonzero(peak_mask) < max(16, peak_r.size // 10)
    ):
        return None

    theta_valid = edge_theta[peak_mask]
    r_valid = peak_r[peak_mask]
    weight_valid = peak_counts[peak_mask]
    edge_points = np.column_stack(
        (
            center[0] + r_valid * np.cos(theta_valid),
            center[1] + r_valid * np.sin(theta_valid),
        )
    )
    if edge_points.shape[0] < 3:
        return None

    fit = circle_fit_kasa(edge_points, weight_valid)
    if fit is None:
        return None
    fit_center = np.array([fit[0], fit[1]], dtype=np.float64)
    fit_radius = float(fit[2])
    for _ in range(2):
        residual = np.abs(np.hypot(edge_points[:, 0] - fit_center[0], edge_points[:, 1] - fit_center[1]) - fit_radius)
        if residual.size < 12:
            break
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        cutoff = max(float(np.quantile(residual, 0.86)), med + 2.5 * max(mad, 1e-6))
        keep = residual <= cutoff
        if np.count_nonzero(keep) < max(12, edge_points.shape[0] // 2) or np.count_nonzero(~keep) == 0:
            break
        fit = circle_fit_kasa(edge_points[keep], weight_valid[keep])
        if fit is None:
            break
        fit_center = np.array([fit[0], fit[1]], dtype=np.float64)
        fit_radius = float(fit[2])
    return fit_center, fit_radius


def _iterative_outer_roi_edge_circle_center(
    points_xy: np.ndarray,
    fallback_xy: tuple[float, float] | np.ndarray,
    *,
    theta_bins: int,
    sample_limit: int,
    target_radius: float,
    target_window: float,
) -> np.ndarray:
    """Iteratively detect the manual-band outer edge and fit a circle to it."""
    pts = np.asarray(points_xy, dtype=np.float64)
    fallback = np.asarray(fallback_xy, dtype=np.float64).reshape(-1)
    if fallback.size < 2:
        fallback = np.array([0.0, 0.0], dtype=np.float64)
    else:
        fallback = fallback[:2].astype(np.float64, copy=False)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0:
        return fallback.copy()

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite, :2]
    if pts.shape[0] == 0:
        return fallback.copy()
    pts_full = pts
    pts_seed = pts_full
    if sample_limit > 0 and pts_full.shape[0] > int(sample_limit):
        step = max(1, int(np.ceil(float(pts_full.shape[0]) / float(sample_limit))))
        pts_seed = pts_full[::step]

    theta_bins_local = max(90, min(int(theta_bins), 180))
    coarse_center = quadrant_symmetry_center(pts_seed, fallback)
    edge_seed = edge_circle_center(pts_seed, coarse_center, angle_bins=theta_bins_local)
    candidate_centers = [
        fallback.astype(np.float64, copy=True),
        coarse_center.astype(np.float64, copy=True),
        edge_seed.astype(np.float64, copy=True),
    ]

    best_center = fallback.astype(np.float64, copy=True)
    best_loss = np.inf
    current_profile: dict[str, np.ndarray | float] | None = None
    for candidate in candidate_centers:
        profile = _scatter_roi_outer_edge_profile(
            pts_full,
            candidate,
            theta_bins=theta_bins_local,
            target_radius=target_radius,
            target_window=target_window,
            sample_limit=sample_limit,
        )
        if profile is None:
            continue
        loss = _outer_edge_profile_loss(profile)
        if loss + 1e-12 < best_loss:
            best_loss = float(loss)
            best_center = candidate.astype(np.float64, copy=True)
            current_profile = profile
    if current_profile is None:
        return edge_seed.astype(np.float64, copy=False)

    # Keep this permissive enough so small-but-consistent improvements can accumulate.
    min_rel_improvement = 6.0e-4
    move_tol = max(1e-3, 1.5e-4 * max(float(target_radius), 1.0))
    max_step = max(0.35, min(3.2, 0.14 * float(target_radius) + 0.30 * float(target_window)))
    for _ in range(14):
        fit = _fit_circle_from_outer_edge_profile(current_profile)
        if fit is None:
            break
        fit_center = np.asarray(fit[0], dtype=np.float64).reshape(2)
        delta = fit_center - best_center
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= move_tol:
            break
        if delta_norm > max_step:
            delta = delta * (max_step / delta_norm)
            fit_center = best_center + delta

        best_trial_center = best_center.copy()
        best_trial_profile = current_profile
        best_trial_loss = best_loss
        for alpha in (1.0, 0.65, 0.40, 0.20, 0.10):
            candidate = best_center + alpha * (fit_center - best_center)
            profile = _scatter_roi_outer_edge_profile(
                pts_full,
                candidate,
                theta_bins=theta_bins_local,
                target_radius=target_radius,
                target_window=target_window,
                sample_limit=sample_limit,
            )
            if profile is None:
                continue
            loss = _outer_edge_profile_loss(profile)
            if loss + 1e-12 < best_trial_loss:
                best_trial_center = candidate.astype(np.float64, copy=True)
                best_trial_profile = profile
                best_trial_loss = float(loss)
        if best_trial_loss + 1e-10 >= best_loss * (1.0 - min_rel_improvement):
            break
        shift = float(np.linalg.norm(best_trial_center - best_center))
        best_center = best_trial_center
        best_loss = best_trial_loss
        current_profile = best_trial_profile
        if shift <= move_tol:
            break

    return best_center


def _build_scatter_peak_line_profile(
    points_xy: np.ndarray,
    center_xy: tuple[float, float] | np.ndarray,
    *,
    peak_mode: str,
    theta_bins: int,
    sample_limit: int = 48_000,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> dict[str, np.ndarray | float] | None:
    """Build a scatter-based polar edge line around one center for display/diagnostics."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0:
        return None
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite, :2]
    if pts.shape[0] == 0:
        return None
    if sample_limit > 0 and pts.shape[0] > int(sample_limit):
        step = max(1, int(np.ceil(float(pts.shape[0]) / float(sample_limit))))
        pts = pts[::step]
    center = np.asarray(center_xy, dtype=np.float64).reshape(-1)
    if center.size < 2:
        return None

    manual_outer_band = (
        str(peak_mode).strip().lower() == "outermost"
        and target_radius is not None
        and target_window is not None
        and np.isfinite(float(target_radius))
        and np.isfinite(float(target_window))
        and float(target_radius) > 0.0
        and float(target_window) > 0.0
    )
    if manual_outer_band and _should_use_iterative_outer_roi_edge_fit(target_radius, target_window):
        profile = _scatter_roi_outer_edge_profile(
            pts,
            center[:2],
            theta_bins=int(theta_bins),
            target_radius=float(target_radius),
            target_window=float(target_window),
            sample_limit=sample_limit,
        )
        if profile is None:
            return None
        peak_r = np.asarray(profile["peak_r"], dtype=np.float64)
        peak_counts = np.asarray(profile["peak_counts"], dtype=np.float64)
        peak_mask = np.asarray(profile["peak_mask"], dtype=bool)
        theta_centers_deg = np.asarray(profile["theta_centers_deg"], dtype=np.float64)
        peak_r_valid = peak_r[peak_mask]
        if peak_r_valid.size == 0:
            return None
        return {
            "theta_centers_deg": theta_centers_deg,
            "peak_r": peak_r,
            "peak_counts": peak_counts,
            "peak_mask": peak_mask,
            "peak_r_mean": float(profile["peak_r_mean"]),
            "peak_r_std": float(profile["peak_r_std"]),
            "valid_theta_fraction": float(profile["valid_theta_fraction"]),
            "straightness_score": float(_outer_edge_profile_loss(profile)),
        }

    profile_theta_bins = max(72, min(int(theta_bins), 180)) if manual_outer_band else int(theta_bins)
    model = _build_scatter_peak_line_model(
        pts,
        center[:2],
        peak_mode=peak_mode,
        theta_bins=profile_theta_bins,
        target_radius=target_radius,
        target_window=target_window,
    )
    if model is None:
        return None

    dx = np.asarray(model["points"], dtype=np.float64)[:, 0] - float(center[0])
    dy = np.asarray(model["points"], dtype=np.float64)[:, 1] - float(center[1])
    rr = np.hypot(dx, dy)
    rr_safe = np.maximum(rr, 1e-12)
    dr = np.column_stack((-dx / rr_safe, -dy / rr_safe))
    loss_mode = str(model.get("loss_mode", "ridge_line")).strip().lower()
    theta_centers = np.asarray(model.get("theta_centers", np.zeros(0, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if theta_centers.size == 0:
        return None

    if loss_mode == "roi_edge_quantile":
        quantile_profile = _scatter_sector_quantile_profile(
            rr_safe,
            np.arctan2(dy, dx),
            int(model.get("sector_count", theta_centers.size)),
            np.asarray(model["weights"], dtype=np.float64),
            quantile=float(model.get("edge_quantile", 0.75)),
            sector_weight_base=None,
        )
        if quantile_profile is None:
            return None
        peak_r = np.asarray(quantile_profile["edge_r"], dtype=np.float64)
        peak_mask = np.asarray(quantile_profile["valid_mask"], dtype=bool)
        peak_counts = np.asarray(quantile_profile["support"], dtype=np.float64)
    elif loss_mode in {"roi_edge_guided", "roi_edge_softmax"} or manual_outer_band:
        profile = _scatter_sector_edge_profile(
            rr_safe,
            dr,
            np.asarray(model.get("sector_index", np.zeros(0, dtype=np.int64)), dtype=np.int64),
            int(model.get("sector_count", theta_centers.size)),
            np.asarray(model["weights"], dtype=np.float64),
            edge_beta=float(model.get("edge_beta", model.get("beta", 1.0))),
            reference_radius=float(model.get("reference_radius", model.get("peak_radius", 1.0))),
            sector_weight_base=np.asarray(model.get("sector_weight", np.zeros(theta_centers.size, dtype=np.float64)), dtype=np.float64),
            return_grad=False,
        )
        if profile is None:
            return None
        peak_r = np.asarray(profile["edge_r"], dtype=np.float64)
        peak_mask = np.asarray(profile["valid_mask"], dtype=bool)
        peak_counts = np.asarray(profile["support"], dtype=np.float64)
    else:
        theta = np.arctan2(dy, dx)
        anchors = np.asarray(model.get("anchor_centers", np.zeros(0, dtype=np.float64)), dtype=np.float64).reshape(-1)
        anchor_weights = np.asarray(model.get("anchor_weights", np.zeros(0, dtype=np.float64)), dtype=np.float64).reshape(-1)
        if anchors.size == 0 or anchor_weights.size != anchors.size:
            return None
        beta = float(model.get("beta", 0.0))
        kappa = float(model.get("kappa", 1.0))
        ref_r = float(model.get("reference_radius", model.get("peak_radius", 1.0)))
        band_lo = float(model.get("band_lo", 0.0))
        band_hi = float(model.get("band_hi", np.inf))
        band_softness = float(model.get("band_softness", 0.0))
        delta = theta[:, None] - anchors[None, :]
        angular_kernel = np.exp(kappa * (np.cos(delta) - 1.0))
        radial_kernel = np.exp(np.clip(beta * (rr_safe - ref_r), -60.0, 60.0))[:, None]
        band_gate = np.ones_like(rr_safe, dtype=np.float64)
        if np.isfinite(band_lo) and np.isfinite(band_hi) and band_hi > band_lo and band_softness > 0.0:
            sigmoid_lo = 1.0 / (1.0 + np.exp(-np.clip((rr_safe - band_lo) / band_softness, -60.0, 60.0)))
            sigmoid_hi = 1.0 / (1.0 + np.exp(-np.clip((band_hi - rr_safe) / band_softness, -60.0, 60.0)))
            band_gate = np.clip(sigmoid_lo * sigmoid_hi, 1e-12, 1.0)
        z = np.asarray(model["weights"], dtype=np.float64)[:, None] * angular_kernel * radial_kernel * band_gate[:, None]
        den = np.sum(z, axis=0)
        if np.any(~np.isfinite(den)):
            return None
        support_floor = max(1e-18, 0.03 * float(np.max(den)))
        peak_mask = den > support_floor
        if np.count_nonzero(peak_mask) == 0:
            return None
        peak_r = np.zeros(anchors.size, dtype=np.float64)
        peak_r[peak_mask] = np.sum(z[:, peak_mask] * rr_safe[:, None], axis=0) / den[peak_mask]
        peak_counts = den
        theta_centers = anchors

    if np.count_nonzero(peak_mask) == 0:
        return None
    weights = peak_counts[peak_mask]
    total = float(np.sum(weights))
    if (not np.isfinite(total)) or total <= 0.0:
        weights = np.ones(np.count_nonzero(peak_mask), dtype=np.float64)
    peak_r_valid = peak_r[peak_mask]
    peak_r_mean = float(np.average(peak_r_valid, weights=weights))
    peak_r_std = float(np.sqrt(np.average((peak_r_valid - peak_r_mean) ** 2, weights=weights)))
    valid_theta_fraction = float(np.mean(peak_mask))
    coverage_penalty = 0.25 if str(peak_mode).strip().lower() == "outermost" else 0.75
    straightness_score = float((peak_r_std / max(peak_r_mean, 1.0)) + coverage_penalty * (1.0 - valid_theta_fraction))
    return {
        "theta_centers_deg": np.degrees(theta_centers),
        "peak_r": peak_r,
        "peak_counts": peak_counts,
        "peak_mask": peak_mask,
        "peak_r_mean": peak_r_mean,
        "peak_r_std": peak_r_std,
        "valid_theta_fraction": valid_theta_fraction,
        "straightness_score": straightness_score,
    }


def _scatter_shell_bulk_loss_grad(
    rr_safe: np.ndarray,
    dr: np.ndarray,
    sector_index: np.ndarray,
    weights: np.ndarray,
    *,
    ref_r: float,
    point_var_weight: float,
    sector_var_weight: float,
    mean_penalty_weight: float,
) -> tuple[float, np.ndarray, float]:
    """Bulk shell regularizer used to stabilize edge-driven optimization."""
    idx = np.asarray(sector_index, dtype=np.int64).reshape(-1)
    w_raw = np.asarray(weights, dtype=np.float64).reshape(-1)
    if idx.size != rr_safe.size or w_raw.size != rr_safe.size:
        return np.inf, np.zeros(2, dtype=np.float64), 0.0
    weight_sum = float(np.sum(w_raw))
    if (not np.isfinite(weight_sum)) or weight_sum <= 0.0:
        return np.inf, np.zeros(2, dtype=np.float64), 0.0
    w = w_raw / weight_sum

    mean_r = float(np.sum(w * rr_safe))
    if (not np.isfinite(mean_r)) or mean_r <= 0.0:
        return np.inf, np.zeros(2, dtype=np.float64), 0.0
    d_mean = np.sum(w[:, None] * dr, axis=0)
    second_moment = float(np.sum(w * rr_safe * rr_safe))
    radial_var = max(second_moment - mean_r * mean_r, 0.0)
    d_second_moment = 2.0 * np.sum((w * rr_safe)[:, None] * dr, axis=0)
    d_radial_var = d_second_moment - 2.0 * mean_r * d_mean

    loss = 0.0
    grad = np.zeros(2, dtype=np.float64)
    if point_var_weight > 0.0:
        loss += float(point_var_weight * radial_var / max(mean_r * mean_r, 1e-12))
        grad = grad + (point_var_weight * d_radial_var / max(mean_r * mean_r, 1e-12)) - (
            2.0 * point_var_weight * radial_var * d_mean / max(mean_r * mean_r * mean_r, 1e-12)
        )

    n_sector = int(np.max(idx)) + 1 if idx.size > 0 else 0
    if n_sector <= 0:
        return np.inf, np.zeros(2, dtype=np.float64), 0.0
    sector_mass = np.bincount(idx, weights=w, minlength=n_sector).astype(np.float64)
    valid_sector = sector_mass > max(
        1e-9,
        0.25 * float(np.mean(sector_mass[sector_mass > 0.0])) if np.any(sector_mass > 0.0) else 1e-9,
    )
    if np.count_nonzero(valid_sector) < max(10, n_sector // 8):
        return np.inf, np.zeros(2, dtype=np.float64), 0.0
    sector_sum_r = np.bincount(idx, weights=w * rr_safe, minlength=n_sector).astype(np.float64)
    sector_mean = sector_sum_r[valid_sector] / sector_mass[valid_sector]
    gamma = sector_mass[valid_sector]
    gamma = gamma / max(float(np.sum(gamma)), 1e-12)
    d_sector_x = np.bincount(idx, weights=w * dr[:, 0], minlength=n_sector).astype(np.float64)
    d_sector_y = np.bincount(idx, weights=w * dr[:, 1], minlength=n_sector).astype(np.float64)
    d_sector_mean = np.column_stack(
        (
            d_sector_x[valid_sector] / sector_mass[valid_sector],
            d_sector_y[valid_sector] / sector_mass[valid_sector],
        )
    )
    sector_avg = float(np.sum(gamma * sector_mean))
    sector_resid = sector_mean - sector_avg
    sector_var = float(np.sum(gamma * sector_resid * sector_resid))
    d_sector_var = 2.0 * np.sum((gamma * sector_resid)[:, None] * d_sector_mean, axis=0)
    if sector_var_weight > 0.0:
        loss += float(sector_var_weight * sector_var / max(mean_r * mean_r, 1e-12))
        grad = grad + (sector_var_weight * d_sector_var / max(mean_r * mean_r, 1e-12)) - (
            2.0 * sector_var_weight * sector_var * d_mean / max(mean_r * mean_r * mean_r, 1e-12)
        )

    if mean_penalty_weight > 0.0:
        mean_offset = (mean_r - ref_r) / max(ref_r, 1e-12)
        loss += float(mean_penalty_weight * mean_offset * mean_offset)
        grad = grad + (2.0 * mean_penalty_weight * mean_offset / max(ref_r, 1e-12)) * d_mean
    return float(loss), grad.astype(np.float64, copy=False), mean_r


def _scatter_quantile_line_loss(
    center_xy: np.ndarray,
    model: dict[str, np.ndarray | float | int],
) -> float:
    """Loss for the user-selected outer band based on a robust scatter quantile line."""
    pts = np.asarray(model["points"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64).reshape(-1)
    sector_count = int(model.get("sector_count", 0))
    quantile = float(model.get("edge_quantile", 0.75))
    roughness_weight = float(model.get("roughness_weight", 0.0))
    curvature_weight = float(model.get("curvature_weight", 0.0))
    mean_penalty_weight = float(model.get("mean_penalty_weight", 0.0))
    ref_r = float(model.get("reference_radius", model.get("peak_radius", 1.0)))
    edge_point_reg_weight = float(model.get("edge_point_reg_weight", 0.0))
    edge_sector_reg_weight = float(model.get("edge_sector_reg_weight", 0.0))
    coverage_penalty_weight = float(model.get("coverage_penalty_weight", 0.0))

    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0 or weights.size != pts.shape[0] or sector_count <= 0:
        return np.inf

    center = np.asarray(center_xy, dtype=np.float64).reshape(-1)
    if center.size < 2:
        return np.inf

    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    rr = np.hypot(dx, dy)
    rr_safe = np.maximum(rr, 1e-12)
    theta = np.arctan2(dy, dx)
    dr = np.column_stack((-dx / rr_safe, -dy / rr_safe))
    sector_index = _theta_bin_indices(theta, sector_count)

    profile = _scatter_sector_quantile_profile(
        rr_safe,
        theta,
        sector_count,
        weights,
        quantile=quantile,
        sector_weight_base=None,
    )
    if profile is None:
        return np.inf

    valid_mask = np.asarray(profile["valid_mask"], dtype=bool)
    edge_r_all = np.asarray(profile["edge_r"], dtype=np.float64)
    gamma = np.asarray(profile["gamma"], dtype=np.float64)
    edge_r = edge_r_all[valid_mask]
    mean_r = float(np.sum(gamma * edge_r))
    if (not np.isfinite(mean_r)) or mean_r <= 0.0:
        return np.inf

    edge_resid = edge_r - mean_r
    loss = float(np.sum(gamma * edge_resid * edge_resid) / max(mean_r * mean_r, 1e-12))
    valid_theta_fraction = float(np.mean(valid_mask))
    if coverage_penalty_weight > 0.0:
        missing = max(0.0, 1.0 - valid_theta_fraction)
        loss += float(coverage_penalty_weight * missing * missing)

    if mean_penalty_weight > 0.0:
        mean_offset = (mean_r - ref_r) / max(ref_r, 1e-12)
        loss += float(mean_penalty_weight * mean_offset * mean_offset)

    if roughness_weight > 0.0 and edge_r.size >= 3:
        diff = np.diff(edge_r)
        loss += float(roughness_weight * np.mean(diff * diff) / max(mean_r * mean_r, 1e-12))
    if curvature_weight > 0.0 and edge_r.size >= 4:
        curve = edge_r[2:] - (2.0 * edge_r[1:-1]) + edge_r[:-2]
        loss += float(curvature_weight * np.mean(curve * curve) / max(mean_r * mean_r, 1e-12))

    if edge_point_reg_weight > 0.0 or edge_sector_reg_weight > 0.0:
        bulk_loss, _bulk_grad, _bulk_mean = _scatter_shell_bulk_loss_grad(
            rr_safe,
            dr,
            sector_index,
            weights,
            ref_r=ref_r,
            point_var_weight=edge_point_reg_weight,
            sector_var_weight=edge_sector_reg_weight,
            mean_penalty_weight=0.0,
        )
        if np.isfinite(bulk_loss):
            loss += float(bulk_loss)
    return float(loss)


def _scatter_peak_line_loss_grad(
    center_xy: np.ndarray,
    model: dict[str, np.ndarray | float | int],
) -> tuple[float, np.ndarray]:
    """Return outer-ridge straightness loss and analytic gradient for one center."""
    pts = np.asarray(model["points"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64).reshape(-1)
    anchors = np.asarray(model.get("anchor_centers", np.zeros(0, dtype=np.float64)), dtype=np.float64).reshape(-1)
    anchor_weights = np.asarray(model.get("anchor_weights", np.zeros(0, dtype=np.float64)), dtype=np.float64).reshape(-1)
    beta = float(model.get("beta", 0.0))
    kappa = float(model.get("kappa", 1.0))
    ref_r = float(model.get("reference_radius", model.get("peak_radius", 1.0)))
    mean_penalty_weight = float(model.get("mean_penalty_weight", 0.0))
    band_lo = float(model.get("band_lo", 0.0))
    band_hi = float(model.get("band_hi", np.inf))
    band_softness = float(model.get("band_softness", 0.0))
    roughness_weight = float(model.get("roughness_weight", 0.0))
    loss_mode = str(model.get("loss_mode", "ridge_line")).strip().lower()
    sector_index = np.asarray(model.get("sector_index", np.zeros(0, dtype=np.int64)), dtype=np.int64).reshape(-1)
    sector_count = int(model.get("sector_count", anchors.size))
    edge_beta = float(model.get("edge_beta", beta))
    point_var_weight = float(model.get("point_var_weight", 0.0))
    sector_var_weight = float(model.get("sector_var_weight", 0.0))
    edge_point_reg_weight = float(model.get("edge_point_reg_weight", 0.0))
    edge_sector_reg_weight = float(model.get("edge_sector_reg_weight", 0.0))
    curvature_weight = float(model.get("curvature_weight", 0.0))

    if (
        pts.ndim != 2
        or pts.shape[1] < 2
        or pts.shape[0] == 0
        or weights.size != pts.shape[0]
        or anchors.size == 0
        or anchor_weights.size != anchors.size
    ):
        return np.inf, np.zeros(2, dtype=np.float64)

    center = np.asarray(center_xy, dtype=np.float64).reshape(-1)
    if center.size < 2:
        return np.inf, np.zeros(2, dtype=np.float64)

    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    rr = np.hypot(dx, dy)
    rr_safe = np.maximum(rr, 1e-12)
    theta = np.arctan2(dy, dx)

    dr = np.column_stack((-dx / rr_safe, -dy / rr_safe))
    inv_r2 = 1.0 / np.maximum(rr_safe * rr_safe, 1e-12)
    dtheta = np.column_stack((dy * inv_r2, -dx * inv_r2))

    if loss_mode == "roi_edge_quantile":
        base_loss = _scatter_quantile_line_loss(center, model)
        if not np.isfinite(base_loss):
            return np.inf, np.zeros(2, dtype=np.float64)
        shell_width = float(model.get("shell_width", max(ref_r * 0.05, 0.12)))
        fd_step = max(0.04, min(0.18, 0.08 * max(shell_width, 0.5)))
        grad = np.zeros(2, dtype=np.float64)
        for axis in range(2):
            step_vec = np.zeros(2, dtype=np.float64)
            step_vec[axis] = fd_step
            loss_hi = _scatter_quantile_line_loss(center + step_vec, model)
            loss_lo = _scatter_quantile_line_loss(center - step_vec, model)
            if np.isfinite(loss_hi) and np.isfinite(loss_lo):
                grad[axis] = (loss_hi - loss_lo) / (2.0 * fd_step)
            elif np.isfinite(loss_hi):
                grad[axis] = (loss_hi - base_loss) / fd_step
            elif np.isfinite(loss_lo):
                grad[axis] = (base_loss - loss_lo) / fd_step
        return float(base_loss), grad.astype(np.float64, copy=False)

    if loss_mode in {"roi_edge_guided", "roi_edge_softmax"}:
        if sector_index.size != pts.shape[0]:
            return np.inf, np.zeros(2, dtype=np.float64)
        profile = _scatter_sector_edge_profile(
            rr_safe,
            dr,
            sector_index,
            sector_count,
            weights,
            edge_beta=edge_beta,
            reference_radius=ref_r,
            sector_weight_base=np.asarray(model.get("sector_weight", np.zeros(max(sector_count, 0), dtype=np.float64)), dtype=np.float64),
            return_grad=True,
        )
        if profile is None:
            return np.inf, np.zeros(2, dtype=np.float64)

        valid_mask = np.asarray(profile["valid_mask"], dtype=bool)
        edge_r_all = np.asarray(profile["edge_r"], dtype=np.float64)
        d_edge_all = np.asarray(profile["d_edge"], dtype=np.float64)
        gamma = np.asarray(profile["gamma"], dtype=np.float64)
        edge_r = edge_r_all[valid_mask]
        d_edge = d_edge_all[valid_mask]
        mean_r = float(np.sum(gamma * edge_r))
        if (not np.isfinite(mean_r)) or mean_r <= 0.0:
            return np.inf, np.zeros(2, dtype=np.float64)
        d_mean = np.sum(gamma[:, None] * d_edge, axis=0)
        edge_resid = edge_r - mean_r
        edge_var = float(np.sum(gamma * edge_resid * edge_resid))
        d_edge_var = 2.0 * np.sum((gamma * edge_resid)[:, None] * d_edge, axis=0)

        loss = float(edge_var / max(mean_r * mean_r, 1e-12))
        grad = (d_edge_var / max(mean_r * mean_r, 1e-12)) - (
            2.0 * edge_var * d_mean / max(mean_r * mean_r * mean_r, 1e-12)
        )

        if point_var_weight > 0.0:
            w = weights / max(float(np.sum(weights)), 1e-12)
            second_moment = float(np.sum(w * rr_safe * rr_safe))
            radial_var = max(second_moment - mean_r * mean_r, 0.0)
            d_second_moment = 2.0 * np.sum((w * rr_safe)[:, None] * dr, axis=0)
            d_radial_var = d_second_moment - 2.0 * mean_r * d_mean
            loss += float(point_var_weight * radial_var / max(mean_r * mean_r, 1e-12))
            grad = grad + (point_var_weight * d_radial_var / max(mean_r * mean_r, 1e-12)) - (
                2.0 * point_var_weight * radial_var * d_mean / max(mean_r * mean_r * mean_r, 1e-12)
            )

        if sector_var_weight > 0.0:
            loss += float((sector_var_weight - 1.0) * edge_var / max(mean_r * mean_r, 1e-12))
            grad = grad + ((sector_var_weight - 1.0) * d_edge_var / max(mean_r * mean_r, 1e-12)) - (
                2.0 * (sector_var_weight - 1.0) * edge_var * d_mean / max(mean_r * mean_r * mean_r, 1e-12)
            )

        mean_offset = (mean_r - ref_r) / max(ref_r, 1e-12)
        mean_loss = mean_penalty_weight * mean_offset * mean_offset
        grad = grad + (2.0 * mean_penalty_weight * mean_offset / max(ref_r, 1e-12)) * d_mean
        if roughness_weight > 0.0 and edge_r.size >= 3:
            diff = np.diff(edge_r)
            d_diff = d_edge[1:, :] - d_edge[:-1, :]
            loss += roughness_weight * float(np.mean(diff * diff)) / max(ref_r * ref_r, 1e-12)
            grad = grad + (2.0 * roughness_weight / max(ref_r * ref_r, 1e-12)) * np.mean(
                diff[:, None] * d_diff,
                axis=0,
            )
        if curvature_weight > 0.0 and edge_r.size >= 4:
            curve = edge_r[2:] - (2.0 * edge_r[1:-1]) + edge_r[:-2]
            d_curve = d_edge[2:, :] - (2.0 * d_edge[1:-1, :]) + d_edge[:-2, :]
            loss += curvature_weight * float(np.mean(curve * curve)) / max(ref_r * ref_r, 1e-12)
            grad = grad + (2.0 * curvature_weight / max(ref_r * ref_r, 1e-12)) * np.mean(
                curve[:, None] * d_curve,
                axis=0,
            )
        if loss_mode == "roi_edge_guided":
            bulk_loss, bulk_grad, _bulk_mean = _scatter_shell_bulk_loss_grad(
                rr_safe,
                dr,
                sector_index,
                weights,
                ref_r=ref_r,
                point_var_weight=edge_point_reg_weight,
                sector_var_weight=edge_sector_reg_weight,
                mean_penalty_weight=0.0,
            )
            if np.isfinite(bulk_loss):
                loss += bulk_loss
                grad = grad + bulk_grad
        return float(loss + mean_loss), grad.astype(np.float64, copy=False)

    if loss_mode == "roi_shell_variance":
        if sector_index.size != pts.shape[0]:
            return np.inf, np.zeros(2, dtype=np.float64)
        loss, grad, _mean_r = _scatter_shell_bulk_loss_grad(
            rr_safe,
            dr,
            sector_index,
            weights,
            ref_r=ref_r,
            point_var_weight=point_var_weight,
            sector_var_weight=sector_var_weight,
            mean_penalty_weight=mean_penalty_weight,
        )
        return float(loss), grad.astype(np.float64, copy=False)

    delta = theta[:, None] - anchors[None, :]
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    angular_kernel = np.exp(kappa * (cos_delta - 1.0))
    radial_kernel = np.exp(np.clip(beta * (rr_safe - ref_r), -60.0, 60.0))[:, None]

    band_grad_term = 0.0
    band_gate = np.ones_like(rr_safe, dtype=np.float64)
    if np.isfinite(band_lo) and np.isfinite(band_hi) and band_hi > band_lo and band_softness > 0.0:
        sigmoid_lo = 1.0 / (1.0 + np.exp(-np.clip((rr_safe - band_lo) / band_softness, -60.0, 60.0)))
        sigmoid_hi = 1.0 / (1.0 + np.exp(-np.clip((band_hi - rr_safe) / band_softness, -60.0, 60.0)))
        band_gate = np.clip(sigmoid_lo * sigmoid_hi, 1e-12, 1.0)
        band_grad_term = ((sigmoid_hi - sigmoid_lo) / band_softness)[:, None, None] * dr[:, None, :]
    z = weights[:, None] * angular_kernel * radial_kernel * band_gate[:, None]

    den = np.sum(z, axis=0)
    if np.any(~np.isfinite(den)):
        return np.inf, np.zeros(2, dtype=np.float64)
    support_floor = max(1e-18, 0.03 * float(np.max(den)))
    valid_anchor = den > support_floor
    if np.count_nonzero(valid_anchor) < max(10, anchors.size // 7):
        return np.inf, np.zeros(2, dtype=np.float64)
    den = den[valid_anchor]
    anchor_weights = anchor_weights[valid_anchor]
    weight_sum = float(np.sum(anchor_weights))
    if (not np.isfinite(weight_sum)) or weight_sum <= 0.0:
        return np.inf, np.zeros(2, dtype=np.float64)
    anchor_weights = anchor_weights / weight_sum
    num = np.sum(z * rr_safe[:, None], axis=0)
    ridge_r = num[valid_anchor] / den
    ridge_mean = float(np.sum(anchor_weights * ridge_r))
    if (not np.isfinite(ridge_mean)) or ridge_mean <= 0.0:
        return np.inf, np.zeros(2, dtype=np.float64)

    angular_grad_term = (-kappa * sin_delta)[:, :, None] * dtheta[:, None, :]
    radial_grad_term = beta * dr[:, None, :]
    h = angular_grad_term + radial_grad_term + band_grad_term
    d_den_full = np.sum(z[:, :, None] * h, axis=0)
    d_num_full = np.sum(z[:, :, None] * (rr_safe[:, None, None] * h + dr[:, None, :]), axis=0)
    d_den = d_den_full[valid_anchor]
    d_num = d_num_full[valid_anchor]
    d_ridge = (d_num - ridge_r[:, None] * d_den) / den[:, None]

    resid = ridge_r - ridge_mean
    ridge_var = float(np.sum(anchor_weights * resid * resid))
    d_mean = np.sum(anchor_weights[:, None] * d_ridge, axis=0)
    d_var = 2.0 * np.sum((anchor_weights * resid)[:, None] * d_ridge, axis=0)

    base_loss = ridge_var / max(ridge_mean * ridge_mean, 1e-12)
    grad = (d_var / max(ridge_mean * ridge_mean, 1e-12)) - (
        2.0 * ridge_var * d_mean / max(ridge_mean * ridge_mean * ridge_mean, 1e-12)
    )
    mean_offset = (ridge_mean - ref_r) / max(ref_r, 1e-12)
    mean_loss = mean_penalty_weight * mean_offset * mean_offset
    grad = grad + (2.0 * mean_penalty_weight * mean_offset / max(ref_r, 1e-12)) * d_mean

    roughness_loss = 0.0
    if roughness_weight > 0.0 and ridge_r.size >= 3:
        diff = np.diff(ridge_r)
        d_diff = d_ridge[1:, :] - d_ridge[:-1, :]
        roughness_loss = roughness_weight * float(np.mean(diff * diff)) / max(ref_r * ref_r, 1e-12)
        grad = grad + (2.0 * roughness_weight / max(ref_r * ref_r, 1e-12)) * np.mean(
            diff[:, None] * d_diff,
            axis=0,
        )
    if curvature_weight > 0.0 and ridge_r.size >= 4:
        curve = ridge_r[2:] - (2.0 * ridge_r[1:-1]) + ridge_r[:-2]
        d_curve = d_ridge[2:, :] - (2.0 * d_ridge[1:-1, :]) + d_ridge[:-2, :]
        roughness_loss += curvature_weight * float(np.mean(curve * curve)) / max(ref_r * ref_r, 1e-12)
        grad = grad + (2.0 * curvature_weight / max(ref_r * ref_r, 1e-12)) * np.mean(
            curve[:, None] * d_curve,
            axis=0,
        )

    return float(base_loss + mean_loss + roughness_loss), grad.astype(np.float64, copy=False)


def _scatter_peak_line_loss(center_xy: np.ndarray, model: dict[str, np.ndarray | float | int]) -> float:
    """Measure how horizontal the shell outer edge stays in `(theta, r)`."""
    loss, _grad = _scatter_peak_line_loss_grad(center_xy, model)
    return float(loss)


def _optimize_scatter_peak_line_model(
    start_center: np.ndarray,
    edge_seed: np.ndarray,
    model: dict[str, np.ndarray | float | int],
    *,
    span_hint: float,
) -> tuple[np.ndarray, float]:
    """Optimize one frozen scatter-shell model with analytic gradients and monotonic updates."""
    best_center = np.asarray(start_center, dtype=np.float64).reshape(2).copy()
    best_loss, best_grad = _scatter_peak_line_loss_grad(best_center, model)

    edge_center = np.asarray(edge_seed, dtype=np.float64).reshape(2).copy()
    edge_loss, edge_grad = _scatter_peak_line_loss_grad(edge_center, model)
    if edge_loss + 1e-12 < best_loss or not np.isfinite(best_loss):
        best_center = edge_center
        best_loss = edge_loss
        best_grad = edge_grad

    peak_radius = float(model.get("peak_radius", 1.0))
    search_radius = max(0.20, min(0.08 * max(peak_radius, 1.0), 0.04 * span_hint, 2.0))
    edge_gap = float(np.linalg.norm(edge_center - best_center))
    if np.isfinite(edge_gap) and edge_gap > 0.0:
        search_radius = max(search_radius, min(edge_gap, 3.0))

    base_center = best_center.copy()
    offsets = np.linspace(-float(search_radius), float(search_radius), 5, dtype=np.float64)
    for dx in offsets:
        for dy in offsets:
            candidate = np.array([base_center[0] + dx, base_center[1] + dy], dtype=np.float64)
            cand_loss, cand_grad = _scatter_peak_line_loss_grad(candidate, model)
            if cand_loss + 1e-12 < best_loss:
                best_center = candidate
                best_loss = float(cand_loss)
                best_grad = cand_grad

    move_tol = max(5e-4, 1e-4 * max(peak_radius, 1.0))
    line_step = max(search_radius * 0.35, 0.06)
    for _ in range(14):
        if not np.isfinite(best_loss):
            break
        grad_norm = float(np.linalg.norm(best_grad))
        if grad_norm <= 1e-6:
            break
        direction = -best_grad / grad_norm
        accepted = False
        step_try = float(line_step)
        while step_try >= move_tol:
            candidate = best_center + step_try * direction
            cand_loss, cand_grad = _scatter_peak_line_loss_grad(candidate, model)
            if cand_loss + 1e-12 < best_loss:
                improvement = float(best_loss - cand_loss)
                best_center = candidate
                best_loss = float(cand_loss)
                best_grad = cand_grad
                accepted = True
                line_step = max(step_try * 1.35, move_tol)
                if step_try <= move_tol or improvement <= 1e-7 * max(abs(best_loss), 1.0):
                    return best_center, best_loss
                break
            step_try *= 0.5
        if not accepted:
            line_step *= 0.5
            if line_step < move_tol:
                break

    if str(model.get("loss_mode", "")).strip().lower() == "roi_edge_quantile":
        pattern_step = max(line_step, move_tol * 2.0)
        directions = (
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
            np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0),
            np.array([-1.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
            np.array([-1.0, -1.0], dtype=np.float64) / np.sqrt(2.0),
        )
        for _ in range(10):
            improved = False
            for direction in directions:
                candidate = best_center + pattern_step * direction
                cand_loss, cand_grad = _scatter_peak_line_loss_grad(candidate, model)
                if cand_loss + 1e-12 < best_loss:
                    best_center = candidate
                    best_loss = float(cand_loss)
                    best_grad = cand_grad
                    improved = True
                    break
            if improved:
                continue
            pattern_step *= 0.5
            if pattern_step < move_tol:
                break

    return best_center, best_loss


def _scatter_peak_line_center(
    points_xy: np.ndarray,
    fallback_xy: tuple[float, float] | np.ndarray,
    *,
    peak_mode: str,
    theta_bins: int,
    sample_limit: int,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> np.ndarray:
    """Optimize a fixed shell from raw scatter points with monotonic loss updates.

    Keep this path scatter-based rather than histogram-only. The user explicitly
    prefers center refinement on raw points because it is typically more accurate
    and more stable than optimizing only on a binned polar matrix.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    fallback = np.asarray(fallback_xy, dtype=np.float64).reshape(-1)
    if fallback.size < 2:
        fallback = np.array([0.0, 0.0], dtype=np.float64)
    else:
        fallback = fallback[:2].astype(np.float64, copy=False)

    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0:
        return fallback.copy()

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite, :2]
    if pts.shape[0] == 0:
        return fallback.copy()

    if sample_limit > 0 and pts.shape[0] > int(sample_limit):
        step = max(1, int(np.ceil(float(pts.shape[0]) / float(sample_limit))))
        pts = pts[::step]

    if pts.shape[0] < 12:
        return np.mean(pts, axis=0).astype(np.float64)

    fallback_center = fallback.astype(np.float64, copy=True)
    span_hint = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    manual_band = (
        target_radius is not None
        and target_window is not None
        and np.isfinite(float(target_radius))
        and np.isfinite(float(target_window))
        and float(target_radius) > 0.0
        and float(target_window) > 0.0
    )
    if (
        manual_band
        and str(peak_mode).strip().lower() == "outermost"
        and _should_use_iterative_outer_roi_edge_fit(target_radius, target_window)
    ):
        return _iterative_outer_roi_edge_circle_center(
            pts,
            fallback_center,
            theta_bins=theta_bins,
            sample_limit=sample_limit,
            target_radius=float(target_radius),
            target_window=float(target_window),
        )

    coarse_center = quadrant_symmetry_center(pts, fallback_center)
    current_center = fallback_center.copy()
    edge_seed = edge_circle_center(pts, coarse_center)
    initial_reference_center = fallback_center if manual_band else coarse_center
    model = _build_scatter_peak_line_model(
        pts,
        initial_reference_center,
        peak_mode=peak_mode,
        theta_bins=theta_bins,
        target_radius=target_radius,
        target_window=target_window,
    )
    if model is None:
        model = _build_scatter_peak_line_model(
            pts,
            edge_seed,
            peak_mode=peak_mode,
            theta_bins=theta_bins,
            target_radius=target_radius,
            target_window=target_window,
        )
        if model is None:
            return edge_seed.astype(np.float64, copy=False)

    candidate_centers = [fallback_center, coarse_center, edge_seed]
    candidate_scores: list[tuple[float, np.ndarray]] = []
    fallback_loss = np.inf
    for candidate in candidate_centers:
        loss = float(_scatter_peak_line_loss(candidate, model))
        if np.isfinite(loss):
            candidate_scores.append((loss, np.asarray(candidate, dtype=np.float64).reshape(2).copy()))
        if np.allclose(candidate, fallback_center, atol=0.0, rtol=0.0):
            fallback_loss = float(loss)
    peak_radius = float(model.get("peak_radius", 1.0))
    min_rel_improvement = 2.0e-3 if str(peak_mode).strip().lower() == "outermost" else 1.0e-3
    if float(np.linalg.norm(fallback_center - coarse_center)) <= max(0.5, 0.02 * max(peak_radius, 1.0)):
        min_rel_improvement = max(min_rel_improvement, 1.0e-2)
    if candidate_scores:
        candidate_scores.sort(key=lambda item: item[0])
        best_candidate_loss, best_candidate_center = candidate_scores[0]
        if np.isfinite(fallback_loss) and fallback_loss <= best_candidate_loss * (1.0 + min_rel_improvement):
            current_center = fallback_center.copy()
        else:
            current_center = best_candidate_center
    else:
        current_center = coarse_center.astype(np.float64, copy=True)

    loss_mode = str(model.get("loss_mode", "")).strip().lower()
    reanchor_iterations = 6 if loss_mode in {"roi_edge_guided", "roi_edge_quantile"} else 2
    for _ in range(reanchor_iterations):
        current_loss = float(_scatter_peak_line_loss(current_center, model))
        best_center, best_loss = _optimize_scatter_peak_line_model(
            current_center,
            edge_seed,
            model,
            span_hint=span_hint,
        )
        if not np.isfinite(current_loss):
            current_center = best_center
            continue
        if best_loss + 1e-10 >= current_loss * (1.0 - min_rel_improvement):
            break

        peak_radius = float(model.get("peak_radius", 1.0))
        move_guard = max(0.30, 0.03 * max(peak_radius, 1.0))
        if float(np.linalg.norm(best_center - current_center)) > move_guard and best_loss >= 0.98 * max(current_loss, 1e-12):
            break

        shift = float(np.linalg.norm(best_center - current_center))
        if shift <= max(1e-3, 2e-4 * max(peak_radius, 1.0)):
            current_center = best_center
            break
        edge_seed = edge_circle_center(pts, best_center)
        next_model = _build_scatter_peak_line_model(
            pts,
            best_center,
            peak_mode=peak_mode,
            theta_bins=theta_bins,
            target_radius=target_radius,
            target_window=target_window,
        )
        if next_model is None:
            next_model = _build_scatter_peak_line_model(
                pts,
                edge_seed,
                peak_mode=peak_mode,
                theta_bins=theta_bins,
                target_radius=target_radius,
                target_window=target_window,
            )
        if next_model is None:
            break
        if loss_mode == "roi_edge_quantile":
            reanchored_current_loss = float(_scatter_peak_line_loss(current_center, next_model))
            reanchored_best_center, reanchored_best_loss = _optimize_scatter_peak_line_model(
                best_center,
                edge_seed,
                next_model,
                span_hint=span_hint,
            )
            if np.isfinite(reanchored_current_loss) and np.isfinite(reanchored_best_loss):
                if reanchored_best_loss + 1e-10 >= reanchored_current_loss * (1.0 - min_rel_improvement):
                    break
                current_center = reanchored_best_center
                edge_seed = edge_circle_center(pts, current_center)
                model = _build_scatter_peak_line_model(
                    pts,
                    current_center,
                    peak_mode=peak_mode,
                    theta_bins=theta_bins,
                    target_radius=target_radius,
                    target_window=target_window,
                )
                if model is None:
                    model = next_model
                continue
        current_center = best_center
        model = next_model

    return current_center


def polar_peak_center(
    points_xy: np.ndarray,
    fallback_xy: tuple[float, float] | np.ndarray,
    *,
    theta_bins: int = 180,
    radial_bins: int = 160,
    sample_limit: int = 32_000,
    target_radius: float | None = None,
    target_window: float | None = None,
) -> np.ndarray:
    """Estimate center by straightening the dominant ring from raw scatter points."""
    _ = radial_bins
    return _scatter_peak_line_center(
        points_xy,
        fallback_xy,
        peak_mode="dominant",
        theta_bins=theta_bins,
        sample_limit=sample_limit,
        target_radius=target_radius,
        target_window=target_window,
    )


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
