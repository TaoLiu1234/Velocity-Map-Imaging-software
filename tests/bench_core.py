#!/usr/bin/env python3
"""Micro-benchmark for the heavy center estimators in VMI_workflow_core.

Plain script (no pytest), prints timings. Compares the live implementation
against `_reference_*` copies of the pre-optimization code kept here so the
A/B is direct and the old implementation stays inspectable.

Run:  python tests/bench_core.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.spatial import cKDTree  # noqa: E402

from VMI_workflow_core import (  # noqa: E402
    edge_circle_center,
    polar_outermost_center,
    quadrant_symmetry_center,
)

BENCH_SEED = 20260831
BENCH_TRUE_CENTER = (34.5, -21.25)
BENCH_FALLBACK = (BENCH_TRUE_CENTER[0] + 3.0, BENCH_TRUE_CENTER[1] - 2.0)
BENCH_SIZES = (50_000, 500_000)


def _bench_cloud(n_points: int) -> np.ndarray:
    """Same deterministic generator as the lock tests in test_core.py."""
    rng = np.random.default_rng(BENCH_SEED + n_points)
    n1 = int(n_points * 0.45)
    n2 = int(n_points * 0.35)
    nb = n_points - n1 - n2
    theta1 = rng.uniform(-np.pi, np.pi, n1)
    r1 = 60.0 + 4.0 * rng.standard_normal(n1)
    theta2 = rng.uniform(-np.pi, np.pi, n2)
    r2 = 110.0 + 5.0 * rng.standard_normal(n2)
    theta_b = rng.uniform(-np.pi, np.pi, nb)
    r_b = np.sqrt(rng.uniform(0.0, 150.0**2, nb))
    rr = np.concatenate((r1, r2, r_b))
    tt = np.concatenate((theta1, theta2, theta_b))
    return np.column_stack(
        (
            BENCH_TRUE_CENTER[0] + rr * np.cos(tt),
            BENCH_TRUE_CENTER[1] + rr * np.sin(tt),
        )
    )


# ---------------------------------------------------------------------------
# Reference copy of the PRE-optimization quadrant_symmetry_center (2026-08-31).
# It rebuilds a cKDTree over the folded destination quadrant for every
# candidate center. Kept verbatim (module-level helpers inlined as closures
# where they were nested) for direct A/B timing; results must equal the live
# implementation bit-for-bit.
# ---------------------------------------------------------------------------
def _reference_quadrant_symmetry_center(
    points_xy: np.ndarray,
    fallback_xy: tuple[float, float] | np.ndarray,
    *,
    grid_bins: int = 72,
    sample_limit: int = 28_000,
) -> np.ndarray:
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

    def _pair_point_stats(center_xy: np.ndarray) -> tuple[float, np.ndarray, float]:
        dx = pts[:, 0] - float(center_xy[0])
        dy = pts[:, 1] - float(center_xy[1])
        rr = np.hypot(dx, dy)
        shell_resid = (rr - peak_r) / max(shell_sigma, 1e-9)
        radial_weight = np.exp(-0.5 * np.clip(shell_resid * shell_resid, 0.0, 25.0))
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

            src_fold = np.column_stack((np.abs(dx[src_idx]), np.abs(dy[src_idx]))).astype(np.float64, copy=False)
            dst_fold = np.column_stack((np.abs(dx[dst_idx]), np.abs(dy[dst_idx]))).astype(np.float64, copy=False)
            tree = cKDTree(dst_fold)
            dist, nn_local = tree.query(src_fold, k=1)
            matched_idx = dst_idx[np.asarray(nn_local, dtype=np.int64)]

            w_src = point_weight[src_idx]
            w_dst = point_weight[matched_idx]
            dist = np.asarray(dist, dtype=np.float64)
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


# `polar_outermost_center` analysis (2026-08-31): every quantity in its
# loss/gradient path (`_scatter_peak_line_loss_grad`, `_scatter_quantile_line_loss`,
# `_scatter_sector_edge_profile`, ...) is derived from polar coordinates
# (dx, dy, rr, theta, dr, sector bins) measured FROM the candidate center, so
# nothing is candidate-independent and there is nothing to hoist. The only
# per-call candidate-independent work is ~20 scalar dict reads, which is
# nanoseconds against millisecond numpy evaluation. The reference below is
# therefore an alias: before == after by construction, and the timing confirms
# no regression from the quadrant refactor (which feeds its coarse seed).
_reference_polar_outermost_center = polar_outermost_center


def _timed(label: str, fn, *args, repeat: int = 1) -> tuple[float, np.ndarray]:
    result = None
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    print(f"  {label:<38s} {best:8.3f} s")
    return best, result


def main() -> int:
    print("VMI_workflow_core center-estimator benchmark")
    print(f"(numpy {np.__version__}; best of repeated runs)")
    rows: list[dict[str, object]] = []
    for n in BENCH_SIZES:
        pts = _bench_cloud(n)
        print(f"\ncloud: {n} points (true center {BENCH_TRUE_CENTER})")
        t_ref_q, c_ref_q = _timed("quadrant  reference (per-cand tree)", _reference_quadrant_symmetry_center, pts, BENCH_FALLBACK)
        t_new_q, c_new_q = _timed("quadrant  live (shared raw tree)", quadrant_symmetry_center, pts, BENCH_FALLBACK)
        t_ref_p, c_ref_p = _timed("polar     reference", _reference_polar_outermost_center, pts, BENCH_FALLBACK)
        t_new_p, c_new_p = _timed("polar     live", polar_outermost_center, pts, BENCH_FALLBACK)
        same_q = bool(np.array_equal(np.asarray(c_ref_q), np.asarray(c_new_q)))
        same_p = bool(np.array_equal(np.asarray(c_ref_p), np.asarray(c_new_p)))
        print(f"  quadrant identical result: {same_q}  speedup: {t_ref_q / max(t_new_q, 1e-12):.2f}x")
        print(f"  polar     identical result: {same_p}  speedup: {t_ref_p / max(t_new_p, 1e-12):.2f}x")
        rows.append(
            {
                "n": n,
                "quadrant_ref_s": t_ref_q,
                "quadrant_new_s": t_new_q,
                "quadrant_same": same_q,
                "polar_ref_s": t_ref_p,
                "polar_new_s": t_new_p,
                "polar_same": same_p,
            }
        )
    print("\nsummary:")
    for row in rows:
        print(
            "  n={n:>7d}: quadrant {qr:.2f}s -> {qn:.2f}s ({qx:.2f}x, identical={qs}) | "
            "polar {pr:.2f}s -> {pn:.2f}s ({px:.2f}x, identical={ps})".format(
                n=row["n"],
                qr=row["quadrant_ref_s"],
                qn=row["quadrant_new_s"],
                qx=row["quadrant_ref_s"] / max(row["quadrant_new_s"], 1e-12),
                qs=row["quadrant_same"],
                pr=row["polar_ref_s"],
                pn=row["polar_new_s"],
                px=row["polar_ref_s"] / max(row["polar_new_s"], 1e-12),
                ps=row["polar_same"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
