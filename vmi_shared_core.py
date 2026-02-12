from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from Abel_forward_simulation import Config, run_simulation


GRID_SIZE = 512
PIXEL_SIZE_MM = 0.1
DBSCAN_EPS = 2.0
DBSCAN_MIN_SAMPLES = 10


@dataclass
class CaseParams:
    r_count: int
    r_size_mm: float
    beta: float
    br_list: List[float]
    n_events: int


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def parse_br_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("br-list is empty.")
    return values


def mm_to_px(points_mm: np.ndarray) -> np.ndarray:
    half_mm = 0.5 * GRID_SIZE * PIXEL_SIZE_MM
    points_px = np.empty_like(points_mm, dtype=np.float64)
    points_px[:, 0] = (points_mm[:, 0] + half_mm) / PIXEL_SIZE_MM - 0.5
    points_px[:, 1] = (points_mm[:, 1] + half_mm) / PIXEL_SIZE_MM - 0.5
    return points_px


def px_to_mm(center_px: np.ndarray) -> np.ndarray:
    half_mm = 0.5 * GRID_SIZE * PIXEL_SIZE_MM
    center_mm = np.empty(2, dtype=np.float64)
    center_mm[0] = (center_px[0] + 0.5) * PIXEL_SIZE_MM - half_mm
    center_mm[1] = (center_px[1] + 0.5) * PIXEL_SIZE_MM - half_mm
    return center_mm


def geometric_median(points: np.ndarray, tol: float = 1e-5, max_iter: int = 500) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Empty points.")
    center = np.mean(points, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - center, axis=1)
        z = d < 1e-12
        if np.any(z):
            return points[np.argmax(z)].copy()
        w = 1.0 / np.clip(d, 1e-12, None)
        new_center = np.sum(points * w[:, None], axis=0) / np.sum(w)
        if np.linalg.norm(new_center - center) < tol:
            return new_center
        center = new_center
    return center


def make_energy_levels(r_count: int) -> List[float]:
    if r_count <= 1:
        return [0.5]
    return np.linspace(0.35, 0.95, r_count).tolist()


def fit_and_normalize_branching_ratios(r_count: int, br_values: List[float]) -> List[float]:
    if r_count <= 0:
        raise ValueError("r_count must be > 0")
    values = np.array(br_values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("br-list cannot be empty.")
    if np.any(values < 0):
        raise ValueError("br-list cannot contain negative values.")

    if values.size == 1 and r_count > 1:
        values = np.repeat(values[0], r_count)
    elif values.size < r_count:
        pad = np.repeat(values[-1], r_count - values.size)
        values = np.concatenate([values, pad])
    elif values.size > r_count:
        values = values[:r_count]

    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("Sum of br-list values must be > 0.")
    values = values / total
    values[-1] = 1.0 - float(np.sum(values[:-1]))
    return values.tolist()


def make_case_profiles(p: CaseParams) -> Tuple[List[float], List[float], List[float]]:
    energies = make_energy_levels(p.r_count)
    betas = [float(np.clip(p.beta, -1.0, 2.0))] * p.r_count
    branching = fit_and_normalize_branching_ratios(p.r_count, p.br_list)
    return energies, betas, branching


def build_config(p: CaseParams) -> Config:
    energies, betas, branching = make_case_profiles(p)
    vmi_k = Config.calculate_vmi_k(E_max_eV=max(energies), r_max_mm=p.r_size_mm)
    return Config(
        E_centers=energies,
        Betas=betas,
        branching_ratios=branching,
        N_events=p.n_events,
        img_res=GRID_SIZE,
        pixel_size=PIXEL_SIZE_MM,
        vmi_k=vmi_k,
        psf_fwhm=0.25,
        dld_resolution=0.015,
        mcp_dark_rate=0.015,
        residual_gas_rate=0.0,
        residual_gas_sigma=1.0,
    )


def generate_scatter_and_truth(
    p: CaseParams,
    bias_x_px: float,
    bias_y_px: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    config = build_config(p)

    np.random.seed(seed)
    ideal_mm, _ = run_simulation(config, add_noise=False, output_mode="xy_ideal")
    true_center = geometric_median(mm_to_px(ideal_mm))

    np.random.seed(seed)
    observed_mm, _ = run_simulation(config, add_noise=True, output_mode="xy_dld")
    observed_px = mm_to_px(observed_mm)
    observed_px[:, 0] += bias_x_px
    observed_px[:, 1] += bias_y_px
    true_center += np.array([bias_x_px, bias_y_px], dtype=np.float64)
    return observed_px, true_center


def dbscan_clusters(points_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(points_px)
    non_noise = labels >= 0
    if not np.any(non_noise):
        raise RuntimeError("DBSCAN found only noise.")
    labels_valid = labels[non_noise]
    unique, counts = np.unique(labels_valid, return_counts=True)
    main_label = unique[np.argmax(counts)]
    main_cluster = points_px[labels == main_label]
    non_noise_points = points_px[non_noise]
    return main_cluster, non_noise_points, labels


def unbinned_loss(center: np.ndarray, points_px: np.ndarray, tree: cKDTree) -> float:
    cx, _ = center
    inv = 2.0 * center - points_px
    mir = np.column_stack((2.0 * cx - points_px[:, 0], points_px[:, 1]))
    d_inv = tree.query(inv, k=1)[0]
    d_mir = tree.query(mir, k=1)[0]
    radii = np.linalg.norm(points_px - center, axis=1)
    return float(np.mean(d_inv**2) + np.mean(d_mir**2) + 0.05 * np.var(radii))


def refine_unbinned(points_px: np.ndarray, coarse: np.ndarray) -> Tuple[np.ndarray, bool, str]:
    tree = cKDTree(points_px)
    bounds = [(-0.5, GRID_SIZE - 0.5), (-0.5, GRID_SIZE - 0.5)]
    res = minimize(
        fun=lambda c: unbinned_loss(c, points_px, tree),
        x0=np.asarray(coarse, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-4},
    )
    return np.asarray(res.x, dtype=np.float64), bool(res.success), str(res.message)


def points_to_image(points_px: np.ndarray) -> np.ndarray:
    edges = np.arange(-0.5, GRID_SIZE + 0.5, 1.0)
    image, _, _ = np.histogram2d(points_px[:, 1], points_px[:, 0], bins=(edges, edges))
    return image.astype(np.float64, copy=False)


def binned_loss(center: np.ndarray, image: np.ndarray) -> float:
    cx, cy = center
    center_ref = (image.shape[0] - 1) / 2.0
    shifted = ndi_shift(
        image,
        shift=(center_ref - cy, center_ref - cx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    inv = shifted[::-1, ::-1]
    mir = shifted[:, ::-1]
    return float(np.sum((shifted - inv) ** 2) + np.sum((shifted - mir) ** 2))


def refine_binned(points_px: np.ndarray, coarse: np.ndarray) -> Tuple[np.ndarray, bool, str]:
    image = points_to_image(points_px)
    bounds = [(-0.5, GRID_SIZE - 0.5), (-0.5, GRID_SIZE - 0.5)]
    res = minimize(
        fun=lambda c: binned_loss(c, image),
        x0=np.asarray(coarse, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    return np.asarray(res.x, dtype=np.float64), bool(res.success), str(res.message)

