#!/usr/bin/env python3
from __future__ import annotations

"""Reconstruction helpers for VMI_workflow.

This module isolates numerical reconstruction code from GUI code.
The GUI only prepares centered/binning data and passes it here.

===============================================================================
rBasex reconstruction (`run_rbasex_reconstruction`)
===============================================================================
Core steps:
1. Read a pre-centered 2D projection image (`hist_image`).
2. Run `abel.rbasex.rbasex_transform(..., direction="inverse")`.
3. Read radial outputs: r, I(r), beta(r).
4. Convert radius from pixel units to data units (`r * bin_size`).
5. Find dominant peaks from I(r), then read beta at those radii.
6. Return reconstruction image + recovered (r, beta) summary.

`settings` keys for rBasex:
- `order`:
  Legendre order used by rBasex (integer, usually 2).
- `odd`:
  Whether odd angular terms are included.
- `reg`:
  Regularization strength (None or non-negative float).
- `rmax`:
  Radial limit used by rBasex (`"MIN"`, `"MAX"`, or positive int).
- `peak_smooth_sigma`:
  Gaussian smoothing sigma for peak detection on I(r).
- `peak_height`:
  Minimum normalized peak height threshold in I(r).
- `peak_prominence`:
  Minimum normalized peak prominence threshold in I(r).
- `peak_min_dist_frac`:
  Minimum peak distance as fraction of profile length.
- `max_peaks`:
  Max number of peaks returned.
- `display_percentile`:
  Suggested percentile for image display scaling in GUI.

Dependencies:
- `pyabel` (for rBasex).
"""

from pathlib import Path
from typing import Callable

import abel
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _rbasex_basis_dir() -> str | None:
    """Return a persistent per-user rBasex basis cache directory (or None).

    pyabel keeps the basis set only in memory while ``basis_dir=None``, so
    every fresh process pays the slow basis generation again. Pointing
    ``basis_dir`` at this stable per-user directory lets pyabel save/load the
    exact basis arrays between runs (``np.save``/``np.load`` are bit-exact,
    so reconstruction results are unchanged). Falls back to ``None`` (the
    previous behaviour) when the directory cannot be created.
    """
    basis_dir = Path.home() / ".cache" / "vmi_workflow" / "abel_basis"
    try:
        basis_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return str(basis_dir)


def extract_peak_r_beta(
    r_axis: np.ndarray,
    intensity: np.ndarray,
    beta_profile: np.ndarray,
    peak_smooth_sigma: float,
    peak_height: float,
    peak_prominence: float,
    peak_min_dist_frac: float,
    max_peaks: int,
) -> list[dict]:
    """Extract dominant (r, beta) peaks from radial profiles.

    Parameters:
    - `r_axis`:
      Radius axis in physical units.
    - `intensity`:
      Radial intensity profile I(r).
    - `beta_profile`:
      Angular anisotropy profile beta(r), same length as `r_axis`.
    - `peak_smooth_sigma`:
      Optional Gaussian smoothing sigma applied to I(r) before peak search.
    - `peak_height`:
      Minimum normalized peak height.
    - `peak_prominence`:
      Minimum normalized peak prominence.
    - `peak_min_dist_frac`:
      Minimum peak distance = len(profile) * this fraction.
    - `max_peaks`:
      Maximum number of peaks returned.

    Returns:
    - A list of dict items with keys: `r`, `beta`, `i`.
    """
    if r_axis.size == 0 or intensity.size == 0 or beta_profile.size == 0:
        return []
    if not np.any(np.isfinite(intensity)):
        return []

    intensity = np.nan_to_num(intensity.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    beta_profile = np.nan_to_num(beta_profile.astype(np.float64), nan=0.0)
    if np.max(intensity) <= 0:
        return []

    if peak_smooth_sigma > 0 and intensity.size >= 7:
        smooth = gaussian_filter1d(intensity, sigma=peak_smooth_sigma, mode="nearest")
    else:
        smooth = intensity

    norm = smooth / (np.max(smooth) + 1e-12)
    min_dist = max(1, int(len(norm) * max(0.0, peak_min_dist_frac)))
    peaks, _ = find_peaks(norm, height=max(0.0, peak_height), prominence=max(0.0, peak_prominence), distance=min_dist)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(norm))], dtype=np.int64)

    order = np.argsort(norm[peaks])[::-1]
    peaks = peaks[order[: max(1, max_peaks)]]
    peaks = np.sort(peaks)

    valley_edges: list[tuple[int, int]] = []
    for pos, idx in enumerate(peaks):
        left_edge = 0
        right_edge = len(norm) - 1
        if pos > 0:
            prev_idx = int(peaks[pos - 1])
            left_edge = prev_idx + int(np.argmin(smooth[prev_idx : idx + 1]))
        else:
            left_edge = int(np.argmin(smooth[: idx + 1]))
        if pos < (len(peaks) - 1):
            next_idx = int(peaks[pos + 1])
            right_edge = idx + int(np.argmin(smooth[idx : next_idx + 1]))
        else:
            right_edge = idx + int(np.argmin(smooth[idx:]))
        left_edge = max(0, min(int(left_edge), int(idx)))
        right_edge = min(len(norm) - 1, max(int(right_edge), int(idx)))
        valley_edges.append((left_edge, right_edge))

    out = []
    for idx, (left_edge, right_edge) in zip(peaks, valley_edges):
        if not (0 <= idx < len(r_axis) and 0 <= idx < len(beta_profile)):
            continue
        area = float(intensity[idx])
        if 0 <= left_edge <= right_edge < len(r_axis):
            r_seg = np.asarray(r_axis[left_edge : right_edge + 1], dtype=np.float64)
            i_seg = np.asarray(intensity[left_edge : right_edge + 1], dtype=np.float64)
            if r_seg.size >= 2 and i_seg.size == r_seg.size and np.isfinite(r_seg).all() and np.isfinite(i_seg).any():
                area = float(np.trapezoid(np.clip(i_seg, 0.0, None), r_seg))
        out.append(
            {
                "r": float(r_axis[idx]),
                "beta": float(np.clip(beta_profile[idx], -2.0, 2.0)),
                "i": float(intensity[idx]),
                "area": max(area, 0.0),
            }
        )
    return out


def format_peak_text(peaks: list[dict]) -> str:
    """Format recovered peaks for display in plot panels."""
    if not peaks:
        return "Recovered peaks:\n(no peak)"
    lines = ["Recovered peaks:"]
    for i, p in enumerate(peaks, start=1):
        area = p.get("area", p.get("i", 0.0))
        lines.append(
            f"{i}. r={float(p['r']):.3g}, beta={float(p['beta']):.3g}, intensity={float(area):.4g}"
        )
    return "\n".join(lines)


def run_rbasex_reconstruction(
    hist_image: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    bin_size: float,
    settings: dict,
) -> dict:
    """Run pyabel rBasex inverse transform and summarize peaks.

    Parameters:
    - `hist_image`:
      2D centered projection image to invert (shape: Ny x Nx).
    - `xedges`, `yedges`:
      Histogram bin edges used to generate `hist_image`.
      They are returned as `extent` so GUI can display correct axis scale.
    - `bin_size`:
      Physical size per pixel/bin; used to convert radial axis to data units.
    - `settings`:
      rBasex parameter dict. Required keys are documented in the module header.

    Returns dict fields:
    - `image`: reconstructed image (or None if failed).
    - `extent`: `(xmin, xmax, ymin, ymax)` for plotting.
    - `r`, `i`, `beta`: recovered radial arrays.
    - `peaks`: extracted dominant peaks.
    - `display_percentile`: GUI display suggestion.
    - `error`: empty on success, exception text on failure.
    """
    try:
        # Step 1: inverse Abel transform with rBasex.
        recon_img, distr = abel.rbasex.rbasex_transform(
            hist_image,
            direction="inverse",
            order=int(settings["order"]),
            odd=bool(settings["odd"]),
            reg=settings["reg"],
            rmax=settings["rmax"],
            basis_dir=_rbasex_basis_dir(),
            verbose=False,
        )
        # Step 2: read radial distributions from pyabel object.
        # NOTE: for order>2, rIbeta() returns extra anisotropy components
        # (e.g. beta4, beta6, ...). Use the first beta component for current GUI.
        ri_beta = distr.rIbeta()
        components = list(ri_beta) if hasattr(ri_beta, "__len__") else []
        if len(components) < 2:
            raise RuntimeError("rBasex rIbeta() returned an unexpected number of outputs.")
        r_px = np.asarray(components[0], dtype=np.float64)
        i_r = np.asarray(components[1], dtype=np.float64)
        if len(components) >= 3:
            beta_r = np.asarray(components[2], dtype=np.float64)
        else:
            beta_r = np.zeros_like(i_r, dtype=np.float64)

        # Step 3: convert radius from pixel unit to data unit and align lengths.
        r_data = np.asarray(r_px, dtype=np.float64) * bin_size
        n = int(min(r_data.size, i_r.size, beta_r.size))
        if n <= 0:
            raise RuntimeError("rBasex returned empty radial outputs.")
        r_data = r_data[:n]
        i_r = i_r[:n]
        beta_r = beta_r[:n]

        # Step 4: extract major (r, beta) peaks from I(r).
        peaks = extract_peak_r_beta(
            r_data,
            i_r,
            beta_r,
            settings["peak_smooth_sigma"],
            settings["peak_height"],
            settings["peak_prominence"],
            settings["peak_min_dist_frac"],
            settings["max_peaks"],
        )
        return {
            "image": np.asarray(recon_img, dtype=np.float64),
            "extent": (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])),
            "r": r_data,
            "i": i_r,
            "beta": beta_r,
            "rbasex_components": int(len(components)),
            "peaks": peaks,
            "display_percentile": settings["display_percentile"],
            "error": "",
        }
    except Exception as exc:
        return {
            "image": None,
            "extent": None,
            "r": np.array([]),
            "i": np.array([]),
            "beta": np.array([]),
            "peaks": [],
            "display_percentile": settings.get("display_percentile", 99.7),
            "error": str(exc),
        }


def run_reconstructions_from_centered_data(
    centered_hist_data: dict | None,
    rbasex_settings: dict,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict | None:
    """Run the rBasex reconstruction from a centered histogram dict.

    Parameters:
    - `centered_hist_data`:
      Dict produced by core binning/denoising step, must include:
      `hist_denoised`, `xedges`, `yedges`, and `bin_size`.
    - `rbasex_settings`:
      Settings dict for the rBasex reconstruction method.

    Important orientation note:
    - `numpy.histogram2d` stores data as (x_bin, y_bin).
    - Reconstruction expects image-like order (y, x).
    - So this function transposes once before reconstruction.
    """
    if centered_hist_data is None:
        return None

    def emit_progress(frac: float, message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(float(frac), message)
        except Exception:
            pass

    emit_progress(0.05, "Preparing reconstruction input...")
    hist_xy = np.asarray(centered_hist_data["hist_denoised"], dtype=np.float64)
    xedges = centered_hist_data["xedges"]
    yedges = centered_hist_data["yedges"]
    if np.sum(hist_xy) <= 0:
        return None

    # histogram2d gives (x_bin, y_bin), while reconstruction code expects (y, x)
    recon_input = np.asarray(hist_xy.T, dtype=np.float64)
    bin_size_eff = float(centered_hist_data.get("bin_size", 1.0))

    emit_progress(0.9, "Running rBasex reconstruction...")
    rbasex_result = run_rbasex_reconstruction(recon_input, xedges, yedges, bin_size_eff, rbasex_settings)
    emit_progress(1.0, "Reconstruction algorithms finished.")
    return rbasex_result
