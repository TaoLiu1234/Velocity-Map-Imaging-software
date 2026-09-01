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
    - `area` is also included per peak: trapezoidal integral of I(r) over the
      valley-clipped segment (fallback: I at the peak index).

    Method / Limitations:
    - Peak positions are found on the Gaussian-smoothed, normalized profile,
      but the reported `i` is the *unsmoothed* intensity at that index, so a
      slightly shifted index can under-report the peak height.
    - Valley clipping (lowest smoothed point between adjacent peaks) bounds
      each peak's support; the trapezoid `area` therefore still contains any
      continuum/background under the peak (no baseline is subtracted), and
      overlap wings are split at the valley.
    - `beta` is clipped to [-2, 2]. For pure P2 physics |beta2| <= 2, so the
      clip only guards numerical noise; if higher odd/even orders leak into
      the extracted component, strong anisotropies can be distorted.
    - The radial grid is integer pixel indices (pyabel uses `arange(rmax+1)`),
      so peak radii are quantized to one pixel and `beta(r)` is reported at
      that raw resolution (no radial averaging, pyabel `window=1`).
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


def format_peak_text(peaks: list[dict], beta_available: bool = True) -> str:
    """Format recovered peaks for display in plot panels.

    Methods that do not recover the anisotropy (everything except rBasex)
    report ``beta=n/a`` instead of a misleading 0.
    """
    if not peaks:
        return "Recovered peaks:\n(no peak)"
    lines = ["Recovered peaks:"]
    for i, p in enumerate(peaks, start=1):
        area = p.get("area", p.get("i", 0.0))
        if beta_available:
            lines.append(
                f"{i}. r={float(p['r']):.3g}, beta={float(p['beta']):.3g}, intensity={float(area):.4g}"
            )
        else:
            lines.append(f"{i}. r={float(p['r']):.3g}, beta=n/a, intensity={float(area):.4g}")
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

    Notes:
    - The image origin must be the exact center of the central pixel (the
      core `build_centered_histogram` grid guarantees this); pyabel's default
      `origin="center"` is relied upon.
    - pyabel returns `r` as integer pixel indices (`arange(rmax+1)`), so
      `r * bin_size` maps the radial axis onto the centered data coordinates
      with zero offset; radial resolution is one pixel.
    - `rIbeta()` is called with the default `window=1`: I(r) and beta(r) are
      per-pixel values (beta is not radially averaged); I(r) is only smoothed
      later, inside `extract_peak_r_beta`, for peak detection.
    - `reg=None` (default) runs the inverse transform unregularized; noise
      amplification at large radii is then possible. For `order > 2`,
      `rIbeta()` returns additional anisotropy components which are ignored.
    - `rmax="MIN"` (the GUI default) limits the transform to the largest
      radius with at least one full quadrant of data.
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


# Abel inversion methods exposed in the GUI (verified against pyabel 0.9.1).
# rBasex is the default and the only method that natively recovers the
# anisotropy parameter beta(r); every other entry returns the inverted image
# only, so peaks are reported with beta = n/a for them.
ABEL_METHODS: list[tuple[str, str]] = [
    ("rbasex", "rBasex (recommended, gives beta)"),
    ("basex", "BASEX"),
    ("daun", "Daun (regularized)"),
    ("direct", "Direct integration"),
    ("hansenlaw", "Hansen-Law"),
    ("linbasex", "Lin-Basex"),
    ("onion_bordas", "Onion-Bordas"),
    ("three_point", "Three-point (Dasch)"),
    ("two_point", "Two-point"),
]

_METHOD_KEYS = frozenset(key for key, _ in ABEL_METHODS)


def normalize_abel_method(method) -> str:
    """Return a registered method key, mapping anything unknown to rbasex."""
    key = str(method or "").strip().lower()
    return key if key in _METHOD_KEYS else "rbasex"


def abel_method_label(method) -> str:
    """Return the human-readable label for a registered method key."""
    key = normalize_abel_method(method)
    for method_key, label in ABEL_METHODS:
        if method_key == key:
            return label
    return key


def run_abel_method_reconstruction(
    hist_image: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    bin_size: float,
    method: str,
    settings: dict,
) -> dict:
    """Run a generic pyabel inverse transform (any non-rBasex method).

    Uses the unified ``abel.Transform`` API. The returned result dict mirrors
    ``run_rbasex_reconstruction`` except that the anisotropy is not recoverable:
    ``beta`` is a zero array, ``beta_available`` is False, and I(r) is obtained
    by angular integration of the inverted image over all angles (mean per
    integer pixel radius). Peak finding reuses the same extractor and peak
    settings as the rBasex path.
    """
    try:
        recon_img = abel.Transform(
            hist_image,
            direction="inverse",
            method=normalize_abel_method(method),
        ).transform
        recon_img = np.asarray(recon_img, dtype=np.float64)

        # Angular-integrated radial profile I(r) on integer pixel radii,
        # centered at the array origin (the core binning grid guarantees the
        # center of the central pixel, same assumption as the rBasex path).
        ny, nx = recon_img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot(yy - (ny - 1) / 2.0, xx - (nx - 1) / 2.0)
        r_idx = np.rint(rr).astype(np.int64)
        r_max = int(min(int(r_idx.max()), min(ny, nx) // 2))
        flat_r = r_idx.ravel()
        flat_i = recon_img.ravel()
        counts = np.bincount(flat_r, minlength=r_max + 1)[: r_max + 1].astype(np.float64)
        sums = np.bincount(flat_r, weights=flat_i, minlength=r_max + 1)[: r_max + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            i_r = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
        i_r = np.nan_to_num(i_r, nan=0.0, posinf=0.0, neginf=0.0)
        r_data = np.arange(r_max + 1, dtype=np.float64) * bin_size
        beta_r = np.zeros_like(i_r)

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
            "image": recon_img,
            "extent": (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])),
            "r": r_data,
            "i": i_r,
            "beta": beta_r,
            "peaks": peaks,
            "display_percentile": settings["display_percentile"],
            "method": normalize_abel_method(method),
            "beta_available": False,
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
            "method": normalize_abel_method(method),
            "beta_available": False,
            "error": str(exc),
        }


def run_reconstructions_from_centered_data(
    centered_hist_data: dict | None,
    rbasex_settings: dict,
    progress_callback: Callable[[float, str], None] | None = None,
    method: str = "rbasex",
) -> dict | None:
    """Run the reconstruction from a centered histogram dict.

    Parameters:
    - `centered_hist_data`:
      Dict produced by core binning/denoising step, must include:
      `hist_denoised`, `xedges`, `yedges`, and `bin_size`.
    - `rbasex_settings`:
      Settings dict. rBasex uses every key; other methods use only the
      peak-finding and display keys.
    - `method`:
      Registered Abel method key (see `ABEL_METHODS`); anything unregistered
      falls back to rBasex.

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

    method_key = normalize_abel_method(method)
    label = abel_method_label(method_key)
    if method_key == "rbasex":
        emit_progress(0.9, "Running rBasex reconstruction...")
        result = run_rbasex_reconstruction(recon_input, xedges, yedges, bin_size_eff, rbasex_settings)
    else:
        emit_progress(0.9, f"Running {label} reconstruction...")
        result = run_abel_method_reconstruction(
            recon_input, xedges, yedges, bin_size_eff, method_key, rbasex_settings
        )
    emit_progress(1.0, "Reconstruction algorithms finished.")
    return result
