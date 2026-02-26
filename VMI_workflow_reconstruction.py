#!/usr/bin/env python3
from __future__ import annotations

"""Reconstruction helpers for VMI_workflow.

This module isolates numerical reconstruction code from GUI code.
The GUI only prepares centered/binning data and passes it here.

===============================================================================
Method A: rBasex reconstruction (`run_rbasex_reconstruction`)
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

===============================================================================
Method B: Backward reconstruction without forward-fit
(`run_backward_reconstruction_no_forward_fit`)
===============================================================================
Core pipeline (actual code path):
1. Build fitter object with mask size and `internal_params`.
2. Phase 0 shared init (`_init_shared_data`):
   - Build polar image `I(r, theta)` with cubic interpolation
     (`scipy.ndimage.map_coordinates(..., order=3)`).
   - Build radial profile and estimate noise region from outer radii.
   - Estimate baseline by frequency-domain rule:
     - if low-frequency ratio is high -> edge percentile baseline;
     - else -> robust noise median baseline.
   - Subtract baseline from radial profile and polar image.
   - Build per-radius noise model (Poisson + readout).
3. Optional external baseline subtraction (`baseline_factor`) in wrapper.
4. Phase 1 radial analysis (`_phase1_radial_analysis`):
   - Projection peak search with Gaussian smoothing.
   - Bayesian 1D denoise + Hansen-Law inverse Abel transform.
   - Savitzky-Golay smoothing (for peak localization only, not sigma fit).
   - Sigma estimation per peak using 3 estimators:
     `peak_widths`/FWHM, local Gaussian fit, second moment.
   - Keep only peaks passing SNR threshold.
5. Phase 2 angular analysis (`_phase2_angular_analysis`):
   - Optional radial window filter to isolate one peak from neighbors.
   - Refine peak radius by local intensity-max search in polar domain.
   - Estimate beta with multi-radius strategy around the refined radius:
     FFT-based beta + folded-profile fit beta, then weighted combination.
6. Reconstruct 2D image from recovered parameters.
7. Build approximate I(r), beta(r), then extract output peaks.
8. Return image + peaks + baseline bookkeeping values.

Backward Phase details (filters, windows, and where they are used):

Phase 0 shared init (before phase 1/2):
- `init_profile_smooth_sigma`
  Gaussian smoothing of radial profile before signal/noise split.
- `init_signal_threshold_frac`
  Threshold = `max(profile) * frac` to find outermost signal radius.
- `init_noise_margin_px`
  Extra radial margin after outermost signal before noise region.
- `init_noise_cap_frac`
  Upper cap so noise start does not exceed this fraction of radius.
- `init_min_noise_region_frac`
  Minimum required radial fraction for noise estimation.
- `baseline_edge_start_frac`, `baseline_edge_percentile`
  Used when low-frequency baseline branch is selected.
- `bayes_prior_sigma`
  Gaussian prior smoothing used to estimate signal spectrum in Bayesian denoise.
- `bayes_lowfreq_sigma`
  Width of low-frequency boost term in Wiener-like filter.
- `bayes_wiener_signal_weight`
  Mix weight between Wiener estimate and low-frequency boost.

Phase 1 radial analysis:
- Step 1.2 projection filtering + peak detection:
  - `phase1_proj_smooth_sigma`: Gaussian smoothing sigma for projection profile.
  - `phase1_peak_height_frac`: `find_peaks` height threshold fraction.
  - `phase1_peak_prominence_frac`: `find_peaks` prominence fraction.
  - `phase1_peak_distance_px`: minimum peak spacing in pixels.
- Step 1.3 Abel domain preprocessing:
  - Bayesian denoise is applied first (uses `bayes_*` params above).
  - 1D inverse Abel uses Hansen-Law transform.
  - `phase1_snr_switch` chooses high/low Savitzky-Golay window branch.
  - `phase1_abel_savgol_window_high`: odd window length at high SNR.
  - `phase1_abel_savgol_window_low`: odd window length at low SNR.
  - `phase1_abel_savgol_polyorder`: polynomial order for Savitzky-Golay.
- Step 1.4 peak sigma/quality filtering:
  - `phase1_snr_low`, `phase1_snr_high`: acceptance threshold below/above
    `phase1_snr_switch`.
  - Sigma comes from weighted combination of:
    `scipy.signal.peak_widths` (if valid), local Gaussian fit, and second moment.

Phase 2 angular analysis:
- Optional radial separation between neighboring peaks:
  - `phase2_radial_filter_sigma_scale`: width scale of Gaussian radial window
    inside neighbor boundaries.
- Radius refinement around each phase-1 peak:
  - `phase2_opt_radius_sigma_scale`: search half-width factor vs sigma.
  - `phase2_opt_radius_min_search`: minimum search half-width in pixels.
- Multi-radius beta extraction:
  - `phase2_multi_sigma_scale`: search half-width factor around refined radius.
  - `phase2_multi_min_search`: minimum multi-radius search half-width.
  - `phase2_multi_n_use`: number of strongest nearby radii used for beta.
  - `phase2_beta_smooth_sigma`: Gaussian smoothing on folded angular profile
    before linear/nonlinear beta fitting.
  - At each used radius:
    - FFT route uses full 0..2pi profile.
    - Fit route folds profile to 0..pi and fits `I(theta)=I0*(1+beta*P2(cos(theta)))`.
  - Final beta is weighted by local intensity and reconciles FFT/fit outputs.

`settings` keys for backward method:
- `n_theta`:
  Number of angular samples used in analysis.
- `mask_radius`:
  Circular mask radius in pixels.
- `baseline_factor`:
  Multiplier applied to estimated external baseline before subtraction.
- `peak_smooth_sigma`:
  Gaussian smoothing sigma for peak extraction from recovered I(r).
- `peak_height`:
  Minimum normalized peak height threshold.
- `peak_prominence`:
  Minimum normalized peak prominence threshold.
- `peak_min_dist_frac`:
  Minimum distance between peaks as fraction of radial profile length.
- `max_peaks`:
  Maximum number of output peaks.
- `display_percentile`:
  Suggested image display percentile in GUI.
- `internal_params`:
  Dictionary forwarded to `PhysicsBasedFitter`; each key controls one
  internal stage (init noise estimation, baseline estimation, phase-1
  radial analysis, phase-2 angular optimization/filtering).

`internal_params` key map (backward method):
- `init_profile_smooth_sigma`:
  Used in `_init_shared_data`; smooths radial profile before finding signal edge.
- `init_signal_threshold_frac`:
  Used in `_init_shared_data`; defines threshold to locate outermost signal radius.
- `init_noise_margin_px`:
  Used in `_init_shared_data`; expands radius after signal edge before noise area.
- `init_noise_cap_frac`:
  Used in `_init_shared_data`; caps maximum start radius of noise region.
- `init_min_noise_region_frac`:
  Used in `_init_shared_data`; ensures enough noise samples.
- `baseline_edge_start_frac`:
  Used in `_estimate_baseline_frequency_domain`; start radius for edge baseline.
- `baseline_edge_percentile`:
  Used in `_estimate_baseline_frequency_domain`; percentile on edge region.
- `bayes_prior_sigma`:
  Used in `_denoise_radial_profile_bayesian`; prior smoothness of radial signal.
- `bayes_lowfreq_sigma`:
  Used in `_denoise_radial_profile_bayesian`; low-frequency boost width in Wiener mix.
- `bayes_wiener_signal_weight`:
  Used in `_denoise_radial_profile_bayesian`; Wiener vs low-frequency boost balance.
- `phase1_proj_smooth_sigma`:
  Used in `_phase1_radial_analysis`; Gaussian smoothing before phase-1 `find_peaks`.
- `phase1_peak_height_frac`:
  Used in `_phase1_find_peaks_in_projection`; relative height threshold.
- `phase1_peak_prominence_frac`:
  Used in `_phase1_find_peaks_in_projection`; relative prominence threshold.
- `phase1_peak_distance_px`:
  Used in `_phase1_find_peaks_in_projection`; minimum peak distance.
- `phase1_snr_switch`:
  Used in `_phase1_radial_analysis`; toggles SNR branch (windows + SNR threshold set).
- `phase1_snr_low`:
  Used in `_phase1_radial_analysis`; acceptance threshold for low-SNR datasets.
- `phase1_snr_high`:
  Used in `_phase1_radial_analysis`; acceptance threshold for high-SNR datasets.
- `phase1_abel_savgol_window_high`:
  Used in `_phase1_radial_analysis`; Savitzky-Golay window length when SNR is high.
- `phase1_abel_savgol_window_low`:
  Used in `_phase1_radial_analysis`; Savitzky-Golay window length when SNR is low.
- `phase1_abel_savgol_polyorder`:
  Used in `_phase1_radial_analysis`; Savitzky-Golay polynomial order.
- `phase2_beta_smooth_sigma`:
  Used in `_phase2_estimate_beta_single_radius`; smooths folded angular profile.
- `phase2_opt_radius_sigma_scale`:
  Used in `_phase2_find_optimal_radius`; search range scale around phase-1 radius.
- `phase2_opt_radius_min_search`:
  Used in `_phase2_find_optimal_radius`; minimum search span in pixels.
- `phase2_multi_sigma_scale`:
  Used in `_phase2_multi_radius_beta`; search range scale for multi-radius beta.
- `phase2_multi_n_use`:
  Used in `_phase2_multi_radius_beta`; number of local strongest radii to combine.
- `phase2_multi_min_search`:
  Used in `_phase2_multi_radius_beta`; minimum local radius search span.
- `phase2_radial_filter_sigma_scale`:
  Used in `_phase2_radial_filter`; Gaussian radial-window width scale for peak isolation.

Dependencies:
- `pyabel` (for rBasex).
- `Abel_backward_reconstruction.py` in the same folder (for backward method).
"""

import contextlib
import io

import abel
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

try:
    from Abel_backward_reconstruction import (
        PhysicsBasedFitter as BackwardPhysicsBasedFitter,
        reconstruct_2d_from_params as backward_reconstruct_2d_from_params,
    )
except Exception:
    BackwardPhysicsBasedFitter = None
    backward_reconstruct_2d_from_params = None


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

    out = []
    for idx in peaks:
        if not (0 <= idx < len(r_axis) and 0 <= idx < len(beta_profile)):
            continue
        out.append(
            {
                "r": float(r_axis[idx]),
                "beta": float(np.clip(beta_profile[idx], -2.0, 2.0)),
                "i": float(intensity[idx]),
            }
        )
    return out


def format_peak_text(peaks: list[dict]) -> str:
    """Format recovered peaks for display in plot panels."""
    if not peaks:
        return "Recovered r, beta:\n(no peak)"
    lines = ["Recovered r, beta:"]
    for i, p in enumerate(peaks, start=1):
        lines.append(f"{i}. r={p['r']:.3g}, beta={p['beta']:.3g}")
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
            basis_dir=None,
            verbose=False,
        )
        # Step 2: read radial distributions from pyabel object.
        r_px, i_r, beta_r = distr.rIbeta()
        # Step 3: convert radius from pixel unit to data unit.
        r_data = np.asarray(r_px, dtype=np.float64) * bin_size
        i_r = np.asarray(i_r, dtype=np.float64)
        beta_r = np.asarray(beta_r, dtype=np.float64)

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


def run_backward_reconstruction_no_forward_fit(
    hist_image: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    bin_size: float,
    settings: dict,
) -> dict:
    """Run backward reconstruction using existing fitter (without forward-fit stage).

    Parameters:
    - `hist_image`:
      2D centered projection image (input for reconstruction).
    - `xedges`, `yedges`:
      Bin-edge arrays used to define output plotting extent.
    - `bin_size`:
      Physical size per bin, used for converting radius units.
    - `settings`:
      Backward method settings dict. Key meanings are documented in module header.

    Returns dict fields:
    - `image`, `extent`, `r`, `i`, `beta`, `peaks`, `display_percentile`, `error`
      (same role as rBasex output),
    - plus baseline diagnostics:
      `baseline_used`, `baseline_external`, `baseline_internal`, `internal_params`.
    """
    internal_params = dict(settings.get("internal_params", {}))
    if BackwardPhysicsBasedFitter is None or backward_reconstruct_2d_from_params is None:
        return {
            "image": None,
            "extent": None,
            "r": np.array([]),
            "i": np.array([]),
            "beta": np.array([]),
            "peaks": [],
            "display_percentile": settings.get("display_percentile", 99.7),
            "baseline_used": 0.0,
            "baseline_external": 0.0,
            "baseline_internal": 0.0,
            "internal_params": internal_params,
            "error": "Abel_backward_reconstruction import failed",
        }

    try:
        # Step 1: create fitter and inject user internal parameters.
        fitter = BackwardPhysicsBasedFitter(
            hist_image.shape[0],
            mask_radius=int(settings["mask_radius"]),
            params=internal_params,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            # Step 2: initialize shared data and estimate baseline.
            baseline_factor = float(settings.get("baseline_factor", 0.0))
            fitter._init_shared_data(hist_image, n_theta=int(settings["n_theta"]))
            baseline_external_ref = float(fitter._shared_noise.get("estimated_baseline", 0.0))
            external_subtract = baseline_external_ref * baseline_factor
            # Step 3: optional baseline subtraction then re-init shared data.
            if external_subtract > 0:
                img_corr = np.maximum(hist_image - external_subtract, 0)
                fitter._init_shared_data(img_corr, n_theta=int(settings["n_theta"]))
            else:
                img_corr = hist_image
            baseline_internal = float(fitter._shared_noise.get("estimated_baseline", 0.0))
            # Step 4: phase-1 radial analysis.
            phase1 = fitter._phase1_radial_analysis(img_corr)
            if not phase1:
                return {
                    "image": np.zeros_like(hist_image, dtype=np.float64),
                    "extent": (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])),
                    "r": np.array([]),
                    "i": np.array([]),
                    "beta": np.array([]),
                    "peaks": [],
                    "display_percentile": settings.get("display_percentile", 99.7),
                    "baseline_used": external_subtract + baseline_internal,
                    "baseline_external": external_subtract,
                    "baseline_internal": baseline_internal,
                    "internal_params": internal_params,
                        "error": "",
                    }
            # Step 5: phase-2 angular analysis to get model parameters.
            params = fitter._phase2_angular_analysis(img_corr, phase1)

        # Step 6: synthesize 2D image from recovered parameters.
        recon_img = np.asarray(backward_reconstruct_2d_from_params(params, hist_image.shape[0]), dtype=np.float64)
        # Step 7: build smooth I(r), beta(r) summaries from parameter set.
        r_px = fitter.r_grid_1d.astype(np.float64)
        i_r = np.zeros_like(r_px)
        beta_num = np.zeros_like(r_px)
        beta_den = np.zeros_like(r_px)
        for p in params:
            r0 = float(p.get("r", 0.0))
            sig = max(float(p.get("sigma", 1.0)), 1e-6)
            amp = max(float(p.get("amp", 0.0)), 0.0)
            beta_val = float(p.get("beta", 0.0))
            radial_w = np.exp(-((r_px - r0) ** 2) / (2.0 * sig**2))
            i_r += amp * radial_w
            beta_num += beta_val * radial_w
            beta_den += radial_w
        beta_r = np.divide(beta_num, beta_den, out=np.zeros_like(beta_num), where=beta_den > 1e-10)
        r_data = r_px * bin_size

        # Step 8: extract dominant peaks.
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
        if not peaks:
            for p in sorted(params, key=lambda q: float(q.get("r", 0.0)))[: settings["max_peaks"]]:
                peaks.append(
                    {
                        "r": float(p.get("r", 0.0) * bin_size),
                        "beta": float(np.clip(p.get("beta", 0.0), -2.0, 2.0)),
                        "i": float(p.get("amp", 0.0)),
                    }
                )

        return {
            "image": recon_img,
            "extent": (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])),
            "r": r_data,
            "i": i_r,
            "beta": beta_r,
            "peaks": peaks,
            "display_percentile": settings.get("display_percentile", 99.7),
            "baseline_used": external_subtract + baseline_internal,
            "baseline_external": external_subtract,
            "baseline_internal": baseline_internal,
            "internal_params": internal_params,
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
            "baseline_used": 0.0,
            "baseline_external": 0.0,
            "baseline_internal": 0.0,
            "internal_params": internal_params,
            "error": str(exc),
        }


def run_reconstructions_from_centered_data(
    centered_hist_data: dict | None,
    rbasex_settings: dict,
    backward_settings: dict,
) -> tuple[dict | None, dict | None]:
    """Run rBasex and backward reconstructions from centered histogram dict.

    Parameters:
    - `centered_hist_data`:
      Dict produced by core binning/denoising step, must include:
      `hist_denoised`, `xedges`, `yedges`, and `bin_size`.
    - `rbasex_settings`, `backward_settings`:
      Settings dicts for each reconstruction method.

    Important orientation note:
    - `numpy.histogram2d` stores data as (x_bin, y_bin).
    - Reconstruction expects image-like order (y, x).
    - So this function transposes once before reconstruction.
    """
    if centered_hist_data is None:
        return None, None

    hist_xy = np.asarray(centered_hist_data["hist_denoised"], dtype=np.float64)
    xedges = centered_hist_data["xedges"]
    yedges = centered_hist_data["yedges"]
    if np.sum(hist_xy) <= 0:
        return None, None

    # histogram2d gives (x_bin, y_bin), while reconstruction code expects (y, x)
    recon_input = np.asarray(hist_xy.T, dtype=np.float64)
    bin_size_eff = float(centered_hist_data.get("bin_size", 1.0))

    rbasex_result = run_rbasex_reconstruction(recon_input, xedges, yedges, bin_size_eff, rbasex_settings)
    backward_result = run_backward_reconstruction_no_forward_fit(
        recon_input,
        xedges,
        yedges,
        bin_size_eff,
        backward_settings,
    )
    return rbasex_result, backward_result
