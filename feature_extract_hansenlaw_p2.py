from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


_EPS = 1e-12
_LAST_PIPELINE_STATE: Dict[str, Any] = {}


@dataclass
class ExtractConfig:
    radial_bins: int = 560
    profile_smooth_sigma: float = 1.1
    peak_scales: Tuple[float, ...] = (0.7, 1.2, 1.8, 2.5)
    min_prominence_frac: float = 0.018
    min_peak_distance_bins: int = 7
    refine_window_bins: int = 16
    beta_window_sigma_mult: float = 1.4


def get_last_hansenlaw_pipeline_state() -> Dict[str, Any]:
    return dict(_LAST_PIPELINE_STATE)


def fit_xy_feature_hansenlaw_p2(
    xy_data: np.ndarray,
    n_peaks: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    ideal_mode = bool(kwargs.get("ideal_mode", False))
    cfg = ExtractConfig(
        radial_bins=int(kwargs.get("radial_bins", 560 if ideal_mode else 512)),
        profile_smooth_sigma=float(kwargs.get("profile_smooth_sigma", 1.1 if ideal_mode else 1.4)),
        peak_scales=tuple(kwargs.get("peak_scales", (0.7, 1.2, 1.8, 2.5) if ideal_mode else (0.8, 1.4, 2.2))),
        min_prominence_frac=float(kwargs.get("min_prominence_frac", 0.018 if ideal_mode else 0.03)),
        min_peak_distance_bins=int(kwargs.get("min_peak_distance_bins", 7 if ideal_mode else 8)),
        refine_window_bins=int(kwargs.get("refine_window_bins", 16 if ideal_mode else 14)),
        beta_window_sigma_mult=float(kwargs.get("beta_window_sigma_mult", 1.4 if ideal_mode else 1.8)),
    )

    peaks, context = extract_hansenlaw(xy_data=xy_data, n_peaks=n_peaks, config=cfg, return_context=True)
    _LAST_PIPELINE_STATE.clear()
    _LAST_PIPELINE_STATE.update(
        {
            "followup_enabled": bool(kwargs.get("run_followup_stages", False)),
            "ideal_mode": ideal_mode,
            "line_shape_requested": str(kwargs.get("line_shape", "gaussian")),
            "line_shape_used": "gaussian",
            "xy_count": int(context.get("xy_count", 0)),
        }
    )
    return peaks


def extract_hansenlaw(
    xy_data: np.ndarray,
    n_peaks: Optional[int] = None,
    *,
    config: Optional[ExtractConfig] = None,
    return_context: bool = False,
) -> Any:
    cfg = config or ExtractConfig()
    arr = _sanitize_xy(xy_data)
    if arr.size == 0:
        return ([], {}) if return_context else []

    x = arr[:, 0]
    y = arr[:, 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    r_max = float(np.percentile(r, 99.7) * 1.04) if len(r) > 0 else 1.0
    r_max = max(r_max, 1.0)
    bins = np.linspace(0.0, r_max, max(int(cfg.radial_bins), 80) + 1)
    m0, edges = np.histogram(r, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    m0 = m0.astype(float)
    dr = float(np.median(np.diff(centers))) if len(centers) > 1 else float(r_max / max(len(centers), 1))
    dr = max(dr, 1e-6)

    m0_smoothed = gaussian_filter1d(m0, sigma=max(float(cfg.profile_smooth_sigma), 0.1), mode="nearest")
    rho0 = _hansen_law_inverse_profile(m0_smoothed, dr=dr)
    rho0 = np.maximum(rho0, 0.0)
    center_mask_bins = max(1, int(round(0.4 / dr)))
    rho0[: min(center_mask_bins, len(rho0))] = 0.0

    dist_bins = max(int(cfg.min_peak_distance_bins), int(round(0.9 / dr)))
    base_idx = _detect_peaks_multiscale(
        profile=rho0,
        radii=centers,
        n_peaks=n_peaks,
        peak_scales=cfg.peak_scales,
        min_prominence_frac=cfg.min_prominence_frac,
        min_peak_distance_bins=dist_bins,
    )
    peak_indices = _augment_peak_candidates_with_m0(
        peak_indices=base_idx,
        m0_profile=m0_smoothed,
        rho0_profile=rho0,
        radii=centers,
        n_peaks=n_peaks,
        peak_scales=cfg.peak_scales,
        min_peak_distance_bins=dist_bins,
        min_prominence_frac=cfg.min_prominence_frac,
    )

    peaks = _refine_peaks_gaussian(
        profile=rho0,
        aux_profile=m0_smoothed,
        radii=centers,
        peak_indices=peak_indices,
        refine_window_bins=cfg.refine_window_bins,
    )
    if not peaks:
        return ([], {}) if return_context else []

    _regularize_sigmas(peaks)
    _estimate_beta_for_peaks(peaks=peaks, x=x, y=y, r=r, theta=theta, beta_window_sigma_mult=max(float(cfg.beta_window_sigma_mult), 1.0))
    _assign_probabilistic_amplitudes(peaks=peaks, r=r)

    if n_peaks is not None and int(n_peaks) > 0:
        nn = int(n_peaks)
        peaks = sorted(peaks, key=lambda p: float(p.get("amp", 0.0)), reverse=True)[:nn]

    peaks = sorted(peaks, key=lambda p: float(p["r0"]))
    context = {
        "r_max": r_max,
        "dr": dr,
        "radial_centers": centers,
        "m0_profile": m0,
        "rho0_profile": rho0,
        "xy_count": int(arr.shape[0]),
    }
    return (peaks, context) if return_context else peaks


def _sanitize_xy(xy_data: np.ndarray) -> np.ndarray:
    if xy_data is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(xy_data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    arr = arr[:, :2]
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def _hansen_law_inverse_profile(profile: np.ndarray, dr: float) -> np.ndarray:
    p = np.maximum(np.asarray(profile, dtype=float), 0.0)
    if p.size <= 2:
        return p.copy()
    try:
        import abel

        try:
            inv = abel.hansenlaw.hansenlaw_transform(p, direction="inverse", dr=float(max(dr, 1e-6)))
        except TypeError:
            inv = abel.hansenlaw.hansenlaw_transform(p, direction="inverse")
        inv = np.asarray(inv, dtype=float)
        inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
        inv = np.maximum(inv, 0.0)
        if inv.size > 5:
            inv = gaussian_filter1d(inv, sigma=0.8, mode="nearest")
        return np.maximum(inv, 0.0)
    except Exception:
        grad = np.gradient(p, edge_order=1)
        inv = p - 0.55 * grad
        inv = gaussian_filter1d(np.maximum(inv, 0.0), sigma=1.0, mode="nearest")
        return np.maximum(inv, 0.0)


def _estimate_profile_noise(profile: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(profile, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0, 0.0
    start = int(0.75 * arr.size)
    tail = arr[start:] if start < arr.size else arr
    if tail.size < 8:
        tail = arr[arr.size // 2 :]
    if tail.size == 0:
        tail = arr
    floor = float(np.median(tail))
    smooth = gaussian_filter1d(arr, sigma=2.2, mode="nearest")
    residual = arr - smooth
    tail_r = residual[start:] if start < residual.size else residual
    if tail_r.size < 8:
        tail_r = residual[residual.size // 2 :]
    if tail_r.size == 0:
        tail_r = residual
    noise_std = float(max(np.std(tail_r), 1e-12))
    snr = float(max((float(np.max(arr)) - floor) / noise_std, 0.0))
    return floor, noise_std, snr


def _detect_peaks_multiscale(
    *,
    profile: np.ndarray,
    radii: np.ndarray,
    n_peaks: Optional[int],
    peak_scales: Sequence[float],
    min_prominence_frac: float,
    min_peak_distance_bins: int,
) -> List[int]:
    if len(profile) < 5:
        return [int(np.argmax(profile))] if len(profile) > 0 else []
    signal = np.maximum(np.asarray(profile, dtype=float), 0.0)
    max_v = float(np.max(signal))
    if max_v <= 0:
        return [int(np.argmax(signal))]

    _, noise_std, snr = _estimate_profile_noise(signal)
    dist = max(int(min_peak_distance_bins), 1)
    score = np.zeros_like(signal, dtype=float)
    for sigma in peak_scales:
        sm = gaussian_filter1d(signal, sigma=max(float(sigma), 0.2), mode="nearest")
        if float(np.max(sm)) <= 0:
            continue
        curv = np.maximum(-np.gradient(np.gradient(sm, edge_order=1), edge_order=1), 0.0)
        curv_scale = float(max(np.percentile(curv, 90), _EPS))
        base_prom = max(float(np.max(sm)) * float(min_prominence_frac), 1.2 * noise_std, _EPS)
        if snr < 10:
            base_prom = max(base_prom, 2.0 * noise_std)
        elif snr < 20:
            base_prom = max(base_prom, 1.5 * noise_std)
        for prom_scale in (1.0, 0.72, 0.5):
            prom = max(base_prom * prom_scale, 0.5 * noise_std, _EPS)
            idx, props = find_peaks(sm, prominence=prom, distance=dist)
            prom_arr = props.get("prominences", np.zeros(0, dtype=float))
            for j, i in enumerate(idx):
                if int(i) <= 1:
                    continue
                radius_norm = float(radii[i] / max(radii[-1], _EPS))
                radius_weight = 1.0 + 0.20 * radius_norm
                curvature_weight = 1.0 + 0.45 * float(curv[i] / curv_scale)
                sc = float(prom_arr[j]) * radius_weight * curvature_weight
                if sc > score[i]:
                    score[i] = sc
    if not np.any(score > 0):
        return [int(np.argmax(signal))]

    ranked = np.argsort(score)[::-1]
    selected: List[int] = []
    for idx in ranked:
        if score[idx] <= 0:
            break
        if int(idx) <= 1:
            continue
        if all(abs(int(idx) - s) >= dist for s in selected):
            selected.append(int(idx))
        if n_peaks is not None and int(n_peaks) > 0 and len(selected) >= int(n_peaks):
            break
    if not selected:
        selected = [int(np.argmax(signal))]
    return sorted(selected)


def _augment_peak_candidates_with_m0(
    *,
    peak_indices: Sequence[int],
    m0_profile: np.ndarray,
    rho0_profile: np.ndarray,
    radii: np.ndarray,
    n_peaks: Optional[int],
    peak_scales: Sequence[float],
    min_peak_distance_bins: int,
    min_prominence_frac: float,
) -> List[int]:
    if n_peaks is None or int(n_peaks) <= 1:
        return sorted(int(i) for i in peak_indices)
    target = int(n_peaks)
    dist = max(int(min_peak_distance_bins), 1)
    m0_idx = _detect_peaks_multiscale(
        profile=np.maximum(np.asarray(m0_profile, dtype=float), 0.0),
        radii=radii,
        n_peaks=max(target * 3, target + 2),
        peak_scales=peak_scales,
        min_prominence_frac=max(float(min_prominence_frac) * 0.5, 0.008),
        min_peak_distance_bins=min_peak_distance_bins,
    )
    raw_candidates = sorted({int(i) for i in list(peak_indices) + list(m0_idx)})
    min_radius = max(0.45, float(3.0 * np.median(np.diff(radii))) if len(radii) > 1 else 0.45)
    candidates = [int(i) for i in raw_candidates if float(radii[int(i)]) >= min_radius]
    if not candidates:
        candidates = raw_candidates
    if not candidates:
        return sorted(int(i) for i in peak_indices)

    m0 = np.maximum(np.asarray(m0_profile, dtype=float), 0.0)
    rho = np.maximum(np.asarray(rho0_profile, dtype=float), 0.0)
    m0_max = float(max(np.max(m0), _EPS))
    rho_max = float(max(np.max(rho), _EPS))

    scored: List[Tuple[float, int]] = []
    for idx in candidates:
        i = int(idx)
        s_rho = float(rho[i] / rho_max)
        s_m0 = float(m0[i] / m0_max)
        rad = float(radii[i] / max(radii[-1], _EPS))
        score = 0.65 * s_rho + 0.55 * s_m0 + 0.05 * rad
        scored.append((score, i))
    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        return sorted(int(i) for i in peak_indices)

    score_map = {int(i): float(s) for s, i in scored}
    selected: List[int] = [int(scored[0][1])]
    while len(selected) < target:
        best_idx: Optional[int] = None
        best_val = -np.inf
        for base_score, idx in scored:
            i = int(idx)
            if i in selected:
                continue
            min_d = min(abs(i - s) for s in selected)
            if min_d < dist:
                continue
            val = float(base_score) * (1.0 + 0.35 * float(min_d / max(dist, 1)))
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            break
        selected.append(int(best_idx))

    if len(selected) < target:
        for idx in sorted(candidates):
            if all(abs(int(idx) - s) >= dist for s in selected):
                selected.append(int(idx))
            if len(selected) >= target:
                break

    if len(selected) >= 2:
        r_span = float(max(radii[s] for s in selected) - min(radii[s] for s in selected))
        total_span = float(max(radii[-1] - radii[0], _EPS))
        if r_span < 0.30 * total_span:
            rest = [i for i in candidates if i not in selected]
            if rest:
                mean_sel = float(np.mean([radii[s] for s in selected]))
                far_idx = max(rest, key=lambda i: abs(float(radii[i]) - mean_sel))
                low_sel = min(selected, key=lambda i: score_map.get(int(i), 0.0))
                if score_map.get(int(far_idx), 0.0) >= 0.55 * score_map.get(int(low_sel), 0.0):
                    selected = [int(far_idx) if int(i) == int(low_sel) else int(i) for i in selected]
    return sorted(selected[:target])


def _gaussian_with_floor(x: np.ndarray, amp: float, r0: float, sigma: float, floor: float) -> np.ndarray:
    sig = max(float(sigma), 1e-4)
    return float(amp) * np.exp(-0.5 * ((x - float(r0)) / sig) ** 2) + float(floor)


def _fit_sigma_from_aux_profile(
    *,
    aux_profile: np.ndarray,
    radii: np.ndarray,
    peak_index: int,
    r0_guess: float,
    dr: float,
    half_window: int,
) -> float:
    n = len(aux_profile)
    i0 = max(int(peak_index) - int(half_window), 0)
    i1 = min(int(peak_index) + int(half_window) + 1, n)
    xw = radii[i0:i1]
    yw = np.maximum(aux_profile[i0:i1], 0.0)
    if len(xw) < 7 or float(np.max(yw)) <= 0:
        return float("nan")
    amp0 = float(max(np.max(yw) - np.min(yw), 1.0))
    floor0 = float(np.min(yw))
    sigma0 = float(max(0.18, 10.0 * dr))
    lo = [0.0, float(xw[0]), max(2.0 * dr, 0.03), 0.0]
    hi = [float(np.max(yw) * 4.0 + 1.0), float(xw[-1]), float((xw[-1] - xw[0]) * 0.9), float(np.max(yw))]
    try:
        popt, _ = curve_fit(
            _gaussian_with_floor,
            xw,
            yw,
            p0=[amp0, float(np.clip(r0_guess, xw[0], xw[-1])), sigma0, floor0],
            bounds=(lo, hi),
            maxfev=12000,
        )
        return float(abs(popt[2]))
    except Exception:
        weights = np.maximum(yw - np.min(yw), 0.0)
        sw = float(np.sum(weights))
        if sw <= _EPS:
            return float("nan")
        r0 = float(np.sum(xw * weights) / sw)
        var = float(np.sum(((xw - r0) ** 2) * weights) / sw)
        return float(np.sqrt(max(var, 0.0)))


def _refine_peaks_gaussian(
    *,
    profile: np.ndarray,
    aux_profile: Optional[np.ndarray],
    radii: np.ndarray,
    peak_indices: Sequence[int],
    refine_window_bins: int,
) -> List[Dict[str, float]]:
    peaks: List[Dict[str, float]] = []
    n = len(profile)
    if n == 0:
        return peaks
    dr = float(np.median(np.diff(radii))) if len(radii) > 1 else 1.0
    dr = max(dr, 1e-4)
    half_window = max(int(refine_window_bins), 4)
    for idx in peak_indices:
        i0 = max(int(idx) - half_window, 0)
        i1 = min(int(idx) + half_window + 1, n)
        xw = radii[i0:i1]
        yw = np.maximum(profile[i0:i1], 0.0)
        if len(xw) < 5 or float(np.max(yw)) <= 0:
            continue
        y_floor = float(np.min(yw))
        y_pos = np.maximum(yw - y_floor, 0.0)
        amp0 = float(max(np.max(y_pos), 1.0))
        r0_0 = float(radii[int(idx)])
        sigma0 = float(max(2.0 * dr, 0.05))
        lo = [0.0, float(xw[0]), 0.25 * dr, 0.0]
        hi = [float(np.max(yw) * 3.0 + 1.0), float(xw[-1]), float((xw[-1] - xw[0]) * 1.5), float(np.max(yw))]
        try:
            popt, _ = curve_fit(
                _gaussian_with_floor,
                xw,
                yw,
                p0=[amp0, r0_0, sigma0, y_floor],
                bounds=(lo, hi),
                maxfev=8000,
            )
            amp_fit, r0_fit, sigma_fit, _ = popt
            weights = np.maximum(y_pos, 0.0)
            sw = float(np.sum(weights))
            if sw > _EPS:
                sigma_moment = float(np.sqrt(max(np.sum(((xw - r0_fit) ** 2) * weights) / sw, 0.0)))
                sigma_fit = 0.7 * float(abs(sigma_fit)) + 0.3 * sigma_moment
        except Exception:
            weights = y_pos
            sw = float(np.sum(weights))
            if sw <= _EPS:
                continue
            r0_fit = float(np.sum(xw * weights) / sw)
            var = float(np.sum(((xw - r0_fit) ** 2) * weights) / sw)
            sigma_fit = float(max(np.sqrt(max(var, 0.0)), 0.25 * dr))
            amp_fit = float(np.max(y_pos))

        if aux_profile is not None and len(aux_profile) == n:
            sigma_aux = _fit_sigma_from_aux_profile(
                aux_profile=np.asarray(aux_profile, dtype=float),
                radii=radii,
                peak_index=int(idx),
                r0_guess=float(r0_fit),
                dr=dr,
                half_window=max(3 * half_window, 28),
            )
            if np.isfinite(sigma_aux):
                sigma_aux = float(np.clip(sigma_aux, 0.04, max(2.5, 0.35 * r0_fit + 0.8)))
                sigma_fit = 0.60 * float(max(sigma_fit, 0.02)) + 0.40 * sigma_aux

        area_local = float(np.trapz(y_pos, xw)) if len(xw) > 1 else float(np.max(y_pos) * dr)
        peaks.append(
            {
                "r0": float(max(r0_fit, 0.0)),
                "sigma": float(max(sigma_fit, 0.02)),
                "beta": 0.0,
                "amp": float(max(area_local, amp_fit, 1e-6)),
            }
        )
    return peaks


def _regularize_sigmas(peaks: List[Dict[str, float]]) -> None:
    if not peaks:
        return
    sig = np.array([max(float(p.get("sigma", 0.05)), 0.02) for p in peaks], dtype=float)
    med = float(np.median(sig))
    for i, peak in enumerate(peaks):
        peak["sigma"] = float(max(0.02, 0.90 * sig[i] + 0.10 * med))


def _estimate_beta_for_peaks(
    *,
    peaks: List[Dict[str, float]],
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    beta_window_sigma_mult: float,
) -> None:
    _ = theta
    for peak in peaks:
        r0 = float(max(peak.get("r0", 0.0), 0.0))
        sigma = float(max(peak.get("sigma", 0.05), 0.02))
        window = max(beta_window_sigma_mult * sigma, 0.15)
        mask = np.abs(r - r0) <= window
        if int(np.sum(mask)) < 50:
            peak["beta"] = 0.0
            continue
        xm = x[mask]
        ym = y[mask]
        rm = np.maximum(r[mask], _EPS)
        phi_y = np.arctan2(xm, ym)
        mu = np.clip(ym / rm, -1.0, 1.0)
        beta_moment, c_moment = _beta_from_moment(mu)
        beta_fft, c_fft = _beta_from_fft(phi_y)
        beta_fit, c_fit = _beta_from_curve_fit(phi_y)
        beta_wls, c_wls = _beta_from_wls(phi_y)

        vals: List[float] = []
        weights: List[float] = []
        candidates = [
            (beta_moment, 0.03, c_moment),
            (beta_fft, 0.45, c_fft),
            (beta_fit, 0.20, c_fit),
            (beta_wls, 0.32, c_wls),
        ]
        for b, base_w, conf in candidates:
            if np.isfinite(b):
                vals.append(float(b))
                weights.append(float(base_w) * (0.2 + float(np.clip(conf, 0.0, 1.0))))
        if not vals:
            peak["beta"] = 0.0
            continue

        vals_arr = np.asarray(vals, dtype=float)
        w_arr = np.asarray(weights, dtype=float)
        if vals_arr.size >= 3:
            med = float(np.median(vals_arr))
            keep = np.abs(vals_arr - med) <= 0.55
            if int(np.sum(keep)) >= 2:
                vals_arr = vals_arr[keep]
                w_arr = w_arr[keep]

        w_arr /= max(float(np.sum(w_arr)), _EPS)
        beta = float(np.sum(vals_arr * w_arr))
        peak["beta"] = float(np.clip(beta, -1.0, 2.0))


def _beta_from_moment(mu: np.ndarray) -> Tuple[float, float]:
    if len(mu) < 30:
        return float("nan"), 0.0
    p2 = 0.5 * (3.0 * np.asarray(mu, dtype=float) ** 2 - 1.0)
    beta = float(np.clip(5.0 * np.mean(p2), -1.0, 2.0))
    sem = float(np.std(p2) / max(np.sqrt(len(p2)), 1.0))
    conf = float(np.clip(1.0 / (1.0 + 3.0 * sem), 0.0, 1.0))
    return beta, conf


def _beta_from_curve_fit(phi: np.ndarray) -> Tuple[float, float]:
    if len(phi) < 40:
        return float("nan"), 0.0
    hist, edges = np.histogram(phi, bins=72, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = hist.astype(float)
    if float(np.max(y)) <= 0:
        return float("nan"), 0.0

    def model(xx: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        mu = np.cos(xx)
        p2 = 0.5 * (3.0 * mu * mu - 1.0)
        return a * (1.0 + b * p2) + c

    try:
        sigma = np.sqrt(np.maximum(y, 1.0))
        popt, _ = curve_fit(
            model,
            centers,
            y,
            p0=[float(np.max(y)), 0.0, float(np.min(y))],
            bounds=([0.0, -1.0, 0.0], [np.inf, 2.0, np.inf]),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=6000,
        )
        pred = model(centers, *popt)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2) + _EPS)
        r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
        conf = float(np.clip(r2 * min(1.0, len(phi) / 600.0), 0.0, 1.0))
        return float(np.clip(popt[1], -1.0, 2.0)), conf
    except Exception:
        return float("nan"), 0.0


def _beta_from_fft(phi: np.ndarray) -> Tuple[float, float]:
    if len(phi) < 40:
        return float("nan"), 0.0
    hist, edges = np.histogram(phi, bins=72, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = hist.astype(float)
    if float(np.max(y)) <= 0:
        return float("nan"), 0.0
    mean_y = float(np.mean(y))
    if mean_y <= _EPS:
        return float("nan"), 0.0
    c2 = float((2.0 / len(y)) * np.sum(y * np.cos(2.0 * centers)))
    ratio = float(c2 / max(mean_y, _EPS))
    denom = 3.0 - ratio
    beta = (4.0 * ratio / denom) if abs(denom) > 1e-8 else (2.0 if ratio > 0 else -1.0)
    beta = float(np.clip(beta, -1.0, 2.0))
    conf = float(np.clip(abs(ratio) * min(1.0, len(phi) / 600.0), 0.0, 1.0))
    return beta, conf


def _beta_from_wls(phi: np.ndarray) -> Tuple[float, float]:
    if len(phi) < 40:
        return float("nan"), 0.0
    hist, edges = np.histogram(phi, bins=72, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = hist.astype(float)
    if float(np.max(y)) <= 0:
        return float("nan"), 0.0
    p2 = 0.5 * (3.0 * np.cos(centers) ** 2 - 1.0)
    X = np.column_stack([np.ones_like(p2), p2])
    w = np.maximum(y, 1.0)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    try:
        coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), 0.0
    a = float(coef[0])
    b = float(coef[1])
    if a <= _EPS:
        return float("nan"), 0.0
    beta = float(np.clip(b / a, -1.0, 2.0))
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + _EPS)
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    conf = float(np.clip(r2 * min(1.0, len(phi) / 600.0), 0.0, 1.0))
    return beta, conf


def _assign_probabilistic_amplitudes(peaks: List[Dict[str, float]], r: np.ndarray) -> None:
    if not peaks or len(r) == 0:
        return
    n = len(peaks)
    resp = np.zeros((len(r), n), dtype=float)
    for k, peak in enumerate(peaks):
        r0 = float(peak.get("r0", 0.0))
        sigma = float(max(peak.get("sigma", 0.05), 0.02))
        core = np.exp(-0.5 * ((r - r0) / sigma) ** 2) / (sigma + _EPS)
        resp[:, k] = core
    bg = np.full((len(r), 1), 1e-4, dtype=float)
    row_sum = np.sum(resp, axis=1, keepdims=True) + bg
    resp = resp / np.maximum(row_sum, _EPS)
    amps = np.sum(resp, axis=0)
    amp_sum = float(np.sum(amps))
    if amp_sum <= _EPS:
        amps = np.ones(n, dtype=float) / float(n)
    else:
        amps = amps / amp_sum
    for k, peak in enumerate(peaks):
        peak["amp"] = float(max(amps[k], 1e-9))
