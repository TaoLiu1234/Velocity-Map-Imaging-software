import argparse
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons

DEFAULT_ORIGIN_X = 199.0
DEFAULT_ORIGIN_Y = -1.0
DEFAULT_ORIGIN_Z = 0.0

# Keep method interfaces visible while optionally disabling heavy recomputation paths.
ENABLE_METHOD_ABS = False
ENABLE_METHOD_LN_RATIO = False
ENABLE_METHOD_SQRT = False


def _safe_float(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return num


def _parse_cell(cell):
    if cell is None:
        return None
    text = str(cell).strip()
    if text == "" or text.lower() in ("none", "nan"):
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _sorted_keys(values):
    def sort_key(item):
        num = _safe_float(item)
        if num is not None:
            return (0, num)
        return (1, str(item))

    return sorted(values, key=sort_key)


def _normalize_axis_limits(limits):
    if limits is None:
        return None
    if not isinstance(limits, (list, tuple, np.ndarray)) or len(limits) != 2:
        raise ValueError(f"Axis limits must be a 2-item tuple/list, got: {limits!r}")
    low = _safe_float(limits[0])
    high = _safe_float(limits[1])
    if low is None or high is None:
        raise ValueError(f"Axis limits must be finite numbers, got: {limits!r}")
    if low == high:
        raise ValueError(f"Axis limits must not be identical, got: {limits!r}")
    if low > high:
        low, high = high, low
    return (float(low), float(high))


def _parse_axis_range(text, axis_name):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--{axis_name} expects 'min,max', got: {text!r}")
    return _normalize_axis_limits(parts)


def _zoom_limits(limits, center, scale):
    low, high = limits
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if scale <= 0:
        return (float(low), float(high))
    if center is None or not np.isfinite(center):
        center = 0.5 * (low + high)
    new_low = center - (center - low) * scale
    new_high = center + (high - center) * scale
    if not np.isfinite(new_low) or not np.isfinite(new_high):
        return None
    if abs(new_high - new_low) <= 1e-15:
        return None
    return (float(new_low), float(new_high))


def _pan_limits(limits, delta):
    low, high = limits
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    shift = float(delta)
    return (float(low + shift), float(high + shift))


def find_latest_summary_file(results_dir):
    if not os.path.isdir(results_dir):
        return None

    candidates = []
    for name in os.listdir(results_dir):
        if not (name.endswith(".pkl") or name.endswith(".csv")):
            continue
        if not (
            name.startswith("processed_data_summary_")
            or name.startswith("processed_data_fallback_")
        ):
            continue
        if "processed_data_runs_" in name:
            continue
        full_path = os.path.join(results_dir, name)
        if os.path.isfile(full_path):
            candidates.append(full_path)

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_summary_from_csv(csv_path):
    summary = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fg = _parse_cell(row.get("field_gradient"))
            lens = _parse_cell(row.get("lens_vmi"))
            ke = _parse_cell(row.get("kinetic_energy_eV"))
            if fg is None or lens is None or ke is None:
                continue

            global_data = {
                "energy_resolution": _parse_cell(row.get("energy_resolution")),
                "energy_resolution_mean": _parse_cell(row.get("energy_resolution_mean")),
                "energy_resolution_variance": _parse_cell(row.get("energy_resolution_variance")),
                "energy_resolution_std": _parse_cell(row.get("energy_resolution_std")),
                "fwhm": _parse_cell(row.get("fwhm")),
                "fwhm_mean": _parse_cell(row.get("fwhm_mean")),
                "fwhm_variance": _parse_cell(row.get("fwhm_variance")),
                "fwhm_std": _parse_cell(row.get("fwhm_std")),
                "max_r": _parse_cell(row.get("max_r")),
                "generated_particles": _parse_cell(row.get("generated_particles")),
                "detected_particles": _parse_cell(row.get("detected_particles")),
                "valid": _parse_cell(row.get("valid")),
                "valid_run_count": _parse_cell(row.get("valid_run_count")),
                "total_runs": _parse_cell(row.get("total_runs")),
                "failure_reason": _parse_cell(row.get("failure_reason")),
            }
            summary.setdefault(fg, {}).setdefault(lens, {})[ke] = {
                "global": global_data,
                "local": {},
            }
    return summary


def load_summary_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_summary_from_csv(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "run_summaries" in data and isinstance(data["run_summaries"], list):
        if data["run_summaries"]:
            return data["run_summaries"][-1]
        return {}
    return data


def count_valid_energy_points(summary_data):
    valid = 0
    total = 0
    for fg_data in summary_data.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if not isinstance(ke_data, dict):
                    continue
                total += 1
                global_data = ke_data.get("global", {})
                er = global_data.get("energy_resolution_mean", global_data.get("energy_resolution"))
                if _safe_float(er) is not None:
                    valid += 1
    return valid, total


def _fmt_num(value, digits=3):
    num = _safe_float(value)
    if num is None:
        return str(value)
    return f"{num:.{digits}f}"


def _extract_er(global_data):
    return _safe_float(global_data.get("energy_resolution_mean", global_data.get("energy_resolution")))


def _extract_er_std(global_data):
    er_std = _safe_float(global_data.get("energy_resolution_std"))
    if er_std is not None:
        return er_std
    er_var = _safe_float(global_data.get("energy_resolution_variance"))
    if er_var is not None:
        return float(np.sqrt(max(er_var, 0.0)))
    return None


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_particle_count_mismatch(global_data):
    generated = _safe_int(global_data.get("generated_particles"))
    detected = _safe_int(global_data.get("detected_particles"))
    if generated is None or detected is None:
        return False
    return detected != generated


def _is_count_check_invalid(global_data):
    if not isinstance(global_data, dict):
        return False
    if global_data.get("plot_marker") == "x":
        return True
    count_flag = global_data.get("count_check_passed")
    if count_flag is False:
        return True
    if count_flag is True:
        return False
    reason = str(global_data.get("failure_reason") or "").lower()
    if "particle mismatch" in reason:
        return True
    return _has_particle_count_mismatch(global_data)


def _build_plot_metadata(global_data):
    if not isinstance(global_data, dict):
        global_data = {}
    return {
        "generated_particles": global_data.get("generated_particles"),
        "detected_particles": global_data.get("detected_particles"),
        "valid_run_count": global_data.get("valid_run_count"),
        "total_runs": global_data.get("total_runs"),
        "failure_reason": global_data.get("failure_reason"),
        "count_check_passed": global_data.get("count_check_passed"),
        "plot_marker": global_data.get("plot_marker"),
        "plot_skip": global_data.get("plot_skip"),
        "pipeline_stage": global_data.get("pipeline_stage"),
    }


def _extract_ra_rb_yz_from_pair(pair, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    # Preferred path for new data: compute radius from stored per-ion final coordinates.
    y_a = _safe_float(pair.get("y_a"))
    z_a = _safe_float(pair.get("z_a"))
    y_b = _safe_float(pair.get("y_b"))
    z_b = _safe_float(pair.get("z_b"))
    if y_a is not None and z_a is not None and y_b is not None and z_b is not None:
        ra = float(np.sqrt((y_a - origin_y) ** 2 + (z_a - origin_z) ** 2))
        rb = float(np.sqrt((y_b - origin_y) ** 2 + (z_b - origin_z) ** 2))
        return ra, rb

    # Backward compatibility for old files (r_a/r_b relative to yz origin 0,0).
    # This does NOT apply origin shift and is only a fallback.
    ra = _safe_float(pair.get("r_a"))
    rb = _safe_float(pair.get("r_b"))
    if ra is None or rb is None:
        return None, None
    return float(abs(ra)), float(abs(rb))


def _sorted_abs_radii(r_a, r_b):
    try:
        a = float(abs(r_a))
        b = float(abs(r_b))
    except (TypeError, ValueError):
        return None, None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None, None
    if a <= b:
        return a, b
    return b, a


def _extract_raw_ion_points_yz(global_data):
    raw_points = global_data.get("raw_ion_points_yz", [])
    if not isinstance(raw_points, list) or not raw_points:
        return []

    parsed = []
    for idx, point in enumerate(raw_points):
        ion_n = None
        y = None
        z = None

        if isinstance(point, dict):
            ion_n = _safe_float(point.get("ion_n"))
            y = _safe_float(point.get("y"))
            z = _safe_float(point.get("z"))
        elif isinstance(point, (list, tuple)):
            if len(point) >= 3:
                ion_n = _safe_float(point[0])
                y = _safe_float(point[1])
                z = _safe_float(point[2])
            elif len(point) >= 2:
                ion_n = float(idx + 1)
                y = _safe_float(point[0])
                z = _safe_float(point[1])
        if y is None or z is None:
            continue
        if not np.isfinite(y) or not np.isfinite(z):
            continue
        ion_sort = int(round(float(ion_n))) if ion_n is not None and np.isfinite(ion_n) else (idx + 1)
        parsed.append((ion_sort, float(y), float(z)))

    parsed.sort(key=lambda item: item[0])
    return parsed


def _build_pair_radii_from_raw_points(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    points = _extract_raw_ion_points_yz(global_data)
    if not points:
        return []

    particles_per_run = _safe_float(global_data.get("generated_particles_per_run"))
    total_runs = _safe_float(global_data.get("total_runs"))
    try:
        particles_per_run = int(particles_per_run) if particles_per_run is not None else 0
    except (TypeError, ValueError):
        particles_per_run = 0
    try:
        total_runs = int(total_runs) if total_runs is not None else 0
    except (TypeError, ValueError):
        total_runs = 0

    if particles_per_run <= 0 and total_runs > 0:
        particles_per_run = len(points) // total_runs
    if particles_per_run <= 0:
        particles_per_run = len(points)

    pair_radii = []
    total_points = len(points)
    if particles_per_run <= 0:
        return pair_radii

    full_blocks = total_points // particles_per_run
    remainder = total_points - (full_blocks * particles_per_run)
    blocks = []
    if full_blocks > 0:
        for block_idx in range(full_blocks):
            start = block_idx * particles_per_run
            end = start + particles_per_run
            blocks.append(points[start:end])
    if remainder >= 2:
        blocks.append(points[full_blocks * particles_per_run:])
    if not blocks:
        blocks = [points]

    for block in blocks:
        left = 0
        right = len(block) - 1
        while left < right:
            _, y_a, z_a = block[left]
            _, y_b, z_b = block[right]
            ra = float(np.sqrt((y_a - origin_y) ** 2 + (z_a - origin_z) ** 2))
            rb = float(np.sqrt((y_b - origin_y) ** 2 + (z_b - origin_z) ** 2))
            if np.isfinite(ra) and np.isfinite(rb):
                pair_radii.append((ra, rb))
            left += 1
            right -= 1
    return pair_radii


def _extract_ra_rb_from_pair_row(row, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    if not isinstance(row, (list, tuple, np.ndarray)) or len(row) < 8:
        return None, None
    y_a = _safe_float(row[4])
    z_a = _safe_float(row[5])
    y_b = _safe_float(row[6])
    z_b = _safe_float(row[7])
    if y_a is None or z_a is None or y_b is None or z_b is None:
        return None, None
    ra = float(np.sqrt((float(y_a) - origin_y) ** 2 + (float(z_a) - origin_z) ** 2))
    rb = float(np.sqrt((float(y_b) - origin_y) ** 2 + (float(z_b) - origin_z) ** 2))
    if not (np.isfinite(ra) and np.isfinite(rb)):
        return None, None
    return float(ra), float(rb)


def _extract_pair_radii(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    pairs = global_data.get("dr_over_r_pairs", [])
    pair_radii = []
    if isinstance(pairs, np.ndarray):
        arr = np.asarray(pairs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2:
            for row in arr:
                ra, rb = _extract_ra_rb_from_pair_row(row, origin_y=origin_y, origin_z=origin_z)
                if ra is not None and rb is not None:
                    pair_radii.append((ra, rb))
    elif isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, dict):
                ra, rb = _extract_ra_rb_yz_from_pair(pair, origin_y=origin_y, origin_z=origin_z)
            else:
                ra, rb = _extract_ra_rb_from_pair_row(pair, origin_y=origin_y, origin_z=origin_z)
            if ra is None or rb is None:
                continue
            if np.isfinite(ra) and np.isfinite(rb):
                pair_radii.append((float(ra), float(rb)))
    if pair_radii:
        return pair_radii, "pairs"

    pair_radii = _build_pair_radii_from_raw_points(global_data, origin_y=origin_y, origin_z=origin_z)
    if pair_radii:
        return pair_radii, "raw_points"
    return [], "missing"


def _extract_rmax_all_points(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    # Preferred: precomputed max radius across all detected points (paired + unpaired).
    rmax = _safe_float(global_data.get("r_max_all_points"))
    if rmax is not None and rmax > 0:
        return float(rmax), "all_points"

    # Optional backup if explicit all-r list is present.
    raw_all_r = global_data.get("all_r_values")
    if isinstance(raw_all_r, list):
        vals = [float(v) for v in raw_all_r if _safe_float(v) is not None]
        if vals:
            vmax = max(vals)
            if vmax > 0:
                return float(vmax), "all_points_list"

    raw_points = _extract_raw_ion_points_yz(global_data)
    if raw_points:
        vals = []
        for _, y, z in raw_points:
            rr = float(np.sqrt((y - origin_y) ** 2 + (z - origin_z) ** 2))
            if np.isfinite(rr):
                vals.append(rr)
        if vals:
            vmax = max(vals)
            if vmax > 0:
                return float(vmax), "raw_points"

    # Fallback for old files: estimate from paired points only.
    pair_r = []
    pair_radii, _ = _extract_pair_radii(global_data, origin_y=origin_y, origin_z=origin_z)
    for ra, rb in pair_radii:
        if np.isfinite(ra):
            pair_r.append(float(ra))
        if np.isfinite(rb):
            pair_r.append(float(rb))
    if pair_r:
        vmax = max(pair_r)
        if vmax > 0:
            return float(vmax), "pairs_fallback"
    return None, "missing"


def _compute_abs_ra_rb_over_rmax_er(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    pair_radii, _ = _extract_pair_radii(global_data, origin_y=origin_y, origin_z=origin_z)
    if not pair_radii:
        return None, None, None, 0, None, "missing"

    rmax, rmax_scope = _extract_rmax_all_points(
        global_data, origin_y=origin_y, origin_z=origin_z
    )
    if rmax is None or rmax <= 0:
        return None, None, None, 0, None, rmax_scope

    ratio_values = []
    for r_a, r_b in pair_radii:
        dr_abs = abs(r_a - r_b)
        ratio = dr_abs / rmax
        if np.isfinite(ratio):
            ratio_values.append(float(ratio))

    if not ratio_values:
        return None, None, None, 0, rmax, rmax_scope

    arr = np.array(ratio_values, dtype=float)
    er_mean = float(np.mean(arr))
    er_var = float(np.var(arr))
    er_std = float(np.sqrt(er_var))
    return er_mean, er_var, er_std, int(arr.size), rmax, rmax_scope


def _compute_sqrt_ra2_minus_rb2_er(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    pair_radii, _ = _extract_pair_radii(global_data, origin_y=origin_y, origin_z=origin_z)
    if not pair_radii:
        return None, None, None, 0, 0

    ratio_values = []
    skipped_negative = 0
    for r_a, r_b in pair_radii:
        r_mean = 0.5 * (r_a + r_b)
        if r_mean <= 0:
            continue

        delta_sq = (r_a * r_a) - (r_b * r_b)
        if delta_sq < 0:
            skipped_negative += 1
            continue

        dr_val = float(np.sqrt(delta_sq))
        ratio = dr_val / r_mean
        if np.isfinite(ratio):
            ratio_values.append(float(ratio))

    if not ratio_values:
        return None, None, None, 0, skipped_negative

    arr = np.array(ratio_values, dtype=float)
    er_mean = float(np.mean(arr))
    er_var = float(np.var(arr))
    er_std = float(np.sqrt(er_var))
    return er_mean, er_var, er_std, int(arr.size), skipped_negative


def _compute_abs_ra_rb_er(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    pair_radii, _ = _extract_pair_radii(global_data, origin_y=origin_y, origin_z=origin_z)
    if not pair_radii:
        return None, None, None, 0, 0

    ratio_values = []
    skipped_negative = 0
    for r_a, r_b in pair_radii:
        r_mean = 0.5 * (r_a + r_b)
        if r_mean <= 0:
            continue

        dr_abs = abs(r_a - r_b)
        ratio = dr_abs / r_mean
        if np.isfinite(ratio):
            ratio_values.append(float(ratio))

    if not ratio_values:
        return None, None, None, 0, skipped_negative

    arr = np.array(ratio_values, dtype=float)
    er_mean = float(np.mean(arr))
    er_var = float(np.var(arr))
    er_std = float(np.sqrt(er_var))
    return er_mean, er_var, er_std, int(arr.size), skipped_negative


def _compute_ln_r2_over_ln_r1_er(global_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    pair_radii, _ = _extract_pair_radii(global_data, origin_y=origin_y, origin_z=origin_z)
    if not pair_radii:
        return None, None, None, 0, 0, 0

    metric_values = []
    skipped_non_positive = 0
    skipped_non_finite = 0

    for r_a, r_b in pair_radii:
        r1, r2 = _sorted_abs_radii(r_a, r_b)
        if r1 is None or r2 is None:
            skipped_non_finite += 1
            continue
        if r1 <= 0 or r2 <= 0:
            skipped_non_positive += 1
            continue
        ratio = float(r2 / r1)
        if ratio <= 0 or not np.isfinite(ratio):
            skipped_non_finite += 1
            continue
        value = abs(float(np.log(ratio)))
        if np.isfinite(value):
            metric_values.append(float(value))
        else:
            skipped_non_finite += 1

    if not metric_values:
        return None, None, None, 0, skipped_non_positive, skipped_non_finite

    arr = np.array(metric_values, dtype=float)
    er_mean = float(np.mean(arr))
    er_var = float(np.var(arr))
    er_std = float(np.sqrt(er_var))
    return er_mean, er_var, er_std, int(arr.size), skipped_non_positive, skipped_non_finite


def build_sqrt_ra2_minus_rb2_summary(
    summary_data,
    origin_y=DEFAULT_ORIGIN_Y,
    origin_z=DEFAULT_ORIGIN_Z,
):
    alt_summary = {}
    for fg, fg_data in summary_data.items():
        if not isinstance(fg_data, dict):
            continue
        alt_summary.setdefault(fg, {})
        for lens, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            alt_summary[fg].setdefault(lens, {})
            for ke, ke_data in lens_data.items():
                global_data = {}
                if isinstance(ke_data, dict):
                    global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    global_data = {}

                metadata = _build_plot_metadata(global_data)
                if _is_count_check_invalid(global_data):
                    er_mean, er_var, er_std, pair_count, skipped_negative = (None, None, None, 0, 0)
                else:
                    er_mean, er_var, er_std, pair_count, skipped_negative = _compute_sqrt_ra2_minus_rb2_er(
                        global_data,
                        origin_y=origin_y,
                        origin_z=origin_z,
                    )
                alt_summary[fg][lens][ke] = {
                    "global": {
                        "energy_resolution": er_mean,
                        "energy_resolution_mean": er_mean,
                        "energy_resolution_variance": er_var,
                        "energy_resolution_std": er_std,
                        "pair_count": pair_count,
                        "skipped_negative_pairs": skipped_negative,
                        "valid": er_mean is not None,
                        **metadata,
                    },
                    "local": {},
                }
    return alt_summary


def build_abs_ra_rb_over_rmax_summary(
    summary_data,
    origin_y=DEFAULT_ORIGIN_Y,
    origin_z=DEFAULT_ORIGIN_Z,
):
    alt_summary = {}
    for fg, fg_data in summary_data.items():
        if not isinstance(fg_data, dict):
            continue
        alt_summary.setdefault(fg, {})
        for lens, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            alt_summary[fg].setdefault(lens, {})
            for ke, ke_data in lens_data.items():
                global_data = {}
                if isinstance(ke_data, dict):
                    global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    global_data = {}

                metadata = _build_plot_metadata(global_data)
                if _is_count_check_invalid(global_data):
                    er_mean, er_var, er_std, pair_count, rmax_used, rmax_scope = (None, None, None, 0, None, "count_mismatch")
                else:
                    er_mean, er_var, er_std, pair_count, rmax_used, rmax_scope = _compute_abs_ra_rb_over_rmax_er(
                        global_data,
                        origin_y=origin_y,
                        origin_z=origin_z,
                    )
                alt_summary[fg][lens][ke] = {
                    "global": {
                        "energy_resolution": er_mean,
                        "energy_resolution_mean": er_mean,
                        "energy_resolution_variance": er_var,
                        "energy_resolution_std": er_std,
                        "pair_count": pair_count,
                        "r_max_used": rmax_used,
                        "r_max_scope": rmax_scope,
                        "valid": er_mean is not None,
                        **metadata,
                    },
                    "local": {},
                }
    return alt_summary


def build_ln_r2_over_ln_r1_summary(
    summary_data,
    origin_y=DEFAULT_ORIGIN_Y,
    origin_z=DEFAULT_ORIGIN_Z,
):
    alt_summary = {}
    for fg, fg_data in summary_data.items():
        if not isinstance(fg_data, dict):
            continue
        alt_summary.setdefault(fg, {})
        for lens, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            alt_summary[fg].setdefault(lens, {})
            for ke, ke_data in lens_data.items():
                global_data = {}
                if isinstance(ke_data, dict):
                    global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    global_data = {}

                metadata = _build_plot_metadata(global_data)
                if _is_count_check_invalid(global_data):
                    er_mean, er_var, er_std, pair_count, skipped_non_positive, skipped_non_finite = (None, None, None, 0, 0, 0)
                else:
                    er_mean, er_var, er_std, pair_count, skipped_non_positive, skipped_non_finite = _compute_ln_r2_over_ln_r1_er(
                        global_data,
                        origin_y=origin_y,
                        origin_z=origin_z,
                    )
                alt_summary[fg][lens][ke] = {
                    "global": {
                        "energy_resolution": er_mean,
                        "energy_resolution_mean": er_mean,
                        "energy_resolution_variance": er_var,
                        "energy_resolution_std": er_std,
                        "pair_count": pair_count,
                        "skipped_non_positive_pairs": skipped_non_positive,
                        "skipped_non_finite_pairs": skipped_non_finite,
                        "valid": er_mean is not None,
                        **metadata,
                    },
                    "local": {},
                }
    return alt_summary


def build_abs_ra_rb_summary(
    summary_data,
    origin_y=DEFAULT_ORIGIN_Y,
    origin_z=DEFAULT_ORIGIN_Z,
):
    alt_summary = {}
    for fg, fg_data in summary_data.items():
        if not isinstance(fg_data, dict):
            continue
        alt_summary.setdefault(fg, {})
        for lens, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            alt_summary[fg].setdefault(lens, {})
            for ke, ke_data in lens_data.items():
                global_data = {}
                if isinstance(ke_data, dict):
                    global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    global_data = {}

                metadata = _build_plot_metadata(global_data)
                if _is_count_check_invalid(global_data):
                    er_mean, er_var, er_std, pair_count, skipped_negative = (None, None, None, 0, 0)
                else:
                    er_mean, er_var, er_std, pair_count, skipped_negative = _compute_abs_ra_rb_er(
                        global_data,
                        origin_y=origin_y,
                        origin_z=origin_z,
                    )
                alt_summary[fg][lens][ke] = {
                    "global": {
                        "energy_resolution": er_mean,
                        "energy_resolution_mean": er_mean,
                        "energy_resolution_variance": er_var,
                        "energy_resolution_std": er_std,
                        "pair_count": pair_count,
                        "skipped_negative_pairs": skipped_negative,
                        "valid": er_mean is not None,
                        **metadata,
                    },
                    "local": {},
                }
    return alt_summary


def summarize_comparison(summary_data, alt_summary):
    total = 0
    original_valid = 0
    alt_valid = 0
    both_valid = 0
    diffs = []
    rel_diffs = []

    for fg, fg_data in summary_data.items():
        if not isinstance(fg_data, dict):
            continue
        for lens, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            for ke, ke_data in lens_data.items():
                if not isinstance(ke_data, dict):
                    continue
                total += 1
                orig_g = ke_data.get("global", {})
                alt_g = alt_summary.get(fg, {}).get(lens, {}).get(ke, {}).get("global", {})
                orig_er = _extract_er(orig_g if isinstance(orig_g, dict) else {})
                alt_er = _extract_er(alt_g if isinstance(alt_g, dict) else {})
                if orig_er is not None:
                    original_valid += 1
                if alt_er is not None:
                    alt_valid += 1
                if orig_er is not None and alt_er is not None:
                    both_valid += 1
                    diff = alt_er - orig_er
                    diffs.append(diff)
                    if abs(orig_er) > 1e-12:
                        rel_diffs.append(diff / orig_er)

    summary = {
        "total": total,
        "original_valid": original_valid,
        "alt_valid": alt_valid,
        "both_valid": both_valid,
        "original_only": max(0, original_valid - both_valid),
        "alt_only": max(0, alt_valid - both_valid),
    }
    if diffs:
        arr = np.array(diffs, dtype=float)
        summary.update(
            {
                "diff_mean": float(np.mean(arr)),
                "diff_std": float(np.std(arr)),
                "diff_min": float(np.min(arr)),
                "diff_max": float(np.max(arr)),
                "abs_diff_mean": float(np.mean(np.abs(arr))),
            }
        )
    else:
        summary.update(
            {
                "diff_mean": None,
                "diff_std": None,
                "diff_min": None,
                "diff_max": None,
                "abs_diff_mean": None,
            }
        )
    if rel_diffs:
        rel = np.array(rel_diffs, dtype=float)
        summary["rel_diff_mean"] = float(np.mean(rel))
        summary["rel_abs_diff_mean"] = float(np.mean(np.abs(rel)))
    else:
        summary["rel_diff_mean"] = None
        summary["rel_abs_diff_mean"] = None
    return summary


def count_pair_coordinate_coverage(summary_data):
    total_pairs = 0
    pairs_with_coords = 0
    for fg_data in summary_data.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if not isinstance(ke_data, dict):
                    continue
                global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    continue
                pairs = global_data.get("dr_over_r_pairs", [])
                if isinstance(pairs, np.ndarray):
                    arr = np.asarray(pairs)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    if arr.ndim == 2:
                        total_pairs += int(arr.shape[0])
                        if arr.shape[1] >= 8:
                            pairs_with_coords += int(arr.shape[0])
                    continue
                if isinstance(pairs, list):
                    for pair in pairs:
                        total_pairs += 1
                        if isinstance(pair, dict):
                            if (
                                _safe_float(pair.get("y_a")) is not None
                                and _safe_float(pair.get("z_a")) is not None
                                and _safe_float(pair.get("y_b")) is not None
                                and _safe_float(pair.get("z_b")) is not None
                            ):
                                pairs_with_coords += 1
                        elif isinstance(pair, (list, tuple, np.ndarray)) and len(pair) >= 8:
                            if (
                                _safe_float(pair[4]) is not None
                                and _safe_float(pair[5]) is not None
                                and _safe_float(pair[6]) is not None
                                and _safe_float(pair[7]) is not None
                            ):
                                pairs_with_coords += 1
    return total_pairs, pairs_with_coords


def count_raw_point_coverage(summary_data):
    total_nodes = 0
    nodes_with_raw = 0
    total_raw_points = 0
    for fg_data in summary_data.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if not isinstance(ke_data, dict):
                    continue
                total_nodes += 1
                global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    continue
                raw_points = _extract_raw_ion_points_yz(global_data)
                if raw_points:
                    nodes_with_raw += 1
                    total_raw_points += len(raw_points)
    return total_nodes, nodes_with_raw, total_raw_points


def count_rmax_all_points_coverage(summary_data):
    total_nodes = 0
    nodes_with_rmax = 0
    for fg_data in summary_data.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if not isinstance(ke_data, dict):
                    continue
                total_nodes += 1
                global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    continue
                rmax = _safe_float(global_data.get("r_max_all_points"))
                if rmax is not None and rmax > 0:
                    nodes_with_rmax += 1
    return total_nodes, nodes_with_rmax


def _build_method_entry(method_key, summary_data, origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    if method_key == "rmax":
        return {
            "key": "rmax",
            "label": "metric = |r2-r1| / rmax",
            "summary": build_abs_ra_rb_over_rmax_summary(summary_data, origin_y=origin_y, origin_z=origin_z),
            "color": "tab:blue",
            "marker": "o",
        }
    if method_key == "abs":
        if not ENABLE_METHOD_ABS:
            return {
                "key": "abs",
                "label": "metric = |r2-r1| / rmean (disabled; showing saved)",
                "summary": summary_data,
                "color": "tab:orange",
                "marker": "s",
            }
        return {
            "key": "abs",
            "label": "metric = |r2-r1| / rmean",
            "summary": build_abs_ra_rb_summary(summary_data, origin_y=origin_y, origin_z=origin_z),
            "color": "tab:orange",
            "marker": "s",
        }
    if method_key == "sqrt":
        if not ENABLE_METHOD_SQRT:
            return {
                "key": "sqrt",
                "label": "metric = sqrt(ra^2-rb^2)/rmean (disabled; showing saved)",
                "summary": summary_data,
                "color": "tab:green",
                "marker": "^",
            }
        return {
            "key": "sqrt",
            "label": "dr = sqrt(ra^2-rb^2) / r_mean(pair)",
            "summary": build_sqrt_ra2_minus_rb2_summary(summary_data, origin_y=origin_y, origin_z=origin_z),
            "color": "tab:green",
            "marker": "^",
        }
    if method_key == "ln_ratio":
        if not ENABLE_METHOD_LN_RATIO:
            return {
                "key": "ln_ratio",
                "label": "metric = |ln(|r2|/|r1|)| (disabled; showing saved)",
                "summary": summary_data,
                "color": "tab:purple",
                "marker": "D",
            }
        return {
            "key": "ln_ratio",
            "label": "metric = |ln(|r2|/|r1|)|, |r2|>|r1|",
            "summary": build_ln_r2_over_ln_r1_summary(summary_data, origin_y=origin_y, origin_z=origin_z),
            "color": "tab:purple",
            "marker": "D",
        }
    raise ValueError(f"Unsupported method: {method_key}")


def launch_multi_method_gui(
    method_entries,
    title_prefix="Energy Resolution Compare",
    y_as_percent=True,
    xlim=None,
    ylim=None,
    group_options=None,
    default_group=None,
):
    if not method_entries:
        return False

    fg_set = set()
    for entry in method_entries:
        fg_set.update(entry["summary"].keys())
    fg_keys = _sorted_keys(fg_set)
    if not fg_keys:
        return False

    scale = 100.0 if y_as_percent else 1.0
    y_label = "Energy Resolution (%)" if y_as_percent else "Energy Resolution (dE/E)"

    initial_fg = fg_keys[0]
    lens_set = set()
    for entry in method_entries:
        lens_set.update(entry["summary"].get(initial_fg, {}).keys())
    initial_lens_keys = _sorted_keys(lens_set)
    if not initial_lens_keys:
        return False

    fig, ax_curve = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.25)
    default_view = {
        "xlim": _normalize_axis_limits(xlim),
        "ylim": _normalize_axis_limits(ylim),
    }
    view_state = {
        "xlim": default_view["xlim"],
        "ylim": default_view["ylim"],
    }
    group_label_map = {}
    if group_options:
        for item in group_options:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            label = item.get("label", key)
            if key is None:
                continue
            group_label_map[str(label)] = key
    active_group = {"key": None}
    if group_label_map:
        if default_group is not None and any(val == default_group for val in group_label_map.values()):
            active_group["key"] = default_group
        else:
            active_group["key"] = next(iter(group_label_map.values()))

    ax_fg_slider = plt.axes([0.2, 0.10, 0.6, 0.03])
    fg_slider = Slider(
        ax=ax_fg_slider,
        label="Field Gradient",
        valmin=0,
        valmax=max(0.1, len(fg_keys) - 1),
        valinit=0,
        valstep=1,
    )

    ax_lens_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    lens_slider = Slider(
        ax=ax_lens_slider,
        label="Lens VMI",
        valmin=0,
        valmax=max(0.1, len(initial_lens_keys) - 1),
        valinit=0,
        valstep=1,
    )

    group_radio = None
    if group_label_map and len(group_label_map) >= 2:
        ax_group_radio = plt.axes([0.02, 0.05, 0.15, 0.18])
        radio_labels = list(group_label_map.keys())
        active_idx = 0
        if active_group["key"] is not None:
            for idx, label in enumerate(radio_labels):
                if group_label_map[label] == active_group["key"]:
                    active_idx = idx
                    break
        group_radio = RadioButtons(ax_group_radio, radio_labels, active=active_idx)
        ax_group_radio.set_title("Method", fontsize=9)

    def update(_=None):
        ax_curve.clear()

        fg_idx = int(fg_slider.val)
        fg_idx = min(max(0, fg_idx), len(fg_keys) - 1)
        fg = fg_keys[fg_idx]
        fg_slider.valtext.set_text(f"{_fmt_num(fg, digits=2)}")

        lens_set_current = set()
        for entry in method_entries:
            lens_set_current.update(entry["summary"].get(fg, {}).keys())
        lens_keys = _sorted_keys(lens_set_current)
        if not lens_keys:
            ax_curve.text(0.5, 0.5, "No lens data", ha="center", va="center", transform=ax_curve.transAxes)
            fig.canvas.draw_idle()
            return

        lens_slider.valmax = max(0.1, len(lens_keys) - 1)
        lens_slider.ax.set_xlim(lens_slider.valmin, lens_slider.valmax)
        lens_idx = int(lens_slider.val)
        lens_idx = min(max(0, lens_idx), len(lens_keys) - 1)
        lens = lens_keys[lens_idx]
        lens_slider.valtext.set_text(_fmt_num(lens, digits=3))

        ke_set = set()
        for entry in method_entries:
            ke_set.update(entry["summary"].get(fg, {}).get(lens, {}).keys())
        ke_keys = _sorted_keys(ke_set)
        if not ke_keys:
            ax_curve.text(0.5, 0.5, "No KE data", ha="center", va="center", transform=ax_curve.transAxes)
            fig.canvas.draw_idle()
            return

        ke_arr = np.array(
            [float(_safe_float(k) if _safe_float(k) is not None else np.nan) for k in ke_keys],
            dtype=float,
        )
        valid_ke = np.isfinite(ke_arr)

        legend_needed = False
        point_counts = []
        invalid_counts = []
        invalid_ke_by_method = []

        for method_idx, entry in enumerate(method_entries):
            entry_group = entry.get("group")
            if active_group["key"] is not None and entry_group is not None and entry_group != active_group["key"]:
                continue
            vals = []
            stds = []
            invalid_flags = []
            summary = entry["summary"]
            for ke in ke_keys:
                global_data = (
                    summary.get(fg, {})
                    .get(lens, {})
                    .get(ke, {})
                    .get("global", {})
                )
                er_val = _extract_er(global_data if isinstance(global_data, dict) else {})
                er_std = _extract_er_std(global_data if isinstance(global_data, dict) else {})
                vals.append(np.nan if er_val is None else er_val * scale)
                stds.append(np.nan if er_std is None else er_std * scale)
                invalid_flags.append(_is_count_check_invalid(global_data if isinstance(global_data, dict) else {}))

            val_arr = np.array(vals, dtype=float)
            std_arr = np.array(stds, dtype=float)
            valid_mask = valid_ke & np.isfinite(val_arr)
            invalid_mask = valid_ke & (~np.isfinite(val_arr)) & np.array(invalid_flags, dtype=bool)
            point_counts.append(int(np.sum(valid_mask)))
            invalid_counts.append(int(np.sum(invalid_mask)))
            invalid_ke_by_method.append(ke_arr[invalid_mask])

            if np.any(valid_mask):
                legend_needed = True
                ax_curve.plot(
                    ke_arr[valid_mask],
                    val_arr[valid_mask],
                    f"{entry['marker']}-",
                    color=entry["color"],
                    linewidth=2,
                    markersize=5,
                    label=entry["label"],
                )
                std_mask = valid_mask & np.isfinite(std_arr)
                if np.any(std_mask):
                    ax_curve.errorbar(
                        ke_arr[std_mask],
                        val_arr[std_mask],
                        yerr=std_arr[std_mask],
                        fmt="none",
                        ecolor=entry["color"],
                        alpha=0.35,
                        capsize=2,
                    )

        total_invalid = int(np.sum(np.array(invalid_counts, dtype=int))) if invalid_counts else 0
        if total_invalid > 0:
            y_min, y_max = ax_curve.get_ylim()
            y_span = y_max - y_min
            if not np.isfinite(y_span) or y_span <= 0:
                y_min = 0.0
                y_span = 1.0
                y_max = y_min + y_span
            cross_y = y_min + 0.08 * y_span
            for invalid_ke_values in invalid_ke_by_method:
                if invalid_ke_values.size <= 0:
                    continue
                ax_curve.scatter(
                    invalid_ke_values,
                    np.full(invalid_ke_values.shape, cross_y, dtype=float),
                    marker="x",
                    color="red",
                    s=50,
                    linewidths=1.5,
                    zorder=6,
                )

        ax_curve.set_xlabel("Kinetic Energy (eV)")
        ax_curve.set_ylabel(y_label)
        ax_curve.set_title(
            f"{title_prefix}\nFG={_fmt_num(fg, 2)}  Lens={_fmt_num(lens, 3)}",
            fontsize=12,
        )
        ax_curve.grid(True, alpha=0.3)
        if legend_needed:
            ax_curve.legend(loc="best", fontsize=9)

        point_text_parts = []
        for idx, count in enumerate(point_counts):
            point_text_parts.append(f"M{idx + 1}={count}")
        if invalid_counts:
            for idx, count in enumerate(invalid_counts):
                point_text_parts.append(f"X{idx + 1}={count}")
        ax_curve.text(
            0.02,
            0.98,
            "points: " + ", ".join(point_text_parts),
            transform=ax_curve.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"),
        )
        ax_curve.text(
            0.02,
            0.03,
            "Wheel: zoom | +/-: zoom | arrows: pan | r: reset",
            transform=ax_curve.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.60, edgecolor="gray"),
        )

        if view_state["xlim"] is not None:
            ax_curve.set_xlim(view_state["xlim"])
        if view_state["ylim"] is not None:
            ax_curve.set_ylim(view_state["ylim"])

        fig.canvas.draw_idle()

    def _apply_zoom(scale, x_center=None, y_center=None):
        cur_xlim = view_state["xlim"] if view_state["xlim"] is not None else tuple(ax_curve.get_xlim())
        cur_ylim = view_state["ylim"] if view_state["ylim"] is not None else tuple(ax_curve.get_ylim())

        x_mid = 0.5 * (cur_xlim[0] + cur_xlim[1]) if x_center is None else x_center
        y_mid = 0.5 * (cur_ylim[0] + cur_ylim[1]) if y_center is None else y_center
        new_xlim = _zoom_limits(cur_xlim, x_mid, scale)
        new_ylim = _zoom_limits(cur_ylim, y_mid, scale)
        if new_xlim is None or new_ylim is None:
            return
        view_state["xlim"] = new_xlim
        view_state["ylim"] = new_ylim
        ax_curve.set_xlim(new_xlim)
        ax_curve.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def _apply_pan(dx_fraction=0.0, dy_fraction=0.0):
        cur_xlim = view_state["xlim"] if view_state["xlim"] is not None else tuple(ax_curve.get_xlim())
        cur_ylim = view_state["ylim"] if view_state["ylim"] is not None else tuple(ax_curve.get_ylim())
        x_span = cur_xlim[1] - cur_xlim[0]
        y_span = cur_ylim[1] - cur_ylim[0]
        new_xlim = _pan_limits(cur_xlim, x_span * float(dx_fraction))
        new_ylim = _pan_limits(cur_ylim, y_span * float(dy_fraction))
        if new_xlim is None or new_ylim is None:
            return
        view_state["xlim"] = new_xlim
        view_state["ylim"] = new_ylim
        ax_curve.set_xlim(new_xlim)
        ax_curve.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes != ax_curve:
            return
        button = str(event.button).lower()
        if button == "up":
            scale = 0.85
        elif button == "down":
            scale = 1.15
        else:
            return
        _apply_zoom(scale, x_center=event.xdata, y_center=event.ydata)

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("r", "home"):
            view_state["xlim"] = default_view["xlim"]
            view_state["ylim"] = default_view["ylim"]
            update()
            return
        if key in ("+", "="):
            _apply_zoom(0.85)
            return
        if key in ("-", "_"):
            _apply_zoom(1.15)
            return
        if key == "left":
            _apply_pan(dx_fraction=-0.10)
            return
        if key == "right":
            _apply_pan(dx_fraction=0.10)
            return
        if key == "up":
            _apply_pan(dy_fraction=0.10)
            return
        if key == "down":
            _apply_pan(dy_fraction=-0.10)
            return

    def on_group_select(label):
        selected_key = group_label_map.get(str(label))
        if selected_key is None:
            return
        active_group["key"] = selected_key
        update()

    fg_slider.on_changed(update)
    lens_slider.on_changed(update)
    if group_radio is not None:
        group_radio.on_clicked(on_group_select)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()
    return True


def launch_comparison_gui(
    summary_data,
    alt_summary,
    title_prefix="Energy Resolution Compare",
    y_as_percent=True,
    method_a_label="Method A",
    method_b_label="Method B",
    xlim=None,
    ylim=None,
):
    method_entries = [
        {
            "key": "a",
            "label": method_a_label,
            "summary": summary_data,
            "color": "tab:blue",
            "marker": "o",
        },
        {
            "key": "b",
            "label": method_b_label,
            "summary": alt_summary,
            "color": "tab:orange",
            "marker": "s",
        },
    ]
    return launch_multi_method_gui(
        method_entries,
        title_prefix=title_prefix,
        y_as_percent=y_as_percent,
        xlim=xlim,
        ylim=ylim,
    )


def _count_rmax_scope(metric_summary):
    scope_counter = {
        "all_points": 0,
        "all_points_list": 0,
        "raw_points": 0,
        "pairs_fallback": 0,
        "missing": 0,
    }
    for fg_data in metric_summary.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if not isinstance(ke_data, dict):
                    continue
                global_data = ke_data.get("global", {})
                if not isinstance(global_data, dict):
                    continue
                scope = str(global_data.get("r_max_scope", "missing"))
                if scope not in scope_counter:
                    scope = "missing"
                scope_counter[scope] += 1
    return scope_counter


def _resolve_method_keys(method):
    if method == "all":
        return ["rmax", "sqrt"]
    if method in ("rmax_vs_abs_legacy", "compare"):
        return ["rmax", "abs"]
    return [method]


def build_method_entries(summary_data, method="rmax", origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    method_keys = _resolve_method_keys(method)
    method_entries = []
    for idx, method_key in enumerate(method_keys):
        entry = _build_method_entry(
            method_key,
            summary_data,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        entry["id"] = f"M{idx + 1}"
        method_entries.append(entry)
    return method_entries


def launch_energy_resolution_gui(
    summary_data,
    method="rmax",
    origin_y=DEFAULT_ORIGIN_Y,
    origin_z=DEFAULT_ORIGIN_Z,
    y_as_percent=True,
    title_prefix="Result Viewer",
    xlim=None,
    ylim=None,
):
    method_entries = build_method_entries(
        summary_data,
        method=method,
        origin_y=origin_y,
        origin_z=origin_z,
    )
    return launch_multi_method_gui(
        method_entries,
        title_prefix=title_prefix,
        y_as_percent=y_as_percent,
        xlim=xlim,
        ylim=ylim,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Standalone viewer for saved SIMION energy-resolution results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to summary pickle/csv. If omitted, loads latest file in --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_v5.3",
        help="Directory containing saved result files.",
    )
    parser.add_argument(
        "--method",
        choices=("rmax", "sqrt", "all"),
        default="rmax",
        help=(
            "Method to display: "
            "rmax=|ra-rb|/rmax(all points), sqrt=sqrt(ra^2-rb^2)/r_mean(pair). "
            "Current build disables sqrt recomputation (falls back to saved metric)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("compare", "single"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--origin-x",
        type=float,
        default=DEFAULT_ORIGIN_X,
        help="Origin x for documentation (not used in yz-only radius).",
    )
    parser.add_argument(
        "--origin-y",
        type=float,
        default=DEFAULT_ORIGIN_Y,
        help="Origin y used to compute ra/rb in yz plane.",
    )
    parser.add_argument(
        "--origin-z",
        type=float,
        default=DEFAULT_ORIGIN_Z,
        help="Origin z used to compute ra/rb in yz plane.",
    )
    parser.add_argument(
        "--fraction",
        action="store_true",
        help="Display y-axis as fractional dE/E instead of percent.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Only print summary stats and exit.",
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default=None,
        help="Initial x-axis limits as 'min,max' in eV.",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default=None,
        help="Initial y-axis limits as 'min,max' (percent unless --fraction).",
    )
    args = parser.parse_args()
    xlim = _parse_axis_range(args.xlim, "xlim")
    ylim = _parse_axis_range(args.ylim, "ylim")

    target_file = args.input
    if not target_file:
        target_file = find_latest_summary_file(args.results_dir)
        if not target_file:
            raise FileNotFoundError(f"No summary files found in '{args.results_dir}'.")

    summary_data = load_summary_from_file(target_file)
    valid_points, total_points = count_valid_energy_points(summary_data)

    print("=" * 70)
    print("Energy Resolution Result Viewer")
    print("=" * 70)
    print(f"Source file: {os.path.abspath(target_file)}")
    print(f"Field gradients: {len(summary_data)}")
    print(
        "Radius origin (yz plane): "
        f"({args.origin_x:.6g}, {args.origin_y:.6g}, {args.origin_z:.6g})"
    )
    if xlim is not None:
        print(f"Initial xlim: [{xlim[0]:.6g}, {xlim[1]:.6g}]")
    if ylim is not None:
        print(f"Initial ylim: [{ylim[0]:.6g}, {ylim[1]:.6g}]")
    print(f"Source valid points (saved metric): {valid_points}/{total_points}")
    total_pairs, pairs_with_coords = count_pair_coordinate_coverage(summary_data)
    if total_pairs > 0:
        print(f"Pair coord coverage (y_a/z_a/y_b/z_b): {pairs_with_coords}/{total_pairs}")
        if pairs_with_coords < total_pairs:
            print(
                "WARNING: Some pairs do not contain per-ion coordinates; "
                "origin-shifted ra/rb uses fallback r_a/r_b for those pairs."
            )
    total_nodes_raw, nodes_with_raw, total_raw_points = count_raw_point_coverage(summary_data)
    if total_nodes_raw > 0:
        print(f"Raw ion y/z coverage: {nodes_with_raw}/{total_nodes_raw} nodes, points={total_raw_points}")
        if nodes_with_raw == 0:
            print("NOTE: This summary file does not contain raw ion payload (legacy metric-only file).")
    total_nodes, nodes_with_rmax = count_rmax_all_points_coverage(summary_data)
    if total_nodes > 0:
        print(f"rmax(all points) coverage: {nodes_with_rmax}/{total_nodes}")
        if nodes_with_rmax < total_nodes:
            print(
                "WARNING: Some nodes do not have r_max_all_points; "
                "viewer will fallback to raw-point max-r or paired-point max-r for those nodes."
            )

    method_arg_present = any(
        arg == "--method" or arg.startswith("--method=") for arg in sys.argv[1:]
    )
    selected_method = args.method
    if args.mode and not method_arg_present:
        # Backward compatibility: keep old --mode behavior when --method is not provided.
        if args.mode == "compare":
            selected_method = "rmax_vs_abs_legacy"
        else:
            selected_method = "rmax"

    method_entries = build_method_entries(
        summary_data,
        method=selected_method,
        origin_y=args.origin_y,
        origin_z=args.origin_z,
    )

    print("-" * 70)
    print("Display methods:")
    for entry in method_entries:
        print(f"{entry['id']}: {entry['label']}")
        metric_valid, metric_total = count_valid_energy_points(entry["summary"])
        print(f"  valid points: {metric_valid}/{metric_total}")
        if entry["key"] == "rmax":
            rmax_scope_counter = _count_rmax_scope(entry["summary"])
            print(
                "  rmax source counts: "
                f"all_points={rmax_scope_counter['all_points']}, "
                f"all_points_list={rmax_scope_counter['all_points_list']}, "
                f"raw_points={rmax_scope_counter['raw_points']}, "
                f"pairs_fallback={rmax_scope_counter['pairs_fallback']}, "
                f"missing={rmax_scope_counter['missing']}"
            )

    if len(method_entries) >= 2:
        print("-" * 70)
        for i in range(len(method_entries)):
            for j in range(i + 1, len(method_entries)):
                left = method_entries[i]
                right = method_entries[j]
                compare_stats = summarize_comparison(left["summary"], right["summary"])
                print(f"Delta({right['id']}-{left['id']}): {right['label']} minus {left['label']}")
                print(f"  overlap valid points: {compare_stats['both_valid']}")
                if compare_stats["diff_mean"] is not None:
                    print(
                        f"  mean={compare_stats['diff_mean']:.6f}, "
                        f"std={compare_stats['diff_std']:.6f}, "
                        f"min={compare_stats['diff_min']:.6f}, "
                        f"max={compare_stats['diff_max']:.6f}"
                    )
                    if compare_stats["rel_diff_mean"] is not None:
                        print(
                            f"  relative mean={compare_stats['rel_diff_mean'] * 100:.3f}%, "
                            f"relative |mean|={compare_stats['rel_abs_diff_mean'] * 100:.3f}%"
                        )

    if args.no_gui:
        return

    launched = launch_multi_method_gui(
        method_entries,
        title_prefix=f"Result Viewer: {os.path.basename(target_file)}",
        y_as_percent=not args.fraction,
        xlim=xlim,
        ylim=ylim,
    )
    if not launched:
        raise RuntimeError("Viewer could not find plottable data.")


if __name__ == "__main__":
    main()
