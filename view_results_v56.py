import argparse
import csv
import glob
import importlib.util
import os
import pickle
import sys
import numpy as np

# Prefer interactive backends for standalone GUI viewer.
def _is_debug_session():
    try:
        if sys.gettrace() is not None:
            return True
    except Exception:
        pass
    return any(name in sys.modules for name in ("debugpy", "pydevd", "pydevd_frame_eval"))


def _has_qt_binding():
    for module_name in ("PyQt6", "PySide6", "PyQt5", "PySide2"):
        if importlib.util.find_spec(module_name) is not None:
            return True
    return False


def _has_tk_binding():
    return importlib.util.find_spec("tkinter") is not None


def _backend_dependencies_available(backend_name):
    key = str(backend_name or "").strip().lower()
    if key in {"qtagg", "qt5agg", "qtcairo"}:
        return _has_qt_binding()
    if key in {"tkagg", "tkcairo"}:
        return _has_tk_binding()
    return True


def _preferred_interactive_backends():
    # TkAgg can trigger Tcl_AsyncDelete crashes under debugpy on Windows.
    has_qt = _has_qt_binding()
    has_tk = _has_tk_binding()
    backends = []
    if has_qt:
        backends.extend(["qtagg", "qt5agg"])
    if has_tk:
        # Prefer avoiding Tk under debug when Qt is available.
        if not (_is_debug_session() and os.name == "nt" and has_qt):
            backends.append("tkagg")
    return tuple(backends)


def _try_switch_backend(target_backend):
    backend_name = str(target_backend or "").strip().lower()
    if not backend_name:
        return False
    if not _backend_dependencies_available(backend_name):
        return False
    try:
        import matplotlib
        matplotlib.use(backend_name, force=True)
    except Exception:
        return False
    try:
        import matplotlib.pyplot as _plt
        _plt.switch_backend(backend_name)
    except Exception:
        pass
    os.environ["MPLBACKEND"] = backend_name
    return True


def _configure_matplotlib_backend():
    try:
        import matplotlib
    except Exception:
        return "unknown"

    forced_backend = str(os.environ.get("SIMION_MPL_BACKEND", "")).strip().lower()
    if forced_backend:
        if _backend_dependencies_available(forced_backend):
            if _try_switch_backend(forced_backend):
                return forced_backend
        else:
            print(
                f"WARNING: SIMION_MPL_BACKEND='{forced_backend}' ignored; "
                f"required GUI binding is not available."
            )

    preferred_backends = _preferred_interactive_backends()
    for backend_name in preferred_backends:
        if _try_switch_backend(backend_name):
            return backend_name

    try:
        matplotlib.use("Agg", force=True)
        os.environ["MPLBACKEND"] = "Agg"
        return "Agg"
    except Exception:
        return "unknown"

MATPLOTLIB_BACKEND_SELECTED = _configure_matplotlib_backend()
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

METHOD_SELECTOR_OPTIONS = [
    ("rmax", "|r2-r1|/rmax"),
]
METHOD_CHOICES = ("selector", "saved", "rmax", "sqrt", "all")
SELECTOR_METHOD_COLORS = {
    "rmax": "tab:blue",
}
SELECTOR_METHOD_MARKERS = {
    "rmax": "o",
}

MAX_OVERLAY_RANGES_DEFAULT = 12
LARGE_BUNDLE_RANGE_THRESHOLD = 10
LARGE_BUNDLE_TOTAL_POINTS_THRESHOLD = 2000
HEAVY_PAYLOAD_GLOBAL_KEYS = ("dr_over_r_pairs", "raw_ion_points_yz", "all_r_values")

DEFAULT_ORIGIN_Y = -1.0
DEFAULT_ORIGIN_Z = 0.0

MAX_PLOT_POINTS_PER_CURVE = 800
MAX_INVALID_MARKERS_PER_CURVE = 300
SERIES_CACHE_LIMIT = 4096


def _normalize_method(selected_method):
    method_key = str(selected_method or "rmax").lower()
    if method_key not in METHOD_CHOICES:
        raise ValueError(
            f"Unsupported method '{selected_method}'. "
            f"Choose from: {', '.join(METHOD_CHOICES)}"
        )
    return method_key


def _method_text_for_title(selected_method):
    method_key = _normalize_method(selected_method)
    if method_key == "saved":
        return "saved energy_resolution"
    if method_key in ("selector", "rmax"):
        return "|r2-r1|/rmax"
    if method_key == "sqrt":
        return "sqrt(ra^2-rb^2)/r_mean(pair) [saved fallback]"
    if method_key == "all":
        return "|r2-r1|/rmax + sqrt(ra^2-rb^2)/r_mean(pair) [saved fallback]"
    return method_key


def _build_title_prefix(base_title, selected_method):
    return f"{base_title}\nMethod: {_method_text_for_title(selected_method)}"


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sorted_numeric_keys(values):
    numeric_items = []
    other_items = []
    for key in values:
        num = _safe_float(key)
        if num is None:
            other_items.append(key)
        else:
            numeric_items.append((num, key))
    numeric_items.sort(key=lambda item: item[0])
    return [item[1] for item in numeric_items] + sorted(other_items, key=str)


def _sorted_keys(values):
    return _sorted_numeric_keys(values)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _fmt_num(value, digits=3):
    num = _safe_float(value)
    if num is None:
        return str(value)
    return f"{num:.{digits}f}"


def _extract_er(global_data):
    if not isinstance(global_data, dict):
        return None
    return _safe_float(global_data.get("energy_resolution_mean", global_data.get("energy_resolution")))


def _extract_er_std(global_data):
    if not isinstance(global_data, dict):
        return None
    er_std = _safe_float(global_data.get("energy_resolution_std"))
    if er_std is not None:
        return er_std
    er_var = _safe_float(global_data.get("energy_resolution_variance"))
    if er_var is not None:
        return float(np.sqrt(max(er_var, 0.0)))
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


def _evenly_spaced_indices(length, max_points):
    if length <= 0:
        return np.array([], dtype=int)
    if max_points is None:
        return np.arange(length, dtype=int)
    try:
        max_points = int(max_points)
    except (TypeError, ValueError):
        return np.arange(length, dtype=int)
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, num=max_points, dtype=int))


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
    if not isinstance(summary_data, dict):
        return valid, total
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
                if _extract_er(global_data) is not None:
                    valid += 1
    return valid, total


def _resolve_method_keys(method):
    if method == "all":
        return ["rmax", "sqrt"]
    if method == "selector":
        return ["rmax"]
    return [method]


def build_method_entries(summary_data, method="rmax", origin_y=DEFAULT_ORIGIN_Y, origin_z=DEFAULT_ORIGIN_Z):
    del origin_y, origin_z
    method_keys = _resolve_method_keys(str(method or "rmax").lower())
    method_entries = []
    labels = {
        "saved": "saved energy_resolution",
        "rmax": "|r2-r1|/rmax (using saved energy_resolution)",
        "sqrt": "sqrt(ra^2-rb^2)/r_mean(pair) [saved fallback]",
    }
    colors = {"saved": "tab:blue", "rmax": "tab:blue", "sqrt": "tab:green"}
    markers = {"saved": "o", "rmax": "o", "sqrt": "^"}
    for idx, method_key in enumerate(method_keys):
        key = method_key if method_key in labels else "saved"
        method_entries.append(
            {
                "key": key,
                "label": labels.get(key, labels["saved"]),
                "summary": summary_data,
                "color": colors.get(key, "tab:blue"),
                "marker": markers.get(key, "o"),
                "id": f"M{idx + 1}",
            }
        )
    return method_entries


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
        summary = entry.get("summary", {})
        if isinstance(summary, dict):
            fg_set.update(summary.keys())
    fg_keys = _sorted_keys(fg_set)
    if not fg_keys:
        return False

    scale = 100.0 if y_as_percent else 1.0
    y_label = "Energy Resolution (%)" if y_as_percent else "Energy Resolution (dE/E)"

    initial_fg = fg_keys[0]
    lens_set = set()
    for entry in method_entries:
        summary = entry.get("summary", {})
        if isinstance(summary, dict):
            lens_set.update(summary.get(initial_fg, {}).keys())
    initial_lens_keys = _sorted_keys(lens_set)
    if not initial_lens_keys:
        return False

    try:
        fig, ax_curve = plt.subplots(figsize=(14, 10))
    except ImportError as exc:
        message = str(exc)
        lower = message.lower()
        qt_failed = (
            "failed to import any of the following qt binding modules" in lower
            or "could not load requested qt binding" in lower
        )
        if not qt_failed:
            raise

        fallback_backends = []
        if _has_tk_binding():
            fallback_backends.append("tkagg")
        fallback_backends.append("agg")

        switched_to = None
        for fallback in fallback_backends:
            if _try_switch_backend(fallback):
                switched_to = fallback
                break

        if switched_to is None:
            raise

        print(
            f"WARNING: Qt backend failed ({message}). "
            f"Switched to '{switched_to}'."
        )
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

    series_cache = {}

    def _build_series_payload(entry_idx, entry, fg, lens):
        cache_key = (entry_idx, fg, lens)
        if cache_key in series_cache:
            return series_cache[cache_key]
        if len(series_cache) >= SERIES_CACHE_LIMIT:
            series_cache.clear()

        summary = entry.get("summary", {})
        lens_data = {}
        if isinstance(summary, dict):
            lens_data = summary.get(fg, {}).get(lens, {})
        if not isinstance(lens_data, dict) or not lens_data:
            payload = {
                "ke": np.array([], dtype=float),
                "val": np.array([], dtype=float),
                "std": np.array([], dtype=float),
                "valid_mask": np.array([], dtype=bool),
                "invalid_mask": np.array([], dtype=bool),
            }
            series_cache[cache_key] = payload
            return payload

        ke_keys = _sorted_keys(lens_data.keys())
        ke_vals = []
        val_vals = []
        std_vals = []
        invalid_flags = []
        for ke in ke_keys:
            ke_num = _safe_float(ke)
            if ke_num is None:
                continue
            node = lens_data.get(ke, {})
            global_data = node.get("global", {}) if isinstance(node, dict) else {}
            if not isinstance(global_data, dict):
                global_data = {}
            er_val = _extract_er(global_data)
            er_std = _extract_er_std(global_data)
            ke_vals.append(float(ke_num))
            val_vals.append(np.nan if er_val is None else float(er_val) * scale)
            std_vals.append(np.nan if er_std is None else float(er_std) * scale)
            invalid_flags.append(_is_count_check_invalid(global_data))

        ke_arr = np.array(ke_vals, dtype=float)
        val_arr = np.array(val_vals, dtype=float)
        std_arr = np.array(std_vals, dtype=float)
        invalid_flag_arr = np.array(invalid_flags, dtype=bool)
        finite_ke = np.isfinite(ke_arr)
        payload = {
            "ke": ke_arr,
            "val": val_arr,
            "std": std_arr,
            "valid_mask": finite_ke & np.isfinite(val_arr),
            "invalid_mask": finite_ke & (~np.isfinite(val_arr)) & invalid_flag_arr,
        }
        series_cache[cache_key] = payload
        return payload

    def update(_=None):
        ax_curve.clear()

        fg_idx = int(fg_slider.val)
        fg_idx = min(max(0, fg_idx), len(fg_keys) - 1)
        fg = fg_keys[fg_idx]
        fg_slider.valtext.set_text(f"{_fmt_num(fg, digits=2)}")

        lens_set_current = set()
        for entry in method_entries:
            summary = entry.get("summary", {})
            if isinstance(summary, dict):
                lens_set_current.update(summary.get(fg, {}).keys())
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

        legend_needed = False
        point_counts = []
        invalid_counts = []
        invalid_ke_by_method = []
        downsampled_valid_total = 0
        downsampled_invalid_total = 0

        for method_idx, entry in enumerate(method_entries):
            entry_group = entry.get("group")
            if active_group["key"] is not None and entry_group is not None and entry_group != active_group["key"]:
                continue
            payload = _build_series_payload(method_idx, entry, fg, lens)
            ke_arr = payload["ke"]
            val_arr = payload["val"]
            std_arr = payload["std"]
            valid_mask = payload["valid_mask"]
            invalid_mask = payload["invalid_mask"]
            if ke_arr.size == 0:
                point_counts.append(0)
                invalid_counts.append(0)
                invalid_ke_by_method.append(np.array([], dtype=float))
                continue

            point_counts.append(int(np.sum(valid_mask)))
            invalid_counts.append(int(np.sum(invalid_mask)))
            invalid_ke_values = ke_arr[invalid_mask]
            if invalid_ke_values.size > MAX_INVALID_MARKERS_PER_CURVE:
                invalid_idx = _evenly_spaced_indices(
                    int(invalid_ke_values.size), MAX_INVALID_MARKERS_PER_CURVE
                )
                invalid_ke_values = invalid_ke_values[invalid_idx]
            invalid_ke_by_method.append(invalid_ke_values)
            downsampled_invalid_total += max(
                0, int(np.sum(invalid_mask)) - int(invalid_ke_values.size)
            )

            if np.any(valid_mask):
                legend_needed = True
                ke_valid = ke_arr[valid_mask]
                val_valid = val_arr[valid_mask]
                std_valid = std_arr[valid_mask]
                valid_count = int(ke_valid.size)
                if valid_count > MAX_PLOT_POINTS_PER_CURVE:
                    valid_idx = _evenly_spaced_indices(valid_count, MAX_PLOT_POINTS_PER_CURVE)
                    ke_valid = ke_valid[valid_idx]
                    val_valid = val_valid[valid_idx]
                    std_valid = std_valid[valid_idx]
                downsampled_valid_total += max(0, valid_count - int(ke_valid.size))

                ax_curve.plot(
                    ke_valid,
                    val_valid,
                    f"{entry.get('marker', 'o')}-",
                    color=entry.get("color", "tab:blue"),
                    linewidth=2,
                    markersize=5,
                    label=entry.get("label", f"M{method_idx + 1}"),
                )
                std_mask = np.isfinite(std_valid)
                if np.any(std_mask):
                    ax_curve.errorbar(
                        ke_valid[std_mask],
                        val_valid[std_mask],
                        yerr=std_valid[std_mask],
                        fmt="none",
                        ecolor=entry.get("color", "tab:blue"),
                        alpha=0.35,
                        capsize=2,
                    )

        total_invalid = int(np.sum(np.array(invalid_counts, dtype=int))) if invalid_counts else 0
        if not legend_needed and total_invalid == 0:
            ax_curve.text(0.5, 0.5, "No KE data", ha="center", va="center", transform=ax_curve.transAxes)
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
        if downsampled_valid_total > 0 or downsampled_invalid_total > 0:
            point_text_parts.append(
                f"thinned V:{downsampled_valid_total} X:{downsampled_invalid_total}"
            )
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

    def _apply_zoom(scale_val, x_center=None, y_center=None):
        cur_xlim = view_state["xlim"] if view_state["xlim"] is not None else tuple(ax_curve.get_xlim())
        cur_ylim = view_state["ylim"] if view_state["ylim"] is not None else tuple(ax_curve.get_ylim())
        x_mid = 0.5 * (cur_xlim[0] + cur_xlim[1]) if x_center is None else x_center
        y_mid = 0.5 * (cur_ylim[0] + cur_ylim[1]) if y_center is None else y_center
        new_xlim = _zoom_limits(cur_xlim, x_mid, scale_val)
        new_ylim = _zoom_limits(cur_ylim, y_mid, scale_val)
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
            scale_val = 0.85
        elif button == "down":
            scale_val = 1.15
        else:
            return
        _apply_zoom(scale_val, x_center=event.xdata, y_center=event.ydata)

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
    # Force blocking show so workflow scripts don't immediately continue
    # and close figures in environments where interactive mode is enabled.
    plt.show(block=True)
    return True


def _parse_axis_range(text, axis_name):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--{axis_name} expects 'min,max', got: {text!r}")
    low = _safe_float(parts[0])
    high = _safe_float(parts[1])
    if low is None or high is None:
        raise ValueError(f"--{axis_name} expects numeric bounds, got: {text!r}")
    if low == high:
        raise ValueError(f"--{axis_name} min and max must be different, got: {text!r}")
    if low > high:
        low, high = high, low
    return (float(low), float(high))


def _interaction_volume_mm3(range_mm):
    numeric = _safe_float(range_mm)
    if numeric is None:
        return None
    return float(numeric) * float(numeric) * float(numeric)


def _format_iv_volume_label(range_mm):
    numeric = _safe_float(range_mm)
    if numeric is None:
        return f"Source={range_mm}*{range_mm}*{range_mm} mm^3"
    token = f"{float(numeric):.6g}"
    return f"Source={token}*{token}*{token} mm^3"


def _find_latest_file(results_dir, patterns):
    candidates = []
    for pattern in patterns:
        search_path = os.path.join(results_dir, pattern)
        candidates.extend(glob.glob(search_path))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_latest_v56_result(results_dir):
    bundle_path = _find_latest_file(results_dir, ["interaction_volume_scan_bundle_*.pkl"])
    if bundle_path:
        return bundle_path
    return _find_latest_file(
        results_dir,
        [
            "energy_resolution_summary_ivr_*.pkl",
            "energy_resolution_ivr_*.csv",
        ],
    )


def load_input_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "summary", load_summary_from_file(path)

    if ext != ".pkl":
        raise ValueError(f"Unsupported file type: {ext}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and isinstance(data.get("scan_summaries"), dict):
        return "bundle", data

    return "summary", load_summary_from_file(path)


def _pick_range_key(scan_summaries, requested_range):
    requested = float(requested_range)
    keys = list(scan_summaries.keys())
    if not keys:
        raise ValueError("scan_summaries is empty.")

    best_key = None
    best_delta = None
    for key in keys:
        num = _safe_float(key)
        if num is None:
            continue
        delta = abs(num - requested)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_key = key

    if best_key is None:
        raise ValueError("No numeric interaction-volume ranges found in scan_summaries.")

    if best_delta is not None and best_delta > 1e-9:
        raise ValueError(
            f"Requested range {requested:g} mm not found exactly. "
            f"Closest available is {float(best_key):g} mm."
        )
    return best_key


def _build_scan_record_index(scan_records):
    index = {}
    if not isinstance(scan_records, list):
        return index
    for record in scan_records:
        if not isinstance(record, dict):
            continue
        key_num = _safe_float(record.get("interaction_volume_range_mm"))
        if key_num is None:
            continue
        index[float(key_num)] = record
    return index


def _lookup_scan_stats(record_index, range_key):
    key_num = _safe_float(range_key)
    if key_num is None:
        return None
    return record_index.get(float(key_num))


def _thin_keys_evenly(keys, max_items):
    if max_items is None:
        return list(keys)
    try:
        max_items = int(max_items)
    except (TypeError, ValueError):
        return list(keys)
    if max_items <= 0 or len(keys) <= max_items:
        return list(keys)
    if max_items == 1:
        return [keys[0]]

    idx_set = set()
    count = len(keys)
    for i in range(max_items):
        idx = int(round((i * (count - 1)) / (max_items - 1)))
        idx = min(max(idx, 0), count - 1)
        idx_set.add(idx)
    return [keys[i] for i in sorted(idx_set)]


def _is_large_bundle(range_count, total_points):
    if int(range_count) >= int(LARGE_BUNDLE_RANGE_THRESHOLD):
        return True
    if total_points is None:
        return False
    return int(total_points) >= int(LARGE_BUNDLE_TOTAL_POINTS_THRESHOLD)


def _strip_heavy_payload_for_saved_view(summary_data):
    if not isinstance(summary_data, dict):
        return 0
    removed = 0
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
                for payload_key in HEAVY_PAYLOAD_GLOBAL_KEYS:
                    if payload_key in global_data:
                        global_data.pop(payload_key, None)
                        removed += 1
    return removed


def _build_method_summary(summary_data, method, origin_y, origin_z):
    method_entries = build_method_entries(
        summary_data,
        method=method,
        origin_y=origin_y,
        origin_z=origin_z,
    )
    if len(method_entries) != 1:
        raise ValueError("Expected exactly one method entry for range comparison.")
    return method_entries[0]["summary"]


def _build_saved_entry(summary_data, key, label, color, marker, group=None):
    entry = {
        "key": key,
        "label": label,
        "summary": summary_data,
        "color": color,
        "marker": marker,
    }
    if group is not None:
        entry["group"] = group
    return entry


def _build_single_method_entry(
    summary_data,
    method_key,
    origin_y,
    origin_z,
    key_name,
    label_prefix,
    color,
    marker,
):
    if method_key == "saved":
        return _build_saved_entry(
            summary_data,
            key_name,
            label_prefix,
            color,
            marker,
            group=method_key,
        )
    method_summary = _build_method_summary(
        summary_data,
        method=method_key,
        origin_y=origin_y,
        origin_z=origin_z,
    )
    return {
        "key": key_name,
        "label": label_prefix,
        "summary": method_summary,
        "color": color,
        "marker": marker,
        "group": method_key,
    }


def _build_entries_for_summary(
    summary_data,
    selected_method,
    origin_y,
    origin_z,
    key_prefix,
    label_prefix,
    color,
    marker,
):
    source_valid, _ = count_valid_energy_points(summary_data)

    if selected_method == "saved":
        return [_build_saved_entry(summary_data, key_prefix, label_prefix, color, marker)]

    if selected_method == "all":
        raw_entries = [
            build_method_entries(
                summary_data,
                method=method_key,
                origin_y=origin_y,
                origin_z=origin_z,
            )[0]
            for method_key in ("rmax", "sqrt")
        ]
        entries = []
        for idx, entry in enumerate(raw_entries):
            entries.append(
                {
                    "key": f"{key_prefix}_{idx}",
                    "label": f"{label_prefix} #{idx + 1}",
                    "summary": entry["summary"],
                    "color": entry["color"],
                    "marker": entry["marker"],
                }
            )
    else:
        method_summary = _build_method_summary(
            summary_data,
            method=selected_method,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        entries = [
            {
                "key": key_prefix,
                "label": label_prefix,
                "summary": method_summary,
                "color": color,
                "marker": marker,
            }
        ]

    derived_valid = 0
    for entry in entries:
        valid_points, _ = count_valid_energy_points(entry["summary"])
        derived_valid += valid_points

    if derived_valid == 0 and source_valid > 0:
        print(
            f"WARNING: method '{selected_method}' has no plottable points for {label_prefix}. "
            "Falling back to saved metric."
        )
        return [_build_saved_entry(summary_data, key_prefix, label_prefix, color, marker)]

    return entries


def print_summary_info(summary_data, prefix="Summary", stats=None):
    if isinstance(stats, dict):
        valid_points = stats.get("valid_points")
        total_points = stats.get("total_points")
        if valid_points is not None and total_points is not None:
            print(f"{prefix}: valid points {int(valid_points)}/{int(total_points)}")
            return
    valid_points, total_points = count_valid_energy_points(summary_data)
    print(f"{prefix}: valid points {valid_points}/{total_points}")


def launch_bundle_view(
    bundle_data,
    selected_method,
    origin_y,
    origin_z,
    y_as_percent,
    xlim=None,
    ylim=None,
    range_mm=None,
    no_gui=False,
    list_only=False,
    max_overlay_ranges=MAX_OVERLAY_RANGES_DEFAULT,
):
    selected_method = _normalize_method(selected_method)
    scan_summaries = bundle_data.get("scan_summaries", {})
    if not isinstance(scan_summaries, dict) or not scan_summaries:
        raise ValueError("Bundle does not contain scan_summaries.")
    scan_record_index = _build_scan_record_index(bundle_data.get("scan_records", []))

    range_keys = _sorted_numeric_keys(scan_summaries.keys())
    range_numbers = [float(_safe_float(k)) for k in range_keys if _safe_float(k) is not None]
    bundle_total_points = None
    if scan_record_index:
        total_points_values = []
        for record in scan_record_index.values():
            point_count = record.get("total_points")
            try:
                point_count = int(point_count)
            except (TypeError, ValueError):
                continue
            if point_count >= 0:
                total_points_values.append(point_count)
        if total_points_values:
            bundle_total_points = int(sum(total_points_values))
    if range_numbers:
        print("Available IV ranges (mm): " + ", ".join(f"{value:g}" for value in range_numbers))
        print(
            "Available IV volumes (mm^3): "
            + ", ".join(f"{_interaction_volume_mm3(value):.6g}" for value in range_numbers)
        )

    if list_only:
        return

    if range_mm is not None:
        selected_key = _pick_range_key(scan_summaries, range_mm)
        summary_data = scan_summaries[selected_key]
        label_prefix = _format_iv_volume_label(selected_key)
        selected_stats = _lookup_scan_stats(scan_record_index, selected_key)

        resolved_method = selected_method
        selected_total_points = selected_stats.get("total_points") if isinstance(selected_stats, dict) else None
        if (
            resolved_method in ("selector", "rmax", "sqrt", "all")
            and _is_large_bundle(range_count=1, total_points=selected_total_points)
        ):
            print(
                "NOTE: Large range payload detected; using saved metric for responsiveness. "
                "Use --method saved to silence this note."
            )
            resolved_method = "saved"
        if resolved_method == "saved":
            _strip_heavy_payload_for_saved_view(summary_data)

        print_summary_info(summary_data, prefix=label_prefix, stats=selected_stats)
        if no_gui:
            return
        group_options = None
        default_group = None
        if resolved_method == "selector":
            entries = []
            for method_key, _ in METHOD_SELECTOR_OPTIONS:
                entries.append(
                    _build_single_method_entry(
                        summary_data=summary_data,
                        method_key=method_key,
                        origin_y=origin_y,
                        origin_z=origin_z,
                        key_name=f"selected_{method_key}",
                        label_prefix=label_prefix,
                        color=SELECTOR_METHOD_COLORS.get(method_key, "tab:blue"),
                        marker=SELECTOR_METHOD_MARKERS.get(method_key, "o"),
                    )
                )
            group_options = [{"key": key, "label": label} for key, label in METHOD_SELECTOR_OPTIONS]
            default_group = "rmax"
        else:
            entries = _build_entries_for_summary(
                summary_data=summary_data,
                selected_method=resolved_method,
                origin_y=origin_y,
                origin_z=origin_z,
                key_prefix="selected_range",
                label_prefix=label_prefix,
                color="tab:blue",
                marker="o",
            )
        launched = launch_multi_method_gui(
            entries,
            title_prefix=_build_title_prefix(f"SIMION v5.6 | {label_prefix}", resolved_method),
            y_as_percent=y_as_percent,
            xlim=xlim,
            ylim=ylim,
            group_options=group_options,
            default_group=default_group,
        )
        if not launched:
            raise RuntimeError("No plottable data for selected interaction-volume range.")
        return

    if no_gui:
        for range_key in range_keys:
            summary_data = scan_summaries[range_key]
            label_prefix = _format_iv_volume_label(range_key)
            range_stats = _lookup_scan_stats(scan_record_index, range_key)
            print_summary_info(summary_data, prefix=label_prefix, stats=range_stats)
        return

    method_for_scan = selected_method
    if method_for_scan == "all":
        print("NOTE: --method all is not supported for multi-range overlay; falling back to rmax.")
        method_for_scan = "rmax"
    if method_for_scan in ("selector", "rmax", "sqrt") and _is_large_bundle(
        range_count=len(range_keys),
        total_points=bundle_total_points,
    ):
        print(
            "NOTE: Large scan bundle detected; using saved metric for multi-range overlay "
            "to reduce memory and recomputation."
        )
        method_for_scan = "saved"

    selected_overlay_keys = _thin_keys_evenly(range_keys, max_overlay_ranges)
    if len(selected_overlay_keys) < len(range_keys):
        print(
            f"NOTE: Overlay limited to {len(selected_overlay_keys)}/{len(range_keys)} ranges "
            f"(set --max-ranges to adjust)."
        )
    if method_for_scan == "saved":
        removed_payload_fields = 0
        for range_key in selected_overlay_keys:
            removed_payload_fields += _strip_heavy_payload_for_saved_view(scan_summaries.get(range_key))
        if removed_payload_fields > 0:
            print(f"NOTE: Removed {removed_payload_fields} heavy payload fields for saved-metric viewing.")

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    markers = ["o", "s", "^", "d", "v", "P", "X", "*", "h"]

    method_entries = []
    group_options = None
    default_group = None
    for idx, range_key in enumerate(selected_overlay_keys):
        summary_data = scan_summaries[range_key]
        label_prefix = _format_iv_volume_label(range_key)
        range_stats = _lookup_scan_stats(scan_record_index, range_key)
        print_summary_info(summary_data, prefix=label_prefix, stats=range_stats)
        if method_for_scan == "selector":
            for method_key, _ in METHOD_SELECTOR_OPTIONS:
                method_entries.append(
                    _build_single_method_entry(
                        summary_data=summary_data,
                        method_key=method_key,
                        origin_y=origin_y,
                        origin_z=origin_z,
                        key_name=f"{method_key}_ivr_{idx}",
                        label_prefix=label_prefix,
                        color=colors[idx % len(colors)],
                        marker=markers[idx % len(markers)],
                    )
                )
            group_options = [{"key": key, "label": label} for key, label in METHOD_SELECTOR_OPTIONS]
            default_group = "rmax"
        else:
            method_entries.extend(
                _build_entries_for_summary(
                    summary_data=summary_data,
                    selected_method=method_for_scan,
                    origin_y=origin_y,
                    origin_z=origin_z,
                    key_prefix=f"ivr_{idx}",
                    label_prefix=label_prefix,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                )
            )

    if no_gui:
        return

    launched = launch_multi_method_gui(
        method_entries,
        title_prefix=_build_title_prefix("SIMION v5.6 | Interaction Volume Scan", method_for_scan),
        y_as_percent=y_as_percent,
        xlim=xlim,
        ylim=ylim,
        group_options=group_options,
        default_group=default_group,
    )
    if not launched:
        raise RuntimeError("No plottable data in bundle viewer.")


def launch_summary_view(summary_data, selected_method, origin_y, origin_z, y_as_percent, xlim=None, ylim=None, no_gui=False):
    selected_method = _normalize_method(selected_method)
    print_summary_info(summary_data)
    if no_gui:
        return

    group_options = None
    default_group = None
    if selected_method == "selector":
        entries = []
        for method_key, _ in METHOD_SELECTOR_OPTIONS:
            entries.append(
                _build_single_method_entry(
                    summary_data=summary_data,
                    method_key=method_key,
                    origin_y=origin_y,
                    origin_z=origin_z,
                    key_name=f"summary_{method_key}",
                    label_prefix="Summary",
                    color=SELECTOR_METHOD_COLORS.get(method_key, "tab:blue"),
                    marker=SELECTOR_METHOD_MARKERS.get(method_key, "o"),
                )
            )
        group_options = [{"key": key, "label": label} for key, label in METHOD_SELECTOR_OPTIONS]
        default_group = "rmax"
    else:
        entries = _build_entries_for_summary(
            summary_data=summary_data,
            selected_method=selected_method,
            origin_y=origin_y,
            origin_z=origin_z,
            key_prefix="summary",
            label_prefix="Summary",
            color="tab:blue",
            marker="o",
        )
    launched = launch_multi_method_gui(
        entries,
        y_as_percent=y_as_percent,
        title_prefix=_build_title_prefix("SIMION v5.6 | Energy Resolution", selected_method),
        xlim=xlim,
        ylim=ylim,
        group_options=group_options,
        default_group=default_group,
    )
    if not launched:
        raise RuntimeError("No plottable data in summary viewer.")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone viewer for SIMION workflow v5.6 saved results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to input file (.pkl/.csv). Supports v5.6 scan bundle "
            "(interaction_volume_scan_bundle_*.pkl) and single summary files."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_v5.6",
        help="Directory used when --input is omitted.",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=None,
        help=(
            "If input is a scan bundle, plot only this interaction-volume "
            "range in mm (exact match required)."
        ),
    )
    parser.add_argument(
        "--list-ranges",
        action="store_true",
        help="If input is a scan bundle, list available ranges and exit.",
    )
    parser.add_argument(
        "--max-ranges",
        type=int,
        default=MAX_OVERLAY_RANGES_DEFAULT,
        help=(
            "When plotting bundle overlay (without --range), cap displayed ranges for "
            "responsiveness using even sampling. Use <=0 to disable cap."
        ),
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="rmax",
        help=(
            "Metric method: selector=GUI switcher (rmax only), "
            "saved=use saved energy_resolution directly, "
            "rmax=|ra-rb|/rmax(all points), "
            "sqrt=sqrt(ra^2-rb^2)/r_mean(pair) (disabled fallback to saved), "
            "all=plot all methods (single summary only)."
        ),
    )
    parser.add_argument(
        "--origin-y",
        type=float,
        default=DEFAULT_ORIGIN_Y,
        help="Origin y for radius reconstruction in yz plane.",
    )
    parser.add_argument(
        "--origin-z",
        type=float,
        default=DEFAULT_ORIGIN_Z,
        help="Origin z for radius reconstruction in yz plane.",
    )
    parser.add_argument(
        "--fraction",
        action="store_true",
        help="Display dE/E as fraction instead of percent.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Only print summary info, do not open the plot window.",
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
        target_file = find_latest_v56_result(args.results_dir)
        if not target_file:
            raise FileNotFoundError(
                f"No v5.6 result files found in '{args.results_dir}'."
            )

    target_file = os.path.abspath(target_file)
    kind, data = load_input_data(target_file)

    print("=" * 70)
    print("SIMION v5.6 Result Viewer")
    print("=" * 70)
    print(f"Input file: {target_file}")
    print(f"Input type: {kind}")
    print(f"Matplotlib backend: {MATPLOTLIB_BACKEND_SELECTED}")
    print(f"Method: {args.method}")
    print(f"Origin (y,z): ({args.origin_y:.6g}, {args.origin_z:.6g})")
    print(f"Y axis mode: {'fraction' if args.fraction else 'percent'}")
    if xlim is not None:
        print(f"Initial xlim: [{xlim[0]:.6g}, {xlim[1]:.6g}]")
    if ylim is not None:
        print(f"Initial ylim: [{ylim[0]:.6g}, {ylim[1]:.6g}]")

    if kind == "bundle":
        launch_bundle_view(
            bundle_data=data,
            selected_method=args.method,
            origin_y=args.origin_y,
            origin_z=args.origin_z,
            y_as_percent=not args.fraction,
            xlim=xlim,
            ylim=ylim,
            range_mm=args.range,
            no_gui=args.no_gui,
            list_only=args.list_ranges,
            max_overlay_ranges=args.max_ranges,
        )
    else:
        if args.list_ranges:
            print("NOTE: --list-ranges is ignored for non-bundle input.")
        launch_summary_view(
            summary_data=data,
            selected_method=args.method,
            origin_y=args.origin_y,
            origin_z=args.origin_z,
            y_as_percent=not args.fraction,
            xlim=xlim,
            ylim=ylim,
            no_gui=args.no_gui,
        )


if __name__ == "__main__":
    main()
