import argparse
import glob
import os
import pickle

import view_energy_resolution_results as viewer

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
        return "summary", viewer.load_summary_from_file(path)

    if ext != ".pkl":
        raise ValueError(f"Unsupported file type: {ext}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and isinstance(data.get("scan_summaries"), dict):
        return "bundle", data

    return "summary", viewer.load_summary_from_file(path)


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


def _build_method_summary(summary_data, method, origin_y, origin_z):
    method_entries = viewer.build_method_entries(
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
    source_valid, _ = viewer.count_valid_energy_points(summary_data)

    if selected_method == "saved":
        return [_build_saved_entry(summary_data, key_prefix, label_prefix, color, marker)]

    if selected_method == "all":
        raw_entries = [
            viewer.build_method_entries(
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
        valid_points, _ = viewer.count_valid_energy_points(entry["summary"])
        derived_valid += valid_points

    if derived_valid == 0 and source_valid > 0:
        print(
            f"WARNING: method '{selected_method}' has no plottable points for {label_prefix}. "
            "Falling back to saved metric."
        )
        return [_build_saved_entry(summary_data, key_prefix, label_prefix, color, marker)]

    return entries


def print_summary_info(summary_data, prefix="Summary"):
    valid_points, total_points = viewer.count_valid_energy_points(summary_data)
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
):
    selected_method = _normalize_method(selected_method)
    scan_summaries = bundle_data.get("scan_summaries", {})
    if not isinstance(scan_summaries, dict) or not scan_summaries:
        raise ValueError("Bundle does not contain scan_summaries.")

    range_keys = _sorted_numeric_keys(scan_summaries.keys())
    range_numbers = [float(_safe_float(k)) for k in range_keys if _safe_float(k) is not None]
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
        print_summary_info(summary_data, prefix=label_prefix)
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
                selected_method=selected_method,
                origin_y=origin_y,
                origin_z=origin_z,
                key_prefix="selected_range",
                label_prefix=label_prefix,
                color="tab:blue",
                marker="o",
            )
        launched = viewer.launch_multi_method_gui(
            entries,
            title_prefix=_build_title_prefix(f"SIMION v5.6 | {label_prefix}", selected_method),
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
            print_summary_info(summary_data, prefix=label_prefix)
        return

    method_for_scan = selected_method
    if method_for_scan == "all":
        print("NOTE: --method all is not supported for multi-range overlay; falling back to rmax.")
        method_for_scan = "rmax"

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
    for idx, range_key in enumerate(range_keys):
        summary_data = scan_summaries[range_key]
        label_prefix = _format_iv_volume_label(range_key)
        print_summary_info(summary_data, prefix=label_prefix)
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

    launched = viewer.launch_multi_method_gui(
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
    launched = viewer.launch_multi_method_gui(
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
        default=viewer.DEFAULT_ORIGIN_Y,
        help="Origin y for radius reconstruction in yz plane.",
    )
    parser.add_argument(
        "--origin-z",
        type=float,
        default=viewer.DEFAULT_ORIGIN_Z,
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
