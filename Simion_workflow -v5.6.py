"""
Streamlined SIMION Workflow - v5.6 (Energy Resolution Only)

Simplified version focused exclusively on energy resolution analysis:
- Removed: Focus filtering, initial simulation workflow, unused visualization
- Kept: Energy resolution analysis, result saving, essential GUI
- Cleaner: Reduced code size, improved clarity, maintained all ER functionality
"""

import os
import csv
import gc
import pickle
import json
import numpy as np
import time
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prefer interactive matplotlib backends for GUI usage. Fall back to Agg only if needed.
def _configure_matplotlib_backend():
    try:
        import matplotlib
    except Exception:
        return "unknown"

    preferred_backends = ("qtagg", "tkagg", "qt5agg")
    for backend_name in preferred_backends:
        try:
            matplotlib.use(backend_name, force=True)
            import matplotlib.pyplot as _plt_probe
            fig = _plt_probe.figure()
            _plt_probe.close(fig)
            os.environ["MPLBACKEND"] = backend_name
            return backend_name
        except Exception:
            continue

    try:
        matplotlib.use("Agg", force=True)
        os.environ["MPLBACKEND"] = "Agg"
        return "Agg"
    except Exception:
        return "unknown"

MATPLOTLIB_BACKEND_SELECTED = _configure_matplotlib_backend()

# Import streamlined Utilis v5.6 module (energy resolution only)
import importlib.util
spec = importlib.util.spec_from_file_location("Utilis", os.path.join(os.path.dirname(__file__), "Utilis_v5.6.py"))
Utilis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Utilis)

# ============================================================================
# CONFIGURATION
# ============================================================================

# === OUTPUT CONTROL ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_v5.6")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# === VISUALIZATION ===
ENABLE_VISUALIZATION = True  # Launch interactive GUI after analysis
SAVE_SCAN_CURVE_PLOTS = False  # Save static multi-curve plots for each (FG, Lens)
PLOT_X_LIMITS = None  # e.g. (1.0, 16.0) in eV
PLOT_Y_LIMITS = None  # e.g. (0.0, 25.0) in percent
VISUALIZATION_METHOD = 'rmax'  # selector(rmax only)|saved|rmax|sqrt|all

# === SAVE OPTIONS ===
SAVE_FULL_PROCESSED_DATA = False  # Save only summary to avoid OOM
SAVE_RANGE_SUMMARY_PKL = False  # Disabled by default: keep only final scan bundle pickle
SAVE_SCAN_BUNDLE_PKL = True  # Save scan bundle pickle
SAVE_RANGE_CSV = False  # Save per-range CSV
SAVE_SCAN_SUMMARY_CSV = False  # Save scan-level summary CSV
SAVE_SCAN_MANIFEST_JSON = False  # Save scan-level JSON manifest
CLEAR_STALE_CHECKPOINTS = False  # Keep checkpoints for resume capability
SAVE_PAIR_PAYLOAD_IN_SUMMARY = True  # Include compact dr_over_r_pairs in summary pickle
SAVE_RAW_POINT_PAYLOAD_IN_SUMMARY = False  # Include raw_ion_points_yz (large payload) in summary pickle
CLEANUP_CHECKPOINTS_AFTER_SUCCESS = True
CLEANUP_TEMP_FILES_AFTER_SUCCESS = True
FORCE_SINGLE_BUNDLE_OUTPUT = True  # If True, workflow only keeps one final output pickle.

if FORCE_SINGLE_BUNDLE_OUTPUT:
    SAVE_RANGE_SUMMARY_PKL = False
    SAVE_FULL_PROCESSED_DATA = False

# === PARAMETER RANGES ===
KE_MIN = 1
KE_MAX = 15.5
NUM_KE_POINTS = 3

FIELD_MIN = 50
FIELD_MAX = 300
NUM_FIELD_POINTS = 3

LENS_MIN = 1.3
LENS_MAX = 3
NUM_LENS_POINTS = 3

# === ENERGY RESOLUTION SETTINGS ===
NUM_PARTICLES_PER_ENERGY = 91
SOURCE_POSITION = (199, 0, 0)
BIN_INTERVAL = 0.01
OUTSIDE_REGION_WIDTH = 2
ENERGY_BATCH_SIZE = 25
MAX_COMBO_RETRIES = 3
REQUIRE_FULL_PARTICLE_CAPTURE = True
RETRY_BACKOFF_S = 0.5
TIMING_VERBOSE = True
CHECKPOINT_INTERVAL = 300
GC_INTERVAL_COMBOS = 50

# === INTERACTION VOLUME (Source Motion) ===
Interaction_volume = True
INTERACTION_VOLUME_RANGE_SCAN = [0.0, 0.5,1,1.5] # mm

NUM_RUNS_PER_NODE = 15 if Interaction_volume else 1

# === DETECTOR ACCEPTANCE ===
DETECTOR_X_MM = 73.0
DETECTOR_X_TOL_MM = 0.5
DETECTOR_Y_RANGE_MM = (-35.0, 35.0)
DETECTOR_Z_RANGE_MM = (-35.0, 35.0)

# Validation
if NUM_PARTICLES_PER_ENERGY <= 0:
    raise ValueError("NUM_PARTICLES_PER_ENERGY must be > 0")
if NUM_RUNS_PER_NODE <= 0:
    raise ValueError("NUM_RUNS_PER_NODE must be > 0")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _normalize_numeric_key(val):
    """Convert value to float for consistent dictionary keys."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return val

def _safe_float(val):
    """Safely convert to float, return None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _safe_int(val):
    """Safely convert to int, return None on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def _sorted_keys(keys):
    """Sort dictionary keys numerically."""
    numeric_keys = []
    other_keys = []
    for k in keys:
        fk = _safe_float(k)
        if fk is not None:
            numeric_keys.append((fk, k))
        else:
            other_keys.append(k)
    numeric_keys.sort(key=lambda x: x[0])
    return [k for _, k in numeric_keys] + sorted(other_keys)

def _to_builtin(obj):
    """Convert numpy types to Python built-ins for JSON/pickle compatibility."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_builtin(item) for item in obj]
    return obj

# ============================================================================
# DATA PROCESSING
# ============================================================================

SUMMARY_BASE_FIELDS = (
    'fwhm',
    'energy_resolution',
    'max_r',
    'generated_particles',
    'detected_particles',
    'valid',
    'failure_reason',
)

SUMMARY_OPTIONAL_FIELDS = (
    'fwhm_mean', 'fwhm_variance', 'fwhm_std', 'fwhm_runs',
    'energy_resolution_mean', 'energy_resolution_variance', 'energy_resolution_std', 'energy_resolution_runs',
    'max_r_mean', 'max_r_variance', 'max_r_std', 'max_r_runs',
    'r_max_all_points', 'r_max_all_points_runs', 'all_point_count_runs',
    'generated_particles_per_run',
    'detected_particles_runs', 'valid_run_count', 'total_runs',
    'pair_count', 'pair_count_runs',
    'raw_point_count', 'raw_point_format',
    'failure_reasons',
    'count_check_passed', 'plot_marker', 'plot_skip', 'pipeline_stage',
)

SUMMARY_PAIR_FIELDS = ('dr_over_r_pairs',)
SUMMARY_RAW_FIELDS = ('raw_ion_points_yz',)

def create_empty_processed_data(field_gradients, lens_vmis, energies):
    """Create empty data structure for energy resolution results."""
    data = {}
    for fg in field_gradients:
        fg_key = _normalize_numeric_key(fg)
        data[fg_key] = {}
        for lens in lens_vmis:
            lens_key = _normalize_numeric_key(lens)
            data[fg_key][lens_key] = {}
            for ke in energies:
                ke_key = _normalize_numeric_key(ke)
                data[fg_key][lens_key][ke_key] = {
                    'local': {},
                    'global': {}
                }
    return data

def build_energy_resolution_summary(
    processed_data,
    include_pair_payload=True,
    include_raw_payload=False
):
    """Extract v5.4-compatible summary structure from processed data."""
    optional_fields = list(SUMMARY_OPTIONAL_FIELDS)
    if include_pair_payload:
        optional_fields.extend(SUMMARY_PAIR_FIELDS)
    if include_raw_payload:
        optional_fields.extend(SUMMARY_RAW_FIELDS)

    summary = {}
    for fg, fg_data in processed_data.items():
        if not isinstance(fg_data, dict):
            continue
        fg_key = _normalize_numeric_key(fg)
        fg_summary = {}
        for lens_vmi, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            lens_key = _normalize_numeric_key(lens_vmi)
            lens_summary = {}
            for ke, ke_data in lens_data.items():
                if not isinstance(ke_data, dict):
                    continue
                global_data = ke_data.get('global')
                if not isinstance(global_data, dict):
                    global_data = {}

                global_summary = {}
                for key in SUMMARY_BASE_FIELDS:
                    global_summary[key] = _to_builtin(global_data.get(key))

                for key in optional_fields:
                    if key in global_data:
                        global_summary[key] = _to_builtin(global_data.get(key))

                lens_summary[_normalize_numeric_key(ke)] = {
                    'global': global_summary,
                    'local': {}
                }
            if lens_summary:
                fg_summary[lens_key] = lens_summary
        if fg_summary:
            summary[fg_key] = fg_summary
    return summary

def collect_summary_statistics(summary_data):
    """
    Single-pass summary statistics to avoid repeated full-tree traversals.
    Returns:
      valid_points, total_points, unique_er_count, raw_nodes, raw_point_total, valid_runs, total_runs
    """
    valid_points = 0
    total_points = 0
    unique_er_values = set()
    raw_nodes = 0
    raw_point_total = 0
    valid_runs = 0
    total_runs = 0

    for fg_data in summary_data.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if not isinstance(ke_data, dict):
                    continue

                total_points += 1
                global_data = ke_data.get('global')
                if not isinstance(global_data, dict):
                    global_data = {}

                # Align with viewer preference: mean first, then fallback to energy_resolution.
                er_num = _safe_float(global_data.get('energy_resolution_mean', global_data.get('energy_resolution')))
                if er_num is not None and np.isfinite(er_num):
                    valid_points += 1
                    unique_er_values.add(round(float(er_num), 6))

                raw_points = global_data.get('raw_ion_points_yz', [])
                if isinstance(raw_points, list) and raw_points:
                    raw_nodes += 1
                    raw_point_total += len(raw_points)

                point_total_runs = _safe_int(global_data.get('total_runs'))
                point_valid_runs = _safe_int(global_data.get('valid_run_count'))
                if point_total_runs is None or point_valid_runs is None:
                    total_runs += 1
                    if bool(global_data.get('valid')):
                        valid_runs += 1
                else:
                    point_total_runs = max(point_total_runs, 0)
                    point_valid_runs = max(min(point_valid_runs, point_total_runs), 0)
                    total_runs += point_total_runs
                    valid_runs += point_valid_runs

    return (
        valid_points,
        total_points,
        len(unique_er_values),
        raw_nodes,
        raw_point_total,
        valid_runs,
        total_runs,
    )

def save_pickle_atomic(obj, filepath):
    """Atomically save pickle file (write to temp, then rename)."""
    temp_path = filepath + '.tmp'
    with open(temp_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, filepath)
    return True

def _serialize_numeric_runs(values):
    if not isinstance(values, (list, tuple, np.ndarray)):
        return ''
    encoded = []
    for value in values:
        numeric = _safe_float(value)
        encoded.append('' if numeric is None else f"{numeric:.8g}")
    return '|'.join(encoded)

def save_energy_resolution_csv(summary_data, output_path):
    row_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'field_gradient',
            'lens_vmi',
            'kinetic_energy_eV',
            'energy_resolution',
            'energy_resolution_mean',
            'energy_resolution_variance',
            'energy_resolution_std',
            'energy_resolution_runs',
            'fwhm',
            'fwhm_mean',
            'fwhm_variance',
            'fwhm_std',
            'fwhm_runs',
            'max_r',
            'generated_particles',
            'detected_particles',
            'detected_particles_runs',
            'valid',
            'valid_run_count',
            'total_runs',
            'failure_reason'
        ])
        for fg in _sorted_keys(summary_data.keys()):
            for lens_vmi in _sorted_keys(summary_data[fg].keys()):
                for ke in _sorted_keys(summary_data[fg][lens_vmi].keys()):
                    global_data = summary_data[fg][lens_vmi][ke].get('global', {})
                    energy_resolution = global_data.get('energy_resolution')
                    energy_resolution_mean = global_data.get('energy_resolution_mean', energy_resolution)
                    fwhm_value = global_data.get('fwhm')
                    fwhm_mean = global_data.get('fwhm_mean', fwhm_value)
                    writer.writerow([
                        fg,
                        lens_vmi,
                        ke,
                        energy_resolution,
                        energy_resolution_mean,
                        global_data.get('energy_resolution_variance'),
                        global_data.get('energy_resolution_std'),
                        _serialize_numeric_runs(global_data.get('energy_resolution_runs')),
                        fwhm_value,
                        fwhm_mean,
                        global_data.get('fwhm_variance'),
                        global_data.get('fwhm_std'),
                        _serialize_numeric_runs(global_data.get('fwhm_runs')),
                        global_data.get('max_r'),
                        global_data.get('generated_particles'),
                        global_data.get('detected_particles'),
                        _serialize_numeric_runs(global_data.get('detected_particles_runs')),
                        global_data.get('valid'),
                        global_data.get('valid_run_count'),
                        global_data.get('total_runs'),
                        global_data.get('failure_reason')
                    ])
                    row_count += 1
    return row_count

def save_interaction_scan_csv(scan_summaries, output_path):
    row_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'interaction_volume_range_mm',
            'field_gradient',
            'lens_vmi',
            'kinetic_energy_eV',
            'energy_resolution_mean',
            'energy_resolution_std',
            'generated_particles',
            'detected_particles',
            'pair_count',
            'valid',
            'valid_run_count',
            'total_runs',
            'failure_reason',
            'pipeline_stage'
        ])
        for range_value in _sorted_keys(scan_summaries.keys()):
            summary_data = scan_summaries.get(range_value, {})
            for fg in _sorted_keys(summary_data.keys()):
                for lens_vmi in _sorted_keys(summary_data[fg].keys()):
                    for ke in _sorted_keys(summary_data[fg][lens_vmi].keys()):
                        global_data = summary_data[fg][lens_vmi][ke].get('global', {})
                        writer.writerow([
                            range_value,
                            fg,
                            lens_vmi,
                            ke,
                            global_data.get('energy_resolution_mean', global_data.get('energy_resolution')),
                            global_data.get('energy_resolution_std'),
                            global_data.get('generated_particles'),
                            global_data.get('detected_particles'),
                            global_data.get('pair_count'),
                            global_data.get('valid'),
                            global_data.get('valid_run_count'),
                            global_data.get('total_runs'),
                            global_data.get('failure_reason'),
                            global_data.get('pipeline_stage')
                        ])
                        row_count += 1
    return row_count

def save_json_atomic(obj, filepath):
    temp_path = filepath + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, filepath)

def _format_scan_token(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    token = f"{numeric:.6g}"
    token = token.replace('-', 'm').replace('.', 'p')
    return token

def _format_interaction_volume_label(range_mm):
    numeric = _safe_float(range_mm)
    if numeric is None:
        return f'Source={range_mm}*{range_mm}*{range_mm} mm^3'
    token = f'{float(numeric):.6g}'
    return f'Source={token}*{token}*{token} mm^3'

def _normalize_visualization_method(method):
    allowed = {'selector', 'saved', 'rmax', 'sqrt', 'all'}
    method_key = str(method or 'rmax').strip().lower()
    if method_key in allowed:
        return method_key
    print(
        f"WARNING: Unsupported VISUALIZATION_METHOD='{method}'. "
        "Falling back to 'rmax'."
    )
    return 'rmax'

def _is_non_interactive_matplotlib_backend():
    try:
        import matplotlib
        backend = str(matplotlib.get_backend() or '').strip().lower()
    except Exception:
        return False
    if not backend:
        return False

    # Explicit static backends only; do not treat qtagg/tkagg as non-interactive.
    non_interactive = {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'}
    if backend in non_interactive:
        return True

    # Jupyter inline backends are non-interactive for desktop GUI windows.
    if backend.startswith('module://matplotlib_inline') or 'backend_inline' in backend:
        return True

    return False

def _get_matplotlib_backend_name():
    try:
        import matplotlib
        return str(matplotlib.get_backend() or 'unknown')
    except Exception:
        return 'unknown'

def _try_switch_matplotlib_backend(target_backend):
    try:
        import matplotlib
        matplotlib.use(target_backend, force=True)
        return True
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend(target_backend)
        return True
    except Exception:
        return False

def _ensure_interactive_matplotlib_backend():
    """Try to switch to an interactive backend when current backend is non-interactive."""
    if not _is_non_interactive_matplotlib_backend():
        return True
    for backend_name in ('qtagg', 'tkagg', 'qt5agg'):
        if _try_switch_matplotlib_backend(backend_name):
            if not _is_non_interactive_matplotlib_backend():
                return True
    return False

def resolve_interaction_volume_ranges():
    if not Interaction_volume:
        return [0.0]

    if not isinstance(INTERACTION_VOLUME_RANGE_SCAN, (list, tuple, np.ndarray)):
        raise ValueError("INTERACTION_VOLUME_RANGE_SCAN must be a list/tuple/ndarray")

    ranges = []
    for value in INTERACTION_VOLUME_RANGE_SCAN:
        numeric = _safe_float(value)
        if numeric is None or not np.isfinite(numeric):
            continue
        ranges.append(float(numeric))

    unique_ranges = sorted(set(ranges))
    if not unique_ranges:
        raise ValueError("No valid numeric values in INTERACTION_VOLUME_RANGE_SCAN")
    return unique_ranges

def launch_interaction_range_gui(scan_summaries, method='selector'):
    """Launch GUI via the standalone v5.6 viewer module."""
    import view_results_v56 as viewer_v56
    bundle_data = {'scan_summaries': scan_summaries}
    return viewer_v56.launch_bundle_view(
        bundle_data=bundle_data,
        selected_method=method,
        origin_y=SOURCE_POSITION[1],
        origin_z=SOURCE_POSITION[2],
        y_as_percent=True,
        xlim=PLOT_X_LIMITS,
        ylim=PLOT_Y_LIMITS,
        range_mm=None,
        no_gui=False,
        list_only=False
    )

def plot_interaction_volume_scan_curves(scan_summaries, output_dir, run_tag, y_as_percent=True):
    import matplotlib.pyplot as plt

    range_values = _sorted_keys(scan_summaries.keys())
    if not range_values:
        return []

    fg_values = set()
    for summary_data in scan_summaries.values():
        fg_values.update(summary_data.keys())
    fg_values = _sorted_keys(fg_values)

    colors = plt.cm.tab10(np.linspace(0, 1, max(2, len(range_values))))
    markers = ['o', 's', '^', 'd', 'v', 'P', 'X', '*', 'h']
    scale = 100.0 if y_as_percent else 1.0
    ylabel = 'Energy Resolution (%)' if y_as_percent else 'Energy Resolution (dE/E)'
    plot_paths = []

    for fg in fg_values:
        lens_values = set()
        for range_value in range_values:
            lens_values.update(scan_summaries[range_value].get(fg, {}).keys())
        lens_values = _sorted_keys(lens_values)

        for lens in lens_values:
            fig, ax = plt.subplots(figsize=(10, 6))
            any_valid_curve = False
            invalid_points = []

            for idx, range_value in enumerate(range_values):
                node_by_ke = scan_summaries[range_value].get(fg, {}).get(lens, {})
                if not node_by_ke:
                    continue
                ke_values = _sorted_keys(node_by_ke.keys())
                x_valid = []
                y_valid = []
                x_invalid = []
                for ke in ke_values:
                    global_data = node_by_ke.get(ke, {}).get('global', {})
                    er_value = global_data.get('energy_resolution_mean', global_data.get('energy_resolution'))
                    er_float = _safe_float(er_value)
                    ke_float = _safe_float(ke)
                    if ke_float is None:
                        continue
                    if er_float is not None and np.isfinite(er_float):
                        x_valid.append(float(ke_float))
                        y_valid.append(float(er_float) * scale)
                    else:
                        if global_data.get('plot_marker') == 'x' or bool(global_data.get('plot_skip')):
                            x_invalid.append(float(ke_float))

                if x_valid:
                    any_valid_curve = True
                    ax.plot(
                        x_valid,
                        y_valid,
                        marker=markers[idx % len(markers)],
                        color=colors[idx],
                        linewidth=2.0,
                        markersize=5,
                        label=_format_interaction_volume_label(range_value)
                    )
                if x_invalid:
                    invalid_points.append((idx, x_invalid))

            if invalid_points:
                y_min, y_max = ax.get_ylim()
                y_span = y_max - y_min
                if not np.isfinite(y_span) or y_span <= 0:
                    y_min = 0.0
                    y_span = 1.0
                cross_y = y_min + 0.08 * y_span
                for idx, x_values in invalid_points:
                    ax.scatter(
                        x_values,
                        np.full(len(x_values), cross_y, dtype=float),
                        marker='x',
                        color=colors[idx],
                        s=55,
                        linewidths=1.5,
                        zorder=6
                    )

            if not any_valid_curve and not invalid_points:
                ax.text(0.5, 0.5, 'No plottable data', ha='center', va='center', transform=ax.transAxes)

            ax.set_xlabel('Kinetic Energy (eV)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Interaction Volume Range Scan\nFG={fg:g}, Lens={lens:g}')
            ax.grid(True, alpha=0.3)
            if any_valid_curve:
                ax.legend(loc='best', fontsize=9)

            fg_token = _format_scan_token(fg)
            lens_token = _format_scan_token(lens)
            plot_path = os.path.join(
                output_dir,
                f'interaction_volume_scan_fg_{fg_token}_lens_{lens_token}_{run_tag}.png'
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
            plot_paths.append(plot_path)

    return plot_paths

def launch_field_gradient_gui(summary_data, method='selector'):
    """Launch interactive GUI for one summary via standalone v5.6 viewer module."""
    import view_results_v56 as viewer_v56
    return viewer_v56.launch_summary_view(
        summary_data=summary_data,
        selected_method=method,
        origin_y=SOURCE_POSITION[1],
        origin_z=SOURCE_POSITION[2],
        y_as_percent=True,
        xlim=PLOT_X_LIMITS,
        ylim=PLOT_Y_LIMITS,
        no_gui=False
    )

def clear_energy_resolution_checkpoints():
    """Clear all energy resolution checkpoint files."""
    import glob
    checkpoint_patterns = [
        'energy_resolution_checkpoint.pkl',
        'energy_resolution_direct_checkpoint.pkl',
        'energy_resolution_checkpoint_merged.pkl',
        'energy_resolution_direct_checkpoint_merged.pkl',
        'energy_resolution_direct_*_checkpoint_merged.pkl',
        'checkpoint_energy_resolution_direct.pkl',
        'checkpoint_energy_resolution_direct_shard_*.pkl',
        'energy_resolution_checkpoint_part_*.pkl',
        'energy_resolution_direct_checkpoint_part_*.pkl',
        'energy_resolution_direct_*_checkpoint_part_*.pkl'
    ]

    checkpoint_files = []
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(pattern))

    removed = 0
    for checkpoint_file in sorted(set(checkpoint_files)):
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                removed += 1
                print(f"  Removed checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"  Failed to remove {checkpoint_file}: {e}")

    if removed > 0:
        print(f"✓ Cleared {removed} checkpoint file(s)")
    else:
        print("No checkpoint files found")

    return removed

def clear_runtime_temp_files():
    """Remove runtime temp/intermediate files without touching final outputs."""
    import glob
    temp_patterns = [
        '*.tmp',
        'temp_out_ke_*.txt',
        'temp_er_*.fly2',
        'temp_er_*.lua',
        'temp_er_*.txt',
        'WORKING_TITLE_tao_ke_*.lua',
        'WORKING_TITLE_energy_resolution_*.fly2',
        'WORKING_TITLE_energy_resolution_*.lua',
        'trapcheck.info'
    ]
    removed = 0
    for pattern in temp_patterns:
        for path in glob.glob(pattern):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
    return removed

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

print("=" * 70)
print("SIMION Energy Resolution Analysis - v5.6")
print("=" * 70)
print(f"Run tag: {RUN_TAG}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Matplotlib backend: {MATPLOTLIB_BACKEND_SELECTED}")
print()

# Clear stale checkpoints if requested
if CLEAR_STALE_CHECKPOINTS:
    print("Clearing stale checkpoints...")
    clear_energy_resolution_checkpoints()
    print()

# === STEP 1: Compute VMI Parameters ===
print("STEP 1: Computing VMI parameters...")
electron_energies = np.linspace(KE_MIN, KE_MAX, NUM_KE_POINTS)

vmi_params = Utilis.compute_vmi_parameters(
    field_min=FIELD_MIN,
    field_max=FIELD_MAX,
    num_points=NUM_FIELD_POINTS,
    lens_min=LENS_MIN,
    lens_max=LENS_MAX,
    num_lens_points=NUM_LENS_POINTS,
    mode='velocity_imaging',
    save_to_file=False
)

# Extract unique values from parameter arrays
unique_fgs = np.unique(vmi_params['field_gradient'])
unique_lens = np.unique(vmi_params['lens_VMI'])
unique_kes = electron_energies

print(f"  Field gradients: {len(unique_fgs)} points from {FIELD_MIN} to {FIELD_MAX}")
print(f"  Lens VMI: {len(unique_lens)} points from {LENS_MIN} to {LENS_MAX}")
print(f"  Kinetic energies: {len(unique_kes)} points from {KE_MIN} to {KE_MAX} eV")

# === STEP 2: Build Combination List ===
print("\nSTEP 2: Building combination list for energy resolution analysis...")

all_combinations = [(fg, lens, ke)
                   for fg in unique_fgs
                   for lens in unique_lens
                   for ke in unique_kes]

print(f"  Total combinations: {len(all_combinations)}")
print(f"  FG points: {len(unique_fgs)}, Lens points: {len(unique_lens)}, KE points: {len(unique_kes)}")

# === STEP 3: Generate Ionization Volume Array ===
print("\nSTEP 3: Setting up interaction volume...")
seq_ionization_position = np.zeros(shape=(15, 3))
face_center_array = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
edge_points_array = np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                              [-1,-1,1], [1,-1,-1], [-1,1,-1], [-1,-1,-1]])
seq_ionization_position[1:7] = face_center_array
seq_ionization_position[7:15] = edge_points_array

interaction_ranges_mm = resolve_interaction_volume_ranges()
if Interaction_volume:
    print(f"  Interaction volume ENABLED: {NUM_RUNS_PER_NODE} positions")
    print(f"  Scan ranges (mm): {interaction_ranges_mm}")
else:
    print("  Interaction volume DISABLED: single source position")

# === STEP 4: Energy Resolution Analysis (per interaction range) ===
print("\nSTEP 4: Running energy resolution analysis scan...")
print(f"  Particles per energy: {NUM_PARTICLES_PER_ENERGY}")
print(f"  Statistical repeats: {NUM_RUNS_PER_NODE}")
print(f"  Detector filter: x={DETECTOR_X_MM}±{DETECTOR_X_TOL_MM} mm")
print(f"  Interaction range nodes: {len(interaction_ranges_mm)}")

scan_summaries = {}
scan_records = []
scan_start_time = time.time()

for range_index, range_mm in enumerate(interaction_ranges_mm, start=1):
    print("\n" + "-" * 70)
    print(f"Range {range_index}/{len(interaction_ranges_mm)}: interaction_volume_range={range_mm:g} mm")
    print("-" * 70)

    range_processed_data = create_empty_processed_data(unique_fgs, unique_lens, unique_kes)
    range_start = time.time()
    range_token = _format_scan_token(range_mm)
    checkpoint_name = f"energy_resolution_direct_ivr_{range_token}"
    range_processed_data = Utilis.energy_resolution_analysis_direct(
        range_processed_data,
        all_combinations=all_combinations,
        source_position=SOURCE_POSITION,
        num_particles_per_energy=NUM_PARTICLES_PER_ENERGY,
        num_statistical_repeats=NUM_RUNS_PER_NODE,
        x_scan_range=(73.0, 166.0),
        bin_interval=BIN_INTERVAL,
        outside_region_width=OUTSIDE_REGION_WIDTH,
        batch_size=ENERGY_BATCH_SIZE,
        enable_memory_optimization=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        max_combo_retries=MAX_COMBO_RETRIES,
        require_full_particle_capture=REQUIRE_FULL_PARTICLE_CAPTURE,
        retry_backoff_s=RETRY_BACKOFF_S,
        timing_verbose=TIMING_VERBOSE,
        detector_x_mm=DETECTOR_X_MM,
        detector_x_tol_mm=DETECTOR_X_TOL_MM,
        detector_y_range_mm=DETECTOR_Y_RANGE_MM,
        detector_z_range_mm=DETECTOR_Z_RANGE_MM,
        gc_interval_combos=GC_INTERVAL_COMBOS,
        intraction_volume=Interaction_volume,
        ionization_volume_array_mm=seq_ionization_position * float(range_mm),
        checkpoint_name=checkpoint_name
    )
    range_elapsed_s = time.time() - range_start

    range_summary = build_energy_resolution_summary(
        range_processed_data,
        include_pair_payload=SAVE_PAIR_PAYLOAD_IN_SUMMARY,
        include_raw_payload=SAVE_RAW_POINT_PAYLOAD_IN_SUMMARY
    )
    (
        valid_points,
        total_points,
        unique_er_count,
        raw_nodes,
        raw_point_total,
        valid_runs,
        total_runs,
    ) = collect_summary_statistics(range_summary)
    print(f"  Range analysis completed in {range_elapsed_s:.1f}s ({range_elapsed_s/60:.1f} min)")
    print(f"  Valid energy-resolution points: {valid_points}/{total_points}")
    print(f"  Unique energy-resolution values: {unique_er_count}")
    print(f"  Raw ion payload nodes: {raw_nodes}/{total_points}, points={raw_point_total}")
    print(f"  Valid run samples: {valid_runs}/{total_runs}")
    if valid_points == 0 and raw_nodes == 0:
        print("  WARNING: No valid energy-resolution values were produced for this range.")

    range_summary_path = None
    if SAVE_RANGE_SUMMARY_PKL:
        range_summary_path = os.path.join(
            OUTPUT_DIR,
            f"energy_resolution_summary_ivr_{range_token}_{RUN_TAG}.pkl"
        )
        save_pickle_atomic(range_summary, range_summary_path)
        print(f"  Saved range summary: {range_summary_path}")
    else:
        print("  Range summary pickle skipped (SAVE_RANGE_SUMMARY_PKL=False)")

    range_csv_path = None
    if SAVE_RANGE_CSV:
        range_csv_path = os.path.join(
            OUTPUT_DIR,
            f"energy_resolution_ivr_{range_token}_{RUN_TAG}.csv"
        )
        range_csv_rows = save_energy_resolution_csv(range_summary, range_csv_path)
        print(f"  Saved range CSV: {range_csv_path} ({range_csv_rows} rows)")
    else:
        print("  Range CSV skipped (SAVE_RANGE_CSV=False)")

    full_path = None
    if SAVE_FULL_PROCESSED_DATA:
        full_path = os.path.join(
            OUTPUT_DIR,
            f"processed_data_full_ivr_{range_token}_{RUN_TAG}.pkl"
        )
        save_pickle_atomic(range_processed_data, full_path)
        print(f"  Saved full processed_data: {full_path}")

    scan_summaries[float(range_mm)] = range_summary
    scan_records.append({
        'interaction_volume_range_mm': float(range_mm),
        'checkpoint_name': checkpoint_name,
        'elapsed_seconds': float(range_elapsed_s),
        'valid_points': int(valid_points),
        'total_points': int(total_points),
        'valid_runs': int(valid_runs),
        'total_runs': int(total_runs),
        'raw_nodes': int(raw_nodes),
        'raw_points': int(raw_point_total),
        'summary_path': range_summary_path,
        'csv_path': range_csv_path,
        'full_processed_data_path': full_path
    })

    del range_processed_data
    gc.collect()

total_scan_elapsed = time.time() - scan_start_time
print(f"\nScan completed in {total_scan_elapsed:.1f} seconds ({total_scan_elapsed/60:.1f} minutes)")

# === STEP 5: Save Scan Results ===
print("\nSTEP 5: Saving scan-level outputs...")

scan_bundle = {
    'run_tag': RUN_TAG,
    'timestamp': datetime.now().isoformat(timespec='seconds'),
    'interaction_volume_enabled': bool(Interaction_volume),
    'interaction_volume_ranges_mm': [float(v) for v in interaction_ranges_mm],
    'num_particles_per_energy': int(NUM_PARTICLES_PER_ENERGY),
    'num_runs_per_node': int(NUM_RUNS_PER_NODE),
    'settings': {
        'field_range': [FIELD_MIN, FIELD_MAX, NUM_FIELD_POINTS],
        'lens_range': [LENS_MIN, LENS_MAX, NUM_LENS_POINTS],
        'ke_range': [KE_MIN, KE_MAX, NUM_KE_POINTS],
        'num_particles_per_energy': NUM_PARTICLES_PER_ENERGY,
        'num_runs_per_node': NUM_RUNS_PER_NODE,
        'require_full_particle_capture': REQUIRE_FULL_PARTICLE_CAPTURE,
        'detector': {
            'x_mm': DETECTOR_X_MM,
            'x_tol_mm': DETECTOR_X_TOL_MM,
            'y_range_mm': DETECTOR_Y_RANGE_MM,
            'z_range_mm': DETECTOR_Z_RANGE_MM
        }
    },
    'scan_records': scan_records,
    'scan_summaries': scan_summaries,
}
scan_bundle_path = None
if SAVE_SCAN_BUNDLE_PKL:
    scan_bundle_path = os.path.join(OUTPUT_DIR, f"interaction_volume_scan_bundle_{RUN_TAG}.pkl")
    save_pickle_atomic(scan_bundle, scan_bundle_path)
    print(f"  Scan bundle saved: {scan_bundle_path}")
else:
    print("  Scan bundle pickle skipped (SAVE_SCAN_BUNDLE_PKL=False)")

if SAVE_SCAN_MANIFEST_JSON:
    scan_manifest_path = os.path.join(OUTPUT_DIR, f"interaction_volume_scan_manifest_{RUN_TAG}.json")
    save_json_atomic({
        'run_tag': RUN_TAG,
        'interaction_volume_ranges_mm': [float(v) for v in interaction_ranges_mm],
        'scan_records': scan_records,
        'settings': {
            'field_range': [FIELD_MIN, FIELD_MAX, NUM_FIELD_POINTS],
            'lens_range': [LENS_MIN, LENS_MAX, NUM_LENS_POINTS],
            'ke_range': [KE_MIN, KE_MAX, NUM_KE_POINTS],
            'num_particles_per_energy': NUM_PARTICLES_PER_ENERGY,
            'num_runs_per_node': NUM_RUNS_PER_NODE,
            'require_full_particle_capture': REQUIRE_FULL_PARTICLE_CAPTURE,
            'detector': {
                'x_mm': DETECTOR_X_MM,
                'x_tol_mm': DETECTOR_X_TOL_MM,
                'y_range_mm': DETECTOR_Y_RANGE_MM,
                'z_range_mm': DETECTOR_Z_RANGE_MM
            }
        }
    }, scan_manifest_path)
    print(f"  Scan manifest saved: {scan_manifest_path}")
else:
    print("  Scan manifest JSON skipped (SAVE_SCAN_MANIFEST_JSON=False)")

if SAVE_SCAN_SUMMARY_CSV:
    scan_csv_path = os.path.join(OUTPUT_DIR, f"interaction_volume_scan_summary_{RUN_TAG}.csv")
    scan_csv_rows = save_interaction_scan_csv(scan_summaries, scan_csv_path)
    print(f"  Scan summary CSV saved: {scan_csv_path} ({scan_csv_rows} rows)")
else:
    print("  Scan summary CSV skipped (SAVE_SCAN_SUMMARY_CSV=False)")

plot_paths = []
if SAVE_SCAN_CURVE_PLOTS:
    plot_paths = plot_interaction_volume_scan_curves(
        scan_summaries=scan_summaries,
        output_dir=OUTPUT_DIR,
        run_tag=RUN_TAG,
        y_as_percent=True
    )
    print(f"  Scan curve plots saved: {len(plot_paths)} file(s)")
else:
    print("  Scan curve plots skipped (SAVE_SCAN_CURVE_PLOTS=False)")

print("\nRange summary:")
for record in scan_records:
    print(
        f"  IV range={record['interaction_volume_range_mm']:g} mm: "
        f"valid points {record['valid_points']}/{record['total_points']}, "
        f"valid runs {record['valid_runs']}/{record['total_runs']}, "
        f"time={record['elapsed_seconds']:.1f}s"
    )

# === STEP 6: Visualization ===
if ENABLE_VISUALIZATION:
    print("\nSTEP 6: Launching visualization...")
    backend_before = _get_matplotlib_backend_name()
    if _is_non_interactive_matplotlib_backend():
        print(f"  Current matplotlib backend is non-interactive: {backend_before}")
        if _ensure_interactive_matplotlib_backend():
            print(f"  Switched backend to interactive: {_get_matplotlib_backend_name()}")
        else:
            print("  Could not switch to an interactive backend; skip GUI launch.")
            print("  Use 'python view_results_v56.py' to open saved results later.")
    if not _is_non_interactive_matplotlib_backend():
        print("Interactive GUI with all interaction-volume-range curves...")
        try:
            visual_method = _normalize_visualization_method(VISUALIZATION_METHOD)
            launched = launch_interaction_range_gui(scan_summaries, method=visual_method)
            if not launched:
                print("  No plottable GUI data for selected method.")
        except Exception as e:
            print(f"Interactive GUI failed: {e}")
            print("  You can still inspect saved outputs with: python view_results_v56.py")
else:
    print("\nVisualization skipped (ENABLE_VISUALIZATION=False)")

# === STEP 7: Post-Run Cleanup ===
print("\nSTEP 7: Post-run cleanup...")
if CLEANUP_TEMP_FILES_AFTER_SUCCESS:
    removed_temp = clear_runtime_temp_files()
    print(f"  Temp files removed: {removed_temp}")
else:
    print("  Temp file cleanup skipped (CLEANUP_TEMP_FILES_AFTER_SUCCESS=False)")

if CLEANUP_CHECKPOINTS_AFTER_SUCCESS:
    removed_ckpt = clear_energy_resolution_checkpoints()
    print(f"  Checkpoint files removed: {removed_ckpt}")
else:
    print("  Checkpoint cleanup skipped (CLEANUP_CHECKPOINTS_AFTER_SUCCESS=False)")

# === CLEANUP ===
print("\n" + "=" * 70)
print("Workflow Complete!")
print("=" * 70)

try:
    import matplotlib.pyplot as plt
    plt.close('all')
except:
    pass

sys.exit(0)
