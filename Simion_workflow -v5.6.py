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
SAVE_SCAN_CURVE_PLOTS = True  # Save static multi-curve plots for each (FG, Lens)

# === SAVE OPTIONS ===
SAVE_FULL_PROCESSED_DATA = False  # Save only summary to avoid OOM
CLEAR_STALE_CHECKPOINTS = False  # Keep checkpoints for resume capability
SAVE_HEAVY_PAYLOAD_IN_SUMMARY = False  # Include raw_ion_points/dr_over_r_pairs in summary pickle
CLEANUP_CHECKPOINTS_AFTER_SUCCESS = True
CLEANUP_TEMP_FILES_AFTER_SUCCESS = True

# === PARAMETER RANGES ===
KE_MIN = 1
KE_MAX = 15.5
NUM_KE_POINTS = 5

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
CHECKPOINT_INTERVAL = 100
GC_INTERVAL_COMBOS = 50

# === INTERACTION VOLUME (Source Motion) ===
Interaction_volume = True
Interaction_volume_range = 1  # mm
SCAN_INTERACTION_VOLUME_RANGE = True
INTERACTION_VOLUME_RANGE_SCAN = [0.0, 0.5, 1.0, 1.5]

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

def build_energy_resolution_summary(processed_data, include_heavy_payload=False):
    """Extract v5.4-compatible summary structure from processed data."""
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
                global_data = ke_data.get('global', {}) if isinstance(ke_data.get('global', {}), dict) else {}
                global_summary = {
                    'fwhm': _to_builtin(global_data.get('fwhm')),
                    'energy_resolution': _to_builtin(global_data.get('energy_resolution')),
                    'max_r': _to_builtin(global_data.get('max_r')),
                    'generated_particles': _to_builtin(global_data.get('generated_particles')),
                    'detected_particles': _to_builtin(global_data.get('detected_particles')),
                    'valid': _to_builtin(global_data.get('valid')),
                    'failure_reason': _to_builtin(global_data.get('failure_reason'))
                }
                optional_fields = [
                    'fwhm_mean', 'fwhm_variance', 'fwhm_std', 'fwhm_runs',
                    'energy_resolution_mean', 'energy_resolution_variance', 'energy_resolution_std', 'energy_resolution_runs',
                    'max_r_mean', 'max_r_variance', 'max_r_std', 'max_r_runs',
                    'r_max_all_points', 'r_max_all_points_runs', 'all_point_count_runs',
                    'generated_particles_per_run',
                    'detected_particles_runs', 'valid_run_count', 'total_runs',
                    'pair_count', 'pair_count_runs',
                    'raw_point_count', 'raw_point_format',
                    'failure_reasons',
                    'count_check_passed', 'plot_marker', 'plot_skip', 'pipeline_stage'
                ]
                if include_heavy_payload:
                    optional_fields.extend([
                        'dr_values', 'r_values', 'dr_over_r_values', 'dr_over_r_pairs',
                        'raw_ion_points_yz',
                    ])
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

def count_valid_energy_points(data):
    valid = 0
    total = 0
    unique_er_values = set()
    for fg_data in data.values():
        for lens_data in fg_data.values():
            for ke_data in lens_data.values():
                total += 1
                global_data = ke_data.get('global', {}) if isinstance(ke_data, dict) else {}
                er = global_data.get('energy_resolution')
                if er is None:
                    er = global_data.get('energy_resolution_mean')
                if er is None:
                    continue
                try:
                    if np.isnan(er):
                        continue
                except Exception:
                    pass
                valid += 1
                try:
                    unique_er_values.add(round(float(er), 6))
                except Exception:
                    pass
    return valid, total, len(unique_er_values)

def count_raw_payload_points(summary_data):
    nodes_with_raw = 0
    total_points = 0
    for fg_data in summary_data.values():
        for lens_data in fg_data.values():
            for ke_data in lens_data.values():
                global_data = ke_data.get('global', {}) if isinstance(ke_data, dict) else {}
                raw_points = global_data.get('raw_ion_points_yz', [])
                if isinstance(raw_points, list) and len(raw_points) > 0:
                    nodes_with_raw += 1
                    total_points += len(raw_points)
                    continue
                raw_point_count = global_data.get('raw_point_count')
                if isinstance(raw_point_count, (int, np.integer)) and int(raw_point_count) > 0:
                    nodes_with_raw += 1
                    total_points += int(raw_point_count)
    return nodes_with_raw, total_points

def count_valid_run_samples(summary_data):
    valid_runs = 0
    total_runs = 0
    for fg_data in summary_data.values():
        for lens_data in fg_data.values():
            for ke_data in lens_data.values():
                global_data = ke_data.get('global', {}) if isinstance(ke_data, dict) else {}
                point_total_runs = global_data.get('total_runs')
                point_valid_runs = global_data.get('valid_run_count')
                if point_total_runs is None or point_valid_runs is None:
                    total_runs += 1
                    if global_data.get('valid'):
                        valid_runs += 1
                    continue
                total_runs += int(point_total_runs)
                valid_runs += int(point_valid_runs)
    return valid_runs, total_runs

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

def resolve_interaction_volume_ranges():
    if not Interaction_volume:
        return [0.0]
    if SCAN_INTERACTION_VOLUME_RANGE:
        if not isinstance(INTERACTION_VOLUME_RANGE_SCAN, (list, tuple, np.ndarray)):
            raise ValueError("INTERACTION_VOLUME_RANGE_SCAN must be a list/tuple/ndarray")
        ranges = []
        for value in INTERACTION_VOLUME_RANGE_SCAN:
            numeric = _safe_float(value)
            if numeric is None:
                continue
            ranges.append(float(numeric))
    else:
        numeric = _safe_float(Interaction_volume_range)
        ranges = [float(numeric if numeric is not None else 0.0)]
    unique_ranges = sorted(set(ranges))
    if not unique_ranges:
        raise ValueError("No valid interaction volume ranges resolved")
    return unique_ranges

def launch_interaction_range_gui(scan_summaries):
    import view_energy_resolution_results as viewer
    range_values = _sorted_keys(scan_summaries.keys())
    if len(range_values) <= 1:
        return launch_field_gradient_gui(scan_summaries[range_values[0]])

    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    markers = ['o', 's', '^', 'd', 'v', 'P', 'X', '*', 'h']
    method_entries = []
    for idx, range_value in enumerate(range_values):
        method_entries.append({
            'key': f'ivr_{idx}',
            'label': f'IV range = {range_value:g} mm',
            'summary': scan_summaries[range_value],
            'color': colors[idx % len(colors)],
            'marker': markers[idx % len(markers)]
        })
    return viewer.launch_multi_method_gui(
        method_entries,
        title_prefix='SIMION Workflow v5.6 - Interaction Volume Range Scan',
        y_as_percent=True
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
                        label=f'IV range={range_value:g} mm'
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

def launch_field_gradient_gui(summary_data):
    """Launch interactive GUI for exploring energy resolution data (uses external viewer)."""
    import view_energy_resolution_results as viewer
    return viewer.launch_energy_resolution_gui(
        summary_data=summary_data,
        method='rmax',  # Visualization method
        origin_y=SOURCE_POSITION[1],
        origin_z=SOURCE_POSITION[2],
        y_as_percent=True,
        title_prefix='SIMION Workflow v5.6'
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
        include_heavy_payload=SAVE_HEAVY_PAYLOAD_IN_SUMMARY
    )
    valid_points, total_points, unique_er_count = count_valid_energy_points(range_summary)
    raw_nodes, raw_point_total = count_raw_payload_points(range_summary)
    valid_runs, total_runs = count_valid_run_samples(range_summary)
    print(f"  Range analysis completed in {range_elapsed_s:.1f}s ({range_elapsed_s/60:.1f} min)")
    print(f"  Valid energy-resolution points: {valid_points}/{total_points}")
    print(f"  Unique energy-resolution values: {unique_er_count}")
    print(f"  Raw ion payload nodes: {raw_nodes}/{total_points}, points={raw_point_total}")
    print(f"  Valid run samples: {valid_runs}/{total_runs}")
    if valid_points == 0 and raw_nodes == 0:
        print("  WARNING: No valid energy-resolution values were produced for this range.")

    range_summary_path = os.path.join(
        OUTPUT_DIR,
        f"energy_resolution_summary_ivr_{range_token}_{RUN_TAG}.pkl"
    )
    save_pickle_atomic(range_summary, range_summary_path)
    print(f"  Saved range summary: {range_summary_path}")

    range_csv_path = os.path.join(
        OUTPUT_DIR,
        f"energy_resolution_ivr_{range_token}_{RUN_TAG}.csv"
    )
    range_csv_rows = save_energy_resolution_csv(range_summary, range_csv_path)
    print(f"  Saved range CSV: {range_csv_path} ({range_csv_rows} rows)")

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
    'scan_records': scan_records,
    'scan_summaries': scan_summaries,
}
scan_bundle_path = os.path.join(OUTPUT_DIR, f"interaction_volume_scan_bundle_{RUN_TAG}.pkl")
save_pickle_atomic(scan_bundle, scan_bundle_path)
print(f"  Scan bundle saved: {scan_bundle_path}")

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

scan_csv_path = os.path.join(OUTPUT_DIR, f"interaction_volume_scan_summary_{RUN_TAG}.csv")
scan_csv_rows = save_interaction_scan_csv(scan_summaries, scan_csv_path)
print(f"  Scan summary CSV saved: {scan_csv_path} ({scan_csv_rows} rows)")

plot_paths = []
if SAVE_SCAN_CURVE_PLOTS:
    plot_paths = plot_interaction_volume_scan_curves(
        scan_summaries=scan_summaries,
        output_dir=OUTPUT_DIR,
        run_tag=RUN_TAG,
        y_as_percent=True
    )
    print(f"  Scan curve plots saved: {len(plot_paths)} file(s)")

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
    print("Interactive GUI with all interaction-volume-range curves...")
    try:
        launched = launch_interaction_range_gui(scan_summaries)
        if not launched:
            raise RuntimeError("No plottable data in GUI helper")
    except Exception as e:
        print(f"Interactive GUI failed: {e}")
        import traceback
        traceback.print_exc()
        first_range = _sorted_keys(scan_summaries.keys())[0]
        print(f"Falling back to Utilis heatmap slider (IV range={first_range:g} mm)...")
        try:
            Utilis.plot_heatmap_all_fg(scan_summaries[first_range])
        except Exception as e2:
            print(f"Fallback slider also failed: {e2}")
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
