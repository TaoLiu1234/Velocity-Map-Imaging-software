"""
Optimized SIMION Workflow - v5.3
- SAME FUNCTIONALITY as v5 (uses Utilis.energy_resolution_analysis)
- FIXED: Proper result saving without memory overflow
- FIXED: GUI with slider for field gradient adjustment
- Shows energy resolution vs lens VMI relationship
- Enhanced memory management for large datasets
"""

import os
import csv
import gc
import glob
import pickle
import traceback
import Utilis
import numpy as np
import time
import psutil
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# === RUNTIME / OUTPUT CONTROL ===
# Always run relative to this script's folder (prevents saving to C:\Windows\System32 by accident)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Centralized output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_v5.4")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# If True: open interactive plots (will BLOCK until you close the windows).
# If False: skip visualization so the script finishes and exits normally.
ENABLE_VISUALIZATION = True
# Visualization metric method for Step 6.
# Options: "rmax", "abs", "sqrt", "all"
VIEWER_METHOD = "rmax"

# Save only analysis essentials by default to avoid OOM when dumping giant nested structures.
SAVE_FULL_PROCESSED_DATA = False
# Keep checkpoints by default so long runs can resume after crashes/interruption.
# Set True only when you intentionally want a clean restart.
CLEAR_STALE_CHECKPOINTS = False

# === CONTROL FLAGS ===
SKIP_INITIAL_WORKFLOW = True   # Skip front simulation stage, go directly to energy resolution analysis
SKIP_FOCUS_FILTERING = True    # Analyze all parameter combinations directly
USE_PARALLEL_SIMION = False    # SIMION is kept serial to avoid race-condition failures

KE_MIN = 1
KE_MAX = 30
NUM_KE_POINTS = 15

electron_energy_sequence = np.linspace(KE_MIN, KE_MAX, NUM_KE_POINTS)
if KE_MIN >= KE_MAX:
    electron_energy_sequence = np.array([KE_MIN])

Theta = 0 * 2 * np.pi / 360

FIELD_MIN = 50
FIELD_MAX = 300
NUM_POINTS = 6

LENS_MIN = 1
LENS_MAX = 3
NUM_LENS_POINTS = 6

NUM_GROUPS = 100

X_SCAN_RANGE = (73.0, 166.0)
X_STEP = 0.25
FOCUS_CRITERION = 'z'

INPUT_FILE = 'out.txt'
OUTPUT_FILENAME_LUA = "WORKING_TITLE_tao.lua"
OUTPUT_FILENAME_FLY2 = 'WORKING_TITLE_tao.fly2'
IOB_FILE = "WORKING_TITLE_tao.iob"
OUT_FILE = "out.txt"

# Energy resolution settings
NUM_PARTICLES_PER_ENERGY = 91
SOURCE_POSITION = (199, 0, 0)
BIN_INTERVAL = 0.01
OUTSIDE_REGION_WIDTH = 2
TOLERABLE_OFFSET = 2.0
ENERGY_BATCH_SIZE = 25
MAX_COMBO_RETRIES = 3
REQUIRE_FULL_PARTICLE_CAPTURE = True
RETRY_BACKOFF_S = 0.5
TIMING_VERBOSE = True

SAVE_PER_RUN_SUMMARY = True     # Save each repeat before aggregation
CHECKPOINT_INTERVAL = 100       # Save checkpoint shard every N attempted combinations
GC_INTERVAL_COMBOS = 50         # Run gc.collect() every N attempted combinations
Intraction_volume = True
Interaction_volume_range = 1 # mm

if Intraction_volume == True:
    NUM_RUNS_PER_NODE = 15      # Repeat count for each (FG, Lens, KE) node
else:
    NUM_RUNS_PER_NODE = 1       # Repeat count for each (FG, Lens, KE) node
# Detector acceptance filter (applied before particle-count matching and dr/r pairing)
DETECTOR_X_MM = 73.0
DETECTOR_X_TOL_MM = 0.5
DETECTOR_Y_RANGE_MM = (-35.0, 35.0)
DETECTOR_Z_RANGE_MM = (-35.0, 35.0)

if NUM_PARTICLES_PER_ENERGY <= 0:
    raise ValueError("NUM_PARTICLES_PER_ENERGY must be > 0")
if NUM_RUNS_PER_NODE <= 0:
    raise ValueError("NUM_RUNS_PER_NODE must be > 0")

# ============================================================================
# DYNAMIC RESOURCE MANAGEMENT (Enhanced for Large Datasets)
# ============================================================================

class DynamicResourceManager:
    """
    Enhanced resource manager for large datasets.
    Monitors memory and temp files, performs cleanup automatically.
    """
    
    # Thresholds (in MB)
    MEMORY_WARNING = 2000      # Start monitoring closely
    MEMORY_HIGH = 4000         # Trigger cleanup
    MEMORY_CRITICAL = 6000     # Aggressive cleanup
    MEMORY_EMERGENCY = 8000    # Emergency cleanup
    
    # Cleanup intervals
    TEMP_CLEANUP_INTERVAL = 5  # Cleanup temp files every N simulations
    MEMORY_CHECK_INTERVAL = 3  # Check memory every N simulations
    
    def __init__(self):
        self.sim_count = 0
        self.total_cleaned = 0
        self.cleanup_count = 0
        self.start_memory = self.get_memory_mb()
        
    def get_memory_mb(self):
        """Get current memory usage in MB."""
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except:
            return -1
    
    def get_available_memory_mb(self):
        """Get available system memory in MB."""
        try:
            return psutil.virtual_memory().available / 1024 / 1024
        except:
            return -1
    
    def cleanup_temp_files(self):
        """Clean all temporary files."""
        patterns = [
            'temp_er_*.fly2', 'temp_er_*.lua', 'temp_er_*.txt',
            'temp_out_ke_*.txt', '*.tmp', 'trj*.tmp',
            'WORKING_TITLE_tao_ke_*.lua', 'energy_resolution_out.txt',
            'trapcheck.info', 'WORKING_TITLE_energy_resolution_*.fly2',
            'WORKING_TITLE_energy_resolution_*.lua'
        ]
        count = 0
        for pattern in patterns:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                    count += 1
                except:
                    pass
        self.total_cleaned += count
        return count
    
    def garbage_collect(self, level=0):
        """
        Perform garbage collection.
        level 0: Normal
        level 1: Thorough
        level 2: Aggressive
        """
        if level == 0:
            gc.collect()
        elif level == 1:
            gc.collect(0)
            gc.collect(1)
            gc.collect()
        else:  # level 2 - aggressive
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            gc.collect()
    
    def check_and_cleanup(self, force=False, verbose=True):
        """
        Check resources and perform cleanup if needed.
        Returns: (cleaned_files, memory_mb, cleanup_performed)
        """
        self.sim_count += 1
        mem_mb = self.get_memory_mb()
        cleaned = 0
        cleanup_performed = False
        
        # Determine cleanup level based on memory
        if mem_mb > self.MEMORY_EMERGENCY or force:
            # Emergency cleanup
            cleaned = self.cleanup_temp_files()
            self.garbage_collect(level=2)
            cleanup_performed = True
            self.cleanup_count += 1
            if verbose and cleaned > 0:
                print(f"    [EMERGENCY CLEANUP] Mem: {mem_mb:.0f}MB, Cleaned: {cleaned} files")
                
        elif mem_mb > self.MEMORY_CRITICAL:
            # Aggressive cleanup
            cleaned = self.cleanup_temp_files()
            self.garbage_collect(level=2)
            cleanup_performed = True
            self.cleanup_count += 1
            if verbose:
                print(f"    [CRITICAL CLEANUP] Mem: {mem_mb:.0f}MB, Cleaned: {cleaned} files")
                
        elif mem_mb > self.MEMORY_HIGH:
            # Normal cleanup
            cleaned = self.cleanup_temp_files()
            self.garbage_collect(level=1)
            cleanup_performed = True
            self.cleanup_count += 1
            if verbose:
                print(f"    [HIGH MEM CLEANUP] Mem: {mem_mb:.0f}MB, Cleaned: {cleaned} files")
                
        elif self.sim_count % self.TEMP_CLEANUP_INTERVAL == 0:
            # Periodic temp file cleanup
            cleaned = self.cleanup_temp_files()
            if cleaned > 0:
                self.garbage_collect(level=0)
                cleanup_performed = True
                
        elif self.sim_count % self.MEMORY_CHECK_INTERVAL == 0:
            # Periodic light garbage collection
            self.garbage_collect(level=0)
        
        return cleaned, mem_mb, cleanup_performed
    
    def get_status(self):
        """Get resource manager status."""
        mem_mb = self.get_memory_mb()
        avail_mb = self.get_available_memory_mb()
        return {
            'current_memory_mb': mem_mb,
            'available_memory_mb': avail_mb,
            'start_memory_mb': self.start_memory,
            'memory_increase_mb': mem_mb - self.start_memory,
            'sim_count': self.sim_count,
            'total_files_cleaned': self.total_cleaned,
            'cleanup_count': self.cleanup_count
        }
    
    def print_status(self):
        """Print resource status."""
        status = self.get_status()
        print(f"\n  Resource Status:")
        print(f"    Memory: {status['current_memory_mb']:.0f}MB (started: {status['start_memory_mb']:.0f}MB, +{status['memory_increase_mb']:.0f}MB)")
        print(f"    Available: {status['available_memory_mb']:.0f}MB")
        print(f"    Simulations: {status['sim_count']}, Cleanups: {status['cleanup_count']}, Files cleaned: {status['total_files_cleaned']}")

# Global resource manager
resource_mgr = DynamicResourceManager()


def _normalize_numeric_key(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    return value


def _safe_float(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return num


def _sorted_keys(values):
    def sort_key(item):
        if isinstance(item, (int, float, np.integer, np.floating)):
            return (0, float(item))
        try:
            return (0, float(item))
        except (TypeError, ValueError):
            return (1, str(item))
    return sorted(values, key=sort_key)


def build_energy_resolution_summary(data):
    summary = {}
    for fg, fg_data in data.items():
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
                    'dr_values', 'r_values', 'dr_over_r_values', 'dr_over_r_pairs',
                    'raw_ion_points_yz', 'raw_point_count', 'raw_point_format',
                    'failure_reasons'
                ]
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


def discard_local_trajectory_data(data):
    removed_slots = 0
    for fg_data in data.values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            for ke_data in lens_data.values():
                if isinstance(ke_data, dict) and 'local' in ke_data:
                    local_obj = ke_data.get('local')
                    if isinstance(local_obj, dict):
                        local_obj.clear()
                    ke_data['local'] = {}
                    removed_slots += 1
    return removed_slots


def count_valid_energy_points(summary_data):
    valid = 0
    total = 0
    unique_er_values = set()
    for fg_data in summary_data.values():
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
    return nodes_with_raw, total_points


def save_pickle_atomic(data, output_file):
    temp_file = f"{output_file}.tmp"
    with open(temp_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, output_file)


def _serialize_numeric_runs(values):
    if not isinstance(values, (list, tuple, np.ndarray)):
        return ''
    encoded = []
    for value in values:
        numeric = _safe_float(value)
        encoded.append('' if numeric is None else f"{numeric:.8g}")
    return '|'.join(encoded)


def save_energy_resolution_csv(summary_data, output_file):
    row_count = 0
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
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


def create_empty_processed_data(field_gradients, lens_values, ke_values):
    template = {}
    for fg in field_gradients:
        fg_key = _normalize_numeric_key(fg)
        if fg_key not in template:
            template[fg_key] = {}
        for lens_vmi in lens_values:
            lens_key = _normalize_numeric_key(lens_vmi)
            if lens_key not in template[fg_key]:
                template[fg_key][lens_key] = {}
            for ke in ke_values:
                ke_key = _normalize_numeric_key(ke)
                template[fg_key][lens_key][ke_key] = {'global': {}, 'local': {}}
    return template


def aggregate_energy_resolution_runs(run_summaries):
    if not run_summaries:
        return {}

    aggregated = {}
    total_runs = len(run_summaries)
    fg_keys = set()
    for summary in run_summaries:
        fg_keys.update(summary.keys())

    for fg in _sorted_keys(fg_keys):
        lens_keys = set()
        for summary in run_summaries:
            fg_data = summary.get(fg, {})
            if isinstance(fg_data, dict):
                lens_keys.update(fg_data.keys())
        if not lens_keys:
            continue
        aggregated[fg] = {}

        for lens_vmi in _sorted_keys(lens_keys):
            ke_keys = set()
            for summary in run_summaries:
                lens_data = summary.get(fg, {}).get(lens_vmi, {})
                if isinstance(lens_data, dict):
                    ke_keys.update(lens_data.keys())
            if not ke_keys:
                continue
            aggregated[fg][lens_vmi] = {}

            for ke in _sorted_keys(ke_keys):
                er_runs = []
                fwhm_runs = []
                max_r_runs = []
                detected_runs = []
                generated_runs = []
                valid_flags = []
                failure_reasons = []

                for summary in run_summaries:
                    global_data = summary.get(fg, {}).get(lens_vmi, {}).get(ke, {}).get('global', {})
                    if not isinstance(global_data, dict):
                        global_data = {}

                    er_value = _safe_float(global_data.get('energy_resolution', global_data.get('energy_resolution_mean')))
                    fwhm_value = _safe_float(global_data.get('fwhm', global_data.get('fwhm_mean')))
                    max_r_value = _safe_float(global_data.get('max_r', global_data.get('max_r_mean')))
                    detected_value = _safe_float(global_data.get('detected_particles'))
                    generated_value = _safe_float(global_data.get('generated_particles'))

                    er_runs.append(er_value)
                    fwhm_runs.append(fwhm_value)
                    max_r_runs.append(max_r_value)
                    detected_runs.append(detected_value)
                    if generated_value is not None:
                        generated_runs.append(generated_value)

                    explicit_valid = global_data.get('valid')
                    run_valid = bool(explicit_valid) if explicit_valid is not None else (er_value is not None)
                    if er_value is None:
                        run_valid = False
                    valid_flags.append(run_valid)

                    reason = global_data.get('failure_reason')
                    if reason:
                        failure_reasons.append(str(reason))

                valid_er = [v for v in er_runs if v is not None]
                valid_fwhm = [v for v in fwhm_runs if v is not None]
                valid_max_r = [v for v in max_r_runs if v is not None]
                valid_detected = [v for v in detected_runs if v is not None]

                er_mean = float(np.mean(valid_er)) if valid_er else None
                er_var = float(np.var(valid_er)) if valid_er else None
                er_std = float(np.sqrt(er_var)) if er_var is not None else None

                fwhm_mean = float(np.mean(valid_fwhm)) if valid_fwhm else None
                fwhm_var = float(np.var(valid_fwhm)) if valid_fwhm else None
                fwhm_std = float(np.sqrt(fwhm_var)) if fwhm_var is not None else None

                max_r_mean = float(np.mean(valid_max_r)) if valid_max_r else None
                max_r_var = float(np.var(valid_max_r)) if valid_max_r else None
                max_r_std = float(np.sqrt(max_r_var)) if max_r_var is not None else None

                detected_mean = int(round(float(np.mean(valid_detected)))) if valid_detected else None
                generated_particles = int(round(float(np.mean(generated_runs)))) if generated_runs else None

                valid_run_count = int(sum(1 for flag in valid_flags if flag))
                unique_failure_reasons = sorted(set(failure_reasons))
                failure_reason = None
                if valid_run_count < total_runs:
                    if unique_failure_reasons:
                        failure_reason = '; '.join(unique_failure_reasons[:3])
                    else:
                        failure_reason = f"{total_runs - valid_run_count}/{total_runs} runs invalid"

                aggregated[fg][lens_vmi][ke] = {
                    'global': {
                        'fwhm': fwhm_mean,
                        'fwhm_mean': fwhm_mean,
                        'fwhm_variance': fwhm_var,
                        'fwhm_std': fwhm_std,
                        'fwhm_runs': fwhm_runs,
                        'energy_resolution': er_mean,
                        'energy_resolution_mean': er_mean,
                        'energy_resolution_variance': er_var,
                        'energy_resolution_std': er_std,
                        'energy_resolution_runs': er_runs,
                        'max_r': max_r_mean,
                        'max_r_mean': max_r_mean,
                        'max_r_variance': max_r_var,
                        'max_r_std': max_r_std,
                        'max_r_runs': max_r_runs,
                        'generated_particles': generated_particles,
                        'detected_particles': detected_mean,
                        'detected_particles_runs': detected_runs,
                        'valid': valid_run_count > 0,
                        'valid_run_count': valid_run_count,
                        'total_runs': total_runs,
                        'failure_reason': failure_reason,
                        'failure_reasons': unique_failure_reasons
                    },
                    'local': {}
                }

    return aggregated


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


def clear_energy_resolution_checkpoints():
    checkpoint_files = []
    checkpoint_patterns = [
        'energy_resolution_checkpoint.pkl',
        'energy_resolution_direct_checkpoint.pkl',
        'energy_resolution_checkpoint_merged.pkl',
        'energy_resolution_direct_checkpoint_merged.pkl',
        'energy_resolution_checkpoint_part_*.pkl',
        'energy_resolution_direct_checkpoint_part_*.pkl'
    ]
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(pattern))

    removed = 0
    for checkpoint_file in sorted(set(checkpoint_files)):
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                removed += 1
                print(f"Removed stale checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"Warning: Could not remove {checkpoint_file}: {e}")
    return removed


def launch_field_gradient_gui(summary_data):
    import view_energy_resolution_results as viewer
    return viewer.launch_energy_resolution_gui(
        summary_data=summary_data,
        method=VIEWER_METHOD,
        origin_y=SOURCE_POSITION[1],
        origin_z=SOURCE_POSITION[2],
        y_as_percent=True,
        title_prefix='SIMION Workflow v5.4'
    )

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

print("=" * 70)
print("SIMION Workflow v5.3 - Fixed Version with Proper Result Saving")
print("=" * 70)
print(f"SKIP_INITIAL_WORKFLOW: {SKIP_INITIAL_WORKFLOW}")
print(f"SKIP_FOCUS_FILTERING: {SKIP_FOCUS_FILTERING}")
print(f"NUM_PARTICLES_PER_ENERGY: {NUM_PARTICLES_PER_ENERGY}")
print(f"NUM_RUNS_PER_NODE: {NUM_RUNS_PER_NODE}")
print(f"CHECKPOINT_INTERVAL: {CHECKPOINT_INTERVAL}")
print(f"GC_INTERVAL_COMBOS: {GC_INTERVAL_COMBOS}")
print(f"MAX_COMBO_RETRIES: {MAX_COMBO_RETRIES}")
print(f"REQUIRE_FULL_PARTICLE_CAPTURE: {REQUIRE_FULL_PARTICLE_CAPTURE}")
print(
    f"Detector acceptance: x={DETECTOR_X_MM}±{DETECTOR_X_TOL_MM} mm, "
    f"y={DETECTOR_Y_RANGE_MM} mm, z={DETECTOR_Z_RANGE_MM} mm"
)
print(f"Working directory: {SCRIPT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Run tag: {RUN_TAG}")
print(f"Initial Memory: {resource_mgr.get_memory_mb():.0f}MB")

workflow_start = time.time()

# Initial cleanup
cleaned = resource_mgr.cleanup_temp_files()
if cleaned > 0:
    print(f"Cleaned {cleaned} temp files from previous run")

if CLEAR_STALE_CHECKPOINTS:
    clear_energy_resolution_checkpoints()

# Step 1: Generate VMI parameters
print("\nStep 1: Computing VMI parameters...")
param = Utilis.compute_vmi_parameters(
    field_min=FIELD_MIN, field_max=FIELD_MAX, num_points=NUM_POINTS,
    lens_min=LENS_MIN, lens_max=LENS_MAX, num_lens_points=NUM_LENS_POINTS,
    save_to_file=False, mode='velocity_imaging'
)

unique_fgs = np.unique(param['field_gradient'])
unique_lens = np.unique(param['lens_VMI'])
total_combinations = len(unique_fgs) * len(unique_lens) * len(electron_energy_sequence)
print(f"  Parameter space: {len(unique_fgs)} FG x {len(unique_lens)} Lens x {len(electron_energy_sequence)} KE = {total_combinations} combinations")
if len(unique_lens) < 2:
    print("  WARNING: Only one Lens VMI value detected. 'energy resolution vs lens VMI' curve will be degenerate.")

if not SKIP_INITIAL_WORKFLOW:
    files_to_clear = [OUT_FILE]
    if INPUT_FILE != OUT_FILE:
        files_to_clear.append(INPUT_FILE)
    Utilis.clear_file_contents(*files_to_clear)

    print("\nStep 2: Running SIMION simulations...")
    expected_per_ke = len(unique_fgs) * len(unique_lens)
    print(f"  Expected simulations per KE: {expected_per_ke}")
    if USE_PARALLEL_SIMION:
        print("  WARNING: Parallel SIMION mode may be unstable due shared Lua file contention.")
    else:
        print("  Using SAFE sequential SIMION mode for reproducible run counts.")
    for ke_idx, ke in enumerate(electron_energy_sequence):
        print(f"  [{ke_idx+1}/{len(electron_energy_sequence)}] Processing KE = {ke:.2f} eV...")
        
        Utilis.generate_particles_fly2(
            num_groups=NUM_GROUPS, 
            filename=OUTPUT_FILENAME_FLY2,
            x_range=(-0.5, 0.5), 
            y_range=(-0.5, 0.5), 
            z_range=(-0.5, 0.5),
            ke=ke, 
            theta=Theta
        )
        
        if USE_PARALLEL_SIMION:
            run_stats = Utilis.run_optimized_simulations_with_ke_parallel(
                param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE
            )
        else:
            run_stats = Utilis.run_optimized_simulations_with_ke(
                param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE
            )

        if isinstance(run_stats, dict):
            print(
                f"    SIMION runs: requested={run_stats.get('requested', expected_per_ke)}, "
                f"successful={run_stats.get('successful', 0)}, failed={run_stats.get('failed', 0)}"
            )
        
        # Dynamic cleanup after each KE
        resource_mgr.check_and_cleanup(force=True)

    Utilis.delete_temp_files()

    print("\nStep 3: Processing simulation data...")
    processed_data = Utilis.process_data(
        x_range=X_SCAN_RANGE,
        file_path=INPUT_FILE,
        focus_axis=FOCUS_CRITERION,
        fly2_file=OUTPUT_FILENAME_FLY2
    )
    print("Initial workflow completed.")
else:
    print("\nStep 2-3: SKIPPED (SKIP_INITIAL_WORKFLOW=True)")
    processed_data = create_empty_processed_data(unique_fgs, unique_lens, electron_energy_sequence)

resource_mgr.check_and_cleanup(force=True)

# ============================================================================
# ENERGY RESOLUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Energy Resolution Analysis")
print("=" * 70)

# Build complete parameter-node list once
all_combinations = []
for fg in unique_fgs:
    for lens_vmi in unique_lens:
        for ke in electron_energy_sequence:
            all_combinations.append((_normalize_numeric_key(fg), _normalize_numeric_key(lens_vmi), _normalize_numeric_key(ke)))

print(f"Total unique parameter nodes: {len(all_combinations)}")
print(
    f"Statistical repeats per node: {NUM_RUNS_PER_NODE} "
    f"(generated particles per node = {NUM_PARTICLES_PER_ENERGY} x {NUM_RUNS_PER_NODE})"
)

run_summaries = []

# Checkpoints are already optionally cleared above via CLEAR_STALE_CHECKPOINTS.
# Do not clear unconditionally here, otherwise resume/breakpoint never works.
#---------------------------
# generate position array for ionization volume
# the seqence is 1 center, 6 face center, 8 edges
seq_ionization_position = np.zeros(shape = (15,3))
face_center_array = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
edge_points_array = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,1],[1,-1,-1],[-1,1,-1],[-1,-1,-1]])
seq_ionization_position[1:7] = face_center_array
seq_ionization_position[7:15] = edge_points_array
#---------------------------
if SKIP_FOCUS_FILTERING:
    print("SKIP_FOCUS_FILTERING=True: Analyzing ALL combinations directly")
    processed_data = create_empty_processed_data(unique_fgs, unique_lens, electron_energy_sequence)
    processed_data = Utilis.energy_resolution_analysis_direct(
        processed_data,
        all_combinations=all_combinations,
        source_position=SOURCE_POSITION,
        num_particles_per_energy=NUM_PARTICLES_PER_ENERGY,
        num_statistical_repeats=NUM_RUNS_PER_NODE,
        x_scan_range=X_SCAN_RANGE,
        bin_interval=BIN_INTERVAL,
        outside_region_width=OUTSIDE_REGION_WIDTH,
        batch_size=ENERGY_BATCH_SIZE,
        enable_memory_optimization=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        gc_interval_combos=GC_INTERVAL_COMBOS,
        max_combo_retries=MAX_COMBO_RETRIES,
        require_full_particle_capture=REQUIRE_FULL_PARTICLE_CAPTURE,
        retry_backoff_s=RETRY_BACKOFF_S,
        timing_verbose=TIMING_VERBOSE,
        detector_x_mm=DETECTOR_X_MM,
        detector_x_tol_mm=DETECTOR_X_TOL_MM,
        detector_y_range_mm=DETECTOR_Y_RANGE_MM,
        detector_z_range_mm=DETECTOR_Z_RANGE_MM,
        intraction_volume = Intraction_volume,
        ionization_volume_array_mm = seq_ionization_position*Interaction_volume_range
    )
else:
    print("SKIP_FOCUS_FILTERING=False: Using focus-filtered analysis (single statistical run)")
    processed_data = Utilis.energy_resolution_analysis(
        processed_data,
        tolerable_offset=TOLERABLE_OFFSET,
        source_position=SOURCE_POSITION,
        num_particles_per_energy=NUM_PARTICLES_PER_ENERGY,
        num_statistical_repeats=NUM_RUNS_PER_NODE,
        x_scan_range=X_SCAN_RANGE,
        bin_interval=BIN_INTERVAL,
        outside_region_width=OUTSIDE_REGION_WIDTH,
        batch_size=ENERGY_BATCH_SIZE,
        enable_memory_optimization=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        gc_interval_combos=GC_INTERVAL_COMBOS,
        detector_x_mm=DETECTOR_X_MM,
        detector_x_tol_mm=DETECTOR_X_TOL_MM,
        detector_y_range_mm=DETECTOR_Y_RANGE_MM,
        detector_z_range_mm=DETECTOR_Z_RANGE_MM
    )

summary_data = build_energy_resolution_summary(processed_data)
if not summary_data:
    raise RuntimeError("No energy resolution results found after analysis.")

resource_mgr.check_and_cleanup(force=True)
resource_mgr.print_status()

# ============================================================================
# SAVE RESULTS (CRITICAL FIX: Ensure results are saved)
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Saving Results")
print("=" * 70)

try:
    if 'summary_data' not in locals() or not summary_data:
        summary_data = build_energy_resolution_summary(processed_data)

    if not summary_data:
        raise RuntimeError("No energy resolution results found to save.")

    valid_points, total_points, unique_er_count = count_valid_energy_points(summary_data)
    raw_nodes, raw_point_total = count_raw_payload_points(summary_data)
    print(f"  Valid energy-resolution points: {valid_points}/{total_points}")
    print(f"  Unique energy-resolution values: {unique_er_count}")
    print(f"  Raw ion payload nodes: {raw_nodes}/{total_points}, points={raw_point_total}")
    valid_runs, total_runs = count_valid_run_samples(summary_data)
    print(f"  Valid run samples: {valid_runs}/{total_runs}")
    if valid_points == 0:
        if raw_nodes > 0:
            print("  INFO: Raw-only mode detected (energy_resolution is computed in viewer).")
        else:
            print("  WARNING: No valid energy_resolution values were produced.")
            print("           Checkpoint mismatch or simulation/Abel inversion may have skipped all combinations.")
    elif unique_er_count <= 1 and valid_points > 1:
        print("  WARNING: Energy-resolution values are nearly identical across combinations.")
        print("           Verify lens/FG scan range and SIMION project sensitivity.")

    out_pkl_summary = os.path.join(OUTPUT_DIR, f"processed_data_summary_{RUN_TAG}.pkl")
    save_pickle_atomic(summary_data, out_pkl_summary)
    summary_size_mb = os.path.getsize(out_pkl_summary) / (1024 * 1024)
    print(f"✓ Saved summary pickle: {out_pkl_summary}")
    print(f"  Summary size: {summary_size_mb:.2f} MB")

    out_csv_summary = os.path.join(OUTPUT_DIR, f"processed_data_summary_{RUN_TAG}.csv")
    csv_rows = save_energy_resolution_csv(summary_data, out_csv_summary)
    csv_size_mb = os.path.getsize(out_csv_summary) / (1024 * 1024)
    print(f"✓ Saved summary CSV: {out_csv_summary}")
    print(f"  CSV rows: {csv_rows}, size: {csv_size_mb:.2f} MB")

    if run_summaries:
        out_pkl_runs = os.path.join(OUTPUT_DIR, f"processed_data_runs_{RUN_TAG}.pkl")
        save_pickle_atomic(
            {
                'num_runs_per_node': NUM_RUNS_PER_NODE,
                'run_summaries': run_summaries
            },
            out_pkl_runs
        )
        runs_size_mb = os.path.getsize(out_pkl_runs) / (1024 * 1024)
        print(f"✓ Saved per-run summaries: {out_pkl_runs}")
        print(f"  Per-run summary size: {runs_size_mb:.2f} MB")
    else:
        print("Per-run summaries are stored in-node as *_runs fields (single-SIMION statistical mode).")

    if SAVE_FULL_PROCESSED_DATA:
        out_pkl_full = os.path.join(OUTPUT_DIR, f"processed_data_full_{RUN_TAG}.pkl")
        save_pickle_atomic(processed_data, out_pkl_full)
        full_size_mb = os.path.getsize(out_pkl_full) / (1024 * 1024)
        print(f"✓ Saved full processed_data: {out_pkl_full}")
        print(f"  Full size: {full_size_mb:.2f} MB")
except Exception as e:
    print(f"✗ ERROR: Could not save processed data: {e}")
    traceback.print_exc()

    # Minimal fallback (keeps only core metrics)
    try:
        summary_data = build_energy_resolution_summary(processed_data)
        out_pkl_fallback = os.path.join(OUTPUT_DIR, f"processed_data_fallback_{RUN_TAG}.pkl")
        save_pickle_atomic(summary_data, out_pkl_fallback)
        print(f"✓ Saved fallback summary: {out_pkl_fallback}")
    except Exception as e2:
        print(f"✗ ERROR: Could not save fallback summary: {e2}")

# Summary
total_time = time.time() - workflow_start
print(f"\nTotal workflow time: {total_time/60:.1f} minutes")

# ============================================================================
# VISUALIZATION (GUI with Slider for Field Gradient)
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: Visualization (Interactive Slider for Field Gradient)")
print("=" * 70)

# GUI: slider controls field gradient; right panel shows energy_resolution vs lens_VMI
if ENABLE_VISUALIZATION:
    print(f"Launching interactive GUI with field-gradient slider (method={VIEWER_METHOD})...")
    try:
        if 'summary_data' not in locals() or not summary_data:
            summary_data = build_energy_resolution_summary(processed_data)
        launched = launch_field_gradient_gui(summary_data)
        if not launched:
            raise RuntimeError("No plottable data in GUI helper.")
    except Exception as e:
        print(f"Interactive GUI failed: {e}")
        traceback.print_exc()
        print("Falling back to Utilis heatmap slider...")
        try:
            Utilis.plot_heatmap_all_fg(summary_data if 'summary_data' in locals() else processed_data)
        except Exception as e2:
            print(f"Fallback slider also failed: {e2}")
            print("Falling back to per-FG static heatmaps...")
            fallback_data = summary_data if 'summary_data' in locals() and summary_data else processed_data
            for fg in _sorted_keys(fallback_data.keys()):
                print(f"  Generating heatmap for FG={fg}...")
                try:
                    Utilis.plot_heatmap_energy_lens(fallback_data, fg=fg)
                except Exception as e3:
                    print(f"    Heatmap for FG={fg} skipped: {e3}")
else:
    print("Visualization skipped (ENABLE_VISUALIZATION=False).")

print("\n" + "=" * 70)
print("Workflow Complete!")
print("=" * 70)

print("\nFinalizing the data analysis...")
try:
    import matplotlib.pyplot as plt
    plt.close('all')
except Exception:
    pass
sys.exit(0)
