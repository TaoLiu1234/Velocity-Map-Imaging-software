"""
Utilis_v5.6.py - Streamlined Energy Resolution Analysis Module

This module contains ONLY the functions needed for energy resolution analysis
in Simion_workflow-v5.6.py. All focus filtering and legacy functions removed.

REDUCTION SUMMARY:
  - Original: 7,034 lines, 83 functions, 305.6 KB
  - Streamlined: 2,368 lines, 34 functions, 94.1 KB
  - Reduction: 66.3% fewer lines, 59.0% fewer functions, 69.2% smaller

CORE FUNCTIONS (Called by workflow):
  1. compute_vmi_parameters() - Computes VMI electrode voltages and field parameters
  2. energy_resolution_analysis_direct() - Main energy resolution analysis engine
  3. plot_heatmap_all_fg() - Fallback visualization for energy resolution heatmaps

SUPPORTING FUNCTIONS (34 total):

Memory & Cleanup (2):
  - get_memory_usage_mb() - Monitor memory usage
  - cleanup_memory() - Garbage collection

Checkpoint Management (15):
  - save_checkpoint() - Save analysis progress
  - load_latest_checkpoint() - Resume from checkpoint
  - consolidate_checkpoints() - Merge checkpoint shards
  - [12 helper functions for checkpoint operations]

File Utilities (2):
  - make_energy_resolution_temp_out_file() - Create temp output files
  - remove_file_with_retry() - Windows-safe file deletion

Particle Generation (2):
  - energy_resolution_utilis() - Generate .fly2 particle files
  - rotate_around_x() - 3D vector rotation helper

SIMION Execution (1):
  - run_optimized_simulations_with_ke() - Execute SIMION with parameters

Parameter Management (2):
  - get_parameters_for_combination() - Extract VMI parameters for (FG, Lens, KE)
  - _resolve_energy_resolution_project_files() - Locate IOB/LUA/FLY2 files

Ion Record Parsing (4):
  - _parse_simion_ion_row() - Parse SIMION output lines
  - _build_detector_acceptance_checker() - Create detector filter
  - _extract_ion_records_from_out_file() - Extract ion data from output
  - _convert_ion_records_to_raw_yz() - Convert to YZ coordinates

Pairing & Statistics (3):
  - _normalize_el_for_pairing() - Normalize elevation angles
  - _compute_dr_over_r_for_block() - YZ plane symmetric pairing
  - _compute_dr_over_r_statistics() - Compute resolution statistics

REMOVED FUNCTIONS (49 functions removed):
  - Focus filtering: process_data, process_data_memory_optimized, find_focus_for_fg
  - Old pairing logic: aberration_estimation, extract_aligned_points_for_all_pairs
  - Visualization: data_viewer, visualize_focus_xyz, para_2d_landscape, _3D_landscape
  - Legacy functions: energy_resolution_utilis_legacy, parse_out_file, etc.
  - Unused utilities: generate_particles_fly2, write_standard_beam, etc.

USAGE:
  import Utilis_v5_6 as Utilis  # Drop-in replacement for Simion_workflow -v5.6.py
"""

import os
import glob
import re
import gc
import pickle
import time
import math
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.io import savemat
from scipy.interpolate import griddata


# ============================================================================
# CONSTANTS
# ============================================================================

_CHECKPOINT_COMPACT_KEYS = {
    'fwhm',
    'fwhm_mean',
    'fwhm_variance',
    'fwhm_std',
    'fwhm_runs',
    'energy_resolution',
    'energy_resolution_mean',
    'energy_resolution_variance',
    'energy_resolution_std',
    'energy_resolution_runs',
    'max_r',
    'max_r_mean',
    'max_r_variance',
    'max_r_std',
    'max_r_runs',
    'r_max_all_points',
    'r_max_all_points_runs',
    'all_point_count_runs',
    'generated_particles',
    'generated_particles_per_run',
    'detected_particles',
    'detected_particles_runs',
    'pair_count',
    'pair_count_runs',
    'dr_values',
    'r_values',
    'dr_over_r_values',
    'dr_over_r_pairs',
    'raw_ion_points_yz',
    'raw_point_count',
    'raw_point_format',
    'total_runs',
    'valid_run_count',
    'valid',
    'failure_reason',
    'count_check_passed',
    'plot_marker',
    'plot_skip',
    'pipeline_stage'
}


# ============================================================================
# MEMORY OPTIMIZATION UTILITIES
# ============================================================================

def get_memory_usage_mb():
    """Get current memory usage in MB (cross-platform)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1  # psutil not available




def cleanup_memory(force=False):
    """
    Perform garbage collection to free memory.

    Args:
        force: If True, perform full garbage collection
    """
    if force:
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
    else:
        gc.collect()


_CHECKPOINT_COMPACT_KEYS = {
    'fwhm',
    'fwhm_mean',
    'fwhm_variance',
    'fwhm_std',
    'fwhm_runs',
    'energy_resolution',
    'energy_resolution_mean',
    'energy_resolution_variance',
    'energy_resolution_std',
    'energy_resolution_runs',
    'max_r',
    'max_r_mean',
    'max_r_variance',
    'max_r_std',
    'max_r_runs',
    'r_max_all_points',
    'r_max_all_points_runs',
    'all_point_count_runs',
    'generated_particles',
    'generated_particles_per_run',
    'detected_particles',
    'detected_particles_runs',
    'pair_count',
    'pair_count_runs',
    'dr_values',
    'r_values',
    'dr_over_r_values',
    'dr_over_r_pairs',
    'raw_ion_points_yz',
    'raw_point_count',
    'raw_point_format',
    'total_runs',
    'valid_run_count',
    'valid',
    'failure_reason',
    'count_check_passed',
    'plot_marker',
    'plot_skip',
    'pipeline_stage'
}




# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def _count_checkpoint_nodes(data):
    count = 0
    for fg_data in (data or {}).values():
        if not isinstance(fg_data, dict):
            continue
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                continue
            count += len(lens_data)
    return count




def _set_checkpoint_node(container, fg, lens_vmi, ke, result):
    if fg not in container:
        container[fg] = {}
    if lens_vmi not in container[fg]:
        container[fg][lens_vmi] = {}
    container[fg][lens_vmi][ke] = result




def _merge_checkpoint_results(target, source):
    for fg, fg_data in (source or {}).items():
        if not isinstance(fg_data, dict):
            continue
        for lens_vmi, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            for ke, result in lens_data.items():
                _set_checkpoint_node(target, fg, lens_vmi, ke, result)




def _is_compact_checkpoint_data(data):
    sample_checked = False
    for fg_data in (data or {}).values():
        if not isinstance(fg_data, dict):
            return False
        for lens_data in fg_data.values():
            if not isinstance(lens_data, dict):
                return False
            for result in lens_data.values():
                sample_checked = True
                if not isinstance(result, dict):
                    return False
                keys = set(result.keys())
                if not keys.issubset(_CHECKPOINT_COMPACT_KEYS):
                    return False
                break
            break
        break
    return sample_checked




def _compact_checkpoint_results(data):
    if not data:
        return {}
    if _is_compact_checkpoint_data(data):
        return data

    essential_data = {}
    for fg, fg_data in data.items():
        if not isinstance(fg_data, dict):
            continue
        for lens_vmi, lens_data in fg_data.items():
            if not isinstance(lens_data, dict):
                continue
            for ke, result in lens_data.items():
                if not isinstance(result, dict):
                    continue
                compact_result = {
                    'fwhm': result.get('fwhm'),
                    'fwhm_mean': result.get('fwhm_mean'),
                    'fwhm_variance': result.get('fwhm_variance'),
                    'fwhm_std': result.get('fwhm_std'),
                    'fwhm_runs': result.get('fwhm_runs'),
                    'energy_resolution': result.get('energy_resolution'),
                    'energy_resolution_mean': result.get('energy_resolution_mean'),
                    'energy_resolution_variance': result.get('energy_resolution_variance'),
                    'energy_resolution_std': result.get('energy_resolution_std'),
                    'energy_resolution_runs': result.get('energy_resolution_runs'),
                    'max_r': result.get('max_r'),
                    'max_r_mean': result.get('max_r_mean'),
                    'max_r_variance': result.get('max_r_variance'),
                    'max_r_std': result.get('max_r_std'),
                    'max_r_runs': result.get('max_r_runs'),
                    'r_max_all_points': result.get('r_max_all_points'),
                    'r_max_all_points_runs': result.get('r_max_all_points_runs', []),
                    'all_point_count_runs': result.get('all_point_count_runs', []),
                    'generated_particles': result.get('generated_particles'),
                    'generated_particles_per_run': result.get('generated_particles_per_run'),
                    'detected_particles': result.get('detected_particles'),
                    'detected_particles_runs': result.get('detected_particles_runs'),
                    'pair_count': result.get('pair_count'),
                    'pair_count_runs': result.get('pair_count_runs'),
                    'dr_values': result.get('dr_values', []),
                    'r_values': result.get('r_values', []),
                    'dr_over_r_values': result.get('dr_over_r_values', []),
                    'dr_over_r_pairs': result.get('dr_over_r_pairs', []),
                    'raw_ion_points_yz': result.get('raw_ion_points_yz', []),
                    'raw_point_count': result.get('raw_point_count'),
                    'raw_point_format': result.get('raw_point_format'),
                    'total_runs': result.get('total_runs'),
                    'valid_run_count': result.get('valid_run_count'),
                    'valid': result.get('valid'),
                    'failure_reason': result.get('failure_reason'),
                    'count_check_passed': result.get('count_check_passed'),
                    'plot_marker': result.get('plot_marker'),
                    'plot_skip': result.get('plot_skip'),
                    'pipeline_stage': result.get('pipeline_stage')
                }
                _set_checkpoint_node(essential_data, fg, lens_vmi, ke, compact_result)
    return essential_data




def _checkpoint_part_pattern(filename):
    return f"{filename}_checkpoint_part_*.pkl"




def _checkpoint_part_file(filename, checkpoint_num):
    try:
        idx = int(checkpoint_num)
    except (TypeError, ValueError):
        idx = 0
    return f"{filename}_checkpoint_part_{idx:08d}.pkl"




def _sorted_checkpoint_parts(filename):
    files = glob.glob(_checkpoint_part_pattern(filename))

    def _extract_num(path):
        base = os.path.basename(path)
        m = re.search(r'_checkpoint_part_(\d+)\.pkl$', base)
        if not m:
            return -1
        try:
            return int(m.group(1))
        except ValueError:
            return -1

    return sorted(files, key=lambda p: (_extract_num(p), p))




def _save_checkpoint_payload_atomic(path, payload):
    tmp_file = f"{path}.tmp"
    with open(tmp_file, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, path)




def save_checkpoint(data, filename, checkpoint_num, completed_combinations=None):
    """
    Save checkpoint shard (incremental chunk) for recovery.

    For long runs this avoids rewriting one ever-growing checkpoint file.
    """
    try:
        node_count = _count_checkpoint_nodes(data)
        if node_count <= 0:
            return None

        checkpoint_file = _checkpoint_part_file(filename, checkpoint_num)
        checkpoint_data = {
            'results': _compact_checkpoint_results(data),
            'completed': list(completed_combinations) if completed_combinations else [],
            'checkpoint_num': int(checkpoint_num) if checkpoint_num is not None else 0,
            'checkpoint_format': 'shard_v2'
        }
        _save_checkpoint_payload_atomic(checkpoint_file, checkpoint_data)
        print(
            f"  Checkpoint shard saved: {checkpoint_file} "
            f"(nodes={node_count}, completed={len(checkpoint_data['completed'])})"
        )
        return checkpoint_file
    except Exception as e:
        print(f"  Warning: Could not save checkpoint: {e}")
        return None




def load_latest_checkpoint(filename):
    """
    Load the latest checkpoint file for recovery.

    Args:
        filename: Base filename (e.g., 'energy_resolution_direct')

    Returns:
        tuple: (fwhm_results dict, completed_combinations set, checkpoint_num) or (None, None, 0)
    """
    part_files = _sorted_checkpoint_parts(filename)
    merged_file = f"{filename}_checkpoint_merged.pkl"
    legacy_file = f"{filename}_checkpoint.pkl"

    if part_files:
        merged_results = {}
        merged_completed = set()
        latest_num = 0
        loaded_parts = 0
        for part_file in part_files:
            try:
                with open(part_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                _merge_checkpoint_results(merged_results, checkpoint_data.get('results', {}))
                merged_completed.update(tuple(c) for c in checkpoint_data.get('completed', []))
                part_num = int(checkpoint_data.get('checkpoint_num', 0) or 0)
                if part_num > latest_num:
                    latest_num = part_num
                loaded_parts += 1
            except Exception as e:
                print(f"  Warning: Could not load checkpoint shard {part_file}: {e}")
        print(
            f"  Loaded {loaded_parts} checkpoint shards: "
            f"{_count_checkpoint_nodes(merged_results)} nodes, {len(merged_completed)} completed markers"
        )
        return merged_results, merged_completed, latest_num

    for checkpoint_file in (merged_file, legacy_file):
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                results = checkpoint_data.get('results', {})
                completed = set(tuple(c) for c in checkpoint_data.get('completed', []))
                checkpoint_num = int(checkpoint_data.get('checkpoint_num', 0) or 0)
                print(
                    f"  Loaded checkpoint: {checkpoint_file} "
                    f"({len(completed)} completed, nodes={_count_checkpoint_nodes(results)})"
                )
                return results, completed, checkpoint_num
        except Exception as e:
            print(f"  Warning: Could not load checkpoint {checkpoint_file}: {e}")

    return None, set(), 0




def consolidate_checkpoints(filename, cleanup_parts=False):
    """
    Merge checkpoint shards into a single consolidated checkpoint file.
    """
    part_files = _sorted_checkpoint_parts(filename)
    if not part_files:
        return None

    merged_results = {}
    merged_completed = set()
    latest_num = 0
    for part_file in part_files:
        try:
            with open(part_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            _merge_checkpoint_results(merged_results, checkpoint_data.get('results', {}))
            merged_completed.update(tuple(c) for c in checkpoint_data.get('completed', []))
            part_num = int(checkpoint_data.get('checkpoint_num', 0) or 0)
            if part_num > latest_num:
                latest_num = part_num
        except Exception as e:
            print(f"  Warning: Could not read checkpoint shard {part_file} during consolidation: {e}")

    merged_file = f"{filename}_checkpoint_merged.pkl"
    payload = {
        'results': _compact_checkpoint_results(merged_results),
        'completed': list(merged_completed),
        'checkpoint_num': latest_num,
        'checkpoint_format': 'merged_v2',
        'source_parts': len(part_files)
    }
    try:
        _save_checkpoint_payload_atomic(merged_file, payload)
        print(
            f"  Consolidated {len(part_files)} checkpoint shards -> {merged_file} "
            f"(nodes={_count_checkpoint_nodes(merged_results)})"
        )
    except Exception as e:
        print(f"  Warning: Could not write consolidated checkpoint {merged_file}: {e}")
        return None

    if cleanup_parts:
        for part_file in part_files:
            try:
                os.remove(part_file)
            except Exception:
                pass
        legacy_file = f"{filename}_checkpoint.pkl"
        if os.path.exists(legacy_file):
            try:
                os.remove(legacy_file)
            except Exception:
                pass

    return merged_file




def _has_valid_energy_value(result_dict):
    if not isinstance(result_dict, dict):
        return False
    er = result_dict.get('energy_resolution_mean', result_dict.get('energy_resolution'))
    try:
        val = float(er)
        return np.isfinite(val)
    except (TypeError, ValueError):
        return False




def _checkpoint_result_has_required_pair_details(result_dict):
    """
    A checkpoint result is considered complete if it has at least one
    recoverable metric payload:
    - valid energy metric with dr_over_r_pairs, or
    - raw ion payload (ion_n,y,z), or
    - explicitly marked invalid node.
    """
    if not isinstance(result_dict, dict):
        return False
    raw_points = result_dict.get('raw_ion_points_yz')
    has_raw_points = isinstance(raw_points, list) and len(raw_points) > 0
    if not _has_valid_energy_value(result_dict):
        if has_raw_points:
            return True
        return result_dict.get('valid') is False
    pairs = result_dict.get('dr_over_r_pairs')
    has_pairs = isinstance(pairs, list) and len(pairs) > 0
    return has_pairs or has_raw_points




def load_checkpoint(filename):
    """
    Load data from a checkpoint file (legacy support).

    Args:
        filename: Checkpoint filename

    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Handle both old and new checkpoint formats
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        return data
    except Exception as e:
        print(f"  Warning: Could not load checkpoint {filename}: {e}")
        return None




# ============================================================================
# FILE UTILITIES
# ============================================================================

def make_energy_resolution_temp_out_file(base_out_file="energy_resolution_out.txt"):
    """
    Create a unique temporary out-file path for one energy-resolution node.

    This avoids repeatedly appending to a long-lived shared output file, which can
    degrade I/O performance in large runs.
    """
    import tempfile

    base_name = os.path.basename(base_out_file) if base_out_file else "energy_resolution_out.txt"
    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".txt"
    prefix = f"{stem}_node_"
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=ext, dir=os.getcwd())
    os.close(fd)
    # We only need the path; runner writes the content.
    try:
        os.remove(temp_path)
    except OSError:
        pass
    return temp_path




def remove_file_with_retry(path, max_wait_s=6.0, step_s=0.2):
    """Best-effort file cleanup for Windows file-lock timing."""
    if not path:
        return True
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(step_s)
    return not os.path.exists(path)


# Define complex type hints for better readability
Trajectory = List[Tuple[float, float, float]]
Particle = List[Trajectory]
ParticleData = Dict[int, Particle]
LensVMIData = Dict[str, ParticleData]
MainData = Dict[float, Dict[float, LensVMIData]]
AnalysisResults = Dict[float, Dict[float, Dict[int, Dict[int, Dict[int, List[float]]]]]]






# ============================================================================
# VMI PARAMETER COMPUTATION
# ============================================================================

def compute_vmi_parameters(field_min=200, field_max=500, num_points=20, lens_min=1.38, lens_max=1.38, num_lens_points=1,
                           mode='velocity_imaging', save_to_file=False, filename='parameters.mat'):
    """
Compute VMI (Velocity Map Imaging) electrode voltages and electric field parameters
based on a range of field gradients and lens factors.

This function performs a parameter sweep over lens_VMI and field_gradient combinations,
iteratively adjusting the ground offset voltage until I_grid ≈ 0 for each combination.

Args:
    field_min (float): Minimum field gradient in V/cm (default: 200)
    field_max (float): Maximum field gradient in V/cm (default: 500)
    num_points (int): Number of points between field_min and field_max (default: 20)
    lens_min (float): Minimum lens focusing factor (lens_VMI), typically around 1.38 (default: 1.38)
    lens_max (float): Maximum lens focusing factor (lens_VMI), typically around 1.38 (default: 1.38)
    num_lens_points (int): Number of points between lens_min and lens_max (default: 1)
    mode (str): Mode of parameter generation ('velocity_imaging' for VMI or 'spatial_imaging' for spatial imaging) (default: 'velocity_imaging')
    save_to_file (bool): Whether to save results to .mat file (default: False)
    filename (str): Output .mat filename if saving (default: 'parameters.mat')

Returns:
    dict: Dictionary containing all computed arrays (length = num_lens_points * num_points):
          - field_gradient: Array of gradient values (repeated for each lens)
          - Offset_to_ground: Adjusted ground offsets after convergence
          - lens_VMI: Lens value array (repeated for each field per lens)
          - I_grid: Final current-equivalent term (should be ~0)
          - VMI2, VMI1: Electrode voltages
          - e_grid: Midpoint electric potential in grid region
          - dt_e: Copy of VMI2, possibly used for time-dependent simulations
"""
    # Generate sequences
    lens_sequence = np.round(np.linspace(lens_min, lens_max, num_lens_points), 3)
    field_sequence = np.linspace(field_min, field_max, num_points)

    total_points = num_lens_points * num_points

    # Initialize output arrays
    field_gradient = np.zeros(total_points)
    Offset_to_ground = np.zeros(total_points)
    lens_VMI = np.zeros(total_points)
    I_grid = np.zeros(total_points)
    VMI2 = np.zeros(total_points)
    VMI1 = np.zeros(total_points)
    e_grid = np.zeros(total_points)
    dt_e = np.zeros(total_points)

    # Parameter generation mode
    # Pre-calculate slope factor: -800 / 120 = -20/3 ≈ -6.6667 for velocity imaging, -3 for spatial imaging
    slope_factor = -800 / 120 if mode == 'velocity_imaging' else -3

    idx = 0
    for lens in lens_sequence:
        for field in field_sequence:
            if mode == 'spatial_imaging':
                lens_correction = 13.0  # Separate lens_correction variable (different from lens_VMI)
                # Initial guess for offset to ground
                Offset_to_ground[idx] = 1000  # Start at 1000 V

                while True:
                    # Recalculate I_grid with spatial imaging formula: I_grid = -3*field + offset
                    I_grid[idx] = -3 * field + Offset_to_ground[idx]

                    # Calculate average potential in the grid region
                    e_grid[idx] = Offset_to_ground[idx]
                    # dt_e set to e_grid for spatial imaging
                    dt_e[idx] = e_grid[idx]
                    # Update VMI electrode voltages using separate lens_correction=13 (not the same as lens_VMI)
                    VMI2[idx] = -(e_grid[idx] - I_grid[idx])/ lens_correction + Offset_to_ground[idx]
                    VMI1[idx] = I_grid[idx] + (20.97/29.5)*(e_grid[idx] - I_grid[idx])

                    # Stop when I_grid is sufficiently close to zero
                    if abs(I_grid[idx]) < 1e-10:
                        break

                    # Adjust Offset_to_ground to drive I_grid toward zero via feedback correction
                    Offset_to_ground[idx] -= I_grid[idx]

            else:  # velocity_imaging
                # Initial guess for offset to ground
                Offset_to_ground[idx] = 1000  # Start at 1000 V

                while True:
                    # Recalculate I_grid: linear relation with field_gradient and offset
                    I_grid[idx] = slope_factor * field + Offset_to_ground[idx]

                    # Update VMI electrode voltages
                    VMI2[idx] = Offset_to_ground[idx]
                    VMI1[idx] = VMI2[idx] - (VMI2[idx] - I_grid[idx]) / lens

                    # Calculate average potential in the grid region
                    e_grid[idx] = VMI1[idx] + 0.5 * (VMI2[idx] - VMI1[idx])

                    # dt_e set to VMI2 (possibly for timing or delay purposes)
                    dt_e[idx] = VMI2[idx]

                    # Stop when I_grid is sufficiently close to zero
                    if abs(I_grid[idx]) < 1e-10:
                        break

                    # Adjust Offset_to_ground to drive I_grid toward zero
                    Offset_to_ground[idx] -= I_grid[idx]  # Feedback correction

            # Store the values
            field_gradient[idx] = field
            lens_VMI[idx] = lens
            idx += 1


    # Pack results into dictionary (like MATLAB struct)
    parameters = {
        'field_gradient': np.round(field_gradient),
        'Offset_to_ground': np.round(Offset_to_ground),
        'lens_VMI': np.round(lens_VMI, 4),
        'I_grid': np.round(I_grid),
        'VMI2': np.round(VMI2),
        'VMI1': np.round(VMI1),
        'e_grid': np.round(e_grid),
        'dt_e': np.round(dt_e)
    }

    # Optionally save to .mat file for MATLAB compatibility
    if save_to_file:
        savemat(filename, {'parameters': parameters})
        print(f"✅ Parameters saved to '{filename}'")

    return parameters



def rotate_around_x(vec, theta):
    """
    Rotate a 3D vector around the x-axis by theta radians.
    Returns a normalized vector.
    """
    x, y, z = vec
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # For rotation around x-axis: x remains unchanged, y and z are rotated
    new_y = y * cos_t - z * sin_t
    new_z = y * sin_t + z * cos_t
    # Normalize
    length = math.sqrt(x**2 + new_y**2 + new_z**2)
    if length == 0:
        return (0, 0, 0)
    return (x / length, new_y / length, new_z / length)



# ============================================================================
# PARTICLE GENERATION
# ============================================================================

def energy_resolution_utilis(filename='energy_resolution_particle.fly2', position=(0.0, 0.0, 0.0),
                             num_particles=100, ke=15, num_groups=1,
                             az_first_deg=0.0, az_step_deg=30.0,az_num=6, el_first_deg=0.0, el_step_deg=30.0,el_num=12,intraction_volume=False,intraction_volume_array_mm = 0):
    """
    Generates a .fly2 file using grouped standard_beam blocks:
      - each group has n=num_particles
      - az is fixed (default 0 deg)
      - el uses SIMION arithmetic_sequence(first, step, n)
      - groups are emitted from the same source position

    This matches the grouped fly2/output format used in energy_resolution_out.txt
    where particles are naturally ordered by group (e.g., first N then second N).

    Args:
        filename (str): Output .fly2 file name.
        position (tuple): Source position (x, y, z).
        num_particles (int): Number of particles per group.
        ke (float): Kinetic energy in eV.
        num_groups (int): Number of groups (default 1).
        azimuth_deg (float): Fixed azimuth angle in degrees.
        el_first_deg (float): Arithmetic sequence first elevation in degrees.
        el_step_deg (float): Arithmetic sequence step in degrees.
    """
    if not isinstance(num_particles, int) or num_particles <= 0:
        print(f"⚠️ Invalid input for `num_particles` (value: {num_particles}). It must be a positive integer. Setting to 100.")
        num_particles = 100
    if not isinstance(num_groups, int) or num_groups <= 0:
        num_groups = 1
    if not filename or not filename.strip():
        filename = 'energy_resolution_particle.fly2'

    try:
        az_first_deg = float(az_first_deg)
    except (TypeError, ValueError):
        az_first_deg = 0.0
    try:
        el_first_deg = float(el_first_deg)
    except (TypeError, ValueError):
        el_first_deg = 0.0
    try:
        el_step_deg = float(el_step_deg)
    except (TypeError, ValueError):
        el_step_deg = 1.0

    x, y, z = position

    try:
        with open(filename, 'w') as fid:
            fid.write('particles {\n')
            fid.write('  coordinates = 0,\n')
            for group_idx in range(num_groups):
                if intraction_volume == True:
                    shift = intraction_volume_array_mm[group_idx]
                    new_position = position + shift
                    x, y, z = new_position
                fid.write('  standard_beam {\n')
                fid.write(f'    n = {num_particles},\n')
                fid.write('    tob = 0,\n')
                fid.write('    mass = 0.000548579903,\n')
                fid.write('    charge = -1,\n')
                fid.write(f'    ke = {ke},\n')
                fid.write('    az = arithmetic_sequence {\n')
                fid.write(f'      first = {az_first_deg:.10g},\n')
                fid.write(f'      step = {az_step_deg:.10g},\n')
                fid.write(f'      n = {int(az_num)}\n')
                fid.write('    },\n')
                fid.write('    el = arithmetic_sequence {\n')
                fid.write(f'      first = {el_first_deg:.10g},\n')
                fid.write(f'      step = {el_step_deg:.10g},\n')
                fid.write(f'      n = {int(el_num)}\n')
                fid.write('    },\n')
                fid.write('    cwf = 1,\n')
                fid.write(f'    color = {group_idx},\n')
                fid.write(f'    position = vector({x:.10g}, {y:.10g}, {z:.10g})\n')
                fid.write('  }')
                fid.write(',\n' if group_idx < num_groups - 1 else '\n')
            fid.write('}\n')
    except Exception as e:
        raise IOError(f"Failed to create file {filename}: {e}")




# ============================================================================
# SIMION EXECUTION
# ============================================================================

def generate_simion_lua_file(field_idx, param, output_filename):
    """
    Generate a SIMION-compatible Lua script that sets electrode voltages based on parameter index.
    This function now matches the format of 'WORKING_TITLE (copy).LUA'.

    Args:
        field_idx (int): Index to select values from param lists.
        param (dict): Must contain lists: 'dt_e', 'VMI2', 'e_grid', 'VMI1'.
        output_filename (str): Full path or filename for the output Lua script. Must be provided.

    Returns:
        None. Writes the Lua file to disk and prints a confirmation message.
    """
    from datetime import datetime

    # Generate timestamp for file header
    now = datetime.now()
    timestamp = now.strftime("%Y             %m             %d             %H             %M       %S.%f")[:-3]

    # Extract parameter values for current field index
    dt_e_val = param['dt_e'][field_idx]
    VMI2_val = param['VMI2'][field_idx]
    e_grid_val = param['e_grid'][field_idx]
    VMI1_val = param['VMI1'][field_idx]

    # Build Lua content with formatted electrode voltages
    lua_content = f"""
-- LUA file, automatically created from the MATLAB function write_GEM_file.
-- written on: [Year month day hour min sec]
-- {timestamp}
simion.workbench_program()
-- called on PA initialization
function segment.init_p_values()
-- before we start the fly, we remove trapcheck.info, because no traps have been detected yet.
os.remove(\"trapcheck.info\")
    -- set electrode voltages
    adj_elect01 = 0.000000
    adj_elect02 = {dt_e_val:.6f}
    adj_elect03 = {dt_e_val:.6f}
    adj_elect04 = {dt_e_val:.6f}
    adj_elect05 = {VMI2_val:.6f}
    adj_elect06 = {e_grid_val:.6f}
    adj_elect07 = {VMI1_val:.6f}
    adj_elect08 = 0.000000
    adj_elect09 = -3000.000000
    adj_elect10 = -3000.000000
    adj_elect11 = -3000.000000
    adj_elect12 = -4000.000000
end


-- this function is called after every time step.
-- in our case, we want to know if we made a trap, if so we terminate the run
function segment.other_actions()
-- we assume that a extremely large TOF means we created a trap:
  if (ion_time_of_flight-ion_time_of_birth > 50.000000 or (ion_px_mm > 207.000000 and ion_vx_mm < 0))

	then ion_splat = -3 	-- this means we remove the ion
	  -- here we write it to a file, so MATLAB can detect it:
	  local trapfile = assert(io.open("trapcheck.info", "a")) -- write mode
	  trapfile:write(ion_number .. string.char(10))
	  trapfile:close()

	return end
end
"""

    # Write generated content to specified file with CRLF line endings
    with open(output_filename, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(lua_content)


def run_optimized_simulations_with_ke(param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE):
    """
    Run optimized SIMION simulations with Lua generation and sequential SIMION runs for a given ke.

    MEMORY OPTIMIZED:
    - Does not buffer SIMION output in memory (prevents MemoryError)
    - Cleans up temp files immediately after use
    - Uses DEVNULL for stdout/stderr to prevent memory buildup
    """
    import subprocess
    import os
    import time
    import errno

    num_simulations = len(param['field_gradient'])
    successful_runs = 0
    failed_runs = 0

    # Verify IOB file exists before starting
    if not os.path.exists(IOB_FILE):
        print(f"ERROR: IOB file '{IOB_FILE}' not found!")
        return {'requested': num_simulations, 'successful': 0, 'failed': num_simulations}

    def _append_output_with_parameters_streaming(temp_file_path, output_file_path, parameter_line):
        """
        Append temp SIMION output to OUT_FILE without loading full file into memory.
        Insert parameter line right after the first separator line.
        """
        separator = "------ Begin Next Fly'm ------"
        inserted = False
        with open(temp_file_path, 'r', errors='ignore') as in_f, open(output_file_path, 'a') as out_f:
            for line in in_f:
                out_f.write(line)
                if (not inserted) and (separator in line):
                    out_f.write(parameter_line)
                    inserted = True
        if not inserted:
            # Fallback: append parameters at end if separator was not found
            with open(output_file_path, 'a') as out_f:
                out_f.write(parameter_line)

    # Generate Lua file (only one needed since we process sequentially)
    for field_idx in range(num_simulations):
        lua_filename = OUTPUT_FILENAME_LUA
        generate_simion_lua_file(field_idx, param, output_filename=lua_filename)

        temp_out_file = f"temp_out_ke_{field_idx}_{os.getpid()}_{int(time.time() * 1000)}.txt"
        if os.path.exists(temp_out_file):
            try:
                os.remove(temp_out_file)
            except OSError:
                pass

        # Run SIMION - DO NOT capture output to avoid memory issues
        command = f"simion.exe --nogui fly --recording-output={temp_out_file} {IOB_FILE}"

        def _read_file_with_retry(path, max_wait_s=8.0, step_s=0.25):
            """Windows: SIMION 可能短时间占用输出文件，做鲁棒重试读取。"""
            deadline = time.time() + max_wait_s
            last_exc = None
            while time.time() < deadline:
                try:
                    with open(path, 'r') as f:
                        return f.read()
                except OSError as e:
                    last_exc = e
                    time.sleep(step_s)
            raise last_exc if last_exc else OSError(errno.ETIMEDOUT, f"Timeout reading {path}")

        def _remove_with_retry(path, max_wait_s=8.0, step_s=0.25):
            """Windows: 删除也可能遇到 WinError 32，占用则重试；最终失败返回 False。"""
            deadline = time.time() + max_wait_s
            while time.time() < deadline:
                try:
                    os.remove(path)
                    return True
                except FileNotFoundError:
                    return True
                except OSError:
                    time.sleep(step_s)
            return False

        # Run with retry to handle transient SIMION/file-lock failures
        max_retries = 3
        run_ok = False
        last_returncode = None

        for attempt in range(1, max_retries + 1):
            # Use DEVNULL to discard stdout/stderr - prevents memory buildup
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            last_returncode = result.returncode

            # Wait briefly for file to become available/non-empty
            max_wait = 12.0
            wait_step = 0.25
            waited = 0.0
            output_ready = False
            while waited < max_wait:
                if os.path.exists(temp_out_file):
                    try:
                        if os.path.getsize(temp_out_file) > 0:
                            output_ready = True
                            break
                    except OSError:
                        pass
                time.sleep(wait_step)
                waited += wait_step

            if result.returncode == 0 and output_ready:
                run_ok = True
                break

            # Some SIMION runs may return non-zero but still produce usable output
            if result.returncode != 0 and output_ready:
                print(f"Warning: SIMION returned code {result.returncode} for field_idx {field_idx}, but output exists; continuing.")
                run_ok = True
                break

            if attempt < max_retries:
                time.sleep(0.5 * attempt)

        if not run_ok:
            print(f"Error: SIMION failed for field_idx {field_idx} after {max_retries} attempts (last return code: {last_returncode})")
            failed_runs += 1
            if os.path.exists(temp_out_file):
                _remove_with_retry(temp_out_file)
            continue

        # Immediately append to output file and delete temp file
        if os.path.exists(temp_out_file):
            try:
                current_parameters = (
                    f"parameters = "
                    f"[{param['field_gradient'][field_idx]},{param['Offset_to_ground'][field_idx]},"
                    f"{param['lens_VMI'][field_idx]},{param['I_grid'][field_idx]},{param['VMI2'][field_idx]},"
                    f"{param['VMI1'][field_idx]},{param['e_grid'][field_idx]},{param['dt_e'][field_idx]},{ke}]\n"
                )
                _append_output_with_parameters_streaming(temp_out_file, OUT_FILE, current_parameters)

                # Immediately delete temp file to free disk space
                _remove_with_retry(temp_out_file)
                successful_runs += 1

            except Exception as e:
                print(f"Error processing temp file: {e}")
                failed_runs += 1
                if os.path.exists(temp_out_file):
                    _remove_with_retry(temp_out_file)
        else:
            failed_runs += 1

    return {'requested': num_simulations, 'successful': successful_runs, 'failed': failed_runs}



# ============================================================================
# PARAMETER EXTRACTION
# ============================================================================

def get_parameters_for_combination(processed_data, fg, lens_vmi, ke):
    """
    Helper function to get parameters for a specific fg, lens_vmi, ke combination.

    Args:
        processed_data: Dictionary containing processed SIMION data
        fg: Field gradient
        lens_vmi: Lens VMI value
        ke: Kinetic energy

    Returns:
        dict: Parameters for the specific combination
    """
    # DEBUG: Add print statements to track function execution
    # Suppressed debug output
    #print(f"  DEBUG: get_parameters_for_combination called with fg={fg}, lens_vmi={lens_vmi}, ke={ke}")
    # Try to find original parameters in processed_data
    for test_fg in processed_data:
        if test_fg == fg:
            for test_lens in processed_data[test_fg]:
                if test_lens == lens_vmi:
                    for test_ke in processed_data[test_fg][test_lens]:
                        if test_ke == ke:
                            # Get the global data which might contain the original parameters
                            global_data = processed_data[test_fg][test_lens][test_ke].get('global', {})
                            # If we don't have the original parameters, we'll use computed ones
                            # DEBUG: Found matching parameters, computing correct ones
                            # Suppressed debug output
                            #print(f"  DEBUG: Found matching parameters in processed_data for fg={fg}, lens={lens_vmi}, ke={ke}")
                            # Use compute_vmi_parameters to get the correct parameters with velocity_imaging mode
                            from Utilis import compute_vmi_parameters
                            computed_params = compute_vmi_parameters(
                                field_min=fg, field_max=fg, num_points=1,
                                lens_min=lens_vmi, lens_max=lens_vmi, num_lens_points=1,
                                mode='velocity_imaging'  # Use same mode as main workflow
                            )
                            # Suppressed debug output
                            #print(f"  DEBUG: Computed parameters: Offset_to_ground={computed_params['Offset_to_ground'][0]}, VMI2={computed_params['VMI2'][0]}, VMI1={computed_params['VMI1'][0]}")
                            return {
                                'field_gradient': np.array([computed_params['field_gradient'][0]]),
                                'Offset_to_ground': np.array([computed_params['Offset_to_ground'][0]]),
                                'lens_VMI': np.array([computed_params['lens_VMI'][0]]),
                                'I_grid': np.array([computed_params['I_grid'][0]]),
                                'VMI2': np.array([computed_params['VMI2'][0]]),
                                'VMI1': np.array([computed_params['VMI1'][0]]),
                                'e_grid': np.array([computed_params['e_grid'][0]]),
                                'dt_e': np.array([computed_params['dt_e'][0]])
                            }

    # If we couldn't find the original parameters, use computed ones
    # Suppressed parameter warning output
    # print(f"  WARNING: Could not find original parameters for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}")
    # Use compute_vmi_parameters to get the correct parameters
    from Utilis import compute_vmi_parameters
    computed_params = compute_vmi_parameters(
        field_min=fg, field_max=fg, num_points=1,
        lens_min=lens_vmi, lens_max=lens_vmi, num_lens_points=1,
        mode='velocity_imaging'  # Use same mode as main workflow
    )
    # Suppressed computed parameters output
    # print(f"  Using computed parameters: Offset_to_ground={computed_params['Offset_to_ground'][0]}, VMI2={computed_params['VMI2'][0]}, VMI1={computed_params['VMI1'][0]}")
    return {
        'field_gradient': np.array([computed_params['field_gradient'][0]]),
        'Offset_to_ground': np.array([computed_params['Offset_to_ground'][0]]),
        'lens_VMI': np.array([computed_params['lens_VMI'][0]]),
        'I_grid': np.array([computed_params['I_grid'][0]]),
        'VMI2': np.array([computed_params['VMI2'][0]]),
        'VMI1': np.array([computed_params['VMI1'][0]]),
        'e_grid': np.array([computed_params['e_grid'][0]]),
        'dt_e': np.array([computed_params['dt_e'][0]])
    }




def _resolve_energy_resolution_project_files():
    """
    Resolve a consistent (IOB, LUA, FLY2) file set for energy-resolution simulations.
    The selected IOB determines which LUA/FLY2 filenames MUST be updated.
    """
    preferred_prefix = "WORKING_TITLE_energy_resolution_tao"
    fallback_prefix = "WORKING_TITLE_tao"

    preferred_iob = f"{preferred_prefix}.iob"
    fallback_iob = f"{fallback_prefix}.iob"

    if os.path.exists(preferred_iob):
        prefix = preferred_prefix
    elif os.path.exists(fallback_iob):
        prefix = fallback_prefix
    else:
        return None, None, None, None

    output_fly2 = f"{prefix}.fly2"
    output_lua = f"{prefix}.lua"
    iob_file = f"{prefix}.iob"
    out_file = "energy_resolution_out.txt"
    return output_fly2, output_lua, iob_file, out_file




# ============================================================================
# ION RECORD PARSING
# ============================================================================

def _parse_simion_ion_row(line):
    """
    Parse one SIMION ion row:
      ion_n, x, y, z
    or:
      ion_n, x, y, z, el
    or:
      ion_n, x, y, z, azm, el
    """
    if not line:
        return None
    parts = [part.strip() for part in line.split(',')]
    if len(parts) < 4:
        return None
    try:
        ion_n = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
    except (TypeError, ValueError):
        return None

    azm = None
    el = None
    if len(parts) >= 6:
        try:
            azm = float(parts[4])
        except (TypeError, ValueError):
            azm = None
        try:
            el = float(parts[5])
        except (TypeError, ValueError):
            el = None
    elif len(parts) >= 5:
        try:
            el = float(parts[4])
        except (TypeError, ValueError):
            el = None
    return ion_n, x, y, z, azm, el




def _build_detector_acceptance_checker(detector_x_mm=None, detector_x_tol_mm=0.5,
                                       detector_y_range_mm=None, detector_z_range_mm=None):
    detector_filter_enabled = (
        detector_x_mm is not None
        and detector_x_tol_mm is not None
        and detector_y_range_mm is not None
        and detector_z_range_mm is not None
    )
    if detector_filter_enabled:
        y_min, y_max = sorted((float(detector_y_range_mm[0]), float(detector_y_range_mm[1])))
        z_min, z_max = sorted((float(detector_z_range_mm[0]), float(detector_z_range_mm[1])))
        detector_x_center = float(detector_x_mm)
        detector_x_tol = abs(float(detector_x_tol_mm))
    else:
        y_min = y_max = z_min = z_max = detector_x_center = detector_x_tol = None

    def _passes_detector_acceptance(x_value, y_value, z_value):
        if not detector_filter_enabled:
            return True
        try:
            x_num = float(x_value)
            y_num = float(y_value)
            z_num = float(z_value)
        except (TypeError, ValueError):
            return False
        return (
            abs(x_num - detector_x_center) <= detector_x_tol
            and y_min <= y_num <= y_max
            and z_min <= z_num <= z_max
        )

    return _passes_detector_acceptance




def _extract_ion_records_from_out_file(out_file_path,
                                       detector_x_mm=None, detector_x_tol_mm=0.5,
                                       detector_y_range_mm=None, detector_z_range_mm=None):
    """
    Parse SIMION energy_resolution_out.txt and return ion records sorted by ion index.

    Each record includes:
      - initial: row dict (x, y, z, el)
      - final: row dict (x, y, z, el)
      - initial_el: pairing key for az-fixed/el-scan workflow
    """
    if not out_file_path or not os.path.exists(out_file_path):
        return [], False

    _passes_detector_acceptance = _build_detector_acceptance_checker(
        detector_x_mm=detector_x_mm,
        detector_x_tol_mm=detector_x_tol_mm,
        detector_y_range_mm=detector_y_range_mm,
        detector_z_range_mm=detector_z_range_mm
    )

    def _finalize_ion_block(block_ion_n, block_rows):
        if block_ion_n is None or not block_rows:
            return None
        initial_row = block_rows[0]
        final_row = block_rows[-1]

        if not _passes_detector_acceptance(final_row[1], final_row[2], final_row[3]):
            return None

        initial_azm = initial_row[4] if initial_row[4] is not None else final_row[4]
        initial_el = initial_row[5] if initial_row[5] is not None else final_row[5]

        return {
            'ion_n': int(block_ion_n),
            'initial': {
                'x': float(initial_row[1]),
                'y': float(initial_row[2]),
                'z': float(initial_row[3]),
                'azm': None if initial_row[4] is None else float(initial_row[4]),
                'el': None if initial_row[5] is None else float(initial_row[5]),
            },
            'final': {
                'x': float(final_row[1]),
                'y': float(final_row[2]),
                'z': float(final_row[3]),
                'azm': None if final_row[4] is None else float(final_row[4]),
                'el': None if final_row[5] is None else float(final_row[5]),
            },
            'initial_azm': None if initial_azm is None else float(initial_azm),
            'initial_el': None if initial_el is None else float(initial_el),
        }

    ion_records = []
    current_ion_n = None
    current_rows = []
    try:
        with open(out_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parsed = _parse_simion_ion_row(line)
                if parsed is None:
                    continue
                ion_n = parsed[0]
                if current_ion_n is None:
                    current_ion_n = ion_n
                    current_rows = [parsed]
                    continue
                if ion_n != current_ion_n:
                    record = _finalize_ion_block(current_ion_n, current_rows)
                    if record is not None:
                        ion_records.append(record)
                    current_ion_n = ion_n
                    current_rows = [parsed]
                    continue
                current_rows.append(parsed)
    except Exception:
        return [], False

    record = _finalize_ion_block(current_ion_n, current_rows)
    if record is not None:
        ion_records.append(record)

    if not ion_records:
        return [], False

    return ion_records, len(ion_records) > 0




def _convert_ion_records_to_raw_yz(ion_records):
    """
    Convert parsed ion records into minimal raw tuples:
      [ion_n, y, z]
    """
    raw_points = []
    if not isinstance(ion_records, list):
        return raw_points
    for record in ion_records:
        if not isinstance(record, dict):
            continue
        final_pos = record.get('final', {})
        if not isinstance(final_pos, dict):
            continue
        try:
            ion_n = int(record.get('ion_n', -1))
            y = float(final_pos.get('y', 0.0))
            z = float(final_pos.get('z', 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(y) and np.isfinite(z):
            raw_points.append([ion_n, y, z])
    return raw_points




# ============================================================================
# PAIRING AND STATISTICS
# ============================================================================

def _normalize_el_for_pairing(el_value, decimals=6, zero_tol=1e-9):
    try:
        value = float(el_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if abs(value) <= zero_tol:
        value = 0.0
    return round(value, decimals)


def _normalize_az_for_pairing(az_value, decimals=6, zero_tol=1e-9):
    try:
        value = float(az_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    if abs(value) <= zero_tol:
        value = 0.0
    return round(value, decimals)




def _compute_dr_over_r_for_block(ion_block, skip_el_deg=90.0, el_tolerance_deg=1e-6,
                                 az_round_decimals=0, el_round_decimals=0):
    """
    Compute dr/r using YZ plane symmetric azimuth pairing.

    YZ plane symmetry: azm1 + azm2 = ±180°

    This pairs ions that are symmetric about the YZ plane (x=0) at the same elevation.
    """
    def _pair_metric(ion_a, ion_b, el_key):
        fa = ion_a.get('final', {})
        fb = ion_b.get('final', {})
        x_a = float(fa.get('x', 0.0))
        y_a = float(fa.get('y', 0.0))
        z_a = float(fa.get('z', 0.0))
        x_b = float(fb.get('x', 0.0))
        y_b = float(fb.get('y', 0.0))
        z_b = float(fb.get('z', 0.0))

        dx = x_a - x_b
        dy = y_a - y_b
        dz = z_a - z_b
        dr = float(math.sqrt(dx * dx + dy * dy + dz * dz))

        r_a = float(math.sqrt(y_a ** 2 + z_a ** 2))
        r_b = float(math.sqrt(y_b ** 2 + z_b ** 2))
        r_mean = float(0.5 * (r_a + r_b))
        ratio = float(dr / r_mean) if r_mean > 0 else None

        return {
            'initial_el': float(el_key),
            'ion_a': int(ion_a.get('ion_n', -1)),
            'ion_b': int(ion_b.get('ion_n', -1)),
            'x_a': x_a,
            'y_a': y_a,
            'z_a': z_a,
            'x_b': x_b,
            'y_b': y_b,
            'z_b': z_b,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'dr': dr,
            'r_a': r_a,
            'r_b': r_b,
            'r': r_mean,
            'dr_over_r': ratio
        }

    def _are_yz_symmetric(azm1, azm2, tolerance=1.0):
        """Check if two azimuth angles are YZ plane symmetric (sum = ±180°)"""
        sum_angles = float(azm1) + float(azm2)
        return abs(abs(sum_angles) - 180) < tolerance

    pair_details = []
    dr_values = []
    r_values = []
    ratio_values = []
    unmatched_ions = []
    pairing_mode = 'yz_plane_symmetric'

    # Group ions by elevation angle
    elv_groups = {}
    used_indices = set()

    for idx, record in enumerate(ion_block):
        el_key = _normalize_el_for_pairing(record.get('initial_el'), decimals=el_round_decimals)
        azm_key = _normalize_az_for_pairing(record.get('initial_azm'), decimals=az_round_decimals)
        if azm_key is None:
            unmatched_ions.append(int(record.get('ion_n', -1)))
            used_indices.add(idx)
            continue

        # Skip azm = ±90° (no symmetric pair)
        if abs(abs(azm_key) - 90.0) < 1.0:
            unmatched_ions.append(int(record.get('ion_n', -1)))
            used_indices.add(idx)
            continue

        # Skip el = skip_el_deg if requested
        if el_key is not None and abs(float(el_key) - float(skip_el_deg)) <= float(el_tolerance_deg):
            unmatched_ions.append(int(record.get('ion_n', -1)))
            used_indices.add(idx)
            continue

        # Group by elevation
        if el_key not in elv_groups:
            elv_groups[el_key] = []
        elv_groups[el_key].append((idx, record, azm_key))

    # For each elevation group, find YZ symmetric pairs
    for el_key, group in elv_groups.items():
        for i, (idx1, ion1, azm1) in enumerate(group):
            if idx1 in used_indices:
                continue

            # Look for YZ symmetric partner
            for j, (idx2, ion2, azm2) in enumerate(group):
                if i >= j or idx2 in used_indices:
                    continue

                # Check if YZ plane symmetric
                if _are_yz_symmetric(azm1, azm2, tolerance=1.0):
                    pair = _pair_metric(ion1, ion2, el_key if el_key is not None else 0.0)
                    pair_details.append(pair)
                    dr_values.append(pair['dr'])
                    r_values.append(pair['r'])
                    ratio = pair.get('dr_over_r')
                    if ratio is not None and np.isfinite(ratio):
                        ratio_values.append(float(ratio))

                    used_indices.add(idx1)
                    used_indices.add(idx2)
                    break

    # Add remaining unpaired ions
    for idx, record in enumerate(ion_block):
        if idx not in used_indices:
            unmatched_ions.append(int(record.get('ion_n', -1)))

    return {
        'pairing_mode': pairing_mode,
        'pair_details': pair_details,
        'dr_values': dr_values,
        'r_values': r_values,
        'ratio_values': ratio_values,
        'unmatched_ions': unmatched_ions
    }




def _compute_dr_over_r_statistics(ion_records, particles_per_run, requested_runs=1,
                                  skip_el_deg=90.0, el_tolerance_deg=1e-6,
                                  radius_origin_y=-1.0, radius_origin_z=0.0):
    try:
        particles_per_run = int(particles_per_run)
    except (TypeError, ValueError):
        particles_per_run = 0
    if particles_per_run <= 0:
        return {
            'success': False,
            'failure_reason': f"Invalid particles_per_run={particles_per_run}"
        }

    try:
        requested_runs = int(requested_runs)
    except (TypeError, ValueError):
        requested_runs = 1
    if requested_runs <= 0:
        requested_runs = 1

    try:
        radius_origin_y = float(radius_origin_y)
    except (TypeError, ValueError):
        radius_origin_y = -1.0
    try:
        radius_origin_z = float(radius_origin_z)
    except (TypeError, ValueError):
        radius_origin_z = 0.0

    detected_particles_total = len(ion_records)
    max_repeat_blocks = detected_particles_total // particles_per_run
    if max_repeat_blocks <= 0:
        return {
            'success': False,
            'failure_reason': (
                f"Insufficient detected particles ({detected_particles_total}) for one repeat block "
                f"(block size={particles_per_run})"
            )
        }

    repeat_blocks = min(requested_runs, max_repeat_blocks)
    er_runs = []
    max_r_runs = []
    detected_runs = []
    pair_count_runs = []
    pairing_mode_runs = []
    repeat_failures = []

    all_dr_values = []
    all_r_values = []
    all_ratio_values = []
    all_pair_details = []
    all_r_all_points = []
    all_r_max_runs_all_points = []
    all_point_count_runs = []

    for repeat_idx in range(repeat_blocks):
        idx_start = repeat_idx * particles_per_run
        idx_end = idx_start + particles_per_run
        ion_block = ion_records[idx_start:idx_end]
        detected_runs.append(len(ion_block))
        run_all_r_values = []
        for ion in ion_block:
            final_row = ion.get('final', {}) if isinstance(ion, dict) else {}
            try:
                y = float(final_row.get('y', 0.0))
                z = float(final_row.get('z', 0.0))
                dy = y - radius_origin_y
                dz = z - radius_origin_z
                r = float(math.sqrt(dy * dy + dz * dz))
            except (TypeError, ValueError):
                continue
            if np.isfinite(r):
                run_all_r_values.append(r)
        all_point_count_runs.append(len(run_all_r_values))
        all_r_max_runs_all_points.append(float(np.max(run_all_r_values)) if run_all_r_values else None)
        all_r_all_points.extend(run_all_r_values)

        if len(ion_block) < particles_per_run:
            er_runs.append(None)
            max_r_runs.append(None)
            pair_count_runs.append(0)
            repeat_failures.append(
                f"repeat#{repeat_idx + 1}: particles {len(ion_block)}/{particles_per_run}"
            )
            continue

        block_metrics = _compute_dr_over_r_for_block(
            ion_block,
            skip_el_deg=skip_el_deg,
            el_tolerance_deg=el_tolerance_deg
        )
        ratio_values = [
            float(v) for v in block_metrics.get('ratio_values', [])
            if v is not None and np.isfinite(v)
        ]
        r_values = [
            float(v) for v in block_metrics.get('r_values', [])
            if v is not None and np.isfinite(v)
        ]
        dr_values = [
            float(v) for v in block_metrics.get('dr_values', [])
            if v is not None and np.isfinite(v)
        ]
        pair_details = block_metrics.get('pair_details', [])
        pair_count = len(ratio_values)
        pairing_mode = block_metrics.get('pairing_mode', 'same_el')

        if pair_count == 0:
            er_runs.append(None)
            max_r_runs.append(None)
            pair_count_runs.append(0)
            pairing_mode_runs.append(pairing_mode)
            repeat_failures.append(
                f"repeat#{repeat_idx + 1}: no valid mirrored pairs (skip el={skip_el_deg})"
            )
            continue

        er_runs.append(float(np.mean(ratio_values)))
        max_r_runs.append(float(np.max(r_values)) if r_values else None)
        pair_count_runs.append(pair_count)
        pairing_mode_runs.append(pairing_mode)
        all_ratio_values.extend(ratio_values)
        all_r_values.extend(r_values)
        all_dr_values.extend(dr_values)

        for pair in pair_details:
            pair_record = dict(pair)
            pair_record['run_index'] = repeat_idx + 1
            all_pair_details.append(pair_record)

        # Unmatched ions can be expected when skipping el=90 in an odd-length sweep.

    valid_er_runs = [float(v) for v in er_runs if v is not None and np.isfinite(v)]
    if not all_ratio_values:
        failure_reason = "All repeat blocks failed dr/r pairing analysis"
        if repeat_failures:
            failure_reason += f" ({'; '.join(repeat_failures[:2])})"
        return {
            'success': False,
            'failure_reason': failure_reason,
            'detected_particles_total': detected_particles_total,
            'detected_particles_runs': detected_runs,
            'pair_count_runs': pair_count_runs
        }

    ratio_mean = float(np.mean(all_ratio_values))
    ratio_var = float(np.var(all_ratio_values))
    ratio_std = float(np.sqrt(ratio_var))

    valid_max_r = [float(v) for v in max_r_runs if v is not None and np.isfinite(v)]
    max_r_mean = float(np.mean(valid_max_r)) if valid_max_r else None
    max_r_var = float(np.var(valid_max_r)) if valid_max_r else None
    max_r_std = float(np.sqrt(max_r_var)) if max_r_var is not None else None
    r_max_all_points = float(np.max(all_r_all_points)) if all_r_all_points else None

    failure_reason_summary = None
    if repeat_failures:
        failure_reason_summary = '; '.join(repeat_failures[:3])
        if len(repeat_failures) > 3:
            failure_reason_summary += f"; +{len(repeat_failures) - 3} more"

    return {
        'success': True,
        'detected_particles_total': detected_particles_total,
        'detected_particles_runs': detected_runs,
        'detected_particles_mean': int(round(float(np.mean(detected_runs)))) if detected_runs else None,
        'pair_count_runs': pair_count_runs,
        'pairing_mode_runs': pairing_mode_runs,
        'pair_count_total': len(all_ratio_values),
        'valid_run_count': len(valid_er_runs),
        'total_runs_used': repeat_blocks,
        'energy_resolution_runs': er_runs,
        'energy_resolution_mean': ratio_mean,
        'energy_resolution_variance': ratio_var,
        'energy_resolution_std': ratio_std,
        'max_r_runs': max_r_runs,
        'max_r_mean': max_r_mean,
        'max_r_variance': max_r_var,
        'max_r_std': max_r_std,
        'r_max_all_points': r_max_all_points,
        'r_max_all_points_runs': all_r_max_runs_all_points,
        'all_point_count_runs': all_point_count_runs,
        'dr_values': all_dr_values,
        'r_values': all_r_values,
        'dr_over_r_values': all_ratio_values,
        'pair_details': all_pair_details,
        'failure_reason': failure_reason_summary
    }


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_invalid_energy_result(expected_particles_total, detected_particles,
                                 particles_per_run, requested_runs,
                                 failure_reason, pipeline_stage='count_check'):
    block_size = _safe_int(particles_per_run)
    if block_size is None or block_size <= 0:
        block_size = 1
    total_runs = _safe_int(requested_runs)
    if total_runs is None or total_runs <= 0:
        total_runs = 1
    detected_total = _safe_int(detected_particles)
    if detected_total is None or detected_total < 0:
        detected_total = 0

    detected_runs = []
    full_blocks = detected_total // block_size
    for _ in range(full_blocks):
        detected_runs.append(block_size)
    remainder = detected_total - (full_blocks * block_size)
    if remainder > 0:
        detected_runs.append(remainder)

    return {
        'fwhm': None,
        'fwhm_mean': None,
        'fwhm_variance': None,
        'fwhm_std': None,
        'fwhm_runs': [],
        'energy_resolution': None,
        'energy_resolution_mean': None,
        'energy_resolution_variance': None,
        'energy_resolution_std': None,
        'energy_resolution_runs': [],
        'max_r': None,
        'max_r_mean': None,
        'max_r_variance': None,
        'max_r_std': None,
        'max_r_runs': [],
        'r_max_all_points': None,
        'r_max_all_points_runs': [],
        'all_point_count_runs': [],
        'generated_particles': _safe_int(expected_particles_total),
        'generated_particles_per_run': block_size,
        'detected_particles': detected_total,
        'detected_particles_runs': detected_runs,
        'pair_count': 0,
        'pair_count_runs': [],
        'dr_values': [],
        'r_values': [],
        'dr_over_r_values': [],
        'dr_over_r_pairs': [],
        'raw_ion_points_yz': [],
        'raw_point_count': 0,
        'raw_point_format': 'ion_n,y,z',
        'total_runs': total_runs,
        'valid_run_count': 0,
        'valid': False,
        'failure_reason': failure_reason,
        'count_check_passed': False,
        'plot_marker': 'x',
        'plot_skip': True,
        'pipeline_stage': pipeline_stage
    }


def _compute_dr_over_rmax_statistics_from_records(ion_records, particles_per_run, requested_runs=1,
                                                  radius_origin_y=0.0, radius_origin_z=0.0):
    """
    Pipeline:
      1) (already done outside) particle-count validation
      2) pair mirrored points inside each run block
      3) compute dr/rmax using rmax across all detected points of this node
    """
    block_size = _safe_int(particles_per_run)
    if block_size is None or block_size <= 1:
        return {
            'success': False,
            'failure_reason': f"Invalid particles_per_run={particles_per_run}"
        }

    runs_requested = _safe_int(requested_runs)
    if runs_requested is None or runs_requested <= 0:
        runs_requested = 1

    try:
        origin_y = float(radius_origin_y)
    except (TypeError, ValueError):
        origin_y = 0.0
    try:
        origin_z = float(radius_origin_z)
    except (TypeError, ValueError):
        origin_z = 0.0

    if not isinstance(ion_records, list) or not ion_records:
        return {
            'success': False,
            'failure_reason': "No ion records available for pairing"
        }

    detected_particles_total = len(ion_records)
    max_repeat_blocks = detected_particles_total // block_size
    if max_repeat_blocks <= 0:
        return {
            'success': False,
            'failure_reason': (
                f"Insufficient detected particles ({detected_particles_total}) for one repeat block "
                f"(block size={block_size})"
            )
        }

    repeat_blocks = min(runs_requested, max_repeat_blocks)
    detected_runs = [block_size for _ in range(repeat_blocks)]

    all_r_values = []
    all_r_max_runs = []
    all_point_count_runs = []
    for run_idx in range(repeat_blocks):
        start = run_idx * block_size
        stop = start + block_size
        ion_block = ion_records[start:stop]
        run_r = []
        for ion in ion_block:
            final_pos = ion.get('final', {}) if isinstance(ion, dict) else {}
            try:
                y = float(final_pos.get('y', 0.0))
                z = float(final_pos.get('z', 0.0))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(y) and np.isfinite(z)):
                continue
            r = float(math.sqrt((y - origin_y) ** 2 + (z - origin_z) ** 2))
            if np.isfinite(r):
                run_r.append(r)
        all_point_count_runs.append(len(run_r))
        all_r_max_runs.append(float(np.max(run_r)) if run_r else None)
        all_r_values.extend(run_r)

    if not all_r_values:
        return {
            'success': False,
            'failure_reason': "No finite detector radii found"
        }

    rmax_all_points = float(np.max(all_r_values))
    if rmax_all_points <= 0:
        return {
            'success': False,
            'failure_reason': "Computed rmax is not positive"
        }

    pair_count_runs = []
    run_er_values = []
    all_ratio_values = []
    all_dr_values = []
    all_r_mean_values = []
    pair_details = []
    repeat_failures = []

    for run_idx in range(repeat_blocks):
        start = run_idx * block_size
        stop = start + block_size
        ion_block = ion_records[start:stop]

        block_metrics = _compute_dr_over_r_for_block(
            ion_block,
            skip_el_deg=90.0,
            el_tolerance_deg=1e-6
        )
        block_pairs = block_metrics.get('pair_details', [])
        if not isinstance(block_pairs, list):
            block_pairs = []

        run_ratios = []
        run_pair_count = 0
        for pair in block_pairs:
            try:
                y_a = float(pair.get('y_a'))
                z_a = float(pair.get('z_a'))
                y_b = float(pair.get('y_b'))
                z_b = float(pair.get('z_b'))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(y_a) and np.isfinite(z_a) and np.isfinite(y_b) and np.isfinite(z_b)):
                continue

            r_a = float(math.sqrt((y_a - origin_y) ** 2 + (z_a - origin_z) ** 2))
            r_b = float(math.sqrt((y_b - origin_y) ** 2 + (z_b - origin_z) ** 2))
            dr = float(abs(r_a - r_b))
            r_mean = float(0.5 * (r_a + r_b))
            ratio = float(dr / rmax_all_points) if rmax_all_points > 0 else None

            if ratio is not None and np.isfinite(ratio):
                run_pair_count += 1
                run_ratios.append(ratio)
                all_ratio_values.append(ratio)
                all_dr_values.append(dr)
                all_r_mean_values.append(r_mean)
                pair_details.append({
                    'run_index': run_idx + 1,
                    'ion_a': int(pair.get('ion_a', -1)),
                    'ion_b': int(pair.get('ion_b', -1)),
                    'y_a': y_a,
                    'z_a': z_a,
                    'y_b': y_b,
                    'z_b': z_b,
                    'r_a': r_a,
                    'r_b': r_b,
                    'dr': dr,
                    'r': r_mean,
                    'dr_over_r': ratio,
                    'initial_el': pair.get('initial_el')
                })

        pair_count_runs.append(run_pair_count)
        if run_ratios:
            run_er_values.append(float(np.mean(run_ratios)))
        else:
            run_er_values.append(None)
            repeat_failures.append(f"repeat#{run_idx + 1}: no valid az/el symmetric pairs")

    if not all_ratio_values:
        return {
            'success': False,
            'failure_reason': "No valid mirrored pairs found for dr/rmax"
        }

    ratio_arr = np.array(all_ratio_values, dtype=float)
    max_r_arr = np.array([v for v in all_r_max_runs if v is not None], dtype=float)
    er_runs_clean = [float(v) for v in run_er_values if v is not None and np.isfinite(v)]

    max_r_mean = float(np.mean(max_r_arr)) if max_r_arr.size > 0 else None
    max_r_var = float(np.var(max_r_arr)) if max_r_arr.size > 0 else None
    max_r_std = float(np.sqrt(max_r_var)) if max_r_var is not None else None
    failure_reason_summary = None
    if repeat_failures:
        failure_reason_summary = '; '.join(repeat_failures[:3])
        if len(repeat_failures) > 3:
            failure_reason_summary += f"; +{len(repeat_failures) - 3} more"

    return {
        'success': True,
        'detected_particles_total': detected_particles_total,
        'detected_particles_runs': detected_runs,
        'detected_particles_mean': int(round(float(np.mean(detected_runs)))) if detected_runs else detected_particles_total,
        'pair_count_runs': pair_count_runs,
        'pair_count_total': int(np.sum(pair_count_runs)),
        'valid_run_count': len(er_runs_clean),
        'total_runs_used': repeat_blocks,
        'energy_resolution_runs': run_er_values,
        'energy_resolution_mean': float(np.mean(ratio_arr)),
        'energy_resolution_variance': float(np.var(ratio_arr)),
        'energy_resolution_std': float(np.std(ratio_arr)),
        'max_r_runs': all_r_max_runs,
        'max_r_mean': max_r_mean,
        'max_r_variance': max_r_var,
        'max_r_std': max_r_std,
        'r_max_all_points': rmax_all_points,
        'r_max_all_points_runs': all_r_max_runs,
        'all_point_count_runs': all_point_count_runs,
        'dr_values': all_dr_values,
        'r_values': all_r_mean_values,
        'dr_over_r_values': all_ratio_values,
        'pair_details': pair_details,
        'failure_reason': failure_reason_summary
    }




# ============================================================================
# ENERGY RESOLUTION ANALYSIS
# ============================================================================

def energy_resolution_analysis_direct(processed_data, all_combinations,
                                      source_position=(199, 0, 0),
                                      num_particles_per_energy=10000,
                                      num_statistical_repeats=1,
                                      x_scan_range=(73.0, 166.0),
                                      bin_interval=0.05,
                                      outside_region_width=2,
                                      batch_size=50,
                                      enable_memory_optimization=True,
                                      checkpoint_interval=100,
                                      max_combo_retries=5,
                                      require_full_particle_capture=True,
                                      retry_backoff_s=0.5,
                                      timing_verbose=True,
                                      detector_x_mm=73.0,
                                      detector_x_tol_mm=0.5,
                                      detector_y_range_mm=(-35.0, 35.0),
                                      detector_z_range_mm=(-35.0, 35.0),
                                      gc_interval_combos=50,
                                      intraction_volume = False,
                                      ionization_volume_array_mm=0,
                                      checkpoint_name='energy_resolution_direct'):
    """
    Perform energy resolution analysis directly on specified combinations WITHOUT focus filtering.

    MEMORY-OPTIMIZED VERSION:
    - Uses regular dicts instead of defaultdict to avoid memory leaks
    - Aggressive cleanup after each combination
    - Periodic forced garbage collection
    - Minimal data retention in fwhm_results

    Args:
        processed_data: Dictionary containing processed SIMION data
        all_combinations: List of (fg, lens_vmi, ke) tuples to analyze
        source_position: Fixed position for particle source (default: (199, 0, 0))
        num_particles_per_energy: Number of particles per energy point (default: 10000)
        num_statistical_repeats: Statistical repeats per node in ONE SIMION run.
            Internally generates num_particles_per_energy * num_statistical_repeats
            particles and splits detected particles into repeat blocks.
        x_scan_range: Range of x-planes to scan for focus analysis (default: (73.0, 166.0))
        bin_interval: Bin size for rectangular coordinates (default: 0.05 mm)
        outside_region_width: Width of region outside data area (default: 2 mm)
        batch_size: Number of combinations to process per batch (default: 50)
        enable_memory_optimization: Enable aggressive memory cleanup (default: True)
        checkpoint_interval: Save checkpoint shard every N attempted combinations (default: 100, 0 to disable)
        gc_interval_combos: Run gc.collect() every N attempted combinations (default: 50, 0 to disable)
        detector_x_mm: Detector x center in mm (None to disable detector filtering)
        detector_x_tol_mm: Detector x acceptance half-width in mm
        detector_y_range_mm: Detector y acceptance range (min, max) in mm
        detector_z_range_mm: Detector z acceptance range (min, max) in mm

    Returns:
        dict: Updated processed_data with raw ion payload (`ion_n,y,z`) in global section
    """
    import numpy as np

    try:
        num_statistical_repeats = int(num_statistical_repeats)
    except (TypeError, ValueError):
        num_statistical_repeats = 1
    if num_statistical_repeats <= 0:
        num_statistical_repeats = 1

    try:
        gc_interval_combos = int(gc_interval_combos)
    except (TypeError, ValueError):
        gc_interval_combos = 0
    if gc_interval_combos < 0:
        gc_interval_combos = 0

    total_combinations = len(all_combinations)
    if total_combinations == 0:
        print("No combinations provided for energy resolution analysis!")
        return processed_data

    print(
        f"Direct energy resolution analysis: {total_combinations} combinations (no focus filtering), "
        f"stat repeats per node={num_statistical_repeats}"
    )
    detector_filter_enabled = (
        detector_x_mm is not None
        and detector_x_tol_mm is not None
        and detector_y_range_mm is not None
        and detector_z_range_mm is not None
    )
    if detector_filter_enabled:
        print(
            f"Detector acceptance enabled: x={detector_x_mm}±{detector_x_tol_mm} mm, "
            f"y={tuple(detector_y_range_mm)} mm, z={tuple(detector_z_range_mm)} mm"
        )

    def _norm_combo(c):
        fg, lens_vmi, ke = c
        return (float(fg), float(lens_vmi), round(float(ke), 9))

    requested_norm = set(_norm_combo(c) for c in all_combinations)

    # Try to load from checkpoint for recovery
    checkpoint_base_name = str(checkpoint_name or 'energy_resolution_direct')
    loaded_results, completed_combinations, last_checkpoint = load_latest_checkpoint(checkpoint_base_name)

    if loaded_results is not None:
        checkpoint_marker_count = len(completed_combinations)
        if checkpoint_marker_count > 0:
            print(f"  Resuming from checkpoint: {checkpoint_marker_count} combinations already completed")
        else:
            print("  Loaded checkpoint file with 0 completion markers; recovering from stored results...")
        # Keep only checkpoint entries that belong to the current parameter request
        fwhm_results = {}
        dropped_incomplete = 0
        for fg, fg_data in (loaded_results or {}).items():
            for lens_vmi, lens_data in (fg_data or {}).items():
                for ke, result in (lens_data or {}).items():
                    combo_norm = _norm_combo((fg, lens_vmi, ke))
                    if combo_norm not in requested_norm:
                        continue
                    if not _checkpoint_result_has_required_pair_details(result):
                        dropped_incomplete += 1
                        continue
                    if fg not in fwhm_results:
                        fwhm_results[fg] = {}
                    if lens_vmi not in fwhm_results[fg]:
                        fwhm_results[fg][lens_vmi] = {}
                    fwhm_results[fg][lens_vmi][ke] = result

        # IMPORTANT: Use actual stored results as source of truth.
        # Do not trust completed-list alone, it can contain stale entries from old runs.
        completed_norm = set()
        for fg, fg_data in fwhm_results.items():
            for lens_vmi, lens_data in fg_data.items():
                for ke in lens_data.keys():
                    completed_norm.add(_norm_combo((fg, lens_vmi, ke)))

        completed_combinations = set(completed_norm)
        stale_markers = checkpoint_marker_count - len(completed_norm)
        if stale_markers > 0:
            print(f"  Ignored {stale_markers} stale checkpoint completion markers")
        if dropped_incomplete > 0:
            print(
                f"  Dropped {dropped_incomplete} incomplete checkpoint nodes "
                f"(missing recoverable payload: dr_over_r_pairs or raw_ion_points_yz)"
            )
        remaining_combinations = [c for c in all_combinations if _norm_combo(c) not in completed_norm]
        print(f"  Remaining combinations: {len(remaining_combinations)}")
    else:
        # Use regular dict instead of defaultdict to avoid memory leaks
        fwhm_results = {}
        completed_combinations = set()
        remaining_combinations = all_combinations

    if len(remaining_combinations) == 0:
        print("  All combinations already completed!")
        # Still need to update processed_data with loaded results
        for fg in fwhm_results:
            if fg not in processed_data:
                continue
            for lens_vmi in fwhm_results[fg]:
                if lens_vmi not in processed_data[fg]:
                    continue
                for ke in fwhm_results[fg][lens_vmi]:
                    if ke not in processed_data[fg][lens_vmi]:
                        continue
                    if 'global' not in processed_data[fg][lens_vmi][ke]:
                        processed_data[fg][lens_vmi][ke]['global'] = {}
                    processed_data[fg][lens_vmi][ke]['global'].update(fwhm_results[fg][lens_vmi][ke])
        consolidate_checkpoints(checkpoint_base_name, cleanup_parts=True)
        return processed_data

    # Calculate number of batches for remaining combinations
    num_batches = (len(remaining_combinations) + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of up to {batch_size} combinations each")

    if enable_memory_optimization:
        initial_memory = get_memory_usage_mb()
        if initial_memory > 0:
            print(f"Initial memory usage: {initial_memory:.1f} MB")

    # Files for energy resolution analysis (must be consistent with selected IOB project)
    OUTPUT_FILENAME_FLY2, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE = _resolve_energy_resolution_project_files()
    if IOB_FILE is None:
        print("ERROR: IOB file not found! Checked 'WORKING_TITLE_energy_resolution_tao.iob' and 'WORKING_TITLE_tao.iob'")
        return processed_data
    print(f"Using project files: IOB={IOB_FILE}, LUA={OUTPUT_FILENAME_LUA}, FLY2={OUTPUT_FILENAME_FLY2}")

    # Track progress
    processed_count = total_combinations - len(remaining_combinations)
    attempted_count = total_combinations - len(remaining_combinations)
    failed_count = 0
    start_time = time.time()
    timing_totals = {
        'particle_gen_s': 0.0,
        'simion_s': 0.0,
        'parse_s': 0.0,
        'extract_s': 0.0,
        'bin_abel_s': 0.0
    }
    timing_attempts = 0
    checkpoint_chunk_results = {}
    checkpoint_chunk_completed = set()

    # Memory monitoring
    MEMORY_WARNING_MB = 4000  # Warn at 4GB
    MEMORY_CRITICAL_MB = 6000  # Force cleanup at 6GB

    # Process in batches
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(remaining_combinations))
        batch_combinations = remaining_combinations[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_combinations)} combinations)")
        print(f"Progress: {processed_count}/{total_combinations} ({100*processed_count/total_combinations:.1f}%)")

        if enable_memory_optimization:
            current_memory = get_memory_usage_mb()
            if current_memory > 0:
                print(f"Current memory usage: {current_memory:.1f} MB")

                # Critical memory check - force aggressive cleanup
                if current_memory > MEMORY_CRITICAL_MB:
                    print(f"  [CRITICAL] Memory above {MEMORY_CRITICAL_MB}MB, forcing aggressive cleanup...")
                    gc.collect(0)
                    gc.collect(1)
                    gc.collect(2)
                    gc.collect()
                    current_memory = get_memory_usage_mb()
                    print(f"  Memory after cleanup: {current_memory:.1f} MB")

        batch_start_time = time.time()

        def _cleanup_single_processed_data(obj):
            if obj is None:
                return
            try:
                for fg_key in list(obj.keys()):
                    for lens_key in list(obj[fg_key].keys()):
                        for ke_key in list(obj[fg_key][lens_key].keys()):
                            if 'local' in obj[fg_key][lens_key][ke_key]:
                                obj[fg_key][lens_key][ke_key]['local'].clear()
                            if 'global' in obj[fg_key][lens_key][ke_key]:
                                obj[fg_key][lens_key][ke_key]['global'].clear()
                        obj[fg_key][lens_key].clear()
                    obj[fg_key].clear()
                obj.clear()
            except Exception:
                pass

        for combo_idx, (fg, lens_vmi, ke) in enumerate(batch_combinations):
            combo_num = attempted_count + 1
            attempted_count += 1
            print(f"  [{combo_num}/{total_combinations}] Processing FG {fg}, Lens {lens_vmi}, KE {ke:.2f} eV...")

            combo_success = False
            last_failure_reason = "unknown"
            last_detected_particles = 0
            combo_timing = {
                'particle_gen_s': 0.0,
                'simion_s': 0.0,
                'parse_s': 0.0,
                'extract_s': 0.0,
                'bin_abel_s': 0.0
            }

            for attempt in range(1, max_combo_retries + 1):
                single_processed_data = None
                ion_records = None
                dr_over_r_stats = None
                attempt_out_file = None
                detected_particles = 0
                expected_particles_total = int(num_particles_per_energy) * int(num_statistical_repeats)
                out_file_size_mb = 0.0
                attempt_timing = {
                    'particle_gen_s': 0.0,
                    'simion_s': 0.0,
                    'parse_s': 0.0,
                    'extract_s': 0.0,
                    'bin_abel_s': 0.0
                }

                try:
                    t0 = time.time()
                    energy_resolution_utilis(
                        filename=OUTPUT_FILENAME_FLY2,
                        position=source_position,
                        num_particles=int(num_particles_per_energy),
                        ke=ke,
                        num_groups=int(num_statistical_repeats),
                        az_first_deg=0.0,
                        az_step_deg=30.0,
                        az_num=7,
                        el_first_deg=0.0,
                        el_step_deg=30.0,
                        el_num =13,
                        intraction_volume = intraction_volume,
                        intraction_volume_array_mm = ionization_volume_array_mm
                    )
                    attempt_timing['particle_gen_s'] = time.time() - t0

                    single_param = get_parameters_for_combination(processed_data, fg, lens_vmi, ke)
                    attempt_out_file = make_energy_resolution_temp_out_file(OUT_FILE)

                    t0 = time.time()
                    sim_stats = run_optimized_simulations_with_ke(
                        single_param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, attempt_out_file
                    )
                    attempt_timing['simion_s'] = time.time() - t0
                    if isinstance(sim_stats, dict) and sim_stats.get('successful', 0) <= 0:
                        last_failure_reason = f"SIMION failed (stats={sim_stats})"
                        raise RuntimeError(last_failure_reason)
                    if not os.path.exists(attempt_out_file):
                        last_failure_reason = f"SIMION output missing: {attempt_out_file}"
                        raise RuntimeError(last_failure_reason)
                    out_file_size_mb = os.path.getsize(attempt_out_file) / (1024 * 1024)
                    if out_file_size_mb <= 0:
                        last_failure_reason = "SIMION output file is empty"
                        raise RuntimeError(last_failure_reason)

                    t0 = time.time()
                    ion_records, success = _extract_ion_records_from_out_file(
                        attempt_out_file,
                        detector_x_mm=detector_x_mm,
                        detector_x_tol_mm=detector_x_tol_mm,
                        detector_y_range_mm=detector_y_range_mm,
                        detector_z_range_mm=detector_z_range_mm
                    )
                    attempt_timing['parse_s'] = time.time() - t0
                    attempt_timing['extract_s'] = 0.0
                    if not success or not ion_records:
                        last_failure_reason = "No ion rows found in output"
                        raise RuntimeError(last_failure_reason)

                    detected_particles = len(ion_records)
                    last_detected_particles = detected_particles
                    if require_full_particle_capture and detected_particles != expected_particles_total:
                        if detector_filter_enabled:
                            last_failure_reason = (
                                f"Particle mismatch after detector filter: detected {detected_particles} "
                                f"!= expected {expected_particles_total}"
                            )
                        else:
                            last_failure_reason = (
                                f"Particle mismatch: detected {detected_particles} != expected {expected_particles_total}"
                            )
                        raise RuntimeError(last_failure_reason)

                    t0 = time.time()
                    raw_ion_points = _convert_ion_records_to_raw_yz(ion_records)
                    if not raw_ion_points:
                        last_failure_reason = "No valid ion_n/y/z rows after extraction"
                        raise RuntimeError(last_failure_reason)

                    # Pipeline step 2-3: pair first, then compute dr/rmax.
                    dr_over_r_stats = _compute_dr_over_rmax_statistics_from_records(
                        ion_records=ion_records,
                        particles_per_run=num_particles_per_energy,
                        requested_runs=num_statistical_repeats,
                        radius_origin_y=source_position[1] if len(source_position) > 1 else 0.0,
                        radius_origin_z=source_position[2] if len(source_position) > 2 else 0.0
                    )
                    attempt_timing['bin_abel_s'] = time.time() - t0
                    if not dr_over_r_stats.get('success'):
                        last_failure_reason = dr_over_r_stats.get('failure_reason', 'Pairing/dr-rmax failed')
                        raise RuntimeError(last_failure_reason)

                    fwhm_runs = []
                    fwhm_mean = None
                    fwhm_var = None
                    fwhm_std = None

                    er_runs = dr_over_r_stats.get('energy_resolution_runs', [])
                    valid_run_count = int(dr_over_r_stats.get('valid_run_count', 0))
                    er_mean = dr_over_r_stats.get('energy_resolution_mean')
                    er_var = dr_over_r_stats.get('energy_resolution_variance')
                    er_std = dr_over_r_stats.get('energy_resolution_std')

                    max_r_runs = dr_over_r_stats.get('max_r_runs', [])
                    max_r_mean = dr_over_r_stats.get('max_r_mean')
                    max_r_var = dr_over_r_stats.get('max_r_variance')
                    max_r_std = dr_over_r_stats.get('max_r_std')

                    detected_runs = dr_over_r_stats.get('detected_particles_runs', [])
                    detected_particles_mean = dr_over_r_stats.get('detected_particles_mean', detected_particles)
                    failure_reason_summary = dr_over_r_stats.get('failure_reason')
                    pair_count_total = int(dr_over_r_stats.get('pair_count_total', 0))

                    if fg not in fwhm_results:
                        fwhm_results[fg] = {}
                    if lens_vmi not in fwhm_results[fg]:
                        fwhm_results[fg][lens_vmi] = {}

                    fwhm_results[fg][lens_vmi][ke] = {
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
                        'r_max_all_points': dr_over_r_stats.get('r_max_all_points'),
                        'r_max_all_points_runs': dr_over_r_stats.get('r_max_all_points_runs', []),
                        'all_point_count_runs': dr_over_r_stats.get('all_point_count_runs', []),
                        'generated_particles': expected_particles_total,
                        'generated_particles_per_run': int(num_particles_per_energy),
                        'detected_particles': dr_over_r_stats.get('detected_particles_total', detected_particles),
                        'detected_particles_runs': detected_runs,
                        'pair_count': pair_count_total,
                        'pair_count_runs': dr_over_r_stats.get('pair_count_runs', []),
                        'dr_values': dr_over_r_stats.get('dr_values', []),
                        'r_values': dr_over_r_stats.get('r_values', []),
                        'dr_over_r_values': dr_over_r_stats.get('dr_over_r_values', []),
                        'dr_over_r_pairs': dr_over_r_stats.get('pair_details', []),
                        'raw_ion_points_yz': raw_ion_points,
                        'raw_point_count': len(raw_ion_points),
                        'raw_point_format': 'ion_n,y,z',
                        'total_runs': int(num_statistical_repeats),
                        'valid_run_count': valid_run_count,
                        'valid': bool(valid_run_count > 0 and er_mean is not None),
                        'failure_reason': failure_reason_summary,
                        'count_check_passed': True,
                        'plot_marker': None,
                        'plot_skip': False,
                        'pipeline_stage': 'dr_over_rmax'
                    }
                    _set_checkpoint_node(
                        checkpoint_chunk_results,
                        fg,
                        lens_vmi,
                        ke,
                        fwhm_results[fg][lens_vmi][ke]
                    )

                    completed_combinations.add((fg, lens_vmi, ke))
                    checkpoint_chunk_completed.add((fg, lens_vmi, ke))
                    processed_count += 1
                    combo_success = True
                    timing_attempts += 1
                    for key in timing_totals:
                        timing_totals[key] += attempt_timing[key]
                        combo_timing[key] += attempt_timing[key]

                    if timing_verbose:
                        attempt_total_s = sum(attempt_timing.values())
                        print(
                            f"    Attempt {attempt} OK: detected={detected_particles}/{expected_particles_total}, "
                            f"pairs={pair_count_total}, "
                            f"valid_repeats={valid_run_count}/{num_statistical_repeats}, "
                            f"out={out_file_size_mb:.2f}MB, "
                            f"timing[s] gen={attempt_timing['particle_gen_s']:.2f} "
                            f"simion={attempt_timing['simion_s']:.2f} "
                            f"parse={attempt_timing['parse_s']:.2f} "
                            f"extract={attempt_timing['extract_s']:.2f} "
                            f"pair-metric={attempt_timing['bin_abel_s']:.2f} "
                            f"total={attempt_total_s:.2f}"
                        )
                    break

                except Exception as e:
                    if not last_failure_reason or last_failure_reason == "unknown":
                        last_failure_reason = str(e)
                    for key in combo_timing:
                        combo_timing[key] += attempt_timing[key]
                    if timing_verbose:
                        attempt_total_s = sum(attempt_timing.values())
                        print(
                            f"    Attempt {attempt} FAILED: {last_failure_reason}; "
                            f"detected={detected_particles}/{expected_particles_total}, "
                            f"out={out_file_size_mb:.2f}MB, "
                            f"timing[s] gen={attempt_timing['particle_gen_s']:.2f} "
                            f"simion={attempt_timing['simion_s']:.2f} "
                            f"parse={attempt_timing['parse_s']:.2f} "
                            f"extract={attempt_timing['extract_s']:.2f} "
                            f"pair-metric={attempt_timing['bin_abel_s']:.2f} "
                            f"total={attempt_total_s:.2f}"
                        )
                    is_particle_mismatch = isinstance(last_failure_reason, str) and last_failure_reason.startswith("Particle mismatch")
                    if is_particle_mismatch:
                        print("    Mark current combination invalid immediately (requires 100% 4π capture).")
                        break
                    if attempt < max_combo_retries:
                        print(f"    Retry {attempt}/{max_combo_retries} due to: {last_failure_reason}")
                        time.sleep(retry_backoff_s * attempt)
                    else:
                        print(f"    ERROR: {last_failure_reason}")

                finally:
                    remove_file_with_retry(attempt_out_file)
                    _cleanup_single_processed_data(single_processed_data)
                    if ion_records is not None:
                        del ion_records
                    if dr_over_r_stats is not None:
                        del dr_over_r_stats

            if not combo_success:
                failed_count += 1
                if fg not in fwhm_results:
                    fwhm_results[fg] = {}
                if lens_vmi not in fwhm_results[fg]:
                    fwhm_results[fg][lens_vmi] = {}
                pipeline_stage = 'count_check' if isinstance(last_failure_reason, str) and 'Particle mismatch' in last_failure_reason else 'failed'
                fwhm_results[fg][lens_vmi][ke] = _build_invalid_energy_result(
                    expected_particles_total=expected_particles_total,
                    detected_particles=last_detected_particles,
                    particles_per_run=num_particles_per_energy,
                    requested_runs=num_statistical_repeats,
                    failure_reason=last_failure_reason,
                    pipeline_stage=pipeline_stage
                )
                _set_checkpoint_node(
                    checkpoint_chunk_results,
                    fg,
                    lens_vmi,
                    ke,
                    fwhm_results[fg][lens_vmi][ke]
                )
            if gc_interval_combos > 0 and attempted_count > 0 and attempted_count % gc_interval_combos == 0:
                gc.collect()

            # Save incremental checkpoint shard by attempts.
            if checkpoint_interval > 0 and attempted_count > 0 and attempted_count % checkpoint_interval == 0:
                if _count_checkpoint_nodes(checkpoint_chunk_results) > 0:
                    save_checkpoint(
                        checkpoint_chunk_results,
                        checkpoint_base_name,
                        attempted_count,
                        checkpoint_chunk_completed
                    )
                    checkpoint_chunk_results = {}
                    checkpoint_chunk_completed = set()

        # End of batch - flush incremental checkpoint shard
        if _count_checkpoint_nodes(checkpoint_chunk_results) > 0:
            save_checkpoint(
                checkpoint_chunk_results,
                checkpoint_base_name,
                attempted_count,
                checkpoint_chunk_completed
            )
            checkpoint_chunk_results = {}
            checkpoint_chunk_completed = set()

        # End of batch - aggressive cleanup
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx + 1} completed in {batch_time:.1f}s")

        if enable_memory_optimization:
            # Aggressive multi-generation garbage collection
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            gc.collect()
            current_memory = get_memory_usage_mb()
            if current_memory > 0:
                print(f"Memory after cleanup: {current_memory:.1f} MB")

    # Store FWHM results in processed_data global section
    print(f"\nStoring results in processed_data...")
    for fg in fwhm_results:
        if fg not in processed_data:
            processed_data[fg] = {}
        for lens_vmi in fwhm_results[fg]:
            if lens_vmi not in processed_data[fg]:
                processed_data[fg][lens_vmi] = {}
            for ke in fwhm_results[fg][lens_vmi]:
                if ke not in processed_data[fg][lens_vmi]:
                    processed_data[fg][lens_vmi][ke] = {'global': {}, 'local': {}}
                if 'global' not in processed_data[fg][lens_vmi][ke]:
                    processed_data[fg][lens_vmi][ke]['global'] = {}

                # Store FWHM results in global section
                processed_data[fg][lens_vmi][ke]['global'].update(fwhm_results[fg][lens_vmi][ke])

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Direct energy resolution analysis completed!")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if processed_count > 0:
        print(f"  Average time per combination: {total_time/processed_count:.2f}s")
    if timing_attempts > 0:
        total_stage_s = sum(timing_totals.values())
        if total_stage_s > 0:
            print("  Timing breakdown (successful attempts):")
            print(f"    particle generation: {timing_totals['particle_gen_s']:.2f}s ({100*timing_totals['particle_gen_s']/total_stage_s:.1f}%)")
            print(f"    SIMION run:          {timing_totals['simion_s']:.2f}s ({100*timing_totals['simion_s']/total_stage_s:.1f}%)")
            print(f"    output parsing:      {timing_totals['parse_s']:.2f}s ({100*timing_totals['parse_s']/total_stage_s:.1f}%)")
            print(f"    final extraction:    {timing_totals['extract_s']:.2f}s ({100*timing_totals['extract_s']/total_stage_s:.1f}%)")
            print(f"    dr/r pairing:        {timing_totals['bin_abel_s']:.2f}s ({100*timing_totals['bin_abel_s']/total_stage_s:.1f}%)")

    # Consolidate checkpoint shards after successful completion
    consolidate_checkpoints(checkpoint_base_name, cleanup_parts=True)

    return processed_data




# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_heatmap_all_fg(processed_data, cmap='viridis', figsize=(14, 10),
                        vmin=None, vmax=None):
    """
    Plot heatmaps of energy resolution for all field gradients in the data.
    Creates a slider to switch between different field gradients.

    Args:
        processed_data: Dictionary containing processed SIMION data
        cmap: Colormap to use (default: 'viridis')
        figsize: Figure size as tuple (width, height) in inches
        vmin: Minimum value for colorbar (default: auto across all data)
        vmax: Maximum value for colorbar (default: auto across all data)

    Returns:
        None (displays interactive plot)
    """
    fg_keys = sorted(processed_data.keys())

    if not fg_keys:
        print("No field gradient data available")
        return

    # Pre-compute all heatmap data and find global min/max for consistent colorbar
    all_heatmaps = {}
    global_min = float('inf')
    global_max = float('-inf')

    for fg in fg_keys:
        heatmap_data, lens_values, ke_values = heatmap_energy_lens(processed_data, fg)
        if heatmap_data is not None:
            all_heatmaps[fg] = (heatmap_data, lens_values, ke_values)
            if not np.all(np.isnan(heatmap_data)):
                global_min = min(global_min, np.nanmin(heatmap_data))
                global_max = max(global_max, np.nanmax(heatmap_data))

    if not all_heatmaps:
        print("No valid heatmap data found")
        return

    # Use provided vmin/vmax or computed global values
    if vmin is None:
        vmin = global_min if global_min != float('inf') else 0
    if vmax is None:
        vmax = global_max if global_max != float('-inf') else 1

    # Create figure with slider
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)

    # Initial plot
    initial_fg = fg_keys[0]
    heatmap_data, lens_values, ke_values = all_heatmaps[initial_fg]

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(ke_values)))
    ax.set_yticks(np.arange(len(lens_values)))
    ax.set_xticklabels([f'{ke:.1f}' for ke in ke_values])
    ax.set_yticklabels([f'{lens:.3f}' for lens in lens_values])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Kinetic Energy (eV)')
    ax.set_ylabel('Lens VMI')
    ax.set_title(f'Energy Resolution (ΔE/E) - Field Gradient: {initial_fg}')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Energy Resolution (ΔE/E)')

    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Field Gradient',
        valmin=0,
        valmax=max(0.1, len(fg_keys) - 1),
        valinit=0,
        valstep=1
    )

    def update(val):
        fg_idx = int(slider.val)
        fg = fg_keys[fg_idx]
        slider.valtext.set_text(f'{fg}')

        if fg in all_heatmaps:
            heatmap_data, lens_values, ke_values = all_heatmaps[fg]

            ax.clear()
            im_new = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(ke_values)))
            ax.set_yticks(np.arange(len(lens_values)))
            ax.set_xticklabels([f'{ke:.1f}' for ke in ke_values])
            ax.set_yticklabels([f'{lens:.3f}' for lens in lens_values])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('Kinetic Energy (eV)')
            ax.set_ylabel('Lens VMI')
            ax.set_title(f'Energy Resolution (ΔE/E) - Field Gradient: {fg}')

            fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()



