import os
import glob
import random
import re
import traceback
import gc
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Any, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from functools import partial
import time  # Added for profiling


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


def save_checkpoint(data, filename, checkpoint_num):
    """
    Save intermediate results to a checkpoint file.
    
    Args:
        data: Data to save
        filename: Base filename for checkpoint
        checkpoint_num: Checkpoint number
    """
    checkpoint_file = f"{filename}_checkpoint_{checkpoint_num}.pkl"
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Checkpoint saved: {checkpoint_file}")
        return checkpoint_file
    except Exception as e:
        print(f"  Warning: Could not save checkpoint: {e}")
        return None


def load_checkpoint(filename):
    """
    Load data from a checkpoint file.
    
    Args:
        filename: Checkpoint filename
    
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not load checkpoint {filename}: {e}")
        return None


def clear_file_contents(*file_paths):
    """
    Clears the contents of the specified files by opening them in write mode with no data.

    Args:
        *file_paths: Variable number of file paths to clear.
    """
    for path in file_paths:
        try:
            with open(path, 'w') as f:
                pass  # Clearing the file by writing nothing
            print(f"Cleared contents of '{path}'")
        except Exception as e:
            print(f"Error clearing '{path}': {e}")


# Define complex type hints for better readability
Trajectory = List[Tuple[float, float, float]]
Particle = List[Trajectory]
ParticleData = Dict[int, Particle]
LensVMIData = Dict[str, ParticleData]
MainData = Dict[float, Dict[float, LensVMIData]]
AnalysisResults = Dict[float, Dict[float, Dict[int, Dict[int, Dict[int, List[float]]]]]]




def parse_fly2_file(filename: str) -> Dict[int, Tuple[int, int]]:
    """
    Parses a SIMION .fly2 file to extract emission angles for each group.

    Args:
        filename: The path to the .fly2 file.

    Returns:
        A dictionary where keys are group indices and values are tuples (az, el) for the first 8 beams.
    """
    # Initialize dictionary to store group angles
    group_angles = {}
    group_idx = 0

    # Read the entire file content into a string for regex processing
    with open(filename, 'r') as f:
        content = f.read()

    # Compile regex pattern to find all 'standard_beam' blocks in the file (using DOTALL to match multiline blocks)
    beam_pattern = re.compile(r'standard_beam\s*\{(.*?)\}', re.DOTALL)
    beams = beam_pattern.findall(content)

    # Iterate through each beam block found, limited to first 8
    for beam in beams:
        if group_idx >= 8:
            break  # Only collect first 8 beams

        # Search for azimuth (az) value within the beam block
        az_match = re.search(r'az\s*=\s*([-\d]+)', beam)
        el_match = re.search(r'el\s*=\s*arithmetic_sequence\s*\{\s*first\s*=\s*([-\d]+)', beam)
        if az_match and el_match:
            az = int(az_match.group(1))
            el = int(el_match.group(1))
            group_angles[group_idx] = (az, el)
            group_idx += 1

    # Return the dictionary of group angles
    return group_angles


def parse_out_file(filename: str, chunk_size: int = 100000) -> MainData:
    """
    Parses a SIMION 'out.txt' file to extract particle trajectories.

    OPTIMIZED: 
    - Read file line-by-line instead of loading entire file into memory
    - Process data in chunks to reduce memory usage for large files
    - Use more efficient data structures

    Args:
        filename: The path to the 'out.txt' file.
        chunk_size: Number of lines to process before forcing garbage collection (default: 100000)

    Returns:
        A nested dictionary containing the parsed data, structured as:
        data[field_gradient][lens_VMI]['local'][particle_idx] = [trajectory_1, ...].
    """
    # Compile regex pattern to match parameter lines in the format "parameters = [val1,val2,val3,...]"
    param_pattern = re.compile(r'parameters\s*=\s*\[([^\]]+)\]')

    # Initialize main data structure: nested defaultdicts for field_gradient -> lens_VMI -> ke -> 'local'/'global' -> particle_idx -> trajectories
    data: MainData = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))

    block_count = 0
    lines_processed = 0

    with open(filename, 'r') as f:
        current_block_data = []  # List of (ion_n, x, y, z) for current block
        current_fg = None
        current_lens_VMI = None
        current_ke = None

        for line_num, line in enumerate(f, 1):
            lines_processed += 1
            
            # Check if this is a block separator line indicating start of new simulation
            if "Begin Next Fly'm" in line:
                # If we have data for the current block, process it immediately
                if current_fg is not None and current_lens_VMI is not None and current_ke is not None and current_block_data:
                    process_simulation_block(data, current_fg, current_lens_VMI, current_ke, current_block_data)
                    block_count += 1
                # Reset for next block
                current_block_data = []
                current_fg = None
                current_lens_VMI = None
                current_ke = None
                continue

            # Try to match parameter line to extract field_gradient, lens_VMI, and ke
            if (param_match := param_pattern.search(line)):
                try:
                    # Extract parameter string and split into list of floats
                    params_str = param_match.group(1)
                    params = [float(p.strip()) for p in params_str.split(',')]
                    # Assign field_gradient (index 0), lens_VMI (index 2), ke (index 8)
                    if len(params) > 8:
                        current_fg = params[0]
                        current_lens_VMI = params[2]
                        current_ke = params[8]
                    else:
                        # Fallback for old format
                        current_fg = params[0]
                        current_lens_VMI = params[2]
                        current_ke = 10.0
                except (ValueError, IndexError):
                    # Warn if parsing fails, but continue
                    print(f"Warning: Could not parse parameters on line {line_num}: {line.strip()}")
                continue

            # Clean the line and skip if empty or starts with quote (comments or headers)
            line = line.strip()
            if not line or line.startswith('"'):
                continue

            # Split line by comma and check if it has 4 parts (ion_n, x, y, z)
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                try:
                    # Parse ion number and coordinates - use regular float instead of float16 to avoid precision issues
                    ion_n, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    # Add to current block data
                    current_block_data.append((ion_n, x, y, z))
                except (ValueError, IndexError):
                    pass  # Ignore malformed lines
            
            # Periodic garbage collection for very large files
            if lines_processed % chunk_size == 0:
                gc.collect()

        # Process the last block if it has data
        if current_fg is not None and current_lens_VMI is not None and current_ke is not None and current_block_data:
            process_simulation_block(data, current_fg, current_lens_VMI, current_ke, current_block_data)
            block_count += 1

    return data


def process_simulation_block(data, fg, lens_VMI, ke, block_data):
    """
    Process a single simulation block and update the data dictionary.

    Args:
        data: Main data dictionary to update.
        fg: Field gradient.
        lens_VMI: Lens VMI value.
        ke: Kinetic energy.
        block_data: List of (ion_n, x, y, z) tuples.
    """
    # Aggregate all trajectory points for each ion number in this block
    ion_trajectories = defaultdict(list)  # ion_n -> list of (x, y, z) points
    for ion_n, x, y, z in block_data:
        ion_trajectories[ion_n].append((x, y, z))

    # Skip if no trajectories
    if not ion_trajectories:
        return

    # Suppressed block processing debug output
    # print(f"Processing block: fg={fg}, lens={lens_VMI}, ke={ke}, ions={list(ion_trajectories.keys())}")  # Debug output

    # Group trajectories into particles using ion number rule
    for ion_n, traj in ion_trajectories.items():
        if not traj or ion_n < 1:
            continue

        # Rule: ion_n 1-8 -> particle_idx 0; 9-16 -> particle_idx 1, etc.
        # This groups 8 consecutive ion numbers into one particle
        p_idx = (ion_n - 1) // 8

        # Initialize the particle data structure if needed
        if p_idx not in data[fg][lens_VMI][ke]['local']:
            data[fg][lens_VMI][ke]['local'][p_idx] = {'trajectories': []}
        
        # Append the full trajectory to the particle's list
        data[fg][lens_VMI][ke]['local'][p_idx]['trajectories'].append(traj)
        # Suppressed trajectory debug output
        # print(f"  Added trajectory for ion {ion_n} -> particle {p_idx}, points={len(traj)}")  # Debug output


def compute_final_stats_for_fg(fg, fg_data):
    """
    Compute final position stats for a single field gradient.
    Returns the global data dict for this fg.
    """
    global_data_per_lens = {}
    # Suppressed detailed progress output
    # print(f"\n🧲 Field Gradient = {fg}")
    for lens_VMI in sorted(fg_data.keys()):
        # print(f"  Lens VMI = {lens_VMI}")
        particles = fg_data[lens_VMI].get('local', {})
        if not particles:
            continue

        # print(f"  🔹 Detected {len(particles)} distinct particles")
        all_final_pos_y, all_final_pos_z = [], []

        for p_idx in sorted(particles.keys()):
            p_data = particles[p_idx]
            if isinstance(p_data, dict):
                trajs = p_data.get('trajectories', [])
            else:
                trajs = p_data
            num_trajs = len(trajs)
            total_points = sum(len(t) for t in trajs)
            avg_length = total_points / num_trajs if num_trajs > 0 else 0
            # Suppressed particle trajectory output
            # print(f"    Particle {p_idx}: {num_trajs} trajectories, ~{avg_length:.1f} pts each")

            if num_trajs == 0:
                continue

            # Extract final positions (y, z) from each trajectory - vectorized
            final_positions = np.array([traj[-1][1:3] for traj in trajs])  # shape (n, 2) where n=number of trajectories

            # For 8 trajectories, no grouping, append all final positions without centering
            if final_positions.shape[0] == 8:
                all_final_pos_y.extend(final_positions[:, 0])
                all_final_pos_z.extend(final_positions[:, 1])
            else:
                # For particles with !=8 trajectories, add positions without centering
                all_final_pos_y.extend(final_positions[:, 0])
                all_final_pos_z.extend(final_positions[:, 1])

        # Calculate standard deviation across all particles for this field gradient - vectorized
        std_dev_y = np.std(all_final_pos_y) if all_final_pos_y else 0.0
        std_dev_z = np.std(all_final_pos_z) if all_final_pos_z else 0.0
        counts_y, bins_y = np.histogram(all_final_pos_y, bins='auto')
        counts_z, bins_z = np.histogram(all_final_pos_z, bins='auto')
        global_data_per_lens[lens_VMI] = {
            'counts_y': counts_y,
            'bins_y': bins_y,
            'counts_z': counts_z,
            'bins_z': bins_z,
            'std_dev_y': std_dev_y,
            'std_dev_z': std_dev_z
        }
    return fg, global_data_per_lens

def calculate_detector_stats(data: MainData, x_range=(73.0, 166.0)):
    """
    Calculates and prints the standard deviation of the y/z positions at detector plane
    for each particle and stores it in the 'global' key of the data dict.
    Also calculates average min distance between particles in same group.

    Optimization: Use multi-threading to parallelize computation across independent field gradients.
    Why: Each fg's stats calculation is independent, so we can process multiple fgs concurrently on different CPU cores.
    This reduces total execution time for datasets with many field gradients.
    """
    # Define grouping for emission angles
    grouping = [
        ([0, 1], (0, 0)),    # (0,0) and (0,180)
        ([2, 4], (90, 45)),  # (90,45) and (90,90)
        ([6], (90, 90)),     # (90,135)
        ([3, 5], (-90, 45)), # (-90,45) and (-90,90)
        ([7], (-90, 90))     # (-90,135)
    ]

    # Optimization: Parallelize over field gradients since each fg is independent
    def compute_detector_stats_for_fg(fg, fg_data):
        # Suppressed detailed progress output
        # print(f"\n🧲 Field Gradient = {fg}")
        for lens_VMI in sorted(fg_data.keys()):
            # print(f"  Lens VMI = {lens_VMI}")
            for ke in sorted(fg_data[lens_VMI].keys()):
                # print(f"    KE = {ke}")
                particles = fg_data[lens_VMI][ke].get('local', {})
                if not particles:
                    continue

                # print(f"  🔹 Detected {len(particles)} distinct particles")
                all_det_pos_y, all_det_pos_z = [], []

                for p_idx in sorted(particles.keys()):
                    p_data = particles[p_idx]
                    if isinstance(p_data, dict):
                        trajs = p_data.get('trajectories', [])
                    else:
                        trajs = p_data
                    num_trajs = len(trajs)
                    total_points = sum(len(t) for t in trajs)
                    avg_length = total_points / num_trajs if num_trajs > 0 else 0
                    # Suppressed particle trajectory output
                    # print(f"    Particle {p_idx}: {num_trajs} trajectories, ~{avg_length:.1f} pts each")

                    if num_trajs == 0:
                        continue

                    # Collect positions at detector (x in x_range)
                    for traj in trajs:
                        traj_array = np.array(traj) if not isinstance(traj, np.ndarray) else traj
                        mask = (traj_array[:, 0] >= x_range[0]) & (traj_array[:, 0] <= x_range[1])
                        if np.any(mask):
                            all_det_pos_y.extend(traj_array[mask, 1])  # y column
                            all_det_pos_z.extend(traj_array[mask, 2])  # z column

                # Calculate standard deviation across all detector positions
                std_dev_y = np.std(all_det_pos_y) if all_det_pos_y else 0.0
                std_dev_z = np.std(all_det_pos_z) if all_det_pos_z else 0.0
                counts_y, bins_y = np.histogram(all_det_pos_y, bins='auto')
                counts_z, bins_z = np.histogram(all_det_pos_z, bins='auto')
                global_data = {
                    'counts_y': counts_y,
                    'bins_y': bins_y,
                    'counts_z': counts_z,
                    'bins_z': bins_z,
                    'std_dev_y': std_dev_y,
                    'std_dev_z': std_dev_z
                }

                # Calculate min distances between groups
                group_distances_y = []
                group_distances_z = []
                group_Y_positions_per_group = {i: [] for i in range(len(grouping))}
                group_Z_positions_per_group = {i: [] for i in range(len(grouping))}

                if particles:  # Assume for first particle with groups
                    for p_idx in sorted(particles.keys()):
                        p_data = particles[p_idx]
                        trajectories = p_data.get('trajectories', []) if isinstance(p_data, dict) else p_data
                        if len(trajectories) != 8:
                            continue
                        for g_idx, (traj_indices, angle_key) in enumerate(grouping):
                            for idx in traj_indices:
                                if idx < len(trajectories) and trajectories[idx]:
                                    final_point = trajectories[idx][-1]
                                    group_Y_positions_per_group[g_idx].append(final_point[1])
                                    group_Z_positions_per_group[g_idx].append(final_point[2])

                    group_distances_y = []
                    group_distances_z = []

                for g_idx in range(len(grouping)):
                    Ys = np.array(group_Y_positions_per_group[g_idx])
                    Zs = np.array(group_Z_positions_per_group[g_idx])
                    if len(Ys) > 1:
                        range_y = np.max(Ys) - np.min(Ys)
                    else:
                        range_y = 0
                    if len(Zs) > 1:
                        range_z = np.max(Zs) - np.min(Zs)
                    else:
                        range_z = 0
                    group_distances_y.append({'range': range_y, 'std': 0})
                    group_distances_z.append({'range': range_z, 'std': 0})
                # Overall average (using group ranges as "distances")
                abs_distances_y = [g['range'] for g in group_distances_y if g['range'] != 0]
                abs_distances_z = [g['range'] for g in group_distances_z if g['range'] != 0]
                global_data['avg_min_dist_y'] = np.mean(abs_distances_y) if abs_distances_y else 0
                global_data['std_min_dist_y'] = np.std(abs_distances_y) if abs_distances_y else 0
                global_data['avg_min_dist_z'] = np.mean(abs_distances_z) if abs_distances_z else 0
                global_data['std_min_dist_z'] = np.std(abs_distances_z) if abs_distances_z else 0
                global_data['group_distances_y'] = group_distances_y
                global_data['group_distances_z'] = group_distances_z

                data[fg][lens_VMI][ke]['global'] = global_data
        return fg

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_detector_stats_for_fg, fg, data[fg]): fg for fg in sorted(data.keys())}
        for future in as_completed(futures):
            fg = future.result()
    return data


def analyze_beam_for_fg(fg, fg_data, x_planes, group_angles, threshold_factor=0.125):
    """
    Analyze beam properties for a single field gradient.
    Returns the results dict for this fg.
    """
    results_fg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Define grouping for focus analysis based on emission angles
    grouping = [
        ([0, 1], (0, 0)),    # (0,0) and (0,180)
        ([2, 6], (90, 45)),  # (90,45) and (90,135)
        ([4, 5], (90, 90)),  # (90,90) and (-90,90)
        ([3, 7], (-90, 45))  # (-90,45) and (-90,135)
    ]

    for lens_VMI, lens_data in fg_data.items():
        for p_idx, p_data in lens_data.get('local', {}).items():
            trajectories = p_data.get('trajectories', []) if isinstance(p_data, dict) else p_data
            if not trajectories:
                continue
            num_trajs = len(trajectories)
            if num_trajs != 8:
                # Suppressed warning message
                # print(f"Warning: particle {p_idx} has {num_trajs} trajectories, expected 8")
                continue

            for traj_indices, key in grouping:
                group_trajs = [trajectories[idx] for idx in traj_indices]
                all_points = np.concatenate([np.array(t) for t in group_trajs])  # Vectorized concatenation

                # Use the defined key for the group
                # If group_angles available, use it, but here we set key directly

                # Efficiently find points near each x_plane - vectorized masking
                for x_plane in x_planes:
                    threshold = threshold_factor
                    mask = np.abs(all_points[:, 0] - x_plane) <= threshold

                    if np.any(mask):
                        y_array = all_points[mask, 1]
                        z_array = all_points[mask, 2]

                        mean_y = np.mean(y_array)
                        mean_z = np.mean(z_array)
                        ptp_y = np.ptp(y_array)  # Peak-to-peak (max - min)
                        ptp_z = np.ptp(z_array)

                        results_fg[lens_VMI][p_idx][x_plane][key] = [mean_y, mean_z, ptp_y, ptp_z]
    return fg, results_fg

def analyze_beam_across_x_planes(data: MainData, x_planes: Iterable[float], group_angles: Dict[int, Tuple[int, int]] = None) -> AnalysisResults:
    """
    Analyzes beam properties (range, mean) at different x-planes.
    Divides trajectories into 4 groups of 2 trajectories each, each group corresponding to different emission angles.

    Optimization: Use multi-threading to parallelize analysis across independent field gradients.
    Why: Each fg's beam analysis is independent, so we can process multiple fgs concurrently on different CPU cores.
    This reduces total execution time for datasets with many field gradients.
    """
    results: AnalysisResults = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))

    # Use a small threshold relative to the plane distance for finding points
    threshold_factor = 0.125  # +- 0.125mm of x_plane

    # Optimization: Parallelize over field gradients since each fg is independent
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(analyze_beam_for_fg, fg, data[fg], x_planes, group_angles, threshold_factor): fg for fg in data}
        for future in as_completed(futures):
            fg, results_fg = future.result()
            for lens_VMI, lens_results in results_fg.items():
                for p_idx, x_plane_data in lens_results.items():
                    for x_plane, key_data in x_plane_data.items():
                        for key, stats in key_data.items():
                            results[fg][lens_VMI][p_idx][x_plane][key] = stats
    return results

def find_focus_for_fg(fg, fg_results, group_angles, focus_axis):
    """
    Find focus for a single field gradient.
    Returns the min_range_results for this fg.
    """
    axis_index = 2 if focus_axis == 'y' else 3

    min_range_results_fg = defaultdict(lambda: defaultdict(list))  # lens_VMI -> key -> list of (ptp, x_plane, mean_y, mean_z, p_idx)

    # Define grouping for focus analysis based on emission angles
    grouping = [
        ([0, 1], (0, 0)),    # (0,0) and (0,180)
        ([2, 6], (90, 45)),  # (90,45) and (90,135)
        ([4, 5], (90, 90)),  # (90,90) and (-90,90)
        ([3, 7], (-90, 45))  # (-90,45) and (-90,135)
    ]

    for lens_VMI, lens_results in fg_results.items():
        for p_idx, x_plane_data in lens_results.items():
            for traj_indices, key in grouping:
                # Since analyze_beam_for_fg used the same key, here we can directly use the key
                valid_planes = {p: s[key] for p, s in x_plane_data.items() if key in s and not np.isnan(s[key][axis_index])}

                # Find the x-plane with the minimum ptp for the chosen axis
                best_plane = min(
                    valid_planes.items(),
                    key=lambda item: item[1][axis_index],  # Use the selected axis_index
                    default=(None, None)
                )

                if best_plane[0] is not None:
                    x_plane, stats = best_plane
                    # stats = [mean_y, mean_z, ptp_y, ptp_z]
                    min_range_results_fg[lens_VMI][key].append((stats[axis_index], x_plane, stats[0], stats[1], p_idx))

    return fg, min_range_results_fg

def find_and_store_focus(data: MainData, results: AnalysisResults, group_angles: Dict[int, Tuple[int, int]] = None):
    """
    Finds the focus for each unique (az, el) combination for both Y and Z axes and stores all focus points in the data dictionary.

    The focus is the x-plane with the minimum beam extent (ptp) along the specified axis for each (az, el) group per particle.

    Optimization: Use multi-threading to parallelize focus finding across independent field gradients.
    Why: Each fg's focus computation is independent, so we can process multiple fgs concurrently on different CPU cores.
    This reduces total execution time for datasets with many field gradients.

    Args:
        data: The main data dictionary to store results in.
        results: The analysis results from `analyze_beam_across_x_planes`.
        group_angles: Dict mapping g_idx to (az, el) if using angle keys.
    """
    for focus_axis in ['y', 'z']:
        if focus_axis == 'y':
            print("\nFinding focus based on minimum beam extent in Y...")
        elif focus_axis == 'z':
            print("\nFinding focus based on minimum beam extent in Z...")

        # Optimization: Parallelize over field gradients since each fg is independent
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(find_focus_for_fg, fg, results[fg], group_angles, focus_axis): fg for fg in results}
            min_range_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # fg -> lens_VMI -> key -> list
            for future in as_completed(futures):
                fg, min_range_results_fg = future.result()
                for lens_VMI, lens_min_data in min_range_results_fg.items():
                    for key, group_data in lens_min_data.items():
                        min_range_results[fg][lens_VMI][key].extend(group_data)

        # Store the focus positions back into the main data structure
        for fg, fg_min_data in min_range_results.items():
            for lens_VMI, lens_min_data in fg_min_data.items():
                if 'global' not in data[fg][lens_VMI]:
                    data[fg][lens_VMI]['global'] = {}
                focus_points = {key: [(x, y, z) for _, x, y, z, _ in sorted(group_data, key=lambda x: x[4])] for key, group_data in lens_min_data.items()}
                data[fg][lens_VMI]['global'][f'focus_points_{focus_axis}'] = focus_points


def process_data(x_range: Tuple[float, float], file_path: str, focus_axis: str, fly2_file: str = None, y_range: Tuple[float, float] = (-0.5, 0.5), z_range: Tuple[float, float] = (-0.5, 0.5)):
    """
    Processes SIMION data to find the focus point.

    Args:
        x_range: A tuple containing the start and end of the x-axis scan range (in mm).
        x_step: The step size for the scan.
        file_path: The path to the data file (e.g., 'out.txt').
        focus_axis: The axis to evaluate for focus ('y' or 'z').
        fly2_file: Optional path to .fly2 file to read emission angles.

    Returns:
        The processed data as a nested dictionary.
    """
    try:
        x_start, x_stop = x_range

        # --- Step 1: Parse the simulation output file ---
        data = parse_out_file(file_path)

        # Trajectory data is now converted to half-precision in parse_out_file.

        # --- Step 2: Calculate and print stats about detector positions ---
        data = calculate_detector_stats(data, x_range=x_range)
        # Suppressed detector stats completion message
        # print("\nFinished calculating detector position stats.")

        extract_aligned_points_for_all_pairs(data,x_range = x_range)

        # --- Compute dr, M_square, M_rectangle ---
        for fg in data:
            for lvmi in data[fg]:
                for ke in data[fg][lvmi]:
                    global_data = data[fg][lvmi][ke]['global']
                    focus_points_y = global_data.get('focus_points_y', [])
                    if focus_points_y:
                        widths_y = [item[3] for item in focus_points_y]
                        avg_focus_width_y = np.mean(widths_y) if widths_y else 0
                    else:
                        avg_focus_width_y = 0
                    focus_points_z = global_data.get('focus_points_z', [])
                    if focus_points_z:
                        widths_z = [item[3] for item in focus_points_z]
                        avg_focus_width_z = np.mean(widths_z) if widths_z else 0
                    else:
                        avg_focus_width_z = 0
                    dr = np.sqrt(avg_focus_width_y**2 + avg_focus_width_z**2)
                    ptp_y = np.ptp(y_range)
                    ptp_z = np.ptp(z_range)
                    M_square = dr**2 / (ptp_y * ptp_z) if ptp_y > 0 and ptp_z > 0 else 0
                    M_rectangle = (avg_focus_width_y * avg_focus_width_z) / (ptp_y * ptp_z) if ptp_y > 0 and ptp_z > 0 else 0
                    global_data['dr'] = dr
                    global_data['M_square'] = M_square
                    global_data['M_rectangle'] = M_rectangle

        # --- Step 3: Get group angles if fly2_file provided ---
        group_angles = None
        if fly2_file:
            group_angles = parse_fly2_file(fly2_file)



        return data

    except FileNotFoundError:
        print(f"\nError: File '{file_path}' not found. Make sure the path is correct.")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        return None


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

def visualize_focus_xyz(processed_data, fly2_file=None):
    """
    Visualizes the focus data in 3D: azimuth vs elevation vs focus x-plane.

    :param processed_data: A dictionary where keys are field gradients and values contain the data.
    :param fly2_file: Optional path to .fly2 file to read actual emission angles.
    """
    fg_keys = sorted(processed_data.keys())
    if not fg_keys:
        print("No data to visualize.")
        return

    # Get emission angles
    if fly2_file:
        group_angles = parse_fly2_file(fly2_file)
    else:
        # Default angles based on focus groups
        # (0,0), (90,45), (90,90), (-90,45)
        group_angles = {(0, 0): (0, 0), (90, 45): (90, 45), (90, 90): (90, 90), (-90, 45): (-90, 45)}

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.1, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Field Gradient',
        valmin=0,
        valmax=len(fg_keys) - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        ax.clear()
        fg_idx = int(slider.val)
        fg = fg_keys[fg_idx]
        lens_VMI = list(processed_data[fg].keys())[0]
        fg_data = processed_data[fg][lens_VMI]

        focus_points = fg_data.get('global', {}).get('focus_points', {})

        if focus_points:
            azs = []
            els = []
            focus_xs = []
            for key, points_list in focus_points.items():
                az, el = key
                for x, y, z in points_list:
                    azs.append(az)
                    els.append(el)
                    focus_xs.append(x)
            if azs:
                ax.scatter(azs, els, focus_xs, marker='o', c='blue', s=50)
                # Optionally add labels
                for i, (az, el, fx) in enumerate(zip(azs, els, focus_xs)):
                    ax.text(az, el, fx, f'({az}°, {el}°)', fontsize=8)

        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Elevation (°)')
        ax.set_zlabel('Focus X-plane')
        ax.set_title(f'Focus X-plane vs Emission Angles\nField Gradient: {fg}')
        slider.valtext.set_text(f'{fg_keys[fg_idx]}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    update(0)
    plt.show()


import math

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

def generate_particles_fly2(num_groups=4, filename='particles_tao.fly2', x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_range=(-0.5, 0.5), ke=15, theta=0):
    """
    Generates a .fly2 file with particle definitions for SIMION.

    This function creates a specified number of particle groups. Each group consists of 8 particles
    generated from 8 separate `standard_beam` definitions. A key feature is that all particles
    within the same group share the same randomized position jitter (dx, dy, dz), simulating a localized event.

    For each position, generates 8 trajectories with specific azimuth and elevation (after theta rotation):
    - az=0, el=0
    - az=0, el=180
    - az=90, el=45
    - az=-90, el=45
    - az=90, el=90
    - az=-90, el=90
    - az=90, el=135
    - az=-90, el=135
    All direction vectors are rotated by theta around the z-axis.

    Summary of generated particles:
    - Number of groups to generate: `num_groups`
    - Particles per group: 8
    - Total particles generated: `num_groups` * 8

    Args:
        num_groups (int): The number of particle groups to generate. If the value is less than 1,
                        no particles will be generated. Defaults to 4.
        filename (str): The name for the output .fly2 file.
                        Defaults to 'particles_tao.fly2'.
        x_range (tuple): A tuple (min, max) specifying the range for random variation in x position.
                         Defaults to (-0.5, 0.5).
        y_range (tuple): A tuple (min, max) specifying the range for random variation in y position.
                         Defaults to (-0.5, 0.5).
        z_range (tuple): A tuple (min, max) specifying the range for random variation in z position.
                         Defaults to (-0.5, 0.5).
        ke (float): The kinetic energy of the electron particles in electron volts (eV).
                    Defaults to 15.
        theta (float): Rotation angle (in radians) around the z-axis for all direction vectors.
                      Defaults to 0.
    """

    # If num_groups is not a positive integer, set to 0 and warn the user.
    if not isinstance(num_groups, int) or num_groups < 0:
        print(f"⚠️ Invalid input for `num_groups` (value: {num_groups}). It must be a non-negative integer. Setting to 0.")
        num_groups = 0

    if num_groups == 0:
        print("`num_groups` is 0, an empty particle file will be created.")

    # Direction vectors for the 8 trajectories
    original_directions = [
        (1, 0, 0), (-1, 0, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 1, 0), (1, -1, 0), (0, 1, 0), (0, -1, 0)
    ]

    # Apply theta rotation around z-axis
    directions = [rotate_around_x(d, theta) for d in original_directions]

    # Color codes based on emission angle groups from extract_aligned_points_for_all_pairs
    # Group 1: trajectories 0,1 - color 1
    # Group 2: trajectories 2,4 - color 2
    # Group 3: trajectory 6 - color 3
    # Group 4: trajectories 3,5 - color 4
    # Group 5: trajectory 7 - color 5
    colors = [1, 1, 2, 4, 2, 4, 3, 5]

    if not filename or not filename.strip():
        filename = 'particles_tao.fly2'

    try:
        with open(filename, 'w') as fid:
            fid.write('particles {\n')

            # Only write coordinates line if there are particles
            if num_groups > 0:
                fid.write('  coordinates = 0,\n')

                for i in range(num_groups):
                    # Shared jitter (dx, dy, dz) for this group
                    dx = random.uniform(x_range[0], x_range[1])
                    dy = random.uniform(y_range[0], y_range[1])
                    dz = random.uniform(z_range[0], z_range[1])

                    # Write 8 standard_beam entries per group, each is a single particle
                    for j in range(8):
                        direction = directions[j]
                        color_for_trajectory = colors[j]  # Color based on emission angle group

                        # The last beam definition in the file must not have a trailing comma.
                        is_last_beam = (i == num_groups - 1 and j == 7)
                        need_comma = not is_last_beam

                        write_standard_beam(fid, 199 + dx, -1 + dy, dz, direction, need_comma, ke, color_for_trajectory)

            fid.write('}\n')

        total_particles = num_groups * 8
        print(f"✅ Successfully generated {num_groups} groups ({total_particles} particles total).")
        print(f"   File saved to: \"{filename}\"")

    except Exception as e:
        raise IOError(f"Failed to create file {filename}: {e}")


def write_standard_beam(fid, x, y, z, direction, need_comma=True, ke=15, color=0):
    """
    Write a single standard_beam block to the file.

    Args:
        fid (file object): Open file handle for writing
        x (float): X position
        y (float): Y position
        z (float): Z position
        direction (tuple): Direction vector (dir_x, dir_y, dir_z)
        need_comma (bool): Whether to add a comma after the block
        ke (float): Kinetic energy
        color (int): Color code for the particle group
    """
    fid.write('  standard_beam {\n')
    fid.write('    n = 1,\n')
    fid.write('    tob = 0,\n')
    fid.write('    mass = 0.000548579903,\n')
    fid.write('    charge = -1,\n')
    fid.write(f'    ke = {ke},\n')
    fid.write('    cwf = 1,\n')
    fid.write(f'    color = {color},\n')
    dir_x, dir_y, dir_z = direction
    fid.write(f'    direction = vector({dir_x}, {dir_y}, {dir_z}),\n')
    fid.write(f'    position = vector({x:.6f}, {y:.6f}, {z:.6f})')
    fid.write('  }')

    # Add comma and newline if needed; otherwise just newline
    fid.write(',\n' if need_comma else '\n')


def energy_resolution_utilis(filename='energy_resolution_particle.fly2', position=(0.0, 0.0, 0.0), num_particles=100, ke=15):
    """
    Generates a .fly2 file with particles at a fixed position with TRUE isotropic (4π) distribution.
    
    OPTIMIZED: Uses SIMION's cone_direction_distribution with fill=true and half_angle=180
    for true isotropic distribution over the full sphere. This creates a compact file
    regardless of particle count.
    
    Args:
        filename (str): The name for the output .fly2 file.
        position (tuple): A tuple (x, y, z) specifying the fixed position of the particle source.
        num_particles (int): The number of particles to generate.
        ke (float): The kinetic energy of the electron particles in electron volts (eV).
    """
    if not isinstance(num_particles, int) or num_particles <= 0:
        print(f"⚠️ Invalid input for `num_particles` (value: {num_particles}). It must be a positive integer. Setting to 100.")
        num_particles = 100

    if not filename or not filename.strip():
        filename = 'energy_resolution_particle.fly2'

    x, y, z = position
    
    # Calculate speed from kinetic energy for electrons
    # For electrons in SIMION: v (mm/μs) = 593.096 * sqrt(KE in eV)
    speed = 593.096 * np.sqrt(ke)

    try:
        with open(filename, 'w') as fid:
            # Write compact fly2 format using cone_direction_distribution for true isotropy
            fid.write('particles {\n')
            fid.write('  coordinates = 0,\n')
            fid.write('  standard_beam {\n')
            fid.write(f'    n = {num_particles},\n')
            fid.write('    tob = 0,\n')
            fid.write('    mass = 0.000548579903,\n')  # Electron mass in amu
            fid.write('    charge = -1,\n')
            fid.write('    cwf = 1,\n')
            fid.write('    color = 0,\n')
            # cone_direction_distribution with half_angle=180 and fill=true gives true 4π isotropy
            fid.write('    direction = cone_direction_distribution {\n')
            fid.write('      axis = vector(1, 0, 0),\n')  # Reference axis (arbitrary for full sphere)
            fid.write('      half_angle = 180,\n')  # Full sphere coverage
            fid.write('      fill = true\n')  # Uniform solid angle distribution
            fid.write('    },\n')
            fid.write(f'    position = vector({x:.6f}, {y:.6f}, {z:.6f}),\n')
            # Fixed speed corresponding to kinetic energy
            fid.write(f'    speed = {speed:.6f},\n')
            fid.write('    format = ""\n')
            fid.write('  }\n')
            fid.write('}\n')

    except Exception as e:
        raise IOError(f"Failed to create file {filename}: {e}")


def energy_resolution_utilis_legacy(filename='energy_resolution_particle.fly2', position=(0.0, 0.0, 0.0), num_particles=100, ke=15):
    """
    LEGACY VERSION: Generates a .fly2 file by writing each particle individually.
    
    This is the original implementation that creates large files for many particles.
    Use energy_resolution_utilis() instead for better performance.
    
    Args:
        filename (str): The name for the output .fly2 file.
        position (tuple): A tuple (x, y, z) specifying the fixed position of the particle source.
        num_particles (int): The number of particles to generate uniformly over 4π solid angle.
        ke (float): The kinetic energy of the electron particles in electron volts (eV).
    """
    if not isinstance(num_particles, int) or num_particles <= 0:
        print(f"⚠️ Invalid input for `num_particles` (value: {num_particles}). It must be a positive integer. Setting to 100.")
        num_particles = 100

    if not filename or not filename.strip():
        filename = 'energy_resolution_particle.fly2'

    x, y, z = position

    try:
        with open(filename, 'w') as fid:
            fid.write('particles {\n')
            fid.write('  coordinates = 0,\n')

            # Generate uniform distribution over sphere for 4π solid angle
            cos_theta = np.random.uniform(-1, 1, num_particles)
            theta = np.arccos(cos_theta)
            phi = np.random.uniform(0, 2*np.pi, num_particles)

            dir_x = np.cos(theta)
            dir_y = np.sin(theta) * np.cos(phi)
            dir_z = np.sin(theta) * np.sin(phi)

            for i in range(num_particles):
                direction = (dir_x[i], dir_y[i], dir_z[i])
                color = 1
                need_comma = (i < num_particles - 1)
                write_standard_beam(fid, x, y, z, direction, need_comma, ke, color)

            fid.write('}\n')

    except Exception as e:
        raise IOError(f"Failed to create file {filename}: {e}")

def add_parameters_to_out_file(parameters,out_file_path="out.txt"):
    """
    Reads 'out.txt', finds the last occurrence of "------ Begin Next Fly'm ------",
    and inserts a parameters line after it.
    """
    try:
        with open(out_file_path, 'r') as f:
            lines = f.readlines()

        separator = "------ Begin Next Fly'm ------"
        # Find the last occurrence of the separator from the end of the list
        last_occurrence_index = -1
        for i in range(len(lines) - 1, -1, -1):
            if separator in lines[i]:
                last_occurrence_index = i
                break

        if last_occurrence_index != -1:
            # Insert the parameters line after the separator
            param_line = f"{parameters}\n"
            lines.insert(last_occurrence_index + 1, param_line)

            # Write the modified content back to the file
            with open(out_file_path, 'w') as f:
                f.writelines(lines)
            print(f"Parameters added to {out_file_path}")
        else:
            print(f"Separator '{separator}' not found in {out_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {out_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


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



def plot_fg_vs_std(processed_data, focus_criterion):
    """
    Calculates the standard deviation for each field gradient (fg) and plots
    fg vs. standard deviation.

    Args:
        processed_data (dict): A dictionary containing the processed data,
                               where keys are field gradients (fg) and values
                               are dictionaries with simulation results.
        focus_criterion (str): The axis ('y' or 'z') to use for determining
                               the focus quality.
    """
    if focus_criterion == 'y':
        std_dev_key = 'std_dev_y'
        ylabel = 'Standard Deviation (Y)'
    elif focus_criterion == 'z':
        std_dev_key = 'std_dev_z'
        ylabel = 'Standard Deviation (Z)'
    else:
        raise ValueError("focus_criterion must be either 'y' or 'z'")

    fgs = []
    stds = []

    # Sort fgs for a clean plot
    sorted_fgs = sorted(processed_data.keys())

    for fg in sorted_fgs:
        lens_VMI = list(processed_data[fg].keys())[0]
        data_fg = processed_data[fg][lens_VMI]
        if 'global' in data_fg and std_dev_key in data_fg['global']:
            fgs.append(fg)
            stds.append(data_fg['global'][std_dev_key])

    if not fgs:
        print("No data available to plot.")
        return

    # Find the lowest std and corresponding fg
    if stds:
        min_std = min(stds)
        min_std_idx = stds.index(min_std)
        min_std_fg = fgs[min_std_idx]
        text_str = f'Lowest std: {min_std:.4f} at fg {min_std_fg}'
    else:
        text_str = 'No std data'

    plt.figure(figsize=(10, 6))
    plt.plot(fgs, stds, marker='o', linestyle='-')
    plt.xlabel('Field Gradient (fg)')
    plt.ylabel(ylabel)
    plt.title(f'Field Gradient vs. {ylabel}')
    plt.grid(True)

    # Add text showing the lowest std
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()

def calculate_data_collection_efficiency(data: MainData, num_groups: int) -> Dict[float, Dict[float, float]]:
    """
    Calculates the efficiency of data collection for each field gradient and lens VMI combination.

    Efficiency = (num_trajectory_received / num_trajectory_generated) * 100%
    num_trajectory_generated = num_particles_generated * 8
    num_trajectory_received = num_particles_received * num_trajectory_per_particles

    Args:
        data: The main data dictionary containing processed simulation data.
        num_groups: The number of particle groups generated (NUM_GROUPS from workflow).

    Returns:
        A dictionary with field_gradient -> lens_VMI -> efficiency value (in percent).
    """
    efficiencies = {}
    for fg in data:
        efficiencies[fg] = {}
        for lens_VMI in data[fg]:
            local_data = data[fg][lens_VMI].get('local', {})
            num_particles_received = len(local_data)
            if local_data:
                first = list(local_data.values())[0]
                if isinstance(first, dict):
                    num_trajectory_per_particles = len(first.get('trajectories', []))
                else:
                    num_trajectory_per_particles = len(first)
            else:
                num_trajectory_per_particles = 0
            num_trajectory_received = num_particles_received * num_trajectory_per_particles
            # num_particles_generated = num_groups
            num_particles_generated = num_groups
            num_trajectory_generated = num_particles_generated * 8
            efficiency = (num_trajectory_received / num_trajectory_generated) * 100 if num_trajectory_generated > 0 else 0
            efficiencies[fg][lens_VMI] = efficiency
    return efficiencies

def compute_for_fg_lens(fg, lens_VMI, lens_data, electron_energy=5):
    """
    Compute preprocessed data for a single fg, lens combination for faster plotting.
    """
    global_data = lens_data['global']

    counts_y = global_data['counts_y']
    bins_y = global_data['bins_y']
    if len(bins_y) > 1:
        bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
        histogram_widths_y = np.diff(bins_y)
    else:
        bin_centers_y = np.array([])
        histogram_widths_y = np.array([])

    counts_z = global_data['counts_z']
    bins_z = global_data['bins_z']
    if len(bins_z) > 1:
        bin_centers_z = (bins_z[:-1] + bins_z[1:]) / 2
        histogram_widths_z = np.diff(bins_z)
    else:
        bin_centers_z = np.array([])
        histogram_widths_z = np.array([])

    peak_pct_y = (np.max(counts_y) / np.sum(counts_y)) * 100 if len(counts_y) > 0 else 0
    peak_pct_z = (np.max(counts_z) / np.sum(counts_z)) * 100 if len(counts_z) > 0 else 0

    # Peak positions
    if len(counts_y) > 0:
        i_y = np.argmax(counts_y)
        peak_y_pos = (bins_y[i_y] + bins_y[i_y+1]) / 2
    else:
        peak_y_pos = 0
    if len(counts_z) > 0:
        i_z = np.argmax(counts_z)
        peak_z_pos = (bins_z[i_z] + bins_z[i_z+1]) / 2
    else:
        peak_z_pos = 0

    # Focus points from global data
    focus_points_y = global_data.get('focus_points_y', [])
    xs_y = np.array([item[0] for item in focus_points_y])  # x
    zs = np.array([item[2] for item in focus_points_y])    # z
    widths_y_per_point = np.array([item[3] for item in focus_points_y])  # widths for y focus (but used for errorbars in XZ)

    focus_points_z = global_data.get('focus_points_z', [])
    xs_z = np.array([item[0] for item in focus_points_z])  # x
    ys = np.array([item[1] for item in focus_points_z])    # y
    widths_z_per_point = np.array([item[3] for item in focus_points_z])  # widths for z focus

    # For compatibility, set other arrays and add the lists
    labels_y = [f'Group {i}' for i in range(len(xs_y))] if len(xs_y) > 0 else []
    labels_z = [f'Group {i}' for i in range(len(xs_z))] if len(xs_z) > 0 else []
    focus_widths_y = np.array([])  # not used anymore
    focus_widths_z = np.array([])  # not used anymore

    # Petzval fit - use local points if available
    fit_text_xz = ''
    if len(xs_y) > 1:
        z2s = zs ** 2
        coeffs_xz = np.polyfit(z2s, xs_y, 1)
        slope_xz, intercept_xz = coeffs_xz
        predicted = slope_xz * z2s + intercept_xz
        ss_res = np.sum((xs_y - predicted)**2)
        ss_tot = np.sum((xs_y - np.mean(xs_y))**2)
        r_squared_xz = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        fit_text_xz = f'Petzval XZ: X = {intercept_xz:.3f} + {slope_xz:.3f} * z², R²={r_squared_xz:.3f}'

    fit_text_xy = ''
    if len(xs_z) > 1:
        y2s = ys ** 2
        coeffs_xy = np.polyfit(y2s, xs_z, 1)
        slope_xy, intercept_xy = coeffs_xy
        predicted = slope_xy * y2s + intercept_xy
        ss_res = np.sum((xs_z - predicted)**2)
        ss_tot = np.sum((xs_z - np.mean(xs_z))**2)
        r_squared_xy = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        fit_text_xy = f'Petzval XY: X = {intercept_xy:.3f} + {slope_xy:.3f} * y², R²={r_squared_xy:.3f}'

    return {
        'bin_centers_y': bin_centers_y,
        'widths_y': histogram_widths_y,
        'counts_y': counts_y,
        'bin_centers_z': bin_centers_z,
        'widths_z': histogram_widths_z,
        'counts_z': counts_z,
        'peak_pct_y': peak_pct_y,
        'peak_pct_z': peak_pct_z,
        'peak_y_pos': peak_y_pos,
        'peak_z_pos': peak_z_pos,
        'xs_y': xs_y,
        'zs': zs,
        'labels_y': labels_y,
        'xs_z': xs_z,
        'ys': ys,
        'labels_z': labels_z,
        'widths_y_per_point': focus_widths_y,
        'widths_z_per_point': focus_widths_z,
        'focus_points_y': global_data.get('focus_points_y', []),
        'focus_points_z': global_data.get('focus_points_z', []),
        'fit_text_xz': fit_text_xz,
        'fit_text_xy': fit_text_xy,
        'dr': global_data.get('dr', float('inf')),
        'M_square': global_data.get('M_square', float('inf')),
        'M_rectangle': global_data.get('M_rectangle', float('inf'))
    }


def compute_metrics_for_fg_lens_ke(fg, lens_VMI, global_data, r2_threshold, param_fg, param_lens, param_ke):
    """
    Compute metrics for a single fg, lens, ke combination for global statistics.
    """
    metrics = {'fg': param_fg, 'lens': param_lens, 'ke': param_ke}

    if 'counts_y' in global_data and len(global_data['counts_y']) > 0:
        peak_y = np.max(global_data['counts_y'])
        metrics['peak_y'] = peak_y
        metrics['std_y'] = global_data.get('std_dev_y', float('inf'))
    else:
        metrics['peak_y'] = 0
        metrics['std_y'] = float('inf')

    if 'counts_z' in global_data and len(global_data['counts_z']) > 0:
        peak_z = np.max(global_data['counts_z'])
        metrics['peak_z'] = peak_z
        metrics['std_z'] = global_data.get('std_dev_z', float('inf'))
    else:
        metrics['peak_z'] = 0
        metrics['std_z'] = float('inf')

    # Focus stds
    focus_points_y = global_data.get('focus_points_y', [])
    if focus_points_y:
        x_focus_y = np.array([item[0] for item in focus_points_y])
        if len(x_focus_y) > 0:
            metrics['std_x_focus_y'] = np.std(x_focus_y)
        else:
            metrics['std_x_focus_y'] = float('inf')
    else:
        metrics['std_x_focus_y'] = float('inf')

    focus_points_z = global_data.get('focus_points_z', [])
    if focus_points_z:
        x_focus_z = np.array([item[0] for item in focus_points_z])
        if len(x_focus_z) > 0:
            metrics['std_x_focus_z'] = np.std(x_focus_z)
        else:
            metrics['std_x_focus_z'] = float('inf')
    else:
        metrics['std_x_focus_z'] = float('inf')

    # Petzval slopes
    if focus_points_y:
        xs_xz = np.array([item[0] for item in focus_points_y])
        z2s = np.array([item[2]**2 for item in focus_points_y])
        if len(xs_xz) > 1:
            coeffs_xz = np.polyfit(z2s, xs_xz, 1)
            slope_xz_abs = abs(coeffs_xz[0])
            predicted = coeffs_xz[0] * z2s + coeffs_xz[1]
            ss_res = np.sum((xs_xz - predicted)**2)
            ss_tot = np.sum((xs_xz - np.mean(xs_xz))**2)
            r_squared_xz = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            if r2_threshold is None or r_squared_xz <= r2_threshold:
                metrics['slope_y'] = slope_xz_abs
            else:
                metrics['slope_y'] = float('inf')
        else:
            metrics['slope_y'] = float('inf')
    else:
        metrics['slope_y'] = float('inf')

    if focus_points_z:
        xs_xy = np.array([item[0] for item in focus_points_z])
        y2s = np.array([item[1]**2 for item in focus_points_z])
        if len(xs_xy) > 1:
            coeffs_xy = np.polyfit(y2s, xs_xy, 1)
            slope_xy_abs = abs(coeffs_xy[0])
            predicted = coeffs_xy[0] * y2s + coeffs_xy[1]
            ss_res = np.sum((xs_xy - predicted)**2)
            ss_tot = np.sum((xs_xy - np.mean(xs_xy))**2)
            r_squared_xy = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            if r2_threshold is None or r_squared_xy <= r2_threshold:
                metrics['slope_z'] = slope_xy_abs
            else:
                metrics['slope_z'] = float('inf')
        else:
            metrics['slope_z'] = float('inf')
    else:
        metrics['slope_z'] = float('inf')
    
    # Calculate focus plane term (|intercept - 73|)
    # For XZ plane (y focus criterion)
    if focus_points_y:
        xs_xz = np.array([item[0] for item in focus_points_y])
        z2s = np.array([item[2]**2 for item in focus_points_y])
        if len(xs_xz) > 1:
            coeffs_xz = np.polyfit(z2s, xs_xz, 1)
            intercept_xz = coeffs_xz[1]
            focus_plane_term_xz = abs(intercept_xz - 73)
            metrics['focus_plane_term_xz'] = focus_plane_term_xz
        else:
            metrics['focus_plane_term_xz'] = float('inf')
    else:
        metrics['focus_plane_term_xz'] = float('inf')
    
    # For XY plane (z focus criterion)
    if focus_points_z:
        xs_xy = np.array([item[0] for item in focus_points_z])
        y2s = np.array([item[1]**2 for item in focus_points_z])
        if len(xs_xy) > 1:
            coeffs_xy = np.polyfit(y2s, xs_xy, 1)
            intercept_xy = coeffs_xy[1]
            focus_plane_term_xy = abs(intercept_xy - 73)
            metrics['focus_plane_term_xy'] = focus_plane_term_xy
        else:
            metrics['focus_plane_term_xy'] = float('inf')
    else:
        metrics['focus_plane_term_xy'] = float('inf')

    return metrics


def data_viewer(data, mode='single', focus_axis='y', fg_idx=None, fly2_file=None, r2_threshold=0.2, num_groups=5, y_range=(-0.5, 0.5), z_range=(-0.5, 0.5), x_range=(73.0, 166.0), electron_energy=None):
    """
    Displays histograms and focus plots of particle data.

    This function can operate in two modes:
    - 'single': Displays a single histogram for a specified or default field gradient.
    - 'multiple': Displays a 2x2 interactive plot with histograms for Y and Z axes on the left,
                  and focus scatter plots for Y and Z on the right, with a slider to switch
                  between different field gradients.

    Args:
        data (dict): A dictionary containing the processed data. Expected structure is
                     data[field_gradient][lens_VMI]['global'][counts/bins_y/z, focus_xyz].
        mode (str, optional): The mode of operation, either 'single' or 'multiple'.
                              Defaults to 'single'.
        focus_axis (str, optional): The axis to focus on in 'single' mode ('y' or 'z').
                                    Defaults to 'y'.
        fg_idx (int, optional): The index of the field gradient to display in 'single' mode.
                                Defaults to None (first one).
        fly2_file (str, optional): Path to .fly2 file to read actual emission angles.
    """
    if data is None:
        print("Error: Data is None, likely file parsing failed.")
        return

    # Get emission angles
    if fly2_file:
        group_angles = parse_fly2_file(fly2_file)
        angles = [f'Az {az}°, El {el}°' for az, el in group_angles.values()]
    else:
        # Default angles
        angles = [f'Az {az}°, El {el}°' for az, el in [(0,0),(0,180),(90,45),(-90,45),(90,90),(-90,90),(90,135),(-90,135)]]

    # Compute metrics for highest peak and closest to zero across all fg and lens_VMI combinations
    fg_keys = sorted(data.keys())
    max_peak_y_fg = None
    max_peak_y_lens = None
    max_peak_y_ke = None
    max_peak_y = 0
    min_std_y_fg = None
    min_std_y_lens = None
    min_std_y_ke = None
    min_std_y = float('inf')
    max_peak_z_fg = None
    max_peak_z_lens = None
    max_peak_z_ke = None
    max_peak_z = 0
    min_std_z_fg = None
    min_std_z_lens = None
    min_std_z_ke = None
    min_std_z = float('inf')
    min_std_x_focus_y_fg = None
    min_std_x_focus_y_lens = None
    min_std_x_focus_y_ke = None
    min_std_x_focus_y = float('inf')
    min_std_x_focus_z_fg = None
    min_std_x_focus_z_lens = None
    min_std_x_focus_z_ke = None
    min_std_x_focus_z = float('inf')
    min_slope_y = float('inf')
    min_slope_y_fg = None
    min_slope_y_lens = None
    min_slope_y_ke = None
    min_slope_z = float('inf')
    min_slope_z_fg = None
    min_slope_z_lens = None
    min_slope_z_ke = None
    
    # Variables for focus plane term (|intercept - 73|)
    min_focus_plane_term_xz = float('inf')
    min_focus_plane_term_xz_fg = None
    min_focus_plane_term_xz_lens = None
    min_focus_plane_term_xz_ke = None
    min_focus_plane_term_xy = float('inf')
    min_focus_plane_term_xy_fg = None
    min_focus_plane_term_xy_lens = None
    min_focus_plane_term_xy_ke = None

    # Optimization: Use multi-threading to parallelize metrics computation across fg/lens combinations
    # Why: Each fg/lens pair's metrics (peaks, stds, focus points) are independent, so we can compute them concurrently on multiple CPU cores
    # This reduces total execution time for large datasets with many combinations
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_metrics_for_fg_lens_ke, fg, lens, data[fg][lens][ke]['global'], r2_threshold, fg, lens, ke): (fg, lens, ke) for fg in fg_keys for lens in data[fg].keys() for ke in data[fg][lens].keys() if 'global' in data[fg][lens][ke]}
        metrics_list = []
        for future in as_completed(futures):
            mets = future.result()
            metrics_list.append(mets)

    for mets in metrics_list:
        fg = mets['fg']
        lens = mets['lens']
        ke = mets['ke']
        if mets['peak_y'] > max_peak_y:
            max_peak_y = mets['peak_y']
            max_peak_y_fg = fg
            max_peak_y_lens = lens
            max_peak_y_ke = ke
        if mets['std_y'] < min_std_y:
            min_std_y = mets['std_y']
            min_std_y_fg = fg
            min_std_y_lens = lens
            min_std_y_ke = ke
        if mets['peak_z'] > max_peak_z:
            max_peak_z = mets['peak_z']
            max_peak_z_fg = fg
            max_peak_z_lens = lens
            max_peak_z_ke = ke
        if mets['std_z'] < min_std_z:
            min_std_z = mets['std_z']
            min_std_z_fg = fg
            min_std_z_lens = lens
            min_std_z_ke = ke
        if mets['std_x_focus_y'] < min_std_x_focus_y:
            min_std_x_focus_y = mets['std_x_focus_y']
            min_std_x_focus_y_fg = mets['fg']
            min_std_x_focus_y_lens = mets['lens']
            min_std_x_focus_y_ke = mets['ke']
        if mets['std_x_focus_z'] < min_std_x_focus_z:
            min_std_x_focus_z = mets['std_x_focus_z']
            min_std_x_focus_z_fg = mets['fg']
            min_std_x_focus_z_lens = mets['lens']
            min_std_x_focus_z_ke = mets['ke']
        if mets['slope_y'] < min_slope_y:
            min_slope_y = mets['slope_y']
            min_slope_y_fg = fg
            min_slope_y_lens = lens
            min_slope_y_ke = ke
        if mets['slope_z'] < min_slope_z:
            min_slope_z = mets['slope_z']
            min_slope_z_fg = fg
            min_slope_z_lens = lens
            min_slope_z_ke = ke
        if mets['focus_plane_term_xz'] < min_focus_plane_term_xz:
            min_focus_plane_term_xz = mets['focus_plane_term_xz']
            min_focus_plane_term_xz_fg = fg
            min_focus_plane_term_xz_lens = lens
            min_focus_plane_term_xz_ke = ke
        if mets['focus_plane_term_xy'] < min_focus_plane_term_xy:
            min_focus_plane_term_xy = mets['focus_plane_term_xy']
            min_focus_plane_term_xy_fg = fg
            min_focus_plane_term_xy_lens = lens
            min_focus_plane_term_xy_ke = ke

    # Prepare info strings
    info_y = f"Y: Highest peak FG {max_peak_y_fg} (Lens {max_peak_y_lens}, KE {max_peak_y_ke}), minimal std FG {min_std_y_fg} (Lens {min_std_y_lens}, KE {min_std_y_ke})"
    info_z = f"Z: Highest peak FG {max_peak_z_fg} (Lens {max_peak_z_lens}, KE {max_peak_z_ke}), minimal std FG {min_std_z_fg} (Lens {min_std_z_lens}, KE {min_std_z_ke})"

    # Find best avg min dist and std min dist for Y and Z
    min_avg_min_dist_y = float('inf')
    min_avg_min_dist_y_fg = None
    min_avg_min_dist_y_lens = None
    min_avg_min_dist_z = float('inf')
    min_avg_min_dist_z_fg = None
    min_avg_min_dist_z_lens = None
    min_std_min_dist_y = float('inf')
    min_std_min_dist_y_fg = None
    min_std_min_dist_y_lens = None
    min_std_min_dist_z = float('inf')
    min_std_min_dist_z_fg = None
    min_std_min_dist_z_lens = None

    for metrics in metrics_list:
        fg = metrics['fg']
        lens = metrics['lens']
        global_data = data[fg][lens]['global']
        avg_min_dist_y = global_data.get('avg_min_dist_y', float('inf'))
        avg_min_dist_z = global_data.get('avg_min_dist_z', float('inf'))
        std_min_dist_y = global_data.get('std_min_dist_y', float('inf'))
        std_min_dist_z = global_data.get('std_min_dist_z', float('inf'))
        if avg_min_dist_y < min_avg_min_dist_y:
            min_avg_min_dist_y = avg_min_dist_y
            min_avg_min_dist_y_fg = fg
            min_avg_min_dist_y_lens = lens
        if avg_min_dist_z < min_avg_min_dist_z:
            min_avg_min_dist_z = avg_min_dist_z
            min_avg_min_dist_z_fg = fg
            min_avg_min_dist_z_lens = lens
        if std_min_dist_y < min_std_min_dist_y:
            min_std_min_dist_y = std_min_dist_y
            min_std_min_dist_y_fg = fg
            min_std_min_dist_y_lens = lens
        if std_min_dist_z < min_std_min_dist_z:
            min_std_min_dist_z = std_min_dist_z
            min_std_min_dist_z_fg = fg
            min_std_min_dist_z_lens = lens

    info_dist = f"Avg min distance Y: smallest {min_avg_min_dist_y:.4f} at FG {min_avg_min_dist_y_fg} (Lens {min_avg_min_dist_y_lens})\nAvg min distance Z: smallest {min_avg_min_dist_z:.4f} at FG {min_avg_min_dist_z_fg} (Lens {min_avg_min_dist_z_lens})\nStd min distance Y: smallest {min_std_min_dist_y:.4f} at FG {min_std_min_dist_y_fg} (Lens {min_std_min_dist_y_lens})\nStd min distance Z: smallest {min_std_min_dist_z:.4f} at FG {min_std_min_dist_z_fg} (Lens {min_std_min_dist_z_lens})"
    info_x = f"X focus lowest std in Y focus criterion: FG {min_std_x_focus_y_fg} (Lens {min_std_x_focus_y_lens}, KE {min_std_x_focus_y_ke})\nX focus lowest std in Z focus criterion: FG {min_std_x_focus_z_fg} (Lens {min_std_x_focus_z_lens}, KE {min_std_x_focus_z_ke})\nSmallest Petzval slope in Y focus criterion: FG {min_slope_y_fg} (Lens {min_slope_y_lens}, KE {min_slope_y_ke})\nSmallest Petzval slope in Z focus criterion: FG {min_slope_z_fg} (Lens {min_slope_z_lens}, KE {min_slope_z_ke})\nSmallest focus plane term (|intercept-73|) in Y focus criterion: FG {min_focus_plane_term_xz_fg} (Lens {min_focus_plane_term_xz_lens}, KE {min_focus_plane_term_xz_ke})\nSmallest focus plane term (|intercept-73|) in Z focus criterion: FG {min_focus_plane_term_xy_fg} (Lens {min_focus_plane_term_xy_lens}, KE {min_focus_plane_term_xy_ke})"
    if mode == 'single':
        if fg_idx is None:
            fg_keys = list(data.keys())
            fg = fg_keys[0]  # default to first
        else:
            fg_keys = list(data.keys())
            fg = fg_keys[fg_idx]
        lens_VMI = list(data[fg].keys())[0]
        if focus_axis == 'y':
            counts, bins = data[fg][lens_VMI]['global']['counts_y'], data[fg][lens_VMI]['global']['bins_y']
            info = info_y
        else:
            counts, bins = data[fg][lens_VMI]['global']['counts_z'], data[fg][lens_VMI]['global']['bins_z']
            info = info_z

        plt.figure()
        bin_centers = (bins[:-1] + bins[1:]) / 2 if len(bins) > 1 else np.array([])
        if len(bins) > 1:
            w = np.diff(bins)
            plt.bar(bin_centers, counts, width=w, align='center', edgecolor='k', alpha=0.7)
        # Calculate peak position
        if len(counts) > 0:
            i = np.argmax(counts)
            peak_pos = bin_centers[i] if len(bin_centers) > i else 0
            peak_pct = (np.max(counts) / np.sum(counts)) * 100 if np.sum(counts) > 0 else 0
        else:
            peak_pos = 0
            peak_pct = 0

        title_str = f"Histogram for {focus_axis.upper()} axis, FG {fg}\n{info}\nPeak position: {peak_pos:.3f}, Peak %: {peak_pct:.2f}%"
        plt.title(title_str)
        plt.xlabel(f'{focus_axis.upper()} position')
        plt.ylabel('Counts')
        plt.show()

    elif mode == 'multiple':

        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(233)
        ax6 = fig.add_subplot(236)
        fg_keys = sorted(data.keys())

        # Precompute all data for faster updates using multi-threading
        all_data = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            for fg in fg_keys:
                for lens in data[fg].keys():
                    for ke in data[fg][lens].keys():
                        # Use the first electron_energy value for precomputation
                        ke_val = electron_energy[0] if electron_energy else 5.0
                        # Only compute for ke values that match our electron_energy sequence
                        # Handle the case where electron_energy might be a string (e.g., 'global')
                        if electron_energy is None or (isinstance(electron_energy, str) and electron_energy == 'global') or (not isinstance(electron_energy, str) and ke in electron_energy):
                            futures[executor.submit(compute_for_fg_lens, fg, lens, data[fg][lens][ke], ke_val)] = (fg, lens, ke)
            
            for future in as_completed(futures):
                fg, lens, ke = futures[future]
                result = future.result()
                if fg not in all_data:
                    all_data[fg] = {}
                if lens not in all_data[fg]:
                    all_data[fg][lens] = {}
                all_data[fg][lens][ke] = result

        # Compute global min avg focus widths from all_data
        min_avg_width_y = float('inf')
        min_avg_width_y_fg = None
        min_avg_width_y_lens = None
        min_avg_width_y_ke = None
        min_avg_width_z = float('inf')
        min_avg_width_z_fg = None
        min_avg_width_z_lens = None
        min_avg_width_z_ke = None

        for fg in fg_keys:
            for lens in data[fg]:
                for ke in data[fg][lens]:
                    if ke not in all_data.get(fg, {}).get(lens, {}):
                        continue
                    d = all_data[fg][lens][ke]
                    
                    global_data_y = d.get('focus_points_y', [])
                    widths_y = [item[3] for item in global_data_y] if global_data_y else []
                    
                    global_data_z = d.get('focus_points_z', [])
                    widths_z = [item[3] for item in global_data_z] if global_data_z else []

                    if widths_y:
                        avg_y = np.mean(widths_y)
                        if not np.isnan(avg_y) and avg_y < min_avg_width_y:
                            min_avg_width_y = avg_y
                            min_avg_width_y_fg = fg
                            min_avg_width_y_lens = lens
                            min_avg_width_y_ke = ke
                    if widths_z:
                        avg_z = np.mean(widths_z)
                        if not np.isnan(avg_z) and avg_z < min_avg_width_z:
                            min_avg_width_z = avg_z
                            min_avg_width_z_fg = fg
                            min_avg_width_z_lens = lens
                            min_avg_width_z_ke = ke

        # If no valid widths found, set FG and Lens to 'N/A'
        if min_avg_width_y_fg is None:
            min_avg_width_y_fg = 'N/A'
            min_avg_width_y_lens = 'N/A'
            min_avg_width_y_ke = 'N/A'
        if min_avg_width_z_fg is None:
            min_avg_width_z_fg = 'N/A'
            min_avg_width_z_lens = 'N/A'
            min_avg_width_z_ke = 'N/A'

        info_focus = (f"Lowest Y focus width: FG {min_avg_width_y_fg} Lens {min_avg_width_y_lens} KE {min_avg_width_y_ke}, width {format_value(min_avg_width_y)}\n"
                      f"Lowest Z focus width: FG {min_avg_width_z_fg} Lens {min_avg_width_z_lens} KE {min_avg_width_z_ke}, width {format_value(min_avg_width_z)}")

        fig.subplots_adjust(bottom=0.45, hspace=1.0, wspace=0.6, top=0.9)
        ax_info = fig.add_axes([0.1, 0.15, 0.8, 0.25])
        ax_info.axis('off')
        # Show fixed info and current parameters text
        info_fixed = f"{info_x}\n{info_focus}\n{info_dist}"
        ax_info.text(0, 1, info_fixed, transform=ax_info.transAxes, fontsize=7, verticalalignment='top', fontfamily='monospace')
        current_params_text = ax_info.text(0, 0, "", transform=ax_info.transAxes, fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax_fg_slider = plt.axes([0.1, 0.2, 0.8, 0.03])
        fg_slider = Slider(
            ax=ax_fg_slider,
            label='Field Gradient',
            valmin=0,
            valmax=max(0.1, len(fg_keys) - 1),
            valinit=0,
            valstep=1
        )

        ax_lens_slider = plt.axes([0.1, 0.15, 0.8, 0.03])
        lens_slider = Slider(
            ax=ax_lens_slider,
            label='Lens VMI',
            valmin=0,
            valmax=0.1,  # Will be updated
            valinit=0,
            valstep=1
        )

        # Handle electron_energy parameter
        if electron_energy is None:
            # If no electron_energy provided, get unique ke values from data
            all_ke = set()
            for fg in data:
                for lens_VMI in data[fg]:
                    for ke in data[fg][lens_VMI]:
                        all_ke.add(float(ke))
            electron_energy = sorted(list(all_ke))
            if not electron_energy:
                electron_energy = [5.0]  # Default value
        else:
            # Convert to list if it's a numpy array
            electron_energy = list(electron_energy) if hasattr(electron_energy, '__iter__') else [electron_energy]
        
        # Handle electron_energy parameter
        if electron_energy is None:
            # If no electron_energy provided, get unique ke values from data
            all_ke = set()
            for fg in data:
                for lens_VMI in data[fg]:
                    for ke in data[fg][lens_VMI]:
                        all_ke.add(float(ke))
            electron_energy = sorted(list(all_ke))
            if not electron_energy:
                electron_energy = [5.0]  # Default value
        
        ax_ke_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
        ke_slider = Slider(
            ax=ax_ke_slider,
            label='Kinetic Energy',
            valmin=0,
            valmax=max(0, len(electron_energy) - 1),
            valinit=0,
            valstep=1
        )

        # Store sliders in figure to prevent garbage collection
        fig.fg_slider = fg_slider
        fig.lens_slider = lens_slider
        fig.ke_slider = ke_slider

        def update_lens_slider(fg_idx, current_lens=None):
            # Ensure fg_idx is within bounds
            fg_idx = min(max(0, fg_idx), len(fg_keys) - 1)
            fg = fg_keys[fg_idx]
            # Check if fg exists in data
            if fg not in data:
                lens_slider.valtext.set_text('No data available')
                return
            lens_keys = sorted(data[fg].keys())
            lens_slider.valmax = max(0.1, len(lens_keys) - 1)
            lens_slider.valmin = 0
            lens_slider.ax.set_xlim(lens_slider.valmin, lens_slider.valmax)
            if current_lens is not None and current_lens in lens_keys:
                lens_slider.val = lens_keys.index(current_lens)
            else:
                lens_slider.val = 0
            # Ensure the slider value is within bounds
            lens_idx = int(lens_slider.val)
            lens_idx = min(lens_idx, len(lens_keys) - 1)
            # Handle empty lens_keys case
            if len(lens_keys) > 0:
                lens_slider.valtext.set_text(f'{lens_keys[lens_idx]}')
            else:
                lens_slider.valtext.set_text('No lenses available')

        def update(val=None):
            fg_idx = int(fg_slider.val)
            # Ensure fg_idx is within bounds
            fg_idx = min(max(0, fg_idx), len(fg_keys) - 1)
            fg = fg_keys[fg_idx]
            fg_slider.valtext.set_text(f'{fg}')

            lens_idx = int(lens_slider.val)
            lens_keys = sorted(data[fg].keys())
            # Ensure the lens_idx is within bounds
            lens_idx = min(lens_idx, len(lens_keys) - 1)
            # Handle empty lens_keys case
            if len(lens_keys) > 0:
                lens_VMI = lens_keys[lens_idx]
                lens_slider.valtext.set_text(f'{lens_VMI}')
            else:
                lens_VMI = None
                lens_slider.valtext.set_text('No lenses available')

            ke_idx = int(ke_slider.val)
            # Ensure ke_idx is within bounds
            ke_idx = min(ke_idx, len(electron_energy) - 1)
            ke = electron_energy[ke_idx]
            ke_slider.valtext.set_text(f'{ke:.1f}')

            # Compute searched parameters for current ke
            current_metrics = [m for m in metrics_list if float(m['ke']) == float(ke)]

            # Reset mins
            max_peak_y_fg_ke = None
            max_peak_y_lens_ke = None
            min_std_y_fg_ke = None
            min_std_y_lens_ke = None
            max_peak_z_fg_ke = None
            max_peak_z_lens_ke = None
            min_std_z_fg_ke = None
            min_std_z_lens_ke = None
            min_std_x_focus_y_fg_ke = None
            min_std_x_focus_y_lens_ke = None
            min_std_x_focus_z_fg_ke = None
            min_std_x_focus_z_lens_ke = None
            min_slope_y_fg_ke = None
            min_slope_y_lens_ke = None
            min_slope_z_fg_ke = None
            min_slope_z_lens_ke = None
            min_avg_min_dist_y_fg_ke = None
            min_avg_min_dist_y_lens_ke = None
            min_avg_min_dist_z_fg_ke = None
            min_avg_min_dist_z_lens_ke = None
            min_std_min_dist_y_fg_ke = None
            min_std_min_dist_y_lens_ke = None
            min_std_min_dist_z_fg_ke = None
            min_std_min_dist_z_lens_ke = None
            
            # Variables for focus plane term (|intercept - 73|) for current ke
            min_focus_plane_term_xz_ke = float('inf')
            min_focus_plane_term_xz_fg_ke = None
            min_focus_plane_term_xz_lens_ke = None
            min_focus_plane_term_xy_ke = float('inf')
            min_focus_plane_term_xy_fg_ke = None
            min_focus_plane_term_xy_lens_ke = None

            max_peak_y_ke = 0
            min_std_y_ke = float('inf')
            max_peak_z_ke = 0
            min_std_z_ke = float('inf')
            min_std_x_focus_y_ke = float('inf')
            min_std_x_focus_z_ke = float('inf')
            min_slope_y_ke = float('inf')
            min_slope_z_ke = float('inf')
            min_avg_min_dist_y_ke = float('inf')
            min_avg_min_dist_z_ke = float('inf')
            min_std_min_dist_y_ke = float('inf')
            min_std_min_dist_z_ke = float('inf')

            for mets in current_metrics:
                fg_m = mets['fg']
                lens_m = mets['lens']
                if mets['peak_y'] > max_peak_y_ke:
                    max_peak_y_ke = mets['peak_y']
                    max_peak_y_fg_ke = fg_m
                    max_peak_y_lens_ke = lens_m
                if mets['std_y'] < min_std_y_ke:
                    min_std_y_ke = mets['std_y']
                    min_std_y_fg_ke = fg_m
                    min_std_y_lens_ke = lens_m
                if mets['peak_z'] > max_peak_z_ke:
                    max_peak_z_ke = mets['peak_z']
                    max_peak_z_fg_ke = fg_m
                    max_peak_z_lens_ke = lens_m
                if mets['std_z'] < min_std_z_ke:
                    min_std_z_ke = mets['std_z']
                    min_std_z_fg_ke = fg_m
                    min_std_z_lens_ke = lens_m
                if mets['std_x_focus_y'] < min_std_x_focus_y_ke:
                    min_std_x_focus_y_ke = mets['std_x_focus_y']
                    min_std_x_focus_y_fg_ke = fg_m
                    min_std_x_focus_y_lens_ke = lens_m
                if mets['std_x_focus_z'] < min_std_x_focus_z_ke:
                    min_std_x_focus_z_ke = mets['std_x_focus_z']
                    min_std_x_focus_z_fg_ke = fg_m
                    min_std_x_focus_z_lens_ke = lens_m
                if mets['slope_y'] < min_slope_y_ke:
                    min_slope_y_ke = mets['slope_y']
                    min_slope_y_fg_ke = fg_m
                    min_slope_y_lens_ke = lens_m
                if mets['slope_z'] < min_slope_z_ke:
                    min_slope_z_ke = mets['slope_z']
                    min_slope_z_fg_ke = fg_m
                    min_slope_z_lens_ke = lens_m
                if mets['focus_plane_term_xz'] < min_focus_plane_term_xz_ke:
                    min_focus_plane_term_xz_ke = mets['focus_plane_term_xz']
                    min_focus_plane_term_xz_fg_ke = fg_m
                    min_focus_plane_term_xz_lens_ke = lens_m
                if mets['focus_plane_term_xy'] < min_focus_plane_term_xy_ke:
                    min_focus_plane_term_xy_ke = mets['focus_plane_term_xy']
                    min_focus_plane_term_xy_fg_ke = fg_m
                    min_focus_plane_term_xy_lens_ke = lens_m

                global_data_m = data[fg_m][lens_m][ke]['global']
                avg_min_dist_y = global_data_m.get('avg_min_dist_y', float('inf'))
                avg_min_dist_z = global_data_m.get('avg_min_dist_z', float('inf'))
                std_min_dist_y = global_data_m.get('std_min_dist_y', float('inf'))
                std_min_dist_z = global_data_m.get('std_min_dist_z', float('inf'))
                if avg_min_dist_y < min_avg_min_dist_y_ke:
                    min_avg_min_dist_y_ke = avg_min_dist_y
                    min_avg_min_dist_y_fg_ke = fg_m
                    min_avg_min_dist_y_lens_ke = lens_m
                if avg_min_dist_z < min_avg_min_dist_z_ke:
                    min_avg_min_dist_z_ke = avg_min_dist_z
                    min_avg_min_dist_z_fg_ke = fg_m
                    min_avg_min_dist_z_lens_ke = lens_m
                if std_min_dist_y < min_std_min_dist_y_ke:
                    min_std_min_dist_y_ke = std_min_dist_y
                    min_std_min_dist_y_fg_ke = fg_m
                    min_std_min_dist_y_lens_ke = lens_m
                if std_min_dist_z < min_std_min_dist_z_ke:
                    min_std_min_dist_z_ke = std_min_dist_z
                    min_std_min_dist_z_fg_ke = fg_m
                    min_std_min_dist_z_lens_ke = lens_m

            # Construct info strings for current ke
            info_x_ke = f"X focus lowest std in Y focus criterion: FG {min_std_x_focus_y_fg_ke} (Lens {min_std_x_focus_y_lens_ke})\nX focus lowest std in Z focus criterion: FG {min_std_x_focus_z_fg_ke} (Lens {min_std_x_focus_z_lens_ke})\nSmallest Petzval slope in Y focus criterion: FG {min_slope_y_fg_ke} (Lens {min_slope_y_lens_ke})\nSmallest Petzval slope in Z focus criterion: FG {min_slope_z_fg_ke} (Lens {min_slope_z_lens_ke})\nSmallest focus plane term (|intercept-73|) in Y focus criterion: FG {min_focus_plane_term_xz_fg_ke} (Lens {min_focus_plane_term_xz_lens_ke})\nSmallest focus plane term (|intercept-73|) in Z focus criterion: FG {min_focus_plane_term_xy_fg_ke} (Lens {min_focus_plane_term_xy_lens_ke})"
            info_dist_ke = f"Avg min distance Y: smallest {min_avg_min_dist_y_ke:.4f} at FG {min_avg_min_dist_y_fg_ke} (Lens {min_avg_min_dist_y_lens_ke})\nAvg min distance Z: smallest {min_avg_min_dist_z_ke:.4f} at FG {min_avg_min_dist_z_fg_ke} (Lens {min_avg_min_dist_z_lens_ke})\nStd min distance Y: smallest {min_std_min_dist_y_ke:.4f} at FG {min_std_min_dist_y_fg_ke} (Lens {min_std_min_dist_y_lens_ke})\nStd min distance Z: smallest {min_std_min_dist_z_ke:.4f} at FG {min_std_min_dist_z_fg_ke} (Lens {min_std_min_dist_z_lens_ke})"

            # Compute min avg focus widths for current ke
            min_avg_width_y_ke = float('inf')
            min_avg_width_y_fg_ke = None
            min_avg_width_y_lens_ke = None
            min_avg_width_z_ke = float('inf')
            min_avg_width_z_fg_ke = None
            min_avg_width_z_lens_ke = None

            for fg_ke in fg_keys:
                for lens_ke in data[fg_ke]:
                    if ke not in data[fg_ke][lens_ke]:
                        continue
                    d_temp = all_data[fg_ke][lens_ke][ke]
                    global_data_y = d_temp.get('focus_points_y', [])
                    widths_y = [item[3] for item in global_data_y] if global_data_y else []
                    global_data_z = d_temp.get('focus_points_z', [])
                    widths_z = [item[3] for item in global_data_z] if global_data_z else []

                    if widths_y:
                        avg_y = np.mean(widths_y)
                        if not np.isnan(avg_y) and avg_y < min_avg_width_y_ke:
                            min_avg_width_y_ke = avg_y
                            min_avg_width_y_fg_ke = fg_ke
                            min_avg_width_y_lens_ke = lens_ke
                    if widths_z:
                        avg_z = np.mean(widths_z)
                        if not np.isnan(avg_z) and avg_z < min_avg_width_z_ke:
                            min_avg_width_z_ke = avg_z
                            min_avg_width_z_fg_ke = fg_ke
                            min_avg_width_z_lens_ke = lens_ke

            if min_avg_width_y_fg_ke is None:
                min_avg_width_y_fg_ke = 'N/A'
                min_avg_width_y_lens_ke = 'N/A'
            if min_avg_width_z_fg_ke is None:
                min_avg_width_z_fg_ke = 'N/A'
                min_avg_width_z_lens_ke = 'N/A'

            info_focus_ke = (f"Lowest Y focus width: FG {min_avg_width_y_fg_ke} Lens {min_avg_width_y_lens_ke}, width {format_value(min_avg_width_y_ke)}\n"
                             f"Lowest Z focus width: FG {min_avg_width_z_fg_ke} Lens {min_avg_width_z_lens_ke}, width {format_value(min_avg_width_z_ke)}")

            # Set info_fixed with current ke data
            info_fixed_new = f"{info_x_ke}\n{info_focus_ke}\n{info_dist_ke}"
            ax_info.texts[0].set_text(info_fixed_new)

            # Get global data for current fg/lens_VMI
            # Check if lens_VMI is None (no lenses available)
            if lens_VMI is None:
                # Skip the rest of the update if no lenses are available
                return
                
            # Check if ke exists in data structure
            if ke not in data[fg][lens_VMI]:
                # Try to find the closest ke value
                available_kes = list(data[fg][lens_VMI].keys())
                if available_kes:
                    ke_float = float(ke)
                    closest_ke = min(available_kes, key=lambda k: abs(float(k) - ke_float))
                    ke = closest_ke
            
            global_data = data[fg][lens_VMI][ke]['global']
            
            # Try to get precomputed data, or compute on-the-fly if not available
            if fg in all_data and lens_VMI in all_data[fg] and ke in all_data[fg][lens_VMI]:
                d = all_data[fg][lens_VMI][ke]
            else:
                # Compute on-the-fly if not precomputed
                d = compute_for_fg_lens(fg, lens_VMI, data[fg][lens_VMI][ke], electron_energy[0] if electron_energy else 5.0)

            # --- Update Y group distances ---
            ax1.clear()
            group_distances_y = global_data.get('group_distances_y', [])
            if group_distances_y:
                x_indices_y = list(range(len(group_distances_y)))
                range_vals = [g['range'] for g in group_distances_y]
                ax1.scatter(x_indices_y, range_vals, color='blue', marker='o', s=50)
                group_labels = ['0/180', '45/135', '90', '-45/-135', '-90'][:len(group_distances_y)]
                ax1.set_xticks(x_indices_y)
                ax1.set_xticklabels(group_labels)
                ax1.set_xlabel('Emission Angle Group')
                ax1.set_ylabel('Final Range (mm)')
                ax1.set_title('final position range along y axis - xy axis (z projection)')
                ax1.grid(True)
            else:
                ax1.set_title('No Y group data')

            # --- Update Y Focus Plot ---
            ax2.clear()
            ax2.set_xlabel('X position')
            ax2.set_ylabel('Z position')
            ax2.set_title(f'XZ Focus Points with Beam Radii')
            ax2.set_xlim(73, 166)
            ax2.set_ylim(-45, 45)
            # Add red circles for focus centers and blue square bars for radii
            if d.get('focus_points_y'):
                for item in d['focus_points_y']:
                    x, _, focus_z, w = item  # unpack x, y, z, w
                    ax2.plot(x, focus_z, 'ro', markersize=5)  # red circle for [x, z]
                    side = np.sqrt(2) * w
                    #ax2.add_patch(patches.Rectangle((x - side/2, focus_y - side/2), side, side, fill=True, alpha=0.5, color='blue'))
            if d['fit_text_xz']:
                ax2.text(0.05, 0.95, d['fit_text_xz'], transform=ax2.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # --- Update Z group distances ---
            ax3.clear()
            group_distances_z = global_data.get('group_distances_z', [])
            if group_distances_z:
                x_indices_z = list(range(len(group_distances_z)))
                range_vals_z = [g['range'] for g in group_distances_z]
                ax3.scatter(x_indices_z, range_vals_z, color='green', marker='s', s=50)
                group_labels_z = ['0/180', '45/135', '90', '-45/-135', '-90'][:len(group_distances_z)]
                ax3.set_xticks(x_indices_z)
                ax3.set_xticklabels(group_labels_z)
                ax3.set_xlabel('Emission Angle Group')
                ax3.set_ylabel('Final Range (mm)')
                ax3.set_title('final position range along z axis - xz axis (y projection)')
                ax3.grid(True)
            else:
                ax3.set_title('No Z group data')

            # --- Update Z Focus Plot ---
            ax4.clear()
            ax4.set_xlabel('X position')
            ax4.set_ylabel('Y position')
            ax4.set_title(f'XY Focus Points with Beam Radii')
            ax4.set_xlim(73, 166)
            ax4.set_ylim(-45, 45)
            # Add red circles for focus centers [x, focus_y] and blue squares for radii
            if d.get('focus_points_z'):
                for item in d['focus_points_z']:
                    x, focus_y, focus_z, w = item  # x, y (focus center y), z (focus center z), width_z
                    # For XY plot, plot at (x, focus_z)
                    ax4.plot(x, focus_y, 'ro', markersize=5)  # red circle
                    # And blue square for radii np.sqrt(2)*focus_width_z
                    side = np.sqrt(2) * w
                    #ax4.add_patch(patches.Rectangle((x - side/2, focus_y - side/2), side, side, fill=True, alpha=0.5, color='blue'))
            if d['fit_text_xy']:
                ax4.text(0.05, 0.95, d['fit_text_xy'], transform=ax4.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Calculate data collection efficiency
            local_data = data[fg][lens_VMI].get('local', {})
            num_particles_received = len(local_data)
            if local_data:
                first = list(local_data.values())[0]
                if isinstance(first, dict):
                    num_trajectory_per_particles = len(first.get('trajectories', []))
                else:
                    num_trajectory_per_particles = len(first)
            else:
                num_trajectory_per_particles = 0
            total_received_trajectory = num_particles_received * num_trajectory_per_particles
            # num_particles_generated * 8 = num_trajectory_generated
            num_particles_generated = num_groups
            num_trajectory_generated = num_particles_generated * 8
            efficiency = (total_received_trajectory / num_trajectory_generated) * 100 if num_trajectory_generated > 0 else 0

            M_square = global_data.get('M_square', [])
            M_rectangle = global_data.get('M_rectangle', [])
            dr = global_data.get('dr', [])
            # Update the text box with current parameters
            info_text = f"Current FG {fg}, Lens {lens_VMI}, Data Collection Efficiency: {efficiency:.1f}%, M_square: {M_square:.4f}, M_rectangle: {M_rectangle:.4f}, dr: {dr:.4f}"
            ax_info.texts[1].set_text(info_text)

            # Restore Focus width plots for Y and Z
            ax5.clear()
            focus_points_y = global_data.get('focus_points_y', [])
            all_widths_y = [item[3] for item in focus_points_y]
            if all_widths_y:
                num_points_y = len(all_widths_y)
                x_indices_y = list(range(num_points_y))
                avg_width_y = sum(all_widths_y) / num_points_y if num_points_y > 0 else 0
                ax5.bar(x_indices_y, all_widths_y, color='blue', alpha=0.7)
                ax5.axhline(y=avg_width_y, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_width_y:.4f}')
                # Compute y_range length
                y_length = y_range[1] - y_range[0]
                ax5.axhline(y=y_length, color='green', linestyle='-', linewidth=1, label=f'Y Range: {y_length}')
                group_labels = ['0/180', '45/135', '90', '-45/-135', '-90'][:num_points_y]
                ax5.set_xticks(x_indices_y)
                ax5.set_xticklabels(group_labels)
                ax5.set_xlabel('Emission Angle Group')
                ax5.set_ylabel('Focus Width XY axis')
                ax5.set_title(f'focus width xy axis(z projection)\nTotal Focus Points: {num_points_y}')
                ax5.legend()
            else:
                ax5.set_title('No Y width data')

            ax6.clear()
            focus_points_z = global_data.get('focus_points_z', [])
            all_widths_z = [item[3] for item in focus_points_z]
            if all_widths_z:
                num_points_z = len(all_widths_z)
                x_indices_z = list(range(num_points_z))
                avg_width_z = sum(all_widths_z) / num_points_z if num_points_z > 0 else 0
                ax6.bar(x_indices_z, all_widths_z, color='green', alpha=0.7)
                ax6.axhline(y=avg_width_z, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_width_z:.4f}')
                # Compute z_range length
                z_length = z_range[1] - z_range[0]
                ax6.axhline(y=z_length, color='green', linestyle='-', linewidth=1, label=f'Z Range: {z_length}')
                group_labels_z = ['0/180', '45/135', '90', '-45/-135', '-90'][:num_points_z]
                ax6.set_xticks(x_indices_z)
                ax6.set_xticklabels(group_labels_z)
                ax6.set_xlabel('Emission Angle Group')
                ax6.set_ylabel('Focus Width XZ aixs')
                ax6.set_title(f'focus width xz axis(y projection)\nTotal Focus Points: {num_points_z}')
                ax6.legend()
            else:
                ax6.set_title('No Z width data')

            fig.canvas.draw_idle()

        def on_fg_change(val):
            # Ensure fg_idx is within bounds
            fg_idx = min(max(0, int(fg_slider.val)), len(fg_keys) - 1)
            current_fg = fg_keys[fg_idx]
            current_lens_keys = sorted(data[current_fg].keys())
            # Handle empty lens_keys case
            if len(current_lens_keys) > 0:
                # Ensure lens_idx is within bounds
                lens_idx = min(max(0, int(lens_slider.val)), len(current_lens_keys) - 1)
                current_lens = current_lens_keys[lens_idx]
                update_lens_slider(fg_idx, current_lens)
            else:
                current_lens = None
                update_lens_slider(fg_idx, None)
            update()

        fg_slider.on_changed(on_fg_change)
        lens_slider.on_changed(update)
        ke_slider.on_changed(update)

        # Initial setup
        update_lens_slider(0)
        update()
        plt.show()
    else:
        raise ValueError("Mode must be 'single' or 'multiple'")


def compute_metrics_for_para(fg, lens_VMI, global_data):
    """
    Compute metrics for a single fg, lens combination for parameter landscape.
    """
    # peak_y
    counts_y = global_data['counts_y']
    peak_y = np.max(counts_y) if len(counts_y) > 0 else 0
    counts_z = global_data['counts_z']
    peak_z = np.max(counts_z) if len(counts_z) > 0 else 0

    # std_y, std_z
    std_y = global_data.get('std_dev_y', float('inf'))
    std_z = global_data.get('std_dev_z', float('inf'))

    # std_x_focus_y, std_x_focus_z
    focus_points_y = global_data.get('focus_points_y', [])
    x_focus_y = np.array([item[0] for item in focus_points_y])
    std_x_focus_y = np.std(x_focus_y) if len(x_focus_y) > 0 else float('inf')
    focus_points_z = global_data.get('focus_points_z', [])
    x_focus_z = np.array([item[0] for item in focus_points_z])
    std_x_focus_z = np.std(x_focus_z) if len(x_focus_z) > 0 else float('inf')

    # slope_y (petzval XZ)
    if len(x_focus_y) > 1:
        z2s = np.array([item[2]**2 for item in focus_points_y])
        coeffs_xz = np.polyfit(z2s, x_focus_y, 1)
        slope_y = abs(coeffs_xz[0])
    else:
        slope_y = float('inf')

    # slope_z (petzval XY)
    if len(x_focus_z) > 1:
        y2s = np.array([item[1]**2 for item in focus_points_z])
        coeffs_xy = np.polyfit(y2s, x_focus_z, 1)
        slope_z = abs(coeffs_xy[0])
    else:
        slope_z = float('inf')

    # Additional metrics: M_square, M_rectangle, dr
    M_square = global_data.get('M_square', float('inf'))
    M_rectangle = global_data.get('M_rectangle', float('inf'))
    dr = global_data.get('dr', float('inf'))

    return (fg, lens_VMI), {
        'peak_y': peak_y,
        'peak_z': peak_z,
        'std_y': std_y,
        'std_z': std_z,
        'std_x_focus_y': std_x_focus_y,
        'std_x_focus_z': std_x_focus_z,
        'slope_y': slope_y,
        'slope_z': slope_z,
        'M_square': M_square,
        'M_rectangle': M_rectangle,
        'dr': dr
    }


def run_optimized_simulations_with_ke(param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE):
    """
    Run optimized SIMION simulations with parallel Lua generation and sequential SIMION runs for a given ke.
    
    IMPROVED: Better error handling and diagnostics for SIMION failures.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import subprocess
    import os

    num_simulations = len(param['field_gradient'])
    
    # Verify IOB file exists before starting
    if not os.path.exists(IOB_FILE):
        print(f"ERROR: IOB file '{IOB_FILE}' not found!")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Available .iob files: {[f for f in os.listdir('.') if f.endswith('.iob')]}")
        return

    # Parallel Lua generation
    def generate_lua(field_idx):
        lua_filename = f"WORKING_TITLE_tao_ke_{field_idx}.lua"
        generate_simion_lua_file(field_idx, param, output_filename=lua_filename)
        return field_idx, lua_filename

    with ThreadPoolExecutor() as executor:
        lua_futures = {executor.submit(generate_lua, field_idx): field_idx for field_idx in range(num_simulations)}
        lua_files = {}
        for future in as_completed(lua_futures):
            field_idx, lua_file = future.result()
            lua_files[field_idx] = lua_file

    # Sequential SIMION runs to maintain correct Lua loading
    temp_files = {}
    for field_idx in range(num_simulations):
        lua_file = lua_files[field_idx]
        temp_out_file = f"temp_out_ke_{field_idx}.txt"
        
        # Copy the Lua to the standard name for loading by IOB
        with open(lua_file, 'r') as f:
            lua_content = f.read()
        with open(OUTPUT_FILENAME_LUA, 'w') as f:
            f.write(lua_content)
        
        command = f"simion.exe --nogui fly --recording-output={temp_out_file} {IOB_FILE}"
        
        # Capture both stdout and stderr for better error diagnosis
        result = subprocess.run(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the command executed successfully and the file was created
        if result.returncode != 0:
            print(f"Error: SIMION command failed with return code {result.returncode}")
            print(f"Command: {command}")
            if result.stdout:
                print(f"STDOUT: {result.stdout[:500]}")  # First 500 chars
            if result.stderr:
                print(f"STDERR: {result.stderr[:500]}")  # First 500 chars
            continue  # Skip to next simulation without adding this file to temp_files
            
        # Wait a moment to ensure the file is created
        import time
        time.sleep(0.5)
        
        # Only add to temp_files if it exists
        if os.path.exists(temp_out_file):
            temp_files[field_idx] = temp_out_file
        else:
            print(f"Warning: Output file {temp_out_file} was not created by SIMION")

    # Merge results sequentially to maintain exact order
    with open(OUT_FILE, 'a') as out_f:
        for field_idx in range(num_simulations):
            # Check if field_idx exists in temp_files before trying to access it
            if field_idx in temp_files:
                temp_file = temp_files[field_idx]
            else:
                print(f"Warning: No temporary file for field_idx {field_idx}, skipping...")
                continue
            # Check if the temp file exists before trying to read it
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as temp_f:
                    content = temp_f.read()
                # Insert parameters after the separator, including ke
                separator = "------ Begin Next Fly'm ------"
                idx = content.find(separator)
                if idx != -1:
                    idx += len(separator)
                    current_parameters = f"parameters = [{param['field_gradient'][field_idx]},{param['Offset_to_ground'][field_idx]},{param['lens_VMI'][field_idx]},{param['I_grid'][field_idx]},{param['VMI2'][field_idx]},{param['VMI1'][field_idx]},{param['e_grid'][field_idx]},{param['dt_e'][field_idx]},{ke}]\n"
                    content = content[:idx] + "\n" + current_parameters + content[idx:]
                out_f.write(content)
            else:
                print(f"Warning: Temporary file {temp_file} not found, skipping...")

    # Clean up temporary files
    for field_idx in range(num_simulations):
        # Check if field_idx exists in lua_files before trying to access it
        if field_idx in lua_files:
            lua_file = lua_files[field_idx]
            if os.path.exists(lua_file):
                os.remove(lua_file)
        # Check if field_idx exists in temp_files before trying to access it
        if field_idx in temp_files:
            temp_file = temp_files[field_idx]
            if os.path.exists(temp_file):
                os.remove(temp_file)

def run_optimized_simulations_with_ke_parallel(param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE, max_workers=None, batch_size=50):
    """
    Run optimized SIMION simulations with parallel Lua generation and parallel SIMION runs for a given ke.
    
    OPTIMIZED: Added batch processing to prevent memory issues with large parameter sets.
    
    This version uses improved error handling and file verification to ensure all simulations complete.
    
    Args:
        param: Dictionary of simulation parameters
        ke: Kinetic energy value
        OUTPUT_FILENAME_LUA: Standard Lua filename
        IOB_FILE: IOB filename
        OUT_FILE: Output file path
        max_workers: Maximum number of parallel workers (default: 75% of CPU cores)
        batch_size: Number of simulations per batch (default: 50)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import subprocess
    import os
    import time
    import shutil
    import multiprocessing
    import threading

    num_simulations = len(param['field_gradient'])
    
    # Determine optimal number of workers based on system resources
    if max_workers is None:
        # Use 75% of available CPUs but leave at least 1 core free
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    # Limit workers to prevent resource exhaustion
    max_workers = min(max_workers, 8)  # Cap at 8 workers to prevent system overload
    
    # Calculate number of batches
    num_batches = (num_simulations + batch_size - 1) // batch_size

    # Parallel Lua generation (already optimized)
    def generate_lua(field_idx):
        lua_filename = f"WORKING_TITLE_tao_ke_{field_idx}.lua"
        # Import function to avoid circular import
        from Utilis import generate_simion_lua_file
        generate_simion_lua_file(field_idx, param, output_filename=lua_filename)
        return field_idx, lua_filename

    # Generate all Lua files first
    lua_files = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        lua_futures = {executor.submit(generate_lua, field_idx): field_idx for field_idx in range(num_simulations)}
        for future in as_completed(lua_futures):
            field_idx, lua_file = future.result()
            lua_files[field_idx] = lua_file

    # Parallel SIMION execution using threads for file isolation
    def run_single_simion_thread(field_idx):
        """Run a single SIMION simulation with proper file isolation"""
        lua_file = lua_files[field_idx]
        temp_out_file = f"temp_out_ke_{field_idx}.txt"
        
        # Use a lock to ensure thread safety for file operations
        file_lock = threading.Lock()
        
        try:
            with file_lock:
                # Copy Lua to standard name for loading by IOB
                with open(lua_file, 'r') as f:
                    lua_content = f.read()
                with open(OUTPUT_FILENAME_LUA, 'w') as f:
                    f.write(lua_content)
            
            # Run SIMION using original working command with output capture for debugging
            command = f"simion.exe --nogui fly --recording-output={temp_out_file} {IOB_FILE}"
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait and verify file creation with retry mechanism
            max_wait = 10  # Maximum wait time in seconds
            wait_interval = 0.5
            wait_time = 0
            
            while wait_time < max_wait:
                if os.path.exists(temp_out_file) and os.path.getsize(temp_out_file) > 0:
                    break
                time.sleep(wait_interval)
                wait_time += wait_interval
            
            # Return results if successful
            if result.returncode == 0 and os.path.exists(temp_out_file) and os.path.getsize(temp_out_file) > 0:
                return field_idx, temp_out_file
            else:
                if result.returncode != 0:
                    print(f"Error: SIMION command failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"Error output: {result.stderr}")
                else:
                    print(f"Error: Output file {temp_out_file} was not created or is empty")
                return field_idx, None
        except Exception as e:
            print(f"Error running simulation {field_idx}: {e}")
            return field_idx, None

    # Execute simulations in batches to prevent resource exhaustion
    temp_files = {}
    completed_count = 0
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_simulations)
        batch_indices = list(range(batch_start, batch_end))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch simulations
            futures = {executor.submit(run_single_simion_thread, field_idx): field_idx for field_idx in batch_indices}
            
            # Wait for batch to complete
            for future in as_completed(futures):
                field_idx, temp_file = future.result()
                if temp_file:
                    temp_files[field_idx] = temp_file
                    completed_count += 1
        
        # Cleanup between batches
        gc.collect()

    # Merge results in order (same as original function)
    with open(OUT_FILE, 'a') as out_f:
        for field_idx in range(num_simulations):
            if field_idx in temp_files:
                temp_file = temp_files[field_idx]
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as temp_f:
                        content = temp_f.read()
                        # Insert parameters after the separator, including ke
                        separator = "------ Begin Next Fly'm ------"
                        idx = content.find(separator)
                        if idx != -1:
                            idx += len(separator)
                            current_parameters = f"parameters = [{param['field_gradient'][field_idx]},{param['Offset_to_ground'][field_idx]},{param['lens_VMI'][field_idx]},{param['I_grid'][field_idx]},{param['VMI2'][field_idx]},{param['VMI1'][field_idx]},{param['e_grid'][field_idx]},{param['dt_e'][field_idx]},{ke}]\n"
                            content = content[:idx] + "\n" + current_parameters + content[idx:]
                        out_f.write(content)
                        # Ensure data is written to disk immediately
                        out_f.flush()
                else:
                    print(f"Warning: Temporary file {temp_file} not found, skipping...")

    # Clean up temporary files (same as original function)
    for field_idx in range(num_simulations):
        # Check if field_idx exists in lua_files before trying to access it
        if field_idx in lua_files:
            lua_file = lua_files[field_idx]
            if os.path.exists(lua_file):
                os.remove(lua_file)
        # Check if field_idx exists in temp_files before trying to access it
        if field_idx in temp_files:
            temp_file = temp_files[field_idx]
            if os.path.exists(temp_file):
                os.remove(temp_file)

def run_optimized_simulations(param, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE):
    """
    Run optimized SIMION simulations with parallel Lua generation and sequential SIMION runs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import subprocess
    import os

    num_simulations = len(param['field_gradient'])

    # Parallel Lua generation
    def generate_lua(field_idx):
        lua_filename = f"WORKING_TITLE_tao_{field_idx}.lua"
        generate_simion_lua_file(field_idx, param, output_filename=lua_filename)
        return field_idx, lua_filename

    print("Generating Lua files in parallel...")
    with ThreadPoolExecutor() as executor:
        lua_futures = {executor.submit(generate_lua, field_idx): field_idx for field_idx in range(num_simulations)}
        lua_files = {}
        for future in as_completed(lua_futures):
            field_idx, lua_file = future.result()
            lua_files[field_idx] = lua_file

    # Sequential SIMION runs to maintain correct Lua loading
    print("Running SIMION simulations sequentially...")
    temp_files = {}
    for field_idx in range(num_simulations):
        lua_file = lua_files[field_idx]
        temp_out_file = f"temp_out_{field_idx}.txt"
        # Copy the Lua to the standard name for loading by IOB
        with open(lua_file, 'r') as f:
            lua_content = f.read()
        with open(OUTPUT_FILENAME_LUA, 'w') as f:
            f.write(lua_content)
        command = f"simion.exe --nogui fly --recording-output={temp_out_file} {IOB_FILE}"
        result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check if the command executed successfully and the file was created
        if result.returncode != 0:
            print(f"Error: SIMION command failed with return code {result.returncode}")
            print(f"Command: {command}")
            continue  # Skip to next simulation without adding this file to temp_files
            
        # Wait a moment to ensure the file is created
        import time
        time.sleep(0.5)
        
        # Only add to temp_files if it exists
        if os.path.exists(temp_out_file):
            temp_files[field_idx] = temp_out_file
        else:
            print(f"Warning: Output file {temp_out_file} was not created by SIMION")

    # Merge results sequentially to maintain exact order
    print("Merging results into out.txt...")
    with open(OUT_FILE, 'w') as out_f:
        for field_idx in range(num_simulations):
            # Check if field_idx exists in temp_files before trying to access it
            if field_idx in temp_files:
                temp_file = temp_files[field_idx]
                with open(temp_file, 'r') as temp_f:
                    content = temp_f.read()
                # Insert parameters after the separator
                separator = "------ Begin Next Fly'm ------"
                idx = content.find(separator)
                if idx != -1:
                    idx += len(separator)
                    current_parameters = f"parameters = [{param['field_gradient'][field_idx]},{param['Offset_to_ground'][field_idx]},{param['lens_VMI'][field_idx]},{param['I_grid'][field_idx]},{param['VMI2'][field_idx]},{param['VMI1'][field_idx]},{param['e_grid'][field_idx]},{param['dt_e'][field_idx]}]\n"
                    content = content[:idx] + "\n" + current_parameters + content[idx:]
                out_f.write(content)
            else:
                print(f"Warning: No temporary file for field_idx {field_idx}, skipping...")

    # Clean up temporary files
    print("Cleaning up temporary files...")
    for field_idx in range(num_simulations):
        # Check if field_idx exists in lua_files before trying to access it
        if field_idx in lua_files:
            lua_file = lua_files[field_idx]
            if os.path.exists(lua_file):
                os.remove(lua_file)
        # Check if field_idx exists in temp_files before trying to access it
        if field_idx in temp_files:
            temp_file = temp_files[field_idx]
            if os.path.exists(temp_file):
                os.remove(temp_file)


def para_2d_landscape(data, target='peak_y', fly2_file=None, initial_ke=None, ke_sequence=None):
    """
    Draws a 3D scatter plot showing the parameter landscape for a specified target function.
    If multiple ke exist, adds a slider to select the kinetic energy.
    Available targets: 'peak_y', 'peak_z', 'std_y', 'std_z', 'std_x_focus_y', 'std_x_focus_z', 'slope_y', 'slope_z', 'M_square', 'M_rectangle', 'dr'
    """
    valid_targets = ['peak_y', 'peak_z', 'std_y', 'std_z', 'std_x_focus_y', 'std_x_focus_z', 'slope_y', 'slope_z', 'M_square', 'M_rectangle', 'dr']
    if target not in valid_targets:
        raise ValueError(f"Invalid target. Choose from {valid_targets}")

    target_labels = {
        'peak_y': 'Y Axis Histogram Peak',
        'peak_z': 'Z Axis Histogram Peak',
        'std_y': 'Y Position Standard Deviation (mm)',
        'std_z': 'Z Position Standard Deviation (mm)',
        'std_x_focus_y': 'X Focus Position Std (Y criterion, mm)',
        'std_x_focus_z': 'X Focus Position Std (Z criterion, mm)',
        'slope_y': 'Petzval Slope (XZ plane)',
        'slope_z': 'Petzval Slope (XY plane)',
        'M_square': 'Magnification coefficient with square beam',
        'M_rectangle': 'Magnification coefficient with rectangle beam',
        'dr': 'Position spread dr (mm)'
    }

    # Check for multiple ke
    all_ke = set()
    for fg in data:
        for lens_VMI in data[fg]:
            for ke in data[fg][lens_VMI]:
                all_ke.add(float(ke))
    unique_kes = sorted(all_ke)

    # Use ke_sequence if provided, else use unique_kes
    kes_to_use = ke_sequence if ke_sequence is not None else unique_kes
    kes_to_use = [float(k) for k in kes_to_use]

    if len(kes_to_use) == 0:
        print("No ke sequence to visualize.")
        return
    elif len(kes_to_use) == 1:
        ke = kes_to_use[0]
        if ke in unique_kes:
            plot_surface(data, ke, target, target_labels)
        else:
            print(f"No data for KE {ke}.")
    else:
        initial_ke_idx = 0 if initial_ke is None else kes_to_use.index(float(initial_ke)) if float(initial_ke) in kes_to_use else 0

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax_slider = plt.axes([0.2, 0.05, 0.7, 0.03])
        ke_slider = Slider(ax=ax_slider, label='Kinetic Energy', valmin=0, valmax=len(kes_to_use)-1, valinit=initial_ke_idx, valstep=1)

        def update(val):
            ax.clear()
            ke_idx = int(round(ke_slider.val))
            ke = kes_to_use[ke_idx]
            ke_slider.valtext.set_text(f'{ke}')
            # Get data for this ke
            filtered_data = {}
            for fg in data:
                filtered_data[fg] = {}
                for lens_VMI in data[fg]:
                    if ke in data[fg][lens_VMI]:
                        filtered_data[fg][lens_VMI] = {'global': data[fg][lens_VMI][ke]['global']}
            if filtered_data:
                plot_surface(filtered_data, ke, target, target_labels, ax=ax)
            else:
                ax.text2D(0.5, 0.5, 'No data for this KE', transform=ax.transAxes)

        ke_slider.on_changed(update)
        update(0)
        plt.show()

def plot_surface(data, ke, target, target_labels, ax=None):
    # Draws the 3D surface for given data at fixed ke
    # Compute metrics for each fg, lens combination using multi-threading
    metrics = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_metrics_for_para, fg, lens, data[fg][lens]['global'] if 'global' in data[fg][lens] else data[fg][lens][ke]['global']): (fg, lens) for fg in data for lens in data[fg]}
        for future in as_completed(futures):
            key, mets = future.result()
            metrics[key] = mets

    # Get unique fg and lens
    fgs = sorted(set(fg for fg, _ in metrics))
    lens_VMIs = sorted(set(lens for _, lens in metrics))
    # Create grid
    fg_grid, lens_grid = np.meshgrid(fgs, lens_VMIs)
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.gca() if ax is None else ax
    # Collect points
    points = []
    values = []
    for (fg, lens), mets in metrics.items():
        val = mets[target]
        if not (np.isnan(val) or np.isinf(val)):  # Only include finite values
            points.append([fg, lens])
            values.append(val)
    if not points:
        print(f"No valid data points for target '{target}' at KE {ke}. Cannot create landscape plot.")
        return
        return
    points = np.array(points)
    values = np.array(values)
    # Check if we have enough data for a surface
    if len(np.unique(points[:, 0])) < 2 or len(np.unique(points[:, 1])) < 2:
        print(f"Insufficient unique parameter combinations for surface plot. Plotting as scatter only.")
        # Plot as scatter
        ax.scatter(points[:, 0], points[:, 1], values, c=values, cmap='viridis', s=50)
        ax.set_xlabel('Field Gradient')
        ax.set_ylabel('Lens VMI')
        ax.set_zlabel(target_labels[target])
        ax.set_title(f'Parameter Scatter: {target_labels[target]} at KE {ke}')
        ax.view_init(elev=20, azim=-60)
    else:
        # Interpolate
        Z = griddata(points, values, (fg_grid, lens_grid), method='nearest')
        Z = np.ma.masked_invalid(Z)
        # Plot surface
        surf = ax.plot_surface(fg_grid, lens_grid, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        # Also plot original points as scatter for visibility
        ax.scatter(points[:, 0], points[:, 1], values, color='red', s=20)
        ax.set_xlabel('Field Gradient')
        ax.set_ylabel('Lens VMI')
        ax.set_zlabel(target_labels[target])
        ax.set_title(f'Parameter Landscape: {target_labels[target]} at KE {ke}')
        ax.view_init(elev=20, azim=-60)
        fig = plt.gcf()
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) if len(fig.axes) == 1 else None
    if ax is None:
        plt.show()


def format_value(val: Any) -> str:
    """
    Format numerical values, handling special cases like infinity and NaN.
    
    This utility function converts numerical values to strings for display,
    properly handling special cases that would otherwise cause formatting errors.

    Args:
        val: The value to format (can be int, float, numpy scalar, or None)
        
    Returns:
        str: Formatted string representation:
             - 'N/A' for None, infinity, or NaN values
             - Formatted value with 4 decimal places for valid numbers
             
    Examples:
        >>> format_value(3.14159)
        '3.1416'
        >>> format_value(float('inf'))
        'N/A'
        >>> format_value(None)
        'N/A'
    """
    if val is None or np.isinf(val) or np.isnan(val):
        return 'N/A'
    return f'{val:.4f}'

def _3D_landscape(data, target='dr', fly2_file=None):
    """
    Draws a 3D surface plot showing the parameter landscape for a specified target function with respect to fg, lens_vmi, and ke.
    The surfaces are colored by ke value.

    Available targets: 'peak_y', 'peak_z', 'std_y', 'std_z', 'std_x_focus_y', 'std_x_focus_z', 'slope_y', 'slope_z', 'M_square', 'M_rectangle', 'dr'
    """
    valid_targets = ['peak_y', 'peak_z', 'std_y', 'std_z', 'std_x_focus_y', 'std_x_focus_z', 'slope_y', 'slope_z', 'M_square', 'M_rectangle', 'dr']
    if target not in valid_targets:
        raise ValueError(f"Invalid target. Choose from {valid_targets}")

    target_labels = {
        'peak_y': 'Y Axis Histogram Peak',
        'peak_z': 'Z Axis Histogram Peak',
        'std_y': 'Y Position Standard Deviation (mm)',
        'std_z': 'Z Position Standard Deviation (mm)',
        'std_x_focus_y': 'X Focus Position Std (Y criterion, mm)',
        'std_x_focus_z': 'X Focus Position Std (Z criterion, mm)',
        'slope_y': 'Petzval Slope (XZ plane)',
        'slope_z': 'Petzval Slope (XY plane)',
        'M_square': 'Magnification coefficient with square beam',
        'M_rectangle': 'Magnification coefficient with rectangle beam',
        'dr': 'Position spread dr (mm)'
    }

    # Collect all data points: fg, lens_VMI, ke, target_value
    all_fgs = []
    all_lens_vmils = []
    all_kes = []
    all_values = []

    def compute_metrics_for_fg_lens_ke(fg, lens_VMI, ke, global_data):
        mets = {}
        counts_y = global_data.get('counts_y')
        if counts_y is not None and len(counts_y) > 0:
            mets['peak_y'] = np.max(counts_y)
            mets['std_y'] = global_data.get('std_dev_y', float('inf'))
        else:
            mets['peak_y'] = 0
            mets['std_y'] = float('inf')
        counts_z = global_data.get('counts_z')
        if counts_z is not None and len(counts_z) > 0:
            mets['peak_z'] = np.max(counts_z)
            mets['std_z'] = global_data.get('std_dev_z', float('inf'))
        else:
            mets['peak_z'] = 0
            mets['std_z'] = float('inf')
        focus_points_y = global_data.get('focus_points_y', [])
        x_focus_y = np.array([item[0] for item in focus_points_y])
        mets['std_x_focus_y'] = np.std(x_focus_y) if len(x_focus_y) > 0 else float('inf')
        if len(x_focus_y) > 1:
            z2s = np.array([item[2]**2 for item in focus_points_y])
            coeffs_xz = np.polyfit(z2s, x_focus_y, 1)
            mets['slope_y'] = abs(coeffs_xz[0])
        else:
            mets['slope_y'] = float('inf')
        focus_points_z = global_data.get('focus_points_z', [])
        x_focus_z = np.array([item[0] for item in focus_points_z])
        mets['std_x_focus_z'] = np.std(x_focus_z) if len(x_focus_z) > 0 else float('inf')
        if len(x_focus_z) > 1:
            y2s = np.array([item[1]**2 for item in focus_points_z])
            coeffs_xy = np.polyfit(y2s, x_focus_z, 1)
            mets['slope_z'] = abs(coeffs_xy[0])
        else:
            mets['slope_z'] = float('inf')
        mets['M_square'] = global_data.get('M_square', float('inf'))
        mets['M_rectangle'] = global_data.get('M_rectangle', float('inf'))
        mets['dr'] = global_data.get('dr', float('inf'))
        return mets

    # Collect all unique combinations
    all_ke = set()
    for fg in data:
        for lens_VMI in data[fg]:
            for ke in data[fg][lens_VMI]:
                if 'global' in data[fg][lens_VMI][ke]:
                    all_fgs.append(fg)
                    all_lens_vmils.append(lens_VMI)
                    ke_float = float(ke)
                    all_kes.append(ke_float)
                    mets = compute_metrics_for_fg_lens_ke(fg, lens_VMI, ke, data[fg][lens_VMI][ke]['global'])
                    val = mets[target]
                    if not (np.isnan(val) or np.isinf(val)):
                        all_values.append(val)
                    else:
                        all_values.append(np.nan)
                # Since we added if global in, need to add ke outside if? No, only when global exists
                if 'global' in data[fg][lens_VMI][ke]:
                    all_ke.add(float(ke))

    if not all_fgs:
        print("No data to visualize.")
        return

    unique_fgs = sorted(set(all_fgs))
    unique_lens = sorted(set(all_lens_vmils))
    unique_kes = sorted(all_ke)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize ke for color
    ke_min = min(unique_kes)
    ke_max = max(unique_kes)
    colormap = plt.get_cmap('viridis')

    # For each ke, create a surface if possible
    for ke_idx, ke in enumerate(unique_kes):
        # Collect points for this ke
        points = []
        values = []
        for i, (fg, lens, k, val) in enumerate(zip(all_fgs, all_lens_vmils, all_kes, all_values)):
            if k == ke and not np.isnan(val):
                points.append([fg, lens])
                values.append(val)
        if len(points) < 4:  # Need at least 4 points for interpolation
            # Plot as scatter instead
            points_arr = np.array(points)
            if len(points_arr) > 0:
                rgba = colormap((ke - ke_min) / (ke_max - ke_min)) if ke_max != ke_min else colormap(0.5)
                ax.scatter(points_arr[:, 0], points_arr[:, 1], values, color=rgba, s=50, label=f'KE={ke}')
            continue
        points_arr = np.array(points)
        values_arr = np.array(values)

        if len(np.unique(points_arr[:, 0])) >= 2 and len(np.unique(points_arr[:, 1])) >= 2:
            fg_grid, lens_grid = np.meshgrid(unique_fgs, unique_lens)
            Z = griddata(points_arr, values_arr, (fg_grid, lens_grid), method='linear')
            Z_masked = np.ma.masked_invalid(Z)
            if not Z_masked.mask.all():
                rgba = colormap((ke - ke_min) / (ke_max - ke_min)) if ke_max != ke_min else colormap(0.5)
                ax.plot_surface(fg_grid, lens_grid, Z_masked, color=rgba, alpha=0.7, label=f'KE={ke}')

    ax.set_xlabel('Field Gradient')
    ax.set_ylabel('Lens VMI')
    ax.set_zlabel(target_labels[target])
    ax.set_title(f'3D Parameter Landscape: {target_labels[target]}')
    # Custom legend for KE
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()


def extract_aligned_points_for_all_pairs(data, x_range=(73.0, 166.0), x_step=1, enable_gc=True):
    """
    Calculate focus positions and widths by grouping trajectories based on emission angles.

    OPTIMIZED: Added periodic garbage collection for large datasets.

    Groups:
    1: emission angle 0,180 - trajectories 0,1
    2: emission angle 45,135 - 2,4
    3: emission angle 90 - 6
    4: emission angle -45,-135 - 3,5
    5: emission angle -90 - 7

    For each group, create x-slices, find minimum spread in y/z, and store focus data.
    
    Args:
        data: Main data dictionary
        x_range: Range of x-planes to analyze
        x_step: Step size for x-slices
        enable_gc: Enable periodic garbage collection (default: True)
    """

    X_SCAN_RANGE = x_range
    X_STEP = x_step

    x_min, x_max = X_SCAN_RANGE

    groups = {1: [0,1], 2: [2,4], 3: [6], 4: [3,5], 5: [7]}
    x_slices = np.arange(x_min, x_max + X_STEP, X_STEP)
    
    # Pre-compute bins once
    bins = np.concatenate([x_slices, [x_slices[-1] + x_step]])
    
    # Track progress for large datasets
    total_combinations = sum(1 for fg in data for lvmi in data[fg] for ke in data[fg][lvmi])
    processed = 0

    for fg in data:
        for lvmi in data[fg]:
            for ke in data[fg][lvmi]:
                local_data = data[fg][lvmi][ke].get('local', {})

                # Collect trajectories per group
                group_trajs = {g: [] for g in groups}
                for p_idx in local_data:
                    p_data = local_data[p_idx]
                    trajectories = p_data.get('trajectories', []) if isinstance(p_data, dict) else p_data
                    for g, indices in groups.items():
                        for idx in indices:
                            if idx < len(trajectories) and trajectories[idx]:
                                group_trajs[g].append(np.array(trajectories[idx]))

                # For each group, calculate focus for y and z
                focus_points_y = []
                focus_points_z = []

                for g, trajs in group_trajs.items():
                    if not trajs:
                        continue

                    # Concatenate all points from trajectories in this group
                    all_points = np.concatenate(trajs, axis=0)  # (N, 3)

                    # Assign points to x-slices
                    slice_indices = np.digitize(all_points[:, 0], bins) - 1

                    # Per slice, calculate ptp in y and z - vectorized approach
                    widths_y = np.full(len(x_slices), np.nan)
                    widths_z = np.full(len(x_slices), np.nan)
                    avg_ys = np.full(len(x_slices), np.nan)
                    avg_zs = np.full(len(x_slices), np.nan)

                    # Use numpy's bincount for faster aggregation where possible
                    for s_idx in range(len(x_slices)):
                        in_slice = slice_indices == s_idx
                        if np.sum(in_slice) > 1:
                            points_slice = all_points[in_slice]
                            y_vals = points_slice[:, 1]
                            z_vals = points_slice[:, 2]
                            widths_y[s_idx] = np.ptp(y_vals)
                            widths_z[s_idx] = np.ptp(z_vals)
                            avg_ys[s_idx] = np.mean(y_vals)
                            avg_zs[s_idx] = np.mean(z_vals)

                    # Find best slice for y (min width)
                    if not np.all(np.isnan(widths_y)):
                        min_idx_y = np.nanargmin(widths_y)
                        x_focus_y = x_slices[min_idx_y]
                        focus_center_y = avg_ys[min_idx_y]
                        focus_center_z = avg_zs[min_idx_y]
                        focus_width_y = widths_y[min_idx_y]
                        focus_points_y.append([x_focus_y, focus_center_y, focus_center_z, focus_width_y])

                    # Find best slice for z (min width)
                    if not np.all(np.isnan(widths_z)):
                        min_idx_z = np.nanargmin(widths_z)
                        x_focus_z = x_slices[min_idx_z]
                        focus_center_y_z = avg_ys[min_idx_z]
                        focus_center_z_z = avg_zs[min_idx_z]
                        focus_width_z = widths_z[min_idx_z]
                        focus_points_z.append([x_focus_z, focus_center_y_z, focus_center_z_z, focus_width_z])
                    
                    # Clear temporary arrays
                    del all_points, slice_indices, widths_y, widths_z, avg_ys, avg_zs

                # Store in global
                data[fg][lvmi][ke]['global']['focus_points_y'] = focus_points_y
                data[fg][lvmi][ke]['global']['focus_points_z'] = focus_points_z
                
                processed += 1
                
                # Periodic garbage collection
                if enable_gc and processed % 100 == 0:
                    gc.collect()


def find_closest_points_by_axis(traj1, traj2, x_range=(73.0, 166.0), axis='y', x_tolerance=0.25):
    """
    在 x ∈ [x_min, x_max] 条件下，找出两条轨迹曲线在投影平面上最接近的一对点，要求 x 位置相似 (在 x_tolerance 范围内)。

    对于 'y' 轴 focus: 投影到 XZ 平面，计算 min sqrt((x1-x2)^2 + (z1-z2)^2) 在 x 相似条件下
    对于 'z' 轴 focus: 投影到 XY 平面，计算 min sqrt((x1-x2)^2 + (y1-y2)^2) 在 x 相似条件下

    参数:
        traj1: list of [x, y, z]
        traj2: list of [x, y, z]
        x_range: (x_min, x_max)
        axis: 'y' or 'z' —— focus 类型
        x_tolerance: x 位置相似度阈值 (mm)

    返回:
        dict 包含 f_point1, f_point2, indices, min_dist
    """
    x_min, x_max = x_range
    traj1 = np.array(traj1)  # shape: (N1, 3)
    traj2 = np.array(traj2)  # shape: (N2, 3)

    # 筛选 x 在范围内的点
    mask1 = (traj1[:, 0] >= x_min) & (traj1[:, 0] <= x_max)
    mask2 = (traj2[:, 0] >= x_min) & (traj2[:, 0] <= x_max)

    filtered_traj1 = traj1[mask1]
    filtered_traj2 = traj2[mask2]

    indices1 = np.where(mask1)[0]
    indices2 = np.where(mask2)[0]

    if len(filtered_traj1) == 0 or len(filtered_traj2) == 0:
        return None

    # 获取 x 和相应的垂直坐标
    x1 = filtered_traj1[:, 0]
    x2 = filtered_traj2[:, 0]

    if axis == 'y':
        # Y focus: XZ 平面距离
        perp1 = filtered_traj1[:, 2]  # z
        perp2 = filtered_traj2[:, 2]
    elif axis == 'z':
        # Z focus: XY 平面距离
        perp1 = filtered_traj1[:, 1]  # y
        perp2 = filtered_traj2[:, 1]
    else:
        raise ValueError("axis must be 'y' or 'z'")

    # 计算限制在 x 相似的点对之间的最小距离 - 使用向量化的方式加速
    x1_mat = x1.reshape(-1, 1)  # N1 x 1
    perp1_mat = perp1.reshape(-1, 1)
    x2_mat = x2.reshape(1, -1)  # 1 x N2
    perp2_mat = perp2.reshape(1, -1)

    dx_mat = x1_mat - x2_mat  # N1 x N2
    perp_diff_mat = perp1_mat - perp2_mat
    dist_sq_mat = dx_mat**2 + perp_diff_mat**2

    mask = np.abs(dx_mat) <= x_tolerance
    valid_dist_sq = np.where(mask, dist_sq_mat, np.inf)

    min_dist_sq = np.nanmin(valid_dist_sq) if np.any(mask) else np.inf

    if np.isinf(min_dist_sq):
        return None  # 没有找到符合 x_tolerance 的点对

    # 找到最小距离的索引
    flat_idx = np.nanargmin(valid_dist_sq)
    best_i, best_j = np.unravel_index(flat_idx, valid_dist_sq.shape)

    min_dist = np.sqrt(min_dist_sq)

    return {
        'f_point1': filtered_traj1[best_i].tolist(),
        'f_point2': filtered_traj2[best_j].tolist(),
        'index1': int(indices1[best_i]),
        'index2': int(indices2[best_j]),
        'min_dist': min_dist,
        'axis': axis,
        'valid': True
    }


def aberration_estimation(
    data: Dict[float, Dict[float, Dict[float, Dict[str, Any]]]],
    theta: float = 0
) -> Dict[float, Dict[float, Dict[float, Dict[str, Any]]]]:
    """
    Estimate optical aberrations for each field gradient (fg), lens_VMI, and kinetic energy (ke).

    This function calculates various optical aberrations from SIMION simulation data:
    
    Aberrations computed:
    1. Field curvature: Petzval slope for XZ and XY planes (slope_y and slope_z)
       - Calculated by fitting x-focus positions vs z² and y²
       - Represents how the focal plane curves with off-axis distance
       
    2. Astigmatism: Difference in focus between orthogonal axes
       - Formula: |(std_x_y - std_x_z) / ((std_x_y + std_x_z) / 2)|
       - Where std_x_y/z are standard deviations of x-focus positions for y/z criteria
       
    3. Spherical aberration: Non-linear focusing error with aperture
       - Fits radial focus position vs elevation angle³: r = a + b * θ³
       - Where θ is emission elevation angle in radians, r is radial focus position
       
    4. Chromatic aberrations (computed per fg, lens_VMI across ke values):
       - Longitudinal: Peak-to-peak of focus plane intercepts over kinetic energy
       - Lateral: Peak-to-peak of focus positions over kinetic energy

    Args:
        data: Processed SIMION data dictionary with structure:
               data[fg][lens_VMI][ke]['global'][focus_points_y/z, etc.]
        theta: Rotation angle (in radians) around z-axis for emission angles.
               Adjusts the effective emission angles for aberration calculations.

    Returns:
        dict: Nested dictionary with aberration results:
               results[fg][lens_VMI][ke] = {
                   'field_curvature_xz': float,
                   'field_curvature_xy': float,
                   'intercept_xz': float,
                   'intercept_xy': float,
                   'astigmatism': float,
                   'spherical_aberration_a': float,
                   'spherical_aberration_b': float,
                   'chromatic_longitudinal': float,
                   'chromatic_lateral': {
                       'y_focus': [ptp_min_y, ptp_max_y, ptp_min_z, ptp_max_z],
                       'z_focus': [ptp_min_y, ptp_max_y, ptp_min_z, ptp_max_z]
                   }
               }
               
    Raises:
        ValueError: If input data is not a dictionary or is empty
        ValueError: If theta is not a numeric value
        
    Note:
        - Returns float('inf') for aberrations that cannot be calculated
        - Handles missing data gracefully with appropriate error messages
        - Chromatic aberrations are computed across all ke values for each fg,lens_VMI pair
    """
    # Input validation
    if not isinstance(data, dict) or not data:
        raise ValueError("Input data must be a non-empty dictionary")
    
    if not isinstance(theta, (int, float)):
        raise ValueError("Theta must be a numeric value")
    
    try:
        # Compute emission elevation angles for each group, adjusted by theta rotation
        # Based on group keys from data_viewer grouping, the elevation angles per trajectory
        original_els = [0, 0, 45, 45, 45, 45, 90, 90]  # Elevation angles for trajectories 0-7 in degrees

        # Adjust for rotation theta (convert theta to degrees for simplicity in fitting)
        theta_deg = np.degrees(theta)
        group_els = [(el - theta_deg) % 360 for el in original_els]  # Adjust elevation by rotation

        # Groups map trajectory indices to group
        groups = {1: [0,1], 2: [2,4], 3: [6], 4: [3,5], 5: [7]}

        group_angles = {}
        for group_idx, traj_indices in groups.items():
            thetas = [group_els[idx] for idx in traj_indices]
            group_angles[group_idx] = np.mean(thetas)  # Average theta for groups with multiple trajectories

        results = {}

        # First, compute chromatic aberrations across ke for each fg, lens_VMI
        chromatic_per_fg_lens = {}
        for fg in sorted(data.keys()):
            chromatic_per_fg_lens[fg] = {}
            for lens_VMI in sorted(data[fg].keys()):
                intercepts_xz = []
                intercepts_xy = []

                # For lateral: collect min/max y and z from focus points per ke, separately for y and z focus
                min_ys_yfocus = []
                max_ys_yfocus = []
                min_zs_yfocus = []
                max_zs_yfocus = []

                min_ys_zfocus = []
                max_ys_zfocus = []
                min_zs_zfocus = []
                max_zs_zfocus = []

                for ke in sorted(data[fg][lens_VMI].keys()):
                    global_data = data[fg][lens_VMI][ke].get('global', {})

                    focus_points_y_dict = global_data.get('focus_points_y', {})
                    focus_points_z_dict = global_data.get('focus_points_z', {})

                    # Collect intercepts from fits
                    if focus_points_y_dict:  # Has data
                        all_pts_y = []
                        if isinstance(focus_points_y_dict, dict):
                            for key, points in focus_points_y_dict.items():
                                all_pts_y.extend(points)
                        elif isinstance(focus_points_y_dict, list):
                            all_pts_y = focus_points_y_dict.copy()
                        if len(all_pts_y) > 1:
                            xs_y_vals = np.array([pt[0] for pt in all_pts_y])
                            z2s = np.array([pt[2]**2 for pt in all_pts_y])
                            if len(z2s) > 1:
                                coeffs_xz = np.polyfit(z2s, xs_y_vals, 1)
                                intercept_xz = coeffs_xz[1]
                                intercepts_xz.append(intercept_xz)

                    if focus_points_z_dict:
                        all_pts_z = []
                        if isinstance(focus_points_z_dict, dict):
                            for key, points in focus_points_z_dict.items():
                                all_pts_z.extend(points)
                        elif isinstance(focus_points_z_dict, list):
                            all_pts_z = focus_points_z_dict.copy()
                        if len(all_pts_z) > 1:
                            xs_z_vals = np.array([pt[0] for pt in all_pts_z])
                            y2s = np.array([pt[1]**2 for pt in all_pts_z])
                            if len(y2s) > 1:
                                coeffs_xy = np.polyfit(y2s, xs_z_vals, 1)
                                intercept_xy = coeffs_xy[1]
                                intercepts_xy.append(intercept_xy)

                    # For lateral with y-focus criterion: use focus_points_y
                    all_ys_yf = []
                    all_zs_yf = []
                    if focus_points_y_dict:
                        if isinstance(focus_points_y_dict, dict):
                            for key, points in focus_points_y_dict.items():
                                all_ys_yf.extend([pt[1] for pt in points])
                                all_zs_yf.extend([pt[2] for pt in points])
                        elif isinstance(focus_points_y_dict, list):
                            all_ys_yf.extend([pt[1] for pt in focus_points_y_dict])
                            all_zs_yf.extend([pt[2] for pt in focus_points_y_dict])
                    if all_ys_yf:
                        min_ys_yfocus.append(np.min(all_ys_yf))
                        max_ys_yfocus.append(np.max(all_ys_yf))
                    if all_zs_yf:
                        min_zs_yfocus.append(np.min(all_zs_yf))
                        max_zs_yfocus.append(np.max(all_zs_yf))

                    # For lateral with z-focus criterion: use focus_points_z
                    all_ys_zf = []
                    all_zs_zf = []
                    if focus_points_z_dict:
                        if isinstance(focus_points_z_dict, dict):
                            for key, points in focus_points_z_dict.items():
                                all_ys_zf.extend([pt[1] for pt in points])
                                all_zs_zf.extend([pt[2] for pt in points])
                        elif isinstance(focus_points_z_dict, list):
                            all_ys_zf.extend([pt[1] for pt in focus_points_z_dict])
                            all_zs_zf.extend([pt[2] for pt in focus_points_z_dict])
                    if all_ys_zf:
                        min_ys_zfocus.append(np.min(all_ys_zf))
                        max_ys_zfocus.append(np.max(all_ys_zf))
                    if all_zs_zf:
                        min_zs_zfocus.append(np.min(all_zs_zf))
                        max_zs_zfocus.append(np.max(all_zs_zf))

                # Longitudinal chromatic: ptp of intercepts
                long_ca_xz = np.ptp(intercepts_xz) if intercepts_xz else float('inf')
                long_ca_xy = np.ptp(intercepts_xy) if intercepts_xy else float('inf')
                long_ca = long_ca_xz if long_ca_xz != float('inf') else long_ca_xy  # Prefer XZ, else XY

                # Lateral chromatic: ptp of each min/max direction per focus criterion
                lat_ca_yf = [np.ptp(min_ys_yfocus) if min_ys_yfocus else float('inf'),
                             np.ptp(max_ys_yfocus) if max_ys_yfocus else float('inf'),
                             np.ptp(min_zs_yfocus) if min_zs_yfocus else float('inf'),
                             np.ptp(max_zs_yfocus) if max_zs_yfocus else float('inf')]
                lat_ca_zf = [np.ptp(min_ys_zfocus) if min_ys_zfocus else float('inf'),
                             np.ptp(max_ys_zfocus) if max_ys_zfocus else float('inf'),
                             np.ptp(min_zs_zfocus) if min_zs_zfocus else float('inf'),
                             np.ptp(max_zs_zfocus) if max_zs_zfocus else float('inf')]

                chromatic_per_fg_lens[fg][lens_VMI] = {
                    'longitudinal_chromatic_xz': long_ca_xz,
                    'longitudinal_chromatic_xy': long_ca_xy,
                    'longitudinal_chromatic': long_ca,
                    'lateral_chromatic_y_focus': lat_ca_yf,  # [ptp_min_y, ptp_max_y, ptp_min_z, ptp_max_z] for y-focus criterion
                    'lateral_chromatic_z_focus': lat_ca_zf   # same for z-focus criterion
                }

        # Now, compute per ke aberrations
        for fg in sorted(data.keys()):
            results[fg] = {}
            for lens_VMI in sorted(data[fg].keys()):
                results[fg][lens_VMI] = {}
                for ke in sorted(data[fg][lens_VMI].keys()):
                    global_data = data[fg][lens_VMI][ke].get('global', {})

                    # Astigmatism: std of x-focus positions for y and z criteria
                    focus_points_y = global_data.get('focus_points_y', [])
                    x_positions_y = np.array([pt[0] for pt in focus_points_y]) if focus_points_y else np.array([])
                    std_x_y = np.std(x_positions_y) if len(x_positions_y) > 0 else float('inf')

                    focus_points_z = global_data.get('focus_points_z', [])
                    x_positions_z = np.array([pt[0] for pt in focus_points_z]) if focus_points_z else np.array([])
                    std_x_z = np.std(x_positions_z) if len(x_positions_z) > 0 else float('inf')

                    # Field curvature: Compute Petzval slopes directly
                    slope_y = float('inf')
                    slope_z = float('inf')
                    intercept_xz = float('inf')
                    intercept_xy = float('inf')

                    if focus_points_y and len(focus_points_y) > 1:
                        xs_y_vals = np.array([pt[0] for pt in focus_points_y])
                        z2s = np.array([pt[2]**2 for pt in focus_points_y])
                        if len(z2s) > 1:
                            try:
                                coeffs_xz = np.polyfit(z2s, xs_y_vals, 1)
                                slope_y = abs(coeffs_xz[0])
                                intercept_xz = coeffs_xz[1]
                            except (ValueError, np.linalg.LinAlgError):
                                # Handle case where polyfit fails
                                pass

                    if focus_points_z and len(focus_points_z) > 1:
                        xs_z_vals = np.array([pt[0] for pt in focus_points_z])
                        y2s = np.array([pt[1]**2 for pt in focus_points_z])
                        if len(y2s) > 1:
                            try:
                                coeffs_xy = np.polyfit(y2s, xs_z_vals, 1)
                                slope_z = abs(coeffs_xy[0])
                                intercept_xy = coeffs_xy[1]
                            except (ValueError, np.linalg.LinAlgError):
                                # Handle case where polyfit fails
                                pass

                    astigmatism = np.abs((std_x_y - std_x_z) / ((std_x_y + std_x_z) / 2)) if std_x_y != float('inf') and std_x_z != float('inf') and (std_x_y + std_x_z) != 0 else float('inf')

                    # Spherical aberration: Fit r = a + b * theta**3
                    spherical_a = None
                    spherical_b = None
                    if group_angles and len(focus_points_y) > 2:  # Need at least 3 points for fitting
                        # Use y-focus points for radial calculation
                        r_values = []
                        theta_values = []
                        groups_present = []
                        for group_idx in range(1, 6):  # Groups 1-5
                            if group_idx in group_angles:
                                # Find corresponding focus point (assuming order matches groups)
                                if group_idx - 1 < len(focus_points_y):
                                    focus_pt = focus_points_y[group_idx - 1]
                                    # r as distance from origin in yz plane: sqrt(y^2 + z^2)
                                    r = np.sqrt(focus_pt[1]**2 + focus_pt[2]**2)  # y, z coordinates
                                    r_values.append(r)
                                    # Convert to radians for proper spherical aberration calculation
                                    theta_rad = np.radians(group_angles[group_idx])
                                    theta_values.append(theta_rad**3)  # theta**3 in radians
                                    groups_present.append(group_idx)

                        if len(theta_values) > 2:
                            try:
                                # Fit r = a + b * theta**3 where theta_values already contain theta**3
                                popt, _ = curve_fit(lambda t, a, b: a + b * t, theta_values, r_values)
                                spherical_a, spherical_b = popt
                            except Exception as e:
                                print(f"Warning: Spherical aberration fit failed for fg={fg}, lens_VMI={lens_VMI}, ke={ke}: {e}")

                    # Get chromatic aberrations for this fg, lens_VMI
                    chromatic_data = chromatic_per_fg_lens.get(fg, {}).get(lens_VMI, {})
                    longitudinal_chromatic = chromatic_data.get('longitudinal_chromatic', float('inf'))
                    lateral_chromatic_y = chromatic_data.get('lateral_chromatic_y_focus', [float('inf')] * 4)
                    lateral_chromatic_z = chromatic_data.get('lateral_chromatic_z_focus', [float('inf')] * 4)

                    results[fg][lens_VMI][ke] = {
                        'field_curvature_xz': slope_y,
                        'field_curvature_xy': slope_z,
                        'intercept_xz': intercept_xz,
                        'intercept_xy': intercept_xy,
                        'astigmatism': astigmatism,
                        'spherical_aberration_a': spherical_a,
                        'spherical_aberration_b': spherical_b,
                        'chromatic_longitudinal': longitudinal_chromatic,
                        'chromatic_lateral': {
                            'y_focus': lateral_chromatic_y,
                            'z_focus': lateral_chromatic_z
                        }
                    }

        return results
    
    except Exception as e:
        print(f"Error in aberration_estimation: {e}")
        traceback.print_exc()
        return {}


def print_aberration_summary(aberration_results: Dict[float, Dict[float, Dict[float, Dict[str, Any]]]]) -> None:
    """
    Print a formatted summary of aberration results from the aberration_estimation function.

    This function displays the calculated aberrations in an organized format,
    showing field curvature, astigmatism, and spherical aberration parameters
    for each combination of field gradient, lens VMI, and kinetic energy.

    Args:
        aberration_results: Nested dictionary output from aberration_estimation function
                          with structure: results[fg][lens_VMI][ke][aberration_type]
                          
    Returns:
        None
        
    Note:
        - Prints 'N/A' for aberration values that could not be calculated
        - Formats spherical aberration as "r = a + b * theta^3" when parameters are available
        - Uses the format_value function to handle infinite and NaN values
    """
    print("=== Aberration Estimation Summary ===\n")

    for fg in sorted(aberration_results.keys()):
        print(f"Field Gradient: {fg}")
        for lens_VMI in sorted(aberration_results[fg].keys()):
            print(f"  Lens VMI: {lens_VMI}")
            for ke in sorted(aberration_results[fg][lens_VMI].keys()):
                results = aberration_results[fg][lens_VMI][ke]
                print(f"    KE: {ke}")
                print(f"      Field Curvature XZ: {format_value(results['field_curvature_xz'])}")
                print(f"      Field Curvature XY: {format_value(results['field_curvature_xy'])}")
                print(f"      Astigmatism: {format_value(results['astigmatism'])}")
                a = results['spherical_aberration_a']
                b = results['spherical_aberration_b']
                if a is not None and b is not None:
                    print(f"      Spherical Aberration: r = {a:.6f} + {b:.6f} * theta^3")
                else:
                    print("      Spherical Aberration: Unable to fit")
        print()


def focus_filtering(processed_data, tolerable_offset=2.0):
    """
    Filter focus points data based on Petzval field curvature fitting.
    
    This function works like fitting the focus points data similar to Petzval field curvature.
    It checks the intercept values and finds indices of processed_data where |intercept - 73| <= tolerable_offset.
    
    Args:
        processed_data: Dictionary containing processed SIMION data with structure:
                      processed_data[fg][lens_VMI][ke]['global'][focus_points_y/z]
        tolerable_offset: Maximum allowed deviation from 73mm for intercept (default: 2.0)
    
    Returns:
        list: List of tuples (fg, lens_VMI, ke) where both XZ and XY planes have intercepts
              within the specified tolerance of 73mm.
    """
    valid_combinations = []
    
    for fg in processed_data:
        for lens_VMI in processed_data[fg]:
            for ke in processed_data[fg][lens_VMI]:
                global_data = processed_data[fg][lens_VMI][ke].get('global', {})
                
                # Initialize validity flags
                valid_xz = False
                valid_xy = False
                
                # Process XZ plane (y focus criterion)
                focus_points_y = global_data.get('focus_points_y', [])
                if True:
                    #focus_points_y and len(focus_points_y) > 1:
                    # Extract x positions and z coordinates for Petzval fitting
                    # xs_y = np.array([item[0] for item in focus_points_y])
                    # zs = np.array([item[2] for item in focus_points_y])
                    # z2s = zs ** 2
                    
                    try:
                        # Fit Petzval field curvature: X = intercept + slope * z²
                        # coeffs_xz = np.polyfit(z2s, xs_y, 1)
                        # intercept_xz = coeffs_xz[1]
                        
                        # Check if intercept is within tolerance
                        if True:
                            #abs(intercept_xz - 73) <= tolerable_offset:
                            valid_xz = True
                    except (ValueError, np.linalg.LinAlgError):
                        # Handle fitting errors
                        pass
                
                # Process XY plane (z focus criterion)
                focus_points_z = global_data.get('focus_points_z', [])
                if True:
                    #focus_points_z and len(focus_points_z) > 1:
                    # Extract x positions and y coordinates for Petzval fitting
                    # xs_z = np.array([item[0] for item in focus_points_z])
                    # ys = np.array([item[1] for item in focus_points_z])
                    # y2s = ys ** 2
                    
                    try:
                        # Fit Petzval field curvature: X = intercept + slope * y²
                        # coeffs_xy = np.polyfit(y2s, xs_z, 1)
                        # intercept_xy = coeffs_xy[1]
                        
                        # Check if intercept is within tolerance
                        if True:
                            #abs(intercept_xy - 73) <= tolerable_offset:
                            valid_xy = True
                    except (ValueError, np.linalg.LinAlgError):
                        # Handle fitting errors
                        pass
                
                # Only add to results if BOTH planes are valid
                if valid_xz and valid_xy:
                    valid_combinations.append((fg, lens_VMI, ke))
    
    return valid_combinations


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


def _extract_final_positions(single_processed_data, fg, lens_vmi, ke):
    """
    Helper function to extract final positions from processed simulation data.
    
    Args:
        single_processed_data: Processed simulation data
        fg: Field gradient
        lens_vmi: Lens VMI value
        ke: Kinetic energy
    
    Returns:
        tuple: (final_positions, success_flag)
    """
    final_positions = []
    
    # Check if the specific fg, lens_vmi, ke combination exists in the processed data
    if fg in single_processed_data and lens_vmi in single_processed_data[fg] and ke in single_processed_data[fg][lens_vmi]:
        local_data = single_processed_data[fg][lens_vmi][ke].get('local', {})
        
        # Suppressed particle processing output
        # print(f"  Processing {len(local_data)} particles for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}")
        
        # Only collect final positions from the specific combination we're analyzing
        for particle_idx in local_data:
            trajectories = local_data[particle_idx].get('trajectories', [])
            for trajectory in trajectories:
                if trajectory:  # Check if trajectory is not empty
                    # Get the final point (last point in the trajectory)
                    final_point = trajectory[-1]
                    y, z = final_point[1], final_point[2]  # Extract y and z coordinates
                    final_positions.append([y, z])
                    
        # Suppressed positions collection output
        # print(f"  Collected {len(final_positions)} final positions for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}")
        return final_positions, True
    else:
        print(f"  ERROR: No data found for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}")
        return [], False


def _center_positions(final_positions):
    """
    Helper function to center the image by subtracting the mean position.
    
    Args:
        final_positions: Array of y,z positions
    
    Returns:
        numpy.array: Centered positions
    """
    if not final_positions:
        return np.array([])
    
    # Convert to numpy array
    final_positions = np.array(final_positions)
    
    # Calculate the center (mean) of y,z positions
    center_y = np.mean(final_positions[:, 0])
    center_z = np.mean(final_positions[:, 1])
    
    # Apply offset to center the image at (0,0)
    final_positions[:, 0] = final_positions[:, 0] - center_y
    final_positions[:, 1] = final_positions[:, 1] - center_z
    
    # Suppressed centering output
    # print(f"  Image centering applied: Y offset = {-center_y:.4f} mm, Z offset = {-center_z:.4f} mm")
    return final_positions


def _estimate_fwhm_and_resolution(rect_binned, bin_interval, fg, lens_vmi, ke):
    """
    Helper function to estimate FWHM and energy resolution from binned data.
    
    Args:
        rect_binned: Binned data
        bin_interval: Bin size
        fg: Field gradient
        lens_vmi: Lens VMI value
        ke: Kinetic energy
    
    Returns:
        dict: FWHM and energy resolution results
    """
    import abel
    
    # Make sure we have valid data for Abel inversion
    if np.sum(rect_binned) == 0:
        # Suppressed warning output
        # print(f"  WARNING: No particles in binned data for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}")
        return None

    # Use correct Abel inversion method for rectangular coordinates
    try:
        recon, distr = abel.rbasex.rbasex_transform(rect_binned, direction='backward')
        r, I, beta = distr.rIbeta()

        # Find peak position and ensure it's at the center
        peak_idx = np.argmax(I)
        peak_r = r[peak_idx]
        
        # Check if peak is close to center (r=0)
       
        # Estimate FWHM of the intensity distribution I
        half_max = np.max(I) / 2
        # Find indices where intensity crosses half maximum
        indices_above_half = np.where(I >= half_max)[0]
        if len(indices_above_half) > 0:
            fwhm_indices = indices_above_half[-1] - indices_above_half[0]
            fwhm_r = r[indices_above_half[-1]] - r[indices_above_half[0]]
            fwhm_value = fwhm_r
            
            # Calculate energy resolution using the peak position (should be near r=0)
            peak_position = r[peak_idx]
            if peak_position != 0:
                energy_resolution = (fwhm_r / peak_position) * 100
            else:
                # If peak is exactly at 0, use the first non-zero r value
                non_zero_indices = np.where(r > 0)[0]
                if len(non_zero_indices) > 0:
                    energy_resolution = (fwhm_r / r[non_zero_indices[0]]) * 100
                else:
                    energy_resolution = 0
            
            # Suppressed FWHM and resolution output
            print(f"  FWHM: {fwhm_value:.4f} mm, Energy Resolution: {energy_resolution:.2f}%")
            
            return {
                'fwhm': fwhm_value,
                'energy_resolution': energy_resolution,
                'max_r': r[np.where(I == np.max(I))[0][0]],
                'rect_binned': rect_binned,
                'recon': recon,
                'r_data': r,  # Add r data for r,I plot
                'I_data': I   # Add I data for r,I plot
            }
        else:
            # Suppressed FWHM estimation warning
            # print(f"  WARNING: Could not estimate FWHM for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}")
            return None
    except Exception as e:
        # Suppressed Abel inversion error output
        # print(f"  ERROR: Abel inversion failed for FG {fg}, Lens {lens_vmi}, KE {ke:.2f}: {e}")
        return None


def energy_resolution_analysis(processed_data, tolerable_offset=2.0,
                              source_position=(199, -1, 0.0),
                              num_particles_per_energy=10000,
                              x_scan_range=(73.0, 166.0),
                              bin_interval=0.05,
                              outside_region_width=2,
                              batch_size=50,
                              enable_memory_optimization=True,
                              checkpoint_interval=25):
    """
    Perform energy resolution analysis for filtered parameter combinations.
    
    OPTIMIZED VERSION: Includes batch processing and memory management for large datasets.
    
    This function follows a 5-step logical flow:
    1. For given fg, lens_vmi, ke, get the final position of all trajectories and store the data
    2. For given fg, lens_vmi, ke, perform binning
    3. For given fg, lens_vmi, ke, perform Abel inversion
    4. For given fg, lens_vmi, ke, estimate the FWHM and energy resolution
    5. Store the data to processed_data and return it
    
    Args:
        processed_data: Dictionary containing processed SIMION data from initial workflow
        tolerable_offset: Maximum allowed deviation from 73mm for intercept (default: 2.0)
        source_position: Fixed position for particle source (default: (199, -1, 0.0))
        num_particles_per_energy: Number of particles per energy point (default: 10000)
        x_scan_range: Range of x-planes to scan for focus analysis (default: (73.0, 166.0))
        bin_interval: Bin size for rectangular coordinates (default: 0.05 mm)
        outside_region_width: Width of region outside data area (default: 2 mm)
        batch_size: Number of combinations to process per batch (default: 50)
        enable_memory_optimization: Enable aggressive memory cleanup (default: True)
        checkpoint_interval: Save checkpoints every N combinations (default: 25, 0 to disable)
    
    Returns:
        dict: Updated processed_data with FWHM results in global section
    """
    import numpy as np
    
    # Step 0: focus_filtering - find out all fg,lens_vmi, ke for filtered parameters
    print("Step 0: Filtering parameter combinations based on focus criteria...")
    filtered_combinations = focus_filtering(processed_data, tolerable_offset)
    
    if not filtered_combinations:
        print("No valid combinations found for energy resolution analysis!")
        return processed_data
    
    total_combinations = len(filtered_combinations)
    print(f"Found {total_combinations} valid combinations for energy resolution analysis")
    
    # Calculate number of batches
    num_batches = (total_combinations + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of up to {batch_size} combinations each")
    
    if enable_memory_optimization:
        initial_memory = get_memory_usage_mb()
        if initial_memory > 0:
            print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Initialize dictionary to store FWHM results
    fwhm_results = {}
    
    # Files for energy resolution analysis
    OUTPUT_FILENAME_FLY2 = 'WORKING_TITLE_energy_resolution_tao.fly2'
    OUTPUT_FILENAME_LUA = "WORKING_TITLE_energy_resolution_tao.lua"
    IOB_FILE = "WORKING_TITLE_energy_resolution_tao.iob"
    OUT_FILE = "energy_resolution_out.txt"
    
    # Track progress
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    checkpoint_files = []
    
    # Process in batches
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_combinations = filtered_combinations[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_combinations)} combinations)")
        print(f"Progress: {processed_count}/{total_combinations} ({100*processed_count/total_combinations:.1f}%)")
        
        if enable_memory_optimization:
            current_memory = get_memory_usage_mb()
            if current_memory > 0:
                print(f"Current memory usage: {current_memory:.1f} MB")
        
        batch_start_time = time.time()
        
        for combo_idx, (fg, lens_vmi, ke) in enumerate(batch_combinations):
            combo_num = batch_start + combo_idx + 1
            
            try:
                # Step 1: Generate corresponding fly2 file
                print(f"  [{combo_num}/{total_combinations}] Processing FG {fg}, Lens {lens_vmi}, KE {ke:.2f} eV...")
                energy_resolution_utilis(
                    filename=OUTPUT_FILENAME_FLY2,
                    position=source_position,
                    num_particles=num_particles_per_energy,
                    ke=ke
                )
                
                # Step 2: Generate corresponding lua file based on filtered fg, lens_vmi, ke
                single_param = get_parameters_for_combination(processed_data, fg, lens_vmi, ke)
                
                # Clear output file for this specific run
                clear_file_contents(OUT_FILE)
                
                # Step 3: Run simion
                run_optimized_simulations_with_ke(single_param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE)
                
                # Process simulation data for this combination
                single_processed_data = process_data(
                    x_range=x_scan_range,
                    file_path=OUT_FILE,
                    focus_axis='z',
                    fly2_file=OUTPUT_FILENAME_FLY2,
                    y_range=(-0.5, 0.5),
                    z_range=(-0.5, 0.5)
                )
                
                if single_processed_data is None:
                    print(f"    ERROR: Data processing failed")
                    failed_count += 1
                    continue
                
                # Step 4: Extract final positions
                final_positions, success = _extract_final_positions(single_processed_data, fg, lens_vmi, ke)
                
                if not success or not final_positions:
                    print(f"    WARNING: No final positions found")
                    failed_count += 1
                    continue
                
                # Center the positions
                final_positions = _center_positions(final_positions)
                
                # Step 5: Apply rectangular binning
                rect_binned, rect_edges, rect_centers = bin_positions(
                    final_positions,
                    bin_type='rectangular',
                    bin_interval=bin_interval,
                    outside_region_width=outside_region_width
                )
                
                # Step 6: Perform Abel inversion and estimate FWHM
                fwhm_result = _estimate_fwhm_and_resolution(rect_binned, bin_interval, fg, lens_vmi, ke)
                
                if fwhm_result is None:
                    failed_count += 1
                    continue
                
                # Store result
                if fg not in fwhm_results:
                    fwhm_results[fg] = {}
                if lens_vmi not in fwhm_results[fg]:
                    fwhm_results[fg][lens_vmi] = {}
                
                fwhm_results[fg][lens_vmi][ke] = fwhm_result
                processed_count += 1
                
                # Clean up single_processed_data to free memory
                del single_processed_data
                del final_positions
                del rect_binned
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                failed_count += 1
                continue
            
            # Save checkpoint if interval reached
            if checkpoint_interval > 0 and combo_num % checkpoint_interval == 0:
                checkpoint_file = save_checkpoint(fwhm_results, 'energy_resolution', combo_num)
                if checkpoint_file:
                    checkpoint_files.append(checkpoint_file)
        
        # End of batch - cleanup
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx + 1} completed in {batch_time:.1f}s")
        
        if enable_memory_optimization:
            cleanup_memory(force=True)
            current_memory = get_memory_usage_mb()
            if current_memory > 0:
                print(f"Memory after cleanup: {current_memory:.1f} MB")
    
    # Store FWHM results in processed_data global section
    print(f"\nStoring results in processed_data...")
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
                
                # Store FWHM results in global section
                processed_data[fg][lens_vmi][ke]['global'].update(fwhm_results[fg][lens_vmi][ke])
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Energy resolution analysis completed!")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if processed_count > 0:
        print(f"  Average time per combination: {total_time/processed_count:.2f}s")
    
    # Cleanup checkpoint files
    if checkpoint_files:
        print(f"  Cleaning up {len(checkpoint_files)} checkpoint files...")
        for cf in checkpoint_files:
            try:
                os.remove(cf)
            except:
                pass
    
    return processed_data


def bin_positions(positions, bin_type='rectangular', bin_interval=0.1, r_bins=None, theta_bins=None,
                  outside_region_width=2.0):
    """
    Bin positions in either rectangular or polar coordinate system.
    
    Args:
        positions: numpy array of shape (n, 2) containing y,z positions
        bin_type: 'rectangular' or 'polar' coordinate system
        bin_interval: bin size for rectangular coordinates (mm) or radial interval for polar (mm)
        r_bins: number of radial bins for polar coordinates (overrides bin_interval if provided)
        theta_bins: number of angular bins for polar coordinates (degrees)
        outside_region_width: width of region outside data area to include (mm)
    
    Returns:
        binned_data: 2D histogram of binned data
        bin_edges: tuple containing bin edges for each dimension
        bin_centers: tuple containing bin centers for each dimension
    """
    if len(positions) == 0:
        return np.array([]), (), ()
    
    y_pos = positions[:, 0]
    z_pos = positions[:, 1]
    
    if bin_type == 'rectangular':
        # Create rectangular bins
        y_min, y_max = y_pos.min(), y_pos.max()
        z_min, z_max = z_pos.min(), z_pos.max()
        
        # Extend range to include outside region
        y_range = (y_min - bin_interval/2 - outside_region_width,
                   y_max + bin_interval/2 + outside_region_width)
        z_range = (z_min - bin_interval/2 - outside_region_width,
                   z_max + bin_interval/2 + outside_region_width)
        
        # Calculate number of bins including outside region
        total_y_range = y_max - y_min + 2 * outside_region_width
        total_z_range = z_max - z_min + 2 * outside_region_width
        y_bins = int(np.ceil(total_y_range / bin_interval)) + 1
        z_bins = int(np.ceil(total_z_range / bin_interval)) + 1
        
        # Create 2D histogram
        binned_data, y_edges, z_edges = np.histogram2d(
            y_pos, z_pos, bins=[y_bins, z_bins], range=[y_range, z_range]
        )
        
        # Calculate bin centers
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        
        return binned_data, (y_edges, z_edges), (y_centers, z_centers)
    
    elif bin_type == 'polar':
        # Convert to polar coordinates
        r = np.sqrt(y_pos**2 + z_pos**2)
        theta = np.arctan2(z_pos, y_pos)  # Angle from y-axis to z-axis
        
        # Convert theta to degrees for easier interpretation
        theta_deg = np.degrees(theta)
        # Adjust to [0, 360) range
        theta_deg = (theta_deg + 360) % 360
        
        # Set up bins with outside region
        r_max = r.max()
        
        if r_bins is None:
            r_bins = int(np.ceil((r_max + outside_region_width) / bin_interval)) + 1
        
        if theta_bins is None:
            theta_bins = 36  # Default: 10-degree bins
        
        # Create bins including outside region
        r_edges = np.linspace(0, r_max + outside_region_width + bin_interval/2, r_bins + 1)
        theta_edges = np.linspace(0, 360, theta_bins + 1)
        
        # Create 2D histogram with theta as first dimension (rows) and r as second (columns)
        # This is important for proper visualization with imshow
        binned_data, _, _ = np.histogram2d(
            theta_deg, r, bins=[theta_edges, r_edges]
        )
        
        # Calculate bin centers
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        
        # No transpose needed - we already created histogram with theta as rows and r as columns
        return binned_data, (r_edges, theta_edges), (r_centers, theta_centers)
    
    else:
        raise ValueError("bin_type must be either 'rectangular' or 'polar'")


def heatmap_energy_lens(processed_data, fg):
    """
    Generate heatmap data for energy resolution vs lens VMI and kinetic energy for given field gradient.
    
    Args:
        processed_data: Dictionary containing processed SIMION data
        fg: Field gradient to analyze
    
    Returns:
        tuple: (heatmap_data, lens_values, ke_values)
            - heatmap_data: 2D numpy array with shape (num_lens, num_ke)
            - lens_values: sorted list of lens VMI values
            - ke_values: sorted list of kinetic energy values
    """
    if fg not in processed_data:
        print(f"Error: Field gradient {fg} not found in processed_data")
        return None, None, None
    
    # Get all unique lens VMI values and ke values for this fg
    lens_values = sorted(processed_data[fg].keys())
    
    # Collect all unique ke values across all lenses
    all_ke_values = set()
    for lens in lens_values:
        for ke in processed_data[fg][lens].keys():
            all_ke_values.add(ke)
    ke_values = sorted(all_ke_values)
    
    if not lens_values or not ke_values:
        print(f"Error: No lens or ke values found for fg={fg}")
        return None, None, None
    
    # Initialize heatmap with NaN (rows = lens, columns = ke)
    heatmap_data = np.full((len(lens_values), len(ke_values)), np.nan)
    
    # Fill in the heatmap data
    for row_idx, lens in enumerate(lens_values):
        for col_idx, ke in enumerate(ke_values):
            if ke in processed_data[fg][lens]:
                global_data = processed_data[fg][lens][ke].get('global', {})
                energy_resolution = global_data.get('energy_resolution', None)
                if energy_resolution is not None:
                    heatmap_data[row_idx, col_idx] = energy_resolution
    
    return heatmap_data, lens_values, ke_values


def plot_heatmap_energy_lens(processed_data, fg, cmap='viridis', figsize=(12, 8),
                              vmin=None, vmax=None, show_values=True):
    """
    Plot a heatmap of energy resolution vs lens VMI and kinetic energy for given field gradient.
    
    Args:
        processed_data: Dictionary containing processed SIMION data
        fg: Field gradient to analyze
        cmap: Colormap to use (default: 'viridis')
        figsize: Figure size as tuple (width, height) in inches
        vmin: Minimum value for colorbar (default: auto)
        vmax: Maximum value for colorbar (default: auto)
        show_values: Whether to annotate cells with values (default: True)
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Get heatmap data
    heatmap_data, lens_values, ke_values = heatmap_energy_lens(processed_data, fg)
    
    if heatmap_data is None:
        print("Cannot create heatmap: no data available")
        return None, None
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using imshow
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(ke_values)))
    ax.set_yticks(np.arange(len(lens_values)))
    ax.set_xticklabels([f'{ke:.1f}' for ke in ke_values])
    ax.set_yticklabels([f'{lens:.3f}' for lens in lens_values])
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add labels and title
    ax.set_xlabel('Kinetic Energy (eV)')
    ax.set_ylabel('Lens VMI')
    ax.set_title(f'Energy Resolution (%) vs Lens VMI and Kinetic Energy\nField Gradient: {fg}')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Energy Resolution (%)')
    
    # Annotate cells with values if requested
    if show_values:
        for i in range(len(lens_values)):
            for j in range(len(ke_values)):
                value = heatmap_data[i, j]
                if not np.isnan(value):
                    # Choose text color based on background
                    text_color = 'white' if value > (np.nanmax(heatmap_data) + np.nanmin(heatmap_data)) / 2 else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


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
        vmax = global_max if global_max != float('-inf') else 100
    
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
    ax.set_title(f'Energy Resolution (%) - Field Gradient: {initial_fg}')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Energy Resolution (%)')
    
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
            ax.set_title(f'Energy Resolution (%) - Field Gradient: {fg}')
            
            fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


def store_all_heatmaps(processed_data, filename='all_heatmaps.pkl', batch_size=10):
    """
    Store all heatmap data for all field gradients to a pickle file.
    
    OPTIMIZED: Process heatmaps in batches to reduce memory usage for large datasets.
    
    This function pre-computes heatmap data for all field gradients in the processed_data
    and stores them in a dictionary that can be saved to disk for later visualization.
    
    Args:
        processed_data: Dictionary containing processed SIMION data with structure:
                       processed_data[fg][lens_VMI][ke]['global']['energy_resolution']
        filename: Output pickle file name (default: 'all_heatmaps.pkl')
        batch_size: Number of field gradients to process per batch (default: 10)
    
    Returns:
        dict: Dictionary containing all heatmap data with structure:
              {fg: (heatmap_data, lens_values, ke_values), ...}
    """
    import pickle
    
    fg_keys = sorted(processed_data.keys())
    
    if not fg_keys:
        print("No field gradient data available")
        return {}
    
    print(f"Processing {len(fg_keys)} field gradients for heatmap storage...")
    
    # Pre-compute all heatmap data in batches
    all_heatmaps = {}
    
    num_batches = (len(fg_keys) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(fg_keys))
        batch_fgs = fg_keys[batch_start:batch_end]
        
        for fg in batch_fgs:
            heatmap_data, lens_values, ke_values = heatmap_energy_lens(processed_data, fg)
            if heatmap_data is not None:
                all_heatmaps[fg] = {
                    'heatmap_data': heatmap_data,
                    'lens_values': lens_values,
                    'ke_values': ke_values
                }
        
        # Cleanup after each batch
        gc.collect()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Processed {batch_end}/{len(fg_keys)} field gradients...")
    
    # Save to pickle file
    if filename:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(all_heatmaps, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Heatmap data saved to '{filename}'")
        except Exception as e:
            print(f"Error saving heatmap data: {e}")
    
    return all_heatmaps
    
    return all_heatmaps


def plot_stored_heatmaps(all_heatmaps, cmap='viridis', figsize=(14, 10),
                          vmin=None, vmax=None):
    """
    Plot heatmaps from pre-stored heatmap data with a slider to switch between field gradients.
    
    Args:
        all_heatmaps: Dictionary containing heatmap data from store_all_heatmaps()
                     Structure: {fg: {'heatmap_data': array, 'lens_values': list, 'ke_values': list}, ...}
        cmap: Colormap to use (default: 'viridis')
        figsize: Figure size as tuple (width, height) in inches
        vmin: Minimum value for colorbar (default: auto across all data)
        vmax: Maximum value for colorbar (default: auto across all data)
    
    Returns:
        None (displays interactive plot)
    """
    if not all_heatmaps:
        print("No heatmap data available")
        return
    
    fg_keys = sorted(all_heatmaps.keys())
    
    # Find global min/max for consistent colorbar
    global_min = float('inf')
    global_max = float('-inf')
    
    for fg in fg_keys:
        heatmap_data = all_heatmaps[fg]['heatmap_data']
        if not np.all(np.isnan(heatmap_data)):
            global_min = min(global_min, np.nanmin(heatmap_data))
            global_max = max(global_max, np.nanmax(heatmap_data))
    
    # Use provided vmin/vmax or computed global values
    if vmin is None:
        vmin = global_min if global_min != float('inf') else 0
    if vmax is None:
        vmax = global_max if global_max != float('-inf') else 100
    
    # Create figure with slider
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)
    
    # Initial plot
    initial_fg = fg_keys[0]
    heatmap_data = all_heatmaps[initial_fg]['heatmap_data']
    lens_values = all_heatmaps[initial_fg]['lens_values']
    ke_values = all_heatmaps[initial_fg]['ke_values']
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(ke_values)))
    ax.set_yticks(np.arange(len(lens_values)))
    ax.set_xticklabels([f'{ke:.1f}' for ke in ke_values])
    ax.set_yticklabels([f'{lens:.3f}' for lens in lens_values])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Kinetic Energy (eV)')
    ax.set_ylabel('Lens VMI')
    ax.set_title(f'Energy Resolution (%) - Field Gradient: {initial_fg}')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Energy Resolution (%)')
    
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
            heatmap_data = all_heatmaps[fg]['heatmap_data']
            lens_values = all_heatmaps[fg]['lens_values']
            ke_values = all_heatmaps[fg]['ke_values']
            
            ax.clear()
            im_new = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(ke_values)))
            ax.set_yticks(np.arange(len(lens_values)))
            ax.set_xticklabels([f'{ke:.1f}' for ke in ke_values])
            ax.set_yticklabels([f'{lens:.3f}' for lens in lens_values])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('Kinetic Energy (eV)')
            ax.set_ylabel('Lens VMI')
            ax.set_title(f'Energy Resolution (%) - Field Gradient: {fg}')
            
            fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


def plot_energy_resolution_vs_ke(processed_data, figsize=(14, 10)):
    """
    Plot energy resolution vs kinetic energy curves with sliders for field gradient and lens VMI.
    
    This function creates an interactive plot showing how energy resolution varies with
    kinetic energy for different combinations of field gradient and lens VMI settings.
    
    Args:
        processed_data: Dictionary containing processed SIMION data with structure:
                       processed_data[fg][lens_VMI][ke]['global']['energy_resolution']
        figsize: Figure size as tuple (width, height) in inches
    
    Returns:
        None (displays interactive plot)
    """
    if not processed_data:
        print("No processed data available")
        return
    
    fg_keys = sorted(processed_data.keys())
    
    if not fg_keys:
        print("No field gradient data available")
        return
    
    # Create figure with sliders
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    
    # Get initial lens values for first fg
    initial_fg = fg_keys[0]
    lens_keys = sorted(processed_data[initial_fg].keys())
    
    if not lens_keys:
        print("No lens VMI data available")
        return
    
    initial_lens = lens_keys[0]
    
    # Create sliders
    ax_fg_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    fg_slider = Slider(
        ax=ax_fg_slider,
        label='Field Gradient',
        valmin=0,
        valmax=max(0.1, len(fg_keys) - 1),
        valinit=0,
        valstep=1
    )
    
    ax_lens_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    lens_slider = Slider(
        ax=ax_lens_slider,
        label='Lens VMI',
        valmin=0,
        valmax=max(0.1, len(lens_keys) - 1),
        valinit=0,
        valstep=1
    )
    
    def update(val=None):
        ax.clear()
        
        # Get current fg and lens from sliders
        fg_idx = int(fg_slider.val)
        fg_idx = min(max(0, fg_idx), len(fg_keys) - 1)
        fg = fg_keys[fg_idx]
        fg_slider.valtext.set_text(f'{fg}')
        
        # Update lens slider range for current fg
        current_lens_keys = sorted(processed_data[fg].keys())
        lens_slider.valmax = max(0.1, len(current_lens_keys) - 1)
        
        lens_idx = int(lens_slider.val)
        lens_idx = min(max(0, lens_idx), len(current_lens_keys) - 1)
        lens_vmi = current_lens_keys[lens_idx]
        lens_slider.valtext.set_text(f'{lens_vmi:.3f}')
        
        # Collect energy resolution data for this fg, lens_vmi combination
        ke_values = []
        energy_resolutions = []
        
        for ke in sorted(processed_data[fg][lens_vmi].keys()):
            global_data = processed_data[fg][lens_vmi][ke].get('global', {})
            energy_resolution = global_data.get('energy_resolution', None)
            if energy_resolution is not None and not np.isnan(energy_resolution):
                ke_values.append(ke)
                energy_resolutions.append(energy_resolution)
        
        if ke_values:
            # Plot the curve
            ax.plot(ke_values, energy_resolutions, 'b-o', linewidth=2, markersize=8)
            ax.set_xlabel('Kinetic Energy (eV)', fontsize=12)
            ax.set_ylabel('Energy Resolution (%)', fontsize=12)
            ax.set_title(f'Energy Resolution vs Kinetic Energy\nField Gradient: {fg}, Lens VMI: {lens_vmi:.3f}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add data point labels
            for ke, er in zip(ke_values, energy_resolutions):
                ax.annotate(f'{er:.1f}%', (ke, er), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9)
            
            # Set axis limits with some padding
            if len(ke_values) > 1:
                ke_range = max(ke_values) - min(ke_values)
                ax.set_xlim(min(ke_values) - ke_range * 0.1, max(ke_values) + ke_range * 0.1)
            
            if len(energy_resolutions) > 1:
                er_range = max(energy_resolutions) - min(energy_resolutions)
                ax.set_ylim(min(energy_resolutions) - er_range * 0.1, max(energy_resolutions) + er_range * 0.1)
        else:
            ax.text(0.5, 0.5, 'No energy resolution data available\nfor this combination',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_xlabel('Kinetic Energy (eV)', fontsize=12)
            ax.set_ylabel('Energy Resolution (%)', fontsize=12)
            ax.set_title(f'Energy Resolution vs Kinetic Energy\nField Gradient: {fg}, Lens VMI: {lens_vmi:.3f}', fontsize=14)
        
        fig.canvas.draw_idle()
    
    fg_slider.on_changed(update)
    lens_slider.on_changed(update)
    
    # Initial plot
    update()
    plt.show()


def delete_temp_files():
    """
    Delete all .tmp files in the current directory.
    
    This function finds and removes all files with the .tmp extension
    in the current working directory.
    
    Returns:
        None
    """
    # Get the current directory
    current_dir = os.getcwd()

    # Find all .tmp files in the current directory
    temp_files = glob.glob(os.path.join(current_dir, '*.tmp'))

    # Delete each temp file
    for file in temp_files:
        os.remove(file)
        #print(f"Deleted: {file}")

    print(f"Total temp files deleted: {len(temp_files)}")