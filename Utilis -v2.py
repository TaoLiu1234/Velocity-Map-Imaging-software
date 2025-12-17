import os
import random
import re
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat
from scipy.interpolate import griddata
from functools import partial


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
        A dictionary where keys are group indices and values are tuples (az, first_el).
    """
    group_angles = {}
    group_idx = 0
    current_az = None
    first_el = None

    with open(filename, 'r') as f:
        content = f.read()

    # Find all standard_beam blocks
    beam_pattern = re.compile(r'standard_beam\s*\{(.*?)\}', re.DOTALL)
    beams = beam_pattern.findall(content)

    for beam in beams:
        # Extract az
        az_match = re.search(r'az\s*=\s*([-\d]+)', beam)
        if az_match:
            az = int(az_match.group(1))
        else:
            continue

        # Extract first el
        el_match = re.search(r'el\s*=\s*arithmetic_sequence\s*\{\s*first\s*=\s*([-\d]+)', beam)
        if el_match:
            el = int(el_match.group(1))
        else:
            continue

        if current_az != az:
            # New group
            if current_az is not None:
                group_angles[group_idx] = (current_az, first_el)
                group_idx += 1
            current_az = az
            first_el = el
        # For same az, keep the first el

    # Last group
    if current_az is not None:
        group_angles[group_idx] = (current_az, first_el)

    return group_angles


def parse_out_file(filename: str) -> MainData:
    """
    Parses a SIMION 'out.txt' file to extract particle trajectories.

    Args:
        filename: The path to the 'out.txt' file.

    Returns:
        A nested dictionary containing the parsed data, structured as:
        `data[field_gradient][lens_VMI]['local'][particle_idx] = [trajectory_1, ...] `.
    """
    param_pattern = re.compile(r'parameters\s*=\s*\[([^\]]+)\]')
    data: MainData = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    all_blocks: List[Tuple[float, float, List[Tuple[int, float, float, float]]]] = []
    current_block_data: List[Tuple[int, float, float, float]] = []
    current_fg = None
    current_lens_VMI = None

    print(f"Reading and parsing '{filename}'...")
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if "Begin Next Fly'm" in line:
                if current_fg is not None and current_lens_VMI is not None and current_block_data:
                    all_blocks.append((current_fg, current_lens_VMI, current_block_data))
                current_block_data = []
                current_fg = None
                current_lens_VMI = None
                continue

            if (param_match := param_pattern.search(line)):
                try:
                    params_str = param_match.group(1)
                    params = [float(p.strip()) for p in params_str.split(',')]
                    current_fg = params[0]
                    current_lens_VMI = params[2]
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse parameters on line {line_num}: {line.strip()}")
                continue

            line = line.strip()
            if not line or line.startswith('"'):
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                try:
                    ion_n, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    current_block_data.append((ion_n, x, y, z))
                except (ValueError, IndexError):
                    pass  # Ignore malformed data lines

    if current_fg is not None and current_lens_VMI is not None and current_block_data:
        all_blocks.append((current_fg, current_lens_VMI, current_block_data))

    print(f"Found {len(all_blocks)} simulation blocks.")
    print("Grouping trajectories into particles based on ion number rule...")

    for fg, lens_VMI, block_data in all_blocks:
        # First, aggregate all points for each ion number into a full trajectory
        ion_trajectories = defaultdict(list)
        for ion_n, x, y, z in block_data:
            ion_trajectories[ion_n].append((x, y, z))

        if not ion_trajectories:
            continue

        # Group trajectories into particles using the fixed rule.
        for ion_n, traj in ion_trajectories.items():
            if not traj or ion_n < 1:
                continue

            # Rule: ion_n 1-8 -> p_idx 0; 9-16 -> p_idx 1, etc.
            # This is a stable grouping method based on the data storage rule.
            p_idx = (ion_n - 1) // 8

            data[fg][lens_VMI]['local'][p_idx].append(traj)

    return data


def calculate_final_position_stats(data: MainData):
    """
    Calculates and prints the standard deviation of the final y/z positions
    for each particle and stores it in the 'global' key of the data dict.
    """
    for fg in sorted(data.keys()):
        for lens_VMI in sorted(data[fg].keys()):
            print(f"\n🧲 Field Gradient = {fg}, Lens VMI = {lens_VMI}")
            particles = data[fg][lens_VMI].get('local', {})
            if not particles:
                continue

            print(f"  🔹 Detected {len(particles)} distinct particles")
            all_final_pos_y, all_final_pos_z = [], []

            for p_idx in sorted(particles.keys()):
                trajs = particles[p_idx]
                num_trajs = len(trajs)
                total_points = sum(len(t) for t in trajs)
                avg_length = total_points / num_trajs if num_trajs > 0 else 0
                print(f"    Particle {p_idx}: {num_trajs} trajectories, ~{avg_length:.1f} pts each")

                if num_trajs == 0:
                    continue

                # Extract final positions (y, z) from each trajectory
                final_positions = np.array([traj[-1][1:3] for traj in trajs])  # shape (8, 2)

                # Reshape into 4 groups of 2 trajectories each, each with (y, z)
                groups = final_positions.reshape(4, 2, 2)  # (4 groups, 2 traj, 2 coords)
                # Calculate mean for each group
                group_means = np.mean(groups, axis=1)  # (4, 2)
                # Center each group by subtracting its mean
                centered_groups = groups - group_means[:, np.newaxis, :]  # broadcast to (4, 2, 2)
                # Flatten and extend
                all_final_pos_y.extend(centered_groups[:, :, 0].flatten())
                all_final_pos_z.extend(centered_groups[:, :, 1].flatten())

            # Calculate standard deviation across all particles for this field gradient
            std_dev_y = np.std(all_final_pos_y) if all_final_pos_y else 0.0
            std_dev_z = np.std(all_final_pos_z) if all_final_pos_z else 0.0
            counts_y, bins_y = np.histogram(all_final_pos_y, bins='auto')
            counts_z, bins_z = np.histogram(all_final_pos_z, bins='auto')
            data[fg][lens_VMI]['global'] = {
                'counts_y': counts_y,
                'bins_y': bins_y,
                'counts_z': counts_z,
                'bins_z': bins_z,
                'std_dev_y': std_dev_y,
                'std_dev_z': std_dev_z
            }
    return data


def analyze_beam_across_x_planes(data: MainData, x_planes: Iterable[float], group_angles: Dict[int, Tuple[int, int]] = None) -> AnalysisResults:
    """
    Analyzes beam properties (range, mean) at different x-planes.
    Divides trajectories into 4 groups of 2 trajectories each, each group corresponding to different emission angles.
    """
    results: AnalysisResults = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))

    # Use a small threshold relative to the plane distance for finding points
    threshold_factor = 0.125  # +- 0.125mm of x_plane

    for fg, fg_data in data.items():
        for lens_VMI, lens_data in fg_data.items():
            for p_idx, trajectories in lens_data.get('local', {}).items():
                if not trajectories:
                    continue
                num_trajs = len(trajectories)
                if num_trajs != 8:
                    print(f"Warning: particle {p_idx} has {num_trajs} trajectories, expected 8")
                    continue

                for g_idx in range(4):
                    group_trajs = trajectories[g_idx*2 : (g_idx+1)*2]
                    all_points = np.concatenate([np.array(t) for t in group_trajs])

                    # Use (az, el) as key if available, else g_idx
                    key = group_angles[g_idx] if group_angles else g_idx

                    # Efficiently find points near each x_plane
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

                            results[fg][lens_VMI][p_idx][x_plane][key] = [mean_y, mean_z, ptp_y, ptp_z]
    return results


def find_and_store_focus(data: MainData, results: AnalysisResults, group_angles: Dict[int, Tuple[int, int]] = None):
    """
    Finds the focus for each unique (az, el) combination for both Y and Z axes and stores all focus points in the data dictionary.

    The focus is the x-plane with the minimum beam extent (ptp) along the specified axis for each (az, el) group per particle.

    Args:
        data: The main data dictionary to store results in.
        results: The analysis results from `analyze_beam_across_x_planes`.
        group_angles: Dict mapping g_idx to (az, el) if using angle keys.
    """
    for focus_axis in ['y', 'z']:
        # --- How to choose the focus evaluation axis ---
        # The `stats` list is ordered as [mean_y, mean_z, ptp_y, ptp_z].
        # To evaluate focus based on the y-axis ptp, we use index 2.
        # To evaluate focus based on the z-axis ptp, we use index 3.
        if focus_axis == 'y':
            axis_index = 2
            print("\nFinding focus based on minimum beam extent in Y...")
        elif focus_axis == 'z':
            axis_index = 3
            print("\nFinding focus based on minimum beam extent in Z...")

        min_range_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # fg -> lens_VMI -> key -> list of (ptp, x_plane, mean_y, mean_z, p_idx)

        for fg, fg_results in results.items():
            for lens_VMI, lens_results in fg_results.items():
                for p_idx, x_plane_data in lens_results.items():
                    for g_idx in range(4):
                        key = group_angles[g_idx] if group_angles else g_idx
                        # Collect valid planes for this group
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
                            min_range_results[fg][lens_VMI][key].append((stats[axis_index], x_plane, stats[0], stats[1], p_idx))

        # Store the focus positions back into the main data structure
        for fg, fg_min_data in min_range_results.items():
            for lens_VMI, lens_min_data in fg_min_data.items():
                if 'global' not in data[fg][lens_VMI]:
                    data[fg][lens_VMI]['global'] = {}
                focus_points = {key: [(x, y, z) for _, x, y, z, _ in sorted(group_data, key=lambda x: x[4])] for key, group_data in lens_min_data.items()}
                data[fg][lens_VMI]['global'][f'focus_points_{focus_axis}'] = focus_points


def process_data(x_range: Tuple[float, float], x_step: float, file_path: str, focus_axis: str, fly2_file: str = None):
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

        # --- Step 2: Calculate and print stats about final positions ---
        data = calculate_final_position_stats(data)
        print("\nFinished calculating final position stats.")

        # --- Step 3: Get group angles if fly2_file provided ---
        group_angles = None
        if fly2_file:
            group_angles = parse_fly2_file(fly2_file)

        # --- Step 4: Analyze beam properties across a range of x-planes ---
        x_planes_to_scan = np.arange(x_start, x_stop, x_step)
        print(f"\nAnalyzing beam focus across {len(x_planes_to_scan)} x-planes (from {x_start} to {x_stop} with step {x_step})...")
        beam_analysis_results = analyze_beam_across_x_planes(data, x_planes_to_scan, group_angles)

        # --- Step 5: Find the focus and store the results ---
        find_and_store_focus(data, beam_analysis_results, group_angles=group_angles)
        print("Finished analyzing beam focus.")

        return data

    except FileNotFoundError:
        print(f"\nError: File '{file_path}' not found. Make sure the path is correct.")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        return None


def compute_vmi_parameters(field_min=200, field_max=500, num_points=20, lens_min=1.38, lens_max=1.38, num_lens_points=1,
                           save_to_file=False, filename='parameters.mat'):
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

    # Pre-calculate slope factor: -800 / 120 = -20/3 ≈ -6.6667
    slope_factor = -800 / 120

    idx = 0
    for lens in lens_sequence:
        for field in field_sequence:
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
        'field_gradient': np.int32(field_gradient),
        'Offset_to_ground': np.int32(Offset_to_ground),
        'lens_VMI': np.round(lens_VMI, 4),
        'I_grid': np.int32(I_grid),
        'VMI2': np.int32(VMI2),
        'VMI1': np.int32(VMI1),
        'e_grid': np.int32(e_grid),
        'dt_e': np.int32(dt_e)
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
        # Default angles
        group_angles = {i: (az, 45) for i, az in enumerate([0, 180, 90, -90])}  # Assume el=45 for default

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


def generate_particles_fly2(num_groups=4, filename='particles_tao.fly2', x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_range=(-0.5, 0.5)):
    """
    Generates a .fly2 file with particle definitions for SIMION.

    This function creates a specified number of particle groups. Each group consists of 4 ion pairs
    (which means 8 particles in total per group), generated from 4 separate `standard_beam`
    definitions. A key feature is that all ion pairs within the same group share the same
    randomized position jitter (dx, dy, dz), simulating a localized event.

    Summary of generated particles:
    - Number of groups to generate: `num_groups`
    - Ion pairs per group: 4
    - Particles per group: 8
    - Total ion pairs generated: `num_groups` * 4
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
    """
    
    # If num_groups is not a positive integer, set to 0 and warn the user.
    if not isinstance(num_groups, int) or num_groups < 0:
        print(f"⚠️ Invalid input for `num_groups` (value: {num_groups}). It must be a non-negative integer. Setting to 0.")
        num_groups = 0
    
    if num_groups == 0:
        print("`num_groups` is 0, an empty particle file will be created.")


    if filename is None or filename.strip() == '':
        filename = 'particles_tao.fly2'

    try:
        with open(filename, 'w') as fid:
            fid.write('particles {\n')
            
            # Only write coordinates line if there are particles
            if num_groups > 0:
                fid.write('  coordinates = 0,\n')

                # Azimuth sequence for each of the 4 beams in a group
                az_sequence = [0, 180, 90, -90]

                for i in range(num_groups):
                    # Shared jitter (dx, dy, dz) for this group
                    dx = random.uniform(x_range[0], x_range[1])
                    dy = random.uniform(y_range[0], y_range[1])
                    dz = random.uniform(z_range[0], z_range[1])

                    # Randomly choose whether el1 + el2 = 180 or -180
                    sum_el = 180 if random.random() < 0.5 else -180

                    el1 = random.randint(0, 180)
                    el2 = sum_el - el1

                    # Write 4 standard_beam entries per group, each is an ion pair (2 particles)
                    for j in range(4):
                        az = az_sequence[j]

                        # The last beam definition in the file must not have a trailing comma.
                        is_last_beam = (i == num_groups - 1 and j == 3)
                        need_comma = not is_last_beam

                        if j % 2 == 0:  # Odd-positioned beam (1st, 3rd): positive el
                            write_standard_beam(fid, el1, el2, 199 + dx, -1 + dy, dz, az, need_comma)
                        else:           # Even-positioned beam (2nd, 4th): negative el
                            write_standard_beam(fid, -el1, -el2, 199 + dx, -1 + dy, dz, az, need_comma)

            fid.write('}\n')
        
        total_pairs = num_groups * 4
        total_particles = num_groups * 8
        print(f"✅ Successfully generated {num_groups} groups ({total_pairs} ion pairs, {total_particles} particles total).")
        print(f"   File saved to: \"{filename}\"")

    except Exception as e:
        raise IOError(f"Failed to create file {filename}: {e}")


def write_standard_beam(fid, el1, el2, x, y, z, az, need_comma=True):
    """
    Write a single standard_beam block to the file.

    Args:
        fid (file object): Open file handle for writing
        el1 (int): First elevation angle
        el2 (int): Second elevation angle (step is computed as el2 - el1)
        x (float): X position
        y (float): Y position
        z (float): Z position
        az (int): Azimuth angle
        need_comma (bool): Whether to add a comma after the block
    """
    fid.write('  standard_beam {\n')
    fid.write('    n = 2,\n')
    fid.write('    tob = 0,\n')
    fid.write('    mass = 0.000548579903,\n')
    fid.write('    charge = -1,\n')
    fid.write('    ke = 15,\n')
    fid.write(f'    az = {az},\n')
    fid.write('    el = arithmetic_sequence {\n')
    fid.write(f'      first = {round(el1)},\n')
    fid.write(f'      step = {round(el2 - el1)},\n')
    fid.write('      n = 2')
    fid.write('    },\n')
    fid.write('    cwf = 1,\n')
    fid.write('    color = 0,\n')
    fid.write(f'    position = vector({x:.6f}, {y:.6f}, {z:.6f})')
    fid.write('  }')
    
    # Add comma and newline if needed; otherwise just newline
    fid.write(',\n' if need_comma else '\n')

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
            param_line = f"parameters = {parameters}\n"
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
    lua_content = f"""-- LUA file, automatically created from the MATLAB function write_GEM_file. 
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

def calculate_data_collection_efficiency(data: MainData) -> Dict[float, Dict[float, float]]:
    """
    Calculates the efficiency of data collection for each field gradient and lens VMI combination.

    Efficiency = num_trajectory_generated / num_trajectory_received
    num_trajectory_generated = NUM_GROUPS * 8
    num_trajectory_received = num_particles_received * num_trajectory_per_particles

    Args:
        data: The main data dictionary containing processed simulation data.

    Returns:
        A dictionary with field_gradient -> lens_VMI -> efficiency value.
    """
    efficiencies = {}
    for fg in data:
        efficiencies[fg] = {}
        for lens_VMI in data[fg]:
            local_data = data[fg][lens_VMI].get('local', {})
            num_particles_received = len(local_data)
            if local_data:
                num_trajectory_per_particles = len(list(local_data.values())[0])
            else:
                num_trajectory_per_particles = 0
            num_trajectory_received = num_particles_received * num_trajectory_per_particles
            # NUM_GROUPS * 8 = num_trajectory_generated
            # Assuming NUM_GROUPS = 2 as per workflow
            num_trajectory_generated = 2 * 8
            efficiency = num_trajectory_generated / num_trajectory_received if num_trajectory_received > 0 else 0
            efficiencies[fg][lens_VMI] = efficiency
    return efficiencies


def data_viewer(data, mode='single', focus_axis='y', fg_idx=None, fly2_file=None, r2_threshold=0.2):
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
    # Get emission angles
    if fly2_file:
        group_angles = parse_fly2_file(fly2_file)
        angles = [f'Az {az}°, El {el}°' for az, el in group_angles.values()]
    else:
        # Default angles
        angles = [f'Az {az}°' for az in [0, 180, 90, -90]]

    # Compute metrics for highest peak and closest to zero across all fg and lens_VMI combinations
    fg_keys = sorted(data.keys())
    max_peak_y_fg = None
    max_peak_y_lens = None
    max_peak_y = 0
    min_std_y_fg = None
    min_std_y_lens = None
    min_std_y = float('inf')
    max_peak_z_fg = None
    max_peak_z_lens = None
    max_peak_z = 0
    min_std_z_fg = None
    min_std_z_lens = None
    min_std_z = float('inf')
    min_std_x_focus_y_fg = None
    min_std_x_focus_y_lens = None
    min_std_x_focus_y = float('inf')
    min_std_x_focus_z_fg = None
    min_std_x_focus_z_lens = None
    min_std_x_focus_z = float('inf')
    min_slope_y = float('inf')
    min_slope_y_fg = None
    min_slope_y_lens = None
    min_slope_z = float('inf')
    min_slope_z_fg = None
    min_slope_z_lens = None

    for fg in fg_keys:
        for lens_VMI in data[fg].keys():
            global_data = data[fg][lens_VMI]['global']
            if 'counts_y' in global_data and global_data['counts_y'].size > 0:
                peak_y = np.max(global_data['counts_y'])
                if peak_y > max_peak_y:
                    max_peak_y = peak_y
                    max_peak_y_fg = fg
                    max_peak_y_lens = lens_VMI
                std_y = global_data.get('std_dev_y', float('inf'))
                if std_y < min_std_y:
                    min_std_y = std_y
                    min_std_y_fg = fg
                    min_std_y_lens = lens_VMI
            if 'counts_z' in global_data and global_data['counts_z'].size > 0:
                peak_z = np.max(global_data['counts_z'])
                if peak_z > max_peak_z:
                    max_peak_z = peak_z
                    max_peak_z_fg = fg
                    max_peak_z_lens = lens_VMI
                std_z = global_data.get('std_dev_z', float('inf'))
                if std_z < min_std_z:
                    min_std_z = std_z
                    min_std_z_fg = fg
                    min_std_z_lens = lens_VMI
            # Compute std of focus x positions for Y
            focus_points_y = global_data.get('focus_points_y', {})
            if focus_points_y:
                x_focus_y = [x for points_list in focus_points_y.values() for x, y, z in points_list]
                if x_focus_y:
                    std_x_focus_y = np.std(x_focus_y)
                    if std_x_focus_y < min_std_x_focus_y:
                        min_std_x_focus_y = std_x_focus_y
                        min_std_x_focus_y_fg = fg
                        min_std_x_focus_y_lens = lens_VMI
            # Compute std of focus x positions for Z
            focus_points_z = global_data.get('focus_points_z', {})
            if focus_points_z:
                x_focus_z = [x for points_list in focus_points_z.values() for x, y, z in points_list]
                if x_focus_z:
                    std_x_focus_z = np.std(x_focus_z)
                    if std_x_focus_z < min_std_x_focus_z:
                        min_std_x_focus_z = std_x_focus_z
                        min_std_x_focus_z_fg = fg
                        min_std_x_focus_z_lens = lens_VMI
            # Compute Petzval slope for Y
            if focus_points_y:
                xs_xz = [x for points_list in focus_points_y.values() for x, y, z in points_list]
                z2s = [z**2 for points_list in focus_points_y.values() for x, y, z in points_list]
                if len(xs_xz) > 1:
                    coeffs_xz = np.polyfit(z2s, xs_xz, 1)
                    slope_xz, intercept_xz = coeffs_xz
                    predicted = slope_xz * np.array(z2s) + intercept_xz
                    ss_res = np.sum((np.array(xs_xz) - predicted)**2)
                    ss_tot = np.sum((np.array(xs_xz) - np.mean(xs_xz))**2)
                    r_squared_xz = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    slope_xz_abs = abs(slope_xz)
                    if (r2_threshold is None or r_squared_xz <= r2_threshold) and slope_xz_abs < abs(min_slope_y):
                        min_slope_y = slope_xz_abs
                        min_slope_y_fg = fg
                        min_slope_y_lens = lens_VMI
            # Compute Petzval slope for Z
            if focus_points_z:
                xs_xy = [x for points_list in focus_points_z.values() for x, y, z in points_list]
                y2s = [y**2 for points_list in focus_points_z.values() for x, y, z in points_list]
                if len(xs_xy) > 1:
                    coeffs_xy = np.polyfit(y2s, xs_xy, 1)
                    slope_xy, intercept_xy = coeffs_xy
                    predicted = slope_xy * np.array(y2s) + intercept_xy
                    ss_res = np.sum((np.array(xs_xy) - predicted)**2)
                    ss_tot = np.sum((np.array(xs_xy) - np.mean(xs_xy))**2)
                    r_squared_xy = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    slope_xy_abs = abs(slope_xy)
                    if (r2_threshold is None or r_squared_xy <= r2_threshold) and slope_xy_abs < abs(min_slope_z):
                        min_slope_z = slope_xy_abs
                        min_slope_z_fg = fg
                        min_slope_z_lens = lens_VMI

    # Prepare info strings
    info_y = f"Y: Highest peak FG {max_peak_y_fg} (Lens {max_peak_y_lens}), minimal std FG {min_std_y_fg} (Lens {min_std_y_lens})"
    info_z = f"Z: Highest peak FG {max_peak_z_fg} (Lens {max_peak_z_lens}), minimal std FG {min_std_z_fg} (Lens {min_std_z_lens})"
    info_x = f"X focus lowest std in Y focus criterion: FG {min_std_x_focus_y_fg} (Lens {min_std_x_focus_y_lens})\nX focus lowest std in Z focus criterion: FG {min_std_x_focus_z_fg} (Lens {min_std_x_focus_z_lens})\nSmallest Petzval slope in Y focus criterion: FG {min_slope_y_fg} (Lens {min_slope_y_lens})\nSmallest Petzval slope in Z focus criterion: FG {min_slope_z_fg} (Lens {min_slope_z_lens})"
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

        fig = plt.figure(figsize=(18, 10))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        fg_keys = sorted(data.keys())

        # Precompute all data for faster updates
        all_data = {}
        for fg in fg_keys:
            all_data[fg] = {}
            lens_keys = sorted(data[fg].keys())
            for lens_VMI in lens_keys:
                counts_y = data[fg][lens_VMI]['global']['counts_y']
                bins_y = data[fg][lens_VMI]['global']['bins_y']
                if len(bins_y) > 1:
                    bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
                    widths_y = np.diff(bins_y)
                else:
                    bin_centers_y = np.array([])
                    widths_y = np.array([])

                counts_z = data[fg][lens_VMI]['global']['counts_z']
                bins_z = data[fg][lens_VMI]['global']['bins_z']
                if len(bins_z) > 1:
                    bin_centers_z = (bins_z[:-1] + bins_z[1:]) / 2
                    widths_z = np.diff(bins_z)
                else:
                    bin_centers_z = np.array([])
                    widths_z = np.array([])
                focus_points_y = data[fg][lens_VMI].get('global', {}).get('focus_points_y', {})
                focus_points_z = data[fg][lens_VMI].get('global', {}).get('focus_points_z', {})
                peak_pct_y = (np.max(counts_y) / np.sum(counts_y)) * 100 if len(counts_y) > 0 else 0
                peak_pct_z = (np.max(counts_z) / np.sum(counts_z)) * 100 if len(counts_z) > 0 else 0
                # Precompute peak positions
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

                # Precompute focus plot data
                xs_y, zs, labels_y = [], [], []
                xs_z, ys, labels_z = [], [], []
                if focus_points_y:
                    for key, points_list in focus_points_y.items():
                        for x, y, z in points_list:
                            xs_y.append(x)
                            zs.append(z)
                            if isinstance(key, tuple) and len(key) == 2:
                                az, el = key
                                labels_y.append(f'Az {az}°, El {el}°')
                            else:
                                labels_y.append(f'Group {key}')
                if focus_points_z:
                    for key, points_list in focus_points_z.items():
                        for x, y, z in points_list:
                            xs_z.append(x)
                            ys.append(y)
                            if isinstance(key, tuple) and len(key) == 2:
                                az, el = key
                                labels_z.append(f'Az {az}°, El {el}°')
                            else:
                                labels_z.append(f'Group {key}')

                # Precompute Petzval fit for XZ and XY planes
                fit_text_xz = ''
                fit_text_xy = ''
                if focus_points_y:
                    xs_xz = np.array([x for points_list in focus_points_y.values() for x, y, z in points_list])
                    z2s = np.array([z**2 for points_list in focus_points_y.values() for x, y, z in points_list])
                    if len(xs_xz) > 1:
                        coeffs_xz = np.polyfit(z2s, xs_xz, 1)
                        slope_xz, intercept_xz = coeffs_xz
                        predicted = slope_xz * z2s + intercept_xz
                        ss_res = np.sum((xs_xz - predicted)**2)
                        ss_tot = np.sum((xs_xz - np.mean(xs_xz))**2)
                        r_squared_xz = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        fit_text_xz = f'Petzval XZ: X = {intercept_xz:.3f} + {slope_xz:.3f} * z², R²={r_squared_xz:.3f}'
                if focus_points_z:
                    xs_xy = np.array([x for points_list in focus_points_z.values() for x, y, z in points_list])
                    y2s = np.array([y**2 for points_list in focus_points_z.values() for x, y, z in points_list])
                    if len(xs_xy) > 1:
                        coeffs_xy = np.polyfit(y2s, xs_xy, 1)
                        slope_xy, intercept_xy = coeffs_xy
                        predicted = slope_xy * y2s + intercept_xy
                        ss_res = np.sum((xs_xy - predicted)**2)
                        ss_tot = np.sum((xs_xy - np.mean(xs_xy))**2)
                        r_squared_xy = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        fit_text_xy = f'Petzval XY: X = {intercept_xy:.3f} + {slope_xy:.3f} * y², R²={r_squared_xy:.3f}'

                all_data[fg][lens_VMI] = {
                    'bin_centers_y': bin_centers_y,
                    'widths_y': widths_y,
                    'counts_y': counts_y,
                    'bin_centers_z': bin_centers_z,
                    'widths_z': widths_z,
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
                    'fit_text_xz': fit_text_xz,
                    'fit_text_xy': fit_text_xy
                }

        fig.suptitle(f"{info_y}\n{info_z}\n{info_x}", fontsize=10)
        fig.subplots_adjust(bottom=0.4, hspace=0.8, wspace=0.3, top=0.8)
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

        # Store sliders in figure to prevent garbage collection
        fig.fg_slider = fg_slider
        fig.lens_slider = lens_slider

        def update_lens_slider(fg_idx):
            fg = fg_keys[fg_idx]
            lens_keys = sorted(data[fg].keys())
            lens_slider.valmax = max(0.1, len(lens_keys) - 1)
            lens_slider.valmin = 0
            lens_slider.val = 0
            lens_slider.ax.set_xlim(lens_slider.valmin, lens_slider.valmax)
            lens_slider.valtext.set_text(f'{lens_keys[0]}')

        def update(val=None):
            fg_idx = int(fg_slider.val)
            fg = fg_keys[fg_idx]
            fg_slider.valtext.set_text(f'{fg}')

            lens_idx = int(lens_slider.val)
            lens_keys = sorted(data[fg].keys())
            lens_VMI = lens_keys[lens_idx]
            lens_slider.valtext.set_text(f'{lens_VMI}')

            d = all_data[fg][lens_VMI]

            # --- Update Y histogram ---
            ax1.clear()
            if len(d['bin_centers_y']) > 0:
                ax1.bar(d['bin_centers_y'], d['counts_y'], width=d['widths_y'], align='center', edgecolor='k', alpha=0.7)
            ax1.set_xlabel('Y position')
            ax1.set_ylabel('Counts')
            ax1.set_title(f'Histogram Y for FG {fg}, Lens {lens_VMI}')
            if d['peak_pct_y'] > 0:
                ax1.text(0.95, 0.95, f'Peak %: {d["peak_pct_y"]:.2f}%', transform=ax1.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # --- Update Y Focus Plot ---
            ax2.clear()
            if d['xs_y']:
                ax2.scatter(d['xs_y'], d['zs'], c='blue', marker='o')
            ax2.set_xlabel('X position')
            ax2.set_ylabel('Z position')
            ax2.set_title(f'XZ Focus Points for FG {fg}, Lens {lens_VMI}')
            if d['fit_text_xz']:
                ax2.text(0.05, 0.95, d['fit_text_xz'], transform=ax2.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # --- Update Z histogram ---
            ax3.clear()
            if len(d['bin_centers_z']) > 0:
                ax3.bar(d['bin_centers_z'], d['counts_z'], width=d['widths_z'], align='center', edgecolor='k', alpha=0.7)
            ax3.set_xlabel('Z position')
            ax3.set_ylabel('Counts')
            ax3.set_title(f'Histogram Z for FG {fg}, Lens {lens_VMI}')
            if d['peak_pct_z'] > 0:
                ax3.text(0.95, 0.95, f'Peak %: {d["peak_pct_z"]:.2f}%', transform=ax3.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # --- Update Z Focus Plot ---
            ax4.clear()
            if d['xs_z']:
                ax4.scatter(d['xs_z'], d['ys'], c='blue', marker='o')
            ax4.set_xlabel('X position')
            ax4.set_ylabel('Y position')
            ax4.set_title(f'XY Focus Points for FG {fg}, Lens {lens_VMI}')
            if d['fit_text_xy']:
                ax4.text(0.05, 0.95, d['fit_text_xy'], transform=ax4.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Calculate data collection efficiency
            local_data = data[fg][lens_VMI].get('local', {})
            num_particles_received = len(local_data)
            if local_data:
                num_trajectory_per_particles = len(list(local_data.values())[0])
            else:
                num_trajectory_per_particles = 0
            total_received_trajectory = num_particles_received * num_trajectory_per_particles
            # NUM_GROUPS * 8 = num trajectory generated
            # Assuming NUM_GROUPS = 2 as per workflow
            num_trajectory_generated = 2 * 8
            efficiency = num_trajectory_generated / total_received_trajectory if total_received_trajectory > 0 else 0

            suptitle_str = f"{info_y}\n{info_z}\n{info_x}\nCurrent FG {fg}, Lens {lens_VMI}, Data Collection Efficiency: {efficiency}"
            fig.suptitle(suptitle_str, fontsize=10)

            fig.canvas.draw_idle()

        def on_fg_change(val):
            update_lens_slider(int(val))
            update()

        fg_slider.on_changed(on_fg_change)
        lens_slider.on_changed(update)

        # Initial setup
        update_lens_slider(0)
        update()
        plt.show()
    else:
        raise ValueError("Mode must be 'single' or 'multiple'")


def para_2d_landscape(data, target='peak_pct_y', fly2_file=None):
    # Draws a 3D scatter plot showing the parameter landscape for a specified target function.
    # target: 'peak_pct_y', 'peak_pct_z', 'std_x_y', 'std_x_z', 'petzval_y', 'petzval_z'
    # Compute metrics for each fg, lens combination
    metrics = {}
    for fg in data:
        for lens_VMI in data[fg]:
            global_data = data[fg][lens_VMI]['global']
            # peak_pct_y
            counts_y = global_data['counts_y']
            peak_pct_y = (np.max(counts_y) / np.sum(counts_y)) * 100 if len(counts_y) > 0 else 0
            counts_z = global_data['counts_z']
            peak_pct_z = (np.max(counts_z) / np.sum(counts_z)) * 100 if len(counts_z) > 0 else 0
            std_y = global_data.get('std_dev_y', 0)
            std_z = global_data.get('std_dev_z', 0)
            # std_x_y
            focus_points_y = global_data.get('focus_points_y', {})
            x_focus_y = [x for points_list in focus_points_y.values() for x, y, z in points_list]
            std_x_y = np.std(x_focus_y) if x_focus_y else 0
            focus_points_z = global_data.get('focus_points_z', {})
            x_focus_z = [x for points_list in focus_points_z.values() for x, y, z in points_list]
            std_x_z = np.std(x_focus_z) if x_focus_z else 0
            # petzval_y
            if focus_points_y:
                xs_xz = np.array([x for points_list in focus_points_y.values() for x, y, z in points_list])
                z2s = np.array([z**2 for points_list in focus_points_y.values() for x, y, z in points_list])
                if len(xs_xz) > 1:
                    coeffs_xz = np.polyfit(z2s, xs_xz, 1)
                    petzval_y = abs(coeffs_xz[0])
                else:
                    petzval_y = 0
            else:
                petzval_y = 0
            # petzval_z
            if focus_points_z:
                xs_xy = np.array([x for points_list in focus_points_z.values() for x, y, z in points_list])
                y2s = np.array([y**2 for points_list in focus_points_z.values() for x, y, z in points_list])
                if len(xs_xy) > 1:
                    coeffs_xy = np.polyfit(y2s, xs_xy, 1)
                    petzval_z = abs(coeffs_xy[0])
                else:
                    petzval_z = 0
            else:
                petzval_z = 0
            metrics[(fg, lens_VMI)] = {
                'peak_pct_y': peak_pct_y,
                'peak_pct_z': peak_pct_z,
                'std_y': std_y,
                'std_z': std_z,
                'std_x_criterion_y': std_x_y,
                'std_x_criterion_z': std_x_z,
                'abs(petzval_y)': petzval_y,
                'abs(petzval_z)': petzval_z
            }
    # Validate target
    valid_targets = ['peak_pct_y', 'peak_pct_z', 'std_x_criterion_y', 'std_x_criterion_z', 'abs(petzval_y)', 'abs(petzval_z)']
    if target not in valid_targets:
        raise ValueError(f"Invalid target. Choose from {valid_targets}")

    target_labels = {
        'peak_pct_y': 'Peak % (Y)',
        'peak_pct_z': 'Peak % (Z)',
        'std_x_criterion_y': 'Std of X focus (Y criterion)',
        'std_x_criterion_z': 'Std of X focus (Z criterion)',
        'abs(petzval_y)': 'Petzval slope (Y)',
        'abs(petzval_z)': 'Petzval slope (Z)'
    }

    # Get unique fg and lens
    fgs = sorted(set(fg for fg, _ in metrics))
    lens_VMIs = sorted(set(lens for _, lens in metrics))
    # Create grid
    fg_grid, lens_grid = np.meshgrid(fgs, lens_VMIs)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Collect points
    points = []
    values = []
    for (fg, lens), mets in metrics.items():
        points.append([fg, lens])
        values.append(mets[target])
    points = np.array(points)
    values = np.array(values)
    # Interpolate
    Z = griddata(points, values, (fg_grid, lens_grid), method='nearest')
    Z = np.ma.masked_invalid(Z)
    # Plot surface
    surf = ax.plot_surface(fg_grid, lens_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Field Gradient')
    ax.set_ylabel('Lens VMI')
    ax.set_zlabel(target_labels[target])
    ax.set_title(f'Parameter Landscape: {target_labels[target]}')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()
