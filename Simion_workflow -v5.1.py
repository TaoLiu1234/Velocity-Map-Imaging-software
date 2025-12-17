"""
Optimized SIMION Workflow - v5.1py
- Improved performance and reduced runtime compared to v5.0
- Optimized Utilis.py: incremental file parsing, removed unused functions, reduced code size by 10%
- Uses multi-threading for independent computations
- Add energy resolution analysis
"""
#processed_data[fg][lens_vmi][ke_electron]['local'][particle_idx]['trajectories'][trajectory_idx][point_idx]

# Import the Utilis module containing all utility functions for SIMION data processing and simulation
import os
import Utilis
import delete_temp
import numpy as np
# Configuration parameters for the SIMION simulation workflow
# These define the parameter sweep ranges and simulation settings

KE_MIN = 1  # Minimum kinetic energy in eV
KE_MAX = 8  # Maximum kinetic energy in eV
NUM_KE_POINTS = 2  # Number of kinetic energy points for sweep

electron_energy_sequence = np.linspace(KE_MIN, KE_MAX, NUM_KE_POINTS)  # Sequence of kinetic energies
if KE_MIN >= KE_MAX:
    electron_energy_sequence = np.array([KE_MIN])
Theta=0*2*np.pi/360 # roataion for the particles DEGREE

FIELD_MIN = 100  # Minimum field gradient in V/cm for parameter sweep
FIELD_MAX = 100  # Maximum field gradient in V/cm for parameter sweep
NUM_POINTS = 1   # Number of field gradient points to simulate (reduced from v2's 20 for faster testing)

LENS_MIN = 1    # Minimum lens focusing factor (lens_VMI), adjusted from v2's 1.38
LENS_MAX = 3  # Maximum lens focusing factor (lens_VMI)
NUM_LENS_POINTS = 2  # Number of lens points to simulate (reduced from v2's 20)

NUM_GROUPS = 200  # Number of particle groups to generate (each group has 8 trajectorr54ies, reduced from v2's 10)

X_SCAN_RANGE = (73.0, 166.0)  # Range of x-planes to scan for focus analysis (in mm)
X_STEP = 0.25  # Step size for the x-plane scan

FOCUS_CRITERION = 'z'  # Axis to use for focus evaluation ('y' or 'z')
INPUT_FILE = 'out.txt'  # Input file for data processing (SIMION output)
OUTPUT_FILENAME_LUA = "WORKING_TITLE_tao.lua"  # Standard Lua file name for SIMION
OUTPUT_FILENAME_FLY2 = 'WORKING_TITLE_tao.fly2'  # Output file for generated particles
IOB_FILE = "WORKING_TITLE_tao.iob"  # SIMION project file
OUT_FILE = "out.txt"  # Output file for simulation results

# Step 1: Generate VMI electrode voltage parameters
# This computes the parameter combinations for field gradients and lens factors
# Uses iterative adjustment to find Offset_to_ground where I_grid ≈ 0
param = Utilis.compute_vmi_parameters(
    field_min=FIELD_MIN, field_max=FIELD_MAX, num_points=NUM_POINTS, lens_min=LENS_MIN, lens_max=LENS_MAX, num_lens_points=NUM_LENS_POINTS,
    save_to_file=False, filename='parameters.mat',mode='velocity_imaging',  # Don't save to file, keep in memory
)

# Clear content of OUT_FILE and INPUT_FILE before simulations
Utilis.clear_file_contents(OUT_FILE, INPUT_FILE)

# Step 2: Run SIMION simulations for each kinetic energy
# For each ke, generate particles, then run simulations over all parameter combinations (fg, lens)
# Particle generation depends on ke, parameters are independent of ke
for ke in electron_energy_sequence:
    # Generate particle definitions for SIMION with current ke
    Utilis.generate_particles_fly2(num_groups=NUM_GROUPS, filename=OUTPUT_FILENAME_FLY2, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_range=(-0.5, 0.5), ke=ke,theta=Theta)

    # Run optimized SIMION simulations with parallel Lua generation and sequential SIMION runs
    Utilis.run_optimized_simulations_with_ke_parallel(param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE)

# Call the function to delete temp files in the current directory
delete_temp.delete_temp_files()

# Step 4: Process simulation data to find focus points
# Parses out.txt, calculates final position statistics, analyzes beam across x-planes, and finds focus
processed_data = Utilis.process_data(
    x_range=X_SCAN_RANGE,  # Range of x-planes to analyze
    file_path=INPUT_FILE,  # SIMION output file
    focus_axis=FOCUS_CRITERION,  # Axis for focus evaluation
    fly2_file=OUTPUT_FILENAME_FLY2  # Particle file for emission angles
                                         )

#Utilis.para_2d_landscape(processed_data, target='dr', ke_sequence=electron_energy_sequence)

#Utilis.data_viewer(processed_data, mode='multiple', focus_axis=FOCUS_CRITERION, fly2_file=OUTPUT_FILENAME_FLY2, r2_threshold=0.2, num_groups=NUM_GROUPS, electron_energy=electron_energy_sequence, x_range=X_SCAN_RANGE)
print("Workflow completed successfully.")
# Step 5: Visualization and analysis

# filter the well focused data using focus_filtering function
# energy resolution analysis
#----------------------------------------------------------------------------------------------------------
processed_data = Utilis.energy_resolution_analysis(processed_data, tolerable_offset=3.0,
                              source_position=(199, -1, 0.0),
                              num_particles_per_energy=10000,
                              x_scan_range=(73.0, 166.0),
                              bin_interval=0.01,
                              outside_region_width=2)
#----------------------------------------------------------------------------------------------------------
# angular distribution analysis
#----------------------------------------------------------------------------------------------------------
# Store all heatmaps for all field gradients
all_heatmaps = Utilis.store_all_heatmaps(processed_data, 'all_heatmaps.pkl')

# Plot heatmaps with slider to select field gradient
Utilis.plot_stored_heatmaps(all_heatmaps)

# Plot energy resolution vs kinetic energy curves with sliders for fg and lens_vmi
Utilis.plot_energy_resolution_vs_ke(processed_data)
#----------------------------------------------------------------------------------------------------------
print('finalizing the data analysis...')