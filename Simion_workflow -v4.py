# Import the Utilis module containing all utility functions for SIMION data processing and simulation
import Utilis
import delete_temp



# Configuration parameters for the SIMION simulation workflow
# These define the parameter sweep ranges and simulation settings
electron_energy = 10 # Electron energy in eV
FIELD_MIN = 50  # Minimum field gradient in V/cm for parameter sweep
FIELD_MAX = 500  # Maximum field gradient in V/cm for parameter sweep
NUM_POINTS = 20   # Number of field gradient points to simulate (reduced from v2's 20 for faster testing)
LENS_MIN = 1    # Minimum lens focusing factor (lens_VMI), adjusted from v2's 1.38
LENS_MAX = 2.5  # Maximum lens focusing factor (lens_VMI)
NUM_LENS_POINTS = 20  # Number of lens points to simulate (reduced from v2's 20)
NUM_GROUPS = 3  # Number of particle groups to generate (each group has 8 trajectorr54ies, reduced from v2's 10)
X_SCAN_RANGE = (73.0, 165.0)  # Range of x-planes to scan for focus analysis (in mm)
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

# Step 2: Generate particle definitions for SIMION
# Creates a .fly2 file with randomized particle groups, each with shared position jitter
# Each group has 4 ion pairs (8 particles total) with varying emission angles
Utilis.generate_particles_fly2(num_groups=NUM_GROUPS, filename=OUTPUT_FILENAME_FLY2, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_range=(-0.5, 0.5),ke=electron_energy)

# Clear content of OUT_FILE and INPUT_FILE before simulations
Utilis.clear_file_contents(OUT_FILE, INPUT_FILE)

# Step 3: Run optimized SIMION simulations
# Optimization: Use run_optimized_simulations for parallel Lua generation and sequential SIMION runs.
# Bottleneck in v2: Sequential loop for Lua generation and SIMION runs, with each simulation waiting for the previous one.
# Why optimized: Lua generation is I/O-bound and independent, so parallelized using ThreadPoolExecutor to generate all Lua files concurrently.
# SIMION runs kept sequential to ensure correct Lua loading and avoid race conditions in file access.
# Result: Faster startup for multiple simulations, reducing total workflow time for parameter sweeps.
Utilis.run_optimized_simulations(param, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE)

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
Utilis.data_viewer(processed_data, mode='multiple', focus_axis=FOCUS_CRITERION, fly2_file=OUTPUT_FILENAME_FLY2, r2_threshold=0.2, num_groups=NUM_GROUPS, electron_energy=electron_energy, x_range=X_SCAN_RANGE)

# Step 5: Visualization and analysis
# Generate 2D parameter landscape plot for focus quality metric
Utilis.para_2d_landscape(processed_data, target='dr')
# Display interactive data viewer with histograms and focus plots
