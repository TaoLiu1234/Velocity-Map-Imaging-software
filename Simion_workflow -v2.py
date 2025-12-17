import matplotlib
# Try to set an interactive backend
try:
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        try:
            matplotlib.use('Qt4Agg')
        except ImportError:
            try:
                matplotlib.use('WXAgg')
            except ImportError:
                pass  # Fall back to default

import Utilis
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import numpy as np
import subprocess

# Configuration parameters
FIELD_MIN = 150
FIELD_MAX = 450
NUM_POINTS = 20 # for field gradient
LENS_MIN = 2 #1.38
LENS_MAX = 2.1
NUM_LENS_POINTS = 20 # for lens sequence

NUM_GROUPS = 10 # num of particle for beam each has 8 trajectories
X_SCAN_RANGE = (73.0, 165.0)
SCAN_STEP = 0.25 # scan steps
FOCUS_CRITERION = 'z'
INPUT_FILE = 'out.txt'
OUTPUT_FILENAME_LUA = "WORKING_TITLE_tao.lua"
OUTPUT_FILENAME_FLY2 = 'WORKING_TITLE_tao.fly2'
IOB_FILE = "WORKING_TITLE_tao.iob"
OUT_FILE = "out.txt"

# Generate parameters
param = Utilis.compute_vmi_parameters(
    field_min=FIELD_MIN, field_max=FIELD_MAX, num_points=NUM_POINTS, lens_min=LENS_MIN, lens_max=LENS_MAX, num_lens_points=NUM_LENS_POINTS,
    save_to_file=False, filename='parameters.mat'
)

# Generate particles
Utilis.generate_particles_fly2(num_groups=NUM_GROUPS, filename=OUTPUT_FILENAME_FLY2, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1))

# Simulation loop
for field_idx in range(len(param['field_gradient'])):
    print(f"Field gradient: {param['field_gradient'][field_idx]} V/cm, lens VMI: {param['lens_VMI'][field_idx]}")
    Utilis.generate_simion_lua_file(field_idx, param, output_filename=OUTPUT_FILENAME_LUA)
    command = f"simion.exe --nogui fly --recording-output={OUT_FILE} {IOB_FILE}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_parameters = f"parameters = [{param['field_gradient'][field_idx]},{param['Offset_to_ground'][field_idx]},{param['lens_VMI'][field_idx]},{param['I_grid'][field_idx]},{param['VMI2'][field_idx]},{param['VMI1'][field_idx]},{param['e_grid'][field_idx]},{param['dt_e'][field_idx]}]"
    Utilis.add_parameters_to_out_file(current_parameters, out_file_path=OUT_FILE)

# Data processing
processed_data = Utilis.process_data(
    x_range=X_SCAN_RANGE,
    x_step=SCAN_STEP,
    file_path=INPUT_FILE,
    focus_axis=FOCUS_CRITERION,
    fly2_file=OUTPUT_FILENAME_FLY2
)

# Visualization and analysis
Utilis.data_viewer(processed_data, mode='multiple', focus_axis=FOCUS_CRITERION, fly2_file=OUTPUT_FILENAME_FLY2,r2_threshold=0.2)
Utilis.para_2d_landscape(processed_data, target='std_x_criterion_y')
# ('peak_pct_y', 'peak_pct_z', 'std_x_criterion_y', 'std_x_criterion_z', 'abs(petzval_y)', 'abs(petzval_z)')
