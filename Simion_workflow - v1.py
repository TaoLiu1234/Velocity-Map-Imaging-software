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
FIELD_MIN = 200
FIELD_MAX = 500
NUM_POINTS = 150 # for field gradient
LENS_VALUE = 1.38
NUM_GROUPS = 30 # for beam
X_SCAN_RANGE = (73.0, 165.0)
SCAN_STEP = 0.5
FOCUS_CRITERION = 'z'
INPUT_FILE = 'out.txt'
OUTPUT_FILENAME_LUA = "WORKING_TITLE_tao.lua"
OUTPUT_FILENAME_FLY2 = 'WORKING_TITLE_tao.fly2'
IOB_FILE = "WORKING_TITLE_tao.iob"
OUT_FILE = "out.txt"

# Generate parameters
param = Utilis.compute_vmi_parameters(
    field_min=FIELD_MIN, field_max=FIELD_MAX, num_points=NUM_POINTS, lens_value=LENS_VALUE,
    save_to_file=False, filename='parameters.mat'
)

# Generate particles
Utilis.generate_particles_fly2(num_groups=NUM_GROUPS, filename=OUTPUT_FILENAME_FLY2)

# Simulation loop
for field_idx in range(len(param['field_gradient'])):
    print(f"Field gradient: {param['field_gradient'][field_idx]} V/cm")
    Utilis.generate_simion_lua_file(field_idx, param, output_filename=OUTPUT_FILENAME_LUA)
    command = f"simion.exe --nogui fly --recording-output={OUT_FILE} {IOB_FILE}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_parameters = f"parameters = [{param['field_gradient'][field_idx]},{param['Offset_to_ground'][field_idx]},{param['lens_VMI'][field_idx]},{param['I_grid'][field_idx]},{param['VMI2'][field_idx]},{param['VMI1'][field_idx]},{param['e_grid'][field_idx]},{param['dt_e'][field_idx]}]"
    Utilis.add_parameters_to_out_file(current_parameters, out_file_path=OUT_FILE)

# Process data
processed_data = Utilis.process_data(
    x_range=X_SCAN_RANGE,
    x_step=SCAN_STEP,
    file_path=INPUT_FILE,
    focus_axis=FOCUS_CRITERION,
    fly2_file=OUTPUT_FILENAME_FLY2
)

# Visualization and analysis
Utilis.data_viewer(processed_data, mode='multiple', focus_axis=FOCUS_CRITERION, fly2_file=OUTPUT_FILENAME_FLY2)
Utilis.plot_fg_vs_std(processed_data, focus_criterion=FOCUS_CRITERION)
