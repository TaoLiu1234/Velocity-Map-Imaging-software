"""
Optimized SIMION Workflow - v5.1.7
- SAME FUNCTIONALITY as v5 (uses Utilis.energy_resolution_analysis)
- FASTER: Pre-computed parameters, optimized parallel execution
- ENHANCED: Dynamic memory and temp file cleanup for large datasets
- Checkpoint recovery support
- Interactive slider visualization (same as v5)
"""

import os
import gc
import glob
import pickle
import Utilis
import numpy as np
import time
import psutil

# ============================================================================
# CONFIGURATION
# ============================================================================

# === CONTROL FLAGS ===
SKIP_INITIAL_WORKFLOW = True   # Set True to skip initial simulations
SKIP_FOCUS_FILTERING = True    # Set True to skip focus filtering

KE_MIN = 1
KE_MAX = 5
NUM_KE_POINTS = 21

electron_energy_sequence = np.linspace(KE_MIN, KE_MAX, NUM_KE_POINTS)
if KE_MIN >= KE_MAX:
    electron_energy_sequence = np.array([KE_MIN])

Theta = 0 * 2 * np.pi / 360

FIELD_MIN = 50
FIELD_MAX = 500
NUM_POINTS = 51

LENS_MIN = 1
LENS_MAX = 5
NUM_LENS_POINTS = 41

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
NUM_PARTICLES_PER_ENERGY = 10000
SOURCE_POSITION = (199, -1, 0.0)
BIN_INTERVAL = 0.01
OUTSIDE_REGION_WIDTH = 2
TOLERABLE_OFFSET = 3.0

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
            'trapcheck.info'
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

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

print("=" * 70)
print("SIMION Workflow v5.1.7 - Enhanced Dynamic Resource Management")
print("=" * 70)
print(f"SKIP_INITIAL_WORKFLOW: {SKIP_INITIAL_WORKFLOW}")
print(f"SKIP_FOCUS_FILTERING: {SKIP_FOCUS_FILTERING}")
print(f"Initial Memory: {resource_mgr.get_memory_mb():.0f}MB")

workflow_start = time.time()

# Initial cleanup
cleaned = resource_mgr.cleanup_temp_files()
if cleaned > 0:
    print(f"Cleaned {cleaned} temp files from previous run")

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

if not SKIP_INITIAL_WORKFLOW:
    Utilis.clear_file_contents(OUT_FILE, INPUT_FILE)

    print("\nStep 2: Running SIMION simulations...")
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
        
        Utilis.run_optimized_simulations_with_ke_parallel(
            param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE
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
    from collections import defaultdict
    processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'global': {}, 'local': {}})))
    
    for fg in unique_fgs:
        for lens_vmi in unique_lens:
            for ke in electron_energy_sequence:
                processed_data[fg][lens_vmi][ke]['global'] = {}
                processed_data[fg][lens_vmi][ke]['local'] = {}

resource_mgr.check_and_cleanup(force=True)

# ============================================================================
# ENERGY RESOLUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Energy Resolution Analysis")
print("=" * 70)

if SKIP_FOCUS_FILTERING:
    print("SKIP_FOCUS_FILTERING=True: Analyzing ALL combinations directly")
    
    all_combinations = []
    for fg in unique_fgs:
        for lens_vmi in unique_lens:
            for ke in electron_energy_sequence:
                all_combinations.append((fg, lens_vmi, ke))
    
    print(f"Total combinations: {len(all_combinations)}")
    
    processed_data = Utilis.energy_resolution_analysis_direct(
        processed_data,
        all_combinations=all_combinations,
        source_position=SOURCE_POSITION,
        num_particles_per_energy=NUM_PARTICLES_PER_ENERGY,
        x_scan_range=X_SCAN_RANGE,
        bin_interval=BIN_INTERVAL,
        outside_region_width=OUTSIDE_REGION_WIDTH,
        batch_size=25,  # Smaller batches for better memory management
        enable_memory_optimization=True,
        checkpoint_interval=10
    )
else:
    processed_data = Utilis.energy_resolution_analysis(
        processed_data,
        tolerable_offset=TOLERABLE_OFFSET,
        source_position=SOURCE_POSITION,
        num_particles_per_energy=NUM_PARTICLES_PER_ENERGY,
        x_scan_range=X_SCAN_RANGE,
        bin_interval=BIN_INTERVAL,
        outside_region_width=OUTSIDE_REGION_WIDTH,
        batch_size=25,
        enable_memory_optimization=True,
        checkpoint_interval=10
    )

resource_mgr.check_and_cleanup(force=True)
resource_mgr.print_status()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Saving Results")
print("=" * 70)

try:
    def convert_to_dict(obj):
        from collections import defaultdict
        if isinstance(obj, defaultdict):
            obj = {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            obj = {k: convert_to_dict(v) for k, v in obj.items()}
        return obj
    
    processed_data_clean = convert_to_dict(processed_data)
    with open('processed_data_final.pkl', 'wb') as f:
        pickle.dump(processed_data_clean, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved: processed_data_final.pkl")
except Exception as e:
    print(f"Warning: Could not save processed data: {e}")

# Summary
total_time = time.time() - workflow_start
print(f"\nTotal workflow time: {total_time/60:.1f} minutes")

# ============================================================================
# VISUALIZATION (Same as v5 - with slider)
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: Visualization (Interactive Slider)")
print("=" * 70)

# Use the slider-based heatmap visualization (same as v5)
# This shows energy resolution vs lens VMI for each field gradient with a slider
print("Launching interactive heatmap viewer with slider...")
try:
    # Use plot_heatmap_all_fg for slider-based visualization across all field gradients
    Utilis.plot_heatmap_all_fg(processed_data)
except Exception as e:
    print(f"Slider visualization failed: {e}")
    # Fallback to individual heatmaps
    print("Falling back to individual heatmaps...")
    for fg in sorted(unique_fgs):
        print(f"  Generating heatmap for FG={fg}...")
        try:
            heatmap_data, lens_values, ke_values = Utilis.heatmap_energy_lens(processed_data, fg=fg)
            Utilis.plot_heatmap_energy_lens(processed_data, fg=fg)
        except Exception as e2:
            print(f"    Heatmap for FG={fg} skipped: {e2}")

print("\n" + "=" * 70)
print("Workflow Complete!")
print("=" * 70)

# Keep plots open and show interactive visualization
print("\nFinalizing the data analysis...")
try:
    import matplotlib.pyplot as plt
    plt.show()
except:
    pass
