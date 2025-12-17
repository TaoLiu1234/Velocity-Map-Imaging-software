"""
Energy Resolution Test Script for SIMION Electron Ion Spectrometer
Tests energy resolution by generating single particles with 4π solid angle distribution
and analyzing their detection patterns.

Based on Simion_workflow -v4.7.py but adapted for energy resolution testing.
"""
import abel
import os
import numpy as np
import Utilis
import delete_temp
import matplotlib.pyplot as plt
# Configuration parameters for energy resolution testing
# These define the parameter sweep for energy resolution analysis

# Energy range to test
KE_MIN = 4.2      # Minimum kinetic energy in eV
KE_MAX = 4.2     # Maximum kinetic energy in eV
NUM_KE_POINTS = 1  # Number of energy points to test

# VMI parameters for velocity imaging mode
FIELD_MIN = 242      # Minimum field gradient in V/cm
FIELD_MAX = 242      # Maximum field gradient in V/cm
NUM_POINTS = 1        # Number of field gradient points to test

LENS_MIN = 1.444     # Minimum lens focusing factor (lens_VMI)
LENS_MAX = 1.444     # Maximum lens focusing factor (lens_VMI)
NUM_LENS_POINTS = 1  # Number of lens points to test

# Particle generation parameters
SOURCE_POSITION = (199, -1, 0.0)  # Fixed position for particle source
NUM_PARTICLES_PER_ENERGY = 10000  # Number of particles per energy point for 4π distribution

# Analysis parameters
X_SCAN_RANGE = (73.0, 166.0)  # Range of x-planes to scan for focus analysis
X_STEP = 0.25               # Step size for the x-plane scan
FOCUS_CRITERION = 'z'       # Axis to use for focus evaluation ('y' or 'z')

# File paths
OUTPUT_FILENAME_FLY2 = 'WORKING_TITLE_energy_resolution_tao.fly2'  # Output file for generated particles
OUTPUT_FILENAME_LUA = "WORKING_TITLE_energy_resolution_tao.lua"      # Standard Lua file name for SIMION
IOB_FILE = "WORKING_TITLE_energy_resolution_tao.iob"              # SIMION project file
OUT_FILE = "energy_resolution_out.txt"              # Output file for simulation results

def plot_yz_positions(final_positions):
    """
    Plot the final y,z positions of all trajectories.
    
    Args:
        final_positions: Dictionary containing final y,z positions matrix
                       with structure final_positions[fg][lens_vmi][ke_electron] = [[y,z], ...]
    """
    plt.figure(figsize=(10, 8))
    
    # Collect all y,z points from all parameter combinations
    all_yz_points = []
    for fg in final_positions:
        for lens_vmi in final_positions[fg]:
            for ke_electron in final_positions[fg][lens_vmi]:
                yz_matrix = final_positions[fg][lens_vmi][ke_electron]
                if len(yz_matrix) > 0:  # Check if matrix is not empty
                    all_yz_points.extend(yz_matrix)
    
    if not all_yz_points:
        print("No y,z positions to plot!")
        return
    
    # Convert to numpy array for easier handling
    yz_array = np.array(all_yz_points)
    y_positions = yz_array[:, 0]
    z_positions = yz_array[:, 1]
    
    

    # Create scatter plot
    plt.scatter(y_positions, z_positions, alpha=0.6, s=10)
    plt.xlabel('Y Position (mm)')
    plt.ylabel('Z Position (mm)')
    plt.title('Final Y,Z Positions of All Trajectories')
    plt.grid(True, alpha=0.3)
    
    # Add histogram on the sides
    # Y histogram
    ax_yhist = plt.axes([0.15, 0.55, 0.2, 0.3])
    ax_yhist.hist(y_positions, bins=30, orientation='horizontal', color='blue', alpha=0.7)
    ax_yhist.set_xticks([])
    ax_yhist.set_yticks([])
    ax_yhist.set_title('Y Distribution')
    
    # Z histogram
    ax_zhist = plt.axes([0.65, 0.15, 0.2, 0.3])
    ax_zhist.hist(z_positions, bins=30, orientation='vertical', color='red', alpha=0.7)
    ax_zhist.set_xticks([])
    ax_zhist.set_yticks([])
    ax_zhist.set_title('Z Distribution')
    
    plt.tight_layout()
    plt.show()


def bin_positions(positions, bin_type='rectangular', bin_interval=0.1, r_bins=None, theta_bins=None,
                  outside_region_width=0.0):
    """
    Bin positions in either rectangular or polar coordinate system.
    
    Args:
        positions: numpy array of shape (n, 2) containing y,z positions
        bin_type: 'rectangular' or 'polar' coordinate system
        bin_interval: bin size for rectangular coordinates (mm) or radial interval for polar (mm)
        r_bins: number of radial bins for polar coordinates (overrides bin_interval if provided)
        theta_bins: number of angular bins for polar coordinates (degrees)
        outside_region_width: width of the region outside the data area to include (mm)
    
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
        
        # No transpose needed - we already created the histogram with theta as rows and r as columns
        return binned_data, (r_edges, theta_edges), (r_centers, theta_centers)
    
    else:
        raise ValueError("bin_type must be either 'rectangular' or 'polar'")


def energy_resolution_analysis(final_positions, bin_interval=0.01, outside_region_width=0.2, visualize=True):
    """
    Perform energy resolution analysis on final positions of particles.
    
    Args:
        final_positions: numpy array of shape (n, 2) containing y,z positions
        bin_interval: bin size for rectangular coordinates (mm)
        outside_region_width: width of region outside data area to include (mm)
        visualize: whether to show visualization plots
        
    Returns:
        energy_resolution: energy resolution as a percentage (ΔE/E)
    """
    if len(final_positions) == 0:
        print("No positions provided for energy resolution analysis")
        return float('inf')
    
    # Find center of final_positions and apply offset to center the image
    center_y = np.mean(final_positions[:, 0])
    center_z = np.mean(final_positions[:, 1])
    
    # Apply offset to center image at (0,0)
    centered_positions = final_positions.copy()
    centered_positions[:, 0] = centered_positions[:, 0] - center_y
    centered_positions[:, 1] = centered_positions[:, 1] - center_z
    
    # Apply rectangular binning
    rect_binned, _, _ = bin_positions(
        centered_positions,
        bin_type='rectangular',
        bin_interval=bin_interval,
        outside_region_width=outside_region_width
    )
    
    # Apply inverse Abel transform
    try:
        recon, distr = abel.rbasex.rbasex_transform(rect_binned)
        r, I, beta = distr.rIbeta()
        
        # Estimate FWHM of intensity distribution I
        half_max = np.max(I) / 2
        indices_above_half = np.where(I >= half_max)[0]
        
        if len(indices_above_half) > 0:
            fwhm_indices = indices_above_half[-1] - indices_above_half[0]
            fwhm_r = r[indices_above_half[-1]] - r[indices_above_half[0]]
        else:
            print("Could not estimate FWHM - no points above half maximum")
            return float('inf')
        
        max_idx = np.where(I == np.max(I))[0]
        max_r = r[max_idx]
        
        # Calculate energy resolution (ΔE/E)
        energy_resolution = (fwhm_r / max_r) * 100
        
        if visualize:
            # Visualize the results
            plt.figure(figsize=(15, 5))
            
            # Original centered positions
            plt.subplot(1, 3, 1)
            plt.scatter(centered_positions[:, 0], centered_positions[:, 1], alpha=0.5, s=1)
            plt.title('Centered Positions')
            plt.xlabel('Y (mm)')
            plt.ylabel('Z (mm)')
            plt.axis('equal')
            
            # Rectangular binned data
            plt.subplot(1, 3, 2)
            plt.imshow(rect_binned.T, origin='lower', cmap='viridis', aspect='equal')
            plt.title('Rectangular Binning')
            plt.xlabel('Y (mm)')
            plt.ylabel('Z (mm)')
            plt.colorbar(label='Count')
            
            # Inverse Abel Transform
            plt.subplot(1, 3, 3)
            plt.imshow(recon, clim=(0, None), cmap='ocean_r')
            plt.title('Inverse Abel Transform')
            plt.colorbar(label='Intensity')
            
            plt.tight_layout()
            plt.show()
            
            # Print results
            print(f"Image centering applied: Y offset = {-center_y:.4f} mm, Z offset = {-center_z:.4f} mm")
            print(f"FWHM of intensity distribution: {fwhm_r:.4f} mm")
            print(f"FWHM in terms of radial bins: {fwhm_indices}")
            print(f"Estimated energy resolution (ΔE/E): {energy_resolution:.4f}%")
        
        return energy_resolution
    
    except Exception as e:
        print(f"Error in energy resolution analysis: {e}")
        return float('inf')


def main():
    """
    Main function to run energy resolution test.
    """
    print("=" * 60)
    print("ENERGY RESOLUTION TEST FOR SIMION ELECTRON ION SPECTROMETER")
    print("=" * 60)
    
    # Generate energy sequence
    energy_sequence = np.linspace(KE_MIN, KE_MAX, NUM_KE_POINTS)
    print(f"Testing energy range: {KE_MIN} eV to {KE_MAX} eV ({NUM_KE_POINTS} points)")
    print(f"Energy values: {energy_sequence}")
    
    # Step 1: Generate VMI electrode voltage parameters
    print("\n" + "="*50)
    print("STEP 1: Generating VMI parameters")
    print("="*50)
    
    param = Utilis.compute_vmi_parameters(
        field_min=FIELD_MIN, field_max=FIELD_MAX, num_points=NUM_POINTS, 
        lens_min=LENS_MIN, lens_max=LENS_MAX, num_lens_points=NUM_LENS_POINTS,
        mode='velocity_imaging', save_to_file=False, filename='energy_resolution_parameters.mat'
    )
    
    print(f"Generated {len(param['field_gradient'])} parameter combinations")
    print(f"Field gradients: {param['field_gradient']}")
    print(f"Lens VMI values: {param['lens_VMI']}")
    
    # Step 2: Clear previous output files
    print("\n" + "="*50)
    print("STEP 2: Clearing previous output files")
    print("="*50)
    
    Utilis.clear_file_contents(OUT_FILE)
    print(f"Cleared contents of '{OUT_FILE}'")
    
    # Step 3: Run energy resolution simulations
    print("\n" + "="*50)
    print("STEP 3: Running energy resolution simulations")
    print("="*50)
    
    for i, ke in enumerate(energy_sequence):
        # Suppressed detailed progress output
        # print(f"\n--- Energy Point {i+1}/{NUM_KE_POINTS}: KE = {ke:.2f} eV ---")
        
        # Generate particles with 4π solid angle distribution for this energy
        # print(f"Generating {NUM_PARTICLES_PER_ENERGY} particles with 4π solid angle distribution...")
        Utilis.energy_resolution_utilis(
            filename=OUTPUT_FILENAME_FLY2,
            position=SOURCE_POSITION,
            num_particles=NUM_PARTICLES_PER_ENERGY,
            ke=ke
        )
        
        # Run optimized SIMION simulations for this energy
        # print(f"Running SIMION simulations for KE = {ke:.2f} eV...")
        Utilis.run_optimized_simulations_with_ke(param, ke, OUTPUT_FILENAME_LUA, IOB_FILE, OUT_FILE)
        
        # print(f"Completed simulations for KE = {ke:.2f} eV")
    
    # Step 4: Process simulation data for energy resolution analysis
    print("\n" + "="*50)
    print("STEP 4: Processing energy resolution data")
    print("="*50)
    
    # Process the accumulated simulation data
    processed_data = Utilis.process_data(
        x_range=X_SCAN_RANGE,
        file_path=OUT_FILE,
        focus_axis=FOCUS_CRITERION,
        fly2_file=OUTPUT_FILENAME_FLY2,
        y_range=(-0.5, 0.5),
        z_range=(-0.5, 0.5)
    )
    
    if processed_data is None:
        print("ERROR: Data processing failed!")
        return
    
    print("Energy resolution data processing completed.")
    
    # Store final y,z positions of all trajectories as a matrix
    final_positions = {}
    for fg in processed_data:
        final_positions[fg] = {}
        for lens_vmi in processed_data[fg]:
            final_positions[fg][lens_vmi] = {}
            for ke_electron in processed_data[fg][lens_vmi]:
                yz_matrix = []
                local_data = processed_data[fg][lens_vmi][ke_electron].get('local', {})
                for particle_idx in local_data:
                    trajectories = local_data[particle_idx].get('trajectories', [])
                    for trajectory in trajectories:
                        if trajectory:  # Check if trajectory is not empty
                            # Get the final point (last point in the trajectory)
                            final_point = trajectory[-1]
                            y, z = final_point[1], final_point[2]  # Extract y and z coordinates
                            yz_matrix.append([y, z])
                
                final_positions[fg][lens_vmi][ke_electron] = np.array(yz_matrix)

    print("Final y,z positions matrix has been stored.")
    
    # Plot y,z positions
    plot_yz_positions(final_positions)
    
    # Step 5: Energy Resolution Analysis
    final_positions = np.array(final_positions[242][1.444][4.2])
    
    # Find the center of the final_positions and apply offset to center the image
    if len(final_positions) > 0:
        # Calculate the center (mean) of the y,z positions
        center_y = np.mean(final_positions[:, 0])
        center_z = np.mean(final_positions[:, 1])
        
        # Apply offset to center the image at (0,0)
        final_positions[:, 0] = final_positions[:, 0] - center_y
        final_positions[:, 1] = final_positions[:, 1] - center_z
        
        print(f"Image centering applied: Y offset = {-center_y:.4f} mm, Z offset = {-center_z:.4f} mm")
    
    # Example of using the binning function
    print("\nApplying binning to centered positions...")
    
    # Rectangular binning example with outside region
    rect_binned, rect_edges, rect_centers = bin_positions(
        final_positions,
        bin_type='rectangular',
        bin_interval=0.01,      # 0.05 mm bins
        outside_region_width = 2  # 0.2 mm outside region
    )
    print(f"Rectangular binning: Created {rect_binned.shape} grid with 0.05 mm bins and 2 mm outside region")
    
    # Polar binning example with outside region
    polar_binned, polar_edges, polar_centers = bin_positions(
        final_positions,
        bin_type='polar',
        bin_interval=0.01,      # 0.01 mm radial bins
        theta_bins=36,          # 10-degree angular bins
        outside_region_width=0.2  # 0.2 mm outside region
    )
    print(f"Polar binning: Created {polar_binned.shape} grid with 0.01 mm radial bins, 36 angular bins, and 0.2 mm outside region")
    
    # Visualize the binned data
    plt.figure(figsize=(15, 5))
    
    # Original centered positions
    plt.subplot(1, 3, 1)
    plt.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.5, s=1)
    plt.title('Centered Positions')
    plt.xlabel('Y (mm)')
    plt.ylabel('Z (mm)')
    plt.axis('equal')
    
    # Rectangular binned data
    plt.subplot(1, 3, 2)
    y_centers, z_centers = rect_centers
    plt.imshow(rect_binned.T, origin='lower', extent=[y_centers[0], y_centers[-1], z_centers[0], z_centers[-1]],
               cmap='viridis', aspect='equal')
    plt.title('Rectangular Binning')
    plt.xlabel('Y (mm)')
    plt.ylabel('Z (mm)')
    plt.colorbar(label='Count')
    
    # Polar binned data
    plt.subplot(1, 3, 3)
    r_centers, theta_centers = polar_centers
    plt.imshow(polar_binned, origin='lower', extent=[r_centers[0], r_centers[-1], theta_centers[0], theta_centers[-1]],
               cmap='viridis', aspect='auto')
    plt.title('Polar Binning')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Angle (degrees)')
    plt.colorbar(label='Count')
    
    plt.tight_layout()
    plt.show()
    

    recon, distr = abel.rbasex.rbasex_transform(rect_binned)
    r, I, beta = distr.rIbeta()
    
    # Estimate FWHM of the intensity distribution I
    half_max = np.max(I) / 2
    # Find indices where intensity crosses half maximum
    indices_above_half = np.where(I >= half_max)[0]
    if len(indices_above_half) > 0:
        fwhm_indices = indices_above_half[-1] - indices_above_half[0]
        fwhm_r = r[indices_above_half[-1]] - r[indices_above_half[0]]
        print(f"FWHM of intensity distribution: {fwhm_r:.4f} mm")
        print(f"FWHM in terms of radial bins: {fwhm_indices}")
    else:
        print("Could not estimate FWHM - no points above half maximum")
    max_idx = np.where(I == np.max(I))[0]
    max_r = r[max_idx]
    energy_resolution =  (fwhm_r / max_r)*100
    print(f"Estimated energy resolution (ΔE/E): {energy_resolution[0]:.4f}%")
    plt.imshow(recon, clim=(0, None), cmap='ocean_r')
    plt.title('Inverse Abel Transform')
    plt.colorbar(label='Intensity')
    plt.show()
    


if __name__ == "__main__":
    main()
