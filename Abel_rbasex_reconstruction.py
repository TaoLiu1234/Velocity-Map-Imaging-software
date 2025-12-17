"""
Abel rBasex Reconstruction for Photoelectron Spectroscopy (VMI)

This module implements reconstruction using the rBasex method from PyAbel.
rBasex is a regularized basis set expansion method that provides:
- Direct extraction of radial intensity distribution
- Beta parameter (anisotropy) extraction
- Robust handling of noisy data

Reference:
    Ryazanov, M. "Development of methods for low-energy velocity map imaging 
    and applications to reactions of atomic chlorine" (2012)

Author: Adapted for VMI analysis
"""

import numpy as np
import abel
from scipy.signal import find_peaks, peak_widths
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import time


def reconstruct_rbasex(image: np.ndarray, config=None, 
                       verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Reconstruct VMI image using rBasex method.
    
    Args:
        image: Input VMI image (2D numpy array, should be centered)
        config: Optional Config object from Abel_forward_simulation for energy conversion
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - params: List of reconstructed peak parameters
        - metadata: Dictionary with reconstruction metadata
    """
    t0 = time.time()
    n_pixels = image.shape[0]
    
    if verbose:
        print("rBasex Reconstruction...")
    
    # Run rBasex transform
    recon_image, distr = abel.rbasex.rbasex_transform(
        image, 
        direction='inverse', 
        basis_dir=None,  # Use default basis
        verbose=False
    )
    
    # Extract radial distribution and beta
    r_rb, I_rb, beta_rb = distr.rIbeta()
    
    # Mask center region (artifacts)
    mask_radius = 10
    I_rb[:mask_radius] = 0
    
    # Normalize for peak finding
    if np.max(I_rb) > 0:
        I_rb_norm = I_rb / np.max(I_rb)
    else:
        I_rb_norm = I_rb
    
    # Find peaks with more stringent criteria to avoid spurious peaks
    peaks_idx, properties = find_peaks(
        I_rb_norm, 
        height=0.08,  # Higher threshold (8% of max) to avoid noise
        distance=12,  # Larger minimum distance between peaks
        prominence=0.05  # Higher prominence requirement
    )
    
    # If no peaks found with stringent criteria, try more lenient
    if len(peaks_idx) == 0:
        peaks_idx, properties = find_peaks(
            I_rb_norm, 
            height=0.05,  # 5% of max
            distance=8,   # Minimum distance between peaks
            prominence=0.03
        )
    
    # Calculate FWHM for each peak
    if len(peaks_idx) > 0:
        widths_res = peak_widths(I_rb_norm, peaks_idx, rel_height=0.5)
        fwhms = widths_res[0]
        sigmas = fwhms / 2.355
    else:
        fwhms = []
        sigmas = []
    
    # Filter peaks based on physical criteria
    filtered_peaks = []
    filtered_fwhms = []
    filtered_sigmas = []
    
    for i, p_idx in enumerate(peaks_idx):
        r_val = r_rb[p_idx]
        fwhm = fwhms[i] if i < len(fwhms) else 0
        sigma = sigmas[i] if i < len(sigmas) else 0
        
        # Physical filtering criteria
        # 1. Radius should be reasonable (not too close to center or edge)
        if r_val < 15 or r_val > len(r_rb) - 10:
            continue
            
        # 2. FWHM should be reasonable (not too narrow or too wide)
        if fwhm < 0.5 or fwhm > 20:
            continue
            
        # 3. Intensity should be significant
        if I_rb_norm[p_idx] < 0.05:
            continue
            
        filtered_peaks.append(p_idx)
        filtered_fwhms.append(fwhm)
        filtered_sigmas.append(sigma)
    
    # Extract parameters for each filtered peak
    params = []
    total_amp = 0
    
    for i, p_idx in enumerate(filtered_peaks):
        r_val = r_rb[p_idx]
        
        # Get beta at this radius (average over small window)
        beta_window = 3
        start_idx = max(0, p_idx - beta_window)
        end_idx = min(len(beta_rb), p_idx + beta_window + 1)
        beta_val = np.mean(beta_rb[start_idx:end_idx])
        
        # Apply reasonable beta constraints (wider than before)
        beta_val = np.clip(beta_val, -2.0, 3.0)
        
        # Get amplitude (use normalized intensity for better comparison)
        amp = I_rb_norm[p_idx]
        total_amp += amp
        
        # Get sigma/FWHM
        sigma = filtered_sigmas[i] if i < len(filtered_sigmas) else 4.0
        fwhm = filtered_fwhms[i] if i < len(filtered_fwhms) else sigma * 2.355
        
        # Ensure minimum sigma for physical realism
        sigma = max(sigma, 0.5)
        fwhm = max(fwhm, 1.0)
        
        params.append({
            'r': r_val,
            'sigma': sigma,
            'fwhm': fwhm,
            'amp': amp,
            'beta': beta_val
        })
    
    # Calculate branching ratios
    if total_amp > 0:
        for p in params:
            p['branching_ratio'] = p['amp'] / total_amp
    
    # Convert to energy if config provided
    if config is not None:
        for p in params:
            p['energy_eV'] = radius_to_energy(
                p['r'], 
                config.pixel_size, 
                config.vmi_k,
                config.mass
            )
    
    # Prepare metadata
    metadata = {
        'n_peaks': len(params),
        'r_grid': r_rb,
        'intensity_profile': I_rb,
        'beta_profile': beta_rb,
        'recon_image': recon_image,
        'image_size': n_pixels,
        'method': 'rBasex'
    }
    
    solver_time = time.time() - t0
    
    if verbose:
        print(f"rBasex Solver Time: {solver_time:.2f}s")
        print("\n" + "="*60)
        print("rBasex RECONSTRUCTION RESULTS")
        print("="*60)
        for i, p in enumerate(params):
            print(f"\nPeak {i+1}:")
            print(f"  Radius: {p['r']:.1f} px")
            if 'energy_eV' in p:
                print(f"  Energy: {p['energy_eV']:.3f} eV")
            print(f"  Sigma: {p['sigma']:.2f} px (FWHM: {p['fwhm']:.2f} px)")
            print(f"  Beta: {p['beta']:.3f}")
            if 'branching_ratio' in p:
                print(f"  Branching ratio: {p['branching_ratio']:.3f}")
        print("="*60)
    
    return params, metadata


def radius_to_energy(radius_px: float, pixel_size_mm: float, vmi_k: float, 
                     mass_amu: float = None) -> float:
    """
    Convert detector radius to electron energy.
    
    Args:
        radius_px: Radius in pixels
        pixel_size_mm: Pixel size in mm
        vmi_k: VMI conversion coefficient (mm/(m/s))
        mass_amu: Particle mass in amu (default: electron)
        
    Returns:
        Energy in eV
    """
    from scipy.constants import electron_mass, elementary_charge, atomic_mass
    
    if mass_amu is None:
        mass_amu = electron_mass / atomic_mass
    
    radius_mm = radius_px * pixel_size_mm
    velocity = radius_mm / vmi_k  # m/s
    
    mass_kg = mass_amu * atomic_mass
    E_joule = 0.5 * mass_kg * velocity**2
    E_eV = E_joule / elementary_charge
    
    return E_eV


def compare_methods(image: np.ndarray, true_params: Dict, config=None,
                   physics_params: List[Dict] = None,
                   rbasex_params: List[Dict] = None,
                   physics_metadata: Dict = None,
                   rbasex_metadata: Dict = None,
                   save_path: Optional[str] = None) -> None:
    """
    Compare PhysicsBasedFitter and rBasex reconstruction results.
    
    Args:
        image: Original VMI image
        true_params: Dictionary with true simulation parameters
        config: Config object for energy conversion
        physics_params: Results from PhysicsBasedFitter
        rbasex_params: Results from rBasex
        physics_metadata: Metadata from PhysicsBasedFitter
        rbasex_metadata: Metadata from rBasex
        save_path: Path to save comparison figure
    """
    n_pixels = image.shape[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # =========================================================================
    # Row 1: Images
    # =========================================================================
    
    # Original image
    ax1 = axes[0, 0]
    vmax = np.percentile(image, 99.5)
    im1 = ax1.imshow(image, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title("Input VMI Image (Projection)")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # rBasex reconstruction
    ax2 = axes[0, 1]
    if rbasex_metadata is not None and 'recon_image' in rbasex_metadata:
        recon_rb = rbasex_metadata['recon_image']
        vmax_rb = np.percentile(recon_rb, 99.5)
        im2 = ax2.imshow(recon_rb, cmap='hot', vmin=0, vmax=vmax_rb)
        ax2.set_title("rBasex Reconstruction (Slice)")
        plt.colorbar(im2, ax=ax2, fraction=0.046)
    else:
        ax2.text(0.5, 0.5, "No rBasex result", ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title("rBasex Reconstruction")
    
    # PhysicsBasedFitter reconstruction
    ax3 = axes[0, 2]
    if physics_params is not None:
        from Abel_backward_reconstruction import reconstruct_2d_from_params
        physics_2d = reconstruct_2d_from_params(physics_params, n_pixels)
        im3 = ax3.imshow(physics_2d, cmap='hot')
        ax3.set_title("PhysicsBasedFitter Reconstruction (Slice)")
        plt.colorbar(im3, ax=ax3, fraction=0.046)
    else:
        ax3.text(0.5, 0.5, "No Physics result", ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title("PhysicsBasedFitter Reconstruction")
    
    # =========================================================================
    # Row 2: Profiles and comparison
    # =========================================================================
    
    # Radial profile comparison
    ax4 = axes[1, 0]
    
    # True profile
    if true_params is not None and config is not None:
        r_grid = np.arange(n_pixels // 2 + 1)
        true_profile = np.zeros_like(r_grid, dtype=float)
        for E, beta, br in zip(true_params['E_centers'], 
                               true_params['Betas'], 
                               true_params['branching_ratios']):
            r_mm = config.get_expected_radius(E)
            r_px = r_mm / config.pixel_size
            sigma_E = true_params.get('sigma_laser', 0.01)
            sigma_r_px = r_px * sigma_E / (2 * E) if E > 0 else 1.0
            sigma_r_px = max(sigma_r_px, 0.5)
            true_profile += br * np.exp(-((r_grid - r_px)**2) / (2 * sigma_r_px**2))
        
        if np.max(true_profile) > 0:
            true_profile_norm = true_profile / np.max(true_profile)
        else:
            true_profile_norm = true_profile
        ax4.plot(r_grid, true_profile_norm, 'k-', linewidth=2.5, alpha=0.7, label='True (3D)')
    
    # rBasex profile
    if rbasex_metadata is not None:
        r_rb = rbasex_metadata['r_grid']
        I_rb = rbasex_metadata['intensity_profile']
        if np.max(I_rb) > 0:
            I_rb_norm = I_rb / np.max(I_rb)
        else:
            I_rb_norm = I_rb
        ax4.plot(r_rb, I_rb_norm, 'b--', linewidth=1.5, label='rBasex')
    
    # PhysicsBasedFitter profile
    if physics_metadata is not None:
        r_phys = physics_metadata['r_grid']
        I_phys = physics_metadata['recon_profile']
        if np.max(I_phys) > 0:
            I_phys_norm = I_phys / np.max(I_phys)
        else:
            I_phys_norm = I_phys
        ax4.plot(r_phys, I_phys_norm, 'r-', linewidth=1.5, label='PhysicsBasedFitter')
    
    ax4.set_xlabel("Radius (pixels)")
    ax4.set_ylabel("Normalized Intensity")
    ax4.set_title("Radial Distribution Comparison (3D Space)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, n_pixels // 2)
    
    # Beta comparison
    ax5 = axes[1, 1]
    
    # True beta
    if true_params is not None and config is not None:
        true_r = []
        true_beta = []
        for E, beta in zip(true_params['E_centers'], true_params['Betas']):
            r_mm = config.get_expected_radius(E)
            r_px = r_mm / config.pixel_size
            true_r.append(r_px)
            true_beta.append(beta)
        ax5.scatter(true_r, true_beta, s=200, c='black', marker='o', 
                   label='True', zorder=10, edgecolors='white', linewidths=2)
    
    # rBasex beta
    if rbasex_params is not None:
        rb_r = [p['r'] for p in rbasex_params]
        rb_beta = [p['beta'] for p in rbasex_params]
        ax5.scatter(rb_r, rb_beta, s=120, c='blue', marker='s', 
                   label='rBasex', zorder=5, edgecolors='white', linewidths=1)
    
    # PhysicsBasedFitter beta
    if physics_params is not None:
        phys_r = [p['r'] for p in physics_params]
        phys_beta = [p['beta'] for p in physics_params]
        ax5.scatter(phys_r, phys_beta, s=100, c='red', marker='^', 
                   label='PhysicsBasedFitter', zorder=5, edgecolors='white', linewidths=1)
    
    ax5.set_xlabel("Radius (pixels)")
    ax5.set_ylabel("Beta (β)")
    ax5.set_title("Anisotropy Parameter Comparison")
    ax5.set_ylim(-1.5, 2.5)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Comparison table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Build comparison table
    table_text = "METHOD COMPARISON\n"
    table_text += "=" * 50 + "\n\n"
    
    if true_params is not None and config is not None:
        table_text += f"{'Peak':<5} | {'True':<15} | {'rBasex':<15} | {'Physics':<15}\n"
        table_text += "-" * 55 + "\n"
        
        for i, (E, beta, br) in enumerate(zip(true_params['E_centers'], 
                                               true_params['Betas'], 
                                               true_params['branching_ratios'])):
            r_mm = config.get_expected_radius(E)
            r_px = r_mm / config.pixel_size
            
            # Find matching peaks
            rb_match = None
            phys_match = None
            
            if rbasex_params:
                rb_match = min(rbasex_params, 
                              key=lambda x: abs(x.get('energy_eV', x['r']/config.pixel_size*config.vmi_k) - E) if config else abs(x['r'] - r_px),
                              default=None)
            if physics_params:
                phys_match = min(physics_params,
                                key=lambda x: abs(x.get('energy_eV', 0) - E) if config else abs(x['r'] - r_px),
                                default=None)
            
            table_text += f"\nPeak {i+1} (E={E:.2f} eV):\n"
            table_text += f"  β:   {beta:>6.2f}      "
            if rb_match:
                table_text += f"{rb_match['beta']:>6.2f}      "
            else:
                table_text += f"{'N/A':>6}      "
            if phys_match:
                table_text += f"{phys_match['beta']:>6.2f}\n"
            else:
                table_text += f"{'N/A':>6}\n"
            
            table_text += f"  BR:  {br:>6.3f}      "
            if rb_match:
                table_text += f"{rb_match.get('branching_ratio', 0):>6.3f}      "
            else:
                table_text += f"{'N/A':>6}      "
            if phys_match:
                table_text += f"{phys_match.get('branching_ratio', 0):>6.3f}\n"
            else:
                table_text += f"{'N/A':>6}\n"
    
    # Add peak counts
    table_text += "\n" + "-" * 55 + "\n"
    table_text += f"Peaks detected:\n"
    if true_params:
        table_text += f"  True: {len(true_params['E_centers'])}\n"
    if rbasex_params:
        table_text += f"  rBasex: {len(rbasex_params)}\n"
    if physics_params:
        table_text += f"  Physics: {len(physics_params)}\n"
    
    ax6.text(0.02, 0.98, table_text, fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def print_comparison_table(true_params: Dict, physics_params: List[Dict],
                          rbasex_params: List[Dict], config=None) -> None:
    """
    Print a detailed comparison table of reconstruction results.
    
    Compares all four key parameters:
    - Peak position (radius in pixels)
    - Beta (anisotropy parameter)
    - FWHM (peak width in pixels)
    - Relative amplitude (branching ratio)
    
    Args:
        true_params: Dictionary with true simulation parameters
        physics_params: Results from PhysicsBasedFitter
        rbasex_params: Results from rBasex
        config: Config object for energy conversion
    """
    print("\n" + "="*120)
    print("DETAILED METHOD COMPARISON: Peak Position, Beta, FWHM, Relative Amplitude")
    print("="*120)
    
    # Header
    print(f"\n{'Peak':<6} | {'Parameter':<12} | {'True':<12} | {'rBasex':<12} | {'Err(rB)':<10} | "
          f"{'Physics':<12} | {'Err(Ph)':<10}")
    print("-"*120)
    
    # Sort reconstructed params by energy/radius
    rb_sorted = sorted(rbasex_params, key=lambda x: x.get('energy_eV', x['r']))
    phys_sorted = sorted(physics_params, key=lambda x: x.get('energy_eV', x['r']))
    
    # Calculate true FWHM from sigma_laser
    sigma_laser = true_params.get('sigma_laser', 0.015)  # eV
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(
            true_params['E_centers'],
            true_params['Betas'],
            true_params['branching_ratios'])):
        
        # Find matching peaks
        rb_match = rb_sorted[i] if i < len(rb_sorted) else None
        phys_match = phys_sorted[i] if i < len(phys_sorted) else None
        
        # Calculate true radius in pixels
        if config is not None:
            r_mm_true = config.get_expected_radius(E_true)
            r_px_true = r_mm_true / config.pixel_size
            # True FWHM in pixels (from energy spread)
            sigma_r_px = r_px_true * sigma_laser / (2 * E_true) if E_true > 0 else 1.0
            fwhm_true = sigma_r_px * 2.355
        else:
            r_px_true = 0
            fwhm_true = 0
        
        # =====================================================================
        # Row 1: Peak Position (radius in pixels)
        # =====================================================================
        r_rb = rb_match['r'] if rb_match else 0
        r_phys = phys_match['r'] if phys_match else 0
        err_r_rb = abs(r_rb - r_px_true) if rb_match else float('nan')
        err_r_phys = abs(r_phys - r_px_true) if phys_match else float('nan')
        
        print(f"{i+1:<6} | {'Position':<12} | {r_px_true:<12.1f} | {r_rb:<12.1f} | {err_r_rb:<10.1f} | "
              f"{r_phys:<12.1f} | {err_r_phys:<10.1f}")
        
        # =====================================================================
        # Row 2: Beta (anisotropy parameter)
        # =====================================================================
        beta_rb = rb_match['beta'] if rb_match else 0
        beta_phys = phys_match['beta'] if phys_match else 0
        err_beta_rb = abs(beta_rb - beta_true) if rb_match else float('nan')
        err_beta_phys = abs(beta_phys - beta_true) if phys_match else float('nan')
        
        print(f"{'':6} | {'Beta':<12} | {beta_true:<12.2f} | {beta_rb:<12.2f} | {err_beta_rb:<10.2f} | "
              f"{beta_phys:<12.2f} | {err_beta_phys:<10.2f}")
        
        # =====================================================================
        # Row 3: FWHM (peak width in pixels)
        # =====================================================================
        fwhm_rb = rb_match.get('fwhm', rb_match.get('sigma', 0) * 2.355) if rb_match else 0
        fwhm_phys = phys_match.get('fwhm', phys_match.get('sigma', 0) * 2.355) if phys_match else 0
        err_fwhm_rb = abs(fwhm_rb - fwhm_true) if rb_match else float('nan')
        err_fwhm_phys = abs(fwhm_phys - fwhm_true) if phys_match else float('nan')
        
        print(f"{'':6} | {'FWHM (px)':<12} | {fwhm_true:<12.2f} | {fwhm_rb:<12.2f} | {err_fwhm_rb:<10.2f} | "
              f"{fwhm_phys:<12.2f} | {err_fwhm_phys:<10.2f}")
        
        # =====================================================================
        # Row 4: Relative Amplitude (branching ratio)
        # =====================================================================
        br_rb = rb_match.get('branching_ratio', 0) if rb_match else 0
        br_phys = phys_match.get('branching_ratio', 0) if phys_match else 0
        err_br_rb = abs(br_rb - br_true) if rb_match else float('nan')
        err_br_phys = abs(br_phys - br_true) if phys_match else float('nan')
        
        print(f"{'':6} | {'Rel. Amp.':<12} | {br_true:<12.3f} | {br_rb:<12.3f} | {err_br_rb:<10.3f} | "
              f"{br_phys:<12.3f} | {err_br_phys:<10.3f}")
        
        print("-"*120)
    
    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\nSUMMARY STATISTICS:")
    print("-"*60)
    
    # Calculate average errors for each method
    n_peaks = len(true_params['E_centers'])
    
    # Initialize error accumulators
    rb_errors = {'position': [], 'beta': [], 'fwhm': [], 'rel_amp': []}
    phys_errors = {'position': [], 'beta': [], 'fwhm': [], 'rel_amp': []}
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(
            true_params['E_centers'],
            true_params['Betas'],
            true_params['branching_ratios'])):
        
        rb_match = rb_sorted[i] if i < len(rb_sorted) else None
        phys_match = phys_sorted[i] if i < len(phys_sorted) else None
        
        if config is not None:
            r_mm_true = config.get_expected_radius(E_true)
            r_px_true = r_mm_true / config.pixel_size
            sigma_r_px = r_px_true * sigma_laser / (2 * E_true) if E_true > 0 else 1.0
            fwhm_true = sigma_r_px * 2.355
        else:
            r_px_true = 0
            fwhm_true = 0
        
        if rb_match:
            rb_errors['position'].append(abs(rb_match['r'] - r_px_true))
            rb_errors['beta'].append(abs(rb_match['beta'] - beta_true))
            rb_errors['fwhm'].append(abs(rb_match.get('fwhm', rb_match.get('sigma', 0) * 2.355) - fwhm_true))
            rb_errors['rel_amp'].append(abs(rb_match.get('branching_ratio', 0) - br_true))
        
        if phys_match:
            phys_errors['position'].append(abs(phys_match['r'] - r_px_true))
            phys_errors['beta'].append(abs(phys_match['beta'] - beta_true))
            phys_errors['fwhm'].append(abs(phys_match.get('fwhm', phys_match.get('sigma', 0) * 2.355) - fwhm_true))
            phys_errors['rel_amp'].append(abs(phys_match.get('branching_ratio', 0) - br_true))
    
    # Print average errors
    print(f"\n{'Parameter':<15} | {'rBasex Avg Err':<18} | {'Physics Avg Err':<18} | {'Better Method':<15}")
    print("-"*75)
    
    for param in ['position', 'beta', 'fwhm', 'rel_amp']:
        rb_avg = np.mean(rb_errors[param]) if rb_errors[param] else float('nan')
        phys_avg = np.mean(phys_errors[param]) if phys_errors[param] else float('nan')
        
        if np.isnan(rb_avg) and np.isnan(phys_avg):
            better = "N/A"
        elif np.isnan(rb_avg):
            better = "Physics"
        elif np.isnan(phys_avg):
            better = "rBasex"
        else:
            better = "rBasex" if rb_avg < phys_avg else "Physics"
        
        param_name = {'position': 'Position (px)', 'beta': 'Beta',
                      'fwhm': 'FWHM (px)', 'rel_amp': 'Rel. Amplitude'}[param]
        print(f"{param_name:<15} | {rb_avg:<18.3f} | {phys_avg:<18.3f} | {better:<15}")
    
    print("-"*75)
    print(f"\nPeaks detected: True={n_peaks}, rBasex={len(rbasex_params)}, Physics={len(physics_params)}")
    print("="*120)


# =============================================================================
# Test function
# =============================================================================
if __name__ == "__main__":
    print("Testing rBasex Reconstruction...")
    
    # Create a simple test image with known parameters
    n = 256
    y, x = np.ogrid[:n, :n]
    center = n // 2
    y, x = y - center, x - center
    r = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    # Create test peaks
    test_peaks = [
        {'r': 40, 'sigma': 3, 'amp': 1.0, 'beta': 1.5},
        {'r': 80, 'sigma': 5, 'amp': 0.7, 'beta': -0.5},
    ]
    
    img_3d = np.zeros_like(r, dtype=float)
    for p in test_peaks:
        radial = p['amp'] * np.exp(-((r - p['r'])**2) / (2 * p['sigma']**2))
        angular = 1 + p['beta'] * P2
        img_3d += radial * angular
    
    # Forward project
    img_proj = abel.Transform(img_3d, method='hansenlaw', direction='forward', verbose=False).transform
    
    # Add some noise
    img_noisy = img_proj + np.random.normal(0, 0.1 * np.max(img_proj), img_proj.shape)
    img_noisy = np.maximum(img_noisy, 0)
    
    # Reconstruct with rBasex
    params, metadata = reconstruct_rbasex(img_noisy, verbose=True)
    
    print("\nTrue parameters:")
    for i, p in enumerate(test_peaks):
        print(f"  Peak {i+1}: r={p['r']}, sigma={p['sigma']}, beta={p['beta']}")
