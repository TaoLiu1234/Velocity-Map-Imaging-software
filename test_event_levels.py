"""
Systematic test: Reconstruction performance at different photon counts.
Compares: Peak position (E), FWHM, Beta, Relative Amplitude
"""
import numpy as np
import sys
from io import StringIO
from typing import List, Dict, Tuple, Optional

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image
from Abel_rbasex_reconstruction import reconstruct_rbasex


def suppress_output(func):
    """Decorator to suppress stdout during function execution."""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper


def get_true_fwhm_px(config: Config, E_center: float) -> float:
    """Calculate true FWHM in pixels from energy spread."""
    r_mm = config.get_expected_radius(E_center)
    r_px = r_mm / config.pixel_size
    sigma_E = config.sigma_laser
    # dE/E = 2 * dr/r for kinetic energy
    sigma_r_px = r_px * sigma_E / (2 * E_center) if E_center > 0 else 1.0
    return sigma_r_px * 2.355


def match_peaks_to_truth(true_E: List[float], params: List[Dict], 
                         config: Config) -> List[Optional[Dict]]:
    """
    Match reconstructed peaks to true energy levels.
    
    Uses radius matching when energy_eV is not available.
    Returns list of matched params (or None if no match).
    """
    if not params:
        return [None] * len(true_E)
    
    matched = []
    for E_true in true_E:
        r_true_mm = config.get_expected_radius(E_true)
        r_true_px = r_true_mm / config.pixel_size
        
        # Find best match by radius (more robust than energy)
        best_match = None
        best_dist = float('inf')
        
        for p in params:
            r_recon = p.get('r', 0)
            dist = abs(r_recon - r_true_px)
            if dist < best_dist:
                best_dist = dist
                best_match = p
        
        # Only accept if within reasonable distance (20% of true radius)
        if best_match and best_dist < 0.2 * r_true_px:
            matched.append(best_match)
        else:
            matched.append(None)
    
    return matched


def run_single_test(N_events: int, config_template: dict, seed: int = 42) -> Tuple[Config, List[Dict], List[Dict]]:
    """
    Run a single simulation and reconstruction test.
    
    Returns: (config, physics_params, rbasex_params)
    """
    np.random.seed(seed)
    
    # Create config
    config = Config(**{**config_template, 'N_events': N_events})
    
    # Generate image (with noise)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        image, _ = run_simulation(config, add_noise=True, add_background=False)
    finally:
        sys.stdout = old_stdout
    
    # PhysicsBasedFitter reconstruction
    sys.stdout = StringIO()
    try:
        physics_params, _ = reconstruct_vmi_image(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    
    # Prepare image for rBasex (subtract offset)
    image_rb = np.maximum(image - config.readout_offset, 0)
    
    # rBasex reconstruction
    sys.stdout = StringIO()
    try:
        rbasex_params, _ = reconstruct_rbasex(image_rb, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    
    return config, physics_params, rbasex_params


def compute_errors(config: Config, physics_params: List[Dict], 
                   rbasex_params: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute reconstruction errors for all parameters.
    
    Returns dict with structure:
    {
        'physics': {'E': [...], 'fwhm': [...], 'beta': [...], 'br': [...]},
        'rbasex': {'E': [...], 'fwhm': [...], 'beta': [...], 'br': [...]}
    }
    """
    errors = {
        'physics': {'E': [], 'fwhm': [], 'beta': [], 'br': []},
        'rbasex': {'E': [], 'fwhm': [], 'beta': [], 'br': []}
    }
    
    # Match peaks
    phys_matched = match_peaks_to_truth(config.E_centers, physics_params, config)
    rb_matched = match_peaks_to_truth(config.E_centers, rbasex_params, config)
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(
            config.E_centers, config.Betas, config.branching_ratios)):
        
        # True values
        r_true_mm = config.get_expected_radius(E_true)
        r_true_px = r_true_mm / config.pixel_size
        fwhm_true = get_true_fwhm_px(config, E_true)
        
        # Physics errors
        p_phys = phys_matched[i]
        if p_phys:
            E_phys = p_phys.get('energy_eV', 0)
            fwhm_phys = p_phys.get('fwhm', p_phys.get('sigma', 0) * 2.355)
            beta_phys = p_phys.get('beta', 0)
            br_phys = p_phys.get('branching_ratio', 0)
            
            errors['physics']['E'].append(abs(E_phys - E_true))
            errors['physics']['fwhm'].append(abs(fwhm_phys - fwhm_true))
            errors['physics']['beta'].append(abs(beta_phys - beta_true))
            errors['physics']['br'].append(abs(br_phys - br_true))
        else:
            errors['physics']['E'].append(float('nan'))
            errors['physics']['fwhm'].append(float('nan'))
            errors['physics']['beta'].append(float('nan'))
            errors['physics']['br'].append(float('nan'))
        
        # rBasex errors
        p_rb = rb_matched[i]
        if p_rb:
            E_rb = p_rb.get('energy_eV', 0)
            fwhm_rb = p_rb.get('fwhm', p_rb.get('sigma', 0) * 2.355)
            beta_rb = p_rb.get('beta', 0)
            br_rb = p_rb.get('branching_ratio', 0)
            
            errors['rbasex']['E'].append(abs(E_rb - E_true))
            errors['rbasex']['fwhm'].append(abs(fwhm_rb - fwhm_true))
            errors['rbasex']['beta'].append(abs(beta_rb - beta_true))
            errors['rbasex']['br'].append(abs(br_rb - br_true))
        else:
            errors['rbasex']['E'].append(float('nan'))
            errors['rbasex']['fwhm'].append(float('nan'))
            errors['rbasex']['beta'].append(float('nan'))
            errors['rbasex']['br'].append(float('nan'))
    
    return errors


def print_test_results(N_events: int, config: Config, 
                       physics_params: List[Dict], rbasex_params: List[Dict]):
    """Print detailed comparison table for a single test."""
    
    phys_matched = match_peaks_to_truth(config.E_centers, physics_params, config)
    rb_matched = match_peaks_to_truth(config.E_centers, rbasex_params, config)
    
    print(f"\n{'Peak':<6} | {'Param':<12} | {'True':<10} | {'Physics':<10} | {'Phys_Err':<10} | "
          f"{'rBasex':<10} | {'rB_Err':<10} | {'Better':<8}")
    print("-" * 100)
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(
            config.E_centers, config.Betas, config.branching_ratios)):
        
        # True values
        r_true_mm = config.get_expected_radius(E_true)
        r_true_px = r_true_mm / config.pixel_size
        fwhm_true = get_true_fwhm_px(config, E_true)
        
        p_phys = phys_matched[i]
        p_rb = rb_matched[i]
        
        # Energy
        E_phys = p_phys.get('energy_eV', 0) if p_phys else 0
        E_rb = p_rb.get('energy_eV', 0) if p_rb else 0
        err_E_phys = abs(E_phys - E_true) if p_phys else float('nan')
        err_E_rb = abs(E_rb - E_true) if p_rb else float('nan')
        better_E = 'Physics' if err_E_phys < err_E_rb else 'rBasex' if err_E_rb < err_E_phys else 'Tie'
        
        print(f"Peak {i+1} | {'Energy(eV)':<12} | {E_true:<10.3f} | {E_phys:<10.3f} | "
              f"{err_E_phys:<10.4f} | {E_rb:<10.3f} | {err_E_rb:<10.4f} | {better_E:<8}")
        
        # FWHM
        fwhm_phys = p_phys.get('fwhm', p_phys.get('sigma', 0) * 2.355) if p_phys else 0
        fwhm_rb = p_rb.get('fwhm', p_rb.get('sigma', 0) * 2.355) if p_rb else 0
        err_fwhm_phys = abs(fwhm_phys - fwhm_true) if p_phys else float('nan')
        err_fwhm_rb = abs(fwhm_rb - fwhm_true) if p_rb else float('nan')
        better_fwhm = 'Physics' if err_fwhm_phys < err_fwhm_rb else 'rBasex' if err_fwhm_rb < err_fwhm_phys else 'Tie'
        
        print(f"       | {'FWHM(px)':<12} | {fwhm_true:<10.2f} | {fwhm_phys:<10.2f} | "
              f"{err_fwhm_phys:<10.2f} | {fwhm_rb:<10.2f} | {err_fwhm_rb:<10.2f} | {better_fwhm:<8}")
        
        # Beta
        beta_phys = p_phys.get('beta', 0) if p_phys else 0
        beta_rb = p_rb.get('beta', 0) if p_rb else 0
        err_beta_phys = abs(beta_phys - beta_true) if p_phys else float('nan')
        err_beta_rb = abs(beta_rb - beta_true) if p_rb else float('nan')
        better_beta = 'Physics' if err_beta_phys < err_beta_rb else 'rBasex' if err_beta_rb < err_beta_phys else 'Tie'
        
        print(f"       | {'Beta':<12} | {beta_true:<10.2f} | {beta_phys:<10.3f} | "
              f"{err_beta_phys:<10.3f} | {beta_rb:<10.3f} | {err_beta_rb:<10.3f} | {better_beta:<8}")
        
        # Branching ratio
        br_phys = p_phys.get('branching_ratio', 0) if p_phys else 0
        br_rb = p_rb.get('branching_ratio', 0) if p_rb else 0
        err_br_phys = abs(br_phys - br_true) if p_phys else float('nan')
        err_br_rb = abs(br_rb - br_true) if p_rb else float('nan')
        better_br = 'Physics' if err_br_phys < err_br_rb else 'rBasex' if err_br_rb < err_br_phys else 'Tie'
        
        print(f"       | {'Rel.Amp':<12} | {br_true:<10.3f} | {br_phys:<10.3f} | "
              f"{err_br_phys:<10.3f} | {br_rb:<10.3f} | {err_br_rb:<10.3f} | {better_br:<8}")
        
        print("-" * 100)


# =============================================================================
# Main Test
# =============================================================================
if __name__ == "__main__":
    # Test configuration template
    E_centers = [0.5, 1.0, 2.0]
    Betas = [1.5, 0.0, -0.5]
    branching_ratios = [0.3, 0.5, 0.2]
    
    E_max = max(E_centers)
    r_max_mm = 20.0
    vmi_k = Config.calculate_vmi_k(E_max_eV=E_max, r_max_mm=r_max_mm)
    
    config_template = {
        'E_centers': E_centers,
        'Betas': Betas,
        'branching_ratios': branching_ratios,
        'vmi_k': vmi_k,
        'sigma_laser': 0.015,
        'T_beam': 10.0,
        'tau_lifetimes': 0.0,
        'photon_energy': 21.2,
        'target_mass': 28.0,
        'vol_sigma': (0.0, 0.0, 0.0),
        'polarization_vec': [0, 1, 0],
        'img_res': 512,
        'pixel_size': 0.1,
        'psf_fwhm': 0.0,
        'dark_rate': 0.1,
        'readout_sigma': 5.0,
        'readout_offset': 100.0,
        'bg_rate': 0.0,
        'bg_energy': 0.15,
        'bg_sigma': 0.08,
    }
    
    # Photon levels to test
    photon_levels = [int(1e5), int(5e5), int(1e6), int(5e6), int(1e7)]
    
    print("=" * 100)
    print("PHOTON COUNT PERFORMANCE TEST")
    print(f"True: E={E_centers} eV, β={Betas}, BR={branching_ratios}")
    print("=" * 100)
    
    for N_events in photon_levels:
        print(f"\n{'='*100}")
        print(f"N_events = {N_events:.0e}")
        print(f"{'='*100}")
        
        config, physics_params, rbasex_params = run_single_test(N_events, config_template)
        
        print(f"\nPeaks detected: Physics={len(physics_params)}, rBasex={len(rbasex_params)}")
        
        print_test_results(N_events, config, physics_params, rbasex_params)
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)
