"""
Systematic test: Compare V1 vs V2 reconstruction at different photon counts.
V2 includes Phase 0 physical constraints based on laser bandwidth.

Compares: Peak position (E), FWHM, Beta, Relative Amplitude
"""
import numpy as np
import sys
from io import StringIO
from typing import List, Dict, Tuple, Optional

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image as reconstruct_v1
from Abel_backward_reconstruction_v2 import reconstruct_vmi_image_v2 as reconstruct_v2


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
    sigma_r_px = r_px * sigma_E / (2 * E_center) if E_center > 0 else 1.0
    return sigma_r_px * 2.355


def match_peaks_to_truth(true_E: List[float], params: List[Dict], 
                         config: Config) -> List[Optional[Dict]]:
    """Match reconstructed peaks to true energy levels."""
    if not params:
        return [None] * len(true_E)
    
    matched = []
    for E_true in true_E:
        r_true_mm = config.get_expected_radius(E_true)
        r_true_px = r_true_mm / config.pixel_size
        
        best_match = None
        best_dist = float('inf')
        
        for p in params:
            r_recon = p.get('r', 0)
            dist = abs(r_recon - r_true_px)
            if dist < best_dist:
                best_dist = dist
                best_match = p
        
        if best_match and best_dist < 0.2 * r_true_px:
            matched.append(best_match)
        else:
            matched.append(None)
    
    return matched


def run_single_test(N_events: int, config_template: dict, seed: int = 42) -> Tuple[Config, List[Dict], List[Dict]]:
    """
    Run a single simulation and reconstruction test.
    
    Returns: (config, v1_params, v2_params)
    """
    np.random.seed(seed)
    
    config = Config(**{**config_template, 'N_events': N_events})
    
    # Generate image
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        image, _ = run_simulation(config, add_noise=True, add_background=False)
    finally:
        sys.stdout = old_stdout
    
    # V1 reconstruction (original)
    sys.stdout = StringIO()
    try:
        v1_params, _ = reconstruct_v1(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    
    # V2 reconstruction (with Phase 0)
    sys.stdout = StringIO()
    try:
        v2_params, _ = reconstruct_v2(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    
    return config, v1_params, v2_params


def compute_errors(config: Config, v1_params: List[Dict], 
                   v2_params: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    """Compute reconstruction errors for all parameters."""
    errors = {
        'v1': {'E': [], 'fwhm': [], 'beta': [], 'br': []},
        'v2': {'E': [], 'fwhm': [], 'beta': [], 'br': []}
    }
    
    v1_matched = match_peaks_to_truth(config.E_centers, v1_params, config)
    v2_matched = match_peaks_to_truth(config.E_centers, v2_params, config)
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(
            config.E_centers, config.Betas, config.branching_ratios)):
        
        r_true_mm = config.get_expected_radius(E_true)
        r_true_px = r_true_mm / config.pixel_size
        fwhm_true = get_true_fwhm_px(config, E_true)
        
        # V1 errors
        p_v1 = v1_matched[i]
        if p_v1:
            E_v1 = p_v1.get('energy_eV', 0)
            fwhm_v1 = p_v1.get('fwhm', p_v1.get('sigma', 0) * 2.355)
            beta_v1 = p_v1.get('beta', 0)
            br_v1 = p_v1.get('branching_ratio', 0)
            
            errors['v1']['E'].append(abs(E_v1 - E_true))
            errors['v1']['fwhm'].append(abs(fwhm_v1 - fwhm_true))
            errors['v1']['beta'].append(abs(beta_v1 - beta_true))
            errors['v1']['br'].append(abs(br_v1 - br_true))
        else:
            errors['v1']['E'].append(float('nan'))
            errors['v1']['fwhm'].append(float('nan'))
            errors['v1']['beta'].append(float('nan'))
            errors['v1']['br'].append(float('nan'))
        
        # V2 errors
        p_v2 = v2_matched[i]
        if p_v2:
            E_v2 = p_v2.get('energy_eV', 0)
            fwhm_v2 = p_v2.get('fwhm', p_v2.get('sigma', 0) * 2.355)
            beta_v2 = p_v2.get('beta', 0)
            br_v2 = p_v2.get('branching_ratio', 0)
            
            errors['v2']['E'].append(abs(E_v2 - E_true))
            errors['v2']['fwhm'].append(abs(fwhm_v2 - fwhm_true))
            errors['v2']['beta'].append(abs(beta_v2 - beta_true))
            errors['v2']['br'].append(abs(br_v2 - br_true))
        else:
            errors['v2']['E'].append(float('nan'))
            errors['v2']['fwhm'].append(float('nan'))
            errors['v2']['beta'].append(float('nan'))
            errors['v2']['br'].append(float('nan'))
    
    return errors


def print_test_results(N_events: int, config: Config, 
                       v1_params: List[Dict], v2_params: List[Dict]):
    """Print detailed comparison table for a single test."""
    
    v1_matched = match_peaks_to_truth(config.E_centers, v1_params, config)
    v2_matched = match_peaks_to_truth(config.E_centers, v2_params, config)
    
    print(f"\n{'Peak':<6} | {'Param':<12} | {'True':<10} | {'V1':<10} | {'V1_Err':<10} | "
          f"{'V2(Phase0)':<10} | {'V2_Err':<10} | {'Better':<8}")
    print("-" * 105)
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(
            config.E_centers, config.Betas, config.branching_ratios)):
        
        r_true_mm = config.get_expected_radius(E_true)
        r_true_px = r_true_mm / config.pixel_size
        fwhm_true = get_true_fwhm_px(config, E_true)
        
        p_v1 = v1_matched[i]
        p_v2 = v2_matched[i]
        
        # Energy
        E_v1 = p_v1.get('energy_eV', 0) if p_v1 else 0
        E_v2 = p_v2.get('energy_eV', 0) if p_v2 else 0
        err_E_v1 = abs(E_v1 - E_true) if p_v1 else float('nan')
        err_E_v2 = abs(E_v2 - E_true) if p_v2 else float('nan')
        better_E = 'V2' if err_E_v2 < err_E_v1 else 'V1' if err_E_v1 < err_E_v2 else 'Tie'
        
        print(f"Peak {i+1} | {'Energy(eV)':<12} | {E_true:<10.3f} | {E_v1:<10.3f} | "
              f"{err_E_v1:<10.4f} | {E_v2:<10.3f} | {err_E_v2:<10.4f} | {better_E:<8}")
        
        # FWHM
        fwhm_v1 = p_v1.get('fwhm', p_v1.get('sigma', 0) * 2.355) if p_v1 else 0
        fwhm_v2 = p_v2.get('fwhm', p_v2.get('sigma', 0) * 2.355) if p_v2 else 0
        err_fwhm_v1 = abs(fwhm_v1 - fwhm_true) if p_v1 else float('nan')
        err_fwhm_v2 = abs(fwhm_v2 - fwhm_true) if p_v2 else float('nan')
        better_fwhm = 'V2' if err_fwhm_v2 < err_fwhm_v1 else 'V1' if err_fwhm_v1 < err_fwhm_v2 else 'Tie'
        
        print(f"       | {'FWHM(px)':<12} | {fwhm_true:<10.2f} | {fwhm_v1:<10.2f} | "
              f"{err_fwhm_v1:<10.2f} | {fwhm_v2:<10.2f} | {err_fwhm_v2:<10.2f} | {better_fwhm:<8}")
        
        # Beta
        beta_v1 = p_v1.get('beta', 0) if p_v1 else 0
        beta_v2 = p_v2.get('beta', 0) if p_v2 else 0
        err_beta_v1 = abs(beta_v1 - beta_true) if p_v1 else float('nan')
        err_beta_v2 = abs(beta_v2 - beta_true) if p_v2 else float('nan')
        better_beta = 'V2' if err_beta_v2 < err_beta_v1 else 'V1' if err_beta_v1 < err_beta_v2 else 'Tie'
        
        print(f"       | {'Beta':<12} | {beta_true:<10.2f} | {beta_v1:<10.3f} | "
              f"{err_beta_v1:<10.3f} | {beta_v2:<10.3f} | {err_beta_v2:<10.3f} | {better_beta:<8}")
        
        # Branching ratio
        br_v1 = p_v1.get('branching_ratio', 0) if p_v1 else 0
        br_v2 = p_v2.get('branching_ratio', 0) if p_v2 else 0
        err_br_v1 = abs(br_v1 - br_true) if p_v1 else float('nan')
        err_br_v2 = abs(br_v2 - br_true) if p_v2 else float('nan')
        better_br = 'V2' if err_br_v2 < err_br_v1 else 'V1' if err_br_v1 < err_br_v2 else 'Tie'
        
        print(f"       | {'Rel.Amp':<12} | {br_true:<10.3f} | {br_v1:<10.3f} | "
              f"{err_br_v1:<10.3f} | {br_v2:<10.3f} | {err_br_v2:<10.3f} | {better_br:<8}")
        
        print("-" * 105)


def print_summary(all_errors: Dict[int, Dict]) -> None:
    """Print summary statistics across all photon levels."""
    print("\n" + "=" * 100)
    print("SUMMARY: Average Errors Across All Photon Levels")
    print("=" * 100)
    
    # Aggregate errors
    v1_totals = {'E': [], 'fwhm': [], 'beta': [], 'br': []}
    v2_totals = {'E': [], 'fwhm': [], 'beta': [], 'br': []}
    
    for N, errors in all_errors.items():
        for key in ['E', 'fwhm', 'beta', 'br']:
            v1_totals[key].extend([e for e in errors['v1'][key] if not np.isnan(e)])
            v2_totals[key].extend([e for e in errors['v2'][key] if not np.isnan(e)])
    
    print(f"\n{'Parameter':<15} | {'V1 Mean Err':<12} | {'V2 Mean Err':<12} | {'Improvement':<12} | {'Winner':<8}")
    print("-" * 70)
    
    params = [('Energy (eV)', 'E'), ('FWHM (px)', 'fwhm'), ('Beta', 'beta'), ('Rel.Amp', 'br')]
    
    v1_wins = 0
    v2_wins = 0
    
    for name, key in params:
        v1_mean = np.mean(v1_totals[key]) if v1_totals[key] else float('nan')
        v2_mean = np.mean(v2_totals[key]) if v2_totals[key] else float('nan')
        
        if not np.isnan(v1_mean) and not np.isnan(v2_mean) and v1_mean > 0:
            improvement = (v1_mean - v2_mean) / v1_mean * 100
            winner = 'V2' if v2_mean < v1_mean else 'V1'
            if winner == 'V1':
                v1_wins += 1
            else:
                v2_wins += 1
        else:
            improvement = 0
            winner = 'N/A'
        
        print(f"{name:<15} | {v1_mean:<12.4f} | {v2_mean:<12.4f} | {improvement:>+10.1f}% | {winner:<8}")
    
    print("-" * 70)
    print(f"\nOverall Winner: {'V2 (Phase 0)' if v2_wins > v1_wins else 'V1' if v1_wins > v2_wins else 'Tie'}")
    print(f"V1 wins: {v1_wins}, V2 wins: {v2_wins}")


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
    
    # Photon levels to test (including low count scenarios where Phase 0 should help)
    photon_levels = [int(1e4), int(5e4), int(1e5), int(5e5), int(1e6)]
    
    print("=" * 105)
    print("V1 vs V2 (Phase 0) PERFORMANCE COMPARISON")
    print(f"True: E={E_centers} eV, β={Betas}, BR={branching_ratios}")
    print("=" * 105)
    print("\nV2 adds Phase 0 physical constraints:")
    print("  - σ_r = C/r relationship from laser bandwidth")
    print("  - Adaptive denoising based on physical frequency limits")
    print("  - Peak validation against physical sigma limits")
    print("  - Dynamic optimization bounds")
    
    all_errors = {}
    
    for N_events in photon_levels:
        print(f"\n{'='*105}")
        print(f"N_events = {N_events:.0e}")
        print(f"{'='*105}")
        
        config, v1_params, v2_params = run_single_test(N_events, config_template)
        
        print(f"\nPeaks detected: V1={len(v1_params)}, V2={len(v2_params)}")
        
        print_test_results(N_events, config, v1_params, v2_params)
        
        # Store errors for summary
        all_errors[N_events] = compute_errors(config, v1_params, v2_params)
    
    # Print overall summary
    print_summary(all_errors)
    
    print("\n" + "=" * 105)
    print("TEST COMPLETE")
    print("=" * 105)
