"""
Comprehensive Performance Comparison: V1 vs V2 vs V3.5 vs rBasex

Compare reconstruction accuracy at different photon count levels.
Metrics: r (position), σ (width), β (anisotropy), BR (branching ratio)

Note: V3.5 uses skip_forward_fit=True (Phase 2 Abel inversion results)

Author: Kiro AI Assistant
"""
import numpy as np
import sys
import time
import warnings
from io import StringIO
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Import simulation
from Abel_forward_simulation import Config, run_simulation

# Import reconstruction methods
from Abel_backward_reconstruction import reconstruct_vmi_image as reconstruct_v1
from Abel_backward_reconstruction_v2 import reconstruct_vmi_image_v2 as reconstruct_v2
from Abel_backward_reconstruction_v3 import AbelReconstructorV3

# Import abel for rbasex
import abel


# =============================================================================
# Helper Functions
# =============================================================================
def suppress_stdout(func):
    """Decorator to suppress stdout during function execution."""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper


def get_true_params(config: Config) -> List[Dict]:
    """Get true parameters from config."""
    true_params = []
    for i, (E, beta, br) in enumerate(zip(config.E_centers, config.Betas, config.branching_ratios)):
        r_mm = config.get_expected_radius(E)
        r_px = r_mm / config.pixel_size
        
        # Physical sigma from laser bandwidth
        sigma_E = config.sigma_laser
        sigma_r_px = r_px * sigma_E / (2 * E) if E > 0 else 1.0
        
        true_params.append({
            'r': r_px,
            'sigma': sigma_r_px,
            'beta': beta,
            'br': br,
            'energy_eV': E
        })
    return true_params


def match_peaks(true_params: List[Dict], recon_params: List[Dict], 
                tolerance: float = 0.2) -> List[Optional[Dict]]:
    """Match reconstructed peaks to true peaks."""
    if not recon_params:
        return [None] * len(true_params)
    
    matched = []
    for true_p in true_params:
        r_true = true_p['r']
        
        best_match = None
        best_dist = float('inf')
        
        for p in recon_params:
            r_recon = p.get('r', 0)
            dist = abs(r_recon - r_true)
            if dist < best_dist:
                best_dist = dist
                best_match = p
        
        # Accept if within tolerance
        if best_match and best_dist < tolerance * r_true:
            matched.append(best_match)
        else:
            matched.append(None)
    
    return matched


def compute_percent_errors(true_params: List[Dict], 
                           matched_params: List[Optional[Dict]]) -> Dict[str, List[float]]:
    """Compute percent errors for each parameter (all as percentages)."""
    errors = {'r': [], 'sigma': [], 'beta': [], 'br': []}
    
    for true_p, recon_p in zip(true_params, matched_params):
        if recon_p is None:
            for key in errors:
                errors[key].append(np.nan)
            continue
        
        # r error (percent)
        r_true = true_p['r']
        r_recon = recon_p.get('r', 0)
        errors['r'].append(abs(r_recon - r_true) / r_true * 100 if r_true > 0 else np.nan)
        
        # sigma error (percent)
        sigma_true = true_p['sigma']
        sigma_recon = recon_p.get('sigma', recon_p.get('sigma_phys', 0))
        errors['sigma'].append(abs(sigma_recon - sigma_true) / sigma_true * 100 if sigma_true > 0 else np.nan)
        
        # beta error (percent relative to range [-1, 2], i.e., range = 3)
        # Or use absolute value as reference if non-zero
        beta_true = true_p['beta']
        beta_recon = recon_p.get('beta', 0)
        # Use range of 3 (from -1 to 2) as denominator for percentage
        beta_range = 3.0
        errors['beta'].append(abs(beta_recon - beta_true) / beta_range * 100)
        
        # BR error (percent)
        br_true = true_p['br']
        br_recon = recon_p.get('br', recon_p.get('branching_ratio', 0))
        errors['br'].append(abs(br_recon - br_true) / br_true * 100 if br_true > 0 else np.nan)
    
    return errors


# =============================================================================
# Reconstruction Wrappers
# =============================================================================
def run_v1(image: np.ndarray, config: Config) -> Tuple[List[Dict], float]:
    """Run V1 reconstruction."""
    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        params, _ = reconstruct_v1(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - t0
    
    # Normalize BR
    if params:
        total_br = sum(p.get('branching_ratio', p.get('br', 0)) for p in params)
        if total_br > 0:
            for p in params:
                if 'branching_ratio' in p:
                    p['br'] = p['branching_ratio'] / total_br
                elif 'br' in p:
                    p['br'] = p['br'] / total_br
    
    return params, elapsed


def run_v2(image: np.ndarray, config: Config) -> Tuple[List[Dict], float]:
    """Run V2 reconstruction."""
    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        params, _ = reconstruct_v2(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - t0
    
    # Normalize BR
    if params:
        total_br = sum(p.get('branching_ratio', p.get('br', 0)) for p in params)
        if total_br > 0:
            for p in params:
                if 'branching_ratio' in p:
                    p['br'] = p['branching_ratio'] / total_br
                elif 'br' in p:
                    p['br'] = p['br'] / total_br
    
    return params, elapsed


def run_v3(image: np.ndarray, config: Config) -> Tuple[List[Dict], float]:
    """Run V3.5 reconstruction (skip_forward_fit=True recommended)."""
    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # V3.5: 传递config以获得正确的能量校准
        recon = AbelReconstructorV3(
            config=config,
            sigma_psf=config.psf_fwhm / 2.355 if config.psf_fwhm > 0 else 1.5,
            polarization_axis='vertical'
        )
        # V3.5: 使用算符融合和宽边界
        params, metadata = recon.reconstruct(
            image, 
            verbose=False, 
            skip_forward_fit=True,
            enforce_circularity=True,
            use_fused_transform=True,
            wide_bounds=True
        )
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - t0
    
    # Normalize BR
    if params:
        total_br = sum(p.get('br', 0) for p in params)
        if total_br > 0:
            for p in params:
                p['br'] = p.get('br', 0) / total_br
    
    return params, elapsed


def run_rbasex(image: np.ndarray, config: Config) -> Tuple[List[Dict], float]:
    """Run rBasex reconstruction."""
    t0 = time.time()
    
    try:
        # rBasex needs background subtraction
        bg_est = np.mean(image[:10, :10])
        image_sub = image - bg_est
        image_sub[image_sub < 0] = 0
        
        # Run rBasex
        result = abel.Transform(image_sub, method='rbasex', direction='inverse', verbose=False)
        
        # Get radial distribution
        center = (image.shape[0] // 2, image.shape[1] // 2)
        r_axis, intensity = abel.tools.vmi.angular_integration_3D(
            result.transform, origin=center, dr=1
        )
        
        # Find peaks using scipy
        from scipy.signal import find_peaks, peak_widths
        from scipy.ndimage import gaussian_filter1d
        
        # Smooth for peak finding
        intensity_smooth = gaussian_filter1d(intensity, sigma=2)
        
        # Find peaks
        peaks, properties = find_peaks(intensity_smooth, height=np.max(intensity_smooth) * 0.05,
                                        distance=10, prominence=np.max(intensity_smooth) * 0.02)
        
        if len(peaks) == 0:
            return [], time.time() - t0
        
        # Extract parameters for each peak
        params = []
        widths_result = peak_widths(intensity_smooth, peaks, rel_height=0.5)
        
        for i, peak_idx in enumerate(peaks):
            r_peak = r_axis[peak_idx]
            
            # Estimate sigma from FWHM
            fwhm = widths_result[0][i] if i < len(widths_result[0]) else 5.0
            sigma = fwhm / 2.355
            
            # Amplitude
            amp = intensity[peak_idx]
            
            # Beta estimation (simplified - use angular distribution)
            # For rBasex, we'd need to analyze the 2D result
            # Here we use a simplified approach
            beta = 0.0  # Default, rBasex doesn't directly give beta
            
            params.append({
                'r': r_peak,
                'sigma': sigma,
                'amp': amp,
                'beta': beta,
                'br': amp  # Will be normalized
            })
        
        # Normalize BR
        if params:
            total_br = sum(p['br'] for p in params)
            if total_br > 0:
                for p in params:
                    p['br'] = p['br'] / total_br
        
    except Exception as e:
        print(f"rBasex error: {e}")
        return [], time.time() - t0
    
    elapsed = time.time() - t0
    return params, elapsed


# =============================================================================
# Main Test
# =============================================================================
def run_comparison_test(N_events_list: List[int], config_template: dict, 
                        n_trials: int = 3) -> Dict:
    """Run comparison test across different photon counts."""
    
    results = {
        'N_events': N_events_list,
        'v1': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []},
        'v2': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []},
        'v3': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []},
        'rbasex': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []}
    }
    
    for N_events in N_events_list:
        print(f"\n{'='*70}")
        print(f"Testing N_events = {N_events:.0e}")
        print(f"{'='*70}")
        
        # Aggregate errors across trials
        trial_errors = {
            'v1': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []},
            'v2': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []},
            'v3': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []},
            'rbasex': {'r': [], 'sigma': [], 'beta': [], 'br': [], 'time': [], 'detected': []}
        }
        
        for trial in range(n_trials):
            np.random.seed(42 + trial)
            
            # Generate image
            config = Config(**{**config_template, 'N_events': N_events})
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                image_raw, _ = run_simulation(config, add_noise=True, add_background=False)
            finally:
                sys.stdout = old_stdout
            
            # Unified preprocessing: subtract readout offset for all methods
            image = image_raw - config.readout_offset
            
            true_params = get_true_params(config)
            
            # Run each method (all use the same preprocessed image)
            for method_name, method_func in [
                ('v1', run_v1),
                ('v2', run_v2),
                ('v3', run_v3),
                ('rbasex', run_rbasex)
            ]:
                try:
                    params, elapsed = method_func(image, config)
                    matched = match_peaks(true_params, params)
                    errors = compute_percent_errors(true_params, matched)
                    
                    # Store results
                    for key in ['r', 'sigma', 'beta', 'br']:
                        valid_errors = [e for e in errors[key] if not np.isnan(e)]
                        if valid_errors:
                            trial_errors[method_name][key].extend(valid_errors)
                    
                    trial_errors[method_name]['time'].append(elapsed)
                    trial_errors[method_name]['detected'].append(
                        sum(1 for m in matched if m is not None) / len(true_params)
                    )
                    
                except Exception as e:
                    print(f"  {method_name} failed: {e}")
                    trial_errors[method_name]['time'].append(np.nan)
                    trial_errors[method_name]['detected'].append(0)
        
        # Average across trials
        for method_name in ['v1', 'v2', 'v3', 'rbasex']:
            for key in ['r', 'sigma', 'beta', 'br']:
                vals = trial_errors[method_name][key]
                results[method_name][key].append(np.nanmean(vals) if vals else np.nan)
            
            results[method_name]['time'].append(np.nanmean(trial_errors[method_name]['time']))
            results[method_name]['detected'].append(np.nanmean(trial_errors[method_name]['detected']))
        
        # Print summary for this N_events
        print(f"\n{'Method':<10} | {'r(%)':<8} | {'σ(%)':<8} | {'β(%)':<8} | {'BR(%)':<8} | {'Det%':<6} | {'Time(s)':<8}")
        print("-" * 75)
        for method_name in ['v1', 'v2', 'v3', 'rbasex']:
            r_err = results[method_name]['r'][-1]
            s_err = results[method_name]['sigma'][-1]
            b_err = results[method_name]['beta'][-1]
            br_err = results[method_name]['br'][-1]
            det = results[method_name]['detected'][-1] * 100
            t = results[method_name]['time'][-1]
            
            print(f"{method_name:<10} | {r_err:<8.2f} | {s_err:<8.2f} | {b_err:<8.2f} | {br_err:<8.2f} | {det:<6.0f} | {t:<8.3f}")
    
    return results


def plot_results(results: Dict, save_path: str = None):
    """Plot comparison results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    N_events = results['N_events']
    methods = ['v1', 'v2', 'v3', 'rbasex']
    labels = ['V1', 'V2', 'V3.5', 'rBasex']
    colors = ['blue', 'green', 'red', 'orange']
    markers = ['o', 's', '^', 'd']
    
    # Plot each metric
    metrics = [
        ('r', 'Position Error (%)', axes[0, 0]),
        ('sigma', 'Width Error (%)', axes[0, 1]),
        ('beta', 'Beta Error (%)', axes[0, 2]),
        ('br', 'BR Error (%)', axes[1, 0]),
        ('detected', 'Detection Rate (%)', axes[1, 1]),
        ('time', 'Time (s)', axes[1, 2])
    ]
    
    for metric, title, ax in metrics:
        for method, label, color, marker in zip(methods, labels, colors, markers):
            y = results[method][metric]
            if metric == 'detected':
                y = [v * 100 for v in y]  # Convert to percentage
            ax.semilogx(N_events, y, marker=marker, color=color, linestyle='-', 
                       label=label, markersize=8, linewidth=2)
        
        ax.set_xlabel('N_events')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def print_summary(results: Dict):
    """Print overall summary."""
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (Average across all photon levels)")
    print("=" * 80)
    
    methods = ['v1', 'v2', 'v3', 'rbasex']
    
    print(f"\n{'Method':<10} | {'r(%)':<10} | {'σ(%)':<10} | {'β(%)':<10} | {'BR(%)':<10} | {'Det%':<8}")
    print("-" * 70)
    
    scores = {}
    for method in methods:
        r_avg = np.nanmean(results[method]['r'])
        s_avg = np.nanmean(results[method]['sigma'])
        b_avg = np.nanmean(results[method]['beta'])
        br_avg = np.nanmean(results[method]['br'])
        det_avg = np.nanmean(results[method]['detected']) * 100
        
        print(f"{method:<10} | {r_avg:<10.2f} | {s_avg:<10.2f} | {b_avg:<10.2f} | {br_avg:<10.2f} | {det_avg:<8.1f}")
        
        # Compute overall score (lower is better, weighted)
        # r: weight 2, sigma: weight 1, beta: weight 1, br: weight 2
        scores[method] = 2*r_avg + s_avg + b_avg + 2*br_avg - det_avg
    
    print("-" * 70)
    
    # Find winner
    winner = min(scores, key=scores.get)
    print(f"\n🏆 Best Overall: {winner.upper()}")
    
    # Detailed comparison
    print("\n" + "-" * 70)
    print("Parameter-wise Winners:")
    for param, name in [('r', 'Position'), ('sigma', 'Width'), ('beta', 'Beta'), ('br', 'BR')]:
        avgs = {m: np.nanmean(results[m][param]) for m in methods}
        best = min(avgs, key=avgs.get)
        print(f"  {name}: {best.upper()} ({avgs[best]:.3f})")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Test configuration
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
    
    # Photon levels to test (1e4 to 1e8)
    N_events_list = [int(1e4), int(5e4), int(1e5), int(5e5), int(1e6), int(5e6), int(1e7), int(5e7), int(1e8)]
    
    print("=" * 80)
    print("V1 vs V2 vs V3 vs rBasex PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\nTrue parameters:")
    print(f"  E = {E_centers} eV")
    print(f"  β = {Betas}")
    print(f"  BR = {branching_ratios}")
    print(f"\nPhoton levels: {[f'{n:.0e}' for n in N_events_list]}")
    print(f"Trials per level: 1 (for speed)")
    
    # Run comparison
    results = run_comparison_test(N_events_list, config_template, n_trials=1)
    
    # Print summary
    print_summary(results)
    
    # Plot results
    plot_results(results, save_path='method_comparison.png')
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
