"""
Compare Hybrid VMI Reconstructor vs rBasex Method
==================================================

Tests both methods on the same test cases to compare performance.
- Hybrid: Uses XY scatter data
- rBasex: Uses 2D image

Author: Kiro AI Assistant
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from Abel_forward_simulation import Config, run_simulation, ELECTRON_MASS_AMU, EV_TO_JOULE, AMU_TO_KG
from vmi_hybrid_reconstructor import HybridVMIReconstructor, HybridConfig
from Abel_rbasex_reconstruction import reconstruct_rbasex

# Constants
N_EVENTS = int(1e5)
VMI_K = 0.01  # mm/(m/s)
PIXEL_SIZE = 0.05  # mm
IMG_RES = 512


@dataclass
class ComparisonResult:
    """Result for a single test case"""
    name: str
    n_peaks: int
    
    # Hybrid results
    hybrid_r0_errors: List[float]
    hybrid_beta_errors: List[float]
    hybrid_time: float
    hybrid_passed: bool
    
    # rBasex results
    rbasex_r0_errors: List[float]
    rbasex_beta_errors: List[float]
    rbasex_time: float
    rbasex_passed: bool


def energy_to_radius(E_eV: float, vmi_k: float) -> float:
    """Convert energy (eV) to radius (mm)."""
    mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
    v = np.sqrt(2.0 * E_eV * EV_TO_JOULE / mass_kg)
    return vmi_k * v


def radius_to_energy(r_mm: float, vmi_k: float) -> float:
    """Convert radius (mm) to energy (eV)."""
    mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
    v = r_mm / vmi_k
    return 0.5 * mass_kg * v**2 / EV_TO_JOULE


def create_test_data(r0_values: List[float], beta_values: List[float], 
                     sigma: float, n_events: int = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create test data in both XY and image formats from the SAME simulation.
    
    The XY data and image are generated from the same particle distribution
    to ensure a fair comparison.
    
    Returns:
        (xy_data, image, ground_truth)
    """
    if n_events is None:
        n_events = N_EVENTS
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    n_peaks = len(r0_values)
    E_centers = [radius_to_energy(r, VMI_K) for r in r0_values]
    avg_E = np.mean(E_centers)
    avg_r = np.mean(r0_values)
    sigma_laser = sigma / avg_r * avg_E if avg_r > 0 else 0.01
    
    config = Config(
        E_centers=E_centers,
        Betas=beta_values,
        branching_ratios=[1.0 / n_peaks] * n_peaks,
        N_events=n_events,
        vmi_k=VMI_K,
        sigma_laser=sigma_laser,
        T_beam=0.0,
        tau_lifetimes=0.0,
        photon_energy=0.0,
        target_mass=28.0,
        vol_sigma=(0.0, 0.0, 0.0),
        polarization_vec=[0, 1, 0],
        img_res=IMG_RES,
        pixel_size=PIXEL_SIZE,
        psf_fwhm=0.1,
        dld_resolution=0.01,
        dark_rate=0.0,
        readout_sigma=0.0,
        readout_offset=0.0,
        bg_rate=0.0,
    )
    
    # Generate XY data with DLD simulation (PSF + quantization)
    xy_data, metadata = run_simulation(config, add_noise=False, add_background=False, 
                                        output_mode='xy_dld')
    
    # Create image from the SAME XY data by histogramming
    # This ensures both methods see exactly the same data
    center = IMG_RES // 2
    
    # Convert XY (mm) to pixel coordinates
    x_px = xy_data[:, 0] / PIXEL_SIZE + center
    y_px = xy_data[:, 1] / PIXEL_SIZE + center
    
    # Filter out points outside image bounds
    valid = (x_px >= 0) & (x_px < IMG_RES) & (y_px >= 0) & (y_px < IMG_RES)
    x_px = x_px[valid]
    y_px = y_px[valid]
    
    # Create 2D histogram (image)
    image, _, _ = np.histogram2d(
        y_px, x_px,  # Note: y first for image coordinates
        bins=IMG_RES,
        range=[[0, IMG_RES], [0, IMG_RES]]
    )
    
    # Apply Gaussian smoothing to simulate PSF effect on image
    # rBasex works better with some smoothing
    from scipy.ndimage import gaussian_filter
    psf_sigma_px = max(config.psf_sigma / PIXEL_SIZE, 1.0)  # At least 1 pixel
    image = gaussian_filter(image, sigma=psf_sigma_px)
    
    ground_truth = {
        'r0_values': r0_values,
        'beta_values': beta_values,
        'sigma': sigma,
        'E_centers': E_centers,
    }
    
    return xy_data, image, ground_truth


def run_hybrid(xy_data: np.ndarray, n_peaks: int) -> Tuple[List[dict], float]:
    """Run hybrid reconstruction."""
    config = HybridConfig(verbose=False)
    recon = HybridVMIReconstructor(xy_data, vmi_k=VMI_K, config=config)
    
    start = time.time()
    peaks = recon.reconstruct(n_peaks=n_peaks, verbose=False)
    elapsed = time.time() - start
    
    results = []
    for p in sorted(peaks, key=lambda x: x.r0):
        results.append({
            'r0': p.r0,
            'beta': p.beta,
            'sigma': p.sigma
        })
    
    return results, elapsed


def run_rbasex(image: np.ndarray, n_peaks: int) -> Tuple[List[dict], float]:
    """Run rBasex reconstruction."""
    start = time.time()
    params, metadata = reconstruct_rbasex(image, verbose=False)
    elapsed = time.time() - start
    
    # Convert pixel radius to mm
    results = []
    for p in sorted(params, key=lambda x: x['r'])[:n_peaks]:
        r_mm = p['r'] * PIXEL_SIZE
        results.append({
            'r0': r_mm,
            'beta': p['beta'],
            'sigma': p['sigma'] * PIXEL_SIZE
        })
    
    return results, elapsed


def evaluate(results: List[dict], ground_truth: dict, 
             tolerance_r0: float = 0.05, tolerance_beta: float = 0.3) -> Tuple[List[float], List[float], bool]:
    """Evaluate reconstruction results."""
    true_r0s = sorted(ground_truth['r0_values'])
    true_betas = [ground_truth['beta_values'][i] for i in np.argsort(ground_truth['r0_values'])]
    
    n_true = len(true_r0s)
    n_detected = len(results)
    
    if n_detected != n_true:
        return [100.0] * n_true, [3.0] * n_true, False
    
    r0_errors = []
    beta_errors = []
    
    for i in range(n_true):
        r0_true = true_r0s[i]
        r0_fitted = results[i]['r0']
        r0_err = abs(r0_fitted - r0_true) / r0_true * 100
        
        beta_true = true_betas[i]
        beta_fitted = results[i]['beta']
        beta_err = abs(beta_fitted - beta_true)
        
        r0_errors.append(r0_err)
        beta_errors.append(beta_err)
    
    passed = all(e < tolerance_r0 * 100 for e in r0_errors) and all(e < tolerance_beta for e in beta_errors)
    
    return r0_errors, beta_errors, passed


def run_comparison(name: str, r0_values: List[float], beta_values: List[float],
                   sigma: float, n_events: int = None, seed: int = None) -> ComparisonResult:
    """Run comparison test."""
    # Use a random seed based on test name if not provided
    if seed is None:
        seed = hash(name) % (2**31)
    
    xy_data, image, ground_truth = create_test_data(r0_values, beta_values, sigma, n_events, seed)
    n_peaks = len(r0_values)
    
    # Run hybrid
    hybrid_results, hybrid_time = run_hybrid(xy_data, n_peaks)
    hybrid_r0_err, hybrid_beta_err, hybrid_passed = evaluate(hybrid_results, ground_truth)
    
    # Run rBasex
    rbasex_results, rbasex_time = run_rbasex(image, n_peaks)
    rbasex_r0_err, rbasex_beta_err, rbasex_passed = evaluate(rbasex_results, ground_truth)
    
    return ComparisonResult(
        name=name,
        n_peaks=n_peaks,
        hybrid_r0_errors=hybrid_r0_err,
        hybrid_beta_errors=hybrid_beta_err,
        hybrid_time=hybrid_time,
        hybrid_passed=hybrid_passed,
        rbasex_r0_errors=rbasex_r0_err,
        rbasex_beta_errors=rbasex_beta_err,
        rbasex_time=rbasex_time,
        rbasex_passed=rbasex_passed,
    )


def print_result(result: ComparisonResult):
    """Print comparison result."""
    h_r0 = ', '.join([f'{e:.1f}%' for e in result.hybrid_r0_errors])
    h_beta = ', '.join([f'{e:.2f}' for e in result.hybrid_beta_errors])
    h_status = "✓" if result.hybrid_passed else "✗"
    
    r_r0 = ', '.join([f'{e:.1f}%' for e in result.rbasex_r0_errors])
    r_beta = ', '.join([f'{e:.2f}' for e in result.rbasex_beta_errors])
    r_status = "✓" if result.rbasex_passed else "✗"
    
    print(f"\n{result.name}")
    print(f"  Hybrid:  r0=[{h_r0}] β=[{h_beta}] {result.hybrid_time:.2f}s {h_status}")
    print(f"  rBasex:  r0=[{r_r0}] β=[{r_beta}] {result.rbasex_time:.2f}s {r_status}")


# =============================================================================
# Test Cases
# =============================================================================

def test_single_peak():
    """Test single peak at different positions and betas."""
    print("\n" + "="*70)
    print("TEST: Single Peak")
    print("="*70)
    
    results = []
    
    # Different positions
    for name, r0 in [("Inner (3mm)", 3.0), ("Middle (10mm)", 10.0), ("Outer (17mm)", 17.0)]:
        result = run_comparison(name, [r0], [0.0], sigma=0.4)
        results.append(result)
        print_result(result)
    
    # Different betas
    for beta in [-1.0, 0.0, 1.0, 2.0]:
        result = run_comparison(f"β={beta:+.1f}", [10.0], [beta], sigma=0.4)
        results.append(result)
        print_result(result)
    
    return results


def test_two_peaks():
    """Test two peaks."""
    print("\n" + "="*70)
    print("TEST: Two Peaks")
    print("="*70)
    
    results = []
    
    # Well separated
    result = run_comparison("Well separated (8, 12mm)", [8.0, 12.0], [0.0, 0.0], sigma=0.4)
    results.append(result)
    print_result(result)
    
    # Different betas
    result = run_comparison("Different betas", [8.0, 12.0], [1.0, -0.5], sigma=0.4)
    results.append(result)
    print_result(result)
    
    # Extreme betas
    result = run_comparison("Extreme betas (2, -1)", [8.0, 12.0], [2.0, -1.0], sigma=0.4)
    results.append(result)
    print_result(result)
    
    return results


def test_three_peaks():
    """Test three peaks."""
    print("\n" + "="*70)
    print("TEST: Three Peaks")
    print("="*70)
    
    results = []
    
    result = run_comparison("Well separated (5, 10, 15mm)", [5.0, 10.0, 15.0], [0.0, 0.0, 0.0], sigma=0.4)
    results.append(result)
    print_result(result)
    
    result = run_comparison("Different betas", [5.0, 10.0, 15.0], [-0.5, 0.0, 1.0], sigma=0.4)
    results.append(result)
    print_result(result)
    
    return results


def test_four_peaks():
    """Test four peaks."""
    print("\n" + "="*70)
    print("TEST: Four Peaks")
    print("="*70)
    
    results = []
    
    result = run_comparison("Well separated (4, 8, 12, 16mm)", [4.0, 8.0, 12.0, 16.0], [0.0]*4, sigma=0.4)
    results.append(result)
    print_result(result)
    
    return results


def test_five_peaks():
    """Test five peaks."""
    print("\n" + "="*70)
    print("TEST: Five Peaks")
    print("="*70)
    
    results = []
    
    result = run_comparison("Well separated (3, 6, 9, 12, 15mm)", [3.0, 6.0, 9.0, 12.0, 15.0], [0.0]*5, sigma=0.4)
    results.append(result)
    print_result(result)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("COMPARISON: Hybrid VMI Reconstructor vs rBasex")
    print("="*70)
    print(f"Events: {N_EVENTS}, Pixel size: {PIXEL_SIZE}mm, Image: {IMG_RES}x{IMG_RES}")
    
    all_results = []
    
    all_results.extend(test_single_peak())
    all_results.extend(test_two_peaks())
    all_results.extend(test_three_peaks())
    all_results.extend(test_four_peaks())
    all_results.extend(test_five_peaks())
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    hybrid_passed = sum(1 for r in all_results if r.hybrid_passed)
    rbasex_passed = sum(1 for r in all_results if r.rbasex_passed)
    total = len(all_results)
    
    hybrid_time = sum(r.hybrid_time for r in all_results)
    rbasex_time = sum(r.rbasex_time for r in all_results)
    
    print(f"\nTotal tests: {total}")
    print(f"\nHybrid:  {hybrid_passed}/{total} passed ({100*hybrid_passed/total:.1f}%), total time: {hybrid_time:.1f}s")
    print(f"rBasex:  {rbasex_passed}/{total} passed ({100*rbasex_passed/total:.1f}%), total time: {rbasex_time:.1f}s")
    
    # Average errors
    hybrid_r0_avg = np.mean([np.mean(r.hybrid_r0_errors) for r in all_results])
    rbasex_r0_avg = np.mean([np.mean(r.rbasex_r0_errors) for r in all_results])
    hybrid_beta_avg = np.mean([np.mean(r.hybrid_beta_errors) for r in all_results])
    rbasex_beta_avg = np.mean([np.mean(r.rbasex_beta_errors) for r in all_results])
    
    print(f"\nAverage r0 error:   Hybrid={hybrid_r0_avg:.1f}%, rBasex={rbasex_r0_avg:.1f}%")
    print(f"Average beta error: Hybrid={hybrid_beta_avg:.2f}, rBasex={rbasex_beta_avg:.2f}")
    
    return all_results


if __name__ == "__main__":
    results = main()
