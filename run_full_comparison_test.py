"""
Comprehensive Orthogonal Test: Hybrid vs rBasex
================================================

Runs the same 40+ test cases on BOTH methods to compare performance.
- Hybrid: Uses XY scatter data
- rBasex: Uses 2D image (created from same XY data)

This ensures a fair comparison where both methods see equivalent data.
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from vmi_test_framework import TestCaseGenerator, SimulationRunner, TestCase, Config
from vmi_hybrid_reconstructor import HybridVMIReconstructor, HybridConfig, HybridPeakResult
from Abel_rbasex_reconstruction import reconstruct_rbasex
from scipy.ndimage import gaussian_filter

# Constants
N_EVENTS = int(1e5)
IMG_RES = 512
PIXEL_SIZE = 0.05  # mm


@dataclass
class ComparisonResult:
    """Result for a single test case comparing both methods"""
    name: str
    n_peaks: int
    
    # Hybrid results
    hybrid_r0_errors: List[float]
    hybrid_beta_errors: List[float]
    hybrid_sigma_errors: List[float]
    hybrid_time: float
    hybrid_passed: bool
    
    # rBasex results
    rbasex_r0_errors: List[float]
    rbasex_beta_errors: List[float]
    rbasex_sigma_errors: List[float]
    rbasex_time: float
    rbasex_passed: bool


def create_test_data(r0_values: List[float], beta_values: List[float], 
                     sigma: float, n_events: int = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray, dict, float]:
    """
    Create test data in both XY and image formats from the SAME simulation.
    """
    if n_events is None:
        n_events = N_EVENTS
    
    if seed is not None:
        np.random.seed(seed)
    
    generator = TestCaseGenerator()
    runner = SimulationRunner(add_noise=True)
    
    tc = TestCase(
        case_id="COMPARISON",
        n_peaks=len(r0_values),
        event_count=n_events,
        peak_separation='well',
        beta_range='mixed',
        amplitude_ratio='equal',
        sigma_range='medium',
        r_position='middle',
        noise_level='low'
    )
    
    tc.r0_values = r0_values
    tc.beta_values = beta_values
    tc.sigma_values = [sigma] * len(r0_values)
    tc.branching_ratios = [1.0 / len(r0_values)] * len(r0_values)
    
    E_centers = [generator._r_to_energy(r) for r in r0_values]
    tc.E_centers = E_centers
    
    config = Config(
        E_centers=E_centers,
        Betas=beta_values,
        branching_ratios=tc.branching_ratios,
        N_events=n_events,
        vmi_k=generator.vmi_k,
        sigma_laser=sigma / np.mean(r0_values) * np.mean(E_centers),
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
    
    # Generate XY data
    xy_data, _ = runner.run(config, tc)
    
    # Create image from the SAME XY data
    center = IMG_RES // 2
    x_px = xy_data[:, 0] / PIXEL_SIZE + center
    y_px = xy_data[:, 1] / PIXEL_SIZE + center
    
    valid = (x_px >= 0) & (x_px < IMG_RES) & (y_px >= 0) & (y_px < IMG_RES)
    x_px = x_px[valid]
    y_px = y_px[valid]
    
    image, _, _ = np.histogram2d(
        y_px, x_px,
        bins=IMG_RES,
        range=[[0, IMG_RES], [0, IMG_RES]]
    )
    
    # Apply PSF smoothing
    psf_sigma_px = max(config.psf_sigma / PIXEL_SIZE, 1.0)
    image = gaussian_filter(image, sigma=psf_sigma_px)
    
    ground_truth = {
        'r0_values': r0_values,
        'beta_values': beta_values,
        'sigma': sigma,
    }
    
    return xy_data, image, ground_truth, generator.vmi_k


def run_hybrid(xy_data: np.ndarray, n_peaks: int, vmi_k: float) -> Tuple[List[dict], float]:
    """Run hybrid reconstruction."""
    config = HybridConfig(verbose=False)
    recon = HybridVMIReconstructor(xy_data, vmi_k=vmi_k, config=config)
    
    start = time.time()
    peaks = recon.reconstruct(n_peaks=n_peaks, verbose=False)
    elapsed = time.time() - start
    
    results = []
    for p in sorted(peaks, key=lambda x: x.r0):
        results.append({'r0': p.r0, 'beta': p.beta, 'sigma': p.sigma})
    
    return results, elapsed


def run_rbasex(image: np.ndarray, n_peaks: int) -> Tuple[List[dict], float]:
    """Run rBasex reconstruction."""
    start = time.time()
    params, metadata = reconstruct_rbasex(image, verbose=False)
    elapsed = time.time() - start
    
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
             tolerance_r0: float = 5.0, tolerance_beta: float = 0.3,
             tolerance_sigma: float = 50.0) -> Tuple[List[float], List[float], List[float], bool]:
    """
    Evaluate reconstruction results.
    
    Returns: (r0_errors, beta_errors, sigma_errors, passed)
    """
    true_r0s = sorted(ground_truth['r0_values'])
    sort_idx = np.argsort(ground_truth['r0_values'])
    true_betas = [ground_truth['beta_values'][i] for i in sort_idx]
    true_sigma = ground_truth['sigma']  # Same sigma for all peaks
    
    n_true = len(true_r0s)
    n_detected = len(results)
    
    if n_detected != n_true:
        return [100.0] * n_true, [3.0] * n_true, [100.0] * n_true, False
    
    r0_errors = []
    beta_errors = []
    sigma_errors = []
    
    for i in range(n_true):
        r0_true = true_r0s[i]
        r0_fitted = results[i]['r0']
        r0_err = abs(r0_fitted - r0_true) / r0_true * 100
        
        beta_true = true_betas[i]
        beta_fitted = results[i]['beta']
        beta_err = abs(beta_fitted - beta_true)
        
        sigma_fitted = results[i].get('sigma', true_sigma)
        sigma_err = abs(sigma_fitted - true_sigma) / true_sigma * 100 if true_sigma > 0 else 0
        
        r0_errors.append(r0_err)
        beta_errors.append(beta_err)
        sigma_errors.append(sigma_err)
    
    # Pass criteria: r0 and beta must be within tolerance
    # Sigma is tracked but not used for pass/fail (too hard to estimate accurately)
    passed = all(e < tolerance_r0 for e in r0_errors) and all(e < tolerance_beta for e in beta_errors)
    
    return r0_errors, beta_errors, sigma_errors, passed


def run_comparison(name: str, r0_values: List[float], beta_values: List[float],
                   sigma: float, n_events: int = None) -> ComparisonResult:
    """Run comparison test on both methods."""
    seed = hash(name) % (2**31)
    xy_data, image, ground_truth, vmi_k = create_test_data(r0_values, beta_values, sigma, n_events, seed)
    n_peaks = len(r0_values)
    
    # Run hybrid
    hybrid_results, hybrid_time = run_hybrid(xy_data, n_peaks, vmi_k)
    hybrid_r0_err, hybrid_beta_err, hybrid_sigma_err, hybrid_passed = evaluate(hybrid_results, ground_truth)
    
    # Run rBasex
    rbasex_results, rbasex_time = run_rbasex(image, n_peaks)
    rbasex_r0_err, rbasex_beta_err, rbasex_sigma_err, rbasex_passed = evaluate(rbasex_results, ground_truth)
    
    return ComparisonResult(
        name=name,
        n_peaks=n_peaks,
        hybrid_r0_errors=hybrid_r0_err,
        hybrid_beta_errors=hybrid_beta_err,
        hybrid_sigma_errors=hybrid_sigma_err,
        hybrid_time=hybrid_time,
        hybrid_passed=hybrid_passed,
        rbasex_r0_errors=rbasex_r0_err,
        rbasex_beta_errors=rbasex_beta_err,
        rbasex_sigma_errors=rbasex_sigma_err,
        rbasex_time=rbasex_time,
        rbasex_passed=rbasex_passed,
    )


def print_result(result: ComparisonResult):
    """Print comparison result."""
    h_status = "✓" if result.hybrid_passed else "✗"
    r_status = "✓" if result.rbasex_passed else "✗"
    
    h_r0 = np.mean(result.hybrid_r0_errors)
    h_beta = np.mean(result.hybrid_beta_errors)
    h_sigma = np.mean(result.hybrid_sigma_errors)
    r_r0 = np.mean(result.rbasex_r0_errors)
    r_beta = np.mean(result.rbasex_beta_errors)
    r_sigma = np.mean(result.rbasex_sigma_errors)
    
    # Format: r0%/β/σ%
    print(f"  {result.name:<35} H:{h_r0:>5.1f}%/{h_beta:.2f}/{h_sigma:>5.1f}% {h_status}  R:{r_r0:>5.1f}%/{r_beta:.2f}/{r_sigma:>5.1f}% {r_status}")


def print_section(title: str):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


# =============================================================================
# Test Functions (same as run_hybrid_orthogonal_test.py)
# =============================================================================

def test_single_peak_positions():
    """Test 1: Single peak at different radial positions."""
    print_section("TEST 1: Single Peak - Radial Position")
    results = []
    for name, r0 in [("Inner (3mm)", 3.0), ("Middle (10mm)", 10.0), ("Outer (17mm)", 17.0)]:
        result = run_comparison(name, [r0], [0.0], sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_single_peak_beta():
    """Test 2: Single peak with different beta values."""
    print_section("TEST 2: Single Peak - Beta Variation")
    results = []
    for beta in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        result = run_comparison(f"β={beta:+.1f}", [10.0], [beta], sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_single_peak_sigma():
    """Test 3: Single peak with different widths."""
    print_section("TEST 3: Single Peak - Width (Sigma)")
    results = []
    for name, sigma in [("Narrow (0.2mm)", 0.2), ("Medium (0.4mm)", 0.4), ("Wide (0.8mm)", 0.8)]:
        result = run_comparison(name, [10.0], [0.0], sigma=sigma)
        results.append(result)
        print_result(result)
    return results


def test_two_peaks_separation():
    """Test 4: Two peaks with different separations."""
    print_section("TEST 4: Two Peaks - Separation")
    results = []
    sigma = 0.4
    center = 10.0
    for name, sep_factor in [("Close (2σ)", 2), ("Medium (4σ)", 4), ("Well (6σ)", 6)]:
        sep = sep_factor * sigma
        r0_1, r0_2 = center - sep/2, center + sep/2
        result = run_comparison(name, [r0_1, r0_2], [0.0, 0.0], sigma=sigma)
        results.append(result)
        print_result(result)
    return results


def test_two_peaks_beta():
    """Test 5: Two peaks with different beta combinations."""
    print_section("TEST 5: Two Peaks - Beta Combinations")
    results = []
    r0_values = [8.0, 12.0]
    combos = [
        ("Both zero (0, 0)", [0.0, 0.0]),
        ("Both positive (1, 1.5)", [1.0, 1.5]),
        ("Both negative (-0.5, -1)", [-0.5, -1.0]),
        ("Mixed (+1, -0.5)", [1.0, -0.5]),
        ("Extreme (2, -1)", [2.0, -1.0]),
    ]
    for name, betas in combos:
        result = run_comparison(name, r0_values, betas, sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_two_peaks_positions():
    """Test 6: Two peaks at different radial regions."""
    print_section("TEST 6: Two Peaks - Position Variation")
    results = []
    positions = [
        ("Inner (3, 7mm)", [3.0, 7.0]),
        ("Middle (8, 12mm)", [8.0, 12.0]),
        ("Outer (14, 18mm)", [14.0, 18.0]),
        ("Span inner-middle (5, 10mm)", [5.0, 10.0]),
        ("Span middle-outer (10, 15mm)", [10.0, 15.0]),
    ]
    for name, r0s in positions:
        result = run_comparison(name, r0s, [0.0, 0.0], sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_three_peaks():
    """Test 7: Three peaks."""
    print_section("TEST 7: Three Peaks")
    results = []
    configs = [
        ("Well separated (5, 10, 15mm)", [5.0, 10.0, 15.0], [0.0, 0.0, 0.0]),
        ("Medium separated (7, 10, 13mm)", [7.0, 10.0, 13.0], [0.0, 0.0, 0.0]),
        ("Different betas", [5.0, 10.0, 15.0], [-0.5, 0.0, 1.0]),
        ("Extreme betas", [5.0, 10.0, 15.0], [-1.0, 0.0, 2.0]),
    ]
    for name, r0s, betas in configs:
        result = run_comparison(name, r0s, betas, sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_four_peaks():
    """Test 8: Four peaks."""
    print_section("TEST 8: Four Peaks")
    results = []
    configs = [
        ("Well separated (4, 8, 12, 16mm)", [4.0, 8.0, 12.0, 16.0], [0.0]*4),
        ("Different betas", [4.0, 8.0, 12.0, 16.0], [-0.5, 0.0, 0.5, 1.0]),
    ]
    for name, r0s, betas in configs:
        result = run_comparison(name, r0s, betas, sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_five_peaks():
    """Test 9: Five peaks."""
    print_section("TEST 9: Five Peaks")
    results = []
    configs = [
        ("Well separated (3, 6, 9, 12, 15mm)", [3.0, 6.0, 9.0, 12.0, 15.0], [0.0]*5),
        ("Different betas", [3.0, 6.0, 9.0, 12.0, 15.0], [-1.0, -0.5, 0.0, 0.5, 1.0]),
    ]
    for name, r0s, betas in configs:
        result = run_comparison(name, r0s, betas, sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_event_counts():
    """Test 10: Different event counts."""
    print_section("TEST 10: Event Count Variation")
    results = []
    for n_events in [int(1e4), int(1e5), int(1e6)]:
        name = f"N={n_events:.0e}"
        result = run_comparison(name, [10.0], [0.0], sigma=0.4, n_events=n_events)
        results.append(result)
        print_result(result)
    return results


def test_event_counts_multipeak():
    """Test 11: Event counts with multiple peaks."""
    print_section("TEST 11: Event Count with 2 Peaks")
    results = []
    for n_events in [int(1e4), int(1e5), int(1e6)]:
        name = f"N={n_events:.0e}"
        result = run_comparison(name, [8.0, 12.0], [0.5, -0.5], sigma=0.4, n_events=n_events)
        results.append(result)
        print_result(result)
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*90)
    print("COMPREHENSIVE ORTHOGONAL TEST: HYBRID vs rBasex")
    print("="*90)
    print(f"Events: {N_EVENTS}, Image: {IMG_RES}x{IMG_RES}, Pixel: {PIXEL_SIZE}mm")
    print("Legend: H=Hybrid, R=rBasex, format: r0_err%/beta_err/sigma_err%")
    
    all_results = []
    
    test_functions = [
        test_single_peak_positions,
        test_single_peak_beta,
        test_single_peak_sigma,
        test_two_peaks_separation,
        test_two_peaks_beta,
        test_two_peaks_positions,
        test_three_peaks,
        test_four_peaks,
        test_five_peaks,
        test_event_counts,
        test_event_counts_multipeak,
    ]
    
    for test_func in test_functions:
        results = test_func()
        all_results.extend(results)
    
    # Summary
    print("\n" + "="*90)
    print("FINAL SUMMARY")
    print("="*90)
    
    n_total = len(all_results)
    hybrid_passed = sum(1 for r in all_results if r.hybrid_passed)
    rbasex_passed = sum(1 for r in all_results if r.rbasex_passed)
    
    hybrid_time = sum(r.hybrid_time for r in all_results)
    rbasex_time = sum(r.rbasex_time for r in all_results)
    
    print(f"\nTotal tests: {n_total}")
    print(f"\nHybrid:  {hybrid_passed}/{n_total} passed ({100*hybrid_passed/n_total:.1f}%), time: {hybrid_time:.1f}s")
    print(f"rBasex:  {rbasex_passed}/{n_total} passed ({100*rbasex_passed/n_total:.1f}%), time: {rbasex_time:.1f}s")
    
    # Average errors for all 3 parameters
    hybrid_r0_avg = np.mean([np.mean(r.hybrid_r0_errors) for r in all_results])
    rbasex_r0_avg = np.mean([np.mean(r.rbasex_r0_errors) for r in all_results])
    hybrid_beta_avg = np.mean([np.mean(r.hybrid_beta_errors) for r in all_results])
    rbasex_beta_avg = np.mean([np.mean(r.rbasex_beta_errors) for r in all_results])
    hybrid_sigma_avg = np.mean([np.mean(r.hybrid_sigma_errors) for r in all_results])
    rbasex_sigma_avg = np.mean([np.mean(r.rbasex_sigma_errors) for r in all_results])
    
    print(f"\nAverage Errors:")
    print(f"  r0 (position):  Hybrid={hybrid_r0_avg:.1f}%, rBasex={rbasex_r0_avg:.1f}%")
    print(f"  beta (aniso):   Hybrid={hybrid_beta_avg:.2f}, rBasex={rbasex_beta_avg:.2f}")
    print(f"  sigma (width):  Hybrid={hybrid_sigma_avg:.1f}%, rBasex={rbasex_sigma_avg:.1f}%")
    
    # Failed tests breakdown
    print("\n" + "-"*90)
    print("FAILED TESTS")
    print("-"*90)
    
    hybrid_failed = [r for r in all_results if not r.hybrid_passed]
    rbasex_failed = [r for r in all_results if not r.rbasex_passed]
    
    print(f"\nHybrid failures ({len(hybrid_failed)}):")
    for r in hybrid_failed:
        r0_str = ', '.join([f'{e:.1f}%' for e in r.hybrid_r0_errors])
        beta_str = ', '.join([f'{e:.2f}' for e in r.hybrid_beta_errors])
        sigma_str = ', '.join([f'{e:.0f}%' for e in r.hybrid_sigma_errors])
        print(f"  {r.name}: r0=[{r0_str}], β=[{beta_str}], σ=[{sigma_str}]")
    
    print(f"\nrBasex failures ({len(rbasex_failed)}):")
    for r in rbasex_failed:
        r0_str = ', '.join([f'{e:.1f}%' for e in r.rbasex_r0_errors])
        beta_str = ', '.join([f'{e:.2f}' for e in r.rbasex_beta_errors])
        sigma_str = ', '.join([f'{e:.0f}%' for e in r.rbasex_sigma_errors])
        print(f"  {r.name}: r0=[{r0_str}], β=[{beta_str}], σ=[{sigma_str}]")
    
    return all_results


if __name__ == "__main__":
    results = main()
