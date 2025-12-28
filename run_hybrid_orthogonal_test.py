"""
Comprehensive Orthogonal Test for Hybrid VMI Reconstructor

Tests across all parameter dimensions:
1. Number of peaks: 1, 2, 3, 4, 5
2. Radial position: inner (3mm), middle (10mm), outer (17mm)
3. Beta values: -1, -0.5, 0, 0.5, 1, 1.5, 2
4. Peak width (sigma): narrow (0.2mm), medium (0.4mm), wide (0.8mm)
5. Peak separation: close (2σ), medium (4σ), well (6σ)
6. Event counts: 1e4, 1e5, 1e6
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from vmi_test_framework import TestCaseGenerator, SimulationRunner, TestCase, Config
from vmi_hybrid_reconstructor import HybridVMIReconstructor, HybridConfig, HybridPeakResult

# Constants
N_EVENTS = int(1e5)  # Use 100k for faster testing


@dataclass
class TestResult:
    """Single test result"""
    name: str
    n_peaks: int
    detected_peaks: int
    r0_errors: List[float]
    beta_errors: List[float]
    sigma_errors: List[float]
    time_seconds: float
    passed: bool


def create_test_data(r0_values: List[float], beta_values: List[float], 
                     sigma: float, n_events: int = None) -> Tuple[np.ndarray, TestCase, float]:
    """Create test data."""
    if n_events is None:
        n_events = N_EVENTS
    
    generator = TestCaseGenerator()
    runner = SimulationRunner(add_noise=True)
    
    tc = TestCase(
        case_id="HYBRID_ORTHO",
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
        img_res=512,
        pixel_size=0.05,
        psf_fwhm=0.1,
        dld_resolution=0.01,
        dark_rate=0.0,
        readout_sigma=0.0,
        readout_offset=0.0,
        bg_rate=0.0,
    )
    
    xy_data, _ = runner.run(config, tc)
    return xy_data, tc, generator.vmi_k


def evaluate_result(peaks: List[HybridPeakResult], tc: TestCase,
                    tolerance_r0: float = 5.0, tolerance_beta: float = 0.3) -> TestResult:
    """Evaluate reconstruction result."""
    n_true = len(tc.r0_values)
    n_detected = len(peaks)
    
    if n_detected != n_true:
        return TestResult(
            name="", n_peaks=n_true, detected_peaks=n_detected,
            r0_errors=[100.0]*n_true, beta_errors=[3.0]*n_true,
            sigma_errors=[100.0]*n_true, time_seconds=0, passed=False
        )
    
    # Match peaks by r0 order
    true_order = np.argsort(tc.r0_values)
    fitted_r0 = [p.r0 for p in peaks]
    fitted_order = np.argsort(fitted_r0)
    
    r0_errors = []
    beta_errors = []
    sigma_errors = []
    
    for i in range(n_true):
        true_idx = true_order[i]
        fitted_idx = fitted_order[i]
        
        r0_true = tc.r0_values[true_idx]
        r0_fitted = peaks[fitted_idx].r0
        r0_err = abs(r0_fitted - r0_true) / r0_true * 100
        
        beta_true = tc.beta_values[true_idx]
        beta_fitted = peaks[fitted_idx].beta
        beta_err = abs(beta_fitted - beta_true)
        
        sigma_true = tc.sigma_values[true_idx]
        sigma_fitted = peaks[fitted_idx].sigma
        sigma_err = abs(sigma_fitted - sigma_true) / sigma_true * 100
        
        r0_errors.append(r0_err)
        beta_errors.append(beta_err)
        sigma_errors.append(sigma_err)
    
    passed = all(e < tolerance_r0 for e in r0_errors) and all(e < tolerance_beta for e in beta_errors)
    
    return TestResult(
        name="", n_peaks=n_true, detected_peaks=n_detected,
        r0_errors=r0_errors, beta_errors=beta_errors,
        sigma_errors=sigma_errors, time_seconds=0, passed=passed
    )


def run_test(name: str, r0_values: List[float], beta_values: List[float],
             sigma: float, n_events: int = None) -> TestResult:
    """Run a single test case."""
    xy_data, tc, vmi_k = create_test_data(r0_values, beta_values, sigma, n_events)
    
    config = HybridConfig(verbose=False)
    recon = HybridVMIReconstructor(xy_data, vmi_k=vmi_k, config=config)
    
    start = time.time()
    peaks = recon.reconstruct(n_peaks=len(r0_values), verbose=False)
    elapsed = time.time() - start
    
    result = evaluate_result(peaks, tc)
    result.name = name
    result.time_seconds = elapsed
    
    return result


def print_section(title: str):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def print_result(result: TestResult):
    status = "✓" if result.passed else "✗"
    r0_str = ', '.join([f'{e:.1f}%' for e in result.r0_errors])
    beta_str = ', '.join([f'{e:.2f}' for e in result.beta_errors])
    print(f"  {result.name:<40} r0:[{r0_str:<20}] β:[{beta_str:<15}] {result.time_seconds:.1f}s {status}")


# =============================================================================
# Test Functions
# =============================================================================

def test_single_peak_positions():
    """Test 1: Single peak at different radial positions."""
    print_section("TEST 1: Single Peak - Radial Position Variation")
    print("  Testing r0 = 3mm (inner), 10mm (middle), 17mm (outer)")
    
    results = []
    for name, r0 in [("Inner (3mm)", 3.0), ("Middle (10mm)", 10.0), ("Outer (17mm)", 17.0)]:
        result = run_test(name, [r0], [0.0], sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_single_peak_beta():
    """Test 2: Single peak with different beta values."""
    print_section("TEST 2: Single Peak - Beta Variation")
    print("  Testing β = -1, -0.5, 0, 0.5, 1, 1.5, 2")
    
    results = []
    for beta in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        result = run_test(f"β={beta:+.1f}", [10.0], [beta], sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_single_peak_sigma():
    """Test 3: Single peak with different widths."""
    print_section("TEST 3: Single Peak - Width (Sigma) Variation")
    print("  Testing σ = 0.2mm (narrow), 0.4mm (medium), 0.8mm (wide)")
    
    results = []
    for name, sigma in [("Narrow (0.2mm)", 0.2), ("Medium (0.4mm)", 0.4), ("Wide (0.8mm)", 0.8)]:
        result = run_test(name, [10.0], [0.0], sigma=sigma)
        results.append(result)
        print_result(result)
    return results


def test_two_peaks_separation():
    """Test 4: Two peaks with different separations."""
    print_section("TEST 4: Two Peaks - Separation Variation")
    print("  Testing separation = 2σ, 4σ, 6σ")
    
    results = []
    sigma = 0.4
    center = 10.0
    
    for name, sep_factor in [("Close (2σ)", 2), ("Medium (4σ)", 4), ("Well (6σ)", 6)]:
        sep = sep_factor * sigma
        r0_1, r0_2 = center - sep/2, center + sep/2
        result = run_test(name, [r0_1, r0_2], [0.0, 0.0], sigma=sigma)
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
        result = run_test(name, r0_values, betas, sigma=0.4)
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
        result = run_test(name, r0s, [0.0, 0.0], sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_three_peaks():
    """Test 7: Three peaks."""
    print_section("TEST 7: Three Peaks - Various Configurations")
    
    results = []
    
    configs = [
        ("Well separated (5, 10, 15mm)", [5.0, 10.0, 15.0], [0.0, 0.0, 0.0]),
        ("Medium separated (7, 10, 13mm)", [7.0, 10.0, 13.0], [0.0, 0.0, 0.0]),
        ("Different betas", [5.0, 10.0, 15.0], [-0.5, 0.0, 1.0]),
        ("Extreme betas", [5.0, 10.0, 15.0], [-1.0, 0.0, 2.0]),
    ]
    
    for name, r0s, betas in configs:
        result = run_test(name, r0s, betas, sigma=0.4)
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
        result = run_test(name, r0s, betas, sigma=0.4)
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
        result = run_test(name, r0s, betas, sigma=0.4)
        results.append(result)
        print_result(result)
    return results


def test_event_counts():
    """Test 10: Different event counts."""
    print_section("TEST 10: Event Count Variation")
    print("  Testing N = 1e4, 1e5, 1e6")
    
    results = []
    
    for n_events in [int(1e4), int(1e5), int(1e6)]:
        name = f"N={n_events:.0e}"
        result = run_test(name, [10.0], [0.0], sigma=0.4, n_events=n_events)
        results.append(result)
        print_result(result)
    
    return results


def test_event_counts_multipeak():
    """Test 11: Event counts with multiple peaks."""
    print_section("TEST 11: Event Count with 2 Peaks")
    
    results = []
    
    for n_events in [int(1e4), int(1e5), int(1e6)]:
        name = f"N={n_events:.0e}"
        result = run_test(name, [8.0, 12.0], [0.5, -0.5], sigma=0.4, n_events=n_events)
        results.append(result)
        print_result(result)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE ORTHOGONAL TEST - HYBRID VMI RECONSTRUCTOR")
    print("="*80)
    
    all_results = []
    total_time = 0
    
    # Run all test suites
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
        total_time += sum(r.time_seconds for r in results)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    n_passed = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    
    print(f"\nTotal tests: {n_total}")
    print(f"Passed: {n_passed} ({100*n_passed/n_total:.1f}%)")
    print(f"Failed: {n_total - n_passed}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per test: {total_time/n_total:.2f}s")
    
    # Breakdown by category
    print("\n" + "-"*80)
    print("BREAKDOWN BY TEST CATEGORY")
    print("-"*80)
    
    # Group results by test category
    categories = {}
    for r in all_results:
        # Extract category from name pattern
        if "Inner" in r.name or "Middle" in r.name or "Outer" in r.name:
            cat = "Position"
        elif "β=" in r.name:
            cat = "Beta"
        elif "Narrow" in r.name or "Medium" in r.name or "Wide" in r.name:
            cat = "Sigma"
        elif "Close" in r.name or "Well" in r.name:
            cat = "Separation"
        elif "N=" in r.name:
            cat = "Event Count"
        else:
            cat = "Multi-peak"
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    for cat, results in categories.items():
        n_pass = sum(1 for r in results if r.passed)
        n_tot = len(results)
        avg_r0 = np.mean([np.mean(r.r0_errors) for r in results])
        avg_beta = np.mean([np.mean(r.beta_errors) for r in results])
        print(f"  {cat:<15}: {n_pass}/{n_tot} passed, avg r0 err={avg_r0:.1f}%, avg β err={avg_beta:.2f}")
    
    # Failed tests
    failed = [r for r in all_results if not r.passed]
    if failed:
        print("\n" + "-"*80)
        print("FAILED TESTS")
        print("-"*80)
        for r in failed:
            r0_str = ', '.join([f'{e:.1f}%' for e in r.r0_errors])
            beta_str = ', '.join([f'{e:.2f}' for e in r.beta_errors])
            print(f"  {r.name}: r0=[{r0_str}], β=[{beta_str}]")
    
    return all_results


if __name__ == "__main__":
    results = main()
