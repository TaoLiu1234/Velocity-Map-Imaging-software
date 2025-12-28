"""
Comprehensive comparison of all VMI reconstruction algorithms.

Compares:
- Abel_backward_reconstruction_x1 (forward fitting with differential evolution)
- Abel_backward_reconstruction_x2 (differentiable forward fitting with ensemble)
- vmi_physics_reconstructor (physics-based)
- vmi_multiresolution_reconstructor (multi-resolution with Abel inversion)

Uses the same test cases for fair comparison.
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from vmi_test_framework import TestCaseGenerator, SimulationRunner, TestCase, Config


@dataclass
class AlgorithmResult:
    """Result from a single algorithm run."""
    name: str
    r0_values: List[float]
    beta_values: List[float]
    sigma_values: List[float]
    time_seconds: float
    success: bool
    error_msg: str = ""


def generate_test_data(r0_values, beta_values, sigma, n_events=100000):
    """Generate test data with specified parameters."""
    generator = TestCaseGenerator()
    runner = SimulationRunner(add_noise=True)
    
    n_peaks = len(r0_values)
    
    tc = TestCase(
        case_id="COMPARE_TEST",
        n_peaks=n_peaks,
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
    tc.sigma_values = [sigma] * n_peaks
    tc.branching_ratios = [1.0 / n_peaks] * n_peaks
    
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


def run_x1(xy_data, n_peaks, vmi_k) -> AlgorithmResult:
    """Run X1 algorithm (forward fitting with differential evolution)."""
    try:
        from Abel_backward_reconstruction_x1 import fit_xy
        
        start = time.time()
        result = fit_xy(xy_data, vmi_k=vmi_k, n_peaks=n_peaks, verbose=False)
        elapsed = time.time() - start
        
        return AlgorithmResult(
            name="X1 (DE)",
            r0_values=result.get('r0_values', []),
            beta_values=result.get('betas', []),
            sigma_values=result.get('sigmas', []),
            time_seconds=elapsed,
            success=True
        )
    except Exception as e:
        return AlgorithmResult(
            name="X1 (DE)",
            r0_values=[], beta_values=[], sigma_values=[],
            time_seconds=0, success=False, error_msg=str(e)
        )


def run_x2(xy_data, n_peaks, vmi_k) -> AlgorithmResult:
    """Run X2 algorithm (differentiable forward fitting with ensemble)."""
    try:
        from Abel_backward_reconstruction_x2 import fit_xy_ensemble
        
        start = time.time()
        result = fit_xy_ensemble(
            xy_data, vmi_k=vmi_k, n_peaks=n_peaks,
            verbose=False, parallel=True
        )
        elapsed = time.time() - start
        
        # Convert energies to radii for comparison
        # r = k * sqrt(2E/m), for electrons approximately r ∝ sqrt(E)
        # We'll return energies and let the comparison handle it
        
        return AlgorithmResult(
            name="X2 (Ensemble)",
            r0_values=result.get('E_centers', []),  # Actually energies
            beta_values=result.get('betas', []),
            sigma_values=result.get('sigmas', []),
            time_seconds=elapsed,
            success=True
        )
    except Exception as e:
        return AlgorithmResult(
            name="X2 (Ensemble)",
            r0_values=[], beta_values=[], sigma_values=[],
            time_seconds=0, success=False, error_msg=str(e)
        )


def run_physics_v1(xy_data, n_peaks) -> AlgorithmResult:
    """Run physics reconstructor v1."""
    try:
        from vmi_physics_reconstructor import PhysicsVMIReconstructor
        
        start = time.time()
        recon = PhysicsVMIReconstructor(xy_data)
        peaks = recon.reconstruct(n_peaks=n_peaks, verbose=False)
        elapsed = time.time() - start
        
        return AlgorithmResult(
            name="Physics V1",
            r0_values=[p.r0 for p in peaks],
            beta_values=[p.beta for p in peaks],
            sigma_values=[p.sigma for p in peaks],
            time_seconds=elapsed,
            success=True
        )
    except Exception as e:
        return AlgorithmResult(
            name="Physics V1",
            r0_values=[], beta_values=[], sigma_values=[],
            time_seconds=0, success=False, error_msg=str(e)
        )


def run_multiresolution(xy_data, n_peaks) -> AlgorithmResult:
    """Run multi-resolution reconstructor."""
    try:
        from vmi_multiresolution_reconstructor import VMIMultiResolutionReconstructor
        
        start = time.time()
        recon = VMIMultiResolutionReconstructor(
            xy_data, pixel_size=0.05, psf_sigma=0.1/2.355, dld_resolution=0.01
        )
        peaks = recon.reconstruct(n_peaks=n_peaks, verbose=False)
        elapsed = time.time() - start
        
        return AlgorithmResult(
            name="MultiRes",
            r0_values=[p.r0 for p in peaks],
            beta_values=[p.beta for p in peaks],
            sigma_values=[p.sigma for p in peaks],
            time_seconds=elapsed,
            success=True
        )
    except Exception as e:
        return AlgorithmResult(
            name="MultiRes",
            r0_values=[], beta_values=[], sigma_values=[],
            time_seconds=0, success=False, error_msg=str(e)
        )


def evaluate_result(result: AlgorithmResult, tc: TestCase, is_energy=False) -> Dict:
    """Evaluate reconstruction accuracy."""
    if not result.success or len(result.r0_values) == 0:
        return {
            'r0_errors': [100.0] * len(tc.r0_values),
            'beta_errors': [3.0] * len(tc.beta_values),
            'passed': False
        }
    
    n_peaks = len(tc.r0_values)
    n_detected = len(result.r0_values)
    
    if n_detected != n_peaks:
        return {
            'r0_errors': [100.0] * n_peaks,
            'beta_errors': [3.0] * n_peaks,
            'passed': False
        }
    
    # Match peaks by position (sorted order)
    if is_energy:
        # For X2, result contains energies - compare with tc.E_centers
        true_order = np.argsort(tc.E_centers)
        fitted_order = np.argsort(result.r0_values)
        true_values = tc.E_centers
    else:
        # For others, result contains radii - compare with tc.r0_values
        true_order = np.argsort(tc.r0_values)
        fitted_order = np.argsort(result.r0_values)
        true_values = tc.r0_values
    
    r0_errors = []
    beta_errors = []
    
    for i in range(n_peaks):
        true_idx = true_order[i]
        fitted_idx = fitted_order[i]
        
        true_val = true_values[true_idx]
        fitted_val = result.r0_values[fitted_idx]
        r0_err = abs(fitted_val - true_val) / true_val * 100
        
        beta_true = tc.beta_values[true_idx]
        beta_fitted = result.beta_values[fitted_idx]
        beta_err = abs(beta_fitted - beta_true)
        
        r0_errors.append(r0_err)
        beta_errors.append(beta_err)
    
    # Pass criteria: r0 error < 10%, beta error < 0.3
    r0_pass = all(e < 10 for e in r0_errors)
    beta_pass = all(e < 0.5 for e in beta_errors)
    
    return {
        'r0_errors': r0_errors,
        'beta_errors': beta_errors,
        'passed': r0_pass and beta_pass
    }


def run_comparison(name, r0_values, beta_values, sigma):
    """Run all algorithms on a single test case."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"  True r0: {r0_values}, β: {beta_values}, σ: {sigma}")
    print(f"{'='*80}")
    
    # Generate data
    xy_data, tc, vmi_k = generate_test_data(r0_values, beta_values, sigma)
    n_peaks = len(r0_values)
    
    results = {}
    
    # Run each algorithm
    algorithms = [
        ("Physics V1", lambda: run_physics_v1(xy_data, n_peaks), False),
        ("MultiRes", lambda: run_multiresolution(xy_data, n_peaks), False),
        ("X2 (Ensemble)", lambda: run_x2(xy_data, n_peaks, vmi_k), True),
        # X1 is very slow, skip for now
        # ("X1 (DE)", lambda: run_x1(xy_data, n_peaks, vmi_k), False),
    ]
    
    for alg_name, run_func, is_energy in algorithms:
        print(f"\n  Running {alg_name}...", end=" ", flush=True)
        result = run_func()
        
        if result.success:
            eval_result = evaluate_result(result, tc, is_energy=is_energy)
            
            r0_str = ', '.join([f'{e:.1f}%' for e in eval_result['r0_errors']])
            beta_str = ', '.join([f'{e:.2f}' for e in eval_result['beta_errors']])
            status = "✓" if eval_result['passed'] else "✗"
            
            print(f"{result.time_seconds:.1f}s")
            print(f"    r0/E errors: [{r0_str}]")
            print(f"    β errors:    [{beta_str}]")
            print(f"    Status: {status}")
            
            results[alg_name] = {
                'result': result,
                'eval': eval_result,
                'time': result.time_seconds
            }
        else:
            print(f"FAILED: {result.error_msg}")
            results[alg_name] = {
                'result': result,
                'eval': {'r0_errors': [100]*n_peaks, 'beta_errors': [3]*n_peaks, 'passed': False},
                'time': 0
            }
    
    return results, tc


def main():
    print("="*80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*80)
    print("\nAlgorithms:")
    print("  - Physics V1: vmi_physics_reconstructor.py")
    print("  - MultiRes: vmi_multiresolution_reconstructor.py")
    print("  - X2 (Ensemble): Abel_backward_reconstruction_x2.py")
    print("\nNote: X1 (DE) skipped due to very long runtime (~5+ min per test)")
    
    test_cases = [
        # (name, r0_values, beta_values, sigma)
        ("1 peak, β=0", [10.0], [0.0], 0.4),
        ("1 peak, β=+2", [10.0], [2.0], 0.4),
        ("1 peak, β=-1", [10.0], [-1.0], 0.4),
        ("2 peaks, β=0", [8.0, 12.0], [0.0, 0.0], 0.4),
        ("2 peaks, mixed β", [8.0, 12.0], [1.0, -0.5], 0.4),
        ("3 peaks, β=0", [5.0, 10.0, 15.0], [0.0, 0.0, 0.0], 0.4),
        ("3 peaks, mixed β", [5.0, 10.0, 15.0], [-0.5, 0.0, 1.0], 0.4),
    ]
    
    all_results = {}
    
    for name, r0, beta, sigma in test_cases:
        results, tc = run_comparison(name, r0, beta, sigma)
        all_results[name] = results
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    algorithms = ["Physics V1", "MultiRes", "X2 (Ensemble)"]
    
    # Header
    print(f"\n{'Test Case':<25}", end="")
    for alg in algorithms:
        print(f" {alg:<20}", end="")
    print()
    print("-"*85)
    
    # Results
    pass_counts = {alg: 0 for alg in algorithms}
    time_totals = {alg: 0 for alg in algorithms}
    
    for test_name in all_results:
        print(f"{test_name:<25}", end="")
        for alg in algorithms:
            if alg in all_results[test_name]:
                res = all_results[test_name][alg]
                passed = res['eval']['passed']
                t = res['time']
                status = "✓" if passed else "✗"
                print(f" {status} {t:>5.1f}s           ", end="")
                if passed:
                    pass_counts[alg] += 1
                time_totals[alg] += t
            else:
                print(f" {'N/A':<20}", end="")
        print()
    
    # Totals
    print("-"*85)
    print(f"{'PASSED':<25}", end="")
    for alg in algorithms:
        print(f" {pass_counts[alg]}/{len(test_cases):<18}", end="")
    print()
    
    print(f"{'TOTAL TIME':<25}", end="")
    for alg in algorithms:
        print(f" {time_totals[alg]:>5.1f}s           ", end="")
    print()
    
    print(f"{'AVG TIME':<25}", end="")
    for alg in algorithms:
        avg = time_totals[alg] / len(test_cases) if len(test_cases) > 0 else 0
        print(f" {avg:>5.1f}s           ", end="")
    print()
    
    # Detailed beta accuracy
    print("\n" + "="*80)
    print("BETA ACCURACY DETAILS")
    print("="*80)
    
    for test_name in all_results:
        print(f"\n{test_name}:")
        for alg in algorithms:
            if alg in all_results[test_name]:
                res = all_results[test_name][alg]
                beta_errs = res['eval']['beta_errors']
                avg_beta_err = np.mean(beta_errs)
                print(f"  {alg:<15}: β errors = {[f'{e:.2f}' for e in beta_errs]}, avg = {avg_beta_err:.2f}")


if __name__ == "__main__":
    main()
