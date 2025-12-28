"""Run the full orthogonal test suite."""
import sys
import time

# Import all test functions
from test_orthogonal_performance import (
    test_single_peak_positions,
    test_single_peak_beta,
    test_single_peak_sigma,
    test_two_peaks_separation,
    test_two_peaks_beta_combinations,
    test_two_peaks_positions,
    test_three_peaks,
    test_sigma_variations_multipeak,
)

print("="*80)
print("Full Orthogonal Test Suite for V3 Reconstructor")
print("="*80)

start_time = time.time()

all_results = []

# Run all tests
test_functions = [
    ("Test 1: Single Peak Positions", test_single_peak_positions),
    ("Test 2: Single Peak Beta", test_single_peak_beta),
    ("Test 3: Single Peak Sigma", test_single_peak_sigma),
    ("Test 4: Two Peaks Separation", test_two_peaks_separation),
    ("Test 5: Two Peaks Beta", test_two_peaks_beta_combinations),
    ("Test 6: Two Peaks Positions", test_two_peaks_positions),
    ("Test 7: Three Peaks", test_three_peaks),
    ("Test 8: Multi-peak Sigma", test_sigma_variations_multipeak),
]

for name, func in test_functions:
    try:
        results = func()
        all_results.extend(results)
        passed = sum(1 for r in results if r.passed)
        print(f"\n{name}: {passed}/{len(results)} passed")
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")

# Summary
elapsed = time.time() - start_time
total = len(all_results)
passed = sum(1 for r in all_results if r.passed)

print("\n" + "="*80)
print(f"FINAL SUMMARY")
print("="*80)
print(f"Total tests: {total}")
print(f"Passed: {passed} ({100*passed/total:.1f}%)")
print(f"Failed: {total - passed} ({100*(total-passed)/total:.1f}%)")
print(f"Time: {elapsed:.1f} seconds")
print("="*80)

# Analyze failures
print("\nFailed tests analysis:")
for r in all_results:
    if not r.passed:
        print(f"  {r.name}")
        print(f"    Peaks: {r.detected_peaks}/{r.n_peaks}")
        if r.r0_errors:
            print(f"    r0 errors: {[f'{e:.1f}%' for e in r.r0_errors]}")
        if r.sigma_errors:
            print(f"    sigma errors: {[f'{e:.1f}%' for e in r.sigma_errors]}")
        if r.beta_errors:
            print(f"    beta errors: {[f'{e:.2f}' for e in r.beta_errors]}")
