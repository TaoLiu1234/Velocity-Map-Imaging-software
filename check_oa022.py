"""Check what OA_022 actually is."""
from vmi_test_framework import OrthogonalTestDesigner, TestCaseGenerator

designer = OrthogonalTestDesigner()
test_cases = designer.generate_test_cases()
generator = TestCaseGenerator()
test_cases = generator.fill_test_cases(test_cases)

for tc in test_cases:
    if tc.case_id == "OA_022":
        print(f"OA_022:")
        print(f"  n_peaks: {tc.n_peaks}")
        print(f"  separation: {tc.peak_separation}")
        print(f"  r_position: {tc.r_position}")
        print(f"  amplitude_ratio: {tc.amplitude_ratio}")
        print(f"  beta_range: {tc.beta_range}")
        print(f"  sigma_range: {tc.sigma_range}")
        print(f"  r0_values: {tc.r0_values}")
        print(f"  branching_ratios: {tc.branching_ratios}")
        break
