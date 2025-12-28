# Requirements Document

## Introduction

本文档定义了 VMI 重建算法综合测试与优化框架的需求规范。目标是通过正交测试设计，系统性地评估 `vmi_reconstruction.py` 算法在各种条件下的性能极限，并改进算法使其：
1. 使用更少的输入参数
2. 自动调整参数以达到最佳性能
3. 在各种条件下都能接近真值

核心设计原则：
1. **正交测试设计**：系统覆盖所有关键参数组合
2. **定量评估**：计算与真值的百分比偏差
3. **自动化参数调整**：减少人工干预
4. **鲁棒性优先**：在极端条件下仍能给出合理结果

## Glossary

- **正交测试 (Orthogonal Testing)**: 使用正交表设计测试用例，以最少的测试覆盖最多的参数组合
- **百分比偏差 (Percentage Deviation)**: |估计值 - 真值| / 真值 × 100%
- **事件数 (N_events)**: 模拟的粒子总数
- **峰分离度 (Peak Separation)**: 相邻峰之间的距离与峰宽的比值
- **信噪比 (SNR)**: 信号强度与噪声水平的比值
- **分支比 (Branching Ratio)**: 各能级的相对强度
- **β 参数**: 角向各向异性参数，范围 [-1, 2]

## Requirements

### Requirement 1: 测试用例生成器

**User Story:** 作为开发者，我希望有一个自动化的测试用例生成器，能够根据正交设计生成覆盖所有关键条件的测试用例。

#### Acceptance Criteria

1. WHEN generating test cases THEN the Test_Generator SHALL support single-peak configurations with varying r0, σ, β, and amplitude
2. WHEN generating test cases THEN the Test_Generator SHALL support multi-peak configurations (2-5 peaks)
3. WHEN generating multi-peak cases THEN the Test_Generator SHALL include well-separated peaks (separation > 5σ)
4. WHEN generating multi-peak cases THEN the Test_Generator SHALL include overlapping peaks (separation < 2σ)
5. WHEN generating test cases THEN the Test_Generator SHALL support event counts from 1e4 (low) to 1e9 (high)
6. WHEN generating test cases THEN the Test_Generator SHALL support β values across full range [-1, 0, 1, 2]
7. WHEN generating test cases THEN the Test_Generator SHALL support varying amplitude ratios (equal, 10:1, 100:1)
8. WHEN generating test cases THEN the Test_Generator SHALL include extreme conditions (very narrow peaks, very wide peaks, edge positions)
9. WHEN generating test cases THEN the Test_Generator SHALL use orthogonal array design to minimize test count while maximizing coverage
10. WHEN a test case is generated THEN the Test_Generator SHALL record all true parameters for later comparison
11. WHEN generating test cases THEN the Test_Generator SHALL support varying radial positions: inner (r0 < 5mm), middle (5-15mm), outer (r0 > 15mm)
12. WHEN generating test cases THEN the Test_Generator SHALL include edge cases where peaks are near detector center (r0 < 2mm) or near detector edge (r0 > 0.9 × r_max)
13. WHEN generating multi-peak cases THEN the Test_Generator SHALL test different radial distributions: all inner, all outer, mixed (inner + outer), evenly spaced

### Requirement 2: 前向模拟接口

**User Story:** 作为开发者，我希望能够使用 `Abel_forward_simulation.py` 生成具有已知真值的测试数据。

#### Acceptance Criteria

1. WHEN simulating test data THEN the Simulator SHALL use Abel_forward_simulation.run_simulation() with output_mode='xy_dld'
2. WHEN simulating test data THEN the Simulator SHALL configure all physical parameters (E_centers, Betas, branching_ratios, N_events)
3. WHEN simulating test data THEN the Simulator SHALL support configurable PSF and noise levels
4. WHEN simulating test data THEN the Simulator SHALL return both XY coordinates and ground truth parameters
5. WHEN simulating with noise THEN the Simulator SHALL support both clean (no noise) and noisy (realistic) conditions
6. WHEN simulating THEN the Simulator SHALL convert energy (eV) to radius (mm) using consistent VMI calibration

### Requirement 3: 算法性能评估

**User Story:** 作为开发者，我希望能够定量评估重建算法的性能，计算各参数与真值的百分比偏差。

#### Acceptance Criteria

1. WHEN evaluating performance THEN the Evaluator SHALL compute percentage error for r0: |r0_est - r0_true| / r0_true × 100%
2. WHEN evaluating performance THEN the Evaluator SHALL compute percentage error for σ: |σ_est - σ_true| / σ_true × 100%
3. WHEN evaluating performance THEN the Evaluator SHALL compute absolute error for β: |β_est - β_true|
4. WHEN evaluating performance THEN the Evaluator SHALL compute percentage error for amplitude ratios
5. WHEN multiple peaks exist THEN the Evaluator SHALL match estimated peaks to true peaks by nearest r0
6. WHEN a peak is missed THEN the Evaluator SHALL record it as 100% error for that peak
7. WHEN a spurious peak is detected THEN the Evaluator SHALL record it as a false positive
8. WHEN all tests complete THEN the Evaluator SHALL generate a summary table with mean, median, and worst-case errors
9. WHEN evaluating THEN the Evaluator SHALL categorize results by test condition (event count, peak separation, β value, etc.)

### Requirement 4: 算法改进 - 自动参数调整

**User Story:** 作为用户，我希望重建算法能够自动调整内部参数，无需手动调参即可获得最佳结果。

#### Acceptance Criteria

1. WHEN reconstructing THEN the Improved_Algorithm SHALL automatically detect the number of peaks
2. WHEN reconstructing THEN the Improved_Algorithm SHALL automatically estimate optimal bin size (dr) based on data density
3. WHEN reconstructing THEN the Improved_Algorithm SHALL automatically adjust peak detection threshold based on SNR
4. WHEN reconstructing THEN the Improved_Algorithm SHALL automatically handle varying event counts (1e4 to 1e9)
5. WHEN reconstructing THEN the Improved_Algorithm SHALL require only XY data as input (no manual parameters)
6. IF the user provides optional hints (e.g., expected number of peaks) THEN the Improved_Algorithm SHALL use them to improve accuracy
7. WHEN reconstructing THEN the Improved_Algorithm SHALL provide confidence estimates for each parameter

### Requirement 5: 算法改进 - 鲁棒性增强

**User Story:** 作为用户，我希望重建算法在各种极端条件下都能给出合理结果。

#### Acceptance Criteria

1. WHEN peaks are heavily overlapping THEN the Improved_Algorithm SHALL use deconvolution or multi-peak fitting
2. WHEN event count is very low (< 1e5) THEN the Improved_Algorithm SHALL use adaptive binning to maintain SNR
3. WHEN event count is very high (> 1e8) THEN the Improved_Algorithm SHALL use efficient algorithms to avoid memory issues
4. WHEN β is at extreme values (-1 or 2) THEN the Improved_Algorithm SHALL correctly extract the angular distribution
5. WHEN amplitude ratios are extreme (> 100:1) THEN the Improved_Algorithm SHALL detect weak peaks without false positives
6. WHEN peaks are near the edge (r0 < 5 or r0 > 0.9 × r_max) THEN the Improved_Algorithm SHALL handle boundary effects
7. WHEN σ is very small (< 0.1) THEN the Improved_Algorithm SHALL not underestimate peak width
8. WHEN σ is very large (> 2.0) THEN the Improved_Algorithm SHALL not overestimate peak width

### Requirement 6: 测试报告生成

**User Story:** 作为开发者，我希望测试框架能够生成详细的性能报告，帮助识别算法的弱点。

#### Acceptance Criteria

1. WHEN tests complete THEN the Reporter SHALL generate a summary table with all test results
2. WHEN tests complete THEN the Reporter SHALL identify conditions where error exceeds 5% for r0 or σ
3. WHEN tests complete THEN the Reporter SHALL identify conditions where error exceeds 0.2 for β
4. WHEN tests complete THEN the Reporter SHALL generate plots showing error vs. each test parameter
5. WHEN tests complete THEN the Reporter SHALL compute overall pass rate (% of tests within tolerance)
6. WHEN tests complete THEN the Reporter SHALL save results to a structured file (JSON or CSV)
7. WHEN comparing algorithm versions THEN the Reporter SHALL show improvement metrics

### Requirement 7: 正交测试设计

**User Story:** 作为开发者，我希望使用正交测试设计来高效覆盖参数空间。

#### Acceptance Criteria

1. WHEN designing tests THEN the Orthogonal_Designer SHALL define factors: n_peaks, event_count, peak_separation, β_range, amplitude_ratio, σ_range, noise_level
2. WHEN designing tests THEN the Orthogonal_Designer SHALL define levels for each factor (at least 3 levels)
3. WHEN designing tests THEN the Orthogonal_Designer SHALL generate an orthogonal array covering all 2-factor interactions
4. WHEN designing tests THEN the Orthogonal_Designer SHALL include corner cases (all factors at extreme values)
5. WHEN designing tests THEN the Orthogonal_Designer SHALL estimate total test count and execution time
6. WHEN tests are too numerous THEN the Orthogonal_Designer SHALL support fractional factorial design

### Requirement 8: 数值稳定性

**User Story:** 作为开发者，我希望改进后的算法在数值上是稳定的。

#### Acceptance Criteria

1. WHEN dividing by a value that could be zero THEN the Improved_Algorithm SHALL add epsilon (1e-10) to the denominator
2. WHEN computing exponentials THEN the Improved_Algorithm SHALL clip arguments to avoid overflow
3. WHEN computing square roots THEN the Improved_Algorithm SHALL ensure arguments are non-negative
4. WHEN the input has very few events THEN the Improved_Algorithm SHALL handle gracefully without crashing
5. WHEN the input has very many events THEN the Improved_Algorithm SHALL handle gracefully without memory overflow

