# Requirements Document

## Introduction

本文档定义了 Abel 反演重建程序 V3 版本的需求规范。V3 版本基于严格的物理模型和统计学原理，实现从 VMI（Velocity Map Imaging）2D 投影图像到 3D 动量分布参数的精确反演。

核心设计原则：
1. **数据预处理（减法）** 与 **前向建模（加法）** 严格分离
2. 每个处理阶段都有明确的物理意义和可验证的检验点
3. 使用加权最小二乘法（WLS）处理减去均值后的数据
4. 最终输出 Branching Ratio (BR) 必须与角向参数 β 解耦

## Glossary

- **Abel 投影**: 3D 球对称分布沿视线方向的积分投影到 2D 平面
- **Abel 逆变换**: 从 2D 投影恢复 3D 分布的数学变换
- **VMI**: Velocity Map Imaging，速度成像技术
- **PSF**: Point Spread Function，点扩散函数
- **β (Beta)**: 角向各向异性参数，范围 [-1, 2]
- **BR (Branching Ratio)**: 分支比，各能级的相对强度
- **σ_phys**: 物理展宽（参与 Abel 投影）
- **σ_sys**: 系统展宽（PSF + 像素化 + 插值，不参与投影，只做 2D 卷积）
- **A_3D**: 3D 空间的振幅密度
- **泊松噪声**: 光子计数的统计涨落，方差等于均值
- **高斯偏置 (Gaussian Pedestal)**: 相机读出噪声的常数偏移
- **WLS**: Weighted Least Squares，加权最小二乘法

## Requirements

### Requirement 1: Phase 0 - 数据净化

**User Story:** 作为物理学家，我希望对原始图像进行净化处理，减去常数背景，使背景区期望值为零，以便后续处理能正确处理负值波动。

#### Acceptance Criteria

1. WHEN the system receives a raw VMI image THEN the system SHALL identify a background region (outer 15% radius) containing no signal
2. WHEN the background region is identified THEN the system SHALL compute the mean pixel value μ_total of that region
3. WHEN μ_total is computed THEN the system SHALL subtract μ_total from the entire image as the constant background offset
4. WHEN background subtraction is complete THEN the system SHALL verify that the background region mean equals zero within tolerance 1e-6
5. WHEN background subtraction is complete THEN the system SHALL compute the background region standard deviation σ_bg and store it for later use in the loss function
6. WHEN the processed image contains negative values THEN the system SHALL preserve those negative values without clipping

### Requirement 2: Phase 1 - 极坐标重采样

**User Story:** 作为物理学家，我希望将笛卡尔坐标图像转换为极坐标矩阵，同时严格保证总计数守恒，以便进行径向和角向分析。

#### Acceptance Criteria

1. WHEN converting from Cartesian to polar coordinates THEN the system SHALL use area-weighted resampling to distribute pixel counts
2. WHEN area-weighted resampling is applied THEN the system SHALL ensure sum(Image_Cartesian) equals sum(Image_Polar) within relative error 1e-6
3. WHEN a single off-center pixel is transformed THEN the system SHALL map it to an arc segment that conserves the total count
4. WHEN the polar matrix is created THEN the system SHALL use configurable angular resolution (default 720 bins for 0.5° resolution)
5. WHEN interpolation is performed THEN the system SHALL use cubic spline (order=3) for smooth reconstruction

### Requirement 3: Phase 2 - 初值提取 (Seed Finding)

**User Story:** 作为物理学家，我希望从极坐标数据中快速提取峰值的初始参数估计，作为后续精细拟合的起点。

#### Acceptance Criteria

1. WHEN the polar image is available THEN the system SHALL compute the angular-integrated 1D radial curve I_2D(R)
2. WHEN I_2D(R) is computed THEN the system SHALL apply 1D Abel inverse transform (Hansen-Law method) to obtain I_3D(r)
3. WHEN I_3D(r) is obtained THEN the system SHALL detect peaks using prominence-based peak finding with adaptive thresholds
4. WHEN peaks are detected THEN the system SHALL extract peak positions r_k with sub-pixel precision using Gaussian fitting
5. WHEN peak positions are found THEN the system SHALL estimate half-width σ_k using FWHM measurement
6. WHEN peak parameters are extracted THEN the system SHALL estimate 3D amplitude A_3D_k from the peak height
7. WHEN r_k is determined THEN the system SHALL fit the angular distribution at r_k to extract β_k
8. WHEN seed parameters are extracted THEN the system SHALL accept up to 5% error as these are initial guesses for Phase 3/4
9. WHEN seed extraction is complete THEN the system SHALL verify that a 1D curve constructed from (r_k, σ_k, β_k) visually covers the experimental curve

### Requirement 4: Phase 3 & 4 - 前向精细拟合

**User Story:** 作为物理学家，我希望通过前向模型拟合来精确确定所有峰值参数，使用物理正确的卷积链和统计学正确的损失函数。

#### Acceptance Criteria

1. WHEN performing forward fitting THEN the system SHALL construct the model M(x,y) using the analytic Abel projection formula: I_2D(R) ∝ A_3D × sqrt(r × σ) × exp(-(R-r)²/(2σ²))
2. WHEN constructing the forward model THEN the system SHALL convolve with PSF using Gaussian kernel
3. WHEN the forward model is constructed THEN the system SHALL NOT add back the Poisson background mean (since Phase 0 subtracted it)
4. WHEN computing the loss function THEN the system SHALL use Weighted Least Squares: Loss = Σ[(M_xy - D_xy)² / (M_xy + σ_bg²)]
5. WHEN computing the loss function THEN the system SHALL include L1 regularization λΣ|A_k| to suppress spurious peaks
6. WHEN fitting is complete THEN the system SHALL verify that the residual image (D - M) shows no visible ring structures
7. WHEN λ is increased THEN the system SHALL observe that weak spurious peaks disappear (false peak suppression test)
8. WHEN fitting parameters THEN the system SHALL constrain A_3D to be non-negative
9. WHEN fitting parameters THEN the system SHALL allow sub-pixel precision for r (not forced to integers)
10. WHEN fitting parameters THEN the system SHALL distinguish σ_phys (participates in projection) from σ_sys (2D convolution only)

### Requirement 5: Phase 5 - Branching Ratio 计算

**User Story:** 作为物理学家，我希望从拟合参数计算各能级的分支比，且分支比必须与角向参数 β 解耦。

#### Acceptance Criteria

1. WHEN computing Branching Ratio THEN the system SHALL use the formula: BR_k = A_3D_k × σ_phys_k × r_k²
2. WHEN computing BR THEN the system SHALL interpret the formula as: A_3D × σ for radial integration, r² for 3D Jacobian
3. WHEN β initial values are changed and refitting is performed THEN the system SHALL produce the same BR_k values within 1% tolerance (β-BR decoupling test)
4. IF BR_k changes when β_k changes THEN the system SHALL indicate parameter coupling ambiguity exists in the model
5. WHEN all BR_k are computed THEN the system SHALL normalize them so that Σ BR_k = 1.0

### Requirement 6: 验证测试框架

**User Story:** 作为开发者，我希望有一套完整的验证测试，确保每个处理阶段的正确性，并最终与真值对比计算百分比误差。

#### Acceptance Criteria

1. WHEN running Test 0.1 THEN the system SHALL verify background region mean equals zero after Phase 0
2. WHEN running Test 0.2 THEN the system SHALL verify background σ_bg equals sqrt(μ_total + σ_readout²) within 10% tolerance
3. WHEN running Test 1.1 THEN the system SHALL verify sum conservation between Cartesian and polar images
4. WHEN running Test 1.2 THEN the system SHALL verify single off-center pixel maps to energy-conserving arc segment
5. WHEN running Test 2.1 THEN the system SHALL verify seed parameters produce a curve that covers the experimental data
6. WHEN running Test 4.1 THEN the system SHALL verify residual image shows no ring structures (flat residual test)
7. WHEN running Test 4.2 THEN the system SHALL verify increasing λ suppresses weak peaks (false peak suppression test)
8. WHEN running Test 5.1 THEN the system SHALL verify BR is invariant to β initial value changes (β-BR decoupling test)
9. WHEN running the final validation THEN the system SHALL compute percentage errors for r, σ, β, and BR against known true values
10. WHEN reporting results THEN the system SHALL display errors in a table format similar to test_v2_percent_error.py

### Requirement 7: 解析投影公式实现

**User Story:** 作为物理学家，我希望使用解析的 Abel 投影公式而非数值积分，以提高计算效率和数值稳定性。

#### Acceptance Criteria

1. WHEN computing the 2D projection of a 3D Gaussian shell THEN the system SHALL use the analytic formula: I_2D(R) ∝ A_3D × sqrt(r × σ) × exp(-(R-r)²/(2σ²))
2. WHEN the analytic formula is used THEN the system SHALL handle the case R > r (outside the shell) correctly
3. WHEN the analytic formula is used THEN the system SHALL handle the case R ≈ r (near the shell peak) correctly
4. WHEN computing angular distribution THEN the system SHALL multiply by (1 + β × P2(cos θ)) where P2 is the second Legendre polynomial
5. WHEN P2 is computed THEN the system SHALL use P2(x) = (3x² - 1) / 2

### Requirement 8: 数值稳定性

**User Story:** 作为开发者，我希望程序在各种输入条件下保持数值稳定，避免除零、溢出等问题。

#### Acceptance Criteria

1. WHEN dividing by a value that could be zero THEN the system SHALL add a small epsilon (1e-10) to the denominator
2. WHEN computing exponentials THEN the system SHALL clip arguments to avoid overflow
3. WHEN computing square roots THEN the system SHALL ensure arguments are non-negative
4. WHEN the input image has very low counts THEN the system SHALL handle gracefully without crashing
5. WHEN the input image has very high counts THEN the system SHALL handle gracefully without overflow

### Requirement 9: 正交性能测试 - 寿命展宽 (Lifetime Broadening)

**User Story:** 作为物理学家，我希望算法能正确处理不同激发态寿命导致的洛伦兹展宽（Voigt 线型），以便准确重建真实实验数据。

#### Acceptance Criteria

1. WHEN testing with non-zero lifetime τ THEN the system SHALL correctly reconstruct peak parameters from Voigt profile data (Gaussian + Lorentzian)
2. WHEN lifetime τ = 0 (no Lorentzian broadening) THEN the system SHALL achieve r0 error < 1%, σ error < 5%, β error < 0.1
3. WHEN lifetime τ = 50 fs (significant Lorentzian, γ=0.0066 eV) THEN the system SHALL achieve r0 error < 2%, σ error < 30%, β error < 0.15 (Note: σ tolerance relaxed due to Voigt profile physics)
4. WHEN lifetime τ = 100 fs (moderate Lorentzian, γ=0.0033 eV) THEN the system SHALL achieve r0 error < 2.5%, σ error < 20%, β error < 0.18
5. WHEN lifetime τ = 200 fs (weak Lorentzian, γ=0.0016 eV) THEN the system SHALL achieve r0 error < 3%, σ error < 15%, β error < 0.2
6. WHEN different peaks have different lifetimes (τ₁ ≠ τ₂) THEN the system SHALL correctly extract parameters for each peak independently with cross-talk error < 5%
7. WHEN the orthogonal test suite runs THEN the system SHALL include lifetime as an independent test dimension alongside r0, β, σ, and peak separation

### Requirement 10: 正交测试框架完整性

**User Story:** 作为开发者，我希望有一套完整的正交测试框架，系统地测试算法在所有独立参数维度上的性能。

#### Acceptance Criteria

1. WHEN running orthogonal tests THEN the system SHALL test the following independent dimensions:
   - Number of peaks: 1, 2, 3, 4, 5
   - Radial position: inner (3mm), middle (10mm), outer (17mm)
   - Beta values: -1, -0.5, 0, 0.5, 1, 1.5, 2
   - Peak width (sigma): narrow (0.1mm), medium (0.4mm), wide (1.0mm)
   - Peak separation: close (2σ), medium (4σ), well (6σ)
   - Lifetime: τ = 0 fs, 50 fs, 100 fs, 200 fs
   - Event count: 1e4, 1e5, 1e6, 1e7
2. WHEN a single peak test fails THEN the system SHALL report detailed diagnostics including ground truth vs reconstructed values
3. WHEN a two-peak test fails THEN the system SHALL report which peak(s) failed and the specific error metrics
4. WHEN all orthogonal tests pass THEN the system SHALL be considered validated for production use
