# Requirements Document

## Introduction

Multi-resolution deconvolution approach for VMI (Velocity Map Imaging) parameter estimation. This algorithm exploits two key insights:

1. **Binning as Convolution**: Coarse binning is a convolution of fine binning with a box kernel. By analyzing at multiple resolutions and applying deconvolution, we can recover fine-resolution information.

2. **Geometric Shape Fitting**: The curvature and statistical moments (mean, variance, skewness, kurtosis) of the distribution encode the underlying physical parameters. Optimization algorithms can fit these geometric features directly.

The algorithm uses dynamic sampling intervals in both radial (r) and angular (θ) dimensions, adapting to local data density and feature scale.

## Glossary

- **VMI_Reconstructor**: The system that extracts peak parameters from VMI scatter point data
- **Multi_Resolution_Analyzer**: Component that analyzes data at multiple bin sizes
- **Deconvolution_Engine**: Component that recovers fine-resolution information from coarse data
- **Geometric_Shape_Fitter**: Component that fits curvature and statistical moments to estimate parameters
- **Parameter_Estimator**: Component that estimates r₀, σ, β from multi-resolution analysis
- **Radial_Profile**: 1D intensity distribution I(r) after angular integration
- **Angular_Distribution**: Distribution I(θ) at a given radius for β estimation
- **Box_Kernel**: Convolution kernel representing binning operation
- **Scale_Pyramid**: Hierarchy of binned data at different resolutions
- **Local_Curvature**: Second derivative of the distribution, related to peak sharpness
- **Statistical_Moments**: Mean, variance, skewness, kurtosis of local distributions

## Requirements

### Requirement 1: Multi-Resolution Radial Analysis

**User Story:** As a physicist, I want to analyze radial distributions at multiple bin sizes, so that I can extract optimal peak parameters regardless of peak width.

#### Acceptance Criteria

1. THE Multi_Resolution_Analyzer SHALL compute radial histograms at multiple dr values (e.g., dr = 0.05, 0.1, 0.2, 0.4, 0.8 mm)
2. WHEN computing coarse histograms, THE Multi_Resolution_Analyzer SHALL treat them as convolutions of fine histograms with box kernels
3. THE Deconvolution_Engine SHALL recover fine-resolution estimates from coarse data using Wiener or Richardson-Lucy deconvolution
4. FOR ALL resolution levels, THE Multi_Resolution_Analyzer SHALL maintain consistent normalization


### Requirement 2: Geometric Shape Fitting for Peak Detection

**User Story:** As a physicist, I want to fit the geometric shape (curvature, moments) of the radial distribution, so that I can detect peaks and estimate their parameters without relying solely on histogram binning.

#### Acceptance Criteria

1. THE Geometric_Shape_Fitter SHALL compute local curvature (second derivative) of the radial distribution
2. WHEN a local maximum in curvature magnitude is detected, THE Geometric_Shape_Fitter SHALL identify it as a potential peak location
3. THE Geometric_Shape_Fitter SHALL compute statistical moments (mean, variance, skewness) in local windows around detected peaks
4. THE Parameter_Estimator SHALL use moment-based estimates: r₀ ≈ local_mean, σ ≈ √(local_variance)
5. THE Geometric_Shape_Fitter SHALL use adaptive window sizes based on local data density

### Requirement 3: Dynamic Angular Binning for β Estimation

**User Story:** As a physicist, I want angular binning to adapt based on radius and event count, so that β estimation is optimal at all radii.

#### Acceptance Criteria

1. WHEN estimating β at small r (few events), THE Parameter_Estimator SHALL use coarser angular bins (larger dθ)
2. WHEN estimating β at large r (many events), THE Parameter_Estimator SHALL use finer angular bins (smaller dθ)
3. THE Parameter_Estimator SHALL target a minimum of 20-30 events per angular bin for valid Poisson statistics
4. THE Parameter_Estimator SHALL use two-fold symmetry (fold to [0, π]) to preserve P₂(sin θ) shape for β fitting

### Requirement 4: Optimization-Based Parameter Refinement

**User Story:** As a physicist, I want to use optimization algorithms to refine parameter estimates, so that I get the best fit to the observed data.

#### Acceptance Criteria

1. THE Parameter_Estimator SHALL define a cost function based on the difference between observed and model distributions
2. THE Parameter_Estimator SHALL use gradient-based optimization (e.g., L-BFGS-B) for continuous parameters (r₀, σ, β)
3. WHEN multiple peaks exist, THE Parameter_Estimator SHALL fit all peaks simultaneously to account for overlap
4. THE Parameter_Estimator SHALL include regularization to prevent overfitting
5. THE Parameter_Estimator SHALL provide uncertainty estimates from the optimization Hessian

### Requirement 5: Multi-Resolution Deconvolution

**User Story:** As a physicist, I want to recover fine-resolution information from coarse binned data, so that I can detect narrow peaks even with limited statistics.

#### Acceptance Criteria

1. THE Deconvolution_Engine SHALL model coarse binning as convolution: H_coarse = H_fine ⊗ Box(dr)
2. THE Deconvolution_Engine SHALL apply Wiener deconvolution with noise-dependent regularization
3. WHEN SNR is low, THE Deconvolution_Engine SHALL increase regularization to suppress noise amplification
4. THE Deconvolution_Engine SHALL combine estimates from multiple resolutions using weighted averaging
5. THE Deconvolution_Engine SHALL weight each resolution by its estimated reliability (SNR-based)

### Requirement 6: Scale-Invariant Peak Detection

**User Story:** As a physicist, I want peak detection to work reliably across all radii, so that both inner and outer peaks are found.

#### Acceptance Criteria

1. THE Multi_Resolution_Analyzer SHALL detect peaks at all radii from r_min to r_max
2. THE Multi_Resolution_Analyzer SHALL use scale-invariant prominence: prominence_scaled = prominence × (1 + r/r_max)
3. WHEN peaks are near the edge of the detector (large r), THE Multi_Resolution_Analyzer SHALL not miss them due to boundary effects
4. THE Multi_Resolution_Analyzer SHALL merge duplicate detections from different resolution levels

### Requirement 7: Noise Model Integration

**User Story:** As a physicist, I want the algorithm to properly account for Poisson and Gaussian noise, so that parameter uncertainties are accurate.

#### Acceptance Criteria

1. THE Parameter_Estimator SHALL model variance as: σ² = N (Poisson) + σ_inst² (Gaussian instrumental)
2. THE Parameter_Estimator SHALL weight fits by inverse variance
3. WHEN computing uncertainties, THE Parameter_Estimator SHALL propagate both statistical and systematic errors
4. THE Parameter_Estimator SHALL report confidence intervals for all estimated parameters


### Requirement 8: Forward Fitting for Sigma/Amplitude Estimation

**User Story:** As a physicist, I want sigma and amplitude to be estimated by forward fitting in projection space, so that I get accurate estimates that account for the full imaging chain (Abel projection + PSF + DLD quantization).

#### Background

The observed radial histogram H(r) from XY scatter points is the result of:
1. 3D Gaussian shell with intrinsic sigma → Energy broadening: `sigma_laser = sigma * E / r0`
2. Abel projection: 3D → 2D by dropping z-coordinate (`r_2d = r_3d * sin(φ)`)
3. PSF broadening: Gaussian noise on x, y positions
4. DLD quantization: Round to nearest `dld_resolution`

Abel inversion amplifies noise, making direct sigma estimation unreliable. Forward fitting compares the projected model against the observed histogram, which has better SNR.

**Key Physics:**
- Test framework uses `sigma_laser = sigma * E / r0`, which means intrinsic radius sigma = sigma/2 at r=r0
- The fitted intrinsic sigma must be multiplied by 2 to match test framework convention

#### Acceptance Criteria

1. THE Forward_Fitter SHALL use Abel inversion only for initial r0 detection (robust to noise)
2. THE Forward_Fitter SHALL fit sigma and amplitude in PROJECTION space (observed histogram)
3. THE Forward_Fitter SHALL implement Monte Carlo forward projection with:
   - 3D Gaussian shell sampling
   - Isotropic angular distribution (cos(φ) uniform)
   - Abel projection by dropping z-coordinate
   - PSF broadening (Gaussian noise on x, y)
   - DLD quantization (round to nearest resolution)
4. THE Forward_Fitter SHALL use at least 100,000 Monte Carlo samples for accuracy
5. THE Forward_Fitter SHALL multiply fitted intrinsic sigma by 2 to match test framework convention
6. THE Forward_Fitter SHALL use two-phase optimization:
   - Phase 1: Fix r0 from Abel inversion, optimize sigma and amplitude
   - Phase 2: Fine-tune all parameters (r0, sigma, amplitude) together
7. THE Forward_Fitter SHALL accept PSF sigma and DLD resolution as constructor parameters
8. WHEN fitting multi-peak data, THE Forward_Fitter SHALL fit all peaks simultaneously


### Requirement 9: Abel Inversion for Peak Position Detection

**User Story:** As a physicist, I want Abel inversion applied to the radial histogram before peak detection, so that I get the TRUE peak positions (not shifted by Abel projection effects).

#### Background

The radial histogram H(r) from XY scatter points is the Abel projection of the true 3D distribution P(r). Abel projection:
- BROADENS peaks by ~100-200%
- SHIFTS peaks inward by ~σ²/(2r₀)
- Effects are LARGER for inner peaks (small r₀)

Inverse Abel transform recovers P(r), giving accurate r0 values.

#### Acceptance Criteria

1. THE Deconvolution_Engine SHALL apply inverse Abel transform after combining resolutions
2. THE Deconvolution_Engine SHALL use PyAbel's Hansen-Law method for inverse Abel transform
3. THE Deconvolution_Engine SHALL smooth the histogram before inversion to reduce noise amplification
4. THE Deconvolution_Engine SHALL ensure non-negative output from Abel inversion
5. THE Peak_Detector SHALL detect peaks on the Abel-inverted distribution P(r)
6. THE Peak_Detector SHALL use the Abel-inverted r0 values as initial guesses for forward fitting
