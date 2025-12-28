# Design Document

## Overview

This design implements a multi-resolution deconvolution approach for VMI reconstruction. The core insight is that binning at different resolutions creates a scale pyramid where coarse bins are convolutions of fine bins. By combining deconvolution with geometric shape fitting and optimization, we achieve robust parameter estimation across all radii and peak widths.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VMI Multi-Resolution Reconstructor           │
├─────────────────────────────────────────────────────────────────┤
│  Input: XY scatter points (N, 2)                                │
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   Center    │───▶│  Scale Pyramid   │───▶│  Deconvolution│  │
│  │   Finder    │    │   Builder        │    │    Engine     │  │
│  └─────────────┘    └──────────────────┘    └───────────────┘  │
│                              │                      │           │
│                              ▼                      ▼           │
│                     ┌──────────────────┐    ┌───────────────┐  │
│                     │ Geometric Shape  │◀───│  Multi-Res    │  │
│                     │    Fitter        │    │   Combiner    │  │
│                     └──────────────────┘    └───────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                     ┌──────────────────┐                       │
│                     │   Peak Detector  │                       │
│                     │ (Scale-Invariant)│                       │
│                     └──────────────────┘                       │
│                              │                                  │
│                              ▼                                  │
│                     ┌──────────────────┐                       │
│                     │   Parameter      │                       │
│                     │   Optimizer      │                       │
│                     └──────────────────┘                       │
│                              │                                  │
│  Output: List[PeakResult(r0, σ, β, uncertainties)]             │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. AdaptiveBinningEngine

```python
class AdaptiveBinningEngine:
    """
    Computes radius-dependent dr and dθ for optimal parameter estimation.
    
    Key insight: At different radii, optimal bin sizes differ:
    - Small r: fewer events → larger dr, larger dθ (coarser bins)
    - Large r: more events → smaller dr, smaller dθ (finer bins)
    
    The relationship: coarse = fine ⊗ Box(ratio)
    So we can deconvolve coarse to estimate fine.
    """
    
    def __init__(self, r_data: np.ndarray, n_events: int, r_max: float):
        self.r_data = r_data
        self.n_events = n_events
        self.r_max = r_max
    
    def compute_optimal_dr(self, r: float) -> float:
        """
        Compute optimal dr at radius r based on local event density.
        
        Physics: Events per annulus ∝ 2πr·dr·ρ(r)
        For constant SNR: dr ∝ 1/√(r·ρ(r))
        
        Scale-invariant: dr/r = α (constant fractional resolution)
        """
        # Estimate local density
        window = max(0.5, r * 0.1)
        mask = (self.r_data >= r - window) & (self.r_data < r + window)
        local_count = np.sum(mask)
        local_density = local_count / (2 * window) if window > 0 else 0
        
        # Base dr from statistics: target ~100 events per bin
        target_events = 100
        if local_density > 0:
            dr_stats = target_events / (2 * np.pi * r * local_density) if r > 0 else 0.1
        else:
            dr_stats = 0.5
        
        # Scale-invariant constraint: dr/r = α
        alpha = 0.03  # 3% fractional resolution
        dr_scale = alpha * max(r, 0.5)
        
        # Use the larger (more conservative)
        dr = max(dr_stats, dr_scale, 0.02)
        
        return np.clip(dr, 0.02, 1.0)
    
    def compute_optimal_dtheta(self, r: float, n_events_local: int) -> float:
        """
        Compute optimal dθ at radius r based on local event count.
        
        For β estimation: need enough events per angular bin (~25)
        At small r: fewer events → larger dθ
        At large r: more events → smaller dθ
        """
        # Target events per angular bin
        target_per_bin = 25
        
        # Number of bins from statistics
        n_bins = max(8, n_events_local // target_per_bin)
        n_bins = min(n_bins, 72)  # Cap at 72 bins (5° resolution)
        
        dtheta = np.pi / n_bins  # For [0, π] range (two-fold symmetry)
        
        return dtheta
    
    def build_adaptive_radial_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build non-uniform radial bins with radius-dependent dr.
        
        Returns: (bin_edges, bin_centers)
        """
        edges = [0.0]
        r = 0.0
        
        while r < self.r_max:
            dr = self.compute_optimal_dr(r)
            r += dr
            if r <= self.r_max:
                edges.append(r)
        
        edges = np.array(edges)
        centers = (edges[:-1] + edges[1:]) / 2
        
        return edges, centers


class ScalePyramidBuilder:
    """Builds radial histograms at multiple resolutions."""
    
    def __init__(self, r_data: np.ndarray, r_max: float):
        self.r_data = r_data
        self.r_max = r_max
        self.scales = [0.05, 0.1, 0.2, 0.4, 0.8]  # dr values in mm
        self.adaptive_engine = AdaptiveBinningEngine(r_data, len(r_data), r_max)
    
    def build_pyramid(self) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns: {dr: (r_centers, histogram)} for each scale
        """
        pyramid = {}
        for dr in self.scales:
            n_bins = int(self.r_max / dr)
            hist, edges = np.histogram(self.r_data, bins=n_bins, range=(0, self.r_max))
            r_centers = (edges[:-1] + edges[1:]) / 2
            pyramid[dr] = (r_centers, hist.astype(float))
        return pyramid
    
    def build_adaptive_histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build histogram with radius-dependent bin sizes.
        
        Returns: (r_centers, histogram, dr_values)
        """
        edges, centers = self.adaptive_engine.build_adaptive_radial_bins()
        hist = np.zeros(len(centers))
        dr_values = np.diff(edges)
        
        for i in range(len(centers)):
            mask = (self.r_data >= edges[i]) & (self.r_data < edges[i+1])
            hist[i] = np.sum(mask)
        
        return centers, hist, dr_values
```

### 2. DeconvolutionEngine

```python
class DeconvolutionEngine:
    """Recovers fine-resolution from coarse using Wiener deconvolution."""
    
    def __init__(self, noise_power: float = 0.01):
        self.noise_power = noise_power
    
    def wiener_deconvolve(self, H_coarse: np.ndarray, dr_coarse: float, 
                          dr_fine: float) -> np.ndarray:
        """
        Deconvolve coarse histogram to estimate fine histogram.
        
        Model: H_coarse = H_fine ⊗ Box(dr_coarse/dr_fine)
        
        Wiener filter: H_fine_est = F^{-1}[ F[H_coarse] * H*(f) / (|H(f)|² + N) ]
        where H(f) is FFT of box kernel, N is noise power
        """
        ratio = int(dr_coarse / dr_fine)
        if ratio <= 1:
            return H_coarse
        
        # Create box kernel in frequency domain
        n = len(H_coarse) * ratio
        kernel = np.ones(ratio) / ratio
        kernel_padded = np.zeros(n)
        kernel_padded[:ratio] = kernel
        
        # Wiener deconvolution
        H_coarse_up = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(H_coarse)),
            H_coarse
        )
        
        H_f = np.fft.fft(kernel_padded)
        Y_f = np.fft.fft(H_coarse_up)
        
        # Wiener filter with regularization
        wiener = np.conj(H_f) / (np.abs(H_f)**2 + self.noise_power)
        H_fine_est = np.real(np.fft.ifft(Y_f * wiener))
        
        return np.maximum(H_fine_est, 0)
    
    def combine_resolutions(self, pyramid: Dict, weights: Dict = None) -> np.ndarray:
        """Combine estimates from multiple resolutions using SNR-weighted average."""
        # Use finest resolution as base grid
        dr_fine = min(pyramid.keys())
        r_centers, H_fine = pyramid[dr_fine]
        
        combined = np.zeros_like(H_fine)
        total_weight = np.zeros_like(H_fine)
        
        for dr, (r, H) in pyramid.items():
            # Deconvolve to fine resolution
            H_deconv = self.wiener_deconvolve(H, dr, dr_fine)
            
            # Weight by SNR (higher counts = higher weight)
            snr = np.sqrt(np.maximum(H_deconv, 1))
            w = snr if weights is None else weights.get(dr, 1.0) * snr
            
            combined += H_deconv * w
            total_weight += w
        
        return combined / np.maximum(total_weight, 1e-10)
```

### 3. GeometricShapeFitter

```python
class GeometricShapeFitter:
    """Fits geometric features (curvature, moments) to estimate parameters."""
    
    def compute_local_curvature(self, r: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Compute second derivative (curvature) using finite differences.
        Peaks have negative curvature at their centers.
        """
        dr = r[1] - r[0]
        # Smooth first to reduce noise
        H_smooth = gaussian_filter1d(H, sigma=2)
        # Second derivative
        curvature = np.gradient(np.gradient(H_smooth, dr), dr)
        return curvature
    
    def compute_local_moments(self, r: np.ndarray, H: np.ndarray, 
                               r0: float, window: float) -> Dict:
        """
        Compute statistical moments in a window around r0.
        
        Returns: {mean, variance, skewness, kurtosis}
        """
        mask = (r >= r0 - window) & (r <= r0 + window)
        r_local = r[mask]
        H_local = H[mask]
        
        if H_local.sum() < 10:
            return {'mean': r0, 'variance': window**2, 'skewness': 0, 'kurtosis': 0}
        
        # Normalize to probability
        p = H_local / H_local.sum()
        
        mean = np.sum(r_local * p)
        variance = np.sum((r_local - mean)**2 * p)
        
        if variance > 0:
            std = np.sqrt(variance)
            skewness = np.sum(((r_local - mean) / std)**3 * p)
            kurtosis = np.sum(((r_local - mean) / std)**4 * p) - 3
        else:
            skewness, kurtosis = 0, 0
        
        return {
            'mean': mean,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def estimate_sigma_from_curvature(self, curvature_at_peak: float, 
                                       amplitude: float) -> float:
        """
        For Gaussian: d²G/dr² at peak = -G(0)/σ²
        So: σ = sqrt(-amplitude / curvature)
        """
        if curvature_at_peak >= 0:
            return 0.2  # Default
        return np.sqrt(-amplitude / curvature_at_peak)
```

### 4. ScaleInvariantPeakDetector

```python
class ScaleInvariantPeakDetector:
    """Detects peaks using scale-invariant criteria."""
    
    def __init__(self, r_max: float):
        self.r_max = r_max
    
    def detect_peaks(self, r: np.ndarray, H: np.ndarray, 
                     curvature: np.ndarray) -> List[Dict]:
        """
        Detect peaks using both amplitude and curvature criteria.
        
        Scale-invariant prominence: prom_scaled = prom * (1 + r/r_max)
        This compensates for lower event density at larger radii.
        """
        dr = r[1] - r[0]
        
        # Find local maxima
        peaks_idx, props = find_peaks(
            H,
            height=np.max(H) * 0.01,  # 1% of max
            prominence=np.max(H) * 0.005,
            distance=max(2, int(0.1 / dr))
        )
        
        candidates = []
        for idx in peaks_idx:
            r_peak = r[idx]
            amp = H[idx]
            prom = props['prominences'][list(peaks_idx).index(idx)]
            curv = curvature[idx]
            
            # Scale-invariant prominence
            prom_scaled = prom * (1 + r_peak / self.r_max)
            
            candidates.append({
                'r': r_peak,
                'idx': idx,
                'amplitude': amp,
                'prominence': prom,
                'prominence_scaled': prom_scaled,
                'curvature': curv
            })
        
        # Sort by scaled prominence
        candidates.sort(key=lambda x: x['prominence_scaled'], reverse=True)
        return candidates
    
    def detect_from_curvature(self, r: np.ndarray, curvature: np.ndarray,
                               H: np.ndarray) -> List[Dict]:
        """
        Alternative detection using curvature minima (negative = peak).
        More robust for overlapping peaks.
        """
        # Find curvature minima (most negative = sharpest peaks)
        neg_curv = -curvature
        peaks_idx, props = find_peaks(
            neg_curv,
            height=0,  # Only negative curvature
            prominence=np.max(neg_curv) * 0.01
        )
        
        candidates = []
        for idx in peaks_idx:
            r_peak = r[idx]
            curv = curvature[idx]
            amp = H[idx]
            
            # Scale-invariant score
            score = -curv * (1 + r_peak / self.r_max)
            
            candidates.append({
                'r': r_peak,
                'idx': idx,
                'amplitude': amp,
                'curvature': curv,
                'score': score
            })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
```

### 5. ParameterOptimizer (with Forward Fitting)

```python
def forward_abel_projection(r_grid: np.ndarray, r0: float, sigma: float, amp: float, dr: float = 0.05,
                            psf_sigma: float = 0.0, dld_resolution: float = 0.0) -> np.ndarray:
    """
    Compute the forward model for radial histogram from 3D Gaussian shell.
    
    Physics: The test framework generates particles with:
    1. Energy E sampled from Gaussian(E_center, sigma_laser)
       where sigma_laser = sigma * E / r0 (so actual radius sigma = sigma/2 at r=r0)
    2. 3D velocity v = sqrt(2E/m) with direction from angular distribution
    3. 2D projection: x = k*v_x, y = k*v_y (drop z component)
    4. PSF broadening: add Gaussian noise to x, y
    5. DLD quantization: round to nearest dld_resolution
    6. Radial histogram: H(r) = histogram of r = sqrt(x² + y²)
    
    For isotropic emission (β=0), the 2D projected radius r_2d = r_3d * sin(φ)
    where φ is the polar angle (uniform in cos(φ)).
    
    IMPORTANT: The sigma parameter here is the INTRINSIC 3D radius sigma,
    which is HALF of the sigma specified in the test framework.
    """
    n_samples = 100000
    
    # Sample 3D radii from Gaussian
    r_3d = np.random.normal(r0, sigma, n_samples)
    r_3d = np.maximum(r_3d, 0.01)
    
    # Sample cos(phi) uniformly for isotropic emission
    cos_phi = np.random.uniform(-1, 1, n_samples)
    sin_phi = np.sqrt(1 - cos_phi**2)
    
    # Sample azimuthal angle uniformly
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # 3D to 2D projection (Abel projection)
    x = r_3d * sin_phi * np.cos(theta)
    y = r_3d * sin_phi * np.sin(theta)
    
    # Apply PSF broadening
    if psf_sigma > 0:
        x += np.random.normal(0, psf_sigma, n_samples)
        y += np.random.normal(0, psf_sigma, n_samples)
    
    # Apply DLD quantization
    if dld_resolution > 0:
        x = np.round(x / dld_resolution) * dld_resolution
        y = np.round(y / dld_resolution) * dld_resolution
    
    # 2D projected radius
    r_2d = np.sqrt(x**2 + y**2)
    
    # Build histogram
    H, _ = np.histogram(r_2d, bins=len(r_grid), range=(r_grid[0] - dr/2, r_grid[-1] + dr/2))
    H = gaussian_filter1d(H.astype(float), sigma=0.5)
    
    # Normalize and scale
    total = np.sum(H)
    if total > 1e-10:
        H = amp * H / total
    
    return H


class ParameterOptimizer:
    """
    Optimizes peak parameters using FORWARD FITTING in projection space.
    
    Key insight: Abel inversion amplifies noise, making sigma estimation unreliable.
    Instead, we:
    1. Use Abel inversion only for initial r0 detection (robust to noise)
    2. Fit sigma and amplitude in PROJECTION space (better SNR)
    3. Compare model projection vs observed radial histogram
    
    IMPORTANT: The sigma we fit is the INTRINSIC 3D radius sigma.
    Due to the energy-to-radius conversion in the test framework,
    the "true" sigma specified in tests is actually 2x the intrinsic sigma.
    So we need to multiply our fitted sigma by 2 to match the test framework.
    """
    
    def __init__(self, r_data: np.ndarray, theta_data: np.ndarray,
                 adaptive_engine: 'AdaptiveBinningEngine' = None,
                 psf_sigma: float = 0.0, dld_resolution: float = 0.0):
        self.r_data = r_data
        self.theta_data = theta_data
        self.adaptive_engine = adaptive_engine
        self.psf_sigma = psf_sigma
        self.dld_resolution = dld_resolution
    
    def model_projected(self, r: np.ndarray, params: List[Tuple], dr: float = 0.05) -> np.ndarray:
        """
        Multi-peak model in PROJECTION space (after Abel transform).
        Uses Monte Carlo forward projection with PSF and DLD effects.
        """
        model = np.zeros_like(r, dtype=float)
        for r0, sigma, amp in params:
            if sigma > 0 and amp > 0:
                model += forward_abel_projection(r, r0, sigma, amp, dr,
                                                  psf_sigma=self.psf_sigma,
                                                  dld_resolution=self.dld_resolution)
        return model
    
    def forward_cost_function(self, flat_params: np.ndarray, r_centers: np.ndarray,
                               H_observed: np.ndarray, n_peaks: int,
                               dr: float = 0.05,
                               fixed_r0: List[float] = None) -> float:
        """
        Forward fitting cost function in PROJECTION space.
        
        Cost = Σ (H_obs - H_model_projected)² / variance + regularization
        """
        # Unpack parameters
        params = []
        if fixed_r0 is not None:
            # Only fitting sigma and amp
            for i in range(n_peaks):
                r0 = fixed_r0[i]
                sigma = max(flat_params[2*i], 0.02)
                amp = max(flat_params[2*i + 1], 0)
                params.append((r0, sigma, amp))
        else:
            # Fitting r0, sigma, amp
            for i in range(n_peaks):
                r0 = flat_params[3*i]
                sigma = max(flat_params[3*i + 1], 0.02)
                amp = max(flat_params[3*i + 2], 0)
                params.append((r0, sigma, amp))
        
        H_model = self.model_projected(r_centers, params, dr)
        
        # Poisson variance
        variance = np.maximum(H_observed, 1)
        
        # Chi-squared cost
        residuals = (H_observed - H_model)**2 / variance
        
        # Regularization
        reg = 0
        for r0, sigma, amp in params:
            if sigma < 0.1:
                reg += 10 * (0.1 - sigma)**2
            if sigma > 1.0:
                reg += 5 * (sigma - 1.0)**2
        
        return np.sum(residuals) + reg
    
    def optimize(self, r_centers: np.ndarray, H_inverted: np.ndarray,
                 initial_peaks: List[Dict]) -> List[Dict]:
        """
        Optimize peak parameters using FORWARD FITTING.
        
        Strategy:
        1. Use r0 from Abel inversion as initial guess (accurate)
        2. Build observed histogram from raw data
        3. Fit sigma and amplitude in projection space
        4. Two-phase optimization: fix r0 first, then fine-tune all
        
        Returns:
            List of optimized peak parameters with sigma multiplied by 2
            to match test framework convention.
        """
        # Phase 1: Fix r0, optimize sigma and amplitude
        # Phase 2: Fine-tune all parameters
        # ... (implementation details)
        
        # IMPORTANT: Multiply fitted sigma by 2 for test framework
        for p in optimized:
            p['sigma'] = p['sigma'] * 2
        
        return optimized
```

## Data Models

```python
@dataclass
class PeakResult:
    """Result for a single detected peak."""
    r0: float           # Peak position (mm)
    sigma: float        # Peak width (mm)
    amp: float          # Amplitude
    beta: float         # Anisotropy parameter
    r0_err: float = 0.0
    sigma_err: float = 0.0
    beta_err: float = 0.0

@dataclass
class ScalePyramid:
    """Multi-resolution histogram pyramid."""
    scales: List[float]                              # dr values
    histograms: Dict[float, np.ndarray]              # {dr: histogram}
    r_centers: Dict[float, np.ndarray]               # {dr: r_centers}
    combined: np.ndarray = None                      # Deconvolved combined estimate
    combined_r: np.ndarray = None                    # r values for combined

@dataclass
class GeometricFeatures:
    """Geometric features extracted from distribution."""
    curvature: np.ndarray                            # Second derivative
    local_moments: Dict[float, Dict]                 # {r0: {mean, var, skew, kurt}}
```

## Algorithm Flow

```
1. INPUT: XY scatter points
   │
2. CENTER FINDING
   │  - Use symmetry optimization
   │  - Convert to polar (r, θ)
   │
3. BUILD SCALE PYRAMID
   │  - Compute histograms at dr = [0.05, 0.1, 0.2, 0.4, 0.8] mm
   │  - Each coarse level = fine ⊗ Box(ratio)
   │
4. MULTI-RESOLUTION DECONVOLUTION + ABEL INVERSION
   │  - Wiener deconvolve each level to finest resolution
   │  - Combine with SNR-weighted averaging
   │  - Apply inverse Abel transform to recover P(r)
   │
5. GEOMETRIC SHAPE ANALYSIS
   │  - Compute curvature (2nd derivative) on P(r)
   │  - Identify peak candidates from curvature minima
   │
6. SCALE-INVARIANT PEAK DETECTION (on Abel-inverted P(r))
   │  - Apply prominence_scaled = prom × (1 + r/r_max)
   │  - Merge duplicates from different methods
   │  - r0 values from P(r) are TRUE positions (not shifted)
   │
7. FORWARD FITTING FOR SIGMA/AMPLITUDE
   │  - Build observed histogram H(r) from raw XY data
   │  - Phase 1: Fix r0 from Abel inversion, fit sigma and amplitude
   │  - Phase 2: Fine-tune all parameters (r0, sigma, amp)
   │  - Forward model: 3D Gaussian → Abel projection → PSF → DLD
   │  - Multiply fitted sigma by 2 for test framework convention
   │
8. BETA ESTIMATION (per peak, outside-in)
   │  - Dynamic angular binning: n_bins ∝ √(n_events)
   │  - Two-fold symmetry for P₂ shape preservation
   │  - Fit I(θ) = A[1 + β·P₂(sin θ)]
   │  - Apply Abel correction for inner peaks (onion peeling)
   │
9. OUTPUT: List[PeakResult]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do.*


**Property 1: Histogram Normalization Preservation**
*For any* input scatter data and *for all* resolution levels in the scale pyramid, the total counts in each histogram SHALL be equal (within numerical precision).
**Validates: Requirements 1.1, 1.4**

**Property 2: Convolution-Binning Equivalence**
*For any* fine-resolution histogram H_fine and box kernel of size k, convolving H_fine with the kernel and downsampling SHALL produce a histogram equivalent to directly binning at coarse resolution.
**Validates: Requirements 1.2, 5.1**

**Property 3: Deconvolution Round-Trip**
*For any* histogram H, convolving with a box kernel then deconvolving with Wiener filter SHALL produce an estimate H' where ||H - H'|| < ε for appropriate regularization.
**Validates: Requirements 1.3, 5.4**

**Property 4: Curvature Sign at Peaks**
*For any* Gaussian peak in the radial distribution, the curvature (second derivative) at the peak center SHALL be negative.
**Validates: Requirements 2.1, 2.2**

**Property 5: Moment-Based Parameter Accuracy**
*For any* synthetic Gaussian distribution with known (r₀, σ), the moment-based estimates SHALL satisfy |r₀_est - r₀_true| < 0.1·σ and |σ_est - σ_true| < 0.2·σ.
**Validates: Requirements 2.3, 2.4**

**Property 6: Adaptive Angular Binning**
*For any* two radii r₁ < r₂ with event counts n₁ < n₂, the number of angular bins SHALL satisfy n_bins(r₁) ≤ n_bins(r₂), and events per bin SHALL be ≥ 20.
**Validates: Requirements 3.1, 3.2, 3.3**

**Property 7: Cost Function Monotonicity**
*For any* observed histogram and model parameters, moving parameters closer to true values SHALL decrease the cost function value.
**Validates: Requirements 4.1**

**Property 8: Multi-Peak Fitting Improvement**
*For any* overlapping peaks, simultaneous fitting SHALL produce lower total error than independent fitting.
**Validates: Requirements 4.3**

**Property 9: Scale-Invariant Peak Detection**
*For any* synthetic data with peaks at radii r ∈ [r_min, r_max], all peaks SHALL be detected regardless of radial position.
**Validates: Requirements 6.1, 6.2, 6.3**

**Property 10: No Duplicate Peaks**
*For any* detected peak set, no two peaks SHALL have |r₁ - r₂| < merge_threshold.
**Validates: Requirements 6.4**

**Property 11: Uncertainty Scaling with Noise**
*For any* two datasets with noise levels σ₁ < σ₂, the parameter uncertainties SHALL satisfy err(σ₁) < err(σ₂).
**Validates: Requirements 7.3, 7.4**

**Property 12: Abel Inversion Peak Position Accuracy**
*For any* synthetic 3D Gaussian shell at r₀, the Abel-inverted distribution P(r) SHALL have its peak within 0.5% of r₀.
**Validates: Requirements 9.1, 9.5**

**Property 13: Forward Model Consistency**
*For any* 3D Gaussian shell with parameters (r₀, σ), the forward-projected histogram SHALL match the observed histogram from Monte Carlo simulation within statistical error.
**Validates: Requirements 8.3, 8.4**

**Property 14: Sigma Conversion Factor**
*For any* test case with specified sigma, the fitted intrinsic sigma multiplied by 2 SHALL match the test framework sigma within tolerance.
**Validates: Requirements 8.5**

**Property 15: Two-Phase Optimization Improvement**
*For any* multi-peak data, two-phase optimization (fix r0 then fine-tune) SHALL produce lower total error than single-phase optimization.
**Validates: Requirements 8.6**

## Error Handling

1. **Insufficient Data**: If n_events < 100, return empty peak list with warning
2. **No Peaks Detected**: Return empty list, log diagnostic information
3. **Deconvolution Instability**: If Wiener filter produces negative values, clip to zero
4. **Optimization Failure**: Fall back to moment-based estimates if L-BFGS-B fails
5. **Invalid β Range**: Clip β to [-1, 2] physical bounds

## Testing Strategy

### Unit Tests
- Test histogram computation at each resolution level
- Test box kernel convolution matches direct coarse binning
- Test Wiener deconvolution on synthetic data
- Test curvature computation on known Gaussian
- Test moment computation accuracy
- Test peak detection on synthetic multi-peak data

### Property-Based Tests
- Use fast-check or hypothesis to generate random peak configurations
- Verify all 11 correctness properties hold across random inputs
- Test edge cases: single peak, many peaks, overlapping peaks, edge peaks
- Minimum 100 iterations per property test

### Integration Tests
- End-to-end reconstruction on synthetic VMI data
- Compare against known ground truth parameters
- Test across range of event counts (1e4 to 1e7)
- Test across range of peak positions (inner, middle, outer)
- Test across range of β values (-1 to 2)
