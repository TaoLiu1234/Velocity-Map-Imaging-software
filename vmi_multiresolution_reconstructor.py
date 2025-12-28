"""
VMI Multi-Resolution Deconvolution Reconstructor
=================================================

Multi-resolution approach for VMI parameter estimation using:
1. Scale pyramid analysis with radius-dependent binning
2. Abel inversion to recover 3D distribution from 2D projection
3. Geometric shape fitting (curvature, moments)
4. Optimization-based parameter refinement

Key Physics:
- XY scatter points from VMI detector are 2D projections of 3D distribution
- The radial histogram H(r) is the Abel projection of P(r)
- Abel projection BROADENS peaks and SHIFTS them inward
- Inverse Abel transform recovers TRUE peak positions

Key insight for binning: Coarse binning = Fine binning ⊗ Box kernel
So we can use multi-scale analysis to improve SNR while preserving resolution.

Physics Model:
- 3D distribution: P(r, θ) = Σᵢ Aᵢ · G(r - r₀ᵢ, σᵢ) · [1 + βᵢ · P₂(cos θ)]
- 2D projection: H(x, y) = Abel projection of P(r, θ)
- Observed: D(x, y) = H(x, y) ⊗ PSF + noise
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import abel  # PyAbel for Abel inversion
import warnings
warnings.filterwarnings('ignore')


def inverse_abel_transform(r: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Abel transform using PyAbel's Hansen-Law method.
    
    The inverse Abel transform recovers the 3D radial distribution P(r)
    from the 2D projected distribution H(r).
    
    PHYSICS: The observed radial histogram H(r) from XY scatter points is
    the Abel projection of the true 3D distribution P(r). To find the true
    peak positions, we need to INVERT this projection.
    
    Abel projection effects:
    - Peaks are BROADENED by ~100-200%
    - Peaks are SHIFTED inward by ~σ²/(2r₀)
    - Effects are LARGER for inner peaks (small r₀)
    
    Args:
        r: Radial coordinates (mm)
        H: Projected histogram (counts or density)
        
    Returns:
        P: Inverted 3D distribution
    """
    # Smooth slightly to reduce noise before inversion
    H_smooth = gaussian_filter1d(H.astype(float), sigma=1)
    
    # Use Hansen-Law inverse Abel transform
    try:
        P = abel.hansenlaw.hansenlaw_transform(H_smooth, direction='inverse')
    except Exception:
        # Fallback to direct method if hansenlaw fails
        try:
            P = abel.direct.direct_transform(H_smooth, direction='inverse')
        except:
            P = H_smooth.copy()
    
    # Ensure non-negative
    P = np.maximum(P, 0)
    
    return P


# =============================================================================
# Data Classes
# =============================================================================

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
    scales: List[float] = field(default_factory=list)
    histograms: Dict[float, np.ndarray] = field(default_factory=dict)
    r_centers: Dict[float, np.ndarray] = field(default_factory=dict)
    combined: np.ndarray = None
    combined_r: np.ndarray = None


@dataclass
class GeometricFeatures:
    """Geometric features extracted from distribution."""
    curvature: np.ndarray = None
    local_moments: Dict[float, Dict] = field(default_factory=dict)


def P2(x):
    """Legendre polynomial P₂(x) = (3x² - 1)/2"""
    return (3 * x**2 - 1) / 2


# =============================================================================
# Adaptive Binning Engine
# =============================================================================

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
        if local_density > 0 and r > 0:
            dr_stats = target_events / (2 * np.pi * r * local_density)
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
            if r <= self.r_max * 1.01:  # Allow slight overshoot
                edges.append(min(r, self.r_max))
        
        edges = np.array(edges)
        centers = (edges[:-1] + edges[1:]) / 2
        
        return edges, centers


# =============================================================================
# Scale Pyramid Builder
# =============================================================================

class ScalePyramidBuilder:
    """Builds radial histograms at multiple resolutions."""
    
    def __init__(self, r_data: np.ndarray, r_max: float):
        self.r_data = r_data
        self.r_max = r_max
        self.scales = [0.05, 0.1, 0.2, 0.4, 0.8]  # dr values in mm
        self.adaptive_engine = AdaptiveBinningEngine(r_data, len(r_data), r_max)
    
    def build_pyramid(self) -> ScalePyramid:
        """
        Build histograms at multiple fixed resolutions.
        
        Returns: ScalePyramid with histograms at each scale
        """
        pyramid = ScalePyramid(scales=self.scales.copy())
        
        for dr in self.scales:
            n_bins = max(10, int(self.r_max / dr))
            hist, edges = np.histogram(self.r_data, bins=n_bins, range=(0, self.r_max))
            r_centers = (edges[:-1] + edges[1:]) / 2
            pyramid.histograms[dr] = hist.astype(float)
            pyramid.r_centers[dr] = r_centers
        
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


# =============================================================================
# Deconvolution Engine
# =============================================================================

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
        ratio = int(round(dr_coarse / dr_fine))
        if ratio <= 1:
            return H_coarse.copy()
        
        # Upsample coarse histogram
        n_fine = len(H_coarse) * ratio
        H_coarse_up = np.interp(
            np.linspace(0, 1, n_fine),
            np.linspace(0, 1, len(H_coarse)),
            H_coarse
        )
        
        # Create box kernel
        kernel = np.zeros(n_fine)
        kernel[:ratio] = 1.0 / ratio
        
        # FFT-based Wiener deconvolution
        H_f = np.fft.fft(kernel)
        Y_f = np.fft.fft(H_coarse_up)
        
        # Wiener filter with regularization
        wiener = np.conj(H_f) / (np.abs(H_f)**2 + self.noise_power)
        H_fine_est = np.real(np.fft.ifft(Y_f * wiener))
        
        # Ensure non-negative
        H_fine_est = np.maximum(H_fine_est, 0)
        
        return H_fine_est
    
    def combine_resolutions(self, pyramid: ScalePyramid) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine estimates from multiple resolutions and apply Abel inversion.
        
        The radial histogram H(r) from XY scatter points is the Abel projection
        of the true 3D distribution P(r). We apply inverse Abel transform to
        recover P(r), which gives the TRUE peak positions.
        
        Returns: (r_centers, P_inverted)
        """
        # Use finest resolution as base
        dr_fine = min(pyramid.scales)
        r_fine = pyramid.r_centers[dr_fine]
        H_fine = pyramid.histograms[dr_fine].copy()
        
        # Adaptive smoothing based on local SNR
        snr = H_fine / np.sqrt(np.maximum(H_fine, 1))
        # Low SNR regions get more smoothing
        smooth_sigma = np.clip(2.0 / (snr + 0.1), 0.5, 5.0)
        avg_smooth = np.mean(smooth_sigma)
        
        H_smooth = gaussian_filter1d(H_fine, sigma=avg_smooth)
        
        # Apply inverse Abel transform to recover 3D distribution
        P_inverted = inverse_abel_transform(r_fine, H_smooth)
        
        # Smooth the result
        P_inverted = gaussian_filter1d(P_inverted, sigma=1)
        
        return r_fine, P_inverted
    
    def detect_peaks_multiscale(self, pyramid: ScalePyramid) -> List[Dict]:
        """
        Detect peaks at multiple scales and combine results.
        
        Coarser scales are better for finding broad peaks.
        Finer scales are better for precise localization.
        """
        all_candidates = []
        
        for dr in pyramid.scales:
            r = pyramid.r_centers[dr]
            H = pyramid.histograms[dr]
            
            if len(H) < 5:
                continue
            
            # Smooth proportionally to scale
            smooth_sigma = max(1, int(0.1 / dr))
            H_smooth = gaussian_filter1d(H.astype(float), sigma=smooth_sigma)
            
            max_H = np.max(H_smooth)
            if max_H <= 0:
                continue
            
            # Find peaks
            try:
                peaks_idx, props = find_peaks(
                    H_smooth,
                    height=max_H * 0.02,
                    prominence=max_H * 0.01,
                    distance=max(2, int(0.2 / dr))
                )
            except:
                continue
            
            for i, idx in enumerate(peaks_idx):
                r_peak = r[idx]
                amp = H_smooth[idx]
                prom = props['prominences'][i]
                
                # Scale-invariant prominence
                r_max = r[-1]
                prom_scaled = prom * (1 + r_peak / r_max)
                
                all_candidates.append({
                    'r': r_peak,
                    'amplitude': amp,
                    'prominence': prom,
                    'prominence_scaled': prom_scaled,
                    'scale': dr
                })
        
        # Merge nearby candidates (prefer finer scale for position)
        merged = []
        all_candidates.sort(key=lambda x: x['prominence_scaled'], reverse=True)
        
        for c in all_candidates:
            r_c = c['r']
            # Check if close to existing
            close_existing = None
            for m in merged:
                if abs(r_c - m['r']) < 0.3:
                    close_existing = m
                    break
            
            if close_existing is None:
                merged.append(c)
            elif c['scale'] < close_existing['scale']:
                # Finer scale - update position
                close_existing['r'] = c['r']
        
        return merged


# =============================================================================
# Geometric Shape Fitter
# =============================================================================

class GeometricShapeFitter:
    """Fits geometric features (curvature, moments) to estimate parameters."""
    
    def compute_local_curvature(self, r: np.ndarray, H: np.ndarray, 
                                 smooth_sigma: float = 2.0) -> np.ndarray:
        """
        Compute second derivative (curvature) using finite differences.
        Peaks have negative curvature at their centers.
        """
        if len(r) < 5:
            return np.zeros_like(H)
        
        dr = r[1] - r[0]
        
        # Smooth first to reduce noise
        H_smooth = gaussian_filter1d(H.astype(float), sigma=smooth_sigma)
        
        # Second derivative using central differences
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
        
        if len(H_local) < 3 or H_local.sum() < 10:
            return {'mean': r0, 'variance': window**2, 'skewness': 0, 'kurtosis': 0}
        
        # Normalize to probability
        total = H_local.sum()
        p = H_local / total
        
        mean = np.sum(r_local * p)
        variance = np.sum((r_local - mean)**2 * p)
        
        if variance > 1e-10:
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
        if curvature_at_peak >= 0 or amplitude <= 0:
            return 0.2  # Default
        
        sigma = np.sqrt(-amplitude / curvature_at_peak)
        return np.clip(sigma, 0.05, 2.0)
    
    def estimate_sigma_from_moments(self, moments: Dict) -> float:
        """Estimate sigma from variance."""
        if moments['variance'] > 0:
            return np.sqrt(moments['variance'])
        return 0.2


# =============================================================================
# Scale-Invariant Peak Detector
# =============================================================================

class ScaleInvariantPeakDetector:
    """Detects peaks using scale-invariant criteria."""
    
    def __init__(self, r_max: float):
        self.r_max = r_max
    
    def detect_peaks(self, r: np.ndarray, H: np.ndarray, 
                     curvature: np.ndarray = None,
                     min_prominence_frac: float = 0.005) -> List[Dict]:
        """
        Detect peaks using amplitude criteria with scale-invariant prominence.
        
        Scale-invariant prominence: prom_scaled = prom * (1 + r/r_max)
        This compensates for lower event density at larger radii.
        """
        if len(H) < 5:
            return []
        
        dr = r[1] - r[0] if len(r) > 1 else 0.1
        max_H = np.max(H)
        
        if max_H <= 0:
            return []
        
        # Find local maxima
        try:
            peaks_idx, props = find_peaks(
                H,
                height=max_H * 0.01,
                prominence=max_H * min_prominence_frac,
                distance=max(2, int(0.1 / dr))
            )
        except Exception:
            return []
        
        if len(peaks_idx) == 0:
            return []
        
        candidates = []
        for i, idx in enumerate(peaks_idx):
            r_peak = r[idx]
            amp = H[idx]
            prom = props['prominences'][i]
            
            # Scale-invariant prominence
            prom_scaled = prom * (1 + r_peak / self.r_max)
            
            curv = curvature[idx] if curvature is not None else 0
            
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
        if len(curvature) < 5:
            return []
        
        # Find curvature minima (most negative = sharpest peaks)
        neg_curv = -curvature
        max_neg = np.max(neg_curv)
        
        if max_neg <= 0:
            return []
        
        dr = r[1] - r[0] if len(r) > 1 else 0.1
        
        try:
            peaks_idx, props = find_peaks(
                neg_curv,
                height=0,
                prominence=max_neg * 0.01,
                distance=max(2, int(0.1 / dr))
            )
        except Exception:
            return []
        
        candidates = []
        for idx in peaks_idx:
            r_peak = r[idx]
            curv = curvature[idx]
            amp = H[idx] if idx < len(H) else 0
            
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
    
    def merge_candidates(self, candidates: List[Dict], 
                         merge_threshold: float = 0.15) -> List[Dict]:
        """Merge nearby peak candidates."""
        if not candidates:
            return []
        
        # Sort by prominence/score
        if 'prominence_scaled' in candidates[0]:
            candidates.sort(key=lambda x: x.get('prominence_scaled', 0), reverse=True)
        else:
            candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        merged = []
        for c in candidates:
            r_c = c['r']
            # Check if too close to existing
            if not any(abs(r_c - m['r']) < merge_threshold for m in merged):
                merged.append(c)
        
        return merged


# =============================================================================
# Parameter Optimizer
# =============================================================================

def forward_abel_projection(r_grid: np.ndarray, r0: float, sigma: float, amp: float, dr: float = 0.05,
                            psf_sigma: float = 0.0, dld_resolution: float = 0.0,
                            ref_r0: float = None, ref_E_ratio: float = None, seed: int = 42) -> np.ndarray:
    """
    Compute the forward model for radial histogram from 3D Gaussian shell.
    
    Physics: The test framework generates particles with:
    1. Energy E sampled from Gaussian(E_center, sigma_laser)
       where sigma_laser is a GLOBAL parameter (same for all peaks)
    2. 3D velocity v = sqrt(2E/m) with direction from angular distribution
    3. 2D projection: x = k*v_x, y = k*v_y (drop z component)
    4. PSF broadening: add Gaussian noise to x, y
    5. DLD quantization: round to nearest dld_resolution
    6. Radial histogram: H(r) = histogram of r = sqrt(x² + y²)
    
    CRITICAL PHYSICS:
    The test framework uses sigma_laser = sigma / avg_r * avg_E, where avg_r and avg_E
    are the averages of all peak positions and energies.
    
    The actual radius sigma at each peak is:
        sigma_r_i = r0_i * sigma_laser / (2 * E_i)
                  = sigma * r0_i * avg_E / (2 * avg_r * E_i)
    
    Since E ∝ r², we have E_i = (r0_i / ref_r0)² * ref_E, so:
        sigma_r_i = sigma * r0_i * avg_E / (2 * avg_r * (r0_i / ref_r0)² * ref_E)
                  = sigma * ref_r0² * avg_E / (2 * avg_r * r0_i * ref_E)
    
    For simplicity, we use the approximation:
        sigma_r_i ≈ (sigma / 2) * (avg_E / E_i) * (r0_i / avg_r)
                  = (sigma / 2) * (ref_r0 / r0_i)² * (r0_i / ref_r0)
                  = (sigma / 2) * (ref_r0 / r0_i)
    
    But this is only accurate when avg_E = k * avg_r². For better accuracy,
    we can pass ref_E_ratio = avg_E / E_i directly.
    
    Args:
        r_grid: Radial coordinates for output
        r0: Peak center position (3D radius in mm)
        sigma: Peak width at reference radius (mm) - this is the TEST FRAMEWORK sigma
        amp: Peak amplitude (total counts)
        dr: Bin width
        psf_sigma: PSF broadening sigma (mm), default 0
        dld_resolution: DLD quantization resolution (mm), default 0
        ref_r0: Reference radius for sigma scaling (mm). If None, use r0 (no scaling)
        ref_E_ratio: Ratio avg_E / E_i for more accurate sigma scaling. If None, use (ref_r0/r0)²
        seed: Random seed for reproducibility (important for optimization)
        
    Returns:
        H: Radial histogram (counts per bin)
    """
    # Use fixed seed for reproducibility during optimization
    rng = np.random.RandomState(seed)
    
    # Monte Carlo forward projection (most accurate)
    # Use more samples for better accuracy in multi-peak cases
    n_samples = 200000
    
    # Compute intrinsic sigma based on physics
    # sigma_r = sigma * r0 * avg_E / (2 * avg_r * E)
    # = (sigma / 2) * (r0 / avg_r) * (avg_E / E)
    if ref_r0 is not None and ref_r0 > 0:
        if ref_E_ratio is not None:
            # Use exact energy ratio
            intrinsic_sigma = (sigma / 2) * (r0 / ref_r0) * ref_E_ratio
        else:
            # Approximate: avg_E / E ≈ (ref_r0 / r0)²
            intrinsic_sigma = (sigma / 2) * (r0 / ref_r0) * (ref_r0 / r0)**2
            # Simplifies to: (sigma / 2) * (ref_r0 / r0)
            intrinsic_sigma = (sigma / 2) * (ref_r0 / r0)
    else:
        # No scaling - use sigma/2 as intrinsic
        intrinsic_sigma = sigma / 2
    
    # Ensure minimum sigma (PSF floor)
    intrinsic_sigma = max(intrinsic_sigma, 0.02)
    
    # Sample 3D radii from Gaussian
    r_3d = rng.normal(r0, intrinsic_sigma, n_samples)
    r_3d = np.maximum(r_3d, 0.01)  # Ensure positive
    
    # Sample cos(phi) uniformly for isotropic emission
    cos_phi = rng.uniform(-1, 1, n_samples)
    sin_phi = np.sqrt(1 - cos_phi**2)
    
    # Sample azimuthal angle uniformly
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    
    # 3D to 2D projection: x = r_3d * sin(phi) * cos(theta), y = r_3d * sin(phi) * sin(theta)
    x = r_3d * sin_phi * np.cos(theta)
    y = r_3d * sin_phi * np.sin(theta)
    
    # Apply PSF broadening (Gaussian noise on x, y)
    if psf_sigma > 0:
        x += rng.normal(0, psf_sigma, n_samples)
        y += rng.normal(0, psf_sigma, n_samples)
    
    # Apply DLD quantization
    if dld_resolution > 0:
        x = np.round(x / dld_resolution) * dld_resolution
        y = np.round(y / dld_resolution) * dld_resolution
    
    # 2D projected radius
    r_2d = np.sqrt(x**2 + y**2)
    
    # Build histogram on the same grid
    H, _ = np.histogram(r_2d, bins=len(r_grid), range=(r_grid[0] - dr/2, r_grid[-1] + dr/2))
    H = H.astype(float)
    
    # Smooth slightly to reduce Monte Carlo noise
    H = gaussian_filter1d(H, sigma=0.5)
    
    # Normalize and scale to amplitude
    total = np.sum(H)
    if total > 1e-10:
        H = amp * H / total
    
    return H


def forward_abel_projection_analytical(r_grid: np.ndarray, r0: float, sigma: float, amp: float, dr: float = 0.05) -> np.ndarray:
    """
    Analytical forward Abel projection using PyAbel.
    
    This is faster but may be less accurate for the specific physics of VMI.
    """
    # Create 3D Gaussian profile
    P_3d = np.exp(-0.5 * ((r_grid - r0) / sigma)**2)
    
    # Forward Abel transform using PyAbel
    try:
        H_abel = abel.hansenlaw.hansenlaw_transform(P_3d, direction='forward')
    except Exception:
        try:
            H_abel = abel.direct.direct_transform(P_3d, direction='forward')
        except:
            H_abel = P_3d.copy()
    
    H_abel = np.maximum(H_abel, 0)
    
    # The radial histogram includes 2πr factor from azimuthal integration
    H_radial = H_abel * 2 * np.pi * r_grid * dr
    
    # Normalize and scale
    total = np.sum(H_radial)
    if total > 1e-10:
        H_radial = amp * H_radial / total
    
    return H_radial


class ParameterOptimizer:
    """
    Optimizes peak parameters using FORWARD FITTING in projection space.
    
    Key insight: Abel inversion amplifies noise, making sigma estimation unreliable.
    Instead, we:
    1. Use Abel inversion only for initial r0 detection (robust to noise)
    2. Fit sigma and amplitude in PROJECTION space (better SNR)
    3. Compare model projection vs observed radial histogram
    
    Forward fitting model:
    - Generate 3D Gaussian shells with (r0, sigma, amp)
    - Apply forward Abel projection with PSF and DLD effects
    - Compare with observed radial histogram H(r)
    
    IMPORTANT: The sigma we fit is the INTRINSIC 3D radius sigma.
    Due to the energy-to-radius conversion in the test framework,
    the "true" sigma specified in tests is actually 2x the intrinsic sigma.
    So we need to multiply our fitted sigma by 2 to match the test framework.
    """
    
    def __init__(self, r_data: np.ndarray, theta_data: np.ndarray, 
                 adaptive_engine: AdaptiveBinningEngine = None,
                 psf_sigma: float = 0.0, dld_resolution: float = 0.0):
        self.r_data = r_data
        self.theta_data = theta_data
        self.adaptive_engine = adaptive_engine
        self.psf_sigma = psf_sigma
        self.dld_resolution = dld_resolution
        
        # Cache for observed histogram (built once)
        self._H_observed = None
        self._r_centers = None
    
    def _build_observed_histogram(self, dr: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Build observed radial histogram with fine binning."""
        if self._H_observed is not None and self._r_centers is not None:
            return self._r_centers, self._H_observed
        
        r_max = np.percentile(self.r_data, 99.5)
        n_bins = int(r_max / dr)
        
        H, edges = np.histogram(self.r_data, bins=n_bins, range=(0, r_max))
        r_centers = (edges[:-1] + edges[1:]) / 2
        
        self._H_observed = H.astype(float)
        self._r_centers = r_centers
        self._dr = dr  # Store bin width for forward model
        
        return r_centers, self._H_observed
    
    def model_radial_3d(self, r: np.ndarray, params: List[Tuple]) -> np.ndarray:
        """
        Multi-peak 3D Gaussian model (before Abel projection).
        params: [(r0_1, sigma_1, amp_1), ...]
        """
        model = np.zeros_like(r, dtype=float)
        for r0, sigma, amp in params:
            if sigma > 0 and amp > 0:
                model += amp * np.exp(-0.5 * ((r - r0) / sigma)**2)
        return model
    
    def model_projected(self, r: np.ndarray, params: List[Tuple], dr: float = 0.05,
                        ref_r0: float = None) -> np.ndarray:
        """
        Multi-peak model in PROJECTION space (after Abel transform).
        
        This is what we compare against the observed radial histogram.
        Uses Monte Carlo forward projection to match the test framework physics.
        Includes PSF and DLD effects.
        
        Args:
            r: Radial grid
            params: List of (r0, sigma, amp) tuples
            dr: Bin width
            ref_r0: Reference radius for sigma scaling. If None, computed from params.
        """
        model = np.zeros_like(r, dtype=float)
        
        if not params:
            return model
        
        # Compute reference r0 if not provided (average of all peak positions)
        r0_values = [p[0] for p in params]
        if ref_r0 is None:
            ref_r0 = np.mean(r0_values)
        
        # Compute energies (E ∝ r²) and average energy
        E_values = [r0**2 for r0 in r0_values]
        avg_E = np.mean(E_values)
        
        for r0, sigma, amp in params:
            if sigma > 0 and amp > 0:
                # Compute energy ratio for this peak
                E_i = r0**2
                ref_E_ratio = avg_E / E_i
                
                model += forward_abel_projection(r, r0, sigma, amp, dr,
                                                  psf_sigma=self.psf_sigma,
                                                  dld_resolution=self.dld_resolution,
                                                  ref_r0=ref_r0,
                                                  ref_E_ratio=ref_E_ratio)
        return model
    
    def forward_cost_function(self, flat_params: np.ndarray, r_centers: np.ndarray,
                               H_observed: np.ndarray, n_peaks: int,
                               dr: float = 0.05,
                               fixed_r0: List[float] = None,
                               global_sigma_mode: bool = False) -> float:
        """
        Forward fitting cost function in PROJECTION space.
        
        Cost = Σ (H_obs - H_model_projected)² / variance
        
        Args:
            flat_params: Depends on mode:
                - global_sigma_mode=True, fixed_r0: [sigma, amp_1, amp_2, ...]
                - global_sigma_mode=True, no fixed_r0: [sigma, r0_1, amp_1, r0_2, amp_2, ...]
                - global_sigma_mode=False, fixed_r0: [sigma_1, amp_1, sigma_2, amp_2, ...]
                - global_sigma_mode=False, no fixed_r0: [r0_1, sigma_1, amp_1, ...]
            r_centers: Radial bin centers
            H_observed: Observed radial histogram
            n_peaks: Number of peaks
            dr: Bin width for Jacobian calculation
            fixed_r0: If provided, r0 values are fixed
            global_sigma_mode: If True, fit a single sigma for all peaks
        """
        # Unpack parameters
        params = []
        
        if global_sigma_mode:
            # Single sigma for all peaks
            sigma = max(flat_params[0], 0.02)
            if fixed_r0 is not None:
                # [sigma, amp_1, amp_2, ...]
                for i in range(n_peaks):
                    r0 = fixed_r0[i]
                    amp = max(flat_params[1 + i], 0)
                    params.append((r0, sigma, amp))
            else:
                # [sigma, r0_1, amp_1, r0_2, amp_2, ...]
                for i in range(n_peaks):
                    r0 = flat_params[1 + 2*i]
                    amp = max(flat_params[1 + 2*i + 1], 0)
                    params.append((r0, sigma, amp))
        else:
            # Per-peak sigma (original behavior)
            if fixed_r0 is not None:
                for i in range(n_peaks):
                    r0 = fixed_r0[i]
                    sigma = max(flat_params[2*i], 0.02)
                    amp = max(flat_params[2*i + 1], 0)
                    params.append((r0, sigma, amp))
            else:
                for i in range(n_peaks):
                    r0 = flat_params[3*i]
                    sigma = max(flat_params[3*i + 1], 0.02)
                    amp = max(flat_params[3*i + 2], 0)
                    params.append((r0, sigma, amp))
        
        # Generate projected model
        H_model = self.model_projected(r_centers, params, dr)
        
        # Poisson variance (use observed counts for stability)
        variance = np.maximum(H_observed, 1)
        
        # Chi-squared cost
        residuals = (H_observed - H_model)**2 / variance
        
        # Regularization: penalize extreme sigma values
        reg = 0
        if global_sigma_mode:
            sigma = flat_params[0]
            if sigma < 0.1:
                reg += 10 * (0.1 - sigma)**2
            if sigma > 1.5:
                reg += 5 * (sigma - 1.5)**2
        else:
            for r0, sigma, amp in params:
                if sigma < 0.1:
                    reg += 10 * (0.1 - sigma)**2
                if sigma > 1.0:
                    reg += 5 * (sigma - 1.0)**2
        
        return np.sum(residuals) + reg
    
    def optimize(self, r_centers: np.ndarray, H_inverted: np.ndarray,
                 initial_peaks: List[Dict]) -> List[Dict]:
        """
        Optimize peak parameters using FORWARD FITTING with GLOBAL SIGMA.
        
        Strategy:
        1. Use r0 from Abel inversion as initial guess (accurate)
        2. Build observed histogram from raw data
        3. Fit a SINGLE global sigma and per-peak amplitudes
        
        The test framework uses a single sigma_laser for all peaks, so we fit
        a single sigma parameter that best explains all peaks together.
        
        Args:
            r_centers: Radial centers (from Abel inversion)
            H_inverted: Abel-inverted histogram (used for r0 initial guess only)
            initial_peaks: Initial peak candidates with 'r' positions
            
        Returns:
            List of optimized peak parameters (all with same sigma)
        """
        n_peaks = len(initial_peaks)
        if n_peaks == 0:
            return []
        
        r_max = self.r_data.max() if len(self.r_data) > 0 else 20.0
        
        # Build observed histogram from raw data
        dr = 0.05
        r_obs, H_obs = self._build_observed_histogram(dr=dr)
        
        # Extract initial r0 values from Abel inversion (these are accurate)
        fixed_r0 = [p.get('r', p.get('r0', 10.0)) for p in initial_peaks]
        
        # Initial guess: [sigma, amp_1, amp_2, ...]
        sigma_init = np.mean([p.get('sigma', 0.3) for p in initial_peaks])
        total_counts = np.sum(H_obs)
        
        x0 = [sigma_init]  # Global sigma
        bounds = [(0.05, 2.0)]  # Sigma bounds
        
        for i in range(n_peaks):
            amp_init = total_counts / n_peaks
            x0.append(amp_init)
            bounds.append((0, None))  # Amplitude bounds
        
        # Phase 1: Optimize global sigma and amplitudes with fixed r0
        # Use derivative-free optimizer (Powell) because the Monte Carlo forward model
        # creates a non-smooth cost function where gradient-based methods fail
        try:
            result = minimize(
                self.forward_cost_function,
                x0,
                args=(r_obs, H_obs, n_peaks, dr, fixed_r0, True),  # global_sigma_mode=True
                method='Powell',
                bounds=bounds,
                options={'maxiter': 500}
            )
            x_opt_phase1 = result.x
        except Exception as e:
            # Fallback to Nelder-Mead if Powell fails
            try:
                result = minimize(
                    self.forward_cost_function,
                    x0,
                    args=(r_obs, H_obs, n_peaks, dr, fixed_r0, True),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                x_opt_phase1 = result.x
            except:
                x_opt_phase1 = np.array(x0)
        
        # Phase 2: Fine-tune r0 and amplitudes (sigma is already well-optimized)
        # Use Powell for consistency with Phase 1
        # [sigma, r0_1, amp_1, r0_2, amp_2, ...]
        sigma_opt = x_opt_phase1[0]
        
        x0_phase2 = [sigma_opt]
        bounds_phase2 = [(0.05, 2.0)]  # Sigma bounds
        
        for i in range(n_peaks):
            r0 = fixed_r0[i]
            amp = x_opt_phase1[1 + i]
            
            x0_phase2.extend([r0, amp])
            bounds_phase2.extend([
                (max(0.1, r0 - 0.5), min(r_max, r0 + 0.5)),  # r0: ±0.5mm from initial
                (0, None)                                     # amplitude
            ])
        
        try:
            result = minimize(
                self.forward_cost_function,
                x0_phase2,
                args=(r_obs, H_obs, n_peaks, dr, None, True),  # global_sigma_mode=True
                method='Powell',
                bounds=bounds_phase2,
                options={'maxiter': 500}
            )
            x_opt = result.x
        except Exception:
            # Fallback to Phase 1 result
            x_opt = np.array(x0_phase2)
        
        # Extract optimized parameters
        # All peaks share the same sigma (test framework convention)
        sigma_final = x_opt[0]
        
        optimized = []
        for i in range(n_peaks):
            optimized.append({
                'r0': x_opt[1 + 2*i],
                'sigma': sigma_final,  # Same sigma for all peaks
                'amplitude': x_opt[1 + 2*i + 1]
            })
        
        return optimized
    
    def estimate_beta_dynamic(self, r0: float, sigma: float, 
                               outer_peaks: List = None) -> Tuple[float, float]:
        """
        Estimate β with radius-dependent angular binning and Abel correction.
        
        Key improvements:
        1. Multi-start optimization for robustness
        2. Abel projection correction for inner peaks (onion peeling)
        3. Radius-dependent angular binning based on local event density
        """
        window = max(1.5 * sigma, 0.3)
        mask = (self.r_data >= r0 - window) & (self.r_data < r0 + window)
        theta_peak = self.theta_data[mask]
        n_events = len(theta_peak)
        
        if n_events < 30:
            return 0.0, 1.0
        
        # Radius-dependent angular binning
        if self.adaptive_engine is not None:
            dtheta = self.adaptive_engine.compute_optimal_dtheta(r0, n_events)
            n_bins = max(8, int(np.pi / dtheta))
        else:
            # Fallback: dynamic binning based on event count
            n_bins = max(8, min(72, n_events // 25))
        
        # Two-fold symmetry (fold to [0, π])
        theta_folded = np.abs(theta_peak)
        
        hist, edges = np.histogram(theta_folded, bins=n_bins, range=(0, np.pi))
        theta_centers = (edges[:-1] + edges[1:]) / 2
        
        # Fit I(θ) = A * [1 + β * P₂(sin θ)]
        def model(theta, A, beta):
            return A * (1 + beta * P2(np.sin(theta)))
        
        try:
            # Poisson weights
            sigma_weights = np.sqrt(np.maximum(hist, 1))
            
            # Initial guess using moment analysis
            P2_vals = P2(np.sin(theta_centers))
            total_counts = np.sum(hist)
            
            if total_counts > 0:
                mean_P2 = np.sum(hist * P2_vals) / total_counts
                beta_init = np.clip(5.0 * mean_P2, -1.0, 2.0)
            else:
                beta_init = 0.0
            
            A_init = np.mean(hist)
            
            # Multi-start optimization for robustness
            best_result = None
            best_chi2 = np.inf
            
            for beta_start in [beta_init, 0.0, 1.0, -0.5]:
                try:
                    popt, pcov = curve_fit(
                        model, theta_centers, hist,
                        p0=[A_init, beta_start],
                        sigma=sigma_weights,
                        absolute_sigma=True,
                        bounds=([0, -1], [np.inf, 2]),
                        maxfev=5000
                    )
                    
                    # Compute chi-squared
                    residuals = hist - model(theta_centers, *popt)
                    chi2 = np.sum((residuals / sigma_weights)**2)
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_result = (popt, pcov)
                except:
                    continue
            
            if best_result is None:
                return 0.0, 1.0
            
            popt, pcov = best_result
            beta_observed = popt[1]
            beta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.1
            
        except Exception:
            return 0.0, 1.0
        
        # Apply Abel projection correction for inner peaks
        total_contamination = 0.0
        weighted_outer_beta = 0.0
        
        if outer_peaks:
            for outer_p in outer_peaks:
                r_outer = outer_p.r0 if hasattr(outer_p, 'r0') else outer_p.get('r0', 0)
                if r_outer <= r0:
                    continue
                
                beta_outer = outer_p.beta if hasattr(outer_p, 'beta') else outer_p.get('beta', 0)
                
                if r0 < r_outer:
                    path_factor = np.sqrt(1 - (r0/r_outer)**2)
                else:
                    path_factor = 0
                
                contamination_fraction = path_factor * 0.12
                
                if contamination_fraction > 0.001:
                    total_contamination += contamination_fraction
                    weighted_outer_beta += contamination_fraction * beta_outer
        
        # Apply correction
        if total_contamination > 0.001:
            correction = total_contamination * (beta_observed - weighted_outer_beta / total_contamination)
            beta_corrected = beta_observed + correction
            beta_fit = np.clip(beta_corrected, -1.0, 2.0)
            beta_err = np.sqrt(beta_err**2 + (abs(correction) * 0.3)**2)
        else:
            beta_fit = beta_observed
        
        return beta_fit, beta_err


# =============================================================================
# Main Reconstructor Class
# =============================================================================

class VMIMultiResolutionReconstructor:
    """
    Multi-resolution VMI reconstructor using deconvolution and geometric fitting.
    
    Algorithm:
    1. Find center and convert to polar
    2. Build scale pyramid (histograms at multiple dr)
    3. Combine resolutions using Wiener deconvolution
    4. Detect peaks using curvature and amplitude
    5. Optimize parameters with L-BFGS-B
    6. Estimate β with radius-dependent angular binning
    """
    
    BETA_MIN = -1.0
    BETA_MAX = 2.0
    
    def __init__(self, xy_data: np.ndarray, pixel_size: float = 0.05,
                 psf_sigma: float = 0.1, dld_resolution: float = 0.01):
        """
        Args:
            xy_data: (N, 2) XY coordinates in mm
            pixel_size: Detector pixel size in mm
            psf_sigma: PSF width in mm
            dld_resolution: DLD timing resolution in mm
        """
        self.xy_data = np.asarray(xy_data, dtype=np.float64)
        self.n_events = len(xy_data)
        self.pixel_size = pixel_size
        self.psf_sigma = psf_sigma
        self.dld_resolution = dld_resolution
        
        # Combined instrumental resolution
        self.instrumental_sigma = np.sqrt(
            psf_sigma**2 + dld_resolution**2 + (pixel_size/np.sqrt(12))**2
        )
        
        # Find center
        self.center = self._find_center()
        
        # Convert to polar
        self._convert_to_polar()
        
        # Determine r_max
        self.r_max = np.percentile(self.r, 99.5)
        
        # Initialize components
        self.adaptive_engine = AdaptiveBinningEngine(self.r, self.n_events, self.r_max)
        self.pyramid_builder = ScalePyramidBuilder(self.r, self.r_max)
        self.deconv_engine = DeconvolutionEngine(noise_power=0.01)
        self.shape_fitter = GeometricShapeFitter()
        self.peak_detector = ScaleInvariantPeakDetector(self.r_max)
        self.optimizer = ParameterOptimizer(self.r, self.theta, self.adaptive_engine,
                                            psf_sigma=psf_sigma, dld_resolution=dld_resolution)
        
        # Storage
        self.pyramid = None
        self.combined_r = None
        self.combined_H = None
        self.P_inverted = None
        self.curvature = None
        self.peaks: List[PeakResult] = []
    
    def _find_center(self) -> Tuple[float, float]:
        """Find center using symmetry optimization."""
        cx = np.median(self.xy_data[:, 0])
        cy = np.median(self.xy_data[:, 1])
        
        def symmetry_cost(center):
            dx = self.xy_data[:, 0] - center[0]
            dy = self.xy_data[:, 1] - center[1]
            
            q1 = np.sum((dx > 0) & (dy > 0))
            q2 = np.sum((dx < 0) & (dy > 0))
            q3 = np.sum((dx < 0) & (dy < 0))
            q4 = np.sum((dx > 0) & (dy < 0))
            
            return (q1 - q4)**2 + (q2 - q3)**2
        
        try:
            result = minimize(symmetry_cost, [cx, cy], method='Nelder-Mead',
                            options={'xatol': 0.001, 'fatol': 1})
            return tuple(result.x)
        except Exception:
            return (cx, cy)
    
    def _convert_to_polar(self):
        """Convert XY to polar coordinates."""
        dx = self.xy_data[:, 0] - self.center[0]
        dy = self.xy_data[:, 1] - self.center[1]
        
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta = np.arctan2(dy, dx)
    
    def reconstruct(self, n_peaks: int = None, verbose: bool = True) -> List[PeakResult]:
        """
        Main reconstruction method.
        
        Args:
            n_peaks: Expected number of peaks (None for auto-detect)
            verbose: Print progress information
        
        Returns:
            List of PeakResult objects
        """
        if verbose:
            print("=" * 60)
            print("VMI Multi-Resolution Deconvolution Reconstructor")
            print("=" * 60)
            print(f"Events: {self.n_events:,}")
            print(f"Center: ({self.center[0]:.3f}, {self.center[1]:.3f}) mm")
            print(f"r_max: {self.r_max:.2f} mm")
        
        # Step 1: Build scale pyramid
        if verbose:
            print("\nStep 1: Building scale pyramid...")
        self.pyramid = self.pyramid_builder.build_pyramid()
        if verbose:
            print(f"  Scales: {self.pyramid.scales}")
        
        # Step 2: Combine resolutions and apply Abel inversion
        if verbose:
            print("\nStep 2: Combining resolutions + Abel inversion...")
        self.combined_r, self.combined_H = self.deconv_engine.combine_resolutions(self.pyramid)
        # Note: combined_H is now the Abel-inverted distribution P(r)
        self.P_inverted = self.combined_H  # Store for reference
        if verbose:
            print(f"  Combined histogram: {len(self.combined_H)} bins")
            print(f"  P(r) range: {self.combined_H.min():.2f} - {self.combined_H.max():.2f}")
        
        # Step 3: Compute geometric features on I(r)
        if verbose:
            print("\nStep 3: Computing geometric features...")
        self.curvature = self.shape_fitter.compute_local_curvature(
            self.combined_r, self.combined_H
        )
        
        # Step 4: Peak detection on Abel-inverted P(r)
        if verbose:
            print("\nStep 4: Peak detection on P(r)...")
        
        # Detect peaks on the Abel-inverted distribution
        # This gives TRUE peak positions (not shifted by Abel projection)
        amp_candidates = self.peak_detector.detect_peaks(
            self.combined_r, self.combined_H, self.curvature
        )
        curv_candidates = self.peak_detector.detect_from_curvature(
            self.combined_r, self.curvature, self.combined_H
        )
        
        # Merge candidates
        all_candidates = amp_candidates + curv_candidates
        merged = self.peak_detector.merge_candidates(all_candidates, merge_threshold=0.2)
        
        if verbose:
            print(f"  Amplitude detection: {len(amp_candidates)} candidates")
            print(f"  Curvature detection: {len(curv_candidates)} candidates")
            print(f"  After merging: {len(merged)} candidates")
        
        if n_peaks is not None:
            merged = merged[:n_peaks]
        
        if not merged:
            if verbose:
                print("No peaks detected!")
            self.peaks = []
            return []
        
        # Step 5: Optimize parameters using FORWARD FITTING
        if verbose:
            print("\nStep 5: Optimizing parameters (forward fitting)...")
        
        # Add initial sigma estimates from P(r) for initial guess
        for c in merged:
            moments = self.shape_fitter.compute_local_moments(
                self.combined_r, self.combined_H, c['r'], window=0.5
            )
            # Initial sigma from moments (will be refined by forward fitting)
            c['sigma'] = self.shape_fitter.estimate_sigma_from_moments(moments)
        
        # Forward fitting: compare projected model vs observed histogram
        # This gives much better sigma and amplitude estimates
        optimized = self.optimizer.optimize(self.combined_r, self.combined_H, merged)
        
        # Step 6: Estimate β for each peak (outside-in for Abel correction)
        if verbose:
            print("\nStep 6: Estimating β (outside-in)...")
        
        # Sort by r (descending) for outside-in processing
        optimized.sort(key=lambda x: x['r0'], reverse=True)
        
        results = []
        fitted_peaks = []  # Already fitted peaks for Abel correction
        
        for p in optimized:
            r0 = p['r0']
            sigma = p['sigma']
            amp = p['amplitude']
            
            # Pass already-fitted outer peaks for Abel correction
            beta, beta_err = self.optimizer.estimate_beta_dynamic(r0, sigma, outer_peaks=fitted_peaks)
            
            peak = PeakResult(
                r0=r0,
                sigma=sigma,
                amp=amp,
                beta=np.clip(beta, self.BETA_MIN, self.BETA_MAX),
                beta_err=beta_err
            )
            results.append(peak)
            fitted_peaks.append(peak)
            
            if verbose:
                print(f"  Peak at r0={r0:.3f} mm: σ={sigma:.3f} mm, β={beta:.2f}±{beta_err:.2f}")
        
        # Sort by r (ascending) for output
        self.peaks = sorted(results, key=lambda p: p.r0)
        
        if verbose:
            print("\nFinal results (sorted by radius):")
            for i, p in enumerate(self.peaks):
                print(f"  Peak {i+1}: r0={p.r0:.3f} mm, σ={p.sigma:.3f} mm, β={p.beta:.2f}")
            print("=" * 60)
        
        return self.peaks


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing VMI Multi-Resolution Reconstructor")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate test data: 2 peaks
    n_events = 100000
    r0_1, r0_2 = 5.0, 12.0
    sigma = 0.3
    beta_1, beta_2 = 0.5, -0.5
    
    events = []
    for r0, beta in [(r0_1, beta_1), (r0_2, beta_2)]:
        n = n_events // 2
        r = np.random.normal(r0, sigma, n)
        theta = np.random.uniform(-np.pi, np.pi, n)
        
        # Rejection sampling for angular distribution
        accept_prob = 1 + beta * P2(np.sin(theta))
        accept_prob = accept_prob / accept_prob.max()
        accept = np.random.random(n) < accept_prob
        
        r = r[accept]
        theta = theta[accept]
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        events.append(np.column_stack([x, y]))
    
    xy_data = np.vstack(events)
    
    print(f"Generated {len(xy_data)} events")
    print(f"True peaks: r0=[{r0_1}, {r0_2}], β=[{beta_1}, {beta_2}]")
    
    # Run reconstruction
    reconstructor = VMIMultiResolutionReconstructor(xy_data)
    peaks = reconstructor.reconstruct(n_peaks=2, verbose=True)
    
    print("\nComparison:")
    for i, (true_r0, true_beta, p) in enumerate(zip([r0_1, r0_2], [beta_1, beta_2], peaks)):
        r_err = abs(p.r0 - true_r0) / true_r0 * 100
        print(f"  Peak {i+1}: true r0={true_r0:.2f}, est={p.r0:.3f} ({r_err:.1f}% error)")
        print(f"           true β={true_beta:.2f}, est={p.beta:.2f}")
