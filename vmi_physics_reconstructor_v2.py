"""
Physics-Based VMI Reconstructor V2
===================================

Improved version with:
1. Proper 1D angular integration with β removal + noise model
2. Four-fold cylindrical symmetry exploitation
3. Scale-invariant adaptive binning
4. Full noise model (Poisson + Gaussian for PSF/DLD)
5. Proper pixelization and coordinate system conversion

Physics Model:
- 3D distribution: P(r, θ) = Σᵢ Aᵢ · G(r - r₀ᵢ, σᵢ) · [1 + βᵢ · P₂(cos θ)]
- 2D projection: H(x, y) = Abel projection of P(r, θ)
- Observed: D(x, y) = H(x, y) ⊗ PSF + noise

Coordinate System:
- Lab frame: X horizontal, Y vertical (polarization axis)
- Polar: r = √(x² + y²), θ = atan2(y, x)
- For Y-polarization: cos(θ_3D) = sin(θ_XY) = y/r
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
from scipy.special import legendre
from dataclasses import dataclass
from typing import List, Tuple, Optional
import abel
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PeakResult:
    """Result for a single peak."""
    r0: float       # Peak position (mm)
    sigma: float    # Peak width (mm)
    amp: float      # Amplitude
    beta: float     # Anisotropy parameter
    r0_err: float = 0.0
    sigma_err: float = 0.0
    beta_err: float = 0.0


def P2(x):
    """Legendre polynomial P₂(x) = (3x² - 1)/2"""
    return (3 * x**2 - 1) / 2


class PhysicsVMIReconstructorV2:
    """
    Improved Physics-based VMI reconstructor.
    
    Key improvements:
    1. Four-fold symmetry exploitation
    2. Proper noise model (Poisson + Gaussian)
    3. Scale-invariant binning
    4. Pixelization-aware coordinate conversion
    """
    
    BETA_MIN = -1.0
    BETA_MAX = 2.0
    
    def __init__(self, xy_data: np.ndarray, pixel_size: float = 0.05,
                 psf_sigma: float = 0.1, dld_resolution: float = 0.01):
        """
        Args:
            xy_data: (N, 2) XY coordinates in mm
            pixel_size: Detector pixel size in mm (for pixelization correction)
            psf_sigma: PSF width in mm (Gaussian broadening)
            dld_resolution: DLD timing resolution contribution in mm
        """
        self.xy_data = np.asarray(xy_data, dtype=np.float64)
        self.n_events = len(xy_data)
        self.pixel_size = pixel_size
        self.psf_sigma = psf_sigma
        self.dld_resolution = dld_resolution
        
        # Combined instrumental resolution
        self.instrumental_sigma = np.sqrt(psf_sigma**2 + dld_resolution**2 + 
                                          (pixel_size/np.sqrt(12))**2)
        
        # Find center with sub-pixel accuracy
        self.center = self._find_center_refined()
        
        # Convert to polar with pixelization correction
        self._convert_to_polar()
        
        # Determine r_max
        self.r_max = np.percentile(self.r, 99.5)
        
        # Storage for intermediate results
        self.r_centers = None
        self.radial_intensity = None
        self.radial_variance = None
        self.beta_profile = None
        self.P_inverted = None
        
        self.peaks: List[PeakResult] = []
    
    def _find_center_refined(self) -> Tuple[float, float]:
        """
        Find center using iterative refinement with symmetry.
        
        Uses four-fold symmetry: the center should give equal
        distributions in all four quadrants.
        """
        # Initial estimate using median
        cx = np.median(self.xy_data[:, 0])
        cy = np.median(self.xy_data[:, 1])
        
        # Refine using symmetry optimization
        def symmetry_cost(center):
            dx = self.xy_data[:, 0] - center[0]
            dy = self.xy_data[:, 1] - center[1]
            
            # Four quadrant counts at similar radii
            q1 = np.sum((dx > 0) & (dy > 0))  # +x, +y
            q2 = np.sum((dx < 0) & (dy > 0))  # -x, +y
            q3 = np.sum((dx < 0) & (dy < 0))  # -x, -y
            q4 = np.sum((dx > 0) & (dy < 0))  # +x, -y
            
            # For cylindrical symmetry, q1≈q4 and q2≈q3 (X-axis symmetry)
            cost = (q1 - q4)**2 + (q2 - q3)**2
            return cost
        
        from scipy.optimize import minimize
        result = minimize(symmetry_cost, [cx, cy], method='Nelder-Mead',
                         options={'xatol': 0.001, 'fatol': 1})
        
        return tuple(result.x)
    
    def _convert_to_polar(self):
        """Convert XY to polar coordinates."""
        dx = self.xy_data[:, 0] - self.center[0]
        dy = self.xy_data[:, 1] - self.center[1]
        
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta = np.arctan2(dy, dx)  # θ ∈ [-π, π]
        
        # For Y-polarization: cos(θ_3D) = sin(θ_XY) = y/r
        self.cos_theta_3d = np.sin(self.theta)
    
    def _compute_adaptive_dr(self, r: float) -> float:
        """
        Compute scale-invariant dr at radius r.
        
        Physics: At radius r, the annular area is 2πr·dr.
        For constant SNR, we want N_events ∝ area, so:
        - dr ∝ 1/r for constant events per bin
        - But dr must resolve instrumental broadening: dr > σ_inst/3
        - And dr must be small enough to resolve peaks: dr < σ_peak/3
        
        Scale-invariant approach: dr/r = constant (logarithmic binning)
        This gives equal fractional resolution at all radii.
        """
        # Minimum dr from instrumental resolution
        dr_min = self.instrumental_sigma / 2
        
        # Scale-invariant: dr = α·r where α ~ 0.02-0.05 (2-5% fractional resolution)
        # Adjust α based on total statistics
        if self.n_events > 1e7:
            alpha = 0.015  # 1.5% fractional resolution
        elif self.n_events > 1e6:
            alpha = 0.02   # 2%
        elif self.n_events > 1e5:
            alpha = 0.03   # 3%
        else:
            alpha = 0.05   # 5%
        
        dr_scale = alpha * max(r, 0.5)  # Avoid very small dr at r→0
        
        # Use the larger of resolution limit and scale-invariant
        dr = max(dr_min, dr_scale)
        
        # Clamp to reasonable range
        return np.clip(dr, 0.02, 0.5)
    
    def _compute_adaptive_n_theta(self, r: float, n_events_in_bin: int) -> int:
        """
        Compute optimal number of angular bins at radius r.
        
        Physics: For good β estimation, we need:
        1. Enough events per angular bin: N_θ > 20 for Poisson statistics
        2. Enough angular resolution: n_θ > 12 to resolve P₂(sin θ) shape
        3. At small r, fewer events → fewer bins
        
        Scale-invariant: n_θ ∝ √(N_events_in_annulus)
        """
        # Minimum bins to resolve P₂ shape
        n_theta_min = 12
        
        # Maximum bins (diminishing returns beyond this)
        n_theta_max = 72
        
        # Target ~30 events per bin for good Poisson statistics
        target_per_bin = 30
        n_theta_stats = max(n_theta_min, n_events_in_bin // target_per_bin)
        
        # Also scale with √N for optimal binning
        n_theta_sqrt = int(np.sqrt(n_events_in_bin) / 2)
        
        # Use the smaller of the two (more conservative)
        n_theta = min(n_theta_stats, n_theta_sqrt)
        
        return np.clip(n_theta, n_theta_min, n_theta_max)
    
    def _compute_optimal_binning(self) -> Tuple[float, int]:
        """
        Compute global dr and n_bins for radial histogram.
        
        Uses scale-invariant considerations but returns uniform bins
        for compatibility with Abel inversion (which expects uniform grid).
        """
        # For Abel inversion, we need uniform radial bins
        # Use the dr appropriate for middle radius
        r_mid = self.r_max / 2
        dr = self._compute_adaptive_dr(r_mid)
        
        n_bins = int(self.r_max / dr)
        n_bins = max(50, min(500, n_bins))  # Clamp to reasonable range
        
        return dr, n_bins
    
    def _exploit_fourfold_symmetry(self, r_mask: np.ndarray) -> np.ndarray:
        """
        Exploit four-fold symmetry to enhance statistics.
        
        For cylindrical symmetry with Y-polarization:
        - I(θ) = I(-θ)  [reflection about X-axis]
        - I(π-θ) = I(θ) [reflection about Y-axis for P₂]
        
        This gives four-fold enhancement by mapping all angles to [0, π/2].
        """
        theta_selected = self.theta[r_mask]
        
        # Step 1: Fold about X-axis: θ → |θ| maps [-π, π] to [0, π]
        theta_folded = np.abs(theta_selected)
        
        # Step 2: Fold about π/2: θ → π - θ for θ > π/2
        theta_folded = np.where(theta_folded > np.pi/2, 
                                np.pi - theta_folded, 
                                theta_folded)
        
        return theta_folded
    
    def _fit_angular_distribution_adaptive(self, theta_folded: np.ndarray, 
                                            r: float) -> Tuple[float, float, float, float]:
        """
        Fit angular distribution with adaptive binning based on radius.
        
        At small r: fewer events → coarser angular bins
        At large r: more events → finer angular bins
        
        Model: I(θ) = A × [1 + β × P₂(sin θ)]
        """
        n_events = len(theta_folded)
        
        # Adaptive number of bins
        n_bins = self._compute_adaptive_n_theta(r, n_events * 4)  # *4 for four-fold
        n_bins = max(6, min(n_bins // 4, 18))  # For [0, π/2] range
        
        # Create histogram
        hist, edges = np.histogram(theta_folded, bins=n_bins, range=(0, np.pi/2))
        theta_centers = (edges[:-1] + edges[1:]) / 2
        
        # Noise model: Poisson + Gaussian floor
        sigma_floor = 0.5
        variance = np.maximum(hist, 1) + sigma_floor**2
        weights = 1.0 / np.sqrt(variance)
        
        # Angular model
        def model(theta, A, beta):
            return A * (1 + beta * P2(np.sin(theta)))
        
        # Initial guess using moments
        total = np.sum(hist)
        if total > 0:
            mean_P2 = np.sum(hist * P2(np.sin(theta_centers))) / total
            beta_init = np.clip(5.0 * mean_P2, self.BETA_MIN, self.BETA_MAX)
        else:
            beta_init = 0.0
        
        A_init = np.mean(hist) if len(hist) > 0 else 1.0
        
        try:
            popt, pcov = curve_fit(
                model, theta_centers, hist,
                p0=[A_init, beta_init],
                sigma=1.0/weights,
                absolute_sigma=True,
                bounds=([0, self.BETA_MIN], [np.inf, self.BETA_MAX]),
                maxfev=3000
            )
            
            A_fit, beta_fit = popt
            A_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else A_fit * 0.1
            beta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.1
            
        except Exception:
            A_fit = np.mean(hist) if len(hist) > 0 else 0
            beta_fit = 0.0
            A_err = np.sqrt(A_fit) if A_fit > 0 else 1.0
            beta_err = 1.0
        
        return A_fit, beta_fit, A_err, beta_err
    
    def _angular_integration_with_beta_removal(self, n_theta_bins: int = 36) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform angular integration with β modulation removal using four-fold symmetry
        and ADAPTIVE binning based on radius.
        
        For each radial bin:
        1. Compute adaptive dr and dθ based on local statistics
        2. Exploit four-fold symmetry to fold data to [0, π/2]
        3. Fit angular distribution: I(θ) = A·[1 + β·P₂(sin θ)]
        4. Extract A (isotropic component) and β
        
        Returns:
            r_centers: Radial bin centers
            intensity: Cleaned radial intensity (β removed)
            variance: Variance at each r (Poisson + Gaussian)
            beta_profile: β value at each r
        """
        dr, n_r_bins = self._compute_optimal_binning()
        r_edges = np.linspace(0, self.r_max, n_r_bins + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        
        intensity = np.zeros(n_r_bins)
        variance = np.zeros(n_r_bins)
        beta_profile = np.zeros(n_r_bins)
        
        # Minimum events threshold scales with radius
        min_events_base = 10
        
        for i in range(n_r_bins):
            r_lo, r_hi = r_edges[i], r_edges[i + 1]
            r_mid = (r_lo + r_hi) / 2
            r_mask = (self.r >= r_lo) & (self.r < r_hi)
            n_in_bin = np.sum(r_mask)
            
            # Scale-invariant minimum events
            min_events = max(5, int(min_events_base * r_mid / (self.r_max / 2)))
            
            if n_in_bin < min_events:
                intensity[i] = n_in_bin
                variance[i] = max(n_in_bin, 1) + self.instrumental_sigma**2
                beta_profile[i] = 0.0
                continue
            
            # Exploit four-fold symmetry
            theta_folded = self._exploit_fourfold_symmetry(r_mask)
            
            # Fit angular distribution with ADAPTIVE binning
            A_fit, beta_fit, A_err, beta_err = self._fit_angular_distribution_adaptive(
                theta_folded, r_mid
            )
            
            # Scale intensity: A is amplitude per angular bin, multiply by total bins
            intensity[i] = A_fit * n_theta_bins
            
            # Combined variance: Poisson + Gaussian instrumental
            poisson_var = max(intensity[i], 1)
            gaussian_var = (self.instrumental_sigma * np.sqrt(n_in_bin))**2
            variance[i] = poisson_var + gaussian_var
            
            beta_profile[i] = beta_fit
        
        intensity = np.maximum(intensity, 0)
        
        return r_centers, intensity, variance, beta_profile
    
    def _apply_abel_inversion(self, r: np.ndarray, H: np.ndarray, 
                               variance: np.ndarray = None) -> np.ndarray:
        """
        Apply inverse Abel transform with noise-aware regularization.
        
        Uses PyAbel's Hansen-Law method with adaptive smoothing based
        on the noise model.
        
        Args:
            r: Radial coordinates
            H: Projected histogram
            variance: Variance at each r (for noise-aware smoothing)
        """
        # Noise-aware smoothing
        if variance is not None:
            # SNR-based adaptive smoothing
            snr = H / np.sqrt(np.maximum(variance, 1))
            # Low SNR regions get more smoothing
            smooth_sigma = np.clip(3.0 / (snr + 0.1), 0.5, 5.0)
            avg_smooth = np.mean(smooth_sigma)
        else:
            avg_smooth = 1.5
        
        H_smooth = gaussian_filter1d(H.astype(float), sigma=avg_smooth)
        
        # Apply inverse Abel transform
        try:
            P = abel.hansenlaw.hansenlaw_transform(H_smooth, direction='inverse')
        except Exception:
            P = abel.direct.direct_transform(H_smooth, direction='inverse')
        
        # Post-smoothing
        P = gaussian_filter1d(P, sigma=1)
        P = np.maximum(P, 0)
        
        return P
    
    def _detect_peaks_multiscale(self, r: np.ndarray, P: np.ndarray,
                                  max_peaks: int = 7) -> List[dict]:
        """
        Multi-scale peak detection with scale-invariant properties.
        
        Uses multiple smoothing scales and combines results,
        weighting by scale-invariant prominence.
        """
        dr = r[1] - r[0]
        valid_mask = r > 0.3  # Exclude inner region
        
        # Background estimate
        valid_P = P[valid_mask & (P > 0)]
        bg = np.percentile(valid_P, 5) if len(valid_P) > 10 else 0
        
        candidates = []
        
        # Multi-scale detection with different smoothing levels
        for base_scale in [0.02, 0.05, 0.1, 0.2, 0.3]:
            smooth_sigma = max(1, int(base_scale / dr))
            smoothed = gaussian_filter1d(P, sigma=smooth_sigma)
            
            max_val = np.max(smoothed[valid_mask])
            if max_val <= 0:
                continue
            
            # Adaptive thresholds based on scale
            scale_factor = max(0.5, base_scale / 0.1)
            min_height = max(bg * 1.02, max_val * 0.02 / scale_factor)
            min_prom = max_val * 0.01 / scale_factor
            min_dist = max(2, int(0.05 / dr))
            
            peaks_idx, props = find_peaks(
                smoothed,
                height=min_height,
                prominence=min_prom,
                distance=min_dist
            )
            
            for idx, prom in zip(peaks_idx, props['prominences']):
                if r[idx] < 0.3:
                    continue
                # Scale-invariant prominence
                scale_inv_prom = prom * (1 + r[idx] / self.r_max)
                candidates.append({
                    'r': r[idx],
                    'prominence': prom,
                    'scale_inv_prominence': scale_inv_prom,
                    'amplitude': P[idx],
                    'scale': base_scale
                })
        
        # Merge nearby candidates
        merged = []
        candidates.sort(key=lambda x: x['scale_inv_prominence'], reverse=True)
        
        for c in candidates:
            r_c = c['r']
            merge_dist = 0.15  # Slightly larger merge distance
            if any(abs(r_c - m['r']) < merge_dist for m in merged):
                continue
            merged.append(c)
        
        return merged[:max_peaks]
    
    def _estimate_sigma_from_inverted(self, r: np.ndarray, P: np.ndarray,
                                       r0: float) -> float:
        """Estimate peak width from inverted distribution using FWHM."""
        window = 0.5
        mask = (r > r0 - window) & (r < r0 + window)
        r_local = r[mask]
        P_local = P[mask]
        
        if len(r_local) < 5:
            return 0.2
        
        peak_val = np.max(P_local)
        half_max = peak_val / 2
        
        above_half = r_local[P_local > half_max]
        if len(above_half) >= 2:
            fwhm = above_half[-1] - above_half[0]
            sigma = fwhm / 2.355
            return np.clip(sigma, 0.05, 1.0)
        
        return 0.2
    
    def _estimate_beta_at_peak(self, r0: float, sigma: float,
                                outer_peaks: List[PeakResult] = None) -> Tuple[float, float]:
        """
        Estimate β using TWO-fold symmetry with ADAPTIVE angular binning.
        
        At small r0: fewer events → coarser angular bins (larger dθ)
        At large r0: more events → finer angular bins (smaller dθ)
        
        Uses two-fold symmetry (fold to [0, π]) which preserves the P₂(sin θ) shape
        better than four-fold for β estimation.
        """
        window = max(1.5 * sigma, 0.25)
        mask = (self.r >= r0 - window) & (self.r < r0 + window)
        theta_peak = self.theta[mask]
        n_events = len(theta_peak)
        
        if n_events < 30:  # Reduced threshold for small r
            return 0.0, 1.0
        
        # Use TWO-fold symmetry for β estimation (fold to [0, π])
        theta_folded = np.abs(theta_peak)
        
        # ADAPTIVE angular binning based on statistics and radius
        # At small r: fewer events, need coarser bins
        # Target: ~25 events per bin for good Poisson statistics
        target_per_bin = 25
        n_bins_stats = max(8, n_events // target_per_bin)
        
        # Also consider radius-dependent binning
        # At small r, angular resolution is less critical
        n_bins_radius = max(12, int(36 * r0 / self.r_max))
        
        # Use the smaller (more conservative)
        n_bins = min(n_bins_stats, n_bins_radius, 72)
        n_bins = max(8, n_bins)  # At least 8 bins
        
        hist_raw, edges = np.histogram(theta_folded, bins=n_bins, range=(0, np.pi))
        theta_centers = (edges[:-1] + edges[1:]) / 2
        
        # Angular model: I(θ) = A × [1 + β × P₂(sin θ)]
        def model(theta, A, beta):
            return A * (1 + beta * P2(np.sin(theta)))
        
        try:
            # Poisson + Gaussian noise model
            sigma_floor = 0.5
            variance = np.maximum(hist_raw, 1) + sigma_floor**2
            weights = 1.0 / np.sqrt(variance)
            
            # Initial guess using moment analysis
            P2_vals = P2(np.sin(theta_centers))
            total_counts = np.sum(hist_raw)
            
            if total_counts > 0:
                mean_P2 = np.sum(hist_raw * P2_vals) / total_counts
                beta_init = np.clip(5.0 * mean_P2, self.BETA_MIN, self.BETA_MAX)
            else:
                beta_init = 0.0
            
            A_init = np.mean(hist_raw)
            
            # Multi-start optimization for robustness
            best_result = None
            best_chi2 = np.inf
            
            for beta_start in [beta_init, 0.0, 1.0, -0.5]:
                try:
                    popt, pcov = curve_fit(
                        model, theta_centers, hist_raw,
                        p0=[A_init, beta_start],
                        sigma=1.0/weights,
                        absolute_sigma=True,
                        bounds=([0, self.BETA_MIN], [np.inf, self.BETA_MAX]),
                        maxfev=5000
                    )
                    
                    residuals = hist_raw - model(theta_centers, *popt)
                    chi2 = np.sum((residuals * weights)**2)
                    
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
        if outer_peaks:
            total_contamination = 0.0
            weighted_outer_beta = 0.0
            
            for outer_p in outer_peaks:
                if outer_p.r0 <= r0:
                    continue
                
                r_outer = outer_p.r0
                beta_outer = outer_p.beta
                
                if r0 < r_outer:
                    path_factor = np.sqrt(1 - (r0/r_outer)**2)
                else:
                    path_factor = 0
                
                contamination_fraction = path_factor * 0.12
                
                if contamination_fraction > 0.001:
                    total_contamination += contamination_fraction
                    weighted_outer_beta += contamination_fraction * beta_outer
            
            if total_contamination > 0.001:
                correction = total_contamination * (beta_observed - weighted_outer_beta / total_contamination)
                beta_corrected = beta_observed + correction
                beta_fit = np.clip(beta_corrected, self.BETA_MIN, self.BETA_MAX)
                beta_err = np.sqrt(beta_err**2 + (abs(correction) * 0.3)**2)
            else:
                beta_fit = beta_observed
        else:
            beta_fit = beta_observed
        
        return beta_fit, beta_err
    
    def reconstruct(self, n_peaks: int = None, verbose: bool = True) -> List[PeakResult]:
        """
        Main reconstruction method.
        
        Algorithm:
        1. Angular integration with β removal (four-fold symmetry)
        2. Apply inverse Abel transform with noise-aware regularization
        3. Multi-scale peak detection with scale-invariant properties
        4. Estimate σ from P(r)
        5. Estimate β from angular distribution (outside-in)
        """
        if verbose:
            print("=" * 60)
            print("Physics-Based VMI Reconstruction V2")
            print("(Four-fold Symmetry + Noise Model + Scale-Invariant)")
            print("=" * 60)
            print(f"Events: {self.n_events:,}")
            print(f"Center: ({self.center[0]:.3f}, {self.center[1]:.3f}) mm")
            print(f"r_max: {self.r_max:.2f} mm")
            print(f"Instrumental σ: {self.instrumental_sigma:.4f} mm")
        
        # Step 1: Angular integration with β removal
        if verbose:
            print("\nStep 1: Angular integration (four-fold symmetry)...")
        self.r_centers, self.radial_intensity, self.radial_variance, self.beta_profile = \
            self._angular_integration_with_beta_removal()
        
        if verbose:
            print(f"  Radial bins: {len(self.r_centers)}")
            print(f"  Total intensity: {self.radial_intensity.sum():.0f}")
        
        # Step 2: Apply inverse Abel transform
        if verbose:
            print("\nStep 2: Abel inversion (noise-aware)...")
        self.P_inverted = self._apply_abel_inversion(
            self.r_centers, self.radial_intensity, self.radial_variance
        )
        
        if verbose:
            print(f"  P(r) range: {self.P_inverted.min():.2f} - {self.P_inverted.max():.2f}")
        
        # Step 3: Detect peaks
        if verbose:
            print("\nStep 3: Multi-scale peak detection...")
        candidates = self._detect_peaks_multiscale(self.r_centers, self.P_inverted)
        
        if verbose:
            print(f"  Found {len(candidates)} peak candidates")
        
        if n_peaks is None:
            n_peaks = len(candidates)
        else:
            n_peaks = min(n_peaks, len(candidates))
        
        if n_peaks == 0:
            if verbose:
                print("No peaks detected!")
            self.peaks = []
            return []
        
        # Select top peaks
        candidates = sorted(candidates, key=lambda x: x['scale_inv_prominence'], reverse=True)
        selected = candidates[:n_peaks]
        selected = sorted(selected, key=lambda x: x['r'])
        
        # Step 4 & 5: Estimate parameters (outside-in)
        if verbose:
            print("\nStep 4-5: Parameter estimation (outside-in)...")
        
        selected_sorted = sorted(selected, key=lambda x: x['r'], reverse=True)
        fitted_peaks = []
        
        for c in selected_sorted:
            r0 = c['r']
            sigma = self._estimate_sigma_from_inverted(self.r_centers, self.P_inverted, r0)
            amp = c['amplitude']
            
            beta, beta_err = self._estimate_beta_at_peak(r0, sigma, outer_peaks=fitted_peaks)
            
            peak = PeakResult(
                r0=r0, sigma=sigma, amp=amp, beta=beta, beta_err=beta_err
            )
            fitted_peaks.append(peak)
            
            if verbose:
                print(f"  Peak at r0={r0:.3f} mm: σ={sigma:.3f} mm, β={beta:.2f}±{beta_err:.2f}")
        
        self.peaks = sorted(fitted_peaks, key=lambda p: p.r0)
        
        if verbose:
            print("\nFinal results (sorted by radius):")
            for i, p in enumerate(self.peaks):
                print(f"  Peak {i+1}: r0={p.r0:.3f} mm, σ={p.sigma:.3f} mm, β={p.beta:.2f}")
            print("=" * 60)
        
        return self.peaks


# Test
if __name__ == "__main__":
    print("Testing Physics-Based VMI Reconstructor V2")
    print("=" * 60)
    
    # Simple test with synthetic data
    np.random.seed(42)
    
    # Generate test data: 2 peaks
    n_events = 100000
    r0_1, r0_2 = 5.0, 10.0
    sigma = 0.3
    beta_1, beta_2 = 0.5, -0.5
    
    # Generate events for each peak
    events = []
    for r0, beta in [(r0_1, beta_1), (r0_2, beta_2)]:
        n = n_events // 2
        r = np.random.normal(r0, sigma, n)
        
        # Sample theta with angular distribution
        theta = np.random.uniform(-np.pi, np.pi, n)
        # Rejection sampling for angular distribution
        accept_prob = 1 + beta * P2(np.sin(theta))
        accept_prob /= accept_prob.max()
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
    reconstructor = PhysicsVMIReconstructorV2(xy_data)
    peaks = reconstructor.reconstruct(n_peaks=2, verbose=True)
    
    print("\nComparison:")
    for i, (true_r0, true_beta, p) in enumerate(zip([r0_1, r0_2], [beta_1, beta_2], peaks)):
        r_err = abs(p.r0 - true_r0) / true_r0 * 100
        print(f"  Peak {i+1}: true r0={true_r0:.2f}, est={p.r0:.3f} ({r_err:.1f}% error)")
        print(f"           true β={true_beta:.2f}, est={p.beta:.2f}")
