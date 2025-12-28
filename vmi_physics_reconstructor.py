"""
Physics-Based VMI Reconstructor
================================

This reconstructor is based on the correct physics of VMI imaging:

1. ANGULAR INTEGRATION WITH β REMOVAL:
   The angular distribution at each r is: I(r,θ) = A(r)·[1 + β(r)·P₂(cos θ)]
   
   To get the true radial distribution, we:
   a) For each r, fit the angular distribution to extract A(r) and β(r)
   b) Remove the β modulation to get the isotropic component A(r)
   c) This gives us statistical information (mean, std) for noise estimation

2. PHYSICS MODEL: The 3D distribution is:
   P(r, θ) = Σᵢ Aᵢ · G(r - r₀ᵢ, σᵢ) · [1 + βᵢ · P₂(cos θ)]
   
   where G is a Gaussian and P₂ is the Legendre polynomial.

3. ABEL INVERSION: Apply Hansen-Law to the cleaned radial distribution
   to recover the 3D distribution P(r) from the 2D projection H(r).

4. PEAK EXTRACTION: Find peaks in P(r) to get r₀, σ, relative amplitude.

Key Physics Insights:
- Abel projection BROADENS peaks by ~100-200%
- Abel projection SHIFTS peaks inward by ~σ²/(2r₀)
- These effects are LARGER for inner peaks (small r₀)
- The inverse Abel transform REMOVES these effects
- Angular integration removes β modulation for cleaner radial profile

The algorithm:
1. Convert XY data to polar coordinates (r, θ)
2. For each radial bin, fit angular distribution to remove β modulation
3. Get cleaned radial distribution with statistical uncertainty
4. Apply Abel inversion (Hansen-Law) to get P(r)
5. Find peaks in P(r) - these are the TRUE peak positions
6. Estimate σ from peak widths in P(r)
7. Estimate β from angular distribution at each r₀
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
from dataclasses import dataclass
from typing import List, Tuple, Optional
import abel  # PyAbel for robust Abel inversion
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


def inverse_abel_transform(r: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Abel transform using PyAbel's Hansen-Law method.
    
    The inverse Abel transform recovers the 3D radial distribution P(r)
    from the 2D projected distribution H(r).
    
    We use PyAbel's hansenlaw method which is:
    - Numerically stable
    - Fast (O(n) complexity)
    - Well-tested
    
    Args:
        r: Radial coordinates (mm)
        H: Projected histogram (counts or density)
        
    Returns:
        P: Inverted 3D distribution
    """
    # PyAbel expects the data to be symmetric around center
    # For a 1D radial profile, we need to create a 2D image first
    # or use the direct 1D transform
    
    # Smooth slightly to reduce noise
    H_smooth = gaussian_filter1d(H.astype(float), sigma=1)
    
    # Use Hansen-Law inverse Abel transform
    # This method works directly on the radial profile
    try:
        P = abel.hansenlaw.hansenlaw_transform(H_smooth, direction='inverse')
    except Exception:
        # Fallback to direct method if hansenlaw fails
        P = abel.direct.direct_transform(H_smooth, direction='inverse')
    
    # Ensure non-negative
    P = np.maximum(P, 0)
    
    return P


def inverse_abel_basex(r: np.ndarray, H: np.ndarray, 
                       regularization: float = 0.01) -> np.ndarray:
    """
    Inverse Abel transform using basis set expansion (BASEX-like).
    
    This method expands the distribution in a basis of Gaussian functions
    and solves for the coefficients. It's more stable than direct inversion.
    
    Args:
        r: Radial coordinates
        H: Projected histogram
        regularization: Tikhonov regularization parameter
        
    Returns:
        P: Inverted distribution
    """
    n = len(r)
    dr = r[1] - r[0]
    
    # Use Gaussian basis functions centered at each r
    # The Abel transform of a Gaussian is known analytically
    
    # Build the Abel projection matrix
    # A[i,j] = how much a Gaussian at r[j] contributes to H[i]
    sigma_basis = dr * 2  # Basis function width
    
    A = np.zeros((n, n))
    for j in range(n):
        r0 = r[j]
        if r0 < 0.1:
            continue
        # Abel projection of Gaussian at r0
        for i in range(n):
            x = r[i]
            if x <= r0 + 3 * sigma_basis:
                # Approximate Abel projection
                if x < r0:
                    # Inside the shell
                    A[i, j] = 2 * np.sqrt(r0**2 - x**2 + sigma_basis**2)
                else:
                    # Outside the shell
                    A[i, j] = 2 * sigma_basis * np.exp(-(x - r0)**2 / (2 * sigma_basis**2))
    
    # Solve with Tikhonov regularization: (A^T A + λI) c = A^T H
    ATA = A.T @ A
    ATH = A.T @ H
    
    # Add regularization
    ATA += regularization * np.eye(n)
    
    # Solve
    try:
        coeffs = np.linalg.solve(ATA, ATH)
    except:
        coeffs = np.linalg.lstsq(ATA, ATH, rcond=None)[0]
    
    # Reconstruct P
    P = np.maximum(coeffs, 0)
    
    return P


class PhysicsVMIReconstructor:
    """
    Physics-based VMI reconstructor using proper angular integration and Abel inversion.
    
    This reconstructor correctly handles the physics of VMI imaging:
    1. Angular integration with β removal to get clean radial distribution
    2. Statistical estimation of signal and noise at each radius
    3. Abel inversion (Hansen-Law) to recover 3D distribution
    4. Physics-constrained β estimation (-1 ≤ β ≤ 2)
    """
    
    BETA_MIN = -1.0
    BETA_MAX = 2.0
    
    def __init__(self, xy_data: np.ndarray):
        """
        Args:
            xy_data: (N, 2) XY coordinates in mm
        """
        self.xy_data = np.asarray(xy_data, dtype=np.float64)
        self.n_events = len(xy_data)
        
        # Find center
        self.center = self._find_center()
        
        # Convert to polar
        dx = self.xy_data[:, 0] - self.center[0]
        dy = self.xy_data[:, 1] - self.center[1]
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta = np.arctan2(dy, dx)
        
        # Determine r_max
        self.r_max = np.percentile(self.r, 99.5)
        
        # Compute optimal bin size
        self.dr = self._compute_bin_size()
        
        # Storage for intermediate results
        self.r_centers = None
        self.radial_intensity = None  # Cleaned radial distribution (β removed)
        self.radial_std = None        # Statistical uncertainty at each r
        self.beta_profile = None      # β(r) profile
        self.P_inverted = None        # Abel-inverted distribution
        
        self.peaks: List[PeakResult] = []
    
    def _find_center(self) -> Tuple[float, float]:
        """Find center using symmetry."""
        # Use median (robust to anisotropy)
        cx = np.median(self.xy_data[:, 0])
        cy = np.median(self.xy_data[:, 1])
        
        # Refine using X-axis symmetry (perpendicular to polarization)
        for _ in range(5):
            dx = self.xy_data[:, 0] - cx
            pos_x = dx[dx > 0]
            neg_x = -dx[dx < 0]
            if len(pos_x) > 100 and len(neg_x) > 100:
                cx += (np.median(pos_x) - np.median(neg_x)) * 0.3
        
        return (cx, cy)
    
    def _compute_bin_size(self) -> float:
        """Compute optimal bin size based on statistics.
        
        PHYSICS: The bin size must be small enough to resolve peaks,
        but large enough to have good statistics per bin.
        
        For narrow peaks (σ ~ 0.1mm), we need dr << σ to resolve them.
        For low event counts, we need larger bins for statistics.
        """
        # Want ~30 counts per bin for good statistics (reduced from 50)
        min_counts = 30
        total_area = np.pi * self.r_max**2
        avg_density = self.n_events / total_area
        r_typical = self.r_max / 2
        
        dr_stats = min_counts / (2 * np.pi * r_typical * avg_density + 1e-10)
        
        # Resolution limit based on event count
        # Use finer bins for better peak resolution
        if self.n_events > 1e7:
            dr_res = 0.01  # Very fine for high statistics
        elif self.n_events > 1e6:
            dr_res = 0.015
        elif self.n_events > 1e5:
            dr_res = 0.03
        elif self.n_events > 1e4:
            dr_res = 0.05
        else:
            dr_res = 0.08
        
        return np.clip(max(dr_stats, dr_res), 0.01, 0.3)
    
    def _angular_integration_with_beta_removal(self, n_theta_bins: int = 36) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform angular integration with β modulation removal.
        
        For each radial bin:
        1. Fit angular distribution: I(θ) = A·[1 + β·P₂(cos θ)]
        2. Extract A (isotropic component) and β
        3. The isotropic component A gives the true radial intensity
        4. Compute statistical uncertainty from fit residuals
        
        PHYSICS CONSTRAINTS:
        - Cylindrical symmetry: I(θ) = I(-θ) = I(π-θ), so we fold the data
          to improve statistics and reduce noise
        - Non-negativity: intensities must be ≥ 0
        
        Returns:
            r_centers: Radial bin centers
            intensity: Cleaned radial intensity (β removed)
            intensity_std: Statistical uncertainty at each r
            beta_profile: β value at each r
        """
        # Adaptive number of radial bins based on statistics and resolution needs
        if self.n_events > 1e7:
            n_r_bins = max(200, int(self.r_max / self.dr))
        elif self.n_events > 1e6:
            n_r_bins = max(150, int(self.r_max / self.dr))
        elif self.n_events > 1e5:
            n_r_bins = max(100, int(self.r_max / self.dr))
        else:
            n_r_bins = max(50, int(self.r_max / self.dr))
        
        r_edges = np.linspace(0, self.r_max, n_r_bins + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        
        # Use half the angular bins since we'll fold for symmetry
        theta_edges = np.linspace(0, np.pi, n_theta_bins // 2 + 1)
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        
        intensity = np.zeros(n_r_bins)
        intensity_std = np.zeros(n_r_bins)
        beta_profile = np.zeros(n_r_bins)
        
        # For Y-polarization: cos(θ_3D) = sin(θ_XY)
        def angular_model(theta, A, beta):
            return A * (1 + beta * P2(np.sin(theta)))
        
        min_events_threshold = 10
        
        for i in range(n_r_bins):
            r_lo, r_hi = r_edges[i], r_edges[i + 1]
            r_mask = (self.r >= r_lo) & (self.r < r_hi)
            
            if np.sum(r_mask) < min_events_threshold:
                intensity[i] = np.sum(r_mask)
                intensity_std[i] = np.sqrt(max(intensity[i], 1))
                beta_profile[i] = 0.0
                continue
            
            theta_in_bin = self.theta[r_mask]
            theta_folded = np.abs(theta_in_bin)
            hist, _ = np.histogram(theta_folded, bins=theta_edges)
            
            try:
                weights = np.sqrt(np.maximum(hist, 1))
                
                popt, pcov = curve_fit(
                    angular_model, theta_centers, hist,
                    p0=[np.mean(hist), 0.0],
                    sigma=weights,
                    bounds=([0, self.BETA_MIN], [np.inf, self.BETA_MAX]),
                    maxfev=2000
                )
                
                A_fit, beta_fit = popt
                intensity[i] = A_fit * n_theta_bins
                
                residuals = hist - angular_model(theta_centers, *popt)
                intensity_std[i] = np.std(residuals) * np.sqrt(n_theta_bins)
                beta_profile[i] = beta_fit
                
            except Exception:
                intensity[i] = np.sum(hist) * 2
                intensity_std[i] = np.sqrt(max(intensity[i], 1))
                beta_profile[i] = 0.0
        
        intensity = np.maximum(intensity, 0)
        
        return r_centers, intensity, intensity_std, beta_profile
    
    def _apply_abel_inversion(self, r: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Apply inverse Abel transform to recover 3D distribution.
        
        The key physics insight: the observed radial distribution H(r) is the
        Abel projection of the 3D distribution P(r). To find the true
        peak positions, we need to INVERT this projection.
        """
        # Smooth slightly to reduce noise before inversion
        H_smooth = gaussian_filter1d(H, sigma=1)
        
        # Apply inverse Abel transform using PyAbel's Hansen-Law method
        P = inverse_abel_transform(r, H_smooth)
        
        # Smooth the result
        P = gaussian_filter1d(P, sigma=1)
        
        return P
    
    def _detect_peaks_in_inverted(self, r: np.ndarray, P: np.ndarray,
                                   max_peaks: int = 7) -> List[dict]:
        """
        Detect peaks in the Abel-inverted distribution.
        
        Since we've inverted the Abel transform, the peaks in P(r)
        correspond to the TRUE peak positions, not the shifted/broadened
        positions in the projected distribution.
        
        Multi-scale detection with adaptive thresholds for
        better detection of narrow peaks and low-statistics scenarios.
        """
        dr = r[1] - r[0]
        
        # Exclude very inner region (r < 0.3 mm)
        valid_mask = r > 0.3
        
        # Background estimate - use robust percentile
        valid_P = P[valid_mask & (P > 0)]
        if len(valid_P) > 10:
            bg = np.percentile(valid_P, 5)
        else:
            bg = 0
        
        # Multi-scale peak detection with finer scales for narrow peaks
        candidates = []
        smooth_scales = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        
        for smooth_mm in smooth_scales:
            smooth_sigma = max(1, int(smooth_mm / dr))
            smoothed = gaussian_filter1d(P, sigma=smooth_sigma)
            
            max_val = np.max(smoothed[valid_mask])
            if max_val <= 0:
                continue
            
            scale_factor = max(0.5, smooth_mm / 0.1)
            min_height = max(bg * 1.05, max_val * 0.03 / scale_factor)
            min_prom = max_val * 0.015 / scale_factor
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
                weighted_prom = prom / scale_factor
                candidates.append({
                    'r': r[idx],
                    'prominence': prom,
                    'weighted_prominence': weighted_prom,
                    'amplitude': P[idx],
                    'scale': smooth_mm
                })
        
        # Merge nearby candidates - prefer higher prominence
        merged = []
        candidates.sort(key=lambda x: x['weighted_prominence'], reverse=True)
        
        for c in candidates:
            r_c = c['r']
            merge_dist = 0.1  # mm
            if any(abs(r_c - m['r']) < merge_dist for m in merged):
                continue
            merged.append(c)
        
        return merged[:max_peaks]
    
    def _estimate_sigma_from_inverted(self, r: np.ndarray, P: np.ndarray,
                                       r0: float) -> float:
        """
        Estimate peak width from the inverted distribution.
        
        Since we've inverted the Abel transform, the width in P(r)
        is the TRUE width, not the broadened width.
        """
        # Use a small window around the peak
        window = 0.5
        mask = (r > r0 - window) & (r < r0 + window)
        r_local = r[mask]
        P_local = P[mask]
        
        if len(r_local) < 5:
            return 0.2
        
        # FWHM method
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
        Estimate β from angular distribution at a specific peak.
        
        Physics: I(θ) = A · [1 + β · P₂(sin θ)]
        
        For Y-polarization, cos(θ_3D) = sin(θ_XY).
        
        PHYSICS CONSTRAINTS:
        1. Cylindrical symmetry: I(θ) = I(-θ), so we fold data to [0, π]
        2. Non-negativity: A ≥ 0, and I(θ) ≥ 0 for all θ
        3. β bounds: -1 ≤ β ≤ 2 (from angular momentum conservation)
        
        IMPROVED: Better handling of extreme β values (β ≈ 2) where the
        angular distribution becomes highly peaked along the polarization axis.
        
        Args:
            r0: Peak position
            sigma: Peak width
            outer_peaks: List of already-fitted outer peaks (r > r0)
        
        Returns:
            beta: Estimated β value
            beta_err: Uncertainty in β
        """
        # Use a tight window to minimize contamination
        window = max(1.5 * sigma, 0.25)
        mask = (self.r >= r0 - window) & (self.r < r0 + window)
        theta_peak = self.theta[mask]
        r_peak = self.r[mask]
        n_events = len(theta_peak)
        
        if n_events < 50:  # Reduced threshold for low statistics
            return 0.0, 1.0
        
        # EXPLOIT CYLINDRICAL SYMMETRY: fold θ to [0, π]
        theta_folded = np.abs(theta_peak)
        
        # Create angular histogram with folded data
        # Use more bins for better resolution of peaked distributions
        n_bins = min(72, max(18, int(np.sqrt(n_events) / 1.5)))
        hist_raw, edges = np.histogram(theta_folded, bins=n_bins, range=(0, np.pi))
        theta_centers = (edges[:-1] + edges[1:]) / 2
        
        # Angular model
        def model(theta, A, beta):
            return A * (1 + beta * P2(np.sin(theta)))
        
        try:
            # Poisson weights
            sigma_weights = np.sqrt(np.maximum(hist_raw, 1))
            
            # IMPROVED INITIAL GUESS using moment analysis
            # For I(θ) = A[1 + β·P₂(sin θ)], the moments give us β directly
            # <P₂(sin θ)> = β·<P₂²(sin θ)> / (1 + β·<P₂(sin θ)>)
            
            # Compute weighted average of P₂(sin θ)
            P2_vals = P2(np.sin(theta_centers))
            total_counts = np.sum(hist_raw)
            
            if total_counts > 0:
                mean_P2 = np.sum(hist_raw * P2_vals) / total_counts
                # For uniform distribution, <P₂> = 0
                # For β > 0, <P₂> > 0; for β < 0, <P₂> < 0
                # Approximate: β ≈ 5 * <P₂> (empirical calibration)
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
                        sigma=sigma_weights,
                        bounds=([0, self.BETA_MIN], [np.inf, self.BETA_MAX]),
                        maxfev=5000
                    )
                    
                    # Compute chi-squared
                    residuals = hist_raw - model(theta_centers, *popt)
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
                if outer_p.r0 <= r0:
                    continue
                
                r_outer = outer_p.r0
                beta_outer = outer_p.beta
                
                if r0 < r_outer:
                    path_factor = np.sqrt(1 - (r0/r_outer)**2)
                else:
                    path_factor = 0
                
                contamination_fraction = path_factor * 0.12  # Slightly reduced
                
                if contamination_fraction > 0.001:
                    total_contamination += contamination_fraction
                    weighted_outer_beta += contamination_fraction * beta_outer
        
        # Apply correction
        if total_contamination > 0.001:
            correction = total_contamination * (beta_observed - weighted_outer_beta / total_contamination)
            beta_corrected = beta_observed + correction
            beta_fit = np.clip(beta_corrected, self.BETA_MIN, self.BETA_MAX)
            beta_err = np.sqrt(beta_err**2 + (abs(correction) * 0.3)**2)
        else:
            beta_fit = beta_observed
        
        return beta_fit, beta_err
    
    def reconstruct(self, n_peaks: int = None, verbose: bool = True) -> List[PeakResult]:
        """
        Main reconstruction method.
        
        Algorithm:
        1. Angular integration with β removal to get clean radial distribution
        2. Apply inverse Abel transform to get P(r)
        3. Find peaks in P(r) - these are TRUE positions
        4. Estimate σ from P(r) - this is TRUE width
        5. Estimate β from angular distribution at each peak
        """
        if verbose:
            print("=" * 60)
            print("Physics-Based VMI Reconstruction")
            print("(Angular Integration + Abel Inversion)")
            print("=" * 60)
            print(f"Events: {self.n_events:,}")
            print(f"Center: ({self.center[0]:.3f}, {self.center[1]:.3f}) mm")
            print(f"r_max: {self.r_max:.2f} mm")
            print(f"Bin size: {self.dr:.3f} mm")
        
        # Step 1: Angular integration with β removal
        if verbose:
            print("\nStep 1: Angular integration with β removal...")
        self.r_centers, self.radial_intensity, self.radial_std, self.beta_profile = \
            self._angular_integration_with_beta_removal()
        
        if verbose:
            print(f"  Radial bins: {len(self.r_centers)}")
            print(f"  Total intensity: {self.radial_intensity.sum():.0f}")
        
        # Step 2: Apply inverse Abel transform
        if verbose:
            print("\nStep 2: Applying Abel inversion (Hansen-Law)...")
        self.P_inverted = self._apply_abel_inversion(self.r_centers, self.radial_intensity)
        
        if verbose:
            print(f"  P(r) range: {self.P_inverted.min():.2f} - {self.P_inverted.max():.2f}")
        
        # Step 3: Detect peaks in inverted distribution
        if verbose:
            print("\nStep 3: Detecting peaks in P(r)...")
        candidates = self._detect_peaks_in_inverted(self.r_centers, self.P_inverted)
        
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
        
        # Select top peaks by prominence
        candidates = sorted(candidates, key=lambda x: x['prominence'], reverse=True)
        selected = candidates[:n_peaks]
        selected = sorted(selected, key=lambda x: x['r'])
        
        # Step 4 & 5: Estimate parameters for each peak
        # IMPORTANT: Process from outside-in (onion peeling) for β estimation
        # because outer peaks contaminate inner peaks due to Abel projection
        if verbose:
            print("\nStep 4-5: Estimating peak parameters (outside-in)...")
        
        # Sort by radius descending (outside first)
        selected_sorted = sorted(selected, key=lambda x: x['r'], reverse=True)
        
        fitted_peaks = []  # Already fitted peaks (for β correction)
        
        for i, c in enumerate(selected_sorted):
            r0 = c['r']
            sigma = self._estimate_sigma_from_inverted(self.r_centers, self.P_inverted, r0)
            amp = c['amplitude']
            
            # Pass already-fitted outer peaks for β correction
            beta, beta_err = self._estimate_beta_at_peak(r0, sigma, outer_peaks=fitted_peaks)
            
            peak = PeakResult(
                r0=r0, sigma=sigma, amp=amp, beta=beta, beta_err=beta_err
            )
            fitted_peaks.append(peak)
            
            if verbose:
                print(f"  Peak at r0={r0:.3f} mm: σ={sigma:.3f} mm, β={beta:.2f}±{beta_err:.2f}")
        
        # Re-sort by radius ascending for output
        self.peaks = sorted(fitted_peaks, key=lambda p: p.r0)
        
        if verbose:
            print("\nFinal results (sorted by radius):")
            for i, p in enumerate(self.peaks):
                print(f"  Peak {i+1}: r0={p.r0:.3f} mm, σ={p.sigma:.3f} mm, β={p.beta:.2f}")
            print("=" * 60)
        
        return self.peaks


# Test the physics-based reconstructor
if __name__ == "__main__":
    from vmi_test_framework import TestCaseGenerator, SimulationRunner, TestCase
    
    print("Testing Physics-Based VMI Reconstructor")
    print("=" * 60)
    
    generator = TestCaseGenerator()
    runner = SimulationRunner(add_noise=True)
    
    # Test case: 2 peaks, well separated, equal amplitude
    tc = TestCase(
        case_id="TEST",
        n_peaks=2,
        event_count=int(1e6),
        peak_separation='well',
        beta_range='zero',
        amplitude_ratio='equal',
        sigma_range='medium',
        r_position='middle',
        noise_level='low'
    )
    
    config = generator.generate_config(tc)
    xy_data, _ = runner.run(config, tc)
    
    print(f"\nGround truth:")
    print(f"  r0 values: {tc.r0_values}")
    print(f"  sigma: {tc.sigma_values[0]}")
    print(f"  beta: {tc.beta_values}")
    
    # Run reconstruction
    reconstructor = PhysicsVMIReconstructor(xy_data)
    peaks = reconstructor.reconstruct(n_peaks=2, verbose=True)
    
    # Compare
    print(f"\nComparison:")
    for i, (true_r0, p) in enumerate(zip(tc.r0_values, peaks)):
        err = abs(p.r0 - true_r0) / true_r0 * 100
        print(f"  Peak {i+1}: true={true_r0:.3f}, est={p.r0:.3f}, error={err:.1f}%")
