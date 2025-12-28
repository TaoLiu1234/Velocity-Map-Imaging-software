"""
VMI Reconstructor v4.0 - Physics-First Design with Improved Accuracy
=====================================================================

Key improvements over v3:
1. Better center finding using X-axis symmetry (perpendicular to polarization)
2. Improved peak position using weighted centroid of particles
3. Simplified β estimation using direct FFT cos(2θ) extraction
4. Better handling of inner peaks (small r) using counts instead of density
5. Improved weak peak detection for multi-peak cases
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import from existing modules
from vmi_reconstruction import PeakResult, abel_projection, P2


@dataclass
class PeakCandidate:
    """Peak candidate with all relevant info"""
    r: float
    sigma: float
    amplitude: float
    prominence: float
    source: str  # 'density', 'counts', or 'refined'


class VMIReconstructorV4:
    """
    Physics-First VMI Reconstructor v4.0
    
    Physical Principles:
    ====================
    1. VMI Imaging: r ∝ √E (radius proportional to sqrt of kinetic energy)
    2. Angular Distribution: I(θ) ∝ 1 + β·P₂(cos θ_3D)
       - For Y-polarization in XY plane: I(θ_XY) ∝ 1 + β·P₂(sin θ_XY)
       - β constrained: -1 ≤ β ≤ 2
    3. Radial Density: ρ(r) = H(r)/(2πr·dr) - Jacobian is crucial
    4. Abel Projection: 3D→2D causes peak broadening
    """
    
    BETA_MIN = -1.0
    BETA_MAX = 2.0
    
    def __init__(self, xy_data: np.ndarray):
        """Initialize with XY coordinate data (N, 2) in mm"""
        self.xy_data = np.asarray(xy_data, dtype=np.float64)
        self.n_events = len(xy_data)
        
        # Find center
        self.center = self._find_center()
        
        # Convert to polar
        dx = self.xy_data[:, 0] - self.center[0]
        dy = self.xy_data[:, 1] - self.center[1]
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta = np.arctan2(dy, dx)
        
        # Determine r_max (use higher percentile to not cut off outer peaks)
        self.r_max = np.percentile(self.r, 99.9)
        
        # Optimal bin size
        self.dr = self._compute_bin_size()
        
        # Results
        self.peaks: List[PeakResult] = []
    
    def _find_center(self) -> Tuple[float, float]:
        """
        Find center using physical symmetry.
        
        For Y-polarization, the X distribution should be symmetric about center.
        Use median for robustness against anisotropy.
        """
        x = self.xy_data[:, 0]
        y = self.xy_data[:, 1]
        
        # Initial: median (robust to outliers and anisotropy)
        cx = np.median(x)
        cy = np.median(y)
        
        # Refine using X-axis symmetry
        for _ in range(5):
            dx = x - cx
            pos_med = np.median(dx[dx > 0]) if np.sum(dx > 0) > 100 else 0
            neg_med = np.median(-dx[dx < 0]) if np.sum(dx < 0) > 100 else 0
            cx += (pos_med - neg_med) * 0.25
        
        # Refine using radial symmetry
        for iteration in range(8):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)
            r_med = np.median(r)
            
            # Use intermediate radii
            mask = (r > r_med * 0.3) & (r < r_med * 1.5)
            if np.sum(mask) < 100:
                break
            
            # Quadrant analysis
            q1 = mask & (dx > 0) & (dy > 0)
            q2 = mask & (dx < 0) & (dy > 0)
            q3 = mask & (dx < 0) & (dy < 0)
            q4 = mask & (dx > 0) & (dy < 0)
            
            r_q1 = np.median(r[q1]) if np.sum(q1) > 20 else r_med
            r_q2 = np.median(r[q2]) if np.sum(q2) > 20 else r_med
            r_q3 = np.median(r[q3]) if np.sum(q3) > 20 else r_med
            r_q4 = np.median(r[q4]) if np.sum(q4) > 20 else r_med
            
            damping = 0.04 / (1 + iteration * 0.2)
            dcx = (r_q1 + r_q4 - r_q2 - r_q3) / 4 * damping
            dcy = (r_q1 + r_q2 - r_q3 - r_q4) / 4 * damping
            
            cx += dcx
            cy += dcy
            
            if abs(dcx) < 0.001 and abs(dcy) < 0.001:
                break
        
        return (cx, cy)
    
    def _compute_bin_size(self) -> float:
        """Compute optimal bin size based on statistics"""
        if self.n_events > 1e6:
            return 0.04
        elif self.n_events > 1e5:
            return 0.08
        elif self.n_events > 1e4:
            return 0.15
        else:
            return 0.25
    
    def _compute_distributions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute radial histogram and density"""
        n_bins = max(50, int(self.r_max / self.dr))
        counts, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        r_centers = (edges[:-1] + edges[1:]) / 2
        dr = edges[1] - edges[0]
        
        # Density with safe division
        r_safe = np.maximum(r_centers, 0.5)
        density = counts / (2 * np.pi * r_safe * dr)
        
        return r_centers, counts.astype(float), density
    
    def _detect_peaks(self, r_centers: np.ndarray, counts: np.ndarray, 
                      density: np.ndarray, max_peaks: int = 7) -> List[PeakCandidate]:
        """
        Multi-scale peak detection using both counts and density.
        
        Key insight: Use counts for inner peaks (small r), density for outer peaks.
        For weak peaks, use lower thresholds and iterative subtraction.
        """
        dr = r_centers[1] - r_centers[0]
        candidates = []
        
        # Background levels (use lower percentile for weak peak detection)
        bg_density = np.percentile(density[density > 0], 3) if np.any(density > 0) else 0
        bg_counts = np.percentile(counts[counts > 0], 3) if np.any(counts > 0) else 0
        
        # Minimum radius to avoid false positives near center
        # The center region (r < 1 mm) often has artifacts
        r_min_valid = 1.0
        
        # Smoothing scales
        scales = [0.08, 0.15, 0.3] if self.n_events > 1e5 else [0.15, 0.3, 0.5]
        
        for scale in scales:
            sigma = max(1, int(scale / dr))
            
            # Detect in density (better for outer peaks)
            smooth_d = gaussian_filter1d(density, sigma=sigma)
            max_d = np.max(smooth_d)
            if max_d > 0:
                # Lower thresholds for weak peak detection
                peaks_idx, props = find_peaks(
                    smooth_d,
                    height=max(bg_density * 1.05, max_d * 0.03),
                    prominence=max_d * 0.008,
                    distance=max(2, int(0.12 / dr))
                )
                for idx, prom in zip(peaks_idx, props['prominences']):
                    r_val = r_centers[idx]
                    # Skip peaks too close to center
                    if r_val < r_min_valid:
                        continue
                    candidates.append(PeakCandidate(
                        r=r_val, sigma=0.3, amplitude=density[idx],
                        prominence=prom, source='density'
                    ))
            
            # Detect in counts (better for inner peaks)
            smooth_c = gaussian_filter1d(counts, sigma=sigma)
            max_c = np.max(smooth_c)
            if max_c > 0:
                peaks_idx, props = find_peaks(
                    smooth_c,
                    height=max(bg_counts * 1.05, max_c * 0.03),
                    prominence=max_c * 0.008,
                    distance=max(2, int(0.12 / dr))
                )
                for idx, prom in zip(peaks_idx, props['prominences']):
                    r_val = r_centers[idx]
                    # Skip peaks too close to center
                    if r_val < r_min_valid:
                        continue
                    prom_d = prom / (2 * np.pi * max(r_val, 0.5) * dr)
                    if not any(abs(c.r - r_val) < 0.15 for c in candidates):
                        candidates.append(PeakCandidate(
                            r=r_val, sigma=0.3, amplitude=density[idx],
                            prominence=prom_d, source='counts'
                        ))
        
        # Merge nearby candidates (smaller merge distance)
        candidates.sort(key=lambda c: c.prominence, reverse=True)
        merged = []
        used_r = set()
        
        for c in candidates:
            too_close = False
            for r_used in used_r:
                if abs(c.r - r_used) < 0.2:
                    too_close = True
                    break
            if not too_close:
                merged.append(c)
                used_r.add(c.r)
        
        # Estimate sigma for each peak
        for c in merged:
            c.sigma = self._estimate_sigma(r_centers, density, c.r)
        
        # If we have fewer candidates than max_peaks, try iterative subtraction
        if len(merged) < max_peaks:
            residual = density.copy()
            for c in merged:
                # Subtract detected peak
                peak_model = c.amplitude * np.exp(-(r_centers - c.r)**2 / (2 * c.sigma**2))
                residual = np.maximum(residual - peak_model * 0.9, 0)
            
            # Look for additional peaks in residual
            smooth_r = gaussian_filter1d(residual, sigma=max(1, int(0.2 / dr)))
            max_r = np.max(smooth_r)
            if max_r > bg_density * 1.1:
                peaks_idx, props = find_peaks(
                    smooth_r,
                    height=bg_density * 1.05,
                    prominence=max_r * 0.05,
                    distance=max(2, int(0.15 / dr))
                )
                for idx, prom in zip(peaks_idx, props['prominences']):
                    r_val = r_centers[idx]
                    if r_val < r_min_valid:
                        continue
                    if not any(abs(c.r - r_val) < 0.3 for c in merged):
                        merged.append(PeakCandidate(
                            r=r_val, sigma=self._estimate_sigma(r_centers, residual, r_val),
                            amplitude=residual[idx], prominence=prom, source='residual'
                        ))
        
        # Sort by prominence (not position) for selection
        # Then the reconstruct method will sort by position
        merged.sort(key=lambda c: c.prominence, reverse=True)
        
        return merged[:max_peaks]
    
    def _estimate_sigma(self, r_centers: np.ndarray, density: np.ndarray, 
                        r0: float, window: float = 2.0) -> float:
        """Estimate peak width using FWHM"""
        mask = (r_centers > r0 - window) & (r_centers < r0 + window)
        r_local = r_centers[mask]
        d_local = density[mask]
        
        if len(r_local) < 5:
            return 0.3
        
        peak_val = np.max(d_local)
        half_max = peak_val / 2
        above = r_local[d_local > half_max]
        
        if len(above) >= 2:
            fwhm = above[-1] - above[0]
            return np.clip(fwhm / 2.355, 0.05, 2.0)
        return 0.3
    
    def _refine_position(self, r0_init: float, sigma: float) -> float:
        """
        Refine peak position using Abel projection correction.
        
        Physics: The Abel projection of a 3D spherical shell onto a 2D plane
        creates a distribution where the histogram peak is shifted inward
        from the true radius. The correction is approximately:
        
        r0_true ≈ r0_peak + 0.2 * FWHM
        
        This correction accounts for the asymmetric broadening caused by
        the Abel projection.
        """
        # Get local data around the peak
        window = max(3 * sigma, 1.5)
        mask = (self.r >= r0_init - window) & (self.r < r0_init + window)
        r_local = self.r[mask]
        
        if len(r_local) < 100:
            return r0_init
        
        # Compute local histogram
        n_bins = min(50, max(20, len(r_local) // 100))
        counts, edges = np.histogram(r_local, bins=n_bins)
        r_centers = (edges[:-1] + edges[1:]) / 2
        
        # Smooth
        counts_smooth = gaussian_filter1d(counts.astype(float), sigma=1)
        
        # Find local peak
        peak_idx = np.argmax(counts_smooth)
        peak_r = r_centers[peak_idx]
        peak_val = counts_smooth[peak_idx]
        
        # Estimate FWHM
        half_max = peak_val / 2
        above_half = r_centers[counts_smooth > half_max]
        if len(above_half) >= 2:
            fwhm = above_half[-1] - above_half[0]
        else:
            fwhm = sigma * 2.355  # Fallback to Gaussian FWHM
        
        # Apply Abel projection correction
        # The correction factor is empirically determined to be ~0.2
        correction = 0.2 * fwhm
        r0_corrected = peak_r + correction
        
        # Sanity check: don't drift too far from initial estimate
        if abs(r0_corrected - r0_init) > 1.5:
            # If correction is too large, use weighted centroid instead
            weights = np.exp(-(r_local - r0_init)**2 / (2 * sigma**2))
            r0_corrected = np.sum(r_local * weights) / np.sum(weights)
        
        return r0_corrected
    
    def _estimate_beta(self, r0: float, sigma: float) -> float:
        """
        Estimate β using FFT of angular distribution.
        
        Physics:
        For Y-polarization, the 2D angular distribution is:
        I(θ_XY) ∝ 1 + β·P₂(sin θ_XY)
        
        where P₂(x) = (3x² - 1)/2
        
        So: I(θ) = A·[1 + β·(3sin²θ - 1)/2]
                 = A·[(1 - β/2) + (3β/2)·sin²θ]
        
        Using sin²θ = (1 - cos(2θ))/2:
        I(θ) = A·[(1 - β/2) + (3β/4)·(1 - cos(2θ))]
             = A·[(1 + β/4) - (3β/4)·cos(2θ)]
        
        So the histogram has:
        - DC component: a₀ ∝ (1 + β/4)
        - cos(2θ) component: a₂ ∝ -(3β/4)
        
        Therefore: β = -4·a₂/(3·a₀) when normalized
        
        But we need to be careful about the FFT normalization.
        """
        window = max(2.5 * sigma, 0.8)
        mask = (self.r >= r0 - window) & (self.r < r0 + window)
        theta_peak = self.theta[mask]
        n_points = len(theta_peak)
        
        if n_points < 100:
            return 0.0
        
        # Histogram of angles
        n_bins = min(max(36, int(np.sqrt(n_points))), 180)
        hist, edges = np.histogram(theta_peak, bins=n_bins, range=(-np.pi, np.pi))
        theta_centers = (edges[:-1] + edges[1:]) / 2
        hist = hist.astype(float)
        
        # Light smoothing
        hist = gaussian_filter1d(hist, sigma=max(1, n_bins // 60))
        
        # Method 1: Direct fitting (more robust for extreme β)
        # Fit I(θ) = A·[1 + β·P₂(sin θ)]
        try:
            from scipy.optimize import curve_fit
            
            def model(theta, A, beta):
                return A * (1 + beta * P2(np.sin(theta)))
            
            # Initial guess
            A_init = np.mean(hist)
            
            # Estimate β from the ratio of max to min
            hist_max = np.max(hist)
            hist_min = np.min(hist)
            if hist_min > 0:
                ratio = hist_max / hist_min
                # For β > 0: max at θ=±π/2, min at θ=0,π
                # I_max/I_min = (1 + β)/(1 - β/2)
                # For β < 0: max at θ=0,π, min at θ=±π/2
                # Need to check which case
                idx_max = np.argmax(hist)
                theta_max = theta_centers[idx_max]
                if abs(abs(theta_max) - np.pi/2) < np.pi/4:
                    # Max near ±π/2 → β > 0
                    beta_init = 2 * (ratio - 1) / (ratio + 2)
                else:
                    # Max near 0 or π → β < 0
                    beta_init = -2 * (ratio - 1) / (2*ratio + 1)
                beta_init = np.clip(beta_init, -1, 2)
            else:
                beta_init = 0.0
            
            popt, _ = curve_fit(
                model, theta_centers, hist,
                p0=[A_init, beta_init],
                bounds=([0, -1], [np.inf, 2]),
                maxfev=2000
            )
            beta = popt[1]
            
        except:
            # Fallback to FFT method
            fft = np.fft.fft(hist)
            a0 = np.abs(fft[0]) / n_bins
            a2 = np.real(fft[2]) * 2 / n_bins
            
            if a0 < 1e-10:
                return 0.0
            
            beta = -4 * a2 / (3 * a0)
        
        return np.clip(beta, self.BETA_MIN, self.BETA_MAX)
    
    def _fit_amplitudes(self, r_centers: np.ndarray, counts: np.ndarray,
                        peaks: List[dict]) -> List[dict]:
        """
        Fit peak amplitudes using linear least squares.
        
        Model: counts(r) = Σ amp_i · model_i(r) + background
        """
        n_peaks = len(peaks)
        if n_peaks == 0:
            return []
        
        dr = r_centers[1] - r_centers[0]
        
        # Build model matrix
        # Each column is the expected counts for unit amplitude
        A = np.zeros((len(r_centers), n_peaks + 1))
        
        for i, p in enumerate(peaks):
            r0, sigma = p['r0'], p['sigma']
            # Abel projection model converted to counts
            model_density = abel_projection(r_centers, r0, sigma)
            model_counts = 2 * np.pi * np.maximum(r_centers, 0.1) * dr * model_density
            A[:, i] = model_counts
        
        # Background column (constant)
        A[:, -1] = 1.0
        
        # Solve least squares with non-negativity
        try:
            from scipy.optimize import nnls
            x, _ = nnls(A, counts)
        except:
            # Fallback to simple least squares
            x, _, _, _ = np.linalg.lstsq(A, counts, rcond=None)
            x = np.maximum(x, 0)
        
        # Update amplitudes
        for i, p in enumerate(peaks):
            p['amp'] = x[i]
        
        return peaks
    
    def reconstruct(self, n_peaks: int = None, verbose: bool = True) -> List[PeakResult]:
        """Main reconstruction method"""
        if verbose:
            print("=" * 60)
            print("VMI Reconstructor v4.0")
            print("=" * 60)
            print(f"Events: {self.n_events:,}")
            print(f"Center: ({self.center[0]:.3f}, {self.center[1]:.3f}) mm")
            print(f"r_max: {self.r_max:.2f} mm, dr: {self.dr:.3f} mm")
        
        # Get distributions
        r_centers, counts, density = self._compute_distributions()
        
        # Detect peaks
        candidates = self._detect_peaks(r_centers, counts, density)
        
        if verbose:
            print(f"Detected {len(candidates)} peak candidates")
        
        # Smart peak selection when n_peaks is specified
        if n_peaks is not None and len(candidates) > n_peaks:
            # Strategy: Select n_peaks that are well-distributed
            # 1. Start with the highest prominence peak
            # 2. Add peaks that are sufficiently separated from already selected
            # 3. Prefer higher prominence among candidates at similar positions
            
            selected = []
            remaining = list(candidates)
            
            # Sort by prominence (highest first)
            remaining.sort(key=lambda c: c.prominence, reverse=True)
            
            # Minimum separation between peaks (adaptive based on r_max)
            min_sep = self.r_max / (n_peaks + 1) * 0.5
            
            while len(selected) < n_peaks and remaining:
                # Take the highest prominence remaining candidate
                best = remaining.pop(0)
                
                # Check if it's sufficiently separated from already selected
                too_close = False
                for s in selected:
                    if abs(best.r - s.r) < min_sep:
                        too_close = True
                        break
                
                if not too_close:
                    selected.append(best)
                elif len(remaining) == 0 and len(selected) < n_peaks:
                    # If we're running out of candidates, accept closer peaks
                    selected.append(best)
            
            candidates = selected
        elif n_peaks is not None:
            candidates = candidates[:n_peaks]
        
        if len(candidates) == 0:
            if verbose:
                print("No peaks detected!")
            return []
        
        # Refine positions and estimate parameters
        peaks_data = []
        for c in sorted(candidates, key=lambda x: x.r):
            r0 = self._refine_position(c.r, c.sigma)
            sigma = self._estimate_sigma(r_centers, density, r0)
            beta = self._estimate_beta(r0, sigma)
            
            peaks_data.append({
                'r0': r0,
                'sigma': sigma,
                'beta': beta,
                'amp': c.amplitude
            })
        
        # Fit amplitudes
        peaks_data = self._fit_amplitudes(r_centers, counts, peaks_data)
        
        # Create results
        self.peaks = []
        for i, p in enumerate(peaks_data):
            self.peaks.append(PeakResult(
                r0=p['r0'], sigma=p['sigma'], amp=p['amp'], beta=p['beta'],
                r0_err=0, sigma_err=0, beta_err=0
            ))
            if verbose:
                print(f"Peak {i+1}: r0={p['r0']:.3f} mm, σ={p['sigma']:.3f} mm, "
                      f"β={p['beta']:.2f}, amp={p['amp']:.1f}")
        
        if verbose:
            print("=" * 60)
        
        return self.peaks


# Alias for compatibility
ImprovedVMIReconstructor = VMIReconstructorV4
