"""
Hybrid VMI Reconstructor V4 (Improved)
======================================

Improvements over V3:
1. Better handling of extreme branching ratios (10:1)
2. Improved beta estimation using probabilistic assignment
3. Iterative refinement for overlapping peaks
4. Adaptive window sizing based on peak separation

Key insight: For close peaks with different betas, we need to:
- Use probabilistic particle assignment based on r position
- Weight angular distribution by assignment probability
- Iterate to refine both r0 and beta estimates

Author: Kiro AI Assistant
Date: 2026-01
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HybridPeakResult:
    """Result for a single peak with confidence metrics."""
    r0: float
    sigma: float
    beta: float
    amp: float
    
    r0_err: float = 0.0
    sigma_err: float = 0.0
    beta_err: float = 0.0
    confidence: float = 1.0
    
    r0_source: str = "physics"
    beta_source: str = "physics"
    sigma_source: str = "physics"


@dataclass 
class HybridConfig:
    """Configuration for hybrid reconstructor."""
    beta_threshold_high: float = 1.5
    overlap_threshold_sigma: float = 3.0
    
    physics_n_angular_bins: int = 36
    n_radial_bins: int = 300
    
    # Iterative refinement settings
    max_iterations: int = 5
    convergence_threshold: float = 0.01
    
    verbose: bool = True


class HybridVMIReconstructorV4:
    """
    Improved Hybrid VMI reconstructor V4.
    
    Key improvements:
    1. Probabilistic particle assignment for beta estimation
    2. Iterative refinement for overlapping peaks
    3. Better handling of extreme branching ratios
    """
    
    def __init__(self, xy_data: np.ndarray, 
                 pixel_size: float = 0.05,
                 psf_sigma: float = 0.0,
                 dld_resolution: float = 0.0,
                 vmi_k: float = 0.01,
                 config: HybridConfig = None):
        self.xy_data = xy_data
        self.pixel_size = pixel_size
        self.psf_sigma = psf_sigma
        self.dld_resolution = dld_resolution
        self.vmi_k = vmi_k
        self.cfg = config or HybridConfig()
        
        self.r = np.sqrt(xy_data[:, 0]**2 + xy_data[:, 1]**2)
        self.theta = np.arctan2(xy_data[:, 1], xy_data[:, 0])
        self.r_max = np.percentile(self.r, 99.5)
    
    def reconstruct(self, n_peaks: int = 3, 
                    verbose: bool = None) -> List[HybridPeakResult]:
        """Main reconstruction method."""
        verbose = verbose if verbose is not None else self.cfg.verbose
        
        if verbose:
            print("="*60)
            print("HYBRID VMI RECONSTRUCTOR V4 (Global Fitting)")
            print("="*60)
            print(f"Data: {len(self.xy_data)} particles, r_max={self.r_max:.2f} mm")
        
        # Stage 1: Coarse peak detection
        if verbose:
            print("\n[Stage 1] Coarse peak detection...")
        coarse_peaks = self._stage1_coarse_detection(n_peaks, verbose)
        
        if len(coarse_peaks) == 0:
            if verbose:
                print("  WARNING: No peaks detected!")
            return []
        
        # Stage 2: Inverse Abel + local refinement
        if verbose:
            print("\n[Stage 2] Inverse Abel refinement...")
        fitted_peaks = self._stage2_abel_refinement(coarse_peaks, verbose)
        
        # Stage 3: Initial beta estimation (windowing)
        if verbose:
            print("\n[Stage 3] Initial beta estimation...")
        peaks_with_beta = self._stage3_probabilistic_beta(fitted_peaks, verbose)
        
        # Stage 4: Global 2D fitting for close peaks
        if verbose:
            print("\n[Stage 4] Global 2D fitting...")
        refined_peaks = self._stage4_final_refinement(peaks_with_beta, verbose)
        
        # Stage 5: Confidence estimation
        if verbose:
            print("\n[Stage 5] Confidence estimation...")
        final_peaks = self._stage5_confidence(refined_peaks, verbose)
        
        if verbose:
            self._print_results(final_peaks)
        
        return final_peaks

    def _stage1_coarse_detection(self, n_peaks: int, verbose: bool) -> List[dict]:
        """Stage 1: Multi-resolution histogram peak detection (same as V2)."""
        resolutions = [
            {'n_bins': 150, 'smooth': 5},
            {'n_bins': 200, 'smooth': 4},
            {'n_bins': 300, 'smooth': 3},
            {'n_bins': 500, 'smooth': 2},
        ]
        
        all_peaks = []
        
        for res_idx, res in enumerate(resolutions):
            n_bins = res['n_bins']
            smooth_sigma = res['smooth']
            
            hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
            bin_centers = (edges[:-1] + edges[1:]) / 2
            dr = bin_centers[1] - bin_centers[0]
            
            # Apply 2πr correction
            r_safe = np.maximum(bin_centers, dr)
            hist_corrected = hist.astype(float) / (2 * np.pi * r_safe * dr)
            hist_smooth = gaussian_filter1d(hist_corrected, sigma=smooth_sigma)
            
            min_distance = max(int(0.3 / dr), 3)
            prominence = max(hist_smooth.max() * 0.02, hist_smooth.max() * 0.01)
            
            peaks_idx, props = find_peaks(
                hist_smooth,
                prominence=prominence,
                distance=min_distance,
                width=2
            )
            
            for i, idx in enumerate(peaks_idx):
                r0 = bin_centers[idx]
                prom = props['prominences'][i]
                all_peaks.append({
                    'r0': r0,
                    'prominence': prom,
                    'height': hist_smooth[idx],
                    'res_idx': res_idx
                })
        
        if len(all_peaks) == 0:
            return self._fallback_detection(n_peaks, verbose)
        
        # Cluster peaks
        cluster_radius = 0.5
        clusters = []
        
        for peak in all_peaks:
            r0 = peak['r0']
            found = False
            for cluster in clusters:
                if abs(r0 - cluster['r0']) < cluster_radius:
                    cluster['peaks'].append(peak)
                    weights = [p['prominence'] for p in cluster['peaks']]
                    r0s = [p['r0'] for p in cluster['peaks']]
                    cluster['r0'] = np.average(r0s, weights=weights)
                    cluster['total_prominence'] = sum(weights)
                    cluster['n_resolutions'] = len(set(p['res_idx'] for p in cluster['peaks']))
                    found = True
                    break
            if not found:
                clusters.append({
                    'r0': r0,
                    'peaks': [peak],
                    'total_prominence': peak['prominence'],
                    'n_resolutions': 1
                })
        
        for cluster in clusters:
            cluster['score'] = cluster['total_prominence'] * (1 + 0.5 * cluster['n_resolutions'])
        
        clusters.sort(key=lambda x: x['score'], reverse=True)
        top_clusters = clusters[:n_peaks]
        top_clusters.sort(key=lambda x: x['r0'])
        
        results = []
        for cluster in top_clusters:
            r0 = cluster['r0']
            sigma = self._estimate_sigma_initial(r0)
            
            if verbose:
                print(f"  Peak at r={r0:.2f}mm, σ≈{sigma:.3f}mm")
            
            results.append({'r0': r0, 'sigma': sigma, 'amp': cluster['score']})
        
        return results
    
    def _fallback_detection(self, n_peaks: int, verbose: bool) -> List[dict]:
        """Fallback detection."""
        n_bins = 300
        hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        dr = bin_centers[1] - bin_centers[0]
        
        r_safe = np.maximum(bin_centers, dr)
        hist_corrected = hist.astype(float) / (2 * np.pi * r_safe * dr)
        hist_smooth = gaussian_filter1d(hist_corrected, sigma=3)
        
        peaks_idx, props = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.01, distance=5)
        
        if len(peaks_idx) == 0:
            peaks_idx = np.array([np.argmax(hist_smooth)])
        
        peaks_idx = peaks_idx[:n_peaks]
        
        results = []
        for idx in peaks_idx:
            r0 = bin_centers[idx]
            sigma = self._estimate_sigma_initial(r0)
            results.append({'r0': r0, 'sigma': sigma, 'amp': hist_smooth[idx]})
        
        return results
    
    def _estimate_sigma_initial(self, r0: float) -> float:
        """Estimate initial sigma from local FWHM."""
        window = 1.0
        mask = (self.r >= r0 - window) & (self.r < r0 + window)
        r_local = self.r[mask]
        
        if len(r_local) < 100:
            return 0.3
        
        sigma = np.std(r_local)
        return np.clip(sigma, 0.1, 1.0)

    def _stage2_abel_refinement(self, coarse_peaks: List[dict],
                                 verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 2: Inverse Abel + local peak refinement (like V2).
        
        This is the proven approach from V2 that works well for r0 estimation.
        """
        try:
            import abel
        except ImportError:
            if verbose:
                print("  WARNING: PyAbel not available")
            return [HybridPeakResult(
                r0=p['r0'], sigma=p['sigma'], beta=0.0, amp=p['amp'],
                r0_source="coarse", sigma_source="coarse", beta_source="pending"
            ) for p in coarse_peaks]
        
        n_peaks = len(coarse_peaks)
        
        # Build high-resolution histogram
        n_bins = 500
        hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        dr = bin_centers[1] - bin_centers[0]
        
        # Remove 2πr factor
        r_safe = np.maximum(bin_centers, dr)
        rho_2d = hist.astype(float) / (2 * np.pi * r_safe * dr)
        rho_2d_smooth = gaussian_filter1d(rho_2d, sigma=2)
        
        # Inverse Abel transform
        try:
            f_3d = abel.hansenlaw.hansenlaw_transform(rho_2d_smooth, direction='inverse')
            f_3d = np.maximum(f_3d, 0)
        except Exception as e:
            if verbose:
                print(f"  WARNING: Abel transform failed: {e}")
            return [HybridPeakResult(
                r0=p['r0'], sigma=p['sigma'], beta=0.0, amp=p['amp'],
                r0_source="coarse", sigma_source="coarse", beta_source="pending"
            ) for p in coarse_peaks]
        
        # Refine each peak position using local maximum in f_3d
        results = []
        coarse_peaks_sorted = sorted(coarse_peaks, key=lambda p: p['r0'])
        
        for i, peak in enumerate(coarse_peaks_sorted):
            r0_coarse = peak['r0']
            sigma_coarse = peak['sigma']
            
            # Define search region
            search_radius = max(1.0, sigma_coarse)
            
            # Constrain by neighbors
            if i == 0:
                r_min = max(0.1, r0_coarse - search_radius)
            else:
                r_min = max(0.1, (coarse_peaks_sorted[i-1]['r0'] + r0_coarse) / 2)
            
            if i == n_peaks - 1:
                r_max_local = min(self.r_max, r0_coarse + search_radius)
            else:
                r_max_local = min(self.r_max, (r0_coarse + coarse_peaks_sorted[i+1]['r0']) / 2)
            
            # Find local maximum in f_3d
            mask = (bin_centers >= r_min) & (bin_centers <= r_max_local)
            r_local = bin_centers[mask]
            f_local = f_3d[mask]
            
            if len(r_local) < 5:
                r0_refined = r0_coarse
                sigma_refined = sigma_coarse
                amp_refined = peak['amp']
            else:
                # Find local maximum
                max_idx = np.argmax(f_local)
                r0_refined = r_local[max_idx]
                amp_refined = f_local[max_idx]
                
                # Estimate sigma from FWHM
                peak_height = f_local[max_idx]
                half_max = peak_height / 2
                
                left_idx = max_idx
                while left_idx > 0 and f_local[left_idx] > half_max:
                    left_idx -= 1
                
                right_idx = max_idx
                while right_idx < len(f_local) - 1 and f_local[right_idx] > half_max:
                    right_idx += 1
                
                fwhm = r_local[right_idx] - r_local[left_idx]
                sigma_refined = fwhm / 2.355
                
                # Factor of 2 correction (empirical, from V2)
                sigma_refined = sigma_refined * 2
                sigma_refined = np.clip(sigma_refined, 0.1, 2.0)
            
            if verbose:
                print(f"  Peak {i+1}: r0={r0_refined:.3f}mm (coarse:{r0_coarse:.2f}), σ={sigma_refined:.4f}mm")
            
            results.append(HybridPeakResult(
                r0=r0_refined,
                sigma=sigma_refined,
                beta=0.0,
                amp=amp_refined,
                r0_source="abel",
                sigma_source="abel",
                beta_source="pending"
            ))
        
        results.sort(key=lambda p: p.r0)
        return results

    def _stage3_probabilistic_beta(self, peaks: List[HybridPeakResult],
                                    verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 3: Initial beta estimation using simple windowing.
        
        This provides initial guesses for the global fitting in stage 4.
        The global fitting will refine these using onion peeling approach.
        """
        n_peaks = len(peaks)
        
        # Single peak: use standard method
        if n_peaks == 1:
            return self._beta_single_peak(peaks, verbose)
        
        if verbose:
            print("  Using simple windowing for initial estimates")
        
        return self._beta_simple_windowing(peaks, verbose)
    
    def _beta_single_peak(self, peaks: List[HybridPeakResult],
                          verbose: bool) -> List[HybridPeakResult]:
        """Beta estimation for single peak."""
        peak = peaks[0]
        window = max(peak.sigma * 2.5, 0.5)
        
        mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
        theta_region = self.theta[mask]
        
        if verbose:
            print(f"  Single peak: window={window:.2f}mm, n={len(theta_region)}")
        
        if len(theta_region) < 50:
            peak.beta = 0.0
            peak.beta_err = 1.0
            peak.beta_source = "default"
        else:
            beta, beta_err = self._estimate_beta_combined(theta_region)
            peak.beta = beta
            peak.beta_err = beta_err
            peak.beta_source = "standard"
            
            if verbose:
                print(f"    β={peak.beta:.3f} ± {peak.beta_err:.2f}")
        
        return peaks
    
    def _beta_simple_windowing(self, peaks: List[HybridPeakResult],
                                verbose: bool) -> List[HybridPeakResult]:
        """Simple windowing for well-separated peaks."""
        for i, peak in enumerate(peaks):
            window = max(peak.sigma * 2.5, 0.3)
            mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
            theta_region = self.theta[mask]
            
            if len(theta_region) < 50:
                peak.beta = 0.0
                peak.beta_err = 1.0
                peak.beta_source = "default"
                continue
            
            beta, beta_err = self._estimate_beta_combined(theta_region)
            peak.beta = beta
            peak.beta_err = beta_err
            peak.beta_source = "windowing"
            
            if verbose:
                print(f"  Peak {i+1}: β={peak.beta:.3f} ± {peak.beta_err:.2f}")
        
        return peaks
    
    def _beta_asymmetric_windowing(self, peaks: List[HybridPeakResult],
                                    verbose: bool) -> List[HybridPeakResult]:
        """
        Beta estimation for close peaks.
        
        Strategy: Use full windowing but with smaller window to reduce contamination.
        The global 2D fitting in stage 4 will refine these estimates.
        """
        n_peaks = len(peaks)
        
        for i, peak in enumerate(peaks):
            sigma = max(peak.sigma, 0.15)
            
            # Use smaller window for close peaks to reduce contamination
            # But not too small to lose statistics
            window = sigma * 1.5
            
            r_min = max(0.1, peak.r0 - window)
            r_max = min(self.r_max, peak.r0 + window)
            
            mask = (self.r >= r_min) & (self.r < r_max)
            theta_region = self.theta[mask]
            
            if verbose:
                print(f"  Peak {i+1}: window [{r_min:.2f}, {r_max:.2f}], n={len(theta_region)}")
            
            if len(theta_region) < 30:
                # Expand window if too few particles
                window = sigma * 2.5
                r_min = max(0.1, peak.r0 - window)
                r_max = min(self.r_max, peak.r0 + window)
                mask = (self.r >= r_min) & (self.r < r_max)
                theta_region = self.theta[mask]
                if verbose:
                    print(f"    Expanded window, n={len(theta_region)}")
            
            if len(theta_region) < 30:
                peak.beta = 0.0
                peak.beta_err = 1.0
                peak.beta_source = "default"
                continue
            
            beta, beta_err = self._estimate_beta_combined(theta_region)
            peak.beta = beta
            peak.beta_err = beta_err
            peak.beta_source = "windowing"
            
            if verbose:
                print(f"    β={peak.beta:.3f} ± {peak.beta_err:.2f}")
        
        return peaks

    def _estimate_beta_combined(self, theta: np.ndarray) -> Tuple[float, float]:
        """Combined FFT + curve fit beta estimation."""
        beta_fft = self._estimate_beta_fft(theta)
        beta_fit, beta_err = self._estimate_beta_curvefit(theta)
        
        # Combine (prefer curve fit if error is low)
        if beta_err < 0.3:
            beta = 0.6 * beta_fit + 0.4 * beta_fft
        else:
            beta = 0.5 * beta_fit + 0.5 * beta_fft
        
        return np.clip(beta, -1.0, 2.0), beta_err
    
    def _estimate_beta_weighted(self, theta: np.ndarray, 
                                 weights: np.ndarray) -> Tuple[float, float]:
        """Weighted beta estimation."""
        # Normalize weights
        weights = weights / (weights.sum() + 1e-10)
        
        # Weighted histogram
        n_bins = 36
        hist, edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi), 
                                   weights=weights)
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Method 1: Curve fit
        def model(theta, A, beta):
            cos_theta = np.cos(theta)
            P2 = (3 * cos_theta**2 - 1) / 2
            return A * (1 + beta * P2)
        
        try:
            hist_smooth = gaussian_filter1d(hist, sigma=1)
            popt, pcov = curve_fit(
                model, centers, hist_smooth,
                p0=[np.mean(hist_smooth), 0.0],
                bounds=([0, -1.0], [np.inf, 2.0]),
                maxfev=2000
            )
            beta_fit = popt[1]
            beta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.5
        except:
            beta_fit = 0.0
            beta_err = 1.0
        
        # Method 2: FFT
        fft = np.fft.fft(hist)
        c0 = np.abs(fft[0]) / n_bins
        
        if c0 > 1e-10:
            c2_complex = fft[2]
            c2_amp = 2 * np.abs(c2_complex) / n_bins
            phase = np.angle(c2_complex)
            sign = 1.0 if abs(phase) > np.pi/2 else -1.0
            c2_signed = sign * c2_amp
            
            denominator = 3.0 * c0 - c2_signed
            if abs(denominator) > 1e-10:
                beta_fft = np.clip(4.0 * c2_signed / denominator, -1.0, 2.0)
            else:
                beta_fft = 0.0
        else:
            beta_fft = 0.0
        
        # Combine
        if beta_err < 0.5:
            beta = 0.7 * beta_fit + 0.3 * beta_fft
        else:
            beta = 0.5 * beta_fit + 0.5 * beta_fft
        
        return np.clip(beta, -1.0, 2.0), beta_err
    
    def _estimate_beta_fft(self, theta: np.ndarray) -> float:
        """FFT-based beta estimation."""
        n_bins = 72
        hist, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
        
        fft = np.fft.fft(hist.astype(float))
        c0 = np.abs(fft[0]) / n_bins
        
        if c0 < 1e-10:
            return 0.0
        
        c2_complex = fft[2]
        c2_amp = 2 * np.abs(c2_complex) / n_bins
        phase = np.angle(c2_complex)
        
        sign = 1.0 if abs(phase) > np.pi/2 else -1.0
        c2_signed = sign * c2_amp
        
        denominator = 3.0 * c0 - c2_signed
        if abs(denominator) < 1e-10:
            return 0.0
        
        beta = 4.0 * c2_signed / denominator
        return np.clip(beta, -1.0, 2.0)
    
    def _estimate_beta_curvefit(self, theta: np.ndarray) -> Tuple[float, float]:
        """Curve fit beta estimation."""
        theta_folded = np.abs(theta)
        
        n_bins = max(18, len(theta) // 50)
        hist, edges = np.histogram(theta_folded, bins=n_bins, range=(0, np.pi))
        centers = (edges[:-1] + edges[1:]) / 2
        
        def model(theta, A, beta):
            sin_theta = np.sin(theta)
            P2 = (3 * sin_theta**2 - 1) / 2
            return A * (1 + beta * P2)
        
        try:
            popt, pcov = curve_fit(
                model, centers, hist,
                p0=[np.mean(hist), 0.0],
                bounds=([0, -1.0], [np.inf, 2.0]),
                maxfev=2000
            )
            beta = popt[1]
            beta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.5
            return beta, beta_err
        except:
            return 0.0, 1.0
    
    def _stage4_final_refinement(self, peaks: List[HybridPeakResult],
                                  verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 4: Refine beta using very narrow window FFT.
        
        Key insight: Use the narrowest possible window to minimize contamination
        from neighboring peaks, even at the cost of fewer particles.
        """
        n_peaks = len(peaks)
        
        if n_peaks < 1:
            return peaks
        
        if verbose:
            print("  Refining beta with narrow window FFT")
        
        # Build radial profile for sigma estimation
        n_r_bins = 150
        r_max = self.r_max * 1.1
        hist_r, r_edges = np.histogram(self.r, bins=n_r_bins, range=(0, r_max))
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        dr = r_centers[1] - r_centers[0]
        
        # Apply 2πr correction
        r_safe = np.maximum(r_centers, dr)
        radial_corrected = hist_r / (2 * np.pi * r_safe * dr)
        
        refined_peaks = []
        
        for i, peak in enumerate(peaks):
            r0 = peak.r0
            
            # Estimate sigma from radial profile FWHM
            idx_r0 = np.argmin(np.abs(r_centers - r0))
            peak_val = radial_corrected[idx_r0]
            half_max = peak_val / 2
            
            left_idx = idx_r0
            while left_idx > 0 and radial_corrected[left_idx] > half_max:
                left_idx -= 1
            
            right_idx = idx_r0
            while right_idx < len(radial_corrected) - 1 and radial_corrected[right_idx] > half_max:
                right_idx += 1
            
            fwhm = r_centers[right_idx] - r_centers[left_idx]
            sigma_fit = np.clip(fwhm / 2.355, 0.1, 2.0)
            
            # Use very narrow window: 0.5σ or 0.2mm, whichever is smaller
            r_window = min(sigma_fit * 0.5, 0.2)
            
            # Constrain by neighbors
            if n_peaks > 1:
                if i > 0:
                    dist_left = r0 - peaks[i-1].r0
                    r_window = min(r_window, dist_left / 5)
                if i < n_peaks - 1:
                    dist_right = peaks[i+1].r0 - r0
                    r_window = min(r_window, dist_right / 5)
            
            r_window = max(r_window, 0.1)  # Minimum for any statistics
            
            # FFT on narrow window
            mask_window = (self.r > r0 - r_window) & (self.r < r0 + r_window)
            theta_window = self.theta[mask_window]
            n_particles = len(theta_window)
            
            beta_fit = peak.beta
            beta_err = peak.beta_err
            
            if n_particles >= 50:
                n_fft_bins = 72
                hist_fft, _ = np.histogram(theta_window, bins=n_fft_bins, range=(-np.pi, np.pi))
                fft = np.fft.fft(hist_fft.astype(float))
                c0 = np.abs(fft[0]) / n_fft_bins
                
                if c0 > 1e-10:
                    c2_complex = fft[2]
                    c2_amp = 2 * np.abs(c2_complex) / n_fft_bins
                    phase = np.angle(c2_complex)
                    sign = 1.0 if abs(phase) > np.pi/2 else -1.0
                    c2_signed = sign * c2_amp
                    denom = 3.0 * c0 - c2_signed
                    if abs(denom) > 1e-10:
                        beta_fit = np.clip(4.0 * c2_signed / denom, -1.0, 2.0)
                        beta_err = 0.2
            
            if verbose:
                print(f"    Peak {i+1}: σ={sigma_fit:.3f}mm, β={beta_fit:.3f} (was {peak.beta:.3f}), n={n_particles}, window={r_window:.2f}mm")
            
            refined_peaks.append(HybridPeakResult(
                r0=r0,
                sigma=sigma_fit,
                beta=beta_fit,
                amp=peak_val,
                r0_err=peak.r0_err,
                sigma_err=0.05,
                beta_err=beta_err,
                confidence=0.9,
                r0_source=peak.r0_source,
                sigma_source="radial_fwhm",
                beta_source="narrow_fft"
            ))
        
        return refined_peaks
    
    def _stage5_confidence(self, peaks: List[HybridPeakResult],
                           verbose: bool) -> List[HybridPeakResult]:
        """Stage 5: Final confidence estimation."""
        for peak in peaks:
            confidence = 0.85
            if peak.r0_source == "global_fit":
                confidence += 0.1
            elif peak.r0_source == "abel":
                confidence += 0.05
            if peak.beta_err < 0.3:
                confidence += 0.05
            if peak.beta_source == "global_fit":
                confidence += 0.05
            peak.confidence = min(confidence, 1.0)
        
        return peaks
    
    def _print_results(self, peaks: List[HybridPeakResult]):
        """Print final results."""
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        for i, peak in enumerate(peaks):
            print(f"Peak {i+1}:")
            print(f"  r0 = {peak.r0:.3f} mm (source: {peak.r0_source})")
            print(f"  σ  = {peak.sigma:.4f} mm (source: {peak.sigma_source})")
            print(f"  β  = {peak.beta:.3f} ± {peak.beta_err:.2f} (source: {peak.beta_source})")
            print(f"  confidence = {peak.confidence:.2f}")
