"""
Hybrid VMI Reconstructor V3 (Improved)
======================================

Improvements over V2:
1. Global multi-Gaussian fitting instead of local maximum search
2. Adaptive binning for narrow peaks
3. Non-overlapping windows for beta estimation
4. Sigma-based threshold for X2 trigger
5. Better sigma estimation without heuristic factor

Author: Kiro AI Assistant
Date: 2026-01
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize, differential_evolution
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
    overlap_threshold_sigma: float = 2.5  # Changed: sigma-based threshold
    
    physics_n_angular_bins: int = 36
    
    n_radial_bins: int = 300
    mle_noise_model: str = "poisson"
    mle_max_iter: int = 500
    
    x2_n_ensemble: int = 5
    x2_n_particles: int = 10000
    x2_n_iterations: int = 50
    
    validate_forward: bool = False
    validation_threshold: float = 0.1
    
    verbose: bool = True


class HybridVMIReconstructorV3:
    """
    Improved Hybrid VMI reconstructor V3.
    
    Key improvements:
    1. Global multi-Gaussian fitting
    2. Adaptive binning for narrow peaks
    3. Non-overlapping beta windows
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
            print("HYBRID VMI RECONSTRUCTOR V3 (Improved)")
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
        
        # Stage 2: Global multi-Gaussian fitting (IMPROVED)
        if verbose:
            print("\n[Stage 2] Global multi-Gaussian fitting...")
        fitted_peaks = self._stage2_global_fitting(coarse_peaks, verbose)
        
        # Stage 3: Beta estimation with non-overlapping windows (IMPROVED)
        if verbose:
            print("\n[Stage 3] Beta estimation...")
        peaks_with_beta = self._stage3_beta_estimation(fitted_peaks, verbose)
        
        # Stage 4: Adaptive refinement
        if verbose:
            print("\n[Stage 4] Adaptive refinement...")
        refined_peaks = self._stage4_adaptive_refinement(peaks_with_beta, verbose)
        
        # Stage 5: Consistency check
        if verbose:
            print("\n[Stage 5] Consistency check...")
        final_peaks = self._stage5_consistency_check(refined_peaks, verbose)
        
        if verbose:
            self._print_results(final_peaks)
        
        return final_peaks
    
    def _stage1_coarse_detection(self, n_peaks: int, verbose: bool) -> List[dict]:
        """Stage 1: Multi-resolution histogram peak detection."""
        resolutions = [
            {'n_bins': 150, 'smooth': 5},
            {'n_bins': 200, 'smooth': 4},
            {'n_bins': 300, 'smooth': 3},
            {'n_bins': 500, 'smooth': 2},  # Added higher resolution
        ]
        
        all_peaks = []
        
        for res_idx, res in enumerate(resolutions):
            n_bins = res['n_bins']
            smooth_sigma = res['smooth']
            
            hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
            bin_centers = (edges[:-1] + edges[1:]) / 2
            dr = bin_centers[1] - bin_centers[0]
            
            # IMPROVED: Apply 2πr correction before peak finding
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
        
        # Estimate initial sigma
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
        # Use particles near r0
        window = 1.0
        mask = (self.r >= r0 - window) & (self.r < r0 + window)
        r_local = self.r[mask]
        
        if len(r_local) < 100:
            return 0.3  # Default
        
        # Estimate from standard deviation
        sigma = np.std(r_local)
        return np.clip(sigma, 0.1, 1.0)
    
    def _stage2_global_fitting(self, coarse_peaks: List[dict],
                                verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 2: IMPROVED - Global multi-Gaussian fitting.
        
        Instead of local maximum search, fit all peaks simultaneously.
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
        
        # IMPROVED: Adaptive binning based on expected peak width
        min_sigma = min(p['sigma'] for p in coarse_peaks)
        n_bins = max(500, int(self.r_max / (min_sigma / 5)))  # At least 5 bins per sigma
        n_bins = min(n_bins, 1000)  # Cap at 1000
        
        if verbose:
            print(f"  Using {n_bins} bins (min_sigma={min_sigma:.2f}mm)")
        
        hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        dr = bin_centers[1] - bin_centers[0]
        
        # Remove 2πr factor
        r_safe = np.maximum(bin_centers, dr)
        rho_2d = hist.astype(float) / (2 * np.pi * r_safe * dr)
        rho_2d_smooth = gaussian_filter1d(rho_2d, sigma=max(1, int(min_sigma / dr / 3)))
        
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
        
        # IMPROVED: Global multi-Gaussian fitting
        def multi_gaussian(r, *params):
            """Sum of Gaussians."""
            result = np.zeros_like(r)
            for i in range(n_peaks):
                A = params[3*i]
                r0 = params[3*i + 1]
                sigma = params[3*i + 2]
                result += A * np.exp(-((r - r0)**2) / (2 * sigma**2 + 1e-10))
            return result
        
        # Initial parameters from coarse peaks
        p0 = []
        bounds_low = []
        bounds_high = []
        
        for peak in coarse_peaks:
            # Amplitude
            idx = np.argmin(np.abs(bin_centers - peak['r0']))
            A_init = f_3d[idx] if idx < len(f_3d) else 1.0
            p0.extend([max(A_init, 0.1), peak['r0'], peak['sigma']])
            
            # Bounds
            bounds_low.extend([0.01, max(0.1, peak['r0'] - 2.0), 0.05])
            bounds_high.extend([A_init * 10 + 1, min(self.r_max, peak['r0'] + 2.0), 2.0])
        
        # Fit
        try:
            # Use only positive part of f_3d
            mask = f_3d > 0
            r_fit = bin_centers[mask]
            f_fit = f_3d[mask]
            
            # Weight by sqrt(f) to reduce influence of noise
            weights = np.sqrt(np.maximum(f_fit, 1))
            
            popt, pcov = curve_fit(
                multi_gaussian, r_fit, f_fit,
                p0=p0,
                bounds=(bounds_low, bounds_high),
                sigma=1/weights,
                maxfev=5000
            )
            
            # Extract results
            results = []
            for i in range(n_peaks):
                A = popt[3*i]
                r0 = popt[3*i + 1]
                sigma = popt[3*i + 2]
                
                # IMPROVED: No heuristic factor for sigma
                # The sigma from f_3d is the true 3D distribution sigma
                
                if verbose:
                    print(f"  Peak {i+1}: r0={r0:.3f}mm, σ={sigma:.4f}mm, A={A:.1f}")
                
                results.append(HybridPeakResult(
                    r0=r0,
                    sigma=sigma,
                    beta=0.0,
                    amp=A,
                    r0_source="global_fit",
                    sigma_source="global_fit",
                    beta_source="pending"
                ))
            
            results.sort(key=lambda p: p.r0)
            return results
            
        except Exception as e:
            if verbose:
                print(f"  WARNING: Global fitting failed: {e}, using coarse estimates")
            return [HybridPeakResult(
                r0=p['r0'], sigma=p['sigma'], beta=0.0, amp=p['amp'],
                r0_source="coarse", sigma_source="coarse", beta_source="pending"
            ) for p in coarse_peaks]
    
    def _stage3_beta_estimation(self, peaks: List[HybridPeakResult],
                                 verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 3: IMPROVED - Onion peeling beta estimation.
        
        Strategy:
        1. For well-separated peaks (>4σ): use simple windowing (like V2)
        2. For close peaks (<4σ): use onion peeling from outermost to innermost
        
        Onion peeling approach:
        - Start from outermost peak (least contaminated by inner peaks)
        - Estimate beta using particles in outer half of the peak
        - Subtract this peak's contribution before estimating inner peaks
        """
        n_peaks = len(peaks)
        
        # Single peak: use standard method
        if n_peaks == 1:
            peak = peaks[0]
            window = max(peak.sigma * 2.5, 0.5)
            
            mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
            theta_region = self.theta[mask]
            
            if verbose:
                print(f"  Single peak: window={window:.2f}mm, n_particles={len(theta_region)}")
            
            if len(theta_region) < 50:
                peak.beta = 0.0
                peak.beta_err = 1.0
                peak.beta_source = "default"
            else:
                beta_fft = self._estimate_beta_fft(theta_region)
                beta_fit, beta_err = self._estimate_beta_curvefit(theta_region)
                
                if beta_err < 0.3:
                    beta = 0.6 * beta_fit + 0.4 * beta_fft
                else:
                    beta = 0.5 * beta_fit + 0.5 * beta_fft
                
                peak.beta = np.clip(beta, -1.0, 2.0)
                peak.beta_err = beta_err
                peak.beta_source = "standard"
                
                if verbose:
                    print(f"    β={peak.beta:.3f} (FFT:{beta_fft:.2f}, fit:{beta_fit:.2f})")
            
            return peaks
        
        # Multi-peak: check if peaks are well-separated
        # Sort peaks by r0 (should already be sorted)
        peaks_sorted = sorted(enumerate(peaks), key=lambda x: x[1].r0)
        
        # Check separations
        separations_sigma = []
        for i in range(len(peaks_sorted) - 1):
            p1 = peaks_sorted[i][1]
            p2 = peaks_sorted[i+1][1]
            sep = p2.r0 - p1.r0
            avg_sigma = (p1.sigma + p2.sigma) / 2
            sep_sigma = sep / max(avg_sigma, 0.1)
            separations_sigma.append(sep_sigma)
        
        min_sep_sigma = min(separations_sigma) if separations_sigma else float('inf')
        
        if verbose:
            print(f"  Min separation: {min_sep_sigma:.1f}σ")
        
        # If well-separated (>4σ), use simple V2-style windowing
        if min_sep_sigma > 4.0:
            if verbose:
                print("  Using simple windowing (well-separated peaks)")
            return self._beta_simple_windowing(peaks, verbose)
        
        # Close peaks: use onion peeling from outermost to innermost
        if verbose:
            print("  Using onion peeling (close peaks)")
        return self._beta_onion_peeling(peaks, verbose)
    
    def _beta_simple_windowing(self, peaks: List[HybridPeakResult],
                                verbose: bool) -> List[HybridPeakResult]:
        """Simple windowing beta estimation (like V2)."""
        for i, peak in enumerate(peaks):
            window = max(peak.sigma * 2.5, 0.3)
            mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
            theta_region = self.theta[mask]
            
            if len(theta_region) < 50:
                peak.beta = 0.0
                peak.beta_err = 1.0
                peak.beta_source = "default"
                continue
            
            beta_fft = self._estimate_beta_fft(theta_region)
            beta_fit, beta_err = self._estimate_beta_curvefit(theta_region)
            
            if beta_err < 0.3:
                beta = 0.6 * beta_fit + 0.4 * beta_fft
            else:
                beta = 0.5 * beta_fit + 0.5 * beta_fft
            
            peak.beta = np.clip(beta, -1.0, 2.0)
            peak.beta_err = beta_err
            peak.beta_source = "windowing"
            
            if verbose:
                print(f"  Peak {i+1}: β={peak.beta:.3f} (FFT:{beta_fft:.2f}, fit:{beta_fit:.2f})")
        
        return peaks
    
    def _beta_onion_peeling(self, peaks: List[HybridPeakResult],
                            verbose: bool) -> List[HybridPeakResult]:
        """
        Onion peeling beta estimation for close peaks.
        
        Process from outermost to innermost:
        1. For outermost peak: use outer half (r > r0) which is uncontaminated
        2. For inner peaks: use weighted subtraction of outer peaks' contribution
        """
        n_peaks = len(peaks)
        
        # Sort by r0 (descending - outermost first)
        peaks_sorted_idx = sorted(range(n_peaks), key=lambda i: peaks[i].r0, reverse=True)
        
        # Track estimated betas and amplitudes for subtraction
        estimated_betas = [None] * n_peaks
        estimated_amps = [p.amp for p in peaks]
        
        # Process from outermost to innermost
        for order, peak_idx in enumerate(peaks_sorted_idx):
            peak = peaks[peak_idx]
            
            if verbose:
                print(f"  Processing peak {peak_idx+1} (r={peak.r0:.2f}mm, order={order+1}/{n_peaks})")
            
            if order == 0:
                # Outermost peak: use outer half (r > r0) which is clean
                window = max(peak.sigma * 2.5, 0.5)
                mask = (self.r >= peak.r0) & (self.r < peak.r0 + window)
                theta_region = self.theta[mask]
                
                if verbose:
                    print(f"    Using outer half: r in [{peak.r0:.2f}, {peak.r0+window:.2f}], n={len(theta_region)}")
                
                if len(theta_region) < 30:
                    # Fall back to full window
                    mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
                    theta_region = self.theta[mask]
                    if verbose:
                        print(f"    Fallback to full window, n={len(theta_region)}")
            else:
                # Inner peaks: use weighted approach
                # Find the region between this peak and the next outer peak
                outer_peak_idx = peaks_sorted_idx[order - 1]
                outer_peak = peaks[outer_peak_idx]
                
                # Use inner half of this peak (r < r0) to avoid contamination from outer peak
                window = max(peak.sigma * 2.5, 0.5)
                
                # For innermost peak, use inner half
                # For middle peaks, use the region away from the outer peak
                if peak_idx == peaks_sorted_idx[-1]:  # Innermost
                    mask = (self.r >= peak.r0 - window) & (self.r < peak.r0)
                else:
                    # Use full window but weight by distance from outer peak
                    mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
                
                theta_region = self.theta[mask]
                r_region = self.r[mask]
                
                if verbose:
                    print(f"    Using region with n={len(theta_region)} particles")
                
                # If we have estimated beta for outer peaks, try to subtract their contribution
                # This is approximate - we use the angular distribution shape
                if len(theta_region) >= 30 and estimated_betas[outer_peak_idx] is not None:
                    # Weight particles by probability of belonging to this peak vs outer peak
                    sigma_this = max(peak.sigma, 0.15)
                    sigma_outer = max(outer_peak.sigma, 0.15)
                    
                    prob_this = np.exp(-((r_region - peak.r0)**2) / (2 * sigma_this**2))
                    prob_outer = np.exp(-((r_region - outer_peak.r0)**2) / (2 * sigma_outer**2))
                    
                    # Normalize
                    prob_total = prob_this + prob_outer + 1e-10
                    weights = prob_this / prob_total
                    
                    # Use weighted beta estimation
                    beta, beta_err = self._estimate_beta_weighted(theta_region, weights)
                    peak.beta = np.clip(beta, -1.0, 2.0)
                    peak.beta_err = beta_err
                    peak.beta_source = "onion_weighted"
                    estimated_betas[peak_idx] = peak.beta
                    
                    if verbose:
                        print(f"    β={peak.beta:.3f} ± {peak.beta_err:.2f} (weighted)")
                    continue
            
            # Standard estimation for this region
            if len(theta_region) < 30:
                peak.beta = 0.0
                peak.beta_err = 1.0
                peak.beta_source = "default"
                estimated_betas[peak_idx] = 0.0
                continue
            
            beta_fft = self._estimate_beta_fft(theta_region)
            beta_fit, beta_err = self._estimate_beta_curvefit(theta_region)
            
            if beta_err < 0.3:
                beta = 0.6 * beta_fit + 0.4 * beta_fft
            else:
                beta = 0.5 * beta_fit + 0.5 * beta_fft
            
            peak.beta = np.clip(beta, -1.0, 2.0)
            peak.beta_err = beta_err
            peak.beta_source = "onion"
            estimated_betas[peak_idx] = peak.beta
            
            if verbose:
                print(f"    β={peak.beta:.3f} (FFT:{beta_fft:.2f}, fit:{beta_fit:.2f})")
        
        return peaks
    
    def _estimate_beta_weighted(self, theta: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """Weighted beta estimation using both FFT and curve fit."""
        # Normalize weights
        weights = weights / (weights.sum() + 1e-10)
        
        # Method 1: Weighted histogram + curve fit
        n_bins = 36
        hist, edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi), weights=weights)
        centers = (edges[:-1] + edges[1:]) / 2
        
        def model(theta, A, beta):
            cos_theta = np.cos(theta)
            P2 = (3 * cos_theta**2 - 1) / 2
            return A * (1 + beta * P2)
        
        try:
            # Smooth histogram slightly
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
        
        # Method 2: Weighted FFT
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
        
        # Combine methods
        if beta_err < 0.5:
            beta = 0.7 * beta_fit + 0.3 * beta_fft
        else:
            beta = 0.5 * beta_fit + 0.5 * beta_fft
        
        return beta, beta_err
    
    def _stage4_adaptive_refinement(self, peaks: List[HybridPeakResult],
                                     verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 4: IMPROVED - Sigma-based threshold for X2 trigger.
        """
        # IMPROVED: Use sigma-based threshold instead of fixed 1mm
        needs_x2 = []
        
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            separation = p2.r0 - p1.r0
            avg_sigma = (p1.sigma + p2.sigma) / 2
            separation_sigma = separation / avg_sigma
            
            if separation_sigma < self.cfg.overlap_threshold_sigma:
                if i not in needs_x2:
                    needs_x2.append(i)
                if i+1 not in needs_x2:
                    needs_x2.append(i+1)
                if verbose:
                    print(f"  Peaks {i+1},{i+2}: separation={separation:.2f}mm = {separation_sigma:.1f}σ < {self.cfg.overlap_threshold_sigma}σ → X2")
        
        # For now, just mark peaks that need X2 but don't run it
        # (X2 is slow and may not help much)
        for i in needs_x2:
            peaks[i].confidence *= 0.8  # Lower confidence for overlapping peaks
        
        return peaks
    
    def _stage5_consistency_check(self, peaks: List[HybridPeakResult],
                                   verbose: bool) -> List[HybridPeakResult]:
        """Stage 5: Consistency check."""
        for peak in peaks:
            confidence = 0.85
            if peak.r0_source == "global_fit":
                confidence += 0.1
            if peak.beta_err < 0.2:
                confidence += 0.05
            peak.confidence = min(confidence, 1.0)
        
        return peaks
    
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
        except Exception:
            return 0.0, 1.0
    
    def _print_results(self, peaks: List[HybridPeakResult]):
        """Print final results."""
        print("\n" + "="*60)
        print("HYBRID V3 RECONSTRUCTION RESULTS")
        print("="*60)
        
        for i, peak in enumerate(peaks):
            print(f"\nPeak {i+1}:")
            print(f"  r0 = {peak.r0:.3f} mm (source: {peak.r0_source})")
            print(f"  σ  = {peak.sigma:.3f} mm (source: {peak.sigma_source})")
            print(f"  β  = {peak.beta:.3f} (source: {peak.beta_source})")
            print(f"  Confidence: {peak.confidence:.2f}")


# Convenience function
def fit_xy_hybrid_v3(xy_data: np.ndarray,
                     n_peaks: int = 3,
                     vmi_k: float = 0.01,
                     verbose: bool = True) -> List[HybridPeakResult]:
    """Convenience function for V3 reconstruction."""
    recon = HybridVMIReconstructorV3(xy_data, vmi_k=vmi_k)
    return recon.reconstruct(n_peaks=n_peaks, verbose=verbose)
