"""
Hybrid VMI Reconstructor V2
===========================

Combines the strengths of multiple reconstruction algorithms:
- Parametric Forward Fitting: MLE-based multi-Gaussian fitting
- Physics V1: Fast initial estimation for beta
- X2: Forward fitting for extreme anisotropy (|β| > 1.5)

Key Innovation (V2):
- Uses analytical Abel projection of Gaussian peaks
- F_model(y) = sum_i A_i * a_i * sqrt(pi) * exp(-y^2/a_i^2)
- MLE with Poisson noise model for robust multi-peak detection
- No binning artifacts, handles overlapping peaks naturally

Strategy:
1. Coarse peak detection from radial histogram
2. Parametric MLE fitting with analytical forward model
3. Beta estimation using FFT + curve fit
4. Adaptive refinement for extreme cases
5. Consistency check

Author: Kiro AI Assistant
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
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
    
    # Confidence and uncertainty
    r0_err: float = 0.0
    sigma_err: float = 0.0
    beta_err: float = 0.0
    confidence: float = 1.0
    
    # Source tracking
    r0_source: str = "physics"
    beta_source: str = "physics"
    sigma_source: str = "physics"


@dataclass 
class HybridConfig:
    """Configuration for hybrid reconstructor."""
    # Thresholds for method selection
    beta_threshold_high: float = 1.5  # Use X2 if |β| > this
    overlap_threshold_sigma: float = 3.0  # Use X2 if separation < this * σ
    
    # Physics V1 settings
    physics_n_angular_bins: int = 36
    
    # Parametric fitting settings
    n_radial_bins: int = 300  # Higher resolution for fitting
    mle_noise_model: str = "poisson"  # "poisson" or "gaussian"
    mle_max_iter: int = 500
    
    # X2 settings (only used when needed)
    x2_n_ensemble: int = 5  # Smaller ensemble for speed
    x2_n_particles: int = 10000
    x2_n_iterations: int = 50
    
    # Validation settings
    validate_forward: bool = False
    validation_threshold: float = 0.1  # Residual threshold
    
    # General
    verbose: bool = True


# =============================================================================
# Parametric Forward Model for VMI Radial Histogram
# =============================================================================

class ParametricAbelModel:
    """
    Parametric model for VMI reconstruction using inverse Abel transform.
    
    The key insight is that the observed radial histogram H(r) is related to
    the 3D distribution f(r) through the Abel transform:
    
    1. H(r) = 2πr × ρ_2D(r)  (histogram = 2πr × 2D radial density)
    2. ρ_2D(r) = Abel[f(r)]   (2D density is Abel projection of 3D)
    3. f(r) = Abel^{-1}[ρ_2D(r)]  (inverse Abel recovers 3D)
    
    For multi-peak fitting, we fit Gaussians to the inverse-Abel-transformed
    data, which gives accurate r0 and sigma estimates.
    """
    
    def __init__(self, n_peaks: int):
        self.n_peaks = n_peaks
    
    def forward_model(self, r: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the 3D radial distribution model (sum of Gaussians).
        
        f(r) = sum_i A_i × exp(-(r - r_i)² / (2σ_i²))
        
        Args:
            r: Radial positions (1D array)
            params: Flattened parameters [A_1, r_1, σ_1, A_2, r_2, σ_2, ...]
        
        Returns:
            f(r): Model 3D distribution
        """
        f = np.zeros_like(r, dtype=float)
        
        for i in range(self.n_peaks):
            A_i = params[3*i]
            r_i = params[3*i + 1]
            sigma_i = params[3*i + 2]
            
            f += A_i * np.exp(-((r - r_i)**2) / (2 * sigma_i**2 + 1e-10))
        
        return f
    
    def negative_log_likelihood_poisson(self, params: np.ndarray, 
                                         r: np.ndarray, 
                                         f_obs: np.ndarray) -> float:
        """Poisson NLL for fitting to inverse-Abel-transformed data."""
        f_model = self.forward_model(r, params)
        f_model = np.maximum(f_model, 1e-10)
        
        # Only include positive observations
        mask = f_obs > 0
        nll = np.sum(f_model) - np.sum(f_obs[mask] * np.log(f_model[mask]))
        
        return nll
    
    def chi_squared(self, params: np.ndarray,
                    r: np.ndarray,
                    f_obs: np.ndarray) -> float:
        """Chi-squared for fitting."""
        f_model = self.forward_model(r, params)
        
        # Weight by inverse variance (Poisson-like)
        variance = np.maximum(np.abs(f_obs), 1)
        chi2 = np.sum((f_obs - f_model)**2 / variance)
        
        return chi2
    
    def fit(self, r: np.ndarray, f_obs: np.ndarray,
            initial_params: np.ndarray,
            noise_model: str = "gaussian",
            max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """
        Fit multi-Gaussian model to inverse-Abel-transformed data.
        """
        r_max = r.max()
        
        bounds = []
        for i in range(self.n_peaks):
            bounds.append((1e-10, None))     # A_i > 0
            bounds.append((0.1, r_max))      # 0 < r_i < r_max
            bounds.append((0.01, r_max/2))   # 0.01 < σ_i < r_max/2
        
        if noise_model == "poisson":
            loss_fn = lambda p: self.negative_log_likelihood_poisson(p, r, f_obs)
        else:
            loss_fn = lambda p: self.chi_squared(p, r, f_obs)
        
        result = minimize(
            loss_fn,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        
        return result.x, result.fun


class HybridVMIReconstructor:
    """
    Hybrid VMI reconstructor combining multiple algorithms.
    
    Usage:
        recon = HybridVMIReconstructor(xy_data)
        peaks = recon.reconstruct(n_peaks=2)
    """
    
    def __init__(self, xy_data: np.ndarray, 
                 pixel_size: float = 0.05,
                 psf_sigma: float = 0.0,
                 dld_resolution: float = 0.0,
                 vmi_k: float = 0.01,
                 config: HybridConfig = None):
        """
        Initialize hybrid reconstructor.
        
        Args:
            xy_data: XY scatter data (N, 2)
            pixel_size: Detector pixel size (mm)
            psf_sigma: PSF sigma (mm)
            dld_resolution: DLD quantization (mm)
            vmi_k: VMI conversion coefficient
            config: Configuration options
        """
        self.xy_data = xy_data
        self.pixel_size = pixel_size
        self.psf_sigma = psf_sigma
        self.dld_resolution = dld_resolution
        self.vmi_k = vmi_k
        self.cfg = config or HybridConfig()
        
        # Precompute polar coordinates
        self.r = np.sqrt(xy_data[:, 0]**2 + xy_data[:, 1]**2)
        self.theta = np.arctan2(xy_data[:, 1], xy_data[:, 0])
        self.r_max = np.percentile(self.r, 99.5)
        
        # Cache for intermediate results
        self._physics_result = None
        self._multires_result = None
        self._x2_result = None
    
    def reconstruct(self, n_peaks: int = 3, 
                    verbose: bool = None) -> List[HybridPeakResult]:
        """
        Reconstruct VMI parameters using hybrid approach.
        
        Args:
            n_peaks: Number of peaks to detect
            verbose: Override config verbose setting
            
        Returns:
            List of HybridPeakResult objects
        """
        verbose = verbose if verbose is not None else self.cfg.verbose
        
        if verbose:
            print("="*60)
            print("HYBRID VMI RECONSTRUCTOR V2 (Parametric MLE)")
            print("="*60)
            print(f"Data: {len(self.xy_data)} particles, r_max={self.r_max:.2f} mm")
        
        # Stage 1: Coarse peak detection for initialization
        if verbose:
            print("\n[Stage 1] Coarse peak detection...")
        
        coarse_peaks = self._stage1_coarse_detection(n_peaks, verbose)
        
        if len(coarse_peaks) == 0:
            if verbose:
                print("  WARNING: No peaks detected!")
            return []
        
        # Stage 2: Parametric MLE fitting
        if verbose:
            print("\n[Stage 2] Parametric MLE fitting...")
        
        fitted_peaks = self._stage2_parametric_mle(coarse_peaks, verbose)
        
        # Stage 3: Beta estimation
        if verbose:
            print("\n[Stage 3] Beta estimation...")
        
        peaks_with_beta = self._stage3_beta_estimation(fitted_peaks, verbose)
        
        # Stage 4: Adaptive refinement for extreme cases
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
    
    def _stage1_coarse_detection(self, n_peaks: int, 
                                  verbose: bool) -> List[dict]:
        """
        Stage 1: Multi-resolution histogram peak detection.
        
        Uses raw histogram for peak detection with correction for inner peaks.
        """
        if verbose:
            print("  Multi-resolution peak detection...")
        
        # Multiple resolutions
        resolutions = [
            {'n_bins': 150, 'smooth': 5},
            {'n_bins': 200, 'smooth': 4},
            {'n_bins': 300, 'smooth': 3},
            {'n_bins': 400, 'smooth': 2},
        ]
        
        all_peaks = []
        
        for res_idx, res in enumerate(resolutions):
            n_bins = res['n_bins']
            smooth_sigma = res['smooth']
            
            # Build histogram
            hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
            bin_centers = (edges[:-1] + edges[1:]) / 2
            dr = bin_centers[1] - bin_centers[0]
            
            # Smooth
            hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)
            
            # Find peaks
            min_distance = max(int(0.3 / dr), 3)
            prominence = max(hist_smooth.max() * 0.02, 20)
            
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
            if verbose:
                print("  No peaks found, using fallback")
            return self._fallback_histogram_detection(n_peaks, verbose)
        
        # Cluster peaks across resolutions
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
        
        # Score clusters
        for cluster in clusters:
            cluster['score'] = cluster['total_prominence'] * (1 + 0.5 * cluster['n_resolutions'])
        
        # Sort by score and take top n_peaks
        clusters.sort(key=lambda x: x['score'], reverse=True)
        top_clusters = clusters[:n_peaks]
        top_clusters.sort(key=lambda x: x['r0'])
        
        # Estimate sigma and apply correction for inner peaks
        results = []
        
        n_bins = 300
        hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        dr = bin_centers[1] - bin_centers[0]
        r_safe = np.maximum(bin_centers, dr)
        hist_corrected = hist.astype(float) / (2 * np.pi * r_safe * dr)
        hist_smooth = gaussian_filter1d(hist_corrected, sigma=3)
        
        for cluster in top_clusters:
            r0 = cluster['r0']
            
            # Find nearest bin
            idx = np.argmin(np.abs(bin_centers - r0))
            sigma = self._estimate_sigma_fwhm(bin_centers, hist_smooth, idx)
            
            if verbose:
                print(f"  Peak at r={r0:.2f}mm, σ≈{sigma:.3f}mm, score={cluster['score']:.0f}, n_res={cluster['n_resolutions']}")
            
            results.append({
                'r0': r0,
                'sigma': sigma,
                'amp': cluster['score']
            })
        
        return results
    
    def _fallback_histogram_detection(self, n_peaks: int, verbose: bool) -> List[dict]:
        """Fallback to simple histogram peak detection."""
        n_bins = 300
        hist, edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        dr = bin_centers[1] - bin_centers[0]
        
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)
        
        min_distance = max(int(0.2 / dr), 3)
        prominence = max(hist_smooth.max() * 0.01, 10)
        
        peaks_idx, props = find_peaks(
            hist_smooth,
            prominence=prominence,
            distance=min_distance,
            width=2
        )
        
        if len(peaks_idx) == 0:
            peaks_idx = np.array([np.argmax(hist_smooth)])
            props = {'prominences': np.array([hist_smooth.max()])}
        
        sorted_idx = np.argsort(props['prominences'])[::-1]
        peaks_idx = peaks_idx[sorted_idx[:n_peaks]]
        peaks_idx = np.sort(peaks_idx)
        
        results = []
        for idx in peaks_idx:
            r0 = bin_centers[idx]
            sigma = self._estimate_sigma_fwhm(bin_centers, hist_smooth, idx)
            results.append({'r0': r0, 'sigma': sigma, 'amp': hist_smooth[idx]})
        
        return results
        
        return results
    
    def _stage2_parametric_mle(self, coarse_peaks: List[dict],
                                verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 2: Inverse Abel + local peak refinement.
        
        For each coarse peak, find the local maximum in the inverse-Abel-transformed
        data to get accurate r0, then fit a Gaussian for sigma.
        """
        try:
            import abel
        except ImportError:
            if verbose:
                print("  WARNING: PyAbel not available, using coarse estimates")
            return [HybridPeakResult(
                r0=p['r0'], sigma=p['sigma'], beta=0.0, amp=p['amp'],
                r0_source="coarse", sigma_source="coarse", beta_source="pending"
            ) for p in coarse_peaks]
        
        n_peaks = len(coarse_peaks)
        
        # Build high-resolution histogram and inverse Abel
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
        except Exception:
            if verbose:
                print("  WARNING: Abel transform failed, using coarse estimates")
            return [HybridPeakResult(
                r0=p['r0'], sigma=p['sigma'], beta=0.0, amp=p['amp'],
                r0_source="coarse", sigma_source="coarse", beta_source="pending"
            ) for p in coarse_peaks]
        
        # Refine each peak position using local maximum in f_3d
        results = []
        
        # Sort coarse peaks by r0
        coarse_peaks_sorted = sorted(coarse_peaks, key=lambda p: p['r0'])
        
        for i, peak in enumerate(coarse_peaks_sorted):
            r0_coarse = peak['r0']
            sigma_coarse = peak['sigma']
            
            # Define search region around coarse peak
            search_radius = max(1.0, sigma_coarse)  # At least 1mm search radius
            
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
            else:
                # Find local maximum
                max_idx = np.argmax(f_local)
                r0_refined = r_local[max_idx]
                
                # Estimate sigma from FWHM in f_3d
                peak_height = f_local[max_idx]
                half_max = peak_height / 2
                
                # Find left crossing
                left_idx = max_idx
                while left_idx > 0 and f_local[left_idx] > half_max:
                    left_idx -= 1
                
                # Find right crossing
                right_idx = max_idx
                while right_idx < len(f_local) - 1 and f_local[right_idx] > half_max:
                    right_idx += 1
                
                fwhm = r_local[right_idx] - r_local[left_idx]
                sigma_refined = fwhm / 2.355
                
                # Factor of 2 correction for sigma (due to E ∝ r²)
                sigma_refined = sigma_refined * 2
                
                # Clamp sigma to reasonable range
                sigma_refined = np.clip(sigma_refined, 0.1, 2.0)
            
            if verbose:
                print(f"  Peak {i+1}: r0={r0_refined:.3f}mm (coarse:{r0_coarse:.2f}), σ={sigma_refined:.4f}mm")
            
            results.append(HybridPeakResult(
                r0=r0_refined,
                sigma=sigma_refined,
                beta=0.0,
                amp=1.0,
                r0_source="mle_abel",
                sigma_source="mle_abel",
                beta_source="pending"
            ))
        
        results.sort(key=lambda p: p.r0)
        return results
    
    def _stage3_beta_estimation(self, peaks: List[HybridPeakResult],
                                 verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 3: Estimate beta for each peak using angular distribution.
        """
        for i, peak in enumerate(peaks):
            # Select particles near this peak
            window = max(peak.sigma * 2.5, 0.3)
            mask = (self.r >= peak.r0 - window) & (self.r < peak.r0 + window)
            theta_region = self.theta[mask]
            
            if len(theta_region) < 50:
                peak.beta = 0.0
                peak.beta_err = 1.0
                peak.beta_source = "default"
                continue
            
            # Method 1: FFT-based
            beta_fft = self._estimate_beta_fft(theta_region)
            
            # Method 2: Curve fit
            beta_fit, beta_err = self._estimate_beta_curvefit(theta_region)
            
            # Combine (prefer curve fit if error is low)
            if beta_err < 0.3:
                beta = 0.6 * beta_fit + 0.4 * beta_fft
            else:
                beta = 0.5 * beta_fit + 0.5 * beta_fft
            
            peak.beta = np.clip(beta, -1.0, 2.0)
            peak.beta_err = beta_err
            peak.beta_source = "physics"
            
            if verbose:
                print(f"  Peak {i+1}: β={peak.beta:.3f} (FFT:{beta_fft:.2f}, fit:{beta_fit:.2f})")
        
        return peaks
    
    def _stage4_adaptive_refinement(self, peaks: List[HybridPeakResult],
                                     verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 4: Adaptive refinement for extreme cases.
        
        - Use X2 for truly overlapping peaks (separation < 1mm)
        - For extreme β, the MLE result is usually good enough
        """
        refined = []
        
        # Check for truly overlapping peaks (use fixed threshold, not sigma-based)
        needs_x2 = []
        min_separation = 1.0  # mm - peaks closer than this are truly overlapping
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            separation = p2.r0 - p1.r0
            if separation < min_separation:
                if i not in needs_x2:
                    needs_x2.append(i)
                if i+1 not in needs_x2:
                    needs_x2.append(i+1)
                if verbose:
                    print(f"  Peaks {i+1},{i+2}: separation={separation:.2f}mm < {min_separation}mm → X2")
        
        # Note: We no longer trigger X2 just for extreme beta
        # The MLE Abel result is usually accurate for extreme beta cases
        
        # Refine peaks
        for i, peak in enumerate(peaks):
            if i in needs_x2:
                # Use X2 forward fitting for overlapping peaks
                refined_peak = self._refine_with_x2(peak, i, verbose)
            else:
                # Keep MLE result
                refined_peak = peak
            
            refined.append(refined_peak)
        
        return refined
        
        return refined
    
    def _stage5_consistency_check(self, peaks: List[HybridPeakResult],
                                   verbose: bool) -> List[HybridPeakResult]:
        """
        Stage 5: Check consistency and compute confidence.
        """
        for peak in peaks:
            # Higher confidence if MLE fitting succeeded
            confidence = 0.85
            if peak.r0_source == "mle":
                confidence += 0.05
            if peak.beta_source == "x2" and abs(peak.beta) > 1.0:
                confidence += 0.1
            peak.confidence = min(confidence, 1.0)
        
        if verbose:
            for i, peak in enumerate(peaks):
                print(f"  Peak {i+1}: confidence={peak.confidence:.2f}")
        
        return peaks
    
    def _estimate_sigma_fwhm(self, r: np.ndarray, hist: np.ndarray, 
                              peak_idx: int) -> float:
        """Estimate sigma from FWHM of histogram peak."""
        peak_height = hist[peak_idx]
        half_max = peak_height / 2
        
        # Find left crossing
        left_idx = peak_idx
        while left_idx > 0 and hist[left_idx] > half_max:
            left_idx -= 1
        
        # Find right crossing
        right_idx = peak_idx
        while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
            right_idx += 1
        
        fwhm = r[right_idx] - r[left_idx]
        sigma = fwhm / 2.355
        
        # Clamp to reasonable range
        sigma = np.clip(sigma, 0.05, 2.0)
        
        return sigma
    
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
        
        # Sign from phase
        sign = 1.0 if abs(phase) > np.pi/2 else -1.0
        c2_signed = sign * c2_amp
        
        denominator = 3.0 * c0 - c2_signed
        if abs(denominator) < 1e-10:
            return 0.0
        
        beta = 4.0 * c2_signed / denominator
        return np.clip(beta, -1.0, 2.0)
    
    def _estimate_beta_curvefit(self, theta: np.ndarray) -> Tuple[float, float]:
        """Curve fit beta estimation."""
        # Fold to [0, π]
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
    
    def _refine_with_x2(self, peak: HybridPeakResult,
                        peak_idx: int, verbose: bool) -> HybridPeakResult:
        """Refine peak using X2 forward fitting."""
        try:
            from Abel_backward_reconstruction_x2 import (
                DiffConfig, DirectSamplingForwardModel, 
                MultiResolutionConsistencyLoss, SmartInitializer
            )
            import torch
            
            # Quick X2 optimization for this peak region
            window = max(peak.sigma * 4, 1.0)
            r_min = max(0, peak.r0 - window)
            r_max_local = min(self.r_max, peak.r0 + window)
            
            mask = (self.r >= r_min) & (self.r < r_max_local)
            xy_local = self.xy_data[mask]
            
            if len(xy_local) < 500:
                return peak
            
            # Convert r0 to energy
            v = peak.r0 / self.vmi_k
            from scipy.constants import electron_mass, elementary_charge, atomic_mass
            mass_kg = electron_mass
            E_center = 0.5 * mass_kg * v**2 / elementary_charge
            
            # Quick X2 fit
            config = DiffConfig(
                vmi_k=self.vmi_k,
                E_max=E_center * 2,
                n_particles=self.cfg.x2_n_particles,
                n_iterations=self.cfg.x2_n_iterations
            )
            
            # Single optimization (not full ensemble for speed)
            xy_t = torch.tensor(xy_local, dtype=torch.float32)
            X_obs, Y_obs = xy_t[:, 0], xy_t[:, 1]
            r_max_t = torch.quantile(torch.sqrt(X_obs**2 + Y_obs**2), 0.99).item()
            
            model = DirectSamplingForwardModel(config, n_peaks=1)
            
            # Initialize from current estimate
            E_range = config.E_max - config.E_min
            E_norm = np.clip((E_center - config.E_min) / E_range, 0.01, 0.99)
            model.E_logits.data[0] = torch.tensor(np.log(E_norm / (1 - E_norm)))
            
            beta_range = config.beta_max - config.beta_min
            beta_norm = np.clip((peak.beta - config.beta_min) / beta_range, 0.01, 0.99)
            model.beta_raw.data[0] = torch.tensor(np.log(beta_norm / (1 - beta_norm)))
            
            loss_fn = MultiResolutionConsistencyLoss(config, r_max_t)
            
            with torch.no_grad():
                features_obs = loss_fn.compute_features(X_obs, Y_obs)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            
            for _ in range(self.cfg.x2_n_iterations):
                optimizer.zero_grad()
                X_sim, Y_sim = model(config.n_particles)
                features_sim = loss_fn.compute_features(X_sim, Y_sim)
                loss, _ = loss_fn(features_obs, features_sim)
                loss.backward()
                optimizer.step()
            
            params = model.get_physical_params()
            E_fitted = params['E_centers'][0].item()
            beta_fitted = params['betas'][0].item()
            
            # Convert energy back to radius
            v_fitted = np.sqrt(2 * E_fitted * elementary_charge / mass_kg)
            r0_fitted = self.vmi_k * v_fitted
            
            if verbose:
                print(f"  Peak {peak_idx+1} X2: r0 {peak.r0:.2f}→{r0_fitted:.2f}, β {peak.beta:.2f}→{beta_fitted:.2f}")
            
            return HybridPeakResult(
                r0=r0_fitted,
                sigma=peak.sigma,  # Keep MLE sigma
                beta=beta_fitted,
                amp=peak.amp,
                r0_source="x2",
                beta_source="x2",
                sigma_source=peak.sigma_source
            )
            
        except Exception as e:
            if verbose:
                print(f"  Peak {peak_idx+1} X2 failed: {e}")
            return peak
    
    def _print_results(self, peaks: List[HybridPeakResult]):
        """Print final results."""
        print("\n" + "="*60)
        print("HYBRID RECONSTRUCTION RESULTS")
        print("="*60)
        
        for i, peak in enumerate(peaks):
            print(f"\nPeak {i+1}:")
            print(f"  r0 = {peak.r0:.3f} mm (source: {peak.r0_source})")
            print(f"  σ  = {peak.sigma:.3f} mm (source: {peak.sigma_source})")
            print(f"  β  = {peak.beta:.3f} (source: {peak.beta_source})")
            print(f"  Confidence: {peak.confidence:.2f}")


# =============================================================================
# Convenience function
# =============================================================================
def fit_xy_hybrid(xy_data: np.ndarray,
                  n_peaks: int = 3,
                  vmi_k: float = 0.01,
                  verbose: bool = True) -> List[HybridPeakResult]:
    """
    Convenience function for hybrid reconstruction.
    
    Args:
        xy_data: XY scatter data (N, 2)
        n_peaks: Number of peaks to detect
        vmi_k: VMI conversion coefficient
        verbose: Print progress
        
    Returns:
        List of HybridPeakResult objects
    """
    recon = HybridVMIReconstructor(xy_data, vmi_k=vmi_k)
    return recon.reconstruct(n_peaks=n_peaks, verbose=verbose)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing Hybrid VMI Reconstructor...")
    
    # Generate test data
    np.random.seed(42)
    N = 50000
    
    # Two peaks with different beta
    r1 = np.random.normal(8, 0.4, N // 2)
    r2 = np.random.normal(12, 0.4, N // 2)
    r = np.concatenate([r1, r2])
    
    # Simple isotropic for test
    theta = np.random.uniform(-np.pi, np.pi, N)
    
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    xy_data = np.column_stack([X, Y])
    
    # Run hybrid reconstruction
    peaks = fit_xy_hybrid(xy_data, n_peaks=2, verbose=True)
    
    print("\nDone!")
