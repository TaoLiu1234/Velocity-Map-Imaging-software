"""
Abel Backward Reconstruction for Photoelectron Spectroscopy (VMI)

This module implements physics-based reconstruction of VMI images to extract:
- Peak positions (radii) -> corresponding to electron energies
- Peak widths (sigma/FWHM) -> energy resolution
- Anisotropy parameters (beta) -> angular distribution
- Relative amplitudes -> branching ratios

The reconstruction uses:
1. Adaptive spectral filtering for noise reduction
2. 1D Abel inversion for radial profile extraction
3. FWHM-based sigma estimation
4. Angular Fourier analysis for beta extraction
5. Forward model optimization with CDF shape constraints

Author: Adapted from test.py for VMI analysis
"""

import numpy as np
import abel
import time
from scipy import signal
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import map_coordinates
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


class PhysicsBasedFitter:
    """
    Physics-based fitter for VMI image reconstruction.
    
    This class implements a multi-phase reconstruction algorithm:
    - Phase 1: Radial analysis with adaptive filtering and sigma estimation
    - Phase 2: Angular analysis for beta parameter extraction
    - Phase 3: Forward model optimization with robust regression
    
    Attributes:
        n: Image size in pixels
        radius: Maximum radius (half of image size)
        r_grid_1d: 1D radial grid
        
    Shared Data (computed once, reused across phases):
        _shared_polar_image: Polar coordinate image (r, θ)
        _shared_theta_grid: Theta grid for polar image
        _shared_noise: Noise parameters (gaussian_mean, gaussian_std, total_noise_std, snr)
        _shared_radial_profile: Radial profile of input image
    """
    
    def __init__(self, n_pixels: int):
        """
        Initialize the fitter with image dimensions.
        
        Args:
            n_pixels: Size of the square image in pixels
        """
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid_1d = np.arange(self.radius + 1, dtype=float)
        
        # Create coordinate grids
        y, x = np.ogrid[:n_pixels, :n_pixels]
        self.Y = y - n_pixels // 2
        self.X = x - n_pixels // 2
        self.R2 = self.X**2 + self.Y**2
        self.R = np.sqrt(self.R2)
        
        # Radial indexing for binning
        self.r_indices = self.R.astype(int)
        self.max_r_idx = int(np.max(self.R))
        self.r_flat = self.r_indices.ravel()
        
        # Pixel counts per radius (for averaging)
        self.pixel_counts = np.bincount(self.r_flat, minlength=self.max_r_idx + 1)
        self.pixel_counts[self.pixel_counts == 0] = 1 
        
        # Angular grids for P2 calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            self.COS_THETA = self.X / self.R
        self.COS_THETA[~np.isfinite(self.COS_THETA)] = 0.0
        self.P2_GRID = 0.5 * (3 * self.COS_THETA**2 - 1)
        
        # Shared data cache (computed once per solve() call)
        self._shared_polar_image = None
        self._shared_theta_grid = None
        self._shared_noise = None
        self._shared_radial_profile = None

    def _compute_full_radial_profile(self, img_2d: np.ndarray) -> np.ndarray:
        """
        Compute complete radial distribution including corner regions.
        
        Used for noise estimation from image corners where signal is minimal.
        
        Args:
            img_2d: Input 2D image
            
        Returns:
            Full radial profile array
        """
        radial_sum = np.bincount(self.r_flat, weights=img_2d.ravel(), minlength=self.max_r_idx + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = radial_sum / self.pixel_counts
        profile[~np.isfinite(profile)] = 0
        return profile

    def _compute_radial_profile(self, img_2d: np.ndarray) -> np.ndarray:
        """
        Compute radial profile within valid radius.
        
        Args:
            img_2d: Input 2D image
            
        Returns:
            Radial profile up to self.radius
        """
        full_prof = self._compute_full_radial_profile(img_2d)
        return full_prof[:len(self.r_grid_1d)]

    def _init_shared_data(self, image_2d: np.ndarray, n_theta: int = 720) -> None:
        """
        初始化共享数据：极坐标图像、噪声参数、径向分布。
        
        这些数据在 Phase 1, 2, 3 中都会用到，只计算一次。
        
        Args:
            image_2d: 输入2D图像
            n_theta: 角度分辨率（默认720 = 0.5°步长）
        """
        cy, cx = self.n // 2, self.n // 2
        n_r = self.radius
        
        # 1. 创建极坐标图像（高分辨率，用于Phase 2）
        theta_grid = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        r_grid = np.arange(n_r)
        theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
        x_cart = cx + r_mesh * np.cos(theta_mesh)
        y_cart = cy + r_mesh * np.sin(theta_mesh)
        
        # 三次插值（更准确）
        self._shared_polar_image = map_coordinates(image_2d, [y_cart, x_cart], order=3, mode='constant', cval=0.0)
        self._shared_theta_grid = theta_grid
        
        # 2. 计算径向分布（先计算，用于找最大peak位置）
        full_profile = self._compute_full_radial_profile(image_2d)
        valid_len = len(self.r_grid_1d)
        signal_region = full_profile[:valid_len]
        self._shared_radial_profile = signal_region
        
        # 3. 改进的噪声估计：从最大peak位置到r_max的整个区域
        # 找到最外层有信号的位置（最大peak之后的区域都是噪声）
        from scipy.ndimage import gaussian_filter1d
        smooth_profile = gaussian_filter1d(signal_region, sigma=3)
        
        # 找所有peaks，取最外层的peak位置
        peak_threshold = np.max(smooth_profile) * 0.05  # 5%阈值
        above_threshold = smooth_profile > peak_threshold
        if np.any(above_threshold):
            # 最外层有信号的位置
            outermost_signal_r = np.max(np.where(above_threshold)[0])
            # 噪声区域从 outermost_signal_r + margin 开始
            noise_margin = 20  # 留一些margin避免peak尾巴
            noise_start_r = min(outermost_signal_r + noise_margin, int(n_r * 0.85))
        else:
            # 如果没找到peak，用默认的边缘区域
            noise_start_r = int(n_r * 0.85)
        
        # 确保噪声区域足够大（至少15%的半径范围）
        min_noise_region = int(n_r * 0.15)
        noise_start_r = min(noise_start_r, n_r - min_noise_region)
        
        # 从极坐标图像提取噪声区域
        noise_pixels = self._shared_polar_image[noise_start_r:, :].ravel()
        
        # Robust统计
        noise_median = np.median(noise_pixels)
        noise_mad = np.median(np.abs(noise_pixels - noise_median))
        noise_std_robust = 1.4826 * noise_mad
        
        # 4. 频域背景估计（改进版）
        # 背景通常是低频成分（DC + 缓慢变化），信号peaks是中高频
        # 使用频域分离可以更干净地去除背景
        baseline = self._estimate_baseline_frequency_domain(signal_region, noise_median)
        signal_region_corrected = np.maximum(signal_region - baseline, 0)
        self._shared_radial_profile = signal_region_corrected
        
        # 同时更新极坐标图像（减去baseline）
        self._shared_polar_image = np.maximum(self._shared_polar_image - baseline, 0)
        
        # 5. 噪声模型（baseline已减去，主要是泊松噪声 + 残余读出噪声）
        total_noise_std = self._estimate_noise_after_baseline_removal(signal_region_corrected, noise_std_robust)
        
        # 6. 整体SNR估计
        snr_estimate = np.mean(signal_region_corrected) / (np.mean(total_noise_std) + 1e-6)
        
        # 存储噪声参数
        self._shared_noise = {
            'readout_std': noise_std_robust,       # 读出噪声标准差（方差仍存在）
            'baseline_removed': True,              # 标记已减去baseline
            'total_noise_std': total_noise_std,    # 每个半径的总噪声（泊松+读出）
            'snr': snr_estimate,
            'estimated_baseline': baseline         # 保存估计的baseline供调试
        }
    
    def _estimate_baseline_frequency_domain(self, radial_profile: np.ndarray, 
                                            noise_median: float) -> float:
        """
        使用频域方法估计背景baseline。
        
        思路：
        - 背景是低频成分（DC + 缓慢变化）
        - 信号peaks是中高频成分
        - 通过分析频谱来分离背景和信号
        
        Args:
            radial_profile: 原始径向分布
            noise_median: 从空域估计的噪声中值（作为参考）
            
        Returns:
            估计的baseline值
        """
        n = len(radial_profile)
        
        # 对径向分布做FFT
        fft = np.fft.rfft(radial_profile)
        freqs = np.fft.rfftfreq(n)
        
        # DC分量（零频率）就是平均值
        dc_component = np.abs(fft[0]) / n
        
        # 分析低频成分（前几个频率）
        # 背景通常只贡献DC和最低的1-2个频率
        n_low_freq = min(3, len(fft) // 10)  # 最低的几个频率
        low_freq_power = np.sum(np.abs(fft[1:n_low_freq+1])**2)
        total_power = np.sum(np.abs(fft[1:])**2)  # 排除DC
        
        # 如果低频功率占比很高，说明有缓慢变化的背景
        low_freq_ratio = low_freq_power / (total_power + 1e-10)
        
        # 综合估计baseline
        # 1. 如果低频占比高，背景可能有空间变化，使用DC分量
        # 2. 否则使用空域估计的noise_median
        if low_freq_ratio > 0.3:
            # 背景有空间变化，使用最小值附近的估计
            # 找到径向分布的最小值区域（通常在边缘）
            edge_region = radial_profile[int(n*0.8):]
            baseline = np.percentile(edge_region, 25)  # 25th percentile更robust
        else:
            # 背景相对平坦，使用空域估计
            baseline = noise_median
        
        return baseline
    
    def _estimate_noise_after_baseline_removal(self, radial_profile: np.ndarray, 
                                                readout_noise_std: float) -> np.ndarray:
        """
        估计减去高斯背景后每个半径处的总噪声标准差。
        
        减去baseline后的噪声模型：
        σ²_total(r) = σ²_poisson(r) + σ²_readout
                    = I(r)/N_pixels(r) + σ²_readout/N_pixels(r)
        
        说明：
        - 泊松噪声（shot noise）：方差 = 信号强度，是主导噪声源
        - 读出噪声：减去均值后方差仍存在，但通常较小
        
        Args:
            radial_profile: 已减去baseline的径向强度分布
            readout_noise_std: 读出噪声的标准差（从边缘区域估计）
            
        Returns:
            每个半径处的总噪声标准差
        """
        pixel_counts = self.pixel_counts[:len(radial_profile)]
        pixel_counts = np.maximum(pixel_counts, 1)  # 避免除零
        
        # 泊松噪声方差（对于平均值）- 主导项
        poisson_variance = np.maximum(radial_profile, 0) / pixel_counts
        
        # 读出噪声方差（对于平均值）- 次要项
        readout_variance = (readout_noise_std ** 2) / pixel_counts
        
        # 总噪声方差
        total_variance = poisson_variance + readout_variance
        
        return np.sqrt(total_variance)
    
    def _denoise_radial_profile_bayesian(self, radial_profile: np.ndarray,
                                          noise_std: np.ndarray) -> np.ndarray:
        """
        使用贝叶斯方法对径向分布进行去噪。
        
        假设真实信号是平滑的（先验），观测值服从泊松噪声模型（baseline已减去）。
        使用Wiener滤波的变体，考虑空间变化的噪声。
        
        注意：输入的radial_profile应该已经减去了baseline。
        
        Args:
            radial_profile: 已减去baseline的径向分布
            noise_std: 每个半径处的噪声标准差
            
        Returns:
            去噪后的径向分布
        """
        from scipy.ndimage import gaussian_filter1d
        
        # 输入已减去baseline，直接使用
        profile_corrected = np.maximum(radial_profile, 0)
        
        n_points = len(profile_corrected)
        
        # 估计信号功率谱（使用平滑版本作为先验）
        # 先做一个初步平滑来估计信号
        smooth_prior = gaussian_filter1d(profile_corrected, sigma=3)
        
        # 频域分析
        fft_obs = np.fft.rfft(profile_corrected)
        fft_prior = np.fft.rfft(smooth_prior)
        
        # 估计信号功率谱
        signal_power = np.abs(fft_prior) ** 2
        
        # 估计噪声功率谱（使用平均噪声方差）
        mean_noise_var = np.mean(noise_std ** 2)
        noise_power = mean_noise_var * n_points  # 噪声功率在频域是均匀分布的
        
        # Wiener滤波器
        # H(f) = S(f) / (S(f) + N(f))
        wiener_filter = signal_power / (signal_power + noise_power + 1e-12)
        
        # 对低频保持更高的增益（信号主要在低频）
        freq = np.fft.rfftfreq(n_points)
        low_freq_boost = np.exp(-freq ** 2 / (2 * 0.1 ** 2))  # 低频增强
        wiener_filter = wiener_filter * 0.7 + low_freq_boost * 0.3
        wiener_filter = np.clip(wiener_filter, 0, 1)
        
        # 应用滤波器
        fft_filtered = fft_obs * wiener_filter
        profile_filtered = np.fft.irfft(fft_filtered, n=n_points)
        
        # 确保非负（baseline已在输入前减去）
        profile_filtered = np.maximum(profile_filtered, 0)
        
        return profile_filtered
    
    # =========================================================================
    # Phase 1: Radial Analysis - Peak Position and Sigma Estimation
    # =========================================================================
    
    def _phase1_find_peaks_in_projection(self, proj_profile: np.ndarray, 
                                          noise_std_array: np.ndarray,
                                          mask_radius: int = 10) -> List[Dict]:
        """
        Step 1.2: 在投影径向分布上找peaks。
        
        投影径向分布 = 原始图像沿theta积分后的结果。
        优点：SNR高，peak位置稳定。
        
        Args:
            proj_profile: 平滑后的投影径向分布（已减去背景）
            noise_std_array: 每个半径处的噪声标准差
            mask_radius: 中心mask半径
            
        Returns:
            List of projection peaks with (position, height, snr)
        """
        max_val = np.max(proj_profile)
        
        # 使用较低的阈值，因为投影的peak更明显
        height_threshold = max_val * 0.03
        prominence_threshold = max_val * 0.02
        
        peaks, _ = find_peaks(
            proj_profile,
            height=height_threshold,
            distance=5,
            prominence=prominence_threshold
        )
        
        # 计算每个peak的SNR
        peak_list = []
        for pk in peaks:
            if pk < mask_radius:
                continue
            height = proj_profile[pk]
            noise = noise_std_array[pk] if pk < len(noise_std_array) else noise_std_array[-1]
            snr = height / (noise + 1e-6)
            peak_list.append({
                'r_proj': int(pk),
                'height': float(height),
                'snr': float(snr)
            })
        
        return peak_list
    
    def _phase1_estimate_sigma_from_abel(self, abel_profile: np.ndarray,
                                          proj_peak_r: int,
                                          search_range: int = 8,
                                          mask_radius: int = 10) -> Tuple[int, float, float]:
        """
        Step 1.4: 在Abel逆变换结果中估计peak的sigma。
        
        使用多种方法估计sigma并取加权平均：
        1. FWHM方法（带基线校正和插值）
        2. 高斯拟合方法
        3. 二阶矩方法
        
        Args:
            abel_profile: Abel逆变换后的径向分布
            proj_peak_r: 投影中peak的位置
            search_range: 搜索范围
            mask_radius: 中心mask半径
            
        Returns:
            Tuple of (abel_peak_r, sigma, amplitude)
        """
        from scipy.optimize import curve_fit
        
        # 在投影peak附近搜索Abel逆变换中的局部最大值
        r_start = max(mask_radius, proj_peak_r - search_range)
        r_end = min(len(abel_profile), proj_peak_r + search_range + 1)
        
        local_region = abel_profile[r_start:r_end]
        if len(local_region) == 0:
            return proj_peak_r, 3.0, 0.0
        
        local_max_idx = np.argmax(local_region)
        abel_pk = r_start + local_max_idx
        abel_amp = abel_profile[abel_pk]
        
        # ===== 方法1: 改进的FWHM方法（带基线校正和线性插值）=====
        # 使用scipy的peak_widths来获得更准确的FWHM
        # 先尝试用peak_widths
        try:
            widths_result = peak_widths(abel_profile, [abel_pk], rel_height=0.5)
            fwhm_scipy = widths_result[0][0]
            if 1.0 < fwhm_scipy < 50:  # 合理范围检查
                sigma_scipy = fwhm_scipy / 2.355
            else:
                sigma_scipy = None
        except:
            sigma_scipy = None
        
        # 估计局部基线（取peak两侧较远处的最小值区域）
        baseline_range = 20
        left_baseline_start = max(mask_radius, abel_pk - baseline_range)
        left_baseline_end = max(mask_radius, abel_pk - baseline_range + 5)
        right_baseline_start = min(len(abel_profile) - 1, abel_pk + baseline_range - 5)
        right_baseline_end = min(len(abel_profile) - 1, abel_pk + baseline_range)
        
        # 使用最小值而不是平均值，更robust
        left_baseline = np.min(abel_profile[left_baseline_start:left_baseline_end]) if left_baseline_end > left_baseline_start else 0
        right_baseline = np.min(abel_profile[right_baseline_start:right_baseline_end]) if right_baseline_end > right_baseline_start else 0
        baseline = max(0, (left_baseline + right_baseline) / 2)
        
        # 校正后的峰高
        corrected_amp = abel_amp - baseline
        if corrected_amp <= 0:
            corrected_amp = abel_amp
            baseline = 0
        
        half_max = baseline + corrected_amp / 2
        
        # 向左找半高点（带线性插值）
        left_idx = abel_pk
        while left_idx > mask_radius and abel_profile[left_idx] > half_max:
            left_idx -= 1
        # 线性插值得到精确位置
        if left_idx < abel_pk and left_idx >= mask_radius:
            y1, y2 = abel_profile[left_idx], abel_profile[left_idx + 1]
            if y2 != y1:
                left_pos = left_idx + (half_max - y1) / (y2 - y1)
            else:
                left_pos = left_idx
        else:
            left_pos = left_idx
        
        # 向右找半高点（带线性插值）
        right_idx = abel_pk
        while right_idx < len(abel_profile) - 1 and abel_profile[right_idx] > half_max:
            right_idx += 1
        # 线性插值
        if right_idx > abel_pk and right_idx < len(abel_profile):
            y1, y2 = abel_profile[right_idx - 1], abel_profile[right_idx]
            if y1 != y2:
                right_pos = right_idx - 1 + (half_max - y1) / (y2 - y1)
            else:
                right_pos = right_idx
        else:
            right_pos = right_idx
        
        fwhm_method1 = right_pos - left_pos
        sigma_fwhm = max(fwhm_method1 / 2.355, 0.5)
        
        # ===== 方法2: 局部高斯拟合 =====
        sigma_gauss = sigma_fwhm  # 默认值
        fit_range = max(5, int(sigma_fwhm * 3))
        fit_start = max(mask_radius, abel_pk - fit_range)
        fit_end = min(len(abel_profile), abel_pk + fit_range + 1)
        
        if fit_end - fit_start >= 5:
            r_fit = np.arange(fit_start, fit_end)
            y_fit = abel_profile[fit_start:fit_end] - baseline
            y_fit = np.maximum(y_fit, 0)
            
            def gaussian(r, amp, r0, sigma):
                return amp * np.exp(-((r - r0)**2) / (2 * sigma**2))
            
            try:
                # 初始猜测
                p0 = [corrected_amp, abel_pk, sigma_fwhm]
                bounds = ([0, fit_start, 0.5], [corrected_amp * 2, fit_end, fit_range])
                popt, _ = curve_fit(gaussian, r_fit, y_fit, p0=p0, bounds=bounds, maxfev=500)
                sigma_gauss = popt[2]
            except:
                sigma_gauss = sigma_fwhm
        
        # ===== 方法3: 二阶矩方法 =====
        moment_range = max(5, int(sigma_fwhm * 2.5))
        m_start = max(mask_radius, abel_pk - moment_range)
        m_end = min(len(abel_profile), abel_pk + moment_range + 1)
        
        r_moment = np.arange(m_start, m_end)
        y_moment = abel_profile[m_start:m_end] - baseline
        y_moment = np.maximum(y_moment, 0)
        
        total_weight = np.sum(y_moment)
        if total_weight > 0:
            mean_r = np.sum(r_moment * y_moment) / total_weight
            variance = np.sum(y_moment * (r_moment - mean_r)**2) / total_weight
            sigma_moment = np.sqrt(max(variance, 0.25))
        else:
            sigma_moment = sigma_fwhm
        
        # ===== 加权平均（优先使用scipy的peak_widths）=====
        sigma_candidates = []
        weights = []
        
        # scipy peak_widths结果（如果可用且合理）
        if sigma_scipy is not None and 0.5 < sigma_scipy < 20:
            sigma_candidates.append(sigma_scipy)
            weights.append(3.0)  # 最高权重
        
        # 高斯拟合结果
        if 0.5 < sigma_gauss < fit_range:
            sigma_candidates.append(sigma_gauss)
            weights.append(2.0)
        
        # FWHM方法
        if 0.5 < sigma_fwhm < 30:
            sigma_candidates.append(sigma_fwhm)
            weights.append(1.5)
        
        # 二阶矩方法
        if 0.5 < sigma_moment < 30:
            sigma_candidates.append(sigma_moment)
            weights.append(1.0)
        
        # 加权平均
        if sigma_candidates:
            weights = np.array(weights)
            sigma_candidates = np.array(sigma_candidates)
            sigma_final = np.sum(sigma_candidates * weights) / np.sum(weights)
        else:
            sigma_final = 3.0  # 默认值
        
        # 限制在合理范围
        sigma_final = np.clip(sigma_final, 1.0, 20.0)
        
        return abel_pk, sigma_final, corrected_amp

    def _phase1_radial_analysis(self, image_2d: np.ndarray) -> List[Dict]:
        """
        Phase 1: 径向分析 - 提取peak位置和sigma估计。
        
        使用共享数据（噪声参数、径向分布）避免重复计算。
        
        Args:
            image_2d: 输入VMI图像（2D投影图像）
            
        Returns:
            List of peak parameters: [{'r', 'sigma', 'amp', 'local_snr', ...}, ...]
        """
        from scipy.ndimage import gaussian_filter1d
        
        print("Phase 1: Radial Analysis")
        print("=" * 60)
        
        mask_radius = 15
        
        # -----------------------------------------------------------------
        # Step 1.1: 使用共享噪声参数
        # -----------------------------------------------------------------
        print("  [Step 1.1] Using Shared Noise Parameters")
        
        readout_std = self._shared_noise['readout_std']
        total_noise_std = self._shared_noise['total_noise_std']
        snr_estimate = self._shared_noise['snr']
        signal_region = self._shared_radial_profile  # 已减去baseline
        
        print(f"    Readout noise std: {readout_std:.3f} (baseline already removed)")
        print(f"    Overall SNR (radial avg): {snr_estimate:.1f}")
        
        # -----------------------------------------------------------------
        # Step 1.2: 在投影径向分布上找peaks
        # -----------------------------------------------------------------
        print("  [Step 1.2] Peak Detection in Projection Profile")
        
        # 平滑投影径向分布（signal_region已减去baseline）
        proj_smooth = gaussian_filter1d(signal_region, sigma=2)
        proj_smooth = np.maximum(proj_smooth, 0)
        proj_smooth[:mask_radius] = 0
        
        # 找peaks
        proj_peaks = self._phase1_find_peaks_in_projection(
            proj_smooth, total_noise_std, mask_radius
        )
        
        print(f"    Found {len(proj_peaks)} candidate peaks:")
        for i, pk in enumerate(proj_peaks):
            print(f"      Peak {i+1}: r={pk['r_proj']}px, SNR={pk['snr']:.1f}")
        
        # -----------------------------------------------------------------
        # Step 1.3: Abel逆变换
        # -----------------------------------------------------------------
        print("  [Step 1.3] Abel Inverse Transform")
        
        # 贝叶斯去噪（signal_region已减去baseline）
        radial_profile_clean = self._denoise_radial_profile_bayesian(
            signal_region, total_noise_std
        )
        
        # 1D Abel逆变换
        abel_profile = abel.hansenlaw.hansenlaw_transform(
            radial_profile_clean, direction='inverse'
        )
        abel_profile[:mask_radius] = 0
        abel_profile = np.maximum(abel_profile, 0)
        
        # 轻度平滑用于peak检测（不用于sigma估计）
        # 使用较小的窗口以减少展宽
        window_len = 7 if snr_estimate > 50 else 11
        abel_smooth = signal.savgol_filter(abel_profile, window_len, 3)
        abel_smooth[:mask_radius] = 0
        abel_smooth = np.maximum(abel_smooth, 0)
        
        # -----------------------------------------------------------------
        # Step 1.4 & 1.5: Sigma估计 + SNR过滤
        # -----------------------------------------------------------------
        print("  [Step 1.4] Sigma Estimation & SNR Filtering")
        
        # SNR阈值（低SNR时放宽）
        snr_threshold = 1.5 if snr_estimate < 30 else 2.5
        
        initial_guesses = []
        for pk in proj_peaks:
            proj_r = pk['r_proj']
            local_snr = pk['snr']
            
            # 在Abel逆变换结果中估计sigma
            # 使用未平滑的profile以获得准确的sigma估计
            abel_r, sigma, amp = self._phase1_estimate_sigma_from_abel(
                abel_profile, proj_r, search_range=8, mask_radius=mask_radius
            )
            
            # SNR过滤
            if local_snr >= snr_threshold:
                peak_info = {
                    'r': float(abel_r),       # Abel逆变换后的位置（更准确）
                    'r_proj': float(proj_r),  # 投影中的位置（参考）
                    'sigma': float(sigma),
                    'amp': float(amp),
                    'fwhm': float(sigma * 2.355),  # 显式保存FWHM
                    'peak_height': float(amp),     # 峰高（用于显式约束）
                    'local_snr': float(local_snr),
                    'noise_std_at_peak': float(total_noise_std[proj_r] if proj_r < len(total_noise_std) else noise_std)
                }
                initial_guesses.append(peak_info)
                print(f"    ✓ Peak: r={abel_r}px (proj:{proj_r}px), σ={sigma:.2f}px, FWHM={sigma*2.355:.2f}px, SNR={local_snr:.1f}")
            else:
                print(f"    ✗ Rejected: r={proj_r}px, SNR={local_snr:.1f} < {snr_threshold}")
        
        print(f"  Final: {len(initial_guesses)} valid peaks")
        print("=" * 60)
        
        # 存储中间结果供调试
        self._last_proj_profile = proj_smooth
        self._last_abel_profile = abel_smooth
        
        return initial_guesses

    # =========================================================================
    # Phase 2: Angular Analysis - Beta Parameter Extraction
    # =========================================================================
    
    def _phase2_get_polar_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取共享的极坐标图像（在 _init_shared_data 中已计算）。
        
        Returns:
            Tuple of (polar_image, theta_grid)
        """
        return self._shared_polar_image, self._shared_theta_grid
    
    def _phase2_find_optimal_radius(self, polar_image: np.ndarray, 
                                     r_center: float, 
                                     sigma_r: float) -> Tuple[int, float]:
        """
        Step 2.2: 在peak附近找到信号最强的半径位置。
        
        Args:
            polar_image: 极坐标图像 (r, θ)
            r_center: Phase 1给出的peak位置
            sigma_r: peak的宽度
            
        Returns:
            Tuple of (optimal_r, max_intensity)
        """
        n_r = polar_image.shape[0]
        search_range = max(3, int(sigma_r * 1.5))
        r_start = max(10, int(r_center - search_range))
        r_end = min(n_r - 1, int(r_center + search_range))
        
        # 计算每个半径的平均强度
        best_r = int(r_center)
        best_intensity = 0
        
        for r in range(r_start, r_end + 1):
            mean_intensity = np.mean(polar_image[r, :])
            if mean_intensity > best_intensity:
                best_intensity = mean_intensity
                best_r = r
        
        return best_r, best_intensity
    
    # NOTE: Removed unused functions:
    # - _phase2_extract_angular_profile: logic inlined in _phase2_angular_analysis
    # - _phase2_estimate_beta_from_fourier: replaced by _phase2_estimate_beta_folded

    def _phase2_radial_filter(self, polar_image: np.ndarray, 
                               all_params: List[Dict], 
                               target_idx: int) -> np.ndarray:
        """
        对极坐标图像进行径向滤波，分离目标peak的信号。
        
        使用基于peak参数的匹配滤波：
        1. 构建目标peak的高斯窗口
        2. 减去其他peaks的估计贡献（基于它们的参数）
        
        Args:
            polar_image: 极坐标图像 (r, θ)
            all_params: 所有peak的参数
            target_idx: 目标peak的索引
            
        Returns:
            滤波后的极坐标图像, 滤波器
        """
        n_r, n_theta = polar_image.shape
        r_grid = np.arange(n_r)
        
        # 获取目标peak参数
        target_r = all_params[target_idx]['r']
        target_sigma = all_params[target_idx].get('sigma', 3.0)
        
        # 方法：构建一个只选择目标peak区域的窗口
        # 使用更窄的窗口来减少其他peaks的影响
        
        # 找到与相邻peaks的边界
        sorted_params = sorted(enumerate(all_params), key=lambda x: x[1]['r'])
        sorted_idx = [i for i, _ in sorted_params]
        target_pos_in_sorted = sorted_idx.index(target_idx)
        
        # 确定左右边界
        if target_pos_in_sorted == 0:
            left_boundary = 0
        else:
            prev_r = sorted_params[target_pos_in_sorted - 1][1]['r']
            prev_sigma = sorted_params[target_pos_in_sorted - 1][1].get('sigma', 3.0)
            # 边界在两个peak之间，偏向sigma小的那个
            left_boundary = int((prev_r + target_r) / 2)
        
        if target_pos_in_sorted == len(sorted_params) - 1:
            right_boundary = n_r
        else:
            next_r = sorted_params[target_pos_in_sorted + 1][1]['r']
            next_sigma = sorted_params[target_pos_in_sorted + 1][1].get('sigma', 3.0)
            right_boundary = int((target_r + next_r) / 2)
        
        # 构建滤波器：在边界内使用高斯窗口
        radial_filter = np.zeros(n_r)
        
        # 高斯窗口，但限制在边界内
        filter_sigma = target_sigma * 1.5
        gaussian_window = np.exp(-0.5 * ((r_grid - target_r) / filter_sigma)**2)
        
        # 只在边界内应用
        mask = (r_grid >= left_boundary) & (r_grid < right_boundary)
        radial_filter[mask] = gaussian_window[mask]
        
        # 归一化
        if np.max(radial_filter) > 0:
            radial_filter = radial_filter / np.max(radial_filter)
        
        # 应用滤波器
        filtered_polar = polar_image * radial_filter[:, np.newaxis]
        
        return filtered_polar, radial_filter

    def _phase2_fold_angular_profile(self, angular_profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        利用cos²函数的对称性折叠角度分布到[0, π]。
        
        对于 I(θ) = I₀[1 + β·P₂(cosθ)]，P₂(cosθ) = (3cos²θ - 1)/2：
        - cos²(θ) = cos²(-θ) = cos²(2π-θ)，所以I(θ) = I(2π-θ)
        
        将[0, 2π)折叠到[0, π]：
        - [0, π): 原始
        - [π, 2π): 翻转后与[0, π)平均
        
        Args:
            angular_profile: 完整的角度分布 [0, 2π)
            
        Returns:
            Tuple of (折叠后的角度分布 [0, π], 对应的theta网格)
        """
        n = len(angular_profile)
        n_half = n // 2
        
        # 两个半圆
        first_half = angular_profile[:n_half]   # [0, π)
        second_half = angular_profile[n_half:]  # [π, 2π)
        
        # 利用对称性：I(θ) = I(2π-θ)
        # second_half翻转后应该与first_half相同
        folded = (first_half + second_half[::-1]) / 2.0
        
        # 对应的theta网格 [0, π)
        theta_folded = np.linspace(0, np.pi, n_half, endpoint=False)
        
        return folded, theta_folded

    def _phase2_estimate_beta_fft(self, angular_profile: np.ndarray) -> Tuple[float, float, Dict]:
        """
        使用角向FFT直接从频谱提取β。
        
        理论推导：
        I(θ) = I₀[1 + β·P₂(cosθ)] = I₀[1 + β·(3cos²θ - 1)/2]
        
        使用 cos²θ = (1 + cos2θ)/2：
        I(θ) = I₀(1 + β/4) + I₀·(3β/4)·cos2θ
        
        所以角向FFT应该看到：
        - DC分量 (k=0): A₀ = I₀(1 + β/4)
        - cos2θ分量 (k=2): 幅度 = I₀·|3β/4|，相位决定符号
          - β > 0: 相位 ≈ 0°
          - β < 0: 相位 ≈ 180°
        
        Args:
            angular_profile: 完整的角度分布 [0, 2π)，未折叠
            
        Returns:
            Tuple of (beta, uncertainty, debug_info)
        """
        n = len(angular_profile)
        
        # 角向FFT
        fft = np.fft.fft(angular_profile)
        
        # DC分量 (k=0)
        dc = np.abs(fft[0]) / n  # 归一化
        
        # cos2θ分量 (k=2)
        # 幅度
        cos2_amp = 2 * np.abs(fft[2]) / n
        # 相位（决定β的符号）
        phase = np.angle(fft[2])
        # 如果相位接近180°（±π），说明β为负
        sign = 1.0 if np.abs(phase) < np.pi/2 else -1.0
        
        # 带符号的cos2分量
        cos2_signed = sign * cos2_amp
        
        # 从频谱计算β
        # DC = I₀(1 + β/4)
        # cos2_signed = I₀·(3β/4)
        # 解出：β = 4·cos2_signed / (3·DC - cos2_signed)
        
        if dc > 1e-6:
            beta_fft = 4.0 * cos2_signed / (3.0 * dc - cos2_signed)
            beta_fft = np.clip(beta_fft, -1.0, 2.0)
        else:
            beta_fft = 0.0
        
        # 估计不确定度（基于高频噪声）
        high_freq_power = np.sum(np.abs(fft[5:n//2])**2)
        signal_power = np.abs(fft[0])**2 + np.abs(fft[2])**2
        noise_ratio = np.sqrt(high_freq_power / (signal_power + 1e-10))
        beta_uncertainty = max(0.02, min(0.5, noise_ratio * 0.5))
        
        debug_info = {
            'dc': dc,
            'cos2_amp': cos2_amp,
            'cos2_signed': cos2_signed,
            'phase_deg': np.degrees(phase),
            'sign': sign,
            'noise_ratio': noise_ratio
        }
        
        return beta_fft, beta_uncertainty, debug_info

    def _phase2_estimate_beta_single_radius(self, angular_profile: np.ndarray, 
                                             theta_grid: np.ndarray) -> Tuple[float, float]:
        """
        从单个半径的角度分布估计β（综合FFT和拟合方法）。
        
        Args:
            angular_profile: 角度分布（已折叠到[0,π]）
            theta_grid: 对应的角度网格
            
        Returns:
            Tuple of (beta, uncertainty)
        """
        from scipy.optimize import curve_fit
        from scipy.ndimage import gaussian_filter1d
        
        # 轻微平滑
        profile_smooth = gaussian_filter1d(angular_profile, sigma=1.0, mode='nearest')
        
        # 计算P2
        cos_theta = np.cos(theta_grid)
        P2 = 0.5 * (3 * cos_theta**2 - 1)
        
        # 线性回归估计
        I0_est = np.mean(profile_smooth)
        if I0_est > 1e-6:
            y_linear = profile_smooth / I0_est - 1
            weights = np.sqrt(np.maximum(profile_smooth, 1))
            numerator = np.sum(weights * y_linear * P2)
            denominator = np.sum(weights * P2**2)
            beta_linear = numerator / (denominator + 1e-10)
            beta_linear = np.clip(beta_linear, -1.0, 2.0)
        else:
            return 0.0, 1.0
        
        # 非线性拟合
        def angular_model(theta, I0, beta):
            cos_t = np.cos(theta)
            P2_t = 0.5 * (3 * cos_t**2 - 1)
            return I0 * (1 + beta * P2_t)
        
        try:
            popt, pcov = curve_fit(
                angular_model, theta_grid, profile_smooth,
                p0=[I0_est, beta_linear],
                bounds=([0, -1.2], [np.inf, 2.2]),
                maxfev=1000
            )
            beta_fit = popt[1]
            beta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.5
        except:
            beta_fit = beta_linear
            beta_err = 0.5
        
        return np.clip(beta_fit, -1.0, 2.0), beta_err

    def _phase2_multi_radius_beta(self, polar_image: np.ndarray, 
                                   theta_grid: np.ndarray,
                                   r_center: float, 
                                   sigma_r: float) -> Dict:
        """
        改进的β估计方法：只使用peak中心附近强度最高的几个半径。
        
        诊断发现：
        1. peak边缘的β估计不准确（信号弱，噪声大）
        2. 高斯加权平均反而降低准确性
        3. 最佳策略是只用强度最大的2-3个半径
        
        Args:
            polar_image: 极坐标图像 (r, θ)
            theta_grid: 角度网格
            r_center: peak中心位置
            sigma_r: peak宽度
            
        Returns:
            Dict with beta estimate and debug info
        """
        n_r = polar_image.shape[0]
        
        # 搜索范围：r ± 1σ（比之前更窄）
        search_range = max(3, int(sigma_r * 1.0))
        r_start = max(10, int(r_center - search_range))
        r_end = min(n_r - 1, int(r_center + search_range + 1))
        
        # 找到强度最大的半径
        r_search = np.arange(r_start, r_end + 1)
        intensities = np.array([np.mean(polar_image[r, :]) for r in r_search])
        max_idx = np.argmax(intensities)
        r_max = r_search[max_idx]
        
        # 只使用r_max附近的2-3个半径（强度最高的区域）
        # 这样可以避免peak边缘的不准确估计
        n_use = 3  # 使用3个半径
        half_n = n_use // 2
        
        r_use_start = max(r_start, r_max - half_n)
        r_use_end = min(r_end, r_max + half_n)
        r_samples = np.arange(r_use_start, r_use_end + 1)
        n_samples = len(r_samples)
        
        if n_samples < 1:
            # 回退到单半径
            r_idx = int(np.clip(r_center, 10, n_r - 1))
            angular_profile = polar_image[r_idx, :]
            angular_folded, theta_folded = self._phase2_fold_angular_profile(angular_profile)
            beta, beta_err = self._phase2_estimate_beta_single_radius(angular_folded, theta_folded)
            return {
                'beta': beta,
                'beta_uncertainty': beta_err,
                'method': 'single_radius',
                'n_samples': 1,
                'snr_overall': 0,
                'beta_fit': beta,
                'beta_linear': beta,
                'beta_fourier': beta,
                'dc_component': np.mean(angular_profile),
                'cos_2theta_amp': 0,
                'smooth_sigma': 1.0
            }
        
        # 对选中的几个半径计算β（同时使用FFT和拟合方法）
        beta_fit_values = []
        beta_fft_values = []
        beta_errors = []
        sample_intensities = []
        fft_debug_list = []
        
        for r_idx in r_samples:
            angular_profile = polar_image[r_idx, :]  # 完整的[0, 2π)
            
            # 方法1: FFT方法（直接从频谱提取）
            beta_fft, beta_fft_err, fft_debug = self._phase2_estimate_beta_fft(angular_profile)
            beta_fft_values.append(beta_fft)
            fft_debug_list.append(fft_debug)
            
            # 方法2: 拟合方法（折叠后拟合）
            angular_folded, theta_folded = self._phase2_fold_angular_profile(angular_profile)
            beta_fit, beta_fit_err = self._phase2_estimate_beta_single_radius(angular_folded, theta_folded)
            beta_fit_values.append(beta_fit)
            
            beta_errors.append(min(beta_fft_err, beta_fit_err))
            sample_intensities.append(np.mean(angular_profile))
        
        beta_fit_values = np.array(beta_fit_values)
        beta_fft_values = np.array(beta_fft_values)
        beta_errors = np.array(beta_errors)
        sample_intensities = np.array(sample_intensities)
        
        # 强度加权平均
        intensity_weights = sample_intensities / (np.sum(sample_intensities) + 1e-10)
        beta_fit_avg = np.sum(beta_fit_values * intensity_weights)
        beta_fft_avg = np.sum(beta_fft_values * intensity_weights)
        
        # 综合两种方法：FFT方法通常更准确（直接从频谱提取）
        # 但如果两种方法差异太大，说明可能有问题，取保守值
        if abs(beta_fft_avg - beta_fit_avg) < 0.3:
            # 两种方法一致，使用FFT结果（更直接）
            beta_avg = beta_fft_avg
            method_used = 'fft'
        else:
            # 差异较大，取平均
            beta_avg = (beta_fft_avg + beta_fit_avg) / 2
            method_used = 'hybrid'
        
        # 估计不确定度
        beta_var = np.sum(intensity_weights * (beta_fft_values - beta_avg)**2)
        beta_std = np.sqrt(beta_var)
        avg_error = np.mean(beta_errors)
        beta_uncertainty = max(beta_std, avg_error * 0.5, 0.02)
        
        # 计算SNR
        mean_intensity = np.mean(sample_intensities)
        noise_est = np.std(sample_intensities) if n_samples > 1 else mean_intensity * 0.1
        snr_overall = mean_intensity / (noise_est + 1e-6)
        
        # 提取FFT调试信息
        avg_dc = np.mean([d['dc'] for d in fft_debug_list])
        avg_cos2 = np.mean([d['cos2_amp'] for d in fft_debug_list])
        
        return {
            'beta': np.clip(beta_avg, -1.0, 2.0),
            'beta_uncertainty': beta_uncertainty,
            'method': f'{method_used}_{n_samples}r',
            'n_samples': n_samples,
            'snr_overall': snr_overall,
            'beta_fit': beta_fit_avg,
            'beta_fft': beta_fft_avg,
            'beta_linear': np.mean(beta_fit_values),
            'beta_fourier': beta_fft_avg,
            'dc_component': avg_dc,
            'cos_2theta_amp': avg_cos2,
            'smooth_sigma': 1.0,
            'beta_fit_values': beta_fit_values.tolist(),
            'beta_fft_values': beta_fft_values.tolist(),
            'r_samples': r_samples.tolist(),
            'r_max': int(r_max)
        }

    def _phase2_estimate_beta_folded(self, angular_profile: np.ndarray, 
                                      theta_grid: np.ndarray) -> Dict:
        """
        从折叠后的角度分布估计β。
        
        折叠后的数据范围是[0, π/2]，使用简单的拟合方法。
        
        Args:
            angular_profile: 折叠后的角度分布 [0, π/2]
            theta_grid: 对应的角度网格
            
        Returns:
            Dict with beta estimate and debug info
        """
        from scipy.optimize import curve_fit
        from scipy.ndimage import gaussian_filter1d
        
        n_theta = len(angular_profile)
        
        # 轻微平滑
        profile_smooth = gaussian_filter1d(angular_profile, sigma=1.0, mode='nearest')
        
        # 计算P2（偏振沿水平方向，theta=0）
        cos_theta = np.cos(theta_grid)
        P2 = 0.5 * (3 * cos_theta**2 - 1)
        
        # 估计SNR
        mean_signal = np.mean(profile_smooth)
        noise_est = np.std(angular_profile - profile_smooth)
        snr_overall = mean_signal / (noise_est + 1e-6)
        
        # ===== 方法1: 线性回归 =====
        I0_est = np.mean(profile_smooth)
        if I0_est > 1e-6:
            y_linear = profile_smooth / I0_est - 1
            # 加权线性回归
            weights = np.sqrt(np.maximum(profile_smooth, 1))
            numerator = np.sum(weights * y_linear * P2)
            denominator = np.sum(weights * P2**2)
            beta_linear = numerator / (denominator + 1e-10)
            beta_linear = np.clip(beta_linear, -1.0, 2.0)
        else:
            beta_linear = 0.0
        
        # ===== 方法2: 非线性拟合 =====
        def angular_model(theta, I0, beta):
            cos_t = np.cos(theta)
            P2_t = 0.5 * (3 * cos_t**2 - 1)
            return I0 * (1 + beta * P2_t)
        
        beta_fit = beta_linear
        beta_fit_err = 0.5
        fit_success = False
        
        try:
            popt, pcov = curve_fit(
                angular_model, theta_grid, profile_smooth,
                p0=[I0_est, beta_linear],
                bounds=([0, -1.2], [np.inf, 2.2]),
                maxfev=2000
            )
            beta_fit = popt[1]
            if pcov[1, 1] > 0:
                beta_fit_err = np.sqrt(pcov[1, 1])
            fit_success = True
        except:
            beta_fit = beta_linear
        
        # ===== 方法3: 端点比值法 =====
        # I(0)/I(π/2) = (1 + β·P₂(1)) / (1 + β·P₂(0))
        # P₂(1) = 1, P₂(0) = -0.5
        # R = I(0)/I(π/2) = (1 + β) / (1 - β/2)
        # 解出: β = 2(R-1)/(R+2)
        # 注意：theta_grid是[0, π)，所以末尾是θ≈π，不是π/2
        # 需要找到θ=π/2的位置
        n_half = n_theta // 2  # θ=π/2的位置
        I_0 = np.mean(profile_smooth[:max(1, n_theta//20)])  # θ≈0附近
        I_90 = np.mean(profile_smooth[n_half-n_theta//20:n_half+n_theta//20])  # θ≈π/2附近
        if I_90 > 1e-6 and I_0 > 1e-6:
            R = I_0 / I_90
            beta_ratio = 2 * (R - 1) / (R + 2)
            beta_ratio = np.clip(beta_ratio, -1.0, 2.0)
        else:
            beta_ratio = beta_linear
        
        # ===== 综合方法 =====
        # 非线性拟合通常更准确，因为它直接拟合物理模型
        if fit_success:
            # 如果拟合成功，主要使用拟合结果
            if abs(beta_fit - beta_linear) < 0.2:
                # 一致，使用拟合结果
                beta_final = beta_fit
                method = 'consistent'
            elif abs(beta_fit - beta_ratio) < 0.2:
                # 拟合与比值法一致
                beta_final = beta_fit
                method = 'fit+ratio'
            else:
                # 使用拟合结果（通常更可靠）
                beta_final = beta_fit
                method = 'fit'
        else:
            beta_final = beta_linear
            method = 'linear'
        
        beta_final = np.clip(beta_final, -1.0, 2.0)
        
        # 计算不确定度
        beta_uncertainty = beta_fit_err if fit_success else 0.1
        
        return {
            'beta': beta_final,
            'beta_uncertainty': beta_uncertainty,
            'method': method,
            'snr_overall': snr_overall,
            'beta_fit': beta_fit,
            'beta_linear': beta_linear,
            'beta_fourier': beta_ratio,  # 用ratio方法代替fourier
            'dc_component': I0_est,
            'cos_2theta_amp': 0,
            'smooth_sigma': 1.0
        }

    def _phase2_angular_analysis(self, image_raw: np.ndarray, partial_params: List[Dict]) -> List[Dict]:
        """
        Phase 2: 角度分析 - 提取β参数。
        
        直接在投影图像上提取角度分布（投影不改变角度分布形式）。
        使用四象限对称性折叠提高SNR。
        
        Args:
            image_raw: 输入VMI图像（投影图像）
            partial_params: Phase 1的peak参数列表
            
        Returns:
            更新后的参数列表（添加了beta）
        """
        print("\nPhase 2: Angular Analysis")
        print("=" * 60)
        
        # -----------------------------------------------------------------
        # Step 2.1: 使用共享的极坐标图像
        # -----------------------------------------------------------------
        print("  [Step 2.1] Using Shared Polar Image")
        polar_image, theta_grid = self._phase2_get_polar_image()
        print(f"    Polar image shape: {polar_image.shape} (r × θ)")
        
        updated_params = []
        n_peaks = len(partial_params)
        
        for i, p in enumerate(partial_params):
            r_center = p['r']
            sigma_r = p.get('sigma', 3.0)
            
            if r_center < 10 or r_center >= self.radius - 1:
                continue
            
            print(f"\n  Processing Peak {i+1}: r = {r_center:.1f} px")
            
            # -----------------------------------------------------------------
            # Step 2.2: 预滤波 - 分离目标peak的信号
            # -----------------------------------------------------------------
            if n_peaks > 1:
                filtered_polar, radial_filter = self._phase2_radial_filter(
                    polar_image, partial_params, i
                )
                filter_width = np.sum(radial_filter > 0.5)
                print(f"    [Step 2.2] Radial filter applied (effective width: {filter_width}px)")
            else:
                filtered_polar = polar_image
                print(f"    [Step 2.2] Single peak - no filtering needed")
            
            # -----------------------------------------------------------------
            # Step 2.3: 优化peak位置（在滤波后的图像上）
            # -----------------------------------------------------------------
            r_optimal, max_intensity = self._phase2_find_optimal_radius(
                filtered_polar, r_center, sigma_r
            )
            print(f"    [Step 2.3] Optimal radius: {r_optimal} px (intensity: {max_intensity:.2f})")
            
            p['r_original'] = r_center
            p['r'] = float(r_optimal)
            
            # -----------------------------------------------------------------
            # Step 2.4 & 2.5: 多半径高斯加权平均提取β
            # -----------------------------------------------------------------
            # 使用新的多半径平均方法：在r±σ范围内对每个半径分别计算β，然后加权平均
            beta_result = self._phase2_multi_radius_beta(
                filtered_polar, theta_grid, r_optimal, sigma_r
            )
            
            print(f"    [Step 2.4-2.5] Multi-radius β estimation (n_samples={beta_result.get('n_samples', 1)})")
            
            p['beta'] = beta_result['beta']
            p['beta_uncertainty'] = beta_result['beta_uncertainty']
            p['beta_debug'] = {
                'method_used': beta_result['method'],
                'snr_overall': beta_result['snr_overall'],
                'beta_fit': beta_result.get('beta_fit', 0),
                'beta_linear': beta_result.get('beta_linear', 0),
                'beta_fourier': beta_result.get('beta_fourier', 0),
                'dc_component': beta_result['dc_component'],
                'cos_2theta_amplitude': beta_result['cos_2theta_amp'],
                'smooth_sigma_used': beta_result['smooth_sigma'],
                'n_samples': beta_result.get('n_samples', 1),
                'r_samples': beta_result.get('r_samples', [])
            }
            
            print(f"    β = {p['beta']:.3f} ± {p['beta_uncertainty']:.3f} ({beta_result['method']})")
            if 'beta_values' in beta_result:
                beta_vals = beta_result['beta_values']
                print(f"              β_range=[{min(beta_vals):.3f}, {max(beta_vals):.3f}], β_linear={beta_result.get('beta_linear', 0):.3f}")
            
            updated_params.append(p)
        
        print("\n" + "=" * 60)
        return updated_params

    def _forward_model_loss(self, params_flat: np.ndarray, image_target: np.ndarray, 
                           target_profile_1d: np.ndarray, target_cdf_norm: np.ndarray,
                           priors: List[Dict], n_peaks: int) -> np.ndarray:
        """
        Compute loss function for forward model optimization.
        
        Combines multiple loss terms:
        - 2D pixel residuals (intensity fitting)
        - 1D radial profile residuals
        - CDF shape constraint (robust to noise)
        - Explicit peak shape constraints (FWHM + peak height)
        
        Args:
            params_flat: Flattened parameter array [r, sigma, amp, beta] per peak
            image_target: Target VMI image
            target_profile_1d: Target radial profile
            target_cdf_norm: Normalized cumulative distribution of target
            priors: Prior parameter estimates for regularization
            n_peaks: Number of peaks
            
        Returns:
            Concatenated residual array for least_squares
        """
        params = params_flat.reshape(n_peaks, 4)
        
        # Build 3D model
        img_3d_model = np.zeros((self.n, self.n))
        
        # 显式峰形约束残差
        fwhm_residuals = []
        height_residuals = []
        sigma_residuals = []
        beta_residuals = []
        
        for i in range(n_peaks):
            r0, sig, amp, beta = params[i]
            sig = max(sig, 0.8)  # Physical minimum
            
            radial = amp * np.exp(-((self.R - r0)**2) / (2 * sig**2))
            angular = 1 + beta * self.P2_GRID
            img_3d_model += radial * angular
            
            # ===== 显式峰形约束 =====
            prior_sigma = priors[i]['sigma']
            prior_fwhm = priors[i].get('fwhm', prior_sigma * 2.355)
            prior_height = priors[i].get('peak_height', priors[i].get('amp', amp))
            prior_beta = priors[i].get('beta', 0.0)
            beta_uncertainty = priors[i].get('beta_uncertainty', 0.5)
            
            # 1. FWHM约束 - 直接约束峰宽（软约束）
            model_fwhm = sig * 2.355
            fwhm_deviation = (model_fwhm - prior_fwhm) / (prior_fwhm + 1e-6)
            fwhm_residuals.append(fwhm_deviation * 3.0)
            
            # 2. 峰高约束 - 打破amp/sigma的ambiguity（软约束）
            height_deviation = (amp - prior_height) / (prior_height + 1e-6)
            height_residuals.append(height_deviation * 2.0)
            
            # 3. Sigma双向正则化（软约束）
            sigma_deviation = (sig - prior_sigma) / (prior_sigma + 1e-6)
            sigma_residuals.append(sigma_deviation * 1.5)
            
            # 4. Beta先验约束（根据不确定度加权）
            # Phase 2的beta估计通常比较准确，给予较强约束
            # 由于2D残差有~262144个元素，需要大幅增加beta约束权重
            # 不确定度越小，约束越强
            beta_weight = 1.0 / (beta_uncertainty + 0.05)
            beta_deviation = (beta - prior_beta) * beta_weight
            beta_residuals.append(beta_deviation * 1000.0)  # 进一步增加权重，防止过度调整

        # Forward projection
        proj_model = abel.Transform(img_3d_model, method='hansenlaw', direction='forward', verbose=False).transform
        
        # 2D pixel residuals with Poisson-like weighting
        weights_2d = 1.0 / np.sqrt(np.abs(image_target) + 1.0)
        res_2d = (proj_model - image_target) * weights_2d
        
        # 1D radial profile
        model_profile_1d = self._compute_radial_profile(proj_model)
        mask_radius = 12
        
        # CDF shape constraint (robust to noise)
        model_cdf = np.cumsum(model_profile_1d)
        model_total = model_cdf[-1] + 1e-9
        model_cdf_norm = model_cdf / model_total
        
        cdf_diff = (model_cdf_norm - target_cdf_norm)
        cdf_diff[:mask_radius] = 0.0
        cdf_weight = 1500.0
        
        # 1D intensity residuals
        weights_1d = 1.0 / np.sqrt(np.abs(target_profile_1d) + 1.0)
        weights_1d[:mask_radius] = 0.0
        res_1d = (model_profile_1d - target_profile_1d) * weights_1d
        
        lambda_factor = 20.0 
        
        # ===== 频域约束（关键改进！）=====
        # 频谱形状直接反映peak宽度：
        # - 窄peak → 高频成分多
        # - 宽peak → 低频成分多
        # 这可以打破 amp↑+sigma↓ vs amp↓+sigma↑ 的ambiguity
        res_spectral = self._compute_spectral_residuals(
            model_profile_1d, target_profile_1d, mask_radius
        )
        spectral_weight = 50.0  # 频谱约束权重
        
        return np.concatenate([
            res_2d.ravel(),
            res_1d * lambda_factor,
            cdf_diff * cdf_weight,
            res_spectral * spectral_weight,
            np.array(fwhm_residuals),
            np.array(height_residuals),
            np.array(sigma_residuals),
            np.array(beta_residuals)
        ])

    def _compute_spectral_residuals(self, model_profile: np.ndarray, 
                                     target_profile: np.ndarray,
                                     mask_radius: int) -> np.ndarray:
        """
        计算频域残差，用于约束peak宽度。
        
        频谱形状直接反映peak宽度：
        - 窄peak → 高频成分多（频谱衰减慢）
        - 宽peak → 低频成分多（频谱衰减快）
        
        这可以打破 amp↑+sigma↓ vs amp↓+sigma↑ 的ambiguity，
        因为虽然总能量相同，但频谱形状不同。
        
        Args:
            model_profile: 模型的径向分布
            target_profile: 目标的径向分布
            mask_radius: 中心mask半径
            
        Returns:
            频谱残差数组
        """
        # 只使用有效区域
        model_valid = model_profile[mask_radius:].copy()
        target_valid = target_profile[mask_radius:].copy()
        
        # 归一化（消除总强度差异，只比较形状）
        model_norm = model_valid / (np.sum(model_valid) + 1e-10)
        target_norm = target_valid / (np.sum(target_valid) + 1e-10)
        
        # 计算FFT
        model_fft = np.fft.rfft(model_norm)
        target_fft = np.fft.rfft(target_norm)
        
        # 计算功率谱（幅度的平方）
        model_power = np.abs(model_fft)**2
        target_power = np.abs(target_fft)**2
        
        # 归一化功率谱（排除DC分量）
        model_power_norm = model_power[1:] / (np.sum(model_power[1:]) + 1e-10)
        target_power_norm = target_power[1:] / (np.sum(target_power[1:]) + 1e-10)
        
        # 频谱残差：比较功率谱形状
        # 使用log尺度可以更好地捕捉高频差异
        eps = 1e-10
        spectral_residuals = np.log(model_power_norm + eps) - np.log(target_power_norm + eps)
        
        # 只使用前一半频率（高频通常是噪声）
        n_use = len(spectral_residuals) // 2
        spectral_residuals = spectral_residuals[:n_use]
        
        # 频率加权：中频最重要（包含peak形状信息）
        freqs = np.arange(1, n_use + 1)
        freq_weights = np.exp(-0.5 * ((freqs - n_use//4) / (n_use//3))**2)
        
        return spectral_residuals * freq_weights

    def solve(self, image_2d: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Main solver: reconstruct peak parameters from VMI image.
        
        Args:
            image_2d: Input VMI image (2D numpy array)
            
        Returns:
            Tuple of:
            - final_params: List of dictionaries with reconstructed parameters
              (r, sigma, fwhm, amp, beta)
            - r_grid: Radial grid array
            - recon_profile: Reconstructed radial intensity profile
        """
        t0 = time.time()
        
        # Initialize shared data (polar image, noise params, radial profile)
        # This is computed once and reused across Phase 1, 2, 3
        print("Initializing shared data...")
        self._init_shared_data(image_2d, n_theta=720)
        
        # 创建减去baseline的图像（用于Phase 3优化）
        baseline = self._shared_noise.get('estimated_baseline', 0.0)
        image_2d_corrected = np.maximum(image_2d - baseline, 0)
        
        # Phase 1: Initial radial analysis (uses shared noise params)
        init_params_no_beta = self._phase1_radial_analysis(image_2d_corrected)
        if not init_params_no_beta: 
            return [], self.r_grid_1d, np.zeros_like(self.r_grid_1d)
            
        # Phase 2: Angular analysis for beta (uses shared polar image, already baseline-corrected)
        full_init_params = self._phase2_angular_analysis(image_2d_corrected, init_params_no_beta)
        
        # Phase 3: Use shared radial profile (already baseline-corrected)
        target_profile_1d = self._shared_radial_profile
        
        # Precompute target CDF
        target_cdf = np.cumsum(target_profile_1d)
        target_total = target_cdf[-1] + 1e-9
        target_cdf_norm = target_cdf / target_total
        
        n_peaks = len(full_init_params)
        x0, lb, ub = [], [], []
        priors = []
        
        print(f"Phase 3: Robust Optimization ({n_peaks} peaks)...")
        
        for p in full_init_params:
            x0.extend([p['r'], p['sigma'], p['amp'], p['beta']])
            priors.append({
                'sigma': p['sigma'],
                'fwhm': p.get('fwhm', p['sigma'] * 2.355),
                'peak_height': p.get('peak_height', p['amp']),
                'amp': p['amp'],
                'beta': p['beta'],
                'beta_uncertainty': p.get('beta_uncertainty', 0.5)
            })
            lb.extend([max(5.0, p['r']-8), 0.5, 0.0, -1.1])
            ub.extend([p['r']+8, 40.0, np.inf, 2.1])
            
        try:
            res = least_squares(
                self._forward_model_loss, 
                x0=np.array(x0), 
                bounds=(np.array(lb), np.array(ub)),
                args=(image_2d_corrected, target_profile_1d, target_cdf_norm, priors, n_peaks), 
                loss='soft_l1', 
                f_scale=1.0,
                method='trf', 
                ftol=1e-4, xtol=1e-4, max_nfev=40
            )
            final_x = res.x
        except Exception as e:
            print(f"Optimization warning: {e}")
            final_x = np.array(x0)
            
        p_reshaped = final_x.reshape(n_peaks, 4)
        
        final_params = []
        recon_profile = np.zeros_like(self.r_grid_1d)
        
        if n_peaks > 0:
            max_amp = np.max(p_reshaped[:, 2])
            amp_threshold = 0.05 * max_amp
        else:
            amp_threshold = 0
            
        for i in range(n_peaks):
            r0, sig, amp, beta = p_reshaped[i]
            if amp < amp_threshold or r0 < 10:
                continue
            
            final_params.append({
                'r': r0, 
                'sigma': sig, 
                'fwhm': 2.355 * sig, 
                'amp': amp, 
                'beta': beta
            })
            recon_profile += amp * np.exp(-((self.r_grid_1d - r0)**2)/(2*sig**2))
            
        print(f"Solver Time: {time.time()-t0:.2f}s")
        return final_params, self.r_grid_1d, recon_profile


def reconstruct_2d_from_params(params: List[Dict], n_pixels: int) -> np.ndarray:
    """
    Reconstruct 2D slice image from fitted parameters.
    
    Args:
        params: List of peak parameter dictionaries
        n_pixels: Image size
        
    Returns:
        2D reconstructed image (slice, not projection)
    """
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y, x = y - center, x - center
    r_grid = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    img = np.zeros_like(r_grid)
    for p in params:
        radial = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        img += radial * (1 + p['beta'] * P2)
    return img


def radius_to_energy(radius_px: float, pixel_size_mm: float, vmi_k: float, 
                     mass_amu: float = None) -> float:
    """
    Convert detector radius to electron energy.
    
    Args:
        radius_px: Radius in pixels
        pixel_size_mm: Pixel size in mm
        vmi_k: VMI conversion coefficient (mm/(m/s))
        mass_amu: Particle mass in amu (default: electron)
        
    Returns:
        Energy in eV
    """
    from scipy.constants import electron_mass, elementary_charge, atomic_mass
    
    if mass_amu is None:
        mass_amu = electron_mass / atomic_mass
    
    radius_mm = radius_px * pixel_size_mm
    velocity = radius_mm / vmi_k  # m/s
    
    mass_kg = mass_amu * atomic_mass
    E_joule = 0.5 * mass_kg * velocity**2
    E_eV = E_joule / elementary_charge
    
    return E_eV


def reconstruct_vmi_image(image: np.ndarray, config=None, 
                          verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Main reconstruction function for VMI images.
    
    Args:
        image: Input VMI image (2D numpy array)
        config: Optional Config object from Abel_forward_simulation for energy conversion
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - params: List of reconstructed peak parameters
        - metadata: Dictionary with reconstruction metadata
    """
    n_pixels = image.shape[0]
    
    # Create fitter and solve
    fitter = PhysicsBasedFitter(n_pixels)
    params, r_grid, recon_profile = fitter.solve(image)
    
    # Convert to energy if config provided
    if config is not None:
        for p in params:
            p['energy_eV'] = radius_to_energy(
                p['r'], 
                config.pixel_size, 
                config.vmi_k,
                config.mass
            )
    
    # Calculate branching ratios from amplitudes
    total_amp = sum(p['amp'] for p in params)
    if total_amp > 0:
        for p in params:
            p['branching_ratio'] = p['amp'] / total_amp
    
    metadata = {
        'n_peaks': len(params),
        'r_grid': r_grid,
        'recon_profile': recon_profile,
        'image_size': n_pixels
    }
    
    if verbose:
        print("\n" + "="*60)
        print("RECONSTRUCTION RESULTS")
        print("="*60)
        for i, p in enumerate(params):
            print(f"\nPeak {i+1}:")
            print(f"  Radius: {p['r']:.1f} px")
            if 'energy_eV' in p:
                print(f"  Energy: {p['energy_eV']:.3f} eV")
            print(f"  Sigma: {p['sigma']:.2f} px (FWHM: {p['fwhm']:.2f} px)")
            print(f"  Beta: {p['beta']:.3f}")
            if 'branching_ratio' in p:
                print(f"  Branching ratio: {p['branching_ratio']:.3f}")
        print("="*60)
    
    return params, metadata


def compare_reconstruction(true_params: Dict, recon_params: List[Dict], 
                          config=None, rbasex_params: List[Dict] = None) -> None:
    """
    Compare reconstructed parameters with ground truth.
    
    Args:
        true_params: Dictionary with true simulation parameters
            Expected keys: E_centers, Betas, branching_ratios, sigma_laser
        recon_params: List of reconstructed peak parameters (PhysicsBasedFitter)
        config: Config object for energy conversion
        rbasex_params: Optional list of reconstructed parameters from rBasex method
    """
    print("\n" + "="*120)
    print("RECONSTRUCTION COMPARISON - PhysicsBasedFitter vs rBasex")
    print("="*120)
    
    # Extract true values
    true_E = true_params.get('E_centers', [])
    true_beta = true_params.get('Betas', [])
    true_br = true_params.get('branching_ratios', [])
    
    # Sort reconstructed by radius/energy
    recon_sorted = sorted(recon_params, key=lambda x: x.get('energy_eV', x['r']))
    rbasex_sorted = sorted(rbasex_params, key=lambda x: x.get('energy_eV', x['r'])) if rbasex_params else []
    
    # Calculate true FWHM from sigma_laser
    sigma_laser = true_params.get('sigma_laser', 0.015)  # eV
    
    # Header
    print(f"\n{'Peak':<6} | {'Parameter':<15} | {'True':<12} | {'Physics':<12} | {'Err(Ph)':<10} | "
          f"{'rBasex':<12} | {'Err(rB)':<10}")
    print("-"*135)
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(true_E, true_beta, true_br)):
        # Find closest reconstructed peaks
        phys_match = None
        rb_match = None
        
        if config is not None:
            if recon_sorted:
                phys_match = min(recon_sorted, 
                               key=lambda x: abs(x.get('energy_eV', 0) - E_true),
                               default=None)
            if rbasex_sorted:
                rb_match = min(rbasex_sorted,
                             key=lambda x: abs(x.get('energy_eV', 0) - E_true),
                             default=None)
        else:
            phys_match = recon_sorted[i] if i < len(recon_sorted) else None
            rb_match = rbasex_sorted[i] if i < len(rbasex_sorted) else None
        
        # Calculate true radius and FWHM in pixels
        if config is not None:
            r_mm_true = config.get_expected_radius(E_true)
            r_px_true = r_mm_true / config.pixel_size
            # True FWHM in pixels (from energy spread)
            sigma_r_px = r_px_true * sigma_laser / (2 * E_true) if E_true > 0 else 1.0
            fwhm_true = sigma_r_px * 2.355
        else:
            r_px_true = 0
            fwhm_true = 0
        
        # Row 1: Peak Position (Energy)
        E_phys = phys_match.get('energy_eV', 0) if phys_match else 0
        E_rb = rb_match.get('energy_eV', 0) if rb_match else 0
        err_E_phys = abs(E_phys - E_true) if phys_match else float('nan')
        err_E_rb = abs(E_rb - E_true) if rb_match else float('nan')
        
        print(f"{i+1:<6} | {'Position (eV)':<15} | {E_true:<12.3f} | {E_phys:<12.3f} | {err_E_phys:<10.3f} | "
              f"{E_rb:<12.3f} | {err_E_rb:<10.3f}")
        
        # Row 2: Beta (Anisotropy)
        beta_phys = phys_match['beta'] if phys_match else 0
        beta_rb = rb_match['beta'] if rb_match else 0
        err_beta_phys = abs(beta_phys - beta_true) if phys_match else float('nan')
        err_beta_rb = abs(beta_rb - beta_true) if rb_match else float('nan')
        
        print(f"{'':6} | {'Beta':<15} | {beta_true:<12.2f} | {beta_phys:<12.2f} | {err_beta_phys:<10.2f} | "
              f"{beta_rb:<12.2f} | {err_beta_rb:<10.2f}")
        
        # Row 3: FWHM (Peak Width)
        fwhm_phys = phys_match.get('fwhm', phys_match.get('sigma', 0) * 2.355) if phys_match else 0
        fwhm_rb = rb_match.get('fwhm', rb_match.get('sigma', 0) * 2.355) if rb_match else 0
        err_fwhm_phys = abs(fwhm_phys - fwhm_true) if phys_match else float('nan')
        err_fwhm_rb = abs(fwhm_rb - fwhm_true) if rb_match else float('nan')
        
        print(f"{'':6} | {'FWHM (px)':<15} | {fwhm_true:<12.2f} | {fwhm_phys:<12.2f} | {err_fwhm_phys:<10.2f} | "
              f"{fwhm_rb:<12.2f} | {err_fwhm_rb:<10.2f}")
        
        # Row 4: Relative Amplitude (Branching Ratio)
        br_phys = phys_match.get('branching_ratio', 0) if phys_match else 0
        br_rb = rb_match.get('branching_ratio', 0) if rb_match else 0
        err_br_phys = abs(br_phys - br_true) if phys_match else float('nan')
        err_br_rb = abs(br_rb - br_true) if rb_match else float('nan')
        
        print(f"{'':6} | {'Rel. Amplitude':<15} | {br_true:<12.3f} | {br_phys:<12.3f} | {err_br_phys:<10.3f} | "
              f"{br_rb:<12.3f} | {err_br_rb:<10.3f}")
        
        print("-"*135)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  True peaks: {len(true_E)}")
    print(f"  PhysicsBasedFitter detected: {len(recon_params)}")
    print(f"  rBasex detected: {len(rbasex_params) if rbasex_params else 0}")
    
    # Calculate average errors
    if recon_params and rbasex_params:
        phys_errors = {'position': [], 'beta': [], 'fwhm': [], 'rel_amp': []}
        rb_errors = {'position': [], 'beta': [], 'fwhm': [], 'rel_amp': []}
        
        for i, (E_true, beta_true, br_true) in enumerate(zip(true_E, true_beta, true_br)):
            # Calculate true FWHM for this peak
            if config is not None:
                r_mm_true = config.get_expected_radius(E_true)
                r_px_true = r_mm_true / config.pixel_size
                sigma_r_px = r_px_true * sigma_laser / (2 * E_true) if E_true > 0 else 1.0
                fwhm_true = sigma_r_px * 2.355
            else:
                fwhm_true = 0
            
            if recon_sorted:
                phys_match = min(recon_sorted, 
                               key=lambda x: abs(x.get('energy_eV', 0) - E_true),
                               default=None)
                if phys_match:
                    phys_errors['position'].append(abs(phys_match.get('energy_eV', 0) - E_true))
                    phys_errors['beta'].append(abs(phys_match['beta'] - beta_true))
                    phys_errors['fwhm'].append(abs(phys_match.get('fwhm', phys_match.get('sigma', 0) * 2.355) - fwhm_true))
                    phys_errors['rel_amp'].append(abs(phys_match.get('branching_ratio', 0) - br_true))
            
            if rbasex_sorted:
                rb_match = min(rbasex_sorted,
                             key=lambda x: abs(x.get('energy_eV', 0) - E_true),
                             default=None)
                if rb_match:
                    rb_errors['position'].append(abs(rb_match.get('energy_eV', 0) - E_true))
                    rb_errors['beta'].append(abs(rb_match['beta'] - beta_true))
                    rb_errors['fwhm'].append(abs(rb_match.get('fwhm', rb_match.get('sigma', 0) * 2.355) - fwhm_true))
                    rb_errors['rel_amp'].append(abs(rb_match.get('branching_ratio', 0) - br_true))
        
        # Print average errors for all four parameters
        if any(phys_errors.values()) and any(rb_errors.values()):
            print(f"\nAverage Errors:")
            print(f"  {'Parameter':<15} | {'Physics':<12} | {'rBasex':<12} | {'Better':<8}")
            print(f"  {'-'*15} | {'-'*12} | {'-'*12} | {'-'*8}")
            
            # Position (Energy)
            phys_avg_pos = np.mean(phys_errors['position']) if phys_errors['position'] else float('nan')
            rb_avg_pos = np.mean(rb_errors['position']) if rb_errors['position'] else float('nan')
            better_pos = 'Physics' if phys_avg_pos < rb_avg_pos else 'rBasex' if not np.isnan(rb_avg_pos) else 'N/A'
            print(f"  {'Position (eV)':<15} | {phys_avg_pos:<12.3f} | {rb_avg_pos:<12.3f} | {better_pos:<8}")
            
            # Beta
            phys_avg_beta = np.mean(phys_errors['beta']) if phys_errors['beta'] else float('nan')
            rb_avg_beta = np.mean(rb_errors['beta']) if rb_errors['beta'] else float('nan')
            better_beta = 'Physics' if phys_avg_beta < rb_avg_beta else 'rBasex' if not np.isnan(rb_avg_beta) else 'N/A'
            print(f"  {'Beta':<15} | {phys_avg_beta:<12.3f} | {rb_avg_beta:<12.3f} | {better_beta:<8}")
            
            # FWHM
            phys_avg_fwhm = np.mean(phys_errors['fwhm']) if phys_errors['fwhm'] else float('nan')
            rb_avg_fwhm = np.mean(rb_errors['fwhm']) if rb_errors['fwhm'] else float('nan')
            better_fwhm = 'Physics' if phys_avg_fwhm < rb_avg_fwhm else 'rBasex' if not np.isnan(rb_avg_fwhm) else 'N/A'
            print(f"  {'FWHM (px)':<15} | {phys_avg_fwhm:<12.3f} | {rb_avg_fwhm:<12.3f} | {better_fwhm:<8}")
            
            # Relative Amplitude
            phys_avg_br = np.mean(phys_errors['rel_amp']) if phys_errors['rel_amp'] else float('nan')
            rb_avg_br = np.mean(rb_errors['rel_amp']) if rb_errors['rel_amp'] else float('nan')
            better_br = 'Physics' if phys_avg_br < rb_avg_br else 'rBasex' if not np.isnan(rb_avg_br) else 'N/A'
            print(f"  {'Rel. Amplitude':<15} | {phys_avg_br:<12.3f} | {rb_avg_br:<12.3f} | {better_br:<8}")
    
    print("="*120)


def visualize_beta_extraction(image: np.ndarray, params: List[Dict], 
                             save_path: Optional[str] = None) -> None:
    """
    可视化β提取过程，显示极坐标变换和角度分布分析。
    
    Args:
        image: 原始VMI图像
        params: 包含β调试信息的参数列表
        save_path: 保存路径
    """
    n_pixels = image.shape[0]
    cy, cx = n_pixels // 2, n_pixels // 2
    
    # 创建极坐标变换
    n_theta = 720
    n_r = n_pixels // 2
    
    theta_grid = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    r_grid = np.arange(n_r)
    
    theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
    x_cart = cx + r_mesh * np.cos(theta_mesh)
    y_cart = cy + r_mesh * np.sin(theta_mesh)
    
    polar_image = map_coordinates(image, [y_cart, x_cart], order=3, mode='constant', cval=0.0)
    
    # 创建图形
    n_peaks = len(params)
    fig, axes = plt.subplots(2, max(3, n_peaks), figsize=(4*max(3, n_peaks), 8))
    if n_peaks == 1:
        axes = axes.reshape(2, -1)
    
    # 第一行：原始图像和极坐标图像
    ax1 = axes[0, 0]
    im1 = ax1.imshow(image, cmap='hot')
    ax1.set_title("原始VMI图像")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 在原始图像上标记peak位置
    for i, p in enumerate(params):
        # 优化后的位置（实际用于β提取的位置）
        circle_opt = plt.Circle((cx, cy), p['r'], fill=False, color='cyan', linewidth=2)
        ax1.add_patch(circle_opt)
        ax1.text(cx + p['r'], cy, f"Peak {i+1}", color='cyan', fontweight='bold')
        
        # 如果有原始位置，也显示出来进行对比
        if 'r_original' in p and abs(p['r_original'] - p['r']) > 1:
            circle_orig = plt.Circle((cx, cy), p['r_original'], fill=False, color='yellow', 
                                   linewidth=1, linestyle='--', alpha=0.7)
            ax1.add_patch(circle_orig)
            ax1.text(cx + p['r_original'], cy - 10, f"原始", color='yellow', fontsize=8)
    
    ax2 = axes[0, 1]
    im2 = ax2.imshow(polar_image, cmap='hot', aspect='auto', 
                     extent=[0, 360, 0, n_r])
    ax2.set_title("极坐标变换 (r-θ)")
    ax2.set_xlabel("角度 (度)")
    ax2.set_ylabel("半径 (像素)")
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 在极坐标图像上标记peak位置
    for i, p in enumerate(params):
        # 优化后的位置
        ax2.axhline(y=p['r'], color='cyan', linewidth=2, alpha=0.7)
        ax2.text(10, p['r'], f"Peak {i+1}", color='cyan', fontweight='bold')
        
        # 原始位置（如果不同）
        if 'r_original' in p and abs(p['r_original'] - p['r']) > 1:
            ax2.axhline(y=p['r_original'], color='yellow', linewidth=1, 
                       linestyle='--', alpha=0.7)
            ax2.text(300, p['r_original'], f"原始{i+1}", color='yellow', fontsize=8)
    
    # 第三个子图：所有角度分布的比较
    ax3 = axes[0, 2] if axes.shape[1] > 2 else axes[0, 1]
    theta_degrees = np.degrees(theta_grid)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, p in enumerate(params):
        r_idx = int(p['r'])
        if r_idx < n_r:
            angular_profile = polar_image[r_idx, :]
            baseline = np.percentile(angular_profile, 10)
            profile_corrected = angular_profile - baseline
            profile_normalized = profile_corrected / np.max(profile_corrected) if np.max(profile_corrected) > 0 else profile_corrected
            
            color = colors[i % len(colors)]
            ax3.plot(theta_degrees, profile_normalized, color=color, linewidth=2, 
                    label=f'Peak {i+1} (r={p["r"]:.1f}, β={p["beta"]:.2f})')
    
    ax3.set_xlabel("角度 (度)")
    ax3.set_ylabel("归一化强度")
    ax3.set_title("角度分布比较")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 360)
    
    # 第二行：每个peak的详细分析
    for i, p in enumerate(params):
        if i >= axes.shape[1]:
            break
            
        ax = axes[1, i]
        
        # 提取角度分布
        r_idx = int(p['r'])
        if r_idx < n_r:
            angular_profile = polar_image[r_idx, :]
            baseline = np.percentile(angular_profile, 10)
            profile_corrected = angular_profile - baseline
            
            # 应用高斯平滑
            from scipy.ndimage import gaussian_filter1d
            profile_smooth = gaussian_filter1d(profile_corrected, sigma=1.0, mode='wrap')
            
            # 绘制角度分布
            ax.plot(theta_degrees, profile_smooth, 'b-', linewidth=2, label='实际分布')
            
            # 理论拟合曲线
            if 'beta_debug' in p:
                dc = p['beta_debug']['dc_component']
                beta = p['beta']
                theta_rad = theta_grid
                
                # VMI理论曲线: I(θ) = I₀[1 + β/2(3cos²θ - 1)]
                P2 = 0.5 * (3 * np.cos(theta_rad)**2 - 1)
                theory_curve = dc * (1 + beta * P2) - baseline
                theory_curve = np.maximum(theory_curve, 0)
                
                ax.plot(theta_degrees, theory_curve, 'r--', linewidth=2, 
                       label=f'理论拟合 (β={beta:.3f})')
            
            ax.set_xlabel("角度 (度)")
            ax.set_ylabel("强度")
            # 标题显示位置优化信息
            title = f"Peak {i+1} 角度分析\n"
            if 'r_original' in p and abs(p['r_original'] - p['r']) > 1:
                title += f"r: {p['r_original']:.1f}→{p['r']:.1f}px, β={p['beta']:.3f}"
            else:
                title += f"r={p['r']:.1f}px, β={p['beta']:.3f}"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 360)
            
            # 添加调试信息
            if 'beta_debug' in p:
                debug = p['beta_debug']
                info_text = f"方法: {debug['method_used']}\n"
                info_text += f"cos(θ)比率: {debug['cos_theta_ratio']:.4f}\n"
                info_text += f"cos(2θ)比率: {debug['cos_2theta_ratio']:.4f}\n"
                info_text += f"径向平均: {debug['radial_averaging_range']}\n"
                if 'r_original' in p:
                    info_text += f"位置优化: {debug['r_original']:.1f}→{debug['r_optimized']:.1f}\n"
                    info_text += f"优化后强度: {debug['intensity_at_optimized']:.2f}"
                
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 隐藏多余的子图
    for i in range(len(params), axes.shape[1]):
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_reconstruction(image: np.ndarray, params: List[Dict],
                            metadata: Dict, config=None,
                            true_params: Dict = None,
                            image_3d: np.ndarray = None,
                            rbasex_params: List[Dict] = None,
                            rbasex_metadata: Dict = None,
                            save_path: Optional[str] = None) -> None:
    """
    Visualize reconstruction results comparing with ground truth.
    
    Args:
        image: Original VMI image (2D projected image from detector)
        params: Reconstructed parameters (from PhysicsBasedFitter)
        metadata: Reconstruction metadata
        config: Optional Config for axis labels and energy conversion
        true_params: Dictionary with true simulation parameters
            Expected keys: E_centers, Betas, branching_ratios, sigma_laser
        image_3d: Deprecated parameter, no longer used.
        rbasex_params: Optional list of reconstructed parameters from rBasex method.
            If provided, rBasex results will be included in the beta comparison plot
            and used to generate the bottom-right panel reconstruction image.
        rbasex_metadata: Optional metadata from rBasex reconstruction.
            If provided, the rBasex radial profile will be included in the
            radial distribution comparison plot. If 'recon_image' key is present,
            it will be displayed in the bottom-right panel.
        save_path: Path to save figure
    """
    n_pixels = image.shape[0]
    r_grid = metadata['r_grid']
    recon_profile = metadata['recon_profile']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # =========================================================================
    # Row 1: Images
    # =========================================================================
    
    # Original projection image
    ax1 = axes[0, 0]
    vmax = np.percentile(image, 99.5)
    im1 = ax1.imshow(image, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title("Input VMI Image (Projection)")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # True 3D slice (if true_params provided)
    ax2 = axes[0, 1]
    if true_params is not None and config is not None:
        # Build true 3D distribution from true parameters
        true_peaks = []
        for E, beta, br in zip(true_params['E_centers'],
                               true_params['Betas'],
                               true_params['branching_ratios']):
            r_mm = config.get_expected_radius(E)
            r_px = r_mm / config.pixel_size
            # Estimate sigma from laser bandwidth
            # dE/E = 2 * dr/r for kinetic energy
            # sigma_r = r * sigma_E / (2 * E)
            sigma_E = true_params.get('sigma_laser', 0.01)
            sigma_r_px = r_px * sigma_E / (2 * E) if E > 0 else 1.0
            sigma_r_px = max(sigma_r_px, 0.5)  # Minimum sigma
            true_peaks.append({
                'r': r_px,
                'sigma': sigma_r_px,
                'amp': br,  # Use branching ratio as amplitude
                'beta': beta
            })
        true_2d = reconstruct_2d_from_params(true_peaks, n_pixels)
        im2 = ax2.imshow(true_2d, cmap='hot')
        ax2.set_title("True 3D Distribution (Slice)")
        plt.colorbar(im2, ax=ax2, fraction=0.046)
    else:
        ax2.text(0.5, 0.5, "No true parameters\nprovided", ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title("True 3D Distribution")
    
    # Reconstructed 3D slice
    ax3 = axes[0, 2]
    recon_2d = reconstruct_2d_from_params(params, n_pixels)
    im3 = ax3.imshow(recon_2d, cmap='hot')
    ax3.set_title("Reconstructed 3D Distribution (Slice)")
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # =========================================================================
    # Row 2: Profiles and comparison
    # =========================================================================
    
    # Radial profile comparison (3D space)
    ax4 = axes[1, 0]
    
    # Reconstructed profile (already in 3D space)
    if np.max(recon_profile) > 0:
        recon_profile_norm = recon_profile / np.max(recon_profile)
    else:
        recon_profile_norm = recon_profile
    
    # True profile (3D space)
    if true_params is not None and config is not None:
        true_profile = np.zeros_like(r_grid)
        for E, beta, br in zip(true_params['E_centers'],
                               true_params['Betas'],
                               true_params['branching_ratios']):
            r_mm = config.get_expected_radius(E)
            r_px = r_mm / config.pixel_size
            sigma_E = true_params.get('sigma_laser', 0.01)
            sigma_r_px = r_px * sigma_E / (2 * E) if E > 0 else 1.0
            sigma_r_px = max(sigma_r_px, 0.5)
            true_profile += br * np.exp(-((r_grid - r_px)**2) / (2 * sigma_r_px**2))
        
        if np.max(true_profile) > 0:
            true_profile_norm = true_profile / np.max(true_profile)
        else:
            true_profile_norm = true_profile
        
        ax4.plot(r_grid, true_profile_norm, 'k-', linewidth=2.5, alpha=0.7, label='True (3D)')
    
    ax4.plot(r_grid, recon_profile_norm, 'r--', linewidth=2, label='PhysicsBasedFitter')
    
    # rBasex profile (if provided)
    if rbasex_metadata is not None and 'intensity_profile' in rbasex_metadata:
        r_rb = rbasex_metadata['r_grid']
        I_rb = rbasex_metadata['intensity_profile']
        if np.max(I_rb) > 0:
            I_rb_norm = I_rb / np.max(I_rb)
        else:
            I_rb_norm = I_rb
        ax4.plot(r_rb, I_rb_norm, 'b-.', linewidth=2, label='rBasex')
    
    ax4.set_xlabel("Radius (pixels)")
    ax4.set_ylabel("Normalized Intensity")
    ax4.set_title("Radial Distribution Comparison (3D Space)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, n_pixels // 2)
    
    # Beta comparison
    ax5 = axes[1, 1]
    if true_params is not None and config is not None:
        # True beta values
        true_r = []
        true_beta = []
        for E, beta in zip(true_params['E_centers'], true_params['Betas']):
            r_mm = config.get_expected_radius(E)
            r_px = r_mm / config.pixel_size
            true_r.append(r_px)
            true_beta.append(beta)
        ax5.scatter(true_r, true_beta, s=200, c='black', marker='o',
                   label='True', zorder=10, edgecolors='white', linewidths=2)
    
    # rBasex beta values (if provided)
    if rbasex_params is not None:
        rb_r = [p['r'] for p in rbasex_params]
        rb_beta = [p['beta'] for p in rbasex_params]
        ax5.scatter(rb_r, rb_beta, s=120, c='blue', marker='s',
                   label='rBasex', zorder=6, edgecolors='white', linewidths=1)
    
    # PhysicsBasedFitter beta values
    recon_r = [p['r'] for p in params]
    recon_beta = [p['beta'] for p in params]
    ax5.scatter(recon_r, recon_beta, s=100, c='red', marker='^',
               label='PhysicsBasedFitter', zorder=5, edgecolors='white', linewidths=1)
    
    ax5.set_xlabel("Radius (pixels)")
    ax5.set_ylabel("Beta (β)")
    ax5.set_title("Anisotropy Parameter Comparison")
    ax5.set_ylim(-1.5, 2.5)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Bottom-right panel: rBasex reconstruction result
    ax6 = axes[1, 2]
    
    if rbasex_metadata is not None and 'recon_image' in rbasex_metadata:
        # Show the rBasex reconstructed 3D distribution slice
        rbasex_recon = rbasex_metadata['recon_image']
        im6 = ax6.imshow(rbasex_recon, cmap='hot')
        ax6.set_title("rBasex Reconstructed 3D Distribution")
        ax6.set_xlabel('X (pixels)')
        ax6.set_ylabel('Y (pixels)')
        plt.colorbar(im6, ax=ax6, fraction=0.046, label='Intensity')
    elif rbasex_params is not None:
        # Reconstruct 2D slice from rBasex parameters
        rbasex_2d = reconstruct_2d_from_params(rbasex_params, n_pixels)
        im6 = ax6.imshow(rbasex_2d, cmap='hot')
        ax6.set_title("rBasex Reconstructed 3D Distribution")
        ax6.set_xlabel('X (pixels)')
        ax6.set_ylabel('Y (pixels)')
        plt.colorbar(im6, ax=ax6, fraction=0.046, label='Intensity')
    else:
        # Fall back to parameter summary table if no rBasex data provided
        ax6.axis('off')
        
        summary = "PARAMETER COMPARISON\n"
        summary += "=" * 40 + "\n\n"
        
        if true_params is not None and config is not None:
            summary += f"{'Peak':<6} {'True':<20} {'Recon':<20}\n"
            summary += "-" * 46 + "\n"
            
            # Sort reconstructed by radius
            recon_sorted = sorted(params, key=lambda x: x['r'])
            
            for i, (E, beta, br) in enumerate(zip(true_params['E_centers'],
                                                   true_params['Betas'],
                                                   true_params['branching_ratios'])):
                r_mm = config.get_expected_radius(E)
                r_px = r_mm / config.pixel_size
                
                # Find matching reconstructed peak
                if i < len(recon_sorted):
                    p = recon_sorted[i]
                    summary += f"\nPeak {i+1}:\n"
                    summary += f"  r:    {r_px:>6.1f} px    {p['r']:>6.1f} px\n"
                    summary += f"  E:    {E:>6.3f} eV    {p.get('energy_eV', 0):>6.3f} eV\n"
                    summary += f"  β:    {beta:>6.2f}       {p['beta']:>6.2f}\n"
                    summary += f"  BR:   {br:>6.3f}       {p.get('branching_ratio', 0):>6.3f}\n"
        else:
            summary += "Reconstructed Parameters:\n\n"
            for i, p in enumerate(params):
                summary += f"Peak {i+1}:\n"
                summary += f"  r = {p['r']:.1f} px"
                if 'energy_eV' in p:
                    summary += f" ({p['energy_eV']:.3f} eV)"
                summary += f"\n  σ = {p['sigma']:.2f} px\n"
                summary += f"  β = {p['beta']:.3f}\n"
                if 'branching_ratio' in p:
                    summary += f"  BR = {p['branching_ratio']:.3f}\n"
                summary += "\n"
        
        ax6.text(0.05, 0.95, summary, fontsize=9, va='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                 transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# Test function
# =============================================================================
if __name__ == "__main__":
    # Test with a simple synthetic image
    print("Testing Abel Backward Reconstruction...")
    
    # Create a simple test image with known parameters
    n = 256
    y, x = np.ogrid[:n, :n]
    center = n // 2
    y, x = y - center, x - center
    r = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    # Create test peaks
    test_peaks = [
        {'r': 40, 'sigma': 3, 'amp': 1.0, 'beta': 1.5},
        {'r': 80, 'sigma': 5, 'amp': 0.7, 'beta': -0.5},
    ]
    
    img_3d = np.zeros_like(r, dtype=float)
    for p in test_peaks:
        radial = p['amp'] * np.exp(-((r - p['r'])**2) / (2 * p['sigma']**2))
        angular = 1 + p['beta'] * P2
        img_3d += radial * angular
    
    # Forward project
    img_proj = abel.Transform(img_3d, method='hansenlaw', direction='forward', verbose=False).transform
    
    # =========================================================================
    # Add realistic noise: Poisson + Gaussian
    # =========================================================================
    print("\nAdding realistic noise (Poisson + Gaussian)...")
    
    # Scale image to realistic photon counts
    # 高光子测试: 500000 counts (更高SNR)
    total_counts = 500000  # Total photon counts - HIGH PHOTON TEST
    img_scaled = img_proj / np.sum(img_proj) * total_counts
    
    # 1. Poisson noise (shot noise) - proportional to sqrt(signal)
    # For each pixel, sample from Poisson distribution
    img_poisson = np.random.poisson(np.maximum(img_scaled, 0).astype(float))
    
    # 2. Gaussian noise (readout noise) - constant background
    readout_noise_sigma = 3.0  # Typical CCD readout noise in counts
    img_gaussian_noise = np.random.normal(0, readout_noise_sigma, img_proj.shape)
    
    # Combined noisy image
    img_noisy = img_poisson + img_gaussian_noise
    img_noisy = np.maximum(img_noisy, 0)  # No negative counts
    
    # Calculate actual SNR
    signal_max = np.max(img_scaled)
    noise_poisson = np.sqrt(signal_max)
    noise_total = np.sqrt(noise_poisson**2 + readout_noise_sigma**2)
    snr_actual = signal_max / noise_total
    print(f"  Total counts: {total_counts}")
    print(f"  Peak signal: {signal_max:.1f} counts")
    print(f"  Poisson noise (at peak): {noise_poisson:.1f}")
    print(f"  Gaussian noise: {readout_noise_sigma:.1f}")
    print(f"  Estimated SNR at peak: {snr_actual:.1f}")
    
    # Reconstruct
    params, metadata = reconstruct_vmi_image(img_noisy, verbose=True)
    
    # Visualize beta extraction process
    print("\n显示β提取过程...")
    visualize_beta_extraction(img_noisy, params)
    
    # Visualize overall reconstruction
    visualize_reconstruction(img_noisy, params, metadata)
    
    print("\n" + "="*70)
    print("COMPARISON: True vs Reconstructed Parameters")
    print("="*70)
    print(f"{'Peak':<6} {'Param':<8} {'True':<12} {'Recon':<12} {'Error':<12} {'Uncertainty':<12}")
    print("-"*70)
    
    # Sort reconstructed params by radius
    params_sorted = sorted(params, key=lambda x: x['r'])
    
    for i, true_p in enumerate(test_peaks):
        if i < len(params_sorted):
            recon_p = params_sorted[i]
            
            # Position
            r_err = abs(recon_p['r'] - true_p['r'])
            print(f"{i+1:<6} {'r (px)':<8} {true_p['r']:<12.1f} {recon_p['r']:<12.1f} {r_err:<12.1f} {'-':<12}")
            
            # Sigma
            sigma_err = abs(recon_p['sigma'] - true_p['sigma'])
            print(f"{'':6} {'sigma':<8} {true_p['sigma']:<12.1f} {recon_p['sigma']:<12.2f} {sigma_err:<12.2f} {'-':<12}")
            
            # Beta with uncertainty
            beta_err = abs(recon_p['beta'] - true_p['beta'])
            beta_unc = recon_p.get('beta_uncertainty', 0)
            print(f"{'':6} {'beta':<8} {true_p['beta']:<12.2f} {recon_p['beta']:<12.3f} {beta_err:<12.3f} {beta_unc:<12.3f}")
            
            print("-"*70)
    
    print("\nNoise Analysis Summary:")
    for i, p in enumerate(params_sorted):
        if 'beta_debug' in p:
            debug = p['beta_debug']
            print(f"  Peak {i+1}:")
            print(f"    SNR overall: {debug.get('snr_overall', 0):.1f}")
            print(f"    SNR cos(2θ): {debug.get('snr_cos2theta', 0):.1f}")
            print(f"    Gaussian noise est: {debug.get('sigma_gaussian_est', 0):.2f}")
            print(f"    Poisson noise est: {debug.get('sigma_poisson_est', 0):.2f}")
            print(f"    Smoothing sigma: {debug.get('smooth_sigma_used', 0):.1f}")
