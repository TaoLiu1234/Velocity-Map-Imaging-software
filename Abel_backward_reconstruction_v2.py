"""
Abel Backward Reconstruction V2 - Complete Physical Model

物理模型（Forward卷积链）：
==========================
3D分布 → Abel投影 → PSF卷积 → 像素化 → (x,y)图像 → 插值 → (r,θ)矩阵

每一步的展宽：
1. 激光带宽：σ_laser(r) = C_laser/r（3D空间，投影后保持）
2. PSF：σ_psf（固定，像素单位）
3. 像素化：σ_pixel（固定，约0.4-0.5像素）
4. xy→rθ插值：σ_interp（固定，约0.3-0.5像素）

综合展宽（投影面上）：
σ_total(r) = sqrt((C_laser/r)² + σ_psf² + σ_pixel² + σ_interp²)

Phase 0: 预处理（滤波）
Phase 1: 径向分析（反卷积后估计）
Phase 2: 角向分析（反卷积后估计）
Phase 3: 前向模型优化（完整卷积链）
"""

import numpy as np
import abel
import time
from scipy.optimize import least_squares, curve_fit
from scipy.signal import find_peaks, peak_widths, stft, istft
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
from typing import List, Dict, Tuple, Optional


class PhysicsBasedFitterV2:
    """Physics-based fitter V2 with complete physical model."""
    
    def __init__(self, n_pixels: int,
                 C_laser: float = None,
                 sigma_psf: float = 0.0,
                 sigma_pixel: float = 0.4,
                 sigma_interp: float = 0.55):
        """
        Args:
            n_pixels: 图像尺寸
            C_laser: 激光常数，σ_laser = C_laser/r
            sigma_psf: PSF的sigma（像素单位）
            sigma_pixel: 像素化的sigma（像素单位，~0.4）
            sigma_interp: xy→rθ插值的sigma（像素单位，3阶样条~0.55）
        """
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid_1d = np.arange(self.radius + 1, dtype=float)

        # 物理参数
        self.sigma_E = None  # 激光带宽 (eV)，固定值
        self.vmi_k = None    # VMI系数 k = E/r²
        self.sigma_psf = sigma_psf
        self.sigma_pixel = sigma_pixel
        self.sigma_interp = sigma_interp  # 3阶样条插值展宽 ~0.55像素
        
        # 坐标网格
        y, x = np.ogrid[:n_pixels, :n_pixels]
        self.Y = y - n_pixels // 2
        self.X = x - n_pixels // 2
        self.R2 = self.X**2 + self.Y**2
        self.R = np.sqrt(self.R2)
        
        self.r_indices = self.R.astype(int)
        self.max_r_idx = int(np.max(self.R))
        self.r_flat = self.r_indices.ravel()
        
        self.pixel_counts = np.bincount(self.r_flat, minlength=self.max_r_idx + 1)
        self.pixel_counts[self.pixel_counts == 0] = 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.COS_THETA = self.X / self.R
        self.COS_THETA[~np.isfinite(self.COS_THETA)] = 0.0
        self.P2_GRID = 0.5 * (3 * self.COS_THETA**2 - 1)
        
        # Phase 0 输出
        self._polar = None
        self._theta_grid = None
        self._noise_params = None

    # =========================================================================
    # 物理模型：展宽计算
    # =========================================================================
    
    def get_sigma_detector(self) -> float:
        """探测器空间的总展宽（PSF + 像素化 + 插值）。"""
        return np.sqrt(self.sigma_psf**2 + self.sigma_pixel**2 + self.sigma_interp**2)
    
    def get_sigma_laser_at_r(self, r: float) -> float:
        """
        半径r处激光带宽导致的径向展宽（像素单位）。
        
        物理推导：
        - E = k × r² (VMI关系)
        - σ_r = |dr/dE| × σ_E = (1/2) × (r/E) × σ_E = r × σ_E / (2E)
        - 代入 E = k×r²: σ_r = r × σ_E / (2×k×r²) = σ_E / (2×k×r)
        
        注意：σ_E是固定的激光带宽，投影操作让其影响随r变化。
        """
        if r < 5 or self.sigma_E is None or self.vmi_k is None:
            return 0.0
        
        # σ_r = σ_E / (2 × k × r)，其中 k = E/r² (eV/px²)
        # 但这里r是像素单位，需要转换
        # 实际上：σ_r(px) = r(px) × σ_E(eV) / (2 × E(eV))
        # E = k × r²，所以 σ_r = r × σ_E / (2 × k × r²) = σ_E / (2 × k × r)
        return self.sigma_E / (2 * self.vmi_k * r)
    
    def get_sigma_radial(self, r: float) -> float:
        """半径r处的径向总展宽（投影面上）。"""
        if r < 5:
            return 10.0
        
        sigma_laser = self.get_sigma_laser_at_r(r)
        sigma_det = self.get_sigma_detector()
        return np.sqrt(sigma_laser**2 + sigma_det**2)
    
    def get_sigma_angular(self, r: float) -> float:
        """半径r处的角向展宽（弧度）。"""
        if r < 5:
            return 1.0
        sigma_det = self.get_sigma_detector()
        return sigma_det / r
    
    def get_f_max_radial(self, r: float) -> float:
        """半径r处的径向最大频率。"""
        sigma = self.get_sigma_radial(r)
        return 1.0 / (2 * np.pi * sigma)
    
    def get_k_max_angular(self, r: float) -> float:
        """半径r处的角向最大频率（k值）。"""
        sigma_theta = self.get_sigma_angular(r)
        if sigma_theta < 0.01:
            return 100
        return 1.0 / (2 * np.pi * sigma_theta)
    
    def calibrate_from_config(self, config) -> None:
        """从config校准参数。"""
        if config is None:
            self.sigma_E = 0.015  # 默认激光带宽 (eV)
            self.vmi_k = 5e-5    # 默认VMI系数
            return
        
        # 激光带宽 (eV) - 固定值
        self.sigma_E = config.sigma_laser if hasattr(config, 'sigma_laser') else 0.015
        
        # VMI系数 k = E/r² (eV/px²)
        E_ref = max(config.E_centers) if hasattr(config, 'E_centers') else 1.0
        r_ref_mm = config.get_expected_radius(E_ref)
        r_ref_px = r_ref_mm / config.pixel_size
        self.vmi_k = E_ref / (r_ref_px ** 2)
        
        # PSF校准
        if hasattr(config, 'psf_fwhm') and config.psf_fwhm > 0:
            self.sigma_psf = (config.psf_fwhm / 2.355) / config.pixel_size
        else:
            self.sigma_psf = 0.0

    # =========================================================================
    # Phase 0: 预处理
    # =========================================================================
    
    def _deconvolve_2d_wiener(self, image: np.ndarray, sigma: float, nsr: float = 0.01) -> np.ndarray:
        """
        2D Wiener反卷积：在笛卡尔坐标下去除PSF展宽。
        
        物理上正确的做法：PSF是在(x,y)空间发生的，所以应该在(x,y)空间去卷积。
        
        Args:
            image: 输入图像 (x,y)
            sigma: PSF的sigma（像素单位）
            nsr: 噪声信号比，用于Wiener滤波正则化
        
        Returns:
            反卷积后的图像
        """
        if sigma < 0.1:
            return image
        
        ny, nx = image.shape
        
        # 频域
        fft_image = np.fft.fft2(image)
        
        # 2D高斯核的频域表示
        fy = np.fft.fftfreq(ny)
        fx = np.fft.fftfreq(nx)
        FY, FX = np.meshgrid(fy, fx, indexing='ij')
        freq_sq = FX**2 + FY**2
        
        # H(f) = exp(-2π²σ²f²)
        gauss_fft = np.exp(-2 * (np.pi * sigma)**2 * freq_sq)
        
        # Wiener滤波: H_wiener = H* / (|H|² + NSR)
        wiener = gauss_fft / (gauss_fft**2 + nsr)
        
        # 反卷积
        fft_deconv = fft_image * wiener
        deconv = np.fft.ifft2(fft_deconv).real
        
        return np.maximum(deconv, 0)
    
    def _phase0_preprocess(self, image_2d: np.ndarray, n_theta: int = 720) -> None:
        """Phase 0: 预处理
        
        物理正确的流程：
        1. 在笛卡尔坐标(x,y)下去除PSF展宽（因为PSF是在探测器平面发生的）
        2. 然后转换到极坐标(r,θ)
        3. 后续分析不再需要考虑PSF展宽
        """
        print("Phase 0: Preprocessing")
        print("=" * 60)
        print(f"  Physical params: σ_E={self.sigma_E:.4f} eV, vmi_k={self.vmi_k:.2e}")
        print(f"                   σ_psf={self.sigma_psf:.2f}, σ_pixel={self.sigma_pixel:.2f}, σ_interp={self.sigma_interp:.2f}")
        sigma_det = self.get_sigma_detector()
        print(f"                   σ_det={sigma_det:.2f} px (total detector broadening)")
        
        cy, cx = self.n // 2, self.n // 2
        n_r = self.radius
        
        # Step 1: 在笛卡尔坐标下去除探测器展宽（PSF + 像素化）
        # 注意：不去除插值展宽，因为那是后面极坐标变换引入的
        sigma_cart = np.sqrt(self.sigma_psf**2 + self.sigma_pixel**2)
        if sigma_cart > 0.1:
            print(f"  [Step 1] 2D Wiener deconvolution in (x,y) space (σ={sigma_cart:.2f} px)")
            # 根据图像质量调整NSR
            image_snr = np.max(image_2d) / (np.std(image_2d[image_2d < np.percentile(image_2d, 10)]) + 1e-6)
            nsr = 0.001 if image_snr > 100 else (0.01 if image_snr > 30 else 0.05)
            print(f"    Image SNR estimate: {image_snr:.1f}, using NSR={nsr}")
            image_deconv = self._deconvolve_2d_wiener(image_2d, sigma_cart, nsr=nsr)
        else:
            print("  [Step 1] Skipping 2D deconvolution (σ_cart too small)")
            image_deconv = image_2d
        
        # Step 2: xy → rθ (引入σ_interp展宽，但这个很小)
        print("  [Step 2] Image (x,y) -> polar (r,θ)")
        theta_grid = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        r_grid = np.arange(n_r)
        theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
        x_cart = cx + r_mesh * np.cos(theta_mesh)
        y_cart = cy + r_mesh * np.sin(theta_mesh)
        
        polar_raw = map_coordinates(image_deconv, [y_cart, x_cart], order=3, mode='constant', cval=0.0)
        self._theta_grid = theta_grid
        
        # Step 3: 噪声估计 + 背景去除
        print("  [Step 3] Noise estimation + background subtraction")
        self._noise_params = self._estimate_noise(polar_raw)
        baseline = self._noise_params['baseline']
        polar = np.maximum(polar_raw - baseline, 0)
        snr = self._noise_params['snr']
        print(f"    Baseline: {baseline:.2f}, SNR: {snr:.1f}")
        
        # Step 4: 99%分位数归一化
        print("  [Step 4] Intensity normalization (99th percentile)")
        norm_factor = np.percentile(polar, 99)
        if norm_factor > 0:
            polar_norm = polar / norm_factor
        else:
            polar_norm = polar
            norm_factor = 1.0
        print(f"    Norm factor: {norm_factor:.2f}")
        
        # Step 5: 角向滤波（在归一化后的图像上）
        print("  [Step 5] Angular filter (DC + k=2)")
        polar_norm = self._filter_angular_simple(polar_norm)
        
        # Step 6: 径向滤波（基于物理模型）- 现在只需要考虑插值展宽
        if snr > 50:
            print("  [Step 6] Skipping radial filter (very high SNR)")
        elif snr > 20:
            print("  [Step 6] Light radial filter (PSF already removed)")
            polar_norm = self._filter_radial_conservative(polar_norm)
        elif snr > 10:
            print("  [Step 6] Conservative radial filter (medium SNR)")
            polar_norm = self._filter_radial_conservative(polar_norm)
        else:
            print("  [Step 6] Adaptive radial filter (low SNR)")
            polar_norm = self._filter_radial_adaptive(polar_norm)
        
        # Step 7: 恢复强度
        self._polar = polar_norm * norm_factor
        self._norm_factor = norm_factor
        
        print("  Phase 0 complete")
        print("=" * 60)
    
    def _filter_angular_simple(self, polar: np.ndarray) -> np.ndarray:
        """
        角向滤波：只保留DC和k=2成分。
        
        简化版本：不做高斯加权，直接保留k=0和k=2。
        这样可以保证beta估计不受滤波影响。
        """
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        
        for r_idx in range(n_r):
            if r_idx < 10:
                result[r_idx, :] = polar[r_idx, :]
                continue
            
            angular = polar[r_idx, :]
            original_mean = np.mean(angular)
            if original_mean < 1e-10:
                result[r_idx, :] = angular
                continue
            
            fft = np.fft.fft(angular)
            n = len(fft)
            
            # 只保留DC (k=0) 和 k=2
            fft_filtered = np.zeros_like(fft)
            fft_filtered[0] = fft[0]  # DC
            fft_filtered[2] = fft[2]  # k=2 正频率
            fft_filtered[-2] = fft[-2]  # k=2 负频率
            
            filtered = np.fft.ifft(fft_filtered).real
            
            # 保持均值
            filtered_mean = np.mean(filtered)
            if filtered_mean > 1e-10:
                filtered = filtered * (original_mean / filtered_mean)
            
            result[r_idx, :] = np.maximum(filtered, 0)
        
        return result
    
    def _estimate_noise(self, polar: np.ndarray) -> Dict:
        """
        噪声估计。
        
        在无信号区域（外围）估计噪声参数。
        baseline用25%分位数（经验上效果最好，对残余信号稳健）。
        """
        n_r = polar.shape[0]
        # 外围区域（85%-100%）应该主要是噪声
        noise_region = polar[int(n_r * 0.85):, :].ravel()
        
        # baseline用25%分位数（比中位数更保守，避免残余信号影响）
        baseline = np.percentile(noise_region, 25)
        
        # 标准差用MAD估计
        noise_median = np.median(noise_region)
        noise_mad = np.median(np.abs(noise_region - noise_median))
        readout_std = 1.4826 * noise_mad
        
        # SNR估计
        signal_max = np.max(np.mean(polar, axis=1))
        snr = signal_max / (readout_std + 1e-6)
        
        return {
            'readout_std': readout_std,
            'baseline': baseline,
            'snr': snr
        }
    
    def _filter_angular(self, polar: np.ndarray) -> np.ndarray:
        """
        角向带通滤波：保留k=0和k=2附近。
        带宽由探测器展宽决定：Δk(r) = r / (2π × σ_det)
        """
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        sigma_det = self.get_sigma_detector()
        
        for r_idx in range(n_r):
            if r_idx < 10:
                result[r_idx, :] = polar[r_idx, :]
                continue
            
            angular = polar[r_idx, :]
            original_mean = np.mean(angular)
            if original_mean < 1e-10:
                result[r_idx, :] = angular
                continue
            
            fft = np.fft.fft(angular)
            n = len(fft)
            
            # 角向频率展宽
            delta_k = r_idx / (2 * np.pi * sigma_det) if sigma_det > 0.1 else 50
            
            fft_filtered = np.zeros_like(fft)
            fft_filtered[0] = fft[0]  # DC
            
            # k=2附近的带通
            k_width = max(1, int(delta_k * 0.5))
            for k in range(max(1, 2-k_width), min(n//2, 2+k_width+1)):
                weight = np.exp(-0.5 * ((k - 2) / max(delta_k * 0.3, 0.5))**2)
                fft_filtered[k] = fft[k] * weight
                if k > 0 and k < n//2:
                    fft_filtered[-k] = fft[-k] * weight
            
            filtered = np.fft.ifft(fft_filtered).real
            
            # 保持均值
            filtered_mean = np.mean(filtered)
            if filtered_mean > 1e-10:
                filtered = filtered * (original_mean / filtered_mean)
            
            result[r_idx, :] = np.maximum(filtered, 0)
        
        return result
    
    def _filter_radial_physical(self, polar: np.ndarray) -> np.ndarray:
        """
        基于物理模型的径向滤波。
        
        关键：截止频率由展宽模型决定，但要考虑强度的影响。
        高强度区域的高频成分可能是真实信号，不应该滤掉。
        
        策略：
        1. 计算每个r处的物理截止频率 f_cutoff(r)
        2. 对每个角度的profile做局部归一化
        3. 应用低通滤波
        4. 恢复强度
        """
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        
        # 计算径向profile用于归一化
        radial_profile = np.mean(polar, axis=1)
        
        for theta_idx in range(n_theta):
            profile = polar[:, theta_idx].copy()
            
            # 局部归一化：除以平滑的包络
            envelope = gaussian_filter1d(profile, sigma=10)
            envelope = np.maximum(envelope, 1e-6)
            normalized = profile / envelope
            
            # 对归一化后的profile做滤波
            # 使用物理模型的截止频率
            filtered_norm = np.zeros_like(normalized)
            
            for r_idx in range(n_r):
                if r_idx < 10:
                    filtered_norm[r_idx] = normalized[r_idx]
                    continue
                
                # 物理截止频率对应的sigma
                sigma_total = self.get_sigma_radial(r_idx)
                # 滤波sigma（略大于物理sigma，留余量）
                sigma_filter = max(0.5, sigma_total * 0.7)
                
                # 局部加权平均
                r_range = int(3 * sigma_filter)
                r_start = max(0, r_idx - r_range)
                r_end = min(n_r, r_idx + r_range + 1)
                
                weights = np.exp(-((np.arange(r_start, r_end) - r_idx)**2) / (2 * sigma_filter**2))
                weights /= np.sum(weights)
                
                filtered_norm[r_idx] = np.sum(normalized[r_start:r_end] * weights)
            
            # 恢复强度
            result[:, theta_idx] = filtered_norm * envelope
        
        return np.maximum(result, 0)
    
    def _filter_radial_conservative(self, polar: np.ndarray) -> np.ndarray:
        """
        保守的径向滤波：只滤除非常高频的噪声。
        用于中等SNR情况，避免滤掉窄peak。
        """
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        
        for theta_idx in range(n_theta):
            profile = polar[:, theta_idx]
            
            # 只用很轻的高斯平滑（sigma=0.5像素）
            # 这只会滤掉非常高频的噪声，不会影响正常的peak
            smoothed = gaussian_filter1d(profile, sigma=0.5)
            result[:, theta_idx] = smoothed
        
        return np.maximum(result, 0)
    
    def _filter_radial_adaptive(self, polar: np.ndarray) -> np.ndarray:
        """
        自适应径向滤波：根据局部信号强度调整滤波强度。
        用于低SNR情况。
        
        策略：
        - 在peak附近（高信号区域）使用轻滤波
        - 在背景区域（低信号区域）使用强滤波
        """
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        
        # 计算径向profile来识别peak区域
        radial_profile = np.mean(polar, axis=1)
        max_signal = np.max(radial_profile)
        
        for theta_idx in range(n_theta):
            profile = polar[:, theta_idx]
            filtered = np.zeros_like(profile)
            
            for r_idx in range(n_r):
                # 根据局部信号强度决定滤波强度
                local_signal = radial_profile[r_idx] / (max_signal + 1e-6)
                
                if local_signal > 0.3:
                    # 高信号区域：轻滤波
                    sigma_filter = 0.5
                elif local_signal > 0.1:
                    # 中等信号区域：中等滤波
                    sigma_filter = 1.0
                else:
                    # 低信号区域：强滤波
                    sigma_filter = 2.0
                
                # 局部加权平均
                r_start = max(0, r_idx - int(3 * sigma_filter))
                r_end = min(n_r, r_idx + int(3 * sigma_filter) + 1)
                
                weights = np.exp(-((np.arange(r_start, r_end) - r_idx)**2) / (2 * sigma_filter**2))
                weights /= np.sum(weights)
                
                filtered[r_idx] = np.sum(profile[r_start:r_end] * weights)
            
            result[:, theta_idx] = filtered
        
        return np.maximum(result, 0)

    def _filter_radial(self, polar: np.ndarray) -> np.ndarray:
        """径向低通滤波：截止频率由σ_total(r)决定。"""
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        
        nperseg = min(64, n_r // 4)
        noverlap = nperseg * 3 // 4
        
        for theta_idx in range(n_theta):
            profile = polar[:, theta_idx]
            
            f, t, Zxx = stft(profile, nperseg=nperseg, noverlap=noverlap)
            
            if len(t) > 1:
                t_to_r = t * (n_r - 1) / t[-1]
            else:
                t_to_r = np.array([n_r // 2])
            
            n_freq, n_time = Zxx.shape
            mask = np.ones_like(Zxx, dtype=float)
            
            for t_idx in range(n_time):
                r = t_to_r[t_idx] if t_idx < len(t_to_r) else n_r // 2
                f_max = self.get_f_max_radial(r) * 1.5  # 1.5倍余量
                
                for f_idx in range(n_freq):
                    if f[f_idx] > f_max:
                        transition = 0.3 * f_max
                        if f[f_idx] > f_max + transition:
                            mask[f_idx, t_idx] = 0.0
                        else:
                            mask[f_idx, t_idx] = 1.0 - (f[f_idx] - f_max) / transition
            
            Zxx_filtered = Zxx * mask
            _, reconstructed = istft(Zxx_filtered, nperseg=nperseg, noverlap=noverlap)
            
            if len(reconstructed) >= n_r:
                result[:, theta_idx] = reconstructed[:n_r]
            else:
                result[:len(reconstructed), theta_idx] = reconstructed
        
        return np.maximum(result, 0)
    
    def _get_radial_profile(self) -> np.ndarray:
        """获取径向分布。"""
        profile = np.mean(self._polar, axis=1)
        if len(profile) < len(self.r_grid_1d):
            profile = np.pad(profile, (0, len(self.r_grid_1d) - len(profile)), mode='edge')
        return profile[:len(self.r_grid_1d)]
    
    def _get_abel_profile(self) -> np.ndarray:
        """获取Abel逆变换后的径向分布。"""
        proj = self._get_radial_profile()
        abel_profile = abel.hansenlaw.hansenlaw_transform(proj, direction='inverse')
        return np.maximum(abel_profile, 0)

    # =========================================================================
    # Phase 1: 径向分析（反卷积后估计）
    # =========================================================================
    
    def _deconvolve_sigma(self, sigma_measured: float, r: float) -> float:
        """
        反卷积：从测量的sigma中去除探测器展宽和Abel逆变换残余展宽。
        
        σ_true² = σ_measured² - σ_detector² - σ_abel_residual²
        
        Abel逆变换后仍有残余展宽，经验因子约为 σ_residual ≈ 0.7 * σ_true
        这意味着 σ_measured ≈ 1.4 * σ_true（当σ_detector很小时）
        """
        sigma_det = self.get_sigma_detector()
        
        # 先去除探测器展宽
        sigma_sq = sigma_measured**2 - sigma_det**2
        if sigma_sq <= 0:
            return sigma_measured * 0.5
        
        sigma_after_det = np.sqrt(sigma_sq)
        
        # Abel逆变换残余展宽校正
        # 经验发现：Abel逆变换后的sigma约为真值的1.4倍
        # 所以 σ_true ≈ σ_after_det / 1.4
        abel_correction_factor = 1.4
        sigma_true = sigma_after_det / abel_correction_factor
        
        return max(sigma_true, 0.3)
    
    def _phase1_find_peaks(self, profile: np.ndarray, mask_radius: int = 15) -> List[Dict]:
        """
        找peaks，平衡灵敏度和假阳性。
        
        策略：
        1. 使用投影面的profile（不是Abel逆变换后的）来检测peaks
        2. 根据SNR自适应调整阈值
        3. 高SNR时使用更严格的形状检查（避免噪声尖峰）
        4. 低SNR时使用更宽松的阈值（避免漏检）
        5. 最后根据相对强度过滤弱peaks
        """
        max_val = np.max(profile)
        snr = self._noise_params.get('snr', 10) if self._noise_params else 10
        
        # 根据SNR调整检测参数
        if snr > 100:
            # 非常高SNR：peak很窄，需要更小的distance
            height_thresh = max_val * 0.03
            distance = 5
            prominence_thresh = max_val * 0.02
        elif snr > 30:
            # 高SNR
            height_thresh = max_val * 0.04
            distance = 6
            prominence_thresh = max_val * 0.025
        elif snr > 10:
            # 中等SNR
            height_thresh = max_val * 0.05
            distance = 8
            prominence_thresh = max_val * 0.03
        else:
            # 低SNR：更严格的阈值避免假阳性
            height_thresh = max_val * 0.10
            distance = 12
            prominence_thresh = max_val * 0.08
        
        # 基本peak检测
        peaks, properties = find_peaks(
            profile, 
            height=height_thresh,
            distance=distance,
            prominence=prominence_thresh,
        )
        
        # 第一轮过滤：基本质量检查
        candidate_peaks = []
        for pk in peaks:
            if pk < mask_radius:
                continue
            
            # 计算局部SNR
            local_region = profile[max(0, pk-15):min(len(profile), pk+16)]
            local_max = profile[pk]
            local_baseline = np.percentile(local_region, 25)
            local_std = np.std(local_region)
            local_snr = (local_max - local_baseline) / (local_std + 1e-6)
            
            # SNR阈值（低全局SNR时更严格）
            snr_threshold = 2.0 if snr > 30 else 3.0
            if local_snr < snr_threshold:
                continue
            
            # 形状检查
            check_range = 3 if snr > 50 else 5
            left_idx = max(mask_radius, pk - check_range)
            right_idx = min(len(profile) - 1, pk + check_range)
            left_val = profile[left_idx]
            right_val = profile[right_idx]
            center_val = profile[pk]
            
            shape_ratio = (left_val + right_val) / (2 * center_val + 1e-6)
            min_shape_ratio = 0.1 if snr > 50 else 0.15
            if shape_ratio < min_shape_ratio and local_snr < 10:
                continue
            
            candidate_peaks.append({
                'r': int(pk), 
                'snr': local_snr,
                'height': local_max - local_baseline,
                'prominence': properties['prominences'][list(peaks).index(pk)] if 'prominences' in properties else local_max
            })
        
        # 第二轮过滤：相对强度过滤
        # 只保留相对强度 > 3% 的peaks（降低阈值以保留弱peaks）
        if len(candidate_peaks) > 0:
            max_height = max(p['height'] for p in candidate_peaks)
            valid_peaks = [p for p in candidate_peaks if p['height'] > max_height * 0.03]
        else:
            valid_peaks = []
        
        # 第三轮过滤：合并太近的peaks
        # 如果两个peaks距离 < 10像素，保留更强的那个
        if len(valid_peaks) > 1:
            valid_peaks = sorted(valid_peaks, key=lambda x: x['r'])
            merged_peaks = [valid_peaks[0]]
            for p in valid_peaks[1:]:
                if p['r'] - merged_peaks[-1]['r'] < 10:
                    # 太近，保留更强的
                    if p['height'] > merged_peaks[-1]['height']:
                        merged_peaks[-1] = p
                else:
                    merged_peaks.append(p)
            valid_peaks = merged_peaks
        
        return valid_peaks

    def _phase1_estimate_sigma(self, profile: np.ndarray, r_center: int,
                               mask_radius: int = 15) -> Tuple[int, float, float]:
        """估计sigma和r位置。
        
        使用高斯拟合估计sigma（低计数时更稳定），
        r位置用整数peak位置（最准确）。
        
        返回: (r, sigma, amp)
        """
        search_range = 15
        r_start = max(mask_radius, r_center - search_range)
        r_end = min(len(profile), r_center + search_range + 1)
        
        local = profile[r_start:r_end]
        if len(local) == 0:
            return r_center, 3.0, 0.0
        
        r_local = np.arange(r_start, r_end)
        
        # 用边缘值作为baseline
        baseline = (local[0] + local[-1]) / 2
        local_corrected = local - baseline
        
        pk_idx = np.argmax(local_corrected)
        pk_val = local_corrected[pk_idx]
        pk_r = r_local[pk_idx]  # 整数peak位置（最准确）
        
        if pk_val <= 0:
            return pk_r, 3.0, local[pk_idx]
        
        # 高斯拟合估计sigma
        def gaussian(x, amp, x0, sigma, offset):
            return amp * np.exp(-((x - x0)**2) / (2 * sigma**2)) + offset
        
        try:
            p0 = [pk_val, pk_r, 1.5, baseline]
            bounds = ([0, pk_r - 5, 0.3, -abs(baseline) - 1],
                      [pk_val * 3, pk_r + 5, 10.0, abs(baseline) + 1])
            
            popt, _ = curve_fit(gaussian, r_local, local, p0=p0, bounds=bounds, maxfev=200)
            amp_fit, r_fit, sigma_fit, offset_fit = popt
            
            if 0.3 <= sigma_fit <= 10:
                # r位置用整数peak位置（更准确），sigma用拟合值
                return float(pk_r), float(sigma_fit), float(amp_fit)
        except:
            pass
        
        # 回退到Linear FWHM
        half_max = pk_val / 2
        
        left = pk_idx
        while left > 0 and local_corrected[left] > half_max:
            left -= 1
        if left < pk_idx and local_corrected[left + 1] != local_corrected[left]:
            t = (half_max - local_corrected[left]) / (local_corrected[left + 1] - local_corrected[left])
            left_r = r_local[left] + t
        else:
            left_r = r_local[left]
        
        right = pk_idx
        while right < len(local_corrected) - 1 and local_corrected[right] > half_max:
            right += 1
        if right > pk_idx and local_corrected[right] != local_corrected[right - 1]:
            t = (half_max - local_corrected[right - 1]) / (local_corrected[right] - local_corrected[right - 1])
            right_r = r_local[right - 1] + t
        else:
            right_r = r_local[right]
        
        fwhm = right_r - left_r
        sigma = max(fwhm / 2.355, 0.3)
        
        return float(pk_r), float(sigma), float(pk_val)
        return float(pk_r), float(sigma), float(corrected_amp)

    def _fit_global_sigma_laser(self, peaks: List[Dict]) -> float:
        """
        全局拟合σ_laser：激光带宽在动量空间的展宽。
        （保留此函数供将来使用，但当前不使用）
        """
        if len(peaks) < 1:
            return 150.0
        
        sigma_r_products = []
        weights = []
        for p in peaks:
            sigma_r = p['sigma_individual'] * p['r']
            sigma_r_products.append(sigma_r)
            weights.append(p['amp'])
        
        sigma_r_products = np.array(sigma_r_products)
        weights = np.array(weights) / np.sum(weights)
        
        return np.sum(sigma_r_products * weights)

    def _phase1_radial_analysis(self) -> List[Dict]:
        """
        Phase 1: 径向分析
        
        注意：PSF展宽已经在Phase 0的笛卡尔坐标下去除了，
        这里只需要做Abel逆变换和高斯拟合。
        """
        print("\nPhase 1: Radial Analysis (PSF already removed in Cartesian)")
        print("=" * 60)
        
        mask_radius = 15
        
        # Step 1: 角向平均得到投影面profile（高SNR）
        print("  [Step 1] Angular average -> projection profile")
        proj_profile = np.mean(self._polar, axis=1)
        
        # Step 2: 在投影面profile上检测peaks位置
        print("  [Step 2] Peak detection on projection profile")
        peaks = self._phase1_find_peaks(proj_profile, mask_radius)
        print(f"    Found {len(peaks)} candidate peaks at r={[p['r'] for p in peaks]}")
        
        # Step 3: Abel逆变换（不需要先做径向反卷积，因为PSF已在(x,y)空间去除）
        print("  [Step 3] Abel inverse transform (no radial deconv needed)")
        abel_profile = abel.hansenlaw.hansenlaw_transform(proj_profile, direction='inverse')
        abel_profile = np.maximum(abel_profile, 0)
        
        # Step 4: 高斯拟合估计sigma和amp
        # 注意：amp是从角向平均的profile估计的，代表"平均强度"
        # 需要乘以r来转换为正确的密度（密度补偿）
        # 
        # 关于sigma：由于PSF已在笛卡尔坐标去除，这里测量的sigma主要来自：
        # 1. 真实的物理展宽（激光带宽等）
        # 2. Abel逆变换的数值误差（很小）
        # 3. 插值展宽（~0.55 px，很小）
        # 所以只需要去除插值展宽
        print("  [Step 4] Sigma/amp estimation (Gaussian fit)")
        results = []
        sigma_interp = self.sigma_interp  # 插值展宽
        
        for pk in peaks:
            r_center = pk['r']
            pk_r, sigma_measured, amp = self._phase1_estimate_sigma(
                abel_profile, r_center, mask_radius=mask_radius
            )
            
            # 只需要去除插值展宽（很小的校正）
            sigma_sq = sigma_measured**2 - sigma_interp**2
            if sigma_sq > 0:
                sigma_corrected = np.sqrt(sigma_sq)
            else:
                sigma_corrected = sigma_measured * 0.9  # 保守估计
            
            # amp校正：从"平均强度"转换为"密度"
            amp_corrected = amp * pk_r
            
            results.append({
                'r': float(pk_r),
                'sigma': float(sigma_corrected),
                'sigma_measured': float(sigma_measured),
                'amp': float(amp_corrected),
                'fwhm': float(sigma_corrected * 2.355),
            })
            print(f"    r={pk_r:.1f}: σ_meas={sigma_measured:.2f} -> σ_corr={sigma_corrected:.2f}, amp={amp_corrected:.2e}")
        
        print(f"  Final: {len(results)} peaks")
        print("=" * 60)
        return results
    
    def _deconvolve_profile(self, profile: np.ndarray, sigma: float) -> np.ndarray:
        """
        对1D profile做高斯反卷积（Wiener滤波）。
        """
        if sigma < 0.1:
            return profile
        
        n = len(profile)
        # 频域
        fft_profile = np.fft.fft(profile)
        freq = np.fft.fftfreq(n)
        
        # 高斯核的频域表示
        gauss_fft = np.exp(-2 * (np.pi * sigma * freq) ** 2)
        
        # Wiener滤波：避免除以0和噪声放大
        # H_wiener = H* / (|H|² + NSR)
        # 这里用简单的正则化
        nsr = 0.01  # 噪声信号比
        wiener = gauss_fft / (gauss_fft ** 2 + nsr)
        
        # 反卷积
        fft_deconv = fft_profile * wiener
        deconv = np.fft.ifft(fft_deconv).real
        
        return np.maximum(deconv, 0)

    # =========================================================================
    # Phase 2: 角向分析（多peak前向模型）
    # =========================================================================
    
    def _forward_model_for_beta(self, peaks_params: List[Tuple]) -> np.ndarray:
        """
        用于beta估计的前向模型。
        peaks_params: list of (r0, sigma_r, amp, beta)
        """
        # 构建3D分布
        img_3d = np.zeros((self.n, self.n))
        for r0, sigma_r, amp, beta in peaks_params:
            radial = amp * np.exp(-((self.R - r0)**2) / (2 * sigma_r**2))
            angular = 1 + beta * self.P2_GRID
            img_3d += radial * angular
        
        # Abel前向投影
        img_proj = abel.Transform(img_3d, method='hansenlaw', 
                                  direction='forward', verbose=False).transform
        
        # PSF卷积
        if self.sigma_psf > 0.1:
            img_proj = gaussian_filter(img_proj, sigma=self.sigma_psf)
        
        # 像素化
        if self.sigma_pixel > 0.1:
            img_proj = gaussian_filter(img_proj, sigma=self.sigma_pixel * 0.5)
        
        # xy -> rtheta
        n_r = self._polar.shape[0]
        n_theta = self._polar.shape[1]
        cy, cx = self.n // 2, self.n // 2
        
        theta_mesh, r_mesh = np.meshgrid(self._theta_grid, np.arange(n_r))
        x_cart = cx + r_mesh * np.cos(theta_mesh)
        y_cart = cy + r_mesh * np.sin(theta_mesh)
        
        polar_model = map_coordinates(img_proj, [y_cart, x_cart], order=3, mode='constant', cval=0.0)
        
        # 插值展宽
        if self.sigma_interp > 0.1:
            polar_model = gaussian_filter1d(polar_model, sigma=self.sigma_interp * 0.5, axis=0)
        
        return polar_model
    
    def _estimate_beta_direct(self, r_idx: int) -> float:
        """直接FFT估计beta（单个r位置）。"""
        polar = self._polar
        n_theta = polar.shape[1]
        angular = polar[r_idx, :]
        
        fft = np.fft.fft(angular)
        dc = np.abs(fft[0]) / n_theta
        cos2_amp = 2 * np.abs(fft[2]) / n_theta
        phase = np.angle(fft[2])
        sign = 1.0 if np.abs(phase) < np.pi/2 else -1.0
        cos2_signed = sign * cos2_amp
        
        if dc > 1e-6:
            beta = 4.0 * cos2_signed / (3.0 * dc - cos2_signed)
            return np.clip(beta, -1.0, 2.0)
        return 0.0
    
    def _estimate_beta_in_range(self, r_center: int, sigma: float) -> Tuple[float, float]:
        """
        在peak范围内估计beta（加权平均）。
        
        物理约束：同一个peak在r±Δr范围内beta应该一致。
        
        返回: (beta, uncertainty)
        """
        polar = self._polar
        n_r = polar.shape[0]
        
        # peak范围：r ± 2σ
        r_range = max(3, int(2 * sigma))
        r_start = max(10, r_center - r_range)
        r_end = min(n_r - 1, r_center + r_range + 1)
        
        betas = []
        weights = []
        
        for r_idx in range(r_start, r_end):
            beta_r = self._estimate_beta_direct(r_idx)
            
            # 权重：距离中心越近权重越大，信号越强权重越大
            dist_weight = np.exp(-0.5 * ((r_idx - r_center) / max(sigma, 1))**2)
            signal_weight = np.mean(polar[r_idx, :])
            weight = dist_weight * signal_weight
            
            betas.append(beta_r)
            weights.append(weight)
        
        if len(betas) == 0 or sum(weights) < 1e-10:
            return 0.0, 1.0
        
        # 加权平均
        weights = np.array(weights)
        weights /= np.sum(weights)
        beta_mean = np.sum(np.array(betas) * weights)
        
        # 不确定度：加权标准差
        beta_std = np.sqrt(np.sum(weights * (np.array(betas) - beta_mean)**2))
        
        return np.clip(beta_mean, -1.0, 2.0), beta_std
    
    def _phase2_angular_analysis(self, params: List[Dict]) -> List[Dict]:
        """Phase 2: 角度分析。
        
        改进流程：
        1. 对每个角度做Abel逆变换，得到3D分布的(r,θ)切片
        2. 对每个r做角向反卷积（去除探测器展宽对k=2成分的衰减）
        3. 从外到内估计beta，减去外层贡献
        """
        print("\nPhase 2: Angular Analysis (improved)")
        print("=" * 60)
        
        if len(params) == 0:
            return params
        
        sigma_det = self.get_sigma_detector()
        
        # Step 1: 对每个角度做Abel逆变换
        print("  [Step 1] Abel inverse for each angle")
        polar_3d = self._get_abel_inverted_polar()
        
        # Step 2: 对每个r做角向反卷积
        print(f"  [Step 2] Angular deconvolution (σ_det={sigma_det:.2f} px)")
        polar_3d_deconv = self._deconvolve_angular(polar_3d, sigma_det)
        
        # Step 3: 从外到内估计beta
        print("  [Step 3] Beta estimation (outer to inner)")
        
        # 按r从大到小排序
        params_sorted = sorted(enumerate(params), key=lambda x: x[1]['r'], reverse=True)
        
        # 用于存储已估计的外层peaks的贡献
        outer_contribution = np.zeros_like(polar_3d_deconv)
        
        for sort_idx, (orig_idx, p) in enumerate(params_sorted):
            r_center = int(p['r'])
            sigma = p['sigma']
            amp = p['amp']
            
            if r_center < 10 or r_center >= polar_3d_deconv.shape[0]:
                p['beta'] = 0.0
                p['beta_uncertainty'] = 1.0
                continue
            
            # 减去外层贡献
            polar_corrected = polar_3d_deconv - outer_contribution
            polar_corrected = np.maximum(polar_corrected, 0)
            
            # 在校正后的分布上估计beta
            beta_3d, beta_std = self._estimate_beta_in_range_simple(
                polar_corrected, r_center, sigma
            )
            
            p['beta'] = beta_3d
            p['beta_uncertainty'] = beta_std
            
            # 计算这个peak对内层的贡献（用于下一个peak的校正）
            # 构建这个peak的模型
            if sort_idx < len(params_sorted) - 1:  # 不是最内层的peak
                n_r, n_theta = polar_3d_deconv.shape
                r_grid = np.arange(n_r)
                theta_grid = self._theta_grid
                
                # 径向高斯
                radial_profile = amp * np.exp(-((r_grid - r_center)**2) / (2 * sigma**2))
                
                # 角向分布 (1 + beta * P2)
                cos_theta = np.cos(theta_grid)
                P2 = 0.5 * (3 * cos_theta**2 - 1)
                angular_profile = 1 + beta_3d * P2
                
                # 外积得到2D贡献
                peak_contribution = np.outer(radial_profile, angular_profile)
                outer_contribution += peak_contribution
            
            print(f"  Peak {orig_idx+1} (r={p['r']:.1f}): β={beta_3d:.3f} ± {beta_std:.3f}")
        
        print("=" * 60)
        return params
    
    def _get_abel_inverted_polar(self) -> np.ndarray:
        """
        对每个角度做Abel逆变换，得到3D分布的(r,θ)切片。
        """
        polar = self._polar
        n_r, n_theta = polar.shape
        polar_3d = np.zeros_like(polar)
        
        for theta_idx in range(n_theta):
            profile = polar[:, theta_idx]
            # Abel逆变换
            profile_3d = abel.hansenlaw.hansenlaw_transform(profile, direction='inverse')
            polar_3d[:, theta_idx] = np.maximum(profile_3d, 0)
        
        return polar_3d
    
    def _deconvolve_angular(self, polar: np.ndarray, sigma_det: float) -> np.ndarray:
        """
        对每个r做角向反卷积，去除探测器展宽对k=2成分的衰减。
        
        角向展宽 σ_θ = σ_det / r（弧度）
        k=2成分的衰减因子 = exp(-0.5 × (2 × σ_θ)²)
        """
        n_r, n_theta = polar.shape
        result = np.zeros_like(polar)
        
        for r_idx in range(n_r):
            if r_idx < 5:
                result[r_idx, :] = polar[r_idx, :]
                continue
            
            angular = polar[r_idx, :]
            fft = np.fft.fft(angular)
            
            # 角向展宽
            sigma_theta = sigma_det / r_idx  # 弧度
            
            # 对每个k成分做反卷积
            fft_deconv = np.zeros_like(fft)
            for k in range(len(fft) // 2 + 1):
                # 衰减因子
                attenuation = np.exp(-0.5 * (k * sigma_theta) ** 2)
                if attenuation > 0.1:  # 避免噪声放大
                    fft_deconv[k] = fft[k] / attenuation
                    if k > 0 and k < len(fft) // 2:
                        fft_deconv[-k] = fft[-k] / attenuation
                else:
                    fft_deconv[k] = fft[k]
                    if k > 0 and k < len(fft) // 2:
                        fft_deconv[-k] = fft[-k]
            
            result[r_idx, :] = np.maximum(np.fft.ifft(fft_deconv).real, 0)
        
        return result
    
    def _estimate_beta_in_range_simple(self, polar: np.ndarray,
                                        r_center: int, sigma: float) -> Tuple[float, float]:
        """
        在给定polar矩阵的peak范围内估计beta（加权平均）。
        简化版：不做反卷积校正（假设已经反卷积过了）。
        """
        n_r = polar.shape[0]
        n_theta = polar.shape[1]
        
        # peak范围：r ± 2σ
        r_range = max(3, int(2 * sigma))
        r_start = max(10, r_center - r_range)
        r_end = min(n_r - 1, r_center + r_range + 1)
        
        betas = []
        weights = []
        
        for r_idx in range(r_start, r_end):
            # 直接FFT估计beta
            angular = polar[r_idx, :]
            fft = np.fft.fft(angular)
            dc = np.abs(fft[0]) / n_theta
            cos2_amp = 2 * np.abs(fft[2]) / n_theta
            phase = np.angle(fft[2])
            sign = 1.0 if np.abs(phase) < np.pi/2 else -1.0
            cos2_signed = sign * cos2_amp
            
            if dc > 1e-6:
                beta_r = 4.0 * cos2_signed / (3.0 * dc - cos2_signed)
                beta_r = np.clip(beta_r, -1.0, 2.0)
            else:
                continue
            
            # 权重：距离中心越近权重越大，信号越强权重越大
            dist_weight = np.exp(-0.5 * ((r_idx - r_center) / max(sigma, 1))**2)
            signal_weight = np.mean(polar[r_idx, :])
            weight = dist_weight * signal_weight
            
            betas.append(beta_r)
            weights.append(weight)
        
        if len(betas) == 0 or sum(weights) < 1e-10:
            return 0.0, 1.0
        
        # 加权平均
        weights = np.array(weights)
        weights /= np.sum(weights)
        beta_mean = np.sum(np.array(betas) * weights)
        
        # 不确定度：加权标准差
        beta_std = np.sqrt(np.sum(weights * (np.array(betas) - beta_mean)**2))
        
        return np.clip(beta_mean, -1.0, 2.0), beta_std
    
    # =========================================================================
    # Phase 3: 前向模型优化
    # =========================================================================
    
    def _forward_model_polar(self, params: np.ndarray, n_peaks: int) -> np.ndarray:
        """
        完整前向模型：3D高斯 → Abel投影 → PSF卷积 → 像素化 → xy→rθ
        
        返回polar矩阵，与Phase 0处理后的self._polar比较。
        """
        params = params.reshape(n_peaks, 4)
        
        # 1. 构建3D分布（在2D切片上表示）
        img_3d = np.zeros((self.n, self.n))
        for i in range(n_peaks):
            r0, sig, amp, beta = params[i]
            sig = max(sig, 0.3)
            
            # 径向高斯（3D空间的sigma）
            radial = amp * np.exp(-((self.R - r0)**2) / (2 * sig**2))
            # 角向分布
            angular = 1 + beta * self.P2_GRID
            img_3d += radial * angular
        
        # 2. Abel前向投影
        img_proj = abel.Transform(img_3d, method='hansenlaw', 
                                  direction='forward', verbose=False).transform
        
        # 3. PSF卷积
        if self.sigma_psf > 0.1:
            img_proj = gaussian_filter(img_proj, sigma=self.sigma_psf)
        
        # 4. 像素化效应（已经是离散的，这里用小高斯模拟）
        if self.sigma_pixel > 0.1:
            img_proj = gaussian_filter(img_proj, sigma=self.sigma_pixel * 0.5)
        
        # 5. xy → rθ 转换
        n_r = self._polar.shape[0]
        n_theta = self._polar.shape[1]
        cy, cx = self.n // 2, self.n // 2
        
        theta_mesh, r_mesh = np.meshgrid(self._theta_grid, np.arange(n_r))
        x_cart = cx + r_mesh * np.cos(theta_mesh)
        y_cart = cy + r_mesh * np.sin(theta_mesh)
        
        polar_model = map_coordinates(img_proj, [y_cart, x_cart], order=3, mode='constant', cval=0.0)
        
        # 6. 插值展宽（用小高斯模拟）
        if self.sigma_interp > 0.1:
            polar_model = gaussian_filter1d(polar_model, sigma=self.sigma_interp * 0.5, axis=0)
        
        return polar_model
    
    def _forward_model_loss(self, params_flat: np.ndarray, priors: List[Dict], 
                           n_peaks: int) -> np.ndarray:
        """
        损失函数：比较前向模型与Phase 0处理后的polar矩阵。
        
        改进：
        1. 增强所有先验约束（Phase 1/2的估计已经比较准确）
        2. 减少2D残差的权重（避免过拟合噪声）
        3. 主要依赖径向profile和先验约束
        """
        
        # 前向模型
        polar_model = self._forward_model_polar(params_flat, n_peaks)
        
        # 目标
        polar_target = self._polar
        
        # 2D残差（大幅降低权重）
        weights = 1.0 / np.sqrt(np.abs(polar_target) + 10.0)
        res_polar = (polar_model - polar_target) * weights * 0.1  # 大幅降低
        
        # 径向profile残差（主要约束）
        profile_model = np.mean(polar_model, axis=1)
        profile_target = np.mean(polar_target, axis=1)
        weights_1d = 1.0 / np.sqrt(np.abs(profile_target) + 1.0)
        res_1d = (profile_model - profile_target) * weights_1d * 30.0
        
        # 先验约束（关键！）
        params = params_flat.reshape(n_peaks, 4)
        prior_penalty = []
        
        for i in range(n_peaks):
            r0, sig, amp, beta = params[i]
            prior = priors[i]
            
            # 位置约束（增强！Phase 1的位置估计很准确）
            r_dev = (r0 - prior['r']) / 2.0  # 更严格
            prior_penalty.append(r_dev * 50.0)  # 大幅增强
            
            # Sigma约束（增强！）
            sigma_dev = (sig - prior['sigma']) / (prior['sigma'] + 0.3)
            prior_penalty.append(sigma_dev * 30.0)  # 增强
            
            # Amp约束
            amp_dev = (amp - prior['amp']) / (prior['amp'] + 0.1)
            prior_penalty.append(amp_dev * 10.0)
            
            # Beta约束（Phase 2的估计准确）
            beta_uncertainty = prior.get('beta_uncertainty', 0.1)
            beta_weight = 1.0 / (beta_uncertainty + 0.02)
            beta_dev = (beta - prior['beta']) * beta_weight
            prior_penalty.append(beta_dev * 200.0)
        
        return np.concatenate([res_polar.ravel(), res_1d, np.array(prior_penalty)])
    
    def _compute_radial_profile(self, img_2d: np.ndarray) -> np.ndarray:
        """计算2D图像的径向分布。"""
        radial_sum = np.bincount(self.r_flat, weights=img_2d.ravel(), 
                                  minlength=self.max_r_idx + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = radial_sum / self.pixel_counts
        profile[~np.isfinite(profile)] = 0
        return profile[:len(self.r_grid_1d)]

    def solve(self, image_2d: np.ndarray, skip_phase3: bool = True) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        主求解器。
        
        Args:
            image_2d: 输入图像
            skip_phase3: 是否跳过Phase 3优化（默认True，因为Phase 1/2已经很准确）
        """
        t0 = time.time()
        
        # Phase 0
        self._phase0_preprocess(image_2d, n_theta=720)
        
        # Phase 1
        params = self._phase1_radial_analysis()
        if not params:
            return [], self.r_grid_1d, np.zeros_like(self.r_grid_1d)
        
        # Phase 2
        params = self._phase2_angular_analysis(params)
        
        # Phase 3: 可选优化
        if skip_phase3:
            print("\nPhase 3: Skipped (using Phase 1/2 estimates)")
            final_params = []
            recon_profile = np.zeros_like(self.r_grid_1d)
            
            for p in params:
                final_params.append({
                    'r': p['r'], 
                    'sigma': p['sigma'], 
                    'sigma_measured': p.get('sigma_measured', p['sigma']),  # 保留用于BR计算
                    'fwhm': 2.355 * p['sigma'],
                    'amp': p['amp'], 
                    'beta': p['beta']
                })
                recon_profile += p['amp'] * np.exp(-((self.r_grid_1d - p['r'])**2) / (2*p['sigma']**2))
            
            print(f"Solver time: {time.time()-t0:.2f}s")
            return final_params, self.r_grid_1d, recon_profile
        
        # Phase 3: 优化（如果需要）
        n_peaks = len(params)
        x0, lb, ub, priors = [], [], [], []
        
        print(f"\nPhase 3: Optimization ({n_peaks} peaks)")
        
        for p in params:
            r_val = p['r']
            
            x0.extend([p['r'], p['sigma'], p['amp'], p['beta']])
            priors.append({
                'r': p['r'],
                'sigma': p['sigma'], 
                'amp': p['amp'], 
                'beta': p['beta'], 
                'beta_uncertainty': p.get('beta_uncertainty', 0.1)
            })
            
            # 非常紧的边界（Phase 1/2的估计已经很准确）
            sigma_init = p['sigma']
            sigma_lb = max(0.3, sigma_init * 0.8)  # 最小为初始值的80%
            sigma_ub = min(5.0, sigma_init * 1.25)  # 最大为初始值的125%
            
            # 位置边界非常紧
            lb.extend([max(5.0, r_val-1.5), sigma_lb, 0.0, -1.1])
            ub.extend([r_val+1.5, sigma_ub, np.inf, 2.1])
            
            print(f"  r={r_val:.1f}: σ_init={p['sigma']:.2f}, bounds=[{sigma_lb:.2f}, {sigma_ub:.2f}]")
        
        try:
            res = least_squares(
                self._forward_model_loss, x0=np.array(x0),
                bounds=(np.array(lb), np.array(ub)),
                args=(priors, n_peaks),
                loss='soft_l1', f_scale=1.0, method='trf', 
                ftol=1e-4, xtol=1e-4, max_nfev=30  # 减少迭代次数
            )
            final_x = res.x
            print(f"  Optimization converged: {res.success}")
        except Exception as e:
            print(f"  Optimization warning: {e}")
            final_x = np.array(x0)
        
        # 构建结果
        p_reshaped = final_x.reshape(n_peaks, 4)
        final_params = []
        recon_profile = np.zeros_like(self.r_grid_1d)
        
        max_amp = np.max(p_reshaped[:, 2]) if n_peaks > 0 else 0
        
        for i in range(n_peaks):
            r0, sig, amp, beta = p_reshaped[i]
            if amp < 0.05 * max_amp or r0 < 10:
                continue
            
            final_params.append({
                'r': r0, 'sigma': sig, 'fwhm': 2.355 * sig,
                'amp': amp, 'beta': beta
            })
            recon_profile += amp * np.exp(-((self.r_grid_1d - r0)**2) / (2*sig**2))
        
        print(f"Solver time: {time.time()-t0:.2f}s")
        return final_params, self.r_grid_1d, recon_profile


# =============================================================================
# Helper Functions
# =============================================================================

def radius_to_energy(radius_px: float, pixel_size_mm: float, vmi_k: float, 
                     mass_amu: float = None) -> float:
    """半径转能量。"""
    from scipy.constants import electron_mass, elementary_charge, atomic_mass
    
    if mass_amu is None:
        mass_amu = electron_mass / atomic_mass
    
    radius_mm = radius_px * pixel_size_mm
    velocity = radius_mm / vmi_k
    mass_kg = mass_amu * atomic_mass
    E_joule = 0.5 * mass_kg * velocity**2
    
    return E_joule / elementary_charge


def reconstruct_vmi_image_v2(image: np.ndarray, config=None, 
                              verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """V2重建主函数。"""
    n_pixels = image.shape[0]
    
    fitter = PhysicsBasedFitterV2(n_pixels)
    fitter.calibrate_from_config(config)
    
    params, r_grid, recon_profile = fitter.solve(image)
    
    if config is not None:
        for p in params:
            p['energy_eV'] = radius_to_energy(p['r'], config.pixel_size, config.vmi_k, config.mass)
    
    # BR计算：精确3D高斯积分公式
    # N = 4π × A₃D × σ × √(2π) × (r² + σ²)
    integrated_intensities = []
    for p in params:
        sigma_for_br = p.get('sigma_measured', p['sigma'])
        r = p['r']
        integrated_intensities.append(p['amp'] * sigma_for_br * (r**2 + sigma_for_br**2))
    
    total_intensity = sum(integrated_intensities)
    if total_intensity > 0:
        for i, p in enumerate(params):
            p['branching_ratio'] = integrated_intensities[i] / total_intensity
    
    metadata = {
        'n_peaks': len(params), 'r_grid': r_grid, 'recon_profile': recon_profile,
        'image_size': n_pixels, 'sigma_E': fitter.sigma_E, 'vmi_k': fitter.vmi_k,
        'sigma_psf': fitter.sigma_psf, 'sigma_pixel': fitter.sigma_pixel,
        'version': 'V2_Complete'
    }
    
    if verbose:
        print("\n" + "="*60)
        print("RECONSTRUCTION RESULTS (V2)")
        print("="*60)
        for i, p in enumerate(params):
            print(f"\nPeak {i+1}:")
            print(f"  r = {p['r']:.1f} px")
            if 'energy_eV' in p:
                print(f"  E = {p['energy_eV']:.3f} eV")
            print(f"  σ = {p['sigma']:.2f} px, FWHM = {p['fwhm']:.2f} px")
            print(f"  β = {p['beta']:.3f}")
            if 'branching_ratio' in p:
                print(f"  BR = {p['branching_ratio']:.3f}")
        print("="*60)
    
    return params, metadata


def reconstruct_vmi_image(image: np.ndarray, config=None, 
                          verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """兼容旧接口。"""
    return reconstruct_vmi_image_v2(image, config, verbose)
