"""
VMI 重建工具
============
从 2D XY 散点数据重建牛顿球参数 (r0, σ, amp, β)

使用方法:
    from vmi_reconstruction import VMIReconstructor
    
    vmi = VMIReconstructor(xy_data)
    vmi.reconstruct(n_peaks=2)
    vmi.plot()
    vmi.summary()

算法:
    1. 多尺度 MLE 提取径向参数 (r0, σ, amp)
    2. 角分布拟合提取各向异性参数 β
    3. 可视化对比实验数据与重建模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import correlate, find_peaks
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import zoom
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ============================================================
# 数据类
# ============================================================

@dataclass
class PeakResult:
    """单个峰的重建结果"""
    r0: float           # 峰位置
    sigma: float        # 高斯宽度
    amp: float          # 振幅
    beta: float         # 各向异性参数
    r0_err: float = 0   # r0 误差 (MAD)
    sigma_err: float = 0  # σ 误差 (MAD)
    beta_err: float = 0   # β 误差 (MAD)


# ============================================================
# 核心物理模型
# ============================================================

def abel_projection(R, r0, sigma):
    """高斯球壳的 Abel 投影"""
    z = np.linspace(0, 5 * sigma + 10, 200)
    dz = z[1] - z[0]
    R_mat = np.atleast_1d(R)[:, np.newaxis]
    Z_mat = z[np.newaxis, :]
    r_val = np.sqrt(Z_mat**2 + R_mat**2)
    gauss = np.exp(-(r_val - r0)**2 / (2 * sigma**2))
    result = 2 * np.trapz(gauss, dx=dz, axis=1)
    return result if len(R.shape) > 0 else result[0]


def P2(cos_theta):
    """Legendre P2 多项式"""
    return (3 * cos_theta**2 - 1) / 2


def mexican_hat(x, sigma):
    """Mexican Hat 零均值模板"""
    norm = 2 / (np.sqrt(3 * sigma) * np.pi**0.25)
    t = (x / sigma)**2
    return norm * (1 - t) * np.exp(-t / 2)


# ============================================================
# VMI 重建器
# ============================================================

class VMIReconstructor:
    """VMI 重建器"""
    
    def __init__(self, xy_data: np.ndarray):
        """
        初始化
        
        Parameters:
        -----------
        xy_data : array (N, 2), XY 坐标数据
        """
        self.xy_data = np.asarray(xy_data)
        self.n_events = len(xy_data)
        
        # 找中心
        self.center = (np.mean(xy_data[:, 0]), np.mean(xy_data[:, 1]))
        
        # 转极坐标
        dx = xy_data[:, 0] - self.center[0]
        dy = xy_data[:, 1] - self.center[1]
        self.r = np.sqrt(dx**2 + dy**2)
        self.theta = np.arctan2(dy, dx)
        self.r_max = np.percentile(self.r, 99)
        
        # 结果
        self.peaks: List[PeakResult] = []
    
    def reconstruct(self, n_peaks: int = 2, verbose: bool = True):
        """
        执行重建
        
        Parameters:
        -----------
        n_peaks : int, 峰数量
        verbose : bool, 是否打印详细信息
        """
        if verbose:
            print("=" * 60)
            print("VMI 重建")
            print("=" * 60)
            print(f"数据点数: {self.n_events}")
            print(f"中心: ({self.center[0]:.2f}, {self.center[1]:.2f})")
            print(f"r_max: {self.r_max:.2f}")
        
        # Step 1: 多尺度 MLE 提取径向参数
        if verbose:
            print("\n--- Step 1: 多尺度 MLE 提取 r0, σ, amp ---")
        radial_results = self._multiscale_mle(n_peaks, verbose)
        
        # Step 2: 提取 β
        if verbose:
            print("\n--- Step 2: 提取各向异性参数 β ---")
        beta_results = self._extract_beta(radial_results, verbose)
        
        # 合并结果
        self.peaks = []
        for i in range(len(radial_results)):
            self.peaks.append(PeakResult(
                r0=radial_results[i]['r0'],
                sigma=radial_results[i]['sigma'],
                amp=radial_results[i]['amp'],
                beta=beta_results[i]['beta'],
                r0_err=radial_results[i].get('r0_mad', 0),
                sigma_err=radial_results[i].get('sigma_mad', 0),
                beta_err=beta_results[i].get('beta_mad', 0)
            ))
        
        if verbose:
            print("\n" + "=" * 60)
            print("重建完成")
            self.summary()
        
        return self.peaks
    
    def _multiscale_mle(self, n_peaks: int, verbose: bool) -> List[dict]:
        """多尺度 MLE 提取径向参数"""
        dr_values = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
        
        results_by_dr = {}
        for dr in dr_values:
            result = self._single_scale_mle(dr, n_peaks)
            if result is not None:
                results_by_dr[dr] = result
        
        if len(results_by_dr) < 2:
            # 回退到单尺度
            return self._single_scale_mle(0.1, n_peaks) or []
        
        # 对每个峰取中位数
        final_results = []
        for peak_idx in range(n_peaks):
            r0_list, sigma_list, amp_list = [], [], []
            
            for dr, peaks in results_by_dr.items():
                if peak_idx < len(peaks):
                    r0_list.append(peaks[peak_idx]['r0'])
                    sigma_list.append(peaks[peak_idx]['sigma'])
                    amp_list.append(peaks[peak_idx]['amp'])
            
            if len(r0_list) >= 2:
                r0_med = np.median(r0_list)
                sigma_med = np.median(sigma_list)
                amp_med = np.median(amp_list)
                
                final_results.append({
                    'r0': r0_med,
                    'sigma': sigma_med,
                    'amp': amp_med,
                    'r0_mad': np.median(np.abs(np.array(r0_list) - r0_med)),
                    'sigma_mad': np.median(np.abs(np.array(sigma_list) - sigma_med))
                })
                
                if verbose:
                    print(f"Peak {peak_idx+1}: r0={r0_med:.3f}±{final_results[-1]['r0_mad']:.4f}, "
                          f"σ={sigma_med:.4f}±{final_results[-1]['sigma_mad']:.4f}")
        
        return final_results
    
    def _single_scale_mle(self, dr: float, n_peaks: int) -> Optional[List[dict]]:
        """单尺度 MLE"""
        # 径向直方图
        n_bins = int(self.r_max / dr)
        if n_bins < 20:
            return None
        
        counts, bin_edges = np.histogram(self.r, bins=n_bins, range=(0, self.r_max))
        R = (bin_edges[:-1] + bin_edges[1:]) / 2
        H_R = counts.astype(float)
        rho_R = H_R / (2 * np.pi * R + 1e-6)
        
        # 峰检测
        peaks = self._detect_peaks(R, rho_R)
        if len(peaks) < n_peaks:
            return None
        peaks = sorted(peaks[:n_peaks])
        
        # MLE 拟合
        return self._fit_peaks_mle(R, H_R, peaks)
    
    def _detect_peaks(self, R: np.ndarray, rho: np.ndarray) -> List[float]:
        """Mexican Hat 峰检测"""
        dr = R[1] - R[0]
        sigma_template = 0.5
        template_r = np.linspace(-5*sigma_template, 5*sigma_template, 
                                  max(int(10*sigma_template/dr), 10))
        template = mexican_hat(template_r, sigma_template)
        
        response = correlate(rho, template, mode='same')
        
        valid_mask = R > 5
        response_masked = response.copy()
        response_masked[~valid_mask] = 0
        
        peaks_idx, props = find_peaks(response_masked,
                                       prominence=np.max(np.abs(response_masked)) * 0.05,
                                       distance=max(3, int(1.0/dr)))
        
        if len(peaks_idx) == 0:
            return []
        
        order = np.argsort(props['prominences'])[::-1]
        return [R[peaks_idx[i]] for i in order]
    
    def _fit_peaks_mle(self, R: np.ndarray, H_R: np.ndarray, 
                       peak_init: List[float]) -> List[dict]:
        """MLE 拟合"""
        n_peaks = len(peak_init)
        
        # sigma_max
        if n_peaks > 1:
            min_sep = min(peak_init[i+1] - peak_init[i] for i in range(n_peaks-1))
            sigma_max = max(min_sep / 3, 0.5)
        else:
            sigma_max = 2.0
        
        def model(params):
            mu = np.zeros_like(R, dtype=float)
            for i in range(n_peaks):
                r0, sigma, amp = params[3*i:3*i+3]
                mu += amp * abel_projection(R, r0, sigma)
            rho_bg, dc_offset = params[-2], params[-1]
            return np.maximum(2 * np.pi * R * (mu + rho_bg) + dc_offset, 1e-9)
        
        def loss(params):
            mu = model(params)
            z = (H_R - mu) / np.sqrt(mu + 400)
            return np.sum(z**2) + len(z) * 10 * (z.mean()**2 + (z.var() - 1)**2)
        
        # 多初值优化
        best_result, best_loss = None, np.inf
        
        for sigma_factor in [0.3, 0.5, 0.7]:
            init = []
            bounds = []
            for r0 in peak_init:
                r0_tol = r0 * 0.05
                init.extend([r0, sigma_max * sigma_factor, 1.0])
                bounds.extend([
                    (max(r0 - r0_tol, 0.1), r0 + r0_tol),
                    (0.05, sigma_max),
                    (0, None)
                ])
            init.extend([0.3, 50])
            bounds.extend([(0, None), (0, None)])
            
            res = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res.x
        
        # 提取结果
        results = []
        for i in range(n_peaks):
            results.append({
                'r0': best_result[3*i],
                'sigma': best_result[3*i + 1],
                'amp': best_result[3*i + 2]
            })
        
        return results
    
    def _extract_beta(self, radial_results: List[dict], verbose: bool) -> List[dict]:
        """提取 β 参数"""
        results = []
        
        for i, p in enumerate(radial_results):
            r0 = p['r0']
            sigma = p['sigma']
            
            # 选取峰附近的点
            mask = (self.r >= r0 - 2*sigma) & (self.r < r0 + 2*sigma)
            theta_peak = self.theta[mask]
            n_points = np.sum(mask)
            
            if n_points < 50:
                results.append({'beta': 0, 'beta_mad': np.nan})
                continue
            
            # 多尺度拟合
            betas = []
            for n_theta in [24, 36, 48, 60]:
                beta = self._fit_beta(theta_peak, n_theta)
                if not np.isnan(beta):
                    betas.append(beta)
            
            if len(betas) >= 2:
                beta_med = np.median(betas)
                beta_mad = np.median(np.abs(np.array(betas) - beta_med))
            else:
                beta_med = betas[0] if betas else 0
                beta_mad = np.nan
            
            results.append({'beta': beta_med, 'beta_mad': beta_mad})
            
            if verbose:
                print(f"Peak {i+1} (r={r0:.2f}): β={beta_med:.3f}±{beta_mad:.4f}, n={n_points}")
        
        return results
    
    def _fit_beta(self, theta_data: np.ndarray, n_bins: int) -> float:
        """拟合单个 β"""
        hist, bin_edges = np.histogram(theta_data, bins=n_bins, range=(-np.pi, np.pi))
        theta_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts = hist.astype(float)
        
        if np.sum(counts) == 0:
            return np.nan
        
        def model(theta, A, beta):
            return A * (1 + beta * P2(np.cos(theta)))
        
        try:
            A_init = np.mean(counts)
            popt, _ = curve_fit(model, theta_centers, counts,
                               p0=[A_init, 0], bounds=([0, -1], [np.inf, 2]))
            return popt[1]
        except:
            return np.nan
    
    def summary(self):
        """打印结果汇总"""
        print("=" * 60)
        print(f"{'Peak':<6} {'r0':<12} {'σ':<12} {'amp':<10} {'β':<12}")
        print("-" * 60)
        for i, p in enumerate(self.peaks):
            print(f"{i+1:<6} {p.r0:<12.3f} {p.sigma:<12.4f} {p.amp:<10.3f} {p.beta:<12.3f}")
        print("=" * 60)
    
    def plot(self, figsize: Tuple[int, int] = (14, 10)):
        """可视化结果"""
        xy_centered = self.xy_data - np.array(self.center)
        extent = self.r_max * 1.1
        
        fig = plt.figure(figsize=figsize)
        
        # 1. 实验 2D 图
        ax1 = fig.add_subplot(2, 3, 1)
        H_exp, xedges, yedges = np.histogram2d(
            xy_centered[:, 0], xy_centered[:, 1],
            bins=150, range=[[-extent, extent], [-extent, extent]])
        ax1.imshow(H_exp.T, extent=[-extent, extent, -extent, extent],
                   origin='lower', cmap='hot', aspect='equal')
        ax1.set_title('Experiment Data')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        
        # 2. 重建模型
        ax2 = fig.add_subplot(2, 3, 2)
        X, Y, I_sim = self._generate_model_image(extent, 300)
        ax2.imshow(I_sim, extent=[-extent, extent, -extent, extent],
                   origin='lower', cmap='hot', aspect='equal')
        ax2.set_title('Reconstructed Model')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        
        # 3. 残差
        ax3 = fig.add_subplot(2, 3, 3)
        H_exp_norm = H_exp.T / np.max(H_exp)
        I_sim_norm = I_sim / np.max(I_sim)
        zf = H_exp.shape[0] / I_sim.shape[0]
        I_sim_resized = zoom(I_sim_norm, zf)[:H_exp.shape[1], :H_exp.shape[0]]
        residual = H_exp_norm - I_sim_resized
        ax3.imshow(residual, extent=[-extent, extent, -extent, extent],
                   origin='lower', cmap='RdBu', aspect='equal', vmin=-0.3, vmax=0.3)
        ax3.set_title('Residual')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        
        # 4. 径向分布
        ax4 = fig.add_subplot(2, 3, 4)
        dr = 0.1
        r_bins = np.arange(0, self.r_max, dr)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        H_r_exp, _ = np.histogram(self.r, bins=r_bins)
        
        # 模型径向分布
        H_r_model = np.zeros_like(r_centers)
        for p in self.peaks:
            H_r_model += p.amp * abel_projection(r_centers, p.r0, p.sigma) * 2 * np.pi * r_centers * dr
        H_r_model = H_r_model / np.max(H_r_model) * np.max(H_r_exp)
        
        ax4.step(r_centers, H_r_exp, where='mid', color='black', lw=1.5, label='Experiment')
        ax4.plot(r_centers, H_r_model, 'r-', lw=2, label='Model')
        for p in self.peaks:
            ax4.axvline(p.r0, ls='--', alpha=0.5, color='blue')
        ax4.set_xlabel('Radius (mm)')
        ax4.set_ylabel('Counts')
        ax4.set_title('Radial Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5-6. 角分布
        for idx, ax in enumerate([fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)]):
            if idx >= len(self.peaks):
                ax.axis('off')
                continue
            
            p = self.peaks[idx]
            mask = (self.r >= p.r0 - 2*p.sigma) & (self.r < p.r0 + 2*p.sigma)
            theta_peak = self.theta[mask]
            
            n_theta = 36
            theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
            theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
            H_theta, _ = np.histogram(theta_peak, bins=theta_bins)
            
            # 模型
            H_model = 1 + p.beta * P2(np.cos(theta_centers))
            H_model = H_model / np.max(H_model) * np.max(H_theta)
            
            ax.bar(np.degrees(theta_centers), H_theta, width=8, alpha=0.7, label='Exp')
            ax.plot(np.degrees(theta_centers), H_model, 'r-', lw=2, label=f'β={p.beta:.2f}')
            ax.set_xlabel('θ (degrees)')
            ax.set_ylabel('Counts')
            ax.set_title(f'Peak {idx+1} (r={p.r0:.1f})')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _generate_model_image(self, extent: float, n_pixels: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成模型 2D 图像"""
        x = np.linspace(-extent, extent, n_pixels)
        y = np.linspace(-extent, extent, n_pixels)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        I_total = np.zeros_like(R)
        for p in self.peaks:
            I_radial = p.amp * abel_projection(R.flatten(), p.r0, p.sigma).reshape(R.shape)
            I_angular = 1 + p.beta * P2(np.cos(Theta))
            I_total += I_radial * I_angular
        
        return X, Y, I_total


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 加载数据
    print("加载数据...")
    mat_data = loadmat('electron_shilpa_XY.mat')
    XY = mat_data['XY']
    
    # 重建
    vmi = VMIReconstructor(XY)
    vmi.reconstruct(n_peaks=2)
    
    # 可视化
    vmi.plot()
