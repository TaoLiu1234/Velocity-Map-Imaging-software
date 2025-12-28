"""
Abel Backward Reconstruction via Forward Fitting (X1)

通过模拟前向过程并优化参数来重建VMI图像的物理参数。
核心思想：生成与观测数据统计特征一致的XY散点分布。

优化参数：
- Peak数量（通过模型选择）
- Peak位置（能量 E_centers）
- 各向异性参数（Betas）
- 能量展宽（sigmas）
- 分支比（branching_ratios）
- 高斯背景参数（bg_fraction, bg_mean, bg_sigma）

统计特征匹配：
- 径向分布 P(r)
- 角度分布 P(theta | r)
- 2D直方图
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import legendre
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings


# =============================================================================
# 物理常数
# =============================================================================
from scipy.constants import electron_mass, elementary_charge, atomic_mass
EV_TO_JOULE = elementary_charge
AMU_TO_KG = atomic_mass
ELECTRON_MASS_AMU = electron_mass / AMU_TO_KG


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class FitConfig:
    """拟合配置参数"""
    # VMI参数
    vmi_k: float = 0.01              # 速度到半径的转换系数 mm/(m/s)
    mass: float = ELECTRON_MASS_AMU  # 粒子质量 (amu)
    
    # 探测器参数
    psf_sigma: float = 0.0           # PSF展宽 (mm)
    dld_resolution: float = 0.0      # DLD量化分辨率 (mm)
    
    # 拟合参数范围
    E_min: float = 0.01              # 最小能量 (eV)
    E_max: float = 5.0               # 最大能量 (eV)
    beta_min: float = -1.0           # 最小beta
    beta_max: float = 2.0            # 最大beta
    sigma_min: float = 0.001         # 最小sigma (eV)
    sigma_max: float = 0.5           # 最大sigma (eV)
    
    # 背景参数范围
    bg_fraction_max: float = 0.3     # 最大背景比例
    bg_E_max: float = 1.0            # 背景能量上限 (eV)
    
    # 优化参数
    n_bins_radial: int = 200         # 径向分布bin数
    n_bins_angular: int = 36         # 角度分布bin数
    n_bins_2d: int = 64              # 2D直方图bin数
    
    # 权重
    weight_radial: float = 1.0       # 径向分布权重
    weight_angular: float = 0.5      # 角度分布权重
    weight_2d: float = 0.3           # 2D直方图权重
    
    # 密度估计方法: 'histogram', 'kde', 'bayesian_blocks'
    density_method: str = 'histogram'
    
    # KDE参数
    kde_bandwidth: str = 'scott'     # KDE带宽选择方法
    kde_n_points: int = 200          # KDE评估点数
    
    # Bayesian Blocks参数
    bb_p0: float = 0.05              # 假阳性概率 (越小越保守，block越少)
    bb_fitness: str = 'events'       # 适应度函数: 'events', 'regular_events', 'measures'


# =============================================================================
# Bayesian Blocks 实现
# =============================================================================
def bayesian_blocks(data: np.ndarray, p0: float = 0.05, 
                    fitness: str = 'events') -> np.ndarray:
    """
    Bayesian Blocks 自适应分箱算法
    
    基于 Scargle et al. (2013) 的算法实现
    
    Args:
        data: 1D数据数组
        p0: 假阳性概率，控制block数量（越小block越少）
        fitness: 适应度函数类型
            - 'events': 事件数据（默认，适合光子计数）
            - 'regular_events': 规则采样的事件
            - 'measures': 带误差的测量值
    
    Returns:
        bin edges数组
    """
    # 排序数据
    data = np.sort(data)
    N = len(data)
    
    if N < 4:
        return np.array([data[0], data[-1]])
    
    # 计算先验：ncp_prior = 4 - log(73.53 * p0 * (N^-0.478))
    # 这是Scargle推荐的经验公式
    ncp_prior = 4 - np.log(73.53 * p0 * (N ** -0.478))
    
    # 创建cell边界（每个数据点是一个cell）
    edges = np.concatenate([[data[0]], 0.5 * (data[1:] + data[:-1]), [data[-1]]])
    block_length = data[-1] - edges[:-1]
    
    # 动态规划
    # best[k] = 到第k个数据点的最优log-likelihood
    # last[k] = 最后一个block的起始位置
    best = np.zeros(N)
    last = np.zeros(N, dtype=int)
    
    for k in range(N):
        # 计算从每个可能起点r到k的fitness
        # 对于events fitness: N_k * log(N_k / T_k) - N_k
        # 其中N_k是block中的事件数，T_k是block宽度
        
        width = block_length[:k+1] - block_length[k+1] if k < N-1 else block_length[:k+1]
        width = np.maximum(width, 1e-10)  # 避免除零
        
        # 事件数（从r到k）
        count = np.arange(k+1, 0, -1)
        
        # Fitness函数
        if fitness == 'events':
            # Cash statistic for events
            fit = count * (np.log(count / width) - 1)
        elif fitness == 'regular_events':
            fit = count * np.log(count / width)
        else:
            fit = count * (np.log(count / width) - 1)
        
        # 加上之前的最优值和先验惩罚
        fit[1:] += best[:k] - ncp_prior
        fit[0] -= ncp_prior
        
        # 找最优分割点
        i_max = np.argmax(fit)
        best[k] = fit[i_max]
        last[k] = i_max
    
    # 回溯找到所有change points
    change_points = []
    idx = N - 1
    while idx >= 0:
        change_points.append(idx)
        idx = last[idx] - 1
    
    change_points = np.array(change_points[::-1])
    
    # 转换为bin edges
    bin_edges = np.concatenate([[edges[0]], edges[change_points + 1]])
    
    return bin_edges


def bayesian_blocks_histogram(data: np.ndarray, p0: float = 0.05,
                               x_range: Optional[Tuple[float, float]] = None,
                               n_eval: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Bayesian Blocks计算直方图，并插值到均匀网格
    
    Args:
        data: 输入数据
        p0: 假阳性概率
        x_range: 数据范围
        n_eval: 输出点数
        
    Returns:
        (x_eval, density) - 均匀网格上的密度估计
    """
    if len(data) < 10:
        # 数据太少，回退到普通直方图
        if x_range is None:
            x_range = (data.min(), data.max())
        x_eval = np.linspace(x_range[0], x_range[1], n_eval)
        counts, _ = np.histogram(data, bins=n_eval, range=x_range)
        density = counts / (counts.sum() + 1e-10)
        return x_eval, density
    
    # 计算Bayesian Blocks bin edges
    edges = bayesian_blocks(data, p0=p0)
    
    # 计算每个block的密度
    counts, _ = np.histogram(data, bins=edges)
    widths = np.diff(edges)
    density_blocks = counts / (widths * len(data) + 1e-10)
    
    # 插值到均匀网格
    if x_range is None:
        x_range = (edges[0], edges[-1])
    x_eval = np.linspace(x_range[0], x_range[1], n_eval)
    
    # 阶梯函数插值
    density = np.zeros(n_eval)
    for i, x in enumerate(x_eval):
        # 找到x所在的block
        idx = np.searchsorted(edges[1:], x)
        if idx < len(density_blocks):
            density[i] = density_blocks[idx]
    
    # 归一化
    density /= (np.trapz(density, x_eval) + 1e-10)
    
    return x_eval, density


# =============================================================================
# 前向模拟器（简化版，仅生成XY散点）
# =============================================================================
class ForwardSimulator:
    """
    前向模拟器：根据物理参数生成XY散点分布
    
    完全模仿 Abel_forward_simulation.py 的物理过程
    """
    
    def __init__(self, config: FitConfig):
        self.cfg = config
    
    def energy_to_velocity(self, E_eV: np.ndarray) -> np.ndarray:
        """能量转换为速度 (m/s)"""
        mass_kg = self.cfg.mass * AMU_TO_KG
        return np.sqrt(2.0 * np.maximum(E_eV, 1e-10) * EV_TO_JOULE / mass_kg)
    
    def velocity_to_radius(self, v: np.ndarray) -> np.ndarray:
        """速度转换为探测器半径 (mm)"""
        return self.cfg.vmi_k * v
    
    def energy_to_radius(self, E_eV: np.ndarray) -> np.ndarray:
        """能量直接转换为半径"""
        return self.velocity_to_radius(self.energy_to_velocity(E_eV))
    
    def sample_cos_theta(self, beta: float, n: int) -> np.ndarray:
        """
        从角度分布采样 cos(theta)
        
        PDF: f(x) = 1 + beta * P2(x), where P2(x) = (3x^2 - 1) / 2
        使用拒绝采样
        """
        if n == 0:
            return np.array([])
        
        # PDF最大值
        f_max = 1 + abs(beta) if beta >= 0 else 1 + abs(beta) / 2
        f_max = max(f_max, 1.0)
        
        samples = []
        while len(samples) < n:
            batch_size = 2 * (n - len(samples))
            x = np.random.uniform(-1, 1, batch_size)
            u = np.random.uniform(0, f_max, batch_size)
            P2 = (3 * x**2 - 1) / 2
            f_x = 1 + beta * P2
            valid = u < f_x
            samples.extend(x[valid])
        
        return np.array(samples[:n])
    
    def generate_peak_particles(self, E_center: float, sigma: float, 
                                 beta: float, n_particles: int) -> np.ndarray:
        """
        生成单个peak的粒子XY坐标
        
        Args:
            E_center: 中心能量 (eV)
            sigma: 能量展宽 (eV)
            beta: 各向异性参数
            n_particles: 粒子数
            
        Returns:
            (n_particles, 2) XY坐标数组
        """
        if n_particles == 0:
            return np.zeros((0, 2))
        
        # 1. 采样能量（高斯分布）
        E = np.random.normal(E_center, sigma, n_particles)
        E = np.maximum(E, 1e-10)  # 确保正能量
        
        # 2. 能量转半径
        r = self.energy_to_radius(E)
        
        # 3. 采样角度
        cos_theta = self.sample_cos_theta(beta, n_particles)
        phi = np.random.uniform(0, 2 * np.pi, n_particles)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # 4. 3D速度方向（假设极化沿Y轴）
        # 在极化坐标系中：z' = cos_theta, x' = sin_theta*cos_phi, y' = sin_theta*sin_phi
        # 旋转到实验室坐标系（极化沿Y）：
        # X_lab = x' = sin_theta * cos_phi
        # Y_lab = z' = cos_theta  
        # Z_lab = y' = sin_theta * sin_phi
        
        vx = sin_theta * np.cos(phi)
        vy = cos_theta  # 极化方向
        # vz = sin_theta * np.sin(phi)  # 投影时丢失
        
        # 5. 投影到XY平面
        X = r * vx
        Y = r * vy
        
        # 6. 添加PSF展宽
        if self.cfg.psf_sigma > 0:
            X += np.random.normal(0, self.cfg.psf_sigma, n_particles)
            Y += np.random.normal(0, self.cfg.psf_sigma, n_particles)
        
        # 7. DLD量化
        if self.cfg.dld_resolution > 0:
            X = np.round(X / self.cfg.dld_resolution) * self.cfg.dld_resolution
            Y = np.round(Y / self.cfg.dld_resolution) * self.cfg.dld_resolution
        
        return np.column_stack([X, Y])
    
    def generate_background(self, bg_E: float, bg_sigma: float, 
                            n_particles: int) -> np.ndarray:
        """
        生成各向同性高斯背景
        
        Args:
            bg_E: 背景中心能量 (eV)
            bg_sigma: 背景能量展宽 (eV)
            n_particles: 粒子数
        """
        if n_particles == 0:
            return np.zeros((0, 2))
        
        # 各向同性 (beta=0)
        return self.generate_peak_particles(bg_E, bg_sigma, 0.0, n_particles)
    
    def simulate(self, E_centers: List[float], sigmas: List[float],
                 betas: List[float], branching_ratios: List[float],
                 N_total: int,
                 bg_fraction: float = 0.0, bg_E: float = 0.1, 
                 bg_sigma: float = 0.05) -> np.ndarray:
        """
        完整前向模拟
        
        Args:
            E_centers: 各peak中心能量列表
            sigmas: 各peak能量展宽列表
            betas: 各peak各向异性参数列表
            branching_ratios: 各peak分支比列表（会自动归一化）
            N_total: 总粒子数
            bg_fraction: 背景占比
            bg_E: 背景中心能量
            bg_sigma: 背景能量展宽
            
        Returns:
            (N_total, 2) XY坐标数组
        """
        # 归一化分支比
        br = np.array(branching_ratios)
        br = br / br.sum()
        
        # 计算各成分粒子数
        N_signal = int(N_total * (1 - bg_fraction))
        N_bg = N_total - N_signal
        
        # 分配信号粒子到各peak
        peak_counts = np.random.multinomial(N_signal, br)
        
        # 生成各peak粒子
        all_particles = []
        for E, sigma, beta, n in zip(E_centers, sigmas, betas, peak_counts):
            particles = self.generate_peak_particles(E, sigma, beta, n)
            all_particles.append(particles)
        
        # 生成背景粒子
        if N_bg > 0:
            bg_particles = self.generate_background(bg_E, bg_sigma, N_bg)
            all_particles.append(bg_particles)
        
        # 合并并打乱
        all_xy = np.vstack(all_particles)
        np.random.shuffle(all_xy)
        
        return all_xy


# =============================================================================
# 统计特征提取
# =============================================================================
class StatisticsExtractor:
    """提取XY散点的统计特征，支持直方图、KDE和Bayesian Blocks三种方法"""
    
    def __init__(self, config: FitConfig):
        self.cfg = config
    
    def _kde_1d(self, data: np.ndarray, x_eval: np.ndarray, 
                bandwidth: Optional[float] = None) -> np.ndarray:
        """1D核密度估计"""
        from scipy.stats import gaussian_kde
        
        if len(data) < 2:
            return np.zeros_like(x_eval)
        
        try:
            if bandwidth is None:
                kde = gaussian_kde(data, bw_method=self.cfg.kde_bandwidth)
            else:
                kde = gaussian_kde(data, bw_method=bandwidth)
            density = kde(x_eval)
            density /= (np.trapz(density, x_eval) + 1e-10)
            return density
        except Exception:
            return self._histogram_1d(data, x_eval)
    
    def _histogram_1d(self, data: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        """1D直方图"""
        bins = np.linspace(x_eval[0], x_eval[-1], len(x_eval) + 1)
        counts, _ = np.histogram(data, bins=bins)
        counts = counts.astype(float)
        counts /= (counts.sum() + 1e-10)
        return counts
    
    def _compute_density_1d(self, data: np.ndarray, x_range: Tuple[float, float],
                            n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一的1D密度估计接口
        
        根据配置选择: histogram, kde, 或 bayesian_blocks
        """
        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        
        if len(data) < 5:
            return x_eval, np.zeros(n_points)
        
        method = self.cfg.density_method
        
        if method == 'bayesian_blocks':
            # Bayesian Blocks
            x_eval, density = bayesian_blocks_histogram(
                data, p0=self.cfg.bb_p0, 
                x_range=x_range, n_eval=n_points
            )
        elif method == 'kde':
            # KDE
            density = self._kde_1d(data, x_eval)
        else:
            # 默认直方图
            bins = np.linspace(x_range[0], x_range[1], n_points + 1)
            counts, _ = np.histogram(data, bins=bins)
            x_eval = 0.5 * (bins[:-1] + bins[1:])
            density = counts.astype(float)
            density /= (density.sum() + 1e-10)
        
        return x_eval, density
    
    def compute_radial_distribution(self, xy: np.ndarray, 
                                     r_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算径向分布 P(r)
        
        支持三种方法: histogram, kde, bayesian_blocks
        
        Returns:
            (r_centers, density) - 归一化的径向分布
        """
        r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
        
        if r_max is None:
            r_max = np.percentile(r, 99)
        
        n_points = self.cfg.kde_n_points if self.cfg.density_method != 'histogram' else self.cfg.n_bins_radial
        
        return self._compute_density_1d(r, (0.001, r_max), n_points)
    
    def compute_angular_distribution(self, xy: np.ndarray, 
                                      r_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算角度分布 P(theta)
        
        Args:
            r_range: 可选的半径范围筛选
            
        Returns:
            (theta_centers, density) - 归一化的角度分布
        """
        r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
        theta = np.arctan2(xy[:, 1], xy[:, 0])  # [-pi, pi]
        
        if r_range is not None:
            mask = (r >= r_range[0]) & (r <= r_range[1])
            theta = theta[mask]
        
        n_points = self.cfg.n_bins_angular
        theta_eval = np.linspace(-np.pi, np.pi, n_points)
        
        if len(theta) < 2:
            return theta_eval, np.zeros(n_points)
        
        # 角度分布用直方图或KDE（Bayesian Blocks对周期性数据不太适合）
        method = self.cfg.density_method
        if method == 'kde':
            density = self._kde_1d(theta, theta_eval, bandwidth=0.2)
        else:
            bins = np.linspace(-np.pi, np.pi, n_points + 1)
            counts, _ = np.histogram(theta, bins=bins)
            theta_eval = 0.5 * (bins[:-1] + bins[1:])
            density = counts.astype(float)
            density /= (density.sum() + 1e-10)
        
        return theta_eval, density
    
    def compute_radial_angular_2d(self, xy: np.ndarray,
                                   r_max: Optional[float] = None) -> np.ndarray:
        """
        计算2D (r, theta) 直方图/KDE
        这比简单的XY直方图更能捕捉角度依赖的径向结构
        """
        r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        
        if r_max is None:
            r_max = np.percentile(r, 99)
        
        n_r = self.cfg.n_bins_radial // 2
        n_theta = self.cfg.n_bins_angular
        
        method = self.cfg.density_method
        
        if method == 'kde' and len(r) > 50:
            # 2D KDE
            from scipy.stats import gaussian_kde
            try:
                data = np.vstack([r, theta])
                kde = gaussian_kde(data, bw_method='scott')
                
                r_grid = np.linspace(0.001, r_max, n_r)
                theta_grid = np.linspace(-np.pi, np.pi, n_theta)
                R, T = np.meshgrid(r_grid, theta_grid, indexing='ij')
                positions = np.vstack([R.ravel(), T.ravel()])
                
                density = kde(positions).reshape(n_r, n_theta)
                density /= (density.sum() + 1e-10)
                return density
            except Exception:
                pass  # 回退到直方图
        
        # 直方图方法（也用于bayesian_blocks，因为2D BB比较复杂）
        r_bins = np.linspace(0, r_max, n_r + 1)
        theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
        
        hist, _, _ = np.histogram2d(r, theta, bins=[r_bins, theta_bins])
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        
        return hist
    
    def compute_2d_histogram(self, xy: np.ndarray,
                              xy_range: Optional[float] = None) -> np.ndarray:
        """
        计算2D XY直方图/KDE
        
        Returns:
            归一化的2D密度
        """
        if xy_range is None:
            xy_range = np.percentile(np.abs(xy), 99)
        
        n_bins = self.cfg.n_bins_2d
        method = self.cfg.density_method
        
        if method == 'kde' and len(xy) > 50:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(xy.T, bw_method='scott')
                
                x_grid = np.linspace(-xy_range, xy_range, n_bins)
                y_grid = np.linspace(-xy_range, xy_range, n_bins)
                X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
                positions = np.vstack([X.ravel(), Y.ravel()])
                
                density = kde(positions).reshape(n_bins, n_bins)
                density /= (density.sum() + 1e-10)
                return density
            except Exception:
                pass  # 回退到直方图
        
        # 直方图方法
        bins = np.linspace(-xy_range, xy_range, n_bins + 1)
        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[bins, bins])
        
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        
        return hist
    
    def compute_angular_in_radial_bins(self, xy: np.ndarray, n_radial_bins: int = 5,
                                        r_max: Optional[float] = None) -> List[np.ndarray]:
        """
        在不同径向区间内计算角度分布
        这对于区分不同能量的peak的beta值很重要
        """
        r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
        
        if r_max is None:
            r_max = np.percentile(r, 99)
        
        r_edges = np.linspace(0, r_max, n_radial_bins + 1)
        
        angular_dists = []
        for i in range(n_radial_bins):
            r_range = (r_edges[i], r_edges[i+1])
            _, angular = self.compute_angular_distribution(xy, r_range)
            angular_dists.append(angular)
        
        return angular_dists
    
    def extract_all_features(self, xy: np.ndarray, 
                              r_max: Optional[float] = None) -> Dict[str, Any]:
        """提取所有统计特征"""
        r_centers, radial = self.compute_radial_distribution(xy, r_max)
        theta_centers, angular = self.compute_angular_distribution(xy)
        hist_2d = self.compute_2d_histogram(xy, r_max)
        r_theta_2d = self.compute_radial_angular_2d(xy, r_max)
        angular_in_bins = self.compute_angular_in_radial_bins(xy, n_radial_bins=5, r_max=r_max)
        
        return {
            'r_centers': r_centers,
            'radial': radial,
            'theta_centers': theta_centers,
            'angular': angular,
            'hist_2d': hist_2d,
            'r_theta_2d': r_theta_2d,
            'angular_in_bins': angular_in_bins,
            'r_max': r_max or np.percentile(np.sqrt(xy[:, 0]**2 + xy[:, 1]**2), 99)
        }


# =============================================================================
# 损失函数
# =============================================================================
class LossFunction:
    """计算模拟数据与观测数据之间的统计差异"""
    
    def __init__(self, config: FitConfig):
        self.cfg = config
    
    def radial_loss(self, radial_obs: np.ndarray, radial_sim: np.ndarray) -> float:
        """径向分布损失（Wasserstein距离 + L2）"""
        # Wasserstein距离捕捉整体形状
        w_dist = wasserstein_distance(radial_obs, radial_sim)
        # L2距离捕捉局部差异
        l2_dist = np.sqrt(np.mean((radial_obs - radial_sim)**2))
        return w_dist + 0.5 * l2_dist
    
    def angular_loss(self, angular_obs: np.ndarray, angular_sim: np.ndarray) -> float:
        """角度分布损失（L2距离）"""
        return np.sqrt(np.mean((angular_obs - angular_sim)**2))
    
    def hist2d_loss(self, hist_obs: np.ndarray, hist_sim: np.ndarray) -> float:
        """2D直方图损失（L2距离）"""
        return np.sqrt(np.mean((hist_obs - hist_sim)**2))
    
    def peak_structure_loss(self, radial_obs: np.ndarray, radial_sim: np.ndarray) -> float:
        """
        Peak结构损失：惩罚peak数量和位置的差异
        通过比较导数来检测peak
        """
        # 平滑后求导
        smooth_obs = gaussian_filter1d(radial_obs, sigma=3)
        smooth_sim = gaussian_filter1d(radial_sim, sigma=3)
        
        grad_obs = np.gradient(smooth_obs)
        grad_sim = np.gradient(smooth_sim)
        
        # 比较梯度
        return np.sqrt(np.mean((grad_obs - grad_sim)**2))
    
    def angular_in_bins_loss(self, angular_bins_obs: List[np.ndarray], 
                              angular_bins_sim: List[np.ndarray]) -> float:
        """不同径向区间内角度分布的损失"""
        total_loss = 0.0
        for obs, sim in zip(angular_bins_obs, angular_bins_sim):
            total_loss += np.sqrt(np.mean((obs - sim)**2))
        return total_loss / len(angular_bins_obs)
    
    def r_theta_2d_loss(self, r_theta_obs: np.ndarray, r_theta_sim: np.ndarray) -> float:
        """(r, theta) 2D直方图损失"""
        return np.sqrt(np.mean((r_theta_obs - r_theta_sim)**2))
    
    def total_loss(self, features_obs: Dict, features_sim: Dict) -> float:
        """总损失"""
        loss = 0.0
        
        loss += self.cfg.weight_radial * self.radial_loss(
            features_obs['radial'], features_sim['radial'])
        
        loss += self.cfg.weight_angular * self.angular_loss(
            features_obs['angular'], features_sim['angular'])
        
        loss += self.cfg.weight_2d * self.hist2d_loss(
            features_obs['hist_2d'], features_sim['hist_2d'])
        
        # 添加peak结构损失
        loss += 0.3 * self.peak_structure_loss(
            features_obs['radial'], features_sim['radial'])
        
        # 添加(r, theta) 2D损失
        if 'r_theta_2d' in features_obs and 'r_theta_2d' in features_sim:
            loss += 0.4 * self.r_theta_2d_loss(
                features_obs['r_theta_2d'], features_sim['r_theta_2d'])
        
        # 添加分区角度分布损失
        if 'angular_in_bins' in features_obs and 'angular_in_bins' in features_sim:
            loss += 0.5 * self.angular_in_bins_loss(
                features_obs['angular_in_bins'], features_sim['angular_in_bins'])
        
        return loss


# =============================================================================
# Peak检测器
# =============================================================================
class PeakDetector:
    """从径向分布中检测peak位置"""
    
    def __init__(self, config: FitConfig):
        self.cfg = config
    
    def detect_peaks(self, r_centers: np.ndarray, radial: np.ndarray, 
                     min_prominence: float = 0.002) -> List[Dict]:
        """
        检测径向分布中的peak
        
        Returns:
            List of dicts with 'r', 'height', 'prominence'
        """
        from scipy.signal import find_peaks
        
        # 平滑 - 使用较小的sigma以保留peak细节
        smooth = gaussian_filter1d(radial, sigma=1.5)
        
        # 找peak - 使用更宽松的参数以检测更多peaks
        peaks, properties = find_peaks(smooth, prominence=min_prominence, 
                                        distance=5, height=0.0003,
                                        width=1)
        
        results = []
        for i, peak_idx in enumerate(peaks):
            results.append({
                'r': r_centers[peak_idx],
                'height': smooth[peak_idx],
                'prominence': properties['prominences'][i] if 'prominences' in properties else 0
            })
        
        # 按prominence排序
        results.sort(key=lambda x: x['prominence'], reverse=True)
        
        return results
    
    def estimate_initial_params(self, xy_obs: np.ndarray, n_peaks: int,
                                 vmi_k: float, mass: float = ELECTRON_MASS_AMU) -> Dict:
        """
        从观测数据估计初始参数
        
        Args:
            xy_obs: 观测XY数据
            n_peaks: 期望的peak数量
            vmi_k: VMI转换系数
            mass: 粒子质量
            
        Returns:
            初始参数字典
        """
        # 使用更多bins以获得更好的peak检测
        cfg_detect = FitConfig(
            vmi_k=vmi_k,
            mass=mass,
            n_bins_radial=300,  # More bins for detection
            density_method='histogram'
        )
        extractor = StatisticsExtractor(cfg_detect)
        r_centers, radial = extractor.compute_radial_distribution(xy_obs)
        
        # 检测peaks - 使用较低的prominence阈值
        detected = self.detect_peaks(r_centers, radial, min_prominence=0.002)
        
        # 半径转能量
        def r_to_E(r):
            mass_kg = mass * AMU_TO_KG
            v = r / vmi_k
            E = 0.5 * mass_kg * v**2 / EV_TO_JOULE
            return E
        
        E_centers = []
        branching_ratios = []
        
        if len(detected) >= n_peaks:
            # 使用检测到的peaks（按prominence排序，取前n_peaks个）
            for i in range(n_peaks):
                E_centers.append(r_to_E(detected[i]['r']))
                branching_ratios.append(detected[i]['height'])
        else:
            # 不够peaks时，先用检测到的，再均匀填充
            r_max = np.percentile(np.sqrt(xy_obs[:, 0]**2 + xy_obs[:, 1]**2), 95)
            
            # 先添加检测到的peaks
            for p in detected:
                E_centers.append(r_to_E(p['r']))
                branching_ratios.append(p['height'])
            
            # 均匀填充剩余的peaks
            n_remaining = n_peaks - len(detected)
            for i in range(n_remaining):
                r = r_max * (i + 1) / (n_remaining + 1)
                E_centers.append(r_to_E(r))
                branching_ratios.append(0.1)
        
        # 归一化分支比
        total = sum(branching_ratios)
        branching_ratios = [b / total for b in branching_ratios]
        
        return {
            'E_centers': E_centers,
            'branching_ratios': branching_ratios,
            'detected_peaks': detected
        }


# =============================================================================
# 参数优化器
# =============================================================================
class ParameterOptimizer:
    """
    通过优化前向模拟参数来拟合观测数据
    
    支持：
    - 固定peak数量的优化
    - 自动peak数量选择（BIC准则）
    """
    
    def __init__(self, config: FitConfig):
        self.cfg = config
        self.simulator = ForwardSimulator(config)
        self.extractor = StatisticsExtractor(config)
        self.loss_fn = LossFunction(config)
        self.peak_detector = PeakDetector(config)
        
        # 缓存观测数据特征
        self._obs_features = None
        self._N_total = None
        self._xy_obs = None
    
    def set_observation(self, xy_obs: np.ndarray):
        """设置观测数据"""
        self._N_total = len(xy_obs)
        self._xy_obs = xy_obs
        self._obs_features = self.extractor.extract_all_features(xy_obs)
    
    def _params_to_arrays(self, params: np.ndarray, n_peaks: int) -> Dict:
        """
        将优化参数向量转换为物理参数
        
        参数布局 (每个peak 4个参数 + 3个背景参数):
        [E1, sigma1, beta1, br1, E2, sigma2, beta2, br2, ..., bg_frac, bg_E, bg_sigma]
        """
        idx = 0
        E_centers = []
        sigmas = []
        betas = []
        branching_ratios = []
        
        for _ in range(n_peaks):
            E_centers.append(params[idx])
            sigmas.append(params[idx + 1])
            betas.append(params[idx + 2])
            branching_ratios.append(params[idx + 3])
            idx += 4
        
        bg_fraction = params[idx]
        bg_E = params[idx + 1]
        bg_sigma = params[idx + 2]
        
        return {
            'E_centers': E_centers,
            'sigmas': sigmas,
            'betas': betas,
            'branching_ratios': branching_ratios,
            'bg_fraction': bg_fraction,
            'bg_E': bg_E,
            'bg_sigma': bg_sigma
        }
    
    def _get_bounds(self, n_peaks: int) -> List[Tuple[float, float]]:
        """获取参数边界"""
        bounds = []
        
        for _ in range(n_peaks):
            bounds.append((self.cfg.E_min, self.cfg.E_max))      # E
            bounds.append((self.cfg.sigma_min, self.cfg.sigma_max))  # sigma
            bounds.append((self.cfg.beta_min, self.cfg.beta_max))    # beta
            bounds.append((0.01, 1.0))                               # branching ratio
        
        # 背景参数
        bounds.append((0.0, self.cfg.bg_fraction_max))    # bg_fraction
        bounds.append((0.01, self.cfg.bg_E_max))          # bg_E
        bounds.append((0.01, 0.5))                        # bg_sigma
        
        return bounds
    
    def _objective(self, params: np.ndarray, n_peaks: int, 
                   n_sim_particles: int = 10000) -> float:
        """目标函数：计算损失"""
        try:
            phys_params = self._params_to_arrays(params, n_peaks)
            
            # 前向模拟
            xy_sim = self.simulator.simulate(
                E_centers=phys_params['E_centers'],
                sigmas=phys_params['sigmas'],
                betas=phys_params['betas'],
                branching_ratios=phys_params['branching_ratios'],
                N_total=n_sim_particles,
                bg_fraction=phys_params['bg_fraction'],
                bg_E=phys_params['bg_E'],
                bg_sigma=phys_params['bg_sigma']
            )
            
            # 提取特征
            features_sim = self.extractor.extract_all_features(
                xy_sim, r_max=self._obs_features['r_max'])
            
            # 计算损失
            loss = self.loss_fn.total_loss(self._obs_features, features_sim)
            
            return loss
            
        except Exception as e:
            return 1e10  # 返回大损失值
    
    def optimize_fixed_peaks(self, n_peaks: int, 
                              n_iterations: int = 50,
                              n_sim_particles: int = 10000,
                              verbose: bool = False) -> Tuple[Dict, float]:
        """
        Optimize with fixed number of peaks
        
        Args:
            n_peaks: number of peaks
            n_iterations: differential evolution iterations
            n_sim_particles: particles per simulation
            verbose: print progress
            
        Returns:
            (best params dict, best loss)
        """
        if self._obs_features is None:
            raise ValueError("Please call set_observation() first")
        
        bounds = self._get_bounds(n_peaks)
        
        # Use peak detection to estimate initial values
        init_params = self.peak_detector.estimate_initial_params(
            self._xy_obs, n_peaks, self.cfg.vmi_k, self.cfg.mass)
        
        if verbose:
            print(f"Optimizing {n_peaks} peaks, {len(bounds)} parameters")
            print(f"  Detected peaks: {len(init_params['detected_peaks'])}")
            print(f"  Initial E estimate: {[f'{E:.3f}' for E in init_params['E_centers']]} eV")
        
        # Build initial vector
        x0 = []
        for i in range(n_peaks):
            E_init = np.clip(init_params['E_centers'][i], self.cfg.E_min, self.cfg.E_max)
            x0.extend([
                E_init,                    # E
                0.05,                      # sigma
                0.0,                       # beta (neutral initial)
                init_params['branching_ratios'][i]  # br
            ])
        x0.extend([0.05, 0.1, 0.05])  # bg_fraction, bg_E, bg_sigma
        
        # Use differential evolution with initial seed
        result = differential_evolution(
            lambda p: self._objective(p, n_peaks, n_sim_particles),
            bounds=bounds,
            maxiter=n_iterations,
            seed=42,
            polish=True,
            disp=verbose,
            workers=1,
            popsize=10,
            tol=0.001,
            atol=0.0001,
            init='latinhypercube',
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        best_params = self._params_to_arrays(result.x, n_peaks)
        best_loss = result.fun
        
        return best_params, best_loss
    
    def optimize_auto_peaks(self, max_peaks: int = 5,
                             n_iterations: int = 100,
                             n_sim_particles: int = 10000,
                             verbose: bool = False) -> Tuple[Dict, float, int]:
        """
        Auto-select number of peaks using AIC/BIC criteria
        
        Args:
            max_peaks: maximum number of peaks
            n_iterations: iterations per model
            n_sim_particles: particles per simulation
            verbose: print progress
            
        Returns:
            (best params dict, best loss, optimal n_peaks)
        """
        # First detect peaks in data
        r_centers, radial = self.extractor.compute_radial_distribution(self._xy_obs)
        detected = self.peak_detector.detect_peaks(r_centers, radial)
        
        # Suggested peak range
        suggested_n = len(detected) if detected else 1
        min_peaks = max(1, suggested_n - 1)
        max_peaks = min(max_peaks, suggested_n + 2)
        
        if verbose:
            print(f"Detected {len(detected)} candidate peaks")
            print(f"Will try {min_peaks} to {max_peaks} peaks")
        
        results = []
        
        for n_peaks in range(min_peaks, max_peaks + 1):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Trying {n_peaks} peaks...")
            
            params, loss = self.optimize_fixed_peaks(
                n_peaks, n_iterations, n_sim_particles, verbose)
            
            # Calculate AIC and BIC
            n_params = 4 * n_peaks + 3
            n_data = self._N_total
            
            # Use negative log-likelihood approximation
            nll = n_data * np.log(loss + 1e-10)
            
            aic = 2 * n_params + 2 * nll
            bic = n_params * np.log(n_data) + 2 * nll
            
            results.append({
                'n_peaks': n_peaks,
                'params': params,
                'loss': loss,
                'aic': aic,
                'bic': bic
            })
            
            if verbose:
                print(f"  Loss: {loss:.6f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        # Select model with minimum AIC
        best = min(results, key=lambda x: x['aic'])
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Best model: {best['n_peaks']} peaks (AIC={best['aic']:.2f})")
        
        return best['params'], best['loss'], best['n_peaks']


# =============================================================================
# 高级优化器（使用多阶段策略）
# =============================================================================
class AdvancedOptimizer:
    """
    高级优化器：多阶段优化策略
    
    阶段1: 粗略搜索（少量粒子，快速）
    阶段2: 精细优化（更多粒子，更准确）
    阶段3: 最终验证（大量粒子）
    """
    
    def __init__(self, config: FitConfig):
        self.cfg = config
        self.optimizer = ParameterOptimizer(config)
    
    def fit(self, xy_obs: np.ndarray, 
            max_peaks: int = 5,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Complete fitting workflow
        
        Args:
            xy_obs: observed XY scatter data (N, 2)
            max_peaks: maximum number of peaks
            verbose: print progress
            
        Returns:
            dict with fitting results
        """
        self.optimizer.set_observation(xy_obs)
        N_total = len(xy_obs)
        
        if verbose:
            print(f"Observed data: {N_total} particles")
            print(f"Max peaks: {max_peaks}")
        
        # Stage 1: Coarse search
        if verbose:
            print("\n" + "="*60)
            print("Stage 1: Coarse Search")
            print("="*60)
        
        params_coarse, loss_coarse, n_peaks = self.optimizer.optimize_auto_peaks(
            max_peaks=max_peaks,
            n_iterations=50,
            n_sim_particles=min(5000, N_total // 4),
            verbose=verbose
        )
        
        # Stage 2: Fine optimization
        if verbose:
            print("\n" + "="*60)
            print(f"Stage 2: Fine Optimization ({n_peaks} peaks)")
            print("="*60)
        
        params_fine, loss_fine = self._refine_optimization(
            params_coarse, n_peaks, 
            n_sim_particles=min(15000, N_total),
            verbose=verbose
        )
        
        # Stage 3: Final validation
        if verbose:
            print("\n" + "="*60)
            print("Stage 3: Final Validation")
            print("="*60)
        
        final_loss = self._validate(params_fine, n_peaks, N_total, verbose)
        
        # Organize results
        result = {
            'n_peaks': n_peaks,
            'E_centers': params_fine['E_centers'],
            'sigmas': params_fine['sigmas'],
            'betas': params_fine['betas'],
            'branching_ratios': self._normalize_br(params_fine['branching_ratios']),
            'bg_fraction': params_fine['bg_fraction'],
            'bg_E': params_fine['bg_E'],
            'bg_sigma': params_fine['bg_sigma'],
            'final_loss': final_loss,
            'N_total': N_total
        }
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _refine_optimization(self, initial_params: Dict, n_peaks: int,
                              n_sim_particles: int, verbose: bool) -> Tuple[Dict, float]:
        """Fine optimization"""
        # Convert dict params to vector
        params_vec = []
        for i in range(n_peaks):
            params_vec.extend([
                initial_params['E_centers'][i],
                initial_params['sigmas'][i],
                initial_params['betas'][i],
                initial_params['branching_ratios'][i]
            ])
        params_vec.extend([
            initial_params['bg_fraction'],
            initial_params['bg_E'],
            initial_params['bg_sigma']
        ])
        
        bounds = self.optimizer._get_bounds(n_peaks)
        
        # Local optimization
        result = minimize(
            lambda p: self.optimizer._objective(p, n_peaks, n_sim_particles),
            x0=params_vec,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': verbose}
        )
        
        return self.optimizer._params_to_arrays(result.x, n_peaks), result.fun
    
    def _validate(self, params: Dict, n_peaks: int, 
                  N_total: int, verbose: bool) -> float:
        """Final validation"""
        simulator = ForwardSimulator(self.cfg)
        extractor = StatisticsExtractor(self.cfg)
        loss_fn = LossFunction(self.cfg)
        
        xy_sim = simulator.simulate(
            E_centers=params['E_centers'],
            sigmas=params['sigmas'],
            betas=params['betas'],
            branching_ratios=params['branching_ratios'],
            N_total=N_total,
            bg_fraction=params['bg_fraction'],
            bg_E=params['bg_E'],
            bg_sigma=params['bg_sigma']
        )
        
        features_sim = extractor.extract_all_features(
            xy_sim, r_max=self.optimizer._obs_features['r_max'])
        
        loss = loss_fn.total_loss(self.optimizer._obs_features, features_sim)
        
        if verbose:
            print(f"Validation loss: {loss:.6f}")
        
        return loss
    
    def _normalize_br(self, br: List[float]) -> List[float]:
        """归一化分支比"""
        total = sum(br)
        return [b / total for b in br]
    
    def _print_results(self, result: Dict):
        """Print results"""
        print("\n" + "="*60)
        print("Fitting Results")
        print("="*60)
        print(f"Number of peaks: {result['n_peaks']}")
        print(f"Total particles: {result['N_total']}")
        print(f"Final loss: {result['final_loss']:.6f}")
        print()
        
        for i in range(result['n_peaks']):
            print(f"Peak {i+1}:")
            print(f"  Energy: {result['E_centers'][i]:.4f} eV")
            print(f"  Sigma: {result['sigmas'][i]:.4f} eV")
            print(f"  Beta: {result['betas'][i]:.3f}")
            print(f"  Branching ratio: {result['branching_ratios'][i]:.3f}")
        
        print(f"\nBackground:")
        print(f"  Fraction: {result['bg_fraction']:.3f}")
        print(f"  Energy: {result['bg_E']:.4f} eV")
        print(f"  Sigma: {result['bg_sigma']:.4f} eV")


# =============================================================================
# 可视化
# =============================================================================
def visualize_fit_result(xy_obs: np.ndarray, result: Dict, 
                          config: FitConfig,
                          save_path: Optional[str] = None):
    """
    Visualize fitting results
    
    Compare observed data and fitted model:
    - 2D scatter distribution
    - Radial distribution
    - Angular distribution
    """
    import matplotlib.pyplot as plt
    
    # Generate fitted model scatter points
    simulator = ForwardSimulator(config)
    xy_fit = simulator.simulate(
        E_centers=result['E_centers'],
        sigmas=result['sigmas'],
        betas=result['betas'],
        branching_ratios=result['branching_ratios'],
        N_total=result['N_total'],
        bg_fraction=result['bg_fraction'],
        bg_E=result['bg_E'],
        bg_sigma=result['bg_sigma']
    )
    
    # Extract features
    extractor = StatisticsExtractor(config)
    r_max = np.percentile(np.sqrt(xy_obs[:, 0]**2 + xy_obs[:, 1]**2), 99)
    
    r_obs, radial_obs = extractor.compute_radial_distribution(xy_obs, r_max)
    r_fit, radial_fit = extractor.compute_radial_distribution(xy_fit, r_max)
    
    theta_obs, angular_obs = extractor.compute_angular_distribution(xy_obs)
    theta_fit, angular_fit = extractor.compute_angular_distribution(xy_fit)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Observed data 2D scatter
    ax = axes[0, 0]
    ax.scatter(xy_obs[:, 0], xy_obs[:, 1], s=0.1, alpha=0.3, c='blue')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Observed Data')
    ax.set_aspect('equal')
    ax.set_xlim(-r_max*1.1, r_max*1.1)
    ax.set_ylim(-r_max*1.1, r_max*1.1)
    
    # 2. Fitted model 2D scatter
    ax = axes[0, 1]
    ax.scatter(xy_fit[:, 0], xy_fit[:, 1], s=0.1, alpha=0.3, c='red')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Fitted Model')
    ax.set_aspect('equal')
    ax.set_xlim(-r_max*1.1, r_max*1.1)
    ax.set_ylim(-r_max*1.1, r_max*1.1)
    
    # 3. 2D histogram residual
    ax = axes[0, 2]
    bins = np.linspace(-r_max, r_max, 64)
    hist_obs, _, _ = np.histogram2d(xy_obs[:, 0], xy_obs[:, 1], bins=[bins, bins])
    hist_fit, _, _ = np.histogram2d(xy_fit[:, 0], xy_fit[:, 1], bins=[bins, bins])
    diff = hist_obs - hist_fit
    im = ax.imshow(diff.T, origin='lower', extent=[-r_max, r_max, -r_max, r_max],
                   cmap='RdBu', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Residual (Obs - Fit)')
    plt.colorbar(im, ax=ax)
    
    # 4. Radial distribution comparison
    ax = axes[1, 0]
    ax.plot(r_obs, radial_obs, 'b-', label='Observed', linewidth=2)
    ax.plot(r_fit, radial_fit, 'r--', label='Fitted', linewidth=2)
    ax.set_xlabel('Radius (mm)')
    ax.set_ylabel('Normalized Counts')
    ax.set_title('Radial Distribution')
    ax.legend()
    
    # Mark peak positions
    for i, E in enumerate(result['E_centers']):
        r_peak = simulator.energy_to_radius(np.array([E]))[0]
        ax.axvline(r_peak, color='green', linestyle=':', alpha=0.7,
                   label=f'Peak {i+1}: {E:.3f} eV' if i == 0 else f'Peak {i+1}: {E:.3f} eV')
    
    # 5. Angular distribution comparison
    ax = axes[1, 1]
    ax.plot(np.degrees(theta_obs), angular_obs, 'b-', label='Observed', linewidth=2)
    ax.plot(np.degrees(theta_fit), angular_fit, 'r--', label='Fitted', linewidth=2)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Normalized Counts')
    ax.set_title('Angular Distribution')
    ax.legend()
    
    # 6. Parameter table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [['Parameter', 'Value']]
    table_data.append(['N_peaks', str(result['n_peaks'])])
    for i in range(result['n_peaks']):
        table_data.append([f'E_{i+1} (eV)', f"{result['E_centers'][i]:.4f}"])
        table_data.append([f'sigma_{i+1} (eV)', f"{result['sigmas'][i]:.4f}"])
        table_data.append([f'beta_{i+1}', f"{result['betas'][i]:.3f}"])
        table_data.append([f'BR_{i+1}', f"{result['branching_ratios'][i]:.3f}"])
    table_data.append(['BG fraction', f"{result['bg_fraction']:.3f}"])
    table_data.append(['Loss', f"{result['final_loss']:.6f}"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Fitted Parameters')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


# =============================================================================
# 便捷接口
# =============================================================================
def fit_xy_data(xy_obs: np.ndarray,
                vmi_k: float = 0.01,
                psf_sigma: float = 0.0,
                dld_resolution: float = 0.0,
                max_peaks: int = 5,
                E_max: float = 5.0,
                verbose: bool = True,
                visualize: bool = True,
                save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    拟合XY散点数据的便捷接口
    
    Args:
        xy_obs: 观测的XY散点数据 (N, 2)
        vmi_k: VMI转换系数 mm/(m/s)
        psf_sigma: PSF展宽 (mm)
        dld_resolution: DLD量化分辨率 (mm)
        max_peaks: 最大peak数量
        E_max: 最大能量 (eV)
        verbose: 是否打印进度
        visualize: 是否可视化结果
        save_path: 图像保存路径
        
    Returns:
        拟合结果字典
    """
    config = FitConfig(
        vmi_k=vmi_k,
        psf_sigma=psf_sigma,
        dld_resolution=dld_resolution,
        E_max=E_max
    )
    
    optimizer = AdvancedOptimizer(config)
    result = optimizer.fit(xy_obs, max_peaks=max_peaks, verbose=verbose)
    
    if visualize:
        visualize_fit_result(xy_obs, result, config, save_path)
    
    return result


# =============================================================================
# Test Code
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Abel Backward Reconstruction via Forward Fitting (X1)")
    print("="*70)
    
    # Import forward simulator to generate test data
    from Abel_forward_simulation import Config, run_simulation
    
    # Set true parameters (2 peaks for quick test)
    true_E_centers = [0.5, 1.5]
    true_Betas = [2.0, -0.5]
    true_branching_ratios = [0.4, 0.6]
    
    # Calculate VMI coefficient
    E_max = max(true_E_centers) * 1.5
    r_max_mm = 20.0
    vmi_k = Config.calculate_vmi_k(E_max, r_max_mm)
    
    print(f"\nTrue parameters:")
    print(f"  Energy: {true_E_centers} eV")
    print(f"  Beta: {true_Betas}")
    print(f"  Branching ratios: {true_branching_ratios}")
    print(f"  VMI coefficient: {vmi_k:.4e} mm/(m/s)")
    
    # =========================================================================
    # Test Bayesian Blocks
    # =========================================================================
    print("\n" + "="*70)
    print("Testing Bayesian Blocks on sample data")
    print("="*70)
    
    # Generate sample radial data
    np.random.seed(42)
    r_sample = np.concatenate([
        np.random.normal(5, 0.5, 1000),   # Peak 1
        np.random.normal(12, 0.8, 1500),  # Peak 2
        np.random.uniform(0, 15, 200)     # Background
    ])
    
    # Compare methods
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(r_sample, bins=50, density=True, alpha=0.7, label='Histogram')
    ax.set_title('Fixed-width Histogram (50 bins)')
    ax.set_xlabel('r')
    ax.legend()
    
    # Bayesian Blocks
    ax = axes[1]
    bb_edges = bayesian_blocks(r_sample, p0=0.05)
    ax.hist(r_sample, bins=bb_edges, density=True, alpha=0.7, color='orange', label='Bayesian Blocks')
    ax.set_title(f'Bayesian Blocks ({len(bb_edges)-1} bins)')
    ax.set_xlabel('r')
    ax.legend()
    
    # Comparison on uniform grid
    ax = axes[2]
    x_eval, density_bb = bayesian_blocks_histogram(r_sample, p0=0.05, x_range=(0, 15), n_eval=200)
    ax.plot(x_eval, density_bb, 'orange', linewidth=2, label='Bayesian Blocks')
    
    # KDE for comparison
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(r_sample)
    ax.plot(x_eval, kde(x_eval) / np.trapz(kde(x_eval), x_eval), 'g--', linewidth=2, label='KDE')
    ax.set_title('Comparison on uniform grid')
    ax.set_xlabel('r')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('bayesian_blocks_demo.png', dpi=150)
    print("Saved bayesian_blocks_demo.png")
    plt.close()
    
    # =========================================================================
    # Test 1: Clean data with Bayesian Blocks
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Clean Data with Bayesian Blocks")
    print("="*70)
    
    config_clean = Config(
        E_centers=true_E_centers,
        Betas=true_Betas,
        branching_ratios=true_branching_ratios,
        N_events=15000,
        vmi_k=vmi_k,
        sigma_laser=0.03,
        T_beam=0.0,
        tau_lifetimes=0.0,
        vol_sigma=(0.0, 0.0, 0.0),
        polarization_vec=[0, 1, 0],
        img_res=512,
        pixel_size=0.1,
        psf_fwhm=0.2,
        dld_resolution=0.01,
        dark_rate=0.0,
        readout_sigma=0.0,
        readout_offset=0.0,
        bg_rate=0.03,
        bg_energy=0.15,
        bg_sigma=0.08,
    )
    
    print(f"\nGenerating {config_clean.N_events} particles (clean)...")
    xy_clean, _ = run_simulation(config_clean, add_noise=False, output_mode='xy_dld')
    print(f"  Data shape: {xy_clean.shape}")
    
    # Fit with Bayesian Blocks
    print("\n--- Fitting with Bayesian Blocks ---")
    fit_config_bb = FitConfig(
        vmi_k=vmi_k,
        psf_sigma=config_clean.psf_sigma,
        dld_resolution=config_clean.dld_resolution,
        E_max=E_max,
        density_method='bayesian_blocks',
        bb_p0=0.05
    )
    
    optimizer_bb = AdvancedOptimizer(fit_config_bb)
    result_bb = optimizer_bb.fit(xy_clean, max_peaks=3, verbose=True)
    
    # =========================================================================
    # Test 2: Noisy data
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Noisy Data with Bayesian Blocks")
    print("="*70)
    
    config_noisy = Config(
        E_centers=true_E_centers,
        Betas=true_Betas,
        branching_ratios=true_branching_ratios,
        N_events=15000,
        vmi_k=vmi_k,
        sigma_laser=0.05,
        T_beam=10.0,
        tau_lifetimes=0.0,
        vol_sigma=(0.3, 0.3, 0.3),
        polarization_vec=[0, 1, 0],
        img_res=512,
        pixel_size=0.1,
        psf_fwhm=0.4,
        dld_resolution=0.02,
        dark_rate=0.0,
        readout_sigma=0.0,
        readout_offset=0.0,
        bg_rate=0.08,
        bg_energy=0.2,
        bg_sigma=0.12,
    )
    
    print(f"\nGenerating {config_noisy.N_events} particles (noisy)...")
    xy_noisy, _ = run_simulation(config_noisy, add_noise=False, output_mode='xy_dld')
    
    # Add additional position noise
    position_noise = np.random.normal(0, 0.08, xy_noisy.shape)
    xy_noisy = xy_noisy + position_noise
    print(f"  Data shape: {xy_noisy.shape}")
    print(f"  Added 0.08 mm position noise")
    
    # Fit noisy data with Bayesian Blocks
    print("\n--- Fitting noisy data with Bayesian Blocks ---")
    fit_config_noisy = FitConfig(
        vmi_k=vmi_k,
        psf_sigma=config_noisy.psf_sigma,
        dld_resolution=config_noisy.dld_resolution,
        E_max=E_max,
        density_method='bayesian_blocks',
        bb_p0=0.05
    )
    
    optimizer_noisy = AdvancedOptimizer(fit_config_noisy)
    result_noisy = optimizer_noisy.fit(xy_noisy, max_peaks=3, verbose=True)
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON: Clean vs Noisy (both with Bayesian Blocks)")
    print("="*70)
    
    def print_comparison(result, label):
        sorted_idx = np.argsort(result['E_centers'])
        print(f"\n{label}:")
        for i, idx in enumerate(sorted_idx[:len(true_E_centers)]):
            true_E = true_E_centers[i]
            fit_E = result['E_centers'][idx]
            true_b = true_Betas[i]
            fit_b = result['betas'][idx]
            E_err = abs(fit_E - true_E) / true_E * 100
            print(f"  Peak {i+1}: E={fit_E:.3f}eV (err={E_err:.1f}%), beta={fit_b:.2f} (true={true_b:.2f})")
    
    print_comparison(result_bb, "Clean data (Bayesian Blocks)")
    print_comparison(result_noisy, "Noisy data (Bayesian Blocks)")
    
    # Visualize
    print("\n--- Generating visualizations ---")
    visualize_fit_result(xy_clean, result_bb, fit_config_bb, save_path="fit_result_clean_bb.png")
    visualize_fit_result(xy_noisy, result_noisy, fit_config_noisy, save_path="fit_result_noisy_bb.png")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)