"""
Comprehensive Method Comparison Test
=====================================
测试不同参数组合下各方法的性能：
1. 不同计数水平 (1e5, 1e6, 1e7)
2. 不同峰宽 (窄峰、正常峰、宽峰)
3. 不同峰间距 (近、正常、远)
4. 不同BR分布 (均匀、偏斜、极端)
5. 不同β值 (各向同性、中等、强各向异性)

统计分析：
- 每个配置运行多次
- 计算均值、标准差、中位数
- 生成热力图和箱线图
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# Import reconstruction methods
from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image as reconstruct_v1_func
from Abel_backward_reconstruction_v2 import reconstruct_vmi_image_v2
from Abel_backward_reconstruction_v3 import AbelReconstructorV3
import abel

@dataclass
class TestConfig:
    """测试配置"""
    name: str
    E_centers: List[float]  # eV
    sigmas: List[float]     # eV (能量展宽)
    betas: List[float]
    BRs: List[float]
    n_events: int = int(1e6)
    
    def __post_init__(self):
        # 归一化BR
        total = sum(self.BRs)
        self.BRs = [br/total for br in self.BRs]


# =============================================================================
# 测试配置定义
# =============================================================================

def get_test_configs() -> List[TestConfig]:
    """生成测试配置列表"""
    configs = []
    
    # =========================================================================
    # 1. 基准配置 (Baseline)
    # =========================================================================
    configs.append(TestConfig(
        name="baseline",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[1.5, 0.0, -0.5],
        BRs=[0.3, 0.5, 0.2]
    ))
    
    # =========================================================================
    # 2. 峰宽变化
    # =========================================================================
    # 2a. 窄峰 (σ = 0.008 eV)
    configs.append(TestConfig(
        name="narrow_peaks",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.008, 0.008, 0.008],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # 2b. 宽峰 (σ = 0.030 eV)
    configs.append(TestConfig(
        name="wide_peaks",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.030, 0.030, 0.030],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # 2c. 混合宽度
    configs.append(TestConfig(
        name="mixed_width",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.008, 0.015, 0.030],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # =========================================================================
    # 3. 峰间距变化
    # =========================================================================
    # 3a. 近峰 (能量接近)
    configs.append(TestConfig(
        name="close_peaks",
        E_centers=[0.8, 1.0, 1.2],
        sigmas=[0.015, 0.015, 0.015],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # 3b. 远峰 (能量分散)
    configs.append(TestConfig(
        name="far_peaks",
        E_centers=[0.3, 1.0, 3.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))

    # =========================================================================
    # 4. BR分布变化
    # =========================================================================
    # 4a. 均匀BR
    configs.append(TestConfig(
        name="uniform_BR",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # 4b. 偏斜BR (一个峰占主导)
    configs.append(TestConfig(
        name="skewed_BR",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.8, 0.15, 0.05]
    ))
    
    # 4c. 极端BR (一个很弱)
    configs.append(TestConfig(
        name="extreme_BR",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.49, 0.49, 0.02]
    ))
    
    # =========================================================================
    # 5. β值变化
    # =========================================================================
    # 5a. 全部各向同性
    configs.append(TestConfig(
        name="isotropic",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[0.0, 0.0, 0.0],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # 5b. 强各向异性
    configs.append(TestConfig(
        name="strong_anisotropy",
        E_centers=[0.5, 1.0, 2.0],
        sigmas=[0.015, 0.015, 0.015],
        betas=[2.0, -1.0, 1.8],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # =========================================================================
    # 6. 挑战性配置
    # =========================================================================
    # 6a. 近峰 + 宽峰 (重叠)
    configs.append(TestConfig(
        name="overlapping",
        E_centers=[0.9, 1.0, 1.1],
        sigmas=[0.025, 0.025, 0.025],
        betas=[1.0, 0.0, -0.5],
        BRs=[0.33, 0.34, 0.33]
    ))
    
    # 6b. 两峰配置
    configs.append(TestConfig(
        name="two_peaks",
        E_centers=[0.5, 2.0],
        sigmas=[0.015, 0.015],
        betas=[1.5, -0.5],
        BRs=[0.6, 0.4]
    ))
    
    # 6c. 四峰配置
    configs.append(TestConfig(
        name="four_peaks",
        E_centers=[0.3, 0.7, 1.2, 2.0],
        sigmas=[0.012, 0.015, 0.015, 0.020],
        betas=[1.5, 0.5, -0.3, -0.8],
        BRs=[0.2, 0.3, 0.3, 0.2]
    ))
    
    return configs


# =============================================================================
# 重建方法封装
# =============================================================================

def run_v1(image, config):
    """运行V1重建"""
    try:
        t0 = time.time()
        params, metadata = reconstruct_v1_func(image, config, verbose=False)
        dt = time.time() - t0
        return params, dt
    except Exception as e:
        return None, 0

def run_v2(image, config):
    """运行V2重建"""
    try:
        t0 = time.time()
        params, metadata = reconstruct_vmi_image_v2(image - config.readout_offset, config, verbose=False)
        dt = time.time() - t0
        return params, dt
    except Exception as e:
        return None, 0

def run_v3(image, config):
    """运行V3重建"""
    try:
        t0 = time.time()
        reconstructor = AbelReconstructorV3(image.shape[0])
        reconstructor.calibrate_from_config(config)
        result = reconstructor.reconstruct(
            image - config.readout_offset,
            skip_forward_fit=True,
            verbose=False
        )
        params = [p.to_dict() for p in result.peaks]
        dt = time.time() - t0
        return params, dt
    except Exception as e:
        return None, 0

def run_rbasex(image, config):
    """运行rBasex重建"""
    try:
        t0 = time.time()
        image_centered = image - config.readout_offset
        recon, distr = abel.Transform(
            image_centered, method='rbasex', direction='inverse',
            transform_options={'reg': 'pos'}
        ).transform
        
        # 提取参数
        radial = np.mean(recon, axis=0)
        n = len(radial)
        radial = radial[n//2:]
        
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(radial, height=np.max(radial)*0.05, distance=5)
        
        params = []
        for pk in peaks_idx:
            if pk < 15:
                continue
            # 简单高斯拟合
            r_range = 10
            r_start = max(0, pk - r_range)
            r_end = min(len(radial), pk + r_range)
            local = radial[r_start:r_end]
            r_local = np.arange(r_start, r_end)
            
            if len(local) > 3 and np.max(local) > 0:
                half_max = np.max(local) / 2
                above = local > half_max
                if np.sum(above) > 1:
                    fwhm = np.sum(above)
                    sigma = fwhm / 2.355
                else:
                    sigma = 2.0
            else:
                sigma = 2.0
            
            params.append({
                'r': float(pk),
                'sigma': float(sigma),
                'amp': float(radial[pk]),
                'beta': 0.0,
                'branching_ratio': 0.0
            })
        
        # 计算BR
        if len(params) > 0:
            total = sum(p['amp'] * p['sigma'] * p['r']**2 for p in params)
            if total > 0:
                for p in params:
                    p['branching_ratio'] = p['amp'] * p['sigma'] * p['r']**2 / total
        
        dt = time.time() - t0
        return params, dt
    except Exception as e:
        return None, 0


def generate_image(sim_config: Config) -> np.ndarray:
    """生成VMI图像"""
    image, metadata = run_simulation(sim_config, add_noise=True, add_background=True)
    return image


# =============================================================================
# 误差计算
# =============================================================================

def match_peaks(true_params: List[Dict], recon_params: List[Dict], 
                sim_config: Config, test_config: TestConfig) -> List[Tuple[Dict, Dict]]:
    """匹配真实峰和重建峰"""
    if recon_params is None or len(recon_params) == 0:
        return []
    
    # 计算真实峰的r位置
    true_r = []
    for i, E in enumerate(test_config.E_centers):
        # r = sqrt(E / vmi_k) / pixel_size
        r_mm = np.sqrt(E / sim_config.vmi_k)
        r_px = r_mm / sim_config.pixel_size
        true_r.append(r_px)
    
    matches = []
    used = set()
    
    for i, r_true in enumerate(true_r):
        best_match = None
        best_dist = float('inf')
        
        for j, p in enumerate(recon_params):
            if j in used:
                continue
            dist = abs(p['r'] - r_true)
            if dist < best_dist and dist < 30:  # 30像素容差
                best_dist = dist
                best_match = j
        
        if best_match is not None:
            used.add(best_match)
            # 计算真实sigma (像素)
            # sigma_r = sigma_E / (2 * vmi_k * sqrt(E)) / pixel_size
            E = test_config.E_centers[i]
            sigma_E = test_config.sigmas[i]
            sigma_r_mm = sigma_E / (2 * sim_config.vmi_k * np.sqrt(E))
            sigma_r_px = sigma_r_mm / sim_config.pixel_size
            
            true_p = {
                'r': r_true,
                'sigma': sigma_r_px,
                'beta': test_config.betas[i],
                'br': test_config.BRs[i]
            }
            matches.append((true_p, recon_params[best_match]))
    
    return matches

def calculate_errors(matches: List[Tuple[Dict, Dict]], n_true: int) -> Dict:
    """计算误差"""
    if len(matches) == 0:
        return {
            'r_err': np.nan, 'sigma_err': np.nan, 
            'beta_err': np.nan, 'br_err': np.nan,
            'detection_rate': 0.0
        }
    
    r_errs = []
    sigma_errs = []
    beta_errs = []
    br_errs = []
    
    for true_p, recon_p in matches:
        # r误差 (%)
        r_errs.append(abs(recon_p['r'] - true_p['r']) / true_p['r'] * 100)
        
        # sigma误差 (%)
        sigma_recon = recon_p.get('sigma', recon_p.get('sigma_phys', 1.0))
        if true_p['sigma'] > 0.1:
            sigma_errs.append(abs(sigma_recon - true_p['sigma']) / true_p['sigma'] * 100)
        
        # beta误差 (% of range [-1, 2])
        beta_recon = recon_p.get('beta', 0.0)
        beta_errs.append(abs(beta_recon - true_p['beta']) / 3.0 * 100)
        
        # BR误差 (%)
        br_recon = recon_p.get('branching_ratio', recon_p.get('br', 0.0))
        if true_p['br'] > 0.01:  # 只计算BR > 1%的峰
            br_errs.append(abs(br_recon - true_p['br']) / true_p['br'] * 100)
    
    return {
        'r_err': np.mean(r_errs) if r_errs else np.nan,
        'sigma_err': np.mean(sigma_errs) if sigma_errs else np.nan,
        'beta_err': np.mean(beta_errs) if beta_errs else np.nan,
        'br_err': np.mean(br_errs) if br_errs else np.nan,
        'detection_rate': len(matches) / n_true * 100
    }


# =============================================================================
# 主测试函数
# =============================================================================

def run_single_test(test_config: TestConfig, sim_config: Config,
                    n_trials: int = 3) -> Dict:
    """运行单个测试配置"""
    results = {method: [] for method in ['v1', 'v2', 'v3', 'rbasex']}
    times = {method: [] for method in ['v1', 'v2', 'v3', 'rbasex']}
    
    for trial in range(n_trials):
        # 生成图像
        np.random.seed(42 + trial)
        image = generate_image(sim_config)
        
        # 运行各方法
        methods = [
            ('v1', run_v1),
            ('v2', run_v2),
            ('v3', run_v3),
            ('rbasex', run_rbasex)
        ]
        
        for name, func in methods:
            params, dt = func(image, sim_config)
            times[name].append(dt)
            
            if params is not None:
                matches = match_peaks(None, params, sim_config, test_config)
                errors = calculate_errors(matches, len(test_config.E_centers))
                results[name].append(errors)
            else:
                results[name].append({
                    'r_err': np.nan, 'sigma_err': np.nan,
                    'beta_err': np.nan, 'br_err': np.nan,
                    'detection_rate': 0.0
                })
    
    # 汇总统计
    summary = {}
    for method in ['v1', 'v2', 'v3', 'rbasex']:
        method_results = results[method]
        summary[method] = {
            'r_err_mean': np.nanmean([r['r_err'] for r in method_results]),
            'r_err_std': np.nanstd([r['r_err'] for r in method_results]),
            'sigma_err_mean': np.nanmean([r['sigma_err'] for r in method_results]),
            'sigma_err_std': np.nanstd([r['sigma_err'] for r in method_results]),
            'beta_err_mean': np.nanmean([r['beta_err'] for r in method_results]),
            'beta_err_std': np.nanstd([r['beta_err'] for r in method_results]),
            'br_err_mean': np.nanmean([r['br_err'] for r in method_results]),
            'br_err_std': np.nanstd([r['br_err'] for r in method_results]),
            'detection_rate': np.mean([r['detection_rate'] for r in method_results]),
            'time_mean': np.mean(times[method])
        }
    
    return summary

def create_sim_config(test_config: TestConfig) -> Config:
    """从测试配置创建模拟配置"""
    # 计算合适的 vmi_k 使最大能量峰落在 ~200 像素处
    E_max = max(test_config.E_centers)
    r_max_mm = 10.0  # 目标半径 10mm
    pixel_size = 0.05  # mm/pixel
    
    # 使用 Config 的静态方法计算 vmi_k
    vmi_k = Config.calculate_vmi_k(E_max, r_max_mm, mass_amu=127.0)
    
    return Config(
        img_res=512,
        E_centers=test_config.E_centers,
        Betas=test_config.betas,
        branching_ratios=test_config.BRs,
        sigma_laser=np.mean(test_config.sigmas),
        N_events=test_config.n_events,
        readout_offset=100.0,
        readout_sigma=10.0,
        psf_fwhm=0.1,
        pixel_size=pixel_size,
        vmi_k=vmi_k,
        mass=127.0
    )


# =============================================================================
# 可视化
# =============================================================================

def plot_heatmap(all_results: Dict, metric: str, title: str, filename: str):
    """绘制热力图"""
    configs = list(all_results.keys())
    methods = ['v1', 'v2', 'v3', 'rbasex']
    
    data = np.zeros((len(configs), len(methods)))
    for i, cfg in enumerate(configs):
        for j, method in enumerate(methods):
            val = all_results[cfg][method].get(f'{metric}_mean', np.nan)
            data[i, j] = val if not np.isnan(val) else 100
    
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['V1', 'V2', 'V3.5', 'rBasex'])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    
    # 添加数值标注
    for i in range(len(configs)):
        for j in range(len(methods)):
            val = data[i, j]
            color = 'white' if val > 25 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label=f'{title} (%)')
    ax.set_title(f'{title} by Configuration and Method')
    ax.set_xlabel('Method')
    ax.set_ylabel('Configuration')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_summary_bars(all_results: Dict, filename: str):
    """绘制汇总柱状图"""
    methods = ['v1', 'v2', 'v3', 'rbasex']
    metrics = ['r_err', 'sigma_err', 'beta_err', 'br_err']
    metric_names = ['Position', 'Width', 'Beta', 'BR']
    
    # 计算每个方法在所有配置上的平均误差
    avg_errors = {method: {metric: [] for metric in metrics} for method in methods}
    
    for cfg, results in all_results.items():
        for method in methods:
            for metric in metrics:
                val = results[method].get(f'{metric}_mean', np.nan)
                if not np.isnan(val):
                    avg_errors[method][metric].append(val)
    
    # 计算均值
    final_avg = {method: {} for method in methods}
    for method in methods:
        for metric in metrics:
            vals = avg_errors[method][metric]
            final_avg[method][metric] = np.mean(vals) if vals else np.nan
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        vals = [final_avg[m][metric] for m in methods]
        bars = ax.bar(['V1', 'V2', 'V3.5', 'rBasex'], vals, color=colors)
        ax.set_ylabel(f'{name} Error (%)')
        ax.set_title(f'Average {name} Error')
        
        # 标注最佳
        min_idx = np.nanargmin(vals)
        bars[min_idx].set_edgecolor('gold')
        bars[min_idx].set_linewidth(3)
        
        # 添加数值
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Overall Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_boxplots(all_results: Dict, filename: str):
    """绘制箱线图"""
    methods = ['v1', 'v2', 'v3', 'rbasex']
    method_labels = ['V1', 'V2', 'V3.5', 'rBasex']
    metrics = ['r_err', 'sigma_err', 'beta_err', 'br_err']
    metric_names = ['Position Error (%)', 'Width Error (%)', 'Beta Error (%)', 'BR Error (%)']
    
    # 收集所有数据
    data = {method: {metric: [] for metric in metrics} for method in methods}
    
    for cfg, results in all_results.items():
        for method in methods:
            for metric in metrics:
                val = results[method].get(f'{metric}_mean', np.nan)
                if not np.isnan(val):
                    data[method][metric].append(val)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        box_data = [data[m][metric] for m in methods]
        
        bp = ax.boxplot(box_data, labels=method_labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(name)
        ax.set_title(f'{name} Distribution')
        ax.grid(True, alpha=0.3)
        
        # 添加均值点
        means = [np.mean(d) if d else np.nan for d in box_data]
        ax.scatter(range(1, 5), means, color='red', marker='D', s=50, zorder=5, label='Mean')
    
    plt.suptitle('Error Distribution Across All Configurations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def print_statistical_summary(all_results: Dict):
    """打印统计摘要"""
    methods = ['v1', 'v2', 'v3', 'rbasex']
    metrics = ['r_err', 'sigma_err', 'beta_err', 'br_err']
    
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    # 收集数据
    data = {method: {metric: [] for metric in metrics} for method in methods}
    for cfg, results in all_results.items():
        for method in methods:
            for metric in metrics:
                val = results[method].get(f'{metric}_mean', np.nan)
                if not np.isnan(val):
                    data[method][metric].append(val)
    
    # 打印表格
    print(f"\n{'Method':<10} | {'Metric':<12} | {'Mean':>8} | {'Std':>8} | {'Median':>8} | {'Min':>8} | {'Max':>8}")
    print("-"*80)
    
    for method in methods:
        for metric in metrics:
            vals = data[method][metric]
            if vals:
                print(f"{method.upper():<10} | {metric:<12} | {np.mean(vals):>8.2f} | {np.std(vals):>8.2f} | "
                      f"{np.median(vals):>8.2f} | {np.min(vals):>8.2f} | {np.max(vals):>8.2f}")
        print("-"*80)
    
    # 胜率统计
    print("\n" + "="*80)
    print("WIN RATE (Best method for each configuration)")
    print("="*80)
    
    wins = {method: {metric: 0 for metric in metrics} for method in methods}
    total_configs = len(all_results)
    
    for cfg, results in all_results.items():
        for metric in metrics:
            best_val = float('inf')
            best_method = None
            for method in methods:
                val = results[method].get(f'{metric}_mean', np.nan)
                if not np.isnan(val) and val < best_val:
                    best_val = val
                    best_method = method
            if best_method:
                wins[best_method][metric] += 1
    
    print(f"\n{'Method':<10} | {'Position':>10} | {'Width':>10} | {'Beta':>10} | {'BR':>10} | {'Total':>10}")
    print("-"*70)
    for method in methods:
        total = sum(wins[method].values())
        print(f"{method.upper():<10} | {wins[method]['r_err']:>10} | {wins[method]['sigma_err']:>10} | "
              f"{wins[method]['beta_err']:>10} | {wins[method]['br_err']:>10} | {total:>10}")
    
    # 找出总冠军
    total_wins = {m: sum(wins[m].values()) for m in methods}
    champion = max(total_wins, key=total_wins.get)
    print(f"\n🏆 Overall Champion: {champion.upper()} with {total_wins[champion]} wins out of {total_configs * 4}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE METHOD COMPARISON TEST")
    print("="*80)
    
    # 获取测试配置
    test_configs = get_test_configs()
    print(f"\nTotal test configurations: {len(test_configs)}")
    
    # 测试参数 - 简化版本
    n_events_levels = [int(1e6)]  # 只测试1e6
    n_trials = 1  # 每个配置运行1次
    
    all_results = {}
    
    for n_events in n_events_levels:
        print(f"\n{'='*80}")
        print(f"Testing with N_events = {n_events:.0e}")
        print("="*80)
        
        for test_cfg in test_configs:
            test_cfg.n_events = n_events
            config_name = f"{test_cfg.name}_{n_events:.0e}"
            
            print(f"\n  Config: {config_name}")
            print(f"    E = {test_cfg.E_centers}")
            print(f"    σ = {test_cfg.sigmas}")
            print(f"    β = {test_cfg.betas}")
            print(f"    BR = {[f'{br:.2f}' for br in test_cfg.BRs]}")
            
            # 创建模拟配置
            sim_config = create_sim_config(test_cfg)
            
            # 运行测试
            summary = run_single_test(test_cfg, sim_config, n_trials=n_trials)
            all_results[config_name] = summary
            
            # 打印结果
            print(f"\n    {'Method':<10} | {'r(%)':>8} | {'σ(%)':>8} | {'β(%)':>8} | {'BR(%)':>8} | {'Det%':>6}")
            print("    " + "-"*60)
            for method in ['v1', 'v2', 'v3', 'rbasex']:
                s = summary[method]
                print(f"    {method.upper():<10} | {s['r_err_mean']:>8.2f} | {s['sigma_err_mean']:>8.2f} | "
                      f"{s['beta_err_mean']:>8.2f} | {s['br_err_mean']:>8.2f} | {s['detection_rate']:>6.0f}")
    
    # 生成可视化
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    plot_heatmap(all_results, 'r_err', 'Position Error', 'comprehensive_r_heatmap.png')
    plot_heatmap(all_results, 'sigma_err', 'Width Error', 'comprehensive_sigma_heatmap.png')
    plot_heatmap(all_results, 'beta_err', 'Beta Error', 'comprehensive_beta_heatmap.png')
    plot_heatmap(all_results, 'br_err', 'BR Error', 'comprehensive_br_heatmap.png')
    plot_summary_bars(all_results, 'comprehensive_summary.png')
    plot_boxplots(all_results, 'comprehensive_boxplots.png')
    
    print("  Saved: comprehensive_r_heatmap.png")
    print("  Saved: comprehensive_sigma_heatmap.png")
    print("  Saved: comprehensive_beta_heatmap.png")
    print("  Saved: comprehensive_br_heatmap.png")
    print("  Saved: comprehensive_summary.png")
    print("  Saved: comprehensive_boxplots.png")
    
    # 统计摘要
    print_statistical_summary(all_results)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
