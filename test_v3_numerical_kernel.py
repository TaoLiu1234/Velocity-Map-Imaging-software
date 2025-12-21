"""
测试 V3 数值投影核的效果

比较三种模式：
1. skip_forward_fit=True（只用 Phase 2 + β 校正）
2. skip_forward_fit=False, use_numerical_kernel=False（解析高斯）
3. skip_forward_fit=False, use_numerical_kernel=True（数值投影核）
"""
import numpy as np
import sys
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction_v3 import AbelReconstructorV3


def run_test(n_events, mode='phase2_only', seed=42):
    """运行单次测试
    
    Args:
        n_events: 事件计数
        mode: 'phase2_only', 'analytic_gaussian', 'numerical_kernel'
        seed: 随机种子
    """
    config = Config(
        E_centers=[0.5, 1.0, 2.0],
        Betas=[1.5, 0.0, -0.5],
        branching_ratios=[0.3, 0.5, 0.2],
        vmi_k=Config.calculate_vmi_k(E_max_eV=2.0, r_max_mm=20.0),
        sigma_laser=0.015,
        N_events=int(n_events),
        img_res=512,
        pixel_size=0.1,
        readout_offset=100.0,
        psf_fwhm=0.0,
    )
    
    # 计算真值
    true_values = []
    for E, beta, br in zip(config.E_centers, config.Betas, config.branching_ratios):
        r_px = config.get_expected_radius(E) / config.pixel_size
        sigma_r = r_px * config.sigma_laser / (2 * E)
        true_values.append({'r': r_px, 'sigma': sigma_r, 'beta': beta, 'br': br})
    
    # 生成模拟图像
    np.random.seed(seed)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    image, _ = run_simulation(config, add_noise=True, add_background=False)
    sys.stdout = old_stdout
    
    # 预处理
    image_corrected = image - config.readout_offset
    
    # V3 重建
    reconstructor = AbelReconstructorV3(config=config, polarization_axis='vertical')
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    if mode == 'phase2_only':
        final_v3, metadata = reconstructor.reconstruct(
            image_corrected, verbose=False, 
            skip_forward_fit=True, use_polar_fit=False
        )
    elif mode == 'analytic_gaussian':
        # 需要修改 fitter 使用解析高斯
        final_v3, metadata = reconstructor.reconstruct(
            image_corrected, verbose=False, 
            skip_forward_fit=False, use_polar_fit=False
        )
    elif mode == 'numerical_kernel':
        final_v3, metadata = reconstructor.reconstruct(
            image_corrected, verbose=False, 
            skip_forward_fit=False, use_polar_fit=False
        )
    
    sys.stdout = old_stdout
    
    # 匹配 peaks
    matched = []
    used_indices = []
    for i, tv in enumerate(true_values):
        best_idx, best_dist = -1, 20
        for j, p in enumerate(final_v3):
            if j not in used_indices:
                dist = abs(p['r'] - tv['r'])
                if dist < best_dist:
                    best_dist, best_idx = dist, j
        if best_idx >= 0:
            used_indices.append(best_idx)
            p = final_v3[best_idx]
            
            # 计算百分比误差
            r_pct = 100 * (p['r'] - tv['r']) / tv['r']
            sigma_pct = 100 * (p['sigma'] - tv['sigma']) / tv['sigma']
            
            if abs(tv['beta']) > 0.01:
                beta_pct = 100 * (p['beta'] - tv['beta']) / tv['beta']
            else:
                beta_pct = 100 * (p['beta'] - tv['beta'])
            
            br_pct = 100 * (p['br'] - tv['br']) / tv['br']
            
            matched.append({
                'peak': i+1,
                'r_pct': r_pct,
                'sigma_pct': sigma_pct,
                'beta_pct': beta_pct,
                'br_pct': br_pct,
            })
    
    # 计算平均误差
    if matched:
        avg_r = np.mean([abs(m['r_pct']) for m in matched])
        avg_sigma = np.mean([abs(m['sigma_pct']) for m in matched])
        avg_beta = np.mean([abs(m['beta_pct']) for m in matched])
        avg_br = np.mean([abs(m['br_pct']) for m in matched])
    else:
        avg_r = avg_sigma = avg_beta = avg_br = float('inf')
    
    return {
        'r': avg_r,
        'sigma': avg_sigma,
        'beta': avg_beta,
        'br': avg_br,
        'converged': metadata.converged if hasattr(metadata, 'converged') else True
    }


def main():
    print("=" * 80)
    print("V3 Numerical Kernel Test: Comparing Different Forward Models")
    print("=" * 80)
    
    print("\nTRUE VALUES:")
    print("  Peak 1: r=100.0 px, σ=1.50 px, β=+1.50, BR=0.30")
    print("  Peak 2: r=141.4 px, σ=1.06 px, β= 0.00, BR=0.50")
    print("  Peak 3: r=200.0 px, σ=0.75 px, β=-0.50, BR=0.20")
    
    n_events = 1e6
    
    print(f"\nN_events = {n_events:.0e}")
    print("-" * 80)
    
    # 测试 Phase 2 only
    print("\n1. Phase 2 Only (skip_forward_fit=True):")
    result = run_test(n_events, mode='phase2_only')
    print(f"   r: {result['r']:.2f}%, σ: {result['sigma']:.2f}%, β: {result['beta']:.2f}%, BR: {result['br']:.2f}%")
    
    # 测试数值核前向拟合
    print("\n2. Forward Fit with Numerical Kernel:")
    result = run_test(n_events, mode='numerical_kernel')
    print(f"   r: {result['r']:.2f}%, σ: {result['sigma']:.2f}%, β: {result['beta']:.2f}%, BR: {result['br']:.2f}%")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
