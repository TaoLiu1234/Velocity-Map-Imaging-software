"""
测试 V3.5 在不同事件计数下各指标与真值的百分比误差

V3.5 改进（响应审查官第三轮批评）：
1. 废除减法预处理 - subtract_background=False
2. 算符融合 - use_fused_transform=True
3. 模板超采样 - 数值核自动使用
4. 放宽边界约束 - wide_bounds=True

测试内容：
1. Phase 0: 背景估算（不减法）
2. Phase 1: 计数守恒（算符融合）
3. Phase 2-4: 参数提取精度
4. Phase 5: β-BR 解耦
5. 最终：与真值对比的百分比误差
"""
import numpy as np
import sys
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction_v3 import (
    AbelReconstructorV3, 
    DataCleaner, 
    PolarTransformer,
    SeedFinder,
    ForwardFitter,
    BRCalculator
)


def run_phase_tests(image, config):
    """运行所有 Phase 检验点"""
    results = {}
    
    # Phase 0: DataCleaner
    cleaner = DataCleaner()
    cleaned, sigma_bg = cleaner.clean(image, auto_center=True)
    verify_0 = cleaner.verify_cleaning(cleaned)
    
    results['test_0_1'] = {
        'name': 'Background mean zero',
        'value': verify_0['bg_mean'],
        'threshold': 1e-6,
        'passed': abs(verify_0['bg_mean']) < 1e-6
    }
    
    results['test_0_2'] = {
        'name': 'Background std',
        'value': verify_0['bg_std'],
        'expected': sigma_bg,
        'passed': True  # 只记录，不判断
    }
    
    # Phase 1: PolarTransformer
    transformer = PolarTransformer()
    center = cleaner.center
    polar = transformer.transform(cleaned, center)
    verify_1 = transformer.verify_conservation(cleaned, polar)
    
    results['test_1_1'] = {
        'name': 'Sum conservation',
        'sum_cart': verify_1['sum_cartesian'],
        'sum_polar': verify_1['sum_polar'],
        'relative_error': verify_1['relative_error'],
        'threshold': 1e-6,
        'passed': verify_1['passed']
    }
    
    return results, cleaned, sigma_bg, center, polar, transformer.theta_grid


def run_test(n_events, seed=42):
    """运行单次测试"""
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
    image_corrected = image - config.readout_offset  # 不裁剪负值
    
    # 运行 Phase 检验
    phase_results, cleaned, sigma_bg, center, polar, theta_grid = run_phase_tests(
        image_corrected, config
    )
    
    # V3.5 重建
    # 审查官建议：Phase 2 的 Abel 逆变换结果已经很准确
    # 使用 skip_forward_fit=True 直接使用 Phase 2 结果
    reconstructor = AbelReconstructorV3(config=config, polarization_axis='vertical')
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    # V3.5: 使用 Phase 2 结果 + 算符融合
    final_v3, metadata = reconstructor.reconstruct(
        image_corrected, 
        verbose=False, 
        skip_forward_fit=True,  # 使用 Phase 2 结果
        enforce_circularity=True,
        use_fused_transform=True,  # 算符融合
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
            
            # beta 特殊处理：真值为 0 时用绝对误差
            if abs(tv['beta']) > 0.01:
                beta_pct = 100 * (p['beta'] - tv['beta']) / tv['beta']
            else:
                beta_pct = 100 * (p['beta'] - tv['beta'])
            
            br_pct = 100 * (p['br'] - tv['br']) / tv['br']
            
            matched.append({
                'peak': i+1,
                'r': p['r'], 'r_true': tv['r'], 'r_pct': r_pct,
                'sigma': p['sigma'], 'sigma_true': tv['sigma'], 'sigma_pct': sigma_pct,
                'beta': p['beta'], 'beta_true': tv['beta'], 'beta_pct': beta_pct,
                'br': p['br'], 'br_true': tv['br'], 'br_pct': br_pct,
            })
    
    return matched, phase_results, metadata


def main():
    """主测试函数"""
    # 测试事件计数
    event_counts = [5e5, 1e6, 5e6, 1e7]
    
    print("=" * 120)
    print("V3.5 Performance: Percentage Error vs True Values")
    print("=" * 120)
    print("\nV3.5 IMPROVEMENTS:")
    print("  1. Operator fusion (single resampling)")
    print("  2. Template oversampling (10x at Abel singularity)")
    print("  3. Wide bounds (r±10px, σ±50%, β free)")
    print("  4. Poisson MLE with background offset model")
    
    print("\nTRUE VALUES:")
    print("  Peak 1: r=100.0 px, σ=1.50 px, β=+1.50, BR=0.30")
    print("  Peak 2: r=141.4 px, σ=1.06 px, β= 0.00, BR=0.50")
    print("  Peak 3: r=200.0 px, σ=0.75 px, β=-0.50, BR=0.20")
    
    # Phase 检验结果
    print("\n" + "=" * 120)
    print("PHASE VERIFICATION TESTS")
    print("=" * 120)
    
    # 用 1e6 事件做 Phase 检验
    _, phase_results, _ = run_test(1e6)
    
    print(f"\nTest 0.1 (Background mean zero):")
    t = phase_results['test_0_1']
    print(f"  Value: {t['value']:.2e}, Threshold: {t['threshold']:.0e}, Passed: {t['passed']}")
    
    print(f"\nTest 1.1 (Sum conservation):")
    t = phase_results['test_1_1']
    print(f"  Sum (Cartesian): {t['sum_cart']:.2f}")
    print(f"  Sum (Polar): {t['sum_polar']:.2f}")
    print(f"  Relative error: {t['relative_error']:.2e}, Threshold: {t['threshold']:.0e}, Passed: {t['passed']}")
    
    # 参数误差测试
    print("\n" + "=" * 120)
    print("PARAMETER ACCURACY TESTS")
    print("=" * 120)
    print(f"{'N_events':<10} | {'Peak':<5} | {'r (px)':<20} | {'σ (px)':<20} | {'β':<20} | {'BR':<20}")
    print(f"{'':10} | {'':5} | {'est → err%':<20} | {'est → err%':<20} | {'est → err%':<20} | {'est → err%':<20}")
    print("-" * 120)
    
    summary = []
    
    for n_events in event_counts:
        results, _, metadata = run_test(n_events)
        
        for i, m in enumerate(results):
            if i == 0:
                n_str = f"{n_events:.0e}"
            else:
                n_str = ""
            
            r_str = f"{m['r']:.1f} → {m['r_pct']:+.2f}%"
            s_str = f"{m['sigma']:.2f} → {m['sigma_pct']:+.1f}%"
            b_str = f"{m['beta']:.3f} → {m['beta_pct']:+.1f}%"
            br_str = f"{m['br']:.3f} → {m['br_pct']:+.1f}%"
            
            print(f"{n_str:<10} | {m['peak']:<5} | {r_str:<20} | {s_str:<20} | {b_str:<20} | {br_str:<20}")
        
        # 平均百分比误差
        avg_r = np.mean([abs(m['r_pct']) for m in results])
        avg_s = np.mean([abs(m['sigma_pct']) for m in results])
        avg_b = np.mean([abs(m['beta_pct']) for m in results])
        avg_br = np.mean([abs(m['br_pct']) for m in results])
        
        print(f"{'Avg |err|':<10} | {'':5} | {avg_r:>6.2f}%{'':<13} | {avg_s:>6.1f}%{'':<13} | {avg_b:>6.1f}%{'':<13} | {avg_br:>6.1f}%{'':<13}")
        print("-" * 120)
        
        summary.append({
            'n_events': n_events,
            'avg_r': avg_r,
            'avg_s': avg_s,
            'avg_b': avg_b,
            'avg_br': avg_br,
            'converged': metadata.converged if hasattr(metadata, 'converged') else metadata.get('converged', False)
        })
    
    # 汇总
    print("\n" + "=" * 120)
    print("SUMMARY: Average Absolute Percentage Error by Event Count")
    print("=" * 120)
    print(f"{'N_events':<12} | {'r (%)':<10} | {'σ (%)':<10} | {'β (%)':<10} | {'BR (%)':<10} | {'Converged':<10}")
    print("-" * 80)
    
    for s in summary:
        print(f"{s['n_events']:<12.0e} | {s['avg_r']:<10.2f} | {s['avg_s']:<10.1f} | {s['avg_b']:<10.1f} | {s['avg_br']:<10.1f} | {s['converged']}")
    
    print("\n" + "=" * 120)
    print("V3.5 TEST COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
