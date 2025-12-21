"""
详细因素分析测试
================
测试以下因素对重建精度的影响：
1. 单峰: 峰宽度(sigma_laser)、峰位置(半径大小)
2. 多峰分离: 峰宽度、峰位置
3. 多峰接近: 峰宽度、峰位置
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image as reconstruct_v1
from Abel_backward_reconstruction_v2 import reconstruct_vmi_image_v2 as reconstruct_v2
from Abel_backward_reconstruction_v3 import AbelReconstructorV3

# =============================================================================
# 测试配置
# =============================================================================

# 峰宽度由 sigma_laser 控制 (能量展宽)
SIGMA_LEVELS = {
    'narrow': 0.005,   # 窄峰
    'medium': 0.015,   # 中等
    'wide': 0.030,     # 宽峰
}

# 峰位置由能量决定 (E -> r)
POSITION_LEVELS = {
    'inner': 0.3,    # 内圈 (小半径)
    'middle': 1.0,   # 中圈
    'outer': 2.5,    # 外圈 (大半径)
}

COUNT_LEVELS = [int(1e5), int(1e6)]
METHODS = ['V1', 'V2', 'V3']

# =============================================================================
# 辅助函数
# =============================================================================
def create_config(E_centers, Betas, BRs, sigma_laser, n_events):
    E_max = max(E_centers)
    img_res = 512
    pixel_size = 0.05
    r_max_mm = img_res * pixel_size * 0.4
    vmi_k = Config.calculate_vmi_k(E_max, r_max_mm, mass_amu=127.0)
    
    return Config(
        img_res=img_res, E_centers=E_centers, Betas=Betas,
        branching_ratios=BRs, sigma_laser=sigma_laser, N_events=n_events,
        readout_offset=100.0, readout_sigma=10.0, psf_fwhm=0.1,
        pixel_size=pixel_size, vmi_k=vmi_k, mass=127.0
    )

def get_true_r(config):
    return [config.get_expected_radius(E) / config.pixel_size for E in config.E_centers]

def match_peaks(true_r, true_beta, true_br, recon_params):
    if recon_params is None or len(recon_params) == 0:
        return None
    
    n_true = len(true_r)
    errors = {'r': [], 'beta': [], 'br': []}
    matched = 0
    used = set()
    
    for i in range(n_true):
        best_j, best_dist = None, float('inf')
        for j, p in enumerate(recon_params):
            if j in used:
                continue
            dist = abs(p['r'] - true_r[i])
            if dist < best_dist and dist < 30:
                best_dist, best_j = dist, j
        
        if best_j is not None:
            used.add(best_j)
            matched += 1
            p = recon_params[best_j]
            errors['r'].append(abs(p['r'] - true_r[i]) / true_r[i] * 100)
            errors['beta'].append(abs(p.get('beta', 0) - true_beta[i]) / 3.0 * 100)
            br_recon = p.get('branching_ratio', p.get('br', 0))
            if true_br[i] > 0.02:
                errors['br'].append(abs(br_recon - true_br[i]) / true_br[i] * 100)
    
    if matched == 0:
        return None
    
    return {
        'r_err': np.mean(errors['r']),
        'beta_err': np.mean(errors['beta']),
        'br_err': np.mean(errors['br']) if errors['br'] else np.nan,
        'det_rate': matched / n_true * 100,
        'n_det': matched, 'n_true': n_true
    }

def run_v1(image, config):
    try:
        params, _ = reconstruct_v1(image - config.readout_offset, config, verbose=False)
        return params
    except:
        return None

def run_v2(image, config):
    try:
        params, _ = reconstruct_v2(image - config.readout_offset, config, verbose=False)
        return params
    except:
        return None

def run_v3(image, config):
    try:
        reconstructor = AbelReconstructorV3(config=config)
        params, _ = reconstructor.reconstruct(image - config.readout_offset, skip_forward_fit=True, verbose=False)
        return [{'r': p['r'], 'beta': p['beta'], 'branching_ratio': p.get('br', 0)} for p in params]
    except:
        return None

METHOD_FUNCS = {'V1': run_v1, 'V2': run_v2, 'V3': run_v3}

def run_test(E_centers, Betas, BRs, sigma_laser, n_events, test_name):
    """运行单个测试并返回结果"""
    config = create_config(E_centers, Betas, BRs, sigma_laser, n_events)
    true_r = get_true_r(config)
    np.random.seed(42)
    image, _ = run_simulation(config)
    
    results = {}
    for name, func in METHOD_FUNCS.items():
        params = func(image, config)
        results[name] = match_peaks(true_r, Betas, BRs, params)
    
    return results, true_r

def print_results(results, test_name):
    """打印测试结果"""
    print(f"\n{test_name}")
    print(f"  {'Method':<6} | {'r(%)':>7} | {'β(%)':>7} | {'BR(%)':>7} | {'Det':>5}")
    print("  " + "-"*42)
    for m in METHODS:
        e = results[m]
        if e:
            br_str = f"{e['br_err']:>7.2f}" if not np.isnan(e['br_err']) else "    N/A"
            print(f"  {m:<6} | {e['r_err']:>7.2f} | {e['beta_err']:>7.2f} | {br_str} | {e['n_det']}/{e['n_true']}")
        else:
            print(f"  {m:<6} | {'FAIL':>7} | {'FAIL':>7} | {'FAIL':>7} | 0/?")

# =============================================================================
# 测试1: 单峰 - 峰宽度和位置的影响
# =============================================================================
def test_single_peak():
    print("\n" + "="*70)
    print("TEST 1: SINGLE PEAK - Width and Position Effects")
    print("="*70)
    
    all_results = {}
    n_events = int(1e6)
    
    for pos_name, E in POSITION_LEVELS.items():
        for width_name, sigma in SIGMA_LEVELS.items():
            test_name = f"Single: {pos_name} ({E}eV), {width_name} (σ={sigma})"
            results, true_r = run_test(
                E_centers=[E], Betas=[1.0], BRs=[1.0],
                sigma_laser=sigma, n_events=n_events, test_name=test_name
            )
            all_results[(pos_name, width_name)] = results
            print_results(results, test_name)
            print(f"  True r: {true_r[0]:.1f} px")
    
    # 汇总分析
    print("\n--- Summary: Single Peak ---")
    print("\nPosition effect (averaged over widths):")
    for pos_name in POSITION_LEVELS:
        print(f"\n  {pos_name}:")
        for m in METHODS:
            r_errs = [all_results[(pos_name, w)][m]['r_err'] 
                     for w in SIGMA_LEVELS if all_results[(pos_name, w)][m]]
            if r_errs:
                print(f"    {m}: r_err={np.mean(r_errs):.2f}%")
    
    print("\nWidth effect (averaged over positions):")
    for width_name in SIGMA_LEVELS:
        print(f"\n  {width_name}:")
        for m in METHODS:
            r_errs = [all_results[(p, width_name)][m]['r_err'] 
                     for p in POSITION_LEVELS if all_results[(p, width_name)][m]]
            if r_errs:
                print(f"    {m}: r_err={np.mean(r_errs):.2f}%")

# =============================================================================
# 测试2: 多峰分离 - 峰宽度和位置的影响
# =============================================================================
def test_separated_peaks():
    print("\n" + "="*70)
    print("TEST 2: SEPARATED PEAKS - Width and Position Effects")
    print("="*70)
    
    # 分离的峰配置 (能量间隔大)
    SEPARATED_CONFIGS = {
        'inner_region': [0.2, 0.5, 0.9],      # 内圈区域
        'middle_region': [0.5, 1.0, 1.8],     # 中圈区域
        'outer_region': [1.0, 2.0, 3.5],      # 外圈区域
        'full_range': [0.3, 1.0, 2.5],        # 全范围
    }
    
    all_results = {}
    n_events = int(1e6)
    
    for region_name, E_centers in SEPARATED_CONFIGS.items():
        for width_name, sigma in SIGMA_LEVELS.items():
            test_name = f"Separated: {region_name}, {width_name} (σ={sigma})"
            results, true_r = run_test(
                E_centers=E_centers, 
                Betas=[1.5, 0.0, -0.5], 
                BRs=[0.33, 0.34, 0.33],
                sigma_laser=sigma, n_events=n_events, test_name=test_name
            )
            all_results[(region_name, width_name)] = results
            print_results(results, test_name)
            print(f"  True r: {[f'{r:.1f}' for r in true_r]} px")
    
    # 汇总分析
    print("\n--- Summary: Separated Peaks ---")
    print("\nRegion effect (averaged over widths):")
    for region_name in SEPARATED_CONFIGS:
        print(f"\n  {region_name}:")
        for m in METHODS:
            det_rates = [all_results[(region_name, w)][m]['det_rate'] 
                        for w in SIGMA_LEVELS if all_results[(region_name, w)][m]]
            r_errs = [all_results[(region_name, w)][m]['r_err'] 
                     for w in SIGMA_LEVELS if all_results[(region_name, w)][m]]
            if det_rates:
                print(f"    {m}: det={np.mean(det_rates):.0f}%, r_err={np.mean(r_errs):.2f}%")
    
    print("\nWidth effect (averaged over regions):")
    for width_name in SIGMA_LEVELS:
        print(f"\n  {width_name}:")
        for m in METHODS:
            det_rates = [all_results[(r, width_name)][m]['det_rate'] 
                        for r in SEPARATED_CONFIGS if all_results[(r, width_name)][m]]
            r_errs = [all_results[(r, width_name)][m]['r_err'] 
                     for r in SEPARATED_CONFIGS if all_results[(r, width_name)][m]]
            if det_rates:
                print(f"    {m}: det={np.mean(det_rates):.0f}%, r_err={np.mean(r_errs):.2f}%")

# =============================================================================
# 测试3: 多峰接近 - 峰宽度和位置的影响
# =============================================================================
def test_close_peaks():
    print("\n" + "="*70)
    print("TEST 3: CLOSE PEAKS - Width and Position Effects")
    print("="*70)
    
    # 接近的峰配置 (能量间隔小)
    # 间隔比例: 约10%的能量差
    CLOSE_CONFIGS = {
        'inner_close': [0.27, 0.30, 0.33],    # 内圈接近
        'middle_close': [0.90, 1.00, 1.10],   # 中圈接近
        'outer_close': [2.25, 2.50, 2.75],    # 外圈接近
    }
    
    all_results = {}
    n_events = int(1e6)
    
    for region_name, E_centers in CLOSE_CONFIGS.items():
        for width_name, sigma in SIGMA_LEVELS.items():
            test_name = f"Close: {region_name}, {width_name} (σ={sigma})"
            results, true_r = run_test(
                E_centers=E_centers, 
                Betas=[1.0, 0.0, -0.5], 
                BRs=[0.33, 0.34, 0.33],
                sigma_laser=sigma, n_events=n_events, test_name=test_name
            )
            all_results[(region_name, width_name)] = results
            print_results(results, test_name)
            print(f"  True r: {[f'{r:.1f}' for r in true_r]} px")
    
    # 汇总分析
    print("\n--- Summary: Close Peaks ---")
    print("\nRegion effect (averaged over widths):")
    for region_name in CLOSE_CONFIGS:
        print(f"\n  {region_name}:")
        for m in METHODS:
            det_rates = [all_results[(region_name, w)][m]['det_rate'] 
                        for w in SIGMA_LEVELS if all_results[(region_name, w)][m]]
            r_errs = [all_results[(region_name, w)][m]['r_err'] 
                     for w in SIGMA_LEVELS if all_results[(region_name, w)][m]]
            if det_rates:
                print(f"    {m}: det={np.mean(det_rates):.0f}%, r_err={np.mean(r_errs):.2f}%")
    
    print("\nWidth effect (averaged over regions):")
    for width_name in SIGMA_LEVELS:
        print(f"\n  {width_name}:")
        for m in METHODS:
            det_rates = [all_results[(r, width_name)][m]['det_rate'] 
                        for r in CLOSE_CONFIGS if all_results[(r, width_name)][m]]
            r_errs = [all_results[(r, width_name)][m]['r_err'] 
                     for r in CLOSE_CONFIGS if all_results[(r, width_name)][m]]
            if det_rates:
                print(f"    {m}: det={np.mean(det_rates):.0f}%, r_err={np.mean(r_errs):.2f}%")

# =============================================================================
# 测试4: 计数水平对各场景的影响
# =============================================================================
def test_count_effect():
    print("\n" + "="*70)
    print("TEST 4: COUNT LEVEL EFFECT")
    print("="*70)
    
    # 选择代表性配置
    TEST_CASES = {
        'single_middle': {'E': [1.0], 'B': [1.0], 'BR': [1.0]},
        'separated': {'E': [0.5, 1.0, 2.0], 'B': [1.5, 0.0, -0.5], 'BR': [0.33, 0.34, 0.33]},
        'close': {'E': [0.9, 1.0, 1.1], 'B': [1.0, 0.0, -0.5], 'BR': [0.33, 0.34, 0.33]},
    }
    
    sigma = 0.015  # 中等宽度
    
    for case_name, cfg in TEST_CASES.items():
        print(f"\n=== {case_name} ===")
        for n_events in COUNT_LEVELS:
            test_name = f"{case_name} @ {n_events:.0e}"
            results, true_r = run_test(
                E_centers=cfg['E'], Betas=cfg['B'], BRs=cfg['BR'],
                sigma_laser=sigma, n_events=n_events, test_name=test_name
            )
            print_results(results, test_name)

# =============================================================================
# 主函数
# =============================================================================
def main():
    print("="*70)
    print("DETAILED FACTOR ANALYSIS")
    print("="*70)
    print("\nFactors being tested:")
    print("  1. Peak width (sigma_laser): narrow/medium/wide")
    print("  2. Peak position (energy -> radius): inner/middle/outer")
    print("  3. Peak separation: separated vs close")
    print("  4. Count level: 1e5 vs 1e6")
    
    # 运行所有测试
    test_single_peak()
    test_separated_peaks()
    test_close_peaks()
    test_count_effect()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
