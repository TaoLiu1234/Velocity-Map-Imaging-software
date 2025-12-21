"""
全面方法对比测试 v2 (简化版)
===========================
测试不同参数组合下各方法的性能
"""
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image as reconstruct_v1
from Abel_backward_reconstruction_v2 import reconstruct_vmi_image_v2 as reconstruct_v2
from Abel_backward_reconstruction_v3 import AbelReconstructorV3

# =============================================================================
# 测试配置
# =============================================================================
TEST_CONFIGS = {
    # 峰数量
    'single_peak': {'E_centers': [1.0], 'Betas': [1.0], 'BRs': [1.0], 'cat': 'count'},
    'three_peaks': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.5, 0.0, -0.5], 'BRs': [0.3, 0.5, 0.2], 'cat': 'count'},
    'four_peaks': {'E_centers': [0.3, 0.7, 1.2, 2.0], 'Betas': [1.5, 0.5, -0.3, -0.8], 'BRs': [0.2, 0.3, 0.3, 0.2], 'cat': 'count'},
    
    # 峰间距
    'close_peaks': {'E_centers': [0.9, 1.0, 1.1], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.33, 0.34, 0.33], 'cat': 'spacing'},
    'far_peaks': {'E_centers': [0.2, 1.0, 3.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.33, 0.34, 0.33], 'cat': 'spacing'},
    
    # β值
    'isotropic': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [0.0, 0.0, 0.0], 'BRs': [0.33, 0.34, 0.33], 'cat': 'beta'},
    'strong_aniso': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [2.0, -1.0, 1.8], 'BRs': [0.33, 0.34, 0.33], 'cat': 'beta'},
    
    # BR分布
    'uniform_BR': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.33, 0.34, 0.33], 'cat': 'br'},
    'skewed_BR': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.7, 0.2, 0.1], 'cat': 'br'},
    'weak_peak': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.48, 0.48, 0.04], 'cat': 'br'},
}

COUNT_LEVELS = [int(1e5), int(1e6)]
METHODS = ['V1', 'V2', 'V3']

# =============================================================================
# 辅助函数
# =============================================================================
def create_config(test_cfg, n_events):
    E_centers = test_cfg['E_centers']
    E_max = max(E_centers)
    img_res = 512
    pixel_size = 0.05
    r_max_mm = img_res * pixel_size * 0.4
    vmi_k = Config.calculate_vmi_k(E_max, r_max_mm, mass_amu=127.0)
    
    return Config(
        img_res=img_res, E_centers=E_centers, Betas=test_cfg['Betas'],
        branching_ratios=test_cfg['BRs'], sigma_laser=0.015, N_events=n_events,
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

# =============================================================================
# 重建方法
# =============================================================================
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

# =============================================================================
# 主测试
# =============================================================================
def main():
    print("="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*80)
    
    all_results = {}
    total = len(TEST_CONFIGS) * len(COUNT_LEVELS)
    current = 0
    
    for config_name, test_cfg in TEST_CONFIGS.items():
        all_results[config_name] = {}
        for n_events in COUNT_LEVELS:
            current += 1
            print(f"\r[{current}/{total}] {config_name} @ {n_events:.0e}...", end='', flush=True)
            
            config = create_config(test_cfg, n_events)
            true_r = get_true_r(config)
            np.random.seed(42)
            image, _ = run_simulation(config)
            
            results = {}
            for name, func in METHOD_FUNCS.items():
                params = func(image, config)
                results[name] = match_peaks(true_r, config.Betas, config.branching_ratios, params)
            all_results[config_name][n_events] = results
    
    print("\n")
    
    # =========================================================================
    # 1. 各场景结果 (1e6 events)
    # =========================================================================
    print("="*80)
    print("1. RESULTS BY SCENARIO (1e6 events)")
    print("="*80)
    
    n_ev = int(1e6)
    for config_name, test_cfg in TEST_CONFIGS.items():
        results = all_results[config_name][n_ev]
        print(f"\n{config_name} ({test_cfg['cat']}, {len(test_cfg['E_centers'])} peaks)")
        print(f"  {'Method':<6} | {'r(%)':>7} | {'β(%)':>7} | {'BR(%)':>7} | {'Det':>5}")
        print("  " + "-"*42)
        for m in METHODS:
            e = results[m]
            if e:
                print(f"  {m:<6} | {e['r_err']:>7.2f} | {e['beta_err']:>7.2f} | {e['br_err']:>7.2f} | {e['n_det']}/{e['n_true']}")
            else:
                print(f"  {m:<6} | {'FAIL':>7} | {'FAIL':>7} | {'FAIL':>7} | 0/?")
    
    # =========================================================================
    # 2. 计数水平影响
    # =========================================================================
    print("\n" + "="*80)
    print("2. COUNT LEVEL IMPACT")
    print("="*80)
    
    for m in METHODS:
        print(f"\n{m}:")
        print(f"  {'Events':<8} | {'r(%)':>7} | {'β(%)':>7} | {'BR(%)':>7} | {'Det%':>6}")
        print("  " + "-"*45)
        for n_ev in COUNT_LEVELS:
            r_e, b_e, br_e, det = [], [], [], []
            for cn in TEST_CONFIGS:
                e = all_results[cn][n_ev][m]
                if e:
                    r_e.append(e['r_err'])
                    b_e.append(e['beta_err'])
                    if not np.isnan(e['br_err']):
                        br_e.append(e['br_err'])
                    det.append(e['det_rate'])
            r = np.mean(r_e) if r_e else np.nan
            b = np.mean(b_e) if b_e else np.nan
            br = np.mean(br_e) if br_e else np.nan
            d = np.mean(det) if det else 0
            print(f"  {n_ev:<8.0e} | {r:>7.2f} | {b:>7.2f} | {br:>7.2f} | {d:>6.1f}")
    
    # =========================================================================
    # 3. 按类别分析
    # =========================================================================
    print("\n" + "="*80)
    print("3. ANALYSIS BY CATEGORY (1e6 events)")
    print("="*80)
    
    n_ev = int(1e6)
    categories = {}
    for cn, cfg in TEST_CONFIGS.items():
        cat = cfg['cat']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(cn)
    
    for cat, configs in categories.items():
        print(f"\n=== {cat.upper()} ===")
        print(f"{'Method':<6} | {'r(%)':>7} | {'β(%)':>7} | {'BR(%)':>7} | {'Det%':>6}")
        print("-"*45)
        for m in METHODS:
            r_e, b_e, br_e, det = [], [], [], []
            for cn in configs:
                e = all_results[cn][n_ev][m]
                if e:
                    r_e.append(e['r_err'])
                    b_e.append(e['beta_err'])
                    if not np.isnan(e['br_err']):
                        br_e.append(e['br_err'])
                    det.append(e['det_rate'])
            r = np.mean(r_e) if r_e else np.nan
            b = np.mean(b_e) if b_e else np.nan
            br = np.mean(br_e) if br_e else np.nan
            d = np.mean(det) if det else 0
            print(f"{m:<6} | {r:>7.2f} | {b:>7.2f} | {br:>7.2f} | {d:>6.1f}")
    
    # =========================================================================
    # 4. 依赖性分析
    # =========================================================================
    print("\n" + "="*80)
    print("4. DEPENDENCY ANALYSIS (1e6 events)")
    print("="*80)
    
    n_ev = int(1e6)
    
    # 峰数量 vs 检测率
    print("\n--- Peak count vs Detection rate ---")
    for m in METHODS:
        print(f"{m}:", end=" ")
        for cn in ['single_peak', 'three_peaks', 'four_peaks']:
            e = all_results[cn][n_ev][m]
            n_pk = len(TEST_CONFIGS[cn]['E_centers'])
            if e:
                print(f"{n_pk}pk:{e['det_rate']:.0f}%", end="  ")
            else:
                print(f"{n_pk}pk:FAIL", end="  ")
        print()
    
    # 峰间距 vs 位置误差
    print("\n--- Peak spacing vs Position error ---")
    for m in METHODS:
        print(f"{m}:", end=" ")
        for cn in ['close_peaks', 'far_peaks']:
            e = all_results[cn][n_ev][m]
            if e:
                print(f"{cn}:r={e['r_err']:.2f}%,det={e['det_rate']:.0f}%", end="  ")
            else:
                print(f"{cn}:FAIL", end="  ")
        print()
    
    # β范围 vs β误差
    print("\n--- Beta range vs Beta error ---")
    for m in METHODS:
        print(f"{m}:", end=" ")
        for cn in ['isotropic', 'strong_aniso']:
            e = all_results[cn][n_ev][m]
            if e:
                print(f"{cn}:β={e['beta_err']:.2f}%", end="  ")
            else:
                print(f"{cn}:FAIL", end="  ")
        print()
    
    # BR分布 vs BR误差
    print("\n--- BR distribution vs BR error ---")
    for m in METHODS:
        print(f"{m}:", end=" ")
        for cn in ['uniform_BR', 'skewed_BR', 'weak_peak']:
            e = all_results[cn][n_ev][m]
            if e and not np.isnan(e['br_err']):
                print(f"{cn}:BR={e['br_err']:.1f}%", end="  ")
            else:
                print(f"{cn}:N/A", end="  ")
        print()
    
    # =========================================================================
    # 5. 总结
    # =========================================================================
    print("\n" + "="*80)
    print("5. OVERALL SUMMARY (1e6 events)")
    print("="*80)
    
    n_ev = int(1e6)
    wins = {m: {'r': 0, 'beta': 0, 'br': 0} for m in METHODS}
    
    for cn in TEST_CONFIGS:
        results = all_results[cn][n_ev]
        for metric in ['r_err', 'beta_err', 'br_err']:
            best_val, best_m = float('inf'), None
            for m in METHODS:
                e = results[m]
                if e:
                    val = e[metric]
                    if not np.isnan(val) and val < best_val:
                        best_val, best_m = val, m
            if best_m:
                wins[best_m][metric.replace('_err', '')] += 1
    
    print(f"\n{'Method':<6} | {'r(%)':>7} | {'β(%)':>7} | {'BR(%)':>7} | {'Det%':>6} | Wins (r/β/BR)")
    print("-"*65)
    
    for m in METHODS:
        r_e, b_e, br_e, det = [], [], [], []
        for cn in TEST_CONFIGS:
            e = all_results[cn][n_ev][m]
            if e:
                r_e.append(e['r_err'])
                b_e.append(e['beta_err'])
                if not np.isnan(e['br_err']):
                    br_e.append(e['br_err'])
                det.append(e['det_rate'])
        r = np.mean(r_e) if r_e else np.nan
        b = np.mean(b_e) if b_e else np.nan
        br = np.mean(br_e) if br_e else np.nan
        d = np.mean(det) if det else 0
        w = wins[m]
        print(f"{m:<6} | {r:>7.2f} | {b:>7.2f} | {br:>7.2f} | {d:>6.1f} | {w['r']}/{w['beta']}/{w['br']}")
    
    # 最佳方法
    total_wins = {m: sum(wins[m].values()) for m in METHODS}
    champion = max(total_wins, key=total_wins.get)
    print(f"\n🏆 Overall Champion: {champion} ({total_wins[champion]} wins)")
    
    print("\nKey Findings:")
    print("  - Low count (1e5): Higher errors, lower detection")
    print("  - Close peaks: Harder to separate, lower detection")
    print("  - Weak peaks (low BR): May be missed")
    print("  - Strong anisotropy: β estimation more challenging")

if __name__ == "__main__":
    main()
