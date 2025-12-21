"""
快速综合测试 - 只运行剩余测试并汇总结果
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
    'uniform_BR': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.33, 0.34, 0.33], 'cat': 'br'},
    'skewed_BR': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.7, 0.2, 0.1], 'cat': 'br'},
    'weak_peak': {'E_centers': [0.5, 1.0, 2.0], 'Betas': [1.0, 0.0, -0.5], 'BRs': [0.48, 0.48, 0.04], 'cat': 'br'},
}

COUNT_LEVELS = [int(1e6)]  # 只测1e6
METHODS = ['V1', 'V2', 'V3']

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

def main():
    print("="*60)
    print("REMAINING TESTS (BR distribution @ 1e6)")
    print("="*60)
    
    n_events = int(1e6)
    
    for config_name, test_cfg in TEST_CONFIGS.items():
        print(f"\n{config_name}:")
        config = create_config(test_cfg, n_events)
        true_r = get_true_r(config)
        np.random.seed(42)
        image, _ = run_simulation(config)
        
        print(f"  {'Method':<6} | {'r(%)':>7} | {'β(%)':>7} | {'BR(%)':>7} | {'Det':>5}")
        print("  " + "-"*42)
        
        for name, func in METHOD_FUNCS.items():
            params = func(image, config)
            e = match_peaks(true_r, config.Betas, config.branching_ratios, params)
            if e:
                print(f"  {name:<6} | {e['r_err']:>7.2f} | {e['beta_err']:>7.2f} | {e['br_err']:>7.2f} | {e['n_det']}/{e['n_true']}")
            else:
                print(f"  {name:<6} | {'FAIL':>7} | {'FAIL':>7} | {'FAIL':>7} | 0/?")

if __name__ == "__main__":
    main()
