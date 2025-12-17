"""
系统性测试：不同光子数下的重建性能
对比指标：Peak位置(E), FWHM, β, Relative Amplitude
"""
import numpy as np
import sys
from io import StringIO

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image
from Abel_rbasex_reconstruction import reconstruct_rbasex

def run_test(N_events, seed=42):
    """运行单次测试"""
    np.random.seed(seed)
    
    # 配置参数
    E_centers = [0.5, 1.0, 2.0]
    Betas = [1.5, 0.0, -0.5]
    branching_ratios = [0.3, 0.5, 0.2]
    sigma_laser = 0.015
    
    E_max = max(E_centers)
    r_max_mm = 20.0
    vmi_k = Config.calculate_vmi_k(E_max_eV=E_max, r_max_mm=r_max_mm)
    
    config = Config(
        E_centers=E_centers,
        Betas=Betas,
        branching_ratios=branching_ratios,
        N_events=N_events,
        vmi_k=vmi_k,
        sigma_laser=sigma_laser,
        T_beam=10.0,
        tau_lifetimes=0.0,
        photon_energy=21.2,
        target_mass=28.0,
        vol_sigma=(0.0, 0.0, 0.0),
        polarization_vec=[0, 1, 0],
        img_res=512,
        pixel_size=0.1,
        psf_fwhm=0.0,
        supersample_factor=4,
        dark_rate=0.1,
        readout_sigma=5.0,
        readout_offset=100.0,
        bg_rate=0.0,
        bg_energy=0.15,
        bg_sigma=0.08,
    )
    
    # 生成图像
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        image, meta = run_simulation(config, add_noise=True, add_background=False)
    finally:
        sys.stdout = old_stdout
    
    # 测试：不减去offset，看改进后的噪声估计能否处理
    # image = image - config.readout_offset
    # image = np.maximum(image, 0)
    
    # PhysicsBasedFitter重建
    sys.stdout = StringIO()
    try:
        physics_params, physics_meta = reconstruct_vmi_image(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    
    # rBasex重建
    image = image - config.readout_offset
    image = np.maximum(image, 0)
    sys.stdout = StringIO()
    try:
        rbasex_params, rbasex_meta = reconstruct_rbasex(image, config=config, verbose=False)
    finally:
        sys.stdout = old_stdout
    
    return config, physics_params, rbasex_params

def get_true_fwhm_px(config, E_center):
    """计算真实的FWHM（像素）"""
    r_mm = config.get_expected_radius(E_center)
    r_px = r_mm / config.pixel_size
    sigma_E = config.sigma_laser
    # dE/E = 2 * dr/r for kinetic energy
    sigma_r_px = r_px * sigma_E / (2 * E_center) if E_center > 0 else 1.0
    return sigma_r_px * 2.355

def match_peaks(true_E, params, config):
    """将重建的peaks与真值匹配"""
    if not params:
        return [None] * len(true_E)
    
    params_sorted = sorted(params, key=lambda x: x.get('energy_eV', x['r']))
    matched = []
    
    for E_true in true_E:
        best_match = min(params_sorted, key=lambda x: abs(x.get('energy_eV', 0) - E_true))
        matched.append(best_match)
    
    return matched

# 测试不同光子数 (从1e5到1e8)
photon_levels = [int(1e5), int(5e5), int(1e6), int(5e6), int(1e7), int(5e7), int(1e8)]

print("=" * 140)
print("光子数性能测试 - 完整对比")
print("True: E=[0.5, 1.0, 2.0] eV, β=[1.5, 0.0, -0.5], BR=[0.3, 0.5, 0.2]")
print("=" * 140)

for N_events in photon_levels:
    print(f"\n{'='*140}")
    print(f"N_events = {N_events:.0e}")
    print(f"{'='*140}")
    
    config, physics_params, rbasex_params = run_test(N_events)
    
    # 匹配peaks
    phys_matched = match_peaks(config.E_centers, physics_params, config)
    rb_matched = match_peaks(config.E_centers, rbasex_params, config)
    
    # 打印表头
    print(f"\n{'Peak':<6} | {'Param':<12} | {'True':<10} | {'Physics':<10} | {'Phys_Err':<10} | {'rBasex':<10} | {'rB_Err':<10} | {'Better':<8}")
    print("-" * 140)
    
    for i, (E_true, beta_true, br_true) in enumerate(zip(config.E_centers, config.Betas, config.branching_ratios)):
        # 计算真实值
        r_mm_true = config.get_expected_radius(E_true)
        r_px_true = r_mm_true / config.pixel_size
        fwhm_true = get_true_fwhm_px(config, E_true)
        
        phys_p = phys_matched[i]
        rb_p = rb_matched[i]
        
        # Peak位置 (Energy)
        E_phys = phys_p.get('energy_eV', 0) if phys_p else 0
        E_rb = rb_p.get('energy_eV', 0) if rb_p else 0
        err_E_phys = abs(E_phys - E_true) if phys_p else float('nan')
        err_E_rb = abs(E_rb - E_true) if rb_p else float('nan')
        better_E = 'Physics' if err_E_phys < err_E_rb else 'rBasex'
        
        print(f"Peak {i+1} | {'Energy(eV)':<12} | {E_true:<10.3f} | {E_phys:<10.3f} | {err_E_phys:<10.4f} | {E_rb:<10.3f} | {err_E_rb:<10.4f} | {better_E:<8}")
        
        # FWHM
        fwhm_phys = phys_p.get('fwhm', phys_p.get('sigma', 0) * 2.355) if phys_p else 0
        fwhm_rb = rb_p.get('fwhm', rb_p.get('sigma', 0) * 2.355) if rb_p else 0
        err_fwhm_phys = abs(fwhm_phys - fwhm_true) if phys_p else float('nan')
        err_fwhm_rb = abs(fwhm_rb - fwhm_true) if rb_p else float('nan')
        better_fwhm = 'Physics' if err_fwhm_phys < err_fwhm_rb else 'rBasex'
        
        print(f"       | {'FWHM(px)':<12} | {fwhm_true:<10.2f} | {fwhm_phys:<10.2f} | {err_fwhm_phys:<10.2f} | {fwhm_rb:<10.2f} | {err_fwhm_rb:<10.2f} | {better_fwhm:<8}")
        
        # Beta
        beta_phys = phys_p.get('beta', 0) if phys_p else 0
        beta_rb = rb_p.get('beta', 0) if rb_p else 0
        err_beta_phys = abs(beta_phys - beta_true) if phys_p else float('nan')
        err_beta_rb = abs(beta_rb - beta_true) if rb_p else float('nan')
        better_beta = 'Physics' if err_beta_phys < err_beta_rb else 'rBasex'
        
        print(f"       | {'Beta':<12} | {beta_true:<10.2f} | {beta_phys:<10.3f} | {err_beta_phys:<10.3f} | {beta_rb:<10.3f} | {err_beta_rb:<10.3f} | {better_beta:<8}")
        
        # Relative Amplitude (Branching Ratio)
        br_phys = phys_p.get('branching_ratio', 0) if phys_p else 0
        br_rb = rb_p.get('branching_ratio', 0) if rb_p else 0
        err_br_phys = abs(br_phys - br_true) if phys_p else float('nan')
        err_br_rb = abs(br_rb - br_true) if rb_p else float('nan')
        better_br = 'Physics' if err_br_phys < err_br_rb else 'rBasex'
        
        print(f"       | {'Rel.Amp':<12} | {br_true:<10.3f} | {br_phys:<10.3f} | {err_br_phys:<10.3f} | {br_rb:<10.3f} | {err_br_rb:<10.3f} | {better_br:<8}")
        
        print("-" * 140)

print("\n" + "=" * 140)
print("测试完成")
print("=" * 140)
