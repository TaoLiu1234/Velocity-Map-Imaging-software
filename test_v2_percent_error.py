"""测试V2在不同事件计数下各指标与真值的百分比误差"""
import numpy as np
import sys
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction_v2 import PhysicsBasedFitterV2 as FitterV2

def run_test(n_events, seed=42):
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
    
    true_values = []
    for E, beta, br in zip(config.E_centers, config.Betas, config.branching_ratios):
        r_px = config.get_expected_radius(E) / config.pixel_size
        sigma_r = r_px * config.sigma_laser / (2 * E)
        true_values.append({'r': r_px, 'sigma': sigma_r, 'beta': beta, 'br': br})
    
    np.random.seed(seed)
    sys.stdout = StringIO()
    image, _ = run_simulation(config, add_noise=True, add_background=False)
    sys.stdout = sys.__stdout__
    
    image_corrected = np.maximum(image - config.readout_offset, 0)
    n_pixels = image.shape[0]
    
    # V2
    fitter_v2 = FitterV2(n_pixels)
    fitter_v2.calibrate_from_config(config)
    sys.stdout = StringIO()
    final_v2, _, _ = fitter_v2.solve(image_corrected)
    sys.stdout = sys.__stdout__
    
    # BR计算：精确3D高斯积分 amp × σ × (r² + σ²)
    integrated = []
    for p in final_v2:
        sigma_for_br = p.get('sigma_measured', p['sigma'])
        r = p['r']
        integrated.append(p['amp'] * sigma_for_br * (r**2 + sigma_for_br**2))
    total_int = sum(integrated)
    for i, p in enumerate(final_v2):
        p['br'] = integrated[i] / total_int if total_int > 0 else 0
    
    # 匹配peaks
    matched = []
    used_indices = []
    for i, tv in enumerate(true_values):
        best_idx, best_dist = -1, 20
        for j, p in enumerate(final_v2):
            if j not in used_indices:
                dist = abs(p['r'] - tv['r'])
                if dist < best_dist:
                    best_dist, best_idx = dist, j
        if best_idx >= 0:
            used_indices.append(best_idx)
            p = final_v2[best_idx]
            # 计算百分比误差
            r_pct = 100 * (p['r'] - tv['r']) / tv['r']
            sigma_pct = 100 * (p['sigma'] - tv['sigma']) / tv['sigma']
            # beta特殊处理：真值为0时用绝对误差
            if abs(tv['beta']) > 0.01:
                beta_pct = 100 * (p['beta'] - tv['beta']) / tv['beta']
            else:
                beta_pct = 100 * (p['beta'] - tv['beta'])  # 绝对误差×100
            br_pct = 100 * (p['br'] - tv['br']) / tv['br']
            
            matched.append({
                'peak': i+1,
                'r': p['r'], 'r_true': tv['r'], 'r_pct': r_pct,
                'sigma': p['sigma'], 'sigma_true': tv['sigma'], 'sigma_pct': sigma_pct,
                'beta': p['beta'], 'beta_true': tv['beta'], 'beta_pct': beta_pct,
                'br': p['br'], 'br_true': tv['br'], 'br_pct': br_pct,
            })
    
    return matched

# 测试 5e5 ~ 1e8
event_counts = [5e5, 1e6, 5e6, 1e7, 5e7, 1e8]

print("=" * 120)
print("V2 Performance: Percentage Error vs True Values")
print("=" * 120)

print("\nTRUE VALUES:")
print("  Peak 1: r=100.0 px, σ=1.50 px, β=+1.50, BR=0.30")
print("  Peak 2: r=141.4 px, σ=1.06 px, β= 0.00, BR=0.50")
print("  Peak 3: r=200.0 px, σ=0.75 px, β=-0.50, BR=0.20")

print("\n" + "=" * 120)
print(f"{'N_events':<10} | {'Peak':<5} | {'r (px)':<20} | {'σ (px)':<20} | {'β':<20} | {'BR':<20}")
print(f"{'':10} | {'':5} | {'est → err%':<20} | {'est → err%':<20} | {'est → err%':<20} | {'est → err%':<20}")
print("-" * 120)

for n_events in event_counts:
    results = run_test(n_events)
    
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

print("\n" + "=" * 120)
print("SUMMARY: Average Absolute Percentage Error by Event Count")
print("=" * 120)
print(f"{'N_events':<12} | {'r (%)':<10} | {'σ (%)':<10} | {'β (%)':<10} | {'BR (%)':<10}")
print("-" * 60)

for n_events in event_counts:
    results = run_test(n_events)
    avg_r = np.mean([abs(m['r_pct']) for m in results])
    avg_s = np.mean([abs(m['sigma_pct']) for m in results])
    avg_b = np.mean([abs(m['beta_pct']) for m in results])
    avg_br = np.mean([abs(m['br_pct']) for m in results])
    print(f"{n_events:<12.0e} | {avg_r:<10.2f} | {avg_s:<10.1f} | {avg_b:<10.1f} | {avg_br:<10.1f}")
