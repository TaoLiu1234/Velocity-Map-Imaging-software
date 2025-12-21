"""测试不同事件计数下V1 vs V2的性能"""
import numpy as np
import sys
from io import StringIO
import time
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import PhysicsBasedFitter as FitterV1
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
    
    # V1
    t0 = time.time()
    fitter_v1 = FitterV1(n_pixels)
    sys.stdout = StringIO()
    final_v1, _, _ = fitter_v1.solve(image_corrected)
    sys.stdout = sys.__stdout__
    v1_time = time.time() - t0
    
    # V2
    t0 = time.time()
    fitter_v2 = FitterV2(n_pixels)
    fitter_v2.calibrate_from_config(config)
    sys.stdout = StringIO()
    final_v2, _, _ = fitter_v2.solve(image_corrected)
    sys.stdout = sys.__stdout__
    v2_time = time.time() - t0
    
    def match_peaks(results, true_vals):
        if len(results) == 0:
            return []
        # BR计算：使用积分强度 (amp × sigma_measured × r²)
        # 在3D空间中，粒子数 ∝ ∫ I(r) × 4πr² dr
        integrated = []
        for p in results:
            sigma_for_br = p.get('sigma_measured', p['sigma'])
            integrated.append(p['amp'] * sigma_for_br * p['r']**2)
        total_int = sum(integrated)
        for i, p in enumerate(results):
            p['br'] = integrated[i] / total_int if total_int > 0 else 0
        
        matched = []
        for i, tv in enumerate(true_vals):
            best_idx, best_dist = -1, 20
            for j, p in enumerate(results):
                if j not in [m[0] for m in matched]:
                    dist = abs(p['r'] - tv['r'])
                    if dist < best_dist:
                        best_dist, best_idx = dist, j
            if best_idx >= 0:
                p = results[best_idx]
                matched.append((best_idx, {
                    'peak': i+1, 'r': p['r'], 'r_err': p['r']-tv['r'],
                    'sigma': p['sigma'], 'sigma_err': p['sigma']-tv['sigma'],
                    'beta': p['beta'], 'beta_err': p['beta']-tv['beta'],
                    'br': p['br'], 'br_err': p['br']-tv['br']
                }))
        return [m[1] for m in matched]
    
    return true_values, match_peaks(final_v1, true_values), match_peaks(final_v2, true_values), v1_time, v2_time, len(final_v1), len(final_v2)

# 测试 5e5 ~ 1e9
event_counts = [5e5, 1e6, 5e6, 1e7, 5e7, 1e8]

print("=" * 100)
print("V1 vs V2 Performance at Different Event Counts")
print("=" * 100)

print("\nTRUE VALUES: Peak 1: r=100.0, σ=1.50, β=1.50, BR=0.30")
print("             Peak 2: r=141.4, σ=1.06, β=0.00, BR=0.50")
print("             Peak 3: r=200.0, σ=0.75, β=-0.50, BR=0.20")

for n_events in event_counts:
    tv, v1_m, v2_m, v1_t, v2_t, v1_n, v2_n = run_test(n_events)
    
    print(f"\n{'='*100}")
    print(f"N = {n_events:.0e} | V1: {v1_n} peaks ({v1_t:.1f}s) | V2: {v2_n} peaks ({v2_t:.1f}s)")
    print(f"{'='*100}")
    
    print(f"\n{'Peak':<6} | {'V1 r':<14} {'V2 r':<14} | {'V1 σ':<14} {'V2 σ':<14} | {'V1 β':<14} {'V2 β':<14} | {'V1 BR':<14} {'V2 BR':<14}")
    print("-" * 130)
    
    for i in range(3):
        v1 = next((m for m in v1_m if m['peak']==i+1), None)
        v2 = next((m for m in v2_m if m['peak']==i+1), None)
        
        v1_r = f"{v1['r']:.1f} ({v1['r_err']:+.1f})" if v1 else "N/A"
        v2_r = f"{v2['r']:.1f} ({v2['r_err']:+.1f})" if v2 else "N/A"
        v1_s = f"{v1['sigma']:.2f} ({v1['sigma_err']:+.2f})" if v1 else "N/A"
        v2_s = f"{v2['sigma']:.2f} ({v2['sigma_err']:+.2f})" if v2 else "N/A"
        v1_b = f"{v1['beta']:.3f} ({v1['beta_err']:+.3f})" if v1 else "N/A"
        v2_b = f"{v2['beta']:.3f} ({v2['beta_err']:+.3f})" if v2 else "N/A"
        v1_br = f"{v1['br']:.2f} ({v1['br_err']:+.2f})" if v1 else "N/A"
        v2_br = f"{v2['br']:.2f} ({v2['br_err']:+.2f})" if v2 else "N/A"
        
        print(f"{i+1:<6} | {v1_r:<14} {v2_r:<14} | {v1_s:<14} {v2_s:<14} | {v1_b:<14} {v2_b:<14} | {v1_br:<14} {v2_br:<14}")
    
    # MAE
    if v1_m and v2_m:
        v1_r_mae = np.mean([abs(m['r_err']) for m in v1_m])
        v2_r_mae = np.mean([abs(m['r_err']) for m in v2_m])
        v1_s_mae = np.mean([abs(m['sigma_err']) for m in v1_m])
        v2_s_mae = np.mean([abs(m['sigma_err']) for m in v2_m])
        v1_b_mae = np.mean([abs(m['beta_err']) for m in v1_m])
        v2_b_mae = np.mean([abs(m['beta_err']) for m in v2_m])
        v1_br_mae = np.mean([abs(m['br_err']) for m in v1_m])
        v2_br_mae = np.mean([abs(m['br_err']) for m in v2_m])
        
        print(f"\nMAE:   | V1={v1_r_mae:.2f} V2={v2_r_mae:.2f}    | V1={v1_s_mae:.2f} V2={v2_s_mae:.2f}    | V1={v1_b_mae:.3f} V2={v2_b_mae:.3f}  | V1={v1_br_mae:.2f} V2={v2_br_mae:.2f}")

