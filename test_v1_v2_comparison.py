"""
V1 vs V2 系统性对比测试

测试不同事件计数下的性能：
- 位置估计精度
- Sigma估计精度  
- Beta估计精度
- 计算时间
"""
import numpy as np
import sys
from io import StringIO
import time
import matplotlib.pyplot as plt

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import PhysicsBasedFitter as FitterV1
from Abel_backward_reconstruction_v2 import PhysicsBasedFitterV2 as FitterV2

# 测试配置
config_base = Config(
    E_centers=[0.5, 1.0, 2.0],
    Betas=[1.5, 0.0, -0.5],
    branching_ratios=[0.3, 0.5, 0.2],
    vmi_k=Config.calculate_vmi_k(E_max_eV=2.0, r_max_mm=20.0),
    sigma_laser=0.015,
    N_events=int(1e6),  # 会被覆盖
    img_res=512,
    pixel_size=0.1,
    readout_offset=100.0,
    psf_fwhm=0.0,
)

# 计算真值
true_values = []
for E, beta in zip(config_base.E_centers, config_base.Betas):
    r_mm = config_base.get_expected_radius(E)
    r_px = r_mm / config_base.pixel_size
    sigma_r = r_px * config_base.sigma_laser / (2 * E)
    true_values.append({'r': r_px, 'sigma': sigma_r, 'beta': beta, 'E': E})

print("=" * 80)
print("TRUE VALUES")
print("=" * 80)
for i, tv in enumerate(true_values):
    print(f"Peak {i+1}: r={tv['r']:.1f}px, σ={tv['sigma']:.2f}px, β={tv['beta']:.2f}")

# 测试不同事件计数
event_counts = [1e4, 1e5, 1e6]
n_trials = 1  # 每个计数重复次数

results = {
    'events': [],
    'v1_r_err': [], 'v2_r_err': [],
    'v1_sigma_err': [], 'v2_sigma_err': [],
    'v1_beta_err': [], 'v2_beta_err': [],
    'v1_time': [], 'v2_time': [],
}

print("\n" + "=" * 80)
print("RUNNING TESTS")
print("=" * 80)

for n_events in event_counts:
    print(f"\nN_events = {n_events:.0e}")
    
    v1_r_errs, v2_r_errs = [], []
    v1_sigma_errs, v2_sigma_errs = [], []
    v1_beta_errs, v2_beta_errs = [], []
    v1_times, v2_times = [], []
    
    for trial in range(n_trials):
        # 生成图像
        config = Config(
            E_centers=config_base.E_centers,
            Betas=config_base.Betas,
            branching_ratios=config_base.branching_ratios,
            vmi_k=config_base.vmi_k,
            sigma_laser=config_base.sigma_laser,
            N_events=int(n_events),
            img_res=config_base.img_res,
            pixel_size=config_base.pixel_size,
            readout_offset=config_base.readout_offset,
            psf_fwhm=config_base.psf_fwhm,
        )
        
        np.random.seed(42 + trial)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        image, _ = run_simulation(config, add_noise=True, add_background=False)
        sys.stdout = old_stdout
        
        n_pixels = image.shape[0]
        baseline = config.readout_offset
        image_corrected = np.maximum(image - baseline, 0)
        
        # V1
        t0 = time.time()
        fitter_v1 = FitterV1(n_pixels)
        sys.stdout = StringIO()
        final_v1, _, _ = fitter_v1.solve(image_corrected)
        sys.stdout = old_stdout
        v1_time = time.time() - t0
        
        # V2
        t0 = time.time()
        fitter_v2 = FitterV2(n_pixels)
        fitter_v2.calibrate_from_config(config)
        sys.stdout = StringIO()
        final_v2, _, _ = fitter_v2.solve(image_corrected)
        sys.stdout = old_stdout
        v2_time = time.time() - t0
        
        # 计算误差
        v1_r_err, v2_r_err = 0, 0
        v1_sigma_err, v2_sigma_err = 0, 0
        v1_beta_err, v2_beta_err = 0, 0
        
        for i, tv in enumerate(true_values):
            if i < len(final_v1):
                v1_r_err += abs(final_v1[i]['r'] - tv['r'])
                v1_sigma_err += abs(final_v1[i]['sigma'] - tv['sigma'])
                v1_beta_err += abs(final_v1[i]['beta'] - tv['beta'])
            if i < len(final_v2):
                v2_r_err += abs(final_v2[i]['r'] - tv['r'])
                v2_sigma_err += abs(final_v2[i]['sigma'] - tv['sigma'])
                v2_beta_err += abs(final_v2[i]['beta'] - tv['beta'])
        
        n_peaks = len(true_values)
        v1_r_errs.append(v1_r_err / n_peaks)
        v2_r_errs.append(v2_r_err / n_peaks)
        v1_sigma_errs.append(v1_sigma_err / n_peaks)
        v2_sigma_errs.append(v2_sigma_err / n_peaks)
        v1_beta_errs.append(v1_beta_err / n_peaks)
        v2_beta_errs.append(v2_beta_err / n_peaks)
        v1_times.append(v1_time)
        v2_times.append(v2_time)
    
    # 平均
    results['events'].append(n_events)
    results['v1_r_err'].append(np.mean(v1_r_errs))
    results['v2_r_err'].append(np.mean(v2_r_errs))
    results['v1_sigma_err'].append(np.mean(v1_sigma_errs))
    results['v2_sigma_err'].append(np.mean(v2_sigma_errs))
    results['v1_beta_err'].append(np.mean(v1_beta_errs))
    results['v2_beta_err'].append(np.mean(v2_beta_errs))
    results['v1_time'].append(np.mean(v1_times))
    results['v2_time'].append(np.mean(v2_times))
    
    print(f"  V1: r_err={results['v1_r_err'][-1]:.2f}, σ_err={results['v1_sigma_err'][-1]:.2f}, β_err={results['v1_beta_err'][-1]:.3f}, time={results['v1_time'][-1]:.1f}s")
    print(f"  V2: r_err={results['v2_r_err'][-1]:.2f}, σ_err={results['v2_sigma_err'][-1]:.2f}, β_err={results['v2_beta_err'][-1]:.3f}, time={results['v2_time'][-1]:.1f}s")

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

events = np.array(results['events'])

# 位置误差
ax = axes[0, 0]
ax.semilogx(events, results['v1_r_err'], 'b-o', label='V1')
ax.semilogx(events, results['v2_r_err'], 'r-s', label='V2')
ax.set_xlabel('N_events')
ax.set_ylabel('Mean |r_error| (px)')
ax.set_title('Position Error')
ax.legend()
ax.grid(True)

# Sigma误差
ax = axes[0, 1]
ax.semilogx(events, results['v1_sigma_err'], 'b-o', label='V1')
ax.semilogx(events, results['v2_sigma_err'], 'r-s', label='V2')
ax.set_xlabel('N_events')
ax.set_ylabel('Mean |σ_error| (px)')
ax.set_title('Sigma Error')
ax.legend()
ax.grid(True)

# Beta误差
ax = axes[1, 0]
ax.semilogx(events, results['v1_beta_err'], 'b-o', label='V1')
ax.semilogx(events, results['v2_beta_err'], 'r-s', label='V2')
ax.set_xlabel('N_events')
ax.set_ylabel('Mean |β_error|')
ax.set_title('Beta Error')
ax.legend()
ax.grid(True)

# 计算时间
ax = axes[1, 1]
ax.semilogx(events, results['v1_time'], 'b-o', label='V1')
ax.semilogx(events, results['v2_time'], 'r-s', label='V2')
ax.set_xlabel('N_events')
ax.set_ylabel('Time (s)')
ax.set_title('Computation Time')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('v1_v2_comparison.png', dpi=150)
print("\nSaved: v1_v2_comparison.png")

# 打印总结
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'N_events':<12} {'V1 r_err':<10} {'V2 r_err':<10} {'V1 σ_err':<10} {'V2 σ_err':<10} {'V1 β_err':<10} {'V2 β_err':<10}")
for i, n in enumerate(results['events']):
    print(f"{n:<12.0e} {results['v1_r_err'][i]:<10.2f} {results['v2_r_err'][i]:<10.2f} {results['v1_sigma_err'][i]:<10.2f} {results['v2_sigma_err'][i]:<10.2f} {results['v1_beta_err'][i]:<10.3f} {results['v2_beta_err'][i]:<10.3f}")

