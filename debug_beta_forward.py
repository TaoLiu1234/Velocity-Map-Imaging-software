"""
测试：用完整前向模型估计Beta

完整前向链：
3D分布(r,sigma_laser,beta) -> Abel投影 -> PSF卷积 -> 像素化 -> xy->rtheta插值

比较低频区域的FFT成分（不只是DC和cos2theta）
"""
import numpy as np
import sys
from io import StringIO
import abel
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize_scalar

from Abel_forward_simulation import Config, run_simulation

# 配置
config = Config(
    E_centers=[0.5, 1.0, 2.0],
    Betas=[1.5, 0.0, -0.5],
    branching_ratios=[0.3, 0.5, 0.2],
    vmi_k=Config.calculate_vmi_k(E_max_eV=2.0, r_max_mm=20.0),
    sigma_laser=0.015,
    N_events=int(1e6),
    img_res=512,
    pixel_size=0.1,
    readout_offset=100.0,
    psf_fwhm=0.0,
)

# 物理参数
sigma_psf = 0.0
sigma_pixel = 0.4
sigma_interp = 0.3

# 计算真值
print("=" * 80)
print("TRUE VALUES")
print("=" * 80)
true_peaks = []
for i, (E, beta, br) in enumerate(zip(config.E_centers, config.Betas, config.branching_ratios)):
    r_mm = config.get_expected_radius(E)
    r_px = r_mm / config.pixel_size
    sigma_r = r_px * config.sigma_laser / (2 * E)
    true_peaks.append({'r': r_px, 'sigma': sigma_r, 'beta': beta, 'E': E})
    print(f"Peak {i+1}: E={E:.2f}eV, r={r_px:.1f}px, sigma={sigma_r:.2f}px, beta={beta:.2f}")

# 生成图像
np.random.seed(42)
old_stdout = sys.stdout
sys.stdout = StringIO()
image, _ = run_simulation(config, add_noise=True, add_background=False)
sys.stdout = old_stdout

n_pixels = image.shape[0]
baseline = config.readout_offset
image_corrected = np.maximum(image - baseline, 0)

# 预计算坐标网格
cy, cx = n_pixels // 2, n_pixels // 2
n_r = n_pixels // 2
n_theta = 720

y, x = np.ogrid[:n_pixels, :n_pixels]
Y = y - cy
X = x - cx
R = np.sqrt(X**2 + Y**2)
with np.errstate(divide='ignore', invalid='ignore'):
    COS_THETA = X / R
COS_THETA[~np.isfinite(COS_THETA)] = 0.0
P2_GRID = 0.5 * (3 * COS_THETA**2 - 1)

theta_grid = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
r_grid_1d = np.arange(n_r)
theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid_1d)
x_cart = cx + r_mesh * np.cos(theta_mesh)
y_cart = cy + r_mesh * np.sin(theta_mesh)

# 观测的polar矩阵
polar_obs = map_coordinates(image_corrected, [y_cart, x_cart], order=3, mode='constant', cval=0.0)


def forward_model_polar(r0, sigma_laser, beta, n_pixels, R, P2_GRID, 
                        x_cart, y_cart, sigma_psf, sigma_pixel, sigma_interp):
    """完整前向模型"""
    radial = np.exp(-((R - r0)**2) / (2 * sigma_laser**2))
    angular = 1 + beta * P2_GRID
    img_3d = radial * angular
    
    img_proj = abel.Transform(img_3d, method='hansenlaw', 
                              direction='forward', verbose=False).transform
    
    if sigma_psf > 0.1:
        img_proj = gaussian_filter(img_proj, sigma=sigma_psf)
    if sigma_pixel > 0.1:
        img_proj = gaussian_filter(img_proj, sigma=sigma_pixel * 0.5)
    
    polar_model = map_coordinates(img_proj, [y_cart, x_cart], order=3, mode='constant', cval=0.0)
    
    if sigma_interp > 0.1:
        polar_model = gaussian_filter1d(polar_model, sigma=sigma_interp * 0.5, axis=0)
    
    return polar_model


def get_low_freq_components(angular_profile, k_max=5):
    """提取低频成分"""
    n = len(angular_profile)
    fft = np.fft.fft(angular_profile)
    amplitudes = np.abs(fft[:k_max+1]) / n
    amplitudes[1:] *= 2
    phases = np.angle(fft[:k_max+1])
    return amplitudes, phases


def compare_low_freq(obs_angular, model_angular, k_max=5):
    """比较低频成分"""
    amp_obs, phase_obs = get_low_freq_components(obs_angular, k_max)
    amp_model, phase_model = get_low_freq_components(model_angular, k_max)
    
    if amp_obs[0] > 1e-10:
        amp_obs_norm = amp_obs / amp_obs[0]
    else:
        amp_obs_norm = amp_obs
    
    if amp_model[0] > 1e-10:
        amp_model_norm = amp_model / amp_model[0]
    else:
        amp_model_norm = amp_model
    
    amp_residual = np.sum((amp_obs_norm - amp_model_norm)**2)
    
    phase_residual = 0
    for k in range(1, k_max+1):
        if amp_obs_norm[k] > 0.01 and amp_model_norm[k] > 0.01:
            phase_diff = np.abs(np.exp(1j*phase_obs[k]) - np.exp(1j*phase_model[k]))
            phase_residual += phase_diff * amp_obs_norm[k]
    
    return amp_residual + phase_residual * 0.5


# 方法1：直接FFT
print("\n" + "=" * 80)
print("METHOD 1: Direct FFT")
print("=" * 80)

for i, tp in enumerate(true_peaks):
    r_idx = int(tp['r'])
    angular = polar_obs[r_idx, :]
    
    fft = np.fft.fft(angular)
    dc = np.abs(fft[0]) / n_theta
    cos2_amp = 2 * np.abs(fft[2]) / n_theta
    phase = np.angle(fft[2])
    sign = 1.0 if np.abs(phase) < np.pi/2 else -1.0
    cos2_signed = sign * cos2_amp
    
    if dc > 1e-6:
        beta_direct = 4.0 * cos2_signed / (3.0 * dc - cos2_signed)
        beta_direct = np.clip(beta_direct, -1.0, 2.0)
    else:
        beta_direct = 0.0
    
    print(f"Peak {i+1}: beta_direct={beta_direct:.3f}, beta_true={tp['beta']:.2f}, error={beta_direct-tp['beta']:.3f}")


# 方法2：前向模型 + 低频比较
print("\n" + "=" * 80)
print("METHOD 2: Forward model + low-freq comparison")
print("=" * 80)

def estimate_beta_forward(r0, sigma_laser, polar_obs, k_max=5):
    r_idx = int(r0)
    search_range = max(2, int(sigma_laser * 1.5))
    r_start = max(10, r_idx - search_range)
    r_end = min(polar_obs.shape[0] - 1, r_idx + search_range)
    
    intensities = [np.mean(polar_obs[r, :]) for r in range(r_start, r_end + 1)]
    r_max = r_start + np.argmax(intensities)
    obs_angular = polar_obs[r_max, :]
    
    def loss_func(beta):
        polar_model = forward_model_polar(
            r0, sigma_laser, beta, n_pixels, R, P2_GRID,
            x_cart, y_cart, sigma_psf, sigma_pixel, sigma_interp
        )
        model_angular = polar_model[r_max, :]
        return compare_low_freq(obs_angular, model_angular, k_max)
    
    result = minimize_scalar(loss_func, bounds=(-1.0, 2.0), method='bounded')
    return result.x, r_max

for i, tp in enumerate(true_peaks):
    beta_forward, r_max = estimate_beta_forward(tp['r'], tp['sigma'], polar_obs, k_max=5)
    print(f"Peak {i+1}: beta_forward={beta_forward:.3f}, beta_true={tp['beta']:.2f}, error={beta_forward-tp['beta']:.3f}")


# 方法3：前向模型，只比较k=0,2
print("\n" + "=" * 80)
print("METHOD 3: Forward model, k=0,2 only")
print("=" * 80)

def estimate_beta_forward_k02(r0, sigma_laser, polar_obs):
    r_idx = int(r0)
    search_range = max(2, int(sigma_laser * 1.5))
    r_start = max(10, r_idx - search_range)
    r_end = min(polar_obs.shape[0] - 1, r_idx + search_range)
    
    intensities = [np.mean(polar_obs[r, :]) for r in range(r_start, r_end + 1)]
    r_max = r_start + np.argmax(intensities)
    
    obs_angular = polar_obs[r_max, :]
    fft_obs = np.fft.fft(obs_angular)
    dc_obs = np.abs(fft_obs[0])
    cos2_obs = np.abs(fft_obs[2])
    phase_obs = np.angle(fft_obs[2])
    sign_obs = 1.0 if np.abs(phase_obs) < np.pi/2 else -1.0
    ratio_obs = sign_obs * cos2_obs / dc_obs if dc_obs > 1e-6 else 0
    
    def loss_func(beta):
        polar_model = forward_model_polar(
            r0, sigma_laser, beta, n_pixels, R, P2_GRID,
            x_cart, y_cart, sigma_psf, sigma_pixel, sigma_interp
        )
        model_angular = polar_model[r_max, :]
        fft_model = np.fft.fft(model_angular)
        dc_model = np.abs(fft_model[0])
        cos2_model = np.abs(fft_model[2])
        phase_model = np.angle(fft_model[2])
        sign_model = 1.0 if np.abs(phase_model) < np.pi/2 else -1.0
        ratio_model = sign_model * cos2_model / dc_model if dc_model > 1e-6 else 0
        return (ratio_model - ratio_obs)**2
    
    result = minimize_scalar(loss_func, bounds=(-1.0, 2.0), method='bounded')
    return result.x, ratio_obs

for i, tp in enumerate(true_peaks):
    beta_k02, ratio_obs = estimate_beta_forward_k02(tp['r'], tp['sigma'], polar_obs)
    print(f"Peak {i+1}: beta_k02={beta_k02:.3f}, beta_true={tp['beta']:.2f}, error={beta_k02-tp['beta']:.3f}")


# 总结
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Peak':<6} {'True':<8} {'Direct':<10} {'Forward(k02)':<15} {'Forward(full)':<15}")

for i, tp in enumerate(true_peaks):
    r_idx = int(tp['r'])
    angular = polar_obs[r_idx, :]
    fft = np.fft.fft(angular)
    dc = np.abs(fft[0]) / n_theta
    cos2_amp = 2 * np.abs(fft[2]) / n_theta
    phase = np.angle(fft[2])
    sign = 1.0 if np.abs(phase) < np.pi/2 else -1.0
    beta_direct = 4.0 * sign * cos2_amp / (3.0 * dc - sign * cos2_amp) if dc > 1e-6 else 0
    beta_direct = np.clip(beta_direct, -1.0, 2.0)
    
    beta_k02, _ = estimate_beta_forward_k02(tp['r'], tp['sigma'], polar_obs)
    beta_full, _ = estimate_beta_forward(tp['r'], tp['sigma'], polar_obs, k_max=5)
    
    print(f"{i+1:<6} {tp['beta']:<8.2f} {beta_direct:<10.3f} {beta_k02:<15.3f} {beta_full:<15.3f}")

print("=" * 80)
