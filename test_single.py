"""单个配置快速测试"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation
from Abel_backward_reconstruction import reconstruct_vmi_image as reconstruct_v1
from Abel_backward_reconstruction_v2 import reconstruct_vmi_image_v2 as reconstruct_v2

# 创建配置
E_centers = [0.5, 1.0, 2.0]
E_max = max(E_centers)
vmi_k = Config.calculate_vmi_k(E_max, 10.0, mass_amu=127.0)

config = Config(
    img_res=512,
    E_centers=E_centers,
    Betas=[1.5, 0.0, -0.5],
    branching_ratios=[0.3, 0.5, 0.2],
    sigma_laser=0.015,
    N_events=int(1e6),
    readout_offset=100.0,
    readout_sigma=10.0,
    psf_fwhm=0.1,
    pixel_size=0.05,
    vmi_k=vmi_k,
    mass=127.0
)

# 真实参数
print('True parameters:')
for i, E in enumerate(E_centers):
    r_px = config.get_expected_radius(E) / config.pixel_size
    print(f'  Peak {i+1}: r={r_px:.1f}px, beta={config.Betas[i]}, BR={config.branching_ratios[i]}')

# 生成图像
np.random.seed(42)
image, _ = run_simulation(config)
print(f'\nImage generated, sum={np.sum(image):.0f}')

# V2重建
print('\nRunning V2...')
params_v2, _ = reconstruct_v2(image - config.readout_offset, config, verbose=False)
print(f'V2 found {len(params_v2)} peaks:')
for p in params_v2:
    br = p.get('branching_ratio', 0)
    print(f'  r={p["r"]:.1f}, sigma={p["sigma"]:.2f}, beta={p["beta"]:.2f}, BR={br:.3f}')
