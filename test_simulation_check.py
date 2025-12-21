"""
快速检查前向模拟参数是否正确
"""
import numpy as np
import matplotlib.pyplot as plt
from Abel_forward_simulation import Config, run_simulation

# 创建测试配置
E_centers = [0.5, 1.0, 2.0]
E_max = max(E_centers)
r_max_mm = 10.0  # 目标：最大能量峰在 10mm 处
pixel_size = 0.05  # mm/pixel -> 10mm = 200 pixels

# 计算 vmi_k
vmi_k = Config.calculate_vmi_k(E_max, r_max_mm, mass_amu=127.0)
print(f"Calculated vmi_k = {vmi_k:.6e}")

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
    pixel_size=pixel_size,
    vmi_k=vmi_k,
    mass=127.0
)

# 打印预期半径
print("\nExpected radii:")
for E in E_centers:
    r_mm = config.get_expected_radius(E)
    r_px = r_mm / pixel_size
    print(f"  E = {E:.1f} eV -> r = {r_mm:.2f} mm = {r_px:.1f} px")

# 生成图像
print("\nGenerating image...")
image, metadata = run_simulation(config, add_noise=True, add_background=True)
print(f"Image shape: {image.shape}")
print(f"Image sum: {np.sum(image):.0f}")

# 计算径向分布
center = config.img_res // 2
y, x = np.ogrid[:config.img_res, :config.img_res]
r = np.sqrt((x - center)**2 + (y - center)**2)
r_int = r.astype(int)

# 角向平均
radial_profile = np.bincount(r_int.ravel(), weights=image.ravel())
radial_counts = np.bincount(r_int.ravel())
radial_counts[radial_counts == 0] = 1
radial_profile = radial_profile / radial_counts

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图像
ax1 = axes[0]
im = ax1.imshow(image - config.readout_offset, cmap='hot', origin='lower')
ax1.set_title('VMI Image (background subtracted)')
plt.colorbar(im, ax=ax1)

# 标记预期峰位置
for E in E_centers:
    r_px = config.get_expected_radius(E) / pixel_size
    circle = plt.Circle((center, center), r_px, fill=False, color='cyan', linestyle='--')
    ax1.add_patch(circle)

# 径向分布
ax2 = axes[1]
ax2.plot(radial_profile, 'b-', label='Radial profile')
ax2.axhline(y=config.readout_offset, color='r', linestyle='--', label='Baseline')

# 标记预期峰位置
for E in E_centers:
    r_px = config.get_expected_radius(E) / pixel_size
    ax2.axvline(x=r_px, color='g', linestyle='--', alpha=0.7)
    ax2.text(r_px, ax2.get_ylim()[1]*0.9, f'{E}eV', ha='center')

ax2.set_xlabel('Radius (pixels)')
ax2.set_ylabel('Intensity')
ax2.set_title('Radial Profile')
ax2.legend()
ax2.set_xlim(0, 256)

plt.tight_layout()
plt.savefig('simulation_check.png', dpi=150)
print("\nSaved: simulation_check.png")
plt.show()
