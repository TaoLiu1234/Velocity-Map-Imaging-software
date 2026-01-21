"""
使用forward simulation生成投影图像并进行三维可视化
xy方向是物理距离(mm)，z方向是强度(counts)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Abel_forward_simulation import Config, run_simulation

# 设置能量级别和参数 - 两个轨道，增大能量间距
E_centers = [1.0, 3.0]  # 增大能量间距：1.0 eV 和 3.0 eV
Betas = [0.0, 1.0]  # beta值：0, 1
branching_ratios = [0.5, 0.5]  # 均匀分布

# 计算VMI校准 - r_max决定数据范围
E_max = max(E_centers)
r_max_mm = 50.0  # 数据范围保持在50mm
vmi_k = Config.calculate_vmi_k(E_max, r_max_mm)

# 创建配置
config = Config(
    E_centers=E_centers,
    Betas=Betas,
    branching_ratios=branching_ratios,
    N_events=500000,  # 增加粒子数量以获得更多计数
    vmi_k=vmi_k,
    sigma_laser=0.015,
    T_beam=0.0,
    tau_lifetimes=[100.0, 50.0],  # 两个能量级别对应两个lifetime
    photon_energy=21.2,
    target_mass=28.0,
    vol_sigma=(0.0, 0.0, 0.0),
    polarization_vec=[0, 1, 0],
    img_res=1000,  # 保持分辨率
    pixel_size=0.15,  # 增大pixel_size以扩大显示范围 (1000*0.15=150mm，覆盖±75mm，数据范围±50mm，外圈有空白)
    psf_fwhm=0.2,  # 添加一些PSF展宽
    dld_resolution=0.0,
    mcp_dark_rate=0.0,
    residual_gas_rate=0.0,
)

print("Running forward simulation...")
print(f"Energy levels: {config.E_centers} eV")
print(f"Beta values: {config.Betas}")
print(f"Number of particles: {config.N_events:,}")
print(f"Image resolution: {config.img_res}x{config.img_res} pixels")
print(f"Detector size: {config.detector_size_mm:.1f} mm")

# 运行simulation生成投影并binning后的2D图像
image, metadata = run_simulation(
    config, 
    add_noise=False, 
    return_particles=False,
    output_mode='image'  # 返回binning后的图像
)

print(f"\nGenerated image:")
print(f"  Image shape: {image.shape}")
print(f"  Total counts: {np.sum(image):.0f}")
print(f"  Max intensity: {np.max(image):.0f}")

# 创建物理距离坐标（mm）
half_size = config.detector_size_mm / 2
x_coords = np.linspace(-half_size, half_size, config.img_res)
y_coords = np.linspace(-half_size, half_size, config.img_res)
X, Y = np.meshgrid(x_coords, y_coords)

# 创建3D可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 使用surface plot显示3D图像
# 为了更好的可视化效果，可以降低采样率
stride = max(1, config.img_res // 128)  # 如果分辨率太高，降低采样
X_plot = X[::stride, ::stride]
Y_plot = Y[::stride, ::stride]
Z_plot = image[::stride, ::stride]

# 创建3D表面图
surf = ax.plot_surface(X_plot, Y_plot, Z_plot, 
                      cmap='hot', 
                      alpha=0.9,
                      linewidth=0,
                      antialiased=True,
                      shade=True)

# 设置标签和标题
ax.set_xlabel('X (mm)', fontsize=12)
ax.set_ylabel('Y (mm)', fontsize=12)
ax.set_zlabel('Intensity (counts)', fontsize=12)
ax.set_title('3D Visualization of Projected and Binned VMI Image', fontsize=14)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Intensity (counts)')

# 设置视角
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

print("\nVisualization complete!")

