# VMI 数据重建实践指南

本文档总结了使用 V1、V2、V3 和 rBasex 方法处理实际 VMI 数据时遇到的问题和解决方案。
基于处理 `electron_shilpa_XY.mat` 数据（323212 个事件）的实际经验。

---

## 1. 数据预处理流程

### 1.1 从原始 XY 坐标构建 VMI 图像

**数据来源**：DLD（Delay Line Detector）记录的 XY 坐标，分辨率约 100-200 able/pixel。

```python
# 加载数据
mat = sio.loadmat('electron_shilpa_XY.mat')
xy_data = mat['XY']  # shape: (N, 2)

# 构建 2D 直方图，以原点为中心
max_range = max(np.abs(xy_data[:,0]).max(), np.abs(xy_data[:,1]).max()) * 1.05
image, x_edges, y_edges = np.histogram2d(
    xy_data[:,0], xy_data[:,1], 
    bins=n_bins, 
    range=[[-max_range, max_range], [-max_range, max_range]]
)
image = image.T  # 转置使 y 轴向上
```

**图像尺寸选择 (n_bins)**：

| 尺寸 | 像素大小 | 适用场景 | 优缺点 |
|------|----------|----------|--------|
| 256x256 | ~0.32 mm | 中等事件数（~30万） | 每像素计数高，峰检测稳定 |
| 512x512 | ~0.16 mm | 高事件数（>100万） | 分辨率高，但计数低 |

**建议**：确保主要峰区域每像素平均计数 > 5，否则降低分辨率。

### 1.2 DLD 数据的结构特点

DLD 数据 binning 成图像后，从中心到边缘有三个区域：

```
┌─────────────────────────────────────┐
│           全 0 区域                  │  ← 图像角落，超出 DLD 有效范围
│      ┌─────────────────────┐        │
│      │    背景区域          │        │  ← 有读出噪声，但无信号
│      │  ┌───────────────┐  │        │
│      │  │   信号区域     │  │        │  ← 有物理信号 + 背景噪声
│      │  │  (VMI 环)     │  │        │
│      │  └───────────────┘  │        │
│      │    r = signal_radius │        │
│      └─────────────────────┘        │
│         r = data_outer              │
└─────────────────────────────────────┘
```

| 区域 | 径向范围 | 内容 | 处理方式 |
|------|----------|------|----------|
| 信号区域 | r < signal_radius | 物理信号 + 背景噪声 | 保留，用于重建 |
| 背景区域 | signal_radius < r < data_outer | 纯背景噪声 | 用于估计背景均值 |
| 全 0 区域 | r > data_outer | 无数据（DLD 边界外） | **不能用于背景估计！** |

### 1.3 背景区域识别（关键步骤）

**核心问题**：图像四角的全 0 区域不能用于背景估计！必须找到真正的背景区域。

**正确的背景区域**：
- 数据边界 (data_outer)：从外向内找第一个非零值
- 信号边界 (signal_radius)：信号衰减到背景水平的位置
- 背景区域：signal_radius < r < data_outer 的环形区域


**自动检测算法**：

```python
def identify_background_region(image):
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    
    # 计算径向分布
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    r_max = n // 2
    
    radial_sum = np.bincount(r.ravel(), weights=image.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count = np.bincount(r.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count
    
    # 1. 找数据边界：从外向内找第一个非零值
    data_outer = r_max
    for r_idx in range(r_max - 1, 0, -1):
        if radial_profile[r_idx] > 0.1:
            data_outer = r_idx
            break
    
    # 2. 估计背景水平（使用数据边界内侧 20% 区域）
    outer_start = int(data_outer * 0.8)
    bg_level = np.mean(radial_profile[outer_start:data_outer])
    bg_std = np.std(radial_profile[outer_start:data_outer])
    
    # 3. 找信号边界：从外向内找信号开始上升的位置
    window_size = 10
    bg_r_inner = data_outer
    for r_idx in range(data_outer - window_size, 20, -1):
        local_mean = np.mean(radial_profile[r_idx:r_idx+window_size])
        if local_mean > bg_level + 3 * bg_std:
            bg_r_inner = r_idx + window_size
            break
    
    bg_r_outer = int(data_outer * 0.9)
    return bg_r_inner, bg_r_outer, data_outer
```

**典型值**（以 `electron_shilpa_XY.mat` 为例）：

| 图像尺寸 | 数据边界 | 信号边界 | 背景区域 |
|----------|----------|----------|----------|
| 256x256 | ~120 px | ~72 px | 72-117 px |
| 512x512 | ~240 px | ~144 px | 144-234 px |

### 1.3 背景减除

背景是均匀的高斯噪声（读出噪声 + 暗电流），覆盖整个图像。

```python
# 1. 从背景区域估计噪声均值
bg_mask = (r >= bg_r_inner) & (r <= bg_r_outer)
bg_pixels = image[bg_mask]
bg_mean = np.mean(bg_pixels)
bg_std = np.std(bg_pixels)

# 2. 从整个图像减去背景均值
image_sub = image - bg_mean

# 3. clip 到非负
image_processed = np.maximum(image_sub, 0)

# 4. 信号区域外设为 0
signal_radius = bg_r_inner
image_processed[r > signal_radius] = 0
```

---

## 2. 物理参数设置

### 2.1 系统展宽的来源和分类

各种展宽发生在数据处理的不同阶段：

| 展宽 | 发生阶段 | 作用对象 | 来源 | 能否避免 |
|------|----------|----------|------|----------|
| sigma_psf | 探测器记录 | 原始 XY 坐标 | DLD 空间分辨率 | 不能，硬件固有 |
| sigma_pixel | histogram binning | 图像 | 连续→离散量化 | 不能（DLD），CCD 无此项 |
| sigma_interp | 极坐标变换 | 图像 | 插值算法 | 可选择低展宽算法 |
| sigma_ellipse | 椭圆校正 | 图像 | 仿射变换插值 | 在原始 XY 上校正可避免 |

**数据流和展宽位置**：
```
物理信号 → [sigma_psf] → 原始 XY 坐标 → [sigma_pixel] → 笛卡尔图像 
         → [sigma_ellipse] → 校正后图像 → [sigma_interp] → 极坐标图像
```

### 2.2 探测器展宽参数

| 参数 | 物理含义 | 典型值 (256px) |
|------|----------|----------------|
| `sigma_psf` | DLD 空间分辨率 | ~0.5 px |
| `sigma_pixel` | 直方图 binning 展宽 | 0.289 px |
| `sigma_interp` | 极坐标插值展宽 | ~0.55 px (3阶样条) |

**总系统展宽**：
```python
sigma_sys = np.sqrt(sigma_psf**2 + sigma_pixel**2 + sigma_interp**2)
# 如果在图像上做了椭圆校正，还要加 sigma_ellipse
```

### 2.3 激光/同步辐射参数

```python
sigma_E = 0.005  # 激光带宽 (eV)，同步辐射很窄
vmi_k = 0.01 / (scale_factor ** 2)  # VMI 系数
```

---

## 3. 各方法的参数调整

### 3.1 V1 (PhysicsBasedFitter)

**文件**：`Abel_backward_reconstruction.py`

```python
fitter = PhysicsBasedFitter(n_pixels)
params, r_grid, recon_profile = fitter.solve(image)
```

**调整位置**：
- 峰检测：`_phase1_find_peaks()` 中的 height, prominence, distance
- beta 估计：`_phase2_angular_analysis()`

### 3.2 V2 (PhysicsBasedFitterV2)

**文件**：`Abel_backward_reconstruction_v2.py`

```python
fitter = PhysicsBasedFitterV2(
    n_pixels,
    sigma_psf=0.5 * scale_factor,
    sigma_pixel=0.3 * scale_factor,
    sigma_interp=0.55 * scale_factor
)
fitter.sigma_E = 0.005
fitter.vmi_k = 0.01 / (scale_factor ** 2)
params, r_grid, recon_profile = fitter.solve(image_raw)  # 传原始图像！
```

**重要**：V2 需要传入**原始图像**，因为它内部做背景估计。

**调整位置**：
- 噪声估计：`_estimate_noise()` - 使用极坐标图像外围 15%
- sigma 估计：`_phase1_estimate_sigma()` - 需要 `use_input_r=True`

**已知问题**：如果外围 15% 超出数据边界（全为 0），噪声估计会失败。

#### V2 峰检测参数调整（重要！）

V2 内部的峰检测可能过于严格或过于宽松。在 `compare_reconstruction_methods.py` 中，可以通过以下参数调整：

| 参数 | 含义 | 默认值 | 调整建议 |
|------|------|--------|----------|
| `v2_height_thresh` | 峰高度阈值（相对于最大值） | 0.08 (8%) | 增大→检测更少峰，减小→检测更多峰 |
| `v2_prominence_thresh` | 峰突出度阈值（相对于最大值） | 0.05 (5%) | 增大→过滤弱峰，减小→保留弱峰 |
| `v2_distance` | 相邻峰最小距离（像素） | 12 | 增大→合并近峰，减小→分离近峰 |

**使用示例**：

```python
# 在 compare_reconstruction_methods.py 中调用
v2_result = run_v2_reconstruction(
    image, 
    image_raw=image_raw, 
    mask_radius=25,
    # V2 峰检测参数
    v2_height_thresh=0.08,      # 8% 高度阈值
    v2_prominence_thresh=0.05,  # 5% 突出度阈值
    v2_distance=12              # 12 像素最小距离
)
```

**参数选择指南**：

| 期望峰数 | height_thresh | prominence_thresh | distance |
|----------|---------------|-------------------|----------|
| 2 个主峰 | 0.08-0.10 | 0.05-0.08 | 12-15 |
| 3-4 个峰 | 0.05-0.08 | 0.03-0.05 | 8-12 |
| 更多峰（敏感模式） | 0.02-0.05 | 0.01-0.03 | 5-8 |

**调试技巧**：
1. 先用宽松参数（height=0.02, prominence=0.01）看能检测到多少峰
2. 逐步增大阈值，直到只保留真实峰
3. 检查径向分布图确认峰位置

### 3.3 V3 (AbelReconstructorV3)

**文件**：`Abel_backward_reconstruction_v3.py`

```python
reconstructor = AbelReconstructorV3(
    sigma_psf=0.5 * scale_factor,
    sigma_pixel=0.3 * scale_factor,
    sigma_interp=0.55 * scale_factor
)
reconstructor.seed_finder.mask_radius = int(20 * scale_factor)

# 自定义背景区域（重要！）
def custom_identify_background_region(img, center=None):
    # 使用上面的自动检测算法
    reconstructor.cleaner._bg_mask = (r >= bg_r_inner) & (r <= bg_r_outer)
    return reconstructor.cleaner._bg_mask

reconstructor.cleaner.identify_background_region = custom_identify_background_region
params, metadata = reconstructor.reconstruct(image_raw, enforce_circularity=False)
```

**调整位置**：
- 背景区域：`DataCleaner.identify_background_region()` - 需要自定义
- 椭圆度：`DataCleaner.check_circularity()` - 容差 5%
- 峰检测：`SeedFinder._detect_peaks()` - 根据 SNR 调整
- 中心遮罩：`SeedFinder.mask_radius`

### 3.4 rBasex

```python
recon_image, distr = abel.rbasex.rbasex_transform(image, direction='inverse')
r, I, beta = distr.rIbeta()
```

**注意**：beta 值符号可能与 V1/V2/V3 相反。显示建议用 `log(1+I)` scale。

---

## 4. 数据椭圆度问题

### 4.1 椭圆度来源

- 探测器平面与飞行轴不垂直
- 残余电场/磁场
- 探测器非均匀性

### 4.2 检测椭圆度

**方法**：计算图像的二阶矩（惯性张量），得到 x 和 y 方向的标准差比值。

```python
# 使用 V3 的检测方法
circularity = reconstructor.cleaner.check_circularity(image)
print(f"sigma_x: {circularity['sigma_x']:.2f}")
print(f"sigma_y: {circularity['sigma_y']:.2f}")
print(f"Aspect ratio (sigma_x/sigma_y): {circularity['aspect_ratio']:.4f}")
print(f"Ellipticity |1 - ratio|: {circularity['ellipticity']:.4f}")
print(f"Is circular (< 5%): {circularity['is_circular']}")
```

**手动检测**：

```python
def detect_ellipticity(image):
    """检测图像椭圆度"""
    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    
    # 创建坐标网格
    y, x = np.ogrid[:ny, :nx]
    Y = y - cy
    X = x - cx
    
    # 使用图像强度作为权重
    weights = np.maximum(image, 0)
    total = np.sum(weights)
    
    # 计算二阶矩
    M_xx = np.sum(weights * X**2) / total
    M_yy = np.sum(weights * Y**2) / total
    M_xy = np.sum(weights * X * Y) / total  # 交叉项，检测旋转
    
    sigma_x = np.sqrt(M_xx)
    sigma_y = np.sqrt(M_yy)
    aspect_ratio = sigma_x / sigma_y
    
    return {
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'aspect_ratio': aspect_ratio,
        'ellipticity': abs(1 - aspect_ratio),
        'M_xy': M_xy  # 非零表示椭圆有旋转
    }
```

### 4.3 椭圆度校正方法

**最佳方法：在原始 XY 坐标上校正（推荐）**

在 binning 之前对原始 XY 坐标做仿射变换，这样只有一次插值，信息损失最小。

```python
def correct_ellipticity_xy(xy_data, aspect_ratio, center=(0, 0)):
    """在原始 XY 坐标上校正椭圆度
    
    Args:
        xy_data: (N, 2) 原始 XY 坐标
        aspect_ratio: sigma_x / sigma_y
        center: 椭圆中心 (x0, y0)
    
    Returns:
        校正后的 XY 坐标
    """
    x = xy_data[:, 0] - center[0]
    y = xy_data[:, 1] - center[1]
    
    # 计算缩放因子：将椭圆变成圆
    # 如果 aspect_ratio > 1，x 方向更宽，需要压缩 x
    # 如果 aspect_ratio < 1，y 方向更宽，需要压缩 y
    scale_x = 1.0 / np.sqrt(aspect_ratio)
    scale_y = np.sqrt(aspect_ratio)
    
    # 应用缩放
    x_corrected = x * scale_x + center[0]
    y_corrected = y * scale_y + center[1]
    
    return np.column_stack([x_corrected, y_corrected])

# 使用示例
aspect_ratio = circularity['aspect_ratio']
if abs(1 - aspect_ratio) > 0.02:  # 椭圆度 > 2%
    xy_corrected = correct_ellipticity_xy(xy_data, aspect_ratio)
    # 然后用校正后的坐标构建图像
    image = build_vmi_image(xy_corrected, n_bins=256)
```

**备选方法：在图像上校正**

如果只有图像没有原始坐标，可以用仿射变换，但会有二次插值损失。

```python
from scipy.ndimage import affine_transform

def correct_ellipticity_image(image, aspect_ratio):
    """在图像上校正椭圆度（有插值损失）"""
    ny, nx = image.shape
    cy, cx = ny / 2, nx / 2
    
    scale_x = 1.0 / np.sqrt(aspect_ratio)
    scale_y = np.sqrt(aspect_ratio)
    
    # 构建变换矩阵
    M = np.array([[scale_y, 0], [0, scale_x]])
    M_inv = np.linalg.inv(M)
    
    # 偏移量
    offset = np.array([cy, cx]) - M_inv @ np.array([cy, cx])
    
    corrected = affine_transform(image, M_inv, offset=offset, order=1)
    return corrected
```

### 4.4 椭圆校正引入的展宽（sigma_ellipse）

**只有在图像上做椭圆校正才会引入展宽！**

| 校正方式 | sigma_ellipse | 说明 |
|----------|---------------|------|
| 在原始 XY 坐标上校正 | **0** | 推荐，无插值，无展宽 |
| 在图像上仿射变换 | ~0.4-0.6 px | 有插值展宽，且各向异性 |

**在图像上校正的展宽估计**（仅供参考）：

```python
def estimate_ellipse_correction_sigma(aspect_ratio, interp_order=1):
    """估计椭圆校正引入的各向异性展宽（仅当在图像上校正时）"""
    scale_x = 1.0 / np.sqrt(aspect_ratio)
    scale_y = np.sqrt(aspect_ratio)
    
    # 插值核的基础展宽
    sigma_interp_base = 0.4 if interp_order == 1 else 0.55
    
    # 各向异性展宽
    sigma_ellipse_x = sigma_interp_base / scale_x if scale_x < 1 else sigma_interp_base * scale_x
    sigma_ellipse_y = sigma_interp_base / scale_y if scale_y < 1 else sigma_interp_base * scale_y
    
    # 等效各向同性展宽
    sigma_ellipse = np.sqrt(sigma_ellipse_x * sigma_ellipse_y)
    return sigma_ellipse
```

### 4.5 展宽如何消除？

**不需要反卷积！**

反卷积会放大噪声，是病态问题。正确的做法是：

1. **前向建模**（V2/V3 的方法）：
   - 模型已经包含系统展宽：`测量信号 = 物理信号 ⊗ 系统展宽`
   - 拟合输出的 sigma_phys 已经是扣除系统展宽后的物理展宽

2. **如果在图像上做了椭圆校正**：把 sigma_ellipse 加到 sigma_sys 里
   ```python
   sigma_sys = np.sqrt(sigma_psf**2 + sigma_pixel**2 + sigma_interp**2 + sigma_ellipse**2)
   ```

3. **如果在原始 XY 上校正**：sigma_ellipse = 0，不需要额外处理

**最佳实践**：在原始 XY 坐标上做椭圆校正，避免所有这些问题。

### 4.6 处理策略选择

| 情况 | 处理方法 | sigma_ellipse | 备注 |
|------|----------|---------------|------|
| 有原始 XY | 在 XY 坐标上校正 | 0 | **推荐** |
| 只有图像 | 在图像上仿射变换 | 加到 sigma_sys | 有信息损失 |
| 椭圆度是物理效应 | 不校正 | 0 | |
| 椭圆度很小 (< 2%) | 可以忽略 | 0 | |

---

## 5. 常见问题和解决方案

### 5.1 峰检测数量不一致

**症状**：V1 检测 3 个峰，V2 检测 2 个，V3 检测 8 个

**解决**：
- 统一背景区域定义
- 调整峰检测阈值
- 过滤信号区域外的假峰

### 5.2 V2 漏检外侧峰

**原因**：`_estimate_noise()` 使用的外围区域可能是全 0

**解决**：传递原始图像给 V2，或修改噪声估计区域

### 5.3 beta 值符号不一致

**原因**：偏振轴方向定义不同

**解决**：确认偏振轴方向，必要时取反

### 5.4 外围假峰

**解决**：
```python
signal_radius = bg_r_inner
params = [p for p in params if p['r'] <= signal_radius]
```

---

## 6. 推荐工作流程

1. **加载数据** -> sio.loadmat()
2. **选择图像尺寸** -> 根据事件数：<50万用256，>100万用512
3. **构建 VMI 图像** -> np.histogram2d()
4. **分析径向分布** -> 找数据边界、信号边界、背景区域
5. **背景减除** -> 减均值，clip，外围设0
6. **运行重建** -> V1/V2/V3/rBasex
7. **比较结果** -> 峰位置、beta 符号
8. **调整参数** -> 如需要

---

## 7. 参数速查表

### 7.1 按图像尺寸缩放

```python
scale_factor = n_pixels / 256.0
sigma_psf = 0.5 * scale_factor
sigma_pixel = 0.3 * scale_factor
sigma_interp = 0.55 * scale_factor
mask_radius = int(20 * scale_factor)
vmi_k = 0.01 / (scale_factor ** 2)
```

### 7.2 各方法输入要求

| 方法 | 输入图像 | 背景处理 | 备注 |
|------|----------|----------|------|
| V1 | 预处理后 | 内部自动 | 最简单 |
| V2 | **原始图像** | 内部自动 | 需要原始图像做噪声估计 |
| V3 | **原始图像** | 需自定义 | 需要自定义背景区域 |
| rBasex | 预处理后 | 无 | beta 符号可能相反 |

---

## 8. 中心区域遮罩（mask_radius）

### 8.1 为什么需要中心遮罩

VMI 图像中心通常有一个很高的峰，来源于：
- **背景气体电离**：真空腔内残余气体被电离，产生低动能电子
- **散射电子**：与腔壁或电极碰撞后的二次电子
- **阈值电子**：零动能光电子

这些信号是真实的物理信号，但通常不是你要分析的目标。如果不处理，会：
- 干扰峰检测（被误认为是最强峰）
- 影响归一化（压低其他峰的相对强度）
- 增加拟合难度

### 8.2 处理方法

**正确做法：在峰检测阶段排除，而不是删除数据**

```python
# 不要这样做（会破坏 Abel 变换）
image[r < mask_radius] = 0  # ❌ 错误

# 正确做法：保留数据，只在峰检测时排除
mask_radius = 25  # 排除 r < 25 px 的区域
params = [p for p in params if p['r'] > mask_radius]  # ✓ 正确
```

**原因**：Abel 逆变换需要完整的投影数据，中心挖洞会引入数值伪影。

### 8.3 各方法的 mask_radius 设置

**V1** (`Abel_backward_reconstruction.py`)：
```python
fitter = PhysicsBasedFitter(n_pixels, mask_radius=25)
```

**V2** (`Abel_backward_reconstruction_v2.py`)：
```python
fitter = PhysicsBasedFitterV2(n_pixels, ..., mask_radius=25)
```

**V3** (`Abel_backward_reconstruction_v3.py`)：
```python
reconstructor = AbelReconstructorV3(..., mask_radius=25)
```

### 8.4 mask_radius 的选择

| 图像尺寸 | 建议 mask_radius | 说明 |
|----------|------------------|------|
| 256×256 | 15-25 px | 约 6-10% 的半径 |
| 512×512 | 25-50 px | 约 5-10% 的半径 |

**选择原则**：
- 覆盖中心高峰的范围
- 不要太大，避免遮盖真实的低能峰
- 可以先画径向分布，看中心峰延伸到哪里

```python
# 从径向分布确定 mask_radius
import matplotlib.pyplot as plt
plt.plot(radial_profile)
plt.xlabel('r (px)')
plt.ylabel('Intensity')
# 找到中心峰下降到背景水平的位置
```

---

## 9. 峰检测参数调整

### 9.1 scipy.signal.find_peaks 参数

所有方法内部都使用 `scipy.signal.find_peaks()` 进行峰检测，关键参数：

| 参数 | 含义 | 调整建议 |
|------|------|----------|
| `height` | 峰的最小高度 | 设为 max_value 的 3-10%，SNR 高时用小值 |
| `prominence` | 峰的突出度 | 设为 max_value 的 2-8%，过滤噪声峰 |
| `distance` | 相邻峰的最小距离 | 5-12 px，防止一个峰被检测为多个 |

### 9.2 V1 峰检测调整

**文件**：`Abel_backward_reconstruction.py`，函数 `_phase1_find_peaks()`

```python
# 在 PhysicsBasedFitter 类中
def _phase1_find_peaks(self, radial_profile):
    max_val = np.max(radial_profile)
    
    # 调整这些参数
    peaks, properties = find_peaks(
        radial_profile,
        height=max_val * 0.05,      # 峰高度阈值：5% of max
        prominence=max_val * 0.03,  # 突出度阈值：3% of max
        distance=8                   # 最小峰间距：8 px
    )
    return peaks
```

### 9.3 V2 峰检测调整

**文件**：`Abel_backward_reconstruction_v2.py`，函数 `_phase1_find_peaks()`

V2 根据 SNR 自适应调整阈值：

```python
def _phase1_find_peaks(self, radial_profile, snr):
    max_val = np.max(radial_profile)
    
    if snr > 50:  # 高 SNR
        height_thresh = max_val * 0.03
        prominence_thresh = max_val * 0.02
        distance = 5
    elif snr > 20:  # 中等 SNR
        height_thresh = max_val * 0.05
        prominence_thresh = max_val * 0.03
        distance = 8
    else:  # 低 SNR
        height_thresh = max_val * 0.10
        prominence_thresh = max_val * 0.08
        distance = 12
    
    peaks, _ = find_peaks(radial_profile, 
                          height=height_thresh,
                          prominence=prominence_thresh,
                          distance=distance)
    return peaks
```

### 9.4 V3 峰检测调整

**文件**：`Abel_backward_reconstruction_v3.py`，类 `SeedFinder`，函数 `_detect_peaks()`

V3 的阈值设置与 V2 类似，但在 `SeedFinder` 类中：

```python
class SeedFinder:
    def __init__(self, mask_radius=15):
        self.mask_radius = mask_radius  # 中心遮罩半径
    
    def _detect_peaks(self, profile_3d, snr=10.0):
        max_val = np.max(profile_3d)
        
        # 根据 SNR 调整阈值
        if snr > 50:
            height_thresh = max_val * 0.03
            prominence_thresh = max_val * 0.02
            distance = 5
        elif snr > 20:
            height_thresh = max_val * 0.05
            prominence_thresh = max_val * 0.03
            distance = 8
        else:
            height_thresh = max_val * 0.10
            prominence_thresh = max_val * 0.08
            distance = 12
        
        peaks, _ = find_peaks(profile_3d,
                              height=height_thresh,
                              prominence=prominence_thresh,
                              distance=distance)
        
        # 过滤中心区域
        peaks = [p for p in peaks if p >= self.mask_radius]
        return peaks
```

### 9.5 调整建议

**检测到太多峰**：
- 增大 `height` 和 `prominence` 阈值
- 增大 `distance` 参数
- 增大 `mask_radius` 排除中心噪声

**漏检峰**：
- 减小 `height` 和 `prominence` 阈值
- 减小 `distance` 参数
- 检查背景是否正确减除

**峰位置不准**：
- 检查图像中心是否正确
- 检查是否有椭圆度问题

---

## 10. 输入参数估计方法

### 9.1 从已知能量峰估计 VMI 系数

如果知道某个峰对应的电子能量 E (eV)，可以估计 VMI 系数：

```python
# VMI 关系：E = k * r^2
# 其中 r 是峰位置（像素），k 是 VMI 系数

# 从已知峰估计
E_known = 1.5  # eV，已知能量
r_peak = 100   # px，对应的峰位置

vmi_k = E_known / (r_peak ** 2)  # eV/px^2
```

### 9.2 sigma_psf 的估计（探测器空间分辨率）

**物理来源**：
- DLD：延迟线的时间分辨率 → 位置分辨率
- MCP：微通道板的孔径（~10-25 μm）是分辨率下限
- 电子光学：离子/电子在飞行过程中的散射

**估计方法**：

**方法 1：从厂商规格**
```python
# DLD 厂商给出的空间分辨率（通常是 FWHM）
dld_fwhm_mm = 0.15  # mm，典型值 100-200 able

# 转换为 sigma
dld_sigma_mm = dld_fwhm_mm / 2.355

# 转换为像素单位
pixel_size_mm = (x_max - x_min) / n_bins
sigma_psf = dld_sigma_mm / pixel_size_mm  # px
```

**方法 2：从已知窄峰测量**
```python
# 如果有一个已知物理展宽很小的峰（sigma_phys ≈ 0）
# 测量的宽度就近似等于系统展宽

sigma_measured = params[0]['sigma']  # 从拟合结果
# 如果 sigma_phys ≈ 0，则 sigma_measured ≈ sigma_sys
# sigma_sys^2 = sigma_psf^2 + sigma_pixel^2 + sigma_interp^2
# 可以反推 sigma_psf
```

**方法 3：从 MCP 孔径估计（下限）**
```python
# MCP 孔径是分辨率的物理下限
mcp_pore_size = 0.025  # mm，典型值 10-25 able
sigma_psf_min = mcp_pore_size / pixel_size_mm  # px
# 实际 sigma_psf 通常比这个大 2-5 倍
```

**典型值**：
| 探测器 | 分辨率 (FWHM) | sigma_psf |
|--------|---------------|-----------|
| 高性能 DLD | ~50 able | ~20 able |
| 普通 DLD | ~100-200 able | ~40-80 able |
| MCP 限制 | ~25 able | ~10 able |

### 9.3 sigma_psf 和 sigma_pixel 的关系（重要！）

**它们是独立的，不重叠**：

```
真实位置 → [sigma_psf] → 测量的连续坐标 → [sigma_pixel] → 像素坐标
           (探测器分辨率)                    (binning 量化)
```

- **sigma_psf**：探测器在**连续空间**的分辨率，发生在 binning 之前
- **sigma_pixel**：**离散化过程**引入的展宽，发生在 binning 时

**但要注意数据来源**：

| sigma_psf 来源 | 是否包含 sigma_pixel | 处理方式 |
|----------------|---------------------|----------|
| 厂商规格（连续空间 FWHM） | 否 | 两者都要算 |
| 从你的数据测量（已 binning） | 是 | 不要再加 sigma_pixel |

**实际建议**：
- 如果用厂商规格：`sigma_sys² = sigma_psf² + sigma_pixel² + sigma_interp²`
- 如果从数据测量：测量值已经是 `sqrt(sigma_psf² + sigma_pixel²)`，只需再加 sigma_interp

### 9.4 sigma_pixel 的计算（像素响应函数展宽）

**本质**：像素响应函数是**矩形函数（方波）**，其标准差 = width/sqrt(12)。

**sigma_pixel 是你做 histogram binning 时产生的**：
```python
# 这一步产生 sigma_pixel
image, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins, ...)
```

DLD 输出的是高精度连续坐标（精度 ~1μm），但 `np.histogram2d()` 把它们量化到更大的像素（比如 320μm/pixel）。这个量化过程引入了 sigma_pixel。

**物理图像**：
- 落在同一个像素内的所有点都被计入该像素中心
- 这等效于用矩形函数（宽度 = 1 像素）对连续信号做卷积
- 矩形函数的标准差 = 1/sqrt(12) ≈ 0.289 像素

**什么时候有 sigma_pixel**：

| 数据类型 | sigma_pixel | 说明 |
|----------|-------------|------|
| DLD 连续坐标 + 你做 binning | **0.289 px** | 你的情况 |
| DLD 已经输出离散像素 | 0 | 已包含在 sigma_psf |
| CCD/CMOS 图像 | 0 | 已包含在 PSF |

**示例**：
- DLD 坐标 binning 到 256×256（320μm/px）：sigma_pixel = 0.289 px = 92μm
- DLD 坐标 binning 到 512×512（160μm/px）：sigma_pixel = 0.289 px = 46μm

**注意**：sigma_pixel 以像素为单位是固定的（0.289 px），但以物理单位（μm）会随 binning 变化。

```python
# DLD 连续坐标 + 你做 binning
sigma_pixel = 1.0 / np.sqrt(12)  # ≈ 0.289 px

# DLD 已输出离散像素 或 CCD/CMOS
sigma_pixel = 0
```

### 9.5 sigma_interp 的计算（极坐标插值展宽）

笛卡尔坐标到极坐标的转换需要插值，不同插值方法引入不同的展宽：

| 插值方法 | 阶数 | sigma_interp (px) | 说明 |
|----------|------|-------------------|------|
| 最近邻 | 0 | ~0.5 | 最差，有锯齿 |
| 双线性 | 1 | ~0.4 | 常用 |
| 3阶样条 | 3 | ~0.55 | 平滑但有展宽 |

```python
# 3阶样条插值的展宽（经验值）
sigma_interp = 0.55  # px
```

**估计方法**：可以用 delta 函数（单像素亮点）测试插值展宽：

```python
def measure_interp_sigma():
    """测量插值展宽"""
    # 创建 delta 函数图像
    n = 256
    image = np.zeros((n, n))
    image[n//2, n//2 + 50] = 1.0  # 在 r=50 处放一个点
    
    # 转换到极坐标（使用 3 阶样条）
    from scipy.ndimage import map_coordinates
    # ... 极坐标转换代码 ...
    
    # 测量极坐标图像中该点的 FWHM
    # sigma_interp = FWHM / 2.355
```

### 9.6 总系统展宽

三个展宽独立，平方和：

```python
sigma_sys = np.sqrt(sigma_psf**2 + sigma_pixel**2 + sigma_interp**2)

# 典型值（256x256）：
# sigma_psf = 0.5, sigma_pixel = 0.3, sigma_interp = 0.55
# sigma_sys = sqrt(0.25 + 0.09 + 0.30) = sqrt(0.64) ≈ 0.8 px
```

### 9.6 如何消除系统展宽

有三种方法处理系统展宽：

**方法 1：反卷积（不推荐）**

直接从图像中去除展宽，但会放大噪声，数值不稳定。

```python
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener

# Wiener 反卷积（需要知道噪声水平）
# 不推荐：对噪声敏感，容易产生伪影
deconvolved = wiener(image, psf_kernel, noise_power)
```

**问题**：
- 放大高频噪声
- 需要精确知道 PSF 形状
- 容易产生振铃伪影
- 对 Poisson 噪声效果差

**方法 2：前向建模（V2/V3 的方法，推荐）**

不去除展宽，而是在前向模型中包含展宽效应。拟合时自动考虑。

```python
# V2/V3 的前向模型
# 3D 分布 → Abel 投影 → PSF 卷积 → 像素化 → 2D 图像
#
# 拟合参数是 3D 分布的参数（r, sigma_phys, beta）
# 系统展宽在前向模型中自动处理

fitter = PhysicsBasedFitterV2(
    n_pixels,
    sigma_psf=0.5,      # 告诉模型 PSF 展宽
    sigma_pixel=0.3,    # 告诉模型像素化展宽
    sigma_interp=0.55   # 告诉模型插值展宽
)
# 拟合结果的 sigma 是物理展宽 sigma_phys，不包含系统展宽
```

**优点**：
- 数值稳定
- 不放大噪声
- 直接得到物理参数

**方法 3：后处理校正（简单情况）**

如果只需要峰宽度，可以从测量值中减去系统展宽。

```python
# 测量的峰宽度
sigma_measured = 3.5  # px

# 系统展宽
sigma_sys = np.sqrt(0.5**2 + 0.3**2 + 0.55**2)  # ≈ 0.8 px

# 物理展宽
sigma_phys = np.sqrt(sigma_measured**2 - sigma_sys**2)  # ≈ 3.4 px
```

**适用条件**：
- 展宽都是高斯型
- sigma_measured >> sigma_sys（否则误差大）

**总结**：

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 反卷积 | 直接得到"真实"图像 | 放大噪声，不稳定 | 高 SNR，已知精确 PSF |
| 前向建模 | 稳定，直接得到物理参数 | 需要正确的前向模型 | **推荐**，V2/V3 使用 |
| 后处理校正 | 简单 | 只能校正峰宽度 | 快速估计 |

### 9.7 从峰宽度反推物理展宽

测量到的峰宽度包含多个贡献：

```
sigma_measured^2 = sigma_phys^2 + sigma_psf^2 + sigma_pixel^2 + sigma_interp^2
                 = sigma_phys^2 + sigma_sys^2
```

反推物理展宽：

```python
# 从拟合结果获取测量宽度
sigma_measured = params[0]['sigma']  # 从重建结果

# 系统展宽
sigma_sys = np.sqrt(sigma_psf**2 + sigma_pixel**2 + sigma_interp**2)

# 物理展宽
sigma_phys = np.sqrt(max(0, sigma_measured**2 - sigma_sys**2))
```

### 9.7 从径向分布估计初始参数

在运行完整重建之前，可以先从径向分布粗略估计参数：

```python
def estimate_initial_params(image):
    """从径向分布估计初始参数"""
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    
    # 计算径向分布
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    r_max = n // 2
    
    radial_sum = np.bincount(r.ravel(), weights=image.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count = np.bincount(r.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count
    
    # 找峰
    from scipy.signal import find_peaks, peak_widths
    peaks, properties = find_peaks(radial_profile, 
                                   height=np.max(radial_profile)*0.05,
                                   prominence=np.max(radial_profile)*0.03)
    
    # 估计峰宽度
    widths, width_heights, left_ips, right_ips = peak_widths(
        radial_profile, peaks, rel_height=0.5
    )
    
    params = []
    for i, peak in enumerate(peaks):
        params.append({
            'r': peak,
            'sigma_est': widths[i] / 2.355,  # FWHM -> sigma
            'height': radial_profile[peak]
        })
    
    return params, radial_profile
```

### 9.8 SNR 估计

```python
def estimate_snr(image, bg_r_inner, bg_r_outer):
    """估计信噪比"""
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    # 信号：最大值
    signal = np.max(image)
    
    # 噪声：背景区域的标准差
    bg_mask = (r >= bg_r_inner) & (r <= bg_r_outer)
    noise = np.std(image[bg_mask])
    
    snr = signal / (noise + 1e-10)
    return snr
```

---

## 11. 参考代码

完整比较脚本见 `compare_reconstruction_methods.py`。
