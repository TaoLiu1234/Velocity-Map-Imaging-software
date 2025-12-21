# V2 vs V3.5 技术细节对比

## 1. 整体架构对比

| 方面 | V2 | V3.5 |
|------|-----|------|
| 设计理念 | 物理反卷积链 | 前向模型拟合 |
| 核心方法 | 2D Wiener反卷积 + Abel逆变换 | 数值Abel核 + Poisson MLE |
| 统计模型 | 加权最小二乘 (WLS) | 泊松最大似然 (MLE) |
| 参数估计 | 逐步反卷积 | 全局优化 |

## 2. Phase 0: 预处理

### V2 (新版本)
```
探测器图像(x,y) 
  → 2D Wiener反卷积(去除PSF+像素化展宽) 
  → 极坐标变换(r,θ) 
  → 背景估计+去除 
  → 角向滤波(保留DC+k=2) 
  → 径向滤波(自适应)
```

**关键改进**：在笛卡尔坐标下做2D反卷积，物理上正确
- PSF展宽发生在探测器平面(x,y)，应该在(x,y)空间去除
- 使用Wiener滤波：`H_wiener = H* / (|H|² + NSR)`
- NSR根据图像SNR自适应调整

### V3.5
```
探测器图像(x,y) 
  → 背景估计(不减法) 
  → 椭圆检测(可选) 
  → 极坐标变换(面积权重) 
  → 角向滤波
```

**特点**：
- 不做反卷积，PSF展宽在前向模型中处理
- 背景不减法，在Phase 4拟合时作为参数

## 3. Phase 1: 径向分析

### V2
```python
# 流程
proj_profile = angular_average(polar)  # 角向平均
peaks = find_peaks(proj_profile)        # 峰值检测
abel_profile = abel_inverse(proj_profile)  # Abel逆变换
sigma, amp = gaussian_fit(abel_profile)    # 高斯拟合
sigma_corrected = sqrt(sigma² - σ_interp²) # 只去除插值展宽
```

**关键点**：
- PSF已在Phase 0去除，这里只需要去除插值展宽(~0.55 px)
- 使用Hansen-Law Abel逆变换
- 高斯拟合估计sigma

### V3.5
```python
# 流程
profile_2d = angular_integrate(polar)   # 角向积分
profile_3d = abel_inverse(profile_2d)   # Abel逆变换
peaks = detect_peaks(profile_3d)        # 峰值检测
sigma, amp = estimate_sigma(profile_3d) # 高斯拟合
```

**关键点**：
- 同样使用Hansen-Law Abel逆变换
- 不做反卷积，sigma包含所有展宽
- 后续在Phase 4用前向模型处理展宽

## 4. Phase 2: 角向分析 (β估计)

### V2 (复杂版本)
```python
# 流程
polar_3d = abel_inverse_each_angle(polar)  # 每个角度做Abel逆变换
polar_deconv = angular_deconvolve(polar_3d) # 角向反卷积
# 从外到内估计beta，减去外层贡献
for peak in sorted_by_r_descending:
    polar_corrected = polar_3d - outer_contribution
    beta = estimate_beta(polar_corrected)
    outer_contribution += build_peak_model(peak)
```

**特点**：
- 对每个角度单独做Abel逆变换
- 角向反卷积去除探测器展宽对k=2成分的衰减
- 从外到内迭代，减去外层峰的贡献

### V3.5 (简单版本)
```python
# 流程
angular = mean(polar[r-3:r+3, :])  # 峰值附近角向平均
fft = FFT(angular)
dc = |fft[0]| / n_theta
cos2_amp = 2 * |fft[2]| / n_theta
beta = 4 * cos2 / (3 * dc - cos2)
```

**特点**：
- 直接在极坐标图像上做FFT
- 不做角向反卷积
- 不考虑多峰重叠

## 5. Phase 3/4: 前向拟合

### V2
- 可选的前向优化
- 使用加权最小二乘 (WLS)
- 通常跳过（Phase 2结果已经足够好）

### V3.5
```python
# 前向模型
model = Σ_k [A_k × numerical_kernel(r_k, σ_k) × (1 + β_k × P2)] + background

# 优化
minimize Cash_statistic(model, data)  # Poisson MLE
# Cash = 2 × Σ [M - D × ln(M)]
```

**特点**：
- 使用数值Abel投影核（预计算模板）
- 泊松统计正确处理低计数
- 直接拟合BR（产率），而不是振幅

## 6. BR计算

### V2
```python
# 3D高斯积分公式
# N = 4π × A_3D × σ × √(2π) × (r² + σ²)
intensity = amp * sigma * (r² + σ²)
BR = intensity / total_intensity
```

**问题**：
- 使用的是测量的sigma（包含系统展宽）
- 公式假设3D高斯分布

### V3.5
```python
# 直接从拟合结果获取
# BR ∝ A_3D × σ_phys × r²
intensity = amp * sigma * r²
BR = intensity / total_intensity

# 或者：Phase 4直接拟合BR
# 参数向量 [r, σ, Y, β] 其中Y就是BR
```

**特点**：
- 区分σ_phys（物理展宽）和σ_sys（系统展宽）
- Phase 4可以直接拟合BR，与β解耦

## 7. 性能对比分析

### 测试结果 (1e4 ~ 1e8 events)

| 参数 | V2 | V3.5 | 分析 |
|------|-----|------|------|
| r(%) | **0.45** | 2.06 | V2更好 |
| σ(%) | **21.72** | 59.23 | V2显著更好 |
| β(%) | **5.31** | 12.75 | V2更好 |
| BR(%) | **14.45** | 29.68 | V2更好 |
| Det% | 92.6 | **100** | V3.5更稳定 |

### 为什么V2现在更好？

1. **2D Wiener反卷积的优势**
   - 在物理正确的坐标系(x,y)下去除PSF
   - 避免了极坐标下反卷积的物理不一致性
   - Wiener滤波平衡了去噪和保真

2. **角向反卷积的优势**
   - V2对每个角度做Abel逆变换后再做角向反卷积
   - 正确补偿了探测器展宽对k=2成分的衰减
   - V3.5直接在投影面上估计β，没有这个校正

3. **多峰处理的优势**
   - V2从外到内迭代，减去外层峰的贡献
   - V3.5的简单FFT方法在多峰重叠时不准确

### 为什么V3.5在高计数时BR更好？

1. **直接拟合BR**
   - V3.5的Phase 4直接拟合产率Y，不需要事后计算
   - 数学上与β解耦

2. **Poisson MLE**
   - 在高计数时，Poisson → Gaussian，MLE更准确
   - 但在低计数时，V3.5的前向模型可能过拟合

## 8. 关键差异总结

| 方面 | V2 | V3.5 |
|------|-----|------|
| PSF处理 | 2D Wiener反卷积 | 前向模型包含 |
| Abel逆变换 | 反卷积后再做 | 直接做 |
| β估计 | 角向反卷积 + 迭代 | 直接FFT |
| 统计模型 | WLS | Poisson MLE |
| BR计算 | 事后积分 | 直接拟合 |
| 复杂度 | 中等 | 高 |
| 速度 | 快 | 慢 |

## 9. 建议

1. **对于一般使用**：推荐V2
   - 更快、更准确、更稳定
   - 2D Wiener反卷积是物理正确的方法

2. **对于高计数、需要精确BR**：考虑V3.5
   - Poisson MLE在高计数时更准确
   - 直接拟合BR避免了事后计算的误差

3. **可能的改进方向**：
   - 将V2的2D Wiener反卷积引入V3.5
   - 将V3.5的Poisson MLE引入V2
   - 统一BR计算方法
