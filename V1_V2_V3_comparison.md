# V1 vs V2 vs V3.5 代码结构与物理实现对比

## 总体架构对比

| 特性 | V1 | V2 | V3.5 |
|------|-----|-----|------|
| **代码行数** | ~2400 | ~1400 | ~3600 |
| **类结构** | 单一类 `PhysicsBasedFitter` | 单一类 `PhysicsBasedFitterV2` | 模块化：`DataCleaner`, `PolarTransformer`, `SeedFinder`, `ForwardFitter` |
| **Phase数量** | 3 (Radial, Angular, Forward) | 3 (Preprocess, Radial, Angular) + 可选Phase3 | 5 (Clean, Polar, Seed, Forward, BR) |
| **默认模式** | 完整前向拟合 | 跳过Phase3 (`skip_phase3=True`) | 跳过前向拟合 (`skip_forward_fit=True`) |

---

## Phase 0: 预处理/数据净化

### V1: 共享数据初始化
```python
def _init_shared_data(self, image_2d, n_theta=720):
    # 1. 极坐标转换（三次插值）
    polar_image = map_coordinates(image_2d, [y_cart, x_cart], order=3)
    
    # 2. 噪声估计：从最大peak位置到r_max的区域
    # 使用MAD估计（robust）
    noise_mad = np.median(np.abs(noise_pixels - noise_median))
    readout_std = 1.4826 * noise_mad
    
    # 3. 频域背景估计
    baseline = self._estimate_baseline_frequency_domain(signal_region, noise_median)
    
    # 4. 减去baseline
    signal_region_corrected = np.maximum(signal_region - baseline, 0)
```

**物理性评价**：
- ✅ 使用MAD估计噪声（对异常值鲁棒）
- ✅ 频域分析分离背景和信号
- ⚠️ 直接减去baseline可能引入负值问题

### V2: 简化预处理
```python
def _phase0_preprocess(self, image_2d, n_theta=720):
    # 1. xy → rθ 转换（引入σ_interp展宽）
    polar_raw = map_coordinates(image_2d, [y_cart, x_cart], order=3)
    
    # 2. 噪声估计：外围85%-100%区域
    baseline = np.percentile(noise_region, 25)  # 25%分位数
    
    # 3. 99%分位数归一化
    norm_factor = np.percentile(polar, 99)
    polar_norm = polar / norm_factor
    
    # 4. 角向滤波：只保留DC和k=2
    # 5. 径向滤波：根据SNR选择策略
```

**物理性评价**：
- ✅ 明确的物理展宽模型（σ_psf, σ_pixel, σ_interp）
- ✅ 角向滤波只保留物理相关成分（DC + k=2）
- ✅ 根据SNR自适应选择滤波策略
- ⚠️ 25%分位数作为baseline可能不够准确

### V3.5: 模块化数据净化
```python
class DataCleaner:
    def clean(self, image, auto_center=True, subtract_background=True):
        # 1. 自动中心精修（互相关方法，精度0.1像素）
        self.auto_center_refinement(image)
        
        # 2. 识别背景区域（外围15%）
        bg_mask = self.identify_background_region(image, self.center)
        
        # 3. 估算背景均值（可选是否减去）
        self.mu_total = np.mean(bg_pixels)
        
        # 4. 椭圆度检测
        circularity = self.check_circularity(image)
```

**物理性评价**：
- ✅ 亚像素级中心精修（互相关方法）
- ✅ 椭圆度检测和校正
- ✅ 支持不减背景模式（用于Poisson MLE）
- ✅ 模块化设计，可独立测试

---

## Phase 1: 径向分析（Peak检测 + σ估计）

### V1: 多方法融合
```python
def _phase1_estimate_sigma_from_abel(self, abel_profile, proj_peak_r):
    # 方法1: scipy peak_widths（最高权重3.0）
    widths_result = peak_widths(abel_profile, [abel_pk], rel_height=0.5)
    sigma_scipy = fwhm_scipy / 2.355
    
    # 方法2: 局部高斯拟合（权重2.0）
    popt, _ = curve_fit(gaussian, r_fit, y_fit, p0=p0, bounds=bounds)
    sigma_gauss = popt[2]
    
    # 方法3: 手动FWHM（权重1.5）
    # 方法4: 二阶矩（权重1.0）
    
    # 加权平均
    sigma_final = np.sum(sigma_candidates * weights) / np.sum(weights)
```

**物理性评价**：
- ✅ 多方法融合提高鲁棒性
- ✅ 使用Abel逆变换后的profile估计σ
- ⚠️ 没有考虑系统展宽的反卷积

### V2: 反卷积 + 高斯拟合
```python
def _phase1_radial_analysis(self):
    # 1. 角向平均得到投影面profile
    proj_profile = np.mean(self._polar, axis=1)
    
    # 2. 反卷积去除探测器展宽
    proj_deconv = self._deconvolve_profile(proj_profile, sigma_det)
    
    # 3. Abel逆变换
    abel_profile = abel.hansenlaw.hansenlaw_transform(proj_deconv, direction='inverse')
    
    # 4. 高斯拟合估计sigma
    # r位置用整数peak位置（最准确）
    return float(pk_r), float(sigma_fit), float(amp_fit)

def _deconvolve_sigma(self, sigma_measured, r):
    # 去除探测器展宽
    sigma_sq = sigma_measured**2 - sigma_det**2
    # Abel逆变换残余展宽校正（经验因子1.4）
    sigma_true = sigma_after_det / 1.4
```

**物理性评价**：
- ✅ 明确的展宽模型：σ_total² = σ_laser² + σ_psf² + σ_pixel² + σ_interp²
- ✅ Wiener滤波反卷积
- ⚠️ Abel逆变换残余展宽使用经验因子1.4

### V3.5: 数值投影核
```python
class NumericalKernelGenerator:
    """
    V3.5改进：Abel奇点附近10×超采样
    
    物理背景：
    - Abel变换的积分核是 1/√(r²-R²)
    - 在R→r时，投影函数变化率趋于无穷大
    - 需要自适应采样网格
    """
    def _precompute_single(self, r_ref):
        # 外侧区域（稀疏采样）
        r_outer_left = np.linspace(0, r_ref - sigma_ref, n_outer // 2)
        
        # 内侧区域（密集采样，奇点附近）
        r_inner = np.linspace(r_ref - sigma_ref, r_ref + sigma_ref, n_inner)
        
        # Forward Abel transform
        profile_2d = abel.hansenlaw.hansenlaw_transform(profile_3d, direction='forward')
```

**物理性评价**：
- ✅ 使用真实Abel投影形状（非高斯近似）
- ✅ 奇点附近超采样消除离散误差
- ✅ 线性插值保证梯度连续性
- ✅ 位移非不变性修正

---

## Phase 2: 角向分析（β提取）

### V1: FFT + 拟合融合
```python
def _phase2_estimate_beta_fft(self, angular_profile):
    """
    理论推导：
    I(θ) = I₀[1 + β·P₂(cosθ)] = I₀[1 + β·(3cos²θ - 1)/2]
    
    使用 cos²θ = (1 + cos2θ)/2：
    I(θ) = I₀(1 + β/4) + I₀·(3β/4)·cos2θ
    
    FFT分析：
    - DC分量 (k=0): A₀ = I₀(1 + β/4)
    - cos2θ分量 (k=2): 幅度 = I₀·|3β/4|
    
    β = 4·cos2_signed / (3·DC - cos2_signed)
    """
    fft = np.fft.fft(angular_profile)
    dc = np.abs(fft[0]) / n
    cos2_amp = 2 * np.abs(fft[2]) / n
    phase = np.angle(fft[2])
    sign = 1.0 if np.abs(phase) < np.pi/2 else -1.0
```

**物理性评价**：
- ✅ 正确的FFT分析公式
- ✅ 相位判断β符号
- ✅ 多半径加权平均
- ⚠️ 没有考虑探测器展宽对k=2成分的衰减

### V2: 角向反卷积
```python
def _deconvolve_angular(self, polar, sigma_det):
    """
    角向展宽 σ_θ = σ_det / r（弧度）
    k=2成分的衰减因子 = exp(-0.5 × (2 × σ_θ)²)
    """
    for r_idx in range(n_r):
        sigma_theta = sigma_det / r_idx
        
        for k in range(len(fft) // 2 + 1):
            attenuation = np.exp(-0.5 * (k * sigma_theta) ** 2)
            if attenuation > 0.1:  # 避免噪声放大
                fft_deconv[k] = fft[k] / attenuation

def _phase2_angular_analysis(self, params):
    # 1. 对每个角度做Abel逆变换
    polar_3d = self._get_abel_inverted_polar()
    
    # 2. 角向反卷积
    polar_3d_deconv = self._deconvolve_angular(polar_3d, sigma_det)
    
    # 3. 从外到内估计beta（减去外层贡献）
```

**物理性评价**：
- ✅ 考虑探测器展宽对角向分布的影响
- ✅ 从外到内估计，减去外层贡献
- ✅ 物理正确的衰减因子公式
- ⚠️ 反卷积可能放大噪声

### V3.5: 极坐标空间直接估计
```python
class SeedFinder:
    def _estimate_beta(self, polar, r_center, theta_grid):
        """
        在峰值附近取平均，FFT分析k=2成分
        """
        r_range = 3
        angular = np.mean(polar[r_start:r_end, :], axis=0)
        
        fft = np.fft.fft(angular)
        dc = np.abs(fft[0]) / n_theta
        cos2_amp = 2 * np.abs(fft[2]) / n_theta
        
        # β = 4 * c2 / (3 * c0 - c2)
        beta = 4.0 * cos2_signed / (3.0 * dc - cos2_signed + EPSILON)
```

**物理性评价**：
- ✅ 简洁的FFT分析
- ⚠️ 没有角向反卷积
- ⚠️ 依赖Phase 4前向拟合来精修β

---

## Phase 3/4: 前向模型拟合

### V1: 完整前向模型
```python
def _forward_model_loss(self, params_flat, image_target, ...):
    # 1. 构建3D分布
    for r0, sig, amp, beta in params:
        radial = amp * np.exp(-((self.R - r0)**2) / (2 * sig**2))
        angular = 1 + beta * self.P2_GRID
        img_3d_model += radial * angular
    
    # 2. Abel前向投影
    proj_model = abel.Transform(img_3d_model, method='hansenlaw', direction='forward')
    
    # 3. 多种残差项
    # - 2D像素残差
    # - 1D径向profile残差
    # - CDF形状约束
    # - 频谱约束（打破amp/sigma ambiguity）
    # - 显式峰形约束（FWHM, peak height）
    # - Beta先验约束（权重1000）
```

**物理性评价**：
- ✅ 完整的前向卷积链
- ✅ 多种约束项防止过拟合
- ⚠️ 使用高斯近似（不是真实Abel投影形状）
- ⚠️ 计算量大

### V2: 简化前向模型（默认跳过）
```python
def _forward_model_polar(self, params, n_peaks):
    # 1. 构建3D分布
    # 2. Abel前向投影
    # 3. PSF卷积
    # 4. 像素化效应
    # 5. xy → rθ 转换
    # 6. 插值展宽

# 默认跳过Phase 3
def solve(self, image_2d, skip_phase3=True):
    if skip_phase3:
        print("Phase 3: Skipped (using Phase 1/2 estimates)")
```

**物理性评价**：
- ✅ 完整的展宽模型
- ✅ 默认跳过（Phase 1/2已经足够准确）
- ⚠️ 同样使用高斯近似

### V3.5: Poisson MLE + 数值核
```python
class ForwardFitter:
    def _numerical_abel_with_beta(self, r, sigma, amp, beta):
        """
        使用数值投影核的Abel投影（V4核心改进）
        
        物理正确性：
        - 真实Abel投影是非对称的（内侧陡峭，外侧平缓）
        - 数值核通过PyAbel forward transform预计算
        - 消除了高斯近似导致的系统性偏差
        """
        kernel = get_numerical_kernel()
        model_2d = kernel.get_template_2d(r, sigma, beta, R, cos_phi)
        return amp * model_2d
    
    def fit_poisson_mle(self, polar_data, seeds, sigma_bg, ...):
        """
        V3.5核心：泊松最大似然估计
        
        Cash statistic: C = 2 × Σ [M - D × ln(M)]
        
        优势：
        1. 正确处理低计数统计
        2. 不会产生WLS的下偏估计
        3. 对噪声大的像素不会过度惩罚
        """
        # 使用泊松残差
        # Deviance residual: sign(D-M) × sqrt(2 × |D × ln(D/M) - (D-M)|)
```

**物理性评价**：
- ✅ 数值投影核（物理正确的非对称形状）
- ✅ Poisson MLE（正确的统计模型）
- ✅ 支持背景偏置模型（不需要预先减背景）
- ✅ 宽边界允许全局优化
- ⚠️ 实际测试中Phase 2结果更准确（`skip_forward_fit=True`）

---

## BR（分支比）计算

### V1: 简单振幅比
```python
# 直接用振幅比
total_amp = sum(p['amp'] for p in params)
for p in params:
    p['branching_ratio'] = p['amp'] / total_amp
```

**物理性评价**：
- ❌ 物理上不正确（没有考虑σ和r的影响）
- ❌ 不同σ的峰会有系统偏差

### V2: 3D高斯积分公式
```python
# N = 4π × A₃D × σ × √(2π) × (r² + σ²)
integrated_intensities = []
for p in params:
    sigma_for_br = p.get('sigma_measured', p['sigma'])
    r = p['r']
    integrated_intensities.append(p['amp'] * sigma_for_br * (r**2 + sigma_for_br**2))

total_intensity = sum(integrated_intensities)
for i, p in enumerate(params):
    p['branching_ratio'] = integrated_intensities[i] / total_intensity
```

**物理性评价**：
- ✅ 考虑了σ和r的影响
- ✅ 使用测量的σ（包含系统展宽）
- ⚠️ 公式假设高斯分布

### V3.5: Yield-Based直接拟合
```python
def fit_poisson_mle(self, polar_data, seeds, ...):
    """
    参数向量包含 [r1, σ1, Y1, β1, ...] 其中 Y 是产率（BR）
    使用归一化模板，Y 直接代表该峰的分支比
    
    BR归一化约束：Σ Y_k = 1
    """
    # 从Phase 2 seeds估计初始BR
    for seed in seeds:
        intensity = amp * sigma * r**2
        raw_intensities.append(intensity)
    
    # 归一化约束
    yields = peak_params[:, 2]
    yield_sum = np.sum(yields)
    normalization_penalty = 10.0 * (yield_sum - 1.0)
```

**物理性评价**：
- ✅ 直接拟合BR（不是事后计算）
- ✅ 使用归一化模板
- ✅ β和BR在数学上彻底解耦
- ✅ 强制归一化约束

---

## 物理正确性总结

| 方面 | V1 | V2 | V3.5 |
|------|-----|-----|------|
| **Abel投影形状** | 高斯近似 ❌ | 高斯近似 ❌ | 数值核 ✅ |
| **系统展宽模型** | 无 ❌ | 完整 ✅ | 完整 ✅ |
| **统计模型** | WLS | WLS | Poisson MLE ✅ |
| **BR计算** | 振幅比 ❌ | 3D积分 ✅ | 直接拟合 ✅ |
| **中心精修** | 无 | 无 | 互相关 ✅ |
| **椭圆校正** | 无 | 无 | 算符融合 ✅ |
| **角向反卷积** | 无 | 有 ✅ | 无 |

---

## 实际性能对比（高计数 ≥5e5）

| 指标 | V1 | V2 | V3.5 |
|------|-----|-----|------|
| **r误差(%)** | ~0.5 | ~0.1 ✅ | ~0.3 |
| **σ误差(%)** | ~15 | ~7 | ~6 ✅ |
| **β误差(%)** | ~3 | ~1 ✅ | ~2 |
| **BR误差(%)** | ~10 | ~5 | ~2 ✅ |
| **检测率** | 100% | 100% | 100% |

---

## 结论

1. **V1**: 最简单的实现，但物理模型不完整（没有系统展宽，BR计算错误）

2. **V2**: 物理模型最完整（明确的展宽链），角向反卷积是独特优势，默认跳过Phase 3是明智选择

3. **V3.5**: 最先进的架构（数值核、Poisson MLE、模块化），但实际测试中Phase 2结果比Phase 4更准确

**推荐**：
- 如果追求**β准确性**：使用V2
- 如果追求**BR准确性**：使用V3.5 (`skip_forward_fit=True`)
- 如果追求**代码简洁**：使用V2 (`skip_phase3=True`)
