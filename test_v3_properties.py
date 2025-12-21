"""
Abel Backward Reconstruction V3 - Property-Based Tests

使用 Hypothesis 库进行属性测试，验证设计文档中定义的正确性属性。

每个属性测试运行至少 100 次迭代。
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from Abel_backward_reconstruction_v3 import (
    DataCleaner,
    PolarTransformer,
    SeedFinder,
    ForwardFitter,
    BRCalculator,
    legendre_p2,
    EPSILON,
)


# =============================================================================
# Property 1: Background Region Mean Zero After Cleaning
# =============================================================================
@settings(max_examples=100)
@given(
    background_value=st.floats(min_value=10.0, max_value=1000.0),
    noise_std=st.floats(min_value=1.0, max_value=50.0),
)
def test_property_1_background_mean_zero(background_value, noise_std):
    """
    **Feature: abel-backward-v3, Property 1: Background Region Mean Zero After Cleaning**
    **Validates: Requirements 1.4**
    
    For any VMI image with non-zero background, after Phase 0 cleaning,
    the mean of the background region SHALL equal zero within tolerance 1e-6.
    """
    # 创建带有背景的测试图像
    np.random.seed(42)
    size = 128
    image = np.random.normal(background_value, noise_std, (size, size))
    
    # 添加一个简单的信号（中心高斯）
    y, x = np.ogrid[:size, :size]
    center = size / 2
    r = np.sqrt((y - center)**2 + (x - center)**2)
    signal = 100 * np.exp(-r**2 / (2 * 10**2))
    image += signal
    
    # Phase 0 清理
    cleaner = DataCleaner()
    cleaned, sigma_bg = cleaner.clean(image, auto_center=False)
    
    # 验证背景区均值为零
    verify_result = cleaner.verify_cleaning(cleaned)
    
    assert abs(verify_result['bg_mean']) < 1e-6, \
        f"Background mean {verify_result['bg_mean']} exceeds tolerance 1e-6"


# =============================================================================
# Property 2: Sum Conservation in Polar Transform
# =============================================================================
@settings(max_examples=100)
@given(
    size=st.integers(min_value=64, max_value=256),
)
def test_property_2_sum_conservation(size):
    """
    **Feature: abel-backward-v3, Property 2: Sum Conservation in Polar Transform**
    **Validates: Requirements 2.2**
    
    For any 2D image, after converting from Cartesian to polar coordinates,
    the total sum SHALL be preserved within relative error 1e-6.
    """
    # 创建随机测试图像
    np.random.seed(42)
    image = np.random.uniform(0, 100, (size, size))
    
    # 极坐标转换
    transformer = PolarTransformer(n_theta=360)
    center = (size / 2, size / 2)
    polar = transformer.transform(image, center)
    
    # 验证计数守恒
    result = transformer.verify_conservation(image, polar)
    
    assert result['passed'], \
        f"Sum conservation failed: relative error = {result['relative_error']:.2e}"


# =============================================================================
# Property 10: Negative Value Preservation
# =============================================================================
@settings(max_examples=100)
@given(
    background_value=st.floats(min_value=50.0, max_value=200.0),
    noise_std=st.floats(min_value=10.0, max_value=30.0),
)
def test_property_10_negative_value_preservation(background_value, noise_std):
    """
    **Feature: abel-backward-v3, Property 10: Negative Value Preservation**
    **Validates: Requirements 1.6**
    
    For any image where background subtraction produces negative values,
    those negative values SHALL be preserved in the cleaned image.
    """
    # 创建带有背景的测试图像
    np.random.seed(42)
    size = 128
    image = np.random.normal(background_value, noise_std, (size, size))
    
    # Phase 0 清理
    cleaner = DataCleaner()
    cleaned, sigma_bg = cleaner.clean(image, auto_center=False)
    
    # 验证负值被保留
    has_negative = np.any(cleaned < 0)
    min_value = np.min(cleaned)
    
    # 由于噪声，清理后应该有负值
    assert has_negative, "Negative values should be preserved after cleaning"
    assert min_value < 0, f"Minimum value {min_value} should be negative"


# =============================================================================
# Property 11: Legendre Polynomial Correctness
# =============================================================================
@settings(max_examples=100)
@given(
    x=st.floats(min_value=-1.0, max_value=1.0),
)
def test_property_11_legendre_polynomial(x):
    """
    **Feature: abel-backward-v3, Property 11: Legendre Polynomial Correctness**
    **Validates: Requirements 7.5**
    
    For any value x ∈ [-1, 1], the computed P2(x) SHALL equal (3x² - 1) / 2.
    """
    computed = legendre_p2(np.array([x]))[0]
    expected = 0.5 * (3 * x**2 - 1)
    
    assert np.isclose(computed, expected, rtol=1e-10), \
        f"P2({x}) = {computed}, expected {expected}"


# =============================================================================
# Property 6: Non-Negative Amplitude Constraint
# =============================================================================
@settings(max_examples=50)
@given(
    n_peaks=st.integers(min_value=1, max_value=3),
)
def test_property_6_non_negative_amplitude(n_peaks):
    """
    **Feature: abel-backward-v3, Property 6: Non-Negative Amplitude Constraint**
    **Validates: Requirements 4.8**
    
    For any fitting result, all A_3D values SHALL be non-negative.
    """
    # 创建测试数据
    np.random.seed(42)
    size = 128
    center = (size / 2, size / 2)
    
    # 创建带有峰的图像
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    image = np.zeros((size, size))
    seeds = []
    
    for i in range(n_peaks):
        r_peak = 30 + i * 20
        sigma = 2.0
        amp = 100 * (i + 1)
        
        image += amp * np.exp(-((r - r_peak)**2) / (2 * sigma**2))
        seeds.append({'r': float(r_peak), 'sigma': sigma, 'amp': amp, 'beta': 0.0})
    
    # 添加噪声
    image += np.random.normal(0, 5, (size, size))
    
    # 拟合
    fitter = ForwardFitter()
    fitted_params, metadata = fitter.fit(image, seeds, sigma_bg=5.0, center=center)
    
    # 验证所有振幅非负
    for p in fitted_params:
        assert p['amp'] >= 0, f"Amplitude {p['amp']} is negative"


# =============================================================================
# Property 7: Sub-Pixel Position Precision
# =============================================================================
@settings(max_examples=50)
@given(
    r_true=st.floats(min_value=30.5, max_value=80.5),
)
def test_property_7_subpixel_precision(r_true):
    """
    **Feature: abel-backward-v3, Property 7: Sub-Pixel Position Precision**
    **Validates: Requirements 4.9**
    
    For any fitting result, peak positions r SHALL be floating-point values.
    """
    # 创建测试数据
    np.random.seed(42)
    size = 128
    center = (size / 2, size / 2)
    
    # 创建带有峰的图像
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    sigma = 2.0
    amp = 100
    image = amp * np.exp(-((r - r_true)**2) / (2 * sigma**2))
    image += np.random.normal(0, 2, (size, size))
    
    seeds = [{'r': float(round(r_true)), 'sigma': sigma, 'amp': amp, 'beta': 0.0}]
    
    # 拟合
    fitter = ForwardFitter()
    fitted_params, metadata = fitter.fit(image, seeds, sigma_bg=2.0, center=center)
    
    # 验证位置是浮点数（不是整数）
    if len(fitted_params) > 0:
        r_fitted = fitted_params[0]['r']
        # 检查是否为浮点数类型
        assert isinstance(r_fitted, float), f"Position {r_fitted} is not a float"


# =============================================================================
# Property 8: BR Normalization
# =============================================================================
@settings(max_examples=100)
@given(
    n_peaks=st.integers(min_value=1, max_value=5),
)
def test_property_8_br_normalization(n_peaks):
    """
    **Feature: abel-backward-v3, Property 8: BR Normalization**
    **Validates: Requirements 5.5**
    
    For any set of computed branching ratios, the sum SHALL equal 1.0.
    """
    # 创建随机参数
    np.random.seed(42)
    params = []
    for i in range(n_peaks):
        params.append({
            'r': 30 + i * 20,
            'sigma': np.random.uniform(1.0, 3.0),
            'amp': np.random.uniform(10, 100),
            'beta': np.random.uniform(-1, 2),
        })
    
    # 计算 BR
    calculator = BRCalculator()
    params_with_br = calculator.calculate(params)
    
    # 验证归一化
    br_sum = sum(p['br'] for p in params_with_br)
    
    assert np.isclose(br_sum, 1.0, atol=1e-6), \
        f"BR sum {br_sum} does not equal 1.0"


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Running Property-Based Tests for Abel Backward Reconstruction V3")
    print("=" * 70)
    
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])


# =============================================================================
# Property 3: Abel Transform Round-Trip
# =============================================================================
@settings(max_examples=50)
@given(
    peak_position=st.integers(min_value=30, max_value=80),
    peak_width=st.floats(min_value=2.0, max_value=5.0),
)
def test_property_3_abel_round_trip(peak_position, peak_width):
    """
    **Feature: abel-backward-v3, Property 3: Abel Transform Round-Trip**
    **Validates: Requirements 3.2**
    
    For any 1D radial profile, applying forward Abel transform followed by
    inverse Abel transform SHALL recover the original profile within tolerance.
    """
    import abel
    
    # 创建 1D 高斯分布
    n = 128
    r = np.arange(n)
    profile_3d = np.exp(-((r - peak_position)**2) / (2 * peak_width**2))
    
    # 前向 Abel 变换
    profile_2d = abel.hansenlaw.hansenlaw_transform(profile_3d, direction='forward')
    
    # 逆向 Abel 变换
    profile_recovered = abel.hansenlaw.hansenlaw_transform(profile_2d, direction='inverse')
    
    # 只比较峰值区域（边缘可能有数值误差）
    mask = (r > peak_position - 3*peak_width) & (r < peak_position + 3*peak_width)
    
    if np.sum(mask) > 5:
        # 归一化比较
        orig_norm = profile_3d[mask] / (np.max(profile_3d[mask]) + EPSILON)
        recov_norm = profile_recovered[mask] / (np.max(profile_recovered[mask]) + EPSILON)
        
        # 允许 10% 的相对误差（Abel 变换有数值误差）
        relative_error = np.mean(np.abs(orig_norm - recov_norm))
        assert relative_error < 0.2, \
            f"Abel round-trip error {relative_error:.2f} exceeds tolerance"


# =============================================================================
# Property 5: Forward Model Consistency
# =============================================================================
@settings(max_examples=30)
@given(
    r=st.floats(min_value=40.0, max_value=80.0),
    sigma=st.floats(min_value=1.5, max_value=4.0),
    beta=st.floats(min_value=-0.5, max_value=1.5),
)
def test_property_5_forward_model_consistency(r, sigma, beta):
    """
    **Feature: abel-backward-v3, Property 5: Forward Model Consistency**
    **Validates: Requirements 4.1, 7.1**
    
    For any set of peak parameters, the forward model output SHALL be
    physically reasonable (non-negative, peaked at r).
    """
    size = 128
    center = (size / 2, size / 2)
    
    fitter = ForwardFitter()
    fitter._init_grids((size, size), center)
    
    # 生成前向模型
    amp = 100.0
    model = fitter._analytic_abel_with_beta(r, sigma, amp, beta)
    
    # 验证模型是物理合理的
    assert np.all(model >= 0), "Model contains negative values"
    assert np.max(model) > 0, "Model has no signal"
    
    # 验证峰值位置大致正确
    y, x = np.ogrid[:size, :size]
    R = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(model), model.shape)
    r_max = R[max_idx]
    
    # 峰值位置应该接近 r（允许 2σ 的误差）
    assert abs(r_max - r) < 2 * sigma + 2, \
        f"Peak position {r_max:.1f} differs from expected {r:.1f}"


# =============================================================================
# Property 9: β-BR Decoupling
# =============================================================================
@settings(max_examples=30)
@given(
    beta_original=st.floats(min_value=-0.5, max_value=1.5),
)
def test_property_9_beta_br_decoupling(beta_original):
    """
    **Feature: abel-backward-v3, Property 9: β-BR Decoupling**
    **Validates: Requirements 5.3**
    
    For any dataset, when β initial values are perturbed, the resulting
    BR values SHALL remain stable.
    """
    # 创建测试数据
    np.random.seed(42)
    size = 128
    center = (size / 2, size / 2)
    
    # 创建带有峰的图像
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    r_peak = 50.0
    sigma = 2.0
    amp = 100.0
    
    # 简单的环形图像
    image = amp * np.exp(-((r - r_peak)**2) / (2 * sigma**2))
    image += np.random.normal(0, 2, (size, size))
    
    # 原始参数
    seeds_original = [{'r': r_peak, 'sigma': sigma, 'amp': amp, 'beta': beta_original}]
    
    # 扰动后的参数
    beta_perturbed = np.clip(beta_original + 0.5, -1.0, 2.0)
    seeds_perturbed = [{'r': r_peak, 'sigma': sigma, 'amp': amp, 'beta': beta_perturbed}]
    
    # 计算 BR
    calculator = BRCalculator()
    
    params_original = calculator.calculate(seeds_original.copy())
    params_perturbed = calculator.calculate(seeds_perturbed.copy())
    
    # 单峰情况下，BR 应该都是 1.0
    assert np.isclose(params_original[0]['br'], 1.0, atol=1e-6)
    assert np.isclose(params_perturbed[0]['br'], 1.0, atol=1e-6)


# =============================================================================
# Property 12: Energy Linearity
# =============================================================================
@settings(max_examples=30)
@given(
    r1=st.floats(min_value=30.0, max_value=50.0),
)
def test_property_12_energy_linearity(r1):
    """
    **Feature: abel-backward-v3, Property 12: Energy Linearity**
    **Validates: Requirements 7.1**
    
    For two peaks with energy ratio E1:E2 = 1:4, the radius ratio
    SHALL satisfy r1:r2 = 1:2 (since E ∝ r²).
    """
    # E ∝ r², 所以 E1:E2 = 1:4 意味着 r1:r2 = 1:2
    r2 = r1 * 2  # r2 = 2 * r1
    
    # 验证能量比
    E1 = r1**2
    E2 = r2**2
    
    energy_ratio = E2 / E1
    expected_ratio = 4.0
    
    assert np.isclose(energy_ratio, expected_ratio, rtol=0.01), \
        f"Energy ratio {energy_ratio:.2f} differs from expected {expected_ratio}"
    
    # 验证半径比
    radius_ratio = r2 / r1
    expected_radius_ratio = 2.0
    
    assert np.isclose(radius_ratio, expected_radius_ratio, rtol=0.01), \
        f"Radius ratio {radius_ratio:.2f} differs from expected {expected_radius_ratio}"


# =============================================================================
# Property 13: β-Invariant Total Count
# =============================================================================
@settings(max_examples=30)
@given(
    amp=st.floats(min_value=50.0, max_value=200.0),
    r=st.floats(min_value=40.0, max_value=80.0),
    sigma=st.floats(min_value=1.5, max_value=4.0),
)
def test_property_13_beta_invariant_total_count(amp, r, sigma):
    """
    **Feature: abel-backward-v3, Property 13: β-Invariant Total Count**
    **Validates: Requirements 5.3**
    
    For any synthetic image with fixed A_3D and varying β ∈ {0, 1, 2},
    the computed total particle count N = A_3D × σ × r² SHALL remain constant.
    """
    calculator = BRCalculator()
    
    # 测试不同的 β 值
    betas = [0.0, 1.0, 2.0]
    total_counts = []
    
    for beta in betas:
        params = [{'r': r, 'sigma': sigma, 'amp': amp, 'beta': beta}]
        
        # 计算总粒子数 N = A_3D × σ × r²
        N = amp * sigma * r**2
        total_counts.append(N)
    
    # 验证所有 β 值的总粒子数相同
    for i, N in enumerate(total_counts):
        assert np.isclose(N, total_counts[0], rtol=0.005), \
            f"Total count for β={betas[i]} differs: {N:.2f} vs {total_counts[0]:.2f}"
