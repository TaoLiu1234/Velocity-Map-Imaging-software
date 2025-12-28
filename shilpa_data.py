"""
VMI径向分布重建工具
===================
从XY坐标数据重建牛顿球的r0和sigma参数

物理模型：
- 牛顿球：以r0为中心，sigma为高斯厚度的球壳
- Abel投影：3D球壳投影到2D探测器
- 噪声模型：Var(r) = μ(r) + σ²_dc (泊松 + DC噪声)

方法：
1. 密度修正：ρ(R) = H(R) / (2πR)，消除面积增长效应
2. Mexican Hat模板匹配：零均值模板，对背景免疫
3. 多尺度验证：不同dr下峰位置应该稳定
4. 分区域高斯约束拟合：确保各区域残差都符合N(0,1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import correlate, find_peaks
from scipy.optimize import minimize


# ============================================================
# 核心函数
# ============================================================

def mexican_hat(x, sigma):
    """Mexican Hat (LoG) 零均值模板，对常数/线性背景免疫"""
    norm = 2 / (np.sqrt(3 * sigma) * np.pi**0.25)
    t = (x / sigma)**2
    return norm * (1 - t) * np.exp(-t / 2)


def abel_projection(R, r0, sigma):
    """高斯球壳的Abel投影"""
    z = np.linspace(0, 5 * sigma + 10, 200)
    dz = z[1] - z[0]
    R_mat = R[:, np.newaxis]
    Z_mat = z[np.newaxis, :]
    r_val = np.sqrt(Z_mat**2 + R_mat**2)
    gauss = np.exp(-(r_val - r0)**2 / (2 * sigma**2))
    return 2 * np.trapz(gauss, dx=dz, axis=1)


def detect_peaks_mexican_hat(R, rho, sigma_template=0.5, min_r=5):
    """
    用Mexican Hat模板检测峰位置
    
    Parameters:
    -----------
    R : array, 半径数组
    rho : array, 密度分布 ρ(R) = H(R)/(2πR)
    sigma_template : float, 模板宽度
    min_r : float, 忽略r < min_r的区域（边界效应）
    
    Returns:
    --------
    peak_positions : list, 检测到的峰位置（按prominence排序）
    """
    dr = R[1] - R[0]
    template_r = np.linspace(-5*sigma_template, 5*sigma_template, max(int(10*sigma_template/dr), 10))
    template = mexican_hat(template_r, sigma_template)
    
    response = correlate(rho, template, mode='same')
    
    # 只在r > min_r区域找峰
    valid_mask = R > min_r
    response_masked = response.copy()
    response_masked[~valid_mask] = 0
    
    peaks_idx, props = find_peaks(response_masked, 
                                   prominence=np.max(np.abs(response_masked)) * 0.05,
                                   distance=max(3, int(1.0/dr)))
    
    if len(peaks_idx) == 0:
        return []
    
    # 按prominence排序
    order = np.argsort(props['prominences'])[::-1]
    return [R[peaks_idx[i]] for i in order]


def multiscale_peak_detection(r_data, r_max, dr_values=[0.1, 0.15, 0.2, 0.25, 0.3], n_peaks=2):
    """
    多尺度峰检测：在不同dr下检测峰位置，取平均值
    
    Returns:
    --------
    peak_means : list, 各峰的平均位置
    peak_stds : list, 各峰位置的标准差
    """
    all_peaks = [[] for _ in range(n_peaks)]
    
    for dr in dr_values:
        n_bins = int(r_max / dr)
        counts, bin_edges = np.histogram(r_data, bins=n_bins, range=(0, r_max))
        R = (bin_edges[:-1] + bin_edges[1:]) / 2
        rho = counts / (2 * np.pi * R + 1e-6)
        
        peaks = detect_peaks_mexican_hat(R, rho)
        
        if len(peaks) >= n_peaks:
            sorted_peaks = sorted(peaks[:n_peaks])
            for i in range(n_peaks):
                all_peaks[i].append(sorted_peaks[i])
    
    peak_means = [np.mean(p) if len(p) > 0 else np.nan for p in all_peaks]
    peak_stds = [np.std(p) if len(p) > 1 else np.nan for p in all_peaks]
    
    return peak_means, peak_stds


# ============================================================
# 尺度空间分析 - r 和 σ 解耦
# ============================================================

def scale_space_analysis(R, rho, sigma_range=(0.3, 3.0), n_scales=30, min_r=5):
    """
    尺度空间最大化法：通过模板库解耦 r 和 σ
    
    原理：
    - 构建不同宽度的Mexican Hat模板库
    - 当模板宽度与物理信号宽度匹配时，响应达到最大
    - r 由响应最强的位置确定
    - σ 由响应最强的尺度确定
    
    Parameters:
    -----------
    R : array, 半径数组
    rho : array, 密度分布
    sigma_range : tuple, 模板宽度搜索范围
    n_scales : int, 尺度数量
    min_r : float, 忽略 r < min_r 的区域
    
    Returns:
    --------
    peaks : list of dict, 每个峰的 {r0, sigma, response}
    scale_space : 2D array, 尺度空间响应图 (n_scales, len(R))
    sigmas : array, 尺度数组
    """
    dr = R[1] - R[0]
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_scales)
    
    # 构建尺度空间
    scale_space = np.zeros((n_scales, len(R)))
    
    for i, sigma in enumerate(sigmas):
        # 构建模板
        template_width = int(10 * sigma / dr)
        if template_width < 5:
            template_width = 5
        template_r = np.linspace(-5*sigma, 5*sigma, template_width)
        template = mexican_hat(template_r, sigma)
        
        # 卷积响应
        response = correlate(rho, template, mode='same')
        scale_space[i, :] = response
    
    # 在 r > min_r 区域找峰
    valid_mask = R > min_r
    scale_space_masked = scale_space.copy()
    scale_space_masked[:, ~valid_mask] = 0
    
    # 对每个尺度，找局部最大值
    from scipy.ndimage import maximum_filter
    
    # 使用局部最大值检测而不是全局最大值
    # 这样可以更好地分离多个峰
    peaks = []
    
    # 先在最小尺度找峰位置（位置最准确）
    small_scale_idx = 0  # 最小尺度
    response_small = scale_space[small_scale_idx, :]
    response_small_masked = response_small.copy()
    response_small_masked[~valid_mask] = 0
    
    peak_indices, props = find_peaks(response_small_masked,
                                      prominence=np.max(response_small_masked) * 0.05,
                                      distance=max(3, int(1.0/dr)))
    
    # 对每个峰，在尺度空间中找最佳尺度
    for peak_idx in peak_indices:
        r0 = R[peak_idx]
        
        # 在该位置附近（±2个bin）找尺度空间的最大响应
        r_window = 3
        r_min_idx = max(0, peak_idx - r_window)
        r_max_idx = min(len(R), peak_idx + r_window)
        
        local_response = scale_space[:, r_min_idx:r_max_idx]
        max_scale_idx = np.unravel_index(np.argmax(local_response), local_response.shape)[0]
        
        sigma_best = sigmas[max_scale_idx]
        max_response = np.max(local_response)
        
        peaks.append({
            'r0': r0,
            'sigma': sigma_best,
            'response': max_response
        })
    
    # 按响应强度排序
    peaks = sorted(peaks, key=lambda x: -x['response'])
    
    return peaks, scale_space, sigmas


def delta_r_method(R, rho, sigma_template=0.5, min_r=5):
    """
    ΔR 方法：利用边缘位置和峰值位置的差异估计 σ
    
    原理：
    - Mexican Hat 响应最大位置 ≈ 信号的最陡上升点 (R_edge)
    - 原始密度曲线的峰值位置 (R_peak)
    - 对于高斯环的Abel投影：ΔR = R_peak - R_edge ∝ σ
    
    Parameters:
    -----------
    R : array, 半径数组
    rho : array, 密度分布
    sigma_template : float, 模板宽度
    min_r : float, 忽略 r < min_r 的区域
    
    Returns:
    --------
    peaks : list of dict, 每个峰的 {r_edge, r_peak, delta_r, sigma_est}
    """
    dr = R[1] - R[0]
    
    # 1. 用Mexican Hat找边缘位置
    template_r = np.linspace(-5*sigma_template, 5*sigma_template, max(int(10*sigma_template/dr), 10))
    template = mexican_hat(template_r, sigma_template)
    response = correlate(rho, template, mode='same')
    
    valid_mask = R > min_r
    response_masked = response.copy()
    response_masked[~valid_mask] = 0
    
    # 找边缘峰
    edge_peaks_idx, edge_props = find_peaks(response_masked, 
                                             prominence=np.max(np.abs(response_masked)) * 0.05,
                                             distance=max(3, int(1.0/dr)))
    
    if len(edge_peaks_idx) == 0:
        return []
    
    # 2. 平滑原始密度曲线，找峰值位置
    from scipy.ndimage import gaussian_filter1d
    rho_smooth = gaussian_filter1d(rho, sigma=1.0/dr)
    
    peaks = []
    for edge_idx in edge_peaks_idx:
        r_edge = R[edge_idx]
        
        # 在边缘附近找密度峰值
        search_range = int(5 / dr)  # 搜索范围 ±5
        search_min = max(0, edge_idx - search_range)
        search_max = min(len(R), edge_idx + search_range)
        
        local_max_idx = search_min + np.argmax(rho_smooth[search_min:search_max])
        r_peak = R[local_max_idx]
        
        delta_r = r_peak - r_edge
        
        # 经验关系：对于Abel投影的高斯环，ΔR ≈ 0.8 * σ
        # 这个系数需要通过模拟校准
        sigma_est = abs(delta_r) / 0.8 if delta_r > 0 else abs(delta_r)
        
        peaks.append({
            'r_edge': r_edge,
            'r_peak': r_peak,
            'delta_r': delta_r,
            'sigma_est': sigma_est,
            'response': response[edge_idx]
        })
    
    # 按响应强度排序
    peaks = sorted(peaks, key=lambda x: -x['response'])
    
    return peaks


def response_width_method(R, rho, sigma_template=0.5, min_r=5):
    """
    响应宽度反推法：通过响应曲线的半高宽估计 σ
    
    原理：
    - 物理信号宽度 σ，模板宽度 σ_temp
    - 响应宽度 W_res² ≈ σ² + σ_temp²
    - 因此 σ = √(W_res² - σ_temp²)
    
    Parameters:
    -----------
    R : array, 半径数组
    rho : array, 密度分布
    sigma_template : float, 模板宽度
    min_r : float, 忽略 r < min_r 的区域
    
    Returns:
    --------
    peaks : list of dict, 每个峰的 {r0, W_res, sigma_est}
    """
    dr = R[1] - R[0]
    
    # 构建模板
    template_r = np.linspace(-5*sigma_template, 5*sigma_template, max(int(10*sigma_template/dr), 10))
    template = mexican_hat(template_r, sigma_template)
    
    # 卷积响应
    response = correlate(rho, template, mode='same')
    
    valid_mask = R > min_r
    response_masked = response.copy()
    response_masked[~valid_mask] = -np.inf
    
    # 找峰
    peaks_idx, props = find_peaks(response_masked, 
                                   prominence=np.max(response_masked[valid_mask]) * 0.05,
                                   distance=max(3, int(1.0/dr)),
                                   width=1)
    
    if len(peaks_idx) == 0:
        return []
    
    peaks = []
    for i, peak_idx in enumerate(peaks_idx):
        r0 = R[peak_idx]
        peak_val = response[peak_idx]
        
        # 测量半高宽
        half_max = peak_val / 2
        
        # 向左找半高点
        left_idx = peak_idx
        while left_idx > 0 and response[left_idx] > half_max:
            left_idx -= 1
        
        # 向右找半高点
        right_idx = peak_idx
        while right_idx < len(R) - 1 and response[right_idx] > half_max:
            right_idx += 1
        
        W_res = R[right_idx] - R[left_idx]
        
        # 反推 σ
        # Mexican Hat 的有效宽度约为 2.5 * sigma_template
        sigma_temp_eff = sigma_template * 2.5
        sigma_sq = W_res**2 - sigma_temp_eff**2
        sigma_est = np.sqrt(max(sigma_sq, 0.01))
        
        peaks.append({
            'r0': r0,
            'W_res': W_res,
            'sigma_est': sigma_est,
            'response': peak_val
        })
    
    # 按响应强度排序
    peaks = sorted(peaks, key=lambda x: -x['response'])
    
    return peaks


def estimate_r_sigma_decoupled(R, rho, n_peaks=2, verbose=True, dr=None, dld_resolution=0.0):
    """
    综合使用多种方法估计 r 和 σ
    
    改进策略：
    1. Mexican Hat 定位 r（导数域，对背景免疫）
    2. 导数域模板匹配估计 σ（比强度域更鲁棒）
    3. ΔR 方法：r_peak - r_steep ∝ σ
    4. 逐层剥离：处理重叠峰
    5. Binning 展宽修正：σ²_true = σ²_obs - dr²/12 - dld²/12
    
    Parameters:
    -----------
    R : array, 半径数组
    rho : array, 密度分布
    n_peaks : int, 峰数量
    verbose : bool, 是否打印详细信息
    dr : float, 径向 bin 宽度（用于展宽修正）
    dld_resolution : float, DLD 量化分辨率（用于展宽修正）
    
    Returns:
    --------
    estimates : list of dict, 每个峰的 {r0, sigma, sigma_raw}
    """
    if verbose:
        print("\n========== r-σ 解耦分析 (改进版) ==========")
    
    # 计算 binning 展宽
    if dr is None:
        dr = R[1] - R[0] if len(R) > 1 else 0.1
    
    sigma_bin_sq = dr**2 / 12  # 径向 binning 展宽
    sigma_dld_sq = dld_resolution**2 / 12  # DLD 量化展宽
    sigma_broadening_sq = sigma_bin_sq + sigma_dld_sq
    sigma_broadening = np.sqrt(sigma_broadening_sq)
    
    if verbose:
        print(f"\nBinning 展宽修正:")
        print(f"  dr = {dr:.3f} mm → σ_bin = {np.sqrt(sigma_bin_sq):.3f} mm")
        print(f"  dld_res = {dld_resolution:.3f} mm → σ_dld = {np.sqrt(sigma_dld_sq):.3f} mm")
        print(f"  总展宽 σ_broadening = {sigma_broadening:.3f} mm")
    
    # Step 1: Mexican Hat 定位 r
    peaks_mh = detect_peaks_mexican_hat(R, rho, sigma_template=0.5, min_r=10)
    peaks_mh = peaks_mh[:n_peaks]
    peaks_mh = sorted(peaks_mh)
    
    if verbose:
        print("\nStep 1 - Mexican Hat 定位:")
        for i, r0 in enumerate(peaks_mh):
            print(f"  Peak {i+1}: r0 = {r0:.2f}")
    
    # 峰间距约束
    if len(peaks_mh) > 1:
        min_separation = min(peaks_mh[i+1] - peaks_mh[i] for i in range(len(peaks_mh)-1))
        sigma_max_constraint = min_separation / 3
    else:
        sigma_max_constraint = 5.0
    
    if verbose:
        print(f"\n峰间距约束: σ_max = {sigma_max_constraint:.2f}")
    
    sigma_range = np.linspace(0.1, min(sigma_max_constraint, 3.0), 30)  # 下限从 0.3 改为 0.1
    
    # 计算导数
    from scipy.ndimage import gaussian_filter1d
    rho_smooth = gaussian_filter1d(rho, sigma=0.5/dr)
    drho_dR = np.gradient(rho_smooth, dr)
    
    estimates = []
    rho_residual = rho.copy()
    
    # 按位置从外向内处理（外侧峰的尾部影响更小）
    process_order = np.argsort(peaks_mh)[::-1]
    
    if verbose:
        print("\n处理顺序: 从外向内 (减少 Abel 尾部干扰)")
    
    for proc_idx in process_order:
        r0 = peaks_mh[proc_idx]
        
        if verbose:
            print(f"\n处理 Peak {proc_idx+1} (r0 = {r0:.2f}):")
        
        # 方法1: 导数域模板匹配
        sigma_deriv = estimate_sigma_derivative_domain(R, rho_residual, r0, sigma_range, dr, verbose)
        
        # 方法2: ΔR 方法（仅作参考，不参与最终估计）
        sigma_delta_r = estimate_sigma_delta_r(R, rho_residual, drho_dR, r0, dr, verbose)
        
        # 方法3: 强度域模板匹配
        sigma_intensity = estimate_sigma_intensity_domain(R, rho_residual, r0, sigma_range, verbose)
        
        # 方法4: Hansen-Law 逆变换后高斯拟合
        sigma_hansen = estimate_sigma_hansen_law(R, rho_residual, r0, verbose)
        
        # ============================================================
        # 综合策略（基于测试结果优化）
        # ============================================================
        # 测试发现：
        # - 强度域匹配：单峰最准（0%误差），但重叠时失效
        # - 导数域匹配：窄峰高估20%，但重叠时稳定
        # - Hansen-Law：系统性低估10%，重叠时比强度域好
        # - ΔR 方法：不可靠，仅作参考
        
        hansen_val = sigma_hansen if sigma_hansen and 0.1 < sigma_hansen < sigma_max_constraint else None
        deriv_val = sigma_deriv if sigma_deriv and 0.1 < sigma_deriv < sigma_max_constraint else None
        intens_val = sigma_intensity if sigma_intensity and 0.1 < sigma_intensity < sigma_max_constraint else None
        
        # 判断是否可能有重叠（基于峰间距）
        is_overlapping = sigma_max_constraint < 2.0  # 间距 < 6σ 认为可能重叠
        
        if intens_val is not None and not is_overlapping:
            # 单峰或间距大：优先使用强度域匹配（最准确）
            sigma_raw = intens_val
            if verbose:
                print(f"  策略: 使用强度域匹配 (间距大)")
        elif hansen_val is not None and deriv_val is not None:
            # 重叠情况：Hansen-Law 和导数域交叉验证
            # Hansen-Law 低估约10%，导数域可能高估
            if abs(hansen_val - deriv_val) < 0.5:
                # 两者一致，取平均
                sigma_raw = (hansen_val + deriv_val) / 2
                if verbose:
                    print(f"  策略: Hansen + 导数域平均 (一致)")
            else:
                # 不一致，Hansen-Law 通常更可靠（重叠时）
                # 但需要修正其低估倾向
                hansen_corrected = hansen_val * 1.1  # 修正10%低估
                sigma_raw = hansen_corrected
                if verbose:
                    print(f"  策略: Hansen (修正) (不一致)")
        elif hansen_val is not None:
            # 只有 Hansen-Law 可用
            sigma_raw = hansen_val * 1.1  # 修正低估
            if verbose:
                print(f"  策略: Hansen (修正)")
        elif deriv_val is not None:
            sigma_raw = deriv_val
            if verbose:
                print(f"  策略: 导数域匹配")
        elif intens_val is not None:
            sigma_raw = intens_val
            if verbose:
                print(f"  策略: 强度域匹配 (fallback)")
        else:
            sigma_raw = sigma_max_constraint / 2
            if verbose:
                print(f"  策略: 默认值")
        
        # 扣除 binning 展宽
        sigma_corrected_sq = sigma_raw**2 - sigma_broadening_sq
        if sigma_corrected_sq > 0:
            sigma_final = np.sqrt(sigma_corrected_sq)
        else:
            sigma_final = 0.1
            if verbose:
                print(f"  警告: σ_raw ({sigma_raw:.2f}) < σ_broadening ({sigma_broadening:.2f}), 设为最小值")
        
        if verbose:
            print(f"  σ_raw = {sigma_raw:.2f}, σ_corrected = {sigma_final:.2f}")
        
        estimates.append({
            'r0': r0,
            'sigma': sigma_final,
            'sigma_raw': sigma_raw,
            'sigma_derivative': sigma_deriv,
            'sigma_delta_r': sigma_delta_r,
            'sigma_intensity': sigma_intensity,
            'sigma_hansen': sigma_hansen,
            'process_order': proc_idx
        })
        
        # 逐层剥离
        mask_peak = (R > r0 - 3*sigma_final) & (R < r0 + 3*sigma_final)
        if np.sum(mask_peak) > 0:
            peak_model_unit = abel_projection(R, r0, sigma_final)
            peak_model_unit = peak_model_unit / (np.max(peak_model_unit) + 1e-10)
            
            rho_peak = rho_residual[mask_peak]
            model_peak = peak_model_unit[mask_peak]
            
            amp_est = np.sum(rho_peak * model_peak) / (np.sum(model_peak**2) + 1e-10)
            amp_est = max(amp_est, 0)
            
            peak_model = amp_est * peak_model_unit
            rho_residual = rho_residual - peak_model * 0.9
    
    # 按位置重新排序
    estimates = sorted(estimates, key=lambda x: x['r0'])
    
    if verbose:
        print("\n最终估计 (已扣除 binning 展宽):")
        for i, e in enumerate(estimates):
            print(f"  Peak {i+1}: r0 = {e['r0']:.2f}, σ = {e['sigma']:.2f} (raw: {e['sigma_raw']:.2f})")
    
    return estimates, {'sigma_range': sigma_range, 'sigma_broadening': sigma_broadening}


def estimate_sigma_derivative_domain(R, rho, r0, sigma_range, dr, verbose=False):
    """
    导数域模板匹配：用 dρ/dR 匹配 d(Abel)/dR
    
    优点：
    - 消除常数背景
    - 对重叠峰更鲁棒
    - σ 的灵敏度更高
    """
    from scipy.ndimage import gaussian_filter1d
    
    # 计算观测数据的导数
    rho_smooth = gaussian_filter1d(rho, sigma=0.3/dr)
    drho_obs = np.gradient(rho_smooth, dr)
    
    # 局部窗口
    r_window = 4.0
    mask = (R > r0 - r_window) & (R < r0 + r_window)
    R_local = R[mask]
    drho_local = drho_obs[mask]
    
    if len(R_local) < 10:
        return None
    
    # 归一化
    drho_norm = drho_local / (np.max(np.abs(drho_local)) + 1e-10)
    
    best_corr = -np.inf
    best_sigma = sigma_range[0]
    
    for sigma in sigma_range:
        # 生成 Abel 投影模板的导数
        template = abel_projection(R_local, r0, sigma)
        dtemplate = np.gradient(template, dr)
        
        if np.max(np.abs(dtemplate)) < 1e-10:
            continue
        
        dtemplate_norm = dtemplate / np.max(np.abs(dtemplate))
        
        # 计算相关系数
        corr = np.corrcoef(drho_norm, dtemplate_norm)[0, 1]
        
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_sigma = sigma
    
    if verbose:
        print(f"  导数域匹配: σ = {best_sigma:.2f}, 相关 = {best_corr:.3f}")
    
    return best_sigma if best_corr > 0.5 else None


def estimate_sigma_hansen_law(R, rho, r0, verbose=False, lowpass_sigma=None, regularization=0.1):
    """
    Hansen-Law 逆变换后直接估计 σ
    
    原理：
    - Abel 投影的逆变换恢复原始 3D 分布
    - 对于高斯球壳，逆变换后是高斯分布
    - 直接拟合高斯得到 σ
    
    改进：
    - 使用 Tikhonov 正则化减少数值误差
    - 只在 r0 附近做局部逆变换（减少干扰）
    - 使用半高宽估计 σ（比拟合更鲁棒）
    
    Parameters:
    -----------
    R : array, 半径数组
    rho : array, 密度分布
    r0 : float, 峰位置
    verbose : bool, 是否打印详细信息
    lowpass_sigma : float, 低通滤波宽度（None 则不滤波）
    regularization : float, 正则化参数（减少数值振荡）
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit
    
    dr = R[1] - R[0]
    
    # 只在 r0 附近做逆变换（减少其他峰的干扰）
    r_window = 6.0
    r_min_idx = max(0, np.searchsorted(R, r0 - r_window))
    r_max_idx = min(len(R), np.searchsorted(R, r0 + r_window))
    
    R_local = R[r_min_idx:r_max_idx]
    rho_local = rho[r_min_idx:r_max_idx]
    
    if len(R_local) < 10:
        if verbose:
            print(f"  Hansen-Law: 数据点不足")
        return None
    
    # 可选低通滤波
    if lowpass_sigma is not None and lowpass_sigma > 0:
        from scipy.fft import fft, ifft, fftfreq
        n_fft = len(rho_local)
        rho_fft = fft(rho_local)
        freq = fftfreq(n_fft, dr)
        filter_kernel = np.exp(-2 * (np.pi * freq * lowpass_sigma)**2)
        rho_local = np.real(ifft(rho_fft * filter_kernel))
    
    # 平滑（使用较小的平滑参数）
    smooth_sigma = max(0.2 / dr, 1)
    rho_smooth = gaussian_filter1d(rho_local, sigma=smooth_sigma)
    
    # 计算导数
    drho_dR = np.gradient(rho_smooth, dr)
    
    # Hansen-Law 逆变换（带正则化）
    # f(r) = -1/π ∫_r^∞ dρ/dR / √(R²-r²) dR
    n = len(R_local)
    f_r = np.zeros(n)
    
    for i in range(n - 2):
        r = R_local[i]
        
        # 积分从 r 到 R_max
        integrand = np.zeros(n - i)
        for j in range(i, n):
            R_j = R_local[j]
            diff_sq = R_j**2 - r**2
            # 正则化：避免奇异点
            if diff_sq > regularization**2:
                integrand[j - i] = drho_dR[j] / np.sqrt(diff_sq)
            else:
                # 在奇异点附近使用正则化
                integrand[j - i] = drho_dR[j] / np.sqrt(diff_sq + regularization**2)
        
        f_r[i] = -np.trapz(integrand, R_local[i:]) / np.pi
    
    # 找到 f_r 在 r0 附近的峰
    r0_local = r0 - R_local[0]  # 相对于局部坐标
    peak_mask = (R_local > r0 - 3) & (R_local < r0 + 3)
    
    if np.sum(peak_mask) < 5:
        if verbose:
            print(f"  Hansen-Law: 峰区域数据点不足")
        return None
    
    R_peak = R_local[peak_mask]
    f_peak = f_r[peak_mask]
    
    # 找峰值
    peak_idx = np.argmax(f_peak)
    peak_val = f_peak[peak_idx]
    r_peak_pos = R_peak[peak_idx]
    
    if peak_val <= 0:
        if verbose:
            print(f"  Hansen-Law: 峰值为负或零")
        return None
    
    # 方法1：半高宽估计 σ
    half_max = peak_val / 2
    
    # 向左找半高点
    left_idx = peak_idx
    while left_idx > 0 and f_peak[left_idx] > half_max:
        left_idx -= 1
    
    # 向右找半高点
    right_idx = peak_idx
    while right_idx < len(f_peak) - 1 and f_peak[right_idx] > half_max:
        right_idx += 1
    
    # 线性插值找精确的半高点
    if left_idx > 0 and left_idx < len(f_peak) - 1:
        if f_peak[left_idx] < half_max < f_peak[left_idx + 1]:
            frac = (half_max - f_peak[left_idx]) / (f_peak[left_idx + 1] - f_peak[left_idx] + 1e-10)
            r_left = R_peak[left_idx] + frac * (R_peak[left_idx + 1] - R_peak[left_idx])
        else:
            r_left = R_peak[left_idx]
    else:
        r_left = R_peak[max(0, left_idx)]
    
    if right_idx > 0 and right_idx < len(f_peak):
        if f_peak[right_idx] < half_max < f_peak[right_idx - 1]:
            frac = (half_max - f_peak[right_idx - 1]) / (f_peak[right_idx] - f_peak[right_idx - 1] + 1e-10)
            r_right = R_peak[right_idx - 1] + frac * (R_peak[right_idx] - R_peak[right_idx - 1])
        else:
            r_right = R_peak[right_idx]
    else:
        r_right = R_peak[min(len(R_peak) - 1, right_idx)]
    
    fwhm = r_right - r_left
    sigma_fwhm = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355 * σ
    
    # 方法2：高斯拟合（作为验证）
    sigma_fit = None
    try:
        def gaussian(r, A, r0_fit, sigma):
            return A * np.exp(-(r - r0_fit)**2 / (2 * sigma**2))
        
        # 只拟合正值部分
        fit_mask = f_peak > peak_val * 0.1
        if np.sum(fit_mask) >= 5:
            R_fit = R_peak[fit_mask]
            f_fit = f_peak[fit_mask]
            
            popt, _ = curve_fit(gaussian, R_fit, f_fit, 
                               p0=[peak_val, r_peak_pos, max(sigma_fwhm, 0.1)],
                               bounds=([0, r0 - 3, 0.05], [np.inf, r0 + 3, 5.0]),
                               maxfev=1000)
            sigma_fit = abs(popt[2])
    except:
        pass
    
    # 综合两种方法
    if sigma_fit is not None and abs(sigma_fit - sigma_fwhm) < 0.5:
        sigma_est = (sigma_fwhm + sigma_fit) / 2
    elif sigma_fit is not None and sigma_fit < sigma_fwhm:
        # 拟合值通常更准确（如果合理）
        sigma_est = sigma_fit
    else:
        sigma_est = sigma_fwhm
    
    # 扣除正则化引入的展宽（近似）
    # 正则化相当于在分母加了 regularization²，会导致约 regularization/2 的展宽
    # 但实际影响较小，只做轻微修正
    sigma_reg_correction = regularization * 0.1
    sigma_est = max(sigma_est - sigma_reg_correction, 0.1)
    
    if verbose:
        print(f"  Hansen-Law: σ_fwhm={sigma_fwhm:.3f}, σ_fit={sigma_fit if sigma_fit else 'N/A'}, "
              f"reg_corr={sigma_reg_correction:.3f}, 最终={sigma_est:.3f}")
    
    return sigma_est


def estimate_sigma_delta_r(R, rho, drho_dR, r0, dr, verbose=False):
    """
    ΔR 方法：利用 r_peak - r_steep ∝ σ
    
    原理：
    - r_peak: 密度曲线的局部最大值
    - r_steep: 导数曲线的局部最大值（最陡上升点）
    - 对于 Abel 投影的高斯环：σ ≈ k * (r_peak - r_steep)
    """
    from scipy.ndimage import gaussian_filter1d
    
    # 在 r0 附近找 r_peak（密度最大值）
    search_range = 3.0
    mask = (R > r0 - search_range) & (R < r0 + search_range)
    R_local = R[mask]
    rho_local = rho[mask]
    
    if len(R_local) < 5:
        return None
    
    # 平滑后找峰
    rho_smooth = gaussian_filter1d(rho_local, sigma=0.3/dr)
    peak_idx = np.argmax(rho_smooth)
    r_peak = R_local[peak_idx]
    
    # 在 r0 附近找 r_steep（导数最大值）
    drho_local = drho_dR[mask]
    drho_smooth = gaussian_filter1d(drho_local, sigma=0.3/dr)
    
    # 导数最大值应该在峰的左侧（上升沿）
    left_mask = R_local < r_peak
    if np.sum(left_mask) < 3:
        return None
    
    steep_idx = np.argmax(drho_smooth[left_mask])
    r_steep = R_local[left_mask][steep_idx]
    
    delta_r = r_peak - r_steep
    
    # 经验系数：通过模拟校准
    # 对于 Abel 投影的高斯环，σ ≈ 0.8 * delta_r
    k = 0.8
    sigma_est = k * delta_r
    
    if verbose:
        print(f"  ΔR 方法: r_peak = {r_peak:.2f}, r_steep = {r_steep:.2f}, "
              f"ΔR = {delta_r:.2f}, σ = {sigma_est:.2f}")
    
    return sigma_est if delta_r > 0.1 else None


def estimate_sigma_intensity_domain(R, rho, r0, sigma_range, verbose=False):
    """
    强度域模板匹配（原方法，作为参考）
    """
    r_window = 4.0
    mask = (R > r0 - r_window) & (R < r0 + r_window)
    R_local = R[mask]
    rho_local = rho[mask]
    
    if len(R_local) < 10:
        return None
    
    rho_norm = rho_local / (np.max(rho_local) + 1e-10)
    
    best_corr = -np.inf
    best_sigma = sigma_range[0]
    
    for sigma in sigma_range:
        template = abel_projection(R_local, r0, sigma)
        if np.max(template) < 1e-10:
            continue
        template_norm = template / np.max(template)
        
        corr = np.corrcoef(rho_norm, template_norm)[0, 1]
        
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_sigma = sigma
    
    if verbose:
        print(f"  强度域匹配: σ = {best_sigma:.2f}, 相关 = {best_corr:.3f}")
    
    return best_sigma if best_corr > 0.5 else None


# ============================================================
# 拟合模型
# ============================================================

def model_n_peaks(params, R, n_peaks):
    """
    N峰模型
    
    params结构: [r0_1, sigma_1, amp_1, ..., r0_n, sigma_n, amp_n, rho_bg, dc_offset]
    """
    rho_bg = params[-2]
    dc_offset = params[-1]
    
    mu = np.zeros_like(R, dtype=float)
    for i in range(n_peaks):
        r0 = params[3*i]
        sigma = params[3*i + 1]
        amp = params[3*i + 2]
        mu += amp * abel_projection(R, r0, sigma)
    
    mu = 2 * np.pi * R * (mu + rho_bg) + dc_offset
    return np.maximum(mu, 1e-9)


def loss_regional_gaussian(params, R, data, n_peaks, sigma_dc=20):
    """
    分区域高斯约束的损失函数
    
    物理约束：
    1. 标准化残差 z = (data - model) / √(model + σ²_dc) 应该是 N(0,1)
    2. 全局和各区域的mean都应该≈0
    3. 全局var应该≈1
    """
    mu = model_n_peaks(params, R, n_peaks)
    total_var = mu + sigma_dc**2
    z = (data - mu) / np.sqrt(total_var)
    
    n = len(z)
    chi2 = np.sum(z**2)
    
    # 全局约束
    global_mean = z.mean()
    global_var = z.var()
    
    # 分区约束：根据峰位置自动划分
    peak_positions = [params[3*i] for i in range(n_peaks)]
    peak_sigmas = [params[3*i + 1] for i in range(n_peaks)]
    
    # 构建区域边界
    boundaries = [0]
    for i in range(n_peaks):
        boundaries.append(peak_positions[i] - 2*peak_sigmas[i])
        boundaries.append(peak_positions[i] + 2*peak_sigmas[i])
    boundaries.append(np.max(R) + 1)
    boundaries = sorted(set(boundaries))
    
    regional_mean_penalty = 0
    for i in range(len(boundaries) - 1):
        mask = (R >= boundaries[i]) & (R < boundaries[i+1])
        if np.sum(mask) < 3:
            continue
        regional_mean_penalty += np.sum(mask) * z[mask].mean()**2
    
    # 总损失
    loss = chi2
    loss += n * 10 * global_mean**2
    loss += n * 10 * (global_var - 1)**2
    loss += 10 * regional_mean_penalty
    
    return loss


def fit_peaks(R, H_R, peak_init, sigma_max=None, sigma_dc=20, sigma_init=None, fix_r0=False, r0_tolerance_pct=5.0):
    """
    拟合峰参数
    
    Parameters:
    -----------
    R : array, 半径数组
    H_R : array, 直方图计数
    peak_init : list, 初始峰位置 [r0_1, r0_2, ...]
    sigma_max : float, sigma上限（默认为峰间距/4）
    sigma_dc : float, DC噪声标准差
    sigma_init : list, 初始sigma值（来自解耦分析）
    fix_r0 : bool, 是否固定 r0（只优化 σ 和 amplitude）
    r0_tolerance_pct : float, r0 允许的相对偏移（百分比，默认 5%）
                       Mexican Hat 检测的 r0 偏移约 2-6%，所以 5% 是合理的默认值
    
    Returns:
    --------
    result : dict, 包含拟合参数和诊断信息
    """
    n_peaks = len(peak_init)
    
    # 确保峰按位置排序
    if sigma_init is not None:
        # 同时排序 peak_init 和 sigma_init
        sorted_pairs = sorted(zip(peak_init, sigma_init), key=lambda x: x[0])
        peak_init = [p[0] for p in sorted_pairs]
        sigma_init = [p[1] for p in sorted_pairs]
    else:
        peak_init = sorted(peak_init)
    
    # 自适应sigma_max
    if sigma_max is None and n_peaks > 1:
        min_separation = min(peak_init[i+1] - peak_init[i] for i in range(n_peaks-1))
        sigma_max = max(min_separation / 4, 0.5)  # 至少0.5
    elif sigma_max is None:
        sigma_max = 2.0
    
    if fix_r0:
        # 固定 r0，只优化 σ 和 amplitude
        return fit_peaks_fixed_r0(R, H_R, peak_init, sigma_max, sigma_dc, sigma_init)
    
    # 构建初值和边界
    init = []
    bounds = []
    for i, r0 in enumerate(peak_init):
        # 使用解耦分析的sigma作为初值（如果提供）
        if sigma_init is not None and i < len(sigma_init):
            sigma_0 = min(sigma_init[i], sigma_max * 0.9)  # 确保在边界内
            sigma_0 = max(sigma_0, 0.15)  # 确保不太小
        else:
            sigma_0 = sigma_max / 2
        
        # r0 容差：相对值
        r0_tol = r0 * r0_tolerance_pct / 100.0
        
        init.extend([r0, sigma_0, 1.0])  # r0, sigma, amp
        bounds.extend([
            (max(r0 - r0_tol, 0.1), r0 + r0_tol),  # r0，限制在检测值 ±1%
            (0.1, max(sigma_max, 0.5)),   # sigma
            (0, None)                      # amp
        ])
    init.extend([0.3, 50])  # rho_bg, dc_offset
    bounds.extend([(0, None), (0, None)])
    
    # 优化
    res = minimize(loss_regional_gaussian, init, 
                   args=(R, H_R, n_peaks, sigma_dc),
                   bounds=bounds, method='L-BFGS-B')
    
    # 提取结果
    result = {
        'n_peaks': n_peaks,
        'peaks': [],
        'rho_bg': res.x[-2],
        'dc_offset': res.x[-1],
        'loss': res.fun,
        'success': res.success
    }
    
    for i in range(n_peaks):
        result['peaks'].append({
            'r0': res.x[3*i],
            'sigma': res.x[3*i + 1],
            'amp': res.x[3*i + 2]
        })
    
    # 计算拟合曲线和残差诊断
    H_fit = model_n_peaks(res.x, R, n_peaks)
    total_var = H_fit + sigma_dc**2
    z = (H_R - H_fit) / np.sqrt(total_var)
    
    result['H_fit'] = H_fit
    result['residual_mean'] = z.mean()
    result['residual_var'] = z.var()
    
    return result


def model_n_peaks_fixed_r0(params, R, r0_fixed, n_peaks):
    """
    N峰模型（r0 固定）
    
    params结构: [sigma_1, amp_1, ..., sigma_n, amp_n, rho_bg, dc_offset]
    r0_fixed: list, 固定的峰位置
    """
    rho_bg = params[-2]
    dc_offset = params[-1]
    
    mu = np.zeros_like(R, dtype=float)
    for i in range(n_peaks):
        r0 = r0_fixed[i]
        sigma = params[2*i]
        amp = params[2*i + 1]
        mu += amp * abel_projection(R, r0, sigma)
    
    mu = 2 * np.pi * R * (mu + rho_bg) + dc_offset
    return np.maximum(mu, 1e-9)


def loss_fixed_r0(params, R, data, r0_fixed, n_peaks, sigma_dc=20):
    """
    固定 r0 的损失函数
    """
    mu = model_n_peaks_fixed_r0(params, R, r0_fixed, n_peaks)
    total_var = mu + sigma_dc**2
    z = (data - mu) / np.sqrt(total_var)
    
    n = len(z)
    chi2 = np.sum(z**2)
    
    # 全局约束
    global_mean = z.mean()
    global_var = z.var()
    
    # 分区约束
    peak_sigmas = [params[2*i] for i in range(n_peaks)]
    
    boundaries = [0]
    for i in range(n_peaks):
        boundaries.append(r0_fixed[i] - 2*peak_sigmas[i])
        boundaries.append(r0_fixed[i] + 2*peak_sigmas[i])
    boundaries.append(np.max(R) + 1)
    boundaries = sorted(set(boundaries))
    
    regional_mean_penalty = 0
    for i in range(len(boundaries) - 1):
        mask = (R >= boundaries[i]) & (R < boundaries[i+1])
        if np.sum(mask) < 3:
            continue
        regional_mean_penalty += np.sum(mask) * z[mask].mean()**2
    
    # 总损失
    loss = chi2
    loss += n * 10 * global_mean**2
    loss += n * 10 * (global_var - 1)**2
    loss += 10 * regional_mean_penalty
    
    return loss


def fit_peaks_fixed_r0(R, H_R, r0_fixed, sigma_max=2.0, sigma_dc=20, sigma_init=None):
    """
    固定 r0 的 MLE 拟合
    
    优点：
    - 减少参数空间维度（从 3n+2 到 2n+2）
    - 避免 r 和 σ 的耦合
    - 更容易收敛
    
    Parameters:
    -----------
    R : array, 半径数组
    H_R : array, 直方图计数
    r0_fixed : list, 固定的峰位置
    sigma_max : float, sigma上限
    sigma_dc : float, DC噪声标准差
    sigma_init : list, 初始sigma值
    
    Returns:
    --------
    result : dict, 包含拟合参数和诊断信息
    """
    n_peaks = len(r0_fixed)
    
    # 构建初值和边界
    init = []
    bounds = []
    for i in range(n_peaks):
        if sigma_init is not None and i < len(sigma_init):
            sigma_0 = min(sigma_init[i], sigma_max * 0.9)
            sigma_0 = max(sigma_0, 0.15)
        else:
            sigma_0 = sigma_max / 2
        
        init.extend([sigma_0, 1.0])  # sigma, amp
        bounds.extend([
            (0.1, max(sigma_max, 0.5)),  # sigma
            (0, None)                     # amp
        ])
    init.extend([0.3, 50])  # rho_bg, dc_offset
    bounds.extend([(0, None), (0, None)])
    
    # 优化
    res = minimize(loss_fixed_r0, init, 
                   args=(R, H_R, r0_fixed, n_peaks, sigma_dc),
                   bounds=bounds, method='L-BFGS-B')
    
    # 提取结果
    result = {
        'n_peaks': n_peaks,
        'peaks': [],
        'rho_bg': res.x[-2],
        'dc_offset': res.x[-1],
        'loss': res.fun,
        'success': res.success,
        'r0_fixed': True
    }
    
    for i in range(n_peaks):
        result['peaks'].append({
            'r0': r0_fixed[i],
            'sigma': res.x[2*i],
            'amp': res.x[2*i + 1]
        })
    
    # 计算拟合曲线和残差诊断
    H_fit = model_n_peaks_fixed_r0(res.x, R, r0_fixed, n_peaks)
    total_var = H_fit + sigma_dc**2
    z = (H_R - H_fit) / np.sqrt(total_var)
    
    result['H_fit'] = H_fit
    result['residual_mean'] = z.mean()
    result['residual_var'] = z.var()
    
    return result


# ============================================================
# 主流程
# ============================================================

def reconstruct_vmi_from_image(image, pixel_size=0.1, dr=None, n_peaks=None, verbose=True, 
                                use_decoupled=True, dld_resolution=0.0):
    """
    从 2D 图像重建 VMI
    
    Parameters:
    -----------
    image : 2D array, VMI 图像
    pixel_size : float, 像素大小 (mm)
    dr : float, 径向 bin 宽度（默认等于 pixel_size）
    n_peaks : int, 峰数量
    verbose : bool, 是否打印详细信息
    use_decoupled : bool, 是否使用 r-σ 解耦分析
    dld_resolution : float, DLD 量化分辨率
    
    Returns:
    --------
    result : dict, 重建结果
    
    注意：
    图像输入的展宽来源：
    1. DLD 量化：σ²_dld = dld_res² / 12
    2. 像素化：σ²_pixel = pixel_size² / 12
    3. 径向 binning：σ²_bin = dr² / 12
    总展宽：σ²_total = σ²_dld + σ²_pixel + σ²_bin
    """
    if dr is None:
        dr = pixel_size
    
    ny, nx = image.shape
    
    # 找中心（质心）
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    total = np.sum(image)
    x_center = np.sum(x_idx * image) / total * pixel_size - nx/2 * pixel_size
    y_center = np.sum(y_idx * image) / total * pixel_size - ny/2 * pixel_size
    
    # 转换为物理坐标
    x_phys = (x_idx - nx/2) * pixel_size
    y_phys = (y_idx - ny/2) * pixel_size
    r_phys = np.sqrt((x_phys - x_center)**2 + (y_phys - y_center)**2)
    
    # 径向积分
    r_max = np.max(r_phys) * 0.95
    n_bins = int(r_max / dr)
    
    R = np.linspace(dr/2, r_max - dr/2, n_bins)
    H_R = np.zeros(n_bins)
    
    for i in range(n_bins):
        r_min = i * dr
        r_max_bin = (i + 1) * dr
        mask = (r_phys >= r_min) & (r_phys < r_max_bin)
        H_R[i] = np.sum(image[mask])
    
    rho_R = H_R / (2 * np.pi * R + 1e-6)
    
    # 计算总展宽（图像输入比 XY 输入多一个像素化展宽）
    sigma_pixel_sq = pixel_size**2 / 12
    sigma_bin_sq = dr**2 / 12
    sigma_dld_sq = dld_resolution**2 / 12
    total_broadening = np.sqrt(sigma_pixel_sq + sigma_bin_sq + sigma_dld_sq)
    
    if verbose:
        print(f"图像大小: {nx} x {ny}, 像素大小: {pixel_size} mm")
        print(f"中心: ({x_center:.2f}, {y_center:.2f})")
        print(f"r_max: {r_max:.2f}, dr: {dr}, bins: {n_bins}")
        print(f"\n展宽来源:")
        print(f"  像素化: σ_pixel = {np.sqrt(sigma_pixel_sq):.3f} mm")
        print(f"  径向 binning: σ_bin = {np.sqrt(sigma_bin_sq):.3f} mm")
        print(f"  DLD 量化: σ_dld = {np.sqrt(sigma_dld_sq):.3f} mm")
        print(f"  总展宽: σ_total = {total_broadening:.3f} mm")
    
    # 峰检测
    peaks_single = detect_peaks_mexican_hat(R, rho_R)
    if n_peaks is None:
        n_peaks = min(len(peaks_single), 5)
    
    # r-σ 解耦分析（传入总展宽）
    if use_decoupled:
        # 对于图像输入，需要传入像素化展宽 + 径向 binning 展宽
        effective_broadening = np.sqrt(sigma_pixel_sq + sigma_bin_sq + sigma_dld_sq)
        estimates, decoupled_data = estimate_r_sigma_decoupled(
            R, rho_R, n_peaks=n_peaks, verbose=verbose,
            dr=np.sqrt(dr**2 + pixel_size**2),  # 等效 dr
            dld_resolution=dld_resolution
        )
        peak_init = [e['r0'] for e in estimates]
        sigma_init = [e['sigma'] for e in estimates]
    else:
        peak_init, _ = multiscale_peak_detection_from_histogram(R, rho_R, n_peaks=n_peaks)
        sigma_init = None
        decoupled_data = None
    
    # 拟合
    if verbose:
        print("\n========== 拟合 ==========")
    
    fit_result = fit_peaks(R, H_R, peak_init, sigma_init=sigma_init)  # 默认 r0_tolerance_pct=5%
    
    if verbose:
        for i, p in enumerate(fit_result['peaks']):
            print(f"Peak {i+1}: r0 = {p['r0']:.2f}, σ = {p['sigma']:.2f}")
        print(f"残差: mean = {fit_result['residual_mean']:.3f}, var = {fit_result['residual_var']:.3f}")
    
    result = {
        'center': (x_center, y_center),
        'R': R,
        'H_R': H_R,
        'rho_R': rho_R,
        'fit': fit_result,
        'decoupled': decoupled_data,
        'broadening': {
            'pixel': np.sqrt(sigma_pixel_sq),
            'bin': np.sqrt(sigma_bin_sq),
            'dld': np.sqrt(sigma_dld_sq),
            'total': total_broadening
        }
    }
    
    return result


def multiscale_peak_detection_from_histogram(R, rho, n_peaks=2):
    """从直方图数据进行多尺度峰检测"""
    peaks = detect_peaks_mexican_hat(R, rho)
    peaks = peaks[:n_peaks]
    peaks = sorted(peaks)
    return peaks, [np.nan] * len(peaks)


def reconstruct_vmi(xy_data, dr=None, n_bins=None, n_peaks=None, verbose=True, use_decoupled=True, 
                    dld_resolution=0.0, sigma_min_expected=None):
    """
    VMI重建主函数
    
    Parameters:
    -----------
    xy_data : array (N, 2), XY坐标数据
    dr : float, 径向bin宽度（None则自动选择）
    n_bins : int, bin数量（与 dr 二选一，优先级更高）
    n_peaks : int, 峰数量（None则自动检测）
    verbose : bool, 是否打印详细信息
    use_decoupled : bool, 是否使用 r-σ 解耦分析
    dld_resolution : float, DLD量化分辨率（用于展宽修正）
    sigma_min_expected : float, 预期最小峰宽（None则从数据估计）
    
    Returns:
    --------
    result : dict, 重建结果
    """
    # 1. 计算中心和径向分布
    x_center = np.mean(xy_data[:, 0])
    y_center = np.mean(xy_data[:, 1])
    r = np.sqrt((xy_data[:, 0] - x_center)**2 + (xy_data[:, 1] - y_center)**2)
    r_max = np.percentile(r, 99)
    N_events = len(xy_data)
    
    # 2. 确定 dr
    if n_bins is not None:
        # 用户指定 n_bins
        dr = r_max / n_bins
    elif dr is not None:
        # 用户指定 dr
        n_bins = int(r_max / dr)
    else:
        # 自动选择 dr
        
        # 如果没有提供 sigma_min，先用粗略 binning 估计
        if sigma_min_expected is None:
            # 用粗略的 dr=0.5 做初步分析
            dr_coarse = 0.5
            n_bins_coarse = int(r_max / dr_coarse)
            counts_coarse, edges_coarse = np.histogram(r, bins=n_bins_coarse, range=(0, r_max))
            R_coarse = (edges_coarse[:-1] + edges_coarse[1:]) / 2
            rho_coarse = counts_coarse / (2 * np.pi * R_coarse + 1e-6)
            
            # 用响应宽度方法粗略估计 sigma
            peaks_rw = response_width_method(R_coarse, rho_coarse, sigma_template=1.0, min_r=5)
            if len(peaks_rw) > 0:
                sigma_min_expected = min(p['sigma_est'] for p in peaks_rw[:3])
                sigma_min_expected = max(sigma_min_expected, 0.2)  # 至少 0.2 mm
            else:
                sigma_min_expected = 0.3  # 默认值
            
            if verbose:
                print(f"从数据估计 σ_min ≈ {sigma_min_expected:.2f} mm")
        
        # 策略1: 基于 sigma（Nyquist 采样）
        dr_nyquist = sigma_min_expected / 3
        
        # 策略2: 基于 DLD 分辨率
        dr_dld = max(dld_resolution, 0.01)
        
        # 策略3: 基于统计量（每个 bin 平均 ≥ 10 个事件）
        dr_stats = r_max * 10 / N_events
        
        # 取最大值
        dr = max(dr_nyquist, dr_dld, dr_stats)
        dr = min(dr, 0.5)  # 最大 0.5 mm
        
        n_bins = int(r_max / dr)
        
        if verbose:
            print(f"自动选择 dr:")
            print(f"  σ_min = {sigma_min_expected:.2f} mm → dr_nyquist = {dr_nyquist:.3f}")
            print(f"  DLD 分辨率 = {dld_resolution:.3f} mm → dr_dld = {dr_dld:.3f}")
            print(f"  统计量 ({N_events} events) → dr_stats = {dr_stats:.3f}")
            print(f"  最终: dr = {dr:.3f} mm, n_bins = {n_bins}")
    
    counts, bin_edges = np.histogram(r, bins=n_bins, range=(0, r_max))
    R = (bin_edges[:-1] + bin_edges[1:]) / 2
    H_R = counts.astype(float)
    rho_R = H_R / (2 * np.pi * R + 1e-6)
    
    if verbose:
        print(f"数据点数: {len(xy_data)}")
        print(f"中心: ({x_center:.2f}, {y_center:.2f})")
        print(f"r_max: {r_max:.2f}, dr: {dr}, bins: {n_bins}")
    
    # 2. 峰检测
    peaks_single = detect_peaks_mexican_hat(R, rho_R)
    if n_peaks is None:
        n_peaks = min(len(peaks_single), 5)
    
    # 3. r-σ 解耦分析
    if use_decoupled:
        estimates, decoupled_data = estimate_r_sigma_decoupled(
            R, rho_R, n_peaks=n_peaks, verbose=verbose, 
            dr=dr, dld_resolution=dld_resolution
        )
        
        # 使用解耦估计作为初值
        peak_init = [e['r0'] for e in estimates]
        sigma_init = [e['sigma'] for e in estimates]
    else:
        # 传统多尺度检测
        if verbose:
            print("\n========== 多尺度峰检测 ==========")
        peak_init, peak_stds = multiscale_peak_detection(r, r_max, n_peaks=n_peaks)
        sigma_init = None
        decoupled_data = None
        
        if verbose:
            for i, (m, s) in enumerate(zip(peak_init, peak_stds)):
                print(f"Peak {i+1}: {m:.2f} ± {s:.2f}")
    
    # 4. 拟合（使用解耦估计的 sigma 作为初值）
    if verbose:
        print("\n========== 拟合 ==========")
    
    fit_result = fit_peaks(R, H_R, peak_init, sigma_init=sigma_init)  # 默认 r0_tolerance_pct=5%
    
    if verbose:
        for i, p in enumerate(fit_result['peaks']):
            print(f"Peak {i+1}: r0 = {p['r0']:.2f}, σ = {p['sigma']:.2f}")
        print(f"残差: mean = {fit_result['residual_mean']:.3f}, var = {fit_result['residual_var']:.3f}")
    
    # 5. 多尺度验证
    if verbose:
        print("\n========== 多尺度验证 ==========")
        print(f"{'dr':<8}", end='')
        for i in range(n_peaks):
            print(f"{'数据P'+str(i+1):<10} {'拟合P'+str(i+1):<10}", end='')
        print()
        
        for dr_test in [0.1, 0.15, 0.2, 0.25, 0.3]:
            n_bins_test = int(r_max / dr_test)
            counts_test, bin_edges_test = np.histogram(r, bins=n_bins_test, range=(0, r_max))
            R_test = (bin_edges_test[:-1] + bin_edges_test[1:]) / 2
            rho_data = counts_test / (2 * np.pi * R_test + 1e-6)
            
            peaks_data = detect_peaks_mexican_hat(R_test, rho_data)[:n_peaks]
            peaks_data = sorted(peaks_data) if len(peaks_data) >= n_peaks else [np.nan]*n_peaks
            
            params = []
            for p in fit_result['peaks']:
                params.extend([p['r0'], p['sigma'], p['amp']])
            params.extend([fit_result['rho_bg'], fit_result['dc_offset']])
            H_fit_test = model_n_peaks(params, R_test, n_peaks)
            rho_fit = H_fit_test / (2 * np.pi * R_test + 1e-6)
            peaks_fit = detect_peaks_mexican_hat(R_test, rho_fit)[:n_peaks]
            peaks_fit = sorted(peaks_fit) if len(peaks_fit) >= n_peaks else [np.nan]*n_peaks
            
            print(f"{dr_test:<8.2f}", end='')
            for i in range(n_peaks):
                pd = peaks_data[i] if i < len(peaks_data) else np.nan
                pf = peaks_fit[i] if i < len(peaks_fit) else np.nan
                print(f"{pd:<10.2f} {pf:<10.2f}", end='')
            print()
    
    # 返回结果
    result = {
        'center': (x_center, y_center),
        'R': R,
        'H_R': H_R,
        'rho_R': rho_R,
        'peak_detection': {'init': peak_init},
        'fit': fit_result,
        'decoupled': decoupled_data
    }
    
    return result


def plot_result(result):
    """绘制结果"""
    R = result['R']
    H_R = result['H_R']
    rho_R = result['rho_R']
    H_fit = result['fit']['H_fit']
    rho_fit = H_fit / (2 * np.pi * R + 1e-6)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 直方图 + 拟合
    ax = axes[0, 0]
    ax.step(R, H_R, where='mid', color='black', lw=1.5, label='Data')
    ax.plot(R, H_fit, 'r-', lw=2, label='Fit')
    for i, p in enumerate(result['fit']['peaks']):
        ax.axvline(p['r0'], ls='--', alpha=0.5, label=f"Peak{i+1}: r0={p['r0']:.2f}, σ={p['sigma']:.2f}")
    ax.set_xlabel('Radius (R)')
    ax.set_ylabel('Counts H(R)')
    ax.set_title('Radial Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # 密度分布
    ax = axes[0, 1]
    ax.step(R, rho_R, where='mid', color='blue', lw=1.5, label='Data ρ(R)')
    ax.plot(R, rho_fit, 'r-', lw=2, label='Fit')
    ax.set_xlabel('Radius (R)')
    ax.set_ylabel('Density ρ(R) = H(R)/(2πR)')
    ax.set_title('Density Corrected')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 残差
    ax = axes[1, 0]
    sigma_dc = 20
    total_var = H_fit + sigma_dc**2
    z = (H_R - H_fit) / np.sqrt(total_var)
    ax.scatter(R, z, s=5, alpha=0.5)
    ax.axhline(0, color='k', ls='--')
    ax.axhline(2, color='r', ls=':', alpha=0.5)
    ax.axhline(-2, color='r', ls=':', alpha=0.5)
    ax.set_xlabel('Radius (R)')
    ax.set_ylabel('Standardized Residual')
    ax.set_title(f'Residuals (mean={z.mean():.2f}, var={z.var():.2f})')
    ax.grid(alpha=0.3)
    
    # 残差直方图
    ax = axes[1, 1]
    ax.hist(z, bins=30, density=True, alpha=0.7)
    x_gauss = np.linspace(-4, 4, 100)
    ax.plot(x_gauss, np.exp(-x_gauss**2/2)/np.sqrt(2*np.pi), 'r-', lw=2, label='N(0,1)')
    ax.set_xlabel('Standardized Residual')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # 加载数据
    print("加载数据...")
    mat_data = loadmat('electron_shilpa_XY.mat')
    XY = mat_data['XY']
    
    # 重建
    result = reconstruct_vmi(XY, dr=0.1, n_peaks=2)
    
    # 绘图
    plot_result(result)



# ============================================================
# 测试：用模拟数据验证重建精度
# ============================================================

def test_reconstruction_accuracy():
    """
    用已知真值的模拟数据测试重建精度
    测试理想数据和带噪声数据
    """
    from Abel_forward_simulation import Config, run_simulation
    
    print("=" * 70)
    print("重建精度测试")
    print("=" * 70)
    
    # 定义测试用例
    test_cases = [
        # (E_centers, sigmas, N_events, description)
        ([1.0], [0.05], 50000, "单峰，窄"),
        ([1.0, 2.0], [0.05, 0.05], 100000, "双峰，间距大"),
        ([1.0, 1.5], [0.05, 0.05], 100000, "双峰，间距小"),
    ]
    
    results_ideal = []
    results_noisy = []
    
    for E_centers, sigmas, N_events, desc in test_cases:
        print(f"\n{'='*60}")
        print(f"测试: {desc}")
        print(f"真值: E = {E_centers}, σ_laser = {sigmas}")
        print(f"{'='*60}")
        
        # 计算真实的r0
        from scipy.constants import electron_mass, elementary_charge
        mass_kg = electron_mass
        
        E_max = max(E_centers) * 1.5
        r_max_mm = 20.0
        v_max = np.sqrt(2.0 * E_max * elementary_charge / mass_kg)
        vmi_k = r_max_mm / v_max
        
        true_r0 = []
        true_sigma_r = []
        for E, sigma_E in zip(E_centers, sigmas):
            v = np.sqrt(2.0 * E * elementary_charge / mass_kg)
            r0 = vmi_k * v
            true_r0.append(r0)
            sigma_r = r0 * sigma_E / (2 * E)
            true_sigma_r.append(sigma_r)
        
        print(f"真实 r0 (mm): {[f'{r:.2f}' for r in true_r0]}")
        print(f"真实 σ_r (mm): {[f'{s:.3f}' for s in true_sigma_r]}")
        
        # 创建配置
        config = Config(
            E_centers=E_centers,
            Betas=[0.0] * len(E_centers),
            branching_ratios=[1.0] * len(E_centers),
            N_events=N_events,
            vmi_k=vmi_k,
            sigma_laser=sigmas[0] if len(set(sigmas)) == 1 else 0.05,
            T_beam=0.0,
            tau_lifetimes=0.0,
            vol_sigma=(0.0, 0.0, 0.0),
            polarization_vec=[0, 0, 1],
            img_res=512,
            pixel_size=0.1,
            psf_fwhm=0.2,  # 有PSF
            dld_resolution=0.01,  # DLD量化
            dark_rate=0.0,
            readout_sigma=0.0,
            readout_offset=0.0,
            bg_rate=0.0,
        )
        
        # 测试1：理想数据（无噪声）
        print(f"\n--- 理想数据 (无噪声) ---")
        xy_ideal, _ = run_simulation(config, add_noise=False, output_mode='xy_ideal')
        result_ideal = reconstruct_vmi(xy_ideal, dr=0.05, n_peaks=len(E_centers), verbose=False)
        
        for i, p in enumerate(result_ideal['fit']['peaks']):
            if i < len(true_r0):
                r0_err = (p['r0'] - true_r0[i]) / true_r0[i] * 100
                sigma_err = (p['sigma'] - true_sigma_r[i]) / true_sigma_r[i] * 100 if true_sigma_r[i] > 0 else np.nan
                print(f"  Peak {i+1}: r0 = {p['r0']:.3f} (误差 {r0_err:+.1f}%), σ = {p['sigma']:.3f} (误差 {sigma_err:+.1f}%)")
                results_ideal.append({
                    'desc': desc, 'peak': i+1,
                    'r0_err_pct': r0_err, 'sigma_err_pct': sigma_err
                })
        
        # 测试2：带噪声数据（PSF + DLD量化）
        print(f"\n--- 带噪声数据 (PSF + DLD) ---")
        xy_noisy, meta = run_simulation(config, add_noise=False, output_mode='xy_dld')
        print(f"  PSF σ = {meta.get('psf_sigma_mm', 0):.3f} mm, DLD分辨率 = {meta.get('dld_resolution_mm', 0):.3f} mm")
        result_noisy = reconstruct_vmi(xy_noisy, dr=0.05, n_peaks=len(E_centers), verbose=False)
        
        for i, p in enumerate(result_noisy['fit']['peaks']):
            if i < len(true_r0):
                r0_err = (p['r0'] - true_r0[i]) / true_r0[i] * 100
                sigma_err = (p['sigma'] - true_sigma_r[i]) / true_sigma_r[i] * 100 if true_sigma_r[i] > 0 else np.nan
                print(f"  Peak {i+1}: r0 = {p['r0']:.3f} (误差 {r0_err:+.1f}%), σ = {p['sigma']:.3f} (误差 {sigma_err:+.1f}%)")
                results_noisy.append({
                    'desc': desc, 'peak': i+1,
                    'r0_err_pct': r0_err, 'sigma_err_pct': sigma_err
                })
    
    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print("\n理想数据 (无噪声):")
    r0_errs = [abs(r['r0_err_pct']) for r in results_ideal]
    sigma_errs = [abs(r['sigma_err_pct']) for r in results_ideal if not np.isnan(r['sigma_err_pct'])]
    print(f"  r0 平均绝对误差: {np.mean(r0_errs):.2f}%")
    print(f"  σ  平均绝对误差: {np.mean(sigma_errs):.2f}%")
    
    print("\n带噪声数据 (PSF + DLD):")
    r0_errs = [abs(r['r0_err_pct']) for r in results_noisy]
    sigma_errs = [abs(r['sigma_err_pct']) for r in results_noisy if not np.isnan(r['sigma_err_pct'])]
    print(f"  r0 平均绝对误差: {np.mean(r0_errs):.2f}%")
    print(f"  σ  平均绝对误差: {np.mean(sigma_errs):.2f}%")
    
    return results_ideal, results_noisy


if __name__ == "__main__":
    # 只运行精度测试
    test_reconstruction_accuracy()
