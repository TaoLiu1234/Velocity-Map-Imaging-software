"""
比较 V1, V2, V3 和 rBasex 四种方法重建 electron_shilpa_XY.mat

从原始XY坐标数据构建VMI图像，然后使用四种不同的 Abel 逆变换方法进行重建。
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import abel
import time
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks, peak_widths

# 导入各个重建方法
from Abel_backward_reconstruction import PhysicsBasedFitter
from Abel_backward_reconstruction_v2 import PhysicsBasedFitterV2
from Abel_backward_reconstruction_v3 import AbelReconstructorV3, DataCleaner, PolarTransformer
from Abel_rbasex_reconstruction import reconstruct_rbasex


def load_raw_data(filepath='electron_shilpa_XY.mat'):
    """加载原始XY坐标数据"""
    mat = sio.loadmat(filepath)
    print("Variables:", mat.keys())
    
    xy_data = mat['XY']
    print(f"XY data shape: {xy_data.shape}")
    print(f"X range: [{xy_data[:,0].min():.2f}, {xy_data[:,0].max():.2f}]")
    print(f"Y range: [{xy_data[:,1].min():.2f}, {xy_data[:,1].max():.2f}]")
    print(f"Total events: {len(xy_data)}")
    
    return xy_data


def build_vmi_image(xy_data, n_bins=512, center=True):
    """从XY坐标构建VMI图像
    
    Args:
        xy_data: (N, 2) array of X, Y coordinates
        n_bins: 图像尺寸
        center: 是否将图像中心对齐到坐标原点
    
    Returns:
        2D histogram image, pixel_size
    """
    x = xy_data[:, 0]
    y = xy_data[:, 1]
    
    # 确定范围
    if center:
        # 以原点为中心
        max_range = max(np.abs(x).max(), np.abs(y).max()) * 1.05
        x_range = [-max_range, max_range]
        y_range = [-max_range, max_range]
    else:
        # 使用数据范围
        margin = 0.05 * (x.max() - x.min())
        x_range = [x.min() - margin, x.max() + margin]
        y_range = [y.min() - margin, y.max() + margin]
    
    # 构建2D直方图
    image, x_edges, y_edges = np.histogram2d(
        x, y, 
        bins=n_bins, 
        range=[x_range, y_range]
    )
    
    # 转置使得y轴向上
    image = image.T
    
    print(f"Built VMI image: {image.shape}")
    print(f"Image value range: [{image.min():.1f}, {image.max():.1f}]")
    print(f"Total counts: {image.sum():.0f}")
    print(f"Non-zero pixels: {np.sum(image > 0)}")
    
    # 计算像素大小
    pixel_size = (x_range[1] - x_range[0]) / n_bins
    print(f"Pixel size: {pixel_size:.4f} mm")
    
    return image, pixel_size


def estimate_background_and_signal_region(image, bg_r_inner=144, bg_r_outer=234):
    """估计背景噪声的均值和方差
    
    使用用户指定的环形背景区域（bg_r_inner < r < bg_r_outer）来估计
    读出噪声+暗电流的均值和方差。这个背景是高斯噪声，均匀覆盖整个图像。
    
    Args:
        image: 输入图像
        bg_r_inner: 背景区域内边界（像素），信号已衰减到背景水平
        bg_r_outer: 背景区域外边界（像素），探测器边缘
    
    Returns:
        bg_mean: 背景均值（高斯噪声均值）
        bg_std: 背景标准差
        signal_radius: 信号区域半径（像素）= bg_r_inner
        radial_profile: 径向分布
    """
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    
    # 计算径向距离
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    r_int = r.astype(int)
    r_max = n // 2
    
    # 计算径向分布
    radial_sum = np.bincount(r_int.ravel(), weights=image.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count = np.bincount(r_int.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count
    
    # 使用用户指定的环形背景区域 (bg_r_inner < r < bg_r_outer)
    bg_mask = (r >= bg_r_inner) & (r <= bg_r_outer)
    
    # 从纯背景区域估计噪声的均值和方差
    bg_pixels = image[bg_mask]
    bg_mean = np.mean(bg_pixels)
    bg_std = np.std(bg_pixels)
    
    # 信号边界就是背景区域的内边界
    signal_radius = bg_r_inner
    
    print(f"\nBackground estimation from annular region ({bg_r_inner} < r < {bg_r_outer} px):")
    print(f"  Number of background pixels: {len(bg_pixels)}")
    print(f"  Background mean (Gaussian noise): {bg_mean:.4f} counts/pixel")
    print(f"  Background std: {bg_std:.4f} counts/pixel")
    print(f"  Signal region boundary: r = {signal_radius} px")
    
    # 也计算这个区域的径向分布均值作为验证
    bg_radial_mean = np.mean(radial_profile[bg_r_inner:bg_r_outer+1])
    print(f"  Radial profile mean in BG region: {bg_radial_mean:.4f}")
    
    return bg_mean, bg_std, signal_radius, radial_profile


def preprocess_image(image, bg_mean, bg_std, signal_radius, normalize=False):
    """预处理图像：从整个图像减去背景均值
    
    背景是均匀的高斯噪声（读出噪声+暗电流），覆盖整个图像。
    1. 整个图像减去背景均值
    2. 减完后 clip 到 0
    3. 背景区域（r > signal_radius）的残余噪声也设为 0
    4. 可选：99分位数归一化
    
    Args:
        image: 原始图像
        bg_mean: 背景均值（从环形纯背景区域估计）
        bg_std: 背景标准差
        signal_radius: 信号区域半径，超出此范围的设为0
        normalize: 是否做99分位数归一化
    
    Returns:
        预处理后的图像
    """
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    
    # 从整个图像减去背景均值
    image_sub = image - bg_mean
    
    # clip 到非负
    image_processed = np.maximum(image_sub, 0)
    
    # 背景区域（r > signal_radius）设为 0
    # 因为那里本来就是纯噪声，减去均值后的残余也是噪声
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    bg_mask = r > signal_radius
    image_processed[bg_mask] = 0
    
    print(f"\nPreprocessing:")
    print(f"  Background mean subtracted from ENTIRE image: {bg_mean:.4f} counts/pixel")
    print(f"  Signal region: r <= {signal_radius} px (kept)")
    print(f"  Background region: r > {signal_radius} px (set to 0)")
    print(f"  Original total counts: {image.sum():.0f}")
    print(f"  After preprocessing: {image_processed.sum():.0f}")
    
    # 可选：99分位数归一化
    if normalize:
        p99 = np.percentile(image_processed[image_processed > 0], 99)
        if p99 > 0:
            image_processed = image_processed / p99
            print(f"  Normalized by 99th percentile: {p99:.2f}")
            print(f"  After normalization: max={image_processed.max():.4f}")
    
    return image_processed


def reconstruct_radial_profile(params, n_r):
    """从峰参数重建径向分布（高斯峰叠加）
    
    Args:
        params: 峰参数列表，每个包含 'r', 'sigma', 'amp'
        n_r: 径向网格点数
    
    Returns:
        重建的径向分布
    """
    r_grid = np.arange(n_r)
    profile = np.zeros(n_r)
    
    for p in params:
        r0 = p['r']
        sigma = p.get('sigma', p.get('sigma_phys', 3.0))
        amp = p.get('amp', 1.0)
        
        # 高斯峰
        profile += amp * np.exp(-((r_grid - r0)**2) / (2 * sigma**2))
    
    return profile
    
    return image_processed


def run_v1_reconstruction(image, mask_radius=25, verbose=True):
    """运行 V1 重建方法
    
    Args:
        image: 输入图像
        mask_radius: 中心遮罩半径（像素），排除中心区域的峰检测
        verbose: 是否打印详细信息
    """
    if verbose:
        print("\n" + "="*60)
        print("V1 Reconstruction (PhysicsBasedFitter)")
        print("="*60)
    
    t0 = time.time()
    n_pixels = image.shape[0]
    fitter = PhysicsBasedFitter(n_pixels, mask_radius=mask_radius)
    
    try:
        params, r_grid, recon_profile = fitter.solve(image)
        elapsed = time.time() - t0
        
        if verbose:
            print(f"\nTime: {elapsed:.2f}s")
            print(f"Found {len(params)} peaks")
            for i, p in enumerate(params[:10]):
                print(f"  Peak {i+1}: r={p['r']:.1f}px, sigma={p['sigma']:.2f}px, beta={p['beta']:.2f}")
        
        return params, r_grid, recon_profile, elapsed
    except Exception as e:
        print(f"V1 reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return [], np.arange(n_pixels//2 + 1), np.zeros(n_pixels//2 + 1), 0


def run_v2_reconstruction(image, image_raw=None, mask_radius=25, verbose=True,
                          v2_height_thresh=0.08, v2_prominence_thresh=0.05, v2_distance=12):
    """运行 V2 重建方法 - 调整参数以检测主要峰
    
    Args:
        image: 预处理后的图像
        image_raw: 原始图像（用于 V2 的背景估计）
        mask_radius: 中心遮罩半径（像素），排除中心区域的峰检测
        verbose: 是否打印详细信息
        v2_height_thresh: 峰高度阈值（相对于最大值），默认 0.08 (8%)
        v2_prominence_thresh: 峰突出度阈值（相对于最大值），默认 0.05 (5%)
        v2_distance: 相邻峰最小距离（像素），默认 12
    """
    if verbose:
        print("\n" + "="*60)
        print("V2 Reconstruction (PhysicsBasedFitterV2 - adjusted)")
        print("="*60)
        print(f"  Peak detection params: height={v2_height_thresh:.0%}, "
              f"prominence={v2_prominence_thresh:.0%}, distance={v2_distance}")
    
    t0 = time.time()
    n_pixels = image.shape[0]
    
    # 使用原始图像让 V2 自己做背景估计
    input_image = image_raw if image_raw is not None else image
    
    try:
        # 使用适中的物理参数
        fitter = PhysicsBasedFitterV2(
            n_pixels,
            sigma_psf=0.5,
            sigma_pixel=0.3,
            sigma_interp=0.4,
            mask_radius=mask_radius
        )
        
        fitter.sigma_E = 0.01
        fitter.vmi_k = 0.001
        
        # 手动执行 Phase 0 预处理
        fitter._phase0_preprocess(input_image, n_theta=720)
        
        # 获取径向分布
        proj_profile = np.mean(fitter._polar, axis=1)
        
        # 使用可调参数进行峰检测
        max_val = np.max(proj_profile)
        peaks, properties = find_peaks(
            proj_profile,
            height=max_val * v2_height_thresh,
            distance=v2_distance,
            prominence=max_val * v2_prominence_thresh,
        )
        
        # 过滤掉中心区域的峰（r < mask_radius）
        peaks = [p for p in peaks if p >= mask_radius]
        
        if verbose:
            print(f"  Found {len(peaks)} peaks at r={peaks}")
        
        # Abel 逆变换
        abel_profile = abel.hansenlaw.hansenlaw_transform(proj_profile, direction='inverse')
        abel_profile = np.maximum(abel_profile, 0)
        
        # 对每个峰估计参数
        params = []
        for pk in peaks:
            # 估计 sigma
            pk_r, sigma, amp = fitter._phase1_estimate_sigma(
                abel_profile, pk, mask_radius=mask_radius, use_input_r=True
            )
            
            # 估计 beta
            beta, beta_std = fitter._estimate_beta_in_range(int(pk_r), sigma)
            
            params.append({
                'r': float(pk_r),
                'sigma': float(sigma),
                'amp': float(amp * pk_r),  # 密度校正
                'beta': float(beta),
                'beta_uncertainty': float(beta_std),
                'fwhm': float(sigma * 2.355),
            })
            
            if verbose:
                print(f"    Peak: r={pk_r:.1f}px, σ={sigma:.2f}px, β={beta:.2f}")
        
        elapsed = time.time() - t0
        
        if verbose:
            print(f"\nTime: {elapsed:.2f}s")
            print(f"Found {len(params)} peaks")
        
        # 从参数重建径向分布
        r_grid = np.arange(n_pixels // 2 + 1)
        recon_profile = reconstruct_radial_profile(params, len(r_grid))
        
        return params, r_grid, recon_profile, elapsed
    except Exception as e:
        print(f"V2 reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return [], np.arange(n_pixels//2 + 1), np.zeros(n_pixels//2 + 1), 0


def run_v3_reconstruction(image, image_raw=None, mask_radius=25, verbose=True):
    """运行 V3 重建方法 - 调用原始的 AbelReconstructorV3
    
    Args:
        image: 预处理后的图像（已减去背景，外围设为0）
        image_raw: 原始图像（用于 V3 的背景估计）
        mask_radius: 中心遮罩半径（像素），排除中心区域的峰检测
        verbose: 是否打印详细信息
    """
    if verbose:
        print("\n" + "="*60)
        print("V3 Reconstruction (AbelReconstructorV3)")
        print("="*60)
    
    t0 = time.time()
    n_pixels = image.shape[0]
    
    # 使用原始图像让 V3 自己做背景估计
    input_image = image_raw if image_raw is not None else image
    
    try:
        # 根据图像尺寸调整物理参数
        scale_factor = n_pixels / 256.0
        
        reconstructor = AbelReconstructorV3(
            sigma_psf=0.5 * scale_factor,
            sigma_pixel=0.3 * scale_factor,
            sigma_interp=0.55 * scale_factor,
            mask_radius=mask_radius
        )
        
        # 重写背景区域识别方法：
        # 1. 找到外围全为0的边界（数据边界）
        # 2. 找到信号衰减到背景水平的位置（信号边界）
        # 3. 背景区域 = 信号边界到数据边界之间
        def custom_identify_background_region(img, center=None):
            ny, nx = img.shape
            if center is None:
                center = (ny / 2, nx / 2)
            cy, cx = center
            y, x = np.ogrid[:ny, :nx]
            r = np.sqrt((y - cy)**2 + (x - cx)**2)
            r_int = r.astype(int)
            r_max = int(min(cy, ny - cy, cx, nx - cx))
            
            # 计算径向分布
            radial_sum = np.bincount(r_int.ravel(), weights=img.ravel(), minlength=r_max+1)[:r_max+1]
            radial_count = np.bincount(r_int.ravel(), minlength=r_max+1)[:r_max+1]
            radial_count[radial_count == 0] = 1
            radial_profile = radial_sum / radial_count
            
            # 找到外围全为0的边界（从外向内找第一个非零值）
            data_outer = r_max
            for r_idx in range(r_max - 1, 0, -1):
                if radial_profile[r_idx] > 0.1:
                    data_outer = r_idx
                    break
            
            # 找到信号峰值区域
            max_signal = np.max(radial_profile[20:data_outer])  # 排除中心
            
            # 从外向内找信号边界：找到径向分布开始明显上升的位置
            # 使用滑动窗口计算局部均值，找到均值开始明显高于背景的位置
            window_size = 10
            bg_inner = data_outer
            
            # 先估计背景水平（使用外围 20% 的数据）
            outer_start = int(data_outer * 0.8)
            bg_level = np.mean(radial_profile[outer_start:data_outer])
            bg_std = np.std(radial_profile[outer_start:data_outer])
            
            # 从外向内找信号边界：找到均值超过 bg_level + 3*bg_std 的位置
            for r_idx in range(data_outer - window_size, 20, -1):
                local_mean = np.mean(radial_profile[r_idx:r_idx+window_size])
                if local_mean > bg_level + 3 * bg_std:
                    bg_inner = r_idx + window_size
                    break
            
            # 背景区域：从信号边界到数据边界
            if verbose:
                print(f"  V3 background region: r={bg_inner}-{data_outer} px (auto-detected)")
                print(f"    Background level: {bg_level:.2f} ± {bg_std:.2f}")
            
            reconstructor.cleaner._bg_mask = (r >= bg_inner) & (r <= data_outer)
            return reconstructor.cleaner._bg_mask
        
        # 替换方法
        reconstructor.cleaner.identify_background_region = custom_identify_background_region
        
        # 禁用椭圆校正
        params, metadata = reconstructor.reconstruct(input_image, enforce_circularity=False)
        elapsed = time.time() - t0
        
        # 过滤掉信号区域外的峰（使用相同的逻辑）
        ny, nx = input_image.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
        r_max = min(cy, ny - cy, cx, nx - cx)
        radial_sum = np.bincount(r.ravel(), weights=input_image.ravel(), minlength=r_max+1)[:r_max+1]
        radial_count = np.bincount(r.ravel(), minlength=r_max+1)[:r_max+1]
        radial_count[radial_count == 0] = 1
        radial_profile = radial_sum / radial_count
        
        # 找到数据边界
        data_outer = r_max
        for r_idx in range(r_max - 1, 0, -1):
            if radial_profile[r_idx] > 0.1:
                data_outer = r_idx
                break
        
        # 找到信号边界
        outer_start = int(data_outer * 0.8)
        bg_level = np.mean(radial_profile[outer_start:data_outer])
        bg_std = np.std(radial_profile[outer_start:data_outer])
        
        signal_radius = data_outer
        window_size = 10
        for r_idx in range(data_outer - window_size, 20, -1):
            local_mean = np.mean(radial_profile[r_idx:r_idx+window_size])
            if local_mean > bg_level + 3 * bg_std:
                signal_radius = r_idx + window_size
                break
        
        # 过滤峰
        params = [p for p in params if p['r'] <= signal_radius]
        
        r_grid = np.arange(n_pixels // 2 + 1)
        
        if verbose:
            print(f"\nTime: {elapsed:.2f}s")
            print(f"Found {len(params)} peaks (after filtering r <= {signal_radius})")
            for i, p in enumerate(params[:10]):
                print(f"  Peak {i+1}: r={p['r']:.1f}px, sigma={p.get('sigma', 0):.2f}px, beta={p.get('beta', 0):.2f}")
        
        # 从参数重建径向分布
        recon_profile = reconstruct_radial_profile(params, len(r_grid))
        
        return params, r_grid, recon_profile, elapsed
    except Exception as e:
        print(f"V3 reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return [], np.arange(n_pixels//2 + 1), np.zeros(n_pixels//2 + 1), 0


def run_rbasex_reconstruction(image, verbose=True):
    """运行 rBasex 重建方法"""
    if verbose:
        print("\n" + "="*60)
        print("rBasex Reconstruction")
        print("="*60)
    
    t0 = time.time()
    
    try:
        # 直接调用 abel.rbasex 获取更多信息
        recon_image, distr = abel.rbasex.rbasex_transform(
            image, 
            direction='inverse', 
            basis_dir=None,
            verbose=verbose
        )
        
        # 提取径向分布和 beta
        r_rb, I_rb, beta_rb = distr.rIbeta()
        
        if verbose:
            print(f"\nrBasex raw output:")
            print(f"  r range: {r_rb[0]:.1f} - {r_rb[-1]:.1f} px")
            print(f"  I range: {I_rb.min():.4f} - {I_rb.max():.4f}")
            print(f"  beta range: {beta_rb.min():.3f} - {beta_rb.max():.3f}")
            
            # 显示中心区域的值
            print(f"\nCenter region (r < 20):")
            for r_idx in range(0, min(20, len(r_rb)), 5):
                print(f"  r={r_rb[r_idx]:.0f}: I={I_rb[r_idx]:.4f}, beta={beta_rb[r_idx]:.3f}")
        
        # 调用原来的函数获取峰参数
        params, metadata = reconstruct_rbasex(image, config=None, verbose=verbose)
        elapsed = time.time() - t0
        
        r_grid = metadata.get('r_grid', np.arange(image.shape[0] // 2 + 1))
        # 从参数重建径向分布（高斯峰叠加）
        recon_profile = reconstruct_radial_profile(params, len(r_grid))
        recon_image = metadata.get('recon_image', None)
        
        if verbose:
            print(f"\nTotal time: {elapsed:.2f}s")
        
        return params, r_grid, recon_profile, elapsed, recon_image
    except Exception as e:
        print(f"rBasex reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        n_pixels = image.shape[0]
        return [], np.arange(n_pixels//2 + 1), np.zeros(n_pixels//2 + 1), 0, None


def reconstruct_2d_image(params, n_pixels):
    """从参数重建2D图像（3D切片）"""
    cy, cx = n_pixels // 2, n_pixels // 2
    y, x = np.ogrid[:n_pixels, :n_pixels]
    Y = y - cy
    X = x - cx
    R = np.sqrt(X**2 + Y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = X / R
    cos_theta[~np.isfinite(cos_theta)] = 0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    image = np.zeros((n_pixels, n_pixels))
    
    for p in params:
        r = p['r']
        sigma = p.get('sigma', p.get('sigma_phys', 3.0))
        amp = p.get('amp', 1.0)
        beta = p.get('beta', 0)
        
        radial = amp * np.exp(-((R - r)**2) / (2 * sigma**2))
        angular = 1 + beta * P2
        image += radial * angular
    
    return image


def save_individual_reconstructions(image, results, image_raw=None, bg_level=None, signal_radius=None, output_dir='.'):
    """保存每种方法的重建图像为单独的文件
    
    Args:
        image: 预处理后的图像
        results: 各方法的重建结果
        image_raw: 原始图像
        bg_level: 背景水平
        signal_radius: 信号区域半径
        output_dir: 输出目录
    """
    import os
    
    n_pixels = image.shape[0]
    
    # 1. 保存原始图像
    if image_raw is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        vmax_raw = np.percentile(image_raw, 99.5)
        im = ax.imshow(image_raw, cmap='hot', vmin=0, vmax=vmax_raw)
        ax.set_title('Original Image (with background)', fontsize=12)
        if signal_radius is not None:
            cy, cx = n_pixels // 2, n_pixels // 2
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(cx + signal_radius*np.cos(theta), cy + signal_radius*np.sin(theta), 
                    'c--', linewidth=2, label=f'Signal boundary r={signal_radius}')
            ax.legend(fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstruction_original.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: reconstruction_original.png")
    
    # 2. 保存预处理后的图像
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.percentile(image, 99.5)
    im = ax.imshow(image, cmap='hot', vmin=0, vmax=vmax)
    title = 'Preprocessed Image'
    if bg_level is not None:
        title += f' (bg={bg_level:.2f} subtracted)'
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_preprocessed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reconstruction_preprocessed.png")
    
    # 3. 保存各方法的重建图像
    for method, result in results.items():
        params = result[0]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if method == 'rBasex' and len(result) > 4 and result[4] is not None:
            # rBasex 有直接的重建图像
            rb_recon = result[4]
            rb_recon_log = np.log1p(np.maximum(rb_recon, 0))
            im = ax.imshow(rb_recon_log, cmap='hot', vmin=0)
            ax.set_title(f'{method} Reconstruction log(1+I)\n({len(params)} peaks)', fontsize=12)
        elif params:
            recon = reconstruct_2d_image(params, n_pixels)
            im = ax.imshow(recon, cmap='hot', vmin=0)
            ax.set_title(f'{method} Reconstruction\n({len(params)} peaks)', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No peaks detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{method} Reconstruction', fontsize=12)
            im = None
        
        if im is not None:
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.tight_layout()
        
        filename = f'reconstruction_{method.lower()}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # 4. 保存径向分布比较图
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'V1': 'blue', 'V2': 'green', 'V3': 'red', 'rBasex': 'purple'}
    mask_center = 50
    
    for method, result in results.items():
        params, r_grid, recon_profile, elapsed = result[:4]
        if len(recon_profile) > 0 and np.max(recon_profile) > 0:
            profile_plot = recon_profile.copy()
            if method in ['V1', 'V2', 'V3']:
                profile_plot[:mask_center] = 0
            profile_norm = profile_plot / np.max(profile_plot) if np.max(profile_plot) > 0 else profile_plot
            ax.plot(r_grid, profile_norm, 
                   label=f'{method} ({elapsed:.2f}s, {len(params)} peaks)', 
                   color=colors.get(method, 'gray'), linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Radius (pixels)', fontsize=11)
    ax.set_ylabel('Normalized Intensity', fontsize=11)
    ax.set_title('Radial Distribution Comparison (3D Space after Abel Inversion)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if signal_radius:
        ax.axvline(x=signal_radius, color='cyan', linestyle='--', alpha=0.5, label='Signal boundary')
    ax.set_xlim(0, n_pixels // 2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_radial_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reconstruction_radial_comparison.png")
    
    # 5. 保存峰位置图
    fig, ax = plt.subplots(figsize=(12, 4))
    method_idx = 0
    for method, result in results.items():
        params = result[0]
        if params:
            r_vals = [p['r'] for p in params]
            ax.scatter(r_vals, [method_idx] * len(r_vals), 
                      s=80, c=colors.get(method, 'gray'), 
                      label=f'{method} ({len(params)})', marker='o', 
                      edgecolors='white', linewidths=0.5, alpha=0.7)
        method_idx += 1
    
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(list(results.keys()))
    ax.set_xlabel('Radius (pixels)', fontsize=11)
    ax.set_title('Detected Peak Positions', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    if signal_radius:
        ax.axvline(x=signal_radius, color='cyan', linestyle='--', alpha=0.5)
    ax.set_xlim(0, n_pixels // 2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_peak_positions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reconstruction_peak_positions.png")
    
    # 6. 保存 Beta 参数图
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, result in results.items():
        params = result[0]
        if params:
            r_vals = [p['r'] for p in params]
            beta_vals = [p.get('beta', 0) for p in params]
            ax.scatter(r_vals, beta_vals, s=60, c=colors.get(method, 'gray'),
                      label=method, marker='o', edgecolors='white', 
                      linewidths=0.5, alpha=0.7)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Radius (pixels)', fontsize=11)
    ax.set_ylabel('Beta Parameter', fontsize=11)
    ax.set_title('Anisotropy Parameter (beta)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2.5, 2.5)
    if signal_radius:
        ax.axvline(x=signal_radius, color='cyan', linestyle='--', alpha=0.5)
    ax.set_xlim(0, n_pixels // 2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_beta_parameters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reconstruction_beta_parameters.png")


def visualize_comparison(image, results, image_raw=None, bg_level=None, signal_radius=None):
    """可视化比较四种方法的结果"""
    
    fig = plt.figure(figsize=(24, 16))
    
    n_pixels = image.shape[0]
    cy, cx = n_pixels // 2, n_pixels // 2
    
    # =========================================================================
    # 第一行：原始图像、预处理后图像、各方法的重建图像
    # =========================================================================
    
    # 原始图像（带信号边界圆）
    ax1 = fig.add_subplot(3, 6, 1)
    if image_raw is not None:
        vmax_raw = np.percentile(image_raw, 99.5)
        im1 = ax1.imshow(image_raw, cmap='hot', vmin=0, vmax=vmax_raw)
        ax1.set_title('Original Image\n(with background)', fontsize=10)
        # 画信号边界圆
        if signal_radius is not None:
            theta = np.linspace(0, 2*np.pi, 100)
            ax1.plot(cx + signal_radius*np.cos(theta), cy + signal_radius*np.sin(theta), 
                    'c--', linewidth=2, label=f'r={signal_radius}')
            ax1.legend(fontsize=8)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
    else:
        ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Original Image', fontsize=10)
    
    # 预处理后的图像
    ax2 = fig.add_subplot(3, 6, 2)
    vmax = np.percentile(image, 99.5)
    im2 = ax2.imshow(image, cmap='hot', vmin=0, vmax=vmax)
    if bg_level is not None:
        ax2.set_title(f'Preprocessed\n(bg={bg_level:.2f} subtracted)', fontsize=10)
    else:
        ax2.set_title('Preprocessed Image', fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # V1 重建图
    ax3 = fig.add_subplot(3, 6, 3)
    v1_params = results['V1'][0]
    if v1_params:
        v1_recon = reconstruct_2d_image(v1_params, n_pixels)
        im3 = ax3.imshow(v1_recon, cmap='hot', vmin=0)
        ax3.set_title(f'V1 Reconstruction\n({len(v1_params)} peaks)', fontsize=10)
        plt.colorbar(im3, ax=ax3, fraction=0.046)
    else:
        ax3.text(0.5, 0.5, 'No peaks', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('V1 Reconstruction', fontsize=10)
    
    # V2 重建图
    ax4 = fig.add_subplot(3, 6, 4)
    v2_params = results['V2'][0]
    if v2_params:
        v2_recon = reconstruct_2d_image(v2_params, n_pixels)
        im4 = ax4.imshow(v2_recon, cmap='hot', vmin=0)
        ax4.set_title(f'V2 Reconstruction\n({len(v2_params)} peaks)', fontsize=10)
        plt.colorbar(im4, ax=ax4, fraction=0.046)
    else:
        ax4.text(0.5, 0.5, 'No peaks', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('V2 Reconstruction', fontsize=10)
    
    # V3 重建图
    ax5 = fig.add_subplot(3, 6, 5)
    v3_params = results['V3'][0]
    if v3_params:
        v3_recon = reconstruct_2d_image(v3_params, n_pixels)
        im5 = ax5.imshow(v3_recon, cmap='hot', vmin=0)
        ax5.set_title(f'V3 Reconstruction\n({len(v3_params)} peaks)', fontsize=10)
        plt.colorbar(im5, ax=ax5, fraction=0.046)
    else:
        ax5.text(0.5, 0.5, 'No peaks', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('V3 Reconstruction', fontsize=10)
    
    # rBasex 重建图 - 使用 log(1+x) scale
    ax6 = fig.add_subplot(3, 6, 6)
    if len(results['rBasex']) > 4 and results['rBasex'][4] is not None:
        rb_recon = results['rBasex'][4]
        # 使用 log(1+x) scale 显示
        rb_recon_log = np.log1p(np.maximum(rb_recon, 0))  # log(1+x)
        im6 = ax6.imshow(rb_recon_log, cmap='hot', vmin=0)
        ax6.set_title(f'rBasex log(1+I)\n({len(results["rBasex"][0])} peaks)', fontsize=10)
        plt.colorbar(im6, ax=ax6, fraction=0.046)
    else:
        rb_params = results['rBasex'][0]
        if rb_params:
            rb_recon = reconstruct_2d_image(rb_params, n_pixels)
            rb_recon_log = np.log1p(np.maximum(rb_recon, 0))
            im6 = ax6.imshow(rb_recon_log, cmap='hot', vmin=0)
            ax6.set_title(f'rBasex log(1+I)\n({len(rb_params)} peaks)', fontsize=10)
            plt.colorbar(im6, ax=ax6, fraction=0.046)
        else:
            ax6.text(0.5, 0.5, 'No peaks', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('rBasex Reconstruction', fontsize=10)
    
    # =========================================================================
    # 第二行：径向分布比较
    # =========================================================================
    
    ax_radial = fig.add_subplot(3, 1, 2)
    colors = {'V1': 'blue', 'V2': 'green', 'V3': 'red', 'rBasex': 'purple'}
    
    mask_center = 50  # 中心 25 像素置为 0 (256 尺度)
    
    for method, result in results.items():
        params, r_grid, recon_profile, elapsed = result[:4]
        if len(recon_profile) > 0 and np.max(recon_profile) > 0:
            profile_plot = recon_profile.copy()
            # V1, V2, V3 的中心 20px 置为 0
            if method in ['V1', 'V2', 'V3']:
                profile_plot[:mask_center] = 0
            profile_norm = profile_plot / np.max(profile_plot) if np.max(profile_plot) > 0 else profile_plot
            ax_radial.plot(r_grid, profile_norm, 
                          label=f'{method} ({elapsed:.2f}s, {len(params)} peaks)', 
                          color=colors.get(method, 'gray'), linewidth=1.5, alpha=0.8)
    
    ax_radial.set_xlabel('Radius (pixels)', fontsize=11)
    ax_radial.set_ylabel('Normalized Intensity', fontsize=11)
    ax_radial.set_title('Radial Distribution Comparison (3D Space after Abel Inversion)', fontsize=12)
    ax_radial.legend(loc='upper right')
    ax_radial.grid(True, alpha=0.3)
    if signal_radius:
        ax_radial.axvline(x=signal_radius, color='cyan', linestyle='--', alpha=0.5, label='Signal boundary')
    ax_radial.set_xlim(0, n_pixels // 2)
    
    # =========================================================================
    # 第三行：峰位置和Beta参数
    # =========================================================================
    
    ax_peaks = fig.add_subplot(3, 2, 5)
    method_idx = 0
    for method, result in results.items():
        params = result[0]
        if params:
            r_vals = [p['r'] for p in params]
            ax_peaks.scatter(r_vals, [method_idx] * len(r_vals), 
                           s=80, c=colors.get(method, 'gray'), 
                           label=f'{method} ({len(params)})', marker='o', 
                           edgecolors='white', linewidths=0.5, alpha=0.7)
        method_idx += 1
    
    ax_peaks.set_yticks(range(len(results)))
    ax_peaks.set_yticklabels(list(results.keys()))
    ax_peaks.set_xlabel('Radius (pixels)', fontsize=11)
    ax_peaks.set_title('Detected Peak Positions', fontsize=12)
    ax_peaks.legend(loc='upper right')
    ax_peaks.grid(True, alpha=0.3, axis='x')
    if signal_radius:
        ax_peaks.axvline(x=signal_radius, color='cyan', linestyle='--', alpha=0.5)
    ax_peaks.set_xlim(0, n_pixels // 2)
    
    ax_beta = fig.add_subplot(3, 2, 6)
    for method, result in results.items():
        params = result[0]
        if params:
            r_vals = [p['r'] for p in params]
            beta_vals = [p.get('beta', 0) for p in params]
            ax_beta.scatter(r_vals, beta_vals, s=60, c=colors.get(method, 'gray'),
                          label=method, marker='o', edgecolors='white', 
                          linewidths=0.5, alpha=0.7)
    
    ax_beta.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_beta.axhline(y=2, color='gray', linestyle=':', alpha=0.3)
    ax_beta.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)
    ax_beta.set_xlabel('Radius (pixels)', fontsize=11)
    ax_beta.set_ylabel('Beta Parameter', fontsize=11)
    ax_beta.set_title('Anisotropy Parameter (beta)', fontsize=12)
    ax_beta.legend(loc='upper right')
    ax_beta.grid(True, alpha=0.3)
    ax_beta.set_ylim(-2.5, 2.5)
    if signal_radius:
        ax_beta.axvline(x=signal_radius, color='cyan', linestyle='--', alpha=0.5)
    ax_beta.set_xlim(0, n_pixels // 2)
    
    plt.suptitle('electron_shilpa_XY.mat - Reconstruction Method Comparison\n(Background subtracted from entire image)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reconstruction_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: reconstruction_comparison_all.png")


def plot_cross_sections(image_raw, image_processed, bg_mean, signal_radius, bg_r_inner=None, bg_r_outer=None):
    """绘制原始图像和预处理后图像的横截面图
    
    显示水平和垂直方向的横截面，以及径向分布。
    标记出用于估计背景的环形区域。
    
    Args:
        bg_r_inner: 背景区域内边界，默认为 signal_radius
        bg_r_outer: 背景区域外边界，默认为 n//2 * 0.9
    """
    n = image_raw.shape[0]
    cy, cx = n // 2, n // 2
    
    # 自动计算背景区域边界（如果未指定）
    if bg_r_inner is None:
        bg_r_inner = signal_radius
    if bg_r_outer is None:
        bg_r_outer = int(n // 2 * 0.9)  # 外边界为半径的 90%
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # =========================================================================
    # 第一行：原始图像的横截面
    # =========================================================================
    
    # 水平横截面 (y = center)
    ax1 = axes[0, 0]
    h_cut_raw = image_raw[cy, :]
    x_coords = np.arange(n) - cx
    ax1.plot(x_coords, h_cut_raw, 'b-', linewidth=1, label='Horizontal cut')
    ax1.axhline(y=bg_mean, color='r', linestyle='--', linewidth=2, label=f'BG mean={bg_mean:.3f}')
    # 标记背景估计区域
    ax1.axvspan(-bg_r_outer, -bg_r_inner, alpha=0.2, color='yellow', label=f'BG region ({bg_r_inner}-{bg_r_outer})')
    ax1.axvspan(bg_r_inner, bg_r_outer, alpha=0.2, color='yellow')
    ax1.axvline(x=-signal_radius, color='c', linestyle=':', alpha=0.7)
    ax1.axvline(x=signal_radius, color='c', linestyle=':', alpha=0.7, label=f'Signal boundary r={signal_radius}')
    ax1.set_xlabel('X (pixels from center)')
    ax1.set_ylabel('Counts')
    ax1.set_title('Original Image - Horizontal Cross Section (y=center)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-n//2, n//2)
    
    # 垂直横截面 (x = center)
    ax2 = axes[0, 1]
    v_cut_raw = image_raw[:, cx]
    y_coords = np.arange(n) - cy
    ax2.plot(y_coords, v_cut_raw, 'g-', linewidth=1, label='Vertical cut')
    ax2.axhline(y=bg_mean, color='r', linestyle='--', linewidth=2, label=f'BG mean={bg_mean:.3f}')
    ax2.axvspan(-bg_r_outer, -bg_r_inner, alpha=0.2, color='yellow', label=f'BG region')
    ax2.axvspan(bg_r_inner, bg_r_outer, alpha=0.2, color='yellow')
    ax2.axvline(x=-signal_radius, color='c', linestyle=':', alpha=0.7)
    ax2.axvline(x=signal_radius, color='c', linestyle=':', alpha=0.7, label=f'Signal boundary r={signal_radius}')
    ax2.set_xlabel('Y (pixels from center)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Original Image - Vertical Cross Section (x=center)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-n//2, n//2)
    
    # 径向分布
    ax3 = axes[0, 2]
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    r_max = n // 2
    radial_sum = np.bincount(r.ravel(), weights=image_raw.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count = np.bincount(r.ravel(), minlength=r_max+1)[:r_max+1]
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count
    
    r_grid = np.arange(len(radial_profile))
    ax3.plot(r_grid, radial_profile, 'm-', linewidth=1, label='Radial profile')
    ax3.axhline(y=bg_mean, color='r', linestyle='--', linewidth=2, label=f'BG mean={bg_mean:.3f}')
    ax3.axvspan(bg_r_inner, bg_r_outer, alpha=0.2, color='yellow', label=f'BG region ({bg_r_inner}-{bg_r_outer})')
    ax3.axvline(x=signal_radius, color='c', linestyle=':', alpha=0.7, label=f'Signal boundary r={signal_radius}')
    ax3.set_xlabel('Radius (pixels)')
    ax3.set_ylabel('Mean counts')
    ax3.set_title('Original Image - Radial Profile')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, r_max)
    
    # =========================================================================
    # 第二行：预处理后图像的横截面
    # =========================================================================
    
    # 水平横截面 (y = center)
    ax4 = axes[1, 0]
    h_cut_proc = image_processed[cy, :]
    ax4.plot(x_coords, h_cut_proc, 'b-', linewidth=1, label='Horizontal cut')
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero level')
    ax4.axvspan(-bg_r_outer, -bg_r_inner, alpha=0.2, color='yellow', label=f'BG region')
    ax4.axvspan(bg_r_inner, bg_r_outer, alpha=0.2, color='yellow')
    ax4.axvline(x=-signal_radius, color='c', linestyle=':', alpha=0.7)
    ax4.axvline(x=signal_radius, color='c', linestyle=':', alpha=0.7, label=f'Signal boundary r={signal_radius}')
    ax4.set_xlabel('X (pixels from center)')
    ax4.set_ylabel('Counts')
    ax4.set_title(f'Preprocessed Image - Horizontal Cross Section\n(BG mean {bg_mean:.3f} subtracted)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-n//2, n//2)
    
    # 垂直横截面 (x = center)
    ax5 = axes[1, 1]
    v_cut_proc = image_processed[:, cx]
    ax5.plot(y_coords, v_cut_proc, 'g-', linewidth=1, label='Vertical cut')
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero level')
    ax5.axvspan(-bg_r_outer, -bg_r_inner, alpha=0.2, color='yellow', label=f'BG region')
    ax5.axvspan(bg_r_inner, bg_r_outer, alpha=0.2, color='yellow')
    ax5.axvline(x=-signal_radius, color='c', linestyle=':', alpha=0.7)
    ax5.axvline(x=signal_radius, color='c', linestyle=':', alpha=0.7, label=f'Signal boundary r={signal_radius}')
    ax5.set_xlabel('Y (pixels from center)')
    ax5.set_ylabel('Counts')
    ax5.set_title(f'Preprocessed Image - Vertical Cross Section\n(BG mean {bg_mean:.3f} subtracted)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-n//2, n//2)
    
    # 径向分布
    ax6 = axes[1, 2]
    radial_sum_proc = np.bincount(r.ravel(), weights=image_processed.ravel(), minlength=r_max+1)[:r_max+1]
    radial_profile_proc = radial_sum_proc / radial_count
    
    ax6.plot(r_grid, radial_profile_proc, 'm-', linewidth=1, label='Radial profile')
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero level')
    ax6.axvspan(bg_r_inner, bg_r_outer, alpha=0.2, color='yellow', label=f'BG region')
    ax6.axvline(x=signal_radius, color='c', linestyle=':', alpha=0.7, label=f'Signal boundary r={signal_radius}')
    ax6.set_xlabel('Radius (pixels)')
    ax6.set_ylabel('Mean counts')
    ax6.set_title(f'Preprocessed Image - Radial Profile\n(BG mean {bg_mean:.3f} subtracted)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, r_max)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Cross-section analysis:")
    print("="*60)
    
    # 背景区域统计（bg_r_inner < r < bg_r_outer）
    bg_radial = radial_profile[bg_r_inner:bg_r_outer+1]
    
    print(f"\nBackground region ({bg_r_inner} < r < {bg_r_outer}) statistics:")
    print(f"  Radial profile in BG region: mean={np.mean(bg_radial):.4f}, std={np.std(bg_radial):.4f}")
    print(f"  Estimated BG mean (used for subtraction): {bg_mean:.4f}")
    
    # 信号区域统计
    inner_r = radial_profile[:signal_radius]
    print(f"\nSignal region (r < {signal_radius}) statistics:")
    print(f"  Radial profile: mean={np.mean(inner_r):.4f}, max={np.max(inner_r):.1f}")
    
    plt.suptitle('Cross-Section Analysis: Original vs Preprocessed Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reconstruction_cross_sections.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: reconstruction_cross_sections.png")


def main():
    """主函数"""
    # ========== 配置选项 ==========
    USE_NORMALIZATION = False  # 是否使用99分位数归一化
    
    # V2 峰检测参数（可调整）
    V2_HEIGHT_THRESH = 0.08      # 峰高度阈值（相对于最大值），默认 8%
    V2_PROMINENCE_THRESH = 0.05  # 峰突出度阈值（相对于最大值），默认 5%
    V2_DISTANCE = 12             # 相邻峰最小距离（像素），默认 12
    
    # 中心遮罩半径
    MASK_RADIUS = 25             # 排除 r < 25 px 的中心区域
    # ==============================
    
    print("="*60)
    print("Loading electron_shilpa_XY.mat (raw XY data)")
    print("="*60)
    
    # 加载原始XY数据
    xy_data = load_raw_data('electron_shilpa_XY.mat')
    
    # 构建VMI图像
    print("\n" + "="*60)
    print("Building VMI image from XY coordinates")
    print("="*60)
    image_raw, pixel_size = build_vmi_image(xy_data, n_bins=512, center=True)
    
    # 估计背景噪声（从外围纯背景区域）
    print("\n" + "="*60)
    print("Estimating background noise from outer region")
    print("="*60)
    bg_mean, bg_std, signal_radius, radial_profile = estimate_background_and_signal_region(image_raw)
    
    # 预处理：从整个图像减去背景均值，背景区域设为0
    print("\n" + "="*60)
    print(f"Preprocessing: subtract background mean, set BG region to 0 (normalize={USE_NORMALIZATION})")
    print("="*60)
    image = preprocess_image(image_raw, bg_mean, bg_std, signal_radius, normalize=USE_NORMALIZATION)
    
    # 先显示横截面图，让用户验证预处理是否正确
    print("\n" + "="*60)
    print("Showing cross-section analysis...")
    print("="*60)
    plot_cross_sections(image_raw, image, bg_mean, signal_radius)
    
    # 存储结果
    results = {}
    
    print("\n" + "="*60)
    print(f"Starting reconstruction (mask_radius={MASK_RADIUS} px)...")
    print("="*60)
    
    # V1
    v1_result = run_v1_reconstruction(image, mask_radius=MASK_RADIUS)
    results['V1'] = v1_result
    
    # V2 - 传递原始图像让 V2 自己做背景估计，使用可配置的峰检测参数
    v2_result = run_v2_reconstruction(
        image, image_raw=image_raw, mask_radius=MASK_RADIUS,
        v2_height_thresh=V2_HEIGHT_THRESH,
        v2_prominence_thresh=V2_PROMINENCE_THRESH,
        v2_distance=V2_DISTANCE
    )
    results['V2'] = v2_result
    
    # V3 - 传递原始图像让 V3 自己做背景估计
    v3_result = run_v3_reconstruction(image, image_raw=image_raw, mask_radius=MASK_RADIUS)
    results['V3'] = v3_result
    
    # rBasex
    rb_result = run_rbasex_reconstruction(image)
    results['rBasex'] = rb_result
    
    # 可视化比较（同时显示原始图像和预处理后的图像）
    print("\nGenerating comparison figure...")
    visualize_comparison(image, results, image_raw, bg_mean, signal_radius)
    
    # 保存各方法的重建图像为单独文件
    print("\nSaving individual reconstruction images...")
    save_individual_reconstructions(image, results, image_raw, bg_mean, signal_radius)
    
    # 打印总结
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)
    print(f"\nBackground noise: mean={bg_mean:.4f}, std={bg_std:.4f} counts/pixel")
    
    for method, result in results.items():
        params, r_grid, recon_profile, elapsed = result[:4]
        print(f"\n{method}:")
        print(f"  Peaks found: {len(params)}")
        print(f"  Time: {elapsed:.2f}s")
        if params:
            sorted_params = sorted(params, key=lambda x: x.get('amp', 0), reverse=True)
            print(f"  Top peaks (by amplitude):")
            for i, p in enumerate(sorted_params[:5]):
                sigma = p.get('sigma', p.get('sigma_phys', 0))
                beta = p.get('beta', 0)
                amp = p.get('amp', 0)
                print(f"    {i+1}: r={p['r']:.1f}px, sigma={sigma:.2f}px, beta={beta:.2f}, amp={amp:.4f}")


if __name__ == '__main__':
    main()
