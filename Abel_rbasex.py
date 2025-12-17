import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve
# 显式导入需要的底层模块，防止作用域错误
import abel.hansenlaw 

# ==========================================
# 1. 高级 B-Spline 求解器 (核心算法)
# ==========================================
class FastBSplineAbel:
    """
    改进版 B-Spline Abel 反演器。
    采用混合正则化 (Hybrid Regularization) 解决 FWHM 展宽和背景抖动问题。
    """
    def __init__(self, n_pixels, n_basis=300, reg_smooth=10.0, reg_ridge=0.01):
        """
        初始化并预计算矩阵。
        """
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid = np.arange(self.radius + 1)
        
        # 1. 定义 B-Spline 节点 (Knots) - 3阶 (Cubic)
        self.knots = np.linspace(0, self.radius, n_basis - 2)
        self.knots = np.r_[[0]*3, self.knots, [self.radius]*3]
        self.n_basis = len(self.knots) - 4
        
        print(f"   [Custom] Init Solver: Basis={self.n_basis}, Smooth={reg_smooth}, Ridge={reg_ridge}")
        
        # 2. 构建投影矩阵 A (Forward Matrix)
        self.A = np.zeros((self.radius + 1, self.n_basis))
        fine_r = np.arange(self.radius + 1, dtype=float)
        
        # 计算每个基函数的 Abel 投影
        for i in range(self.n_basis):
            c = np.zeros(self.n_basis)
            c[i] = 1.0
            bs = BSpline(self.knots, c, k=3, extrapolate=False)
            b_vals = bs(fine_r)
            b_vals[np.isnan(b_vals)] = 0
            
            # 使用 HansenLaw 计算基函数的正向投影 (支持 1D)
            # 这里直接调用全局导入的 abel.hansenlaw
            proj = abel.hansenlaw.hansenlaw_transform(b_vals, direction='forward', dr=1)
                
            self.A[:, i] = proj

        # 3. 构建正则化矩阵
        # L: 二阶导数矩阵 (拉普拉斯算子)，惩罚曲线弯曲 -> 保证光滑
        self.L = np.zeros((self.n_basis, self.n_basis))
        for i in range(1, self.n_basis - 1):
            self.L[i, i-1] = 1
            self.L[i, i] = -2
            self.L[i, i+1] = 1
        self.L[0, 0] = 1; self.L[-1, -1] = 1 # 边界
        
        # I: 单位矩阵，惩罚系数大小 -> 压制背景噪声
        self.I = np.eye(self.n_basis)
            
        # 4. 预计算逆矩阵 K (混合正则化核心公式)
        # Formula: x = (A'A + λ_smooth*L'L + λ_ridge*I)^-1 A'b
        ATA = self.A.T @ self.A
        LTL = self.L.T @ self.L
        
        # 求解线性方程组
        Matrix_Left = ATA + (reg_smooth * LTL) + (reg_ridge * self.I)
        self.K = np.linalg.solve(Matrix_Left, self.A.T)

    def solve(self, projected_profile):
        """
        输入: 1D 投影分布 (NumPy array)
        输出: (r_grid, intensity_profile)
        """
        # 1. 极速矩阵乘法
        coeffs = self.K @ projected_profile.astype(float)
        
        # 2. B-Spline 重建
        bs = BSpline(self.knots, coeffs, k=3, extrapolate=False)
        recon_profile = bs(self.r_grid)
        
        # 3. 物理约束: 强度非负 (解决残留的微小负值震荡)
        recon_profile = np.nan_to_num(recon_profile)
        recon_profile = np.clip(recon_profile, 0, None)
        
        return self.r_grid, recon_profile

# ==========================================
# 2. 模拟与辅助函数
# ==========================================

def generate_ground_truth_slice(r_peak, width, intensity, r_max, N):
    """生成真实的径向分布 (Ground Truth)"""
    r = np.linspace(0, r_max, N//2 + 1)
    f_r = intensity * np.exp(-((r - r_peak)**2) / (2 * width**2))
    return r, f_r

def create_simulated_image(r, f_r, N, noise_level=0):
    """
    生成模拟实验投影图像:
    3D Sphere Slice -> Forward Abel -> 2D Image -> Poisson Noise
    """
    center = N // 2
    y, x = np.ogrid[-center:N-center, -center:N-center]
    r_grid = np.sqrt(x**2 + y**2)
    # 1. 3D 密度切片
    density_slice = np.interp(r_grid, r, f_r, left=0, right=0)
    
    # 2. 正向投影 (模拟相机)
    # 这里我们确保 density_slice 是 2D 数组
    forward_transform = abel.Transform(density_slice, method='hansenlaw', direction='forward', verbose=False)
    projection_image = forward_transform.transform
    
    # 3. 添加噪声
    if noise_level > 0:
        np.random.seed(42)
        scale = 1.0 / noise_level
        projection_image = np.random.poisson(projection_image * scale) / scale
        
    return density_slice, projection_image

def find_peak_params(r, intensity):
    """精确计算峰位和 FWHM"""
    peak_idx = np.argmax(intensity)
    peak_val = intensity[peak_idx]
    
    # 1. 重心法求峰位
    window = 15
    start = max(0, peak_idx - window)
    end = min(len(r), peak_idx + window)
    if np.sum(intensity[start:end]) == 0: return 0, 0
    peak_pos = np.sum(r[start:end] * intensity[start:end]) / np.sum(intensity[start:end])
    
    # 2. 插值法求 FWHM
    half_max = peak_val / 2.0
    # 寻找穿过半高值的点
    diffs = np.sign(intensity - half_max)
    crossings = np.where(np.diff(diffs))[0]
    
    if len(crossings) >= 2:
        # 找最左和最右的交点 (应对可能的中间抖动)
        left_idx = crossings[0]
        right_idx = crossings[-1]
        
        # 线性插值
        def get_x(idx):
            y1, y2 = intensity[idx], intensity[idx+1]
            x1, x2 = r[idx], r[idx+1]
            return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1 + 1e-10)
            
        fwhm = get_x(right_idx) - get_x(left_idx)
    else:
        fwhm = 0
        
    return peak_pos, fwhm

# ==========================================
# 3. 主程序对比
# ==========================================

def main():
    # --- 实验参数设置 ---
    N = 501           # 图像尺寸
    r_peak = 200.0    # 峰位置
    width = 20.0       # 峰宽 (Sigma)，FWHM ≈ 7.0

    noise_lvl = 0.005 # 噪声水平 (模拟中等信噪比)
    
    # --- 算法参数调优 (关键) ---
    # reg_smooth: 设为 10.0 (较小)，保证峰不被磨平，FWHM 准
    # reg_ridge:  设为 0.05 (适中)，压制平坦区抖动
    custom_basis = 300
    custom_smooth = 10.0
    custom_ridge = 0.05
    
    print("=== Advanced Abel Inversion Benchmark ===")
    print(f"Target: Peak @ {r_peak}, Sigma={width} (Expected FWHM={2.355*width:.2f})")
    
    # 1. 生成数据
    r_true, f_true = generate_ground_truth_slice(r_peak, width, 1.0, N//2, N)
    _, proj_image = create_simulated_image(r_true, f_true, N, noise_level=noise_lvl)
    proj_profile_1d = proj_image[N//2, N//2:] # 取中心切片作为输入
    
    # 2. rBasex (Baseline)
    print("\n[Running] rBasex...")
    t0 = time.time()
    res_rbasex = abel.Transform(proj_image, method='rbasex', direction='inverse', verbose=False)
    # 获取径向分布
    r_rb, I_rb = abel.tools.vmi.angular_integration_3D(res_rbasex.transform, origin=(N//2, N//2), dr=1)
    t_rbasex = time.time() - t0
    I_rb /= np.max(I_rb)

    # 3. Custom Hybrid B-Spline
    print("\n[Running] Custom Hybrid B-Spline...")
    t_init_s = time.time()
    # 初始化求解器
    solver = FastBSplineAbel(N, n_basis=custom_basis, reg_smooth=custom_smooth, reg_ridge=custom_ridge)
    t_init = time.time() - t_init_s
    
    t_solve_s = time.time()
    # 求解
    r_my, I_my = solver.solve(proj_profile_1d)
    t_solve = time.time() - t_solve_s
    I_my /= np.max(I_my)
    
    # 4. 指标计算
    gt_pos, gt_fwhm = find_peak_params(r_true, f_true)
    rb_pos, rb_fwhm = find_peak_params(r_rb, I_rb)
    my_pos, my_fwhm = find_peak_params(r_my, I_my)
    
    # 5. 打印结果
    print("\n" + "="*75)
    print(f"{'Metric':<15} | {'Ground Truth':<12} | {'rBasex':<12} | {'Custom (Hybrid)':<15}")
    print("-" * 75)
    print(f"{'Peak Pos':<15} | {gt_pos:<12.2f} | {rb_pos:<12.2f} | {my_pos:<12.2f}")
    print(f"{'Error (px)':<15} | {'-':<12} | {abs(rb_pos-gt_pos):<12.3f} | {abs(my_pos-gt_pos):<12.3f}")
    print("-" * 75)
    print(f"{'FWHM':<15} | {gt_fwhm:<12.2f} | {rb_fwhm:<12.2f} | {my_fwhm:<12.2f}")
    print(f"{'Diff FWHM':<15} | {'-':<12} | {abs(rb_fwhm-gt_fwhm):<12.3f} | {abs(my_fwhm-gt_fwhm):<12.3f}")
    print("-" * 75)
    print(f"{'Speed (Solve)':<15} | {'-':<12} | {t_rbasex*1000:<12.1f} ms | {t_solve*1000:<12.3f} ms (FAST!)")
    print("="*75)
    
    # 6. 绘图
    plt.figure(figsize=(12, 10))
    
    # A. 全谱图
    plt.subplot(2, 2, 1)
    plt.plot(r_true, f_true/np.max(f_true), 'k', alpha=0.3, linewidth=3, label='Truth')
    plt.plot(r_rb, I_rb, 'b-', alpha=0.6, linewidth=1, label='rBasex')
    plt.plot(r_my, I_my, 'r-', linewidth=1.5, label='Custom Hybrid')
    plt.title("Full Radial Distribution")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # B. 背景噪声特写 (Flat Region)
    plt.subplot(2, 2, 2)
    # 取一个远离峰的区域观察噪声
    bg_start = int(r_peak + 50)
    bg_end = int(r_peak + 100)
    plt.plot(r_true[bg_start:bg_end], np.zeros_like(r_true[bg_start:bg_end]), 'k--', label='Truth (0)')
    plt.plot(r_rb[bg_start:bg_end], I_rb[bg_start:bg_end], 'b.-', label='rBasex')
    plt.plot(r_my[bg_start:bg_end], I_my[bg_start:bg_end], 'r.-', label='Custom Hybrid')
    plt.title("Baseline Noise (Zoom-in)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # C. 峰形特写 (FWHM Check)
    plt.subplot(2, 2, 3)
    plt.plot(r_true, f_true/np.max(f_true), 'k', alpha=0.2, linewidth=5, label='Truth')
    plt.plot(r_rb, I_rb, 'b.-', label='rBasex')
    plt.plot(r_my, I_my, 'r.-', label='Custom Hybrid')
    plt.xlim(r_peak - width*4, r_peak + width*4)
    plt.title(f"Peak Quality (Target FWHM: {gt_fwhm:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # D. 模拟图像
    plt.subplot(2, 2, 4)
    plt.imshow(proj_image, cmap='gray')
    plt.title("Simulated Projection (Input)")
    plt.colorbar(shrink=0.8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# only physical forward fitting
#-----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import abel
import abel.hansenlaw
from scipy.optimize import curve_fit
import time

# ==========================================
# 1. 定义物理模型 (核心思想)
# ==========================================
def physical_model_projection(r_coords, r0, sigma, amplitude, background=0):
    """
    这是一个"黑盒"函数：
    输入：物理参数 (位置, 宽度, 强度)
    过程：生成 3D 壳层 -> 做前向 Abel 投影
    输出：模拟的投影数据
    """
    # 1. 生成 1D 径向分布 (假设是高斯峰)
    # 注意：这里 r_coords 是投影后的 x 轴坐标
    # 我们需要在同样的网格上定义 r
    
    # 构造径向分布 f(r)
    # 稍微延伸一下 r 轴以防边缘截断，虽然 hansenlaw 支持同长度
    f_r = amplitude * np.exp(-((r_coords - r0)**2) / (2 * sigma**2))
    
    # 2. 关键步骤：前向投影 (Forward Abel)
    # 将物理的 f(r) 变成相机看到的 projection(x)
    # hansenlaw 非常快，适合放在拟合循环里
    proj = abel.hansenlaw.hansenlaw_transform(f_r, direction='forward', dr=1)
    
    # 3. 加上背景 (可选)
    proj = proj + background
    
    return proj

# ==========================================
# 2. 模拟实验数据
# ==========================================
def generate_experiment_data(N=501):
    r_coords = np.arange(N//2 + 1, dtype=float)
    
    # 真实的物理参数
    true_params = [150.0, 4.0, 1000.0] # r0, sigma, amp
    
    # 生成无噪投影
    clean_proj = physical_model_projection(r_coords, *true_params)
    
    # 加噪声 (这是实验的本质)
    np.random.seed(42)
    # 泊松噪声
    noisy_proj = np.random.poisson(clean_proj)
    
    return r_coords, noisy_proj, true_params

# ==========================================
# 3. 执行拟合 (Forward Fitting)
# ==========================================
def main():
    N = 501
    x, y_exp, true_params = generate_experiment_data(N)
    
    print("=== Forward Fitting Approach ===")
    print("Instead of inverting the noise, we fit a clean physical model to it.")
    
    # 1. 初始猜测 (Guess)
    # 你可以写一个简单的逻辑来猜，比如 argmax
    peak_idx = np.argmax(y_exp)
    guess_r0 = x[peak_idx]-40

    guess_sigma = 5.0
    guess_amp = np.max(y_exp) # 注意：投影的峰值不等于 radial 的峰值，但作为一个 guess 够用了
    p0 = [guess_r0, guess_sigma, guess_amp, 0]
    
    print(f"Initial Guess: r0={guess_r0}, sigma={guess_sigma}")
    
    # 2. 调用优化器 (scipy.optimize.curve_fit)
    # 它会自动调节 r0, sigma, amp 来让 physical_model_projection(x) 逼近 y_exp
    t0 = time.time()
    
    # bounds: 设置参数范围 (r0>0, sigma>0, amp>0) 防止拟合出非物理值
    popt, pcov = curve_fit(
        physical_model_projection, 
        x, 
        y_exp, 
        p0=p0,
        bounds=([0, 0.1, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]) 
    )
    
    t_fit = time.time() - t0
    
    # 3. 解析结果
    fit_r0, fit_sigma, fit_amp, fit_bg = popt
    perr = np.sqrt(np.diag(pcov)) # 参数误差
    
    print(f"\nOptimization Finished in {t_fit*1000:.2f} ms")
    print("-" * 40)
    print(f"{'Param':<10} | {'True':<10} | {'Fitted':<10} | {'Error':<10}")
    print("-" * 40)
    print(f"{'r0':<10} | {true_params[0]:<10.2f} | {fit_r0:<10.2f} | {abs(fit_r0-true_params[0]):<10.4f}")
    print(f"{'sigma':<10} | {true_params[1]:<10.2f} | {fit_sigma:<10.2f} | {abs(fit_sigma-true_params[1]):<10.4f}")
    print(f"{'amp':<10} | {true_params[2]:<10.2f} | {fit_amp:<10.2f} | {abs(fit_amp-true_params[2]):<10.4f}")
    print("-" * 40)
    
    # 4. 重建“纯净”的径向分布
    # 既然我们拿到了最佳的 r0, sigma，我们直接画出那个高斯函数即可
    # 这就是你想要的“无毛刺、无噪声”的分布
    fit_curve_proj = physical_model_projection(x, *popt)
    
    # 重建真实的 I(r)
    # 注意：这里不需要做反演了！因为我们已经fit出了 r0 和 sigma
    # 直接用解析公式画出来就行
    r_recon = x
    I_recon = fit_amp * np.exp(-((r_recon - fit_r0)**2) / (2 * fit_sigma**2))
    
    # 5. 绘图
    plt.figure(figsize=(10, 8))
    
    # 图 1: 拟合效果 (投影域)
    plt.subplot(2, 1, 1)
    plt.scatter(x, y_exp, s=5, c='k', alpha=0.3, label='Experimental Projection (Noisy)')
    plt.plot(x, fit_curve_proj, 'r-', linewidth=2, label='Fitted Model Projection')
    plt.title("Step 1: Forward Fit in Projection Domain")
    plt.legend()
    
    # 图 2: 最终结果 (径向域)
    plt.subplot(2, 1, 2)
    # 画 Ground Truth (为了对比)
    I_true = true_params[2] * np.exp(-((r_recon - true_params[0])**2) / (2 * true_params[1]**2))
    plt.plot(r_recon, I_true, 'k--', linewidth=3, alpha=0.3, label='Ground Truth')
    plt.plot(r_recon, I_recon, 'b-', label='Recovered Distribution (Analytic)')
    plt.title(f"Step 2: Recovered I(r) (r0={fit_r0:.2f}, FWHM={2.355*fit_sigma:.2f})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# an improved multi-peak fitting approach
#-----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy import signal
from scipy.optimize import least_squares
import abel.hansenlaw 

# ==========================================
# 核心算法: Precision Multi-Peak Fitter
# ==========================================
class PrecisionMultiPeakFitter:
    def __init__(self, n_pixels):
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid = np.arange(self.radius + 1, dtype=float)
        
    def _forward_model_one_peak(self, r_axis, r0, sigma, amp):
        if amp <= 1e-9: return np.zeros_like(r_axis)
        f_r = amp * np.exp(-((r_axis - r0)**2) / (2 * sigma**2))
        proj = abel.hansenlaw.hansenlaw_transform(f_r, direction='forward', dr=1)
        return proj

    def _residual_single_with_bg(self, params, x_sub, y_sub):
        r0, sig, amp, bg = params
        signal = self._forward_model_one_peak(x_sub, r0, sig, amp)
        if len(signal) > len(x_sub): signal = signal[:len(x_sub)]
        elif len(signal) < len(x_sub): signal = np.pad(signal, (0, len(x_sub)-len(signal)))
        model = signal + bg
        return model - y_sub

    def _residual_global(self, params, x_data, y_data, n_peaks):
        model = np.zeros_like(x_data, dtype=float)
        for i in range(n_peaks):
            r0 = params[i*3]
            sig = params[i*3+1]
            amp = params[i*3+2]
            try:
                proj = self._forward_model_one_peak(x_data, r0, sig, amp)
                if len(proj) > len(x_data): proj = proj[:len(x_data)]
                elif len(proj) < len(x_data): proj = np.pad(proj, (0, len(x_data)-len(proj)))
                model += proj
            except: pass
        return model - y_data

    def _verify_peak_locally(self, data, r_guess):
        window_half = max(15, int(400 / (r_guess + 1))) 
        idx_start = max(0, int(r_guess) - window_half)
        idx_end = min(len(data), int(r_guess) + window_half)
        
        x_sub = self.r_grid[idx_start:idx_end]
        y_sub = data[idx_start:idx_end]
        
        bg_guess = np.min(y_sub)
        proj_height = np.max(y_sub) - bg_guess
        sig_guess = max(2.0, 300.0/(r_guess+1))
        amp_guess = proj_height / (sig_guess * 2.5) 
        
        p0 = [r_guess, sig_guess, amp_guess, bg_guess]
        lb = [x_sub[0], 0.8, 0.0, 0.0]
        ub = [x_sub[-1], 60.0, np.inf, np.max(y_sub)*1.5]
        
        try:
            res = least_squares(
                self._residual_single_with_bg, p0, args=(x_sub, y_sub),
                bounds=(lb, ub), ftol=1e-4
            )
            r_fit, sig_fit, amp_fit, bg_fit = res.x
            
            if abs(r_fit - r_guess) > window_half * 0.9: return None 
            if sig_fit < 0.9 or sig_fit > 50.0: return None 
            
            proj_h = amp_fit * sig_fit * 2.5
            # 这里的阈值取决于是否扣除了背景，如果是纯信号，2%是合理的
            if proj_h < (np.max(data) - np.min(data)) * 0.02: return None 
            
            return [r_fit, sig_fit, amp_fit]
        except:
            return None

    def solve(self, projected_profile):
        x_axis = self.r_grid
        data = projected_profile.astype(float)
        
        # 1. CWT
        cwt_widths = np.linspace(2, 25, 30)
        peak_indices = signal.find_peaks_cwt(data, cwt_widths, min_snr=1.0)
        found_r = x_axis[peak_indices]
        valid_mask = (found_r > 10) & (found_r < self.radius - 10)
        found_r = found_r[valid_mask]
        
        # 2. Local Verify
        verified_peaks = []
        for r_val in found_r:
            result = self._verify_peak_locally(data, r_val)
            if result:
                is_duplicate = False
                for vp in verified_peaks:
                    if abs(result[0] - vp[0]) < max(3.0, vp[1]*1.5):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    verified_peaks.append(result)

        if len(verified_peaks) == 0:
            return self.r_grid, np.zeros_like(self.r_grid), []

        # 3. Global Optimization
        n_peaks = len(verified_peaks)
        p0 = []
        lb = []
        ub = []
        
        for p in verified_peaks:
            r_val, sig_val, amp_val = p
            p0.extend(p)
            lb.extend([r_val - 10.0, 0.8, 0.0])
            ub.extend([r_val + 10.0, 60.0, np.inf])

        try:
            res = least_squares(
                self._residual_global, p0, args=(x_axis, data, n_peaks), 
                bounds=(lb, ub), ftol=1e-5, xtol=1e-5
            )
            p_final = res.x
        except:
            p_final = np.array(verified_peaks).flatten()

        recon_profile = np.zeros_like(x_axis, dtype=float)
        fitted_params_list = []
        for i in range(n_peaks):
            r0 = p_final[i*3]
            sig = p_final[i*3+1]
            amp = p_final[i*3+2]
            fitted_params_list.append({'r': r0, 'sigma': sig, 'amp': amp})
            recon_profile += amp * np.exp(-((x_axis - r0)**2) / (2 * sig**2))
            
        return self.r_grid, recon_profile, fitted_params_list

# ==========================================
# 辅助函数: 用于从 rbasex 结果中提取参数
# ==========================================
def extract_peak_params_from_array(r_axis, intensity_array, target_r):
    """从数组中找峰参数用于对比"""
    # 在目标 r 附近寻找最大值
    idx_center = np.argmin(np.abs(r_axis - target_r))
    window = 20
    idx_start = max(0, idx_center - window)
    idx_end = min(len(r_axis), idx_center + window)
    
    sub_r = r_axis[idx_start:idx_end]
    sub_i = intensity_array[idx_start:idx_end]
    
    if np.sum(sub_i) == 0: return 0, 0
    
    # 1. 峰位 (重心)
    peak_pos = np.sum(sub_r * sub_i) / np.sum(sub_i)
    
    # 2. Sigma (从 FWHM 估算)
    peak_val = np.max(sub_i)
    half_max = peak_val / 2.0
    above_half = sub_i > half_max
    if np.sum(above_half) > 0:
        fwhm = np.sum(above_half) # 粗略估算
        # 更精细的插值
        indices = np.where(above_half)[0]
        if len(indices) >= 2:
            left = sub_r[indices[0]]
            right = sub_r[indices[-1]]
            fwhm = right - left
        sigma = fwhm / 2.355
    else:
        sigma = 0
        
    return peak_pos, sigma

# ==========================================
# 主程序
# ==========================================
def main():
    N = 501
    noise_lvl = 0.012 # 中等偏强噪声
    
    # 定义 3 个峰
    true_peaks = [
        [40.0,  5.0, 0.8], 
        [120.0, 3.0, 1.0], 
        [220.0, 1.5, 0.6] 
    ]
    
    print("=== Benchmark: rBasex vs. Forward Fitter ===")
    
    # --- 1. 生成数据 ---
    r = np.linspace(0, N//2, N//2 + 1)
    f_total = np.zeros_like(r)
    for p in true_peaks: f_total += p[2] * np.exp(-((r - p[0])**2) / (2 * p[1]**2))
    
    center = N // 2
    y, x = np.ogrid[-center:N-center, -center:N-center]
    r_grid = np.sqrt(x**2 + y**2)
    density_2d = np.interp(r_grid, r, f_total, left=0, right=0)
    
    # 前向投影
    proj_res = abel.Transform(density_2d, method='hansenlaw', direction='forward', verbose=False)
    proj_image = proj_res.transform
    
    np.random.seed(42)
    scale = 1.0 / noise_lvl
    # 添加背景 +20，测试抗干扰能力
    bg_level = 20
    proj_noisy = np.random.poisson(proj_image * scale + bg_level) / scale
    
    # 提取中心切片给 Fitter
    proj_profile_1d = proj_noisy[N//2, N//2:]

    # --- 2. 运行 rBasex ---
    print("\n[Running] rBasex...")
    t0 = time.time()
    # 注意：rBasex 假设边界为0。为了公平，我们需要帮它扣除背景。
    # 如果不扣背景，rBasex 会严重发散。
    # 我们用边缘值作为背景估计
    bg_est = np.mean(proj_noisy[:10, :10])
    proj_noisy_sub = proj_noisy - bg_est
    proj_noisy_sub[proj_noisy_sub < 0] = 0 # 非负约束
    
    res_rbasex = abel.Transform(proj_noisy_sub, method='rbasex', direction='inverse', verbose=False)
    r_rb, I_rb = abel.tools.vmi.angular_integration_3D(res_rbasex.transform, origin=(N//2, N//2), dr=1)
    t_rb = time.time() - t0
    
    # 归一化 (匹配真值高度)
    peak_idx = np.argmin(np.abs(r_rb - 120.0))
    scale_rb = 1.0 / I_rb[peak_idx] # 假设中间峰是 1.0
    I_rb *= scale_rb

    # --- 3. 运行 Precision Fitter ---
    print("\n[Running] Precision Fitter (Yours)...")
    t0 = time.time()
    solver = PrecisionMultiPeakFitter(N)
    # Fitter 不需要手动扣背景，它的局部拟合会自动处理 bg 参数
    r_my, I_my, fitted_params = solver.solve(proj_profile_1d)
    t_my = time.time() - t0
    
    # Fitter 直接输出物理强度，不需要额外归一化
    
    # --- 4. 数据对比 ---
    print("\n" + "="*80)
    print(f"{'ID':<3} | {'Truth (r,sig)':<15} | {'rBasex (r,sig)':<18} | {'Fitter (r,sig)':<18} | {'Fitter Err':<10}")
    print("-" * 80)
    
    for i, true_p in enumerate(true_peaks):
        tr, tsig = true_p[0], true_p[1]
        
        # 获取 rBasex 参数 (估算)
        rb_r, rb_sig = extract_peak_params_from_array(r_rb, I_rb, tr)
        
        # 获取 Fitter 参数 (精确)
        best_match = None
        min_dist = 999
        for fp in fitted_params:
            if abs(fp['r'] - tr) < min_dist:
                min_dist = abs(fp['r'] - tr)
                best_match = fp
        
        if best_match:
            fr, fsig = best_match['r'], best_match['sigma']
            ferr = abs(fr - tr)
            print(f"{i+1:<3} | {tr:<5.1f}, {tsig:<5.1f}       | {rb_r:<6.1f}, {rb_sig:<6.1f}      | {fr:<6.2f}, {fsig:<6.2f}      | {ferr:<6.3f}")
        else:
            print(f"{i+1:<3} | {tr:<5.1f}, {tsig:<5.1f}       | {rb_r:<6.1f}, {rb_sig:<6.1f}      | {'Missed':<18} | -")
            
    print("="*80)
    print(f"Time: rBasex={t_rb*1000:.1f}ms, Fitter={t_my*1000:.1f}ms")

    # --- 5. 绘图 ---
    plt.figure(figsize=(15, 8))
    
    # 子图 1: 全谱对比
    plt.subplot(2, 2, 1)
    plt.plot(r, f_total, 'k', alpha=0.3, linewidth=4, label='Ground Truth')
    plt.plot(r_rb, I_rb, 'b-', linewidth=1, alpha=0.8, label='rBasex')
    plt.plot(r_my, I_my, 'r-', linewidth=2, label='Fitter (Yours)')
    plt.title("Full Reconstruction Comparison")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 子图 2: 内层宽峰细节 (r=40)
    # 这里展示 rBasex 的基线噪声问题 vs Fitter 的平滑性
    plt.subplot(2, 2, 2)
    plt.plot(r, f_total, 'k', alpha=0.3, linewidth=4, label='Truth')
    plt.plot(r_rb, I_rb, 'b.-', markersize=3, label='rBasex')
    plt.plot(r_my, I_my, 'r.-', markersize=3, label='Fitter')
    plt.xlim(10, 70)
    plt.title("Inner Peak (Wide, r=40) - Check Baseline Noise")
    plt.legend()
    plt.grid(True, alpha=0.2)

    # 子图 3: 外层窄峰细节 (r=220)
    # 展示分辨率
    plt.subplot(2, 2, 3)
    plt.plot(r, f_total, 'k', alpha=0.3, linewidth=4, label='Truth')
    plt.plot(r_rb, I_rb, 'b.-', markersize=3, label='rBasex')
    plt.plot(r_my, I_my, 'r.-', markersize=3, label='Fitter')
    plt.xlim(200, 240)
    plt.title("Outer Peak (Narrow, r=220) - Check Resolution")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 子图 4: 中心奇点噪声 (r < 20)
    plt.subplot(2, 2, 4)
    plt.plot(r, f_total, 'k', alpha=0.3, linewidth=4, label='Truth')
    plt.plot(r_rb, I_rb, 'b.-', markersize=3, label='rBasex')
    plt.plot(r_my, I_my, 'r.-', markersize=3, label='Fitter')
    plt.xlim(0, 20)
    plt.ylim(-0.2, 0.4) # 关注 0 附近的震荡
    plt.title("Center Singularity Noise (r < 20)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# refined version for multi-peak fitting
#------------------------------------------ 
import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy import signal
from scipy.optimize import least_squares
import abel.hansenlaw 

# ==========================================
# 核心算法: Scale-Adaptive Fitter (可调参版)
# ==========================================
class ScaleAdaptiveFitter:
    def __init__(self, n_pixels, 
                 cwt_width_range=(0.005, 0.25), 
                 cwt_snr=1.0, 
                 cluster_thresh=0.06, 
                 fit_window_ratio=0.15):
        """
        初始化反演器，所有可调参数在此定义。

        :param n_pixels: 图像大小 (N)
        
        --- CWT 找峰参数 ---
        :param cwt_width_range: (min_ratio, max_ratio) 
                                CWT 搜索的宽度范围，相对于半径的比例。
                                (0.005, 0.25) 意味着能找到从 0.5% 到 25% 半径宽度的峰。
                                如果你的峰特别宽，调大第二个值。
        :param cwt_snr:         信噪比阈值。
                                1.0 是标准值。如果峰很弱没找到，设为 0.5；如果假峰太多，设为 2.0。

        --- 聚类与拟合参数 ---
        :param cluster_thresh:  聚类距离阈值 (相对于半径)。
                                默认为 0.06 (6%)。如果两个峰距离 < 6% 半径，会被视为重叠峰一起拟合。
                                如果你发现两个明显分开的峰被粘在一起了，调小这个值。
        :param fit_window_ratio: 局部拟合窗口大小 (相对于半径)。
                                 默认为 0.15。窗口必须大到能包含宽峰的拖尾。
                                 对于巨宽的峰，可能需要调大到 0.2 或 0.3。
        """
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid = np.arange(self.radius + 1, dtype=float)
        
        # 保存参数
        self.min_w_ratio, self.max_w_ratio = cwt_width_range
        self.cwt_snr = cwt_snr
        self.cluster_thresh = cluster_thresh
        self.fit_window_ratio = fit_window_ratio

    def _forward_model_multi_peak(self, r_axis, params, n_peaks):
        model = np.zeros_like(r_axis)
        for i in range(n_peaks):
            r0 = params[i*3]
            sig = params[i*3+1]
            amp = params[i*3+2]
            if sig < 0.1 or amp < 1e-9: continue
            f_r = amp * np.exp(-((r_axis - r0)**2) / (2 * sig**2))
            proj = abel.hansenlaw.hansenlaw_transform(f_r, direction='forward', dr=1)
            if len(proj) > len(r_axis): proj = proj[:len(r_axis)]
            elif len(proj) < len(r_axis): proj = np.pad(proj, (0, len(r_axis)-len(proj)))
            model += proj
        return model

    def _residual_cluster(self, params, x_sub, y_sub, n_peaks_in_cluster):
        peak_params = params[:-1]
        bg = params[-1]
        model = self._forward_model_multi_peak(x_sub, peak_params, n_peaks_in_cluster)
        return (model + bg) - y_sub

    def _residual_global(self, params, x_data, y_data, n_peaks):
        model = self._forward_model_multi_peak(x_data, params, n_peaks)
        return model - y_data

    def solve(self, projected_profile):
        x_axis = self.r_grid
        data = projected_profile.astype(float)
        max_val = np.max(data)
        
        # --- Step 1: 动态 CWT 找峰 ---
        min_width = max(1, self.radius * self.min_w_ratio)
        max_width = max(10, self.radius * self.max_w_ratio)
        cwt_widths = np.logspace(np.log10(min_width), np.log10(max_width), num=40)
        
        peak_indices = signal.find_peaks_cwt(data, cwt_widths, min_snr=self.cwt_snr)
        found_r = x_axis[peak_indices]
        
        # 边缘过滤
        margin = max(5, self.radius * 0.02)
        found_r = found_r[(found_r > margin) & (found_r < self.radius - margin)]
        
        if len(found_r) == 0:
            return self.r_grid, np.zeros_like(self.r_grid), []

        print(f"   [CWT] Candidates: {np.round(found_r, 1)}")

        # --- Step 2: 峰聚类 (Clustering) ---
        clusters = [] 
        if len(found_r) > 0:
            current_cluster = [found_r[0]]
            for i in range(1, len(found_r)):
                # dist_threshold 控制参数在这里生效
                dist_abs = max(15, self.radius * self.cluster_thresh)
                if found_r[i] - found_r[i-1] < dist_abs:
                    current_cluster.append(found_r[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [found_r[i]]
            clusters.append(current_cluster)
            
        # --- Step 3: 局部簇验证 ---
        verified_peaks = []
        
        for cluster in clusters:
            r_center = np.mean(cluster)
            n_in = len(cluster)
            span = cluster[-1] - cluster[0]
            
            # fit_window_ratio 控制参数在这里生效
            window_half = int(max(20, self.radius * self.fit_window_ratio + span))
            idx_start = max(0, int(r_center) - window_half)
            idx_end = min(len(data), int(r_center) + window_half)
            
            x_sub = x_axis[idx_start:idx_end]
            y_sub = data[idx_start:idx_end]
            
            p0 = []
            lb = []
            ub = []
            bg_guess = np.min(y_sub)
            
            for r_val in cluster:
                sig_guess = max(3.0, 300.0/(r_val+1)) 
                proj_h = max(0, data[int(r_val)] - bg_guess)
                amp_guess = proj_h / (sig_guess * 2.5 + 1e-6)
                p0.extend([r_val, sig_guess, amp_guess])
                lb.extend([r_val - window_half*0.8, 0.5, 0.0])
                ub.extend([r_val + window_half*0.8, self.radius*0.6, np.inf])
            
            p0.append(bg_guess)
            lb.append(0.0)
            ub.append(max_val)
            
            try:
                res = least_squares(
                    self._residual_cluster, p0, args=(x_sub, y_sub, n_in),
                    bounds=(lb, ub), ftol=1e-3
                )
                p_opt = res.x[:-1]
                
                for i in range(n_in):
                    r_fit = p_opt[i*3]
                    sig_fit = p_opt[i*3+1]
                    amp_fit = p_opt[i*3+2]
                    # 判据：不是纯噪声
                    proj_contrib = amp_fit * sig_fit * 2.5
                    if proj_contrib > max_val * 0.015: # 1.5% 阈值
                        verified_peaks.append([r_fit, sig_fit, amp_fit])
            except: pass

        if len(verified_peaks) == 0:
            return self.r_grid, np.zeros_like(self.r_grid), []

        # --- Step 4: 全局优化 ---
        verified_peaks.sort(key=lambda x: x[0])
        n_peaks = len(verified_peaks)
        p0 = []
        lb = []
        ub = []
        
        for p in verified_peaks:
            r_val, sig_val, amp_val = p
            p0.extend(p)
            lb.extend([r_val - 10.0, 0.5, 0.0])
            ub.extend([r_val + 10.0, self.radius/2.5, np.inf])

        try:
            res = least_squares(
                self._residual_global, p0, args=(x_axis, data, n_peaks), 
                bounds=(lb, ub), ftol=1e-5
            )
            p_final = res.x
        except:
            p_final = np.array(verified_peaks).flatten()

        recon_profile = np.zeros_like(x_axis, dtype=float)
        fitted_params_list = []
        for i in range(n_peaks):
            r0 = p_final[i*3]
            sig = p_final[i*3+1]
            amp = p_final[i*3+2]
            fitted_params_list.append({'r': r0, 'sigma': sig, 'amp': amp})
            recon_profile += amp * np.exp(-((x_axis - r0)**2) / (2 * sig**2))
            
        return self.r_grid, recon_profile, fitted_params_list

# ==========================================
# 辅助函数
# ==========================================
def extract_peak_params_from_array(r_axis, intensity_array, target_r):
    """从 rBasex 结果中找峰参数"""
    idx_center = np.argmin(np.abs(r_axis - target_r))
    window = 30
    idx_start = max(0, idx_center - window)
    idx_end = min(len(r_axis), idx_center + window)
    sub_r = r_axis[idx_start:idx_end]
    sub_i = intensity_array[idx_start:idx_end]
    if np.sum(sub_i) == 0: return 0, 0
    
    peak_pos = np.sum(sub_r * sub_i) / np.sum(sub_i)
    
    # 简单的 FWHM 估算
    peak_val = np.max(sub_i)
    above_half = sub_i > peak_val / 2.0
    if np.sum(above_half) > 1:
        fwhm = (np.sum(above_half)) * (sub_r[1]-sub_r[0])
        sigma = fwhm / 2.355
    else:
        sigma = 0
    return peak_pos, sigma

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 模拟设置：大矩阵 + 宽峰 + 重叠峰
    N = 1001 
    noise_lvl = 0.005
    




    # True Peaks [r, sigma, amp]
    true_peaks = [
        [150.0, 25.0, 0.5],   # 宽峰 (Sigma=25, FWHM~60)
        [350.0, 15.0,  1.0],   # 重叠峰 A
        [370.0, 5.0,  0.9],   # 重叠峰 B (间距 20px = 4 sigma)
        [380.0, 2.0,  0.6]    # 边缘窄峰
    ]
    
    print(f"=== Comparative Benchmark (N={N}) ===")
    
    # 生成数据
    r = np.linspace(0, N//2, N//2 + 1)
    f_total = np.zeros_like(r)
    for p in true_peaks: f_total += p[2] * np.exp(-((r - p[0])**2) / (2 * p[1]**2))
    
    center = N // 2
    y, x = np.ogrid[-center:N-center, -center:N-center]
    r_grid = np.sqrt(x**2 + y**2)
    density_2d = np.interp(r_grid, r, f_total, left=0, right=0)
    
    # 前向投影
    proj_res = abel.Transform(density_2d, method='hansenlaw', direction='forward', verbose=False)
    proj_image = proj_res.transform
    
    # 加噪
    np.random.seed(42)
    scale = 1.0 / noise_lvl
    # 加上大背景 (50)，测试基线稳定性
    proj_noisy = np.random.poisson(proj_image * scale + 50) / scale
    proj_profile_1d = proj_noisy[N//2, N//2:]
    
    # --- 2. 运行 rBasex ---
    print("\n[Running] rBasex...")
    t0 = time.time()
    # 手动扣背景给 rBasex 一个机会，否则它会崩
    bg_est = np.mean(proj_noisy[:50, :50])
    proj_rb_in = np.maximum(proj_noisy - bg_est, 0)
    res_rb = abel.Transform(proj_rb_in, method='rbasex', direction='inverse', verbose=False)
    r_rb, I_rb = abel.tools.vmi.angular_integration_3D(res_rb.transform, origin=(N//2, N//2), dr=1)
    t_rb = time.time() - t0
    
    # 归一化 rBasex (对齐最高的峰)
    I_rb *= (1.0 / np.max(I_rb))

    # --- 3. 运行 ScaleAdaptiveFitter (你的方法) ---
    print("\n[Running] Adaptive Fitter...")
    # ==========================
    # 在这里调节参数！
    # ==========================
    solver = ScaleAdaptiveFitter(
        N, 
        cwt_width_range=(0.002, 0.1),  # 宽峰需要上限设大一点 (0.3)
        cwt_snr=0.4,                   # 宽峰很平，SNR设低一点
        cluster_thresh=0.01,           # 8% 半径内的峰会被合并 (~40px)
        fit_window_ratio=0.10          # 窗口大一点，包住宽峰
    )

    t0 = time.time()
    r_my, I_my, fitted_params = solver.solve(proj_profile_1d)
    t_my = time.time() - t0
    
    # Fitter 输出是物理强度，也归一化以便画图对比
    if np.max(I_my) > 0: I_my *= (1.0 / np.max(I_my))

    # --- 4. 统计表格 ---
    print("\n" + "="*85)
    print(f"{'True r':<10} | {'True Sig':<10} | {'rBasex r':<10} | {'rBasex Sig':<10} | {'Yours r':<10} | {'Yours Sig':<10}")
    print("-" * 85)
    
    for tp in true_peaks:
        tr, tsig = tp[0], tp[1]
        
        # rBasex 估算
        rb_r, rb_sig = extract_peak_params_from_array(r_rb, I_rb, tr)
        
        # Yours 精确值
        best = None
        min_d = 999
        for fp in fitted_params:
            if abs(fp['r'] - tr) < min_d:
                min_d = abs(fp['r'] - tr)
                best = fp
        
        if best:
            print(f"{tr:<10.1f} | {tsig:<10.1f} | {rb_r:<10.2f} | {rb_sig:<10.2f} | {best['r']:<10.2f} | {best['sigma']:<10.2f}")
        else:
            print(f"{tr:<10.1f} | {tsig:<10.1f} | {rb_r:<10.2f} | {rb_sig:<10.2f} | {'Missed':<10} | -")
            
    print("="*85)
    print(f"Time: rBasex={t_rb*1000:.1f}ms, Yours={t_my*1000:.1f}ms")

    # --- 5. 详细绘图 ---
    plt.figure(figsize=(15, 10))
    
    # 子图 1: 全谱
    plt.subplot(2, 2, 1)
    plt.plot(r, f_total/np.max(f_total), 'k', alpha=0.3, linewidth=3, label='Truth')
    plt.plot(r_rb, I_rb, 'b-', alpha=0.6, label='rBasex')
    plt.plot(r_my, I_my, 'r-', linewidth=1.5, label='Yours')
    plt.title("Full Spectrum Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 宽峰细节 (r=150)
    plt.subplot(2, 2, 2)
    plt.plot(r, f_total/np.max(f_total), 'k', alpha=0.3, linewidth=3)
    plt.plot(r_rb, I_rb, 'b.-', label='rBasex')
    plt.plot(r_my, I_my, 'r.-', label='Yours')
    plt.xlim(50, 250)
    plt.title("Wide Peak Challenge (r=150, Sig=25)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 3: 重叠峰细节 (r=350, 370)
    plt.subplot(2, 2, 3)
    plt.plot(r, f_total/np.max(f_total), 'k', alpha=0.3, linewidth=3)
    plt.plot(r_rb, I_rb, 'b.-', label='rBasex')
    plt.plot(r_my, I_my, 'r.-', label='Yours')
    plt.xlim(300, 420)
    plt.title("Overlapping Peaks (r=350 & 370)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 4: 中心/基线稳定性
    plt.subplot(2, 2, 4)
    plt.plot(r_rb, I_rb, 'b-', label='rBasex')
    plt.plot(r_my, I_my, 'r-', label='Yours')
    plt.xlim(0, 50)
    plt.ylim(-0.1, 0.2)
    plt.title("Center/Baseline Noise Check")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# add reverse operation to find valley first
#------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy import signal, ndimage
from scipy.optimize import least_squares
import abel.hansenlaw 

# ==========================================
# 核心类: Morphology + Gradient Fitter
# ==========================================
class MorphoGradientFitter:
    def __init__(self, n_pixels, grad_weight=20.0):
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid = np.arange(self.radius + 1, dtype=float)
        self.grad_weight = grad_weight

    def _forward_model_multi_peak(self, r_axis, params, n_peaks):
        model = np.zeros_like(r_axis)
        for i in range(n_peaks):
            r0 = params[i*3]
            sig = params[i*3+1]
            amp = params[i*3+2]
            if sig < 0.1 or amp < 1e-9: continue
            
            f_r = amp * np.exp(-((r_axis - r0)**2) / (2 * sig**2))
            proj = abel.hansenlaw.hansenlaw_transform(f_r, direction='forward', dr=1)
            
            if len(proj) > len(r_axis): proj = proj[:len(r_axis)]
            elif len(proj) < len(r_axis): proj = np.pad(proj, (0, len(r_axis)-len(proj)))
            model += proj
        return model

    def _residual_gradient_enhanced(self, params, x_data, y_data, y_grad_target, n_peaks):
        """能量 + 梯度 联合优化"""
        y_model = self._forward_model_multi_peak(x_data, params, n_peaks)
        y_model_grad = np.gradient(y_model)
        
        res_intensity = y_model - y_data
        res_gradient = (y_model_grad - y_grad_target) * self.grad_weight
        return np.concatenate([res_intensity, res_gradient])

    def solve(self, projected_profile):
        x_axis = self.r_grid
        data = projected_profile.astype(float)
        max_val = np.max(data)
        
        # --- Step 0: 梯度目标提取 ---
        # 依然需要一点点平滑来计算梯度 Loss 的目标，但这不影响分区
        sg_window = max(11, int(self.radius * 0.03) | 1)
        data_smooth_grad = signal.savgol_filter(data, window_length=sg_window, polyorder=3)
        target_gradient = np.gradient(data_smooth_grad)

        # --- Step 1: 形态学找谷底 (Morphological Valley Detection) ---
        # 1. 图像取反 (谷底变成峰)
        data_inv = np.max(data) - data
        
        # 2. 形态学开运算 (Opening) - 核心改进
        # 它可以抹除掉像“刺”一样的噪声窄峰，但保留宽阔的结构峰
        # structure size 决定了我们要忽略多窄的噪声坑。设为 5-10 像素通常很好。
        morph_struct_size = max(5, int(self.radius * 0.015))
        data_inv_clean = ndimage.grey_opening(data_inv, size=morph_struct_size)
        
        # 3. 在清洗后的反转图上找峰 (即原图的谷底)
        # prominence: 忽略太浅的坑
        valley_indices, _ = signal.find_peaks(data_inv_clean, prominence=max_val * 0.02)
        
        # 4. 加上两端边界
        boundaries = np.concatenate(([0], x_axis[valley_indices], [self.radius]))
        boundaries = np.sort(np.unique(boundaries))
        
        print(f"   [Morphology] Valleys found at r={np.round(x_axis[valley_indices], 1)}")
        
        # --- Step 2: 分区间 CWT 找峰 ---
        all_candidates = [] 
        cwt_widths = np.logspace(np.log10(2), np.log10(self.radius*0.3), 40)
        
        for i in range(len(boundaries) - 1):
            b_start = int(boundaries[i])
            b_end = int(boundaries[i+1])
            
            # 截取区间
            idx_s = max(0, b_start)
            idx_e = min(len(data), b_end)
            
            # 区间过小跳过
            if idx_e - idx_s < 3: continue
            
            sub_data = data[idx_s:idx_e]
            sub_axis = x_axis[idx_s:idx_e]
            
            # 判据：区间内是否有显著信号
            # 使用 data_inv_clean 的反面(原数据的大致包络)来判断更稳健
            # 或者简单判断最大值
            if np.max(sub_data) < max_val * 0.05: continue
            
            # 局部 CWT
            peak_idx_local = signal.find_peaks_cwt(sub_data, cwt_widths, min_snr=0.2)
            r_in_interval = sub_axis[peak_idx_local]
            
            # --- 强制保底机制 ---
            # 如果区间很显著(肯定有峰)，但 CWT 没找到(可能太宽/太重叠)
            # 我们强制在区间最大值处加一个峰
            if len(r_in_interval) == 0:
                local_max_idx = np.argmax(sub_data)
                r_force = sub_axis[local_max_idx]
                r_in_interval = [r_force]
                print(f"      -> Interval [{b_start}-{b_end}]: Force added peak at {r_force}")
            
            for r_val in r_in_interval:
                # 记录峰及其硬约束边界
                all_candidates.append({
                    'r': r_val,
                    'lb_r': b_start,
                    'ub_r': b_end
                })

        n_peaks = len(all_candidates)
        if n_peaks == 0: return self.r_grid, np.zeros_like(self.r_grid), []

        # --- Step 3: 构建全局优化 ---
        p0 = []
        lb = []
        ub = []
        
        for p in all_candidates:
            r_val = p['r']
            sig_guess = max(3.0, 300.0/(r_val+1))
            amp_guess = data[int(r_val)] / (sig_guess * 2.5 + 1e-9)
            
            p0.extend([r_val, sig_guess, amp_guess])
            
            # 硬约束：峰中心不能跑出谷底围成的栅栏
            # 留 0.1 余量
            r_min = p['lb_r'] + 0.1
            r_max = p['ub_r'] - 0.1
            if r_max <= r_min: r_min, r_max = r_val - 2, r_val + 2
                
            lb.extend([r_min, 0.5, 0.0])
            ub.extend([r_max, 80.0, np.inf])

        # --- Step 4: 执行优化 ---
        print(f"   [Fit] Optimizing {n_peaks} peaks...")
        try:
            res = least_squares(
                self._residual_gradient_enhanced, 
                p0, 
                args=(x_axis, data, target_gradient, n_peaks), 
                bounds=(lb, ub), 
                ftol=1e-6, xtol=1e-6
            )
            p_final = res.x
        except Exception as e:
            print(f"Fit failed: {e}")
            return self.r_grid, np.zeros_like(self.r_grid), []

        # 重建
        recon_profile = np.zeros_like(x_axis)
        fitted_params_list = []
        
        # 最终模型用于画图验证
        final_model_proj = self._forward_model_multi_peak(x_axis, p_final, n_peaks)
        final_model_grad = np.gradient(final_model_proj)
        
        for i in range(n_peaks):
            r0 = p_final[i*3]
            sig = p_final[i*3+1]
            amp = p_final[i*3+2]
            
            # 后处理：过滤被压扁的无效峰
            if amp * sig * 2.5 > max_val * 0.01:
                fitted_params_list.append({'r': r0, 'sigma': sig, 'amp': amp})
                recon_profile += amp * np.exp(-((x_axis - r0)**2) / (2 * sig**2))
            
        # 返回 data_inv_clean 用于画图展示形态学的效果
        return self.r_grid, recon_profile, fitted_params_list, data_inv_clean, boundaries, target_gradient, final_model_grad

# ==========================================
# 辅助函数
# ==========================================
def extract_peak_params_from_array(r_axis, intensity_array, target_r):
    idx_center = np.argmin(np.abs(r_axis - target_r))
    window = 30
    idx_start = max(0, idx_center - window)
    idx_end = min(len(r_axis), idx_center + window)
    sub_r = r_axis[idx_start:idx_end]
    sub_i = intensity_array[idx_start:idx_end]
    if np.sum(sub_i) == 0: return 0, 0
    peak_pos = np.sum(sub_r * sub_i) / np.sum(sub_i)
    peak_val = np.max(sub_i)
    above_half = sub_i > peak_val / 2.0
    if np.sum(above_half) > 1:
        fwhm = (np.sum(above_half)) * (sub_r[1]-sub_r[0])
        sigma = fwhm / 2.355
    else:
        sigma = 0
    return peak_pos, sigma

# ==========================================
# 主程序
# ==========================================
def main():
    N = 1001 
    noise_lvl = 0.005
    
    # 构造极难的重叠峰 (测试形态学分割能力)
    # r=340 和 r=360 之间有一个非常浅的谷底，SavGol 可能会把它抹平，但 Morph 应该能抓住
    true_peaks = [
        [100.0, 10.0, 0.6], 
        [300.0, 8.0,  1.0], 
        [340.0, 8.0,  0.8], 
        [360.0, 5.0,  0.6], # 紧密重叠
    ]
    
    print(f"=== Morphology-Guided Gradient Fitter (N={N}) ===")
    
    # 1. 生成数据
    r = np.linspace(0, N//2, N//2 + 1)
    f_total = np.zeros_like(r)
    for p in true_peaks: f_total += p[2] * np.exp(-((r - p[0])**2) / (2 * p[1]**2))
    
    center = N // 2
    y, x = np.ogrid[-center:N-center, -center:N-center]
    r_grid = np.sqrt(x**2 + y**2)
    density_2d = np.interp(r_grid, r, f_total, left=0, right=0)
    proj_res = abel.Transform(density_2d, method='hansenlaw', direction='forward', verbose=False)
    
    np.random.seed(42)
    proj_noisy = np.random.poisson(proj_res.transform * 200 + 50) / 200.0
    proj_1d = proj_noisy[N//2, N//2:]
    
    # 2. rBasex
    t0 = time.time()
    bg_est = np.mean(proj_noisy[:50, :50])
    proj_rb_in = np.maximum(proj_noisy - bg_est, 0)
    res_rb = abel.Transform(proj_rb_in, method='rbasex', direction='inverse', verbose=False)
    r_rb, I_rb = abel.tools.vmi.angular_integration_3D(res_rb.transform, origin=(N//2, N//2), dr=1)
    t_rb = time.time() - t0
    I_rb *= (1.0 / np.max(I_rb))

    # 3. Yours (Morphology + Gradient)
    solver = MorphoGradientFitter(N, grad_weight=15.0)
    t0 = time.time()
    # 返回值包含 morph_clean 数据，用于画图展示
    r_my, I_my, params, morph_clean, boundaries, g_target, g_fit = solver.solve(proj_1d)
    t_my = time.time() - t0
    if np.max(I_my) > 0: I_my *= (1.0 / np.max(I_my))

    # 4. 统计
    print("\n" + "="*85)
    print(f"{'True r':<10} | {'rBasex r':<10} | {'Yours r':<10} | {'Err(Y)':<10}")
    print("-" * 85)
    for tp in true_peaks:
        tr = tp[0]
        rb_r, _ = extract_peak_params_from_array(r_rb, I_rb, tr)
        best = min(params, key=lambda x: abs(x['r']-tr)) if params else None
        if best:
            print(f"{tr:<10.1f} | {rb_r:<10.2f} | {best['r']:<10.2f} | {abs(best['r']-tr):<10.3f}")
        else:
            print(f"{tr:<10.1f} | {rb_r:<10.2f} | Missed     | -")
    print("="*85)
    print(f"Time: rBasex={t_rb*1000:.1f}ms, Yours={t_my*1000:.1f}ms")

    # 5. 绘图
    plt.figure(figsize=(12, 12))
    
    # 图 1: 形态学处理展示 (核心改进)
    plt.subplot(3, 1, 1)
    # 画出反转后的原始数据 (含噪)
    data_inv_raw = np.max(proj_1d) - proj_1d
    plt.plot(r, data_inv_raw, 'k-', alpha=0.2, label='Inverted Raw Data (Noisy)')
    # 画出形态学处理后的数据 (干净)
    plt.plot(r, morph_clean, 'b-', linewidth=2, label='Morphological Cleaned (Valleys->Peaks)')
    
    # 标出谷底
    for b in boundaries:
        if 0 < b < N//2:
            plt.axvline(b, color='orange', linestyle='--', linewidth=2, label='Detected Boundary' if b==boundaries[1] else "")
            
    plt.title("Step 1: Morphology-Based Valley Detection")
    plt.legend()
    plt.xlim(0, 500) # 聚焦重叠区
    plt.grid(True, alpha=0.3)
    
    # 图 2: 拟合结果
    plt.subplot(3, 1, 2)
    plt.plot(r, f_total/np.max(f_total), 'k', alpha=0.3, linewidth=4, label='Truth')
    plt.plot(r_rb, I_rb, 'b-', alpha=0.6, label='rBasex')
    plt.plot(r_my, I_my, 'r-', linewidth=2, label='Yours (Morph+Grad)')
    plt.title("Step 2: Final Fit Result")
    plt.legend()
    plt.xlim(0, 500)
    
    # 图 3: 梯度约束
    plt.subplot(3, 1, 3)
    plt.plot(g_target, 'g-', alpha=0.5, label='Target Gradient')
    plt.plot(g_fit, 'r--', linewidth=2, label='Fitted Gradient')
    plt.title("Step 3: Gradient Matching")
    plt.legend()
    plt.xlim(0, 500)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


#update angular integration and beta estimation
#------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy import signal
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from abel.tools.vmi import angular_integration_3D

# ==========================================
# 1. 物理场模拟 (VMI Simulation)
# ==========================================
def generate_vmi_image(n_pixels, peaks, total_photons, bg_sigma):
    """
    模拟 VMI 实验图像：
    I(r, theta) ~ Amp * exp(...) * [1 + Beta * P2(cos theta)]
    """
    # 1. 严格生成指定尺寸的网格
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y = y - center
    x = x - center
    r_grid = np.sqrt(x**2 + y**2)
    
    # 2. 安全计算 cos(theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    
    # P2 Legendre 多项式
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    img_3d = np.zeros_like(r_grid, dtype=float)
    
    for p in peaks:
        radial_term = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        angular_term = 1 + p['beta'] * P2
        img_3d += radial_term * angular_term

    # Abel 投影
    proj_clean = abel.Transform(img_3d, method='hansenlaw', direction='forward', verbose=False).transform
    
    # 加噪
    if np.sum(proj_clean) == 0: return proj_clean
    prob = proj_clean / np.sum(proj_clean)
    img_counts = np.random.poisson(prob * total_photons)
    bg_noise = np.random.normal(0, bg_sigma, proj_clean.shape)
    
    return np.maximum(img_counts + bg_noise, 0)

# ==========================================
# 2. 勒让德矩提取 (Bug Fixed)
# ==========================================
def extract_angular_projections(image):
    """
    将 2D 图像分解为 P0 (各向同性) 和 P2 (各向异性) 投影。
    修复: 强制截断到 radius 长度，忽略角落数据。
    """
    h, w = image.shape
    cy, cx = h//2, w//2
    max_radius = min(cy, cx)  # 定义有效半径
    
    y, x = np.ogrid[:h, :w]
    y, x = y - cy, x - cx
    r = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    
    P2_map = 0.5 * (3 * cos_theta**2 - 1)
    
    # r 坐标离散化
    r_bins = r.astype(int)
    
    # 1. P0 投影
    proj_0_raw = np.bincount(r_bins.ravel(), weights=image.ravel())
    
    # 2. P2 投影
    proj_2_raw = np.bincount(r_bins.ravel(), weights=(image * P2_map).ravel())
    
    # 3. 计数
    counts_raw = np.bincount(r_bins.ravel())
    
    # === 关键修复: 截断数组到 max_radius + 1 ===
    limit = max_radius + 1
    
    # 如果 bincount 结果比 limit 长 (因为角落像素 r > radius)，则截断
    if len(proj_0_raw) > limit:
        proj_0_raw = proj_0_raw[:limit]
        proj_2_raw = proj_2_raw[:limit]
        counts_raw = counts_raw[:limit]
    # 如果结果比 limit 短 (虽然不太可能)，则补零
    elif len(proj_0_raw) < limit:
        pad_len = limit - len(proj_0_raw)
        proj_0_raw = np.pad(proj_0_raw, (0, pad_len))
        proj_2_raw = np.pad(proj_2_raw, (0, pad_len))
        counts_raw = np.pad(counts_raw, (0, pad_len))
        
    # 归一化
    p0_out = np.zeros(limit)
    p2_out = np.zeros(limit)
    mask = counts_raw > 0
    
    p0_out[mask] = proj_0_raw[mask] / counts_raw[mask]
    p2_out[mask] = proj_2_raw[mask] / counts_raw[mask]
    
    return p0_out, p2_out

# ==========================================
# 3. 核心求解器
# ==========================================
class AnisoGradientFitter:
    def __init__(self, n_pixels, grad_weight=50.0):
        self.n = n_pixels
        self.radius = n_pixels // 2
        # r_grid 长度严格等于 radius + 1
        self.r_grid = np.arange(self.radius + 1, dtype=float)
        self.grad_weight = grad_weight

    def _forward_iso_vectorized(self, r_axis, params, n_peaks):
        p = params.reshape(n_peaks, 3)
        r0 = p[:, 0][:, None]
        sig = p[:, 1][:, None]
        amp = p[:, 2][:, None]
        sig = np.maximum(sig, 0.5)
        
        delta = r_axis[None, :] - r0
        f_r = np.sum(amp * np.exp(-(delta**2) / (2 * sig**2)), axis=0)
        
        proj = abel.hansenlaw.hansenlaw_transform(f_r, direction='forward', dr=1)
        
        n = len(r_axis)
        if len(proj) > n: proj = proj[:n]
        elif len(proj) < n: proj = np.pad(proj, (0, n-len(proj)))
        return proj

    def _loss_iso(self, params, x, y_data, y_grad_stat, n_peaks):
        y_model = self._forward_iso_vectorized(x, params, n_peaks)
        y_model_grad = signal.savgol_filter(np.gradient(y_model), 11, 2)
        
        # 物理权重
        weights = 1.0 / (np.sqrt(np.abs(y_data)) + 1.0)
        
        res_E = (y_model - y_data) * weights
        res_G = (y_model_grad - y_grad_stat) * self.grad_weight
        
        return np.concatenate([res_E, res_G])

    def solve(self, image_2d):
        # Step 1: 提取投影 (长度已修复为 radius+1)
        proj_0, proj_2 = extract_angular_projections(image_2d)
        
        # Step 2: 统计梯度侦察
        grad_stat = signal.savgol_filter(np.gradient(proj_0), window_length=25, polyorder=2)
        neg_deriv = np.maximum(-grad_stat, 0)
        
        # 找峰
        peak_idxs, _ = signal.find_peaks(neg_deriv, height=np.max(neg_deriv)*0.02, distance=8)
        n_peaks = len(peak_idxs)
        
        if n_peaks == 0: return [], self.r_grid, np.zeros_like(self.r_grid)

        # Step 3: 拟合 P0
        p0 = []
        lb = []
        ub = []
        for idx in peak_idxs:
            # 这里的 idx 不会再越界了
            r_val = self.r_grid[idx]
            sig_guess = max(4.0, 300.0/(r_val+10))
            amp_guess = proj_0[idx] / 10.0
            p0.extend([r_val, sig_guess, amp_guess])
            lb.extend([r_val - 12, 0.5, 0.0])
            ub.extend([r_val + 12, 100.0, np.inf])

        try:
            res_iso = least_squares(
                self._loss_iso, p0, 
                args=(self.r_grid, proj_0, grad_stat, n_peaks), 
                bounds=(lb, ub), ftol=1e-6
            )
            p_iso = res_iso.x
        except:
            return [], self.r_grid, np.zeros_like(self.r_grid)

        # Step 4: 估算 Beta
        final_params = []
        recon_profile = np.zeros_like(self.r_grid)
        
        for i in range(n_peaks):
            r0, sig, amp = p_iso[i*3:(i+1)*3]
            
            idx_int = int(r0)
            if idx_int < len(proj_2):
                denom = proj_0[idx_int] if proj_0[idx_int] > 1e-9 else 1.0
                ratio = proj_2[idx_int] / denom
                # Beta 校正因子
                beta_est = ratio * 5.0 
            else:
                beta_est = 0
            
            beta_est = np.clip(beta_est, -1.0, 2.0)

            final_params.append({
                'r': r0, 'sigma': sig, 'fwhm': 2.355*sig, 'amp': amp, 'beta': beta_est
            })
            
            if amp > 1e-6:
                recon_profile += amp * np.exp(-((self.r_grid - r0)**2)/(2*sig**2))

        return final_params, self.r_grid, recon_profile

# ==========================================
# 5. 重建2D图像
# ==========================================
def reconstruct_2d_image(params, n_pixels):
    """
    根据提取的参数重建2D图像
    """
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y = y - center
    x = x - center
    r_grid = np.sqrt(x**2 + y**2)
    
    # 安全计算 cos(theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    
    # P2 Legendre 多项式
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    img_recon = np.zeros_like(r_grid, dtype=float)
    
    for p in params:
        radial_term = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        angular_term = 1 + p['beta'] * P2
        img_recon += radial_term * angular_term
        
    return img_recon

# ==========================================
# 6. 从径向分布提取峰参数
# ==========================================
def extract_peaks_from_radial(r_axis, intensity):
    """
    从径向分布中提取峰的位置、幅度和FWHM
    """
    # 找到峰值
    peaks, properties = find_peaks(intensity, height=np.max(intensity)*0.1, distance=10)
    
    extracted_params = []
    for i, peak_idx in enumerate(peaks):
        peak_r = r_axis[peak_idx]
        peak_amp = intensity[peak_idx]
        
        # 计算FWHM - 找到半最大值点
        half_max = peak_amp / 2.0
        
        # 向左找半最大值点
        left_idx = peak_idx
        while left_idx > 0 and intensity[left_idx] > half_max:
            left_idx -= 1
        
        # 向右找半最大值点
        right_idx = peak_idx
        while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
            right_idx += 1
        
        # 插值计算精确的半最大值点
        if left_idx >= 0 and right_idx < len(r_axis):
            fwhm = r_axis[right_idx] - r_axis[left_idx]
        else:
            fwhm = 0  # 如果找不到合适的点，则设为0
        
        extracted_params.append({
            'r': peak_r,
            'amp': peak_amp,
            'fwhm': fwhm
        })
    
    return extracted_params

# ==========================================
# 4. 对比评估
# ==========================================
def compare_methods():
    N = 601
    PHOTONS = 1e9 
    BG_SIGMA = 1.0 

    print(f"Running Test: Photons={PHOTONS}, BgNoise={BG_SIGMA}")
    
    true_peaks = [
        {'r': 120.0, 'sigma': 8.0, 'amp': 1.0, 'beta': 1.8}, 
        {'r': 200.0, 'sigma': 6.0, 'amp': 0.8, 'beta': 0.0}, 
        {'r': 220.0, 'sigma': 6.0, 'amp': 0.7, 'beta': -0.5},
    ]
    
    # 创建真实的3D图像用于比较
    true_3d_img = np.zeros((N, N))
    y, x = np.ogrid[:N, :N]
    center = N // 2
    y = y - center
    x = x - center
    r_grid = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    for p in true_peaks:
        radial_term = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        angular_term = 1 + p['beta'] * P2
        true_3d_img += radial_term * angular_term
    
    # 1. 生成数据
    img_input = generate_vmi_image(N, true_peaks, PHOTONS, BG_SIGMA)
    
    # 2. rBasex
    t0 = time.time()
    out_rb = abel.Transform(img_input, method='rbasex', direction='inverse', verbose=False)
    img_rb_2d = out_rb.transform
    t_rb = time.time() - t0
    
    # 计算 rBasex 径向分布
    r_rb, I_rb = abel.tools.vmi.angular_integration_3D(img_rb_2d, origin=(N//2, N//2), dr=1)
    if np.max(I_rb) > 0: I_rb /= np.max(I_rb)
    
    # 提取rBasex的峰参数
    rb_params = extract_peaks_from_radial(r_rb, I_rb)

    # 3. Yours
    solver = AnisoGradientFitter(N, grad_weight=4.0)
    t0 = time.time()
    params_my, r_my, I_my = solver.solve(img_input)
    if np.max(I_my) > 0: I_my /= np.max(I_my)
    t_my = time.time() - t0
    
    # 重建我的2D图像
    img_my_2d = reconstruct_2d_image(params_my, N)
    
    # 4. 指标对比
    print("\n" + "="*140)
    print(f"{'True R':<8} | {'True Amp':<8} | {'True FWHM':<9} | {'True Beta':<9} || "
          f"{'rBasex R':<10} | {'rBasex Amp':<10} | {'rBasex FWHM':<10} || "
          f"{'Yours R':<10} | {'Yours Amp':<10} | {'Yours FWHM':<10} | {'Yours Beta':<10}")
    print("-" * 140)
    
    # 获取最大真实幅度用于归一化
    max_true_amp = max(p['amp'] for p in true_peaks)
    
    for tp in true_peaks:
        tr = tp['r']
        t_amp = tp['amp']
        t_fwhm = 2.355 * tp['sigma']
        t_beta = tp['beta']
        
        # rBasex 测量 - 寻找最接近的峰
        rb_param_match = None
        min_rb_dist = float('inf')
        for rb_p in rb_params:
            dist = abs(rb_p['r'] - tr)
            if dist < min_rb_dist and dist < 15:  # 只考虑距离小于15的匹配
                min_rb_dist = dist
                rb_param_match = rb_p
        
        if rb_param_match:
            rb_r_val = rb_param_match['r']
            rb_amp_val = rb_param_match['amp'] * max_true_amp  # 反归一化
            rb_fwhm_val = rb_param_match['fwhm']
        else:
            rb_r_val, rb_amp_val, rb_fwhm_val = np.nan, np.nan, np.nan
            
        # Yours 测量
        best_p = None
        min_d = 999
        for p in params_my:
            d = abs(p['r'] - tr)
            if d < min_d:
                min_d = d
                best_p = p
        
        if best_p and min_d < 15:
            my_r = best_p['r']
            my_amp = best_p['amp']
            my_fwhm = best_p['fwhm']
            my_b = best_p['beta']
        else:
            my_r, my_amp, my_fwhm, my_b = np.nan, np.nan, np.nan, np.nan
            
        def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "-"
        print(f"{tr:<8.1f} | {t_amp:<8.2f} | {t_fwhm:<9.2f} | {t_beta:<9.2f} || "
              f"{fmt(rb_r_val):<10} | {fmt(rb_amp_val):<10} | {fmt(rb_fwhm_val):<10} || "
              f"{fmt(my_r):<10} | {fmt(my_amp):<10} | {fmt(my_fwhm):<10} | {fmt(my_b):<10}")
        
    print("="*140)
    print(f"Time: rBasex={t_rb*1000:.1f}ms, Yours={t_my*1000:.1f}ms")

    # 5. 绘图 - 创建一个更大的子图布局来包含所有对比
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 第一行：输入图像、投影图、真实3D图像和重建图像对比
    im1 = axes[0, 0].imshow(img_input, cmap='gray', vmin=0, vmax=np.percentile(img_input, 99))
    axes[0, 0].set_title(f"Input Projection Image\n(Photons={PHOTONS})")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 显示算法输入的投影图（径向投影）
    proj_0, proj_2 = extract_angular_projections(img_input)
    axes[0, 1].plot(proj_0, 'b-', label='P0 (Isotropic)', linewidth=2)
    axes[0, 1].plot(proj_2, 'r--', label='P2 (Anisotropic)', linewidth=2)
    axes[0, 1].set_title("Algorithm Input: Radial Projections")
    axes[0, 1].set_xlabel("Radius (pixels)")
    axes[0, 1].set_ylabel("Intensity")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    im2 = axes[0, 2].imshow(true_3d_img, cmap='gray', vmin=0, vmax=np.percentile(true_3d_img, 99))
    axes[0, 2].set_title("True 3D Image (Ground Truth)")
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 重建图像与原图像的差值
    diff_rb = np.abs(img_rb_2d - true_3d_img)
    diff_my = np.abs(img_my_2d - true_3d_img)
    
    # 创建一个组合差值图
    combined_diff = np.zeros((true_3d_img.shape[0], true_3d_img.shape[1]*2))
    combined_diff[:, :true_3d_img.shape[1]] = diff_rb
    combined_diff[:, true_3d_img.shape[1]:] = diff_my
    
    im3 = axes[0, 3].imshow(combined_diff, cmap='hot', vmin=0, vmax=np.percentile(combined_diff, 99))
    axes[0, 3].set_title("Reconstruction Errors\n(Left: rBasex, Right: Ours)")
    plt.colorbar(im3, ax=axes[0, 3])
    
    # 第二行：重建图像对比
    im4 = axes[1, 0].imshow(img_rb_2d, cmap='gray', vmin=0, vmax=np.percentile(img_rb_2d, 99))
    axes[1, 0].set_title("rBasex Reconstructed 2D")
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(img_my_2d, cmap='gray', vmin=0, vmax=np.percentile(img_my_2d, 99))
    axes[1, 1].set_title("Our Method Reconstructed 2D")
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 径向分布比较
    f_true = np.zeros_like(r_my)
    for p in true_peaks: f_true += p['amp'] * np.exp(-((r_my - p['r'])**2)/(2*p['sigma']**2))
    f_true = f_true / np.max(f_true)
    
    axes[1, 2].plot(r_my, f_true, 'k', alpha=0.3, linewidth=5, label='Truth', zorder=1)
    axes[1, 2].plot(r_rb, I_rb, 'b--', label='rBasex', linewidth=2, zorder=2)
    axes[1, 2].plot(r_my, I_my, 'r-', linewidth=2, label='Ours', zorder=3)
    
    # 标记检测到的峰
    for p in params_my:
        axes[1, 2].axvline(x=p['r'], color='r', linestyle=':', alpha=0.5)
        axes[1, 2].text(p['r'], 0.8, f"β={p['beta']:.1f}", 
                       color='r', rotation=90, va='bottom', fontsize=8, ha='center')
    
    for p in rb_params:
        axes[1, 2].axvline(x=p['r'], color='b', linestyle=':', alpha=0.5)
        
    axes[1, 2].set_title("Radial Distribution Comparison")
    axes[1, 2].legend()
    axes[1, 2].set_xlim(80, 260)
    axes[1, 2].set_ylim(0, 1.1)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlabel("Radius (pixels)")
    axes[1, 2].set_ylabel("Normalized Intensity")
    
    # 重建质量指标
    mse_rb = np.mean((img_rb_2d - true_3d_img)**2)
    mse_my = np.mean((img_my_2d - true_3d_img)**2)
    
    metrics_text = f"Reconstruction Quality Metrics:\n\n"
    metrics_text += f"rBasex MSE: {mse_rb:.2e}\n"
    metrics_text += f"Ours MSE: {mse_my:.2e}\n"
    metrics_text += f"Improvement: {(mse_rb - mse_my)/mse_rb*100:.1f}%\n\n"
    metrics_text += f"rBasex Time: {t_rb*1000:.1f}ms\n"
    metrics_text += f"Ours Time: {t_my*1000:.1f}ms"
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    axes[1, 3].text(0.05, 0.95, metrics_text, transform=axes[1, 3].transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    axes[1, 3].axis('off')
    axes[1, 3].set_title("Quality Metrics")
    
    # 第三行：详细参数对比表格
    # 创建详细的参数对比
    param_comparison = "Detailed Parameter Comparison:\n\n"
    param_comparison += f"{'True':<8} {'rBasex':<10} {'Ours':<10} {'True':<8} {'rBasex':<10} {'Ours':<10}\n"
    param_comparison += f"{'R':<8} {'R':<10} {'R':<10} {'Amp':<8} {'Amp':<10} {'Amp':<10}\n"
    param_comparison += "-" * 60 + "\n"
    
    for i, tp in enumerate(true_peaks):
        tr = tp['r']
        t_amp = tp['amp']
        t_sigma = tp['sigma']
        t_fwhm = 2.355 * t_sigma
        t_beta = tp['beta']
        
        # rBasex匹配
        rb_param_match = None
        min_rb_dist = float('inf')
        for rb_p in rb_params:
            dist = abs(rb_p['r'] - tr)
            if dist < min_rb_dist and dist < 15:
                min_rb_dist = dist
                rb_param_match = rb_p
        
        rb_r_val = rb_param_match['r'] if rb_param_match else np.nan
        rb_amp_val = rb_param_match['amp'] * max_true_amp if rb_param_match else np.nan
        rb_fwhm_val = rb_param_match['fwhm'] if rb_param_match else np.nan
        
        # Ours匹配
        best_p = None
        min_d = 999
        for p in params_my:
            d = abs(p['r'] - tr)
            if d < min_d:
                min_d = d
                best_p = p
        
        my_r = best_p['r'] if best_p and min_d < 15 else np.nan
        my_amp = best_p['amp'] if best_p and min_d < 15 else np.nan
        my_fwhm = best_p['fwhm'] if best_p and min_d < 15 else np.nan
        my_beta = best_p['beta'] if best_p and min_d < 15 else np.nan
        
        def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "-"
        
        if i == 0:  # 第一行显示R和Amp
            param_comparison += f"{tr:<8.1f} {fmt(rb_r_val):<10} {fmt(my_r):<10} {t_amp:<8.2f} {fmt(rb_amp_val):<10} {fmt(my_amp):<10}\n"
        elif i == 1:  # 第二行显示Sigma和Beta
            param_comparison += f"{'Sigma':<8} {'FWHM':<10} {'FWHM':<10} {'Beta':<8} {'-':<10} {fmt(my_beta):<10}\n"
            param_comparison += f"{t_sigma:<8.1f} {fmt(rb_fwhm_val):<10} {fmt(my_fwhm):<10} {t_beta:<8.2f} {'-':<10} {fmt(my_beta):<10}\n"
    
    # 显示完整参数对比
    full_params = "Complete Extracted Parameters:\n\n"
    full_params += "True Parameters:\n"
    for i, p in enumerate(true_peaks):
        full_params += f"  Peak {i+1}: R={p['r']:.1f}, Amp={p['amp']:.2f}, Sigma={p['sigma']:.1f}, Beta={p['beta']:.2f}\n"
    
    full_params += "\nrBasex Parameters:\n"
    for i, p in enumerate(rb_params):
        full_params += f"  Peak {i+1}: R={p['r']:.1f}, Amp={p['amp']:.2f}, FWHM={p['fwhm']:.1f}\n"
    
    full_params += "\nOur Method Parameters:\n"
    for i, p in enumerate(params_my):
        full_params += f"  Peak {i+1}: R={p['r']:.1f}, Amp={p['amp']:.2f}, FWHM={p['fwhm']:.1f}, Beta={p['beta']:.2f}\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[2, 0].text(0.05, 0.95, param_comparison, transform=axes[2, 0].transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')
    axes[2, 0].axis('off')
    axes[2, 0].set_title("Parameter Comparison")
    
    axes[2, 1].text(0.05, 0.95, full_params, transform=axes[2, 1].transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
    axes[2, 1].axis('off')
    axes[2, 1].set_title("Complete Parameters")
    
    # 创建误差分析图
    error_types = ['MSE', 'MAE', 'Max Error']
    rb_errors = [mse_rb, np.mean(np.abs(img_rb_2d - true_3d_img)), np.max(np.abs(img_rb_2d - true_3d_img))]
    my_errors = [mse_my, np.mean(np.abs(img_my_2d - true_3d_img)), np.max(np.abs(img_my_2d - true_3d_img))]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    axes[2, 2].bar(x - width/2, rb_errors, width, label='rBasex', alpha=0.7, color='blue')
    axes[2, 2].bar(x + width/2, my_errors, width, label='Ours', alpha=0.7, color='red')
    axes[2, 2].set_xlabel('Error Type')
    axes[2, 2].set_ylabel('Error Value')
    axes[2, 2].set_title('Reconstruction Error Comparison')
    axes[2, 2].set_xticks(x)
    axes[2, 2].set_xticklabels(error_types)
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # 峰检测准确性分析
    true_r_values = [p['r'] for p in true_peaks]
    rb_detected_r = [p['r'] for p in rb_params]
    my_detected_r = [p['r'] for p in params_my]
    
    axes[2, 3].scatter(true_r_values, [0]*len(true_r_values), s=100, c='black', marker='o', label='True', zorder=3)
    axes[2, 3].scatter(rb_detected_r, [0.5]*len(rb_detected_r), s=80, c='blue', marker='s', label='rBasex', zorder=2)
    axes[2, 3].scatter(my_detected_r, [1]*len(my_detected_r), s=80, c='red', marker='^', label='Ours', zorder=2)
    
    axes[2, 3].set_xlabel('Radius (pixels)')
    axes[2, 3].set_ylabel('Method')
    axes[2, 3].set_title('Peak Detection Accuracy')
    axes[2, 3].set_yticks([0, 0.5, 1])
    axes[2, 3].set_yticklabels(['Truth', 'rBasex', 'Ours'])
    axes[2, 3].set_xlim(50, 300)
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods()


#improve performance in low flux
#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy import signal
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import map_coordinates
from abel.tools.vmi import angular_integration_3D

# ==========================================
# 1. 物理场模拟
# ==========================================
def generate_vmi_image(n_pixels, peaks, total_photons, bg_sigma):
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y = y - center
    x = x - center
    r_grid = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    img_3d = np.zeros_like(r_grid, dtype=float)
    
    for p in peaks:
        radial_term = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        angular_term = 1 + p['beta'] * P2
        img_3d += radial_term * angular_term

    proj_clean = abel.Transform(img_3d, method='hansenlaw', direction='forward', verbose=False).transform
    
    if np.sum(proj_clean) == 0: return proj_clean
    prob = proj_clean / np.sum(proj_clean)
    img_counts = np.random.poisson(prob * total_photons)
    bg_noise = np.random.normal(0, bg_sigma, proj_clean.shape)
    
    return np.maximum(img_counts + bg_noise, 0)

# ==========================================
# 2. 核心求解器
# ==========================================
class PhysicsBasedFitter:
    def __init__(self, n_pixels):
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid_1d = np.arange(self.radius + 1, dtype=float)
        
        y, x = np.ogrid[:n_pixels, :n_pixels]
        self.Y = y - n_pixels // 2
        self.X = x - n_pixels // 2
        self.R2 = self.X**2 + self.Y**2
        self.R = np.sqrt(self.R2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.COS_THETA = self.X / self.R
        self.COS_THETA[~np.isfinite(self.COS_THETA)] = 0.0
        self.P2_GRID = 0.5 * (3 * self.COS_THETA**2 - 1)

    def _phase1_radial_analysis(self, image_2d):
        print("Phase 1: Radial Analysis...")
        recon = abel.Transform(image_2d, method='hansenlaw', direction='inverse', verbose=False).transform
        r_dist, intensity = abel.tools.vmi.angular_integration_3D(recon, origin=(self.n//2, self.n//2), dr=1)
        intensity_smooth = signal.savgol_filter(intensity, 11, 3)
        peaks, _ = find_peaks(intensity_smooth, height=np.max(intensity_smooth)*0.05, distance=8)
        
        initial_guesses = []
        for p_idx in peaks:
            initial_guesses.append({
                'r': r_dist[p_idx], 'sigma': 4.0, 'amp': intensity_smooth[p_idx]
            })
        return recon, initial_guesses

    def _phase2_angular_analysis(self, recon_image, partial_params):
        print("Phase 2: Angular Analysis (FFT Filtering)...")
        cy, cx = self.n // 2, self.n // 2
        updated_params = []
        thetas = np.linspace(0, 2*np.pi, 256, endpoint=False)
        
        for p in partial_params:
            r_center = p['r']
            sample_x = cx + r_center * np.cos(thetas)
            sample_y = cy + r_center * np.sin(thetas)
            angular_profile = map_coordinates(recon_image, [sample_y, sample_x], order=1, mode='wrap')
            
            fft_vals = np.fft.rfft(angular_profile)
            dc_comp = np.abs(fft_vals[0])
            w2_comp = np.abs(fft_vals[2])
            
            if dc_comp > 1e-6:
                R_phys = 2.0 * w2_comp / dc_comp
                denom = 3.0 - R_phys
                beta_est = 4.0 * R_phys / denom if denom > 0.1 else 2.0
            else:
                beta_est = 0.0
            
            p['beta'] = np.clip(beta_est, -1.0, 2.0)
            updated_params.append(p)
        return updated_params

    def _forward_model_loss(self, params_flat, image_target, n_peaks):
        params = params_flat.reshape(n_peaks, 4)
        img_3d_model = np.zeros((self.n, self.n))
        for i in range(n_peaks):
            r0, sig, amp, beta = params[i]
            sig = max(sig, 0.5) 
            radial = amp * np.exp(-((self.R - r0)**2) / (2 * sig**2))
            angular = 1 + beta * self.P2_GRID
            img_3d_model += radial * angular
            
        proj_model = abel.Transform(img_3d_model, method='hansenlaw', direction='forward', verbose=False).transform
        weights = 1.0 / np.sqrt(np.abs(image_target) + 1.0)
        return ((proj_model - image_target) * weights).ravel()

    def solve(self, image_2d):
        t0 = time.time()
        recon_img, init_params = self._phase1_radial_analysis(image_2d)
        if not init_params: return [], self.r_grid_1d, np.zeros_like(self.r_grid_1d)
            
        full_init_params = self._phase2_angular_analysis(recon_img, init_params)
        n_peaks = len(full_init_params)
        x0, lb, ub = [], [], []
        
        print(f"Phase 3: Joint Optimization ({n_peaks} peaks)...")
        for p in full_init_params:
            x0.extend([p['r'], p['sigma'], p['amp'], p['beta']])
            lb.extend([p['r']-8, 0.5, 0.0, -1.1])
            ub.extend([p['r']+8, 50.0, np.inf, 2.1])
            
        try:
            res = least_squares(
                self._forward_model_loss, x0=np.array(x0), bounds=(np.array(lb), np.array(ub)),
                args=(image_2d, n_peaks), method='trf', ftol=1e-3, xtol=1e-3, max_nfev=30
            )
            final_x = res.x
        except Exception as e:
            print(f"Opt warning: {e}")
            final_x = np.array(x0)

        final_params = []
        recon_profile = np.zeros_like(self.r_grid_1d)
        p_reshaped = final_x.reshape(n_peaks, 4)
        for i in range(n_peaks):
            r0, sig, amp, beta = p_reshaped[i]
            # 这里 FWHM = 2.355 * sigma
            final_params.append({'r': r0, 'sigma': sig, 'fwhm': 2.355*sig, 'amp': amp, 'beta': beta})
            recon_profile += amp * np.exp(-((self.r_grid_1d - r0)**2)/(2*sig**2))

        print(f"Solver Time: {time.time()-t0:.2f}s")
        return final_params, self.r_grid_1d, recon_profile

# ==========================================
# 辅助函数
# ==========================================
def get_rbasex_params(img_2d, r_peaks):
    try:
        beta_profile, _ = abel.tools.vmi.anisotropy_parameter(img_2d)
    except:
        return [0.0] * len(r_peaks)

    results = []
    for r_val in r_peaks:
        idx = int(r_val)
        if 0 <= idx < len(beta_profile):
            s, e = max(0, idx-2), min(len(beta_profile), idx+3)
            results.append(np.clip(np.mean(beta_profile[s:e]), -1.5, 2.5))
        else:
            results.append(0.0)
    return results

def reconstruct_2d_from_params(params, n_pixels):
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y, x = y - center, x - center
    r_grid = np.sqrt(x**2 + y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    img = np.zeros_like(r_grid)
    for p in params:
        radial = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        img += radial * (1 + p['beta'] * P2)
    return img

# ==========================================
# 3. 对比评估 (包含 FWHM)
# ==========================================
def compare_methods():
    N = 401
    PHOTONS = 1e6 
    BG_SIGMA = 2.0 

    print(f"Running Test: Size={N}, Photons={PHOTONS:.0e}")
    
    true_peaks = [
        {'r': 80.0, 'sigma': 4.0, 'amp': 1.0, 'beta': 1.8}, 
        {'r': 140.0, 'sigma': 3.5, 'amp': 0.7, 'beta': -0.6}, 
        {'r': 155.0, 'sigma': 3.5, 'amp': 0.5, 'beta': 0.2}, 
    ]
    
    img_input = generate_vmi_image(N, true_peaks, PHOTONS, BG_SIGMA)
    
    # --- rBasex ---
    t0 = time.time()
    out_rb = abel.Transform(img_input, method='rbasex', direction='inverse', verbose=False)
    img_rb_2d = out_rb.transform
    r_rb, I_rb = abel.tools.vmi.angular_integration_3D(img_rb_2d, origin=(N//2, N//2), dr=1)
    if np.max(I_rb) > 0: I_rb /= np.max(I_rb)
    
    p_idx, _ = find_peaks(I_rb, height=0.1, distance=10)
    rb_detected_r = r_rb[p_idx]
    rb_betas = get_rbasex_params(img_rb_2d, rb_detected_r)
    
    # 新增: 计算 rBasex 的 FWHM
    # peak_widths 返回 (widths, width_heights, left_ips, right_ips)
    # 我们只取 widths, 单位是样本数(dr=1时即像素)
    rb_widths_res = peak_widths(I_rb, p_idx, rel_height=0.5)
    rb_fwhms = rb_widths_res[0] 
    
    # --- Ours ---
    solver = PhysicsBasedFitter(N)
    params_my, r_my, I_my = solver.solve(img_input)
    if np.max(I_my) > 0: I_my /= np.max(I_my)
    img_my_2d = reconstruct_2d_from_params(params_my, N)
    
    # --- 计算真值径向分布 (用于画图) ---
    I_true = np.zeros_like(r_my)
    for p in true_peaks:
        I_true += p['amp'] * np.exp(-((r_my - p['r'])**2) / (2 * p['sigma']**2))
    if np.max(I_true) > 0: I_true /= np.max(I_true)

    # --- 打印详细对比表 (新增 FWHM 列) ---
    print("\n" + "="*125)
    print(f"{'COMPARISON TABLE (Metric: FWHM added)':^125}")
    print("="*125)
    # 格式化字符串
    header = f"{'Method':<8} | {'R (px)':<8} | {'Amp':<6} | {'FWHM':<6} | {'Beta':<6} | {'Err(R)':<7} | {'Err(FW)':<7} | {'Err(β)':<7}"
    print(header)
    print("-" * 125)
    
    for i, tp in enumerate(true_peaks):
        tr, tb, t_sig = tp['r'], tp['beta'], tp['sigma']
        t_fwhm = 2.355 * t_sig
        
        print(f"--- Peak {i+1} (True: R={tr:.1f}, FWHM={t_fwhm:.1f}, B={tb:.2f}) ---")
        
        # rBasex Match
        best_rb = None
        min_dist = 999
        for k, r_val in enumerate(rb_detected_r):
            dist = abs(r_val - tr)
            if dist < min_dist:
                # 存储元组: (R, Beta, FWHM)
                min_dist = dist
                best_rb = (r_val, rb_betas[k], rb_fwhms[k])
        
        if best_rb and min_dist < 10:
            err_r = abs(best_rb[0] - tr)
            err_fw = abs(best_rb[2] - t_fwhm)
            err_b = abs(best_rb[1] - tb)
            print(f"{'rBasex':<8} | {best_rb[0]:<8.1f} | {'-':<6} | {best_rb[2]:<6.1f} | {best_rb[1]:<6.2f} | {err_r:<7.2f} | {err_fw:<7.2f} | {err_b:<7.2f}")
        else:
            print(f"{'rBasex':<8} | {'Missed':<8} | ...")
            
        # Ours Match
        best_my = None
        min_dist = 999
        for p in params_my:
            dist = abs(p['r'] - tr)
            if dist < min_dist: 
                min_dist = dist
                best_my = p
        
        if best_my and min_dist < 10:
            err_r = abs(best_my['r'] - tr)
            err_fw = abs(best_my['fwhm'] - t_fwhm)
            err_b = abs(best_my['beta'] - tb)
            print(f"{'Ours':<8} | {best_my['r']:<8.1f} | {best_my['amp']:<6.2f} | {best_my['fwhm']:<6.1f} | {best_my['beta']:<6.2f} | {err_r:<7.2f} | {err_fw:<7.2f} | {err_b:<7.2f}")
        else:
            print(f"{'Ours':<8} | {'Missed':<8} | ...")

    print("-" * 125)
    # --- 峰数量统计 ---
    print(f"\n[Peak Detection Summary]")
    print(f"Ground Truth : {len(true_peaks)} peaks")
    print(f"rBasex Found : {len(rb_detected_r)} peaks")
    print(f"Ours Found   : {len(params_my)} peaks")
    print("="*125)
    
    # --- 绘图 (保持要求) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 第一行
    vmax = np.percentile(img_input, 99.5)
    im1 = axes[0,0].imshow(img_input, cmap='gray', vmin=0, vmax=vmax)
    axes[0,0].set_title(f"Input Raw Photoelectron Image\n(Photons={PHOTONS:.0e})")
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    vmax_rec = np.percentile(img_rb_2d, 99.5)
    im2 = axes[0,1].imshow(img_rb_2d, cmap='hot', vmin=0, vmax=vmax_rec)
    axes[0,1].set_title("rBasex Reconstruction (Slice)")
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    im3 = axes[0,2].imshow(img_my_2d, cmap='hot', vmin=0, vmax=vmax_rec)
    axes[0,2].set_title("Ours Reconstruction (Slice)")
    plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)

    # 第二行
    axes[1,0].plot(r_my, I_true, 'k-', linewidth=3, alpha=0.6, label='Ground Truth')
    axes[1,0].plot(r_rb, I_rb, 'b--', linewidth=1.5, label='rBasex')
    axes[1,0].plot(r_my, I_my, 'r-', linewidth=1.5, label='Ours')
    axes[1,0].set_title("Radial Distribution Comparison")
    axes[1,0].set_xlabel("Radius (px)")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter([p['r'] for p in true_peaks], [p['beta'] for p in true_peaks], 
                      s=150, c='k', marker='o', label='True', zorder=10)
    valid_idx = [k for k, r in enumerate(rb_detected_r) if abs(r - N//2) > 10]
    if valid_idx:
        axes[1,1].scatter(rb_detected_r[valid_idx], np.array(rb_betas)[valid_idx], 
                          s=80, c='b', marker='s', label='rBasex', alpha=0.6)
    axes[1,1].scatter([p['r'] for p in params_my], [p['beta'] for p in params_my], 
                      s=80, c='r', marker='^', label='Ours', alpha=0.9)
    axes[1,1].set_ylim(-1.5, 2.5)
    axes[1,1].set_xlabel("Radius (px)")
    axes[1,1].set_ylabel("Beta")
    axes[1,1].set_title("Beta Parameter Accuracy")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].axis('off')
    summary_text = f"Performance Summary:\n\n"
    summary_text += f"Ground Truth Peaks: {len(true_peaks)}\n"
    summary_text += f"rBasex Detected:    {len(rb_detected_r)}\n"
    summary_text += f"Ours Detected:      {len(params_my)}\n\n"
    summary_text += "Metric Added: FWHM (Full Width Half Max)\n"
    summary_text += "- Evaluates how well the peak width\n  is resolved.\n"
    summary_text += "- True FWHM = 2.355 * Sigma"
    
    axes[1,2].text(0.1, 0.9, summary_text, fontsize=12, va='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods()

#very great improvement
#----------------------
import numpy as np
import matplotlib.pyplot as plt
import abel
import time
from scipy import signal
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import map_coordinates
from abel.tools.vmi import angular_integration_3D

# ==========================================
# 1. 物理场模拟
# ==========================================
def generate_vmi_image(n_pixels, peaks, total_photons, bg_sigma):
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y = y - center
    x = x - center
    r_grid = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    img_3d = np.zeros_like(r_grid, dtype=float)
    
    for p in peaks:
        radial_term = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        angular_term = 1 + p['beta'] * P2
        img_3d += radial_term * angular_term

    proj_clean = abel.Transform(img_3d, method='hansenlaw', direction='forward', verbose=False).transform
    
    if np.sum(proj_clean) == 0: return proj_clean
    prob = proj_clean / np.sum(proj_clean)
    img_counts = np.random.poisson(prob * total_photons)
    bg_noise = np.random.normal(0, bg_sigma, proj_clean.shape)
    
    return np.maximum(img_counts + bg_noise, 0)

# ==========================================
# 2. 核心求解器
# ==========================================
class PhysicsBasedFitter:
    def __init__(self, n_pixels):
        # ... (保持不变) ...
        self.n = n_pixels
        self.radius = n_pixels // 2
        self.r_grid_1d = np.arange(self.radius + 1, dtype=float)
        
        y, x = np.ogrid[:n_pixels, :n_pixels]
        self.Y = y - n_pixels // 2
        self.X = x - n_pixels // 2
        self.R2 = self.X**2 + self.Y**2
        self.R = np.sqrt(self.R2)
        
        self.r_indices = self.R.astype(int)
        self.max_r_idx = int(np.max(self.R))
        self.r_flat = self.r_indices.ravel()
        
        self.pixel_counts = np.bincount(self.r_flat, minlength=self.max_r_idx + 1)
        self.pixel_counts[self.pixel_counts == 0] = 1 
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.COS_THETA = self.X / self.R
        self.COS_THETA[~np.isfinite(self.COS_THETA)] = 0.0
        self.P2_GRID = 0.5 * (3 * self.COS_THETA**2 - 1)

    def _compute_full_radial_profile(self, img_2d):
        """
        新增辅助函数：计算完整的径向分布（包含角落区域），用于噪声估计
        """
        radial_sum = np.bincount(self.r_flat, weights=img_2d.ravel(), minlength=self.max_r_idx + 1)
        # pixel_counts 已在 init 中计算到 max_r_idx
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = radial_sum / self.pixel_counts
        profile[~np.isfinite(profile)] = 0
        return profile

    def _compute_radial_profile(self, img_2d):
        # 保持兼容性，只返回有效半径部分
        full_prof = self._compute_full_radial_profile(img_2d)
        return full_prof[:len(self.r_grid_1d)]

    def _adaptive_spectral_filter(self, full_profile, valid_radius):
        """
        改进 0: 基于噪声先验的自适应频域滤波
        """
        # 1. 分离信号区与噪声区 (先验估计)
        # valid_radius 对应图像内切圆半径
        # 图像四角 (Corners) 的数据 r > valid_radius 视为纯噪声
        noise_region = full_profile[valid_radius:]
        signal_region = full_profile[:valid_radius]
        
        # 2. 估计基线 (Baseline)
        if len(noise_region) > 10:
            # 使用噪声区的平均值作为更准确的 DC 基线
            baseline = np.mean(noise_region)
        else:
            baseline = np.min(signal_region) # 回退方案
            
        sig_corr = signal_region - baseline
        
        # 3. 频域分析
        n_points = len(sig_corr)
        fft_sig = np.fft.rfft(sig_corr)
        power_sig = np.abs(fft_sig)**2
        
        # 4. 估计噪声基底 (Noise Floor)
        # 即使在信号区，高频部分通常也是由噪声主导。
        # 我们可以结合 corner noise 的方差来校验，但直接用信号的高频尾部更稳健。
        # 取最后 25% 的频率分量平均能量作为噪声基底估计
        high_freq_idx = int(len(power_sig) * 0.75)
        noise_floor = np.mean(power_sig[high_freq_idx:])
        
        # 5. 构建 Wiener-like 滤波器
        # H(f) = P_signal(f) / (P_signal(f) + alpha * P_noise)
        # 这种滤波器能自动保留强信号频率，压制接近底噪的频率
        alpha = 5.0 # 激进一点的去噪系数，保证干净
        filter_gain = power_sig / (power_sig + alpha * noise_floor + 1e-12)
        
        # 平滑滤波器增益曲线，防止频域突变产生时域振铃
        filter_gain = signal.savgol_filter(filter_gain, window_length=5, polyorder=2)
        filter_gain = np.clip(filter_gain, 0, 1)
        
        # 6. 应用滤波
        filtered_fft = fft_sig * filter_gain
        filtered_sig = np.fft.irfft(filtered_fft, n=n_points)
        
        # 恢复基线并保证非负
        return np.maximum(filtered_sig + baseline, 0)

    def _phase1_radial_analysis(self, image_2d):
        print("Phase 1: Radial Analysis (Adaptive Filtering & Sigma Est)...")
        
        # 1. 获取包含角落的完整分布
        full_profile = self._compute_full_radial_profile(image_2d)
        valid_len = len(self.r_grid_1d)
        
        # 2. 【改进 0】基于先验的自适应滤波
        radial_profile_clean = self._adaptive_spectral_filter(full_profile, valid_len)
        
        # 3. 1D Abel 逆变换
        recon_1d = abel.hansenlaw.hansenlaw_transform(
            radial_profile_clean, 
            direction='inverse'
        )

        # 4. 中心屏蔽 (Center Masking)
        mask_radius = 12
        recon_1d[:mask_radius] = 0 
        
        # 5. 二次平滑与 【改进 2】非负约束
        # 窗口可以小一点，因为频域已经去噪了
        intensity_smooth = signal.savgol_filter(recon_1d, 15, 3)
        intensity_smooth[:mask_radius] = 0
        intensity_smooth = np.maximum(intensity_smooth, 0) # Non-negative constraint
        
        # 6. 寻峰
        max_val = np.max(intensity_smooth)
        peaks, properties = find_peaks(intensity_smooth, 
                              height=max_val * 0.05,  # 降低一点阈值，相信滤波效果
                              distance=8, 
                              prominence=max_val * 0.03)
        
        # 【改进 1】Sigma 估计
        # 使用 peak_widths 计算 FWHM，进而推算 sigma
        # rel_height=0.5 即半高全宽
        if len(peaks) > 0:
            widths_res = peak_widths(intensity_smooth, peaks, rel_height=0.5)
            # widths_res[0] 是宽度数组
            calculated_sigmas = widths_res[0] / 2.355
        else:
            calculated_sigmas = []

        initial_guesses = []
        for i, p_idx in enumerate(peaks):
            # 获取计算出的 sigma，如果计算失败或异常则回退到 4.0
            est_sigma = calculated_sigmas[i] if i < len(calculated_sigmas) else 4.0
            est_sigma = max(1.0, est_sigma) # 物理限制
            
            initial_guesses.append({
                'r': float(p_idx), 
                'sigma': float(est_sigma),  # 使用估计值
                'amp': intensity_smooth[p_idx]
            })
            
        return initial_guesses

    # ... (Phase 2, Forward Model, Solve 等其他方法保持不变) ...
    def _phase2_angular_analysis(self, image_raw, partial_params):
        return super(PhysicsBasedFitter, self)._phase2_angular_analysis(image_raw, partial_params) if hasattr(super(PhysicsBasedFitter, self), '_phase2_angular_analysis') else self._phase2_impl(image_raw, partial_params)

    # 这里的 helper 仅为了完整性，实际并未改动逻辑
    def _phase2_impl(self, image_raw, partial_params):
        cy, cx = self.n // 2, self.n // 2
        updated_params = []
        thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
        for p in partial_params:
            r_center = p['r']
            if r_center < 10 or r_center >= self.radius - 1: continue
            sample_x = cx + r_center * np.cos(thetas)
            sample_y = cy + r_center * np.sin(thetas)
            angular_profile = map_coordinates(image_raw, [sample_y, sample_x], order=1, mode='wrap')
            fft_vals = np.fft.rfft(angular_profile)
            dc_comp = np.abs(fft_vals[0])
            w2_comp = np.abs(fft_vals[2])
            if dc_comp > 1e-6:
                R_val = 2.0 * w2_comp / dc_comp
                denom = 3.0 - R_val
                beta_est = 4.0 * R_val / denom if denom > 0.1 else 2.0
            else:
                beta_est = 0.0
            p['beta'] = np.clip(beta_est, -1.0, 2.0)
            updated_params.append(p)
        return updated_params

    def _forward_model_loss(self, params_flat, image_target, target_profile_1d, n_peaks):
        params = params_flat.reshape(n_peaks, 4)
        img_3d_model = np.zeros((self.n, self.n))
        for i in range(n_peaks):
            r0, sig, amp, beta = params[i]
            sig = max(sig, 1.0) 
            radial = amp * np.exp(-((self.R - r0)**2) / (2 * sig**2))
            angular = 1 + beta * self.P2_GRID
            img_3d_model += radial * angular
        proj_model = abel.Transform(img_3d_model, method='hansenlaw', direction='forward', verbose=False).transform
        weights_2d = 1.0 / np.sqrt(np.abs(image_target) + 1.0)
        res_2d = (proj_model - image_target) * weights_2d
        model_profile_1d = self._compute_radial_profile(proj_model)
        mask_radius = 12
        weights_1d = 1.0 / np.sqrt(np.abs(target_profile_1d) + 1.0)
        weights_1d[:mask_radius] = 0.0 
        res_1d = (model_profile_1d - target_profile_1d) * weights_1d
        lambda_factor = 30.0 
        return np.concatenate([res_2d.ravel(), res_1d * lambda_factor])

    def solve(self, image_2d):
        t0 = time.time()
        init_params_no_beta = self._phase1_radial_analysis(image_2d)
        if not init_params_no_beta: return [], self.r_grid_1d, np.zeros_like(self.r_grid_1d)
        full_init_params = self._phase2_impl(image_2d, init_params_no_beta)
        target_profile_1d = self._compute_radial_profile(image_2d)
        n_peaks = len(full_init_params)
        x0, lb, ub = [], [], []
        print(f"Phase 3: Joint Optimization ({n_peaks} peaks)...")
        for p in full_init_params:
            x0.extend([p['r'], p['sigma'], p['amp'], p['beta']])
            lb.extend([max(10.0, p['r']-6), 1.0, 0.0, -1.1])
            ub.extend([p['r']+6, 50.0, np.inf, 2.1])
        try:
            res = least_squares(
                self._forward_model_loss, x0=np.array(x0), bounds=(np.array(lb), np.array(ub)),
                args=(image_2d, target_profile_1d, n_peaks), 
                method='trf', ftol=1e-3, xtol=1e-3, max_nfev=30
            )
            final_x = res.x
        except Exception as e:
            print(f"Opt warning: {e}")
            final_x = np.array(x0)
        p_reshaped = final_x.reshape(n_peaks, 4)
        if n_peaks > 0:
            max_amp = np.max(p_reshaped[:, 2])
            amp_threshold = 0.05 * max_amp
        else:
            amp_threshold = 0
        final_params = []
        recon_profile = np.zeros_like(self.r_grid_1d)
        for i in range(n_peaks):
            r0, sig, amp, beta = p_reshaped[i]
            if amp < amp_threshold or r0 < 10: continue
            final_params.append({'r': r0, 'sigma': sig, 'fwhm': 2.355*sig, 'amp': amp, 'beta': beta})
            recon_profile += amp * np.exp(-((self.r_grid_1d - r0)**2)/(2*sig**2))
        print(f"Solver Time: {time.time()-t0:.2f}s")
        return final_params, self.r_grid_1d, recon_profile

# ==========================================
# 辅助函数
# ==========================================
def get_rbasex_params(img_2d, r_peaks):
    try:
        beta_profile, _ = abel.tools.vmi.anisotropy_parameter(img_2d)
    except:
        return [0.0] * len(r_peaks)

    results = []
    for r_val in r_peaks:
        idx = int(r_val)
        if 0 <= idx < len(beta_profile):
            s, e = max(0, idx-2), min(len(beta_profile), idx+3)
            results.append(np.clip(np.mean(beta_profile[s:e]), -1.5, 2.5))
        else:
            results.append(0.0)
    return results

def reconstruct_2d_from_params(params, n_pixels):
    y, x = np.ogrid[:n_pixels, :n_pixels]
    center = n_pixels // 2
    y, x = y - center, x - center
    r_grid = np.sqrt(x**2 + y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r_grid
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    img = np.zeros_like(r_grid)
    for p in params:
        radial = p['amp'] * np.exp(-((r_grid - p['r'])**2) / (2 * p['sigma']**2))
        img += radial * (1 + p['beta'] * P2)
    return img

# ==========================================
# 3. 对比评估 (包含 FWHM)
# ==========================================
def compare_methods():
    N = 401
    PHOTONS = 1e6 
    BG_SIGMA = 2.0 

    print(f"Running Test: Size={N}, Photons={PHOTONS:.0e}")
    
    true_peaks = [
        {'r': 80.0, 'sigma': 4.0, 'amp': 1.0, 'beta': 1.8}, 
        {'r': 140.0, 'sigma': 3.5, 'amp': 0.7, 'beta': -0.6}, 
        {'r': 155.0, 'sigma': 3.5, 'amp': 0.5, 'beta': 0.2}, 
    ]
    
    img_input = generate_vmi_image(N, true_peaks, PHOTONS, BG_SIGMA)
    
    # --- rBasex ---
    recon_rb, distr_rb = abel.rbasex.rbasex_transform(
        img_input, direction='inverse', basis_dir=None, verbose=False
    )
    r_rb, I_rb, beta_rb = distr_rb.rIbeta()
    I_rb[0:10] = 0 
    
    if np.max(I_rb) > 0: I_rb_norm = I_rb / np.max(I_rb)
    else: I_rb_norm = I_rb
    
    p_idx, _ = find_peaks(I_rb_norm, height=0.1, distance=10)
    rb_detected_r = r_rb[p_idx]
    rb_detected_betas = beta_rb[p_idx]
    rb_widths_res = peak_widths(I_rb_norm, p_idx, rel_height=0.5)
    rb_fwhms = rb_widths_res[0] 
    
    # --- Ours ---
    solver = PhysicsBasedFitter(N)
    params_my, r_my, I_my = solver.solve(img_input)
    if np.max(I_my) > 0: I_my /= np.max(I_my)
    img_my_2d = reconstruct_2d_from_params(params_my, N)
    
    # --- 计算真值径向分布 (用于画图) ---
    I_true = np.zeros_like(r_my)
    for p in true_peaks:
        I_true += p['amp'] * np.exp(-((r_my - p['r'])**2) / (2 * p['sigma']**2))
    if np.max(I_true) > 0: I_true /= np.max(I_true)

    # --- 打印详细对比表 (新增 FWHM 列) ---
    print("\n" + "="*125)
    print(f"{'COMPARISON TABLE (Metric: FWHM added)':^125}")
    print("="*125)
    # 格式化字符串
    header = f"{'Method':<8} | {'R (px)':<8} | {'Amp':<6} | {'FWHM':<6} | {'Beta':<6} | {'Err(R)':<7} | {'Err(FW)':<7} | {'Err(β)':<7}"
    print(header)
    print("-" * 125)
    
    for i, tp in enumerate(true_peaks):
        tr, tb, t_sig = tp['r'], tp['beta'], tp['sigma']
        t_fwhm = 2.355 * t_sig
        
        print(f"--- Peak {i+1} (True: R={tr:.1f}, FWHM={t_fwhm:.1f}, B={tb:.2f}) ---")
        
        # rBasex Match
        best_rb = None
        min_dist = 999 
        for k, r_val in enumerate(rb_detected_r):
            dist = abs(r_val - tr)
            if dist < min_dist:
                # 存储元组: (R, Beta, FWHM)
                min_dist = dist
                best_rb = (r_val, rb_detected_betas[k], rb_fwhms[k])
        
        if best_rb and min_dist < 10:
            err_r = abs(best_rb[0] - tr)
            err_fw = abs(best_rb[2] - t_fwhm)
            err_b = abs(best_rb[1] - tb)
            print(f"{'rBasex':<8} | {best_rb[0]:<8.1f} | {'-':<6} | {best_rb[2]:<6.1f} | {best_rb[1]:<6.2f} | {err_r:<7.2f} | {err_fw:<7.2f} | {err_b:<7.2f}")
        else:
            print(f"{'rBasex':<8} | {'Missed':<8} | ...")
            
        # Ours Match
        best_my = None
        min_dist = 999
        for p in params_my:
            dist = abs(p['r'] - tr)
            if dist < min_dist: 
                min_dist = dist
                best_my = p
        
        if best_my and min_dist < 10:
            err_r = abs(best_my['r'] - tr)
            err_fw = abs(best_my['fwhm'] - t_fwhm)
            err_b = abs(best_my['beta'] - tb)
            print(f"{'Ours':<8} | {best_my['r']:<8.1f} | {best_my['amp']:<6.2f} | {best_my['fwhm']:<6.1f} | {best_my['beta']:<6.2f} | {err_r:<7.2f} | {err_fw:<7.2f} | {err_b:<7.2f}")
        else:
            print(f"{'Ours':<8} | {'Missed':<8} | ...")

    print("-" * 125)
    # --- 峰数量统计 ---
    print(f"\n[Peak Detection Summary]")
    print(f"Ground Truth : {len(true_peaks)} peaks")
    print(f"rBasex Found : {len(rb_detected_r)} peaks")
    print(f"Ours Found   : {len(params_my)} peaks")
    print("="*125)
    
    # --- 绘图 (保持要求) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 第一行
    vmax = np.percentile(img_input, 99.5)
    im1 = axes[0,0].imshow(img_input, cmap='gray', vmin=0, vmax=vmax)
    axes[0,0].set_title(f"Input Raw Photoelectron Image\n(Photons={PHOTONS:.0e})")
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    vmax_rec = np.percentile(recon_rb, 99.5)
    im2 = axes[0,1].imshow(recon_rb, cmap='hot', vmin=0, vmax=vmax_rec)
    axes[0,1].set_title("rBasex Reconstruction (Slice)")
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    im3 = axes[0,2].imshow(img_my_2d, cmap='hot', vmin=0, vmax=vmax_rec)
    axes[0,2].set_title("Ours Reconstruction (Slice)")
    plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)

    # 第二行
    axes[1,0].plot(r_my, I_true, 'k-', linewidth=3, alpha=0.6, label='Ground Truth')
    axes[1,0].plot(r_rb, I_rb, 'b--', linewidth=1.5, label='rBasex')
    axes[1,0].plot(r_my, I_my, 'r-', linewidth=1.5, label='Ours')
    axes[1,0].set_title("Radial Distribution Comparison")
    axes[1,0].set_xlabel("Radius (px)")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter([p['r'] for p in true_peaks], [p['beta'] for p in true_peaks], 
                      s=150, c='k', marker='o', label='True', zorder=10)
    valid_idx = [k for k, r in enumerate(rb_detected_r) if abs(r - N//2) > 10]
    if valid_idx:
        axes[1,1].scatter(rb_detected_r[valid_idx], np.array(rb_detected_betas)[valid_idx], 
                          s=80, c='b', marker='s', label='rBasex', alpha=0.6)
    axes[1,1].scatter([p['r'] for p in params_my], [p['beta'] for p in params_my], 
                      s=80, c='r', marker='^', label='Ours', alpha=0.9)
    axes[1,1].set_ylim(-1.5, 2.5)
    axes[1,1].set_xlabel("Radius (px)")
    axes[1,1].set_ylabel("Beta")
    axes[1,1].set_title("Beta Parameter Accuracy")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].axis('off')
    summary_text = f"Performance Summary:\n\n"
    summary_text += f"Ground Truth Peaks: {len(true_peaks)}\n"
    summary_text += f"rBasex Detected:    {len(rb_detected_r)}\n"
    summary_text += f"Ours Detected:      {len(params_my)}\n\n"
    summary_text += "Metric Added: FWHM (Full Width Half Max)\n"
    summary_text += "- Evaluates how well the peak width\n  is resolved.\n"
    summary_text += "- True FWHM = 2.355 * Sigma"
    
    axes[1,2].text(0.1, 0.9, summary_text, fontsize=12, va='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods()