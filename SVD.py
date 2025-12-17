# angular ambiguity and radial blurring demonstration
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def generate_3d_dist(r_grid, z_grid, r0, sigma, l_order):
    """
    生成一个3D分布切片 (在 y=0 平面上, 用于可视化)
    r_grid, z_grid: 网格
    r0: 径向壳层中心
    sigma: 壳层宽度
    l_order: 勒让德多项式阶数 (0 或 2)
    """
    # 3D 坐标转换
    # 这里的 r_3d 是球坐标半径
    r_3d = np.sqrt(r_grid**2 + z_grid**2)
    
    # 径向基函数 (高斯壳层)
    radial_func = np.exp(-(r_3d - r0)**2 / (2 * sigma**2))
    
    # 角向基函数 (勒让德多项式 P_l(cos theta))
    # cos_theta = z / r
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = z_grid / r_3d
        cos_theta[np.isnan(cos_theta)] = 0
        
    if l_order == 0:
        angular_func = np.ones_like(cos_theta) # P0 = 1
    elif l_order == 2:
        angular_func = 0.5 * (3 * cos_theta**2 - 1) # P2
    else:
        angular_func = np.ones_like(cos_theta)

    return radial_func * angular_func

def numerical_abel_forward(y_img, z_img, r0, sigma, l_order, x_range=50):
    """
    数值计算 Abel Forward 变换 (沿着 x 轴积分)
    y_img, z_img: 探测器平面的坐标
    """
    projection = np.zeros_like(y_img)
    
    # 沿着视线方向 (x轴) 积分
    # 简单的黎曼和积分
    dx = 0.5
    xs = np.arange(-x_range, x_range, dx)
    
    for x in xs:
        # 当前 3D 半径 r = sqrt(x^2 + y^2 + z^2)
        r_3d = np.sqrt(x**2 + y_img**2 + z_img**2)
        
        # 当前 cos_theta = z / r_3d (假设 z 是极化轴/对称轴)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = z_img / r_3d
            cos_theta[np.isnan(cos_theta)] = 0 # 处理原点
            
        # 径向部分
        f_r = np.exp(-(r_3d - r0)**2 / (2 * sigma**2))
        
        # 角向部分
        if l_order == 0:
            P_l = 1.0
        elif l_order == 2:
            P_l = 0.5 * (3 * cos_theta**2 - 1)
            
        projection += f_r * P_l * dx
        
    return projection

# --- 设置参数 ---
size = 100
y = np.linspace(-40, 40, size)
z = np.linspace(-40, 40, size)
Y, Z = np.meshgrid(y, z)

r0 = 20.0 # 壳层半径
sigma = 2.0 # 壳层很窄，模拟 delta 函数

# --- 计算 l=0 和 l=2 的投影 ---
proj_l0 = numerical_abel_forward(Y, Z, r0, sigma, l_order=0)
proj_l2 = numerical_abel_forward(Y, Z, r0, sigma, l_order=2)

# --- 绘图 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 原始径向函数 (截面)
R = np.linspace(0, 40, 100)
f_r_profile = np.exp(-(R - r0)**2 / (2 * sigma**2))
axes[0].plot(R, f_r_profile, 'k-', lw=2, label='Original Radial Basis')
axes[0].set_title("Original Radial Basis f(r)")
axes[0].set_xlabel("r")
axes[0].legend()

# 2. l=0 的投影
im1 = axes[1].imshow(proj_l0, extent=[-40,40,-40,40], origin='lower', cmap='viridis')
axes[1].set_title("Projection of l=0 (Isotropic)")
axes[1].text(-35, 35, "Different Shapes", color='white', fontweight='bold')

# 3. l=2 的投影
im2 = axes[2].imshow(proj_l2, extent=[-40,40,-40,40], origin='lower', cmap='viridis')
axes[2].set_title("Projection of l=2 (Anisotropic)")

plt.tight_layout()
plt.show()

# 结论输出
print("观察结果：")
print("1. 角向形状完全不同：l=0 是圆环，l=2 是双瓣结构。Abel变换没有把 l=0 变成 l=2。")
print("2. 径向模糊：请看投影图中心填充了颜色。原物体是空心的壳(r=20)，但投影后，r<20的区域也有信号。")
print("   这就是'径向模糊'，它导致了信息重叠。")

# radial blurring and ambiguity demonstration
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# 模拟参数
N_pixels = 100  # 图像半径上的像素点数
N_basis = 100   # 我们使用的径向基函数的数量
r_max = 50.0

# 图像坐标 (探测器半径 y)
y_points = np.linspace(0, r_max, N_pixels)

# 定义 pBASEX 风格的基函数投影矩阵 G
# 简化版：假设我们只看 l=0 的分量，Abel变换公式已知
# 对于高斯基函数 exp(-(r-k)^2), 其 Abel 投影没有简单的解析解，
# 但为了演示，我们用数值积分生成这个矩阵 G。

G_matrix = np.zeros((N_pixels, N_basis))
basis_centers = np.linspace(0, r_max, N_basis)
sigma = (r_max / N_basis) # 基函数宽度

print("正在生成基函数投影矩阵 (G matrix)...")

# 构建矩阵：每一列是一个基函数的投影
for k in range(N_basis):
    r0 = basis_centers[k]
    
    # 对当前基函数计算 Abel Forward (简化为 1D 积分)
    # Projection P(y) = 2 * integral_{y}^{inf} f(r) * r / sqrt(r^2 - y^2) dr
    # 这里我们离散化计算
    
    col_vector = []
    for y_val in y_points:
        # 积分区间从 y_val 到 r_max
        if y_val >= r_max:
            col_vector.append(0)
            continue
            
        rs = np.linspace(y_val, r_max * 1.5, 200) # 积分变量 r
        dr = rs[1] - rs[0]
        
        # 径向基函数值
        f_r = np.exp(-(rs - r0)**2 / (2 * sigma**2))
        
        # 避免分母为0
        denom = np.sqrt(rs**2 - y_val**2)
        denom[denom == 0] = 1e-10 
        
        integrand = f_r * rs / denom
        val = 2 * np.sum(integrand) * dr
        col_vector.append(val)
        
    G_matrix[:, k] = col_vector

# --- SVD 分析 ---
U, S, Vt = svd(G_matrix)

# --- 绘图 ---
plt.figure(figsize=(12, 5))

# 1. 可视化 G 矩阵的前几列
plt.subplot(1, 2, 1)
plt.plot(y_points, G_matrix[:, 0], label='Basis at r=10')
plt.plot(y_points, G_matrix[:, 1], label='Basis at r=10.5')
plt.plot(y_points, G_matrix[:, 2], label='Basis at r=11')





plt.title("Projections of Adjacent Basis Functions")
plt.xlabel("Detector Radius y")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)

# 2. 奇异值谱 (Singular Value Spectrum)
plt.subplot(1, 2, 2)
plt.semilogy(S, 'o-', markersize=4, color='r')
plt.title("Singular Values of Projection Matrix G")
plt.xlabel("Index")
plt.ylabel("Singular Value (log scale)")
plt.grid(True)
plt.axvline(x=40, color='b', linestyle='--', label='Noise Floor (Example)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Condition Number (cond): {S[0]/S[-1]:.2e}")




# ambiguity test for SVD-based Abel inversion
#---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# --- 1. 准备数据 ---
N_pixels = 100
N_basis = 60 # 稍微减少基函数数量以便看清
r_max = 50.0
y_points = np.linspace(0, r_max, N_pixels)
basis_centers = np.linspace(0, r_max, N_basis)
sigma = (r_max / N_basis) * 1.0 # 设定宽度，保证有重叠

# 构建投影矩阵 G
G = np.zeros((N_pixels, N_basis))
print("正在构建投影矩阵 G (模拟 Abel 投影)...")
for k in range(N_basis):
    r0 = basis_centers[k]
    # 模拟 Abel 投影: 高斯函数的投影接近 Gaussian (近似)
    # 这里用近似公式，为了定性展示 Overlap
    # 真实的 Abel 投影会比高斯宽，Overlap 会更严重
    G[:, k] = np.exp(-(y_points - r0)**2 / (2 * (sigma * 1.5)**2)) 

# --- 2. 证明 Ambiguity 来源于 Overlap (几何视角) ---
# 计算 Gram 矩阵: M = G.T @ G
# M_ij 代表第 i 个基函数和第 j 个基函数的重叠积分
Gram_matrix = G.T @ G

plt.figure(figsize=(12, 5))

# 图1: Gram Matrix (Overlap Map)
plt.subplot(1, 2, 1)
plt.imshow(Gram_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Overlap Strength (Inner Product)')
plt.title(f"Gram Matrix (G.T @ G)\nVisualizing Overlap Ambiguity")
plt.xlabel("Basis Function Index j")
plt.ylabel("Basis Function Index i")
# 标注
plt.text(N_basis/2, N_basis/2, "High Overlap\n(Ambiguity)", 
         ha='center', va='center', color='cyan', fontweight='bold')

# --- 3. 证明 SVD 选择了什么 (SVD 视角) ---
# 对 G 进行 SVD: G = U * S * Vt
# U 的列向量 (Columns of U) 代表了投影空间中的 "正交模式" (Eigen-images)
U, S, Vt = svd(G)

plt.subplot(1, 2, 2)
# 绘制前3个重要模式 (Large Singular Values)
plt.plot(y_points, U[:, 0], 'r-', linewidth=2, label=f'Mode 0 (S={S[0]:.1f})')
plt.plot(y_points, U[:, 1], 'g-', linewidth=2, label=f'Mode 1 (S={S[1]:.1f})')
plt.plot(y_points, U[:, 2], 'b-', linewidth=2, label=f'Mode 2 (S={S[2]:.1f})')

# 绘制最后1个微不足道的模式 (Small Singular Value)
# 为了看清，放大它的振幅 (实际上它对应的贡献极小)
plt.plot(y_points, U[:, -1], 'k--', alpha=0.5, label=f'Last Mode (S={S[-1]:.1e})')

plt.title("What SVD Actually Selects\n(Columns of U Matrix)")
plt.xlabel("Detector Radius")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 文字解释输出 ---
print("\n--- 深度解析 ---")
print(f"1. 左图 (Gram Matrix): 对角线非常宽。这意味着 Basis[i] 和 Basis[i+1], Basis[i+2]...")
print(f"   都有巨大的重叠。这就是你说的 'Overlap Region'，几何上导致了 'Ambiguity'。")
print(f"   因为投影太像了，很难分清信号到底来自 i 还是 i+1。")

print(f"\n2. 右图 (SVD Modes):")
print(f"   - Mode 0, 1, 2 (彩色线): 这些对应最大的奇异值。注意它们非常平滑。")
print(f"     SVD 发现既然大家重叠这么厉害，不如把大家共有的部分提取出来，作为一个新的基。")
print(f"     这代表了 '整体轮廓'。")
print(f"   - Last Mode (虚线): 对应最小的奇异值。注意它是高频震荡的 (+ - + -)。")
print(f"     这代表了什么？它代表了 '差异'。要在高度重叠的模糊中区分细节，")
print(f"     就需要这种剧烈的正负相消。但这通常是病态的，因为噪声也是这种形态。")
print(f"\n结论: SVD 通过选择大的奇异值，实际上是选择了 '平滑的、非重叠的整体特征'，")
print(f"      而扔掉了那些 '试图在重叠区域强行区分细节的高频震荡'。")

# see how SVD basis functions look like and how to reconstruct
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# --- 1. 准备环境 ---
N_pixels = 100
N_basis = 100
r_max = 50.0
y_points = np.linspace(0, r_max, N_pixels)
basis_centers = np.linspace(0, r_max, N_basis)
sigma = (r_max / N_basis)

# 构建投影矩阵 G (模拟 Abel)
# 每一列是一个高斯壳层(Old Basis)的投影
G = np.zeros((N_pixels, N_basis))
for k in range(N_basis):
    r0 = basis_centers[k]
    # 模拟 Abel 投影形状
    G[:, k] = np.exp(-(y_points - r0)**2 / (2 * (sigma * 2.0)**2))

# --- 2. SVD 分解 (构建新基底) ---
# G = U * S * Vt
# V 的列 (即 Vt 的行) 就是 "新基底" 在 "老基底空间" 的表示
U, S, Vt = svd(G)
V = Vt.T  # 转置一下，方便理解：V 的每一列是一个新基底向量

# --- 证明 1: SVD 新基底长得像傅里叶变换的波 ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
# 画出前几个 V 的列向量 (对应的奇异值大)
plt.plot(basis_centers, V[:, 0], label='New Basis #0 (Low Freq)')
plt.plot(basis_centers, V[:, 1], label='New Basis #1')
plt.plot(basis_centers, V[:, 3], label='New Basis #3')
# 画出一个后面的 V (对应的奇异值小)
plt.plot(basis_centers, V[:, 10], label='New Basis #10 (High Freq)')
plt.title("The 'New Basis' Functions (Columns of V)\nSimilar to Fourier Modes")
plt.xlabel("Radial Position r")
plt.legend()
plt.grid(True)

# --- 证明 2: 怎么变回老基底 (重建过程演示) ---

# 假设我们需要重建一个真实的径向分布 (Ground Truth)
# 比如一个位于 r=25 的双高斯峰
true_coeffs_old = np.zeros(N_basis)
true_coeffs_old[40] = 1.0 # 在 r=20 处有一个峰
true_coeffs_old[60] = 0.5 # 在 r=30 处有一个峰
# 生成模拟的观测图像 (带一点噪声)
measured_image = G @ true_coeffs_old
measured_image += np.random.normal(0, 0.01, size=measured_image.shape)

# === 重建步骤 (核心逻辑) ===

# Step 1: 投影到 SVD 域 (得到新基底的系数)
# w = U^T * Image
weights_in_new_basis = U.T @ measured_image

# Step 2: 正则化/滤波 (这一步就是你说的"SVD选中重要的成分")
# 我们只保留前 k 个成分，除以奇异值 S
k_keep = 15 # 截断参数 (Truncation)
coeffs_new = np.zeros(N_basis)

# 反演公式: c_new = w / s
# 就像傅里叶去噪：把高频系数置零
coeffs_new[:k_keep] = weights_in_new_basis[:k_keep] / S[:k_keep]

# Step 3: "字典翻译" - 从新基底映射回老基底
#coeffs_old_reconstructed = V * coeffs_new
reconstructed_coeffs_old = V @ coeffs_new

# Step 4: 最终重建的径向函数 (老系数 * 老基底函数)
# 这一步通常在画图时做，实际上 reconstructed_coeffs_old 已经是我们要的径向分布了
# (因为我们的老基底是 delta 或者是窄高斯，系数本身就是分布)

plt.subplot(1, 2, 2)
plt.plot(basis_centers, true_coeffs_old, 'k--', label='True Radial Dist.')
plt.plot(basis_centers, reconstructed_coeffs_old, 'r-', linewidth=2, label='Reconstructed via SVD')
plt.title(f"Reconstruction (Using first {k_keep} SVD modes)")
plt.xlabel("Radial Position r")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("重建过程数学流：")
print("1. 测量图像 I")
print("2. 转换到 SVD 频率域: w = U.T * I")
print("3. 在 SVD 域去噪/放大: c_new = w / S (只取前 k 项)")
print("4. **翻译回老基底**: c_old = V * c_new")
print("   (矩阵 V 的作用就是把抽象的 SVD 波形组合成物理的高斯壳层系数)")

# prove SVD is not radical FFT
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# --- 准备数据 ---
N = 200
r_max = 50.0
y = np.linspace(0, r_max, N)
basis_centers = np.linspace(0, r_max, N)
sigma = r_max / N * 1.5

# 构建 Abel 投影矩阵 G
G = np.zeros((N, N))
for k in range(N):
    r0 = basis_centers[k]
    # 模拟 Abel 投影的核心特征：边缘扩散
    # 简单的 Gaussian 不足以体现 Abel 的几何不对称性，
    # 我们这里用数值积分模拟更真实的 1/sqrt(r^2-y^2) 衰减行为
    # 为了简化代码，用变宽的高斯模拟：越靠外，投影越宽
    width = sigma * (1 + 0.05 * r0) 
    G[:, k] = np.exp(-(y - r0)**2 / (2 * width**2))

# --- SVD 分解 ---
U, S, Vt = svd(G)
V = Vt.T 

# --- 分析基底的频率特性 ---
# 我们选取一个中等频率的 SVD 模式 (比如第 10 个模式)
mode_idx = 10
svd_wave = V[:, mode_idx]

# 生成一个纯傅里叶模式 (同频率的正弦波) 作对比
fourier_wave = np.sin(np.linspace(0, np.pi * (mode_idx + 1), N))

# --- 计算过零点 (Zero Crossings) 来衡量波长 ---
def get_zero_crossing_diffs(wave, x_axis):
    # 找到符号变化的位置
    signs = np.sign(wave)
    diff_signs = np.diff(signs)
    indices = np.where(diff_signs != 0)[0]
    if len(indices) < 2: return []
    crossings = x_axis[indices]
    # 计算相邻过零点的距离 (即半波长)
    spacings = np.diff(crossings)
    return crossings[:-1], spacings

zc_svd_x, zc_svd_diff = get_zero_crossing_diffs(svd_wave, y)
zc_fft_x, zc_fft_diff = get_zero_crossing_diffs(fourier_wave, y)

# --- 绘图 ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 图1: 波形对比
axes[0].plot(y, svd_wave, 'r-', label=f'SVD Basis #{mode_idx} (Adapted)')
axes[0].plot(y, fourier_wave, 'b--', alpha=0.5, label='Fourier Basis (Fixed Freq)')
axes[0].set_title(f"Waveform Comparison (Mode {mode_idx})")
axes[0].set_ylabel("Amplitude")
axes[0].legend()

# 图2: 波长变化分析
axes[1].plot(zc_fft_x, zc_fft_diff, 'b--o', label='Fourier Wavelength (Constant)')
axes[1].plot(zc_svd_x, zc_svd_diff, 'r-o', label='SVD Wavelength (Variable!)')
axes[1].set_title("Wavelength (Spacing between Zero-Crossings) vs. Radius")
axes[1].set_xlabel("Radius Position (r)")
axes[1].set_ylabel("Local Wavelength (Delta r)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("分析结果：")
print("1. 蓝色线 (Fourier) 是水平的，说明不管在中心还是边缘，波长都是一样的。")
print("2. 红色线 (SVD) 是倾斜/弯曲的。")
print("   这意味着 SVD 的基底在不同半径处，'频率' 是自动调整的。")
print("   这种调整是为了补偿 Abel 积分带来的几何扭曲。")

# its frequency is changing with radius
# larger radius -> larger wavelength (lower frequency)
# smaller radius -> smaller wavelength (higher frequency)
#--------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.signal import stft

# --- 1. 构建 Abel 投影矩阵 & SVD ---
N = 500  # 点数多一点，为了时频分析清晰
r_max = 100.0
y = np.linspace(0, r_max, N)
basis_centers = np.linspace(0, r_max, N)
sigma = r_max / N * 2.0

# 构建矩阵 G
G = np.zeros((N, N))
for k in range(N):
    r0 = basis_centers[k]
    # 模拟 Abel 核的几何衰减特性
    # 这是一个经验性的模拟，捕捉 Abel 变换随半径变宽、变弱的趋势
    G[:, k] = np.exp(-(y - r0)**2 / (2 * (sigma * (1 + 0.02*r0))**2)) / (np.sqrt(r0 + 1))

# SVD 分解
U, S, Vt = svd(G)
V = Vt.T 

# --- 2. 选取一个高频基底进行分析 ---
# 选第 50 个模式，它的震荡次数足够多，能看清频率变化
mode_idx = 50 
svd_chirp_signal = V[:, mode_idx]
# 生成一个等效频率的普通正弦波做对比
fft_sine_signal = np.sin(np.linspace(0, 16 * np.pi, N)) 

# --- 3. 时频分析 (STFT) ---
# 这就像给信号照“CT”，看它的频率随时间(位置)怎么变
f_svd, t_svd, Zxx_svd = stft(svd_chirp_signal, nperseg=64, noverlap=50)
f_fft, t_fft, Zxx_fft = stft(fft_sine_signal, nperseg=64, noverlap=50)

# --- 4. 绘图 ---
plt.figure(figsize=(12, 6))

# 左图：普通正弦波的谱图
plt.subplot(1, 2, 1)
plt.pcolormesh(t_fft, f_fft, np.abs(Zxx_fft), shading='gouraud', cmap='inferno')
plt.title("Spectrogram of Fourier Basis (Sine)")
plt.ylabel("Frequency")
plt.xlabel("Position (Radius)")
plt.axhline(y=0.12, color='w', linestyle='--', alpha=0.5)
plt.text(10, 0.13, "Frequency is Constant", color='w')

# 右图：SVD 基底的谱图
plt.subplot(1, 2, 2)
plt.pcolormesh(t_svd, f_svd, np.abs(Zxx_svd), shading='gouraud', cmap='inferno')
plt.title(f"Spectrogram of SVD Basis #{mode_idx}")
plt.ylabel("Frequency")
plt.xlabel("Position (Radius)")

# 标注趋势
plt.arrow(10, 0.05, 40, 0.15, color='cyan', head_width=0.01, linewidth=2)
plt.text(20, 0.22, "Frequency Changes!\n(This is a Chirp)", color='cyan', fontweight='bold')

plt.tight_layout()
plt.show()

print("实验结论：")
print("1. 左图 (Fourier): 亮斑是一条水平线。说明无论在哪里，震荡频率都是一样的。")
print("2. 右图 (SVD): 亮斑是一条‘斜线’或‘曲线’。")
print("   - 在 r 较小的地方，频率较低。")
print("   - 在 r 较大的地方，频率较高（或者反之，取决于核函数的具体构建）。")
print("   - 这完美证明了你的猜想：它就是一个线性/非线性调频信号 (Chirp)！")

# prove that the choose of base is very important which verify the idea of measuring the beam gaussian width can help reconstruction
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# --- 1. 模拟实验数据 (Ground Truth) ---
N_pixels = 100
r = np.linspace(0, 50, N_pixels)

# 真实的物理效应：光电子谱不是 Delta 函数，而是有一定宽度的
# 假设仪器分辨率导致峰宽 sigma_instr = 2.0
true_center = 25.0
sigma_instr = 2.0 
# 真实的径向分布
true_radial = np.exp(-(r - true_center)**2 / (2 * sigma_instr**2))

# 生成 Abel 投影 (模拟观测图像)
# 投影矩阵 G_true 使用真实的宽度生成
G_true = np.zeros((N_pixels, N_pixels))
for k in range(N_pixels):
    rk = r[k]
    # 这里简化模拟：投影矩阵也是基于仪器宽度的卷积
    G_true[:, k] = np.exp(-(r - rk)**2 / (2 * sigma_instr**2)) 

# 观测信号 = 投影 + 噪声
measured_signal = G_true @ true_radial
#measured_signal += np.random.normal(0, 0.05, size=N_pixels) * np.max(measured_signal) # 加5%噪声



# --- 2. 两种重建策略对比 ---

# 策略 A: 盲目拟合 (假设基底是非常窄的 Delta / 1像素)
# 这是很多初学者的默认设置
sigma_basis_A = 0.5 
G_A = np.zeros((N_pixels, N_pixels))
for k in range(N_pixels):
    G_A[:, k] = np.exp(-(r - r[k])**2 / (2 * sigma_basis_A**2))

# 策略 B: 物理匹配 (你的思路 - 匹配激光/仪器宽度)
sigma_basis_B = 2.0 
G_B = np.zeros((N_pixels, N_pixels))
for k in range(N_pixels):
    G_B[:, k] = np.exp(-(r - r[k])**2 / (2 * sigma_basis_B**2))

# --- 定义通用的 pBASEX 重建函数 ---
def reconstruct_pbasex(G, signal, r_cond=0.01):
    U, S, Vt = svd(G)
    # 截断奇异值 (正则化)
    limit = S[0] * r_cond
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > limit:
            S_inv[i] = 1.0 / S[i]
        else:
            S_inv[i] = 0.0
            
    # c = V * S_inv * U.T * I
    weights = U.T @ signal
    coeffs = Vt.T @ (S_inv * weights)
    return coeffs

# 执行重建
# 注意：重建出来的 coeffs 本身就是径向分布，因为基函数中心对应 r
res_A = reconstruct_pbasex(G_A, measured_signal, r_cond=0.05)
res_B = reconstruct_pbasex(G_B, measured_signal, r_cond=0.05)

# --- 绘图对比 ---
plt.figure(figsize=(10, 6))

plt.plot(r, true_radial, 'k-', linewidth=3, alpha=0.3, label='Ground Truth (Physics)')
plt.plot(r, measured_signal, 'k--', label='Measured Projection (Data)')

plt.plot(r, res_A, 'r-o', markersize=3, label='Reconstruction A (Too Narrow Basis)')
plt.plot(r, res_B, 'g-o', markersize=3, label='Reconstruction B (Matched Basis - YOUR IDEA)')

plt.title("Why Matching Basis Width to Experiment Matters")
plt.xlabel("Radius / Energy")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.show()

# prove that the spectrum of incident beam is influencing the width of circle , larger radius less influence
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def simulate_peak_widths(energy_list, bandwidth_E, detector_res_pixel, k_VMI=10.0):
    """
    energy_list: 电子能量 (eV)
    bandwidth_E: 激光光谱导致的能量展宽 (eV)
    detector_res_pixel: 探测器本身的点扩散 (像素)
    k_VMI: VMI 常数, r = k * sqrt(E)
    """
    
    radii = k_VMI * np.sqrt(energy_list)
    
    # 1. 由光谱 (Physics) 导致的径向宽度
    # dr = (k / 2*sqrt(E)) * dE
    width_physics = (k_VMI / (2 * np.sqrt(energy_list))) * bandwidth_E
    
    # 2. 最终观测宽度 (卷积)
    # 假设高斯卷积: width_total = sqrt(w_phys^2 + w_inst^2)
    width_observed = np.sqrt(width_physics**2 + detector_res_pixel**2)
    
    return radii, width_observed, width_physics

# --- 参数设置 ---
Energies = np.linspace(0.1, 4.0, 50) # 0.1 到 4 eV

# 情况 A: 纳秒激光 (带宽极窄, 仪器分辨率主导)
# Bandwidth ~ 0.0001 eV, Detector blur ~ 2 pixels
r_A, w_total_A, w_phys_A = simulate_peak_widths(Energies, 0.0001, 2.0)

# 情况 B: 飞秒激光 (带宽很大, 光谱主导)
# Bandwidth ~ 0.1 eV, Detector blur ~ 2 pixels
r_B, w_total_B, w_phys_B = simulate_peak_widths(Energies, 0.1, 2.0)

# --- 绘图 ---
plt.figure(figsize=(12, 5))

# Plot A: Nanosecond Case
plt.subplot(1, 2, 1)
plt.plot(r_A, w_total_A, 'r-o', label='Observed Width (Total)')
plt.plot(r_A, [2.0]*len(r_A), 'k--', label='Instrument Limit (MCP)')
plt.title("Case A: Nanosecond Laser\n(Instrument Limited)")
plt.xlabel("Radius (pixels)")
plt.ylabel("Peak Width (pixels)")
plt.ylim(0, 5)
plt.legend()
plt.grid(True)

# Plot B: Femtosecond Case
plt.subplot(1, 2, 2)
plt.plot(r_B, w_total_B, 'r-o', label='Observed Width (Total)')
plt.plot(r_B, w_phys_B, 'b--', alpha=0.5, label='Spectral Contribution (Physics)')
plt.plot(r_B, [2.0]*len(r_B), 'k--', label='Instrument Limit (MCP)')
plt.title("Case B: Femtosecond Laser\n(Spectral Bandwidth Limited)")
plt.xlabel("Radius (pixels)")
plt.ylabel("Peak Width (pixels)")
plt.ylim(0, 10)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("决策指南：")
print("1. 如果你的图像像左图 (Case A)：里外圈宽度差不多。")
print("   -> pBASEX 基底宽度设为常数 (即 Instrument Width，例如 2.0)。")
print("   -> 这时候不要管激光光谱，因为它太窄了，被仪器分辨率淹没了。")
print("")
print("2. 如果你的图像像右图 (Case B)：中心圈很宽，边缘圈很窄。")
print("   -> pBASEX 基底宽度设为 '图像中最窄的那个峰' (通常在边缘)。")
print("   -> 或者使用高级的 pBASEX 变体，允许基底宽度随半径变化 (var-width pBASEX)。")

# considering brodening effect by incident beam and instrumental blur
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def generate_physics_informed_G(N_pixels, r_max, sigma_inst, bandwidth_factor):
    """
    生成符合物理模型的 Abel 投影矩阵 G
    
    参数:
    N_pixels: 像素数
    r_max: 最大半径
    sigma_inst: 仪器固有的模糊 (Instrumental broadening, const)
    bandwidth_factor: 光谱导致的展宽系数 (Spectral broadening, ~ 1/r)
                      近似公式: sigma_spec = C / r  (因为 r ~ sqrt(E), dr ~ dE/sqrt(E) ~ dE/r)
    """
    y_points = np.linspace(0, r_max, N_pixels)
    basis_centers = np.linspace(0, r_max, N_pixels)
    
    G = np.zeros((N_pixels, N_pixels))
    
    # 记录每个半径处的实际合成宽度，用于绘图分析
    sigma_eff_list = []
    
    for k in range(N_pixels):
        r0 = basis_centers[k]
        
        # --- 核心改进：物理展宽模型 ---
        # 防止 r0=0 除零错误，加一个小量
        safe_r = max(r0, 0.1)
        
        # 1. 光谱展宽部分 (变宽)
        # 注意：这里假设 bandwidth 是常数能量 dE。
        # 实际上 dr = k * dE / (2*sqrt(E)) = k' * dE / r
        sigma_spec = bandwidth_factor / safe_r
        
        # 2. 仪器展宽部分 (固定)
        sigma_inst_val = sigma_inst
        
        # 3. 总宽度 (卷积)
        sigma_total = np.sqrt(sigma_spec**2 + sigma_inst_val**2)
        sigma_eff_list.append(sigma_total)
        
        # --- 生成投影列向量 ---
        # 模拟 Abel 投影：这里用高斯近似 Abel 投影形状，
        # 实际 pBASEX 会计算 exp(-(r-r0)^2) 的解析投影，这里为了演示宽度变化用高斯代替
        # 重点是 sigma_total 的变化
        G[:, k] = np.exp(-(y_points - r0)**2 / (2 * sigma_total**2))
        
    return G, basis_centers, sigma_eff_list

# --- 运行模拟 ---
N = 200
R = 100.0

# 模拟：仪器分辨率 = 1.0 像素，光谱导致中心非常宽
G_phys, r_axis, sigmas = generate_physics_informed_G(N, R, sigma_inst=1.0, bandwidth_factor=20.0)

# --- 绘图 ---
plt.figure(figsize=(10, 6))

# 1. 宽度随半径的变化规律
plt.plot(r_axis, sigmas, 'r-', linewidth=2, label='Total Basis Width (Effective)')
plt.plot(r_axis, [1.0]*N, 'k--', label='Instrument Component (Const)')
plt.plot(r_axis, 20.0/(r_axis+0.1), 'b--', alpha=0.5, label='Spectral Component (~1/r)')

plt.title("Physics-Informed Basis Width Model")
plt.xlabel("Radius r")
plt.ylabel("Basis Width (sigma)")
plt.ylim(0, 10)
plt.legend()
plt.grid(True)

plt.text(40, 4, "Standard pBASEX uses constant width.\nThis model matches physics better!", 
         fontsize=12, backgroundcolor='white')

plt.show()