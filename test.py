import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理模型：2D 极坐标下的期望密度
# ==========================================
def expected_density_map(R, sigma, beta, W_sig, W_bg, r_edges, p_edges, n_total):
    """
    计算极坐标网格中每个格子的“期望电子数”
    """
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2
    R_grid, P_grid = np.meshgrid(r_centers, p_centers)
    
    # --- 信号分布 (Abel-Gaussian) ---
    dr = np.abs(R_grid - R)
    # 平滑核，防止奇异性
    radial = np.exp(-0.5 * (dr / sigma)**2) / (np.sqrt(np.abs(R**2 - R_grid**2) + sigma**2))
    angular = 1 + beta * 0.5 * (3 * np.cos(P_grid)**2 - 1)
    
    signal_pdf = radial * angular
    # 信号归一化：使其在极坐标下的积分为 W_sig
    # 极坐标面积元是 r dr dphi
    norm_s = np.sum(signal_pdf * R_grid) 
    signal_map = (signal_pdf * R_grid) * (W_sig / (norm_s + 1e-9))
    
    # --- 背景分布 (Uniform in 2D) ---
    # 背景在 r 方向随 r 线性增长 (P(r) dr dphi = C * r dr dphi)
    bg_pdf = R_grid
    norm_b = np.sum(bg_pdf)
    bg_map = bg_pdf * (W_bg / (norm_b + 1e-9))
    
    # 期望总点数
    return (signal_map + bg_map) * n_total

# ==========================================
# 2. 核心损失函数：泊松分布一致性 (Poisson Deviance)
# ==========================================
def poisson_loss(params, obs_map, r_edges, p_edges, n_total, sigma_min, n_peaks):
    # 解析参数
    Rs = params[0:n_peaks]
    sigs = sigma_min + params[n_peaks:2*n_peaks]
    betas = params[2*n_peaks:3*n_peaks]
    w_raw = params[3*n_peaks:]
    w_exp = np.exp(w_raw - np.max(w_raw))
    weights = w_exp / np.sum(w_exp) # [w1, w2, ..., w_bg]
    
    # 强制有序
    if n_peaks > 1 and np.any(np.diff(Rs) < 5): return 1e15

    expected_map = np.zeros_like(obs_map)
    # 背景贡献
    bg_grid = expected_density_map(0, 1, 0, 0, 1.0, r_edges, p_edges, n_total)
    expected_map += weights[-1] * bg_grid
    
    # 信号贡献
    for k in range(n_peaks):
        expected_map += expected_density_map(Rs[k], sigs[k], betas[k], weights[k], 0, r_edges, p_edges, n_total)
    
    # 泊松对数似然 (Poisson Deviance)
    # 这是衡量“分布一致性”的最科学指标
    # 只有当数据点的空间密度与 PDF 完美重合时，该值最小
    mu = expected_map + 1e-9
    loss = np.sum(mu - obs_map * np.log(mu))
    
    # 额外的物理约束惩罚
    penalty = 0.5 * np.sum((np.log(sigs/3.0))**2)
    
    return loss + penalty

# ==========================================
# 3. 执行流程
# ==========================================
def run_distribution_fit(x_raw, y_raw, n_peaks, truth):
    r_data = np.sqrt(x_raw**2 + y_raw**2)
    phi_data = np.arctan2(y_raw, x_raw)
    n_total = len(r_data)
    
    # --- 关键：建立极坐标网格 (Polar Binning) ---
    # r轴分 60 份, phi轴分 36 份 (每10度一份)
    r_edges = np.linspace(0, 150, 61)
    p_edges = np.linspace(-np.pi, np.pi, 37)
    obs_map, _, _ = np.histogram2d(phi_data, r_data, bins=[p_edges, r_edges])

    # 设定 Bounds
    bounds = [(10, 150)]*n_peaks + [(0.5, 10)]*n_peaks + [(-1, 2)]*n_peaks + [(-5, 5)]*(n_peaks+1)

    print("Running Distribution Consistency Fit...")
    res = differential_evolution(
        poisson_loss, bounds, args=(obs_map, r_edges, p_edges, n_total, 0.5, n_peaks),
        popsize=20, strategy='best1bin', disp=True
    )

    # 结果提取与展示
    p = res.x
    Rs, sigs, betas = p[0:n_peaks], 0.5 + p[n_peaks:2*n_peaks], p[2*n_peaks:3*n_peaks]
    w_e = np.exp(p[3*n_peaks:] - np.max(p[3*n_peaks:]))
    weights = w_e / np.sum(w_e)

    print("\n" + "="*65)
    print(f"{'Parameter':<15} | {'Truth':<10} | {'Estimate':<10} | {'Error %':<10}")
    print("-" * 65)
    for i in range(n_peaks):
        for name, val, true_val in [("R", Rs[i], truth['R'][i]), ("Sig", sigs[i], truth['Sig'][i]), ("Beta", betas[i], truth['Beta'][i])]:
            err = abs(val - true_val) / true_val * 100
            print(f"Peak {i+1} {name:<7} | {true_val:<10.2f} | {val:<10.2f} | {err:<10.2f}%")
    print(f"BG Weight       | {truth['W_bg']:<10.2f} | {weights[-1]:<10.2f} | {abs(weights[-1]-truth['W_bg'])/0.2*100:.2f}%")
    print("="*65)

# --- 生成数据 ---
if __name__ == "__main__":
    N = 4000
    truth = {'R':[60.0, 110.0], 'Sig':[2.5, 4.0], 'Beta':[1.6, -0.6], 'W_sig':[0.5, 0.3], 'W_bg':0.2}
    r_list, phi_list = [], []
    for i in range(2):
        n = int(N * truth['W_sig'][i])
        p_phi = np.random.uniform(-np.pi, np.pi, n*10)
        p_acc = 1 + truth['Beta'][i]*0.5*(3*np.cos(p_phi)**2-1)
        p_phi = p_phi[np.random.rand(len(p_phi)) < (p_acc/2.5)][:n]
        r_list.append(np.random.normal(truth['R'][i], truth['Sig'][i], n))
        phi_list.append(p_phi)
    r_list.append(np.sqrt(np.random.uniform(0, 150**2, int(N*0.2))))
    phi_list.append(np.random.uniform(-np.pi, np.pi, int(N*0.2)))
    X, Y = np.concatenate(r_list)*np.cos(np.concatenate(phi_list)), np.concatenate(r_list)*np.sin(np.concatenate(phi_list))
    
    run_distribution_fit(X, Y, 2, truth)