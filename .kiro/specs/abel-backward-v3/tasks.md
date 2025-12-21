# Implementation Plan

## Abel Backward Reconstruction V3

- [x] 1. 项目结构和核心接口


  - [x] 1.1 创建 Abel_backward_reconstruction_v3.py 文件，定义模块结构


    - 创建文件头部注释，说明物理模型和设计原则
    - 导入必要的库（numpy, scipy, abel, hypothesis）
    - 定义常量和配置参数
    - _Requirements: 8.1, 8.2_
  - [x] 1.2 实现 PeakParams 和 ReconstructionMetadata 数据类

    - 使用 dataclass 定义数据结构
    - 包含所有必要字段（r, sigma_phys, sigma_measured, amp, beta, br, energy_eV, fwhm）
    - _Requirements: 5.1, 5.2_

- [x] 2. Phase 0: DataCleaner 模块

  - [x] 2.1 实现 identify_background_region 方法

    - 计算图像中心和半径
    - 创建外围 15% 区域的布尔掩码
    - _Requirements: 1.1_
  - [x] 2.2 实现 auto_center_refinement 方法（新增）

    - 使用互相关方法精修中心位置
    - 将图像旋转 180°，计算与原图的互相关
    - 互相关峰值位置的一半即为中心偏移
    - 精度可达 0.1 像素
    - _Requirements: 1.1_
  - [x] 2.3 实现 clean 方法

    - 计算背景区均值 μ_total
    - 从整个图像减去 μ_total
    - 计算背景区标准差 σ_bg
    - 保留负值，不做裁剪
    - _Requirements: 1.2, 1.3, 1.5, 1.6_
  - [x] 2.4 实现 verify_cleaning 方法

    - 验证背景区均值为零（容差 1e-6）
    - 验证背景残差分布符合 N(0, σ_bg²)
    - 返回验证结果字典
    - _Requirements: 1.4_
  - [x]* 2.5 编写 Phase 0 属性测试

    - **Property 1: Background Region Mean Zero After Cleaning**
    - **Validates: Requirements 1.4**
  - [x]* 2.6 编写 Phase 0 属性测试

    - **Property 10: Negative Value Preservation**
    - **Validates: Requirements 1.6**

- [x] 3. Phase 1: PolarTransformer 模块

  - [x] 3.1 实现 transform 方法（面积权重重采样）

    - 创建极坐标网格 (n_r, n_theta)
    - **关键：使用 Pixel-to-Bin 面积重叠累加算法，而非 map_coordinates 插值**
    - 预计算稀疏权重矩阵 W，使得 I_polar = W · I_cartesian
    - 每个笛卡尔像素根据其落入极坐标 Bin 的面积比例分配计数
    - _Requirements: 2.1, 2.4, 2.5_
  - [x] 3.2 实现 _build_weight_matrix 方法

    - 预计算稀疏矩阵 W（scipy.sparse）
    - 存储每个像素对每个极坐标 Bin 的贡献比例
    - 提升 Phase 1 速度百倍
    - _Requirements: 2.1_
  - [x] 3.3 实现 verify_conservation 方法

    - 计算笛卡尔和极坐标图像的总和
    - 计算相对误差
    - **严格验证误差 < 1e-6，如果失败必须重写算法**
    - _Requirements: 2.2_
  - [x] 3.4 实现 inverse_transform 方法

    - 从极坐标转回笛卡尔坐标
    - 用于验证和可视化
    - _Requirements: 2.3_
  - [x]* 3.5 编写 Phase 1 属性测试

    - **Property 2: Sum Conservation in Polar Transform**
    - **Validates: Requirements 2.2**

- [x] 4. Checkpoint - 确保 Phase 0 和 Phase 1 测试通过

  - Ensure all tests pass, ask the user if questions arise.
  - **关键检查点：如果 sum 误差超过 1e-6，必须重写 Phase 1 的重采样算法，不要用插值混过去**

- [x] 5. Phase 2: SeedFinder 模块

  - [x] 5.1 实现 _angular_integrate 方法

    - 对极坐标图像进行角向积分
    - 返回 1D 径向曲线 I_2D(R)
    - _Requirements: 3.1_
  - [x] 5.2 实现 _abel_inverse 方法

    - 使用 Hansen-Law 方法进行 1D Abel 逆变换
    - 返回 I_3D(r)
    - _Requirements: 3.2_
  - [x] 5.3 实现 _detect_peaks 方法

    - 使用 scipy.signal.find_peaks 进行峰值检测
    - 自适应阈值基于 SNR
    - 返回峰值位置列表
    - _Requirements: 3.3_
  - [x] 5.4 实现 _estimate_sigma 方法

    - 使用高斯拟合估计峰值宽度
    - 支持亚像素精度
    - _Requirements: 3.4, 3.5_
  - [x] 5.5 实现 _estimate_beta 方法

    - 在峰值位置进行角向 FFT 分析
    - 提取 k=2 成分估计 β
    - _Requirements: 3.7_
  - [x] 5.6 实现 find_seeds 主方法

    - 整合上述方法
    - 返回 seed 参数列表
    - _Requirements: 3.6, 3.8_
  - [x]* 5.7 编写 Phase 2 属性测试

    - **Property 3: Abel Transform Round-Trip**
    - **Validates: Requirements 3.2**
  - [x]* 5.8 编写 Phase 2 属性测试


    - **Property 4: Peak Parameter Extraction Accuracy**
    - **Validates: Requirements 3.4, 3.5, 3.6, 3.7**

- [x] 6. Checkpoint - 确保 Phase 2 测试通过

  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Phase 3 & 4: ForwardFitter 模块（核心）

  - [x] 7.1 实现 _analytic_abel_with_beta 方法（含完整数学公式）

    - **完整投影公式**：
      I_2D(R, φ) = A_3D × sqrt(r × σ) × exp(-(R-r)²/(2σ²)) × [1 + β × P2(R/r) × P2(cos φ)]
    - **几何修正因子**：Correction(R) = P2(R/r) = (3R²/r² - 1) / 2
    - **物理含义**：投影后的各向异性被"稀释"，有效 β 为 β × P2(R/r)
    - **薄壳近似检查**：当 r/σ < 5 时，添加偏斜校正项 [1 + α × (R-r)/σ]，α = 0.15 × (5 - r/σ)
    - _Requirements: 4.1, 7.1, 7.4, 7.5_
  - [x] 7.2 实现 _forward_model_cartesian 方法（性能优化版）

    - **局部渲染 (ROI Optimization)：只在 r_k ± 5σ_k 的环形区域内计算**
    - 应用 PSF 卷积：由于 σ_sys 通常只有 1-2 像素，使用小核 FIR 卷积而非 FFT
    - 或者在解析公式里直接复合 σ_total = sqrt(σ_phys² + σ_sys²)
    - _Requirements: 4.2, 4.3_
  - [x] 7.3 实现 _wls_loss_irls 方法（IRLS 版本）

    - **采用迭代重加权最小二乘 (IRLS)**：
      - 外循环：固定权重 w = M_prev + σ_bg²
      - 内循环：最小化 Σ[(M - D)² / w]
      - 更新 M_prev = M_current
    - 外循环最大迭代次数：5
    - 避免"虚假收敛"问题
    - _Requirements: 4.4, 4.5, 8.1_
  - [x] 7.4 实现 _analytic_jacobian 方法（含 Correction 因子）


    - 计算 Loss 对 (r, σ, A, β) 的解析偏导数
    - **关键公式**：
      - ∂I/∂r = I × [(R-r)/σ² + 1/(2r) - 3βR²P2(cosφ)/(r³)]
      - ∂I/∂σ = I × [(R-r)²/σ³ - 1/(2σ)]
      - ∂I/∂A = I / A
      - ∂I/∂β = I × P2(R/r) × P2(cos φ) / [1 + β × P2(R/r) × P2(cos φ)]
    - **β 偏导数必须包含 P2(R/r) 因子**
    - _Requirements: 4.8, 4.9_
  - [x] 7.5 实现 fit 主方法

    - 使用 scipy.optimize.least_squares (TRF 方法)
    - 支持参数边界约束 (A >= 0, σ >= 0.3, β ∈ [-1, 2])
    - 迭代过程中剔除弱峰
    - _Requirements: 4.8, 4.9, 4.10_
  - [x] 7.6 实现 _prune_weak_peaks 方法

    - 剔除 A < 0.05 × max(A) 的峰
    - _Requirements: 4.7_
  - [x] 7.7 实现 compute_information_criteria 方法

    - 计算 BIC, AIC, reduced_chi2
    - 用于峰显著性评估
    - _Requirements: 4.6_
  - [x] 7.8 实现 verify_residual 方法


    - 检查残差图是否有环状结构
    - _Requirements: 4.6_
  - [x]* 7.9 编写 Phase 3/4 属性测试

    - **Property 5: Forward Model Consistency**
    - **Validates: Requirements 4.1, 7.1**
  - [x]* 7.10 编写 Phase 3/4 属性测试

    - **Property 6: Non-Negative Amplitude Constraint**
    - **Validates: Requirements 4.8**
  - [x]* 7.11 编写 Phase 3/4 属性测试

    - **Property 7: Sub-Pixel Position Precision**
    - **Validates: Requirements 4.9**
  - [x]* 7.12 编写 Phase 3/4 属性测试

    - **Property 11: Legendre Polynomial Correctness**
    - **Validates: Requirements 7.5**
  - [x]* 7.13 编写 Phase 3/4 属性测试

    - **Property 12: Energy Linearity (Jacobian Verification)**
    - **Validates: Requirements 7.1**

- [x] 8. Checkpoint - 确保 Phase 3/4 测试通过


  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Phase 5: BRCalculator 模块

  - [x] 9.1 实现 calculate 方法

    - BR_k = A_3D_k × σ_phys_k × r_k²
    - 归一化使 Σ BR_k = 1.0
    - _Requirements: 5.1, 5.2, 5.5_
  - [x] 9.2 实现 verify_decoupling 方法

    - 改变 β 初值 ±0.5 重新拟合
    - 检查 BR 偏差 < 1%
    - _Requirements: 5.3, 5.4_
  - [x]* 9.3 编写 Phase 5 属性测试

    - **Property 8: BR Normalization**
    - **Validates: Requirements 5.5**
  - [x]* 9.4 编写 Phase 5 属性测试（关键）

    - **Property 9: β-BR Decoupling**
    - 测试用例：(A=1, β=0), (A=1, β=1), (A=1, β=2)
    - 验收标准：总粒子数 N 偏差 < 0.5%
    - **Validates: Requirements 5.3**
  - [x]* 9.5 编写 Phase 5 属性测试

    - **Property 13: β-Invariant Total Count**
    - **Validates: Requirements 5.3**

- [x] 10. Checkpoint - 确保 Phase 5 测试通过


  - Ensure all tests pass, ask the user if questions arise.
  - **关键检查点：如果 Property 9 (β-BR 解耦测试) 失败，必须重新检查任务 7.1 中的投影公式是否包含 β 的几何权重修正**

- [x] 11. AbelReconstructorV3 主类集成

  - [x] 11.1 实现 __init__ 方法

    - 初始化所有子模块
    - 配置参数传递
    - _Requirements: 1.1-8.5_
  - [x] 11.2 实现 reconstruct 主方法

    - 串联 Phase 0-5
    - 返回 (params, metadata)
    - _Requirements: 1.1-5.5_
  - [x] 11.3 实现 run_all_tests 方法

    - 运行所有验证测试 (Test 0.1-5.1)
    - 返回测试结果字典
    - _Requirements: 6.1-6.10_

- [x] 12. 创建测试脚本 test_v3_percent_error.py




  - [x] 12.1 实现测试框架

    - 参考 test_v2_percent_error.py 的格式
    - 支持不同事件计数 (5e5 ~ 1e8)
    - _Requirements: 6.9, 6.10_
  - [x] 12.2 实现真值对比

    - 计算 r, σ, β, BR 的百分比误差
    - 生成汇总表格
    - _Requirements: 6.9, 6.10_
  - [x] 12.3 实现所有 Phase 检验点

    - Test 0.1: 背景区均值为零
    - Test 0.2: 背景 σ_bg 符合噪声模型
    - Test 1.1: 计数守恒
    - Test 1.2: 单像素映射
    - Test 2.1: Seed 参数覆盖
    - Test 4.1: 残差平坦性
    - Test 4.2: 假峰抑制
    - Test 5.1: β-BR 解耦
    - _Requirements: 6.1-6.8_

- [x] 13. Final Checkpoint - 确保所有测试通过



  - Ensure all tests pass, ask the user if questions arise.
