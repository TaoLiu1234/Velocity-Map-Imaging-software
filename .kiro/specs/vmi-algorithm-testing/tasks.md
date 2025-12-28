# Implementation Plan: VMI Algorithm Testing and Improvement Framework

## Overview

本实现计划将设计文档转换为可执行的编码任务。主要目标：
1. 创建正交测试框架
2. 使用 Abel_forward_simulation 生成带噪声的测试数据
3. 评估 vmi_reconstruction.py 的性能极限
4. 改进算法以减少输入参数并提高鲁棒性

## Current Status (2024-12-23)

### 测试结果摘要
- **总体通过率**: 27% (10/37 测试用例)
- **单峰通过率**: 57%
- **双峰通过率**: 17%
- **三峰及以上**: 0%

### 按条件分析
| 条件 | 通过率 |
|------|--------|
| 事件数 1e4 | 30% |
| 事件数 1e6 | 33% |
| 事件数 1e8 | 20% |
| 良好分离 | 44% |
| 中等分离 | 11% |
| 重叠 | 10% |
| 内侧位置 | 0% |
| 中间位置 | 44% |
| 外侧位置 | 27% |

### 算法性能
- **r0 估计**: 中位误差 ~15%，单峰情况下 <5%
- **β 估计**: 中位误差 ~0.27，单峰情况下 <0.1
- **σ 估计**: 由于 Abel 投影效应，误差较大

### 已知限制
1. 多峰检测（3+峰）需要改进
2. 内侧位置峰（r < 5mm）检测困难
3. 重叠峰分辨率有限
4. σ 估计受 Abel 投影影响

## Tasks

- [x] 1. 项目结构和核心接口

  - [x] 1.1 创建 vmi_test_framework.py 文件
    - 创建文件头部注释
    - 导入必要的库（numpy, scipy, hypothesis, pandas, matplotlib）
    - 导入 Abel_forward_simulation 和 vmi_reconstruction
    - _Requirements: 1.1, 2.1_

  - [x] 1.2 实现数据类定义
    - TestCase 数据类
    - EvaluationResult 数据类
    - TestSummary 数据类
    - _Requirements: 1.10, 3.8_

- [x] 2. 正交测试设计器

  - [x] 2.1 实现 OrthogonalTestDesigner 类
    - 定义测试因子和水平
    - 因子：n_peaks, event_count, peak_separation, beta_range, amplitude_ratio, sigma_range, r_position, noise_level
    - 每个因子至少 3 个水平
    - _Requirements: 7.1, 7.2_

  - [x] 2.2 实现 generate_orthogonal_array 方法
    - 使用 L27(3^13) 或类似正交表
    - 确保所有 2-因子交互被覆盖
    - _Requirements: 7.3_

  - [x] 2.3 实现 add_corner_cases 方法
    - 添加极端条件组合
    - 所有因子在极端值的情况
    - _Requirements: 7.4, 1.8_

  - [ ]* 2.4 编写正交设计属性测试
    - **Property 4: Orthogonal Array Balance**
    - **Validates: Requirements 7.2, 7.3**

- [-] 3. 测试用例生成器

  - [x] 3.1 实现 TestCaseGenerator 类
    - 参数映射定义（SIGMA_MAP, R_POSITION_MAP 等）
    - VMI 校准系数计算
    - _Requirements: 1.1, 2.6_

  - [x] 3.2 实现 _generate_peak_positions 方法
    - 根据 n_peaks, r_position, separation 生成 r0 值
    - 支持 inner (< 5mm), middle (5-15mm), outer (> 15mm)
    - 支持 well-separated (> 5σ), moderate, overlapping (< 2σ)
    - _Requirements: 1.3, 1.4, 1.11, 1.12, 1.13_

  - [x] 3.3 实现 _generate_betas 方法
    - 根据 beta_range 生成 β 值
    - 支持 negative (-1 to 0), zero (0), positive (0 to 2)
    - _Requirements: 1.6_

  - [x] 3.4 实现 _generate_amplitudes 方法
    - 根据 amplitude_ratio 生成分支比
    - 支持 equal, 10:1, 100:1
    - _Requirements: 1.7_

  - [x] 3.5 实现 generate_config 方法
    - 将 TestCase 转换为 Abel_forward_simulation.Config
    - 设置 PSF, noise 等参数
    - _Requirements: 2.2, 2.3_

  - [ ]* 3.6 编写测试用例生成属性测试
    - **Property 1: Test Case Generation Validity**
    - **Property 2: Peak Separation Consistency**
    - **Property 3: Radial Position Coverage**
    - **Validates: Requirements 1.1-1.13**

- [x] 4. Checkpoint - 确保测试生成模块通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. 模拟运行器

  - [x] 5.1 实现 SimulationRunner 类
    - 使用 Abel_forward_simulation.run_simulation()
    - 默认 output_mode='xy_dld'（带 PSF 和 DLD 噪声）
    - _Requirements: 2.1, 2.3_

  - [x] 5.2 实现 run 方法
    - 调用 run_simulation 生成 XY 数据
    - 返回 (xy_data, ground_truth)
    - ground_truth 包含 E_centers, r0_values, sigma_values, beta_values, branching_ratios
    - _Requirements: 2.4, 2.5_

  - [x] 5.3 实现 run_batch 方法
    - 批量运行模拟
    - 支持进度回调
    - _Requirements: 2.2_

  - [ ]* 5.4 编写模拟运行器属性测试
    - **Property 5: Simulation Round-Trip Consistency**
    - **Property 6: Energy-Radius Conversion Consistency**
    - **Validates: Requirements 2.2, 2.4, 2.6**

- [ ] 6. 性能评估器

  - [x] 6.1 实现 PerformanceEvaluator 类
    - 定义容差参数
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 6.2 实现 _match_peaks 方法
    - 使用最近邻匹配（按 r0）
    - 处理漏检和误检
    - _Requirements: 3.5, 3.6, 3.7_

  - [x] 6.3 实现 compute_errors 方法
    - 计算 r0 百分比误差
    - 计算 σ 百分比误差
    - 计算 β 绝对误差
    - 计算振幅比误差
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 6.4 实现 evaluate 方法
    - 整合峰匹配和误差计算
    - 返回 EvaluationResult
    - _Requirements: 3.5_

  - [x] 6.5 实现 aggregate_results 方法
    - 计算 mean, median, std, worst_case
    - 按因子分组统计
    - _Requirements: 3.8, 3.9_

  - [ ]* 6.6 编写性能评估属性测试
    - **Property 7: Error Calculation Correctness**
    - **Property 8: Peak Matching Correctness**
    - **Validates: Requirements 3.1-3.5**

- [x] 7. Checkpoint - 确保评估模块通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. 改进的 VMI 重建器

  - [x] 8.1 实现 ImprovedVMIReconstructor 类基础结构
    - 继承或包装原 VMIReconstructor
    - 添加自动参数检测
    - _Requirements: 4.5_

  - [x] 8.2 实现 _auto_center 方法
    - 使用迭代质心法
    - 或使用互相关法（旋转 180° 对称）
    - _Requirements: 4.2_

  - [x] 8.3 实现 _auto_bin_size 方法
    - 基于数据密度自动选择 dr
    - 确保每个 bin 至少 ~100 个事件
    - 处理低统计量情况
    - _Requirements: 4.2, 5.2_

  - [x] 8.4 实现 _auto_snr_threshold 方法
    - 基于背景噪声水平设置阈值
    - 自适应峰检测灵敏度
    - _Requirements: 4.3_

  - [x] 8.5 实现 _auto_detect_n_peaks 方法
    - 使用多尺度分析
    - 使用 BIC/AIC 准则选择最优峰数
    - _Requirements: 4.1_

  - [x] 8.6 实现 _handle_overlapping_peaks 方法
    - 多峰联合拟合
    - 使用约束优化分离重叠峰
    - _Requirements: 5.1_

  - [x] 8.7 实现 _handle_low_statistics 方法
    - 自适应 binning
    - 或使用 KDE 估计
    - _Requirements: 5.2_

  - [x] 8.8 实现 _handle_high_statistics 方法
    - 分块处理避免内存问题
    - 或使用下采样
    - _Requirements: 5.3_

  - [x] 8.9 实现 reconstruct 方法
    - 整合所有自动化功能
    - 支持可选的 n_peaks 提示
    - 返回带置信度的结果
    - _Requirements: 4.4, 4.6, 4.7_

  - [x] 8.10 添加数值稳定性保护
    - 除零保护（epsilon）
    - 指数溢出保护（clip）
    - 负数开方保护（max(0, x)）
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 8.11 编写改进算法属性测试
    - **Property 9: Auto-Detection Accuracy**
    - **Property 10: Adaptive Binning Correctness**
    - **Property 11: Reconstruction Accuracy Under Standard Conditions**
    - **Validates: Requirements 4.1, 4.2, 4.4**

  - [ ]* 8.12 编写鲁棒性属性测试
    - **Property 12: Robustness Under Extreme Conditions**
    - **Property 13: Overlapping Peak Resolution**
    - **Property 14: Weak Peak Detection**
    - **Property 15: Numerical Stability**
    - **Validates: Requirements 5.1-5.8, 8.1-8.5**

- [x] 9. Checkpoint - 确保改进算法通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. 测试报告生成器

  - [x] 10.1 实现 TestReporter 类
    - 接收评估结果和测试用例
    - _Requirements: 6.1_

  - [x] 10.2 实现 generate_summary_table 方法
    - 生成 pandas DataFrame
    - 包含所有测试结果
    - _Requirements: 6.1_

  - [x] 10.3 实现 generate_error_plots 方法
    - 误差 vs 事件数
    - 误差 vs 峰分离度
    - 误差 vs β 值
    - 误差 vs 径向位置
    - _Requirements: 6.4_

  - [x] 10.4 实现 identify_failure_conditions 方法
    - 找出 r0 误差 > 5% 的条件
    - 找出 β 误差 > 0.2 的条件
    - _Requirements: 6.2, 6.3_

  - [x] 10.5 实现 compute_pass_rate 方法
    - 计算各参数的通过率
    - 计算总体通过率
    - _Requirements: 6.5_

  - [x] 10.6 实现 save_results 方法
    - 支持 JSON 格式
    - 支持 CSV 格式
    - _Requirements: 6.6_

  - [x] 10.7 实现 compare_algorithms 方法
    - 比较两个算法版本
    - 计算改进指标
    - _Requirements: 6.7_

  - [ ]* 10.8 编写报告生成属性测试
    - **Property 16: Pass Rate Calculation Correctness**
    - **Validates: Requirements 6.5**

- [ ] 11. 集成测试脚本

  - [x] 11.1 创建 run_vmi_tests.py 主脚本
    - 命令行参数解析
    - 支持选择测试子集
    - _Requirements: 1.9_

  - [x] 11.2 实现完整测试流程
    - 生成测试用例
    - 运行模拟（带噪声）
    - 执行重建
    - 评估性能
    - 生成报告
    - _Requirements: 1.1-6.7_

  - [x] 11.3 实现性能极限分析
    - 找出达到 5% 误差所需最小事件数
    - 找出达到 5% 误差所需最小分离度
    - 识别算法弱点
    - _Requirements: 3.8, 6.2, 6.3_

- [x] 12. Final Checkpoint - 运行完整测试套件
  - [x] 运行完整测试套件（37个测试用例）
  - [x] 生成最终性能报告（vmi_test_results.json）
  - [x] 识别需要进一步改进的领域

## Notes

- Tasks marked with `*` are optional property-based tests
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- 所有模拟默认使用带噪声的 XY 数据（output_mode='xy_dld'）
- 性能目标：标准条件下 r0 < 5%, σ < 50%, β < 0.2

## 改进建议（未来工作）

1. **多峰检测改进**
   - 实现迭代峰检测（先找最强峰，减去后找下一个）
   - 使用贝叶斯方法估计峰数量

2. **内侧位置峰**
   - 改进中心检测算法
   - 使用更小的内边界阈值

3. **重叠峰分辨**
   - 实现多峰联合拟合
   - 使用正则化约束

4. **σ 估计**
   - 考虑 Abel 投影效应的校正
   - 使用前向模型拟合

