# VMI_workflow 架构文档

> 生成时间:2026-08-25
> 范围:本仓库根目录下三个源文件 + 数据目录(数据文件不入库)
> 阅读对象:后续维护者 / 希望理解系统如何工作的学生

---

## 0. 运行环境与依赖

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 Pro (x64),AMD Ryzen 7 7840HS |
| Python | 3.11.11 (conda-forge) |
| numpy / scipy | 2.4.6 / 1.16.0 |
| PySide6 | 6.7.2 (matplotlib `QT_API=pyside6` + `QtAgg`) |
| pyabel | 0.9.1 |
| matplotlib | 3.10.8 |
| 可选(未用) | pandas 2.3.1, pyarrow 19.0.0(基准测试验证,见 §7.1);numba 不可用(numpy 2.4 不兼容) |

数据文件:CSV 文本,每行 4 列 `事件序号, 离子TOF时间, 电子索引, 离子索引`,列值可为 `NaN`(表示该事件该通道缺失)。参考文件 220–260 MB,数千万行。

---

## 1. 代码库全景

| 文件 | 行数(2026-08-31) | 职责 | 依赖 Qt? |
|---|---|---|---|
| `VMI_workflow.py` | 23,177 | GUI、交互、全部业务编排(`MainWindow` 单类) | 是 |
| `VMI_workflow_core.py` | 3,139 | 纯 numpy/scipy 科学计算:配对、中心估计、极坐标投影、去噪分箱 | 否 |
| `VMI_workflow_reconstruction.py` | 325 | pyabel rBasex 调用 + 峰值提取(backward 回退引擎已在 §16 移除) | 否 |

设计意图(文件头注释):计算核心与 GUI 分离,便于测试与阅读。实际执行:两个计算模块确实无 Qt 依赖,但**绝大多数算法仍直接写在 `MainWindow` 方法里**(背景拟合、TOF 对齐、旋转、校准、画布交互),分离不彻底。

### 1.1 顶层小类
- `FileDropFrame(QFrame)`(L162):拖放或浏览单文件,`file_dropped` 信号。
- `_SelectorToggleProxy`(L223):替代 matplotlib selector 的最小代理(仅 `active` 标志),用于离子-TOF ROI 自定义拾取。

### 1.2 常量(节流/性能护栏)
L66–86 定义了 ~20 个调优常量:`MAX_SCATTER_POINTS=25_000`(散点图最大点数)、`MAX_ION_COINCIDENCE_POINTS=120_000`、`SCATTER_HEATMAP_THRESHOLD=8_000`、`DRAG_PREVIEW_INTERVAL_MS=16`、`OVERLAY_EDIT_DEBOUNCE_MS=70`、`WHEEL_SCROLL_COALESCE_MS=12` 等。(原 `ION_TOF_BG_POINTWISE_K=28` 对应的 kd-tree 逐点后端曾是死代码,已在 §16 删除。)

---

## 2. UI 结构

```
MainWindow (1380×980)
├─ 顶部文件栏 file_bar_row:Files btn / Settings[Tab] toggle / Load / Process and Plot / Save Session /
│  Load Session / Clear / Trigger Mode 下拉
├─ 状态栏:Status 标签 + QProgressBar(完成时隐藏)
├─ plot_settings_splitter (QSplitter.Vertical)
│  ├─ plot_panel(绘图区)
│  │  ├─ NavigationToolbar (matplotlib QtAgg)
│  │  ├─ plot_scroll (QScrollArea, both scrollbars 常显)
│  │  │  └─ plot_canvas_host (QStackedLayout StackAll)
│  │  │     ├─ figure+canvas (27.0×8.6 英寸)
│  │  │     └─ plot_scroll_preview_label(滚动手势时的实时位图预览蒙层)
│  │  └─ h_view_slider(水平视图滑块,通常隐藏)
│  └─ settings_panel(Settings Tray,最小高 140px,启动时隐藏,按 Tab 显示)
│     └─ control_tabs (QTabWidget,7 个标签)
│        ├─ File(文件来源 + Workflow 须知)
│        ├─ Ion Histogram(ROI/分箱/MQ 参考/背景拟合/显示)
│        ├─ Ion Coincidence(TOF 图参数、TOF 背景模型、TOF 对齐拟合)
│        ├─ Electron Scatter(中心估计、圆参数、过滤、极坐标 ROI)
│        ├─ Ion Scatter(离子过滤、旋转、TOF 中心校正)
│        ├─ Electron Binned Image(中心 bin 大小、θ 剖面、电压)
│        └─ Reconstruction(rBasex 参数)
└─ status bar(status_label + progress_bar)
```

### 2.1 子图网格(2×4 gridspec)
```
[ion_histogram     | electron_scatter | centered_bin      | rBasex recon]
[ion_tof_xy        | ion_scatter      | theta_profile     | rBasex radial profile]
```
8 个轴对象均注册于 `self.subplot_axes`,被刷新函数与子图保存/复制逻辑索引。(旧版为 2×5,含 backward recon 与 summary/info 两块;这两块面板连同整个 backward 机制已分别于 §15/§16 移除。)

### 2.2 交互层
- 画布全局事件: `mpl_connect` 了 `button_press/motion/button_release/pick/draw` 五个事件,统一进 `_on_canvas_press/move/release/pick`。
- **滚轮滚动**:全局 `eventFilter`(L2 级)把滚轮事件按控件身份派发到滚动区或画布;画布内滚轮进入**滚动爆发(burst)模式**:`_capture_plot_scroll_preview_pixmap` 抓取当前画布为 QPixmap 铺在顶层蒙层,底层 `plot_scroll` 快速滚动,松手后 `_flush_deferred_plot_scroll_restore` 恢复。这是为了让高分辨率 matplotlib 图滚动不卡顿的巧思。
- **拖拽预览**:所有画布拖拽(环拖动、离子滤罩拖动、极坐标 ROI、θ 线、rBasex 范围)只更新 `pending_*` 变量,由节流 QTimer(16ms)触发 `_flush_*` 重绘 overlay artists。
- **Overlay 绘制**:`_draw_circle_overlay`/`_draw_ion_overlay` 等纯 matplotlib artists;快路径 `_present_*_from_background` 用 `restore_region`+`draw_artist` 做 blit,但 **Windows 上全部被 `_use_safe_*_redraw()` 强制关闭**(L8397-+),降级为全画布重绘(`_draw_canvas_without_overlay_draw_event_sync` + 重新捕获背景)。→ 交互流畅性瓶颈之一。

---

## 3. 数据管线(7 步官方工作流)

### Step 1 文件加载与缓存
`load_cache()`(L7 主程序):`np.loadtxt` 三个文件 → `CacheData(trigger_indices[electron,ion], electron_points[i,t,x,y?], ion_points[...])`。`select_*` 触发配对。注:trigger 文件按 `[ion,electron]` 读入后交换为内部 `[electron,ion]`(见代码注释"Input trigger order switched for test")。

- `np.loadtxt(usecols=...)`:注意**它仍会解析整行所有 token**。基准:`np.loadtxt`(200 万行 95.6MB)1.44s;真实 260MB 文件约 3–5 s。三文件串行,期间 UI 阻塞(WaitCursor)。

### Step 2 触发(TDC)配对
`select_increment_pairs` / `select_*_one_pairs` / `select_1e2i` / `select_1e3i`(core.py L41-190)全部骨架向量化:相邻行差分 `Δe, Δi` 比较,取满足 `(Δe,Δi)=(1,1)` 等条件的行;1e/2i 与 1e/3i 展开为 `(e,i-1),(e,i)` 或三倍行。已经**:O(N) `interable,无 Python 循环**——没有明显优化空间。

结果 `paired_lookup_e_idx/paired_lookup_i_idx`` 直接当作索引 lookup electron_points/ion_points 的行,**不物化整表复本**(`matched_electron=empty`),下游按需 `_paired_points(mask)`。

### Step 3 离子直方图选择
- bin 度、ROI(coarse x-range)、fine ROI;绘制在 `ax_hist_ion`。
- `_ion_hist_cache`(key = data_version+bins+coarse ROI+axis 变换 tag+背景 key)缓存 counts/edges;增性失效 `_invalidate_ion_hist_cache`。
- 额外带 **背景拟合子系统**(§4.6)与 **m/q 轴**(§4.7)。
- `_selected_mask()`(L6)根据 fine ROI + 峰值 marker 组焊 bool mask,缓存于 `_selected_mask_cache_key`。

### Step 4 散射过滤
`_selected_pairs_after_optional_ion_filter()`:
1. `_selected_mask` → fine ROI 基准;
2. 若启用, 应用 TOF 背景 keep mask(bg_keep);
3. `_paired_points(mask)` 物化 `(x,y,t)` 电子点与离子点;
4. 可选:离子矩形/密度过滤(`_density_filter_mask`);可选电子密度过滤;返回携带各阶段掩码与 lookup 索引的 dict。

**注意**:密度/过滤掩码对 **所有**被选点每轮重算;`_density_filter_mask` 内部 `density_counts_from_bins`(2D bin count,线性) + top-k % 或 bottom-M 剔除。用 `np.digitize` + `np.bincount`, 复杂度 O(n),但每次重算。

### 5 电子散射中心估计(bundle)
`estimate_center_once`(L5)→ 一次估计(单次,无迭代循环),中心模式下拉 `center_mode_combo` 决定算法:
1. `centroid` - 均值;
2. `geo_median` - `geometric_median`(Weiszfeld 型固定点迭代,最多 120 步);
3. `edge_fit` - `edge_circle_center`(边缘包络→Kasa 圆拟合);
4. `polar_outermost` - `polar_outermost_center`(冻结外层壳的扇形模型 + 解析梯度优化);
5. `quadrant_symmetry` - `quadrant_symmetry_center`(对角象限对称性,raw 点匹配)。
后续 `_center_curve_metrics` 用 `build_polar_histogram` 计算“直线度”评估候选,接受条件 `_center_metrics_better`(score/sigma/valid 综合)。polar_outermost 还有多探针回退(polar ROI 下的确定性探针)。

全部在 GUI 线程同步执行,**期间完整 `_refresh_plots`**。

### Step 6 `apply_circle_selection`(核心投影)
1. 获取 filtered pairs(Step 4);
2. 环选择: `dist2<=inner²`,可选 `outer_ring_filter` 收集外环噪声点;
3. 中心化: `centered_signal` = electron−center;
4. `build_denoised_centered_histogram`(core.py,细节 §4.8): 外环噪声密度均匀化减到内环 bin;`hist_denoised`、`hist_signal`、edges、bin_size 字典返回;
5. 清除旧的重建/剖面选择状态, `_refresh_after_circle_clear(partial)`。

### Step 7 重建
`run_reconstruction_now`(2026-08-31 现状,backward 已于 §16 移除):
- 设置 `_get_rbasex_settings`;
- rBasex 在 **后台线程** `_ReconWorker` 中调 pyabel `rbasex_transform(direction="inverse")`(见 §13.3b),带进度回调 `10–90%`,点击后 UI 立即返回;
- 完成后由主线程 `_collect_recon_results` 刷新所有面板;重复点击受 `_recon_busy` 门控。

---

## 4. 关键算法实现详解

### 4.1 触发配对(core.py:41–189)
- `_select_strict_delta_pairs`: 相邻行差分向量化。`valid` 要求两行列都非 NaN,rint 取整后比较。O(N)。
- 1e2i 展开:对每个匹配行输出 `(e,i-1),(e,i)`,剪除 `i<0`。
- 优点:纯向量,极快;多个模式统一 `delta_e/delta_i` 参数;等。缺点:相邻行必须是**连续数据行**(丢失跨文件边界的配对行,末行);对数据文件已排序假设,无排序防御;行间差依赖原始 CSV 顺序,若文件由多次采集拼接(如 `merge_trigger`),边界处会产生假配对——文件命名带 `merged_trigger` 暗示实际数据即拼接,此点值得注意。

### 4.2 重建(reconstruction.py)
**(a) rBasex**:`abel.rbasex.rbasex_transform(direction="inverse", order=settings["order"], odd=..., reg=..., rmax=...)`,`basis_dir=None`(**每次都重算基集——pyabel 默认缓存于 `abel` 数据目录",实际 `basis_dir=None` 会用默认缓存目录 ~/.abel),`rIbeta()` 取出 `(r,I,β,...)`, `r*=bin_size`,峰值提取:
`extract_peak_r_beta`:高斯平滑 → 归一化 → `scipy.signal.find_peaks` → 山谷裁剪 → 对每个峰 `np.trapezoid` 积分区间面积。返回 `(r,β,i,area)`。

**(b) backward 模型(内置回退)——已于 §16 整体移除,以下为历史记录**:`Abel_backward_reconstruction.py` **不存在**,因此**实际执行的是文件内回退引擎**(L56-后面的的三段:phase0/1/2):
- Phase0 `_init_shared_data`:笛卡尔—极坐标立方插值 `map_coordinates(order=3)`,径向平均轮廓,尾部(80%半径以外)噪声 std;
- Phase1 `_phase1_radial_analysis`:平滑+FWHM 峰定位 +`find_peaks`,过滤 mask_radius;峰 → (r,σ,amp,SNR);
- Phase2 `_phase2_angular_analysis`:多半径均值极角谱 → `P2(cosθ)` 最小二乘 `[1,P2]` 拟合 → β=c2/c0(约束 −2..2);
- `reconstruct_2d_from_params`: 叠加各峰高斯环 `amp·exp(−(r−r0)²/2σ²)·(1+β·P2)`。
- 注意:**docs 声称的完整 forward-fit 引擎不存在**,该回退是“参数化高斯环拟合”,不是严格 Abel 逆变换;`reg`/`baseline` 等很多设置实际未用。

### 4.3 中心估计器(core.py)
| 估计器 | 算法 | 复杂度 | 优点 | 缺点 |
|---|---|---|---|---|
| `geometric_median`(L23) | Weiszfeld 迭代 | O(k·n) k≤120 | 抗噪声 | 对均匀环对称数据不敏感,偏质心 |
| `circle_fit_kasa`(L25) | 代数最小二乘圆 | O(n) | 快 | Kasa 对短弧/噪声有系统偏移(代数偏差)|
| `edge_circle_center`(L29) | 1°/bin 包络→圆拟合 | O(n+180²) | 快速已有拟合 | **每 bin 只保留最远 1 点,对离群值脆弱**,相角覆盖不完整时劣化 |
| `quadrant_symmetry_center`(L331) | 方位角配对: 每个角度扇区内部对称残差; `_coarse_search` 网格 + 迭代 | 每次候选都建 `cKDTree` + `query_ball_point` | 对环形分布好 | **每次候选重建 KDTree,O(k·n·query)**;初始化半径依赖 |
| `polar_outermost_center`(L990) | 冻结“最外层 ring”散射模型, `_scatter_peak_line_loss_grad` 解析梯度 + `_optimize_scatter_peak_line_model` 单调更新 | 每轮 loss 重新计算所有点的 `(dx,dy,rr,θ)` | 收敛稳定 | **跨迭代无缓存,每 loss 重复 O(n) 极坐标** + 模型重建 |
| `_iterative_outer_roi_edge_circle_center`(L1922) | 窄环形 ROI 的对比图 →平滑圆周路径 →圆拟合→ 迭代 | 每次迭代重做整幅对比图 | 手动 ROI 场景稳 | 对 2D 直方图重建瓶颈({对比图全图}) |

共享前处理:`points[::step]` 采样 ≤ ~ 64k 点再进入估计器(`_scatter_peak_line…`)。

### 4.4 极坐标直方图与峰值线
`build_polar_histogram`(L819):以某点为圆心将 (x,y) 按 `θ=atan2`, `r=hypot` 分 bin(θ 默认 360 列? 由调用者定;r bin 数沿动态),得到二维热图;`_select_polar_peak_line` 逐 θ 列找峰值;`*_loss_grad` 用“峰位水平平直”作为切向约束(straightness)求梯度,梯度把 `∂r/∂center` 传给解析公式。

### 4.5 m/q 校准
`_ion_mq_calibration_params`(L 8980):平方律 `m/q = a·t² + b`, `b=0`;参考点 `(m/q_ref, t_ref)` 定 `a = m_ref/t_ref²`,若给 TOF 范围则对 `[t_lo,t_hi]→[m_ref−½, m_ref+½]` 再做最小二乘标定 `a`。整数 m/q bin ↔ 对应 TOF 区间(确定性)。优:物理清晰;弱:拟合器只有一个自由参数,若 TOF 偏移(如 时间零点 delta)则系统误差。

### 台位 4.6 离子直方图背景拟合(121-13690)
两条并行路线 + 一个回退:
1. 包络基线 `_estimate_ion_hist_log_envelope_baseline`(熵): 对数空间滚动分位数(纯 Python 滚动循环 → 慢)→ 可分离平滑 → 加权 isotonic 增回归 → 指数 → 最小累积(`np.minimum.accumulate`) 得 **under-signal 单调基线**。
2. 自适应组件 `_fit_ion_hist_background_curves_adaptive_raw`(L34):候选 m 幂集合 + 软偏移,权重最小二乘线性组件(`_solve_weighted_nonnegative_components`),目标下打分;再 `_project_nonnegative_components_under_target` 强制非负与 “不越过唯一信号区域”。
3. 非参数回退 `_build_nonparametric_bg_state`(L46):log-velope smoothing profile。
 评分:自适应用 under-target SSE;NNLS 搜索用 BIC 选模型(complexity 加入选择)。

性能⚠️:`_rolling_quantile`、isotonic 块合并、逐 shape-profile 拟合都是 **Python 循环 + O(n) 每迭代**,有窗口 几万–几十万级数据时可达数百 ms–秒级;且 **每次 `_ensure_ion_hist_background_fit` 仅在 cache key 变更时重算** —— 否则复用,是缓存设计对的。

### 4.7 Ion TOF 背景模型(1-TO-F 图, visualization)
(实际代码,非点式 kd-tree):
- `_fit_ion_tof_background_model_raw`(L~14742):先 平滑 XY 直方图(密度),再 `_fit_radial_floor_profile`(径向地板轮廓),逐点 `score = bg_density/all_density`, `_choose_score_threshold` 用分位数 / 直方图质量自动定阈值, mask 掉低分事件报背景 keep。
- `_ensure_ion_tof_bg_model` 按 (source key, xy transform key, params) 记忆化。
- **死代码(已删除,§16)**: `_adaptive_xy_kde_density` 及 `ION_TOF_BG_POINTWISE_K`(K=28) 的逐点 KNN 路径曾定义且无调用者,实际用的是 histogram-density + radial floor 版本(更快、更稳,但也更粗);两已于 §16 移除。

### 4.8 去噪居中分箱(core.py:2946–3018)
```
信号直方图(signal_hist)  ← centered_signal
外环密度 = noise_count / (π(outer²−inner²))
expected_per_bin = density × bin_size²
内环 mask(圆内)全部减 expected_per_bin → clamp ≥0
removed_total = Σsignal − Σdenoised
```
- 优点:单次同型减法,极快,数学可解析。
- 缺点(准确度):**假设背景在内环内均匀**(实际 VMI 背景在 r 方向有径向涨落、心部常偏高);对 Poisson 计数噪声不做双侧校正; `denoised<0→0` 造成截断偏差(负值丢失信息,均值上移)。改进:径向自适应背景或随 r 的 Poisson 校正(见 §10 改进项)。

### 4.9 旋转 / 对齐
- `_apply_ion_rotation`(L8152): 以 `ion_rotation_center` 为基准旋转 `(x,y)` 按 `rotation_deg`;`_transform_ion_xy` 级联:旋转 → TOF 对齐平移 → TOF Z 中心。
- `_fit_ion_tof_main_line`(L):histogram2d 降采样 → 平滑 → 每列 argmax 脊线 → MAD-稳健加权线性回归(line 拟合);`_fit_ion_tof_box_density_line` 逐 box 密度 top-%。 结果存 `ion_tof_fit_result_by_axis`,应用移动到 `(x,t)` 使直线水平。
- 两处之间**变换数学在同一文件内被复制 3–4 遍**(`_apply_ion_tof_alignment_to_xy`/_apply_ion_scatter_tof_center_to_xy/-terms/_ion_tof_display_coord_values)——维护风险点。

---

## 5. 缓存与失效机制

| 缓存 | 键 | 失效触发 |
|---|---|---|
| 触发配对 `_pair_cache_*` | (mode, trigger_ref 对象身份) | `process_and_plot` / `clear` |
| 离子直方图 `_ion_hist_cache` | (data_version, bins, coarse ROI, axis_tag, bg key) | `process_and_plot` 或 controls 改动 |
| fine-ROI 选中 `_mask_cache` | (data_version, ROI, 峰值 marker, bg key) | ROI/标记变更 |
| 散射显示 `_current_scatter_display_* | (selection, filters, subsample) | filters/controls 变更 |
| TOF 背景模型 | (source data, xy transform, params) | `fit_ion_tof_bg_model` 显式重算 |
| TOF 对齐结果 | (axis, transform) | fit/clear |
| rBasex | 基集持久化: `basis_dir=~/.cache/vmi_workflow/abel_basis`(进程内全局 + 跨进程磁盘缓存) | — |
| ion_tof_xy map | (data_version, pair 表 id, paired_count, n, **coarse ROI**, **BG keep 状态**, axis, bins, z-range, 旋转/对齐项)(2026-08-31 修复: 原键不含 ROI/数据指纹) | 数据或选择变化即失效 |

整体缓存设计成熟(键含版本),但 `ion_tof_xy` 缓存键遗漏数据依赖、`display_data` 每次 `_refresh_*` 重算(offset 重复)、background fit 的 isotonic/rolling 循环没有复用为缓存对象。

---

## 6. 性能实测(pandas/pyarrow 对照)

| 操作(200 万行 95.6MB) | 耗时 |
|---|---|
| `np.loadtxt`(现行) | 1.44 s |
| `pandas.read_csv(engine="c")` | 1.33 s |
| `pyarrow.csv.read_csv` | **0.15 s** |

结论:单文件解析差异有限,但三文件× 260MB 的串行加载 + 解析仍是启动冷门;pyarrow 带来 **~10× 读速**。更重要的事实:**GUI 线程内同步阻塞是体验首要问题**, IO 应挪到后台线程。

### 性能瓶颈清单(按优先级)

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| P0-2 ✅已解决(§13.3b 后台线程;§16 backward 移除) | `run_reconstruction` L~18 | 曾为同步 rBasex(基集生成本来就慢)+ backward, 无 worker | 曾卡界面数秒–数十秒 |
| P0-3 ✅已解决(§14.2) | `_use_safe_*_redraw()` Windows | 曾强制全画布重绘代替 blit;每次拖动都全量 draw + recapture background | 拖动交互卡顿 |
| P0-4 ✅基本解决(§16) | `_selected_pairs_after_ion_filter` / `display_data` | 每次 `_refresh_*` 重做过滤 + 密度分箱;多处 refresh 重复 40 行 scatter 块 | 过滤参数热更时重绘风暴(现 selected-pairs 派生已记忆化,refresh 尾部已去重) |
| P0-5 ◐部分解决(§16) | 中心估计器: `quadrant_symmetry` 每候选重建 cKDTree | O(k²·n) 级(现 KDTree 单树提升,逐位一致) | 中心估计耗时秒级(polar_outermost 跨迭代仍无缓存) |
| P1-1 | `_fit_ion_histogram_background_rules_*` | Python 循环滚动分位数/同调 | 百 ms–秒级 |
| P1-2 | `build_denoised_centered_histogram` | 每圈重建含 meshgrid;同一次 应用时可接受 | 低 |
| P1-3 ✅已解决(§16) | `run_rbasex` `basis_dir=None` | 无显式基集缓存管理(依赖 pyabel 内部磁盘缓存) | 首次重建慢(现持久化于 ~/.cache/vmi_workflow/abel_basis) |
| P1-4 ✅已解决(§16) | `_ion_tof_xy_cache` 键缺数据 | 参数不变时数据变化仍命中旧图(现键含 coarse-ROI/BG-mask 指纹) | 数据不一致 |

---

## 10. 准确度风险清单

1. **去噪模型假设均匀内环噪声**(§4.8): 忽略径向涨落, 可能低估/高估信号峰底部 → 影响 rBasex 输入准确性。
2. **Kasa 圆拟合存在代数偏差**(core L252-293): 对低信噪、短弧数据中心估计有偏。
3. **edge_circle_center 每角度仅保留最远点**: 外缘离群点直接污染Kasa。
4. **(历史,§16 已移除)backward 引擎曾是简化高斯模型**(§4.2b): 返回 β 来自峰值局部角度拟合,全局 β(r) 是 `Σβ·radial/Σradial` 加权,不是真 Abel 重建 → 与 rBasex 结果差异大,尤其多峰重叠/阴影。该引擎连同其 UI/会话键已于 §16 删除,现在重建只走 rBasex。
5. **m/q 平方律 `b=0`**: 若 TOF 零点慢,系统偏差; 范围校准用最小二乘但无回归误差反馈。
6. **背景拟合 under-target SSE**: 天然把基线往信号底下压(低估背景) — 信号峰间的谷可能被低估。
7. **峰值线性.Fringes**: `extract_peak_r_beta` 采用 `intensity[idx]` 作为面积 fallback; 若 `beta_profile` 硬 clip −2..2 会扭曲极各向异性分布。
8. **离子/电子索引直接索引**: out-of-range 被裁剪,但不警告用户;多数指数成对但 sample 计数可能被丢弃。

---

## 9. 可维护性问题

1. **巨型上帝类**: `MainWindow` 巨型单类、~23k 行(2026-08-31)。方法边界清楚但**职责未分离**: 间隙/坐标/拟合/plot/缓存全部在同一类。无障碍重构:按域拆 mixin 或 utils(保持 import 兼容)。
2. **重复代码**: 1) 变换数学 3–4 份;2) `_refresh_plots` 中 electron/ion 散射绘制块与 `_refresh_scatter_panels_only` 重复 40+ 行;3) 四个配对模式的进度回调闭包几乎相同;4) 两个颜色条删除+创建模式。
3. **死代码(2026-08-31 修订)**: `_SelectorToggleProxy` **是活代码**(L217 定义,L2540/2547 创建;`.active` 在画布事件处理 L~19063/19088 被 TOF-ROI/TOF-背景拾取消费,clear 时调用 `.clear()`)——本节早先"可能已无使用者"的判断是错的。`_clear_ion_tof_fit_preview_background` 已更名为 `_invalidate_ion_tof_fit_preview_background`(L3549,5 处调用,活代码)。`_adaptive_xy_kde_density`/`ION_TOF_BG_POINTWISE_K` 逐点 KNN 后端已确认死代码并于 §16 删除;`_center_curve_metrics` 中探针逻辑仅少数模式使用(保留)。
4. **依赖外部文件(已解决,§16)**: `Abel_backward_reconstruction.py` 不存在、模块 try-import 与整个 backward 回退引擎已随 §16 移除。
5. 常量/硬编码: `sys.platform` 分支、比例常数散落方法中(names 无常量)。
6. 无单测(`VMI_workflow_core` 设计为易测但仓库中无测试文件——未验证,扫描未发现)。

---

## 11. 改进路线图(保持交互逻辑不变)

### P0 性能(高杠杆,低风险)
1. **异步加载 + pyarrow/pandas C 解析**(✅ 已完成 §13.1): 把 `load_cache` 放 `QThread`/`concurrent.futures`; 用 `pyarrow.csv.read_csv`(或 pandas C 引擎)+ 结果转 numpy; 进度条继续,UI不冻结。列选择用 pyarrow `include_columns`。**风险**: 无,纯粹替换读路径;保留 dtype float64。
2. **异步重建**(✅ 已完成 §13.3b;backward 已于 §16 移除): `run_reconstruction_now` 的 rBasex 与 backward 放入后台线程,完成后回主线程刷新;期间画布不动,状态栏转菊花。**保持行为**: 点击按钮后结果一致。
3. **Windows 安全 blit 改优化**(✅ 已完成 §14.2): 替代外更稳的路径——用 `canvas.copy_from_bbox` + `restore_region` + `draw_artist` 包裹 `try/except` 自动降级;或把 overlay 绘制延迟到 `timerEvent` 批量合并。**风险**: 中(Win 专属绘图),需要人工验证。
4. **`display_data` 派生缓存**(◐ 基本完成 §16:selected-pairs 派生已记忆化): 在 `_refresh_*` 前用 `(selection_version, filter params, data_version)` 判断数组是否相同,相同则复用旧 `electron_show/ion_show/颜色数组`,避免重复密度过滤。
5. **中心估计缓存**(◐ 部分完成 §16:quadrant_symmetry 单树提升,逐位一致): 在 `_scatter_peak_line_model` 冻结点集后预计算 `(r²,θ,cosθ,sinθ)` 常量向量;所有 `loss/grad` 迭代复用;`quadrant_symmetry` 对每个候选用 `KD.query_ball_point` 但 **KDTree 只构建一次**(候选中心集内移动点不变!),直接重大提速。
6. **`basis_dir` 显式持久缓存**(✅ 已完成 §16): 设 `basis_dir=os.path.join(缓存)`, 首个 rBasex 慢,后续毫秒级。
7. **`ion_tof_xy` 缓存键加入数据指纹**(✅ 已完成 §16): `hash(self._current_pair_data_version + coarse_mask sum)` 强制失效。

### P1 精度
 - 去噪改径向自适应(annular density per radius)或 Poisson 模型; 保持 API `build_denoised_centered_histogram(signal, noise, inner, outer, bin)` 输出 dict 结构不变。默认 `flat uniform → 可选 'radial'` 校验收(不改默认可避免科学断裂)。
 - 圆拟合 Kasa → Pratt/Taubin(代数恒定)或者 `scipy.optimize` 非线性最小二乘(radius weighting),`circle_fit_kasa` 作为 fallback 保兼容。
 - `edge_circle_center` 每 bin 存 top-N% 分位数点集代替单一最远点。
 - m/q 允许 `b≠0`(挡: 需要 UI 牺牲精度,默认保持)。

### P2 UI 现代化(不碰交互逻辑)
1. **全局 QSS 主题**: 在 app 启动 `main()` 里 `app.setStyleSheet`(浅色现代化: 圆角、对比度、字体栈 "Segoe UI Variable")替换窗口散落的 inline stylesheet;保留 `QTabWidget/GroupBox` 现有结构规则。
2. **matplotlib 样式**: `rcParams` 全局主题(轴色、网格线、字体大小、Figure 底色 #fafafa),保持子图/坐标系布局不改。
3. **面板与状态栏**: 进度条细化(0-100 分段色)、状态提示弱化;图标按钮换文字+icon 不合格?保持。
4. **绘图速度**: 对已缓存 image 型面板(centered bin、recon、hist) 用 blit/局部图表流畅;text 对象避免每次全部重建。
5. 滚动预览保留(已是亮点)。

### P3 前端
- 拆 `MainWindow` 为几大类(交互、布局、核心、绘图), 保持公共方法名/类名不变 → 行为不变, 但可测。
- 把 public 算法(触发配对、圆心、背景)做成 `VMI_workflow_core` 的纯函数并加单测(np.testing).
- 移除死代码(2.2 列出的), 注意先 grep 调用点再删。
- `.vendor_site`(L1) 与 `_prepend_local_vendor_sitepackages` 若无目录则 no-op, 保留。

---

以下行为将被视为"交互契约",优化时不得改变:
1. 七步工作流按钮顺序与功能完全一致(Load→Process→ROI→Fine→Filter→Estimate→Apply+Bin→Recon)。
2. 所有 `QLineEdit` 文本输入即监听器布局不变(`textChanged` 触发覆绘制/重算)。
3. 拖拽手势(环、滤罩、θ、rBasex 范围、标记、TOF box)语义不变。
4. 会话保存/恢复格式兼容(既有 npz + meta JSON)。
5. 窗口默认尺寸/停靠/标签结构不变。
6. 科学输出: 峰值 r/β/强度 数值可复现(同一数据+同参数)。

---

## 13. 本次已实施的改动清单(2026-08-25)

> 遵循 §12 契约:未改动任何交互逻辑、未改动任何科学计算数值路径。

### 13.1 高速数据加载(性能)
- `VMI_workflow_core.py`:新增 `fast_read_csv_float64(path, *, n_columns, use_columns)`。
  - 首选 **pyarrow C++ CSV 解析器**(SIMD,解析 260MB 文件 ~10× 快于 `np.loadtxt`);
  - `NaN` 文本正确解析为 float64 NaN(真实数据含 `NaN`);
  - 解析异常自动回退(原 `np.loadtxt` 完全同语义),无 pyarrow 环境不破坏功能;
  - 列选择按 `use_columns` 输出顺序,兼容 trigger 文件 4 列取 (2,3)、点文件 3 列取 (0,1,2)。
- `VMI_workflow.py::load_cache` 改用 `fast_read_csv_float64`,进度条分段更新不变。
- 实测(真实参考文件 3 个,共 ~24M 行):trigger 8.6M 行由 `np.loadtxt` 1.53 s → 0.65 s;三文件读取部分合计约 1.4 s。
- 回退路径兼容(若 pyarrow 不可用则 `np.loadtxt` 原样)。

### 13.2 界面现代化(QSS + matplotlib 主题)
- `main()` 增加全局 `app.setStyleSheet` 浅色现代主题:圆角按钮/输入框、聚焦高亮、扁平面板、
  Segoe UI Variable 字体栈、复选框圆角指示器、滚动条现代化、ToolTip 深色。
  作用域限定在通用控件类型,保留既有 `QTabWidget/GroupBox/SettingsPanel` 内联样式优先级。
- `MainWindow.__init__` 尾新增 `matplotlib.rcParams.update`(浅色画布、浅灰网格、清晰的刻度与标签色),
  不触碰任何轴布局/几何参数。
- 可视化验证:窗口背景 `${#f7f8fa}` 生效;全部子图 placeholder 正常渲染。

### 13.3b 后台线程重建(性能)
- `run_reconstruction_now` 重写为**后台线程执行**:
  - 新增 `_ReconWorker(QObject)` 辅助类:纯 numpy/pyabel 计算,**不接触任何控件**,
    进度/结果通过内存字段发布,由主线程定时轮询(--`_recon_progress_timer`, 120ms);
  - 新增 `_collect_recon_results` 主线程收结果并刷新面板;线程经 `quit()+wait(1500)`+`deleteLater`
    优雅回收,不依赖易丢的 `finished` 信号;
  - 运行时点击`Start Reconstruction`后**立即返回**(UI 不冻结),后台算完回主线程刷新,
    结果与原同步路径**逐位一致**;
  - 重复点击受 `_recon_busy` 门控,连续运行稳定。
- 实测(参考数据 + 145×145 直方图):点击即返回(0.01s),1.7s 完成;rBasex 2 峰、backward 1 峰,
  无 Python 警告。

---

## 14. 全 app 交互 30–60Hz 优化(2026-08-25)

> 目标:所有交互动画/刷新/等待达到 30Hz 以上(用户认可 30Hz 即可),**不改变任何交互方式、不损失准确性/清晰度**。

### 14.1 根因(实测)
- 拖动 overlay 时,Windows `_use_safe_*_redraw()` 恒 True → 每次拖动都**整幅 canvas.draw() + 重新截取背景**,实测 **183–408ms/帧**(≈5fps)。
- `_draw_axes_immediate` / `_draw_axes_preview` 在 Windows 上强制整幅 `canvas.draw()`(注释:"QtAgg partial blits crash")→ 所有局部刷新也全幅重绘。
- 686 万行配对数据,散点/投影物化与全量重绘让"操作后刷新"达数秒。

### 14.2 改动
1. **删除 Windows 全幅分支**:`_draw_axes_immediate`、`_draw_axes_preview` 不再按平台降级,统一走 `ax.draw(renderer) + canvas.blit(bbox)` 局部路径(自带 try/except 全幅兜底)。
2. **Overlay 拖动 60Hz**:
   - 新增 `_present_circle_overlay_from_background`(镜像 ion 版):`restore_region + draw_artist + blit`,拖动不再全幅重绘。
   - `_update_circle_overlay_only` / `_update_ion_overlay_only` 的 fast_drag 分支改为:背景缺失才截取(`_capture_scatter_blit_backgrounds`),否则直接贴局部。
   - `_ensure_scatter_overlay_backgrounds` 拖动中复用背景缓存,不再每次重截。
   - `_flush_ion_rotation_preview` 改单轴局部重绘。
   - ion TOF fit 预览已自带局部 blit(启用)。
3. **大点云自动转热图**:`_plot_density_scatter` 在 `count > SCATTER_HEATMAP_THRESHOLD(8000)` 时改走 `_plot_density_heatmap`(image 渲染,61fps 级),低点保留点迹 scatter(清晰度不降)。热图路径是代码里已存在但从未接线的孤儿,现接上。
   - **`load_cache` → 新增 `_LoadWorker(QObject)`**:三个大文件读取/校验在后台线程,进度条轮询,读完成回主线程安装 cache。UI 全程响应。
   - **`estimate_center_once` → 新增 `_CenterWorker(QObject)`**:几何中心估计/metrics/确定性探针在后台线程;主线程只保留输入准备(选区物化,带进度)与结果写回画布。与原逻辑输出逐位一致(纯计算,无 Qt)。

### 14.4 用户回归修复(2026-08-25 第二轮)
用户报告两个我引入的回归,已全部修复并验证:
1. **electron/ion scatter 背景变黑、点迹看不清**:我加的"大点云自动转热图"把点迹换成深色密度图。已撤销该分支,恢复原始 PathCollection 点迹渲染(白底彩色密度点)。验证:两轴 images=0、PathCollections=2。
2. **拖动 filter 出现两个圆心/旧圆残影**:fast-blit `restore_region+draw_artist` 在真机画布会贴旧 overlay 像素。已将 circle/ion 的 fast_drag 改为**整轴 `_draw_axes_immediate` 局部重绘**(失败自动全量兜底),彻底消除残影。验证:拖动 30 帧 circle center marker 恒 1、ion rect 恒 1。
5. **Tab 后子图重新缩放(Tab 卡顿真因)**:面板高度变化会带着 viewport 拉伸画布,matplotlib 因此对所有子图重布局重排。已改为 **canvas 尺寸稳定**(viewport 不再 resize 画布;仅显式窗口模式变化时 `_configure_plot_canvas_size` 重新设目标)。验证:Tab 前后画布 2214×705 不变、8 轴位置逐轴一致、呼出仅 22ms。
6. **electron/ion scatter 的 filter 环被数据点覆盖**:环/滤罩创建时未设 zorder,低于点云 zorder=2。已为 inner/outer 环 z=10、圆心 marker z=11、ion 滤罩 z=10、滤罩中心 z=11。验证:全部 >2,显示在数据上层。

### 15. 布局调整: 移除 Backward Recon 与 Summary 面板
- 按用户要求,`ax_reserved_bottom`(Backward Recon)与 `ax_info`(Summary)面板已移除。
- gridspec 从 2×5 改为 **2×4**:
  ```
  row0: [ion hist | e scatter | centered bin | rBasex recon]
  row1: [ion-tof xy | i scatter | theta profile | rBasex radial profile]
  ```
- **rBasex radial profile 移到原 Backward Recon 位置(row1,col3)**,原 [0,4] 位置取消。
- `ax_reserved_bottom`/`ax_info` 置为 None,所有下游引用带 None 守卫(`_plot_reconstruction_panel`/`_plot_info_panel` 及绘制/标记/占位逻辑),删除 `subplot_axes` 中 `summary`/`backward_reconstruction` 键。
- 验证:figure 恰 8 轴 2×4,radial profile 位于右下(0.80, 0.08),全套流程 + 会话引用无警告。

### 13.4 后续建议(部分已在本轮实施,见 §11/§16)
- ~~把 rBasex/backward 重建与中心估计移入后台线程~~(✅ 已完成 §13.3b/§14.2;backward 已于 §16 移除);
- ~~Windows 安全 blit 改局部重绘~~(✅ 已完成 §14.2);
- ~~`_selected_pairs_after_optional_ion_filter` 派生缓存~~(✅ 已完成 §16 记忆化);
- Kasa→Pratt/Taubin 圆拟合、径向自适应去噪(需您确认科学结果,未实施);
- 拆 `MainWindow` 巨型类、补单测(单测已落地:`tests/test_core.py` + `tests/test_smoke.py`,见 §16e)。

以下行为将被视为“交互契约”,优化时不得改变:
1. 七步工作流按钮顺序与功能完全一致(Load→Process→ROI→Fine→Filter→Estimate→Apply+Bin→Recon)。
2. 所有 `QLineEdit` 文本输入即监听器布局不变(`textChanged` 触发覆绘制/重算)。
3. 拖拽手势(环、滤罩、θ、rBasex 范围、标记、TOF box)语义不变。
4. 会话保存/恢复格式兼容(既有 npz + meta JSON)。
5. 窗口默认尺寸/停靠/标签结构不变。
6. 科学输出: 峰值 r/β/强度 数值可复现(同一数据+同参数)。

---

## 16. 发布加固(Release hardening,2026-08-31)

> 本轮目标:面向 GitHub 公开发布做回归修复、死代码清理、去重、性能收尾与仓库整理。
> 全程遵守上方"交互契约";科学数值路径经 `tests/golden_core.json` / `tests/golden_smoke.json` 逐位锁定。

### 16a. 回归修复
- **`estimate_center_once` 的 `NameError: source_label`**: polar-ROI 路径与 ring-empty 回退路径引用了未定义的 `source_label`(只定义了 `source_prefix`)。两处崩溃路径已修复,并由 `tests/test_smoke.py::run_regression_checks` 的 `check_polar_outermost_center` / `check_ring_empty_center_fallback` 覆盖。
- **启动期 `__init__` 调用恢复**: 占位面板渲染(`_draw_placeholder`)、trigger 模式下拉标签(`_update_trigger_mode_combo_labels`,启动时带 "[events: n/a]" 后缀)、TOF 控件同步,均恢复为启动即调用(`check_startup_placeholders` 锁定)。
- **空选区散点分支恢复**: 离子滤罩选 0 事件时,electron scatter 显示 "No selected points" 标注并清除颜色条,同时保留灰底 context 点(`check_empty_selection_scatter` 锁定)。
- **状态栏文案对齐**: 与 classic 版本的状态文字表述对齐。

### 16b. 移除
- **整个 backward-recon 机制**(§4.2b): UI 控件、`_get_backward_settings`、内置 phase0/1/2 回退引擎、compute 路径与会话键全部删除;三个主文件中已无任何 "backward" 引用。会话恢复对**遗留会话**(含 backward 键)保持宽容——未知键被忽略,不报错。重建现在只有 rBasex 一条路径。
- **summary/backward 僵尸脚手架**: §15 移除两块面板后残留的绘制/占位分支(`_plot_info_panel` 相关逻辑等)清理干净。
- **死 KDE 后端**: `_adaptive_xy_kde_density` 与 `ION_TOF_BG_POINTWISE_K` 的逐点 KNN 路径(§4.7、§9.3)。
- **孤儿 blit 辅助函数**: 无调用者的 `*_from_background`/背景截取残件。
- **`_plot_density_heatmap`**: §14.2 曾接线、§14.4 又撤销的"大点云转热图"路径最终删除(散点保持 PathCollection 点迹)。

### 16c. 去重(行为不变)
- **TOF 变换助手**: `_apply_ion_tof_alignment_to_xy` / `_apply_ion_scatter_tof_center_to_xy` / `-terms` / `_ion_tof_display_coord_values` 的重复变换数学(§4.9 指出的 3–4 份拷贝)收敛到共享助手;`check_ion_tof_alignment` 以 9 位小数锁定变换输出。
- **配对进度工厂**: 四个配对模式几乎相同的进度回调闭包统一为工厂函数。
- **颜色条助手**: 两处"删除+重建 colorbar"模式合并为共享助手。
- **refresh 尾部助手**: electron/ion 散点刷新的重复尾部(§6 P0-4 的 40+ 行重复)抽成公共尾部函数。

### 16d. 性能
- **`ion_tof_xy` 缓存键修复**(§6 P1-4): 键加入 coarse-ROI 与 BG-keep mask 指纹;同点数不同选区不再命中旧图(`check_ion_tof_xy_cache_invalidation` 锁定)。
- **selected-pairs 派生记忆化**(§6 P0-4): 过滤参数与数据版本不变时复用已物化的配对点,不再每轮 `_refresh_*` 重算。
- **rBasex 基集持久化**(§6 P1-3): `basis_dir=~/.cache/vmi_workflow/abel_basis`(进程内 + 跨进程磁盘缓存,`VMI_workflow_reconstruction.py::_rbasex_basis_dir`);首次重建慢,之后毫秒级(`tests/bench_rbasex_basis.py` 验证)。
- **`quadrant_symmetry` 单树提升**(§6 P0-5 部分): 候选搜索的 cKDTree 只构建一次(点集不变),结果与提升前**逐位一致**(`tests/bench_core.py` A/B 对照,锁值测试覆盖)。

### 16e. 仓库与打包
- **独立 git 仓库**: 在项目文件夹根 `git init`(父目录的个人仓库不受影响;无 remote、不推送)。
- **测试基线**: `tests/test_core.py`(数值 goldens)+ `tests/test_smoke.py`(离屏端到端,含 §16a 回归扩展与 TOF 对齐/缓存失效检查);样例数据由 `tests/make_sample_data.py` 确定性再生;详见 `tests/README.md`。
- **打包文件**: `README.md`(英文,含截图)、`LICENSE`(MIT)、`requirements.txt`(numpy>=2.2 / scipy>=1.16 / matplotlib>=3.10 / PySide6>=6.7 / pyabel>=0.9,可选 pyarrow>=15)、`docs/screenshot_main.png`(离屏驱动完整 7 步工作流后 `canvas.grab()` 抓取的真实主题画面)、`.gitignore`(排除 `*.dat`/`*.npz`/`workflow_outputs/`/`Refence data/`/`tests/sample_data/` 等)。
- **真实数据终验**: ~700MB 参考三件套异步加载 → 1e+1i 配对 → 边缘拟合定心 → 环选分箱 → rBasex 完成 → 会话保存/恢复,全部通过(数字见发布记录,不入库)。