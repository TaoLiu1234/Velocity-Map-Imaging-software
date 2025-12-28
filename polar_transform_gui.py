"""
极坐标转换GUI工具
功能：
1. 读取electron_shilpa_XY.mat数据
2. 显示原始散点图并找中心
3. 极坐标转换
4. 可调节dr, dtheta的binning显示
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox


class PolarTransformGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("极坐标转换工具 - Polar Transform Tool")
        self.root.geometry("1400x900")
        
        # 数据存储
        self.xy_data = None
        self.center = None
        self.r_data = None
        self.theta_data = None
        self.cbar_binned = None
        
        # ROI圆圈参数
        self.roi_x = 0
        self.roi_y = 0
        self.roi_radius = 50
        
        # 椭圆校准参数
        self.scale_x = 1.0  # X方向缩放
        self.scale_y = 1.0  # Y方向缩放
        self.rotation = 0.0  # 旋转角度（度）
        self.xy_calibrated = None  # 校准后的数据
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        # 加载数据按钮
        ttk.Button(control_frame, text="加载数据", command=self.load_data).pack(pady=10, padx=10, fill=tk.X)
        
        # 中心设置
        center_frame = ttk.LabelFrame(control_frame, text="中心设置")
        center_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(center_frame, text="X中心:").grid(row=0, column=0, padx=5, pady=2)
        self.center_x_var = tk.StringVar(value="0")
        self.center_x_entry = ttk.Entry(center_frame, textvariable=self.center_x_var, width=10)
        self.center_x_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(center_frame, text="Y中心:").grid(row=1, column=0, padx=5, pady=2)
        self.center_y_var = tk.StringVar(value="0")
        self.center_y_entry = ttk.Entry(center_frame, textvariable=self.center_y_var, width=10)
        self.center_y_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Button(center_frame, text="自动找中心", command=self.find_center).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(center_frame, text="应用中心", command=self.apply_center).grid(row=3, column=0, columnspan=2, pady=5)
        
        # ROI圆圈设置
        roi_frame = ttk.LabelFrame(control_frame, text="ROI圆圈设置 (找中心区域)")
        roi_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(roi_frame, text="ROI X:").grid(row=0, column=0, padx=5, pady=2)
        self.roi_x_var = tk.StringVar(value="0")
        self.roi_x_entry = ttk.Entry(roi_frame, textvariable=self.roi_x_var, width=10)
        self.roi_x_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(roi_frame, text="ROI Y:").grid(row=1, column=0, padx=5, pady=2)
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_y_entry = ttk.Entry(roi_frame, textvariable=self.roi_y_var, width=10)
        self.roi_y_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(roi_frame, text="半径:").grid(row=2, column=0, padx=5, pady=2)
        self.roi_radius_var = tk.StringVar(value="50")
        self.roi_radius_entry = ttk.Entry(roi_frame, textvariable=self.roi_radius_var, width=10)
        self.roi_radius_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # ROI半径滑块
        ttk.Label(roi_frame, text="半径滑块:").grid(row=3, column=0, padx=5, pady=2)
        self.roi_radius_scale = ttk.Scale(roi_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                           command=self.on_roi_radius_scale)
        self.roi_radius_scale.set(50)
        self.roi_radius_scale.grid(row=3, column=1, padx=5, pady=2, sticky='ew')
        
        ttk.Button(roi_frame, text="更新ROI", command=self.update_roi).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(roi_frame, text="ROI居中到数据", command=self.center_roi_to_data).grid(row=5, column=0, columnspan=2, pady=2)
        
        # 椭圆校准设置
        calib_frame = ttk.LabelFrame(control_frame, text="椭圆校准 (非正圆校正)")
        calib_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(calib_frame, text="X缩放:").grid(row=0, column=0, padx=5, pady=2)
        self.scale_x_var = tk.StringVar(value="1.0")
        self.scale_x_entry = ttk.Entry(calib_frame, textvariable=self.scale_x_var, width=8)
        self.scale_x_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(calib_frame, text="Y缩放:").grid(row=1, column=0, padx=5, pady=2)
        self.scale_y_var = tk.StringVar(value="1.0")
        self.scale_y_entry = ttk.Entry(calib_frame, textvariable=self.scale_y_var, width=8)
        self.scale_y_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(calib_frame, text="旋转(°):").grid(row=2, column=0, padx=5, pady=2)
        self.rotation_var = tk.StringVar(value="0")
        self.rotation_entry = ttk.Entry(calib_frame, textvariable=self.rotation_var, width=8)
        self.rotation_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # X缩放滑块
        self.scale_x_scale = ttk.Scale(calib_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL,
                                        command=self.on_scale_x_change)
        self.scale_x_scale.set(1.0)
        self.scale_x_scale.grid(row=0, column=2, padx=5, pady=2, sticky='ew')
        
        # Y缩放滑块
        self.scale_y_scale = ttk.Scale(calib_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL,
                                        command=self.on_scale_y_change)
        self.scale_y_scale.set(1.0)
        self.scale_y_scale.grid(row=1, column=2, padx=5, pady=2, sticky='ew')
        
        # 旋转滑块
        self.rotation_scale = ttk.Scale(calib_frame, from_=-45, to=45, orient=tk.HORIZONTAL,
                                         command=self.on_rotation_change)
        self.rotation_scale.set(0)
        self.rotation_scale.grid(row=2, column=2, padx=5, pady=2, sticky='ew')
        
        ttk.Button(calib_frame, text="自动拟合椭圆", command=self.auto_fit_ellipse).grid(row=3, column=0, columnspan=3, pady=3)
        ttk.Button(calib_frame, text="应用校准", command=self.apply_calibration).grid(row=4, column=0, columnspan=3, pady=3)
        ttk.Button(calib_frame, text="重置校准", command=self.reset_calibration).grid(row=5, column=0, columnspan=3, pady=3)
        
        # 极坐标转换
        ttk.Button(control_frame, text="极坐标转换", command=self.polar_transform).pack(pady=10, padx=10, fill=tk.X)
        
        # Binning设置
        bin_frame = ttk.LabelFrame(control_frame, text="Binning设置")
        bin_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(bin_frame, text="dr (像素):").grid(row=0, column=0, padx=5, pady=2)
        self.dr_var = tk.StringVar(value="1.0")
        self.dr_entry = ttk.Entry(bin_frame, textvariable=self.dr_var, width=10)
        self.dr_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(bin_frame, text="dθ (度):").grid(row=1, column=0, padx=5, pady=2)
        self.dtheta_var = tk.StringVar(value="1.0")
        self.dtheta_entry = ttk.Entry(bin_frame, textvariable=self.dtheta_var, width=10)
        self.dtheta_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # dr滑块
        ttk.Label(bin_frame, text="dr滑块:").grid(row=2, column=0, padx=5, pady=2)
        self.dr_scale = ttk.Scale(bin_frame, from_=0.5, to=10, orient=tk.HORIZONTAL, 
                                   command=self.on_dr_scale)
        self.dr_scale.set(1.0)
        self.dr_scale.grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        
        # dtheta滑块
        ttk.Label(bin_frame, text="dθ滑块:").grid(row=3, column=0, padx=5, pady=2)
        self.dtheta_scale = ttk.Scale(bin_frame, from_=0.5, to=10, orient=tk.HORIZONTAL,
                                       command=self.on_dtheta_scale)
        self.dtheta_scale.set(1.0)
        self.dtheta_scale.grid(row=3, column=1, padx=5, pady=2, sticky='ew')
        
        ttk.Button(bin_frame, text="更新Binning图", command=self.update_binning).grid(row=4, column=0, columnspan=2, pady=10)
        
        # 信息显示
        info_frame = ttk.LabelFrame(control_frame, text="数据信息")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=10, width=28)
        self.info_text.pack(padx=5, pady=5)
        
        # 右侧图形区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建2x2的图形布局
        self.fig = Figure(figsize=(12, 9), dpi=100)
        self.axes = {
            'scatter': self.fig.add_subplot(221),
            'polar_scatter': self.fig.add_subplot(222),
            'binned': self.fig.add_subplot(223),
            'radial': self.fig.add_subplot(224)
        }
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        self.fig.tight_layout()
        
    def load_data(self):
        """加载.mat数据文件"""
        try:
            data = sio.loadmat('electron_shilpa_XY.mat')
            self.xy_data = data['XY']
            
            self.update_info(f"数据加载成功!\n点数: {len(self.xy_data)}\nX范围: [{self.xy_data[:,0].min():.2f}, {self.xy_data[:,0].max():.2f}]\nY范围: [{self.xy_data[:,1].min():.2f}, {self.xy_data[:,1].max():.2f}]")
            
            # 自动将ROI居中到数据
            self.center_roi_to_data()
            self.plot_scatter()
            self.find_center()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败: {str(e)}")
    
    def find_center(self):
        """自动找中心 - 只在ROI圆圈内找"""
        if self.xy_data is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        
        # 筛选ROI圆圈内的点
        dist_to_roi = np.sqrt((self.xy_data[:, 0] - self.roi_x)**2 + 
                              (self.xy_data[:, 1] - self.roi_y)**2)
        mask = dist_to_roi <= self.roi_radius
        roi_data = self.xy_data[mask]
        
        if len(roi_data) == 0:
            messagebox.showwarning("警告", "ROI圆圈内没有数据点！请调整ROI位置或大小。")
            return
        
        # 方法1: ROI内的质心
        center_x = np.mean(roi_data[:, 0])
        center_y = np.mean(roi_data[:, 1])
        
        # 方法2: 使用2D直方图找峰值（更鲁棒）
        hist, xedges, yedges = np.histogram2d(roi_data[:, 0], roi_data[:, 1], bins=50)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        peak_x = (xedges[max_idx[0]] + xedges[max_idx[0]+1]) / 2
        peak_y = (yedges[max_idx[1]] + yedges[max_idx[1]+1]) / 2
        
        # 使用质心作为默认
        self.center = np.array([center_x, center_y])
        self.center_x_var.set(f"{center_x:.2f}")
        self.center_y_var.set(f"{center_y:.2f}")
        
        self.update_info(f"中心已找到 (ROI内{len(roi_data)}点)!\n质心: ({center_x:.2f}, {center_y:.2f})\n峰值: ({peak_x:.2f}, {peak_y:.2f})")
        
        self.plot_scatter()
        
    def apply_center(self):
        """应用手动设置的中心"""
        try:
            center_x = float(self.center_x_var.get())
            center_y = float(self.center_y_var.get())
            self.center = np.array([center_x, center_y])
            self.plot_scatter()
            self.update_info(f"中心已更新: ({center_x:.2f}, {center_y:.2f})")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字!")
    
    def polar_transform(self):
        """执行极坐标转换"""
        if self.xy_data is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        if self.center is None:
            messagebox.showwarning("警告", "请先找中心!")
            return
        
        # 使用校准后的数据或原始数据
        if self.xy_calibrated is not None:
            # 校准后的数据已经是相对于中心的
            x_centered = self.xy_calibrated[:, 0]
            y_centered = self.xy_calibrated[:, 1]
            data_type = "校准后"
        else:
            # 相对于中心的坐标
            x_centered = self.xy_data[:, 0] - self.center[0]
            y_centered = self.xy_data[:, 1] - self.center[1]
            data_type = "原始"
        
        # 转换为极坐标
        self.r_data = np.sqrt(x_centered**2 + y_centered**2)
        self.theta_data = np.arctan2(y_centered, x_centered)  # 弧度 [-π, π]
        self.theta_data_deg = np.degrees(self.theta_data)  # 转换为度 [-180, 180]
        
        self.update_info(f"极坐标转换完成 ({data_type}数据)!\nr范围: [{self.r_data.min():.2f}, {self.r_data.max():.2f}]\nθ范围: [{self.theta_data_deg.min():.1f}°, {self.theta_data_deg.max():.1f}°]")
        
        self.plot_polar_scatter()
        self.update_binning()
        
    def update_binning(self):
        """更新binning图"""
        if self.r_data is None or self.theta_data is None:
            messagebox.showwarning("警告", "请先进行极坐标转换!")
            return
        
        try:
            dr = float(self.dr_var.get())
            dtheta = float(self.dtheta_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的dr和dθ值!")
            return
        
        # 计算bin数量
        r_max = self.r_data.max()
        n_r_bins = int(np.ceil(r_max / dr))
        n_theta_bins = int(np.ceil(360 / dtheta))
        
        # 创建2D直方图 (r, theta)
        r_bins = np.linspace(0, r_max, n_r_bins + 1)
        theta_bins = np.linspace(-180, 180, n_theta_bins + 1)
        
        hist, r_edges, theta_edges = np.histogram2d(
            self.r_data, self.theta_data_deg, 
            bins=[r_bins, theta_bins]
        )
        
        self.plot_binned(hist, r_edges, theta_edges, dr, dtheta)
        self.plot_radial_distribution(r_bins)
        
    def on_dr_scale(self, value):
        """dr滑块回调"""
        self.dr_var.set(f"{float(value):.1f}")
        # 实时更新binning图
        if self.r_data is not None:
            self.update_binning()
        
    def on_dtheta_scale(self, value):
        """dtheta滑块回调"""
        self.dtheta_var.set(f"{float(value):.1f}")
        # 实时更新binning图
        if self.r_data is not None:
            self.update_binning()
    
    def on_roi_radius_scale(self, value):
        """ROI半径滑块回调"""
        self.roi_radius_var.set(f"{float(value):.0f}")
    
    def update_roi(self):
        """更新ROI圆圈"""
        try:
            self.roi_x = float(self.roi_x_var.get())
            self.roi_y = float(self.roi_y_var.get())
            self.roi_radius = float(self.roi_radius_var.get())
            if self.xy_data is not None:
                self.plot_scatter()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的ROI参数!")
    
    def center_roi_to_data(self):
        """将ROI圆圈居中到数据中心"""
        if self.xy_data is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        
        # 使用数据的质心作为ROI中心
        self.roi_x = np.mean(self.xy_data[:, 0])
        self.roi_y = np.mean(self.xy_data[:, 1])
        self.roi_x_var.set(f"{self.roi_x:.2f}")
        self.roi_y_var.set(f"{self.roi_y:.2f}")
        self.plot_scatter()
    
    def on_scale_x_change(self, value):
        """X缩放滑块回调"""
        self.scale_x_var.set(f"{float(value):.3f}")
    
    def on_scale_y_change(self, value):
        """Y缩放滑块回调"""
        self.scale_y_var.set(f"{float(value):.3f}")
    
    def on_rotation_change(self, value):
        """旋转滑块回调"""
        self.rotation_var.set(f"{float(value):.1f}")
    
    def auto_fit_ellipse(self):
        """自动拟合椭圆参数"""
        if self.xy_data is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        if self.center is None:
            messagebox.showwarning("警告", "请先找中心!")
            return
        
        # 相对于中心的坐标
        x_centered = self.xy_data[:, 0] - self.center[0]
        y_centered = self.xy_data[:, 1] - self.center[1]
        
        # 使用协方差矩阵拟合椭圆
        cov = np.cov(x_centered, y_centered)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # 排序特征值（从大到小）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 计算椭圆参数
        a = np.sqrt(eigenvalues[0])  # 长轴（标准差）
        b = np.sqrt(eigenvalues[1])  # 短轴（标准差）
        
        # 旋转角度：主轴方向
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # 校正策略：先旋转使主轴对齐坐标轴，然后缩放使长短轴相等
        # 旋转角度：将主轴旋转到X轴
        rotation_correction = -angle
        
        # 缩放：将长轴缩短到短轴长度（或将短轴拉长到长轴长度）
        # 这里选择缩短长轴
        scale_ratio = b / a if a > b else 1.0
        
        self.scale_x_var.set(f"{scale_ratio:.3f}")
        self.scale_y_var.set("1.000")
        self.rotation_var.set(f"{rotation_correction:.1f}")
        
        # 更新滑块
        self.scale_x_scale.set(min(max(scale_ratio, 0.5), 2.0))
        self.scale_y_scale.set(1.0)
        self.rotation_scale.set(min(max(rotation_correction, -45), 45))
        
        self.update_info(f"Ellipse fit done!\nMajor axis: {a:.2f}\nMinor axis: {b:.2f}\nRatio: {a/b:.3f}\nAngle: {angle:.1f} deg\n\nSuggested params set")
    
    def apply_calibration(self):
        """应用椭圆校准"""
        if self.xy_data is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        if self.center is None:
            messagebox.showwarning("警告", "请先找中心!")
            return
        
        try:
            scale_x = float(self.scale_x_var.get())
            scale_y = float(self.scale_y_var.get())
            rotation = float(self.rotation_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的校准参数!")
            return
        
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.rotation = rotation
        
        # 相对于中心的坐标
        x_centered = self.xy_data[:, 0] - self.center[0]
        y_centered = self.xy_data[:, 1] - self.center[1]
        
        # 步骤1: 先旋转（将椭圆主轴对齐到坐标轴）
        theta_rad = np.radians(rotation)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        x_rot = x_centered * cos_t - y_centered * sin_t
        y_rot = x_centered * sin_t + y_centered * cos_t
        
        # 步骤2: 再缩放（将椭圆变成圆）
        x_scaled = x_rot * scale_x
        y_scaled = y_rot * scale_y
        
        # 保存校准后的数据（相对于中心，即以原点为中心）
        self.xy_calibrated = np.column_stack([x_scaled, y_scaled])
        
        self.update_info(f"Calibration applied!\nX scale: {scale_x:.3f}\nY scale: {scale_y:.3f}\nRotation: {rotation:.1f} deg\n\nData points: {len(self.xy_calibrated)}")
        
        self.plot_scatter()
        # 自动更新极坐标转换和binning
        self.polar_transform()
    
    def reset_calibration(self):
        """重置校准参数"""
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.rotation = 0.0
        self.xy_calibrated = None
        
        self.scale_x_var.set("1.0")
        self.scale_y_var.set("1.0")
        self.rotation_var.set("0")
        self.scale_x_scale.set(1.0)
        self.scale_y_scale.set(1.0)
        self.rotation_scale.set(0)
        
        if self.xy_data is not None:
            self.plot_scatter()
            # 如果已有极坐标数据，重新转换
            if self.r_data is not None:
                self.polar_transform()
        
        self.update_info("Calibration reset")
        
    def plot_scatter(self):
        """绘制散点图"""
        ax = self.axes['scatter']
        ax.clear()
        
        # 判断是否有校准数据
        if self.xy_calibrated is not None:
            # 校准后的数据已经是以原点为中心
            x_plot_full = self.xy_calibrated[:, 0]
            y_plot_full = self.xy_calibrated[:, 1]
            is_calibrated = True
        else:
            # 原始数据
            x_plot_full = self.xy_data[:, 0]
            y_plot_full = self.xy_data[:, 1]
            is_calibrated = False
        
        # 采样显示（数据量大时）
        n_points = len(x_plot_full)
        if n_points > 50000:
            idx = np.random.choice(n_points, 50000, replace=False)
            x_plot = x_plot_full[idx]
            y_plot = y_plot_full[idx]
            title_suffix = f" ({50000}/{n_points})"
        else:
            x_plot = x_plot_full
            y_plot = y_plot_full
            title_suffix = ""
        
        ax.scatter(x_plot, y_plot, s=0.1, alpha=0.3, c='blue')
        
        # 绘制ROI圆圈和中心
        if is_calibrated:
            # 校准后数据以原点为中心
            roi_x_display = self.roi_x - self.center[0]
            roi_y_display = self.roi_y - self.center[1]
            center_x_display = 0
            center_y_display = 0
        else:
            roi_x_display = self.roi_x
            roi_y_display = self.roi_y
            center_x_display = self.center[0] if self.center is not None else 0
            center_y_display = self.center[1] if self.center is not None else 0
        
        roi_circle = plt.Circle((roi_x_display, roi_y_display), self.roi_radius, 
                                 fill=False, color='green', linewidth=2, linestyle='--', label='ROI')
        ax.add_patch(roi_circle)
        
        if self.center is not None:
            ax.scatter(center_x_display, center_y_display, s=100, c='red', marker='+', linewidths=2, label='Center')
            ax.axhline(y=center_y_display, color='r', linestyle='--', alpha=0.5)
            ax.axvline(x=center_x_display, color='r', linestyle='--', alpha=0.5)
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        calib_str = " [Calibrated]" if is_calibrated else ""
        ax.set_title(f'Scatter Plot{calib_str}{title_suffix}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def plot_polar_scatter(self):
        """绘制极坐标散点图"""
        ax = self.axes['polar_scatter']
        ax.clear()
        
        n_points = len(self.r_data)
        if n_points > 50000:
            idx = np.random.choice(n_points, 50000, replace=False)
            r_plot = self.r_data[idx]
            theta_plot = self.theta_data_deg[idx]
        else:
            r_plot = self.r_data
            theta_plot = self.theta_data_deg
        
        scatter = ax.scatter(theta_plot, r_plot, s=0.1, alpha=0.3, c=r_plot, cmap='viridis')
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('r')
        ax.set_title('Polar Scatter (r vs theta)')
        ax.set_xlim(-180, 180)
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def plot_binned(self, hist, r_edges, theta_edges, dr, dtheta):
        """绘制binning后的热图"""
        ax = self.axes['binned']
        
        # 清除旧的colorbar（如果存在）
        if hasattr(self, 'cbar_binned') and self.cbar_binned is not None:
            try:
                self.cbar_binned.ax.remove()
            except:
                pass
            self.cbar_binned = None
        
        ax.clear()
        
        # 使用pcolormesh绘制
        im = ax.pcolormesh(theta_edges, r_edges, hist, cmap='hot', shading='auto')
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('r')
        ax.set_title(f'Polar Binning (dr={dr:.1f}, dtheta={dtheta:.1f} deg)')
        
        # 添加新的colorbar
        self.cbar_binned = self.fig.colorbar(im, ax=ax, label='Counts')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_radial_distribution(self, r_bins):
        """绘制径向分布"""
        ax = self.axes['radial']
        ax.clear()
        
        hist, _ = np.histogram(self.r_data, bins=r_bins)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        ax.plot(r_centers, hist, 'b-', linewidth=1)
        ax.fill_between(r_centers, hist, alpha=0.3)
        ax.set_xlabel('r')
        ax.set_ylabel('Counts')
        ax.set_title('Radial Distribution')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def update_info(self, text):
        """更新信息显示"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)


def main():
    root = tk.Tk()
    app = PolarTransformGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
