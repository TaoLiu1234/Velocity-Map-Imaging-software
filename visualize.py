import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
from PyQt5 import QtWidgets, QtCore
import sys

class ThreeDVisualizer(MainWindow):
    def __init__(self, ion_data=None, electron_data=None):
        super().__init__()

        # 创建主框架
        self.frame = QtWidgets.QFrame()
        self.h_layout = QtWidgets.QHBoxLayout()  # 水平布局：3D视图 + 右侧控制面板

        # 创建PyVista渲染窗口
        self.plotter = QtInteractor(self.frame)
        self.h_layout.addWidget(self.plotter, 4)  # 占4/5宽度

        # 创建控制面板
        self.create_control_panel()
        self.h_layout.addWidget(self.control_panel, 1)  # 占1/5宽度

        self.frame.setLayout(self.h_layout)
        self.setCentralWidget(self.frame)

        # 初始化数据
        # 如果没有提供数据，生成示例数据 (100,000点)
        if ion_data is None:
            print("生成示例 Ion 数据集 (100,000点)...")
            np.random.seed(42)
            x = np.random.normal(0, 50, 100000)
            y = np.random.normal(0, 50, 100000)
            z = np.random.normal(0, 50, 100000)
            ion_data = np.column_stack((x, y, z))
        if electron_data is None:
            print("生成示例 Electron 数据集 (100,000点)...")
            np.random.seed(42)
            # 生成与 Ion 数据相关的 2D 数据，例如能量和角度
            energy = np.random.normal(100, 10, 100000)  # 能量
            angle = np.random.uniform(0, 2*np.pi, 100000)  # 角度
            electron_data = np.column_stack((energy, angle))

        self.ion_data = ion_data
        self.electron_data = electron_data
        self.original_ion_data = ion_data.copy()
        self.original_electron_data = electron_data.copy()
        
        # 索引掩码 - 两个数据集共享同一个掩码
        self.shared_indices_mask = np.ones(len(ion_data), dtype=bool)

        # 滤波相关
        self.filtered_out_indices_history = []  # 被过滤点的历史记录
        self.filtered_out_ion_data_history = []   # 被过滤点的 Ion 坐标历史记录
        self.filtered_out_electron_data_history = [] # 被过滤点的 Electron 坐标历史记录
        self.filter_type = "box"  # 当前滤波器类型
        self.show_filtered_out = True  # 是否显示被过滤的点
        self.show_filter_volume = False  # 是否显示滤波器

        # 当前滤波器actor
        self.current_filter_actor = None
        self.filtered_out_actors = []  # 所有被过滤点的actors
        self.history_filter_actor = None  # 历史滤波器actor
        self.history_filtered_out_actor = None  # 历史滤波点actor

        # 当前显示的actor
        self.current_ion_actor = None
        self.current_electron_actor = None # 2D视图用

        # 网格和刻度线
        self.grid_actors = []
        self.range_text_actor = None  # 用于显示有效数据范围的文本
        self.range_bg_actor = None    # 用于显示背景矩形
        self.scale_text_actors = []   # 用于存储刻度标签的actors

        # 历史记录
        self.filter_history = []  # 滤波历史记录
        self.selected_history_index = None  # 当前选中的历史记录索引

        # 视图模式
        self.current_view_mode = "ion" # "ion" or "electron"

        # 设置初始视图
        self.setup_scene()
        
        # 添加键盘事件过滤器以捕获空格键
        self.installEventFilter(self)

        # 创建悬浮滤波器工具栏
        self.create_filter_toolbar()

        # 创建参数设置面板 (dock widget)
        self.create_parameter_dock()

        # 为2D视图设置相机控制器
        self.setup_camera_controller()

    def eventFilter(self, obj, event):
        """处理键盘事件"""
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Space:
                self.reset_view()
                return True  # 消费事件，防止进一步传播
        return super().eventFilter(obj, event)

    def setup_camera_controller(self):
        """为2D视图设置相机控制器，限制Z轴旋转"""
        # 这里我们通过监听鼠标事件来限制相机行为
        # PyVista 没有直接的 API 来限制相机旋转，所以我们重写 mouseMoveEvent
        # 但这需要更复杂的交互管理
        # 一个简单的替代方法是，在切换到2D视图时，手动设置相机位置并禁用某些交互
        pass

    def create_filter_toolbar(self):
        """创建悬浮在3D视图右上角的滤波器选择工具栏"""
        # 创建工具栏
        self.filter_toolbar = QtWidgets.QToolBar("滤波器")
        # 设置样式，使其看起来像一个悬浮按钮组
        self.filter_toolbar.setStyleSheet("""
            QToolBar {
                background-color: rgba(255, 255, 255, 0.8); /* 半透明背景 */
                border: 1px solid gray; /* 边框 */
                border-radius: 4px; /* 圆角 */
                padding: 2px; /* 内边距 */
            }
            QToolButton {
                background-color: rgba(255, 255, 255, 0.9); /* 按钮背景 */
                border: 1px solid lightgray; /* 按钮边框 */
                border-radius: 3px; /* 按钮圆角 */
                padding: 2px; /* 按钮内边距 */
            }
            QToolButton:checked {
                background-color: lightblue; /* 选中时的背景 */
            }
        """)
        # 设置为浮动工具栏
        self.filter_toolbar.setMovable(False)  # 固定位置
        self.filter_toolbar.setFloatable(False) # 不可拖动
        self.filter_toolbar.setOrientation(QtCore.Qt.Horizontal)  # 水平布局

        # 创建 Ion/Electron 通道按钮
        self.ion_channel_action = self.filter_toolbar.addAction("Ion")
        self.ion_channel_action.setCheckable(True)
        self.ion_channel_action.setChecked(True)  # 默认选中 Ion
        self.ion_channel_action.triggered.connect(lambda: self.set_view_mode("ion"))

        self.electron_channel_action = self.filter_toolbar.addAction("Electron")
        self.electron_channel_action.setCheckable(True)
        self.electron_channel_action.triggered.connect(lambda: self.set_view_mode("electron"))

        # 创建动作（按钮）
        self.box_action = self.filter_toolbar.addAction("立方体")
        self.box_action.setCheckable(True)
        self.box_action.setChecked(True)  # 默认选中立方体
        self.box_action.triggered.connect(lambda: self.set_filter_type("box"))

        self.sphere_action = self.filter_toolbar.addAction("球形")
        self.sphere_action.setCheckable(True)
        self.sphere_action.triggered.connect(lambda: self.set_filter_type("sphere"))

        self.cylinder_action = self.filter_toolbar.addAction("圆柱形")
        self.cylinder_action.setCheckable(True)
        self.cylinder_action.triggered.connect(lambda: self.set_filter_type("cylinder"))

        self.torus_action = self.filter_toolbar.addAction("圆环筒型") # Renamed from 'tube' to 'torus'
        self.torus_action.setCheckable(True)
        self.torus_action.triggered.connect(lambda: self.set_filter_type("torus")) # Renamed from 'tube' to 'torus'

        # 为2D视图添加滤波器按钮
        self.rect_2d_action = self.filter_toolbar.addAction("矩形 (2D)")
        self.rect_2d_action.setCheckable(True)
        self.rect_2d_action.triggered.connect(lambda: self.set_filter_type("rect_2d"))

        self.circle_2d_action = self.filter_toolbar.addAction("圆形 (2D)")
        self.circle_2d_action.setCheckable(True)
        self.circle_2d_action.triggered.connect(lambda: self.set_filter_type("circle_2d"))

        self.ring_2d_action = self.filter_toolbar.addAction("环形 (2D)")
        self.ring_2d_action.setCheckable(True)
        self.ring_2d_action.triggered.connect(lambda: self.set_filter_type("ring_2d"))

        # 将工具栏添加到主窗口的顶部
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.filter_toolbar)
        # 手动设置工具栏位置到右上角
        self.filter_toolbar.move(self.width() - self.filter_toolbar.width() - 10, 10)
        # 重写resizeEvent以动态调整工具栏位置
        self.old_size = self.size()
        self.resizeEvent = self._resize_toolbar

    def _resize_toolbar(self, event):
        """重写resizeEvent以动态调整工具栏位置"""
        super().resizeEvent(event)
        # 检查窗口大小是否改变
        if self.size() != self.old_size:
            self.old_size = self.size()
            # 重新设置工具栏位置到右上角
            self.filter_toolbar.move(self.width() - self.filter_toolbar.width() - 10, 10)

    def create_parameter_dock(self):
        """创建参数设置面板 (dock widget)"""
        # 创建DockWidget
        self.param_dock = QtWidgets.QDockWidget("滤波器参数", self)
        self.param_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        # 确保DockWidget具有关闭按钮
        self.param_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable | QtWidgets.QDockWidget.DockWidgetClosable)

        # 设置DockWidget的最小和最大尺寸
        self.param_dock.setMinimumSize(250, 300)
        self.param_dock.setMaximumSize(400, 600)

        # 创建参数设置的中央小部件
        param_widget = QtWidgets.QWidget()
        param_layout = QtWidgets.QFormLayout()

        # 位置参数 (对于3D滤波器)
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1000, 1000)
        self.x_spin.setValue(0)
        self.x_spin.setSingleStep(1)

        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1000, 1000)
        self.y_spin.setValue(0)
        self.y_spin.setSingleStep(1)

        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setRange(-1000, 1000)
        self.z_spin.setValue(0)
        self.z_spin.setSingleStep(1)

        # 3D滤波器参数
        self.x_size_spin = QtWidgets.QDoubleSpinBox()
        self.x_size_spin.setRange(0.1, 1000)
        self.x_size_spin.setValue(10)
        self.x_size_spin.setSingleStep(0.5)

        self.y_size_spin = QtWidgets.QDoubleSpinBox()
        self.y_size_spin.setRange(0.1, 1000)
        self.y_size_spin.setValue(10)
        self.y_size_spin.setSingleStep(0.5)

        self.z_size_spin = QtWidgets.QDoubleSpinBox()
        self.z_size_spin.setRange(0.1, 1000)
        self.z_size_spin.setValue(10)
        self.z_size_spin.setSingleStep(0.5)

        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 500)
        self.radius_spin.setValue(10)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setVisible(False)  # 初始隐藏，球形时显示

        self.cylinder_radius_spin = QtWidgets.QDoubleSpinBox()
        self.cylinder_radius_spin.setRange(0.1, 500)
        self.cylinder_radius_spin.setValue(10)
        self.cylinder_radius_spin.setSingleStep(0.5)
        self.cylinder_radius_spin.setVisible(False)  # 初始隐藏，圆柱形时显示
        self.cylinder_height_spin = QtWidgets.QDoubleSpinBox()
        self.cylinder_height_spin.setRange(0.1, 1000)
        self.cylinder_height_spin.setValue(20)
        self.cylinder_height_spin.setSingleStep(0.5)
        self.cylinder_height_spin.setVisible(False)  # 初始隐藏，圆柱形时显示

        self.torus_inner_radius_spin = QtWidgets.QDoubleSpinBox()
        self.torus_inner_radius_spin.setRange(0.1, 500)
        self.torus_inner_radius_spin.setValue(5)
        self.torus_inner_radius_spin.setSingleStep(0.5)
        self.torus_inner_radius_spin.setVisible(False)  # 初始隐藏，圆环筒型时显示
        self.torus_outer_radius_spin = QtWidgets.QDoubleSpinBox()
        self.torus_outer_radius_spin.setRange(0.1, 500)
        self.torus_outer_radius_spin.setValue(10)
        self.torus_outer_radius_spin.setSingleStep(0.5)
        self.torus_outer_radius_spin.setVisible(False)  # 初始隐藏，圆环筒型时显示
        self.torus_height_spin = QtWidgets.QDoubleSpinBox()
        self.torus_height_spin.setRange(0.1, 1000)
        self.torus_height_spin.setValue(20)
        self.torus_height_spin.setSingleStep(0.5)
        self.torus_height_spin.setVisible(False)  # 初始隐藏，圆环筒型时显示

        # 2D滤波器参数
        self.electron_x_spin = QtWidgets.QDoubleSpinBox()
        self.electron_x_spin.setRange(-1000, 1000)
        self.electron_x_spin.setValue(0)
        self.electron_x_spin.setSingleStep(1)
        self.electron_x_spin.setVisible(False)  # 初始隐藏，2D滤波器时显示

        self.electron_y_spin = QtWidgets.QDoubleSpinBox()
        self.electron_y_spin.setRange(-1000, 1000)
        self.electron_y_spin.setValue(0)
        self.electron_y_spin.setSingleStep(1)
        self.electron_y_spin.setVisible(False)  # 初始隐藏，2D滤波器时显示

        self.electron_width_spin = QtWidgets.QDoubleSpinBox()
        self.electron_width_spin.setRange(0.1, 1000)
        self.electron_width_spin.setValue(10)
        self.electron_width_spin.setSingleStep(0.5)
        self.electron_width_spin.setVisible(False)  # 初始隐藏，矩形2D时显示

        self.electron_height_spin = QtWidgets.QDoubleSpinBox()
        self.electron_height_spin.setRange(0.1, 1000)
        self.electron_height_spin.setValue(10)
        self.electron_height_spin.setSingleStep(0.5)
        self.electron_height_spin.setVisible(False)  # 初始隐藏，矩形2D时显示

        self.electron_radius_spin = QtWidgets.QDoubleSpinBox()
        self.electron_radius_spin.setRange(0.1, 500)
        self.electron_radius_spin.setValue(10)
        self.electron_radius_spin.setSingleStep(0.5)
        self.electron_radius_spin.setVisible(False)  # 初始隐藏，圆形2D时显示

        self.electron_ring_inner_radius_spin = QtWidgets.QDoubleSpinBox()
        self.electron_ring_inner_radius_spin.setRange(0.1, 500)
        self.electron_ring_inner_radius_spin.setValue(5)
        self.electron_ring_inner_radius_spin.setSingleStep(0.5)
        self.electron_ring_inner_radius_spin.setVisible(False)  # 初始隐藏，环形2D时显示
        self.electron_ring_outer_radius_spin = QtWidgets.QDoubleSpinBox()
        self.electron_ring_outer_radius_spin.setRange(0.1, 500)
        self.electron_ring_outer_radius_spin.setValue(10)
        self.electron_ring_outer_radius_spin.setSingleStep(0.5)
        self.electron_ring_outer_radius_spin.setVisible(False)  # 初始隐藏，环形2D时显示

        param_layout.addRow("X位置 (3D):", self.x_spin)
        param_layout.addRow("Y位置 (3D):", self.y_spin)
        param_layout.addRow("Z位置 (3D):", self.z_spin)
        param_layout.addRow("X长度 (3D):", self.x_size_spin)
        param_layout.addRow("Y宽度 (3D):", self.y_size_spin)
        param_layout.addRow("Z高度 (3D):", self.z_size_spin)
        param_layout.addRow("半径 (3D):", self.radius_spin)
        param_layout.addRow("圆柱半径 (3D):", self.cylinder_radius_spin)
        param_layout.addRow("圆柱高度 (3D):", self.cylinder_height_spin)
        param_layout.addRow("圆环内径 (3D):", self.torus_inner_radius_spin)
        param_layout.addRow("圆环外径 (3D):", self.torus_outer_radius_spin)
        param_layout.addRow("圆环高度 (3D):", self.torus_height_spin)
        param_layout.addRow("X位置 (2D):", self.electron_x_spin)
        param_layout.addRow("Y位置 (2D):", self.electron_y_spin)
        param_layout.addRow("宽度 (2D):", self.electron_width_spin)
        param_layout.addRow("高度 (2D):", self.electron_height_spin)
        param_layout.addRow("半径 (2D):", self.electron_radius_spin)
        param_layout.addRow("环形内径 (2D):", self.electron_ring_inner_radius_spin)
        param_layout.addRow("环形外径 (2D):", self.electron_ring_outer_radius_spin)

        param_widget.setLayout(param_layout)

        # 设置DockWidget的中央小部件
        self.param_dock.setWidget(param_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.param_dock)

        # 隐藏DockWidget，因为初始时不需要显示
        self.param_dock.hide()

    def create_control_panel(self):
        """创建右侧控制面板"""
        self.control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout()

        # 滤波器控制
        filter_control_group = QtWidgets.QGroupBox("滤波器控制")
        filter_control_layout = QtWidgets.QVBoxLayout()

        self.show_filter_checkbox = QtWidgets.QCheckBox("显示滤波器")
        self.show_filter_checkbox.setChecked(False)
        self.show_filter_checkbox.stateChanged.connect(self.toggle_filter_display)

        self.create_filter_btn = QtWidgets.QPushButton("创建滤波器")
        self.create_filter_btn.clicked.connect(self.create_filter)

        self.apply_filter_btn = QtWidgets.QPushButton("应用滤波")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.apply_filter_btn.setEnabled(False)  # 初始禁用，需要先创建滤波器

        filter_control_layout.addWidget(self.show_filter_checkbox)
        filter_control_layout.addWidget(self.create_filter_btn)
        filter_control_layout.addWidget(self.apply_filter_btn)
        filter_control_group.setLayout(filter_control_layout)

        # 显示选项
        display_group = QtWidgets.QGroupBox("显示选项")
        display_layout = QtWidgets.QVBoxLayout()

        self.show_filtered_out_checkbox = QtWidgets.QCheckBox("显示被过滤的数据点")
        self.show_filtered_out_checkbox.setChecked(True)
        self.show_filtered_out_checkbox.stateChanged.connect(self.toggle_filtered_out_points)

        self.reset_view_btn = QtWidgets.QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)

        self.reset_all_btn = QtWidgets.QPushButton("重置所有")
        self.reset_all_btn.clicked.connect(self.reset_all)

        display_layout.addWidget(self.show_filtered_out_checkbox)
        display_layout.addWidget(self.reset_view_btn)
        display_layout.addWidget(self.reset_all_btn)
        display_group.setLayout(display_layout)

        # 统计信息
        stats_group = QtWidgets.QGroupBox("数据统计")
        stats_layout = QtWidgets.QVBoxLayout()
        self.stats_label = QtWidgets.QLabel("总点数: 0\n剩余点数: 0\n被过滤点数: 0")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)

        # 历史记录
        history_group = QtWidgets.QGroupBox("滤波历史")
        history_layout = QtWidgets.QVBoxLayout()

        self.history_list = QtWidgets.QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)

        self.undo_last_btn = QtWidgets.QPushButton("撤销选中的滤波")
        self.undo_last_btn.clicked.connect(self.undo_selected_filter) # Changed functionality

        history_layout.addWidget(self.history_list)
        history_layout.addWidget(self.undo_last_btn)
        history_group.setLayout(history_layout)

        # 添加到控制面板
        control_layout.addWidget(filter_control_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(stats_group)  # 添加统计信息组
        control_layout.addWidget(history_group)
        control_layout.addStretch()

        self.control_panel.setLayout(control_layout)
        self.control_panel.setMaximumWidth(300)

    def set_view_mode(self, mode):
        """设置当前视图模式 (Ion 或 Electron)"""
        if mode == self.current_view_mode:
            # 如果点击的是当前已选中的按钮，不执行任何操作
            return

        self.current_view_mode = mode
        
        # 重置所有按钮的选中状态
        self.ion_channel_action.setChecked(False)
        self.electron_channel_action.setChecked(False)
        
        # 选中当前按钮
        if mode == "ion":
            self.ion_channel_action.setChecked(True)
            # 重新设置为3D视图
            self.plotter.set_background('white')
            # 移除2D点云（如果存在）
            if self.current_electron_actor:
                self.plotter.remove_actor(self.current_electron_actor)
                self.current_electron_actor = None
            # 移除2D坐标轴和网格
            for actor in self.grid_actors:
                self.plotter.remove_actor(actor)
            self.grid_actors = []
            if self.range_text_actor:
                self.plotter.remove_actor(self.range_text_actor)
                self.range_bg_actor = None
            if self.range_bg_actor:
                self.plotter.remove_actor(self.range_bg_actor)
                self.range_bg_actor = None
            # 添加3D点云
            current_ion_points = self.original_ion_data[self.shared_indices_mask]
            points_polydata = pv.PolyData(current_ion_points)
            self.current_ion_actor = self.plotter.add_mesh(
                points_polydata,
                render_points_as_spheres=False,
                point_size=2.5,
                color='#1f77b4',  # 科研风格深蓝色
                opacity=0.7,
                name='current_ion_points'
            )
            # 重新添加坐标轴和网格
            self.plotter.add_axes(line_width=3, color='black', labels_off=False,
                                 x_color='red', y_color='green', z_color='blue')
            self.update_grid_with_scales()
            # 重置相机
            self.plotter.camera_position = 'iso'
            self.plotter.reset_camera()
            # 启用默认的相机交互
            # self.plotter.update_axes() # Remove this line
            # 恢复默认的交互样式
            try:
                from vtkmodules.vtkRendering.UI import vtkInteractorStyleTrackballCamera
                self.plotter.ren_win.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            except ImportError:
                # 如果 vtkmodules.vtkRendering.UI 不可用，尝试其他可能的路径或跳过
                try:
                    from vtkmodules.vtkInteraction.Style import vtkInteractorStyleTrackballCamera
                    self.plotter.ren_win.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
                except ImportError:
                    # 如果都失败了，保持当前的交互样式
                    print("警告: 无法导入 vtkInteractorStyleTrackballCamera，将使用默认交互样式。")
                    pass
            self.plotter.ren_win.interactor.Enable()
            
        elif mode == "electron":
            self.electron_channel_action.setChecked(True)
            # 重新设置为2D视图
            self.plotter.set_background('white')
            # 移除3D点云（如果存在）
            if self.current_ion_actor:
                self.plotter.remove_actor(self.current_ion_actor)
                self.current_ion_actor = None
            # 移除3D坐标轴和网格
            for actor in self.grid_actors:
                self.plotter.remove_actor(actor)
            self.grid_actors = []
            if self.range_text_actor:
                self.plotter.remove_actor(self.range_text_actor)
                self.range_text_actor = None
            if self.range_bg_actor:
                self.plotter.remove_actor(self.range_bg_actor)
                self.range_bg_actor = None
            # 添加2D点云
            current_electron_points = self.original_electron_data[self.shared_indices_mask]
            # 将2D点转换为3D点，Z坐标设为0
            points_3d = np.column_stack([current_electron_points, np.zeros(len(current_electron_points))])
            points_polydata = pv.PolyData(points_3d)
            self.current_electron_actor = self.plotter.add_mesh(
                points_polydata,
                render_points_as_spheres=False,
                point_size=2.5,
                color='#d62728',  # 红色
                opacity=0.7,
                name='current_electron_points'
            )
            # 设置相机为正交投影或调整视角以模拟2D
            # 这里我们简单地将相机位置调整到Z轴正上方，看Z=0的平面
            self.plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]
            # 恢复默认的交互样式 for 2D mode as well
            # The user is expected to know to primarily zoom and pan in 2D mode
            try:
                from vtkmodules.vtkRendering.UI import vtkInteractorStyleTrackballCamera
                self.plotter.ren_win.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            except ImportError:
                # 如果 vtkmodules.vtkRendering.UI 不可用，尝试其他可能的路径或跳过
                try:
                    from vtkmodules.vtkInteraction.Style import vtkInteractorStyleTrackballCamera
                    self.plotter.ren_win.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
                except ImportError:
                    # 如果都失败了，保持当前的交互样式
                    print("警告: 无法导入 vtkInteractorStyleTrackballCamera，将使用默认交互样式。")
                    pass
            self.plotter.ren_win.interactor.Enable()
            # 重置相机
            self.plotter.reset_camera()
            # 添加2D网格和标签
            self.update_2d_grid_with_scales()
            # 添加2D坐标轴
            self.plotter.add_axes(line_width=3, color='black', labels_off=False,
                                 x_color='red', y_color='green', z_color='blue')
        
        # 更新参数面板的可见性
        self.update_parameter_panel_visibility()

    def update_2d_grid_with_scales(self):
        """为2D视图添加网格和刻度尺"""
        try:
            # 获取当前电子数据的边界
            current_electron_points = self.original_electron_data[self.shared_indices_mask]
            if len(current_electron_points) == 0:
                print("警告：没有可用的电子数据点来创建网格")
                return
                
            # 确保数据是2D的
            if current_electron_points.shape[1] < 2:
                print("警告：电子数据不是2D的，无法创建网格")
                return
                
            x_min, x_max = current_electron_points[:, 0].min(), current_electron_points[:, 0].max()
            y_min, y_max = current_electron_points[:, 1].min(), current_electron_points[:, 1].max()
            
            # 如果所有点都在同一位置，添加一些边界
            if x_max == x_min:
                x_min -= 1
                x_max += 1
            if y_max == y_min:
                y_min -= 1
                y_max += 1
            
            # 创建网格线 (Z坐标为0)
            x_range = x_max - x_min
            y_range = y_max - y_min
            steps = 2

            if x_range > 0:
                x_step = x_range / steps
                for i in range(steps + 1):
                    x_pos = x_min + i * x_step
                    line = pv.Line((x_pos, y_min, 0), (x_pos, y_max, 0))
                    actor = self.plotter.add_mesh(line, color='lightgray', line_width=1, name=f'2d_grid_x_{i}')
                    self.grid_actors.append(actor)

            if y_range > 0:
                y_step = y_range / steps
                for i in range(steps + 1):
                    y_pos = y_min + i * y_step
                    line = pv.Line((x_min, y_pos, 0), (x_max, y_pos, 0))
                    actor = self.plotter.add_mesh(line, color='lightgray', line_width=1, name=f'2d_grid_y_{i}')
                    self.grid_actors.append(actor)

            # 添加有效数据范围文本 (右下角)
            text_content = f"X: {x_min:.1f} ~ {x_max:.1f}\nY: {y_min:.1f} ~ {y_max:.1f}"
            self.range_text_actor = self.plotter.add_text(
                text_content,
                position=(0.7, 0.05),  # 右下角
                color='black',
                font_size=10,
                name='effective_range_2d'
            )
            # 添加半透明背景矩形
            rect_points = pv.PolyData(np.array([
                [0.68, 0.04, 0], [0.98, 0.04, 0],
                [0.98, 0.15, 0], [0.68, 0.15, 0]
            ]))
            rect_points.faces = np.array([4, 0, 1, 2, 3])
            self.range_bg_actor = self.plotter.add_mesh(
                rect_points,
                color='white',
                opacity=0.6,
                name='range_background_2d'
            )
            # 为了确保背景在文本后面，需要重新添加文本
            self.range_text_actor = self.plotter.add_text(
                text_content,
                position=(0.7, 0.05),
                color='black',
                font_size=10,
                name='effective_range_2d'
            )
        except Exception as e:
            print(f"创建2D网格时发生错误: {str(e)}")

    def set_filter_type(self, filter_type):
        """设置当前滤波器类型"""
        # 如果点击的是当前已选中的按钮，则取消选中并隐藏参数窗口
        if self.filter_type == filter_type:
            # 重置所有按钮的选中状态
            self.box_action.setChecked(False)
            self.sphere_action.setChecked(False)
            self.cylinder_action.setChecked(False)
            self.torus_action.setChecked(False)
            self.rect_2d_action.setChecked(False)
            self.circle_2d_action.setChecked(False)
            self.ring_2d_action.setChecked(False)
            # 更新 filter_type 为 None 或保持不变，这里设为 None 以表示未选中
            self.filter_type = None
            # 隐藏参数设置面板
            self.param_dock.hide()
            return

        self.filter_type = filter_type
        
        # 重置所有按钮的选中状态
        self.box_action.setChecked(False)
        self.sphere_action.setChecked(False)
        self.cylinder_action.setChecked(False)
        self.torus_action.setChecked(False)
        self.rect_2d_action.setChecked(False)
        self.circle_2d_action.setChecked(False)
        self.ring_2d_action.setChecked(False)
        
        # 选中当前按钮
        if filter_type == "box":
            self.box_action.setChecked(True)
        elif filter_type == "sphere":
            self.sphere_action.setChecked(True)
        elif filter_type == "cylinder":
            self.cylinder_action.setChecked(True)
        elif filter_type == "torus": # Renamed from tube
            self.torus_action.setChecked(True)
        elif filter_type == "rect_2d":
            self.rect_2d_action.setChecked(True)
        elif filter_type == "circle_2d":
            self.circle_2d_action.setChecked(True)
        elif filter_type == "ring_2d":
            self.ring_2d_action.setChecked(True)
        
        # 显示/隐藏相应的参数控件
        # 3D滤波器参数
        self.x_spin.setVisible(filter_type in ["box", "sphere", "cylinder", "torus"])
        self.y_spin.setVisible(filter_type in ["box", "sphere", "cylinder", "torus"])
        self.z_spin.setVisible(filter_type in ["box", "sphere", "cylinder", "torus"])
        self.x_size_spin.setVisible(filter_type == "box")
        self.y_size_spin.setVisible(filter_type == "box")
        self.z_size_spin.setVisible(filter_type == "box")
        self.radius_spin.setVisible(filter_type == "sphere")
        self.cylinder_radius_spin.setVisible(filter_type == "cylinder")
        self.cylinder_height_spin.setVisible(filter_type == "cylinder")
        self.torus_inner_radius_spin.setVisible(filter_type == "torus") # Renamed
        self.torus_outer_radius_spin.setVisible(filter_type == "torus") # Renamed
        self.torus_height_spin.setVisible(filter_type == "torus") # Renamed

        # 2D滤波器参数
        self.electron_x_spin.setVisible(filter_type in ["rect_2d", "circle_2d", "ring_2d"])
        self.electron_y_spin.setVisible(filter_type in ["rect_2d", "circle_2d", "ring_2d"])
        self.electron_width_spin.setVisible(filter_type == "rect_2d")
        self.electron_height_spin.setVisible(filter_type == "rect_2d")
        self.electron_radius_spin.setVisible(filter_type == "circle_2d")
        self.electron_ring_inner_radius_spin.setVisible(filter_type == "ring_2d")
        self.electron_ring_outer_radius_spin.setVisible(filter_type == "ring_2d")

        # 显示参数设置面板
        self.param_dock.show()

    def update_parameter_panel_visibility(self):
        """根据当前视图模式更新参数面板的可见性"""
        if self.current_view_mode == "ion":
            # 3D视图模式：显示3D滤波器参数，隐藏2D滤波器参数
            if self.filter_type in ["box", "sphere", "cylinder", "torus"]:
                # 显示3D参数
                self.x_spin.setVisible(True)
                self.y_spin.setVisible(True)
                self.z_spin.setVisible(True)
                # 隐藏2D参数
                self.electron_x_spin.setVisible(False)
                self.electron_y_spin.setVisible(False)
                self.electron_width_spin.setVisible(False)
                self.electron_height_spin.setVisible(False)
                self.electron_radius_spin.setVisible(False)
                self.electron_ring_inner_radius_spin.setVisible(False)
                self.electron_ring_outer_radius_spin.setVisible(False)
            elif self.filter_type in ["rect_2d", "circle_2d", "ring_2d"]:
                # 如果当前选中的是2D滤波器，取消选中
                self.filter_type = None
                self.param_dock.hide()
        elif self.current_view_mode == "electron":
            # 2D视图模式：显示2D滤波器参数，隐藏3D滤波器参数
            if self.filter_type in ["rect_2d", "circle_2d", "ring_2d"]:
                # 显示2D参数
                self.electron_x_spin.setVisible(True)
                self.electron_y_spin.setVisible(True)
                self.electron_width_spin.setVisible(True)
                self.electron_height_spin.setVisible(True)
                self.electron_radius_spin.setVisible(True)
                self.electron_ring_inner_radius_spin.setVisible(True)
                self.electron_ring_outer_radius_spin.setVisible(True)
                # 隐藏3D参数
                self.x_spin.setVisible(False)
                self.y_spin.setVisible(False)
                self.z_spin.setVisible(False)
                self.x_size_spin.setVisible(False)
                self.y_size_spin.setVisible(False)
                self.z_size_spin.setVisible(False)
                self.radius_spin.setVisible(False)
                self.cylinder_radius_spin.setVisible(False)
                self.cylinder_height_spin.setVisible(False)
                self.torus_inner_radius_spin.setVisible(False)
                self.torus_outer_radius_spin.setVisible(False)
                self.torus_height_spin.setVisible(False)
            elif self.filter_type in ["box", "sphere", "cylinder", "torus"]:
                # 如果当前选中的是3D滤波器，取消选中
                self.filter_type = None
                self.param_dock.hide()

    def setup_scene(self):
        """设置初始场景"""
        # 保存原始数据和索引掩码
        self.original_ion_data = self.ion_data.copy()
        self.original_electron_data = self.electron_data.copy()
        self.shared_indices_mask = np.ones(len(self.ion_data), dtype=bool)

        # 创建点云 (Ion)
        points_polydata = pv.PolyData(self.original_ion_data)

        # 添加点云到场景 (使用点精灵提高性能)
        self.current_ion_actor = self.plotter.add_mesh(
            points_polydata,
            render_points_as_spheres=False,
            point_size=2.5,
            color='#1f77b4',  # 科研风格深蓝色
            opacity=0.7,
            name='current_ion_points'
        )

        # 添加坐标轴和刻度
        self.plotter.add_axes(line_width=3, color='black', labels_off=False,
                             x_color='red', y_color='green', z_color='blue')

        # 添加网格和刻度尺
        self.update_grid_with_scales()

        # 设置初始相机位置
        self.plotter.camera_position = 'iso'
        self.plotter.add_text("科研级3D散点图可视化系统", font_size=12)

        # 更新统计信息
        self.update_stats()

    def update_grid_with_scales(self):
        """更新网格和刻度尺"""
        # 移除旧的网格和刻度线
        for actor in self.grid_actors:
            self.plotter.remove_actor(actor)
        self.grid_actors = []

        # 移除旧的有效数据范围文本和背景
        if self.range_text_actor is not None:
            self.plotter.remove_actor(self.range_text_actor)
            self.range_text_actor = None
        if self.range_bg_actor is not None:
            self.plotter.remove_actor(self.range_bg_actor)
            self.range_bg_actor = None

        # 移除旧的刻度标签
        for actor in self.scale_text_actors:
            self.plotter.remove_actor(actor)
        self.scale_text_actors = []

        # 获取当前点云的边界 - 只基于当前显示的点
        current_ion_points = self.original_ion_data[self.shared_indices_mask]
        points_polydata = pv.PolyData(current_ion_points)
        bounds = points_polydata.bounds

        # 创建网格线
        self.create_grid_lines(bounds)

        # 在右下角添加有效数据范围文本 (基于当前显示的点)
        self.add_effective_range_text(bounds)

    def create_grid_lines(self, bounds):
        """创建稀疏的淡灰色网格线，不添加任何刻度标签"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        # 计算网格间距 - 每个轴只显示3个网格线
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # 每个轴显示3个网格线
        x_steps = 2  # 3个网格意味着2个步骤
        y_steps = 2
        z_steps = 2

        # X轴方向网格线（平行于Y-Z平面）
        if x_range > 0:
            x_step = x_range / x_steps
            for i in range(x_steps + 1):
                x_pos = x_min + i * x_step
                # 创建Y-Z平面内的网格线
                line_y = pv.Line((x_pos, y_min, z_min), (x_pos, y_max, z_min))
                line_z = pv.Line((x_pos, y_min, z_min), (x_pos, y_min, z_max))

                actor_y = self.plotter.add_mesh(line_y, color='lightgray', line_width=1, name=f'grid_x_{i}_y')
                actor_z = self.plotter.add_mesh(line_z, color='lightgray', line_width=1, name=f'grid_x_{i}_z')

                self.grid_actors.extend([actor_y, actor_z])

        # Y轴方向网格线（平行于X-Z平面）
        if y_range > 0:
            y_step = y_range / y_steps
            for i in range(y_steps + 1):
                y_pos = y_min + i * y_step
                # 创建X-Z平面内的网格线
                line_x = pv.Line((x_min, y_pos, z_min), (x_max, y_pos, z_min))
                line_z = pv.Line((x_min, y_pos, z_min), (x_min, y_pos, z_max))

                actor_x = self.plotter.add_mesh(line_x, color='lightgray', line_width=1, name=f'grid_y_{i}_x')
                actor_z = self.plotter.add_mesh(line_z, color='lightgray', line_width=1, name=f'grid_y_{i}_z')

                self.grid_actors.extend([actor_x, actor_z])

        # Z轴方向网格线（平行于X-Y平面）
        if z_range > 0:
            z_step = z_range / z_steps
            for i in range(z_steps + 1):
                z_pos = z_min + i * z_step
                # 创建X-Y平面内的网格线
                line_x = pv.Line((x_min, y_min, z_pos), (x_max, y_min, z_pos))
                line_y = pv.Line((x_min, y_min, z_pos), (x_min, y_max, z_pos))

                actor_x = self.plotter.add_mesh(line_x, color='lightgray', line_width=1, name=f'grid_z_{i}_x')
                actor_y = self.plotter.add_mesh(line_y, color='lightgray', line_width=1, name=f'grid_z_{i}_y')

                self.grid_actors.extend([actor_x, actor_y])

    def add_effective_range_text(self, bounds):
        """在3D场景的右下角（即立方体网格的右下角）添加有效数据范围文本"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        # 文本内容
        text_content = f"X: {x_min:.1f} ~ {x_max:.1f}\nY: {y_min:.1f} ~ {y_max:.1f}\nZ: {z_min:.1f} ~ {z_max:.1f}"

        # 移除旧的文本和背景
        if self.range_text_actor:
            self.plotter.remove_actor(self.range_text_actor)
        if self.range_bg_actor:
            self.plotter.remove_actor(self.range_bg_actor)

        # 添加文本到右下角 (使用相对屏幕坐标 0.7, 0.05)
        self.range_text_actor = self.plotter.add_text(
            text_content,
            position=(0.7, 0.05),  # 右下角
            color='black',
            font_size=10,
            name='effective_range'
        )

        # 添加半透明背景矩形
        # 估算文本大小以调整背景矩形大小 (这是一个近似值)
        # PyVista的add_text不直接提供尺寸，所以使用一个固定大小的矩形
        # 位置稍微调整以适应文本
        rect_points = pv.PolyData(np.array([
            [0.68, 0.04, 0], [0.98, 0.04, 0],
            [0.98, 0.15, 0], [0.68, 0.15, 0]
        ]))
        rect_points.faces = np.array([4, 0, 1, 2, 3])

        self.range_bg_actor = self.plotter.add_mesh(
            rect_points,
            color='white',
            opacity=0.6,
            name='range_background'
        )
        # 为了确保背景在文本后面，需要重新添加文本
        self.range_text_actor = self.plotter.add_text(
            text_content,
            position=(0.7, 0.05),
            color='black',
            font_size=10,
            name='effective_range'
        )

    def toggle_filter_display(self, state):
        """切换滤波器显示"""
        if state:
            # 如果当前没有滤波器但有创建过，尝试重新创建
            if self.current_filter_actor is None and self.apply_filter_btn.isEnabled():
                self.create_filter()
        else:
            # 隐藏滤波器
            self.hide_filter()

    def hide_filter(self):
        """隐藏当前滤波器"""
        if self.current_filter_actor is not None:
            self.plotter.remove_actor(self.current_filter_actor)
            self.current_filter_actor = None
        self.apply_filter_btn.setEnabled(False)

    def hide_history_filter(self):
        """隐藏历史滤波器和相关点"""
        if self.history_filter_actor is not None:
            self.plotter.remove_actor(self.history_filter_actor)
            self.history_filter_actor = None
        if self.history_filtered_out_actor is not None:
            self.plotter.remove_actor(self.history_filtered_out_actor)
            self.history_filtered_out_actor = None

    def create_filter(self):
        """创建滤波器几何体"""
        # 隐藏当前滤波器
        self.hide_filter()

        # 创建新滤波器
        if self.current_view_mode == "ion":
            # 3D滤波器
            x, y, z = self.x_spin.value(), self.y_spin.value(), self.z_spin.value()

            if self.filter_type == "box":
                x_size = self.x_size_spin.value()
                y_size = self.y_size_spin.value()
                z_size = self.z_size_spin.value()
                filter_mesh = pv.Box(bounds=(
                    x - x_size/2, x + x_size/2,
                    y - y_size/2, y + y_size/2,
                    z - z_size/2, z + z_size/2
                ))
                color = 'darkred'
            elif self.filter_type == "sphere":
                radius = self.radius_spin.value()
                filter_mesh = pv.Sphere(radius=radius, center=(x, y, z))
                color = 'darkgreen'
            elif self.filter_type == "cylinder":
                radius = self.cylinder_radius_spin.value()
                height = self.cylinder_height_spin.value()
                # 创建圆柱体，中心在 (x, y, z)，沿 Z 轴
                filter_mesh = pv.Cylinder(center=(x, y, z), radius=radius, height=height, direction=(0, 0, 1), resolution=32)
                color = 'darkblue'
            elif self.filter_type == "torus": # 圆环筒型 - 使用 boolean_difference
                inner_radius = self.torus_inner_radius_spin.value()
                outer_radius = self.torus_outer_radius_spin.value()
                height = self.torus_height_spin.value()
                
                # 创建外圆柱
                outer_cylinder = pv.Cylinder(center=(x, y, z), radius=outer_radius, height=height, direction=(0, 0, 1), resolution=32)
                # 创建内圆柱
                inner_cylinder = pv.Cylinder(center=(x, y, z), radius=inner_radius, height=height, direction=(0, 0, 1), resolution=32)
                
                # 使用 triangulate 和 boolean_difference 创建圆环
                outer_cylinder_triangulated = outer_cylinder.triangulate()
                inner_cylinder_triangulated = inner_cylinder.triangulate()
                
                filter_mesh = outer_cylinder_triangulated.boolean_difference(inner_cylinder_triangulated)
                
                color = 'darkorange'
        else: # self.current_view_mode == "electron"
            # 2D滤波器
            x, y = self.electron_x_spin.value(), self.electron_y_spin.value()

            if self.filter_type == "rect_2d":
                width = self.electron_width_spin.value()
                height = self.electron_height_spin.value()
                # 创建一个2D矩形，Z坐标为0
                # 使用 pv.Box 但高度为0，或者使用 pv.Polygon
                # 这里我们创建一个非常薄的长方体
                filter_mesh = pv.Box(bounds=(
                    x - width/2, x + width/2,
                    y - height/2, y + height/2,
                    -0.01, 0.01  # 非常薄的高度
                ))
                color = 'darkred'
            elif self.filter_type == "circle_2d":
                radius = self.electron_radius_spin.value()
                # 创建一个2D圆形，Z坐标为0
                # 使用正确的 pv.Circle 参数
                try:
                    # 创建一个圆形并平移到正确位置
                    filter_mesh = pv.Circle(radius=radius, resolution=32)
                    # 平移到正确位置
                    filter_mesh.points[:, 0] += x
                    filter_mesh.points[:, 1] += y
                    filter_mesh.points[:, 2] = 0  # 确保Z坐标为0
                    # 将 Circle 转换为 PolyData
                    filter_mesh = filter_mesh.triangulate()
                    color = 'darkgreen'
                except Exception as e:
                    print(f"创建2D圆形滤波器时发生错误: {str(e)}")
                    return
            elif self.filter_type == "ring_2d":
                inner_radius = self.electron_ring_inner_radius_spin.value()
                outer_radius = self.electron_ring_outer_radius_spin.value()
                # 创建一个2D环形，Z坐标为0
                # 使用正确的 pv.Circle 参数创建内外两个圆，然后进行布尔差集
                
                try:
                    # 检查半径是否有效
                    if inner_radius >= outer_radius:
                        print("错误：内半径必须小于外半径")
                        return
                        
                    # 外圆
                    outer_circle = pv.Circle(radius=outer_radius, resolution=32)
                    # 平移到正确位置
                    outer_circle.points[:, 0] += x
                    outer_circle.points[:, 1] += y
                    outer_circle.points[:, 2] = 0  # 确保Z坐标为0
                    
                    # 内圆
                    inner_circle = pv.Circle(radius=inner_radius, resolution=32)
                    # 平移到正确位置
                    inner_circle.points[:, 0] += x
                    inner_circle.points[:, 1] += y
                    inner_circle.points[:, 2] = 0  # 确保Z坐标为0
                    
                    # 转换为 PolyData
                    outer_circle_triangulated = outer_circle.triangulate()
                    inner_circle_triangulated = inner_circle.triangulate()
                    
                    # 使用 boolean_difference
                    filter_mesh = outer_circle_triangulated.boolean_difference(inner_circle_triangulated)
                    color = 'darkblue'
                except Exception as e:
                    print(f"创建2D环形滤波器时发生错误: {str(e)}")
                    return

        # 添加滤波器到场景并保存actor引用
        # 为圆环筒型和环形设置更低的不透明度，以更好地显示其空心结构
        opacity = 0.2 if self.filter_type in ["torus", "ring_2d"] else 0.3
        
        self.current_filter_actor = self.plotter.add_mesh(
            filter_mesh,
            color=color,
            opacity=opacity,
            name='current_filter'
        )

        # 启用应用按钮
        self.apply_filter_btn.setEnabled(True)

        # 显示滤波器（如果用户选择了显示）
        if self.show_filter_checkbox.isChecked():
            self.show_filter_volume = True
        else:
            self.hide_filter()

    def apply_filter(self):
        """应用滤波器并隐藏被选中的点"""
        if self.current_filter_actor is None:
            return

        if self.current_view_mode == "ion":
            # 3D视图下的过滤
            # 获取当前显示的 Ion 点
            current_ion_points = self.original_ion_data[self.shared_indices_mask]
            current_ion_polydata = pv.PolyData(current_ion_points)

            # 执行空间查询 - 使用PyVista的select_enclosed_points
            # 添加 check_surface=False 以处理可能非封闭的几何体（如圆环）
            selection = current_ion_polydata.select_enclosed_points(self.current_filter_actor.mapper.dataset, tolerance=1e-3, check_surface=False)

            # 获取被选中（将被过滤掉）的点的索引（相对于当前显示的点）
            mask = selection["SelectedPoints"].astype(bool)
            filtered_out_local_indices = np.where(mask)[0]

            if len(filtered_out_local_indices) == 0:
                print("警告：滤波器内没有点，无法应用滤波")
                return

            # 将局部索引转换为原始数据索引
            original_indices = np.where(self.shared_indices_mask)[0]
            filtered_out_original_indices = original_indices[filtered_out_local_indices]

        else: # self.current_view_mode == "electron"
            # 2D视图下的过滤
            # 获取当前显示的 Electron 点
            current_electron_points = self.original_electron_data[self.shared_indices_mask]

            # 根据滤波器类型计算掩码
            x, y = self.electron_x_spin.value(), self.electron_y_spin.value()

            if self.filter_type == "rect_2d":
                width = self.electron_width_spin.value()
                height = self.electron_height_spin.value()
                x_min, x_max = x - width/2, x + width/2
                y_min, y_max = y - height/2, y + height/2
                mask = (
                    (current_electron_points[:, 0] >= x_min) & (current_electron_points[:, 0] <= x_max) &
                    (current_electron_points[:, 1] >= y_min) & (current_electron_points[:, 1] <= y_max)
                )
            elif self.filter_type == "circle_2d":
                radius = self.electron_radius_spin.value()
                distances = np.linalg.norm(current_electron_points - np.array([x, y]), axis=1)
                mask = distances <= radius
            elif self.filter_type == "ring_2d":
                inner_radius = self.electron_ring_inner_radius_spin.value()
                outer_radius = self.electron_ring_outer_radius_spin.value()
                distances = np.linalg.norm(current_electron_points - np.array([x, y]), axis=1)
                mask = (distances >= inner_radius) & (distances <= outer_radius)

            filtered_out_local_indices = np.where(mask)[0]

            if len(filtered_out_local_indices) == 0:
                print("警告：滤波器内没有点，无法应用滤波")
                return

            # 将局部索引转换为原始数据索引
            original_indices = np.where(self.shared_indices_mask)[0]
            filtered_out_original_indices = original_indices[filtered_out_local_indices]

        # 记录这次滤波操作
        filter_params = {
            'type': self.filter_type,
            'center': (
                self.x_spin.value(), self.y_spin.value(), self.z_spin.value()
            ) if self.current_view_mode == "ion" else (
                self.electron_x_spin.value(), self.electron_y_spin.value()
            ),
            'size': (
                self.x_size_spin.value(), self.y_size_spin.value(), self.z_size_spin.value()
            ) if self.filter_type == 'box' and self.current_view_mode == "ion" else (
                self.radius_spin.value(),
            ) if self.filter_type == 'sphere' and self.current_view_mode == "ion" else (
                self.cylinder_radius_spin.value(), self.cylinder_height_spin.value()
            ) if self.filter_type == 'cylinder' and self.current_view_mode == "ion" else (
                self.torus_inner_radius_spin.value(), self.torus_outer_radius_spin.value(), self.torus_height_spin.value() # Renamed
            ) if self.filter_type == 'torus' and self.current_view_mode == "ion" else (
                self.electron_width_spin.value(), self.electron_height_spin.value()
            ) if self.filter_type == 'rect_2d' and self.current_view_mode == "electron" else (
                self.electron_radius_spin.value(),
            ) if self.filter_type == 'circle_2d' and self.current_view_mode == "electron" else (
                self.electron_ring_inner_radius_spin.value(), self.electron_ring_outer_radius_spin.value()
            ),
            'filtered_count': len(filtered_out_original_indices),
            'filtered_indices': filtered_out_original_indices.copy()
        }

        self.filter_history.append(filter_params)

        # 更新共享索引掩码（在原始数据上操作）
        new_mask = self.shared_indices_mask.copy()
        new_mask[filtered_out_original_indices] = False
        self.shared_indices_mask = new_mask

        # 更新 Ion 和 Electron 数据
        filtered_out_ion_points = self.original_ion_data[filtered_out_original_indices]
        filtered_out_electron_points = self.original_electron_data[filtered_out_original_indices]

        # 更新主视图显示
        self.update_main_view()

        # 保存被过滤的点
        self.filtered_out_ion_data_history.append(filtered_out_ion_points)
        self.filtered_out_electron_data_history.append(filtered_out_electron_points)
        self.filtered_out_indices_history.append(filtered_out_original_indices)

        # 更新被过滤点的显示
        self.update_filtered_out_points()

        # 更新历史记录列表
        self.update_history_list()

        # 更新网格和刻度尺
        self.update_grid_with_scales()

        # 隐藏滤波器
        self.hide_filter()

        # 更新统计信息
        self.update_stats()

    def update_main_view(self):
        """更新主视图显示"""
        # 根据当前视图模式更新显示
        if self.current_view_mode == "ion":
            # 移除当前的 Ion 点云
            if self.current_ion_actor is not None:
                self.plotter.remove_actor(self.current_ion_actor)

            # 获取当前显示的 Ion 点
            current_ion_points = self.original_ion_data[self.shared_indices_mask]

            # 如果还有点要显示，添加新的点云
            if len(current_ion_points) > 0:
                points_polydata = pv.PolyData(current_ion_points)
                self.current_ion_actor = self.plotter.add_mesh(
                    points_polydata,
                    render_points_as_spheres=False,
                    point_size=2.5,
                    color='#1f77b4',  # 科研风格深蓝色
                    opacity=0.7,
                    name='current_ion_points'
                )
            else:
                # 如果没有点要显示，创建一个空的polydata
                empty_points = pv.PolyData(np.array([[0, 0, 0]]))
                self.current_ion_actor = self.plotter.add_mesh(
                    empty_points,
                    render_points_as_spheres=False,
                    point_size=0,
                    color='#1f77b4',
                    opacity=0,
                    name='current_ion_points'
                )

            # 更新网格和刻度尺
            self.update_grid_with_scales()
            
        elif self.current_view_mode == "electron":
            # 移除当前的 Electron 点云
            if self.current_electron_actor is not None:
                self.plotter.remove_actor(self.current_electron_actor)

            # 获取当前显示的 Electron 点
            current_electron_points = self.original_electron_data[self.shared_indices_mask]

            # 如果还有点要显示，添加新的点云 (转换为3D)
            if len(current_electron_points) > 0:
                points_3d = np.column_stack([current_electron_points, np.zeros(len(current_electron_points))])
                points_polydata = pv.PolyData(points_3d)
                self.current_electron_actor = self.plotter.add_mesh(
                    points_polydata,
                    render_points_as_spheres=False,
                    point_size=2.5,
                    color='#d62728',  # 红色
                    opacity=0.7,
                    name='current_electron_points'
                )
            else:
                # 如果没有点要显示，创建一个空的polydata
                empty_points = pv.PolyData(np.array([[0, 0, 0]]))
                self.current_electron_actor = self.plotter.add_mesh(
                    empty_points,
                    render_points_as_spheres=False,
                    point_size=0,
                    color='#d62728',
                    opacity=0,
                    name='current_electron_points'
                )
            
            # 更新2D网格和刻度尺
            self.update_2d_grid_with_scales()

    def update_filtered_out_points(self):
        """更新被过滤点的可视化"""
        # 移除所有旧的被过滤点actors
        for actor in self.filtered_out_actors:
            self.plotter.remove_actor(actor)
        self.filtered_out_actors = []

        # 如果应该显示被过滤的点
        if self.show_filtered_out:
            for i, (filtered_out_ion_points, filtered_out_electron_points) in enumerate(zip(self.filtered_out_ion_data_history, self.filtered_out_electron_data_history)):
                # 根据当前视图模式显示被过滤的点
                if self.current_view_mode == "ion":
                    if len(filtered_out_ion_points) > 0:
                        filtered_out_polydata = pv.PolyData(filtered_out_ion_points)
                        actor = self.plotter.add_mesh(
                            filtered_out_polydata,
                            render_points_as_spheres=False,
                            point_size=2,
                            color='#A9A9A9',  # 暗灰色，更清晰
                            opacity=0.6,  # 降低透明度，更清晰
                            name=f'filtered_out_points_{i}'
                        )
                        self.filtered_out_actors.append(actor)
                elif self.current_view_mode == "electron":
                    if len(filtered_out_electron_points) > 0:
                        points_3d = np.column_stack([filtered_out_electron_points, np.zeros(len(filtered_out_electron_points))])
                        filtered_out_polydata = pv.PolyData(points_3d)
                        actor = self.plotter.add_mesh(
                            filtered_out_polydata,
                            render_points_as_spheres=False,
                            point_size=2,
                            color='#A9A9A9',  # 暗灰色，更清晰
                            opacity=0.6,  # 降低透明度，更清晰
                            name=f'filtered_out_points_{i}'
                        )
                        self.filtered_out_actors.append(actor)

    def toggle_filtered_out_points(self, state):
        """切换是否显示被过滤的数据点"""
        self.show_filtered_out = bool(state)
        self.update_filtered_out_points()

    def update_stats(self):
        """更新数据统计信息"""
        total = len(self.original_ion_data) # Ion 和 Electron 数据长度相同
        remaining = len(self.original_ion_data[self.shared_indices_mask])
        filtered_total = total - remaining
        self.stats_label.setText(f"总点数: {total:,}\n剩余点数: {remaining:,} ({remaining/total:.1%})\n被过滤点数: {filtered_total:,} ({filtered_total/total:.1%})")

    def reset_view(self):
        """重置视图"""
        if self.current_view_mode == "ion":
            self.plotter.camera_position = 'iso'
            self.plotter.reset_camera()
        elif self.current_view_mode == "electron":
            # 重置到2D视图的默认视角
            self.plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]
            self.plotter.reset_camera()

    def reset_all(self):
        """重置所有滤波操作"""
        # 恢复到原始状态
        self.shared_indices_mask = np.ones(len(self.original_ion_data), dtype=bool)

        # 清空历史记录
        self.filtered_out_indices_history = []
        self.filtered_out_ion_data_history = []
        self.filtered_out_electron_data_history = []
        self.filter_history = []

        # 清空选中的历史索引
        self.selected_history_index = None

        # 更新视图
        self.update_main_view()
        self.update_filtered_out_points()
        self.hide_history_filter()  # 隐藏历史滤波器和点

        # 清空历史记录列表
        self.update_history_list()

        # 更新统计信息
        self.update_stats()

    def update_history_list(self):
        """更新历史记录列表"""
        self.history_list.clear()
        for i, filter_info in enumerate(self.filter_history):
            item_text = f"滤波 {i+1}: {filter_info['type']} | "
            if filter_info['type'] == 'box':
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"尺寸({filter_info['size'][0]:.1f}, {filter_info['size'][1]:.1f}, {filter_info['size'][2]:.1f}) | "
            elif filter_info['type'] == 'sphere':
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"半径({filter_info['size'][0]:.1f}) | "
            elif filter_info['type'] == 'cylinder':
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"半径({filter_info['size'][0]:.1f}), 高度({filter_info['size'][1]:.1f}) | "
            elif filter_info['type'] == 'torus': # Renamed from tube
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"内径({filter_info['size'][0]:.1f}), 外径({filter_info['size'][1]:.1f}), 高度({filter_info['size'][2]:.1f}) | " # Renamed
            elif filter_info['type'] == 'rect_2d':
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"宽度({filter_info['size'][0]:.1f}), 高度({filter_info['size'][1]:.1f}) | "
            elif filter_info['type'] == 'circle_2d':
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"半径({filter_info['size'][0]:.1f}) | "
            elif filter_info['type'] == 'ring_2d':
                item_text += f"位置({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"内径({filter_info['size'][0]:.1f}), 外径({filter_info['size'][1]:.1f}) | "
            item_text += f"过滤点数: {filter_info['filtered_count']}"

            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, i)  # 存储索引
            self.history_list.addItem(item)

    def on_history_item_clicked(self, item):
        """点击历史记录项"""
        index = item.data(QtCore.Qt.UserRole)
        if index is not None:
            # 如果当前选中的就是这个项，取消选中
            if self.selected_history_index == index:
                self.selected_history_index = None
                self.hide_history_filter()
                return

            # 否则选中这个项
            filter_info = self.filter_history[index]

            # 设置参数
            if self.current_view_mode == "ion":
                self.x_spin.setValue(filter_info['center'][0])
                self.y_spin.setValue(filter_info['center'][1])
                self.z_spin.setValue(filter_info['center'][2])
            else: # self.current_view_mode == "electron"
                self.electron_x_spin.setValue(filter_info['center'][0])
                self.electron_y_spin.setValue(filter_info['center'][1])

            if filter_info['type'] == 'box':
                self.box_action.trigger() # This will also show the parameter dock
                self.x_size_spin.setValue(filter_info['size'][0])
                self.y_size_spin.setValue(filter_info['size'][1])
                self.z_size_spin.setValue(filter_info['size'][2])
            elif filter_info['type'] == 'sphere':
                self.sphere_action.trigger() # This will also show the parameter dock
                self.radius_spin.setValue(filter_info['size'][0])
            elif filter_info['type'] == 'cylinder':
                self.cylinder_action.trigger() # This will also show the parameter dock
                self.cylinder_radius_spin.setValue(filter_info['size'][0])
                self.cylinder_height_spin.setValue(filter_info['size'][1])
            elif filter_info['type'] == 'torus': # Renamed from tube
                self.torus_action.trigger() # This will also show the parameter dock
                self.torus_inner_radius_spin.setValue(filter_info['size'][0]) # Renamed
                self.torus_outer_radius_spin.setValue(filter_info['size'][1]) # Renamed
                self.torus_height_spin.setValue(filter_info['size'][2]) # Renamed
            elif filter_info['type'] == 'rect_2d':
                self.rect_2d_action.trigger() # This will also show the parameter dock
                self.electron_width_spin.setValue(filter_info['size'][0])
                self.electron_height_spin.setValue(filter_info['size'][1])
            elif filter_info['type'] == 'circle_2d':
                self.circle_2d_action.trigger() # This will also show the parameter dock
                self.electron_radius_spin.setValue(filter_info['size'][0])
            elif filter_info['type'] == 'ring_2d':
                self.ring_2d_action.trigger() # This will also show the parameter dock
                self.electron_ring_inner_radius_spin.setValue(filter_info['size'][0])
                self.electron_ring_outer_radius_spin.setValue(filter_info['size'][1])

            # 隐藏之前的历史滤波器和点
            self.hide_history_filter()

            # 记录当前选中的历史索引
            self.selected_history_index = index

            # 显示选中的历史滤波器和相关点
            self.show_history_filter(index)
        else:
            # 点击了空白区域，清除选择
            self.selected_history_index = None
            self.hide_history_filter()

    def show_history_filter(self, index):
        """显示历史记录对应的滤波器和相关点"""
        # 隐藏当前滤波器
        self.hide_filter()

        # 获取历史记录信息
        filter_info = self.filter_history[index]

        # 创建历史滤波器
        if self.current_view_mode == "ion":
            x, y, z = filter_info['center']
            filter_type = filter_info['type']

            if filter_type == 'box':
                x_size, y_size, z_size = filter_info['size']
                filter_mesh = pv.Box(bounds=(
                    x - x_size/2, x + x_size/2,
                    y - y_size/2, y + y_size/2,
                    z - z_size/2, z + z_size/2
                ))
                color = 'purple'
            elif filter_type == 'sphere':
                radius = filter_info['size'][0]
                filter_mesh = pv.Sphere(radius=radius, center=(x, y, z))
                color = 'orange'
            elif filter_type == 'cylinder':
                radius = filter_info['size'][0]
                height = filter_info['size'][1]
                filter_mesh = pv.Cylinder(center=(x, y, z), radius=radius, height=height, direction=(0, 0, 1), resolution=32)
                color = 'magenta'
            elif filter_type == 'torus': # 圆环筒型 - 使用 boolean_difference
                inner_radius = filter_info['size'][0] # [0] is inner_radius
                outer_radius = filter_info['size'][1] # [1] is outer_radius
                height = filter_info['size'][2] # [2] is height
                
                # 创建外圆柱
                outer_cylinder = pv.Cylinder(center=(x, y, z), radius=outer_radius, height=height, direction=(0, 0, 1), resolution=32)
                # 创建内圆柱
                inner_cylinder = pv.Cylinder(center=(x, y, z), radius=inner_radius, height=height, direction=(0, 0, 1), resolution=32)
                
                # 使用 triangulate 和 boolean_difference 创建圆环
                outer_cylinder_triangulated = outer_cylinder.triangulate()
                inner_cylinder_triangulated = inner_cylinder.triangulate()
                
                filter_mesh = outer_cylinder_triangulated.boolean_difference(inner_cylinder_triangulated)
                
                color = 'cyan'
        else: # self.current_view_mode == "electron"
            x, y = filter_info['center']
            filter_type = filter_info['type']

            if filter_type == 'rect_2d':
                width, height = filter_info['size']
                # 创建一个2D矩形，Z坐标为0
                filter_mesh = pv.Box(bounds=(
                    x - width/2, x + width/2,
                    y - height/2, y + height/2,
                    -0.01, 0.01  # 非常薄的高度
                ))
                color = 'purple'
            elif filter_type == 'circle_2d':
                radius = filter_info['size'][0]
                # 创建一个2D圆形，Z坐标为0
                # 创建一个圆形并平移到正确位置
                filter_mesh = pv.Circle(radius=radius, resolution=32)
                # 平移到正确位置
                filter_mesh.points[:, 0] += x
                filter_mesh.points[:, 1] += y
                filter_mesh.points[:, 2] = 0  # 确保Z坐标为0
                # 将 Circle 转换为 PolyData
                filter_mesh = filter_mesh.triangulate()
                color = 'orange'
            elif filter_type == 'ring_2d':
                inner_radius, outer_radius = filter_info['size']
                # 创建一个2D环形，Z坐标为0
                # 使用正确的 pv.Circle 参数并进行布尔差集
                outer_circle = pv.Circle(radius=outer_radius, resolution=32)
                # 平移到正确位置
                outer_circle.points[:, 0] += x
                outer_circle.points[:, 1] += y
                outer_circle.points[:, 2] = 0  # 确保Z坐标为0
                
                inner_circle = pv.Circle(radius=inner_radius, resolution=32)
                # 平移到正确位置
                inner_circle.points[:, 0] += x
                inner_circle.points[:, 1] += y
                inner_circle.points[:, 2] = 0  # 确保Z坐标为0
                
                # 转换为 PolyData
                outer_circle_triangulated = outer_circle.triangulate()
                inner_circle_triangulated = inner_circle.triangulate()
                # 使用 boolean_difference
                filter_mesh = outer_circle_triangulated.boolean_difference(inner_circle_triangulated)
                color = 'magenta'

        # 添加历史滤波器到场景
        # 为圆环筒型和环形设置更低的不透明度，以更好地显示其空心结构
        opacity = 0.2 if filter_type in ["torus", "ring_2d"] else 0.4
        
        self.history_filter_actor = self.plotter.add_mesh(
            filter_mesh,
            color=color,
            opacity=opacity,
            name=f'history_filter_{index}'
        )

        # 显示历史滤波器内的点（使用科研常用颜色）
        if self.current_view_mode == "ion":
            filtered_points = self.original_ion_data[filter_info['filtered_indices']]
            if len(filtered_points) > 0:
                filtered_polydata = pv.PolyData(filtered_points)
                self.history_filtered_out_actor = self.plotter.add_mesh(
                    filtered_polydata,
                    render_points_as_spheres=False,
                    point_size=3,
                    color='#FF6B6B',  # 科研常用红色，用于历史滤波点
                    opacity=0.8,
                    name=f'history_filtered_points_{index}'
                )
        else: # self.current_view_mode == "electron"
            filtered_points = self.original_electron_data[filter_info['filtered_indices']]
            if len(filtered_points) > 0:
                points_3d = np.column_stack([filtered_points, np.zeros(len(filtered_points))])
                filtered_polydata = pv.PolyData(points_3d)
                self.history_filtered_out_actor = self.plotter.add_mesh(
                    filtered_polydata,
                    render_points_as_spheres=False,
                    point_size=3,
                    color='#FF6B6B',  # 科研常用红色，用于历史滤波点
                    opacity=0.8,
                    name=f'history_filtered_points_{index}'
                )

    def undo_selected_filter(self):
        """撤销选中的滤波"""
        if self.selected_history_index is None or len(self.filter_history) == 0:
            print("没有选中要撤销的滤波记录。")
            return

        # 获取选中项的索引
        index_to_undo = self.selected_history_index

        # 获取要撤销的滤波信息
        filter_to_undo = self.filter_history[index_to_undo]
        indices_to_restore = filter_to_undo['filtered_indices']

        # 恢复被过滤的点到当前显示
        new_mask = self.shared_indices_mask.copy()
        new_mask[indices_to_restore] = True  # 恢复这些点

        # 应用新的掩码
        self.shared_indices_mask = new_mask

        # 从历史记录中移除选中的记录
        self.filter_history.pop(index_to_undo)
        self.filtered_out_ion_data_history.pop(index_to_undo)
        self.filtered_out_electron_data_history.pop(index_to_undo)
        self.filtered_out_indices_history.pop(index_to_undo)

        # 由于移除了历史记录中的一个元素，需要更新后续元素的索引
        # 并且如果选中的索引等于当前选中的历史索引，需要重置选中状态
        self.selected_history_index = None
        self.hide_history_filter()

        # 更新视图
        self.update_main_view()
        self.update_filtered_out_points()

        # 更新历史记录列表
        self.update_history_list()

        # 更新统计信息
        self.update_stats()

    def undo_last_filter(self):
        """撤销上一次滤波 (保留此函数，但不用于按钮)"""
        if not self.filter_history:
            return

        # 获取最后一步的滤波信息
        last_filter = self.filter_history.pop()
        last_filtered_indices = last_filter['filtered_indices']

        # 恢复被过滤的点到当前显示
        new_mask = self.shared_indices_mask.copy()
        new_mask[last_filtered_indices] = True  # 恢复这些点

        # 应用新的掩码
        self.shared_indices_mask = new_mask

        # 从历史记录中移除最后的记录
        self.filtered_out_ion_data_history.pop()
        self.filtered_out_electron_data_history.pop()
        self.filtered_out_indices_history.pop()

        # 如果撤销的是当前选中的历史记录，隐藏历史滤波器
        if self.selected_history_index is not None and self.selected_history_index == len(self.filter_history):
            self.selected_history_index = None
            self.hide_history_filter()
        elif self.selected_history_index is not None and self.selected_history_index >= len(self.filter_history):
            # 如果选中的索引超出范围，隐藏历史滤波器
            self.selected_history_index = None
            self.hide_history_filter()

        # 更新视图
        self.update_main_view()
        self.update_filtered_out_points()

        # 更新历史记录列表
        self.update_history_list()

        # 更新统计信息
        self.update_stats()


def load_large_dataset(file_path=None):
    """
    加载超大量数据集的函数
    实际应用中，这里可以添加数据加载逻辑
    """
    # 示例：模拟加载大数据
    if file_path:
        print(f"正在加载数据集: {file_path}")
        # 这里添加实际的数据加载代码，例如:
        # data = np.loadtxt(file_path)
        # return data
        pass
    else:
        # 生成100,000个随机点作为示例
        return None, None

def downsample_points(points, max_points=500000):
    """如果点太多，进行随机下采样"""
    if len(points) > max_points:
        print(f"数据点过多 ({len(points):,})，进行下采样至 {max_points} 点")
        indices = np.random.choice(len(points), max_points, replace=False)
        return points[indices]
    return points

if __name__ == "__main__":
    # 创建Qt应用
    app = QtWidgets.QApplication(sys.argv)

    # 生成或加载数据 (实际使用时替换为真实数据加载)
    np.random.seed(42)
    x = np.random.normal(0, 50, 100000)
    y = np.random.normal(0, 50, 100000)
    z = np.random.normal(0, 50, 100000)
    ion_data = np.column_stack((x, y, z))
    
    energy = np.random.normal(100, 10, 100000)  # 能量
    angle = np.random.uniform(0, 2*np.pi, 100000)  # 角度
    electron_data = np.column_stack((energy, angle))

    # 下采样大数据集（如果需要）
    #ion_data = downsample_points(ion_data)
    #electron_data = downsample_points(electron_data)

    # 创建可视化窗口
    window = ThreeDVisualizer(ion_data=ion_data, electron_data=electron_data)
    window.setWindowTitle("科研级3D/2D散点图可视化系统")
    window.resize(1400, 900)  # 增加窗口大小
    window.show()

    # 启动应用
    sys.exit(app.exec_())