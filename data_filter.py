import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle, Circle, Ellipse, Wedge
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
import sys
import warnings
import pickle
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

# Suppress the SIP deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

@dataclass
class FilterOperation:
    """Data class to store filter operation information"""
    type: str
    center: Tuple[float, ...]
    size: Tuple[float, ...]
    filtered_count: int
    filtered_indices: np.ndarray
    dimension: str  # '2D' or '3D'

class SharedDataManager:
    """Manages shared data between 2D and 3D views"""
    
    def __init__(self, num_2d_points=100000, num_3d_points=100000):
        # Generate 2D dataset (electron data)
        np.random.seed(42)
        x_2d = np.random.normal(0, 50, num_2d_points)
        y_2d = np.random.normal(0, 50, num_2d_points)
        self.data_2d = np.column_stack((x_2d, y_2d))
        
        # Generate 3D dataset (ion data) - share same seed for correlation
        np.random.seed(42)
        x_3d = np.random.normal(0, 50, num_3d_points)
        y_3d = np.random.normal(0, 50, num_3d_points)
        z_3d = np.random.normal(0, 50, num_3d_points)
        self.data_3d = np.column_stack((x_3d, y_3d, z_3d))
        
        # Shared indices mask (start with all points visible)
        min_points = min(len(self.data_2d), len(self.data_3d))
        self.shared_indices_mask = np.ones(min_points, dtype=bool)
        
        # Filter history (shared between both views)
        self.filter_history: List[FilterOperation] = []
        
        # Current view states
        self.current_2d_indices_mask = self.shared_indices_mask.copy()
        self.current_3d_indices_mask = self.shared_indices_mask.copy()
        
    def apply_filter(self, filter_op: FilterOperation, dimension: str):
        """Apply filter and update shared indices"""
        self.filter_history.append(filter_op)
        
        # Update shared indices mask
        self.shared_indices_mask[filter_op.filtered_indices] = False
        
        # Update dimension-specific masks
        if dimension == '2D':
            self.current_2d_indices_mask = self.shared_indices_mask.copy()
        else:
            self.current_3d_indices_mask = self.shared_indices_mask.copy()
    
    def undo_filter(self, index: int):
        """Undo a specific filter operation"""
        if 0 <= index < len(self.filter_history):
            filter_op = self.filter_history.pop(index)
            
            # Restore filtered indices
            self.shared_indices_mask[filter_op.filtered_indices] = True
            
            # Update both masks
            self.current_2d_indices_mask = self.shared_indices_mask.copy()
            self.current_3d_indices_mask = self.shared_indices_mask.copy()
            
            return filter_op
        return None
    
    def reset_all(self):
        """Reset all filters"""
        min_points = min(len(self.data_2d), len(self.data_3d))
        self.shared_indices_mask = np.ones(min_points, dtype=bool)
        self.current_2d_indices_mask = self.shared_indices_mask.copy()
        self.current_3d_indices_mask = self.shared_indices_mask.copy()
        self.filter_history.clear()

class TwoDVisualizerWidget(QtWidgets.QWidget):
    """2D visualization widget"""
    
    def __init__(self, data_manager: SharedDataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setup_ui()
        
        # Filter related
        self.filter_type = "box"
        self.show_filtered_out = True
        self.show_filter_volume = False
        self.current_filter_patch = None
        self.filtered_out_scatter = None
        self.current_scatter = None
        
        # History related
        self.history_filter_patch = None
        self.history_filtered_scatter = None
        
        # Setup scene
        self.setup_scene()
        
    def setup_ui(self):
        """Setup UI layout"""
        layout = QtWidgets.QVBoxLayout()
        
        # Create matplotlib figure and canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def setup_scene(self):
        """Set up initial scene"""
        # Get current 2D data based on shared indices
        min_points = min(len(self.data_manager.data_2d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_2d[:min_points][self.data_manager.current_2d_indices_mask[:min_points]]
        
        # Create scatter plot
        if len(current_data) > 0:
            self.current_scatter = self.ax.scatter(
                current_data[:, 0], current_data[:, 1],
                s=6, c='#1f77b4', alpha=0.7, label='Electron Data (2D)'
            )
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        self.ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        
        # Set title
        self.ax.set_title("2D Electron View", fontsize=12)
        self.ax.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()
    
    def update_view(self):
        """Update 2D view based on shared indices"""
        # Clear current plot
        self.ax.clear()
        
        # Get current 2D data
        min_points = min(len(self.data_manager.data_2d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_2d[:min_points][self.data_manager.current_2d_indices_mask[:min_points]]
        
        # Get filtered out data
        filtered_mask = ~self.data_manager.current_2d_indices_mask[:min_points]
        filtered_data = self.data_manager.data_2d[:min_points][filtered_mask]
        
        # Plot current data
        if len(current_data) > 0:
            self.current_scatter = self.ax.scatter(
                current_data[:, 0], current_data[:, 1],
                s=6, c='#1f77b4', alpha=0.7, label='Electron Data (2D)'
            )
        
        # Plot filtered data if enabled
        if self.show_filtered_out and len(filtered_data) > 0:
            self.filtered_out_scatter = self.ax.scatter(
                filtered_data[:, 0], filtered_data[:, 1],
                s=4, c='#A9A9A9', alpha=0.6, label='Filtered Data'
            )
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        self.ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        
        # Set title
        self.ax.set_title("2D Electron View", fontsize=12)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend()
        
        self.canvas.draw()
        
    def show_history_filter(self, filter_op):
        """Display filter corresponding to history record and related points"""
        # Hide current filter
        if self.current_filter_patch:
            self.current_filter_patch.remove()
            self.current_filter_patch = None
        
        # Hide previous history filter and points
        self.hide_history_filter()
        
        # Create history filter with different color
        x, y = filter_op.center
        filter_type = filter_op.type
        
        if filter_type == 'box':
            width, height = filter_op.size
            filter_patch = Rectangle((x - width/2, y - height/2), width, height,
                                   fill=True, alpha=0.4, color='purple')
        elif filter_type == 'circle':
            radius = filter_op.size[0]
            filter_patch = Circle((x, y), radius, fill=True, alpha=0.4, color='orange')
        elif filter_type == 'ellipse':
            width, height = filter_op.size
            filter_patch = Ellipse((x, y), width, height, fill=True, alpha=0.4, color='magenta')
        elif filter_type == 'ring':
            inner_radius = filter_op.size[0]
            outer_radius = filter_op.size[1]
            filter_patch = Wedge((x, y), outer_radius, 0, 360, width=outer_radius-inner_radius,
                               fill=True, alpha=0.4, color='cyan')
        else:
            return
        
        # Add history filter to scene
        self.history_filter_patch = self.ax.add_patch(filter_patch)
        
        # Display points within history filter with special color
        filtered_points = self.data_manager.data_2d[filter_op.filtered_indices]
        if len(filtered_points) > 0:
            self.history_filtered_scatter = self.ax.scatter(
                filtered_points[:, 0], filtered_points[:, 1],
                s=8, c='#FF6B6B', alpha=0.8, label='History Filtered Points'
            )
            self.ax.legend()
        
        self.canvas.draw()
        
    def hide_history_filter(self):
        """Hide history filter and related points"""
        if self.history_filter_patch:
            try:
                self.history_filter_patch.remove()
            except (ValueError, NotImplementedError):
                pass  # Patch already removed or can't be removed
            self.history_filter_patch = None
        if self.history_filtered_scatter:
            try:
                self.history_filtered_scatter.remove()
            except (ValueError, NotImplementedError):
                pass  # Scatter already removed or can't be removed
            self.history_filtered_scatter = None
        self.canvas.draw()

class ThreeDVisualizerWidget(QtWidgets.QWidget):
    """3D visualization widget"""
    
    def __init__(self, data_manager: SharedDataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setup_ui()
        
        # Filter related
        self.show_filtered_out = True
        self.show_filter_volume = False
        self.current_points_actor = None
        self.filtered_out_actors = []
        
        # History related
        self.history_filter_actor = None
        self.history_filtered_actor = None
        
        # Grid and scale lines
        self.grid_actors = []
        self.range_text_actor = None  # Text for displaying effective data range
        
        # Setup scene
        self.setup_scene()
        
    def setup_ui(self):
        """Setup UI layout"""
        layout = QtWidgets.QVBoxLayout()
        
        # Create PyVista plotter
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.setLayout(layout)
        
    def setup_scene(self):
        """Set up initial scene"""
        # Get current 3D data based on shared indices
        min_points = min(len(self.data_manager.data_3d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_3d[:min_points][self.data_manager.current_3d_indices_mask[:min_points]]
        
        # Create point cloud
        if len(current_data) > 0:
            points_polydata = pv.PolyData(current_data)
            self.current_points_actor = self.plotter.add_mesh(
                points_polydata,
                render_points_as_spheres=False,
                point_size=2.5,
                color='#1f77b4',
                opacity=0.7,
                name='ion_points'
            )
        
        # Add coordinate axes
        self.plotter.add_axes(line_width=3, color='black', labels_off=False,
                             x_color='red', y_color='green', z_color='blue')
        
        # Set initial camera position
        self.plotter.camera_position = 'iso'
        self.plotter.add_text("3D Ion View", font_size=12)
        
        # Add grid and scale rulers
        self.update_grid_with_scales()
        
    def update_view(self):
        """Update 3D view based on shared indices"""
        # Remove existing actors
        if self.current_points_actor is not None:
            self.plotter.remove_actor(self.current_points_actor)
        
        for actor in self.filtered_out_actors:
            self.plotter.remove_actor(actor)
        self.filtered_out_actors.clear()
        
        # Get current 3D data
        min_points = min(len(self.data_manager.data_3d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_3d[:min_points][self.data_manager.current_3d_indices_mask[:min_points]]
        
        # Get filtered out data
        filtered_mask = ~self.data_manager.current_3d_indices_mask[:min_points]
        filtered_data = self.data_manager.data_3d[:min_points][filtered_mask]
        
        # Plot current data
        if len(current_data) > 0:
            points_polydata = pv.PolyData(current_data)
            self.current_points_actor = self.plotter.add_mesh(
                points_polydata,
                render_points_as_spheres=False,
                point_size=2.5,
                color='#1f77b4',
                opacity=0.7,
                name='ion_points'
            )
        
        # Plot filtered data if enabled
        if self.show_filtered_out and len(filtered_data) > 0:
            filtered_polydata = pv.PolyData(filtered_data)
            actor = self.plotter.add_mesh(
                filtered_polydata,
                render_points_as_spheres=False,
                point_size=2,
                color='#A9A9A9',
                opacity=0.6,
                name='filtered_points'
            )
            self.filtered_out_actors.append(actor)
        
        # Update grid and scale rulers
        self.update_grid_with_scales()
            
    def show_history_filter(self, filter_op):
        """Display filter corresponding to history record and related points"""
        # Hide previous history filter and points
        self.hide_history_filter()
        
        # Create history filter with different color
        x, y, z = filter_op.center
        filter_type = filter_op.type
        
        if filter_type == 'box':
            x_size, y_size, z_size = filter_op.size
            filter_mesh = pv.Box(bounds=(
                x - x_size/2, x + x_size/2,
                y - y_size/2, y + y_size/2,
                z - z_size/2, z + z_size/2
            ))
            color = 'purple'
        elif filter_type == 'sphere':
            radius = filter_op.size[0]
            filter_mesh = pv.Sphere(radius=radius, center=(x, y, z))
            color = 'orange'
        elif filter_type == 'cylinder':
            radius = filter_op.size[0]
            height = filter_op.size[1]
            filter_mesh = pv.Cylinder(center=(x, y, z), radius=radius, height=height, direction=(0, 0, 1))
            color = 'magenta'
        elif filter_type == 'torus':
            inner_radius = filter_op.size[0]
            outer_radius = filter_op.size[1]
            height = filter_op.size[2]
            filter_mesh = pv.ParametricTorus(
                ringradius=(inner_radius + outer_radius) / 2,
                crosssectionradius=(outer_radius - inner_radius) / 2
            )
            filter_mesh.translate((x, y, z))
            color = 'cyan'
        else:
            return
        
        # Add history filter to scene
        self.history_filter_actor = self.plotter.add_mesh(
            filter_mesh,
            color=color,
            opacity=0.4,
            name=f'history_filter'
        )
        
        # Display points within history filter with special color
        filtered_points = self.data_manager.data_3d[filter_op.filtered_indices]
        if len(filtered_points) > 0:
            filtered_polydata = pv.PolyData(filtered_points)
            self.history_filtered_actor = self.plotter.add_mesh(
                filtered_polydata,
                render_points_as_spheres=False,
                point_size=3,
                color='#FF6B6B',
                opacity=0.8,
                name=f'history_filtered_points'
            )
        
    def hide_history_filter(self):
        """Hide history filter and related points"""
        if self.history_filter_actor:
            self.plotter.remove_actor(self.history_filter_actor)
            self.history_filter_actor = None
        if self.history_filtered_actor:
            self.plotter.remove_actor(self.history_filtered_actor)
            self.history_filtered_actor = None

    def update_grid_with_scales(self):
        """Update grid and scale rulers"""
        # Remove old grid and scale lines
        for actor in self.grid_actors:
            self.plotter.remove_actor(actor)
        self.grid_actors = []

        # Remove old effective data range text
        if self.range_text_actor is not None:
            self.plotter.remove_actor(self.range_text_actor)
            self.range_text_actor = None

        # Get current point cloud boundaries - based only on currently displayed points
        min_points = min(len(self.data_manager.data_3d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_3d[:min_points][self.data_manager.current_3d_indices_mask[:min_points]]
        
        if len(current_data) > 0:
            points_polydata = pv.PolyData(current_data)
            bounds = points_polydata.bounds

            # Create grid lines
            self.create_grid_lines(bounds)

            # Add effective data range text in bottom-right corner (based on currently displayed points)
            self.add_effective_range_text(bounds)

    def create_grid_lines(self, bounds):
        """Create sparse light gray grid lines without any scale labels"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        # Calculate grid spacing - only show 3 grid lines per axis
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Show 3 grid lines per axis
        x_steps = 2
        y_steps = 2
        z_steps = 2

        # X-axis direction grid lines (parallel to Y-Z plane)
        if x_range > 0:
            x_step = x_range / x_steps
            for i in range(x_steps + 1):
                x_pos = x_min + i * x_step
                line_y = pv.Line((x_pos, y_min, z_min), (x_pos, y_max, z_min))
                line_z = pv.Line((x_pos, y_min, z_min), (x_pos, y_min, z_max))
                actor_y = self.plotter.add_mesh(line_y, color='lightgray', line_width=1, name=f'grid_x_{i}_y')
                actor_z = self.plotter.add_mesh(line_z, color='lightgray', line_width=1, name=f'grid_x_{i}_z')
                self.grid_actors.extend([actor_y, actor_z])

        # Y-axis direction grid lines (parallel to X-Z plane)
        if y_range > 0:
            y_step = y_range / y_steps
            for i in range(y_steps + 1):
                y_pos = y_min + i * y_step
                line_x = pv.Line((x_min, y_pos, z_min), (x_max, y_pos, z_min))
                line_z = pv.Line((x_min, y_pos, z_min), (x_min, y_pos, z_max))
                actor_x = self.plotter.add_mesh(line_x, color='lightgray', line_width=1, name=f'grid_y_{i}_x')
                actor_z = self.plotter.add_mesh(line_z, color='lightgray', line_width=1, name=f'grid_y_{i}_z')
                self.grid_actors.extend([actor_x, actor_z])

        # Z-axis direction grid lines (parallel to X-Y plane)
        if z_range > 0:
            z_step = z_range / z_steps
            for i in range(z_steps + 1):
                z_pos = z_min + i * z_step
                line_x = pv.Line((x_min, y_min, z_pos), (x_max, y_min, z_pos))
                line_y = pv.Line((x_min, y_min, z_pos), (x_min, y_max, z_pos))
                actor_x = self.plotter.add_mesh(line_x, color='lightgray', line_width=1, name=f'grid_z_{i}_x')
                actor_y = self.plotter.add_mesh(line_y, color='lightgray', line_width=1, name=f'grid_z_{i}_y')
                self.grid_actors.extend([actor_x, actor_y])

    def add_effective_range_text(self, bounds):
        """Add effective data range text in bottom-right corner of 3D scene"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        text_content = f"X: {x_min:.1f} ~ {x_max:.1f}\nY: {y_min:.1f} ~ {y_max:.1f}\nZ: {z_min:.1f} ~ {z_max:.1f}"

        # Remove old text
        if self.range_text_actor:
            self.plotter.remove_actor(self.range_text_actor)

        # Add text to bottom-right corner
        self.range_text_actor = self.plotter.add_text(
            text_content,
            position=(0.7, 0.05),
            color='black',
            font_size=10,
            name='effective_range'
        )

class DataFilterMainWindow(QtWidgets.QMainWindow):
    """Main window for combined 2D/3D filtering"""
    
    def __init__(self):
        super().__init__()
        
        # Current view mode (set before setup_ui)
        self.current_view = "2D"
        self.selected_history_index = None  # Track selected history item
        
        # Initialize shared data manager
        self.data_manager = SharedDataManager()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup main UI"""
        self.setWindowTitle("2D/3D Data Filter - Electron/Ion Visualization")
        self.resize(1400, 900)
        
        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create button toolbar
        self.create_button_toolbar()
        main_layout.addWidget(self.button_toolbar)
        
        # Create stacked widget for 2D/3D views
        self.stacked_widget = QtWidgets.QStackedWidget()
        
        # Create 2D and 3D visualizers
        self.visualizer_2d = TwoDVisualizerWidget(self.data_manager)
        self.visualizer_3d = ThreeDVisualizerWidget(self.data_manager)
        
        self.stacked_widget.addWidget(self.visualizer_2d)
        self.stacked_widget.addWidget(self.visualizer_3d)
        
        main_layout.addWidget(self.stacked_widget)
        
        # Create control panel
        self.create_control_panel()
        
        # Create horizontal layout for view and controls
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(self.stacked_widget, 4)
        h_layout.addWidget(self.control_panel, 1)
        
        main_layout.addLayout(h_layout)
        
        # Create filter toolbar (initially hidden)
        self.create_filter_toolbar()
        
        # Create parameter dock (initially hidden)
        self.create_parameter_dock()
        
    def create_button_toolbar(self):
        """Create ion/electron toggle buttons"""
        self.button_toolbar = QtWidgets.QToolBar()
        self.button_toolbar.setMovable(False)
        
        # Create electron button (2D)
        self.electron_btn = QtWidgets.QPushButton("Electron (2D)")
        self.electron_btn.setCheckable(True)
        self.electron_btn.setChecked(True)
        self.electron_btn.clicked.connect(self.show_2d_view)
        self.electron_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #45a049;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        # Create ion button (3D)
        self.ion_btn = QtWidgets.QPushButton("Ion (3D)")
        self.ion_btn.setCheckable(True)
        self.ion_btn.clicked.connect(self.show_3d_view)
        self.ion_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                border: 2px solid #1976D2;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        self.button_toolbar.addWidget(self.electron_btn)
        self.button_toolbar.addWidget(self.ion_btn)
        
        # Add spacer
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.button_toolbar.addWidget(spacer)
        
        # Add statistics label
        self.stats_label = QtWidgets.QLabel("Total Points: 0 | Remaining: 0 | Filtered: 0")
        self.button_toolbar.addWidget(self.stats_label)
        self.update_stats()
        
    def create_control_panel(self):
        """Create control panel"""
        self.control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout()
        
        # Filter control group
        filter_group = QtWidgets.QGroupBox("Filter Control")
        filter_layout = QtWidgets.QVBoxLayout()
        
        self.show_filter_checkbox = QtWidgets.QCheckBox("Show Filter")
        self.show_filter_checkbox.stateChanged.connect(self.toggle_filter_display)
        
        self.create_filter_btn = QtWidgets.QPushButton("Create Filter")
        self.create_filter_btn.clicked.connect(self.create_filter)
        
        self.apply_filter_btn = QtWidgets.QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.apply_filter_btn.setEnabled(False)
        
        filter_layout.addWidget(self.show_filter_checkbox)
        filter_layout.addWidget(self.create_filter_btn)
        filter_layout.addWidget(self.apply_filter_btn)
        filter_group.setLayout(filter_layout)
        
        # Display options
        display_group = QtWidgets.QGroupBox("Display Options")
        display_layout = QtWidgets.QVBoxLayout()
        
        self.show_filtered_out_checkbox = QtWidgets.QCheckBox("Show Filtered Data")
        self.show_filtered_out_checkbox.setChecked(True)
        self.show_filtered_out_checkbox.stateChanged.connect(self.toggle_filtered_out_points)
        
        self.reset_view_btn = QtWidgets.QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        
        self.reset_all_btn = QtWidgets.QPushButton("Reset All Filters")
        self.reset_all_btn.clicked.connect(self.reset_all)
        
        display_layout.addWidget(self.show_filtered_out_checkbox)
        display_layout.addWidget(self.reset_view_btn)
        display_layout.addWidget(self.reset_all_btn)
        display_group.setLayout(display_layout)
        
        # History group
        history_group = QtWidgets.QGroupBox("Filter History (Shared)")
        history_layout = QtWidgets.QVBoxLayout()
        
        self.history_list = QtWidgets.QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        
        self.undo_btn = QtWidgets.QPushButton("Undo Selected Filter")
        self.undo_btn.clicked.connect(self.undo_selected_filter)
        
        history_layout.addWidget(self.history_list)
        history_layout.addWidget(self.undo_btn)
        history_group.setLayout(history_layout)
        
        # Save group
        save_group = QtWidgets.QGroupBox("Save Options")
        save_layout = QtWidgets.QVBoxLayout()
        
        self.save_indices_btn = QtWidgets.QPushButton("Save Indices")
        self.save_indices_btn.clicked.connect(self.save_indices)
        
        self.save_data_btn = QtWidgets.QPushButton("Save Data")
        self.save_data_btn.clicked.connect(self.save_data)
        
        save_layout.addWidget(self.save_indices_btn)
        save_layout.addWidget(self.save_data_btn)
        save_group.setLayout(save_layout)
        
        # Add all groups to control panel
        control_layout.addWidget(filter_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(history_group)
        control_layout.addWidget(save_group)
        control_layout.addStretch()
        
        self.control_panel.setLayout(control_layout)
        self.control_panel.setMinimumWidth(350)
        self.control_panel.setMaximumWidth(400)
        
    def create_filter_toolbar(self):
        """Create filter type selection toolbar"""
        self.filter_toolbar = QtWidgets.QToolBar("Filter Types")
        self.filter_toolbar.setStyleSheet("""
            QToolBar {
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid gray;
                border-radius: 4px;
                padding: 2px;
            }
            QToolButton {
                background-color: rgba(255, 255, 255, 0.9);
                border: 1px solid lightgray;
                border-radius: 3px;
                padding: 2px;
            }
            QToolButton:checked {
                background-color: lightblue;
            }
        """)
        
        # 2D filter types
        if self.current_view == "2D":
            self.box_action = self.filter_toolbar.addAction("Rectangle")
            self.box_action.setCheckable(True)
            self.box_action.setChecked(True)
            self.box_action.triggered.connect(lambda: self.set_filter_type("box"))
            
            self.circle_action = self.filter_toolbar.addAction("Circle")
            self.circle_action.setCheckable(True)
            self.circle_action.triggered.connect(lambda: self.set_filter_type("circle"))
            
            self.ellipse_action = self.filter_toolbar.addAction("Ellipse")
            self.ellipse_action.setCheckable(True)
            self.ellipse_action.triggered.connect(lambda: self.set_filter_type("ellipse"))
            
            self.ring_action = self.filter_toolbar.addAction("Ring")
            self.ring_action.setCheckable(True)
            self.ring_action.triggered.connect(lambda: self.set_filter_type("ring"))
        
        # 3D filter types
        else:
            self.box_action = self.filter_toolbar.addAction("Box")
            self.box_action.setCheckable(True)
            self.box_action.setChecked(True)
            self.box_action.triggered.connect(lambda: self.set_filter_type("box"))
            
            self.sphere_action = self.filter_toolbar.addAction("Sphere")
            self.sphere_action.setCheckable(True)
            self.sphere_action.triggered.connect(lambda: self.set_filter_type("sphere"))
            
            self.cylinder_action = self.filter_toolbar.addAction("Cylinder")
            self.cylinder_action.setCheckable(True)
            self.cylinder_action.triggered.connect(lambda: self.set_filter_type("cylinder"))
            
            self.torus_action = self.filter_toolbar.addAction("Torus")
            self.torus_action.setCheckable(True)
            self.torus_action.triggered.connect(lambda: self.set_filter_type("torus"))
        
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.filter_toolbar)
        
        # Set filter type
        self.filter_type = "box"
        
    def create_parameter_dock(self):
        """Create parameter settings dock widget"""
        self.param_dock = QtWidgets.QDockWidget("Filter Parameters", self)
        self.param_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        param_widget = QtWidgets.QWidget()
        param_layout = QtWidgets.QFormLayout()
        
        # Position parameters
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1000, 1000)
        self.x_spin.setValue(0)
        self.x_spin.setSingleStep(1)
        
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1000, 1000)
        self.y_spin.setValue(0)
        self.y_spin.setSingleStep(1)
        
        # Add position controls
        param_layout.addRow("X Position:", self.x_spin)
        param_layout.addRow("Y Position:", self.y_spin)
        
        # 2D specific controls
        if self.current_view == "2D":
            self.x_size_spin = QtWidgets.QDoubleSpinBox()
            self.x_size_spin.setRange(0.1, 1000)
            self.x_size_spin.setValue(10)
            self.x_size_spin.setSingleStep(0.5)
            
            self.y_size_spin = QtWidgets.QDoubleSpinBox()
            self.y_size_spin.setRange(0.1, 1000)
            self.y_size_spin.setValue(10)
            self.y_size_spin.setSingleStep(0.5)
            
            self.radius_spin = QtWidgets.QDoubleSpinBox()
            self.radius_spin.setRange(0.1, 500)
            self.radius_spin.setValue(10)
            self.radius_spin.setSingleStep(0.5)
            
            param_layout.addRow("Width:", self.x_size_spin)
            param_layout.addRow("Height:", self.y_size_spin)
            param_layout.addRow("Radius:", self.radius_spin)
        
        # 3D specific controls
        else:
            self.z_spin = QtWidgets.QDoubleSpinBox()
            self.z_spin.setRange(-1000, 1000)
            self.z_spin.setValue(0)
            self.z_spin.setSingleStep(1)
            
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
            
            param_layout.addRow("Z Position:", self.z_spin)
            param_layout.addRow("X Length:", self.x_size_spin)
            param_layout.addRow("Y Width:", self.y_size_spin)
            param_layout.addRow("Z Height:", self.z_size_spin)
            param_layout.addRow("Radius:", self.radius_spin)
        
        param_widget.setLayout(param_layout)
        self.param_dock.setWidget(param_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.param_dock)
        self.param_dock.hide()
        
    def show_2d_view(self):
        """Switch to 2D electron view"""
        self.electron_btn.setChecked(True)
        self.ion_btn.setChecked(False)
        self.current_view = "2D"
        self.stacked_widget.setCurrentIndex(0)
        
        # Recreate toolbar and parameters for 2D
        self.removeToolBar(self.filter_toolbar)
        self.create_filter_toolbar()
        
        self.removeDockWidget(self.param_dock)
        self.create_parameter_dock()
        
    def show_3d_view(self):
        """Switch to 3D ion view"""
        self.ion_btn.setChecked(True)
        self.electron_btn.setChecked(False)
        self.current_view = "3D"
        self.stacked_widget.setCurrentIndex(1)
        
        # Recreate toolbar and parameters for 3D
        self.removeToolBar(self.filter_toolbar)
        self.create_filter_toolbar()
        
        self.removeDockWidget(self.param_dock)
        self.create_parameter_dock()
        
    def set_filter_type(self, filter_type):
        """Set current filter type"""
        # If clicking the currently selected button, deselect and hide parameter window
        if self.filter_type == filter_type:
            for action in self.filter_toolbar.actions():
                action.setChecked(False)
            self.filter_type = None
            self.param_dock.hide()
            return
        
        self.filter_type = filter_type
        
        # Reset all button states
        for action in self.filter_toolbar.actions():
            action.setChecked(False)
        
        # Set current button
        for action in self.filter_toolbar.actions():
            if filter_type in action.text().lower():
                action.setChecked(True)
                break
        
        self.param_dock.show()
        
    def create_filter(self):
        """Create filter visualization"""
        if self.current_view == "2D":
            self.create_2d_filter()
        else:
            self.create_3d_filter()
            
        self.apply_filter_btn.setEnabled(True)
        
    def create_2d_filter(self):
        """Create 2D filter"""
        # Remove existing filter
        if hasattr(self.visualizer_2d, 'current_filter_patch') and self.visualizer_2d.current_filter_patch:
            self.visualizer_2d.current_filter_patch.remove()
        
        x, y = self.x_spin.value(), self.y_spin.value()
        
        if self.filter_type == "box":
            width = self.x_size_spin.value()
            height = self.y_size_spin.value()
            filter_patch = Rectangle((x - width/2, y - height/2), width, height, 
                                   fill=True, alpha=0.3, color='darkred')
        elif self.filter_type == "circle":
            radius = self.radius_spin.value()
            filter_patch = Circle((x, y), radius, fill=True, alpha=0.3, color='darkgreen')
        elif self.filter_type == "ellipse":
            width = self.x_size_spin.value()
            height = self.y_size_spin.value()
            filter_patch = Ellipse((x, y), width, height, fill=True, alpha=0.3, color='darkblue')
        elif self.filter_type == "ring":
            # For ring, use inner radius from radius_spin and outer radius from x_size_spin
            inner_radius = self.radius_spin.value()
            outer_radius = self.x_size_spin.value()
            filter_patch = Wedge((x, y), outer_radius, 0, 360, width=outer_radius-inner_radius, 
                               fill=True, alpha=0.3, color='darkorange')
        else:
            return
        
        self.visualizer_2d.current_filter_patch = self.visualizer_2d.ax.add_patch(filter_patch)
        self.visualizer_2d.canvas.draw()
        
    def create_3d_filter(self):
        """Create 3D filter"""
        # Remove existing filter
        if hasattr(self.visualizer_3d, 'current_filter_actor') and self.visualizer_3d.current_filter_actor:
            self.visualizer_3d.plotter.remove_actor(self.visualizer_3d.current_filter_actor)
        
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
            radius = self.radius_spin.value()
            height = self.z_size_spin.value()
            filter_mesh = pv.Cylinder(center=(x, y, z), radius=radius, height=height, direction=(0, 0, 1))
            color = 'darkblue'
        elif self.filter_type == "torus":
            inner_radius = self.radius_spin.value()
            outer_radius = self.x_size_spin.value()
            height = self.z_size_spin.value()
            filter_mesh = pv.ParametricTorus(
                ringradius=(inner_radius + outer_radius) / 2,
                crosssectionradius=(outer_radius - inner_radius) / 2
            )
            filter_mesh.translate((x, y, z))
            color = 'darkorange'
        else:
            return
        
        self.visualizer_3d.current_filter_actor = self.visualizer_3d.plotter.add_mesh(
            filter_mesh, color=color, opacity=0.3
        )
        
    def apply_filter(self):
        """Apply filter to shared data"""
        if self.current_view == "2D":
            filtered_indices = self.apply_2d_filter()
        else:
            filtered_indices = self.apply_3d_filter()
        
        if filtered_indices is not None and len(filtered_indices) > 0:
            # Create filter operation
            if self.current_view == "2D":
                x, y = self.x_spin.value(), self.y_spin.value()
                center = (x, y)
                if self.filter_type == "box":
                    size = (self.x_size_spin.value(), self.y_size_spin.value())
                elif self.filter_type == "circle":
                    size = (self.radius_spin.value(),)
                elif self.filter_type == "ellipse":
                    size = (self.x_size_spin.value(), self.y_size_spin.value())
                elif self.filter_type == "ring":
                    size = (self.radius_spin.value(), self.x_size_spin.value())
            else:
                x, y, z = self.x_spin.value(), self.y_spin.value(), self.z_spin.value()
                center = (x, y, z)
                if self.filter_type == "box":
                    size = (self.x_size_spin.value(), self.y_size_spin.value(), self.z_size_spin.value())
                elif self.filter_type == "sphere":
                    size = (self.radius_spin.value(),)
                elif self.filter_type == "cylinder":
                    size = (self.radius_spin.value(), self.z_size_spin.value())
                elif self.filter_type == "torus":
                    size = (self.radius_spin.value(), self.x_size_spin.value(), self.z_size_spin.value())
            
            filter_op = FilterOperation(
                type=self.filter_type,
                center=center,
                size=size,
                filtered_count=len(filtered_indices),
                filtered_indices=filtered_indices,
                dimension=self.current_view
            )
            
            # Apply to shared data manager
            self.data_manager.apply_filter(filter_op, self.current_view)
            
            # Update both views
            self.visualizer_2d.update_view()
            self.visualizer_3d.update_view()
            
            # Update UI
            self.update_history_list()
            self.update_stats()
            
            # Hide filter
            self.hide_filter()
            
            print(f"Applied {self.current_view} filter: {len(filtered_indices)} points filtered")
            
    def apply_2d_filter(self):
        """Apply 2D filter and return filtered indices"""
        min_points = min(len(self.data_manager.data_2d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_2d[:min_points]
        current_mask = self.data_manager.current_2d_indices_mask[:min_points]
        
        # Get currently active points
        active_indices = np.where(current_mask)[0]
        active_points = current_data[active_indices]
        
        x, y = self.x_spin.value(), self.y_spin.value()
        
        if self.filter_type == "box":
            x_min, x_max = x - self.x_size_spin.value()/2, x + self.x_size_spin.value()/2
            y_min, y_max = y - self.y_size_spin.value()/2, y + self.y_size_spin.value()/2
            local_mask = (
                (active_points[:, 0] >= x_min) & (active_points[:, 0] <= x_max) &
                (active_points[:, 1] >= y_min) & (active_points[:, 1] <= y_max)
            )
        elif self.filter_type == "circle":
            radius = self.radius_spin.value()
            distances = np.linalg.norm(active_points - np.array([x, y]), axis=1)
            local_mask = distances <= radius
        elif self.filter_type == "ellipse":
            width = self.x_size_spin.value()
            height = self.y_size_spin.value()
            normalized_x = (active_points[:, 0] - x) / (width/2)
            normalized_y = (active_points[:, 1] - y) / (height/2)
            local_mask = normalized_x**2 + normalized_y**2 <= 1
        elif self.filter_type == "ring":
            inner_radius = self.radius_spin.value()
            outer_radius = self.x_size_spin.value()
            distances = np.linalg.norm(active_points - np.array([x, y]), axis=1)
            local_mask = (distances >= inner_radius) & (distances <= outer_radius)
        else:
            return None
        
        # Get filtered indices
        filtered_local_indices = np.where(local_mask)[0]
        filtered_global_indices = active_indices[filtered_local_indices]
        
        return filtered_global_indices
        
    def apply_3d_filter(self):
        """Apply 3D filter and return filtered indices"""
        min_points = min(len(self.data_manager.data_3d), len(self.data_manager.shared_indices_mask))
        current_data = self.data_manager.data_3d[:min_points]
        current_mask = self.data_manager.current_3d_indices_mask[:min_points]
        
        # Get currently active points
        active_indices = np.where(current_mask)[0]
        active_points = current_data[active_indices]
        
        x, y, z = self.x_spin.value(), self.y_spin.value(), self.z_spin.value()
        
        if self.filter_type == "box":
            x_min, x_max = x - self.x_size_spin.value()/2, x + self.x_size_spin.value()/2
            y_min, y_max = y - self.y_size_spin.value()/2, y + self.y_size_spin.value()/2
            z_min, z_max = z - self.z_size_spin.value()/2, z + self.z_size_spin.value()/2
            local_mask = (
                (active_points[:, 0] >= x_min) & (active_points[:, 0] <= x_max) &
                (active_points[:, 1] >= y_min) & (active_points[:, 1] <= y_max) &
                (active_points[:, 2] >= z_min) & (active_points[:, 2] <= z_max)
            )
        elif self.filter_type == "sphere":
            radius = self.radius_spin.value()
            distances = np.linalg.norm(active_points - np.array([x, y, z]), axis=1)
            local_mask = distances <= radius
        elif self.filter_type == "cylinder":
            radius = self.radius_spin.value()
            height = self.z_size_spin.value()
            xy_distances = np.linalg.norm(active_points[:, :2] - np.array([x, y]), axis=1)
            z_distances = np.abs(active_points[:, 2] - z)
            local_mask = (xy_distances <= radius) & (z_distances <= height / 2)
        elif self.filter_type == "torus":
            inner_radius = self.radius_spin.value()
            outer_radius = self.x_size_spin.value()
            height = self.z_size_spin.value()
            
            major_radius = (inner_radius + outer_radius) / 2
            minor_radius = (outer_radius - inner_radius) / 2
            
            rel_points = active_points - np.array([x, y, z])
            angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
            
            local_x = np.cos(angles)
            local_y = np.sin(angles)
            
            local_pos_x = rel_points[:, 0] * local_x + rel_points[:, 1] * local_y - major_radius
            local_pos_y = -rel_points[:, 0] * local_y + rel_points[:, 1] * local_x
            
            square_condition = (np.abs(local_pos_x) <= minor_radius) & (np.abs(local_pos_y) <= minor_radius)
            height_condition = np.abs(rel_points[:, 2]) <= height / 2
            
            local_mask = square_condition & height_condition
        else:
            return None
        
        # Get filtered indices
        filtered_local_indices = np.where(local_mask)[0]
        filtered_global_indices = active_indices[filtered_local_indices]
        
        return filtered_global_indices
        
    def toggle_filter_display(self, state):
        """Toggle filter display"""
        if state:
            if self.apply_filter_btn.isEnabled():
                self.create_filter()
        else:
            self.hide_filter()
            
    def hide_filter(self):
        """Hide current filter"""
        if self.current_view == "2D":
            if hasattr(self.visualizer_2d, 'current_filter_patch') and self.visualizer_2d.current_filter_patch:
                try:
                    self.visualizer_2d.current_filter_patch.remove()
                except (ValueError, NotImplementedError):
                    pass  # Patch already removed or can't be removed
                self.visualizer_2d.current_filter_patch = None
                self.visualizer_2d.canvas.draw()
        else:
            if hasattr(self.visualizer_3d, 'current_filter_actor') and self.visualizer_3d.current_filter_actor:
                self.visualizer_3d.plotter.remove_actor(self.visualizer_3d.current_filter_actor)
                self.visualizer_3d.current_filter_actor = None
        
        self.apply_filter_btn.setEnabled(False)
        
    def toggle_filtered_out_points(self, state):
        """Toggle filtered points display"""
        self.visualizer_2d.show_filtered_out = bool(state)
        self.visualizer_3d.show_filtered_out = bool(state)
        
        self.visualizer_2d.update_view()
        self.visualizer_3d.update_view()
        
    def reset_view(self):
        """Reset current view"""
        if self.current_view == "2D":
            self.visualizer_2d.ax.set_aspect('equal', adjustable='box')
            self.visualizer_2d.canvas.draw()
        else:
            self.visualizer_3d.plotter.camera_position = 'iso'
            self.visualizer_3d.plotter.reset_camera()
            
    def reset_all(self):
        """Reset all filters"""
        self.data_manager.reset_all()
        
        self.visualizer_2d.update_view()
        self.visualizer_3d.update_view()
        
        self.update_history_list()
        self.update_stats()
        
    def update_history_list(self):
        """Update filter history list"""
        self.history_list.clear()
        for i, filter_op in enumerate(self.data_manager.filter_history):
            item_text = f"Filter {i+1}: {filter_op.dimension} {filter_op.type} | "
            item_text += f"Position({', '.join(f'{c:.1f}' for c in filter_op.center)}) | "
            item_text += f"Size({', '.join(f'{s:.1f}' for s in filter_op.size)}) | "
            item_text += f"Count: {filter_op.filtered_count}"
            
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, i)
            self.history_list.addItem(item)
            
    def on_history_item_clicked(self, item):
        """Handle history item click"""
        index = item.data(QtCore.Qt.UserRole)
        if index is not None and index < len(self.data_manager.filter_history):
            filter_op = self.data_manager.filter_history[index]
            
            # If clicking the same item, deselect and hide history filter
            if self.selected_history_index == index:
                self.selected_history_index = None
                self.visualizer_2d.hide_history_filter()
                self.visualizer_3d.hide_history_filter()
                # Update both views to clear the history display
                self.visualizer_2d.update_view()
                self.visualizer_3d.update_view()
                return
            
            # Otherwise select this item
            self.selected_history_index = index
            
            # Switch to appropriate view first
            if filter_op.dimension == "2D":
                self.show_2d_view()
            else:
                self.show_3d_view()
            
            # Set parameters (after view switch to ensure widgets exist)
            self.x_spin.setValue(filter_op.center[0])
            self.y_spin.setValue(filter_op.center[1])
            
            if len(filter_op.center) > 2 and hasattr(self, 'z_spin'):
                self.z_spin.setValue(filter_op.center[2])
            
            # Set size parameters based on filter type
            if filter_op.type == "box":
                self.x_size_spin.setValue(filter_op.size[0])
                self.y_size_spin.setValue(filter_op.size[1])
                if len(filter_op.size) > 2 and hasattr(self, 'z_size_spin'):
                    self.z_size_spin.setValue(filter_op.size[2])
            elif filter_op.type in ["circle", "sphere"]:
                self.radius_spin.setValue(filter_op.size[0])
            elif filter_op.type == "ellipse":
                self.x_size_spin.setValue(filter_op.size[0])
                self.y_size_spin.setValue(filter_op.size[1])
            elif filter_op.type in ["ring", "torus"]:
                self.radius_spin.setValue(filter_op.size[0])
                self.x_size_spin.setValue(filter_op.size[1])
                if len(filter_op.size) > 2 and hasattr(self, 'z_size_spin'):
                    self.z_size_spin.setValue(filter_op.size[2])
            
            # Set filter type and show filter
            self.set_filter_type(filter_op.type)
            
            # Show history filter with colored points
            if filter_op.dimension == "2D":
                self.visualizer_2d.show_history_filter(filter_op)
            else:
                self.visualizer_3d.show_history_filter(filter_op)
            
    def undo_selected_filter(self):
        """Undo selected filter"""
        if self.selected_history_index is not None and self.selected_history_index < len(self.data_manager.filter_history):
            filter_op = self.data_manager.undo_filter(self.selected_history_index)
            
            if filter_op:
                # Reset selected history index
                self.selected_history_index = None
                
                # Hide history filters
                self.visualizer_2d.hide_history_filter()
                self.visualizer_3d.hide_history_filter()
                
                # Update both views
                self.visualizer_2d.update_view()
                self.visualizer_3d.update_view()
                
                self.update_history_list()
                self.update_stats()
                
                print(f"Undid {filter_op.dimension} {filter_op.type} filter: {filter_op.filtered_count} points restored")
                
    def update_stats(self):
        """Update statistics display"""
        total_points = len(self.data_manager.shared_indices_mask)
        remaining_points = np.sum(self.data_manager.shared_indices_mask)
        filtered_points = total_points - remaining_points
        
        self.stats_label.setText(
            f"Total: {total_points:,} | Remaining: {remaining_points:,} ({remaining_points/total_points:.1%}) | Filtered: {filtered_points:,} ({filtered_points/total_points:.1%})"
        )
        
    def save_indices(self):
        """Save current indices"""
        try:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Indices", "filtered_indices.pkl", "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if filename:
                current_indices = np.where(self.data_manager.shared_indices_mask)[0]
                save_data = {
                    'indices': current_indices,
                    'total_original_points': len(self.data_manager.shared_indices_mask),
                    'current_points_count': len(current_indices),
                    'filter_history': self.data_manager.filter_history
                }
                
                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)
                
                print(f"Saved {len(current_indices)} indices to {filename}")
                QtWidgets.QMessageBox.information(self, "Save Successful", f"Saved {len(current_indices):,} indices to:\n{filename}")
                
        except Exception as e:
            print(f"Error saving indices: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save indices:\n{str(e)}")
            
    def save_data(self):
        """Save current data"""
        try:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Data", "filtered_data.pkl", "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if filename:
                min_2d_points = min(len(self.data_manager.data_2d), len(self.data_manager.shared_indices_mask))
                min_3d_points = min(len(self.data_manager.data_3d), len(self.data_manager.shared_indices_mask))
                
                current_2d_data = self.data_manager.data_2d[:min_2d_points][self.data_manager.shared_indices_mask[:min_2d_points]]
                current_3d_data = self.data_manager.data_3d[:min_3d_points][self.data_manager.shared_indices_mask[:min_3d_points]]
                
                save_data = {
                    'data_2d': current_2d_data,
                    'data_3d': current_3d_data,
                    'indices': np.where(self.data_manager.shared_indices_mask)[0],
                    'total_original_points': len(self.data_manager.shared_indices_mask),
                    'current_points_count': len(current_2d_data),
                    'filter_history': self.data_manager.filter_history
                }
                
                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)
                
                print(f"Saved data to {filename}")
                QtWidgets.QMessageBox.information(self, "Save Successful", f"Saved {len(current_2d_data):,} points to:\n{filename}")
                
        except Exception as e:
            print(f"Error saving data: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save data:\n{str(e)}")

if __name__ == "__main__":
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create main window
    window = DataFilterMainWindow()
    window.show()
    
    # Start application
    sys.exit(app.exec_())
