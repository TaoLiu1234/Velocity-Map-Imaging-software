import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle, Circle, Ellipse, Wedge
from PyQt5 import QtWidgets, QtCore
import sys
import warnings
import pickle

# Suppress the SIP deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

class TwoDVisualizer(QtWidgets.QMainWindow):
    def __init__(self, point_data=None):
        super().__init__()

        # Create main frame
        self.frame = QtWidgets.QFrame()
        self.h_layout = QtWidgets.QHBoxLayout()  # Horizontal layout: 2D view + right control panel

        # Create matplotlib figure and canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create matplotlib layout
        self.matplotlib_layout = QtWidgets.QVBoxLayout()
        self.matplotlib_layout.addWidget(self.toolbar)
        self.matplotlib_layout.addWidget(self.canvas)
        
        self.h_layout.addLayout(self.matplotlib_layout, 4)  # Takes 4/5 width

        # Create control panel
        self.create_control_panel()
        self.h_layout.addWidget(self.control_panel, 1)  # Takes 1/5 width

        self.frame.setLayout(self.h_layout)
        self.setCentralWidget(self.frame)

        # Initialize data
        self.point_data = point_data
        self.original_points = None  # Original data
        self.current_points = None   # Currently displayed data
        self.current_indices_mask = None  # Index mask for currently displayed points
        self.original_indices_mask = None  # Original index mask (for cumulative filtering)

        # Filter related
        self.filtered_out_indices_history = []  # History of filtered point indices
        self.filtered_out_points_history = []   # History of filtered point coordinates
        self.filter_type = "box"  # Current filter type
        self.show_filtered_out = True  # Whether to show filtered points
        self.show_filter_volume = False  # Whether to show filter

        # Current filter patch
        self.current_filter_patch = None
        self.filtered_out_scatter = None  # Filtered out points scatter plot
        self.history_filter_patch = None  # History filter patch
        self.history_filtered_out_scatter = None  # History filtered out scatter plot

        # Currently displayed scatter plot
        self.current_scatter = None

        # Grid and range text
        self.grid_lines = []
        self.range_text = None  # Text for displaying effective data range

        # History
        self.filter_history = []  # Filter history
        self.selected_history_index = None  # Currently selected history index

        # Set up initial view
        self.setup_scene()
        
        # Create floating filter toolbar
        self.create_filter_toolbar()

        # Create parameter settings panel (dock widget)
        self.create_parameter_dock()

    def create_filter_toolbar(self):
        """Create filter selection toolbar floating in the top-right corner of 2D view"""
        # Create toolbar
        self.filter_toolbar = QtWidgets.QToolBar("Filter")
        # Set style to make it look like a floating button group
        self.filter_toolbar.setStyleSheet("""
            QToolBar {
                background-color: rgba(255, 255, 255, 0.8);
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
        # Set as floating toolbar
        self.filter_toolbar.setMovable(False)
        self.filter_toolbar.setFloatable(False)
        self.filter_toolbar.setOrientation(QtCore.Qt.Horizontal)

        # Create actions (buttons) - 2D equivalents
        self.box_action = self.filter_toolbar.addAction("Rectangle")
        self.box_action.setCheckable(True)
        self.box_action.setChecked(True)
        self.box_action.triggered.connect(lambda: self.set_filter_type("box"))

        self.sphere_action = self.filter_toolbar.addAction("Circle")
        self.sphere_action.setCheckable(True)
        self.sphere_action.triggered.connect(lambda: self.set_filter_type("circle"))

        self.cylinder_action = self.filter_toolbar.addAction("Ellipse")
        self.cylinder_action.setCheckable(True)
        self.cylinder_action.triggered.connect(lambda: self.set_filter_type("ellipse"))

        self.torus_action = self.filter_toolbar.addAction("Ring")
        self.torus_action.setCheckable(True)
        self.torus_action.triggered.connect(lambda: self.set_filter_type("ring"))

        # Add toolbar to top of main window
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.filter_toolbar)
        # Manually set toolbar position to top-right corner
        self.filter_toolbar.move(self.width() - self.filter_toolbar.width() - 10, 10)
        # Override resizeEvent to dynamically adjust toolbar position
        self.old_size = self.size()
        self.resizeEvent = self._resize_toolbar

    def _resize_toolbar(self, event):
        """Override resizeEvent to dynamically adjust toolbar position"""
        super().resizeEvent(event)
        # Check if window size has changed
        if self.size() != self.old_size:
            self.old_size = self.size()
            # Reset toolbar position to top-right corner
            self.filter_toolbar.move(self.width() - self.filter_toolbar.width() - 10, 10)

    def create_parameter_dock(self):
        """Create parameter settings panel (dock widget)"""
        # Create DockWidget
        self.param_dock = QtWidgets.QDockWidget("Filter Parameters", self)
        self.param_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.param_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable | QtWidgets.QDockWidget.DockWidgetClosable)

        # Set minimum and maximum size for DockWidget
        self.param_dock.setMinimumSize(250, 300)
        self.param_dock.setMaximumSize(400, 600)

        # Create central widget for parameter settings
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

        # Box (Rectangle) parameters
        self.x_size_spin = QtWidgets.QDoubleSpinBox()
        self.x_size_spin.setRange(0.1, 1000)
        self.x_size_spin.setValue(10)
        self.x_size_spin.setSingleStep(0.5)

        self.y_size_spin = QtWidgets.QDoubleSpinBox()
        self.y_size_spin.setRange(0.1, 1000)
        self.y_size_spin.setValue(10)
        self.y_size_spin.setSingleStep(0.5)

        # Circle parameters
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 500)
        self.radius_spin.setValue(10)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setVisible(False)

        # Ellipse parameters
        self.ellipse_width_spin = QtWidgets.QDoubleSpinBox()
        self.ellipse_width_spin.setRange(0.1, 500)
        self.ellipse_width_spin.setValue(20)
        self.ellipse_width_spin.setSingleStep(0.5)
        self.ellipse_width_spin.setVisible(False)
        
        self.ellipse_height_spin = QtWidgets.QDoubleSpinBox()
        self.ellipse_height_spin.setRange(0.1, 500)
        self.ellipse_height_spin.setValue(10)
        self.ellipse_height_spin.setSingleStep(0.5)
        self.ellipse_height_spin.setVisible(False)

        # Ring parameters
        self.ring_inner_radius_spin = QtWidgets.QDoubleSpinBox()
        self.ring_inner_radius_spin.setRange(0.1, 500)
        self.ring_inner_radius_spin.setValue(5)
        self.ring_inner_radius_spin.setSingleStep(0.5)
        self.ring_inner_radius_spin.setVisible(False)
        
        self.ring_outer_radius_spin = QtWidgets.QDoubleSpinBox()
        self.ring_outer_radius_spin.setRange(0.1, 500)
        self.ring_outer_radius_spin.setValue(10)
        self.ring_outer_radius_spin.setSingleStep(0.5)
        self.ring_outer_radius_spin.setVisible(False)

        # Add parameters to layout
        param_layout.addRow("X Position:", self.x_spin)
        param_layout.addRow("Y Position:", self.y_spin)
        param_layout.addRow("Width:", self.x_size_spin)
        param_layout.addRow("Height:", self.y_size_spin)
        param_layout.addRow("Radius:", self.radius_spin)
        param_layout.addRow("Ellipse Width:", self.ellipse_width_spin)
        param_layout.addRow("Ellipse Height:", self.ellipse_height_spin)
        param_layout.addRow("Ring Inner Radius:", self.ring_inner_radius_spin)
        param_layout.addRow("Ring Outer Radius:", self.ring_outer_radius_spin)
        
        param_widget.setLayout(param_layout)
        self.param_dock.setWidget(param_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.param_dock)
        self.param_dock.hide()

    def create_control_panel(self):
        """Create right control panel"""
        self.control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout()

        # Filter control
        filter_control_group = QtWidgets.QGroupBox("Filter Control")
        filter_control_layout = QtWidgets.QVBoxLayout()

        self.show_filter_checkbox = QtWidgets.QCheckBox("Show Filter")
        self.show_filter_checkbox.setChecked(False)
        self.show_filter_checkbox.stateChanged.connect(self.toggle_filter_display)

        self.create_filter_btn = QtWidgets.QPushButton("Create Filter")
        self.create_filter_btn.clicked.connect(self.create_filter)

        self.apply_filter_btn = QtWidgets.QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        self.apply_filter_btn.setEnabled(False)

        filter_control_layout.addWidget(self.show_filter_checkbox)
        filter_control_layout.addWidget(self.create_filter_btn)
        filter_control_layout.addWidget(self.apply_filter_btn)
        filter_control_group.setLayout(filter_control_layout)

        # Display options
        display_group = QtWidgets.QGroupBox("Display Options")
        display_layout = QtWidgets.QVBoxLayout()

        self.show_filtered_out_checkbox = QtWidgets.QCheckBox("Show Filtered Data Points")
        self.show_filtered_out_checkbox.setChecked(True)
        self.show_filtered_out_checkbox.stateChanged.connect(self.toggle_filtered_out_points)

        self.reset_view_btn = QtWidgets.QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)

        self.reset_all_btn = QtWidgets.QPushButton("Reset All")
        self.reset_all_btn.clicked.connect(self.reset_all)

        display_layout.addWidget(self.show_filtered_out_checkbox)
        display_layout.addWidget(self.reset_view_btn)
        display_layout.addWidget(self.reset_all_btn)
        display_group.setLayout(display_layout)

        # Statistics
        stats_group = QtWidgets.QGroupBox("Data Statistics")
        stats_layout = QtWidgets.QVBoxLayout()
        self.stats_label = QtWidgets.QLabel("Total Points: 0\nRemaining Points: 0\nFiltered Points: 0")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)

        # Save Options
        save_group = QtWidgets.QGroupBox("Save Options")
        save_layout = QtWidgets.QVBoxLayout()

        self.save_indices_btn = QtWidgets.QPushButton("Save Current Indices")
        self.save_indices_btn.clicked.connect(self.save_current_indices)

        self.save_data_btn = QtWidgets.QPushButton("Save Current Data")
        self.save_data_btn.clicked.connect(self.save_current_data)

        save_layout.addWidget(self.save_indices_btn)
        save_layout.addWidget(self.save_data_btn)
        save_group.setLayout(save_layout)

        # History
        history_group = QtWidgets.QGroupBox("Filter History")
        history_layout = QtWidgets.QVBoxLayout()

        self.history_list = QtWidgets.QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)

        self.undo_last_btn = QtWidgets.QPushButton("Undo Selected Filter")
        self.undo_last_btn.clicked.connect(self.undo_selected_filter)

        history_layout.addWidget(self.history_list)
        history_layout.addWidget(self.undo_last_btn)
        history_group.setLayout(history_layout)

        # Add to control panel
        control_layout.addWidget(filter_control_group)
        control_layout.addWidget(display_group)
        control_layout.addWidget(stats_group)
        control_layout.addWidget(save_group)
        control_layout.addWidget(history_group)
        control_layout.addStretch()

        self.control_panel.setLayout(control_layout)
        self.control_panel.setMinimumWidth(350)
        self.control_panel.setMaximumWidth(400)

    def set_filter_type(self, filter_type):
        """Set current filter type"""
        # If clicking the currently selected button, deselect and hide parameter window
        if self.filter_type == filter_type:
            self.box_action.setChecked(False)
            self.sphere_action.setChecked(False)
            self.cylinder_action.setChecked(False)
            self.torus_action.setChecked(False)
            self.filter_type = None
            self.param_dock.hide()
            return

        self.filter_type = filter_type
        
        # Reset all button selected states
        self.box_action.setChecked(False)
        self.sphere_action.setChecked(False)
        self.cylinder_action.setChecked(False)
        self.torus_action.setChecked(False)
        
        # Select current button
        if filter_type == "box":
            self.box_action.setChecked(True)
        elif filter_type == "circle":
            self.sphere_action.setChecked(True)
        elif filter_type == "ellipse":
            self.cylinder_action.setChecked(True)
        elif filter_type == "ring":
            self.torus_action.setChecked(True)
        
        # Show/hide corresponding parameter controls
        self.x_size_spin.setVisible(filter_type == "box")
        self.y_size_spin.setVisible(filter_type == "box")
        self.radius_spin.setVisible(filter_type == "circle")
        self.ellipse_width_spin.setVisible(filter_type == "ellipse")
        self.ellipse_height_spin.setVisible(filter_type == "ellipse")
        self.ring_inner_radius_spin.setVisible(filter_type == "ring")
        self.ring_outer_radius_spin.setVisible(filter_type == "ring")

        self.param_dock.show()

    def setup_scene(self):
        """Set up initial scene"""
        # If no data provided, generate sample data (100,000 points)
        if self.point_data is None:
            print("Generating sample dataset (100,000 points)...")
            np.random.seed(42)
            x = np.random.normal(0, 50, 100000)
            y = np.random.normal(0, 50, 100000)
            self.point_data = np.column_stack((x, y))

        # Save original data and index mask
        self.original_points = self.point_data.copy()
        self.current_points = self.point_data.copy()
        self.original_indices_mask = np.ones(len(self.point_data), dtype=bool)
        self.current_indices_mask = np.ones(len(self.point_data), dtype=bool)

        # Create scatter plot
        self.update_main_view()

        # Add coordinate axes
        self.ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        self.ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)

        # Add grid
        self.update_grid_with_scales()

        # Set initial view limits
        self.reset_view()
        
        # Add title
        self.ax.set_title("Scientific 2D Scatter Plot Visualization System", fontsize=12)
        
        # Set equal aspect ratio to prevent deformation
        self.ax.set_aspect('equal', adjustable='box')

        # Update statistics
        self.update_stats()
        
        # Refresh canvas
        self.canvas.draw()

    def update_grid_with_scales(self):
        """Update grid and scale rulers"""
        # Remove old grid lines
        for line in self.grid_lines:
            line.remove()
        self.grid_lines = []

        # Remove old effective data range text
        if self.range_text is not None:
            self.range_text.remove()
            self.range_text = None

        # Get current point cloud boundaries - based only on currently displayed points
        if len(self.current_points) > 0:
            x_min, x_max = self.current_points[:, 0].min(), self.current_points[:, 0].max()
            y_min, y_max = self.current_points[:, 1].min(), self.current_points[:, 1].max()
        else:
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10

        # Add effective data range text in bottom-right corner
        text_content = f"X: {x_min:.1f} ~ {x_max:.1f}\nY: {y_min:.1f} ~ {y_max:.1f}"
        self.range_text = self.ax.text(0.95, 0.05, text_content, transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def toggle_filter_display(self, state):
        """Toggle filter display"""
        if state:
            if self.current_filter_patch is None and self.apply_filter_btn.isEnabled():
                self.create_filter()
        else:
            self.hide_filter()

    def hide_filter(self):
        """Hide current filter"""
        if self.current_filter_patch is not None:
            self.current_filter_patch.remove()
            self.current_filter_patch = None
            self.canvas.draw()
        self.apply_filter_btn.setEnabled(False)

    def hide_history_filter(self):
        """Hide history filter and related points"""
        if self.history_filter_patch is not None:
            self.history_filter_patch.remove()
            self.history_filter_patch = None
        if self.history_filtered_out_scatter is not None:
            self.history_filtered_out_scatter.remove()
            self.history_filtered_out_scatter = None
        self.canvas.draw()

    def create_filter(self):
        """Create filter geometry"""
        self.hide_filter()

        x, y = self.x_spin.value(), self.y_spin.value()

        filter_patch = None
        color = 'gray'

        if self.filter_type == "box":
            width = self.x_size_spin.value()
            height = self.y_size_spin.value()
            filter_patch = Rectangle((x - width/2, y - height/2), width, height, 
                                   fill=True, alpha=0.3, color='darkred')
        elif self.filter_type == "circle":
            radius = self.radius_spin.value()
            filter_patch = Circle((x, y), radius, fill=True, alpha=0.3, color='darkgreen')
        elif self.filter_type == "ellipse":
            width = self.ellipse_width_spin.value()
            height = self.ellipse_height_spin.value()
            filter_patch = Ellipse((x, y), width, height, fill=True, alpha=0.3, color='darkblue')
        elif self.filter_type == "ring":
            inner_radius = self.ring_inner_radius_spin.value()
            outer_radius = self.ring_outer_radius_spin.value()
            filter_patch = Wedge((x, y), outer_radius, 0, 360, width=outer_radius-inner_radius, 
                               fill=True, alpha=0.3, color='darkorange')

        if filter_patch is not None:
            # Add filter to scene and save patch reference
            self.current_filter_patch = self.ax.add_patch(filter_patch)
            self.canvas.draw()
        else:
            print("Error: Failed to create filter patch")
            return

        self.apply_filter_btn.setEnabled(True)

        if not self.show_filter_checkbox.isChecked():
            self.hide_filter()

    def apply_filter(self):
        """Apply filter based on indices - works on index mask rather than modifying data directly"""
        if self.current_filter_patch is None:
            return

        # Get currently active indices (where mask is True)
        current_active_indices = np.where(self.current_indices_mask)[0]
        
        # Get points corresponding to currently active indices
        current_active_points = self.original_points[current_active_indices]
        
        x, y = self.x_spin.value(), self.y_spin.value()

        # Calculate mask based on filter type for currently active points
        if self.filter_type == "box":
            x_min, x_max = x - self.x_size_spin.value()/2, x + self.x_size_spin.value()/2
            y_min, y_max = y - self.y_size_spin.value()/2, y + self.y_size_spin.value()/2
            local_mask = (
                (current_active_points[:, 0] >= x_min) & (current_active_points[:, 0] <= x_max) &
                (current_active_points[:, 1] >= y_min) & (current_active_points[:, 1] <= y_max)
            )
        elif self.filter_type == "circle":
            radius = self.radius_spin.value()
            distances = np.linalg.norm(current_active_points - np.array([x, y]), axis=1)
            local_mask = distances <= radius
        elif self.filter_type == "ellipse":
            width = self.ellipse_width_spin.value()
            height = self.ellipse_height_spin.value()
            # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
            normalized_x = (current_active_points[:, 0] - x) / (width/2)
            normalized_y = (current_active_points[:, 1] - y) / (height/2)
            local_mask = normalized_x**2 + normalized_y**2 <= 1
        elif self.filter_type == "ring":
            inner_radius = self.ring_inner_radius_spin.value()
            outer_radius = self.ring_outer_radius_spin.value()
            distances = np.linalg.norm(current_active_points - np.array([x, y]), axis=1)
            local_mask = (distances >= inner_radius) & (distances <= outer_radius)

        # Find local indices of points to filter out
        local_filtered_indices = np.where(local_mask)[0]

        if len(local_filtered_indices) == 0:
            print("Warning: No points in filter, cannot apply filter")
            return

        # Convert local indices to original data indices
        filtered_out_original_indices = current_active_indices[local_filtered_indices]

        # Record this filter operation
        filter_params = {
            'type': self.filter_type,
            'center': (x, y),
            'size': (
                self.x_size_spin.value(), self.y_size_spin.value()
            ) if self.filter_type == 'box' else (
                self.radius_spin.value(),
            ) if self.filter_type == 'circle' else (
                self.ellipse_width_spin.value(), self.ellipse_height_spin.value()
            ) if self.filter_type == 'ellipse' else (
                self.ring_inner_radius_spin.value(), self.ring_outer_radius_spin.value()
            ),
            'filtered_count': len(filtered_out_original_indices),
            'filtered_indices': filtered_out_original_indices.copy()
        }

        self.filter_history.append(filter_params)

        # Update index mask by setting filtered indices to False
        new_mask = self.current_indices_mask.copy()
        new_mask[filtered_out_original_indices] = False

        # Update current state (indices-based approach)
        self.current_indices_mask = new_mask
        
        # Update current points by applying new mask to original data
        self.current_points = self.original_points[self.current_indices_mask]

        # Update main view display
        self.update_main_view()

        # Save filtered points for visualization (keep for backward compatibility)
        filtered_out_points = self.original_points[filtered_out_original_indices]
        self.filtered_out_points_history.append(filtered_out_points)
        self.filtered_out_indices_history.append(filtered_out_original_indices)

        # Update filtered points display
        self.update_filtered_out_points()

        # Update history list
        self.update_history_list()

        # Update grid and scale rulers
        self.update_grid_with_scales()

        # Hide filter
        self.hide_filter()

        # Update statistics
        self.update_stats()
        
        print(f"Filter applied: {len(filtered_out_original_indices)} points filtered out using index-based approach")

    def update_main_view(self):
        """Update main view display"""
        # Remove current scatter plot
        if self.current_scatter is not None:
            self.current_scatter.remove()

        # If there are still points to display, add new scatter plot
        if len(self.current_points) > 0:
            self.current_scatter = self.ax.scatter(
                self.current_points[:, 0], self.current_points[:, 1],
                s=6, c='#1f77b4', alpha=0.7, label='Current Points'
            )
        else:
            # If no points to display, create empty scatter
            self.current_scatter = self.ax.scatter([], [], s=0, c='#1f77b4', alpha=0)

        # Update grid and scale rulers
        self.update_grid_with_scales()
        self.canvas.draw()

    def update_filtered_out_points(self):
        """Update visualization of filtered points"""
        # Remove old filtered points scatter plot
        if self.filtered_out_scatter is not None:
            self.filtered_out_scatter.remove()
            self.filtered_out_scatter = None

        # If filtered points should be displayed
        if self.show_filtered_out and len(self.filtered_out_points_history) > 0:
            all_filtered_points = np.vstack(self.filtered_out_points_history)
            if len(all_filtered_points) > 0:
                self.filtered_out_scatter = self.ax.scatter(
                    all_filtered_points[:, 0], all_filtered_points[:, 1],
                    s=4, c='#A9A9A9', alpha=0.6, label='Filtered Points'
                )
        
        self.canvas.draw()

    def toggle_filtered_out_points(self, state):
        """Toggle whether to display filtered data points"""
        self.show_filtered_out = bool(state)
        self.update_filtered_out_points()

    def update_stats(self):
        """Update data statistics"""
        total = len(self.original_points)
        remaining = len(self.current_points)
        filtered_total = total - remaining
        self.stats_label.setText(f"Total Points: {total:,}\nRemaining Points: {remaining:,} ({remaining/total:.1%})\nFiltered Points: {filtered_total:,} ({filtered_total/total:.1%})")

    def reset_view(self):
        """Reset view"""
        if len(self.current_points) > 0:
            # Set view limits based on current data
            x_min, x_max = self.current_points[:, 0].min(), self.current_points[:, 0].max()
            y_min, y_max = self.current_points[:, 1].min(), self.current_points[:, 1].max()
            
            # Add some padding
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            
            # Calculate the range to ensure equal aspect ratio
            x_range = (x_max - x_min) + 2 * x_padding
            y_range = (y_max - y_min) + 2 * y_padding
            
            # Use the larger range to maintain equal aspect ratio
            max_range = max(x_range, y_range)
            
            # Center the view
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            
            # Ensure equal aspect ratio is maintained
            self.ax.set_aspect('equal', adjustable='box')
        else:
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()

    def reset_all(self):
        """Reset all filter operations"""
        # Restore to original state
        self.current_points = self.original_points.copy()
        self.current_indices_mask = np.ones(len(self.original_points), dtype=bool)

        # Clear history
        self.filtered_out_indices_history = []
        self.filtered_out_points_history = []
        self.filter_history = []

        # Clear selected history index
        self.selected_history_index = None

        # Update view
        self.update_main_view()
        self.update_filtered_out_points()
        self.hide_history_filter()

        # Clear history list
        self.update_history_list()

        # Update statistics
        self.update_stats()

    def update_history_list(self):
        """Update history list"""
        self.history_list.clear()
        for i, filter_info in enumerate(self.filter_history):
            item_text = f"Filter {i+1}: {filter_info['type']} | "
            if filter_info['type'] == 'box':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"Size({filter_info['size'][0]:.1f}, {filter_info['size'][1]:.1f}) | "
            elif filter_info['type'] == 'circle':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"Radius({filter_info['size'][0]:.1f}) | "
            elif filter_info['type'] == 'ellipse':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"Width({filter_info['size'][0]:.1f}), Height({filter_info['size'][1]:.1f}) | "
            elif filter_info['type'] == 'ring':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}) | "
                item_text += f"Inner Radius({filter_info['size'][0]:.1f}), Outer Radius({filter_info['size'][1]:.1f}) | "
            item_text += f"Filtered Count: {filter_info['filtered_count']}"

            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, i)
            self.history_list.addItem(item)

    def on_history_item_clicked(self, item):
        """Click on history item"""
        index = item.data(QtCore.Qt.UserRole)
        if index is not None:
            # If currently selected is this item, deselect
            if self.selected_history_index == index:
                self.selected_history_index = None
                self.hide_history_filter()
                return

            # Otherwise select this item
            filter_info = self.filter_history[index]

            # Set parameters
            self.x_spin.setValue(filter_info['center'][0])
            self.y_spin.setValue(filter_info['center'][1])

            if filter_info['type'] == 'box':
                self.box_action.trigger()
                self.x_size_spin.setValue(filter_info['size'][0])
                self.y_size_spin.setValue(filter_info['size'][1])
            elif filter_info['type'] == 'circle':
                self.sphere_action.trigger()
                self.radius_spin.setValue(filter_info['size'][0])
            elif filter_info['type'] == 'ellipse':
                self.cylinder_action.trigger()
                self.ellipse_width_spin.setValue(filter_info['size'][0])
                self.ellipse_height_spin.setValue(filter_info['size'][1])
            elif filter_info['type'] == 'ring':
                self.torus_action.trigger()
                self.ring_inner_radius_spin.setValue(filter_info['size'][0])
                self.ring_outer_radius_spin.setValue(filter_info['size'][1])

            # Hide previous history filter and points
            self.hide_history_filter()

            # Record currently selected history index
            self.selected_history_index = index

            # Show selected history filter and related points
            self.show_history_filter(index)
        else:
            # Clicked on blank area, clear selection
            self.selected_history_index = None
            self.hide_history_filter()

    def show_history_filter(self, index):
        """Display filter corresponding to history record and related points"""
        # Hide current filter
        self.hide_filter()

        # Get history information
        filter_info = self.filter_history[index]

        # Create history filter
        x, y = filter_info['center']
        filter_type = filter_info['type']

        if filter_type == 'box':
            width, height = filter_info['size']
            filter_patch = Rectangle((x - width/2, y - height/2), width, height, 
                                   fill=True, alpha=0.4, color='purple')
        elif filter_type == 'circle':
            radius = filter_info['size'][0]
            filter_patch = Circle((x, y), radius, fill=True, alpha=0.4, color='orange')
        elif filter_type == 'ellipse':
            width, height = filter_info['size']
            filter_patch = Ellipse((x, y), width, height, fill=True, alpha=0.4, color='magenta')
        elif filter_type == 'ring':
            inner_radius, outer_radius = filter_info['size']
            filter_patch = Wedge((x, y), outer_radius, 0, 360, width=outer_radius-inner_radius, 
                               fill=True, alpha=0.4, color='cyan')

        # Add history filter to scene
        self.history_filter_patch = self.ax.add_patch(filter_patch)

        # Display points within history filter
        filtered_points = self.original_points[filter_info['filtered_indices']]
        if len(filtered_points) > 0:
            self.history_filtered_out_scatter = self.ax.scatter(
                filtered_points[:, 0], filtered_points[:, 1],
                s=8, c='#FF6B6B', alpha=0.8, label='History Filtered Points'
            )
        
        self.canvas.draw()

    def undo_selected_filter(self):
        """Undo selected filter"""
        if self.selected_history_index is None or len(self.filter_history) == 0:
            print("No filter record selected to undo.")
            return

        # Get index of selected item
        index_to_undo = self.selected_history_index

        # Get filter information to undo
        filter_to_undo = self.filter_history[index_to_undo]
        indices_to_restore = filter_to_undo['filtered_indices']

        # Restore filtered points to current display
        new_mask = self.current_indices_mask.copy()
        new_mask[indices_to_restore] = True

        # Apply new mask
        self.current_indices_mask = new_mask
        self.current_points = self.original_points[new_mask]

        # Remove selected record from history
        self.filter_history.pop(index_to_undo)
        self.filtered_out_points_history.pop(index_to_undo)
        self.filtered_out_indices_history.pop(index_to_undo)

        # Reset selected state
        self.selected_history_index = None
        self.hide_history_filter()

        # Update view
        self.update_main_view()
        self.update_filtered_out_points()

        # Update history list
        self.update_history_list()

        # Update statistics
        self.update_stats()

    def undo_last_filter(self):
        """Undo last filter (keep this function, but not used for button)"""
        if not self.filter_history:
            return

        # Get last filter information
        last_filter = self.filter_history.pop()
        last_filtered_indices = last_filter['filtered_indices']

        # Restore filtered points to current display
        new_mask = self.current_indices_mask.copy()
        new_mask[last_filtered_indices] = True

        # Apply new mask
        self.current_indices_mask = new_mask
        self.current_points = self.original_points[new_mask]

        # Remove last record from history
        self.filtered_out_points_history.pop()
        self.filtered_out_indices_history.pop()

        # If undoing currently selected history record, hide history filter
        if self.selected_history_index is not None and self.selected_history_index == len(self.filter_history):
            self.selected_history_index = None
            self.hide_history_filter()
        elif self.selected_history_index is not None and self.selected_history_index >= len(self.filter_history):
            self.selected_history_index = None
            self.hide_history_filter()

        # Update view
        self.update_main_view()
        self.update_filtered_out_points()

        # Update history list
        self.update_history_list()

        # Update statistics
        self.update_stats()

    def save_current_indices(self):
        """Save current indices to file"""
        try:
            # Get current indices (where mask is True)
            current_indices = np.where(self.current_indices_mask)[0]
            
            # Open file dialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Save Current Indices", 
                "current_indices_2D.pkl",
                "Pickle Files (*.pkl);;NumPy Files (*.npy);;All Files (*)"
            )
            
            if filename:
                if filename.endswith('.npy'):
                    np.save(filename, current_indices)
                else:
                    # Default to pickle format
                    save_data = {
                        'indices': current_indices,
                        'total_original_points': len(self.original_points),
                        'current_points_count': len(current_indices),
                        'filter_history': self.filter_history
                    }
                    with open(filename, 'wb') as f:
                        pickle.dump(save_data, f)
                
                print(f"Successfully saved {len(current_indices)} indices to {filename}")
                QtWidgets.QMessageBox.information(
                    self, 
                    "Save Successful", 
                    f"Saved {len(current_indices):,} indices to:\n{filename}"
                )
        except Exception as e:
            print(f"Error saving indices: {e}")
            QtWidgets.QMessageBox.critical(
                self, 
                "Save Error", 
                f"Failed to save indices:\n{str(e)}"
            )

    def save_current_data(self):
        """Save current data (filtered points) to file"""
        try:
            # Get current data points
            current_data = self.current_points
            
            # Open file dialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Save Current Data", 
                "current_data_2D.pkl",
                "Pickle Files (*.pkl);;CSV Files (*.csv);;NumPy Files (*.npy);;All Files (*)"
            )
            
            if filename:
                if filename.endswith('.csv'):
                    # Save as CSV with headers
                    header = "X,Y"
                    np.savetxt(filename, current_data, delimiter=',', header=header, comments='')
                elif filename.endswith('.npy'):
                    np.save(filename, current_data)
                else:
                    # Default to pickle format
                    save_data = {
                        'points': current_data,
                        'indices': np.where(self.current_indices_mask)[0],
                        'total_original_points': len(self.original_points),
                        'current_points_count': len(current_data),
                        'filter_history': self.filter_history
                    }
                    with open(filename, 'wb') as f:
                        pickle.dump(save_data, f)
                
                print(f"Successfully saved {len(current_data)} data points to {filename}")
                QtWidgets.QMessageBox.information(
                    self, 
                    "Save Successful", 
                    f"Saved {len(current_data):,} data points to:\n{filename}"
                )
        except Exception as e:
            print(f"Error saving data: {e}")
            QtWidgets.QMessageBox.critical(
                self, 
                "Save Error", 
                f"Failed to save data:\n{str(e)}"
            )

def downsample_points(points, max_points=500000):
    """If too many points, perform random downsampling"""
    if len(points) > max_points:
        print(f"Too many data points ({len(points):,}), downsampling to {max_points} points")
        indices = np.random.choice(len(points), max_points, replace=False)
        return points[indices]
    return points

if __name__ == "__main__":
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Generate or load data
    np.random.seed(42)
    x = np.random.normal(0, 50, 100000)
    y = np.random.normal(0, 50, 100000)
    point_data = np.column_stack((x, y))

    # Create visualization window
    window = TwoDVisualizer(point_data=point_data)
    window.setWindowTitle("Scientific 2D Scatter Plot Visualization System")
    window.resize(1400, 900)
    window.show()

    # Start application
    sys.exit(app.exec_())
