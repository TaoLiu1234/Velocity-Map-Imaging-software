import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
import sys
import warnings
import pickle

# Suppress the SIP deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

class ThreeDVisualizer(QtWidgets.QMainWindow):
    def __init__(self, point_data=None):
        super().__init__()

        # Create main frame
        self.frame = QtWidgets.QFrame()
        self.h_layout = QtWidgets.QHBoxLayout()  # Horizontal layout: 3D view + right control panel

        # Create PyVista render window
        self.plotter = QtInteractor(self.frame)
        self.h_layout.addWidget(self.plotter.interactor, 4)  # Takes 4/5 width

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

        # Current filter actor
        self.current_filter_actor = None
        self.filtered_out_actors = []  # All filtered point actors
        self.history_filter_actor = None  # History filter actor
        self.history_filtered_out_actor = None  # History filtered point actor

        # Currently displayed actor
        self.current_points_actor = None

        # Grid and scale lines
        self.grid_actors = []
        self.range_text_actor = None  # Text for displaying effective data range

        # History
        self.filter_history = []  # Filter history
        self.selected_history_index = None  # Currently selected history index

        # Set up initial view
        self.setup_scene()
        
        # Add keyboard event filter to capture space key on the plotter interactor
        self.plotter.interactor.installEventFilter(self)

        # Create floating filter toolbar
        self.create_filter_toolbar()

        # Create parameter settings panel (dock widget)
        self.create_parameter_dock()

    def eventFilter(self, obj, event):
        """Handle keyboard events"""
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Space:
                self.reset_view()
                return True  # Consume event to prevent further propagation
        return super().eventFilter(obj, event)

    def create_filter_toolbar(self):
        """Create filter selection toolbar floating in the top-right corner of 3D view"""
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

        # Create actions (buttons)
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

        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setRange(-1000, 1000)
        self.z_spin.setValue(0)
        self.z_spin.setSingleStep(1)

        # Box parameters
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

        # Sphere parameters
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 500)
        self.radius_spin.setValue(10)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setVisible(False)

        # Cylinder parameters
        self.cylinder_radius_spin = QtWidgets.QDoubleSpinBox()
        self.cylinder_radius_spin.setRange(0.1, 500)
        self.cylinder_radius_spin.setValue(10)
        self.cylinder_radius_spin.setSingleStep(0.5)
        self.cylinder_radius_spin.setVisible(False)
        
        self.cylinder_height_spin = QtWidgets.QDoubleSpinBox()
        self.cylinder_height_spin.setRange(0.1, 1000)
        self.cylinder_height_spin.setValue(20)
        self.cylinder_height_spin.setSingleStep(0.5)
        self.cylinder_height_spin.setVisible(False)

        # Torus parameters
        self.torus_inner_radius_spin = QtWidgets.QDoubleSpinBox()
        self.torus_inner_radius_spin.setRange(0.1, 500)
        self.torus_inner_radius_spin.setValue(5)
        self.torus_inner_radius_spin.setSingleStep(0.5)
        self.torus_inner_radius_spin.setVisible(False)
        
        self.torus_outer_radius_spin = QtWidgets.QDoubleSpinBox()
        self.torus_outer_radius_spin.setRange(0.1, 500)
        self.torus_outer_radius_spin.setValue(10)
        self.torus_outer_radius_spin.setSingleStep(0.5)
        self.torus_outer_radius_spin.setVisible(False)
        
        self.torus_height_spin = QtWidgets.QDoubleSpinBox()
        self.torus_height_spin.setRange(0.1, 1000)
        self.torus_height_spin.setValue(20)
        self.torus_height_spin.setSingleStep(0.5)
        self.torus_height_spin.setVisible(False)

        # Add parameters to layout
        param_layout.addRow("X Position:", self.x_spin)
        param_layout.addRow("Y Position:", self.y_spin)
        param_layout.addRow("Z Position:", self.z_spin)
        param_layout.addRow("X Length:", self.x_size_spin)
        param_layout.addRow("Y Width:", self.y_size_spin)
        param_layout.addRow("Z Height:", self.z_size_spin)
        param_layout.addRow("Radius:", self.radius_spin)
        param_layout.addRow("Cylinder Radius:", self.cylinder_radius_spin)
        param_layout.addRow("Cylinder Height:", self.cylinder_height_spin)
        param_layout.addRow("Torus Inner Radius:", self.torus_inner_radius_spin)
        param_layout.addRow("Torus Outer Radius:", self.torus_outer_radius_spin)
        param_layout.addRow("Torus Height:", self.torus_height_spin)
        
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
        elif filter_type == "sphere":
            self.sphere_action.setChecked(True)
        elif filter_type == "cylinder":
            self.cylinder_action.setChecked(True)
        elif filter_type == "torus":
            self.torus_action.setChecked(True)
        
        # Show/hide corresponding parameter controls
        self.x_size_spin.setVisible(filter_type == "box")
        self.y_size_spin.setVisible(filter_type == "box")
        self.z_size_spin.setVisible(filter_type == "box")
        self.radius_spin.setVisible(filter_type == "sphere")
        self.cylinder_radius_spin.setVisible(filter_type == "cylinder")
        self.cylinder_height_spin.setVisible(filter_type == "cylinder")
        self.torus_inner_radius_spin.setVisible(filter_type == "torus")
        self.torus_outer_radius_spin.setVisible(filter_type == "torus")
        self.torus_height_spin.setVisible(filter_type == "torus")

        self.param_dock.show()

    def setup_scene(self):
        """Set up initial scene"""
        # If no data provided, generate sample data (100,000 points)
        if self.point_data is None:
            print("Generating sample dataset (100,000 points)...")
            np.random.seed(42)
            x = np.random.normal(0, 50, 100000)
            y = np.random.normal(0, 50, 100000)
            z = np.random.normal(0, 50, 100000)
            self.point_data = np.column_stack((x, y, z))

        # Save original data and index mask
        self.original_points = self.point_data.copy()
        self.current_points = self.point_data.copy()
        self.original_indices_mask = np.ones(len(self.point_data), dtype=bool)
        self.current_indices_mask = np.ones(len(self.point_data), dtype=bool)

        # Create point cloud
        points_polydata = pv.PolyData(self.current_points)

        # Add point cloud to scene (using point sprites for performance)
        self.current_points_actor = self.plotter.add_mesh(
            points_polydata,
            render_points_as_spheres=False,
            point_size=2.5,
            color='#1f77b4',
            opacity=0.7,
            name='current_points'
        )

        # Add coordinate axes and scales
        self.plotter.add_axes(line_width=3, color='black', labels_off=False,
                             x_color='red', y_color='green', z_color='blue')

        # Add grid and scale rulers
        self.update_grid_with_scales()

        # Set initial camera position
        self.plotter.camera_position = 'iso'
        self.plotter.add_text("Scientific 3D Scatter Plot Visualization System", font_size=12)

        # Update statistics
        self.update_stats()

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
        points_polydata = pv.PolyData(self.current_points)
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

    def toggle_filter_display(self, state):
        """Toggle filter display"""
        if state:
            if self.current_filter_actor is None and self.apply_filter_btn.isEnabled():
                self.create_filter()
        else:
            self.hide_filter()

    def hide_filter(self):
        """Hide current filter"""
        if self.current_filter_actor is not None:
            self.plotter.remove_actor(self.current_filter_actor)
            self.current_filter_actor = None
        self.apply_filter_btn.setEnabled(False)

    def hide_history_filter(self):
        """Hide history filter and related points"""
        if self.history_filter_actor is not None:
            self.plotter.remove_actor(self.history_filter_actor)
            self.history_filter_actor = None
        if self.history_filtered_out_actor is not None:
            self.plotter.remove_actor(self.history_filtered_out_actor)
            self.history_filtered_out_actor = None

    def create_filter(self):
        """Create filter geometry"""
        self.hide_filter()

        x, y, z = self.x_spin.value(), self.y_spin.value(), self.z_spin.value()

        filter_mesh = None
        color = 'gray'

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
            filter_mesh = pv.Cylinder(center=(x, y, z), radius=radius, height=height, direction=(0, 0, 1), resolution=32)
            color = 'darkblue'
        elif self.filter_type == "torus":
            inner_radius = self.torus_inner_radius_spin.value()
            outer_radius = self.torus_outer_radius_spin.value()
            height = self.torus_height_spin.value()
            filter_mesh = self.create_polygonal_torus(x, y, z, inner_radius, outer_radius, height)
            color = 'darkorange'

        if filter_mesh is not None:
            # Add filter to scene and save actor reference
            self.current_filter_actor = self.plotter.add_mesh(
                filter_mesh,
                color=color,
                opacity=0.3,
                name='current_filter'
            )
        else:
            print("Error: Failed to create filter mesh")
            return

        self.apply_filter_btn.setEnabled(True)

        if self.show_filter_checkbox.isChecked():
            self.show_filter_volume = True
        else:
            self.hide_filter()

    def create_polygonal_torus(self, x, y, z, inner_radius, outer_radius, height):
        """Create a torus using PyVista's ParametricTorus for robustness"""
        try:
            major_radius = (inner_radius + outer_radius) / 2
            minor_radius = (outer_radius - inner_radius) / 2
            
            filter_mesh = pv.ParametricTorus(
                ringradius=major_radius,
                crosssectionradius=minor_radius,
                u_res=32,
                v_res=16
            )
            
            # Scale torus to match desired height
            current_height = 2 * minor_radius
            if current_height > 0:
                scale_factor = height / current_height
                filter_mesh.scale((1, 1, scale_factor), inplace=True)
            
            filter_mesh.translate((x, y, z), inplace=True)
            return filter_mesh
            
        except Exception as e:
            print(f"Error creating parametric torus: {e}")
            return self.create_torus_fallback(x, y, z, inner_radius, outer_radius, height)
    
    def create_torus_fallback(self, x, y, z, inner_radius, outer_radius, height):
        """Fallback method using boolean operations to create a torus"""
        try:
            outer_cylinder = pv.Cylinder(
                center=(x, y, z), 
                radius=outer_radius, 
                height=height,
                direction=(0, 0, 1), 
                resolution=64
            )
            
            inner_cylinder = pv.Cylinder(
                center=(x, y, z), 
                radius=inner_radius, 
                height=height,
                direction=(0, 0, 1), 
                resolution=64
            )
            
            outer_cylinder = outer_cylinder.triangulate()
            inner_cylinder = inner_cylinder.triangulate()
            
            torus_mesh = outer_cylinder.boolean_difference(inner_cylinder)
            
            if torus_mesh.n_points > 0 and torus_mesh.n_cells > 0:
                return torus_mesh
            else:
                print("Boolean operation failed, trying alternative approach")
                return self.create_torus_alternative(x, y, z, inner_radius, outer_radius, height)
                
        except Exception as e:
            print(f"Boolean operation fallback failed: {e}")
            return self.create_torus_alternative(x, y, z, inner_radius, outer_radius, height)
    
    def create_torus_alternative(self, x, y, z, inner_radius, outer_radius, height):
        """Alternative method using multiple small cylinders to approximate a torus"""
        try:
            major_radius = (inner_radius + outer_radius) / 2
            minor_radius = (outer_radius - inner_radius) / 2
            
            theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
            meshes = []
            
            for t in theta:
                cx = x + major_radius * np.cos(t)
                cy = y + major_radius * np.sin(t)
                
                small_cyl = pv.Cylinder(
                    center=(cx, cy, z),
                    radius=minor_radius,
                    height=height,
                    direction=(0, 0, 1),
                    resolution=16
                )
                meshes.append(small_cyl)
            
            if meshes:
                result_mesh = meshes[0]
                for mesh in meshes[1:]:
                    result_mesh = result_mesh + mesh
                
                if result_mesh.n_points > 0 and result_mesh.n_cells > 0:
                    return result_mesh
            
            print("All torus creation methods failed, using cylinder fallback")
            return pv.Cylinder(
                center=(x, y, z), 
                radius=outer_radius, 
                height=height,
                direction=(0, 0, 1), 
                resolution=32
            )
            
        except Exception as e:
            print(f"Alternative torus creation failed: {e}")
            return pv.Cylinder(
                center=(x, y, z), 
                radius=outer_radius, 
                height=height,
                direction=(0, 0, 1), 
                resolution=32
            )

    def apply_filter(self):
        """Apply filter based on indices - works on index mask rather than modifying data directly"""
        if self.current_filter_actor is None:
            return

        # Get currently active indices (where mask is True)
        current_active_indices = np.where(self.current_indices_mask)[0]
        
        # Get points corresponding to currently active indices
        current_active_points = self.original_points[current_active_indices]
        
        x, y, z = self.x_spin.value(), self.y_spin.value(), self.z_spin.value()

        # Calculate mask based on filter type for currently active points
        if self.filter_type == "box":
            x_min, x_max = x - self.x_size_spin.value()/2, x + self.x_size_spin.value()/2
            y_min, y_max = y - self.y_size_spin.value()/2, y + self.y_size_spin.value()/2
            z_min, z_max = z - self.z_size_spin.value()/2, z + self.z_size_spin.value()/2
            local_mask = (
                (current_active_points[:, 0] >= x_min) & (current_active_points[:, 0] <= x_max) &
                (current_active_points[:, 1] >= y_min) & (current_active_points[:, 1] <= y_max) &
                (current_active_points[:, 2] >= z_min) & (current_active_points[:, 2] <= z_max)
            )
        elif self.filter_type == "sphere":
            radius = self.radius_spin.value()
            distances = np.linalg.norm(current_active_points - np.array([x, y, z]), axis=1)
            local_mask = distances <= radius
        elif self.filter_type == "cylinder":
            radius = self.cylinder_radius_spin.value()
            height = self.cylinder_height_spin.value()
            xy_distances = np.linalg.norm(current_active_points[:, :2] - np.array([x, y]), axis=1)
            z_distances = np.abs(current_active_points[:, 2] - z)
            local_mask = (xy_distances <= radius) & (z_distances <= height / 2)
        elif self.filter_type == "torus":
            inner_radius = self.torus_inner_radius_spin.value()
            outer_radius = self.torus_outer_radius_spin.value()
            height = self.torus_height_spin.value()
            
            major_radius = (inner_radius + outer_radius) / 2
            minor_radius = (outer_radius - inner_radius) / 2
            
            rel_points = current_active_points - np.array([x, y, z])
            angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
            
            local_x = np.cos(angles)
            local_y = np.sin(angles)
            
            local_pos_x = rel_points[:, 0] * local_x + rel_points[:, 1] * local_y - major_radius
            local_pos_y = -rel_points[:, 0] * local_y + rel_points[:, 1] * local_x
            
            square_condition = (np.abs(local_pos_x) <= minor_radius) & (np.abs(local_pos_y) <= minor_radius)
            height_condition = np.abs(rel_points[:, 2]) <= height / 2
            
            local_mask = square_condition & height_condition

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
            'center': (x, y, z),
            'size': (
                self.x_size_spin.value(), self.y_size_spin.value(), self.z_size_spin.value()
            ) if self.filter_type == 'box' else (
                self.radius_spin.value(),
            ) if self.filter_type == 'sphere' else (
                self.cylinder_radius_spin.value(), self.cylinder_height_spin.value()
            ) if self.filter_type == 'cylinder' else (
                self.torus_inner_radius_spin.value(), self.torus_outer_radius_spin.value(), self.torus_height_spin.value()
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
        # Remove current point cloud
        if self.current_points_actor is not None:
            self.plotter.remove_actor(self.current_points_actor)

        # If there are still points to display, add new point cloud
        if len(self.current_points) > 0:
            points_polydata = pv.PolyData(self.current_points)
            self.current_points_actor = self.plotter.add_mesh(
                points_polydata,
                render_points_as_spheres=False,
                point_size=2.5,
                color='#1f77b4',
                opacity=0.7,
                name='current_points'
            )
        else:
            # If no points to display, create empty polydata
            empty_points = pv.PolyData(np.array([[0, 0, 0]]))
            self.current_points_actor = self.plotter.add_mesh(
                empty_points,
                render_points_as_spheres=False,
                point_size=0,
                color='#1f77b4',
                opacity=0,
                name='current_points'
            )

        # Update grid and scale rulers
        self.update_grid_with_scales()

    def update_filtered_out_points(self):
        """Update visualization of filtered points"""
        # Remove all old filtered point actors
        for actor in self.filtered_out_actors:
            self.plotter.remove_actor(actor)
        self.filtered_out_actors = []

        # If filtered points should be displayed
        if self.show_filtered_out:
            for i, filtered_out_points in enumerate(self.filtered_out_points_history):
                if len(filtered_out_points) > 0:
                    filtered_out_polydata = pv.PolyData(filtered_out_points)
                    actor = self.plotter.add_mesh(
                        filtered_out_polydata,
                        render_points_as_spheres=False,
                        point_size=2,
                        color='#A9A9A9',
                        opacity=0.6,
                        name=f'filtered_out_points_{i}'
                    )
                    self.filtered_out_actors.append(actor)

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
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

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
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"Size({filter_info['size'][0]:.1f}, {filter_info['size'][1]:.1f}, {filter_info['size'][2]:.1f}) | "
            elif filter_info['type'] == 'sphere':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"Radius({filter_info['size'][0]:.1f}) | "
            elif filter_info['type'] == 'cylinder':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"Radius({filter_info['size'][0]:.1f}), Height({filter_info['size'][1]:.1f}) | "
            elif filter_info['type'] == 'torus':
                item_text += f"Position({filter_info['center'][0]:.1f}, {filter_info['center'][1]:.1f}, {filter_info['center'][2]:.1f}) | "
                item_text += f"Inner Radius({filter_info['size'][0]:.1f}), Outer Radius({filter_info['size'][1]:.1f}), Height({filter_info['size'][2]:.1f}) | "
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
            self.z_spin.setValue(filter_info['center'][2])

            if filter_info['type'] == 'box':
                self.box_action.trigger()
                self.x_size_spin.setValue(filter_info['size'][0])
                self.y_size_spin.setValue(filter_info['size'][1])
                self.z_size_spin.setValue(filter_info['size'][2])
            elif filter_info['type'] == 'sphere':
                self.sphere_action.trigger()
                self.radius_spin.setValue(filter_info['size'][0])
            elif filter_info['type'] == 'cylinder':
                self.cylinder_action.trigger()
                self.cylinder_radius_spin.setValue(filter_info['size'][0])
                self.cylinder_height_spin.setValue(filter_info['size'][1])
            elif filter_info['type'] == 'torus':
                self.torus_action.trigger()
                self.torus_inner_radius_spin.setValue(filter_info['size'][0])
                self.torus_outer_radius_spin.setValue(filter_info['size'][1])
                self.torus_height_spin.setValue(filter_info['size'][2])

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
        elif filter_type == 'torus':
            inner_radius = filter_info['size'][0]
            outer_radius = filter_info['size'][1]
            height = filter_info['size'][2]
            filter_mesh = self.create_polygonal_torus(x, y, z, inner_radius, outer_radius, height)
            color = 'cyan'

        # Add history filter to scene
        self.history_filter_actor = self.plotter.add_mesh(
            filter_mesh,
            color=color,
            opacity=0.4,
            name=f'history_filter_{index}'
        )

        # Display points within history filter
        filtered_points = self.original_points[filter_info['filtered_indices']]
        if len(filtered_points) > 0:
            filtered_polydata = pv.PolyData(filtered_points)
            self.history_filtered_out_actor = self.plotter.add_mesh(
                filtered_polydata,
                render_points_as_spheres=False,
                point_size=3,
                color='#FF6B6B',
                opacity=0.8,
                name=f'history_filtered_points_{index}'
            )

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
                "current_indices.pkl",
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
                "current_data.pkl",
                "Pickle Files (*.pkl);;CSV Files (*.csv);;NumPy Files (*.npy);;All Files (*)"
            )
            
            if filename:
                if filename.endswith('.csv'):
                    # Save as CSV with headers
                    header = "X,Y,Z"
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
    z = np.random.normal(0, 50, 100000)
    point_data = np.column_stack((x, y, z))

    # Create visualization window
    window = ThreeDVisualizer(point_data=point_data)
    window.setWindowTitle("Scientific 3D Scatter Plot Visualization System")
    window.resize(1400, 900)
    window.show()

    # Start application
    sys.exit(app.exec_())
