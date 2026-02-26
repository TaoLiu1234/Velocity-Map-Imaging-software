#!/usr/bin/env python3
from __future__ import annotations

"""VMI_workflow: interactive GUI for trigger-based electron/ion event filtering and reconstruction.

Recommended operation order in this GUI (the same order users should click):
1. Load files to cache and run first plot:
   - Drop Trigger/Electron/Ion files.
   - Click `Load to Cache` -> `Process and Plot`.
   - Main functions: `load_cache()`, `process_and_plot()`.
2. Coarse tune ion TOF histogram range:
   - Set `Hist X ROI min/max` and click `Update Hist ROI`.
   - This changes histogram rendering window (coarse view).
   - Main function: `apply_ion_hist_x_roi_from_inputs()`.
3. Fine tune ion TOF selection:
   - Drag on ion histogram to select the exact TOF peak region.
   - This fine ROI controls downstream point selection.
   - Main functions: `_on_ion_span_selected()` -> `_apply_ion_selection_range()`.
4. Filter and align ion/electron scatter:
   - Optional: enable ion rectangle filter and ion rotation/alignment.
   - Set electron ring center/inner/outer radii.
5. Build centered projection inputs:
   - Click `Apply Ring Selection and Center`.
   - App intersects current filters, recenters selected electrons, estimates outer-ring
     noise, then generates denoised binned image.
   - Main function: `apply_circle_selection()`.
6. Set reconstruction parameters:
   - Edit rBasex/backward settings in control panel.
7. Run reconstruction:
   - Click `Start Reconstruction`.
   - Main function: `run_reconstruction_now()`.

Code split (3 main modules):
- VMI_workflow.py: Qt/Matplotlib GUI and user interaction.
- VMI_workflow_core.py: data filtering, center helpers, and projection binning.
- VMI_workflow_reconstruction.py: rBasex/backward reconstruction utilities.
"""

import contextlib
import os
import sys

# Keep matplotlib on the same Qt binding as the GUI toolkit.
os.environ["QT_API"] = "pyside6"

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from VMI_workflow_core import (
    CacheData,
    build_denoised_centered_histogram,
    density_counts_from_bins,
    edge_circle_center,
    ensure_2d,
    geometric_median,
    select_increment_pairs,
)
from VMI_workflow_reconstruction import (
    format_peak_text,
    run_reconstructions_from_centered_data,
)

matplotlib.use("QtAgg")

ROLE_LABELS = {
    "trigger": "Trigger",
    "electron": "Electron",
    "ion": "Ion",
}

MAX_SCATTER_POINTS = 80_000
OVERLAY_EDIT_DEBOUNCE_MS = 70
DRAG_PREVIEW_INTERVAL_MS = 16
ION_FINE_ROI_DEBOUNCE_MS = 130


class FileDropFrame(QFrame):
    """Small reusable widget: drag-drop (or browse) one file path."""

    file_dropped = Signal(str)

    def __init__(self, title: str, placeholder: str, parent: QWidget | None = None):
        """Build a labeled frame with path display and browse button."""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600;")

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(placeholder)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_file)

        layout.addWidget(title_label)
        layout.addWidget(self.path_edit)
        layout.addWidget(browse_btn)

    def set_path(self, file_path: str) -> None:
        """Display the chosen file path."""
        self.path_edit.setText(file_path)

    def clear_path(self) -> None:
        """Clear displayed path."""
        self.path_edit.clear()

    def dragEnterEvent(self, event):  # noqa: N802
        """Accept drag if MIME data contains local URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802
        """Emit the first local file dropped onto this frame."""
        urls = event.mimeData().urls()
        for url in urls:
            if url.isLocalFile():
                self.file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()

    def _browse_file(self) -> None:
        """Open file dialog and emit selected path."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose data file", "", "Data files (*.dat);;All files (*.*)"
        )
        if file_path:
            self.file_dropped.emit(file_path)


class MainWindow(QMainWindow):
    """Main application window.

    Design notes for students:
    - UI widgets are created in `__init__`.
    - Data filtering/reconstruction logic is implemented as methods in this class.
    - Matplotlib axes are reused; plots are refreshed when controls change.
    """

    def __init__(self):
        """Create UI, initialize state, and wire all callbacks."""
        super().__init__()
        self.setWindowTitle("VMI_workflow")
        self.resize(1380, 980)

        # ------------------------------
        # Runtime data state
        # ------------------------------
        self.file_paths = {"trigger": "", "electron": "", "ion": ""}
        self.cache: CacheData | None = None

        self.matched_electron = np.empty((0, 3), dtype=np.float64)
        self.matched_ion = np.empty((0, 3), dtype=np.float64)
        self.ion_range: tuple[float, float] | None = None
        self.ion_hist_x_roi: tuple[float, float] | None = None
        self.current_hist_bins = 120

        self.ring_inner_selected_electron = np.empty((0, 3), dtype=np.float64)
        self.ring_outer_noise_electron = np.empty((0, 3), dtype=np.float64)
        self.ion_filter_selected_electron = np.empty((0, 3), dtype=np.float64)
        self.ion_filter_selected_ion = np.empty((0, 3), dtype=np.float64)
        self.intersection_indices = np.empty(0, dtype=np.int64)
        self.circle_centered_electron = np.empty((0, 3), dtype=np.float64)
        self.noise_ring_centered_electron = np.empty((0, 3), dtype=np.float64)
        self.circle_centroid: tuple[float, float] | None = None
        self.center_residual: tuple[float, float] | None = None
        self.centered_hist_data: dict | None = None
        self.noise_removed_total = 0.0
        self.rbasex_recon_result: dict | None = None
        self.backward_recon_result: dict | None = None

        # ------------------------------
        # Overlay/interaction state
        # ------------------------------
        self.inner_ring_patch: Circle | None = None
        self.outer_ring_patch: Circle | None = None
        self.circle_center_marker = None
        self.ion_filter_patch: Rectangle | None = None
        self.ion_filter_center_marker = None
        self.dragging_circle = False
        self.dragging_ion_filter = False
        self.drag_offset_x = 0.0
        self.drag_offset_y = 0.0
        self.ion_drag_offset_x = 0.0
        self.ion_drag_offset_y = 0.0
        self.pending_drag_center: tuple[float, float] | None = None
        self.pending_drag_ion_center: tuple[float, float] | None = None
        self.preview_circle_center: tuple[float, float] | None = None
        self.preview_ion_center: tuple[float, float] | None = None
        self.pending_ion_span_range: tuple[float, float] | None = None
        self.bg_scatter_e = None
        self.bg_scatter_i = None
        self.ion_rotation_matrix = np.eye(2, dtype=np.float64)
        self.ion_rotation_angle_deg = 0.0
        self.ion_auto_angle_deg = 0.0
        self.ion_user_rotation_deg = 0.0
        self.ion_selector = None
        self.ion_main_axis_line = None
        self.ion_main_axis_marker = None
        self.ion_main_axis_angle_deg = None
        self.ion_selection_patch = None

        # ------------------------------
        # Timers for smooth interaction (debounce/throttle)
        # ------------------------------
        self.circle_preview_timer = QTimer(self)
        self.circle_preview_timer.setInterval(OVERLAY_EDIT_DEBOUNCE_MS)
        self.circle_preview_timer.setSingleShot(True)
        self.circle_preview_timer.timeout.connect(self._update_circle_overlay_only)
        self.ion_overlay_preview_timer = QTimer(self)
        self.ion_overlay_preview_timer.setInterval(OVERLAY_EDIT_DEBOUNCE_MS)
        self.ion_overlay_preview_timer.setSingleShot(True)
        self.ion_overlay_preview_timer.timeout.connect(self._update_ion_overlay_only)
        self.drag_preview_timer = QTimer(self)
        self.drag_preview_timer.setInterval(DRAG_PREVIEW_INTERVAL_MS)
        self.drag_preview_timer.setSingleShot(True)
        self.drag_preview_timer.timeout.connect(self._flush_drag_preview)
        self.ion_drag_preview_timer = QTimer(self)
        self.ion_drag_preview_timer.setInterval(DRAG_PREVIEW_INTERVAL_MS)
        self.ion_drag_preview_timer.setSingleShot(True)
        self.ion_drag_preview_timer.timeout.connect(self._flush_ion_drag_preview)
        self.ion_span_apply_timer = QTimer(self)
        self.ion_span_apply_timer.setInterval(ION_FINE_ROI_DEBOUNCE_MS)
        self.ion_span_apply_timer.setSingleShot(True)
        self.ion_span_apply_timer.timeout.connect(self._flush_pending_ion_span_selection)

        # ------------------------------
        # Main split layout: controls (top) + plots (bottom)
        # ------------------------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        self.setCentralWidget(central)

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(12)
        root.addWidget(self.main_splitter, stretch=1)

        top_panel = QWidget()
        top_panel.setObjectName("TopPanel")
        self.top_layout = QVBoxLayout(top_panel)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(6)

        bottom_panel = QWidget()
        bottom_panel.setObjectName("BottomPanel")
        self.bottom_layout = QVBoxLayout(bottom_panel)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(6)

        self.setStyleSheet(
            "QWidget#TopPanel, QWidget#BottomPanel { border: 1px solid #9a9a9a; border-radius: 3px; }"
            "QSplitter::handle:vertical { background: #707070; border-top: 1px solid #555; border-bottom: 1px solid #555; }"
        )

        self.main_splitter.addWidget(top_panel)
        self.main_splitter.addWidget(bottom_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([300, 880])

        quick_guide = QLabel(
            "Workflow: 1) Load to Cache + Process and Plot  2) Coarse Ion Hist X ROI  3) Fine TOF Span ROI  "
            "4) Ion/Electron filter+align  5) Apply Ring Selection and Center  6) Set reconstruction params  "
            "7) Start Reconstruction"
        )
        quick_guide.setWordWrap(True)
        self.top_layout.addWidget(quick_guide)
        self.top_layout.addWidget(QLabel("Drag the gray splitter bar to resize Control Panel and Plot Area."))

        # ------------------------------
        # File input area
        # ------------------------------
        file_grid = QGridLayout()
        self.drop_frames = {}
        specs = [
            ("trigger", "Trigger File", "Drop file here"),
            ("electron", "Electron File", "Drop file here"),
            ("ion", "Ion File", "Drop file here"),
        ]
        for col, (role, title, placeholder) in enumerate(specs):
            frame = FileDropFrame(title, placeholder)
            frame.file_dropped.connect(lambda path, r=role: self._set_file_path(r, path))
            self.drop_frames[role] = frame
            file_grid.addWidget(frame, 0, col)
        self.top_layout.addLayout(file_grid)

        action_row = QHBoxLayout()
        self.load_btn = QPushButton("Load to Cache")
        self.load_btn.clicked.connect(self.load_cache)
        self.clear_btn = QPushButton("Clear Cache")
        self.clear_btn.clicked.connect(self.clear_cache)
        self.process_btn = QPushButton("Process and Plot")
        self.process_btn.clicked.connect(self.process_and_plot)
        action_row.addWidget(self.load_btn)
        action_row.addWidget(self.clear_btn)
        action_row.addWidget(self.process_btn)
        action_row.addStretch(1)
        self.top_layout.addLayout(action_row)

        # ------------------------------
        # Control panel sections
        # ------------------------------
        control_group = QGroupBox("Controls")
        control_grid = QGridLayout(control_group)
        control_grid.setHorizontalSpacing(12)
        control_grid.setVerticalSpacing(7)

        # Step 2 + Step 3 controls (coarse/fine TOF selection)
        ion_hist_group = QGroupBox("Ion Histogram")
        ion_hist_grid = QGridLayout(ion_hist_group)
        ion_hist_grid.setHorizontalSpacing(10)
        ion_hist_grid.addWidget(QLabel("Histogram bins"), 0, 0)
        self.bins_edit = QLineEdit("120")
        self.bins_edit.setMaximumWidth(100)
        ion_hist_grid.addWidget(self.bins_edit, 0, 1)
        self.clear_ion_sel_btn = QPushButton("Clear Fine ROI")
        self.clear_ion_sel_btn.clicked.connect(self.clear_ion_selection)
        ion_hist_grid.addWidget(self.clear_ion_sel_btn, 0, 2)
        self.clear_ion_hist_roi_btn = QPushButton("Reset Hist X ROI")
        self.clear_ion_hist_roi_btn.clicked.connect(self.clear_ion_hist_x_roi)
        ion_hist_grid.addWidget(self.clear_ion_hist_roi_btn, 0, 3)
        self.selection_label = QLabel("Ion t selection (fine): all | Histogram X ROI (coarse): full")
        ion_hist_grid.addWidget(self.selection_label, 1, 0, 1, 4)
        ion_hist_grid.addWidget(QLabel("Hist X ROI min"), 2, 0)
        self.ion_hist_xmin_edit = QLineEdit("")
        self.ion_hist_xmin_edit.setPlaceholderText("auto")
        self.ion_hist_xmin_edit.setMaximumWidth(110)
        self.ion_hist_xmin_edit.returnPressed.connect(self.apply_ion_hist_x_roi_from_inputs)
        ion_hist_grid.addWidget(self.ion_hist_xmin_edit, 2, 1)
        ion_hist_grid.addWidget(QLabel("Hist X ROI max"), 2, 2)
        self.ion_hist_xmax_edit = QLineEdit("")
        self.ion_hist_xmax_edit.setPlaceholderText("auto")
        self.ion_hist_xmax_edit.setMaximumWidth(110)
        self.ion_hist_xmax_edit.returnPressed.connect(self.apply_ion_hist_x_roi_from_inputs)
        ion_hist_grid.addWidget(self.ion_hist_xmax_edit, 2, 3)
        self.apply_ion_hist_roi_btn = QPushButton("Update Hist ROI")
        self.apply_ion_hist_roi_btn.clicked.connect(self.apply_ion_hist_x_roi_from_inputs)
        ion_hist_grid.addWidget(self.apply_ion_hist_roi_btn, 3, 0, 1, 2)
        control_grid.addWidget(ion_hist_group, 0, 0, 1, 12)

        # Step 4 + Step 5 controls (electron ring filter and centering)
        e_scatter_group = QGroupBox("Electron Scatter Plot")
        e_scatter_grid = QGridLayout(e_scatter_group)
        e_scatter_grid.setHorizontalSpacing(10)
        e_scatter_grid.addWidget(QLabel("Center estimator"), 0, 0)
        self.center_mode_combo = QComboBox()
        self.center_mode_combo.addItem("Edge circle fit (recommended)", "edge_fit")
        self.center_mode_combo.addItem("Centroid (mean)", "centroid")
        self.center_mode_combo.addItem("Geometric median", "geo_median")
        e_scatter_grid.addWidget(self.center_mode_combo, 0, 1)
        self.auto_recenter_checkbox = QCheckBox("Auto recenter ring to estimated center on Apply")
        self.auto_recenter_checkbox.setChecked(True)
        e_scatter_grid.addWidget(self.auto_recenter_checkbox, 0, 2, 1, 3)
        e_scatter_grid.addWidget(QLabel("Ring center X"), 1, 0)
        self.circle_cx_edit = QLineEdit("0")
        self.circle_cx_edit.setMaximumWidth(100)
        e_scatter_grid.addWidget(self.circle_cx_edit, 1, 1)
        e_scatter_grid.addWidget(QLabel("Ring center Y"), 1, 2)
        self.circle_cy_edit = QLineEdit("0")
        self.circle_cy_edit.setMaximumWidth(100)
        e_scatter_grid.addWidget(self.circle_cy_edit, 1, 3)
        e_scatter_grid.addWidget(QLabel("Inner radius (signal)"), 2, 0)
        self.inner_r_edit = QLineEdit("8")
        self.inner_r_edit.setMaximumWidth(100)
        e_scatter_grid.addWidget(self.inner_r_edit, 2, 1)
        e_scatter_grid.addWidget(QLabel("Outer radius (noise)"), 2, 2)
        self.outer_r_edit = QLineEdit("14")
        self.outer_r_edit.setMaximumWidth(100)
        e_scatter_grid.addWidget(self.outer_r_edit, 2, 3)
        self.apply_circle_btn = QPushButton("Apply Ring Selection and Center")
        self.apply_circle_btn.clicked.connect(self.apply_circle_selection)
        e_scatter_grid.addWidget(self.apply_circle_btn, 2, 4)
        control_grid.addWidget(e_scatter_group, 1, 0, 1, 7)

        # Step 4 controls (ion rectangle filter and alignment)
        ion_scatter_group = QGroupBox("Ion Scatter Plot")
        ion_scatter_grid = QGridLayout(ion_scatter_group)
        ion_scatter_grid.setHorizontalSpacing(10)
        ion_scatter_grid.addWidget(QLabel("Filter center X"), 0, 0)
        self.ion_filter_cx_edit = QLineEdit("0")
        self.ion_filter_cx_edit.setMaximumWidth(100)
        ion_scatter_grid.addWidget(self.ion_filter_cx_edit, 0, 1)
        ion_scatter_grid.addWidget(QLabel("Filter center Y"), 0, 2)
        self.ion_filter_cy_edit = QLineEdit("0")
        self.ion_filter_cy_edit.setMaximumWidth(100)
        ion_scatter_grid.addWidget(self.ion_filter_cy_edit, 0, 3)
        ion_scatter_grid.addWidget(QLabel("Filter width"), 1, 0)
        self.ion_filter_w_edit = QLineEdit("12")
        self.ion_filter_w_edit.setMaximumWidth(100)
        ion_scatter_grid.addWidget(self.ion_filter_w_edit, 1, 1)
        ion_scatter_grid.addWidget(QLabel("Filter height"), 1, 2)
        self.ion_filter_h_edit = QLineEdit("12")
        self.ion_filter_h_edit.setMaximumWidth(100)
        ion_scatter_grid.addWidget(self.ion_filter_h_edit, 1, 3)
        self.ion_filter_enable_checkbox = QCheckBox("Enable ion filter")
        self.ion_filter_enable_checkbox.setChecked(False)
        ion_scatter_grid.addWidget(self.ion_filter_enable_checkbox, 2, 0, 1, 2)
        self.ion_align_checkbox = QCheckBox("Auto rotate major axis to horizontal")
        self.ion_align_checkbox.setChecked(False)
        ion_scatter_grid.addWidget(self.ion_align_checkbox, 2, 2, 1, 2)
        ion_scatter_grid.addWidget(QLabel("Rotation offset (deg)"), 3, 0)
        self.ion_rot_offset_edit = QLineEdit("0")
        self.ion_rot_offset_edit.setMaximumWidth(100)
        ion_scatter_grid.addWidget(self.ion_rot_offset_edit, 3, 1)
        self.apply_ion_rot_btn = QPushButton("Apply Rotation")
        self.apply_ion_rot_btn.clicked.connect(self.apply_ion_rotation_offset)
        ion_scatter_grid.addWidget(self.apply_ion_rot_btn, 3, 2)
        self.ion_dirline_btn = QPushButton("Show Main Direction Line")
        self.ion_dirline_btn.setCheckable(True)
        self.ion_dirline_btn.setChecked(False)
        self.ion_dirline_btn.toggled.connect(self._on_toggle_ion_direction_line)
        ion_scatter_grid.addWidget(self.ion_dirline_btn, 3, 3)
        control_grid.addWidget(ion_scatter_group, 1, 7, 1, 5)

        # Step 5 + Step 7 controls (projection bin size + start reconstruction)
        projection_group = QGroupBox("Electron Projection Image")
        projection_grid = QGridLayout(projection_group)
        projection_grid.addWidget(QLabel("Centered bin size"), 0, 0)
        self.center_bin_edit = QLineEdit("0.1")
        self.center_bin_edit.setMaximumWidth(100)
        projection_grid.addWidget(self.center_bin_edit, 0, 1)
        self.reconstruct_btn = QPushButton("Start Reconstruction")
        self.reconstruct_btn.clicked.connect(self.run_reconstruction_now)
        projection_grid.addWidget(self.reconstruct_btn, 0, 2, 1, 2)
        control_grid.addWidget(projection_group, 2, 0, 1, 6)

        # Step 6 controls: reconstruction parameter panels (rBasex + backward)
        control_grid.addWidget(QLabel("rBasex peak smooth sigma"), 3, 0)
        self.rbasex_peak_smooth_sigma_edit = QLineEdit("0")
        self.rbasex_peak_smooth_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_peak_smooth_sigma_edit, 3, 1)

        control_grid.addWidget(QLabel("rBasex peak height"), 3, 2)
        self.rbasex_peak_height_edit = QLineEdit("0.12")
        self.rbasex_peak_height_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_peak_height_edit, 3, 3)

        control_grid.addWidget(QLabel("rBasex peak prominence"), 3, 4)
        self.rbasex_peak_prominence_edit = QLineEdit("0.08")
        self.rbasex_peak_prominence_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_peak_prominence_edit, 3, 5)

        control_grid.addWidget(QLabel("rBasex max peaks"), 3, 6)
        self.rbasex_max_peaks_edit = QLineEdit("5")
        self.rbasex_max_peaks_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_max_peaks_edit, 3, 7)

        control_grid.addWidget(QLabel("rBasex min-dist frac"), 4, 0)
        self.rbasex_peak_min_dist_frac_edit = QLineEdit("0.06")
        self.rbasex_peak_min_dist_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_peak_min_dist_frac_edit, 4, 1)

        control_grid.addWidget(QLabel("rBasex display percentile"), 4, 2)
        self.rbasex_display_percentile_edit = QLineEdit("99.7")
        self.rbasex_display_percentile_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_display_percentile_edit, 4, 3)

        control_grid.addWidget(QLabel("Backward n_theta"), 4, 4)
        self.backward_n_theta_edit = QLineEdit("720")
        self.backward_n_theta_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_n_theta_edit, 4, 5)

        control_grid.addWidget(QLabel("Backward mask radius"), 4, 6)
        self.backward_mask_radius_edit = QLineEdit("25")
        self.backward_mask_radius_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_mask_radius_edit, 4, 7)

        control_grid.addWidget(QLabel("Backward peak smooth sigma"), 5, 0)
        self.backward_peak_smooth_sigma_edit = QLineEdit("0")
        self.backward_peak_smooth_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_peak_smooth_sigma_edit, 5, 1)

        control_grid.addWidget(QLabel("Backward peak height"), 5, 2)
        self.backward_peak_height_edit = QLineEdit("0.12")
        self.backward_peak_height_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_peak_height_edit, 5, 3)

        control_grid.addWidget(QLabel("Backward peak prominence"), 5, 4)
        self.backward_peak_prominence_edit = QLineEdit("0.08")
        self.backward_peak_prominence_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_peak_prominence_edit, 5, 5)

        control_grid.addWidget(QLabel("Backward max peaks"), 5, 6)
        self.backward_max_peaks_edit = QLineEdit("5")
        self.backward_max_peaks_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_max_peaks_edit, 5, 7)

        control_grid.addWidget(QLabel("Backward min-dist frac"), 6, 0)
        self.backward_peak_min_dist_frac_edit = QLineEdit("0.06")
        self.backward_peak_min_dist_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_peak_min_dist_frac_edit, 6, 1)

        control_grid.addWidget(QLabel("Backward display percentile"), 6, 2)
        self.backward_display_percentile_edit = QLineEdit("99.7")
        self.backward_display_percentile_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_display_percentile_edit, 6, 3)

        control_grid.addWidget(QLabel("rBasex order"), 7, 0)
        self.rbasex_order_edit = QLineEdit("2")
        self.rbasex_order_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_order_edit, 7, 1)

        control_grid.addWidget(QLabel("rBasex reg (blank=None)"), 7, 2)
        self.rbasex_reg_edit = QLineEdit("")
        self.rbasex_reg_edit.setPlaceholderText("e.g. 200")
        self.rbasex_reg_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_reg_edit, 7, 3)

        control_grid.addWidget(QLabel("rBasex rmax (MIN/MAX/int)"), 7, 4)
        self.rbasex_rmax_edit = QLineEdit("MIN")
        self.rbasex_rmax_edit.setMaximumWidth(100)
        control_grid.addWidget(self.rbasex_rmax_edit, 7, 5)

        self.rbasex_odd_checkbox = QCheckBox("rBasex odd terms")
        self.rbasex_odd_checkbox.setChecked(False)
        control_grid.addWidget(self.rbasex_odd_checkbox, 7, 6, 1, 2)

        control_grid.addWidget(QLabel("Backward baseline factor"), 8, 0)
        self.backward_baseline_factor_edit = QLineEdit("0")
        self.backward_baseline_factor_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_baseline_factor_edit, 8, 1)

        control_grid.addWidget(QLabel("B init smooth sigma"), 9, 0)
        self.backward_init_smooth_sigma_edit = QLineEdit("3")
        self.backward_init_smooth_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_init_smooth_sigma_edit, 9, 1)

        control_grid.addWidget(QLabel("B init signal frac"), 9, 2)
        self.backward_init_signal_frac_edit = QLineEdit("0.05")
        self.backward_init_signal_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_init_signal_frac_edit, 9, 3)

        control_grid.addWidget(QLabel("B noise margin px"), 9, 4)
        self.backward_noise_margin_edit = QLineEdit("20")
        self.backward_noise_margin_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_noise_margin_edit, 9, 5)

        control_grid.addWidget(QLabel("B min-noise frac"), 9, 6)
        self.backward_min_noise_frac_edit = QLineEdit("0.15")
        self.backward_min_noise_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_min_noise_frac_edit, 9, 7)

        control_grid.addWidget(QLabel("B baseline edge frac"), 10, 0)
        self.backward_baseline_edge_frac_edit = QLineEdit("0.8")
        self.backward_baseline_edge_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_baseline_edge_frac_edit, 10, 1)

        control_grid.addWidget(QLabel("B baseline percentile"), 10, 2)
        self.backward_baseline_edge_percentile_edit = QLineEdit("25")
        self.backward_baseline_edge_percentile_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_baseline_edge_percentile_edit, 10, 3)

        control_grid.addWidget(QLabel("B bayes prior sigma"), 10, 4)
        self.backward_bayes_prior_sigma_edit = QLineEdit("3")
        self.backward_bayes_prior_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_bayes_prior_sigma_edit, 10, 5)

        control_grid.addWidget(QLabel("B bayes lowfreq sigma"), 10, 6)
        self.backward_bayes_lowfreq_sigma_edit = QLineEdit("0.1")
        self.backward_bayes_lowfreq_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_bayes_lowfreq_sigma_edit, 10, 7)

        control_grid.addWidget(QLabel("B bayes signal weight"), 11, 0)
        self.backward_bayes_signal_weight_edit = QLineEdit("0.7")
        self.backward_bayes_signal_weight_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_bayes_signal_weight_edit, 11, 1)

        control_grid.addWidget(QLabel("B proj smooth sigma"), 11, 2)
        self.backward_proj_smooth_sigma_edit = QLineEdit("2")
        self.backward_proj_smooth_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_proj_smooth_sigma_edit, 11, 3)

        control_grid.addWidget(QLabel("B peak height frac"), 11, 4)
        self.backward_phase1_peak_height_frac_edit = QLineEdit("0.03")
        self.backward_phase1_peak_height_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_peak_height_frac_edit, 11, 5)

        control_grid.addWidget(QLabel("B peak prom frac"), 11, 6)
        self.backward_phase1_peak_prom_frac_edit = QLineEdit("0.02")
        self.backward_phase1_peak_prom_frac_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_peak_prom_frac_edit, 11, 7)

        control_grid.addWidget(QLabel("B peak dist px"), 12, 0)
        self.backward_phase1_peak_dist_edit = QLineEdit("5")
        self.backward_phase1_peak_dist_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_peak_dist_edit, 12, 1)

        control_grid.addWidget(QLabel("B SNR switch"), 12, 2)
        self.backward_phase1_snr_switch_edit = QLineEdit("30")
        self.backward_phase1_snr_switch_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_snr_switch_edit, 12, 3)

        control_grid.addWidget(QLabel("B SNR low"), 12, 4)
        self.backward_phase1_snr_low_edit = QLineEdit("1.5")
        self.backward_phase1_snr_low_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_snr_low_edit, 12, 5)

        control_grid.addWidget(QLabel("B SNR high"), 12, 6)
        self.backward_phase1_snr_high_edit = QLineEdit("2.5")
        self.backward_phase1_snr_high_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_snr_high_edit, 12, 7)

        control_grid.addWidget(QLabel("B savgol win high"), 13, 0)
        self.backward_phase1_savgol_win_high_edit = QLineEdit("7")
        self.backward_phase1_savgol_win_high_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_savgol_win_high_edit, 13, 1)

        control_grid.addWidget(QLabel("B savgol win low"), 13, 2)
        self.backward_phase1_savgol_win_low_edit = QLineEdit("11")
        self.backward_phase1_savgol_win_low_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_savgol_win_low_edit, 13, 3)

        control_grid.addWidget(QLabel("B savgol poly"), 13, 4)
        self.backward_phase1_savgol_poly_edit = QLineEdit("3")
        self.backward_phase1_savgol_poly_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase1_savgol_poly_edit, 13, 5)

        control_grid.addWidget(QLabel("B beta smooth sigma"), 13, 6)
        self.backward_phase2_beta_smooth_sigma_edit = QLineEdit("1")
        self.backward_phase2_beta_smooth_sigma_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_beta_smooth_sigma_edit, 13, 7)

        control_grid.addWidget(QLabel("B opt-r sigma scale"), 14, 0)
        self.backward_phase2_opt_radius_scale_edit = QLineEdit("1.5")
        self.backward_phase2_opt_radius_scale_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_opt_radius_scale_edit, 14, 1)

        control_grid.addWidget(QLabel("B opt-r min search"), 14, 2)
        self.backward_phase2_opt_radius_min_edit = QLineEdit("3")
        self.backward_phase2_opt_radius_min_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_opt_radius_min_edit, 14, 3)

        control_grid.addWidget(QLabel("B multi sigma scale"), 14, 4)
        self.backward_phase2_multi_scale_edit = QLineEdit("1")
        self.backward_phase2_multi_scale_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_multi_scale_edit, 14, 5)

        control_grid.addWidget(QLabel("B multi n use"), 14, 6)
        self.backward_phase2_multi_n_use_edit = QLineEdit("3")
        self.backward_phase2_multi_n_use_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_multi_n_use_edit, 14, 7)

        control_grid.addWidget(QLabel("B multi min search"), 15, 0)
        self.backward_phase2_multi_min_edit = QLineEdit("3")
        self.backward_phase2_multi_min_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_multi_min_edit, 15, 1)

        control_grid.addWidget(QLabel("B radial filt sigma scale"), 15, 2)
        self.backward_phase2_radial_filter_scale_edit = QLineEdit("1.5")
        self.backward_phase2_radial_filter_scale_edit.setMaximumWidth(100)
        control_grid.addWidget(self.backward_phase2_radial_filter_scale_edit, 15, 3)

        self.control_scroll = QScrollArea()
        self.control_scroll.setWidget(control_group)
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setFrameShape(QFrame.NoFrame)
        self.top_layout.addWidget(self.control_scroll, stretch=1)

        self.status_label = QLabel("Status: waiting for files")
        self.top_layout.addWidget(self.status_label)

        # ------------------------------
        # Plot canvas area
        # ------------------------------
        self.figure = Figure(figsize=(23.5, 8.6))
        self.figure.subplots_adjust(left=0.03, right=0.99, bottom=0.08, top=0.95, wspace=0.35, hspace=0.35)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.bottom_layout.addWidget(self.toolbar)

        self.canvas.setMinimumSize(2500, 760)
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidget(self.canvas)
        self.plot_scroll.setWidgetResizable(False)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.bottom_layout.addWidget(self.plot_scroll, stretch=1)

        self.h_view_slider = QSlider(Qt.Horizontal)
        self.h_view_slider.setMinimum(0)
        self.h_view_slider.setMaximum(0)
        self.bottom_layout.addWidget(self.h_view_slider)

        # 2 rows x 4 columns:
        # [ion hist | e scatter | centered bin | rbasex]
        # [summary  | i scatter | centered bin | backward]
        gs = self.figure.add_gridspec(2, 4, width_ratios=[1.1, 1.0, 1.0, 1.0])
        self.ax_hist_ion = self.figure.add_subplot(gs[0, 0])
        self.ax_info = self.figure.add_subplot(gs[1, 0])
        self.ax_scatter_e = self.figure.add_subplot(gs[0, 1])
        self.ax_scatter_i = self.figure.add_subplot(gs[1, 1])
        self.ax_centered_bin = self.figure.add_subplot(gs[:, 2])
        self.ax_reserved_top = self.figure.add_subplot(gs[0, 3])
        self.ax_reserved_bottom = self.figure.add_subplot(gs[1, 3])

        self.ion_selector = self._create_span_selector()
        self._wire_horizontal_view_slider()
        for edit in (self.circle_cx_edit, self.circle_cy_edit, self.inner_r_edit, self.outer_r_edit):
            edit.textChanged.connect(self._schedule_circle_overlay_update)
        for edit in (self.ion_filter_cx_edit, self.ion_filter_cy_edit, self.ion_filter_w_edit, self.ion_filter_h_edit):
            edit.textChanged.connect(self._schedule_ion_overlay_update)
        self.ion_filter_enable_checkbox.toggled.connect(self._on_ion_filter_toggled)
        self.ion_align_checkbox.toggled.connect(self._on_ion_alignment_toggled)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_move)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._draw_placeholder()

    # ------------------------------------------------------------------
    # UI wiring and quick interaction callbacks
    # ------------------------------------------------------------------
    def _create_span_selector(self):
        """Create the ion-TOF fine ROI selector on the histogram axis."""
        try:
            return SpanSelector(
                self.ax_hist_ion,
                self._on_ion_span_selected,
                "horizontal",
                useblit=True,
                props={"alpha": 0.22, "facecolor": "#00b4d8"},
                interactive=True,
                drag_from_anywhere=True,
            )
        except TypeError:
            return SpanSelector(
                self.ax_hist_ion,
                self._on_ion_span_selected,
                "horizontal",
                useblit=True,
                rectprops={"alpha": 0.22, "facecolor": "#00b4d8"},
            )

    def _wire_horizontal_view_slider(self) -> None:
        """Bind the custom bottom slider to the plot area's horizontal scroll bar."""
        hbar = self.plot_scroll.horizontalScrollBar()

        def update_slider_range() -> None:
            self.h_view_slider.setMinimum(hbar.minimum())
            self.h_view_slider.setMaximum(hbar.maximum())
            self.h_view_slider.setValue(hbar.value())

        def on_hbar_changed(value: int) -> None:
            was = self.h_view_slider.blockSignals(True)
            self.h_view_slider.setValue(value)
            self.h_view_slider.blockSignals(was)

        def on_slider_changed(value: int) -> None:
            hbar.setValue(value)

        hbar.valueChanged.connect(on_hbar_changed)
        self.h_view_slider.valueChanged.connect(on_slider_changed)
        hbar.rangeChanged.connect(lambda _a, _b: update_slider_range())
        update_slider_range()

    def _on_ion_alignment_toggled(self, checked: bool) -> None:
        """Recompute view when automatic ion major-axis alignment is toggled."""
        _ = checked
        self._clear_circle_result()
        if self.matched_ion.shape[0] > 0:
            self._refresh_plots()
            if self.ion_align_checkbox.isChecked():
                self._set_status("Ion major-axis horizontal alignment enabled.")
            else:
                self._set_status("Ion major-axis alignment disabled.")
        else:
            self._update_ion_overlay_only()

    def _on_ion_filter_toggled(self, checked: bool) -> None:
        """Refresh plot state when ion rectangle filter is enabled/disabled."""
        self._clear_circle_result()
        if self.matched_ion.shape[0] > 0:
            self._refresh_plots()
            self._set_status("Ion filter enabled." if checked else "Ion filter disabled.")
        else:
            self._update_ion_overlay_only()

    def apply_ion_rotation_offset(self) -> None:
        """Apply user-entered extra rotation (degrees) for ion scatter display."""
        try:
            offset = float(self.ion_rot_offset_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, "Invalid rotation", "Ion rotation offset must be a valid number (degrees).")
            return
        self.ion_user_rotation_deg = offset
        if self.matched_ion.shape[0] > 0:
            self._clear_circle_result()
            self._refresh_plots()
        else:
            self._update_ion_overlay_only()
        self._set_status(f"Ion rotation offset set to {offset:.6g} deg.")

    def _on_toggle_ion_direction_line(self, checked: bool) -> None:
        """Show or hide the ion main-direction reference line."""
        if not checked:
            self._clear_ion_main_direction_artists()
        if self.matched_ion.shape[0] > 0:
            self._refresh_plots()
        else:
            self._update_ion_overlay_only()

    def _clear_ion_main_direction_artists(self) -> None:
        """Remove ion main-direction line artists from the axis."""
        if self.ion_main_axis_line is not None:
            with contextlib.suppress(Exception):
                self.ion_main_axis_line.remove()
            self.ion_main_axis_line = None
        self.ion_main_axis_marker = None
        self.ion_main_axis_angle_deg = None

    def _draw_ion_main_direction_line(self, ion_points: np.ndarray) -> None:
        """Estimate and draw dominant ion direction through highest-density region."""
        if not self.ion_dirline_btn.isChecked():
            self._clear_ion_main_direction_artists()
            return
        if ion_points.shape[0] < 8:
            self._clear_ion_main_direction_artists()
            return

        x = ion_points[:, 0].astype(np.float64, copy=False)
        y = ion_points[:, 1].astype(np.float64, copy=False)

        bins = int(np.clip(np.sqrt(ion_points.shape[0]) / 2.5, 32, 140))
        hist2d, xedges, yedges = np.histogram2d(x, y, bins=bins)
        if hist2d.size == 0 or float(np.max(hist2d)) <= 0.0:
            self._clear_ion_main_direction_artists()
            return

        max_idx = np.unravel_index(int(np.argmax(hist2d)), hist2d.shape)
        cx = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
        cy = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])

        cov = np.cov(np.column_stack((x, y)).T)
        evals, evecs = np.linalg.eigh(cov)
        direction = evecs[:, int(np.argmax(evals))]
        norm = float(np.hypot(direction[0], direction[1]))
        if norm <= 1e-12:
            self._clear_ion_main_direction_artists()
            return
        direction = direction / norm
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        self.ion_main_axis_angle_deg = angle

        x0, x1 = self.ax_scatter_i.get_xlim()
        y0, y1 = self.ax_scatter_i.get_ylim()
        span = max(abs(x1 - x0), abs(y1 - y0), 1e-9)
        t = 1.6 * span
        x_line = np.array([cx - direction[0] * t, cx + direction[0] * t], dtype=np.float64)
        y_line = np.array([cy - direction[1] * t, cy + direction[1] * t], dtype=np.float64)

        if self.ion_main_axis_line is None or self.ion_main_axis_line.axes is not self.ax_scatter_i:
            (self.ion_main_axis_line,) = self.ax_scatter_i.plot(
                x_line,
                y_line,
                color="#00e5ff",
                linewidth=1.6,
                alpha=0.9,
                zorder=6,
                scalex=False,
                scaley=False,
            )
            self.ion_main_axis_line.set_gid("ion_main_axis_line")
        else:
            self.ion_main_axis_line.set_data(x_line, y_line)

    def _compute_ion_rotation(self, ion_xy: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute rotation matrix that aligns ion major axis to horizontal."""
        if ion_xy.shape[0] < 2:
            return np.eye(2, dtype=np.float64), 0.0

        centered = ion_xy - np.mean(ion_xy, axis=0, keepdims=True)
        try:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            principal = evecs[:, int(np.argmax(evals))]
            theta = float(np.arctan2(principal[1], principal[0]))
            if theta > 0.5 * np.pi:
                theta -= np.pi
            elif theta <= -0.5 * np.pi:
                theta += np.pi
        except Exception:
            return np.eye(2, dtype=np.float64), 0.0

        # Rotate principal axis to x-axis (horizontal).
        alpha = -theta
        c, s = float(np.cos(alpha)), float(np.sin(alpha))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return rot, float(np.degrees(theta))

    def _update_ion_rotation_from_points(self, ion_xy: np.ndarray) -> None:
        """Update final ion rotation = (user rotation) x (optional auto alignment)."""
        if self.ion_align_checkbox.isChecked():
            rot_auto, auto_angle_deg = self._compute_ion_rotation(ion_xy)
        else:
            rot_auto = np.eye(2, dtype=np.float64)
            auto_angle_deg = 0.0

        user_angle_deg = float(self.ion_user_rotation_deg)
        a = np.deg2rad(user_angle_deg)
        c, s = float(np.cos(a)), float(np.sin(a))
        rot_user = np.array([[c, -s], [s, c]], dtype=np.float64)

        self.ion_auto_angle_deg = auto_angle_deg
        self.ion_rotation_matrix = rot_user @ rot_auto
        self.ion_rotation_angle_deg = auto_angle_deg + user_angle_deg

    def _transform_ion_xy(self, ion_xy: np.ndarray) -> np.ndarray:
        """Apply current ion rotation matrix to XY points."""
        if ion_xy.size == 0:
            return ion_xy.copy()
        return ion_xy @ self.ion_rotation_matrix.T

    # ------------------------------------------------------------------
    # Overlay artist lifecycle and blitting
    # ------------------------------------------------------------------
    def _invalidate_blit_background(self) -> None:
        """Mark cached background as dirty (must be recaptured before blit)."""
        self.bg_scatter_e = None
        self.bg_scatter_i = None

    def _dedupe_overlay_artists(self) -> None:
        """Remove stale duplicate overlay artists from scatter axes."""
        changed = False

        if self.inner_ring_patch is not None and self.inner_ring_patch.axes is not self.ax_scatter_e:
            self.inner_ring_patch = None
            changed = True
        if self.outer_ring_patch is not None and self.outer_ring_patch.axes is not self.ax_scatter_e:
            self.outer_ring_patch = None
            changed = True
        if self.circle_center_marker is not None and self.circle_center_marker.axes is not self.ax_scatter_e:
            self.circle_center_marker = None
            changed = True
        if self.ion_filter_patch is not None and self.ion_filter_patch.axes is not self.ax_scatter_i:
            self.ion_filter_patch = None
            changed = True
        if self.ion_filter_center_marker is not None and self.ion_filter_center_marker.axes is not self.ax_scatter_i:
            self.ion_filter_center_marker = None
            changed = True
        if self.ion_main_axis_line is not None and self.ion_main_axis_line.axes is not self.ax_scatter_i:
            self.ion_main_axis_line = None
            changed = True
        if self.ion_main_axis_marker is not None and self.ion_main_axis_marker.axes is not self.ax_scatter_i:
            self.ion_main_axis_marker = None
            changed = True

        keep_ids = {
            id(a)
            for a in (
                self.inner_ring_patch,
                self.outer_ring_patch,
                self.circle_center_marker,
                self.ion_filter_patch,
                self.ion_filter_center_marker,
                self.ion_main_axis_line,
                self.ion_main_axis_marker,
            )
            if a is not None
        }

        for p in list(self.ax_scatter_e.patches):
            if id(p) not in keep_ids:
                with contextlib.suppress(Exception):
                    p.remove()
                changed = True
        for l in list(self.ax_scatter_e.lines):
            if id(l) not in keep_ids:
                with contextlib.suppress(Exception):
                    l.remove()
                changed = True
        for p in list(self.ax_scatter_i.patches):
            if id(p) not in keep_ids:
                with contextlib.suppress(Exception):
                    p.remove()
                changed = True
        for l in list(self.ax_scatter_i.lines):
            if id(l) not in keep_ids:
                with contextlib.suppress(Exception):
                    l.remove()
                changed = True

        if changed:
            self._invalidate_blit_background()

    def _rebuild_blit_background(self) -> None:
        """Capture clean backgrounds for fast overlay redraw (blitting)."""
        self._dedupe_overlay_artists()

        artists = []
        for artist in (
            self.inner_ring_patch,
            self.outer_ring_patch,
            self.circle_center_marker,
            self.ion_filter_patch,
            self.ion_filter_center_marker,
        ):
            if artist is not None:
                try:
                    artists.append((artist, artist.get_visible()))
                    artist.set_visible(False)
                except Exception:
                    pass

        # Draw once without overlays, so backgrounds never contain stale circles/markers.
        self.canvas.draw()
        try:
            self.bg_scatter_e = self.canvas.copy_from_bbox(self.ax_scatter_e.bbox)
            self.bg_scatter_i = self.canvas.copy_from_bbox(self.ax_scatter_i.bbox)
        except Exception:
            self._invalidate_blit_background()

        for artist, visible in artists:
            with contextlib.suppress(Exception):
                artist.set_visible(visible)

        self._blit_overlays()

    def _capture_blit_background_from_current_canvas(self) -> None:
        """Capture scatter axis backgrounds from current full canvas draw."""
        try:
            self.bg_scatter_e = self.canvas.copy_from_bbox(self.ax_scatter_e.bbox)
            self.bg_scatter_i = self.canvas.copy_from_bbox(self.ax_scatter_i.bbox)
        except Exception:
            self._invalidate_blit_background()

    # ------------------------------------------------------------------
    # File/cache state and base processing
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        """Update one-line status text shown below the control panel."""
        self.status_label.setText(f"Status: {text}")

    def _set_file_path(self, role: str, file_path: str) -> None:
        """Register selected file path for one role: trigger/electron/ion."""
        file_path = os.path.normpath(file_path)
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "Invalid path", "Dropped path is not a file.")
            return
        self.file_paths[role] = file_path
        self.drop_frames[role].set_path(file_path)

        if self.cache is not None:
            self.cache = None
            self._clear_processed_data()
            self._set_status("File changed. Cache invalidated, please load again.")
        else:
            self._set_status(f"{ROLE_LABELS[role]} file selected.")

    def _clear_processed_data(self) -> None:
        """Reset derived data/selection state while keeping UI structure."""
        self.matched_electron = np.empty((0, 3), dtype=np.float64)
        self.matched_ion = np.empty((0, 3), dtype=np.float64)
        self.ion_range = None
        self.ion_hist_x_roi = None
        self.pending_ion_span_range = None
        self.ion_span_apply_timer.stop()
        self.preview_circle_center = None
        self.preview_ion_center = None
        self.pending_drag_center = None
        self.pending_drag_ion_center = None
        self.ion_rotation_matrix = np.eye(2, dtype=np.float64)
        self.ion_rotation_angle_deg = 0.0
        self.ion_auto_angle_deg = 0.0
        self.ion_user_rotation_deg = 0.0
        if hasattr(self, "ion_rot_offset_edit"):
            was = self.ion_rot_offset_edit.blockSignals(True)
            self.ion_rot_offset_edit.setText("0")
            self.ion_rot_offset_edit.blockSignals(was)
        self._clear_circle_result()
        self._sync_hist_roi_inputs(force=True)
        self._update_ion_selection_label()
        self._draw_placeholder()

    def _clear_circle_result(self) -> None:
        """Reset results produced by ring selection and projection/reconstruction."""
        self.ring_inner_selected_electron = np.empty((0, 3), dtype=np.float64)
        self.ring_outer_noise_electron = np.empty((0, 3), dtype=np.float64)
        self.ion_filter_selected_electron = np.empty((0, 3), dtype=np.float64)
        self.ion_filter_selected_ion = np.empty((0, 3), dtype=np.float64)
        self.intersection_indices = np.empty(0, dtype=np.int64)
        self.circle_centered_electron = np.empty((0, 3), dtype=np.float64)
        self.noise_ring_centered_electron = np.empty((0, 3), dtype=np.float64)
        self.circle_centroid = None
        self.center_residual = None
        self.centered_hist_data = None
        self.noise_removed_total = 0.0
        self.rbasex_recon_result = None
        self.backward_recon_result = None

    def clear_cache(self) -> None:
        """Clear file paths and in-memory cache."""
        self.cache = None
        for role in self.file_paths:
            self.file_paths[role] = ""
            self.drop_frames[role].clear_path()
        self._clear_processed_data()
        self._set_status("Cache and file paths cleared.")

    def load_cache(self) -> None:
        """Step 1a: read three files into numeric arrays and cache them."""
        missing = [role for role, path in self.file_paths.items() if not path]
        if missing:
            names = ", ".join(ROLE_LABELS[m] for m in missing)
            QMessageBox.warning(self, "Missing file", f"Please select: {names}")
            return

        self._set_status("Loading files into cache...")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            trigger_indices = np.loadtxt(
                self.file_paths["trigger"],
                delimiter=",",
                dtype=np.float64,
                usecols=(2, 3),
            )
            electron_points = np.loadtxt(
                self.file_paths["electron"],
                delimiter=",",
                dtype=np.float64,
                usecols=(0, 1, 2),
            )
            ion_points = np.loadtxt(
                self.file_paths["ion"],
                delimiter=",",
                dtype=np.float64,
                usecols=(0, 1, 2),
            )

            trigger_indices = ensure_2d(trigger_indices, 2, "Trigger file")
            electron_points = ensure_2d(electron_points, 3, "Electron file")
            ion_points = ensure_2d(ion_points, 3, "Ion file")

            self.cache = CacheData(
                trigger_indices=trigger_indices,
                electron_points=electron_points,
                ion_points=ion_points,
            )
            self._clear_processed_data()
        except Exception as exc:
            self.cache = None
            QMessageBox.critical(self, "Load failed", f"Failed to read files:\n{exc}")
            self._set_status("Load failed.")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._set_status(
            "Cache ready: trigger=%d rows, electron=%d rows, ion=%d rows"
            % (
                self.cache.trigger_indices.shape[0],
                self.cache.electron_points.shape[0],
                self.cache.ion_points.shape[0],
            )
        )

    def process_and_plot(self) -> None:
        """Step 1b: apply trigger +1/+1 rule and produce first set of plots."""
        if self.cache is None:
            QMessageBox.warning(self, "No cache", "Please load files into cache first.")
            return

        try:
            bins = int(self.bins_edit.text().strip())
            if bins <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid bins", "Histogram bins must be a positive integer.")
            return

        e_idx, i_idx = select_increment_pairs(self.cache.trigger_indices)
        if e_idx.size == 0:
            self.current_hist_bins = bins
            self._clear_processed_data()
            self._set_status("No rows satisfy the +1/+1 trigger rule.")
            return

        # Keep only rows where indices point to existing electron/ion rows.
        in_bounds = (
            (e_idx >= 0)
            & (e_idx < self.cache.electron_points.shape[0])
            & (i_idx >= 0)
            & (i_idx < self.cache.ion_points.shape[0])
        )
        out_of_range_count = int((~in_bounds).sum())

        self.matched_electron = self.cache.electron_points[e_idx[in_bounds]]
        self.matched_ion = self.cache.ion_points[i_idx[in_bounds]]
        self.current_hist_bins = bins
        self._clear_circle_result()
        self._refresh_plots()

        self._set_status(
            "Matched pairs=%d, out-of-range discarded=%d"
            % (self.matched_ion.shape[0], out_of_range_count)
        )

    # ------------------------------------------------------------------
    # Ion TOF selection (fine + coarse ROI)
    # ------------------------------------------------------------------
    def clear_ion_selection(self) -> None:
        """Clear fine ROI selection from ion TOF histogram."""
        self.pending_ion_span_range = None
        self.ion_span_apply_timer.stop()
        if self.ion_range is None:
            return
        self.ion_range = None
        self._clear_circle_result()
        self._update_ion_selection_label()
        self._refresh_plots()
        self._set_status("Ion t fine selection cleared.")

    def clear_ion_hist_x_roi(self) -> None:
        """Clear coarse X-range ROI used to render ion TOF histogram."""
        if self.ion_hist_x_roi is None:
            self._sync_hist_roi_inputs(force=True)
            return
        self.ion_hist_x_roi = None
        self._sync_hist_roi_inputs(force=True)
        self._update_ion_selection_label()
        self._refresh_ion_histogram_only()
        self._set_status("Ion histogram X ROI reset to full range.")

    def _sync_hist_roi_inputs(self, force: bool = False) -> None:
        """Synchronize coarse ROI line edits with current ROI state."""
        if not hasattr(self, "ion_hist_xmin_edit") or not hasattr(self, "ion_hist_xmax_edit"):
            return
        if not force and (self.ion_hist_xmin_edit.hasFocus() or self.ion_hist_xmax_edit.hasFocus()):
            return
        if self.ion_hist_x_roi is None:
            min_text, max_text = "", ""
        else:
            min_text, max_text = f"{self.ion_hist_x_roi[0]:.6g}", f"{self.ion_hist_x_roi[1]:.6g}"
        edits = (self.ion_hist_xmin_edit, self.ion_hist_xmax_edit)
        values = (min_text, max_text)
        for edit, value in zip(edits, values):
            was = edit.blockSignals(True)
            edit.setText(value)
            edit.blockSignals(was)

    def apply_ion_hist_x_roi_from_inputs(self) -> None:
        """Step 2: read coarse histogram X ROI from inputs and redraw histogram."""
        if self.matched_ion.shape[0] == 0:
            QMessageBox.warning(self, "No data", "Run Process and Plot first.")
            return

        txt_min = self.ion_hist_xmin_edit.text().strip() if hasattr(self, "ion_hist_xmin_edit") else ""
        txt_max = self.ion_hist_xmax_edit.text().strip() if hasattr(self, "ion_hist_xmax_edit") else ""
        if txt_min == "" and txt_max == "":
            self.clear_ion_hist_x_roi()
            return
        try:
            if txt_min == "":
                low = float(np.nanmin(self.matched_ion[:, 2]))
            else:
                low = float(txt_min)
            if txt_max == "":
                high = float(np.nanmax(self.matched_ion[:, 2]))
            else:
                high = float(txt_max)
        except Exception:
            QMessageBox.warning(self, "Invalid ROI", "Histogram X ROI min/max must be valid numbers.")
            return
        self._apply_ion_hist_x_roi(low, high, source="input")

    def _on_ion_span_selected(self, xmin: float, xmax: float) -> None:
        """Step 3: receive fine TOF ROI from SpanSelector (debounced)."""
        if self.matched_ion.shape[0] == 0:
            return
        if np.isclose(xmin, xmax, rtol=0.0, atol=1e-12):
            return
        low, high = sorted((float(xmin), float(xmax)))
        self.pending_ion_span_range = (low, high)
        self.ion_span_apply_timer.start()

    def _flush_pending_ion_span_selection(self) -> None:
        """Apply latest pending fine ROI once debounce timer fires."""
        if self.pending_ion_span_range is None:
            return
        low, high = self.pending_ion_span_range
        self.pending_ion_span_range = None
        if self.ion_range is not None:
            span = max(abs(high - low), 1.0)
            tol = 1e-6 * span
            if abs(self.ion_range[0] - low) <= tol and abs(self.ion_range[1] - high) <= tol:
                return
        self._apply_ion_selection_range(low, high, source="span")

    def _apply_ion_selection_range(self, low: float, high: float, source: str = "span") -> None:
        """Apply fine ROI to downstream filtering and redraw all affected panels."""
        if self.matched_ion.shape[0] == 0:
            return
        if np.isclose(low, high, rtol=0.0, atol=1e-12):
            return
        self.pending_ion_span_range = None
        low, high = sorted((float(low), float(high)))
        self.ion_range = (low, high)
        self._clear_circle_result()
        self._update_ion_selection_label()
        self._refresh_plots()

        selected_count = int(self._selected_mask().sum())
        self._set_status(
            "Ion t fine selection [%.6g, %.6g], selected=%d/%d (%s)"
            % (low, high, selected_count, self.matched_ion.shape[0], source)
        )

    def _apply_ion_hist_x_roi(self, low: float, high: float, source: str = "input") -> None:
        """Apply coarse ROI to histogram axis range and histogram data window."""
        if self.matched_ion.shape[0] == 0:
            return
        if np.isclose(low, high, rtol=0.0, atol=1e-12):
            return
        low, high = sorted((float(low), float(high)))
        self.ion_hist_x_roi = (low, high)
        self._sync_hist_roi_inputs(force=True)
        self._update_ion_selection_label()
        self._refresh_ion_histogram_only()
        self._set_status("Ion histogram X ROI set to [%.6g, %.6g] (%s)" % (low, high, source))

    def _update_ion_selection_label(self) -> None:
        """Update label that summarizes fine/coarse ROI settings."""
        fine_text = "all"
        if self.ion_range is not None:
            fine_text = f"[{self.ion_range[0]:.6g}, {self.ion_range[1]:.6g}]"
        coarse_text = "full"
        if self.ion_hist_x_roi is not None:
            coarse_text = f"[{self.ion_hist_x_roi[0]:.6g}, {self.ion_hist_x_roi[1]:.6g}]"
        self.selection_label.setText(f"Ion t selection (fine): {fine_text} | Histogram X ROI (coarse): {coarse_text}")

    def _refresh_ion_histogram_only(self, draw_canvas: bool = True) -> None:
        """Redraw only ion TOF histogram panel (fast path)."""
        self.ax_hist_ion.clear()
        ion_t_full = self.matched_ion[:, 2] if self.matched_ion.size else np.array([])
        ion_t = ion_t_full
        if self.ion_hist_x_roi is not None and ion_t_full.size:
            x_low, x_high = self.ion_hist_x_roi
            hist_mask = (ion_t_full >= x_low) & (ion_t_full <= x_high)
            ion_t = ion_t_full[hist_mask]
        self.ax_hist_ion.hist(ion_t, bins=self.current_hist_bins, color="#d62728", alpha=0.88)
        if self.ion_hist_x_roi is not None:
            self.ax_hist_ion.set_title(f"Ion t Histogram (x-ROI n={ion_t.size}, total={ion_t_full.size})")
            self.ax_hist_ion.set_xlim(self.ion_hist_x_roi[0], self.ion_hist_x_roi[1])
        else:
            self.ax_hist_ion.set_title(f"Ion t Histogram (n={ion_t.size})")
        self.ax_hist_ion.set_xlabel("t (ns)")
        self.ax_hist_ion.set_ylabel("counts")
        self.ax_hist_ion.grid(alpha=0.2)
        if self.ion_range is not None:
            low, high = self.ion_range
            self.ion_selection_patch = self.ax_hist_ion.axvspan(low, high, color="#00b4d8", alpha=0.22)
        else:
            self.ion_selection_patch = None
        if draw_canvas:
            self.canvas.draw_idle()

    def _selected_mask(self) -> np.ndarray:
        """Return boolean mask of rows selected by ion fine ROI."""
        n = self.matched_ion.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.ion_range is None:
            return np.ones(n, dtype=bool)
        low, high = self.ion_range
        ion_t = self.matched_ion[:, 2]
        return (ion_t >= low) & (ion_t <= high)

    # ------------------------------------------------------------------
    # Parameter parsing and reconstruction settings
    # ------------------------------------------------------------------
    def _parse_circle_params(self, show_dialog: bool) -> tuple[float, float, float, float] | None:
        """Parse electron ring center/radii from controls."""
        try:
            cx = float(self.circle_cx_edit.text().strip())
            cy = float(self.circle_cy_edit.text().strip())
            inner_r = float(self.inner_r_edit.text().strip())
            outer_r = float(self.outer_r_edit.text().strip())
            if inner_r <= 0 or outer_r <= inner_r:
                raise ValueError
        except ValueError:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "Invalid ring parameters",
                    "Ring requires valid numbers and must satisfy 0 < inner < outer.",
                )
            return None
        return cx, cy, inner_r, outer_r

    def _parse_ion_filter_params(self, show_dialog: bool) -> tuple[float, float, float, float] | None:
        """Parse ion rectangle filter center/size from controls."""
        try:
            cx = float(self.ion_filter_cx_edit.text().strip())
            cy = float(self.ion_filter_cy_edit.text().strip())
            width = float(self.ion_filter_w_edit.text().strip())
            height = float(self.ion_filter_h_edit.text().strip())
            if width <= 0 or height <= 0:
                raise ValueError
        except ValueError:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "Invalid ion filter",
                    "Ion filter requires valid center and positive width/height.",
                )
            return None
        return cx, cy, width, height

    def _parse_center_bin_size(self, show_dialog: bool) -> float | None:
        """Parse 2D bin size used to build centered projection image."""
        try:
            size = float(self.center_bin_edit.text().strip())
            if size <= 0:
                raise ValueError
            return size
        except ValueError:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "Invalid bin size",
                    "Centered bin size must be a positive number, e.g. 0.1.",
                )
            return None

    def _parse_float_edit(
        self, edit: QLineEdit, default: float, min_value: float | None = None, max_value: float | None = None
    ) -> float:
        """Read float from line edit, with default and clamping."""
        try:
            value = float(edit.text().strip())
        except Exception:
            value = default
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    def _parse_int_edit(
        self, edit: QLineEdit, default: int, min_value: int | None = None, max_value: int | None = None
    ) -> int:
        """Read int from line edit, with default and clamping."""
        try:
            value = int(float(edit.text().strip()))
        except Exception:
            value = default
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    def _parse_odd_int_edit(
        self, edit: QLineEdit, default: int, min_value: int | None = None, max_value: int | None = None
    ) -> int:
        """Read odd integer from line edit (used for Savitzky-Golay windows)."""
        value = self._parse_int_edit(edit, default, min_value, max_value)
        if value % 2 == 0:
            value += 1
            if max_value is not None and value > max_value:
                value -= 2
        return max(1, value)

    def _get_rbasex_settings(self) -> dict:
        """Collect all rBasex-related settings from UI controls."""
        rmax_text = self.rbasex_rmax_edit.text().strip()
        if not rmax_text:
            rmax_value: str | int = "MIN"
        else:
            rmax_upper = rmax_text.upper()
            if rmax_upper in {"MIN", "MAX"}:
                rmax_value = rmax_upper
            else:
                try:
                    rmax_value = max(1, int(float(rmax_text)))
                except Exception:
                    rmax_value = "MIN"

        reg_text = self.rbasex_reg_edit.text().strip()
        if not reg_text or reg_text.lower() == "none":
            reg_value = None
        else:
            try:
                reg_value = float(reg_text)
                if reg_value < 0:
                    reg_value = 0.0
            except Exception:
                reg_value = None

        return {
            "order": self._parse_int_edit(self.rbasex_order_edit, 2, 0, 8),
            "odd": self.rbasex_odd_checkbox.isChecked(),
            "reg": reg_value,
            "rmax": rmax_value,
            "peak_smooth_sigma": self._parse_float_edit(self.rbasex_peak_smooth_sigma_edit, 0.0, 0.0, 20.0),
            "peak_height": self._parse_float_edit(self.rbasex_peak_height_edit, 0.12, 0.0, 1.0),
            "peak_prominence": self._parse_float_edit(self.rbasex_peak_prominence_edit, 0.08, 0.0, 1.0),
            "peak_min_dist_frac": self._parse_float_edit(self.rbasex_peak_min_dist_frac_edit, 0.06, 0.0, 1.0),
            "max_peaks": self._parse_int_edit(self.rbasex_max_peaks_edit, 5, 1, 20),
            "display_percentile": self._parse_float_edit(self.rbasex_display_percentile_edit, 99.7, 50.0, 100.0),
        }

    def _get_backward_settings(self) -> dict:
        """Collect all backward-reconstruction settings from UI controls."""
        internal_params = {
            "init_profile_smooth_sigma": self._parse_float_edit(self.backward_init_smooth_sigma_edit, 3.0, 0.0, 50.0),
            "init_signal_threshold_frac": self._parse_float_edit(self.backward_init_signal_frac_edit, 0.05, 0.0, 1.0),
            "init_noise_margin_px": self._parse_int_edit(self.backward_noise_margin_edit, 20, 0, 10000),
            "init_min_noise_region_frac": self._parse_float_edit(self.backward_min_noise_frac_edit, 0.15, 0.0, 1.0),
            "baseline_edge_start_frac": self._parse_float_edit(
                self.backward_baseline_edge_frac_edit, 0.8, 0.0, 0.999
            ),
            "baseline_edge_percentile": self._parse_float_edit(
                self.backward_baseline_edge_percentile_edit, 25.0, 0.0, 100.0
            ),
            "bayes_prior_sigma": self._parse_float_edit(self.backward_bayes_prior_sigma_edit, 3.0, 0.0, 50.0),
            "bayes_lowfreq_sigma": self._parse_float_edit(self.backward_bayes_lowfreq_sigma_edit, 0.1, 1e-6, 10.0),
            "bayes_wiener_signal_weight": self._parse_float_edit(
                self.backward_bayes_signal_weight_edit, 0.7, 0.0, 1.0
            ),
            "phase1_proj_smooth_sigma": self._parse_float_edit(
                self.backward_proj_smooth_sigma_edit, 2.0, 0.0, 50.0
            ),
            "phase1_peak_height_frac": self._parse_float_edit(
                self.backward_phase1_peak_height_frac_edit, 0.03, 0.0, 1.0
            ),
            "phase1_peak_prominence_frac": self._parse_float_edit(
                self.backward_phase1_peak_prom_frac_edit, 0.02, 0.0, 1.0
            ),
            "phase1_peak_distance_px": self._parse_int_edit(self.backward_phase1_peak_dist_edit, 5, 1, 10000),
            "phase1_snr_switch": self._parse_float_edit(self.backward_phase1_snr_switch_edit, 30.0, 0.0, 1e6),
            "phase1_snr_low": self._parse_float_edit(self.backward_phase1_snr_low_edit, 1.5, 0.0, 1e6),
            "phase1_snr_high": self._parse_float_edit(self.backward_phase1_snr_high_edit, 2.5, 0.0, 1e6),
            "phase1_abel_savgol_window_high": self._parse_odd_int_edit(
                self.backward_phase1_savgol_win_high_edit, 7, 3, 10001
            ),
            "phase1_abel_savgol_window_low": self._parse_odd_int_edit(
                self.backward_phase1_savgol_win_low_edit, 11, 3, 10001
            ),
            "phase1_abel_savgol_polyorder": self._parse_int_edit(self.backward_phase1_savgol_poly_edit, 3, 1, 100),
            "phase2_beta_smooth_sigma": self._parse_float_edit(
                self.backward_phase2_beta_smooth_sigma_edit, 1.0, 0.0, 50.0
            ),
            "phase2_opt_radius_sigma_scale": self._parse_float_edit(
                self.backward_phase2_opt_radius_scale_edit, 1.5, 0.0, 100.0
            ),
            "phase2_opt_radius_min_search": self._parse_int_edit(
                self.backward_phase2_opt_radius_min_edit, 3, 1, 10000
            ),
            "phase2_multi_sigma_scale": self._parse_float_edit(
                self.backward_phase2_multi_scale_edit, 1.0, 0.0, 100.0
            ),
            "phase2_multi_n_use": self._parse_int_edit(self.backward_phase2_multi_n_use_edit, 3, 1, 101),
            "phase2_multi_min_search": self._parse_int_edit(self.backward_phase2_multi_min_edit, 3, 1, 10000),
            "phase2_radial_filter_sigma_scale": self._parse_float_edit(
                self.backward_phase2_radial_filter_scale_edit, 1.5, 0.0, 100.0
            ),
        }
        return {
            "n_theta": self._parse_int_edit(self.backward_n_theta_edit, 720, 90, 4000),
            "mask_radius": self._parse_int_edit(self.backward_mask_radius_edit, 25, 0, 2000),
            "baseline_factor": self._parse_float_edit(self.backward_baseline_factor_edit, 0.0, 0.0, 10.0),
            "peak_smooth_sigma": self._parse_float_edit(self.backward_peak_smooth_sigma_edit, 0.0, 0.0, 20.0),
            "peak_height": self._parse_float_edit(self.backward_peak_height_edit, 0.12, 0.0, 1.0),
            "peak_prominence": self._parse_float_edit(self.backward_peak_prominence_edit, 0.08, 0.0, 1.0),
            "peak_min_dist_frac": self._parse_float_edit(self.backward_peak_min_dist_frac_edit, 0.06, 0.0, 1.0),
            "max_peaks": self._parse_int_edit(self.backward_max_peaks_edit, 5, 1, 20),
            "display_percentile": self._parse_float_edit(self.backward_display_percentile_edit, 99.7, 50.0, 100.0),
            "internal_params": internal_params,
        }

    def _current_center_mode(self) -> str:
        """Return selected center estimator key."""
        mode = self.center_mode_combo.currentData()
        if isinstance(mode, str):
            return mode
        return "edge_fit"

    def _estimate_center(self, points_xy: np.ndarray, fallback_xy: tuple[float, float]) -> np.ndarray:
        """Estimate ring center using current center mode."""
        if points_xy.shape[0] == 0:
            return np.array([fallback_xy[0], fallback_xy[1]], dtype=np.float64)

        mode = self._current_center_mode()
        if mode == "centroid":
            return np.mean(points_xy, axis=0).astype(np.float64)
        if mode == "geo_median":
            return geometric_median(points_xy)
        return edge_circle_center(points_xy, np.array([fallback_xy[0], fallback_xy[1]], dtype=np.float64))

    # ------------------------------------------------------------------
    # Main analysis actions: Apply selection and run reconstruction
    # ------------------------------------------------------------------
    def apply_circle_selection(self) -> None:
        """Step 5: apply filters, center selected electrons, build denoised projection."""
        if self.matched_electron.shape[0] == 0:
            QMessageBox.warning(self, "No data", "Run Process and Plot first.")
            return

        circle = self._parse_circle_params(show_dialog=True)
        if circle is None:
            return

        center_bin_size = self._parse_center_bin_size(show_dialog=True)
        if center_bin_size is None:
            return

        # Step 1: start from ion fine-ROI selected rows.
        selected_mask = self._selected_mask()
        selected_indices = np.flatnonzero(selected_mask)
        electron_now = self.matched_electron[selected_mask]
        ion_now = self.matched_ion[selected_mask]
        if electron_now.shape[0] == 0:
            QMessageBox.warning(self, "No points", "Current ion t selection contains no points.")
            return

        # Step 2: update ion display rotation and optional ion rectangle filter.
        self._update_ion_rotation_from_points(ion_now[:, :2])
        ion_filter_on = self.ion_filter_enable_checkbox.isChecked()
        if ion_filter_on:
            ion_filter = self._parse_ion_filter_params(show_dialog=True)
            if ion_filter is None:
                return
            ion_cx, ion_cy, ion_w, ion_h = ion_filter
            ion_xy_view = self._transform_ion_xy(ion_now[:, :2])
            ion_inside = (
                (ion_xy_view[:, 0] >= ion_cx - 0.5 * ion_w)
                & (ion_xy_view[:, 0] <= ion_cx + 0.5 * ion_w)
                & (ion_xy_view[:, 1] >= ion_cy - 0.5 * ion_h)
                & (ion_xy_view[:, 1] <= ion_cy + 0.5 * ion_h)
            )
            electron_now = electron_now[ion_inside]
            ion_now = ion_now[ion_inside]
            selected_indices = selected_indices[ion_inside]
            if electron_now.shape[0] == 0:
                self._clear_circle_result()
                self._refresh_plots()
                self._set_status("Ion filter enabled, but no events inside ion filter rectangle.")
                return

        # Step 3: split electron points into inner signal ring and outer noise ring.
        cx, cy, inner_r, outer_r = circle
        dist2 = (electron_now[:, 0] - cx) ** 2 + (electron_now[:, 1] - cy) ** 2
        inner_mask = dist2 <= inner_r**2
        outer_mask = (dist2 > inner_r**2) & (dist2 <= outer_r**2)
        inner_selected = electron_now[inner_mask]
        outer_noise = electron_now[outer_mask]

        if inner_selected.shape[0] == 0:
            self._clear_circle_result()
            self._refresh_plots()
            self._set_status("No electron points inside current inner ring.")
            return

        # Step 4: estimate center and optionally recenter once more.
        mode = self._current_center_mode()
        center_xy = self._estimate_center(inner_selected[:, :2], (cx, cy))
        auto_recenter = self.auto_recenter_checkbox.isChecked()
        if auto_recenter:
            cx_ref, cy_ref = float(center_xy[0]), float(center_xy[1])
            dist2_ref = (electron_now[:, 0] - cx_ref) ** 2 + (electron_now[:, 1] - cy_ref) ** 2
            inner_ref = electron_now[dist2_ref <= inner_r**2]
            outer_ref = electron_now[(dist2_ref > inner_r**2) & (dist2_ref <= outer_r**2)]
            if inner_ref.shape[0] > 0:
                inner_selected = inner_ref
                outer_noise = outer_ref
                center_xy = self._estimate_center(inner_selected[:, :2], (cx_ref, cy_ref))

        # Step 5: center both signal and noise coordinates around estimated center.
        centered_signal = inner_selected.copy()
        centered_signal[:, 0] -= center_xy[0]
        centered_signal[:, 1] -= center_xy[1]
        centered_noise = outer_noise.copy()
        if centered_noise.shape[0] > 0:
            centered_noise[:, 0] -= center_xy[0]
            centered_noise[:, 1] -= center_xy[1]
        residual_xy = np.mean(centered_signal[:, :2], axis=0)

        self.ring_inner_selected_electron = inner_selected.copy()
        self.ring_outer_noise_electron = outer_noise.copy()
        self.ion_filter_selected_electron = electron_now.copy()
        self.ion_filter_selected_ion = ion_now.copy()
        self.intersection_indices = selected_indices.copy()
        self.circle_centered_electron = centered_signal
        self.noise_ring_centered_electron = centered_noise
        self.circle_centroid = (float(center_xy[0]), float(center_xy[1]))
        self.center_residual = (float(residual_xy[0]), float(residual_xy[1]))
        if auto_recenter:
            self._set_circle_inputs(self.circle_centroid[0], self.circle_centroid[1])
        # Step 6: bin and denoise centered signal for reconstruction input.
        self.centered_hist_data = build_denoised_centered_histogram(
            self.circle_centered_electron,
            self.noise_ring_centered_electron,
            inner_r,
            outer_r,
            center_bin_size,
        )
        self.noise_removed_total = (
            float(self.centered_hist_data.get("removed_total", 0.0)) if self.centered_hist_data is not None else 0.0
        )
        self.rbasex_recon_result = None
        self.backward_recon_result = None
        self._refresh_plots()
        self._set_status(
            "IonFilter=%s (n=%d), IonRot=%.3fdeg, Ring inner=%d, ring outer=%d, center=(%.6g, %.6g), mode=%s, residual=(%.3e, %.3e), "
            "auto_recenter=%s, bin=%.6g, removed_noise=%.3f. Projection updated. Click Start Reconstruction."
            % (
                "on" if ion_filter_on else "off",
                electron_now.shape[0],
                self.ion_rotation_angle_deg,
                inner_selected.shape[0],
                outer_noise.shape[0],
                center_xy[0],
                center_xy[1],
                mode,
                residual_xy[0],
                residual_xy[1],
                "on" if auto_recenter else "off",
                center_bin_size,
                self.noise_removed_total,
            )
        )

    def run_reconstruction_now(self) -> None:
        """Step 7: run rBasex and backward reconstruction from current centered image."""
        if self.centered_hist_data is None:
            QMessageBox.warning(self, "No projection", "Run Apply Ring Selection and Center first.")
            return

        center_bin_size = self._parse_center_bin_size(show_dialog=True)
        if center_bin_size is None:
            return

        self._set_status("Running rBasex and backward reconstructions...")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._run_reconstructions_from_centered_data(center_bin_size)
        finally:
            QApplication.restoreOverrideCursor()

        self._refresh_reconstruction_panels_only()
        rb_n = len(self.rbasex_recon_result.get("peaks", [])) if self.rbasex_recon_result else 0
        bw_n = len(self.backward_recon_result.get("peaks", [])) if self.backward_recon_result else 0
        self._set_status(f"Reconstruction finished: rBasex peaks={rb_n}, backward peaks={bw_n}.")

    # ------------------------------------------------------------------
    # Mouse/overlay interaction for ring and ion rectangle
    # ------------------------------------------------------------------
    def _schedule_circle_overlay_update(self) -> None:
        """Debounced ring overlay update after text edit."""
        if not self.dragging_circle:
            self.preview_circle_center = None
        self.circle_preview_timer.start()

    def _schedule_ion_overlay_update(self) -> None:
        """Debounced ion filter overlay update after text edit."""
        if not self.dragging_ion_filter:
            self.preview_ion_center = None
        self.ion_overlay_preview_timer.start()

    def _set_circle_inputs(self, cx: float, cy: float) -> None:
        """Write ring center coordinates to UI edits without recursive signals."""
        edits = (self.circle_cx_edit, self.circle_cy_edit)
        values = (cx, cy)
        for edit, val in zip(edits, values):
            txt = f"{val:.6g}"
            if edit.text() == txt:
                continue
            was_blocked = edit.blockSignals(True)
            edit.setText(txt)
            edit.blockSignals(was_blocked)

    def _set_ion_filter_inputs(self, cx: float, cy: float) -> None:
        """Write ion filter center coordinates to UI edits without recursive signals."""
        edits = (self.ion_filter_cx_edit, self.ion_filter_cy_edit)
        values = (cx, cy)
        for edit, val in zip(edits, values):
            txt = f"{val:.6g}"
            if edit.text() == txt:
                continue
            was_blocked = edit.blockSignals(True)
            edit.setText(txt)
            edit.blockSignals(was_blocked)

    def _on_canvas_press(self, event) -> None:
        """Start dragging ring/rectangle when left-click occurs inside overlay."""
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes == self.ax_scatter_e:
            circle = self._parse_circle_params(show_dialog=False)
            if circle is None:
                return
            cx, cy, _inner_r, outer_r = circle
            dx = float(event.xdata) - cx
            dy = float(event.ydata) - cy
            dist = float(np.hypot(dx, dy))

            x0, x1 = self.ax_scatter_e.get_xlim()
            y0, y1 = self.ax_scatter_e.get_ylim()
            tol = 0.03 * max(abs(x1 - x0), abs(y1 - y0), 1e-9)
            if dist <= outer_r + tol:
                self.dragging_circle = True
                self.drag_offset_x = dx
                self.drag_offset_y = dy
                self.preview_circle_center = (cx, cy)
            return

        if event.inaxes == self.ax_scatter_i:
            ion_rect = self._parse_ion_filter_params(show_dialog=False)
            if ion_rect is None:
                return
            cx, cy, width, height = ion_rect
            dx = float(event.xdata) - cx
            dy = float(event.ydata) - cy

            x0, x1 = self.ax_scatter_i.get_xlim()
            y0, y1 = self.ax_scatter_i.get_ylim()
            tol = 0.03 * max(abs(x1 - x0), abs(y1 - y0), 1e-9)
            if (abs(dx) <= 0.5 * width + tol) and (abs(dy) <= 0.5 * height + tol):
                self.dragging_ion_filter = True
                self.ion_drag_offset_x = dx
                self.ion_drag_offset_y = dy
                self.preview_ion_center = (cx, cy)

    def _on_canvas_move(self, event) -> None:
        """Queue drag-preview updates while mouse moves."""
        if self.dragging_circle:
            if event.inaxes != self.ax_scatter_e or event.xdata is None or event.ydata is None:
                return
            new_cx = float(event.xdata) - self.drag_offset_x
            new_cy = float(event.ydata) - self.drag_offset_y
            self.pending_drag_center = (new_cx, new_cy)
            if not self.drag_preview_timer.isActive():
                self.drag_preview_timer.start()
            return

        if self.dragging_ion_filter:
            if event.inaxes != self.ax_scatter_i or event.xdata is None or event.ydata is None:
                return
            new_cx = float(event.xdata) - self.ion_drag_offset_x
            new_cy = float(event.ydata) - self.ion_drag_offset_y
            self.pending_drag_ion_center = (new_cx, new_cy)
            if not self.ion_drag_preview_timer.isActive():
                self.ion_drag_preview_timer.start()

    def _on_canvas_release(self, event) -> None:
        """Commit drag result to controls on mouse release."""
        _ = event
        was_dragging_e = self.dragging_circle
        was_dragging_i = self.dragging_ion_filter
        self.dragging_circle = False
        self.dragging_ion_filter = False
        if was_dragging_e and self.pending_drag_center is not None:
            self._flush_drag_preview()
        if was_dragging_e and self.preview_circle_center is not None:
            self._set_circle_inputs(self.preview_circle_center[0], self.preview_circle_center[1])
            self.preview_circle_center = None
            self._update_circle_overlay_only()
        if was_dragging_i and self.pending_drag_ion_center is not None:
            self._flush_ion_drag_preview()
        if was_dragging_i and self.preview_ion_center is not None:
            self._set_ion_filter_inputs(self.preview_ion_center[0], self.preview_ion_center[1])
            self.preview_ion_center = None
            self._update_ion_overlay_only()

    def _flush_drag_preview(self) -> None:
        """Apply latest pending electron-ring drag position to preview overlay."""
        if self.pending_drag_center is None:
            return
        cx, cy = self.pending_drag_center
        self.pending_drag_center = None
        self.preview_circle_center = (cx, cy)
        self._update_circle_overlay_only()

    def _flush_ion_drag_preview(self) -> None:
        """Apply latest pending ion-filter drag position to preview overlay."""
        if self.pending_drag_ion_center is None:
            return
        cx, cy = self.pending_drag_ion_center
        self.pending_drag_ion_center = None
        self.preview_ion_center = (cx, cy)
        self._update_ion_overlay_only()

    def _clear_circle_overlay_artists(self) -> None:
        """Remove all electron overlay artists and reset references."""
        self._invalidate_blit_background()
        self.preview_circle_center = None
        # Hard cleanup to avoid any stale overlays from older versions/state.
        for p in list(self.ax_scatter_e.patches):
            try:
                p.remove()
            except Exception:
                pass
        for l in list(self.ax_scatter_e.lines):
            try:
                l.remove()
            except Exception:
                pass
        if self.inner_ring_patch is not None:
            try:
                self.inner_ring_patch.remove()
            except Exception:
                pass
            self.inner_ring_patch = None
        if self.outer_ring_patch is not None:
            try:
                self.outer_ring_patch.remove()
            except Exception:
                pass
            self.outer_ring_patch = None
        if self.circle_center_marker is not None:
            try:
                self.circle_center_marker.remove()
            except Exception:
                pass
            self.circle_center_marker = None

    def _clear_ion_overlay_artists(self) -> None:
        """Remove all ion overlay artists and reset references."""
        self._invalidate_blit_background()
        self.preview_ion_center = None
        # Hard cleanup to avoid any stale overlays from older versions/state.
        for p in list(self.ax_scatter_i.patches):
            try:
                p.remove()
            except Exception:
                pass
        for l in list(self.ax_scatter_i.lines):
            try:
                l.remove()
            except Exception:
                pass
        if self.ion_filter_patch is not None:
            try:
                self.ion_filter_patch.remove()
            except Exception:
                pass
            self.ion_filter_patch = None
        if self.ion_filter_center_marker is not None:
            try:
                self.ion_filter_center_marker.remove()
            except Exception:
                pass
            self.ion_filter_center_marker = None
        self.ion_main_axis_line = None
        self.ion_main_axis_marker = None
        self.ion_main_axis_angle_deg = None

    def _draw_circle_overlay(self) -> None:
        """Create/update electron ring + center marker artists."""
        circle = self._parse_circle_params(show_dialog=False)
        if circle is None:
            return
        cx, cy, inner_r, outer_r = circle
        if self.preview_circle_center is not None:
            cx, cy = self.preview_circle_center
        if self.inner_ring_patch is None or self.inner_ring_patch.axes is not self.ax_scatter_e:
            self.inner_ring_patch = Circle(
                (cx, cy), inner_r, fill=False, linewidth=1.8, edgecolor="#00d4ff", animated=True
            )
            self.inner_ring_patch.set_gid("electron_inner_ring")
            self.inner_ring_patch.set_transform(self.ax_scatter_e.transData)
            self.ax_scatter_e.add_patch(self.inner_ring_patch)
        self.inner_ring_patch.set_center((cx, cy))
        self.inner_ring_patch.set_radius(inner_r)
        if self.outer_ring_patch is None or self.outer_ring_patch.axes is not self.ax_scatter_e:
            self.outer_ring_patch = Circle(
                (cx, cy), outer_r, fill=False, linewidth=1.6, edgecolor="#7fffd4", linestyle="--", animated=True
            )
            self.outer_ring_patch.set_gid("electron_outer_ring")
            self.outer_ring_patch.set_transform(self.ax_scatter_e.transData)
            self.ax_scatter_e.add_patch(self.outer_ring_patch)
        self.outer_ring_patch.set_center((cx, cy))
        self.outer_ring_patch.set_radius(outer_r)
        if self.circle_center_marker is None or self.circle_center_marker.axes is not self.ax_scatter_e:
            (self.circle_center_marker,) = self.ax_scatter_e.plot(
                [cx], [cy], marker="o", markersize=4.5, color="#00d4ff", linestyle="", animated=True
            )
            self.circle_center_marker.set_gid("electron_center_marker")
        self.circle_center_marker.set_data([cx], [cy])

    def _draw_ion_overlay(self) -> None:
        """Create/update ion filter rectangle + center marker artists."""
        ion_rect = self._parse_ion_filter_params(show_dialog=False)
        if ion_rect is None:
            return
        cx, cy, width, height = ion_rect
        if self.preview_ion_center is not None:
            cx, cy = self.preview_ion_center
        enabled = self.ion_filter_enable_checkbox.isChecked()
        edgecolor = "#33c759" if enabled else "#9a9a9a"
        linestyle = "-" if enabled else ":"

        if self.ion_filter_patch is None or self.ion_filter_patch.axes is not self.ax_scatter_i:
            self.ion_filter_patch = Rectangle(
                (cx - 0.5 * width, cy - 0.5 * height),
                width,
                height,
                fill=False,
                linewidth=1.7,
                edgecolor=edgecolor,
                linestyle=linestyle,
                animated=True,
            )
            self.ion_filter_patch.set_gid("ion_filter_rect")
            self.ion_filter_patch.set_transform(self.ax_scatter_i.transData)
            self.ax_scatter_i.add_patch(self.ion_filter_patch)
        self.ion_filter_patch.set_xy((cx - 0.5 * width, cy - 0.5 * height))
        self.ion_filter_patch.set_width(width)
        self.ion_filter_patch.set_height(height)
        self.ion_filter_patch.set_edgecolor(edgecolor)
        self.ion_filter_patch.set_linestyle(linestyle)
        if self.ion_filter_center_marker is None or self.ion_filter_center_marker.axes is not self.ax_scatter_i:
            (self.ion_filter_center_marker,) = self.ax_scatter_i.plot(
                [cx], [cy], marker="o", markersize=4.3, color=edgecolor, linestyle="", animated=True
            )
            self.ion_filter_center_marker.set_gid("ion_filter_center")
        self.ion_filter_center_marker.set_data([cx], [cy])
        self.ion_filter_center_marker.set_color(edgecolor)

    # ------------------------------------------------------------------
    # Fast redraw helpers for interactive overlays
    # ------------------------------------------------------------------
    def _blit_overlays(self) -> None:
        """Redraw only overlay artists (ring/rectangle) using cached backgrounds."""
        if self.bg_scatter_e is None or self.bg_scatter_i is None:
            self.canvas.draw_idle()
            return
        try:
            self.canvas.restore_region(self.bg_scatter_e)
            self.canvas.restore_region(self.bg_scatter_i)
            if self.inner_ring_patch is not None and self.inner_ring_patch.axes is self.ax_scatter_e:
                self.ax_scatter_e.draw_artist(self.inner_ring_patch)
            if self.outer_ring_patch is not None and self.outer_ring_patch.axes is self.ax_scatter_e:
                self.ax_scatter_e.draw_artist(self.outer_ring_patch)
            if self.circle_center_marker is not None and self.circle_center_marker.axes is self.ax_scatter_e:
                self.ax_scatter_e.draw_artist(self.circle_center_marker)

            if self.ion_filter_patch is not None and self.ion_filter_patch.axes is self.ax_scatter_i:
                self.ax_scatter_i.draw_artist(self.ion_filter_patch)
            if self.ion_filter_center_marker is not None and self.ion_filter_center_marker.axes is self.ax_scatter_i:
                self.ax_scatter_i.draw_artist(self.ion_filter_center_marker)

            self.canvas.blit(self.ax_scatter_e.bbox)
            self.canvas.blit(self.ax_scatter_i.bbox)
        except Exception:
            self._invalidate_blit_background()
            self.canvas.draw_idle()

    def _update_circle_overlay_only(self) -> None:
        """Update only electron ring overlay without recomputing data plots."""
        if self.ax_scatter_e is None:
            return
        self._enforce_square_axis(self.ax_scatter_e)
        with contextlib.suppress(Exception):
            self.ax_scatter_e.apply_aspect()
        self._draw_circle_overlay()
        if self.bg_scatter_e is None or self.bg_scatter_i is None:
            self._rebuild_blit_background()
            return
        self._blit_overlays()

    def _update_ion_overlay_only(self) -> None:
        """Update only ion filter overlay without recomputing data plots."""
        if self.ax_scatter_i is None:
            return
        self._enforce_square_axis(self.ax_scatter_i)
        with contextlib.suppress(Exception):
            self.ax_scatter_i.apply_aspect()
        self._draw_ion_overlay()
        if self.bg_scatter_e is None or self.bg_scatter_i is None:
            self._rebuild_blit_background()
            return
        self._blit_overlays()

    def _enforce_square_axis(self, ax) -> None:
        """Keep axis box visually square."""
        ax.set_aspect("equal", adjustable="box")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

    def _lock_equal_xy_limits(self, ax) -> None:
        """Force equal data scales for x/y so circles stay circles."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            return
        half = 0.5 * max(abs(x1 - x0), abs(y1 - y0), 1e-9)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        ax.set_xlim(cx - half, cx + half, auto=False)
        ax.set_ylim(cy - half, cy + half, auto=False)
        self._enforce_square_axis(ax)
        with contextlib.suppress(Exception):
            ax.apply_aspect()

    def _count_points_in_inner_ring(self, points_xy: np.ndarray) -> int | None:
        """Count points inside current electron inner ring."""
        circle = self._parse_circle_params(show_dialog=False)
        if circle is None:
            return None
        if points_xy.shape[0] == 0:
            return 0
        cx, cy, inner_r, _outer_r = circle
        if self.preview_circle_center is not None:
            cx, cy = self.preview_circle_center
        dist2 = (points_xy[:, 0] - cx) ** 2 + (points_xy[:, 1] - cy) ** 2
        return int(np.count_nonzero(dist2 <= inner_r**2))

    def _count_points_in_ion_filter(self, points_xy: np.ndarray) -> int | None:
        """Count points inside current ion rectangle filter."""
        if not self.ion_filter_enable_checkbox.isChecked():
            return int(points_xy.shape[0])
        ion_rect = self._parse_ion_filter_params(show_dialog=False)
        if ion_rect is None:
            return None
        if points_xy.shape[0] == 0:
            return 0
        cx, cy, width, height = ion_rect
        if self.preview_ion_center is not None:
            cx, cy = self.preview_ion_center
        inside = (
            (points_xy[:, 0] >= cx - 0.5 * width)
            & (points_xy[:, 0] <= cx + 0.5 * width)
            & (points_xy[:, 1] >= cy - 0.5 * height)
            & (points_xy[:, 1] <= cy + 0.5 * height)
        )
        return int(np.count_nonzero(inside))

    def _plot_density_scatter(
        self,
        ax,
        points: np.ndarray,
        title: str,
        cmap: str,
        density_bin_size: float | None,
        empty_text: str,
    ) -> int:
        """Draw density-colored scatter; return number of actually plotted points."""
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.2)
        self._enforce_square_axis(ax)

        count = points.shape[0]
        if count == 0:
            ax.text(0.5, 0.5, empty_text, transform=ax.transAxes, ha="center", va="center")
            return 0

        if count > MAX_SCATTER_POINTS:
            step = int(np.ceil(count / MAX_SCATTER_POINTS))
            points = points[::step]
            if points.shape[0] > MAX_SCATTER_POINTS:
                points = points[:MAX_SCATTER_POINTS]
            plotted = points.shape[0]
        else:
            plotted = count

        x = points[:, 0].astype(np.float32, copy=False)
        y = points[:, 1].astype(np.float32, copy=False)
        dens = density_counts_from_bins(x, y, bin_size=density_bin_size)
        dmin = max(1.0, float(np.min(dens)))
        dmax = float(np.max(dens))
        norm = PowerNorm(gamma=0.65, vmin=dmin, vmax=dmax) if dmax > dmin else None
        ax.scatter(
            x,
            y,
            c=dens,
            cmap=cmap,
            norm=norm,
            s=4,
            alpha=0.9,
            linewidths=0,
            marker=",",
            rasterized=True,
            antialiased=False,
        )
        self._lock_equal_xy_limits(ax)
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        return plotted

    def _plot_centered_bin_image(self, ax, bin_data: dict | None, bin_size: float) -> None:
        """Draw centered electron projection image after ring-based denoising."""
        ax.clear()
        overlap_count = int(bin_data.get("signal_count", 0)) if bin_data is not None else 0
        if bin_data is not None:
            h2d_shape = np.asarray(bin_data.get("hist_denoised", np.zeros((0, 0)))).shape
            if len(h2d_shape) == 2:
                pixel_text = f"{int(h2d_shape[0])}x{int(h2d_shape[1])}"
            else:
                pixel_text = "n/a"
        else:
            pixel_text = "n/a"
        ax.set_title(
            f"Centered Electron Bin Map (ring-denoised, bin={bin_size:g}, overlap={overlap_count}, pixels={pixel_text})"
        )
        ax.set_xlabel("x centered")
        ax.set_ylabel("y centered")
        ax.grid(alpha=0.15)
        self._enforce_square_axis(ax)

        if bin_data is None:
            ax.text(
                0.5,
                0.5,
                "Click 'Apply Ring Selection and Center' first",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        h2d = bin_data["hist_denoised"]
        xedges = bin_data["xedges"]
        yedges = bin_data["yedges"]
        ax.imshow(
            h2d.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot",
            interpolation="nearest",
            aspect="equal",
        )
        ax.axhline(0.0, color="#8f8f8f", linewidth=1.0, alpha=0.7)
        ax.axvline(0.0, color="#8f8f8f", linewidth=1.0, alpha=0.7)
        ax.text(
            0.01,
            0.99,
            (
                f"overlap={int(bin_data.get('signal_count', 0))}\n"
                f"pixels={h2d.shape[0]}x{h2d.shape[1]}\n"
                f"signal={float(np.sum(bin_data.get('hist_signal', 0))):.1f}\n"
                f"removed={float(bin_data.get('removed_total', 0.0)):.1f}\n"
                f"outer_noise={float(bin_data.get('outer_noise_count', 0.0)):.1f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.25},
            color="white",
        )

    def _run_reconstructions_from_centered_data(self, bin_size: float) -> None:
        """Run both reconstruction methods from currently centered/denoised image."""
        _ = bin_size
        rbasex_settings = self._get_rbasex_settings()
        backward_settings = self._get_backward_settings()
        self.rbasex_recon_result, self.backward_recon_result = run_reconstructions_from_centered_data(
            self.centered_hist_data,
            rbasex_settings,
            backward_settings,
        )

    def _plot_reconstruction_panel(self, ax, title: str, result: dict | None) -> None:
        """Render one reconstruction panel (image + recovered peak text)."""
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("x centered")
        ax.set_ylabel("y centered")
        ax.grid(alpha=0.15)
        self._enforce_square_axis(ax)
        if result is None:
            ax.text(0.5, 0.5, "Click Start Reconstruction", transform=ax.transAxes, ha="center", va="center")
            return
        if result.get("error"):
            ax.text(
                0.03,
                0.97,
                f"Reconstruction error:\n{result['error']}",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
            return
        img = result.get("image")
        extent = result.get("extent")
        if img is None or extent is None:
            ax.text(0.5, 0.5, "No reconstruction output", transform=ax.transAxes, ha="center", va="center")
            return
        img_view = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        pct = float(result.get("display_percentile", 99.7))
        pct = min(max(pct, 50.0), 100.0)
        vmax = float(np.percentile(img_view, pct)) if img_view.size else 0.0
        if vmax <= 0:
            vmax = float(np.max(img_view)) if img_view.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        ax.imshow(
            img_view,
            origin="lower",
            extent=[extent[0], extent[1], extent[2], extent[3]],
            cmap="hot",
            interpolation="nearest",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
        )
        ax.text(
            0.02,
            0.98,
            format_peak_text(result.get("peaks", [])),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.35},
            color="white",
        )
        footer = f"display pctl={pct:.3g}"
        if "baseline_used" in result:
            footer += f", baseline(total)={float(result.get('baseline_used', 0.0)):.3g}"
        if "baseline_external" in result:
            footer += f", ext={float(result.get('baseline_external', 0.0)):.3g}"
        if "baseline_internal" in result:
            footer += f", int={float(result.get('baseline_internal', 0.0)):.3g}"
        if footer:
            ax.text(
                0.02,
                0.03,
                footer,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.25},
                color="white",
            )

    def _plot_info_panel(self, selected_count: int) -> None:
        """Render summary panel text for current selection and reconstruction state."""
        self.ax_info.clear()
        self.ax_info.set_title("Selection Summary")
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.set_frame_on(True)

        lines = [
            f"Matched pairs: {self.matched_ion.shape[0]}",
            f"Ion t selected: {selected_count}",
            f"Ion filter: {'ON' if self.ion_filter_enable_checkbox.isChecked() else 'OFF'}",
            f"Ion-rectangle intersection events: {self.ion_filter_selected_electron.shape[0]}",
            (
                f"Ion major-axis align: ON (auto={self.ion_auto_angle_deg:.3f} deg, total={self.ion_rotation_angle_deg:.3f} deg)"
                if self.ion_align_checkbox.isChecked()
                else f"Ion major-axis align: OFF (manual={self.ion_user_rotation_deg:.3f} deg)"
            ),
            (
                f"Ion main-direction line: ON (angle={self.ion_main_axis_angle_deg:.3f} deg)"
                if self.ion_dirline_btn.isChecked() and self.ion_main_axis_angle_deg is not None
                else f"Ion main-direction line: {'ON' if self.ion_dirline_btn.isChecked() else 'OFF'}"
            ),
            f"Inner-ring electrons: {self.ring_inner_selected_electron.shape[0]}",
            f"Outer-ring noise points: {self.ring_outer_noise_electron.shape[0]}",
            f"Center estimator: {self.center_mode_combo.currentText()}",
            f"Auto recenter on apply: {'ON' if self.auto_recenter_checkbox.isChecked() else 'OFF'}",
        ]
        rb_n = len(self.rbasex_recon_result.get("peaks", [])) if self.rbasex_recon_result else 0
        bw_n = len(self.backward_recon_result.get("peaks", [])) if self.backward_recon_result else 0
        lines.append(f"rBasex peaks: {rb_n}")
        lines.append(f"Backward(no forward-fit) peaks: {bw_n}")
        if self.circle_centroid is not None:
            lines.append(
                "Estimated center: "
                f"({self.circle_centroid[0]:.6g}, {self.circle_centroid[1]:.6g})"
            )
        if self.center_residual is not None:
            lines.append(
                "Centered mean residual: "
                f"({self.center_residual[0]:.3e}, {self.center_residual[1]:.3e})"
            )
        if self.centered_hist_data is not None:
            lines.append(f"Estimated removed noise: {self.centered_hist_data.get('removed_total', 0.0):.3f}")
            lines.append(
                f"Expected inner noise from ring: {self.centered_hist_data.get('expected_inner_noise_total', 0.0):.3f}"
            )

        self.ax_info.text(0.03, 0.97, "\n".join(lines), transform=self.ax_info.transAxes, ha="left", va="top")

    def _refresh_reconstruction_panels_only(self) -> None:
        """Refresh summary/projection/reconstruction panels only (fast post-reconstruct path)."""
        mask = self._selected_mask()
        selected_count = int(mask.sum())
        self._plot_info_panel(selected_count)
        centered_bin_size = self._parse_center_bin_size(show_dialog=False)
        if centered_bin_size is None:
            centered_bin_size = 0.1
        self._plot_centered_bin_image(self.ax_centered_bin, self.centered_hist_data, centered_bin_size)
        if self.circle_centroid is not None:
            self.ax_centered_bin.text(
                0.01,
                0.03,
                f"centroid=({self.circle_centroid[0]:.4g}, {self.circle_centroid[1]:.4g})",
                transform=self.ax_centered_bin.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="white",
                bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.25},
            )
        self._plot_reconstruction_panel(self.ax_reserved_top, "rBasex Reconstruction", self.rbasex_recon_result)
        self._plot_reconstruction_panel(
            self.ax_reserved_bottom,
            "Backward Reconstruction (no forward-fit)",
            self.backward_recon_result,
        )
        overlay_artists = []
        for artist in (
            self.inner_ring_patch,
            self.outer_ring_patch,
            self.circle_center_marker,
            self.ion_filter_patch,
            self.ion_filter_center_marker,
        ):
            if artist is not None:
                with contextlib.suppress(Exception):
                    overlay_artists.append((artist, artist.get_visible()))
                    artist.set_visible(False)
        self.canvas.draw()
        self._capture_blit_background_from_current_canvas()
        for artist, visible in overlay_artists:
            with contextlib.suppress(Exception):
                artist.set_visible(visible)
        self._update_circle_overlay_only()
        self._update_ion_overlay_only()

    def _refresh_plots(self) -> None:
        """Full refresh of all panels based on current data and filter settings."""
        self._clear_circle_overlay_artists()
        self._clear_ion_overlay_artists()
        self._refresh_ion_histogram_only(draw_canvas=False)

        mask = self._selected_mask()
        selected_count = int(mask.sum())
        electron_show = self.matched_electron[mask] if mask.size else np.empty((0, 3))
        ion_show_raw = self.matched_ion[mask] if mask.size else np.empty((0, 3))
        self._update_ion_rotation_from_points(ion_show_raw[:, :2] if ion_show_raw.size else np.empty((0, 2)))
        ion_show = ion_show_raw.copy()
        if ion_show.size:
            ion_show[:, :2] = self._transform_ion_xy(ion_show_raw[:, :2])
        self._plot_info_panel(selected_count)
        electron_inner_count = self._count_points_in_inner_ring(electron_show[:, :2] if electron_show.size else np.empty((0, 2)))
        ion_filter_count = self._count_points_in_ion_filter(ion_show[:, :2] if ion_show.size else np.empty((0, 2)))
        e_inner_text = "n/a" if electron_inner_count is None else f"{electron_inner_count}"
        if self.ion_filter_enable_checkbox.isChecked():
            i_filter_text = "n/a" if ion_filter_count is None else f"{ion_filter_count}"
            ion_title = (
                f"Ion Scatter (selected={ion_show.shape[0]}, filter-in={i_filter_text}"
                + (
                    f", rotated={self.ion_rotation_angle_deg:.3f} deg)"
                    if self.ion_align_checkbox.isChecked()
                    else ")"
                )
            )
        else:
            ion_title = (
                f"Ion Scatter (selected={ion_show.shape[0]}"
                + (
                    f", rotated={self.ion_rotation_angle_deg:.3f} deg)"
                    if self.ion_align_checkbox.isChecked()
                    else ")"
                )
            )

        plotted_e = self._plot_density_scatter(
            self.ax_scatter_e,
            electron_show,
            f"Electron Scatter (selected={electron_show.shape[0]}, inner-ring={e_inner_text})",
            cmap="inferno",
            density_bin_size=None,
            empty_text="No points",
        )

        plotted_i = self._plot_density_scatter(
            self.ax_scatter_i,
            ion_show,
            ion_title,
            cmap="inferno",
            density_bin_size=None,
            empty_text="No points",
        )
        if self.ion_dirline_btn.isChecked():
            lim_x = self.ax_scatter_i.get_xlim()
            lim_y = self.ax_scatter_i.get_ylim()
            self.ax_scatter_i.set_autoscalex_on(False)
            self.ax_scatter_i.set_autoscaley_on(False)
            self._draw_ion_main_direction_line(ion_show[:, :2] if ion_show.size else np.empty((0, 2)))
            self.ax_scatter_i.set_xlim(lim_x[0], lim_x[1], auto=False)
            self.ax_scatter_i.set_ylim(lim_y[0], lim_y[1], auto=False)
        else:
            self._draw_ion_main_direction_line(np.empty((0, 2)))

        centered_bin_size = self._parse_center_bin_size(show_dialog=False)
        if centered_bin_size is None:
            centered_bin_size = 0.1

        self._plot_centered_bin_image(self.ax_centered_bin, self.centered_hist_data, centered_bin_size)
        if self.circle_centroid is not None:
            self.ax_centered_bin.text(
                0.01,
                0.03,
                f"centroid=({self.circle_centroid[0]:.4g}, {self.circle_centroid[1]:.4g})",
                transform=self.ax_centered_bin.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="white",
                bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.25},
            )

        self._plot_reconstruction_panel(self.ax_reserved_top, "rBasex Reconstruction", self.rbasex_recon_result)
        self._plot_reconstruction_panel(
            self.ax_reserved_bottom,
            "Backward Reconstruction (no forward-fit)",
            self.backward_recon_result,
        )

        if electron_show.shape[0] > plotted_e:
            self.ax_scatter_e.text(
                0.01,
                0.99,
                f"Plotted {plotted_e}/{electron_show.shape[0]}",
                transform=self.ax_scatter_e.transAxes,
                ha="left",
                va="top",
            )
        if ion_show.shape[0] > plotted_i:
            self.ax_scatter_i.text(
                0.01,
                0.99,
                f"Plotted {plotted_i}/{ion_show.shape[0]}",
                transform=self.ax_scatter_i.transAxes,
                ha="left",
                va="top",
            )
        self.canvas.draw()
        self._capture_blit_background_from_current_canvas()
        self._update_circle_overlay_only()
        self._update_ion_overlay_only()

    def _draw_placeholder(self) -> None:
        """Draw initial empty-state panels before any data is processed."""
        self._clear_circle_overlay_artists()
        self._clear_ion_overlay_artists()
        self.ax_hist_ion.clear()
        self.ax_info.clear()
        self.ax_scatter_e.clear()
        self.ax_scatter_i.clear()
        self.ax_centered_bin.clear()
        self.ax_reserved_top.clear()
        self.ax_reserved_bottom.clear()

        self.ax_hist_ion.set_title("Ion t Histogram")
        self.ax_hist_ion.set_xlabel("t (ns)")
        self.ax_hist_ion.set_ylabel("counts")
        self.ax_hist_ion.grid(alpha=0.2)
        self._sync_hist_roi_inputs()
        self.ax_info.set_title("Selection Summary")
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.text(0.03, 0.97, "No processed data", transform=self.ax_info.transAxes, ha="left", va="top")

        self.ax_scatter_e.set_title("Electron Scatter")
        self.ax_scatter_e.set_xlabel("x")
        self.ax_scatter_e.set_ylabel("y")
        self.ax_scatter_e.grid(alpha=0.2)
        self._enforce_square_axis(self.ax_scatter_e)

        self.ax_scatter_i.set_title("Ion Scatter")
        self.ax_scatter_i.set_xlabel("x")
        self.ax_scatter_i.set_ylabel("y")
        self.ax_scatter_i.grid(alpha=0.2)
        self._enforce_square_axis(self.ax_scatter_i)

        centered_bin_size = self._parse_center_bin_size(show_dialog=False)
        if centered_bin_size is None:
            centered_bin_size = 0.1
        self._plot_centered_bin_image(self.ax_centered_bin, None, centered_bin_size)
        self._plot_reconstruction_panel(self.ax_reserved_top, "rBasex Reconstruction", None)
        self._plot_reconstruction_panel(self.ax_reserved_bottom, "Backward Reconstruction (no forward-fit)", None)

        self.canvas.draw()
        self._capture_blit_background_from_current_canvas()
        self._update_circle_overlay_only()
        self._update_ion_overlay_only()


def main() -> int:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("VMI_workflow")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
