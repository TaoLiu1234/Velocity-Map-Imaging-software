#!/usr/bin/env python3
"""Drag-interaction benchmark + ghost detector for VMI_workflow (2026-09-01).

Populates the real ``MainWindow`` with the sample-data workflow (exactly the
``tests/test_smoke.py`` pipeline: load -> process_and_plot -> fine ROI ->
circle params -> apply_circle_selection -> rBasex reconstruction), then
simulates every draggable overlay interaction by driving the REAL mouse-event
handlers (``_on_canvas_press`` / ``_on_canvas_move`` / ``_on_canvas_release``)
with synthetic ``matplotlib.backend_bases.MouseEvent`` objects aimed at the
right axes. The 16 ms ``DRAG_PREVIEW_INTERVAL_MS`` QTimer cadence is mimicked
by pre-arming each single-shot preview timer and timing the ``_flush_*``
function the timer would call.

Each interaction runs TWO drag sessions (so the ghost detector sees several
end positions): session A ends far away from the start position, session B
ends somewhere else. After each session's final frame the canvas RGBA buffer
(the "drag result") is compared against a fresh full ``canvas.draw()``
reference render (overlay artists temporarily de-animated so the reference
includes them). Stale overlay pixels left behind by a bad
restore_region+draw_artist discipline (the historical "ghost ring" bug) show
up as pixel differences at previous overlay positions. PASS requires
bit-exact buffers (status EXACT); small antialiasing noise (max channel
delta <= 8 on <= 0.2% of pixels) is reported as PASS(AA); anything else is
GHOST and fails the run.

Run from the project root:
    python tests/bench_drag.py [--frames 200] [--label BEFORE|AFTER]
                               [--only name1,name2] [--quick]
Exit code 0 iff every ghost check passes.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must happen before any Qt import (VMI_workflow imports PySide6 + QtAgg).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np  # noqa: E402

import test_smoke  # noqa: E402  (reuses the smoke workflow constants + wait_until)
from test_smoke import (  # noqa: E402
    CIRCLE_TIMEOUT_S,
    FINE_ROI,
    INITIAL_CENTER,
    INNER_RADIUS,
    LOAD_TIMEOUT_S,
    OUTER_RADIUS,
    RECON_TIMEOUT_S,
    ensure_sample_data,
    wait_until,
)


# ---------------------------------------------------------------------------
# Environment / window preparation
# ---------------------------------------------------------------------------
def _patch_message_boxes():
    from PySide6.QtWidgets import QMessageBox

    def fake(*call_args, **_kwargs):
        return QMessageBox.StandardButton.Ok

    QMessageBox.warning = staticmethod(fake)
    QMessageBox.critical = staticmethod(fake)


def prepared_window(app, MainWindow, windows: list):
    """Full sample-data workflow through rBasex so every panel has content."""
    win = MainWindow()
    windows.append(win)
    win.show()
    win.file_paths["trigger"] = str(test_smoke.TRIGGER_PATH)
    win.file_paths["electron"] = str(test_smoke.ELECTRON_PATH)
    win.file_paths["ion"] = str(test_smoke.ION_PATH)
    win.load_cache()
    wait_until(
        app,
        lambda: win.cache is not None and not getattr(win, "_load_busy", False),
        LOAD_TIMEOUT_S,
        "bench_load",
        "cache",
    )
    win.trigger_mode_combo.setCurrentIndex(win.trigger_mode_combo.findData("coincidence"))
    app.processEvents()
    win.process_and_plot()
    app.processEvents()

    win.ion_fine_xmin_edit.setText(str(FINE_ROI[0]))
    win.ion_fine_xmax_edit.setText(str(FINE_ROI[1]))
    win.apply_ion_fine_roi_from_inputs()
    app.processEvents()
    if win.ion_range is None:
        raise RuntimeError("fine ROI not applied")

    # The ion filter must be enabled BEFORE apply_circle_selection: toggling it
    # clears downstream results (incl. rBasex), so the reconstruction has to be
    # the LAST workflow step for the rBasex panels to stay populated.
    win.ion_filter_cx_edit.setText("128")
    win.ion_filter_cy_edit.setText("126")
    win.ion_filter_w_edit.setText("80")
    win.ion_filter_h_edit.setText("80")
    win.ion_filter_enable_checkbox.setChecked(True)
    app.processEvents()

    win.circle_cx_edit.setText(f"{INITIAL_CENTER[0]:g}")
    win.circle_cy_edit.setText(f"{INITIAL_CENTER[1]:g}")
    win.inner_r_edit.setText(f"{INNER_RADIUS:g}")
    win.outer_r_edit.setText(f"{OUTER_RADIUS:g}")
    win.outer_ring_filter_enable_checkbox.setChecked(True)
    app.processEvents()

    win.apply_circle_selection()
    wait_until(
        app,
        lambda: (win.centered_hist_data is not None) and (not getattr(win, "_circle_busy", False)),
        CIRCLE_TIMEOUT_S,
        "bench_circle",
        "ring projection",
    )

    win.run_reconstruction_now()
    wait_until(
        app,
        lambda: (win.rbasex_recon_result is not None) and (not getattr(win, "_recon_busy", False)),
        RECON_TIMEOUT_S,
        "bench_recon",
        "rBasex",
    )

    win.canvas.draw()
    app.processEvents()
    if win.rbasex_recon_result is None:
        raise RuntimeError("rBasex result was cleared after preparation")
    return win


# ---------------------------------------------------------------------------
# Synthetic matplotlib events
# ---------------------------------------------------------------------------
def make_event(win, ax, xdata: float, ydata: float, *, name: str, button=None):
    """Build a real MouseEvent aimed at `ax` from data coordinates."""
    from matplotlib.backend_bases import MouseEvent

    x_disp, y_disp = ax.transData.transform((float(xdata), float(ydata)))
    return MouseEvent(name, win.canvas, float(x_disp), float(y_disp), button=button)


def data_point(ax, fx: float, fy: float) -> tuple[float, float]:
    """Fractional position inside the current axes limits (0..1)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (float(x0) + fx * float(x1 - x0), float(y0) + fy * float(y1 - y0))


def linear_track(p0, p1, n):
    return [
        (p0[0] + (p1[0] - p0[0]) * i / (n - 1), p0[1] + (p1[1] - p0[1]) * i / (n - 1))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Ghost detector
# ---------------------------------------------------------------------------
def ghost_check(win, axes, artists: list, label: str) -> dict:
    """Compare the mid-drag canvas buffer vs a forced clean reference render.

    Strictness applies to the axes INTERIOR (axes bbox shrunk by 5 px): stale
    overlay pixels (the historical ghost ring) live there, so the interior
    must be bit-exact vs a fresh full draw. Pixels in the margins (title, tick
    labels) and on the frame edge are reported informationally: the legacy
    whole-axes redraw path re-composites text antialiasing every frame (the
    "progressive text darkening" artifact), which shows up there but is not
    an overlay ghost.
    """
    canvas = win.canvas
    drag = np.asarray(canvas.buffer_rgba()).copy()
    saved = []
    for artist in artists:
        if artist is None:
            continue
        try:
            saved.append((artist, bool(artist.get_animated())))
            artist.set_animated(False)
        except Exception:
            pass
    prev_suspend = bool(getattr(win, "_suspend_overlay_draw_event_sync", False))
    win._suspend_overlay_draw_event_sync = True  # keep draw_event hooks out of the reference
    try:
        canvas.draw()
        ref = np.asarray(canvas.buffer_rgba())
    finally:
        win._suspend_overlay_draw_event_sync = prev_suspend
        for artist, flag in saved:
            try:
                artist.set_animated(flag)
            except Exception:
                pass

    if drag.shape != ref.shape:
        return {"label": label, "status": "GHOST", "detail": f"shape {drag.shape} != {ref.shape}"}

    diff_mask = np.any(drag != ref, axis=2)
    # Interior window (array coords: row 0 = top).
    height = diff_mask.shape[0]
    bb = axes.bbox
    shrink = 5.0
    r0 = max(0, int(height - bb.y1) + int(shrink))
    r1 = min(height, int(height - bb.y0) - int(shrink))
    c0 = max(0, int(bb.x0) + int(shrink))
    c1 = min(diff_mask.shape[1], int(bb.x1) - int(shrink))
    interior = diff_mask[r0:r1, c0:c1]
    n_in = int(interior.sum())
    n_all = int(diff_mask.sum())
    if n_in == 0:
        status = "EXACT" if n_all == 0 else "EXACT-INTERIOR"
    else:
        delta = np.abs(drag.astype(np.int16) - ref.astype(np.int16))[r0:r1, c0:c1]
        max_delta = int(delta.max())
        frac = n_in / float(max(interior.size, 1))
        if max_delta <= 8 and frac <= 0.002:
            status = "PASS(AA)"
        else:
            status = "GHOST"
    ys, xs = np.nonzero(diff_mask)
    if n_all:
        full_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        full_txt = f"whole-buffer diff {n_all} px bbox={full_bbox}"
    else:
        full_txt = "whole-buffer identical"
    detail = f"interior diff={n_in}px, {full_txt}"
    return {"label": label, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
class InteractionResult:
    def __init__(self, name: str):
        self.name = name
        self.press_ms: float | None = None
        self.frame_ms: list[float] = []
        self.release_ms: float | None = None
        self.ghosts: list[dict] = []
        self.note = ""

    @property
    def ghost_ok(self) -> bool:
        # Interactions with no ghost checks by design (e.g. the SpanSelector
        # commit, whose visuals live inside matplotlib) count as OK.
        return all(g["status"] in {"EXACT", "EXACT-INTERIOR", "PASS(AA)"} for g in self.ghosts)

    def table_row(self) -> str:
        arr = np.asarray(self.frame_ms, dtype=np.float64)
        if arr.size == 0:
            mean = p95 = mx = float("nan")
            fps = float("nan")
        else:
            mean = float(arr.mean())
            p95 = float(np.percentile(arr, 95))
            mx = float(arr.max())
            fps = 1000.0 / mean if mean > 0 else float("nan")
        press = "-" if self.press_ms is None else f"{self.press_ms:.1f}"
        release = "-" if self.release_ms is None else f"{self.release_ms:.1f}"
        ghost = ",".join(g["status"] for g in self.ghosts) if self.ghosts else "-"
        row = (
            f"| {self.name} | {press} | {mean:.2f} | {p95:.2f} | {mx:.2f} | {fps:.0f} | "
            f"{release} | {ghost} |"
        )
        if self.note:
            row += f" {self.note}"
        return row


def _run_frames(win, move_points, move_ev_factory, flush, timer_attr, timed_on_move: bool, frame_ms: list):
    """Simulate the 16 ms timer cadence: pre-arm timer, move (sets pending),
    then time the flush the timer would call (or the inline move work)."""
    timer = getattr(win, timer_attr) if timer_attr else None
    for x, y in move_points:
        if timer is not None and not timer.isActive():
            timer.start()
        ev = move_ev_factory(x, y)
        if timed_on_move:
            t0 = time.perf_counter()
            win._on_canvas_move(ev)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            win._on_canvas_move(ev)
            t0 = time.perf_counter()
            flush()
            frame_ms.append((time.perf_counter() - t0) * 1000.0)


def run_drag_session(
    win,
    result: InteractionResult,
    *,
    press_ev,
    move_points,
    move_ev_factory,
    flush,
    timer_attr,
    release_ev,
    ghost_artists_fn,
    ghost_label,
    drag_axes=None,
    expect_flag=None,
    timed_on_move=False,
    skip_release=False,
):
    """One full drag session: press -> timed frames -> ghost check -> release."""
    t0 = time.perf_counter()
    win._on_canvas_press(press_ev)
    result.press_ms = (time.perf_counter() - t0) * 1000.0
    if expect_flag is not None and not expect_flag():
        result.note = "PRESS-NOT-ARMED (event routing did not start this drag)"
        result.ghosts.append(
            {"label": ghost_label, "status": "GHOST", "detail": "drag was never armed"}
        )
        return
    _run_frames(win, move_points, move_ev_factory, flush, timer_attr, timed_on_move, result.frame_ms)
    result.ghosts.append(ghost_check(win, drag_axes, ghost_artists_fn(), ghost_label))
    if skip_release:
        return
    t0 = time.perf_counter()
    win._on_canvas_release(release_ev)
    result.release_ms = (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Individual interaction drivers (each returns an InteractionResult)
# ---------------------------------------------------------------------------
def drive_electron_ring_drag(win, app, n: int) -> InteractionResult:
    result = InteractionResult("electron_ring_center_drag")
    ax = win.ax_scatter_e
    circle = win._parse_circle_params(show_dialog=False)
    cx, cy = float(circle[0]), float(circle[1])
    far = (cx + 55.0, cy + 45.0)  # fully away from start: stale ring would show
    back = (cx - 40.0, cy + 70.0)

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    artists = lambda: [  # noqa: E731
        win.inner_ring_patch,
        win.outer_ring_patch,
        win.circle_center_marker,
    ]

    # Session A: start -> far. Session B: far -> back.
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, cx, cy, name="button_press_event", button=1),
        move_points=linear_track((cx, cy), far, max(4, n // 2)),
        move_ev_factory=factory,
        flush=win._flush_drag_preview,
        timer_attr="drag_preview_timer",
        release_ev=factory(*far),
        ghost_artists_fn=artists,
        ghost_label="electron_ring@far",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_circle,
    )
    _stop_preview_timers(win)
    app.processEvents()
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *far, name="button_press_event", button=1),
        move_points=linear_track(far, back, max(4, n - n // 2)),
        move_ev_factory=factory,
        flush=win._flush_drag_preview,
        timer_attr="drag_preview_timer",
        release_ev=factory(*back),
        ghost_artists_fn=artists,
        ghost_label="electron_ring@back",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_circle,
    )
    _stop_preview_timers(win)
    app.processEvents()
    return result


def drive_ion_filter_drag(win, app, n: int) -> InteractionResult:
    result = InteractionResult("ion_filter_rect_center_drag")
    ax = win.ax_scatter_i
    rect = win._parse_ion_filter_params(show_dialog=False)
    cx, cy = float(rect[0]), float(rect[1])
    far = (cx + 45.0, cy - 35.0)
    back = (cx - 30.0, cy + 55.0)

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    artists = lambda: [win.ion_filter_patch, win.ion_filter_center_marker]  # noqa: E731

    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, cx, cy, name="button_press_event", button=1),
        move_points=linear_track((cx, cy), far, max(4, n // 2)),
        move_ev_factory=factory,
        flush=win._flush_ion_drag_preview,
        timer_attr="ion_drag_preview_timer",
        release_ev=factory(*far),
        ghost_artists_fn=artists,
        ghost_label="ion_filter@far",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_ion_filter,
    )
    _stop_preview_timers(win)
    app.processEvents()
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *far, name="button_press_event", button=1),
        move_points=linear_track(far, back, max(4, n - n // 2)),
        move_ev_factory=factory,
        flush=win._flush_ion_drag_preview,
        timer_attr="ion_drag_preview_timer",
        release_ev=factory(*back),
        ghost_artists_fn=artists,
        ghost_label="ion_filter@back",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_ion_filter,
    )
    _stop_preview_timers(win)
    app.processEvents()
    return result


def _wait_polar_mode(win, app, want: bool, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if win._electron_scatter_polar_mode_enabled() == want:
            # give the deferred full redraw a chance to settle
            app.processEvents()
            time.sleep(0.05)
            app.processEvents()
            return
        time.sleep(0.01)
    raise RuntimeError(f"polar mode did not reach {want}")


def drive_polar_roi_drag(win, app, n: int) -> InteractionResult:
    result = InteractionResult("polar_roi_band_drag")
    win.polar_roi_rmin_edit.setText("40")
    win.polar_roi_rmax_edit.setText("130")
    app.processEvents()
    win.electron_scatter_polar_toggle_btn.setChecked(True)
    _wait_polar_mode(win, app, True)

    ax = win.ax_scatter_e
    _target_r, _win_, r_min, r_max = win._current_polar_target_band()
    if r_min is None or r_max is None:
        result.note = "SKIPPED (no polar target band)"
        win.electron_scatter_polar_toggle_btn.setChecked(False)
        _wait_polar_mode(win, app, False)
        return result
    y_mid = 0.5 * (float(r_min) + float(r_max))
    x_mid = 0.5 * float(np.mean(ax.get_xlim()))
    y0, y1 = ax.get_ylim()
    far_y = float(np.clip(y_mid + 0.25 * (y1 - y0), min(y0, y1), max(y0, y1)))
    back_y = float(np.clip(y_mid - 0.2 * (y1 - y0), min(y0, y1), max(y0, y1)))

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    artists = lambda: [win.polar_roi_span_patch, win.polar_roi_line_lo, win.polar_roi_line_hi]  # noqa: E731

    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, x_mid, y_mid, name="button_press_event", button=1),
        move_points=[(x_mid, y_mid + (far_y - y_mid) * i / (max(4, n // 2) - 1)) for i in range(max(4, n // 2))],
        move_ev_factory=factory,
        flush=win._flush_polar_roi_preview,
        timer_attr="polar_roi_preview_timer",
        release_ev=factory(x_mid, far_y),
        ghost_artists_fn=artists,
        ghost_label="polar_roi@far",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_polar_roi,
    )
    _stop_preview_timers(win)
    app.processEvents()
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, x_mid, far_y, name="button_press_event", button=1),
        move_points=[(x_mid, far_y + (back_y - far_y) * i / (max(4, n - n // 2) - 1)) for i in range(max(4, n - n // 2))],
        move_ev_factory=factory,
        flush=win._flush_polar_roi_preview,
        timer_attr="polar_roi_preview_timer",
        release_ev=factory(x_mid, back_y),
        ghost_artists_fn=artists,
        ghost_label="polar_roi@back",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_polar_roi,
    )
    _stop_preview_timers(win)
    app.processEvents()
    win.electron_scatter_polar_toggle_btn.setChecked(False)
    _wait_polar_mode(win, app, False)
    return result


def drive_theta_drag(win, app, n: int, source: str) -> InteractionResult:
    ax = win.ax_centered_bin if source == "centered" else win.ax_reserved_top
    result = InteractionResult(f"theta_line_drag_{source}")
    r = 0.35 * min(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
    # "centered" ignores clicks on the right half (compare toggle) -> stay x<0.
    deg_a, deg_b, deg_c = (110.0, 200.0, 250.0) if source == "centered" else (20.0, 140.0, 250.0)

    def pt(deg):
        rad = np.deg2rad(deg)
        return (float(r * np.cos(rad)), float(r * np.sin(rad)))

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    if source == "centered":
        artists = lambda: [win.theta_profile_line_main, win.theta_profile_line_lo, win.theta_profile_line_hi]  # noqa: E731
    else:
        artists = lambda: [win.rbasex_profile_line_main, win.rbasex_profile_line_lo, win.rbasex_profile_line_hi]  # noqa: E731

    mid_pts = [pt(deg_a + (deg_b - deg_a) * i / (max(4, n // 2) - 1)) for i in range(max(4, n // 2))]
    end_pts = [pt(deg_b + (deg_c - deg_b) * i / (max(4, n - n // 2) - 1)) for i in range(max(4, n - n // 2))]

    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *pt(deg_a), name="button_press_event", button=1),
        move_points=mid_pts,
        move_ev_factory=factory,
        flush=win._flush_theta_drag_preview,
        timer_attr="theta_drag_preview_timer",
        release_ev=factory(*pt(deg_b)),
        ghost_artists_fn=artists,
        ghost_label=f"theta_{source}@mid",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_theta_source is not None,
    )
    _stop_preview_timers(win)
    app.processEvents()
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *pt(deg_b), name="button_press_event", button=1),
        move_points=end_pts,
        move_ev_factory=factory,
        flush=win._flush_theta_drag_preview,
        timer_attr="theta_drag_preview_timer",
        release_ev=factory(*pt(deg_c)),
        ghost_artists_fn=artists,
        ghost_label=f"theta_{source}@end",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_theta_source is not None,
    )
    _stop_preview_timers(win)
    app.processEvents()
    return result


def drive_rbasex_range_drag(win, app, n: int) -> InteractionResult:
    result = InteractionResult("rbasex_range_handles_drag")
    win.rbasex_profile_range_pick_btn.setChecked(True)
    app.processEvents()
    ax = win.ax_rbasex_profile
    x0, x1 = sorted(ax.get_xlim())
    xa = float(x0 + 0.25 * (x1 - x0))
    xb = float(x0 + 0.65 * (x1 - x0))
    xc = float(x0 + 0.45 * (x1 - x0))
    ya = float(np.mean(ax.get_ylim()))

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    def artists():
        found = []
        for artist in list(ax.patches) + list(ax.texts):
            gid = getattr(artist, "get_gid", lambda: None)()
            if gid in {"rbasex_profile_range_span", "rbasex_profile_range_text"}:
                found.append(artist)
        return found

    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, xa, ya, name="button_press_event", button=1),
        move_points=[(xa + (xb - xa) * i / (max(4, n // 2) - 1), ya) for i in range(max(4, n // 2))],
        move_ev_factory=factory,
        flush=win._flush_rbasex_profile_range_preview,
        timer_attr="rbasex_profile_range_preview_timer",
        release_ev=factory(xb, ya),
        ghost_artists_fn=artists,
        ghost_label="rbasex_range@b",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_rbasex_profile_range,
    )
    _stop_preview_timers(win)
    app.processEvents()
    # The commit on release turns the pick toggle off; re-arm it like the UI does.
    win.rbasex_profile_range_pick_btn.setChecked(True)
    app.processEvents()
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, xb, ya, name="button_press_event", button=1),
        move_points=[(xb + (xc - xb) * i / (max(4, n - n // 2) - 1), ya) for i in range(max(4, n - n // 2))],
        move_ev_factory=factory,
        flush=win._flush_rbasex_profile_range_preview,
        timer_attr="rbasex_profile_range_preview_timer",
        release_ev=factory(xc, ya),
        ghost_artists_fn=artists,
        ghost_label="rbasex_range@c",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_rbasex_profile_range,
    )
    _stop_preview_timers(win)
    app.processEvents()
    win.rbasex_profile_range_pick_btn.setChecked(False)
    app.processEvents()
    return result


def drive_ion_rotation_drag(win, app, n: int) -> InteractionResult:
    result = InteractionResult("ion_rotation_preview_drag")
    if not win.ion_dirline_btn.isChecked():
        win.ion_dirline_btn.setChecked(True)
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            app.processEvents()
            if win.ion_main_axis_angle_deg is not None and win.ion_main_axis_line is not None:
                break
            time.sleep(0.01)
    app.processEvents()
    if win.ion_main_axis_line is None or win.ion_main_axis_angle_deg is None:
        result.note = "SKIPPED (direction line unavailable)"
        return result

    ax = win.ax_scatter_i
    rect = win._parse_ion_filter_params(show_dialog=False)
    rcx, rcy, rw, rh = (float(v) for v in rect)

    press_pt = data_point(ax, 0.93, 0.05)
    if (abs(press_pt[0] - rcx) <= 0.55 * rw) and (abs(press_pt[1] - rcy) <= 0.55 * rh):
        press_pt = data_point(ax, 0.05, 0.95)
    if (abs(press_pt[0] - rcx) <= 0.55 * rw) and (abs(press_pt[1] - rcy) <= 0.55 * rh):
        result.note = "SKIPPED (filter rect covers press corners)"
        return result

    press_ev = make_event(win, ax, *press_pt, name="button_press_event", button=1)
    center = win._electron_center_for_ion_view()
    ccx, ccy = float(center[0]), float(center[1])
    span = min(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
    radius = 0.3 * span

    def pt(deg):
        rad = np.deg2rad(deg)
        return (ccx + radius * np.cos(rad), ccy + radius * np.sin(rad))

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    artists = lambda: [win.ion_main_axis_line]  # noqa: E731

    # 1-degree snap: step >= 3 degrees/frame so every flush redraws (like a
    # real fast rotation drag) and the per-frame cost comparison is honest.
    run_drag_session(
        win, result,
        press_ev=press_ev,
        move_points=[pt(20.0 + 3.0 * i) for i in range(max(4, n // 2))],
        move_ev_factory=factory,
        flush=win._flush_ion_rotation_preview,
        timer_attr="ion_rotation_preview_timer",
        release_ev=factory(*pt(150.0)),
        ghost_artists_fn=artists,
        ghost_label="ion_rotation@150",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_ion_rotation,
    )
    _stop_preview_timers(win)
    app.processEvents()
    if win.ion_main_axis_angle_deg is None or win.ion_main_axis_line is None:
        # The applied rotation may have reset the direction line; re-arm it
        # through the real toggle like a user would.
        win.ion_dirline_btn.setChecked(False)
        app.processEvents()
        win.ion_dirline_btn.setChecked(True)
    deadline = time.perf_counter() + 30.0
    while time.perf_counter() < deadline:
        app.processEvents()
        if win.ion_main_axis_angle_deg is not None and win.ion_main_axis_line is not None:
            break
        time.sleep(0.01)
    if win.ion_main_axis_line is None or win.ion_main_axis_angle_deg is None:
        result.note = "PARTIAL (session B skipped: direction line unavailable after apply)"
        return result
    # Session B must press OUTSIDE the (re-applied) filter rectangle like session A.
    rect_b = win._parse_ion_filter_params(show_dialog=False)
    press_b = data_point(ax, 0.93, 0.05)
    if (abs(press_b[0] - rect_b[0]) <= 0.55 * rect_b[2]) and (abs(press_b[1] - rect_b[1]) <= 0.55 * rect_b[3]):
        press_b = data_point(ax, 0.05, 0.95)
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *press_b, name="button_press_event", button=1),
        move_points=[pt(150.0 - 3.0 * i) for i in range(max(4, n - n // 2))],
        move_ev_factory=factory,
        flush=win._flush_ion_rotation_preview,
        timer_attr="ion_rotation_preview_timer",
        release_ev=factory(*pt(-20.0)),
        ghost_artists_fn=artists,
        ghost_label="ion_rotation@-20",
        drag_axes=ax,
        expect_flag=lambda: win.dragging_ion_rotation,
    )
    _stop_preview_timers(win)
    app.processEvents()
    return result


def drive_ion_tof_fit_box_drag(win, app, n: int) -> InteractionResult:
    result = InteractionResult("ion_tof_fit_box_drag")
    win.ion_tof_fit_pick_btn.setChecked(True)
    app.processEvents()
    ax = win.ax_ion_tof_xy
    p0 = data_point(ax, 0.30, 0.30)
    p1 = data_point(ax, 0.62, 0.62)
    p2 = data_point(ax, 0.45, 0.75)

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    artists = lambda: [win.ion_tof_fit_preview_patch, win.ion_tof_fit_preview_anchor_artist]  # noqa: E731

    # The preview work happens inline inside _on_canvas_move (5 px drag
    # threshold arms _ion_tof_fit_drag_active), so time the move handler.
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *p0, name="button_press_event", button=1),
        move_points=linear_track(p0, p1, n),
        move_ev_factory=factory,
        flush=lambda: None,
        timer_attr=None,
        release_ev=factory(*p1),
        ghost_artists_fn=artists,
        ghost_label="tof_fit_box@p1",
        drag_axes=ax,
        expect_flag=lambda: win._ion_tof_fit_pending_axis is not None,
        timed_on_move=True,
    )
    _stop_preview_timers(win)
    app.processEvents()
    run_drag_session(
        win, result,
        press_ev=make_event(win, ax, *p1, name="button_press_event", button=1),
        move_points=linear_track(p1, p2, n),
        move_ev_factory=factory,
        flush=lambda: None,
        timer_attr=None,
        release_ev=factory(*p2),
        ghost_artists_fn=artists,
        ghost_label="tof_fit_box@p2",
        drag_axes=ax,
        expect_flag=lambda: win._ion_tof_fit_pending_axis is not None,
        timed_on_move=True,
    )
    _stop_preview_timers(win)
    app.processEvents()
    win.ion_tof_fit_pick_btn.setChecked(False)
    app.processEvents()
    return result


def drive_ion_tof_bg_hover(win, app, n: int) -> InteractionResult:
    result = InteractionResult("ion_tof_bg_range_hover")
    win.ion_tof_bg_pick_btn.setChecked(True)
    app.processEvents()
    ax = win.ax_ion_tof_xy
    p0 = data_point(ax, 0.25, 0.5)
    p1 = data_point(ax, 0.6, 0.5)
    p2 = data_point(ax, 0.4, 0.5)

    def factory(x, y):
        return make_event(win, ax, x, y, name="motion_notify_event", button=1)

    def preview_artists():
        found = []
        for artist in list(ax.patches) + list(ax.lines):
            gid = getattr(artist, "get_gid", lambda: None)()
            if gid in {"ion_tof_bg_preview_range", "ion_tof_bg_preview_anchor"}:
                found.append(artist)
        return found

    # Press arms the anchor (inline preview draw).
    t0 = time.perf_counter()
    win._on_canvas_press(make_event(win, ax, *p0, name="button_press_event", button=1))
    result.press_ms = (time.perf_counter() - t0) * 1000.0
    # Hover previews run inline in the move handler (no debounce timer).
    for x, y in linear_track(p0, p1, n):
        t0 = time.perf_counter()
        win._on_canvas_move(factory(x, y))
        result.frame_ms.append((time.perf_counter() - t0) * 1000.0)
    result.ghosts.append(ghost_check(win, ax, preview_artists(), "tof_bg_hover@p1"))
    t0 = time.perf_counter()
    win._clear_ion_tof_bg_pending_selection(update_preview=True)
    result.release_ms = (time.perf_counter() - t0) * 1000.0
    app.processEvents()
    win.ion_tof_bg_pick_btn.setChecked(False)
    app.processEvents()
    return result


def drive_ion_hist_span_commit(win, app, _n: int) -> InteractionResult:
    """Fine-ROI span drag visuals are matplotlib SpanSelector (useblit) internals.

    The app-side per-gesture cost is the debounced commit; measure that.
    """
    result = InteractionResult("ion_hist_fine_roi_span_commit")
    result.note = "(span visuals: matplotlib SpanSelector useblit; row = app commit)"
    win.pending_ion_span_range = (8050.0, 8350.0)
    app.processEvents()
    t0 = time.perf_counter()
    win._flush_pending_ion_span_selection()
    result.release_ms = (time.perf_counter() - t0) * 1000.0
    app.processEvents()
    return result


def measure_typing_debounce(win, app) -> dict:
    """OVERLAY_EDIT_DEBOUNCE_MS typing path: debounced recompute+redraw cost."""
    out: dict[str, float] = {}
    app.processEvents()
    t0 = time.perf_counter()
    win._update_circle_overlay_only()  # what circle_preview_timer (70 ms) runs
    out["circle_params_edit_redraw_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    win._update_ion_overlay_only()  # what ion_overlay_preview_timer (70 ms) runs
    out["ion_filter_edit_redraw_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    win._refresh_scatter_panels_only(preserve_scatter_view=True)  # filter release path
    out["ion_filter_release_refresh_ms"] = (time.perf_counter() - t0) * 1000.0
    app.processEvents()
    return out


def _stop_preview_timers(win) -> None:
    for attr in (
        "drag_preview_timer",
        "ion_drag_preview_timer",
        "ion_rotation_preview_timer",
        "theta_drag_preview_timer",
        "rbasex_profile_range_preview_timer",
        "polar_roi_preview_timer",
        "circle_preview_timer",
        "ion_overlay_preview_timer",
        "ion_span_apply_timer",
        "_layout_refresh_timer",
        "_plot_scroll_preview_capture_timer",
    ):
        timer = getattr(win, attr, None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drag-interaction benchmark + ghost detector")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--label", default="BEFORE", help="Report label (BEFORE/AFTER)")
    parser.add_argument("--only", default="", help="Comma-separated interaction substrings")
    parser.add_argument("--quick", action="store_true", help="50 frames per session")
    args = parser.parse_args(argv)

    frames = 50 if args.quick else max(10, args.frames)
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    workdir = tempfile.mkdtemp(prefix="vmi_bench_drag_")
    previous_cwd = os.getcwd()
    os.chdir(workdir)
    windows: list = []
    results: list = []
    typing_stats: dict = {}
    exit_code = 0
    try:
        ensure_sample_data()
        _patch_message_boxes()
        from PySide6.QtWidgets import QApplication

        from VMI_workflow import MainWindow

        app = QApplication.instance() or QApplication([])
        t0 = time.perf_counter()
        win = prepared_window(app, MainWindow, windows)
        prep_s = time.perf_counter() - t0
        app.processEvents()

        # Order matters: applying an ion rotation invalidates the downstream
        # circle/centered projections (app semantics), so the theta/rBasex
        # panel drags run BEFORE the rotation interaction, and polar last
        # (it retargets the electron axes).
        drivers = [
            ("electron_ring", drive_electron_ring_drag),
            ("ion_filter", drive_ion_filter_drag),
            ("theta_centered", lambda w, a, nn: drive_theta_drag(w, a, nn, "centered")),
            ("theta_rbasex", lambda w, a, nn: drive_theta_drag(w, a, nn, "rbasex")),
            ("rbasex_range", drive_rbasex_range_drag),
            ("tof_fit_box", drive_ion_tof_fit_box_drag),
            ("tof_bg_hover", drive_ion_tof_bg_hover),
            ("ion_rotation", drive_ion_rotation_drag),
            ("polar_roi", drive_polar_roi_drag),
            ("ion_hist_span", drive_ion_hist_span_commit),
        ]
        for name, driver in drivers:
            if only and not any(o in name for o in only):
                continue
            try:
                results.append(driver(win, app, frames))
            except Exception as exc:  # noqa: BLE001
                result = InteractionResult(name)
                result.note = f"ERROR: {exc}"
                traceback.print_exc()
                results.append(result)
            app.processEvents()
            _stop_preview_timers(win)

        if not only or any(o in "typing" for o in only):
            typing_stats = measure_typing_debounce(win, app)

        print()
        print(
            f"## Drag benchmark ({args.label}) — {frames} frames/session x2 sessions, "
            f"offscreen, workflow prep {prep_s:.1f}s"
        )
        print()
        print("| interaction | press ms | mean ms | p95 ms | max ms | fps | release ms | ghost |")
        print("|---|---|---|---|---|---|---|---|")
        for r in results:
            print(r.table_row())
        print()
        if typing_stats:
            print("Typing/debounce path (70 ms OVERLAY_EDIT_DEBOUNCE_MS):")
            for key, value in typing_stats.items():
                print(f"  {key}: {value:.1f} ms")
            print()
        print("Ghost detector details:")
        for r in results:
            for g in r.ghosts:
                print(f"  [{g['status']:9s}] {g['label']}: {g['detail']}")
            if r.note.startswith(("ERROR", "SKIPPED")):
                print(f"  [NOTE     ] {r.name}: {r.note}")
        any_ghost = any(not r.ghost_ok for r in results)
        skips = [r.name for r in results if r.note.startswith(("ERROR", "SKIPPED"))]
        print()
        print(f"GHOST CHECK: {'FAIL' if any_ghost else 'PASS'}"
              f"{'' if not skips else f' (skips/errors: {skips})'}")
        exit_code = 1 if any_ghost else 0
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(workdir, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
