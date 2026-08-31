#!/usr/bin/env python3
"""Offscreen end-to-end smoke test for VMI_workflow (regression safety net).

Drives the REAL ``MainWindow`` through the full 7-step workflow on the
deterministic synthetic triplet from ``tests/make_sample_data.py``:

  1. construction + 2x4 axes grid;
  2. async file load (``_LoadWorker`` + polling timer driven by processEvents);
  3. ``process_and_plot`` in "1e+1i coincidence" mode (paired counts, ion
     histogram with peaks);
  4. ion-TOF fine ROI via the real QLineEdit widgets (``_selected_mask``);
  5. ``estimate_center_once`` with the DEFAULT edge_fit mode
     (polar_outermost and the ring-empty fallback are covered separately by
     ``run_regression_checks`` below);
  6. ``apply_circle_selection`` (denoised centered histogram + panel);
  7. ``run_reconstruction_now`` (async rBasex via ``_ReconWorker``);
  8. session save + restore into a fresh ``MainWindow``;
  9. no error dialogs and no non-deprecation warnings;
 10. regression extensions (``run_regression_checks``): startup placeholder
     panels + "[events: n/a]" trigger-combo labels, ``polar_outermost`` and
     ring-empty-fallback center estimation (the former
     ``NameError: source_label`` crash), and the empty-selection scatter
     branch ("No selected points" annotation, cleared colorbar). These
     checks intentionally do NOT contribute to the golden dict. The
     ``check_ion_tof_alignment`` extension (2026-08-31) additionally drives
     the ion-TOF fit + "Apply Align to 0" + ion-scatter temp-centering
     workflow and locks the transformed coordinate outputs (rounded to 9
     decimals) against hardcoded values in this file.

Golden workflow:
- ``python tests/test_smoke.py --update-golden`` writes
  ``tests/golden_smoke.json`` from the current run (paired counts, selected
  mask count, center estimate, denoised histogram sum, rBasex peak list).
- plain run asserts the same numbers match the golden file exactly (values
  are rounded to 6 decimals before comparison).

Run from the project root:
    python tests/test_smoke.py [--update-golden]
    # or: pytest tests/test_smoke.py -q
(QT_QPA_PLATFORM is forced to "offscreen" automatically.)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must happen before any Qt import (VMI_workflow imports PySide6 + QtAgg).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

SAMPLE_DIR = TESTS_DIR / "sample_data"
GOLDEN_PATH = TESTS_DIR / "golden_smoke.json"
GOLDEN_VERSION = 1

TRIGGER_PATH = SAMPLE_DIR / "synth_ar100_vmi_DAn.dat"
ELECTRON_PATH = SAMPLE_DIR / "synth_ar100_vmi.lmf_elec_DAn.dat"
ION_PATH = SAMPLE_DIR / "synth_ar100_vmi.lmf_ion_DAn.dat"

# Fixed workflow inputs (mirrors the generator's physics; do not change
# without regenerating the sample data AND both golden files).
TRUE_CENTER = (128.0, 125.5)
INITIAL_CENTER = (126.0, 123.0)  # deliberately slightly off, estimator must move it
INNER_RADIUS = 118.0  # contains both rings (60, 110)
OUTER_RADIUS = 140.0  # noise annulus 118..140 (background disk radius is 148)
FINE_ROI = (7900.0, 8600.0)  # ns, around the main ion TOF peak at 8250 ns
EXPECTED_SUBPLOT_KEYS = {
    "ion_histogram",
    "ion_tof_xy",
    "electron_scatter",
    "ion_scatter",
    "centered_bin",
    "centered_theta_profile",
    "rbasex_reconstruction",
    "rbasex_radial_profile",
}

# Timeout budgets (seconds) for the async worker waits.
LOAD_TIMEOUT_S = 120.0
CENTER_TIMEOUT_S = 240.0
CIRCLE_TIMEOUT_S = 180.0
RECON_TIMEOUT_S = 600.0


class SmokeFailure(AssertionError):
    """Raised on any failed check; carries the step name for reporting."""


def _fail(step: str, message: str) -> "SmokeFailure":
    return SmokeFailure(f"[{step}] {message}")


def wait_until(app, predicate, timeout_s: float, step: str, what: str) -> None:
    """Pump the Qt event loop until predicate() is true or timeout.

    This is how the async workers are driven offscreen: every worker
    (_LoadWorker/_CenterWorker/_ProjectionWorker/_ReconWorker) publishes its
    result into plain Python fields and a 120 ms QTimer on the main window
    polls those fields.  QApplication.processEvents() dispatches those timer
    events, so a plain polling loop drives the whole machinery
    deterministically without needing a real event loop.
    """
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        try:
            if predicate():
                return
        except SmokeFailure:
            raise
        except Exception as exc:  # predicate raised => real failure
            raise _fail(step, f"wait predicate for {what!r} raised: {exc}\n{traceback.format_exc()}") from exc
        time.sleep(0.01)
    raise _fail(step, f"timeout after {timeout_s:g}s waiting for {what!r}")


def ensure_sample_data() -> None:
    needed = (TRIGGER_PATH, ELECTRON_PATH, ION_PATH)
    if all(p.is_file() for p in needed):
        return
    sys.path.insert(0, str(TESTS_DIR))
    import make_sample_data

    make_sample_data.generate(SAMPLE_DIR)


# ---------------------------------------------------------------------------
# Golden helpers
# ---------------------------------------------------------------------------
def _round6(value: float) -> float:
    return round(float(value), 6)


def _rounded(value) -> float | int | list | dict | str:
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, (float,)):
        if not math.isfinite(value):
            return str(value)
        return _round6(value)
    return value


def _compare(golden, live, path: str = "") -> list[str]:
    mismatches: list[str] = []
    if isinstance(golden, dict) and isinstance(live, dict):
        for key in sorted(set(golden) | set(live)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in golden:
                mismatches.append(f"{sub}: new key not in golden (live={live[key]!r})")
            elif key not in live:
                mismatches.append(f"{sub}: key missing in live result")
            else:
                mismatches.extend(_compare(golden[key], live[key], sub))
    elif isinstance(golden, list) and isinstance(live, list):
        if len(golden) != len(live):
            mismatches.append(f"{path}: length golden={len(golden)} live={len(live)}")
        else:
            for idx, (g, l) in enumerate(zip(golden, live)):
                mismatches.extend(_compare(g, l, f"{path}[{idx}]"))
    elif golden != live:
        mismatches.append(f"{path}: golden={golden!r} live={live!r}")
    return mismatches


# ---------------------------------------------------------------------------
# The actual workflow driver
# ---------------------------------------------------------------------------
def run_smoke(update_golden: bool) -> dict:
    """Run the full offscreen workflow; return the golden dict on success."""
    ensure_sample_data()

    from PySide6.QtWidgets import QApplication, QMessageBox

    from VMI_workflow import MainWindow, SESSION_OUTPUT_DIRNAME

    app = QApplication.instance() or QApplication([])

    # Capture modal dialogs so offscreen runs can never block, and so we can
    # assert the workflow produced no warning/error dialogs.
    dialogs: dict[str, list[dict[str, str]]] = {"warning": [], "critical": []}
    original_warning = QMessageBox.warning
    original_critical = QMessageBox.critical

    def fake_warning(*call_args, **_kwargs):
        dialogs["warning"].append(
            {
                "title": str(call_args[1]) if len(call_args) > 1 else "",
                "text": str(call_args[2]) if len(call_args) > 2 else "",
            }
        )
        return QMessageBox.StandardButton.Ok

    def fake_critical(*call_args, **_kwargs):
        dialogs["critical"].append(
            {
                "title": str(call_args[1]) if len(call_args) > 1 else "",
                "text": str(call_args[2]) if len(call_args) > 2 else "",
            }
        )
        return QMessageBox.StandardButton.Ok

    QMessageBox.warning = fake_warning
    QMessageBox.critical = fake_critical

    # Session output is written to <cwd>/workflow_outputs -> isolate in a
    # temp dir so tests never pollute the repository checkout.
    workdir = tempfile.mkdtemp(prefix="vmi_smoke_run_")
    previous_cwd = os.getcwd()
    os.chdir(workdir)

    windows: list[MainWindow] = []
    collected_warnings: list[str] = []
    golden: dict = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            golden = _run_workflow(app, MainWindow, SESSION_OUTPUT_DIRNAME, windows, workdir)
            # Regression extensions (2026-08-31): startup visuals,
            # polar_outermost / ring-empty center estimation and the
            # empty-selection scatter branch. These intentionally do NOT
            # contribute to the golden dict.
            run_regression_checks(app, MainWindow, windows)
        for w in caught:
            category = w.category.__name__ if isinstance(w.category, type) else str(w.category)
            filename = str(w.filename or "")
            allowed = (
                "matplotlib" in filename
                or issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
            )
            if not allowed:
                collected_warnings.append(
                    f"{category}: {w.message} ({filename}:{w.lineno})"
                )
    finally:
        os.chdir(previous_cwd)
        QMessageBox.warning = original_warning
        QMessageBox.critical = original_critical
        if os.environ.get("VMI_SMOKE_KEEP_TMP") != "1":
            shutil.rmtree(workdir, ignore_errors=True)

    if dialogs["warning"] or dialogs["critical"]:
        raise _fail(
            "dialogs",
            f"app raised {len(dialogs['warning'])} warning and "
            f"{len(dialogs['critical'])} critical dialogs: "
            f"{dialogs['warning'][:3]} {dialogs['critical'][:3]}",
        )
    if collected_warnings:
        raise _fail(
            "warnings",
            f"{len(collected_warnings)} non-deprecation warning(s) during run:\n"
            + "\n".join(collected_warnings[:20]),
        )

    if update_golden:
        payload = {"golden_version": GOLDEN_VERSION, "workflow": _rounded(golden["workflow"])}
        GOLDEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Golden smoke file written: {GOLDEN_PATH}")
    else:
        if not GOLDEN_PATH.is_file():
            print(
                f"NOTE: {GOLDEN_PATH.name} not found; run with --update-golden once to create it. "
                "Structural checks still passed."
            )
        else:
            golden_file = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
            mismatches = _compare(golden_file.get("workflow", {}), _rounded(golden["workflow"]), "workflow")
            if mismatches:
                raise _fail(
                    "golden",
                    f"{len(mismatches)} mismatch(es) against {GOLDEN_PATH.name}:\n"
                    + "\n".join(mismatches[:40]),
                )
    print(f"Timing: {golden.get('timing', {})}")
    return golden


def _run_workflow(app, MainWindow, session_output_dirname: str, windows: list, workdir: str) -> dict:
    import numpy as np
    from scipy.signal import find_peaks

    stats = json.loads((SAMPLE_DIR / "generation_stats.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Step 1: construction + subplot grid
    # ------------------------------------------------------------------
    step = "construct"
    win = MainWindow()
    windows.append(win)
    win.show()
    app.processEvents()
    axes = list(win.figure.axes)
    if len(axes) != 8:
        raise _fail(step, f"figure must have exactly 8 axes in 2x4 grid, found {len(axes)}")
    if set(win.subplot_axes.keys()) != EXPECTED_SUBPLOT_KEYS:
        raise _fail(step, f"subplot_axes keys mismatch: {sorted(win.subplot_axes.keys())}")
    # Startup placeholders are NOT asserted here (known regression being fixed
    # separately); a plain full draw must still work offscreen.
    win.canvas.draw()
    app.processEvents()

    # ------------------------------------------------------------------
    # Step 2: async file load (_LoadWorker)
    # ------------------------------------------------------------------
    step = "load"
    win.file_paths["trigger"] = str(TRIGGER_PATH)
    win.file_paths["electron"] = str(ELECTRON_PATH)
    win.file_paths["ion"] = str(ION_PATH)
    win.load_cache()
    wait_until(
        app,
        lambda: win.cache is not None and not getattr(win, "_load_busy", False),
        LOAD_TIMEOUT_S,
        step,
        "cache installed by _LoadWorker",
    )
    if win.cache is None:
        raise _fail(step, "cache is None after load")
    if win.cache.trigger_indices.shape[0] != int(stats["trigger_rows"]):
        raise _fail(
            step,
            f"trigger rows {win.cache.trigger_indices.shape[0]} != {stats['trigger_rows']}",
        )
    if win.cache.electron_points.shape[0] != int(stats["electron_rows"]):
        raise _fail(step, "electron rows mismatch vs generation stats")
    if win.cache.ion_points.shape[0] != int(stats["ion_rows"]):
        raise _fail(step, "ion rows mismatch vs generation stats")

    # ------------------------------------------------------------------
    # Step 3: process_and_plot in 1e+1i coincidence mode
    # ------------------------------------------------------------------
    step = "process_and_plot"
    mode_idx = win.trigger_mode_combo.findData("coincidence")
    if mode_idx < 0:
        raise _fail(step, "trigger_mode_combo has no 'coincidence' entry")
    win.trigger_mode_combo.setCurrentIndex(int(mode_idx))
    app.processEvents()
    win.process_and_plot()
    app.processEvents()
    paired_count = int(win._paired_count())
    if paired_count < 15000:
        raise _fail(step, f"expected >15000 coincidence pairs, got {paired_count}")
    counts = np.asarray(win._ion_hist_cache_counts, dtype=np.float64).reshape(-1)
    if counts.size == 0 or float(np.max(counts)) <= 0:
        raise _fail(step, "ion histogram was not populated")
    peak_idx, _props = find_peaks(counts, height=0.05 * float(np.max(counts)), prominence=0.05 * float(np.max(counts)))
    if peak_idx.size < 2:
        raise _fail(
            step,
            f"ion histogram should show >=2 TOF peaks (main+secondary), found {peak_idx.size}",
        )
    ion_hist_artists = [
        child
        for child in win.ax_hist_ion.get_children()
        if getattr(child, "get_gid", lambda: None)() == "ion_histogram_display"
    ]
    if not ion_hist_artists:
        raise _fail(step, "ion histogram artist (gid=ion_histogram_display) missing")

    # ------------------------------------------------------------------
    # Step 4: ion-TOF fine ROI through the real widgets
    # ------------------------------------------------------------------
    step = "fine_roi"
    win.ion_fine_xmin_edit.setText(str(FINE_ROI[0]))
    win.ion_fine_xmax_edit.setText(str(FINE_ROI[1]))
    win.apply_ion_fine_roi_from_inputs()
    app.processEvents()
    if win.ion_range is None:
        raise _fail(step, f"fine ROI {FINE_ROI} was not applied (ion_range is None)")
    mask = win._selected_mask()
    selected_mask_count = int(np.count_nonzero(mask))
    if selected_mask_count < 10000:
        raise _fail(step, f"fine ROI selected only {selected_mask_count} events (<10000)")

    # ------------------------------------------------------------------
    # Step 5: estimate_center_once with DEFAULT edge_fit mode
    # (polar_outermost + ring-empty fallback are covered in
    # run_regression_checks below)
    # ------------------------------------------------------------------
    step = "estimate_center"
    if str(win._current_center_mode()) != "edge_fit":
        raise _fail(step, f"default center mode should be edge_fit, got {win._current_center_mode()}")
    win.circle_cx_edit.setText(f"{INITIAL_CENTER[0]:g}")
    win.circle_cy_edit.setText(f"{INITIAL_CENTER[1]:g}")
    win.inner_r_edit.setText(f"{INNER_RADIUS:g}")
    win.outer_r_edit.setText(f"{OUTER_RADIUS:g}")
    win.outer_ring_filter_enable_checkbox.setChecked(True)
    app.processEvents()
    win.estimate_center_once()
    wait_until(
        app,
        lambda: (not getattr(win, "_center_busy", False)) and getattr(win, "_center_worker_thread", None) is None,
        CENTER_TIMEOUT_S,
        step,
        "center estimation worker",
    )
    if win.circle_centroid is None:
        raise _fail(step, "circle_centroid was not set (estimator bailed or kept None)")
    cx_est, cy_est = (float(win.circle_centroid[0]), float(win.circle_centroid[1]))
    center_error = math.hypot(cx_est - TRUE_CENTER[0], cy_est - TRUE_CENTER[1])
    if center_error > 10.0:
        raise _fail(
            step,
            f"center estimate ({cx_est:.4f}, {cy_est:.4f}) is {center_error:.2f}px from "
            f"true center {TRUE_CENTER} (>10px)",
        )

    # ------------------------------------------------------------------
    # Step 6: apply_circle_selection (ring projection + denoised binning)
    # ------------------------------------------------------------------
    step = "apply_circle"
    win.apply_circle_selection()
    wait_until(
        app,
        lambda: (win.centered_hist_data is not None) and (not getattr(win, "_circle_busy", False)),
        CIRCLE_TIMEOUT_S,
        step,
        "ring projection worker",
    )
    hist_data = win.centered_hist_data
    if hist_data is None:
        raise _fail(step, "centered_hist_data is None after apply_circle_selection")
    denoised_sum = float(np.sum(hist_data["hist_denoised"]))
    if denoised_sum <= 0:
        raise _fail(step, f"denoised histogram sum is {denoised_sum} (expected > 0)")
    ring_inner_count = int(np.asarray(win.ring_inner_selected_electron).shape[0])
    ring_outer_count = int(np.asarray(win.ring_outer_noise_electron).shape[0])
    if ring_inner_count < 10000:
        raise _fail(step, f"ring inner count {ring_inner_count} unexpectedly small")
    if ring_outer_count <= 0:
        raise _fail(step, "outer-ring noise filter collected 0 noise points")
    if len(win.ax_centered_bin.images) < 1:
        raise _fail(step, "centered-bin panel has no image drawn")

    # ------------------------------------------------------------------
    # Step 7: run_reconstruction_now (async rBasex via _ReconWorker)
    # ------------------------------------------------------------------
    step = "reconstruction"
    recon_started = time.perf_counter()
    win.run_reconstruction_now()
    wait_until(
        app,
        lambda: (win.rbasex_recon_result is not None) and (not getattr(win, "_recon_busy", False)),
        RECON_TIMEOUT_S,
        step,
        "rBasex reconstruction worker",
    )
    recon_seconds = time.perf_counter() - recon_started
    rb = win.rbasex_recon_result
    if rb is None:
        raise _fail(step, "rbasex_recon_result is None")
    if str(rb.get("error", "missing")) != "":
        raise _fail(step, f"rBasex reported error: {rb.get('error')}")
    peaks = [
        {
            "r": float(p["r"]),
            "beta": float(p["beta"]),
            "i": float(p["i"]),
            "area": float(p.get("area", 0.0)),
        }
        for p in rb.get("peaks", [])
    ]
    if len(peaks) < 1:
        raise _fail(step, "rBasex returned no peaks")
    if len(win.ax_reserved_top.images) < 1:
        raise _fail(step, "rBasex reconstruction panel has no image drawn")

    # ------------------------------------------------------------------
    # Step 8: session save + restore into a fresh MainWindow
    # ------------------------------------------------------------------
    step = "session_roundtrip"
    win.save_session_output()
    app.processEvents()
    base_dir = Path(workdir) / session_output_dirname
    session_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir()) if base_dir.is_dir() else []
    if not session_dirs:
        raise _fail(step, "save_session_output created no session directory")
    session_dir = session_dirs[-1]
    meta_path = session_dir / "session_metadata.json"
    npz_path = session_dir / "session_data.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        raise _fail(step, f"session artifacts missing in {session_dir}")

    win2 = MainWindow()
    windows.append(win2)
    if len(win2.figure.axes) != 8:
        raise _fail(step, "restored window also must construct 8 axes")
    if not win2._load_session_output_from_metadata_path(str(meta_path)):
        raise _fail(step, "_load_session_output_from_metadata_path returned False")
    app.processEvents()

    if win2.circle_centroid is None or win2.ion_range is None:
        raise _fail(step, "restored center/ROI state is None")
    # The app round-trips the center through the UI line edits at "%.6g"
    # precision, so require 6-significant-digit fidelity rather than exact
    # float equality.
    if abs(float(win2.circle_centroid[0]) - cx_est) > 1e-3 or abs(float(win2.circle_centroid[1]) - cy_est) > 1e-3:
        raise _fail(step, f"restored center {win2.circle_centroid} != {cx_est, cy_est}")
    if tuple(float(v) for v in win2.ion_range) != tuple(float(v) for v in win.ion_range):
        raise _fail(step, f"restored ion_range {win2.ion_range} != {win.ion_range}")
    restored_paired = int(win2._paired_count())
    if restored_paired != selected_mask_count:
        raise _fail(
            step,
            f"restored pair count {restored_paired} != selected mask count {selected_mask_count}",
        )
    if win2.centered_hist_data is None:
        raise _fail(step, "restored centered_hist_data is None")
    restored_den_sum = float(np.sum(win2.centered_hist_data["hist_denoised"]))
    if abs(restored_den_sum - denoised_sum) > 1e-6:
        raise _fail(step, f"restored denoised sum {restored_den_sum} != {denoised_sum}")
    restored_peaks = [
        {"r": float(p["r"]), "beta": float(p["beta"]), "i": float(p["i"]), "area": float(p.get("area", 0.0))}
        for p in (win2.rbasex_recon_result or {}).get("peaks", [])
    ]
    if _rounded(restored_peaks) != _rounded(peaks):
        raise _fail(step, "restored rBasex peaks differ from live peaks")

    workflow = {
        "paired_count": paired_count,
        "fine_roi": [float(FINE_ROI[0]), float(FINE_ROI[1])],
        "selected_mask_count": selected_mask_count,
        "center_estimate": [cx_est, cy_est],
        "center_error_px": center_error,
        "ring_inner_count": ring_inner_count,
        "ring_outer_count": ring_outer_count,
        "denoised_hist_sum": denoised_sum,
        "denoised_removed_total": float(hist_data.get("removed_total", 0.0)),
        "rbasex_peak_count": len(peaks),
        "rbasex_peaks": peaks,
        "restored_paired_count": restored_paired,
    }
    # wall-clock timing is reported but intentionally NOT part of the golden
    timing = {"recon_seconds": recon_seconds}
    print(
        "Smoke workflow OK: paired={paired_count}, selected={selected_mask_count}, "
        "center=({cx:.4f}, {cy:.4f}) err={err:.3f}px, denoised_sum={den:.3f}, "
        "rbasex_peaks={npk}, recon={secs:.1f}s".format(
            paired_count=paired_count,
            selected_mask_count=selected_mask_count,
            cx=cx_est,
            cy=cy_est,
            err=center_error,
            den=denoised_sum,
            npk=len(peaks),
            secs=recon_seconds,
        )
    )
    return {"workflow": workflow, "timing": timing}


# ---------------------------------------------------------------------------
# Regression extensions (2026-08-31)
# ---------------------------------------------------------------------------
def _make_prepared_window(app, MainWindow, windows: list, *, fine_roi: bool) -> "MainWindow":
    """Fresh MainWindow with sample data loaded and processed (coincidence mode).

    Shared preparation for the regression checks: async load via _LoadWorker,
    process_and_plot, and (optionally) the fine ion-TOF ROI plus the same
    circle parameters the main workflow uses, so the inner ring has points.
    """
    step = "regression_prepare"
    win = MainWindow()
    windows.append(win)
    win.show()
    win.file_paths["trigger"] = str(TRIGGER_PATH)
    win.file_paths["electron"] = str(ELECTRON_PATH)
    win.file_paths["ion"] = str(ION_PATH)
    win.load_cache()
    wait_until(
        app,
        lambda: win.cache is not None and not getattr(win, "_load_busy", False),
        LOAD_TIMEOUT_S,
        step,
        "cache installed by _LoadWorker",
    )
    win.trigger_mode_combo.setCurrentIndex(win.trigger_mode_combo.findData("coincidence"))
    app.processEvents()
    win.process_and_plot()
    app.processEvents()
    if fine_roi:
        win.ion_fine_xmin_edit.setText(str(FINE_ROI[0]))
        win.ion_fine_xmax_edit.setText(str(FINE_ROI[1]))
        win.apply_ion_fine_roi_from_inputs()
        app.processEvents()
        if win.ion_range is None:
            raise _fail(step, f"fine ROI {FINE_ROI} was not applied (ion_range is None)")
        win.circle_cx_edit.setText(f"{INITIAL_CENTER[0]:g}")
        win.circle_cy_edit.setText(f"{INITIAL_CENTER[1]:g}")
        win.inner_r_edit.setText(f"{INNER_RADIUS:g}")
        win.outer_r_edit.setText(f"{OUTER_RADIUS:g}")
        win.outer_ring_filter_enable_checkbox.setChecked(True)
        app.processEvents()
    return win


def check_startup_placeholders(app, MainWindow, windows: list) -> None:
    """Startup visuals: titled placeholder panels + '[events: n/a]' combo labels.

    Covers the re-added ``__init__`` calls ``_draw_placeholder`` and
    ``_update_trigger_mode_combo_labels``.
    """
    step = "startup_placeholders"
    win = MainWindow()
    windows.append(win)
    win.show()
    app.processEvents()

    # Every subplot panel must carry a non-empty placeholder title, except the
    # radial-profile panel, which (matching the old startup) uses a placeholder
    # text instead of a title.
    for key, ax in win.subplot_axes.items():
        title = str(ax.get_title())
        if key == "rbasex_radial_profile":
            continue
        if not title.strip():
            raise _fail(step, f"subplot {key!r} has no placeholder title after construction")

    # In-axes placeholder hint texts must be present (placeholder artists).
    reserved_top_texts = {str(t.get_text()) for t in win.ax_reserved_top.texts}
    if "Click Start Reconstruction" not in reserved_top_texts:
        raise _fail(step, f"rBasex panel placeholder text missing: {sorted(reserved_top_texts)}")
    theta_texts = {str(t.get_text()) for t in win.ax_centered_theta_profile.texts}
    if "Click 'Apply Ring Selection and Bin' first" not in theta_texts:
        raise _fail(step, f"radial-profile placeholder text missing: {sorted(theta_texts)}")
    profile_texts = {str(t.get_text()) for t in win.ax_rbasex_profile.texts}
    if "Click Start Reconstruction" not in profile_texts:
        raise _fail(step, f"radial-profile-panel placeholder text missing: {sorted(profile_texts)}")

    # Trigger-mode dropdown items must show the startup '[events: n/a]' suffix.
    if win.trigger_mode_combo.count() < 4:
        raise _fail(step, f"trigger_mode_combo has only {win.trigger_mode_combo.count()} items")
    for idx in range(win.trigger_mode_combo.count()):
        text = str(win.trigger_mode_combo.itemText(idx))
        if not text.endswith("[events: n/a]"):
            raise _fail(
                step,
                f"trigger combo item {idx} does not end with '[events: n/a]': {text!r}",
            )
    print("Startup placeholders OK: titled panels, hint texts, [events: n/a] combo labels")


def check_polar_outermost_center(app, MainWindow, windows: list) -> None:
    """polar_outermost center estimation (the former NameError: source_label crash)."""
    step = "center_polar_outermost"
    win = _make_prepared_window(app, MainWindow, windows, fine_roi=True)

    # Pre-seed the centroid so the finite-result assertion below is about THIS
    # estimation call (matches a UI session where a previous estimate exists).
    win.circle_centroid = (float(INITIAL_CENTER[0]), float(INITIAL_CENTER[1]))

    # Polar ROI band covering both Newton-sphere rings (60 px and 110 px) so
    # the polar branch finds >= 24 ROI points (no early status bailout).
    win.polar_roi_rmin_edit.setText("40")
    win.polar_roi_rmax_edit.setText("130")
    app.processEvents()

    mode_idx = win.center_mode_combo.findData("polar_outermost")
    if mode_idx < 0:
        raise _fail(step, "center_mode_combo has no 'polar_outermost' entry")
    win.center_mode_combo.setCurrentIndex(int(mode_idx))
    app.processEvents()

    # This call raised NameError: source_label before the fix.
    win.estimate_center_once()
    wait_until(
        app,
        lambda: (not getattr(win, "_center_busy", False))
        and getattr(win, "_center_worker_thread", None) is None,
        CENTER_TIMEOUT_S,
        step,
        "polar_outermost center estimation worker",
    )

    ctx = getattr(win, "_center_result_context", None) or {}
    if str(ctx.get("mode_now", "")) != "polar_outermost":
        raise _fail(step, f"polar_outermost branch did not run (context mode={ctx.get('mode_now')!r})")
    if "ROI" not in str(ctx.get("estimation_source", "")):
        raise _fail(
            step,
            f"polar ROI estimation source missing (context source={ctx.get('estimation_source')!r})",
        )
    if win.circle_centroid is None:
        raise _fail(step, "circle_centroid is None after polar_outermost estimation")
    if not all(math.isfinite(float(v)) for v in win.circle_centroid):
        raise _fail(step, f"polar_outermost center result is not finite: {win.circle_centroid}")
    if win.progress_bar.isVisible():
        raise _fail(step, "progress bar still visible after polar_outermost estimation")
    print(
        "polar_outermost center OK: center=({:.4f}, {:.4f}), progress bar hidden".format(
            float(win.circle_centroid[0]), float(win.circle_centroid[1])
        )
    )


def check_ring_empty_center_fallback(app, MainWindow, windows: list) -> None:
    """Ring-empty fallback estimation (also hit the former source_label NameError)."""
    step = "center_ring_empty"
    win = _make_prepared_window(app, MainWindow, windows, fine_roi=True)

    # Move the inner ring completely off the data so ring_inner_selected is
    # empty; edge_fit with >= 24 candidates must take the "(fallback: full
    # set)" branch (pre-fix this raised NameError: source_label).
    win.circle_centroid = (400.0, 400.0)
    win.circle_cx_edit.setText("400")
    win.circle_cy_edit.setText("400")
    win.inner_r_edit.setText("5")
    win.outer_r_edit.setText("10")
    app.processEvents()

    win.estimate_center_once()
    wait_until(
        app,
        lambda: (not getattr(win, "_center_busy", False))
        and getattr(win, "_center_worker_thread", None) is None,
        CENTER_TIMEOUT_S,
        step,
        "ring-empty center estimation worker",
    )

    ctx = getattr(win, "_center_result_context", None) or {}
    if "fallback: full set" not in str(ctx.get("estimation_source", "")):
        raise _fail(
            step,
            "ring-empty fallback did not trigger (context source="
            f"{ctx.get('estimation_source')!r})",
        )
    if win.circle_centroid is None:
        raise _fail(step, "circle_centroid is None after ring-empty fallback estimation")
    if not all(math.isfinite(float(v)) for v in win.circle_centroid):
        raise _fail(step, f"ring-empty fallback center result is not finite: {win.circle_centroid}")
    if win.progress_bar.isVisible():
        raise _fail(step, "progress bar still visible after ring-empty fallback estimation")
    print("ring-empty fallback OK: full-set fallback ran, progress bar hidden")


def check_empty_selection_scatter(app, MainWindow, windows: list) -> None:
    """Empty selection with context points: 'No selected points' + cleared colorbar."""
    step = "empty_selection"
    win = _make_prepared_window(app, MainWindow, windows, fine_roi=False)

    # Move the ion scatter filter rectangle fully off the detector (raw ion
    # coordinates live inside the ~256 px detector), then enable the filter:
    # selects 0 events while all other points remain as gray context.
    win.ion_filter_cx_edit.setText("500")
    win.ion_filter_cy_edit.setText("500")
    win.ion_filter_w_edit.setText("2")
    win.ion_filter_h_edit.setText("2")
    app.processEvents()
    win.ion_filter_enable_checkbox.setChecked(True)
    app.processEvents()

    texts = {str(t.get_text()): t for t in win.ax_scatter_e.texts}
    if "No selected points" not in texts:
        raise _fail(step, f"'No selected points' annotation missing (texts={sorted(texts)})")
    annotation = texts["No selected points"]
    pos = tuple(float(v) for v in annotation.get_position())
    if pos != (0.5, 0.98):
        raise _fail(step, f"'No selected points' annotation at unexpected position {pos} (want (0.5, 0.98))")
    if win.electron_scatter_colorbar is not None:
        raise _fail(step, "electron scatter colorbar was not cleared on empty selection")
    context_gids = {
        str(getattr(c, "get_gid", lambda: None)()) for c in win.ax_scatter_e.collections
    }
    if "electron_scatter_context" not in context_gids:
        raise _fail(step, f"gray context scatter missing (collection gids={sorted(context_gids)})")
    print("Empty selection OK: annotation drawn, colorbar cleared, context points kept")


# ---------------------------------------------------------------------------
# Ion-TOF alignment regression (2026-08-31 dedup safety net)
# ---------------------------------------------------------------------------
# Fixed x-TOF fit ROI box (t0, t1, c0, c1) in ns x px for
# ``fit_ion_tof_alignment_line``; sample data ions cluster around x ~ 128 px
# with the main TOF peak at 8250 ns, so this box contains the bulk.
ION_TOF_FIT_ROI = (8000.0, 8600.0, 60.0, 200.0)
# Number of paired points (evenly spaced fixed indices) captured from the
# transformed-output functions below.
ION_TOF_SAMPLE_N = 12

# Locked transformed outputs for check_ion_tof_alignment (rounded to 9
# decimals). Captured 2026-08-31 from the un-refactored code; the ion-TOF
# alignment apply path was previously untested.
ION_TOF_ALIGNMENT_EXPECTED: dict = {
    'fit_display_slope': 0.076285421,
    'fit_display_intercept': -500.241230734,
    'fit_raw_slope': 0.076285421,
    'fit_raw_intercept': -500.241230734,
    'align_slope': 0.076285421,
    'align_intercept': -500.241230734,
    'transform_ion_xy': [
        [23.524112753, 148.445615],
        [-257.65868874, 115.328861],
        [-285.30431581, 150.989046],
        [17.364221764, 121.442954],
        [37.587764237, 125.305411],
        [-6.101670191, 155.015229],
        [-18.570547617, 130.122608],
        [-5.659854046, 137.202193],
        [-28.69533037, 149.267632],
        [-5.805761847, 117.198074],
        [12.578777396, 131.867263],
        [-12.191763841, 141.415993],
    ],
    'transform_ion_xy_for_scatter_pre_center': [
        [23.524112753, 148.445615],
        [-257.65868874, 115.328861],
        [-285.30431581, 150.989046],
        [17.364221764, 121.442954],
        [37.587764237, 125.305411],
        [-6.101670191, 155.015229],
        [-18.570547617, 130.122608],
        [-5.659854046, 137.202193],
        [-28.69533037, 149.267632],
        [-5.805761847, 117.198074],
        [12.578777396, 131.867263],
        [-12.191763841, 141.415993],
    ],
    'transform_ion_xy_for_scatter': [
        [23.524112753, 148.445615],
        [-257.65868874, 115.328861],
        [-285.30431581, 150.989046],
        [17.364221764, 121.442954],
        [37.587764237, 125.305411],
        [-6.101670191, 155.015229],
        [-18.570547617, 130.122608],
        [-5.659854046, 137.202193],
        [-28.69533037, 149.267632],
        [-5.805761847, 117.198074],
        [12.578777396, 131.867263],
        [-12.191763841, 141.415993],
    ],
    'fit_y_display_slope': 0.004348376,
    'fit_y_display_intercept': 91.460133211,
    'transform_ion_xy_for_scatter_y_center': [
        [23.524112753, 21.226458778],
        [-257.65868874, -27.538678237],
        [-285.30431581, 8.073850227],
        [17.364221764, -5.672887644],
        [37.587764237, -1.868388042],
        [-6.101670191, 27.606186588],
        [-18.570547617, 2.702790305],
        [-5.659854046, 9.893290653],
        [-28.69533037, 21.724002266],
        [-5.805761847, -10.338064353],
        [12.578777396, 4.945020428],
        [-12.191763841, 13.680022566],
    ],
    'alignment_terms_x': [23.524112753, -257.65868874, -285.30431581, 17.364221764, 37.587764237, -6.101670191, -18.570547617, -5.659854046, -28.69533037, -5.805761847, 12.578777396, -12.191763841],
    'display_coord_values': [23.524112753, -257.65868874, -285.30431581, 17.364221764, 37.587764237, -6.101670191, -18.570547617, -5.659854046, -28.69533037, -5.805761847, 12.578777396, -12.191763841],
}


def check_ion_tof_alignment(app, MainWindow, windows: list) -> None:
    """Ion-TOF fit + 'Apply Align to 0' + ion-scatter temp-centering workflow.

    Drives the REAL public controls the UI uses (``fit_ion_tof_alignment_line``
    with a fixed ROI box, the Apply-Align button handler
    ``apply_ion_tof_alignment_to_zero`` and the ion-scatter temp-centering
    button handler ``apply_ion_scatter_tof_center_correction``), then locks the
    transformed coordinate outputs (``_transform_ion_xy``,
    ``_transform_ion_xy_for_scatter``, ``_apply_ion_tof_alignment_terms`` and
    ``_ion_tof_display_coord_values``) on a fixed index subset at 9 decimals.
    This pins the shared ion-TOF transform math against regressions.
    """
    import numpy as np

    step = "ion_tof_alignment"
    win = _make_prepared_window(app, MainWindow, windows, fine_roi=False)

    # Pin the coincidence-map axis and the scatter-centering axis to "x"
    # through the real combos (both default to "y").
    idx = win.ion_tof_coord_axis_combo.findData("x")
    if idx < 0:
        raise _fail(step, "ion_tof_coord_axis_combo has no 'x' entry")
    win.ion_tof_coord_axis_combo.setCurrentIndex(int(idx))
    idx = win.ion_scatter_tof_center_axis_combo.findData("x")
    if idx < 0:
        raise _fail(step, "ion_scatter_tof_center_axis_combo has no 'x' entry")
    win.ion_scatter_tof_center_axis_combo.setCurrentIndex(int(idx))
    app.processEvents()

    # 1) Fit through the real public driver (fixed ROI box, no dialogs).
    if not win.fit_ion_tof_alignment_line(roi_override=ION_TOF_FIT_ROI):
        raise _fail(step, f"fit_ion_tof_alignment_line failed for ROI {ION_TOF_FIT_ROI}")
    fit = win.ion_tof_fit_result_by_axis.get("x")
    if not isinstance(fit, dict):
        raise _fail(step, "no stored x-TOF fit after fit_ion_tof_alignment_line")

    # 2) Apply-align through the real button handler.
    win.apply_ion_tof_alignment_to_zero()
    app.processEvents()
    align = win.ion_tof_alignment_by_axis.get("x", {})
    if not align.get("enabled", False):
        raise _fail(step, "x-TOF alignment not enabled after apply_ion_tof_alignment_to_zero")

    # 3) Capture transformed outputs on a fixed index subset (round 9dp).
    mask = win._coarse_ion_roi_mask()
    _electron_pts, ion_raw = win._paired_points(mask)
    n = int(ion_raw.shape[0])
    if n < 1000:
        raise _fail(step, f"too few paired ion points for TOF-alignment capture: {n}")
    subset = np.linspace(0, n - 1, ION_TOF_SAMPLE_N, dtype=int)
    xy = np.asarray(ion_raw[subset, :2], dtype=np.float64)
    tof = np.asarray(ion_raw[subset, 2], dtype=np.float64).reshape(-1)

    out_align = win._transform_ion_xy(xy, tof)
    out_scatter_pre = win._transform_ion_xy_for_scatter(xy, tof)  # centering still off
    out_terms = win._apply_ion_tof_alignment_terms(xy[:, 0], tof, "x")
    disp_vals = win._ion_tof_display_coord_values(out_align, tof, "x")

    # 4) Temporary ion-scatter centering through the real button handler.
    win.apply_ion_scatter_tof_center_correction()
    app.processEvents()
    if not win.ion_scatter_tof_center_by_axis.get("x", {}).get("enabled", False):
        raise _fail(
            step,
            "x scatter temp centering not enabled after apply_ion_scatter_tof_center_correction",
        )
    out_scatter = win._transform_ion_xy_for_scatter(xy, tof)

    # 5) Give the scatter-centering branch a NONZERO correction: fit the y
    # axis as well (fit only -- no alignment applied to y), then center the
    # ion scatter on y, so the displayed y-fit terms are nonzero and the
    # y column really shifts.
    idx = win.ion_tof_coord_axis_combo.findData("y")
    if idx < 0:
        raise _fail(step, "ion_tof_coord_axis_combo has no 'y' entry")
    win.ion_tof_coord_axis_combo.setCurrentIndex(int(idx))
    app.processEvents()
    if not win.fit_ion_tof_alignment_line(roi_override=ION_TOF_FIT_ROI):
        raise _fail(step, f"y-axis fit_ion_tof_alignment_line failed for ROI {ION_TOF_FIT_ROI}")
    fit_y = win.ion_tof_fit_result_by_axis.get("y")
    if not isinstance(fit_y, dict):
        raise _fail(step, "no stored y-TOF fit after fit_ion_tof_alignment_line")
    idx = win.ion_scatter_tof_center_axis_combo.findData("y")
    if idx < 0:
        raise _fail(step, "ion_scatter_tof_center_axis_combo has no 'y' entry")
    win.ion_scatter_tof_center_axis_combo.setCurrentIndex(int(idx))
    app.processEvents()
    win.apply_ion_scatter_tof_center_correction()
    app.processEvents()
    out_scatter_y = win._transform_ion_xy_for_scatter(xy, tof)

    captured = {
        "fit_display_slope": round(float(fit["display_slope"]), 9),
        "fit_display_intercept": round(float(fit["display_intercept"]), 9),
        "fit_raw_slope": round(float(fit["raw_slope"]), 9),
        "fit_raw_intercept": round(float(fit["raw_intercept"]), 9),
        "align_slope": round(float(align.get("slope", 0.0)), 9),
        "align_intercept": round(float(align.get("intercept", 0.0)), 9),
        "transform_ion_xy": [[round(float(v), 9) for v in row] for row in out_align],
        "transform_ion_xy_for_scatter_pre_center": [
            [round(float(v), 9) for v in row] for row in out_scatter_pre
        ],
        "transform_ion_xy_for_scatter": [[round(float(v), 9) for v in row] for row in out_scatter],
        "fit_y_display_slope": round(float(fit_y["display_slope"]), 9),
        "fit_y_display_intercept": round(float(fit_y["display_intercept"]), 9),
        "transform_ion_xy_for_scatter_y_center": [
            [round(float(v), 9) for v in row] for row in out_scatter_y
        ],
        "alignment_terms_x": [round(float(v), 9) for v in out_terms],
        "display_coord_values": [round(float(v), 9) for v in disp_vals],
    }

    expected = ION_TOF_ALIGNMENT_EXPECTED
    if expected is None:
        print("Ion TOF alignment values to lock (ION_TOF_ALIGNMENT_EXPECTED is None):")
        print(json.dumps(captured, indent=2))
        return
    mismatches = _compare(expected, captured, "ion_tof_alignment")
    if mismatches:
        raise _fail(
            step,
            f"{len(mismatches)} mismatch(es) against locked ion-TOF alignment values:\n"
            + "\n".join(mismatches[:20]),
        )
    print(
        "Ion TOF alignment OK: fit=({:.6g}*TOF + {:.6g}), apply+scatter-center transforms locked".format(
            float(fit["raw_slope"]), float(fit["raw_intercept"])
        )
    )


def check_ion_tof_xy_cache_invalidation(app, MainWindow, windows: list) -> None:
    """ion TOF XY map cache must carry a data fingerprint (ARCHITECTURE P1-4).

    The cache used to be keyed only by display parameters, so two different
    coarse-ROI selections with the *same point count* replayed the stale
    plot. This check applies two shifted same-count ROI windows and asserts
    the cache key and the binned data actually change (recompute, not reuse).
    """
    import numpy as np

    step = "ion_tof_xy_cache"
    win = _make_prepared_window(app, MainWindow, windows, fine_roi=False)

    # Two coarse-ROI windows with EXACTLY the same selected row count, built
    # from consecutive sorted finite ion-TOF values: [s0..s(m-1)] and
    # [s1..s(m)] both contain exactly m rows but have different bounds.
    ion_t = np.asarray(win._paired_ion_t(), dtype=np.float64).reshape(-1)
    s = np.sort(ion_t[np.isfinite(ion_t)])
    m = 5000
    if s.size < m + 2:
        raise _fail(step, f"not enough finite ion TOF rows for the cache check ({s.size})")
    window_a = (float(s[0]), float(s[m - 1]))
    window_b = (float(s[1]), float(s[m]))
    for name, window in (("A", window_a), ("B", window_b)):
        mask = (ion_t >= window[0]) & (ion_t <= window[1])
        if int(np.count_nonzero(mask)) != m:
            raise _fail(step, f"window {name} {window} selects {int(np.count_nonzero(mask))} rows, want {m}")

    # Populate the cache with selection A (full refresh like the UI does).
    win._apply_ion_hist_x_roi(window_a[0], window_a[1], source="smoke")
    app.processEvents()
    win._refresh_ion_tof_pos_panel_only(draw_canvas=False)
    app.processEvents()
    key_a = win._ion_tof_xy_cache_key
    edges_a = np.asarray(win._ion_tof_xy_cache_x_edges, dtype=np.float64).copy()
    if key_a is None or edges_a.size < 2:
        raise _fail(step, "ion TOF XY cache was not populated for selection A")

    # Switch to same-count selection B: every display parameter is unchanged,
    # only the underlying data selection differs.
    win._apply_ion_hist_x_roi(window_b[0], window_b[1], source="smoke")
    app.processEvents()
    win._refresh_ion_tof_pos_panel_only(draw_canvas=False)
    app.processEvents()
    key_b = win._ion_tof_xy_cache_key
    edges_b = np.asarray(win._ion_tof_xy_cache_x_edges, dtype=np.float64)
    counts_b = np.asarray(win._ion_tof_xy_cache_counts, dtype=np.float32)
    if key_b == key_a:
        raise _fail(step, "ion TOF XY cache key unchanged after same-count coarse ROI change (stale plot risk)")
    if not (edges_b[0] > edges_a[0] and abs(float(edges_b[0]) - float(s[1])) < 1e-6):
        raise _fail(
            step,
            f"cache did not recompute for selection B (edges[0] {edges_b[0]!r}, want ~{float(s[1])!r}; "
            f"previous {edges_a[0]!r})",
        )
    if int(counts_b.sum()) != m:
        raise _fail(step, f"recomputed ion TOF XY map bins {int(counts_b.sum())} points, want {m}")
    print(
        "Ion TOF XY cache OK: same-count ROI change recomputed "
        f"(key changed, edges[0] {edges_a[0]:.4f} -> {edges_b[0]:.4f}, counts={m})"
    )


def run_regression_checks(app, MainWindow, windows: list) -> None:
    """Run the post-fix regression extensions (not part of the golden dict)."""
    check_startup_placeholders(app, MainWindow, windows)
    check_polar_outermost_center(app, MainWindow, windows)
    check_ring_empty_center_fallback(app, MainWindow, windows)
    check_empty_selection_scatter(app, MainWindow, windows)
    check_ion_tof_alignment(app, MainWindow, windows)
    check_ion_tof_xy_cache_invalidation(app, MainWindow, windows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offscreen E2E smoke test for VMI_workflow")
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Write tests/golden_smoke.json from this run instead of comparing",
    )
    args = parser.parse_args(argv)
    try:
        run_smoke(update_golden=args.update_golden)
    except SmokeFailure as exc:
        print(f"SMOKE FAIL: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"SMOKE ERROR (unexpected): {exc}\n{traceback.format_exc()}")
        return 2
    print("SMOKE OK")
    return 0


def test_smoke_e2e():
    """pytest entry point."""
    assert main([]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
