# VMI_workflow Regression Safety Net

Baseline test suite pinning the behaviour of `VMI_workflow.py` /
`VMI_workflow_core.py` / `VMI_workflow_reconstruction.py` (baseline
2026-08-31; center-estimator pruning re-baselined 2026-09-01, see
"Baseline" below). Other agents refactoring the app must keep these tests
green.
The scientific numbers live in `golden_core.json` and `golden_smoke.json`;
do not regenerate them unless an intentional, reviewed behaviour change
occurred (see "Regenerating goldens").

## Files

| File | Purpose |
|---|---|
| `make_sample_data.py` | Deterministic synthetic VMI data triplet generator (seed 20260831) |
| `test_core.py` | Numerics golden tests for `VMI_workflow_core` (pairing, denoise checksum) + hardcoded lock tests for the two kept center estimators `quadrant_symmetry_center` / `polar_outermost_center` (pin pre-refactor results; run under pytest). The golden cases for the pruned estimators (`geometric_median`, `circle_fit_kasa`, `edge_circle_center`) were removed on 2026-09-01 |
| `test_smoke.py` | Offscreen end-to-end smoke test driving the real `MainWindow` through the full 7-step workflow |
| `golden_core.json` | Golden values for `test_core.py` (generated via `--update-golden`) |
| `golden_smoke.json` | Golden E2E numbers for `test_smoke.py` (generated via `--update-golden`) |
| `sample_data/` | Generated triplet (`synth_ar100_vmi*_DAn.dat`) + `generation_stats.json` manifest |
| `bench_core.py` | Plain A/B timing script for the center estimators (keeps the pre-hoist `quadrant_symmetry_center` as `_reference_*` copy) |
| `bench_rbasex_basis.py` | Times `run_rbasex_reconstruction` in fresh processes to verify the persistent rBasex basis cache (`~/.cache/vmi_workflow/abel_basis`); run twice, second run must be faster with identical peaks |
| `bench_drag.py` | Drag-interaction benchmark + ghost detector (2026-09-01): drives every draggable overlay through the real mouse-event handlers at the 16 ms timer cadence, reports per-frame ms, and compares the mid-drag canvas buffer against a forced clean reference render (axes interior must be pixel-exact; catches the historical ghost-ring bug). Run: `python tests/bench_drag.py [--frames 200] [--label AFTER]` (offscreen, ~2 min) |
| `_capture_center_locks.py` | One-shot recipe that captured the hardcoded center-estimator lock values from the pre-refactor code |

All three main app `.py` files are untouched by this suite.

## Running

From the project root (the repository checkout):

```bash
# full core numerics suite (~15 s)
python tests/test_core.py
# or: python -m pytest tests/test_core.py -q

# offscreen end-to-end smoke test (~60 s)
QT_QPA_PLATFORM=offscreen python tests/test_smoke.py
# or: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_smoke.py -q
```

`test_smoke.py` forces `QT_QPA_PLATFORM=offscreen` itself if not already set.
Both scripts are plain-runnable (`exit code != 0` on failure) and pytest
collectable. The smoke test writes session outputs into a temp dir
(`workflow_outputs` is resolved from the CWD, so the test `chdir`s away and
never pollutes the repository; set `VMI_SMOKE_KEEP_TMP=1` to keep the temp
workspace for debugging).

## Regenerating sample data and goldens

```bash
python tests/make_sample_data.py                 # regenerate tests/sample_data/
python tests/test_core.py --update-golden        # regenerate tests/golden_core.json
QT_QPA_PLATFORM=offscreen python tests/test_smoke.py --update-golden   # regenerate tests/golden_smoke.json
```

IMPORTANT: the goldens are only reproducible if the sample data is
byte-identical (fixed RNG seed). If you regenerate the sample data you MUST
regenerate BOTH goldens and expect every number to shift. Values are compared
exactly after rounding to 6 decimals.

## What the sample data looks like

~30,000 trigger rows; electrons in two anisotropic Newton-sphere rings
(r = 60 px with beta = +1.2 and r = 110 px with beta = -0.4, sigma_r = 4 px)
plus 4% uniform background inside r = 148 px, all around a true detector
center at **(128.0, 125.5)**. Each event carries a correlated ion whose TOF
has a main peak at 8250 ns (85%), a secondary peak at 11900 ns (10%) and a
flat 7000-14000 ns background (5%). Trigger rows are ~92.25% strict +1/+1
coincidences, plus deliberate +1/+2 and +1/+3 index-jump rows (feeding the
1e+2i / 1e+3i selectors) and NaN single-channel rows (chain breakers that
exercise the NaN handling and the "all valid rows" mode).

Trigger file column layout matches the real `Refence data` files:
`event_no, ion_tof, ion_index, electron_index` (the app reads columns
2-3 with `use_columns=(2,3)` and swaps to internal `[electron, ion]` order).
File naming follows the real suffix pattern so the historic QA script's
auto-discovery works: `synth_ar100_vmi_DAn.dat`,
`synth_ar100_vmi.lmf_elec_DAn.dat`, `synth_ar100_vmi.lmf_ion_DAn.dat`.

## Baseline (green) numbers, 2026-08-31

`test_core.py` / `golden_core.json`:

| Quantity | Value |
|---|---|
| coincidence 1e+1i pairs (in bounds) | 25658 |
| all-valid-rows pairs | 27841 |
| 1e+2i pairs | 310 |
| 1e+3i pairs | 123 |
| denoise checksum (fixed input) | signal 6000 -> denoised 5756.610539, removed 243.389461 |

`test_smoke.py` / `golden_smoke.json` (E2E, defaults: coincidence mode,
fine ROI [7900, 8600] ns, ring center start (126, 123), inner 118, outer 140,
outer-ring filter ON, bin 0.5):

| Quantity | Value |
|---|---|
| paired count (1e+1i) | 25658 |
| selected mask count (fine ROI) | 21871 |
| center estimate (quadrant_symmetry, 2026-09-01) | (128.679982, 125.321352), error 0.703 px from true center (pre-pruning edge_fit default was (126.634982, 124.088852), 1.963 px) |
| ring inner / outer counts | 21421 / 343 |
| denoised histogram sum | 21344.284303 (removed 76.715697) |
| rBasex peaks | r=61.0 (beta=-0.893267, i=2601.926298) and r=112.0 (beta=0.199947, i=1161.280871) |
| restored pair count after session roundtrip | 21871 |
| rBasex wall time | ~3-7 s (timing is NOT part of the golden) |

Re-baseline note (2026-09-01 center-estimator pruning): the default center
mode changed from `edge_fit` to `quadrant_symmetry`. Because
`estimate_center_once` writes the estimated center back into the
`circle_cx/cy_edit` widgets and `apply_circle_selection` reads them, every
value downstream of the ring center shifted deterministically (ring counts,
denoised sum, rBasex peaks). Values upstream of the center estimation (pair
counts, fine-ROI selected mask) are bit-identical to the 2026-08-31
baseline.

## Known app bugs worked around in the tests (do not "fix" the expectations silently)

1. **FIXED (2026-08-31): `polar_outermost` center mode used to crash with
   `NameError: source_label`**
   (`estimate_center_once`: `source_prefix` was defined but `source_label`
   was referenced; the same NameError also hit the (since 2026-09-01
   removed) edge_fit/geo_median FALLBACK path whenever no electron points
   fell inside the current inner ring; the default-mode fallback branch is
   now taken by `quadrant_symmetry`). The main workflow only exercises the
   default mode with a non-empty inner ring (cx=126, cy=123, inner=118) to
   keep the golden numbers stable; the previously-crashing paths are now
   covered by
   `run_regression_checks` in `test_smoke.py`:
   `check_polar_outermost_center` (polar ROI band [40, 130] so the
   formerly-crashing ROI branch runs), `check_ring_empty_center_fallback`
   (off-data inner ring -> "(fallback: full set)" branch) and
   `check_empty_selection_scatter` (0-event ion filter -> "No selected
   points" annotation + cleared colorbar).
3. **FIXED (2026-09-01): theta-guide drag freeze + hard crash (screen-only).**
   Dragging the radial-profile theta guide through 360 degrees after switching
   the electron panel to polar view froze the app and then segfaulted it: (a)
   the draw-event handler re-captured blit backgrounds on every draw during a
   drag (re-render storm), and (b) `restore_region` of a background captured
   at a different canvas size writes out of bounds on the real Qt backing
   store (offscreen Agg only clips, so the crash itself is not reproducible
   offscreen). Fixed in the app (ARCHITECTURE.md section 16i): geometry-safe
   blits, draw-event suspension for the whole drag session, self-healing
   recapture, and invalidation on content change. The offscreen guards are
   locked by `check_theta_drag_blit_safety` (suspend window, mid-drag resize
   rejection, press recapture, 360-degree sweep) and
   `check_compare_toggle_blit_invalidation` (compare refused without a
   reconstruction; invalidation helper clears both buffers + geometry keys).
2. **Session restore of the ring center has `%.6g` fidelity.**
   `_load_session_output_from_metadata_path` round-trips the center through
   the `circle_cx/cy_edit` QLineEdits (formatted `%.6g`), so the restored
   center equals the original only to ~6 significant digits. The smoke test
   therefore compares the restored center with a 1e-3 px tolerance (other
   restored quantities, e.g. denoised histogram sums and rBasex peaks, are
   compared exactly).

## Notes for later agents: driving the app offscreen

- **Async workers.** `load_cache`, `estimate_center_once`,
  `apply_circle_selection` and `run_reconstruction_now` each spawn a
  `QObject` worker on a `QThread` (`_LoadWorker`, `_CenterWorker`,
  `_ProjectionWorker`, `_ReconWorker`) and poll it with a 120 ms
  `QTimer` into the main window. Offscreen (no `app.exec()`), drive
  everything with:

  ```python
  deadline = time.perf_counter() + timeout_s
  while time.perf_counter() < deadline:
      app.processEvents()          # dispatches the 120 ms polling timers
      if done_predicate():
          break
      time.sleep(0.01)
  ```

  Each step's completion predicate: load -> `win.cache is not None and not
  win._load_busy`; center -> `not win._center_busy and win._center_worker_thread
  is None`; circle -> `win.centered_hist_data is not None and not
  win._circle_busy`; recon -> `win.rbasex_recon_result is not None and not
  win._recon_busy`. Call the public method first (e.g. `win.load_cache()`);
  the busy flag is set synchronously before it returns.

- **Modal dialogs.** Patch `QMessageBox.warning` / `QMessageBox.critical`
  (like the historic `qa_vmi_subplot_layout.py` does) BEFORE running the
  workflow; otherwise any app warning would block the offscreen run forever.
  The smoke test also asserts zero dialogs were raised.

- **File loading.** Set `win.file_paths = {"trigger": ..., "electron": ...,
  "ion": ...}` directly and call `win.load_cache()`; no need to touch the
  drop frames or file dialogs.

- **Fine ROI.** `win.ion_fine_xmin_edit.setText("7900")` +
  `win.ion_fine_xmax_edit.setText("8600")` then
  `win.apply_ion_fine_roi_from_inputs()` (equivalent to pressing Return in
  the UI). The selection lands in `win.ion_range` and `win._selected_mask()`.

- **Session roundtrip.** Save with `win.save_session_output()` (writes
  `<cwd>/workflow_outputs/<timestamp>_<tag>/session_data.npz` +
  `session_metadata.json`); restore without a file dialog via
  `win2._load_session_output_from_metadata_path(meta_path)`.
