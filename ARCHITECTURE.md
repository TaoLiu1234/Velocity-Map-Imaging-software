# VMI_workflow Architecture Document

> Generated: 2026-08-25 (Chinese original); fully translated to English on 2026-09-01 (see §16h)
> Scope: the three source files in the repository root + data directories (data files are not committed)
> Audience: future maintainers / students who want to understand how the system works

---

## 0. Runtime environment and dependencies

| Item | Value |
|---|---|
| OS | Windows 11 Pro (x64), AMD Ryzen 7 7840HS |
| Python | 3.11.11 (conda-forge) |
| numpy / scipy | 2.4.6 / 1.16.0 |
| PySide6 | 6.7.2 (matplotlib `QT_API=pyside6` + `QtAgg`) |
| pyabel | 0.9.1 |
| matplotlib | 3.10.8 |
| Optional (not used directly) | pandas 2.3.1, pyarrow 19.0.0 (verified by benchmark, see §6); numba unusable (incompatible with numpy 2.4) |

Data files: CSV text, 4 columns per row — `event number, ion TOF, electron index, ion index`; column values may be `NaN` (that channel is missing for the event). Reference files are 220–260 MB, tens of millions of rows.

---

## 1. Codebase overview

| File | Lines (2026-09-01) | Responsibility | Qt-dependent? |
|---|---|---|---|
| `VMI_workflow.py` | 23,797 | GUI, interactions, all workflow orchestration (a single `MainWindow` class) | yes |
| `VMI_workflow_core.py` | 3,235 | Pure numpy/scipy numerics: pairing, center estimation, polar projection, denoised binning | no |
| `VMI_workflow_reconstruction.py` | 358 | pyabel rBasex driver + peak extraction (the backward fallback engine was removed in §16) | no |

Design intent (file-header comment): keep the computation core separate from the GUI so it is easier to test and read. Actual state: the two computation modules indeed have no Qt dependency, but **the majority of the algorithms still live directly in `MainWindow` methods** (background fitting, TOF alignment, rotation, calibration, canvas interaction) — the separation is incomplete.

### 1.1 Top-level helper classes
- `FileDropFrame(QFrame)` (L162 at generation time): drag-drop or browse a single file; emits the `file_dropped` signal.
- `_SelectorToggleProxy` (L223): minimal stand-in for matplotlib selectors (only an `active` flag), used for the ion-TOF ROI custom picking.

### 1.2 Constants (throttling / performance guardrails)
L66–86 (at generation time) define ~20 tuning constants: `MAX_SCATTER_POINTS=25_000` (maximum scatter points), `MAX_ION_COINCIDENCE_POINTS=120_000`, `DRAG_PREVIEW_INTERVAL_MS=16`, `OVERLAY_EDIT_DEBOUNCE_MS=70`, `WHEEL_SCROLL_COALESCE_MS=12`, etc. (The former `SCATTER_HEATMAP_THRESHOLD=8_000` was removed in §16b together with `_plot_density_heatmap`; the former `ION_TOF_BG_POINTWISE_K=28` and its kd-tree point-wise backend were dead code, also deleted in §16.)

---

## 2. UI structure

```
MainWindow (1380×980)
├─ Top file bar file_bar_row: Files btn / Settings[Tab] toggle / Load / Process and Plot / Save Session /
│  Load Session / Clear / Trigger Mode dropdown
├─ Status bar: Status label + QProgressBar (hidden when done)
├─ plot_settings_splitter (QSplitter.Vertical)
│  ├─ plot_panel (plot area)
│  │  ├─ NavigationToolbar (matplotlib QtAgg)
│  │  ├─ plot_scroll (QScrollArea, both scrollbars always on)
│  │  │  └─ plot_canvas_host (QStackedLayout StackAll)
│  │  │     ├─ figure+canvas (27.0×8.6 inches)
│  │  │     └─ plot_scroll_preview_label (live bitmap preview mask during scroll gestures)
│  │  └─ h_view_slider (horizontal view slider, usually hidden)
│  └─ settings_panel (Settings Tray, min height 140 px, hidden at startup, shown with Tab)
│     └─ control_tabs (QTabWidget, 7 tabs)
│        ├─ File (data sources + workflow notes)
│        ├─ Ion Histogram (ROI / binning / m-q reference / background fit / display)
│        ├─ Ion Coincidence (TOF map parameters, TOF background model, TOF alignment fit)
│        ├─ Electron Scatter (center estimation, circle parameters, filters, polar ROI)
│        ├─ Ion Scatter (ion filters, rotation, TOF center correction)
│        ├─ Electron Binned Image (centered bin size, θ profile, voltage)
│        └─ Reconstruction (rBasex parameters)
└─ status bar (status_label + progress_bar)
```

### 2.1 Subplot grid (2×4 gridspec)
```
[ion_histogram     | electron_scatter | centered_bin      | rBasex recon]
[ion_tof_xy        | ion_scatter      | theta_profile     | rBasex radial profile]
```
All 8 axis objects are registered in `self.subplot_axes` and indexed by the refresh functions and the subplot save/copy logic. (The old layout was 2×5, including the backward recon and summary/info panels; those two panels and the entire backward mechanism were removed in §15/§16 respectively.)

### 2.2 Interaction layer
- Canvas global events: `mpl_connect` binds five events — `button_press/motion/button_release/pick/draw` — all funnelled into `_on_canvas_press/move/release/pick`.
- **Wheel scrolling**: a global `eventFilter` dispatches wheel events by widget identity to the scroll area or the canvas; inside the canvas the wheel enters **scroll-burst mode**: `_capture_plot_scroll_preview_pixmap` grabs the current canvas as a QPixmap laid over a top mask layer while the underlying `plot_scroll` scrolls fast, and on release `_flush_deferred_plot_scroll_restore` restores. A trick that keeps high-resolution matplotlib figures scrolling smoothly.
- **Drag previews**: all canvas drags (ring drag, ion filter-box drag, polar ROI, θ lines, rBasex range) only update `pending_*` variables; a throttled QTimer (16 ms) triggers the `_flush_*` overlay redraws (a blit fast path since §16f).
- **Overlay drawing**: `_draw_circle_overlay`/`_draw_ion_overlay` etc. are pure matplotlib artists; the `_present_*_from_background` fast path does `restore_region`+`draw_artist` blitting. Historically **everything was force-disabled on Windows by `_use_safe_*_redraw()`** (L8397+ at generation time), degrading to full-canvas redraws (`_draw_canvas_without_overlay_draw_event_sync` + background recapture) — one of the interaction-smoothness bottlenecks. (Since §14.2/§16f the blit path is the default on all platforms; `_use_safe_*_redraw()` now only gates the non-fast stable path.)

---

## 3. Data pipeline (the official 7-step workflow)

### Step 1 File loading and caching
`load_cache()`: reads the three files (`fast_read_csv_float64`, see §13.1) → `CacheData(trigger_indices[electron,ion], electron_points[i,t,x,y?], ion_points[...])`. `select_*` triggers pairing. Note: the trigger file is read as `[ion,electron]` and then swapped to the internal `[electron,ion]` order (see the code comment "Input trigger order switched for test").

- `np.loadtxt(usecols=...)`: note **it still parses every token of the whole line**. Benchmark: `np.loadtxt` (2M rows, 95.6 MB) 1.44 s; a real 260 MB file ~3–5 s. Historically the three files loaded serially and blocked the UI (WaitCursor); since §13.1/§14.2 the read runs on the `_LoadWorker` background thread.

### Step 2 Trigger (TDC) pairing
`select_increment_pairs` / `select_*_one_pairs` / `select_1e2i` / `select_1e3i` (core.py L41-190 at generation time) are all skeleton-vectorized: adjacent-row differences `Δe, Δi` are compared and rows satisfying conditions such as `(Δe,Δi)=(1,1)` are selected; 1e/2i and 1e/3i expand to `(e,i-1),(e,i)` or triple rows. Already O(N) with no Python loops — no obvious optimization headroom.

The resulting `paired_lookup_e_idx`/`paired_lookup_i_idx` are used directly as row indices into `electron_points`/`ion_points` — **no full-table copies are materialized** (`matched_electron` stays empty); downstream materializes on demand via `_paired_points(mask)`.

### Step 3 Ion-histogram selection
- Bin count, ROI (coarse x-range), fine ROI; drawn on `ax_hist_ion`.
- `_ion_hist_cache` (key = data_version + bins + coarse ROI + axis-transform tag + background key) caches counts/edges; incremental invalidation via `_invalidate_ion_hist_cache`.
- Additionally carries the **background-fitting subsystem** (§4.6) and the **m/q axis** (§4.5).
- `_selected_mask()` assembles the bool mask from the fine ROI + peak markers, cached under `_selected_mask_cache_key`.

### Step 4 Scatter filtering
`_selected_pairs_after_optional_ion_filter()`:
1. `_selected_mask` → the fine-ROI baseline;
2. if enabled, apply the TOF background keep mask (bg_keep);
3. `_paired_points(mask)` materializes the `(x,y,t)` electron points and ion points;
4. optional: ion rectangle/density filter (`_density_filter_mask`); optional electron density filter; returns a dict carrying the per-stage masks and lookup indices.

**Note**: density/filter masks are recomputed over **all** selected points on every round; `_density_filter_mask` internally uses `density_counts_from_bins` (2D bin count, linear) + top-k % or bottom-M exclusion. It uses `np.digitize` + `np.bincount`, O(n), but recomputes each time. (Historical: since §16d the selected-pairs derivation is memoized when the filter parameters and data version are unchanged.)

### Step 5 Electron-scatter center estimation (bundle)
`estimate_center_once` → one-shot estimation (single pass, no iterative loop); the `center_mode_combo` dropdown selects the algorithm
(after the 2026-09-01 center-estimator pruning only 2 modes remain, default `quadrant_symmetry`, see §16g):
1. ~~`centroid` - mean~~ (**REMOVED** 2026-09-01);
2. ~~`geo_median` - `geometric_median` (Weiszfeld fixed-point iteration)~~ (**REMOVED**, function deleted from the core module);
3. ~~`edge_fit` - `edge_circle_center` (edge envelope -> Kasa circle fit)~~ (**UI mode removed**; function kept as an internal seed helper);
4. `polar_outermost` - `polar_outermost_center` (frozen outermost-shell fan model + analytic-gradient optimization; still requires a Polar ROI band);
5. `quadrant_symmetry` - `quadrant_symmetry_center` (diagonal-quadrant symmetry on raw points; **new default**, no prerequisites).

Downstream, `_center_curve_metrics` uses `build_polar_histogram` to compute a "straightness" score for candidates; the acceptance condition is `_center_metrics_better` (combined score/sigma/valid). `polar_outermost` additionally has a multi-probe fallback (deterministic probes under the polar ROI).

Historically all of this ran synchronously on the GUI thread with a full `_refresh_plots` afterwards; since §14.2 the estimation/metrics/probes run on the `_CenterWorker` background thread.

### Step 6 `apply_circle_selection` (core projection)
1. Get the filtered pairs (Step 4);
2. Ring selection: `dist2 <= inner²`; the optional `outer_ring_filter` collects outer-ring noise points;
3. Centering: `centered_signal` = electron − center;
4. `build_denoised_centered_histogram` (core.py, details §4.8): the uniformized outer-ring noise density is subtracted from the inner-ring bins; returns a dict with `hist_denoised`, `hist_signal`, edges, bin_size;
5. Clears the old reconstruction/profile selection state, `_refresh_after_circle_clear(partial)`.

### Step 7 Reconstruction
`run_reconstruction_now` (state as of 2026-08-31; backward was removed in §16):
- sets `_get_rbasex_settings`;
- rBasex runs on the **background thread** `_ReconWorker`, calling pyabel `rbasex_transform(direction="inverse")` (see §13.3b) with a progress callback of 10–90%; the click returns immediately;
- on completion the main thread's `_collect_recon_results` refreshes all panels; repeated clicks are gated by `_recon_busy`.

---

## 4. Key algorithms in detail

### 4.1 Trigger pairing (core.py:41–189)
- `_select_strict_delta_pairs`: vectorized adjacent-row differences. `valid` requires both rows' columns to be non-NaN; compares after `rint` rounding. O(N).
- 1e2i expansion: for each matched row output `(e,i-1),(e,i)`, clipping `i<0`.
- Pros: pure vector ops, extremely fast; unified `delta_e/delta_i` parameters across modes. Cons: adjacent rows must be **contiguous data rows** (pairs across file boundaries and the last row are lost); the data file is assumed sorted with no sort defense; row deltas depend on the raw CSV order — if a file was stitched from multiple acquisitions (cf. `merge_trigger`), fake pairs appear at the stitch boundaries. The `merged_trigger` filename pattern suggests the real data is stitched; worth noting.

### 4.2 Reconstruction (reconstruction.py)
**(a) rBasex**: `abel.rbasex.rbasex_transform(direction="inverse", order=settings["order"], odd=..., reg=..., rmax=...)`. Originally `basis_dir=None` (**the basis set was recomputed every time — pyabel's default cache lives in the `abel` data directory; since §16d it is explicitly persisted under `~/.cache/vmi_workflow/abel_basis`**), `rIbeta()` extracts `(r,I,β,...)`, `r *= bin_size`; peak extraction:
`extract_peak_r_beta`: Gaussian smoothing → normalize → `scipy.signal.find_peaks` → valley clipping → per-peak `np.trapezoid` interval area. Returns `(r,β,i,area)`.

**(b) backward model (built-in fallback) — removed entirely in §16; kept here as history**: `Abel_backward_reconstruction.py` **did not exist**, so **what actually ran was the in-file fallback engine** (three phases, L56+ at generation time):
- Phase0 `_init_shared_data`: Cartesian–polar cubic interpolation `map_coordinates(order=3)`, radially averaged profile, tail (outside 80% radius) noise std;
- Phase1 `_phase1_radial_analysis`: smoothing + FWHM peak localization + `find_peaks`, filtered by mask_radius; peaks → (r,σ,amp,SNR);
- Phase2 `_phase2_angular_analysis`: multi-radius mean polar-angle spectra → `P2(cosθ)` least-squares `[1,P2]` fit → β=c2/c0 (constrained to −2..2);
- `reconstruct_2d_from_params`: superimposes per-peak Gaussian rings `amp·exp(−(r−r0)²/2σ²)·(1+β·P2)`.
- Note: **the full forward-fit engine claimed by the early docs never existed**; this fallback was a "parametric Gaussian-ring fit", not a true Abel inverse transform; many settings (`reg`/`baseline`, etc.) were actually unused.

### 4.3 Center estimators (core.py)
| Estimator | Algorithm | Complexity | Pros | Cons |
|---|---|---|---|---|
| ~~`geometric_median`~~ (**DELETED** 2026-09-01, see §16g) | Weiszfeld iteration | O(k·n), k≤120 | noise-robust | insensitive to symmetric uniform-ring data, biased toward the centroid |
| `circle_fit_kasa` (internal helper, not a UI mode) | algebraic least-squares circle | O(n) | fast | Kasa has a systematic offset for short arcs / noise (algebraic bias) |
| `edge_circle_center` (internal seed helper, **UI mode removed** 2026-09-01, see §16g) | outer-envelope per angle bin → Kasa circle fit, 3 refinement iterations (**2°/bin, `angle_bins=180` default**) | worst case O(iterations·bins·n) (per-bin `np.where` rescan) | fast, reuses an existing fit | **keeps only the single farthest point per bin — fragile to outliers**; degrades with incomplete angular coverage |
| `quadrant_symmetry_center` (**default UI mode**, see §16g) | diagonal-quadrant raw-point nearest-neighbour matching (180-degree rotation symmetry), single hoisted KD-tree | shared cKDTree built once + per-candidate queries | good for ring distributions, no prerequisites | depends on the seed radius estimate |
| `polar_outermost_center` (L990 at generation time) | frozen "outermost ring" scatter model, `_scatter_peak_line_loss_grad` analytic gradient + `_optimize_scatter_peak_line_model` monotone updates | each loss round recomputes every point's `(dx,dy,rr,θ)` | stable convergence | **no cross-iteration caching; each loss repeats O(n) polar-coordinate work** + model rebuild |
| `_iterative_outer_roi_edge_circle_center` (L1922 at generation time) | narrow annular-ROI contrast map → smoothed circular path → circle fit → iterate | each iteration redoes the full contrast map | stable for manual-ROI scenarios | bottlenecked by the 2D-histogram rebuild (the full contrast map) |

Shared preprocessing: `points[::step]` subsampling to ≤ ~64k points before entering the estimators (`_scatter_peak_line…`).

### 4.4 Polar histogram and peak line
`build_polar_histogram` (L819 at generation time): bins (x,y) by `θ=atan2`, `r=hypot` around a given center into a 2D heatmap (θ columns are set by the caller; the r bin count is dynamic); `_select_polar_peak_line` finds the peak per θ column; `*_loss_grad` uses "peak position horizontal straightness" as the tangential constraint (straightness) and differentiates; the gradient feeds `∂r/∂center` into analytic formulas.

### 4.5 m/q calibration
`_ion_mq_calibration_params` (L 8980 at generation time): quadratic law `m/q = a·t² + b`, `b=0`; a reference point `(m/q_ref, t_ref)` fixes `a = m_ref/t_ref²`; given a TOF range, `[t_lo,t_hi]→[m_ref−½, m_ref+½]` is least-squares calibrated for `a`. Integer m/q bins ↔ corresponding TOF intervals (deterministic). Strength: physically clear; weakness: the fit has a single free parameter — any TOF offset (e.g. a time-zero delta) becomes a systematic error.

### 4.6 Ion-histogram background fitting (L~121-13690)
Two parallel routes + one fallback:
1. Envelope baseline `_estimate_ion_hist_log_envelope_baseline`: log-space rolling quantiles (pure-Python rolling loop → slow) → separable smoothing → weighted isotonic (monotone increasing) regression → exponential → running minimum (`np.minimum.accumulate`) gives the **under-signal monotone baseline**.
2. Adaptive components `_fit_ion_hist_background_curves_adaptive_raw`: candidate power set + soft offsets, weighted least-squares linear components (`_solve_weighted_nonnegative_components`), scored under the target; then `_project_nonnegative_components_under_target` enforces non-negativity and "do not cross the unique signal region".
3. Non-parametric fallback `_build_nonparametric_bg_state`: log-envelope smoothing profile.

Scoring: the adaptive route uses under-target SSE; the NNLS search selects the model by BIC (complexity enters the selection).

Performance ⚠️: `_rolling_quantile`, the isotonic block merging and the per-shape-profile fits are **Python loops + O(n) per iteration**; with windows of tens to hundreds of thousands of points this can reach hundreds of ms to seconds. **`_ensure_ion_hist_background_fit` only recomputes when the cache key changes** — otherwise it reuses, which is the correct cache design.

### 4.7 Ion TOF background model (the ion TOF map, visualization)
(the code that actually runs; not the point-wise kd-tree):
- `_fit_ion_tof_background_model_raw`: first smooth the XY histogram (density), then `_fit_radial_floor_profile` (radial floor profile); per point `score = bg_density/all_density`; `_choose_score_threshold` picks the threshold automatically from quantiles / histogram mass; low-score events are masked as background keep.
- `_ensure_ion_tof_bg_model` memoizes by (source key, xy transform key, params).
- **Dead code (deleted, §16)**: the `_adaptive_xy_kde_density` / `ION_TOF_BG_POINTWISE_K` (K=28) point-wise KNN path was defined with no callers; the histogram-density + radial-floor variant is what actually ran (faster and more stable, but coarser). Both were removed in §16 (the leftover feature-construction helper `_adaptive_xy_bg_features` was finally deleted in §16h).

### 4.8 Denoised centered binning (core.py:2946–3018)
```
signal histogram (signal_hist)  ← centered_signal
outer-ring density = noise_count / (π(outer²−inner²))
expected_per_bin = density × bin_size²
inner-ring mask (inside circle) subtracts expected_per_bin → clamp ≥0
removed_total = Σsignal − Σdenoised
```
- Pros: a single uniform-type subtraction, extremely fast, analytically transparent.
- Cons (accuracy): **assumes the background is uniform inside the inner ring** (real VMI background fluctuates radially and is often elevated near the center); no two-sided correction of Poisson counting noise; `denoised<0→0` causes truncation bias (negative values discard information, the mean shifts up). Improvements: a radially adaptive background or a Poisson correction as a function of r (see §10 improvement items).

### 4.9 Rotation / alignment
- `_apply_ion_rotation` (L8152 at generation time): rotates `(x,y)` by `rotation_deg` around `ion_rotation_center`; `_transform_ion_xy` cascades: rotation → TOF-alignment shift → TOF Z centering.
- `_fit_ion_tof_main_line`: histogram2d downsampling → smoothing → per-column argmax ridge → MAD-robust weighted linear regression (line fit); `_fit_ion_tof_box_density_line` per-box density top-%. Results are stored in `ion_tof_fit_result_by_axis`; applying the fit shifts `(x,t)` so the line is horizontal.
- The **transform math used to be duplicated 3–4 times within the same file** (`_apply_ion_tof_alignment_to_xy`/`_apply_ion_scatter_tof_center_to_xy`/`-terms`/`_ion_tof_display_coord_values`) — a maintenance risk. (Deduplicated in §16c.)

---

## 5. Caches and invalidation

| Cache | Key | Invalidated by |
|---|---|---|
| trigger pairing `_pair_cache_*` | (mode, trigger_ref object identity) | `process_and_plot` / `clear` |
| ion histogram `_ion_hist_cache` | (data_version, bins, coarse ROI, axis_tag, bg key) | `process_and_plot` or control changes |
| fine-ROI selection `_mask_cache` | (data_version, ROI, peak marker, bg key) | ROI/marker changes |
| scatter display `_current_scatter_display_*` | (selection, filters, subsample) | filter/control changes |
| TOF background model | (source data, xy transform, params) | explicit recompute via `fit_ion_tof_bg_model` |
| TOF alignment result | (axis, transform) | fit/clear |
| rBasex | persisted basis: `basis_dir=~/.cache/vmi_workflow/abel_basis` (in-process global + cross-process disk cache) | — |
| ion_tof_xy map | (data_version, pair-table id, paired_count, n, **coarse ROI**, **BG-keep state**, axis, bins, z-range, rotation/alignment terms) (fixed 2026-08-31: the key previously lacked the ROI/data fingerprint) | invalidated on any data or selection change |

The overall cache design is mature (versioned keys), but the `ion_tof_xy` cache key used to omit data dependencies, `display_data` was recomputed on every `_refresh_*` (duplicate offsets), and the background-fit isotonic/rolling loops are not reused as cached objects. (The first two were addressed in §16d.)

---

## 6. Measured performance (pandas/pyarrow comparison)

| Operation (2M rows, 95.6 MB) | Time |
|---|---|
| `np.loadtxt` (then-current) | 1.44 s |
| `pandas.read_csv(engine="c")` | 1.33 s |
| `pyarrow.csv.read_csv` | **0.15 s** |

Conclusion: per-file parsing differences are limited, but the serial load+parse of three 260 MB files still dominated startup; pyarrow brings a **~10x read speedup**. The more important fact: **synchronous blocking on the GUI thread was the top UX problem**; IO belongs on a background thread. (Addressed in §13.1/§14.2.)

### Bottleneck list (by priority)

| # | Location | Problem | Impact |
|---|---|---|---|
| P0-2 ✅ resolved (§13.3b background thread; §16 backward removal) | `run_reconstruction` L~18 | was synchronous rBasex (basis generation is inherently slow) + backward, no worker | froze the UI for seconds to tens of seconds |
| P0-3 ✅ resolved (§14.2) | `_use_safe_*_redraw()` Windows | forced full-canvas redraw instead of blit; every drag did a full draw + background recapture | laggy drag interaction |
| P0-4 ✅ mostly resolved (§16) | `_selected_pairs_after_ion_filter` / `display_data` | re-did filtering + density binning on every `_refresh_*`; several refresh sites duplicated a 40-line scatter block | redraw storm on hot filter updates (selected-pairs derivation now memoized; refresh tails deduplicated) |
| P0-5 ◐ partially resolved (§16) | center estimators: `quadrant_symmetry` rebuilt a cKDTree per candidate | O(k²·n)-class (now a single hoisted KDTree, bit-identical results) | center estimation took seconds (polar_outermost still has no cross-iteration caching) |
| P1-1 | `_fit_ion_histogram_background_rules_*` | Python-loop rolling quantiles / isotonic | 100 ms–seconds |
| P1-2 | `build_denoised_centered_histogram` | rebuilds the meshgrid each run; acceptable for a single apply | low |
| P1-3 ✅ resolved (§16) | `run_rbasex` `basis_dir=None` | no explicit basis cache management (relied on pyabel's internal disk cache) | first reconstruction slow (now persisted under ~/.cache/vmi_workflow/abel_basis) |
| P1-4 ✅ resolved (§16) | `_ion_tof_xy_cache` key lacked data | data changes still hit the old map when params were unchanged (the key now carries coarse-ROI/BG-mask fingerprints) | data inconsistency |

---

## 10. Accuracy risk list

1. **The denoise model assumes uniform inner-ring noise** (§4.8): ignores radial fluctuation; may under-/overestimate the bases of signal peaks → affects rBasex input accuracy.
2. **Kasa circle fit has an algebraic bias** (core L252-293): biased center estimation for low-SNR, short-arc data.
3. **edge_circle_center keeps only the farthest point per angle**: outer-edge outliers pollute Kasa directly.
4. **(historical, removed in §16) the backward engine was a simplified Gaussian model** (§4.2b): the returned β came from per-peak local angular fits; the global β(r) was `Σβ·radial/Σradial` weighted, not a true Abel reconstruction → large divergence from rBasex, especially with overlapping/shadowed peaks. The engine, its UI and session keys were deleted in §16; reconstruction now runs rBasex only.
5. **m/q quadratic law `b=0`**: a TOF zero offset gives a systematic bias; range calibration uses least squares but provides no regression-error feedback.
6. **Background-fit under-target SSE**: naturally pushes the baseline below the signal (underestimates the background) — valleys between signal peaks may be underestimated.
7. **Peak-area fringes**: `extract_peak_r_beta` falls back to `intensity[idx]` for the area; a hard clip of `beta_profile` to −2..2 distorts strongly anisotropic distributions.
8. **Ion/electron indices used directly as indexes**: out-of-range is clipped without warning to the user; most indices are paired but some sample counts may be dropped.

---

## 9. Maintainability issues

1. **Giant God class**: `MainWindow` is a single class of ~23k lines (2026-08-31; ~23.8k after §16f/§16g). Method boundaries are clear but **responsibilities are not separated**: pairing/coordinates/fitting/plots/caching all live in one class. Low-risk refactor: split into domain mixins or utils (keeping import compatibility).
2. **Duplicated code**: 1) transform math ×3–4; 2) the electron/ion scatter drawing blocks in `_refresh_plots` duplicate `_refresh_scatter_panels_only` by 40+ lines; 3) the four pairing modes' progress-callback closures are nearly identical; 4) the two colorbar delete+create patterns. (Items 1, 3, 4 and the refresh tails were deduplicated in §16c.)
3. **Dead code (revised 2026-08-31)**: `_SelectorToggleProxy` **is live code** (defined L217, created L2540/2547 at generation time; `.active` is consumed by the TOF-ROI/TOF-background picking in the canvas event handlers L~19063/19088, `.clear()` called on clear) — this section's earlier "may have no users" judgment was wrong. `_clear_ion_tof_fit_preview_background` was renamed `_invalidate_ion_tof_fit_preview_background` (L3549, 5 call sites, live). The `_adaptive_xy_kde_density`/`ION_TOF_BG_POINTWISE_K` point-wise KNN backend was confirmed dead and deleted in §16 (its leftover feature helper `_adaptive_xy_bg_features` in §16h); the probe logic in `_center_curve_metrics` is used only by a few modes (kept).
4. **External-file dependency (resolved, §16)**: `Abel_backward_reconstruction.py` did not exist; the module try-import and the entire backward fallback engine were removed with §16.
5. Constants / hard-coding: `sys.platform` branches, scale constants scattered through methods (no named constants).
6. No unit tests (`VMI_workflow_core` was designed to be testable but the repo had no test files — unverified at the time; since §16e the repo has `tests/test_core.py` + `tests/test_smoke.py`).

---

## 11. Improvement roadmap (interaction logic unchanged)

### P0 performance (high leverage, low risk)
1. **Async loading + pyarrow/pandas C parser** (✅ done, §13.1): move `load_cache` onto a `QThread`/`concurrent.futures`; use `pyarrow.csv.read_csv` (or the pandas C engine) + convert results to numpy; keep the progress bar, UI does not freeze; column selection via pyarrow `include_columns`. **Risk**: none, a pure read-path replacement; keep dtype float64.
2. **Async reconstruction** (✅ done §13.3b; backward removed in §16): move rBasex onto a background thread and return to the main thread to refresh; the canvas is untouched meanwhile, spinner in the status bar. **Behaviour kept**: results identical after the click.
3. **Windows safe-blit rework** (✅ done §14.2): a more robust path — wrap `canvas.copy_from_bbox` + `restore_region` + `draw_artist` in try/except with automatic fallback; or defer overlay drawing into batched `timerEvent` merges. **Risk**: medium (Windows-specific painting), needs manual verification.
4. **`display_data` derived cache** (◐ mostly done §16: the selected-pairs derivation is memoized): before `_refresh_*`, check `(selection_version, filter params, data_version)`; if unchanged, reuse the old `electron_show/ion_show`/color arrays and skip the repeated density filtering.
5. **Center-estimation cache** (◐ partially done §16: quadrant_symmetry single-tree hoist, bit-identical): after `_scatter_peak_line_model` freezes the point set, precompute the `(r²,θ,cosθ,sinθ)` constant vectors; all `loss/grad` iterations reuse them; `quadrant_symmetry` uses `KD.query_ball_point` per candidate but the **KDTree is built only once** (the points do not move within the candidate-center set!) — a major direct speedup.
6. **`basis_dir` explicit persistent cache** (✅ done §16): set `basis_dir` under the cache directory; the first rBasex is slow, afterwards millisecond-level.
7. **`ion_tof_xy` cache key gains a data fingerprint** (✅ done §16): `hash(data_version + coarse_mask sum)` forces invalidation.

### P1 accuracy
- Denoising → radially adaptive (annular density per radius) or a Poisson model; keep the API `build_denoised_centered_histogram(signal, noise, inner, outer, bin)` output dict structure unchanged. Default `flat uniform`, an optional `'radial'` mode behind a switch (not changing the default avoids a scientific break).
- Circle fit Kasa → Pratt/Taubin (algebraically unbiased) or `scipy.optimize` nonlinear least squares (radius weighting), with `circle_fit_kasa` kept as a compatible fallback.
- `edge_circle_center`: keep a top-N% quantile point set per bin instead of the single farthest point.
- m/q: allow `b≠0` (trade-off: needs UI; keep the default).

### P2 UI modernization (touch no interaction logic)
1. **Global QSS theme**: `app.setStyleSheet` in `main()` (light, modern: rounded corners, contrast, a "Segoe UI Variable" font stack) replacing the scattered inline styles; keep the existing `QTabWidget`/GroupBox structure rules.
2. **matplotlib styling**: a global `rcParams` theme (axis colors, gridlines, font sizes, figure background #fafafa), subplot/axes layout untouched.
3. **Panels and status bar**: finer progress bar (0-100 segmented colors), softened status hints; icon buttons stay text-based.
4. **Plotting speed**: blit/partial redraws for the already-cached image-type panels (centered bin, recon, hist); avoid rebuilding all text objects every time.
5. Keep the scroll preview (already a highlight).

### P3 frontend
- Split `MainWindow` into several large classes (interactions, layout, core, plotting), keeping public method/class names unchanged → behaviour unchanged but testable.
- Turn the public algorithms (trigger pairing, circle centers, background) into pure functions of `VMI_workflow_core` with unit tests (np.testing).
- Remove dead code (listed in §9.3), grepping call sites before each deletion.
- `.vendor_site` (L1) and `_prepend_local_vendor_sitepackages` stay as no-ops when the directory is absent.

---

## 12. Interaction contract

The following will be treated as the "interaction contract"; optimizations must not change:
1. The seven-step workflow button order and functions exactly (Load→Process→ROI→Fine→Filter→Estimate→Apply+Bin→Recon).
2. All `QLineEdit` text inputs remain live listeners with unchanged wiring (`textChanged` triggers overlay redraw/recompute).
3. Drag gestures (ring, filter box, θ, rBasex range, markers, TOF box) keep their semantics.
4. Session save/restore format stays compatible (the existing npz + meta JSON).
5. Window default size/docking/tab structure unchanged.
6. Scientific output: peak r/β/intensity values reproducible (same data + same parameters).

---

## 13. Changes implemented in this round (2026-08-25)

> Follows the §12 contract: no interaction logic changed, no scientific-numerics path changed.

### 13.1 High-speed data loading (performance)
- `VMI_workflow_core.py`: new `fast_read_csv_float64(path, *, n_columns, use_columns)`.
  - Prefers the **pyarrow C++ CSV parser** (SIMD; parses a 260 MB file ~10x faster than `np.loadtxt`);
  - `NaN` text parses correctly to float64 NaN (real data contains `NaN`);
  - automatic fallback on parse errors (exactly the original `np.loadtxt` semantics); functionality intact without pyarrow;
  - column selection follows the `use_columns` output order, compatible with trigger files (4 columns, take (2,3)) and point files (3 columns, take (0,1,2)).
- `VMI_workflow.py::load_cache` switched to `fast_read_csv_float64`; the segmented progress-bar updates are unchanged.
- Measured (three real reference files, ~24M rows total): the 8.6M-row trigger went from `np.loadtxt` 1.53 s → 0.65 s; the read portion of all three files totals ~1.4 s.
- Fallback path compatible (`np.loadtxt` as before when pyarrow is unavailable).

### 13.2 UI modernization (QSS + matplotlib theme)
- `main()` gained a global `app.setStyleSheet` light modern theme: rounded buttons/inputs, focus highlight, flat panels, a Segoe UI Variable font stack, rounded checkbox indicators, modernized scrollbars, dark ToolTips.
  Scoped to generic control types; the existing `QTabWidget/GroupBox/SettingsPanel` inline styles keep priority.
- The end of `MainWindow.__init__` gained a `matplotlib.rcParams.update` (light canvas, light-grey grid, clear tick/label colors), touching no axes layout/geometry parameters.
- Visual verification: the window background `#f7f8fa` applies; all subplot placeholders render correctly.

### 13.3b Background-thread reconstruction (performance)
- `run_reconstruction_now` rewritten to **execute on a background thread**:
  - new `_ReconWorker(QObject)` helper: pure numpy/pyabel computation, **touches no widgets**; progress/results published via in-memory fields and polled by the main thread (`_recon_progress_timer`, 120 ms);
  - new `_collect_recon_results` collects the results on the main thread and refreshes the panels; threads are gracefully reclaimed via `quit()+wait(1500)`+`deleteLater`, not relying on the lossy `finished` signal;
  - clicking `Start Reconstruction` now **returns immediately** (UI not frozen); when the background finishes the main thread refreshes, and the results are **bit-identical** to the old synchronous path;
  - repeated clicks are gated by `_recon_busy`; stable across consecutive runs.
- Measured (reference data + 145×145 histogram): click returns in 0.01 s, completes in 1.7 s; rBasex 2 peaks, backward 1 peak, no Python warnings.

---

## 14. App-wide 30–60 Hz interaction optimization (2026-08-25)

> Goal: all interaction animations/refreshes/waits at 30 Hz or above (the user accepted 30 Hz), **without changing any interaction style and without losing accuracy/clarity**.

### 14.1 Root causes (measured)
- When dragging overlays, Windows `_use_safe_*_redraw()` was always True → every drag did a **full canvas.draw() + background recapture**, measured **183–408 ms/frame (≈5 fps)**.
- `_draw_axes_immediate` / `_draw_axes_preview` forced a full `canvas.draw()` on Windows (comment: "QtAgg partial blits crash") → all partial refreshes also repainted fully.
- With 6.86M paired rows, scatter/projection materialization plus full repaints made "refresh after operation" take seconds.

### 14.2 Changes
1. **Deleted the Windows full-repaint branch**: `_draw_axes_immediate` / `_draw_axes_preview` no longer degrade by platform; both use the partial `ax.draw(renderer) + canvas.blit(bbox)` path (with its own try/except full-repaint fallback).
2. **Overlay dragging at 60 Hz**:
   - new `_present_circle_overlay_from_background` (mirroring the ion version): `restore_region + draw_artist + blit`; dragging no longer repaints fully.
   - the fast_drag branches of `_update_circle_overlay_only` / `_update_ion_overlay_only` now capture the background only when missing (`_capture_scatter_blit_backgrounds`), otherwise paste the partial update directly.
   - `_ensure_scatter_overlay_backgrounds` reuses the cached background during a drag instead of recapturing every time.
   - `_flush_ion_rotation_preview` switched to a single-axis partial redraw.
   - the ion TOF fit preview already had partial blitting (enabled).
3. **Large point clouds auto-switch to heatmap**: `_plot_density_scatter` switched to `_plot_density_heatmap` (image rendering, 61 fps-class) when `count > SCATTER_HEATMAP_THRESHOLD (8000)`; low counts kept the point scatter (no clarity loss). The heatmap path existed in the code but was never wired up; it was wired here. (This branch was reverted in §14.4 and finally deleted in §16b.)
   - **`load_cache` → new `_LoadWorker(QObject)`**: the three large file reads/validation run on a background thread with progress polling; the cache is installed on the main thread when reading completes. UI responsive throughout.
   - **`estimate_center_once` → new `_CenterWorker(QObject)`**: geometric center estimation/metrics/deterministic probes on a background thread; the main thread keeps only input preparation (selection materialization, with progress) and writing results back to the canvas. Bit-identical output to the original logic (pure computation, no Qt).

### 14.4 User-reported regression fixes (2026-08-25, second round)
The user reported two regressions introduced by this round; all fixed and verified:
1. **electron/ion scatter backgrounds turned black, points hard to see**: the "large point clouds auto-switch to heatmap" change had replaced the point scatter with a dark density map. That branch was reverted, restoring the original PathCollection point rendering (colored density points on a white background). Verified: images=0, PathCollections=2 on both axes.
2. **Dragging the filter showed two circle centers / old-ring ghosts**: fast-blit `restore_region+draw_artist` pasted stale overlay pixels on a real canvas. The circle/ion fast_drag was switched to **whole-axis `_draw_axes_immediate` partial redraw** (automatic full-repaint fallback on failure), eliminating the ghosts. Verified: over a 30-frame drag the circle center marker stayed at exactly 1 and the ion rect at exactly 1.
5. **Subplots rescale after toggling the Tab (the real cause of Tab lag)**: the panel height change stretched the canvas via the viewport, forcing matplotlib to re-layout all subplots. Fixed by **keeping the canvas size stable** (the viewport no longer resizes the canvas; `_configure_plot_canvas_size` re-targets only on explicit window-mode changes). Verified: canvas 2214×705 unchanged across Tab toggles, all 8 axes positionally identical, reveal takes only 22 ms.
6. **The electron/ion scatter filter rings were covered by data points**: rings/filters were created without zorder, below the point cloud's zorder=2. Now inner/outer rings z=10, circle-center marker z=11, ion filter z=10, filter center z=11. Verified: all >2, drawn above the data.

## 15. Layout adjustment: removal of the Backward Recon and Summary panels
- As requested by the user, the `ax_reserved_bottom` (Backward Recon) and `ax_info` (Summary) panels were removed.
- gridspec changed from 2×5 to **2×4**:
  ```
  row0: [ion hist | e scatter | centered bin | rBasex recon]
  row1: [ion-tof xy | i scatter | theta profile | rBasex radial profile]
  ```
- **The rBasex radial profile moved to the old Backward Recon position (row1, col3)**; the former [0,4] slot is gone.
- `ax_reserved_bottom`/`ax_info` set to None; all downstream references carry None guards (`_plot_reconstruction_panel`/`_plot_info_panel` and the drawing/marker/placeholder logic); the `summary`/`backward_reconstruction` keys were removed from `subplot_axes`.
- Verified: the figure has exactly 8 axes in 2×4, the radial profile sits bottom-right (0.80, 0.08); the full workflow + session roundtrip run without warnings.

### 13.4 Follow-up suggestions (partially implemented in the rounds above; see §11/§16)
- ~~move rBasex/backward reconstruction and center estimation to background threads~~ (✅ done §13.3b/§14.2; backward removed in §16);
- ~~Windows safe-blit → partial redraw~~ (✅ done §14.2);
- ~~`_selected_pairs_after_optional_ion_filter` derived cache~~ (✅ done §16 memoization);
- Kasa→Pratt/Taubin circle fit, radially adaptive denoising (require scientific confirmation, not implemented);
- split the `MainWindow` giant class, add unit tests (unit tests are in place: `tests/test_core.py` + `tests/test_smoke.py`, see §16e).

The following will again be treated as the "interaction contract"; optimizations must not change:
1. The seven-step workflow button order and functions exactly (Load→Process→ROI→Fine→Filter→Estimate→Apply+Bin→Recon).
2. All `QLineEdit` text inputs remain live listeners with unchanged wiring (`textChanged` triggers overlay redraw/recompute).
3. Drag gestures (ring, filter box, θ, rBasex range, markers, TOF box) keep their semantics.
4. Session save/restore format stays compatible (the existing npz + meta JSON).
5. Window default size/docking/tab structure unchanged.
6. Scientific output: peak r/β/intensity values reproducible (same data + same parameters).

---

## 16. Release hardening (2026-08-31)

> This round's goal: regression fixes, dead-code removal, deduplication, performance wrap-up and repository clean-up for the public GitHub release.
> The interaction contract above was honoured throughout; the scientific-numerics paths are locked bit-exactly by `tests/golden_core.json` / `tests/golden_smoke.json`.

### 16a. Regression fixes
- **`NameError: source_label` in `estimate_center_once`**: the polar-ROI path and the ring-empty fallback path referenced the undefined `source_label` (only `source_prefix` was defined). Both crash paths were fixed and are covered by `tests/test_smoke.py::run_regression_checks` (`check_polar_outermost_center` / `check_ring_empty_center_fallback`).
- **Startup `__init__` calls restored**: placeholder panel rendering (`_draw_placeholder`), the trigger-mode combo labels (`_update_trigger_mode_combo_labels`, with the "[events: n/a]" suffix at startup) and TOF control syncing are all called at startup again (locked by `check_startup_placeholders`).
- **Empty-selection scatter branch restored**: when the ion filter selects 0 events, the electron scatter shows a "No selected points" annotation and clears the colorbar while keeping the grey context points (locked by `check_empty_selection_scatter`).
- **Status-bar wording aligned** with the classic version's status texts.

### 16b. Removals
- **The entire backward-recon mechanism** (§4.2b): UI controls, `_get_backward_settings`, the built-in phase0/1/2 fallback engine, the compute path and session keys all deleted; no "backward" references remain in the three main files. Session restore stays tolerant of **legacy sessions** (containing backward keys) — unknown keys are ignored without error. Reconstruction now has exactly one path: rBasex.
- **summary/backward zombie scaffolding**: leftover drawing/placeholder branches from before §15 (`_plot_info_panel`-related logic etc.) cleaned out.
- **Dead KDE backend**: the `_adaptive_xy_kde_density` / `ION_TOF_BG_POINTWISE_K` point-wise KNN path (§4.7, §9.3).
- **Orphan blit helpers**: caller-less `*_from_background`/background-capture remnants.
- **`_plot_density_heatmap`**: the "large point cloud → heatmap" path wired in §14.2 and reverted in §14.4 was deleted for good (scatter keeps PathCollection points).

### 16c. Deduplication (no behaviour change)
- **TOF transform helpers**: the duplicated transform math of `_apply_ion_tof_alignment_to_xy` / `_apply_ion_scatter_tof_center_to_xy` / `-terms` / `_ion_tof_display_coord_values` (the 3–4 copies flagged in §4.9) converged into shared helpers; `check_ion_tof_alignment` locks the transform output to 9 decimals.
- **Pairing progress factory**: the four pairing modes' near-identical progress-callback closures unified into a factory function.
- **Colorbar helper**: the two "delete + recreate colorbar" patterns merged into a shared helper.
- **Refresh-tail helper**: the duplicated electron/ion scatter refresh tails (the 40+ duplicated lines of §6 P0-4) extracted into a common tail function.

### 16d. Performance
- **`ion_tof_xy` cache-key fix** (§6 P1-4): the key now includes coarse-ROI and BG-keep mask fingerprints; the same point counts with different selections no longer hit the old map (locked by `check_ion_tof_xy_cache_invalidation`).
- **Selected-pairs derivation memoization** (§6 P0-4): when filter parameters and the data version are unchanged, the materialized paired points are reused instead of recomputed on every `_refresh_*`.
- **rBasex basis persistence** (§6 P1-3): `basis_dir=~/.cache/vmi_workflow/abel_basis` (in-process + cross-process disk cache, `VMI_workflow_reconstruction.py::_rbasex_basis_dir`); the first reconstruction is slow, afterwards millisecond-level (verified by `tests/bench_rbasex_basis.py`).
- **`quadrant_symmetry` single-tree hoist** (§6 P0-5 partial): the candidate search's cKDTree is built only once (the point set does not change); results are **bit-identical** to the pre-hoist code (`tests/bench_core.py` A/B comparison; the lock tests cover it).

### 16e. Repository and packaging
- **Standalone git repo**: `git init` at the project folder root (the parent directory's personal repo is unaffected; no remote, not pushed).
- **Test baseline**: `tests/test_core.py` (numerics goldens) + `tests/test_smoke.py` (offscreen end-to-end, including the §16a regression extensions and TOF-alignment/cache-invalidation checks); sample data regenerated deterministically by `tests/make_sample_data.py`; see `tests/README.md`.
- **Packaging files**: `README.md` (English, with screenshot), `LICENSE` (MIT), `requirements.txt` (numpy>=2.2 / scipy>=1.16 / matplotlib>=3.10 / PySide6>=6.7 / pyabel>=0.9, optional pyarrow>=15), `docs/screenshot_main.png` (real themed window captured via `canvas.grab()` after driving the full 7-step workflow offscreen), `.gitignore` (excluding `*.dat`/`*.npz`/`workflow_outputs/`/`Refence data/`/`tests/sample_data/` etc.).
- **Real-data final verification**: ~700 MB reference triplet async load → 1e+1i pairing → edge-fit centering → ring selection + binning → rBasex → session save/restore, all passing (numbers in the release notes, not committed).

### 16f. Drag-interaction 60 Hz blit path (2026-09-01)

> The user reported that dragging the electron circle filter ring and the ion filter box was "too laggy". The benchmark script `tests/bench_drag.py`
> confirmed: on the whole-axis redraw path, electron ring dragging ran at **91.8 ms/frame (11 fps)**, the ion filter box at **70.9 ms/frame (14 fps)**,
> and the polar ROI at **235 ms/frame (4 fps)** — every preview frame re-rasterized the 25k-point scatter.
> This section migrated all draggable overlays to the same
> **animated-artist blit protocol** as the ion-TOF fit preview (`restore_region` + `draw_artist` + `blit`).
> Interaction semantics, hit testing and the final released state are **completely unchanged**; the scientific-numerics path is untouched (goldens locked bit-exactly).

#### 16f.1 Mechanism (copies the TOF-fit preview's successful discipline, and fixes two of its latent defects)

1. **Session-scoped animated artists** (`_begin/_end_scatter_overlay_blit_sessions`,
   `_begin_ion_rotation_blit_session`): on drag press, the dragged axis's overlay family is
   `set_animated(True)` (electron: inner/outer rings + center marker, or the polar-ROI trio;
   ion: filter box + center point; rotation: direction line). Animated artists are skipped by all regular drawing,
   so **the background captured during a session cannot have stale overlay pixels baked in** — the root cause of the §14.4 ghosting.
   On release (`_on_canvas_release`) all sessions end first (un-animate, discard backgrounds, clear the fallback table);
   all release-commit paths keep using the regular refresh (`_draw_axes_immediate` / full refresh).
2. **Clean single background capture** (`_capture_scatter_overlay_blit_background`):
   each drag does exactly one "single-axis `ax.draw(renderer)` (animated artists skipped) + `copy_from_bbox(ax.bbox)`",
   turning the 70–250 ms re-rasterization from a per-frame cost into a one-time per-drag cost.
   Any mid-drag invalidation (`_invalidate_blit_background`) automatically recaptures on the next frame.
3. **Per-frame presentation** (`_present_scatter_overlay_blit` / `_present_ion_rotation_blit` /
   `_present_rbasex_range_blit`): `restore_region(bg)` → `draw_artist` each overlay in zorder
   → redraw the `ax.texts` above the overlay (`_axes_above_texts`, preserving the correct stacking of the `[raw]/[copy]` buttons and
   translucent annotations; these texts are hidden during capture to avoid double compositing) → `blit(ax.bbox)`.
   The rotation presentation additionally redraws the filter box/center point in order and `draw_artist(ax.title)`
   (the preview-angle suffix); the capture region = axes bbox ∪ title baseline/longest-suffix bounding box.
4. **Per-frame try/except fallback**: any blit exception → that interaction session permanently degrades to
   the `_draw_axes_immediate` whole-axis redraw (the §14.4 safe path, kept alive), logged once to stdout.
   Non-fast (typing/refresh) paths still use the stable path; `_use_safe_*_redraw()` now only gates
   the non-fast stable path and no longer blocks drag blitting (drag blitting enabled by default on all platforms).
5. **Faster typing-debounce path** (the fastest stable path since §14.4): the non-fast branches of `_update_circle_overlay_only` /
   `_update_ion_overlay_only` changed from "full canvas.draw" to a single-axis
   `_draw_axes_immediate(..., include_tight=True)` (which carries its own full-repaint fallback);
   the background cache is discarded outright and recaptured at the next drag session.
6. **Two pre-existing latent defects fixed**:
   - `_ensure_ion_tof_fit_preview_background` used to copy directly from the shared buffer,
     baking the previous frame's stationary anchor into the background (the ghost detector caught a 44 px double composite);
     it now recaptures with a single-axis redraw.
   - `_capture_theta_line_blit_background`'s former "fast path" copied from the shared buffer —
     the buffer always contained the guide line drawn by the previous present (a stale-angle residue when dragging θ),
     while the "slow path"'s hide+`canvas.draw()` recursed through the draw_event hook.
     Both image panels now capture with their own single-axis redraw (no recursion, guaranteed guide-line-free).

#### 16f.2 Ghost detector (`tests/bench_drag.py`, regression guard)

The benchmark reproduces the full test_smoke workflow (load → pair → fine ROI → ring selection → rBasex; the filter box is enabled before ring
selection and rBasex runs last, otherwise enabling the filter clears the reconstruction), then drives every draggable overlay through the real
`MouseEvent` handlers `_on_canvas_press/_move/_release`, timed frame by frame at the 16 ms throttle cadence via `_flush_*`. At the far endpoint of
each session it runs a **ghost check**: the canvas RGBA buffer (the drag result) is compared pixel-by-pixel against a forced clean `canvas.draw()`
reference render (rendered with the overlays temporarily un-animated so they composite in) —
the axes interior (shrunk 5 px) must be **bit-exactly equal** (EXACT/EXACT-INTERIOR); re-compositing differences in the margin text (the old path's
inherent "text darkening frame by frame" artifact) are logged but not counted. On the old code the detector indeed caught the θ guide-line stale-angle
residue (4×GHOST) and the TOF fit anchor double composite (2×GHOST); on the new code everything is EXACT-INTERIOR — the detector has real killing
power against both historical ghost classes.

#### 16f.3 Before/after comparison (200 frames × 2 sessions, offscreen, full workflow on sample data)

| Interaction | Before mean/p95/max ms (fps) | After mean/p95/max ms (fps) |
|---|---|---|
| electron ring/center drag | 91.8 / 107.9 / 127.3 (11 fps) | **4.4 / 4.6 / 84.6 (230 fps)** |
| ion filter box/center drag | 70.9 / 116.3 / 148.5 (14 fps) | **3.8 / 4.3 / 99.1 (262 fps)** |
| θ guide-line drag (centered) | 1.5 / 2.2 / 2.8 (685 fps)† | 5.0 / 6.7 / 9.1 (202 fps)† |
| θ guide-line drag (rbasex) | 1.7 / 2.3 / 2.7 (608 fps)† | 7.4 / 9.0 / 10.8 (135 fps)† |
| rBasex range-handle drag | 40.7 / 48.4 / 53.4 (25 fps) | **6.9 / 8.9 / 10.2 (145 fps)** |
| ion TOF fit box drag | 3.3 / 4.6 / 6.7 (301 fps) | 5.2 / 6.7 / 30.0 (191 fps) |
| ion TOF BG range hover | 0.7 / 1.0 / 1.3 (1444 fps) | 0.6 / 1.1 / 1.3 (1622 fps) |
| ion rotation preview drag | 91.6 / 111.0 / 147.7 (11 fps) | **5.2 / 5.9 / 85.6 (192 fps)** |
| polar ROI drag | 235.1 / 268.3 / 329.7 (4 fps) | **5.0 / 6.5 / 29.0 (202 fps)** |
| ion histogram fine-ROI commit (one release) | 943.2 | 516.0 |
| typing debounce 70 ms: circle-parameter redraw | 667.2 (median) | **80.6 (median)** |
| typing debounce 70 ms: filter-box redraw | 675.0 (median) | **74.6 (median)** |

† The θ guide line redraws the translucent annotations/buttons above it every frame to keep exact stacking; the per-frame cost went 1.5→5–7.5 ms,
still 135–200 fps. That is the price of fixing the pre-existing "background with baked-in guide line" ghost into bit-exact output.
All drag p95 ≤ 9 ms (≥110 fps), meeting the 60 Hz target; the "ion histogram peak marker" is a Ctrl+click action with no drag;
the fine-ROI span drag visuals are handled internally by matplotlib's SpanSelector (useblit); the table's entry is the app-side debounced commit cost.
Note: the offscreen measurements exclude Qt on-screen upload overhead (the `blit` partial upload is a small texture upload on real hardware, far
smaller than a full repaint).

### 16g. Center-estimator pruning (2026-09-01)

> Decision: the Electron Scatter `Center estimator` dropdown offered five
> modes; only two kept, user-approved estimators remain. All repo text in
> this section is English; the rest of the document was translated to
> English in §16h.

#### What was removed / kept

- **Kept modes** (`center_mode_combo`, in this order):
  1. `quadrant_symmetry` — `VMI_workflow_core.quadrant_symmetry_center`
     (**new default**; raw-point diagonal-quadrant symmetry matching,
     robust for ring distributions, no prerequisites);
  2. `polar_outermost` — `polar_outermost_center` (unchanged; still
     requires a valid Polar ROI band `[r min, r max]`).
- **Removed UI modes**: `edge_fit` ("Edge circle fit (recommended)"),
  `centroid` ("Centroid (mean)"), `geo_median` ("Geometric median").
- **Core function deletions (orphan audit before each deletion)**:
  - `geometric_median` — deleted (only caller was the GUI mode dispatch);
  - `circle_fit_kasa` — **kept**: still used by `edge_circle_center` and
    `_fit_circle_from_outer_edge_profile` (both on kept paths);
  - `edge_circle_center` — **kept as an internal helper** (UI mode removed):
    `quadrant_symmetry_center` uses it as its seed / no-scipy fallback, and
    `_scatter_peak_line_center` / `_iterative_outer_roi_edge_circle_center`
    (both behind `polar_outermost_center`) use it as seed and re-anchor.
- **GUI removals**: the dead `MainWindow._estimate_center` method (no
  callers left since the `_CenterWorker` refactor; it still referenced the
  removed estimators), the `centroid`/`geo_median` branches of
  `_CenterWorker._estimate_center_pure`, and the `centroid`/`geo_median`/
  `edge_fit` branches of the ring-empty fallback ladder in
  `estimate_center_once` (the default mode now takes the same
  "(fallback: full set)" branch for `>= 24` candidates).
- `polar_peak_center` and the `polar_peak`-related internals are untouched.

#### Straightness-gate semantics (first estimate)

The straightness acceptance gate (`_center_metrics_better`) still guards
re-estimates exactly as before. For the **first** estimate
(`self.circle_centroid is None`, i.e. no previously applied center) the
gate is bypassed: the pending center is the user's manual guess, and the
default `quadrant_symmetry` estimator optimizes quadrant symmetry rather
than the dominant-peak straightness metric — on two-ring data that metric
is noisy (the "dominant" polar peak flips between rings), and without the
bypass the first click of "Estimate Center Once" could keep the rough
manual center (`_CenterWorker` gained a `gate_on_metrics` flag).
`check_polar_outermost_center` keeps its pre-seeded center, so the polar
path is still gate-exercised and its behaviour is unchanged.

#### Legacy session restore mapping

Sessions saved before the pruning persist the mode STRING in
`combo_boxes.center_mode_combo.data`. `_restore_ui_state` now tolerantly
remaps every value that is not one of the kept modes (`edge_fit`,
`centroid`, `geo_median`, the historic `polar_peak`, or missing data) to
the new default `quadrant_symmetry`; `polar_outermost` restores as itself.
Restoring an old session therefore never crashes and never falls back to a
stale combo index. (Before this change the mapping was the inverse:
historic `quadrant_symmetry`/`polar_peak` strings were mapped to
`edge_fit`.)

#### Golden re-baseline (intentional shifts)

`estimate_center_once` writes the estimated center back into the
`circle_cx/cy_edit` widgets and `apply_circle_selection` reads them, so
every golden value downstream of the ring center shifted when the default
mode changed:

| Quantity | 2026-08-31 (edge_fit default) | 2026-09-01 (quadrant_symmetry default) |
|---|---|---|
| center estimate | (126.634982, 124.088852), 1.963 px error | (128.679982, 125.321352), 0.703 px error |
| ring inner / outer counts | 21388 / 386 | 21421 / 343 |
| denoised histogram sum | 21301.769706 (removed 86.230294) | 21344.284303 (removed 76.715697) |
| rBasex peaks | r=63.0 / r=111.0 | r=61.0 (beta=-0.893267, i=2601.926298) / r=112.0 (beta=0.199947, i=1161.280871) |

Values upstream of the center estimation (pair counts 25658, fine-ROI
selected mask 21871, session-roundtrip restored count 21871) are
bit-identical. `golden_core.json` lost its `centers` section (golden cases
of the pruned estimators); the lock tests for the two kept estimators are
unchanged and still pass bit-exactly.

### 16h. Open-source preparation (2026-09-01)

> Final compliance pass before publishing. No interaction logic changed;
> numerics untouched except for dead-code deletion verified caller-free.

- **Documentation**: `docs/user-guide.md` (step-by-step walkthrough with
  screenshots under `docs/img/`) and `docs/science.md` (algorithms and
  physics reference), both linked from `README.md`.
- **CI**: `.github/workflows/ci.yml` — an ubuntu-latest job (Python 3.12,
  Qt system libraries via apt, deterministic sample-data generation, then
  the core and offscreen smoke suites under pytest, 20 min timeout) and a
  lean windows-latest job running the core suite only (no display on
  Windows CI runners either; kept core-only to avoid flaky offscreen Qt).
- **Community files**: `CONTRIBUTING.md` (dev setup, test commands, style
  and goldens discipline), `CODE_OF_CONDUCT.md` (Contributor Covenant
  v2.1), `CITATION.cff` (repository URL is an explicit placeholder —
  update it after the GitHub repo is created), `.gitattributes`
  (`* text=auto`, `*.png binary`).
- **Full English translation**: this document was translated from Chinese
  to English (§0–§16g above); a repo-wide grep confirms zero CJK
  characters remain in any tracked text file.
- **Lint pass**: pyflakes-level only (`ruff check --select F,E9`) —
  removed two unused imports (`matplotlib.widgets.RectangleSelector`,
  `PySide6.QtCore.QRect`) and the dead `_adaptive_xy_bg_features` helper
  (last leftover of the KDE backend removed in §16b; zero callers).
  Four unused-local (F841) findings and one benign quoted-annotation
  (F821) finding in working code were investigated and intentionally left
  untouched. Both test suites re-run green afterwards.

### 16i. Polar/theta-guide blit freeze-and-crash fix (2026-09-01)

> User report: after switching the electron scatter to the polar view, dragging
> the radial-profile theta guide through a full 360-degree sweep froze the app
> and then hard-crashed it. Clicking the right half of the centered image
> (right-half = rBasex recon compare) without a reconstruction was suspected.
> That toggle was verified to be correctly refused without a reconstruction;
> the crash came from the drag-blit machinery itself.

Root causes (two compounding bugs left by the 16f refactor):

1. **Draw-event recapture storm (the freeze).** `_on_canvas_draw_event`
   re-captures the blit backgrounds (a full single-axes re-render) on every
   canvas draw. During a theta drag, any `draw_idle` fallback fired a draw,
   which re-entered the handler, re-captured, and re-blitted — escalating the
   60 Hz drag into a full re-render storm. The existing
   `_suspend_overlay_draw_event_sync` guard flag existed but the theta drag
   never set it.
2. **Stale / geometry-mismatched background (the segfault).**
   `_toggle_centered_right_half_compare` and `_on_electron_scatter_polar_toggled`
   replace panel image artists without invalidating `bg_theta_centered` /
   `bg_theta_rbasex`, and nothing validated that a captured background still
   matched the current canvas size. `restore_region` of a mismatched buffer is
   silently clipped by the offscreen Agg buffer, but on the real Qt backing
   store it writes out of bounds and hard-crashes the process (no Python
   traceback — why the offscreen harness could not reproduce it).

Fix (all interactive blit backgrounds, uniformly):

- **Geometry-safe blit**: every `copy_from_bbox` capture site records the
  canvas size (`_blit_bg_geom`, keys `theta_centered`, `theta_rbasex`,
  `scatter_e`, `scatter_i`, `ion_rotation`, `rbasex_range`,
  `ion_tof_xy_preview`); every fast path validates the geometry via
  `_blit_bg_size_matches` BEFORE `restore_region` and on mismatch drops the
  buffer and falls back to the whole-axes safe redraw. A stale region can
  never be restored again.
- **Suspend for the whole drag session**: the theta-drag press sets
  `_suspend_overlay_draw_event_sync = True` and captures the clean guide
  background once against the current content; the release commit resets the
  flag in a `finally` (and the drag-cancel helper resets it too), so no draw
  can trigger the recapture storm mid-drag and the flag cannot leak.
- **Self-healing drag frames**: `_preview_theta_line_only` now recaptures the
  background once and retries the blit when the geometry check rejects it
  (e.g. a canvas resize mid-drag), instead of degrading every remaining frame
  to full canvas redraws.
- **Invalidation on content change**: `_invalidate_theta_blit_backgrounds()`
  is called when the centered/recon images are replaced — compare toggle (both
  directions), `_apply_circle_projection_result`, `_clear_circle_result`,
  `_collect_recon_results`, and both session-restore paths. The electron polar
  toggle does not touch those images (the geometry guard covers its canvas
  resize side effect).

Regression tests (offscreen; they lock the guards, since the segfault itself
is screen-only): `check_theta_drag_blit_safety` (suspend window press->release,
handler short-circuit while suspended, mid-drag resize rejects the stale
background and the press recaptures, full 360-degree sweep) and
`check_compare_toggle_blit_invalidation` (refused without recon; helper
invalidates both buffers + geometry keys; both toggle directions leave no
stale-geometry background). All suites green; goldens byte-unchanged;
`bench_drag` ghost detector 17/17 EXACT-INTERIOR.

### 16j. UI polish round (2026-09-01)

> User-reported: (1) compare-mode left colorbar overlapped text, (2) settings
> tabs felt disorganized, (3) the rBasex radial-profile panel had no title,
> (4) full screen clipped the app with no scrollbars until exiting.

**16j-1 Compare colorbar + profile title.** In compare mode the projection
colorbar was squeezed into the narrow left inter-panel gap (8-17 px wide)
with its outward-facing label overlapping neighbouring text. Side-by-side
bars in the right gap cannot work (each bar's right-facing tick labels are
wider than any safe inter-bar gap), so compare mode now stacks BOTH colorbars
vertically in the right gap (projection upper, rBasex lower) with compact
labels ("Counts"/"Counts (log)", "rBasex"); single-colorbar mode keeps the
historical geometry and label. `_create_centered_bin_colorbar_axis` gained
index/count stacking; the dead "left" branch was removed. A new bbox
intersection check (`check_compare_colorbar_no_overlap`) asserts zero >1px^2
conflicts between every colorbar cax and all texts/axes/other cax, in both
compare ON and OFF states. The rBasex radial-profile panel now carries the
"rBasex Recovered Profile" title (placeholder path included); the startup
placeholder check no longer exempts it.

**16j-2 Settings tab reorganization.** The control builder still contained
the entire pre-tab legacy layout (`control_grid` + six outer group boxes,
~200 statements) that is never displayed — every widget it added was
re-parented by the later per-tab sections (verified: zero widgets existed
only in the legacy grids; the constructed widget tree is byte-identical
before/after deletion). The dead layout was removed. The live sections were
renamed from generic titles ("Controls"/"Parameters"/"Display"/"Actions"/
"Voltages") to panel-driven ones (e.g. "Center Estimation, Rings & Filters",
"Ion X/Y-TOF Map — Binning & ROI", "Centered Bin Map & Radial Profile",
"Spectrometer Voltages & Calibration" — the F/Lens/V_offset calibration rows
moved in with the detector voltages they belong to), and every tab now opens
with a "Drives: <panels>" hint line mapping the tab to its dashboard panels.
"Start Reconstruction" moved into the Reconstruction tab (rBasex — Run),
replacing the duplicate "Update Reconstruction" button (same slot, zero
external references). No signal connections changed; the widget-set dump is
identical apart from the six new hint labels.

**16j-3 Window-mode fit.** The main window's minimum width was pinned at
2404 px by the file bar (the trigger-mode combo's 726 px widest-item hint
plus text-width labels), so in full screen the central widget overflowed the
screen and the clipped right edge carried the scrollbars — the reported
"no scrollbars until exiting full screen". The combo now uses
AdjustToMinimumContentsLengthWithIcon and the three file-bar labels are
horizontally Ignored, bringing the window minimum to ~1534 px (the push
button text floor). `_configure_plot_canvas_size(fit_to_viewport=)` clamps
the canvas to the plot viewport, and a debounced (250 ms) `resizeEvent` hook
re-fits on WINDOW resizes only — settings-tray toggles do not resize the
window, preserving the §14.4 stable-canvas guarantee. New
`check_window_mode_fit` locks: shrinkable window, viewport-fit canvas at
resized sizes, scrollbars visible with real ranges when below the canvas
floors, and no canvas re-target on tray toggles. All suites green; goldens
byte-unchanged; bench_drag ghost detector PASS.

### 16k. Tab reserved for the tray; cross-machine theme pinning (2026-09-01)

> User request: pressing Tab should toggle the settings tray instead of
> moving keyboard focus, and the app must look the same on every
> Python/computer setup.

**Tab behavior.** The global event filter previously intercepted Tab only
while the plot region had focus; everywhere else Tab performed standard
focus navigation. It now intercepts plain Tab AND Shift+Tab from ANY widget
of the main window (key Tab or Backtab, no auto-repeat, modifiers only
none/Shift, no active popup, event target inside this window) and toggles
the settings tray. Modifier combos (Ctrl+Tab) and dialogs pass through.
Because hiding the tray can hide the focused control, the shortcut parks
focus on the plot canvas whenever the app focus widget ends up hidden — a
deterministic landing spot that keeps canvas keyboard/wheel handling alive;
Tab never lands on an arbitrary tab-stop. `check_tab_toggle_and_theme`
locks this (toggle from a tray line edit, round-trip with Shift+Tab, focus
invariants).

**Look pinning.** The previous theme was QSS-only, so native widget
geometries still varied with the OS/Qt build. `apply_application_theme()`
(factor-out of `main()`) now applies the **Fusion** style — identical
geometry on all platforms — plus a light `QPalette` (window/base/text/
highlight/tooltip/disabled roles) and an application `QFont` with a family
fallback chain (Segoe UI Variable → Segoe UI → Noto Sans → DejaVu Sans →
Arial) so text metrics do not depend on installed fonts. The QSS was kept
and extended (radio buttons, menus, dialogs, item views, hover states).
Note: `app.style().objectName()` reads empty once a stylesheet is active
(QStyleSheetStyle wrapper); assert via the style class instead.

### 16l. Pluggable pyAbel inversion methods (2026-09-01)

> User request: pyAbel offers more than rBasex — expose a method dropdown in
> the Reconstruction tab.

The Reconstruction tab gained an **Inversion method** combo backed by
`ABEL_METHODS` in `VMI_workflow_reconstruction.py` (9 entries verified
against pyabel 0.9.1's unified `abel.Transform` API: rBasex default, BASEX,
Daun, Direct, Hansen-Law, Lin-Basex, Onion-Bordas, Three-point, Two-point;
`onion_drying`/`pbasex` are not registered by this pyabel build and are
omitted). `normalize_abel_method` maps any unknown/legacy value to rBasex.

Dispatch: rBasex keeps its dedicated path unchanged (rIbeta -> peaks with
beta, bit-identical results); every other method runs
`run_abel_method_reconstruction` — `abel.Transform(direction="inverse")`
followed by angular integration of the inverted image over integer pixel
radii for I(r), beta reported as zeros with `beta_available=False`, and the
same peak extractor/peak settings as rBasex. `format_peak_text` renders
`beta=n/a` for such results instead of a misleading 0. Panel titles are
method-aware ("Hansen-Law Recon" / "<Method> Recovered Profile"), as are the
progress and status messages. Sessions persist `reconstruction.method` and
restore it tolerantly (legacy sessions without the key default to rBasex).
Order/Odd/Reg/rmax are documented as rBasex-only; Peak-* and Display
percentile apply to all methods. `check_alt_method_reconstruction` drives
Hansen-Law through the full async GUI path (image shape, peaks, beta n/a,
method-aware titles). All suites green; goldens byte-unchanged.

### 16n. Reconstruction busy-state popup crash + liveness (2026-09-01)

> User report: pressed Start Reconstruction with a non-rBasex method — the
> app appeared stuck — then clicked the inversion-method dropdown and the app
> hard-crashed.

Root causes (two, compounding):
1. **Apparent freeze**: basis-set-generating methods (BASEX/Daun/Lin-Basex
   first runs) compute for tens of seconds to minutes between the worker's
   5% and 90% progress checkpoints, so the progress bar sat at ~13% with no
   feedback — not a deadlock.
2. **Segfault**: `_progress_update`/`_progress_start` call
   `_pump_progress_events_if_safe()` -> `QApplication.processEvents()`. The
   reconstruction poll is a 120 ms QTimer; opening the QComboBox popup enters
   a native modal loop (mouse grab on Windows) in which the timer keeps
   firing, so progress updates re-entered `processEvents()` inside the active
   popup — event re-ordering/re-entrant painting crashes Qt natively. (The
   `_suspend_progress_event_pump` flag existed for save/export paths but did
   not cover this, and `_collect_recon_results` doing a full panel refresh +
   `thread.wait(1500)` inside the modal loop is the same danger class.)

Fixes:
- **Popup guard in the pump** (systemic): `_pump_progress_events_if_safe`
  skips `processEvents()` whenever `QApplication.activePopupWidget()` is set.
- **Deferred collection**: `_poll_recon_progress` does not call
  `_collect_recon_results` while a popup is open; the next 120 ms tick
  (popup closed) collects.
- **Combo locked while busy**: `recon_method_combo` is disabled during a run
  and re-enabled on every exit path of `_collect_recon_results`; the
  completion status now labels the method from the RESULT dict, not the
  live combo, so a mid-run change can never mislabel.
- **Liveness**: the running poll appends an elapsed-seconds counter
  (`_recon_started_at`) to the status message, and non-rBasex runs warn that
  first runs may generate basis sets.

`check_recon_busy_popup_safety` drives a slow (1.5 s) monkeypatched
Transform on the worker thread and locks in: combo disabled while busy, the
pump leaves a sentinel event pending while a popup is active, collection
defers while the popup is open, exactly one collection after it closes, and
the elapsed-time status. All suites green; goldens byte-unchanged.

Correction (see 16o): the popup re-entrancy above was provably NOT the whole
story — the user's second crash occurred with no popup open at all, and the
popup guard alone did not stop the crash. The pump is now also suppressed
for the whole duration of a reconstruction (16o).

### 16o. Reconstruction press-crash on Qt 6.7 (2026-09-01)

> User report: hard crash (no traceback, whole python.exe dies) on pressing
> Start Reconstruction — recorded twice, one with the inversion-method combo
> popup open, one with no popup at all.

Evidence (Windows Application event log, faulting process
`c:\Users\Tao\.conda\envs\my-env\python.exe`): faulting module
`Qt6Widgets.dll` version 6.7.3.0, exception code `0xc0000005`, fault offset
`0x000000000038e0f0` — IDENTICAL offset in both crashes (2026-09-01 7:40 PM
and 8:28 PM local; a same-day crash in `python312.dll` at a different
offset in the dev env is a separate, unrelated event).

Root cause (environment-parity class): the crash is specific to the USER
ENV (Python 3.11.11 + PySide6 6.7.2 / Qt 6.7.3). The dev env (Python 3.12.9
+ PySide6 6.11.1 / Qt 6.11.1) never reproduces; numerics (all 9 pyAbel
methods benchmarked as subprocesses in the user env) and full offscreen GUI
runs pass in BOTH envs. That is exactly why earlier offscreen tests passed
while the user crashed: a class of Qt 6.7 slot-re-entrancy/painting bugs
only manifests on a real window and in the older Qt build. The second crash
(no popup) shows the 16n popup re-entrancy explanation alone could not cover
it — `_progress_start`/`_progress_update` call
`_pump_progress_events_if_safe()` -> `QApplication.processEvents()`, and
re-entrant event dispatch from inside the button-click/poll slots is the
crash class; a popup only makes it (sometimes) more reachable.

Fix (busy-pump guard): `_pump_progress_events_if_safe` returns early while
`self._recon_busy` is True. The reconstruction pipeline is fully async
(worker thread + 120 ms poll timer), so the normal event loop repaints
everything without any pumping during a run.

Reproduction matrix (on-screen `_repro_real.py`, user env, PySide6 6.7.2,
synthetic sample data, tray = settings tray opened before the click; run
sequentially, ~30 s per cell):

| Cell | Unfixed | Fixed |
|---|---|---|
| tray x hansenlaw | pass | pass |
| tray x basex | pass | pass |
| fullscreen + tray x rbasex | pass | pass |

Causality NOT proven on the synthetic data: the unfixed build completed
every cell (exit 0, "DONE", no new 0xc0000005 application event), so on
this data the busy-pump guard remains hardening. The synthetic busy windows
were short (rBasex still busy at the 1 s tick, done by 2 s; the other cells
finished before the 1 s tick) while the real crashes involved longer
basis-generation busy periods, real data and DPI scaling. The fix stands
(the documented crash class is real and observed twice in the user env) but
the user must re-verify their real workflow on the fixed build.

Regression coverage: `check_recon_busy_popup_safety` now additionally
asserts the busy guard WITHOUT any popup — while `_recon_busy` is True a
sentinel `QTimer.singleShot(0, ...)` stays pending across one
`_pump_progress_events_if_safe()` call, and after the run finishes (busy
False, no popup) the pump dispatches a new sentinel again (guard scoped to
the run). All suites green in both envs; goldens byte-unchanged.

Dual-env testing requirement: UI crash classes are Qt-version-sensitive —
before releases run `tests/test_smoke.py` in the oldest supported env (the
user's Qt 6.7.3) AND in the dev env (Qt 6.11.1); see tests/README.md,
"Testing in multiple environments".

### 16p. Completion-refresh re-entrancy freeze (QPainter not active) (2026-09-01)

> User report: during reconstruction COMPLETION — at the
> "Refreshing reconstruction panels..." (90%) stage — the app froze and the
> console showed `QPainter::fillRect: Painter not active` /
> `QPainter::end: Painter not active, aborted`; the app hung (backing store
> aborted). Reproduced in the user env (`c:\Users\Tao\.conda\envs\my-env`,
> Python 3.11.11, PySide6 6.7.2 / Qt 6.7.3) only; the dev env (Python 3.12.9,
> PySide6 6.11.1 / Qt 6.11.1) tolerates the same call sequence.

Root mechanism (verified by code reading, and consistent with the observed
signature): `_collect_recon_results` set `self._recon_busy = False` at its
very TOP, which disengages the 16o busy-guard (`_pump_progress_events_if_safe`
skips `processEvents()` while `_recon_busy` is True) for the entire
completion path. Then:
1. `_progress_update(90, "Refreshing reconstruction panels...")` →
   `_pump_progress_events_if_safe()` → `QApplication.processEvents()` —
   RE-ENTRANT event dispatch in the middle of the collection slot.
2. `_refresh_reconstruction_panels_only()` re-plots the reconstruction image +
   radial profile (+ the centered compare left half when compare mode is on)
   and calls `_force_full_canvas_redraw_for_layout_change()` — a full-canvas
   raster of the large figure, so paint events delivered re-entrantly land in
   a half-finished render.
3. `_progress_update(100, ...)` pumps again; `_reset_toolbar_navigation_history()`
   follows.
On Qt 6.7 (real window) the re-entrant paint into the incomplete render is the
"Painter not active" + backing-store abort; Qt 6.11 tolerates it (why the
offscreen suites were green in both).

Fix (unconditional pump suspension for the whole slot): the collection body
(including both `_progress_update` pumps, the panel refresh, and the early
return paths) is now wrapped in `try/finally` that first saves the current
value, sets a NEW flag `_suspend_progress_event_pump_unconditional = True`,
and restores it in the `finally`. `_pump_progress_events_if_safe` honors this
flag unconditionally (all platforms), BEFORE the pre-existing Windows-only
`_suspend_progress_event_pump` (its only setter is `save_session_output`,
which deliberately scopes it to Windows; the new flag was kept separate so the
save/export semantics stay untouched). `_recon_busy = False` was deliberately
NOT moved to the end — the combo unlock, thread teardown and status paths in
the slot rely on it being False at their point of use (grep-verified: the only
`_recon_busy` consumers are the busy gate in `run_reconstruction_now`, the 16o
pump guard and the slot itself). No behavior change otherwise: same draws,
same final visuals, same status text (the normal event loop repaints after the
slot returns; pumping is never required for correctness of the visuals).

Completed-path audit (no other re-entrancy): `_force_full_canvas_redraw_for_layout_change`,
`_refresh_reconstruction_panels_only`, `_draw_canvas_without_overlay_draw_event_sync`,
`_present_canvas_now`, `_draw_axes_immediate`, `_reset_toolbar_navigation_history`
and `_add_subplot_save_markers` contain NO `processEvents`/`flush_events`/`repaint`
calls — the only `processEvents` sites in the app are the progress pump, the
clipboard retry helper (`_try_set_image_on_clipboard_qt`, unrelated) and the
save/export pump (already guarded). `_present_canvas_now` only schedules
(`canvas.update()`/`draw_idle`), never dispatches. So the pump route is fully
covered by the new slot-wide flag; no helper needed routing.

Phase 1 reproduction (`_repro_refresh.py`, scratch — deleted after the run):
real event loop, tray OPEN, hansenlaw with a 4 s-sleeping `abel.Transform`
monkeypatch so completion lands with realistic timing; two Start clicks (the
second with the centered right-half compare enabled); plain-thread heartbeat
every 300 ms; auto-quit at 34 s; stderr captured (Qt writes "Painter not
active" at C level):

| Cell | Platform | Fixed | Observer result |
|---|---|---|---|
| user env (Qt 6.7) | offscreen | unfixed | pass: both completions, no stall, stderr clean |
| dev env (Qt 6.11) | offscreen | unfixed | pass: both completions, no stall, stderr clean |
| user env (Qt 6.7) | on-screen | unfixed | **fail**: heartbeat/logs stop after the Start click; the 34 s auto-quit never fires (event loop frozen — the reported freeze); process AV-crashes ~89 s in — Windows event log: `Qt6Widgets.dll` 6.7.3.0, `0xc0000005`, offset `0x38e114` (the 16o documented offset family 0x38e0f0; same DLL, same class) |
| user env (Qt 6.7) | on-screen | fixed | not clean: process aborted with exit code 127 shortly after the click WITHOUT a WER "Application Error" record (no access violation logged) — distinct from the unfixed AV but also not a clean pass |

Honest caveats: the exact "QPainter: Painter not active" warning text did NOT
reproduce on the synthetic data (it appeared in neither offscreen nor
on-screen cells); the on-screen failure appeared as the freeze + delayed AV on
the unfixed build, i.e. the same class but a harder manifestation. The
on-screen fixed cell could not be re-run (cell budget), so the fix was not
validated on-screen in this session: the actual crash/freeze might also
depend on event timing (the unfixed freeze began before the first completion's
refresh could print, i.e. inside or right before the 90% pump window — timing
consistent with the reported symptom but not proven on the synthetic path).

Regression coverage: `check_recon_collect_no_pump` drives a slow-Transform
reconstruction, stops the poll timer so the completion is NOT collected
automatically, then calls `win._collect_recon_results(worker)` DIRECTLY (popup
closed, worker finished — the exact fix window) while `QApplication.processEvents`
is replaced with a counting wrapper (the pump calls the class-level static
method, so the instance-level attribute alone could not intercept it): ZERO
`processEvents` calls may occur between slot entry and the finished status,
and `_suspend_progress_event_pump_unconditional` must be restored False
afterwards (plus the collection results in `rbasex_recon_result` set and the
method combo unlocked). On the unfixed code this fails with `collection pumped
the event loop 2x`. All suites green in both envs; goldens byte-unchanged.

### 16q. Synchronous blit repaint vs canvas resize (final crash layer) (2026-09-01)

> User report (after the 16p fix): during reconstruction the app still
> froze/crashed at the "Refreshing reconstruction panels..." (90%) stage, and
> stderr now showed `QPainter::begin: Paint device returned engine == 0,
> type: 1` / `QPainter::fillRect: Painter not active` /
> `QPainter::end: Painter not active, aborted`.

Root cause (verified, not re-derived): matplotlib's Qt backend implements
`FigureCanvasQT.blit()` as a SYNCHRONOUS immediate Qt paint. `inspect.getsource`
of `matplotlib.backends.backend_qt.FigureCanvasQT.blit` (identical in both
envs: matplotlib 3.10.8 in the user env, 3.10.7 in the dev env) ends with

```python
l, b, w, h = (int(pt / self.device_pixel_ratio) for pt in bbox.bounds)
t = b + h
self.repaint(l, self.rect().height() - t, w, h)
```

`QWidget::repaint()` paints OUTSIDE the normal paint cycle, and on Qt 6.7.3 it
aborts app-wide ("Paint device returned engine == 0" → "Painter not active" →
backing-store aborted → freeze → `Qt6Widgets.dll` AV, the 16o offset family)
when the widget's native paint surface is not available at that instant. The
app's 250 ms `_canvas_refit_timer` (16j-3) resizes `self.canvas` on window
resizes; a `repaint()` arriving right around a `canvas.resize()` /
native-surface re-creation hits exactly that window. The completion path fired
several such synchronous repaints from a timer slot: the panel refresh's
`_force_full_canvas_redraw_for_layout_change` (background recapture +
`_blit_theta_guides`, plus `_update_circle_overlay_only` /
`_update_ion_overlay_only` → `_blit_overlays`) and the
`_draw_axes_immediate`-based fast paths all end in `canvas.blit(...)`.

Fix part 1 — completion path with ZERO immediate paints. `_collect_recon_results`
is now LIGHT: stop the poll timer, clear `_recon_busy`, unlock the method
combo, tear the thread down, store `self.rbasex_recon_result = result`, set the
90% "Refreshing reconstruction panels..." status, then schedule the heavy part
with `QTimer.singleShot(0, self._finalize_recon_collection)`. The early-return
branches (`worker.error` / `worker.result is None`) set a status and skip the
scheduling. The 16p unconditional pump suspension REMAINS around the light slot
(the 90% pump still sits inside it). The new `_finalize_recon_collection` runs
from a clean event-loop turn: invalidate the theta-guide caches
(`_invalidate_theta_blit_backgrounds`) and the scatter/rotation/range caches
(`_invalidate_blit_background`) — they lazily recapture at the next drag press
(16f/16i), so NO immediate capture happens here — then
`_refresh_reconstruction_panels_only()` (panel re-plot; its
`_force_full_canvas_redraw_for_layout_change` tail is now repaint-free, see
below), `_reset_toolbar_navigation_history()`, and the final 100% status. The
async-completion contract is explicit via a new `_recon_finalize_pending` flag
(set when the finalize is scheduled, cleared in its `finally`): the run is
fully finished only when BOTH stages are done. A new helper
`_full_redraw_scheduled()` implements the repaint-free redraw tail (pure Agg
`_draw_canvas_without_overlay_draw_event_sync()` — the draw-event hook is
suspended, so it cannot re-enter blits — plus `_axis_last_blit_extents`/
`_invalidate_blit_background()` cache invalidation plus a SCHEDULED present
via `_present_canvas_now` = `canvas.update()`).
`_force_full_canvas_redraw_for_layout_change` keeps its signature but now
delegates to it; the audit of its other callers (`_refresh_after_circle_clear`,
`_on_electron_scatter_polar_toggled` (already deferred via singleShot),
`_refresh_rbasex_profile_display_settings`,
`_on_radial_profile_mode_changed`, both ion-histogram toggles,
`_refresh_after_ion_position_transform_change`) showed every one tolerates
losing the immediate blits (identical visuals one frame later), so no caller
kept the immediate form. The overlay ARTISTS are still recreated by those
callers' flags: `_update_circle_overlay_only` / `_update_ion_overlay_only`
gained a keyword-only `present=True` argument, and `present=False` stops after
the artist updates (parameters, cleared panels) so the scheduled full redraw
renders them — no caller behavior change otherwise.

Fix part 2 — canvas-geometry ↔ blit mutual exclusion (kills the crash class
app-wide, for every drag/interaction path too). After
`self.canvas.resize(...)` in `_configure_plot_canvas_size`, the app sets
`_canvas_geometry_changing = True` and clears it via
`QTimer.singleShot(80, ...)` (after matplotlib's own 30 ms resize debounce and
the native-surface re-creation settle). EVERY immediate-paint fast path checks
the flag FIRST and takes its non-blit fallback instead — `_blit_theta_guides`,
`_blit_overlays`, `_present_scatter_overlay_blit`, `_present_ion_rotation_blit`,
`_present_rbasex_range_blit`, `_present_ion_tof_fit_preview_from_background`,
and `_draw_axes_immediate`'s per-axes blit branch (therefore also
`_draw_axes_preview`, which delegates). End-to-end fallback-chain audit: with
the flag set, `_blit_theta_guides` returns False (its callers either ignore
the return, or — the theta drag `_preview_theta_line_only` — recapture once,
retry, and end in `canvas.draw_idle()`); `_blit_overlays` drops the caches and
uses `canvas.draw_idle()`; every `_present_*` returns False so the drag frame
falls back to `_draw_axes_immediate([ax])`, whose 16q branch (and every other
early-exit/exception branch inside it) terminates in a full `canvas.draw()` +
`_present_canvas_now` scheduled present — never another blit. There are
exactly seven `canvas.blit(` sites in the app (seven functions, counting the
two blits in `_blit_overlays`); each is now behind the flag check, so no path
can reach `canvas.blit()` while `_canvas_geometry_changing` is set. The
capture helpers are intentionally NOT guarded: `copy_from_bbox` snapshots the
Agg buffer (independent of the widget surface) and matplotlib resizes that
buffer within its own resize handling inside the 80 ms window.

Reproduction matrix (`_repro_final.py`, scratch driver, deleted after the run;
user env PySide6 6.7.2 / Qt 6.7.3, ON-SCREEN, real event loop, tray OPEN,
Hansen-Law with a 4 s-sleeping `abel.Transform`, TWO recon cycles per cell,
each cycle with a `canvas.resize()` fired mid-busy plus the debounced
`_configure_plot_canvas_size(fit_to_viewport=True)` refit right after;
plain-thread heartbeat + Qt tick observers; auto-quit at 30 s; sequential
cells, ~30 s each):

| Cell | Result |
|---|---|
| user env (Qt 6.7), on-screen, UNFIXED (fix stashed via `git stash push`) | **fail**: cycle-1 collect began (worker finished, `error=''`, 90% stage) and `_collect_recon_results` NEVER returned; Qt ticks stopped (event loop frozen) while the heartbeat thread continued ~2 s; then the whole process died with a native access violation (shell-reported segfault, exit 139). Reproduced TWICE (also with a plain rbasex completion, before the slow-Transform patch). No `engine == 0` / "Painter not active" stderr lines (Qt dies before flushing them — the harder manifestation, as in 16p) and no WER Id=1000 record for this process |
| user env (Qt 6.7), on-screen, FIXED (`git stash pop`, guard verified present) | **pass**: both cycles complete — light collect returns at the 90% status, the deferred finalize renders in ~0.7 s, both mid-busy resizes + refits handled, loop alive throughout (58 ticks), clean auto-quit, exit 0, stderr clean |

(The two `python312.dll` Id=1000 records from `C:\App\MiniConda\python.exe`
shortly before the cells are the dev-interpreter crash class already noted as
a separate, unrelated event in 16o — different process, different module; no
new Qt6Widgets.dll record appeared during any cell.)

Regression coverage (offscreen; locks the invariants, since the crash itself
is screen-only): `check_recon_finalize_no_repaint` drives a slow-Transform
reconstruction, patches `win.canvas.repaint` with a counting instance
wrapper BEFORE the completion (and forces `_canvas_geometry_changing = False`,
so the completion must be blit-free by construction rather than suppressed by
the geometry guard), then drives `_collect_recon_results` +
`processEvents` until the deferred `_finalize_recon_collection` ran and
asserts ZERO `canvas.repaint` calls across the WHOLE completion, the result
stored, the combo unlocked, `_recon_busy` False, the pump-suspension flag
restored and the 100% "Reconstruction finished" status.
`check_blit_geometry_change_guard` sets `win._canvas_geometry_changing = True`
with valid caches and asserts `_blit_theta_guides("centered")` returns False,
`_blit_overlays()` takes its fallback, and `_draw_axes_immediate` ends in a
scheduled present (`canvas.update()` counted) with zero repaint calls; clearing
the flag restores the blit paths (repaint called, `_blit_theta_guides` True).
All suites green in BOTH envs; goldens byte-unchanged.

Note: the 16p unconditional pump suspension for the collection/finalize slots
is deliberately kept (the 90%/100% `_progress_update` pumps still sit inside
them); 16q removes the synchronous-immediate-paint class itself, which 16p
could not cover.

### 16r. Method-specific parameter sections (2026-09-01)

> User request: reorganize the Reconstruction tab so the parameter sections
> SWAP with the selected pyAbel inversion method, and the "Recovered radial
> profile" X/Y-axis display settings become a section separate from the
> model parameters.

The tab went from 2 groups ("Abel Reconstruction — Run" + the former
"rBasex Model Parameters", which packed EVERYTHING) to 4 groups, all built
in `MainWindow.__init__` and wrapped by the same
`_build_control_tab_scroll_area(...)` call (drives text updated to
"rBasex/selected-method Reconstruction image (top row, 4th) + Recovered
Profile (bottom row, 4th)"):

1. **Abel Reconstruction — Run** — unchanged (method combo + Start button +
   hint). `recon_method_combo.currentIndexChanged` is now additionally wired
   to `_on_recon_method_combo_changed`, which shows the matching page of the
   new `self.recon_method_stack` (`QStackedWidget`).
2. **Peak Finding & Display — all methods** (`self.recon_peak_group`): the
   five peak-finding edits (`rbasex_peak_smooth_sigma_edit`,
   `rbasex_peak_height_edit`, `rbasex_peak_prominence_edit`,
   `rbasex_max_peaks_edit`, `rbasex_peak_min_dist_frac_edit`) plus
   `rbasex_display_percentile_edit`, moved with identical widget names,
   defaults and signals — they apply to every method.
3. **Method Parameters** (`self.method_params_group`): one stacked page per
   registered method, built in `ABEL_METHODS` order (combo index == stack
   index; pages also keyed `self._recon_method_pages[method_key]` with
   objectName `recon_method_page_<key>`). The rbasex page keeps the existing
   four controls (Order / Odd terms / Reg / rmax — same names and signals);
   new pages: basex = Sigma ("1.0") + Reg ("0.0") + Correction checkbox
   (checked, `basex_correction_checkbox`); daun = Reg ("0.0") + Degree
   combo 0/1/2 (`daun_degree_combo`); linbasex = Smoothing ("0") + Rcond
   ("0.0005") + Threshold ("0.2") + Legendre orders ("[0, 2]"); the five
   parameter-less methods (direct, hansenlaw, onion_bordas, three_point,
   two_point) get a "This method has no additional parameters." hint label.
   The stacked widget naturally preserves each page's edit values across
   method switches. `_restore_ui_state` re-syncs the stack after its
   signal-blocked combo restore.
4. **Recovered Radial Profile — Axes & Display**
   (`self.recon_profile_axes_group`): ALL `rbasex_profile_*` display
   controls moved out of the old model-parameters box (Profile theta /
   dTheta / Update button, Profile r tags + Clear r Tags, X axis mode +
   Swap top/bottom, Energy c/b/hv + Apply X Axis, Normalize max, x min/max
   + Apply X Range, Top space x + Apply Top Space, Pick Range on Plot +
   range label + Clear Range) — relaid out row-wise, names and signal
   connections unchanged; they drive the bottom-right Recovered Profile
   panel only.

Backend plumbing (non-rBasex paths only; the rBasex numeric path is
deliberately untouched so the goldens stay byte-identical):

- `_get_method_params(method=None)` reads the CURRENT method page into a
  per-method dict — basex `{"sigma", "reg", "correction"}`, daun
  `{"reg", "degree"}`, linbasex `{"smoothing", "rcond", "threshold",
  "legendre_orders"}` (others `{}`) — with tolerant parsing (invalid text
  falls back to the pyabel defaults via `_parse_float_edit` /
  `_parse_legendre_orders_edit`, never raises). `_get_rbasex_settings` is
  unchanged.
- `run_reconstruction_now` snapshots `method_params` for the selected
  method and passes it to `_ReconWorker(..., method_params=...)`, which
  forwards it to `run_reconstructions_from_centered_data(...,
  method_params=...)` and on to `run_abel_method_reconstruction`.
- `VMI_workflow_reconstruction.sanitize_abel_method_params(method, params)`
  filters the dict through a per-method whitelist
  (`_METHOD_PARAM_KEYS`: basex sigma/reg/correction; daun reg/degree;
  linbasex smoothing/rcond/threshold/legendre_orders), coerces/validates
  every value (float/int casts, finiteness, ranges; daun degree 0-3,
  legendre orders ints 0-6, max 4 entries) and DROPS unknown or invalid
  keys so the pyabel default applies; it never raises. The sanitized dict
  is forwarded as `abel.Transform(..., transform_options=...)` (pyabel's
  quadrant dispatch resolves `basex.basex_transform` / `daun.daun_transform`
  etc. at call time, and linbasex runs `linbasex_transform_full` with the
  same option names — the whitelist matches those signatures exactly).
- Sessions: `save_session_output` persists
  `reconstruction.method_params` (jsonified params of the saved method);
  `_load_session_output_from_metadata_path` calls the new
  `_restore_method_params(recon_meta)`, which merges the saved values with
  `RECON_METHOD_PARAM_DEFAULTS` for missing keys and writes them back into
  the page widgets (signal-blocked). Legacy sessions without the key are a
  no-op (the generic line-edit/check-box UI-state restore already covers
  those widgets).

Regression coverage: `check_method_param_sections` in `tests/test_smoke.py`
(registered in `run_regression_checks`) locks (a) stacked-page swap
(rbasex vs basex visible page + Order edit on the rbasex page + Sigma edit
on the basex page) and per-page value persistence across switches; (b) the
three sections being separate `QGroupBox`es in the Reconstruction tab with
the right widgets contained in exactly one section each; (c) BASEX
sigma=1.7/reg=120 plumbing — a delegating capture wrapper around
`abel.basex.basex_transform` must receive sigma=1.7/reg=120/correction=True
and the run must produce a valid result (peaks, image, beta n/a); (d) Daun
degree=1 plumbing the same way; (e) session save/restore roundtrip of
`method_params` into a fresh window's widgets plus the tolerant merge
(partial keys fall back to defaults; legacy-shaped metadata is a no-op).
All suites green in BOTH envs; goldens byte-unchanged.

### 16s. Physical-unit axis labels + screenshot refresh (2026-09-01)

> User request: the radial-profile panels must name their axis units, and
> every published screenshot predates the "rBasex Recovered Profile" title
> (16j-1), the Fusion theme (16k) and the settings-tab reorganization
> (16j-2/16r) and had to be regenerated from the current build.

**Axis labels.** The radial-profile x axes now name their physical unit.
The theta-slice Radial Profile of the centered map
(`_plot_theta_radial_profile_panel`) labels x as `r (px)` (was bare `r`);
the profile x values are the binned-pixel radius of the centered map.
The rBasex Recovered Profile label factory `_rbasex_profile_x_axis_label`
returned the misleading `Radial Position (mm)` for the raw-radius mode
(`mode="r"`), although `_rbasex_profile_radius_to_display_x` passes pixel
radii through unchanged in that mode — it now returns `r (px)`. The
`ke`/`be` modes already carried units (`Kinetic Energy (eV)` /
`Binding Energy (eV)`) and are unchanged; `both` delegates to the primary
energy mode. For consistency the raw-mode naming was fixed wherever it
surfaced: the X-axis mode combo item (`Radial Position (mm)` ->
`Radial Position (px)`), the hard-coded mode status string, and the
`rbasex_profile_last_axis_label` reset defaults (used by the range-integration
status text/overlay, previously `r`/`Radial Position (mm)`). The reconstruction
image panel (`_plot_reconstruction_panel`), the Centered Bin Map
(`_plot_centered_bin_image`) and the exported reconstruction PNG
(`_save_reconstruction_image`) label their axes `x centered (px)` /
`y centered (px)` (was `x centered` / `y centered`); their extents are the
binned-pixel coordinates of the centered map. No numerics touched: labels
and status strings only, goldens byte-unchanged.

**Screenshot refresh.** All seven published screenshots were regenerated
offscreen from the current build with a scratch driver mirroring
tests/test_smoke.py techniques (async workers pumped via processEvents,
monkeypatched QMessageBox, cwd in a TEMP dir, `apply_application_theme`
for the Fusion look, `QT_QPA_FONTDIR=C:\Windows\Fonts` so widget text
renders offscreen). `docs/screenshot_main.png` is the post-reconstruction
dashboard `canvas.grab()` at the canonical 2214x705; the five step grabs in
docs/img are full-window grabs at the dashboard window size that exactly
fits the canonical canvas (2246x854; the old 2878x916 shots predate the
16j-3 fixed-size canvas and relied on the then-stretched canvas);
`step3b_settings_tray.png` is a 1720x1060 window grab with the Settings
tray open on the Electron Scatter tab. The driver asserts per shot:
dimensions, file size and pixel variance (PIL), plus programmatically the
`rBasex Recovered Profile` title, the new `r (px)` / `x centered (px)`
labels, the golden center estimate and rBasex peaks. Note (observed, not
changed): the 16j-3 resize refit derives the preferred canvas size from
`figure.get_figwidth()`, which matplotlib keeps synced to the canvas, so
successive window resizes ratchet the canvas down to the 1480x700 floor;
the screenshot driver restores the design figure size
(27.0x8.6 in) before each grab to pin the canonical 2214x705 dashboard.
