# Science Reference

This document describes what each pipeline stage of the VMI workflow computes, the
numerical method used, its assumptions, known limitations/biases, and practical
guidance. It complements `ARCHITECTURE.md` (software structure) and the docstrings
in `VMI_workflow_core.py`, `VMI_workflow_reconstruction.py` and `VMI_workflow.py`.

All stages operate on coincidence-list data: each trigger row carries an electron
index and an ion index into per-particle tables `(x, y, t)`.

---

## Pipeline overview

| Step | Panel / action | Core computation |
|---|---|---|
| 1 | Load | CSV parsing into trigger + electron + ion tables |
| 2 | Process and Plot | Trigger (TDC) pairing into electron–ion events |
| 3 | Ion Histogram | TOF/mq histogram, peak selection, background trend fit |
| 4 | Ion Coincidence | TOF background model (keep mask) + X/Y–TOF alignment |
| 5 | Electron Scatter | Ring-center estimation |
| 6 | Apply Ring Selection | Centered binning + uniform-background denoising |
| 7 | Reconstruction | rBasex inverse Abel transform, peak (r, β) extraction |

---

## Step 1–2: Trigger pairing (coincidence mode selection)

**What it computes.** A set of index pairs $(e_k, i_k)$ selecting which electron and
ion table rows belong to the same ionization event, under the chosen coincidence
mode: all valid rows (1e/1i assumption), strict 1e+1i, 1e+2i, or 1e+3i.

**Method.** The strict modes vectorize an adjacent-row difference test on the raw
trigger columns: a row $k$ is accepted when $(\Delta e, \Delta i)_k =
(e_k - e_{k-1},\, i_k - i_{k-1})$ equals $(1,1)$, $(1,2)$ or $(1,3)$ exactly
(integers compared after rounding; both rows must be non-NaN). Accepted 1e/2i and
1e/3i rows are expanded into two or three pairs sharing the same electron
$(e, i-1), (e, i)$ / $(e, i-2), (e, i-1), (e, i)$. Complexity is $O(N)$ with no
Python loop.

**Assumptions.**
- Trigger rows are stored in acquisition order; array adjacency equals time
  adjacency. No sort or monotonicity check is performed.
- The TDC increments indices exactly as the mode assumes.

**Known limitations.**
- *Concatenated acquisitions:* if several acquisitions are merged into one
  trigger file (the sample data include "merged" trigger files), pairs that
  would straddle a file boundary are lost — the pair anchored on the first row
  of the next segment has no valid predecessor. The loss is $O(1)$ per boundary
  but systematic; accidental deltas of exactly $(1,1)$ across a boundary can in
  principle form a false pair.
- *NaN rows reset pairing:* two events separated by a NaN trigger row are never
  paired (strictness trades recall for zero false pairs).
- The "all valid rows" mode does no coincidence filtering at all; use it only
  for data where every row is a true 1e/1i event.

**Guidance.** Prefer the strict mode matching your experiment's multiplicity. For
merged acquisitions, keep the per-file merge boundaries in mind when interpreting
absolute counts (they do not bias distributions, only total yield, by ~1 event per
boundary).

---

## Step 3: Ion histogram, m/q calibration and background trend

### 3.1 Histogram and selection

The ion histogram is binned in flight time $t$ (or $m/q$) over a coarse ROI, then
a fine ROI plus peak markers define the accepted ion mask. The mask selects which
paired events feed all downstream panels.

### 3.2 m/q calibration

**What it computes.** A mapping between TOF and mass-to-charge ratio.

**Method.** A square law through the reference point:

$$
\frac{m}{q} = a\,t^2 + b, \qquad b = 0, \qquad
a = \frac{(m/q)_{\mathrm{ref}}}{t_{\mathrm{ref}}^2}.
$$

With a reference TOF *range* selected, the range edges are treated as the edges of
one integer m/q bin, $t_{\mathrm{lo}} \mapsto (m/q)_{\mathrm{ref}} - \tfrac12$ and
$t_{\mathrm{hi}} \mapsto (m/q)_{\mathrm{ref}} + \tfrac12$, and $a$ is refitted by
least squares through the two points $(t^2, m/q)$.

**Assumptions.**
- Negligible time-zero offset: the physical law is
  $t = t_0 + k\sqrt{m/q}$ with $t_0 \approx 0$, so $m/q = a t^2$ holds.
- In range mode, the user-selected range spans exactly one full integer m/q bin.

**Known limitations.**
- `b = 0` forces $t_0 = 0$. With a real offset $t_0$ (extraction pulse delay,
  cable delays) the true law is $m/q = a(t - t_0)^2$, and the fitted square law is
  systematically stretched away from the reference: the relative m/q error grows
  roughly like $2 t_0 / t$ near $t_{\mathrm{ref}}$ and worsens for small $t$.
  *Calibrate with a reference near the low-m/q end of interest to limit this.*
- The single-parameter range fit has no residual feedback: a wrong range silently
  skews the whole axis.
- The scale inherits the error of the reference pair multiplicatively
  ($\delta(m/q)/(m/q) \approx 2\,\delta t / t$).

### 3.3 Ion histogram background trend fit

**What it computes.** A smooth background curve under the ion TOF/mq spectrum
(display/diagnostic overlay; it does not modify counts).

**Method.** Two layers:
1. A monotone *lower envelope* baseline: a low rolling quantile (q = 0.22–0.30)
   of $\log(\mathrm{counts})$, lightly smoothed, followed by a weighted isotonic
   regression applied to the reversed array (so the baseline decays monotonically
   along the axis), clipped to the data.
2. *Adaptive power-law components* fitted to that envelope: each branch is
   $A \exp\!\left[-(p_0 z + \tfrac12 s z^2)\right]$ with
   $z = \log\!\big((u + s_0)/(u_0 + s_0)\big)$ and $u = t^2$ (or $u = m/q$),
   i.e. a power law whose local exponent drifts from $p_0$ to $p_0 + s$.
   Amplitudes come from a weighted log-space fit; candidate branches are ranked
   by log-space SSE against the envelope plus tail/preference penalties; one
   optional second branch on the residual is accepted if it improves the SSE by
   ≥ 5% with amplitude ≥ 8% of the first. A non-negative least-squares (NNLS)
   path with BIC selection exists for fixed-exponent laws, and the final curve is
   projected to *never exceed* the envelope (and the raw counts), then made
   monotone non-increasing.

**Why the fit sits under the signal peaks.** The fit target is a low rolling
quantile — a valley-floor envelope. Inside peaks the envelope is pulled down, and
the "never exceed target" projection plus under-target SSE scoring keep the model
hugging that envelope. This is deliberate (the curve never cuts into peaks and
peak contrast is preserved), but it means:

**Known limitations.**
- The displayed background is systematically *below* the true baseline, most
  visibly in the valleys between peaks and under wide peaks; using it for
  quantitative subtraction would overestimate peak yields.
- Monotonicity forbids a background that rises toward long times/high m/q.
- The rolling-quantile window saturates at 31 bins, so structure much wider than
  the window cannot be followed on long histograms.

**Guidance.** Treat the curve as a trend/shape diagnostic. For quantitative
background removal of the ion image, use the Step 4 point-level background model
or an off-line side-by-side reference measurement.

---

## Step 4: Ion coincidence map — background model and alignment

### 4.1 TOF background model (point keep mask)

**What it computes.** A per-point score $s_j = \rho_{\mathrm{bg}}(x_j, y_j) /
\rho_{\mathrm{all}}(x_j, y_j)$ for each transformed ion point, and a keep mask
`s_j < threshold` used to suppress background events in all panels when enabled.

**Method.** Points inside the user-drawn signal boxes are the "mixed" training
set; everything outside is the "background" training set. Smoothed 2D histograms
give $\rho_{\mathrm{all}}$ and a local $\rho_{\mathrm{bg}}$, which is blended
(geometrically, weighted by per-cell support) with a radially symmetric floor
profile fitted from the outside-box points (low quantile of angular sector
densities per radius, blended with a log-log power law, made monotone). Scores
are rescaled so the median outside-box score is 1; the threshold is chosen by
scanning candidate quantiles of the combined score distribution with the
objective $0.82\,P(\text{bg removed}) + 0.18\,P(\text{mixed kept})$, penalized if
more than 28% of in-box points would be removed or fewer than 45% kept.

**Assumptions.**
- The drawn boxes cover the signal; whatever is outside is background.
- Background density is a smooth function of $(x, y)$ only (after the applied
  rotation/alignment), not of TOF.

**Known limitations.**
- This is a density-ratio classifier, not a physics model: a box that under-covers
  a broad signal leaks signal into the background training set and depresses the
  threshold; over-tight boxes around strong signal make the model remove
  legitimate weak signal elsewhere.
- Masking semantics: while the model is enabled, events with non-finite
  $(x, y, t)$ are *always* removed (the keep mask is False for them), and the
  final removal acts on all paired rows, including those outside the current view.
- The operating point allows up to ~28% in-box removal by construction.

**Guidance.** Draw boxes tightly around genuine signal islands; re-fit the model
after changing the rotation/alignment transform (the cache key tracks the
transform, so panels refresh correctly, but the model itself is not re-fit
automatically).

### 4.2 X/Y–TOF alignment line fit

**What it computes.** A linear drift correction
$\Delta = \mathrm{slope}\cdot t + \mathrm{intercept}$ applied to one spatial axis
of the ion coincidence map, so the dominant line becomes horizontal (spatial
position independent of TOF).

**Method.** The ROI points are binned into a 2D histogram, transformed with
$\log(1 + \mathrm{counts})$, smoothed with a $3\times3$ binomial kernel; the
per-column (TOF-bin) argmax gives the ridge; the brightest 70% of columns feed an
intensity-weighted linear regression, iterated up to 4 times with MAD-based
outlier rejection ($2.5\sigma$, floored at 1.5 coordinate bins). The stored
correction accumulates with any previous fit (`raw = display + previous`).

**Assumptions / limitations.**
- Exactly one straight ridge dominates the ROI. Crossing or parallel lines make
  the argmax hop between them; the MAD filter then keeps whichever line has more
  columns, not necessarily the intended one. *Draw a tight ROI around a single
  line.*
- Slope precision is quantized by the coordinate bin size of the fit grid.
- The fit is performed on transformed (rotated) coordinates: after changing the
  rotation angle, re-fit the alignment.

### 4.3 Ion transform chain

Ion coordinates are transformed identically everywhere via one shared helper:
subtract peak shift → rotate about the effective center → subtract
$\mathrm{slope}\cdot t + \mathrm{intercept}$ per axis (active alignment), with an
optional display-only TOF centering for the scatter panel. Order is fixed; the
alignment operates in the rotated frame. Consistency across panels is guaranteed
by routing all transforms through `_transform_ion_xy` and
`_apply_ion_tof_terms_to_xy`.

---

## Step 5: Electron ring-center estimation

Two estimators are available; both operate on raw scatter points (subsampled to
≤ ~64k) rather than binned images.

### 5.1 Quadrant symmetry (default)

**What it computes.** The center that makes the image 180°-rotationally
symmetric.

**Method.** Around a candidate center, points are assigned to four quadrants.
Each point is matched to the nearest point of the diagonally opposite quadrant —
implemented exactly as the nearest raw point to the antipode
$\mathbf{p}^* = 2\mathbf{c} - \mathbf{p}$ using a single KD-tree built once per
estimation (points are frozen). The score is a weighted mean squared folded-space
pair distance, with Gaussian weighting around the dominant radial shell
(estimated from the seed center), reduced leverage for points near the candidate
axes, and a quadrant-mass balance penalty; the update step moves the center to
the weighted mean of matched-pair midpoints. A bounded coarse grid search around
the seed, monotone backtracking line search, and a final local polish make the
iteration stable and order-independent.

**Assumptions.**
- The distribution has approximate 180° rotational symmetry about the true
  center (true for field-projected VMI images of an isotropic or cylindrically
  symmetric cloud).

**Known limitations.**
- Genuine asymmetries (dead detector sectors, beam-stop shadows, asymmetric
  dissociation, uneven detection efficiency) pull the estimate toward the
  symmetric majority.
- The shell weighting is seeded from an edge-circle estimate; if the outer
  envelope is noise-dominated the seed can mis-set the weighting shell, though
  the bounded coarse search around the seed limits the damage.
- Needs points in both diagonally opposite quadrants; a strongly one-sided cloud
  degenerates to the seed/fallback.

### 5.2 Polar outermost ring line

**What it computes.** The center that makes the *outermost* ring's radius
constant across all angles.

**Method.** A radial shell is frozen from the reference center (outermost
significant peak of the radial histogram, or a user Polar-ROI band), then the
flatness of its $(\theta \mapsto r)$ ridge is minimized over center candidates
with analytic gradients (ridge = support-weighted mean radius per angular
sector, with edge-quantile variants for narrow manual bands) and monotone
updates; the shell model may be re-anchored a bounded number of times. For
narrow manual ROIs a dedicated iterative variant detects the outer edge by
outward-step contrast per sector, smooths the edge path, and fits a (trimmed)
Kasa circle to the edge line.

**Assumptions / limitations.**
- The outermost shell must be a real ring, not detector-edge noise — supply a
  Polar ROI band (the GUI requires one) when the image edge is noisy.
- Incomplete angular coverage weakens the constraint; the coverage penalty in
  the straightness metric exposes this.
- Local optimizer: result depends mildly on the candidate seeds (manual guess,
  quadrant symmetry, edge circle).

### 5.3 Straightness metric and acceptance gate

**What it computes.** For any candidate center, `build_polar_histogram` extracts
the per-angle peak-radius line; the metric is

$$
S = \frac{\sigma_w(r_{\mathrm{peak}})}{\max(\bar r_{\mathrm{peak}}, 1)} +
    \gamma\,\big(1 - f_{\mathrm{valid}}\big),
$$

with count-weighted mean/std, valid-angle fraction $f_{\mathrm{valid}}$, and
$\gamma = 0.25$ (outermost) / $0.75$ (dominant).

**Acceptance gate.** A re-estimate replaces an already-applied center only if
$S$ improves by a relative margin (5×10⁻⁴), with ties broken on $\sigma_w$ plus a
validity guard. The **first** estimate bypasses the gate: without a previous
center the edits hold the user's manual guess, and on multi-ring data the
dominant-ridge metric is noisy (the "dominant" peak can flip between rings along
θ), so gating the first click could keep a rough manual center.

**Known limitations.**
- The metric is histogram-binned: its noise floor is one radial bin.
- On images with two comparable rings, the dominant-mode ridge flips between
  rings; prefer `polar_outermost` with a ROI band for such data, or judge the
  result by the polar peak line, not the score alone.

**Guidance.** `quadrant_symmetry` is the robust default (no prerequisites,
works on ring distributions). Use `polar_outermost` when you have a clean,
isolated outer ring or a well-chosen ROI band — it is the more accurate of the
two in that regime. With incomplete angular coverage, neither estimator is
trustworthy; fix the acquisition instead.

---

## Step 6: Ring selection and denoised centered binning

**What it computes.** A centered 2D histogram of the electron points inside the
inner circle, with a uniform background subtracted.

**Method.** With center $\mathbf{c}$, signal points are those with
$\|\mathbf{p} - \mathbf{c}\| \le r_{\mathrm{in}}$; with the outer-ring filter
enabled, points in the annulus
$r_{\mathrm{in}} < \|\mathbf{p} - \mathbf{c}\| \le r_{\mathrm{out}}$ are the
background sample. With $n_{\mathrm{out}}$ background points, the areal density
is $\lambda = n_{\mathrm{out}} / \pi (r_{\mathrm{out}}^2 - r_{\mathrm{in}}^2)$
and every bin whose center lies inside the inner circle has

$$
\widehat{N}_{\mathrm{bg}} = \lambda\,(\Delta x)^2
$$

subtracted. Negative results are clamped to zero; `removed_total` is the mass
difference after clamping. Bin edges sit at half-integer multiples of
`bin_size`, so bin centers lie exactly on integer multiples of `bin_size` and
$(0,0)$ is the central pixel — this grid is what lets the rBasex pixel radii map
onto the data coordinates without offset.

**Assumptions.**
- The background is spatially uniform across the inner disk and equal to the
  outer-annulus areal density.
- The annulus is signal-free (`r_out` beyond the outermost real ring).

**Known limitations.**
- Real background can vary radially (often larger near the center), so the flat
  model under-/over-subtracts locally and slightly reshapes peak wings.
- Clamping at zero (required because the Abel inversion needs a non-negative
  image) truncates negative Poisson fluctuations and biases the residual
  background upward; the bias is largest where $\lambda (\Delta x)^2$ is
  comparable to the signal.
- The inner set is defined by bin centers, so boundary bins are included
  coarsely.

**Guidance.** Choose `r_in` to enclose all rings of interest and `r_out` in a
genuinely empty region; if the removed fraction (`removed_noise`) is a large
share of the signal, prefer disabling denoising and letting the rBasex
regularization (`reg`) handle the background instead.

---

## Step 7: rBasex reconstruction and peak extraction

**What it computes.** The inverse Abel transform of the (denoised, centered)
projection image, its radial intensity $I(r)$, and the radial anisotropy
$\beta(r)$; then dominant peaks with their $(r, \beta, \mathrm{intensity})$.

**Method.** `pyabel.rbasex.rbasex_transform(direction="inverse")` expands the
image in basis functions that are analytic in the radial distributions — a
pBasex-like method formulated in terms of $I(r)$ and $\beta_n(r)$ directly, so
the inversion is a single basis-coefficient solve (regularizable with `reg`).
The angular expansion order `order` (default 2) and an `odd`-terms switch
control the angular physics; `rmax` limits the transform radius (default
"MIN": the largest radius with at least one full quadrant of data). The basis
set is cached on disk (`~/.cache/vmi_workflow/abel_basis`) and is bit-exact
across runs.

The recovered 3D distribution is parameterized as

$$
I(r, \theta) = \frac{I(r)}{4\pi}\Big[1 + \beta_2(r)\,P_2(\cos\theta)\Big]
$$

(for `order = 2`; higher orders add $P_4, \dots$). With $\beta_2 = 2$ the
distribution is parallel to the detector polarization axis, $\beta_2 = 0$
isotropic, $\beta_2 = -1$ perpendicular.

**Peak extraction.** $I(r)$ is Gaussian-smoothed and normalized;
`scipy.signal.find_peaks` (height/prominence/distance) selects peaks; each
peak's support is clipped at the valley minimum toward its neighbors; `area` is
the trapezoidal integral of the *raw* $I(r)$ over that segment (fallback: the
raw value at the peak index); `beta` is read at the peak index and clipped to
$[-2, 2]$.

**Units and grid notes (verified against pyabel 0.9.1).**
- The radial grid is integer pixel indices (`arange(rmax+1)`); the app
  multiplies by `bin_size`, which maps exactly onto the centered data
  coordinates (Step 6 grid). Radial resolution is therefore one pixel — peak
  radii are quantized to whole pixels, and $\beta(r)$ is reported at that raw
  resolution (pyabel's radial averaging window is left at 1).
- `numpy.histogram2d` output is transposed before the transform (image row = y).
- `I(r)` returned by pyabel is the full-sphere radial intensity; only the peak
  *detection* is smoothed, reported `i` values are unsmoothed.

**Assumptions / limitations.**
- Cylindrical symmetry about the vertical axis of the centered image; `odd`
  terms only allow top–bottom asymmetry, not tilt.
- `reg = None` (default) runs unregularized: noise amplification at large
  radius is possible, particularly with `order` high.
- Peak `area` includes any continuum under the peak (no baseline subtracted)
  and splits overlap wings at the valleys.
- The $\beta$ clip $[-2,2]$ matches pure-$P_2$ physics; it only guards numerical
  noise at `order = 2`, but at higher orders a leaked component can be
  distorted.
- Input is the *denoised* histogram; Step 6 biases (uniform background model,
  clamp-at-zero) propagate here.

**Guidance.** Use `order = 2` unless you know higher harmonics are present;
raise `order`/enable `odd` only with enough counts. Consider a small `reg`
when the radial profile is noisy at large r. Prefer reporting peak `area`
over `i` for relative yields of well-separated peaks.

---

## Known methodological limitations (summary)

| # | Where | Limitation / bias | Class |
|---|---|---|---|
| 1 | Trigger pairing | Adjacent-row matching loses pairs across merged-file boundaries; no sort defense | RISK |
| 2 | m/q calibration | `b = 0` ignores time-zero offset $t_0$; systematic m/q stretch away from reference; range mode assumes exactly one bin | RISK |
| 3 | Ion histogram BG fit | Envelope target + never-exceed projection bias the background *under* true baseline (esp. in valleys); monotone model forbids rising tails; display/diagnostic only | RISK (by design) |
| 4 | Ion TOF BG model | Density-ratio classifier: box quality decides everything; non-finite rows always removed; up to 28% in-box removal allowed | RISK |
| 5 | TOF alignment fit | Per-column argmax ridge hops between crossing/parallel lines; bin-quantized slope | RISK |
| 6 | Denoised binning | Uniform annular-density assumption; clamp-at-zero truncation biases residual background up | RISK |
| 7 | Center: quadrant symmetry | Requires 180° symmetry; asymmetries and one-sided coverage bias the estimate; seed-dependent shell | RISK (mild) |
| 8 | Center: polar outermost | Outermost-shell selection can follow noise without a ROI band; local optimizer | RISK (mild) |
| 9 | Straightness gate | Histogram-binned metric; dominant-ridge flips on multi-ring data (first estimate bypasses the gate) | NOTE |
| 10 | Center seeds | `edge_circle_center` keeps only the farthest point per angle bin (outlier-fragile); Kasa fit has algebraic bias on short arcs — acceptable because both are seeds with robust refinement | NOTE |
| 11 | rBasex radial grid | Radii quantized to whole pixels; $\beta(r)$ not radially averaged (window = 1); `r * bin_size` alignment exact | NOTE |
| 12 | Peak extraction | `area` includes underlying continuum; valleys split overlaps; `i` read from unsmoothed profile at smoothed peak index | NOTE |
| 13 | β clip | `beta` clipped to $[-2, 2]$ — exact guard for $P_2$, can distort leaked higher-order components | NOTE |

---

## References

1. A. T. J. B. Eppink and D. H. Parker,
   *Velocity map imaging of ions and electrons using electrostatic lenses:
   Application in photoelectron and photofragment ion imaging of molecular oxygen*,
   Rev. Sci. Instrum. **68**, 3477 (1997). doi:10.1063/1.1148310
   — original velocity-map imaging lens.
2. V. Dribinski, A. Ossadtchi, V. A. Mandelshtam and H. Reisler,
   *Reconstruction of Abel-transformable images: The Gaussian basis-set
   expansion Abel transform method* (BASEX),
   Rev. Sci. Instrum. **73**, 2634 (2002). doi:10.1063/1.1480904
3. G. A. Garcia, L. Nahon and I. Powis,
   *Two-dimensional charged particle image inversion using a polar basis
   function expansion* (pBasex),
   Rev. Sci. Instrum. **75**, 4989 (2004). doi:10.1063/1.1787603
   — the polar basis-set approach rBasex builds upon.
4. D. D. Hickstein, S. T. Gibson, R. Yurchak, D. D. Das and M. Ryazanov,
   *PyAbel — A Python package for Abel transforms*, Zenodo (2016),
   doi:10.5281/zenodo.47423 — the package used by this application
   (cite the Zenodo DOI of the pyabel version you use; there is no JOSS paper).
5. D. D. Hickstein, S. T. Gibson, R. Yurchak, D. D. Das and M. Ryazanov,
   *A direct comparison of high-speed methods for the numerical Abel transform*,
   Rev. Sci. Instrum. **90**, 065115 (2019). doi:10.1063/1.5092635;
   arXiv:1902.09007 — background on method choice and accuracy.
6. M. Ryazanov,
   *Development and implementation of methods for sliced velocity map imaging.
   Studies of overtone-induced dissociation and isomerization dynamics of
   hydroxymethyl radical (CH₂OH)*, Ph.D. dissertation, University of Southern
   California (2012) — origin of the rBasex basis functions; see also the
   PyAbel rBasex documentation (pyabel.readthedocs.io, "rBasex" method page).
