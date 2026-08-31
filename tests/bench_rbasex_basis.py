#!/usr/bin/env python3
"""Measure rBasex basis persistence across processes (Task: basis_dir cache).

Builds the same centered histogram the smoke workflow feeds to rBasex (sample
triplet, smoke center/ring/bin parameters), then runs
`run_rbasex_reconstruction` once and reports the wall time and peaks.

Run it TWICE in two fresh processes:
    python tests/bench_rbasex_basis.py
    python tests/bench_rbasex_basis.py
The first process generates the basis (and saves it under
~/.cache/vmi_workflow/abel_basis); the second loads it from disk and should
be substantially faster. Peak values must be identical between runs.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VMI_workflow_core import build_denoised_centered_histogram, fast_read_csv_float64  # noqa: E402
from VMI_workflow_reconstruction import run_rbasex_reconstruction  # noqa: E402

SAMPLE_BASE = TESTS_DIR / "sample_data" / "synth_ar100_vmi"
# Smoke workflow parameters (see tests/README.md).
CENTER = (126.0, 123.0)
INNER_R = 118.0
OUTER_R = 140.0
BIN = 0.5

SETTINGS = {
    "order": 2,
    "odd": False,
    "reg": None,
    "rmax": "MIN",
    "peak_smooth_sigma": 0.0,
    "peak_height": 0.12,
    "peak_prominence": 0.08,
    "peak_min_dist_frac": 0.06,
    "max_peaks": 5,
    "display_percentile": 99.7,
}


def _build_input() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    trigger = fast_read_csv_float64(str(SAMPLE_BASE) + "_DAn.dat", n_columns=4, use_columns=(2, 3))
    trigger = trigger[:, [1, 0]]  # file [ion, electron] -> internal [electron, ion]
    electron = fast_read_csv_float64(str(SAMPLE_BASE) + ".lmf_elec_DAn.dat", n_columns=3, use_columns=(0, 1, 2))
    ion = fast_read_csv_float64(str(SAMPLE_BASE) + ".lmf_ion_DAn.dat", n_columns=3, use_columns=(0, 1, 2))
    in_bounds = (
        (trigger[:, 0] >= 0)
        & (trigger[:, 0] < electron.shape[0])
        & (trigger[:, 1] >= 0)
        & (trigger[:, 1] < ion.shape[0])
    )
    trigger = trigger[in_bounds]
    e_pts = electron[trigger[:, 0].astype(np.int64)]
    i_pts = ion[trigger[:, 1].astype(np.int64)]
    dist2 = (e_pts[:, 0] - CENTER[0]) ** 2 + (e_pts[:, 1] - CENTER[1]) ** 2
    ring = e_pts[dist2 <= INNER_R**2]
    ring_ion = i_pts[dist2 <= INNER_R**2]
    hist = build_denoised_centered_histogram(ring[:, :2], ring_ion[:, :2], INNER_R, OUTER_R, BIN)
    assert hist is not None
    recon_input = np.asarray(hist["hist_denoised"].T, dtype=np.float64)
    return recon_input, hist["xedges"], hist["yedges"], float(hist.get("bin_size", BIN))


def main() -> int:
    image, xedges, yedges, bin_size = _build_input()
    print(f"input image: {image.shape[0]}x{image.shape[1]} (bin {bin_size})")
    t0 = time.perf_counter()
    result = run_rbasex_reconstruction(image, xedges, yedges, bin_size, SETTINGS)
    elapsed = time.perf_counter() - t0
    if result.get("error"):
        print(f"rBasex FAILED: {result['error']}")
        return 1
    peaks = result["peaks"]
    print(f"rBasex wall time: {elapsed:.2f} s")
    for p in peaks:
        print(
            "  peak r={:.1f} beta={:.6f} i={:.6f}".format(float(p["r"]), float(p["beta"]), float(p["i"]))
        )
    out_path = Path(tempfile.gettempdir()) / "vmi_rbasex_bench_last_peaks.json"
    import json

    out_path.write_text(json.dumps(peaks, indent=2), encoding="utf-8")
    prev = Path(tempfile.gettempdir()) / "vmi_rbasex_bench_prev_peaks.json"
    if prev.is_file():
        previous = json.loads(prev.read_text(encoding="utf-8"))
        print(f"peaks identical to previous run: {previous == peaks}")
    else:
        print("peaks identical to previous run: (no previous run)")
    prev.write_text(json.dumps(peaks, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
