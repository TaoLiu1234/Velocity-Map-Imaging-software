#!/usr/bin/env python3
"""Numerics golden tests for VMI_workflow_core (regression safety net).

These tests pin the *current* numerical behaviour of the pure numpy/scipy
core module on fixed inputs (synthetic arrays + the generated sample data
triplet in ``tests/sample_data``).  They are the contract that must survive
refactors of the app.

Golden workflow:
- ``python tests/test_core.py --update-golden`` regenerates ``tests/golden_core.json``
  from the current code (do this ONLY when an intentional behaviour change
  has been reviewed).
- plain run (or pytest) compares live results against the golden file with
  exact equality on 6-decimal rounded values.

Covered (per regression plan):
- ``fast_read_csv_float64`` on the sample triplet (shapes + spot values);
- all four trigger pairing selectors and their pair counts;
- ``build_denoised_centered_histogram`` checksum on a fixed synthetic input;
- ``geometric_median``, ``circle_fit_kasa``, ``edge_circle_center`` on fixed
  synthetic point clouds;
- (rBasex is intentionally NOT covered here -- it is exercised by the E2E
  smoke test.)

Run:  python tests/test_core.py [--update-golden]
  or: pytest tests/test_core.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VMI_workflow_core import (  # noqa: E402
    build_denoised_centered_histogram,
    circle_fit_kasa,
    edge_circle_center,
    fast_read_csv_float64,
    geometric_median,
    polar_outermost_center,
    quadrant_symmetry_center,
    select_all_one_pairs,
    select_increment_pairs,
    select_one_e_three_i_pairs,
    select_one_e_two_i_pairs,
)

SAMPLE_DIR = TESTS_DIR / "sample_data"
GOLDEN_PATH = TESTS_DIR / "golden_core.json"
GOLDEN_VERSION = 1

# Fixed parameters of the synthetic fixed-input cases (do not change).
DENOISE_SEED = 7
DENOISE_INNER_R = 55.0
DENOISE_OUTER_R = 90.0
DENOISE_BIN = 1.0

CENTER_SEED = 11
CENTER_TRUE = (12.5, -7.25)
CENTER_RING_R = 40.0
CENTER_RING_SIGMA = 1.5
CENTER_N_POINTS = 800

# Lock-test fixtures for the heavy center estimators. Expected values were
# captured from the pre-refactor code (see _capture_center_locks.py) and are
# hardcoded here on purpose: any deviation means a refactor changed numerics.
LOCK_SEED = 20260831
LOCK_TRUE_CENTER = (34.5, -21.25)
LOCK_SIZES = (5_000, 50_000, 500_000)
# Values are compared at 10 decimals; captured 2026-08-31 pre KD-tree hoisting.
CENTER_LOCK_EXPECTED = {
    "quadrant_5000": (35.4208280258751, -20.9121156655083),
    "quadrant_50000": (34.5561108518855, -21.1512696614041),
    "quadrant_500000": (34.0303500323096, -21.2739400661696),
    "polar_5000": (35.4208280258751, -20.9121156655083),
    "polar_50000": (34.5078924300084, -21.1309842061115),
    "polar_500000": (35.3678735011624, -21.4700301073210),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_sample_data() -> None:
    """Generate the sample triplet on demand (keeps the suite self-contained)."""
    needed = [
        SAMPLE_DIR / "synth_ar100_vmi_DAn.dat",
        SAMPLE_DIR / "synth_ar100_vmi.lmf_elec_DAn.dat",
        SAMPLE_DIR / "synth_ar100_vmi.lmf_ion_DAn.dat",
    ]
    if not all(p.is_file() for p in needed):
        import make_sample_data  # same directory

        make_sample_data.generate(SAMPLE_DIR)


def _load_sample_arrays() -> dict[str, np.ndarray]:
    """Load the sample triplet exactly like MainWindow._LoadWorker does."""
    base = SAMPLE_DIR / "synth_ar100_vmi"
    trigger = fast_read_csv_float64(str(base) + "_DAn.dat", n_columns=4, use_columns=(2, 3))
    trigger = ensure_trigger_internal_order(trigger)
    electron = fast_read_csv_float64(str(base) + ".lmf_elec_DAn.dat", n_columns=3, use_columns=(0, 1, 2))
    ion = fast_read_csv_float64(str(base) + ".lmf_ion_DAn.dat", n_columns=3, use_columns=(0, 1, 2))
    return {"trigger": trigger, "electron": electron, "ion": ion}


def ensure_trigger_internal_order(trigger: np.ndarray) -> np.ndarray:
    """Mirror the app's column swap: file [ion, electron] -> internal [electron, ion]."""
    trigger = np.asarray(trigger, dtype=np.float64)
    if trigger.ndim == 1:
        trigger = trigger.reshape(1, -1)
    return trigger[:, [1, 0]]


def _count_in_bounds(e_idx: np.ndarray, i_idx: np.ndarray, arrays: dict[str, np.ndarray]) -> int:
    if e_idx.size == 0:
        return 0
    in_bounds = (
        (e_idx >= 0)
        & (e_idx < arrays["electron"].shape[0])
        & (i_idx >= 0)
        & (i_idx < arrays["ion"].shape[0])
    )
    return int(in_bounds.sum())


def _fixed_denoise_input() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic centered ring + annulus noise for the denoise checksum."""
    rng = np.random.default_rng(DENOISE_SEED)
    n_sig = 6000
    theta = rng.uniform(-np.pi, np.pi, n_sig)
    r = 40.0 + 3.0 * rng.standard_normal(n_sig)
    signal = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    n_noise = 1500
    theta_n = rng.uniform(-np.pi, np.pi, n_noise)
    r_n = np.sqrt(rng.uniform(DENOISE_INNER_R**2, DENOISE_OUTER_R**2, n_noise))
    noise = np.column_stack((r_n * np.cos(theta_n), r_n * np.sin(theta_n)))
    return signal, noise


def _fixed_center_input() -> np.ndarray:
    """Deterministic noisy ring around CENTER_TRUE for center-estimator pins."""
    rng = np.random.default_rng(CENTER_SEED)
    theta = rng.uniform(-np.pi, np.pi, CENTER_N_POINTS)
    r = CENTER_RING_R + CENTER_RING_SIGMA * rng.standard_normal(CENTER_N_POINTS)
    return np.column_stack(
        (
            CENTER_TRUE[0] + r * np.cos(theta),
            CENTER_TRUE[1] + r * np.sin(theta),
        )
    )


def _lock_cloud(n_points: int) -> np.ndarray:
    """Deterministic two-ring + background cloud for the center lock tests."""
    rng = np.random.default_rng(LOCK_SEED + n_points)
    n1 = int(n_points * 0.45)
    n2 = int(n_points * 0.35)
    nb = n_points - n1 - n2
    theta1 = rng.uniform(-np.pi, np.pi, n1)
    r1 = 60.0 + 4.0 * rng.standard_normal(n1)
    theta2 = rng.uniform(-np.pi, np.pi, n2)
    r2 = 110.0 + 5.0 * rng.standard_normal(n2)
    theta_b = rng.uniform(-np.pi, np.pi, nb)
    r_b = np.sqrt(rng.uniform(0.0, 150.0**2, nb))
    rr = np.concatenate((r1, r2, r_b))
    tt = np.concatenate((theta1, theta2, theta_b))
    return np.column_stack(
        (
            LOCK_TRUE_CENTER[0] + rr * np.cos(tt),
            LOCK_TRUE_CENTER[1] + rr * np.sin(tt),
        )
    )


def _lock_fallback() -> tuple[float, float]:
    """Fixed off-center fallback used by both lock tests."""
    return (LOCK_TRUE_CENTER[0] + 3.0, LOCK_TRUE_CENTER[1] - 2.0)


def _rounded(value) -> float | int | list | dict:
    """JSON-safe structure with all floats rounded to 6 decimals."""
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not np.isfinite(f):
            return str(f)
        return round(f, 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def compute_golden() -> dict:
    """Run every pinned computation and return the golden dict."""
    _require_sample_data()
    arrays = _load_sample_arrays()

    # 1) raw loading facts -------------------------------------------------
    trigger = arrays["trigger"]
    electron = arrays["electron"]
    ion = arrays["ion"]
    loading = {
        "trigger_shape": [int(trigger.shape[0]), int(trigger.shape[1])],
        "electron_shape": [int(electron.shape[0]), int(electron.shape[1])],
        "ion_shape": [int(ion.shape[0]), int(ion.shape[1])],
        "trigger_first_row": [round(float(v), 6) if np.isfinite(v) else str(v) for v in trigger[0]],
        "electron_first_row": [round(float(v), 6) for v in electron[0]],
        "ion_first_row": [round(float(v), 6) for v in ion[0]],
        "trigger_nan_rows": int(np.count_nonzero(np.isnan(trigger).any(axis=1))),
    }

    # 2) pairing counts (post in-bounds filter, like process_and_plot) -----
    pair_functions = {
        "coincidence_1e1i": select_increment_pairs,
        "all_one": select_all_one_pairs,
        "one_e_two_i": select_one_e_two_i_pairs,
        "one_e_three_i": select_one_e_three_i_pairs,
    }
    pairing: dict[str, dict[str, int]] = {}
    for name, fn in pair_functions.items():
        e_idx, i_idx = fn(trigger)
        pairing[name] = {
            "selected_rows": int(e_idx.size),
            "in_bounds_pairs": _count_in_bounds(e_idx, i_idx, arrays),
        }

    # 3) denoised centered histogram checksum ------------------------------
    signal, noise = _fixed_denoise_input()
    hist = build_denoised_centered_histogram(
        signal, noise, DENOISE_INNER_R, DENOISE_OUTER_R, DENOISE_BIN
    )
    assert hist is not None
    denoise = {
        "hist_signal_sum": float(np.sum(hist["hist_signal"])),
        "hist_denoised_sum": float(np.sum(hist["hist_denoised"])),
        "removed_total": float(hist["removed_total"]),
        "expected_inner_noise_total": float(hist["expected_inner_noise_total"]),
        "shape": [int(hist["hist_denoised"].shape[0]), int(hist["hist_denoised"].shape[1])],
        "signal_count": int(hist["signal_count"]),
        "noise_count": int(hist["noise_count"]),
        "hist_denoised_max": float(np.max(hist["hist_denoised"])),
    }

    # 4) center estimation primitives on fixed input ------------------------
    pts = _fixed_center_input()
    geo = geometric_median(pts)
    kasa = circle_fit_kasa(pts)
    assert kasa is not None
    edge = edge_circle_center(pts, np.array([10.0, -5.0]))
    centers = {
        "geometric_median": [float(geo[0]), float(geo[1])],
        "circle_fit_kasa": [float(kasa[0]), float(kasa[1]), float(kasa[2])],
        "edge_circle_center": [float(edge[0]), float(edge[1])],
    }

    return {
        "golden_version": GOLDEN_VERSION,
        "loading": loading,
        "pairing": pairing,
        "denoise": denoise,
        "centers": centers,
    }


def _compare(golden: dict, live: dict, path: str = "") -> list[str]:
    mismatches: list[str] = []
    if isinstance(golden, dict) and isinstance(live, dict):
        for key in sorted(set(golden) | set(live)):
            sub_path = f"{path}.{key}" if path else str(key)
            if key not in golden:
                mismatches.append(f"{sub_path}: new key not in golden (value={live[key]!r})")
            elif key not in live:
                mismatches.append(f"{sub_path}: key missing in live result")
            else:
                mismatches.extend(_compare(golden[key], live[key], sub_path))
    elif isinstance(golden, list) and isinstance(live, list):
        if len(golden) != len(live):
            mismatches.append(f"{path}: length {golden!r} != {live!r}")
        else:
            for idx, (g, l) in enumerate(zip(golden, live)):
                mismatches.extend(_compare(g, l, f"{path}[{idx}]"))
    else:
        if golden != live:
            mismatches.append(f"{path}: golden={golden!r} live={live!r}")
    return mismatches


# ---------------------------------------------------------------------------
# pytest-compatible tests (each pins one section of the golden file)
# ---------------------------------------------------------------------------
def _live_rounded() -> dict:
    """Live golden content, rounded exactly like the stored golden file."""
    return _rounded(compute_golden())  # type: ignore[arg-type]


def _load_golden() -> dict:
    if not GOLDEN_PATH.is_file():
        raise AssertionError(
            f"Golden file {GOLDEN_PATH} missing. Regenerate with: "
            "python tests/test_core.py --update-golden"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_fast_read_csv_and_pairing_golden():
    golden = _load_golden()
    live = _live_rounded()
    for section in ("loading", "pairing"):
        mismatches = _compare(golden[section], live[section], section)
        assert not mismatches, "Golden mismatch:\n" + "\n".join(mismatches)


def test_denoise_checksum_golden():
    golden = _load_golden()
    live = _live_rounded()
    mismatches = _compare(golden["denoise"], live["denoise"], "denoise")
    assert not mismatches, "Golden mismatch:\n" + "\n".join(mismatches)


def test_center_estimators_golden():
    golden = _load_golden()
    live = _live_rounded()
    mismatches = _compare(golden["centers"], live["centers"], "centers")
    assert not mismatches, "Golden mismatch:\n" + "\n".join(mismatches)


def test_sample_data_sanity():
    """Structural sanity independent of the golden file."""
    arrays = _load_sample_arrays()
    assert arrays["trigger"].shape[1] == 2
    assert arrays["electron"].shape[1] == 3
    assert arrays["ion"].shape[1] == 3
    e_idx, i_idx = select_increment_pairs(arrays["trigger"])
    assert e_idx.size > 5000, "coincidence pairing should find thousands of pairs"
    assert _count_in_bounds(e_idx, i_idx, arrays) == int(e_idx.size)


def test_quadrant_symmetry_center_lock():
    """Pin quadrant_symmetry_center on fixed clouds (hardcoded pre-change values)."""
    fallback = _lock_fallback()
    for n in LOCK_SIZES:
        pts = _lock_cloud(n)
        got = quadrant_symmetry_center(pts, fallback)
        exp = CENTER_LOCK_EXPECTED[f"quadrant_{n}"]
        assert round(float(got[0]), 10) == round(exp[0], 10), f"quadrant cx mismatch at n={n}: {got[0]!r} != {exp!r}"
        assert round(float(got[1]), 10) == round(exp[1], 10), f"quadrant cy mismatch at n={n}: {got[1]!r} != {exp!r}"


def test_polar_outermost_center_lock():
    """Pin polar_outermost_center on fixed clouds (hardcoded pre-change values)."""
    fallback = _lock_fallback()
    for n in LOCK_SIZES:
        pts = _lock_cloud(n)
        got = polar_outermost_center(pts, fallback)
        exp = CENTER_LOCK_EXPECTED[f"polar_{n}"]
        assert round(float(got[0]), 10) == round(exp[0], 10), f"polar cx mismatch at n={n}: {got[0]!r} != {exp!r}"
        assert round(float(got[1]), 10) == round(exp[1], 10), f"polar cy mismatch at n={n}: {got[1]!r} != {exp!r}"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    update = "--update-golden" in argv
    live = _live_rounded()
    if update:
        GOLDEN_PATH.write_text(json.dumps(live, indent=2), encoding="utf-8")
        print(f"Golden file written: {GOLDEN_PATH}")
        return 0
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    mismatches = _compare(golden, live)
    if mismatches:
        print(f"FAIL: {len(mismatches)} golden mismatch(es) in {GOLDEN_PATH.name}:")
        for m in mismatches[:40]:
            print(f"  - {m}")
        return 1
    print(f"OK: all core numerics match {GOLDEN_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
