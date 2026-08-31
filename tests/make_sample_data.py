#!/usr/bin/env python3
"""Generate a small deterministic synthetic VMI data triplet for regression tests.

The output mimics the real reference data layout:

- trigger file ``<base>_DAn.dat``
      4 columns: ``event_no, ion_tof, ion_index, electron_index``
      (the app reads columns 2 and 3 with ``use_columns=(2, 3)`` and then
      swaps them to internal ``[electron_index, ion_index]`` order).
- electron file ``<base>.lmf_elec_DAn.dat``
      3 columns: ``x, y, t`` (pixel coordinates + electron TOF).
- ion file ``<base>.lmf_ion_DAn.dat``
      3 columns: ``x, y, t`` (pixel coordinates + ion TOF).

Physics sketch:
- electrons form two anisotropic Newton-sphere rings (r=60 px strong with
  beta=+1.2, r=110 px weaker with beta=-0.4, radial sigma 4 px) around a
  true detector center at (128.0, 125.5), plus ~4% uniform disk background;
- each electron event carries one correlated ion hit (position shrunk toward
  the center + jitter) whose TOF lands in a main peak (85%), a secondary
  peak (10%) or a flat background (5%).

Trigger-row structure (this is what the pairing code in
``VMI_workflow_core`` matches on: adjacent-row deltas of the two index
columns):

- ~92% coincidence rows: both indices advance by exactly +1  ->  feeds the
  strict 1e+1i selector;
- ~0.6% "two ions one electron" rows: electron +1, ion +2 (the intermediate
  ion is still appended to the ion file so lookups stay in bounds)  ->  feeds
  the 1e+2i selector;
- ~0.15% "three ions one electron" rows (ion +3)  ->  feeds the 1e+3i selector;
- ~3.5% electron-only rows (electron +1, ion column NaN) and ~3.5% ion-only
  rows (ion +1, electron column NaN)  ->  chain breakers, exercising the NaN
  handling; they are counted by the "all valid rows" mode.

Everything is driven by a fixed RNG seed so goldens are reproducible.

Run:  python tests/make_sample_data.py [--out tests/sample_data] [--rows 30000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Deterministic configuration (do not change: goldens depend on these).
# --------------------------------------------------------------------------
SEED = 20260831
BASENAME = "synth_ar100_vmi"
TRIGGER_NAME = f"{BASENAME}_DAn.dat"
ELECTRON_NAME = f"{BASENAME}.lmf_elec_DAn.dat"
ION_NAME = f"{BASENAME}.lmf_ion_DAn.dat"

CENTER_XY = (128.0, 125.5)

# (radius px, radial sigma px, fraction of electrons, beta2 anisotropy)
RING1 = (60.0, 4.0, 0.72, 1.2)
RING2 = (110.0, 4.0, 0.24, -0.4)
BG_FRACTION = 0.04  # remainder of the electron fractions above
BG_RADIUS = 148.0  # uniform background disk radius (px)

ELECTRON_T_MEAN = 3520.0
ELECTRON_T_SIGMA = 4.0

# (mean ns, sigma ns, fraction of ions)
ION_T_MAIN = (8250.0, 55.0, 0.85)
ION_T_SECONDARY = (11900.0, 70.0, 0.10)
ION_T_BG_RANGE = (7000.0, 14000.0)  # remaining 0.05: flat background

# Trigger row mixture.
P_SKIP_2I = 0.006  # electron +1 / ion +2 rows (1e+2i selector food)
P_SKIP_3I = 0.0015  # electron +1 / ion +3 rows (1e+3i selector food)
P_ELECTRON_ONLY = 0.035  # electron +1, ion column NaN
P_ION_ONLY = 0.035  # ion +1, electron column NaN
# Remainder (~0.9225): strict +1/+1 coincidence rows.

ION_SHRINK_FACTOR = 0.25  # ion xy = center + shrink * (e_xy - center)
ION_XY_JITTER = 8.0  # px gaussian jitter added on top


def _sample_anisotropic_theta(rng: np.random.Generator, beta: float, n: int) -> np.ndarray:
    """Sample n angles from I(theta) proportional to 1 + beta * P2(cos theta)."""
    bound = 1.0 + abs(beta)
    out = np.empty(0, dtype=np.float64)
    while out.size < n:
        need = max(1024, int((n - out.size) * 2))
        cand = rng.uniform(-np.pi, np.pi, need)
        p2 = 0.5 * (3.0 * np.cos(cand) ** 2 - 1.0)
        weight = 1.0 + beta * p2
        keep = rng.uniform(0.0, bound, need) <= weight
        out = np.concatenate((out, cand[keep]))
    return out[:n]


def _sample_electron_xy(rng: np.random.Generator, n: int) -> np.ndarray:
    """Electron detector hits: two anisotropic rings + uniform disk background."""
    u = rng.uniform(0.0, 1.0, n)
    kind = np.full(n, 2, dtype=np.int64)  # 2 = background
    kind[u < RING1[2]] = 0
    kind[(u >= RING1[2]) & (u < RING1[2] + RING2[2])] = 1

    xy = np.empty((n, 2), dtype=np.float64)
    cx, cy = CENTER_XY
    for k, (r0, sigma, _frac, beta) in ((0, RING1), (1, RING2)):
        sel = np.flatnonzero(kind == k)
        if sel.size == 0:
            continue
        theta = _sample_anisotropic_theta(rng, beta, sel.size)
        r = r0 + sigma * rng.standard_normal(sel.size)
        xy[sel, 0] = cx + r * np.cos(theta)
        xy[sel, 1] = cy + r * np.sin(theta)
    sel = np.flatnonzero(kind == 2)
    if sel.size:
        theta = rng.uniform(-np.pi, np.pi, sel.size)
        r = BG_RADIUS * np.sqrt(rng.uniform(0.0, 1.0, sel.size))
        xy[sel, 0] = cx + r * np.cos(theta)
        xy[sel, 1] = cy + r * np.sin(theta)
    return xy


def _sample_ion_tof(rng: np.random.Generator, n: int) -> np.ndarray:
    """Ion TOF values: main peak + secondary peak + flat background."""
    u = rng.uniform(0.0, 1.0, n)
    t = np.empty(n, dtype=np.float64)
    main_sel = u < ION_T_MAIN[2]
    second_sel = (u >= ION_T_MAIN[2]) & (u < ION_T_MAIN[2] + ION_T_SECONDARY[2])
    bg_sel = ~(main_sel | second_sel)
    t[main_sel] = ION_T_MAIN[0] + ION_T_MAIN[1] * rng.standard_normal(int(main_sel.sum()))
    t[second_sel] = ION_T_SECONDARY[0] + ION_T_SECONDARY[1] * rng.standard_normal(int(second_sel.sum()))
    if bg_sel.any():
        lo, hi = ION_T_BG_RANGE
        t[bg_sel] = rng.uniform(lo, hi, int(bg_sel.sum()))
    return t


def _sample_ion_xy(rng: np.random.Generator, electron_xy: np.ndarray) -> np.ndarray:
    """Ion hits correlated with their partner electron hit."""
    cx, cy = CENTER_XY
    base = (cx, cy) + ION_SHRINK_FACTOR * (electron_xy - np.array(CENTER_XY))
    jitter = ION_XY_JITTER * rng.standard_normal(electron_xy.shape)
    return base + jitter


def generate(out_dir: str | Path = Path(__file__).resolve().parent / "sample_data", n_rows: int = 30_000) -> dict:
    """Generate the triplet plus a ``generation_stats.json`` manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    electron_rows: list[tuple[float, float, float]] = []  # x, y, t
    ion_rows: list[tuple[float, float, float]] = []  # x, y, t
    trigger_rows: list[str] = []
    counts = {"coincidence": 0, "skip2": 0, "skip3": 0, "e_only": 0, "i_only": 0}

    e_idx = -1  # last recorded electron index (file row of electron file)
    i_idx = -1  # last recorded ion index (file row of ion file)
    event_no = 0
    last_ion_t = 0.0

    def next_electron() -> tuple[np.ndarray, float]:
        nonlocal e_idx
        (xy,) = _sample_electron_xy(rng, 1)
        t = ELECTRON_T_MEAN + ELECTRON_T_SIGMA * float(rng.standard_normal())
        electron_rows.append((float(xy[0]), float(xy[1]), t))
        e_idx += 1
        return xy, t

    def next_ion(electron_xy: np.ndarray) -> float:
        """Append one correlated ion hit; return its TOF."""
        nonlocal i_idx, last_ion_t
        ion_xy = _sample_ion_xy(rng, electron_xy.reshape(1, 2))
        t = float(_sample_ion_tof(rng, 1)[0])
        ion_rows.append((float(ion_xy[0, 0]), float(ion_xy[0, 1]), t))
        i_idx += 1
        last_ion_t = t
        return t

    for row in range(n_rows):
        u = rng.uniform()
        if row == 0:
            kind = "coincidence"  # keep indices non-negative from the start
        elif u < P_SKIP_3I:
            kind = "skip3"
        elif u < P_SKIP_3I + P_SKIP_2I:
            kind = "skip2"
        elif u < P_SKIP_3I + P_SKIP_2I + P_ELECTRON_ONLY:
            kind = "e_only"
        elif u < P_SKIP_3I + P_SKIP_2I + P_ELECTRON_ONLY + P_ION_ONLY:
            kind = "i_only"
        else:
            kind = "coincidence"
        counts[kind] += 1
        event_no += int(rng.integers(1, 4))  # ragged event numbering like real data

        if kind == "coincidence":
            e_xy, _e_t = next_electron()
            ion_t = next_ion(e_xy)
            trigger_rows.append(f"{event_no},{ion_t:.6f},{i_idx},{e_idx}")
        elif kind == "skip2":
            e_xy, _e_t = next_electron()
            next_ion(e_xy)  # swallowed ion: in the ion file, not on any trigger row
            ion_t = next_ion(e_xy)  # recorded ion (index i_idx)
            trigger_rows.append(f"{event_no},{ion_t:.6f},{i_idx},{e_idx}")
        elif kind == "skip3":
            e_xy, _e_t = next_electron()
            next_ion(e_xy)
            next_ion(e_xy)
            ion_t = next_ion(e_xy)
            trigger_rows.append(f"{event_no},{ion_t:.6f},{i_idx},{e_idx}")
        elif kind == "e_only":
            next_electron()
            trigger_rows.append(f"{event_no},NaN,NaN,{e_idx}")
        else:  # i_only
            _ = next_ion(_sample_electron_xy(rng, 1)[0])
            trigger_rows.append(f"{event_no},{last_ion_t:.6f},{i_idx},NaN")

    trigger_path = out_dir / TRIGGER_NAME
    electron_path = out_dir / ELECTRON_NAME
    ion_path = out_dir / ION_NAME
    trigger_path.write_text("\n".join(trigger_rows) + "\n", encoding="ascii")
    electron_path.write_text(
        "\n".join(f"{x:.6f},{y:.6f},{t:.6f}" for x, y, t in electron_rows) + "\n",
        encoding="ascii",
    )
    ion_path.write_text(
        "\n".join(f"{x:.6f},{y:.6f},{t:.6f}" for x, y, t in ion_rows) + "\n",
        encoding="ascii",
    )

    stats = {
        "seed": SEED,
        "trigger_rows": n_rows,
        "electron_rows": len(electron_rows),
        "ion_rows": len(ion_rows),
        "row_kinds": counts,
        "expected_pairing": {
            # adjacent (+1/+1) pairs: coincidence rows whose previous row is
            # also a coincidence row (row 0 has no previous).
            "one_e_one_i_approx": max(0, counts["coincidence"] - 1),
            "one_e_two_i_pairs_approx": 2 * counts["skip2"],
            "one_e_three_i_pairs_approx": 3 * counts["skip3"],
            "all_valid_rows_approx": counts["coincidence"] + counts["skip2"] + counts["skip3"],
        },
        "center_xy": list(CENTER_XY),
        "ring_radii_px": [RING1[0], RING2[0]],
        "ion_tof_peaks_ns": [ION_T_MAIN[0], ION_T_SECONDARY[0]],
        "files": {
            "trigger": TRIGGER_NAME,
            "electron": ELECTRON_NAME,
            "ion": ION_NAME,
        },
    }
    (out_dir / "generation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "sample_data",
        help="Output directory (default: tests/sample_data)",
    )
    parser.add_argument("--rows", type=int, default=30_000, help="Number of trigger rows")
    args = parser.parse_args(argv)
    stats = generate(args.out, args.rows)
    print(f"Sample data written to {args.out}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
