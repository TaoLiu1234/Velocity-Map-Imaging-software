#!/usr/bin/env python3
"""One-shot helper: capture center-estimator lock values from the CURRENT code.

Used once before the quadrant_symmetry_center KD-tree hoisting refactor so
tests/test_core.py can pin the pre-change (reference) results as hardcoded
expected values. Kept in the repo so the capture recipe stays reproducible.
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

from VMI_workflow_core import polar_outermost_center, quadrant_symmetry_center  # noqa: E402

LOCK_SEED = 20260831
LOCK_TRUE_CENTER = (34.5, -21.25)


def _lock_cloud(n_points: int) -> np.ndarray:
    """Deterministic two-ring + background cloud like the sample generator."""
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


def main() -> int:
    out = {}
    for n in (5_000, 50_000, 500_000):
        pts = _lock_cloud(n)
        fallback = (LOCK_TRUE_CENTER[0] + 3.0, LOCK_TRUE_CENTER[1] - 2.0)
        c_q = quadrant_symmetry_center(pts, fallback)
        out[f"quadrant_{n}"] = [float(c_q[0]), float(c_q[1])]
        print(f"quadrant n={n}: {float(c_q[0]):.12f} {float(c_q[1]):.12f}", flush=True)
        c_p = polar_outermost_center(pts, fallback)
        out[f"polar_{n}"] = [float(c_p[0]), float(c_p[1])]
        print(f"polar    n={n}: {float(c_p[0]):.12f} {float(c_p[1]):.12f}", flush=True)
    dest = TESTS_DIR / "_center_lock_capture.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
