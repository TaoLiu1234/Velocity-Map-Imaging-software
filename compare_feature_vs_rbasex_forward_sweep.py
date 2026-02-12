from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from abel_benchmark_framework import BenchmarkMetrics, BenchmarkRunner
from feature_hansenlaw_forward_param_sweep import SweepPoint, build_sweep_plan


def _metric_row(point: SweepPoint, algorithm: str, metrics: BenchmarkMetrics) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "scenario": point.scenario.name,
        "design": point.design,
        "factor": point.factor,
        "level": point.level_name,
        "replicate": int(point.replicate),
        "npk": float(metrics.peak_count_error_pct),
        "r0": float(metrics.r0_error_pct_mean),
        "sigma": float(metrics.sigma_error_pct_mean),
        "beta": float(metrics.beta_error_abs_mean),
        "br": float(metrics.br_error_pct_mean),
        "time": float(metrics.computation_time),
    }


def _group_vals(rows: List[Dict[str, Any]], algorithm: str, key: str) -> np.ndarray:
    vals = [float(r[key]) for r in rows if r["algorithm"] == algorithm and np.isfinite(float(r[key]))]
    return np.asarray(vals, dtype=float)


def run_compare_sweep(
    sweep_points: List[SweepPoint],
    *,
    feature_kwargs: Dict[str, Any],
    rbasex_kwargs: Dict[str, Any],
    progress_every: int = 120,
) -> List[Dict[str, Any]]:
    runner = BenchmarkRunner(
        algorithms=["feature_hansenlaw_fusion", "rbasex"],
        algorithm_kwargs={
            "feature_hansenlaw_fusion": dict(feature_kwargs),
            "rbasex": dict(rbasex_kwargs),
        },
    )

    rows: List[Dict[str, Any]] = []
    total = len(sweep_points)
    for idx, point in enumerate(sweep_points, start=1):
        metrics_map = runner.run_scenario(point.scenario)
        for alg in ("feature_hansenlaw_fusion", "rbasex"):
            m = metrics_map.get(alg)
            if m is not None:
                rows.append(_metric_row(point, alg, m))
        if (idx % max(1, progress_every) == 0) or idx == total:
            msg = [f"[{idx:5d}/{total}] {point.design:<8} {point.factor}"]
            for alg in ("feature_hansenlaw_fusion", "rbasex"):
                m = metrics_map.get(alg)
                if m is None:
                    continue
                msg.append(f"{alg}:r0={m.r0_error_pct_mean:.2f}% sigma={m.sigma_error_pct_mean:.2f}% beta={m.beta_error_abs_mean:.3f}")
            print(" | ".join(msg))
    return rows


def _mean_dict(rows: List[Dict[str, Any]], algorithm: str) -> Dict[str, float]:
    return {k: float(np.mean(_group_vals(rows, algorithm, k))) for k in ["npk", "r0", "sigma", "beta", "br", "time"]}


def _build_pair_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        out[(str(r["scenario"]), str(r["algorithm"]))] = r
    return out


def _compute_wins(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    idx = _build_pair_index(rows)
    wins = {
        "npk": {"feature_hansenlaw_fusion": 0, "rbasex": 0, "tie": 0},
        "r0": {"feature_hansenlaw_fusion": 0, "rbasex": 0, "tie": 0},
        "sigma": {"feature_hansenlaw_fusion": 0, "rbasex": 0, "tie": 0},
        "beta": {"feature_hansenlaw_fusion": 0, "rbasex": 0, "tie": 0},
        "br": {"feature_hansenlaw_fusion": 0, "rbasex": 0, "tie": 0},
        "time": {"feature_hansenlaw_fusion": 0, "rbasex": 0, "tie": 0},
    }
    scenarios = sorted({str(r["scenario"]) for r in rows})
    for sc in scenarios:
        a = idx.get((sc, "feature_hansenlaw_fusion"))
        b = idx.get((sc, "rbasex"))
        if a is None or b is None:
            continue
        for k in wins.keys():
            va = float(a[k])
            vb = float(b[k])
            if not np.isfinite(va) or not np.isfinite(vb):
                continue
            if abs(va - vb) <= 1e-12:
                wins[k]["tie"] += 1
            elif va < vb:
                wins[k]["feature_hansenlaw_fusion"] += 1
            else:
                wins[k]["rbasex"] += 1
    return wins


def _repeat_stability(rows: List[Dict[str, Any]], algorithm: str, key: str) -> np.ndarray:
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        if r["algorithm"] != algorithm:
            continue
        gk = (str(r["design"]), str(r["factor"]), str(r["level"]))
        v = float(r[key])
        if np.isfinite(v):
            grouped[gk].append(v)
    stds: List[float] = []
    for vals in grouped.values():
        if len(vals) >= 2:
            stds.append(float(np.std(np.asarray(vals, dtype=float))))
    return np.asarray(stds, dtype=float)


def write_summary(
    rows: List[Dict[str, Any]],
    output_path: Path,
    *,
    design: str,
    repeats: int,
) -> None:
    metrics = [("npk", "Npk_err%"), ("r0", "r0_err%"), ("sigma", "sigma_err%"), ("beta", "beta_abs_err"), ("br", "BR_err%"), ("time", "Time(s)")]
    lines: List[str] = []
    lines.append("Feature Hansen-Law vs rBasex forward-driven sweep")
    lines.append(f"Design mode: {design}, repeats={repeats}")
    lines.append("Data mode: xy_ideal, add_noise=False, T_beam=0 K")
    lines.append("")
    lines.append(f"Total scenario-evaluations: {len(rows)}")
    lines.append(f"Total scenarios: {len(set(str(r['scenario']) for r in rows))}")
    lines.append("")
    for alg in ["feature_hansenlaw_fusion", "rbasex"]:
        lines.append(f"[{alg}]")
        for key, label in metrics:
            vals = _group_vals(rows, alg, key)
            if vals.size == 0:
                lines.append(f"- {label}: n/a")
                continue
            lines.append(
                f"- {label}: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, "
                f"p90={np.percentile(vals, 90):.4f}, p99={np.percentile(vals, 99):.4f}"
            )
        lines.append("- Repeat stability (std across repeats per same factor-level):")
        for key, label in metrics:
            stds = _repeat_stability(rows, alg, key)
            if stds.size == 0:
                lines.append(f"  - {label}: n/a")
                continue
            lines.append(
                f"  - {label}: mean_std={np.mean(stds):.4f}, median_std={np.median(stds):.4f}, "
                f"p90_std={np.percentile(stds, 90):.4f}"
            )
        lines.append("")

    wins = _compute_wins(rows)
    lines.append("Per-scenario lower-is-better win counts:")
    for key, label in metrics:
        w = wins[key]
        total = w["feature_hansenlaw_fusion"] + w["rbasex"] + w["tie"]
        lines.append(
            f"- {label}: feature={w['feature_hansenlaw_fusion']}, "
            f"rbasex={w['rbasex']}, tie={w['tie']} (total={total})"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_overall_means(rows: List[Dict[str, Any]], out_path: Path) -> None:
    metrics = [("npk", "Npk_err%"), ("r0", "r0_err%"), ("sigma", "sigma_err%"), ("beta", "beta_abs_err"), ("br", "BR_err%"), ("time", "Time(s)")]
    algs = ["feature_hansenlaw_fusion", "rbasex"]
    means = {alg: _mean_dict(rows, alg) for alg in algs}

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, (key, label) in zip(axes.flatten(), metrics):
        vals = [means[alg][key] for alg in algs]
        ax.bar([0, 1], vals, color=["#1f77b4", "#ff7f0e"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["feature", "rbasex"])
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Overall mean metric comparison", fontsize=12)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(output_dir: Path, repeats: int = 1, design: str = "both") -> None:
    sweep_points, _factor_spec, base = build_sweep_plan(repeats=repeats, design=design)
    print(f"Sweep scenarios: {len(sweep_points)}")
    rows = run_compare_sweep(
        sweep_points,
        feature_kwargs={"ideal_mode": True},
        rbasex_kwargs={"pixel_size_mm": float(base["pixel_size_mm"]), "n_pixels": 512},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, output_dir / "comparison_summary.txt", design=design, repeats=repeats)
    plot_overall_means(rows, output_dir / "overall_mean_comparison.png")
    print(f"Saved outputs to: {output_dir}")
    print(f"  - {output_dir / 'comparison_summary.txt'}")
    print(f"  - {output_dir / 'overall_mean_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare feature_hansenlaw_fusion and rbasex under identical forward sweep design")
    parser.add_argument("--output-dir", type=Path, default=Path("feature-vs-rbasex-forward-sweep"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--design", type=str, choices=["ofat", "pairwise", "both"], default="both")
    args = parser.parse_args()

    main(output_dir=args.output_dir, repeats=max(1, int(args.repeats)), design=str(args.design))
