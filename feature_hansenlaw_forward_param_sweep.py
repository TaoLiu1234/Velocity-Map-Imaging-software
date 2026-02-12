from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from abel_benchmark_framework import BenchmarkMetrics, BenchmarkRunner, GroundTruthPeak, TestScenario


@dataclass
class SweepPoint:
    factor: str
    level_name: str
    numeric_value: float
    scenario: TestScenario
    replicate: int
    design: str
    pair_factor_a: str = ""
    pair_factor_b: str = ""
    pair_level_a: str = ""
    pair_level_b: str = ""


def _positions_from_center(n_peaks: int, center_r: float, separation: float) -> List[float]:
    if n_peaks <= 1:
        return [float(center_r)]
    start = center_r - separation * (n_peaks - 1) / 2.0
    return [float(start + i * separation) for i in range(n_peaks)]


def _beta_list(n_peaks: int, beta_value: float) -> List[float]:
    return [float(beta_value)] * n_peaks


def _br_list(n_peaks: int, dominant_ratio: float) -> List[float]:
    if n_peaks <= 1:
        return [1.0]
    return [float(dominant_ratio)] + [1.0] * (n_peaks - 1)


def _make_scenario(
    *,
    name: str,
    n_peaks: int,
    center_r_mm: float,
    separation_mm: float,
    sigma_mm: float,
    beta_value: float,
    dominant_ratio: float,
    n_events: int,
    seed: int,
    psf_sigma_mm: float,
    dld_resolution_mm: float,
    pixel_size_mm: float,
    vmi_k: float,
    forward_temperature_k: float,
) -> TestScenario:
    r_list = _positions_from_center(n_peaks=n_peaks, center_r=center_r_mm, separation=separation_mm)
    beta_vals = _beta_list(n_peaks=n_peaks, beta_value=beta_value)
    amp_list = _br_list(n_peaks=n_peaks, dominant_ratio=dominant_ratio)
    peaks = [
        GroundTruthPeak(r0=float(r_list[i]), sigma=float(sigma_mm), beta=float(beta_vals[i]), amp=float(amp_list[i]))
        for i in range(n_peaks)
    ]
    return TestScenario(
        name=name,
        peaks=peaks,
        n_events=int(n_events),
        psf_sigma=float(psf_sigma_mm),
        dld_resolution=float(dld_resolution_mm),
        pixel_size=float(pixel_size_mm),
        vmi_k=float(vmi_k),
        forward_temperature_k=float(max(forward_temperature_k, 0.0)),
        add_noise=False,
        forward_output_mode="xy_ideal",
        seed=int(seed),
    )


def build_factor_spec() -> Dict[str, Dict[str, Any]]:
    return {
        "n_peaks": {"levels": [1, 2, 3, 4], "unit": "count", "interval": "step=1", "description": "number of rings"},
        "center_r_mm": {"levels": [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], "unit": "mm", "interval": "step=2.0 mm", "description": "overall radius location"},
        "separation_mm": {"levels": [1.2, 2.0, 3.0, 4.5, 6.0], "unit": "mm", "interval": "non-uniform dense points", "description": "distance between adjacent rings"},
        "sigma_mm": {"levels": [0.15, 0.25, 0.35, 0.45, 0.55, 0.70], "unit": "mm", "interval": "approx step=0.10 mm", "description": "radial width"},
        "beta_value": {"levels": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], "unit": "dimensionless", "interval": "step=0.25", "description": "anisotropy beta"},
        "dominant_ratio": {"levels": [1.0, 1.5, 2.0, 3.0, 5.0, 10.0], "unit": "ratio", "interval": "non-uniform dense points", "description": "peak-1 BR ratio vs others"},
        "n_events": {"levels": [5_000, 10_000, 20_000, 50_000, 100_000, 200_000], "unit": "events", "interval": "roughly log-spaced", "description": "detected event count"},
    }


def build_base_params() -> Dict[str, Any]:
    return {
        "n_peaks": 2,
        "center_r_mm": 10.0,
        "separation_mm": 3.0,
        "sigma_mm": 0.30,
        "beta_value": 0.5,
        "dominant_ratio": 1.0,
        "n_events": 50_000,
        "psf_sigma_mm": 0.0,
        "dld_resolution_mm": 0.0,
        "pixel_size_mm": 0.05,
        "vmi_k": 0.01,
        "forward_temperature_k": 0.0,
    }


def _normalize_params(params: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if int(out["n_peaks"]) <= 1:
        out["separation_mm"] = float(base["separation_mm"])
        out["dominant_ratio"] = 1.0
    return out


def build_sweep_plan(repeats: int = 2, design: str = "both") -> Tuple[List[SweepPoint], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    factor_spec = build_factor_spec()
    base = build_base_params()
    factor_names = list(factor_spec.keys())
    sweep_points: List[SweepPoint] = []
    seed_counter = 500_000
    repeats = max(1, int(repeats))

    include_ofat = design in ("ofat", "both")
    include_pairwise = design in ("pairwise", "both")

    if include_ofat:
        for factor, spec in factor_spec.items():
            for level in spec["levels"]:
                for rep in range(1, repeats + 1):
                    params = dict(base)
                    params[factor] = level
                    params = _normalize_params(params, base)
                    scenario = _make_scenario(
                        name=f"ofat_{factor}_{level}_rep{rep}",
                        n_peaks=int(params["n_peaks"]),
                        center_r_mm=float(params["center_r_mm"]),
                        separation_mm=float(params["separation_mm"]),
                        sigma_mm=float(params["sigma_mm"]),
                        beta_value=float(params["beta_value"]),
                        dominant_ratio=float(params["dominant_ratio"]),
                        n_events=int(params["n_events"]),
                        seed=seed_counter,
                        psf_sigma_mm=float(params["psf_sigma_mm"]),
                        dld_resolution_mm=float(params["dld_resolution_mm"]),
                        pixel_size_mm=float(params["pixel_size_mm"]),
                        vmi_k=float(params["vmi_k"]),
                        forward_temperature_k=float(params["forward_temperature_k"]),
                    )
                    sweep_points.append(
                        SweepPoint(
                            factor=factor,
                            level_name=f"{level}",
                            numeric_value=float(level),
                            scenario=scenario,
                            replicate=rep,
                            design="ofat",
                        )
                    )
                    seed_counter += 1

    if include_pairwise:
        for factor_a, factor_b in combinations(factor_names, 2):
            levels_a = factor_spec[factor_a]["levels"]
            levels_b = factor_spec[factor_b]["levels"]
            for level_a in levels_a:
                for level_b in levels_b:
                    for rep in range(1, repeats + 1):
                        params = dict(base)
                        params[factor_a] = level_a
                        params[factor_b] = level_b
                        params = _normalize_params(params, base)
                        scenario = _make_scenario(
                            name=f"pair_{factor_a}_{level_a}_{factor_b}_{level_b}_rep{rep}",
                            n_peaks=int(params["n_peaks"]),
                            center_r_mm=float(params["center_r_mm"]),
                            separation_mm=float(params["separation_mm"]),
                            sigma_mm=float(params["sigma_mm"]),
                            beta_value=float(params["beta_value"]),
                            dominant_ratio=float(params["dominant_ratio"]),
                            n_events=int(params["n_events"]),
                            seed=seed_counter,
                            psf_sigma_mm=float(params["psf_sigma_mm"]),
                            dld_resolution_mm=float(params["dld_resolution_mm"]),
                            pixel_size_mm=float(params["pixel_size_mm"]),
                            vmi_k=float(params["vmi_k"]),
                            forward_temperature_k=float(params["forward_temperature_k"]),
                        )
                        sweep_points.append(
                            SweepPoint(
                                factor=f"{factor_a}__{factor_b}",
                                level_name=f"{factor_a}={level_a}|{factor_b}={level_b}",
                                numeric_value=float("nan"),
                                scenario=scenario,
                                replicate=rep,
                                design="pairwise",
                                pair_factor_a=factor_a,
                                pair_factor_b=factor_b,
                                pair_level_a=str(level_a),
                                pair_level_b=str(level_b),
                            )
                        )
                        seed_counter += 1
    return sweep_points, factor_spec, base


def _metrics_to_row(point: SweepPoint, metrics: BenchmarkMetrics) -> Dict[str, Any]:
    return {
        "scenario": point.scenario.name,
        "design": point.design,
        "factor": point.factor,
        "level": point.level_name,
        "replicate": int(point.replicate),
        "pair_factor_a": point.pair_factor_a,
        "pair_factor_b": point.pair_factor_b,
        "pair_level_a": point.pair_level_a,
        "pair_level_b": point.pair_level_b,
        "npk": float(metrics.peak_count_error_pct),
        "r0": float(metrics.r0_error_pct_mean),
        "sigma": float(metrics.sigma_error_pct_mean),
        "beta": float(metrics.beta_error_abs_mean),
        "br": float(metrics.br_error_pct_mean),
        "time": float(metrics.computation_time),
    }


def run_sweep(sweep_points: List[SweepPoint], algorithm_kwargs: Optional[Dict[str, Any]] = None, progress_every: int = 150) -> List[Dict[str, Any]]:
    alg_kwargs = {"ideal_mode": True}
    if algorithm_kwargs:
        alg_kwargs.update(dict(algorithm_kwargs))
    runner = BenchmarkRunner(algorithms=["feature_hansenlaw_fusion"], algorithm_kwargs={"feature_hansenlaw_fusion": alg_kwargs})
    rows: List[Dict[str, Any]] = []
    total = len(sweep_points)
    for idx, point in enumerate(sweep_points, start=1):
        result = runner.run_scenario(point.scenario, algorithm_name="feature_hansenlaw_fusion")
        metrics = result.get("feature_hansenlaw_fusion")
        if metrics is None:
            continue
        rows.append(_metrics_to_row(point, metrics))
        if (idx % max(1, progress_every) == 0) or idx == total:
            print(
                f"[{idx:5d}/{total}] {point.design:<8} {point.factor} "
                f"Npk={metrics.peak_count_error_pct:.2f}% r0={metrics.r0_error_pct_mean:.2f}% "
                f"sigma={metrics.sigma_error_pct_mean:.2f}% beta={metrics.beta_error_abs_mean:.3f}"
            )
    return rows


def _group_vals(rows: List[Dict[str, Any]], mask_fn, key: str) -> np.ndarray:
    vals = [float(r[key]) for r in rows if mask_fn(r) and np.isfinite(float(r[key]))]
    return np.asarray(vals, dtype=float)


def _group_mean(rows: List[Dict[str, Any]], mask_fn, key: str) -> float:
    vals = _group_vals(rows, mask_fn, key)
    return float(np.mean(vals)) if vals.size else float("nan")


def _group_std(rows: List[Dict[str, Any]], mask_fn, key: str) -> float:
    vals = _group_vals(rows, mask_fn, key)
    return float(np.std(vals)) if vals.size else float("nan")


def _group_median(rows: List[Dict[str, Any]], mask_fn, key: str) -> float:
    vals = _group_vals(rows, mask_fn, key)
    return float(np.median(vals)) if vals.size else float("nan")


def _group_p90(rows: List[Dict[str, Any]], mask_fn, key: str) -> float:
    vals = _group_vals(rows, mask_fn, key)
    return float(np.percentile(vals, 90)) if vals.size else float("nan")


def _group_p99(rows: List[Dict[str, Any]], mask_fn, key: str) -> float:
    vals = _group_vals(rows, mask_fn, key)
    return float(np.percentile(vals, 99)) if vals.size else float("nan")


def _sorted_levels(levels: List[str]) -> List[str]:
    try:
        return sorted(levels, key=lambda x: float(x))
    except Exception:
        return sorted(levels)


def plot_factor(rows: List[Dict[str, Any]], factor: str, out_path: Path) -> None:
    levels = _sorted_levels(sorted({str(r["level"]) for r in rows if r["design"] == "ofat" and r["factor"] == factor}))
    if not levels:
        return
    metrics = [("r0", "r0_err%"), ("sigma", "sigma_err%"), ("beta", "beta_abs_err"), ("br", "BR_err%")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flatten(), metrics):
        mean_vals = []
        std_vals = []
        for lv in levels:
            mean_vals.append(_group_mean(rows, lambda r, lv=lv: r["design"] == "ofat" and r["factor"] == factor and r["level"] == lv, metric))
            std_vals.append(_group_std(rows, lambda r, lv=lv: r["design"] == "ofat" and r["factor"] == factor and r["level"] == lv, metric))
        xx = np.arange(len(levels))
        yy = np.asarray(mean_vals, dtype=float)
        ss = np.asarray(std_vals, dtype=float)
        ax.plot(xx, yy, marker="o", linewidth=1.5)
        low = np.maximum(yy - ss, 0.0)
        high = yy + ss
        ax.fill_between(xx, low, high, alpha=0.2)
        ax.set_title(label)
        ax.set_xticks(xx)
        ax.set_xticklabels(levels, rotation=30, ha="right")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"OFAT impact: {factor}", fontsize=12)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pairwise_heatmaps(
    rows: List[Dict[str, Any]],
    metric: str,
    output_dir: Path,
    factor_spec: Dict[str, Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = {"npk": "Npk_err%", "r0": "r0_err%", "sigma": "sigma_err%", "beta": "beta_abs_err", "br": "BR_err%", "time": "Time(s)"}[metric]
    pair_keys = sorted({(r["pair_factor_a"], r["pair_factor_b"]) for r in rows if r["design"] == "pairwise"})
    for fa, fb in pair_keys:
        levels_a = [str(v) for v in factor_spec[fa]["levels"]]
        levels_b = [str(v) for v in factor_spec[fb]["levels"]]
        mat = np.full((len(levels_a), len(levels_b)), np.nan, dtype=float)
        for ia, la in enumerate(levels_a):
            for ib, lb in enumerate(levels_b):
                val = _group_mean(
                    rows,
                    lambda r, fa=fa, fb=fb, la=la, lb=lb: r["design"] == "pairwise" and r["pair_factor_a"] == fa and r["pair_factor_b"] == fb and r["pair_level_a"] == la and r["pair_level_b"] == lb,
                    metric,
                )
                mat[ia, ib] = val
        fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(f"{fa} x {fb} ({label})")
        ax.set_xticks(np.arange(len(levels_b)))
        ax.set_yticks(np.arange(len(levels_a)))
        ax.set_xticklabels(levels_b, rotation=35, ha="right")
        ax.set_yticklabels(levels_a)
        ax.set_xlabel(fb)
        ax.set_ylabel(fa)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(label)
        fig.savefig(output_dir / f"{fa}__{fb}.png", dpi=160)
        plt.close(fig)


def write_summary(
    rows: List[Dict[str, Any]],
    factor_spec: Dict[str, Dict[str, Any]],
    base: Dict[str, Any],
    output_path: Path,
    design: str,
    repeats: int,
    followup_enabled: bool = False,
    line_shape: str = "gaussian",
    uncertainty_enabled: bool = False,
) -> None:
    metrics = [("npk", "Npk_err%"), ("r0", "r0_err%"), ("sigma", "sigma_err%"), ("beta", "beta_abs_err"), ("br", "BR_err%"), ("time", "Time(s)")]
    lines: List[str] = []
    lines.append("Feature Hansen-Law forward-driven comprehensive sweep")
    lines.append("Data mode: xy_ideal, add_noise=False, psf_sigma_mm=0, dld_resolution_mm=0")
    lines.append("Forward thermal setting: T_beam=0 K (no thermal motion broadening)")
    lines.append(
        "Reconstructor mode: "
        f"feature_hansenlaw_fusion(ideal_mode=True, run_followup_stages={bool(followup_enabled)}, "
        f"enable_uncertainty_analysis={bool(uncertainty_enabled)}, line_shape={line_shape})"
    )
    lines.append(f"Design mode: {design}, repeats={repeats}")
    lines.append("")
    lines.append(f"Total scenarios: {len(rows)}")
    lines.append(f"- OFAT scenarios: {sum(1 for r in rows if r['design'] == 'ofat')}")
    lines.append(f"- Pairwise scenarios: {sum(1 for r in rows if r['design'] == 'pairwise')}")
    lines.append("")
    lines.append("Algorithm flow (feature_hansenlaw_fusion):")
    lines.append("- XY points -> radial m0 profile histogram -> Hansen-Law inverse (rho0)")
    lines.append("- Multi-scale peak detection (prominence + curvature + radius weighting)")
    lines.append("- Local Gaussian refinement for each peak (r0, sigma, amp)")
    lines.append("- Beta estimation fusion (moment + FFT + curve fit + WLS)")
    lines.append("- Area/BR estimation via probabilistic responsibility assignment")
    if followup_enabled:
        lines.append("- Stage 3/4/5 followup enabled")
    lines.append("")
    lines.append("Design (range / points / interval):")
    for factor, spec in factor_spec.items():
        levels = spec["levels"]
        lines.append(
            f"- {factor}: points={len(levels)}, levels={levels}, unit={spec['unit']}, "
            f"interval={spec['interval']}, desc={spec['description']}"
        )
    lines.append("")
    lines.append("Fixed baseline params:")
    for k in ["psf_sigma_mm", "dld_resolution_mm", "pixel_size_mm", "vmi_k", "forward_temperature_k"]:
        lines.append(f"- {k}={base[k]}")
    lines.append("")
    lines.append("Overall performance:")
    for key, label in metrics:
        vals = _group_vals(rows, lambda _r: True, key)
        if vals.size:
            lines.append(
                f"- {label}: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, "
                f"p90={np.percentile(vals, 90):.4f}, p99={np.percentile(vals, 99):.4f}"
            )
        else:
            lines.append(f"- {label}: n/a")
    lines.append("")
    lines.append("OFAT per-factor level stats (mean / median / p90):")
    for factor, spec in factor_spec.items():
        lines.append(f"- {factor}:")
        for lv in spec["levels"]:
            lv_str = str(lv)
            r0 = (_group_mean(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "r0"),
                  _group_median(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "r0"),
                  _group_p90(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "r0"))
            sigma = (_group_mean(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "sigma"),
                     _group_median(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "sigma"),
                     _group_p90(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "sigma"))
            beta = (_group_mean(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "beta"),
                    _group_median(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "beta"),
                    _group_p90(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "beta"))
            br = (_group_mean(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "br"),
                  _group_median(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "br"),
                  _group_p90(rows, lambda r, f=factor, l=lv_str: r["design"] == "ofat" and r["factor"] == f and r["level"] == l, "br"))
            lines.append(
                f"  - level={lv_str}, r0%=({r0[0]:.4f}/{r0[1]:.4f}/{r0[2]:.4f}), "
                f"sigma%=({sigma[0]:.4f}/{sigma[1]:.4f}/{sigma[2]:.4f}), "
                f"beta_abs=({beta[0]:.4f}/{beta[1]:.4f}/{beta[2]:.4f}), "
                f"BR%=({br[0]:.4f}/{br[1]:.4f}/{br[2]:.4f})"
            )

    lines.append("")
    lines.append("Pairwise interaction sensitivity (delta=max-min mean):")
    for key, label in [("r0", "r0_err%"), ("sigma", "sigma_err%"), ("beta", "beta_abs_err"), ("br", "BR_err%")]:
        lines.append(f"- {label}:")
        pair_deltas: List[Tuple[str, float]] = []
        pair_keys = sorted({(r["pair_factor_a"], r["pair_factor_b"]) for r in rows if r["design"] == "pairwise"})
        for fa, fb in pair_keys:
            vals: List[float] = []
            levels_a = [str(v) for v in factor_spec[fa]["levels"]]
            levels_b = [str(v) for v in factor_spec[fb]["levels"]]
            for la in levels_a:
                for lb in levels_b:
                    v = _group_mean(
                        rows,
                        lambda r, fa=fa, fb=fb, la=la, lb=lb: r["design"] == "pairwise" and r["pair_factor_a"] == fa and r["pair_factor_b"] == fb and r["pair_level_a"] == la and r["pair_level_b"] == lb,
                        key,
                    )
                    if np.isfinite(v):
                        vals.append(float(v))
            if vals:
                pair_deltas.append((f"{fa} x {fb}", float(max(vals) - min(vals))))
        pair_deltas.sort(key=lambda x: x[1], reverse=True)
        for nm, dv in pair_deltas[:8]:
            lines.append(f"  - {nm}: delta={dv:.4f}")

    strict = [
        r
        for r in rows
        if (r["npk"] == 0.0 and r["r0"] <= 5.0 and r["beta"] <= 0.05 and r["sigma"] <= 10.0 and r["br"] <= 15.0)
    ]
    lines.append("")
    lines.append("Strict target hit count (Npk=0, r0<=5%, beta<=0.05, sigma<=10%, BR<=15%):")
    lines.append(f"- strict count={len(strict)}/{len(rows)} ({100.0 * len(strict) / max(1, len(rows)):.2f}%)")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(
    output_dir: Path,
    repeats: int = 2,
    design: str = "both",
    pairwise_metrics: Optional[List[str]] = None,
    run_followup_stages: bool = False,
    line_shape: str = "gaussian",
    enable_uncertainty_analysis: bool = False,
) -> None:
    sweep_points, factor_spec, base = build_sweep_plan(repeats=repeats, design=design)
    print(f"Sweep scenarios: {len(sweep_points)}")
    rows = run_sweep(
        sweep_points,
        algorithm_kwargs={
            "run_followup_stages": bool(run_followup_stages),
            "line_shape": str(line_shape),
            "enable_uncertainty_analysis": bool(enable_uncertainty_analysis),
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for factor in factor_spec.keys():
        plot_factor(rows, factor, output_dir / f"impact_ofat_{factor}.png")

    metrics = pairwise_metrics or ["r0", "sigma", "beta", "br"]
    for metric in metrics:
        plot_pairwise_heatmaps(rows, metric, output_dir / f"pairwise_heatmaps_{metric}", factor_spec)

    write_summary(
        rows=rows,
        factor_spec=factor_spec,
        base=base,
        output_path=output_dir / "forward_sweep_summary.txt",
        design=design,
        repeats=repeats,
        followup_enabled=bool(run_followup_stages),
        line_shape=str(line_shape),
        uncertainty_enabled=bool(enable_uncertainty_analysis),
    )

    print(f"Saved outputs to: {output_dir}")
    for factor in factor_spec.keys():
        print(f"  - {output_dir / f'impact_ofat_{factor}.png'}")
    for metric in metrics:
        print(f"  - {output_dir / f'pairwise_heatmaps_{metric}'}")
    print(f"  - {output_dir / 'forward_sweep_summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Hansen-Law forward-driven parameter sweep")
    parser.add_argument("--output-dir", type=Path, default=Path("feature-hansenlaw-forward-sweep"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--design", type=str, choices=["ofat", "pairwise", "both"], default="both")
    parser.add_argument("--pairwise-metrics", type=str, default="r0,sigma,beta,br")
    parser.add_argument("--run-followup-stages", action="store_true")
    parser.add_argument("--line-shape", type=str, default="gaussian")
    parser.add_argument("--enable-uncertainty-analysis", action="store_true")
    args = parser.parse_args()

    metrics = [x.strip() for x in args.pairwise_metrics.split(",") if x.strip() in {"npk", "r0", "sigma", "beta", "br", "time"}]
    if not metrics:
        metrics = ["r0", "sigma", "beta", "br"]

    main(
        output_dir=args.output_dir,
        repeats=int(args.repeats),
        design=str(args.design),
        pairwise_metrics=metrics,
        run_followup_stages=bool(args.run_followup_stages),
        line_shape=str(args.line_shape),
        enable_uncertainty_analysis=bool(args.enable_uncertainty_analysis),
    )

