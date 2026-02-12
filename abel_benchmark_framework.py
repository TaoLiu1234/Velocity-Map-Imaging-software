from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import numpy as np

from Abel_forward_simulation import Config, ELECTRON_MASS_AMU, run_simulation


@dataclass
class GroundTruthPeak:
    r0: float
    sigma: float
    beta: float
    amp: float


@dataclass
class TestScenario:
    name: str
    peaks: List[GroundTruthPeak]
    n_events: int
    psf_sigma: float = 0.0
    dld_resolution: float = 0.0
    pixel_size: float = 0.05
    vmi_k: float = 0.01
    forward_temperature_k: float = 0.0
    add_noise: bool = False
    forward_output_mode: str = "xy_ideal"
    seed: int = 0


@dataclass
class ReconstructionResult:
    algorithm_name: str
    peaks: List[Dict[str, float]]
    computation_time: float
    success: bool
    error_message: str = ""


@dataclass
class BenchmarkMetrics:
    algorithm_name: str
    scenario_name: str
    n_peaks_detected: int
    n_peaks_true: int
    peak_count_abs_error: float
    peak_count_error_pct: float
    r0_error_mean: float = 0.0
    r0_error_std: float = 0.0
    r0_error_pct_mean: float = 0.0
    sigma_error_mean: float = 0.0
    sigma_error_std: float = 0.0
    sigma_error_pct_mean: float = 0.0
    beta_error_mean: float = 0.0
    beta_error_std: float = 0.0
    beta_error_abs_mean: float = 0.0
    br_error_mean: float = 0.0
    br_error_std: float = 0.0
    br_error_pct_mean: float = 0.0
    computation_time: float = 0.0
    success_rate: float = 0.0


class TestCaseGenerator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _radial_sigma_mm_to_energy_sigma_ev(sigma_r_mm: float, r_ref_mm: float, vmi_k: float) -> float:
        if sigma_r_mm <= 0 or r_ref_mm <= 0 or vmi_k <= 0:
            return 0.0
        mass_kg = ELECTRON_MASS_AMU * 1.66053906660e-27
        charge = 1.602176634e-19
        dE_dr = (mass_kg * r_ref_mm) / (charge * (vmi_k**2))
        return float(max(dE_dr * sigma_r_mm, 0.0))

    def generate_scenario(self, scenario: TestScenario) -> np.ndarray:
        if not scenario.peaks:
            return np.zeros((0, 2), dtype=float)

        amps = np.array([max(float(p.amp), 1e-12) for p in scenario.peaks], dtype=float)
        amp_sum = float(np.sum(amps))
        br = (amps / amp_sum).tolist() if amp_sum > 0 else [1.0 / len(scenario.peaks)] * len(scenario.peaks)

        energies: List[float] = []
        betas: List[float] = []
        sigma_weighted = 0.0
        r_weighted = 0.0
        for i, peak in enumerate(scenario.peaks):
            rr = float(max(peak.r0, 1e-6))
            ee = 0.5 * (ELECTRON_MASS_AMU * 1.66053906660e-27) * (rr / float(max(scenario.vmi_k, 1e-9))) ** 2 / 1.602176634e-19
            energies.append(float(max(ee, 1e-9)))
            betas.append(float(np.clip(peak.beta, -1.0, 2.0)))
            sigma_weighted += float(br[i]) * float(max(peak.sigma, 1e-6))
            r_weighted += float(br[i]) * rr

        sigma_e = self._radial_sigma_mm_to_energy_sigma_ev(
            sigma_r_mm=float(max(sigma_weighted, 0.0)),
            r_ref_mm=float(max(r_weighted, 1e-6)),
            vmi_k=float(max(scenario.vmi_k, 1e-9)),
        )

        config = Config(
            mass=ELECTRON_MASS_AMU,
            E_centers=energies,
            Betas=betas,
            branching_ratios=br,
            N_events=int(max(scenario.n_events, 1)),
            vmi_k=float(max(scenario.vmi_k, 1e-9)),
            sigma_laser=float(max(sigma_e, 0.0)),
            T_beam=float(max(scenario.forward_temperature_k, 0.0)),
            tau_lifetimes=0.0,
            photon_energy=0.0,
            target_mass=28.0,
            vol_sigma=(0.0, 0.0, 0.0),
            polarization_vec=np.array([0.0, 1.0, 0.0], dtype=float),
            img_res=512,
            pixel_size=float(max(scenario.pixel_size, 1e-6)),
            psf_fwhm=float(max(scenario.psf_sigma, 0.0) * (2.0 * np.sqrt(2.0 * np.log(2.0)))),
            dld_resolution=float(max(scenario.dld_resolution, 0.0)),
            readout_sigma=0.0,
            readout_offset=0.0,
            mcp_dark_rate=0.0,
            residual_gas_rate=0.0,
            residual_gas_sigma=1.0,
        )

        np.random.seed(int(scenario.seed))
        out, _ = run_simulation(
            config=config,
            add_noise=bool(scenario.add_noise),
            return_particles=False,
            output_mode=str(getattr(scenario, "forward_output_mode", "xy_ideal")),
        )
        arr = np.asarray(out, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.zeros((0, 2), dtype=float)
        return arr[:, :2]


class AlgorithmWrapper:
    def __init__(self, name: str, reconstruct_func: Callable, default_kwargs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.reconstruct_func = reconstruct_func
        self.default_kwargs = default_kwargs or {}

    def run(self, xy_data: np.ndarray, n_peaks: Optional[int] = None, **kwargs: Any) -> ReconstructionResult:
        start_time = time.time()
        try:
            all_kwargs = {**self.default_kwargs, **kwargs}
            peaks = self.reconstruct_func(xy_data, n_peaks, **all_kwargs)
            standardized = self._standardize_peaks(peaks)
            return ReconstructionResult(
                algorithm_name=self.name,
                peaks=standardized,
                computation_time=float(time.time() - start_time),
                success=True,
            )
        except Exception as exc:
            return ReconstructionResult(
                algorithm_name=self.name,
                peaks=[],
                computation_time=float(time.time() - start_time),
                success=False,
                error_message=str(exc),
            )

    @staticmethod
    def _standardize_peaks(peaks: Any) -> List[Dict[str, float]]:
        if not peaks:
            return []
        out: List[Dict[str, float]] = []
        for peak in peaks:
            if isinstance(peak, dict):
                out.append(
                    {
                        "r0": float(peak.get("r0", peak.get("r", 0.0))),
                        "sigma": float(peak.get("sigma", 0.0)),
                        "beta": float(peak.get("beta", 0.0)),
                        "amp": float(peak.get("amp", peak.get("amplitude", 0.0))),
                    }
                )
            elif hasattr(peak, "__dict__"):
                out.append(
                    {
                        "r0": float(getattr(peak, "r0", getattr(peak, "r", 0.0))),
                        "sigma": float(getattr(peak, "sigma", 0.0)),
                        "beta": float(getattr(peak, "beta", 0.0)),
                        "amp": float(getattr(peak, "amp", getattr(peak, "amplitude", 0.0))),
                    }
                )
            elif isinstance(peak, (list, tuple)):
                out.append(
                    {
                        "r0": float(peak[0]) if len(peak) > 0 else 0.0,
                        "sigma": float(peak[1]) if len(peak) > 1 else 0.0,
                        "beta": float(peak[2]) if len(peak) > 2 else 0.0,
                        "amp": float(peak[3]) if len(peak) > 3 else 0.0,
                    }
                )
        return out


class AlgorithmRegistry:
    def __init__(self) -> None:
        self.algorithms: Dict[str, AlgorithmWrapper] = {}
        self._register_algorithms()

    def _register_algorithms(self) -> None:
        try:
            from feature_extract_hansenlaw_p2 import fit_xy_feature_hansenlaw_p2

            self.algorithms["feature_hansenlaw_fusion"] = AlgorithmWrapper("feature_hansenlaw_fusion", fit_xy_feature_hansenlaw_p2)
        except Exception:
            pass
        try:
            from rbasex_benchmark_adapter import fit_xy_rbasex_benchmark

            self.algorithms["rbasex"] = AlgorithmWrapper("rbasex", fit_xy_rbasex_benchmark)
        except Exception:
            pass

    def get_algorithm(self, name: str) -> Optional[AlgorithmWrapper]:
        return self.algorithms.get(name)

    def list_algorithms(self) -> List[str]:
        return list(self.algorithms.keys())


class MetricsCalculator:
    def __init__(self, max_distance: float = np.inf):
        self.max_distance = max_distance

    def calculate_metrics(self, result: ReconstructionResult, scenario: TestScenario) -> BenchmarkMetrics:
        true_peaks = scenario.peaks
        n_true = len(true_peaks)
        n_detected = len(result.peaks) if result.success else 0
        peak_count_abs_error = abs(n_detected - n_true)
        peak_count_error_pct = (peak_count_abs_error / n_true * 100.0) if n_true > 0 else 0.0
        if not result.success:
            return BenchmarkMetrics(
                algorithm_name=result.algorithm_name,
                scenario_name=scenario.name,
                n_peaks_detected=n_detected,
                n_peaks_true=n_true,
                peak_count_abs_error=peak_count_abs_error,
                peak_count_error_pct=peak_count_error_pct,
                computation_time=float(result.computation_time),
                success_rate=0.0,
            )

        est_peaks = self._harmonize_peak_units(result.peaks, scenario)
        matches = self._match_peaks(est_peaks, true_peaks)
        if not matches:
            return BenchmarkMetrics(
                algorithm_name=result.algorithm_name,
                scenario_name=scenario.name,
                n_peaks_detected=n_detected,
                n_peaks_true=n_true,
                peak_count_abs_error=peak_count_abs_error,
                peak_count_error_pct=peak_count_error_pct,
                computation_time=float(result.computation_time),
                success_rate=0.0,
            )

        r0_errors: List[float] = []
        sigma_errors: List[float] = []
        beta_errors: List[float] = []
        br_errors: List[float] = []
        true_br_vals: List[float] = []

        true_total_amp = float(sum(max(float(p.amp), 0.0) for p in true_peaks))
        est_total_amp = float(sum(max(float(p.get("amp", 0.0)), 0.0) for p in est_peaks))
        for est_idx, true_idx, _ in matches:
            est = est_peaks[est_idx]
            true = true_peaks[true_idx]
            r0_errors.append(float(est["r0"] - true.r0))
            sigma_errors.append(float(est["sigma"] - true.sigma))
            beta_errors.append(float(est["beta"] - true.beta))
            true_br = (float(true.amp) / true_total_amp) if true_total_amp > 0 else 0.0
            est_br = (float(est.get("amp", 0.0)) / est_total_amp) if est_total_amp > 0 else 0.0
            br_errors.append(float(est_br - true_br))
            true_br_vals.append(float(true_br))

        r0_pct = [abs(e / t.r0) * 100.0 for e, (_, ti, _) in zip(r0_errors, matches) for t in [true_peaks[ti]] if t.r0 > 0]
        sigma_pct = [abs(e / t.sigma) * 100.0 for e, (_, ti, _) in zip(sigma_errors, matches) for t in [true_peaks[ti]] if t.sigma > 0]
        br_pct = [abs(e / max(t, 1e-12)) * 100.0 for e, t in zip(br_errors, true_br_vals) if t > 0]

        return BenchmarkMetrics(
            algorithm_name=result.algorithm_name,
            scenario_name=scenario.name,
            n_peaks_detected=n_detected,
            n_peaks_true=n_true,
            peak_count_abs_error=peak_count_abs_error,
            peak_count_error_pct=peak_count_error_pct,
            r0_error_mean=float(np.mean(r0_errors)),
            r0_error_std=float(np.std(r0_errors)),
            r0_error_pct_mean=float(np.mean(r0_pct)) if r0_pct else 0.0,
            sigma_error_mean=float(np.mean(sigma_errors)),
            sigma_error_std=float(np.std(sigma_errors)),
            sigma_error_pct_mean=float(np.mean(sigma_pct)) if sigma_pct else 0.0,
            beta_error_mean=float(np.mean(beta_errors)),
            beta_error_std=float(np.std(beta_errors)),
            beta_error_abs_mean=float(np.mean(np.abs(beta_errors))),
            br_error_mean=float(np.mean(br_errors)),
            br_error_std=float(np.std(br_errors)),
            br_error_pct_mean=float(np.mean(br_pct)) if br_pct else 0.0,
            computation_time=float(result.computation_time),
            success_rate=min(len(matches) / max(n_true, 1), 1.0),
        )

    @staticmethod
    def _harmonize_peak_units(est_peaks: List[Dict[str, float]], scenario: TestScenario) -> List[Dict[str, float]]:
        if not est_peaks:
            return est_peaks
        true_r = np.array([p.r0 for p in scenario.peaks if p.r0 > 0], dtype=float)
        est_r = np.array([p.get("r0", 0.0) for p in est_peaks if p.get("r0", 0.0) > 0], dtype=float)
        if true_r.size == 0 or est_r.size == 0:
            return est_peaks
        ratio = float(np.median(est_r) / max(np.median(true_r), 1e-12))
        pixel_size = float(max(scenario.pixel_size, 1e-12))
        expected_px_ratio = 1.0 / pixel_size
        convert = bool(ratio > 3.0 or abs(ratio - expected_px_ratio) <= max(2.0, 0.35 * expected_px_ratio))
        if not convert:
            return est_peaks
        out: List[Dict[str, float]] = []
        for peak in est_peaks:
            p = dict(peak)
            p["r0"] = float(p.get("r0", 0.0)) * pixel_size
            p["sigma"] = float(p.get("sigma", 0.0)) * pixel_size
            out.append(p)
        return out

    def _match_peaks(self, est_peaks: List[Dict[str, float]], true_peaks: List[GroundTruthPeak]) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[float, int, int]] = []
        for i, est in enumerate(est_peaks):
            for j, true in enumerate(true_peaks):
                d = abs(float(est["r0"]) - float(true.r0))
                if d <= self.max_distance:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda x: x[0])
        used_est = set()
        used_true = set()
        matches: List[Tuple[int, int, float]] = []
        for d, i, j in pairs:
            if i in used_est or j in used_true:
                continue
            used_est.add(i)
            used_true.add(j)
            matches.append((i, j, float(d)))
        return matches


class BenchmarkRunner:
    def __init__(self, algorithms: Optional[List[str]] = None, algorithm_kwargs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.generator = TestCaseGenerator()
        self.registry = AlgorithmRegistry()
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.algorithms = algorithms or self.registry.list_algorithms()
        self.algorithms = [a for a in self.algorithms if self.registry.get_algorithm(a) is not None]
        self.results: Dict[str, Dict[str, BenchmarkMetrics]] = {}

    def run_scenario(self, scenario: TestScenario, algorithm_name: Optional[str] = None) -> Dict[str, BenchmarkMetrics]:
        xy_data = self.generator.generate_scenario(scenario)
        alg_names = [algorithm_name] if algorithm_name else self.algorithms
        out: Dict[str, BenchmarkMetrics] = {}
        calculator = MetricsCalculator()
        for alg_name in alg_names:
            wrapper = self.registry.get_algorithm(alg_name)
            if wrapper is None:
                continue
            per_alg_kwargs = self.algorithm_kwargs.get(alg_name, {})
            result = wrapper.run(xy_data, n_peaks=len(scenario.peaks), **per_alg_kwargs)
            metrics = calculator.calculate_metrics(result, scenario)
            out[alg_name] = metrics
            self.results.setdefault(alg_name, {})
            self.results[alg_name][scenario.name] = metrics
        return out
