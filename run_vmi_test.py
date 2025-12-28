"""
Comprehensive VMI Algorithm Testing - Optimized Version
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from Abel_forward_simulation import Config, run_simulation, ELECTRON_MASS_AMU, EV_TO_JOULE, AMU_TO_KG
from vmi_reconstruction import PeakResult

@dataclass
class TestResult:
    case_name: str
    n_peaks: int
    true_r0: List[float]
    true_sigma: List[float]
    true_beta: List[float]
    true_amp: List[float]
    est_r0: List[float]
    est_sigma: List[float]
    est_beta: List[float]
    est_amp: List[float]
    r0_err_pct: List[float]
    sigma_err_pct: List[float]
    beta_err_abs: List[float]
    amp_err_pct: List[float]
    n_detected: int
    n_matched: int
    passed: bool
    exec_time: float

def energy_to_radius(E_eV, vmi_k):
    mass_kg = ELECTRON_MASS_AMU * AMU_TO_KG
    v = np.sqrt(2.0 * E_eV * EV_TO_JOULE / mass_kg)
    return vmi_k * v

def create_config(E_centers, betas, branching_ratios, n_events, sigma_laser=0.03, noise='low'):
    r_max = 20.0
    E_max = max(E_centers) * 1.5
    vmi_k = Config.calculate_vmi_k(E_max, r_max)
    noise_params = {'clean': (0.0, 0.0), 'low': (0.1, 0.01), 'high': (0.3, 0.05)}
    psf, dld = noise_params.get(noise, (0.1, 0.01))
    return Config(
        E_centers=E_centers, Betas=betas, branching_ratios=branching_ratios,
        N_events=n_events, vmi_k=vmi_k, sigma_laser=sigma_laser,
        T_beam=0.0, tau_lifetimes=0.0, photon_energy=0.0, target_mass=28.0,
        vol_sigma=(0.0, 0.0, 0.0), polarization_vec=[0, 1, 0],
        img_res=512, pixel_size=0.05, psf_fwhm=psf, dld_resolution=dld,
        dark_rate=0.0, readout_sigma=0.0, readout_offset=0.0, bg_rate=0.0,
    )

def match_peaks(est_peaks, true_r0s):
    if not est_peaks or not true_r0s:
        return []
    distances = [(abs(est_peaks[i].r0 - true_r0s[j]), i, j) 
                 for i in range(len(est_peaks)) for j in range(len(true_r0s))]
    distances.sort()
    matches, used_est, used_true = [], set(), set()
    for dist, i, j in distances:
        if i not in used_est and j not in used_true:
            matches.append((i, j))
            used_est.add(i)
            used_true.add(j)
    return matches

def run_test(config, name, recon_class, verbose=False):
    start = time.time()
    n_peaks = len(config.E_centers)
    true_r0 = [energy_to_radius(E, config.vmi_k) for E in config.E_centers]
    true_beta = list(config.Betas)
    true_amp = list(config.branching_ratios)
    true_sigma = [config.sigma_laser * r / (2 * E) if E > 0 else 0.1 
                  for r, E in zip(true_r0, config.E_centers)]
    
    xy_data, _ = run_simulation(config, output_mode='xy_dld', add_noise=False, add_background=False)
    recon = recon_class(xy_data)
    peaks = recon.reconstruct(n_peaks=n_peaks, verbose=False)
    
    matches = match_peaks(peaks, true_r0)
    total_amp = sum(p.amp for p in peaks) if peaks else 1.0
    
    est_r0, est_sigma, est_beta, est_amp = [], [], [], []
    r0_err, sigma_err, beta_err, amp_err = [], [], [], []
    
    for ei, ti in sorted(matches, key=lambda x: x[1]):
        p = peaks[ei]
        est_r0.append(p.r0)
        est_sigma.append(p.sigma)
        est_beta.append(p.beta)
        est_amp.append(p.amp / total_amp)
        r0_err.append(abs(p.r0 - true_r0[ti]) / true_r0[ti] * 100)
        sigma_err.append(abs(p.sigma - true_sigma[ti]) / true_sigma[ti] * 100 if true_sigma[ti] > 0 else 0)
        beta_err.append(abs(p.beta - true_beta[ti]))
        amp_err.append(abs(p.amp / total_amp - true_amp[ti]) / true_amp[ti] * 100 if true_amp[ti] > 0 else 0)
    
    for idx in set(range(n_peaks)) - set(j for _, j in matches):
        est_r0.append(None); est_sigma.append(None); est_beta.append(None); est_amp.append(None)
        r0_err.append(100.0); sigma_err.append(100.0); beta_err.append(3.0); amp_err.append(100.0)
    
    passed = len(matches) == n_peaks and all(e < 5.0 for e in r0_err) and all(e < 0.2 for e in beta_err)
    
    return TestResult(name, n_peaks, true_r0, true_sigma, true_beta, true_amp,
                      est_r0, est_sigma, est_beta, est_amp, r0_err, sigma_err, beta_err, amp_err,
                      len(peaks), len(matches), passed, time.time() - start)

def generate_tests():
    tests = []
    # Single peak tests
    for pos, E in [('inner', 0.2), ('mid', 0.8), ('outer', 1.6)]:
        tests.append((f"1p_{pos}", create_config([E], [0.0], [1.0], int(1e6))))
    for bn, b in [('b-1', -1.0), ('b0', 0.0), ('b1', 1.0), ('b2', 2.0)]:
        tests.append((f"1p_{bn}", create_config([0.8], [b], [1.0], int(1e6))))
    for sn, s in [('narrow', 0.01), ('wide', 0.08)]:
        tests.append((f"1p_{sn}", create_config([0.8], [0.5], [1.0], int(1e6), sigma_laser=s)))
    for nn, n in [('10k', int(1e4)), ('1M', int(1e6))]:
        tests.append((f"1p_{nn}", create_config([0.8], [0.5], [1.0], n)))
    # Two peak tests
    tests.append(("2p_well", create_config([0.4, 1.2], [0.5, 0.5], [0.5, 0.5], int(1e6))))
    tests.append(("2p_close", create_config([0.7, 0.85], [0.5, 0.5], [0.5, 0.5], int(1e6))))
    for rn, r in [('eq', [0.5, 0.5]), ('5:1', [0.83, 0.17])]:
        tests.append((f"2p_{rn}", create_config([0.4, 1.0], [0.5, 0.5], r, int(1e6))))
    # Three peak
    tests.append(("3p_well", create_config([0.3, 0.7, 1.3], [0.0, 0.5, 1.0], [0.33, 0.34, 0.33], int(1e6))))
    return tests

def run_all(recon_class):
    print("=" * 60)
    print("VMI COMPREHENSIVE TEST")
    print("=" * 60)
    tests = generate_tests()
    results = []
    for name, cfg in tests:
        r = run_test(cfg, name, recon_class)
        results.append(r)
        s = "PASS" if r.passed else "FAIL"
        print(f"{s} {name}: r0={np.mean(r.r0_err_pct):.1f}% b={np.mean(r.beta_err_abs):.3f} amp={np.mean(r.amp_err_pct):.1f}%")
    
    passed = sum(1 for r in results if r.passed)
    print(f"\nTotal: {passed}/{len(results)} ({100*passed/len(results):.0f}%)")
    
    # By category
    for cat in ['1p', '2p', '3p']:
        sub = [r for r in results if r.case_name.startswith(cat)]
        if sub:
            p = sum(1 for r in sub if r.passed)
            print(f"  {cat}: {p}/{len(sub)}")
    return results

if __name__ == "__main__":
    from vmi_test_framework import ImprovedVMIReconstructor
    run_all(ImprovedVMIReconstructor)
