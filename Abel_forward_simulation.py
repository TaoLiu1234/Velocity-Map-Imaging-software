"""
Abel Forward Simulation for Velocity Map Imaging (VMI) Photoelectron Spectroscopy

This module implements a Monte Carlo forward simulation for VMI experiments.
It simulates the complete imaging chain from quantum mechanical photoemission
to final camera image.

Physical Pipeline:
    1. PhysicsSource: Generate 3D velocity distribution (energy + angular sampling)
    2. VMIInstrument: Project to 2D detector with PSF convolution
    3. CameraElectronics: Add noise sources (dark current, readout, background)

Units: Energy (eV), Length (mm), Time (fs), Mass (amu), Velocity (m/s)
"""

import numpy as np
from scipy.special import erf
from scipy.constants import electron_mass, elementary_charge, atomic_mass, Boltzmann, speed_of_light
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt


# =============================================================================
# Physical Constants
# =============================================================================
EV_TO_JOULE = elementary_charge      # 1 eV = 1.602e-19 J
AMU_TO_KG = atomic_mass              # 1 amu = 1.661e-27 kg
HBAR_EV_FS = 0.6582119569            # hbar in eV·fs
C_M_S = speed_of_light               # Speed of light in m/s
ELECTRON_MASS_AMU = electron_mass / AMU_TO_KG


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """
    Configuration container for VMI forward simulation.
    
    Parameters are organized into groups:
    - Particle: mass
    - Energy levels: E_centers, Betas, branching_ratios
    - Broadening: sigma_laser, T_beam, tau_lifetimes, photon_energy, target_mass
    - Geometry: vol_sigma, polarization_vec
    - Detector: img_res, pixel_size, psf_fwhm
    - Electronics: dark_rate, readout_sigma, readout_offset
    - Background: bg_rate, bg_energy, bg_sigma
    - Simulation: N_events, vmi_k
    """
    # Particle properties
    mass: float = ELECTRON_MASS_AMU  # Particle mass in amu
    
    # Energy levels and angular distributions
    E_centers: List[float] = field(default_factory=lambda: [1.0])  # Peak energies (eV)
    Betas: List[float] = field(default_factory=lambda: [2.0])      # Anisotropy parameters
    branching_ratios: List[float] = field(default_factory=lambda: [1.0])  # Relative intensities
    
    # Simulation parameters
    N_events: int = 100000           # Number of particles to simulate
    vmi_k: float = 0.01              # Velocity-to-radius coefficient mm/(m/s)
    
    # Broadening parameters
    sigma_laser: float = 0.01        # Laser bandwidth sigma (eV)
    T_beam: float = 10.0             # Molecular beam temperature (K)
    tau_lifetimes: Union[float, List[float]] = 0.0  # Excited state lifetime (fs)
    photon_energy: float = 0.0       # Ionizing photon energy (eV)
    target_mass: float = 28.0        # Target molecule mass (amu)
    
    # Geometry parameters
    vol_sigma: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Interaction volume (mm)
    polarization_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    
    # Detector parameters
    img_res: int = 512               # Image resolution (pixels)
    pixel_size: float = 0.05         # Pixel size (mm)
    psf_fwhm: float = 0.2            # Point spread function FWHM (mm)
    dld_resolution: float = 0.005    # DLD digitization resolution (mm), 0 = no quantization
    supersample_factor: int = 4      # Deprecated, kept for compatibility
    
    # Electronics parameters
    dark_rate: float = 0.1           # Dark current (counts/pixel)
    readout_sigma: float = 5.0       # Readout noise sigma (counts)
    readout_offset: float = 100.0    # Readout offset/bias (counts)
    
    # Background gas parameters
    bg_rate: float = 0.01            # Background fraction of N_events
    bg_energy: float = 0.1           # Background electron energy (eV)
    bg_sigma: float = 0.05           # Background energy spread (eV)
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize polarization vector
        self.polarization_vec = np.asarray(self.polarization_vec, dtype=float)
        self.polarization_vec /= np.linalg.norm(self.polarization_vec)
        
        # Normalize branching ratios
        total = sum(self.branching_ratios)
        self.branching_ratios = [r / total for r in self.branching_ratios]
        
        # Validate array lengths
        n_levels = len(self.E_centers)
        if len(self.Betas) != n_levels:
            raise ValueError(f"Betas length ({len(self.Betas)}) must match E_centers ({n_levels})")
        if len(self.branching_ratios) != n_levels:
            raise ValueError(f"branching_ratios length must match E_centers")
        
        # Convert scalar tau_lifetimes to list
        if isinstance(self.tau_lifetimes, (int, float)):
            self.tau_lifetimes = [float(self.tau_lifetimes)] * n_levels
        elif len(self.tau_lifetimes) != n_levels:
            raise ValueError(f"tau_lifetimes length must match E_centers")
    
    @property
    def detector_size_mm(self) -> float:
        """Total detector size in mm."""
        return self.img_res * self.pixel_size
    
    @property
    def psf_sigma(self) -> float:
        """PSF standard deviation in mm (converted from FWHM)."""
        return self.psf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    @staticmethod
    def calculate_vmi_k(E_max_eV: float, r_max_mm: float, 
                        mass_amu: float = ELECTRON_MASS_AMU) -> float:
        """
        Calculate vmi_k to map energy E_max to radius r_max.
        
        Uses: r = k * v, where v = sqrt(2E/m)
        """
        mass_kg = mass_amu * AMU_TO_KG
        v_max = np.sqrt(2.0 * E_max_eV * EV_TO_JOULE / mass_kg)
        return r_max_mm / v_max
    
    def get_expected_radius(self, E_eV: float) -> float:
        """Calculate expected detector radius for given energy."""
        mass_kg = self.mass * AMU_TO_KG
        v = np.sqrt(2.0 * E_eV * EV_TO_JOULE / mass_kg)
        return self.vmi_k * v


# =============================================================================
# Physics Source
# =============================================================================
class PhysicsSource:
    """
    Generates 3D particle distribution based on quantum mechanics.
    
    Handles:
    - Energy sampling with Voigt profile (Gaussian + Lorentzian broadening)
    - Angular distribution sampling with beta parameter
    - Interaction volume sampling
    """
    
    def __init__(self, config: Config):
        self.cfg = config
        self._rotation_matrix = self._compute_rotation_matrix()
    
    def _compute_rotation_matrix(self) -> np.ndarray:
        """
        Compute rotation matrix from intrinsic frame (Z' = polarization) to lab frame.
        """
        z_prime = self.cfg.polarization_vec
        
        # Find perpendicular vector for x'
        ref = np.array([1, 0, 0]) if abs(z_prime[0]) < 0.9 else np.array([0, 1, 0])
        x_prime = np.cross(z_prime, ref)
        x_prime /= np.linalg.norm(x_prime)
        
        # y' completes right-handed system
        y_prime = np.cross(z_prime, x_prime)
        
        return np.column_stack([x_prime, y_prime, z_prime])
    
    def _calculate_doppler_sigma(self, E_center: float) -> float:
        """
        Calculate Doppler broadening sigma (eV) for given energy level.
        
        Two contributions:
        1. Photon Doppler shift: sigma_E = E_photon * (sigma_v / c)
        2. Momentum transfer: sigma_E = sqrt(2 * m_e * E) * sigma_v / e
        """
        if self.cfg.T_beam <= 0:
            return 0.0
        
        # Thermal velocity spread: sigma_v = sqrt(kT/M)
        target_mass_kg = self.cfg.target_mass * AMU_TO_KG
        sigma_v = np.sqrt(Boltzmann * self.cfg.T_beam / target_mass_kg)
        
        sigma_sq = 0.0
        
        # Photon Doppler contribution
        if self.cfg.photon_energy > 0:
            sigma_photon = self.cfg.photon_energy * sigma_v / C_M_S
            sigma_sq += sigma_photon**2
        
        # Momentum transfer contribution
        if E_center > 0:
            sigma_recoil = np.sqrt(2 * electron_mass * E_center * EV_TO_JOULE) * sigma_v / EV_TO_JOULE
            sigma_sq += sigma_recoil**2
        
        return np.sqrt(sigma_sq)
    
    def _calculate_lifetime_gamma(self, level_idx: int) -> float:
        """
        Calculate Lorentzian HWHM (eV) from lifetime using Heisenberg uncertainty.
        
        gamma = hbar / (2 * tau)
        """
        tau = self.cfg.tau_lifetimes[level_idx]
        return HBAR_EV_FS / (2.0 * tau) if tau > 0 else 0.0
    
    def _sample_truncated_cauchy(self, gamma: float, n: int) -> np.ndarray:
        """Sample from truncated Cauchy distribution with HWHM gamma."""
        max_width = 10 * gamma
        samples = []
        while len(samples) < n:
            x = np.random.standard_cauchy(2 * (n - len(samples))) * gamma
            valid = np.abs(x) < max_width
            samples.extend(x[valid])
        return np.array(samples[:n])
    
    def _sample_cos_theta(self, beta: float, n: int) -> np.ndarray:
        """
        Sample cos(theta) from angular distribution using rejection sampling.
        
        PDF: f(x) = 1 + beta * P2(x), where P2(x) = (3x^2 - 1) / 2
        """
        # Maximum of PDF for rejection sampling
        f_max = 1 + beta if beta >= 0 else 1 - beta / 2
        
        samples = []
        while len(samples) < n:
            x = np.random.uniform(-1, 1, 2 * (n - len(samples)))
            u = np.random.uniform(0, f_max, len(x))
            P2 = (3 * x**2 - 1) / 2
            f_x = 1 + beta * P2
            samples.extend(x[u < f_x])
        
        return np.array(samples[:n])
    
    def sample_energy(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample particle energies with Voigt profile broadening.
        
        Returns: (velocity_magnitudes, level_indices)
        """
        # Allocate particles to energy levels
        level_counts = np.random.multinomial(N, self.cfg.branching_ratios)
        
        velocities = []
        level_indices = []
        
        for level_idx, (E_center, n_particles) in enumerate(zip(self.cfg.E_centers, level_counts)):
            if n_particles == 0:
                continue
            
            # Gaussian broadening: laser + Doppler (in quadrature)
            sigma_gauss = np.sqrt(
                self.cfg.sigma_laser**2 + 
                self._calculate_doppler_sigma(E_center)**2
            )
            
            # Lorentzian broadening: lifetime
            gamma_lorentz = self._calculate_lifetime_gamma(level_idx)
            
            # Sample energy deviations
            E_gauss = np.random.normal(0, sigma_gauss, n_particles) if sigma_gauss > 0 else 0
            E_lorentz = self._sample_truncated_cauchy(gamma_lorentz, n_particles) if gamma_lorentz > 0 else 0
            
            # Final energy (Voigt = Gaussian + Lorentzian)
            E_final = np.maximum(E_center + E_gauss + E_lorentz, 1e-10)
            
            # Convert to velocity: v = sqrt(2E/m)
            mass_kg = self.cfg.mass * AMU_TO_KG
            v_mag = np.sqrt(2.0 * E_final * EV_TO_JOULE / mass_kg)
            
            velocities.append(v_mag)
            level_indices.append(np.full(n_particles, level_idx, dtype=int))
        
        return np.concatenate(velocities), np.concatenate(level_indices)
    
    def sample_direction(self, N: int, level_indices: np.ndarray) -> np.ndarray:
        """
        Sample emission directions based on beta parameters.
        
        Angular distribution: I(theta) = 1 + beta * P2(cos(theta))
        """
        directions = np.zeros((N, 3))
        
        for level_idx, beta in enumerate(self.cfg.Betas):
            mask = level_indices == level_idx
            n_level = np.sum(mask)
            if n_level == 0:
                continue
            
            # Sample angles in intrinsic frame
            phi = np.random.uniform(0, 2 * np.pi, n_level)
            cos_theta = self._sample_cos_theta(beta, n_level)
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian (intrinsic frame)
            v_intrinsic = np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta
            ])
            
            # Rotate to lab frame
            directions[mask] = v_intrinsic @ self._rotation_matrix.T
        
        return directions
    
    def sample_origin(self, N: int) -> np.ndarray:
        """Sample starting positions from interaction volume (3D Gaussian)."""
        return np.column_stack([
            np.random.normal(0, max(s, 1e-10), N) for s in self.cfg.vol_sigma
        ])
    
    def generate_particles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete particle data.
        
        Returns: (origins, velocities, level_indices)
        - origins: (N, 3) starting positions in mm
        - velocities: (N, 3) velocity vectors in m/s
        - level_indices: (N,) energy level index
        """
        N = self.cfg.N_events
        
        v_mag, level_indices = self.sample_energy(N)
        directions = self.sample_direction(N, level_indices)
        velocities = v_mag[:, np.newaxis] * directions
        origins = self.sample_origin(N)
        
        return origins, velocities, level_indices


# =============================================================================
# VMI Instrument
# =============================================================================
class VMIInstrument:
    """
    Simulates VMI instrument response.
    
    Handles:
    - 3D to 2D projection (Abel projection)
    - PSF convolution using exact Gaussian integral
    - Pixel binning
    """
    
    def __init__(self, config: Config):
        self.cfg = config
        # Precompute pixel edges
        half_size = config.detector_size_mm / 2
        self._edges = np.linspace(-half_size, half_size, config.img_res + 1)
        self._half_size = half_size
    
    def project_to_detector(self, origins: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """
        Project 3D trajectories to 2D detector positions.
        
        VMI mapping: r_det = r_origin + k * v
        Z-component is lost (Abel projection).
        """
        X_det = origins[:, 0] + self.cfg.vmi_k * velocities[:, 0]
        Y_det = origins[:, 1] + self.cfg.vmi_k * velocities[:, 1]
        return np.column_stack([X_det, Y_det])
    
    def apply_psf_and_pixelize(self, hits: np.ndarray) -> np.ndarray:
        """
        Apply PSF and pixelize using exact Gaussian integral method.
        
        For each particle at position (x, y), compute the exact integral of
        the 2D Gaussian PSF over each affected pixel using the error function:
        
        Integral over [x1,x2] x [y1,y2] = 
            [Phi((x2-x)/sigma) - Phi((x1-x)/sigma)] *
            [Phi((y2-y)/sigma) - Phi((y1-y)/sigma)]
        
        where Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        """
        img = np.zeros((self.cfg.img_res, self.cfg.img_res))
        psf_sigma = self.cfg.psf_sigma
        pixel_size = self.cfg.pixel_size
        img_res = self.cfg.img_res
        half_size = self._half_size
        edges = self._edges
        
        # If PSF is negligible, use simple histogram
        if psf_sigma < 1e-10:
            img, _, _ = np.histogram2d(hits[:, 0], hits[:, 1], bins=[edges, edges])
            return img
        
        # PSF truncation at 4 sigma (captures 99.994% of distribution)
        cutoff = 4.0 * psf_sigma
        sqrt2_sigma = np.sqrt(2.0) * psf_sigma
        
        # Process each particle
        for x_hit, y_hit in hits:
            # Skip particles far outside detector
            if abs(x_hit) > half_size + cutoff or abs(y_hit) > half_size + cutoff:
                continue
            
            # Find affected pixel range
            i_min = max(0, int((x_hit - cutoff + half_size) / pixel_size))
            i_max = min(img_res, int((x_hit + cutoff + half_size) / pixel_size) + 1)
            j_min = max(0, int((y_hit - cutoff + half_size) / pixel_size))
            j_max = min(img_res, int((y_hit + cutoff + half_size) / pixel_size) + 1)
            
            if i_min >= i_max or j_min >= j_max:
                continue
            
            # Compute erf at pixel edges
            erf_x = erf((edges[i_min:i_max + 1] - x_hit) / sqrt2_sigma)
            erf_y = erf((edges[j_min:j_max + 1] - y_hit) / sqrt2_sigma)
            
            # Integral over each pixel = 0.5 * diff(erf)
            dx = 0.5 * np.diff(erf_x)
            dy = 0.5 * np.diff(erf_y)
            
            # Add contribution (outer product for 2D)
            img[i_min:i_max, j_min:j_max] += np.outer(dx, dy)
        
        return img
    
    def process(self, origins: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Complete instrument pipeline: project + PSF + pixelize."""
        hits = self.project_to_detector(origins, velocities)
        return self.apply_psf_and_pixelize(hits)


# =============================================================================
# Camera Electronics
# =============================================================================
class CameraElectronics:
    """
    Simulates camera electronics and noise sources.
    
    Noise model:
    - Background gas: low-energy isotropic electrons
    - Dark current: Poisson distributed
    - Readout noise: Gaussian distributed with offset
    """
    
    def __init__(self, config: Config):
        self.cfg = config
    
    def add_background(self, img: np.ndarray) -> np.ndarray:
        """Add background gas contribution (isotropic, low-energy electrons)."""
        if self.cfg.bg_rate <= 0:
            return img
        
        # Create background config
        bg_config = Config(
            mass=self.cfg.mass,
            E_centers=[self.cfg.bg_energy],
            Betas=[0.0],  # Isotropic
            branching_ratios=[1.0],
            N_events=int(self.cfg.N_events * self.cfg.bg_rate),
            vmi_k=self.cfg.vmi_k,
            sigma_laser=self.cfg.bg_sigma,
            T_beam=0.0,
            tau_lifetimes=0.0,
            vol_sigma=self.cfg.vol_sigma,
            polarization_vec=self.cfg.polarization_vec,
            img_res=self.cfg.img_res,
            pixel_size=self.cfg.pixel_size,
            psf_fwhm=self.cfg.psf_fwhm,
        )
        
        physics = PhysicsSource(bg_config)
        instrument = VMIInstrument(bg_config)
        origins, velocities, _ = physics.generate_particles()
        
        return img + instrument.process(origins, velocities)
    
    def add_dark_current(self, img: np.ndarray) -> np.ndarray:
        """Add Poisson-distributed dark current noise."""
        if self.cfg.dark_rate <= 0:
            return img
        return img + np.random.poisson(self.cfg.dark_rate, img.shape)
    
    def add_readout_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian readout noise with offset."""
        return img + np.random.normal(self.cfg.readout_offset, self.cfg.readout_sigma, img.shape)
    
    def process(self, img: np.ndarray, add_background: bool = True) -> np.ndarray:
        """Apply all noise sources."""
        if add_background:
            img = self.add_background(img)
        img = self.add_dark_current(img)
        img = self.add_readout_noise(img)
        return img


# =============================================================================
# Main Simulation Function
# =============================================================================
def run_simulation(config: Config, 
                   add_noise: bool = True,
                   add_background: bool = True,
                   return_particles: bool = False,
                   output_mode: str = 'image') -> Tuple[np.ndarray, dict]:
    """
    Run complete VMI forward simulation.
    
    Args:
        config: Simulation configuration
        add_noise: Whether to add electronic noise
        add_background: Whether to add background gas
        return_particles: If True, include particle data in metadata
        output_mode: 
            'image' - 返回像素化图像（默认），完整链路
            'xy_ideal' - 返回理想 XY 坐标（无任何展宽，理论极限）
            'xy_dld' - 返回模拟 DLD 输出的 XY 坐标（PSF + DLD量化）
    
    Returns:
        output_mode='image': (image, metadata) tuple
        output_mode='xy_ideal' or 'xy_dld': (xy_coords, metadata) tuple
    
    数据流：
        真实位置 → [PSF展宽] → 连续位置 → [DLD量化] → DLD输出坐标 → [histogram] → 图像
        
        xy_ideal: 真实位置（无展宽）
        xy_dld: DLD输出坐标（有PSF+量化，无histogram）
        image: 最终图像（完整链路）
    """
    physics = PhysicsSource(config)
    instrument = VMIInstrument(config)
    camera = CameraElectronics(config)
    
    # Generate and project particles
    origins, velocities, level_indices = physics.generate_particles()
    
    # 投影到探测器平面，得到理想的 XY 坐标
    hits_ideal = instrument.project_to_detector(origins, velocities)
    
    if output_mode == 'xy_ideal':
        # 直接返回理想 XY 坐标（无任何展宽）
        # 这是"真实"位置，实际探测器无法达到
        metadata = {
            'N_events': config.N_events,
            'E_centers': config.E_centers,
            'Betas': config.Betas,
            'output_mode': 'xy_ideal',
            'note': 'Ideal XY coordinates without any broadening (theoretical limit)'
        }
        if return_particles:
            metadata['origins'] = origins
            metadata['velocities'] = velocities
            metadata['level_indices'] = level_indices
        return hits_ideal, metadata
    
    elif output_mode in ['xy_dld', 'xy_with_psf']:  # xy_with_psf 保留兼容
        # 模拟完整的 DLD 输出：PSF 展宽 + DLD 数字化量化 + 背景噪声
        hits_processed = hits_ideal.copy()
        
        # Step 1: PSF 展宽（MCP + 延迟线的空间分辨率）
        psf_sigma = config.psf_sigma
        if psf_sigma > 0:
            noise_x = np.random.normal(0, psf_sigma, len(hits_ideal))
            noise_y = np.random.normal(0, psf_sigma, len(hits_ideal))
            hits_processed = hits_processed + np.column_stack([noise_x, noise_y])
        
        # Step 2: DLD 数字化量化（TDC 时间分辨率 → 位置精度）
        dld_res = config.dld_resolution
        if dld_res > 0:
            hits_processed = np.round(hits_processed / dld_res) * dld_res
        
        # Step 3: 添加高斯背景噪声（类似图像模式）
        # 噪声区域比实际数据区域大，模拟真实探测器的背景噪声
        n_bg_events = int(config.N_events * config.bg_rate) if config.bg_rate > 0 else 0
        
        if n_bg_events > 0:
            # 计算数据的实际范围
            data_r_max = np.sqrt(np.max(hits_processed[:, 0]**2 + hits_processed[:, 1]**2))
            
            # 背景噪声区域比数据区域大 50%（覆盖更大范围）
            bg_extent = data_r_max * 1.5
            
            # 生成均匀分布的背景噪声点（在圆形区域内）
            # 使用极坐标采样确保均匀分布
            bg_r = np.sqrt(np.random.uniform(0, 1, n_bg_events)) * bg_extent
            bg_theta = np.random.uniform(0, 2 * np.pi, n_bg_events)
            bg_x = bg_r * np.cos(bg_theta)
            bg_y = bg_r * np.sin(bg_theta)
            bg_hits = np.column_stack([bg_x, bg_y])
            
            # 对背景噪声也应用 PSF 展宽
            if psf_sigma > 0:
                bg_noise_x = np.random.normal(0, psf_sigma, n_bg_events)
                bg_noise_y = np.random.normal(0, psf_sigma, n_bg_events)
                bg_hits = bg_hits + np.column_stack([bg_noise_x, bg_noise_y])
            
            # 对背景噪声也应用 DLD 量化
            if dld_res > 0:
                bg_hits = np.round(bg_hits / dld_res) * dld_res
            
            # 合并信号和背景
            hits_processed = np.vstack([hits_processed, bg_hits])
        
        # 计算等效展宽
        sigma_dld = dld_res / np.sqrt(12) if dld_res > 0 else 0
        
        metadata = {
            'N_events': config.N_events,
            'N_signal': len(hits_ideal),
            'N_background': n_bg_events,
            'E_centers': config.E_centers,
            'Betas': config.Betas,
            'output_mode': 'xy_dld',
            'psf_sigma_mm': psf_sigma,
            'dld_resolution_mm': dld_res,
            'sigma_dld_mm': sigma_dld,
            'bg_rate': config.bg_rate,
            'note': 'Simulated DLD output: PSF broadening + digitization + background noise'
        }
        if return_particles:
            metadata['origins'] = origins
            metadata['velocities'] = velocities
            metadata['level_indices'] = level_indices
        return hits_processed, metadata
    
    else:  # output_mode == 'image'
        # 原来的行为：生成像素化图像
        image_ideal = instrument.process(origins, velocities)
        
        # Add noise
        final_image = camera.process(image_ideal, add_background) if add_noise else image_ideal
        
        # Metadata
        metadata = {
            'N_events': config.N_events,
            'E_centers': config.E_centers,
            'Betas': config.Betas,
            'output_mode': 'image',
            'image_ideal_sum': float(np.sum(image_ideal)),
            'image_final_sum': float(np.sum(final_image)),
        }
        
        if return_particles:
            metadata['origins'] = origins
            metadata['velocities'] = velocities
            metadata['level_indices'] = level_indices
        
        return final_image, metadata



# =============================================================================
# Visualization Functions
# =============================================================================
def visualize_simulation(image: np.ndarray, config: Config,
                         title: str = "VMI Simulation",
                         save_path: Optional[str] = None):
    """
    Visualize simulation result with linear and log scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    extent = [-config.detector_size_mm/2, config.detector_size_mm/2,
              -config.detector_size_mm/2, config.detector_size_mm/2]
    
    # Linear scale
    im1 = axes[0].imshow(image.T, origin='lower', extent=extent, cmap='hot')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title(f'{title} - Linear')
    plt.colorbar(im1, ax=axes[0], label='Counts')
    
    # Log scale
    img_log = np.log10(np.maximum(image, 1))
    im2 = axes[1].imshow(img_log.T, origin='lower', extent=extent, cmap='hot')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Y (mm)')
    axes[1].set_title(f'{title} - Log scale')
    plt.colorbar(im2, ax=axes[1], label='log10(Counts)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_velocity_slices(velocities: np.ndarray, config: Config,
                           n_bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create 2D projections of 3D velocity distribution.
    
    Returns: (slice_xy, slice_xz, slice_yz, v_max)
    """
    E_max = max(config.E_centers) * 1.5
    mass_kg = config.mass * AMU_TO_KG
    v_max = np.sqrt(2.0 * E_max * EV_TO_JOULE / mass_kg)
    
    bins = np.linspace(-v_max, v_max, n_bins + 1)
    
    slice_xy, _, _ = np.histogram2d(velocities[:, 0], velocities[:, 1], bins=[bins, bins])
    slice_xz, _, _ = np.histogram2d(velocities[:, 0], velocities[:, 2], bins=[bins, bins])
    slice_yz, _, _ = np.histogram2d(velocities[:, 1], velocities[:, 2], bins=[bins, bins])
    
    return slice_xy, slice_xz, slice_yz, v_max


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VMI Forward Simulation Demo")
    print("=" * 70)
    
    # Define energy levels
    E_centers = [0.5, 1.0, 2.0]
    Betas = [2.0, 0.0, -0.5]
    branching_ratios = [0.3, 0.5, 0.2]
    
    # Calculate VMI calibration
    E_max = max(E_centers)
    r_max_mm = 20.0
    vmi_k = Config.calculate_vmi_k(E_max, r_max_mm)
    
    print(f"\nVMI calibration: k = {vmi_k:.4e} mm/(m/s)")
    print(f"  Maps {E_max} eV -> {r_max_mm} mm radius")
    
    # Create configuration
    config = Config(
        E_centers=E_centers,
        Betas=Betas,
        branching_ratios=branching_ratios,
        N_events=int(1e6),
        vmi_k=vmi_k,
        sigma_laser=0.015,
        T_beam=10.0,
        tau_lifetimes=[100.0, 50.0, 200.0],
        photon_energy=21.2,
        target_mass=28.0,
        vol_sigma=(0.0, 0.0, 0.0),
        polarization_vec=[0, 1, 0],
        img_res=512,
        pixel_size=0.1,
        psf_fwhm=0.0,
        dark_rate=0.1,
        readout_sigma=5.0,
        readout_offset=0.0,
        bg_rate=0.02,
        bg_energy=0.15,
        bg_sigma=0.08,
    )
    
    print(f"\nConfiguration:")
    print(f"  Energy levels: {config.E_centers} eV")
    print(f"  Beta values: {config.Betas}")
    print(f"  Branching ratios: {config.branching_ratios}")
    print(f"  Detector: {config.img_res} px, {config.detector_size_mm:.1f} mm")
    print(f"  PSF FWHM: {config.psf_fwhm} mm")
    
    # Expected radii
    print(f"\nExpected radii:")
    for E in config.E_centers:
        r = config.get_expected_radius(E)
        print(f"  {E} eV -> {r:.2f} mm")
    
    # Run simulation
    print(f"\nRunning simulation with {config.N_events:,} particles...")
    image_noisy, meta_noisy = run_simulation(config, add_noise=True, add_background=True)
    
    print(f"  Ideal sum: {meta_noisy['image_ideal_sum']:.0f}")
    print(f"  Final sum: {meta_noisy['image_final_sum']:.0f}")
    
    # Clean image for reconstruction
    image_clean, _ = run_simulation(config, add_noise=False, add_background=False)
    
    # =========================================================================
    # Reconstruction Comparison
    # =========================================================================
    RUN_RECONSTRUCTION = True
    
    if RUN_RECONSTRUCTION:
        print("\n" + "=" * 70)
        print("RECONSTRUCTION TEST")
        print("=" * 70)
        
        from Abel_backward_reconstruction import (
            reconstruct_vmi_image,
            compare_reconstruction,
            visualize_reconstruction
        )
        from Abel_rbasex_reconstruction import reconstruct_rbasex
        
        # True parameters
        true_params = {
            'E_centers': config.E_centers,
            'Betas': config.Betas,
            'branching_ratios': config.branching_ratios,
            'sigma_laser': config.sigma_laser
        }
        
        print("\nTrue parameters:")
        print(f"  Energy levels: {true_params['E_centers']} eV")
        print(f"  Beta values: {true_params['Betas']}")
        print(f"  Branching ratios: {true_params['branching_ratios']}")
        
        # PhysicsBasedFitter reconstruction
        print("\n" + "-" * 60)
        print("Running PhysicsBasedFitter reconstruction...")
        print("-" * 60)
        physics_params, physics_metadata = reconstruct_vmi_image(
            image_clean, config=config, verbose=True
        )
        
        # rBasex reconstruction
        print("\n" + "-" * 60)
        print("Running rBasex reconstruction...")
        print("-" * 60)
        rbasex_params, rbasex_metadata = reconstruct_rbasex(
            image_clean, config=config, verbose=True
        )
        
        # Compare with ground truth
        compare_reconstruction(true_params, physics_params, config=config, rbasex_params=rbasex_params)
        
        # Generate 3D distribution for visualization
        print("\nGenerating 3D distribution for visualization...")
        _, meta_3d = run_simulation(
            config, add_noise=False, add_background=False, return_particles=True
        )
        
        # Create XY slice of 3D velocity distribution
        slice_xy, _, _, v_max = create_velocity_slices(
            meta_3d['velocities'], config, n_bins=config.img_res
        )
        
        # Visualize reconstruction
        visualize_reconstruction(
            image_clean, physics_params, physics_metadata,
            config=config, true_params=true_params,
            image_3d=slice_xy,
            rbasex_params=rbasex_params,
            rbasex_metadata=rbasex_metadata,
            save_path="reconstruction_comparison.png"
        )
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
