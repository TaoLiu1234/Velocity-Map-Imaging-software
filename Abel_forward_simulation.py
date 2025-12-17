"""
Abel Forward Simulation for Photoelectron Spectroscopy (VMI)

This module implements a Monte Carlo forward simulation for Velocity Map Imaging (VMI)
photoelectron spectroscopy. It simulates the complete imaging chain from quantum
mechanical photoemission to final camera image.

Architecture:
    - Config: Global parameters container
    - PhysicsSource: Quantum/classical mechanics simulation (particle generation)
    - VMIInstrument: Optical projection, PSF convolution, spatial mapping
    - CameraElectronics: Photoelectric conversion, noise, final image generation

Author: Generated for VMI simulation
Units: Energy (eV), Length (mm), Time (ns), Mass (amu)
"""

import numpy as np
from scipy import ndimage
from scipy.constants import electron_mass, elementary_charge, atomic_mass, Boltzmann, speed_of_light
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt


# =============================================================================
# Physical Constants
# =============================================================================
# Conversion factors
EV_TO_JOULE = elementary_charge  # 1 eV = 1.602e-19 J
AMU_TO_KG = atomic_mass  # 1 amu = 1.661e-27 kg
MM_TO_M = 1e-3
M_TO_MM = 1e3
KB_EV = Boltzmann / elementary_charge  # Boltzmann constant in eV/K (~8.617e-5)
HBAR_EV_FS = 0.6582119569  # ℏ in eV·fs (reduced Planck constant)
C_M_S = speed_of_light  # Speed of light in m/s


# =============================================================================
# Configuration Class
# =============================================================================
@dataclass
class Config:
    """
    Configuration container for all simulation parameters.
    
    This class holds all parameters needed for the VMI forward simulation,
    organized into logical groups: particle properties, energy levels,
    broadening mechanisms, geometry, detector, and electronics.
    
    Attributes:
        -------------------------------------------------------------------------
        PARTICLE PROPERTIES
        -------------------------------------------------------------------------
        mass (float): Particle mass in atomic mass units (amu).
            Default: electron mass (~5.486e-4 amu).
            For ions, use the appropriate atomic/molecular mass.
        
        -------------------------------------------------------------------------
        ENERGY LEVELS AND ANGULAR DISTRIBUTIONS
        -------------------------------------------------------------------------
        E_centers (List[float]): List of energy level centers in eV.
            Each entry represents a distinct photoelectron peak/orbital.
            Example: [0.5, 1.0, 2.0] for three energy levels.
        
        Betas (List[float]): List of anisotropy parameters β for each energy level.
            β determines the angular distribution: I(θ) ∝ 1 + β·P₂(cos θ)
            where P₂ is the second Legendre polynomial.
            Valid range: -1 ≤ β ≤ 2
            - β = 2: Parallel transition (cos²θ distribution, emission along polarization)
            - β = 0: Isotropic emission (uniform in all directions)
            - β = -1: Perpendicular transition (sin²θ distribution)
            Must have same length as E_centers.
        
        branching_ratios (List[float]): Relative intensities for each energy level.
            Determines the fraction of particles emitted at each energy.
            Will be normalized to sum to 1.0.
            Must have same length as E_centers.
        
        -------------------------------------------------------------------------
        SIMULATION PARAMETERS
        -------------------------------------------------------------------------
        N_events (int): Total number of particles to simulate.
            Higher values give better statistics but longer computation time.
            Typical values: 1e5 for quick tests, 1e7-1e8 for publication quality.
        
        vmi_k (float): Velocity-to-radius conversion coefficient in mm/(m/s).
            This is the key VMI calibration parameter that maps particle velocity
            to detector position: r_detector = k × v_particle
            Use Config.calculate_vmi_k() to compute from known energy-radius pairs.
            Depends on electrode voltages and geometry of the VMI spectrometer.
        
        -------------------------------------------------------------------------
        BROADENING PARAMETERS
        -------------------------------------------------------------------------
        sigma_laser (float): Laser bandwidth (Gaussian σ) in eV.
            Represents the spectral width of the ionizing laser.
            Typical values: 0.001-0.1 eV depending on laser type.
            This contributes to the Gaussian component of energy broadening.
        
        T_beam (float): Molecular beam temperature in Kelvin.
            Used for Doppler broadening calculation.
            This is a GLOBAL parameter - same for all energy levels/orbitals,
            as it represents the physical temperature of the molecular beam.
            Typical values: 1-50 K for supersonic expansion, 300 K for effusive.
            The Doppler broadening is calculated using the full relativistic
            formula accounting for the thermal velocity distribution.
        
        tau_lifetimes (List[float] or float): Excited state lifetime(s) in femtoseconds.
            Determines Lorentzian (natural) linewidth via Heisenberg uncertainty:
            Γ = ℏ / (2τ)
            Can be specified as:
            - Single float: Same lifetime for all energy levels
            - List[float]: Different lifetime for each energy level/orbital
            Set to 0.0 to disable Lorentzian broadening.
            Typical values: 1-1000 fs depending on the electronic state.
        
        photon_energy (float): Photon energy in eV for Doppler broadening calculation.
            Used in the relativistic Doppler shift formula.
            Default: 0.0 (uses simplified non-relativistic approximation).
        
        target_mass (float): Mass of the target molecule/atom in amu.
            Used for Doppler broadening calculation (thermal velocity).
            Default: 28.0 (N₂ molecule).
        
        -------------------------------------------------------------------------
        GEOMETRY PARAMETERS
        -------------------------------------------------------------------------
        vol_sigma (Tuple[float, float, float]): Interaction volume size (σx, σy, σz) in mm.
            Represents the 3D Gaussian extent of the laser-molecule interaction region.
            Affects the spatial blurring of the VMI image.
            Typical values: (0.1-1.0, 0.1-1.0, 0.1-1.0) mm.
        
        polarization_vec (np.ndarray): Polarization direction in lab frame [x, y, z].
            Defines the quantization axis for angular distributions.
            Will be normalized to unit length.
            Default: [0, 1, 0] (Y-axis, vertical polarization).
        
        -------------------------------------------------------------------------
        DETECTOR PARAMETERS
        -------------------------------------------------------------------------
        img_res (int): Final image resolution in pixels (square image).
            Determines the output image size: img_res × img_res pixels.
            Typical values: 256, 512, 1024.
        
        pixel_size (float): Physical size of each pixel in mm.
            Total detector size = img_res × pixel_size.
            Typical values: 0.01-0.1 mm depending on detector.
        
        psf_fwhm (float): Point spread function full-width at half-maximum in mm.
            Models the optical/detector blurring of the image.
            Includes contributions from MCP pore size, phosphor screen,
            and camera optics.
            Typical values: 0.1-0.5 mm.
        
        supersample_factor (int): Supersampling factor for PSF convolution.
            The simulation first creates a high-resolution grid with
            (img_res × supersample_factor)² pixels, applies PSF convolution,
            then downsamples to final resolution by block-summing.
            This ensures accurate PSF application without aliasing artifacts.
            Higher values give more accurate PSF but use more memory.
            Typical values: 2-8. Value of 4 is usually sufficient.
        
        -------------------------------------------------------------------------
        ELECTRONICS/NOISE PARAMETERS
        -------------------------------------------------------------------------
        dark_rate (float): Dark current rate in counts per pixel.
            Models thermal electron emission from the detector (MCP/CCD).
            Added as Poisson-distributed noise to each pixel.
            Represents the average number of dark counts per pixel per exposure.
            Typical values: 0.01-1.0 counts/pixel depending on detector
            temperature and exposure time.
        
        readout_sigma (float): Readout noise standard deviation in counts.
            Models electronic noise from the camera readout circuitry.
            Added as Gaussian-distributed noise to each pixel.
            Typical values: 1-20 counts depending on camera quality.
        
        readout_offset (float): Readout offset (bias) in counts.
            Constant offset added to all pixels (camera bias level).
            Typical values: 50-500 counts.
        
        -------------------------------------------------------------------------
        BACKGROUND GAS PARAMETERS
        -------------------------------------------------------------------------
        bg_rate (float): Background gas contribution rate (fraction of N_events).
            Fraction of total events that come from background gas ionization.
            Example: 0.01 means 1% of events are background.
        
        bg_energy (float): Mean energy of background gas electrons in eV.
            Background electrons are typically low-energy and isotropic.
            Default: 0.1 eV.
        
        bg_sigma (float): Energy spread (Gaussian σ) of background electrons in eV.
            Default: 0.05 eV.
    """
    # =========================================================================
    # Particle properties
    # =========================================================================
    mass: float = electron_mass / AMU_TO_KG  # electron mass in amu
    
    # =========================================================================
    # Energy levels and angular distributions
    # =========================================================================
    E_centers: List[float] = field(default_factory=lambda: [1.0])  # eV
    Betas: List[float] = field(default_factory=lambda: [2.0])  # anisotropy
    branching_ratios: List[float] = field(default_factory=lambda: [1.0])
    
    # =========================================================================
    # Simulation parameters
    # =========================================================================
    N_events: int = 100000
    vmi_k: float = 0.01  # mm / (m/s), needs calibration for specific setup
    
    # =========================================================================
    # Broadening parameters
    # =========================================================================
    sigma_laser: float = 0.01  # eV (Gaussian laser bandwidth)
    T_beam: float = 10.0  # K (molecular beam temperature - GLOBAL for all orbitals)
    tau_lifetimes: Union[float, List[float]] = 0.0  # fs (can be per-orbital)
    photon_energy: float = 0.0  # eV (for Doppler calculation, 0 = simplified)
    target_mass: float = 28.0  # amu (target molecule mass for Doppler)
    
    # =========================================================================
    # Geometry parameters
    # =========================================================================
    vol_sigma: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # mm (σx, σy, σz)
    polarization_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    
    # =========================================================================
    # Detector parameters
    # =========================================================================
    img_res: int = 512  # pixels
    pixel_size: float = 0.05  # mm per pixel
    psf_fwhm: float = 0.2  # mm
    supersample_factor: int = 4  # for accurate PSF convolution
    
    # =========================================================================
    # Electronics/noise parameters
    # =========================================================================
    dark_rate: float = 0.1  # counts/pixel (Poisson dark current)
    readout_sigma: float = 5.0  # counts (Gaussian readout noise)
    readout_offset: float = 100.0  # counts (bias level)
    
    # =========================================================================
    # Background gas parameters
    # =========================================================================
    bg_rate: float = 0.01  # fraction of N_events from background
    bg_energy: float = 0.1  # eV (mean energy of background electrons)
    bg_sigma: float = 0.05  # eV (energy spread of background electrons)
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize polarization vector
        self.polarization_vec = np.array(self.polarization_vec, dtype=float)
        self.polarization_vec = self.polarization_vec / np.linalg.norm(self.polarization_vec)
        
        # Normalize branching ratios
        total = sum(self.branching_ratios)
        self.branching_ratios = [r / total for r in self.branching_ratios]
        
        # Validate lengths match
        n_levels = len(self.E_centers)
        if len(self.Betas) != n_levels:
            raise ValueError(f"Betas length ({len(self.Betas)}) must match E_centers ({n_levels})")
        if len(self.branching_ratios) != n_levels:
            raise ValueError(f"branching_ratios length ({len(self.branching_ratios)}) must match E_centers ({n_levels})")
        
        # Handle tau_lifetimes: convert single value to list if needed
        if isinstance(self.tau_lifetimes, (int, float)):
            self.tau_lifetimes = [float(self.tau_lifetimes)] * n_levels
        elif len(self.tau_lifetimes) != n_levels:
            raise ValueError(f"tau_lifetimes length ({len(self.tau_lifetimes)}) must match E_centers ({n_levels})")
    
    @property
    def detector_size_mm(self) -> float:
        """Total detector size in mm."""
        return self.img_res * self.pixel_size
    
    @property
    def psf_sigma(self) -> float:
        """PSF standard deviation in mm (from FWHM)."""
        return self.psf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    @staticmethod
    def calculate_vmi_k(E_max_eV: float, r_max_mm: float, mass_amu: float = None) -> float:
        """
        Calculate vmi_k coefficient to map maximum energy to desired radius.
        
        This helper function computes the velocity-to-position conversion factor
        such that electrons with energy E_max will hit the detector at radius r_max.
        
        Args:
            E_max_eV: Maximum electron energy in eV
            r_max_mm: Desired radius on detector for E_max electrons (mm)
            mass_amu: Particle mass in amu (default: electron mass)
        
        Returns:
            vmi_k coefficient in mm/(m/s)
        
        Example:
            # For 2 eV electrons to hit at 10 mm radius:
            >>> vmi_k = Config.calculate_vmi_k(E_max_eV=2.0, r_max_mm=10.0)
        """
        if mass_amu is None:
            mass_amu = electron_mass / AMU_TO_KG
        
        # v = sqrt(2E/m)
        mass_kg = mass_amu * AMU_TO_KG
        E_joule = E_max_eV * EV_TO_JOULE
        v_max = np.sqrt(2.0 * E_joule / mass_kg)  # m/s
        
        # r = k * v → k = r / v
        vmi_k = r_max_mm / v_max
        
        return vmi_k
    
    def get_expected_radius(self, E_eV: float) -> float:
        """
        Calculate expected detector radius for given energy.
        
        Args:
            E_eV: Electron energy in eV
        
        Returns:
            Expected radius on detector in mm
        """
        mass_kg = self.mass * AMU_TO_KG
        E_joule = E_eV * EV_TO_JOULE
        v = np.sqrt(2.0 * E_joule / mass_kg)
        return self.vmi_k * v


# =============================================================================
# Physics Source Class
# =============================================================================
class PhysicsSource:
    """
    Generates 3D electron/ion cloud data based on quantum mechanics.
    
    Handles:
        - Energy distribution sampling (Gaussian + Lorentzian broadening)
        - Angular distribution sampling (β parameter, rejection sampling)
        - Interaction volume sampling
        - Coordinate transformations (intrinsic → lab frame)
    """
    
    def __init__(self, config: Config):
        """
        Initialize physics source with configuration.
        
        Args:
            config: Configuration object with all parameters
        """
        self.cfg = config
        self._rotation_matrix = self._compute_rotation_matrix()
    
    def _compute_rotation_matrix(self) -> np.ndarray:
        """
        Compute rotation matrix from intrinsic frame (Z' = polarization) to lab frame.
        
        In the intrinsic frame, Z' is along the polarization direction.
        This matrix rotates vectors from intrinsic to lab coordinates.
        
        Returns:
            3x3 rotation matrix
        """
        # Target: polarization vector in lab frame
        z_prime = self.cfg.polarization_vec
        
        # Find a perpendicular vector for x'
        if abs(z_prime[0]) < 0.9:
            x_prime = np.cross(z_prime, np.array([1, 0, 0]))
        else:
            x_prime = np.cross(z_prime, np.array([0, 1, 0]))
        x_prime = x_prime / np.linalg.norm(x_prime)
        
        # y' completes the right-handed system
        y_prime = np.cross(z_prime, x_prime)
        y_prime = y_prime / np.linalg.norm(y_prime)
        
        # Rotation matrix: columns are the new basis vectors
        R = np.column_stack([x_prime, y_prime, z_prime])
        return R
    
    def calculate_doppler_broadening(self, E_center: float) -> float:
        """
        Calculate Doppler broadening for a given energy level.
        
        This implements the full Doppler broadening calculation for photoelectron
        spectroscopy, accounting for the thermal velocity distribution of the
        target molecules in the molecular beam.
        
        The Doppler effect in photoelectron spectroscopy arises because:
        1. The target molecule has thermal velocity v_mol from the beam temperature
        2. This causes a Doppler shift in the photon energy seen by the molecule
        3. The photoelectron energy is affected by both the photon Doppler shift
           and the recoil/momentum transfer
        
        For a molecule moving with velocity v_mol along the laser propagation direction:
        - First-order Doppler shift: ΔE_photon = E_photon × (v_mol / c)
        - This directly affects the photoelectron kinetic energy
        
        The thermal velocity distribution is Maxwell-Boltzmann:
        - σ_v = sqrt(k_B × T / M) where M is the target molecule mass
        
        The resulting energy broadening (Gaussian σ) is:
        - σ_E_doppler = E_photon × σ_v / c  (for photon energy Doppler shift)
        
        Additionally, there's a contribution from the initial momentum of the molecule
        being transferred to the photoelectron, which gives:
        - σ_E_recoil ≈ sqrt(2 × E_electron × k_B × T × m_e / M)
        
        where m_e is electron mass and M is target mass.
        
        Args:
            E_center: Center energy of the photoelectron peak in eV
        
        Returns:
            Doppler broadening σ in eV (Gaussian standard deviation)
        """
        if self.cfg.T_beam <= 0:
            return 0.0
        
        # Thermal velocity spread of target molecules
        # σ_v = sqrt(k_B × T / M)
        target_mass_kg = self.cfg.target_mass * AMU_TO_KG
        sigma_v = np.sqrt(Boltzmann * self.cfg.T_beam / target_mass_kg)  # m/s
        
        sigma_doppler_total = 0.0
        
        # Contribution 1: Photon energy Doppler shift
        # When molecule moves toward/away from laser, it sees shifted photon energy
        # ΔE_photon / E_photon = v_mol / c (first-order Doppler)
        if self.cfg.photon_energy > 0:
            # σ_E = E_photon × (σ_v / c)
            sigma_photon_doppler = self.cfg.photon_energy * (sigma_v / C_M_S)
            sigma_doppler_total += sigma_photon_doppler**2
        
        # Contribution 2: Momentum transfer from molecule to electron
        # The electron inherits some of the molecule's momentum
        # For an electron with energy E_e, velocity v_e = sqrt(2 E_e / m_e)
        # The momentum transfer causes energy spread:
        # ΔE ≈ m_e × v_e × v_mol = sqrt(2 × m_e × E_e) × v_mol
        # σ_E_recoil = sqrt(2 × m_e × E_e) × σ_v
        if E_center > 0:
            electron_mass_kg = electron_mass
            # σ_E = sqrt(2 × m_e × E_e) × σ_v, with E_e in Joules
            E_center_J = E_center * EV_TO_JOULE
            sigma_recoil = np.sqrt(2 * electron_mass_kg * E_center_J) * sigma_v
            # Convert back to eV
            sigma_recoil_eV = sigma_recoil / EV_TO_JOULE
            sigma_doppler_total += sigma_recoil_eV**2
        
        return np.sqrt(sigma_doppler_total)
    
    def calculate_lifetime_broadening(self, level_idx: int) -> float:
        """
        Calculate Lorentzian (natural) linewidth from excited state lifetime.
        
        Uses the Heisenberg uncertainty principle:
        ΔE × Δt ≥ ℏ/2
        
        For an exponentially decaying state with lifetime τ:
        Γ (HWHM) = ℏ / (2τ)
        
        The full width at half maximum (FWHM) is 2Γ = ℏ/τ
        
        Args:
            level_idx: Index of the energy level
        
        Returns:
            Lorentzian HWHM (γ) in eV
        """
        tau = self.cfg.tau_lifetimes[level_idx]
        if tau <= 0:
            return 0.0
        
        # γ = ℏ / (2τ)
        # ℏ = 0.6582 eV·fs
        gamma = HBAR_EV_FS / (2.0 * tau)
        return gamma
    
    def calculate_gaussian_sigma(self, E_center: float) -> float:
        """
        Calculate total Gaussian broadening combining laser and Doppler contributions.
        
        The total Gaussian width is the quadrature sum of:
        - Laser bandwidth (σ_laser)
        - Doppler broadening (σ_doppler)
        
        σ_total = sqrt(σ_laser² + σ_doppler²)
        
        Args:
            E_center: Center energy of the peak in eV
        
        Returns:
            Total Gaussian σ in eV
        """
        sigma_laser = self.cfg.sigma_laser
        sigma_doppler = self.calculate_doppler_broadening(E_center)
        
        return np.sqrt(sigma_laser**2 + sigma_doppler**2)
    
    def sample_energy(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample particle energies and convert to velocity magnitudes.
        
        For each energy level, applies:
        - Gaussian broadening (laser + Doppler) - same T_beam for all levels
        - Lorentzian broadening (lifetime) - can be different per level
        
        The energy distribution is a Voigt profile (convolution of Gaussian and
        Lorentzian), which we approximate by adding independent Gaussian and
        Lorentzian samples.
        
        Args:
            N: Number of particles to generate
        
        Returns:
            Tuple of (velocity_magnitudes in m/s, level_indices)
        """
        # Allocate particles to energy levels based on branching ratios
        level_counts = np.random.multinomial(N, self.cfg.branching_ratios)
        
        velocities = []
        level_indices = []
        
        for level_idx, (E_center, n_particles) in enumerate(zip(self.cfg.E_centers, level_counts)):
            if n_particles == 0:
                continue
            
            # Calculate broadening for this level
            # Gaussian: laser + Doppler (T_beam is global, same for all levels)
            sigma_gauss = self.calculate_gaussian_sigma(E_center)
            
            # Lorentzian: lifetime (can be different per level)
            gamma_lorentz = self.calculate_lifetime_broadening(level_idx)
            
            # Sample Gaussian energy distribution
            if sigma_gauss > 0:
                E_gauss = np.random.normal(0, sigma_gauss, n_particles)
            else:
                E_gauss = np.zeros(n_particles)
            
            # Sample Lorentzian energy distribution (truncated Cauchy)
            if gamma_lorentz > 0:
                E_lorentz = self._sample_truncated_cauchy(gamma_lorentz, n_particles,
                                                          max_width=10*gamma_lorentz)
            else:
                E_lorentz = np.zeros(n_particles)
            
            # Final energy (Voigt-like profile from sum of Gaussian + Lorentzian)
            E_final = E_center + E_gauss + E_lorentz
            
            # Ensure positive energies
            E_final = np.maximum(E_final, 1e-10)
            
            # Convert energy to velocity: v = sqrt(2E/m)
            mass_kg = self.cfg.mass * AMU_TO_KG
            E_joule = E_final * EV_TO_JOULE
            v_mag = np.sqrt(2.0 * E_joule / mass_kg)  # m/s
            
            velocities.append(v_mag)
            level_indices.append(np.full(n_particles, level_idx, dtype=int))
        
        return np.concatenate(velocities), np.concatenate(level_indices)
    
    def _sample_truncated_cauchy(self, gamma: float, n: int, max_width: float) -> np.ndarray:
        """
        Sample from truncated Cauchy (Lorentzian) distribution.
        
        Args:
            gamma: HWHM of Lorentzian
            n: Number of samples
            max_width: Maximum deviation from center
        
        Returns:
            Array of samples
        """
        samples = []
        while len(samples) < n:
            # Standard Cauchy samples
            x = np.random.standard_cauchy(n - len(samples)) * gamma
            # Keep only those within bounds
            valid = np.abs(x) < max_width
            samples.extend(x[valid])
        return np.array(samples[:n])
    
    def sample_direction(self, N: int, level_indices: np.ndarray) -> np.ndarray:
        """
        Sample emission directions based on β parameters using rejection sampling.
        
        The angular distribution follows: I(θ) ∝ 1 + β * P₂(cos θ)
        where P₂(x) = (3x² - 1) / 2 is the second Legendre polynomial.
        
        Args:
            N: Number of particles
            level_indices: Array indicating which energy level each particle belongs to
        
        Returns:
            Direction unit vectors in lab frame (N x 3)
        """
        directions_lab = np.zeros((N, 3))
        
        for level_idx, beta in enumerate(self.cfg.Betas):
            mask = level_indices == level_idx
            n_level = np.sum(mask)
            if n_level == 0:
                continue
            
            # Sample angles in intrinsic frame (Z' = polarization)
            phi = np.random.uniform(0, 2*np.pi, n_level)
            cos_theta = self._sample_cos_theta_rejection(beta, n_level)
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian in intrinsic frame
            vx_prime = sin_theta * np.cos(phi)
            vy_prime = sin_theta * np.sin(phi)
            vz_prime = cos_theta
            
            # Stack into matrix (n_level x 3)
            v_intrinsic = np.column_stack([vx_prime, vy_prime, vz_prime])
            
            # Rotate to lab frame
            v_lab = v_intrinsic @ self._rotation_matrix.T
            
            directions_lab[mask] = v_lab
        
        return directions_lab
    
    def _sample_cos_theta_rejection(self, beta: float, n: int) -> np.ndarray:
        """
        Sample cos(θ) using rejection sampling for angular distribution.
        
        PDF: f(x) ∝ 1 + β * P₂(x) where x = cos(θ), P₂(x) = (3x² - 1) / 2
        
        Args:
            beta: Anisotropy parameter (-1 ≤ β ≤ 2)
            n: Number of samples
        
        Returns:
            Array of cos(θ) values
        """
        # Maximum value of the distribution for rejection sampling
        # f(x) = 1 + β * (3x² - 1) / 2
        # For β > 0: max at x = ±1, f(±1) = 1 + β
        # For β < 0: max at x = 0, f(0) = 1 - β/2
        if beta >= 0:
            f_max = 1 + beta
        else:
            f_max = 1 - beta / 2
        
        samples = []
        while len(samples) < n:
            # Generate candidates
            n_needed = n - len(samples)
            x = np.random.uniform(-1, 1, n_needed * 2)  # oversample
            u = np.random.uniform(0, f_max, n_needed * 2)
            
            # Evaluate PDF
            P2 = (3 * x**2 - 1) / 2
            f_x = 1 + beta * P2
            
            # Accept/reject
            accepted = x[u < f_x]
            samples.extend(accepted)
        
        return np.array(samples[:n])
    
    def sample_origin(self, N: int) -> np.ndarray:
        """
        Sample particle starting positions within interaction volume.
        
        The interaction volume is modeled as a 3D Gaussian distribution.
        
        Args:
            N: Number of particles
        
        Returns:
            Starting positions (N x 3) in mm
        """
        x0 = np.random.normal(0, self.cfg.vol_sigma[0], N)
        y0 = np.random.normal(0, self.cfg.vol_sigma[1], N)
        z0 = np.random.normal(0, self.cfg.vol_sigma[2], N)
        
        return np.column_stack([x0, y0, z0])
    
    def generate_particles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete particle data (positions and velocities).
        
        This is the main entry point for particle generation. It:
        1. Samples energies from the configured distribution (with broadening)
        2. Samples emission directions based on β parameters
        3. Samples starting positions within the interaction volume
        
        Returns:
            Tuple of (origins, velocities, level_indices)
            - origins: (N x 3) starting positions in mm
            - velocities: (N x 3) velocity vectors in m/s
            - level_indices: (N,) energy level index for each particle
        """
        N = self.cfg.N_events
        
        # Sample energies and convert to velocity magnitudes
        # (broadening is calculated per-level inside sample_energy)
        v_mag, level_indices = self.sample_energy(N)
        
        # Sample directions
        directions = self.sample_direction(N, level_indices)
        
        # Combine into velocity vectors
        velocities = v_mag[:, np.newaxis] * directions
        
        # Sample starting positions
        origins = self.sample_origin(N)
        
        return origins, velocities, level_indices


# =============================================================================
# VMI Instrument Class
# =============================================================================
class VMIInstrument:
    """
    Simulates VMI instrument response including projection and PSF.
    
    Handles:
        - 3D to 2D projection (Abel projection)
        - Point spread function convolution
        - Pixel binning and image formation
    """
    
    def __init__(self, config: Config):
        """
        Initialize VMI instrument with configuration.
        
        Args:
            config: Configuration object with all parameters
        """
        self.cfg = config
    
    def project_to_detector(self, origins: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """
        Project 3D particle trajectories to 2D detector positions.
        
        The VMI maps velocity to position: X_det = x0 + k * vx, Y_det = y0 + k * vy
        The z-component is lost (Abel projection).
        
        Args:
            origins: Starting positions (N x 3) in mm
            velocities: Velocity vectors (N x 3) in m/s
        
        Returns:
            Hit positions on detector (N x 2) in mm
        """
        # VMI projection: position = origin + k * velocity
        # k converts velocity (m/s) to displacement (mm)
        X_det = origins[:, 0] + self.cfg.vmi_k * velocities[:, 0]
        Y_det = origins[:, 1] + self.cfg.vmi_k * velocities[:, 1]
        
        return np.column_stack([X_det, Y_det])
    
    def apply_psf_and_grid(self, hits: np.ndarray) -> np.ndarray:
        """
        Apply PSF convolution and bin hits into pixel grid.
        
        Uses supersampling for accurate PSF convolution:
        1. Create high-resolution grid
        2. Histogram hits onto high-res grid
        3. Convolve with PSF
        4. Downsample to final resolution
        
        Args:
            hits: Hit positions (N x 2) in mm
        
        Returns:
            2D image array (img_res x img_res)
        """
        # High-resolution grid parameters
        supersample = self.cfg.supersample_factor
        high_res = self.cfg.img_res * supersample
        pixel_size_high = self.cfg.pixel_size / supersample
        
        # Detector extent (centered at origin)
        half_size = self.cfg.detector_size_mm / 2
        
        # Create high-resolution histogram
        bins = np.linspace(-half_size, half_size, high_res + 1)
        grid_high, _, _ = np.histogram2d(
            hits[:, 0], hits[:, 1],
            bins=[bins, bins]
        )
        
        # Apply PSF convolution
        # Convert PSF sigma from mm to high-res pixels
        psf_sigma_pixels = self.cfg.psf_sigma / pixel_size_high
        if psf_sigma_pixels > 0.5:  # Only convolve if PSF is significant
            grid_high = ndimage.gaussian_filter(grid_high, sigma=psf_sigma_pixels)
        
        # Downsample to final resolution (block sum)
        img_final = self._block_sum(grid_high, supersample)
        
        return img_final
    
    def _block_sum(self, arr: np.ndarray, block_size: int) -> np.ndarray:
        """
        Downsample array by summing blocks.
        
        Args:
            arr: Input array (must be divisible by block_size)
            block_size: Size of blocks to sum
        
        Returns:
            Downsampled array
        """
        shape = (arr.shape[0] // block_size, block_size,
                 arr.shape[1] // block_size, block_size)
        return arr.reshape(shape).sum(axis=(1, 3))
    
    def process(self, origins: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """
        Complete instrument processing pipeline.
        
        Args:
            origins: Starting positions (N x 3) in mm
            velocities: Velocity vectors (N x 3) in m/s
        
        Returns:
            Ideal (noise-free) detector image
        """
        hits = self.project_to_detector(origins, velocities)
        image = self.apply_psf_and_grid(hits)
        return image


# =============================================================================
# Camera Electronics Class
# =============================================================================
class CameraElectronics:
    """
    Simulates camera electronics and noise sources.
    
    Handles:
        - Background gas contribution
        - Dark current (Poisson)
        - Readout noise (Gaussian)
    """
    
    def __init__(self, config: Config, physics_source: Optional[PhysicsSource] = None,
                 instrument: Optional[VMIInstrument] = None):
        """
        Initialize camera electronics.
        
        Args:
            config: Configuration object
            physics_source: PhysicsSource for background generation (optional)
            instrument: VMIInstrument for background projection (optional)
        """
        self.cfg = config
        self.physics = physics_source
        self.instrument = instrument
    
    def add_background_gas(self, img: np.ndarray) -> np.ndarray:
        """
        Add background gas contribution.
        
        Generates low-energy, isotropic electrons from background gas ionization
        and adds them to the image. Background electrons typically come from
        residual gas in the vacuum chamber being ionized by the laser.
        
        The background is characterized by:
        - bg_rate: Fraction of total events from background (set in Config)
        - bg_energy: Mean energy of background electrons in eV (set in Config)
        - bg_sigma: Energy spread of background electrons in eV (set in Config)
        - Isotropic angular distribution (β = 0)
        
        Args:
            img: Input image
        
        Returns:
            Image with background added
        """
        if self.cfg.bg_rate <= 0 or self.physics is None or self.instrument is None:
            return img
        
        # Create temporary config for background using user-specified parameters
        bg_config = Config(
            mass=self.cfg.mass,
            E_centers=[self.cfg.bg_energy],  # User-specified background energy
            Betas=[0.0],  # Isotropic (background gas has no preferred direction)
            branching_ratios=[1.0],
            N_events=int(self.cfg.N_events * self.cfg.bg_rate),
            vmi_k=self.cfg.vmi_k,
            sigma_laser=self.cfg.bg_sigma,  # User-specified background energy spread
            T_beam=0.0,  # No Doppler for background (already thermalized)
            tau_lifetimes=0.0,  # No lifetime broadening for background
            vol_sigma=self.cfg.vol_sigma,
            polarization_vec=self.cfg.polarization_vec,
            img_res=self.cfg.img_res,
            pixel_size=self.cfg.pixel_size,
            psf_fwhm=self.cfg.psf_fwhm,
            supersample_factor=self.cfg.supersample_factor
        )
        
        # Generate background particles
        bg_physics = PhysicsSource(bg_config)
        origins, velocities, _ = bg_physics.generate_particles()
        
        # Project to detector
        bg_instrument = VMIInstrument(bg_config)
        img_bg = bg_instrument.process(origins, velocities)
        
        return img + img_bg
    
    def add_dark_current(self, img: np.ndarray) -> np.ndarray:
        """
        Add dark current noise (Poisson distributed).
        
        Args:
            img: Input image
        
        Returns:
            Image with dark current added
        """
        if self.cfg.dark_rate <= 0:
            return img
        
        dark_noise = np.random.poisson(self.cfg.dark_rate, img.shape)
        return img + dark_noise
    
    def add_readout_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Add readout noise (Gaussian) and offset.
        
        Args:
            img: Input image
        
        Returns:
            Image with readout noise and offset
        """
        readout_noise = np.random.normal(
            self.cfg.readout_offset,
            self.cfg.readout_sigma,
            img.shape
        )
        return img + readout_noise
    
    def process(self, img: np.ndarray, add_background: bool = True) -> np.ndarray:
        """
        Complete electronics processing pipeline.
        
        Args:
            img: Ideal detector image
            add_background: Whether to add background gas
        
        Returns:
            Final image with all noise sources
        """
        if add_background:
            img = self.add_background_gas(img)
        img = self.add_dark_current(img)
        img = self.add_readout_noise(img)
        return img


# =============================================================================
# Main Simulation Function
# =============================================================================
def run_simulation(config: Config, add_noise: bool = True,
                   add_background: bool = True,
                   return_3d_data: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Run complete VMI forward simulation.
    
    Args:
        config: Simulation configuration
        add_noise: Whether to add electronic noise
        add_background: Whether to add background gas
        return_3d_data: If True, include 3D velocity distribution in metadata
    
    Returns:
        Tuple of (final_image, metadata_dict)
        If return_3d_data=True, metadata includes:
            - 'velocities': (N x 3) velocity vectors in m/s
            - 'origins': (N x 3) starting positions in mm
            - 'level_indices': (N,) energy level index for each particle
            - 'velocity_3d_hist': 3D histogram of velocity distribution
            - 'velocity_bins': bin edges for the 3D histogram
    """
    # Initialize modules
    physics = PhysicsSource(config)
    instrument = VMIInstrument(config)
    camera = CameraElectronics(config, physics, instrument)
    
    # Generate particles
    origins, velocities, level_indices = physics.generate_particles()
    
    # Project to detector
    image_ideal = instrument.process(origins, velocities)
    
    # Add noise
    if add_noise:
        final_image = camera.process(image_ideal, add_background=add_background)
    else:
        final_image = image_ideal
    
    # Collect metadata
    metadata = {
        'N_events': config.N_events,
        'E_centers': config.E_centers,
        'Betas': config.Betas,
        'sigma_laser': config.sigma_laser,
        'vmi_k': config.vmi_k,
        'psf_fwhm': config.psf_fwhm,
        'image_ideal_sum': np.sum(image_ideal),
        'image_final_sum': np.sum(final_image),
    }
    
    # Optionally include 3D data for visualization
    if return_3d_data:
        metadata['velocities'] = velocities
        metadata['origins'] = origins
        metadata['level_indices'] = level_indices
        
        # Create 3D velocity histogram
        # Determine velocity range from maximum energy
        E_max = max(config.E_centers) * 1.5  # Add margin
        mass_kg = config.mass * AMU_TO_KG
        v_max = np.sqrt(2.0 * E_max * EV_TO_JOULE / mass_kg)
        
        # Create 3D histogram
        n_bins_3d = 128  # Resolution for 3D histogram
        bins_3d = np.linspace(-v_max, v_max, n_bins_3d + 1)
        hist_3d, edges = np.histogramdd(
            velocities,
            bins=[bins_3d, bins_3d, bins_3d]
        )
        metadata['velocity_3d_hist'] = hist_3d
        metadata['velocity_bins'] = bins_3d
        metadata['v_max'] = v_max
    
    return final_image, metadata


def create_3d_slice_visualization(velocities: np.ndarray, config: Config,
                                   n_bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 2D slices through the 3D velocity distribution for visualization.
    
    This function creates three orthogonal slices through the center of the
    3D velocity distribution, showing the XY, XZ, and YZ planes.
    
    Args:
        velocities: (N x 3) velocity vectors in m/s
        config: Configuration object
        n_bins: Number of bins for each dimension
    
    Returns:
        Tuple of (slice_xy, slice_xz, slice_yz) - 2D arrays
        Each slice is a 2D histogram of velocities in that plane
    """
    # Determine velocity range
    E_max = max(config.E_centers) * 1.5
    mass_kg = config.mass * AMU_TO_KG
    v_max = np.sqrt(2.0 * E_max * EV_TO_JOULE / mass_kg)
    
    bins = np.linspace(-v_max, v_max, n_bins + 1)
    
    # XY slice (vz ≈ 0) - this is what the detector sees after Abel projection
    # For a true slice, we'd filter by vz, but for visualization we project
    slice_xy, _, _ = np.histogram2d(velocities[:, 0], velocities[:, 1], bins=[bins, bins])
    
    # XZ slice (vy ≈ 0)
    slice_xz, _, _ = np.histogram2d(velocities[:, 0], velocities[:, 2], bins=[bins, bins])
    
    # YZ slice (vx ≈ 0)
    slice_yz, _, _ = np.histogram2d(velocities[:, 1], velocities[:, 2], bins=[bins, bins])
    
    return slice_xy, slice_xz, slice_yz, v_max


def visualize_3d_distribution(velocities: np.ndarray, config: Config,
                               projected_image: np.ndarray,
                               title: str = "3D Velocity Distribution",
                               save_path: Optional[str] = None):
    """
    Visualize the 3D velocity distribution alongside the projected 2D image.
    
    Creates a figure showing:
    - Top row: XY, XZ, YZ velocity slices (the "real" 3D distribution)
    - Bottom row: Projected 2D image (what the detector sees)
    
    This allows comparison between the true 3D distribution and the
    Abel-projected 2D image.
    
    Args:
        velocities: (N x 3) velocity vectors in m/s
        config: Configuration object
        projected_image: 2D projected image from VMI
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Create velocity slices
    slice_xy, slice_xz, slice_yz, v_max = create_3d_slice_visualization(velocities, config)
    
    # Convert velocity to mm using vmi_k for consistent comparison
    r_max = config.vmi_k * v_max
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Velocity extent in mm (using vmi_k conversion)
    v_extent = [-r_max, r_max, -r_max, r_max]
    
    # Top row: 3D velocity distribution slices
    # XY slice (projection along Z - this is what Abel inversion recovers)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(slice_xy.T, origin='lower', extent=v_extent, cmap='hot', aspect='equal')
    ax1.set_xlabel('X (mm equivalent)')
    ax1.set_ylabel('Y (mm equivalent)')
    ax1.set_title('3D Distribution: XY Projection\n(Sum over Vz - True distribution)')
    plt.colorbar(im1, ax=ax1, label='Counts')
    
    # XZ slice
    ax2 = axes[0, 1]
    im2 = ax2.imshow(slice_xz.T, origin='lower', extent=v_extent, cmap='hot', aspect='equal')
    ax2.set_xlabel('X (mm equivalent)')
    ax2.set_ylabel('Z (mm equivalent)')
    ax2.set_title('3D Distribution: XZ Projection\n(Sum over Vy)')
    plt.colorbar(im2, ax=ax2, label='Counts')
    
    # YZ slice
    ax3 = axes[0, 2]
    im3 = ax3.imshow(slice_yz.T, origin='lower', extent=v_extent, cmap='hot', aspect='equal')
    ax3.set_xlabel('Y (mm equivalent)')
    ax3.set_ylabel('Z (mm equivalent)')
    ax3.set_title('3D Distribution: YZ Projection\n(Sum over Vx)')
    plt.colorbar(im3, ax=ax3, label='Counts')
    
    # Bottom row: Projected 2D image and comparisons
    # Detector image
    det_extent = [-config.detector_size_mm/2, config.detector_size_mm/2,
                  -config.detector_size_mm/2, config.detector_size_mm/2]
    
    ax4 = axes[1, 0]
    im4 = ax4.imshow(projected_image.T, origin='lower', extent=det_extent, cmap='hot', aspect='equal')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title('Detector Image (2D Projection)\n(Abel-projected, what camera sees)')
    plt.colorbar(im4, ax=ax4, label='Counts')
    
    # Log scale of detector image
    ax5 = axes[1, 1]
    img_log = np.log10(np.maximum(projected_image, 1))
    im5 = ax5.imshow(img_log.T, origin='lower', extent=det_extent, cmap='hot', aspect='equal')
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    ax5.set_title('Detector Image (Log scale)')
    plt.colorbar(im5, ax=ax5, label='log₁₀(Counts)')
    
    # Radial profile comparison
    ax6 = axes[1, 2]
    
    # Calculate radial profiles
    # For 3D XY slice
    center_3d = slice_xy.shape[0] // 2
    y_3d, x_3d = np.ogrid[:slice_xy.shape[0], :slice_xy.shape[1]]
    r_3d = np.sqrt((x_3d - center_3d)**2 + (y_3d - center_3d)**2)
    r_3d_flat = r_3d.flatten()
    slice_xy_flat = slice_xy.flatten()
    
    # Bin the radial profile
    r_bins = np.arange(0, center_3d, 1)
    r_profile_3d = np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins) - 1):
        mask = (r_3d_flat >= r_bins[i]) & (r_3d_flat < r_bins[i+1])
        if np.sum(mask) > 0:
            r_profile_3d[i] = np.mean(slice_xy_flat[mask])
    
    # For detector image
    center_det = projected_image.shape[0] // 2
    y_det, x_det = np.ogrid[:projected_image.shape[0], :projected_image.shape[1]]
    r_det = np.sqrt((x_det - center_det)**2 + (y_det - center_det)**2)
    r_det_flat = r_det.flatten()
    det_flat = projected_image.flatten()
    
    r_bins_det = np.arange(0, center_det, 1)
    r_profile_det = np.zeros(len(r_bins_det) - 1)
    for i in range(len(r_bins_det) - 1):
        mask = (r_det_flat >= r_bins_det[i]) & (r_det_flat < r_bins_det[i+1])
        if np.sum(mask) > 0:
            r_profile_det[i] = np.mean(det_flat[mask])
    
    # Convert radius to mm
    r_mm_3d = (r_bins[:-1] + 0.5) * (2 * r_max / slice_xy.shape[0])
    r_mm_det = (r_bins_det[:-1] + 0.5) * config.pixel_size
    
    # Normalize for comparison
    r_profile_3d_norm = r_profile_3d / np.max(r_profile_3d) if np.max(r_profile_3d) > 0 else r_profile_3d
    r_profile_det_norm = r_profile_det / np.max(r_profile_det) if np.max(r_profile_det) > 0 else r_profile_det
    
    ax6.plot(r_mm_3d, r_profile_3d_norm, 'b-', label='3D XY (True)', linewidth=2)
    ax6.plot(r_mm_det, r_profile_det_norm, 'r--', label='2D Detector', linewidth=2)
    ax6.set_xlabel('Radius (mm)')
    ax6.set_ylabel('Normalized Intensity')
    ax6.set_title('Radial Profile Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, min(r_max, config.detector_size_mm/2))
    
    # Add expected energy positions
    for E in config.E_centers:
        r_expected = config.get_expected_radius(E)
        ax6.axvline(r_expected, color='g', linestyle=':', alpha=0.7)
        ax6.text(r_expected, 0.95, f'{E} eV', rotation=90, va='top', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_simulation(image: np.ndarray, config: Config, 
                         title: str = "VMI Simulation", 
                         save_path: Optional[str] = None):
    """
    Visualize simulation result.
    
    Args:
        image: Simulated image
        config: Configuration used
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1 = axes[0]
    extent = [-config.detector_size_mm/2, config.detector_size_mm/2,
              -config.detector_size_mm/2, config.detector_size_mm/2]
    im1 = ax1.imshow(image.T, origin='lower', extent=extent, cmap='hot')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title(f'{title} - Linear')
    plt.colorbar(im1, ax=ax1, label='Counts')
    
    # Log scale
    ax2 = axes[1]
    img_log = np.log10(np.maximum(image, 1))
    im2 = ax2.imshow(img_log.T, origin='lower', extent=extent, cmap='hot')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title(f'{title} - Log scale')
    plt.colorbar(im2, ax=ax2, label='log₁₀(Counts)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# Example Usage and Testing
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VMI FORWARD SIMULATION - DEMONSTRATION OF ALL PARAMETERS")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Define energy levels and their properties
    # =========================================================================
    # E_centers: List of photoelectron kinetic energies in eV
    #   - Each entry represents a distinct peak from a different orbital/state
    #   - Example: ionization from different molecular orbitals
    E_centers = [0.5, 1.0, 2.0]  # Three energy levels in eV
    
    # Betas: Anisotropy parameter for each energy level
    #   - β = 2: Parallel transition (cos²θ, emission along polarization)
    #   - β = 0: Isotropic emission
    #   - β = -1: Perpendicular transition (sin²θ)
    #   - Must have same length as E_centers
    Betas = [2.0, 0.0, -0.5]  # Different anisotropies for each level
    
    # branching_ratios: Relative intensities of each peak
    #   - Will be normalized to sum to 1.0
    #   - Represents relative ionization cross-sections
    branching_ratios = [0.3, 0.5, 0.2]  # Middle peak is strongest
    
    # =========================================================================
    # STEP 2: Define broadening parameters
    # =========================================================================
    # sigma_laser: Laser bandwidth (Gaussian σ) in eV
    #   - Contributes to Gaussian energy broadening
    #   - Typical: 0.001-0.1 eV depending on laser type
    sigma_laser = 0.015  # 15 meV laser bandwidth
    
    # T_beam: Molecular beam temperature in Kelvin
    #   - GLOBAL parameter: same for ALL energy levels (physical beam temperature)
    #   - Used for Doppler broadening calculation
    #   - Typical: 1-50 K for supersonic expansion, 300 K for effusive beam
    T_beam = 10.0  # 10 K cold molecular beam
    
    # tau_lifetimes: Excited state lifetime(s) in femtoseconds
    #   - Can be DIFFERENT for each energy level (per-orbital property)
    #   - Determines Lorentzian (natural) linewidth: Γ = ℏ/(2τ)
    #   - Can specify as single float (same for all) or list (per-level)
    #   - Set to 0.0 to disable lifetime broadening
    tau_lifetimes = [100.0, 50.0, 200.0]  # Different lifetimes per orbital (fs)
    # Alternative: tau_lifetimes = 100.0  # Same lifetime for all levels
    
    # photon_energy: Ionizing photon energy in eV
    #   - Used for Doppler broadening calculation (photon Doppler shift)
    #   - Set to 0.0 to use simplified approximation
    photon_energy = 21.2  # He I line (21.2 eV)
    
    # target_mass: Mass of target molecule in amu
    #   - Used for Doppler broadening (thermal velocity calculation)
    target_mass = 28.0  # N₂ molecule
    
    # =========================================================================
    # STEP 3: Define VMI calibration
    # =========================================================================
    # vmi_k: Velocity-to-radius conversion coefficient in mm/(m/s)
    #   - Maps particle velocity to detector position: r = k × v
    #   - Depends on VMI electrode voltages and geometry
    #   - Use calculate_vmi_k() to compute from known energy-radius pair
    E_max = max(E_centers)
    r_max_mm = 20.0  # Desired radius for maximum energy electrons
    vmi_k = Config.calculate_vmi_k(E_max_eV=E_max, r_max_mm=r_max_mm)
    print(f"\nVMI calibration: vmi_k = {vmi_k:.4e} mm/(m/s)")
    print(f"  (Maps {E_max} eV electrons to {r_max_mm} mm radius)")
    
    # =========================================================================
    # STEP 4: Define detector parameters
    # =========================================================================
    # img_res: Image resolution in pixels (square image)
    img_res = 512  # 512 × 512 pixel image
    
    # pixel_size: Physical size of each pixel in mm
    #   - Total detector size = img_res × pixel_size
    pixel_size = 0.1  # 0.1 mm/pixel → 51.2 mm total detector size
    
    # psf_fwhm: Point spread function FWHM in mm
    #   - Models optical/detector blurring (MCP pores, phosphor, camera)
    #   - Set to 0.0 for ideal (no blurring)
    psf_fwhm = 0.0  # 0.3 mm PSF FWHM
    
    # supersample_factor: Supersampling factor for PSF convolution
    #   - Creates high-res grid (img_res × supersample_factor)² for PSF
    #   - Then downsamples by block-summing to final resolution
    #   - Ensures accurate PSF without aliasing artifacts
    #   - Higher = more accurate but more memory
    #   - Typical: 2-8, value of 4 is usually sufficient
    supersample_factor = 4
    
    # =========================================================================
    # STEP 5: Define geometry parameters
    # =========================================================================
    # vol_sigma: Interaction volume size (σx, σy, σz) in mm
    #   - 3D Gaussian extent of laser-molecule interaction region
    #   - Set to (0,0,0) for point source (ideal)
    vol_sigma = (0.0, 0.0, 0.0)  # Elongated along z (laser propagation)
    
    # polarization_vec: Laser polarization direction [x, y, z]
    #   - Defines quantization axis for angular distributions
    #   - Will be normalized to unit length
    polarization_vec = [0, 1, 0]  # Vertical polarization (Y-axis)
    
    # =========================================================================
    # STEP 6: Define electronics/noise parameters
    # =========================================================================
    # dark_rate: Dark current rate in counts per pixel
    #   - Models thermal electron emission from detector (MCP/CCD)
    #   - Added as Poisson-distributed noise
    #   - Represents average dark counts per pixel per exposure
    #   - Typical: 0.01-1.0 counts/pixel
    dark_rate = 0.1  # 0.1 counts/pixel average dark current
    
    # readout_sigma: Readout noise standard deviation in counts
    #   - Models electronic noise from camera readout
    #   - Added as Gaussian-distributed noise
    #   - Typical: 1-20 counts
    readout_sigma = 5.0  # 5 counts readout noise σ
    
    # readout_offset: Camera bias level in counts
    #   - Constant offset added to all pixels
    #   - Typical: 50-500 counts
    readout_offset = 100.0  # 100 counts bias
    
    # =========================================================================
    # STEP 7: Define background gas parameters
    # =========================================================================
    # bg_rate: Background gas contribution (fraction of N_events)
    #   - Fraction of events from residual gas ionization
    #   - Example: 0.01 = 1% background
    bg_rate = 0.02  # 2% background contribution
    
    # bg_energy: Mean energy of background electrons in eV
    #   - Background electrons are typically low-energy
    bg_energy = 0.15  # 150 meV mean background energy
    
    # bg_sigma: Energy spread of background electrons in eV
    #   - Gaussian σ for background energy distribution
    bg_sigma = 0.08  # 80 meV background energy spread
    
    # =========================================================================
    # STEP 8: Define simulation size
    # =========================================================================
    # N_events: Total number of particles to simulate
    #   - Higher = better statistics but longer computation
    #   - Typical: 1e5 for quick tests, 1e7-1e8 for publication quality
    N_events = int(1e6)  # 1 million particles
    
    # =========================================================================
    # CREATE CONFIGURATION
    # =========================================================================
    print("\n" + "-" * 70)
    print("CREATING SIMULATION CONFIGURATION")
    print("-" * 70)
    
    config = Config(
        # Energy levels and angular distributions
        E_centers=E_centers,
        Betas=Betas,
        branching_ratios=branching_ratios,
        
        # Simulation size
        N_events=N_events,
        
        # VMI calibration
        vmi_k=vmi_k,
        
        # Broadening parameters
        sigma_laser=sigma_laser,
        T_beam=T_beam,  # GLOBAL: same for all orbitals
        tau_lifetimes=tau_lifetimes,  # PER-ORBITAL: can be different
        photon_energy=photon_energy,
        target_mass=target_mass,
        
        # Geometry
        vol_sigma=vol_sigma,
        polarization_vec=polarization_vec,
        
        # Detector
        img_res=img_res,
        pixel_size=pixel_size,
        psf_fwhm=psf_fwhm,
        supersample_factor=supersample_factor,
        
        # Electronics/noise
        dark_rate=dark_rate,
        readout_sigma=readout_sigma,
        readout_offset=readout_offset,
        
        # Background gas
        bg_rate=bg_rate,
        bg_energy=bg_energy,
        bg_sigma=bg_sigma,
    )
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Energy levels: {config.E_centers} eV")
    print(f"  Beta values: {config.Betas}")
    print(f"  Branching ratios: {config.branching_ratios}")
    print(f"  Tau lifetimes: {config.tau_lifetimes} fs")
    print(f"  T_beam (global): {config.T_beam} K")
    print(f"  Laser bandwidth: {config.sigma_laser} eV")
    print(f"  Photon energy: {config.photon_energy} eV")
    print(f"  Target mass: {config.target_mass} amu")
    print(f"  Detector size: {config.detector_size_mm:.1f} mm ({config.img_res} pixels)")
    print(f"  PSF FWHM: {config.psf_fwhm} mm")
    print(f"  Supersample factor: {config.supersample_factor}")
    print(f"  Dark rate: {config.dark_rate} counts/pixel")
    print(f"  Background: {config.bg_rate*100:.1f}% at {config.bg_energy} eV (σ={config.bg_sigma} eV)")
    
    # Print expected radii
    print("\nExpected radii on detector:")
    for E in config.E_centers:
        r_mm = config.get_expected_radius(E)
        r_px = r_mm / config.pixel_size
        print(f"  E = {E} eV: r = {r_mm:.2f} mm = {r_px:.1f} px")
    print(f"  Detector half-size: {config.detector_size_mm/2:.1f} mm")
    
    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    print("\n" + "-" * 70)
    print("RUNNING SIMULATION")
    print("-" * 70)
    
    # Run with noise and background
    image_noisy, meta_noisy = run_simulation(config, add_noise=True, add_background=True)
    print(f"Simulation complete!")
    print(f"  Ideal image sum: {meta_noisy['image_ideal_sum']:.0f} counts")
    print(f"  Final image sum: {meta_noisy['image_final_sum']:.0f} counts")
    
    # Also run without noise for comparison
    image_clean, meta_clean = run_simulation(config, add_noise=False, add_background=False)
    # =========================================================================
    # OPTIONAL: Reconstruction Test
    # =========================================================================
    RUN_RECONSTRUCTION = True  # Set to True to run reconstruction tests
    
    if RUN_RECONSTRUCTION:
        print("\n" + "=" * 70)
        print("RECONSTRUCTION TEST")
        print("=" * 70)
        
        # Import reconstruction modules
        from Abel_backward_reconstruction import (
            reconstruct_vmi_image,
            compare_reconstruction,
            visualize_reconstruction
        )
        from Abel_rbasex_reconstruction import reconstruct_rbasex
        
        # Store true parameters for comparison
        true_params = {
            'E_centers': config.E_centers,
            'Betas': config.Betas,
            'branching_ratios': config.branching_ratios,
            'sigma_laser': config.sigma_laser
        }
        
        print("\nTrue simulation parameters:")
        print(f"  Energy levels: {true_params['E_centers']} eV")
        print(f"  Beta values: {true_params['Betas']}")
        print(f"  Branching ratios: {true_params['branching_ratios']}")
        print(f"  Laser sigma: {true_params['sigma_laser']} eV")
        
        # Run PhysicsBasedFitter reconstruction
        print("\n" + "-" * 60)
        print("Running PhysicsBasedFitter reconstruction...")
        print("-" * 60)
        physics_params, physics_metadata = reconstruct_vmi_image(
            image_clean, config=config, verbose=True
        )
        
        # Run rBasex reconstruction
        print("\n" + "-" * 60)
        print("Running rBasex reconstruction...")
        print("-" * 60)
        rbasex_params, rbasex_metadata = reconstruct_rbasex(
            image_clean, config=config, verbose=True
        )
        
        # Compare with ground truth
        compare_reconstruction(true_params, physics_params, config=config)
        
        # Generate 3D distribution image for visualization
        # This shows the "real" 3D velocity distribution before Abel projection
        print("\nGenerating 3D distribution for visualization...")
        _, meta_3d_recon = run_simulation(
            config, add_noise=False, add_background=False, return_3d_data=True
        )
        
        # Create XY slice of 3D velocity distribution (what Abel inversion recovers)
        slice_xy, _, _, v_max = create_3d_slice_visualization(
            meta_3d_recon['velocities'], config, n_bins=config.img_res
        )
        
        # Visualize reconstruction result with 3D distribution and rBasex comparison
        visualize_reconstruction(
            image_clean, physics_params, physics_metadata,
            config=config, true_params=true_params,
            image_3d=slice_xy,  # Pass the 3D distribution image
            rbasex_params=rbasex_params,  # Pass rBasex results for beta comparison
            rbasex_metadata=rbasex_metadata,  # Pass rBasex metadata for radial profile
            save_path="reconstruction_physics.png"
        )
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)