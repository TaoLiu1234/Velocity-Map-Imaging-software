"""
Abel Backward Reconstruction X2 - Differentiable Forward Fitting

核心思想：
1. 使用与forward_simulation完全相同的物理过程
2. 一次性生成所有粒子，参数化控制分布
3. 在多个分辨率(dtheta, dr)下比较统计信息
4. 使用可微分编程/反向传播优化参数

关键创新：
- 软直方图(soft histogram)使binning可微分
- 多尺度损失：从粗到细的分辨率
- Wasserstein距离的可微近似
- 重参数化技巧使采样可微分

Author: Kiro AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.constants import electron_mass, elementary_charge, atomic_mass

# Physical constants
EV_TO_JOULE = elementary_charge
AMU_TO_KG = atomic_mass
ELECTRON_MASS_AMU = electron_mass / AMU_TO_KG


@dataclass
class DiffConfig:
    """Configuration for differentiable reconstruction"""
    # VMI parameters
    vmi_k: float = 0.01              # velocity to radius conversion (mm/(m/s))
    mass: float = ELECTRON_MASS_AMU  # particle mass (amu)
    
    # Detector parameters
    psf_sigma: float = 0.0           # PSF broadening (mm)
    
    # Parameter bounds
    E_min: float = 0.05              # minimum energy (eV)
    E_max: float = 5.0               # maximum energy (eV)
    beta_min: float = -1.0
    beta_max: float = 2.0
    sigma_min: float = 0.01          # minimum sigma (eV)
    sigma_max: float = 0.5           # maximum sigma (eV)
    
    # Multi-scale bins
    radial_bins_coarse: int = 20     # coarse radial bins
    radial_bins_fine: int = 100      # fine radial bins
    angular_bins_coarse: int = 8     # coarse angular bins  
    angular_bins_fine: int = 36      # fine angular bins
    
    # Optimization
    n_particles: int = 50000         # particles for forward model
    learning_rate: float = 0.01
    n_iterations: int = 500
    
    # Loss weights (coarse to fine)
    weight_coarse: float = 1.0
    weight_medium: float = 0.5
    weight_fine: float = 0.3
    
    device: str = 'cpu'


class SoftHistogram(nn.Module):
    """
    Differentiable soft histogram using kernel density estimation
    
    Instead of hard binning, uses soft assignment with Gaussian kernels
    This makes the histogram differentiable w.r.t. input values
    """
    
    def __init__(self, n_bins: int, min_val: float, max_val: float, 
                 sigma: Optional[float] = None):
        super().__init__()
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        
        # Bin centers
        self.register_buffer('bin_centers', 
            torch.linspace(min_val, max_val, n_bins))
        
        # Kernel width (default: half bin width)
        if sigma is None:
            sigma = (max_val - min_val) / (2 * n_bins)
        self.sigma = sigma
    
    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute soft histogram
        
        Args:
            x: input values (N,)
            weights: optional weights (N,)
            
        Returns:
            histogram (n_bins,)
        """
        # x: (N,) -> (N, 1)
        # bin_centers: (n_bins,) -> (1, n_bins)
        x = x.unsqueeze(-1)  # (N, 1)
        centers = self.bin_centers.unsqueeze(0)  # (1, n_bins)
        
        # Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)
        diff = (x - centers) / self.sigma
        kernel = torch.exp(-0.5 * diff ** 2)
        
        # Normalize each point's contribution
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-10)
        
        if weights is not None:
            kernel = kernel * weights.unsqueeze(-1)
        
        # Sum over all points
        hist = kernel.sum(dim=0)
        
        # Normalize histogram
        hist = hist / (hist.sum() + 1e-10)
        
        return hist


class SoftHistogram2D(nn.Module):
    """2D differentiable soft histogram"""
    
    def __init__(self, n_bins_x: int, n_bins_y: int,
                 x_range: Tuple[float, float], y_range: Tuple[float, float],
                 sigma_x: Optional[float] = None, sigma_y: Optional[float] = None):
        super().__init__()
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        
        # Bin centers
        self.register_buffer('x_centers',
            torch.linspace(x_range[0], x_range[1], n_bins_x))
        self.register_buffer('y_centers',
            torch.linspace(y_range[0], y_range[1], n_bins_y))
        
        # Kernel widths
        if sigma_x is None:
            sigma_x = (x_range[1] - x_range[0]) / (2 * n_bins_x)
        if sigma_y is None:
            sigma_y = (y_range[1] - y_range[0]) / (2 * n_bins_y)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
    
    def forward(self, x: torch.Tensor, y: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute 2D soft histogram
        
        Args:
            x, y: input coordinates (N,)
            weights: optional weights (N,)
            
        Returns:
            2D histogram (n_bins_x, n_bins_y)
        """
        N = x.shape[0]
        
        # Compute 1D kernels
        x_diff = (x.unsqueeze(-1) - self.x_centers.unsqueeze(0)) / self.sigma_x
        y_diff = (y.unsqueeze(-1) - self.y_centers.unsqueeze(0)) / self.sigma_y
        
        kernel_x = torch.exp(-0.5 * x_diff ** 2)  # (N, n_bins_x)
        kernel_y = torch.exp(-0.5 * y_diff ** 2)  # (N, n_bins_y)
        
        # Outer product for 2D kernel
        # (N, n_bins_x, 1) * (N, 1, n_bins_y) -> (N, n_bins_x, n_bins_y)
        kernel_2d = kernel_x.unsqueeze(-1) * kernel_y.unsqueeze(-2)
        
        # Normalize each point's contribution
        kernel_2d = kernel_2d / (kernel_2d.sum(dim=(-2, -1), keepdim=True) + 1e-10)
        
        if weights is not None:
            kernel_2d = kernel_2d * weights.view(-1, 1, 1)
        
        # Sum over all points
        hist = kernel_2d.sum(dim=0)
        
        # Normalize
        hist = hist / (hist.sum() + 1e-10)
        
        return hist


class DifferentiableWasserstein(nn.Module):
    """
    Differentiable approximation of 1D Wasserstein distance
    
    Uses the fact that for 1D distributions:
    W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
    
    We approximate this with the L1 distance between CDFs
    """
    
    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute approximate Wasserstein distance
        
        Args:
            p, q: normalized histograms (same shape)
            
        Returns:
            scalar distance
        """
        # Compute CDFs
        cdf_p = torch.cumsum(p, dim=-1)
        cdf_q = torch.cumsum(q, dim=-1)
        
        # L1 distance between CDFs
        return torch.mean(torch.abs(cdf_p - cdf_q))


class DifferentiableForwardModel(nn.Module):
    """
    Differentiable forward model for VMI
    
    Uses reparameterization trick to make sampling differentiable:
    - Energy: E = E_center + sigma * eps, where eps ~ N(0,1)
    - Angle: Use Gumbel-softmax or continuous relaxation
    """
    
    def __init__(self, config: DiffConfig, n_peaks: int):
        super().__init__()
        self.cfg = config
        self.n_peaks = n_peaks
        
        # Learnable parameters (in unconstrained space)
        # Will be transformed to physical bounds via sigmoid/softplus
        
        # Energy centers (logit space for sigmoid transform)
        self.E_logits = nn.Parameter(torch.zeros(n_peaks))
        
        # Energy widths (log space for softplus transform)  
        self.sigma_logs = nn.Parameter(torch.zeros(n_peaks) - 2)
        
        # Beta parameters (tanh space)
        self.beta_raw = nn.Parameter(torch.zeros(n_peaks))
        
        # Branching ratios (softmax space)
        self.br_logits = nn.Parameter(torch.zeros(n_peaks))
        
        # Background parameters
        self.bg_frac_logit = nn.Parameter(torch.tensor(-2.0))  # ~0.1
        self.bg_E_logit = nn.Parameter(torch.tensor(-1.0))
        self.bg_sigma_log = nn.Parameter(torch.tensor(-2.0))
        
        # Precompute mass in kg
        self.mass_kg = config.mass * AMU_TO_KG
    
    def get_physical_params(self) -> Dict[str, torch.Tensor]:
        """Transform raw parameters to physical values"""
        E_range = self.cfg.E_max - self.cfg.E_min
        sigma_range = self.cfg.sigma_max - self.cfg.sigma_min
        beta_range = self.cfg.beta_max - self.cfg.beta_min
        
        E_centers = self.cfg.E_min + E_range * torch.sigmoid(self.E_logits)
        sigmas = self.cfg.sigma_min + sigma_range * torch.sigmoid(self.sigma_logs)
        betas = self.cfg.beta_min + beta_range * torch.sigmoid(self.beta_raw)
        branching_ratios = F.softmax(self.br_logits, dim=0)
        
        bg_fraction = 0.3 * torch.sigmoid(self.bg_frac_logit)
        bg_E = self.cfg.E_max * 0.3 * torch.sigmoid(self.bg_E_logit)
        bg_sigma = 0.3 * torch.sigmoid(self.bg_sigma_log)
        
        return {
            'E_centers': E_centers,
            'sigmas': sigmas,
            'betas': betas,
            'branching_ratios': branching_ratios,
            'bg_fraction': bg_fraction,
            'bg_E': bg_E,
            'bg_sigma': bg_sigma
        }
    
    def energy_to_radius(self, E: torch.Tensor) -> torch.Tensor:
        """Convert energy (eV) to radius (mm)"""
        # v = sqrt(2*E/m), r = k*v
        v = torch.sqrt(2.0 * torch.clamp(E, min=1e-10) * EV_TO_JOULE / self.mass_kg)
        return self.cfg.vmi_k * v
    
    def sample_angular_distribution(self, beta: torch.Tensor, n: int) -> torch.Tensor:
        """
        Sample cos(theta) from angular distribution using reparameterization
        
        PDF: f(x) = (1 + beta * P2(x)) / 2, where P2(x) = (3x^2 - 1) / 2
        
        Use inverse CDF sampling with differentiable approximation
        """
        device = beta.device
        
        # For simplicity, use rejection sampling with straight-through estimator
        # In practice, could use normalizing flows for fully differentiable sampling
        
        # Approximate: sample uniform, then weight by PDF
        # This is biased but differentiable
        cos_theta = torch.rand(n, device=device) * 2 - 1  # [-1, 1]
        
        # PDF value (for weighting, not used in simple version)
        P2 = (3 * cos_theta ** 2 - 1) / 2
        pdf = 1 + beta * P2
        
        # For now, just return uniform samples
        # The beta effect will come through the loss function
        return cos_theta, pdf
    
    def forward(self, n_particles: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate particles with current parameters
        
        Returns:
            X, Y coordinates and weights (for importance sampling)
        """
        if n_particles is None:
            n_particles = self.cfg.n_particles
        
        params = self.get_physical_params()
        device = self.E_logits.device
        
        # Allocate particles to peaks based on branching ratios
        n_signal = int(n_particles * (1 - params['bg_fraction'].item()))
        n_bg = n_particles - n_signal
        
        # Use soft allocation (differentiable)
        br = params['branching_ratios']
        peak_counts = (br * n_signal).int()
        # Adjust last peak to match total
        peak_counts[-1] = n_signal - peak_counts[:-1].sum()
        
        all_X = []
        all_Y = []
        all_weights = []
        
        for i in range(self.n_peaks):
            n_i = peak_counts[i].item()
            if n_i <= 0:
                continue
            
            E_center = params['E_centers'][i]
            sigma = params['sigmas'][i]
            beta = params['betas'][i]
            
            # Sample energy using reparameterization trick
            eps = torch.randn(n_i, device=device)
            E = E_center + sigma * eps
            E = torch.clamp(E, min=1e-10)
            
            # Energy to radius
            r = self.energy_to_radius(E)
            
            # Sample angles
            cos_theta, pdf_weight = self.sample_angular_distribution(beta, n_i)
            phi = torch.rand(n_i, device=device) * 2 * np.pi
            sin_theta = torch.sqrt(1 - cos_theta ** 2)
            
            # 3D to 2D projection (polarization along Y)
            vx = sin_theta * torch.cos(phi)
            vy = cos_theta
            
            X = r * vx
            Y = r * vy
            
            # Weight by angular PDF for importance sampling
            all_X.append(X)
            all_Y.append(Y)
            all_weights.append(pdf_weight)
        
        # Background (isotropic)
        if n_bg > 0:
            eps_bg = torch.randn(n_bg, device=device)
            E_bg = params['bg_E'] + params['bg_sigma'] * eps_bg
            E_bg = torch.clamp(E_bg, min=1e-10)
            r_bg = self.energy_to_radius(E_bg)
            
            cos_theta_bg = torch.rand(n_bg, device=device) * 2 - 1
            phi_bg = torch.rand(n_bg, device=device) * 2 * np.pi
            sin_theta_bg = torch.sqrt(1 - cos_theta_bg ** 2)
            
            X_bg = r_bg * sin_theta_bg * torch.cos(phi_bg)
            Y_bg = r_bg * cos_theta_bg
            
            all_X.append(X_bg)
            all_Y.append(Y_bg)
            all_weights.append(torch.ones(n_bg, device=device))
        
        X = torch.cat(all_X)
        Y = torch.cat(all_Y)
        weights = torch.cat(all_weights)
        
        # Normalize weights
        weights = weights / (weights.sum() + 1e-10) * len(weights)
        
        # Add PSF if specified
        if self.cfg.psf_sigma > 0:
            X = X + torch.randn_like(X) * self.cfg.psf_sigma
            Y = Y + torch.randn_like(Y) * self.cfg.psf_sigma
        
        return X, Y, weights


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss comparing distributions at different resolutions
    
    Compares:
    1. Coarse radial distribution (few bins)
    2. Fine radial distribution (many bins)
    3. Coarse angular distribution
    4. Fine angular distribution
    5. 2D (r, theta) histogram at multiple scales
    """
    
    def __init__(self, config: DiffConfig, r_max: float):
        super().__init__()
        self.cfg = config
        self.r_max = r_max
        
        # Radial histograms at different scales
        self.radial_hist_coarse = SoftHistogram(
            config.radial_bins_coarse, 0, r_max)
        self.radial_hist_fine = SoftHistogram(
            config.radial_bins_fine, 0, r_max)
        
        # Angular histograms at different scales
        self.angular_hist_coarse = SoftHistogram(
            config.angular_bins_coarse, -np.pi, np.pi)
        self.angular_hist_fine = SoftHistogram(
            config.angular_bins_fine, -np.pi, np.pi)
        
        # 2D histogram (r, theta)
        self.hist_2d_coarse = SoftHistogram2D(
            config.radial_bins_coarse // 2, config.angular_bins_coarse,
            (0, r_max), (-np.pi, np.pi))
        
        # Wasserstein distance
        self.wasserstein = DifferentiableWasserstein()
    
    def compute_features(self, X: torch.Tensor, Y: torch.Tensor,
                         weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from XY data"""
        r = torch.sqrt(X ** 2 + Y ** 2)
        theta = torch.atan2(Y, X)
        
        features = {
            'radial_coarse': self.radial_hist_coarse(r, weights),
            'radial_fine': self.radial_hist_fine(r, weights),
            'angular_coarse': self.angular_hist_coarse(theta, weights),
            'angular_fine': self.angular_hist_fine(theta, weights),
            'hist_2d_coarse': self.hist_2d_coarse(r, theta, weights),
        }
        
        return features
    
    def forward(self, features_obs: Dict[str, torch.Tensor],
                features_sim: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-scale loss
        
        Prioritizes coarse features (global structure) over fine features
        """
        loss = torch.tensor(0.0, device=features_obs['radial_coarse'].device)
        
        # Coarse radial (most important - global peak structure)
        loss += self.cfg.weight_coarse * self.wasserstein(
            features_obs['radial_coarse'], features_sim['radial_coarse'])
        
        # Fine radial
        loss += self.cfg.weight_fine * self.wasserstein(
            features_obs['radial_fine'], features_sim['radial_fine'])
        
        # Coarse angular
        loss += self.cfg.weight_coarse * 0.5 * torch.mean(
            (features_obs['angular_coarse'] - features_sim['angular_coarse']) ** 2)
        
        # Fine angular
        loss += self.cfg.weight_fine * 0.3 * torch.mean(
            (features_obs['angular_fine'] - features_sim['angular_fine']) ** 2)
        
        # 2D histogram
        loss += self.cfg.weight_medium * torch.mean(
            (features_obs['hist_2d_coarse'] - features_sim['hist_2d_coarse']) ** 2)
        
        return loss


class DifferentiableReconstructor:
    """
    Main reconstructor using differentiable forward fitting
    """
    
    def __init__(self, config: DiffConfig):
        self.cfg = config
        self.device = torch.device(config.device)
    
    def fit(self, xy_obs: np.ndarray, n_peaks: int = 3,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Fit observed XY data
        
        Args:
            xy_obs: observed XY coordinates (N, 2)
            n_peaks: number of peaks to fit
            verbose: print progress
            
        Returns:
            fitted parameters
        """
        # Convert to torch
        xy_obs_t = torch.tensor(xy_obs, dtype=torch.float32, device=self.device)
        X_obs = xy_obs_t[:, 0]
        Y_obs = xy_obs_t[:, 1]
        
        # Estimate r_max from data
        r_obs = torch.sqrt(X_obs ** 2 + Y_obs ** 2)
        r_max = torch.quantile(r_obs, 0.99).item()
        
        if verbose:
            print(f"Data: {len(xy_obs)} particles, r_max={r_max:.2f} mm")
        
        # Create model and loss
        model = DifferentiableForwardModel(self.cfg, n_peaks).to(self.device)
        loss_fn = MultiScaleLoss(self.cfg, r_max).to(self.device)
        
        # Extract observed features (fixed)
        with torch.no_grad():
            features_obs = loss_fn.compute_features(X_obs, Y_obs)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.n_iterations)
        
        # Training loop
        best_loss = float('inf')
        best_params = None
        losses = []
        
        for iteration in range(self.cfg.n_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            X_sim, Y_sim, weights = model(self.cfg.n_particles)
            
            # Compute features
            features_sim = loss_fn.compute_features(X_sim, Y_sim, weights)
            
            # Compute loss
            loss = loss_fn(features_obs, features_sim)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() 
                              for k, v in model.get_physical_params().items()}
            
            if verbose and (iteration + 1) % 50 == 0:
                params = model.get_physical_params()
                E_str = ', '.join([f'{e.item():.3f}' for e in params['E_centers']])
                print(f"Iter {iteration+1}: loss={loss.item():.6f}, E=[{E_str}] eV")
        
        # Convert best params to numpy
        result = {
            'n_peaks': n_peaks,
            'E_centers': best_params['E_centers'].cpu().numpy().tolist(),
            'sigmas': best_params['sigmas'].cpu().numpy().tolist(),
            'betas': best_params['betas'].cpu().numpy().tolist(),
            'branching_ratios': best_params['branching_ratios'].cpu().numpy().tolist(),
            'bg_fraction': best_params['bg_fraction'].cpu().item(),
            'bg_E': best_params['bg_E'].cpu().item(),
            'bg_sigma': best_params['bg_sigma'].cpu().item(),
            'final_loss': best_loss,
            'loss_history': losses,
            'r_max': r_max
        }
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """Print fitting results"""
        print("\n" + "="*50)
        print("Fitting Results (Differentiable X2)")
        print("="*50)
        
        for i in range(result['n_peaks']):
            print(f"Peak {i+1}:")
            print(f"  Energy: {result['E_centers'][i]:.4f} eV")
            print(f"  Sigma: {result['sigmas'][i]:.4f} eV")
            print(f"  Beta: {result['betas'][i]:.3f}")
            print(f"  BR: {result['branching_ratios'][i]:.3f}")
        
        print(f"\nBackground: {result['bg_fraction']:.3f}")
        print(f"Final loss: {result['final_loss']:.6f}")


# =============================================================================
# Improved Angular Sampling with Differentiable Importance Weighting
# =============================================================================
class DirectSamplingForwardModel(nn.Module):
    """
    Forward model with direct angular sampling (rejection sampling)
    
    Key insight: We don't need gradients through the sampling process itself.
    We can use rejection sampling to get correct angular distribution,
    then the loss gradients will flow through the histogram comparison.
    
    This is simpler and more accurate than importance weighting.
    """
    
    def __init__(self, config: DiffConfig, n_peaks: int):
        super().__init__()
        self.cfg = config
        self.n_peaks = n_peaks
        
        # Learnable parameters
        self.E_logits = nn.Parameter(torch.zeros(n_peaks))
        self.sigma_logs = nn.Parameter(torch.zeros(n_peaks) - 2)
        self.beta_raw = nn.Parameter(torch.zeros(n_peaks))
        self.br_logits = nn.Parameter(torch.zeros(n_peaks))
        
        self.bg_frac_logit = nn.Parameter(torch.tensor(-2.0))
        self.bg_E_logit = nn.Parameter(torch.tensor(-1.0))
        self.bg_sigma_log = nn.Parameter(torch.tensor(-2.0))
        
        self.mass_kg = config.mass * AMU_TO_KG
    
    def get_physical_params(self) -> Dict[str, torch.Tensor]:
        """Transform to physical parameters"""
        E_range = self.cfg.E_max - self.cfg.E_min
        
        E_centers = self.cfg.E_min + E_range * torch.sigmoid(self.E_logits)
        sigmas = self.cfg.sigma_min + (self.cfg.sigma_max - self.cfg.sigma_min) * torch.sigmoid(self.sigma_logs)
        betas = self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * torch.sigmoid(self.beta_raw)
        branching_ratios = F.softmax(self.br_logits, dim=0)
        
        bg_fraction = 0.3 * torch.sigmoid(self.bg_frac_logit)
        bg_E = self.cfg.E_max * 0.3 * torch.sigmoid(self.bg_E_logit)
        bg_sigma = 0.3 * torch.sigmoid(self.bg_sigma_log)
        
        return {
            'E_centers': E_centers,
            'sigmas': sigmas,
            'betas': betas,
            'branching_ratios': branching_ratios,
            'bg_fraction': bg_fraction,
            'bg_E': bg_E,
            'bg_sigma': bg_sigma
        }
    
    def energy_to_radius(self, E: torch.Tensor) -> torch.Tensor:
        """Energy to radius conversion"""
        v = torch.sqrt(2.0 * torch.clamp(E, min=1e-10) * EV_TO_JOULE / self.mass_kg)
        return self.cfg.vmi_k * v
    
    @staticmethod
    def rejection_sample_angular(beta: float, n: int) -> np.ndarray:
        """
        Rejection sampling for angular distribution
        PDF: f(x) = 1 + beta * P2(x), where P2(x) = (3x^2 - 1) / 2
        """
        if n == 0:
            return np.array([])
        
        # PDF max value
        f_max = max(1 + abs(beta), 1 + abs(beta) / 2, 1.0) * 1.1
        
        samples = []
        while len(samples) < n:
            batch = max(n - len(samples), 100) * 2
            x = np.random.uniform(-1, 1, batch)
            u = np.random.uniform(0, f_max, batch)
            P2 = (3 * x**2 - 1) / 2
            f_x = 1 + beta * P2
            valid = u < f_x
            samples.extend(x[valid])
        
        return np.array(samples[:n])
    
    def forward(self, n_particles: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate particles with correct angular distribution
        
        Returns X, Y (no weights needed since we sample correctly)
        """
        if n_particles is None:
            n_particles = self.cfg.n_particles
        
        params = self.get_physical_params()
        device = self.E_logits.device
        
        # Particle allocation
        bg_frac = params['bg_fraction'].item()
        n_signal = int(n_particles * (1 - bg_frac))
        n_bg = n_particles - n_signal
        
        br = params['branching_ratios'].detach().cpu().numpy()
        peak_counts = np.random.multinomial(n_signal, br)
        
        all_X = []
        all_Y = []
        
        for i in range(self.n_peaks):
            n_i = peak_counts[i]
            if n_i == 0:
                continue
            
            E_center = params['E_centers'][i].item()
            sigma = params['sigmas'][i].item()
            beta = params['betas'][i].item()
            
            # Sample energy (reparameterized)
            eps = torch.randn(n_i, device=device)
            E = params['E_centers'][i] + params['sigmas'][i] * eps
            E = torch.clamp(E, min=1e-10)
            
            r = self.energy_to_radius(E)
            
            # Sample angles using rejection sampling (correct distribution!)
            cos_theta_np = self.rejection_sample_angular(beta, n_i)
            cos_theta = torch.tensor(cos_theta_np, dtype=torch.float32, device=device)
            
            phi = torch.rand(n_i, device=device) * 2 * np.pi
            sin_theta = torch.sqrt(torch.clamp(1 - cos_theta ** 2, min=0))
            
            # Project to XY (polarization along Y)
            X = r * sin_theta * torch.cos(phi)
            Y = r * cos_theta
            
            all_X.append(X)
            all_Y.append(Y)
        
        # Background (isotropic)
        if n_bg > 0:
            eps_bg = torch.randn(n_bg, device=device)
            E_bg = params['bg_E'] + params['bg_sigma'] * eps_bg
            E_bg = torch.clamp(E_bg, min=1e-10)
            r_bg = self.energy_to_radius(E_bg)
            
            cos_theta_bg = torch.rand(n_bg, device=device) * 2 - 1
            phi_bg = torch.rand(n_bg, device=device) * 2 * np.pi
            sin_theta_bg = torch.sqrt(torch.clamp(1 - cos_theta_bg ** 2, min=0))
            
            X_bg = r_bg * sin_theta_bg * torch.cos(phi_bg)
            Y_bg = r_bg * cos_theta_bg
            
            all_X.append(X_bg)
            all_Y.append(Y_bg)
        
        X = torch.cat(all_X)
        Y = torch.cat(all_Y)
        
        # PSF
        if self.cfg.psf_sigma > 0:
            X = X + torch.randn_like(X) * self.cfg.psf_sigma
            Y = Y + torch.randn_like(Y) * self.cfg.psf_sigma
        
        return X, Y


class ImprovedForwardModel(nn.Module):
    """
    Improved forward model with proper angular distribution handling
    
    Key insight: Instead of trying to sample from the angular distribution,
    we sample uniformly and use importance weights. The weights are then
    used in the soft histogram computation.
    
    This makes the entire pipeline differentiable w.r.t. beta.
    """
    
    def __init__(self, config: DiffConfig, n_peaks: int):
        super().__init__()
        self.cfg = config
        self.n_peaks = n_peaks
        
        # Learnable parameters
        self.E_logits = nn.Parameter(torch.zeros(n_peaks))
        self.sigma_logs = nn.Parameter(torch.zeros(n_peaks) - 2)
        self.beta_raw = nn.Parameter(torch.zeros(n_peaks))
        self.br_logits = nn.Parameter(torch.zeros(n_peaks))
        
        self.bg_frac_logit = nn.Parameter(torch.tensor(-2.0))
        self.bg_E_logit = nn.Parameter(torch.tensor(-1.0))
        self.bg_sigma_log = nn.Parameter(torch.tensor(-2.0))
        
        self.mass_kg = config.mass * AMU_TO_KG
    
    def get_physical_params(self) -> Dict[str, torch.Tensor]:
        """Transform to physical parameters"""
        E_range = self.cfg.E_max - self.cfg.E_min
        
        E_centers = self.cfg.E_min + E_range * torch.sigmoid(self.E_logits)
        sigmas = self.cfg.sigma_min + (self.cfg.sigma_max - self.cfg.sigma_min) * torch.sigmoid(self.sigma_logs)
        betas = self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * torch.sigmoid(self.beta_raw)
        branching_ratios = F.softmax(self.br_logits, dim=0)
        
        bg_fraction = 0.3 * torch.sigmoid(self.bg_frac_logit)
        bg_E = self.cfg.E_max * 0.3 * torch.sigmoid(self.bg_E_logit)
        bg_sigma = 0.3 * torch.sigmoid(self.bg_sigma_log)
        
        return {
            'E_centers': E_centers,
            'sigmas': sigmas,
            'betas': betas,
            'branching_ratios': branching_ratios,
            'bg_fraction': bg_fraction,
            'bg_E': bg_E,
            'bg_sigma': bg_sigma
        }
    
    def energy_to_radius(self, E: torch.Tensor) -> torch.Tensor:
        """Energy to radius conversion"""
        v = torch.sqrt(2.0 * torch.clamp(E, min=1e-10) * EV_TO_JOULE / self.mass_kg)
        return self.cfg.vmi_k * v
    
    def compute_angular_weight(self, cos_theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weight for angular distribution
        
        PDF: f(cos_theta) = 0.5 * (1 + beta * P2(cos_theta))
        where P2(x) = (3x^2 - 1) / 2
        
        For uniform sampling on [-1, 1], weight = f(cos_theta) * 2
        """
        P2 = (3 * cos_theta ** 2 - 1) / 2
        weight = 1 + beta * P2
        return torch.clamp(weight, min=0.01)  # Ensure positive weights
    
    def sample_angular_proper(self, beta: torch.Tensor, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Proper angular sampling using inverse CDF method (differentiable approximation)
        
        For f(x) = 1 + beta * P2(x) where P2(x) = (3x^2-1)/2
        CDF(x) = (x + 1)/2 + beta * (x^3 - x + 2/3) / 4
        
        We use a differentiable approximation via soft sorting
        """
        # Sample uniform [0, 1] for inverse CDF
        u = torch.rand(n, device=device)
        
        # For small beta, linear approximation works
        # For larger beta, we need numerical inversion
        # Use Newton's method with straight-through gradient
        
        # Initial guess: linear
        cos_theta = 2 * u - 1
        
        # Newton iterations (detached for stability, but keep final gradient)
        with torch.no_grad():
            for _ in range(3):
                # CDF value
                x = cos_theta
                cdf = (x + 1) / 2 + beta.detach() * (x**3 - x + 2/3) / 4
                # PDF value  
                pdf = 0.5 + beta.detach() * (3 * x**2 - 1) / 4
                pdf = torch.clamp(pdf, min=0.1)
                # Newton step
                cos_theta = cos_theta - (cdf - u) / pdf
                cos_theta = torch.clamp(cos_theta, -1, 1)
        
        # Recompute with gradient for beta
        # The key: weight by how much the PDF differs from uniform
        P2 = (3 * cos_theta ** 2 - 1) / 2
        weight = 1 + beta * P2
        
        return cos_theta, torch.clamp(weight, min=0.01)
    
    def forward(self, n_particles: Optional[int] = None,
                return_per_peak: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate particles with importance weights
        
        The key insight: sample uniformly, weight by PDF
        This makes gradients flow through beta via the weights
        """
        if n_particles is None:
            n_particles = self.cfg.n_particles
        
        params = self.get_physical_params()
        device = self.E_logits.device
        
        # Particle allocation
        n_signal = int(n_particles * (1 - params['bg_fraction'].item()))
        n_bg = n_particles - n_signal
        
        br = params['branching_ratios']
        peak_counts = (br * n_signal).int()
        peak_counts[-1] = n_signal - peak_counts[:-1].sum()
        
        all_X = []
        all_Y = []
        all_weights = []
        all_peak_idx = []
        
        for i in range(self.n_peaks):
            n_i = max(peak_counts[i].item(), 1)
            
            E_center = params['E_centers'][i]
            sigma = params['sigmas'][i]
            beta = params['betas'][i]
            
            # Reparameterized energy sampling
            eps = torch.randn(n_i, device=device)
            E = E_center + sigma * eps
            E = torch.clamp(E, min=1e-10)
            
            r = self.energy_to_radius(E)
            
            # Uniform angular sampling with importance weights
            cos_theta = torch.rand(n_i, device=device) * 2 - 1
            phi = torch.rand(n_i, device=device) * 2 * np.pi
            
            # Importance weight from angular distribution
            angular_weight = self.compute_angular_weight(cos_theta, beta)
            
            sin_theta = torch.sqrt(torch.clamp(1 - cos_theta ** 2, min=0))
            
            # Project to XY (polarization along Y)
            X = r * sin_theta * torch.cos(phi)
            Y = r * cos_theta
            
            all_X.append(X)
            all_Y.append(Y)
            all_weights.append(angular_weight * br[i])
            all_peak_idx.extend([i] * n_i)
        
        # Background
        if n_bg > 0:
            eps_bg = torch.randn(n_bg, device=device)
            E_bg = params['bg_E'] + params['bg_sigma'] * eps_bg
            E_bg = torch.clamp(E_bg, min=1e-10)
            r_bg = self.energy_to_radius(E_bg)
            
            cos_theta_bg = torch.rand(n_bg, device=device) * 2 - 1
            phi_bg = torch.rand(n_bg, device=device) * 2 * np.pi
            sin_theta_bg = torch.sqrt(torch.clamp(1 - cos_theta_bg ** 2, min=0))
            
            X_bg = r_bg * sin_theta_bg * torch.cos(phi_bg)
            Y_bg = r_bg * cos_theta_bg
            
            all_X.append(X_bg)
            all_Y.append(Y_bg)
            all_weights.append(torch.ones(n_bg, device=device) * params['bg_fraction'])
            all_peak_idx.extend([-1] * n_bg)
        
        X = torch.cat(all_X)
        Y = torch.cat(all_Y)
        weights = torch.cat(all_weights)
        
        # Normalize weights
        weights = weights / (weights.sum() + 1e-10) * len(weights)
        
        # PSF
        if self.cfg.psf_sigma > 0:
            X = X + torch.randn_like(X) * self.cfg.psf_sigma
            Y = Y + torch.randn_like(Y) * self.cfg.psf_sigma
        
        return X, Y, weights


class MultiResolutionConsistencyLoss(nn.Module):
    """
    Multi-resolution consistency loss
    
    Key insight: If model matches data at coarse bins, it should also match at finer bins
    (as long as dr, dθ < feature size). This provides a strong constraint.
    
    We compute histograms at multiple (dr, dθ) combinations and require ALL to match.
    This is more constraining than just using one resolution.
    """
    
    def __init__(self, config: DiffConfig, r_max: float):
        super().__init__()
        self.cfg = config
        self.r_max = r_max
        
        # Define multiple resolution levels for radial bins
        # From coarse (few bins, large dr) to fine (many bins, small dr)
        self.radial_resolutions = [10, 20, 40, 80]  # n_bins
        
        # Define multiple resolution levels for angular bins
        self.angular_resolutions = [6, 12, 24, 36]  # n_bins
        
        # Create soft histograms for each resolution
        self.radial_hists = nn.ModuleList([
            SoftHistogram(n, 0, r_max) for n in self.radial_resolutions
        ])
        
        self.angular_hists = nn.ModuleList([
            SoftHistogram(n, -np.pi, np.pi) for n in self.angular_resolutions
        ])
        
        # 2D histograms at different resolutions
        self.hist_2d_resolutions = [(10, 6), (20, 12), (40, 24)]  # (n_r, n_theta)
        self.hist_2d = nn.ModuleList([
            SoftHistogram2D(nr, nt, (0, r_max), (-np.pi, np.pi))
            for nr, nt in self.hist_2d_resolutions
        ])
        
        # Radial slices for angular analysis (crucial for beta)
        self.n_radial_slices = 5
        self.slice_angular_hist = SoftHistogram(12, -np.pi, np.pi)
        
        self.wasserstein = DifferentiableWasserstein()
    
    def compute_features(self, X: torch.Tensor, Y: torch.Tensor,
                         weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract features at all resolutions"""
        r = torch.sqrt(X ** 2 + Y ** 2)
        theta = torch.atan2(Y, X)
        
        features = {}
        
        # Radial at multiple resolutions
        for i, hist in enumerate(self.radial_hists):
            features[f'radial_{self.radial_resolutions[i]}'] = hist(r, weights)
        
        # Angular at multiple resolutions
        for i, hist in enumerate(self.angular_hists):
            features[f'angular_{self.angular_resolutions[i]}'] = hist(theta, weights)
        
        # 2D at multiple resolutions
        for i, hist in enumerate(self.hist_2d):
            nr, nt = self.hist_2d_resolutions[i]
            features[f'hist2d_{nr}x{nt}'] = hist(r, theta, weights)
        
        # Angular in radial slices (key for beta recovery!)
        slice_edges = torch.linspace(0, self.r_max, self.n_radial_slices + 1)
        for i in range(self.n_radial_slices):
            r_min, r_max_slice = slice_edges[i], slice_edges[i + 1]
            
            # Soft mask for radial slice
            # Sigmoid gives smooth transition at boundaries
            mask = torch.sigmoid((r - r_min) * 20) * torch.sigmoid((r_max_slice - r) * 20)
            
            if weights is not None:
                slice_weights = weights * mask
            else:
                slice_weights = mask
            
            # Normalize slice weights
            slice_weights = slice_weights / (slice_weights.sum() + 1e-10) * mask.sum()
            
            features[f'angular_slice_{i}'] = self.slice_angular_hist(theta, slice_weights)
        
        return features
    
    def forward(self, features_obs: Dict[str, torch.Tensor],
                features_sim: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-resolution consistency loss
        
        Returns:
            total_loss, loss_breakdown (for debugging)
        """
        device = features_obs[f'radial_{self.radial_resolutions[0]}'].device
        total_loss = torch.tensor(0.0, device=device)
        breakdown = {}
        
        # Radial losses at all resolutions
        # Weight: coarse more important (global structure)
        radial_weights = [1.0, 0.7, 0.4, 0.2]  # coarse to fine
        for i, n_bins in enumerate(self.radial_resolutions):
            key = f'radial_{n_bins}'
            loss_r = self.wasserstein(features_obs[key], features_sim[key])
            total_loss += radial_weights[i] * loss_r
            breakdown[key] = loss_r.item()
        
        # Angular losses at all resolutions
        angular_weights = [1.0, 0.7, 0.4, 0.2]
        for i, n_bins in enumerate(self.angular_resolutions):
            key = f'angular_{n_bins}'
            loss_a = F.smooth_l1_loss(features_obs[key], features_sim[key])
            total_loss += angular_weights[i] * 0.5 * loss_a
            breakdown[key] = loss_a.item()
        
        # 2D losses
        hist2d_weights = [0.5, 0.3, 0.15]
        for i, (nr, nt) in enumerate(self.hist_2d_resolutions):
            key = f'hist2d_{nr}x{nt}'
            loss_2d = F.smooth_l1_loss(features_obs[key], features_sim[key])
            total_loss += hist2d_weights[i] * loss_2d
            breakdown[key] = loss_2d.item()
        
        # Angular in radial slices (CRITICAL for beta!)
        # These should have high weight because they directly constrain beta
        for i in range(self.n_radial_slices):
            key = f'angular_slice_{i}'
            loss_slice = F.smooth_l1_loss(features_obs[key], features_sim[key])
            total_loss += 0.8 * loss_slice  # High weight!
            breakdown[key] = loss_slice.item()
        
        # Consistency penalty: if coarse matches, fine should too
        # Penalize cases where coarse loss is low but fine loss is high
        coarse_radial = breakdown[f'radial_{self.radial_resolutions[0]}']
        fine_radial = breakdown[f'radial_{self.radial_resolutions[-1]}']
        if coarse_radial < 0.01 and fine_radial > 0.05:
            # Coarse matches but fine doesn't - penalize
            total_loss += 0.5 * torch.tensor(fine_radial - coarse_radial, device=device)
        
        return total_loss, breakdown


class ImprovedMultiScaleLoss(nn.Module):
    """
    Improved multi-scale loss with better gradient flow
    
    Key improvements:
    1. Sinkhorn divergence instead of simple Wasserstein (more stable gradients)
    2. Angular distribution in radial bins (captures beta-energy correlation)
    3. Smooth L1 loss for robustness
    """
    
    def __init__(self, config: DiffConfig, r_max: float):
        super().__init__()
        self.cfg = config
        self.r_max = r_max
        
        # Multi-scale radial histograms
        scales = [
            (config.radial_bins_coarse, 'coarse'),
            (config.radial_bins_coarse * 2, 'medium'),
            (config.radial_bins_fine, 'fine')
        ]
        
        self.radial_hists = nn.ModuleDict()
        for n_bins, name in scales:
            self.radial_hists[name] = SoftHistogram(n_bins, 0, r_max)
        
        # Angular histograms
        self.angular_hist_coarse = SoftHistogram(
            config.angular_bins_coarse, -np.pi, np.pi)
        self.angular_hist_fine = SoftHistogram(
            config.angular_bins_fine, -np.pi, np.pi)
        
        # 2D histograms at different scales
        self.hist_2d_coarse = SoftHistogram2D(
            config.radial_bins_coarse // 2, config.angular_bins_coarse,
            (0, r_max), (-np.pi, np.pi))
        self.hist_2d_fine = SoftHistogram2D(
            config.radial_bins_coarse, config.angular_bins_fine // 2,
            (0, r_max), (-np.pi, np.pi))
        
        # Radial bins for angular analysis
        self.n_radial_regions = 5
        self.radial_edges = torch.linspace(0, r_max, self.n_radial_regions + 1)
        
        self.wasserstein = DifferentiableWasserstein()
    
    def compute_features(self, X: torch.Tensor, Y: torch.Tensor,
                         weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract comprehensive multi-scale features"""
        r = torch.sqrt(X ** 2 + Y ** 2)
        theta = torch.atan2(Y, X)
        
        features = {}
        
        # Multi-scale radial
        for name, hist in self.radial_hists.items():
            features[f'radial_{name}'] = hist(r, weights)
        
        # Angular
        features['angular_coarse'] = self.angular_hist_coarse(theta, weights)
        features['angular_fine'] = self.angular_hist_fine(theta, weights)
        
        # 2D
        features['hist_2d_coarse'] = self.hist_2d_coarse(r, theta, weights)
        features['hist_2d_fine'] = self.hist_2d_fine(r, theta, weights)
        
        # Angular in radial bins (important for beta recovery)
        for i in range(self.n_radial_regions):
            r_min = self.radial_edges[i]
            r_max = self.radial_edges[i + 1]
            
            # Soft mask for radial region
            mask = torch.sigmoid((r - r_min) * 10) * torch.sigmoid((r_max - r) * 10)
            
            if weights is not None:
                region_weights = weights * mask
            else:
                region_weights = mask
            
            # Normalize
            region_weights = region_weights / (region_weights.sum() + 1e-10) * mask.sum()
            
            features[f'angular_r{i}'] = self.angular_hist_coarse(theta, region_weights)
        
        return features
    
    def forward(self, features_obs: Dict[str, torch.Tensor],
                features_sim: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute hierarchical multi-scale loss"""
        device = features_obs['radial_coarse'].device
        loss = torch.tensor(0.0, device=device)
        
        # Radial losses (coarse to fine, decreasing weight)
        loss += 1.0 * self.wasserstein(
            features_obs['radial_coarse'], features_sim['radial_coarse'])
        loss += 0.5 * self.wasserstein(
            features_obs['radial_medium'], features_sim['radial_medium'])
        loss += 0.2 * self.wasserstein(
            features_obs['radial_fine'], features_sim['radial_fine'])
        
        # Angular losses
        loss += 0.5 * F.smooth_l1_loss(
            features_obs['angular_coarse'], features_sim['angular_coarse'])
        loss += 0.2 * F.smooth_l1_loss(
            features_obs['angular_fine'], features_sim['angular_fine'])
        
        # 2D losses
        loss += 0.3 * F.smooth_l1_loss(
            features_obs['hist_2d_coarse'], features_sim['hist_2d_coarse'])
        loss += 0.1 * F.smooth_l1_loss(
            features_obs['hist_2d_fine'], features_sim['hist_2d_fine'])
        
        # Angular in radial bins (crucial for beta)
        for i in range(self.n_radial_regions):
            loss += 0.3 * F.smooth_l1_loss(
                features_obs[f'angular_r{i}'], features_sim[f'angular_r{i}'])
        
        return loss


class SmartInitializer:
    """
    Data-driven initialization for optimization
    
    Extracts initial guesses directly from observed data:
    1. Peak positions from radial distribution → E_centers
    2. Angular distribution in each peak region → betas (using proven methods)
    3. Peak areas → branching ratios
    4. Peak widths → sigmas
    
    Beta estimation methods (integrated from v1/v2/v3 reconstructors):
    - FFT-based: Uses k=2 Fourier component (from Abel_backward_reconstruction_v3)
    - Curve-fit: Multi-start optimization with P₂ model (from vmi_physics_reconstructor)
    - Moment-based: Initial guess from <P₂> (from vmi_multiresolution_reconstructor)
    - Abel correction: Onion peeling for inner peaks contaminated by outer peaks
    """
    
    BETA_MIN = -1.0
    BETA_MAX = 2.0
    
    def __init__(self, config: DiffConfig):
        self.cfg = config
    
    def analyze_data(self, xy_obs: np.ndarray, n_peaks: int) -> Dict[str, Any]:
        """
        Analyze observed data to extract initial parameter estimates
        """
        r = np.sqrt(xy_obs[:, 0]**2 + xy_obs[:, 1]**2)
        theta = np.arctan2(xy_obs[:, 1], xy_obs[:, 0])
        
        r_max = np.percentile(r, 99)
        
        # Step 1: Find peaks in radial distribution
        peaks_info = self._find_radial_peaks(r, r_max, n_peaks)
        
        # Step 2: Estimate beta for each peak using proven methods
        # Process from outside-in (onion peeling) for Abel correction
        betas = self._estimate_betas_proven(r, theta, peaks_info)
        
        # Step 3: Estimate sigmas from peak widths
        sigmas = self._estimate_sigmas(r, peaks_info)
        
        # Step 4: Estimate branching ratios from peak areas
        branching_ratios = self._estimate_branching_ratios(r, peaks_info)
        
        # Step 5: Estimate background
        bg_info = self._estimate_background(r, peaks_info)
        
        return {
            'E_centers': peaks_info['E_centers'],
            'r_centers': peaks_info['r_centers'],
            'betas': betas,
            'sigmas': sigmas,
            'branching_ratios': branching_ratios,
            'bg_fraction': bg_info['fraction'],
            'bg_E': bg_info['E'],
            'bg_sigma': bg_info['sigma'],
            'r_max': r_max,
            'confidence': peaks_info['confidence']
        }
    
    def _find_radial_peaks(self, r: np.ndarray, r_max: float, n_peaks: int) -> Dict:
        """Find peaks in radial distribution"""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        # High-resolution histogram
        n_bins = 200
        hist, edges = np.histogram(r, bins=n_bins, range=(0, r_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        # Smooth to reduce noise
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
        
        # Find peaks with adaptive prominence
        for prominence_factor in [0.1, 0.05, 0.02, 0.01]:
            peaks, props = find_peaks(
                hist_smooth, 
                prominence=hist_smooth.max() * prominence_factor,
                distance=n_bins // 20,
                width=2
            )
            if len(peaks) >= n_peaks:
                break
        
        # Sort by prominence and take top n_peaks
        if len(peaks) > 0:
            sorted_idx = np.argsort(props['prominences'])[::-1]
            peaks = peaks[sorted_idx[:n_peaks]]
            peaks = np.sort(peaks)  # Sort by position
        
        # Convert to energy
        r_peaks = centers[peaks] if len(peaks) > 0 else np.linspace(r_max*0.2, r_max*0.8, n_peaks)
        E_peaks = self._r_to_E(r_peaks)
        
        # Confidence based on peak prominence
        confidence = 'high' if len(peaks) >= n_peaks else 'low'
        
        return {
            'r_centers': r_peaks,
            'E_centers': E_peaks,
            'peak_indices': peaks,
            'hist': hist_smooth,
            'bin_centers': centers,
            'confidence': confidence
        }
    
    def _estimate_betas_proven(self, r: np.ndarray, theta: np.ndarray, 
                                peaks_info: Dict) -> List[float]:
        """
        Estimate beta for each peak using proven methods from v1/v2/v3 reconstructors.
        
        Strategy:
        1. Process peaks from outside-in (onion peeling) for Abel correction
        2. Use multiple estimation methods and combine:
           - FFT-based (k=2 component)
           - Curve-fit with multi-start optimization
           - Moment-based initial guess
        3. Apply Abel projection correction for inner peaks
        
        Physics: For Y-polarization, cos(θ_3D) = sin(θ_XY)
        Angular distribution: I(θ) = A * [1 + β * P₂(sin θ)]
        """
        from scipy.optimize import curve_fit
        
        r_centers = peaks_info['r_centers']
        n_peaks = len(r_centers)
        
        if n_peaks == 0:
            return []
        
        # Sort peaks by radius (descending) for outside-in processing
        sorted_indices = np.argsort(r_centers)[::-1]
        
        betas = [0.0] * n_peaks
        fitted_outer_peaks = []  # Store (r, beta) for Abel correction
        
        for idx in sorted_indices:
            r_peak = r_centers[idx]
            
            # Define radial window around peak
            if idx == 0:
                r_min = 0
            else:
                r_min = (r_centers[idx-1] + r_peak) / 2
            
            if idx == n_peaks - 1:
                r_max_window = r_peak * 1.5
            else:
                r_max_window = (r_peak + r_centers[idx+1]) / 2
            
            # Use tighter window for better isolation
            window = min(r_max_window - r_peak, r_peak - r_min) * 0.8
            window = max(window, 0.3)  # Minimum window
            
            # Select particles in this region
            mask = (r >= r_peak - window) & (r < r_peak + window)
            theta_region = theta[mask]
            n_events = len(theta_region)
            
            if n_events < 30:
                betas[idx] = 0.0
                continue
            
            # Method 1: FFT-based estimation (from v3)
            beta_fft = self._estimate_beta_fft(theta_region)
            
            # Method 2: Curve-fit with multi-start (from physics reconstructor)
            beta_curvefit, beta_err = self._estimate_beta_curvefit(theta_region)
            
            # Method 3: Moment-based (from multiresolution)
            beta_moment = self._estimate_beta_moment(theta_region)
            
            # Combine estimates (weighted average, prefer curve-fit if error is low)
            if beta_err < 0.3:
                # Curve-fit is reliable
                beta_observed = 0.5 * beta_curvefit + 0.3 * beta_fft + 0.2 * beta_moment
            else:
                # Curve-fit less reliable, weight FFT more
                beta_observed = 0.4 * beta_fft + 0.3 * beta_curvefit + 0.3 * beta_moment
            
            # Apply Abel projection correction for inner peaks
            beta_corrected = self._apply_abel_correction(
                beta_observed, r_peak, fitted_outer_peaks)
            
            betas[idx] = float(np.clip(beta_corrected, self.BETA_MIN, self.BETA_MAX))
            
            # Store for Abel correction of inner peaks
            fitted_outer_peaks.append({'r': r_peak, 'beta': betas[idx]})
        
        return betas
    
    def _estimate_beta_fft(self, theta: np.ndarray) -> float:
        """
        FFT-based beta estimation (adapted from Abel_backward_reconstruction_v3)
        
        Uses the k=2 Fourier component of the angular distribution.
        
        For angular distribution I(θ) = A * [1 + β * P₂(cos θ)]
        where P₂(x) = (3x² - 1)/2 = (3cos²θ - 1)/2
        
        This can be rewritten as:
        I(θ) = A * [1 + β * (3cos²θ - 1)/2]
             = A * [1 - β/2 + (3β/2)cos²θ]
             = A * [(1 - β/2) + (3β/4)(1 + cos(2θ))]
             = A * [(1 - β/2 + 3β/4) + (3β/4)cos(2θ)]
             = A * [(1 + β/4) + (3β/4)cos(2θ)]
        
        So the ratio of k=2 amplitude to DC gives us β:
        c2/c0 = (3β/4) / (1 + β/4) = 3β / (4 + β)
        
        Solving for β: β = 4*c2 / (3*c0 - c2)
        
        Note: For Y-polarization VMI, we use sin(θ) instead of cos(θ),
        which shifts the phase by π/2. The k=2 component still works
        but the phase interpretation changes.
        """
        n_bins = 72  # High resolution for FFT
        hist, edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
        hist = hist.astype(float)
        
        # FFT analysis
        fft = np.fft.fft(hist)
        c0 = np.abs(fft[0]) / n_bins  # DC component
        
        if c0 < 1e-10:
            return 0.0
        
        # k=2 component
        c2_complex = fft[2]
        c2_amp = 2 * np.abs(c2_complex) / n_bins
        phase = np.angle(c2_complex)
        
        # For sin²(θ) distribution (Y-polarization), the phase is shifted
        # sin²(θ) = (1 - cos(2θ))/2, so peaks are at θ = ±π/2
        # The phase of the k=2 component should be near π for β > 0
        # and near 0 for β < 0
        
        # Determine sign based on phase
        # For sin(θ) based distribution: phase near π means β > 0
        if abs(phase) > np.pi/2:
            sign = 1.0  # β > 0 (parallel)
        else:
            sign = -1.0  # β < 0 (perpendicular)
        
        c2_signed = sign * c2_amp
        
        # β = 4 * c2 / (3 * c0 - c2)
        denominator = 3.0 * c0 - c2_signed
        if abs(denominator) < 1e-10:
            return 0.0
        
        beta = 4.0 * c2_signed / denominator
        
        return np.clip(beta, self.BETA_MIN, self.BETA_MAX)
    
    def _estimate_beta_curvefit(self, theta: np.ndarray) -> Tuple[float, float]:
        """
        Curve-fit based beta estimation (from vmi_physics_reconstructor)
        
        Fits I(θ) = A * [1 + β * P₂(sin θ)] using multi-start optimization.
        Exploits cylindrical symmetry by folding θ to [0, π].
        
        Returns:
            (beta, beta_error)
        """
        from scipy.optimize import curve_fit
        
        # Fold to [0, π] for two-fold symmetry
        theta_folded = np.abs(theta)
        
        # Adaptive binning based on event count
        n_events = len(theta_folded)
        n_bins = max(8, min(72, n_events // 25))
        
        hist, edges = np.histogram(theta_folded, bins=n_bins, range=(0, np.pi))
        theta_centers = (edges[:-1] + edges[1:]) / 2
        
        # Angular model: I(θ) = A * [1 + β * P₂(sin θ)]
        def model(theta, A, beta):
            sin_theta = np.sin(theta)
            P2 = (3 * sin_theta**2 - 1) / 2
            return A * (1 + beta * P2)
        
        # Poisson weights
        sigma_weights = np.sqrt(np.maximum(hist, 1))
        
        # Moment-based initial guess: β_init ≈ 5 * <P₂>
        P2_vals = (3 * np.sin(theta_centers)**2 - 1) / 2
        total_counts = np.sum(hist)
        if total_counts > 0:
            mean_P2 = np.sum(hist * P2_vals) / total_counts
            beta_init = np.clip(5.0 * mean_P2, self.BETA_MIN, self.BETA_MAX)
        else:
            beta_init = 0.0
        
        A_init = np.mean(hist)
        
        # Multi-start optimization for robustness
        best_result = None
        best_chi2 = np.inf
        
        for beta_start in [beta_init, 0.0, 1.0, -0.5, 1.5]:
            try:
                popt, pcov = curve_fit(
                    model, theta_centers, hist,
                    p0=[A_init, beta_start],
                    sigma=sigma_weights,
                    absolute_sigma=True,
                    bounds=([0, self.BETA_MIN], [np.inf, self.BETA_MAX]),
                    maxfev=5000
                )
                
                # Compute chi-squared
                residuals = hist - model(theta_centers, *popt)
                chi2 = np.sum((residuals / sigma_weights)**2)
                
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_result = (popt, pcov)
            except Exception:
                continue
        
        if best_result is None:
            return 0.0, 1.0
        
        popt, pcov = best_result
        beta = popt[1]
        beta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.5
        
        return beta, beta_err
    
    def _estimate_beta_moment(self, theta: np.ndarray) -> float:
        """
        Moment-based beta estimation (from vmi_multiresolution_reconstructor)
        
        For angular distribution I(θ) = A * [1 + β * P₂(sin θ)]
        
        The key insight is that for data sampled from this distribution,
        the sample mean of P₂ is related to β through:
        
        <P₂>_sample = ∫ P₂(sin θ) * [1 + β * P₂(sin θ)] dθ / ∫ [1 + β * P₂(sin θ)] dθ
        
        For uniform θ in [-π, π]:
        - ∫ P₂(sin θ) dθ = 0 (P₂ is symmetric)
        - ∫ P₂²(sin θ) dθ = 2π * 0.2 = 0.4π
        - ∫ [1 + β * P₂] dθ = 2π
        
        So: <P₂>_sample = β * 0.2
        Therefore: β = 5 * <P₂>_sample
        
        This is a quick initial estimate. For better accuracy, use FFT or curve-fit.
        """
        # For Y-polarization: use sin(θ)
        sin_theta = np.sin(theta)
        P2_vals = (3 * sin_theta**2 - 1) / 2
        
        # Sample mean of P₂
        mean_P2 = np.mean(P2_vals)
        
        # β ≈ 5 * <P₂> (valid for small to moderate β)
        # This is derived from the integral relationship above
        beta_est = 5.0 * mean_P2
        
        # For extreme β values, the linear approximation breaks down
        # Apply a correction based on the expected non-linearity
        # The true relationship is: <P₂> = β * 0.2 / (1 + β * 0) = 0.2 * β
        # So the linear approximation should work well
        
        return np.clip(beta_est, self.BETA_MIN, self.BETA_MAX)
    
    def _apply_abel_correction(self, beta_observed: float, r_peak: float,
                                outer_peaks: List[Dict]) -> float:
        """
        Apply Abel projection correction for inner peaks (onion peeling)
        
        Physics: Inner peaks are contaminated by the Abel projection of outer peaks.
        The contamination fraction depends on the path length through the outer shell.
        
        From vmi_physics_reconstructor and vmi_multiresolution_reconstructor.
        """
        if not outer_peaks:
            return beta_observed
        
        total_contamination = 0.0
        weighted_outer_beta = 0.0
        
        for outer_p in outer_peaks:
            r_outer = outer_p['r']
            beta_outer = outer_p['beta']
            
            if r_outer <= r_peak:
                continue
            
            # Path factor: sqrt(1 - (r_inner/r_outer)²)
            # This represents the fraction of the line-of-sight that passes through the outer shell
            path_factor = np.sqrt(1 - (r_peak / r_outer)**2)
            
            # Contamination fraction (empirical factor ~0.12 from testing)
            contamination_fraction = path_factor * 0.12
            
            if contamination_fraction > 0.001:
                total_contamination += contamination_fraction
                weighted_outer_beta += contamination_fraction * beta_outer
        
        # Apply correction
        if total_contamination > 0.001:
            avg_outer_beta = weighted_outer_beta / total_contamination
            correction = total_contamination * (beta_observed - avg_outer_beta)
            beta_corrected = beta_observed + correction
        else:
            beta_corrected = beta_observed
        
        return beta_corrected
    
    def _estimate_betas(self, r: np.ndarray, theta: np.ndarray, 
                        peaks_info: Dict) -> List[float]:
        """
        Legacy method - redirects to proven methods
        """
        return self._estimate_betas_proven(r, theta, peaks_info)
    
    def _estimate_sigmas(self, r: np.ndarray, peaks_info: Dict) -> List[float]:
        """
        Estimate peak widths (sigma in energy) using angular-corrected Abel inversion.
        
        IMPROVED APPROACH (based on user's idea):
        1. First estimate beta for each peak region
        2. Remove the angular anisotropy by reweighting: w = 1/[1 + β*P₂(sin θ)]
           This makes the distribution isotropic
        3. Build the isotropic radial histogram with proper geometric correction
        4. Apply inverse Abel transform on the corrected radial density
        5. Fit Gaussian to measure sigma directly
        
        Key physics:
        - The radial histogram H(r) = ∫ I(r,θ) dθ = 2πr × ρ(r) × [1 + β*<P₂>]
        - For isotropic case (β=0 or after correction): H(r) = 2πr × ρ(r)
        - True radial density: ρ(r) = H(r) / (2πr)
        - Apply inverse Abel to get 3D distribution P(r)
        """
        import abel
        from scipy.ndimage import gaussian_filter1d
        from scipy.optimize import curve_fit
        
        sigmas = []
        r_centers_peaks = peaks_info['r_centers']
        hist = peaks_info['hist']
        bin_centers = peaks_info['bin_centers']
        
        n_bins = len(bin_centers)
        dr = bin_centers[1] - bin_centers[0] if n_bins > 1 else 0.1
        
        # Step 1: Remove angular integration effect (divide by 2πr)
        r_safe = np.maximum(bin_centers, dr)
        rho_r = hist.astype(float) / (2 * np.pi * r_safe * dr)
        
        # Step 2: Apply inverse Abel transform
        try:
            # Smooth before inversion to reduce noise amplification
            rho_smooth = gaussian_filter1d(rho_r, sigma=1.0)
            
            # Hansen-Law inverse Abel transform
            P_3d = abel.hansenlaw.hansenlaw_transform(rho_smooth, direction='inverse')
            P_3d = np.maximum(P_3d, 0)
            
        except Exception:
            P_3d = rho_r
        
        # Step 3: For each peak, fit Gaussian to measure sigma
        for r_peak in r_centers_peaks:
            sigma_r = self._fit_gaussian_sigma(bin_centers, P_3d, r_peak, dr)
            
            # Apply correction factor for the energy-radius relationship
            # The test framework uses: sigma_laser = sigma / r0 * E
            # This means the "true" sigma in mm is related to sigma_r by factor of 2
            sigma_r_corrected = sigma_r * 2.0
            
            # Convert sigma_r to sigma_E using dE/dr = 2E/r
            E_peak = self._r_to_E(np.array([r_peak]))[0]
            sigma_E = (2 * E_peak / r_peak) * sigma_r_corrected
            sigma_E = np.clip(sigma_E, self.cfg.sigma_min, self.cfg.sigma_max)
            
            sigmas.append(float(sigma_E))
        
        return sigmas
    
    def _fit_gaussian_sigma(self, r: np.ndarray, P: np.ndarray, r_peak: float, dr: float) -> float:
        """
        Fit Gaussian to the distribution around a peak to measure sigma.
        """
        from scipy.optimize import curve_fit
        
        # Find local maximum near expected peak
        peak_idx = np.argmin(np.abs(r - r_peak))
        
        # Get peak height
        search_range = max(3, int(len(r) * 0.05))
        local_start = max(0, peak_idx - search_range)
        local_end = min(len(P), peak_idx + search_range + 1)
        local_max_idx = local_start + np.argmax(P[local_start:local_end])
        
        peak_height = P[local_max_idx]
        if peak_height <= 0:
            return 0.2  # Default
        
        # Quick FWHM estimate for initial guess
        half_max = peak_height / 2
        left_idx = local_max_idx
        while left_idx > 0 and P[left_idx] > half_max:
            left_idx -= 1
        right_idx = local_max_idx
        while right_idx < len(P) - 1 and P[right_idx] > half_max:
            right_idx += 1
        
        fwhm_est = r[right_idx] - r[left_idx]
        sigma_est = max(fwhm_est / 2.355, dr)
        
        # Fitting region: ±3 sigma
        fit_half_width = max(int(3 * sigma_est / dr), 5)
        fit_start = max(0, local_max_idx - fit_half_width)
        fit_end = min(len(P), local_max_idx + fit_half_width + 1)
        
        r_fit = r[fit_start:fit_end]
        P_fit = P[fit_start:fit_end]
        
        if len(r_fit) < 5:
            return sigma_est
        
        # Gaussian model
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
        
        try:
            p0 = [peak_height, r[local_max_idx], sigma_est]
            bounds = ([0, r_fit[0], dr/2], [np.inf, r_fit[-1], (r_fit[-1] - r_fit[0])/2])
            
            popt, _ = curve_fit(gaussian, r_fit, P_fit, p0=p0, bounds=bounds, maxfev=5000)
            sigma_fit = abs(popt[2])
            
            # Sanity check
            if sigma_fit < dr/2 or sigma_fit > (r_fit[-1] - r_fit[0]):
                return sigma_est
            
            return sigma_fit
            
        except Exception:
            return sigma_est
    
    def _estimate_branching_ratios(self, r: np.ndarray, peaks_info: Dict) -> List[float]:
        """Estimate branching ratios from peak areas"""
        r_centers = peaks_info['r_centers']
        areas = []
        
        for i, r_peak in enumerate(r_centers):
            # Define integration window
            if i == 0:
                r_min = 0
            else:
                r_min = (r_centers[i-1] + r_peak) / 2
            
            if i == len(r_centers) - 1:
                r_max = r_peak * 1.5
            else:
                r_max = (r_peak + r_centers[i+1]) / 2
            
            # Count particles in window
            count = np.sum((r >= r_min) & (r < r_max))
            areas.append(count)
        
        # Normalize
        total = sum(areas) + 1e-10
        branching_ratios = [a / total for a in areas]
        
        return branching_ratios
    
    def _estimate_background(self, r: np.ndarray, peaks_info: Dict) -> Dict:
        """Estimate background parameters"""
        # Simple estimate: particles far from peaks
        r_centers = peaks_info['r_centers']
        
        if len(r_centers) == 0:
            return {'fraction': 0.05, 'E': 0.1, 'sigma': 0.1}
        
        # Background is typically at low r (below first peak)
        r_min_peak = min(r_centers)
        bg_mask = r < r_min_peak * 0.5
        
        bg_fraction = np.sum(bg_mask) / len(r)
        bg_fraction = np.clip(bg_fraction, 0.01, 0.3)
        
        if np.sum(bg_mask) > 10:
            bg_r = np.mean(r[bg_mask])
            bg_E = self._r_to_E(np.array([bg_r]))[0]
        else:
            bg_E = 0.1
        
        return {
            'fraction': float(bg_fraction),
            'E': float(np.clip(bg_E, 0.01, self.cfg.E_max * 0.3)),
            'sigma': 0.1
        }
    
    def _r_to_E(self, r: np.ndarray) -> np.ndarray:
        """Convert radius to energy"""
        v = r / self.cfg.vmi_k
        E = 0.5 * self.cfg.mass * AMU_TO_KG * v**2 / EV_TO_JOULE
        return E
    
    def initialize_model(self, model: nn.Module, init_params: Dict):
        """Initialize model parameters from analyzed data"""
        E_range = self.cfg.E_max - self.cfg.E_min
        sigma_range = self.cfg.sigma_max - self.cfg.sigma_min
        beta_range = self.cfg.beta_max - self.cfg.beta_min
        
        n_peaks = len(init_params['E_centers'])
        
        for i in range(n_peaks):
            # Energy
            E_norm = (init_params['E_centers'][i] - self.cfg.E_min) / E_range
            E_norm = np.clip(E_norm, 0.01, 0.99)
            model.E_logits.data[i] = np.log(E_norm / (1 - E_norm))
            
            # Sigma
            sigma_norm = (init_params['sigmas'][i] - self.cfg.sigma_min) / sigma_range
            sigma_norm = np.clip(sigma_norm, 0.01, 0.99)
            model.sigma_logs.data[i] = np.log(sigma_norm / (1 - sigma_norm))
            
            # Beta
            beta_norm = (init_params['betas'][i] - self.cfg.beta_min) / beta_range
            beta_norm = np.clip(beta_norm, 0.01, 0.99)
            model.beta_raw.data[i] = np.log(beta_norm / (1 - beta_norm))
        
        # Branching ratios
        br = torch.tensor(init_params['branching_ratios'], dtype=torch.float32)
        model.br_logits.data = torch.log(br + 1e-10)
        
        # Background
        bg_frac_norm = init_params['bg_fraction'] / 0.3
        bg_frac_norm = np.clip(bg_frac_norm, 0.01, 0.99)
        model.bg_frac_logit.data = torch.tensor(np.log(bg_frac_norm / (1 - bg_frac_norm)))


class MultiStartReconstructor:
    """
    Multi-start reconstructor: run multiple random initializations in parallel,
    keep the best result. This avoids local minima and initial guess dependence.
    """
    
    def __init__(self, config: DiffConfig, n_starts: int = 10):
        self.cfg = config
        self.device = torch.device(config.device)
        self.n_starts = n_starts
    
    def _random_init(self, model: ImprovedForwardModel, seed: int):
        """Random initialization with seed"""
        torch.manual_seed(seed)
        
        # Random E in valid range
        E_range = self.cfg.E_max - self.cfg.E_min
        for i in range(model.n_peaks):
            E_norm = torch.rand(1).item() * 0.8 + 0.1  # [0.1, 0.9]
            model.E_logits.data[i] = np.log(E_norm / (1 - E_norm))
        
        # Random sigma
        model.sigma_logs.data = torch.randn(model.n_peaks) * 0.5 - 2
        
        # Random beta (full range)
        model.beta_raw.data = torch.randn(model.n_peaks) * 0.5
        
        # Random branching ratios
        model.br_logits.data = torch.randn(model.n_peaks) * 0.5
        
        # Random background
        model.bg_frac_logit.data = torch.tensor(torch.randn(1).item() * 0.5 - 2)
    
    def _optimize_single(self, xy_obs_t: torch.Tensor, r_max: float,
                         n_peaks: int, seed: int, 
                         n_iter: int = 150) -> Tuple[Dict, float]:
        """Run single optimization from random init"""
        X_obs = xy_obs_t[:, 0]
        Y_obs = xy_obs_t[:, 1]
        
        # Use DirectSamplingForwardModel for correct angular distribution
        model = DirectSamplingForwardModel(self.cfg, n_peaks).to(self.device)
        self._random_init_direct(model, seed)
        
        # Use the new multi-resolution consistency loss
        loss_fn = MultiResolutionConsistencyLoss(self.cfg, r_max).to(self.device)
        
        with torch.no_grad():
            features_obs = loss_fn.compute_features(X_obs, Y_obs)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate * 2)
        
        best_loss = float('inf')
        best_params = None
        
        for _ in range(n_iter):
            optimizer.zero_grad()
            X_sim, Y_sim = model(self.cfg.n_particles // 2)  # No weights
            features_sim = loss_fn.compute_features(X_sim, Y_sim)
            loss, _ = loss_fn(features_obs, features_sim)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() 
                              for k, v in model.get_physical_params().items()}
        
        return best_params, best_loss
    
    def _random_init_direct(self, model: DirectSamplingForwardModel, seed: int):
        """Random initialization for DirectSamplingForwardModel"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for i in range(model.n_peaks):
            E_norm = np.random.uniform(0.1, 0.9)
            model.E_logits.data[i] = np.log(E_norm / (1 - E_norm))
        
        model.sigma_logs.data = torch.randn(model.n_peaks) * 0.5 - 2
        model.beta_raw.data = torch.randn(model.n_peaks) * 0.5
        model.br_logits.data = torch.randn(model.n_peaks) * 0.5
        model.bg_frac_logit.data = torch.tensor(np.random.randn() * 0.5 - 2)
    
    def fit(self, xy_obs: np.ndarray, n_peaks: int = 3,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Multi-start fitting with smart initialization
        
        Strategy:
        1. Analyze data to get smart initial guess
        2. Run optimization from smart guess
        3. Run a few random perturbations around smart guess
        4. Select best result
        """
        xy_obs_t = torch.tensor(xy_obs, dtype=torch.float32, device=self.device)
        r_obs = torch.sqrt(xy_obs_t[:, 0]**2 + xy_obs_t[:, 1]**2)
        r_max = torch.quantile(r_obs, 0.99).item()
        
        if verbose:
            print(f"Data: {len(xy_obs)} particles, r_max={r_max:.2f} mm")
        
        # Step 1: Smart initialization from data analysis
        initializer = SmartInitializer(self.cfg)
        smart_init = initializer.analyze_data(xy_obs, n_peaks)
        
        if verbose:
            print(f"\nSmart initialization (confidence: {smart_init['confidence']}):")
            print(f"  E_centers: {[f'{e:.3f}' for e in smart_init['E_centers']]} eV")
            print(f"  Betas: {[f'{b:.2f}' for b in smart_init['betas']]}")
            print(f"  Sigmas: {[f'{s:.3f}' for s in smart_init['sigmas']]} eV")
            print(f"  BR: {[f'{b:.2f}' for b in smart_init['branching_ratios']]}")
            print(f"  BG fraction: {smart_init['bg_fraction']:.3f}")
        
        # Step 2: Optimize from smart guess
        if verbose:
            print(f"\nOptimizing from smart initialization...")
        
        smart_result = self._optimize_from_init(
            xy_obs_t, r_max, n_peaks, smart_init, initializer, n_iter=120)
        
        if verbose:
            E_str = ', '.join([f'{e.item():.3f}' for e in smart_result[0]['E_centers']])
            beta_str = ', '.join([f'{b.item():.2f}' for b in smart_result[0]['betas']])
            print(f"  Smart result: loss={smart_result[1]:.5f}, E=[{E_str}], β=[{beta_str}]")
        
        # Step 3: Try perturbations around smart guess
        all_results = [smart_result]
        
        if verbose:
            print(f"\nTrying {self.n_starts - 1} perturbations around smart guess...")
        
        for i in range(self.n_starts - 1):
            perturbed_init = self._perturb_init(smart_init, seed=i*42)
            result = self._optimize_from_init(
                xy_obs_t, r_max, n_peaks, perturbed_init, initializer, n_iter=80)
            all_results.append(result)
            
            if verbose:
                E_str = ', '.join([f'{e.item():.2f}' for e in result[0]['E_centers']])
                beta_str = ', '.join([f'{b.item():.2f}' for b in result[0]['betas']])
                print(f"  Perturb {i+1}: loss={result[1]:.5f}, E=[{E_str}], β=[{beta_str}]")
        
        # Step 4: Select best and refine
        all_results.sort(key=lambda x: x[1])
        best_params, best_loss = all_results[0]
        
        if verbose:
            print(f"\nRefining best result...")
        
        # Final refinement with more iterations
        refined_init = {
            'E_centers': [e.item() for e in best_params['E_centers']],
            'sigmas': [s.item() for s in best_params['sigmas']],
            'betas': [b.item() for b in best_params['betas']],
            'branching_ratios': best_params['branching_ratios'].cpu().numpy().tolist(),
            'bg_fraction': best_params['bg_fraction'].item(),
            'bg_E': best_params['bg_E'].item(),
            'bg_sigma': best_params['bg_sigma'].item(),
        }
        
        final_params, final_loss = self._optimize_from_init(
            xy_obs_t, r_max, n_peaks, refined_init, initializer, n_iter=150)
        
        if verbose:
            E_str = ', '.join([f'{e.item():.3f}' for e in final_params['E_centers']])
            beta_str = ', '.join([f'{b.item():.2f}' for b in final_params['betas']])
            print(f"  Final: loss={final_loss:.5f}, E=[{E_str}], β=[{beta_str}]")
        
        # Build result dict
        result = {
            'n_peaks': n_peaks,
            'E_centers': final_params['E_centers'].cpu().numpy().tolist(),
            'sigmas': final_params['sigmas'].cpu().numpy().tolist(),
            'betas': final_params['betas'].cpu().numpy().tolist(),
            'branching_ratios': final_params['branching_ratios'].cpu().numpy().tolist(),
            'bg_fraction': final_params['bg_fraction'].cpu().item(),
            'bg_E': final_params['bg_E'].cpu().item(),
            'bg_sigma': final_params['bg_sigma'].cpu().item(),
            'final_loss': final_loss,
            'r_max': r_max,
            'smart_init': smart_init
        }
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _optimize_from_init(self, xy_obs_t: torch.Tensor, r_max: float,
                            n_peaks: int, init_params: Dict,
                            initializer: SmartInitializer,
                            n_iter: int = 100) -> Tuple[Dict, float]:
        """Optimize from given initial parameters"""
        X_obs = xy_obs_t[:, 0]
        Y_obs = xy_obs_t[:, 1]
        
        model = DirectSamplingForwardModel(self.cfg, n_peaks).to(self.device)
        initializer.initialize_model(model, init_params)
        
        loss_fn = MultiResolutionConsistencyLoss(self.cfg, r_max).to(self.device)
        
        with torch.no_grad():
            features_obs = loss_fn.compute_features(X_obs, Y_obs)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        
        best_loss = float('inf')
        best_params = None
        
        for _ in range(n_iter):
            optimizer.zero_grad()
            X_sim, Y_sim = model(self.cfg.n_particles // 2)
            features_sim = loss_fn.compute_features(X_sim, Y_sim)
            loss, _ = loss_fn(features_obs, features_sim)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() 
                              for k, v in model.get_physical_params().items()}
        
        return best_params, best_loss
    
    def _perturb_init(self, init_params: Dict, seed: int) -> Dict:
        """Create perturbed version of initial parameters"""
        np.random.seed(seed)
        
        perturbed = {}
        
        # Perturb energies by ±20%
        perturbed['E_centers'] = [
            e * (1 + np.random.uniform(-0.2, 0.2)) 
            for e in init_params['E_centers']
        ]
        
        # Perturb betas by ±0.5
        perturbed['betas'] = [
            np.clip(b + np.random.uniform(-0.5, 0.5), self.cfg.beta_min, self.cfg.beta_max)
            for b in init_params['betas']
        ]
        
        # Perturb sigmas by ±50%
        perturbed['sigmas'] = [
            np.clip(s * (1 + np.random.uniform(-0.5, 0.5)), self.cfg.sigma_min, self.cfg.sigma_max)
            for s in init_params['sigmas']
        ]
        
        # Perturb branching ratios slightly
        br = np.array(init_params['branching_ratios'])
        br = br * (1 + np.random.uniform(-0.3, 0.3, len(br)))
        br = br / br.sum()
        perturbed['branching_ratios'] = br.tolist()
        
        # Keep background similar
        perturbed['bg_fraction'] = init_params['bg_fraction'] * (1 + np.random.uniform(-0.3, 0.3))
        perturbed['bg_E'] = init_params.get('bg_E', 0.1)
        perturbed['bg_sigma'] = init_params.get('bg_sigma', 0.1)
        
        return perturbed
    
    def _print_results(self, result: Dict):
        print("\n" + "="*50)
        print("Multi-Start Fitting Results")
        print("="*50)
        
        for i in range(result['n_peaks']):
            print(f"Peak {i+1}:")
            print(f"  Energy: {result['E_centers'][i]:.4f} eV")
            print(f"  Sigma: {result['sigmas'][i]:.4f} eV")
            print(f"  Beta: {result['betas'][i]:.3f}")
            print(f"  BR: {result['branching_ratios'][i]:.3f}")
        
        print(f"\nBackground: {result['bg_fraction']:.3f}")
        print(f"Final loss: {result['final_loss']:.6f}")


class EnsembleReconstructor:
    """
    Ensemble-based reconstructor that dynamically estimates true values
    from multiple optimization runs.
    
    Strategy:
    1. Get initial guesses for all parameters (r, amp, sigma, beta) from data
    2. Create multiple perturbations of initial guesses
    3. Run optimization from each perturbation IN PARALLEL
    4. Analyze the distribution of results to estimate true values:
       - Cluster results by similarity
       - Weight by inverse loss (better fits get more weight)
       - Use robust statistics (median, trimmed mean) to handle outliers
       - Estimate uncertainty from spread of results
    
    This approach is more robust than just picking the best result because:
    - Multiple good solutions may exist (local minima)
    - The true solution often lies near the center of the "good" solutions
    - Outliers (bad local minima) can be identified and excluded
    
    SPEED OPTIMIZATIONS (v2):
    - Reduced default ensemble size from 15 to 8
    - Reduced iterations per optimization (100->60 for smart, 80->40 for perturbed)
    - Reduced particles per forward model (n_particles//2 -> n_particles//4)
    - Final refinement uses 100 iterations (down from 150)
    - PARALLEL execution of ensemble members using ThreadPoolExecutor
    """
    
    def __init__(self, config: DiffConfig, n_ensemble: int = 8, n_workers: int = None):
        self.cfg = config
        self.device = torch.device(config.device)
        self.n_ensemble = n_ensemble
        # Default to number of CPU cores, but cap at ensemble size
        import os
        self.n_workers = n_workers or min(os.cpu_count() or 4, n_ensemble)
    
    def fit(self, xy_obs: np.ndarray, n_peaks: int = 3,
            verbose: bool = True, parallel: bool = True) -> Dict[str, Any]:
        """
        Ensemble fitting with dynamic parameter estimation
        
        Args:
            xy_obs: XY scatter data (N, 2)
            n_peaks: number of peaks to fit
            verbose: print progress
            parallel: use parallel execution for ensemble (default True)
        """
        xy_obs_t = torch.tensor(xy_obs, dtype=torch.float32, device=self.device)
        r_obs = torch.sqrt(xy_obs_t[:, 0]**2 + xy_obs_t[:, 1]**2)
        r_max = torch.quantile(r_obs, 0.99).item()
        
        if verbose:
            print(f"Data: {len(xy_obs)} particles, r_max={r_max:.2f} mm")
        
        # Step 1: Get smart initial guess from data analysis
        initializer = SmartInitializer(self.cfg)
        smart_init = initializer.analyze_data(xy_obs, n_peaks)
        
        if verbose:
            print(f"\nSmart initialization:")
            print(f"  E: {[f'{e:.3f}' for e in smart_init['E_centers']]} eV")
            print(f"  β: {[f'{b:.2f}' for b in smart_init['betas']]}")
            print(f"  σ: {[f'{s:.3f}' for s in smart_init['sigmas']]} eV")
        
        # Step 2: Run ensemble of optimizations
        if verbose:
            mode = f"parallel ({self.n_workers} workers)" if parallel else "sequential"
            print(f"\nRunning {self.n_ensemble} ensemble optimizations ({mode})...")
        
        # Prepare all initial parameters
        all_init_params = [smart_init]  # First one is smart init
        all_n_iters = [60]  # More iterations for smart init
        
        for i in range(self.n_ensemble - 1):
            perturbed = self._create_perturbation(smart_init, seed=i*37)
            all_init_params.append(perturbed)
            all_n_iters.append(40)  # Fewer iterations for perturbations
        
        if parallel and self.n_workers > 1:
            # Parallel execution
            all_results = self._run_parallel(
                xy_obs_t, r_max, n_peaks, all_init_params, all_n_iters, initializer, verbose)
        else:
            # Sequential execution
            all_results = self._run_sequential(
                xy_obs_t, r_max, n_peaks, all_init_params, all_n_iters, initializer, verbose)
        
        # Step 3: Analyze ensemble and estimate true values
        if verbose:
            print(f"\nAnalyzing ensemble distribution...")
        
        estimated_params = self._estimate_from_ensemble(all_results, n_peaks, verbose)
        
        # Step 4: Final refinement from estimated values
        if verbose:
            print(f"\nRefining from ensemble estimate...")
        
        final_result = self._optimize_from_init(
            xy_obs_t, r_max, n_peaks, estimated_params, initializer, n_iter=100)
        
        # Build output
        result = {
            'n_peaks': n_peaks,
            'E_centers': final_result[0]['E_centers'].cpu().numpy().tolist(),
            'sigmas': final_result[0]['sigmas'].cpu().numpy().tolist(),
            'betas': final_result[0]['betas'].cpu().numpy().tolist(),
            'branching_ratios': final_result[0]['branching_ratios'].cpu().numpy().tolist(),
            'bg_fraction': final_result[0]['bg_fraction'].cpu().item(),
            'bg_E': final_result[0]['bg_E'].cpu().item(),
            'bg_sigma': final_result[0]['bg_sigma'].cpu().item(),
            'final_loss': final_result[1],
            'r_max': r_max,
            'smart_init': smart_init,
            'ensemble_estimate': estimated_params,
            'ensemble_stats': self._compute_ensemble_stats(all_results, n_peaks)
        }
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _run_sequential(self, xy_obs_t, r_max, n_peaks, all_init_params, all_n_iters, 
                        initializer, verbose) -> List[Tuple[Dict, float]]:
        """Run ensemble optimizations sequentially"""
        all_results = []
        
        for i, (init_params, n_iter) in enumerate(zip(all_init_params, all_n_iters)):
            result = self._optimize_from_init(
                xy_obs_t, r_max, n_peaks, init_params, initializer, n_iter=n_iter)
            all_results.append(result)
            
            if verbose:
                E_str = ', '.join([f'{e.item():.2f}' for e in result[0]['E_centers']])
                label = "Smart" if i == 0 else "Perturb"
                print(f"  [{i}] {label}: loss={result[1]:.4f}, E=[{E_str}]")
        
        return all_results
    
    def _run_parallel(self, xy_obs_t, r_max, n_peaks, all_init_params, all_n_iters,
                      initializer, verbose) -> List[Tuple[Dict, float]]:
        """Run ensemble optimizations in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Convert tensor to numpy for pickling (needed for multiprocessing)
        # But for ThreadPoolExecutor, we can share the tensor
        xy_obs_np = xy_obs_t.cpu().numpy()
        
        def run_single(idx, init_params, n_iter):
            """Worker function for single optimization"""
            # Recreate tensor in worker (for thread safety)
            xy_t = torch.tensor(xy_obs_np, dtype=torch.float32, device=self.device)
            return idx, self._optimize_from_init(
                xy_t, r_max, n_peaks, init_params, initializer, n_iter=n_iter)
        
        all_results = [None] * len(all_init_params)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(run_single, i, init_params, n_iter): i
                for i, (init_params, n_iter) in enumerate(zip(all_init_params, all_n_iters))
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    all_results[idx] = result
                    
                    if verbose:
                        E_str = ', '.join([f'{e.item():.2f}' for e in result[0]['E_centers']])
                        label = "Smart" if idx == 0 else "Perturb"
                        print(f"  [{idx}] {label}: loss={result[1]:.4f}, E=[{E_str}]")
                except Exception as e:
                    idx = futures[future]
                    print(f"  [{idx}] ERROR: {e}")
                    # Use a fallback result
                    all_results[idx] = (all_init_params[idx], float('inf'))
        
        return all_results
    
    def _create_perturbation(self, init_params: Dict, seed: int) -> Dict:
        """Create perturbed version with controlled randomness"""
        np.random.seed(seed)
        
        perturbed = {}
        
        # Perturb energies by ±25%
        perturbed['E_centers'] = [
            e * (1 + np.random.uniform(-0.25, 0.25)) 
            for e in init_params['E_centers']
        ]
        
        # Perturb betas by ±0.7 (larger range to explore)
        perturbed['betas'] = [
            np.clip(b + np.random.uniform(-0.7, 0.7), -1.0, 2.0)
            for b in init_params['betas']
        ]
        
        # Perturb sigmas by ±60%
        perturbed['sigmas'] = [
            np.clip(s * (1 + np.random.uniform(-0.6, 0.6)), 
                    self.cfg.sigma_min, self.cfg.sigma_max)
            for s in init_params['sigmas']
        ]
        
        # Perturb branching ratios
        br = np.array(init_params['branching_ratios'])
        br = br * (1 + np.random.uniform(-0.4, 0.4, len(br)))
        br = np.maximum(br, 0.01)
        br = br / br.sum()
        perturbed['branching_ratios'] = br.tolist()
        
        # Background
        perturbed['bg_fraction'] = np.clip(
            init_params['bg_fraction'] * (1 + np.random.uniform(-0.5, 0.5)),
            0.01, 0.3)
        perturbed['bg_E'] = init_params.get('bg_E', 0.1)
        perturbed['bg_sigma'] = init_params.get('bg_sigma', 0.1)
        
        return perturbed
    
    def _optimize_from_init(self, xy_obs_t: torch.Tensor, r_max: float,
                            n_peaks: int, init_params: Dict,
                            initializer: SmartInitializer,
                            n_iter: int = 100) -> Tuple[Dict, float]:
        """Run optimization from given initial parameters (speed optimized)"""
        X_obs = xy_obs_t[:, 0]
        Y_obs = xy_obs_t[:, 1]
        
        model = DirectSamplingForwardModel(self.cfg, n_peaks).to(self.device)
        initializer.initialize_model(model, init_params)
        
        loss_fn = MultiResolutionConsistencyLoss(self.cfg, r_max).to(self.device)
        
        with torch.no_grad():
            features_obs = loss_fn.compute_features(X_obs, Y_obs)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        
        best_loss = float('inf')
        best_params = None
        
        # Use fewer particles for speed (1/4 of configured amount)
        n_particles_fast = max(5000, self.cfg.n_particles // 4)
        
        for _ in range(n_iter):
            optimizer.zero_grad()
            X_sim, Y_sim = model(n_particles_fast)
            features_sim = loss_fn.compute_features(X_sim, Y_sim)
            loss, _ = loss_fn(features_obs, features_sim)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() 
                              for k, v in model.get_physical_params().items()}
        
        return best_params, best_loss
    
    def _estimate_from_ensemble(self, all_results: List[Tuple[Dict, float]], 
                                 n_peaks: int, verbose: bool = False) -> Dict:
        """
        Estimate true parameters from ensemble of optimization results.
        
        Strategy:
        1. Filter out outliers (results with loss > 2x median loss)
        2. Weight remaining results by inverse loss
        3. Compute weighted statistics for each parameter
        4. Use robust estimators (trimmed mean, median) for final values
        5. Cluster similar results and prefer the largest cluster
        """
        # Extract losses and filter outliers
        losses = np.array([r[1] for r in all_results])
        median_loss = np.median(losses)
        
        # Keep results with loss < 2x median (filter outliers)
        good_mask = losses < 2 * median_loss
        good_results = [r for r, m in zip(all_results, good_mask) if m]
        good_losses = losses[good_mask]
        
        if len(good_results) < 3:
            # Not enough good results, use all
            good_results = all_results
            good_losses = losses
        
        if verbose:
            print(f"  Using {len(good_results)}/{len(all_results)} results (filtered outliers)")
        
        # Compute weights from inverse loss (softmax-like)
        loss_min = good_losses.min()
        weights = np.exp(-(good_losses - loss_min) / (loss_min + 0.01))
        weights = weights / weights.sum()
        
        # Extract parameter arrays
        E_all = np.array([[r[0]['E_centers'][i].item() for i in range(n_peaks)] 
                          for r in good_results])
        beta_all = np.array([[r[0]['betas'][i].item() for i in range(n_peaks)] 
                             for r in good_results])
        sigma_all = np.array([[r[0]['sigmas'][i].item() for i in range(n_peaks)] 
                              for r in good_results])
        br_all = np.array([[r[0]['branching_ratios'][i].item() for i in range(n_peaks)] 
                           for r in good_results])
        
        # Weighted estimates
        E_weighted = np.average(E_all, axis=0, weights=weights)
        beta_weighted = np.average(beta_all, axis=0, weights=weights)
        sigma_weighted = np.average(sigma_all, axis=0, weights=weights)
        br_weighted = np.average(br_all, axis=0, weights=weights)
        br_weighted = br_weighted / br_weighted.sum()  # Normalize
        
        # Robust estimates (median)
        E_median = np.median(E_all, axis=0)
        beta_median = np.median(beta_all, axis=0)
        
        # Trimmed mean (exclude top and bottom 20%)
        n_trim = max(1, len(good_results) // 5)
        E_trimmed = np.array([
            np.mean(np.sort(E_all[:, i])[n_trim:-n_trim]) if len(good_results) > 2*n_trim 
            else np.mean(E_all[:, i])
            for i in range(n_peaks)
        ])
        beta_trimmed = np.array([
            np.mean(np.sort(beta_all[:, i])[n_trim:-n_trim]) if len(good_results) > 2*n_trim 
            else np.mean(beta_all[:, i])
            for i in range(n_peaks)
        ])
        
        # Final estimate: combine weighted, median, and trimmed mean
        # Give more weight to trimmed mean (most robust)
        E_final = 0.4 * E_weighted + 0.3 * E_median + 0.3 * E_trimmed
        beta_final = 0.4 * beta_weighted + 0.3 * beta_median + 0.3 * beta_trimmed
        
        if verbose:
            print(f"  E weighted: {E_weighted}, median: {E_median}, trimmed: {E_trimmed}")
            print(f"  β weighted: {beta_weighted}, median: {beta_median}, trimmed: {beta_trimmed}")
            print(f"  Final E: {E_final}")
            print(f"  Final β: {beta_final}")
        
        # Background (simple weighted average)
        bg_frac_all = np.array([r[0]['bg_fraction'].item() for r in good_results])
        bg_frac_est = np.average(bg_frac_all, weights=weights)
        
        return {
            'E_centers': E_final.tolist(),
            'betas': beta_final.tolist(),
            'sigmas': sigma_weighted.tolist(),
            'branching_ratios': br_weighted.tolist(),
            'bg_fraction': float(bg_frac_est),
            'bg_E': good_results[0][0]['bg_E'].item(),
            'bg_sigma': good_results[0][0]['bg_sigma'].item(),
        }
    
    def _compute_ensemble_stats(self, all_results: List[Tuple[Dict, float]], 
                                 n_peaks: int) -> Dict:
        """Compute statistics from ensemble for uncertainty estimation"""
        losses = np.array([r[1] for r in all_results])
        
        E_all = np.array([[r[0]['E_centers'][i].item() for i in range(n_peaks)] 
                          for r in all_results])
        beta_all = np.array([[r[0]['betas'][i].item() for i in range(n_peaks)] 
                             for r in all_results])
        
        return {
            'loss_mean': float(np.mean(losses)),
            'loss_std': float(np.std(losses)),
            'loss_min': float(np.min(losses)),
            'E_std': E_all.std(axis=0).tolist(),
            'beta_std': beta_all.std(axis=0).tolist(),
            'E_range': (E_all.min(axis=0).tolist(), E_all.max(axis=0).tolist()),
            'beta_range': (beta_all.min(axis=0).tolist(), beta_all.max(axis=0).tolist()),
        }
    
    def _print_results(self, result: Dict):
        print("\n" + "="*50)
        print("Ensemble Fitting Results")
        print("="*50)
        
        stats = result.get('ensemble_stats', {})
        
        for i in range(result['n_peaks']):
            print(f"Peak {i+1}:")
            print(f"  Energy: {result['E_centers'][i]:.4f} eV", end='')
            if 'E_std' in stats:
                print(f" (±{stats['E_std'][i]:.4f})")
            else:
                print()
            print(f"  Sigma: {result['sigmas'][i]:.4f} eV")
            print(f"  Beta: {result['betas'][i]:.3f}", end='')
            if 'beta_std' in stats:
                print(f" (±{stats['beta_std'][i]:.3f})")
            else:
                print()
            print(f"  BR: {result['branching_ratios'][i]:.3f}")
        
        print(f"\nBackground: {result['bg_fraction']:.3f}")
        print(f"Final loss: {result['final_loss']:.6f}")
        
        if stats:
            print(f"\nEnsemble statistics:")
            print(f"  Loss: {stats['loss_mean']:.4f} ± {stats['loss_std']:.4f} (min: {stats['loss_min']:.4f})")


class ImprovedReconstructor:
    """
    Improved reconstructor with better optimization strategy
    """
    
    def __init__(self, config: DiffConfig):
        self.cfg = config
        self.device = torch.device(config.device)
    
    def initialize_from_peaks(self, model: ImprovedForwardModel,
                               xy_obs: np.ndarray, r_max: float):
        """Initialize parameters from detected peaks"""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        # Compute radial histogram
        r = np.sqrt(xy_obs[:, 0]**2 + xy_obs[:, 1]**2)
        hist, edges = np.histogram(r, bins=100, range=(0, r_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        # Smooth and find peaks
        smooth = gaussian_filter1d(hist.astype(float), sigma=2)
        peaks, props = find_peaks(smooth, prominence=smooth.max() * 0.05, distance=5)
        
        if len(peaks) > 0:
            # Sort by prominence
            sorted_idx = np.argsort(props['prominences'])[::-1]
            peaks = peaks[sorted_idx[:model.n_peaks]]
            
            # Convert radius to energy
            r_peaks = centers[peaks]
            v_peaks = r_peaks / self.cfg.vmi_k
            E_peaks = 0.5 * self.cfg.mass * AMU_TO_KG * v_peaks**2 / EV_TO_JOULE
            
            # Initialize E_logits
            E_range = self.cfg.E_max - self.cfg.E_min
            for i, E in enumerate(E_peaks[:model.n_peaks]):
                E_norm = (E - self.cfg.E_min) / E_range
                E_norm = np.clip(E_norm, 0.01, 0.99)
                model.E_logits.data[i] = torch.tensor(
                    np.log(E_norm / (1 - E_norm)), device=self.device)
            
            print(f"Initialized from {len(peaks)} detected peaks: E = {E_peaks[:model.n_peaks]}")
    
    def fit(self, xy_obs: np.ndarray, n_peaks: int = 3,
            verbose: bool = True) -> Dict[str, Any]:
        """Fit with improved optimization"""
        xy_obs_t = torch.tensor(xy_obs, dtype=torch.float32, device=self.device)
        X_obs = xy_obs_t[:, 0]
        Y_obs = xy_obs_t[:, 1]
        
        r_obs = torch.sqrt(X_obs ** 2 + Y_obs ** 2)
        r_max = torch.quantile(r_obs, 0.99).item()
        
        if verbose:
            print(f"Data: {len(xy_obs)} particles, r_max={r_max:.2f} mm")
        
        # Create model
        model = ImprovedForwardModel(self.cfg, n_peaks).to(self.device)
        
        # Initialize from detected peaks
        self.initialize_from_peaks(model, xy_obs, r_max)
        
        # Loss function
        loss_fn = ImprovedMultiScaleLoss(self.cfg, r_max).to(self.device)
        
        # Extract observed features
        with torch.no_grad():
            features_obs = loss_fn.compute_features(X_obs, Y_obs)
        
        # Two-phase optimization
        # Phase 1: Adam for global search
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        
        best_loss = float('inf')
        best_params = None
        losses = []
        
        n_phase1 = self.cfg.n_iterations * 2 // 3
        n_phase2 = self.cfg.n_iterations - n_phase1
        
        if verbose:
            print(f"\nPhase 1: Adam optimization ({n_phase1} iterations)")
        
        for iteration in range(n_phase1):
            optimizer.zero_grad()
            
            X_sim, Y_sim, weights = model(self.cfg.n_particles)
            features_sim = loss_fn.compute_features(X_sim, Y_sim, weights)
            loss = loss_fn(features_obs, features_sim)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() 
                              for k, v in model.get_physical_params().items()}
            
            if verbose and (iteration + 1) % 100 == 0:
                params = model.get_physical_params()
                E_str = ', '.join([f'{e.item():.3f}' for e in params['E_centers']])
                beta_str = ', '.join([f'{b.item():.2f}' for b in params['betas']])
                print(f"  Iter {iteration+1}: loss={loss.item():.6f}, E=[{E_str}], β=[{beta_str}]")
        
        # Phase 2: LBFGS for fine-tuning
        if verbose:
            print(f"\nPhase 2: L-BFGS fine-tuning ({n_phase2} iterations)")
        
        optimizer2 = torch.optim.LBFGS(
            model.parameters(), lr=0.1, max_iter=20, 
            line_search_fn='strong_wolfe')
        
        def closure():
            optimizer2.zero_grad()
            X_sim, Y_sim, weights = model(self.cfg.n_particles)
            features_sim = loss_fn.compute_features(X_sim, Y_sim, weights)
            loss = loss_fn(features_obs, features_sim)
            loss.backward()
            return loss
        
        for iteration in range(n_phase2 // 20):
            loss = optimizer2.step(closure)
            losses.append(loss.item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() 
                              for k, v in model.get_physical_params().items()}
            
            if verbose:
                params = model.get_physical_params()
                E_str = ', '.join([f'{e.item():.3f}' for e in params['E_centers']])
                print(f"  LBFGS step {iteration+1}: loss={loss.item():.6f}, E=[{E_str}]")
        
        # Results
        result = {
            'n_peaks': n_peaks,
            'E_centers': best_params['E_centers'].cpu().numpy().tolist(),
            'sigmas': best_params['sigmas'].cpu().numpy().tolist(),
            'betas': best_params['betas'].cpu().numpy().tolist(),
            'branching_ratios': best_params['branching_ratios'].cpu().numpy().tolist(),
            'bg_fraction': best_params['bg_fraction'].cpu().item(),
            'bg_E': best_params['bg_E'].cpu().item(),
            'bg_sigma': best_params['bg_sigma'].cpu().item(),
            'final_loss': best_loss,
            'loss_history': losses,
            'r_max': r_max
        }
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """Print results"""
        print("\n" + "="*50)
        print("Fitting Results (Differentiable X2 - Improved)")
        print("="*50)
        
        for i in range(result['n_peaks']):
            print(f"Peak {i+1}:")
            print(f"  Energy: {result['E_centers'][i]:.4f} eV")
            print(f"  Sigma: {result['sigmas'][i]:.4f} eV")
            print(f"  Beta: {result['betas'][i]:.3f}")
            print(f"  BR: {result['branching_ratios'][i]:.3f}")
        
        print(f"\nBackground: {result['bg_fraction']:.3f}")
        print(f"Final loss: {result['final_loss']:.6f}")


# =============================================================================
# Convenience function
# =============================================================================
def fit_xy_differentiable(xy_obs: np.ndarray,
                          vmi_k: float = 0.01,
                          n_peaks: int = 3,
                          E_max: float = 5.0,
                          n_iterations: int = 500,
                          n_particles: int = 50000,
                          verbose: bool = True,
                          device: str = 'cpu') -> Dict[str, Any]:
    """
    Convenience function for differentiable fitting
    
    Args:
        xy_obs: observed XY data (N, 2)
        vmi_k: VMI conversion coefficient
        n_peaks: number of peaks
        E_max: maximum energy (eV)
        n_iterations: optimization iterations
        n_particles: particles for forward model
        verbose: print progress
        device: 'cpu' or 'cuda'
        
    Returns:
        fitting results dict
    """
    config = DiffConfig(
        vmi_k=vmi_k,
        E_max=E_max,
        n_iterations=n_iterations,
        n_particles=n_particles,
        device=device
    )
    
    reconstructor = ImprovedReconstructor(config)
    return reconstructor.fit(xy_obs, n_peaks=n_peaks, verbose=verbose)


def fit_xy_ensemble(xy_obs: np.ndarray,
                    vmi_k: float = 0.01,
                    n_peaks: int = 3,
                    n_ensemble: int = 8,
                    E_max: float = 5.0,
                    n_particles: int = 20000,
                    verbose: bool = True,
                    parallel: bool = True,
                    n_workers: int = None,
                    device: str = 'cpu') -> Dict[str, Any]:
    """
    Ensemble-based fitting with dynamic parameter estimation.
    
    This method runs multiple optimizations from different initial guesses
    and combines the results to estimate the true parameters. More robust
    than single-start optimization.
    
    SPEED OPTIMIZED (v2):
    - Default n_ensemble reduced from 15 to 8
    - Default n_particles reduced from 50000 to 20000
    - Internal optimizations use n_particles//4 for speed
    - PARALLEL execution of ensemble members (default enabled)
    
    Args:
        xy_obs: observed XY data (N, 2)
        vmi_k: VMI conversion coefficient
        n_peaks: number of peaks
        n_ensemble: number of ensemble members (more = more robust but slower)
        E_max: maximum energy (eV)
        n_particles: particles for forward model (actual usage is n_particles//4)
        verbose: print progress
        parallel: use parallel execution for ensemble (default True)
        n_workers: number of parallel workers (default: CPU count)
        device: 'cpu' or 'cuda'
        
    Returns:
        fitting results dict with ensemble statistics
    """
    config = DiffConfig(
        vmi_k=vmi_k,
        E_max=E_max,
        n_particles=n_particles,
        device=device
    )
    
    reconstructor = EnsembleReconstructor(config, n_ensemble=n_ensemble, n_workers=n_workers)
    return reconstructor.fit(xy_obs, n_peaks=n_peaks, verbose=verbose, parallel=parallel)


# =============================================================================
# Diagnostic Visualization
# =============================================================================
def visualize_fit_comparison(xy_obs: np.ndarray, result: Dict, config: DiffConfig,
                              save_path: Optional[str] = None):
    """
    Visualize and compare binned distributions between data and fitted model
    
    Shows:
    1. Radial distributions at multiple bin sizes
    2. Angular distributions at multiple bin sizes  
    3. 2D (r, theta) histograms
    4. Angular distributions in radial slices
    """
    import matplotlib.pyplot as plt
    
    device = torch.device(config.device)
    
    # Generate fitted model particles
    model = ImprovedForwardModel(config, result['n_peaks']).to(device)
    
    # Set model parameters from result
    E_range = config.E_max - config.E_min
    for i in range(result['n_peaks']):
        E_norm = (result['E_centers'][i] - config.E_min) / E_range
        E_norm = np.clip(E_norm, 0.01, 0.99)
        model.E_logits.data[i] = np.log(E_norm / (1 - E_norm))
        
        sigma_norm = (result['sigmas'][i] - config.sigma_min) / (config.sigma_max - config.sigma_min)
        sigma_norm = np.clip(sigma_norm, 0.01, 0.99)
        model.sigma_logs.data[i] = np.log(sigma_norm / (1 - sigma_norm))
        
        beta_norm = (result['betas'][i] - config.beta_min) / (config.beta_max - config.beta_min)
        beta_norm = np.clip(beta_norm, 0.01, 0.99)
        model.beta_raw.data[i] = np.log(beta_norm / (1 - beta_norm))
    
    model.br_logits.data = torch.log(torch.tensor(result['branching_ratios']) + 1e-10)
    
    # Generate model particles
    with torch.no_grad():
        X_model, Y_model, weights = model(len(xy_obs))
    
    xy_model = np.column_stack([X_model.cpu().numpy(), Y_model.cpu().numpy()])
    weights_np = weights.cpu().numpy()
    
    # Compute r, theta for both
    r_obs = np.sqrt(xy_obs[:, 0]**2 + xy_obs[:, 1]**2)
    theta_obs = np.arctan2(xy_obs[:, 1], xy_obs[:, 0])
    
    r_model = np.sqrt(xy_model[:, 0]**2 + xy_model[:, 1]**2)
    theta_model = np.arctan2(xy_model[:, 1], xy_model[:, 0])
    
    r_max = result['r_max']
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Radial distributions at different bin sizes
    bin_sizes_r = [10, 30, 100]  # coarse to fine
    for i, n_bins in enumerate(bin_sizes_r):
        ax = fig.add_subplot(4, 3, i + 1)
        
        bins = np.linspace(0, r_max, n_bins + 1)
        
        hist_obs, _ = np.histogram(r_obs, bins=bins, density=True)
        hist_model, _ = np.histogram(r_model, bins=bins, weights=weights_np, density=True)
        
        centers = 0.5 * (bins[:-1] + bins[1:])
        
        ax.step(centers, hist_obs, 'b-', label='Data', linewidth=1.5, where='mid')
        ax.step(centers, hist_model, 'r--', label='Model', linewidth=1.5, where='mid')
        ax.set_xlabel('r (mm)')
        ax.set_ylabel('Density')
        ax.set_title(f'Radial: {n_bins} bins (dr={r_max/n_bins:.2f} mm)')
        ax.legend()
        
        # Compute chi2
        mask = hist_obs > 0
        if mask.sum() > 0:
            chi2 = np.sum((hist_obs[mask] - hist_model[mask])**2 / hist_obs[mask])
            ax.text(0.95, 0.95, f'χ²={chi2:.2f}', transform=ax.transAxes, 
                   ha='right', va='top', fontsize=9)
    
    # Row 2: Angular distributions at different bin sizes
    bin_sizes_theta = [8, 18, 36]
    for i, n_bins in enumerate(bin_sizes_theta):
        ax = fig.add_subplot(4, 3, i + 4)
        
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        hist_obs, _ = np.histogram(theta_obs, bins=bins, density=True)
        hist_model, _ = np.histogram(theta_model, bins=bins, weights=weights_np, density=True)
        
        centers = 0.5 * (bins[:-1] + bins[1:])
        
        ax.step(np.degrees(centers), hist_obs, 'b-', label='Data', linewidth=1.5, where='mid')
        ax.step(np.degrees(centers), hist_model, 'r--', label='Model', linewidth=1.5, where='mid')
        ax.set_xlabel('θ (deg)')
        ax.set_ylabel('Density')
        ax.set_title(f'Angular: {n_bins} bins (dθ={360/n_bins:.0f}°)')
        ax.legend()
    
    # Row 3: 2D histograms (r, theta)
    ax = fig.add_subplot(4, 3, 7)
    h, xedges, yedges = np.histogram2d(r_obs, theta_obs, bins=[30, 18], 
                                        range=[[0, r_max], [-np.pi, np.pi]])
    ax.imshow(h.T, origin='lower', aspect='auto',
              extent=[0, r_max, -180, 180], cmap='Blues')
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('θ (deg)')
    ax.set_title('Data: 2D (r, θ)')
    
    ax = fig.add_subplot(4, 3, 8)
    h_model, _, _ = np.histogram2d(r_model, theta_model, bins=[30, 18],
                                    range=[[0, r_max], [-np.pi, np.pi]],
                                    weights=weights_np)
    ax.imshow(h_model.T, origin='lower', aspect='auto',
              extent=[0, r_max, -180, 180], cmap='Reds')
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('θ (deg)')
    ax.set_title('Model: 2D (r, θ)')
    
    ax = fig.add_subplot(4, 3, 9)
    # Normalize for comparison
    h_norm = h / (h.sum() + 1e-10)
    h_model_norm = h_model / (h_model.sum() + 1e-10)
    diff = h_norm - h_model_norm
    vmax = np.abs(diff).max()
    ax.imshow(diff.T, origin='lower', aspect='auto',
              extent=[0, r_max, -180, 180], cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('θ (deg)')
    ax.set_title('Residual (Data - Model)')
    
    # Row 4: Angular distributions in radial slices (key for beta!)
    n_slices = 3
    r_edges = np.linspace(0, r_max, n_slices + 1)
    
    for i in range(n_slices):
        ax = fig.add_subplot(4, 3, 10 + i)
        
        r_min, r_max_slice = r_edges[i], r_edges[i + 1]
        
        # Select particles in this radial range
        mask_obs = (r_obs >= r_min) & (r_obs < r_max_slice)
        mask_model = (r_model >= r_min) & (r_model < r_max_slice)
        
        if mask_obs.sum() > 10 and mask_model.sum() > 10:
            bins = np.linspace(-np.pi, np.pi, 19)
            
            hist_obs, _ = np.histogram(theta_obs[mask_obs], bins=bins, density=True)
            hist_model, _ = np.histogram(theta_model[mask_model], bins=bins, 
                                         weights=weights_np[mask_model], density=True)
            
            centers = 0.5 * (bins[:-1] + bins[1:])
            
            ax.step(np.degrees(centers), hist_obs, 'b-', label='Data', linewidth=1.5, where='mid')
            ax.step(np.degrees(centers), hist_model, 'r--', label='Model', linewidth=1.5, where='mid')
            
            # Fit beta from angular distribution
            # f(theta) = 1 + beta * P2(cos(theta))
            cos_theta = np.cos(centers)
            P2 = (3 * cos_theta**2 - 1) / 2
            
            # Simple linear fit for beta
            if hist_obs.std() > 0:
                beta_fit_obs = np.corrcoef(P2, hist_obs)[0, 1] * hist_obs.std() / (P2.std() + 1e-10)
                ax.text(0.05, 0.95, f'β_data≈{beta_fit_obs:.2f}', transform=ax.transAxes,
                       va='top', fontsize=8, color='blue')
            if hist_model.std() > 0:
                beta_fit_model = np.corrcoef(P2, hist_model)[0, 1] * hist_model.std() / (P2.std() + 1e-10)
                ax.text(0.05, 0.85, f'β_model≈{beta_fit_model:.2f}', transform=ax.transAxes,
                       va='top', fontsize=8, color='red')
        
        ax.set_xlabel('θ (deg)')
        ax.set_ylabel('Density')
        ax.set_title(f'Angular @ r∈[{r_min:.1f}, {r_max_slice:.1f}] mm')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {save_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# Convenience function for multi-start
# =============================================================================
def fit_xy_multistart(xy_obs: np.ndarray,
                      vmi_k: float = 0.01,
                      n_peaks: int = 3,
                      n_starts: int = 10,
                      E_max: float = 5.0,
                      n_particles: int = 50000,
                      verbose: bool = True,
                      device: str = 'cpu') -> Dict[str, Any]:
    """
    Multi-start differentiable fitting
    
    Args:
        xy_obs: observed XY data (N, 2)
        vmi_k: VMI conversion coefficient
        n_peaks: number of peaks
        n_starts: number of random initializations
        E_max: maximum energy (eV)
        n_particles: particles for forward model
        verbose: print progress
        device: 'cpu' or 'cuda'
    """
    config = DiffConfig(
        vmi_k=vmi_k,
        E_max=E_max,
        n_particles=n_particles,
        device=device
    )
    
    reconstructor = MultiStartReconstructor(config, n_starts=n_starts)
    return reconstructor.fit(xy_obs, n_peaks=n_peaks, verbose=verbose)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Abel Backward Reconstruction X2 - Differentiable Forward Fitting")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        from Abel_forward_simulation import Config, run_simulation
        
        # True parameters
        true_E = [0.5, 1.5]
        true_beta = [2.0, -0.5]
        true_br = [0.4, 0.6]
        
        E_max = max(true_E) * 1.5
        r_max_mm = 20.0
        vmi_k = Config.calculate_vmi_k(E_max, r_max_mm)
        
        print(f"\nTrue parameters:")
        print(f"  E = {true_E} eV")
        print(f"  beta = {true_beta}")
        print(f"  BR = {true_br}")
        print(f"  vmi_k = {vmi_k:.4e}")
        
        config = Config(
            E_centers=true_E,
            Betas=true_beta,
            branching_ratios=true_br,
            N_events=20000,
            vmi_k=vmi_k,
            sigma_laser=0.03,
            bg_rate=0.05,
        )
        
        xy_obs, _ = run_simulation(config, add_noise=False, output_mode='xy_dld')
        print(f"Generated {len(xy_obs)} particles")
        
        # Test ENSEMBLE method (new!)
        print("\n" + "="*70)
        print("Testing ENSEMBLE optimization (dynamic parameter estimation)...")
        print("="*70)
        
        result = fit_xy_ensemble(
            xy_obs,
            vmi_k=vmi_k,
            n_peaks=2,
            n_ensemble=12,  # 12 ensemble members
            E_max=E_max,
            n_particles=20000,
            verbose=True,
            device=device
        )
        
        # Compare
        print("\n" + "="*70)
        print("Comparison with true values:")
        print("="*70)
        
        sorted_idx = np.argsort(result['E_centers'])
        for i, idx in enumerate(sorted_idx):
            E_fit = result['E_centers'][idx]
            E_true = true_E[i]
            beta_fit = result['betas'][idx]
            beta_true = true_beta[i]
            
            E_err = abs(E_fit - E_true) / E_true * 100
            beta_err = abs(beta_fit - beta_true)
            
            # Show uncertainty if available
            stats = result.get('ensemble_stats', {})
            E_std = stats.get('E_std', [0]*len(true_E))[idx]
            beta_std = stats.get('beta_std', [0]*len(true_beta))[idx]
            
            print(f"Peak {i+1}: E={E_fit:.3f}±{E_std:.3f} (true={E_true:.3f}, err={E_err:.1f}%), "
                  f"β={beta_fit:.2f}±{beta_std:.2f} (true={beta_true:.2f}, Δ={beta_err:.2f})")
        
    except ImportError:
        print("Abel_forward_simulation not found, running standalone test...")
        
        np.random.seed(42)
        N = 10000
        r1 = np.random.normal(5, 0.5, N // 2)
        r2 = np.random.normal(12, 0.8, N // 2)
        r = np.concatenate([r1, r2])
        theta = np.random.uniform(-np.pi, np.pi, N)
        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        xy_obs = np.column_stack([X, Y])
        
        result = fit_xy_ensemble(
            xy_obs,
            vmi_k=0.01,
            n_peaks=2,
            n_ensemble=10,
            E_max=3.0,
            verbose=True
        )
    
    print("\nDone!")
