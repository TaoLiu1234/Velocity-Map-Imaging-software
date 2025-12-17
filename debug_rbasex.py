"""
Debug script for rBasex reconstruction to identify why it's returning zeros
"""

import numpy as np
import abel
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt


def debug_rbasex(image: np.ndarray) -> None:
    """
    Debug rBasex reconstruction step by step
    """
    print("Debugging rBasex reconstruction...")
    print(f"Input image shape: {image.shape}")
    print(f"Input image range: {np.min(image):.3f} to {np.max(image):.3f}")
    
    # Step 1: Run rBasex transform
    print("\n1. Running rBasex transform...")
    try:
        recon_image, distr = abel.rbasex.rbasex_transform(
            image, 
            direction='inverse', 
            basis_dir=None,
            verbose=True  # Enable verbose to see what's happening
        )
        print(f"Reconstruction successful!")
        print(f"Recon image shape: {recon_image.shape}")
        print(f"Recon image range: {np.min(recon_image):.3f} to {np.max(recon_image):.3f}")
    except Exception as e:
        print(f"Error in rBasex transform: {e}")
        return
    
    # Step 2: Extract radial distribution and beta
    print("\n2. Extracting r, I, beta...")
    try:
        r_rb, I_rb, beta_rb = distr.rIbeta()
        print(f"r range: {np.min(r_rb):.1f} to {np.max(r_rb):.1f}")
        print(f"I range: {np.min(I_rb):.3f} to {np.max(I_rb):.3f}")
        print(f"beta range: {np.min(beta_rb):.3f} to {np.max(beta_rb):.3f}")
        print(f"Number of points: {len(r_rb)}")
    except Exception as e:
        print(f"Error extracting rIbeta: {e}")
        return
    
    # Step 3: Check for valid intensity values
    print("\n3. Checking intensity profile...")
    if np.max(I_rb) <= 0:
        print("ERROR: Intensity profile is all zeros or negative!")
        print("This suggests the rBasex transform failed.")
        return
    
    # Step 4: Mask center region
    print("\n4. Masking center region...")
    mask_radius = 10
    I_rb_masked = I_rb.copy()
    I_rb_masked[:mask_radius] = 0
    print(f"After masking, max intensity: {np.max(I_rb_masked):.3f}")
    
    # Step 5: Normalize for peak finding
    print("\n5. Normalizing for peak finding...")
    if np.max(I_rb_masked) > 0:
        I_rb_norm = I_rb_masked / np.max(I_rb_masked)
        print(f"Normalization successful. Max normalized value: {np.max(I_rb_norm):.3f}")
    else:
        print("ERROR: Cannot normalize - max intensity is 0!")
        return
    
    # Step 6: Try different peak finding parameters
    print("\n6. Testing peak finding with different parameters...")
    
    # Original parameters
    peaks_orig, _ = find_peaks(I_rb_norm, height=0.05, distance=8, prominence=0.03)
    print(f"Original parameters - found {len(peaks_orig)} peaks at indices: {peaks_orig}")
    
    # More lenient parameters
    peaks_lenient, _ = find_peaks(I_rb_norm, height=0.01, distance=5, prominence=0.01)
    print(f"Lenient parameters - found {len(peaks_lenient)} peaks at indices: {peaks_lenient}")
    
    # Very lenient parameters
    peaks_very_lenient, _ = find_peaks(I_rb_norm, height=0.001, distance=3, prominence=0.001)
    print(f"Very lenient parameters - found {len(peaks_very_lenient)} peaks at indices: {peaks_very_lenient}")
    
    # Step 7: Visualize the results
    print("\n7. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    ax1 = axes[0, 0]
    im1 = ax1.imshow(image, cmap='hot')
    ax1.set_title("Original VMI Image")
    plt.colorbar(im1, ax=ax1)
    
    # Reconstructed image
    ax2 = axes[0, 1]
    im2 = ax2.imshow(recon_image, cmap='hot')
    ax2.set_title("rBasex Reconstructed Image")
    plt.colorbar(im2, ax=ax2)
    
    # Intensity profile
    ax3 = axes[1, 0]
    ax3.plot(r_rb, I_rb, 'b-', label='Original', linewidth=2)
    ax3.plot(r_rb, I_rb_masked, 'r--', label='Masked', linewidth=2)
    ax3.plot(r_rb, I_rb_norm, 'g:', label='Normalized', linewidth=2)
    
    # Mark peaks
    if len(peaks_orig) > 0:
        ax3.plot(r_rb[peaks_orig], I_rb_norm[peaks_orig], 'ro', markersize=8, label='Peaks (orig)')
    if len(peaks_lenient) > 0:
        ax3.plot(r_rb[peaks_lenient], I_rb_norm[peaks_lenient], 'bs', markersize=6, label='Peaks (lenient)')
    
    ax3.set_xlabel("Radius (pixels)")
    ax3.set_ylabel("Intensity")
    ax3.set_title("Radial Intensity Profile")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Beta profile
    ax4 = axes[1, 1]
    ax4.plot(r_rb, beta_rb, 'purple', linewidth=2)
    ax4.set_xlabel("Radius (pixels)")
    ax4.set_ylabel("Beta")
    ax4.set_title("Angular Anisotropy (Beta) Profile")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('rbasex_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 8: Extract parameters with the best peak set
    print("\n8. Extracting parameters...")
    
    # Use the most successful peak finding result
    if len(peaks_lenient) > 0:
        peaks_idx = peaks_lenient
        print(f"Using {len(peaks_idx)} peaks found with lenient parameters")
    else:
        print("ERROR: No peaks found even with lenient parameters!")
        return
    
    # Calculate FWHM
    if len(peaks_idx) > 0:
        widths_res = peak_widths(I_rb_norm, peaks_idx, rel_height=0.5)
        fwhms = widths_res[0]
        sigmas = fwhms / 2.355
        print(f"FWHMs: {fwhms}")
        print(f"Sigmas: {sigmas}")
    
    # Extract parameters
    params = []
    for i, p_idx in enumerate(peaks_idx):
        r_val = r_rb[p_idx]
        
        # Get beta at this radius
        beta_window = 3
        start_idx = max(0, p_idx - beta_window)
        end_idx = min(len(beta_rb), p_idx + beta_window + 1)
        beta_val = np.mean(beta_rb[start_idx:end_idx])
        beta_val = np.clip(beta_val, -1.0, 2.0)
        
        # Get amplitude
        amp = I_rb[p_idx]
        
        # Get sigma/FWHM
        sigma = sigmas[i] if i < len(sigmas) else 4.0
        fwhm = fwhms[i] if i < len(fwhms) else sigma * 2.355
        
        params.append({
            'r': r_val,
            'sigma': sigma,
            'fwhm': fwhm,
            'amp': amp,
            'beta': beta_val
        })
        
        print(f"Peak {i+1}: r={r_val:.1f}, sigma={sigma:.2f}, fwhm={fwhm:.2f}, beta={beta_val:.3f}, amp={amp:.3f}")
    
    return params


def create_test_image():
    """Create a simple test VMI image"""
    n = 256
    y, x = np.ogrid[:n, :n]
    center = n // 2
    y, x = y - center, x - center
    r = np.sqrt(x**2 + y**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = x / r
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    P2 = 0.5 * (3 * cos_theta**2 - 1)
    
    # Create test peaks with different parameters
    test_peaks = [
        {'r': 40, 'sigma': 3, 'amp': 1.0, 'beta': 1.5},
        {'r': 80, 'sigma': 5, 'amp': 0.7, 'beta': -0.5},
        {'r': 120, 'sigma': 4, 'amp': 0.5, 'beta': 0.0},
    ]
    
    img_3d = np.zeros_like(r, dtype=float)
    for p in test_peaks:
        radial = p['amp'] * np.exp(-((r - p['r'])**2) / (2 * p['sigma']**2))
        angular = 1 + p['beta'] * P2
        img_3d += radial * angular
    
    # Forward project
    img_proj = abel.Transform(img_3d, method='hansenlaw', direction='forward', verbose=False).transform
    
    # Add some noise
    img_noisy = img_proj + np.random.normal(0, 0.05 * np.max(img_proj), img_proj.shape)
    img_noisy = np.maximum(img_noisy, 0)
    
    return img_noisy, test_peaks


if __name__ == "__main__":
    print("Creating test image...")
    test_image, true_peaks = create_test_image()
    
    print("\nTrue parameters:")
    for i, p in enumerate(true_peaks):
        print(f"  Peak {i+1}: r={p['r']}, sigma={p['sigma']}, beta={p['beta']}, amp={p['amp']}")
    
    print("\n" + "="*60)
    params = debug_rbasex(test_image)
    
    if params:
        print("\n" + "="*60)
        print("rBasex successfully extracted parameters:")
        for i, p in enumerate(params):
            print(f"  Peak {i+1}: r={p['r']:.1f}, sigma={p['sigma']:.2f}, fwhm={p['fwhm']:.2f}, beta={p['beta']:.3f}, amp={p['amp']:.3f}")
    else:
        print("\n" + "="*60)
        print("rBasex failed to extract any parameters!")
