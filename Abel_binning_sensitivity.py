import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
import gc
import os
import pickle
import math
from itertools import islice, product
import abel
from functools import lru_cache

def get_available_memory():
    """Get available memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def estimate_memory_usage(matrix_size, num_rings, num_bin_sizes, dtype=np.float32):
    """Estimate memory usage for processing."""
    bytes_per_element = 4 if dtype == np.float32 else 8
    memory_per_matrix = (matrix_size ** 2) * bytes_per_element / (1024**3)
    total_memory = memory_per_matrix * num_rings * (1 + num_bin_sizes)
    return total_memory, memory_per_matrix


@lru_cache(maxsize=128)
def _get_grid_cache(matrix_size):
    """Cache coordinate grids to avoid recomputation."""
    center = matrix_size // 2
    y, x = np.ogrid[:matrix_size, :matrix_size]
    return np.sqrt((x - center)**2 + (y - center)**2)

def rectangle_binning(matrix, bin_size):
    """Memory-efficient rectangle binning with proper centering."""
    if bin_size <= 1:
        return matrix.copy()
    
    h, w = matrix.shape
    new_h, new_w = h // bin_size, w // bin_size
    
    # Trim to ensure exact divisibility by bin_size
    trimmed = matrix[:new_h*bin_size, :new_w*bin_size]
    binned = trimmed.reshape(new_h, bin_size, new_w, bin_size).sum(axis=(1, 3))
    
    # Ensure the binned matrix has the correct centering
    # For odd original dimensions, the center should be preserved
    if h % 2 == 1 and new_h % 2 == 0:
        # If original was odd but new is even, we need to adjust
        # Add a row/column to maintain centering
        binned = np.pad(binned, ((0, 1), (0, 1)), mode='constant')
    elif h % 2 == 0 and new_h % 2 == 1:
        # If original was even but new is odd, we need to adjust
        # Remove a row/column to maintain centering
        binned = binned[:-1, :-1] if binned.shape[0] > 1 and binned.shape[1] > 1 else binned
        
    return binned


def process_single_ring(args):
    """Process a single ring with all bin sizes - ultra memory efficient with proper centering."""
    r, dr, max_r, matrix_size, bin_sizes, dtype = args
    
    # Generate ring matrix with proper centering
    # Ensure matrix_size is odd for proper centering
    if matrix_size % 2 == 0:
        matrix_size += 1
    
    # Use cached grid for better performance
    R = _get_grid_cache(matrix_size)
    
    inner_radius = r - dr/2
    outer_radius = r + dr/2
    ring_matrix = ((R >= inner_radius) & (R < outer_radius)).astype(dtype)
    
    # Apply binning with centering preservation - parallelize bin sizes
    binned_versions = {}
    if len(bin_sizes) > 1:  # Only use threading if multiple bin sizes
        with ThreadPoolExecutor(max_workers=min(4, len(bin_sizes))) as executor:
            futures = {executor.submit(rectangle_binning, ring_matrix, bin_size): bin_size
                      for bin_size in bin_sizes}
            for future in futures:
                binned_versions[futures[future]] = future.result()
    else:
        # For single bin size, just process directly
        for bin_size in bin_sizes:
            binned_versions[bin_size] = rectangle_binning(ring_matrix, bin_size)
    
    # Store both original and binned matrices
    result = {
        'r': r, 'dr': dr,
        'original_matrix': ring_matrix.copy(),  # Store original ring matrix
        'binned_matrices': binned_versions      # Store all binned matrices
    }
    
    # Clear temporary arrays
    gc.collect()
    
    return result


def save_checkpoint(results, checkpoint_file, batch_idx):
    """Save results to checkpoint file."""
    try:
        with open(checkpoint_file, 'ab') as f:
            pickle.dump((batch_idx, results), f)
        return True
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")
        return False


def load_checkpoints(checkpoint_file):
    """Load results from checkpoint files."""
    results = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                while True:
                    try:
                        batch_idx, batch_results = pickle.load(f)
                        results[batch_idx] = batch_results
                    except EOFError:
                        break
        except Exception as e:
            print(f"Warning: Failed to load checkpoints: {e}")
    return results


def combination_generator(r_values, dr_values, batch_size):
    """Generate combinations in batches to avoid memory issues."""
    r_iter = iter(r_values)
    dr_iter = iter(dr_values)
    
    batch = []
    for r, dr in product(r_values, dr_values):
        batch.append((r, dr))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:  # Yield remaining items
        yield batch


def process_extreme_dataset(r_values, dr_values, bin_sizes, checkpoint_file="checkpoint.pkl", 
                           max_memory_gb=2.0, num_processes=None, dtype=np.float32, resume=False):
    """
    Process extremely large datasets (up to 1 billion combinations) with streaming.
    
    Parameters:
    -----------
    r_values, dr_values : array-like
        Arrays with up to 1000 values each
    bin_sizes : list
        List of bin sizes to apply
    checkpoint_file : str
        File to save/load checkpoints
    max_memory_gb : float
        Maximum memory per batch (default: 2GB)
    num_processes : int, optional
        Number of processes (auto-adaptive)
    dtype : numpy dtype, optional
        Data type for matrices
    resume : bool
        Whether to resume from checkpoint
    """
    # Convert to arrays
    r_values = np.asarray(r_values)
    dr_values = np.asarray(dr_values)
    
    # Calculate matrix size with proper centering
    max_r = np.max(r_values) + np.max(dr_values)/2
    matrix_size = int(2 * max_r) + 3
    
    # Ensure matrix size is odd for proper centering
    if matrix_size % 2 == 0:
        matrix_size += 1
    
    # Calculate total combinations (this could be 1 billion!)
    total_combinations = len(r_values) * len(dr_values)
    
    print(f"EXTREME Dataset Processing:")
    print(f"  r_values: {len(r_values)}")
    print(f"  dr_values: {len(dr_values)}")
    print(f"  bin_sizes: {len(bin_sizes)}")
    print(f"  TOTAL COMBINATIONS: {total_combinations:,}")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    
    # Memory estimation
    _, memory_per_matrix = estimate_memory_usage(matrix_size, 1, len(bin_sizes), dtype)
    available_memory = get_available_memory()
    safe_memory = min(available_memory * 0.6, max_memory_gb)
    
    # Calculate batch size based on memory
    rings_per_batch = max(1, int(safe_memory / (memory_per_matrix * (1 + len(bin_sizes)))))
    
    print(f"  Memory per ring: {memory_per_matrix:.4f} GB")
    print(f"  Safe memory limit: {safe_memory:.1f} GB")
    print(f"  Batch size: {rings_per_batch} rings")
    print(f"  Total batches: {(total_combinations + rings_per_batch - 1) // rings_per_batch:,}")
    
    # Adaptive process count
    if num_processes is None:
        num_processes = min(cpu_count(), rings_per_batch, max(1, int(safe_memory * 2)))
    
    print(f"  Using {num_processes} processes")
    
    # Load existing checkpoints if resuming
    completed_batches = {}
    if resume:
        completed_batches = load_checkpoints(checkpoint_file)
        print(f"  Found {len(completed_batches)} completed batches")
    
    # Process in streaming batches
    all_results = []
    batch_idx = 0
    
    for batch_combinations in combination_generator(r_values, dr_values, rings_per_batch):
        # Skip if already completed
        if batch_idx in completed_batches:
            print(f"Batch {batch_idx + 1}: Skipping (already completed)")
            all_results.extend(completed_batches[batch_idx])
            batch_idx += 1
            continue
        
        print(f"\nBatch {batch_idx + 1}: Processing {len(batch_combinations)} rings...")
        
        # Prepare arguments
        args_list = [(r, dr, max_r, matrix_size, bin_sizes, dtype) for r, dr in batch_combinations]
        
        # Process batch
        start_time = time.time()
        try:
            with Pool(processes=num_processes) as pool:
                batch_results = pool.map(process_single_ring, args_list)
            
            batch_time = time.time() - start_time
            rings_per_sec = len(batch_combinations) / batch_time
            
            print(f"  Completed in {batch_time:.2f}s ({rings_per_sec:.1f} rings/sec)")
            
            # Save checkpoint
            if save_checkpoint(batch_results, checkpoint_file, batch_idx):
                print(f"  Checkpoint saved")
            
            all_results.extend(batch_results)
            
        except Exception as e:
            print(f"  ERROR in batch {batch_idx}: {e}")
            print(f"  Continuing with next batch...")
        
        # Force cleanup
        gc.collect()
        batch_idx += 1
        
        # Progress estimate
        if batch_idx % 10 == 0:
            progress = (batch_idx * rings_per_batch) / total_combinations * 100
            print(f"  Progress: {progress:.1f}% ({batch_idx * rings_per_batch:,}/{total_combinations:,})")
    
    return all_results

def analyze_energy_resolution_sensitivity(results, transform_params=None, save_results=True,
                                         output_file="energy_resolution_results.pkl"):
    """
    Analyze energy resolution sensitivity from processed ring data.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing ring data from process_extreme_dataset
    transform_params : dict, optional
        Parameters for Abel transforms
    save_results : bool
        Whether to save results to file
    output_file : str
        Output file path
        
    Returns:
    --------
    dict
        Dictionary containing energy resolution analysis results
    """
    if transform_params is None:
        transform_params = {
            'basis_dir': None,
            'reg': None,
            'verbose': False
        }
    
    print("\nANALYZING ENERGY RESOLUTION SENSITIVITY")
    print("=" * 50)
    
    start_time = time.time()
    
    # Extract data from results
    original_matrices = []
    r_values = []
    dr_values = []
    bin_sizes = []
    binned_matrices_dict = {}
    
    # First pass: collect all data and organize by bin size
    for i, result in enumerate(results):
        r_values.append(result['r'])
        dr_values.append(result['dr'])
        original_matrices.append(result['original_matrix'])
        
        # Collect binned matrices by bin size
        for bin_size, binned_matrix in result['binned_matrices'].items():
            if bin_size not in binned_matrices_dict:
                binned_matrices_dict[bin_size] = []
            binned_matrices_dict[bin_size].append(binned_matrix)
    
    # Get unique bin sizes
    bin_sizes = sorted(binned_matrices_dict.keys())
    
    print(f"Found {len(original_matrices)} matrices")
    print(f"Bin sizes to analyze: {bin_sizes}")
    
    # Memory check before processing
    available_memory = get_available_memory()
    estimated_memory = len(original_matrices) * 0.1  # Rough estimate in GB
    if estimated_memory > available_memory * 0.7:
        print(f"WARNING: Estimated memory usage ({estimated_memory:.1f} GB) exceeds available memory")
        print("Consider processing in smaller batches")
    
    # Perform batch energy resolution analysis
    resolution_results = Abel_energy_resolution_estimation_batch(
        r_values, dr_values, bin_sizes, original_matrices, binned_matrices_dict,
        transform_params=transform_params, save_intermediate=save_results,
        output_file=output_file if save_results else None
    )
    
    # Calculate statistics for each bin size
    statistics = {}
    for bin_size in bin_sizes:
        diffs = resolution_results['resolutions'][bin_size]
        if diffs:
            stats = {
                'mean': np.mean(diffs),
                'std': np.std(diffs),
                'min': np.min(diffs),
                'max': np.max(diffs),
                'median': np.median(diffs),
                'count': len(diffs)
            }
            statistics[bin_size] = stats
        else:
            statistics[bin_size] = {}
    
    # Create final results dictionary
    final_results = {
        'resolution_data': resolution_results,
        'statistics': statistics,
        'metadata': {
            'total_matrices': len(original_matrices),
            'processing_time': time.time() - start_time,
            'transform_params': transform_params
        }
    }
    
    # Print summary statistics
    print("\nENERGY RESOLUTION SUMMARY:")
    print("-" * 30)
    for bin_size in bin_sizes:
        stats = statistics[bin_size]
        if stats:
            print(f"Bin {bin_size}x{bin_size}:")
            print(f"  Mean ΔFWHM: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Median: {stats['median']:.4f}")
    
    # Save final results
    if save_results:
        with open(output_file, 'wb') as f:
            pickle.dump(final_results, f)
        print(f"\nResults saved to {output_file}")
    
    return final_results

def process_and_analyze_energy_resolution(r_values, dr_values, bin_sizes,
                                        checkpoint_file="energy_checkpoint.pkl",
                                        results_file="energy_resolution_results.pkl",
                                        max_memory_gb=2.0, num_processes=None,
                                        dtype=np.float32, resume=False,
                                        transform_params=None):
    """
    Complete pipeline: process rings and analyze energy resolution sensitivity.
    
    Parameters:
    -----------
    r_values, dr_values : array-like
        Arrays of ring parameters
    bin_sizes : list
        List of bin sizes to analyze
    checkpoint_file : str
        Checkpoint file for ring processing
    results_file : str
        Output file for energy resolution results
    max_memory_gb : float
        Memory limit for processing
    num_processes : int, optional
        Number of processes
    dtype : numpy dtype
        Data type for matrices
    resume : bool
        Whether to resume from checkpoint
    transform_params : dict, optional
        Parameters for Abel transforms
        
    Returns:
    --------
    dict
        Energy resolution analysis results
    """
    print("COMPLETE ENERGY RESOLUTION ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Process rings with different bin sizes
    print("\nSTEP 1: Processing rings with binning...")
    ring_results = process_extreme_dataset(
        r_values, dr_values, bin_sizes,
        checkpoint_file=checkpoint_file,
        max_memory_gb=max_memory_gb,
        num_processes=num_processes,
        dtype=dtype,
        resume=resume
    )
    
    # Step 2: Analyze energy resolution sensitivity
    print("\nSTEP 2: Analyzing energy resolution sensitivity...")
    analysis_results = analyze_energy_resolution_sensitivity(
        ring_results,
        transform_params=transform_params,
        save_results=True,
        output_file=results_file
    )
    
    return analysis_results
def calculate_fwhm(r, I):
    """Calculate FWHM efficiently."""
    if len(I) == 0 or np.max(I) <= 0:
        return 0.0
    
    peak_idx = np.argmax(I)
    peak_intensity = I[peak_idx]
    half_max = peak_intensity / 2
    indices_above_half = np.where(I >= half_max)[0]
    
    if len(indices_above_half) < 2:
        return 0.0
    
    return np.abs(r[indices_above_half[-1]] - r[indices_above_half[0]])

def _transform_single_matrix(matrix, transform_params):
    """Helper function to transform a single matrix."""
    try:
        recon, distr = abel.rbasex.rbasex_transform(matrix, direction='backward', **transform_params)
        r, I, beta = distr.rIbeta()
        return (r.astype(np.float32), I.astype(np.float32), beta.astype(np.float32))
    except Exception:
        return (np.array([]), np.array([]), np.array([]))

def abel_transform_batch(matrices, direction='backward', transform_params=None):
    """
    Perform Abel transforms on a batch of matrices efficiently.
    
    Parameters:
    -----------
    matrices : list or array
        List of matrices to transform
    direction : str
        Transform direction ('forward' or 'backward')
    transform_params : dict, optional
        Parameters for the Abel transform (basis functions, etc.)
        
    Returns:
    --------
    list of tuples
        Each tuple contains (r, I, beta) for each matrix
    """
    if transform_params is None:
        transform_params = {
            'basis_dir': None,  # Use default
            'reg': None,        # No regularization by default
            'verbose': False    # Suppress output
        }
    
    results = []
    
    # Process matrices in smaller sub-batches to manage memory
    sub_batch_size = max(1, len(matrices) // 4)  # Process in 4 sub-batches
    
    for i in range(0, len(matrices), sub_batch_size):
        sub_batch = matrices[i:i+sub_batch_size]
        
        # Use threading for sub-batch processing
        if len(sub_batch) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(sub_batch))) as executor:
                futures = [executor.submit(_transform_single_matrix, matrix, transform_params) for matrix in sub_batch]
                for future in futures:
                    results.append(future.result())
        else:
            # For single matrix, process directly
            results.append(_transform_single_matrix(sub_batch[0], transform_params))
        
        # Force garbage collection after each sub-batch
        gc.collect()
    
    return results

def Abel_energy_resolution_estimation_batch(r_values, dr_values, bin_sizes, original_matrices, binned_matrices_dict,
                                           transform_params=None, save_intermediate=False, output_file=None):
    """
    Optimized batch processing for energy resolution estimation on large datasets.
    
    Parameters:
    -----------
    r_values : array-like
        Array of radius values
    dr_values : array-like
        Array of dr values
    bin_sizes : list
        List of bin sizes applied
    original_matrices : list
        List of original matrices
    binned_matrices_dict : dict
        Dictionary with bin_size as key and list of binned matrices as values
    transform_params : dict, optional
        Parameters for Abel transforms
    save_intermediate : bool
        Whether to save intermediate results
    output_file : str, optional
        File to save results
        
    Returns:
    --------
    dict
        Dictionary containing resolution results for all combinations
    """
    if transform_params is None:
        transform_params = {
            'basis_dir': None,
            'reg': None,
            'verbose': False
        }
    
    print(f"Processing {len(original_matrices)} matrices for energy resolution estimation...")
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        'r_values': r_values,
        'dr_values': dr_values,
        'bin_sizes': bin_sizes,
        'resolutions': {bin_size: [] for bin_size in bin_sizes},
        'metadata': {
            'total_matrices': len(original_matrices),
            'transform_params': transform_params,
            'processing_time': None
        }
    }
    
    # Process original matrices in batch
    print("Performing Abel transforms on original matrices...")
    original_transforms = abel_transform_batch(original_matrices, transform_params=transform_params)
    
    # Calculate FWHM for original matrices
    original_fwhms = []
    for r, I, _ in original_transforms:
        fwhm = calculate_fwhm(r, I)
        original_fwhms.append(fwhm)
    
    # Process each bin size
    for bin_size in bin_sizes:
        print(f"Processing bin size {bin_size}x{bin_size}...")
        binned_matrices = binned_matrices_dict[bin_size]
        
        # Perform Abel transforms on binned matrices
        binned_transforms = abel_transform_batch(binned_matrices, transform_params=transform_params)
        
        # Calculate FWHM for binned matrices
        binned_fwhms = []
        for r, I, _ in binned_transforms:
            fwhm = calculate_fwhm(r, I)
            binned_fwhms.append(fwhm)
        
        # Calculate resolution differences
        resolution_diffs = []
        for orig_fwhm, binned_fwhm in zip(original_fwhms, binned_fwhms):
            delta_fwhm = orig_fwhm - binned_fwhm
            resolution_diffs.append(delta_fwhm)
        
        results['resolutions'][bin_size] = resolution_diffs
        
        # Save intermediate results if requested
        if save_intermediate and output_file:
            temp_file = f"{output_file}_bin{bin_size}.pkl"
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'bin_size': bin_size,
                    'original_fwhms': original_fwhms,
                    'binned_fwhms': binned_fwhms,
                    'resolution_diffs': resolution_diffs
                }, f)
        
        # Clean up to free memory
        del binned_transforms, binned_fwhms, resolution_diffs
        gc.collect()
    
    # Record processing time
    results['metadata']['processing_time'] = time.time() - start_time
    print(f"Energy resolution estimation completed in {results['metadata']['processing_time']:.2f} seconds")
    
    # Save final results
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
    
    return results

def Abel_energy_resolution_estimation(r, dr, bin_size, original_matrix, binned_matrix):
    """
    Estimate energy resolution based on ring parameters and binned data.
    Performs Abel inversion on both original and binned matrices and stores all relevant variables.
    
    Parameters:
    -----------
    r : float
        Radius value
    dr : float
        dr value
    bin_size : int
        Bin size applied
    original_matrix : ndarray
        Original matrix
    binned_matrix : ndarray
        Binned matrix
        
    Returns:
    --------
    dict
        Dictionary containing all relevant variables:
        - r_original, r_binned: radius arrays from Abel transforms
        - I_original, I_binned: intensity arrays from Abel transforms
        - recon_original, recon_binned: reconstructed matrices from Abel transforms
        - fwhm_original, fwhm_binned: FWHM values
        - delta_fwhm: difference in FWHM
    """
    # Calculate the effective radius after binning
    effective_r = r / bin_size
    effective_dr = dr / bin_size
    
    # Perform Abel transform on original matrix
    recon_original, distr = abel.rbasex.rbasex_transform(original_matrix, direction='backward')
    r_original, I_original, beta_original = distr.rIbeta()

    # Perform Abel transform on binned matrix
    recon_binned, distr = abel.rbasex.rbasex_transform(binned_matrix, direction='backward')
    r_binned, I_binned, beta_binned = distr.rIbeta()
    
    # Estimate resolution (simple model) - calculate FWHM for original
    peak_idx_original = np.argmax(I_original)
    peak_intensity_original = I_original[peak_idx_original]

    half_max_original = peak_intensity_original / 2
    indices_above_half_original = np.where(I_original >= half_max_original)[0]
    fwhm_r_original = np.abs(r_original[indices_above_half_original[-1]] - r_original[indices_above_half_original[0]])

    # Calculate FWHM for binned
    peak_idx_binned = np.argmax(I_binned)
    peak_intensity_binned = I_binned[peak_idx_binned]
    # Full Width at Half Maximum (FWHM) estimation
    half_max_binned = peak_intensity_binned / 2
    indices_above_half_binned = np.where(I_binned >= half_max_binned)[0]
    fwhm_r_binned = np.abs(r_binned[indices_above_half_binned[-1]] - r_binned[indices_above_half_binned[0]])
    
    # Calculate delta FWHM
    delta_fwhm = fwhm_r_original - fwhm_r_binned
    
    # Store all relevant variables in a dictionary
    result = {
        'r': r,
        'dr': dr,
        'bin_size': bin_size,
        'effective_r': effective_r,
        'effective_dr': effective_dr,
        'r_original': r_original,
        'r_binned': r_binned,
        'I_original': I_original,
        'I_binned': I_binned,
        'beta_original': beta_original,
        'beta_binned': beta_binned,
        'recon_original': recon_original,
        'recon_binned': recon_binned,
        'fwhm_original': fwhm_r_original,
        'fwhm_binned': fwhm_r_binned,
        'delta_fwhm': delta_fwhm
    }
    
    return result

def abel_energy_resolution_estimation_batch(r_values, dr_values, bin_sizes, original_matrices, binned_matrices_dict,
                                           save_results=True, output_file="abel_results.pkl", batch_size=10):
    """
    Optimized batch processing for energy resolution estimation on large datasets.
    Processes matrices in batches to manage memory efficiently.
    
    Parameters:
    -----------
    r_values : array-like
        Array of radius values
    dr_values : array-like
        Array of dr values
    bin_sizes : list
        List of bin sizes applied
    original_matrices : list
        List of original matrices
    binned_matrices_dict : dict
        Dictionary with bin_size as key and list of binned matrices as values
    save_results : bool
        Whether to save results to file
    output_file : str
        File to save results
    batch_size : int
        Number of matrices to process in each batch
        
    Returns:
    --------
    dict
        Dictionary containing all results for all combinations
    """
    print(f"Processing {len(original_matrices)} matrices for energy resolution estimation...")
    start_time = time.time()
    
    # Initialize results dictionary
    all_results = {
        'metadata': {
            'total_matrices': len(original_matrices),
            'bin_sizes': bin_sizes,
            'processing_time': None
        },
        'results': []
    }
    
    # Process matrices in batches to manage memory
    total_combinations = len(original_matrices) * len(bin_sizes)
    processed_count = 0
    
    for i in range(0, len(original_matrices), batch_size):
        batch_end = min(i + batch_size, len(original_matrices))
        batch_original = original_matrices[i:batch_end]
        batch_r = r_values[i:batch_end]
        batch_dr = dr_values[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}: matrices {i+1}-{batch_end}...")
        
        # Process each matrix in the batch
        for j, (r, dr, original_matrix) in enumerate(zip(batch_r, batch_dr, batch_original)):
            for bin_size in bin_sizes:
                binned_matrix = binned_matrices_dict[bin_size][i + j]
                
                # Perform energy resolution estimation
                result = Abel_energy_resolution_estimation(r, dr, bin_size, original_matrix, binned_matrix)
                all_results['results'].append(result)
                
                processed_count += 1
                
                # Progress update
                if processed_count % 50 == 0:
                    progress = processed_count / total_combinations * 100
                    print(f"  Progress: {progress:.1f}% ({processed_count}/{total_combinations})")
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Record processing time
    all_results['metadata']['processing_time'] = time.time() - start_time
    print(f"Energy resolution estimation completed in {all_results['metadata']['processing_time']:.2f} seconds")
    
    # Save results if requested
    if save_results:
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Results saved to {output_file}")
    
    return all_results

def process_rings_and_estimate_energy_resolution(r_values, dr_values, bin_sizes,
                                               checkpoint_file="rings_checkpoint.pkl",
                                               results_file="abel_results.pkl",
                                               max_memory_gb=2.0, num_processes=None,
                                               dtype=np.float32, resume=False,
                                               batch_size=10):
    """
    Complete pipeline: process rings and perform energy resolution estimation.
    
    Parameters:
    -----------
    r_values, dr_values : array-like
        Arrays of ring parameters
    bin_sizes : list
        List of bin sizes to analyze
    checkpoint_file : str
        Checkpoint file for ring processing
    results_file : str
        Output file for Abel results
    max_memory_gb : float
        Memory limit for processing
    num_processes : int, optional
        Number of processes
    dtype : numpy dtype
        Data type for matrices
    resume : bool
        Whether to resume from checkpoint
    batch_size : int
        Batch size for energy resolution processing
        
    Returns:
    --------
    dict
        Energy resolution estimation results
    """
    print("COMPLETE ENERGY RESOLUTION ESTIMATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Process rings with different bin sizes
    print("\nSTEP 1: Processing rings with binning...")
    ring_results = process_extreme_dataset(
        r_values, dr_values, bin_sizes,
        checkpoint_file=checkpoint_file,
        max_memory_gb=max_memory_gb,
        num_processes=num_processes,
        dtype=dtype,
        resume=resume
    )
    
    # Step 2: Extract data and organize for energy resolution estimation
    print("\nSTEP 2: Preparing data for energy resolution estimation...")
    original_matrices = []
    r_values_list = []
    dr_values_list = []
    binned_matrices_dict = {}
    
    # Extract data from ring results
    for result in ring_results:
        r_values_list.append(result['r'])
        dr_values_list.append(result['dr'])
        original_matrices.append(result['original_matrix'])
        
        # Collect binned matrices by bin size
        for bin_size, binned_matrix in result['binned_matrices'].items():
            if bin_size not in binned_matrices_dict:
                binned_matrices_dict[bin_size] = []
            binned_matrices_dict[bin_size].append(binned_matrix)
    
    # Step 3: Perform energy resolution estimation
    print("\nSTEP 3: Performing energy resolution estimation...")
    abel_results = abel_energy_resolution_estimation_batch(
        r_values_list, dr_values_list, bin_sizes,
        original_matrices, binned_matrices_dict,
        save_results=True, output_file=results_file,
        batch_size=batch_size
    )
    
    return abel_results

def main():
    """Main function for extreme dataset processing with energy resolution analysis."""
    print("ULTRA-EFFICIENT EXTREME DATASET PROCESSING WITH ENERGY RESOLUTION ANALYSIS")
    print("=" * 80)
    
    # Test parameters - simulate extreme case
    # For real use, these could be 1000 each = 1 billion combinations
    r_values = np.linspace(300, 500, 5)   # 5 r values
    dr_values = np.linspace(1, 10, 5)      # 5 dr values
    bin_sizes = [1, 2, 4, 8, 16, 32]       # 6 bin sizes
    
    total_combinations = len(r_values) * len(dr_values)
    print(f"Test Dataset: {total_combinations:,} ring combinations")
    print(f"Real scenario could be 1000³ = 1,000,000,000 combinations")
    
    # Option 1: Just process rings (original functionality)
    print("\n" + "="*60)
    print("OPTION 1: PROCESS RINGS ONLY")
    print("="*60)
    
    ring_results = process_extreme_dataset(
        r_values, dr_values, bin_sizes,
        checkpoint_file="extreme_checkpoint.pkl",
        max_memory_gb=2.0,  # Very memory efficient
        num_processes=None,  # Auto-adaptive
        dtype=np.float32,
        resume=False
    )
    
    print(f"\nRING PROCESSING RESULTS:")
    print(f"  Total rings processed: {len(ring_results):,}")
    print(f"  Bin sizes applied: {bin_sizes}")
    
    # Show sample
    if ring_results:
        sample = ring_results[0]
        print(f"  Original matrix shape: {sample['original_matrix'].shape}")
        print(f"  Binned matrices:")
        for bin_size in bin_sizes[:3]:  # Show first 3
            shape = sample['binned_matrices'][bin_size].shape
            print(f"    Bin {bin_size}x{bin_size}: {shape}")
    
    # Option 2: Energy resolution estimation (new functionality)
    print("\n" + "="*60)
    print("OPTION 2: ENERGY RESOLUTION ESTIMATION")
    print("="*60)
    
    # Process rings and estimate energy resolution
    abel_results = process_rings_and_estimate_energy_resolution(
        r_values, dr_values, bin_sizes,
        checkpoint_file="rings_checkpoint.pkl",
        results_file="abel_results.pkl",
        max_memory_gb=2.0,
        num_processes=None,
        dtype=np.float32,
        resume=False,
        batch_size=5  # Small batch size for demo
    )
    
    # Print summary of energy resolution estimation
    print("\nENERGY RESOLUTION ESTIMATION SUMMARY:")
    print("-" * 40)
    print(f"Total results: {len(abel_results['results'])}")
    print(f"Processing time: {abel_results['metadata']['processing_time']:.2f} seconds")
    
    # Show sample result
    if abel_results['results']:
        sample = abel_results['results'][0]
        print(f"\nSample result:")
        print(f"  r: {sample['r']}, dr: {sample['dr']}, bin_size: {sample['bin_size']}")
        print(f"  FWHM original: {sample['fwhm_original']:.4f}")
        print(f"  FWHM binned: {sample['fwhm_binned']:.4f}")
        print(f"  Delta FWHM: {sample['delta_fwhm']:.4f}")
    
    return ring_results, abel_results


if __name__ == "__main__":
    ring_results,abel_results = main()
    print("\nEXTREME PROCESSING COMPLETE.")