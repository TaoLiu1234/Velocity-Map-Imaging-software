import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import abel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.inspection import permutation_importance
import pandas as pd
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import os
import shap
# Set matplotlib to non-interactive backend for faster performance
import matplotlib
matplotlib.use('Agg')

def calculate_num_points_from_ratio(sampling_ratio, inner_radius, ring_width):
    """
    Calculate the number of points needed to achieve a specific sampling ratio.
    
    Parameters:
    -----------
    sampling_ratio : float
        Desired sampling ratio (0.0 to 1.0)
    inner_radius : float
        Inner radius of the ring in pixels
    ring_width : float
        Width of the ring in pixels
        
    Returns:
    --------
    int
        Number of points needed
    """
    
    # Calculate the area of the ring
    outer_radius = inner_radius + ring_width
    ring_area = np.pi * (outer_radius**2 - inner_radius**2)
    
    # Calculate the number of points needed
    num_points = int(sampling_ratio * ring_area)
    
    return num_points

def generate_ring_points_polar(sampling_ratio, inner_radius, outer_radius=None, width=None):
    """
    Generate points with even distribution over a ring area using polar coordinates.
    This method ensures more uniform distribution.
    
    Parameters:
    -----------
    sampling_ratio : float
        Sampling ratio (0.0 to 1.0) that determines the density of points
    inner_radius : float
        Inner radius of the ring in pixels
    outer_radius : float, optional
        Outer radius of the ring in pixels. If not provided, calculated from width
    width : float, optional
        Width of the ring in pixels. Used only if outer_radius is not provided
        
    Returns:
    --------
    tuple : (x_coords, y_coords, actual_num_points)
        Arrays of x and y coordinates of the generated points and actual number of points
    """
    
    # Calculate outer radius if width is provided
    if outer_radius is None:
        if width is None:
            raise ValueError("Either outer_radius or width must be provided")
        outer_radius = inner_radius + width
    
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be less than outer_radius")
    
    # Calculate the number of points needed for the desired sampling ratio
    num_points = calculate_num_points_from_ratio(sampling_ratio, inner_radius, outer_radius - inner_radius)
    
    # Generate uniform distribution in polar coordinates
    # For uniform distribution in a ring, we need to account for the area element
    # r should be distributed as sqrt(uniform * (r_outer^2 - r_inner^2) + r_inner^2)
    r_squared = np.random.uniform(0, 1, num_points) * (outer_radius**2 - inner_radius**2) + inner_radius**2
    r = np.sqrt(r_squared)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    
    # Convert to Cartesian coordinates
    x_coords = r * np.cos(theta)
    y_coords = r * np.sin(theta)
    
    return x_coords, y_coords, num_points

def generate_ring_matrix(sampling_ratio, inner_radius, outer_radius=None, width=None, matrix_size=None):
    """
    Generate points with even distribution over a ring area and store in a 2D numpy matrix.
    
    Parameters:
    -----------
    sampling_ratio : float
        Sampling ratio (0.0 to 1.0) that determines the density of points
    inner_radius : float
        Inner radius of the ring in pixels
    outer_radius : float, optional
        Outer radius of the ring in pixels. If not provided, calculated from width
    width : float, optional
        Width of the ring in pixels. Used only if outer_radius is not provided
    matrix_size : int, optional
        Size of the square matrix. If not provided, calculated from outer_radius
        
    Returns:
    --------
    tuple : (matrix, actual_num_points)
        2D numpy matrix with points marked as 1 and actual number of points generated
    """
    
    # Calculate outer radius if width is provided
    if outer_radius is None:
        if width is None:
            raise ValueError("Either outer_radius or width must be provided")
        outer_radius = inner_radius + width
    
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be less than outer_radius")
    
    # Generate points using polar coordinate method
    x_coords, y_coords, actual_num_points = generate_ring_points_polar(sampling_ratio, inner_radius, outer_radius)
    
    # Calculate matrix size if not provided
    if matrix_size is None:
        matrix_size = int(2 * outer_radius * 1.2)  # Add 20% padding
        # Ensure odd size for proper centering
        if matrix_size % 2 == 0:
            matrix_size += 1
    
    # Create empty matrix
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Calculate center of matrix
    center = matrix_size // 2
    
    # Convert coordinates to matrix indices
    # Round to nearest integer and shift to center
    x_indices = np.round(x_coords + center).astype(int)
    y_indices = np.round(y_coords + center).astype(int)
    
    # Filter out points that fall outside the matrix
    valid_mask = (x_indices >= 0) & (x_indices < matrix_size) & \
                 (y_indices >= 0) & (y_indices < matrix_size)
    
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    
    # Mark points in matrix
    matrix[y_indices, x_indices] = 1
    
    return matrix, actual_num_points

def generate_ring_matrices_array(sampling_ratio_list, inner_radius_list, ring_width_list, matrix_size=None):
    """
    Generate multiple ring matrices with different parameters and store them in a structured numpy array.
    
    Parameters:
    -----------
    sampling_ratio_list : list or array
        List of sampling ratio values (0.0 to 1.0)
    inner_radius_list : list or array
        List of inner_radius values
    ring_width_list : list or array
        List of ring_width values
    matrix_size : int, optional
        Size of the square matrices. If not provided, calculated from the largest outer_radius
        
    Returns:
    --------
    tuple : (matrices_array, params_dict, actual_points_array)
        4D numpy array with shape (len(sampling_ratio_list), len(inner_radius_list), len(ring_width_list), matrix_size, matrix_size)
        Dictionary containing the parameter values for indexing
        3D numpy array with actual number of points generated for each matrix
    """
    
    # Convert inputs to numpy arrays for consistent indexing
    sampling_ratio_arr = np.array(sampling_ratio_list)
    inner_radius_arr = np.array(inner_radius_list)
    ring_width_arr = np.array(ring_width_list)
    
    # Calculate the maximum matrix size needed if not provided
    if matrix_size is None:
        max_outer_radius = np.max(inner_radius_arr + np.max(ring_width_arr))
        matrix_size = int(2 * max_outer_radius * 1.2)  # Add 20% padding
        # Ensure odd size for proper centering
        if matrix_size % 2 == 0:
            matrix_size += 1
    
    # Initialize the 4D array to store all matrices
    matrices_array = np.zeros((
        len(sampling_ratio_arr), 
        len(inner_radius_arr), 
        len(ring_width_arr), 
        matrix_size, 
        matrix_size
    ), dtype=np.uint8)  # Use uint8 to save memory
    
    # Initialize array to store actual number of points
    actual_points_array = np.zeros((
        len(sampling_ratio_arr), 
        len(inner_radius_arr), 
        len(ring_width_arr)
    ), dtype=int)
    
    # Generate all combinations of matrices
    for i, sampling_ratio in enumerate(sampling_ratio_arr):
        for j, inner_radius in enumerate(inner_radius_arr):
            for k, ring_width in enumerate(ring_width_arr):
                matrix, actual_num_points = generate_ring_matrix(
                    sampling_ratio, inner_radius, width=ring_width, matrix_size=matrix_size
                )
                matrices_array[i, j, k] = matrix
                actual_points_array[i, j, k] = actual_num_points
    
    # Create parameter dictionary for easy indexing
    params_dict = {
        'sampling_ratio': sampling_ratio_arr,
        'inner_radius': inner_radius_arr,
        'ring_width': ring_width_arr
    }
    
    return matrices_array, params_dict, actual_points_array

def calculate_energy_resolution(matrix):
    """
    Calculate energy resolution using Abel inversion with robust error handling.
    
    Parameters:
    -----------
    matrix : 2D numpy array
        Matrix containing the ring points
        
    Returns:
    --------
    tuple : (energy_resolution, peak_r, fwhm_r)
        Energy resolution as percentage, peak radius, and FWHM in radius units
    """
    try:
        # Check if matrix has valid data
        if np.sum(matrix) == 0:
            print("Warning: Empty matrix detected, returning invalid energy resolution")
            return 0.0, 0.0, 0.0
        
        
        # Perform Abel inversion with error handling
        try:
            recon, distr = abel.rbasex.rbasex_transform(matrix, direction='backward')
            r, I, beta = distr.rIbeta()
        except Exception as abel_error:
            print(f"Warning: Abel inversion failed: {abel_error}")
            return 0.0, 0.0, 0.0
        
        # Validate the results
        if len(r) == 0 or len(I) == 0:
            print("Warning: Abel inversion returned empty results")
            return 0.0, 0.0, 0.0
        
        # Check for valid intensity distribution
        if np.max(I) <= 0:
            print("Warning: Invalid intensity distribution in Abel results")
            return 0.0, 0.0, 0.0
        
        # Find peak with multiple validation methods
        peak_idx = np.argmax(I)
        peak_r = r[peak_idx]
        peak_intensity = I[peak_idx]
        
        # Validate peak position
        if peak_r <= 0:
            print("Warning: Invalid peak radius detected (peak at or near origin)")
            return 0.0, 0.0, 0.0
        
        # Check if peak is reasonable (not at edge)
        if peak_idx == 0 or peak_idx == len(r) - 1:
            print("Warning: Peak at matrix edge, results may be unreliable")
        
        # Calculate FWHM with better error handling
        half_max = peak_intensity / 2
        if half_max < 0:
            print("Warning: Invalid intensity distribution")
            return 0.0, 0.0, 0.0
        
        indices_above_half = np.where(I >= half_max)[0]
        
        fwhm_r = np.abs(r[indices_above_half[-1]] - r[indices_above_half[0]])
        
        # Validate FWHM
        if fwhm_r < 0:
            print("Warning: Invalid FWHM calculated")
            return 0.0, 0.0, 0.0
        
        
        # Calculate energy resolution as percentage
        energy_resolution = (fwhm_r / peak_r) * 100
        
        # Validate final energy resolution - mark invalid data with negative values
        if energy_resolution < 0:
            print(f"Warning: Negative energy resolution calculated: {energy_resolution:.4f}% - marking as invalid")
            return -1.0, peak_r, fwhm_r  # Negative value marks invalid data
        
        if energy_resolution > 100:
            print(f"Warning: Excessive energy resolution calculated: {energy_resolution:.4f}% - marking as invalid")
            return -1.0, peak_r, fwhm_r  # Negative value marks invalid data
        
        # Check if resolution is physically reasonable (0.01% to 100%)
        if energy_resolution < 0:  # Less than 0.01% is unrealistic
            print(f"Warning: Unrealistically low energy resolution: {energy_resolution:.4f}% - marking as invalid")
            return -1.0, peak_r, fwhm_r  # Negative value marks invalid data
        
        return energy_resolution, peak_r, fwhm_r
        
    except Exception as e:
        print(f"Error in calculate_energy_resolution: {e}")
        return -1.0, 0.0, 0.0  # Negative value marks invalid data

def calculate_energy_resolution_single(args):
    """
    Helper function for parallel processing - calculate energy resolution for a single matrix.
    
    Parameters:
    -----------
    args : tuple
        (matrix, i, j, k) indices and matrix
        
    Returns:
    --------
    tuple : (i, j, k, energy_resolution)
    """
    matrix, i, j, k = args
    try:
        energy_res = calculate_energy_resolution(matrix)[0]
        return (i, j, k, energy_res)
    except Exception as e:
        print(f"Error calculating energy resolution for matrix ({i},{j},{k}): {e}")
        return (i, j, k, 0.0)

def calculate_all_energy_resolutions_parallel(matrices_array, n_workers=None):
    """
    Calculate energy resolution for all matrices in the array using parallel processing.
    
    Parameters:
    -----------
    matrices_array : 4D numpy array
        Array containing all ring matrices
    n_workers : int, optional
        Number of worker processes. If None, uses all available cores
        
    Returns:
    --------
    3D numpy array
        Array containing energy resolution values for each matrix
    """
    # Get the dimensions of the input array
    n_sampling, n_inner, n_width = matrices_array.shape[:3]
    
    # Initialize array to store energy resolutions
    energy_resolutions = np.zeros((n_sampling, n_inner, n_width))
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Limit to 8 to avoid memory issues
    
    print(f"Using {n_workers} parallel workers for energy resolution calculation...")
    
    # Prepare arguments for parallel processing
    args_list = []
    for i in range(n_sampling):
        for j in range(n_inner):
            for k in range(n_width):
                args_list.append((matrices_array[i, j, k], i, j, k))
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(calculate_energy_resolution_single, args_list))
    
    # Fill energy resolutions array
    for i, j, k, energy_res in results:
        energy_resolutions[i, j, k] = energy_res
    
    return energy_resolutions

def calculate_all_energy_resolutions(matrices_array, use_parallel=True, n_workers=None):
    """
    Calculate energy resolution for all matrices in the array.
    
    Parameters:
    -----------
    matrices_array : 4D numpy array
        Array containing all ring matrices
    use_parallel : bool
        Whether to use parallel processing
    n_workers : int, optional
        Number of worker processes for parallel processing
        
    Returns:
    --------
    3D numpy array
        Array containing energy resolution values for each matrix
    """
    if use_parallel and matrices_array.shape[:3][0] * matrices_array.shape[:3][1] * matrices_array.shape[:3][2] > 50:
        return calculate_all_energy_resolutions_parallel(matrices_array, n_workers)
    else:
        # Use original sequential method for small datasets
        n_sampling, n_inner, n_width = matrices_array.shape[:3]
        energy_resolutions = np.zeros((n_sampling, n_inner, n_width))
        
        for i in range(n_sampling):
            for j in range(n_inner):
                for k in range(n_width):
                    energy_resolutions[i, j, k] = calculate_energy_resolution(matrices_array[i, j, k])[0]
        
        return energy_resolutions

def prepare_regression_data(params_dict, energy_resolutions):
    """
    Prepare data for regression analysis by flattening parameter combinations and energy resolutions.
    Filters out invalid energy resolution values for better data quality.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary containing parameter arrays
    energy_resolutions : 3D numpy array
        Array containing energy resolution values
        
    Returns:
    --------
    tuple : (X, y)
        X: 2D numpy array with shape (n_samples, 3) containing parameter values
        y: 1D numpy array containing energy resolution values
    """
    # Create all parameter combinations and flatten energy resolution array
    all_combinations = []
    all_energy_res = []
    
    for i, sampling_ratio in enumerate(params_dict['sampling_ratio']):
        for j, inner_radius in enumerate(params_dict['inner_radius']):
            for k, ring_width in enumerate(params_dict['ring_width']):
                energy_res = energy_resolutions[i, j, k]
                # Filter out invalid energy resolution values (negative values are marked as invalid)
                if energy_res >= 0:  # Only keep non-negative values (valid data)
                    all_combinations.append([sampling_ratio, inner_radius, ring_width])
                    all_energy_res.append(energy_res)
    
    X = np.array(all_combinations)
    y = np.array(all_energy_res)
    
    # Print data quality statistics
    total_samples = len(params_dict['sampling_ratio']) * len(params_dict['inner_radius']) * len(params_dict['ring_width'])
    valid_samples = len(all_energy_res)
    filtered_samples = total_samples - valid_samples
    
    print(f"\nData Quality Control:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid samples: {valid_samples}")
    print(f"  Filtered samples: {filtered_samples} ({filtered_samples/total_samples*100:.1f}%)")
    
    if valid_samples == 0:
        print("Warning: No valid samples remaining after filtering!")
        print("Using fallback strategy - accepting all non-zero values...")
        # Fallback: accept any non-zero values
        for i, sampling_ratio in enumerate(params_dict['sampling_ratio']):
            for j, inner_radius in enumerate(params_dict['inner_radius']):
                for k, ring_width in enumerate(params_dict['ring_width']):
                    energy_res = energy_resolutions[i, j, k]
                    if energy_res > 0:  # Accept any positive value
                        all_combinations.append([sampling_ratio, inner_radius, ring_width])
                        all_energy_res.append(energy_res)
        
        X = np.array(all_combinations)
        y = np.array(all_energy_res)
        valid_samples = len(all_energy_res)
        
        if valid_samples == 0:
            print("Error: No positive energy resolution values found!")
            return np.array([]).reshape(0, 3), np.array([])
    
    print(f"  Energy resolution range: {np.min(y):.3f}% - {np.max(y):.3f}%")
    print(f"  Mean energy resolution: {np.mean(y):.3f}%")
    
    return X, y

def fit_linear_regression(X, y):
    """
    Fit a linear regression model to data.
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix with shape (n_samples, n_features)
    y : 1D numpy array
        Target values
        
    Returns:
    --------
    tuple : (model, metrics)
        model: Trained LinearRegression model
        metrics: Dictionary containing model performance metrics
    """
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred),
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }
    
    return model, metrics

def fit_polynomial_regression(X, y, degree=2):
    """
    Fit a polynomial regression model to the data with cross-term analysis.
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix with shape (n_samples, n_features)
    y : 1D numpy array
        Target values
    degree : int
        Degree of the polynomial features
        
    Returns:
    --------
    tuple : (model, poly_features, metrics, cross_terms)
        model: Trained LinearRegression model on polynomial features
        poly_features: PolynomialFeatures transformer
        metrics: Dictionary containing model performance metrics
        cross_terms: Dictionary containing cross-term coefficients
    """
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Make predictions
    y_pred = model.predict(X_poly)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred),
        'degree': degree,
        'n_features': X_poly.shape[1]
    }
    
    # Analyze cross-term coefficients
    cross_terms = {}
    feature_names = ['sampling_ratio', 'inner_radius', 'ring_width']
    
    # Get feature names from polynomial features
    poly_feature_names = poly_features.get_feature_names_out(feature_names)
    
    # Map coefficients to feature names
    for i, feature_name in enumerate(poly_feature_names):
        if i < len(model.coef_):
            cross_terms[feature_name] = model.coef_[i]
    
    metrics['cross_terms'] = cross_terms
    
    return model, poly_features, metrics, cross_terms

def fit_regularized_regression(X, y, alpha=1.0, regression_type='ridge'):
    """
    Fit a regularized regression model (Ridge or Lasso) to the data.
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix with shape (n_samples, n_features)
    y : 1D numpy array
        Target values
    alpha : float
        Regularization strength
    regression_type : str
        Type of regularization ('ridge' or 'lasso')
        
    Returns:
    --------
    tuple : (model, metrics)
        model: Trained regression model
        metrics: Dictionary containing model performance metrics
    """
    if regression_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif regression_type == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("regression_type must be 'ridge' or 'lasso'")
    
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred),
        'alpha': alpha,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'type': regression_type
    }
    
    return model, metrics

def fit_random_forest(X, y, n_estimators=100, max_depth=None):
    """
    Fit a Random Forest regression model to the data.
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix with shape (n_samples, n_features)
    y : 1D numpy array
        Target values
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of the trees
        
    Returns:
    --------
    tuple : (model, metrics)
        model: Trained RandomForestRegressor model
        metrics: Dictionary containing model performance metrics
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred),
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_importances': model.feature_importances_
    }
    
    return model, metrics

def compare_regression_models(params_dict, energy_resolutions):
    """
    Compare different regression models on energy resolution data.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary containing parameter arrays
    energy_resolutions : 3D numpy array
        Array containing energy resolution values
        
    Returns:
    --------
    dict
        Dictionary containing all trained models and their metrics
    """
    # Prepare data
    X, y = prepare_regression_data(params_dict, energy_resolutions)
    
    # Check if we have valid data
    if len(X) == 0 or len(y) == 0:
        print("Warning: No valid data available for regression analysis!")
        print("Creating synthetic demonstration data...")
        
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 200
        X = np.random.rand(n_samples, 3)
        X[:, 0] = X[:, 0] * 0.7 + 0.3  # sampling_ratio: 0.3-1.0
        X[:, 1] = X[:, 1] * 130 + 20   # inner_radius: 20-150
        X[:, 2] = X[:, 2] * 35 + 5      # ring_width: 5-40
        
        # Create synthetic energy resolution with realistic relationships
        y = (2.5 + 0.5 * X[:, 0] + 0.01 * X[:, 1] - 0.02 * X[:, 2] +
             0.3 * np.random.randn(n_samples))
        y = np.clip(y, 0.5, 8.0)  # Keep in realistic range
        
        print(f"Created synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Synthetic target range: {np.min(y):.2f}% - {np.max(y):.2f}%")
    else:
        print(f"Prepared regression data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target variable range: {np.min(y):.2f}% - {np.max(y):.2f}%")
    
    # Initialize results dictionary
    results = {}
    
    # 1. Linear Regression
    print("\nFitting Linear Regression...")
    linear_model, linear_metrics = fit_linear_regression(X, y)
    results['linear'] = {
        'model': linear_model,
        'metrics': linear_metrics
    }
    print(f"Linear Regression - R²: {linear_metrics['r2']:.4f}, RMSE: {linear_metrics['rmse']:.4f}")
    
    # 2. Polynomial Regression (degree 2)
    print("\nFitting Polynomial Regression (degree=2)...")
    poly_model, poly_features, poly_metrics, cross_terms = fit_polynomial_regression(X, y, degree=2)
    results['polynomial'] = {
        'model': poly_model,
        'poly_features': poly_features,
        'metrics': poly_metrics,
        'cross_terms': cross_terms
    }
    print(f"Polynomial Regression - R²: {poly_metrics['r2']:.4f}, RMSE: {poly_metrics['rmse']:.4f}")
    
    # 3. Ridge Regression
    print("\nFitting Ridge Regression...")
    ridge_model, ridge_metrics = fit_regularized_regression(X, y, alpha=1.0, regression_type='ridge')
    results['ridge'] = {
        'model': ridge_model,
        'metrics': ridge_metrics
    }
    print(f"Ridge Regression - R²: {ridge_metrics['r2']:.4f}, RMSE: {ridge_metrics['rmse']:.4f}")
    
    # 4. Lasso Regression
    print("\nFitting Lasso Regression...")
    lasso_model, lasso_metrics = fit_regularized_regression(X, y, alpha=0.1, regression_type='lasso')
    results['lasso'] = {
        'model': lasso_model,
        'metrics': lasso_metrics
    }
    print(f"Lasso Regression - R²: {lasso_metrics['r2']:.4f}, RMSE: {lasso_metrics['rmse']:.4f}")
    
    # 5. Random Forest
    print("\nFitting Random Forest...")
    rf_model, rf_metrics = fit_random_forest(X, y, n_estimators=100)
    results['random_forest'] = {
        'model': rf_model,
        'metrics': rf_metrics
    }
    print(f"Random Forest - R²: {rf_metrics['r2']:.4f}, RMSE: {rf_metrics['rmse']:.4f}")
    
    return results, X, y

def analyze_model_coefficients(results, params_dict):
    """
    Analyze and print coefficients/importances of different models (optimized for performance).
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all trained models and their metrics
    params_dict : dict
        Dictionary containing parameter arrays
    """
    # Get feature names
    feature_names = ['Sampling Ratio', 'Inner Radius', 'Ring Width']
    
    print("\n" + "="*60)
    print("MODEL COEFFICIENTS AND FEATURE IMPORTANCES")
    print("="*60)
    
    # 1. Linear Regression Coefficients
    linear_coefs = results['linear']['metrics']['coefficients']
    print(f"\nLinear Regression Coefficients:")
    for name, coef in zip(feature_names, linear_coefs):
        print(f"  {name}: {coef:.6f}")
    
    # 2. Ridge Regression Coefficients
    ridge_coefs = results['ridge']['metrics']['coefficients']
    print(f"\nRidge Regression Coefficients:")
    for name, coef in zip(feature_names, ridge_coefs):
        print(f"  {name}: {coef:.6f}")
    
    # 3. Lasso Regression Coefficients
    lasso_coefs = results['lasso']['metrics']['coefficients']
    print(f"\nLasso Regression Coefficients:")
    for name, coef in zip(feature_names, lasso_coefs):
        print(f"  {name}: {coef:.6f}")
    
    # 4. Random Forest Feature Importances
    rf_importances = results['random_forest']['metrics']['feature_importances']
    print(f"\nRandom Forest Feature Importances:")
    for name, importance in zip(feature_names, rf_importances):
        print(f"  {name}: {importance:.6f}")
    
    print("="*60)

def analyze_cross_terms(results):
    """
    Analyze and visualize cross-term interactions from polynomial regression.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all trained models and their metrics
    """
    if 'polynomial' not in results or 'cross_terms' not in results['polynomial']['metrics']:
        print("\nNo cross-term analysis available. Polynomial regression not performed with cross-terms.")
        return None, None, None
    
    cross_terms = results['polynomial']['metrics']['cross_terms']
    
    print("\n" + "="*60)
    print("CROSS-TERM ANALYSIS")
    print("="*60)
    
    # Separate linear terms from cross-terms
    linear_terms = {}
    interaction_terms = {}
    quadratic_terms = {}
    
    for term, coef in cross_terms.items():
        if ' ' in term and '*' in term:
            # This is an interaction term (cross-term)
            interaction_terms[term] = coef
        elif '^2' in term:
            # This is a quadratic term
            quadratic_terms[term] = coef
        else:
            # This is a linear term
            linear_terms[term] = coef
    
    print("\nLinear Terms:")
    for term, coef in linear_terms.items():
        print(f"  {term}: {coef:.6f}")
    
    print("\nQuadratic Terms:")
    for term, coef in quadratic_terms.items():
        print(f"  {term}: {coef:.6f}")
    
    print("\nCross-Term Interactions:")
    for term, coef in interaction_terms.items():
        print(f"  {term}: {coef:.6f}")
    
    # Find most significant cross-terms
    if interaction_terms:
        sorted_interactions = sorted(interaction_terms.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\nTop 3 Most Significant Cross-Terms:")
        for i, (term, coef) in enumerate(sorted_interactions[:3]):
            print(f"  {i+1}. {term}: {coef:.6f}")
    
    return linear_terms, quadratic_terms, interaction_terms

def plot_random_forest_analysis(X, y, results, params_dict):
    """
    Create visualization plots for Random Forest analysis with parameter plots only (PDP separated).
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix
    y : 1D numpy array
        Target values
    results : dict
        Dictionary containing all trained models and their metrics
    params_dict : dict
        Dictionary containing parameter arrays
    """
    if 'random_forest' not in results:
        print("No Random Forest model available for analysis.")
        return
    
    rf_model = results['random_forest']['model']
    rf_metrics = results['random_forest']['metrics']
    
    # Set optimized matplotlib parameters
    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['agg.path.chunksize'] = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Random Forest Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Get predictions
    y_pred = rf_model.predict(X)
    
    # 1. Actual vs Predicted Scatter Plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y, y_pred, alpha=0.6, c=y, cmap='viridis', edgecolor='black', s=30)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Energy Resolution (%)')
    ax1.set_ylabel('Predicted Energy Resolution (%)')
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Energy Resolution (%)')
    
    # Add R² annotation
    r2 = rf_metrics['r2']
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Separate Parameter Impact Analysis - Sampling Ratio
    ax2 = axes[0, 1]
    sampling_ratios = params_dict['sampling_ratio']
    mean_res_sampling = []
    std_res_sampling = []
    
    for sr in sampling_ratios:
        mask = np.isclose(X[:, 0], sr, atol=0.01)
        mean_res_sampling.append(np.mean(y[mask]))
        std_res_sampling.append(np.std(y[mask]))
    
    optimal_sr_idx = np.argmin(mean_res_sampling)
    optimal_sr = sampling_ratios[optimal_sr_idx]
    
    # Convert to numpy arrays for easier handling
    mean_res_sampling = np.array(mean_res_sampling)
    std_res_sampling = np.array(std_res_sampling)
    
    ax2.errorbar(sampling_ratios, mean_res_sampling, yerr=std_res_sampling,
                fmt='b-o', linewidth=2, markersize=6, alpha=0.7, capsize=5, capthick=2)
    ax2.axvline(x=optimal_sr, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    ax2.plot(optimal_sr, mean_res_sampling[optimal_sr_idx], 'bo', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    sr_min_idx = max(0, optimal_sr_idx - 1)
    sr_max_idx = min(len(sampling_ratios) - 1, optimal_sr_idx + 1)
    ax2.axvspan(sampling_ratios[sr_min_idx], sampling_ratios[sr_max_idx], alpha=0.2, color='blue')
    
    ax2.set_xlabel('Sampling Ratio')
    ax2.set_ylabel('Mean Energy Resolution (%)')
    ax2.set_title('Sampling Ratio Impact', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Optimal: {optimal_sr:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # 3. Separate Parameter Impact Analysis - Inner Radius
    ax3 = axes[1, 0]
    inner_radii = params_dict['inner_radius']
    mean_res_inner = []
    std_res_inner = []
    
    for ir in inner_radii:
        mask = np.isclose(X[:, 1], ir, atol=1.0)
        mean_res_inner.append(np.mean(y[mask]))
        std_res_inner.append(np.std(y[mask]))
    
    optimal_ir_idx = np.argmin(mean_res_inner)
    optimal_ir = inner_radii[optimal_ir_idx]
    
    # Convert to numpy arrays for easier handling
    mean_res_inner = np.array(mean_res_inner)
    std_res_inner = np.array(std_res_inner)
    
    ax3.errorbar(inner_radii, mean_res_inner, yerr=std_res_inner,
                fmt='r-s', linewidth=2, markersize=6, alpha=0.7, capsize=5, capthick=2)
    ax3.axvline(x=optimal_ir, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.plot(optimal_ir, mean_res_inner[optimal_ir_idx], 'rs', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ir_min_idx = max(0, optimal_ir_idx - 1)
    ir_max_idx = min(len(inner_radii) - 1, optimal_ir_idx + 1)
    ax3.axvspan(inner_radii[ir_min_idx], inner_radii[ir_max_idx], alpha=0.2, color='red')
    
    ax3.set_xlabel('Inner Radius (pixels)')
    ax3.set_ylabel('Mean Energy Resolution (%)')
    ax3.set_title('Inner Radius Impact', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'Optimal: {optimal_ir:.0f} px', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # 4. Separate Parameter Impact Analysis - Ring Width
    ax4 = axes[1, 1]
    ring_widths = params_dict['ring_width']
    mean_res_width = []
    std_res_width = []
    
    for rw in ring_widths:
        mask = np.isclose(X[:, 2], rw, atol=1.0)
        mean_res_width.append(np.mean(y[mask]))
        std_res_width.append(np.std(y[mask]))
    
    optimal_rw_idx = np.argmin(mean_res_width)
    optimal_rw = ring_widths[optimal_rw_idx]
    
    # Convert to numpy arrays for easier handling
    mean_res_width = np.array(mean_res_width)
    std_res_width = np.array(std_res_width)
    
    ax4.errorbar(ring_widths, mean_res_width, yerr=std_res_width,
                fmt='g-^', linewidth=2, markersize=6, alpha=0.7, capsize=5, capthick=2)
    ax4.axvline(x=optimal_rw, color='green', linestyle='--', alpha=0.8, linewidth=2)
    ax4.plot(optimal_rw, mean_res_width[optimal_rw_idx], 'g^', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    rw_min_idx = max(0, optimal_rw_idx - 1)
    rw_max_idx = min(len(ring_widths) - 1, optimal_rw_idx + 1)
    ax4.axvspan(ring_widths[rw_min_idx], ring_widths[rw_max_idx], alpha=0.2, color='green')
    
    ax4.set_xlabel('Ring Width (pixels)')
    ax4.set_ylabel('Mean Energy Resolution (%)')
    ax4.set_title('Ring Width Impact', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, f'Optimal: {optimal_rw:.0f} px', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('random_forest_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Random Forest parameter analysis plot saved as 'random_forest_analysis.png'")

def plot_shap_analysis(X, y, results, params_dict):
    """
    Create SHAP (SHapley Additive exPlanations) analysis to replace PDP.
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix
    y : 1D numpy array
        Target values
    results : dict
        Dictionary containing all trained models and their metrics
    params_dict : dict
        Dictionary containing parameter arrays
    """
    if 'random_forest' not in results:
        print("No Random Forest model available for SHAP analysis.")
        return
    
    rf_model = results['random_forest']['model']
    feature_names = ['Sampling Ratio', 'Inner Radius', 'Ring Width']
    
    # Set optimized matplotlib parameters
    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['agg.path.chunksize'] = 10000
    
    print("Computing SHAP values...")
    
    # Create SHAP explainer - use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(rf_model)
    
    # Calculate SHAP values for a subset of data for efficiency
    max_samples = min(1000, len(X))  # Limit to 1000 samples for efficiency
    sample_indices = np.random.choice(len(X), max_samples, replace=False)
    X_sample = X[sample_indices]
    
    shap_values = explainer.shap_values(X_sample)
    
    # If shap_values is a list (for multi-class), take the first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Create comprehensive SHAP analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('SHAP (SHapley Additive exPlanations) Analysis', fontsize=16, fontweight='bold')
    
    # 1. SHAP Summary Plot (beeswarm)
    ax1 = axes[0, 0]
    # Create custom beeswarm plot
    for i, feature_name in enumerate(feature_names):
        # Get SHAP values for this feature
        feature_shap = shap_values[:, i]
        feature_values = X_sample[:, i]
        
        # Create scatter plot with color based on feature value
        scatter = ax1.scatter(feature_shap, np.full(len(feature_shap), i),
                            c=feature_values, cmap='viridis', alpha=0.6, s=20)
        
        # Add mean SHAP value line
        mean_shap = np.mean(feature_shap)
        ax1.axvline(mean_shap, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add feature name and mean SHAP value
        ax1.text(mean_shap + 0.01, i, f'{mean_shap:.3f}',
                fontsize=8, va='center', fontweight='bold')
    
    ax1.set_xlabel('SHAP Value (Impact on Energy Resolution)')
    ax1.set_ylabel('Features')
    ax1.set_title('SHAP Summary Plot', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add colorbar for feature values
    cbar1 = plt.colorbar(scatter, ax=ax1, orientation='horizontal', pad=0.1)
    cbar1.set_label('Feature Value', fontsize=10)
    
    # 2. SHAP Feature Importance Bar Plot
    ax2 = axes[0, 1]
    # Calculate mean absolute SHAP values for feature importance
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax2.bar(range(len(feature_names)), feature_importance[sorted_idx],
                   color=[colors[i] for i in sorted_idx], alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Mean |SHAP Value|')
    ax2.set_title('SHAP Feature Importance', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels([feature_names[i] for i in sorted_idx])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, feature_importance[sorted_idx])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. SHAP Dependence Plots for each feature
    ax3 = axes[1, 0]
    
    # Create dependence plot for the most important feature
    most_important_idx = sorted_idx[0]
    feature_values = X_sample[:, most_important_idx]
    feature_shap = shap_values[:, most_important_idx]
    
    # Sort by feature value for better visualization
    sort_idx = np.argsort(feature_values)
    sorted_feature_values = feature_values[sort_idx]
    sorted_feature_shap = feature_shap[sort_idx]
    
    # Find second most important feature for color interaction
    second_important_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
    interaction_feature_values = X_sample[:, second_important_idx]
    sorted_interaction_values = interaction_feature_values[sort_idx]
    
    # Create scatter plot with trend line - color by interaction feature
    scatter3 = ax3.scatter(sorted_feature_values, sorted_feature_shap,
                           alpha=0.6, s=30, c=sorted_interaction_values, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(sorted_feature_values, sorted_feature_shap, 1)
    p = np.poly1d(z)
    ax3.plot(sorted_feature_values, p(sorted_feature_values), "r--", alpha=0.8, linewidth=2)
    
    ax3.set_xlabel(f'{feature_names[most_important_idx]}')
    ax3.set_ylabel('SHAP Value')
    ax3.set_title(f'SHAP Dependence: {feature_names[most_important_idx]}',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label(f'{feature_names[second_important_idx]} Value', fontsize=10)
    
    # 4. SHAP Interaction Summary
    ax4 = axes[1, 1]
    
    # Calculate SHAP interaction values (simplified version)
    interaction_matrix = np.zeros((len(feature_names), len(feature_names)))
    
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            if i != j:
                # Calculate interaction strength as correlation between SHAP values
                interaction_matrix[i, j] = np.abs(np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1])
            else:
                interaction_matrix[i, j] = np.mean(np.abs(shap_values[:, i]))
    
    # Create heatmap
    im = ax4.imshow(interaction_matrix, cmap='viridis', aspect='auto')
    
    # Add labels and colorbar
    ax4.set_xticks(range(len(feature_names)))
    ax4.set_yticks(range(len(feature_names)))
    ax4.set_xticklabels(feature_names, rotation=45, ha='right')
    ax4.set_yticklabels(feature_names)
    ax4.set_title('SHAP Interaction Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax4.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax4, label='Interaction Strength')
    
    plt.tight_layout()
    plt.savefig('shap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print SHAP insights
    print("\n" + "="*60)
    print("SHAP ANALYSIS INSIGHTS")
    print("="*60)
    
    print(f"\nFeature Importance (Mean |SHAP Value|):")
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    print(f"\nMost Important Feature: {feature_names[most_important_idx]}")
    print(f"  Mean SHAP Value: {np.mean(shap_values[:, most_important_idx]):.4f}")
    print(f"  SHAP Value Range: [{np.min(shap_values[:, most_important_idx]):.4f}, {np.max(shap_values[:, most_important_idx]):.4f}]")
    
    # Find optimal values based on SHAP analysis
    print(f"\nOptimal Parameter Ranges (based on SHAP analysis):")
    for i, feature_name in enumerate(feature_names):
        feature_values = X_sample[:, i]
        feature_shap = shap_values[:, i]
        
        # Find values that minimize energy resolution (negative SHAP values)
        optimal_mask = feature_shap < np.percentile(feature_shap, 25)  # Bottom 25%
        if np.sum(optimal_mask) > 0:
            optimal_range = [np.min(feature_values[optimal_mask]), np.max(feature_values[optimal_mask])]
            print(f"  {feature_name}: {optimal_range[0]:.3f} - {optimal_range[1]:.3f}")
    
    print("="*60)
    print("SHAP analysis plot saved as 'shap_analysis.png'")

def plot_regression_results_optimized(X, y, results, best_model_name):
    """
    Create optimized visualization plots for regression results with parameter importance analysis.
    
    Parameters:
    -----------
    X : 2D numpy array
        Feature matrix
    y : 1D numpy array
        Target values
    results : dict
        Dictionary containing all trained models and their metrics
    best_model_name : str
        Name of the best performing model
    """
    # Set optimized matplotlib parameters for faster rendering
    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['agg.path.chunksize'] = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Regression Results Analysis', fontsize=16, fontweight='bold')
    
    # Get predictions from best model
    if best_model_name == 'polynomial':
        poly_features = results['polynomial']['poly_features']
        X_transformed = poly_features.transform(X)
        y_pred = results['polynomial']['model'].predict(X_transformed)
    else:
        y_pred = results[best_model_name]['model'].predict(X)
    
    # 1. Actual vs Predicted Plot
    ax = axes[0, 0]
    ax.scatter(y, y_pred, alpha=0.6, color='blue', edgecolor='black', s=30)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Energy Resolution (%)')
    ax.set_ylabel('Predicted Energy Resolution (%)')
    ax.set_title(f'Actual vs Predicted ({best_model_name.title()})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add R² annotation
    r2 = results[best_model_name]['metrics']['r2']
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Feature Importance/Coefficients Comparison
    ax = axes[0, 1]
    feature_names = ['Sampling\nRatio', 'Inner\nRadius', 'Ring\nWidth']
    
    # Get coefficients or feature importances from different models
    model_types = ['linear', 'ridge', 'lasso', 'random_forest']
    colors = ['blue', 'red', 'green', 'orange']
    
    x_pos = np.arange(len(feature_names))
    width = 0.2
    
    for i, (model_type, color) in enumerate(zip(model_types, colors)):
        if model_type in results:
            if model_type == 'random_forest':
                values = results[model_type]['metrics']['feature_importances']
            else:
                values = np.abs(results[model_type]['metrics']['coefficients'])
            
            ax.bar(x_pos + i*width, values, width, label=model_type.title(), color=color, alpha=0.7)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Coefficient Magnitude / Importance')
    ax.set_title('Feature Importance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width*1.5)
    ax.set_xticklabels(feature_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Parameter Importance Analysis (previously empty space)
    ax = axes[1, 0]
    
    if 'random_forest' in results:
        # Use Random Forest for parameter importance analysis
        rf_model = results['random_forest']['model']
        rf_metrics = results['random_forest']['metrics']
        
        # Feature importances bar chart
        feature_names_pretty = ['Sampling Ratio', 'Inner Radius', 'Ring Width']
        importances = rf_metrics['feature_importances']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = ax.bar(feature_names_pretty, importances, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title('Random Forest Feature Importances', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add permutation importance if available
        try:
            perm_importance = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42, n_jobs=2)
            
            # Create a second y-axis for permutation importance
            ax2 = ax.twinx()
            sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
            ax2.bar(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx],
                   alpha=0.3, color='red', edgecolor='black', label='Permutation')
            ax2.set_ylabel('Permutation Importance', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add value labels for permutation importance
            for i, idx in enumerate(sorted_idx):
                ax2.text(i, perm_importance.importances_mean[idx] + 0.001,
                        f'{perm_importance.importances_mean[idx]:.3f}',
                        ha='center', va='bottom', fontweight='bold', color='red')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
        except ImportError:
            ax.text(0.02, 0.98, 'Permutation importance\nrequires sklearn >= 0.22',
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Random Forest model\nnot available for\nparameter importance analysis',
               ha='center', va='center', fontsize=10)
        ax.set_title('Parameter Importance Analysis', fontsize=12, fontweight='bold')
    
    # 4. Model Performance Comparison
    ax = axes[1, 1]
    models = list(results.keys())
    r2_scores = [results[model]['metrics']['r2'] for model in models]
    rmse_scores = [results[model]['metrics']['rmse'] for model in models]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    ax.bar(x_pos - width/2, r2_scores, width, label='R²', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, rmse_scores, width, label='RMSE', color='red', alpha=0.7)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Regression results plot saved as 'regression_results.png'")

def create_model_performance_comparison(results):
    """
    Print a comparison of model performance metrics (optimized for performance).
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all trained models and their metrics
    """
    # Extract metrics for all models
    models = list(results.keys())
    r2_scores = [results[model]['metrics']['r2'] for model in models]
    rmse_scores = [results[model]['metrics']['rmse'] for model in models]
    mse_scores = [results[model]['metrics']['mse'] for model in models]
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MSE':<10}")
    print("-"*60)
    
    for i, model in enumerate(models):
        print(f"{model:<15} {r2_scores[i]:<10.4f} {rmse_scores[i]:<10.4f} {mse_scores[i]:<10.4f}")
    
    print("-"*60)
    
    # Find best model
    best_idx = np.argmax(r2_scores)
    print(f"\nBest performing model: {models[best_idx]} (R² = {r2_scores[best_idx]:.4f})")
    print("="*60)

def predict_with_best_model(results, X, y, params_dict):
    """
    Use the best performing model to make predictions and analyze the results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all trained models and their metrics
    X : 2D numpy array
        Feature matrix
    y : 1D numpy array
        Target values
    params_dict : dict
        Dictionary containing parameter arrays
    """
    # Find the best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]['metrics']
    
    print(f"\nBest model: {best_model_name}")
    print(f"R²: {best_metrics['r2']:.4f}")
    print(f"RMSE: {best_metrics['rmse']:.4f}")
    
    # Make predictions
    if best_model_name == 'polynomial':
        # For polynomial regression, we need to transform X first
        poly_features = results['polynomial']['poly_features']
        X_transformed = poly_features.transform(X)
        y_pred = best_model.predict(X_transformed)
    else:
        y_pred = best_model.predict(X)
    
    # Calculate residual statistics
    residuals = y - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    print(f"\nPrediction Statistics:")
    print(f"  Mean residual: {mean_residual:.6f}")
    print(f"  Std residual: {std_residual:.6f}")
    print(f"  Max residual: {np.max(np.abs(residuals)):.6f}")
    
    return best_model_name, best_model, y_pred

def optimize_parameters_with_model(results, params_dict):
    """
    Use the best regression model to find optimal parameter combinations.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all trained models and their metrics
    params_dict : dict
        Dictionary containing parameter arrays
    """
    # Find the best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nOptimizing parameters using {best_model_name} model...")
    
    # Create a fine grid for optimization
    fine_sampling = np.linspace(min(params_dict['sampling_ratio']), max(params_dict['sampling_ratio']), 20)
    fine_inner = np.linspace(min(params_dict['inner_radius']), max(params_dict['inner_radius']), 20)
    fine_width = np.linspace(min(params_dict['ring_width']), max(params_dict['ring_width']), 20)
    
    # Generate all combinations
    fine_combinations = []
    for sr in fine_sampling:
        for ir in fine_inner:
            for rw in fine_width:
                fine_combinations.append([sr, ir, rw])
    
    X_fine = np.array(fine_combinations)
    
    # Make predictions
    if best_model_name == 'polynomial':
        poly_features = results['polynomial']['poly_features']
        X_fine_transformed = poly_features.transform(X_fine)
        y_fine_pred = best_model.predict(X_fine_transformed)
    else:
        y_fine_pred = best_model.predict(X_fine)
    
    # Find optimal parameters (minimum energy resolution)
    optimal_idx = np.argmin(y_fine_pred)
    optimal_params = X_fine[optimal_idx]
    optimal_resolution = y_fine_pred[optimal_idx]
    
    print(f"\nOptimal parameters found:")
    print(f"  Sampling Ratio: {optimal_params[0]:.3f}")
    print(f"  Inner Radius: {optimal_params[1]:.1f} pixels")
    print(f"  Ring Width: {optimal_params[2]:.1f} pixels")
    print(f"  Predicted Energy Resolution: {optimal_resolution:.2f}%")
    
    return optimal_params, optimal_resolution

def create_interactive_interface(results, params_dict):
    """
    Create an interactive interface with sliders for real-time energy resolution estimation.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all trained models and their metrics
    params_dict : dict
        Dictionary containing parameter arrays
    """
    try:
        import tkinter as tk
        from tkinter import ttk
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError:
        print("Error: tkinter is not available. Cannot create interactive interface.")
        print("Please install tkinter: pip install tk")
        return
    
    if 'random_forest' not in results:
        print("No Random Forest model available for interactive interface.")
        return
    
    rf_model = results['random_forest']['model']
     
    # Get parameter ranges from the original settings
    sampling_min = float(params_dict['sampling_ratio'].min())
    sampling_max = float(params_dict['sampling_ratio'].max())
    inner_min = float(params_dict['inner_radius'].min())
    inner_max = float(params_dict['inner_radius'].max())
    width_min = float(params_dict['ring_width'].min())
    width_max = float(params_dict['ring_width'].max())
    
    # Create main window
    root = tk.Tk()
    root.title("Energy Resolution Estimator - Interactive Interface")
    root.geometry("800x600")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Title
    title_label = ttk.Label(main_frame, text="Energy Resolution Estimator",
                          font=('Arial', 16, 'bold'))
    title_label.grid(row=0, column=0, columnspan=3, pady=10)
    
    # Parameter sliders frame
    sliders_frame = ttk.LabelFrame(main_frame, text="Parameter Controls", padding="10")
    sliders_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
    
    # Sampling Ratio Slider
    ttk.Label(sliders_frame, text="Sampling Ratio:").grid(row=0, column=0, sticky=tk.W, pady=5)
    sampling_var = tk.DoubleVar(value=(sampling_min + sampling_max) / 2)
    sampling_slider = ttk.Scale(sliders_frame, from_=sampling_min, to=sampling_max,
                             variable=sampling_var, orient=tk.HORIZONTAL, length=300)
    sampling_slider.grid(row=0, column=1, pady=5, padx=10)
    sampling_label = ttk.Label(sliders_frame, text=f"{sampling_var.get():.3f}")
    sampling_label.grid(row=0, column=2, pady=5)
    
    # Inner Radius Slider
    ttk.Label(sliders_frame, text="Inner Radius (pixels):").grid(row=1, column=0, sticky=tk.W, pady=5)
    inner_var = tk.DoubleVar(value=round((inner_min + inner_max) / 2))
    inner_slider = ttk.Scale(sliders_frame, from_=inner_min, to=inner_max,
                          variable=inner_var, orient=tk.HORIZONTAL, length=300)
    inner_slider.grid(row=1, column=1, pady=5, padx=10)
    inner_label = ttk.Label(sliders_frame, text=f"{int(inner_var.get())}")
    inner_label.grid(row=1, column=2, pady=5)
    
    # Ring Width Slider
    ttk.Label(sliders_frame, text="Ring Width (pixels):").grid(row=2, column=0, sticky=tk.W, pady=5)
    width_var = tk.DoubleVar(value=round((width_min + width_max) / 2))
    width_slider = ttk.Scale(sliders_frame, from_=width_min, to=width_max,
                          variable=width_var, orient=tk.HORIZONTAL, length=300)
    width_slider.grid(row=2, column=1, pady=5, padx=10)
    width_label = ttk.Label(sliders_frame, text=f"{int(width_var.get())}")
    width_label.grid(row=2, column=2, pady=5)
    
    # Results frame
    results_frame = ttk.LabelFrame(main_frame, text="Energy Resolution Estimate", padding="10")
    results_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
    
    # Energy resolution display
    resolution_var = tk.StringVar(value="0.00%")
    resolution_label = ttk.Label(results_frame, textvariable=resolution_var,
                             font=('Arial', 20, 'bold'), foreground='blue')
    resolution_label.grid(row=0, column=0, pady=10)
    
    # Parameter display
    params_text = tk.StringVar(value="")
    params_label = ttk.Label(results_frame, textvariable=params_text,
                         font=('Arial', 10))
    params_label.grid(row=1, column=0, pady=5)
    
    # Create matplotlib figure for real-time visualization
    fig = Figure(figsize=(8, 4), dpi=100)  # Increased figure size from (6, 3) to (8, 4)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Energy Resolution (%)')
    ax.set_title('Real-time Energy Resolution Estimation', pad=20)  # Added padding
    ax.grid(True, alpha=0.3)
    
    # Embed matplotlib figure in tkinter
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
    
    # Store plot data
    plot_data = {'sampling': [], 'inner': [], 'width': [], 'resolution': []}
    
    def update_estimation():
        """Update energy resolution estimation based on current slider values"""
        # Get current parameter values
        sampling_ratio = sampling_var.get()
        inner_radius = inner_var.get()
        ring_width = width_var.get()
        
        # Create input for prediction
        X_input = np.array([[sampling_ratio, inner_radius, ring_width]])
        
        # Make prediction
        predicted_resolution = rf_model.predict(X_input)[0]
        
        # Update labels - ensure both inner radius and ring width are displayed as integers
        sampling_label.config(text=f"{sampling_ratio:.3f}")
        inner_label.config(text=f"{int(inner_radius)}")
        width_label.config(text=f"{int(ring_width)}")
        resolution_var.set(f"{predicted_resolution:.2f}%")
        params_text.set(f"SR: {sampling_ratio:.3f}, IR: {int(inner_radius)}px, RW: {int(ring_width)}px")
        
        # Update plot data (keep last 50 points) - ensure both inner radius and ring width are stored as integers
        plot_data['sampling'].append(sampling_ratio)
        plot_data['inner'].append(int(inner_radius))
        plot_data['width'].append(int(ring_width))
        plot_data['resolution'].append(predicted_resolution)
        
        # Keep only last 50 points for visualization
        max_points = 50
        if len(plot_data['resolution']) > max_points:
            for key in plot_data:
                plot_data[key] = plot_data[key][-max_points:]
        
        # Update plot
        ax.clear()
        
        if len(plot_data['resolution']) > 1:
            # Plot line graph of estimation history
            ax.plot(range(len(plot_data['resolution'])), plot_data['resolution'], 'b-', alpha=0.7, linewidth=2)
            # Highlight current point
            ax.scatter(len(plot_data['resolution'])-1, plot_data['resolution'][-1],
                     color='red', s=80, zorder=5, edgecolor='black', linewidth=2)
            
            # Add parameter annotations for current point
            current_sr = plot_data['sampling'][-1] if plot_data['sampling'] else 0.5
            current_ir = plot_data['inner'][-1] if plot_data['inner'] else 100
            current_rw = plot_data['width'][-1] if plot_data['width'] else 15
            
            # Add text box with current parameters
            param_text = f"SR: {current_sr:.3f}\nIR: {current_ir}px\nRW: {current_rw}px"
            ax.text(len(plot_data['resolution'])-1, plot_data['resolution'][-1],
                    param_text, fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        else:
            # Handle case with no data yet
            ax.text(0.5, 0.5, 'Adjust sliders to see estimations',
                    ha='center', va='center', fontsize=12, alpha=0.7)
        
        ax.set_xlabel('Estimation Sequence Number', fontsize=10, labelpad=15)
        ax.set_ylabel('Energy Resolution (%)', fontsize=10, labelpad=10)
        ax.set_title('Real-time Energy Resolution Estimation History', fontsize=12, pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        
        # Adjust subplot parameters to prevent title/label truncation
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add padding around the plot
        
        canvas.draw()
    
    # Bind slider events to update function
    sampling_var.trace('w', lambda *args: update_estimation())
    inner_var.trace('w', lambda *args: update_estimation())
    width_var.trace('w', lambda *args: update_estimation())
    
    # Add control buttons with more spacing
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=4, column=0, columnspan=3, pady=20)  # Increased pady from 10 to 20
    
    def reset_sliders():
        """Reset sliders to default values"""
        sampling_var.set((sampling_min + sampling_max) / 2)
        inner_var.set((inner_min + inner_max) / 2)
        width_var.set((width_min + width_max) / 2)
        plot_data['sampling'].clear()
        plot_data['inner'].clear()
        plot_data['width'].clear()
        plot_data['resolution'].clear()
        update_estimation()
    
    def save_current():
        """Save current parameters and estimation"""
        sampling_ratio = sampling_var.get()
        inner_radius = int(inner_var.get())  # Ensure integer value
        ring_width = int(width_var.get())  # Ensure integer value
        predicted_resolution = rf_model.predict([[sampling_ratio, inner_radius, ring_width]])[0]
        
        with open('energy_resolution_estimates.log', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"SR: {sampling_ratio:.3f}, IR: {inner_radius}px, "
                    f"RW: {ring_width}px -> Resolution: {predicted_resolution:.2f}%\n")
        print(f"Estimation saved to energy_resolution_estimates.log")
    
    reset_button = ttk.Button(button_frame, text="Reset Sliders", command=reset_sliders)
    reset_button.grid(row=0, column=0, padx=5)
    
    save_button = ttk.Button(button_frame, text="Save Current", command=save_current)
    save_button.grid(row=0, column=1, padx=5)
    
    quit_button = ttk.Button(button_frame, text="Quit", command=root.quit)
    quit_button.grid(row=0, column=2, padx=5)
    
    # Initial estimation
    update_estimation()
    
    # Start the GUI
    root.mainloop()

# Main execution - Enhanced with separate parameter plots, SHAP, and permutation importance
if __name__ == "__main__":
    start_time = time.time()
    
    # Define parameter ranges for analysis - adjusted for better data quality
    sampling_ratio_list = np.linspace(0.1, 1.0, 20,dtype=np.float16)  # 50 points from 0.1 to 1.0 (higher density)
    inner_radius_list = np.linspace(5, 150, 20,dtype=int)  # 50 points from 5 to 150 (more reasonable range)
    ring_width_list = np.linspace(1, 30, 20,dtype=int)  # 50 points from 1 to 40 (wider rings for better signal)
    
    print(f"Enhanced Random Forest analysis: {len(sampling_ratio_list)}×{len(inner_radius_list)}×{len(ring_width_list)} = {len(sampling_ratio_list)*len(inner_radius_list)*len(ring_width_list)} total combinations")
    print("Note: Using parallel processing with separate parameter plots, SHAP, and permutation importance")
    
    print("Generating ring matrices with different parameters...")
    print(f"Sampling ratios: {sampling_ratio_list}")
    print(f"Inner radii: {inner_radius_list}")
    print(f"Ring widths: {ring_width_list}")
    
    # Generate all matrices
    matrices_array, params_dict, actual_points_array = generate_ring_matrices_array(
        sampling_ratio_list, inner_radius_list, ring_width_list
    )
    
    print(f"Generated matrices array with shape: {matrices_array.shape}")
    print(f"Total matrices: {np.prod(matrices_array.shape[:3])}")
    
    # Calculate energy resolution for all matrices using parallel processing
    print("\nCalculating energy resolution for all matrices (parallel processing)...")
    energy_resolutions = calculate_all_energy_resolutions(matrices_array, use_parallel=True)
    
    print(f"Energy resolutions calculated. Shape: {energy_resolutions.shape}")
    print(f"Mean energy resolution: {np.mean(energy_resolutions):.2f}%")
    print(f"Min energy resolution: {np.min(energy_resolutions):.2f}%")
    print(f"Max energy resolution: {np.max(energy_resolutions):.2f}%")
    
    # REGRESSION ANALYSIS
    print("\n" + "="*50)
    print("STARTING REGRESSION ANALYSIS")
    print("="*50)
    
    # Compare different regression models
    results, X, y = compare_regression_models(params_dict, energy_resolutions)
    
    # Analyze model coefficients and feature importances
    print("\nAnalyzing model coefficients and feature importances...")
    analyze_model_coefficients(results, params_dict)
    
    # Analyze cross-terms
    print("\nAnalyzing cross-term interactions...")
    linear_terms, quadratic_terms, interaction_terms = analyze_cross_terms(results)
    
    # Compare model performance
    print("\nComparing model performance...")
    create_model_performance_comparison(results)
    
    # Use the best model for predictions
    print("\nUsing best model for predictions...")
    best_model_name, best_model, y_pred = predict_with_best_model(results, X, y, params_dict)
    
    # Create comprehensive Random Forest analysis plots
    print("\nCreating Random Forest comprehensive analysis plots...")
    plot_random_forest_analysis(X, y, results, params_dict)
    
    # Create SHAP analysis plots
    print("\nCreating SHAP analysis plots...")
    plot_shap_analysis(X, y, results, params_dict)
    
    # Create regression results visualization
    print("\nCreating regression results plots...")
    plot_regression_results_optimized(X, y, results, best_model_name)
    
    # Optimize parameters using the best model
    print("\nOptimizing parameters using best model...")
    optimal_params, optimal_resolution = optimize_parameters_with_model(results, params_dict)
    
    # Create a summary table of all model results
    print("\n" + "="*50)
    print("REGRESSION MODEL SUMMARY")
    print("="*50)
    
    model_summary_data = []
    for model_name in results.keys():
        metrics = results[model_name]['metrics']
        model_summary_data.append([
            model_name,
            f"{metrics['r2']:.4f}",
            f"{metrics['rmse']:.4f}",
            f"{metrics['mse']:.4f}"
        ])
    
    # Display summary table
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MSE':<10}")
    print("-"*50)
    
    for row in model_summary_data:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10}")
        print("-"*50)
    
    # Display optimal parameters
    print(f"\nOptimal Parameters (using {best_model_name}):")
    print(f"  Sampling Ratio: {optimal_params[0]:.3f}")
    print(f"  Inner Radius: {optimal_params[1]:.1f} pixels")
    print(f"  Ring Width: {optimal_params[2]:.1f} pixels")
    print(f"  Predicted Energy Resolution: {optimal_resolution:.2f}%")
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    print(f"Performance: {np.prod(matrices_array.shape[:3])/execution_time:.1f} matrix calculations per second")
    
    print("\nEnhanced Random Forest analysis complete.")
    print("Generated plots:")
    print("  - random_forest_analysis.png")
    print("  - shap_analysis.png")
    print("  - regression_results.png")
    
    # Launch interactive interface for real-time energy resolution estimation
    print("\nLaunching interactive parameter interface...")
    print("Use the sliders to adjust parameters and get real-time energy resolution estimates.")
    create_interactive_interface(results, params_dict)