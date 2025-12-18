"""
Utility Functions for Phase Diagram Gaussian Process Models

This module provides essential utility functions for data handling, kernel
configuration, active learning, and mathematical operations used throughout
the phaseGP library.

Key Functionality:
- Data type conversions between NumPy and PyTorch
- Kernel initialization and configuration
- Active learning point selection with spacing constraints
- Grid generation for phase diagram evaluation
- Input scaling and normalization
- Random seed management

Author: Eduardo Gonzalez Garcia (e.gonzalez.garcia@tue.nl)
Version: 0.1.0
"""
import gpytorch
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from botorch.optim import optimize_acqf

# =============================================================================
# Data Type Conversion Utilities
# =============================================================================

def ensure_tensor(x, dtype=torch.float32, device="cpu"):
    """
    Convert input to PyTorch tensor if not already.
    
    Ensures consistent tensor format throughout the library, converting
    NumPy arrays, lists, or other array-like objects to PyTorch tensors
    in the provided device.
    
    Args:
        x: Input data (array-like, tensor, or scalar)
        dtype (torch.dtype): Desired tensor data type
        device ("cpu" or "cuda"): Desired device
        
    Returns:
        torch.Tensor: Input as a PyTorch tensor
    """
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype).to(device)
    return x.to(device)

def ensure_numpy(x):
    """
    Convert input to NumPy array if not already.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

# =============================================================================
# Kernel Configuration
# =============================================================================

def get_base_kernel(kernel_choice, lengthscale=None, **kwargs):
    """
    Create and configure a GPyTorch kernel based on string specification.
    
    Kernel properties:
    - RBF: Infinitely differentiable, very smooth boundaries
    - Matern52: Twice differentiable, moderately smooth boundaries
    - Matern32: Once differentiable, less smooth boundaries (default)
    
    Args:
        kernel_choice (str): Type of kernel ('rbf', 'matern52', 'matern32')
        lengthscale (float, optional): Initial lengthscale parameter
        **kwargs: Additional kernel parameters (e.g., lengthscale_prior)
        
    Returns:
        gpytorch.kernels.Kernel: Configured kernel instance
        
    Raises:
        ValueError: If kernel_choice is not recognized
    """
    kernel_choice = kernel_choice.lower()
    
    if kernel_choice == 'rbf':
        kernel = gpytorch.kernels.RBFKernel(**kwargs)
    elif kernel_choice == 'matern52':
        kernel = gpytorch.kernels.MaternKernel(nu=2.5, **kwargs)
    elif kernel_choice == 'matern32':
        #(default)
        kernel = gpytorch.kernels.MaternKernel(nu=1.5, **kwargs)
    else:
        raise ValueError(f"Unknown kernel_choice: {kernel_choice}")
    
    # Set initial lengthscale if provided
    if lengthscale is not None:
        kernel.lengthscale = lengthscale
    
    return kernel

# =============================================================================
# Active Learning Utilities
# =============================================================================

def brute_sample_new_points(model, candidates, sampled_points=None, n_sample = 1 ,frac_distance_thresh = 0.1,epsilon=0.05, vanilla_acq=False, distance_acq=True, return_index = False):
    """
    Select new sampling points using acquisition function with spacing constraints when batch sampling.
    
    This function implements the active learning selection strategy, choosing
    points that maximize the acquisition function while maintaining minimum
    spacing between selected points through brue sampling. The spacing constraint prevents clustering
    of samples and ensures better coverage of the phase diagram.
    
    The selection process:
    1. Compute acquisition values for all candidates
    2. Sort candidates by acquisition value (descending)
    3. Greedily select points that satisfy spacing constraints
    
    Args:
        model: GP model with acquisition method
        candidates (torch.Tensor): Candidate points for selection, shape (n, d)
        sampled_points (torch.Tensor): Previously sampled points, shape (m, d)
        n_sample (int): Number of points to select
        frac_distance_thresh (float): Minimum distance between points as fraction of domain
        epsilon (float): Small value for numerical stability in acquisition
        vanilla_acq (bool): Whether to use vanilla acquisition function
        return_index (bool): If True, also return indices of selected points
        
    Returns:
        torch.Tensor or tuple:
            - If return_index=False: Selected points, shape (n_sample, d)
            - If return_index=True: (selected_points, selected_indices)
    """
    # Compute acquisition values for all candidates
    acq_values = model.acquisition(candidates, sampled_points, epsilon=epsilon, vanilla_acq=vanilla_acq, distance_acq=distance_acq)

    if n_sample <= 0:
        return torch.tensor([], dtype=torch.long)
    
    # Limit selection to available candidates
    n_sample = min(n_sample, len(candidates))
    
    # Scale candidates to [0,1] for distance computation
    scaled_candidates = scaler(candidates, model.min_scale, model.max_scale)
    
    # Sort candidates by acquisition value (best first)
    sorted_indices = torch.argsort(acq_values, descending=True)
    
    selected_indices = []
    
    # Greedy selection with spacing constraint
    for idx in sorted_indices:
        coord = scaled_candidates[idx].unsqueeze(0)
        
        # First point is always selected
        if len(selected_indices) == 0:
            selected_indices.append(idx.item())
        else:
            # Check distance to all previously selected points
            selected_coords = scaled_candidates[selected_indices]
            
            # Compute distances using L2 norm
            diffs = torch.cdist(selected_coords, coord)
            
            # Add point only if it's far enough from all selected points
            if not torch.any(diffs < frac_distance_thresh):
                selected_indices.append(idx.item())
        
        # Stop when we have enough points
        if len(selected_indices) == n_sample:
            break
    
    #if(n_sample == 1):
    #    selected_indices = selected_indices[0]

    if(return_index):
        return candidates[selected_indices], selected_indices
    else:
        return candidates[selected_indices]

def gradient_sample_new_points(model, sampled_points=None, n_sample = 1 ,frac_distance_thresh = 0.1,epsilon=0.05, vanilla_acq=False, distance_acq=True,
                               num_restarts = 10, raw_samples = 512):
    """
    Select new sampling points using gradient-based acquisition optimization with spacing constraints when batch sampling.
    
    This function implements active learning selection using gradient-based optimization
    to find points that maximize the acquisition function while maintaining minimum
    spacing between selected points (when batch sampling). The optimization uses BoTorch's optimize_acqf
    to search the continuous domain.
    
    The selection process:
    1. Define acquisition function with distance penalty for spacing
    2. Iteratively optimize to find next best point
    3. Add selected point and repeat until n_sample points selected
    
    Args:
        model: GP model with acquisition method
        sampled_points (torch.Tensor, optional): Previously sampled points, shape (m, d)
        n_sample (int): Number of points to select
        frac_distance_thresh (float): Minimum distance between points as fraction of domain
        epsilon (float): Small value for numerical stability in acquisition
        vanilla_acq (bool): Whether to use vanilla acquisition function
        distance_acq (bool): Whether to include distance-based acquisition component
        num_restarts (int): Number of random restarts for optimization
        raw_samples (int): Number of raw samples for initialization
        
    Returns:
        torch.Tensor: Selected points, shape (n_sample, d)
    """
    bounds = torch.stack((model.min_scale, model.max_scale)).to(device=model.device)
    bounds = bounds.type(torch.float32)
    selected_candidates = torch.empty((0, bounds.size(1)), device=model.device)

    for i in range(n_sample):
        acq_batch_distance_penalty = partial(_acq_batch_distance_penalty, model=model,selected_candidates=selected_candidates ,sampled_points=sampled_points, frac_distance_thresh = frac_distance_thresh,
                                             epsilon = epsilon, vanilla_acq = vanilla_acq, distance_acq = distance_acq)

        candidate, value = optimize_acqf(
            acq_function=acq_batch_distance_penalty,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,  # Increase for better global search
            raw_samples=raw_samples,
        )

        selected_candidates = torch.cat([selected_candidates, candidate], dim=0)

    return selected_candidates

def _acq_batch_distance_penalty(X, model, selected_candidates, sampled_points=None, frac_distance_thresh = 0.1, epsilon=0.05, vanilla_acq=False, distance_acq=True):
    """
    Compute acquisition function with soft distance penalty when selecting a batch.
    
    This internal helper function calculates a penalized acquisition value that
    discourages selecting points too close to already-selected candidates (when sampling a batch). The
    penalty uses a soft ReLU-based formulation to maintain differentiability
    for gradient-based optimization.
    
    The acquisition value is computed as:
        final_acq = base_acquisition - penalty
    
    where penalty grows as points get closer to selected_candidates within the
    threshold distance.
    
    Args:
        X (torch.Tensor): Candidate points to evaluate, shape (batch_size, d) or (batch_size, 1, d)
        model: GP model with acquisition method
        selected_candidates (torch.Tensor): Already selected points in current batch, shape (n_selected, d)
        sampled_points (torch.Tensor, optional): Previously sampled points, shape (m, d)
        frac_distance_thresh (float): Distance threshold for penalty activation
        epsilon (float): Small value for numerical stability in acquisition
        vanilla_acq (bool): Whether to use vanilla acquisition function
        distance_acq (bool): Whether to include distance-based acquisition component
        
    Returns:
        torch.Tensor: Penalized acquisition values, shape (batch_size,)
        
    Note:
        This is a private function intended only for use within gradient_sample_new_points.
        The penalty_strength parameter is hardcoded to 1000.0 for consistent behavior.
    """
    penalty_strength = 1000.0
    # Handle dimensions for BoTorch (batch_size, q=1, d) -> (batch_size, d)
    if X.ndim == 3:
        X = X.squeeze(1)
    
    # Calculate Base Acquisition enabling gradients for the optimizer
    base_acquisition = model.acquisition(X, sampled_points, requires_grad=True, epsilon=epsilon, vanilla_acq=vanilla_acq, distance_acq=distance_acq)

    if len(selected_candidates) == 0:
        return base_acquisition
    
    # Compute minimum distance to selected candidates
    dists = torch.cdist(X, selected_candidates)
    closest_dist = dists.min(dim=1).values
    
    # Create a "Soft" Penalty using ReLU
    # If closest_dist < frac_distance_thresh: Penalty grows larger as we get closer
    # If closest_dist >= frac_distance_thresh: Penalty is 0
    # We use ReLU to keep the function differentiable
    penalty = F.relu(frac_distance_thresh - closest_dist) * penalty_strength
    
    return base_acquisition - penalty

# =============================================================================
# Grid Generation
# =============================================================================
    
def get_grid(x_min=0, x_max=1, grid_size=100, return_coordinates=False, 
             y_min=None, y_max=None, device="cpu"):
    """
    Generate a regular N-D grid for phase diagram evaluation in the desired device.
    
    Args:
        x_min (float or list): Minimum coordinate(s). If list, creates N-D grid.
        x_max (float or list): Maximum coordinate(s). If list, creates N-D grid.
        grid_size (int): Number of points along each axis
        return_coordinates (bool): If True, also return coordinate vectors
        y_min (float, optional): Minimum y-coordinate (only for 2D, defaults to x_min)
        y_max (float, optional): Maximum y-coordinate (only for 2D, defaults to x_max)
        device ("cpu" or "cuda"): Desired device
    
    Returns:
        torch.Tensor or tuple:
            - If return_coordinates=False: Grid points, shape (grid_size^N, N)
            - If return_coordinates=True: (grid_points, coord_list)
              where coord_list is a list of coordinate vectors for each dimension
    """
    # Check if we're in N-D mode (x_min/x_max are lists)
    if isinstance(x_min, (list, tuple)) or isinstance(x_max, (list, tuple)):
        # N-D mode
        if not isinstance(x_min, (list, tuple)) or not isinstance(x_max, (list, tuple)):
            raise ValueError("x_min and x_max must both be lists/tuples for N-D grids")
        
        if len(x_min) != len(x_max):
            raise ValueError("x_min and x_max must have the same length")
        
        n_dims = len(x_min)
        
        # Create linearly spaced coordinates for each dimension
        coords = [torch.linspace(x_min[i], x_max[i], grid_size) for i in range(n_dims)]
        
        # Create N-D mesh grid
        mesh = torch.meshgrid(*coords, indexing='ij')
        
        # Flatten and stack to create list of points
        grid_points = torch.stack([m.flatten() for m in mesh], dim=1).to(device)
        
        if return_coordinates:
            return grid_points, coords
        else:
            return grid_points
    
    else:
        # Original 2D mode (backward compatible)
        # Default to square domain if y bounds not specified
        if y_min is None:
            y_min = x_min
        if y_max is None:
            y_max = x_max
        
        # Create linearly spaced coordinates
        x1 = torch.linspace(x_min, x_max, grid_size)
        x2 = torch.linspace(y_min, y_max, grid_size)
        
        # Create 2D mesh grid
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        
        # Flatten to list of points
        grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1).to(device)
        
        if return_coordinates:
            return grid_points, x1, x2
        else:
            return grid_points


# =============================================================================
# Scaling and Normalization
# =============================================================================

def scaler(x, min_scale, max_scale):
    """
    Scale input data to [0, 1] range using min-max normalization.
    
    Args:
        x (torch.Tensor): Input data to scale
        min_scale (torch.Tensor): Minimum values for each dimension
        max_scale (torch.Tensor): Maximum values for each dimension
        
    Returns:
        torch.Tensor: Scaled data in [0, 1] range
    """
    return (x - min_scale) / (max_scale - min_scale)

# =============================================================================
# Random Seed Management
# =============================================================================

def set_seeds(seed):
    """
    Set random seeds for reproducibility across PyTorch and NumPy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)