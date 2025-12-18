"""
Source Model Pruning for Transfer Learning

This module provides functionality to automatically select diverse source models
from a collection of pre-trained models. The goal is to identify a subset of
source models that are sufficiently different from each other, ensuring that
transfer learning benefits from diverse prior knowledge.

The pruning process uses topological and geometric similarity metrics to assess
whether two phase diagrams represent similar or different physical behaviors.
This helps avoid redundancy in the source model ensemble and improves transfer
learning efficiency.

Key Features:
- Topological analysis using persistent homology
- Tolerant phase boundary comparison with dilation
- Connected component and hole counting
- Automatic selection of diverse source models

Dependencies:
    - gudhi: For persistent homology computations
    - scipy.ndimage: For morphological operations
    - sklearn.metrics: For accuracy computation

Author: Eduardo Gonzalez Garcia (e.gonzalez.garcia@tue.nl)
Version: 0.1.0
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.ndimage import binary_dilation, label
import gudhi as gd
import warnings

from .utils import get_grid, ensure_numpy

def tolerant_phase_disagreement(A, B, radius=1):
    """
    Compute disagreement between two phase diagrams with spatial tolerance.
    
    This function identifies regions where two phase diagrams disagree, but
    with a tolerance for small boundary shifts. A point is considered to
    agree if it matches the other diagram OR is within 'radius' pixels of
    a matching region.
    
    This tolerance is important because:
    - Numerical errors can cause slight boundary shifts
    - Different models may represent the same physics with slightly different boundaries
    - Small variations shouldn't be considered fundamental differences
    
    Args:
        A (np.ndarray): First binary phase diagram (2D array of 0s and 1s)
        B (np.ndarray): Second binary phase diagram (2D array of 0s and 1s)
        radius (int): Dilation radius for tolerance (pixels)
        
    Returns:
        np.ndarray: Binary map where 1 indicates disagreement, shape same as input
    """
    # Dilate both phase diagrams to create tolerance zones
    A_dil = binary_dilation(A, iterations=radius)
    B_dil = binary_dilation(B, iterations=radius)

    # Points agree if:
    # - A=1 and B_dilated=1 (A's phase 1 overlaps with dilated B's phase 1)
    # - B=1 and A_dilated=1 (B's phase 1 overlaps with dilated A's phase 1)
    agree = (A & B_dil) | (B & A_dil)
    
    # Disagreement occurs where there's a phase but no agreement
    disagreement = ~(agree) & (A | B)
    
    return disagreement.astype(np.uint8)

def num_connected_regions(map):
    """
    Count the number of connected regions in a binary map.
    """
    _, n = label(map)
    return n

def num_holes(map):
    """
    Count the number of topological holes in a binary phase diagram.
    
    Uses persistent homology to identify 1-dimensional topological features
    (holes/voids) in the phase diagram. Holes represent regions where one
    phase is completely surrounded by another.
    
    The computation treats phase 1 regions as "high elevation" in a height
    function, then computes persistent homology to find significant holes.
    
    Args:
        map (np.ndarray): Binary 2D phase diagram
        
    Returns:
        int: Number of 1-dimensional holes in the diagram
    """
    # Convert to height function: -1 for phase 1 (high), 0 for phase 0 (low)
    # Negative values ensure phase 1 regions are treated as peaks
    f = -map.astype(float)
    
    # Create cubical complex for 2D grid data
    cc = gd.CubicalComplex(dimensions=f.shape, top_dimensional_cells=f.flatten())
    
    # Compute persistent homology
    cc.compute_persistence()
    
    # Extract 1-dimensional features (holes)
    dgm_1 = cc.persistence_intervals_in_dimension(1)
    
    return len(dgm_1)

def topology_metrics(map):
    """
    Compute topological invariants of a phase diagram.
    
    Args:
        map (np.ndarray): Binary 2D phase diagram
        
    Returns:
        tuple: (num_regions, num_holes)
            - num_regions: Number of connected components
            - num_holes: Number of topological holes
    """
    return num_connected_regions(map), num_holes(map)

def similarity_checker(phase_diagram1, phase_diagram2, acc_threshold = 0.8, min_intersection_regions = 3, intersection_tol = 5):
    """
    Determine if two phase diagrams are similar based on multiple criteria.
    
    Two diagrams are considered similar if they satisfy ALL of:
    1. High pixel-wise accuracy (overall agreement)
    2. Same topological structure (regions and holes)
    3. Few disconnected disagreement regions (smooth boundaries)
    
    This multi-criteria approach ensures that similar diagrams truly
    represent the same underlying physics.
    
    Args:
        phase_diagram1 (np.ndarray): First binary phase diagram
        phase_diagram2 (np.ndarray): Second binary phase diagram
        acc_threshold (float): Minimum required pixel-wise accuracy (0-1)
        min_intersection_regions (int): Maximum allowed disagreement regions
        intersection_tol (int): Dilation radius for boundary tolerance (pixels)
        
    Returns:
        bool: True if diagrams are similar, False otherwise
    """
    # Criterion 1: Overall pixel-wise agreement
    acc = accuracy_score(phase_diagram1.flatten(), phase_diagram2.flatten())
    
    if(acc > acc_threshold):
        # Criterion 2: Same topological structure
        num_regions1, num_holes1 = topology_metrics(phase_diagram1)
        num_regions2, num_holes2 = topology_metrics(phase_diagram2)
        
        if(num_regions1 == num_regions2 and num_holes1 == num_holes2):
            # Criterion 3: Few disconnected disagreement regions
            # Many small disagreement regions indicate fundamentally different boundaries
            diff_diagram = tolerant_phase_disagreement(phase_diagram1, phase_diagram2, radius=intersection_tol)
            diff_num_regions = num_connected_regions(diff_diagram)
            
            if(diff_num_regions < min_intersection_regions):
                return True
    
    return False

def source_model_pruner(source_model_list, x_min = None, x_max = None, y_min = None, y_max = None, grid_size = 100,
                         acc_threshold = 0.9, min_intersection_regions = 3, intersection_tol = 5, return_index=False):
    """
    Select diverse source models from a collection based on phase diagram similarity.
    
    This function evaluates all source models on a common grid and selects a subset
    where each model produces a sufficiently different phase diagram. This ensures
    diversity in the transfer learning ensemble while avoiding redundant sources.
    
    The selection process is greedy: models are considered in order, and each is
    added to the selected set only if it differs from all previously selected models.
    
    Args:
        source_model_list (list): List of trained phase classification models
        x_min (float, optional): Minimum x-coordinate for evaluation grid
        x_max (float, optional): Maximum x-coordinate for evaluation grid
        y_min (float, optional): Minimum y-coordinate for evaluation grid
        y_max (float, optional): Maximum y-coordinate for evaluation grid
        grid_size (int): Number of grid points in each dimension
        acc_threshold (float): Similarity threshold for pixel-wise accuracy
        min_intersection_regions (int): Maximum disagreement regions for similarity
        intersection_tol (int): Boundary tolerance in pixels
        return_index (bool): If True, also return indices of selected models
        
    Returns:
        list or tuple: 
            - If return_index=False: List of selected diverse models
            - If return_index=True: (selected_models, selected_indices)
    """
    chosen_list_index = []
    chosen_models = []

    # Set default grid bounds if not specified
    if(x_min is None):
        warnings.warn("WARNING: No minimum value selected for interval -> x_min = 0")
        x_min = 0
    if(x_max is None):
        warnings.warn("WARNING: No maximum value selected for interval -> x_max = 1")
        x_max = 1
    if(y_min is None):
        y_min = x_min
    if(y_max is None):
        y_max = x_max

    # Create evaluation grid
    grid_points = get_grid(x_min, x_max, y_min = y_min, y_max=y_max, grid_size=grid_size)

    # Generate phase diagrams for all source models
    phase_diagram_list = [source_model.predict(grid_points).reshape(grid_size,grid_size) for source_model in source_model_list]

    # Convert to numpy arrays and ensure integer type
    for i, phase_diagram in enumerate(phase_diagram_list):
        phase_diagram = ensure_numpy(phase_diagram)
        phase_diagram_list[i] = phase_diagram.astype(int)
    
    # Greedy selection of diverse models
    for i, phase_diagram in enumerate(phase_diagram_list):
        to_save = True
        
        # Check similarity with all previously selected models
        for idx in chosen_list_index:
            if(i == idx):
                continue
                
            similar = similarity_checker(
                phase_diagram,
                phase_diagram_list[idx],
                acc_threshold=acc_threshold,
                min_intersection_regions=min_intersection_regions,
                intersection_tol=intersection_tol  
            )
            
            if(similar):
                # This model is too similar to an already selected one
                to_save = False
                break

        if(to_save):
            # This model is sufficiently different from all selected models
            chosen_list_index.append(i)
            chosen_models.append(source_model_list[i])
    
    if(return_index):
        return chosen_models, chosen_list_index
    else:
        return chosen_models
    

def source_diagram_pruner(phase_diagram_list, acc_threshold = 0.9, min_intersection_regions = 3, intersection_tol = 5):
    """
    Select diverse phase diagrams from a collection based on similarity.
    
    Similar to source_model_pruner but operates directly on phase diagram arrays
    rather than models. Useful when you have pre-computed phase diagrams or want
    to analyze diagram diversity without model evaluation.
    
    Args:
        phase_diagram_list (list): List of 2D binary phase diagram arrays
        acc_threshold (float): Similarity threshold for pixel-wise accuracy
        min_intersection_regions (int): Maximum disagreement regions for similarity
        intersection_tol (int): Boundary tolerance in pixels
        
    Returns:
        list: Selected diverse phase diagrams
    """
    chosen_list_index = []
    chosen_diagrams = []

    # Greedy selection process
    for i, phase_diagram in enumerate(phase_diagram_list):
        to_save = True
        
        # Check against all previously selected diagrams
        for idx in chosen_list_index:
            if(i == idx):
                continue
                
            similar = similarity_checker(
                phase_diagram,
                phase_diagram_list[idx],
                acc_threshold=acc_threshold,
                min_intersection_regions=min_intersection_regions,
                intersection_tol=intersection_tol  
            )
            
            if(similar):
                to_save = False
                break

        if(to_save):
            # Add to selected set
            chosen_list_index.append(i)
            chosen_diagrams.append(phase_diagram_list[i])
    
    return chosen_diagrams