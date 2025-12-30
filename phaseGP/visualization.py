"""
Visualization Tools for Phase Diagram Analysis

This module provides comprehensive visualization functions for phase diagrams,
probability maps, and acquisition functions. It supports both model-based
visualization (directly from GP models) and data-based visualization (from
pre-computed arrays).

The visualizations are designed to provide intuitive understanding of:
- Phase boundaries and transitions
- Model uncertainty and confidence
- Active learning acquisition targets
- Transfer learning source contributions

Author: Eduardo Gonzalez Garcia (e.gonzalez.garcia@tue.nl)
Version: 0.1.0
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
import warnings
import matplotlib

from .utils import get_grid, ensure_numpy

__all__ = ["model_diagram_plot", "phase_diagram_plot", "phase_diagram_probability_plot", "phase_acquisition_plot"]
# =============================================================================
# Model-Based Visualization
# =============================================================================

def model_diagram_plot(model, plot_type="phase", x_min=None, x_max=None, y_min=None, y_max=None, 
                       grid_size=100, sampled_points=None, phase_labels=None, title=None, 
                       xlabel="Parameter 1", ylabel="Parameter 2", figsize=(7,6), cmap=None, 
                       plot_boundary=False):
    """
    Generate various 2D plots from a trained GP model.
    
    This is the main interface for model visualization, supporting multiple
    plot types to analyze different aspects of the model's predictions and
    behavior. It automatically evaluates the model on a grid and creates
    the appropriate visualization.
    
    Plot types:
    - 'phase': Binary phase classification with clear boundaries
    - 'probability': Continuous probability map showing model confidence
    - 'acquisition': Acquisition function values for active learning
    
    Args:
        model: Trained GP model with predict/predict_proba/acquisition methods
        plot_type (str): Type of plot ('phase', 'probability', 'acquisition')
        x_min (float, optional): Minimum x-coordinate for plot domain
        x_max (float, optional): Maximum x-coordinate for plot domain
        y_min (float, optional): Minimum y-coordinate for plot domain
        y_max (float, optional): Maximum y-coordinate for plot domain
        grid_size (int): Number of grid points in each dimension
        sampled_points (torch.Tensor, optional): Already sampled points (for acquisition)
        phase_labels (list, optional): Names for phase 0 and phase 1
        title (str, optional): Plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (tuple): Figure size as (width, height) in inches
        cmap (str, optional): Colormap name (defaults based on plot_type)
        plot_boundary (bool): Whether to show phase boundaries as contour lines
        
    Returns:
        tuple: (fig, ax) - Matplotlib figure and axes objects
        
    Warnings:
        Issues warning if plot_type is not recognized
    """
    # Set default colormaps for each plot type
    if(cmap is None):
        if(plot_type=="phase"):
            #cmap='coolwarm'  # Blue/red for binary phases
            cmap = matplotlib.colors.ListedColormap(["#be96c5ff","#fdc684ff"])
        elif(plot_type=="probability"):
            cmap="viridis"  # Sequential colormap for probabilities
        elif(plot_type=="acquisition"):
            cmap='plasma'  # High contrast for acquisition values
        else:
            print("Wrong plot type")
            return
    
    # Set default domain bounds if not specified
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

    # Generate evaluation grid
    grid_points, x_coords, y_coords = get_grid(
        x_min, x_max, y_min=y_min, y_max=y_max, 
        grid_size=grid_size, return_coordinates=True, device=model.device
    )

    # Generate appropriate plot based on type
    if(plot_type=="phase"):
        # Binary phase classification plot
        phase_diagram = model.predict(grid_points).reshape(grid_size, grid_size).T
        phase_diagram = ensure_numpy(phase_diagram)
        phase_diagram = phase_diagram.astype(int)

        return phase_diagram_plot(
            phase_diagram, x_coords, y_coords,
            phase_labels=phase_labels, title=title, xlabel=xlabel, ylabel=ylabel, 
            figsize=figsize, cmap=cmap, plot_boundary=plot_boundary
        )

    elif(plot_type=="probability"):
        # Probability heatmap with optional decision boundary
        phase_probability = model.predict_proba(grid_points)
        
        # Handle sklearn models that return 2D array
        if(phase_probability.ndim == 2):
            phase_probability = phase_probability[:,1]
            
        phase_probability = phase_probability.reshape(grid_size, grid_size).T
        phase_probability = ensure_numpy(phase_probability)

        return phase_diagram_probability_plot(
            phase_probability, x_coords, y_coords, title=title, xlabel=xlabel, 
            ylabel=ylabel, figsize=figsize, cmap=cmap, plot_boundary=plot_boundary
        )
    
    elif(plot_type=="acquisition"):
        # Acquisition function visualization for active learning
        acq_values = model.acquisition(
            grid_points, sampled_points=sampled_points
        ).reshape(grid_size, grid_size).T
        acq_values = ensure_numpy(acq_values)

        return phase_acquisition_plot(
            acq_values, x_coords, y_coords, title=title, xlabel=xlabel, 
            ylabel=ylabel, figsize=(7, 6), cmap=cmap, show_maximum=True
        )
    else:
        warnings.warn("Wrong plot type")
        return

# =============================================================================
# Core Visualization Functions
# =============================================================================

def phase_diagram_plot(phase_diagram, x_coords=None, y_coords=None, phase_labels=None, 
                      title=None, xlabel="Parameter 1", ylabel="Parameter 2", 
                      figsize=(7,6), cmap=matplotlib.colors.ListedColormap(["#be96c5ff","#fdc684ff"]), plot_boundary=False):
    """
    Plot a phase diagram with optional contour lines at phase boundaries.
    
    Creates a filled contour plot of the phase diagram, clearly showing
    distinct phases and their boundaries. Supports both binary and multi-phase
    systems with appropriate level selection.
    
    Args:
        phase_diagram (np.ndarray): 2D array of phase indices, shape (n_y, n_x)
        x_coords (array-like, optional): X-axis coordinate values
        y_coords (array-like, optional): Y-axis coordinate values
        phase_labels (list, optional): Names for each phase
        title (str, optional): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        figsize (tuple): Figure size (width, height) in inches
        cmap (str or colormap): Colormap for phases
        plot_boundary (bool): Whether to draw black contour lines at boundaries
    
    Returns:
        tuple: (fig, ax) - Matplotlib figure and axes objects
    """
    if title is None: 
        title = "Phase Diagram"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default coordinates if not provided
    if x_coords is None:
        x_coords = np.arange(phase_diagram.shape[1])
    if y_coords is None:
        y_coords = np.arange(phase_diagram.shape[0])
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Determine contour levels based on number of phases
    unique_phases = np.unique(phase_diagram)
    if len(unique_phases) == 2:  # Binary phase diagram
        # Levels at -0.5, 0.5, 1.5 create boundaries at 0 and 1
        levels = [-0.5, 0.5, 1.5]
    else:
        # For multi-phase, create boundaries around each phase
        levels = np.arange(min(unique_phases) - 0.5, max(unique_phases) + 1.5, 1)
    
    # Create filled contour plot
    im = ax.contourf(X, Y, phase_diagram, levels=levels, cmap=cmap, alpha=1)
    
    # Add black contour lines at phase boundaries if requested
    if(plot_boundary):
        contours = ax.contour(X, Y, phase_diagram, levels=levels, 
                         colors='black', linewidths=2, alpha=1)
    
    # Add colorbar with phase labels
    cbar = plt.colorbar(im, ax=ax)
    
    # Set colorbar ticks to show only actual phases
    cbar.set_ticks(unique_phases)
    if phase_labels:
        cbar.set_ticklabels(phase_labels[:len(unique_phases)], fontsize=10)
    else:
        cbar.set_ticklabels([f'Phase {int(p)}' for p in unique_phases], fontsize=10)
    
    # Set labels and title with specified font sizes
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=20)
    
    plt.tight_layout()
    return fig, ax


def phase_diagram_probability_plot(phase_probabilities, x_coords=None, y_coords=None,
                                 title=None, xlabel="Parameter 1", 
                                 ylabel="Parameter 2", figsize=(7,6), cmap='viridis', 
                                 plot_boundary=True):
    """
    Plot phase probabilities with decision boundary at p=0.5.
    
    Creates a continuous heatmap showing the model's predicted probability
    of phase 1, with contour lines indicating confidence levels. The red
    decision boundary at p=0.5 clearly shows the predicted phase transition.
    
    Args:
        phase_probabilities (np.ndarray): 2D array of probabilities [0,1], shape (n_y, n_x)
        x_coords (array-like, optional): X-axis coordinate values
        y_coords (array-like, optional): Y-axis coordinate values
        title (str, optional): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        figsize (tuple): Figure size (width, height) in inches
        cmap (str or colormap): Colormap for probability values
        plot_boundary (bool): Whether to highlight the p=0.5 decision boundary
    
    Returns:
        tuple: (fig, ax) - Matplotlib figure and axes objects
    """
    if title is None: 
        title = "Phase Diagram Probabilities"

    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default coordinates if not provided
    if x_coords is None:
        x_coords = np.arange(phase_probabilities.shape[1])
    if y_coords is None:
        y_coords = np.arange(phase_probabilities.shape[0])
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Plot probability contours with 11 levels from 0 to 1
    levels = np.linspace(0, 1, 11)
    im = ax.contourf(X, Y, phase_probabilities, levels=levels, cmap=cmap, alpha=1)
    
    # Add gray contour lines at key probability levels
    contours = ax.contour(X, Y, phase_probabilities, levels=levels[::4],  # Every 4th level
                         colors='gray', linewidths=0.5, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    if(plot_boundary):
        # Highlight decision boundary at p=0.5 in red
        decision_boundary = ax.contour(X, Y, phase_probabilities, levels=[0.5], 
                                  colors='red', linewidths=2.5)
        ax.clabel(decision_boundary, inline=True, fontsize=15, fmt='%.1f', 
              colors='red')
    
    # Add colorbar showing probability scale
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=20)
    
    # Add legend for decision boundary
    if(plot_boundary):
        red_line = plt.Line2D([0], [0], color='red', linewidth=2.5, label='Phase Boundary')
        ax.legend(handles=[red_line], loc='upper right', fontsize=15)
    
    plt.tight_layout()
    return fig, ax


def phase_acquisition_plot(acquisition_values, x_coords=None, y_coords=None,
                          title=None, xlabel="Parameter 1", 
                          ylabel="Parameter 2", figsize=(7, 6), cmap='plasma',
                          show_maximum=True):
    """
    Plot acquisition function values for active learning visualization.
    
    Shows where the model suggests sampling next, with higher values indicating
    more informative locations. The acquisition function  combines
    uncertainty (exploration) with proximity to the decision boundary (exploitation).
    
    Args:
        acquisition_values (np.ndarray): 2D array of acquisition values, shape (n_y, n_x)
        x_coords (array-like, optional): X-axis coordinate values
        y_coords (array-like, optional): Y-axis coordinate values
        title (str, optional): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        figsize (tuple): Figure size (width, height) in inches
        cmap (str or colormap): Colormap for acquisition values
        show_maximum (bool): Whether to mark the maximum acquisition point
    
    Returns:
        tuple: (fig, ax) - Matplotlib figure and axes objects
    """
    if title is None: 
        title = "Acquisition Function"

    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default coordinates if not provided
    if x_coords is None:
        x_coords = np.arange(acquisition_values.shape[1])
    if y_coords is None:
        y_coords = np.arange(acquisition_values.shape[0])
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Plot acquisition function with 20 contour levels
    im = ax.contourf(X, Y, acquisition_values, levels=20, cmap=cmap, alpha=1)
    
    # Add subtle contour lines for better readability
    contours = ax.contour(X, Y, acquisition_values, levels=10, 
                         colors='black', linewidths=0.6, alpha=0.7)
    
    # Mark the point with maximum acquisition value
    if show_maximum:
        # Find location of maximum
        max_idx = np.unravel_index(np.argmax(acquisition_values), acquisition_values.shape)
        max_x = x_coords[max_idx[1]]
        max_y = y_coords[max_idx[0]]
        
        # Plot as red star with white edge for visibility
        ax.plot(max_x, max_y, 'r*', markersize=15, markeredgecolor='white', 
                markeredgewidth=1, label=f'Max Acquisition')
        ax.legend(fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Acquisition Value', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=20)
    
    plt.tight_layout()
    return fig, ax