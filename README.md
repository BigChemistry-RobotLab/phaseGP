# phaseGP

A Gaussian Process library for efficient phase diagram mapping using active learning and transfer learning. Intelligently select sampling points and leverage knowledge from related systems to minimize expensive simulations and experiments.

## Introduction

**phaseGP** is a Python library for implementing active learning loops for efficient phase diagram mapping using Gaussian Process (GP) classification with active learning and transfer learning capabilities. Based on the algorithms described in our paper "PhaseTransfer: A transfer learning framework for efficient phase diagram mapping", this library provides state-of-the-art methods for exploring phase spaces with minimal computational cost.

The library includes the PhaseTransfer method, a novel transfer learning approach that enables efficient knowledge reuse from related phase diagrams, significantly reducing the number of expensive evaluations needed to map new systems.

### Use Cases
phaseGP is designed for researchers and engineers working on:
- Materials science phase diagram exploration
- Chemical system phase behavior
- Soft matter self-assembly
- Colloidal phase transitions
- Any binary classification problem with expensive evaluations

### Library Structure
The library consists of five main modules:
- `models.py` - GP models and transfer learning implementations (PhaseGP, PhaseTransferGP and SKPhaseTransferGP)
- `priors.py` - Informed prior functions for transfer learning
- `source_pruner.py` - Source model selection algorithms
- `utils.py` - Utility functions for data processing and active learning
- `visualization.py` - Plotting and visualization tools

## Installation and Setup

### Requirements
phaseGP requires Python 3.7+ and the following dependencies:
```
numpy >= 1.19.0
torch >= 1.8.0
gpytorch >= 1.4.0
botorch >= 0.8.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
scipy >= 1.6.0
gudhi >= 3.4.0  # For topological analysis
```

### Installation
Install phaseGP using pip:
```bash
pip install git+https://github.com/BigChemistry-RobotLab/phaseGP.git
```

Or install from source:
```bash
git clone https://github.com/BigChemistry-RobotLab/phaseGP.git
cd phaseGP
pip install -e .
```

### Basic Import
```python
import phaseGP
from phaseGP.models import PhaseGP, PhaseTransferGP
from phaseGP.visualization import model_diagram_plot
from phaseGP.utils import brute_sample_new_points, set_seeds
```

## Quick Start Guide

### Basic Phase Diagram Mapping
Here's a simple example of using phaseGP for phase diagram exploration:

```python
# Import required modules
import torch
from phaseGP.models import PhaseGP
from phaseGP.visualization import model_diagram_plot
from phaseGP.utils import set_seeds

# Set random seed for reproducibility
set_seeds(42)

# Generate initial training data
n_initial = 50
X_train = torch.rand(n_initial, 2)  # Random points in [0,1]²
y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).float()  # Simple linear boundary

# Create and train GP model
model = PhaseGP(
    train_x=X_train,
    min_scale=[0, 0],
    max_scale=[1, 1],
)

model.fit(X_train, y_train)

# Visualize the phase diagram
fig, ax = model_diagram_plot(
    model, 
    plot_type="phase",
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    title="Phase Diagram",
    phase_labels=["Phase A", "Phase B"],
    plot_boundary=True
)
```

### Active Learning Loop
Implement an active learning strategy to efficiently explore the phase space:

```python
import torch
from phaseGP.models import PhaseGP
from phaseGP.visualization import model_diagram_plot
from phaseGP.utils import set_seeds, get_grid, brute_sample_new_points

# Set random seed for reproducibility
set_seeds(42)

# Active learning loop
n_iterations = 20
n_samples_per_iter = 1

# Generate initial training data
n_initial = 10
X_train = torch.rand(n_initial, 2)  # Random points in [0,1]²
y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).float()  # Simple linear boundary

# Generate candidate points
candidates = get_grid(0, 1, grid_size=30)
for iteration in range(n_iterations):
    model = PhaseGP(
        train_x=X_train,
        min_scale=[0, 0],
        max_scale=[1, 1],
        )
    # Select new points using acquisition function
    # Note: For problems with more than 3 dimensions, it is recommended to use
    # gradient_sample_new_points instead of brute_sample_new_points for better efficiency
    new_points = brute_sample_new_points(
        model, 
        candidates, 
        X_train,
        n_sample=n_samples_per_iter
    )
    
    # Evaluate true labels (replace with your expensive function)
    new_labels = (new_points[:, 0] + new_points[:, 1] > 1.0).float()
    
    # Add to training set
    X_train = torch.cat([X_train, new_points])
    y_train = torch.cat([y_train, new_labels])
    
    # Retrain model
    model.fit(X_train, y_train)
    
    print(f"Iteration {iteration+1}: Added {len(new_points)} points")

# Visualize the phase diagram
fig, ax = model_diagram_plot(
    model, 
    plot_type="phase",
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    title="Phase Diagram",
    phase_labels=["Phase A", "Phase B"],
    plot_boundary=True
)
```

### Transfer Learning Example
Leverage knowledge from related phase diagrams:

```python
# Assume we have pre-trained source models
import torch
from phaseGP.models import PhaseTransferGP
from phaseGP.visualization import model_diagram_plot
from phaseGP.utils import set_seeds, get_grid, brute_sample_new_points

# Set random seed for reproducibility
set_seeds(42)

source_models = [model1, model2, model3]  # Previously trained PhaseGP models

# Generate initial training data
n_initial = 50
X_train_new = torch.rand(n_initial, 2)  # Random points in [0,1]²
# Simple linear boundary
y_train_new = (X_train_new[:, 0] + X_train_new[:, 1] > 1.2).float()  

# Prune similar source models
from phaseGP.source_pruner import source_model_pruner
diverse_sources = source_model_pruner(
    source_models,
    x_min=0, x_max=1
)

# Create transfer learning model
tl_model = PhaseTransferGP(
    source_model_list=diverse_sources,
    train_x=X_train_new,
    min_scale=[0, 0],
    max_scale=[1, 1]
)

# Train with transfer learning
tl_model.fit(X_train_new, y_train_new)

# The model now benefits from source knowledge!

# Visualize the phase diagram
fig, ax = model_diagram_plot(
    tl_model, 
    plot_type="phase",
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    title="Phase Diagram",
    phase_labels=["Phase A", "Phase B"],
    plot_boundary=True
)
```
## GPU support
For utilizing a GPU, set the class device parameter to your desired cuda device.
Ensure that your training data is in the proper device.

```python
model = PhaseGP(
        train_x=X_train,
        min_scale=[0, 0],
        max_scale=[1, 1],
        device = "cuda"
        )
```
or

```python
tl_model = PhaseTransferGP(
    source_model_list=diverse_sources,
    train_x=X_train_new,
    min_scale=[0, 0],
    max_scale=[1, 1],
    device = "cuda"
)
```
For a more detailed example refer to the GPU sine benchmark notebook under the folder [benchmarks/demos](benchmarks/demos)

## Documentation & Extended Examples

For detailed API reference, please refer to the [documentation PDF](phaseGP_api_reference.pdf).

For detailed examples refer to the notebooks [Sine Wave Benchmak](benchmarks/demos/sine_benchmark.ipynb), [3-Component Biological Condensate BenchMark](benchmarks/demos/biological_condensate_benchmark.ipynb) and [Supramolecular Copolymerization Benchmark](benchmarks/demos/supramolecular_copolymerization_benchmark.ipynb) inside the folder [benchmarks/demos](benchmarks/demos)

## Paper Reproducibility

To recalculate the error curves for all models (except GP-ECA), run the following script "calculate_error_curves.sh" while inside the folder [benchmarks/paper_reproducibility]


For visualizing the results, use the notebook [error_curve_visualizer.ipynb](benchmarks/paper_reproducibility/error_curve_visualizer.ipynb) under the location [benchmarks/paper_reproducibility](benchmarks/paper_reproducibility).

The data from our experimental results is available in [benchmarks/paper_reproducibility/experimental_data](benchmarks/paper_reproducibility/experimental_data).

## Citation

If you use phaseGP in your research, please cite:

```bibtex
[Citation to be added]
```
