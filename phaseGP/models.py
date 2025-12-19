#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPyTorch Implementation of Phase Diagram Mapping with Active Learning and Transfer Learning

This module implements Gaussian Process models for binary phase classification.

1. PhaseGP: Base GP model with variational inference for phase classification
2. PhaseTransferGP: Transfer learning model using multiple source models with adaptive weighting
3. SKPhaseTransferGP: Transfer learning with scikit-learn compatible source models

Key Features:
- Active learning through uncertainty-based acquisition functions
- Transfer learning with automatic source model weighting

Author: Eduardo Gonzalez Garcia (e.gonzalez.garcia@tue.nl)
Version: 0.1.0
"""
import warnings
import torch
import gpytorch

from .priors import MultiGPMeanModule, SKMultiGPMeanModule
from .utils import ensure_tensor, get_base_kernel, scaler

# =============================================================================
# Unified GP Model
# =============================================================================


class PhaseGP(gpytorch.models.ApproximateGP):
    """
    Variational Gaussian Process model for binary phase classification.
    
    This class implements a GP classifier using variational inference, suitable for
    mapping phase diagrams where the output represents different phases (0 or 1).
    The model uses inducing points for computational efficiency and includes
    regularization through prior distributions on kernel hyperparameters.
    
    Attributes:
        learning_rate (float): Learning rate for model optimization
        lengthscale (float): Initial/target lengthscale for the kernel
        training_iterations (int): Number of optimization iterations
        lengthscale_interval (tuple): Prior bounds for lengthscale parameter
        outputscale_interval (tuple): Prior bounds for outputscale parameter
        min_scale (torch.Tensor): Minimum values for input normalization
        max_scale (torch.Tensor): Maximum values for input normalization
        inducing_points_size (int): Number of inducing points
        mean_module: GP mean function (constant by default)
        covar_module: GP covariance function with specified kernel
        likelihood: Bernoulli likelihood for binary classification
        device: Device where the model is stored (cpu or cuda)
    """
    def __init__(
            self,
            train_x,
            min_scale = None,
            max_scale = None,
            kernel_choice='matern32',
            lengthscale=0.3,
            learning_rate=0.1,
            training_iterations = 120,
            lengthscale_interval = (0.2,0.3),
            outputscale_interval = (1.0,4.0),
            device = "cpu"
            ):
        """
        Initialize the Phase GP model.
        """
        self.learning_rate = learning_rate
        # Store lengthscale for kernel initialization
        self.lengthscale  = lengthscale
        self.training_iterations = training_iterations
        self.lengthscale_interval = lengthscale_interval
        self.outputscale_interval = outputscale_interval
        self.device = device
        
        # Set up input scaling parameters for normalization to [0,1]
        if(min_scale is None):
            warnings.warn("WARNING: No minimum value selected for interval -> min_scale = 0")
            self.min_scale = torch.zeros(train_x.size(1)).to(device)
        else:
            self.min_scale = torch.tensor(min_scale).to(device)
            if self.min_scale.dim() == 0:
                self.min_scale = self.min_scale.repeat(train_x.size(1))

        if(max_scale is None):
            warnings.warn("WARNING: No maximum value selected for interval -> max_scale = 1")
            self.max_scale = torch.ones(train_x.size(1)).to(device)
        else:
            self.max_scale = torch.tensor(max_scale).to(device)
            if self.max_scale.dim() == 0:
                self.max_scale = self.max_scale.repeat(train_x.size(1))
        
        # Initialize variational GP components
        train_x = ensure_tensor(train_x, device=self.device)
        train_x = train_x.float()
        self.inducing_points_size = train_x.size(0)

        # Set up variational distribution using Cholesky decomposition for efficiency
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(self.inducing_points_size)
        
        # Create variational strategy with scaled inducing points
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, 
            scaler(train_x, self.min_scale, self.max_scale).float(), 
            variational_distribution
        )
        
        super(PhaseGP, self).__init__(variational_strategy)
        
        # Initialize mean function (constant mean)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Set up kernel with smoothed box priors for regularization        
        lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(
            self.lengthscale_interval[0],
            self.lengthscale_interval[1]
        )
        outputscale_prior = gpytorch.priors.SmoothedBoxPrior(
            self.outputscale_interval[0], 
            self.outputscale_interval[1]
        )

        # Create base kernel with specified type and wrap with scale kernel
        base_kernel = get_base_kernel(
            kernel_choice, 
            lengthscale=self.lengthscale, 
            lengthscale_prior=lengthscale_prior
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, 
            outputscale_prior=outputscale_prior
        )
        #Ensure that the model is in the proper device
        self.to(self.device)
    
    def _move_to_device(self, device):
        self.device = device
        self.min_scale = self.min_scale.to(device)
        self.max_scale = self.max_scale.to(device)
        self.likelihood.to(device)
        self.to(device)

    def forward(self, x):
        """
        Forward pass through the GP model.
        
        Computes the GP prior distribution at input locations. Note that inputs
        are scaled to [0,1].
        
        Args:
            x (torch.Tensor): Input locations of shape (n, d)
            
        Returns:
            gpytorch.distributions.MultivariateNormal: GP prior distribution
        """
        with torch.no_grad():
            x = x.clone() # Create a copy to avoid modifying original input
            # Scale non-inducing point inputs to [0,1] range
            x[self.inducing_points_size:] = scaler(
                x[self.inducing_points_size:], 
                self.min_scale, 
                self.max_scale
            )
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self, train_x, train_y, epsilon=0.05, verbose=False):
        """
        Train the GP model on binary classification data.
        
        Uses variational inference to fit the model parameters. The binary labels
        are transformed to continuous latent values using a logit transformation
        with epsilon regularization to avoid numerical issues.
        
        Args:
            train_x (torch.Tensor): Training inputs of shape (n, d)
            train_y (torch.Tensor): Binary training labels (0 or 1) of shape (n,)
            epsilon (float): Regularization parameter for logit transformation
            verbose (bool): Whether to print training progress
        """
        train_x = ensure_tensor(train_x, device=self.device)
        train_y = ensure_tensor(train_y, device=self.device)

        train_x = train_x.float()
        train_y = train_y.flatten()

        # Transform binary labels to continuous latent values using regularized logit
        # This maps [0,1] to ~[-3, 3] with epsilon=0.05 preventing extreme values
        latent_y = torch.log((train_y + epsilon) / (1 - train_y + epsilon))

        # Train the model using variational inference
        self, self.likelihood = train_gp_model(
            self,
            train_x,
            latent_y,
            learning_rate=self.learning_rate,
            training_iterations=self.training_iterations,
            verbose=verbose,
            device = self.device
        )
        return
    
    def predict(self, x):
        """
        Predict binary phase labels for input points.
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Binary predictions (0 or 1) of shape (n,) in the model's device
        """
        y_pred = self.predict_proba(x)
        return (y_pred > 0.5).int()
    
    def predict_proba(self, x):
        """
        Predict phase probabilities for input points.
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Probability of phase 1 for each point, shape (n,) in the model's device
        """
        x = ensure_tensor(x, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self(x)
            y_pred = self.likelihood(f_pred).mean

        return y_pred
    
    def acquisition(self, x, sampled_points=None, epsilon=0.05, vanilla_acq=True,distance_acq=False ,requires_grad = False):
        """
        Compute acquisition function values for active learning.
        
        The acquisition function identifies points with high uncertainty near the
        phase boundary. It combines predictive variance with proximity to the
        decision boundary (where mean prediction is near 0).
        
        Args:
            x (torch.Tensor): Candidate points for evaluation, shape (n, d)
            sampled_points (torch.Tensor, optional): unused in base model
            epsilon (float): Small value to prevent division by zero
            vanilla_acq (bool): unused in base model
            distance_acq (bool): unused in base model
            requires_grad (bool): If True, computes gradients of the acquisition for gradient based optimization
        Returns:
            torch.Tensor: Acquisition values for each candidate point, shape (n,)
        """
        x = ensure_tensor(x, device=self.device)
        # Expand single point to batch dimension
        if(x.ndim == 1):
            x = x.unsqueeze(0)

        if requires_grad:
            with gpytorch.settings.fast_pred_var():
                target_f_pred = self(x)
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                target_f_pred = self(x)

        # Acquisition favors high variance points near decision boundary
        # High when: variance is large AND mean is close to 0 (uncertain boundary)
        target_boundary_term = torch.sqrt(target_f_pred.variance) / (torch.abs(target_f_pred.mean) + epsilon)
        
        return target_boundary_term
    
class PhaseTransferGP(torch.nn.Module):
    """
    Transfer Learning GP model for phase classification with multiple source models.
    
    This model implements transfer learning by combining predictions from multiple
    source models (previously trained on related phase diagrams) with a target model
    trained on the current task. Each source model's contribution is weighted by an
    auxiliary GP that learns the reliability of that source for different regions
    of the input space.
    
    Key Features:
    - Adaptive weighting of source models based on their local accuracy
    - Power transformation for emphasizing confident source predictions
    - Exploration-exploitation balance in acquisition function
    - Support for both GP and non-GP source models
    
    Attributes:
        source_model_list (list): Pre-trained source models for transfer
        weight_model_list (list): GP models learning source reliability weights
        target_model (PhaseGP): Target GP model for current task
        prior_aggregation (str): Method for combining source predictions
        max_adaptive_power (float): Maximum exponent for weight adaptation
        explorative_threshold (float): Threshold for exploration vs exploitation
    """

    def __init__(
            self,
            source_model_list,
            train_x,
            min_scale = None,
            max_scale = None,
            prior_aggregation = "linear",
            max_adaptive_power=5,
            explorative_threshold = 0.4,
            kernel_choice='matern32',
            lengthscale=0.3,
            learning_rate=0.1,
            training_iterations = 120,
            lengthscale_interval = (0.2,0.3),
            outputscale_interval = (1.0,4.0),
            device = "cpu"
            ):
        """
        Initialize the transfer learning GP model.
        
        Args:
            source_model_list (list): List of pre-trained source models
            train_x (torch.Tensor): Initial training inputs
            min_scale (array-like, optional): Minimum values for input scaling
            max_scale (array-like, optional): Maximum values for input scaling
            prior_aggregation (str): Method for aggregating priors ('linear' or 'highest')
            max_adaptive_power (float): Maximum power for adaptive weighting
            explorative_threshold (float): Threshold for exploration (0 to 1)
            kernel_choice (str): Kernel type for GP models
            lengthscale (float): Initial kernel lengthscale
            learning_rate (float): Optimization learning rate
            training_iterations (int): Number of training iterations
            lengthscale_interval (tuple): Prior bounds for lengthscale
            outputscale_interval (tuple): Prior bounds for outputscale
            device: Device where the model is stored (cpu or cuda)
        """
        super().__init__()
        
        if(source_model_list is None):
            raise ValueError('Source model list for transfer learning is None')

        # Store configuration parameters
        self.source_model_list = source_model_list
        self.prior_aggregation = prior_aggregation
        self.max_adaptive_power = max_adaptive_power
        self.kernel_choice = kernel_choice
        self.lengthscale = lengthscale
        self.learning_rate = learning_rate
        self.training_iterations = training_iterations
        self.lengthscale_interval = lengthscale_interval
        self.outputscale_interval = outputscale_interval
        self.explorative_threshold = explorative_threshold
        self.device = device
        # Set up input scaling parameters
        if(min_scale is None):
            warnings.warn("WARNING: No minimum value selected for interval -> min_scale = 0")
            self.min_scale = torch.zeros(train_x.size(1)).to(self.device)
        else:
            self.min_scale = torch.tensor(min_scale).to(self.device)
            if self.min_scale.dim() == 0:
                self.min_scale = self.min_scale.repeat(train_x.size(1))

        if(max_scale is None):
            warnings.warn("WARNING: No maximum value selected for interval -> max_scale = 1")
            self.max_scale = torch.ones(train_x.size(1)).to(self.device)
        else:
            self.max_scale = torch.tensor(max_scale).to(self.device)
            if self.max_scale.dim() == 0:
                self.max_scale = self.max_scale.repeat(train_x.size(1))
        
        #Move source models to the correct device
        for source_model in self.source_model_list:
            if hasattr(source_model, '_move_to_device'):
                source_model._move_to_device(self.device)

        # Create weight models for each source model
        # These learn how reliable each source is at different input locations
        n_models = len(self.source_model_list)
        self.weight_model_list = [PhaseGP(
            train_x,
            min_scale = min_scale,
            max_scale = max_scale,
            kernel_choice=kernel_choice,
            lengthscale=lengthscale,
            learning_rate=learning_rate,
            training_iterations = training_iterations,
            lengthscale_interval = lengthscale_interval,
            outputscale_interval = outputscale_interval,
            device = self.device
        ).to(device) for i in range(n_models)]
        
        # Create target model for the current task
        self.target_model = PhaseGP(
            train_x,
            min_scale = min_scale,
            max_scale = max_scale,
            kernel_choice=kernel_choice,
            lengthscale=lengthscale,
            learning_rate=learning_rate,
            training_iterations = training_iterations,
            lengthscale_interval = lengthscale_interval,
            outputscale_interval = outputscale_interval,
            device = self.device
        ).to(device)

    def forward(self, x):
        """
        Forward pass combining source and target model predictions.
        
        Computes weighted combination of source and target predictions where:
        1. Each source model makes a prediction
        2. Weight models determine source reliability at x
        3. The most reliable source is selected
        4. Final prediction blends source and target based on adaptive weighting
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            tuple: (y_pred_mean, f_pred_mean, f_pred_var)
                - y_pred_mean: Combined probability predictions
                - f_pred_mean: Combined latent function mean
                - f_pred_var: Combined latent function variance
        """
        # Get target model predictions
        target_f_pred = self.target_model(x)
        target_y_pred = self.target_model.likelihood(target_f_pred)

        # Collect predictions and weights from all source models
        weight_list = []
        source_y_mean_list = []
        source_f_var_list = []
        source_f_mean_list = []
        
        for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
            # Get weight for this source model
            latent_weight = weight_model(x)
            weight = weight_model.likelihood(latent_weight).mean
            weight_list.append(weight)

            # Get source model predictions
            source_f_pred = source_model(x)
            source_y_pred = source_model.likelihood(source_f_pred)
            source_f_mean_list.append(source_f_pred.mean)
            source_f_var_list.append(source_f_pred.variance)
            source_y_mean_list.append(source_y_pred.mean)
        
        # Stack all predictions for vectorized operations
        weight_list = torch.stack(weight_list)
        source_y_mean_list = torch.stack(source_y_mean_list)
        source_f_mean_list = torch.stack(source_f_mean_list)
        source_f_var_list = torch.stack(source_f_var_list)

        # Select the best source model for each input point
        cols = torch.arange(x.size(0))
        max_indices = torch.argmax(weight_list, dim=0)

        weight = weight_list[max_indices, cols]
        source_y_mean = source_y_mean_list[max_indices, cols]
        source_f_mean = source_f_mean_list[max_indices, cols]
        source_f_var = source_f_var_list[max_indices, cols]

        # Apply adaptive power transformation to weights
        # Higher power when source is confident (far from 0.5 probability)
        # This emphasizes reliable source predictions
        power = torch.log(torch.e + 2*(torch.e**self.max_adaptive_power-torch.e)*torch.abs(source_y_mean - 0.5))

        # Apply power transformation to weight
        weight = weight**power
        
        # Compute weighted combination of source and target predictions
        y_pred_mean = weight*source_y_mean + (1-weight)*target_y_pred.mean
        f_pred_var = weight*source_f_var + (1-weight)*target_f_pred.variance
        f_pred_mean = weight*source_f_mean + (1-weight)*target_f_pred.mean

        return y_pred_mean, f_pred_mean, f_pred_var
    
    def fit(self, train_x, train_y, epsilon=0.05, verbose=False):
        """
        Train the transfer learning model.
        
        Training process:
        1. Train weight models to predict source model accuracy
        2. Initialize target model with informed prior from weighted sources
        3. Train target model on current task data
        
        Args:
            train_x (torch.Tensor): Training inputs of shape (n, d)
            train_y (torch.Tensor): Binary training labels of shape (n,)
            epsilon (float): Regularization parameter for numerical stability
            verbose (bool): Whether to print training progress
        """
        train_x = ensure_tensor(train_x, device=self.device)
        train_y = ensure_tensor(train_y, device=self.device)

        train_x = train_x.float()
        train_y = train_y.flatten()
        i = 0
        
        # Train weight models for each source
        for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
            i += 1
            if(verbose):
                print(f"Training weight model {i}/{len(self.source_model_list)}")
            
            # Create labels for weight model: 1 if source agrees with true label, 0 otherwise
            source_phases = source_model.predict(train_x)
            auxiliary_y = torch.eq(source_phases, train_y).int()
            
            # Train weight model to predict source reliability
            weight_model.fit(train_x, auxiliary_y, epsilon=epsilon)

        # Initialize target model with informed prior combining weighted sources
        self.target_model.mean_module = MultiGPMeanModule(
            self.source_model_list,
            self.weight_model_list,
            prior_aggregation=self.prior_aggregation,
            device = self.device
        )
        
        # Train target model with informed prior
        self.target_model.fit(train_x, train_y, epsilon=epsilon)
    
    def predict(self, x):
        """
        Predict binary phase labels.
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Binary predictions (0 or 1) of shape (n,)
        """
        y_pred = self.predict_proba(x)
        return (y_pred > 0.5).int()
    
    def predict_proba(self, x):
        """
        Predict phase probabilities using weighted combination.
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Probability of phase 1 for each point, shape (n,) in the model's device
        """
        x = ensure_tensor(x, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred_mean, f_pred_mean, f_pred_var = self(x)

        return y_pred_mean
    
    def get_weight(self, x, requires_grad=False):
        """
        Get the weight of the best source model at each input point.
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Weight values for best source at each point, shape (n,) in the model's device
        """
        weight_list = []
        if requires_grad:
            with gpytorch.settings.fast_pred_var():
                for weight_model in self.weight_model_list:
                    latent_weight = weight_model(x)
                    weight = weight_model.likelihood(latent_weight).mean
                    weight_list.append(weight)
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for weight_model in self.weight_model_list:
                    latent_weight = weight_model(x)
                    weight = weight_model.likelihood(latent_weight).mean
                    weight_list.append(weight)

        weight_list = torch.stack(weight_list)
        cols = torch.arange(x.size(0))
        max_indices = torch.argmax(weight_list, dim=0)

        weight = weight_list[max_indices, cols]
        return weight
    
    def acquisition(self, x, sampled_points=None, epsilon=0.05, vanilla_acq=False, distance_acq=True, requires_grad=False):
        """
        Compute acquisition function for transfer learning active learning.
        
        The acquisition function balances:
        - Exploitation: High uncertainty near phase boundary (from target model)
        - Exploration: High variance and distance from sampled points
        
        The balance is controlled by source model reliability:
        - Low reliability (weight < threshold): Fall back to standard exploration/exploitation
        acquisition function
        - High reliability: Focus on exploration
        
        Args:
            x (torch.Tensor): Candidate points for evaluation, shape (n, d)
            sampled_points (torch.Tensor): Already sampled points, shape (m, d)
            epsilon (float): Small value to prevent division by zero
            vanilla_acq (bool): If True, use only standard exploration/exploitation
            acquisition function
            distance_acq (bool): If True, include euclidean distance in exploration
            requires_grad (bool): If True, computes gradients of the acquisition for gradient based optimization
        Returns:
            torch.Tensor: Acquisition values for each candidate, shape (n,)
        """
        x = ensure_tensor(x, device=self.device)
        if(distance_acq):
            sampled_points = ensure_tensor(sampled_points, device=self.device)
        
        # Ensure batch dimension
        if(x.ndim == 1):
            x = x.unsqueeze(0)

        if requires_grad:
            with gpytorch.settings.fast_pred_var():
                weight = self.get_weight(x, requires_grad)
                target_f_pred = self.target_model(x)
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                weight = self.get_weight(x)
                target_f_pred = self.target_model(x)

        #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #    weight = self.get_weight(x)
        #    target_f_pred = self.target_model(x)

        # Transform weight to exploration weight [0,1]
        # 0 when weight < (1 - threshold): exploration/exploitation tradeoff
        # Linear increase above threshold: increasing exploration
        weight = torch.where(weight > 1 - self.explorative_threshold, 2*weight-1, 0)
        
        # Standard acq function
        target_boundary_term = torch.sqrt(target_f_pred.variance) / (torch.abs(target_f_pred.mean) + epsilon)
        
        
        
        if(vanilla_acq):
            return target_boundary_term
        elif(distance_acq):
            # Distance-based exploration term
            distances = torch.cdist(
                scaler(sampled_points, self.min_scale, self.max_scale), 
                scaler(x, self.min_scale, self.max_scale)
            )
            min_distance = torch.min(distances, dim=0)[0]
            return ((1-weight)*target_boundary_term + weight*(target_f_pred.variance + min_distance))
        else:
            # Weighted combination of exploitation and exploration
            return ((1-weight)*target_boundary_term + weight*target_f_pred.variance)


class SKPhaseTransferGP(PhaseTransferGP):
    """
    Transfer Learning GP with scikit-learn compatible source models.
    
    This variant of PhaseTransferGP supports source models from scikit-learn
    or other libraries that follow the scikit-learn API (predict, predict_proba).
    The main difference is in how source model predictions are obtained and
    processed.
    
    Attributes:
        Inherits all attributes from PhaseTransferGP
    """
    def __init__(
            self,
            source_model_list,
            train_x,
            min_scale = None,
            max_scale = None,
            prior_aggregation = "linear",
            max_adaptive_power=5,
            explorative_threshold = 0.4,
            kernel_choice='matern32',
            lengthscale=0.3,
            learning_rate=0.1,
            training_iterations = 120,
            lengthscale_interval = (0.2,0.3),
            outputscale_interval = (1.0,4.0),
            device = "cpu"
            ):
        """
        Initialize the scikit-learn compatible transfer learning GP.
        
        Args:
            Same as PhaseTransferGP, but source_model_list contains 
            scikit-learn compatible models
        """
        super().__init__(
            source_model_list = source_model_list,
            train_x = train_x,
            min_scale = min_scale,
            max_scale = max_scale,
            prior_aggregation = prior_aggregation,
            max_adaptive_power= max_adaptive_power,
            explorative_threshold = explorative_threshold,
            kernel_choice=kernel_choice,
            lengthscale=lengthscale,
            learning_rate=learning_rate,
            training_iterations = training_iterations,
            lengthscale_interval = lengthscale_interval,
            outputscale_interval = outputscale_interval,
            device = device
        )
    
    def forward(self, x):
        """
        Forward pass with scikit-learn source models.
        
        Differs from parent class by calling predict_proba on sklearn models
        and handling potential 2D output (for binary classification).
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Combined probability predictions, shape (n,)
        """
        x = ensure_tensor(x, device=self.device)
        # Get target model predictions
        target_f_pred = self.target_model(x)
        target_y_pred = self.target_model.likelihood(target_f_pred)

        source_y_mean_list = []
        weight_list = []
        
        # Process each source model
        for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
            # Get weight for this source
            latent_weight = weight_model(x)
            weight = weight_model.likelihood(latent_weight).mean
            weight_list.append(weight)

            # Get sklearn model predictions (convert to CPU for sklearn)
            y_pred = source_model.predict_proba(x.cpu())
            
            # Handle binary classification output (n_samples, 2)
            if(y_pred.ndim == 2):
                y_pred = y_pred[:,1]  # Take probability of class 1

            source_y_mean_list.append(torch.tensor(y_pred, device=self.device))
        
        weight_list = torch.stack(weight_list)
        source_y_mean_list = torch.stack(source_y_mean_list)

        # Select best source for each point
        cols = torch.arange(x.size(0))
        max_indices = torch.argmax(weight_list, dim=0)

        weight = weight_list[max_indices, cols]
        source_y_mean = source_y_mean_list[max_indices, cols]

        # Apply adaptive power transformation
        power = torch.log(torch.e + 2*(torch.e**self.max_adaptive_power-torch.e)*torch.abs(source_y_mean - 0.5))

        # Weight transformation and combination
        weight = weight**power
        y_pred_mean = weight*source_y_mean + (1-weight)*target_y_pred.mean

        return y_pred_mean
    
    def fit(self, train_x, train_y, epsilon=0.05, verbose=False):
        """
        Train the model with scikit-learn source models.
        
        Uses SKMultiGPMeanModule for the informed prior which handles
        sklearn model predictions appropriately.
        
        Args:
            train_x (torch.Tensor): Training inputs of shape (n, d)
            train_y (torch.Tensor): Binary training labels of shape (n,)
            epsilon (float): Regularization parameter
            verbose (bool): Whether to print training progress
        """
        train_x = ensure_tensor(train_x, device= self.device)
        train_y = ensure_tensor(train_y, device = self.device)
        i = 0
        
        # Train weight models
        for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
            i += 1
            if(verbose):
                print(f"Training weight model {i}/{len(self.source_model_list)}")
            
            # Get source predictions using sklearn interface
            source_phases = source_model.predict(train_x.cpu())
            source_phases = ensure_tensor(source_phases, device=self.device)
            auxiliary_y = torch.eq(source_phases, train_y).int()
            
            # Train weight model
            weight_model.fit(train_x, auxiliary_y, epsilon=epsilon)

        # Initialize informed prior for sklearn models
        self.target_model.mean_module = SKMultiGPMeanModule(
            self.source_model_list,
            self.weight_model_list,
            prior_aggregation=self.prior_aggregation,
            epsilon=epsilon,
            device = self.device
        )
        
        # Train target model
        self.target_model.fit(train_x, train_y, epsilon=epsilon)

    
    def predict_proba(self, x):
        """
        Predict probabilities with sklearn source models.
        
        Args:
            x (torch.Tensor): Input points of shape (n, d)
            
        Returns:
            torch.Tensor: Probability predictions, shape (n,)
        """
        x = ensure_tensor(x, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred_mean = self(x)

        return y_pred_mean

def train_gp_model(model, train_x, train_y, learning_rate, training_iterations, verbose = False, device="cpu"):
    """
    Train a Gaussian Process model using variational inference.
    
    This function performs optimization of the variational parameters and
    hyperparameters using the Evidence Lower Bound (ELBO) as the objective.
    
    Args:
        model (gpytorch.models.ApproximateGP): GP model to train
        train_x (torch.Tensor): Training inputs of shape (n, d)
        train_y (torch.Tensor): Training targets of shape (n,)
        learning_rate (float): Learning rate for Adam optimizer
        training_iterations (int): Number of optimization iterations
        verbose (bool): Whether to print training progress
        device: Device where the training is performed (cpu or cuda)
    Returns:
        tuple: (model, likelihood) - Trained model and likelihood
    """
    train_x = ensure_tensor(train_x, device=device)
    train_y = ensure_tensor(train_y, device=device)
    
    # Set up Bernoulli likelihood for binary classification
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    
    # Initialize Adam optimizer for all model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Variational ELBO objective for approximate inference
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimization loop
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)  # Negative ELBO to minimize
        loss.backward()
        optimizer.step()
        
        if verbose and (i+1) % 25 == 0:
            current_ls = model.covar_module.base_kernel.lengthscale.item()
            print(f'Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}, Lengthscale: {current_ls:.3f}')
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    # Print final hyperparameters if verbose
    final_ls = model.covar_module.base_kernel.lengthscale.item()
    if verbose:
        print(f"Trained hyperparameters: lengthscale={final_ls:.3f}, outputscale={model.covar_module.outputscale.item():.3f}")
    
    return model, likelihood