"""
Informed Prior Functions for Transfer Learning in Gaussian Processes

This module implements custom mean functions that serve as informed priors
for transfer learning in GP models. These priors incorporate knowledge from
pre-trained source models to improve learning efficiency on new but related tasks.

The key idea is to use predictions from source models as the mean function
of the target GP, weighted by learned reliability scores. This allows the
target model to leverage relevant knowledge while adapting to task-specific patterns.

Classes:
    GPMeanModule: Single source model prior (not used right now)
    MultiGPMeanModule: Multiple GP source models with aggregation (when using one model, behaves like GPMeanModule)
    SKMultiGPMeanModule: Multiple scikit-learn compatible source models

Author: Eduardo Gonzalez Garcia (e.gonzalez.garcia@tue.nl)
Version: 0.1.0
"""
import torch
import gpytorch

__all__ = ["GPMeanModule", "MultiGPMeanModule", "SKMultiGPMeanModule"]

class GPMeanModule(gpytorch.means.Mean):
    """
    Informed prior mean function using a single source GP model.
    
    This class creates a mean function for the target GP that incorporates
    predictions from a pre-trained source model. The source predictions are
    weighted by a reliability model that learns where the source is trustworthy.
    
    The resulting prior function is: μ(x) = source_mean(x) * weight(x)
    
    This allows the target model to start with informed predictions in regions
    where the source is reliable, while reverting to standard GP behavior elsewhere.
    
    Attributes:
        source_model: Pre-trained GP model providing prior knowledge
        weight_model: GP model learning source reliability at each location
    """

    def __init__(self, source_model, weight_model):
        """
        Initialize the informed prior with single source.
        
        Args:
            source_model: Pre-trained PhaseGP model to use as prior
            weight_model: PhaseGP model that predicts source reliability
        
        Note:
            Both models are set to eval mode with gradients disabled
            to prevent updates during target model training.
        """
        super().__init__()
        
        # Store source model and freeze its parameters
        self.source_model = source_model
        self.source_model.eval()
        
        # Disable gradient computation for source model
        for param in self.source_model.parameters():
            param.requires_grad = False

        # Store weight model and freeze its parameters
        self.weight_model = weight_model
        self.weight_model.eval()
        self.weight_model.likelihood.eval()
        
        # Disable gradient computation for weight model and likelihood
        for param in self.weight_model.parameters():
            param.requires_grad = False
        for param in self.weight_model.likelihood.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Compute the informed prior mean at input locations.
        
        The prior combines source predictions with learned weights:
        prior(x) = source_mean(x) * weight(x)
        
        Args:
            x (torch.Tensor): Input locations of shape (n, d)
            
        Returns:
            torch.Tensor: Prior mean values of shape (n,)
        """
        with torch.no_grad():
            # Get source model's latent mean prediction
            mean_pred = self.source_model(x).mean
            
            # Get reliability weight for this location
            weight_pred = self.weight_model(x)
            weight_pred = self.weight_model.likelihood(weight_pred).mean

            # Return weighted source prediction as prior
            return mean_pred * weight_pred
        

class MultiGPMeanModule(gpytorch.means.Mean):
    """
    Informed prior mean function using multiple source GP models.
    
    This class extends the single-source concept to multiple source models,
    each with its own reliability weight. The prior can aggregate source
    predictions using different strategies:
    
    - 'linear': Weighted sum of all sources with adaptive exponentiation
    - 'highest': Use only the most reliable source at each location
    
    The adaptive exponentiation gives more weight to confident predictions
    (those far from the decision boundary), allowing the model to leverage
    strong source knowledge while being cautious about uncertain predictions.
    
    Attributes:
        source_model_list (list): List of pre-trained source GP models
        weight_model_list (list): List of GP models for source reliability
        prior_aggregation (str): Method for combining source predictions
    """

    def __init__(self, source_model_list, weight_model_list, prior_aggregation="linear", device="cpu"):
        """
        Initialize the multi-source informed prior.
        
        Args:
            source_model_list (list): List of pre-trained PhaseGP models
            weight_model_list (list): List of PhaseGP weight models
            prior_aggregation (str): Aggregation method ('linear' or 'highest')
            device: Device where the prior is stored (cpu or cuda)
        """
        super().__init__()
        self.device= device
        self.prior_aggregation = prior_aggregation
        self.source_model_list = source_model_list
        self.eval_mode(self.source_model_list)
        
        self.weight_model_list = weight_model_list
        self.eval_mode(self.weight_model_list)
    def eval_mode(self, model_list):
        """
        Set a list of models to evaluation mode and freeze parameters.
        
        This ensures source models don't update during target training,
        maintaining the transfer learning paradigm where source knowledge
        is fixed and only the target model adapts.
        
        Args:
            model_list (list): List of GP models to freeze
        """
        for model in model_list:
            # Set model to eval mode
            model.eval()
            
            # Disable gradients for model parameters
            for param in model.parameters():
                param.requires_grad = False

            # Set likelihood to eval mode
            model.likelihood.eval()
            
            # Disable gradients for likelihood parameters
            for param in model.likelihood.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Compute the aggregated prior mean from multiple sources.
        
        Two aggregation strategies are supported:
        
        1. Linear aggregation:
           prior(x) = Σ source_mean(x) * weight(x)^(|source_mean(x)| + 1)
           The adaptive exponent emphasizes confident source predictions
        
        2. Highest aggregation:
           prior(x) = best_source_mean(x) * best_weight(x)
           Uses only the most reliable source at each location
        
        Args:
            x (torch.Tensor): Input locations of shape (n, d)
            
        Returns:
            torch.Tensor: Aggregated prior mean values of shape (n,)
            
        Raises:
            Exception: If prior_aggregation is not 'linear' or 'highest'
        """
        if(self.prior_aggregation == "linear"):
            with torch.no_grad():
                full_pred = torch.zeros(x.shape[0], device=self.device)
                
                # Sum weighted contributions from all sources
                for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
                    # Get source latent mean
                    mean_pred = source_model(x).mean
                    
                    # Get reliability weight
                    weight_pred = weight_model(x)
                    weight_pred = weight_model.likelihood(weight_pred).mean
                    
                    # Adaptive exponentiation: higher power for confident predictions
                    # This emphasizes sources that are far from the decision boundary
                    adaptive_exponent = torch.abs(mean_pred) + 1

                    # Add weighted contribution
                    full_pred += mean_pred * (weight_pred**adaptive_exponent)

                return full_pred
                
        elif(self.prior_aggregation == "highest"):
            with torch.no_grad():
                weight_list = []
                source_f_mean_list = []
                
                # Collect predictions and weights from all sources
                for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
                    # Get weight prediction
                    latent_weight_pred = weight_model(x)
                    weight_pred = weight_model.likelihood(latent_weight_pred).mean
                    weight_list.append(weight_pred)
                    
                    # Get source mean prediction
                    source_f_mean_list.append(source_model(x).mean)
                
                weight_list = torch.stack(weight_list)
                source_f_mean_list = torch.stack(source_f_mean_list)

                # Select the source with highest weight at each location
                cols = torch.arange(x.size(0))
                max_indices = torch.argmax(weight_list, dim=0)

                weight = weight_list[max_indices, cols]
                source_f_mean = source_f_mean_list[max_indices, cols]

                # Return weighted prediction from best source
                return weight * source_f_mean
            
        else:
            raise Exception('Unknown prior_aggregation value, use linear or highest')
    

class SKMultiGPMeanModule(gpytorch.means.Mean):
    """
    Informed prior mean function for scikit-learn compatible source models.
    
    This class adapts the multi-source prior concept to work with non-GP
    models that follow the scikit-learn API (e.g., Random Forests, SVMs).
    Since these models don't provide latent function values, their probability
    predictions are transformed to the latent space using a logit function.
    
    The logit transformation with epsilon regularization maps probabilities
    from [0,1] to ~(-3,3) for ε=0.05, making them suitable as GP mean function values.
    
    Attributes:
        source_model_list (list): List of sklearn-compatible models
        weight_model_list (list): List of GP models for reliability weights
        prior_aggregation (str): Method for combining predictions
        epsilon (float): Regularization for logit transformation
    """
    
    def __init__(self, source_model_list, weight_model_list, prior_aggregation="linear", epsilon=0.05, device="cpu"):
        """
        Initialize the scikit-learn compatible multi-source prior.
        
        Args:
            source_model_list (list): List of sklearn-compatible models
            weight_model_list (list): List of PhaseGP weight models
            prior_aggregation (str): Aggregation method ('linear' or 'highest')
            epsilon (float): Regularization parameter for logit transform
        """
        super().__init__()
        self.prior_aggregation = prior_aggregation
        self.source_model_list = source_model_list
        self.weight_model_list = weight_model_list
        self.epsilon = epsilon
        self.device = device

    def forward(self, x):
        """
        Compute aggregated prior from scikit-learn source models.
        
        The process differs from GP sources:
        1. Get probability predictions from sklearn models
        2. Transform probabilities to latent space using regularized logit
        3. Apply same aggregation strategies as MultiGPMeanModule
        
        The logit transformation is: f = log((p + ε) / (1 - p + ε))
        where p is the predicted probability and ε prevents numerical issues.
        
        Args:
            x (torch.Tensor): Input locations of shape (n, d)
            
        Returns:
            torch.Tensor: Aggregated prior mean in latent space, shape (n,)
            
        Raises:
            Exception: If prior_aggregation is not 'linear' or 'highest'
        """
        if(self.prior_aggregation == "linear"):
            with torch.no_grad():
                full_pred = torch.zeros(x.shape[0], device=self.device)
                
                for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
                    # Get probability predictions from sklearn model
                    mean_pred = source_model.predict_proba(x.cpu())
                    
                    # Handle binary classification output shape
                    if(mean_pred.ndim == 2):
                        mean_pred = mean_pred[:,1]  # Take probability of positive class
                    mean_pred = torch.tensor(mean_pred, device=self.device)

                    # Transform probability to latent space using regularized logit
                    # This maps [0,1] to ~(-3,3) for ε=0.05 suitable for GP mean function
                    mean_pred = torch.log((mean_pred + self.epsilon) / (1 - mean_pred + self.epsilon))

                    # Get reliability weight from GP model
                    weight_pred = weight_model(x)
                    weight_pred = weight_model.likelihood(weight_pred).mean
                    
                    # Adaptive exponentiation based on confidence
                    adaptive_exponent = torch.abs(mean_pred) + 1
                    
                    # Add weighted contribution
                    full_pred += mean_pred * (weight_pred**adaptive_exponent)
                    
                return full_pred
                
        elif(self.prior_aggregation == "highest"):
            with torch.no_grad():
                weight_list = []
                source_mean_list = []
                
                for source_model, weight_model in zip(self.source_model_list, self.weight_model_list):
                    # Get weight prediction
                    latent_weight_pred = weight_model(x)
                    weight_pred = weight_model.likelihood(latent_weight_pred).mean
                    weight_list.append(weight_pred)

                    # Get sklearn model probability prediction
                    mean_pred = source_model.predict_proba(x.cpu())[:,1]
                    mean_pred = torch.tensor(mean_pred, device=self.device)

                    # Transform to latent space
                    mean_pred = torch.log((mean_pred + self.epsilon) / (1 - mean_pred + self.epsilon))

                    source_mean_list.append(mean_pred)
                
                weight_list = torch.stack(weight_list)
                source_mean_list = torch.stack(source_mean_list)

                # Select best source at each location
                cols = torch.arange(x.size(0))
                max_indices = torch.argmax(weight_list, dim=0)

                weight = weight_list[max_indices, cols]
                source_f_mean = source_mean_list[max_indices, cols]

                # Return weighted prediction from best source
                return weight * source_f_mean
            
        else:
            raise Exception('Unknown prior_aggregation value, use linear or highest')