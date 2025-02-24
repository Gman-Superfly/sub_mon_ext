#!/usr/bin/env python3
"""
Sublinear Monotonicity Score Extractor
A sublinear time tool to compute monotonicity scores for 1D sequences in PyTorch.
Useful for ML preprocessing, feature engineering, and interpretability.
You can find the ref to the 2011 paper in readme. ps: it has other fun stuff you might find useful.
"""

import torch
import math
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple

# ------------------------------
# Input Validation
# ------------------------------

def validate_inputs(input_tensor: torch.Tensor, epsilon: float) -> None:
    """
    Validate inputs for monotonicity score calculation to ensure robustness.

    Purpose: Ensures the input tensor and epsilon parameter meet requirements before
    proceeding with computation, preventing runtime errors or illogical results.

    Args:
        input_tensor (torch.Tensor): The input sequence to evaluate.
        epsilon (float): Error tolerance parameter controlling sample size.

    Raises:
        ValueError: If the input tensor isn’t a tensor, isn’t 1D, or if epsilon is invalid.

    Why: Robust input validation is critical for ML tools to handle diverse use cases
    and provide clear error messages to users.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    if input_tensor.dim() != 1:
        raise ValueError("Input must be a 1D tensor")
    if not (0 < epsilon <= 1):
        raise ValueError("Epsilon must be in range (0, 1]")

# ------------------------------
# Sample Size Calculation
# ------------------------------

def calculate_sample_size(n: int, epsilon: float, c: float = 2.0) -> int:
    """
    Calculate the number of samples needed for sublinear computation.

    Purpose: Determines how many points to sample based on sequence length and
    error tolerance, ensuring sublinear complexity O((1/ε) log n).

    Args:
        n (int): Length of the input sequence.
        epsilon (float): Error tolerance (smaller values increase sample size).
        c (float): Sampling constant, default 2.0 (tunable for accuracy vs. speed).

    Returns:
        int: Number of samples to use, capped at sequence length.

    Why: Sublinear sampling reduces computational cost, making this practical for
    large datasets in ML preprocessing or feature extraction.
    """
    num_samples = int(math.ceil(c / epsilon * math.log(n)))
    return min(num_samples, n)

# ------------------------------
# Vectorized Violation Counting
# ------------------------------

def compute_violations_vectorized(values: torch.Tensor, indices: torch.Tensor) -> Tuple[int, int]:
    """
    Compute monotonicity violations using vectorized operations for efficiency.

    Purpose: Counts instances where a later value in the sequence is less than an
    earlier value (a violation), leveraging PyTorch’s tensor operations.

    Args:
        values (torch.Tensor): Sampled values from the sequence.
        indices (torch.Tensor): Corresponding original indices of sampled values.

    Returns:
        Tuple[int, int]: Total number of violations and total checks performed.

    Why: Vectorized computation replaces loops, speeding up the process significantly
    for ML applications where performance is key.
    """
    n = len(values)
    # Create comparison matrices
    idx_matrix = indices.unsqueeze(0) < indices.unsqueeze(1)  # True where i < j
    val_matrix = values.unsqueeze(0) > values.unsqueeze(1)    # True where v_i > v_j
    
    # Count violations (i < j but v_i > v_j)
    violations = torch.logical_and(idx_matrix, val_matrix).sum().item()
    total_checks = idx_matrix.sum().item()  # Total number of comparisons
    
    return violations, total_checks

# ------------------------------
# Monotonicity Score Function
# ------------------------------

def monotonicity_score(
    input_tensor: torch.Tensor,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
    return_details: bool = False
) -> Union[float, Tuple[float, dict]]:
    """
    Compute a sublinear monotonicity score for a 1D tensor.

    Purpose: Quantifies how close a sequence is to being monotonically increasing,
    using sublinear sampling inspired by sublinear time algorithms. Useful as a
    feature in ML models or for data validation.

    Args:
        input_tensor (torch.Tensor): 1D tensor representing a sequence.
        epsilon (float): Error tolerance (0 < ε ≤ 1), default 0.1.
        seed (Optional[int]): Random seed for reproducibility.
        return_details (bool): If True, returns additional statistics.

    Returns:
        Union[float, Tuple[float, dict]]:
            - float: Score between 0 (non-monotone) and 1 (monotone) if return_details=False.
            - Tuple[float, dict]: (score, details) if return_details=True.

    Examples:
        >>> seq = torch.tensor([1.0, 2.0, 3.0, 2.5, 4.0])
        >>> score = monotonicity_score(seq, epsilon=0.1)
        >>> score, details = monotonicity_score(seq, epsilon=0.1, return_details=True)

    Why: Offers a fast, scalable way to assess monotonicity, enhancing ML workflows
    for time series, tabular data, or ranking tasks.
    """
    # Validate inputs
    validate_inputs(input_tensor, epsilon)
    
    n = input_tensor.size(0)
    if n < 2:
        # Trivially monotone for short sequences
        return (1.0, {"samples": 0, "violations": 0, "checks": 0}) if return_details else 1.0

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Determine sample size
    num_samples = calculate_sample_size(n, epsilon)
    
    # Sample and sort indices and values
    sampled_indices = torch.randperm(n)[:num_samples]
    sampled_values = input_tensor[sampled_indices]
    sorted_indices = torch.argsort(sampled_indices)
    sampled_indices = sampled_indices[sorted_indices]
    sampled_values = sampled_values[sorted_indices]

    # Compute violations
    total_violations, total_checks = compute_violations_vectorized(sampled_values, sampled_indices)
    
    # Calculate score
    score = 1.0 if total_checks == 0 else max(0.0, 1.0 - total_violations / total_checks)
    
    # Return with details if requested
    if return_details:
        details = {
            "samples": num_samples,
            "violations": total_violations,
            "checks": total_checks,
            "epsilon": epsilon,
            "sequence_length": n
        }
        return score, details
    
    return score

# ------------------------------
# Enhanced Neural Network Model
# ------------------------------

class EnhancedModel(nn.Module):
    """
    Enhanced neural network incorporating monotonicity scores as features.

    Purpose: Demonstrates integration of the monotonicity score into a deep learning
    model, with multiple layers for improved capacity and regularization.

    Why: Shows how structural features like monotonicity can enhance predictive
    models in ML, especially for tasks where trends matter.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the model with flexible architecture.

        Args:
            input_dim (int): Number of input features (excluding monotonicity score).
            hidden_dims (list): List of hidden layer sizes, default [64, 32].
            dropout_rate (float): Dropout probability for regularization, default 0.3.
        """
        super().__init__()
        
        # Dynamically build layers
        layers = []
        prev_dim = input_dim + 1  # Add 1 for monotonicity score
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Normalize for stability
                nn.ReLU(),                   # Non-linearity
                nn.Dropout(dropout_rate)     # Prevent overfitting
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mono_scores: torch.Tensor) -> torch.Tensor:
        """
        Forward pass incorporating monotonicity scores.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            mono_scores (torch.Tensor): Monotonicity scores of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).

        Why: Combines raw features with monotonicity information for richer input.
        """
        mono_scores = mono_scores.unsqueeze(1)  # Shape: (batch_size, 1)
        x = torch.cat([x, mono_scores], dim=1)  # Shape: (batch_size, input_dim + 1)
        return self.network(x)

# ------------------------------
# Test Sequence Generation
# ------------------------------

def generate_test_sequences(n: int = 1000, seed: Optional[int] = None) -> dict:
    """
    Generate various test sequences for evaluation.

    Purpose: Creates diverse sequences to test the monotonicity score function,
    simulating real-world ML data scenarios.

    Args:
        n (int): Length of sequences, default 1000.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        dict: Dictionary of named test sequences.

    Why: Provides a benchmark to verify the function’s behavior across different cases.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    return {
        "monotone": torch.arange(n, dtype=torch.float32),
        "random": torch.rand(n),
        "decreasing": torch.flip(torch.arange(n + 1, dtype=torch.float32), dims=[0]),
        "noisy_monotone": torch.arange(n, dtype=torch.float32) + torch.randn(n) * 0.1,
        "partially_monotone": torch.cat([
            torch.arange(n // 2, dtype=torch.float32),
            torch.rand(n - n // 2) * (n // 2)
        ])
    }

# ------------------------------
# Main Testing Block
# ------------------------------

if __name__ == "__main__":
    # Generate test sequences
    sequences = generate_test_sequences(seed=42)
    epsilon = 0.1

    # Test monotonicity score function
    print("\nTesting Monotonicity Score Function:")
    for name, seq in sequences.items():
        score, details = monotonicity_score(seq, epsilon, seed=42, return_details=True)
        print(f"\n{name.capitalize()} Sequence:")
        print(f"Score: {score:.3f}")
        print(f"Details: {details}")

    # Test model integration
    print("\n\nTesting Model Integration:")
    batch_size = 32
    seq_length = 10
    data = torch.rand(batch_size, seq_length)
    
    # Compute monotonicity scores for each sample
    mono_scores = torch.tensor(
        [monotonicity_score(data[i], epsilon) for i in range(batch_size)],
        dtype=torch.float32
    )
    
    # Initialize and run model
    model = EnhancedModel(seq_length)
    output = model(data, mono_scores)
    
    # Display results
    print(f"Input shape: {data.shape}")
    print(f"Scores shape: {mono_scores.shape}")
    print(f"Output shape: {output.shape}")
