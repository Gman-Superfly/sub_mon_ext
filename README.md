# Sublinear Monotonicity Score Extractor

## What It Does
This code provides a **monotonicity score** for 1D sequences (e.g., time series or ordered features), measuring how close they are to being monotonically increasing:
- **Score = 1.0**: Fully monotone (e.g., `[1, 2, 3, 4]`)
- **Score = 0.0**: Highly non-monotone (e.g., `[4, 3, 2, 1]` or random)

It uses a sublinear number of samples (`O((1/ε) log n)`), making it fast even for large datasets.

## Features
- **Vectorized Implementation**: Fast computation using PyTorch operations
- **Detailed Analysis**: Optional return of sampling statistics and violation details
- **Type Safety**: Full type hints and input validation
- **Enhanced Model**: Neural network with batch normalization and dropout
- **Comprehensive Testing**: Various sequence types for thorough evaluation


## Zoom Zoom

### Basic Start
```python
import torch
from monotonicity_score import monotonicity_score

# Simple score calculation
sequence = torch.tensor([1.0, 2.0, 3.0, 2.5, 4.0])
score = monotonicity_score(sequence, epsilon=0.1)
print(f"Monotonicity score: {score}")

# Get detailed statistics
score, details = monotonicity_score(sequence, epsilon=0.1, return_details=True)
print(f"Details: {details}")
```

### Neural Network Integration
```python
from monotonicity_score import EnhancedModel

# Create and use the enhanced model
data = torch.rand(32, 10)  # Batch of 32 samples, 10 features each
scores = torch.tensor([monotonicity_score(data[i]) for i in range(32)])
model = EnhancedModel(
    input_dim=10, 
    hidden_dims=[64, 32], 
    dropout_rate=0.3
)
output = model(data, scores)
```

## Implementation 

### Monotonicity Score Function
```python
def monotonicity_score(
    input_tensor: torch.Tensor, 
    epsilon: float = 0.1, 
    seed: Optional[int] = None,
    return_details: bool = False
) -> Union[float, Tuple[float, dict]]
```

#### Parameters
- **input_tensor**: 1D PyTorch tensor representing the sequence
- **epsilon**: Error tolerance (0 < ε ≤ 1). Lower values mean more samples and higher accuracy
- **seed**: Optional random seed for reproducible results
- **return_details**: If True, returns additional statistics about the computation

#### Returns
- Score between 0 and 1, or
- Tuple of (score, details_dict) if return_details=True

### Enhanced Neural Network
```python
class EnhancedModel(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout_rate: float = 0.3
    )
```

#### Features
- Configurable architecture with multiple hidden layers
- Batch normalization for training stability
- Dropout for regularization
- ReLU activations for non-linearity

## Extras and Adv Usage

### Generating Test Sequences
```python
sequences = generate_test_sequences(n=1000, seed=42)
for name, seq in sequences.items():
    score = monotonicity_score(seq)
    print(f"{name}: {score:.3f}")
```

### Types of Test Sequences
- Monotone increasing
- Random
- Strictly decreasing
- Noisy monotone (with Gaussian noise)
- Partially monotone

## Performance 

### Sampling Efficiency
- Sublinear complexity: `O((1/ε) log n)`
- Vectorized violation counting
- Configurable accuracy vs. speed trade-off via epsilon

### Memory Usage
- Efficient tensor operations
- No redundant memory allocation
- Batch processing support

### Optimization Tips
- For sequences > 10⁶ elements: Use larger epsilon
- For real-time applications: Cache scores for static sequences
- For sequences < 100 elements: Consider exact computation

## Considerations

### Parameter Selection
- **epsilon**: Start with 0.1, adjust based on needs
  - Lower values (e.g., 0.01) for higher accuracy
  - Higher values (e.g., 0.2) for faster computation
- **hidden_dims**: Adjust based on data complexity
  - More/larger layers for complex relationships
  - Fewer/smaller layers for simpler data

### Model Training
- Normalize input features
- Use appropriate learning rate scheduling
- Monitor validation metrics
- Consider early stopping

## The Algo Came From Here: https://people.csail.mit.edu/ronitt/papers/TR11-013.pdf

## License
MIT License - see LICENSE file for details
