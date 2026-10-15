# Weight normalization, the one thing every weighted metric needs

# Libraries
import torch


def normalize_weights(weights, n=None, dtype=None, device=None):
    """Normalize importance weights, treating None as uniform

    Args:
        * weights (torch.Tensor of shape (n,) or None): Unnormalized weights
        * n (int or None): Number of samples, required when weights is None (default is None)
        * dtype (torch.dtype or None): Output dtype when weights is None (default is None)
        * device (torch.device or None): Output device when weights is None (default is None)

    Returns:
        * weights (torch.Tensor of shape (n,)): Weights summing to one
    """
    if weights is None:
        return torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    return weights / weights.sum()
