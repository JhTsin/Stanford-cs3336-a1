import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on 
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W_init = self.initialize_Weights(out_features, in_features)
        self.weight = nn.Parameter(W_init)

    def initialize_Weights(self, out_dim: int, in_dim: int) -> torch.Tensor:
        """
        Initialize the weights W using truncated normal method
        """
        W = torch.empty(out_dim, in_dim)
        mean = 0
        std = np.sqrt(2 / (in_dim + out_dim))

        nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ self.weight.T
        return output