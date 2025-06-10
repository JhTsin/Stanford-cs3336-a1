import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, 
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on 
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps

        w_init = self.initialize_Weights(d_model, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def initialize_Weights(self, d_model: int, factory_kwargs: dict):
        """
        Initialize the weights w using truncated normal method
        """
        w = torch.ones(d_model, **factory_kwargs)
        return w

    def RMS(self, x: torch.tensor, d_model: int, eps: float):
        """
        Calculate the root mean square of input tensor x
        """
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + eps)
        return rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = self.RMS(x, self.d_model, self.eps)
        result = (x / rms) * self.weight

        return result.to(in_dtype)