import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .Attention import MultiheadSelfAttention
from .PositionwiseFeedForward import PositionwiseFeedForward

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        """
        d_model (int): Dimensionality of the Transformer block inputs.
        num_heads (int): Number of heads to use in multi-head self-attention. 
        d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache.
        theta (float): RoPE parameter.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.rms_norm2 = RMSNorm(d_model=d_model)
        self.attn = MultiheadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            use_rope=True, 
            max_seq_len=max_seq_len,
            theta=theta
        )
        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
            running the Transformer block on the input features while using RoPE.
        """
        # First sublayer for multihead self attention
        y = x + self.attn(self.rms_norm1(x))

        # Second sublayer for feed-forward network
        output = y + self.ff(self.rms_norm2(y))
        return output
