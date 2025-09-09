import torch
import torch.nn as nn
from .Linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu
            w1 (Float[Tensor, "d_ff d_model"]): Stored weights for W1
            w2 (Float[Tensor, "d_model d_ff"]): Stored weights for W2
            w3 (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the SwiGLU feed-forward network
        Args:
            x (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer

        Returns:
            Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings
        """
        # 分割输入为两部分（假设输入维度为偶数）
        assert x.shape[-1] % 2 == 0, "输入维度需为偶数"
        a, b = x.chunk(2, dim=-1)
        
        # 直接计算输出
        output = self.w2(silu(self.w1(a)) * self.w3(b))
        # 线性层（silu（线性层（））*线性层（））
        return output
    