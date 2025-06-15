import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

def softmax(x: torch.Tensor, dim: int):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        x (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `x` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `x` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = torch.max(x, dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return output

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = K.shape[-1]
    attention_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k) 

    if mask is not None:
        attention_score = attention_score.masked_fill(~mask, float('-inf')) # fill the mask false value with -inf
    
    output = softmax(attention_score, -1) @ V
    return output

    