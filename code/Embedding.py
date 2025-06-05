import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors
        device: torch.device | None = None Device to store the parameters on 
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        W_init = self.initialize_Weights(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(W_init)

    def initialize_Weights(self, vocab_size: int, d_model: int) -> torch.Tensor:
        """
        Initialize the weights W using truncated normal method
        """
        W = torch.empty(vocab_size, d_model)
        mean = 0
        std = np.sqrt(2 / (vocab_size + d_model))

        nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        return W

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        token_ids: (batch_size, sequence_length)
        output: (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length = token_ids.shape
        output = torch.empty(batch_size, sequence_length, self.embedding_dim)

        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                output[i][j] = self.weight[token_id]
        return output





