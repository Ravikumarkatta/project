# src/model/embeddings.py
"""
Embedding modules for the BiblicalTransformer model.

Includes token embeddings for vocabulary and positional encodings for sequence context.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TokenEmbeddings(nn.Module):
    """
    Token embeddings for the model's vocabulary.

    Maps input token IDs to dense embeddings, with padding support.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
        dropout: float = 0.1
    ):
        """
        Initialize token embeddings.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            padding_idx (int): Index of the padding token.
            dropout (float): Dropout rate for regularization.
        """
        super(TokenEmbeddings, self).__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            nn.init.zeros_(self.weight[padding_idx])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input token IDs to embeddings.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs [batch_size, seq_length].

        Returns:
            torch.Tensor: Embedded tokens [batch_size, seq_length, embedding_dim].
        """
        embeddings = self.weight[input_ids]
        return self.dropout(embeddings)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Adds positional information to token embeddings to capture sequence order.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.

        Args:
            d_model (int): Dimension of the model (embedding size).
            max_seq_length (int): Maximum sequence length to precompute encodings.
            dropout (float): Dropout rate for regularization.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings [batch_size, seq_length, d_model].

        Returns:
            torch.Tensor: Embeddings with positional encoding added.
        """
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)