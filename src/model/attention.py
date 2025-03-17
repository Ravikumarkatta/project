# src/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module specialized for biblical text processing.
    
    This implementation includes:
    - Scaled dot-product attention
    - Multiple attention heads
    - Optional attention masking
    - Special handling for Bible verse references
    """
    
    def __init__(
        self, 
        hidden_size: int = 768, 
        num_attention_heads: int = 12, 
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by the number of attention heads ({num_attention_heads})"
            )
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, key, and value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)
        
        # Special attention for theological content
        self.theological_bias = nn.Parameter(torch.zeros(num_attention_heads, 1, 1))
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the tensor for multi-head attention computation."""
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_size]
        
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        verse_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query_states: Tensor of shape [batch_size, seq_length, hidden_size]
            key_states: Tensor of shape [batch_size, seq_length, hidden_size]
            value_states: Tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional tensor of shape [batch_size, 1, 1, seq_length]
            output_attentions: Whether to return attention weights
            verse_positions: Optional tensor indicating positions of Bible verse references
            
        Returns:
            context_layer: Tensor of shape [batch_size, seq_length, hidden_size]
            attention_probs: Optional tensor of attention probabilities
        """
        # Project query, key, and value
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate dot-product attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Add theological bias to attention (for theological concepts)
        if verse_positions is not None:
            verse_attention_bias = verse_positions.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores + verse_attention_bias * self.theological_bias.unsqueeze(0)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add -10000 to masked positions to make their softmax value effectively zero
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention weights to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to [batch_size, seq_length, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), context_layer.size(1), self.hidden_size)
        
        # Apply output projection
        output_layer = self.output(context_layer)
        output_layer = self.output_dropout(output_layer)
        
        if output_attentions:
            return output_layer, attention_probs
        return output_layer, None


class BiblicalSelfAttention(MultiHeadAttention):
    """
    Specialized self-attention module for Biblical content.
    
    Extends MultiHeadAttention with:
    - Enhanced verse reference handling
    - Theological concept attention guidance
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        dropout_prob: float = 0.1,
        verse_attention_factor: float = 1.2,
    ):
        super().__init__(hidden_size, num_attention_heads, dropout_prob)
        self.verse_attention_factor = verse_attention_factor
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        verse_positions: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass that specially handles Bible verse references.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask
            verse_positions: Optional tensor indicating positions of Bible verse references
            output_attentions: Whether to return attention weights
            
        Returns:
            attention_output: Output tensor after self-attention
            attention_probs: Optional tensor of attention probabilities
        """
        # Enhance verse positions with attention factor
        if verse_positions is not None:
            verse_positions = verse_positions * self.verse_attention_factor
        
        # Apply standard multi-head attention
        return super().forward(
            hidden_states, hidden_states, hidden_states,
            attention_mask, output_attentions, verse_positions
        )q
