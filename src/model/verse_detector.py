import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Dict, Tuple, Optional, Union


class VerseDetector(nn.Module):
    """
    A module for detecting and classifying verse patterns in text.
    
    This module is responsible for analyzing text to identify structural patterns
    that indicate different types of verse formats (e.g., poetry, scripture, etc.).
    It is referenced by architecture.py as noted in the priority list.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_verse_types: int = 5,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the verse detector module.
        
        Args:
            hidden_dim (int): Dimension of the hidden representations from the transformer
            num_verse_types (int): Number of different verse types to detect
            dropout_rate (float): Dropout probability for regularization
        """
        super(VerseDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_verse_types = num_verse_types
        
        # Layers for verse detection
        self.verse_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        self.verse_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Classification head for verse type detection
        self.verse_classifier = nn.Linear(hidden_dim, num_verse_types)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for verse detection.
        
        Args:
            hidden_states (torch.Tensor): Hidden representations from transformer
                                         [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens
                                                    [batch_size, seq_len]
                                                    
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'verse_logits': Logits for verse type classification [batch_size, seq_len, num_verse_types]
                - 'verse_features': Enhanced features for verse detection [batch_size, seq_len, hidden_dim]
        """
        # Convert attention mask format if provided
        key_padding_mask = None
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, seq_len] with False for tokens to attend to
            # and True for tokens to ignore
            key_padding_mask = (1 - attention_mask).bool()
        
        # Self-attention for verse detection
        # Transpose for attention: [seq_len, batch_size, hidden_dim]
        hidden_states_t = hidden_states.transpose(0, 1)
        # Simulate sparse attention by masking low-attention tokens
        # Simplified: only attend to nearby tokens (±5 positions)
        seq_len = hidden_states_t.size(0)
        sparse_mask = torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool)
        for i in range(seq_len):
        start = max(0, i - 5)
        end = min(seq_len, i + 6)
        sparse_mask[i, start:end] = 0
        sparse_mask = sparse_mask.bool()
        
        # Apply self-attention
        attn_output, _ = self.verse_attention(
            query=hidden_states_t,
            key=hidden_states_t,
            value=hidden_states_t,
            key_padding_mask=key_padding_mask
            attn_mask=sparse_mask
        )
        
        # Transpose back: [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.transpose(0, 1)
        
        # First residual connection and layer norm
        hidden_states = self.norm1(hidden_states + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.verse_ff(hidden_states)
        
        # Second residual connection and layer norm
        verse_features = self.norm2(hidden_states + self.dropout(ff_output))
        
        # Verse type classification
        verse_logits = self.verse_classifier(verse_features)
        
        return {
            'verse_logits': verse_logits,
            'verse_features': verse_features
        }
    
    def detect_verse_patterns(
        self,
        text: str,
        tokenizer
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Detect verse patterns in raw text.
        
        Args:
            text (str): Raw input text to analyze
            tokenizer: Tokenizer object used for the model
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of detected verse segments with their types and confidence scores
        """
        # This is a helper method to use the model for inference on raw text
        # Split text into lines for analysis
        lines = text.split('\n')
        results = []
        
        # Basic regex patterns for different verse types (simplified)
        verse_patterns = {
            'numbered_verse': r'^\d+[:\.]\d+',  # e.g., "3:16" or "3.16"
            'poetry_stanza': r'^(\s{2,}|\t+)',  # Indented lines often indicate poetry
            'list_item': r'^\s*[\*\-\•]\s+',    # Bullet points
            'quotation': r'^[\"\']',            # Starting with quotation marks
        }
        
        for line in lines:
            verse_type = 'prose'  # Default
            confidence = 0.5      # Default confidence
            
            # Check against regex patterns
            for pattern_name, pattern in verse_patterns.items():
                if re.match(pattern, line):
                    verse_type = pattern_name
                    confidence = 0.8  # Higher confidence for pattern matches
                    break
            
            results.append({
                'text': line,
                'verse_type': verse_type,
                'confidence': confidence
            })
        
        return results


class VerseSegmenter(nn.Module):
    """
    A module for segmenting text into verse units.
    
    This is a companion module to the VerseDetector that focuses specifically
    on determining verse boundaries and structure.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_segment_length: int = 256
    ):
        """
        Initialize the verse segmenter.
        
        Args:
            hidden_dim (int): Dimension of the hidden representations
            max_segment_length (int): Maximum length of a verse segment
        """
        super(VerseSegmenter, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_segment_length = max_segment_length
        
        # Boundary detection network
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for verse segmentation.
        
        Args:
            hidden_states (torch.Tensor): Hidden representations [batch_size, seq_len, hidden_dim]
            attention_mask (torch.Tensor, optional): Mask to avoid padding tokens [batch_size, seq_len]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'boundary_scores': Probability of token being a verse boundary [batch_size, seq_len, 1]
        """
        # Detect potential verse boundaries
        boundary_scores = self.boundary_detector(hidden_states)
        
        # Apply mask if provided (0 probability for padding tokens)
        if attention_mask is not None:
            boundary_scores = boundary_scores * attention_mask.unsqueeze(-1)
        
        return {
            'boundary_scores': boundary_scores
        }
    
    @staticmethod
    def segment_text(
        boundary_scores: torch.Tensor,
        tokens: List[str],
        threshold: float = 0.5
    ) -> List[str]:
        """
        Segment text based on boundary predictions.
        
        Args:
            boundary_scores (torch.Tensor): Predicted boundary scores [seq_len, 1]
            tokens (List[str]): Original tokens corresponding to the scores
            threshold (float): Threshold for boundary detection
            
        Returns:
            List[str]: List of segmented verse units
        """
        boundaries = [0] + [i+1 for i, score in enumerate(boundary_scores) if score > threshold]
        
        if boundaries[-1] != len(tokens):
            boundaries.append(len(tokens))
        
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            segment_tokens = tokens[start:end]
            segments.append(" ".join(segment_tokens))
        
        return segments
