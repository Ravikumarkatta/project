# src/model/architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.model.attention import MultiHeadAttention
from src.model.embeddings import TokenEmbeddings, PositionalEncoding
from src.model.verse_detector import VerseReferenceDetector


class BiblicalTransformerConfig:
    """Configuration class for BiblicalTransformer model."""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,  # Extended context window for longer theological discussions
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = True,
        classifier_dropout: float = 0.1,
        theological_embedding_size: int = 128,  # Special embeddings for theological concepts
        verse_embedding_size: int = 64,  # Special embeddings for Bible verse references
        num_bible_books: int = 66,  # Number of books in the Bible
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.theological_embedding_size = theological_embedding_size
        self.verse_embedding_size = verse_embedding_size
        self.num_bible_books = num_bible_books


class BiblicalTransformerLayer(nn.Module):
    """
    Single transformer layer for the BiblicalTransformer model.
    Includes self-attention, feed-forward network, and special biblical context handling.
    """
    
    def __init__(self, config: BiblicalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Self-attention mechanism
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob
        )
        
        # Modified feed-forward network with gradient checkpointing
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob if self.training else 0),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob if self.training else 0)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Optional: Theological context integration
        self.theological_context_gate = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        theological_context: Optional[torch.Tensor] = None,
        verse_references: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attention_output, attention_weights = self.attention(
            hidden_states, hidden_states, hidden_states, attention_mask, output_attentions
        )
        hidden_states = residual + attention_output
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = residual + feed_forward_output
        
        # Integrate theological context if provided
        if theological_context is not None:
            context_gate = self.theological_context_gate(hidden_states)
            hidden_states = hidden_states * (1 - context_gate) + theological_context * context_gate
        
        if output_attentions:
            return hidden_states, attention_weights
        return hidden_states, None


class BiblicalTransformer(nn.Module):
    """
    Transformer model specialized for biblical content and interpretation.
    Features enhanced context understanding for theological concepts and verse references.
    """
    
    def __init__(self, config: BiblicalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Core embeddings
        self.token_embedding = TokenEmbeddings(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embedding = PositionalEncoding(
            d_model=config.hidden_size,
            max_seq_length=config.max_position_embeddings,
            dropout=config.hidden_dropout_prob
        )
        
        # Special embeddings for biblical content
        self.verse_reference_detector = VerseReferenceDetector(config.num_bible_books)
        self.verse_embedding = nn.Embedding(config.num_bible_books * 200, config.verse_embedding_size)  # Approximating verses per book
        self.verse_projection = nn.Linear(config.verse_embedding_size, config.hidden_size)
        
        # Theological concept embedding (for known theological categories)
        self.theological_embedding = nn.Embedding(1000, config.theological_embedding_size)  # 1000 theological concepts
        self.theological_projection = nn.Linear(config.theological_embedding_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BiblicalTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output components
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings and LM head weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Additional prediction heads for specialized biblical tasks
        self.verse_prediction_head = nn.Linear(config.hidden_size, config.num_bible_books * 200)
        self.theological_classification_head = nn.Linear(config.hidden_size, 1000)  # 1000 theological concepts
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.verse_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.theological_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_verse_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract and embed Bible verse references from input tokens."""
        verse_indices = self.verse_reference_detector(input_ids)
        verse_embeds = self.verse_embedding(verse_indices)
        return self.verse_projection(verse_embeds)
    
    def get_theological_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract theological context from current hidden states."""
        # This is a simplified version - in practice would use a more sophisticated mechanism
        theological_logits = self.theological_classification_head(hidden_states[:, 0, :])
        theological_probs = F.softmax(theological_logits, dim=-1)
        theological_embeds = self.theological_embedding.weight.unsqueeze(0) * theological_probs.unsqueeze(-1)
        theological_embeds = theological_embeds.sum(dim=1)
        return self.theological_projection(theological_embeds).unsqueeze(1).expand_as(hidden_states)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        theological_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.size()
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Create embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Apply positional encoding directly to token embeddings
        hidden_states = self.position_embedding(token_embeds)
        
        # Detect and embed verse references
        verse_embeds = self.get_verse_embeddings(input_ids)
        
        # Add verse embeddings
        hidden_states = hidden_states + verse_embeds
        
        # Initialize tracking variables
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get theological context (every 3rd layer)
            theological_context = None
            if i % 3 == 0:
                theological_context = self.get_theological_context(hidden_states)
            
            hidden_states, attention_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                theological_context=theological_context,
                output_attentions=output_attentions
            )
            
            if output_attentions:
                all_attentions = all_attentions + (attention_weights,)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Get language modeling logits
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # Calculate additional biblically-relevant predictions
        verse_logits = self.verse_prediction_head(hidden_states)
        theological_logits = self.theological_classification_head(hidden_states)
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "verse_logits": verse_logits,
            "theological_logits": theological_logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }