import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class TheologicalLoss(nn.Module):
    """
    Custom loss function combining multiple objectives for biblical text generation.
    Includes main language modeling loss and auxiliary losses for verse detection
    and theological accuracy.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {
            "main_loss": "cross_entropy",
            "auxiliary_losses": {
                "verse_detection": {
                    "loss_type": "cross_entropy",
                    "weight": 0.3
                },
                "theological_accuracy": {
                    "loss_type": "kl_divergence",
                    "weight": 0.5
                }
            }
        }
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        verse_logits: Optional[torch.Tensor] = None,
        verse_labels: Optional[torch.Tensor] = None,
        theological_logits: Optional[torch.Tensor] = None,
        theological_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss.
        
        Args:
            logits: Main language model logits (B, S, V)
            labels: Target labels (B, S)
            verse_logits: Verse detection logits (optional)
            verse_labels: Verse detection labels (optional)
            theological_logits: Theological classification logits (optional)
            theological_labels: Theological classification labels (optional)
            attention_mask: Attention mask for padding (optional)
        
        Returns:
            Dictionary containing total loss and individual components
        """
        # Calculate main loss (language modeling)
        if self.config["main_loss"] == "cross_entropy":
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100  # Standard padding index
            )
        else:
            raise ValueError(f"Unsupported main loss type: {self.config['main_loss']}")
        
        losses = {"main_loss": main_loss}
        total_loss = main_loss
        
        # Add verse detection loss if provided
        if verse_logits is not None and verse_labels is not None:
            verse_loss_config = self.config["auxiliary_losses"]["verse_detection"]
            if verse_loss_config["loss_type"] == "cross_entropy":
                verse_loss = F.cross_entropy(
                    verse_logits.view(-1, verse_logits.size(-1)),
                    verse_labels.view(-1),
                    ignore_index=-100
                )
                losses["verse_loss"] = verse_loss
                total_loss += verse_loss_config["weight"] * verse_loss
        
        # Add theological accuracy loss if provided
        if theological_logits is not None and theological_labels is not None:
            theo_loss_config = self.config["auxiliary_losses"]["theological_accuracy"]
            if theo_loss_config["loss_type"] == "kl_divergence":
                # Apply softmax to get distributions
                theo_probs = F.softmax(theological_logits, dim=-1)
                theo_loss = F.kl_div(
                    theo_probs.log(),
                    theological_labels,
                    reduction='batchmean'
                )
                losses["theological_loss"] = theo_loss
                total_loss += theo_loss_config["weight"] * theo_loss
        
        losses["total_loss"] = total_loss
        return losses