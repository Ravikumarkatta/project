import torch
from transformers import (  # Assuming you're using Hugging Face's tokenizer
    PreTrainedTokenizer,
)

from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig


def test_biblical_transformer() -> None:
    """
    Test the forward pass of the BiblicalTransformer model.

    Ensures that the model produces the expected output shapes for logits and verse_logits.
    """
    # Initialize the tokenizer
    tokenizer = PreTrainedTokenizer.from_pretrained("bert-base-uncased")

    # Define the model configuration
    config = BiblicalTransformerConfig(
        vocab_size=1000, hidden_size=768, num_hidden_layers=2
    )

    # Initialize the model with the configuration and tokenizer
    model = BiblicalTransformer(config, tokenizer=tokenizer)

    # Create random input IDs
    input_ids = torch.randint(0, 1000, (2, 10))

    # Perform a forward pass
    output = model(input_ids)

    # Assertions to verify the output shapes
    assert output["logits"].shape == (2, 10, 1000)
    assert output["verse_logits"].shape == (2, 10, config.num_bible_books * 200)
