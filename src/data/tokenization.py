# bible/src/data/tokenization.py
"""
Custom tokenization for Bible-AI project.

This module provides tokenization utilities for biblical texts, preserving verse references,
theological terms, and handling special cases like Hebrew/Greek terms. It integrates with
HuggingFace tokenizers and supports the data pipeline for model training.
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedTokenizer, AutoTokenizer
from nltk.tokenize import word_tokenize
import nltk

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Project-specific imports
try:
    from src.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger("BiblicalTokenizer")


class BiblicalTokenizer:
    """Custom tokenizer for biblical texts with verse reference preservation."""

    def __init__(
        self,
        base_tokenizer_name: str = "bert-base-uncased",
        config_path: Optional[str] = "config/data_config.json",
        max_length: int = 512
    ):
        """
        Initialize the tokenizer with a base HuggingFace tokenizer and custom rules.

        Args:
            base_tokenizer_name: Name of the base tokenizer (e.g., 'bert-base-uncased').
            config_path: Path to configuration file for tokenization rules.
            max_length: Maximum sequence length for tokenization.
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        self.max_length = max_length
        self.config = self._load_config(config_path)
        
        # Compile regex patterns for verse references and special terms
        self.verse_pattern = re.compile(r'\[(\d+:\d+)\]')  # From preprocessing.py: [chapter:verse]
        self.book_chapter_verse_pattern = re.compile(
            r'([1-3]?\s*[A-Za-z]+)\s+(\d+):(\d+)(?:-(\d+))?'
        )  # e.g., "John 3:16" or "John 3:16-18"
        self.theological_terms = set(self.config.get("theological_terms", []))
        self.special_terms = self.config.get("special_terms", {"YHWH", "JHVH", "LORD", "Son of Man"})
        logger.info("Initialized BiblicalTokenizer with base tokenizer %s", base_tokenizer_name)

    def _load_config(self, config_path: str) -> Dict:
        """Load tokenization configuration from file."""
        default_config = {
            "theological_terms": [
                "god", "jesus", "christ", "holy spirit", "messiah", "sin", "salvation",
                "grace", "faith", "prophet", "apostle", "gospel", "covenant"
            ],
            "special_terms": {"YHWH", "JHVH", "LORD", "Son of Man"},
            "preserve_verse_references": True,
            "handle_special_terms": True
        }
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                default_config.update(config)
                logger.info("Loaded tokenization config from %s", config_path)
            except Exception as e:
                logger.error("Failed to load config: %s, using defaults", e)
        return default_config

    def _preserve_special_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace special terms with placeholders to preserve them during tokenization.

        Args:
            text: Input text.

        Returns:
            Tuple of (modified text, mapping of placeholders to original terms).
        """
        placeholder_map = {}
        if not self.config.get("handle_special_terms", True):
            return text, placeholder_map

        # Preserve verse references (e.g., [3:16])
        for match in self.verse_pattern.finditer(text):
            verse_ref = match.group(0)  # e.g., [3:16]
            placeholder = f"__VERSE_REF_{len(placeholder_map)}__"
            placeholder_map[placeholder] = verse_ref
            text = text.replace(verse_ref, placeholder)

        # Preserve book chapter:verse references (e.g., John 3:16)
        for match in self.book_chapter_verse_pattern.finditer(text):
            full_ref = match.group(0)  # e.g., John 3:16
            placeholder = f"__BOOK_REF_{len(placeholder_map)}__"
            placeholder_map[placeholder] = full_ref
            text = text.replace(full_ref, placeholder)

        # Preserve theological and special terms
        for term in self.theological_terms | self.special_terms:
            placeholder = f"__TERM_{len(placeholder_map)}__"
            text = re.sub(fr'\b{term}\b', placeholder, text, flags=re.IGNORECASE)
            placeholder_map[placeholder] = term

        return text, placeholder_map

    def _restore_special_terms(self, tokens: List[str], placeholder_map: Dict[str, str]) -> List[str]:
        """
        Restore special terms from placeholders after tokenization.

        Args:
            tokens: List of tokenized strings.
            placeholder_map: Mapping of placeholders to original terms.

        Returns:
            List of tokens with restored terms.
        """
        restored_tokens = []
        for token in tokens:
            if token in placeholder_map:
                # Split the restored term into sub-tokens if necessary
                restored_term = placeholder_map[token]
                if token.startswith("__VERSE_REF_") or token.startswith("__BOOK_REF_"):
                    restored_tokens.append(restored_term)
                else:
                    sub_tokens = word_tokenize(restored_term)
                    restored_tokens.extend(sub_tokens)
            else:
                restored_tokens.append(token)
        return restored_tokens

    def tokenize(self, text: str, return_tensors: str = "pt") -> Dict[str, Union[torch.Tensor, List]]:
        """
        Tokenize text while preserving verse references and theological terms.

        Args:
            text: Input text to tokenize.
            return_tensors: Format of returned tensors ('pt' for PyTorch, 'np' for NumPy, None for list).

        Returns:
            Dictionary with tokenized outputs (input_ids, attention_mask, etc.).
        """
        # Step 1: Preserve special terms
        modified_text, placeholder_map = self._preserve_special_terms(text)

        # Step 2: Tokenize with base tokenizer
        base_encoding = self.base_tokenizer(
            modified_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None  # We'll handle tensor conversion later
        )

        # Step 3: Restore special terms in the tokenized output
        tokens = self.base_tokenizer.convert_ids_to_tokens(base_encoding["input_ids"])
        restored_tokens = self._restore_special_terms(tokens, placeholder_map)

        # Step 4: Convert restored tokens back to IDs
        restored_ids = []
        for token in restored_tokens:
            if token in placeholder_map.values():
                # If it's a verse reference or special term, encode as-is
                token_ids = self.base_tokenizer.encode(
                    token, add_special_tokens=False, return_tensors=None
                )
                restored_ids.extend(token_ids)
            else:
                token_id = self.base_tokenizer.convert_tokens_to_ids(token)
                restored_ids.append(token_id)

        # Truncate to max_length if necessary
        restored_ids = restored_ids[:self.max_length]
        attention_mask = [1] * len(restored_ids)

        # Pad to max_length
        padding_length = self.max_length - len(restored_ids)
        restored_ids.extend([self.base_tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)

        # Convert to tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(restored_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }
        elif return_tensors == "np":
            return {
                "input_ids": np.array(restored_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64)
            }
        else:
            return {
                "input_ids": restored_ids,
                "attention_mask": attention_mask
            }

    def tokenize_instruction_data(
        self,
        instruction: str,
        input_text: str,
        output: str,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize instruction data for fine-tuning (used by BibleInstructionDataset).

        Args:
            instruction: Instruction text (e.g., "Explain the verse").
            input_text: Input text (e.g., "John 3:16").
            output: Expected output text.
            return_tensors: Format of returned tensors ('pt' for PyTorch, 'np' for NumPy).

        Returns:
            Dictionary with tokenized inputs, attention mask, and labels.
        """
        # Format prompt as in BibleInstructionDataset
        prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "
        prompt_encoding = self.tokenize(prompt, return_tensors=None)
        output_encoding = self.tokenize(output, return_tensors=None)

        # Combine input_ids: prompt + output
        input_ids = prompt_encoding["input_ids"] + output_encoding["input_ids"]
        input_ids = input_ids[:self.max_length]

        # Combine attention masks
        attention_mask = prompt_encoding["attention_mask"] + output_encoding["attention_mask"]
        attention_mask = attention_mask[:self.max_length]

        # Create labels: -100 for prompt tokens, actual IDs for output tokens
        prompt_length = len(prompt_encoding["input_ids"])
        labels = [-100] * prompt_length + output_encoding["input_ids"]
        labels = labels[:self.max_length]

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.base_tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        labels.extend([-100] * padding_length)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        elif return_tensors == "np":
            return {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "labels": np.array(labels, dtype=np.int64)
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

    def detokenize(self, input_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Convert token IDs back to text, preserving special terms.

        Args:
            input_ids: List or tensor of token IDs.

        Returns:
            Decoded text string.
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # Remove padding tokens
        input_ids = [id_ for id_ in input_ids if id_ != self.base_tokenizer.pad_token_id]

        # Decode tokens
        text = self.base_tokenizer.decode(input_ids, skip_special_tokens=True)
        return text


# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = BiblicalTokenizer(base_tokenizer_name="bert-base-uncased")

    # Example text with verse reference and theological term
    text = "John 3:16 For God so loved the world, [3:16] YHWH said."

    # Tokenize
    encoding = tokenizer.tokenize(text, return_tensors="pt")
    print("Input IDs:", encoding["input_ids"])
    print("Attention Mask:", encoding["attention_mask"])

    # Detokenize
    decoded_text = tokenizer.detokenize(encoding["input_ids"])
    print("Decoded Text:", decoded_text)

    # Example instruction data
    instruction = "Explain the verse."
    input_text = "John 3:16"
    output = "For God so loved the world that he gave his only Son."
    instruction_encoding = tokenizer.tokenize_instruction_data(
        instruction, input_text, output, return_tensors="pt"
    )
    print("Instruction Input IDs:", instruction_encoding["input_ids"])
    print("Instruction Labels:", instruction_encoding["labels"])
